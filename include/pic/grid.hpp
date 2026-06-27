// ArcWarden — grid geometry (Step 5).
//
// Two small POD structs, both safe to pass into kernels by value:
//
//   Grid   — REAL-space geometry, the SINGLE SOURCE OF TRUTH for nx,ny,dx,dy,
//            Lx,Ly and the flat index map. deposit / push / migrate /
//            diagnostics all take a Grid; RunParams deliberately stores no
//            geometry (design §3/§4) so the two can never drift.
//   KGrid  — K-space (Fourier) geometry for the R2C half-spectrum: nkx,nky and
//            the wavenumbers kx(i), ky(j), k2(i,j) used by the Poisson solver
//            (phi_k = rho_k/(eps0 k²), E_k = -i k phi_k) and spectral filters.
//
// ---------------------------------------------------------------------------
// R2C LAYOUT CONTRACT (must match cuFFT plan + every kernel that indexes a
// field; nailed down here so it lives in exactly one place):
//
//   real field   : row-major [ny][nx], x (i) is the FAST/contiguous axis,
//                  flat index  idx(i,j) = j*nx + i,   i in [0,nx), j in [0,ny).
//   cuFFT plan   : cufftPlan2d(&p, ny, nx)  — slow dim ny first, fast dim nx.
//   R2C output   : complex [ny][nkx], nkx = nx/2 + 1 (the FAST x axis is the
//                  halved one), flat index  kidx(i,j) = j*nkx + i.
//
// Wavenumber convention (numpy-fftfreq style, scaled by 2π/L):
//   kx(i): x is the REDUCED (half) axis, so i in [0,nkx) are all the
//          non-negative modes:           kx(i) = i * (2π/Lx).
//   ky(j): y is the FULL axis, so modes wrap to negative past Nyquist:
//          ky(j) = (j <= ny/2 ? j : j-ny) * (2π/Ly).   (design §4: "j>ny/2 取负")
//   The k=0 (DC) and Nyquist modes are zeroed by the solver (design §7.2), so
//   the sign chosen exactly at the Nyquist point is immaterial.
// ---------------------------------------------------------------------------

#ifndef ARC_PIC_GRID_HPP
#define ARC_PIC_GRID_HPP

#include <cuda_runtime.h>  // __host__ __device__ qualifiers (empty on host TUs)

namespace arc {

// 2π as a constexpr (avoids pulling in <cmath>/M_PI portability quirks).
inline constexpr double kTwoPi = 6.28318530717958647692;

// ---- Grid : real-space geometry (single source of truth) ------------------

struct Grid {
    int    nx = 0, ny = 0;
    double dx = 0.0, dy = 0.0;   // cell sizes
    double Lx = 0.0, Ly = 0.0;   // domain extents

    Grid() = default;

    // Build from cell counts + physical extents; dx,dy follow (periodic: a cell
    // of width L/n, so n cells exactly tile [0,L)).
    Grid(int nx_, int ny_, double Lx_, double Ly_)
        : nx(nx_), ny(ny_),
          dx(Lx_ / nx_), dy(Ly_ / ny_),
          Lx(Lx_), Ly(Ly_) {}

    // Flat row-major index into a real [ny][nx] field. x (i) is contiguous.
    __host__ __device__ int idx(int i, int j) const { return j * nx + i; }

    // Number of cells / real-field elements.
    __host__ __device__ int real_size() const { return nx * ny; }

    // Wrap a single index into [0,n) for periodic boundaries. Assumes the input
    // is at most one period out of range (true for CIC i+1 and a leapfrog step),
    // so two branches beat a modulo. Used by deposit/gather/migrate so the
    // wrapping convention lives in one place.
    __host__ __device__ static int wrap(int i, int n) {
        if (i >= n) i -= n;
        if (i < 0)  i += n;
        return i;
    }

    // Flat index with both coordinates wrapped periodically.
    __host__ __device__ int idx_periodic(int i, int j) const {
        return wrap(j, ny) * nx + wrap(i, nx);
    }
};

// ---- KGrid : k-space geometry for the R2C half-spectrum -------------------

struct KGrid {
    int    nkx = 0, nky = 0;     // nkx = nx/2+1 (reduced x), nky = ny (full y)
    double dkx = 0.0, dky = 0.0; // fundamental wavenumbers 2π/Lx, 2π/Ly

    KGrid() = default;

    explicit KGrid(const Grid& g)
        : nkx(g.nx / 2 + 1), nky(g.ny),
          dkx(kTwoPi / g.Lx), dky(kTwoPi / g.Ly) {}

    // Flat row-major index into a complex [ny][nkx] spectral array.
    __host__ __device__ int kidx(int i, int j) const { return j * nkx + i; }

    // Number of complex elements in the half-spectrum.
    __host__ __device__ int complex_size() const { return nkx * nky; }

    // x is the reduced axis: i in [0,nkx) are exactly the non-negative modes.
    __host__ __device__ double kx(int i) const { return i * dkx; }

    // y is the full axis: modes past Nyquist alias to negative wavenumbers.
    __host__ __device__ double ky(int j) const {
        const int m = (j <= nky / 2) ? j : j - nky;
        return m * dky;
    }

    // |k|² = kx² + ky². The solver special-cases k2==0 (DC) to avoid 1/0.
    __host__ __device__ double k2(int i, int j) const {
        const double kxi = kx(i);
        const double kyj = ky(j);
        return kxi * kxi + kyj * kyj;
    }
};

// ---- CIC interpolation stencil (the ONE definition deposit & gather share) --

// The 4 cloud-in-cell cells touched by a particle and their bilinear weights.
// Deposit (scatter q -> rho) and gather (interp E -> particle) MUST use the same
// weights; defining the stencil once here is what guarantees that (plan §16, the
// classic PIC bug source). Positions are in cell units, already wrapped to
// [0,nx)×[0,ny) by migrate/init, so floor == truncation (x >= 0).
struct CicStencil {
    int   cell[4];   // 4 wrapped flat cell indices (j*nx+i)
    float w[4];      // 4 weights, sum == 1
};

__host__ __device__ inline CicStencil cic_stencil(const Grid& g, float x, float y) {
    const int   i0 = static_cast<int>(x);     // floor for x >= 0
    const int   j0 = static_cast<int>(y);
    const float fx = x - static_cast<float>(i0);
    const float fy = y - static_cast<float>(j0);

    const int wi0 = Grid::wrap(i0,     g.nx), wi1 = Grid::wrap(i0 + 1, g.nx);
    const int wj0 = Grid::wrap(j0,     g.ny), wj1 = Grid::wrap(j0 + 1, g.ny);

    CicStencil s;
    s.cell[0] = wj0 * g.nx + wi0;  s.w[0] = (1.0f - fx) * (1.0f - fy);
    s.cell[1] = wj0 * g.nx + wi1;  s.w[1] =         fx  * (1.0f - fy);
    s.cell[2] = wj1 * g.nx + wi0;  s.w[2] = (1.0f - fx) *         fy;
    s.cell[3] = wj1 * g.nx + wi1;  s.w[3] =         fx  *         fy;
    return s;
}

} // namespace arc

#endif // ARC_PIC_GRID_HPP
