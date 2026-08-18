// pic2d — P3d marker sort (counting sort by cell + permutation gather).
//
// Why: the fused push kernel does 6 CIC gathers per marker from ~100 MB
// field arrays; with load-ordered (random) markers every warp lane misses
// L2 independently. Sorting markers by cell every sort_every steps makes
// warp neighbours touch the same cells — gathers and deposit atomics
// become local. Markers drift ~0.06 cells/step, so the legacy cadence
// (25) keeps order fresh.
//
// Counting sort (cells ≪ markers): k_cellkey (histogram) → CUB exclusive
// scan → k_scatter (stable within nothing — order inside a cell is
// arbitrary, physics-irrelevant) → permute the 7 marker arrays through a
// single reusable temp buffer with pointer swaps (no copy-back).
// Transient memory: count+offset (2×4 B/cell) + perm (4 B/marker) + one
// f32 temp (4 B/marker) + CUB scratch — ~8 B/marker + 8 B/cell, freed
// between sorts only if the owner drops the Sorter2D.

#ifndef ARC_PIC2D_SORT2D_HPP
#define ARC_PIC2D_SORT2D_HPP

#include "pic2d/particles2d.hpp"

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

namespace arc2d {
namespace s2d {

#ifdef __CUDACC__

static __global__ void k_cellkey(MarkerViews p, float x0, float z0, float idx_,
                                 float idz_, int nx, int nz, uint32_t* count,
                                 uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    int i = int((p.x[m] - x0) * idx_), k = int((p.z[m] - z0) * idz_);
    i = max(0, min(nx - 1, i));
    k = max(0, min(nz - 1, k));
    const uint32_t c = uint32_t(k) * nx + i;
    p.cell[m] = c;
    atomicAdd(&count[c], 1u);
}

static __global__ void k_scatter(MarkerViews p, uint32_t* offset,
                                 uint32_t* perm, uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    perm[atomicAdd(&offset[p.cell[m]], 1u)] = uint32_t(m);
}

static __global__ void k_permute(const float* src, float* dst,
                                 const uint32_t* perm, uint64_t n) {
    const uint64_t m = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    dst[m] = src[perm[m]];
}

#endif  // __CUDACC__

}  // namespace s2d

struct Sorter2D {
    arc::DeviceArray<uint32_t> count, offset, perm;
    arc::DeviceArray<float> tmp;
    arc::DeviceArray<uint8_t> scan_tmp;
    size_t scan_bytes = 0;
    size_t ncell = 0;
    uint64_t cap = 0;

#ifdef __CUDACC__
    void ensure(size_t ncell_, uint64_t n) {
        if (ncell_ > ncell) {
            count = arc::DeviceArray<uint32_t>(ncell_);
            offset = arc::DeviceArray<uint32_t>(ncell_);
            ncell = ncell_;
            size_t b = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, b, count.data(),
                                          offset.data(), int(ncell));
            scan_tmp = arc::DeviceArray<uint8_t>(b);
            scan_bytes = b;
        }
        if (n > cap) {
            perm = arc::DeviceArray<uint32_t>(n);
            tmp = arc::DeviceArray<float>(n);
            cap = n;
        }
    }

    void sort(MarkerStore& mk, float x0, float z0, float dx, float dz, int nx,
              int nz) {
        const uint64_t n = mk.n;
        ensure(size_t(nx) * nz, n);
        count.zero();
        MarkerViews p = mk.views();
        const int tb = 256, nb = int((n + tb - 1) / tb);
        s2d::k_cellkey<<<nb, tb>>>(p, x0, z0, 1.f / dx, 1.f / dz, nx, nz,
                                   count.data(), n);
        size_t b = scan_bytes;
        cub::DeviceScan::ExclusiveSum(scan_tmp.data(), b, count.data(),
                                      offset.data(), int(size_t(nx) * nz));
        s2d::k_scatter<<<nb, tb>>>(p, offset.data(), perm.data(), n);
        // permute → temp → copy back. (A pointer swap would hand the shared
        // temp buffer to this species and leave a SMALLER one behind for the
        // next, larger species — the first-run overflow. D2D copy-back is
        // ~0.5 ms per array at W10 scale, amortized over sort_every.)
        for (auto* a : {&mk.x, &mk.z, &mk.ux, &mk.uy, &mk.uz, &mk.w, &mk.wd}) {
            s2d::k_permute<<<nb, tb>>>(a->data(), tmp.data(), perm.data(), n);
            CUDA_CHECK(cudaMemcpyAsync(a->data(), tmp.data(), n * 4,
                                       cudaMemcpyDeviceToDevice));
        }
    }
#endif
};

}  // namespace arc2d

#endif  // ARC_PIC2D_SORT2D_HPP
