// pic2d — P3c checkpoint/restart (PLAN_2D_REBORN §2.6).
//
// Binary layout (little-endian, all sizes explicit):
//   char[8]  "AW2DCKP2"  (v2: sparse tile pool; v1 dense still loadable)
//   char[64] git hash (CMake configure-time AW_GIT_HASH — closes the
//            legacy provenance gap; stale-by-one-commit is accepted and
//            recorded as such)
//   i64 nstep, f64 time, i32 nx, i32 nz, i32 nspecies
//   u64 field_cells (v2 only; storage length of every field array —
//       dense nx·nz or pool nslots·256; must match the rebuilt deck)
//   12 × field arrays [field_cells] f32 (ex..bz, jx..jz, vcx..vcz)
//   per species: char[32] name, u64 n, 7 × [n] f32 (x,z,ux,uy,uz,w,wd)
//   optional trailer (2026-08-20, Gauss closure): char[4] "RHOC" +
//       [field_cells] f32 accumulated cold charge — written only when the
//       run has gauss_clean_every != 0; absence on load is tolerated with
//       a warning (the cold-charge ledger restarts from zero).
// Written to <path>.tmp then atomically renamed (legacy convention: a
// killed job never leaves a torn checkpoint). Device↔host streamed in
// 64 MB chunks.

#ifndef ARC_PIC2D_CHECKPOINT2D_HPP
#define ARC_PIC2D_CHECKPOINT2D_HPP

#include "pic2d/sim2d.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AW_GIT_HASH
#define AW_GIT_HASH "unknown"
#endif

namespace arc2d {

namespace ckpt {

constexpr size_t CHUNK = 64u << 20;

inline void stream_out(FILE* f, const float* dev, size_t n) {
    std::vector<float> h(std::min(n, CHUNK / 4));
    for (size_t off = 0; off < n; off += h.size()) {
        const size_t c = std::min(h.size(), n - off);
        CUDA_CHECK(cudaMemcpy(h.data(), dev + off, c * 4, cudaMemcpyDeviceToHost));
        if (std::fwrite(h.data(), 4, c, f) != c)
            throw std::runtime_error("ckpt: short write");
    }
}
inline void stream_in(FILE* f, float* dev, size_t n) {
    std::vector<float> h(std::min(n, CHUNK / 4));
    for (size_t off = 0; off < n; off += h.size()) {
        const size_t c = std::min(h.size(), n - off);
        if (std::fread(h.data(), 4, c, f) != c)
            throw std::runtime_error("ckpt: short read");
        CUDA_CHECK(cudaMemcpy(dev + off, h.data(), c * 4, cudaMemcpyHostToDevice));
    }
}

}  // namespace ckpt

inline void save_checkpoint(Sim2D& S, const std::string& path) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("ckpt: cannot open " + tmp);
    std::fwrite("AW2DCKP2", 1, 8, f);
    char gh[64] = {0};
    std::strncpy(gh, AW_GIT_HASH, sizeof gh - 1);
    std::fwrite(gh, 1, 64, f);
    const int64_t ns = S.nstep;
    const int32_t nx = S.F.nx, nz = S.F.nz, nsp = int32_t(S.sp.size());
    std::fwrite(&ns, 8, 1, f);
    std::fwrite(&S.time, 8, 1, f);
    std::fwrite(&nx, 4, 1, f);
    std::fwrite(&nz, 4, 1, f);
    std::fwrite(&nsp, 4, 1, f);
    const uint64_t fc = S.F.field_cells;
    std::fwrite(&fc, 8, 1, f);
    const size_t nc = fc;
    for (auto* a : {&S.F.ex, &S.F.ey, &S.F.ez, &S.F.bx, &S.F.by, &S.F.bz,
                    &S.F.jx, &S.F.jy, &S.F.jz, &S.F.vcx, &S.F.vcy, &S.F.vcz})
        ckpt::stream_out(f, a->data(), nc);
    for (auto& s : S.sp) {
        char nm[32] = {0};
        std::strncpy(nm, s.name.c_str(), sizeof nm - 1);
        std::fwrite(nm, 1, 32, f);
        const uint64_t n = s.mk->n;
        std::fwrite(&n, 8, 1, f);
        for (auto* a : {&s.mk->x, &s.mk->z, &s.mk->ux, &s.mk->uy, &s.mk->uz,
                        &s.mk->w, &s.mk->wd})
            ckpt::stream_out(f, a->data(), n);
    }
    if (S.gauss_every != 0 && S.rho_c.size()) {
        std::fwrite("RHOC", 1, 4, f);
        ckpt::stream_out(f, S.rho_c.data(), nc);
    }
    std::fclose(f);
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("ckpt: rename failed");
}

// Sim2D must already be built from the SAME deck; restores dynamic state.
inline void load_checkpoint(Sim2D& S, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ckpt: cannot open " + path);
    char magic[8], gh[64];
    if (std::fread(magic, 1, 8, f) != 8 ||
        (std::memcmp(magic, "AW2DCKP1", 8) && std::memcmp(magic, "AW2DCKP2", 8)))
        throw std::runtime_error("ckpt: bad magic");
    const bool v2 = magic[7] == '2';
    (void)!std::fread(gh, 1, 64, f);
    int64_t ns;
    int32_t nx, nz, nsp;
    (void)!std::fread(&ns, 8, 1, f);
    (void)!std::fread(&S.time, 8, 1, f);
    (void)!std::fread(&nx, 4, 1, f);
    (void)!std::fread(&nz, 4, 1, f);
    (void)!std::fread(&nsp, 4, 1, f);
    if (nx != S.F.nx || nz != S.F.nz || nsp != int32_t(S.sp.size()))
        throw std::runtime_error("ckpt: geometry/species mismatch vs deck");
    uint64_t fc = size_t(nx) * nz;      // v1: always dense
    if (v2) (void)!std::fread(&fc, 8, 1, f);
    if (fc != S.F.field_cells)
        throw std::runtime_error("ckpt: field layout mismatch vs deck "
                                 "(sparse band changed?)");
    S.nstep = ns;
    const size_t nc = fc;
    for (auto* a : {&S.F.ex, &S.F.ey, &S.F.ez, &S.F.bx, &S.F.by, &S.F.bz,
                    &S.F.jx, &S.F.jy, &S.F.jz, &S.F.vcx, &S.F.vcy, &S.F.vcz})
        ckpt::stream_in(f, a->data(), nc);
    for (auto& s : S.sp) {
        char nm[32];
        uint64_t n;
        (void)!std::fread(nm, 1, 32, f);
        (void)!std::fread(&n, 8, 1, f);
        if (n != s.mk->n)
            throw std::runtime_error("ckpt: marker count mismatch for " + s.name);
        for (auto* a : {&s.mk->x, &s.mk->z, &s.mk->ux, &s.mk->uy, &s.mk->uz,
                        &s.mk->w, &s.mk->wd})
            ckpt::stream_in(f, a->data(), n);
    }
    char tag[4];
    if (std::fread(tag, 1, 4, f) == 4 && std::memcmp(tag, "RHOC", 4) == 0) {
        if (S.rho_c.size())
            ckpt::stream_in(f, S.rho_c.data(), nc);
        else  // deck runs with gauss off: skip the ledger
            std::fseek(f, long(nc) * 4, SEEK_CUR);
    } else if (S.rho_c.size()) {
        std::fprintf(stderr,
                     "ckpt: no RHOC block in %s — cold-charge ledger "
                     "restarts from zero (residual monitor will re-settle)\n",
                     path.c_str());
    }
    std::fclose(f);
    std::printf("resumed from %s at step %ld (t = %.1f), written by git %.12s\n",
                path.c_str(), long(S.nstep), S.time, gh);
}

}  // namespace arc2d

#endif  // ARC_PIC2D_CHECKPOINT2D_HPP
