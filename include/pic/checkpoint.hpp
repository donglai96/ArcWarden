// ArcWarden — L-shell plan M0: checkpoint/restart FORMAT (schema first).
//
// Plan v3 §12.6: production runs are multi-day; the format is defined at M0
// and unit-tested, the full array streaming lands with M7 (restart 投产).
//
// Layout (little-endian, single file):
//   [Header]      magic "ARCW", u32 version, u64 rng_seed, i64 step,
//                 f64 time, f64 S_d, f64 eps_L
//   [DeckBlock]   u64 nbytes + verbatim deck text (self-describing physics)
//   [GitBlock]    u64 nbytes + git hash string
//   [Manifest]    u32 narrays, then per array: u32 name_len + name,
//                 u32 dtype (0=f32,1=f64,2=i32,3=u64), u64 count
//   [Arrays]      raw payloads in manifest order (M7; may be absent in a
//                 schema-only file, manifest counts say what WOULD follow)
//
// Versioning rule: readers refuse a higher major version; fields are only
// ever appended (never reordered) within a major version.

#ifndef ARC_PIC_CHECKPOINT_HPP
#define ARC_PIC_CHECKPOINT_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace arc {

struct CheckpointArray {
    std::string name;
    uint32_t    dtype = 0;      // 0=f32 1=f64 2=i32 3=u64
    uint64_t    count = 0;
};

struct CheckpointHeader {
    static constexpr uint32_t MAGIC   = 0x57435241u;   // "ARCW"
    static constexpr uint32_t VERSION = 1;

    uint64_t rng_seed = 0;
    int64_t  step     = 0;
    double   time     = 0.0;
    double   sd       = 1.0;
    double   eps_l    = 0.0;
    std::string deck;           // verbatim deck snapshot
    std::string git_hash;
    std::vector<CheckpointArray> manifest;
};

namespace detail {
inline void ck_write(std::FILE* f, const void* p, size_t n) {
    if (std::fwrite(p, 1, n, f) != n) throw std::runtime_error("checkpoint: short write");
}
inline void ck_read(std::FILE* f, void* p, size_t n) {
    if (std::fread(p, 1, n, f) != n) throw std::runtime_error("checkpoint: short read");
}
inline void ck_write_str(std::FILE* f, const std::string& s) {
    const uint64_t n = s.size();
    ck_write(f, &n, 8); ck_write(f, s.data(), s.size());
}
inline std::string ck_read_str(std::FILE* f) {
    uint64_t n = 0; ck_read(f, &n, 8);
    if (n > (1ull << 30)) throw std::runtime_error("checkpoint: absurd string length");
    std::string s(n, '\0'); ck_read(f, s.data(), n);
    return s;
}
} // namespace detail

inline void checkpoint_write_header(const std::string& path, const CheckpointHeader& h) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
    try {
        const uint32_t magic = CheckpointHeader::MAGIC, ver = CheckpointHeader::VERSION;
        detail::ck_write(f, &magic, 4); detail::ck_write(f, &ver, 4);
        detail::ck_write(f, &h.rng_seed, 8);
        detail::ck_write(f, &h.step, 8);
        detail::ck_write(f, &h.time, 8);
        detail::ck_write(f, &h.sd, 8);
        detail::ck_write(f, &h.eps_l, 8);
        detail::ck_write_str(f, h.deck);
        detail::ck_write_str(f, h.git_hash);
        const uint32_t na = (uint32_t)h.manifest.size();
        detail::ck_write(f, &na, 4);
        for (const auto& a : h.manifest) {
            const uint32_t nl = (uint32_t)a.name.size();
            detail::ck_write(f, &nl, 4);
            detail::ck_write(f, a.name.data(), nl);
            detail::ck_write(f, &a.dtype, 4);
            detail::ck_write(f, &a.count, 8);
        }
    } catch (...) { std::fclose(f); throw; }
    std::fclose(f);
}

inline CheckpointHeader checkpoint_read_header(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
    CheckpointHeader h;
    try {
        uint32_t magic = 0, ver = 0;
        detail::ck_read(f, &magic, 4); detail::ck_read(f, &ver, 4);
        if (magic != CheckpointHeader::MAGIC)
            throw std::runtime_error("checkpoint: bad magic");
        if (ver > CheckpointHeader::VERSION)
            throw std::runtime_error("checkpoint: version " + std::to_string(ver) +
                                     " newer than reader");
        detail::ck_read(f, &h.rng_seed, 8);
        detail::ck_read(f, &h.step, 8);
        detail::ck_read(f, &h.time, 8);
        detail::ck_read(f, &h.sd, 8);
        detail::ck_read(f, &h.eps_l, 8);
        h.deck = detail::ck_read_str(f);
        h.git_hash = detail::ck_read_str(f);
        uint32_t na = 0; detail::ck_read(f, &na, 4);
        for (uint32_t i = 0; i < na; ++i) {
            CheckpointArray a;
            uint32_t nl = 0; detail::ck_read(f, &nl, 4);
            a.name.resize(nl); detail::ck_read(f, a.name.data(), nl);
            detail::ck_read(f, &a.dtype, 4);
            detail::ck_read(f, &a.count, 8);
            h.manifest.push_back(std::move(a));
        }
    } catch (...) { std::fclose(f); throw; }
    std::fclose(f);
    return h;
}

} // namespace arc

#endif // ARC_PIC_CHECKPOINT_HPP
