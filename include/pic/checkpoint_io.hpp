// ArcWarden — M7 pulled forward: full checkpoint/restart array streaming on
// the M0 schema (checkpoint.hpp). Everything checkpoint-shaped lives HERE;
// the only touches elsewhere are two additive accessors on MaxwellSimulation
// (step_count / set_step_count) and the guarded calls in tools/chirp2d.cu.
//
// File layout (single file, little-endian):
//   [M0 header incl. manifest]  then raw payloads in manifest order.
// Saved state: Yee fields (ex..bz), cold fluid (vcy,vcz if present),
// particles (x,y,ux,uy,uz,w,cell; wd/wc/wdd if allocated), step count.
// Scratch (bins, jx..jz, masks, filters) is rebuilt, not saved. The particle
// RNG is hashed-stateless, the antenna is a function of step*dt: no hidden
// state. Writes go to <path>.tmp then rename — a killed run never corrupts
// the previous checkpoint.

#ifndef ARC_PIC_CHECKPOINT_IO_HPP
#define ARC_PIC_CHECKPOINT_IO_HPP

#include "pic/checkpoint.hpp"
#include "pic/simulation_maxwell.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace arc {

namespace ckio {

constexpr size_t CHUNK = 64ull << 20;   // 64M elements per staging chunk

template <class T>
inline void dump(std::FILE* f, const DeviceArray<T>& a, std::vector<char>& buf) {
    const size_t n = a.size();
    buf.resize(std::min(n, CHUNK) * sizeof(T));
    for (size_t off = 0; off < n; off += CHUNK) {
        const size_t c = std::min(CHUNK, n - off);
        CUDA_CHECK(cudaMemcpy(buf.data(), a.data() + off, c * sizeof(T),
                              cudaMemcpyDeviceToHost));
        detail::ck_write(f, buf.data(), c * sizeof(T));
    }
}

template <class T>
inline void slurp(std::FILE* f, DeviceArray<T>& a, uint64_t count,
                  std::vector<char>& buf) {
    if (a.size() != count)
        throw std::runtime_error("checkpoint: array size mismatch (deck/ppc "
                                 "changed?): have " + std::to_string(a.size()) +
                                 " want " + std::to_string(count));
    buf.resize(std::min<size_t>(count, CHUNK) * sizeof(T));
    for (size_t off = 0; off < count; off += CHUNK) {
        const size_t c = std::min<size_t>(CHUNK, count - off);
        detail::ck_read(f, buf.data(), c * sizeof(T));
        CUDA_CHECK(cudaMemcpy(a.data() + off, buf.data(), c * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
}

template <class T> constexpr uint32_t dtype();
template <> constexpr uint32_t dtype<float>()  { return 0; }
template <> constexpr uint32_t dtype<double>() { return 1; }
template <> constexpr uint32_t dtype<int>()    { return 2; }

struct Entry {
    const char* name;
    uint32_t    dt;
    uint64_t    count;
};

// The manifest is built from what is actually allocated, so full-f and
// delta-f runs (and wprec variants) round-trip without special cases.
template <class Fn>
inline void for_each_array(MaxwellSimulation& sim, Fn&& fn) {
    auto& F = sim.fields();
    auto& P = sim.particles();
    fn("ex", F.ex_); fn("ey", F.ey_); fn("ez", F.ez_);
    fn("bx", F.bx_); fn("by", F.by_); fn("bz", F.bz_);
    if (sim.vcy().size()) { fn("vcy", sim.vcy()); fn("vcz", sim.vcz()); }
    fn("px", P.x); fn("py", P.y);
    fn("pux", P.ux); fn("puy", P.uy); fn("puz", P.uz);
    fn("pw", P.w); fn("pcell", P.cell);
    if (P.wd.size())  fn("pwd", P.wd);
    if (P.wc.size())  fn("pwc", P.wc);
    if (P.wdd.size()) fn("pwdd", P.wdd);
}

} // namespace ckio

inline void checkpoint_save(const std::string& path, MaxwellSimulation& sim,
                            long step, double time, uint64_t rng_seed,
                            const std::string& deck_text) {
    CheckpointHeader h;
    h.rng_seed = rng_seed;
    h.step = step;
    h.time = time;
    h.deck = deck_text;
    ckio::for_each_array(sim, [&](const char* name, auto& arr) {
        using T = std::remove_pointer_t<decltype(arr.data())>;
        h.manifest.push_back({name, ckio::dtype<T>(), (uint64_t)arr.size()});
    });

    const std::string tmp = path + ".tmp";
    checkpoint_write_header(tmp, h);
    std::FILE* f = std::fopen(tmp.c_str(), "ab");
    if (!f) throw std::runtime_error("checkpoint: cannot append " + tmp);
    std::vector<char> buf;
    try {
        ckio::for_each_array(sim, [&](const char*, auto& arr) {
            ckio::dump(f, arr, buf);
        });
    } catch (...) { std::fclose(f); throw; }
    std::fclose(f);
    std::filesystem::rename(tmp, path);
}

// Returns the saved step count. Arrays must already be allocated with the
// same sizes (run the normal init first, then load on top of it).
inline long checkpoint_load(const std::string& path, MaxwellSimulation& sim,
                            std::string* deck_out = nullptr) {
    CheckpointHeader h = checkpoint_read_header(path);
    if (deck_out) *deck_out = h.deck;

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("checkpoint: cannot open " + path);
    std::vector<char> buf;
    try {
        // skip the header by re-reading it through the same parser
        uint32_t u32; uint64_t u64; double d64;
        detail::ck_read(f, &u32, 4); detail::ck_read(f, &u32, 4);
        detail::ck_read(f, &u64, 8);
        detail::ck_read(f, &u64, 8);
        detail::ck_read(f, &d64, 8); detail::ck_read(f, &d64, 8);
        detail::ck_read(f, &d64, 8);
        (void)detail::ck_read_str(f); (void)detail::ck_read_str(f);
        uint32_t na = 0; detail::ck_read(f, &na, 4);
        for (uint32_t i = 0; i < na; ++i) {
            uint32_t nl = 0; detail::ck_read(f, &nl, 4);
            std::fseek(f, nl, SEEK_CUR);
            detail::ck_read(f, &u32, 4); detail::ck_read(f, &u64, 8);
        }
        // payloads, in the same order the manifest (and saver) used
        size_t idx = 0;
        ckio::for_each_array(sim, [&](const char* name, auto& arr) {
            if (idx >= h.manifest.size())
                throw std::runtime_error("checkpoint: manifest shorter than "
                                         "expected at " + std::string(name));
            const auto& m = h.manifest[idx++];
            if (m.name != name)
                throw std::runtime_error("checkpoint: manifest order mismatch: "
                                         "have " + m.name + " want " + name);
            ckio::slurp(f, arr, m.count, buf);
        });
        if (idx != h.manifest.size())
            throw std::runtime_error("checkpoint: extra arrays in file");
    } catch (...) { std::fclose(f); throw; }
    std::fclose(f);
    return (long)h.step;
}

} // namespace arc

#endif // ARC_PIC_CHECKPOINT_IO_HPP
