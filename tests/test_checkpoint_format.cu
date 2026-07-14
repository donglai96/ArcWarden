// ArcWarden — L-shell plan M0: checkpoint format round-trip unit test.
// Write a header with deck text, scaling metadata and an array manifest,
// reread it, and require field-exact equality; also verify magic/version
// rejection paths.

#include "pic/checkpoint.hpp"

#include <cstdio>
#include <cstdlib>

int main() {
    using namespace arc;
    const std::string path = "test_checkpoint_format.arcw";
    bool ok = true;

    CheckpointHeader h;
    h.rng_seed = 0xDEADBEEFCAFEBABEull;
    h.step = 123456789012345ll;
    h.time = 9876.5432;
    h.sd = 250.0;
    h.eps_l = 0.37;
    h.deck = "[chirp]\nnx = 6554\n# snapshot text with\nunicode Ωe0 ✓\n";
    h.git_hash = "59d9eb07e56e79eed94ed10ee9f50cab0fbcf4ab";
    h.manifest = {{"xp", 1, 11404826}, {"ux", 0, 11404826}, {"cell", 2, 11404826},
                  {"rng_state", 3, 1}};

    checkpoint_write_header(path, h);
    const CheckpointHeader r = checkpoint_read_header(path);

    auto expect = [&](bool c, const char* what) {
        if (!c) { std::printf("FAIL: %s\n", what); ok = false; }
    };
    expect(r.rng_seed == h.rng_seed, "rng_seed");
    expect(r.step == h.step, "step");
    expect(r.time == h.time, "time");
    expect(r.sd == h.sd, "S_d");
    expect(r.eps_l == h.eps_l, "eps_L");
    expect(r.deck == h.deck, "deck text");
    expect(r.git_hash == h.git_hash, "git hash");
    expect(r.manifest.size() == 4, "manifest size");
    for (size_t i = 0; i < r.manifest.size() && i < h.manifest.size(); ++i) {
        expect(r.manifest[i].name == h.manifest[i].name, "manifest name");
        expect(r.manifest[i].dtype == h.manifest[i].dtype, "manifest dtype");
        expect(r.manifest[i].count == h.manifest[i].count, "manifest count");
    }

    // corrupt magic -> must throw
    {
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        const uint32_t bad = 0x12345678;
        std::fwrite(&bad, 4, 1, f);
        std::fclose(f);
        bool threw = false;
        try { (void)checkpoint_read_header(path); } catch (...) { threw = true; }
        expect(threw, "bad magic rejected");
    }

    std::remove(path.c_str());
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
