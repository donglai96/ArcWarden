// ArcWarden — L-shell plan M0: run provenance metadata.
//
// Every deck-driven run writes <outdir>/RUN_META.txt (timestamp, git hash if
// available, command line, scaling metadata) and a verbatim deck snapshot, so
// any output directory is self-describing and reproducible (plan v3 §12.6,
// VII.3 discipline item 2).

#ifndef ARC_PIC_RUN_META_HPP
#define ARC_PIC_RUN_META_HPP

#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

namespace arc {

inline std::string git_hash_best_effort() {
    std::string h = "unknown";
#ifndef _WIN32
    if (FILE* p = ::popen("git rev-parse HEAD 2>/dev/null", "r")) {
        char buf[64] = {0};
        if (std::fgets(buf, sizeof(buf), p)) {
            h = buf;
            while (!h.empty() && (h.back() == '\n' || h.back() == '\r')) h.pop_back();
        }
        ::pclose(p);
    }
#endif
    return h;
}

inline void write_run_meta(const std::string& outdir, const std::string& deck_path,
                           int argc, char** argv,
                           double sd = 1.0, double eps_l = 0.0) {
    // deck snapshot (verbatim copy)
    {
        std::ifstream in(deck_path);
        std::ofstream out(outdir + "/deck_snapshot.ini");
        out << in.rdbuf();
    }
    std::ofstream m(outdir + "/RUN_META.txt");
    const std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S %Z", std::localtime(&now));
    m << "timestamp = " << ts << "\n"
      << "git_hash  = " << git_hash_best_effort() << "\n"
      << "deck      = " << deck_path << " (snapshot: deck_snapshot.ini)\n"
      << "S_d       = " << sd << "\n"
      << "eps_L     = " << eps_l << "\n"
      << "cmdline   =";
    for (int i = 0; i < argc; ++i) m << " " << argv[i];
    m << "\n";
}

} // namespace arc

#endif // ARC_PIC_RUN_META_HPP
