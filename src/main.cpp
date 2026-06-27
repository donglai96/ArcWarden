// ArcWarden — 2D3V electrostatic spectral GPU PIC.
// Step 1: smoke test — report the CUDA device and exit 0.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

bool false_env_value(const char* v) {
    return v && ((std::strcmp(v, "0") == 0) || (std::strcmp(v, "false") == 0) ||
                 (std::strcmp(v, "off") == 0) || (std::strcmp(v, "no") == 0));
}

bool color_enabled() {
    if (const char* force = std::getenv("ARCWARDEN_COLOR")) {
        return !false_env_value(force);
    }
    if (std::getenv("NO_COLOR")) {
        return false;
    }
    const char* term = std::getenv("TERM");
    if (!term || std::strcmp(term, "dumb") == 0) {
        return false;
    }
#ifdef _WIN32
    return true;
#else
    return isatty(fileno(stdout));
#endif
}

void print_ascii_banner() {
    std::printf("                 ~~===={ ARCWARDEN ONLINE }====~~\n");
}

// ArcWarden's signature cyan voice color (truecolor SGR).
constexpr const char* ARC_VOICE = "\033[38;2;126;240;255m";

// Make ArcWarden "speak": one decorated cyan line. Honors color_enabled().
// Use this for any flavor/voice line so the styling lives in one place.
void arc_say(const char* line) {
    if (color_enabled()) {
        std::printf("%s  ~~===={ %s }====~~\033[0m\n", ARC_VOICE, line);
    } else {
        std::printf("  ~~===={ %s }====~~\n", line);
    }
}

// Arc Warden (Dota 2) voice lines — pick one explicitly with arc_say(...).
// (Reference pool; not auto-selected.)
//   "The Self arrives at last."
//   "The self knows this power."
//   "Two minds, but a single thought."
//   "I am beside myself."
//   "Know thyself."
//   "Tempest Double!"
//   "A clone, and yet myself."
//   "Doubt is removed."
//   "We are of one mind."
//   "Reduplicate."

// One aligned "label   value" device-info row: dim label, bright value.
void print_info(const char* label, const char* value) {
    if (color_enabled()) {
        std::printf("   \033[38;2;118;130;156m%-8s\033[0m  \033[38;2;226;232;245m%s\033[0m\n",
                    label, value);
    } else {
        std::printf("   %-8s  %s\n", label, value);
    }
}

// ArcWarden's self-introduction, printed as a cyan quoted block with a
// vertical gutter (│). Lines are pre-wrapped to stay narrow and tidy.
void print_intro() {
    static const char* const lines[] = {
        "Welcome to my world. I am a hero of Dota,",
        "commander of plasma and the electromagnetic field.",
        "My name is ArcWarden, or Zet",
        "...In truth, a 2D3V gpu particle-in-cell",
        "plasma simulator, forged by Donglai Ma.",
        "Reach out: donglaima96@gmail.com",
        "           dma96@atmos.ucla.edu",
        "           donglai.ma@kla.com",
    };
    const bool c = color_enabled();
    for (const char* l : lines) {
        if (c) {
            std::printf("   %s\342\224\202\033[0m \033[38;2;176;224;240m%s\033[0m\n", ARC_VOICE, l);
        } else {
            std::printf("   | %s\n", l);
        }
    }
}

// Fixed bead/pixel raster from the Python rows in the attached file.
// Keep row order exact; edit the row characters directly for pixel changes.
static constexpr int ARCWARDEN_W = 43;
static constexpr int ARCWARDEN_H = 45;

static constexpr const char* ARCWARDEN_BITMAP[ARCWARDEN_H] = {
    ".................GCCLCLLLG.................",
    ".................GLLLBBBBLGGG..............",
    "..............RLLBBBBBBBBBLLLGGGPP.........",
    "...........BCGLLLBLBBLBBBBLLLLLLGPPM.......",
    ".........PPGLCLLLBBBBBLLBLBLLLLLLPPGG......",
    "........PPLLLLLLLLLBLBBLLLLBLCGCRLPPP......",
    ".......VPBLLCLLLWWWBBLLLLLLBBLLLLLPPP......",
    "......VPVLCGCLLBWWLBBLLLLLLBLBLMMLLBP......",
    "......VVVLMLLBBLWWLBLBBLBLBBLLBMLLLLB..B...",
    ".....KVVLLMMLBBLWWWBBBBBBBLLBBBLLMLGBVBB...",
    ".....KVLLLMMBBBBLLLLBBLBBLLLBBLLLMLLBBBBBB.",
    "....BKVBLMLLBBBBBBBBBLBBBLLLLBLLLMLLPPPPBG.",
    "...VKKVVLMLLBBBBLBLBBLLBBBBBBLBLLMLLLPPBBB.",
    "..KVVVVLLMLLBBBLBBLBBBBBBLBBBBLLLRLLLPPPVG.",
    "..KKVVLCRPLLBBBBBLBBBBBBBBLBBBBBBRLLLBPPVG.",
    ".KKKVVLCLRBBBBLLBLLBBBBBBBBLLBBLLGLLLPPBVG.",
    ".KKVVVLLLRBBBBBBBBLBBBBLBBBLBBLRRRLLLLLBVBB",
    ".KKVVVLLLRBBBLBBBBBBBBBBBLLBLLLWRLBLLGBBBVB",
    ".KKKVLLLBLRRBLLBBBBBBBBBBBBBBLWRRLBLLLGBPVB",
    ".KKVVLLLBBRWRBLBBBBBBBBBBBBLLWWRRBBLLLBBBBB",
    "KKVVVLLLBBRRWWPBBBBBBBBBBBBMWWRGMBBLLLBBBVB",
    "KKVVVLLLBVRRWWRMLBBBLBBBMWWWWRRMMBBLLGGPBVB",
    "KKVVVLLLBBPMRRWWWMMBBBBBMWWWRRLBBBBLGGBBVBB",
    "KKKVVLLLBBBPLRRWWWRBBBLBWWWRRLBBBBBLGBVVVVB",
    "KKKKVGGLBVBBBMRRWWGBBBBLRRRMBBBBBBBGGBVVBVR",
    "KKKVVVMLBBBBBBPMGRLBBBBBGLLLBBBBBPMRRBBBBPB",
    "KKKVVVGMPVBBBBLLLBBBBBBBBLLLBBBBVVGGRVVVBPB",
    "KVVVVVMMMVBBBBLCLBBBBBBBBLLCLBBBBVGRGVVVPPB",
    "KVVVKKGRGKVBBBGLBBBBBBBBBLLCLLBBMMRGGVBPPBB",
    ".VVVVKBGRPVBBLCCLBBBBBBBBLLCLLBBPMRVVGBPPPB",
    ".VVVVKVBRMVBBLCCLLCLBBLCLLLLLCBBPMGVVGPPPPB",
    ".VVVVGVVMMVBBLLLLLLCLLLCLCWLBBBVMMMBVMMMPPB",
    ".VVVMPVVMMVVBVLLLLLLLLLCLCCCBBGMPMVGGMMMPPB",
    ".VVVMMBGVMMMLBGCCLCCLWWWWCCGBPMMMMVGGMMMMPB",
    ".VVVMMGGKMMMGBBLCLLCWWWWWCGBBPMRGVBMMMMMMPV",
    ".VPMPMMPVKRRMVBBGCLCWWCWCGBBPGRVBBMMMMMMMVV",
    "..PMPPMMMVNGRMBBBGCCLLLCCBBBMRRVBBMMMMMMVV.",
    "..MPMMPMMLVVGRRRLLCCBBLCCLLRRRVBGRGMMMMMVV.",
    "....PMMMMGBVKRRWLLCGLBGCRRWWVVBBPPRRGMMMV..",
    "...MPMMGGVVVVKRWRWWCLLCCWWRRVVBVVPMGRGVVK..",
    "....PMGGPVVVVVKKKRWWRWWWRVKKVPVVPVPGGGVV...",
    ".....GMGVVVVVVVKVGVPGVVPPVVVVPPPPPPPVVV....",
    "......BVVVVVVVVVVVVVVVVVVVKVVVVVVVVVVK.....",
    ".......KKKKKKKKKKKKKKKVVKVKKKVVKKKKVV......",
    "........KKKKKKKKKKVVVVVVVVVVVVVVVVV........",
};

static_assert(sizeof(ARCWARDEN_BITMAP) / sizeof(ARCWARDEN_BITMAP[0]) == ARCWARDEN_H,
              "bad ArcWarden bitmap height");

struct Rgb {
    int r, g, b;
    bool solid;  // false for transparent '.' background
};

Rgb dot_rgb(char px) {
    switch (px) {
        case 'K': return {42, 32, 78, true};
        case 'V': return {73, 43, 135, true};
        case 'P': return {124, 71, 185, true};
        case 'M': return {208, 82, 180, true};
        case 'R': return {246, 174, 218, true};
        case 'N': return {26, 43, 126, true};
        case 'B': return {37, 91, 190, true};
        case 'L': return {91, 160, 235, true};
        case 'C': return {24, 201, 216, true};
        case 'W': return {244, 246, 255, true};
        case 'G': return {151, 150, 164, true};
        default:  return {0, 0, 0, false};
    }
}

const char* dot_color(char px) {
    switch (px) {
        case 'K': return "\033[38;2;42;32;78m";
        case 'V': return "\033[38;2;73;43;135m";
        case 'P': return "\033[38;2;124;71;185m";
        case 'M': return "\033[38;2;208;82;180m";
        case 'R': return "\033[38;2;246;174;218m";
        case 'N': return "\033[38;2;26;43;126m";
        case 'B': return "\033[38;2;37;91;190m";
        case 'L': return "\033[38;2;91;160;235m";
        case 'C': return "\033[38;2;24;201;216m";
        case 'W': return "\033[38;2;244;246;255m";
        case 'G': return "\033[38;2;151;150;164m";
        default:  return "\033[0m";
    }
}

void print_pixel_banner() {
    const bool use_color = color_enabled();
    if (!use_color) {
        print_ascii_banner();
        return;
    }

    std::printf("\n");
    for (int y = 0; y < ARCWARDEN_H; ++y) {
        const char* row = ARCWARDEN_BITMAP[y];
        for (int x = 0; x < ARCWARDEN_W; ++x) {
            const char px = row[x];
            if (px == '.') {
                std::printf("  ");
            } else {
                std::printf("%s\342\227\217 ", dot_color(px));
            }
        }
        std::printf("\033[0m\n");
    }

    std::printf("%s  ~~===={ ARCWARDEN ONLINE }====~~\033[0m\n", ARC_VOICE);
}

// Half-size renderer: packs two vertical pixels into one text cell using the
// upper-half-block glyph (foreground = top pixel, background = bottom pixel),
// and one column per pixel. Result is ~1/4 the area of print_pixel_banner()
// with no pixel loss, and stays square because a cell is ~2x taller than wide.
void print_pixel_banner_compact() {
    const bool use_color = color_enabled();
    if (!use_color) {
        print_ascii_banner();
        return;
    }

    std::printf("\n");
    for (int y = 0; y < ARCWARDEN_H; y += 2) {
        for (int x = 0; x < ARCWARDEN_W; ++x) {
            const Rgb top = dot_rgb(ARCWARDEN_BITMAP[y][x]);
            const Rgb bot =
                (y + 1 < ARCWARDEN_H) ? dot_rgb(ARCWARDEN_BITMAP[y + 1][x]) : Rgb{0, 0, 0, false};

            if (top.solid && bot.solid) {
                // ▀ with fg=top, bg=bottom
                std::printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm\342\226\200",
                            top.r, top.g, top.b, bot.r, bot.g, bot.b);
            } else if (top.solid) {
                // ▀ with fg=top, default bg
                std::printf("\033[49m\033[38;2;%d;%d;%dm\342\226\200", top.r, top.g, top.b);
            } else if (bot.solid) {
                // ▄ (lower half) with fg=bottom, default bg
                std::printf("\033[49m\033[38;2;%d;%d;%dm\342\226\204", bot.r, bot.g, bot.b);
            } else {
                std::printf("\033[0m ");
            }
        }
        std::printf("\033[0m\n");
    }

    std::printf("%s      ~~===={ ARCWARDEN ONLINE }====~~\033[0m\n", ARC_VOICE);
}

// Dispatch: ARCWARDEN_BANNER = compact (default) | full | off.
void print_banner() {
    const char* mode = std::getenv("ARCWARDEN_BANNER");
    if (mode && std::strcmp(mode, "off") == 0) {
        return;
    }
    if (mode && std::strcmp(mode, "full") == 0) {
        print_pixel_banner();
        return;
    }
    print_pixel_banner_compact();
}

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::fprintf(stderr, "ArcWarden: Futile effort. No CUDA device found.\n");
        return 1;
    }

    int dev = 0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    print_banner();

    std::printf("\n");
    print_intro();
    std::printf("\n");

    char buf[320];

    std::snprintf(buf, sizeof(buf), "%s  (%d/%d)", prop.name, dev, n);
    print_info("device", buf);

    std::snprintf(buf, sizeof(buf), "sm_%d%d", prop.major, prop.minor);
    print_info("compute", buf);

    std::snprintf(buf, sizeof(buf), "%.1f GB",
                  static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    print_info("memory", buf);

    std::printf("\n");
    arc_say("The Self knows this power.");
    return 0;
}
