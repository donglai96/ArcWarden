// ArcWarden — DiagManager: builds the enabled diagnostic modules from the deck's
// `[diagnostics] enable = ...` list and fans out sample()/finalize() each step.
// Non-templated: it consumes the concrete const Fields&/Particles& that Simulation
// exposes, so one manager drives both ES and Darwin runs.

#ifndef ARC_PIC_DIAG_MANAGER_HPP
#define ARC_PIC_DIAG_MANAGER_HPP

#include "pic/deck.hpp"
#include "pic/diag/module.hpp"
#include "pic/diag/spectrum.hpp"
#include "pic/diag/kt.hpp"
#include "pic/diag/phase_video.hpp"
#include "pic/diag/snapshot.hpp"
#include "pic/diag/em_energy.hpp"
#include "pic/diag/phase_frames.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace arc { namespace diag {

class DiagManager {
    std::vector<std::unique_ptr<IDiag>> mods_;
    double bmag_, cost_, sint_, vth_, vr_;
public:
    // pref = output file prefix (e.g. "out/an2019_sim3_" or "an2019_sim3_");
    // outdir = directory for phase_frames/manifest ("" → CWD).
    DiagManager(const Deck& d, const RunParams& rp,
                const std::string& pref, const std::string& outdir) {
        bmag_ = std::sqrt((double)rp.B0[0]*rp.B0[0] + (double)rp.B0[1]*rp.B0[1]
                        + (double)rp.B0[2]*rp.B0[2]);
        cost_ = bmag_ > 0 ? rp.B0[0]/bmag_ : 1.0;      // b̂_x
        sint_ = bmag_ > 0 ? rp.B0[2]/bmag_ : 0.0;      // b̂_z
        vth_  = d.species.empty() ? 1.0 : d.species[0].uth[0];
        vr_   = (rp.pump_k0 > 0 && cost_ > 0) ? (rp.pump_w0/(rp.pump_k0*cost_))/vth_ : 0.0;

        for (const std::string& k : d.diag_enable) {
            if      (k == "spectrum")     mods_.emplace_back(new SpectrumDiag(d));
            else if (k == "kt")           mods_.emplace_back(new KTDump(d, pref));
            else if (k == "phase_video")  mods_.emplace_back(new PhaseVideo(d, pref));
            else if (k == "snapshot")     mods_.emplace_back(new Snapshot(d, pref));
            else if (k == "em_energy")    mods_.emplace_back(new EMEnergy(d, pref));
            else if (k == "phase_frames") mods_.emplace_back(new PhaseFrames(d, outdir));
            else std::fprintf(stderr, "DiagManager: unknown diagnostic '%s' (ignored)\n", k.c_str());
        }
    }

    bool empty() const { return mods_.empty(); }

    void sample(const Fields& f, const Particles& p, const Grid& g, const RunParams& rp,
                long step, double t) {
        if (mods_.empty()) return;
        DiagFrame fr{ f, p, g, rp, step, t, bmag_, cost_, sint_, vth_, vr_ };
        for (auto& m : mods_) m->sample(fr);
    }

    void finalize() { for (auto& m : mods_) m->finalize(); }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_MANAGER_HPP
