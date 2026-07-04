// ArcWarden diagnostic — EMEnergy: Darwin field energies (E_total, B, E_T).
// (Lifted from the run_deck.cu Darwin block.) Logs 0.5·dx·dy·Σ(component²) for each
// field to <pref>emenergy.csv every nsteps/40 steps, and prints the final line.

#ifndef ARC_PIC_DIAG_EM_ENERGY_HPP
#define ARC_PIC_DIAG_EM_ENERGY_HPP

#include "pic/diag/module.hpp"
#include "pic/deck.hpp"
#include <fstream>

namespace arc { namespace diag {

class EMEnergy : public IDiag {
    long   every_;
    std::ofstream csv_;
    double eE_ = 0, eB_ = 0, eT_ = 0;
public:
    EMEnergy(const Deck& d, const std::string& pref)
        : every_(d.rp.nsteps / 40 > 0 ? d.rp.nsteps / 40 : 1) {
        csv_.open(pref + "emenergy.csv");
        if (csv_) csv_ << "step,time,E_total,B,E_T\n";
    }
    void sample(const DiagFrame& fr) override {
        if (fr.step % every_ != 0 || fr.f.Ez.size() == 0) return;   // Darwin only
        cudaDeviceSynchronize();
        const double cell = 0.5 * fr.g.dx * fr.g.dy;
        eE_ = cell * (sum_sq(fr.f.Ex) + sum_sq(fr.f.Ey) + sum_sq(fr.f.Ez));
        eB_ = cell * (sum_sq(fr.f.Bx) + sum_sq(fr.f.By) + sum_sq(fr.f.Bz));
        eT_ = cell * (sum_sq(fr.f.ETx) + sum_sq(fr.f.ETy) + sum_sq(fr.f.ETz));
        if (csv_) csv_ << fr.step << ',' << fr.t << ',' << eE_ << ',' << eB_ << ',' << eT_ << '\n';
    }
    void finalize() override {
        std::printf("  final field energy: electric(E_total)=%.4e  magnetic=%.4e (%.2e of E)"
                    "  transverse-E=%.4e (%.2e of E)\n",
                    eE_, eB_, eE_ > 0 ? eB_/eE_ : 0.0, eT_, eE_ > 0 ? eT_/eE_ : 0.0);
    }
};

}} // namespace arc::diag

#endif // ARC_PIC_DIAG_EM_ENERGY_HPP
