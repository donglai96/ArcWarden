# δf restart day — triggered chirping in the true-2D dipole (2026-08-20)

User directive (remote): restart the δf-in-2D project; final goal = a CLEAR
chirping element; iterate until achieved; run-card figure before every big run.

## 1. Unblocking: Gauss closure (commit 6d9cc30)

The 08-20 retraction (live δf continuity residual 350× the frozen floor)
mandated a correction before any E∥/element claims. Landed and gated — see
CONTINUITY_GATE.md "GAUSS CLOSURE LANDED + GATED" and scripts/gauss_gate.py
(4/4 PASS). Key numerical finding: the properly-paired divE−ρ monitor
(filtered ρ_hot + cold-charge ledger, interior-only norm) shows the
ACCUMULATED Gauss error is amplitude-tracking, not secular — 2.8e-6 after
8000 driven steps — i.e. the per-step weight residual largely oscillates
rather than integrates. Cleaning halves it at zero measured physics cost.

## 2. Trigger pilot p3_trig = PASS (the WHY_NO §5-A decisive experiment)

Deck decks/warden2d_p3_trig.ini (δf + antenna, L0=1086 = x10-equivalent
corridor, w0=0.06=0.30Ωe_eq OPEN corridor, amp 2e-4, toff 800, 25k steps,
gauss_clean=20, seed 20260820). Frozen readouts, results:

- P-A1 placement: δB/B0eq(±5°, t≈1000) = 8.5e-3 — marginally above the
  3–8e-3 window, sitting AT B_opt. Reported as marginal-high.
- P-A2 DECISION = **UPSWEEP**: ridge climbs 0.30 → 0.45–0.48 Ωe_local by
  t=3750 at +5° and +10°; W_EM grows ×6.7 AFTER antenna off
  (2.5e-3 → 1.67e-2). The 2D corridor physics is intact; the lateral-escape
  stall applies to noise-seeded spontaneous growth, not to an
  above-threshold coherent seed.
- P-A3 validity: gauss_res ≤ 2.1e-6, wdrms end 0.172 < 0.25, healthy.
- Transport: element sustains ~6e-3 out to ±20° (arrival-time ordered:
  ±5° at t≈1000, ±10° ≈ 2400 — v_g ≈ 0.08c; an early "amplitude collapse
  past 5°" read is a TRAVEL-TIME artifact, logged as a measurement trap).
- Corridor bookkeeping: with the consistent lre_eff = 1330 mapping
  (2D B/Beq = 1+3(s/L0)² ⇒ a = 3/L0² = 4.5/lre_eff²), B_th(0.30) =
  1.6e-3, B_th(0.25) = 5.5e-3 ≈ B_opt(0.25) (marginal selector). The
  CHIRPING_DIAGNOSIS table's "2D x10" row used lre = 1086 in the 1D
  formula — inconsistent with its own map; corrected here.

Limitations of the pilot AS a "clear element": the drive ran to B_opt while
on, so packet and element mix; ridge instantaneous width ~0.1 Ωe; single
element only.

## 3. Production ladder

- **ARM-1 p4_el1** (decks/warden2d_p4_el1.ini, running): threshold-level
  trigger (amp 1e-4 → seed ≈ 4e-3 > B_th=1.6e-3, toff 600), t=6000.
  Frozen readouts P-B1..B4 in the deck header; judgment tool
  scripts/element_gate.py FROZEN before the data.
- **ARM-2 element train** (pending ARM-1): new [antenna] tper knob
  (envelope cycles, carrier phase continuous) → repeated triggered
  elements = the discrete-element-sequence figure. Run card to user first.
- δf validity budget: wdrms grew ~3e-5/ωpe⁻¹ late in the pilot from 0.17
  at t=3750 → the 0.25 rms gate bounds single-seed runs to t ≲ 6000.
  Longer runs need either the multi-criteria battery to rule (tail
  fractions stayed ≪ gates) or a τ_D arm — decision deferred until an
  element catalog exists.

## 4. ARM-1 ladder outcome (same day): CLEAR ELEMENT ACHIEVED

Calibration ladder (frozen bars untouched; amp/toff recalibrated between
arms from measured points — buildup is LINEAR in amp, toff-insensitive
beyond ~600):

| arm | amp | seed@±5°(t 700-900) | verdict |
|---|---|---|---|
| p4_el1  | 1e-4   | 1.37e-3 (< B_th 1.6e-3) | narrow riser 0.34→0.40 FADES (sub-threshold control, unplanned but clean) |
| p4_el1b | 1.5e-4 | 2.06e-3 | P-B2 PASS: riser 0.30→0.50, dies slowly; seed/transport under bars |
| p4_el1c | 3e-4   | 4.12e-3 (predicted 4.1e-3) | **FULL PASS 4/4** |

p4_el1c: narrow monotonic riser 0.30 → 0.50 Ωe_local over Δt≈3000/ωpe at
±5° AND ±10° (N=4096 STFT), amplitude peaks 1.57e-2 at ±5°, transits
1.16e-2 at ±10°, 8.9e-3 at ±20°, N/S symmetric; wdrms max 0.212 < 0.25,
gauss_res 4.7e-6. The three-arm ladder doubles as a measured threshold
bracket: 1.4e-3 dies / 2.1e-3 marginal / 4.1e-3 robust vs B_th(0.30) =
1.6e-3 (Omura, lre_eff = 1330) — threshold confirmed within a factor ~1.3.

GATE AMENDMENT (v2, flagged post-hoc): P-B3 transport was written as
"mean over the last 500/ωpe" — but the element is a transient; that
window measures the channel AFTER passage (travel-time-trap family).
v2 = peak of the 200/ωpe-smoothed envelope, bars unchanged. Both 1b and
1c pass B3 under v2; ARM-1 (sub-threshold) still fails everything but B4.
