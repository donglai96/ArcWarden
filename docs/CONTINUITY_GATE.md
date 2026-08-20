# δf continuity gate — frozen criteria (2026-08-20)

Six-arm matrix (decks/cont_*.ini, 201 steps, --contcheck=100, quiet smoke):

| arm | R/||drho/dt|| | note |
|---|---|---|
| full-f fixed-weight | 5.40e-5 | Esirkepov float floor (identity verified) |
| δf FROZEN weights | 1.310e-3 | = δf deposit float floor (defines the floor) |
| δf LIVE weights | 1.310e-3 | == frozen -> variable-weight residual BELOW floor |
| jfilter=0 raw | 4.89e-4 | filter-free floor |
| dt/2 | 1.215e-3 | unchanged -> floor is float noise, not O(dt) |
| ppc 200 | 1.062e-3 | weak ppc dependence |

FROZEN PASS CRITERIA (production δf runs, via --contcheck + contcheck.csv):
  C-A  R_live / R_frozen-floor <= 2.0   (smoke-calibrated; floor rerun
       whenever deck geometry/ppc/dt changes)
  C-B  absolute R/||divJ|| recorded every check; a x10 jump between
       consecutive checks = investigate before trusting E_par physics.
Caveat: at element amplitudes wd excursions grow — C-A is exactly what
--contcheck monitors in-run; if it fails, weight-source correction or
Gauss projection becomes mandatory (audit ruling).
