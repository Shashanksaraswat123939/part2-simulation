# Part 2 — Physics upgrades: correct flow, the whole car in the CFD, a measurable noise floor

**Status:** plan. **Date:** 2026-09-25. **Repo audited:** `part2-simulation` @ 1590660 (2026-09-02).
Line numbers refer to that commit. OpenFOAM features marked *(verify)* are documented for ESI v2312
and must be confirmed on the v2412 install before building on them.

---

## 1. What runs today

```
stl_half_path
 ├─ forward: cfd_wrapper.run_half_car_cfd -> openfoam_case.invoke
 │    snappy (medium: levels 3-4, underbody +1, 3 layers) -> simpleFoam kOmegaSST, 2000 iters
 │    D20 = mean of last 20 % of force.dat; converged = last p residual <= 5e-3
 │    -> x2 half-to-full -> D20, L, A -> locked race objective -> T, dT/dD20
 └─ adjoint: cfd_wrapper.run_half_car_adjoint -> openfoam_adjoint.invoke_adjoint
      NEW case, NEW snappy mesh, 1000 primal + 1000 adjoint (frozen turbulence, ATC nSmooth 30)
      pointSensNormal -> nearest CFD point per STL vertex (> 5 mm -> frozen at 0)
      x rho x 2.0 (half car) x dT/dD20  -> "s/m^3" -> Part 1
```

Measured, from comments: 25–55 min per iteration; force peak-to-peak 18–27 %; mean drift 2–4 % per decade
of iterations; remesh spread 1.2–3 %; 21 % of STL vertices frozen; adjoint 2.03× faster on 4 ranks, but
production runs on 1.

---

## 2. Findings, most important first

| # | sev | finding | evidence | fix |
|---|---|---|---|---|
| 1 | **CRIT** | **The freestream is ≈ 10,500 times too viscous.** Inlet k from 5 % intensity and ω from a length scale equal to the car length give ν_t = 0.156 m²/s against ν = 1.48e-5. It decays only ≈ 3 % over the 3L upstream run. The car is solved at an effective Reynolds number near 30, not 3×10⁵. Every drag, lift, oscillation, residual plateau and sensitivity in the repo was measured in this flow. The test suite pins k = 1.5. | `openfoam_case.py:244-258, 1021, 1070, 113`; `tests/test_openfoam_case.py:107-112`; recomputed here: ν_t/ν = 10,561 | a car moving through still air: I ≈ 0.5 %, and ω set from a target ν_t/ν of 1–5 (ω = k / (5ν)); assert ν_t/ν < 20 at the inlet; then **re-measure everything** (mesh study, oscillation, drift, adjoint stability) before tuning any threshold |
| 2 | CRIT | **The CFD object is a bare body with open wheel wells.** No wheels, supports, halo/helmet, cartridge protrusion, wings, tether guides; four empty cavities where the wheels sit. | `stl_assembler.py:46-51`; Part 1 keep-out columns | §4 below |
| 3 | HIGH | **The thrust curve is a simulated trajectory.** force = mass × acceleration to 1e-15 on every row, starting at 0.22 N and peaking at 8.5 N at 0.18 s; a punctured cartridge peaks in 10–20 ms. The sanity checker exists and has no production caller. | `co2_thrust_data.csv`; `race_objective_adapter.py:289` | load-cell trace of real Denford cartridges; store its hash in every record; label race times "relative" until then |
| 4 | HIGH | **µ and wheel MOI have no measured value, and the defaults in reach differ 40× and 10×.** `SearchConfig` defaults µ = 0.4 (a sliding coefficient; ≈ 0.59 N of friction on a 150 g car, comparable to all the drag) and I = 1e-6 kg·m² (≈ 20 g of effective mass). `fit_mu_from_track_test` raises NotImplementedError. | `bayesian_outer_search.py:156-157`; `stage1_search.py:75`; `calibration.py:119` | coast-down for µ; I from Part 4's wheel geometry; both required arguments, no defaults; warn when µ > 0.1 |
| 5 | HIGH | **Two meshes and two primal solves per iteration, with different numerics.** The gradient is linearised around a different D20 on a different mesh than the value it is paired with. | `openfoam_adjoint.py:655-746`; schemes differ (`openfoam_case.py:1357-1371` vs `openfoam_adjoint.py:1087-1151`) | one case: forward primal, then adjoint started from its mesh and fields (≈ 100 primal iterations), one scheme set, D20 read from the same solve. Halves wall-clock |
| 6 | HIGH | **The sensitivity's units are asserted, not checked, and now decide the aero/mass balance.** If `surfacePoints` is a per-area density the chain gives s/m³; if it is a per-point total, the aero term is ≈ 1e6 too small and mesh-density dependent. `adjoint_contract.py` still says the magnitude is inert. | `cfd_wrapper.py:445-459`; `phi_updater.py:663-665`; `adjoint_contract.py:27-32` | read `sensitivitySurfacePoints.C` on the machine; then the finite-difference check in #7 |
| 7 | HIGH | **Sign and gain rest on one 2-iteration run whose change (−5.4 %) was smaller than that mesh's remesh spread (7.7 %)**, in the flow of #1. | `phi_updater.py:706-729` | inflate the body 0.2 mm along the normal, solve once, compare with Σ s·A·δ·ρ. Settles sign, area convention and the ρ and 2× factors together |
| 7a | HIGH | **The half-car ×2 on the sensitivity probably double-counts on the unified path.** The half domain with a symmetry plane is already the symmetric car. Moving area dA at a point and its mirror changes full-car drag by 2·s·dA and displaced volume by 2·dA, so the per-area aero density is w·s, not 2·w·s. Part 1 splats onto both halves and adds ρ·dT/dm per local area, so aero enters 2× too strong against mass. `adjoint_contract.py:27-32` still calls the magnitude inert. | `adjoint_contract.py:33`; `cfd_wrapper.py:458`; `phi_updater.py:652` | settle with the finite-difference test in #7 before changing it; expect 1.0 on the unified path |
| 7b | HIGH | **The background mesh is rebuilt from each STL's own bounding box.** Domain, cell size (L/12), `locationInMesh` and the inlet length scale all follow the STL bounds, so a 65 nm change rescales every cell. This is the likeliest main cause of the 1.2–3 % remesh spread and the 7.7 % jump. | `openfoam_case.py:995-999` | freeze domain, cell size, `locationInMesh` and inlet turbulence from the regulation envelope for the whole run |
| 7c | HIGH | **Sensitivity mapping searches every mesh point, most of them interior zeros.** STL vertices near sharp edges, the ground gap or the symmetry plane can snap to a zero-valued interior, `lowerWall` or symmetry point and be frozen silently. This inflates the 21 % "unmapped" figure. | `openfoam_adjoint.py:853, 893` | build the tree from `car`-patch points only (from `polyMesh/boundary`); interpolate over the 3 nearest |
| 8 | HIGH | **The convergence gate is one residual sample; the statistic that matters is not gated.** The last p residual (the corrector's) against 5e-3 on a 2.1–3.2e-3 plateau is a coin flip; mean stderr and drift only warn. | `cfd_wrapper.py:47, 68, 297-331`; `openfoam_case.py:685-695` | residual = median of the last 10 %; gate on drift < stderr and stderr ≤ 1 %; when drift dominates, continue the run from `latestTime` instead of failing the candidate |
| 9 | HIGH | **Nothing makes consecutive solves comparable.** Fresh castellation each time; sub-micron changes move cell boundaries. | `openfoam_case.py:1180` | §3 below |
| 10 | HIGH | **Frozen-turbulence adjoint on a separated body.** Its justification ("only the sign survives normalisation") went stale when normalisation was removed; the SIGFPE that forced it happened in the flow of #1. | `openfoam_adjoint.py:194-229` | retry `adjointkOmegaSST` after #1; if it still fails, keep frozen turbulence and down-weight regions with reversed wall shear |
| 11 | HIGH | **The ride-height gap is 2–3 cells, wall-functioned.** 0.61 mm cells in a 1.5 mm gap; first layer centre at y+ 14–28. | `openfoam_case.py:84-91, 464-469` | underbody level 2–3 (≥ 5 cells), thinner layers in the box; the box is small, the cost modest |
| 12 | MED | y+ 18–239 with wall functions, fully turbulent at Re 3×10⁵; y+ parsed then dropped and never gated. | `openfoam_case.py:716`; part3 `pipeline_interface.py:346-353` | gate y+min ≥ 20 on the wall-function path; `kOmegaSSTLM` with y+ < 1 on the body for final validation *(verify field set-up against the T3A tutorial)* |
| 13 | MED | Lift is differentiated then excluded (w_L = 0). Its weight relative to drag is µ: 0.5 % at µ = 0.01, 20 % at the default 0.4. | `adjoint_contract.py:148`; `race_objective.py:423-426` | after µ is measured, add a weighted `lift` objective beside `drag` if it is above noise |
| 14 | MED | Early records linearise the objective around a 150–160 g car; dT/dD20 there is far below its value at 48 g, which starves the aero channel. | `tests/test_candidate_record.py:255`; `phi_updater.py:711` | Part 1's ballast model holds the car at 48 g; warn when car mass > 80 g |
| 15 | MED | Single-point aero at 20 m/s; a constant-acceleration launch from the sheet's 0.22 N; no tether friction, pod release or reaction. Shifts absolute T, not ranking. | `race_objective.py:389, 418-423` | reference speed = distance-weighted mean speed (≈ 21–23 m/s); document the rest |
| 16 | MED | COM terms: nine placeholder points; the fabricated com_x term stays inside T_raw (adapter zeroes only its subtraction); a sign comment is reversed. | `race_objective.py:93-110, 487-489`; `race_objective_adapter.py:102-107, 250` | subtract com_x in the adapter; set its k to 0 until measured; fix the comment |
| 17 | MED | **Cartridge mass is inconsistent with the regulations.** Appendix vi: full Race Power Pack 28.9–29.4 g. The objective carries 23 g inside `car_weight_kg` as the shell and adds 7.9 g of propellant: 30.9 g at launch. The CAD split (15.25 g steel + 7.97 g CO2) says 23 g is the full cartridge. | `race_objective.py:256-314`; `fixed_hardware.py:722-728` | weigh a full and an empty cartridge; set shell and charge from that |
| 18 | MED | A record cannot reconstruct its run: no CFD or adjoint config, cell count, y+, OpenFOAM version, STL or thrust hash, µ/MOI; logs deleted on success; two penalty fields dropped on read. | `candidate_record.py:171-213, 270-294`; `openfoam_case.py:1347-1348` | `cfd_config`, `adjoint_config`, `provenance` dicts in every record |
| 19 | MED | `D20_half = abs(fx)` hides a reversed or diverged solve. | `openfoam_case.py:1327` | return fx; the D20 ≥ 0 guard then catches it |
| 20 | MED | Blockage 4.7 % with slip walls. Background cells are 19 mm, so larger margins are nearly free. | `openfoam_case.py:304-305` | 10× lateral and vertical margins |
| 21 | MED | The half-STL's symmetry cap is handed to snappy as a wall lying on the symmetry patch: feature edges and refinement along the centreline, a snapping instability. | `stl_assembler.py:49-51`; `openfoam_case.py:962-979` | drop cap triangles when writing `car.stl` (after the watertight check) |
| 22 | MED | 21 % of STL vertices frozen; which ones is not recorded; one CFD value copied to many vertices. | `openfoam_adjoint.py:915-950` | decimate the STL to the CFD facet size before the adjoint; save the frozen mask; prefer per-face sensitivities mapped by area |
| 23 | MED | Production runs single-core. | `openfoam_case.py:83`; `openfoam_adjoint.py:107`; `cfd_wrapper.py:200, 355` | default `n_subdomains` to half the cores |
| 24 | MED | `mesh_validation.py` and `calibration.py` have no caller; "laminar" comparison now runs SST; calibration's CSV schema cannot read the objective's; the mesh study result (1.1 % over 19× cells) is not stored. | grep | delete the dead helpers; keep the mesh-study CLI and store its JSON as provenance |
| 25 | MED | `compute_adjoint_objective` returns ×2 while its docstring says ×0.5; no production caller; `CHANGES_BY_GLM.md` still documents 0.5. | `adjoint_contract.py:64-90` | delete or correct |
| 26 | LOW | `race_objective_hash.txt` is stale and unchecked; "locked" is untrue. | hash differs | regenerate and check, or delete |
| 27 | LOW | Dead field `force_oscillation` on `FullCarQuantities`; `forceCoeffs` output never read; Cm uses a √A placeholder length. | `physics_contract.py:136-177`; `openfoam_case.py:614-631` | remove, or use W for Cm |
| 28 | LOW | NaN forces pass the `< 0` guards. | `physics_contract.py:111-114` | `math.isfinite` checks |
| 29 | LOW | `cfd_case_template/README.md` and BUILD_REPORT describe a different pipeline (laminar default, v2206, R_WHEEL 0.015). | files | rewrite |
| 30 | LOW | Tests shell out to `merge_results.py`, which is not in any repo. | `tests/test_merge_results.py:270` | commit the script |
| 31 | LOW | ASCII-only STL parsed 4+ times per solve (4.3 s each, 14.8 s manifold check). | `cfd_wrapper.py:154-158` | parse once; accept binary |

**Earlier audit claims checked:** COM extrapolation (fixed as a clamp), mesh-study false pass (fixed, module
dead), adjoint scaling (resolved as 2.0, two docs still say 0.5), lifecycle at construction (fixed),
thrust-CSV sanity (function exists, never called, so not effectively closed), test runner (fixed).

**Tests:** 12 pure-Python modules; none touches output from a real OpenFOAM run, and there is no captured
fixture from one (the "real" formats are retyped 5-line snippets). Nothing checks the physical values the
dicts produce, which is how #1 passed every test. Add: real `force.dat`, log tail, `yPlus.dat` and a
`pointSensNormal` slice as fixtures; an inlet ν_t/ν < 20 test; a unit-gain test of the sensitivity chain.

---

## 3. Noise floor — make consecutive solves comparable

Order matters. Each step is cheap and only the last two are real work.

1. **Fix the inlet (#1).** Much of the measured unsteadiness and drift may belong to the wrong flow.
   Re-measure oscillation, drift and remesh spread first.
2. **Deterministic meshing (#7b).** Same background grid origin and size for every candidate of a run (fix the
   box from the envelope, not from each STL's bounds), same `locationInMesh`, same inlet length scale, same
   feature settings. This is probably the single biggest noise reduction available, and it is a few lines.
3. **One case per iteration (#5)** — removes the second mesh.
4. **Warm start** from the previous candidate's fields (`mapFields`), so the transient is short and the
   averaging window long.
5. **Morph, do not remesh.** Two routes:
   - **Route A (inside OpenFOAM):** `adjointOptimisationFoam` with `optimisationManager steadyOptimisation`,
     `designVariables { type shape; shapeType volumetricBSplines; sensitivityType shapeFI; patches (...); }`,
     morphing boxes in `constant/dynamicMeshDict` (`volumetricBSplinesMotionSolver`), BFGS/SQP update with line
     search, and a `partialVolume` objective as a volume constraint. The mesh is morphed between cycles, never
     rebuilt, so cycle-to-cycle drag differences are free of remesh noise. *(verify: all documented in ESI's
     adjointOptimisationFoam manual v2312 and shipped in the sbend/naca0012/motorBike tutorials.)*
     Morphing boxes map naturally onto Part 1's parametric stations.
   - **Route B (Python-driven):** keep iteration 0's mesh and move the `car` patch points by Part 1's surface
     displacement with `displacementLaplacian`; remesh only when `checkMesh` fails or every N iterations.
6. **Measure the floor.** Solve one geometry three times with tiny perturbations; the spread is the error bar
   Part 3 uses for every decision.
7. **URANS** (`pimpleFoam` + `fieldAverage`) only for 2–3 finalists.

Target: repeat-solve spread below 1 % of D20 at the fidelity used for descent.

---

## 4. The whole car in the CFD

### 4.1 Patch layout

| patch | contents | primal U BC | adjoint Ua BC |
|---|---|---|---|
| `car` | body (Part 1), cap triangles removed | noSlip | adjointWallVelocity |
| `nose` | printed nose (Part 4) | noSlip | adjointWallVelocity |
| `wingF`, `wingR` | wings and endplates (Part 4) | noSlip | adjointWallVelocity |
| `supports` | wheel supports, axle stubs, tether guides | noSlip | adjointWallVelocity |
| `halo` | halo + helmet | noSlip | adjointWallVelocity |
| `cartridge` | protruding cartridge (≥ 5 mm) | noSlip | adjointWallVelocity |
| `wheelFront`, `wheelRear` | right-side wheels, sunk 0.2–0.3 mm into the track | `rotatingWallVelocity`, origin at the axle, axis (0 1 0), ω = −U/R ≈ −1415 rad/s | `adjointRotatingWallVelocity` *(verify: listed in the manual; no shipped tutorial)* |
| `lowerWall` | rolling road | fixedValue (U 0 0) | adjointWallVelocity |

Sign check for ω: with the road moving +x in the car frame, the contact point moves +x, so the wheel top moves
−x; with axis +y that needs ω < 0. Confirm on the machine by sampling U at the wheel top ≈ (−20, 0, 0).
Closed wheels only; an open-spoke wheel needs an MRF zone to be honest.

Each patch is its own named STL solid with its own feature file, refinement level (wheels one level finer),
and layers. Standard snappy syntax.

### 4.2 Objective and sensitivities

- Objective `drag` on **all** patches, so the body adjoint sees wheel and wing interference drag.
- Design patches separate from objective patches: `car` and `nose` for the body, wing patches for Part 4's
  projected dJ/dp. The point sensitivity field already covers every mesh point, so mapping is unchanged.
- Per-patch `forces` function objects for the drag breakdown Part 5 reports.
- Frontal area from all solids combined.

### 4.3 Order

Body + wheels (static, then rotating) first: three solves on the leader (body / + static wheels / + rotating)
tell you how much of the drag the wheels carry. Then supports, halo, cartridge. Then wings, once Part 4 has them.

---

## 5. Objective inputs to measure (phase 0 of the master plan)

| input | today | measure |
|---|---|---|
| thrust curve | simulated trajectory | load cell, several cartridges, store mean and spread |
| cartridge shell / charge | 23 g + 7.9 g, inconsistent with Appendix vi | scale, full and empty |
| µ | 0.010 or 0.4 depending on caller | coast-down or incline test on the track |
| wheel MOI | 1e-7 or 1e-6 depending on caller | from Part 4's wheel geometry (exact) |
| COM penalty | nine placeholder points | the ballast experiment already planned |

Each goes into one constants file with its source, and every record stores the hash of the thrust data.

---

## 6. Build order and the test that closes each step

| # | change | test |
|---|---|---|
| 1 | fix the inlet turbulence (#1) and its test | ν_t/ν < 20 at the inlet; leader re-solved; new D20, oscillation, drift recorded |
| 2 | fixed background mesh from the envelope (#7b); 10× domain margins, underbody level 2–3, cap removal (#11, #20, #21) | same STL meshed twice gives identical cells; mesh study at three resolutions within 2 % |
| 3 | median residual gate + drift/stderr gate + continue-from-latestTime (#8) | a known-good solve is never failed on residual noise |
| 4 | one case per iteration, unified schemes (#5) | D20 from the adjoint primal matches the forward D20 within the error bar |
| 5 | units check, half-car factor, patch-only mapping, finite-difference gain test (#6, #7, #7a, #7c) | predicted and measured ΔD20 agree within 20 % |
| 6 | multi-patch export and wheels (§4) | per-patch drag; wheel share of total measured |
| 7 | noise floor steps (§3.2–3.6) | repeat-solve spread < 1 % |
| 8 | Route A morphing optimisation on the leader | 10 cycles reduce D20 by more than 2σ without remeshing |
| 9 | provenance in records (#18) | a record alone reproduces its solve |
| 10 | measured inputs (§5) | placeholders gone from the constants file |

Step 1 is one function and one test, and it changes every number the project has produced. Do it before
anything else in any part.
