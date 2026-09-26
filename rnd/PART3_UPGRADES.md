# Part 3 — Optimiser upgrades: from 150 days of CFD to a few dozen meaningful solves

**Status:** plan. **Date:** 2026-09-25. **Repo audited:** `part3-simulation` @ 4b444cd (2026-09-02),
cross-checked against part1 @ 3db4607 and part2 @ 1590660. Line numbers refer to those commits.

---

## 1. What runs today

### 1.1 Call path for one d_halo

```
run_two_stage.py                      (driver — referenced in 9 places, exists in NO repo)
 ├─ Stage 1  stage1_search.run_stage1 -> scalars_for_stage2 (W, x_front, cargo, seed field)
 └─ Stage 2  orchestrator.run_stage2_dhalo_search            (orchestrator.py:120)
     └─ wheelbase_sweep.run_d_halo_sweep                     (wheelbase_sweep.py:380)
         └─ per d_halo: optimize_single_w                    (wheelbase_sweep.py:126)
             └─ per round: run_candidates_parallel(run_inner_loop)
                 └─ per iteration: gates -> mass -> CFD -> objective -> T3.6 -> adjoint
                                   -> apply_adjoint_to_unified -> record
```

`merge_results.py` and `run_optimization.py` are also referenced and missing. The project's
`../ARCHITECTURE.md`, cited by README and orchestrator, is not on disk either.

### 1.2 Cost

Each inner iteration meshes twice and solves the primal twice: `run_half_car_cfd` (snappy + simpleFoam,
2000 iters) then `run_half_car_adjoint` (snappy again + 1000 primal + 1000 adjoint). The code quotes 25–55 min.

| budget | pairs | serial time |
|---|---|---|
| live run: 6 d_halo × 25 iterations, 1 candidate | 150 | ≈ 100 h (≈ 17 h on 6 shards) |
| defaults: coarse 6 × 5 candidates × 3 rounds × 10 | 900 | ≈ 600 h |
| defaults + refine (15 values × 10 × 3 × 10) | ≈ 5400 | ≈ 150 days |

### 1.3 How far the car can move

The CFL cap is 0.3 × 0.5 mm = 0.15 mm per iteration. The measured rate is about 40× lower:
0.057 g per iteration near the floor, against ≈ 2.3 g if the whole surface moved at the cap
(bayesian_outer_search.py ≈ 333). Twenty-five iterations reshape the car by a fraction of a millimetre.

The leader in the image sits at 48.20 g, below the 48.4 g entry threshold of the aero phase. **It never
entered the aero phase.** Its shape is the Stage-1 mass carve plus masks.

---

## 2. Findings, most important first

| # | sev | finding | evidence | fix |
|---|---|---|---|---|
| 1 | CRIT | Step is throttled by adjoint speckle: dt is sized on the max |V|, and after the p99.9 clip about 1000 spiky cells still set dt for 8 M cells | `phi_updater.py:412-413, 927-938`; "bulk moving 0.017 cells/step while a spike moves 0.3" | size dt on a band statistic (RMS or p90 of |V| where |φ|<2dx); smooth the sensitivity on the surface (Helmholtz/Sobolev filter, radius 2–3 mm) before extension |
| 2 | CRIT | One HJ step per 40-minute CFD+adjoint pair; no sub-steps, no line search; `hj_dt` never binds | `inner_loop.py:683-686`; `optimizer_contract.py:345` | K sub-steps per adjoint up to a trust radius (1–2 mm, about one CFD cell), mass/COM recomputed each sub-step (free), backtrack on the linearised objective |
| 3 | CRIT | No mass-conserving aero step. The aero phase sets `w_mass=0`, which also switches off the T3.6 barrier, so the car flaps between a mass-only and an aero-only descent | `inner_loop.py:360-362`; `phi_updater.py:872`; margins `optimizer_contract.py:166-167` | replace with a constrained step: V_aero − λρ with λ making ∫ρV dA = 0 on the band, or the ballast model of PART1 §3 which removes the need |
| 4 | HIGH | Convergence: gradient-norm stop is dead; D20 stop at 1.5 % sits inside the 1.2–3 % remesh noise and can fire on noise; tracker and phase state reset every evolutionary round | `inner_loop.py:587-606`; `optimizer_contract.py:171`; `convergence.py:169`; `wheelbase_sweep.py:213` | converge on D20 change relative to the measured `force_mean_stderr` (already in `CFDOutcome`, unused); keep tracker state across rounds; delete the gradient-norm stop |
| 5 | HIGH | `perturb_phi_fields` mutates in place, so a survivor and its perturbed clone share one geometry | `pipeline_interface.py:812-817`; `wheelbase_sweep.py:288-296` | `copy.deepcopy` before perturbing; the population test's dict fake copies, so it cannot see this |
| 6 | HIGH | Evolutionary "kill and perturb" on a level set: perturbation is 10 % of φ RMS, several mm, versus sub-mm descent; selection is on differences inside the 15 ms noise | `evolutionary.py:100-101, 58` | drop it from Stage 2; structural diversity moves to Part 5's parametric search |
| 7 | HIGH | d_halo is swept as N separately optimised cars; the halo loft is worth single-digit ms, below the 15 ms ranking noise | `orchestrator.py:175-179` | optimise one car, remap it to each d_halo, one forward solve each, repeated 3× for the error bar: 18 solves instead of 150 pairs. Later, d_halo becomes a continuous variable in Part 5 |
| 8 | HIGH | Stage 1 hands over 0.8–4.7 g above the floor (5 g margin minus the remap loss), and Stage 2 burns CFD iterations removing it | `bayesian_outer_search.py ≈ 317-345` | run the free no-CFD mass descent after the remap, before iteration 1 |
| 9 | HIGH | Stage 1 optimises `0.5 m/0.055 + 0.3 h/0.025 + 0.2 W/130`; at the floor this is a hand-set reward for short W. The leader's W = 120.3 mm is the predicted corner | `bayesian_outer_search.py:295, 615-620`; `stage1_search.py:4` | score with the real objective at nominal D20 (`make_race_objective_cargo_scorer` already exists, `stage1_search.py:73-102`) |
| 10 | HIGH | Final ranking uses T_raw among lifecycle-valid states and can pick an underweight car; Part 2's merge test ranks on T_pen and excludes underweight cars | `objective_policy.py:71-72`; `orchestrator.py:176`; part2 `tests/test_merge_results.py:140-184` | add competition mass ≥ 48 g to `is_fully_valid`; one ranking policy in one place |
| 11 | HIGH | Up to 30 % of the surface gets zero sensitivity (21 % measured) because the STL is shipped undecimated when decimation fails the angle gate | `openfoam_adjoint.py:118-124`; `pipeline_interface.py:602-620, 843-855` | coarser extraction grid, a quality-preserving decimator, or area-weighted face-to-vertex mapping; the Part 2 morphing path removes the mapping entirely |
| 12 | MED | Two meshes and two primal solves per iteration | `pipeline_interface.py:897-918` | take D20 from the adjoint run's primal; one case per iteration |
| 13 | MED | No coarse-to-fine, though `remap_geometry` exists and its docstring measures a 12× / 60× win | `unified_phi.py:704-716` | spacing schedule 2 mm → 1 mm → 0.5 mm with matching CFD resolution |
| 14 | MED | No surrogate or step model on the CFD path; each iteration's data is used once | `inner_loop.py:673` | trust-region model from the last few (D20, dJ/dp) pairs |
| 15 | MED | `best` is the min-T_pen iteration (noise-selected) but the returned field is the final one | `inner_loop.py:382-384`; `wheelbase_sweep.py:310-320` | return the field that matches the reported best, or define best as the last feasible iteration |
| 16 | MED | Manufacturing penalty only affects ranking; `rule_margin_penalty_s` is always 0 | `inner_loop.py:105-146`; `phi_updater.py:689-696` | keep `enforce_machinability` as the projection; delete the dead penalty field |
| 17 | MED | The two validation flags are self-attested CLI switches; no experiment is recorded anywhere; `--smoke` also relaxes convergence and mapping | `optimizer_contract.py:335-336`; `test_stage2_dhalo_sweep.py:302-306` | the flags are set by Part 5's calibration loop from stored measurements, not by a switch |
| 18 | MED | µ and wheel MOI have no production source; defaults in reach disagree 40× (µ 0.4 vs 0.010; MOI 1e-6 vs 1e-7) | `bayesian_outer_search.py:156-157`; `stage1_search.py:75`; `optimizer_contract.py:337-338` | one constant pair in `physics_contract`, fitted by `calibration.py`; wheel MOI from Part 4's wheel |
| 19 | MED | Four-grid API and the unified object coexist; `real_bindings` is the dead four-grid path yet README's documented wiring; `combine_gradients` and weight calibration are bypassed | `pipeline_interface.py:169-176, 230-438, 865-868` | delete `real_bindings`, type the interface as the unified geometry, reduce weights to `w_aero, w_mass` |
| 20 | MED | The "Oracle Sweep" caption is not produced by any code in the repos; it matches the missing `merge_results.py`'s BEST line | `wheelbase_sweep.py:81` gives d_halo 43.72 at W 120 | commit the driver and merge scripts |
| 21 | MED | ≈ 300 MB per iteration on disk (ASCII full STL ≈ 235 MB); the promoted "converged" state never reaches disk | `pipeline_interface.py:445`; `inner_loop.py:431-443` | binary STL, full STL only at the end; write the promoted record |
| 22 | MED | Default d_halo sampling puts 2 of 6 values in a legal-but-unbuildable band (36–75 mm) | `orchestrator.py:145-153` | filter with `unified_phi.cargo_placement_is_buildable` |
| 23 | MED | Refinement multiplies cost 5× on differences below noise | `orchestrator.py:182-188` | `refine=False` until the noise floor is below the effect |
| 24 | LOW-MED | Robustness hooks never run; stability Tier 1 recorded, unused | `orchestrator.py:84-86`; `robustness.py:141-171` | mesh-refinement spread on finalists (3–6 forward solves) is the one that matters |
| 25 | LOW-MED | Warm-start flag is overridden whenever a Stage-1 seed exists | `pipeline_interface.py:792` | delete the flag |
| 26 | LOW | ΔT_pen stop (4 ms) is measured on a quantity whose machinability penalty flickers by tens of ms | `optimizer_contract.py:133`; `inner_loop.py:123-124` | converge on D20 and mass, not on T_pen |
| 27 | LOW | Solver timeouts are recorded as design failures and can blacklist a d_halo | `wheelbase_sweep.py:255-266` | separate infrastructure failure from geometry failure |
| 28 | LOW | `CFDOutcome` stored where `FullCarQuantities` is expected; `force_mean_stderr` and `force_drift` lost on read | `inner_loop.py:757`; `candidate_record.py:127-131` | add the fields to the record type |
| 29 | LOW | Decimation backoff doubles to 960 k then ships the original; the angle envelope is stricter than anything the solver needed | `pipeline_interface.py:566, 848-855` | set the envelope from the two solved cases |
| 30 | LOW | Dead code and stale docs: `run_full_search`, `real_bindings`, `combine_gradients`, `zero_penalties`, `failure_penalty_for_state`, W-sweep helpers; README entry point; d_halo range comments | listed in the audit | delete; update README |

---

## 3. The redesign

### 3.1 Principle

Spend CFD only where it is the only way to learn something, and make each solve move the car as far as its
linearisation is valid. Everything that is cheap (mass, COM, inertia, legality, the ballast balance) runs
between solves, many times.

### 3.2 New Stage structure

| stage | what | CFD | replaces |
|---|---|---|---|
| **S0 components** | wheels, bearings (Part 4) | none | nothing (new) |
| **S1 layout** | (W, x_front, d_halo, structural switches) scored by the **real objective** with a drag surrogate from frontal/wetted area; body at the floor via ballast | none | proxy BO |
| **S2 structure** | Part 5 screens discrete configurations at coarse CFD | ≈ 30–50 coarse forward solves | d_halo sweep of separate cars |
| **S3 descent** | parametric dJ/dp steps with line search inside OpenFOAM's morphing loop, then bounded level-set polish | ≈ 10–15 adjoint cycles | 25–100 HJ steps per d_halo |
| **S4 validate** | fine mesh / longer averaging / unsteady on 2–3 finalists; mesh-refinement spread | ≈ 6–10 solves | robustness hooks that never ran |

Total ≈ 50–80 solves, most of them coarse, against ≈ 900–5400 pairs today.

### 3.3 Inner loop, rewritten

Per outer iteration k:

1. **Legality and mass (free).** Build the body from parameters p_k + level-set residual; ballast fills to
   48.0 g + margin; if the container overflows, hollowing channels grow (PART1 §4).
2. **One CFD case** (primal + adjoint together, morphing mesh from the previous iteration, warm-started fields).
3. **Project** the surface sensitivity onto parameters: dT/dp = dT/dD20 · Σ s_i (∂x_i/∂p · n_i) A_i
   plus analytic mass/COM/inertia terms. The ∂x/∂p come from finite differences of the parametric geometry,
   no CFD.
4. **Step** with a trust region: L-BFGS or SQP in parameter space, constraints (legality margins, mass
   balance) linearised, step length capped by the trust radius.
5. **Accept or shrink** by comparing predicted and measured ΔD20 at the next solve, inflated by the measured
   error bar. If the measured change is within noise, count it as zero, not as a success.
6. **Stop** when the predicted improvement of the best available step is below 2× the noise floor, or the
   budget is used.

Option A (simplest to build first): do steps 2–5 in Python with Part 2's existing single-run adjoint.
Option B (fewer moving parts once proven): configure `adjointOptimisationFoam` in `steadyOptimisation` mode
with volumetric B-spline morphing boxes aligned to the body parameters, and let it run the cycles itself
(PART2 §B). Start with A, move to B when the parameter set is stable.

The level set comes back only in the last 3–5 iterations, as a bounded residual (|δφ| ≤ 1–2 mm) with the
fixes from findings 1–3, to catch local detail the parameters cannot express.

### 3.4 Noise-aware decisions everywhere

`force_mean_stderr` and the measured repeat-solve spread become the unit of every decision:
convergence (finding 4), acceptance (step 5), ranking (a difference below 2σ is a tie), refinement
(finding 23). The error bar is recorded with every result and printed beside every ranking.

### 3.5 Parallelism that exists

Use MPI inside solves (measured 2.03× at 4 ranks) and run independent coarse screening solves in parallel
across machines or shards. Keep candidate threads at 1 until the aliasing bug (finding 5) is fixed and a
test covers `max_workers > 1`.

---

## 4. Build order and the test that closes each step

| # | change | test |
|---|---|---|
| 1 | commit `run_two_stage.py` and `merge_results.py` to the repo | the tests that import them run |
| 2 | fix finding 5 (deepcopy) and finding 10 (legal ranking) | population and ranking tests with real geometry objects |
| 3 | band-statistic dt + sensitivity smoothing (finding 1) | spiky synthetic sensitivity: band-median displacement ≥ 0.5 × CFL × dx |
| 4 | sub-steps + trust radius (finding 2) | mass removed per adjoint ≥ 10× today on a no-CFD fake |
| 5 | constrained aero step or ballast model (finding 3) | aero-only fake sensitivity moves the shape at constant mass |
| 6 | noise-floor convergence (finding 4) | random-walk D20 at 2 % never triggers convergence |
| 7 | single CFD case per iteration (finding 12) | D20 from the adjoint primal equals the forward D20 within the error bar |
| 8 | remap-only d_halo evaluation (finding 7) | 18 solves give dT/dd_halo with an error bar |
| 9 | real objective in Stage 1 (finding 9) | Stage 1's chosen W moves off the 120 mm corner or proves it belongs there |
| 10 | parametric dJ/dp loop (§3.3, option A) | 10 iterations reduce D20 by more than 2σ on the coarse mesh |
| 11 | delete dead paths, retype the interface (findings 19, 30) | test count holds, no four-grid import left |

Steps 2–6 are small and fix the current pipeline even if nothing else changes.

---

## 5. Constants to move into one place

Every tunable listed in the audit (spacing, CFL, dt, clip percentile, budgets, thresholds, margins, candidate
counts, mesh envelope, CFD iterations and timeouts, unmapped fraction, µ, wheel MOI) goes into one
`run_config.py` with the measurement or regulation behind each value. Today they are spread over at least
fifteen files in three repos, and µ and wheel MOI have no production value at all.
