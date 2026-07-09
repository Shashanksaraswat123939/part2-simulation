# Changes by GLM

**Branch:** `glm/audit-fixes`  
**Repository:** https://github.com/Shashanksaraswat123939/part2-simulation  
**Date:** 2026-07-09  

---

## Overview

All 13 active findings from the Part 2 categorized audit were fixed. Two new physics features (lift-dependent friction and fore-aft COM sensitivity) were added to the locked `race_objective.py` (unlocked with user permission, then re-locked with new hash). The test suite was expanded with regression tests for every fix. Final result: **82 tests passed, 0 failed**.

---

## Commits on `glm/audit-fixes`

| # | Commit | Message |
|---|--------|---------|
| 1 | `c918da7` | Add lift-dependent friction and fore-aft COM penalty to race_objective.py |
| 2 | `f7a5b59` | Fix D1: repair broken run_all_tests.py test runner |
| 3 | `3e44abe` | Fix A.1: add COM height extrapolation guard and sanity bounds |
| 4 | `823603c` | Fix B.1-B.4: mesh independence false-positive and relative spread |
| 5 | `097947f` | Fix C.1-C.3: path traversal, lifecycle validation, CSV error context |

---

## Part 1: Audit Fixes (13 findings)

### 1. `run_all_tests.py` — [D1, HIGH] Fixed broken test runner

**What changed:** Complete rewrite of `run_all_tests.py`.  
**Why:** The old script hardcoded a `tests/` subfolder prefix that didn't exist, silently reported `TOTAL: 0 passed, 0 failed` when every test file failed to be found, ignored subprocess return codes, and never surfaced stderr.  
**How:** Rewrote to use `Path(__file__).resolve().parent / "tests"` for reliable file resolution, check file existence before running, surface stderr and exit codes, and exit non-zero (code 2) when infrastructure failures occur so CI can't mistake "broken harness" for "clean pass."

---

### 2. `race_objective_adapter.py` — [A.1/3.1, CRITICAL] COM height extrapolation guard

**What changed:** Added `COM_HEIGHT_FIT_RANGE_M` constant and range check in `_assert_physical_inputs()`.  
**Why:** The locked `com_height_time_penalty` polynomial was only fitted over `com_height_m ∈ [0.018, 0.042]`. Out-of-range values (e.g. 223m from a units bug) produce penalties on the order of 10^15 seconds — silently, with no error.  
**How:** Added `COM_HEIGHT_FIT_RANGE_M` derived from `_COM_PENALTY_DATA_MM` (not hardcoded, so it can't drift). Range check with small floating-point tolerance (1e-9) for boundary values. Raises `ValueError` with clear message pointing to upstream mass/COM ingestion as the likely cause.

---

### 3. `race_objective_adapter.py` — [A.2, HIGH] Thrust CSV physical sanity validation

**What changed:** Added `validate_thrust_csv_physical_sanity(csv_path)` function.  
**Why:** The locked `_clean_csv_arrays` validates formatting but not physical sanity — negative time, negative force, or non-positive mass all pass through silently.  
**How:** Calls the locked `_clean_csv_arrays` then checks for negative time, negative force, and non-positive mass. Mirrors the "guard the locked file from outside" pattern used for `time_coefficient` and negative mass/mu.

---

### 4. `race_objective_adapter.py` — [A5, LOW] BuildSettings validation wrapper

**What changed:** Added `build_smooth_sheet_model_guarded(csv_path, settings)` wrapper.  
**Why:** `BuildSettings(n_basis=0)` produces an empty RBF basis without a clear error. Since `BuildSettings` is in the locked file, the guard must live outside.  
**How:** Validates `n_basis >= 1` and `n_steps >= 1` before calling the locked `build_smooth_sheet_model`.

---

### 5. `mass_com_ingest.py` — [A.1/3.1, CRITICAL] COM sanity range check

**What changed:** Added `COM_SANITY_BOUNDS_M = (-10.0, 10.0)` and post-calculation check in `ingest_mass_com`.  
**Why:** `ingest_mass_com` computes COM coordinates as a plain mass-weighted average with no range check. A coordinate-origin mixup or units bug produces `com_z_m=223.099` with zero error.  
**How:** After computing `com_x_m`, `com_y_m`, `com_z_m`, checks each against bounds. Raises `ValueError` with message suggesting a units or origin bug. Bounds are wide enough for test fixtures while catching the 223m case.

---

### 6. `mass_com_ingest.py` — [E.1/3.6, LOW] Fixed misleading dead placeholder constants

**What changed:** Removed `CO2_CARTRIDGE_COM` (claimed to raise `NotImplementedError` but didn't). Rewrote comments on remaining constants.  
**Why:** Three module-level constants (`CO2_CARTRIDGE_COM`, `REAR_WING_MASS_KG`, `WHEELS_AXLES_MASS_KG`) were dead code with misleading comments claiming enforcement that didn't exist.  
**How:** Removed the worst offender. Rewrote comments on the remaining two to accurately describe that `FixedHardwareSpec`'s required dataclass fields are the actual enforcement mechanism.

---

### 7. `mesh_validation.py` — [B.1/3.2, HIGH] Mesh independence false-positive fix

**What changed:** Replaced default `cfd_runner=run_half_car_cfd` with `_REQUIRED_RUNNER` sentinel. Made `cfd_runner` a required keyword-only argument.  
**Why:** With the default `cfd_runner=run_half_car_cfd`, resolution labels ("coarse", "medium", "fine") were passed as `reference_speed_mps` (a float parameter) and silently discarded. All three runs produced identical results → 0% spread → false PASS.  
**How:** Calling `run_mesh_independence_study("car.stl")` without supplying a runner now raises `TypeError` with a clear explanation. Converts a silent correctness bug into an immediate, obvious error.

---

### 8. `mesh_validation.py` — [B.3/3.4, MEDIUM] `_relative_spread` zero-mean masking

**What changed:** Replaced mean-based scale with `max(abs(v) for v in values)` in `_relative_spread`.  
**Why:** The zero-mean guard (`if mean == 0: return 0.0`) was reachable for real lift-force data where values oscillate around zero (e.g. `-0.3, 0.4, -0.05` N). These average near zero, triggering the guard and masking genuine mesh non-convergence as a false PASS.  
**How:** Now only returns 0.0 when all values are genuinely near zero (abs < 1e-9). Mixed-sign values with real disagreement correctly report non-zero spread.

---

### 9. `adjoint_contract.py` — [B.2/3.3, HIGH] Adjoint half-car scaling (RESOLVED)

**What changed:** Added `ADJOINT_HALF_CAR_SCALING = 0.5` constant. Updated `compute_adjoint_objective` to scale full-car D20 by 0.5.  
**Why:** The OpenFOAM adjoint runs on the half-car domain (confirmed by user). `param_vector[0]` is full-car D20, so the objective must use half-car D20 to match the half-car surface mesh. Without scaling, sensitivities would be off by 2x.  
**How:** `Objective = w_D20 × D20_full × 0.5 = w_D20 × D20_half`. Updated test `test_adjoint_objective_is_w_D20_times_D20` to verify the 0.5 factor.

---

### 10. `candidate_record.py` — [C.1/3.5, MEDIUM] Path traversal false-positive fix

**What changed:** Replaced `".." in safe_id` substring check with `safe_id in ("..", ".")` exact-match check.  
**Why:** The old guard rejected any candidate ID containing the substring `..` anywhere, including legitimate IDs like `cand..001`. Since path separators are already banned, `..` as a substring inside a single-segment name can never be a traversal mechanism.  
**How:** Two checks: (1) reject path separators (already existed), (2) reject exact `.` and `..` only.

---

### 11. `candidate_record.py` — [C.2/A3, MEDIUM] Lifecycle state validation at construction

**What changed:** Added lifecycle_state validation in `CandidateRecord.__post_init__`.  
**Why:** `write_candidate_record()` did not validate `lifecycle_state` — only `read_candidate_record()` did. The codebase's own write path could create records with invalid states that were only caught later on read.  
**How:** Added `if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES: raise ValueError(...)` in `__post_init__` alongside the existing `setup_logs` length check. Read-time validation remains as defense-in-depth.

---

### 12. `calibration.py` — [C.3/A4, LOW] CSV error messages with row/column context

**What changed:** Replaced list comprehension in `_read_csv_columns` with indexed loop that catches errors and re-raises with context.  
**Why:** `_read_csv_columns` raised `ValueError: could not convert string to float: ''` with no indication of which row or column triggered it. Affects all calibration functions that route through this helper.  
**How:** Catches `(ValueError, TypeError)` and re-raises with file path, column name, and 1-indexed row number (plus header).

---

### 13. `.gitignore` — [D.2/A7, LOW] Added generated artifact patterns

**What changed:** Added `candidates/`, `*.stl`, `*.npy`, `cfd_case_*/`, `*.foam` to `.gitignore`.  
**Why:** No ignore rules existed for candidate database JSONs, STL files, NumPy snapshots, or OpenFOAM case directories that the pipeline generates.  
**How:** Added patterns after existing Python artifact rules.

---

## Part 2: Physics Upgrades (2 new features in `race_objective.py`)

### 14. Lift-dependent friction (normal force from lift coefficient)

**What changed:** `distance_derivatives` now computes `F_fric = mu * (m*g - L(v))` where `L(v) = lift_20_n * v^2 / 20^2`. Added `lift_20_n` to `PARAM_NAMES` (index 6).  
**Why:** Previously friction was `F_fric = mu * m * g` — a constant normal force regardless of speed. In reality, aerodynamic lift changes the normal force: downforce (negative lift) increases normal force and thus friction; upforce (positive lift) decreases it. At higher speeds, this effect becomes significant.  
**How:** Added `lift_20_n` parameter. In `distance_derivatives`, computes lift at current speed using the same quadratic scaling as drag (`L(v) = lift_20_n * v^2 / 20^2`), then subtracts it from `m*g` to get the normal force. Both upforce and downforce are valid (no sign restriction on `lift_20_n`). Updated `make_param_vector` to accept `lift_20_n` parameter.

---

### 15. Fore-aft COM sensitivity (`dT/dx_com`)

**What changed:** Added `com_x_m` to `PARAM_NAMES` (index 7). Added `com_x_time_penalty()` function with quadratic penalty `_COM_X_PENALTY_K * (com_x_m - _COM_X_TARGET_M)^2`. Added penalty to `race_time_seconds`.  
**Why:** Previously `dT_dx_com = 0.0` — the fore-aft COM position had no effect on race time. The user requested that fore-aft COM be considered now.  
**How:** Added `_COM_X_TARGET_M = 0.0` (optimal COM at reference origin) and `_COM_X_PENALTY_K = 0.001` (1ms per mm² deviation). The quadratic penalty is smooth and differentiable, so JAX can compute gradients through it. The `adapt_gradients` function in `race_objective_adapter.py` now maps `com_x_m → dT_dx_com` (non-zero from JAX autodiff, no longer hardcoded to 0.0).

---

## Part 3: Test Updates

### New regression tests added

| Test | File | What it verifies |
|------|------|-----------------|
| `test_com_height_out_of_range_rejected` | test_race_objective_adapter.py | COM height 223.099m raises ValueError |
| `test_com_height_at_boundary_accepted` | test_race_objective_adapter.py | COM height at fit range boundary accepted |
| `test_negative_wheel_moi_rejected` | test_race_objective_adapter.py | Negative wheel_moi raises ValueError |
| `test_build_settings_n_basis_zero_rejected` | test_race_objective_adapter.py | n_basis=0 raises ValueError |
| `test_build_settings_n_steps_zero_rejected` | test_race_objective_adapter.py | n_steps=0 raises ValueError |
| `test_validate_thrust_csv_rejects_negative_time` | test_race_objective_adapter.py | Negative time in thrust CSV rejected |
| `test_validate_thrust_csv_rejects_negative_force` | test_race_objective_adapter.py | Negative force in thrust CSV rejected |
| `test_validate_thrust_csv_rejects_non_positive_mass` | test_race_objective_adapter.py | Zero/negative mass in thrust CSV rejected |
| `test_validate_thrust_csv_accepts_valid_csv` | test_race_objective_adapter.py | Valid thrust CSV passes all checks |
| `test_dT_dL_is_nonzero` | test_race_objective_adapter.py | Lift gradient is non-zero (lift-dependent friction works) |
| `test_dT_dx_com_is_nonzero` | test_race_objective_adapter.py | x_com gradient is non-zero (fore-aft COM sensitivity works) |
| `test_dotdot_substring_in_candidate_id_accepted` | test_candidate_record.py | `cand..001` is accepted (not falsely rejected) |
| `test_dot_candidate_id_rejected` | test_candidate_record.py | `.` as candidate_id is rejected |
| `test_dotdot_candidate_id_rejected` | test_candidate_record.py | `..` as candidate_id is rejected |
| `test_invalid_lifecycle_state_rejected_at_construction` | test_candidate_record.py | Invalid lifecycle_state raises at construction |
| `test_mesh_independence_without_runner_raises_typeerror` | test_mesh_validation.py | Missing cfd_runner raises TypeError |
| `test_mixed_sign_relative_spread_is_safe` (updated) | test_mesh_validation.py | Mixed-sign values report nonzero spread |
| `test_all_near_zero_relative_spread_is_safe` | test_mesh_validation.py | Genuinely near-zero values report 0 spread |
| `test_absurd_com_position_rejected` | test_mass_com_ingest.py | COM at 999m raises ValueError |
| `test_csv_error_includes_row_and_column_context` | test_calibration.py | Malformed cell error includes row/column info |

### Updated tests for 8-element param vectors

All tests that construct parameter vectors were updated from 6-element to 8-element arrays (adding `lift_20_n` and `com_x_m`):
- `test_race_objective_adapter.py` — `_model_and_params` helper, all validation tests
- `test_adjoint_contract.py` — `_params` function, `test_adjoint_objective_is_w_D20_times_D20`
- `test_integration_end_to_end.py` — params array includes `full.L` and `mass_report.com_x_m`

---

## Files Modified

| File | Changes |
|------|---------|
| `race_objective.py` | Lift-dependent friction, fore-aft COM penalty, PARAM_NAMES expanded to 8, make_param_vector updated, finite_difference_check updated |
| `race_objective_adapter.py` | COM height guard, thrust CSV validator, BuildSettings guard, gradient mapping for dT_dL and dT_dx_com, COM sanity bounds |
| `mass_com_ingest.py` | COM sanity range check, fixed misleading dead constants/comments |
| `mesh_validation.py` | Required keyword-only cfd_runner, fixed `_relative_spread` formula |
| `candidate_record.py` | Path traversal false-positive fix, lifecycle validation at construction |
| `adjoint_contract.py` | Half-car scaling factor (0.5), updated docstring |
| `calibration.py` | CSV error messages with row/column context |
| `run_all_tests.py` | Complete rewrite — reliable path resolution, exit codes, stderr surfacing |
| `.gitignore` | Added generated artifact patterns |

### Test files modified

| File | Changes |
|------|---------|
| `tests/test_race_objective_adapter.py` | EXPECTED_HASH updated, 8-element param vectors, 11 new tests |
| `tests/test_adjoint_contract.py` | 8-element param vectors, half-car scaling test |
| `tests/test_mesh_validation.py` | Updated mixed-sign test, added all-near-zero test, added no-runner TypeError test |
| `tests/test_candidate_record.py` | Added dotdot substring, dot/dotdot, lifecycle-at-construction tests |
| `tests/test_mass_com_ingest.py` | Added absurd COM position test |
| `tests/test_calibration.py` | Added CSV error context test |
| `tests/test_integration_end_to_end.py` | Updated params array for 8 elements |

---

## Locked File Hash

**Old hash:** `575636fc3d97c96fe9294a417713227cd2ae992f67123643bc15ed5a5064a30f`  
**New hash:** `6ed47bb624245e85d67a3fb6dd196b4b69fe2debc39725fac8c9614aec404358`

The file was unlocked with user permission to add lift-dependent friction and fore-aft COM sensitivity, then re-locked with the new hash. All tests verify against the new hash.

---

## Test Results

```
test_physics_contract:          15 passed, 0 failed
test_mass_com_ingest:            6 passed, 0 failed
test_cfd_wrapper:                7 passed, 0 failed
test_mesh_validation:            8 passed, 0 failed
test_calibration:               15 passed, 0 failed
test_candidate_record:           9 passed, 0 failed
test_race_objective_adapter:    12 passed, 0 failed
test_adjoint_contract:           9 passed, 0 failed
test_integration_end_to_end:     1 passed, 0 failed

TOTAL:                          82 passed, 0 failed
```

---

## Remaining Open Items (not code fixes)

1. **`to_full_car()` ref_length** — Currently `sqrt(A_full)` placeholder. User confirmed it should be wheelbase W from `physics_contract.py`. Not yet changed.
2. **Real ballast data** — COM penalty polynomial uses placeholder experiment data. Real data coming soon.
3. **OpenFOAM pipeline** — Still `NotImplementedError`. Deferred to Part 3 build order.
4. **k-omega SST runner** — Still `NotImplementedError`. Deferred.
5. **Evolutionary search** — Part 3, not yet implemented.