# Part 2 Audit Fix Build Report

**Date:** 2026-07-09  
**Implemented by:** GLM 5.2 (via OpenClaw)  
**Source audit docs:** `PART2_AUDIT_FULL---75639623-4754-4727-9a05-2ff1ea1fe5ee.md`, `PART2_AUDIT_CATEGORIZED---e05fb97e-2cf6-4c1f-8847-d4885de1a835.md`  
**Locked file constraint:** `race_objective.py` was NOT modified. SHA-256 hash confirmed unchanged: `575636fc3d97c96fe9294a417713227cd2ae992f67123643bc15ed5a5064a30f`

---

## Summary

All 13 active findings from the two audit documents were addressed. The test suite was expanded with new regression tests for every fix. Final result: **96 tests passed, 0 failed** (up from 87 tests before the new ones were added).

---

## Changes Made

### 1. `run_all_tests.py` — [D1, HIGH] Fixed broken test runner

**Problem:** Hardcoded `tests/` subfolder prefix that didn't exist relative to the script's working directory, silently reported `TOTAL: 0 passed, 0 failed` when every test file failed to be found, ignored subprocess return codes, and never surfaced stderr.

**Fix:** Rewrote to use `Path(__file__).resolve().parent / "tests"` for reliable file resolution, check file existence before running, surface stderr and exit codes, and exit non-zero (code 2) when infrastructure failures occur so CI can't mistake "broken harness" for "clean pass."

### 2. `race_objective_adapter.py` — [A.1/3.1, CRITICAL] COM height extrapolation guard

**Problem:** No validation on `com_height_m` before it reaches the locked polynomial, which was only fitted over `com_height_m ∈ [0.018, 0.042]`. Out-of-range values produce penalties on the order of 10^14–10^15 seconds.

**Fix:** Added `COM_HEIGHT_FIT_RANGE_M` constant derived from the locked file's `_COM_PENALTY_DATA_MM` (not hardcoded, so it can't drift). Added range check in `_assert_physical_inputs()` with a small floating-point tolerance for boundary values. Raises `ValueError` with a clear message pointing to upstream mass/COM ingestion as the likely cause.

### 3. `race_objective_adapter.py` — [A.2, HIGH] Thrust CSV physical sanity validation

**Problem:** The locked `_clean_csv_arrays` validates formatting but not physical sanity — negative time, negative force, or non-positive mass all pass through silently.

**Fix:** Added `validate_thrust_csv_physical_sanity(csv_path)` function that calls the locked `_clean_csv_arrays` and then checks for negative time, negative force, and non-positive mass. Mirrors the "guard the locked file from outside" pattern used for `time_coefficient` and negative mass/mu.

### 4. `race_objective_adapter.py` — [A5, LOW] BuildSettings validation wrapper

**Problem:** `BuildSettings(n_basis=0)` produces an empty RBF basis without a clear error. Since `BuildSettings` is in the locked file, the guard must live outside.

**Fix:** Added `build_smooth_sheet_model_guarded(csv_path, settings)` wrapper that validates `n_basis >= 1` and `n_steps >= 1` before calling the locked `build_smooth_sheet_model`.

### 5. `mass_com_ingest.py` — [A.1/3.1, CRITICAL] COM sanity range check

**Problem:** `ingest_mass_com` computes COM coordinates as a plain mass-weighted average with no range/sanity check beyond `mass_kg > 0`. A coordinate-origin mixup or units bug produces `com_z_m=223.099` with zero error.

**Fix:** Added `COM_SANITY_BOUNDS_M = (-10.0, 10.0)` and a post-calculation check that rejects any COM coordinate outside these bounds with a clear message suggesting a units or origin bug. Bounds are generous enough for test fixtures while catching the 223m case.

### 6. `mass_com_ingest.py` — [E.1/3.6, LOW] Fixed misleading dead placeholder constants

**Problem:** `CO2_CARTRIDGE_COM`, `REAR_WING_MASS_KG`, `WHEELS_AXLES_MASS_KG` had comments claiming they raise `NotImplementedError` — they don't, and are never referenced. Misleading for future readers.

**Fix:** Removed `CO2_CARTRIDGE_COM` (the worst offender with the `NotImplementedError` claim). Rewrote comments on the remaining two to accurately describe that `FixedHardwareSpec`'s required dataclass fields are the actual enforcement mechanism, and these names exist only as documentation anchors.

### 7. `mesh_validation.py` — [B.1/3.2, HIGH] Mesh independence false-positive fix

**Problem:** `run_mesh_independence_study()` with default `cfd_runner=run_half_car_cfd` silently passes resolution labels as `reference_speed_mps` (a float parameter), which the stub discards. All three runs produce identical results → 0% spread → false PASS.

**Fix:** Replaced the default with a `_REQUIRED_RUNNER` sentinel. `cfd_runner` is now a required keyword-only argument. Calling `run_mesh_independence_study("car.stl")` without supplying a runner raises `TypeError` with a clear explanation. This converts a silent correctness bug into an immediate, obvious error at the call site.

### 8. `mesh_validation.py` — [B.3/3.4, MEDIUM] `_relative_spread` zero-mean masking

**Problem:** The zero-mean guard (`if mean == 0: return 0.0`) is reachable for real lift-force data where values oscillate around zero (e.g. `-0.3, 0.4, -0.05` N). These average near zero, triggering the guard and masking genuine mesh non-convergence as a false PASS.

**Fix:** Replaced the mean-based scale with `max(abs(v) for v in values)`, which only returns 0.0 when all values are genuinely near zero (abs < 1e-9). Mixed-sign values with real disagreement now correctly report non-zero spread.

### 9. `adjoint_contract.py` — [B.2/3.3, HIGH] Half/full D20 scaling ambiguity flag

**Problem:** `compute_adjoint_objective` uses full-car D20 with no scaling, but the OpenFOAM adjoint will likely run on a half-car mesh. No scaling factor or documentation exists. This is a spec ambiguity, not a definite code bug.

**Fix:** Added a `⚠ UNRESOLVED` comment block to `compute_adjoint_objective`'s docstring documenting the ambiguity and stating it must be resolved by a human before Stage 6 of the Part 3 build order. Follows the established pattern from `race_objective_adapter.py`'s existing `time_coefficient` conflict comment.

### 10. `candidate_record.py` — [C.1/3.5, MEDIUM] Path traversal false-positive fix

**Problem:** The guard `".." in safe_id` rejects any candidate ID containing the substring `..` anywhere, including legitimate IDs like `cand..001`. Not a security bug, but silently rejects valid candidates.

**Fix:** Split into two checks: (1) reject path separators (already existed), (2) reject exact `.` and `..` only. Since separators are already banned, a bare ID with no separators can never be a multi-segment path traversal — `..` as a substring inside a single-segment name is harmless.

### 11. `candidate_record.py` — [C.2/A3, MEDIUM] Lifecycle state validation at construction

**Problem:** `write_candidate_record()` does not validate `lifecycle_state`, only `read_candidate_record()` does. The codebase's own write path can create records with invalid states that are only caught later on read.

**Fix:** Added lifecycle_state validation in `CandidateRecord.__post_init__` alongside the existing `setup_logs` length check. `read_candidate_record`'s existing check remains as defense-in-depth against externally-edited JSON.

### 12. `calibration.py` — [C.3/A4, LOW] CSV error messages with row/column context

**Problem:** `_read_csv_columns` raises `ValueError: could not convert string to float: ''` with no indication of which row or column triggered it. Affects all calibration functions that route through this helper.

**Fix:** Replaced the list comprehension with an indexed loop that catches `(ValueError, TypeError)` and re-raises with file path, column name, and 1-indexed row number (plus header).

### 13. `.gitignore` — [D.2/A7, LOW] Added generated artifact patterns

**Problem:** No ignore rules for candidate database JSONs, STL files, NumPy snapshots, or OpenFOAM case directories that the pipeline generates.

**Fix:** Added patterns: `candidates/`, `*.stl`, `*.npy`, `cfd_case_*/`, `*.foam`.

---

## New Regression Tests Added

| Test | File | What it verifies |
|------|------|-----------------|
| `test_com_height_out_of_range_rejected` | test_race_objective_adapter.py | COM height 223.099m raises ValueError |
| `test_com_height_at_boundary_accepted` | test_race_objective_adapter.py | COM height at fit range boundary (0.042m) accepted |
| `test_negative_wheel_moi_rejected` | test_race_objective_adapter.py | Negative wheel_moi_kg_m2 raises ValueError |
| `test_build_settings_n_basis_zero_rejected` | test_race_objective_adapter.py | n_basis=0 raises ValueError via guarded wrapper |
| `test_build_settings_n_steps_zero_rejected` | test_race_objective_adapter.py | n_steps=0 raises ValueError via guarded wrapper |
| `test_validate_thrust_csv_rejects_negative_time` | test_race_objective_adapter.py | Negative time in thrust CSV rejected |
| `test_validate_thrust_csv_rejects_negative_force` | test_race_objective_adapter.py | Negative force in thrust CSV rejected |
| `test_validate_thrust_csv_rejects_non_positive_mass` | test_race_objective_adapter.py | Zero/negative mass in thrust CSV rejected |
| `test_validate_thrust_csv_accepts_valid_csv` | test_race_objective_adapter.py | Valid thrust CSV passes all checks |
| `test_dotdot_substring_in_candidate_id_accepted` | test_candidate_record.py | `cand..001` is accepted (not falsely rejected) |
| `test_dot_candidate_id_rejected` | test_candidate_record.py | `.` as candidate_id is rejected |
| `test_dotdot_candidate_id_rejected` | test_candidate_record.py | `..` as candidate_id is rejected |
| `test_invalid_lifecycle_state_rejected_at_construction` | test_candidate_record.py | Invalid lifecycle_state raises at construction |
| `test_mesh_independence_without_runner_raises_typeerror` | test_mesh_validation.py | Missing cfd_runner raises TypeError |
| `test_mixed_sign_relative_spread_is_safe` (updated) | test_mesh_validation.py | Mixed-sign values report nonzero spread |
| `test_all_near_zero_relative_spread_is_safe` | test_mesh_validation.py | Genuinely near-zero values report 0 spread |
| `test_absurd_com_position_rejected` | test_mass_com_ingest.py | COM at 999m raises ValueError |
| `test_csv_error_includes_row_and_column_context` | test_calibration.py | Malformed cell error includes row/column info |

---

## Test Results

```
test_physics_contract: 15 passed, 0 failed
test_mass_com_ingest: 7 passed, 0 failed
test_cfd_wrapper: 7 passed, 0 failed
test_mesh_validation: 9 passed, 0 failed
test_calibration: 16 passed, 0 failed
test_candidate_record: 13 passed, 0 failed
test_race_objective_adapter: 19 passed, 0 failed
test_adjoint_contract: 9 passed, 0 failed
test_integration_end_to_end: 1 passed, 0 failed

TOTAL: 96 passed, 0 failed
```

Exit code: 0

---

## Locked File Integrity

`race_objective.py` was not modified. SHA-256 hash confirmed:
```
575636fc3d97c96fe9294a417713227cd2ae992f67123643bc15ed5a5064a30f
```
This matches `EXPECTED_HASH` in `test_race_objective_adapter.py::test_hash_of_locked_file_matches`, which continues to pass.

---

## Files Modified

| File | Changes |
|------|---------|
| `run_all_tests.py` | Complete rewrite — reliable path resolution, exit codes, stderr surfacing |
| `race_objective_adapter.py` | COM height guard, thrust CSV validator, BuildSettings guard, COM sanity bounds |
| `mass_com_ingest.py` | COM sanity range check, fixed misleading dead constants/comments |
| `mesh_validation.py` | Required keyword-only cfd_runner, fixed `_relative_spread` formula |
| `candidate_record.py` | Path traversal false-positive fix, lifecycle validation at construction |
| `adjoint_contract.py` | UNRESOLVED comment for half/full D20 scaling ambiguity |
| `calibration.py` | CSV error messages with row/column context |
| `.gitignore` | Added generated artifact patterns |

## Test Files Modified

| File | Changes |
|------|---------|
| `tests/test_mesh_validation.py` | Updated mixed-sign test, added all-near-zero test, added no-runner TypeError test |
| `tests/test_candidate_record.py` | Added dotdot substring accepted, dot/dotdot rejected, lifecycle-at-construction tests |
| `tests/test_race_objective_adapter.py` | Added 8 new tests for COM height, BuildSettings, thrust CSV validation |
| `tests/test_mass_com_ingest.py` | Added absurd COM position rejected test |
| `tests/test_calibration.py` | Added CSV error context test |

---

## Findings Not Requiring Code Changes

- **Original findings #1–#17** (from `audit_notes.md`): 14 confirmed fixed, 1 was still open (now fixed as #7 above), 2 had residual gaps (now fixed as #2 and #8 above). All 17 are now fully resolved.
- **Spec compliance** (Section 2 of the audit): All confirmed compliant — Half-Car CFD Contract, Rolling Friction Policy, CO2 cartridge mass, CandidateRecord fields, lifecycle states.
- **Things explicitly checked and found correct**: Coordinate conventions, unit conventions, SHA-256 hash safeguard, `N_WHEELS`/`R_WHEEL` hardcoded constants.

---

## Remaining Human Decisions

1. **Adjoint half/full scaling** (B.2/3.3): The `⚠ UNRESOLVED` comment documents the ambiguity. A human must decide whether the OpenFOAM adjoint runs on half-car or full-car domain before Stage 6 of the Part 3 build order. No code change needed until then.
2. **CO2 cartridge fixed legal position** (A.4, spec gap): `FixedHardwareSpec` enforces the 23g mass but `co2_cartridge_com` is caller-supplied with no validation against a known legal coordinate. The spec says "fixed legal position" but doesn't give the coordinate. This requires a spec clarification, not a code fix.