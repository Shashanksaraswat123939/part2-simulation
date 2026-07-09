# Part 2 Critical Audit Findings

## CRITICAL (can cause crashes, wrong results, or security issues)

1. **PATH TRAVERSAL in candidate_record.py** — `candidate_id='../../evil'` escapes output directory
2. **NaN/Infinity accepted in JSON** — Python json.dump allows NaN/Inf by default, producing non-standard JSON that breaks strict parsers
3. **ZeroDivisionError in mesh_validation._relative_spread** — all-zero or symmetric values cause division by zero
4. **API MISMATCH: mesh_validation calls cfd_runner(stl_path, resolution_string)** — `run_half_car_cfd` expects `reference_speed_mps: float` as second arg, string 'coarse' would crash at runtime

## HIGH (wrong physics or silent failures)

5. **COM penalty polynomial goes slightly negative** near delta≈0.61mm — T_penalized < T_raw possible in edge case
6. **to_full_car() ignores custom q_ref** — q_full is recomputed from module constants, silently discarding any override
7. **to_full_car() produces absurd Cm for tiny areas** — A=1e-20 gives Cm=1.4e26, no guard
8. **No validation for negative forces/area** in HalfCarQuantities/FullCarQuantities — negative D20, L, A silently accepted
9. **T_raw formula is only correct for tc=1.0** — `T_raw = T_penalized - com_penalty` should be `T_raw = T_penalized - tc * com_penalty`. Safe because adapter guards tc=1.0, but formula is mathematically wrong and will break silently if guard is ever removed.

## MEDIUM (robustness/edge cases)

10. **No size limit on setup_logs** — 10MB string produces 10MB JSON file, no limit enforced
11. **polyfit with duplicate x_com values** — rank-deficient fit succeeds silently with RankWarning
12. **compute_held_out_residual with 2-row dataset** — LinAlgError (SVD fails), no graceful handling
13. **Single-point RBF** — fit_thrust_surrogate crashes with cryptic scipy error for 1 data point
14. **Binary STL files** — _read_ascii_stl_triangles crashes with UnicodeDecodeError instead of clear error message

## LOW (design/defensive)

15. **No input validation for negative mass in race_objective** — smooth_positive masks it, produces meaningless results
16. **No input validation for negative mu** — negative friction makes car faster (anti-physics)
17. **time_coefficient=0 gives T=0** — trivially zeros everything, no guard in locked file (adapter guards it)