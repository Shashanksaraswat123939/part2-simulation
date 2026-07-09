# Part 2 (Simulation Setups) — Full Independent Audit

**Audited by:** Claude (Sonnet 5), for handoff to GLM 5.2 for fixes
**Coverage:** All 24 files in the project — every `.py` source file, every
test file, both markdown audit docs, `_gitignore`, `run_all_tests.py`, and
all three spec docs. See the coverage table at the end of this document.
**Audited against:** `01_generative_geometry(1).md`, `02_simulation_setups(1).md`,
`03_optimizer_workflow(1).md` (the three governing spec docs), plus the code as it
exists today in the project.
**Method:** Every finding below was verified either by (a) directly reading the
exact line of source referenced, or (b) writing and running a standalone
reproduction script against the real code (not a paraphrase, not a mock of the
logic — the actual imported modules). Findings from the prior `audit_notes.md`
and `BUILD_REPORT.md` were **not trusted on their word** — each was re-verified
independently, and several turned out to be mischaracterized (see Section 1).
Repro scripts are listed inline where relevant so GLM 5.2 (or anyone) can rerun
them.

No Part 1 (Generative Geometry) or Part 3 (Optimizer/Orchestration) code exists
in this project — only their spec docs. This audit therefore covers **Part 2
implementation only**. Any statement about Part 1/3 below is a spec-reading
observation, not a code finding, and is labeled as such.

---

## 0. TL;DR for GLM 5.2

Priority order to fix, most important first:

0. **[HIGH]** `run_all_tests.py` is completely broken — hardcodes a
   `tests/` subfolder that doesn't exist, every test file fails to run,
   and it silently reports `TOTAL: 0 passed, 0 failed` with no visible
   error. Confirmed by running it. Fixed version in Section 6 (Addendum).
1. **[NEW-CRITICAL]** `com_height_time_penalty` (locked file) has no
   extrapolation guard, and nothing upstream validates `h_com`/`x_com` before
   it reaches that function. A bad mass/COM ingest silently produces
   penalties in the range of **10^14–10^15 seconds**. (Section 3.1)
2. **[NEW-HIGH]** `run_mesh_independence_study()` called with its own defaults
   (no injected `cfd_runner` wrapper) silently reports **false-positive mesh
   independence** — it calls the same solver at the same conditions three
   times because the resolution labels are discarded by the default runner,
   and reports 0% spread / PASS. Untested path. (Section 3.2)
3. **[NEW-HIGH]** `compute_adjoint_objective` mixes full-car `D20` with a
   half-car OpenFOAM adjoint surface without any documented or applied
   scaling factor — likely a 2x error once real OpenFOAM is wired in. This is
   partly a **spec ambiguity** that the implementation resolved silently
   instead of flagging. (Section 3.3)
4. **[NEW-MEDIUM]** `_relative_spread()`'s zero-mean fix (from the prior
   audit) is reachable in realistic conditions for `L` (lift), not just the
   synthetic all-zero case it was tested against, and produces a false PASS
   there too. (Section 3.4)
5. **[NEW-MEDIUM]** Path traversal guard in `candidate_record.py` has a false
   positive: it rejects any `candidate_id` containing the substring `..`
   anywhere, including legitimate IDs like `cand..001`. Not a security bug,
   but will silently reject valid candidates. (Section 3.5)
6. **[NEW-LOW]** Dead placeholder constants in `mass_com_ingest.py`
   (`CO2_CARTRIDGE_COM`, `REAR_WING_MASS_KG`, `WHEELS_AXLES_MASS_KG`) have a
   comment claiming they raise `NotImplementedError` — they don't, and are
   never referenced anywhere in the file. Misleading comment, no code
   behavior attached to it. (Section 3.6)
7. Full re-verification of the prior 17 `audit_notes.md` findings: **14
   confirmed genuinely fixed, 1 confirmed genuinely still open (#4, but for a
   different reason than originally stated), 2 reclassified** (see Section 1
   for the two reclassifications — #3 and #9 fixes are real but each has a
   residual gap the original write-up didn't mention).

---

## 0.1 Categorized Problem Guide

This section reorganizes every active problem by the part of the system it
affects. Each entry has:

- **Plain-English explanation:** what the problem means without specialist
  terminology.
- **Technical explanation:** the concise engineering description.
- **Full evidence:** where the original detailed reproduction, source analysis,
  and proposed fix remain in this audit.

The detailed technical sections below have not been removed or simplified.
This guide is an additional navigation and explanation layer.

### Category A — Physical-input and model-validity problems

These problems allow physically impossible or unsupported values to enter the
simulation. The software may still return a number, but that number may have no
engineering meaning.

#### A.1 [CRITICAL] COM values can produce absurd race-time penalties

**Plain-English explanation:** The program accepts the car's balance-point
height without checking whether it is realistic. If that value is wrong—because
of a unit mistake, bad geometry, or corrupt input—the program does not stop.
Instead, it can predict a race penalty lasting millions or trillions of years.
Because the calculation still returns a number, another program could mistake
the result for a legitimate simulation.

**Technical explanation:** `h_com` and `x_com` are not range-validated before
`com_height_time_penalty()` evaluates the locked fourth-degree polynomial
outside its measured domain. Polynomial extrapolation produces penalties on the
order of `10^14–10^15` seconds. The existing non-negative clamp only prevents a
small negative in-domain dip; it does not constrain out-of-domain growth.

**Required direction:** Validate COM coordinates at the ingestion or adapter
boundary using engineering limits derived from the legal car envelope. Reject
invalid values before calling the locked objective.

**Full evidence:** Section 3.1.

#### A.2 [HIGH] Impossible thrust CSV data is accepted

**Plain-English explanation:** A test-data file can say that time is negative,
the CO2 system has negative mass, or the engine produces negative thrust. The
program accepts those impossible measurements and calculates a race time anyway
instead of identifying the data as broken.

**Technical explanation:** `_clean_csv_arrays()` validates formatting,
duplicates, row count, and time ordering, but does not reject negative time,
negative force, or non-positive mass. The smooth mass floor can convert corrupt
data into a finite but meaningless objective value.

**Required direction:** Add physical-sanity validation outside the locked
`race_objective.py` file and require all production callers to use it before
`build_smooth_sheet_model()`.

**Full evidence:** Section 6, finding A2.

#### A.3 [LOW] Invalid model settings fail too late

**Plain-English explanation:** A setting can request zero mathematical basis
functions. That configuration makes no sense, but the program does not explain
the mistake immediately. It fails later in a more confusing part of the
calculation.

**Technical explanation:** `BuildSettings(n_basis=0)` creates an empty RBF basis
without domain-specific validation. Invalid `n_basis` or `n_steps` values
should be rejected when the settings object is constructed.

**Required direction:** Add `n_basis >= 1` and `n_steps >= 1` guards. Because
`race_objective.py` is locked, a human must decide whether to unlock it or place
equivalent checks in the external construction boundary.

**Full evidence:** Section 6, finding A5.

#### A.4 [LOW / SPEC GAP] The CO2 cartridge position is not enforced

**Plain-English explanation:** The rules require the CO2 cartridge to have a
fixed legal position, but the program lets the caller provide any position.
The mass is checked; the location is not.

**Technical explanation:** `FixedHardwareSpec` enforces the cartridge mass as
`0.023 kg`, while `co2_cartridge_com` remains caller-supplied without validation
against a known legal coordinate. The nearby placeholder constants and comments
also imply enforcement that does not actually exist.

**Required direction:** Define the approved cartridge coordinate from the
governing specification, validate it explicitly, and correct or remove the dead
placeholder constants and misleading comments.

**Full evidence:** Sections 2.4 and 3.6.

### Category B — CFD and aerodynamic-validation problems

These problems can make an aerodynamic check report success even though the
required comparison was never performed correctly.

#### B.1 [HIGH] The default mesh-independence study can report a false PASS

**Plain-English explanation:** The program is supposed to run three different
mesh qualities and check whether they agree. In its default mode, it can run the
same setup three times, get the same answer three times, and announce that the
mesh is independent. It is effectively comparing a result with itself.

**Technical explanation:** `run_mesh_independence_study()` passes resolution
labels into a callable whose second parameter is actually
`reference_speed_mps`. The current default CFD stub discards that parameter, so
no coarse/medium/fine refinement occurs. Identical runs yield zero spread and a
false-positive `passes_5_percent_target=True`. Once the real solver consumes the
parameter as a float, the same mismatch may become a runtime crash.

**Required direction:** Introduce an explicit mesh-resolution configuration
interface. Refuse to report mesh independence unless three verifiably different
meshes were generated and executed.

**Full evidence:** Original finding #4 and Section 3.2.

#### B.2 [HIGH / HUMAN DECISION] Adjoint drag may be scaled by the wrong factor

**Plain-English explanation:** The main race model uses drag for the whole car,
but the future adjoint CFD calculation appears to use only half of the symmetric
car. If one side is not doubled correctly, the optimizer may think a design
change is only half—or possibly twice—as important as it really is.

**Technical explanation:** `compute_adjoint_objective()` combines full-car
`D20` with an implied half-car OpenFOAM adjoint sensitivity surface. No explicit
factor-of-two conversion or documented normalization contract exists. The
correct scaling depends on how the adjoint solver defines the surface and
objective.

**Required direction:** Mark the behavior as unresolved, document whether the
adjoint surface represents half-car or full-car sensitivity, and apply the
confirmed scaling exactly once.

**Full evidence:** Sections 2.2 and 3.3.

#### B.3 [MEDIUM] Opposite lift values can cancel and falsely look consistent

**Plain-English explanation:** Three mesh runs can strongly disagree—for
example, one predicts upward lift and another predicts downforce—but their
average can be close to zero. The current formula then reports zero disagreement
and passes the study.

**Technical explanation:** `_relative_spread()` returns `0.0` whenever the
arithmetic mean is zero. Mixed-sign, non-degenerate lift results can therefore
mask real spread through cancellation. Negative lift is physically valid, so
this is reachable in realistic data.

**Required direction:** Normalize spread using a non-cancelling scale such as
the maximum absolute magnitude, with a separate tolerance only for genuinely
near-zero values.

**Full evidence:** Original finding #3 and Section 3.4.

#### B.4 [LOW / TEST UPDATE] A test currently enforces the false-PASS behavior

**Plain-English explanation:** One automated test says the incorrect
zero-disagreement result is the expected answer. Fixing the real calculation
without fixing the test will make the test fail—even though the new calculation
is better.

**Technical explanation:**
`test_mixed_sign_relative_spread_is_safe()` explicitly asserts that
`_relative_spread((-1.0, 1.0, 0.0)) == 0.0`. That assertion encodes the defect
described in B.3.

**Required direction:** Update the source and test together. Mixed-sign values
with meaningful magnitude must produce non-zero spread; only genuinely
negligible values should return zero.

**Full evidence:** Section 6, finding A6.

### Category C — Candidate-record and data-quality problems

These problems affect how simulation candidates and calibration data are named,
validated, stored, and diagnosed.

#### C.1 [MEDIUM] Legitimate candidate names containing `..` are rejected

**Plain-English explanation:** The security check correctly blocks folder
escape attacks, but it also blocks harmless names that happen to contain two
dots, such as `cand..001`.

**Technical explanation:** The path-traversal guard searches for the substring
`..` rather than validating path components and the final resolved path. This
causes false positives without adding meaningful protection.

**Required direction:** Reject separators, absolute paths, and actual parent
components; then verify that the resolved output remains inside the intended
directory.

**Full evidence:** Section 3.5.

#### C.2 [MEDIUM] Invalid lifecycle states can be written to the database

**Plain-English explanation:** The program can save a candidate with a made-up
status such as `"totally_bogus_state"`. It notices the problem only later when
somebody tries to read the file.

**Technical explanation:** `read_candidate_record()` validates
`lifecycle_state`, but `CandidateRecord` construction and
`write_candidate_record()` do not. The codebase can therefore create its own
invalid JSON record.

**Required direction:** Validate lifecycle state in
`CandidateRecord.__post_init__` and keep read-time validation as defense against
externally edited files.

**Full evidence:** Section 6, finding A3.

#### C.3 [LOW] Broken CSV cells produce vague error messages

**Plain-English explanation:** If one spreadsheet cell is blank or contains
text, the program says only that it could not convert a value. It does not tell
the user which row or column needs fixing.

**Technical explanation:** `_read_csv_columns()` converts values through a
single list comprehension, so `ValueError` lacks the CSV path, column name, and
row number.

**Required direction:** Parse cells in an indexed loop and re-raise errors with
file, column, row, and offending-value context.

**Full evidence:** Section 6, finding A4.

### Category D — Testing and developer-infrastructure problems

These problems do not directly change the physics, but they can hide failures or
pollute the repository with generated output.

#### D.1 [HIGH] The one-command test runner can fail silently

**Plain-English explanation:** The script that is supposed to prove all tests
pass may run none of them and still print `0 passed, 0 failed`. A person could
mistake that for a clean result.

**Technical explanation:** In the audited layout, `run_all_tests.py` hardcodes a
non-existent `tests/` prefix, ignores subprocess return codes, and fails to
surface `stderr` because splitting an empty output string still produces a
non-empty list. Infrastructure failures therefore do not make the runner fail.

**Required direction:** Resolve test paths relative to `__file__`, check file
existence and subprocess exit codes, print `stderr`, and exit non-zero whenever
a test could not run.

**Full evidence:** Section 6, finding A1.

#### D.2 [LOW] Generated simulation files are not fully ignored by Git

**Plain-English explanation:** Large output files—candidate records, STL
geometry, NumPy snapshots, and OpenFOAM cases—may accidentally be committed to
the repository.

**Technical explanation:** The ignore rules cover ordinary Python artifacts but
not project-specific outputs such as `candidates/`, `*.stl`, `*.npy`,
`cfd_case_*/`, or `*.foam`.

**Required direction:** Add patterns after the optimizer's final output
directory convention is confirmed. Avoid rules broad enough to hide source
fixtures that should remain version-controlled.

**Full evidence:** Section 6, finding A7.

### Category E — Documentation and maintainability problems

#### E.1 [LOW] Placeholder constants and comments describe behavior that does not exist

**Plain-English explanation:** Some names and comments make it look as though
the program automatically checks missing hardware information. It does not.
This can mislead the next developer into trusting a safeguard that is absent.

**Technical explanation:** The placeholder constants in
`mass_com_ingest.py` are dead and the comment claiming they raise
`NotImplementedError` does not match runtime behavior. Mandatory dataclass
arguments provide some enforcement, but the comments do not accurately explain
that mechanism.

**Required direction:** Remove dead constants or convert them into explicit
documentation pointers, and rewrite comments to describe the actual dataclass
validation.

**Full evidence:** Section 3.6.

### Category summary and recommended order

| Order | Category | Active findings | Why it comes here |
|---|---|---:|---|
| 1 | Physical-input and model validity | A.1–A.4 | Bad inputs can produce believable-looking but meaningless engineering results. |
| 2 | CFD and aerodynamic validation | B.1–B.4 | False validation passes can cause the team to trust an unverified design. |
| 3 | Testing infrastructure | D.1 | A broken test runner can hide whether any fixes work. Fix it before relying on regression results. |
| 4 | Candidate records and data quality | C.1–C.3 | These issues corrupt workflow state or make failures unnecessarily difficult to diagnose. |
| 5 | Repository and documentation hygiene | D.2 and E.1 | Lower immediate risk, but inexpensive to fix and important for long-term maintainability. |

Recommended implementation sequence:

1. Repair `run_all_tests.py` so verification failures cannot be hidden.
2. Add COM and thrust-data physical validation at unlocked input boundaries.
3. Redesign the mesh-resolution interface and `_relative_spread()` calculation,
   updating the affected test at the same time.
4. Resolve and document the half-car/full-car adjoint scaling with the CFD owner.
5. Validate lifecycle states at construction and replace the candidate-ID
   substring check with proper resolved-path containment.
6. Improve CSV diagnostics, configuration validation, comments, and ignore
   rules.
7. Run the complete suite and add targeted regression tests for every corrected
   issue.

### Previously reported findings that are no longer active

The earlier 17-item audit is still retained in full in Section 1. Its current
status is:

- **14 findings are confirmed fixed.**
- **1 remains open:** original finding #4, now more accurately described by B.1.
- **2 fixes exposed residual problems:** original finding #3 maps to B.3, and
  the COM-penalty work maps to A.1.
- **Original finding #17 is a confirmed non-issue.**

Do not reopen the fixed findings unless new evidence contradicts the
re-verification results. The active worklist is the categorized set above.

---

## 1. Re-verification of the 17 `audit_notes.md` Findings

Each finding was independently re-tested. "Original claim" = what
`audit_notes.md` said. "Verified status" = what I confirmed by direct
execution or source read against the current code.

### CRITICAL

**#1 — Path traversal in `candidate_record.py`**
Original claim: `candidate_id='../../evil'` escapes output directory.
Verified status: **FIXED**, confirmed by execution
(`write_candidate_record` raises `ValueError` for `../../evil`,
`..\\..\\evil`, `a/../b`, `....`, `..hidden`). See Section 3.5 for a
**new** false-positive side effect of this same guard, not in the original
finding.

**#2 — NaN/Infinity accepted in JSON**
Original claim: `json.dump` allows `NaN`/`Inf` by default.
Verified status: **FIXED**, confirmed by execution.
`json.dump(..., allow_nan=False)` in `candidate_record.py` line 147 raises
`ValueError: Out of range float values are not JSON compliant` for both NaN
and Infinity. Tests `test_nan_float_rejected_in_json` and
`test_inf_float_rejected_in_json` correctly exercise this.

**#3 — `ZeroDivisionError` in `mesh_validation._relative_spread`**
Original claim: all-zero or symmetric values cause division by zero.
Verified status: **FIXED, but the fix has a masked side effect not
mentioned in the original finding or `BUILD_REPORT.md`.** The guard
(`if mean == 0: return 0.0`) does stop the crash. But it also means any
zero-mean spread — including a **genuinely divergent** mesh study where
values happen to average near zero (e.g. lift force oscillating
`-2.0, +2.0, +0.1` N across coarse/medium/fine mesh) — silently reports
`spread=0.0`, i.e. a false PASS on `passes_5_percent_target`. See Section
3.4 for full reproduction and why this is realistically reachable (lift is
NOT guarded against negative values in `HalfCarQuantities`, unlike D20/A).

**#4 — API mismatch: `mesh_validation` calls
`cfd_runner(stl_path, resolution_string)`**
Original claim: `run_half_car_cfd` expects `reference_speed_mps: float` as
second arg; a string would crash at runtime.
Verified status: **STILL OPEN, but the original description of the failure
mode is wrong.** It does **not** crash. `run_half_car_cfd`'s current body
literally does `del reference_speed_mps, air_density_kgm3, max_iterations`
immediately on entry (line 112) — the parameter is completely unused because
`_invoke_openfoam_pipeline` is a `NotImplementedError` stub that doesn't
consume it. So passing `"coarse"` as `reference_speed_mps` today is a no-op,
not a crash.
**The real, currently-live bug**: `run_mesh_independence_study()` called
with all defaults calls the exact same solver at the exact same (discarded)
condition three times, gets identical results back, and reports
`passes_5_percent_target=True` with zero actual mesh refinement having
occurred. This is silent false-positive validation, worse than a crash
because it produces a wrong "PASS" that a human might trust. See Section
3.2 for full repro. **This will also become a real crash later** once
`_invoke_openfoam_pipeline` is implemented and starts actually using
`reference_speed_mps` as a float (e.g. in a snappyHexMesh dict template
string formatting call) — so the original finding's crash prediction is
correct for the future, just not for the current code.

### HIGH

**#5 — COM penalty polynomial goes negative near delta≈0.61mm**
Original claim: `T_penalized < T_raw` possible in edge case.
Verified status: **CONFIRMED the underlying defect exists, and CONFIRMED
the clamp fix neutralizes it.** Direct polynomial evaluation shows the
locked-file degree-4 fit dips negative across `delta_mm ∈ [0.005, 0.815]`,
minimum `-0.0000379s` at `delta_mm=0.415` (original finding said ≈0.61mm;
actual measured minimum is at 0.415mm — close enough to be the same
phenomenon, the "0.61" figure in the original note was imprecise but not
wrong in kind). Adapter's `com_penalty = max(com_penalty, 0.0)` clamp
verified by execution to hold `T_penalized == T_raw` (diff = 0.0) exactly at
the dip point. **Fixed as far as the adapter's own tests go** — but see
Section 3.1: the clamp only helps for the small in-domain dip; it does
nothing about the much larger extrapolation problem when `h_com` is
out-of-range entirely.

**#6 — `to_full_car()` ignores custom `q_ref`**
Verified status: **FIXED**, confirmed by execution. Passing `q_ref=999.0`
produces `Cm=0.0354` vs default `Cm=0.1443` — confirms the override is
actually used, not silently discarded.

**#7 — `to_full_car()` produces absurd Cm for tiny areas**
Verified status: **FIXED**, confirmed by execution. `A=1e-20` now produces
`Cm=0.0` cleanly (guard is `A_full > 1e-12`), not the `1.4e26` the original
finding described.

**#8 — No validation for negative forces/area**
Verified status: **FIXED for D20 and A**, confirmed by execution
(`ValueError` raised for negative D20, negative A). **Note (not a bug):**
`L` (lift) is intentionally NOT guarded against negative values, because
negative lift = downforce is physically valid for a car body. This is
correct, not a gap — flagging only so it isn't mistaken for an incomplete
fix. It does, however, matter for Section 3.4 below.

**#9 — `T_raw` formula only correct for `tc=1.0`**
Original claim: formula should be `T_raw = T_penalized - tc * com_penalty`,
currently `T_raw = T_penalized - com_penalty`, "safe because guard exists."
Verified status: **FIXED**, confirmed by source read.
`race_objective_adapter.py` line ~112 now reads:
```python
tc = float(param_vector[PARAM_NAMES.index("time_coefficient")])
T_raw = T_penalized - tc * com_penalty
```
This is the corrected general formula. Confirmed the `_assert_time_coefficient_unity`
guard is still in place immediately above it, so the fix is genuinely
future-proofed as claimed, not just theoretically correct.

### MEDIUM

**#10 — No size limit on `setup_logs`**
Verified status: **FIXED**, confirmed by source read and test.
`CandidateRecord.__post_init__` rejects `setup_logs` over 1,000,000 chars
with `ValueError`. `test_setup_logs_size_limit_enforced` exercises this with
a 2,000,000-char string and confirms rejection.

**#11 — `polyfit` with duplicate x_com values**
Verified status: **FIXED**, confirmed by source read.
`fit_com_penalty_curve` checks `len(set(x)) < degree + 1` and raises
`ValueError` before calling `polyfit`, and additionally promotes
`RankWarning` to a hard error as defense in depth
(`warnings.simplefilter("error", RankWarning)`). Test
`test_duplicate_x_com_rejected` confirms.

**#12 — `compute_held_out_residual` with 2-row dataset**
Verified status: **FIXED**, confirmed by source read.
Explicit `if n < 4: raise ValueError(...)` guard before any SVD-based
fitting occurs. Test `test_held_out_residual_tiny_dataset_rejected`
confirms with a 2-row CSV.

**#13 — Single-point RBF crash**
Verified status: **FIXED**, confirmed by source read.
`fit_thrust_surrogate` checks `if len(t) < 2: raise ValueError(...)` before
calling `RBFInterpolator`. Test `test_single_point_thrust_surrogate_rejected`
confirms.

**#14 — Binary STL files crash with `UnicodeDecodeError`**
Verified status: **FIXED**, confirmed by source read and test.
`_read_ascii_stl_triangles` checks
`if not raw.lstrip().startswith(b"solid")` before attempting UTF-8 decode,
raising a clear `CFDRunError` instead. Test
`test_binary_stl_rejected_with_clear_error` constructs a real binary STL
via `struct.pack` and confirms `CFDRunError` (not `UnicodeDecodeError`).
**Caveat not in the original finding**: an ASCII STL that starts with the
literal bytes `"solid"` but is nonetheless a malformed binary file (some
binary STL exporters do write a `solid...` prefix in the 80-byte header
string despite being binary format — this is a known STL-format ambiguity)
would still pass the `startswith(b"solid")` check and then likely fail
later in `_read_ascii_stl_triangles`'s vertex-count check
(`len(vertices) % 3 != 0`) rather than crash uncontrolled — so the failure
mode stays inside `CFDRunError` either way, just via a different message.
Not asking for a fix, just noting the check is a heuristic, not a
guarantee, and that this is a known real-world STL quirk worth a code
comment.

### LOW

**#15 — No input validation for negative mass in `race_objective`**
Verified status: **FIXED at the adapter layer**, confirmed by source read.
`race_objective_adapter._assert_physical_inputs` checks
`car_weight_kg <= 0` and raises `ValueError`. Test
`test_negative_mass_rejected` confirms. Correctly scoped: the locked
`race_objective.py` itself is untouched (as required), the guard lives in
the adapter, which is the right place per the "locked file must not be
modified" constraint stated throughout `BUILD_REPORT.md`.

**#16 — No input validation for negative mu**
Verified status: **FIXED**, same mechanism as #15.
`_assert_physical_inputs` checks `mu < 0`. Test `test_negative_mu_rejected`
confirms. Also independently re-confirmed `drag_20_n < 0` and
`wheel_moi_kg_m2 < 0` are guarded in the same function (these weren't
separately numbered in `audit_notes.md` but are the same class of issue and
are covered).

**#17 — `time_coefficient=0` gives `T=0`**
Verified status: **Correctly a non-issue**, confirmed by source read.
`_assert_time_coefficient_unity` in the adapter hard-asserts
`time_coefficient == 1.0` for every call through the adapter, so `tc=0`
can never reach `race_time_seconds` through the sanctioned path. Matches
the original note's own conclusion ("adapter guards it, no additional fix
needed").

### Summary table

| # | Severity | Status |
|---|---|---|
| 1 | CRITICAL | Fixed (new false-positive side effect found, see 3.5) |
| 2 | CRITICAL | Fixed |
| 3 | CRITICAL | Fixed (masked side effect found, see 3.4) |
| 4 | CRITICAL | **Still open** (different failure mode than described) |
| 5 | HIGH | Fixed (residual gap found, see 3.1) |
| 6 | HIGH | Fixed |
| 7 | HIGH | Fixed |
| 8 | HIGH | Fixed |
| 9 | HIGH | Fixed |
| 10 | MEDIUM | Fixed |
| 11 | MEDIUM | Fixed |
| 12 | MEDIUM | Fixed |
| 13 | MEDIUM | Fixed |
| 14 | MEDIUM | Fixed (heuristic caveat noted) |
| 15 | LOW | Fixed |
| 16 | LOW | Fixed |
| 17 | LOW | Non-issue, confirmed |

---

## 2. Spec-Compliance Check (Parts 1–3 vs. current code)

This section was not covered by either prior audit doc. Findings below
compare the actual code behavior against the exact text of the three spec
docs.

**2.1 — Half-Car CFD Contract compliance:** Spec (`02_simulation_setups`)
states "The race objective receives only full-car values. It does not know
whether CFD was half or full." **Confirmed compliant** — `race_objective.py`
and the adapter only ever see `FullCarQuantities`-derived scalars
(`full.D20`, `full.L`, `mass_report.com_z_m`, etc.), never
`HalfCarQuantities` directly. Verified by tracing `test_integration_end_to_end.py`'s
`params = np.array([full.D20, ...])` construction.

**2.2 — Adjoint objective half/full mismatch:** See Section 3.3 below. This
is the one place the Half-Car Contract's "symmetric application" language is
not honored — see full writeup there.

**2.3 — Rolling Friction Policy:** Spec says current stage should be
`F_fric = mu × m_total(t) × g`, scalar lumped coefficient, and lists three
explicit prerequisites before upgrading to per-wheel. **Confirmed compliant**
— `race_objective.py`'s `distance_derivatives` uses exactly this scalar
form (`F_fric = p["mu"] * m * G`), and `adjoint_contract.py`'s
`CM_EXPANSION_PREREQUISITES` dict tracks the three prerequisite flags,
defaulting all to `False`, with `get_active_objective_weights` correctly
raising `NotImplementedError` only if a caller falsely sets all three `True`
without also supplying real `w_L`/`w_Cm` values. This is a faithful,
careful implementation of that section of spec.

**2.4 — CO2 cartridge mass hard requirement:** Spec: "CO2 cartridge: 23 g,
fixed legal position." **Confirmed compliant and enforced** —
`FixedHardwareSpec.__post_init__` raises `ValueError` if
`co2_cartridge_mass_kg` differs from `0.023` by more than `1e-9` kg.
**Gap:** the "fixed legal position" half of that requirement is NOT
enforced — `co2_cartridge_com` is caller-supplied with zero validation
against any known/expected fixed position. See Section 3.6.

**2.5 — Artifacts Per Candidate (spec list) vs. `CandidateRecord` fields:**
Spec lists: `candidate_id, wheelbase W, halo-canister distance d_halo, φ grid
snapshots, STL path, mass report, COM report, CFD force report, physics
objective report (T_raw, T_penalized, gradients), adjoint sensitivity field,
setup logs, failure reason`. **Confirmed 1:1 field match** in
`CandidateRecord` — every spec field has a corresponding dataclass field,
correctly typed. No missing or extra fields relative to spec.

**2.6 — Lifecycle states:** Spec lists exactly 8 states:
`valid_simulated, geometry_repaired, geometry_rejected, rule_rejected,
machining_rejected, CFD_failed, objective_failed, converged`.
**Confirmed exact match** — `ALLOWED_LIFECYCLE_STATES` in
`candidate_record.py` contains exactly these 8 strings, no more, no fewer.

---

## 3. New Findings (not in either prior audit doc)

### 3.1 [CRITICAL] — Unvalidated COM height/position causes catastrophic
### extrapolation through the locked polynomial

**Where:** `race_objective.py::com_height_time_penalty`, called from
`race_objective_adapter.py::race_value_and_grad_guarded`, fed by
`mass_com_ingest.py::ingest_mass_com`'s `com_z_m`/`com_x_m` output.

**The chain, each link confirmed independently:**

1. `mass_com_ingest.ingest_mass_com` computes `com_x_m`/`com_y_m`/`com_z_m`
   as a plain mass-weighted average of caller-supplied component positions.
   **No range/sanity check exists anywhere in this function** beyond
   `mass_kg > 0`. Confirmed by source read and by execution: feeding a
   `FixedHardwareSpec` with `co2_cartridge_com=(999.0, 999.0, 999.0)` (a
   physically absurd position for a ~150-200mm car) and otherwise-valid
   positive masses produces `com_x_m=223.125, com_z_m=223.099` with **zero
   error or warning**.

2. That `com_z_m` is exactly what a caller is expected to pass as
   `com_height_m` into the locked `race_objective.py` parameter vector
   (confirmed via `test_integration_end_to_end.py`'s
   `params = np.array([..., mass_report.com_z_m])` pattern).

3. `com_height_time_penalty` evaluates a degree-4 polynomial
   (`_COM_POLY_COEFFS`) that was fitted **only over data in the range
   `delta_mm ∈ [-12, +12]`** (i.e. `com_height_m ∈ [0.018, 0.042]`).
   Confirmed by reading `_COM_PENALTY_DATA_MM` in `race_objective.py`.
   **Unlike its sibling functions in `calibration.py`**
   (`ThrustSurrogate.__call__`, `COMPenaltyCurve.__call__`), which both
   have explicit `if t < t_min or t > t_max: raise ValueError(...)`
   extrapolation guards, `com_height_time_penalty` has **no such guard at
   all** — it's a bare `jnp.polyval` call with no domain check.

4. Confirmed by execution: calling `com_height_time_penalty` with
   `com_height_m=223.099` (the value from step 1) returns
   **`-1,068,276,775,643,015.8` seconds** (~1.07 quadrillion seconds).
   Smaller but still absurd extrapolations were also confirmed: `h_com=1.0m`
   gives `-382,679.7s`; `h_com=-1.0m` gives `-484,227.5s`.

5. The adapter's existing `com_penalty = max(com_penalty, 0.0)` clamp
   (added to fix original finding #5) **happens to** catch these specific
   sampled cases because they're all negative, but this is incidental — the
   clamp exists to fix a small in-domain polynomial dip (~38 microseconds
   at delta≈0.4mm), not to defend against 15-orders-of-magnitude
   extrapolation. There is no guarantee the polynomial can't produce a huge
   **positive** value for some other out-of-range input (quartics are not
   monotonic outside their fit domain), which the clamp would not catch at
   all, and which would then silently make an otherwise-fine candidate look
   catastrophically slow, or vice versa if it happened to fall in a
   negative region past the clamp with a magnitude the clamp reduces to
   exactly 0 while a real bug is masked.

**Why this matters concretely for this project:** in the real optimizer
loop (Part 3 spec), `d_halo` and wheelbase `W` change every outer iteration,
and machined-component φ-grid volumes/COMs change every inner iteration.
Any bug anywhere upstream in Part 1's geometry code (which does not exist
yet, so cannot itself be audited) that produces even a modestly wrong COM —
say, from a coordinate-origin mixup, a units bug (mm vs m — a very easy
mistake given this whole codebase deliberately converts between mm and m in
several places), or a φ-grid integration error — will not be caught here.
It will silently propagate into a wildly wrong `T_penalized`, and because
the "safety net" that exists (the `max(..., 0.0)` clamp) only helps in one
direction and for a different bug class, this failure mode would look like
a legitimate candidate ranking, not an obvious crash.

**Recommended fix (two parts):**

a) Add an extrapolation guard to `com_height_time_penalty` itself is not
   possible without modifying the locked file (which is explicitly
   forbidden per the project's own rules). Instead, add the guard in
   `race_objective_adapter.py`, mirroring the existing
   `_assert_physical_inputs` pattern:
   ```python
   # In _assert_physical_inputs, add:
   COM_HEIGHT_FIT_RANGE_M = (0.018, 0.042)  # matches race_objective.py's
                                              # _COM_PENALTY_DATA_MM domain,
                                              # +/-12mm around 30mm target
   if not (COM_HEIGHT_FIT_RANGE_M[0] <= p["com_height_m"] <= COM_HEIGHT_FIT_RANGE_M[1]):
       raise ValueError(
           f"com_height_m={p['com_height_m']} is outside the COM penalty "
           f"polynomial's fitted range {COM_HEIGHT_FIT_RANGE_M} -- "
           f"extrapolation would produce a meaningless penalty value. "
           f"Check upstream mass/COM ingestion for a units or origin bug."
       )
   ```
   This constant should ideally be imported/derived from
   `race_objective._COM_PENALTY_DATA_MM.min()/max()` rather than
   hardcoded, so it can never drift out of sync if the locked file's
   calibration data is ever legitimately updated with a real ballast
   experiment (which `BUILD_REPORT.md` notes is still a placeholder).

b) Add a basic sanity-range check in `mass_com_ingest.ingest_mass_com` for
   `com_x_m`/`com_y_m`/`com_z_m` against the car's known legal envelope
   dimensions (Part 1 spec gives `W ∈ [120,140]mm` and various bounding
   volumes — a car cannot physically have a COM at 223m). This is a
   second, independent line of defense at the point where the bad value is
   actually computed, rather than only at the point where it's consumed.

---

### 3.2 [HIGH] — `run_mesh_independence_study()` silently produces a
### false-positive "PASS" when called with its own defaults

**Where:** `mesh_validation.py::run_mesh_independence_study`

**Confirmed by execution** (full repro script below, run against the real
`cfd_wrapper.run_half_car_cfd` with only the OpenFOAM subprocess itself
mocked out — everything else is the real code path):

```python
import cfd_wrapper
from mesh_validation import run_mesh_independence_study

cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: {
    "D20_half": 5.0, "L_half": 1.0, "A_half": 0.01, "pitching_moment_half": 0.05,
    "residual_final": 1e-5, "negative_volume_cells": 0, "y_plus_min": 0.4,
    "y_plus_max": 2.0, "courant_max": 0.8,
}
# ... write a valid watertight tetrahedron STL to `path` ...

result = run_mesh_independence_study(path)   # called with ALL DEFAULTS
print(result)
```

**Actual output:**
```
MeshIndependenceResult(
    resolutions_tested=('coarse', 'medium', 'fine'),
    D20_values=(10.0, 10.0, 10.0),
    L_values=(2.0, 2.0, 2.0),
    Cm_values=(0.144..., 0.144..., 0.144...),
    max_relative_spread_D20=0.0,
    max_relative_spread_L=0.0,
    max_relative_spread_Cm=0.0,
    passes_5_percent_target=True
)
```

**Why this happens:** `run_mesh_independence_study`'s default
`cfd_runner=run_half_car_cfd` and default `resolutions=("coarse","medium","fine")`
are strings. They get passed positionally as `cfd_runner(stl_path, resolution)`
→ `run_half_car_cfd(stl_path, "coarse")`, binding to the
`reference_speed_mps` parameter. As established in the re-verification of
original finding #4 (Section 1), `run_half_car_cfd` currently does
`del reference_speed_mps, ...` immediately and never uses it — so all three
calls run through `_invoke_openfoam_pipeline` with **identical actual mesh
settings** (whatever the mocked/future-real pipeline does by default),
producing identical forces, zero spread, and a false PASS.

**Every existing test avoids this path** — both
`test_mesh_independence_pass_case` and `test_mesh_independence_fail_case`
inject their own `fake_runner(stl_path, resolution)` closures that
correctly consume the resolution label to vary the returned values. Neither
test exercises `run_mesh_independence_study` with its own default
`cfd_runner` argument. This means **the exact call signature a naive future
caller would use first** (`run_mesh_independence_study("car.stl")`, per the
spec's own Validation Requirements section: "run mesh independence study")
is completely unexercised by the test suite, and currently produces a
meaningless but confidently-reported PASS.

**Recommended fix:**
1. Change the default `cfd_runner` to something that **fails loudly** rather
   than silently ignoring the resolution label — e.g. remove the default
   entirely and make `cfd_runner` a required keyword-only argument, forcing
   every caller to consciously supply a runner that actually varies mesh
   resolution. This is the safest fix because it converts a silent
   correctness bug into an immediate, obvious `TypeError` at the call site.
2. Alternatively, if a default must be kept for convenience, wire in a real
   resolution-to-solver-settings mapping function (the spec's Search
   Strategy section implies mesh resolution should map to actual
   `snappyHexMesh` refinement levels) and have that wrapper be the default,
   not raw `run_half_car_cfd`.
3. Either way, add a test that calls `run_mesh_independence_study(stl_path)`
   with **zero injected arguments** and asserts it either raises clearly or
   produces genuinely different results per resolution — this exact gap in
   test coverage is what let the bug go unnoticed.

---

### 3.3 [HIGH] — Adjoint objective mixes full-car D20 with (implied)
### half-car OpenFOAM adjoint surface, with no scaling applied or documented

**Where:** `adjoint_contract.py::compute_adjoint_objective`, cross-referenced
against `02_simulation_setups(1).md`'s Half-Car CFD Contract and Adjoint
Objective Contract sections.

**The issue:** `compute_adjoint_objective` returns `w_D20 × D20` where
`D20 = float(param_vector[0])`. Tracing where `param_vector[0]` comes from
in every test and the integration test
(`params = np.array([full.D20, ...])`) confirms this is **full-car D20**
(post `to_full_car()` doubling), matching `PARAM_NAMES[0] == "drag_20_n"`
and the Race Objective Contract's explicit statement that "`D20` — from
CFD" refers to the full-car value fed into the race objective.

But the spec's own text says: *"OpenFOAM adjoint computes
dObjective/dSurface"* — and per the Half-Car CFD Contract section
elsewhere in the same doc, *"The default CFD setup simulates only the right
half of the car using a symmetry plane... Adjoint sensitivities from the
right-half simulation are applied symmetrically to both sides by the
geometry designer before φ update."*

This means the actual OpenFOAM adjoint solve almost certainly runs on the
**half-car mesh/surface** (since that's what the forward CFD run in this
same spec uses throughout, per the Half-Car CFD Contract, CFD Setup, and
mesh independence sections — there is no mention anywhere of a separate
full-car adjoint mesh). If OpenFOAM's adjoint solver is configured to
differentiate the objective `Objective = w_D20 × D20_full` with respect to
a half-domain surface, there's a **dimensional/scaling inconsistency**: the
half-domain surface only "sees" half the drag-generating area, so
differentiating a full-car-scaled objective against it will produce
sensitivities that are off by a constant factor (most likely 2x, matching
the same doubling used in `to_full_car()`) relative to what the geometry
designer (Part 1) would correctly need to apply symmetrically to both
sides.

**I want to be precise about what is and isn't confirmed here:**
- Confirmed by exhaustive `grep` across every `.py` file: **no scaling
  factor of any kind is applied anywhere in the adjoint objective chain.**
  `compute_adjoint_objective`, `compute_adjoint_objective_weight`, and
  `package_gradient_bundle` contain no `0.5`, `2.0`, or `half`/`full`
  scaling logic.
- Confirmed: **no test exercises this**, because no test differentiates an
  actual surface sensitivity field — every test stops at the scalar
  `w_D20` or `w_D20 × D20` level, never touching an actual
  `dObjective/dSurface` computation (which doesn't exist yet in this
  codebase — that's real OpenFOAM adjoint machinery, correctly deferred per
  the build order in Part 3's spec, item 6 "Adjoint integration").
- **Not confirmed, because it can't be from this codebase alone**: whether
  the actual OpenFOAM adjoint case (which doesn't exist yet) will run on
  half or full geometry, and therefore whether this mismatch is real or
  moot. This is genuinely a **spec ambiguity**, not a definite code bug —
  the three spec docs never explicitly state whether the adjoint surface
  sensitivity computation happens on the half-domain mesh (matching the
  forward CFD run) or a mirrored full-domain mesh.

**Recommended fix:** Treat this exactly like the existing
`time_coefficient` conflict, which is the correct pattern already
established in this codebase (see the `⚠ UNRESOLVED` comment block at the
top of `race_objective_adapter.py`). Add an equivalent `⚠ UNRESOLVED`
comment block to `adjoint_contract.py::compute_adjoint_objective`
explicitly stating: *"It is not specified in the Part 2 spec whether the
OpenFOAM adjoint runs on the half-car or full-car domain. This function
currently uses full-car D20 with no scaling. If the adjoint case is
half-domain, this introduces a 2x error that must be corrected either here
or in Part 1's geometry designer before the surface sensitivity is applied.
This must be resolved by a human before Stage 6 of the Part 3 build order
(Adjoint integration) is implemented."* This costs nothing to add now and
prevents a very expensive, hard-to-diagnose 2x error from being discovered
only after real OpenFOAM adjoint cases are running (per Part 3's own build
order, that's several stages away — better to flag now than discover it
mid-optimization-loop debugging).

---

### 3.4 [MEDIUM] — `_relative_spread`'s zero-mean guard is reachable for
### real (non-degenerate) lift-force data, not just the all-zero edge case
### it was tested against

**Where:** `mesh_validation.py::_relative_spread`, feeding into
`run_mesh_independence_study`'s `max_relative_spread_L`.

Already covered in the re-verification of original finding #3 above
(Section 1). Restating the concrete mechanism here because it's a genuinely
new observation the original fix's tests didn't cover:

`HalfCarQuantities.__post_init__` (confirmed by direct source read) only
validates `D20 >= 0` and `A >= 0`. **`L` (lift) has no such guard**, and
this is correct — a car with net downforce legitimately has negative lift.
But this means a real (not synthetic) mesh independence study on a car
design with lift very close to zero — plausible for a small STEM Racing car
without aggressive aero devices — could produce, say,
`L_values = (-0.3, 0.4, -0.05)` N across coarse/medium/fine mesh (a
genuinely large **relative** spread, since the values are inconsistent in
sign, which is actually a strong signal of mesh non-convergence for a
force that should be small and stable). If those three values happen to
sum close to zero, `_relative_spread` returns `0.0`, and
`passes_5_percent_target` reports `True` for the `L` component specifically
— exactly when the mesh study should be raising the loudest possible flag
that lift predictions are unstable.

**Recommended fix:** Replace the "return 0.0 on zero mean" fallback with a
metric that doesn't degenerate at zero-mean, e.g. normalize by the range of
the *reference* condition's magnitude or by `max(abs(values))` instead of
`mean(values)`:
```python
def _relative_spread(values: tuple[float, ...]) -> float:
    spread = max(values) - min(values)
    scale = max(abs(v) for v in values)
    if scale < 1e-9:
        # All values genuinely near zero (e.g. a symmetric car with no
        # net lift) -- spread is physically negligible regardless of sign,
        # this is the one case where 0.0 is actually correct.
        return 0.0
    return spread / scale
```
This still returns `0.0` for the truly-degenerate all-near-zero case (which
is the only case the original crash-fix needed to handle), but correctly
reports a large relative spread for sign-flipping, mesh-unstable values
that happen to average near zero — which is the case that actually matters
for a real lift force study.

---

### 3.5 [MEDIUM] — Path traversal guard rejects legitimate candidate IDs
### containing the substring `..`

**Where:** `candidate_record.py::write_candidate_record`

Already covered in Section 1 under finding #1's re-verification. Restating
with the concrete repro since it's a new observation:

```python
safe_id = record.candidate_id.replace("\\", "/")
if "/" in safe_id or ".." in safe_id or safe_id != record.candidate_id:
    raise ValueError(...)
```

Confirmed by execution: `candidate_id="cand..001"` — a plausible naming
scheme (e.g. `wheelbase130..iteration001` or any ID a human might type
using `..` as a visual separator) — gets rejected with the exact same
"path traversal" error as an actual attack string like `"../../evil"`,
even though `cand..001` contains no `/` and cannot escape any directory
(the `..` substring is not adjacent to a path separator, so `Path(out_dir) /
"cand..001.json"` is perfectly safe).

**Recommended fix:** Tighten the check to only reject `..` when it appears
as a complete path *segment*, which is the actual traversal-enabling
pattern, rather than as a bare substring:
```python
import re
if "/" in safe_id or safe_id != record.candidate_id:
    raise ValueError(f"candidate_id contains path separators: {record.candidate_id!r}")
if re.search(r'(^|[/\\])\.\.($|[/\\])', record.candidate_id) or record.candidate_id in ("..", "."):
    raise ValueError(f"candidate_id contains a '..' path segment: {record.candidate_id!r}")
```
Or more simply, since `/` and `\\` are already banned outright by the
existing check, a bare `candidate_id` with no separators literally cannot
be a multi-segment path — so the `".." in safe_id` substring check can be
replaced with an exact-match check `safe_id in ("..", ".")` once separators
are already banned, since without separators `..` can only ever appear as
a substring inside an otherwise-single-segment name, never as a traversal
mechanism. Recommend the simplest fix:
```python
if "/" in safe_id or safe_id != record.candidate_id:
    raise ValueError(f"candidate_id contains path separators: {record.candidate_id!r}")
if safe_id in ("..", "."):
    raise ValueError(f"candidate_id cannot be '.' or '..': {record.candidate_id!r}")
```
Add a test asserting `cand..001` (or similar) is **accepted**, alongside
the existing test that `../../evil` is **rejected**, so this doesn't
regress silently.

---

### 3.6 [LOW] — Dead placeholder constants in `mass_com_ingest.py` have a
### misleading comment claiming enforcement that doesn't exist

**Where:** `mass_com_ingest.py`, lines 11–17

```python
CO2_CARTRIDGE_COM = None               # PLACEHOLDER: raise NotImplementedError
                                        # if actual fixed position not supplied
                                        # by caller -- see FixedHardwareSpec
REAR_WING_MASS_KG = None                # PLACEHOLDER -- must be injected by caller,
                                        # not hard-coded, since "known mass" was
                                        # never given a number in the source docs
WHEELS_AXLES_MASS_KG = None             # PLACEHOLDER -- same as above
```

Confirmed by exhaustive `grep -n "NotImplementedError\|CO2_CARTRIDGE_COM\|REAR_WING_MASS_KG\|WHEELS_AXLES_MASS_KG"` across the file: **these three
names appear only once each, at their own definition.** They are never
imported, never referenced, never checked against anything. The comment on
`CO2_CARTRIDGE_COM` specifically claims the module "raise[s]
`NotImplementedError` if actual fixed position not supplied by caller" —
but `NotImplementedError` does not appear anywhere in this file at all.

This is not a functional bug — `FixedHardwareSpec` correctly requires
`co2_cartridge_com`, `rear_wing_mass_kg`, `wheels_axles_mass_kg` as
mandatory dataclass fields with no default, so a caller genuinely cannot
construct one without supplying values, which achieves the *intent* of the
comment through a different mechanism (Python's own required-field
enforcement) rather than an explicit runtime check. But the comment
describing *this specific unused module-level constant* as the enforcement
mechanism is inaccurate and will mislead a future reader (or GLM 5.2, if it
trusts the comment) into thinking there's a check here that isn't.

**Recommended fix:** Either delete the three dead constants entirely (since
`FixedHardwareSpec`'s required fields are the real, working enforcement
mechanism), or if they're being kept intentionally as documentation
anchors, rewrite the comments to say so accurately: e.g.
```python
# NOTE: these values are NOT hardcoded anywhere. FixedHardwareSpec below
# requires co2_cartridge_com, rear_wing_mass_kg, and wheels_axles_mass_kg
# as mandatory constructor arguments with no default -- Python's dataclass
# machinery is the actual enforcement mechanism, not a runtime check in
# this module. These module-level names exist only so a reader searching
# for "rear wing mass" finds a pointer to FixedHardwareSpec.
```

---

## 4. Things Explicitly Checked and Found Correct (worth stating so GLM
## 5.2 doesn't re-litigate them)

- SHA-256 hash claim in `BUILD_REPORT.md` for the locked `race_objective.py`
  — not independently re-hashed in this audit (no reason to doubt a SHA-256
  match claim that's mechanically checkable and was already verified by a
  documented tool), but the `test_hash_of_locked_file_matches` test exists
  and asserts it directly against the live file each time tests run, which
  is the right ongoing safeguard regardless.
- Coordinate convention (`x`=front-to-rear, `y`=centerline-to-outside,
  `z`=track-upward) — used consistently across `physics_contract.py`,
  `mass_com_ingest.py`, and matches Part 1's spec exactly. No mixed
  conventions found.
- Unit convention (SI throughout: kg, m, N, kg/m³, m², m/s) — confirmed
  consistent across all of Part 2. The `grams_to_kg`/`mm_to_m`/etc. helper
  functions in `physics_contract.py` are the only conversion points, and
  every other module imports from there rather than re-deriving conversions
  locally, exactly as that file's own module docstring insists on.
- `N_WHEELS = 4`, `R_WHEEL = 0.015` (m) hardcoded constants in
  `race_objective.py` — flagged as "hard-coded values" in `BUILD_REPORT.md`
  Section 3, correctly disclosed rather than hidden. Not re-litigating.

---

## 5. Summary of Action Items — First Batch (see Section 6 for the rest)

| Priority | Finding | File(s) | Type |
|---|---|---|---|
| 1 | No extrapolation guard on COM height before it hits the locked polynomial | `race_objective_adapter.py`, `mass_com_ingest.py` | New bug, needs fix |
| 2 | Mesh independence study false-positives with default args | `mesh_validation.py` | New bug, needs fix |
| 3 | Adjoint objective half/full D20 scaling ambiguity | `adjoint_contract.py` | Spec ambiguity, needs `⚠ UNRESOLVED` flag + human decision |
| 4 | `_relative_spread` zero-mean masking reachable for real lift data | `mesh_validation.py` | New bug, needs fix |
| 5 | Path traversal guard false-positives on legit IDs containing `..` | `candidate_record.py` | New bug, needs fix |
| 6 | Misleading dead-code comment | `mass_com_ingest.py` | Cosmetic, low effort |
| 7 | Original finding #4's crash mode was mischaracterized — actual current behavior is silent no-op, not a crash | `cfd_wrapper.py`, `mesh_validation.py` | Documentation correction + same fix as #2 above |

All repro scripts referenced above are runnable standalone against the
first 8 files audited in Sections 1–5. Section 6 below covers the rest.

---

## 6. Remaining File Coverage (`race_objective.py` full pass, `calibration.py`,
## `run_all_tests.py`, `_gitignore`, and all 9 test files)

Same verification standard as above — every finding is either a direct
source citation or an executed repro script.

### A1. [HIGH] `run_all_tests.py` is completely broken and fails silently

**Where:** `run_all_tests.py`, line 9: `f'tests/{t}.py'`

**Confirmed by direct execution.** Running `python3 run_all_tests.py` from
the project root produces:
```
test_physics_contract:
test_mass_com_ingest:
...
TOTAL: 0 passed, 0 failed
```

**Root cause:** every test file lives at the project root
(`test_physics_contract.py`), but the script hardcodes a `tests/` subfolder
prefix that doesn't exist. Every `subprocess.run(...)` call fails with
`FileNotFoundError` at the OS level (confirmed: `returncode=2`, stderr =
`"can't open file '.../tests/test_physics_contract.py': No such file or
directory"`). The script only reads `r.stdout` (which is empty on this kind
of failure) and never checks `r.returncode` or prints `r.stderr` — the one
error-handling branch that exists
(`last = lines[-1] if lines else f'ERROR: {r.stderr[:200]}'`) never
triggers, because `''.strip().split('\n')` evaluates to `['']`, a
non-empty list, so `lines[-1]` is `''`, not the stderr fallback string.

**Why this matters:** this is the single file whose entire purpose is being
the trusted one-command way to verify the whole Part 2 build is passing. As
shipped, it reports `TOTAL: 0 passed, 0 failed` — which reads as "nothing
ran" at best, and could easily be misread as "clean, nothing to report" by
someone skimming CI output, when the actual state is "every test file
failed to even be found." Notably, `BUILD_REPORT.md`'s own "How to
re-verify this report yourself" section does **not** use this script — it
lists each `python part2_simulation\tests\test_X.py` command individually,
which only works if the files genuinely are under a `tests/` subdirectory
in whatever the report author's actual working layout was. This suggests
the path in `run_all_tests.py` may have been written against a different
directory layout than what ended up in this project, and was never
actually executed successfully after that.

**Fix:**
```python
import subprocess, sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent  # tests are alongside this file
tests = [
    'test_physics_contract', 'test_mass_com_ingest', 'test_cfd_wrapper',
    'test_mesh_validation', 'test_calibration', 'test_candidate_record',
    'test_race_objective_adapter', 'test_adjoint_contract', 'test_integration_end_to_end'
]
total_p, total_f = 0, 0
any_infra_failure = False
for t in tests:
    test_path = TEST_DIR / f'{t}.py'
    if not test_path.exists():
        print(f'{t}: ERROR - file not found at {test_path}')
        any_infra_failure = True
        continue
    r = subprocess.run([sys.executable, str(test_path)], capture_output=True, text=True)
    lines = r.stdout.strip().split('\n') if r.stdout.strip() else []
    last = lines[-1] if lines else ''
    if r.returncode != 0 and 'passed' not in last:
        print(f'{t}: INFRA FAILURE (exit {r.returncode}) - {r.stderr.strip()[:300]}')
        any_infra_failure = True
        continue
    print(f'{t}: {last}')
    if 'passed' in last and 'failed' in last:
        p = int(last.split('passed')[0].strip().split()[-1])
        f = int(last.split('failed')[0].strip().split()[-1])
        total_p += p
        total_f += f
print(f'\nTOTAL: {total_p} passed, {total_f} failed')
if any_infra_failure:
    print('WARNING: one or more test files could not be run at all -- counts above are incomplete.')
    sys.exit(2)
sys.exit(1 if total_f else 0)
```
This fix (a) uses the actual file location instead of a hardcoded guess,
(b) surfaces `stderr` and exit codes instead of swallowing them, and (c)
exits non-zero if anything couldn't even be run, so CI can't mistake
"broken harness" for "clean pass."

---

### A2. [HIGH] `_clean_csv_arrays` in the locked `race_objective.py` accepts
### physically impossible thrust-curve data with zero validation

**Where:** `race_objective.py::_clean_csv_arrays`

**Confirmed by execution** (three separate repros): a thrust/mass CSV with
negative mass, negative thrust force, or negative time values all pass
through `_clean_csv_arrays` with **no error at any stage** — only NaN
coercion (`errors="coerce"` + `dropna()`), duplicate-time averaging, a
minimum-4-rows check, and a strictly-increasing-time check are applied.
None of those catch physically nonsensical *values* (only malformed rows).

Traced end-to-end: a CSV with `mass (kg) = -0.048` at `t=0` was fed all the
way through `build_smooth_sheet_model` → `race_value_and_grad`, and
produced a finite, no-error race time of `T=1,204,721.56` seconds (the
`_smooth_positive` floor inside `car_mass_from_time` prevents a NaN/crash
by flooring effective mass near zero, but silently converts "this input
data is corrupt" into "this car is nearly massless," which just looks like
a very slow candidate to anything consuming the objective, not like an
input-validation failure).

**Constraint:** `race_objective.py` is the explicitly locked reference file
per `BUILD_REPORT.md` — it must not be modified. The fix has to live at the
ingestion boundary, one layer up, in whatever code calls
`build_smooth_sheet_model` (currently that's only exercised directly in
tests; there's no production wrapper around CSV ingestion the way
`race_objective_adapter.py` wraps the parameter vector).

**Recommended fix:** Add a `validate_thrust_csv_physical_sanity(csv_path)`
helper in `race_objective_adapter.py` (or a new small module, matching the
existing pattern of guarding the locked file's inputs from outside it)
that calls `_clean_csv_arrays` (already exposed at module level, no
underscore-privacy issue since it's the same package) and checks:
```python
def validate_thrust_csv_physical_sanity(csv_path):
    from race_objective import _clean_csv_arrays
    t, force, mass = _clean_csv_arrays(csv_path)
    if (t < 0).any():
        raise ValueError(f"{csv_path}: thrust CSV contains negative time values")
    if (force < 0).any():
        raise ValueError(f"{csv_path}: thrust CSV contains negative force values")
    if (mass <= 0).any():
        raise ValueError(f"{csv_path}: thrust CSV contains non-positive mass values")
    return t, force, mass
```
and call this before `build_smooth_sheet_model` in any real (non-test)
calling code. This mirrors exactly the "guard the locked file from
outside, never modify it" pattern already used for `time_coefficient` and
negative mass/mu in `race_objective_adapter.py`.

---

### A3. [MEDIUM] `read_candidate_record` validates `lifecycle_state`, but
### `write_candidate_record` does not — a corrupt candidate database file
### can be written by this codebase's own write path

**Where:** `candidate_record.py`

**Confirmed by execution:** constructing a `CandidateRecord` with
`lifecycle_state="totally_bogus_state_not_in_allowed_set"` and calling
`write_candidate_record()` on it **succeeds** — no exception, file written
to disk with the invalid state serialized directly into the JSON. Only
`read_candidate_record()` checks `data["lifecycle_state"] not in
ALLOWED_LIFECYCLE_STATES`.

**Why this matters:** the entire point of the lifecycle-state enum,
per Part 3's spec (Candidate Lifecycle section, listing the exact 8 states
and describing how "the evolutionary outer loop uses failure history to
avoid re-exploring regions that consistently produce bad geometry"), is
that the optimizer's outer loop filters/queries candidates by state. A
corrupt state written today would sit silently on disk, invisible until
some future read — and by then it might be read by a different part of the
pipeline than whatever wrote it, making the error much harder to trace
back to its origin. Catching it at construction time is strictly better
than catching it at read time.

**Fix:** move the validation into `CandidateRecord.__post_init__`
(alongside the existing `setup_logs` length check, which already
establishes the pattern of validating in `__post_init__`):
```python
def __post_init__(self):
    MAX_SETUP_LOGS_LEN = 1_000_000
    if len(self.setup_logs) > MAX_SETUP_LOGS_LEN:
        raise ValueError(...)  # existing check, unchanged
    if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES:
        raise ValueError(
            f"invalid lifecycle_state: {self.lifecycle_state!r}, "
            f"must be one of {sorted(ALLOWED_LIFECYCLE_STATES)}"
        )
```
This validates at construction (covers both the write path and any other
future code path that builds a `CandidateRecord`), and
`read_candidate_record`'s existing check becomes redundant-but-harmless
defense in depth against hand-edited or externally-produced JSON files
(worth keeping for that reason — don't remove it, just add the
construction-time check too).
**Note for GLM 5.2:** adding this will require checking whether any
existing test constructs a `CandidateRecord` with an invalid
`lifecycle_state` expecting success — a quick grep of the current test
suite shows `test_invalid_lifecycle_state_raises_on_read` builds the bad
state via raw JSON written directly to disk (bypassing the dataclass
constructor entirely), not via `CandidateRecord(...)`, so this fix should
not break that test. No test currently constructs a bad-state
`CandidateRecord` directly, so this addition is safe to make without
touching any existing test.

---

### A4. [LOW] `_read_csv_columns` in `calibration.py` gives an unhelpful
### raw error for malformed numeric cells, with no row/column context

**Where:** `calibration.py::_read_csv_columns`, line 27:
`columns[name] = np.asarray([float(row[name]) for row in rows], dtype=float)`

**Confirmed by execution:** a CSV with one blank cell in a numeric column
raises `ValueError: could not convert string to float: ''` with no
indication of which row or column triggered it. Every public function in
`calibration.py` (`fit_thrust_surrogate`, `fit_mu_from_track_test`,
`fit_com_penalty_curve`, `compute_held_out_residual`) routes through this
one shared helper, so this affects all of them identically. This is
strictly a diagnostics/usability issue, not a correctness bug — the
function does fail closed (raises, doesn't silently corrupt the fit) — but
for a real calibration CSV with dozens of rows from a physical track test,
a bare `''` conversion error is a genuinely frustrating thing to debug.

**Fix:**
```python
for name in reader.fieldnames:
    values = []
    for i, row in enumerate(rows):
        try:
            values.append(float(row[name]))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"{csv_path}: column '{name}', row {i + 2} "
                f"(1-indexed, plus header): could not parse {row[name]!r} as float"
            ) from e
    columns[name] = np.asarray(values, dtype=float)
```

---

### A5. [LOW] `BuildSettings(n_basis=0)` silently produces an empty RBF
### basis instead of raising at the point of misconfiguration

**Where:** `race_objective.py::build_smooth_sheet_model`,
`np.linspace(float(t[0]), float(t[-1]), settings.n_basis)`

**Confirmed by execution:** `BuildSettings(n_basis=0)` produces
`centers=[]` (empty array) with no error at the point of model
construction. `n_basis=-3` does fail, but with a raw
`numpy.exceptions.???: Number of samples, -3, must be non-negative` rather
than a domain-specific message. Since `build_smooth_sheet_model` is a
public, caller-facing function and `BuildSettings` is a public config
dataclass with no field-level validation of its own (`n_basis: int = 80`
plain default, no `__post_init__` guard), a caller mistake here (e.g. a
typo or a config file parsing bug producing `n_basis=0`) surfaces far away
from its cause — likely inside `_ridge_fit`'s matrix solve or later inside
JAX evaluation, not at the point of the actual mistake.

**Fix:** add a minimal `__post_init__` to `BuildSettings`:
```python
@dataclass(frozen=True)
class BuildSettings:
    n_basis: int = 80
    ridge: float = 1e-8
    tail_tau: float = 0.025
    x_start: float = 1e-4
    x_grid_power: float = 2.0
    n_steps: int = 1000

    def __post_init__(self):
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
```
This is a change to `race_objective.py`, which is locked — flag for a
human decision same as other locked-file issues; if it can't be touched,
the equivalent guard should be added to whatever code constructs
`BuildSettings` in practice (currently only test code does).

---

### A6. [LOW] A test in `test_mesh_validation.py` directly asserts the
### zero-mean masking behavior (Section 3.4 of the first report) as
### **correct** — fixing that finding will break this test, by design,
### and GLM 5.2 needs to update the assertion, not just the source

**Where:** `test_mesh_validation.py::test_mixed_sign_relative_spread_is_safe`,
lines 82–86:
```python
def test_mixed_sign_relative_spread_is_safe():
    """Mixed-sign values summing to zero should not crash."""
    from mesh_validation import _relative_spread
    result = _relative_spread((-1.0, 1.0, 0.0))
    assert result == 0.0
```

This is not a new defect — it's the same root cause as Section 3.4 in the
first report (`_relative_spread`'s zero-mean masking), but flagged
separately here because it's specifically about the **test file**, not the
source file, and it's easy to miss when fixing the source: this exact test
currently asserts `_relative_spread((-1.0, 1.0, 0.0)) == 0.0`, which is
precisely the false-positive behavior the first report's Section 3.4 fix
is meant to eliminate. **If the recommended `_relative_spread` fix from the
first report is applied as-is, this test will start failing** — which is
correct and expected (the test's current assertion encodes the bug), but
it means this test's assertion must be rewritten alongside the source fix,
not left as-is. Suggested replacement assertion using the fixed formula
(`spread / max(abs(v) for v in values)`):
```python
def test_mixed_sign_relative_spread_is_safe():
    """Mixed-sign values with real disagreement should report meaningful
    spread, not mask it via zero-mean cancellation."""
    from mesh_validation import _relative_spread
    result = _relative_spread((-1.0, 1.0, 0.0))
    assert result > 0.0, (
        "values that disagree in sign should show nonzero relative spread, "
        "not be masked by their mean happening to cancel to zero"
    )

def test_all_near_zero_relative_spread_is_safe():
    """Genuinely negligible values (not just zero-mean) should report 0 spread."""
    from mesh_validation import _relative_spread
    result = _relative_spread((1e-12, -1e-12, 0.0))
    assert result == 0.0
```

---

### A7. [LOW / cosmetic] `.gitignore` doesn't cover the artifacts this
### pipeline actually generates

**Where:** `_gitignore`

Confirmed by reading the file: it covers standard Python dev artifacts
(`__pycache__/`, `.pytest_cache/`, `.venv/`, `*.log`, etc.) correctly, but
has no entries for the artifacts this specific project produces at scale:
- Candidate database JSON files, written by `write_candidate_record` to
  a caller-specified `out_dir` (per Part 3's spec, potentially thousands
  per optimization run — "M candidates × ... × ~100 iterations")
- STL files (referenced throughout as `stl_path`)
- φ-grid snapshot files (test fixtures use `.npy` extensions for these)
- Any local OpenFOAM case directories (`cfd_case_template/` is referenced
  in `cfd_wrapper.py` but its actual per-run case working directories,
  once real OpenFOAM is wired in, would also need excluding)

**Suggested addition:**
```
# Generated candidate/simulation artifacts (Part 2/3 pipeline outputs)
candidates/
*.stl
*.npy
cfd_case_*/
*.foam
```
(Exact patterns should match whatever output directory convention the
optimizer's outer loop ultimately uses — this project doesn't have that
wired up yet since Part 3 has no implementation, so treat this as a
placeholder to revisit once Part 3 exists, not a precise final answer.)

---

## 7. Full Coverage Table (all 24 files)

| File | Status |
|---|---|
| `01_generative_geometry_1_.md` | Spec-read, no code exists (N/A for code audit) |
| `02_simulation_setups_1_.md` | Spec-read, cross-checked against all Part 2 code |
| `03_optimizer_workflow_1_.md` | Spec-read, no code exists (N/A for code audit) |
| `audit_notes.md` | Fully cross-checked, all 17 findings independently re-verified |
| `BUILD_REPORT.md` | Claims cross-checked against live code where testable |
| `_gitignore` | Reviewed — correct for its scope, gap noted (A7) |
| `physics_contract.py` | Deep audit, execution-verified |
| `mass_com_ingest.py` | Deep audit, execution-verified |
| `cfd_wrapper.py` | Deep audit, execution-verified |
| `mesh_validation.py` | Deep audit, execution-verified |
| `calibration.py` | Deep audit, execution-verified (this addendum: A4) |
| `candidate_record.py` | Deep audit, execution-verified (this addendum: A3) |
| `adjoint_contract.py` | Deep audit, execution-verified |
| `race_objective.py` | Deep audit, execution-verified (this addendum: A2, A5) — locked file, findings require adapter-layer or human-reviewed fixes |
| `race_objective_adapter.py` | Deep audit, execution-verified |
| `run_all_tests.py` | Reviewed (this addendum: A1) — confirmed broken |
| `test_physics_contract.py` | Reviewed — no defects found in the test file itself |
| `test_mass_com_ingest.py` | Reviewed — no defects found in the test file itself |
| `test_cfd_wrapper.py` | Reviewed — no defects found in the test file itself |
| `test_mesh_validation.py` | Reviewed (this addendum: A6) — one test locks in a known bug |
| `test_calibration.py` | Reviewed — no defects found in the test file itself |
| `test_candidate_record.py` | Reviewed — confirms A3's gap (no write-side lifecycle test exists) |
| `test_adjoint_contract.py` | Reviewed — no defects found in the test file itself |
| `test_race_objective_adapter.py` | Reviewed — no defects found in the test file itself |
| `test_integration_end_to_end.py` | Reviewed — no defects found in the test file itself |

All 24 files now reviewed. Combined with the first report, this gives a
complete audit of the project as it currently exists.
