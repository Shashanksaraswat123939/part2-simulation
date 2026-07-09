# Part 2 Simulation

This repository contains the second-stage simulation and optimization layer for a small race-car CFD workflow. It connects physical unit contracts, mass and center-of-mass data, CFD result validation, calibration models, a differentiable race objective, adjoint weights, and candidate-result storage.

The implementation is intentionally defensive: invalid geometry, impossible physical values, unsupported calibration inputs, failed CFD runs, and unresolved model assumptions are surfaced explicitly instead of being silently accepted.

## Current status

- 79 automated tests pass.
- The coordinate, unit, mass, COM, calibration, objective, gradient, and candidate-record contracts are implemented.
- The differentiable race objective uses JAX.
- The candidate database serializes simulation results to JSON.
- The real OpenFOAM execution pipeline is not connected yet.
- Several physical inputs still require real experimental data.

This is a development-stage engineering model. Passing tests establish software behavior; they do not validate the underlying aerodynamic model against physical measurements.

## Pipeline

1. **Physics contracts** — defines units, reference conditions, half-car/full-car conversions, and physical validation.
2. **Mass and COM ingestion** — combines component masses and centers of mass into a full-car report.
3. **CFD wrapper** — validates STL input and packages aerodynamic force and solver-health results.
4. **Mesh and solver validation** — checks mesh independence, solver agreement, and speed sensitivity.
5. **Calibration** — fits thrust and COM-penalty surrogate models while preventing silent extrapolation.
6. **Race objective** — predicts 20 m race time with a differentiable JAX model.
7. **Adjoint contract** — exposes objective sensitivities and packages optimizer gradients.
8. **Candidate records** — stores candidate geometry, CFD results, timing, gradients, logs, and lifecycle state.

## Repository layout

| Path | Purpose |
| --- | --- |
| `physics_contract.py` | Units, reference constants, physical dataclasses, and half/full-car conversion |
| `mass_com_ingest.py` | Component mass and center-of-mass aggregation |
| `cfd_wrapper.py` | STL checks, CFD execution contract, and health reporting |
| `mesh_validation.py` | Mesh-independence and solver-validation calculations |
| `calibration.py` | Thrust, friction, and COM-penalty calibration |
| `race_objective.py` | Locked differentiable race-time implementation |
| `race_objective_adapter.py` | Guarded interface between the locked objective and optimizer |
| `adjoint_contract.py` | Adjoint weights and gradient packaging |
| `candidate_record.py` | Candidate lifecycle model and JSON persistence |
| `cfd_case_template/` | Placeholder for real OpenFOAM case dictionaries |
| `tests/` | Standalone automated tests |
| `run_all_tests.py` | Full test runner |
| `BUILD_REPORT.md` | Detailed implementation, validation, and unresolved-issue report |
| `audit_notes.md` | OpenClaw audit notes produced during review |

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- pandas
- JAX
- jaxlib

Install the Python dependencies:

```powershell
python -m pip install numpy scipy pandas jax jaxlib
```

## Run the test suite

From the repository root:

```powershell
python run_all_tests.py
```

Expected result:

```text
TOTAL: 79 passed, 0 failed
```

The tests are plain Python scripts, so individual stages can also be run directly:

```powershell
python tests\test_physics_contract.py
python tests\test_cfd_wrapper.py
python tests\test_race_objective_adapter.py
```

## Important limitations

- `_invoke_openfoam_pipeline` is a deliberate placeholder. No real `snappyHexMesh` or `simpleFoam` workflow is currently executed.
- The `cfd_case_template` directory does not contain production OpenFOAM dictionaries.
- The rolling-friction fitting schema is unfinished; current calibration requires a precomputed `mu_fitted` column.
- Some mass and COM inputs must still be supplied by upstream geometry/manufacturing code.
- Fore-aft COM sensitivity remains a zero placeholder because the locked race objective does not model it.
- Lift and pitching-moment objective weights remain disabled until track and ballast experiments provide defensible values.
- The locked objective exposes `time_coefficient` as a parameter, while the physics contract freezes it at `1.0`. The adapter rejects any other value until that design conflict is resolved.
- The pitching-moment reference length uses a documented stand-in and must be replaced with the confirmed engineering convention.

Read `BUILD_REPORT.md` before using results for design decisions. It records the exact test outcomes, hard-coded values, placeholder inventory, and unresolved specification conflicts.

## Recommended next steps

1. Supply production OpenFOAM case dictionaries and implement the subprocess pipeline.
2. Add measured thrust, rolling-friction, ballast, and COM datasets.
3. Resolve the `time_coefficient` contract conflict.
4. Confirm the pitching-moment reference length.
5. Validate predicted race times against held-out physical track tests.
6. Add continuous integration so all tests run automatically on every commit.

## Data and safety

Do not commit private experimental data, credentials, API keys, or proprietary geometry unless the repository's access controls have been reviewed. The repository is private, but private Git hosting is not a substitute for proper secret management.
