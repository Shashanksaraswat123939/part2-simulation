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

The table below explains each important file twice: first in simple language, then with enough technical detail for someone working on the code.

| Path | Easy explanation | What it does technically |
| --- | --- | --- |
| `physics_contract.py` | The project's rulebook for measurements and basic physics. It makes sure every part of the program agrees on units such as metres, millimetres, kilograms, newtons, and seconds. | Defines reference air density and speed, typed data structures for aerodynamic and mass results, unit conversions, input validation, and the rules for converting half-car CFD forces and areas into full-car values. It also calculates the pitching-moment coefficient while guarding against invalid or near-zero areas. |
| `mass_com_ingest.py` | Combines the weight and position of every car component to find the total car mass and its balance point. | Accepts component-level mass and center-of-mass coordinates, validates that masses are physically valid, includes required fixed components such as the CO2 cartridge, and returns a mass-weighted full-car center of mass in metres. |
| `cfd_wrapper.py` | Checks that a 3D car file is usable, sends it to the CFD process, and turns the solver output into a clean result. | Validates the STL path and basic mesh topology, rejects unsupported or invalid geometry, defines the interface for the future OpenFOAM runner, converts half-car solver results to full-car aerodynamic quantities, and records convergence, residuals, negative cells, y-plus, and Courant-number health information. The real OpenFOAM subprocess is not connected yet. |
| `mesh_validation.py` | Checks whether the CFD answer is trustworthy instead of accepting the first result blindly. | Compares coarse, medium, and fine mesh results against the mesh-independence tolerance; compares solver outputs; calculates relative differences safely when values are zero or change sign; and estimates drag-area sensitivity across different speeds. |
| `calibration.py` | Learns useful physics values from test data, such as the CO2 thrust curve, rolling resistance, and how COM position affects time. | Reads and validates CSV datasets, fits a radial-basis-function thrust surrogate, validates or imports a fitted rolling-friction coefficient, fits a polynomial COM-penalty curve, blocks extrapolation outside measured ranges, and calculates held-out R² scores to check fit quality. |
| `race_objective.py` | Predicts how long the car should take to travel 20 metres and shows how changing each design value affects that time. | Contains the locked JAX-differentiable race model. It fits smooth thrust and mass curves, models drag, rolling friction, wheel inertia, changing CO2 mass, and COM-height penalty, integrates the distance-domain equations with RK4, and uses automatic differentiation to return race-time gradients. |
| `race_objective_adapter.py` | Safely connects the race-time model to the rest of the optimizer without changing the locked model. | Validates the optimizer parameter vector, forces the unresolved `time_coefficient` to remain `1.0`, separates raw race time from penalized race time, renames gradients into the Stage 7 contract, and leaves fore-aft COM sensitivity as an explicit zero placeholder because the locked model does not calculate it. |
| `adjoint_contract.py` | Packages the model's sensitivity results so an optimizer knows which design changes should improve race time. | Calculates the drag objective weight from the guarded race model, creates a consistent gradient bundle for drag, mass, COM height, COM position, and manufacturing sensitivity, and keeps lift and pitching-moment weights disabled until the required experiments and CFD connections exist. |
| `candidate_record.py` | Saves every attempted car design and records whether it worked, failed, or was rejected. | Defines the candidate lifecycle states and serializes candidate geometry settings, file paths, mass/COM reports, CFD forces, objective values, gradients, logs, and failure reasons to JSON. It validates lifecycle values and rejects data that cannot be safely serialized. |
| `cfd_case_template/` | The future home of the OpenFOAM setup files. It is currently only a placeholder. | Intended to contain production `snappyHexMesh`, `simpleFoam`, turbulence, boundary-condition, and solver dictionaries. These files have not been supplied, so the repository cannot yet launch a real OpenFOAM case. |
| `tests/` | Small programs that deliberately check whether each part of the project behaves correctly. | Contains separate test scripts for physics contracts, mass/COM ingestion, STL and CFD handling, mesh validation, calibration, candidate storage, the objective adapter, adjoint calculations, and the end-to-end mocked pipeline. The tests cover expected results and important failure cases. |
| `run_all_tests.py` | Runs every test and gives one simple pass/fail total. | Launches each standalone test module with the current Python interpreter, captures its output, extracts the number of passes and failures, and prints the combined result. The current expected total is `79 passed, 0 failed`. |
| `BUILD_REPORT.md` | A detailed engineering diary explaining what was built, tested, assumed, and left unfinished. | Records stage-by-stage test results, hard-coded constants, placeholder locations, integrity checks, specification conflicts, hardening changes, and commands for independently reproducing the validation. Read this before treating the model as production-ready. |
| `audit_notes.md` | Notes from the automated review that inspected the project for mistakes and weak assumptions. | Summarizes the OpenClaw review activity and findings. It is supporting audit evidence, not executable code and not a substitute for the formal build report or independent engineering validation. |

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
