# Part 2 — Physics: CFD and the race objective

| module | what |
|---|---|
| `openfoam_case.py` | the forward CFD case for a half car on ESI OpenFOAM v2412: domain (3 lengths ahead, 8 behind, symmetry plane, moving ground), snappyHexMesh with wake boxes, k-omega SST, Spalding wall function or a wall-resolved (y+ <= 2) layer stack, every Part 4 part as its own patch (wheels as rotating walls), forces per patch, y+ distribution per patch |
| `cfd_wrapper.py` | `run_half_car_cfd`: validate the STL, run the case, return full-car drag and lift with health (residual, standard error and drift of the force mean); `run_half_car_adjoint` for the drag adjoint |
| `openfoam_adjoint.py` | the adjoint case (`adjointOptimisationFoam`, frozen turbulence); it failed its gradient check on this car (right sign 3 of 8 modes), so the search uses direct CFD |
| `race_objective.py` | the locked, differentiable race-time model (JAX): thrust curve, rolling and aero drag, wheel inertia. Its hash is in `race_objective_hash.txt` |
| `race_objective_adapter.py`, `adjoint_contract.py` | the objective's parameter vector, guarded evaluation, and the drag weight dT/dD20 |
| `physics_contract.py`, `mass_com_ingest.py`, `candidate_record.py` | units and the half-car contract, mass and COM of the whole car, one JSON record per candidate |

Resolutions (`resolution=`): `coarse`, `medium` (the search, ~0.65 M cells with the wake
boxes), `fine`, `resolved` (~5 M cells, wall-resolved).

```bash
pip install -r ../part1-simulation/requirements.txt
python -m pytest tests -q
```
