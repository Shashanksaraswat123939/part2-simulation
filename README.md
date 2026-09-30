# Part 2 — Physics: CFD on SimScale and the race objective (simscale branch)

| module | what |
|---|---|
| `simscale_case.py` | the half-car CFD on SimScale through the Python SDK: builds the flow domain (box minus the half car, one named face per boundary), imports it, meshes it (Simmetrix, a 15 µm first layer for y+ <= 2, wake boxes), runs k-omega SST with a moving ground and rotating wheels, and reads the forces per part. `python simscale_case.py probe <run_car folder>` checks the face mapping for free. |
| `cfd_wrapper.py` | `run_half_car_cfd`: validate the STL, run it on SimScale, return half-car forces with health (standard error and drift of the force mean, final pressure residual) |
| `race_objective.py` | the locked, differentiable race-time model (JAX): thrust curve, rolling and aero drag, wheel inertia. Its hash is in `race_objective_hash.txt` |
| `race_objective_adapter.py`, `adjoint_contract.py` | the objective's parameter vector, guarded evaluation, and the drag weight dT/dD20 |
| `physics_contract.py`, `mass_com_ingest.py`, `candidate_record.py` | units and the half-car contract, mass and COM of the whole car, one JSON record per candidate |

Mesh presets (`resolution=`): `coarse`, `medium`, `fine`, `resolved` (~5 M cells).
Credentials come from the environment: `SIMSCALE_API_KEY`, `SIMSCALE_PROJECT_ID`.

```bash
pip install "git+https://github.com/SimScaleGmbH/simscale-python-sdk.git@19.1.0"
python -m pytest tests -q
```
