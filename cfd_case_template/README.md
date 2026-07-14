# Half-car CFD case (ESI OpenFOAM)

This directory is the base path handed to `openfoam_case.invoke()`. It no longer
holds static dictionaries — the entire case is **generated from Python**
(`openfoam_case.py`) for each candidate, so geometry-dependent values (domain
box, `locationInMesh`, turbulence inlet values, `Aref`) are always consistent
with the STL being solved. Generated runs land under `runs/` (git-ignored).

## Why ESI (openfoam.com), not the Foundation (openfoam.org)

The project's adjoint shape sensitivity is produced by `adjointOptimisationFoam`,
which ships only in the ESI line. The primal case here uses the same skeleton
(symmetry plane on the centreline, `simpleFoam`, force function objects) so the
adjoint solve can be added later without re-deriving the setup. Tested against
ESI **v2206+**.

## What the pipeline does (matches the Part 2 spec "CFD Setup")

Half-car domain, right side only (`y >= 0`, `symmetryPlane` at `y = 0`), 20 m/s,
air density 1.225 kg/m³.

1. `surfaceFeatureExtract` → edge features for snapping (ESI's utility name;
   NOT `surfaceFeatures`, despite that being the more commonly documented one)
2. `blockMesh` → wind-tunnel background mesh
3. `snappyHexMesh -overwrite` → body-fitted mesh (boundary layers when
   `turbulence_model="kOmegaSST"`)
4. `checkMesh` → negative-volume-cell gate
5. `simpleFoam` → steady solve; `forces`/`forceCoeffs`/`yPlus` function objects

Returns (all **half-car**): `D20_half`, `L_half`, `A_half`,
`pitching_moment_half`, plus health: `residual_final`, `negative_volume_cells`,
`y_plus_min`, `y_plus_max`, `courant_max`. `cfd_wrapper.run_half_car_cfd`
doubles forces/area to full-car via `to_full_car()`.

`A_half` is computed from the STL geometry (front-facing projected silhouette on
the y–z plane), not read back out of the solve — per the spec ("A = frontal
projected area from geometry").

## Running locally

Requires ESI OpenFOAM sourced in the shell (so `$WM_PROJECT_DIR` / `$FOAM_BASHRC`
resolve), plus a `bash` on PATH. Then:

```python
from cfd_wrapper import run_half_car_cfd
half, health = run_half_car_cfd("car_0001_half.stl")          # laminar baseline
half, health = run_half_car_cfd("car_0001_half.stl",
                                turbulence_model="kOmegaSST")  # validation model
```

If no ESI environment is found, `run_half_car_cfd` raises `CFDRunError`
(→ `CFD_failed` lifecycle state), never a bare crash.

## Turbulence

- `laminar` — spec baseline (Re ≈ 270k, marginal). Default.
- `kOmegaSST` — validation model; adds inlet `k`/`omega`/`nut`, wall functions,
  and 3 snappy boundary layers.
