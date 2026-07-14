"""Stage 3 half-car CFD wrapper around a future OpenFOAM pipeline."""

from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from physics_contract import (
    AIR_DENSITY_KGM3,
    MOMENT_REFERENCE_POINT_M,
    REFERENCE_SPEED_MPS,
    HalfCarQuantities,
)


@dataclass(frozen=True)
class CFDHealthReport:
    """CFD health fields.

    All force inputs are reported elsewhere. residual_final is dimensionless,
    negative_volume_cells is a count, y_plus_min/y_plus_max are dimensionless,
    and courant_max is dimensionless or None.

    Invalid input behavior:
        This dataclass performs no validation; run_half_car_cfd raises
        CFDRunError for invalid mesh conditions.
    """

    converged: bool
    residual_final: float
    negative_volume_cells: int
    y_plus_min: float
    y_plus_max: float
    courant_max: Optional[float] = None


class CFDRunError(Exception):
    """Raised when meshing or solving fails in a way that cannot produce
    a usable force report. Callers (Stage 6/7/8) must catch this and route
    to the 'CFD_failed' candidate lifecycle state -- never let it propagate
    unhandled into the optimizer loop."""


def _invoke_openfoam_pipeline(stl_path, case_dir, run_config=None) -> dict:
    """Run the real ESI OpenFOAM half-car pipeline and return the force/health
    dict. Delegates to openfoam_case.invoke (case generation → snappyHexMesh →
    simpleFoam → force/residual/y+ parsing).

    A missing OpenFOAM install is turned into CFDRunError so the caller routes
    it to the 'CFD_failed' lifecycle state instead of crashing the optimizer.
    The run_config keyword is optional and defaulted so the historical
    two-argument call site (and tests that monkeypatch this function) keep
    working unchanged.
    """
    import openfoam_case

    try:
        return openfoam_case.invoke(stl_path, case_dir, cfg=run_config)
    except openfoam_case.OpenFOAMNotFoundError as exc:
        raise CFDRunError(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise CFDRunError(
            f"OpenFOAM stage failed (exit {exc.returncode}): {getattr(exc, 'cmd', '?')}. "
            "See logs/ in the run directory."
        ) from exc


def _read_ascii_stl_triangles(stl_path: str) -> list[tuple[tuple[float, float, float], ...]]:
    path = Path(stl_path)
    raw = path.read_bytes()
    # Check for binary STL (starts with binary header, not 'solid')
    if not raw.lstrip().startswith(b"solid"):
        raise CFDRunError(
            "STL file is not ASCII format (binary STL detected). "
            "Only ASCII STL with vertex lines is supported."
        )
    vertices = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append(tuple(float(v) for v in parts[1:4]))
    if len(vertices) % 3 != 0 or not vertices:
        raise CFDRunError("STL does not contain a valid triangle vertex list")
    # Enforce right-half-only contract (SPEC §16): all vertices must have y >= -1e-6.
    # A full-car STL fed here would double forces silently via to_full_car() — P2-1.
    min_y = min(v[1] for v in vertices)
    if min_y < -1e-6:
        raise CFDRunError(
            f"Half-car STL has vertex with y={min_y:.6f} < -1e-6. "
            "Expected right-half only (y >= 0). Part 1 must export a right-half STL; "
            "a full-car STL would silently double aerodynamic forces."
        )
    return [tuple(vertices[i:i + 3]) for i in range(0, len(vertices), 3)]


def _assert_watertight_stl(stl_path: str) -> None:
    triangles = _read_ascii_stl_triangles(stl_path)
    edge_counts = Counter()
    for tri in triangles:
        arr = np.asarray(tri, dtype=float)
        if arr.shape != (3, 3):
            raise CFDRunError("STL triangle has invalid shape")
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(arr[i]), tuple(arr[j]))))
            edge_counts[edge] += 1
    bad_edges = [edge for edge, count in edge_counts.items() if count != 2]
    if bad_edges:
        raise CFDRunError("STL is not watertight: edge manifold check failed")


def run_half_car_cfd(
    stl_path: str,
    reference_speed_mps: float = REFERENCE_SPEED_MPS,
    air_density_kgm3: float = AIR_DENSITY_KGM3,
    max_iterations: int = 2000,
    turbulence_model: str = "laminar",
    resolution: str = "medium",
) -> tuple[HalfCarQuantities, CFDHealthReport]:
    """
    Validate a half-car STL and package OpenFOAM half-domain outputs.

    Args:
        stl_path: filesystem path to an STL file.
        reference_speed_mps: reference speed in m/s. Honored (sets the inlet U
            and the coefficient magUInf) — previously discarded (audit P2-11).
        air_density_kgm3: air density in kg/m^3. Honored (sets rhoInf and the
            kinematic viscosity nu = mu_air / rho).
        max_iterations: steady solver iteration cap (controlDict endTime).
        turbulence_model: "laminar" (spec baseline) or "kOmegaSST" (validation
            model; grows boundary layers in snappyHexMesh).
        resolution: "coarse" | "medium" | "fine" — snappyHexMesh refinement
            level, used by the mesh-independence study.

    Returns:
        (HalfCarQuantities, CFDHealthReport). HalfCarQuantities uses N, m^2,
        N*m, and Pa-derived defaults from physics_contract.py. CFD health
        fields are residual/count/dimensionless quantities.

    Invalid input behavior:
        Raises CFDRunError if the STL path is missing, the STL is not
        watertight, negative_volume_cells > 0, or OpenFOAM is unavailable/
        errors. Non-convergence (residual_final > 1e-3) does not raise; it
        sets converged=False.
    """
    import openfoam_case

    path = Path(stl_path)
    if not path.exists():
        raise CFDRunError(f"STL path does not exist: {stl_path}")
    _assert_watertight_stl(str(path))

    run_config = openfoam_case.OpenFOAMRunConfig(
        reference_speed_mps=reference_speed_mps,
        air_density_kgm3=air_density_kgm3,
        kinematic_viscosity_m2s=1.813e-5 / air_density_kgm3,
        max_iterations=max_iterations,
        turbulence_model=turbulence_model,
        resolution=resolution,
        moment_reference_point_m=MOMENT_REFERENCE_POINT_M,
    )

    case_dir = Path(__file__).resolve().parent / "cfd_case_template"
    result = _invoke_openfoam_pipeline(str(path), str(case_dir), run_config=run_config)

    negative_volume_cells = int(result["negative_volume_cells"])
    if negative_volume_cells > 0:
        raise CFDRunError("OpenFOAM mesh contains negative-volume cells")

    half = HalfCarQuantities(
        D20=float(result["D20_half"]),
        L=float(result["L_half"]),
        A=float(result["A_half"]),
        pitching_moment_half=float(result["pitching_moment_half"]),
    )
    residual_final = float(result["residual_final"])
    health = CFDHealthReport(
        converged=residual_final <= 1e-3,
        residual_final=residual_final,
        negative_volume_cells=negative_volume_cells,
        y_plus_min=float(result["y_plus_min"]),
        y_plus_max=float(result["y_plus_max"]),
        courant_max=None if result.get("courant_max") is None else float(result["courant_max"]),
    )
    return half, health


def run_half_car_adjoint(
    stl_path: str,
    objective_weight: float,
    reference_speed_mps: float = REFERENCE_SPEED_MPS,
    air_density_kgm3: float = AIR_DENSITY_KGM3,
    resolution: str = "medium",
    primal_iters: int = 1000,
    adjoint_iters: int = 1000,
) -> np.ndarray:
    """Drag-adjoint surface sensitivity for a half-car STL. This is what
    Part 3's `pipeline_interface.real_bindings.run_adjoint` needs and what
    was previously a hard `? UNRESOLVED NotImplementedError` stub there.

    Args:
        stl_path: right-half STL (same contract as run_half_car_cfd).
        objective_weight: w_D20 = dT/dD20 in s/N, from
            adjoint_contract.compute_adjoint_objective_weight. The OpenFOAM
            adjoint solve itself computes the raw (unweighted) dD20_half/
            dSurface; this function applies objective_weight AND
            adjoint_contract.ADJOINT_HALF_CAR_SCALING here, matching the
            Objective = w_D20 x D20 x 0.5 convention in Part 2's Adjoint
            Objective Contract (SPEC.txt Part 2 section) -- see
            openfoam_adjoint.py's module docstring for why the scaling is
            applied in Python rather than baked into the OpenFOAM dict.
        reference_speed_mps, air_density_kgm3: same reference condition as
            the forward CFD run this candidate's D20 came from.
        resolution: snappyHexMesh refinement label.
        primal_iters, adjoint_iters: iteration caps for the primal and
            adjoint solves inside the single adjointOptimisationFoam run.

    Returns:
        np.ndarray of length == len(trimesh.load(stl_path).vertices), one
        weighted sensitivity scalar per vertex, in the SAME order as those
        vertices -- exactly the right_half_sensitivity shape
        phi_updater.apply_adjoint_sensitivity_symmetric (update_phi) requires.

    Invalid input behavior:
        Raises CFDRunError if the STL is missing/not watertight, OpenFOAM is
        unavailable, a solve stage fails, or the sensitivity mesh doesn't
        cover the full STL surface within tolerance -- never a silent
        wrong-shaped or partially-populated array.
    """
    import openfoam_adjoint
    from adjoint_contract import ADJOINT_HALF_CAR_SCALING

    path = Path(stl_path)
    if not path.exists():
        raise CFDRunError(f"STL path does not exist: {stl_path}")
    _assert_watertight_stl(str(path))

    run_config = openfoam_adjoint.AdjointRunConfig(
        reference_speed_mps=reference_speed_mps,
        air_density_kgm3=air_density_kgm3,
        kinematic_viscosity_m2s=1.813e-5 / air_density_kgm3,
        resolution=resolution,
        primal_iters=primal_iters,
        adjoint_iters=adjoint_iters,
    )
    case_dir = Path(__file__).resolve().parent / "cfd_case_template"

    try:
        raw_sensitivity = openfoam_adjoint.invoke_adjoint(str(path), str(case_dir), cfg=run_config)
    except openfoam_adjoint.oc.OpenFOAMNotFoundError as exc:
        raise CFDRunError(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise CFDRunError(
            f"OpenFOAM adjoint stage failed (exit {exc.returncode}): "
            f"{getattr(exc, 'cmd', '?')}. See logs/ in the run directory."
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise CFDRunError(f"Adjoint sensitivity extraction failed: {exc}") from exc

    return raw_sensitivity * float(objective_weight) * ADJOINT_HALF_CAR_SCALING
