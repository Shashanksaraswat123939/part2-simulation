"""Stage 3 half-car CFD wrapper around a future OpenFOAM pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from physics_contract import AIR_DENSITY_KGM3, REFERENCE_SPEED_MPS, HalfCarQuantities


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


def _invoke_openfoam_pipeline(stl_path, case_dir) -> dict:
    raise NotImplementedError(
        "PLACEHOLDER: wire in snappyHexMesh + simpleFoam subprocess "
        "calls here. Must return a dict with keys: "
        "'D20_half', 'L_half', 'A_half', 'pitching_moment_half', "
        "'residual_final', 'negative_volume_cells', "
        "'y_plus_min', 'y_plus_max', 'courant_max'."
    )


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
) -> tuple[HalfCarQuantities, CFDHealthReport]:
    """
    Validate a half-car STL and package OpenFOAM half-domain outputs.

    Args:
        stl_path: filesystem path to an STL file.
        reference_speed_mps: reference speed in m/s.
        air_density_kgm3: air density in kg/m^3.
        max_iterations: solver iteration cap, count.

    Returns:
        (HalfCarQuantities, CFDHealthReport). HalfCarQuantities uses N, m^2,
        N*m, and Pa-derived defaults from physics_contract.py. CFD health
        fields are residual/count/dimensionless quantities.

    Invalid input behavior:
        Raises CFDRunError if the STL path is missing, the STL is not
        watertight, or negative_volume_cells > 0. Non-convergence
        (residual_final > 1e-3) does not raise; it sets converged=False.
    """
    del reference_speed_mps, air_density_kgm3, max_iterations

    path = Path(stl_path)
    if not path.exists():
        raise CFDRunError(f"STL path does not exist: {stl_path}")
    _assert_watertight_stl(str(path))

    case_dir = Path(__file__).resolve().parent / "cfd_case_template"
    result = _invoke_openfoam_pipeline(str(path), str(case_dir))

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
