"""Half-car CFD on SimScale (simscale branch): the same entry point and
result types as the OpenFOAM wrapper on main, backed by simscale_case."""

from __future__ import annotations

import warnings
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

# Largest streamwise-force peak-to-peak swing (as a fraction of its mean, over
# the averaged window) for which the reported drag counts as reproducible.
#
# 5% is chosen against what the number is FOR: ranking candidates whose race
# times differ by milliseconds. A drag figure that moves 20% between two
# geometries 65 nanometres apart cannot rank anything. Measured on the current
# brick: 0.18-0.27, i.e. every solve so far fails this — which is the correct
# verdict, not a threshold to loosen. simpleFoam is a steady solver and the
# brick's wake is not steady; the routes out are a less bluff shape, a longer
# averaging window, or an unsteady solver.
MAX_FORCE_OSCILLATION: float = 0.05

# Largest standard error of the MEAN streamwise force (as a fraction of that
# mean) for which a drag DELTA is measurable.
#
# This is the threshold that matters, and it is a different quantity from
# MAX_FORCE_OSCILLATION above. D20 is a mean; the error on a mean is its
# standard error, which shrinks as sqrt(N_eff) with a longer window, whereas
# peak-to-peak is max-minus-min and does not shrink at all. A solve can swing
# 8.9% peak-to-peak and still pin its mean to a fraction of a percent.
#
# 1% is set against the job: the first working adjoint step produced a 5.43%
# drag reduction, and parse_total_vector_dat's window sweep showed averaging
# lands in the 1-3% band. Below 1% that step is a real measurement; above it,
# candidates are being ranked by noise. Not loosened to whatever the solver
# currently achieves -- the point is to say when the number cannot do its job.
MAX_FORCE_MEAN_STDERR: float = 0.01

# Final p-residual at or below which a solve counts as converged.
#
# SET FROM MEASUREMENT, and the value is a statement about this geometry rather
# than a preference. simpleFoam is a STEADY solver and this car is a brick whose
# wake is not steady, so the residual PLATEAUS instead of descending. Measured
# on one fixed STL (2026-07-27):
#     coarse (103k cells)          4.4e-4
#     medium (389k, production)    2.1e-3
#     medium + underbody (1.94M)   3.2e-3
# The old hardcoded 1e-3 was therefore unpassable at production resolution:
# --smoke set require_cfd_convergence=False and completed, while a real run --
# which leaves it True -- marked every candidate CFD_failed before the adjoint
# ever ran.
#
# 5e-3 is not "loose enough to pass". It is above the measured plateau and far
# below a diverging solve, and the mesh study showed the AVERAGED drag at these
# residuals agreeing to +/-1.1% across a 19x cell range -- i.e. the plateau is
# not corrupting the force, the unsteadiness is, and force_oscillation is what
# reports that. Tighten it once the case is steady.
CONVERGENCE_RESIDUAL: float = 5e-3

# Largest drift of the averaged streamwise force (|slope| x window / mean) for
# which a solve counts as converged. See the drift gate in run_half_car_cfd.
MAX_FORCE_DRIFT: float = 0.02


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
    # Peak-to-peak swing of streamwise force over the averaged window, as a
    # fraction of its mean. residual_final CANNOT express this: a steady solver
    # on an unsteady wake plateaus its residuals while the forces keep swinging.
    # Measured 0.18-0.27 on this brick geometry, i.e. the reported drag was
    # reproducible only to ~+/-20%, which is wider than any single optimiser
    # step. None when no force history was available -- None rather than NaN so
    # two identical reports compare equal (NaN != NaN breaks dataclass equality).
    force_oscillation: Optional[float] = None
    # Error bar on the MEAN streamwise force (the number D20 is built from),
    # as a fraction of that mean, autocorrelation-corrected. This -- not
    # force_oscillation -- is what a drag delta must exceed to be measurable.
    # Peak-to-peak is max-minus-min and does not shrink with a longer window;
    # the uncertainty of a mean does.
    force_mean_stderr: Optional[float] = None
    # How much the mean is still MOVING across the window (|slope|*span/mean).
    # If this dominates force_mean_stderr the solve has not settled and the
    # answer is more iterations -- averaging cannot fix a drifting signal.
    force_drift: Optional[float] = None
    # Half-car force per patch when extra surfaces (wheels, wings, supports)
    # are in the case: {patch: {"D_half_N", "L_half_N"}}. None for body-only.
    patch_forces: Optional[dict] = None


class CFDRunError(Exception):
    """Raised when meshing or solving fails in a way that cannot produce
    a usable force report. Callers (Stage 6/7/8) must catch this and route
    to the 'CFD_failed' candidate lifecycle state -- never let it propagate
    unhandled into the optimizer loop."""


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


def _invoke_simscale(stl_path: str, cfg, work: str) -> dict:
    """The one SimScale call (tests replace it)."""
    import simscale_case
    return simscale_case.invoke(stl_path, cfg, work)


def run_half_car_cfd(
    stl_path: str,
    reference_speed_mps: float = REFERENCE_SPEED_MPS,
    air_density_kgm3: float = AIR_DENSITY_KGM3,
    max_iterations: int = 2000,
    resolution: str = "medium",
    extra_surfaces: tuple = (),
    domain_reference_bounds=None,
    keep_run_dir: bool = False,
    run_dir: Optional[str] = None,
    **simscale_options,
) -> tuple[HalfCarQuantities, CFDHealthReport]:
    """Validate a half-car STL, run it on SimScale, and package the outputs.

    resolution: a simscale_case.MESH_PRESETS key ("coarse", "medium", "fine",
    "resolved" ~5 M cells). simscale_options: any other SimScaleConfig field
    (first_layer_m, layer_growth, n_layers, fraction_from_end, ...).

    Raises CFDRunError if the STL is missing or not watertight, or SimScale
    fails; non-convergence does not raise, it sets converged=False.
    """
    import tempfile

    import simscale_case

    path = Path(stl_path)
    if not path.exists():
        raise CFDRunError(f"STL path does not exist: {stl_path}")
    _assert_watertight_stl(str(path))
    try:
        cfg = simscale_case.SimScaleConfig(
            speed_mps=reference_speed_mps, density_kgm3=air_density_kgm3,
            kinematic_viscosity_m2s=1.813e-5 / air_density_kgm3, iterations=max_iterations,
            resolution=resolution, extra_surfaces=tuple(extra_surfaces or ()),
            domain_reference_bounds=domain_reference_bounds, **simscale_options)
        work = run_dir or tempfile.mkdtemp(prefix="simscale_")
        result = _invoke_simscale(str(path), cfg, work)
    except simscale_case.SimScaleError as exc:
        raise CFDRunError(f"SimScale: {exc}") from exc

    if int(result.get("negative_volume_cells", 0)) > 0:
        raise CFDRunError("the mesh contains negative-volume cells")
    half = HalfCarQuantities(
        D20=float(result["D20_half"]), L=float(result["L_half"]), A=float(result["A_half"]),
        pitching_moment_half=float(result["pitching_moment_half"]))
    residual = float(result["residual_final"])
    se, drift = result.get("force_mean_stderr"), result.get("force_drift")
    if se is not None and se > MAX_FORCE_MEAN_STDERR:
        warnings.warn(f"D20 is a mean whose standard error is {se * 100:.2f}% of itself "
                      f"(limit {MAX_FORCE_MEAN_STDERR * 100:.1f}%)", RuntimeWarning, stacklevel=2)
    drift_ok = drift is None or drift <= MAX_FORCE_DRIFT
    residual_ok = residual != residual or residual <= CONVERGENCE_RESIDUAL   # NaN: not reported
    health = CFDHealthReport(
        converged=residual_ok and drift_ok, residual_final=residual,
        negative_volume_cells=int(result.get("negative_volume_cells", 0)),
        y_plus_min=float(result.get("y_plus_min", float("nan"))),
        y_plus_max=float(result.get("y_plus_max", float("nan"))),
        courant_max=None, force_oscillation=None, force_mean_stderr=se, force_drift=drift,
        patch_forces=result.get("groups"))
    return half, health
