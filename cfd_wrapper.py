"""Stage 3 half-car CFD wrapper around a future OpenFOAM pipeline."""

from __future__ import annotations

import subprocess
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
            "See logs/ in the run directory (kept on failure)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        # Was uncaught: it reached Part 3's broad `except` as a raw
        # TimeoutExpired and became an opaque "CFD failed". Name it, so a
        # 2-hour wall-clock kill is distinguishable from a solver crash.
        raise CFDRunError(
            f"OpenFOAM stage timed out after {exc.timeout}s: {getattr(exc, 'cmd', '?')}. "
            "Raise OpenFOAMRunConfig.stage_timeout_s, lower max_iterations, or "
            "coarsen `resolution`."
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
    turbulence_model: str = "kOmegaSST",
    resolution: str = "medium",
    n_subdomains: int = 1,
    keep_run_dir: bool = False,
    stage_timeout_s: int = 7200,
    underbody_refinement_level: int = 1,
    extra_surfaces: tuple = (),
    domain_reference_bounds=None,
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
        errors. Non-convergence (residual_final > CONVERGENCE_RESIDUAL, 5e-3) does not raise; it
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
        # These four were unreachable from the pipeline before: run_cfd called
        # with all defaults, so n_subdomains was permanently 1 and every solve
        # ran single-core on a 360 GB machine, run dirs were always deleted,
        # and the timeout was hardcoded at 2 h.
        n_subdomains=n_subdomains,
        keep_run_dir=keep_run_dir,
        stage_timeout_s=stage_timeout_s,
        underbody_refinement_level=underbody_refinement_level,
        moment_reference_point_m=MOMENT_REFERENCE_POINT_M,
        extra_surfaces=tuple(extra_surfaces or ()),
        domain_reference_bounds=domain_reference_bounds,
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
    raw_osc = result.get("force_oscillation")
    force_oscillation = (
        None if raw_osc is None or raw_osc != raw_osc else float(raw_osc)
    )
    # `converged` stays RESIDUAL-ONLY. An earlier version of this ANDed in the
    # force-oscillation check, which was wrong as a gate even though it is right
    # as a diagnosis: Part 3's require_cfd_convergence defaults to True and
    # routes a non-converged solve to CFD_failed, so folding a criterion that
    # every real solve currently fails (10-27% measured) into `converged` would
    # have killed every candidate in a production sweep.
    #
    # The signal is reported instead, and warned about. Whether an unsteady
    # force is fatal is a policy decision for the caller who knows what the
    # number is being used for — ranking candidates needs it small, exercising
    # the pipeline does not.
    def _opt(key):
        v = result.get(key)
        return None if v is None or v != v else float(v)

    force_mean_stderr = _opt("force_mean_stderr")
    force_drift = _opt("force_drift")

    # Warn on the statistic that answers the question actually being asked --
    # "is this drag delta measurable" -- which is the error on the MEAN, not
    # the peak-to-peak of the signal. Peak-to-peak is max-minus-min: one
    # outlier sets it and it does not shrink with a longer window, so an 8.9%
    # swing says nothing about whether the mean is good to 0.5% or 5%.
    if force_mean_stderr is not None and force_mean_stderr > MAX_FORCE_MEAN_STDERR:
        drift_txt = (
            f" and still drifting {force_drift*100:.1f}% across the window"
            if force_drift is not None and force_drift > force_mean_stderr
            else ""
        )
        advice = (
            "the mean has not settled, so a longer averaging window cannot fix "
            "it -- run more iterations"
            if force_drift is not None and force_drift > force_mean_stderr
            else "average over more iterations, or accept this as the noise floor"
        )
        warnings.warn(
            f"D20 is a mean whose standard error is "
            f"{force_mean_stderr*100:.2f}% of itself (limit "
            f"{MAX_FORCE_MEAN_STDERR*100:.1f}%){drift_txt}. Drag deltas below "
            f"that are not measurable, so candidates closer than it are being "
            f"ranked by noise. {advice[0].upper()}{advice[1:]}. Residual "
            f"convergence does not cover this — a steady solver on an unsteady "
            f"wake plateaus its residuals while the forces keep moving.",
            RuntimeWarning, stacklevel=2,
        )
    elif (force_oscillation is not None
            and force_oscillation > MAX_FORCE_OSCILLATION
            and force_mean_stderr is None):
        # No force history detail available; fall back to the coarse signal.
        warnings.warn(
            f"streamwise force is swinging {force_oscillation*100:.1f}% "
            f"peak-to-peak over the averaged window (limit "
            f"{MAX_FORCE_OSCILLATION*100:.0f}%), and the per-sample history "
            f"needed to compute the error on the mean was not available.",
            RuntimeWarning, stacklevel=2,
        )
    # DRIFT GATE (2026-09-25). Measured on GitHub Actions: a medium solve with
    # a final residual of 1.7e-3 (passes the 5e-3 residual gate) was still
    # drifting 9.2 % across its averaging window, and its D20 was 4.7 % off the
    # neighbouring solves. Drift is what makes a single number unrepeatable, so
    # it gates. Healthy solves measured 0.0-1.3 % drift.
    drift_ok = force_drift is None or force_drift <= MAX_FORCE_DRIFT
    health = CFDHealthReport(
        converged=residual_final <= CONVERGENCE_RESIDUAL and drift_ok,
        residual_final=residual_final,
        negative_volume_cells=negative_volume_cells,
        y_plus_min=float(result["y_plus_min"]),
        y_plus_max=float(result["y_plus_max"]),
        courant_max=None if result.get("courant_max") is None else float(result["courant_max"]),
        force_oscillation=force_oscillation,
        force_mean_stderr=force_mean_stderr,
        force_drift=force_drift,
        patch_forces=result.get("groups"),
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
    keep_run_dir: bool = False,
    stage_timeout_s: int = 14400,
    underbody_refinement_level: int = 1,
    n_subdomains: int = 1,
    extra_surfaces: tuple = (),
    domain_reference_bounds=None,
    # None means "whatever AdjointRunConfig says", rather than restating it.
    #
    # This signature duplicated every one of the config's defaults and exactly
    # one drifted: max_unmapped_fraction stayed 0.05 here when the config moved
    # to 0.30, so the production adjoint still died at "20.8% ... above the 5%
    # limit" while the config said 30%. A duplicated default is a second source
    # of truth that nothing keeps in step. openfoam_adjoint is imported inside
    # the function body, so the signature cannot name the config directly --
    # hence the sentinel. test_wrapper_defaults_match_the_config compares all of
    # them so the next drift fails a test instead of a run.
    max_unmapped_fraction: Optional[float] = None,
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
        keep_run_dir=keep_run_dir,
        stage_timeout_s=stage_timeout_s,
        underbody_refinement_level=underbody_refinement_level,
        # Omit when None so the dataclass supplies its own default -- one source
        # of truth for this value.
        **({} if max_unmapped_fraction is None
           else {"max_unmapped_fraction": max_unmapped_fraction}),
        n_subdomains=n_subdomains,
        extra_surfaces=tuple(extra_surfaces or ()),
        domain_reference_bounds=domain_reference_bounds,
    )
    case_dir = Path(__file__).resolve().parent / "cfd_case_template"

    try:
        try:
            raw_sensitivity = openfoam_adjoint.invoke_adjoint(str(path), str(case_dir), cfg=run_config)
        except RuntimeError as exc:
            # The fallback the ATC note in openfoam_adjoint names: `cancel`
            # drops adjoint transpose convection, a small inconsistency against
            # a gradient that is otherwise garbage.
            if "DIVERGED" not in str(exc) or run_config.atc_model == "cancel":
                raise
            print(f"[adjoint] {str(exc)[:80]}... retrying with ATCModel cancel", flush=True)
            from dataclasses import replace as _replace
            run_config = _replace(run_config, atc_model="cancel")
            raw_sensitivity = openfoam_adjoint.invoke_adjoint(str(path), str(case_dir), cfg=run_config)
    except openfoam_adjoint.oc.OpenFOAMNotFoundError as exc:
        raise CFDRunError(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise CFDRunError(
            f"OpenFOAM adjoint stage failed (exit {exc.returncode}): "
            f"{getattr(exc, 'cmd', '?')}. See logs/ in the run directory (kept on failure)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CFDRunError(
            f"OpenFOAM adjoint stage timed out after {exc.timeout}s: "
            f"{getattr(exc, 'cmd', '?')}. Raise AdjointRunConfig.stage_timeout_s "
            "or lower primal_iters/adjoint_iters."
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise CFDRunError(f"Adjoint sensitivity extraction failed: {exc}") from exc

    # The adjoint objective is now the drag FORCE, not a coefficient:
    # build_optimisation_dict sets Aref = 2/UInf^2 so ESI's
    # denom() = 0.5*UInf^2*Aref collapses to exactly 1. See the long note there
    # for why a coefficient was the wrong thing to minimise.
    #
    # One conversion remains. objectiveForce reads rhoInf and never uses it
    # (objectiveForce.C:72 is its only appearance), and both p and devReff are
    # kinematic in the incompressible solver, so `force` is kinematic and
    # reaching newtons takes a factor of rho.
    #
    #   dD20_half/dSurface = rho * (kinematic dF/dSurface)
    #
    # ADJOINT_HALF_CAR_SCALING (2.0) then converts half-car to full car.
    return (raw_sensitivity * float(objective_weight)
            * air_density_kgm3 * ADJOINT_HALF_CAR_SCALING)
