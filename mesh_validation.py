"""Stage 4 mesh independence and solver validation harness."""

from __future__ import annotations

from dataclasses import dataclass

from cfd_wrapper import run_half_car_cfd
from physics_contract import AIR_DENSITY_KGM3

# Sentinel for detecting when cfd_runner was not supplied.
# Using this instead of defaulting to run_half_car_cfd prevents the silent
# false-positive mesh independence study that occurs when the default runner
# ignores the resolution label.
_REQUIRED_RUNNER = object()


@dataclass(frozen=True)
class MeshIndependenceResult:
    """Mesh independence result.

    D20 and L are in N, Cm is dimensionless, and spreads are dimensionless.
    Invalid input behavior: invalid CFD runner behavior propagates from the
    injected cfd_runner.
    """

    resolutions_tested: tuple[str, ...]
    D20_values: tuple[float, ...]
    L_values: tuple[float, ...]
    Cm_values: tuple[float, ...]
    max_relative_spread_D20: float
    max_relative_spread_L: float
    max_relative_spread_Cm: float
    passes_5_percent_target: bool


def _relative_spread(values: tuple[float, ...]) -> float:
    spread = max(values) - min(values)
    scale = max(abs(v) for v in values)
    if scale < 1e-9:
        # All values genuinely near zero (e.g. a symmetric car with no
        # net lift) -- spread is physically negligible regardless of sign,
        # this is the one case where 0.0 is actually correct.
        return 0.0
    return spread / scale


def _full_quantities_from_runner_result(result):
    half, _health = result
    return half.to_full_car()


def run_mesh_independence_study(
    stl_path: str,
    *,
    cfd_runner=_REQUIRED_RUNNER,
    resolutions: tuple[str, ...] = ("coarse", "medium", "fine"),
) -> MeshIndependenceResult:
    """
    Calls cfd_runner once per resolution label.

    Args:
        stl_path: STL path string.
        cfd_runner: callable accepting (stl_path, resolution_label) and
            returning (HalfCarQuantities, CFDHealthReport). This is a REQUIRED
            keyword-only argument. The previous default (run_half_car_cfd)
            silently ignored the resolution label and produced false-positive
            mesh independence results. Callers must explicitly supply a runner
            that actually varies mesh resolution.
        resolutions: opaque resolution labels passed through to cfd_runner.

    Returns:
        MeshIndependenceResult with D20/L in N, Cm dimensionless, and relative
        spreads dimensionless.

    Invalid input behavior:
        Raises TypeError if cfd_runner is not supplied (preventing silent
        false-positive mesh independence). Exceptions from cfd_runner propagate;
        this function does not clamp or warn.
    """
    if cfd_runner is _REQUIRED_RUNNER:
        raise TypeError(
            "run_mesh_independence_study requires a cfd_runner keyword argument. "
            "The default was removed because run_half_car_cfd ignores the "
            "resolution label, producing false-positive mesh independence. "
            "Supply a runner that actually varies mesh resolution."
        )
    full_values = [
        _full_quantities_from_runner_result(cfd_runner(stl_path, resolution))
        for resolution in resolutions
    ]
    D20_values = tuple(value.D20 for value in full_values)
    L_values = tuple(value.L for value in full_values)
    Cm_values = tuple(value.Cm for value in full_values)
    spread_D20 = _relative_spread(D20_values)
    spread_L = _relative_spread(L_values)
    spread_Cm = _relative_spread(Cm_values)
    return MeshIndependenceResult(
        resolutions_tested=resolutions,
        D20_values=D20_values,
        L_values=L_values,
        Cm_values=Cm_values,
        max_relative_spread_D20=spread_D20,
        max_relative_spread_L=spread_L,
        max_relative_spread_Cm=spread_Cm,
        passes_5_percent_target=spread_D20 <= 0.05 and spread_L <= 0.05 and spread_Cm <= 0.05,
    )


@dataclass(frozen=True)
class SolverComparisonResult:
    """Laminar versus k-omega SST solver comparison.

    D20 and L are in N, Cm and relative deltas are dimensionless.
    Invalid input behavior: raises NotImplementedError when
    komega_sst_runner is None; runner exceptions otherwise propagate.
    """

    D20_laminar: float
    D20_komega_sst: float
    L_laminar: float
    L_komega_sst: float
    Cm_laminar: float
    Cm_komega_sst: float
    relative_delta_D20: float
    relative_delta_L: float
    relative_delta_Cm: float


def run_laminar_vs_komega_sst_comparison(
    stl_path: str,
    laminar_runner=run_half_car_cfd,
    komega_sst_runner=None,
) -> SolverComparisonResult:
    """Runs both solvers on the same STL, reports relative deltas.

    Args:
        stl_path: STL path string.
        laminar_runner: callable returning laminar half-car CFD output.
        komega_sst_runner: callable returning k-omega SST half-car CFD output.

    Returns:
        SolverComparisonResult with D20/L in N, Cm dimensionless, and relative
        deltas dimensionless.

    Invalid input behavior:
        Raises NotImplementedError if komega_sst_runner is None; runner
        exceptions otherwise propagate.
    """
    if komega_sst_runner is None:
        raise NotImplementedError("PLACEHOLDER: k-omega SST runner not wired in yet")
    laminar = _full_quantities_from_runner_result(laminar_runner(stl_path))
    komega = _full_quantities_from_runner_result(komega_sst_runner(stl_path))
    return SolverComparisonResult(
        D20_laminar=laminar.D20,
        D20_komega_sst=komega.D20,
        L_laminar=laminar.L,
        L_komega_sst=komega.L,
        Cm_laminar=laminar.Cm,
        Cm_komega_sst=komega.Cm,
        relative_delta_D20=abs(komega.D20 - laminar.D20) / laminar.D20,
        relative_delta_L=abs(komega.L - laminar.L) / laminar.L,
        relative_delta_Cm=abs(komega.Cm - laminar.Cm) / laminar.Cm,
    )


@dataclass(frozen=True)
class SpeedSensitivityResult:
    """Speed sensitivity result.

    D20 values are in N, CdA values are in m^2, and relative_delta_CdA is
    dimensionless. Invalid input behavior: runner exceptions propagate.
    """

    D20_at_5mps: float
    D20_at_20mps: float
    CdA_at_5mps: float
    CdA_at_20mps: float
    relative_delta_CdA: float


def run_speed_sensitivity_check(
    stl_path: str,
    cfd_runner=run_half_car_cfd,
) -> SpeedSensitivityResult:
    """
    Runs cfd_runner at 5.0 m/s and 20.0 m/s and back-solves CdA.

    Args:
        stl_path: STL path string.
        cfd_runner: callable accepting reference_speed_mps and returning
            (HalfCarQuantities, CFDHealthReport).

    Returns:
        SpeedSensitivityResult with D20 in N, CdA in m^2, relative delta
        dimensionless.

    Invalid input behavior:
        Exceptions from cfd_runner propagate; this function does not clamp or
        warn.
    """
    half_5, _health_5 = cfd_runner(stl_path, reference_speed_mps=5.0)
    half_20, _health_20 = cfd_runner(stl_path, reference_speed_mps=20.0)
    cda_5 = half_5.D20 / (0.5 * AIR_DENSITY_KGM3 * 5.0 ** 2)
    cda_20 = half_20.D20 / (0.5 * AIR_DENSITY_KGM3 * 20.0 ** 2)
    return SpeedSensitivityResult(
        D20_at_5mps=half_5.D20,
        D20_at_20mps=half_20.D20,
        CdA_at_5mps=cda_5,
        CdA_at_20mps=cda_20,
        relative_delta_CdA=abs(cda_20 - cda_5) / cda_5,
    )
