"""Adapter from the locked race_objective.py interface to Stage 7 keys.

This adapter bridges the locked race_objective.py (which must not be modified)
to the Stage 7 adjoint contract layer. It enforces the Part 2 spec contract:

    Returns:
        T_raw       = predicted 20 m race time (physics only, no penalties)
        T_penalized = T_raw + COM_penalty + manufacturing_penalties
        gradients  = dict of dT/d(param) for each optimizable input

The locked race_objective.py bakes the COM penalty into race_time_seconds,
so its return value is actually T_penalized. This adapter extracts T_raw by
subtracting the COM penalty component, exposing both values per the spec.
"""

from __future__ import annotations

from mass_com_ingest import COM_SANITY_BOUNDS_M
from race_objective import (
    PARAM_NAMES,
    _COM_PENALTY_DATA_MM,
    _clean_csv_arrays,
    BuildSettings,
    COM_TARGET_HEIGHT_M,
    build_smooth_sheet_model,
    com_height_time_penalty,
    com_x_time_penalty,
    race_value_and_grad,
)

# Derived from the locked race_objective.py's _COM_PENALTY_DATA_MM domain.
# The COM penalty polynomial was fitted over delta_mm in [-12, +12],
# i.e. com_height_m in [COM_TARGET_HEIGHT_M - 0.012, COM_TARGET_HEIGHT_M + 0.012].
# Evaluating outside this range produces meaningless extrapolation.
_COM_PENALTY_HALF_RANGE_M = max(abs(v) for v in _COM_PENALTY_DATA_MM) / 1000.0
COM_HEIGHT_FIT_RANGE_M = (
    COM_TARGET_HEIGHT_M - _COM_PENALTY_HALF_RANGE_M,
    COM_TARGET_HEIGHT_M + _COM_PENALTY_HALF_RANGE_M,
)


# ⚠ UNRESOLVED: race_objective.py (locked, verbatim) treats
# time_coefficient as a live optimizer input. physics_contract.py
# defines TIME_COEFFICIENT = 1.0 as a frozen, non-optimizable constant.
# These two are in direct conflict. This adapter does NOT resolve the
# conflict. It only enforces that, until a human explicitly decides,
# every call through this adapter uses exactly 1.0 for
# time_coefficient, matching physics_contract.py's frozen value --
# NOT because this is confirmed correct, but because it is the safer
# default until a human resolves the conflict explicitly.
def _assert_time_coefficient_unity(param_vector) -> None:
    assert param_vector[PARAM_NAMES.index("time_coefficient")] == 1.0, (
        "time_coefficient must be 1.0 through this adapter until the human "
        "explicitly resolves the conflict between race_objective.py "
        "(locked) treating it as a variable and physics_contract.py "
        "freezing it at 1.0 -- see the UNRESOLVED CONFLICT note in the "
        "build spec."
    )


def _assert_physical_inputs(param_vector) -> None:
    """Guard against physically meaningless inputs that would produce
    silently wrong results (e.g. negative mass, negative mu). The locked
    race_objective.py uses smooth_positive to avoid crashes, but this
    masks garbage inputs. This guard catches them early with a clear error."""
    p = {name: float(param_vector[i]) for i, name in enumerate(PARAM_NAMES)}
    if p["car_weight_kg"] <= 0:
        raise ValueError(f"car_weight_kg must be positive, got {p['car_weight_kg']}")
    if p["mu"] < 0:
        raise ValueError(f"mu (friction) must be non-negative, got {p['mu']}")
    if p["drag_20_n"] < 0:
        raise ValueError(f"drag_20_n must be non-negative, got {p['drag_20_n']}")
    if p["wheel_moi_kg_m2"] < 0:
        raise ValueError(f"wheel_moi_kg_m2 must be non-negative, got {p['wheel_moi_kg_m2']}")
    # lift_20_n: allow any value (both downforce and upforce are physically valid)
    # COM height extrapolation guard: the locked polynomial was fitted only
    # over COM_HEIGHT_FIT_RANGE_M. Outside that range, the degree-4 polynomial
    # produces meaningless values (up to 10^14 seconds penalty).
    _tol = 1e-9  # small float tolerance for boundary comparisons
    if not (COM_HEIGHT_FIT_RANGE_M[0] - _tol <= p["com_height_m"] <= COM_HEIGHT_FIT_RANGE_M[1] + _tol):
        raise ValueError(
            f"com_height_m={p['com_height_m']} is outside the COM penalty "
            f"polynomial's fitted range {COM_HEIGHT_FIT_RANGE_M} -- "
            f"extrapolation would produce a meaningless penalty value. "
            f"Check upstream mass/COM ingestion for a units or origin bug."
        )
    # COM x sanity bounds (COM should be within a reasonable range of the car)
    if not (COM_SANITY_BOUNDS_M[0] <= p["com_x_m"] <= COM_SANITY_BOUNDS_M[1]):
        raise ValueError(
            f"com_x_m={p['com_x_m']} outside sanity bounds {COM_SANITY_BOUNDS_M}, "
            f"likely a units or origin bug"
        )


def adapt_gradients(raw_grads: dict) -> dict:
    """
    raw_grads is the dict returned by race_value_and_grad(...)[1], keyed by
    PARAM_NAMES ('drag_20_n', 'car_weight_kg', 'mu', 'wheel_moi_kg_m2',
    'time_coefficient', 'com_height_m', 'lift_20_n', 'com_x_m').

    Returns a dict with keys 'dT_dD20', 'dT_dmass', 'dT_dh_com',
    'dT_dx_com', 'dT_dL'. Values are seconds per input unit: s/N for dT_dD20,
    s/kg for dT_dmass, s/m for dT_dh_com, s/m for dT_dx_com, s/N for dT_dL.

    Invalid input behavior:
        Raises KeyError if raw_grads is missing any required locked-file key.
    """
    return {
        "dT_dD20": raw_grads["drag_20_n"],
        "dT_dmass": raw_grads["car_weight_kg"],
        "dT_dh_com": raw_grads["com_height_m"],
        "dT_dx_com": raw_grads["com_x_m"],
        "dT_dL": raw_grads["lift_20_n"],
    }


def race_value_and_grad_guarded(param_vector, model):
    """Call locked race_value_and_grad with the time_coefficient guard.

    Implements the Part 2 spec Race Objective Contract:

        Returns:
            T_raw       = predicted 20 m race time (physics dynamics only)
            T_penalized = T_raw + COM_penalty (+ manufacturing_penalties
                          added by the optimizer, not this adapter)
            gradients   = adapted gradient dict with keys dT_dD20, dT_dmass,
                          dT_dh_com, dT_dx_com, dT_dL

    The locked race_objective.py's race_time_seconds() returns a value that
    includes the COM penalty baked in (via com_height_time_penalty and
    com_x_time_penalty). This adapter splits that out:
        T_penalized = locked race_time_seconds output
        T_raw       = T_penalized - com_height_penalty - com_x_penalty

    Gradients from the locked file are wrt T_penalized (the full objective).
    This is correct for the adjoint contract: the optimizer needs dT_penalized
    /d(param) to minimize T_penalized, and ultimately T_raw among valid
    candidates.

    Args:
        param_vector: locked-file parameter vector; units follow PARAM_NAMES.
        model: locked-file SmoothSheetModel.

    Returns:
        (T_raw, T_penalized, adapted gradient dict).

    Invalid input behavior:
        Raises AssertionError unless time_coefficient is exactly 1.0. Other
        validation errors propagate from the locked race_objective.py file.
    """
    import jax.numpy as jnp
    _assert_time_coefficient_unity(param_vector)
    _assert_physical_inputs(param_vector)
    T_penalized, raw_grads = race_value_and_grad(param_vector, model)
    params_jax = jnp.asarray(param_vector, dtype=jnp.float64)
    com_h_penalty = float(com_height_time_penalty(params_jax))
    com_x_pen = float(com_x_time_penalty(params_jax))
    # Guard: the fitted COM penalty polynomial goes slightly negative near
    # delta≈0.61mm (a polyfit artifact). Clamp to >= 0 so T_raw <= T_penalized
    # always holds. The magnitude is ~29 microseconds but must not invert the
    # T_raw/T_penalized relationship.
    com_h_penalty = max(com_h_penalty, 0.0)
    com_x_pen = max(com_x_pen, 0.0)
    # NOTE: T_raw = T_penalized - com_penalty is only correct because the
    # adapter guards time_coefficient == 1.0. The locked file computes
    # T_penalized = tc * (t_finish + low_speed_penalty + com_h_penalty + com_x_pen),
    # so the correct general formula would be:
    #   T_raw = T_penalized - tc * (com_h_penalty + com_x_pen)
    # If the time_coefficient guard is ever removed, this MUST be updated.
    tc = float(param_vector[PARAM_NAMES.index("time_coefficient")])
    T_raw = T_penalized - tc * (com_h_penalty + com_x_pen)
    return T_raw, T_penalized, adapt_gradients(raw_grads)


def build_smooth_sheet_model_guarded(csv_path: str, settings) -> object:
    """Guarded wrapper around locked build_smooth_sheet_model.

    Validates BuildSettings before calling the locked function so that
    invalid n_basis or n_steps are caught early with a clear error message,
    rather than failing deep inside the RBF interpolation with a cryptic error.

    Args:
        csv_path: path to thrust CSV file.
        settings: race_objective.BuildSettings instance.

    Returns:
        Locked SmoothSheetModel instance.

    Raises:
        ValueError if n_basis < 1 or n_steps < 1.
    """
    from race_objective import build_smooth_sheet_model
    if settings.n_basis < 1:
        raise ValueError(
            f"BuildSettings.n_basis must be >= 1, got {settings.n_basis}"
        )
    if settings.n_steps < 1:
        raise ValueError(
            f"BuildSettings.n_steps must be >= 1, got {settings.n_steps}"
        )
    return build_smooth_sheet_model(csv_path, settings)


def validate_thrust_csv_physical_sanity(csv_path: str) -> None:
    """Check thrust CSV for physically impossible data.

    Calls race_objective._clean_csv_arrays to parse the CSV, then checks:
      - No negative time values
      - No negative force values
      - No non-positive mass values

    This catches corrupt test data that _clean_csv_arrays would accept
    (it validates formatting but not physical sanity), preventing
    meaningless objective values from being computed.

    Args:
        csv_path: path to thrust CSV file.

    Raises:
        ValueError if any physical sanity check fails.
        FileNotFoundError if csv_path doesn't exist.
    """
    from race_objective import _clean_csv_arrays
    t, force, mass = _clean_csv_arrays(csv_path)
    if any(ti < 0 for ti in t):
        raise ValueError(
            f"Thrust CSV {csv_path} contains negative time values"
        )
    if any(fi < 0 for fi in force):
        raise ValueError(
            f"Thrust CSV {csv_path} contains negative force values"
        )
    if any(mi <= 0 for mi in mass):
        raise ValueError(
            f"Thrust CSV {csv_path} contains non-positive mass values"
        )
