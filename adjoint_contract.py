"""Stage 7 adjoint objective contract."""

from __future__ import annotations

from typing import Optional

from race_objective_adapter import race_value_and_grad_guarded

# Half-car scaling factor for the adjoint objective.
# The forward CFD runs on a half-car mesh (symmetry plane), and D20 in the
# param_vector is the full-car value (post to_full_car() doubling). The
# adjoint objective differentiates w_D20 × D20 w.r.t. the half-car surface,
# so we scale D20 by 0.5 to get the half-car contribution, preventing a 2x
# error in surface sensitivities once real OpenFOAM adjoint is wired in.
ADJOINT_HALF_CAR_SCALING = 0.5


def compute_adjoint_objective_weight(
    param_vector,
    model,
) -> float:
    """Returns w_D20 = dT/dD20 from the locked race objective.

    This is the sensitivity of the penalized race time to drag force. It is
    used as the weight in the adjoint objective: Objective = w_D20 × D20.

    Args:
        param_vector: locked race_objective.py parameter vector with units
            defined by PARAM_NAMES: drag_20_n in N, car_weight_kg in kg, mu
            dimensionless, wheel_moi_kg_m2 in kg*m^2, time_coefficient
            dimensionless and guarded to 1.0, com_height_m in m.
        model: locked race_objective.py SmoothSheetModel fitted from thrust CSV.

    Returns:
        w_D20 in seconds per N.

    Invalid input behavior:
        Raises AssertionError through race_objective_adapter if
        time_coefficient is not exactly 1.0. Other validation/JAX errors
        propagate from the locked race objective.
    """
    _T_raw, _T_penalized, gradients = race_value_and_grad_guarded(param_vector, model)
    return gradients["dT_dD20"]


# ⚠ UNRESOLVED: The Part 2 spec does not state whether the OpenFOAM adjoint
# case runs on the half-car mesh (matching the forward CFD run per the
# Half-Car CFD Contract) or a mirrored full-car mesh. This function
# currently uses full-car D20 (post to_full_car() doubling) with no scaling.
# If the adjoint case is half-domain, this introduces a 2x error that must
# be corrected either here or in Part 1's geometry designer before the
# surface sensitivity is applied. This must be resolved by a human before
# Stage 6 of the Part 3 build order (Adjoint integration) is implemented.

def compute_adjoint_objective(
    param_vector,
    model,
) -> float:
    """Returns the adjoint objective value: w_D20 × D20.

    Per the Part 2 Adjoint Objective Contract:
        Objective = w_D20 × D20

    This is the quantity that OpenFOAM adjoint differentiates to get
    surface sensitivities dObjective/dSurface.

    ⚠ Half-car scaling applied: D20 is multiplied by ADJOINT_HALF_CAR_SCALING
    (0.5) before computing the objective, because the OpenFOAM adjoint will
    run on the half-car mesh per the Half-Car CFD Contract. See the comment
    at ADJOINT_HALF_CAR_SCALING above.

    Args:
        param_vector: locked race_objective.py parameter vector.
        model: locked race_objective.py SmoothSheetModel.

    Returns:
        Objective value in N×s/N = seconds (w_D20 × D20 × 0.5).
    """
    w_D20 = compute_adjoint_objective_weight(param_vector, model)
    D20 = float(param_vector[0])  # drag_20_n is index 0 in PARAM_NAMES
    return w_D20 * D20 * ADJOINT_HALF_CAR_SCALING


def package_gradient_bundle(
    w_D20: float,
    dT_dmass: float,
    dT_dh_com: float,
    dT_dx_com: float,
    manufacturing_gradient: Optional[object] = None,
) -> dict:
    """
    Returns a dict with EXACTLY these keys, always present even when a
    value is a placeholder zero or None:
      'w_D20', 'dT_dmass', 'dT_dh_com', 'dT_dx_com', 'manufacturing_gradient'

    Units:
        w_D20 is s/N, dT_dmass is s/kg, dT_dh_com and dT_dx_com are s/m.
        manufacturing_gradient units are unspecified by Part 1.

    Invalid input behavior:
        Does not validate or clamp. manufacturing_gradient defaults to None
        and is stored explicitly, not replaced with a made-up value.
    """
    return {
        "w_D20": w_D20,
        "dT_dmass": dT_dmass,
        "dT_dh_com": dT_dh_com,
        "dT_dx_com": dT_dx_com,
        "manufacturing_gradient": manufacturing_gradient,
    }


CM_EXPANSION_PREREQUISITES = {
    "mu_fitted_from_track_data": False,         # PLACEHOLDER flags --
    "ballast_com_experiment_completed": False,  # caller must set these True
    "cm_and_l_wired_into_rtc": False,           # only once real data exists
}


def get_active_objective_weights(prerequisites: dict = CM_EXPANSION_PREREQUISITES) -> dict:
    """
    Returns {'w_D20': 1.0, 'w_L': 0.0, 'w_Cm': 0.0} if not all three
    prerequisite flags in `prerequisites` are True.

    Units:
        All returned weights are dimensionless multipliers for objective
        composition at this contract layer.

    Invalid input behavior:
        Raises NotImplementedError when all three prerequisite flags are True
        because real w_L/w_Cm sensitivity values have not been supplied.
    """
    if all(prerequisites.values()):
        raise NotImplementedError(
            "PLACEHOLDER: prerequisites met but w_L/w_Cm sensitivity "
            "analysis values have not been supplied -- see Gradient "
            "Combination section of the optimizer spec"
        )
    return {"w_D20": 1.0, "w_L": 0.0, "w_Cm": 0.0}
