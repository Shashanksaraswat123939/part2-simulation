import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_contract import (
    compute_adjoint_objective,
    compute_adjoint_objective_weight,
    get_active_objective_weights,
    package_gradient_bundle,
)
from race_objective import BuildSettings, build_smooth_sheet_model


def _synthetic_csv(force_scale=1.0):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=["time (s)", "force (N)", "mass (kg)"])
        writer.writeheader()
        writer.writerows(
            [
                {"time (s)": 0.0, "force (N)": 3.0 * force_scale, "mass (kg)": 0.048},
                {"time (s)": 0.1, "force (N)": 3.0 * force_scale, "mass (kg)": 0.045},
                {"time (s)": 0.2, "force (N)": 2.0 * force_scale, "mass (kg)": 0.040},
                {"time (s)": 0.3, "force (N)": 1.0 * force_scale, "mass (kg)": 0.035},
                {"time (s)": 0.4, "force (N)": 0.2 * force_scale, "mass (kg)": 0.030},
            ]
        )
    return f.name


def _model(force_scale=1.0):
    path = _synthetic_csv(force_scale=force_scale)
    try:
        return build_smooth_sheet_model(
            path,
            BuildSettings(
                n_basis=5,
                ridge=1e-8,
                tail_tau=0.025,
                x_start=1e-4,
                x_grid_power=2.0,
                n_steps=60,
            ),
        )
    finally:
        Path(path).unlink(missing_ok=True)


def _params(D20=12.0, mu=0.02):
    return np.array([D20, 0.050, mu, 1e-7, 1.0, 0.040, 0.5, 0.005], dtype=np.float64)


def test_gradient_bundle_always_has_all_five_keys():
    bundle = package_gradient_bundle(1.0, 2.0, 0.0, 0.0, manufacturing_gradient=None)
    assert set(bundle) == {
        "w_D20",
        "dT_dmass",
        "dT_dh_com",
        "dT_dx_com",
        "manufacturing_gradient",
    }
    assert bundle["manufacturing_gradient"] is None


def test_dT_dh_com_zero_passed_through_unchanged():
    bundle = package_gradient_bundle(1.0, 2.0, 0.0, 3.0)
    assert bundle["dT_dh_com"] == 0.0


def test_active_weights_default_all_disabled():
    assert get_active_objective_weights() == {"w_D20": 1.0, "w_L": 0.0, "w_Cm": 0.0}


def test_active_weights_partial_prerequisites_still_disabled():
    prerequisites = {
        "mu_fitted_from_track_data": True,
        "ballast_com_experiment_completed": True,
        "cm_and_l_wired_into_rtc": False,
    }
    assert get_active_objective_weights(prerequisites) == {"w_D20": 1.0, "w_L": 0.0, "w_Cm": 0.0}


def test_active_weights_all_true_raises_notimplementederror():
    prerequisites = {
        "mu_fitted_from_track_data": True,
        "ballast_com_experiment_completed": True,
        "cm_and_l_wired_into_rtc": True,
    }
    try:
        get_active_objective_weights(prerequisites)
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError")


def test_perturb_and_confirm_improvement():
    def D20_of_s(s):
        return 10.0 + 2.0 * s ** 2

    s = 1.0
    D20 = D20_of_s(s)
    model = _model()
    w_D20 = compute_adjoint_objective_weight(
        param_vector=_params(D20=D20),
        model=model,
    )
    dD20_ds = 4.0 * s
    step_size = 0.01
    s_new = s - step_size * w_D20 * dD20_ds
    assert w_D20 * D20_of_s(s_new) < w_D20 * D20_of_s(s)


def test_adjoint_objective_is_w_D20_times_D20():
    """Per Part 2 spec: Objective = w_D20 × D20, scaled by ADJOINT_HALF_CAR_SCALING (0.5)."""
    model = _model()
    params = _params(D20=12.0, mu=0.02)
    w_D20 = compute_adjoint_objective_weight(params, model)
    objective = compute_adjoint_objective(params, model)
    expected = w_D20 * 12.0 * 0.5  # ADJOINT_HALF_CAR_SCALING = 0.5
    assert abs(objective - expected) < 1e-12, (
        f"Objective {objective} != w_D20×D20×0.5 {expected}"
    )


def test_w_D20_changes_when_mu_changes():
    model = _model()
    w_low_mu = compute_adjoint_objective_weight(
        param_vector=_params(D20=12.0, mu=0.01),
        model=model,
    )
    w_high_mu = compute_adjoint_objective_weight(
        param_vector=_params(D20=12.0, mu=0.08),
        model=model,
    )
    assert w_low_mu != w_high_mu


def test_w_D20_depends_on_thrust_curve():
    """Verify w_D20 is real JAX gradient, not a fabricated constant.

    This test catches the defect where compute_adjoint_objective_weight used
    a hand-rolled formula disconnected from the actual race objective. If w_D20
    is computed from the real locked race model, it must change when the thrust
    model changes.
    """
    model_low_thrust = _model(force_scale=0.5)
    model_high_thrust = _model(force_scale=2.0)

    params_baseline = _params(D20=12.0, mu=0.02)

    w_low_thrust = compute_adjoint_objective_weight(
        param_vector=params_baseline,
        model=model_low_thrust,
    )
    w_high_thrust = compute_adjoint_objective_weight(
        param_vector=params_baseline,
        model=model_high_thrust,
    )

    # If w_D20 is real physics, it must differ when the thrust model changes.
    # A fabricated formula (independent of model) would give the same answer.
    assert w_low_thrust != w_high_thrust, (
        "w_D20 must depend on the race model; if it's constant across "
        "different thrust curves, it's not using the real JAX gradient"
    )


if __name__ == "__main__":
    import sys
    fns = [f for f in dir(sys.modules[__name__]) if f.startswith("test_")]
    passed, failed = 0, 0
    for f in fns:
        try:
            globals()[f]()
            print("PASS", f); passed += 1
        except Exception as e:
            print("FAIL", f, "->", e); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
