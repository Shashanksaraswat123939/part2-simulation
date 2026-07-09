import csv
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from race_objective import BuildSettings, build_smooth_sheet_model, race_value_and_grad
from race_objective_adapter import adapt_gradients, race_value_and_grad_guarded


EXPECTED_HASH = "575636fc3d97c96fe9294a417713227cd2ae992f67123643bc15ed5a5064a30f"


def _synthetic_csv():
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=["time (s)", "force (N)", "mass (kg)"])
        writer.writeheader()
        writer.writerows(
            [
                {"time (s)": 0.0, "force (N)": 3.0, "mass (kg)": 0.048},
                {"time (s)": 0.1, "force (N)": 3.0, "mass (kg)": 0.045},
                {"time (s)": 0.2, "force (N)": 2.0, "mass (kg)": 0.040},
                {"time (s)": 0.3, "force (N)": 1.0, "mass (kg)": 0.035},
                {"time (s)": 0.4, "force (N)": 0.2, "mass (kg)": 0.030},
            ]
        )
    return f.name


def _model_and_params(com_height_m=0.040, time_coefficient=1.0):
    path = _synthetic_csv()
    try:
        model = build_smooth_sheet_model(
            path,
            BuildSettings(n_basis=5, ridge=1e-8, tail_tau=0.025, x_start=1e-4, x_grid_power=2.0, n_steps=60),
        )
    finally:
        Path(path).unlink(missing_ok=True)
    params = np.array([1.0, 0.050, 0.02, 1e-7, time_coefficient, com_height_m], dtype=np.float64)
    return model, params


def test_hash_of_locked_file_matches():
    path = Path(__file__).resolve().parents[1] / "race_objective.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_HASH


def test_adapt_gradients_key_mapping():
    model, params = _model_and_params()
    _T_raw, _T_penalized, raw = race_value_and_grad_guarded(params, model)
    adapted = raw
    assert set(adapted) == {"dT_dD20", "dT_dmass", "dT_dh_com", "dT_dx_com"}
    # Verify the adapter correctly maps locked-file gradient keys
    _value, locked_grads = race_value_and_grad(params, model)
    assert adapted["dT_dD20"] == locked_grads["drag_20_n"]
    assert adapted["dT_dmass"] == locked_grads["car_weight_kg"]
    assert adapted["dT_dx_com"] == 0.0


def test_dT_dh_com_is_now_nonzero():
    model, params = _model_and_params(com_height_m=0.040)
    _T_raw, _T_penalized, adapted = race_value_and_grad_guarded(params, model)
    assert adapted["dT_dh_com"] != 0.0


def test_time_coefficient_guard_blocks_non_unity():
    model, params = _model_and_params(time_coefficient=1.05)
    try:
        race_value_and_grad_guarded(params, model)
    except AssertionError:
        return
    raise AssertionError("Expected AssertionError")


def test_time_coefficient_guard_allows_unity():
    model, params = _model_and_params(time_coefficient=1.0)
    race_value_and_grad_guarded(params, model)


def test_returns_three_tuple_T_raw_T_penalized_gradients():
    """Per the Part 2 spec, the adapter must return (T_raw, T_penalized, gradients).
    T_raw excludes the COM penalty; T_penalized includes it."""
    model, params = _model_and_params(com_height_m=0.040)
    T_raw, T_penalized, gradients = race_value_and_grad_guarded(params, model)
    assert isinstance(T_raw, float)
    assert isinstance(T_penalized, float)
    assert isinstance(gradients, dict)
    assert T_penalized > T_raw, "T_penalized must include COM penalty above T_raw"
    assert set(gradients) == {"dT_dD20", "dT_dmass", "dT_dh_com", "dT_dx_com"}


def test_T_raw_at_target_com_equals_T_penalized():
    """When COM height is exactly at the 30mm target, COM penalty is ~0,
    so T_raw ≈ T_penalized."""
    model, params = _model_and_params(com_height_m=0.030)
    T_raw, T_penalized, _grads = race_value_and_grad_guarded(params, model)
    assert abs(T_penalized - T_raw) < 1e-6, (
        f"COM penalty at target should be ~0, got diff {T_penalized - T_raw}"
    )


def test_negative_mass_rejected():
    """Negative car_weight_kg must raise ValueError, not produce garbage."""
    model, _ = _model_and_params()
    params = np.array([12.0, -0.050, 0.02, 1e-7, 1.0, 0.040], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative mass")


def test_negative_mu_rejected():
    """Negative mu must raise ValueError."""
    model, _ = _model_and_params()
    params = np.array([12.0, 0.050, -0.05, 1e-7, 1.0, 0.040], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative mu")


def test_negative_drag_rejected():
    """Negative D20 must raise ValueError."""
    model, _ = _model_and_params()
    params = np.array([-12.0, 0.050, 0.02, 1e-7, 1.0, 0.040], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative drag")


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
