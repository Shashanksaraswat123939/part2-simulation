import csv
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import (
    FrictionCoefficient,
    compute_held_out_residual,
    fit_com_penalty_curve,
    fit_mu_from_track_test,
    fit_thrust_surrogate,
)


def _csv_path(fieldnames, rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return f.name


def test_thrust_surrogate_missing_file_raises():
    try:
        fit_thrust_surrogate("missing.csv")
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")


def test_thrust_surrogate_missing_column_raises():
    path = _csv_path(["time_s"], [{"time_s": 0.0}, {"time_s": 1.0}])
    try:
        try:
            fit_thrust_surrogate(path)
        except ValueError:
            return
        raise AssertionError("Expected ValueError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_thrust_surrogate_extrapolation_guard():
    path = _csv_path(
        ["time_s", "thrust_N"],
        [{"time_s": 0.0, "thrust_N": 10.0}, {"time_s": 0.5, "thrust_N": 9.0}, {"time_s": 1.0, "thrust_N": 8.0}],
    )
    try:
        surrogate = fit_thrust_surrogate(path)
        try:
            surrogate(5.0)
        except ValueError:
            return
        raise AssertionError("Expected ValueError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_thrust_surrogate_interpolates_known_points():
    path = _csv_path(
        ["time_s", "thrust_N"],
        [{"time_s": 0.0, "thrust_N": 10.0}, {"time_s": 0.5, "thrust_N": 9.0}, {"time_s": 1.0, "thrust_N": 8.0}],
    )
    try:
        surrogate = fit_thrust_surrogate(path)
        assert math.isclose(surrogate(0.5), 9.0, rel_tol=1e-6, abs_tol=1e-6)
    finally:
        Path(path).unlink(missing_ok=True)


def test_mu_missing_file_raises():
    try:
        fit_mu_from_track_test("missing.csv")
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError")


def test_mu_without_mu_fitted_column_raises_notimplementederror():
    path = _csv_path(["deceleration_mps2"], [{"deceleration_mps2": 1.0}])
    try:
        try:
            fit_mu_from_track_test(path)
        except NotImplementedError:
            return
        raise AssertionError("Expected NotImplementedError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_mu_with_mu_fitted_column_returns_mean():
    path = _csv_path(
        ["mu_fitted"],
        [{"mu_fitted": 0.1}, {"mu_fitted": 0.12}, {"mu_fitted": 0.11}],
    )
    try:
        result = fit_mu_from_track_test(path)
        assert math.isclose(result.mu, 0.11)
    finally:
        Path(path).unlink(missing_ok=True)


def test_mu_out_of_range_raises():
    try:
        FrictionCoefficient(mu=1.5)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_com_penalty_missing_column_raises():
    path = _csv_path(["x_com_m"], [{"x_com_m": 0.0}, {"x_com_m": 1.0}])
    try:
        try:
            fit_com_penalty_curve(path)
        except ValueError:
            return
        raise AssertionError("Expected ValueError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_com_penalty_extrapolation_guard():
    path = _csv_path(
        ["x_com_m", "penalty_s"],
        [{"x_com_m": 0.0, "penalty_s": 1.0}, {"x_com_m": 1.0, "penalty_s": 2.0}, {"x_com_m": 2.0, "penalty_s": 5.0}, {"x_com_m": 3.0, "penalty_s": 10.0}],
    )
    try:
        curve = fit_com_penalty_curve(path, degree=2)
        try:
            curve(5.0)
        except ValueError:
            return
        raise AssertionError("Expected ValueError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_com_penalty_fits_known_polynomial():
    rows = [{"x_com_m": x, "penalty_s": 2.0 * x * x + 3.0 * x + 1.0} for x in [0.0, 1.0, 2.0, 3.0, 4.0]]
    path = _csv_path(["x_com_m", "penalty_s"], rows)
    try:
        curve = fit_com_penalty_curve(path, degree=2)
        for x in [0.5, 1.5, 2.5]:
            assert math.isclose(curve(x), 2.0 * x * x + 3.0 * x + 1.0, rel_tol=1e-9, abs_tol=1e-9)
    finally:
        Path(path).unlink(missing_ok=True)


def test_held_out_residual_perfect_fit_gives_r2_near_one():
    rows = [{"x_com_m": float(x), "penalty_s": 2.0 * float(x) + 1.0} for x in range(10)]
    path = _csv_path(["x_com_m", "penalty_s"], rows)
    try:
        r2 = compute_held_out_residual(
            path,
            lambda p: fit_com_penalty_curve(p, degree=1),
            holdout_fraction=0.2,
            seed=0,
        )
        assert r2 > 0.99
    finally:
        Path(path).unlink(missing_ok=True)


def test_single_point_thrust_surrogate_rejected():
    """Single data point should give a clear ValueError, not a scipy crash."""
    path = _csv_path(["time_s", "thrust_N"], [{"time_s": 0.5, "thrust_N": 10.0}])
    try:
        try:
            fit_thrust_surrogate(path)
        except ValueError:
            return
        raise AssertionError("Expected ValueError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_duplicate_x_com_rejected():
    """Duplicate x_com values should be rejected, not silently rank-deficient."""
    path = _csv_path(
        ["x_com_m", "penalty_s"],
        [{"x_com_m": 1.0, "penalty_s": 1.0}, {"x_com_m": 1.0, "penalty_s": 1.0}, {"x_com_m": 1.0, "penalty_s": 1.0}, {"x_com_m": 2.0, "penalty_s": 2.0}],
    )
    try:
        try:
            fit_com_penalty_curve(path, degree=2)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for duplicate x_com")
    finally:
        Path(path).unlink(missing_ok=True)


def test_held_out_residual_tiny_dataset_rejected():
    """Fewer than 4 rows should give a clear ValueError, not a LinAlgError."""
    path = _csv_path(
        ["x_com_m", "penalty_s"],
        [{"x_com_m": 0.0, "penalty_s": 1.0}, {"x_com_m": 1.0, "penalty_s": 2.0}],
    )
    try:
        try:
            compute_held_out_residual(path, lambda p: fit_com_penalty_curve(p, degree=1))
        except ValueError:
            return
        raise AssertionError("Expected ValueError for tiny dataset")
    finally:
        Path(path).unlink(missing_ok=True)


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
