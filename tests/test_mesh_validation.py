import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfd_wrapper import CFDHealthReport
from mesh_validation import (
    run_laminar_vs_komega_sst_comparison,
    run_mesh_independence_study,
    run_speed_sensitivity_check,
)
from physics_contract import HalfCarQuantities


def _runner_result(D20=5.0, L=1.0, moment=0.05, A=0.01):
    return (
        HalfCarQuantities(D20=D20, L=L, A=A, pitching_moment_half=moment),
        CFDHealthReport(True, 1e-5, 0, 0.4, 2.0, 0.8),
    )


def test_mesh_independence_pass_case():
    values = iter([5.0, 5.02, 4.98])

    def fake_runner(stl_path, resolution):
        return _runner_result(D20=next(values), L=1.0, moment=0.05)

    result = run_mesh_independence_study("car.stl", cfd_runner=fake_runner)
    assert result.passes_5_percent_target is True


def test_mesh_independence_fail_case():
    values = iter([5.0, 6.0, 4.0])

    def fake_runner(stl_path, resolution):
        return _runner_result(D20=next(values), L=1.0, moment=0.05)

    result = run_mesh_independence_study("car.stl", cfd_runner=fake_runner)
    assert result.passes_5_percent_target is False


def test_solver_comparison_relative_delta_formula():
    laminar = lambda stl_path: _runner_result(D20=5.0, L=1.0, moment=0.05)
    komega = lambda stl_path: _runner_result(D20=5.5, L=1.1, moment=0.055)
    result = run_laminar_vs_komega_sst_comparison(
        "car.stl",
        laminar_runner=laminar,
        komega_sst_runner=komega,
    )
    assert math.isclose(result.relative_delta_D20, abs(11.0 - 10.0) / 10.0)


def test_komega_runner_none_raises_notimplementederror():
    try:
        run_laminar_vs_komega_sst_comparison("car.stl")
    except NotImplementedError:
        return
    raise AssertionError("Expected NotImplementedError")


def test_speed_sensitivity_cda_backsolve():
    def fake_runner(stl_path, reference_speed_mps=20.0):
        if reference_speed_mps == 5.0:
            return _runner_result(D20=1.225)
        if reference_speed_mps == 20.0:
            return _runner_result(D20=24.5)
        raise AssertionError("unexpected speed")

    result = run_speed_sensitivity_check("car.stl", cfd_runner=fake_runner)
    assert math.isclose(result.CdA_at_5mps, 1.225 / (0.5 * 1.225 * 5.0 ** 2))
    assert math.isclose(result.CdA_at_20mps, 24.5 / (0.5 * 1.225 * 20.0 ** 2))


def test_zero_relative_spread_is_safe():
    """All-zero values should not cause ZeroDivisionError."""
    from mesh_validation import _relative_spread
    result = _relative_spread((0.0, 0.0, 0.0))
    assert result == 0.0


def test_mixed_sign_relative_spread_is_safe():
    """Mixed-sign values that cancel to zero mean should NOT report zero spread.
    The old mean-based formula returned 0.0 for (-1, 1, 0), masking real spread.
    The new max(abs)-based formula correctly reports nonzero spread."""
    from mesh_validation import _relative_spread
    result = _relative_spread((-1.0, 1.0, 0.0))
    assert result > 0.0, (
        "values that disagree in sign should show nonzero relative spread, "
        "not be masked by their mean happening to cancel to zero"
    )


def test_all_near_zero_relative_spread_is_safe():
    """Genuinely negligible values (not just zero-mean) should report 0 spread."""
    from mesh_validation import _relative_spread
    result = _relative_spread((1e-12, -1e-12, 0.0))
    assert result == 0.0




def test_mesh_independence_without_runner_raises_typeerror():
    """Calling run_mesh_independence_study without a cfd_runner must raise
    TypeError, not silently produce a false-positive PASS."""
    try:
        run_mesh_independence_study("car.stl")
    except TypeError:
        return
    raise AssertionError("Expected TypeError when cfd_runner not supplied")

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
