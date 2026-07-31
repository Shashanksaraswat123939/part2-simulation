import csv
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from race_objective import (BuildSettings, PARAM_NAMES, build_smooth_sheet_model,
                            race_value_and_grad)
from race_objective_adapter import adapt_gradients, race_value_and_grad_guarded


# The former SHA-256 pin, kept only as a provenance marker. The lock was
# removed on the owner's instruction (2026-07-28); see
# test_objective_physics_is_unchanged_by_edits for what replaced it.
_ORIGINAL_LOCKED_HASH = "9d8c49c488b5c2f960daca9ab575f03ccb1eba99534a24303233f7ccc61d2c84"


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


def _model_and_params(com_height_m=0.040, time_coefficient=1.0, lift_20_n=0.5, com_x_m=0.005):
    path = _synthetic_csv()
    try:
        model = build_smooth_sheet_model(
            path,
            BuildSettings(n_basis=5, ridge=1e-8, tail_tau=0.025, x_start=1e-4, x_grid_power=2.0, n_steps=60),
        )
    finally:
        Path(path).unlink(missing_ok=True)
    params = np.array([1.0, 0.050, 0.02, 1e-7, time_coefficient, com_height_m, lift_20_n, com_x_m], dtype=np.float64)
    return model, params


def test_objective_physics_is_unchanged_by_edits():
    """The hash lock is gone (owner's instruction, 2026-07-28). What replaces it
    is a check on BEHAVIOUR, which is what the lock was actually protecting.

    The lock blocked two necessary changes in a row -- the cartridge
    double-count, then `jax.value_and_grad` being rebuilt per call until Stage 1
    exhausted vm.max_map_count and segfaulted -- while never once catching a
    physics regression. A hash cannot tell a fix from a corruption; these
    invariants can.
    """
    import numpy as np
    from race_objective import build_smooth_sheet_model, race_value_and_grad

    csv = str(Path(__file__).resolve().parents[1] / "co2_thrust_data.csv")
    model = build_smooth_sheet_model(csv)
    p = np.array([0.7076, 0.158598, 0.06, 1.5e-7, 1.0, 0.025, 0.02, 0.10],
                 dtype=np.float64)
    T, grads = race_value_and_grad(p, model)

    assert 1.0 < T < 20.0, f"race time {T} s is not physical for a 20 m run"
    # Signs are the physics: heavier and draggier are slower, a lower COM helps.
    assert grads["car_weight_kg"] > 0, "a heavier car must be slower"
    assert grads["drag_20_n"] > 0, "more drag must be slower"
    assert grads["com_height_m"] < 0, "a lower COM must not be penalised"
    # Determinism: repeated calls must agree exactly. This is also what caught
    # the retrace leak -- a fresh trace per call still returns the same number,
    # so only the mapping count exposed it. Kept as a cheap sanity anchor.
    T2, _ = race_value_and_grad(p, model)
    assert T == T2, f"objective is not deterministic: {T} != {T2}"


def test_com_height_clamps_instead_of_killing_a_working_optimiser():
    """dT/dh_com is negative, so the optimiser lowers the COM toward the fitted
    range's floor. Raising there meant the better it worked, the sooner it
    killed its own candidate -- ~320 iterations from the measured 29.2 mm start
    against an 18 mm floor.
    """
    import warnings as _w
    from race_objective_adapter import COM_HEIGHT_FIT_RANGE_M, _COM_CLAMP_WARNED

    model, params = _model_and_params()
    lo, _hi = COM_HEIGHT_FIT_RANGE_M
    params = np.asarray(params, dtype=np.float64).copy()
    params[PARAM_NAMES.index("com_height_m")] = lo - 0.002   # 2 mm below the floor

    _COM_CLAMP_WARNED.clear()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        T_raw, T_pen, grads = race_value_and_grad_guarded(params, model)
    assert any("clamping" in str(c.message) for c in caught), (
        "clamping must be announced, not silent")
    assert T_raw > 0 and T_pen > 0, "clamped call must still return a race time"
    # Flat beyond the data: two points below the floor must agree.
    p2 = params.copy()
    p2[PARAM_NAMES.index("com_height_m")] = lo - 0.004
    T2_raw, T2_pen, _ = race_value_and_grad_guarded(p2, model)
    assert abs(T_pen - T2_pen) < 1e-12, (
        f"penalty is not flat below the fitted range: {T_pen} vs {T2_pen}")


def test_com_height_far_outside_is_still_a_units_bug():
    """Clamping must not swallow a genuine units/origin error."""
    from race_objective_adapter import COM_HEIGHT_FIT_RANGE_M
    model, params = _model_and_params()
    lo, hi = COM_HEIGHT_FIT_RANGE_M
    params = np.asarray(params, dtype=np.float64).copy()
    params[PARAM_NAMES.index("com_height_m")] = hi + 2.0 * (hi - lo)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError as exc:
        assert "units or origin bug" in str(exc)
        return
    raise AssertionError("a far-outside COM height was silently clamped")


def test_clamping_does_not_disable_the_com_x_guard():
    """Regression: an early return on the clamp path skipped the com_x check."""
    from race_objective_adapter import COM_HEIGHT_FIT_RANGE_M
    model, params = _model_and_params()
    lo, _hi = COM_HEIGHT_FIT_RANGE_M
    params = np.asarray(params, dtype=np.float64).copy()
    params[PARAM_NAMES.index("com_height_m")] = lo - 0.002    # triggers clamping
    params[PARAM_NAMES.index("com_x_m")] = 999.0              # absurd
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError as exc:
        assert "com_x_m" in str(exc), f"wrong guard fired: {exc}"
        return
    raise AssertionError("com_x sanity guard was skipped on the clamp path")


def test_mapping_leak_is_documented_where_it_bites():
    """The leak is MITIGATED (vm.max_map_count), not fixed in code.

    An earlier version of this test asserted that hoisting
    jax.value_and_grad out of the call fixed it. That was wrong:
    value_and_grad is not a caching transform, only jax.jit is, and hoisting
    changed nothing (999 vs 1017 ms/call measured). Asserting a false invariant
    is worse than asserting none -- it would have let someone "restore" the real
    fix and think they had broken something.

    What this pins instead is that the next person meets the explanation.
    """
    import inspect
    import race_objective as ro

    src = inspect.getsource(ro)
    head = src[:src.index("def race_value_and_grad(")]
    for needed in ("max_map_count", "jax.jit", "not bit-identical"):
        assert needed in head, (
            f"the mapping-leak note lost its mention of {needed!r}; whoever "
            f"hits the segfault next needs the measurement, the real fix, and "
            f"why it was not applied")


def test_cartridge_mass_is_not_double_counted():
    """car_weight_kg owns the cartridge HARDWARE; the objective owns only the
    unspent propellant.

    Regression guard for the 2026-07-24 correction. The old formula was
    `car_weight_kg + 0.021 + sheet(t) - 0.048`, which added a SECOND cartridge
    shell on top of the CO2_CARTRIDGE_MASS_KG already summed into
    car_weight_kg by mass_com_ingest -- a 50 g car entered the integrator at
    78.9 g. Two properties pin it:

      1. at the finish the vehicle must weigh EXACTLY car_weight_kg (all
         propellant spent, nothing else added);
      2. at the line the excess must equal the CSV's own propellant mass,
         not a cartridge-shell-sized lump.
    """
    import jax.numpy as jnp
    from race_objective import build_smooth_sheet_model, car_mass_from_time, sheet_mass

    path = _synthetic_csv()
    try:
        model = build_smooth_sheet_model(path, BuildSettings(
            n_basis=5, ridge=1e-8, tail_tau=0.025, x_start=1e-4,
            x_grid_power=2.0, n_steps=60))
    finally:
        Path(path).unlink(missing_ok=True)

    cw = 0.050
    end = float(car_mass_from_time(jnp.asarray(1e3), jnp.asarray(cw), model))
    assert abs(end - cw) < 1e-6, (
        f"at the finish the car must weigh exactly car_weight_kg ({cw} kg), got "
        f"{end} kg -- something other than propellant is being added"
    )
    start = float(car_mass_from_time(jnp.asarray(0.0), jnp.asarray(cw), model))
    expected = cw + float(sheet_mass(jnp.asarray(0.0), model) - model.mass_sheet_final)
    assert abs(start - expected) < 1e-9, (
        f"start mass {start} != car_weight_kg + propellant {expected}"
    )
    # Same two properties against the REAL project thrust curve, where the
    # numbers are meaningful: its mass column runs 0.05589 -> 0.04802, so the
    # excess at the line must be ~7.9 g of CO2. The old formula gave 28.9 g.
    real_csv = Path(__file__).resolve().parents[1] / "co2_thrust_data.csv"
    if real_csv.exists():
        real = build_smooth_sheet_model(str(real_csv))
        r_start = float(car_mass_from_time(jnp.asarray(0.0), jnp.asarray(cw), real))
        r_end = float(car_mass_from_time(jnp.asarray(1e3), jnp.asarray(cw), real))
        assert abs(r_end - cw) < 1e-6, f"real CSV: finish mass {r_end} != {cw}"
        excess_g = (r_start - cw) * 1000.0
        assert 5.0 < excess_g < 12.0, (
            f"real CSV: excess at the line is {excess_g:.2f} g, expected ~7.9 g of "
            "CO2. Near 29 g means the cartridge shell is being double-counted again."
        )


def test_adapt_gradients_key_mapping():
    model, params = _model_and_params()
    _T_raw, _T_penalized, raw = race_value_and_grad_guarded(params, model)
    adapted = raw
    assert set(adapted) == {"dT_dD20", "dT_dmass", "dT_dh_com", "dT_dx_com", "dT_dL"}
    # Verify the adapter correctly maps locked-file gradient keys
    _value, locked_grads = race_value_and_grad(params, model)
    assert adapted["dT_dD20"] == locked_grads["drag_20_n"]
    assert adapted["dT_dmass"] == locked_grads["car_weight_kg"]
    assert adapted["dT_dx_com"] != 0.0  # now non-zero with com_x penalty


def test_dT_dh_com_is_now_nonzero():
    model, params = _model_and_params(com_height_m=0.040)
    _T_raw, _T_penalized, adapted = race_value_and_grad_guarded(params, model)
    assert adapted["dT_dh_com"] != 0.0


def test_dT_dL_is_nonzero():
    """Lift gradient must be non-zero now that lift-dependent friction is modeled."""
    model, params = _model_and_params(lift_20_n=0.5)
    _T_raw, _T_penalized, adapted = race_value_and_grad_guarded(params, model)
    assert "dT_dL" in adapted
    assert adapted["dT_dL"] != 0.0


def test_dT_dx_com_is_nonzero():
    """Fore-aft COM gradient must be non-zero now that com_x penalty is modeled."""
    model, params = _model_and_params(com_x_m=0.005)
    _T_raw, _T_penalized, adapted = race_value_and_grad_guarded(params, model)
    assert adapted["dT_dx_com"] != 0.0


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
    assert set(gradients) == {"dT_dD20", "dT_dmass", "dT_dh_com", "dT_dx_com", "dT_dL"}


def test_T_raw_at_target_com_equals_T_penalized():
    """When COM height is exactly at the 30mm target, COM penalty is ~0,
    so T_raw ≈ T_penalized."""
    model, params = _model_and_params(com_height_m=0.030, com_x_m=0.0)
    T_raw, T_penalized, _grads = race_value_and_grad_guarded(params, model)
    assert abs(T_penalized - T_raw) < 1e-6, (
        f"COM penalty at target should be ~0, got diff {T_penalized - T_raw}"
    )


def test_negative_mass_rejected():
    """Negative car_weight_kg must raise ValueError, not produce garbage."""
    model, _ = _model_and_params()
    params = np.array([12.0, -0.050, 0.02, 1e-7, 1.0, 0.040, 0.5, 0.005], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative mass")


def test_negative_mu_rejected():
    """Negative mu must raise ValueError."""
    model, _ = _model_and_params()
    params = np.array([12.0, 0.050, -0.05, 1e-7, 1.0, 0.040, 0.5, 0.005], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative mu")


def test_negative_drag_rejected():
    """Negative D20 must raise ValueError."""
    model, _ = _model_and_params()
    params = np.array([-12.0, 0.050, 0.02, 1e-7, 1.0, 0.040, 0.5, 0.005], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative drag")




def test_com_height_out_of_fitted_range_is_clamped_not_rejected():
    """Superseded 2026-07-28: outside the FITTED range now clamps.

    This used to assert that 50 mm raises. It does not any more, and that is
    deliberate: dT/dh_com is negative, so a working optimiser walks the COM
    toward the fitted range's 18 mm floor and out of it -- raising there meant
    the better the optimiser worked, the sooner it killed its own candidate.
    50 mm is a legal COM height on a 65 mm car, so it is clamped to the fitted
    ceiling and the penalty goes flat. What still raises is a PHYSICALLY
    impossible height (see the next test).
    """
    model, _ = _model_and_params()
    params = np.array([12.0, 0.050, 0.02, 1e-7, 1.0, 0.050, 0.5, 0.005],
                      dtype=np.float64)
    T_raw, T_pen, _ = race_value_and_grad_guarded(params, model)
    assert T_raw > 0 and T_pen > 0, "a clamped call must still produce a time"


def test_com_height_outside_the_car_is_still_rejected():
    """A COM above the 65 mm car or below the track is a units/origin bug."""
    model, _ = _model_and_params()
    for bad in (0.50, -0.01):     # 500 mm up, and below the track
        params = np.array([12.0, 0.050, 0.02, 1e-7, 1.0, bad, 0.5, 0.005],
                          dtype=np.float64)
        try:
            race_value_and_grad_guarded(params, model)
        except ValueError as exc:
            assert "units or origin bug" in str(exc), f"wrong guard: {exc}"
            continue
        raise AssertionError(f"com_height_m={bad} was accepted")


def test_com_height_at_boundary_accepted():
    """COM height at the boundary of the fitted range should be accepted
    (with small float tolerance)."""
    model, _ = _model_and_params()
    # 0.018 is the lower boundary
    params = np.array([12.0, 0.050, 0.02, 1e-7, 1.0, 0.018, 0.5, 0.005], dtype=np.float64)
    race_value_and_grad_guarded(params, model)


def test_negative_wheel_moi_rejected():
    """Negative wheel_moi_kg_m2 must raise ValueError."""
    model, _ = _model_and_params()
    params = np.array([12.0, 0.050, 0.02, -1e-7, 1.0, 0.040, 0.5, 0.005], dtype=np.float64)
    try:
        race_value_and_grad_guarded(params, model)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative wheel_moi")


def test_build_settings_n_basis_zero_rejected():
    """BuildSettings with n_basis=0 must be rejected by the guarded wrapper."""
    from race_objective_adapter import build_smooth_sheet_model_guarded
    from race_objective import BuildSettings
    csv_path = _synthetic_csv()
    try:
        try:
            build_smooth_sheet_model_guarded(csv_path, BuildSettings(n_basis=0, n_steps=60, ridge=1e-8, tail_tau=0.025, x_start=1e-4, x_grid_power=2.0))
        except ValueError:
            return
        raise AssertionError("Expected ValueError for n_basis=0")
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_build_settings_n_steps_zero_rejected():
    """BuildSettings with n_steps=0 must be rejected by the guarded wrapper."""
    from race_objective_adapter import build_smooth_sheet_model_guarded
    from race_objective import BuildSettings
    csv_path = _synthetic_csv()
    try:
        try:
            build_smooth_sheet_model_guarded(csv_path, BuildSettings(n_basis=5, n_steps=0, ridge=1e-8, tail_tau=0.025, x_start=1e-4, x_grid_power=2.0))
        except ValueError:
            return
        raise AssertionError("Expected ValueError for n_steps=0")
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_validate_thrust_csv_rejects_negative_time():
    """Thrust CSV with negative time values must be rejected."""
    from race_objective_adapter import validate_thrust_csv_physical_sanity
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=["time (s)", "force (N)", "mass (kg)"])
        writer.writeheader()
        writer.writerows([
            {"time (s)": -0.1, "force (N)": 3.0, "mass (kg)": 0.048},
            {"time (s)": 0.1, "force (N)": 3.0, "mass (kg)": 0.045},
        ])
    try:
        try:
            validate_thrust_csv_physical_sanity(f.name)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for negative time")
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_validate_thrust_csv_rejects_negative_force():
    """Thrust CSV with negative force values must be rejected."""
    from race_objective_adapter import validate_thrust_csv_physical_sanity
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=["time (s)", "force (N)", "mass (kg)"])
        writer.writeheader()
        writer.writerows([
            {"time (s)": 0.0, "force (N)": -3.0, "mass (kg)": 0.048},
            {"time (s)": 0.1, "force (N)": 3.0, "mass (kg)": 0.045},
        ])
    try:
        try:
            validate_thrust_csv_physical_sanity(f.name)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for negative force")
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_validate_thrust_csv_rejects_non_positive_mass():
    """Thrust CSV with non-positive mass values must be rejected."""
    from race_objective_adapter import validate_thrust_csv_physical_sanity
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=["time (s)", "force (N)", "mass (kg)"])
        writer.writeheader()
        writer.writerows([
            {"time (s)": 0.0, "force (N)": 3.0, "mass (kg)": 0.048},
            {"time (s)": 0.1, "force (N)": 3.0, "mass (kg)": 0.0},
        ])
    try:
        try:
            validate_thrust_csv_physical_sanity(f.name)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for non-positive mass")
    finally:
        Path(f.name).unlink(missing_ok=True)


def test_validate_thrust_csv_accepts_valid_csv():
    """A physically valid thrust CSV must pass the sanity check."""
    from race_objective_adapter import validate_thrust_csv_physical_sanity
    csv_path = _synthetic_csv()
    try:
        validate_thrust_csv_physical_sanity(csv_path)
    finally:
        Path(csv_path).unlink(missing_ok=True)



def test_the_com_penalty_is_never_negative():
    """A penalty that pays you back breaks the layer above.

    The adapter derives T_raw = T_penalized - tc*com_h_penalty, so a negative
    penalty puts T_raw ABOVE T_penalized and CandidateOutcome.__post_init__
    rejects the iteration outright ("penalties are additive and non-negative,
    this is a bug upstream"). That raise killed d_halo=16 three iterations into
    the live run of 2026-07-30, with a reported gap of 2.03e-7 s.

    The cause was a fitting artefact: the degree-4 fit through nine placeholder
    points bottoms out at +0.4 mm rather than at the 30 mm target, so
    subtracting the value AT the target left a well from 30.05 mm to 30.80 mm
    reaching -3.79e-5 s. A 2 mm sample grid steps straight over it, which is
    how it survived an earlier check -- so this samples finely, and across the
    whole physical range rather than near the target only.
    """
    import numpy as np
    import jax.numpy as jnp
    from race_objective import com_height_time_penalty, PARAM_NAMES

    def pen(h_mm):
        p = np.zeros(len(PARAM_NAMES))
        p[0], p[1], p[2], p[3] = 0.64, 0.149, 0.010, 1e-7
        p[4], p[5], p[6], p[7] = 1.0, h_mm / 1000.0, 0.0, 0.122
        return float(com_height_time_penalty(jnp.asarray(p)))

    heights = np.linspace(15.0, 45.0, 6001)          # 5 micron steps
    values = np.array([pen(h) for h in heights])
    worst = values.min()
    assert worst >= -1e-12, (
        f"COM height penalty goes to {worst:.3e} s at "
        f"h_com={heights[values.argmin()]:.3f} mm. T_raw is derived by "
        f"subtracting this, so a negative value inverts T_raw and T_penalized "
        f"and the candidate is rejected upstream.")

    # And it must still be ~zero at the declared target, or the target means
    # nothing -- the two constraints together are what forced a clamp rather
    # than a re-baseline to the fit's own minimum.
    assert abs(pen(30.0)) < 1e-9, (
        f"penalty at the 30 mm target is {pen(30.0):.3e}, not ~0")


if __name__ == "__main__":
    # Collected by name; a hand-written call list silently drops tests appended
    # after it, which has already hidden several tests in this repo.
    _mod = sys.modules[__name__]
    _p = _f = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print("PASS " + _n)
            _p += 1
        except Exception as _e:  # noqa: BLE001
            print("FAIL %s -> %s" % (_n, _e))
            _f += 1
    print("%d passed, %d failed" % (_p, _f))
    sys.exit(1 if _f else 0)
