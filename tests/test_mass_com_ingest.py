import math
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mass_com_ingest import FixedHardwareSpec, ingest_mass_com
from physics_contract import ComponentMassCOM


def _fixed_hardware(
    co2_mass=0.023,
    co2_com=(0.0, 0.0, 0.0),
    rear_mass=1.0,
    rear_com=(0.0, 0.0, 0.0),
    wheels_mass=1.0,
    wheels_com=(0.0, 0.0, 0.0),
):
    return FixedHardwareSpec(
        co2_cartridge_mass_kg=co2_mass,
        co2_cartridge_com=co2_com,
        rear_wing_mass_kg=rear_mass,
        rear_wing_com=rear_com,
        wheels_axles_mass_kg=wheels_mass,
        wheels_axles_com=wheels_com,
    )


def test_mass_weighted_com_matches_hand_calculation():
    machined = [
        ComponentMassCOM("a", 1.0, 0.0, 0.0, 0.0),
        ComponentMassCOM("b", 1.0, 1.0, 0.0, 0.0),
    ]
    fixed = _fixed_hardware(
        co2_mass=0.023,
        co2_com=(0.0, 0.0, 0.0),
        rear_mass=0.977,
        rear_com=(2.0, 0.0, 0.0),
        wheels_mass=1.0,
        wheels_com=(1.0, 0.0, 0.0),
    )
    result = ingest_mass_com(machined, fixed)
    assert math.isclose(result.total_mass_kg, 4.0)
    assert math.isclose(result.com_x_m, (1.0 + 0.977 * 2.0 + 1.0) / 4.0)
    assert math.isclose(result.com_y_m, 0.0)
    assert math.isclose(result.com_z_m, 0.0)


def test_empty_machined_components_raises():
    try:
        ingest_mass_com([], _fixed_hardware())
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_negative_mass_component_raises():
    machined = [ComponentMassCOM("bad", -1.0, 0.0, 0.0, 0.0)]
    try:
        ingest_mass_com(machined, _fixed_hardware())
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_zero_mass_component_is_excluded_and_warned_not_fatal():
    """A carved-away component is a valid answer, not an error.

    This used to raise. That killed a run: only the mandatory cargo block is
    pinned hard-solid, so nothing stops the mass term removing a small
    component, and 'nose' reached zero on the 4th iteration of a sweep that was
    otherwise descending nicely (151 g -> 102 g -> 85 g, 2.826 s -> 1.979 s ->
    1.697 s). The warm start then carried the dead nose to the next d_halo,
    which failed on its first iteration too.

    Whether a car missing a component is LEGAL is a rules question and the rule
    checker's job. The mass rollup's job is to describe what is there.
    """
    import warnings
    machined = [ComponentMassCOM("gone", 0.0, 0.0, 0.0, 0.0),
                ComponentMassCOM("body", 0.040, 0.100, 0.0, 0.030)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ingest_mass_com(machined, _fixed_hardware())
    assert any("zero mass" in str(c.message) for c in caught), (
        "a component vanishing is far more often a signal than an intent; it "
        "must warn")
    # The survivor plus fixed hardware, with the empty one contributing nothing.
    assert out.total_mass_kg > 0.040
    assert math.isfinite(out.com_x_m) and math.isfinite(out.com_z_m)


def test_negative_mass_component_still_raises():
    """Zero is a carve; negative is a broken volume integral."""
    machined = [ComponentMassCOM("bad", -1e-6, 0.0, 0.0, 0.0)]
    try:
        ingest_mass_com(machined, _fixed_hardware())
    except ValueError as exc:
        assert "NEGATIVE" in str(exc) or "negative" in str(exc)
        return
    raise AssertionError("Expected ValueError for negative mass")


def test_every_machined_component_gone_still_describes_the_hardware():
    """All machined material carved away is still describable.

    The fixed hardware -- cartridge, wheels, axles, rear wing -- does not go
    away with it, and its masses are validated positive at construction, so the
    rollup has something real to report. My first version of this test asserted
    a raise here and was simply wrong about what is left.

    The rollup describing it does not make it legal; that is the rule gate's
    call. It does warn.
    """
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ingest_mass_com([ComponentMassCOM("gone", 0.0, 0.0, 0.0, 0.0)],
                              _fixed_hardware())
    assert any("zero mass" in str(c.message) for c in caught)
    assert out.total_mass_kg > 0, "fixed hardware still has mass"
    assert math.isfinite(out.com_x_m)


def test_cartridge_mass_must_equal_23g():
    try:
        _fixed_hardware(co2_mass=0.025)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_h_com_and_x_com_extraction():
    machined = [ComponentMassCOM("body", 1.0, 5.0, 0.0, 2.0)]
    fixed = _fixed_hardware(
        co2_mass=0.023,
        co2_com=(5.0, 0.0, 2.0),
        rear_mass=1.0,
        rear_com=(5.0, 0.0, 2.0),
        wheels_mass=1.0,
        wheels_com=(5.0, 0.0, 2.0),
    )
    result = ingest_mass_com(machined, fixed)
    # Downstream reads com_z_m as h_com and com_x_m as x_com; no new fields.
    assert math.isclose(result.com_z_m, 2.0)
    assert math.isclose(result.com_x_m, 5.0)


def test_absurd_com_position_rejected():
    """COM position outside sanity bounds (e.g. 999m) must be rejected,
    catching likely units (mm vs m) or coordinate-origin bugs."""
    machined = [ComponentMassCOM("body", 1.0, 999.0, 999.0, 999.0)]
    fixed = _fixed_hardware(
        co2_mass=0.023,
        co2_com=(999.0, 999.0, 999.0),
        rear_mass=1.0,
        rear_com=(999.0, 999.0, 999.0),
        wheels_mass=1.0,
        wheels_com=(999.0, 999.0, 999.0),
    )
    try:
        ingest_mass_com(machined, fixed)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for absurd COM position")


def test_com_sanity_bounds_exactly_at_limit_accepted():
    """COM exactly at the sanity boundary (10.0m) must be accepted."""
    machined = [ComponentMassCOM("body", 1.0, 10.0, 0.0, 0.0)]
    fixed = _fixed_hardware(
        co2_mass=0.023,
        co2_com=(10.0, 0.0, 0.0),
        rear_mass=1.0,
        rear_com=(10.0, 0.0, 0.0),
        wheels_mass=1.0,
        wheels_com=(10.0, 0.0, 0.0),
    )
    # Should not raise — exactly at boundary is within bounds
    result = ingest_mass_com(machined, fixed)
    # Tolerance, not `== 10.0`: com_x is a mass-weighted sum of four components
    # all at 10.0 m, which lands on 10.000000000000002. Asserting exact float
    # equality on that made this test fail on representation error alone while
    # testing nothing about the boundary behaviour it is named for. The guard
    # itself now carries COM_SANITY_TOL_M for the same reason.
    assert abs(result.com_x_m - 10.0) < 1e-9, result.com_x_m


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
