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


def test_zero_mass_component_raises():
    machined = [ComponentMassCOM("bad", 0.0, 0.0, 0.0, 0.0)]
    try:
        ingest_mass_com(machined, _fixed_hardware())
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


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
