import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_contract import (
    AIR_DENSITY_KGM3,
    REFERENCE_SPEED_MPS,
    TIME_COEFFICIENT,
    FullCarQuantities,
    HalfCarQuantities,
    gcm3_to_kgm3,
    grams_to_kg,
    kg_to_grams,
    m_to_mm,
    mm_to_m,
)


def test_grams_kg_round_trip():
    assert math.isclose(kg_to_grams(grams_to_kg(23.0)), 23.0)


def test_mm_m_round_trip():
    assert math.isclose(m_to_mm(mm_to_m(130.0)), 130.0)


def test_cartridge_mass_conversion_exact():
    assert math.isclose(grams_to_kg(23.0), 0.023)


def test_density_conversion():
    assert math.isclose(gcm3_to_kgm3(1.0), 1000.0)
    assert math.isclose(gcm3_to_kgm3(0.163), 163.0)


def test_forces_and_area_double():
    half = HalfCarQuantities(D20=5.0, L=1.2, A=0.01, pitching_moment_half=0.05)
    full = half.to_full_car()
    assert math.isclose(full.D20, 10.0)
    assert math.isclose(full.L, 2.4)
    assert math.isclose(full.A, 0.02)


def test_half_to_full_cm_not_simply_doubled():
    half = HalfCarQuantities(D20=5.0, L=1.2, A=0.01, pitching_moment_half=0.05)
    full = half.to_full_car()
    naive_doubled_cm = 2.0 * (
        half.pitching_moment_half / (half.q_ref * half.A * (half.A ** 0.5))
    )
    assert not math.isclose(
        full.Cm,
        naive_doubled_cm,
        rel_tol=1e-9,
    ), "Cm must not be computed as a flat doubling of the half-car value"


def test_symmetric_shape_zero_moment_gives_zero_cm():
    half = HalfCarQuantities(D20=5.0, L=1.2, A=0.01, pitching_moment_half=0.0)
    full = half.to_full_car()
    assert math.isclose(full.Cm, 0.0)


def test_zero_area_does_not_divide_by_zero():
    half = HalfCarQuantities(D20=0.0, L=0.0, A=0.0, pitching_moment_half=0.0)
    full = half.to_full_car()
    assert full.Cm == 0.0


def test_full_car_quantities_is_distinct_type_from_half():
    half = HalfCarQuantities(D20=1.0, L=1.0, A=1.0, pitching_moment_half=1.0)
    full = half.to_full_car()
    assert isinstance(full, FullCarQuantities) is True
    assert isinstance(full, HalfCarQuantities) is False


def test_time_coefficient_is_frozen_at_one():
    assert TIME_COEFFICIENT == 1.0


def test_reference_condition_constants():
    assert REFERENCE_SPEED_MPS == 20.0
    assert math.isclose(AIR_DENSITY_KGM3, 1.225)


def test_negative_D20_raises():
    try:
        HalfCarQuantities(D20=-5.0, L=1.2, A=0.01, pitching_moment_half=0.05)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative D20")


def test_negative_area_raises():
    try:
        HalfCarQuantities(D20=5.0, L=1.2, A=-0.01, pitching_moment_half=0.05)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for negative area")


def test_tiny_area_cm_is_zero_not_huge():
    half = HalfCarQuantities(D20=5.0, L=1.2, A=1e-20, pitching_moment_half=0.05)
    full = half.to_full_car()
    assert full.Cm == 0.0, f"Expected Cm=0 for degenerate area, got {full.Cm}"


def test_custom_q_ref_is_used_in_to_full_car():
    custom_q = 999.0
    half = HalfCarQuantities(D20=5.0, L=1.2, A=0.01, pitching_moment_half=0.05, q_ref=custom_q)
    full = half.to_full_car()
    # With q_ref=999, Cm should be much smaller than with default q_ref
    default_half = HalfCarQuantities(D20=5.0, L=1.2, A=0.01, pitching_moment_half=0.05)
    default_full = default_half.to_full_car()
    assert abs(full.Cm) < abs(default_full.Cm), (
        f"Custom q_ref should change Cm: {full.Cm} vs {default_full.Cm}"
    )




def test_constants_defined_twice_still_agree():
    """Four physical constants have two independent definitions each.

    race_objective.py carries its own TRACK_LENGTH, REFERENCE_SPEED, R_WHEEL and
    N_WHEELS alongside physics_contract's TRACK_LENGTH_M / REFERENCE_SPEED_MPS
    and geometry_contract's R_WHEEL_M / N_WHEELS. They agree today and nothing
    makes them. Changing the reference speed in one place and not the other
    would leave the CFD solving one condition while the race objective
    integrates another, with no error anywhere -- just a wrong answer.

    Deliberately a test rather than an import. race_objective is a fitted-model
    file whose coefficients were regressed against these exact values; pointing
    it at another module's constants at import time is a bigger change than the
    risk warrants, and this catches the drift either way.
    """
    import os
    import sys
    _p1 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "part1-simulation")
    if _p1 not in sys.path:
        sys.path.insert(0, _p1)

    import physics_contract as pc
    import race_objective as ro
    import geometry_contract as gc

    pairs = [
        ("track length m", pc.TRACK_LENGTH_M, ro.TRACK_LENGTH),
        ("reference speed m/s", pc.REFERENCE_SPEED_MPS, ro.REFERENCE_SPEED),
        ("wheel radius m", gc.R_WHEEL_M, ro.R_WHEEL),
        ("wheel count", gc.N_WHEELS, ro.N_WHEELS),
    ]
    for label, a, b in pairs:
        assert abs(float(a) - float(b)) < 1e-12, (
            f"{label} is defined twice and the two disagree: {a} vs {b}. "
            f"Whichever is wrong, something is solving a different problem "
            f"than something else.")


if __name__ == "__main__":
    # Collected by name; a hand-written call list silently drops every test
    # appended after it, which has already hidden several tests in this repo.
    _mod = sys.modules[__name__]
    _passed = _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print("PASS " + _n)
            _passed += 1
        except Exception as _e:  # noqa: BLE001
            print("FAIL %s: %r" % (_n, _e))
            _failed += 1
    print("%d passed, %d failed" % (_passed, _failed))
    sys.exit(1 if _failed else 0)
