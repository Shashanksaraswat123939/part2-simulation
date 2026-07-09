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
