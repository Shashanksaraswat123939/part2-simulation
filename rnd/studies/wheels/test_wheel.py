"""Runnable check for wheel.py: python test_wheel.py"""
import json
import math

import wheel as W


def test_thin_ring_and_disc_closed_form():
    # A rim-only wheel (no spokes, no hub) is a thick ring: I = m (R^2 + r^2) / 2.
    w = W.Wheel(R=14.0, t_rim=0.5, w=13.0, n_spokes=0, l_hub=0.0)
    m, i = w.parts()["rim"]
    rho = 1.04e-3
    m_ref = rho * math.pi * (14.0**2 - 13.5**2) * 13.0
    assert abs(m - m_ref) / m_ref < 1e-9
    assert abs(i - m_ref * (14.0**2 + 13.5**2) / 2) / i < 1e-9


def test_model_reproduces_cad_wheels():
    cad = json.load(open("cad_wheels.json"))
    for w, key in ((W.CAD_FRONT, "front_wheel"), (W.CAD_REAR, "rear_wheel")):
        assert abs(w.inertia / cad[key]["I_gmm2"] - 1) < 0.03, (key, w.inertia)
        assert abs(w.mass / cad[key]["mass_g"] - 1) < 0.05, (key, w.mass)


def test_stiffness_is_relative_to_cad():
    assert W.CAD_FRONT.check(W.CAD_FRONT) == (1.0, 1.0)
    thinner = W.Wheel(w=13.25, t_rim=0.3)
    assert max(thinner.check(W.CAD_FRONT)) < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
