"""Multi-patch cases (wheels, wings, supports as their own patches) and the drift gate."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _box_stl(path, lo, hi, name="s"):
    import itertools
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))), float)
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
         (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    lines = [f"solid {name}"]
    for a, b, c in f:
        n = np.cross(v[b] - v[a], v[c] - v[a]); n = n / np.linalg.norm(n)
        lines += [f"facet normal {n[0]} {n[1]} {n[2]}", "outer loop"]
        lines += [f"vertex {v[i][0]} {v[i][1]} {v[i][2]}" for i in (a, b, c)]
        lines += ["endloop", "endfacet"]
    lines.append(f"endsolid {name}")
    Path(path).write_text("\n".join(lines))


def _surfaces(td):
    _box_stl(f"{td}/w.stl", (0.03, 0.02, 0.0), (0.06, 0.035, 0.028))
    _box_stl(f"{td}/hw.stl", (0.0, 0.0, 0.005), (0.02, 0.03, 0.008))
    return ({"name": "wheelF", "stl": f"{td}/w.stl",
             "rotating": {"origin": (0.045, 0.03, 0.014), "axis": (0, 1, 0), "omega": -1415.4}},
            {"name": "hardware", "stl": f"{td}/hw.stl", "rotating": None})


def test_forward_case_gets_patches_bcs_and_force_groups():
    import openfoam_case as oc
    with tempfile.TemporaryDirectory() as td:
        _box_stl(f"{td}/car.stl", (0.02, 0.0, 0.002), (0.2, 0.02, 0.04))
        cfg = oc.OpenFOAMRunConfig(resolution="coarse", extra_surfaces=_surfaces(td))
        oc.build_case(f"{td}/run", f"{td}/car.stl", cfg)
        run = Path(td) / "run"
        u = (run / "0" / "U").read_text()
        assert "wheelF { type rotatingWallVelocity;" in u and "omega -1415.4" in u
        assert "hardware" in u and u.count("noSlip") == 2
        for fld in ("p", "k", "omega", "nut"):
            t = (run / "0" / fld).read_text()
            assert "wheelF" in t and "hardware" in t, fld
        sd = (run / "system" / "snappyHexMeshDict").read_text()
        assert 'file "wheelF.stl"' in sd and "wheelF { level (2 4)" in sd
        assert '"(car|wheelF|hardware)"' in sd
        cd = (run / "system" / "controlDict").read_text()
        assert "patches         (car wheelF hardware);" in cd
        assert "forces_wheelF" in cd and "forces_car" in cd
        assert (run / "constant" / "triSurface" / "wheelF.stl").read_text().startswith("solid wheelF")


def test_adjoint_objective_covers_all_patches_design_only_car():
    import openfoam_adjoint as oa
    with tempfile.TemporaryDirectory() as td:
        _box_stl(f"{td}/car.stl", (0.02, 0.0, 0.002), (0.2, 0.02, 0.04))
        cfg = oa.AdjointRunConfig(resolution="coarse", extra_surfaces=_surfaces(td))
        oa.build_adjoint_case(f"{td}/run", f"{td}/car.stl", cfg)
        run = Path(td) / "run"
        od = (run / "system" / "optimisationDict").read_text()
        assert "patches    (car wheelF hardware);" in od
        assert "patches            (car);" in od          # design variables: body only
        for fld in ("Ua", "pa", "ka", "wa", "U"):
            assert "wheelF" in (run / "0" / fld).read_text(), fld


def test_car_patch_points_from_a_polymesh():
    import openfoam_adjoint as oa
    with tempfile.TemporaryDirectory() as td:
        poly = Path(td) / "constant" / "polyMesh"; poly.mkdir(parents=True)
        (poly / "boundary").write_text(
            "2\n(\ninlet\n{\n type patch;\n nFaces 1;\n startFace 1;\n}\n"
            "car\n{\n type wall;\n nFaces 2;\n startFace 2;\n}\n)\n")
        (poly / "faces").write_text("x\n4\n(\n4(0 1 2 3)\n3(4 5 6)\n4(7 8 9 10)\n3(10 11 7)\n)\n")
        ids = oa.car_patch_point_ids(td)
        assert ids.tolist() == [7, 8, 9, 10, 11]


def test_drift_fails_convergence():
    import cfd_wrapper as cw
    assert cw.MAX_FORCE_DRIFT == 0.02
    base = {"D20_half": 0.14, "L_half": 0.0, "A_half": 0.001, "pitching_moment_half": 0.0,
            "residual_final": 1e-3, "negative_volume_cells": 0, "y_plus_min": 1.0,
            "y_plus_max": 2.0, "courant_max": None, "force_mean_stderr": 0.005}
    with tempfile.TemporaryDirectory() as td:
        _box_stl(f"{td}/car.stl", (0.02, 0.0, 0.002), (0.2, 0.02, 0.04))
        orig = cw._invoke_openfoam_pipeline
        try:
            for drift, ok in ((0.004, True), (0.09, False)):
                cw._invoke_openfoam_pipeline = lambda *a, _d=drift, **k: dict(base, force_drift=_d)
                _half, health = cw.run_half_car_cfd(f"{td}/car.stl")
                assert health.converged is ok, (drift, health)
        finally:
            cw._invoke_openfoam_pipeline = orig

if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _fails = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)(); print("PASS", _n)
        except Exception as e:  # noqa: BLE001
            _fails += 1; print("FAIL", _n, "->", repr(e))
    print(f"{_fails} failed")
    sys.exit(1 if _fails else 0)
