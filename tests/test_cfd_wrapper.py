import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cfd_wrapper
from cfd_wrapper import CFDRunError, run_half_car_cfd
from physics_contract import HalfCarQuantities


def _write_stl(triangles):
    f = tempfile.NamedTemporaryFile("w", suffix=".stl", delete=False, encoding="utf-8")
    with f:
        f.write("solid test\n")
        for tri in triangles:
            f.write("  facet normal 0 0 0\n")
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid test\n")
    return f.name


def _tetrahedron_path():
    triangles = [
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ]
    return _write_stl(triangles)


def _open_stl_path():
    triangles = [
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
    ]
    return _write_stl(triangles)


def _fake_dict(**overrides):
    data = {
        "D20_half": 5.0,
        "L_half": 1.2,
        "A_half": 0.01,
        "pitching_moment_half": 0.05,
        "residual_final": 1e-5,
        "negative_volume_cells": 0,
        "y_plus_min": 0.4,
        "y_plus_max": 2.0,
        "courant_max": 0.8,
    }
    data.update(overrides)
    return data


def test_missing_stl_raises_cfdrunerror():
    try:
        run_half_car_cfd("does_not_exist.stl")
    except CFDRunError:
        return
    raise AssertionError("Expected CFDRunError")


def test_non_watertight_stl_raises():
    path = _open_stl_path()
    try:
        try:
            run_half_car_cfd(path)
        except CFDRunError:
            return
        raise AssertionError("Expected CFDRunError")
    finally:
        Path(path).unlink(missing_ok=True)


def test_watertight_tetrahedron_passes_manifold_check():
    original = cfd_wrapper._invoke_openfoam_pipeline
    cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: _fake_dict()
    path = _tetrahedron_path()
    try:
        half, health = run_half_car_cfd(path)
        assert isinstance(half, HalfCarQuantities)
        assert half.D20 == 5.0
        assert half.L == 1.2
        assert half.A == 0.01
        assert half.pitching_moment_half == 0.05
        assert health.converged is True
    finally:
        cfd_wrapper._invoke_openfoam_pipeline = original
        Path(path).unlink(missing_ok=True)


def test_negative_volume_cells_raises():
    original = cfd_wrapper._invoke_openfoam_pipeline
    cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: _fake_dict(
        negative_volume_cells=1
    )
    path = _tetrahedron_path()
    try:
        try:
            run_half_car_cfd(path)
        except CFDRunError:
            return
        raise AssertionError("Expected CFDRunError")
    finally:
        cfd_wrapper._invoke_openfoam_pipeline = original
        Path(path).unlink(missing_ok=True)


def test_non_convergence_does_not_raise_but_flags_health_report():
    original = cfd_wrapper._invoke_openfoam_pipeline
    cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: _fake_dict(
        residual_final=1e-2
    )
    path = _tetrahedron_path()
    try:
        half, health = run_half_car_cfd(path)
        assert half.D20 == 5.0
        assert health.converged is False
    finally:
        cfd_wrapper._invoke_openfoam_pipeline = original
        Path(path).unlink(missing_ok=True)


def test_determinism_of_dict_to_dataclass_packaging():
    original = cfd_wrapper._invoke_openfoam_pipeline
    cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: _fake_dict()
    path = _tetrahedron_path()
    try:
        half_1, health_1 = run_half_car_cfd(path)
        half_2, health_2 = run_half_car_cfd(path)
        assert half_1 == half_2
        assert health_1 == health_2
    finally:
        cfd_wrapper._invoke_openfoam_pipeline = original
        Path(path).unlink(missing_ok=True)


def test_binary_stl_rejected_with_clear_error():
    """Binary STL files should give CFDRunError, not UnicodeDecodeError."""
    import struct
    f = tempfile.NamedTemporaryFile("wb", suffix=".stl", delete=False)
    with f:
        f.write(b"\x00" * 84)
        f.write(struct.pack("<f", 0.0) * 3)
        f.write(struct.pack("<f", 0.0) * 3)
        f.write(struct.pack("<f", 1.0) * 3)
        f.write(struct.pack("<f", 0.0) * 3)
        f.write(b"\x00\x00")
    try:
        try:
            run_half_car_cfd(f.name)
        except CFDRunError:
            return
        raise AssertionError("Expected CFDRunError for binary STL")
    finally:
        Path(f.name).unlink(missing_ok=True)


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
