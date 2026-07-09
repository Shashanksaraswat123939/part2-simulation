import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjoint_contract import package_gradient_bundle
from candidate_record import CandidateRecord, read_candidate_record, write_candidate_record
from physics_contract import ComponentMassCOM, FullCarMassCOM, FullCarQuantities


def _mass_report():
    return FullCarMassCOM(
        total_mass_kg=0.1,
        com_x_m=0.05,
        com_y_m=0.0,
        com_z_m=0.02,
        components=(ComponentMassCOM("body", 0.1, 0.05, 0.0, 0.02),),
    )


def _record(**overrides):
    mass = _mass_report()
    data = {
        "candidate_id": "cand_001",
        "W_mm": 130.0,
        "d_halo_mm": 5.0,
        "phi_grid_snapshot_paths": {
            "nose": "nose.npy",
            "sidepod": "sidepod.npy",
            "rearpod": "rearpod.npy",
            "main_body": "main_body.npy",
        },
        "stl_path": "car.stl",
        "mass_report": mass,
        "com_report": mass,
        "cfd_force_report": FullCarQuantities(D20=10.0, L=2.0, Cm=0.1, A=0.02),
        "T_raw": 1.25,
        "T_penalized": 1.30,
        "gradients": package_gradient_bundle(1.0, 2.0, 3.0, 0.0, None),
        "adjoint_sensitivity_field_path": "sensitivities.dat",
        "setup_logs": "ok",
        "failure_reason": None,
        "lifecycle_state": "valid_simulated",
    }
    data.update(overrides)
    return CandidateRecord(**data)


def _assert_record_fields_equal(left, right):
    assert left.candidate_id == right.candidate_id
    assert left.W_mm == right.W_mm
    assert left.d_halo_mm == right.d_halo_mm
    assert left.phi_grid_snapshot_paths == right.phi_grid_snapshot_paths
    assert left.stl_path == right.stl_path
    assert left.mass_report == right.mass_report
    assert left.com_report == right.com_report
    assert left.cfd_force_report == right.cfd_force_report
    assert left.T_raw == right.T_raw
    assert left.T_penalized == right.T_penalized
    assert left.gradients == right.gradients
    assert left.adjoint_sensitivity_field_path == right.adjoint_sensitivity_field_path
    assert left.setup_logs == right.setup_logs
    assert left.failure_reason == right.failure_reason
    assert left.lifecycle_state == right.lifecycle_state


def test_round_trip_valid_candidate():
    with tempfile.TemporaryDirectory() as d:
        original = _record()
        path = write_candidate_record(original, d)
        loaded = read_candidate_record(path)
        _assert_record_fields_equal(original, loaded)


def test_round_trip_cfd_failed_candidate():
    with tempfile.TemporaryDirectory() as d:
        gradients = package_gradient_bundle(0.0, 0.0, 0.0, 0.0, None)
        original = _record(
            candidate_id="cand_failed",
            T_raw=None,
            T_penalized=None,
            failure_reason="mesh generation failed: non-manifold edges",
            gradients=gradients,
            lifecycle_state="CFD_failed",
        )
        path = write_candidate_record(original, d)
        loaded = read_candidate_record(path)
        _assert_record_fields_equal(original, loaded)
        assert loaded.T_raw is None
        assert loaded.T_penalized is None


def test_invalid_lifecycle_state_raises_on_read():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.json"
        data = {
            "candidate_id": "bad",
            "W_mm": 1.0,
            "d_halo_mm": 1.0,
            "phi_grid_snapshot_paths": {},
            "stl_path": "bad.stl",
            "mass_report": {"total_mass_kg": 1.0, "com_x_m": 0.0, "com_y_m": 0.0, "com_z_m": 0.0, "components": []},
            "com_report": {"total_mass_kg": 1.0, "com_x_m": 0.0, "com_y_m": 0.0, "com_z_m": 0.0, "components": []},
            "cfd_force_report": {"D20": 1.0, "L": 1.0, "Cm": 1.0, "A": 1.0},
            "T_raw": None,
            "T_penalized": None,
            "gradients": {},
            "adjoint_sensitivity_field_path": None,
            "setup_logs": "",
            "failure_reason": None,
            "lifecycle_state": "bogus_state",
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            read_candidate_record(str(path))
        except ValueError:
            return
        raise AssertionError("Expected ValueError")


def test_write_rejects_non_serializable_field():
    with tempfile.TemporaryDirectory() as d:
        record = _record(setup_logs=np.asarray([1.0, 2.0]))
        try:
            write_candidate_record(record, d)
        except TypeError:
            return
        raise AssertionError("Expected TypeError")


def test_phi_snapshot_paths_stored_as_strings_not_arrays():
    with tempfile.TemporaryDirectory() as d:
        path = write_candidate_record(_record(), d)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert all(isinstance(value, str) for value in data["phi_grid_snapshot_paths"].values())


def test_path_traversal_in_candidate_id_rejected():
    """Path traversal via candidate_id containing ../ must be rejected."""
    mass = _mass_report()
    record = _record(candidate_id="../../evil")
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidate_record(record, d)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for path traversal candidate_id")


def test_nan_float_rejected_in_json():
    """NaN and Infinity must not be serializable to JSON (allow_nan=False)."""
    record = _record(W_mm=float("nan"))
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidate_record(record, d)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for NaN in record")


def test_inf_float_rejected_in_json():
    """Infinity must not be serializable to JSON."""
    record = _record(T_raw=float("inf"))
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidate_record(record, d)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for Inf in record")


def test_setup_logs_size_limit_enforced():
    """setup_logs longer than 1MB must be rejected at construction."""
    try:
        _record(setup_logs="x" * 2_000_000)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for oversized setup_logs")




def test_dotdot_substring_in_candidate_id_accepted():
    """A candidate_id containing '..' as a substring (not a path segment)
    should be accepted, since it cannot cause path traversal."""
    record = _record(candidate_id="cand..001")
    with tempfile.TemporaryDirectory() as d:
        path = write_candidate_record(record, d)
        loaded = read_candidate_record(path)
        assert loaded.candidate_id == "cand..001"


def test_dot_candidate_id_rejected():
    """candidate_id '.' must be rejected (current directory)."""
    record = _record(candidate_id=".")
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidate_record(record, d)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for '.' candidate_id")


def test_dotdot_candidate_id_rejected():
    """candidate_id '..' must be rejected (parent directory)."""
    record = _record(candidate_id="..")
    with tempfile.TemporaryDirectory() as d:
        try:
            write_candidate_record(record, d)
        except ValueError:
            return
        raise AssertionError("Expected ValueError for '..' candidate_id")


def test_invalid_lifecycle_state_rejected_at_construction():
    """CandidateRecord construction must reject invalid lifecycle_state,
    not just at read time."""
    try:
        _record(lifecycle_state="totally_bogus_state")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid lifecycle_state at construction")

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
