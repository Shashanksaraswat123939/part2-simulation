import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_end_to_end_pipeline_with_mocks():
    import cfd_wrapper
    from adjoint_contract import compute_adjoint_objective_weight, package_gradient_bundle
    from candidate_record import CandidateRecord, read_candidate_record, write_candidate_record
    from mass_com_ingest import FixedHardwareSpec, ingest_mass_com
    from physics_contract import ComponentMassCOM
    from race_objective import BuildSettings, build_smooth_sheet_model
    from race_objective_adapter import race_value_and_grad_guarded

    def write_tetrahedron():
        triangles = [
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
            ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ]
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

    def write_thrust_csv():
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

    def finite_float_values(obj):
        if isinstance(obj, float):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from finite_float_values(value)
        elif isinstance(obj, tuple):
            for value in obj:
                yield from finite_float_values(value)
        elif hasattr(obj, "__dataclass_fields__"):
            for name in obj.__dataclass_fields__:
                yield from finite_float_values(getattr(obj, name))

    stl_path = write_tetrahedron()
    csv_path = write_thrust_csv()
    original = cfd_wrapper._invoke_openfoam_pipeline
    cfd_wrapper._invoke_openfoam_pipeline = lambda stl_path, case_dir: {
        "D20_half": 5.0,
        "L_half": 1.0,
        "A_half": 0.01,
        "pitching_moment_half": 0.05,
        "residual_final": 1e-5,
        "negative_volume_cells": 0,
        "y_plus_min": 0.4,
        "y_plus_max": 2.0,
        "courant_max": 0.8,
    }
    try:
        mass_report = ingest_mass_com(
            [ComponentMassCOM("body", 0.050, 0.05, 0.0, 0.030)],
            FixedHardwareSpec(
                co2_cartridge_mass_kg=0.023,
                co2_cartridge_com=(0.02, 0.0, 0.025),
                rear_wing_mass_kg=0.010,
                rear_wing_com=(0.12, 0.0, 0.040),
                wheels_axles_mass_kg=0.020,
                wheels_axles_com=(0.06, 0.0, 0.015),
            ),
        )
        half, _health = cfd_wrapper.run_half_car_cfd(stl_path)
        full = half.to_full_car()
        model = build_smooth_sheet_model(csv_path, BuildSettings(n_basis=5, n_steps=60))
        params = np.array([full.D20, mass_report.total_mass_kg, 0.02, 1e-7, 1.0, mass_report.com_z_m, full.L, mass_report.com_x_m], dtype=np.float64)
        T_raw, T_penalized, adapted_grads = race_value_and_grad_guarded(params, model)
        w_D20 = compute_adjoint_objective_weight(params, model)
        gradients = package_gradient_bundle(
            w_D20,
            adapted_grads["dT_dmass"],
            adapted_grads["dT_dh_com"],
            adapted_grads["dT_dx_com"],
            None,
        )
        with tempfile.TemporaryDirectory() as d:
            record = CandidateRecord(
                candidate_id="integration",
                W_mm=130.0,
                d_halo_mm=5.0,
                phi_grid_snapshot_paths={"nose": "nose.npy", "sidepod": "sidepod.npy", "rearpod": "rearpod.npy", "main_body": "main_body.npy"},
                stl_path=stl_path,
                mass_report=mass_report,
                com_report=mass_report,
                cfd_force_report=full,
                T_raw=T_raw,
                T_penalized=T_penalized,
                gradients=gradients,
                adjoint_sensitivity_field_path=None,
                setup_logs="integration ok",
                failure_reason=None,
                lifecycle_state="valid_simulated",
            )
            path = write_candidate_record(record, d)
            loaded = read_candidate_record(path)
            assert loaded.candidate_id == record.candidate_id
            assert loaded.T_penalized >= loaded.T_raw
            for value in finite_float_values(loaded):
                assert math.isfinite(value)
    finally:
        cfd_wrapper._invoke_openfoam_pipeline = original
        Path(stl_path).unlink(missing_ok=True)
        Path(csv_path).unlink(missing_ok=True)


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
