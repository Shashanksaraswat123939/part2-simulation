"""Unit tests for openfoam_case.py.

These cover everything that does NOT require a live OpenFOAM solve: the
geometry math, dictionary generation, turbulence inlet values, and the log/
force/moment/y+ parsers (against crafted ESI-format fixtures). The actual
subprocess solve is exercised on a machine with ESI OpenFOAM installed.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import openfoam_case as oc
from openfoam_case import OpenFOAMRunConfig


# ---------------------------------------------------------------------------
# STL fixtures (ASCII, explicit winding so normals are deterministic)
# ---------------------------------------------------------------------------

def _write_stl(triangles, name="fixture"):
    f = tempfile.NamedTemporaryFile("w", suffix=".stl", delete=False, encoding="utf-8")
    with f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            f.write("  facet normal 0 0 0\n    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")
    return f.name


# Front panel at x=0 (outward normal -x → front-facing) + back panel at x=1
# (outward normal +x → not front-facing). Frontal silhouette area = 1.0 m^2.
_PANELS = [
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),   # x=0, normal -x
    ((0, 1, 0), (0, 0, 1), (0, 1, 1)),   # x=0, normal -x
    ((1, 0, 0), (1, 1, 0), (1, 0, 1)),   # x=1, normal +x
    ((1, 1, 0), (1, 1, 1), (1, 0, 1)),   # x=1, normal +x
]


def test_config_validation():
    OpenFOAMRunConfig()  # defaults OK
    for bad in (
        dict(turbulence_model="spalart"),
        dict(resolution="ultra"),
        dict(reference_speed_mps=0.0),
    ):
        try:
            OpenFOAMRunConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_stl_bounds():
    p = _write_stl(_PANELS)
    try:
        (mn, mx) = oc.stl_bounds(p)
        assert mn == (0.0, 0.0, 0.0), mn
        assert mx == (1.0, 1.0, 1.0), mx
    finally:
        Path(p).unlink(missing_ok=True)


def test_frontal_area_only_counts_front_facing():
    p = _write_stl(_PANELS)
    try:
        a = oc.compute_frontal_area_half(p)
        assert abs(a - 1.0) < 1e-9, a  # only the x=0 panel counts
    finally:
        Path(p).unlink(missing_ok=True)


def test_frontal_area_ignores_back_facing_only():
    # Just the back panel (normal +x) → zero frontal area.
    p = _write_stl(_PANELS[2:])
    try:
        assert oc.compute_frontal_area_half(p) == 0.0
    finally:
        Path(p).unlink(missing_ok=True)


def test_domain_box_clamps_symmetry_plane_and_encloses():
    bounds = ((0.0, 0.0, 0.0), (0.2, 0.05, 0.08))
    box_min, box_max = oc.domain_box(bounds)
    assert box_min[1] == 0.0, "y_min must sit on the symmetry plane"
    assert box_min[0] < 0.0 and box_max[0] > 0.2, "must enclose the car in x"
    assert box_max[1] > 0.05 and box_max[2] > 0.08


def test_location_in_mesh_is_outside_car_and_inside_box():
    bounds = ((0.0, 0.0, 0.0), (0.2, 0.05, 0.08))
    loc = oc.location_in_mesh(bounds)
    box_min, box_max = oc.domain_box(bounds)
    assert loc[0] < 0.0, "seed point should be upstream of the car (in fluid)"
    for i in range(3):
        assert box_min[i] <= loc[i] <= box_max[i]


def test_turbulence_inlet_values_positive_and_formula():
    cfg = OpenFOAMRunConfig(turbulence_model="kOmegaSST", reference_speed_mps=20.0)
    k, omega, nut = oc.turbulence_inlet_values(cfg, ref_length_m=0.2)
    assert k > 0 and omega > 0 and nut > 0
    # k = 1.5 (I U)^2 = 1.5 (0.05*20)^2 = 1.5
    assert abs(k - 1.5) < 1e-9, k


def test_blockmesh_dict_has_symmetry_and_cell_counts():
    d = oc.build_blockmesh_dict((-1.0, 0.0, -1.0), (2.0, 1.0, 1.0), cell_size_m=0.5)
    assert "symmetryPlane" in d
    assert "blocks" in d and "hex (0 1 2 3 4 5 6 7)" in d
    assert "lowerWall" in d and "inlet" in d and "outlet" in d


def test_snappy_layers_toggle():
    loc = (0.0, 0.1, 0.0)
    with_layers = oc.build_snappy_dict("car.stl", loc, (3, 4), add_layers=True)
    without = oc.build_snappy_dict("car.stl", loc, (3, 4), add_layers=False)
    assert "addLayers       true" in with_layers and "nSurfaceLayers" in with_layers
    assert "addLayers       false" in without


def test_control_dict_has_force_objects_and_cofr():
    cfg = OpenFOAMRunConfig(moment_reference_point_m=(0.1, 0.0, 0.02), max_iterations=1500)
    d = oc.build_control_dict(cfg, ref_area_half=0.01)
    assert "forces" in d and "forceCoeffs" in d and "yPlus" in d
    assert "CofR            (0.1 0.0 0.02)" in d
    assert "endTime         1500" in d
    assert "pitchAxis       (0 1 0)" in d


def test_surface_feature_extract_dict_matches_esi_contract():
    # Verified against openfoam2412's own tutorials: the utility is
    # `surfaceFeatureExtract` (NOT `surfaceFeatures`), and the dict is keyed
    # by the STL filename with extractFromSurfaceCoeffs, not a flat
    # surfaces()/includedAngle block.
    body = oc._SURFACE_FEATURE_EXTRACT_BODY
    assert "car.stl" in body
    assert "extractFromSurfaceCoeffs" in body
    assert "includedAngle" in body


def test_transport_and_turbulence_properties():
    lam = oc.build_turbulence_properties(OpenFOAMRunConfig(turbulence_model="laminar"))
    sst = oc.build_turbulence_properties(OpenFOAMRunConfig(turbulence_model="kOmegaSST"))
    assert "laminar" in lam
    assert "kOmegaSST" in sst and "RAS" in sst
    tp = oc.build_transport_properties(OpenFOAMRunConfig(air_density_kgm3=1.225))
    assert "nu" in tp and "Newtonian" in tp


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_negative_volume_cells():
    assert oc.parse_negative_volume_cells("Checking geometry...\n  Mesh OK.\n") == 0
    assert oc.parse_negative_volume_cells("  ***Error in mesh: 5 negative volume cells") == 5
    assert oc.parse_negative_volume_cells("Number of negative volume cells: 12") == 12


def test_parse_final_p_residual_takes_last():
    log = (
        "Time = 1\n"
        "GAMG:  Solving for p, Initial residual = 0.5, Final residual = 0.004, No Iterations 10\n"
        "Time = 500\n"
        "GAMG:  Solving for p, Initial residual = 8.3e-05, Final residual = 9e-07, No Iterations 3\n"
    )
    assert abs(oc.parse_final_p_residual(log) - 8.3e-05) < 1e-12
    try:
        oc.parse_final_p_residual("no residuals here")
    except ValueError:
        return
    raise AssertionError("expected ValueError when no residual present")


def test_parse_max_courant():
    log = "Courant Number mean: 0.01 max: 0.5\nCourant Number mean: 0.02 max: 0.85\n"
    assert abs(oc.parse_max_courant(log) - 0.85) < 1e-12
    assert oc.parse_max_courant("steady run, no courant line") is None


def test_parse_yplus_log_and_dat():
    # Real ESI (v2412, verified against an actual solver run) uses "=", not
    # ":", and reports every wall patch — the parser must target the "car"
    # patch specifically and ignore e.g. lowerWall.
    log_form = (
        "yPlus write:\n"
        "    patch lowerWall y+ : min = 40.0, max = 500.0, average = 200.0\n"
        "    patch car y+ : min = 0.5, max = 3.2, average = 1.1\n"
    )
    assert oc.parse_yplus_range(log_form) == (0.5, 3.2)

    dat_form = (
        "# Time\tpatch\tmin\tmax\taverage\n"
        "7\tlowerWall\t40.0\t500.0\t200.0\n"
        "7\tcar\t1.0\t9.0\t3.0\n"
        "14\tlowerWall\t41.0\t510.0\t201.0\n"
        "14\tcar\t0.4\t2.8\t1.0\n"
    )
    assert oc.parse_yplus_range(dat_form) == (0.4, 2.8)  # latest time, car patch only


def test_parse_total_vector_dat():
    force_dat = (
        "# Forces\n# CofR : (0 0 0)\n# Time forces\n"
        "1 (10.0 0.0 -2.0) (8.0 0.0 -1.5) (2.0 0.0 -0.5)\n"
        "2 (12.5 0.1 -3.0) (10.0 0.1 -2.0) (2.5 0.0 -1.0)\n"
    )
    fx, fy, fz = oc.parse_total_vector_dat(force_dat)
    assert (abs(fx - 12.5) < 1e-9 and abs(fy - 0.1) < 1e-9 and abs(fz + 3.0) < 1e-9)


def test_read_force_and_moment_from_postprocessing():
    with tempfile.TemporaryDirectory() as d:
        fdir = Path(d) / "postProcessing" / "forces" / "0"
        fdir.mkdir(parents=True)
        (fdir / "force.dat").write_text(
            "# Time forces\n2 (12.5 0.1 -3.0) (10 0 -2) (2.5 0.1 -1)\n", encoding="utf-8"
        )
        (fdir / "moment.dat").write_text(
            "# Time moments\n2 (0.0 1.5 0.0) (0 1 0) (0 0.5 0)\n", encoding="utf-8"
        )
        force, moment = oc.read_force_and_moment(d)
        assert abs(force[0] - 12.5) < 1e-9 and abs(force[2] + 3.0) < 1e-9
        assert abs(moment[1] - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# Case assembly + environment
# ---------------------------------------------------------------------------

def test_build_case_generates_expected_files():
    stl = _write_stl(_PANELS)
    try:
        with tempfile.TemporaryDirectory() as d:
            run_dir = str(Path(d) / "run")
            meta = oc.build_case(run_dir, stl, OpenFOAMRunConfig(turbulence_model="kOmegaSST"))
            run = Path(run_dir)
            for rel in (
                "system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict",
                "system/fvSchemes", "system/fvSolution", "system/decomposeParDict",
                "system/surfaceFeatureExtractDict",
                "constant/transportProperties", "constant/turbulenceProperties",
                "constant/triSurface/car.stl", "0/U", "0/p", "0/k", "0/omega", "0/nut",
            ):
                assert (run / rel).is_file(), f"missing {rel}"
            # STL solid renamed to 'car' so the snappy patch name is deterministic.
            assert "solid car" in (run / "constant/triSurface/car.stl").read_text()
            assert abs(meta["frontal_area_half"] - 1.0) < 1e-9
    finally:
        Path(stl).unlink(missing_ok=True)


def test_laminar_case_has_no_turbulence_fields():
    stl = _write_stl(_PANELS)
    try:
        with tempfile.TemporaryDirectory() as d:
            run_dir = str(Path(d) / "run")
            oc.build_case(run_dir, stl, OpenFOAMRunConfig(turbulence_model="laminar"))
            assert not (Path(run_dir) / "0" / "k").exists()
            assert not (Path(run_dir) / "0" / "nut").exists()
    finally:
        Path(stl).unlink(missing_ok=True)


def test_find_bashrc_missing_returns_none():
    assert oc.find_openfoam_bashrc("/definitely/not/a/real/bashrc") is None


def test_invoke_raises_when_openfoam_absent():
    stl = _write_stl(_PANELS)
    try:
        with tempfile.TemporaryDirectory() as d:
            try:
                oc.invoke(stl, d, bashrc="/definitely/not/a/real/bashrc")
            except oc.OpenFOAMNotFoundError:
                return
            raise AssertionError("expected OpenFOAMNotFoundError")
    finally:
        Path(stl).unlink(missing_ok=True)


if __name__ == "__main__":
    fns = [f for f in dir(sys.modules[__name__]) if f.startswith("test_")]
    passed, failed = 0, 0
    for name in fns:
        try:
            globals()[name]()
            print("PASS", name)
            passed += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", name, "->", repr(e))
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
