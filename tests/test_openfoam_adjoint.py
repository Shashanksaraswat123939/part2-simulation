"""Unit tests for openfoam_adjoint.py.

Covers everything that does NOT require a live OpenFOAM solve: config
validation, dict generation, and the sensitivity-file/points-file parsers.
The parser fixtures below are built to match the REAL formats confirmed by
actually running adjointOptimisationFoam (openfoam2412, kOmegaSST/
adjointkOmegaSST) end-to-end during development of this module — including
two real format surprises that were wrong on first guess and only caught by
execution: (1) the sensitivity file is named `pointSensNormal<solverName>...`
and is a full pointScalarField over every mesh point (mostly zeros), not a
small list of (point, value) pairs on the design patch; (2) primal solve
needed `consistent yes` (SIMPLEC) or it diverged to NaN within ~55 iterations
on plain SIMPLE with the schemes used here.

The full pipeline (case generation through a real solve) was verified once
via a live WSL run against a tiny hand-built watertight half-box STL; see
project memory / conversation history for that run's output. These tests
cover the pure functions only.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import openfoam_adjoint as oa
from openfoam_adjoint import AdjointRunConfig


def test_config_validation():
    AdjointRunConfig()  # defaults OK
    for bad in (
        dict(resolution="ultra"),
        dict(reference_speed_mps=0.0),
        dict(primal_iters=0),
        dict(adjoint_iters=-1),
    ):
        try:
            AdjointRunConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_as_forward_config_forces_komega_sst():
    cfg = AdjointRunConfig(reference_speed_mps=15.0, resolution="fine")
    fwd = cfg.as_forward_config()
    assert fwd.turbulence_model == "kOmegaSST"
    assert fwd.reference_speed_mps == 15.0
    assert fwd.resolution == "fine"


def test_adjoint_ras_properties_names_komega_sst():
    body = oa.build_adjoint_ras_properties()
    assert "adjointkOmegaSST" in body
    assert "adjointTurbulence on" in body
    # No SA-specific coeffs sub-dict for this pairing (verified against the
    # naca0012 kOmegaSST tutorial's adjointRASProperties).
    assert "adjointSpalartAllmarasCoeffs" not in body


def test_adjoint_control_dict_end_time_is_sum_of_iters():
    cfg = AdjointRunConfig(primal_iters=300, adjoint_iters=700)
    d = oa.build_adjoint_control_dict(cfg)
    assert "application     adjointOptimisationFoam;" in d
    assert "endTime         1000;" in d
    assert "writeFormat     ascii;" in d  # must NOT be binary -- we parse text output


def test_optimisation_dict_has_single_sensitivity_type_and_consistent_solve():
    cfg = AdjointRunConfig(primal_iters=50, adjoint_iters=60)
    d = oa.build_optimisation_dict(cfg, ref_area_half=0.004)
    assert "optimisationManager singleRun;" in d
    assert "sensitivityType    surfacePoints;" in d
    assert d.count("sensitivityType") == 1  # no duplicate-key bug (regression: an
    # earlier draft had both "sensitivityType single;" and
    # "sensitivityType surfacePoints;" in the same dict block)
    assert "patches    (car);" in d
    assert "direction  (1 0 0);" in d  # drag = +x
    assert "weight     1.;" in d  # unweighted -- scaling applied in Python
    # consistent yes (SIMPLEC) is required: plain SIMPLE diverged to NaN
    # within ~55 iterations in a live test of this exact case/scheme set.
    assert d.count("consistent yes;") == 2  # primal AND adjoint solversControls


def test_adjoint_fields_cover_all_seven_patches():
    with tempfile.TemporaryDirectory() as d:
        zero_dir = Path(d)
        oa._write_adjoint_fields(zero_dir)
        for name in ("Ua", "pa", "ka", "wa"):
            text = (zero_dir / name).read_text()
            for patch in ("inlet", "outlet", "symmetry", "outer", "upperWall", "lowerWall", "car"):
                assert patch in text, f"{name} missing patch {patch}"


def test_fv_solution_has_ma_solver():
    # Regression: an earlier version crashed at the mesh-movement/eikonal
    # sensitivity step with "Entry 'ma' not found in dictionary
    # system/fvSolution/solvers" -- verified live, fixed against the
    # sensitivityMaps/motorBike reference's fvSolution.
    assert "ma" in oa._FV_SOLUTION_ADJOINT
    assert "preconditioner   DIC;" in oa._FV_SOLUTION_ADJOINT


# ---------------------------------------------------------------------------
# Parsers -- fixtures match the REAL confirmed formats
# ---------------------------------------------------------------------------

_REAL_POINTS_FIXTURE = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       vectorField;
    location    "constant/polyMesh";
    object      points;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


5
(
(-0.6 0 -0.32)
(0.01 0.02 0.03)
(0.011 0.021 0.031)
(1.8 0.3 0.48)
(0.5 0.1 0.1)
)

// ************************************************************************* //
"""

_REAL_SENSITIVITY_FIXTURE = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       pointScalarField;
    location    "120";
    object      pointSensNormaladjS1ESI;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   nonuniform List<scalar>
5
(
0
1.5e10
2.3e10
0
0
)
;

boundaryField
{
    car
    {
        type            calculated;
        value           nonuniform 0();
    }
}

// ************************************************************************* //
"""


def test_parse_foam_vector_list_matches_real_points_format():
    pts = oa._parse_foam_vector_list(_REAL_POINTS_FIXTURE)
    assert pts.shape == (5, 3)
    assert np.allclose(pts[0], [-0.6, 0.0, -0.32])
    assert np.allclose(pts[4], [0.5, 0.1, 0.1])


def test_parse_foam_scalar_list_matches_real_sensitivity_format():
    vals = oa._parse_foam_scalar_list(_REAL_SENSITIVITY_FIXTURE)
    assert vals.shape == (5,)
    assert np.allclose(vals, [0.0, 1.5e10, 2.3e10, 0.0, 0.0])


def test_find_sensitivity_file_matches_real_naming():
    # Regression: the file is named pointSensNormal<solverName><suffix>, e.g.
    # pointSensNormaladjS1ESI -- NOT "pointSensitivity" or "SensitivityMap",
    # which an earlier version of find_sensitivity_file searched for and
    # would never have matched the real output.
    with tempfile.TemporaryDirectory() as d:
        run = Path(d)
        (run / "60").mkdir()
        (run / "120").mkdir()
        (run / "60" / "pointSensNormaladjS1ESI").write_text("old")
        (run / "120" / "pointSensNormaladjS1ESI").write_text("new")
        import time
        time.sleep(0.01)
        (run / "120" / "pointSensNormaladjS1ESI").write_text("new")  # bump mtime
        found = oa.find_sensitivity_file(str(run))
        assert found.parent.name == "120"


def test_parse_sensitivity_points_pairs_by_index():
    pts = oa._parse_foam_vector_list(_REAL_POINTS_FIXTURE)
    sens_points, sens_values = oa.parse_sensitivity_points(_REAL_SENSITIVITY_FIXTURE, pts)
    assert sens_points.shape == (5, 3)
    assert np.allclose(sens_values, [0.0, 1.5e10, 2.3e10, 0.0, 0.0])


def test_parse_sensitivity_points_length_mismatch_raises():
    pts = oa._parse_foam_vector_list(_REAL_POINTS_FIXTURE)[:3]  # wrong length
    try:
        oa.parse_sensitivity_points(_REAL_SENSITIVITY_FIXTURE, pts)
    except ValueError:
        return
    raise AssertionError("expected ValueError on point/value count mismatch")


def _write_stl(triangles):
    f = tempfile.NamedTemporaryFile("w", suffix=".stl", delete=False, encoding="utf-8")
    with f:
        f.write("solid car\n")
        for tri in triangles:
            f.write("  facet normal 0 0 0\n    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid car\n")
    return f.name


def test_map_sensitivity_to_stl_vertices_nearest_neighbour():
    # Two STL vertices, each very close to one of two sensitivity points with
    # distinct values -- nearest-neighbour must pick the right one, not just
    # whichever appears first in the array.
    stl = _write_stl([
        ((0.0001, 0.0, 0.0), (0.9999, 1.0, 0.0), (0.0001, 1.0, 0.0)),
    ])
    try:
        sens_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [10.0, 10.0, 10.0]])
        sens_values = np.array([100.0, 200.0, 300.0, 999999.0])
        result = oa.map_sensitivity_to_stl_vertices(stl, sens_points, sens_values, max_distance_m=0.01)
        assert result.shape == (3,)
        assert result[0] == 100.0  # vertex (0.0001,0,0) nearest to (0,0,0)
        assert result[1] == 200.0  # vertex (0.9999,1,0) nearest to (1,1,0)
        assert result[2] == 300.0  # vertex (0.0001,1,0) nearest to (0,1,0)
    finally:
        Path(stl).unlink(missing_ok=True)


def test_map_sensitivity_raises_when_vertex_unmatched():
    stl = _write_stl([
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ])
    try:
        # sensitivity points are far from all STL vertices
        sens_points = np.array([[500.0, 500.0, 500.0]])
        sens_values = np.array([1.0])
        try:
            oa.map_sensitivity_to_stl_vertices(stl, sens_points, sens_values, max_distance_m=0.01)
        except ValueError:
            return
        raise AssertionError("expected ValueError when no nearby sensitivity point exists")
    finally:
        Path(stl).unlink(missing_ok=True)


def test_find_openfoam_bashrc_missing_returns_none_reused_from_case_module():
    # search_roots=[] + cleared env vars (2026-07-16 fix): see
    # test_openfoam_case.py's identical fix for why an invalid explicit path
    # alone is not sufficient on a machine with real ESI OpenFOAM installed.
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FOAM_BASHRC", None)
        os.environ.pop("WM_PROJECT_DIR", None)
        assert oa.oc.find_openfoam_bashrc("/definitely/not/real", search_roots=[]) is None


def test_invoke_adjoint_raises_when_openfoam_absent():
    # search_roots=[] + cleared env vars: without this, on a machine with
    # real ESI OpenFOAM installed, invoke_adjoint() silently finds it and
    # launches a genuine primal+adjoint solve instead of raising (same class
    # of bug as test_openfoam_case.py's invoke() test, fixed 2026-07-16).
    stl = _write_stl([((0, 0, 0), (1, 0, 0), (0, 1, 0))])
    try:
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FOAM_BASHRC", None)
                os.environ.pop("WM_PROJECT_DIR", None)
                try:
                    oa.invoke_adjoint(
                        stl, d, bashrc="/definitely/not/real", search_roots=[]
                    )
                except oa.oc.OpenFOAMNotFoundError:
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
