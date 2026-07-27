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


def test_adjoint_ras_model_is_frozen_turbulence():
    """adjointLaminar, NOT adjointkOmegaSST.

    adjointkOmegaSST crashed with SIGFPE inside its own divDevReff at adjoint
    iteration 277 of 500, after a healthy primal and healthy pa/Ua/ka/wa. On the
    installed v2412, adjointLaminar is used by 14 shipped tutorials against
    adjointkOmegaSST's 2, and no tutorial freezes via `adjointTurbulence off`.

    Frozen turbulence assumes d(nu_t)/d(shape) = 0. That is nearly free here:
    combine_gradients normalises the aero field to unit RMS, so only the
    spatial pattern and sign survive into the level-set update anyway.
    """
    body = oa.build_adjoint_ras_properties()
    assert "adjointLaminar" in body
    assert "adjointkOmegaSST" not in body, (
        "adjointkOmegaSST SIGFPEs in divDevReff on this geometry")
    assert "adjointTurbulence on" in body
    assert "adjointSpalartAllmarasCoeffs" not in body


def test_primal_stays_komega_sst():
    """Only the ADJOINT turbulence is frozen -- the primal still produces the
    drag VALUE and must keep the model the forward solve validated."""
    assert oa.AdjointRunConfig().as_forward_config().turbulence_model == "kOmegaSST"
    assert "kOmegaSST" in oc_turb(), "primal turbulenceProperties lost kOmegaSST"


def oc_turb():
    import openfoam_case as _oc
    return _oc.build_turbulence_properties(
        oa.AdjointRunConfig().as_forward_config())


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
    # v2412 schema, VERIFIED against the shipped tutorials on a real install
    # (2026-07-26). sensitivityMaps/motorBike -- the external-aero case closest
    # to ours -- uses:
    #     optimisation { designVariables { sensitivityType surfacePoints; ... } }
    # There is no `sensitivities` block in v2412 at all. An intermediate version
    # of this test asserted exactly that wrong shape; the preflight caught it.
    assert "designVariables" in d
    assert "sensitivities\n" not in d
    assert f"sensitivityType    {oa.SENSITIVITY_TYPE};" in d
    assert oa.SENSITIVITY_TYPE == "surfacePoints"
    # Without this the solver runs to completion and writes NO sensitivity field.
    assert "computeSensitivities   true;" in d
    assert "patches    (car);" in d
    assert "direction  (1 0 0);" in d  # drag = +x
    assert "weight     1.;" in d  # unweighted -- scaling applied in Python
    # consistent yes (SIMPLEC) is required: plain SIMPLE diverged to NaN
    # within ~55 iterations in a live test of this exact case/scheme set.
    assert d.count("consistent yes;") == 2  # primal AND adjoint solversControls



def test_turbulence_transport_is_upwind_not_linearUpwind():
    """Regression guard for the SIGFPE on the first medium-resolution solve.

    linearUpwind on k/omega undershoots to negative omega near walls; bounding
    clips it, but the SST blending functions multiply k/omega and that is where
    Foam::multiply threw. omega reached 2.6e+135 and adjointOptimisationFoam
    died in solvePrimalEquations before ever reaching the adjoint.

    The forward case solves the same model on the same geometry and converges
    using `bounded Gauss upwind` for both, so the adjoint must match it.
    """
    import re
    s = oa._FV_SCHEMES_ADJOINT
    for field in ("div(phi,k)", "div(phi,omega)", "div(-phi,ka)", "div(-phi,wa)"):
        m = re.search(re.escape(field) + r"\s+([^;]+);", s)
        assert m, f"{field} missing from adjoint fvSchemes"
        scheme = m.group(1)
        assert "linearUpwind" not in scheme, (
            f"{field} uses {scheme.strip()!r}; linearUpwind on turbulence "
            "transport diverged to omega=1e135 and crashed the solver")
        assert "upwind" in scheme, f"{field} should be upwind, got {scheme.strip()!r}"
    # momentum legitimately keeps linearUpwind -- U was never the problem
    assert "linearUpwind" in re.search(r"div\(phi,U\)\s+([^;]+);", s).group(1)

def test_adjoint_fields_cover_all_seven_patches():
    with tempfile.TemporaryDirectory() as d:
        zero_dir = Path(d)
        oa._write_adjoint_fields(zero_dir)
        for name in ("Ua", "pa", "ka", "wa"):
            text = (zero_dir / name).read_text()
            for patch in ("inlet", "outlet", "symmetry", "outer", "upperWall", "lowerWall", "car"):
                assert patch in text, f"{name} missing patch {patch}"



def test_adjoint_momentum_is_relaxed_harder_than_the_primal():
    """Ua relaxation must stay below the primal's U.

    The SIGFPE landed in divDevReff -> fvmLaplacian(nuEff, Ua) with BOTH
    adjointkOmegaSST and adjointLaminar, at adjoint iteration ~277/~282 --
    model-independent, reproducible, on a clean mesh. Uaz plateaued near 10%
    residual and oscillated until overflow while Uax/Uay sat at 0.002. That is
    the adjoint momentum equation being stepped too hard, not a turbulence
    problem.
    """
    import re
    s = oa._FV_SOLUTION_ADJOINT
    ua = float(re.search(r"^\s*Ua\s+([0-9.]+);", s, re.M).group(1))
    u = float(re.search(r"^\s*U\s+([0-9.]+);", s, re.M).group(1))
    assert ua < u, f"Ua relaxation {ua} should be below the primal U {u}"
    assert ua <= 0.5, f"Ua relaxation {ua} is too aggressive for a bluff body"


def test_invoke_adjoint_actually_passes_max_unmapped_fraction():
    """The config value must reach map_sensitivity_to_stl_vertices.

    It did not: the dataclass field and the function signature both existed and
    both looked right, but invoke_adjoint's CALL omitted the argument, so it
    silently used the 5% default while the config said 25%. Verifying that a
    field exists proves nothing about whether it is wired. Assert on the call.
    """
    import inspect, re
    src = inspect.getsource(oa.invoke_adjoint)
    call = re.search(r"map_sensitivity_to_stl_vertices\((.*?)\)", src, re.S)
    assert call, "call site not found"
    args = call.group(1)
    assert "max_unmapped_fraction" in args, (
        "invoke_adjoint does not pass cfg.max_unmapped_fraction; the config "
        "value would be silently ignored")
    assert "max_point_match_distance_m" in args

def test_atc_smoothing_is_measured_not_copied_from_the_tutorial():
    """nSmooth 0 (the sensitivityMaps/motorBike value) DIVERGES on this case.

    Measured over 150 adjoint iterations: nSmooth 0 -> |Ua| max 4.33e+13 with
    99.97% of the sensitivity's sum of squares in ten points; nSmooth 10 ->
    |Ua| max 28.1 and 20.30%. Restoring the tutorial value silently returns the
    optimiser to running on the mass gradient alone, because the diverged
    sensitivity is normalised into silence rather than raising.
    """
    import re
    d = oa.build_optimisation_dict(AdjointRunConfig(), ref_area_half=0.00375)
    m = re.search(r"nSmooth\s+(\d+)\s*;", d)
    assert m, "nSmooth entry missing from optimisationDict"
    assert int(m.group(1)) >= 3, (
        f"nSmooth is {m.group(1)}; anything below 3 diverged on the measured "
        f"case. See the ladder in build_optimisation_dict's ATCModel comment.")
    assert "includeMeshMovement false" in d, (
        "includeMeshMovement defaults to TRUE and pulls the diverging adjoint "
        "mesh-movement field into the sensitivity chain")


def _write_ua(dirpath, mags):
    """Minimal volVectorField Ua with the given x-magnitudes."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"({m} 0 0)" for m in mags)
    (d / "Ua").write_text(
        "FoamFile\n{\n version 2.0;\n format ascii;\n class volVectorField;\n"
        " object Ua;\n}\n\ndimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   nonuniform List<vector>\n{len(mags)}\n(\n{body}\n)\n;\n\n"
        "boundaryField\n{\n}\n", encoding="utf-8")


def test_check_adjoint_magnitude_accepts_a_sane_solve():
    # Measured with ATCModel=cancel: |Ua| max 6.96 m/s, p50 0.28, against a
    # 20 m/s freestream. That must pass.
    with tempfile.TemporaryDirectory() as td:
        _write_ua(os.path.join(td, "300"), [0.28, 1.65, 6.96])
        oa.check_adjoint_magnitude(td, AdjointRunConfig())


def test_check_adjoint_magnitude_catches_the_diverged_solve():
    """The 2026-07-27 failure: |Ua| ~ 1e38 reported as converged.

    Residuals said converged for hundreds of iterations (a constant relative
    residual is exactly what steady exponential growth produces), and the
    resulting 1e50 sensitivity was normalised into silence rather than raising.
    Nothing in the pipeline noticed. This is the trap.
    """
    with tempfile.TemporaryDirectory() as td:
        _write_ua(os.path.join(td, "300"), [1.0, 6.19e38, 7.2e45])
        try:
            oa.check_adjoint_magnitude(td, AdjointRunConfig())
        except RuntimeError as exc:
            assert "DIVERGED" in str(exc)
            assert "ATCModel" in str(exc), "error should name the usual cause"
            return
        raise AssertionError("diverged adjoint was not detected")


def test_check_adjoint_magnitude_ignores_uniform_unsolved_fields():
    # A time dir holding only the uniform initial Ua must not be mistaken for a
    # solution -- the real run writes 0/, 150/ and 300/ and only one is solved.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "0"
        d.mkdir(parents=True)
        (d / "Ua").write_text(
            "dimensions [0 1 -1 0 0 0 0];\ninternalField   uniform (0 0 0);\n",
            encoding="utf-8")
        oa.check_adjoint_magnitude(td, AdjointRunConfig())


def test_invoke_adjoint_actually_calls_the_divergence_check():
    # Same lesson as max_unmapped_fraction: assert on the CALL, not that the
    # function exists.
    import inspect
    src = inspect.getsource(oa.invoke_adjoint)
    assert "check_adjoint_magnitude(" in src, (
        "invoke_adjoint does not call check_adjoint_magnitude; a diverged "
        "adjoint would be read as a gradient")


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
