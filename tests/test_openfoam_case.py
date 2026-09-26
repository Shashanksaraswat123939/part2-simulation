"""Unit tests for openfoam_case.py.

These cover everything that does NOT require a live OpenFOAM solve: the
geometry math, dictionary generation, turbulence inlet values, and the log/
force/moment/y+ parsers (against crafted ESI-format fixtures). The actual
subprocess solve is exercised on a machine with ESI OpenFOAM installed.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

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
    # k = 1.5 (I U)^2 = 1.5 (0.005*20)^2 = 0.015
    assert abs(k - 0.015) < 1e-12, k


def test_freestream_eddy_viscosity_is_near_laminar():
    """A car in still air sees an almost laminar freestream. The legacy inlet
    (I=5 %, omega from the car length) gave nut/nu ~ 10,500 -- a freestream
    four orders of magnitude too viscous. Guard the default against that."""
    cfg = OpenFOAMRunConfig(turbulence_model="kOmegaSST", reference_speed_mps=20.0)
    _k, _omega, nut = oc.turbulence_inlet_values(cfg, ref_length_m=0.233)
    ratio = nut / cfg.kinematic_viscosity_m2s
    assert ratio < 20.0, ratio
    legacy = OpenFOAMRunConfig(turbulence_model="kOmegaSST", reference_speed_mps=20.0,
                               turbulence_intensity=0.05, turbulent_viscosity_ratio=None)
    _k, _omega, nut_legacy = oc.turbulence_inlet_values(legacy, ref_length_m=0.233)
    assert nut_legacy / legacy.kinematic_viscosity_m2s > 5000.0


def test_fixed_meshing_frame_ignores_stl_bounds():
    """With domain_reference_bounds set, two STLs of different size get the
    same meshing bounds, so the background mesh does not move between them."""
    ref = ((0.0, 0.0, 0.0), (0.26, 0.0425, 0.065))
    cfg = OpenFOAMRunConfig(domain_reference_bounds=ref)
    a = oc.meshing_bounds(((0.01, 0.0, 0.0015), (0.20, 0.03, 0.05)), cfg)
    b = oc.meshing_bounds(((0.0100001, 0.0, 0.0015), (0.2000003, 0.03, 0.05)), cfg)
    assert a == b == ref
    legacy = OpenFOAMRunConfig()
    assert oc.meshing_bounds(((0.01, 0.0, 0.0015), (0.2, 0.03, 0.05)), legacy)[0][0] == 0.01
    try:
        oc.meshing_bounds(((0.0, 0.0, 0.0), (0.30, 0.03, 0.05)), cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("STL outside the fixed frame must be rejected")


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


def test_location_in_mesh_never_lands_on_a_cell_face():
    """snappyHexMesh findCell() fails intermittently for points on cell faces.

    This background mesh is perfectly self-similar -- domain width 12*lx, cell
    lx/12, so nx == 144 for ANY car length. The old x0 - 1.0*lx put the point at
    exactly 1/6 of the domain = cell index 24.0, dead on a face, and whether
    snappy found it came down to floating-point rounding. It meshed iteration 1
    and refused iteration 2 after a 0.3% volume change.
    """
    import re
    for lx_mm in (120.0, 150.0, 232.3, 232.26, 300.0, 199.999, 401.7):
        lx = lx_mm / 1000.0
        bounds = ((0.0002, 0.0, 0.0017), (0.0002 + lx, 0.0353, 0.0643))
        loc = oc.location_in_mesh(bounds)
        bmin, bmax = oc.domain_box(bounds)
        d = oc.build_blockmesh_dict(bmin, bmax, max(lx / 12.0, 1e-4))
        nx, ny, nz = (int(v) for v in
                      re.search(r"hex \(0 1 2 3 4 5 6 7\) \((\d+) (\d+) (\d+)\)",
                                d).groups())
        for ax, n in zip(range(3), (nx, ny, nz)):
            width = bmax[ax] - bmin[ax]
            idx = (loc[ax] - bmin[ax]) / width * n
            # snapped to a cell CENTRE -> index fraction must be ~0.5
            assert 0.4 < idx % 1.0 < 0.6, (
                f"lx={lx_mm}mm axis {'xyz'[ax]}: locationInMesh at cell index "
                f"{idx:.4f} -- too close to a cell face; snappy findCell() is "
                "a coin flip there")
            assert 0 < idx < n, f"locationInMesh outside the mesh on axis {ax}"


def test_write_control_is_a_valid_time_enum():
    """controlDict's writeControl must be a Time::writeControls value.

    Regression guard: `writeControl onEnd;` shipped in BOTH controlDicts.
    `onEnd` is valid for a FUNCTION OBJECT's writeControl but not the global
    one, and v2412 aborts with a Foam::Enum readEntry error before meshing
    starts. Found on the first real solve, 2026-07-26.
    """
    import re
    import openfoam_adjoint as oa
    VALID = {"timeStep", "runTime", "adjustableRunTime", "cpuTime", "clockTime"}

    fwd = oc.build_control_dict(OpenFOAMRunConfig(max_iterations=250), 0.004)
    adj = oa.build_adjoint_control_dict(
        oa.AdjointRunConfig(primal_iters=40, adjoint_iters=60))

    for label, text in (("forward", fwd), ("adjoint", adj)):
        # The FIRST writeControl is the global (Time) one; later ones belong to
        # function objects, which legitimately accept writeTime.
        first = re.search(r"^writeControl\s+(\w+);", text, re.M)
        assert first, f"{label}: no global writeControl"
        assert first.group(1) in VALID, (
            f"{label} writeControl={first.group(1)!r} not in {sorted(VALID)}")
        interval = re.search(r"^writeInterval\s+(\d+);", text, re.M)
        end = re.search(r"^endTime\s+(\d+);", text, re.M)
        assert interval and end, f"{label}: missing writeInterval/endTime"
        assert int(interval.group(1)) == int(end.group(1)), (
            f"{label}: writeInterval {interval.group(1)} != endTime "
            f"{end.group(1)}; fields would be written more than once")


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


def _oscillating_force_dat(n=1000, mean=0.35, amp=0.04):
    """A force history like the real ones: settled residuals, swinging force."""
    import math
    rows = ["# Forces", "# Time forces"]
    for i in range(1, n + 1):
        fx = mean + amp * math.sin(i * 0.37)
        rows.append(f"{i} ({fx} 0.0 -0.05) ({fx*0.8} 0 -0.04) ({fx*0.2} 0 -0.01)")
    return "\n".join(rows) + "\n"


def test_parse_total_vector_dat_averages_the_tail_not_the_last_sample():
    """Reading the final row samples an oscillation at an arbitrary phase.

    Measured 2026-07-27: real solves swing 18-27% peak-to-peak over the last
    20% of iterations, so two geometries 65 NANOMETRES apart reported drag 7.7%
    apart. Averaging makes the reported number reproducible.
    """
    dat = _oscillating_force_dat()
    fx, _fy, _fz = oc.parse_total_vector_dat(dat)
    # The mean of the window must be far closer to the true mean than the last
    # sample is, which is the whole point.
    last = float(dat.strip().splitlines()[-1].replace("(", " ").replace(")", " ").split()[1])
    assert abs(fx - 0.35) < abs(last - 0.35), (
        f"averaged {fx} is no closer to the true mean than the last sample {last}")
    assert abs(fx - 0.35) < 0.01, f"averaged force {fx} should sit near the mean 0.35"


def test_force_oscillation_fraction_reports_what_residuals_cannot():
    dat = _oscillating_force_dat(mean=0.35, amp=0.04)
    osc = oc.force_oscillation_fraction(dat)
    # amplitude 0.04 about a mean of 0.35 -> peak-to-peak ~0.08 -> ~23%
    assert 0.15 < osc < 0.30, f"expected ~23% peak-to-peak, got {osc}"

    steady = "# Time forces\n" + "\n".join(
        f"{i} (0.35 0.0 -0.05) (0.28 0 -0.04) (0.07 0 -0.01)" for i in range(1, 501))
    assert oc.force_oscillation_fraction(steady) < 1e-9


def test_convergence_threshold_is_reachable_at_production_resolution():
    """The gate must be passable by the resolution production actually uses.

    Measured p-residual plateaus on one fixed STL: coarse 4.4e-4, medium
    (production) 2.1e-3, medium+underbody 3.2e-3. The old hardcoded 1e-3 was
    unpassable at medium, so --smoke (require_cfd_convergence=False) completed
    while a real run marked every candidate CFD_failed before the adjoint ran.
    """
    import cfd_wrapper as cw
    assert cw.CONVERGENCE_RESIDUAL > 3.2e-3, (
        f"threshold {cw.CONVERGENCE_RESIDUAL} sits below the measured "
        f"production plateau of 3.2e-3; no candidate could ever be scored")
    assert cw.CONVERGENCE_RESIDUAL < 5e-2, (
        f"threshold {cw.CONVERGENCE_RESIDUAL} is loose enough to accept a "
        f"solve that never converged at all")


def test_force_oscillation_warns_but_does_not_gate_convergence():
    """The oscillation check must NOT be folded into `converged`.

    Part 3's require_cfd_convergence defaults to True and routes a
    non-converged solve to CFD_failed. Every real solve currently oscillates
    10-27%, so ANDing that into `converged` killed every candidate in a
    production sweep. It is a diagnosis, not a gate — reported and warned
    about, with the policy left to the caller.
    """
    import inspect
    import cfd_wrapper as cw
    assert cw.MAX_FORCE_OSCILLATION < 0.18, (
        "threshold must flag the 18-27% oscillation measured on the brick")
    src = inspect.getsource(cw.run_half_car_cfd)
    conv = [ln for ln in src.splitlines() if "converged=" in ln]
    assert conv, "no converged= assignment found"
    assert all("force_oscillation" not in ln and "force_steady" not in ln
               for ln in conv), (
        "converged must stay residual-only; folding the force-oscillation "
        "check into it fails every candidate in a production sweep")
    assert any("residual_final" in ln for ln in conv), (
        "converged must still be based on the residual")
    assert "force_oscillation=force_oscillation" in src, (
        "the oscillation must still be reported on the health report")


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
    # search_roots=[] disables the common-install-root fallback scan --
    # required for this to be deterministic (2026-07-16 fix): an invalid
    # explicit path alone does NOT guarantee "not found", since discovery
    # falls through to $WM_PROJECT_DIR / $FOAM_BASHRC / common roots. On any
    # machine that actually has ESI OpenFOAM installed (e.g. this one), the
    # old version of this test silently found the real environment instead
    # of testing the "absent" case at all.
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FOAM_BASHRC", None)
        os.environ.pop("WM_PROJECT_DIR", None)
        assert oc.find_openfoam_bashrc(
            "/definitely/not/a/real/bashrc", search_roots=[]
        ) is None


def test_invoke_raises_when_openfoam_absent():
    # search_roots=[] + cleared env vars: see test_find_bashrc_missing_returns_none.
    # Without this, on a machine with real ESI OpenFOAM installed, invoke()
    # silently finds it and launches a genuine multi-stage solve instead of
    # raising -- verified live, 2026-07-16 (a ~15-minute simpleFoam run got
    # triggered by this exact test before the fix, on a machine where
    # OpenFOAM had just been set up).
    stl = _write_stl(_PANELS)
    try:
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FOAM_BASHRC", None)
                os.environ.pop("WM_PROJECT_DIR", None)
                try:
                    oc.invoke(
                        stl, d, bashrc="/definitely/not/a/real/bashrc",
                        search_roots=[],
                    )
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


def test_peak_to_peak_ranks_solve_quality_backwards_and_stderr_does_not():
    """The error on D20 is the error on a MEAN, not the swing of the signal.

    force_oscillation_fraction is max-minus-min: one outlier sets it, and it
    does not shrink as you average over more samples. But D20 is a mean, whose
    uncertainty does shrink. So the two statistics disagree about which solves
    are trustworthy -- and on these three signals peak-to-peak gets the order
    exactly backwards, calling the drifting solve the best of the three.

    That matters because MAX_FORCE_OSCILLATION gates on it: a solve whose mean
    is pinned to 0.35% was being reported as unusable at 26% peak-to-peak.
    """
    import math
    import random

    from openfoam_case import force_mean_convergence, force_oscillation_fraction

    def _dat(vals):
        head = "# Time (total_x total_y total_z) (p_x p_y p_z) (v_x v_y v_z)\n"
        return head + "".join(
            f"{i} ({v} 0 0) (0 0 0) (0 0 0)\n" for i, v in enumerate(vals))

    random.seed(0)
    n = 1000
    settled = [10.0 + random.gauss(0, 0.5) for _ in range(n)]
    oscillating = [10.0 + 2.0 * math.sin(i / 5.0) for i in range(n)]
    drifting = [10.0 + 0.004 * i + random.gauss(0, 0.2) for i in range(n)]

    pp = {k: force_oscillation_fraction(_dat(v)) for k, v in
          (("settled", settled), ("osc", oscillating), ("drift", drifting))}
    se = {}
    dr = {}
    for k, v in (("settled", settled), ("osc", oscillating), ("drift", drifting)):
        se[k], dr[k] = force_mean_convergence(_dat(v))

    # A settled solve has a big peak-to-peak (it is noisy) but a mean that is
    # nailed down. Peak-to-peak alone would reject it.
    assert pp["settled"] > 0.20, pp["settled"]
    assert se["settled"] < 0.01, f"mean of a settled solve should be tight, got {se['settled']}"

    # The drifting solve has the SMALLEST peak-to-peak of the three and is the
    # least trustworthy: its mean is still moving.
    assert pp["drift"] < pp["settled"], "the premise of this test"
    assert dr["drift"] > 4 * dr["settled"], (
        f"drift should dominate for a drifting signal: {dr['drift']} vs {dr['settled']}")
    assert dr["drift"] > se["drift"], (
        "when drift exceeds the standard error the mean has not settled, and "
        "that is the case a longer averaging window cannot fix")

    # A stationary oscillation is honestly reported as noisier in the mean than
    # the settled case, without being confused for drift.
    assert se["osc"] > se["settled"]

    # And a perfectly flat signal is zero on both.
    flat_se, flat_dr = force_mean_convergence(_dat([7.0] * 100))
    assert flat_se == 0.0 and flat_dr == 0.0, (flat_se, flat_dr)

    # Too few samples must read as "unknown", never as "converged".
    short_se, short_dr = force_mean_convergence(_dat([7.0, 7.1]))
    assert short_se == float("inf") and short_dr == float("inf")
