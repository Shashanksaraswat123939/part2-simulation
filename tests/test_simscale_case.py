"""simscale_case without the network: the domain, its labels, the STL, the
face mapping, the models the SDK would send, and the force statistics."""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simscale_case as sc  # noqa: E402

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("simscale_sdk")


def _car(td):
    body = trimesh.creation.box(bounds=[[0.03, 0.0, 0.004], [0.22, 0.03, 0.04]])
    body.export(f"{td}/car.stl", file_type="stl_ascii")
    wheel = trimesh.creation.cylinder(radius=0.014, height=0.013, sections=48)
    wheel.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    wheel.apply_translation([0.046, 0.035, 0.014 - 0.0003])
    wheel.export(f"{td}/wheelF.stl", file_type="stl_ascii")
    rot = {"origin": (0.046, 0.0415, 0.0137), "axis": (0, 1, 0), "omega": -20.0 / 0.014}
    return f"{td}/car.stl", ({"name": "wheelF", "stl": f"{td}/wheelF.stl", "rotating": rot},)


def test_domain_is_one_closed_fluid_with_every_boundary_labelled():
    with tempfile.TemporaryDirectory() as td:
        car, extra = _car(td)
        cfg = sc.SimScaleConfig(extra_surfaces=extra)
        fluid, labels, parts = sc.build_domain(car, cfg)
        assert fluid.is_watertight and parts == ["car", "wheelF"]
        assert set(labels) == set(sc.BOX_FACES) | {"car", "wheelF"}
        # the fluid volume is the box minus the parts
        lo, hi = sc.domain_box(sc.case_bounds(car, cfg))
        box_v = np.prod(np.array(hi) - np.array(lo))
        assert fluid.volume < box_v and fluid.volume > 0.99 * box_v
        # wheel faces sit on the wheel: within its radius of the axle
        c = fluid.triangles_center[labels == "wheelF"]
        r = np.hypot(c[:, 0] - 0.046, c[:, 2] - 0.0137)
        assert r.max() < 0.0141
        names = sc.write_multisolid_stl(fluid, labels, f"{td}/domain.stl")
        txt = Path(f"{td}/domain.stl").read_text()
        assert names[:6] == list(sc.BOX_FACES) and txt.count("solid ") == 2 * len(names)
        back = trimesh.load(f"{td}/domain.stl", force="mesh")   # one mesh per solid, concatenated
        back.merge_vertices()                                     # weld them, as SimScale does
        assert len(back.faces) == len(fluid.faces)
        assert back.is_watertight and abs(back.volume - fluid.volume) < 1e-9


def test_face_mapping_by_name_then_by_order():
    names = ["inlet", "outlet", "car"]
    by_name = [{"name": f"F{i}", "originate_from": [{"body": "B1", "entity": n}]}
               for i, n in enumerate(names)]
    assert sc.faces_by_label(by_name, names) == {"inlet": ["F0"], "outlet": ["F1"], "car": ["F2"]}
    anon = [{"name": f"F{i}", "originate_from": []} for i in range(3)]
    assert sc.faces_by_label(anon, names)["car"] == ["F2"]
    with pytest.raises(sc.SimScaleError):
        sc.faces_by_label(anon[:2], names)


def test_models_serialise_with_rotating_wheels_layers_and_wake_boxes():
    from simscale_sdk import ApiClient, SimulationSpec
    faces = {n: [f"F_{n}"] for n in sc.BOX_FACES + ("car", "wheelF")}
    wheels = {"wheelF": ((0.046, 0.0415, 0.0137), -1428.6)}
    cfg = sc.SimScaleConfig(resolution="resolved")
    spec = SimulationSpec(name="t", geometry_id="g", model=sc.build_model(cfg, faces, ["R1"], wheels))
    d = ApiClient().sanitize_for_serialization(spec)
    bcs = {b["name"]: b for b in d["model"]["boundaryConditions"]}
    assert bcs["wheelF"]["velocity"]["type"] == "ROTATING_WALL_VELOCITY"
    assert bcs["car"]["velocity"]["type"] == "NO_SLIP"
    assert bcs["ground"]["velocity"]["type"] == "MOVING_WALL_VELOCITY"
    assert {f["name"] for f in d["model"]["resultControl"]["forcesMoments"]} == {"F_car", "F_wheelF"}
    mesh = ApiClient().sanitize_for_serialization(
        sc.build_mesh_model(cfg, faces, wheels, {"wakeNear": "p1", "wakeFar": "p2"}))
    kinds = [r["type"] for r in mesh["refinements"]]
    assert "SIMMETRIX_BOUNDARY_LAYER_V13" in kinds and kinds.count("REGION_LENGTH") == 2
    layer = next(r for r in mesh["refinements"] if r["type"] == "SIMMETRIX_BOUNDARY_LAYER_V13")
    assert layer["layerType"]["firstLayerSize"]["value"] == cfg.first_layer_m
    assert set(layer["topologicalReference"]["entities"]) == {"F_car", "F_wheelF"}


def test_force_series_uses_the_total_force_over_the_last_fraction():
    rows = "Iteration,PRESSURE_FORCE_X,TOTAL_FORCE_X,TOTAL_FORCE_Y,TOTAL_FORCE_Z\n" + "".join(
        f"{i},{0.5},{1.0 + (0.1 if i < 80 else 0.0)},0,{0.2}\n" for i in range(100))
    s = sc.force_series(rows, 0.2)
    assert len(s["fx"]) == 20 and s["fx"] == [1.0] * 20 and s["fz"] == [0.2] * 20
    se, drift = sc.force_mean_convergence(s["fx"])
    assert se == 0.0 and drift == 0.0
