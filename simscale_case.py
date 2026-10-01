"""
simscale_case.py -- the half-car CFD on SimScale, through the SimScale Python SDK.

The same physics as the OpenFOAM case on main (inlet at 20 m/s, moving ground,
slip top and side, symmetry plane, rotating wheels, every part its own force
group, k-omega SST), run by SimScale:

  1. DOMAIN   built here: the wind-tunnel box minus the half car (every Part 4
              part included), ONE closed fluid solid, written as an ASCII STL
              with one named `solid` per boundary. SimScale imports each named
              solid as its own sheet (in "#1", "#2" pieces if disconnected);
              sewing joins them into one fluid region whose faces keep those
              names, so every boundary condition and force group can address
              them.
  2. IMPORT   upload + geometry import (STL, metres), then the face mapping:
              SimScale face name -> our boundary name.
  3. MESH     Simmetrix: manual sizing, finer surface sizing on the car and the
              wheels, wake boxes (geometry primitives), and a boundary-layer
              stack with an explicit first-layer height for y+ <= 2.
  4. RUN      incompressible, k-omega SST, forces per part averaged over the
              last `fraction_from_end` of the run, the y+ field.
  5. RESULTS  per-part forces (half car), the mean over the averaging window,
              its standard error and drift, the final pressure residual.

Credentials come from the environment and never from this file:
    SIMSCALE_API_KEY      the account's API key
    SIMSCALE_PROJECT_ID   the project every geometry, mesh and run goes into; if
                          the key's account may not write to it, the account's own
                          project OWN_PROJECT_NAME is found or created and used
    SIMSCALE_API_BASE     optional, default https://api.simscale.com/v0
"""
from __future__ import annotations

import csv
import io
import json
import re
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Boundary names in the domain STL. The car parts follow Part 4's patch names.
BOX_FACES = ("inlet", "outlet", "symmetry", "side", "ground", "top")

# Mesh presets: (car surface size, wheel surface size, far-field size, wake
# near size, wake far size), metres. "resolved" is the ~5 M cell target.
MESH_PRESETS = {
    "coarse": (2.0e-3, 1.0e-3, 40e-3, 4.0e-3, 8.0e-3),
    "medium": (1.2e-3, 0.6e-3, 30e-3, 2.4e-3, 5.0e-3),
    "fine": (0.8e-3, 0.4e-3, 25e-3, 1.6e-3, 3.2e-3),
    "resolved": (0.6e-3, 0.3e-3, 20e-3, 1.2e-3, 2.4e-3),
}


@dataclass(frozen=True)
class SimScaleConfig:
    speed_mps: float = 20.0
    density_kgm3: float = 1.225
    kinematic_viscosity_m2s: float = 1.813e-5 / 1.225
    turbulence_intensity: float = 0.005
    iterations: int = 2000
    resolution: str = "medium"
    # Boundary layers (team requirement: y+ <= 2): first layer, growth, count.
    first_layer_m: float = 1.5e-5
    layer_growth: float = 1.2
    n_layers: int = 12
    fraction_from_end: float = 0.2     # force averaging window
    # Geometry clean-up before the domain boolean (SimScale rejects "face
    # slits": air layers of almost zero thickness). Features under
    # `simplify_m` are removed from every part; stationary parts grow by
    # `close_gap_m` so gaps under twice that close. Both are far below the
    # 0.3-2 mm surface cells. Wheels are left exact (rotating walls).
    simplify_m: float = 2e-5
    close_gap_m: float = 5e-5
    poll_s: float = 15.0
    max_run_time_s: float = 36000.0
    extra_surfaces: tuple = ()
    domain_reference_bounds: Optional[tuple] = None
    run_name: str = "car"
    keep: bool = True                  # keep the SimScale project entities after the run

    def __post_init__(self):
        if self.resolution not in MESH_PRESETS:
            raise ValueError(f"resolution must be one of {sorted(MESH_PRESETS)}")


class SimScaleError(RuntimeError):
    pass


OWN_PROJECT_NAME = "STEM Racing optimiser"
# Parts never grown along their normals: the halo CAD is too detailed (growing
# it 0.2 mm folded it into itself: SimScale "face self_int" faults, probe 3,
# 2026-10-01). As drawn it imports cleanly (probe 2).
EXACT_PARTS = {"halo"}
# The bodywork behind the halo leans in until it touches the halo's back face,
# leaving a wedge of air 0-1.1 mm wide between two nearly parallel faces
# (probe 4: 'fault-face-slit' at the halo's back, 2026-10-01). The halo's
# last 3 mm are stretched this far back, into the bodywork. The stretch is
# monotone in x, so no face turns over. ponytail: fixed length; if a car's
# gap exceeds it, the probe says so.
HALO_BACK_STRETCH_M = 1.2e-3
_PROJECT: Optional[str] = None          # the project in use (see _project)


# --------------------------------------------------------------------------- geometry
def stl_bounds(path: str) -> tuple:
    import trimesh
    b = trimesh.load(str(path), force="mesh").bounds
    return tuple(map(tuple, b))


def case_bounds(car_stl: str, cfg: SimScaleConfig) -> tuple:
    if cfg.domain_reference_bounds is not None:
        return tuple(map(tuple, cfg.domain_reference_bounds))
    lo, hi = np.array(stl_bounds(car_stl))
    for s in cfg.extra_surfaces:
        b = np.array(stl_bounds(s["stl"]))
        lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
    return (tuple(lo), tuple(hi))


def domain_box(bounds) -> tuple:
    """3 car lengths ahead, 8 behind, 5x the half width and height around,
    track at z = 0, symmetry plane at y = 0 (as the OpenFOAM case on main)."""
    (x0, _y0, _z0), (x1, y1, z1) = bounds
    lx, ly, lz = x1 - x0, max(y1, 1e-6), max(z1, 1e-6)
    return (x0 - 3 * lx, 0.0, 0.0), (x1 + 8 * lx, y1 + 5 * ly, z1 + 5 * lz)


def wake_boxes(bounds) -> tuple:
    """(name, min, max): the car plus one length behind it; four more lengths."""
    (x0, _y0, _z0), (x1, y1, z1) = bounds
    lx = x1 - x0
    return (("wakeNear", (x0 - 0.05 * lx, 0.0, 0.0), (x1 + lx, 1.25 * y1, 1.25 * z1)),
            ("wakeFar", (x1, 0.0, 0.0), (x1 + 4 * lx, 1.5 * y1, 2.0 * z1)))


def frontal_area_half(car_stl: str) -> float:
    """Projected area on the y-z plane of the front-facing triangles, m^2."""
    import trimesh
    m = trimesh.load(str(car_stl), force="mesh")
    nx = m.face_normals[:, 0] * m.area_faces
    return float(-nx[nx < 0].sum())


def build_domain(car_stl: str, cfg: SimScaleConfig):
    """(fluid mesh, face labels, part names). The fluid is the box minus the
    union of every car part. The boolean library records which input every
    output triangle came from, so each face is labelled exactly: box faces by
    the side they lie on, the rest by their part."""
    import manifold3d as m3
    import trimesh
    parts = {"car": trimesh.load(str(car_stl), force="mesh")}
    for s_ in cfg.extra_surfaces:
        parts[s_["name"]] = trimesh.load(str(s_["stl"]), force="mesh")
    for n, m in parts.items():
        if not m.is_watertight:
            raise SimScaleError(f"part {n} is not a closed surface")

    def man(t):
        return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                                   tri_verts=np.asarray(t.faces, np.uint32))).as_original()

    rotating = {s_["name"] for s_ in cfg.extra_surfaces if s_.get("rotating")}
    for n in list(parts):
        t = parts[n]
        if cfg.simplify_m > 0:
            o = man(t).simplify(cfg.simplify_m).to_mesh()
            t = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts),
                                process=False)
        if n == "halo":
            v = np.array(t.vertices)
            x0 = v[:, 0].max() - 3e-3
            v[:, 0] += HALO_BACK_STRETCH_M * np.clip((v[:, 0] - x0) / 3e-3, 0, 1)
            t = trimesh.Trimesh(v, t.faces, process=False)
        if cfg.close_gap_m > 0 and n not in rotating | EXACT_PARTS:
            t = trimesh.Trimesh(t.vertices + t.vertex_normals * cfg.close_gap_m, t.faces,
                                process=False)
        # Each part is a right half closed on y = 0, exactly where the box's
        # symmetry face is: coincident faces leave slivers. Parts grown above
        # (and the half body, whose surface runs into that face in a strip of
        # nearly flat triangles up to ~0.14 mm off it) have every vertex within
        # 0.2 mm of the plane moved through it. An exact part, whose closing
        # face is exactly flat, is moved through the plane whole instead:
        # moving only its curved surface turns faces over (9 on the halo), and
        # joining it to its mirror image left the closing face inside it
        # (probe 6).
        if n in EXACT_PARTS:
            t = trimesh.Trimesh(t.vertices - [0.0, max(cfg.close_gap_m, 5e-5), 0.0], t.faces,
                                process=False)
        else:
            v = np.array(t.vertices)
            v[v[:, 1] < 2e-4, 1] = -max(cfg.close_gap_m, 5e-5)
            t = trimesh.Trimesh(v, t.faces, process=False)
        parts[n] = t
    lo, hi = domain_box(case_bounds(car_stl, cfg))
    box = man(trimesh.creation.box(bounds=[lo, hi]))
    owner = {box.original_id(): None}
    solids = []
    for n, t in parts.items():
        mm = man(t)
        owner[mm.original_id()] = n
        solids.append(mm)
    fluid_m = box - m3.Manifold.batch_boolean(solids, m3.OpType.Add)
    out = fluid_m.to_mesh()
    fluid = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts),
                            process=False)
    if not fluid.is_watertight:
        raise SimScaleError("the fluid domain is not a closed solid")
    labels = np.full(len(fluid.faces), "", dtype=object)
    starts = np.asarray(out.run_index) // 3
    for k, oid in enumerate(np.asarray(out.run_original_id)):
        labels[starts[k]:starts[k + 1]] = owner.get(int(oid)) or "__box__"
    on_box = np.nonzero(labels == "__box__")[0]
    c = fluid.triangles_center[on_box]
    side = np.stack([np.abs(c[:, 0] - lo[0]), np.abs(c[:, 0] - hi[0]), np.abs(c[:, 1] - lo[1]),
                     np.abs(c[:, 1] - hi[1]), np.abs(c[:, 2] - lo[2]), np.abs(c[:, 2] - hi[2])])
    labels[on_box] = np.array(BOX_FACES, dtype=object)[side.argmin(axis=0)]
    return fluid, labels, list(parts)


def thin_air(fluid, labels, below_m: float = 1.5e-4) -> list:
    """Places where the fluid is a thin sheet between two nearly parallel
    walls (SimScale's "face slit"): ((x, y, z) mm, gap mm, label, label).
    From each wall face a segment `below_m` long goes into the fluid; a face
    it crosses that points back at it is the other side of a slit. Wedges
    (wheel contact lines) meet at an angle and are not reported."""
    walls = np.nonzero(~np.isin(labels, BOX_FACES))[0]
    tri = fluid.triangles
    nrm = fluid.face_normals                       # outward of the fluid: into the part
    o = fluid.triangles_center[walls] - nrm[walls] * 1e-9
    d = -nrm[walls]
    e = o + d * below_m
    tree = fluid.triangles_tree
    pairs = []
    for k in range(len(walls)):
        lo, hi = np.minimum(o[k], e[k]), np.maximum(o[k], e[k])
        for t in tree.intersection(np.r_[lo, hi]):
            if t != walls[k] and nrm[t] @ nrm[walls[k]] < -0.9:
                pairs.append((k, t))
    out = []
    if not pairs:
        return out
    P = np.array(pairs)
    a, b, c = tri[P[:, 1], 0], tri[P[:, 1], 1], tri[P[:, 1], 2]
    oo, dd = o[P[:, 0]], d[P[:, 0]]
    e1, e2 = b - a, c - a                          # Moller-Trumbore, vectorised
    h = np.cross(dd, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-30
    f = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    sv = oo - a
    u = f * np.einsum("ij,ij->i", sv, h)
    q = np.cross(sv, e1)
    v = f * np.einsum("ij,ij->i", dd, q)
    tt = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (tt > 0) & (tt < below_m)
    best = {}
    for (k, t), g, h_ in zip(pairs, tt, hit):
        if h_ and (k not in best or g < best[k][0]):
            best[k] = (g, t)
    for k, (g, t) in best.items():
        out.append((tuple(np.round(fluid.triangles_center[walls[k]] * 1e3, 3)), round(float(g) * 1e3, 4),
                    labels[walls[k]], labels[t]))
    return out


def write_multisolid_stl(mesh, labels, path: str) -> list:
    """ASCII STL, metres, one `solid <name>` per label. Returns the names in
    file order."""
    names = [n for n in BOX_FACES if (labels == n).any()]
    names += sorted(set(labels) - set(BOX_FACES))
    with open(path, "w", encoding="ascii") as f:
        for n in names:
            f.write(f"solid {n}\n")
            for i in np.nonzero(labels == n)[0]:
                a, b, c = mesh.vertices[mesh.faces[i]]
                nrm = mesh.face_normals[i]
                f.write(f" facet normal {nrm[0]:.7e} {nrm[1]:.7e} {nrm[2]:.7e}\n  outer loop\n")
                for v in (a, b, c):
                    f.write(f"   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {n}\n")
    return names


# --------------------------------------------------------------------------- statistics
def force_mean_convergence(fx: list) -> tuple:
    """(stderr fraction, drift fraction) of the mean of a force series, the
    same estimator as the OpenFOAM case on main (autocorrelation-corrected
    standard error; least-squares drift across the window)."""
    n = len(fx)
    if n < 3:
        return float("inf"), float("inf")
    mean = sum(fx) / n
    if mean == 0.0:
        return float("inf"), float("inf")
    dev = [v - mean for v in fx]
    var = sum(d * d for d in dev) / (n - 1)
    if var <= 0.0:
        return 0.0, 0.0
    tau = 1.0
    for lag in range(1, n // 2):
        rho = sum(dev[i] * dev[i + lag] for i in range(n - lag)) / ((n - lag) * var)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    stderr = (var ** 0.5 / max(1.0, n / tau) ** 0.5) / abs(mean)
    xm = (n - 1) / 2.0
    sxx = sum((x - xm) ** 2 for x in range(n))
    slope = sum((x - xm) * d for x, d in zip(range(n), dev)) / sxx
    return stderr, abs(slope) * (n - 1) / abs(mean)


def force_series(csv_text: str, fraction: float) -> dict:
    """{"fx": [...], "fz": [...]} over the last `fraction` of a SimScale
    FORCE_PLOT CSV (TOTAL_FORCE_X/Z: pressure + viscous)."""
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if not rows:
        raise SimScaleError("empty force CSV")

    def col(prefix):
        c = next((k for k in rows[0] if k and k.strip().upper().startswith(prefix)), None)
        if c is None:
            raise SimScaleError(f"no {prefix} column in {list(rows[0])}")
        return c
    fx, fz = col("TOTAL_FORCE_X"), col("TOTAL_FORCE_Z")
    k = max(int(len(rows) * fraction), 1)
    tail = rows[-k:]
    return {"fx": [float(r[fx]) for r in tail], "fz": [float(r[fz]) for r in tail]}


# --------------------------------------------------------------------------- SimScale
def _clients():
    from simscale_sdk import (ApiClient, Configuration, GeometriesApi, GeometryImportsApi,
                              MeshesApi, MeshOperationsApi, ProjectsApi, SimulationRunsApi,
                              SimulationsApi, StorageApi)
    key = os.environ.get("SIMSCALE_API_KEY")
    if not key:
        raise SimScaleError("SIMSCALE_API_KEY is not set")
    conf = Configuration()
    conf.host = os.environ.get("SIMSCALE_API_BASE", "https://api.simscale.com/v0")
    conf.api_key = {"X-API-KEY": key}
    client = ApiClient(configuration=conf)
    return {"client": client, "storage": StorageApi(client), "imports": GeometryImportsApi(client),
            "geometries": GeometriesApi(client), "sims": SimulationsApi(client),
            "mesh": MeshOperationsApi(client), "meshes": MeshesApi(client),
            "runs": SimulationRunsApi(client), "projects": ProjectsApi(client)}


def _project() -> str:
    if _PROJECT:
        return _PROJECT
    p = os.environ.get("SIMSCALE_PROJECT_ID")
    if not p:
        raise SimScaleError("SIMSCALE_PROJECT_ID is not set")
    return p


def own_project(api) -> str:
    """The key's own project OWN_PROJECT_NAME, created if it does not exist.
    Used when the configured project refuses writes (seen 2026-10-01: a key
    that may not write to the TurboFlow project)."""
    from simscale_sdk import Project
    page = 1
    while True:
        res = api["projects"].get_projects(limit=100, page=page)
        items = res.embedded or []
        for pr in items:
            if pr.name == OWN_PROJECT_NAME:
                return pr.project_id
        if len(items) < 100:
            break
        page += 1
    pr = api["projects"].create_project(Project(
        name=OWN_PROJECT_NAME, measurement_system="SI",
        description="Part 5 pattern search: half-car domains, meshes and runs"))
    return pr.project_id


def _use_own_project_if_refused(api, call):
    """Run call(); on a 403 from the configured project switch to the key's
    own project, say so, and run it again."""
    global _PROJECT
    from simscale_sdk.exceptions import ApiException
    try:
        return call()
    except ApiException as exc:
        if exc.status != 403 or _PROJECT:
            raise
        _PROJECT = own_project(api)
        print(f"[simscale] project {os.environ.get('SIMSCALE_PROJECT_ID')} refused writes; "
              f"using the account's own project '{OWN_PROJECT_NAME}' ({_PROJECT}). "
              f"Set SIMSCALE_PROJECT_ID to {_PROJECT} to skip this step.", flush=True)
        return call()


def _poll(get, done=("FINISHED", "SUCCESS"), failed=("FAILED", "ERROR", "CANCELED"),
          every=15.0, what="operation"):
    while True:
        obj = get()
        status = (getattr(obj, "status", "") or "").upper()
        if status in done:
            return obj
        if status in failed:
            raise SimScaleError(f"{what} {status}: {getattr(obj, 'failure_reason', '')}")
        time.sleep(every)


def import_geometry(api, stl_path: str, name: str, every: float = 15.0) -> str:
    from simscale_sdk import (GeometryImportRequest, GeometryImportRequestLocation,
                              GeometryImportRequestOptions)
    storage = api["storage"].create_storage()
    with open(stl_path, "rb") as f:
        api["client"].rest_client.PUT(url=storage.url, body=f.read(),
                                      headers={"Content-Type": "application/octet-stream"})
    imp = _use_own_project_if_refused(api, lambda: api["imports"].import_geometry(
        _project(), GeometryImportRequest(
            name=name, location=GeometryImportRequestLocation(storage_id=storage.storage_id),
            format="STL", input_unit="m",
            options=GeometryImportRequestOptions(facet_split=False, sewing=True, improve=True,
                                                 optimize_for_lbm_solver=False))))
    done = _poll(lambda: api["imports"].get_geometry_import(_project(), imp.geometry_import_id),
                 every=every, what="geometry import")
    return done.geometry_id


def face_mapping(api, geometry_id: str) -> list:
    """Every face entity as a dict (name, originate_from, ...)."""
    out, page = [], 1
    while True:
        res = api["geometries"].get_geometry_mappings(_project(), geometry_id, _class="face",
                                                      limit=100, page=page)
        batch = [e.to_dict() for e in (res.embedded or [])]
        out += batch
        if len(batch) < 100:
            return out
        page += 1


def faces_by_label(mappings: list, names: list) -> dict:
    """{our boundary name: [SimScale face names]}. A face records where it came
    from; the STL solid's name is matched first, and if SimScale does not
    report it, the faces are taken in file order (one face per solid)."""
    out = {n: [] for n in names}
    for m in mappings:
        # a solid in disconnected patches comes back as "car#1", "car#2", ...
        orig = " ".join(re.sub(r"#\d+$", "", str(v))
                        for o in (m.get("originate_from") or []) for v in o.values())
        hit = [n for n in names if n and (f" {n} " in f" {orig} " or orig.endswith(n))]
        if len(hit) == 1:
            out[hit[0]].append(m["name"])
    if all(out.values()):
        return out
    if len(mappings) == len(names):
        return {n: [m["name"]] for n, m in zip(names, mappings)}
    raise SimScaleError(
        "cannot map SimScale faces to boundary names: "
        + json.dumps([{k: m.get(k) for k in ("name", "originate_from")} for m in mappings[:20]],
                     default=str)[:2000])


def region_names(api, geometry_id: str) -> list:
    res = api["geometries"].get_geometry_mappings(_project(), geometry_id, _class="region",
                                                  limit=100, page=1)
    return [e.to_dict().get("name", "") for e in (res.embedded or [])]


def build_model(cfg: SimScaleConfig, faces: dict, regions: list, wheels: dict,
                x_moment_m: float = 0.0):
    """The Incompressible model. `wheels`: {part name: (centre m, omega rad/s)}."""
    from simscale_sdk import (
        AdvancedConcepts, AngularRotation, ComponentVectorFunction, ConstantFunction,
        DecimalVector, DimensionalDensity, DimensionalFunctionDimensionless, DimensionalFunctionPressure,
        DimensionalFunctionRotationSpeed, DimensionalKinematicViscosity, DimensionalPressure,
        DimensionalTime, DimensionalVectorFunctionSpeed, DimensionalVectorLength,
        DimensionalVectorSpeed, FieldCalculationsTurbulenceResultControl, FixedValuePBC,
        FixedValueVBC, FluidInitialConditions, FluidModel, FluidNumerics, FluidResultControls,
        FluidSimulationControl, FluidSolvers, ForcesMomentsResultControl, Incompressible,
        IncompressibleFluidMaterials, IncompressibleMaterial, MovingWallVBC,
        NewtonianViscosityModel, NoSlipVBC, PressureOutletBC, RelaxationFactor, ResidualControls,
        RotatingWallVBC, Schemes, ScotchDecomposeAlgorithm, SlipVBC, SymmetryBC,
        TimeStepWriteControl, Tolerance, TopologicalReference, TurbulenceIntensityTIBC,
        VelocityInletBC, WallBC, YPlusRASResultType)

    def topo(*labels):
        return TopologicalReference(entities=[f for n in labels for f in faces.get(n, [])], sets=[])
    U = cfg.speed_mps
    vel = DimensionalVectorFunctionSpeed(value=ComponentVectorFunction(
        x=ConstantFunction(value=U), y=ConstantFunction(value=0.0), z=ConstantFunction(value=0.0)),
        unit="m/s")
    parts = [n for n in faces if n not in BOX_FACES]
    bcs = [VelocityInletBC(name="inlet", velocity=FixedValueVBC(value=vel),
                           turbulence_intensity=TurbulenceIntensityTIBC(value=DimensionalFunctionDimensionless(
                               value=ConstantFunction(value=100.0 * cfg.turbulence_intensity), unit="%")),
                           topological_reference=topo("inlet")),
           PressureOutletBC(name="outlet", gauge_pressure=FixedValuePBC(
               value=DimensionalFunctionPressure(value=ConstantFunction(value=0.0), unit="Pa")),
               topological_reference=topo("outlet")),
           WallBC(name="ground", velocity=MovingWallVBC(value=DimensionalVectorSpeed(
               value=DecimalVector(x=U, y=0.0, z=0.0), unit="m/s")), topological_reference=topo("ground")),
           WallBC(name="side_top", velocity=SlipVBC(), topological_reference=topo("side", "top")),
           SymmetryBC(name="symmetry", topological_reference=topo("symmetry"))]
    for n in parts:
        if n in wheels:
            (cx, cy, cz), omega = wheels[n]
            v = RotatingWallVBC(rotation=AngularRotation(
                rotation_center=DimensionalVectorLength(value=DecimalVector(x=cx, y=cy, z=cz), unit="m"),
                rotation_axis=DimensionalVectorLength(value=DecimalVector(x=0.0, y=1.0, z=0.0), unit="m"),
                angular_velocity=DimensionalFunctionRotationSpeed(
                    value=ConstantFunction(value=omega), unit="rad/s")))
        else:
            v = NoSlipVBC()
        bcs.append(WallBC(name=n, velocity=v, topological_reference=topo(n)))
    cor = DimensionalVectorLength(value=DecimalVector(x=x_moment_m, y=0.0, z=0.0), unit="m")
    forces = [ForcesMomentsResultControl(
        name=f"F_{n}", center_of_rotation=cor, write_control=TimeStepWriteControl(write_interval=1),
        fraction_from_end=cfg.fraction_from_end, export_statistics=True, group_assignments=False,
        topological_reference=topo(n)) for n in parts]
    air = IncompressibleMaterial(
        name="Air", viscosity_model=NewtonianViscosityModel(
            kinematic_viscosity=DimensionalKinematicViscosity(value=cfg.kinematic_viscosity_m2s,
                                                              unit="m²/s")),
        density=DimensionalDensity(value=cfg.density_kgm3, unit="kg/m³"),
        topological_reference=TopologicalReference(entities=regions, sets=[]))
    tol = 1e-5
    return Incompressible(
        turbulence_model="KOMEGASST", model=FluidModel(),
        initial_conditions=FluidInitialConditions(), advanced_concepts=AdvancedConcepts(),
        materials=IncompressibleFluidMaterials(fluids=[air]), boundary_conditions=bcs,
        numerics=FluidNumerics(
            relaxation_factor=RelaxationFactor(),
            pressure_reference_value=DimensionalPressure(value=0, unit="Pa"),
            residual_controls=ResidualControls(
                velocity=Tolerance(absolute_tolerance=tol), pressure=Tolerance(absolute_tolerance=tol),
                turbulent_kinetic_energy=Tolerance(absolute_tolerance=tol),
                omega_dissipation_rate=Tolerance(absolute_tolerance=tol)),
            solvers=FluidSolvers(), schemes=Schemes()),
        simulation_control=FluidSimulationControl(
            end_time=DimensionalTime(value=cfg.iterations, unit="s"),
            delta_t=DimensionalTime(value=1, unit="s"),
            write_control=TimeStepWriteControl(write_interval=cfg.iterations),
            max_run_time=DimensionalTime(value=cfg.max_run_time_s, unit="s"),
            decompose_algorithm=ScotchDecomposeAlgorithm()),
        result_control=FluidResultControls(
            forces_moments=forces,
            field_calculations=[FieldCalculationsTurbulenceResultControl(
                name="yplus", result_type=YPlusRASResultType())]))


def build_mesh_model(cfg: SimScaleConfig, faces: dict, wheels: dict, primitive_ids: dict):
    from simscale_sdk import (AutomaticLayerOff, CustomMeshSizingSimmetrix, DimensionalLength,
                              FirstLayerGrowth, InsideRegionRefinementWithLength,
                              ManualMeshSizingSimmetrix, RegionRefinementWithLength,
                              SimmetrixBoundaryLayerRefinement, SimmetrixMeshingFluid,
                              SurfaceCustomSizing, TopologicalReference)
    car_h, wheel_h, far_h, near_h, farwake_h = MESH_PRESETS[cfg.resolution]
    parts = [n for n in faces if n not in BOX_FACES]
    body = [f for n in parts if n not in wheels for f in faces[n]]
    wheel = [f for n in parts if n in wheels for f in faces[n]]
    L = lambda v: DimensionalLength(value=v, unit="m")  # noqa: E731
    ref = [SurfaceCustomSizing(name="car_surface", sizing=CustomMeshSizingSimmetrix(
               default_size=L(car_h), min_size=L(car_h / 4)),
               topological_reference=TopologicalReference(entities=body, sets=[])),
           SurfaceCustomSizing(name="wheel_surface", sizing=CustomMeshSizingSimmetrix(
               default_size=L(wheel_h), min_size=L(wheel_h / 4)),
               topological_reference=TopologicalReference(entities=wheel, sets=[])),
           SimmetrixBoundaryLayerRefinement(
               name="wall_layers", layer_type=FirstLayerGrowth(
                   number_of_layers=cfg.n_layers, growth_rate=cfg.layer_growth,
                   first_layer_size=L(cfg.first_layer_m)),
               topological_reference=TopologicalReference(entities=body + wheel, sets=[]))]
    for name, h in (("wakeNear", near_h), ("wakeFar", farwake_h)):
        if name in primitive_ids:
            ref.append(RegionRefinementWithLength(
                name=name, refinement=InsideRegionRefinementWithLength(length=L(h)),
                geometry_primitive_uuids=[primitive_ids[name]]))
    return SimmetrixMeshingFluid(
        sizing=ManualMeshSizingSimmetrix(maximum_edge_length=L(far_h), minimum_edge_length=L(wheel_h / 4)),
        refinements=ref, automatic_layer_settings=AutomaticLayerOff(), physics_based_meshing=False)


def create_wake_boxes(api, bounds) -> dict:
    from simscale_sdk import DecimalVector, DimensionalVectorLength, GeometryPrimitive
    out = {}
    for name, lo, hi in wake_boxes(bounds):
        v = lambda p: DimensionalVectorLength(value=DecimalVector(x=p[0], y=p[1], z=p[2]), unit="m")  # noqa: E731
        res = api["sims"].create_geometry_primitive(_project(), GeometryPrimitive(
            type="CARTESIAN_BOX", name=name, min=v(lo), max=v(hi)))
        out[name] = res.geometry_primitive_id
    return out


def results_items(api, sim_id: str, run_id: str) -> list:
    res = api["runs"].get_simulation_run_results(_project(), sim_id, run_id)
    return [r.to_dict() if hasattr(r, "to_dict") else r for r in (res.embedded or [])]


def _download(api, url: str) -> str:
    import requests
    r = requests.get(url, headers={"X-API-KEY": os.environ["SIMSCALE_API_KEY"]}, timeout=120)
    if not r.ok:
        raise SimScaleError(f"result download failed: {r.status_code}")
    return r.text


def read_forces(api, sim_id: str, run_id: str, parts: list, fraction: float) -> dict:
    """{part: {"D_half_N", "L_half_N", "series": [...]}} from the per-part FORCE_PLOTs."""
    items = results_items(api, sim_id, run_id)
    out = {}
    for n in parts:
        it = next((r for r in items if r.get("name") == f"F_{n}"
                   and str(r.get("category", "")).upper() == "FORCE_PLOT"), None)
        if it is None:
            continue
        s = force_series(_download(api, it["download"]["url"]), fraction)
        out[n] = {"D_half_N": float(np.mean(s["fx"])), "L_half_N": float(np.mean(s["fz"])),
                  "series": s["fx"]}
    return out


def read_final_residual(api, sim_id: str, run_id: str) -> float:
    items = results_items(api, sim_id, run_id)
    it = next((r for r in items if str(r.get("category", "")).upper() == "RESIDUALS_PLOT"), None)
    if it is None:
        return float("nan")
    rows = list(csv.DictReader(io.StringIO(_download(api, it["download"]["url"]))))
    col = next((k for k in rows[0] if k and k.strip().lower().startswith("p")), None) if rows else None
    return float(rows[-1][col]) if col else float("nan")


def invoke(car_stl: str, cfg: SimScaleConfig, workdir: str) -> dict:
    """Run one half car on SimScale. Returns the same result dict the OpenFOAM
    case returns on main (half-car forces, health, per-part forces)."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    bounds = case_bounds(car_stl, cfg)
    fluid, labels, _parts = build_domain(car_stl, cfg)
    names = write_multisolid_stl(fluid, labels, str(work / "domain.stl"))
    api = _clients()
    tag = f"{cfg.run_name}_{int(time.time())}"
    geometry_id = import_geometry(api, str(work / "domain.stl"), tag, cfg.poll_s)
    ids = {"tag": tag, "project_id": _project(), "geometry_id": geometry_id}
    (work / "simscale.json").write_text(json.dumps(ids, indent=1))
    faces = faces_by_label(face_mapping(api, geometry_id), names)
    regions = region_names(api, geometry_id)
    R = {s["name"]: s for s in cfg.extra_surfaces if s.get("rotating")}
    wheels = {n: (tuple(s["rotating"]["origin"]), float(s["rotating"]["omega"])) for n, s in R.items()}
    from simscale_sdk import MeshOperation, SimulationRun, SimulationSpec
    spec = SimulationSpec(name=tag, geometry_id=geometry_id,
                          model=build_model(cfg, faces, regions, wheels))
    sim_id = api["sims"].create_simulation(_project(), spec).simulation_id
    prims = create_wake_boxes(api, bounds)
    op = api["mesh"].create_mesh_operation(_project(), MeshOperation(
        name=f"{tag}_mesh", geometry_id=geometry_id,
        model=build_mesh_model(cfg, faces, wheels, prims)))
    api["mesh"].start_mesh_operation(_project(), op.mesh_operation_id, simulation_id=sim_id)
    mesh_op = _poll(lambda: api["mesh"].get_mesh_operation(_project(), op.mesh_operation_id),
                    every=cfg.poll_s, what="mesh")
    s = api["sims"].get_simulation(_project(), sim_id)
    s.mesh_id = mesh_op.mesh_id
    api["sims"].update_simulation(_project(), sim_id, s)
    cells = None
    try:
        cells = api["meshes"].get_mesh(_project(), mesh_op.mesh_id).number_of_cells
    except Exception:  # noqa: BLE001 -- statistics are a report, not a result
        pass
    ids.update(simulation_id=sim_id, mesh_id=mesh_op.mesh_id, cells=cells)
    (work / "simscale.json").write_text(json.dumps(ids, indent=1))
    run = api["runs"].create_simulation_run(_project(), sim_id, SimulationRun(name="run"))
    api["runs"].start_simulation_run(_project(), sim_id, run.run_id)
    _poll(lambda: api["runs"].get_simulation_run(_project(), sim_id, run.run_id),
          every=cfg.poll_s, what="run")
    parts = [n for n in faces if n not in BOX_FACES]
    groups = read_forces(api, sim_id, run.run_id, parts, cfg.fraction_from_end)
    if not groups:
        raise SimScaleError("the run produced no force plots")
    total = [sum(v) for v in zip(*(g["series"] for g in groups.values()))]
    se, drift = force_mean_convergence(total)
    return {
        "D20_half": sum(g["D_half_N"] for g in groups.values()),
        "L_half": sum(g["L_half_N"] for g in groups.values()),
        "A_half": frontal_area_half(car_stl),
        "pitching_moment_half": 0.0,
        "residual_final": read_final_residual(api, sim_id, run.run_id),
        "force_mean_stderr": se, "force_drift": drift, "force_oscillation": None,
        "negative_volume_cells": 0, "y_plus_min": float("nan"), "y_plus_max": float("nan"),
        "courant_max": None,
        "groups": {n: {k: v for k, v in g.items() if k != "series"} for n, g in groups.items()},
        "simscale": {"geometry_id": geometry_id, "simulation_id": sim_id, "run_id": run.run_id,
                     "mesh_id": mesh_op.mesh_id, "cells": cells},
    }


def probe(domain_stl: str, names: list) -> dict:
    """Import one domain STL and return SimScale's face mapping against our
    boundary names. No mesh, no run, no core hours: the first live check."""
    api = _clients()
    gid = import_geometry(api, domain_stl, f"probe_{int(time.time())}")
    maps = face_mapping(api, gid)
    try:
        by = faces_by_label(maps, names)
    except SimScaleError as exc:
        by = {"_error": str(exc)}
    return {"project_id": _project(), "geometry_id": gid, "n_faces": len(maps),
            "regions": region_names(api, gid),
            "raw": [{k: m.get(k) for k in ("name", "originate_from")} for m in maps[:30]],
            "mapped": by}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SimScale checks that cost no core hours")
    ap.add_argument("cmd", choices=("domain", "probe"))
    ap.add_argument("run_dir", help="a run_car output folder (body_half.stl, parts/)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    A = json.loads((run / "parts" / "assembly.json").read_text())
    extra = tuple({"name": s_["name"], "stl": str(run / "parts" / Path(s_["stl"]).name),
                   "rotating": s_["rotating"]} for s_ in A["extra_surfaces"])
    cfg = SimScaleConfig(extra_surfaces=extra)
    fluid, labels, _p = build_domain(str(run / "body_half.stl"), cfg)
    names = write_multisolid_stl(fluid, labels, str(run / "domain.stl"))
    slits = thin_air(fluid, labels)
    print(json.dumps({"faces": len(fluid.faces), "closed": bool(fluid.is_watertight), "solids": names,
                      "thin_air": slits[:20], "n_thin_air": len(slits)}, default=str))
    if a.cmd == "probe":
        r = probe(str(run / "domain.stl"), names)
        print(json.dumps(r, indent=1, default=str))
        if "_error" in r["mapped"] or not r["regions"]:
            raise SystemExit("probe: faces not mapped or no fluid region")
