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
    # The car goes to SimScale as one rebuilt surface (see solid_car): filled
    # on a grid of `voxel_m`, crevices narrower than 2 x `close_m` sealed,
    # resurfaced within `surface_tol_m`. All far below the 0.3-2 mm surface
    # cells; the sealed crevices are glue-fillet sized.
    voxel_m: float = 1.25e-4
    close_m: float = 2.5e-4
    surface_tol_m: float = 3e-5
    facet_m: float = 1.0e-3            # triangle size of the rebuilt car surface
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
    car rebuilt as one surface (solid_car). Box faces are labelled by the side
    they lie on, car faces by the part whose surface is nearest."""
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

    car = solid_car(parts, cfg)
    lo, hi = domain_box(case_bounds(car_stl, cfg))
    box = man(trimesh.creation.box(bounds=[lo, hi]))
    owner = {box.original_id(): None, car.original_id(): "__car__"}
    fluid_m = box - car
    # Air the car seals in (the cockpit under the halo, 651 mm3 on the start
    # car) is a region of its own to SimScale and plays no part in the flow.
    fluid_m = max(fluid_m.decompose(), key=lambda m: m.volume())
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
    on_car = np.nonzero(labels == "__car__")[0]
    labels[on_car] = nearest_part(parts, fluid.triangles_center[on_car])
    # a face the cut left lying in a box plane belongs to that side, whatever
    # its origin (a handful per car after simplification)
    T = fluid.triangles
    for k, (axis, val) in enumerate(((0, lo[0]), (0, hi[0]), (1, lo[1]), (1, hi[1]), (2, lo[2]), (2, hi[2]))):
        labels[np.all(np.abs(T[:, :, axis] - val) < 1e-9, axis=1)] = BOX_FACES[k]
    labels = merge_label_islands(fluid, labels, 0.5e-6)
    return fluid, labels, list(parts)


def merge_label_islands(mesh, labels, min_area_m2: float) -> np.ndarray:
    """A patch of one label smaller than `min_area_m2` (a few triangles the
    nearest-part labelling gave to a neighbour part) joins the label it
    shares the longest boundary with. SimScale makes every patch a body."""
    import trimesh
    labels = np.array(labels, dtype=object)
    adj = mesh.face_adjacency
    elen = np.linalg.norm(np.diff(mesh.vertices[mesh.face_adjacency_edges], axis=1)[:, 0], axis=1)
    for _ in range(4):
        same = labels[adj[:, 0]] == labels[adj[:, 1]]
        comp = trimesh.graph.connected_component_labels(adj[same], node_count=len(mesh.faces))
        small = np.bincount(comp, weights=mesh.area_faces)[comp] < min_area_m2
        if not small.any():
            break
        votes = {}
        for (a, b), L in zip(adj[~same], elen[~same]):
            for x, y in ((a, b), (b, a)):
                if small[x]:
                    v = votes.setdefault(comp[x], {})
                    v[labels[y]] = v.get(labels[y], 0.0) + L
        for k, v in votes.items():
            labels[comp == k] = max(v, key=v.get)
    return labels


def solid_car(parts: dict, cfg: SimScaleConfig):
    """Every part as one closed, smooth surface (a manifold3d Manifold).

    The parts are separate closed surfaces that meet at near-tangent angles
    and micron gaps; SimScale cannot sew the resulting domain (slits, face
    intersections, micron edges and stalled imports, probes 2-10, 2026-10-01).
    So the union is filled on a grid (each slice even-odd, so a micron crack
    never holds a cell centre), crevices narrower than 2 x close_m are sealed,
    and the surface is rebuilt by marching cubes and simplified.

    The half parts do not meet the symmetry plane square-on (the body runs
    into it in a strip up to 0.14 mm off; mirrored, that is a groove the box
    cut slices at a glancing angle: 30 slits, run 3). So within close_m of
    the plane the cross-section at close_m is extruded straight through it,
    and on past the plane, which the box then cuts at exactly 90 degrees."""
    import manifold3d as m3
    from scipy import ndimage
    from skimage import draw, measure

    def man(t):
        return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                                   tri_verts=np.asarray(t.faces, np.uint32)))
    h = cfg.voxel_m
    r = max(1, int(round(cfg.close_m / h)))
    pad = (r + 3) * h
    U = m3.Manifold.batch_boolean([man(t) for t in parts.values()], m3.OpType.Add)
    b = U.bounding_box()
    lo, hi = np.array(b[:3]) - pad, np.array(b[3:]) + pad
    lo[1] = -pad                       # rows straddle y = 0 symmetrically
    xs, ys, zs = (np.arange(lo[k] + h / 2, hi[k], h) for k in range(3))
    occ = np.zeros((len(xs), len(ys), len(zs)), bool)
    for k, z in enumerate(zs):
        for poly in U.slice(z).to_polygons():
            occ[:, :, k] ^= draw.polygon2mask((len(xs), len(ys)), (np.asarray(poly) - lo[:2]) / h - 0.5)
    strip = ys < cfg.close_m
    occ[:, strip, :] = occ[:, [np.argmax(~strip)], :]
    g = np.indices((2 * r + 1,) * 3) - r
    ball = (g ** 2).sum(0) <= r * r
    occ = ndimage.binary_closing(occ, structure=ball)
    occ[:, strip, :] = occ[:, [np.argmax(~strip)], :]
    occ[:, :2, :] = False              # closed off beyond the plane, outside the box
    f = ndimage.gaussian_filter(occ.astype(np.float32), 0.7)
    del occ
    v, faces, _n, _ = measure.marching_cubes(f, 0.5)
    del f
    v = (v * h + lo + h / 2).astype(np.float32)
    M = m3.Manifold(m3.Mesh(vert_properties=v, tri_verts=faces[:, ::-1].astype(np.uint32)))
    if M.volume() < 0:
        M = m3.Manifold(m3.Mesh(vert_properties=v, tri_verts=faces.astype(np.uint32)))
    # specks of a few cells are marching-cubes debris; real parts are far bigger
    keep = [m for m in M.decompose() if m.volume() > 1e-9]
    M = m3.Manifold.batch_boolean(keep, m3.OpType.Add).simplify(cfg.surface_tol_m)
    o = M.to_mesh()
    v, faces = isotropic_remesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), cfg.facet_m)
    # Cut flush with the symmetry plane and the track here, where the slivers
    # the cuts leave can still be collapsed (vertices on a plane stay on it),
    # so the box boolean meets the car along existing edges.
    M = _manifold(v, faces).trim_by_plane((0, 1, 0), 0.0).trim_by_plane((0, 0, 1), 0.0)
    o = M.to_mesh()
    v, faces = np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts)
    pinned = (np.abs(v[:, 1]) < 1e-9) | (np.abs(v[:, 2]) < 1e-9)
    v, faces = collapse_short_edges(v, faces, 0.4 * cfg.facet_m, pinned=pinned)
    M = _manifold(v, faces)
    if M.status() != m3.Error.NoError:
        raise SimScaleError(f"rebuilt car surface: {M.status()}")
    return M.as_original()


def _manifold(v, f):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(vert_properties=np.array(v, np.float32, order="C"),
                               tri_verts=np.array(f, np.uint32, order="C")))


def _tri_normals(v, f):
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])


def collapse_short_edges(v, f, tol: float, passes: int = 10, pinned=None):
    """Edges shorter than `tol` collapse to their midpoint, or onto the
    `pinned` end if one end is pinned (a vertex on a cut plane stays on it;
    two pinned ends collapse to their midpoint, which is on the plane too).
    A collapse is skipped if it would pinch the surface (link condition) or
    turn a face over. Returns new (vertices, faces)."""
    v, f = np.array(v, float), np.array(f)
    pinned = np.zeros(len(v), bool) if pinned is None else np.array(pinned, bool)
    for _ in range(passes):
        E = np.unique(np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
        ln = np.linalg.norm(v[E[:, 0]] - v[E[:, 1]], axis=1)
        order = np.argsort(ln)
        short = E[order][ln[order] < tol]
        if not len(short):
            break
        vf = [[] for _ in range(len(v))]
        for i, t in enumerate(f):
            for w in t:
                vf[w].append(i)
        touched, dead, done = set(), np.zeros(len(f), bool), 0
        for a, b in short:
            if a in touched or b in touched:
                continue
            both = [i for i in vf[a] if b in f[i]]
            if len(both) != 2:
                continue
            ring_a = {w for i in vf[a] for w in f[i]} - {a}
            ring_b = {w for i in vf[b] for w in f[i]} - {b}
            if ring_a & ring_b != {w for i in both for w in f[i]} - {a, b}:
                continue
            moved = [i for i in set(vf[a]) | set(vf[b]) if i not in both]
            T = f[moved].copy()
            T[T == b] = a
            p = v[a] if pinned[a] and not pinned[b] else v[b] if pinned[b] and not pinned[a]                 else 0.5 * (v[a] + v[b])
            v2 = v[T]
            v2[T == a] = p
            n_new = np.cross(v2[:, 1] - v2[:, 0], v2[:, 2] - v2[:, 0])
            if np.any(np.einsum("ij,ij->i", n_new, _tri_normals(v, f[moved])) <= 0):
                continue
            v[a] = p
            pinned[a] = pinned[a] or pinned[b]
            f[moved] = T
            dead[both] = True
            touched |= ring_a | ring_b | {a, b}
            done += 1
        f = f[~dead]
        if not done:
            break
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def tangential_smooth(v, f, iters: int = 3, lam: float = 0.5):
    """Move each vertex toward the mean of its neighbours, along its tangent
    plane only (the shape stays, the triangles even out). A move that would
    turn a face over is dropped."""
    from scipy import sparse
    v = np.array(v, float)
    E = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                          shape=(len(v), len(v))).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(iters):
        n0 = _tri_normals(v, f)
        vn = np.zeros_like(v)
        np.add.at(vn, f.ravel(), np.repeat(n0, 3, axis=0))
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
        d = (A @ v) / deg[:, None] - v
        d -= np.einsum("ij,ij->i", d, vn)[:, None] * vn
        v2 = v + lam * d
        bad = np.einsum("ij,ij->i", _tri_normals(v2, f), n0) <= 0
        v2[np.unique(f[bad])] = v[np.unique(f[bad])]
        v = v2
    return v


def isotropic_remesh(v, f, length: float, passes: int = 3):
    """Triangles of about `length` everywhere, with no slivers: SimScale's
    facet checks flag long thin triangles (20 mm x 0.4 mm after plain
    simplification, run 4) and heal them by splitting faces, which breaks
    the sewing. Split long edges, collapse short ones, relax tangentially."""
    for _ in range(passes):
        M = _manifold(v, f).refine_to_length(1.4 * length)
        o = M.to_mesh()
        v, f = np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts)
        v, f = collapse_short_edges(v, f, 0.6 * length)
        v = tangential_smooth(v, f)
    return v, f


def nearest_part(parts: dict, points) -> np.ndarray:
    """The name of the part whose surface is nearest each point."""
    import trimesh
    from scipy.spatial import cKDTree
    pts, names = [], []
    for n, t in parts.items():
        # ~0.1 mm spacing: label boundaries land within a cell of the true junction
        k = int(max(2000, t.area / 1e-8))
        pts.append(trimesh.sample.sample_surface(t, k, seed=0)[0])
        names += [n] * k
    _d, idx = cKDTree(np.vstack(pts)).query(points)
    return np.array(names, dtype=object)[idx]


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
          every=15.0, what="operation", timeout_s=None):
    t0 = time.time()
    while True:
        obj = get()
        status = (getattr(obj, "status", "") or "").upper()
        if status in done:
            return obj
        if status in failed:
            raise SimScaleError(f"{what} {status}: {getattr(obj, 'failure_reason', '')}")
        if timeout_s and time.time() - t0 > timeout_s:
            raise SimScaleError(f"{what} still {status} after {timeout_s / 60:.0f} min")
        time.sleep(every)


def import_geometry(api, stl_path: str, name: str, every: float = 15.0, improve: bool = True,
                    sewing: bool = True) -> str:
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
            options=GeometryImportRequestOptions(facet_split=False, sewing=sewing, improve=improve,
                                                 optimize_for_lbm_solver=False))))
    try:
        # an import takes ~20 s; one sat at "Sewing 0 of 35 bodies" for 10+ min (2026-10-01)
        done = _poll(lambda: api["imports"].get_geometry_import(_project(), imp.geometry_import_id),
                     every=every, what="geometry import", timeout_s=1200)
    finally:
        log = api["imports"].get_geometry_import_event_log(_project(), imp.geometry_import_id)
        for e in log.entries or []:
            print(f"[simscale import] {e.to_dict()}", flush=True)
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
    print(f"[simscale] geometry {geometry_id}: {sum(map(len, faces.values()))} faces, "
          f"regions {regions}", flush=True)
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
    print(f"[simscale] mesh {mesh_op.mesh_id}: {cells} cells; solving", flush=True)
    (work / "simscale.json").write_text(json.dumps(ids, indent=1))
    run = api["runs"].create_simulation_run(_project(), sim_id, SimulationRun(name="run"))
    api["runs"].start_simulation_run(_project(), sim_id, run.run_id)
    _poll(lambda: api["runs"].get_simulation_run(_project(), sim_id, run.run_id),
          every=cfg.poll_s, what="run")
    print(f"[simscale] run {run.run_id} finished", flush=True)
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


def probe_variants(fluid, labels, workdir: str) -> list:
    """The same fluid mesh imported with its faces grouped several ways, to
    see where SimScale's sewing stops joining them (probe 12: 21 clean
    faces, no faults, yet 5 solids and 9 sheets). Free: no mesh, no run."""
    labels = np.asarray(labels)
    box = np.isin(labels, BOX_FACES)
    one = np.where(box, "box", "car")
    grouped = {"one": np.full(len(labels), "fluid", dtype=object),
               "car_box": one.astype(object),
               "box6_car1": np.where(box, labels, "car").astype(object),
               "all": labels}
    api = _clients()
    out = []
    for name, lab in grouped.items():
        for improve in ((True, False) if name == "all" else (True,)):
            path = str(Path(workdir) / f"variant_{name}.stl")
            names = write_multisolid_stl(fluid, lab, path)
            tag = f"variant_{name}{'' if improve else '_noimprove'}_{int(time.time())}"
            print(f"[variant] {tag}: {len(names)} solids", flush=True)
            try:
                gid = import_geometry(api, path, tag, improve=improve)
                maps = face_mapping(api, gid)
                regions = region_names(api, gid)
                bodies = sorted({m["name"].split("_")[0] for m in maps})
                res = {"variant": tag, "faces": len(maps), "bodies": len(bodies), "regions": len(regions)}
            except SimScaleError as exc:
                res = {"variant": tag, "error": str(exc)[:300]}
            print(f"[variant] {res}", flush=True)
            out.append(res)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SimScale checks that cost no core hours")
    ap.add_argument("cmd", choices=("domain", "probe", "variants"))
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
    if a.cmd == "variants":
        print(json.dumps(probe_variants(fluid, labels, str(run)), indent=1))
    if a.cmd == "probe":
        r = probe(str(run / "domain.stl"), names)
        print(json.dumps(r, indent=1, default=str))
        if "_error" in r["mapped"] or len(r["regions"]) != 1:
            raise SystemExit(f"probe: faces not mapped, or {len(r['regions'])} regions, not 1")
