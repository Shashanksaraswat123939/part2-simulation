"""
lean.py -- how much leaner does the car get when legal ballast is modelled?

Uses the project's OWN Stage-1 carve (bayesian_outer_search._level2_evaluate_unified)
and changes only two things, by patching module attributes at run time:

  1. BALLAST: the mass/COM state gains ballast in the legal container under the
     halo (Appendix ix), so the foam has to supply only
     48.0 g + margin - hardware - ballast. Ballast sits at the capsule centroid.
  2. BLOCK: the milled envelope is capped to the 223 x 65 x 50 mm Model Block
     (half-width 32.5 mm, top 1.5 + 50 mm). The nose (printed) keeps T8.5.1.

Three cars at the leader's scalars (W 120.3, x_front 46, d_halo 43.72):
  baseline  no ballast, carved to 48.2 g (what the pipeline does today)
  lead      container full of lead      (11.34 g/cm3)
  tungsten  container full of W alloy   (18.0 g/cm3)

Outputs in this folder: <car>_half.stl (ASCII, CFD input), <car>_full.stl,
metrics.json, views_<car>.png, compare.png.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent.parent / "part1-simulation"
sys.path[:0] = [str(P1), str(P1 / "sandbox")]

from coarse import use_spacing  # noqa: E402

use_spacing(2.0)

import bayesian_outer_search as bos   # noqa: E402
import bounding_volumes as bvm        # noqa: E402
import halo_pocket as hp              # noqa: E402
import unified_phi as up              # noqa: E402

W_MM, XF_MM, DH_MM = 120.3, 46.0, 43.72
TARGET_COMP_KG = 0.0482                 # the leader's competition mass
CARTRIDGE_KG = 0.023

# Capsule volume: rectangle + two half-discs, times depth.
_r = hp.BALLAST_SLOT_WIDTH_MM / 2
_straight = hp.BALLAST_SLOT_LENGTH_MM - 2 * _r
CAPSULE_MM3 = (_straight * hp.BALLAST_SLOT_WIDTH_MM + math.pi * _r**2) * hp.BALLAST_DEPTH_MM
BALLAST = {"baseline": 0.0,
           "lead": 11.34e-3 * CAPSULE_MM3 / 1000,
           "tungsten": 18.0e-3 * CAPSULE_MM3 / 1000}


def _block_envelope():
    re = bvm.default_rule_envelope()
    re.y_sidepod_outer_m = 0.0325                  # 65 mm block width
    top = re.z_floor_m + 0.050                     # 50 mm block height
    re.z_body_top_m = re.z_sidepod_top_m = re.z_rearpod_top_m = top
    return re


def _ballast_xz(geom):
    ref_A = geom.landmarks["ref_plane_A_m"]
    box = hp.compute_halo_pocket_box_m(ref_A, DH_MM)
    x = box["x_min_m"] + hp.BALLAST_CENTRE_FROM_POCKET_FRONT_MM / 1000
    z = box["z_min_m"] - hp.BALLAST_DEPTH_MM / 2000
    return x, z


def build(car: str, iters: int = 250):
    ballast_kg = BALLAST[car]
    orig_state = bos._unified_mass_com_state

    def state_with_ballast(geom):
        s = orig_state(geom)
        if s is None or ballast_kg == 0:
            return s
        bx, bz = _ballast_xz(geom)
        M = s["total_mass_kg"] + ballast_kg
        return {"total_mass_kg": M,
                "com_x_m": (s["com_x_m"] * s["total_mass_kg"] + bx * ballast_kg) / M,
                "com_z_m": (s["com_z_m"] * s["total_mass_kg"] + bz * ballast_kg) / M}

    bos._unified_mass_com_state = state_with_ballast
    bos.PROXY_MIN_MASS_KG = TARGET_COMP_KG - 0.0002      # descend to 48.2 g, not 53 g
    bos.PROXY_MASS_TARGET_MARGIN_KG = 0.0002
    up.default_rule_envelope = _block_envelope
    try:
        t0 = time.time()
        res, geom = bos._level2_evaluate_unified(
            W_MM, XF_MM, DH_MM, n_iters=iters, output_dir=str(HERE / f"_{car}"),
            eval_id=0, return_geom=True)
        st = state_with_ballast(geom)
    finally:
        bos._unified_mass_com_state = orig_state
    return geom, st, ballast_kg, time.time() - t0


def metrics(car, geom, st, ballast_kg, secs):
    comps = up.compute_mass_com(geom)
    body_g = sum(c.mass_kg for c in comps) * 1e3
    use_spacing(1.0)
    g1 = up.remap_geometry(geom)
    half = up.extract_half_surface(g1)
    half.vertices[half.vertices[:, 1] < 0, 1] = 0.0
    half.export(str(HERE / f"{car}_half.stl"), file_type="stl_ascii")
    full = up.extract_unified_surface(g1, allow_inaccessible=True)[0]
    full.export(str(HERE / f"{car}_full.stl"))
    use_spacing(2.0)
    # Frontal area: rasterise the full body onto the y-z plane at 0.25 mm.
    v = np.asarray(full.vertices)
    from matplotlib.path import Path as MPath
    px = 0.25e-3
    ys = np.arange(v[:, 1].min(), v[:, 1].max() + px, px)
    zs = np.arange(v[:, 2].min(), v[:, 2].max() + px, px)
    Y, Z = np.meshgrid(ys, zs)
    grid = np.zeros(Y.shape, bool)
    pts = np.c_[Y.ravel(), Z.ravel()]
    for tri in v[full.faces][:, :, 1:]:
        lo, hi = tri.min(0), tri.max(0)
        if (hi - lo).min() <= 0:
            continue
        i0, i1 = np.searchsorted(ys, [lo[0], hi[0]])
        j0, j1 = np.searchsorted(zs, [lo[1], hi[1]])
        if i1 <= i0 or j1 <= j0:
            continue
        sub = np.c_[Y[j0:j1, i0:i1].ravel(), Z[j0:j1, i0:i1].ravel()]
        inside = MPath(tri).contains_points(sub).reshape(j1 - j0, i1 - i0)
        grid[j0:j1, i0:i1] |= inside
    frontal_mm2 = grid.sum() * (px * 1e3) ** 2
    lab, n = __import__("scipy.ndimage", fromlist=["label"]).label(g1.phi.grid < 0)
    return dict(
        car=car, ballast_g=ballast_kg * 1e3, body_foam_and_nose_g=body_g,
        competition_g=(st["total_mass_kg"] - CARTRIDGE_KG) * 1e3,
        body_volume_cm3=abs(full.volume) * 1e6, wetted_area_mm2=full.area * 1e6,
        frontal_area_mm2=frontal_mm2,
        width_mm=(v[:, 1].max() - v[:, 1].min()) * 1e3,
        height_top_mm=v[:, 2].max() * 1e3, length_mm=(v[:, 0].max() - v[:, 0].min()) * 1e3,
        com_x_mm=st["com_x_m"] * 1e3, com_z_mm=st["com_z_m"] * 1e3,
        connected_bodies=int(n), half_faces=len(half.faces), seconds=round(secs, 1),
        components_g={c.name: round(c.mass_kg * 1e3, 2) for c in comps})


def render(cars):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    sys.path.insert(0, str(P1))
    import hardware_geometry as hg
    ref_A = (XF_MM - 16) / 1000
    try:
        hw = hg.build_all_hardware(W_MM, XF_MM, DH_MM, ref_A, (208.2, 0.0, 35.0))
    except Exception:
        hw = {}
    views = [("SIDE (x-z)", 0, 2), ("TOP (x-y)", 0, 1), ("FRONT (y-z)", 1, 2)]
    fig, axs = plt.subplots(len(cars), 3, figsize=(18, 4.2 * len(cars)))
    for r, car in enumerate(cars):
        body = trimesh.load(str(HERE / f"{car}_full.stl"))
        for c, (title, a, b) in enumerate(views):
            ax = axs[r, c]
            for name, m, col in [("body", body, "#9fb4c7")] + [
                    (k, v, "#c8553d" if "support" in k else ("#e0b000" if "halo" in k else "#333"))
                    for k, v in hw.items()]:
                tri = np.asarray(m.vertices)[m.faces][:, :, [a, b]] * 1000
                from matplotlib.collections import PolyCollection
                ax.add_collection(PolyCollection(tri, facecolor=col, edgecolor="none", alpha=0.9))
            ax.autoscale(); ax.set_aspect("equal")
            ax.set_title(f"{car}: {title}")
    plt.tight_layout()
    plt.savefig(HERE / "compare.png", dpi=80)


if __name__ == "__main__":
    out = []
    for car in ("baseline", "lead", "tungsten"):
        geom, st, bk, secs = build(car)
        m = metrics(car, geom, st, bk, secs)
        out.append(m)
        print(json.dumps(m))
    json.dump(out, open(HERE / "metrics.json", "w"), indent=2)
    render([m["car"] for m in out])
    print("capsule volume mm3", round(CAPSULE_MM3, 1))
