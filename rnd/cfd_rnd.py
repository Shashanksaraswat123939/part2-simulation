"""
cfd_rnd.py -- CFD research runs for Part 2, designed to run on GitHub Actions.

Three questions, each answered with real OpenFOAM solves:

  1. INLET: how much does the legacy freestream (I = 5 %, omega from the car
     length -> nut/nu ~ 10,500) change drag against a near-laminar freestream
     (I = 0.5 %, nut/nu = 5)?
  2. NOISE: how much does drag move when the SAME car is shifted 1 micrometre,
     with the legacy per-STL meshing frame vs a fixed meshing frame?
  3. ADJOINT GAIN: does the adjoint's surface sensitivity predict the drag
     change of a uniform 0.3 mm inflation, and under which convention
     (per-area density vs per-point total), with which sign?

Subcommands (all paths explicit, one JSON or NPZ out per call):

  fixture  --out car.stl                       leader-like car from Part 1 code
  inflate  --stl car.stl --delta-mm 0.3 --out car_p.stl
  forward  --stl car.stl --inlet {legacy,fixed} --frame {stl,fixed}
           --shift-um 0 --res coarse --np 4 --out f.json
  adjoint  --stl car.stl --frame fixed --res medium --np 4 --out sens.npz
  summary  --dir results/ --out SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PART2 = HERE.parent
sys.path.insert(0, str(PART2))

# Fixed meshing frame (metres). Contains every legal car: nose tip at x >= 0,
# rear face <= x_front + W + 16 + 40 <= 252 mm, half-width <= 42.5 mm (T3.4),
# height <= 65 mm (T3.5).
FIXED_FRAME = ((0.0, 0.0, 0.0), (0.26, 0.045, 0.07))
RHO = 1.225
LEADER = dict(W_mm=120.3, x_front_mm=46.0, d_halo_mm=43.72)


# --------------------------------------------------------------------------- #
#  STL helpers
# --------------------------------------------------------------------------- #

def _load(stl):
    import trimesh
    return trimesh.load(str(stl), force="mesh", process=True)


def _export_ascii(mesh, out):
    mesh.vertices[mesh.vertices[:, 1] < 0.0, 1] = 0.0
    mesh.export(str(out), file_type="stl_ascii")


def cmd_fixture(a):
    """Leader-like car from the project's own Stage-1 path, half surface at 1 mm."""
    p1 = Path(a.part1).resolve()
    sys.path.insert(0, str(p1))
    sys.path.insert(0, str(p1 / "sandbox"))
    from coarse import use_spacing
    use_spacing(2.0)
    import bayesian_outer_search as bos
    import unified_phi as up
    t0 = time.time()
    res, geom = bos._level2_evaluate_unified(
        LEADER["W_mm"], LEADER["x_front_mm"], LEADER["d_halo_mm"],
        n_iters=100, output_dir=str(Path(a.out).parent / "_stage1"),
        eval_id=0, return_geom=True)
    if geom is None:
        raise SystemExit(f"Stage-1 carve failed: {res.lifecycle}")
    use_spacing(1.0)
    geom1 = up.remap_geometry(geom)
    mesh = up.extract_half_surface(geom1)
    _export_ascii(mesh, a.out)
    info = dict(stage1_mass_g=res.mass_kg * 1e3, faces=len(mesh.faces),
                watertight=bool(mesh.is_watertight),
                bounds_m=np.asarray(mesh.bounds).tolist(),
                seconds=round(time.time() - t0, 1), **LEADER)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def cmd_inflate(a):
    """Move every surface vertex delta along its outward normal.

    Vertices on the symmetry plane keep y = 0: their normal's y component is
    dropped, so they slide in-plane. Pure cap vertices (normal = -y) stay put.
    """
    import trimesh
    m = _load(a.stl)
    n = np.asarray(m.vertex_normals, dtype=float).copy()
    on_plane = np.abs(m.vertices[:, 1]) < 1e-9
    n[on_plane, 1] = 0.0
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 0.3
    n[ok] /= norm[ok, None]
    n[~ok] = 0.0
    d = a.delta_mm / 1000.0
    m2 = trimesh.Trimesh(m.vertices + d * n, m.faces, process=False)
    _export_ascii(m2, a.out)
    moved = float(np.mean(ok))
    print(json.dumps(dict(delta_mm=a.delta_mm, frac_vertices_moved=moved,
                          watertight=bool(m2.is_watertight),
                          dvol_mm3=float((abs(m2.volume) - abs(m.volume)) * 1e9))))


def _shift(stl, shift_um, out):
    import trimesh
    m = _load(stl)
    m2 = trimesh.Trimesh(m.vertices + np.array([shift_um * 1e-6, 0, 0]), m.faces,
                         process=False)
    _export_ascii(m2, out)
    return out


# --------------------------------------------------------------------------- #
#  Forward solve
# --------------------------------------------------------------------------- #

def _cells(checkmesh_log: str):
    m = re.search(r"cells:\s+(\d+)", checkmesh_log)
    return int(m.group(1)) if m else None


def cmd_forward(a):
    import openfoam_case as oc
    stl = a.stl
    if a.shift_um:
        stl = _shift(a.stl, a.shift_um, Path(a.out).with_suffix(".shifted.stl"))
    kw = dict(resolution=a.res, n_subdomains=a.np, max_iterations=a.iters)
    if a.inlet == "legacy":
        kw.update(turbulence_intensity=0.05, turbulent_viscosity_ratio=None)
    if a.frame == "fixed":
        kw.update(domain_reference_bounds=FIXED_FRAME)
    cfg = oc.OpenFOAMRunConfig(**kw)
    bashrc = oc.find_openfoam_bashrc()
    if bashrc is None:
        raise SystemExit("OpenFOAM not found")
    run = Path(a.workdir) / f"run_{Path(a.out).stem}"
    t0 = time.time()
    meta = oc.build_case(str(run), str(stl), cfg)
    logs = oc.run_stages(str(run), cfg, bashrc)
    (fx, fy, fz), (_mx, my, _mz) = oc.read_force_and_moment(str(run))
    se, drift = oc.read_force_mean_convergence(str(run))
    k, om, nut = oc.turbulence_inlet_values(cfg, meta["bounds"][1][0] - meta["bounds"][0][0])
    try:
        ypmin, ypmax = oc.parse_yplus_range(oc._read_yplus(str(run), logs["solver_log"]))
    except ValueError:
        ypmin = ypmax = None
    res = dict(
        label=Path(a.out).stem, inlet=a.inlet, frame=a.frame, shift_um=a.shift_um,
        res=a.res, iters=a.iters, np=a.np, stl=str(a.stl),
        D20_half_N=fx, L_half_N=fz, My_half=my, A_half_m2=meta["frontal_area_half"],
        force_mean_stderr=se, force_drift=drift,
        force_oscillation=oc.read_force_oscillation(str(run)),
        residual_final=oc.parse_final_p_residual(logs["solver_log"]),
        cells=_cells(logs["checkmesh_log"]),
        nut_over_nu_inlet=nut / cfg.kinematic_viscosity_m2s,
        yplus_min=ypmin, yplus_max=ypmax,
        seconds=round(time.time() - t0, 1),
    )
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- #
#  Adjoint solve -> sensitivity on the car patch with point areas
# --------------------------------------------------------------------------- #

def _read_boundary(text: str) -> dict:
    out = {}
    for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", text):
        body = m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            out[m.group(1)] = (int(sf.group(1)), int(nf.group(1)))
    return out


def _read_faces(text: str, start: int, n: int) -> list:
    """Faces start..start+n from an ASCII polyMesh/faces (faceList or faceCompactList)."""
    if "faceCompactList" in text[:2000]:
        lists = re.findall(r"\n(\d+)\s*\n\(\s*\n?([\d\s]*?)\n?\)", text)
        offs = np.array(lists[0][1].split(), dtype=np.int64)
        labs = np.array(lists[1][1].split(), dtype=np.int64)
        return [labs[offs[i]:offs[i + 1]] for i in range(start, start + n)]
    faces = re.findall(r"\d+\(([\d\s]+)\)", text)
    return [np.array(f.split(), dtype=np.int64) for f in faces[start:start + n]]


def cmd_adjoint(a):
    import openfoam_adjoint as oa
    import openfoam_case as oc
    kw = dict(resolution=a.res, n_subdomains=a.np,
              primal_iters=a.iters, adjoint_iters=a.iters, keep_run_dir=True)
    if a.frame == "fixed":
        kw.update(domain_reference_bounds=FIXED_FRAME)
    cfg = oa.AdjointRunConfig(**kw)
    bashrc = oc.find_openfoam_bashrc()
    run = Path(a.workdir) / "adjoint_run"
    t0 = time.time()
    oa.build_adjoint_case(str(run), a.stl, cfg)
    logs = oa.run_adjoint_stages(str(run), cfg, bashrc)
    oa.check_adjoint_magnitude(str(run), cfg)
    sens_file = oa.find_sensitivity_file(str(run))
    pts = oa.read_polymesh_points(str(run))
    _, s_all = oa.parse_sensitivity_points(
        sens_file.read_text(encoding="utf-8", errors="replace"), pts)
    poly = run / "constant" / "polyMesh"
    bnd = _read_boundary((poly / "boundary").read_text(errors="replace"))
    start, n = bnd["car"]
    faces = _read_faces((poly / "faces").read_text(errors="replace"), start, n)
    area = np.zeros(len(pts))
    nrm = np.zeros((len(pts), 3))
    for f in faces:
        p = pts[f]
        c = p.mean(axis=0)
        # Polygon area vector (points out of the fluid, i.e. INTO the car).
        av = 0.5 * np.sum(np.cross(p - c, np.roll(p, -1, axis=0) - c), axis=0)
        area[f] += np.linalg.norm(av) / len(f)
        nrm[f] += av / len(f)
    on = np.unique(np.concatenate(faces))
    nn = nrm[on] / np.maximum(np.linalg.norm(nrm[on], axis=1, keepdims=True), 1e-30)
    np.savez_compressed(a.out, points=pts[on], area=area[on], normal_into_car=nn,
                        sens=s_all[on], n_car_faces=n, seconds=time.time() - t0,
                        adjoint_log_tail=logs["solver_log"][-4000:])
    print(json.dumps(dict(car_points=int(len(on)), car_area_m2=float(area[on].sum()),
                          sens_rms=float(np.sqrt(np.mean(s_all[on] ** 2))),
                          sens_min=float(s_all[on].min()), sens_max=float(s_all[on].max()),
                          nonzero_frac_all_points=float(np.mean(s_all != 0)),
                          seconds=round(time.time() - t0, 1)), indent=2))


# --------------------------------------------------------------------------- #
#  Summary
# --------------------------------------------------------------------------- #

def cmd_summary(a):
    d = Path(a.dir)
    runs = {}
    for f in d.rglob("*.json"):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        if "D20_half_N" in j:
            runs[j["label"]] = j
    L = ["# CFD R&D results", ""]
    L += ["| run | inlet | frame | shift µm | res | cells | D20 full (N) | stderr | drift | osc p-p | p resid | ν_t/ν inlet | s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(runs):
        r = runs[k]
        L.append(f"| {k} | {r['inlet']} | {r['frame']} | {r['shift_um']} | {r['res']} | {r['cells']} | "
                 f"{2*r['D20_half_N']:.5f} | {r['force_mean_stderr']:.4f} | {r['force_drift']:.4f} | "
                 f"{(r['force_oscillation'] or 0):.3f} | {r['residual_final']:.2e} | "
                 f"{r['nut_over_nu_inlet']:.1f} | {r['seconds']:.0f} |")

    def rel(x, y):
        return abs(runs[x]["D20_half_N"] - runs[y]["D20_half_N"]) / abs(runs[y]["D20_half_N"])

    L += ["", "## Findings", ""]
    pairs = [("legacy inlet, per-STL frame", "F2_legacy_stl_shift", "F1_legacy_stl_base"),
             ("fixed inlet, per-STL frame", "F4_fixed_stl_shift", "F3_fixed_stl_base"),
             ("fixed inlet, fixed frame", "F6_fixed_frame_shift", "F5_fixed_frame_base")]
    for name, x, y in pairs:
        if x in runs and y in runs:
            L.append(f"- Remesh noise, {name}: 1 µm shift changes D20 by {100*rel(x, y):.3f} %.")
    if "F1_legacy_stl_base" in runs and "F3_fixed_stl_base" in runs:
        a1, a3 = runs["F1_legacy_stl_base"]["D20_half_N"], runs["F3_fixed_stl_base"]["D20_half_N"]
        L.append(f"- Inlet correction changes D20 by {100*(a3-a1)/a1:+.1f} % "
                 f"({2*a1:.4f} N legacy → {2*a3:.4f} N corrected).")

    sens = list(d.rglob("sens.npz"))
    if sens and all(k in runs for k in ("M_plus", "M_minus")):
        z = np.load(sens[0])
        s, A = z["sens"], z["area"]
        delta = float(a.delta_mm) / 1000.0
        meas = (runs["M_plus"]["D20_half_N"] - runs["M_minus"]["D20_half_N"]) / (2 * delta)
        pred_density = RHO * float(np.sum(s * A))
        pred_point = RHO * float(np.sum(s))
        L += ["", "## Adjoint gain check (half car, per metre of outward inflation)", "",
              "| quantity | value |", "|---|---|",
              f"| measured dD20_half/dδ (central difference, ±{a.delta_mm} mm) | {meas:.4e} N/m |",
              f"| predicted, per-area density ρ·Σ s·A | {pred_density:.4e} N/m |",
              f"| predicted, per-point total ρ·Σ s | {pred_point:.4e} N/m |",
              f"| ratio measured / density | {meas/pred_density if pred_density else float('nan'):+.3f} |",
              f"| ratio measured / per-point | {meas/pred_point if pred_point else float('nan'):+.3f} |",
              "",
              "A ratio near ±1 identifies the convention; its sign is the sign Part 1 must apply "
              "for OUTWARD motion (patch normals point into the car). The current pipeline "
              "multiplies by ρ·2·dT/dD20 and treats the value as a per-area density.",
              ""]
        if "M_base" in runs:
            b = runs["M_base"]["D20_half_N"]
            L.append(f"- Base medium D20 full {2*b:.5f} N; +δ {2*runs['M_plus']['D20_half_N']:.5f}; "
                     f"−δ {2*runs['M_minus']['D20_half_N']:.5f}.")
            curv = (runs["M_plus"]["D20_half_N"] + runs["M_minus"]["D20_half_N"] - 2 * b) / abs(b)
            L.append(f"- Asymmetry (+δ and −δ about base, relative): {100*curv:+.3f} % "
                     "(large values mean the difference is noise-dominated, not linear).")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fixture"); f.add_argument("--out", required=True)
    f.add_argument("--part1", default=str(PART2.parent / "part1-simulation"))
    i = sp.add_parser("inflate"); i.add_argument("--stl", required=True)
    i.add_argument("--delta-mm", type=float, required=True); i.add_argument("--out", required=True)
    w = sp.add_parser("forward")
    for x in ("--stl", "--out"):
        w.add_argument(x, required=True)
    w.add_argument("--inlet", choices=("legacy", "fixed"), default="fixed")
    w.add_argument("--frame", choices=("stl", "fixed"), default="fixed")
    w.add_argument("--shift-um", type=float, default=0.0)
    w.add_argument("--res", default="coarse"); w.add_argument("--np", type=int, default=4)
    w.add_argument("--iters", type=int, default=2000); w.add_argument("--workdir", default="work")
    j = sp.add_parser("adjoint")
    j.add_argument("--stl", required=True); j.add_argument("--out", required=True)
    j.add_argument("--frame", choices=("stl", "fixed"), default="fixed")
    j.add_argument("--res", default="medium"); j.add_argument("--np", type=int, default=4)
    j.add_argument("--iters", type=int, default=1000); j.add_argument("--workdir", default="work")
    s = sp.add_parser("summary"); s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True); s.add_argument("--delta-mm", default="0.3")
    a = ap.parse_args()
    {"fixture": cmd_fixture, "inflate": cmd_inflate, "forward": cmd_forward,
     "adjoint": cmd_adjoint, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
