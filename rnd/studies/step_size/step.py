"""
step.py -- why does one CFD+adjoint iteration barely move the car, and what fixes it?

Uses the REAL adjoint sensitivity computed on GitHub Actions (rnd-cfd run, ADJ
job, medium mesh) on the same leader-like car, mapped to the STL vertices the
way production does (nearest CFD point within 5 mm, else 0), scaled the way
cfd_wrapper does (x w_D20 x rho x 2.0).

Each variant performs ONE shape update from the same starting field and we
measure how far the zero level set moved (median / p90 over the interface),
how much mass moved, and what set the time step.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent.parent / "part1-simulation"
sys.path[:0] = [str(P1), str(P1 / "sandbox")]
from coarse import use_spacing  # noqa: E402

RHO, W_D20, HALF = 1.225, 0.449, 2.0
DT_DMASS, DT_DH = 16.7, -0.092


def build(sens_npz: str):
    use_spacing(2.0)
    import bayesian_outer_search as bos
    import unified_phi as up
    _res, geom = bos._level2_evaluate_unified(120.3, 46.0, 43.72, n_iters=100,
                                              output_dir=str(HERE / "_s1"), eval_id=0,
                                              return_geom=True)
    use_spacing(1.0)
    g = up.remap_geometry(geom)
    mesh = up.extract_half_surface(g)
    z = np.load(sens_npz)
    from scipy.spatial import cKDTree
    d, i = cKDTree(z["points"]).query(mesh.vertices)
    s = z["sens"][i].astype(float)
    s[d > 5e-3] = 0.0
    sens = s * W_D20 * RHO * HALF
    return g, mesh, sens, dict(frozen_frac=float(np.mean(d > 5e-3)),
                               median_match_mm=float(np.median(d) * 1e3))


def mass_report(g):
    import unified_phi as up
    comps = up.compute_mass_com(g)
    m = sum(c.mass_kg for c in comps) + 0.0385   # + fixed hardware incl. cartridge
    x = sum(c.mass_kg * c.com_x_m for c in comps) / m
    z = sum(c.mass_kg * c.com_z_m for c in comps) / m
    return {"total_mass_kg": m, "com_x_m": x, "com_z_m": z}


def surface_smooth(mesh, s, length_m):
    """Helmholtz filter on the surface: (I + L^2 * Lap) s_f = s, uniform graph Laplacian."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = len(mesh.vertices)
    e = mesh.edges_unique
    el = mesh.edges_unique_length
    w = 1.0 / np.maximum(el, 1e-9) ** 2
    A = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    Lap = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    M = sp.identity(n) + (length_m ** 2) * Lap
    return spla.spsolve(M.tocsc(), s)


def one_update(g0, mesh, sens, variant: dict):
    """One shape update in the style of apply_adjoint_to_unified, with switches."""
    import phi_updater as pu
    import unified_phi as up
    g = copy.deepcopy(g0)
    phi = g.phi
    dx = pu.GRID_SPACING_M
    t = {}
    t0 = time.time(); pu.reinitialise_sdf(phi); t["reinit"] = time.time() - t0
    phi_before = phi.grid.astype(np.float64).copy()
    s = -sens.copy()                                   # descent sign, as production
    if variant.get("smooth_mm"):
        s = surface_smooth(mesh, s, variant["smooth_mm"] / 1000)
    nz = np.abs(s)[np.abs(s) > 0]
    if nz.size:
        cap = float(np.percentile(nz, 99.9)); s = np.clip(s, -cap, cap)
    v = np.asarray(mesh.vertices)
    t0 = time.time()
    vel = pu._splat_vertex_sensitivity_to_grid(s, v, phi)
    vl = v.copy(); vl[:, 1] *= -1
    vel = 0.5 * (vel + pu._splat_vertex_sensitivity_to_grid(s, vl, phi))
    t["splat"] = time.time() - t0
    t0 = time.time(); aero = pu.extend_velocity(phi.grid.astype(np.float64), vel); t["extend"] = time.time() - t0
    rho = up.density_field(g)
    if variant.get("uniform_density"):
        rho = np.where(rho > 0, 163.0, 0.0)
    mr = mass_report(g)
    masscom = pu.scalar_objective_velocity(phi, rho, {"dT_dmass": DT_DMASS, "dT_dh_com": DT_DH,
                                                      "dT_dx_com": 0.0}, mr)
    comb = variant.get("w_aero", 1.0) * aero + variant.get("w_mass", 1.0) * masscom
    band = np.abs(phi.grid) < 2 * dx
    a_rms = float(np.sqrt(np.mean(aero[band] ** 2))); m_rms = float(np.sqrt(np.mean(masscom[band] ** 2)))
    absc = np.abs(comb)
    if variant.get("band_dt"):
        ref = float(np.percentile(absc[band], variant["band_dt"]))
        comb = np.clip(comb, -3 * ref, 3 * ref)
        vmax = 3 * ref
    else:
        nzc = absc[absc > 0]
        comb = np.clip(comb, -np.percentile(nzc, 99.9), np.percentile(nzc, 99.9))
        vmax = float(np.max(np.abs(comb)))
    # where is the max?  label of the arg-max cell
    am = np.unravel_index(np.argmax(np.abs(comb) * band), comb.shape)
    who = up.LABEL_NAMES.get(int(g.labels[am]), "none")
    dt = pu.CFL_NUMBER * dx / vmax
    steps = variant.get("substeps", 1)
    t0 = time.time()
    for _ in range(steps):
        pu.hj_update(phi, comb, dt)
    t["hj"] = time.time() - t0
    up.enforce_symmetry(g)
    t0 = time.time(); filled = up.enforce_machinability(g); t["machinability"] = time.time() - t0
    # displacement of the old interface = -(phi_new - phi_old) at old |phi| < dx
    iface = np.abs(phi_before) < dx
    disp = -(phi.grid.astype(np.float64) - phi_before)[iface]
    mr2 = mass_report(g)
    return dict(
        variant=variant, dt=dt, vmax_set_by=who, aero_rms_band=a_rms, mass_rms_band=m_rms,
        aero_share=a_rms / (a_rms + m_rms),
        disp_median_mm=float(np.median(np.abs(disp)) * 1e3),
        disp_p90_mm=float(np.percentile(np.abs(disp), 90) * 1e3),
        disp_max_mm=float(np.abs(disp).max() * 1e3),
        frac_iface_moved_over_0p05mm=float(np.mean(np.abs(disp) > 5e-5)),
        dmass_g=(mr2["total_mass_kg"] - mr["total_mass_kg"]) * 1e3,
        machinability_filled=int(filled), seconds={k: round(v, 2) for k, v in t.items()})


if __name__ == "__main__":
    g, mesh, sens, info = build(sys.argv[1])
    print(json.dumps(info))
    variants = [
        {"name": "production (1 step)"},
        {"name": "aero only", "w_mass": 0.0},
        {"name": "uniform density (nose not 6x)", "uniform_density": True},
        {"name": "band p90 dt", "band_dt": 90},
        {"name": "smooth 2 mm", "smooth_mm": 2.0},
        {"name": "smooth 2 mm + band p90 dt", "smooth_mm": 2.0, "band_dt": 90},
        {"name": "smooth 2 mm + band p90 dt + uniform density", "smooth_mm": 2.0, "band_dt": 90,
         "uniform_density": True},
        {"name": "smooth + band dt + uniform density, 5 substeps", "smooth_mm": 2.0, "band_dt": 90,
         "uniform_density": True, "substeps": 5},
    ]
    out = []
    for v in variants:
        r = one_update(g, mesh, sens, v)
        out.append(r)
        print(json.dumps(r))
    json.dump(dict(info=info, results=out), open(HERE / "step_results.json", "w"), indent=2)
