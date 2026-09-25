"""search.py -- grid search over wheel designs; writes designs.json and prints the best."""
import itertools
import json
from dataclasses import replace, asdict

import numpy as np

import wheel as W

TAB = json.load(open("race_T_vs_I.json"))
MS_PER_GMM2 = 1e3 * (np.interp(160, TAB["I"], TAB["T"]) - np.interp(120, TAB["I"], TAB["T"])) / 40.0
I_CAD = (W.CAD_FRONT.inertia + W.CAD_REAR.inertia) / 2

# Contact width at the T7.4 minimum plus 0.1 mm print margin (CAD: +0.25).
W_FRONT, W_REAR = 13.10, 17.10

results = []
grid = dict(
    t_rim=[0.30, 0.35, 0.40, 0.45, 0.50, 0.60],
    n_spokes=[3, 4, 5, 6, 7, 8, 10, 12],
    b_spoke=[0.6, 0.8, 1.0, 1.29],
    t_plate=[1.2, 1.6, 2.0, 2.93, 4.0, 6.0],
    r_fillet=[0.5, 1.0, 1.5],
    R=[14.05, 14.12],
    mat=["ABS_FDM", "PLA_FDM", "SLA_standard", "CF_tube_rim"],
)
for t, n, b, tp, rf, R, mat in itertools.product(*grid.values()):
    body = "SLA_standard" if mat in ("SLA_standard", "CF_tube_rim") else mat
    base = dict(R=R, t_rim=t, n_spokes=n, b_spoke=b, t_plate=tp, l_hub=tp,
                r_fillet=rf, rim_material=mat, body_material=body)
    f = W.Wheel(w=W_FRONT, **base)
    r = W.Wheel(w=W_REAR, **base)
    if not (f.printable() and r.printable()):
        continue
    ok = min(*f.check(W.CAD_FRONT), *r.check(W.CAD_REAR))
    if ok < 1.0:
        continue
    I = (f.inertia + r.inertia) / 2
    results.append(dict(
        material=mat, t_rim=t, n_spokes=n, b_spoke=b, t_plate=tp, r_fillet=rf, R=R,
        I_front=round(f.inertia, 1), I_rear=round(r.inertia, 1), I_mean=round(I, 1),
        m_front=round(f.mass, 3), m_rear=round(r.mass, 3),
        stiff_min_ratio=round(ok, 2),
        gain_ms=round((I - I_CAD) * MS_PER_GMM2, 1)))

results.sort(key=lambda d: d["I_mean"])
json.dump(dict(ms_per_gmm2=MS_PER_GMM2, I_cad=I_CAD, designs=results[:400]),
          open("designs.json", "w"), indent=1)
print(f"CAD mean I (model) {I_CAD:.1f} g.mm2; {MS_PER_GMM2:.3f} ms per g.mm2; {len(results)} feasible designs")
seen = set()
for d in results:
    key = d["material"]
    if key in seen:
        continue
    seen.add(key)
    print(d)
print("--- best 8 overall")
for d in results[:8]:
    print(d)
