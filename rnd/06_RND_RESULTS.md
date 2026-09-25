# R&D results — measured, not assumed

**Date:** 2026-09-25. Everything here was run, not estimated, unless marked.
**Where it ran:** CFD on GitHub Actions (ESI OpenFOAM v2412, 4 threads per job, up to 10
jobs in parallel), everything else locally.
**Code:** branch `rnd/cfd-validation` of `part2-simulation` (CFD fixes, `rnd/cfd_rnd.py`,
`.github/workflows/rnd-cfd.yml`, fixtures). Studies: `rnd/wheels`, `rnd/lean_body`,
`rnd/step_size` (reports beside each).

This file **overrides** the plan documents where they disagree. Section 7 lists the
corrections.

---

## 1. Headline

1. **Wheels are 64–75 % of the car's drag.** The body is 13–22 %. The pipeline has been
   optimising the body in a flow with no wheels, i.e. the smaller part of the problem, in the
   wrong flow.
2. **A leaner body alone makes the full car slower.** Ballast shrinks the body 45–75 % and cuts
   body-only drag 5–40 %, but with wheels in the flow the lean cars were 10–26 % *worse*: the
   wider body was shielding the wheels. The body's job is managing wheel flow.
3. **A front wing placed ahead of the front wheels costs nothing net.** A flat wing cut wheel
   drag 6.5 %, exactly its own drag. Since a wing is mandatory, that is where it goes; shaped
   (twisted) wings are the next gain.
4. **The optimiser's step is throttled 6–50× below its own cap**, mainly by the printed nose's
   density setting the time step and by one step per 20-minute CFD solve. Measured fixes give
   9× per solve locally, ~15–20× with a trust-radius loop.
5. **The thrust curve has about 60 % of the real impulse.** With it the model predicts
   1.52 s for the leader; scaled 1.5× it reproduces the 2022 World Finals fastest car
   (1.05 s predicted vs 1.06 s measured). Every ms figure in the plans is ~35–40 % too big;
   rankings do not change.
6. **Printed wheels are near their floor (−7 to −12 ms). A thin carbon-tube rim is worth
   −17 to −24 ms** at equal or better stiffness.
7. **GitHub Actions is a free CFD cluster for this project** while the repos are public:
   10 medium solves in ~25 minutes of wall time.

---

## 2. CFD experiments (three rounds, GitHub Actions)

Car: the leader's scalars (W 120.3, x_front 46, d_halo 43.72), carved by the project's own
Stage-1 code. Coarse ≈ 90–113 k cells, medium ≈ 383–400 k. 2000 SIMPLE iterations.

### 2.1 Inlet turbulence (corrected on the branch)

| inlet | ν_t/ν at inlet | D20 (full car, coarse) |
|---|---|---|
| legacy: I = 5 %, ω from car length | 8,887 | 0.2861 N |
| corrected: I = 0.5 %, ν_t/ν = 5 | 5 | 0.2907 N (+1.6 %) |

A real correctness bug, but on this bluff body it moves drag only 1.6 %. It will matter more
on a streamlined car where transition and skin friction count.

### 2.2 Noise — how repeatable is one number

| perturbation | frame | resolution | ΔD20 |
|---|---|---|---|
| 1 µm translation | per-STL (legacy) | coarse | 0.08 % |
| 1 µm translation | fixed | coarse | 0.93 % |
| 1e-5 stretch in x | per-STL | coarse | 0.27 % |
| 1e-5 stretch in x | fixed | coarse | 0.28 % |
| 1 µm translation | fixed | medium | **0.02 %** |

- The "fixed meshing frame" I proposed does **not** reduce noise; retracted.
- Remeshing noise is 0.3–1 % at coarse and ~0.02 % at medium for sub-micron changes.
- Averaged-force standard error: 0.4 % (coarse), 0.9 % (medium), **0.1–0.3 % once wheels are
  in the flow** (the car with wheels is steadier: peak-to-peak 2–9 % vs 9–28 %).
- The dangerous case is **drift**: a medium solve with 1.2e-3 residual (passes the 5e-3 gate)
  was still drifting 9 % at 2000 iterations. Gate on drift, not residual.

### 2.3 Wheels, supports and halo in the flow

Wheels as closed cylinders sunk 0.3 mm into the rolling road, CAD supports, halo; each its own
patch with its own force function object.

| car | total | body | wheels | supports + halo | wheels share |
|---|---|---|---|---|---|
| baseline body only | 0.2549 N | 0.2549 | — | — | — |
| + static wheels | 0.3816 N | 0.0854 | 0.2444 | 0.0518 | 64 % |
| + rotating wheels (coarse) | 0.3978 N | 0.0621 | 0.2831 | 0.0527 | 71 % |
| + rotating wheels (medium) | 0.3900 N | 0.0502 | 0.2911 | 0.0487 | 75 % |
| lead-ballast lean body, rotating | 0.4360 N | 0.0583 | 0.3263 | 0.0513 | 75 % |
| tungsten lean body, rotating | 0.5024 N | 0.0929 | 0.3477 | 0.0618 | 69 % |

Rotation adds 4 % over static wheels. A World Finals team reported front-wheel faces at 54 %
of their car's drag; this is consistent.

### 2.4 Front wing as wheel shield (medium, rotating wheels)

NACA 0015, chord 20 mm, LE at x = 6 mm, TE 5.9 mm clear of the front wheel (T7.9.1),
span to y = 38.5 mm, bottom 6.5 mm above the track.

| variant | total | wheels | wing+supports+halo | vs no wing |
|---|---|---|---|---|
| no wing | 0.3900 | 0.2911 | 0.0487 | — |
| flat | 0.3900 | 0.2721 | 0.0684 | **−0.01 %** |
| flat + endplate (5–20 mm tall) | 0.4136 | 0.2706 | 0.0847 | +6.1 % |
| 10° trailing-edge-up + endplate | 0.4036 | 0.2748 | 0.0805 | +3.5 % |

The wing's shielding exactly paid for its own drag; my simple endplates added more than they
saved. The top team's twisted front wing (turning flow over the wheel) measured −6 %. This is
where Part 4's wing optimisation should start.

### 2.5 Adjoint checks

- Converged cleanly on the hosted runner (medium: 1985 adjoint iterations, 9 min on 4 threads).
- **Units:** the per-area density convention is right; a per-point reading is off by ~10⁶.
- **Gain:** not pinned. Uniform ±0.3 mm (medium) and ±1 mm (coarse) inflations were
  non-linear: +1 mm cut drag 12 % while −1 mm changed it 0.5 %. The blob's wake switches state
  at millimetre scale, and a uniform inflation's front and rear contributions cancel. Next
  test: a localised bump in a high-sensitivity region, on a streamlined body with wheels.
- The adjoint builder does not yet support the wheel/hardware patches; that is the next code
  change (PART2 §4).

---

## 3. Wheels (local)

Measured on the v2 CAD: front 0.806 g / 125 g·mm², rear 0.951 g / 153 g·mm²; a 0.40 mm rim
shell holds 61–67 % of the mass. Mean I = **1.39e-7 kg·m²** — the value the race objective
should use (a default elsewhere is 1e-6).

| best design | mean I | race time (project thrust) | ×0.6 for real thrust |
|---|---|---|---|
| printed ABS, stiffness ≥ today | 117 | −6.9 ms | ≈ −4 ms |
| printed SLA, stiffness ≥ 0.5× today | 102 | −12.0 ms | ≈ −7 ms |
| 0.20 mm carbon rim, 2.9× stiffer | 86 | −17.4 ms | ≈ −10 ms |
| 0.15 mm carbon rim, 1.2× stiffer | 66 | −23.9 ms | ≈ −14 ms |

Keep stiffness: a World Finals team measured stiff 1.28 g wheels beating flexible 0.80 g ones.
Bearings matter as much: the same team measured hybrid ceramic bearings (SMR73C) at 54 s
spin-down vs 9.9 s for steel. Details: `rnd/wheels/WHEEL_RND.md`.

---

## 4. Lean body with ballast (local + CFD)

| | baseline | lead | tungsten |
|---|---|---|---|
| ballast | 0 | 15.8 g | 25.1 g |
| body volume | 186 cm³ | 101 cm³ | 47 cm³ |
| frontal area | 1996 mm² | 1545 mm² | 1218 mm² |
| body-only D20 | 0.2549 N | 0.2412 N | 0.1518 N |
| full car D20 (rotating wheels) | 0.3978 N | 0.4360 N | 0.5024 N |

Ballast works as a mass tool. A mass-only carve does not produce a good full car. Once the
body shrinks, the regulated solids set the shape: the 55 mm wide end of the T4.2 cargo
(wide-end forward, z 14–24 mm), the forced loft deck and the cartridge shroud (top fixed at
47.6 mm). Details: `rnd/lean_body/LEAN_BODY_RND.md`.

---

## 5. Optimiser step size (local, real adjoint field)

| variant | median surface move per update (cap 0.30 mm) |
|---|---|
| production | 0.049 mm |
| drag term alone | 0.006 mm |
| nose density not dominating dt | 0.132 mm |
| + band-p90 dt, 2 mm smoothing | 0.090 mm (rear pod sets dt) |
| + 5 sub-steps | **0.451 mm** |

Redistancing is 26 of 34 s per update. Details and the exact change: `rnd/step_size/STEP_SIZE_RND.md`.

---

## 6. External benchmarks

| quantity | value | source |
|---|---|---|
| world record, 20 m | 0.916 s (2016) | Warwick press release via web search |
| 2022 World Finals fastest average | 1.060 s (Hydron) | Raceteq / Formula1.com |
| Hydron complete-car drag | 0.23–0.26 N (Fluent), 0.216 N wind tunnel | Hydron portfolio |
| Hydron front-wheel faces share | 54 % of total drag | Hydron portfolio |
| Hydron twisted front wing | −0.0169 N (−6 %) vs flat | Hydron portfolio |
| Hydron wheel test | 1.28 g stiff 1.182 s vs 0.80 g flexible 1.185 s | Hydron portfolio |
| Hydron bearings | hybrid ceramic 54 s spin vs steel 9.9 s | Hydron portfolio |
| CO2 thrust | measured at 1 ms, ~0.5 s duration, 20 % cartridge scatter | Hydron portfolio |
| thrust applies over | ~1/3 of the track | Hydron portfolio |
| COM target | within 5 mm of the thrust line, as far rearward as possible | Hydron portfolio |

Our leader's **body alone** (0.29 N at 20 m/s) is already draggier than Hydron's **complete**
car; with wheels it is 0.39–0.40 N. (The portfolio does not state its CFD speed; the comparison
assumes the usual 20 m/s.) At the realistic 0.29 s/N that gap is worth ~45–55 ms.

---

## 7. Corrections to the plan documents

| document | claim | correction |
|---|---|---|
| 00, PART2 #1 | inlet turbulence invalidates every number, fix first | real bug, measured +1.6 % drag; still fix, no longer "first" |
| PART2 #7b, §3.2 | fixed meshing frame is the biggest noise reduction | retracted: no effect on shape-change noise |
| 00 §3, PART4 §1 | wheel inertia worth −48 ms | that is the zero-inertia bound; achievable −7 to −24 ms (×0.6 for real thrust) |
| 00 §3 | all ms figures | ~35–40 % too large (thrust curve); ranking unchanged |
| 00 §2, PART1 §3 | ballast lean body → skin follows drag | necessary but not sufficient: the body must be optimised with wheels in the CFD, or it gets worse |
| PART4 §4.2 | front wing's job is wheel shielding | confirmed as net-free for a flat wing; endplates as tried here hurt; shaped wings are the lever |
| PART2 #7a | half-car ×2 double-counts | not settled; the gain check was noise- and regime-limited |
| PART1 A-list | — | add: `hardware_geometry.build_wheel_assembly` asserts on the current wheel STLs (their width is along x) |

---

## 8. Room for upgrade, ranked by measured evidence

1. **Put wheels, supports, halo and wings in every CFD, and optimise the body in that flow.**
   Wheels are 64–75 % of drag; body-only optimisation moved the wrong 15–20 %. (Code for the
   forward case exists on the branch; the adjoint needs the same patches.)
2. **Front wheel flow management**: wing shape and position ahead of the front wheels, nose
   shape turning flow away from the wheel faces, rear wheels shielded by body width. Measured
   net-zero with a flat wing; the top team found −6 % with a twisted wing.
3. **Measure the thrust curve** (load cell, 1 ms). It sets every sensitivity; ours has ~60 % of
   real impulse.
4. **Carbon-rim wheels + tested bearings**: −10 to −14 ms (real thrust), no CFD.
5. **Optimiser mechanics**: nose out of the mass velocity, band-p90 dt, smoothing, sub-steps to
   a trust radius, narrow-band reinit. ~15–20× more motion per solve.
6. **Gate solves on drift and stderr**, not residual.
7. **Use GitHub Actions for candidate evaluation**: 10+ parallel medium solves per 25 minutes.
   Caveats: public repos make code and results public; 6-hour job cap; fair-use limits.
8. **Streamlined parametric start** instead of the carved blob: the blob's wake switches state
   at mm scale, which defeats gradients and finite differences alike.

## 9. Reproduce

```bash
gh workflow run rnd-cfd -R Shashanksaraswat123939/part2-simulation --ref rnd/cfd-validation
```

Edit the matrix in `.github/workflows/rnd-cfd.yml`; each entry is one `rnd/cfd_rnd.py` call.
Results land as artifacts and in the run's summary page.
