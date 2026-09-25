# Master plan — from "blob that passes the gates" to the fastest legal car we can build

**Date:** 2026-09-25. **Status:** plan. No code in Parts 1–3 has been changed.
**Read in this order:** this file → **06_RND_RESULTS (measured; overrides this file where they disagree)** → PART1 → PART2 → PART3 → PART4 → PART5.

> **Update after R&D (2026-09-25, same day):** real CFD on GitHub Actions showed wheels are 64–75 % of the car's drag, a mass-only lean body makes the full car slower, the inlet fix is worth only +1.6 % drag, the fixed meshing frame does not reduce noise, and the thrust curve has ~60 % of real impulse (every ms figure below is ~35–40 % too large). See 06 §7 for the line-by-line corrections.

| file | scope |
|---|---|
| `00_MASTER_PLAN.md` | diagnosis, the core idea, priorities, roadmap, DESIGN_features.md review |
| `PART1_UPGRADES.md` | geometry: parametric body + organic skin, ballast, floor, hollowing, masks |
| `PART2_UPGRADES.md` | physics: CFD fidelity, noise floor, adjoint, objective inputs |
| `PART3_UPGRADES.md` | optimiser: efficiency, search strategy, convergence, records |
| `PART4_COMPONENTS_PLAN.md` | new: wheels, hubcaps, supports, wings, nose, tether guides |
| `PART5_DESIGN_ADVISOR_PLAN.md` | new: assembly search, legality engine, structural recommendations, CAD handoff |
| `06_RND_RESULTS.md` | measured results: CFD on GitHub Actions, wheels, lean body, optimiser step, benchmarks |

---

## 1. Why the car looks like it does

The leader image is not a bad optimum. It is what this formulation produces, for five structural reasons.

### 1.1 Foam is being used as ballast (the root cause)

The pipeline has no ballast variable. When the car is under the 48 g floor (T3.6), the barrier in both
stages pushes **foam** back out (`t36_descent_gradient` in Part 3, `_proxy_objective_gradients` in Part 1).
Model Block foam is 0.163 g/cm³, so making weight with it takes a lot of volume.

The regulations allow ballast in the container under the halo (T1.22, Appendix ix). That capsule is
12.7 × 20 × 6.35 mm, about 1.39 cm³:

| ballast | fits in the container | foam it replaces |
|---|---|---|
| lead (11.3 g/cm³) | ≈ 15.8 g | ≈ 97 cm³ |
| tungsten alloy (≈ 18 g/cm³) | ≈ 25 g | ≈ 153 cm³ |

The leader carries about 32.7 g of body (48.2 g minus 15.5 g of hardware), roughly 200 cm³ of foam.
With the container full of lead the body only needs about 16.7 g, roughly half the volume.
With tungsten it needs far less. **The car is fat because foam is doing ballast's job.**
Every other defect below is made worse by this one.

### 1.2 Mass steers the shape; aero cannot

`apply_adjoint_to_unified` sums the drag velocity and `dT/dmass × ρ` into one surface velocity.
The code's own logging reports mass/COM ≈ 94 % of the update and aero ≈ 5–6 %.
At the floor the mass term is not physics any more. It is the barrier, and it wins.

With a ballast model, a car at the floor has **dT/d(body volume) = 0** as long as ballast can absorb the
difference. The outer skin is then free to follow drag alone. That single change is what "increase aero
sensitivity" should mean. Multiplying the aero weight instead would only amplify CFD noise.

### 1.3 The shape can barely move, so the envelope is the design

The body starts as a full block inside the rule envelope. Each CFD+adjoint iteration moves the surface at
most CFL 0.3 × 0.5 mm = 0.15 mm, so a 25-iteration budget moves it less than 4 mm. What you see is the rule
envelope with bites taken out by the hard masks: wheel keep-out columns with 4 mm margin, halo visibility
shadows, the loft ceiling and forced-solid deck, the canister safety tube and contact shell. The humps in
the image are masks, not aerodynamics.

### 1.4 The CFD sees the wrong car

The CFD STL is the body only: no wheels, supports, halo, helmet, wings, or protruding cartridge.
On this car the front wheels sit directly in the free stream ahead of the pods. The adjoint is optimising a
body in flow that does not exist on race day.

### 1.5 Differences are smaller than the noise

Measured in the code comments: force peak-to-peak 18–27 % on the bluff body, mean still drifting 2–4 % per
decade of iterations, remesh spread 1.2–3 % from sub-micron geometry changes. A 0.15 mm step changes drag by
much less than that, so the optimiser cannot tell whether a step helped. In practice the step is smaller still:
adjoint speckle throttles the time step, and the car moved about 0.057 g per iteration, roughly 40× below the
cap (PART3 #1). The leader, at 48.20 g, never reached the aero phase at all.

All of these numbers were measured with a freestream about 10,500 times too viscous (§4), so the true noise
floor is unknown until that is fixed.

---

## 2. The core idea

**Parametric skeleton, organic skin, and physical constraints handled as constraints, not as gradients.**

1. **Mass is a constraint, met by ballast first and hollowing second.** Competition mass is held at
   48.0 g + margin. Ballast fills any shortfall up to container capacity. If the aero-best outer shape is
   *heavier* than 48 g, the body is hollowed with open floor channels cut from below (a covered void is not
   manufacturable as one milled piece, T1.3/T4.1, and a cover would be an extra component, T3.3).
   Either way the outer surface answers to drag.
2. **The body starts from a sensible aerodynamic shape**, defined by ~30–60 parameters (lofted sections along
   a spine, floor line, top line, pod switches). The level set is a bounded refinement on top of it, not the
   whole design.
3. **Big moves come from parameters, fine detail from the level set.** Adjoint sensitivities are projected
   onto the parameters (dJ/dp), so one CFD iteration can move the body by millimetres. The level set only
   polishes, inside a trust region.
4. **Consecutive solves share a mesh.** Use OpenFOAM's own morphing optimisation
   (`adjointOptimisationFoam` `steadyOptimisation` with `volumetricBSplines` morphing boxes, `shapeFI`
   sensitivities, BFGS/SQP with line search, volume constraint through `partialVolume`). Documented in ESI's
   manual for v2312; confirm on the v2412 install. Remesh only between stages.
5. **Every part of the car is in the CFD.** Wheels as rotating walls (`adjointRotatingWallVelocity` exists
   for the adjoint), supports, halo and helmet, wings, protruding cartridge.
6. **Components are exact parametric geometry** (Part 4), assembled at the mesh level. Thin parts do not
   belong on the body's 0.5 mm grid.
7. **Structure is a decision, not an accident.** Sidepods, floor channels, wing element count, axle through
   body: explicit switches that Part 5 evaluates at low fidelity and recommends with numbers.

---

## 3. What is worth what — race-time ledger

From the project's own race objective at the leader's operating point (µ = 0.010, D20 = 0.287 N,
48.2 g + 23 g cartridge). Absolute times depend on the placeholder thrust curve; the differences are what matter.

| lever | size | where it is handled |
|---|---|---|
| wheel inertia: today's rims → zero | −48 ms | Part 4 §2, **no CFD** |
| wheel inertia: rim mass halved | ≈ −23 ms | Part 4 §2 |
| drag −10 % | −12.8 ms | Parts 1, 2, 4 |
| being 1 g under the floor | +16.7 ms (plus the regulation penalty) | Part 1 ballast model |
| µ doubled 0.010 → 0.020 | +7.1 ms | Part 4 §2.5, bearings |
| 0.1 N downforce | +0.45 ms | ignore |
| a capped hubcap, cost side | ≈ +2 ms per wheel | Part 4 §2.4 |

Priority order that follows: **wheels and bearings first (cheap, large, no CFD), then a lean body enabled
by ballast, then drag with every part in the CFD, then wings for legality at minimum drag.**

---

## 4. Fix these first — they invalidate results or legality

1. **The CFD inlet makes the air ≈ 10,500 times too viscous** (PART2 #1). Inlet turbulence from 5 % intensity
   with the car length as the length scale gives ν_t = 0.156 m²/s against ν = 1.48e-5, and it barely decays
   before the car. Every drag number, oscillation figure and adjoint field in the project was measured in that
   flow. One function and one test fix it; then re-measure everything.
2. **The milled envelope is bigger than the Model Block** (PART1 A1). The block is 223 × 65 × 50 mm; the body
   envelope reaches 71 mm wide and 63.5 mm tall.
3. **The rear wheel keep-out cuts the T5.5 chamber wall to 1.25–2.0 mm** against a 3.0 mm minimum, a Safety
   regulation (PART1 A2).
4. **µ and wheel MOI have no production value.** Defaults in reach are µ = 0.4 or 0.010 and I = 1e-6 or 1e-7.
   At µ = 0.4 friction is comparable to all the drag (PART2 #4, PART3 #18).
5. **Wings, tether guides and the decal areas are not modelled at all** (PART4). A car from this pipeline is
   missing mandatory parts.

## 4a. Findings that change absolute numbers

1. **Cartridge mass is counted inconsistently.** Appendix vi puts a full Race Power Pack at 28.9–29.4 g.
   The objective adds 23 g as the empty shell (inside `car_weight_kg`) plus 7.9 g of propellant from the
   thrust sheet, 30.9 g at launch. The CAD split (15.25 g steel + 7.97 g CO2 = 23.2 g) suggests 23 g is the
   *full* cartridge, not the shell. Roughly 1.5–8 g of phantom mass, depending on which is right. It shifts
   every predicted time and slightly re-weights drag against mass. Weigh a full and an empty cartridge.
2. **The thrust curve is simulated, not measured.** In `co2_thrust_data.csv`, acceleration equals force /
   mass exactly, with no drag or friction, and the car reaches 20 m at 0.976 s. It is a model output
   (impulse 1.21 N·s, peak 8.5 N at 0.18 s). Replace it with a load-cell measurement before trusting
   absolute times.
3. **µ = 0.010 is a placeholder.** Worth 7 ms per doubling.

---

## 5. Roadmap

| phase | goal | parts | CFD needed | exit test |
|---|---|---|---|---|
| **−1. Correct** | inlet turbulence, block envelope, T5.5 wall, commit the missing driver and merge scripts | 1, 2, 3 | one re-solve of the leader | leader re-measured in correct flow; legality fixes pass |
| **0. Measure** | real thrust curve, µ, cartridge masses, ballast material, T7.13 clearance | 2, 4 | no | three numbers replace three placeholders |
| **1. Wheels** | lowest-inertia wheels that pass the hang test; bearing choice | 4 | no | printed wheels, 100 g hang test, lower I than v2 |
| **2. Ballast + lean body** | ballast model; parametric body; mass as a constraint | 1, 3 | no (L0) | legal 48.0 g car with a body ≈ half today's volume |
| **3. Full-car CFD** | wheels, supports, halo, wings, cartridge as patches; noise floor measured | 2, 4 | yes | per-patch drag; repeat-solve error bar < 1 % |
| **4. Wings + nose** | legal wings at minimum drag, front-wheel shielding | 4 | yes | scrutineer gates pass; drag vs no-wing measured |
| **5. Efficient descent** | forward+adjoint merged, morphing mesh, parametric dJ/dp with line search | 2, 3 | yes | 10–15 iterations reach a converged D20 |
| **6. Structural search** | discrete switches evaluated, recommendations to the designer | 5 | yes (L1) | ranked list with error bars; designer handoff |
| **7. Validate** | fine / unsteady on 2–3 finalists; build; track test | 2, 5 | yes (L3) | predicted vs measured race time |

Phases 0–2 need no CFD at all and address the two biggest levers.

---

## 6. DESIGN_features.md, reasoned

Short version: its direction is right (wheels first, wings from the regulation numbers, wheels in the CFD,
noise floor before fine tuning). Its numbers and three of its structural choices need correcting.

| claim | verdict |
|---|---|
| Wheel freedom ≈ 90 ms (1.628 → 1.538 s) | **Overstated.** Its 8.7 g effective mass implies ≈ 2.7 g wheels (v1). With v2 CAD wheels: ≈ 48 ms to zero inertia, ≈ 30 ms for k 0.8 → 0.3. Still the largest lever. |
| "Mass only matters through k" | **Wrong target.** The objective prices I = k·m·r². Adding hub mass lowers k and raises I. Minimise rim mass. |
| "Car sits at the 48 g floor and ballast refills" | **Right in principle, false in the code.** No ballast exists in the pipeline; foam does the refilling. This is the root cause of the blob (§1.1). |
| Voxel FE for the hang test first | **Reorder.** Ring and beam models plus five printed samples first; FE as a cross-check. |
| One generative engine; every attachment as a label in the body's φ | **Change.** Thin parts cannot live on a 0.5 mm grid. Exact parametric components, assembled at mesh level. |
| Organic wheel supports via FE + adjoint | **Reorder.** Parametric skeleton first. Organic refinement only if the adjoint shows the support's drag is above the noise floor. |
| Rear wing bounds | **Incomplete.** Missing T9.5.4 (≤ 15 mm height deviation) and the placement consequence of T9.7 front-view visibility. |
| Front wing and nose | **Extend.** Add T7.11 (wheels may be hidden up to 20 mm, so the wing can shield them), T7.9.1 (5 mm clear ahead of front wheels), T8.2 applying to the whole nose assembly, finish-beam trigger at ≈ 7 mm. |
| Wings run to the minimum-drag corner | **Likely for the rear wing, unlikely for the front** (shielding needs camber and incidence). Test both. |
| Rotating wheels in CFD, noise floor, morphing mesh | **Agree.** Use `adjointRotatingWallVelocity` and `volumetricBSplines` morphing. Open-spoke wheels need MRF. |
| Lift out of the gradient | **Agree.** 0.45 ms per 0.1 N. |
| Stage 0 wheels before everything | **Agree.** |
| Not mentioned | Ballast, body parametrisation, floor and hollowing, tether guides (T6), decal areas (T4.6, T4.7), cartridge mass inconsistency, simulated thrust curve. |

---

## 7. Questions only you can answer

1. Ballast material you will use: lead or tungsten alloy. It sets the smallest legal body.
2. T7.13 clearance figure from the diagram.
3. Bearings or plain bore; wheel and wing print process and material.
4. Support-to-body fastening.
5. Weight of one full and one empty cartridge on a 0.01 g scale.
6. Whether you can borrow a load cell for a thrust curve, and track time for a coast-down.
