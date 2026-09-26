# Part 5 — Design advisor: assembly search, legality engine, and structural recommendations

**Status:** plan only. **Date:** 2026-09-25.
**Purpose:** the "solver/optimiser that actually recommends full-on structural changes to the 3D designer".
**Depends on:** Part 1 (parametric body), Part 2 (multi-patch CFD, adjoint, calibrated objective),
Part 3 (efficient inner loop), Part 4 (component parameters and gates).

---

## 0. The problem this solves

Today the pipeline's output is a marching-cubes STL of a level set. A designer cannot open that in Fusion 360
and improve it, and nothing tells them *why* the shape is what it is, or what to change next.
The search also cannot make structural moves: it can only nudge a surface 0.15 mm per CFD iteration inside
one topology (PART3_UPGRADES.md). Adding or removing a sidepod, switching to a two-element wing, cutting floor
channels, or moving the axle into the body are all invisible to it.

Part 5 does three jobs:

1. **Searches the whole assembly**, including discrete structural choices, at the cheapest fidelity that can
   tell candidates apart.
2. **Explains**: turns adjoint fields and parameter gradients into ranked, quantified statements in
   milliseconds, with a confidence flag from the measured noise floor.
3. **Hands off**: produces CAD-friendly geometry (sections, guide curves, exact component parameters) and a
   legality report the designer and the scrutineering portfolio can both use.

```
part5-advisor/
  design_space.py        one registry: body params (Part 1), component params (Part 4),
                         outer scalars (W, x_front, d_halo), discrete switches; bounds from regs
  legality.py            every regulation as a measured margin on the assembled car
  fidelity.py            L0 analytic -> L1 coarse CFD -> L2 medium CFD+adjoint -> L3 validation
  attribution.py         dJ/dn -> per-parameter dJ/dp and per-region integrals, in ms
  recommender.py         continuous moves + discrete structural candidates, ranked
  search.py              multi-fidelity Bayesian search over (discrete x continuous)
  robustness.py          tolerance Monte Carlo, expected race SCORE, DNF risk
  handoff.py             report (MD/HTML with images), CAD export (sections, DXF, STEP), what-if API
  calibration_loop.py    track results back into the model
  tests/
```

---

## 1. Design space registry

One file lists every design variable with its type, bounds, units, owning part, and the regulation that
bounds it. Continuous: body loft stations, wing and nose parameters, wheel profile, W, x_front, d_halo.
Discrete switches, each a structural decision:

| switch | options |
|---|---|
| sidepods | none / short / full |
| floor | flat / raked / channels (1–3) |
| rear wing | 1 element / 2 elements; central pylon / twin pylons / swan neck |
| front wing | 1 / 2 / 3 elements; endplate style |
| wheel mounting | outrigger supports / axle through body pod |
| wheel web | solid / spokes / capped |
| cartridge shroud | full / minimal (T5.5 wall only) |
| ballast material | lead / tungsten alloy (changes the smallest legal body) |

Today these are either hard-coded in masks or absent. Making them explicit is what lets the search change
structure rather than polish one structure.

---

## 2. Legality engine

Every rule as a function of the assembled geometry returning a signed margin in mm (or g),
measured the way a scrutineer would. It serves as the constraint set for the search and as the compliance
table in the report (Specification judging is 110 of the competition's points, and any Performance-rule
breach makes the car ineligible for Fastest Car).

Coverage target, grouped:

| group | rules |
|---|---|
| assembled car | T3.4 width, T3.5 height, T3.6 mass, T3.7 clearance (cartridge in, car on four wheels) |
| body | T4.1 single piece rear of Ref A, T4.2 cargo, T4.4 halo (pocket z, visibility front/side/top), T4.6 logo decal area visible in each side view, T4.7 team number area in plan view, T4.8 |
| chamber | T5.1–T5.6 including radial visibility of the 5 mm protrusion |
| tether | T6.1 position, T6.2 bore, T6.3 closed |
| wheels | T7.1–T7.13 including T7.9 zones, T7.10, T7.11, T7.12.1 cylinder, T7.13 clearance |
| nose/front wing | T8.2, T8.5.1–T8.5.3, T8.6.1–T8.6.3, T8.7 ball-on-rod, T8.8 |
| rear wing | T9.4.1–T9.4.3, T9.5.1–T9.5.4, T9.6 ball-on-rod, T9.7 |

Part 1's hard masks stay as the fast way to keep the body legal during descent. `legality.py` is the
independent audit that runs on every candidate that reaches CFD and on every exported design.
Anything the masks and the audit disagree on is a bug in one of them.

T4.6 and T4.7 are new: a very slim body can run out of flat side area for the 30 × 15 mm logo decal,
or plan-view area ahead of the halo for 8 mm numerals. Neither is modelled anywhere today.

---

## 3. Fidelity ladder

| level | what | cost | used for |
|---|---|---|---|
| L0 | mass, COM, wheel inertia, frontal and wetted area, legality margins, analytic race time with a drag surrogate | milliseconds | filtering, Stage 1, discrete screening |
| L1 | coarse RANS, warm-started, fixed deterministic mesh settings | ≈ 5 min target | ranking structural candidates |
| L2 | medium RANS + adjoint, morphing mesh (PART2 §B) | ≈ 20–30 min | continuous refinement of the chosen structure |
| L3 | fine mesh, longer averaging or unsteady | hours | final validation of 2–3 finalists only |

A candidate climbs only when the level below cannot separate it from its competitors. The measured noise
floor at each level (PART2 §B) decides that, not a fixed rule.

---

## 4. Attribution — turning the adjoint into ms per mm

From one L2 solve:

1. **Per parameter:** `dT/dp_k = dT/dD20 · Σ_i s_i · (∂x_i/∂p_k · n_i) · A_i`, with ∂x/∂p_k from a
   finite difference of the parametric geometry (no CFD). Plus the analytic mass, COM and inertia parts.
   Result: a table "moving this parameter 1 mm is worth X ms".
2. **Per region:** integrate the drag sensitivity over named regions (nose, front-wheel faces, pod fronts,
   floor, halo trench, canister shroud, base, wings, supports). Result: where the drag comes from and where
   it could be removed.
3. **Confidence:** each number is compared with the drag noise floor at that fidelity. Anything below it is
   reported as "not measurable" instead of as a recommendation.

This is also the analysis the Design & Engineering portfolio (180 points) asks teams to show.

---

## 5. Recommender — structural changes, not just nudges

### 5.1 Continuous moves

Top-N parameter moves from §4, each with predicted Δt, its error bar, the binding regulation if any
("width is at the T3.4 limit"), and the knock-on effects (mass → ballast, COM).

### 5.2 Structural moves

Generated two ways:

- **Rule-driven from the sensitivity pattern.** Examples: large drag sensitivity on pod leading faces with
  little pressure recovery behind them → evaluate "no sidepods"; base-pressure-dominated drag → evaluate a
  longer boat-tail and a smaller cartridge shroud; high sensitivity on front-wheel faces → evaluate a wider
  front wing with endplates that shield them (T7.11).
- **Search-driven.** Every discrete switch in §1 is a dimension of the search in §6.

Each structural candidate is built by Part 1/Part 4 from parameters, gated by §2, screened at L0, ranked at L1.
The report shows the winners with their measured Δt and error bars, and says which are worth building.

### 5.3 Why this is safer than letting the level set invent topology

A structural change proposed from parameters is manufacturable and legal by construction, and a designer can
understand it. A level set drifting into a new topology over hundreds of CFD iterations is none of those
things, and the budget does not exist for it anyway.

---

## 6. Search over the assembly

- Continuous variables inside a fixed structure: gradient-based with line search, using §4's dJ/dp
  (Part 3 inner loop, upgraded).
- Discrete × continuous: one Gaussian-process surrogate per structural configuration, fed by L1 results,
  with expected improvement to pick the next L1 solve. Configurations whose best predicted time is worse than
  the incumbent by more than twice the noise floor are dropped.
- Outer scalars (W, x_front, d_halo): continuous variables of the same search, **not** a sweep of separate
  cars (PART3_UPGRADES.md). Their mass/COM effect is L0-cheap; their aero effect enters through L1.
- Budget example: 8 structural configurations × 6 L1 solves ≈ 48 coarse solves ≈ 4 h; then 10–15 L2 adjoint
  iterations on the winner; then L3 on 2–3 finalists.

---

## 7. Robustness and race-day score

- Time trials score the **average of the 2nd and 3rd best** of four runs (C9.6), and a DNF costs points.
  Optimise the expected score, not the single best time.
- Monte Carlo over manufacturing tolerance (milling ±0.1 mm, print shrink), wheel alignment, µ scatter,
  mass tolerance against the 48.0 g floor (T2.8 rounding).
- Structural margins (hang tests, wing rigidity, tether guide 200 g) turned into a DNF risk flag.

---

## 8. Designer handoff

1. **Report** (MD/HTML): current vs recommended car, images of pressure and sensitivity, the ms-per-mm table,
   structural candidates tried and their results, mass budget (body, hardware, ballast), legality margins,
   and what is not measurable at the current noise floor.
2. **CAD export:** body as station sections plus guide curves (CSV and DXF, and a Fusion 360 script to loft
   them), so the designer rebuilds a clean surface instead of importing a mesh. Wings, nose, wheels and
   supports as exact STEP from their parameters (cadquery/OCC). STL only for CFD.
3. **What-if API:** the designer changes a parameter; L0 answers at once and L1 in minutes.

---

## 9. Closing the loop with the track

The model is only as good as three measured inputs, none of which is measured today:

| input | today | how to measure |
|---|---|---|
| thrust curve | `co2_thrust_data.csv` is a drag-free simulated trajectory (acceleration = force / mass exactly) | load-cell test of real cartridges |
| µ | 0.010 placeholder | coast-down on the track, or incline test |
| COM penalty | nine placeholder points | ballast experiment already planned in the code |

Then compare predicted and measured race times for two or three built cars and fit a correction.
The orchestrator already refuses to run without the two validation flags; this module is where they get
earned honestly.

---

## 10. Build order

| # | item | test |
|---|---|---|
| 1 | `design_space.py` | every parameter has bounds that trace to a regulation or a stated design choice |
| 2 | `legality.py` on today's leader | reproduces the known status; reports T4.6/T4.7/T6/T8/T9 as missing |
| 3 | L0 evaluator | matches Part 2's objective to 1e-6 s on a fixed input |
| 4 | attribution on one saved adjoint field (records already store them) | per-region integrals sum to the total |
| 5 | handoff report + section export | designer can loft the exported sections in Fusion 360 |
| 6 | L1 fidelity + noise-floor measurement | repeated solves of one geometry give the error bar |
| 7 | discrete screening at L0/L1 | ranks at least 6 structural variants with error bars |
| 8 | multi-fidelity search | beats the hand-picked baseline by more than the noise floor |
| 9 | robustness | expected score and DNF flags for finalists |
