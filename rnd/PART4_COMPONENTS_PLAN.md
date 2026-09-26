# Part 4 — Component generators: wheels, hubcaps, wheel supports, wings, nose cone, tether guides

**Status:** plan only, nothing built. **Date:** 2026-09-25.
**Replaces:** DESIGN_features.md §1–§2 (kept where right, corrected where not — see §9).
**Depends on:** Part 2 multi-patch CFD (PART2_UPGRADES.md §C) for anything aerodynamic.
**Feeds:** Part 1 (interfaces the body must respect), Part 5 (parameters and gates the search drives).

---

## 0. Why this is its own part

Today every component except the body is either fixed CAD (`hardware_cad/*.stl`), a mass placeholder
(rear wing = 5 g at a guessed point), or absent (front wing, tether guides). The CFD sees the body only.
So the car in the leader image is a body with bolted-on parts that nobody optimised and the solver never saw.

Components are thin (wings 2–6 mm, rims < 1 mm, struts 1–3 mm). A level set on a 0.5 mm grid
represents a 2 mm wing with 4 cells and marching cubes turns it into a lumpy slab. So Part 4 builds
components as **exact parametric geometry** (analytic SDF for gates, direct triangle mesh / B-rep for
CFD and CAD), not as labels in the body's φ field. This is the main disagreement with DESIGN_features §2.1.

```
part4-components/
  component_contract.py   parameter registry, bounds from regs, units, one place
  wheel.py                2-D axisymmetric profile -> revolve; m, I closed form; hang-test check
  hubcap.py               (inside wheel.py profile; separate only for the A/B switch)
  wheel_support.py        parametric skeleton (pad, strut spline, boss) + beam/FE check
  wing.py                 front + rear wing SDF and mesh, 1..3 elements, endplates
  wing_support.py         rear pylon(s); front wing mounts to nose
  nose.py                 parametric nose cone (printed), front-wing mount
  tether_guide.py         T6 eyelets, streamlined
  gates.py                scrutineer-style measurements on the MESH (span, chord, thickness,
                          clear-air ball-on-rod, front-view visibility, T7.9/T7.11 zones)
  fe_beam.py              beam/ring models for stiffness (first); voxel FE only if needed
  export.py               STL for CFD (named solids = separate patches), STEP/DXF for CAD
  tests/
```

---

## 1. What each component is worth, in race time

Measured with the project's own locked objective (`race_objective.py`) at the leader's operating point
(D20 = 0.287 N, 48.2 g competition mass + 23 g cartridge, µ = 0.010, CO2 sheet in `co2_thrust_data.csv`):

| lever | change | race time |
|---|---|---|
| wheel inertia, 4 wheels | today's v2 CAD wheels (rim-heavy, I ≈ 144 g·mm² each) → zero inertia | **−48 ms** |
| wheel inertia | rim mass halved | ≈ −23 ms |
| drag D20 | −10 % | −12.8 ms |
| mass below the 48 g floor | per gram | +16.7 ms |
| rolling friction µ | 0.010 → 0.020 | +7.1 ms |
| lift | 0.1 N downforce | +0.45 ms |
| finish-beam trigger point | 10 mm further forward at 20 m/s | ≈ −0.5 ms |

Reading of the table:

- **Wheels are the single biggest lever and need no CFD.** Build them first (Stage 0), exactly as
  DESIGN_features §4 proposes.
- Wing lift is worthless; wing **drag** is the only wing term that matters. Wings are mandatory
  (T8.3, T9.2) and their dimensions are Performance regulations (T8.6, T9.5), so they are a legality
  requirement first and a drag minimisation second.
- µ is worth protecting: bearing choice and alignment are worth more than most body tweaks.

---

## 2. Wheels and hubcaps

### 2.1 What to minimise — I, not k

The objective prices each wheel as effective mass `I / r²`. For a wheel whose contact band must be a
full-width cylinder at radius R (T7.4, T7.7), `I / r² ≈ m_rim + (small hub and web terms)`.
So **the target is the lightest rim that still passes the hang test**, then the lightest web and hub.

DESIGN_features §1.1 frames it as lowering the shape factor `k = I/(m r²)` by concentrating mass at the hub.
Adding hub mass lowers k and **raises** I, which makes the car slower. At the 48 g floor ballast refills any
mass the wheel gives up, so wheel *mass* is free and wheel *inertia* is not. Optimise I directly.

Rim minimum is a stiffness problem. For a thin ring under a point load, deflection scales with
`R³ / (E · w · t³)`, so rim effective mass at fixed stiffness scales with the material index `ρ / E^(1/3)`
(Ashby's panel index). Two consequences for the search:

1. Material is a design variable: printed ABS ≈ 0.80, PLA ≈ 0.82, carbon-filled nylon ≈ 0.66,
   carbon-fibre tube section ≈ 0.38 (lower is better).
2. Geometry beats thickness: an inboard stiffening lip or I-section rim buys section inertia without
   putting mass at full radius on the whole band. The optimiser should be allowed to find it.

### 2.2 Representation

A 2-D profile in (r, y) revolved about the axle, plus an angular pattern:

| parameter | bound | source |
|---|---|---|
| R outer radius | 14.0–16.0 mm | T7.5 (Ø 28–32) |
| contact width w | ≥ 13 front, ≥ 17 rear, flat across the band | T7.4, T7.7 |
| rim shell t_rim | ≥ process minimum (0.4 SLA, 0.8 FDM) | process |
| inner lip height h_lip, thickness t_lip | free | stiffness |
| web: solid disc t_web, or N spokes × width × thickness | free | inertia vs drag |
| hub bore Ø, boss length | from bearing/axle choice | T7.8 |
| hubcap: none / flat / dished, t_cap, dish depth | free | aero A/B |
| outboard chamfer | excluded from contact width | T7.4 |

`m` and `I` come from Pappus integrals over the profile (exact, instant). Spokes multiply the web term
by their fill fraction.

### 2.3 Stiffness: the hang test

T7.13: 100 g hung from each wheel, no change of diameter or shape.

- First pass: analytic thin-ring model (closed form) for rim ovalisation, plus a spoke/web bending check.
- Calibrate on the current v2 CAD wheel: its computed deflection is the ceiling, because it passes today.
- Then **print five variants and hang 100 g on them.** At this scale a physical test costs an afternoon
  and is more trustworthy than a voxel FE with 2–3 voxels through a 0.5 mm wall. This replaces
  DESIGN_features §1.4's voxel-FE-first plan.

### 2.4 Aero: hubcap and spoke A/B (DESIGN_features W2, kept)

Only meaningful once wheels are in the CFD (Part 2 §C). Cost side is known now:
a 0.4 mm ABS full-face cap adds ≈ 0.26 g at k ≈ 0.5, about 0.13 g of effective mass per wheel,
**≈ 2 ms per wheel**. A cap must save more than ≈ 2 ms of drag per wheel to earn its place.

Rotating-wall BCs are correct for a closed (capped or solid-web) wheel. An open-spoke wheel needs an
MRF zone or sliding mesh to be honest; flag open-spoke drag numbers as approximate.

### 2.5 Bearings and axle

µ = 0.010 is a placeholder. Doubling it costs 7 ms, more than most body changes. Deliverables:
bearing vs plain-bore choice, axle diameter, a coast-down or incline test that fits µ
(feeds `calibration.py`), and the bore geometry the support boss must match.

---

## 3. Wheel support systems — parametric skeleton, organic refinement second

### 3.1 Rules that shape it

- T7.12.1: support exists only inside the cylinder through the diameters of the two opposing wheels.
- T3.7: ≥ 1.5 mm above the track.
- T7.10: supports are the one component allowed to hide the wheel in side view.
- T7.13: clearance from the wheel's inner corner to the body (figure is in the diagram, not the text —
  **open question, needs the number**).
- T3.9: two spares each, identical geometry.

### 3.2 Representation

| element | parameters |
|---|---|
| body pad (anchor) | x/z position on the body wall, pad size, fastening (adhesive / dowel / screw) |
| strut(s) | count 1–2, centreline spline (3–4 control points), section ellipse or thin aerofoil, t/c |
| axle boss | bore from §2.5, wall ≥ 1.5 mm, length |
| optional web | fill between struts, thickness |

Built as a sweep along the spline, so it is exact and printable. The body gets a matching
forced-solid **landing pad** in its own field (Part 1), so the milled body cannot remove the face the
support bonds to.

### 3.3 Structure

Load cases from DESIGN_features §2.2 (kept): L1 hang test 0.98 N at the boss, L2 side load ≈ 0.5 N,
L3 launch. Check with a **beam model** of the strut (closed form, instant). Ceiling = the v2 CAD support's
deflection under the same model. Voxel FE only for the final design, as a cross-check.

### 3.4 Where "organic" comes in

The support sits inside the wheel's diameter cylinder, largely in the wheel's own wake. Its drag
sensitivity is small, so a full topology optimisation is not the first move. The order is:

1. Parametric skeleton, optimised for mass (free at the floor) and stiffness, with an aerofoil section.
2. Once wheels and supports are in the CFD, read the adjoint sensitivity on the support patch.
   If the support's integrated drag sensitivity is above the CFD noise floor, run a **bounded level-set
   refinement**: φ = φ_skeleton + δφ with |δφ| ≤ 1 mm, FE constraint active, connectivity gate
   (pad must reach boss). If it is below the noise floor, stop — organic refinement would be tuning noise.

### 3.5 A structural alternative to put in front of the designer

Axle-through-body (bearings pressed into a milled body pod) instead of outrigger supports.
It removes two parts per axle but forces body material inside the wheel cylinder. It is a discrete
choice for Part 5 to evaluate, not something Part 4 decides.

---

## 4. Wings (front and rear)

### 4.1 Rear wing — T9

| rule | constraint |
|---|---|
| T9.4.1 | wing **and support** wholly aft of Ref Plane B |
| T9.4.2 | extreme rear ≤ Ref B + 40 mm |
| T9.4.3 | highest point ≤ 65 mm |
| T9.5.1 | single unbroken span ≥ 50 mm, measured as the shortest of LE, TE, top, bottom |
| T9.5.2 | chord 15–25 mm throughout the span, ≤ 2 elements, elements must overlap (Appendix v) |
| T9.5.3 | thickness 2–6 mm, existing across the span at some chord station |
| T9.5.4 | top-surface height deviation along the minimum span ≤ 15 mm (**missing from DESIGN_features**) |
| T9.6 | 5 mm clear air to every other part **and the track**, ball-on-rod test |
| T9.7 | wing unobstructed in front view (**hard placement constraint** — see below) |
| T9.3 | span unchanged during races (rigid) |
| T5.6 | support must not hide the protruding ≥ 5 mm of cartridge in any radial view |

T9.7 matters more than it looks. The whole wing must be visible from the front, so it cannot hide behind
the body, halo, canister or rear wheels in front view. With the canister top near 47 mm, a centred wing
must sit above the body silhouette, or reach outboard of it. The optimiser has to see that as a hard
constraint, computed as a 2-D front-view occlusion raster.

### 4.2 Front wing — T8

| rule | constraint |
|---|---|
| T8.2 | whole nose assembly, wing included, within 40 mm forward of Ref A |
| T8.5.2 | forward of Ref A; ≤ 20 mm above track wherever |y| > 15 mm |
| T8.5.3 | endplates outside the minimum span, ≤ 10 mm wide, ≤ 25 mm high beyond the front wheels |
| T8.6.1 | span ≥ 50 single, or 2 × 25 split by the nose, shortest-edge measurement |
| T8.6.2 | chord 15–25, **up to 3 elements**, overlapping |
| T8.6.3 | thickness 2–6 mm |
| T8.7 | 5 mm clear air to every other part and the track |
| T7.9.1 | nothing within 5 mm in front of the front wheels, full height, from inside contact edge outward |
| T7.11 | front wheels may be hidden in front view only up to 20 mm above the track |

The front wing's real job is **front-wheel shielding**. T7.11 allows the wheels to be covered up to 20 mm,
and T8.5.2 allows the wing up to 20 mm. A wing and endplate that sit ahead of the wheels, 5 mm clear of
them, can take the wheels partly out of the free stream. That is where camber and angle of attack
earn their keep: turning flow around the wheels, not making downforce.

A second, small, free gain: the finish beam sits about 7 mm above the track (Appendix i). The first part
of the car to cross it at that height stops the clock. A wing or endplate leading edge low and far
forward gains about 0.5 ms per 10 mm.

### 4.3 Representation

- Section: NACA 4-digit (camber m, position p, thickness) or thin cambered plate with round LE.
- Per element: chord share, angle of attack, gap and overlap to the previous element.
- Planform: span, sweep, dihedral / height variation (bounded by T9.5.4 at the rear), taper.
- Endplates: height, width (≤ 10 front), chord.
- Placement: x, z of the leading edge.
- Built as an exact SDF (for gates and clear-air distance) and a direct triangle mesh (for CFD and CAD).

### 4.4 Gates, measured like a scrutineer

Implemented on the extracted mesh, not on parameters, so a parameter bug cannot pass a non-compliant wing:
span as the shortest of LE/TE/top/bottom lines; chord at every span station; thickness existing across the
span; overlap between elements (Appendix v projection check); clear air as the distance field of all other
parts plus the track, with the 2.5 mm fillet allowance at the support junction; front-view visibility raster.

### 4.5 Rigidity

Printed wing, 50+ mm span, 2 mm thick: T9.3 / T8.4 may bind before drag does. Beam check under aero load
at 20 m/s plus a handling load, calibrated on one printed sample. If it binds, thickness or material moves,
not the rule.

### 4.6 Expected optimum (to test, not assume)

DESIGN_features §2.3 predicts every wing runs to its minimum-drag corner: chord 15, thickness 2, zero camber.
That is likely right for the rear wing. It is probably wrong for the front wing, whose shielding role makes
camber and incidence useful. Treat both as hypotheses the search checks.

### 4.7 Gradients

Same as DESIGN_features §2.3 (kept): finite-difference each wing parameter on the SDF, convert to normal
displacement at the surface vertices, dot with the adjoint's per-vertex sensitivity on the wing patch.
No extra CFD. Acceptance test: one angle-of-attack finite difference with two real solves.

---

## 5. Nose cone

T1.7: the nose is the front wing support. T8.2: ≤ 40 mm overhang from Ref A. T8.5.1: ≤ 25 mm tall, ±15 mm wide.
Printed (geometry_contract), so it has no milling constraint and can carry mass where the designer wants it.

| parameter | notes |
|---|---|
| length forward of Ref A | ≤ 40 mm; also sets x_front via Part 1 |
| tip height, tip radius | tip low for the finish beam (§4.2) |
| cross-section superellipse w(x), h(x), exponent | ≤ 15 half-width, ≤ 25 height |
| underside rake | ground effect vs clearance (T3.7) |
| wing mount | single pylon, twin pylons, or wing through nose |
| infill / density | free mass placement: forward mass moves COM forward |
| body blend at Ref A | the legal step to the ≤ 65 mm body; Part 1 shapes the body side of it |

Today's nose is a label in the body's φ field with a 1000 kg/m³ density. Moving it to a parametric
printed part gives it a real shape and makes its density a declared choice rather than a constant.

---

## 6. Tether line guides — T6, not modelled anywhere today

Two guides, one within 10 mm of each axle line, closed, internal 3.5–6 mm, robust to a 200 g hang test (T6.3).
They sit under the car on the centreline, add drag, and the tether contact is a friction source.
Parametric eyelet: internal Ø, wall, streamlined outer section, attachment to the floor. The body floor
must provide the mount (Part 1 landing pad). A search over the code found no reference to tether guides
in any of the three parts.

---

## 7. Interfaces this part publishes

| to | what |
|---|---|
| Part 1 | wheel R (axle height), wheel inner-face y, support pad footprints, rear-wing pylon root, tether guide mounts, nose rear face at Ref A |
| Part 2 | one named STL solid per component (patches: body, nose, wing_f, wing_r, supports, wheel_fl/fr/rl/rr, halo, guides) plus wheel axis and ω for rotating walls |
| Part 3 / 5 | parameter vector, bounds, mass/COM/inertia per component, gate margins, dJ/dp projector |

---

## 8. Build order and the test that closes each step

| # | item | test |
|---|---|---|
| 1 | `component_contract` + `gates.py` on hand-made meshes | known-good and known-bad wings from Appendix iii/iv pass and fail as the appendix says |
| 2 | `wheel.py` profile, m, I | reproduces the v2 CAD wheel's volume within 2 %, and its I |
| 3 | ring + beam stiffness | v2 CAD wheel and support give the ceilings; hand check |
| 4 | wheel search (no CFD) + 5 printed variants hung at 100 g | lower I at equal or better deflection, confirmed physically |
| 5 | `wing.py` + gates | scrutineer measurements pass at the bounds, fail just past them |
| 6 | `nose.py`, `wing_support.py`, `tether_guide.py` | legality gates, printability |
| 7 | component STLs through Part 2 multi-patch CFD | per-patch drag, sensitivities finite on every patch |
| 8 | wheel hubcap/spoke A/B | ΔD per variant with its error bar, priced against its Δ effective mass |
| 9 | dJ/dp projection for wings | finite-difference check on one angle of attack |
| 10 | support organic refinement (only if §3.4 says it is above noise) | connected, stiffer or equal, less drag |

---

## 9. DESIGN_features.md, checked

| § | verdict | note |
|---|---|---|
| 1.1 table | numbers off | Its 8.7 g effective mass implies ≈ 2.7 g wheels (v1). With the v2 CAD wheels the freedom is ≈ 48 ms to zero inertia, ≈ 30 ms for k 0.8 → 0.3. Still the largest lever. |
| 1.1 "mass only matters through k" | wrong framing | The objective prices I. Lowering k by adding hub mass raises I. Minimise rim mass. |
| 1.2 rule table | mostly right | Add T7.6 (full-width contact on all four wheels), T7.10, T7.11. |
| 1.4 voxel FE first | changed | Ring/beam model plus five printed samples first; voxel FE as a later cross-check. |
| 1.5 rotating wheels, azimuthal averaging | kept | Rotating wall is honest only for closed wheels; open spokes need MRF. |
| 1.6 W1/W2/W3 staging | kept | W3 only if W2 shows the wheel's own drag is above the noise floor. |
| 2.1 one engine, everything in the body's φ | changed | Thin parts cannot live on a 0.5 mm grid. Components are exact geometry, assembled at the mesh level. |
| 2.2 wheel support | kept, reordered | Parametric skeleton first; organic refinement only if the adjoint shows it matters. |
| 2.3 rear wing | kept, extended | Add T9.5.4 height deviation and the T9.7 front-view placement consequence. |
| 2.4 front wing and nose | kept, extended | Add wheel shielding under T7.11, T7.9.1's 5 mm zone, T8.2 applying to the whole nose assembly, finish-beam trigger. |
| 2.5 manufacturing split | kept | |
| 3 aero needs | kept | See PART2_UPGRADES.md. |
| 4 stage layout | kept | Wheels first. |
| — | **missing** | Tether guides (T6), decal areas (T4.6 logo 30 × 15 mm visible in each side view, T4.7 team number ≥ 8 mm in plan view ahead of the halo), ballast model, body parametrisation. Covered in Parts 1 and 5. |

---

## 10. Open questions that change the work

1. T7.13 minimum clearance figure (from the diagram). Sets the support region's outboard limit.
2. Bearings or plain bore? Sets bore, boss, µ.
3. Wheel and wing process and material (FDM, SLA resin, nylon, carbon). Sets density, minimum wall, and the material index.
4. Support-to-body fastening. Defines the landing pad.
5. Ballast material you will actually use (lead ≈ 15.8 g fits the container, tungsten alloy ≈ 25 g). Changes how small the body can be (Part 1).
