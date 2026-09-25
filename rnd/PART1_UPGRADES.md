# Part 1 — Geometry upgrades: parametric body, organic skin, ballast, floor, hollowing

**Status:** plan. **Date:** 2026-09-25. **Repo audited:** `part1-simulation` @ 3db4607 (2026-08-12).
Line numbers refer to that commit.

---

## 1. What runs today

```
build_unified_geometry(W, x_front, d_halo)          unified_phi.py:399
  labels by zone (nose / sidepod / rearpod / main_body)    _zone_masks :268
  hard AIR : T7.9 zones, wheel keep-clear columns (+4 mm margin), halo pocket, ballast slot,
             cartridge bore, wheel discs, halo visibility shadows, halo->canister loft CEILING
  hard SOLID: T4.2 cargo, T5.5 safety tube + end cap, loft DECK
  init "full": every legal cell solid (constant -dx, not a distance field)      :786
Stage 1 (2 mm grid): 100 HJ steps on a mass/COM proxy, redistance + machinability every 10
remap_geometry -> 0.5 mm grid                                                    :694
Stage 2 per iteration: apply_adjoint_to_unified (phi_updater.py:625)
  reinit (50 steps) -> splat adjoint -> extend -> + dT/dmass*rho + COM terms -> clip p99.9
  -> one CFL-limited HJ step -> symmetry -> enforce_machinability
extract_half_surface: y<0 forced air, marching cubes, repair                     :871
decimate to 120 k triangles (falls back to the full mesh when the angle gate fails)
ASCII STL of the BODY ONLY -> Part 2
```

The halo, helmet, wheels, supports, canister protrusion and wings are voids or masses. None of them are in
the surface the CFD sees.

---

## 2. Diagnosis — why the body is a blob

| # | sev | cause | evidence | consequence |
|---|---|---|---|---|
| 1 | CRIT | **No ballast.** Underweight is fixed by growing foam. | T3.6 barrier replaces `dT_dmass` (`inner_loop.py:227-271`); Stage 1 targets floor + 5 g with foam (`bayesian_outer_search.py:344, 411-424`) | ≈ 200 cm³ of 0.163 g/cm³ foam doing the job of a ≈ 1.4 cm³ ballast capsule |
| 2 | CRIT | **Mass drives the surface.** `dT/dmass·ρ` is a volume velocity everywhere on the skin | `phi_updater.py:841-872`; logged share aero 5–7 %, mass/COM 93–95 % | the outer shape answers to the mass barrier, not to drag |
| 3 | CRIT | **The start is the envelope.** Full-block init; the descent then moves ≤ 0.15 mm per CFD iteration | `unified_phi.py:780-786`; `phi_updater.py:375` | the final car is the rule envelope minus the masks |
| 4 | HIGH | **Design choices encoded as hard masks.** 4 mm wheel keep-out beyond the 2 mm clearance; loft ceiling and forced-solid loft deck; T5.5 contact shell; fixed 35 mm chamber height | `unified_phi.py:86, 541-555, 623-641`; `fixed_hardware.py:608, 901-948` | humps and trenches in the image are these masks; several exist only to stop the mass gradient carving legal features away |
| 5 | HIGH | **Nose is a label in the body field** at 1000 kg/m³ with no shape model | `geometry_contract.py:97`; `unified_phi.py:305-309` | a 25 × 30 mm stub; no front-wing mount; its density is a constant, not a choice |
| 6 | HIGH | **CFD surface is the body only** | `hardware_geometry.py:37-39` "do not make any wings"; `extract_half_surface` | the adjoint shapes the body for flow with no wheels in it |
| 7 | HIGH | **Regulations not modelled at all:** T6 tether guides, T4.6 logo decal area in side view, T4.7 team number area in plan view, T7.11, T8 front wing, T9 rear wing | grep over all three repos finds no reference | a car that passes every gate here can still fail scrutineering or be missing mandatory parts |
| 8 | MED | **Chamber height and depth fixed** at 35 mm and 50 mm; T5.2 allows 30–40, T5.3 45–58 | `fixed_hardware.py:607-608` | lowering the thrust line 5 mm lowers the whole rear shroud and the frontal area with it |
| 9 | MED | **Marching cubes on a 0.5 mm staircase + decimation** gives slivers; the decimator often ships the full 240 k-triangle mesh | `pipeline_interface.py:602-620, 843-855` | 21 % of vertices get no adjoint sensitivity (Part 2) |
| 10 | MED | **Initial field is not a distance function** (constant −dx), so the first step of every candidate does nothing until reinit | `unified_phi.py:786`; `phi_updater.py:787-829` | the fix is in place (reinit first) but costs 50 Godunov passes on 8 M cells every iteration |

*Findings from the Part 1 audit agent are merged in §9.*

---

## 3. Ballast model — the change that unlocks everything else

### 3.1 Rule basis

- T1.22 and Appendix ix: legal ballast only in the container under the halo, reached through the helmet aperture.
- Container: capsule 12.7 × 20 × 6.35 mm, ≈ 1.39 cm³ (`halo_pocket.py:45-48`).
- T3.6: 48.0 g minimum, excluding the cartridge. Under it costs 10 points per gram, and the car is raced
  with 0.2 g added per 0.1 g short. Being under is never optimal.
- T2.8 / T2.7.1: 48.0 passes, 47.9 fails. The scale reads to 0.1 g.

### 3.2 Model

```
m_hw     = wheels + supports + halo/helmet + wings + guides + nose        (Part 4, per design)
m_body   = ρ_foam · V_body(φ)
M_target = 48.0 g + margin (0.2 g: one scale digit plus machining scatter)
b        = clamp(M_target − m_body − m_hw, 0, B_cap)                       ballast mass
B_cap    = ρ_ballast · 1.39 cm³      (lead ≈ 15.8 g, tungsten alloy ≈ 25 g)
m_comp   = m_body + m_hw + b
COM      includes b at the container centroid (x from d_halo, z ≈ 20.8 mm)
```

Derivative seen by the shape update:

| regime | condition | dT/dV_body |
|---|---|---|
| ballast absorbing | 0 < b < B_cap | **0** — body volume is free; the skin follows drag only |
| over-heavy | b = 0 and m_comp > M_target | ρ·dT/dm > 0 — remove material (outer skin or floor channels, §4) |
| container full | b = B_cap and m_comp < M_target | barrier — body must add mass (or a denser printed nose, §5.3) |

The COM term stays live in every regime, and it now includes the ballast. Ballast sits low (≈ 21 mm) and at
the halo, so d_halo becomes a weight-distribution lever as well as an aero one.

### 3.3 What it replaces

`t36_descent_gradient` (Part 3), `PROXY_MASS_TARGET_MARGIN_KG` and `_proxy_objective_gradients`'s barrier
(Stage 1), the aero-phase switch and its hysteresis margins (Part 3). One function, used by both stages.

### 3.4 Expected effect

With lead ballast the body needs ≈ 16.7 g instead of ≈ 32.7 g, about half today's foam. With tungsten it needs
less than the regulations' own minimum structure (cargo, chamber wall, halo pocket floor), so the body shrinks
to what the rules and the aero want. Check which: that is the first question in §10.

---

## 4. Floor and hollowing

### 4.1 What "hollow inside" can legally mean

- T1.3 / T4.1: the body is one uninterrupted piece of Model Block rear of Ref A.
- T3.1.2: made by CNC material removal. Tools reach from ±Z and ±Y only (`geometry_contract.py:262-273`).
- A cover plate would be an extra component (T3.3). A decal must be 100 % adhered to a surface (T1.11),
  so it cannot bridge a pocket.

So a sealed internal void is not buildable. **Every hollow is an open pocket**, and a pocket open to the
floor is part of the underbody the air sees. Hollowing is therefore a floor-design problem, and it belongs in
the CFD. `enforce_machinability` already fills any void with no straight ±Y/±Z run to the outside; it stays.

### 4.2 When hollowing is needed

Only in the "over-heavy" regime of §3.2: the aero-best outer skin encloses more foam than the mass budget
allows. With lead ballast that is likely only for tall, wide bodies. With a lean body and tungsten it will
not happen. The mechanism is still needed, because the search must be free to choose a big outer shape
if the drag says so.

### 4.3 Floor parameters

| parameter | bound |
|---|---|
| ride height along x, h(x) | ≥ 1.5 mm with the cartridge in and the car on four wheels (T3.7) |
| rake, leading-edge radius, rear kick-up (boat-tail on the underside) | ≥ 3.15 mm radii on milled edges |
| channels: count 0–3, width ≥ 6.3 mm (one tool diameter), depth, x-start, x-end, entry and exit ramp angles | cut from −Z only |
| wall under channels | ≥ 2 mm (design choice, check with the 200 g tether test and handling) |
| keep-solid | T4.2 cargo (fully encompassed by body), T5.5 3 mm around the chamber, halo pocket floor at 24 mm, ballast container walls, support landing pads, tether guide mounts |

At 20 m/s downforce is worth 0.45 ms per 0.1 N, so the floor is about drag and mass, not ground effect.
Whether a low floor helps or hurts is for the CFD to say, once the underbody gap is resolved (Part 2).

---

## 5. Body representation — parametric skeleton, organic skin

### 5.1 Parametric body

A loft along the car's axis from Ref A to the rear face, with 8–12 stations. Each station has:

| per station | meaning |
|---|---|
| half-width w | body side |
| top height z_top, floor height z_bot | vertical extent |
| superellipse exponents n_top, n_bot | from round (2) to boxy (4–6) |
| optional pod offset and size | sidepods as a switch (Part 5) |

Plus globals: chamber height (T5.2, 30–40 mm) and depth (T5.3, 45–58 mm), shroud length, boat-tail length and
angle, halo region height, nose blend at Ref A. About 40–60 numbers.

Build it as an **SDF**: smooth union of the loft, the regulated solids (cargo, T5.5 tube, landing pads), and
smooth subtraction of the regulated voids (chamber, halo pocket, ballast slot, wheel clearance, T7.9 zones,
halo visibility shadows). Use a blend radius ≥ 3.15 mm: every concave edge then meets the milling radius by
construction instead of being smoothed after the fact.

Start from a sensible dragster, not the envelope: low nose blend at ≤ 25 mm (T8.5.1 makes the step at Ref A
legal but the body side can meet it), body top ahead of the halo held below the halo's base + 4 mm fillet
(T4.4.2 front-view visibility already forces this within the halo's width), a smooth rise to the chamber,
a boat-tail around the protruding cartridge.

### 5.2 Organic skin

φ = SDF_param(p) + δφ, with |δφ| ≤ δ_max (1–2 mm) as a trust region. The level-set machinery that exists
(HJ update, reinit, extension, machinability, symmetry) evolves only δφ, and only in the last few iterations
(PART3 §3.3). Large moves come from p, through the projected gradient dT/dp.

### 5.3 Nose as a Part 4 component

The nose leaves the body field. Part 4 builds it as an exact printed part. Its density becomes a declared
design choice (printed infill), which is a legitimate way to place mass forward and low. It is not ballast
under T1.22, so keep it to the material the part is made of; do not load a nose with lead.

### 5.4 Which masks stay

| mask | basis | keep? |
|---|---|---|
| T7.9 wheel zones | regulation | keep |
| wheel disc clearance 2 mm | design choice until the T7.13 figure is known | keep, make it a parameter |
| wheel keep-out +4 mm, full height | design choice ("outrigger look") | **drop**; T7.9/T7.10 and the disc clearance are the rules |
| halo pocket, ballast slot | regulation | keep |
| halo visibility shadows | T4.4.2 / T4.4.3 | keep |
| halo→canister loft ceiling | design choice | **drop**; let drag decide the deck |
| loft forced-solid deck | counter to the mass gradient | **drop** once §3 lands |
| T5.5 tube and end cap | regulation (3 mm around the minimum depth) | keep, length from the chamber-depth parameter |
| T5.5 contact shell | counter to the mass gradient | **drop** once §3 lands |
| T4.2 cargo | regulation | keep |
| new: T4.6 logo patch, T4.7 number patch | regulation | add as visibility checks on the side and plan views |
| new: support pads, tether guide mounts, rear-wing pylon root | Part 4 interfaces | add as forced-solid pads |

---

## 6. Numerics and surface quality

1. **Extract from the SDF, not from a staircase.** The parametric SDF is smooth; marching cubes on it gives
   well-shaped triangles and the decimator stops failing. That shrinks the 21 % unmapped-vertex problem.
2. **Narrow band.** Evolve δφ only where |φ| < 3–4 mm. Reinit and extension then touch ≈ 10 % of the 8 M cells.
3. **Coarse-to-fine.** 2 mm for layout and structure, 1 mm for descent, 0.5 mm for the final polish.
   `remap_geometry` already exists and is tested.
4. **Sensitivity smoothing.** Filter the splatted adjoint on the surface before extension (PART3 finding 1).
5. **STL size.** Binary for the full car; the decimated half as the CFD input.

---

## 7. Interfaces

| with | Part 1 provides | Part 1 needs |
|---|---|---|
| Part 2 | body STL as its own named solid, landing-pad and channel geometry, frontal area | per-vertex (or per-parameter) sensitivity on the body patch |
| Part 4 | pad and mount locations, Ref A face shape, body silhouette for the T9.7 front-view test | wheel R, inner-face y, support pad footprints, nose rear face, pylon root, guide mounts |
| Part 5 | parameter vector with bounds, dx/dp for projection, mass/COM/ballast breakdown, mask margins | parameter updates, structural switches |

---

## 8. Build order and the test that closes each step

| # | change | test |
|---|---|---|
| 0 | legality fixes A1 (block envelope) and A2 (T5.5 wall) | envelope fits 223 × 65 × 50; 3 mm ball test passes all round the bore |
| 1 | ballast model (§3) as one shared function | Stage 1 lands at 48.0–48.2 g with ballast > 0 and a body lighter than today's |
| 2 | drop the design-choice masks (§5.4) behind flags | same inputs build; legality gates still pass |
| 3 | chamber height and depth as parameters | legal range builds; bore and T5.5 wall follow |
| 4 | parametric body SDF | reproduces a hand-drawn reference body; every concave radius ≥ 3.15 mm |
| 5 | dx/dp finite differences | a parameter step moves the surface by the expected mm |
| 6 | floor and channel parameters, mass surplus path | over-heavy body sheds mass through channels, cargo and T5.5 intact |
| 7 | organic δφ with trust region and narrow band | δφ never exceeds δ_max; runtime per step drops ≥ 5× |
| 8 | T4.6/T4.7 checks, Part 4 mounts | a deliberately slim body fails T4.6 and the check says why |

---

## 9. Audit findings not covered above

From a line-by-line read of the rest of Part 1. Items already in §2 are not repeated.

| # | sev | finding | evidence | fix |
|---|---|---|---|---|
| A1 | CRIT | **The milled envelope is bigger than the Model Block.** Block is 223 × 65 × 50 mm; the body envelope reaches 71 mm wide (sidepods to ±35.5) and 63.5 mm tall. It exceeds the block's cross-section whichever way the block is oriented. T3.5's 65 mm is the assembled car, not the milled body. | `bounding_volumes.py:557-562`; `unified_phi.py:102-107`; `machined_length_mm` is only called from a test | cap milled labels to the block cross-section in `default_rule_envelope`; add block length, width and height to the finished-surface checks |
| A2 | CRIT | **The rear wheel keep-out column breaks the T5.5 wall** (Safety, 10 points plus T3.2). The +4 mm inboard margin reaches y = 10.25 mm, full height, and air wins over the forced-solid ring. Measured wall 2.0 mm at 1 mm spacing, 1.25 mm at 0.5 mm, against a 3.0 mm minimum. | `unified_phi.py:258-264, 667` | give the T5.5 ring precedence over wheel masks; drop the 4 mm margin (§5.4); add a finished-surface 3 mm-ball check around the bore |
| A3 | HIGH | **The machining-radius gate never fires.** It inverts trimesh's curvature *measure* (an area-weighted quantity), so a 3 mm fillet reads as about 25 m. | `surface_extraction.py:234-243`; only test is "returns an array" | field-space opening test: air not reachable by a 3.15 mm ball is an unmillable corner |
| A4 | HIGH | **The mass term eats the nose six times faster than the body**, because the velocity scales with density (1000 vs 163). The aero-critical part goes first. | `phi_updater.py:483`; measured nose 20.7 g of 85.3 g as built | nose leaves the field (Part 4); until then, exclude it from the mass velocity |
| A5 | HIGH | **Quality gates rewrite the field.** The nose thin-wall repair overwrites the whole grid with a voxel distance transform, re-staircasing the car between the adjoint step and the CFD. | `surface_extraction.py:292-296, 420-428`; called at `unified_phi.py:1006, 1029` | gates work on a copy and report; repairs happen in the update step, locally |
| A6 | HIGH | **Disconnected pieces are dropped silently.** Repair keeps the largest component, so `connected_bodies` is always 1 and a detached sidepod vanishes from the CFD while its mass is still counted. T4.1 is unverified. | `surface_extraction.py:114-116`; `unified_phi.py:1062` | count components on the field (`ndimage.label`) and fail on more than one |
| A7 | HIGH | **No finished-surface rule checks.** The rule stage compares the mesh with the envelope box and always passes. Legality rests entirely on masks, and A1 and A2 show two are wrong. | `unified_phi.py:1057`; `surface_extraction.py:568-595` | the scrutineer engine in PART5 §2, run on every candidate that reaches CFD |
| A8 | HIGH | **The halo is modelled as a 45 mm box**; the real part is 34–39 mm. Visibility shadows are cast from the box, and no helmet exception. Half the legal envelope is forced air before any optimisation (50.7 % measured). | `fixed_hardware.py:773-774, 790-799, 951-1030` | voxelise the placed `halo_helmet.stl` and derive the pocket and view silhouettes from it |
| A9 | MED | Cargo erosion tolerance of 5 % knowingly allows a T4.2 cargo under its minimum (25 points). | `unified_phi.py:180, 657` | zero tolerance for interior voids; move the placement instead |
| A10 | MED | Two inconsistent machinability models, neither with a tool radius: a 1-cell slot passes one, any occlusion fails the other. | `unified_phi.py:819-827`; `surface_extraction.py:471-542` | one field-space test: air reachable by the tool disc along ±Y/±Z |
| A11 | MED | Hard masks are ±dx plateaus, so every mask boundary is a permanent voxel staircase, likely a source of the adjoint spikes. | `phi_grid.py:112-117`; `phi_updater.py:196-201` | store masks as signed distances and combine with max/min |
| A12 | MED | Loft ceiling extends across the full width to the tail, forbidding any bodywork above ≈ 47 mm aft of the halo (including a rear-wing support root) with no rule behind it. | `fixed_hardware.py:1168-1178` | drop it (§5.4) |
| A13 | MED | Without Part 2 on the path, Stage 1 silently builds cars with no hardware voids. | `unified_phi.py:556-560` | raise when `fixed_hardware is None` |
| A14 | MED | Full-car ASCII STL written every iteration (≈ 235 MB), reintroducing a disk-fill bug fixed elsewhere. | part3 `pipeline_interface.py:831` | binary, or accepted candidates only |
| A15 | LOW | Cargo pinned at 14–24 mm under the halo; T4.2 only requires "between the axles". It limits floor pockets under a 55 × 60 mm footprint. | `virtual_cargo.py:32-48` | revisit once pockets exist |
| A16 | LOW | Half-surface cap forms at y = −dx/2 and is snapped to 0, making slivers on the symmetry plane. | `unified_phi.py:893-902` | set the y = 0 row to φ = 0, or cut at the cell face |
| A17 | LOW | Grid spacing is a sandbox monkey-patch of module globals; the driver that sets it is missing. | `sandbox/coarse.py:74-90` | spacing as a builder argument (`UnifiedGeometry.spacing_m` already exists) |
| A18 | LOW | Dead and duplicated code: the four-grid path (`_level2_evaluate`, `phi_grid_factory`, `quality_gates`, `stl_assembler.assemble_stl`, `mass_com_calculator`, `apply_adjoint_sensitivity_symmetric`), `canister_zone_contact_shell_mask` (0 callers), duplicate density and hardware-COM helpers. | audit list | delete after the unified path has a test for each behaviour they covered |
| A20 | HIGH | T7.9 zones are anchored on the regulation-minimum wheel gaps (front y = 19 mm), not the real wheels (23.25 front, 16.25 rear). T7.9 is measured from the actual inside contact edge, so the front zones sit 4.25 mm too far inboard and cost body volume. | `wheel_visibility_zones.py:135-136` | pass the real inner-face y from `geometry_contract` |
| A21 | HIGH | Nothing bounds the roof ahead of or beside the halo except 65 mm: floating overhangs up to 63.5 mm at x = 40 are representable. | measured; `fixed_hardware.py:1002-1028, 1177-1178` | block cap (A1) plus the parametric seed (§5.1) |
| A22 | MED | Mass is voxel-counted, so it moves in whole-cell steps: roughly ±2 g of noise at 0.5 mm against a 0.5 g T3.6 margin, and a discontinuous gradient. | `unified_phi.py:1088-1101` | smoothed-Heaviside volume fraction |
| A23 | MED | `extend_velocity` updates interface cells too, advecting the splatted surface velocity away from where it was set. | `phi_updater.py:261, 301` | freeze the interface band, smoothed sign |
| A24 | MED | The per-vertex splat is a Python loop, run twice per iteration. | `phi_updater.py:359-366` | `np.add.at` |
| A25 | LOW | The Stage 1 cargo scorer uses the uncarved brick's mass and COM (≥ 85 g), biasing cargo placement. | `stage1_search.py:236-245` | score on the carved car |
| A26 | LOW | The printed nose is modelled as solid at 1000 kg/m³ (20.7 g as built); it can be a 2 mm shell. | `geometry_contract.py:97`; `surface_extraction.py:41-45` | Part 4 nose with declared wall and infill |
| A27 | MED | `hardware_geometry.build_wheel_assembly` fails on the current wheel CAD: the wheel STLs store their width along x, the code measures it along y and asserts "width = 28.2 mm". Nothing on the production path calls it, which is why it went unnoticed; any rendering or full-car CFD export through it breaks. | `hardware_geometry.py:143-150`; found in R&D 2026-09-25 | measure width and axis from the mesh's principal axes, not a fixed axis |
| A19 | LOW | Stale docs: BUILD_REPORT, PLACEHOLDERS items 1–4, 11, 16; SPEC_ASCII d_halo bound; `geometry_contract.py:74-75`; `stage1_search.py:123-133` ("30 steps" vs 100); T5.5 "not consumed". | audit list | update with this plan |

**Test gaps that matter:** the radius gate on a known fillet (A3); the T5.5 wall on the sides of the bore (A2);
any finished-surface rule (A7); block dimensions (A1); component count before pruning (A6); gate side effects on
φ (A5). Every test runs at 1.5–3 mm spacing, so behaviour at the 0.5 mm production spacing is untested.

### 9.1 Pocket implementation, concretely (refines §4)

- `UnifiedGeometry` gains `pockets` (bool grid, empty at build) and `ballast_kg`. φ stays the aero skin.
  Manufactured solid = (φ < 0) ∧ ¬pockets.
- Admissible pocket cells: deeper than the skin thickness T_skin (≈ 3 mm) from the outer surface; not in
  `hard_mask_solid` (cargo, T5.5 ring); open to the floor (every cell below is pocket down to the underside,
  the `_clear_run` test already in `enforce_machinability`); opened by a 3.15 mm disc per z-slice so walls and
  corners meet the cutter; depth within tool reach.
- Mass controller, closed form, no level set: surplus mass dilates pockets one layer inside the admissible set;
  deficit erodes them; ballast takes the remainder up to capacity.
- CFD sees the skin only if the pocket mouths are sealed by a floor plate — they are not (§4.1), so the CFD STL
  must be extracted from max(φ, sdf_pockets) whenever pockets exist. This is the one place the audit's
  suggestion ("pocket mouths sit in the ground gap, close them for CFD") is not safe to adopt: an open channel
  in a 1.5 mm gap changes the underbody flow.
- `compute_mass_com` and `density_field` use the manufactured solid; ballast enters at the container COM.

---

## 10. Questions that change this part

1. Ballast material (lead ≈ 15.8 g or tungsten alloy ≈ 25 g in the container).
2. Is the ≈ 5 g rear-wing placeholder, the 1.07 g halo and the v2 support masses what you will build?
   They set m_hw and so the body's size.
3. Any objection to dropping the "outrigger look" 4 mm wheel keep-out and the loft deck?
