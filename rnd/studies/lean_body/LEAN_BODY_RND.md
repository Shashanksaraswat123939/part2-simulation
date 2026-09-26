# Lean-body R&D — what legal ballast does to the car

Files: `lean.py` (study), `metrics.json`, `compare.png`, `*_half.stl` (CFD inputs, also
committed gzipped on the `rnd/cfd-validation` branch), `*_full.stl`.

## Method

The project's own Stage-1 carve (`_level2_evaluate_unified`, 250 steps at 2 mm) at the
leader's scalars (W 120.3, x_front 46, d_halo 43.72), changed in two ways only:

1. **Ballast**: the mass/COM state gains ballast at the centroid of the legal capsule under
   the halo (12.7 × 20 × 6.35 mm = 1.39 cm³), so foam supplies only what is left of 48.2 g.
2. **Block**: the milled envelope is capped to the 223 × 65 × 50 mm Model Block (today's
   envelope allows 71 mm wide and 63.5 mm tall).

## Result

| | baseline (today) | lead ballast | tungsten ballast |
|---|---|---|---|
| ballast | 0 g | 15.8 g | 25.1 g |
| foam + nose | 32.5 g | 16.5 g | 7.3 g |
| competition mass | 47.9 g | 47.7 g | 47.9 g |
| body volume | 186 cm³ | 101 cm³ (−45 %) | 47 cm³ (−75 %) |
| frontal area | 1996 mm² | 1545 mm² (−23 %) | 1218 mm² (−39 %) |
| wetted area | 29,841 mm² | 22,449 mm² (−25 %) | 17,098 mm² (−43 %) |
| width / height | 62.1 / 47.6 mm | 56.1 / 47.6 mm | 56.1 / 47.6 mm |

CFD (GitHub Actions, coarse, corrected inlet, fixed frame):

| | body only | with rotating wheels, supports, halo |
|---|---|---|
| baseline | 0.2549 N | 0.3978 N |
| lead | 0.2412 N (−5.4 %) | 0.4360 N (**+9.6 %**) |
| tungsten | 0.1518 N (−40.4 %) | see round 3 |

## What it shows

- Ballast lets the body shrink dramatically at the same competition mass. This part of the
  plan is confirmed.
- **But shrinking the body with a mass-only carve made the full car slower.** With wheels in
  the flow, the lean car's wheels carried 0.326 N against 0.283 N on the baseline: the wider
  baseline body was shielding the rear wheels. Body shape must be optimised *with the wheels
  in the CFD*, never alone.
- **Regulated solids now set the shape.** In the lean cars the 55 mm wide end of the T4.2
  virtual cargo (placed wide-end-forward at z 14–24 mm) makes a flat slab across the car, and
  the loft deck plus the cartridge shroud fix the top at 47.6 mm in all three cars. Cargo
  orientation and height, the deck, and the chamber height (T5.2 allows 30–40 mm; the code
  fixes 35) are the next shaping levers.
- The carve lost the nose entirely in the ballast cars (nose mass 0). Those cars are aero
  estimates, not legal cars; Part 4's parametric nose and wings replace this.
- All three land a few tenths of a gram under the 48.2 g target; the proxy's known
  undershoot. The ballast model should aim at 48.0 + margin exactly.
