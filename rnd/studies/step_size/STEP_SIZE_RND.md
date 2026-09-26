# Step-size R&D — why one CFD+adjoint iteration barely moves the car

Files: `step.py` (experiment), `step_results.json`.

## Setup

- The leader-like car (W 120.3, x_front 46, d_halo 43.72) from the project's own Stage-1
  carve, remapped to a 1.0 mm grid. At CFL 0.3 the most any cell can move in one HJ step
  is 0.30 mm.
- The **real** adjoint sensitivity from the GitHub Actions run (medium mesh, 14,583 car
  points), mapped to the STL vertices as production does (nearest point within 5 mm:
  10.3 % frozen, median match 0.56 mm), scaled as `cfd_wrapper` does (× 0.449 × 1.225 × 2).
- Mass gradients at the leader's operating point (dT/dm 16.7 s/kg, dT/dh −0.092 s/m).
- One shape update per variant from the same starting field. Displacement is measured at
  the old interface (|φ| < dx) as the change in φ.

## Results

| variant | median move | p90 move | interface cells moving > 0.05 mm | mass change | what set dt |
|---|---|---|---|---|---|
| production, 1 step | 0.049 mm | 0.062 mm | 39 % | −1.30 g | nose |
| aero only | 0.006 mm | 0.057 mm | 12 % | −0.36 g | main body |
| uniform density (nose not 6×) | 0.132 mm | 0.164 mm | 99 % | −2.28 g | main body |
| dt from band p90, clip 3× | 0.088 mm | 0.112 mm | 98 % | −2.20 g | nose |
| smooth 2 mm | 0.049 mm | 0.059 mm | 40 % | −1.31 g | nose |
| smooth + band dt + uniform density | 0.090 mm | 0.108 mm | 99 % | −2.17 g | rear pod |
| same, 5 sub-steps | **0.451 mm** | 0.529 mm | 100 % | −3.10 g | rear pod |
| aero only, smooth 2 mm | 0.012 mm | 0.081 mm | 19 % | −0.43 g | |
| aero only, smooth 2 mm + band dt | 0.021 mm | 0.144 mm | 33 % | −0.53 g | |
| aero only, smooth 4 mm + band dt | 0.024 mm | 0.131 mm | 34 % | −0.49 g | |

Time per update: 34 s, of which **reinitialisation is 26 s**, velocity extension 5.7 s,
splatting 1.1 s, the HJ step itself 0.4 s.

## What it shows

1. **The production step moves the surface at one-sixth of the cap**, and the drag term alone
   at one-fiftieth (median). That is the ~40× shortfall seen in production.
2. **The nose sets the time step.** The mass velocity is ρ·dT/dm, and the printed nose is
   modelled at 1000 kg/m³ against 163 for the foam, so the nose moves 6× faster and caps dt
   for the whole car. Removing that alone gives 2.7×.
3. **dt from the maximum is dominated by a few cells.** Sizing it on the interface p90 and
   clipping at 3× gives 1.8×.
4. **The drag sensitivity is genuinely concentrated.** Even smoothed and properly stepped, the
   aero-only median stays small while the p90 reaches 0.14 mm. That is correct behaviour: a
   drag step should move the high-sensitivity regions most. The way to use it is more motion
   per adjoint, not a different normalisation.
5. **Sub-steps are the big multiplier.** Five sub-steps on one adjoint give 0.45 mm median,
   9× production. A trust radius of about 1 mm of p90 motion per adjoint (≈ 7 sub-steps,
   re-extending the velocity each time) moves the car roughly 15–20× further per CFD solve.
6. **Reinitialisation is 75 % of the cost.** A narrow-band reinit (|φ| < 4 dx) and fewer
   pseudo-steps would cut the update from 34 s to a few seconds.

## Recommended change to `phi_updater.apply_adjoint_to_unified`

- Nose (and every printed part) excluded from ρ·dT/dm on the skin, or the ballast model of
  PART1 §3 so the skin carries no mass velocity at all.
- dt from the interface p90 of |V|, with |V| clipped at 3× that value.
- Surface Helmholtz smoothing of the sensitivity, radius 2 mm, before splatting: it barely
  changes speed but removes the vertex speckle that makes steps jagged.
- K sub-steps per adjoint until the p90 displacement reaches the trust radius (≈ 1 mm, below
  one medium CFD cell of 1.35 mm), recomputing mass/COM each sub-step (free).
- Narrow-band reinitialisation.

Expected effect: 25 CFD iterations that today move the car about 1 mm in total would move it
about 15–25 mm where the gradient is strongest.
