# Wheel R&D — measured CAD wheels, a validated model, and what a better wheel is worth

Files: `wheel.py` (model), `search.py` (design search), `test_wheel.py` (runnable check),
`cad_wheels.json`, `race_T_vs_I.json`, `designs.json`, `tradeoff.json`, `cad_wheel_*.png`.

## 1. The real wheels (hardware_cad v2, ABS 1.04 g/cm³)

| | front | rear |
|---|---|---|
| outer radius | 14.12 mm | 14.12 mm |
| contact width | 13.25 mm | 17.25 mm |
| mass | 0.806 g | 0.951 g |
| inertia about the axle | 125.0 g·mm² | 153.0 g·mm² |
| shape factor k = I/(m R²) | 0.78 | 0.81 |
| mass in the outer 0.6 mm | 61 % | 67 % |

Structure, from sections: a 0.40 mm rim shell over the full contact width, seven curved
spokes in a 2.9 mm central plate, a 9 mm bore. The rim carries about 75–80 % of the inertia.

**Correct value for the race objective:** mean I = 139 g·mm² = 1.39e-7 kg·m².
Stage 1 uses 1e-7 (close). `SearchConfig`'s default 1e-6 is 7× too high and adds about
20 g of phantom effective mass.

## 2. Model

A rim shell + spokes with root fillets + hub, integrated numerically. Against the CAD:
inertia within 1.1 %, mass within 3.1 % (front and rear). Stiffness is scored relative to
the CAD wheel, which passes the T7.13 100 g hang test, with two scaling laws: rim bending
between spokes (E·w·t³/L³) and the unsupported rim overhang either side of the spoke plate.

## 3. Race value of inertia

Linear over the useful range: **0.333 ms per g·mm² of mean wheel inertia** (all four wheels,
the project's thrust curve). With a realistic thrust curve (about 1.5× impulse, see the
master R&D report) multiply every ms figure here by about 0.6.

## 4. Best designs (contact width at T7.4 minimum + 0.1 mm, R = 14.05 mm)

| rim material | stiffness floor vs CAD | mean I (g·mm²) | race time | rim t | spokes | spoke w | plate t |
|---|---|---|---|---|---|---|---|
| ABS FDM | 1.0 | 117.0 | −6.9 ms | 0.40 | 7 | 0.6 | 2.93 |
| ABS FDM | 0.5 | 109.2 | −9.5 ms | 0.40 | 6 | 0.6 | 1.2 |
| SLA resin | 1.0 | 122.9 | −4.9 ms | 0.35 | 8 | 0.6 | 4.0 |
| SLA resin | 0.5 | 101.6 | −12.0 ms | 0.30 | 7 | 0.6 | 2.93 |
| carbon tube rim, 0.15 mm | 1.2 | 66.0 | **−23.9 ms** | 0.15 | 7 | 0.6 | 1.2 |
| carbon tube rim, 0.20 mm | 2.9 | 85.6 | −17.4 ms | 0.20 | 7 | 0.6 | 1.2 |
| carbon tube rim, 0.25 mm | 5.3 | 104.9 | −10.9 ms | 0.25 | 7 | 0.6 | 1.2 |

Reading:
- **Printed wheels are close to their floor.** The rim is already at the 0.4 mm FDM minimum,
  and the contact band's area is fixed by T7.4. Thinner spokes and a thinner plate save
  7–12 ms.
- **The rim material is the lever.** A rolled carbon-fibre tube of 0.15–0.20 mm wall as the
  rim, with a printed hub and spokes, saves 17–24 ms while being *stiffer* than today's
  wheel. T1.8 allows a wheel to be an assembly.
- **Do not trade stiffness for inertia.** A World Finals team measured stiff 1.28 g wheels
  beating flexible 0.80 g ones on the track (1.182 vs 1.185 s) because rim deformation costs
  rolling losses. Every design above is at least as stiff as the current wheel or states
  its floor.
- **A hubcap costs about 1.6 ms per wheel** (a 0.3 mm full-face disc adds ≈ 19 g·mm²).
  CFD round 2 shows wheels are 64–75 % of the car's drag, so a cap that trims even 2 % of
  wheel drag pays for itself. Test it in CFD (closed wheel) before printing.

## 5. What must be confirmed physically

Print 3–5 designs, hang 100 g on each (T7.13), spin-down test with the chosen bearing
(the same team measured hybrid ceramic SMR73C bearings at 54 s spin time vs 9.9 s for
steel), and weigh. The stiffness model ranks designs; the hang test accepts them.
