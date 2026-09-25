"""
wheel.py -- parametric STEM Racing wheel: mass, rotational inertia, relative stiffness.

Geometry (matches the team's v2 CAD wheel family, measured in cad_wheels.json):
  rim   : cylindrical shell, outer radius R, radial thickness t_rim, axial width w
          (the whole contact width, T7.4/T7.7), optional inboard lip
  plate : a central spoke plate of axial thickness t_plate spanning r_hub..R-t_rim,
          with N spokes of tangential width b_spoke (straight-spoke approximation)
  hub   : ring r_bore..r_hub over axial length l_hub
  cap   : optional full-face disc of thickness t_cap (hubcap)

All lengths mm, density g/cm^3, results in g and g*mm^2.

Stiffness is RELATIVE to the CAD wheel, which passes the T7.13 100 g hang test.
Two failure modes are tracked, both from thin-shell/beam scaling:
  S_span  rim bending between spokes, a curved beam of span L = 2*pi*R/N:
          stiffness ~ E * w * t^3 / L^3
  S_edge  the unsupported rim overhang either side of the plate, a cylindrical
          shell cantilever of length a = (w - t_plate)/2 under a radial edge load:
          stiffness ~ E * t^3 / a^2 * sqrt(t/R)^-1 ... simplified to E * t^2.5 / (a^2 * R^0.5)
          (shell bending length sqrt(R t) sets the loaded zone)
A design is acceptable when both ratios to the CAD wheel are >= 1. The model
is a scaling law, not an FE result: it ranks designs, and the physical 100 g
hang test on printed samples is the acceptance test.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

MATERIALS = {
    # name: (density g/cm3, E GPa, min printable wall mm, note)
    "ABS_FDM":       (1.04, 2.2, 0.40, "current wheels; 0.4 mm nozzle"),
    "PLA_FDM":       (1.24, 3.5, 0.40, "stiffer, denser"),
    "PETG_FDM":      (1.27, 2.1, 0.40, ""),
    "SLA_standard":  (1.15, 2.6, 0.30, "thin walls, brittle"),
    "SLA_tough":     (1.14, 1.9, 0.30, ""),
    "PA12_SLS":      (1.01, 1.7, 0.60, "SLS nylon"),
    "CF_tube_rim":   (1.55, 50.0, 0.15, "rolled carbon tube as the rim only"),
}


@dataclass(frozen=True)
class Wheel:
    R: float = 14.12
    w: float = 13.25
    t_rim: float = 0.40
    t_plate: float = 2.93
    r_hub: float = 5.0
    r_bore: float = 4.5
    l_hub: float = 2.93
    n_spokes: int = 7
    b_spoke: float = 1.29         # tangential spoke width mid-span (mm), CAD-calibrated
    r_fillet: float = 1.5         # root fillet radius at hub and rim ends (mm)
    plate_fill_extra: float = 0.0 # extra plate area fraction (fillets, webs)
    t_cap: float = 0.0
    rim_material: str = "ABS_FDM"
    body_material: str = "ABS_FDM"

    # ---- mass properties -------------------------------------------------
    def parts(self):
        rho_r = MATERIALS[self.rim_material][0] * 1e-3      # g/mm3
        rho_b = MATERIALS[self.body_material][0] * 1e-3
        R, ri = self.R, self.R - self.t_rim
        out = {}
        v = math.pi * (R**2 - ri**2) * self.w
        out["rim"] = (rho_r * v, rho_r * v * (R**2 + ri**2) / 2)
        # spokes: N spokes of width b plus root fillets of radius rf at both
        # ends (width grows by 2*rf*(1-d/rf)^2 within rf of each root).
        # Integrated numerically over r in the plate of thickness t_plate.
        n = 400
        rr = [self.r_hub + (ri - self.r_hub) * (k + 0.5) / n for k in range(n)]
        dr = (ri - self.r_hub) / n
        m_sp = i_sp = 0.0
        for r in rr:
            b = self.b_spoke
            for d in (r - self.r_hub, ri - r):
                if d < self.r_fillet:
                    b += 2 * self.r_fillet * (1 - d / self.r_fillet) ** 2
            b = min(b, 2 * math.pi * r / max(self.n_spokes, 1))
            dm = rho_b * self.n_spokes * b * self.t_plate * dr
            m_sp += dm
            i_sp += dm * r * r
        out["spokes"] = (m_sp, i_sp)
        if self.plate_fill_extra:
            v2 = math.pi * (ri**2 - self.r_hub**2) * self.t_plate * self.plate_fill_extra
            out["plate_fill"] = (rho_b * v2, rho_b * v2 * (ri**2 + self.r_hub**2) / 2)
        v = math.pi * (self.r_hub**2 - self.r_bore**2) * self.l_hub
        out["hub"] = (rho_b * v, rho_b * v * (self.r_hub**2 + self.r_bore**2) / 2)
        if self.t_cap:
            v = math.pi * (ri**2 - self.r_hub**2) * self.t_cap
            out["cap"] = (rho_b * v, rho_b * v * (ri**2 + self.r_hub**2) / 2)
        return out

    @property
    def mass(self):
        return sum(m for m, _ in self.parts().values())

    @property
    def inertia(self):
        return sum(i for _, i in self.parts().values())

    # ---- relative stiffness ----------------------------------------------
    def _E_rim(self):
        return MATERIALS[self.rim_material][1]

    def s_span(self):
        L = 2 * math.pi * self.R / self.n_spokes
        return self._E_rim() * self.w * self.t_rim**3 / L**3

    def s_edge(self):
        a = max((self.w - self.t_plate) / 2, 1e-6)
        return self._E_rim() * self.t_rim**2.5 / (a**2 * self.R**0.5)

    def check(self, ref: "Wheel"):
        return self.s_span() / ref.s_span(), self.s_edge() / ref.s_edge()

    def printable(self):
        return (self.t_rim >= MATERIALS[self.rim_material][2] - 1e-9 and
                self.b_spoke >= MATERIALS[self.body_material][2] and
                (self.t_cap == 0 or self.t_cap >= MATERIALS[self.body_material][2]))


CAD_FRONT = Wheel(w=13.25)
CAD_REAR = Wheel(w=17.25)
