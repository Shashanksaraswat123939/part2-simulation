"""
physics_contract.py

Stage 1 of the Part 2 build: the shared coordinate/unit contract layer.

Every other module in Part 2 imports FROM this module and never re-derives
axis conventions or unit conversions locally. This is deliberate: a unit or
axis bug here silently corrupts every downstream stage (CFD wrapper, RTC,
adjoint), and is the hardest class of bug to catch later. Keep this module
small, dependency-free, and paranoid.

Coordinate convention (must match Part 1 exactly):
    x = front to rear
    y = centerline to outside
    z = track upward

Internal unit convention (SI, enforced everywhere past this boundary):
    mass     -> kg
    length   -> m
    force    -> N
    density  -> kg/m^3
    area     -> m^2
    speed    -> m/s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def grams_to_kg(g: float) -> float:
    """Convert grams to kilograms. No rounding, no clamping."""
    return g / 1000.0


def kg_to_grams(kg: float) -> float:
    """Convert kilograms to grams. Inverse of grams_to_kg."""
    return kg * 1000.0


def mm_to_m(mm: float) -> float:
    """Convert millimeters to meters."""
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    """Convert meters to millimeters. Inverse of mm_to_m."""
    return m * 1000.0


def gcm3_to_kgm3(density_g_cm3: float) -> float:
    """Convert g/cm^3 to kg/m^3. 1 g/cm^3 = 1000 kg/m^3 exactly."""
    return density_g_cm3 * 1000.0


REFERENCE_SPEED_MPS: float = 20.0          # m/s, CFD reference condition
AIR_DENSITY_KGM3: float = 1.225            # kg/m^3
GRAVITY_MPS2: float = 9.81                 # m/s^2
TRACK_LENGTH_M: float = 20.0               # m, regulation race distance

# Center of rotation the CFD pitching moment is reported about (SPEC "pitching
# moment about car reference point"). Part 1 emits the STL in world coordinates
# with x=0 at the nose tip on the ground plane, so (0,0,0) is that point. This
# is the single source of truth for the OpenFOAM forces/forceCoeffs CofR and
# for any later Cm convention — closes audit P2-12 ("moment reference pinned
# nowhere"). It is a documented convention, not a physical claim; if the team
# later prefers the front axle or mid-wheelbase, change it here only.
MOMENT_REFERENCE_POINT_M: tuple[float, float, float] = (0.0, 0.0, 0.0)

# One source for the two unmeasured objective inputs (2026-09-25). Callers used
# mu 0.4 / 0.02 / 0.010 and wheel MOI 1e-6 / 1e-7 for the same car.
#   DEFAULT_WHEEL_MOI_KG_M2: v2 CAD wheels, mean of front 125.0 and rear 153.0
#       g.mm2 at ABS 1.04 g/cm3 (rnd/wheels). Part 4's designed wheel overrides.
#   DEFAULT_ROLLING_MU: PLACEHOLDER. Replace with a coast-down fit.
DEFAULT_WHEEL_MOI_KG_M2: float = 1.39e-7
DEFAULT_ROLLING_MU: float = 0.010

# time_coefficient is FROZEN. It must never be exposed as a callable
# parameter anywhere in Part 2 (see Stage 6). Any code that needs it reads
# this constant directly; nothing may accept it as a function argument.
TIME_COEFFICIENT: float = 1.0


@dataclass(frozen=True)
class HalfCarQuantities:
    """Raw forces/moments straight off the right-half CFD domain.

    All values SI. Cm is the pitching moment COEFFICIENT computed about
    the car reference point using the HALF-domain force/area convention --
    it is NOT yet in the full-car convention. Use `to_full_car()` to convert.

    Fields:
      D20                     : half-car drag force at reference speed, N
      L                       : half-car lift force at reference speed, N
      A                       : half-car frontal projected area, m^2
      pitching_moment_half    : half-car raw pitching moment about the car
                                 reference point, N*m
      q_ref                   : dynamic pressure at reference speed, Pa.
                                 Computed as a default value from
                                 AIR_DENSITY_KGM3 and REFERENCE_SPEED_MPS
                                 so callers do not have to pass it
                                 explicitly in the ordinary case, but it is
                                 still a real field (not a hidden module
                                 global) so tests can override it if a
                                 non-default reference condition is ever
                                 needed.
    """

    D20: float
    L: float
    A: float
    pitching_moment_half: float
    q_ref: float = 0.5 * AIR_DENSITY_KGM3 * REFERENCE_SPEED_MPS ** 2

    def __post_init__(self):
        # Guard against negative forces (physically meaningless) and
        # negative area. These indicate a CFD bug or mesh corruption.
        if self.D20 < 0:
            raise ValueError(f"D20 (drag) must be non-negative, got {self.D20}")
        if self.A < 0:
            raise ValueError(f"A (area) must be non-negative, got {self.A}")

    def to_full_car(self) -> "FullCarQuantities":
        """Convert half-car CFD outputs to full-car values.

        Arguments:
            self: half-car forces in N, half-car area in m^2, half-car
                pitching moment in N*m, and q_ref in Pa.

        Returns:
            FullCarQuantities with D20 and L in N, A in m^2, and Cm
            dimensionless.

        Invalid input behavior:
            Degenerate zero-area shapes do not raise ZeroDivisionError;
            Cm is returned as 0.0 when A_full <= 0.
        """
        D20_full = 2.0 * self.D20
        L_full = 2.0 * self.L
        A_full = 2.0 * self.A
        moment_full = 2.0 * self.pitching_moment_half

        # PLACEHOLDER -- replace with real reference length
        # (likely wheelbase W) once confirmed; sqrt(A_full) is a
        # stand-in only, chosen so the formula is dimensionally
        # sane (area * length has units of m^3, matching moment
        # units of N*m when divided into q*A*L) and not because
        # it is physically correct.
        ref_length = A_full ** 0.5

        # Use self.q_ref (which defaults to the module constant but can be
        # overridden) instead of recomputing from AIR_DENSITY_KGM3 and
        # REFERENCE_SPEED_MPS. This preserves any caller-specified q_ref.
        q_full = self.q_ref
        if A_full > 1e-12:
            Cm_full = moment_full / (q_full * A_full * ref_length)
        else:
            Cm_full = 0.0
        return FullCarQuantities(D20=D20_full, L=L_full, Cm=Cm_full, A=A_full)


@dataclass(frozen=True)
class FullCarQuantities:
    """Full-car aerodynamic quantities.

    Fields:
      D20 : N, full-car drag force at reference speed
      L   : N, full-car lift force at reference speed
      Cm  : dimensionless, full-car pitching moment coefficient
      A   : m^2, full-car frontal projected area
      force_oscillation : peak-to-peak swing of streamwise force over the
          averaged window, as a fraction of its mean. None when not measured.
          The ERROR BAR on D20, and the reason it lives here rather than in a
          log line: measured 26-33% on the real geometry against a 5% limit, so
          drag differences under ~15% are not resolved by the solve that
          produced them. A ranking that carries D20 but not this has thrown
          away the only thing that says whether the ranking means anything.
    """

    D20: float
    L: float
    Cm: float
    A: float
    force_oscillation: Optional[float] = None


@dataclass(frozen=True)
class ComponentMassCOM:
    """Mass and center-of-mass for a single component.

    Fields:
      name    : component identifier string.
      mass_kg : mass in kilograms.
      com_x_m, com_y_m, com_z_m : center of mass position in meters.
    """

    name: str
    mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float


@dataclass(frozen=True)
class FullCarMassCOM:
    """Aggregated full-car mass and center of mass.

    Fields:
      total_mass_kg : kg, sum of every component's mass.
      com_x_m       : m, mass-weighted fore-aft COM position.
      com_y_m       : m, mass-weighted lateral COM position.
      com_z_m       : m, mass-weighted vertical COM position.
      components    : tuple of ComponentMassCOM.
    """

    total_mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float
    components: tuple = field(default_factory=tuple)
    # CO2 PROPELLANT, carried separately and deliberately EXCLUDED from
    # total_mass_kg / com_*_m above. Those describe the car as it crosses the
    # line: everything permanently aboard. The propellant is aboard only at the
    # start (~7.9 g) and is gone by t ~ 0.2 s, and race_objective owns its mass
    # via `sheet_mass(t) - mass_sheet_final`. Including it here would recreate
    # the double-count corrected on 2026-07-24.
    #
    # It is reported because it MOVES THE COM WHILE IT IS THERE: at the design
    # default it sits at x=208 mm / z=35 mm, well aft and above the dry COM, so
    # at launch it pulls com_x +13.5 mm and com_z +0.75 mm. That is worth <=0.11 ms
    # of race time (negligible), but it lands squarely on the static-stability
    # check at peak thrust -- the worst instant for a wheelie. Consumers wanting
    # launch-condition COM should blend it in via com_with_propellant().
    propellant_mass_kg: float = 0.0
    propellant_com: tuple = (0.0, 0.0, 0.0)

    def com_with_propellant(self) -> tuple:
        """(total_mass_kg, com_x_m, com_y_m, com_z_m) at the START LINE.

        The mass-weighted blend of the dry car and a full propellant charge.
        Use for launch-instant analysis (static stability, wheelie margin);
        use the plain fields for anything describing the car after ~0.2 s.
        """
        m = self.total_mass_kg + self.propellant_mass_kg
        if m <= 0:
            return (0.0, self.com_x_m, self.com_y_m, self.com_z_m)
        px, py, pz = self.propellant_com
        w, p = self.total_mass_kg, self.propellant_mass_kg
        return (
            m,
            (w * self.com_x_m + p * px) / m,
            (w * self.com_y_m + p * py) / m,
            (w * self.com_z_m + p * pz) / m,
        )
