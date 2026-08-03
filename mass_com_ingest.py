"""Stage 2 mass/COM ingestion from Part 1 outputs."""

from __future__ import annotations

import warnings
from typing import Optional

from dataclasses import dataclass

from physics_contract import ComponentMassCOM, FullCarMassCOM


# Broad sanity bounds for COM coordinates (m). A STEM Racing car is ~150mm long;
# COM at 10m is clearly a units or origin bug, not a legitimate design.
COM_SANITY_BOUNDS_M = (-10.0, 10.0)
# The bounds are a units/origin smoke alarm three orders of magnitude away from
# any real value, so comparing to them with exact float `<=` is pointless
# precision: a mass-weighted sum landing on 10.000000000000002 is "10", not a
# units bug. Without this, the boundary case fails on representation error
# alone (test_com_sanity_bounds_exactly_at_limit_accepted).
COM_SANITY_TOL_M = 1e-9

# 23 g, fixed. This is the EMPTY (spent) cartridge -- the hardware that stays
# with the car for the whole run. The CO2 PROPELLANT is NOT included here and
# must not be: it leaves the vehicle during the run, so it is supplied by
# race_objective.car_mass_from_time as `sheet_mass(t) - mass_sheet_final`
# (7.89 g at the line, 0 g at the finish).
#
# The boundary was previously undefined, and both sides assumed they owned the
# cartridge: this constant went into car_weight_kg AND the objective added its
# own 21 g shell on top. Corrected 2026-07-24 -- geometry owns what stays,
# the objective owns what burns off.
CO2_CARTRIDGE_MASS_KG = 0.023


@dataclass(frozen=True)
class FixedHardwareSpec:
    """Fixed hardware masses and COMs.

    Arguments:
        co2_cartridge_mass_kg: CO2 cartridge mass in kg; must equal 0.023.
        co2_cartridge_com: CO2 cartridge COM in m as (x, y, z).
        rear_wing_mass_kg: rear wing mass in kg, supplied by caller.
        rear_wing_com: rear wing COM in m as (x, y, z).
        wheels_axles_mass_kg: wheels plus axles mass in kg, supplied by caller.
        wheels_axles_com: wheels plus axles COM in m as (x, y, z).

    Returns:
        FixedHardwareSpec instance; no unit conversion is performed.

    Invalid input behavior:
        Raises ValueError if co2_cartridge_mass_kg differs from 0.023 by more
        than 1e-9 kg. All fields are mandatory constructor arguments with no
        defaults — Python's dataclass machinery enforces that callers supply
        every value. There is no runtime NotImplementedError check; the
        enforcement is structural via required fields.
    """

    co2_cartridge_mass_kg: float
    co2_cartridge_com: tuple[float, float, float]
    rear_wing_mass_kg: float
    rear_wing_com: tuple[float, float, float]
    wheels_axles_mass_kg: float
    wheels_axles_com: tuple[float, float, float]
    # Optional so existing callers and tests keep working with the lumped
    # wheels-and-axles entry above.
    #
    # SPLIT WHEELS. Measured 2026-08-03: front 5 g, rear 6 g, both sides
    # combined. They differ AND they sit a wheelbase apart, so a single mass at
    # the axle midpoint puts the wheel COM (6/11 - 1/2)*W = 5.5 mm too far
    # forward at W=120. When these are supplied they REPLACE the lumped entry.
    wheels_front_mass_kg: Optional[float] = None
    wheels_front_com: Optional[tuple[float, float, float]] = None
    wheels_rear_mass_kg: Optional[float] = None
    wheels_rear_com: Optional[tuple[float, float, float]] = None
    # HALO. Measured 2026-08-03 at 3 g, and it had NO MASS AT ALL here before:
    # the fixed components were cartridge, rear wing and wheels only. The halo
    # is modelled as a void that forces phi > 0, so its geometry was respected
    # and its weight was not -- while Stage 1's proxy path carried an 8 g stub
    # for it, so the two stages disagreed about the mass of the same car.
    halo_mass_kg: Optional[float] = None
    halo_com: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if abs(self.co2_cartridge_mass_kg - CO2_CARTRIDGE_MASS_KG) > 1e-9:
            raise ValueError("co2_cartridge_mass_kg must equal 0.023 kg")


# Mass of a full CO2 charge, kg. Reported alongside (never inside) the totals,
# so launch-condition COM can be computed without re-introducing the
# double-count. The authoritative value at run time is the thrust CSV's own
# `sheet_mass(0) - mass_sheet_final` (7.87 g for co2_thrust_data.csv); this is
# the nominal 8 g charge used when no model is on hand.
CO2_PROPELLANT_MASS_KG = 0.008


def ingest_mass_com(
    machined_components: list[ComponentMassCOM],
    fixed_hardware: FixedHardwareSpec,
    propellant_mass_kg: float = CO2_PROPELLANT_MASS_KG,
) -> FullCarMassCOM:
    """
    Combine Part 1's machined-component mass/COM report with fixed hardware.

    Units: all inputs/outputs in kg and m per physics_contract.py convention.

    total_mass = sum of all component masses (machined + fixed)
    COM_total  = sum(mass_i * COM_i) / sum(mass_i), computed independently
    per x, y, z.

    h_com = COM_total.z  (height above track)
    x_com = COM_total.x  (fore-aft position from front axle -- caller is
    responsible for ensuring the x=0 origin used by Part 1's phi grids is the
    front-axle reference; this function does not re-origin anything)

    Raises ValueError if:
      - machined_components is empty
      - any component has mass_kg <= 0
      - the CO2 cartridge is not present in fixed_hardware (this is enforced
        structurally since fixed_hardware.co2_cartridge_mass_kg is required)
    """
    if not machined_components:
        raise ValueError("machined_components must not be empty")

    fixed_components = [
        ComponentMassCOM(
            name="co2_cartridge",
            mass_kg=fixed_hardware.co2_cartridge_mass_kg,
            com_x_m=fixed_hardware.co2_cartridge_com[0],
            com_y_m=fixed_hardware.co2_cartridge_com[1],
            com_z_m=fixed_hardware.co2_cartridge_com[2],
        ),
        ComponentMassCOM(
            name="rear_wing",
            mass_kg=fixed_hardware.rear_wing_mass_kg,
            com_x_m=fixed_hardware.rear_wing_com[0],
            com_y_m=fixed_hardware.rear_wing_com[1],
            com_z_m=fixed_hardware.rear_wing_com[2],
        ),
        ComponentMassCOM(
            name="wheels_axles",
            mass_kg=fixed_hardware.wheels_axles_mass_kg,
            com_x_m=fixed_hardware.wheels_axles_com[0],
            com_y_m=fixed_hardware.wheels_axles_com[1],
            com_z_m=fixed_hardware.wheels_axles_com[2],
        ),
    ]
    # Split wheels replace the lumped entry when supplied; the halo is added
    # when supplied. Both default to absent so older callers are unaffected.
    if (fixed_hardware.wheels_front_mass_kg is not None
            and fixed_hardware.wheels_rear_mass_kg is not None):
        fixed_components = [c for c in fixed_components if c.name != "wheels_axles"]
        fixed_components.append(ComponentMassCOM(
            name="wheels_front",
            mass_kg=fixed_hardware.wheels_front_mass_kg,
            com_x_m=fixed_hardware.wheels_front_com[0],
            com_y_m=fixed_hardware.wheels_front_com[1],
            com_z_m=fixed_hardware.wheels_front_com[2]))
        fixed_components.append(ComponentMassCOM(
            name="wheels_rear",
            mass_kg=fixed_hardware.wheels_rear_mass_kg,
            com_x_m=fixed_hardware.wheels_rear_com[0],
            com_y_m=fixed_hardware.wheels_rear_com[1],
            com_z_m=fixed_hardware.wheels_rear_com[2]))
    if fixed_hardware.halo_mass_kg is not None:
        fixed_components.append(ComponentMassCOM(
            name="halo",
            mass_kg=fixed_hardware.halo_mass_kg,
            com_x_m=fixed_hardware.halo_com[0],
            com_y_m=fixed_hardware.halo_com[1],
            com_z_m=fixed_hardware.halo_com[2]))

    components = tuple(machined_components) + tuple(fixed_components)

    # A machined component may legitimately reach ZERO mass -- this is a topology
    # optimiser, and deciding not to use a region is a valid answer. It raised
    # instead, and that killed a run: with only the mandatory cargo block pinned
    # as hard-solid, nothing stops the mass term carving a small component away,
    # and 'nose' hit zero on the 4th iteration of a mocked sweep (151 g -> 102 g
    # -> 85 g, race time 2.826 s -> 1.979 s -> 1.697 s, then this raise). The
    # warm start then carried the dead nose to the next d_halo, which failed on
    # ITS first iteration too.
    #
    # Whether a car with no nose is LEGAL is a rules question, and the rule
    # checker's job -- not the mass rollup's. It contributes zero to a
    # mass-weighted mean, so it is simply excluded, with a warning because a
    # vanished component is much more often a signal than an intent.
    #
    # NEGATIVE mass still raises. That cannot come from a volume integral and
    # means something upstream is broken.
    for component in components:
        if component.mass_kg < 0:
            raise ValueError(
                f"component {component.name!r} has NEGATIVE mass "
                f"({component.mass_kg}); a volume integral cannot produce this")
    empty = [c.name for c in components if c.mass_kg == 0]
    if empty:
        warnings.warn(
            f"machined component(s) {empty} have been carved to zero mass and "
            f"are excluded from the mass/COM rollup. Only the mandatory cargo "
            f"block is pinned as hard-solid, so nothing stops the mass term "
            f"removing a component entirely -- check the rule gates still "
            f"reject the result if a car without them is illegal.",
            RuntimeWarning, stacklevel=2)
        components = tuple(c for c in components if c.mass_kg > 0)
    if not components:
        raise ValueError(
            "every component has zero mass: the optimiser has removed the "
            "entire car, which no rollup can describe")

    total_mass = sum(component.mass_kg for component in components)
    com_x = sum(component.mass_kg * component.com_x_m for component in components) / total_mass
    com_y = sum(component.mass_kg * component.com_y_m for component in components) / total_mass
    com_z = sum(component.mass_kg * component.com_z_m for component in components) / total_mass

    # Sanity check: COM coordinates should be within a physically reasonable range.
    # A COM at 10m for a ~150mm car indicates a units bug (mm vs m) or origin error.
    for name, value in (("com_x_m", com_x), ("com_y_m", com_y), ("com_z_m", com_z)):
        if not (COM_SANITY_BOUNDS_M[0] - COM_SANITY_TOL_M
                <= value <=
                COM_SANITY_BOUNDS_M[1] + COM_SANITY_TOL_M):
            raise ValueError(
                f"{name}={value} is outside sanity bounds {COM_SANITY_BOUNDS_M}; "
                f"check for a units (mm vs m) or coordinate-origin bug upstream"
            )

    return FullCarMassCOM(
        total_mass_kg=total_mass,
        com_x_m=com_x,
        com_y_m=com_y,
        com_z_m=com_z,
        components=components,
        # Propellant rides at the cartridge's own COM and is NOT in the totals
        # above -- see FullCarMassCOM. Zero unless the caller supplies a charge.
        propellant_mass_kg=propellant_mass_kg,
        propellant_com=tuple(fixed_hardware.co2_cartridge_com),
    )
