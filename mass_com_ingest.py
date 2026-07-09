"""Stage 2 mass/COM ingestion from Part 1 outputs."""

from __future__ import annotations

from dataclasses import dataclass

from physics_contract import ComponentMassCOM, FullCarMassCOM


# Broad sanity bounds for COM coordinates (m). A STEM Racing car is ~150mm long;
# COM at 10m is clearly a units or origin bug, not a legitimate design.
COM_SANITY_BOUNDS_M = (-10.0, 10.0)

CO2_CARTRIDGE_MASS_KG = 0.023          # 23 g, fixed
CO2_CARTRIDGE_COM = None               # PLACEHOLDER: raise NotImplementedError
                                        # if actual fixed position not supplied
                                        # by caller -- see FixedHardwareSpec
REAR_WING_MASS_KG = None                # PLACEHOLDER -- must be injected by caller,
                                        # not hard-coded, since "known mass" was
                                        # never given a number in the source docs
WHEELS_AXLES_MASS_KG = None             # PLACEHOLDER -- same as above


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
        than 1e-9 kg.
    """

    co2_cartridge_mass_kg: float
    co2_cartridge_com: tuple[float, float, float]
    rear_wing_mass_kg: float
    rear_wing_com: tuple[float, float, float]
    wheels_axles_mass_kg: float
    wheels_axles_com: tuple[float, float, float]

    def __post_init__(self) -> None:
        if abs(self.co2_cartridge_mass_kg - CO2_CARTRIDGE_MASS_KG) > 1e-9:
            raise ValueError("co2_cartridge_mass_kg must equal 0.023 kg")


def ingest_mass_com(
    machined_components: list[ComponentMassCOM],
    fixed_hardware: FixedHardwareSpec,
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
    components = tuple(machined_components) + tuple(fixed_components)

    for component in components:
        if component.mass_kg <= 0:
            raise ValueError(f"component {component.name!r} has non-positive mass")

    total_mass = sum(component.mass_kg for component in components)
    com_x = sum(component.mass_kg * component.com_x_m for component in components) / total_mass
    com_y = sum(component.mass_kg * component.com_y_m for component in components) / total_mass
    com_z = sum(component.mass_kg * component.com_z_m for component in components) / total_mass

    # Sanity check: COM coordinates should be within a physically reasonable range.
    # A COM at 10m for a ~150mm car indicates a units bug (mm vs m) or origin error.
    for name, value in (("com_x_m", com_x), ("com_y_m", com_y), ("com_z_m", com_z)):
        if not (COM_SANITY_BOUNDS_M[0] <= value <= COM_SANITY_BOUNDS_M[1]):
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
    )
