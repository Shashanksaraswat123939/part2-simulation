"""
openfoam_case.py — real half-car external-aero case generation, execution, and
output parsing for ESI OpenFOAM (openfoam.com, v2206+).

Why ESI (.com) and not the Foundation (.org) build: the project's adjoint shape
sensitivity (SPEC.txt §15, Part 2 "Adjoint Objective Contract") is produced by
`adjointOptimisationFoam`, which ships only in the ESI line. Targeting ESI here
means the same case skeleton (0/, constant/, system/, symmetryPlane on the
centreline) can later drive the adjoint solve without a second setup.

Design:
  * Everything is generated from Python (no static dict files to keep in sync).
    Geometry-dependent quantities (domain box, locationInMesh, turbulence inlet
    values) are computed from the STL bounding box and the run config, so a
    single function produces a self-consistent case.
  * The three phases the spec calls for — snappyHexMesh, steady simpleFoam,
    force extraction (SPEC "CFD Setup") — are run as subprocesses that source
    the ESI bashrc. Nothing here talks to AWS or Firebase; runs are local.
  * The pure functions (dict builders, bbox, frontal area, log/force parsers)
    are unit-tested in tests/test_openfoam_case.py. The subprocess orchestration
    (`invoke`) is exercised on a machine that actually has ESI OpenFOAM.

Coordinate convention (matches physics_contract / Part 1):
    x = front→rear (freestream +x), y = centreline→outside, z = track→up.
    Half-car domain keeps y >= 0 with a symmetryPlane at y = 0.

Contract: `invoke()` returns exactly the dict keys cfd_wrapper expects:
    D20_half, L_half, A_half, pitching_moment_half,
    residual_final, negative_volume_cells, y_plus_min, y_plus_max, courant_max.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Air properties at ~15 °C. nu = mu / rho; kept explicit so a caller varying
# air_density also gets a matching viscosity if they compute it.
_MU_AIR_PA_S: float = 1.813e-5


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

# snappyHexMesh surface refinement (min, max) levels per resolution label.
# Used by mesh_validation.run_mesh_independence_study, which drives the same
# STL at three resolutions and checks the force spread.
RESOLUTION_REFINEMENT: dict[str, tuple[int, int]] = {
    "coarse": (2, 3),
    "medium": (3, 4),
    "fine": (4, 5),
}


@dataclass(frozen=True)
class OpenFOAMRunConfig:
    """All knobs for one half-car solve. Defaults match the spec reference
    condition (20 m/s, 1.225 kg/m^3, laminar baseline)."""

    reference_speed_mps: float = 20.0
    air_density_kgm3: float = 1.225
    kinematic_viscosity_m2s: float = _MU_AIR_PA_S / 1.225
    # kOmegaSST is the default (audit fix 2026-07-24). The old "laminar"
    # default was not viable: Re = U*L/nu = 20*0.233/1.48e-5 = 3.15e5, where a
    # laminar steady solve does not settle — residuals stall around 1e-2..1e-3,
    # and Part 3's require_cfd_convergence gate (residual <= CONVERGENCE_RESIDUAL,
    # 5e-3 -- 1e-3 was below the measured 2.1e-3 plateau) then routes
    # every iteration to CFD_failed until the candidate dies on 3 consecutive
    # failures. kOmegaSST also matches openfoam_adjoint, which is hardwired to
    # it — so the drag VALUE and the shape GRADIENT now come from the same
    # closure instead of two different ones.
    turbulence_model: str = "kOmegaSST"
    resolution: str = "medium"
    max_iterations: int = 2000
    n_subdomains: int = 1  # 1 = serial; >1 decomposes with scotch + mpirun
    # Extra refinement levels (above the surface max level) inside
    # underbody_box, so the ride-height gap under the car is actually resolved
    # once lowerWall sits on the track. Each level halves the cell size.
    # ponytail: +1 gives ~0.6 mm cells in a ~1.6 mm gap at "medium" — about
    # 2-3 cells across, coarse but not blocked. Raise to 2 (0.30 mm, ~5 cells)
    # when you want the ground-effect number to be trustworthy rather than
    # merely present; it costs roughly 8x the cells in that box.
    underbody_refinement_level: int = 1
    # Per-stage subprocess timeout. Was hardcoded at 7200 s inside run_stages
    # with no way to change it; a big case silently burned 2 h and then died.
    stage_timeout_s: int = 7200
    # Pitching moment is reported about this point (SPEC "pitching moment about
    # car reference point"). Pinned here to close audit P2-12; the STL arrives
    # in Part 1 world coords with x=0 at the nose tip, so (0,0,0) is the nose
    # tip on the ground plane. Documented, not silently chosen.
    moment_reference_point_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # lRef/Aref only affect the *coefficient* function object; raw force and
    # moment (what physics_contract consumes) are independent of them.
    reference_length_m: float = 1.0
    # FREESTREAM TURBULENCE (changed 2026-09-25). The car runs through still
    # air, so the oncoming flow is nearly laminar. The old inlet (I = 5 %,
    # omega from a length scale equal to the car length) gave
    # nut/nu ~ 10,500 at the car -- the freestream was ~10^4 times too viscous
    # and every force was computed at an effective Re of order 30.
    # Now: I = 0.5 % and omega set from a target viscosity ratio nut/nu.
    # turbulent_viscosity_ratio=None restores the legacy length-scale formula
    # (kept only so the two can be compared in rnd/cfd_rnd.py).
    turbulence_intensity: float = 0.005
    turbulent_viscosity_ratio: Optional[float] = 5.0
    # FIXED MESHING FRAME. When set to ((x0,y0,z0),(x1,y1,z1)) in metres, the
    # domain box, background cell size, locationInMesh and underbody box are
    # derived from THESE bounds instead of from each STL's own bounding box.
    # With the old behaviour a sub-micron change to the car moved every
    # background cell, which is the likeliest source of the 1.2-3 % remesh
    # spread. None keeps the legacy per-STL behaviour.
    domain_reference_bounds: Optional[tuple] = None
    keep_run_dir: bool = False

    def __post_init__(self):
        if self.turbulence_model not in ("laminar", "kOmegaSST"):
            raise ValueError(
                f"turbulence_model must be 'laminar' or 'kOmegaSST', "
                f"got {self.turbulence_model!r}"
            )
        if self.resolution not in RESOLUTION_REFINEMENT:
            raise ValueError(
                f"resolution must be one of {sorted(RESOLUTION_REFINEMENT)}, "
                f"got {self.resolution!r}"
            )
        if self.reference_speed_mps <= 0:
            raise ValueError("reference_speed_mps must be > 0")


class OpenFOAMNotFoundError(RuntimeError):
    """Raised when no ESI OpenFOAM environment can be located. Callers in
    cfd_wrapper translate this into a CFDRunError so it maps to the
    'CFD_failed' lifecycle state rather than crashing the optimizer."""


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------

_DEFAULT_BASHRC_SEARCH_ROOTS = ("/usr/lib/openfoam", "/opt", os.path.expanduser("~"))


def find_openfoam_bashrc(
    explicit: Optional[str] = None,
    search_roots: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Locate an ESI OpenFOAM etc/bashrc.

    Order: explicit arg → $FOAM_BASHRC → derived from $WM_PROJECT_DIR →
    common install roots. Returns the path or None (never raises), so callers
    can decide whether a missing install is fatal.

    IMPORTANT for tests: an invalid `explicit` path does NOT disable the
    later fallback candidates -- it's just one candidate among several, by
    design (a caller can suggest a path that doesn't happen to exist and
    still get a real environment from $WM_PROJECT_DIR or a common install
    root). This means "pass an obviously-fake bashrc path" is NOT a
    reliable way to simulate "no OpenFOAM available" on a machine that
    actually has ESI OpenFOAM installed -- verified live, 2026-07-16: a
    test doing exactly that silently found the real local install and
    triggered a genuine multi-hour simpleFoam run instead of testing the
    absent-environment path at all. Pass `search_roots=[]` (and clear
    $FOAM_BASHRC/$WM_PROJECT_DIR) to deterministically force "not found"
    regardless of what's actually installed on the host.
    """
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env_bashrc = os.environ.get("FOAM_BASHRC")
    if env_bashrc:
        candidates.append(env_bashrc)
    wm = os.environ.get("WM_PROJECT_DIR")
    if wm:
        candidates.append(os.path.join(wm, "etc", "bashrc"))
    # Common ESI install roots (Linux packages, Docker images, module installs).
    if search_roots is None:
        search_roots = _DEFAULT_BASHRC_SEARCH_ROOTS
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            for name in sorted(os.listdir(root)):
                if name.lower().startswith("openfoam"):
                    candidates.append(os.path.join(root, name, "etc", "bashrc"))
        except OSError:
            pass
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Geometry helpers (pure, unit-tested)
# ---------------------------------------------------------------------------

def read_ascii_stl_vertices(stl_path: str) -> list[tuple[float, float, float]]:
    """Return every `vertex` line as an (x, y, z) tuple. ASCII STL only."""
    raw = Path(stl_path).read_bytes()
    if not raw.lstrip().startswith(b"solid"):
        raise ValueError("Not an ASCII STL (missing 'solid' header)")
    verts: list[tuple[float, float, float]] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise ValueError("ASCII STL contains no vertices")
    return verts


def stl_bounds(stl_path: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Axis-aligned bounding box (min_xyz, max_xyz) of the STL, in metres."""
    verts = read_ascii_stl_vertices(stl_path)
    xs, ys, zs = zip(*verts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def compute_frontal_area_half(stl_path: str) -> float:
    """Frontal (projected) area of the half-car onto the y–z plane, in m^2.

    Freestream is +x, so frontal area is the silhouette seen from upstream.
    For a closed surface the signed projected areas of front- and rear-facing
    triangles cancel, so summing only *front-facing* triangles (outward normal
    has a component pointing upstream, n_x < 0) gives the silhouette area
    exactly for a convex body and a very good approximation otherwise.

    This is the `compute_frontal_area` the audit (P2-14) noted nobody owned.
    It is deliberately geometry-derived (SPEC: "A = frontal projected area
    from geometry"), not read back out of the CFD.
    """
    verts = read_ascii_stl_vertices(stl_path)
    if len(verts) % 3 != 0:
        raise ValueError("STL vertex count is not a multiple of 3")
    area = 0.0
    for i in range(0, len(verts), 3):
        a, b, c = verts[i], verts[i + 1], verts[i + 2]
        # Full triangle normal (cross product); its x-component equals twice
        # the signed area projected onto the y–z plane.
        ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        nx = uy * vz - uz * vy  # x-component of (u × v)
        projected_signed = 0.5 * nx  # signed area on y–z plane
        if projected_signed < 0.0:  # front-facing (normal points upstream)
            area += -projected_signed
    return area


# ---------------------------------------------------------------------------
# Turbulence inlet values (pure)
# ---------------------------------------------------------------------------

def turbulence_inlet_values(cfg: OpenFOAMRunConfig, ref_length_m: float) -> tuple[float, float, float]:
    """Return (k, omega, nut) inlet values for a k-omega SST run.

    k     = 1.5 (I * U)^2
    omega = k / (r * nu)                 when turbulent_viscosity_ratio r is set
          = k^0.5 / (Cmu^0.25 * L)       legacy (r is None): L = car length
    nut   = k / omega
    """
    u = cfg.reference_speed_mps
    intensity = cfg.turbulence_intensity
    k = 1.5 * (intensity * u) ** 2
    ratio = getattr(cfg, "turbulent_viscosity_ratio", None)
    if ratio is not None:
        if ratio <= 0:
            raise ValueError("turbulent_viscosity_ratio must be > 0")
        omega = k / (ratio * cfg.kinematic_viscosity_m2s)
    else:
        length = max(ref_length_m, 1e-6)
        omega = math.sqrt(k) / (0.09 ** 0.25 * length)
    nut = k / omega if omega > 0 else 0.0
    return k, omega, nut


def meshing_bounds(stl_bounds_: tuple, cfg) -> tuple:
    """Bounds that drive the domain, background mesh and locationInMesh.

    The STL's own bounds unless cfg.domain_reference_bounds is set, in which
    case the fixed frame is used and the STL must lie inside it.
    """
    ref = getattr(cfg, "domain_reference_bounds", None)
    if ref is None:
        return stl_bounds_
    (x0, y0, z0), (x1, y1, z1) = stl_bounds_
    (rx0, ry0, rz0), (rx1, ry1, rz1) = ref
    tol = 1e-6
    if x0 < rx0 - tol or y0 < ry0 - tol or z0 < rz0 - tol \
            or x1 > rx1 + tol or y1 > ry1 + tol or z1 > rz1 + tol:
        raise ValueError(
            f"STL bounds {stl_bounds_} fall outside domain_reference_bounds {ref}")
    return (tuple(ref[0]), tuple(ref[1]))


# ---------------------------------------------------------------------------
# Dictionary builders (pure string generators, unit-tested)
# ---------------------------------------------------------------------------

_FOAM_HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
| Generated by openfoam_case.py — half-car external aero (ESI OpenFOAM)      |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _header(cls: str, obj: str) -> str:
    return _FOAM_HEADER.format(cls=cls, obj=obj)


def domain_box(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Wind-tunnel box around the car. Upstream 3L, downstream 8L, 5·(h,w)
    lateral/vertical margins — standard external-aero clearances.

    y_min is clamped to 0 so the symmetry plane sits exactly on the centreline.
    z_min is clamped to 0 — the TRACK SURFACE — so `lowerWall` is the track the
    car actually runs on.

    Audit fix (2026-07-24): z_min was `z0 - 4*lz`, which for a real car
    (min z = 1.6 mm, lz = 63 mm) put the lower wall 252 mm BELOW the track. The
    car was simulated flying in free air a quarter of a metre off the ground,
    so ground effect — a first-order term for a 1.5 mm-ride-height dragster
    (T3.7) — was entirely absent from D20 and L. The gap now has to be meshed;
    see `underbody_refinement_level` in OpenFOAMRunConfig.
    """
    (x0, y0, z0), (x1, y1, z1) = bounds
    lx = max(x1 - x0, 1e-6)
    ly = max(y1 - y0, 1e-6)
    lz = max(z1 - z0, 1e-6)
    box_min = (x0 - 3.0 * lx, 0.0, 0.0)
    box_max = (x1 + 8.0 * lx, y1 + 5.0 * ly, z1 + 5.0 * lz)
    return box_min, box_max


def underbody_box(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Refinement box spanning the ride-height gap between track and car floor.

    Now that `lowerWall` sits on the track (see domain_box), the ~1.5 mm gap
    under the car has to be resolved or the ground-effect flow is simply
    blocked by cells too big to fit in it — which would be worse physics than
    the free-air case it replaces. This box covers the whole planform plus a
    small margin, from the track up to just above the car's lowest point.
    """
    (x0, y0, z0), (x1, y1, z1) = bounds
    margin = 0.05 * max(x1 - x0, 1e-6)
    # Ride height is z0 (car floor above the track). Cover the gap and as much
    # again above it, with a 5 mm floor so the box is never degenerate.
    z_top = max(2.0 * z0, 0.005)
    return ((x0 - margin, 0.0, -margin), (x1 + margin, y1 + margin, z_top))


def location_in_mesh(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> tuple[float, float, float]:
    """A point guaranteed to be in the fluid (inside the box, outside the car):
    upstream of the car, above the ground, just off the symmetry plane.

    The multipliers are deliberately NOT round numbers, and that is the whole
    point. This geometry makes the background mesh perfectly self-similar:

        domain width = 12*lx           (domain_box: -3lx .. +8lx around lx)
        cell size    = lx/12           (build_case)
        => nx        = 144  ALWAYS, for any car length

    With the old `x0 - 1.0*lx`, the point landed at exactly 2lx/12lx = 1/6 of
    the domain, i.e. cell index 144/6 = 24.0 -- EXACTLY on a cell face. Whether
    snappyHexMesh's findCell() then succeeds is decided by floating-point
    rounding, so it worked on one geometry and failed on the next:

        --> FOAM FATAL ERROR
        Point (-0.23149355 0.026552572 0.033000793) is not inside the mesh
        or on a face or edge.
        Bounding box of the mesh: (-0.69524729 0 0) (2.0872751 0.21162078 ...)

    -- with the point plainly inside that bounding box. Iteration 1 meshed
    fine; iteration 2, after the phi update moved the surface by 0.3% in
    volume, flipped the rounding and snappy refused. Found 2026-07-27.

    Nudging the multipliers off round numbers is NOT enough on its own: nx is
    pinned at 144 but ny and nz follow the car's aspect ratio, so a fraction
    that is safely interior in x can still land on a face in y for a different
    car (caught by the test at lx=300 mm, y index 0.9866). So the point is
    snapped to the CENTRE of whichever background cell contains it -- the
    furthest it can possibly be from every face, for any geometry.
    """
    (x0, y0, z0), (x1, y1, z1) = bounds
    lx = max(x1 - x0, 1e-6)

    # Where we want it: upstream of the car, off the symmetry plane, mid-height.
    want = (
        x0 - 1.2731 * lx,
        0.5 * (y0 + y1) + 0.2371 * (y1 - y0) + 1e-4,
        0.4871 * (z0 + z1),
    )

    # Snap to the containing cell's centre. Must mirror build_case's cell_size
    # and build_blockmesh_dict's rounding exactly, or the snap targets a grid
    # that does not exist.
    box_min, box_max = domain_box(bounds)
    nominal = max(lx / 12.0, 1e-4)
    out = []
    for ax in range(3):
        lo, hi = box_min[ax], box_max[ax]
        n = max(int(round((hi - lo) / nominal)), 1)
        cell = (hi - lo) / n
        idx = min(max(int((want[ax] - lo) / cell), 0), n - 1)
        out.append(lo + (idx + 0.5) * cell)
    return tuple(out)


def build_blockmesh_dict(
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
    cell_size_m: float,
) -> str:
    (xmin, ymin, zmin) = box_min
    (xmax, ymax, zmax) = box_max
    nx = max(int(round((xmax - xmin) / cell_size_m)), 1)
    ny = max(int(round((ymax - ymin) / cell_size_m)), 1)
    nz = max(int(round((zmax - zmin) / cell_size_m)), 1)
    return _header("dictionary", "blockMeshDict") + f"""
scale   1;

vertices
(
    ({xmin} {ymin} {zmin})
    ({xmax} {ymin} {zmin})
    ({xmax} {ymax} {zmin})
    ({xmin} {ymax} {zmin})
    ({xmin} {ymin} {zmax})
    ({xmax} {ymin} {zmax})
    ({xmax} {ymax} {zmax})
    ({xmin} {ymax} {zmax})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet     {{ type patch;         faces ((0 4 7 3)); }}
    outlet    {{ type patch;         faces ((1 2 6 5)); }}
    symmetry  {{ type symmetryPlane; faces ((0 1 5 4)); }}
    outer     {{ type patch;         faces ((3 7 6 2)); }}
    lowerWall {{ type wall;          faces ((0 3 2 1)); }}
    upperWall {{ type patch;         faces ((4 5 6 7)); }}
);

mergePatchPairs ();
"""


def build_snappy_dict(
    car_stl_name: str,
    loc: tuple[float, float, float],
    refinement: tuple[int, int],
    add_layers: bool,
    underbody: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = None,
    underbody_extra_levels: int = 1,
) -> str:
    lo, hi = refinement
    underbody_geometry = ""
    underbody_region = ""
    if underbody is not None and underbody_extra_levels > 0:
        (ux0, uy0, uz0), (ux1, uy1, uz1) = underbody
        underbody_geometry = f"""
    underbody
    {{
        type searchableBox;
        min  ({ux0} {uy0} {uz0});
        max  ({ux1} {uy1} {uz1});
    }}
"""
        underbody_region = (
            f"underbody {{ mode inside; levels ((1e15 {hi + underbody_extra_levels})); }}"
        )
    layers_block = ""
    if add_layers:
        layers_block = f"""
    layers
    {{
        "car.*"
        {{
            nSurfaceLayers 3;
        }}
    }}
    relativeSizes true;
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.05;
    nGrow 0;
    featureAngle 120;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
"""
    return _header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       {str(add_layers).lower()};

geometry
{{
    car
    {{
        type triSurfaceMesh;
        file "{car_stl_name}";
    }}
{underbody_geometry}}}

castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    maxLoadUnbalance 0.10;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;

    features
    (
        {{ file "car.eMesh"; level {hi}; }}
    );

    refinementSurfaces
    {{
        car
        {{
            level ({lo} {hi});
            patchInfo {{ type wall; }}
        }}
    }}

    refinementRegions {{ {underbody_region} }}

    locationInMesh ({loc[0]} {loc[1]} {loc[2]});
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{{layers_block}
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""


def build_control_dict(cfg: OpenFOAMRunConfig, ref_area_half: float) -> str:
    """controlDict with the force + forceCoeffs + yPlus function objects the
    spec's Extract step needs (D20, L, pitching moment, y+)."""
    (cx, cy, cz) = cfg.moment_reference_point_m
    rho = cfg.air_density_kgm3
    u = cfg.reference_speed_mps
    aref = max(ref_area_half, 1e-9)
    lref = cfg.reference_length_m
    return _header("dictionary", "controlDict") + f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {cfg.max_iterations};
deltaT          1;
// Write the field set ONCE, at the end: writeInterval == endTime means the
// only write is the final step. (`writeControl onEnd` is NOT valid here --
// onEnd belongs to the FUNCTION OBJECT writeControl enum, not
// Time::writeControls, which accepts only timeStep/runTime/adjustableRunTime/
// cpuTime/clockTime. Setting it made v2412 abort with a Foam::Enum readEntry
// error before meshing even started; caught on the first real solve.)
// The old every-500 setting dumped U/p/phi/k/omega/nut for a ~1M-cell case
// four times per solve in ASCII at 8 digits -- multi-GB per iteration, for
// data nothing reads. The forces/yPlus function objects below still write
// every step; they are what invoke() actually parses.
writeControl    timeStep;
writeInterval   {cfg.max_iterations};
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (car);
        rho             rhoInf;
        rhoInf          {rho};
        CofR            ({cx} {cy} {cz});
        log             false;
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (car);
        rho             rhoInf;
        rhoInf          {rho};
        liftDir         (0 0 1);
        dragDir         (1 0 0);
        pitchAxis       (0 1 0);
        CofR            ({cx} {cy} {cz});
        magUInf         {u};
        lRef            {lref};
        Aref            {aref};
        log             false;
    }}

    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        log             true;
    }}
}}
"""


def build_transport_properties(cfg: OpenFOAMRunConfig) -> str:
    return _header("dictionary", "transportProperties") + f"""
transportModel  Newtonian;
nu              {cfg.kinematic_viscosity_m2s};
"""


def build_turbulence_properties(cfg: OpenFOAMRunConfig) -> str:
    if cfg.turbulence_model == "laminar":
        return _header("dictionary", "turbulenceProperties") + """
simulationType  laminar;
"""
    return _header("dictionary", "turbulenceProperties") + """
simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""


# ---------------------------------------------------------------------------
# Output parsers (pure, unit-tested against fixtures)
# ---------------------------------------------------------------------------

def parse_negative_volume_cells(checkmesh_log: str) -> int:
    """checkMesh prints e.g. '  ***Error in ... : 12 negative volume cells' or
    '  Min volume = ... . Negative volume cells: 0'. Return the count; if the
    reassuring 'Mesh OK' line is present and no explicit count, return 0."""
    m = re.search(r"(\d+)\s+negative\s+volume\s+cells", checkmesh_log, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"negative\s+volume\s+cells\s*[:=]\s*(\d+)", checkmesh_log, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def parse_final_p_residual(solver_log: str) -> float:
    """Final initial-residual of the pressure equation from a simpleFoam log.
    Lines look like: 'GAMG:  Solving for p, Initial residual = 3.1e-04, ...'.
    Returns the last such value; raises if none found (an empty/failed solve
    must not silently look converged)."""
    vals = re.findall(
        r"Solving for p,\s*Initial residual\s*=\s*([0-9.eE+-]+)", solver_log
    )
    if not vals:
        raise ValueError("No pressure residual found in solver log")
    return float(vals[-1])


def parse_max_courant(solver_log: str) -> Optional[float]:
    """Last 'Courant Number mean: ... max: X'. simpleFoam (steady) may not
    print Courant; return None in that case rather than inventing a value."""
    vals = re.findall(r"Courant Number mean:\s*[0-9.eE+-]+\s*max:\s*([0-9.eE+-]+)", solver_log)
    if not vals:
        return None
    return float(vals[-1])


def parse_yplus_range(yplus_text: str, patch: str = "car") -> tuple[float, float]:
    """Parse the y+ min/max for `patch` (default "car", i.e. the vehicle
    surface — deliberately excludes lowerWall/other wall patches, which the
    yPlus function object also reports and which are not what the CFD health
    report's y_plus_min/max is meant to validate).

    Handles two real ESI output shapes, verified against an actual v2412
    solver run (not just assumed from documentation):
      1. Solver log line:
         "    patch car y+ : min = 17.99, max = 239.1, average = 88.5"
         (equals sign, not colon — colon-based 'min:'/'max:' does not occur
         in practice and was a wrong assumption in an earlier version of this
         parser).
      2. postProcessing/yPlus/<time>/yPlus.dat, tab-separated columns
         "Time patch min max average", one row per patch per write time.
    Returns (min, max) for the LATEST time step of the requested patch.
    """
    log_hits = re.findall(
        rf"patch\s+{re.escape(patch)}\s+y\+\s*:\s*min\s*=\s*([0-9.eE+-]+),?\s*max\s*=\s*([0-9.eE+-]+)",
        yplus_text,
    )
    if log_hits:
        last_min, last_max = log_hits[-1]  # last occurrence = latest write
        return float(last_min), float(last_max)

    # dat-file: filter to the requested patch, take the row at the latest time.
    best: Optional[tuple[float, float, float]] = None  # (time, min, max)
    for line in yplus_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols = s.split()
        if len(cols) < 4 or cols[1] != patch:
            continue
        try:
            t, mn, mx = float(cols[0]), float(cols[2]), float(cols[3])
        except ValueError:
            continue
        if best is None or t >= best[0]:
            best = (t, mn, mx)
    if best is not None:
        return best[1], best[2]
    raise ValueError(f"Could not parse y+ range for patch {patch!r}")


def _last_data_row(dat_text: str) -> list[str]:
    last = None
    for line in dat_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        last = s
    if last is None:
        raise ValueError("No data rows in force file")
    # ESI wraps force vectors in parentheses: strip them to get flat floats.
    return last.replace("(", " ").replace(")", " ").split()


# Fraction of the force history averaged to produce the reported force, and
# over which the oscillation is measured. 20% of a 1000-iteration solve is 200
# samples — long enough to average several shedding periods, short enough to
# exclude the initial transient.
FORCE_AVERAGE_FRACTION: float = 0.2


def _force_history(dat_text: str) -> list[list[float]]:
    """All parsable data rows of a force/moment .dat as flat float lists."""
    rows: list[list[float]] = []
    for line in dat_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("(", " ").replace(")", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return rows


def parse_total_vector_dat(
    dat_text: str, average_fraction: float = FORCE_AVERAGE_FRACTION
) -> tuple[float, float, float]:
    """Parse the *total* 3-vector from an ESI v2206+ `force.dat` or `moment.dat`.

    Layout (after stripping parentheses):
        time  (total_x total_y total_z) (pressure_x..) (viscous_x..)
    The total already includes pressure + viscous (+ porous), so we take the
    first vector directly — no summation, which avoids the version-dependent
    ambiguity of the older combined `forces.dat` column order.

    Returns the MEAN over the last `average_fraction` of the history, not the
    final row.

    WHY (measured 2026-07-27). This car is a brick, and a brick at Re 3.15e5
    sheds vortices. simpleFoam is a STEADY solver, so it cannot converge on a
    genuinely unsteady wake: the force history oscillates and never settles.
    Measured across four real solves, the drag over the last 20% of iterations
    had a peak-to-peak spread of 18.1%, 20.2%, 22.1% and 26.9% of its own mean,
    still drifting 2-4% per decade of iterations.

    Reading the final row therefore sampled that oscillation at an arbitrary
    phase. Two geometries differing by 65 NANOMETRES (418 of 360,000 vertices
    snapped onto the symmetry plane) reported drag 7.7% apart — not a physical
    difference, just two different points on the same oscillation. That noise
    is larger than the 5.43% drag reduction the first working adjoint step
    produced, which is why that step must not be read as a measured improvement.

    Averaging does not make the solve steady, and it is NOT SUFFICIENT on its
    own. Measured on that same 65-nm-apart pair, sweeping the window:

        final row   7.39% disagreement
        last  5%    1.20%
        last 10%    2.10%
        last 20%    2.75%   <- FORCE_AVERAGE_FRACTION
        last 40%    1.54%
        last 60%    0.70%
        last 80%    5.67%

    So averaging cuts the noise from ~7.4% to ~1-3%, but never reliably reaches
    the <1% that ranking candidates by race time needs, and the answer depends
    on the window. That non-monotonicity is the diagnosis: the solve is still
    DRIFTING (2-4% per decade of iterations), not oscillating about a converged
    mean, so each window averages a different part of a moving signal. 20% is
    kept because no window is defensibly better -- picking 60% because it scored
    best on one pair would be fitting noise.

    What would actually fix it: many more iterations so the transient is well
    past and many shedding cycles are averaged; or a less bluff geometry; or an
    unsteady solver with proper time-averaging. `force_oscillation_fraction`
    reports what averaging is hiding, so the caller can see when the number is
    not trustworthy rather than inferring it from a residual that cannot tell.
    """
    rows = _force_history(dat_text)
    if not rows:
        raise ValueError("force/moment .dat contained no parsable data rows")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    if width < 4:
        raise ValueError(f"Unexpected .dat width: {width} columns")
    if not 0.0 < average_fraction <= 1.0:
        raise ValueError("average_fraction must be in (0, 1]")
    k = max(int(len(rows) * average_fraction), 1)
    tail = rows[-k:]
    return tuple(sum(r[i] for r in tail) / len(tail) for i in (1, 2, 3))


def force_mean_convergence(
    dat_text: str, average_fraction: float = FORCE_AVERAGE_FRACTION
) -> tuple[float, float]:
    """Error bar on the MEAN streamwise force, and how much it is still drifting.

    Returns (stderr_fraction, drift_fraction), both as fractions of the mean.

    Why this exists alongside force_oscillation_fraction: peak-to-peak answers
    "how unsteady is the signal", but the number the optimiser ranks on is the
    MEAN over the window, and those are different questions. Peak-to-peak is
    max-minus-min, so one outlier sets it and it does NOT shrink as you average
    longer. The uncertainty of a mean does shrink, which is why a solve can
    swing 8.9% peak-to-peak and still deliver a mean good to a fraction of a
    percent -- or not, and peak-to-peak cannot tell you which.

      stderr_fraction: standard error of the mean, corrected for
        autocorrelation. Consecutive SIMPLE iterations are highly correlated,
        so the naive std/sqrt(N) understates the error by the square root of
        the integrated autocorrelation time; N_eff = N / (1 + 2*sum rho_k),
        summed until rho_k first goes non-positive (Sokal's window). THIS is
        the error bar on D20: a drag delta smaller than it is not measurable.

      drift_fraction: |slope| * window_length / |mean|, from a least-squares
        line through the window. This is the one that matters most here.
        parse_total_vector_dat's own measurements found the force still moving
        2-4% per decade of iterations -- the solve is DRIFTING, not oscillating
        about a settled value. A drifting mean is not a converged mean, and
        averaging cannot fix it: each window averages a different part of a
        moving signal, which is exactly why that docstring's window sweep is
        non-monotonic (1.20%, 2.10%, 2.75%, 1.54%, 0.70%, 5.67%). When drift
        dominates, the fix is more iterations, not a longer window.
    """
    rows = _force_history(dat_text)
    if not rows:
        raise ValueError("force/moment .dat contained no parsable data rows")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    if not 0.0 < average_fraction <= 1.0:
        raise ValueError("average_fraction must be in (0, 1]")
    k = max(int(len(rows) * average_fraction), 1)
    fx = [r[1] for r in rows[-k:]]
    n = len(fx)
    mean = sum(fx) / n
    if mean == 0.0:
        return float("inf"), float("inf")
    if n < 3:
        # Too few samples to say anything about either statistic. inf rather
        # than 0.0: "unknown" must not read as "perfectly converged".
        return float("inf"), float("inf")

    dev = [v - mean for v in fx]
    var = sum(d * d for d in dev) / (n - 1)
    if var <= 0.0:
        return 0.0, 0.0

    # Integrated autocorrelation time, Sokal's automatic window.
    tau = 1.0
    for lag in range(1, n // 2):
        rho = sum(dev[i] * dev[i + lag] for i in range(n - lag)) / ((n - lag) * var)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    n_eff = max(1.0, n / tau)
    stderr_fraction = (var ** 0.5 / n_eff ** 0.5) / abs(mean)

    # Least-squares slope over the window, in force units per sample.
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    sxx = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * d for x, d in zip(xs, dev)) / sxx if sxx else 0.0
    drift_fraction = abs(slope) * (n - 1) / abs(mean)
    return stderr_fraction, drift_fraction


def force_oscillation_fraction(
    dat_text: str, average_fraction: float = FORCE_AVERAGE_FRACTION
) -> float:
    """Peak-to-peak swing of the streamwise force over the averaging window,
    as a fraction of its mean. 0.0 means a fully settled solve.

    This is the number `residual_final` cannot express. A steady solver on an
    unsteady wake plateaus its residuals while the forces keep swinging, so
    residual convergence says nothing about whether the reported force is
    reproducible. Measured 18-27% here; a trustworthy drag delta needs this
    well below the delta being claimed.
    """
    rows = _force_history(dat_text)
    if not rows:
        raise ValueError("force/moment .dat contained no parsable data rows")
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    k = max(int(len(rows) * average_fraction), 1)
    fx = [r[1] for r in rows[-k:]]
    mean = sum(fx) / len(fx)
    if mean == 0.0:
        return float("inf")
    return (max(fx) - min(fx)) / abs(mean)


# ---------------------------------------------------------------------------
# Case assembly + orchestration
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalise_solid_name(stl_path: str, dest: Path, solid_name: str = "car") -> None:
    """Copy the STL to `dest`, forcing the ASCII 'solid <name>' header so the
    snappy patch is named deterministically ('car')."""
    raw = Path(stl_path).read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        st = line.strip()
        if st.startswith("solid"):
            out.append(f"solid {solid_name}")
        elif st.startswith("endsolid"):
            out.append(f"endsolid {solid_name}")
        else:
            out.append(line)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")


def build_case(run_dir: str, stl_path: str, cfg: OpenFOAMRunConfig) -> dict:
    """Generate a complete ESI case under run_dir for the given half-car STL.
    Returns a small metadata dict (bounds, frontal area, cell size)."""
    run = Path(run_dir)
    if run.exists():
        shutil.rmtree(run)
    (run / "system").mkdir(parents=True, exist_ok=True)
    (run / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (run / "0").mkdir(parents=True, exist_ok=True)

    bounds = meshing_bounds(stl_bounds(stl_path), cfg)
    (x0, y0, z0), (x1, y1, z1) = bounds
    lx, ly, lz = (x1 - x0), (y1 - y0), (z1 - z0)
    ref_len = max(lx, 1e-4)
    frontal_area_half = compute_frontal_area_half(stl_path)
    box_min, box_max = domain_box(bounds)
    loc = location_in_mesh(bounds)
    # Background cell ~ 1/12 of the car length keeps blockMesh modest; snappy
    # refines down from there per resolution.
    cell_size = max(ref_len / 12.0, 1e-4)

    _normalise_solid_name(stl_path, run / "constant" / "triSurface" / "car.stl")

    add_layers = cfg.turbulence_model == "kOmegaSST"
    refinement = RESOLUTION_REFINEMENT[cfg.resolution]

    _write(run / "system" / "blockMeshDict", build_blockmesh_dict(box_min, box_max, cell_size))
    _write(run / "system" / "snappyHexMeshDict",
           build_snappy_dict("car.stl", loc, refinement, add_layers,
                             underbody=underbody_box(bounds),
                             underbody_extra_levels=cfg.underbody_refinement_level))
    _write(run / "system" / "controlDict", build_control_dict(cfg, frontal_area_half))
    _write(run / "system" / "fvSchemes", _FV_SCHEMES)
    _write(run / "system" / "fvSolution", _FV_SOLUTION)
    _write(run / "system" / "surfaceFeatureExtractDict",
           _header("dictionary", "surfaceFeatureExtractDict") + _SURFACE_FEATURE_EXTRACT_BODY)
    _write(run / "system" / "meshQualityDict", _MESH_QUALITY_DICT)
    _write(run / "system" / "decomposeParDict", _decompose_dict(cfg.n_subdomains))
    _write(run / "constant" / "transportProperties", build_transport_properties(cfg))
    _write(run / "constant" / "turbulenceProperties", build_turbulence_properties(cfg))

    _write_initial_fields(run / "0", cfg, ref_len)

    return {
        "bounds": bounds,
        "frontal_area_half": frontal_area_half,
        "cell_size_m": cell_size,
        "domain_box": (box_min, box_max),
        "location_in_mesh": loc,
    }


def _write_initial_fields(zero_dir: Path, cfg: OpenFOAMRunConfig, ref_len: float) -> None:
    u = cfg.reference_speed_mps
    _write(zero_dir / "U", _header("volVectorField", "U") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u} 0 0);
boundaryField
{{
    inlet       {{ type fixedValue; value uniform ({u} 0 0); }}
    outlet      {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({u} 0 0); }}
    symmetry    {{ type symmetryPlane; }}
    outer       {{ type slip; }}
    upperWall   {{ type slip; }}
    lowerWall   {{ type fixedValue; value uniform ({u} 0 0); }}
    car         {{ type noSlip; }}
}}
""")
    # lowerWall is a ROLLING ROAD, not a stationary wall. In the car's frame the
    # track moves backwards under it at the reference speed, so the ground must
    # carry U, not zero. A no-slip stationary floor would grow a ~0.7 m boundary
    # layer over the 3L upstream run and arrive at the car with the wrong
    # velocity profile — a large error precisely in the ride-height gap that
    # domain_box's track-level lowerWall now exists to capture.
    _write(zero_dir / "p", _header("volScalarField", "p") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    symmetry    { type symmetryPlane; }
    outer       { type zeroGradient; }
    upperWall   { type zeroGradient; }
    lowerWall   { type zeroGradient; }
    car         { type zeroGradient; }
}
""")
    if cfg.turbulence_model != "kOmegaSST":
        return
    k, omega, nut = turbulence_inlet_values(cfg, ref_len)
    _write(zero_dir / "k", _header("volScalarField", "k") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {k}; }}
    outlet      {{ type inletOutlet; inletValue uniform {k}; value uniform {k}; }}
    symmetry    {{ type symmetryPlane; }}
    outer       {{ type slip; }}
    upperWall   {{ type slip; }}
    lowerWall   {{ type kqRWallFunction; value uniform {k}; }}
    car         {{ type kqRWallFunction; value uniform {k}; }}
}}
""")
    _write(zero_dir / "omega", _header("volScalarField", "omega") + f"""
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {omega}; }}
    outlet      {{ type inletOutlet; inletValue uniform {omega}; value uniform {omega}; }}
    symmetry    {{ type symmetryPlane; }}
    outer       {{ type slip; }}
    upperWall   {{ type slip; }}
    lowerWall   {{ type omegaWallFunction; value uniform {omega}; }}
    car         {{ type omegaWallFunction; value uniform {omega}; }}
}}
""")
    _write(zero_dir / "nut", _header("volScalarField", "nut") + f"""
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut};
boundaryField
{{
    inlet       {{ type calculated; value uniform {nut}; }}
    outlet      {{ type calculated; value uniform {nut}; }}
    symmetry    {{ type symmetryPlane; }}
    outer       {{ type calculated; value uniform {nut}; }}
    upperWall   {{ type calculated; value uniform {nut}; }}
    lowerWall   {{ type nutkWallFunction; value uniform 0; }}
    car         {{ type nutkWallFunction; value uniform 0; }}
}}
""")


def _decompose_dict(n: int) -> str:
    return _header("dictionary", "decomposeParDict") + f"""
numberOfSubdomains {n};
method          scotch;
"""


def _run(cmd: str, cwd: Path, bashrc: str, log_name: str, timeout: int) -> str:
    """Run one OpenFOAM command in a bash shell that sources the ESI bashrc.
    Writes combined stdout/stderr to logs/<log_name> and returns the text.
    Raises subprocess.CalledProcessError on non-zero exit."""
    (cwd / "logs").mkdir(exist_ok=True)
    full = f"source '{bashrc}' && {cmd}"
    proc = subprocess.run(
        ["bash", "-lc", full],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    (cwd / "logs" / log_name).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout


def run_stages(run_dir: str, cfg: OpenFOAMRunConfig, bashrc: str,
               timeout_s: Optional[int] = None) -> dict:
    """Execute mesh + solve stages in order and return parsed logs.

    Stages (SPEC "Solver path"): surfaceFeatureExtract → blockMesh →
    snappyHexMesh → checkMesh → simpleFoam. Serial unless cfg.n_subdomains > 1.
    """
    timeout_s = cfg.stage_timeout_s if timeout_s is None else timeout_s
    run = Path(run_dir)
    solver = "simpleFoam"
    _run("surfaceFeatureExtract", run, bashrc, "surfaceFeatureExtract.log", timeout_s)
    _run("blockMesh", run, bashrc, "blockMesh.log", timeout_s)
    if cfg.n_subdomains > 1:
        # MESH SERIALLY, THEN DECOMPOSE. The obvious ordering -- decomposePar,
        # then snappyHexMesh -parallel -- is what the tutorials do, but it only
        # works with a `restore0Dir -processor` step in between, and without it
        # the solve dies with:
        #     Cannot find patchField entry for car
        #     file: processor0/0/p/boundaryField
        # Reason: decomposePar splits the 0/ fields against the BLOCKMESH, which
        # has no `car` patch -- snappy is what creates it. The decomposed field
        # files are written before `car` exists and never gain an entry for it.
        # Caught on the first real parallel solve, 2026-07-26; checkMesh had
        # already said "Mesh OK, 103395 cells", so the mesh was never the issue.
        #
        # Meshing serially and decomposing the FINISHED mesh sidesteps the whole
        # class of bug: decomposePar then splits a mesh that already has `car`
        # and 0/ fields that already match it. Costs a little wall-clock
        # (measured 13.7 s on 4 ranks, so tens of seconds serially) against a
        # solve measured in tens of minutes -- a trade worth taking for a stage
        # that is not the bottleneck.
        _run("snappyHexMesh -overwrite", run, bashrc, "snappyHexMesh.log", timeout_s)
        checkmesh = _run("checkMesh", run, bashrc, "checkMesh.log", timeout_s)
        _run("decomposePar -force", run, bashrc, "decomposePar.log", timeout_s)
        solver_log = _run(f"mpirun -np {cfg.n_subdomains} {solver} -parallel",
                          run, bashrc, "solver.log", timeout_s)
        _run("reconstructPar -latestTime", run, bashrc, "reconstructPar.log", timeout_s)
    else:
        _run("snappyHexMesh -overwrite", run, bashrc, "snappyHexMesh.log", timeout_s)
        checkmesh = _run("checkMesh", run, bashrc, "checkMesh.log", timeout_s)
        solver_log = _run(solver, run, bashrc, "solver.log", timeout_s)
    return {"checkmesh_log": checkmesh, "solver_log": solver_log}


def _find_latest(root: Path, names: tuple[str, ...]) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob("*.dat"), key=lambda p: p.stat().st_mtime)
    for name in names:
        for c in reversed(candidates):
            if c.name == name:
                return c
    return None


def read_force_oscillation(run_dir: str) -> Optional[float]:
    """Streamwise-force peak-to-peak over the averaging window, as a fraction
    of its mean. None if no force history is available."""
    root = Path(run_dir) / "postProcessing" / "forces"
    if not root.is_dir():
        return None
    force_f = _find_latest(root, ("force.dat",)) or _find_latest(root, ("forces.dat",))
    if force_f is None:
        return None
    try:
        return force_oscillation_fraction(
            force_f.read_text(encoding="utf-8", errors="replace")
        )
    except ValueError:
        return None


def read_force_mean_convergence(run_dir: str) -> tuple[Optional[float], Optional[float]]:
    """(stderr_fraction, drift_fraction) for the streamwise force, or (None, None).

    Same file discovery as read_force_oscillation; see force_mean_convergence
    for what the two numbers mean and why peak-to-peak does not replace them.
    """
    root = Path(run_dir) / "postProcessing" / "forces"
    if not root.is_dir():
        return None, None
    force_f = _find_latest(root, ("force.dat",)) or _find_latest(root, ("forces.dat",))
    if force_f is None:
        return None, None
    try:
        return force_mean_convergence(
            force_f.read_text(encoding="utf-8", errors="replace")
        )
    except ValueError:
        return None, None


def read_force_and_moment(run_dir: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ((Fx,Fy,Fz), (Mx,My,Mz)) from postProcessing/forces.

    ESI v2206+ writes separate force.dat / moment.dat; some builds/config write
    a combined forces.dat. Prefer the separate files (unambiguous total-vector
    layout); fall back to the combined file for both.
    """
    root = Path(run_dir) / "postProcessing" / "forces"
    if not root.is_dir():
        raise FileNotFoundError("postProcessing/forces not found — solve produced no forces")
    force_f = _find_latest(root, ("force.dat",))
    moment_f = _find_latest(root, ("moment.dat",))
    if force_f and moment_f:
        f = parse_total_vector_dat(force_f.read_text(encoding="utf-8", errors="replace"))
        m = parse_total_vector_dat(moment_f.read_text(encoding="utf-8", errors="replace"))
        return f, m
    # NB: the combined-forces.dat fallback below still reads a single row. It is
    # a legacy path for builds that do not write force.dat/moment.dat; if it
    # ever becomes the live path, give it the same averaging treatment.
    combined = _find_latest(root, ("forces.dat",))
    if combined is None:
        raise FileNotFoundError("No force.dat/moment.dat/forces.dat found under postProcessing/forces")
    text = combined.read_text(encoding="utf-8", errors="replace")
    cols = _last_data_row(text)
    nums = [float(c) for c in cols[1:]]
    if len(nums) < 6:
        raise ValueError(f"Combined forces.dat too narrow: {len(nums)} columns")
    # Combined layout: time (total F)(pressure F)(viscous F) (total M)(pressure M)(viscous M).
    # Total force is the first triple; total moment is the triple after all
    # three force triples (index 9) when present, else immediately after force.
    force = (nums[0], nums[1], nums[2])
    moment = (nums[9], nums[10], nums[11]) if len(nums) >= 12 else (nums[3], nums[4], nums[5])
    return force, moment


def _read_yplus(run_dir: str, solver_log: str) -> str:
    """Prefer postProcessing/yPlus dat; fall back to the solver log text."""
    root = Path(run_dir) / "postProcessing" / "yPlus"
    if root.is_dir():
        dats = sorted(root.rglob("*.dat"), key=lambda p: p.stat().st_mtime)
        if dats:
            return dats[-1].read_text(encoding="utf-8", errors="replace")
    return solver_log


def new_run_dir_name() -> str:
    """Unique per-run directory name, shared by the primal and adjoint invokes.

    uuid4, not `hash(stl_path) % 10_000`: Part 3 runs candidates as THREADS in
    one process, so pid is shared and a 1-in-10k hash collision meant
    build_case's opening `shutil.rmtree(run)` deleted a sibling's live case.
    """
    return f"run_{os.getpid()}_{uuid.uuid4().hex[:12]}"


def invoke(stl_path: str, case_dir: str, cfg: Optional[OpenFOAMRunConfig] = None,
           bashrc: Optional[str] = None,
           search_roots: Optional[Sequence[str]] = None) -> dict:
    """Full pipeline: build case, run stages, parse, return the contract dict.

    Raises OpenFOAMNotFoundError if no ESI environment is found (cfd_wrapper
    turns that into a CFDRunError → 'CFD_failed'). All returned quantities are
    half-car (D20_half, L_half, A_half, pitching_moment_half) plus health.

    search_roots: passed through to find_openfoam_bashrc -- pass [] to force
    "not found" deterministically in tests, regardless of what's actually
    installed on the host (see find_openfoam_bashrc's docstring).
    """
    cfg = cfg or OpenFOAMRunConfig()
    resolved_bashrc = find_openfoam_bashrc(bashrc, search_roots=search_roots)
    if resolved_bashrc is None:
        raise OpenFOAMNotFoundError(
            "No ESI OpenFOAM environment found. Set $WM_PROJECT_DIR or "
            "$FOAM_BASHRC (source the ESI etc/bashrc), or pass bashrc=... . "
            "This build targets openfoam.com (ESI) so the adjoint solver is "
            "available; the Foundation (.org) build will not provide it."
        )
    run_dir = str(Path(case_dir) / "runs" / new_run_dir_name())
    meta = build_case(run_dir, stl_path, cfg)
    succeeded = False
    try:
        logs = run_stages(run_dir, cfg, resolved_bashrc)
        neg = parse_negative_volume_cells(logs["checkmesh_log"])
        residual = parse_final_p_residual(logs["solver_log"])
        courant = parse_max_courant(logs["solver_log"])
        (fx, _fy, fz), (_mx, my, _mz) = read_force_and_moment(run_dir)
        force_osc = read_force_oscillation(run_dir)
        force_se, force_drift = read_force_mean_convergence(run_dir)
        try:
            yp_min, yp_max = parse_yplus_range(_read_yplus(run_dir, logs["solver_log"]))
        except ValueError:
            yp_min, yp_max = float("nan"), float("nan")
        result = {
            # fx, not abs(fx): a reversed or diverged solve must trip the
            # D20 >= 0 guard in physics_contract instead of reading as drag.
            "D20_half": fx,
            "L_half": fz,
            "A_half": meta["frontal_area_half"],
            "pitching_moment_half": my,
            "residual_final": residual,
            "force_oscillation": force_osc,
            "force_mean_stderr": force_se,
            "force_drift": force_drift,
            "negative_volume_cells": neg,
            "y_plus_min": yp_min,
            "y_plus_max": yp_max,
            "courant_max": courant,
        }
        succeeded = True
        return result
    finally:
        # Keep the run directory on FAILURE, always. The old unconditional
        # rmtree deleted logs/ on the error path too — while CFDRunError's own
        # message told the reader to "See logs/ in the run directory". The first
        # real failure on the VM was therefore undiagnosable by construction.
        if succeeded and not cfg.keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)
        elif not succeeded:
            print(f"[openfoam_case] stage failed — run dir KEPT for diagnosis: {run_dir}")


# ---------------------------------------------------------------------------
# Static dict bodies (do not depend on geometry)
# ---------------------------------------------------------------------------

_FV_SCHEMES = _header("dictionary", "fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""

_FV_SOLUTION = _header("dictionary", "fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|omega)"     1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        "(k|omega)"     0.7;
    }
}
"""

# car.stl { ... } keyed by filename — matches the shipped ESI utility
# `surfaceFeatureExtract` / dict `surfaceFeatureExtractDict` (verified against
# openfoam2412's own tutorials; the utility is NOT named `surfaceFeatures` in
# this ESI release, despite that being the more commonly documented name).
_SURFACE_FEATURE_EXTRACT_BODY = """
car.stl
{
    extractionMethod extractFromSurface;
    writeObj        no;

    extractFromSurfaceCoeffs
    {
        includedAngle   150;
    }
}
"""

_MESH_QUALITY_DICT = _header("dictionary", "meshQualityDict") + """
#includeEtc "caseDicts/meshQualityDict"
"""
