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
    # "laminar" is the spec baseline (Re ~ 270k, marginal); "kOmegaSST" is the
    # validation model and the one that grows boundary layers in snappy.
    turbulence_model: str = "laminar"
    resolution: str = "medium"
    max_iterations: int = 2000
    n_subdomains: int = 1  # 1 = serial; >1 decomposes with scotch + mpirun
    # Pitching moment is reported about this point (SPEC "pitching moment about
    # car reference point"). Pinned here to close audit P2-12; the STL arrives
    # in Part 1 world coords with x=0 at the nose tip, so (0,0,0) is the nose
    # tip on the ground plane. Documented, not silently chosen.
    moment_reference_point_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # lRef/Aref only affect the *coefficient* function object; raw force and
    # moment (what physics_contract consumes) are independent of them.
    reference_length_m: float = 1.0
    turbulence_intensity: float = 0.05
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
    omega = k^0.5 / (Cmu^0.25 * L)
    nut   = k / omega   (a physical estimate; the field is calculated anyway)
    """
    u = cfg.reference_speed_mps
    intensity = cfg.turbulence_intensity
    length = max(ref_length_m, 1e-6)
    k = 1.5 * (intensity * u) ** 2
    c_mu = 0.09
    omega = math.sqrt(k) / (c_mu ** 0.25 * length)
    nut = k / omega if omega > 0 else 0.0
    return k, omega, nut


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
    lateral/vertical margins — standard external-aero clearances. y_min is
    clamped to 0 so the symmetry plane sits exactly on the centreline."""
    (x0, y0, z0), (x1, y1, z1) = bounds
    lx = max(x1 - x0, 1e-6)
    ly = max(y1 - y0, 1e-6)
    lz = max(z1 - z0, 1e-6)
    box_min = (x0 - 3.0 * lx, 0.0, z0 - 4.0 * lz)
    box_max = (x1 + 8.0 * lx, y1 + 5.0 * ly, z1 + 5.0 * lz)
    return box_min, box_max


def location_in_mesh(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> tuple[float, float, float]:
    """A point guaranteed to be in the fluid (inside the box, outside the car):
    upstream of the car, above the ground, just off the symmetry plane."""
    (x0, y0, z0), (x1, y1, z1) = bounds
    lx = max(x1 - x0, 1e-6)
    return (x0 - 1.0 * lx, 0.5 * (y0 + y1) + 0.25 * (y1 - y0) + 1e-4, 0.5 * (z0 + z1))


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
) -> str:
    lo, hi = refinement
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
}}

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

    refinementRegions {{}}

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
writeControl    timeStep;
writeInterval   {max(cfg.max_iterations // 4, 1)};
purgeWrite      2;
writeFormat     ascii;
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


def parse_total_vector_dat(dat_text: str) -> tuple[float, float, float]:
    """Parse the *total* 3-vector from an ESI v2206+ `force.dat` or `moment.dat`.

    Layout (after stripping parentheses):
        time  (total_x total_y total_z) (pressure_x..) (viscous_x..)
    The total already includes pressure + viscous (+ porous), so we take the
    first vector directly — no summation, which avoids the version-dependent
    ambiguity of the older combined `forces.dat` column order. Returns the last
    time-step's (x, y, z).
    """
    cols = _last_data_row(dat_text)
    nums = [float(c) for c in cols[1:]]  # drop the time column
    if len(nums) < 3:
        raise ValueError(f"Unexpected .dat width: {len(nums)} numeric columns")
    return nums[0], nums[1], nums[2]


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

    bounds = stl_bounds(stl_path)
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
           build_snappy_dict("car.stl", loc, refinement, add_layers))
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
    lowerWall   {{ type noSlip; }}
    car         {{ type noSlip; }}
}}
""")
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


def run_stages(run_dir: str, cfg: OpenFOAMRunConfig, bashrc: str, timeout_s: int = 7200) -> dict:
    """Execute mesh + solve stages in order and return parsed logs.

    Stages (SPEC "Solver path"): surfaceFeatureExtract → blockMesh →
    snappyHexMesh → checkMesh → simpleFoam. Serial unless cfg.n_subdomains > 1.
    """
    run = Path(run_dir)
    solver = "simpleFoam"
    _run("surfaceFeatureExtract", run, bashrc, "surfaceFeatureExtract.log", timeout_s)
    _run("blockMesh", run, bashrc, "blockMesh.log", timeout_s)
    if cfg.n_subdomains > 1:
        _run("decomposePar -force", run, bashrc, "decomposePar.log", timeout_s)
        _run(f"mpirun -np {cfg.n_subdomains} snappyHexMesh -overwrite -parallel",
             run, bashrc, "snappyHexMesh.log", timeout_s)
        checkmesh = _run(f"mpirun -np {cfg.n_subdomains} checkMesh -parallel",
                         run, bashrc, "checkMesh.log", timeout_s)
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
    run_dir = str(Path(case_dir) / "runs" / f"run_{os.getpid()}_{abs(hash(stl_path)) % 10_000}")
    meta = build_case(run_dir, stl_path, cfg)
    try:
        logs = run_stages(run_dir, cfg, resolved_bashrc)
        neg = parse_negative_volume_cells(logs["checkmesh_log"])
        residual = parse_final_p_residual(logs["solver_log"])
        courant = parse_max_courant(logs["solver_log"])
        (fx, _fy, fz), (_mx, my, _mz) = read_force_and_moment(run_dir)
        try:
            yp_min, yp_max = parse_yplus_range(_read_yplus(run_dir, logs["solver_log"]))
        except ValueError:
            yp_min, yp_max = float("nan"), float("nan")
        return {
            "D20_half": abs(fx),
            "L_half": fz,
            "A_half": meta["frontal_area_half"],
            "pitching_moment_half": my,
            "residual_final": residual,
            "negative_volume_cells": neg,
            "y_plus_min": yp_min,
            "y_plus_max": yp_max,
            "courant_max": courant,
        }
    finally:
        if not cfg.keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


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
