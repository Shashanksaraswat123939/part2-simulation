"""Stage 8 candidate database and logging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from physics_contract import ComponentMassCOM, FullCarMassCOM, FullCarQuantities


ALLOWED_LIFECYCLE_STATES = {
    "valid_simulated",
    "geometry_repaired",
    "geometry_rejected",
    "rule_rejected",
    "machining_rejected",
    "CFD_failed",
    "objective_failed",
    "converged",
}


@dataclass(frozen=True)
class CandidateRecord:
    """Candidate database record.

    Numeric units: W_mm and d_halo_mm in mm; STL/phi paths are strings;
    mass_report/com_report use kg and m; cfd_force_report uses N, m^2, and
    dimensionless Cm; T_raw/T_penalized are seconds.

    Invalid input behavior:
        lifecycle_state is validated at construction time in __post_init__
        and again in read_candidate_record (defense-in-depth for externally
        edited JSON). write_candidate_record rejects non-JSON-serializable
        fields with TypeError from json.dump.
    """

    candidate_id: str
    W_mm: float
    d_halo_mm: float
    lifecycle_state: str
    # x_front is a per-RUN scalar under the two-stage split: Stage 1 chooses it
    # and Stage 2 freezes it. A record that does not say which x_front produced
    # it cannot be merged with records from another machine, which is the whole
    # point of sharding the d_halo sweep. It was absent, and the inner loop
    # passed it anyway -- so every CandidateRecord(**payload) raised TypeError
    # and _try_write_record swallowed it. No record has ever been written.
    x_front_mm: float = 0.0
    phi_grid_snapshot_paths: dict = field(default_factory=dict)
    # Everything below is optional so a SUMMARY record (what the inner loop can
    # supply: identity, lifecycle, times) is constructible. The full physics
    # payload is written when the caller has it; the ranking that Stage 2 exists
    # to do needs only T_raw plus the identifying scalars, and a record layer
    # that can only represent complete records ends up representing none.
    stl_path: str = ""
    mass_report: Optional[FullCarMassCOM] = None
    # Source docs list mass report and COM report separately, but
    # physics_contract.FullCarMassCOM already bundles both. Store the same
    # object in both fields rather than inventing a redundant second type.
    com_report: Optional[FullCarMassCOM] = None
    cfd_force_report: Optional[FullCarQuantities] = None
    T_raw: Optional[float] = None
    T_penalized: Optional[float] = None
    gradients: dict = field(default_factory=dict)
    adjoint_sensitivity_field_path: Optional[str] = None
    setup_logs: str = ""
    failure_reason: Optional[str] = None
    # Stability outcome, passed by inner_loop's `extra`. Real per-candidate
    # results, so they belong in the record rather than being dropped.
    statically_stable: Optional[bool] = None
    stability_notes: str = ""

    def __post_init__(self):
        # Guard: prevent unbounded setup_logs from creating huge JSON files.
        # 1MB is generous for log text; larger values indicate a bug or abuse.
        MAX_SETUP_LOGS_LEN = 1_000_000
        if len(self.setup_logs) > MAX_SETUP_LOGS_LEN:
            raise ValueError(
                f"setup_logs exceeds {MAX_SETUP_LOGS_LEN} chars "
                f"(got {len(self.setup_logs)}); truncate before writing"
            )
        # Guard: validate lifecycle_state at construction time, not just at read.
        # This prevents the codebase from creating its own invalid records.
        if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES:
            raise ValueError(
                f"invalid lifecycle_state: {self.lifecycle_state!r}; "
                f"must be one of {sorted(ALLOWED_LIFECYCLE_STATES)}"
            )


def _component_to_dict(component: ComponentMassCOM) -> dict:
    return {
        "name": component.name,
        "mass_kg": component.mass_kg,
        "com_x_m": component.com_x_m,
        "com_y_m": component.com_y_m,
        "com_z_m": component.com_z_m,
    }


def _mass_com_to_dict(report) -> dict:
    """Duck-typed on the four scalars.

    Part 3's MassReport carries the same COM scalars but no per-component
    breakdown, and it is what the inner loop actually holds at record-write
    time. Requiring `.components` would mean the record could only be written
    from a layer that does not exist there -- which is how the record ended up
    carrying no mass at all.
    """
    # null, not [], when the report has no breakdown to give. Every real record
    # is written from Part 3's MassReport, so `components` was always [] -- which
    # reads as "the car is made of nothing" rather than "this layer does not
    # carry the breakdown". A reader cannot tell those apart from an empty list.
    components = getattr(report, "components", None)
    return {
        "total_mass_kg": report.total_mass_kg,
        "com_x_m": report.com_x_m,
        "com_y_m": report.com_y_m,
        "com_z_m": report.com_z_m,
        "components": (None if components is None
                       else [_component_to_dict(c) for c in components]),
        "propellant_mass_kg": getattr(report, "propellant_mass_kg", None),
    }


def _full_car_quantities_from_dict(data: dict) -> FullCarQuantities:
    import dataclasses
    names = {f.name for f in dataclasses.fields(FullCarQuantities)}
    return FullCarQuantities(**{k: v for k, v in data.items() if k in names})


def _mass_com_from_dict(data: dict) -> FullCarMassCOM:
    return FullCarMassCOM(
        total_mass_kg=data["total_mass_kg"],
        com_x_m=data["com_x_m"],
        com_y_m=data["com_y_m"],
        com_z_m=data["com_z_m"],
        # `or []` because components is now null when the writing layer had no
        # breakdown to give; iterating None would raise on every real record.
        components=tuple(ComponentMassCOM(**component)
                         for component in (data.get("components") or [])),
    )


def _record_to_dict(record: CandidateRecord) -> dict:
    return {
        "candidate_id": record.candidate_id,
        "W_mm": record.W_mm,
        "x_front_mm": record.x_front_mm,
        "d_halo_mm": record.d_halo_mm,
        "phi_grid_snapshot_paths": record.phi_grid_snapshot_paths,
        "stl_path": record.stl_path,
        # None for a summary record — see the field comments. Serialising null
        # keeps the schema stable so a reader can tell "not captured" from
        # "captured as zero".
        "mass_report": (None if record.mass_report is None
                        else _mass_com_to_dict(record.mass_report)),
        "com_report": (None if record.com_report is None
                       else _mass_com_to_dict(record.com_report)),
        # Hand-listed field names, so anything added upstream stops here. That
        # is how force_oscillation -- the reproducibility bar on D20, measured
        # at 26-33% against a 5% limit -- was plumbed from Part 2 into
        # CFDOutcome and then silently dropped one layer before the file it was
        # meant to reach. getattr, so a report that predates a field (or a test
        # double that never had it) still serialises instead of raising.
        "cfd_force_report": (None if record.cfd_force_report is None else {
            k: getattr(record.cfd_force_report, k, None)
            for k in ("D20", "L", "Cm", "A", "converged", "residual_final",
                      "force_oscillation")
        }),
        "T_raw": record.T_raw,
        "T_penalized": record.T_penalized,
        "gradients": record.gradients,
        "adjoint_sensitivity_field_path": record.adjoint_sensitivity_field_path,
        "setup_logs": record.setup_logs,
        "failure_reason": record.failure_reason,
        "lifecycle_state": record.lifecycle_state,
        "statically_stable": record.statically_stable,
        "stability_notes": record.stability_notes,
    }


def write_candidate_record(record: CandidateRecord, out_dir: str) -> str:
    """
    Serializes record to JSON at {out_dir}/{candidate_id}.json.
    Non-JSON-serializable fields must be rejected with a clear TypeError
    rather than silently coerced. Returns the full path written.

    Invalid input behavior:
        Raises ValueError if candidate_id contains path separators or '..'
        (path traversal guard). Rejects NaN/Infinity float values with
        ValueError (non-standard JSON). Rejects non-JSON-serializable fields
        with TypeError from json.dump.
    """
    # Path traversal guard: candidate_id must be a safe filename.
    # Only reject exact '.' and '..' (not substring '..'), since a bare
    # candidate_id with no separators cannot be a multi-segment path traversal.
    # Separator rejection is handled by the '/' and '\\' checks above.
    safe_id = record.candidate_id.replace("\\", "/")
    if "/" in safe_id or safe_id != record.candidate_id:
        raise ValueError(
            f"candidate_id contains path separators: {record.candidate_id!r}"
        )
    if safe_id in ("..", "."):
        raise ValueError(
            f"candidate_id cannot be '.' or '..': {record.candidate_id!r}"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"{record.candidate_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_record_to_dict(record), f, indent=2, allow_nan=False)
    return str(path)


def read_candidate_record(path: str) -> CandidateRecord:
    """Deserializes JSON back into CandidateRecord. Raises FileNotFoundError
    if path doesn't exist, raises ValueError if lifecycle_state is not one
    of the exact 8 allowed strings listed above."""
    record_path = Path(path)
    if not record_path.exists():
        raise FileNotFoundError(path)
    with record_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data["lifecycle_state"] not in ALLOWED_LIFECYCLE_STATES:
        raise ValueError(f"invalid lifecycle_state: {data['lifecycle_state']}")
    # Tolerant of SUMMARY records: the physics-heavy blocks are optional (the
    # inner loop does not hold them), and older records predate x_front_mm and
    # the stability fields. A reader that cannot read what the writer emits is
    # worse than no reader -- and until the write path was fixed on 2026-07-28,
    # nothing had ever exercised this round trip on a real record.
    def _opt(key, fn):
        v = data.get(key)
        return None if v is None else fn(v)

    return CandidateRecord(
        candidate_id=data["candidate_id"],
        W_mm=data["W_mm"],
        x_front_mm=data.get("x_front_mm", 0.0),
        d_halo_mm=data["d_halo_mm"],
        phi_grid_snapshot_paths=data.get("phi_grid_snapshot_paths") or {},
        stl_path=data.get("stl_path") or "",
        mass_report=_opt("mass_report", _mass_com_from_dict),
        com_report=_opt("com_report", _mass_com_from_dict),
        # Filtered: the record now also stores the health fields (converged,
        # residual_final, force_oscillation) that FullCarQuantities does not
        # model. Splatting the whole dict raised TypeError on every record
        # written after they were added. They stay readable from the raw json.
        cfd_force_report=_opt("cfd_force_report", _full_car_quantities_from_dict),
        T_raw=data.get("T_raw"),
        T_penalized=data.get("T_penalized"),
        gradients=data.get("gradients") or {},
        adjoint_sensitivity_field_path=data.get("adjoint_sensitivity_field_path"),
        setup_logs=data.get("setup_logs") or "",
        failure_reason=data.get("failure_reason"),
        lifecycle_state=data["lifecycle_state"],
        statically_stable=data.get("statically_stable"),
        stability_notes=data.get("stability_notes") or "",
    )
