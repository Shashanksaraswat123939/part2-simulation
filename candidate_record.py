"""Stage 8 candidate database and logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
        Validation is performed by read_candidate_record for lifecycle_state.
        write_candidate_record rejects non-JSON-serializable fields with
        TypeError from json.dump.
    """

    candidate_id: str
    W_mm: float
    d_halo_mm: float
    phi_grid_snapshot_paths: dict
    stl_path: str
    mass_report: FullCarMassCOM
    # Source docs list mass report and COM report separately, but
    # physics_contract.FullCarMassCOM already bundles both. Store the same
    # object in both fields rather than inventing a redundant second type.
    com_report: FullCarMassCOM
    cfd_force_report: FullCarQuantities
    T_raw: Optional[float]
    T_penalized: Optional[float]
    gradients: dict
    adjoint_sensitivity_field_path: Optional[str]
    setup_logs: str
    failure_reason: Optional[str]
    lifecycle_state: str

    def __post_init__(self):
        # Guard: prevent unbounded setup_logs from creating huge JSON files.
        # 1MB is generous for log text; larger values indicate a bug or abuse.
        MAX_SETUP_LOGS_LEN = 1_000_000
        if len(self.setup_logs) > MAX_SETUP_LOGS_LEN:
            raise ValueError(
                f"setup_logs exceeds {MAX_SETUP_LOGS_LEN} chars "
                f"(got {len(self.setup_logs)}); truncate before writing"
            )


def _component_to_dict(component: ComponentMassCOM) -> dict:
    return {
        "name": component.name,
        "mass_kg": component.mass_kg,
        "com_x_m": component.com_x_m,
        "com_y_m": component.com_y_m,
        "com_z_m": component.com_z_m,
    }


def _mass_com_to_dict(report: FullCarMassCOM) -> dict:
    return {
        "total_mass_kg": report.total_mass_kg,
        "com_x_m": report.com_x_m,
        "com_y_m": report.com_y_m,
        "com_z_m": report.com_z_m,
        "components": [_component_to_dict(component) for component in report.components],
    }


def _mass_com_from_dict(data: dict) -> FullCarMassCOM:
    return FullCarMassCOM(
        total_mass_kg=data["total_mass_kg"],
        com_x_m=data["com_x_m"],
        com_y_m=data["com_y_m"],
        com_z_m=data["com_z_m"],
        components=tuple(ComponentMassCOM(**component) for component in data.get("components", [])),
    )


def _record_to_dict(record: CandidateRecord) -> dict:
    return {
        "candidate_id": record.candidate_id,
        "W_mm": record.W_mm,
        "d_halo_mm": record.d_halo_mm,
        "phi_grid_snapshot_paths": record.phi_grid_snapshot_paths,
        "stl_path": record.stl_path,
        "mass_report": _mass_com_to_dict(record.mass_report),
        "com_report": _mass_com_to_dict(record.com_report),
        "cfd_force_report": {
            "D20": record.cfd_force_report.D20,
            "L": record.cfd_force_report.L,
            "Cm": record.cfd_force_report.Cm,
            "A": record.cfd_force_report.A,
        },
        "T_raw": record.T_raw,
        "T_penalized": record.T_penalized,
        "gradients": record.gradients,
        "adjoint_sensitivity_field_path": record.adjoint_sensitivity_field_path,
        "setup_logs": record.setup_logs,
        "failure_reason": record.failure_reason,
        "lifecycle_state": record.lifecycle_state,
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
    safe_id = record.candidate_id.replace("\\", "/")
    if "/" in safe_id or ".." in safe_id or safe_id != record.candidate_id:
        raise ValueError(
            f"candidate_id contains path separators or '..': {record.candidate_id!r}"
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
    return CandidateRecord(
        candidate_id=data["candidate_id"],
        W_mm=data["W_mm"],
        d_halo_mm=data["d_halo_mm"],
        phi_grid_snapshot_paths=data["phi_grid_snapshot_paths"],
        stl_path=data["stl_path"],
        mass_report=_mass_com_from_dict(data["mass_report"]),
        com_report=_mass_com_from_dict(data["com_report"]),
        cfd_force_report=FullCarQuantities(**data["cfd_force_report"]),
        T_raw=data["T_raw"],
        T_penalized=data["T_penalized"],
        gradients=data["gradients"],
        adjoint_sensitivity_field_path=data["adjoint_sensitivity_field_path"],
        setup_logs=data["setup_logs"],
        failure_reason=data["failure_reason"],
        lifecycle_state=data["lifecycle_state"],
    )
