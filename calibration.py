"""Stage 5 calibration data ingestion."""

from __future__ import annotations

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

from physics_contract import GRAVITY_MPS2


def _read_csv_columns(csv_path: str) -> dict[str, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        rows = list(reader)
        columns = {}
        for name in reader.fieldnames:
            columns[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        return columns


def _write_train_csv(columns: dict[str, np.ndarray], indices: np.ndarray) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns.keys()))
        writer.writeheader()
        for index in indices:
            writer.writerow({name: columns[name][index] for name in columns})
    return handle.name


@dataclass(frozen=True)
class ThrustSurrogate:
    """RBF surrogate over CO2 thrust curve, fitted from CSV."""

    _rbf: object
    t_min: float
    t_max: float

    def __call__(self, t: float) -> float:
        """
        Returns thrust in Newtons at time t (seconds).
        Raises ValueError if t < self.t_min or t > self.t_max
        (extrapolation guard -- never silently extrapolate).
        """
        if t < self.t_min or t > self.t_max:
            raise ValueError("time outside fitted thrust surrogate range")
        return float(self._rbf(np.asarray([[t]], dtype=float))[0])


def fit_thrust_surrogate(csv_path: str) -> ThrustSurrogate:
    """
    CSV must have columns 'time_s', 'thrust_N'. Fits RBFInterpolator.
    Raises FileNotFoundError if csv_path doesn't exist.
    Raises ValueError if CSV is missing either required column or has
    fewer than 2 data points (RBFInterpolator requires >= 2 points).
    """
    columns = _read_csv_columns(csv_path)
    required = {"time_s", "thrust_N"}
    if not required.issubset(columns):
        raise ValueError("CSV missing required columns: 'time_s', 'thrust_N'")
    t = columns["time_s"]
    thrust = columns["thrust_N"]
    if len(t) < 2:
        raise ValueError(f"thrust surrogate needs at least 2 data points, got {len(t)}")
    rbf = RBFInterpolator(t[:, None], thrust)
    return ThrustSurrogate(_rbf=rbf, t_min=float(np.min(t)), t_max=float(np.max(t)))


@dataclass(frozen=True)
class FrictionCoefficient:
    """Rolling friction coefficient.

    mu is dimensionless. Raises ValueError if mu is outside [0.0, 1.0].
    """

    mu: float

    def __post_init__(self):
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu out of physically sane range: {self.mu}")


def fit_mu_from_track_test(csv_path: str) -> FrictionCoefficient:
    """
    CSV must have columns 'deceleration_mps2' or equivalent raw
    measurement columns needed to back out mu = deceleration / g.
    PLACEHOLDER: exact column schema for the real track-test CSV is not
    given in the source docs. Implemented behavior:
      1. raise FileNotFoundError if csv_path doesn't exist
      2. raise NotImplementedError("PLACEHOLDER: real track-test CSV "
         "schema not finalized -- see calibration.py fit_mu_from_track_test "
         "docstring") if the CSV does not contain a column literally named
         'mu_fitted'. Until the real schema is defined, this function only
         accepts a CSV that already contains a precomputed 'mu_fitted' column
         and takes its mean.
    """
    columns = _read_csv_columns(csv_path)
    if "mu_fitted" not in columns:
        raise NotImplementedError(
            "PLACEHOLDER: real track-test CSV schema not finalized -- see "
            "calibration.py fit_mu_from_track_test docstring"
        )
    return FrictionCoefficient(mu=float(np.mean(columns["mu_fitted"])))


@dataclass(frozen=True)
class COMPenaltyCurve:
    """Polynomial COM penalty curve.

    _poly_coeffs are numpy.polyfit coefficients. x_com_min and x_com_max are
    in m. Calling returns additive penalty in seconds. Raises ValueError if
    x_com is outside the fitted range.
    """

    _poly_coeffs: tuple[float, ...]
    x_com_min: float
    x_com_max: float

    def __call__(self, x_com: float) -> float:
        """
        Returns COM_penalty (seconds, additive to T_raw) at given x_com (m).
        Raises ValueError if x_com outside [x_com_min, x_com_max]
        (extrapolation guard).
        """
        if x_com < self.x_com_min or x_com > self.x_com_max:
            raise ValueError("x_com outside fitted COM penalty range")
        return float(np.polyval(np.asarray(self._poly_coeffs), x_com))


def fit_com_penalty_curve(csv_path: str, degree: int = 3) -> COMPenaltyCurve:
    """
    CSV must have columns 'x_com_m', 'penalty_s'. Fits numpy.polyfit of
    the given degree. Raises FileNotFoundError / ValueError as above.
    Raises ValueError if x values are all identical (rank-deficient fit).
    """
    columns = _read_csv_columns(csv_path)
    required = {"x_com_m", "penalty_s"}
    if not required.issubset(columns):
        raise ValueError("CSV missing required columns: 'x_com_m', 'penalty_s'")
    x = columns["x_com_m"]
    penalty = columns["penalty_s"]
    if len(set(x)) < degree + 1:
        raise ValueError(
            f"x_com_m needs at least {degree + 1} distinct points for degree-{degree} polyfit, "
            f"got {len(set(x))} distinct"
        )
    import warnings
    from numpy.exceptions import RankWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", RankWarning)
        coeffs = tuple(float(v) for v in np.polyfit(x, penalty, degree))
    return COMPenaltyCurve(_poly_coeffs=coeffs, x_com_min=float(np.min(x)), x_com_max=float(np.max(x)))


def compute_held_out_residual(
    csv_path: str,
    fit_fn,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> float:
    """
    Splits the CSV rows deterministically into train/holdout and returns R^2.

    Args:
        csv_path: CSV path.
        fit_fn: callable fitting on a CSV path and returning a callable model.
        holdout_fraction: fraction of rows held out, dimensionless.
        seed: numpy RandomState seed, count.

    Returns:
        R^2 as a dimensionless float.

    Invalid input behavior:
        File and schema exceptions propagate from the fit function and CSV
        reader. No clamping or warning is performed.
    """
    columns = _read_csv_columns(csv_path)
    n = len(next(iter(columns.values())))
    if n < 4:
        raise ValueError(
            f"need at least 4 rows for held-out residual, got {n}"
        )
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    holdout_idx = order[:n_holdout]
    train_idx = order[n_holdout:]
    train_path = _write_train_csv(columns, train_idx)
    try:
        model = fit_fn(train_path)
    finally:
        Path(train_path).unlink(missing_ok=True)

    if {"time_s", "thrust_N"}.issubset(columns):
        x = columns["time_s"][holdout_idx]
        y = columns["thrust_N"][holdout_idx]
    elif {"x_com_m", "penalty_s"}.issubset(columns):
        x = columns["x_com_m"][holdout_idx]
        y = columns["penalty_s"][holdout_idx]
    else:
        raise ValueError("CSV schema is not supported for held-out residual")

    pred = np.asarray([model(float(value)) for value in x], dtype=float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot
