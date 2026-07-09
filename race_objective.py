"""
JAX differentiable race-time objective for a STEM Racing / F1 in Schools car.

Version 3 adds a differentiable COM-height term for the 3D optimizer.

Purpose
-------
This file is meant to be the optimizer/adjoint-facing race model, not just a
post-processing simulator.

It converts the CSV thrust/mass sheet into smooth analytic functions, then
integrates the car dynamics in distance-domain:

    independent variable: x from 0 m to 20 m
    state: [t, q]
        t = elapsed race time
        q = v^2

The equations are:

    dt/dx = 1 / v
    dq/dx = 2a

where:

    a = [F_thrust(t) - D20*q/20^2 - mu*m(t)*g] / [m(t) + N*I/r^2]

Using q = v^2 avoids the old timestep loop and avoids a branch-based finish
event. Race time is simply t at x = 20 m, so JAX can differentiate it.

Inputs to the differentiable objective:
    D20              drag force at 20 m/s, N
    car_weight_kg    entered car weight, kg
    mu               rolling friction coefficient
    wheel_moi_kg_m2  MOI of one wheel, kg*m^2
    time_coefficient coefficient multiplier; use 1.0 for normal time
    com_height_m     COM height above the track/thrust reference, m

Outputs:
    race time in seconds
    gradients wrt D20, car_weight_kg, mu, wheel_moi_kg_m2, time_coefficient, COM height

Notes
-----
- This keeps your old mass logic:
      m_car(t) = car_weight_kg + 0.021 + m_sheet(t) - 0.048
- Wheel inertia is converted into effective mass:
      m_eff = m_car + N*I/r^2
- The CSV is not used as a lookup table inside the objective. A smooth RBF
  surrogate is fitted once, then evaluated with pure JAX.
- This is single-point aero for now: D(v)=D20*(v/20)^2.
- Version 2 uses a non-uniform distance grid clustered near launch so the
  distance-domain singularity at v=0 is resolved without needing 20k steps.
- Version 3 adds a smooth COM-height penalty. By default the target COM height
  is 30 mm and the penalty is +0.001 s per mm away from that target. This sign
  makes 30 mm the optimum; using a negative sign would make the optimizer push
  the COM away from the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


G = 9.81
TRACK_LENGTH = 20.0
REFERENCE_SPEED = 20.0
N_WHEELS = 4
R_WHEEL = 0.015

# COM penalty model.
# Placeholder data points: (offset_mm, penalty_s) where offset = com_height_mm - 30.
# Slightly asymmetric: nose-up (negative offset, COM below thrust line) costs more.
# REPLACE these with real ballast experiment measurements when available.
_COM_PENALTY_DATA_MM = np.array([-12.0, -8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 12.0])
_COM_PENALTY_DATA_S  = np.array([ 0.028, 0.015, 0.007, 0.002, 0.0, 0.001, 0.005, 0.011, 0.020])

# Fit degree-4 polynomial (offset in metres). Subtract value at delta=0 so
# penalty is exactly 0 at the 30 mm target regardless of polyfit residuals.
_COM_PENALTY_DATA_M = _COM_PENALTY_DATA_MM * 1e-3
_COM_POLY_COEFFS_NP = np.polyfit(_COM_PENALTY_DATA_M, _COM_PENALTY_DATA_S, 4)
_COM_POLY_BASELINE  = float(np.polyval(_COM_POLY_COEFFS_NP, 0.0))
_COM_POLY_COEFFS    = jnp.asarray(_COM_POLY_COEFFS_NP, dtype=jnp.float64)
COM_TARGET_HEIGHT_M = 0.030

# Parameter order for the optimizer-facing vector.
PARAM_NAMES = (
    "drag_20_n",
    "car_weight_kg",
    "mu",
    "wheel_moi_kg_m2",
    "time_coefficient",
    "com_height_m",
)


class SmoothSheetModel(NamedTuple):
    centers: jnp.ndarray
    width: jnp.ndarray
    force_coeffs: jnp.ndarray
    mass_coeffs: jnp.ndarray
    t_min: jnp.ndarray
    t_max: jnp.ndarray
    mass_sheet_final: jnp.ndarray
    tail_tau: jnp.ndarray
    x_start: jnp.ndarray
    x_grid_power: jnp.ndarray
    n_steps: int


@dataclass(frozen=True)
class BuildSettings:
    n_basis: int = 80
    ridge: float = 1e-8
    tail_tau: float = 0.025
    x_start: float = 1e-4
    x_grid_power: float = 2.0
    n_steps: int = 1000


def _clean_csv_arrays(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = ["time (s)", "force (N)", "mass (kg)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    clean = df[required].copy()
    clean = clean.apply(pd.to_numeric, errors="coerce").dropna()
    clean = clean.sort_values("time (s)")
    clean = clean.groupby("time (s)", as_index=False).mean()

    t = clean["time (s)"].to_numpy(dtype=np.float64)
    force = clean["force (N)"].to_numpy(dtype=np.float64)
    mass = clean["mass (kg)"].to_numpy(dtype=np.float64)

    if len(t) < 4:
        raise ValueError("CSV needs at least 4 valid data points.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time column must be strictly increasing after cleaning.")

    return t, force, mass


def _rbf_design(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    z = (t[:, None] - centers[None, :]) / width
    rbf = np.exp(-0.5 * z * z)
    # Add constant + linear term so the fit handles the slow mass tail better.
    t_scaled = (t - t[0]) / max(t[-1] - t[0], 1e-12)
    return np.column_stack([rbf, np.ones_like(t), t_scaled])


def _ridge_fit(Phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    b = Phi.T @ y
    return np.linalg.solve(A, b)


def build_smooth_sheet_model(
    csv_path: str | Path = "co2_thrust_data.csv",
    settings: BuildSettings = BuildSettings(),
) -> SmoothSheetModel:
    """Fit smooth JAX-evaluable thrust and mass-sheet functions from the CSV."""
    t, force, mass_sheet = _clean_csv_arrays(csv_path)

    centers = np.linspace(float(t[0]), float(t[-1]), settings.n_basis)
    spacing = centers[1] - centers[0] if len(centers) > 1 else (t[-1] - t[0])
    width = max(2.5 * spacing, 1e-6)

    Phi = _rbf_design(t, centers, width)
    force_coeffs = _ridge_fit(Phi, force, settings.ridge)
    mass_coeffs = _ridge_fit(Phi, mass_sheet, settings.ridge)

    return SmoothSheetModel(
        centers=jnp.asarray(centers, dtype=jnp.float64),
        width=jnp.asarray(width, dtype=jnp.float64),
        force_coeffs=jnp.asarray(force_coeffs, dtype=jnp.float64),
        mass_coeffs=jnp.asarray(mass_coeffs, dtype=jnp.float64),
        t_min=jnp.asarray(float(t[0]), dtype=jnp.float64),
        t_max=jnp.asarray(float(t[-1]), dtype=jnp.float64),
        mass_sheet_final=jnp.asarray(float(mass_sheet[-1]), dtype=jnp.float64),
        tail_tau=jnp.asarray(settings.tail_tau, dtype=jnp.float64),
        x_start=jnp.asarray(settings.x_start, dtype=jnp.float64),
        x_grid_power=jnp.asarray(settings.x_grid_power, dtype=jnp.float64),
        n_steps=settings.n_steps,
    )


def _smooth_positive(x: jnp.ndarray, scale: float = 1e-6) -> jnp.ndarray:
    """Smooth approximation of max(x, 0), retaining gradients."""
    s = jnp.asarray(scale, dtype=x.dtype)
    return s * jax.nn.softplus(x / s)


def _eval_rbf_model(t: jnp.ndarray, model: SmoothSheetModel, coeffs: jnp.ndarray) -> jnp.ndarray:
    z = (t - model.centers) / model.width
    rbf = jnp.exp(-0.5 * z * z)
    t_scaled = (t - model.t_min) / jnp.maximum(model.t_max - model.t_min, 1e-12)
    features = jnp.concatenate([rbf, jnp.array([1.0, t_scaled], dtype=t.dtype)])
    return jnp.dot(features, coeffs)


def thrust_force(t: jnp.ndarray, model: SmoothSheetModel) -> jnp.ndarray:
    """
    Smooth thrust function F(t).

    The RBF fit handles the in-sheet curve. A smooth logistic tail turns thrust
    off after the sheet's final time instead of using a hard if-statement.
    """
    raw = _eval_rbf_model(t, model, model.force_coeffs)
    nonnegative = _smooth_positive(raw, scale=1e-4)
    end_gate = jax.nn.sigmoid((model.t_max - t) / model.tail_tau)
    return nonnegative * end_gate


def sheet_mass(t: jnp.ndarray, model: SmoothSheetModel) -> jnp.ndarray:
    """
    Smooth version of the CSV mass column.

    After the CSV ends, it smoothly approaches the final sheet mass instead of
    branching to a constant value.
    """
    raw = _eval_rbf_model(t, model, model.mass_coeffs)
    end_gate = jax.nn.sigmoid((model.t_max - t) / model.tail_tau)
    return model.mass_sheet_final + end_gate * (raw - model.mass_sheet_final)


def car_mass_from_time(
    t: jnp.ndarray,
    car_weight_kg: jnp.ndarray,
    model: SmoothSheetModel,
) -> jnp.ndarray:
    # Preserves old logic:
    # m_car = 0.021 + m_sheet - 0.048 + entered_car_weight
    m = car_weight_kg + 0.021 + sheet_mass(t, model) - 0.048
    return _smooth_positive(m, scale=1e-5)


def com_height_time_penalty(params: jnp.ndarray) -> jnp.ndarray:
    """COM-height penalty fitted from ballast experiment data.

    Evaluates a degree-4 polynomial in (com_height_m - 30mm).
    _COM_POLY_COEFFS are fitted at module load from _COM_PENALTY_DATA_*.
    Replace those data arrays with real measurements when available.
    """
    p = unpack_params(params)
    delta = p["com_height_m"] - COM_TARGET_HEIGHT_M
    return jnp.polyval(_COM_POLY_COEFFS, delta) - _COM_POLY_BASELINE


def unpack_params(params: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {name: params[i] for i, name in enumerate(PARAM_NAMES)}


def initial_state(params: jnp.ndarray, model: SmoothSheetModel) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Start slightly after x=0 using a local constant-acceleration launch estimate.

    Distance-domain dynamics has dt/dx = 1/v, which is singular exactly at
    v=0. The singularity is physically integrable, but numerically awkward.
    We therefore initialize at x_start using:

        x = 0.5*a0*t^2
        q = v^2 = 2*a0*x

    where a0 is the initial acceleration at t=0, v=0.
    """
    p = unpack_params(params)
    t0 = jnp.asarray(0.0, dtype=params.dtype)
    m0 = car_mass_from_time(t0, p["car_weight_kg"], model)
    m_eff0 = m0 + N_WHEELS * p["wheel_moi_kg_m2"] / (R_WHEEL ** 2)
    friction0 = p["mu"] * m0 * G
    a0 = (thrust_force(t0, model) - friction0) / m_eff0

    # Valid cars should have positive launch acceleration. The smooth floor keeps
    # gradients finite for bad inputs instead of crashing.
    a0_safe = _smooth_positive(a0, scale=1e-4) + 1e-8

    q0 = 2.0 * a0_safe * model.x_start
    t_start = jnp.sqrt(2.0 * model.x_start / a0_safe)
    return t_start, q0


def distance_derivatives(
    state: tuple[jnp.ndarray, jnp.ndarray],
    _x: jnp.ndarray,
    params: jnp.ndarray,
    model: SmoothSheetModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return dt/dx and dq/dx at the current distance location."""
    t, q = state
    p = unpack_params(params)

    # q = v^2. Keep it positive smoothly so invalid designs become slow/penalized
    # instead of producing NaNs.
    q_safe = _smooth_positive(q, scale=1e-6) + 1e-10

    m = car_mass_from_time(t, p["car_weight_kg"], model)
    m_eff = m + N_WHEELS * p["wheel_moi_kg_m2"] / (R_WHEEL ** 2)

    F_thrust = thrust_force(t, model)
    F_drag = p["drag_20_n"] * q_safe / (REFERENCE_SPEED ** 2)
    F_fric = p["mu"] * m * G

    a = (F_thrust - F_drag - F_fric) / m_eff

    dt_dx = 1.0 / jnp.sqrt(q_safe)
    dq_dx = 2.0 * a
    return dt_dx, dq_dx


def rk4_step(
    state: tuple[jnp.ndarray, jnp.ndarray],
    x: jnp.ndarray,
    dx: jnp.ndarray,
    params: jnp.ndarray,
    model: SmoothSheetModel,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def add_state(s, k, factor):
        return (s[0] + factor * dx * k[0], s[1] + factor * dx * k[1])

    k1 = distance_derivatives(state, x, params, model)
    k2 = distance_derivatives(add_state(state, k1, 0.5), x + 0.5 * dx, params, model)
    k3 = distance_derivatives(add_state(state, k2, 0.5), x + 0.5 * dx, params, model)
    k4 = distance_derivatives(add_state(state, k3, 1.0), x + dx, params, model)

    t_new = state[0] + (dx / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    q_new = state[1] + (dx / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    return t_new, q_new


def race_time_seconds(params: jnp.ndarray, model: SmoothSheetModel) -> jnp.ndarray:
    """Differentiable predicted race time in seconds."""
    p = unpack_params(params)

    x0 = model.x_start
    x1 = jnp.asarray(TRACK_LENGTH, dtype=params.dtype)

    # Non-uniform distance grid.
    # The distance-domain equations contain dt/dx = 1/v, which is singular
    # near launch. Uniform dx needs many steps to resolve the first few
    # centimeters. A powered grid clusters points near x=0 while keeping the
    # whole integration fixed-shape and JAX-differentiable.
    u_edges = jnp.linspace(0.0, 1.0, model.n_steps + 1, dtype=params.dtype)
    x_edges = x0 + (x1 - x0) * (u_edges ** model.x_grid_power)
    xs = x_edges[:-1]
    dxs = x_edges[1:] - x_edges[:-1]

    state0 = initial_state(params, model)

    def scan_body(state, step_data):
        x, dx = step_data
        new_state = rk4_step(state, x, dx, params, model)
        return new_state, new_state

    final_state, history = jax.lax.scan(scan_body, state0, (xs, dxs))
    t_finish, q_finish = final_state

    # Smooth penalty if final v^2 is near/below zero. For normal finishing cars,
    # this is approximately zero.
    q_safe_finish = _smooth_positive(q_finish, scale=1e-6)
    low_speed_penalty = 0.05 * _smooth_positive(0.01 - q_safe_finish, scale=1e-4)

    com_penalty = com_height_time_penalty(params)

    return p["time_coefficient"] * (t_finish + low_speed_penalty + com_penalty)


def race_value_and_grad(params: np.ndarray, model: SmoothSheetModel) -> tuple[float, dict[str, float]]:
    params_jax = jnp.asarray(params, dtype=jnp.float64)
    value, grad_vec = jax.value_and_grad(race_time_seconds)(params_jax, model)
    return float(value), {name: float(grad_vec[i]) for i, name in enumerate(PARAM_NAMES)}


def finite_difference_check(
    params: np.ndarray,
    model: SmoothSheetModel,
    relative_eps: float = 1e-4,
) -> list[dict[str, float]]:
    """Compare JAX gradients to central finite differences."""
    base = np.asarray(params, dtype=np.float64)
    value, grads = race_value_and_grad(base, model)
    rows = []

    fallback_steps = {
        "drag_20_n": 1e-5,
        "car_weight_kg": 1e-6,
        "mu": 1e-6,
        "wheel_moi_kg_m2": 1e-11,
        "time_coefficient": 1e-5,
        "com_height_m": 1e-6,
    }

    for i, name in enumerate(PARAM_NAMES):
        # Use a scale-aware perturbation. Some parameters, especially wheel MOI
        # in kg*m^2, are around 1e-7, so max(abs(x), 1) would be catastrophically
        # too large for a finite-difference test.
        step = max(relative_eps * abs(base[i]), fallback_steps[name])
        p_plus = base.copy()
        p_minus = base.copy()
        p_plus[i] += step
        p_minus[i] -= step

        f_plus = float(race_time_seconds(jnp.asarray(p_plus), model))
        f_minus = float(race_time_seconds(jnp.asarray(p_minus), model))
        fd = (f_plus - f_minus) / (2.0 * step)
        jg = grads[name]
        rows.append(
            {
                "param": name,
                "jax_grad": jg,
                "finite_diff": fd,
                "abs_error": abs(jg - fd),
                "base_value": base[i],
                "race_time_s": value,
            }
        )

    return rows


def make_param_vector(
    drag_20_n: float,
    car_weight_g: float,
    mu: float,
    wheel_moi_g_mm2: float,
    time_coefficient_percent: float = 100.0,
    com_height_mm: float = 30.0,
) -> np.ndarray:
    """Helper matching the old script's user-facing units."""
    return np.array(
        [
            drag_20_n,
            car_weight_g / 1000.0,
            mu,
            wheel_moi_g_mm2 * 1e-9,
            time_coefficient_percent / 100.0,
            com_height_mm / 1000.0,
        ],
        dtype=np.float64,
    )


def main() -> None:
    # Enable float64 for better gradient checks and physics accuracy.
    # If your install disables this globally, run with:
    #   export JAX_ENABLE_X64=True
    jax.config.update("jax_enable_x64", True)

    csv_path = Path(input("CSV path [co2_thrust_data.csv]: ").strip() or "co2_thrust_data.csv")
    model = build_smooth_sheet_model(csv_path)

    params = make_param_vector(
        drag_20_n=float(input("Enter drag force at 20 m/s (N): ")),
        car_weight_g=float(input("Weight of car (g): ")),
        mu=float(input("Enter rolling friction coefficient: ")),
        wheel_moi_g_mm2=float(input("Enter wheel MOI (g·mm^2): ")),
        time_coefficient_percent=float(input("Enter Time Coefficient (in %) [100 = unchanged]: ") or "100"),
        com_height_mm=float(input("Enter COM height (mm) [30 = ideal]: ") or "30"),
    )

    T, grads = race_value_and_grad(params, model)
    print("\nPredicted differentiable race time:")
    print(f"  T = {T:.6f} s")
    params_jax = jnp.asarray(params, dtype=jnp.float64)
    print(f"  COM penalty = {float(com_height_time_penalty(params_jax)):.6f} s")

    print("\nGradients:")
    for name, grad in grads.items():
        print(f"  dT/d{name:>18s} = {grad:.9e}")

    print("\nFinite-difference check:")
    for row in finite_difference_check(params, model):
        print(
            f"  {row['param']:>18s}: "
            f"JAX={row['jax_grad']:.9e}, "
            f"FD={row['finite_diff']:.9e}, "
            f"abs_err={row['abs_error']:.3e}"
        )


if __name__ == "__main__":
    main()
