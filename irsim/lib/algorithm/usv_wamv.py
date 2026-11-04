"""3DOF WAM-V USV dynamics (Fossen surge/sway/yaw).

The equations match the project simulator in
``usv_calib/simulator/wamv_3dof.py``, whose coefficients come from
Klinger et al., IEEE JOE 2017, "Control of an Unmanned Surface Vehicle
With Uncertain Displacement and Drag" (Tables I-III).

State (column vector)::

    eta, nu = [x, y, psi, u, v, r]

    eta_dot = J(psi) nu
    M nu_dot = tau - C(nu) nu - D(nu)

Control is the two waterjet thrusts ``[T_port, T_stbd]`` in Newtons,
clamped to ``[0, Twj_max]``. Yaw moment is ``(T_port - T_stbd) * B / 2``.
"""

from __future__ import annotations

import numpy as np

from irsim.lib.algorithm.wamv_params import (
    TWJ_MAX,
    XUDOT_FRAC_DEFAULT,
    WAMVCoeffs,
    coeffs_at_mass,
    mass_from_payload,
    velocity_linear_drag,
)
from irsim.util.util import validate_shape


def _sat(x: float, x_min: float, x_max: float) -> float:
    return float(min(max(x, x_min), x_max))


def _j_psi(psi: float) -> np.ndarray:
    """Body-to-world Jacobian for 3DOF (Klinger et al. 2017, eq. 6)."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _mass_matrix(c: WAMVCoeffs) -> np.ndarray:
    return np.array(
        [
            [c.m - c.X_udot, 0.0, 0.0],
            [0.0, c.m - c.Y_vdot, -c.Y_rdot],
            [0.0, -c.N_vdot, c.I_z - c.N_rdot],
        ],
        dtype=float,
    )


def _coriolis(nu: np.ndarray, c: WAMVCoeffs) -> np.ndarray:
    u, v, r = float(nu[0]), float(nu[1]), float(nu[2])
    crb = np.array(
        [
            [0.0, 0.0, -c.m * v],
            [0.0, 0.0, c.m * u],
            [c.m * v, -c.m * u, 0.0],
        ],
        dtype=float,
    )
    # Added-mass Coriolis. The /200 scale on the (1,3) and (3,1) terms is
    # retained from the project WAM-V simulator; it is not a Table III entry.
    mid = 0.5 * (c.Y_rdot + c.N_vdot)
    ca = np.array(
        [
            [0.0, 0.0, (c.Y_vdot * v + mid * r) / 200.0],
            [0.0, 0.0, -c.X_udot * u],
            [(-c.Y_vdot * v - mid * r) / 200.0, c.X_udot * u, 0.0],
        ],
        dtype=float,
    )
    return crb + ca


def _surge_pontoon_drag(nu: np.ndarray, c: WAMVCoeffs) -> tuple[float, float]:
    """Port/stbd surge drag (Table II X_u, X_uu) and the yaw they produce."""
    u, r = float(nu[0]), float(nu[2])
    y_port = -c.B / 2.0
    y_stbd = c.B / 2.0
    u_port = u - r * y_port
    u_stbd = u - r * y_stbd
    d_port = (c.X_uu / 2.0) * abs(u_port) * u_port + (c.X_u / 2.0) * u_port
    d_stbd = (c.X_uu / 2.0) * abs(u_stbd) * u_stbd + (c.X_u / 2.0) * u_stbd
    return float(d_port + d_stbd), float((d_stbd - d_port) * c.B / 2.0)


def _sway_yaw_drag(nu: np.ndarray, c: WAMVCoeffs) -> np.ndarray:
    u, v, r = float(nu[0]), float(nu[1]), float(nu[2])
    lin = velocity_linear_drag(u, v, c.T)
    fy = lin["Y_v"] * v + lin["Y_r"] * r
    fn = lin["N_v"] * v + lin["N_r"] * r
    fy += (c.Y_vv * abs(v) + c.Y_vr * abs(r)) * v + (
        c.Y_rv * abs(v) + c.Y_rr * abs(r)
    ) * r
    fn += (c.N_vv * abs(v) + c.N_vr * abs(r)) * v + (
        c.N_rv * abs(v) + c.N_rr * abs(r)
    ) * r
    return np.array([0.0, fy, fn], dtype=float)


class WAMV3DOF:
    """One WAM-V hull: rebuilds M when payload/mass changes."""

    def __init__(
        self,
        mass: float | None = None,
        payload: float = 0.0,
        twj_max: float = TWJ_MAX,
        xudot_frac: float = XUDOT_FRAC_DEFAULT,
    ) -> None:
        self.twj_max = float(twj_max)
        self.xudot_frac = float(xudot_frac)
        self.coeffs: WAMVCoeffs
        self.M = np.eye(3)
        self.Minv = np.eye(3)
        if mass is not None:
            self.set_displacement(float(mass))
        else:
            self.set_payload(payload)

    def set_displacement(self, mass: float) -> None:
        """Update mass/draft and rebuild M from Tables II-III."""
        self.coeffs = coeffs_at_mass(float(mass), xudot_frac=self.xudot_frac)
        self.M = _mass_matrix(self.coeffs)
        self.Minv = np.linalg.inv(self.M)

    def set_payload(self, payload: float | None = 0.0) -> None:
        """Payload relative to empty ``M0``; clamped to ``[0, P_FULL]``."""
        self.set_displacement(mass_from_payload(payload))

    def allocate(self, tau_x: float, tau_n: float) -> tuple[float, float]:
        """Map desired surge force / yaw moment to waterjet thrusts [N]."""
        half_b = self.coeffs.B / 2.0
        t_p = 0.5 * (tau_x + tau_n / half_b)
        t_s = 0.5 * (tau_x - tau_n / half_b)
        return _sat(t_p, 0.0, self.twj_max), _sat(t_s, 0.0, self.twj_max)

    def tau_from_thrust(self, thrust: np.ndarray) -> np.ndarray:
        t_p = _sat(float(thrust[0]), 0.0, self.twj_max)
        t_s = _sat(float(thrust[1]), 0.0, self.twj_max)
        return np.array(
            [t_p + t_s, 0.0, (t_p - t_s) * self.coeffs.B / 2.0],
            dtype=float,
        )

    def nu_dot(
        self,
        nu: np.ndarray,
        thrust: np.ndarray,
        tau_env: np.ndarray | None = None,
    ) -> np.ndarray:
        c = self.coeffs
        tau = self.tau_from_thrust(thrust)
        if tau_env is not None:
            tau = tau + np.asarray(tau_env, dtype=float).reshape(3)
        x_drag, n_drag = _surge_pontoon_drag(nu, c)
        f_sy = _sway_yaw_drag(nu, c)
        dnu = np.array([x_drag, -f_sy[1], -f_sy[2] + n_drag], dtype=float)
        return self.Minv @ (tau - _coriolis(nu, c) @ nu - dnu)

    def step_eta_nu(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        thrust: np.ndarray,
        dt: float,
        tau_env: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nu = np.asarray(nu, dtype=float).reshape(3)
        eta = np.asarray(eta, dtype=float).reshape(3)
        nu_next = nu + dt * self.nu_dot(nu, thrust, tau_env)
        eta_next = eta + dt * (_j_psi(float(eta[2])) @ nu)
        return eta_next, nu_next


@validate_shape(state=6, velocity=2)
def usv_wamv_kinematics(
    state: np.ndarray,
    velocity: np.ndarray,
    step_time: float,
    model: WAMV3DOF,
    tau_env: np.ndarray | None = None,
) -> np.ndarray:
    """Advance a 6x1 WAM-V state ``[x, y, psi, u, v, r]`` by one Euler step.

    Args:
        state: (6, 1) pose and body velocity.
        velocity: (2, 1) port/stbd waterjet thrust [N].
        step_time: Integration step [s].
        model: Coefficient / mass-matrix container.
        tau_env: Optional environmental wrench ``[X, Y, N]``.

    Returns:
        np.ndarray: Next (6, 1) state.
    """
    eta = state[0:3, 0]
    nu = state[3:6, 0]
    thrust = np.asarray(velocity, dtype=float).reshape(2)
    eta_next, nu_next = model.step_eta_nu(eta, nu, thrust, step_time, tau_env)
    next_state = np.zeros((6, 1), dtype=float)
    next_state[0:3, 0] = eta_next
    next_state[3:6, 0] = nu_next
    return next_state
