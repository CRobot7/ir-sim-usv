"""WAM-V USV14 hydrodynamic coefficients.

Parameter source
----------------
Klinger, W. B., Bertaska, I. R., von Ellenrieder, K. D., and Dhanak, M. R.,
"Control of an Unmanned Surface Vehicle With Uncertain Displacement and Drag,"
IEEE Journal of Oceanic Engineering, vol. 42, no. 2, pp. 458-477, Apr. 2017.

Tables used here (same numbering as the paper):

* Table I  — principal characteristics (L, LWL, beam, draft, LCG, AWP, ...).
* Table II — surge linear / quadratic drag ``X_u``, ``X_uu`` at the three
  measured displacements (slick 150 kg, lightship 220 kg, full 246 kg).
* Table III — added-mass and nonlinear sway/yaw drag, evaluated from the
  paper's closed-form expressions (with the same empirical scale factors
  as the project simulator in ``usv_calib/simulator``).

The 3DOF Fossen model is surge/sway/yaw only. Dual waterjets are assumed
aligned with the hulls: each jet produces a surge force in ``[0, Twj_max]``
(no reverse), and the yaw moment is ``(T_port - T_stbd) * B / 2``.

Load convention (project extension of Table I, not a paper table):

    m = M0 + payload,  payload in [0, P_FULL]
    M0 = 150 kg (slick / empty), P_FULL = 130 kg → design full 280 kg.

Table II is piecewise-linear in mass and extrapolated outside the paper
nodes; a small positive floor keeps extrapolated ``X_u``, ``X_uu`` from
changing sign (which would turn drag into thrust).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Table I — principal characteristics
# ---------------------------------------------------------------------------
L = 4.29  # overall length [m]
LWL = 3.21  # length on waterline [m]
T_AFT_LIGHTSHIP = 0.127  # aft draft at lightship [m]
T_MID_LIGHTSHIP = 0.105  # midship draft at lightship [m]
T_REF = 0.5 * (T_AFT_LIGHTSHIP + T_MID_LIGHTSHIP)
BOA = 2.20  # beam overall [m]
BWL = 2.19  # beam on waterline [m]
AWP = 1.1  # waterplane area [m^2]
B = 1.83  # hull-center spacing, used as the waterjet lever [m]
NABLA_LIGHTSHIP = 0.21  # lightship displacement volume [m^3]
LCG = 1.27  # longitudinal center of gravity from the transom [m]
B_HULL = BOA - B  # single-hull beam [m]

# Paper Table I / body-text displacement nodes for Table II interpolation
M_SLICK = 150.0
M_LIGHTSHIP = 220.0
M_FULL_PAPER = 246.0
M_NODES = np.array([M_SLICK, M_LIGHTSHIP, M_FULL_PAPER], dtype=float)

# Continuous load domain used by the project simulator
M0 = M_SLICK  # empty
P_FULL = 130.0  # max add-on payload [kg]
M_FULL = M0 + P_FULL  # 280 kg design full load

# Table II nodes (paper): X_uu, X_u at slick / lightship / full
XU_UU_NODES = np.array([5.8722, 6.9627, 2.6693], dtype=float)
XU_NODES = np.array([50.897, 55.771, 47.341], dtype=float)

RHO = 1025.0  # seawater density [kg/m^3]
CD = 1.1  # cross-flow drag coefficient (Table III)
TWJ_MAX = 204.0  # per-waterjet thrust limit [N]

# Lightship yaw inertia, then scaled linearly with mass
I_Z_REF = M_LIGHTSHIP * ((B / 2.0) ** 2 + (0.25 * L) ** 2)

# Table III uses X_udot = xudot_frac * (-m). Paper table value is 0.075;
# the project simulator defaults to 0.16 (identified).
XUDOT_FRAC_TABLE = 0.075
XUDOT_FRAC_DEFAULT = 0.16


@dataclass
class WAMVCoeffs:
    """Hydrodynamic coefficients at one displacement."""

    m: float
    T: float
    payload: float
    X_u: float
    X_uu: float
    X_udot: float
    Y_vdot: float
    Y_rdot: float
    N_vdot: float
    N_rdot: float
    Y_vv: float
    Y_vr: float
    Y_rv: float
    Y_rr: float
    N_vv: float
    N_vr: float
    N_rv: float
    N_rr: float
    I_z: float
    B: float = B
    B_hull: float = B_HULL
    L: float = L
    LCG: float = LCG


def clamp_payload(payload: float | None) -> float:
    """Default empty (0); never negative; never above P_FULL."""
    if payload is None:
        return 0.0
    return float(np.clip(float(payload), 0.0, P_FULL))


def mass_from_payload(payload: float | None) -> float:
    return M0 + clamp_payload(payload)


def draft_from_mass(m: float, m_ref: float = M_LIGHTSHIP, T_ref: float = T_REF) -> float:
    """Draft from mass via waterplane area (hydrostatic, Table I AWP)."""
    return float(T_ref + (m - m_ref) / (RHO * AWP))


def interp_table_ii(m: float, drag_floor: float = 1e-3) -> Tuple[float, float]:
    """Piecewise-linear Table II; extrapolate outside the paper nodes.

    Magnitudes are floored to ``drag_floor``: raw extrapolation beyond 246 kg
    drives ``X_uu`` (and eventually ``X_u``) negative, which makes surge drag
    act as thrust.
    """
    m = float(m)
    if m <= M_NODES[0]:
        dm = M_NODES[1] - M_NODES[0]
        t = (m - M_NODES[0]) / dm
        xuu = XU_UU_NODES[0] + t * (XU_UU_NODES[1] - XU_UU_NODES[0])
        xu = XU_NODES[0] + t * (XU_NODES[1] - XU_NODES[0])
    elif m >= M_NODES[-1]:
        dm = M_NODES[-1] - M_NODES[-2]
        t = (m - M_NODES[-1]) / dm
        xuu = XU_UU_NODES[-1] + t * (XU_UU_NODES[-1] - XU_UU_NODES[-2])
        xu = XU_NODES[-1] + t * (XU_NODES[-1] - XU_NODES[-2])
    else:
        xuu = float(np.interp(m, M_NODES, XU_UU_NODES))
        xu = float(np.interp(m, M_NODES, XU_NODES))
    return float(max(xuu, drag_floor)), float(max(xu, drag_floor))


def _table_iii_added_mass(m: float, T: float, xudot_frac: float) -> Dict[str, float]:
    """Table III added-mass terms, with the project simulator scale factors."""
    pi = np.pi
    rho = RHO
    L_ = L
    lcg = LCG
    bh = B_HULL

    N_vdot = 2.5 * (-pi * rho * T**2 * ((L_ - lcg) ** 2 + lcg**2) / 2.0)
    N_rdot = 1.2 * (
        -(
            (4.75 / 2.0) * pi * rho * (bh / 2.0) * T**4
            + pi * rho * T**2 * ((L_ - lcg) ** 3 + lcg**3) / 3.0
        )
    )
    X_udot = xudot_frac * (-m)
    Y_rdot = 0.2 * (-pi * rho * T**2 * ((L_ - lcg) ** 2 + lcg**2) / 2.0)
    Y_vdot = 0.9 * (-pi * rho * T**2 * L_)

    return {
        "X_udot": float(X_udot),
        "Y_vdot": float(Y_vdot),
        "Y_rdot": float(Y_rdot),
        "N_vdot": float(N_vdot),
        "N_rdot": float(N_rdot),
    }


def _table_iii_nonlinear_drag(T: float) -> Dict[str, float]:
    """Table III nonlinear cross-flow drag (paper closed-form, scale 1.0)."""
    rho = RHO
    L_ = L
    lcg = LCG
    cd = CD

    Y_vv = 1.0 * (-rho * T * cd * L_)
    Y_vr = 1.0 * (-rho * T * (1.1 / 2.0) * ((L_ - lcg) ** 2 - lcg**2))
    Y_rv = Y_vr
    Y_rr = 1.0 * (-rho * T * (cd / 3.0) * ((L_ - lcg) ** 3 + lcg**3))
    N_vv = Y_vr
    N_vr = Y_rr
    N_rv = Y_rr
    N_rr = 1.0 * (-rho * T * (cd / 4.0) * ((L_ - lcg) ** 4 + lcg**4))

    return {
        "Y_vv": float(Y_vv),
        "Y_vr": float(Y_vr),
        "Y_rv": float(Y_rv),
        "Y_rr": float(Y_rr),
        "N_vv": float(N_vv),
        "N_vr": float(N_vr),
        "N_rv": float(N_rv),
        "N_rr": float(N_rr),
    }


def velocity_linear_drag(u: float, v: float, T: float) -> Dict[str, float]:
    """Speed-dependent linear sway/yaw drag (project simulator, Table III family)."""
    pi = np.pi
    rho = RHO
    L_ = L
    bh = B_HULL
    speed = float(np.sqrt(u * u + v * v) + 1e-12)
    abs_v = abs(v)

    bracket = (
        1.1
        + 0.0045 * (L_ / max(T, 1e-6))
        - 0.1 * (bh / max(T, 1e-6))
        + 0.016 * (bh / max(T, 1e-6)) ** 2
    )
    Y_v = 0.5 * (-40.0 * rho * abs_v * bracket * (pi * T * L_ / 2.0))
    N_r = 0.02 * (-pi * rho * speed * T**2 * L_**2)
    N_v = 0.06 * (-pi * rho * speed * T**2 * L_)
    Y_r = 6.0 * (-pi * rho * speed * T**2 * L_)

    return {
        "Y_v": float(Y_v),
        "Y_r": float(Y_r),
        "N_v": float(N_v),
        "N_r": float(N_r),
    }


def coeffs_at_mass(
    m: float,
    xudot_frac: float = XUDOT_FRAC_DEFAULT,
) -> WAMVCoeffs:
    """Rebuild all coefficients at displacement ``m`` [kg]."""
    m = float(m)
    T = draft_from_mass(m)
    xuu, xu = interp_table_ii(m)
    am = _table_iii_added_mass(m, T, xudot_frac)
    nd = _table_iii_nonlinear_drag(T)
    I_z = I_Z_REF * (m / M_LIGHTSHIP)
    return WAMVCoeffs(
        m=m,
        T=T,
        payload=m - M0,
        X_u=xu,
        X_uu=xuu,
        I_z=float(I_z),
        **am,
        **nd,
    )
