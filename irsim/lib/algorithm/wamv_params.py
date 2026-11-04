"""WAM-V defaults, loaded from ``irsim/vessels/wamv.yaml``.

Klinger 2017 closed-form hydro lives in :mod:`irsim.lib.algorithm.vessel_spec`.
This module re-exports the built-in WAM-V numbers so existing imports keep working.
"""

from __future__ import annotations

from irsim.lib.algorithm.vessel_spec import (
    FossenCoeffs as WAMVCoeffs,
)
from irsim.lib.algorithm.vessel_spec import (
    klinger_velocity_linear_drag,
    load_vessel,
)

_SPEC = load_vessel("wamv")

L = _SPEC.L
LWL = _SPEC.LWL
T_AFT_LIGHTSHIP = _SPEC.T_aft
T_MID_LIGHTSHIP = _SPEC.T_mid
T_REF = _SPEC.T_ref
BOA = _SPEC.BOA
BWL = _SPEC.B
AWP = _SPEC.AWP
B = _SPEC.B
LCG = _SPEC.LCG
B_HULL = _SPEC.B_hull
M_SLICK = _SPEC.M0
M_LIGHTSHIP = _SPEC.M_lightship
M_FULL_PAPER = float(_SPEC.M_nodes[-1])
M_NODES = _SPEC.M_nodes
M0 = _SPEC.M0
P_FULL = _SPEC.P_full
M_FULL = M0 + P_FULL
XU_UU_NODES = _SPEC.X_uu_nodes
XU_NODES = _SPEC.X_u_nodes
RHO = _SPEC.rho
CD = _SPEC.Cd
TWJ_MAX = _SPEC.twj_max
I_Z_REF = M_LIGHTSHIP * ((B / 2.0) ** 2 + (0.25 * L) ** 2)
XUDOT_FRAC_TABLE = 0.075
XUDOT_FRAC_DEFAULT = _SPEC.xudot_frac_default


def clamp_payload(payload: float | None) -> float:
    return _SPEC.clamp_payload(payload)


def mass_from_payload(payload: float | None) -> float:
    return _SPEC.mass_from_payload(payload)


def draft_from_mass(
    m: float, m_ref: float | None = None, T_ref: float | None = None
) -> float:
    del m_ref, T_ref
    return _SPEC.draft_from_mass(m)


def interp_table_ii(m: float, drag_floor: float = 1e-3) -> tuple[float, float]:
    return _SPEC.interp_table_ii(m, drag_floor=drag_floor)


def velocity_linear_drag(u: float, v: float, T: float) -> dict[str, float]:
    return klinger_velocity_linear_drag(u, v, T, _SPEC)


def coeffs_at_mass(
    m: float,
    xudot_frac: float = XUDOT_FRAC_DEFAULT,
) -> WAMVCoeffs:
    return _SPEC.coeffs_at_mass(m, xudot_frac=xudot_frac)
