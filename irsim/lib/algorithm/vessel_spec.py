"""Load a USV vessel YAML into hydro coefficients and hull geometry."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from irsim.lib.algorithm.wave.hull import HullGeom, hull_from_mapping

_VESSEL_DIR = Path(__file__).resolve().parents[2] / "vessels"


@dataclass
class FossenCoeffs:
    """3DOF Fossen surge/sway/yaw coefficients at one displacement."""

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
    B: float = 1.83
    B_hull: float = 0.37
    L: float = 4.29
    LCG: float = 1.27
    rho: float = 1025.0
    Y_v: float = 0.0
    Y_r: float = 0.0
    N_v: float = 0.0
    N_r: float = 0.0
    linear_drag_model: str = "none"
    ca_scale: float = 1.0


@dataclass
class VesselSpec:
    """One vessel YAML: hydro recipe plus hull stations."""

    name: str
    recipe: str | None
    L: float
    LWL: float
    BOA: float
    B: float
    B_hull: float
    LCG: float
    AWP: float
    T_ref: float
    T_aft: float
    T_mid: float
    rho: float
    Cd: float
    M0: float
    M_lightship: float
    P_full: float
    M_nodes: np.ndarray
    X_uu_nodes: np.ndarray
    X_u_nodes: np.ndarray
    table_iii_scale: dict[str, float]
    xudot_frac_default: float
    twj_max: float
    ca_scale: float
    hull_mapping: dict[str, Any]
    explicit: dict[str, float] | None
    hull: HullGeom

    def clamp_payload(self, payload: float | None) -> float:
        if payload is None:
            return 0.0
        return float(np.clip(float(payload), 0.0, self.P_full))

    def mass_from_payload(self, payload: float | None) -> float:
        return self.M0 + self.clamp_payload(payload)

    def draft_from_mass(self, m: float) -> float:
        return float(self.T_ref + (m - self.M_lightship) / (self.rho * self.AWP))

    def interp_table_ii(
        self, m: float, drag_floor: float = 1e-3
    ) -> tuple[float, float]:
        m = float(m)
        nodes = self.M_nodes
        xuu_n = self.X_uu_nodes
        xu_n = self.X_u_nodes
        if m <= nodes[0]:
            dm = nodes[1] - nodes[0]
            t = (m - nodes[0]) / dm
            xuu = xuu_n[0] + t * (xuu_n[1] - xuu_n[0])
            xu = xu_n[0] + t * (xu_n[1] - xu_n[0])
        elif m >= nodes[-1]:
            dm = nodes[-1] - nodes[-2]
            t = (m - nodes[-1]) / dm
            xuu = xuu_n[-1] + t * (xuu_n[-1] - xuu_n[-2])
            xu = xu_n[-1] + t * (xu_n[-1] - xu_n[-2])
        else:
            xuu = float(np.interp(m, nodes, xuu_n))
            xu = float(np.interp(m, nodes, xu_n))
        return float(max(xuu, drag_floor)), float(max(xu, drag_floor))

    def _table_iii_added_mass(
        self, m: float, T: float, xudot_frac: float
    ) -> dict[str, float]:
        pi = np.pi
        rho = self.rho
        L_ = self.L
        lcg = self.LCG
        bh = self.B_hull
        s = self.table_iii_scale
        N_vdot = s.get("N_vdot", 2.5) * (
            -pi * rho * T**2 * ((L_ - lcg) ** 2 + lcg**2) / 2.0
        )
        N_rdot = s.get("N_rdot", 1.2) * (
            -(
                (4.75 / 2.0) * pi * rho * (bh / 2.0) * T**4
                + pi * rho * T**2 * ((L_ - lcg) ** 3 + lcg**3) / 3.0
            )
        )
        X_udot = xudot_frac * (-m)
        Y_rdot = s.get("Y_rdot", 0.2) * (
            -pi * rho * T**2 * ((L_ - lcg) ** 2 + lcg**2) / 2.0
        )
        Y_vdot = s.get("Y_vdot", 0.9) * (-pi * rho * T**2 * L_)
        return {
            "X_udot": float(X_udot),
            "Y_vdot": float(Y_vdot),
            "Y_rdot": float(Y_rdot),
            "N_vdot": float(N_vdot),
            "N_rdot": float(N_rdot),
        }

    def _table_iii_nonlinear_drag(self, T: float) -> dict[str, float]:
        rho = self.rho
        L_ = self.L
        lcg = self.LCG
        cd = self.Cd
        s = self.table_iii_scale
        Y_vv = s.get("Y_vv", 1.0) * (-rho * T * cd * L_)
        Y_vr = s.get("Y_vr", 1.0) * (
            -rho * T * (1.1 / 2.0) * ((L_ - lcg) ** 2 - lcg**2)
        )
        Y_rv = Y_vr
        Y_rr = s.get("Y_rr", 1.0) * (-rho * T * (cd / 3.0) * ((L_ - lcg) ** 3 + lcg**3))
        N_vv = Y_vr
        N_vr = Y_rr
        N_rv = Y_rr
        N_rr = s.get("N_rr", 1.0) * (-rho * T * (cd / 4.0) * ((L_ - lcg) ** 4 + lcg**4))
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

    def coeffs_at_mass(self, m: float, xudot_frac: float | None = None) -> FossenCoeffs:
        m = float(m)
        if self.recipe == "klinger2017":
            frac = self.xudot_frac_default if xudot_frac is None else float(xudot_frac)
            T = self.draft_from_mass(m)
            xuu, xu = self.interp_table_ii(m)
            am = self._table_iii_added_mass(m, T, frac)
            nd = self._table_iii_nonlinear_drag(T)
            I_z_ref = self.M_lightship * ((self.B / 2.0) ** 2 + (0.25 * self.L) ** 2)
            I_z = I_z_ref * (m / self.M_lightship)
            return FossenCoeffs(
                m=m,
                T=T,
                payload=m - self.M0,
                X_u=xu,
                X_uu=xuu,
                I_z=float(I_z),
                B=self.B,
                B_hull=self.B_hull,
                L=self.L,
                LCG=self.LCG,
                rho=self.rho,
                linear_drag_model="klinger",
                ca_scale=self.ca_scale,
                **am,
                **nd,
            )
        if not self.explicit:
            raise ValueError(
                f"vessel {self.name!r} has no hydro.recipe and no hydro.coeffs"
            )
        data = dict(self.explicit)
        data.setdefault("m", m)
        data.setdefault("payload", m - float(data.get("m", m)))
        if xudot_frac is not None and "X_udot" not in self.explicit:
            data["X_udot"] = float(xudot_frac) * (-float(data["m"]))
        data["m"] = m
        data["payload"] = m - self.M0
        if "T" not in self.explicit:
            data["T"] = self.draft_from_mass(m)
        if "I_z" in self.explicit and abs(float(self.explicit.get("m", m)) - m) > 1e-9:
            m0 = float(self.explicit["m"])
            data["I_z"] = float(self.explicit["I_z"]) * (m / m0)
        allowed = {f.name for f in fields(FossenCoeffs)}
        kwargs = {k: data[k] for k in allowed if k in data}
        kwargs.setdefault("B", self.B)
        kwargs.setdefault("B_hull", self.B_hull)
        kwargs.setdefault("L", self.L)
        kwargs.setdefault("LCG", self.LCG)
        kwargs.setdefault("rho", self.rho)
        kwargs.setdefault("ca_scale", self.ca_scale)
        kwargs.setdefault("linear_drag_model", "none")
        return FossenCoeffs(**kwargs)

    def hull_at_draft(self, T: float) -> HullGeom:
        return self.hull.at_draft(T)


def resolve_vessel_path(name_or_path: str) -> Path:
    """Resolve ``wamv``, ``wamv.yaml``, or a filesystem path to a YAML file."""
    raw = Path(name_or_path)
    if raw.is_file():
        return raw
    candidate = Path(str(name_or_path) + ".yaml")
    if candidate.is_file():
        return candidate
    stem = raw.stem if raw.suffix in {".yaml", ".yml"} else raw.name
    bundled = _VESSEL_DIR / f"{stem}.yaml"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"vessel spec {name_or_path!r} not found (looked in irsim/vessels/)"
    )


def load_vessel(name_or_path: str = "wamv") -> VesselSpec:
    path = resolve_vessel_path(name_or_path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return vessel_spec_from_dict(data, source=str(path))


def vessel_spec_from_dict(data: dict[str, Any], source: str = "") -> VesselSpec:
    name = str(data.get("name") or Path(source).stem or "vessel")
    geom = data.get("geometry") or {}
    load = data.get("load") or {}
    hydro = data.get("hydro") or {}
    hull_map = data.get("hull") or {}
    actuators = data.get("actuators") or {}

    L = float(geom["L"])
    LWL = float(geom.get("LWL", L))
    BOA = float(geom.get("BOA", geom.get("B", L)))
    B = float(geom.get("B", BOA))
    B_hull = float(geom.get("B_hull", BOA - B if BOA > B else B))
    LCG = float(geom.get("LCG", 0.5 * L))
    AWP = float(geom.get("AWP", 1.0))
    rho = float(geom.get("rho", 1025.0))
    Cd = float(geom.get("Cd", 1.1))
    t_aft = float(geom.get("T_aft_lightship", geom.get("T", 0.1)))
    t_mid = float(geom.get("T_mid_lightship", t_aft))
    T_ref = 0.5 * (t_aft + t_mid)

    M0 = float(
        load.get("M_slick", load.get("M0", hydro.get("coeffs", {}).get("m", 1.0)))
    )
    M_lightship = float(load.get("M_lightship", M0))
    M_full_paper = float(load.get("M_full_paper", M_lightship))
    P_full = float(load.get("P_full", 0.0))

    recipe = hydro.get("recipe")
    if recipe is not None:
        recipe = str(recipe)
    explicit = hydro.get("coeffs")
    if explicit is not None:
        explicit = {str(k): float(v) for k, v in explicit.items()}

    xuu = np.asarray(hydro.get("X_uu", [0.0, 0.0, 0.0]), dtype=float)
    xu = np.asarray(hydro.get("X_u", [0.0, 0.0, 0.0]), dtype=float)
    m_nodes = np.array([M0, M_lightship, M_full_paper], dtype=float)
    scales = {str(k): float(v) for k, v in (hydro.get("table_iii_scale") or {}).items()}

    hull = hull_from_mapping(
        name,
        hull_map,
        L=L,
        LWL=LWL,
        B=B,
        B_hull=B_hull,
        LCG=LCG,
        AWP=AWP,
        T=T_ref,
    )
    return VesselSpec(
        name=name,
        recipe=recipe,
        L=L,
        LWL=LWL,
        BOA=BOA,
        B=B,
        B_hull=B_hull,
        LCG=LCG,
        AWP=AWP,
        T_ref=T_ref,
        T_aft=t_aft,
        T_mid=t_mid,
        rho=rho,
        Cd=Cd,
        M0=M0,
        M_lightship=M_lightship,
        P_full=P_full,
        M_nodes=m_nodes,
        X_uu_nodes=xuu,
        X_u_nodes=xu,
        table_iii_scale=scales,
        xudot_frac_default=float(hydro.get("xudot_frac_default", 0.16)),
        twj_max=float(actuators.get("twj_max", 204.0)),
        ca_scale=float(hydro.get("coriolis_ca_scale", 1.0)),
        hull_mapping=hull_map,
        explicit=explicit,
        hull=hull,
    )


def klinger_velocity_linear_drag(
    u: float, v: float, T: float, spec: VesselSpec
) -> dict[str, float]:
    """Speed-dependent linear sway/yaw drag (Klinger / project simulator)."""
    pi = np.pi
    rho = spec.rho
    L_ = spec.L
    bh = spec.B_hull
    speed = float(np.sqrt(u * u + v * v) + 1e-12)
    abs_v = abs(v)
    T = max(float(T), 1e-6)
    bracket = 1.1 + 0.0045 * (L_ / T) - 0.1 * (bh / T) + 0.016 * (bh / T) ** 2
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
