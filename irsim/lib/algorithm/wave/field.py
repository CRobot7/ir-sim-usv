"""Shared irregular WaveField: elevation, pressure, and FK wrench queries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from irsim.lib.algorithm.wave.hull import HullGeom
from irsim.lib.algorithm.wave.spectrum import G, discrete_components

RHO = 1025.0


class WaveField:
    """One irregular sea, shared by every vessel in the world.

    Components ``zeta_a, k, omega, mu, epsilon`` are identical for all
    queries. Elevation and pressure depend only on ``(x, y, z, t)``.
    The 3DOF wrench also needs hull stations and encounter frequency.
    """

    def __init__(
        self,
        Hs: float = 0.2,
        Tp: float | None = 5.0,
        heading: float = 0.0,
        n_freq: int = 20,
        n_dir: int = 10,
        seed: int = 0,
        rho: float = RHO,
        g: float = G,
    ) -> None:
        self.Hs = float(Hs)
        self.Tp = None if Tp is None else float(Tp)
        self.heading = float(heading)
        self.n_freq = int(n_freq)
        self.n_dir = int(n_dir)
        self.seed = int(seed)
        self.rho = float(rho)
        self.g = float(g)
        comps = discrete_components(
            self.Hs,
            self.Tp,
            heading=self.heading,
            n_freq=self.n_freq,
            n_dir=self.n_dir,
            seed=self.seed,
        )
        self.zeta_a = comps["zeta_a"]
        self.k = comps["k"]
        self.omega = comps["omega"]
        self.mu = comps["mu"]
        self.epsilon = comps["epsilon"]

    @classmethod
    def from_config(cls, cfg: dict | None) -> WaveField | None:
        if not cfg:
            return None
        return cls(
            Hs=float(cfg.get("Hs", 0.2)),
            Tp=cfg.get("Tp", 5.0),
            heading=float(cfg.get("heading", 0.0)),
            n_freq=int(cfg.get("n_freq", 20)),
            n_dir=int(cfg.get("n_dir", 10)),
            seed=int(cfg.get("seed", 0)),
        )

    def _phase(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float,
        omega: np.ndarray | None = None,
    ) -> np.ndarray:
        """Phase with trailing component axis.

        ``x, y`` broadcast against ``(n_comp,)``. ``omega`` defaults to the
        intrinsic frequencies (spatial field). Pass encounter frequencies
        of the same shape as ``self.omega`` or broadcast ``(..., n_comp)``.
        """
        omega_use = self.omega if omega is None else omega
        return (
            self.k * (x * np.cos(self.mu) + y * np.sin(self.mu))
            - omega_use * t
            + self.epsilon
        )

    def elevation(self, x, y, t: float) -> np.ndarray:
        """Free-surface elevation [m] at world ``(x, y)`` and time ``t``."""
        x_a = np.asarray(x, dtype=float)[..., None]
        y_a = np.asarray(y, dtype=float)[..., None]
        eta = np.sum(self.zeta_a * np.cos(self._phase(x_a, y_a, t)), axis=-1)
        if eta.ndim == 0:
            return float(eta)
        return eta

    def pressure(self, x, y, z, t: float, omega: np.ndarray | None = None):
        """Dynamic pressure [Pa] (hydrostatic mean omitted). ``z`` up, 0 at SWL."""
        x_a = np.asarray(x, dtype=float)[..., None]
        y_a = np.asarray(y, dtype=float)[..., None]
        z_a = np.asarray(z, dtype=float)[..., None]
        decay = np.exp(self.k * np.minimum(z_a, 0.0))
        p = (
            self.rho
            * self.g
            * np.sum(
                self.zeta_a * decay * np.cos(self._phase(x_a, y_a, t, omega=omega)),
                axis=-1,
            )
        )
        if np.isscalar(x) and np.isscalar(y):
            return float(np.asarray(p).reshape(-1)[0])
        return p

    def _encounter_omega(self, u: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """``omega_e`` with shape ``(N, n_comp)``."""
        u = np.asarray(u, dtype=float).reshape(-1, 1)
        psi = np.asarray(psi, dtype=float).reshape(-1, 1)
        return self.omega[None, :] - (self.omega[None, :] ** 2 / self.g) * u * np.cos(
            self.mu[None, :] - psi
        )

    def _pressures_at_stations(
        self,
        etas: np.ndarray,
        nus: np.ndarray,
        hull: HullGeom,
        t: float,
    ) -> dict[str, np.ndarray]:
        """Named dynamic pressures, each ``(N,)``."""
        etas = np.asarray(etas, dtype=float).reshape(-1, 3)
        nus = np.asarray(nus, dtype=float).reshape(-1, 3)
        n = etas.shape[0]
        psi = etas[:, 2]
        c = np.cos(psi)
        s = np.sin(psi)
        z = np.full(n, -0.5 * max(float(hull.T), 1e-6))
        omega_e = self._encounter_omega(nus[:, 0], psi)
        out: dict[str, np.ndarray] = {}
        for label, (xb, yb) in hull.points.items():
            xw = etas[:, 0] + xb * c - yb * s
            yw = etas[:, 1] + xb * s + yb * c
            # pressure() expects omega broadcast; pass (N, n_comp)
            x_a = xw[:, None]
            y_a = yw[:, None]
            z_a = z[:, None]
            decay = np.exp(self.k[None, :] * np.minimum(z_a, 0.0))
            phase = (
                self.k[None, :] * (x_a * np.cos(self.mu) + y_a * np.sin(self.mu))
                - omega_e * t
                + self.epsilon[None, :]
            )
            out[label] = (
                self.rho
                * self.g
                * np.sum(self.zeta_a[None, :] * decay * np.cos(phase), axis=1)
            )
        return out

    def _fk_from_pressures(
        self, hull: HullGeom, pressures: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Planar wrench ``(N, 3)`` = ``[X, Y, N]`` from named station pressures."""
        n = next(iter(pressures.values())).shape[0]
        p_bow = pressures.get("bow", np.zeros(n))
        p_stern = pressures.get("stern", np.zeros(n))
        p_port = pressures.get("port", np.zeros(n))
        p_stbd = pressures.get("stbd", np.zeros(n))
        x_bow = hull.points.get("bow", (0.0, 0.0))[0]
        x_stern = hull.points.get("stern", (0.0, 0.0))[0]
        X = (p_bow - p_stern) * hull.A_x
        Y = (p_port - p_stbd) * hull.A_y
        N = (p_port - p_stbd) * hull.A_x * (hull.B / 4.0) + (
            p_bow * x_bow + p_stern * x_stern
        ) * (hull.A_y / 2.0)
        return np.column_stack((X, Y, N))

    def wrench(self, eta, nu, hull: HullGeom, t: float) -> np.ndarray:
        """Body-frame ``[X, Y, N]`` at one pose. Shape ``(3,)``."""
        return self.wrench_many(
            np.asarray(eta, dtype=float).reshape(1, 3),
            np.asarray(nu, dtype=float).reshape(1, 3),
            [hull],
            t,
        )[0]

    def wrench_many(
        self,
        etas: np.ndarray,
        nus: np.ndarray,
        hulls: Sequence[HullGeom],
        t: float,
    ) -> np.ndarray:
        """Same-time batch query. ``etas, nus`` are ``(N, 3)``; result ``(N, 3)``."""
        etas = np.asarray(etas, dtype=float).reshape(-1, 3)
        nus = np.asarray(nus, dtype=float).reshape(-1, 3)
        n = etas.shape[0]
        if n == 0:
            return np.zeros((0, 3))
        if len(hulls) != n:
            raise ValueError("hulls must match the number of poses")
        out = np.zeros((n, 3), dtype=float)
        groups: dict[tuple, list[int]] = {}
        for i, hull in enumerate(hulls):
            groups.setdefault(hull.key(), []).append(i)
        for idxs in groups.values():
            ii = np.asarray(idxs, dtype=int)
            hull = hulls[int(ii[0])]
            press = self._pressures_at_stations(etas[ii], nus[ii], hull, t)
            out[ii] = self._fk_from_pressures(hull, press)
        return out
