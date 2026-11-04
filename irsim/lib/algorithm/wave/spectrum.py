"""Pierson-Moskowitz spectrum with ITTC directional spreading."""

from __future__ import annotations

import numpy as np

G = 9.81


def peak_omega(Hs: float, Tp: float | None = None) -> float:
    """Peak angular frequency [rad/s].

    If ``Tp`` is given it wins. Otherwise the ITTC two-parameter default
    ``Tp = 3.86 * sqrt(Hs)`` (seconds, Hs in metres) is used.
    """
    Hs = max(float(Hs), 0.0)
    if Tp is not None and float(Tp) > 1e-6:
        return 2.0 * np.pi / float(Tp)
    if Hs <= 1e-12:
        return 1.0
    tp = 3.86 * np.sqrt(Hs)
    return 2.0 * np.pi / tp


def pierson_moskowitz(omega: np.ndarray, Hs: float, omega_p: float) -> np.ndarray:
    """ITTC two-parameter PM density ``S(omega)`` [m^2 s / rad]."""
    omega = np.asarray(omega, dtype=float)
    Hs = max(float(Hs), 0.0)
    omega_p = max(float(omega_p), 1e-9)
    out = np.zeros_like(omega, dtype=float)
    if Hs <= 1e-12:
        return out
    w = np.maximum(omega, 1e-9)
    return (
        (5.0 / 16.0) * Hs**2 * (omega_p**4) / w**5 * np.exp(-1.25 * (omega_p / w) ** 4)
    )


def ittc_spreading(theta: np.ndarray) -> np.ndarray:
    """``D(theta) = (2/pi) cos^2(theta)`` for ``|theta| < pi/2``, else 0."""
    theta = np.asarray(theta, dtype=float)
    d = np.zeros_like(theta, dtype=float)
    mask = np.abs(theta) < (0.5 * np.pi)
    d[mask] = (2.0 / np.pi) * np.cos(theta[mask]) ** 2
    return d


def discrete_components(
    Hs: float,
    Tp: float | None = None,
    heading: float = 0.0,
    n_freq: int = 20,
    n_dir: int = 10,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Discretize PM + ITTC spreading into shared regular-wave components.

    Returns arrays ``zeta_a, k, omega, mu, epsilon`` of shape ``(n_freq * n_dir,)``.
    """
    Hs = max(float(Hs), 0.0)
    n_freq = max(int(n_freq), 1)
    n_dir = max(int(n_dir), 1)
    omega_p = peak_omega(Hs, Tp)
    omega_min = 0.5 * omega_p
    omega_max = 3.0 * omega_p
    omega = np.linspace(omega_min, omega_max, n_freq)
    d_omega = float(omega[1] - omega[0]) if n_freq > 1 else omega_p

    # Midpoints across (-pi/2, pi/2) relative to mean heading.
    d_theta = np.pi / n_dir
    theta_rel = -0.5 * np.pi + (np.arange(n_dir) + 0.5) * d_theta
    mu = heading + theta_rel
    spreading = ittc_spreading(theta_rel)
    spec = pierson_moskowitz(omega, Hs, omega_p)

    omega_g, mu_g = np.meshgrid(omega, mu, indexing="ij")
    spec_g, spr_g = np.meshgrid(spec, spreading, indexing="ij")
    variance = spec_g * spr_g * d_omega * d_theta
    zeta_a = np.sqrt(2.0 * np.maximum(variance, 0.0)).reshape(-1)
    omega_f = omega_g.reshape(-1)
    mu_f = mu_g.reshape(-1)
    k = omega_f**2 / G

    rng = np.random.default_rng(int(seed))
    epsilon = rng.uniform(0.0, 2.0 * np.pi, size=zeta_a.shape)
    if Hs <= 1e-12:
        zeta_a[:] = 0.0
    return {
        "zeta_a": zeta_a.astype(float),
        "k": k.astype(float),
        "omega": omega_f.astype(float),
        "mu": mu_f.astype(float),
        "epsilon": epsilon.astype(float),
    }
