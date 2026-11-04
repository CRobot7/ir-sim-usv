"""Shared WaveField queries: single, batch, mixed hulls, env hook."""

from pathlib import Path

import numpy as np

import irsim
from irsim.lib.algorithm.vessel_spec import load_vessel
from irsim.lib.algorithm.wave import HullGeom, WaveField

_TESTS = Path(__file__).resolve().parent


def _wamv_hull():
    spec = load_vessel("wamv")
    return spec.hull_at_draft(spec.T_ref)


def _flat_hull(wamv: HullGeom) -> HullGeom:
    return HullGeom(
        name="flat",
        points=wamv.points,
        A_x=wamv.A_x * 0.2,
        A_y=wamv.A_y * 3.0,
        A_z=5.0,
        T=0.03,
        B=0.8,
        A_x_ref=wamv.A_x * 0.2,
        A_y_ref=wamv.A_y * 3.0,
        T_ref=0.03,
        hull_type="board",
    )


def test_hs_zero_wrench_is_zero():
    hull = _wamv_hull()
    wf = WaveField(Hs=0.0, Tp=4.0, seed=0)
    tau = wf.wrench([10.0, 10.0, 0.0], [0.0, 0.0, 0.0], hull, 1.0)
    assert np.allclose(tau, 0.0)


def test_same_query_is_repeatable():
    hull = _wamv_hull()
    wf = WaveField(Hs=0.6, Tp=4.0, seed=1)
    eta = [10.0, 10.0, 0.2]
    nu = [0.4, 0.0, 0.0]
    a = wf.wrench(eta, nu, hull, 2.0)
    b = wf.wrench(eta, nu, hull, 2.0)
    assert np.allclose(a, b)


def test_two_positions_differ_at_same_t():
    hull = _wamv_hull()
    wf = WaveField(Hs=0.6, Tp=4.0, seed=1)
    t = 2.0
    nu = [0.0, 0.0, 0.0]
    a = wf.wrench([10.0, 10.0, 0.0], nu, hull, t)
    b = wf.wrench([28.0, 10.0, 0.0], nu, hull, t)
    assert not np.allclose(a, b)


def test_wrench_many_matches_loop():
    hull = _wamv_hull()
    wf = WaveField(Hs=0.6, Tp=4.0, seed=2)
    etas = np.array([[10.0, 10.0, 0.0], [20.0, 12.0, 0.2], [8.0, 30.0, -0.4]])
    nus = np.zeros((3, 3))
    t = 1.5
    batch = wf.wrench_many(etas, nus, [hull, hull, hull], t)
    loop = np.stack([wf.wrench(e, n, hull, t) for e, n in zip(etas, nus, strict=True)])
    assert np.allclose(batch, loop)


def test_different_hulls_same_pose():
    wamv = _wamv_hull()
    flat = _flat_hull(wamv)
    wf = WaveField(Hs=0.8, Tp=4.0, seed=3)
    eta = [12.0, 12.0, 0.0]
    nu = [0.0, 0.0, 0.0]
    t = 3.0
    a = wf.wrench(eta, nu, wamv, t)
    b = wf.wrench(eta, nu, flat, t)
    mixed = wf.wrench_many(np.array([eta, eta]), np.array([nu, nu]), [wamv, flat], t)
    assert not np.allclose(a, b)
    assert np.allclose(mixed[0], a)
    assert np.allclose(mixed[1], b)


def test_env_wave_moves_uncontrolled_boats():
    env = irsim.make(
        str(_TESTS / "test_usv_wave_world.yaml"), save_ani=False, display=False
    )
    assert env.wave is not None
    s0 = [obj.state.copy() for obj in env.robot_list]
    for _i in range(80):
        env.step()
    moved = [
        float(np.linalg.norm(obj.state - s))
        for obj, s in zip(env.robot_list, s0, strict=True)
    ]
    assert moved[0] > 1e-4
    assert moved[1] > 1e-4
    # spatial field: two boats do not copy each other
    d0 = env.robot_list[0].state.flatten() - s0[0].flatten()
    d1 = env.robot_list[1].state.flatten() - s0[1].flatten()
    assert not np.allclose(d0, d1, atol=1e-6)
    tau = env.wave_wrench(env.robot)
    assert tau.shape == (3,)
    env.end()


def test_env_without_wave_stays_put():
    env = irsim.make(str(_TESTS / "test_usv_world.yaml"), save_ani=False, display=False)
    assert env.wave is None
    s0 = env.robot.state.copy()
    for _i in range(20):
        env.step()
    assert np.allclose(env.robot.state, s0)
    assert np.allclose(env.wave_wrench(), 0.0)
    env.end()
