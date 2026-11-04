"""WAM-V USV 3DOF kinematics (Klinger et al. 2017)."""

import numpy as np
import pytest

from irsim.lib.algorithm.usv_wamv import WAMV3DOF, usv_wamv_kinematics
from irsim.lib.algorithm.wamv_params import M0, M_LIGHTSHIP, TWJ_MAX, coeffs_at_mass
from irsim.lib.handler.kinematics_handler import KinematicsFactory, UsvKinematics


def _rest_state():
    return np.zeros((6, 1))


def test_rest_with_zero_thrust():
    model = WAMV3DOF(payload=0.0)
    next_state = usv_wamv_kinematics(_rest_state(), np.zeros((2, 1)), 0.05, model)
    assert np.allclose(next_state, 0.0)


def test_equal_thrust_produces_surge():
    model = WAMV3DOF(payload=0.0)
    next_state = usv_wamv_kinematics(
        _rest_state(), np.array([[TWJ_MAX], [TWJ_MAX]]), 0.05, model
    )
    assert next_state[3, 0] > 0.0  # u increases
    assert abs(next_state[5, 0]) < 1e-9  # no yaw at rest + equal thrust
    assert next_state[0, 0] == pytest.approx(0.0)  # pose uses current nu (0)


def test_port_thrust_produces_positive_yaw():
    model = WAMV3DOF(payload=0.0)
    next_state = usv_wamv_kinematics(
        _rest_state(), np.array([[TWJ_MAX], [0.0]]), 0.05, model
    )
    assert next_state[5, 0] > 0.0  # r > 0
    assert next_state[3, 0] > 0.0  # some surge as well


def test_matches_project_simulator_one_step():
    """One Euler step at empty load must match usv_calib VariableLoadWAMV."""
    import sys
    from pathlib import Path

    sim_root = Path.home() / "Qsync" / "HNU" / "program" / "zero_shot" / "usv_calib"
    if not (sim_root / "simulator" / "wamv_3dof.py").is_file():
        pytest.skip("project WAM-V simulator not on this machine")

    sys.path.insert(0, str(sim_root))
    from simulator.wamv_3dof import VariableLoadWAMV

    vessel = VariableLoadWAMV(payload=0.0)
    x = np.array([1.0, 2.0, 0.3, 0.4, -0.1, 0.05], dtype=float)
    thrust = np.array([120.0, 80.0], dtype=float)
    x_ref = vessel.step(x, thrust, payload=0.0, dt=0.05)

    model = WAMV3DOF(payload=0.0)
    state = x.reshape(6, 1)
    x_irsim = usv_wamv_kinematics(state, thrust.reshape(2, 1), 0.05, model)
    assert np.allclose(x_irsim.flatten(), x_ref, atol=1e-10)


def test_handler_and_factory():
    handler = KinematicsFactory.create_kinematics(
        name="usv", payload=0.0, twj_max=TWJ_MAX
    )
    assert isinstance(handler, UsvKinematics)
    assert handler.state_dim == 6
    assert handler.action_dim == 2

    import irsim

    env = irsim.make("test_usv_world.yaml", save_ani=False, display=False)
    robot = env.robot
    assert robot.kinematics == "usv"
    assert robot.state.shape == (6, 1)
    env.step([100.0, 100.0])
    assert robot.state[3, 0] > 0.0
    env.end()


def test_payload_changes_mass():
    empty = coeffs_at_mass(M0)
    light = coeffs_at_mass(M_LIGHTSHIP)
    assert empty.m == pytest.approx(150.0)
    assert light.m == pytest.approx(220.0)
    assert light.T > empty.T

    handler = UsvKinematics("usv")
    handler.set_payload(70.0)
    assert handler.model.coeffs.m == pytest.approx(220.0)
