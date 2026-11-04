"""Vessel YAML loader and WAM-V coefficient regression."""

import numpy as np
import pytest

from irsim.lib.algorithm.usv_wamv import WAMV3DOF, Fossen3DOF
from irsim.lib.algorithm.vessel_spec import load_vessel
from irsim.lib.algorithm.wamv_params import M0, M_LIGHTSHIP, coeffs_at_mass


def test_wamv_yaml_matches_klinger_nodes():
    spec = load_vessel("wamv")
    assert spec.name == "wamv"
    assert spec.recipe == "klinger2017"
    empty = spec.coeffs_at_mass(M0)
    assert empty.m == pytest.approx(150.0)
    assert empty.X_u == pytest.approx(50.897)
    assert empty.X_uu == pytest.approx(5.8722)
    light = spec.coeffs_at_mass(M_LIGHTSHIP)
    assert light.m == pytest.approx(220.0)
    assert light.T > empty.T
    assert light.X_u == pytest.approx(55.771)


def test_wamv_params_shim_matches_loader():
    spec = load_vessel("wamv")
    a = spec.coeffs_at_mass(150.0)
    b = coeffs_at_mass(150.0)
    assert a.m == pytest.approx(b.m)
    assert a.X_u == pytest.approx(b.X_u)
    assert a.Y_vdot == pytest.approx(b.Y_vdot)
    assert a.I_z == pytest.approx(b.I_z)


def test_fossen_alias_and_hull():
    model = WAMV3DOF(payload=0.0)
    assert isinstance(model, Fossen3DOF)
    assert model.hull.name == "wamv"
    assert "bow" in model.hull.points
    assert model.hull.T > 0.0
    heavier = Fossen3DOF(vessel="wamv", payload=70.0)
    assert heavier.coeffs.m == pytest.approx(220.0)
    assert heavier.hull.T > model.hull.T


def test_explicit_hydro_without_recipe(tmp_path):
    yaml_text = """
name: stub
geometry:
  L: 3.0
  LWL: 2.8
  BOA: 0.8
  B: 0.8
  LCG: 1.5
  AWP: 2.0
  T_aft_lightship: 0.04
  T_mid_lightship: 0.04
  rho: 1025.0
  Cd: 1.1
load:
  M_slick: 25.0
  M_lightship: 25.0
  M_full_paper: 25.0
  P_full: 0.0
hydro:
  coeffs:
    m: 25.0
    I_z: 5.0
    T: 0.04
    X_u: 2.0
    X_uu: 8.0
    X_udot: -4.0
    Y_vdot: -15.0
    Y_rdot: -1.0
    N_vdot: -1.0
    N_rdot: -2.0
    Y_vv: -10.0
    Y_vr: 0.0
    Y_rv: 0.0
    Y_rr: -2.0
    N_vv: 0.0
    N_vr: -2.0
    N_rv: -2.0
    N_rr: -1.0
hull:
  type: board
  sample_points:
    cg: [0.0, 0.0]
    bow: [1.5, 0.0]
    stern: [-1.5, 0.0]
    port: [0.0, 0.4]
    stbd: [0.0, -0.4]
actuators:
  type: dual_waterjet
  twj_max: 40.0
"""
    path = tmp_path / "stub.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    spec = load_vessel(str(path))
    assert spec.recipe is None
    c = spec.coeffs_at_mass(25.0)
    assert c.m == pytest.approx(25.0)
    assert c.X_u == pytest.approx(2.0)
    assert c.linear_drag_model == "none"
    model = Fossen3DOF(spec=spec)
    _eta, nu_next = model.step_eta_nu(
        np.zeros(3), np.zeros(3), np.array([10.0, 10.0]), 0.05
    )
    assert nu_next[0] > 0.0  # surge u increases
