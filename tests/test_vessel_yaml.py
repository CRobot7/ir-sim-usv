"""Vessel YAML loader and WAM-V coefficient regression."""

import numpy as np
import pytest

from irsim.lib.algorithm.usv_wamv import WAMV3DOF, Fossen3DOF
from irsim.lib.algorithm.vessel_outline import hull_kind, outline_shape
from irsim.lib.algorithm.vessel_spec import load_vessel
from irsim.lib.algorithm.wamv_params import M0, M_LIGHTSHIP, coeffs_at_mass
from irsim.lib.handler.kinematics_handler import UsvKinematics


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
    assert spec.twj_min == pytest.approx(0.0)


def test_wamv_clips_negative_thrust_to_zero():
    spec = load_vessel("wamv")
    assert spec.twj_min == pytest.approx(0.0)
    model = Fossen3DOF(vessel="wamv")
    tau = model.tau_from_thrust(np.array([-80.0, -80.0]))
    assert tau[0] == pytest.approx(0.0)
    handler = UsvKinematics("usv", vessel="wamv")
    assert handler.vel_min == [0.0, 0.0]
    assert handler.vel_max[0] == pytest.approx(spec.twj_max)
    assert handler.vel_max[1] == pytest.approx(spec.twj_max)


def test_wamv16_allows_reverse_thrust():
    spec = load_vessel("wamv16")
    assert spec.twj_min == pytest.approx(-500.0)
    assert spec.twj_max == pytest.approx(1000.0)
    assert pytest.approx(5.0) == spec.L
    c = spec.coeffs_at_mass(400.0)
    assert c.m == pytest.approx(400.0)
    assert c.I_z == pytest.approx(450.0)
    model = Fossen3DOF(vessel="wamv16")
    tau = model.tau_from_thrust(np.array([-200.0, -200.0]))
    assert tau[0] == pytest.approx(-400.0)
    handler = UsvKinematics("usv", vessel="wamv16")
    assert handler.vel_min == [-500.0, -500.0]
    assert handler.vel_max == [1000.0, 1000.0]


def _polygon_count(shape: dict) -> int:
    return sum(1 for p in shape["parts"] if p.get("name") == "polygon")


def test_outline_catamaran_has_two_pontoons():
    spec = load_vessel("wamv")
    assert hull_kind(spec) == "catamaran"
    shape = outline_shape(spec)
    assert shape["name"] == "compound"
    assert _polygon_count(shape) >= 2
    port = shape["parts"][0]["vertices"]
    assert port[0][0] == pytest.approx(-1.27, abs=1e-2)
    spec16 = load_vessel("wamv16")
    shape16 = outline_shape(spec16)
    assert _polygon_count(shape16) >= 2
    xs14 = [v[0] for v in shape["parts"][0]["vertices"]]
    xs16 = [v[0] for v in shape16["parts"][0]["vertices"]]
    ys14 = [v[1] for v in shape["parts"][0]["vertices"]]
    ys16 = [v[1] for v in shape16["parts"][0]["vertices"]]
    assert max(xs16) - min(xs16) > max(xs14) - min(xs14)
    assert max(ys16) > max(ys14)


def test_outline_monohull_has_one_hull():
    spec = load_vessel("monohull")
    assert hull_kind(spec) == "monohull"
    shape = outline_shape(spec)
    assert shape["name"] == "compound"
    assert _polygon_count(shape) == 1
    assert len(shape["parts"]) == 1
