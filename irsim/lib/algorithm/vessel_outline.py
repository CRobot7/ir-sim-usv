"""Top-view plot shapes generated from a vessel YAML.

``hull.type`` is ``catamaran`` (two pontoons + crossbeams) or ``monohull``
(one pointed hull). ``board`` is treated as a monohull alias. Dimensions
come from ``geometry`` (``L``, ``LCG``, ``B``, ``B_hull`` / ``BOA``).
"""

from __future__ import annotations

from typing import Any

from irsim.lib.algorithm.vessel_spec import VesselSpec


def hull_kind(spec: VesselSpec) -> str:
    """Return ``catamaran`` or ``monohull`` from ``hull.type``."""
    raw = str(
        (spec.hull_mapping or {}).get("type")
        or getattr(spec.hull, "hull_type", "")
        or "catamaran"
    ).lower()
    if raw in ("monohull", "board"):
        return "monohull"
    return "catamaran"


def outline_shape(spec: VesselSpec) -> dict[str, Any]:
    """Build an ir-sim ``shape`` dict (compound) for top-view rendering."""
    if hull_kind(spec) == "monohull":
        return {"name": "compound", "parts": [_monohull_polygon(spec)]}
    return {"name": "compound", "parts": _catamaran_parts(spec)}


def _catamaran_parts(spec: VesselSpec) -> list[dict[str, Any]]:
    """Two pointed pontoons plus deck / aft / forward crossbeams.

    Scale factors match the former hand-written USV14 compound in
    ``usage/27usv_world/usv_world.yaml`` (L=4.29, B=1.83, B_hull=0.37).
    """
    length = float(spec.L)
    b = float(spec.B)
    bh = float(spec.B_hull)
    boa = float(spec.BOA)
    lcg = float(spec.LCG)
    stern_x = -lcg
    bow_x = length - lcg
    taper = 0.16317 * length
    taper_x = bow_x - taper
    half_b = 0.5 * b
    half_bh = 0.5 * bh
    inner = half_b - half_bh
    outer = half_b + half_bh
    port = [
        [stern_x, inner],
        [taper_x, inner],
        [bow_x, half_b],
        [taper_x, outer],
        [stern_x, outer],
    ]
    stbd = [
        [stern_x, -outer],
        [taper_x, -outer],
        [bow_x, -half_b],
        [taper_x, -inner],
        [stern_x, -inner],
    ]
    deck_w = 0.6818 * boa
    return [
        {"name": "polygon", "vertices": port},
        {"name": "polygon", "vertices": stbd},
        {
            "name": "rectangle",
            "length": 0.4196 * length,
            "width": deck_w,
            "pose": [0.0350 * length, 0.0, 0.0],
        },
        {
            "name": "rectangle",
            "length": 0.0932 * length,
            "width": b,
            "pose": [-lcg + 0.0466 * length, 0.0, 0.0],
        },
        {
            "name": "rectangle",
            "length": 0.0653 * length,
            "width": deck_w,
            "pose": [0.3147 * length, 0.0, 0.0],
        },
    ]


def _monohull_polygon(spec: VesselSpec) -> dict[str, Any]:
    """Single transom hull with a pointed bow, envelope ``L x BOA``."""
    length = float(spec.L)
    boa = float(spec.BOA)
    lcg = float(spec.LCG)
    stern_x = -lcg
    bow_x = length - lcg
    half_b = 0.5 * boa
    taper = 0.22 * length
    taper_x = bow_x - taper
    return {
        "name": "polygon",
        "vertices": [
            [stern_x, -half_b],
            [taper_x, -half_b],
            [bow_x, 0.0],
            [taper_x, half_b],
            [stern_x, half_b],
        ],
    }
