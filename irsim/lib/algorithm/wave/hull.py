"""Hull geometry for linear Froude-Krylov wave-force queries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


def _as_xy(value: list | tuple) -> tuple[float, float]:
    arr = tuple(float(v) for v in value)
    if len(arr) < 2:
        raise ValueError("sample point needs [x, y]")
    return arr[0], arr[1]


@dataclass(frozen=True)
class HullGeom:
    """Body-frame pressure stations and projected areas for one hull.

    Points are in the body frame (origin at CG, +x forward, +y to port).
    ``A_x`` / ``A_y`` / ``A_z`` are the frontal, lateral and waterplane
    areas used by the five-point Froude-Krylov difference.
    """

    name: str
    points: dict[str, tuple[float, float]]
    A_x: float
    A_y: float
    A_z: float
    T: float
    B: float
    A_x_ref: float = 0.0
    A_y_ref: float = 0.0
    T_ref: float = 0.0
    hull_type: str = "catamaran"
    labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.labels:
            object.__setattr__(self, "labels", tuple(self.points))
        if self.T_ref <= 0.0:
            object.__setattr__(self, "T_ref", float(self.T) if self.T > 0.0 else 1.0)
        if self.A_x_ref <= 0.0:
            object.__setattr__(self, "A_x_ref", float(self.A_x))
        if self.A_y_ref <= 0.0:
            object.__setattr__(self, "A_y_ref", float(self.A_y))

    def at_draft(self, T: float) -> HullGeom:
        """Scale submerged frontal/lateral area with draft; A_z unchanged."""
        T = float(T)
        scale = T / self.T_ref if self.T_ref > 1e-12 else 1.0
        return replace(self, T=T, A_x=self.A_x_ref * scale, A_y=self.A_y_ref * scale)

    def key(self) -> tuple:
        pts = tuple(
            sorted((k, round(v[0], 6), round(v[1], 6)) for k, v in self.points.items())
        )
        return (
            self.name,
            round(self.T, 6),
            round(self.A_x, 6),
            round(self.A_y, 6),
            round(self.B, 6),
            pts,
        )


def hull_from_mapping(
    name: str,
    mapping: dict,
    *,
    L: float,
    LWL: float,
    B: float,
    B_hull: float,
    LCG: float,
    AWP: float,
    T: float,
) -> HullGeom:
    """Build a HullGeom from a vessel YAML ``hull:`` block."""
    hull_type = str(mapping.get("type", "catamaran"))
    raw_pts = mapping.get("sample_points") or {}
    if raw_pts:
        points = {str(k): _as_xy(v) for k, v in raw_pts.items()}
    elif hull_type == "catamaran":
        points = {
            "cg": (0.0, 0.0),
            "bow": (float(L) - float(LCG), 0.0),
            "stern": (-float(LCG), 0.0),
            "port": (0.0, float(B) / 2.0),
            "stbd": (0.0, -float(B) / 2.0),
        }
    else:
        points = {
            "cg": (0.0, 0.0),
            "bow": (0.5 * float(L), 0.0),
            "stern": (-0.5 * float(L), 0.0),
            "port": (0.0, 0.5 * float(B)),
            "stbd": (0.0, -0.5 * float(B)),
        }

    T = max(float(T), 1e-6)
    A_x = mapping.get("A_x")
    A_y = mapping.get("A_y")
    A_z = mapping.get("A_z", AWP)
    if A_x is None:
        A_x = 2.0 * T * float(B_hull) if hull_type == "catamaran" else T * float(B)
    if A_y is None:
        A_y = 2.0 * T * float(LWL) if hull_type == "catamaran" else T * float(L)

    return HullGeom(
        name=name,
        points=points,
        A_x=float(A_x),
        A_y=float(A_y),
        A_z=float(A_z),
        T=T,
        B=float(B),
        A_x_ref=float(A_x),
        A_y_ref=float(A_y),
        T_ref=T,
        hull_type=hull_type,
    )
