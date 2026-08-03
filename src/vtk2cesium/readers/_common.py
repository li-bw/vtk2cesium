"""Shared helpers for non-VTI structured-grid readers.

All adapters emit the same contract as the VTK reader: a local ENU frame in
metres where ``origin`` is the south-west-bottom corner and ``spacing`` is the
per-axis cell size. The pipeline georeference anchors that local origin to the
globe, so adapters never need WGS84 coordinates themselves.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

# Average metres per degree of latitude / longitude (good enough for the local
# tangent-plane approximation the voxel pipeline already uses).
DEG_TO_M_LAT = 111_320.0
DEG_TO_M_LON = 111_320.0

_DEGREE_UNIT_KEYWORDS = (
    "deg",
    "degree",
    "degrees",
    "rad",
    "radian",
    "radians",
    "lat",
    "lon",
    "longitude",
    "latitude",
)


def is_degree_units(units: str | None) -> bool:
    """Heuristically decide whether a coordinate's units are angular degrees."""

    if not units:
        return False
    token = units.strip().lower()
    if not token:
        return False
    # Common angular unit spellings, including CF "degrees_*" prefixes.
    if token in ("deg", "degree", "degrees", "rad", "radian", "radians"):
        return True
    if token.startswith("degrees_") or token.startswith("degree_"):
        return True
    if token in ("lat", "latitude", "lon", "long", "longitude"):
        return True
    return False


def meters_per_degree(role: str, reference_latitude_deg: float) -> float:
    """Return metres per degree for the given ENU axis role at a reference lat."""

    if role == "x":  # east-west: depends on latitude
        return DEG_TO_M_LON * math.cos(math.radians(reference_latitude_deg))
    return DEG_TO_M_LAT  # north-south


def regular_spacing(coords: Iterable[float]) -> float:
    """Uniform spacing of a strictly monotonic coordinate, or 1.0 if degenerate.

    A non-uniform coordinate is reduced to its mean delta; callers document that
    the voxel pipeline assumes a regular grid.
    """

    array = np.asarray(list(coords), dtype=float)
    if array.size < 2:
        return 1.0
    deltas = np.diff(array)
    if not np.all(np.isfinite(deltas)):
        return 1.0
    return float(np.mean(np.abs(deltas)))


def make_bounds(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    point_dimensions: tuple[int, int, int],
) -> tuple[float, float, float, float, float, float]:
    """Axis-aligned bounds (xmin,xmax,ymin,ymax,zmin,zmax) in local ENU metres."""

    x_size, y_size, z_size = point_dimensions
    ox, oy, oz = origin
    sx, sy, sz = spacing
    return (
        ox,
        ox + (x_size - 1) * sx,
        oy,
        oy + (y_size - 1) * sy,
        oz,
        oz + (z_size - 1) * sz,
    )
