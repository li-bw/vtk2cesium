"""Explicit WGS84 georeferencing for local VTK ENU coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import numpy.typing as npt

WGS84_SEMI_MAJOR_AXIS = 6_378_137.0
WGS84_INVERSE_FLATTENING = 298.257_223_563
WGS84_FLATTENING = 1.0 / WGS84_INVERSE_FLATTENING
WGS84_FIRST_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GeoReference:
    """A WGS84 anchor whose local axes are East, North, and Up.

    Longitude and latitude are degrees. Height is ellipsoidal height in metres,
    not orthometric height above mean sea level.
    """

    longitude: float
    latitude: float
    height: float = 0.0

    def __post_init__(self) -> None:
        coordinates = (self.longitude, self.latitude, self.height)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("longitude, latitude, and height must be finite")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180] degrees")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90] degrees")

    @property
    def ecef_origin(self) -> tuple[float, float, float]:
        """Return the WGS84 anchor in Earth-centred Earth-fixed metres."""

        return geodetic_to_ecef(self.longitude, self.latitude, self.height)

    @property
    def enu_basis(self) -> Float64Array:
        """Return a 3x3 matrix whose columns are East, North, and Up in ECEF."""

        longitude = math.radians(self.longitude)
        latitude = math.radians(self.latitude)
        sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
        sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
        return np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ],
            dtype=np.float64,
        )

    @property
    def enu_to_ecef_matrix(self) -> Float64Array:
        """Return a 4x4 matrix for ``ecef = matrix @ [east,north,up,1]``."""

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.enu_basis
        matrix[:3, 3] = self.ecef_origin
        return matrix

    def tileset_transform(self) -> tuple[float, ...]:
        """Return the transform as the 16 column-major values required by 3D Tiles."""

        return tuple(float(value) for value in self.enu_to_ecef_matrix.flatten(order="F"))

    def transform_points(self, points_enu: Iterable[Iterable[float]]) -> Float64Array:
        """Transform an ``(n, 3)`` collection of local ENU points to ECEF."""

        points = np.asarray(points_enu, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_enu must have shape (n, 3)")
        if not np.isfinite(points).all():
            raise ValueError("points_enu must contain only finite coordinates")
        return points @ self.enu_basis.T + np.asarray(self.ecef_origin)


def geodetic_to_ecef(
    longitude: float,
    latitude: float,
    height: float = 0.0,
) -> tuple[float, float, float]:
    """Convert WGS84 longitude, latitude, and ellipsoidal height to ECEF metres."""

    reference = (float(longitude), float(latitude), float(height))
    if not all(math.isfinite(value) for value in reference):
        raise ValueError("longitude, latitude, and height must be finite")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be in [-180, 180] degrees")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be in [-90, 90] degrees")

    lon = math.radians(longitude)
    lat = math.radians(latitude)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    prime_vertical_radius = WGS84_SEMI_MAJOR_AXIS / math.sqrt(
        1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat
    )
    x = (prime_vertical_radius + height) * cos_lat * math.cos(lon)
    y = (prime_vertical_radius + height) * cos_lat * math.sin(lon)
    z = (
        prime_vertical_radius * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED) + height
    ) * sin_lat
    return float(x), float(y), float(z)


def local_box_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[float, ...]:
    """Build a 3D Tiles BOX in local ENU coordinates from axis-aligned bounds."""

    xmin, xmax, ymin, ymax, zmin, zmax = _validated_bounds(bounds)
    center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    half = ((xmax - xmin) / 2.0, (ymax - ymin) / 2.0, (zmax - zmin) / 2.0)
    return (
        *center,
        half[0],
        0.0,
        0.0,
        0.0,
        half[1],
        0.0,
        0.0,
        0.0,
        half[2],
    )


def bounds_corners_enu(
    bounds: tuple[float, float, float, float, float, float],
) -> Float64Array:
    """Return all eight corners of local ``(xmin,xmax,ymin,ymax,zmin,zmax)`` bounds."""

    xmin, xmax, ymin, ymax, zmin, zmax = _validated_bounds(bounds)
    return np.array(
        [
            (x, y, z)
            for z in (zmin, zmax)
            for y in (ymin, ymax)
            for x in (xmin, xmax)
        ],
        dtype=np.float64,
    )


def _validated_bounds(
    bounds: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    if len(bounds) != 6 or not all(math.isfinite(value) for value in bounds):
        raise ValueError("bounds must contain six finite values")
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if xmin > xmax or ymin > ymax or zmin > zmax:
        raise ValueError("bounds minima must not exceed maxima")
    return xmin, xmax, ymin, ymax, zmin, zmax
