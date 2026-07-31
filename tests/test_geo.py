import numpy as np
import pytest

from vtk2cesium.geo import (
    GeoReference,
    bounds_corners_enu,
    geodetic_to_ecef,
    local_box_from_bounds,
)


def test_known_wgs84_axes() -> None:
    assert geodetic_to_ecef(0.0, 0.0, 0.0) == pytest.approx((6_378_137.0, 0.0, 0.0))
    assert geodetic_to_ecef(90.0, 0.0, 0.0) == pytest.approx((0.0, 6_378_137.0, 0.0), abs=1e-8)
    assert geodetic_to_ecef(0.0, 90.0, 0.0) == pytest.approx(
        (0.0, 0.0, 6_356_752.314245179), abs=1e-8
    )


def test_enu_basis_is_orthonormal_and_right_handed() -> None:
    basis = GeoReference(116.3913, 39.9075, 1200.0).enu_basis
    east, north, up = basis[:, 0], basis[:, 1], basis[:, 2]

    assert basis.T @ basis == pytest.approx(np.eye(3), abs=1e-14)
    assert np.cross(east, north) == pytest.approx(up, abs=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.0, abs=1e-14)


def test_tileset_transform_is_column_major() -> None:
    reference = GeoReference(0.0, 0.0, 0.0)

    assert reference.tileset_transform() == pytest.approx(
        (
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            6_378_137.0,
            0.0,
            0.0,
            1.0,
        )
    )


def test_transform_points_maps_enu_axes() -> None:
    reference = GeoReference(0.0, 0.0, 0.0)
    origin = np.array(reference.ecef_origin)
    transformed = reference.transform_points(np.eye(3))

    assert transformed[0] - origin == pytest.approx((0.0, 1.0, 0.0))
    assert transformed[1] - origin == pytest.approx((0.0, 0.0, 1.0))
    assert transformed[2] - origin == pytest.approx((1.0, 0.0, 0.0))


def test_bounds_corners_cover_each_minimum_and_maximum() -> None:
    corners = bounds_corners_enu((1.0, 3.0, 2.0, 6.0, -1.0, 4.0))

    assert corners.shape == (8, 3)
    assert np.array_equal(corners.min(axis=0), (1.0, 2.0, -1.0))
    assert np.array_equal(corners.max(axis=0), (3.0, 6.0, 4.0))


def test_local_tileset_box_reconstructs_all_bounds_corners() -> None:
    bounds = (1.0, 3.0, 2.0, 6.0, -1.0, 4.0)
    box = local_box_from_bounds(bounds)
    center = np.array(box[:3])
    axes = np.array(box[3:]).reshape((3, 3))
    reconstructed = np.array(
        [
            center + sx * axes[0] + sy * axes[1] + sz * axes[2]
            for sz in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sx in (-1.0, 1.0)
        ]
    )

    assert reconstructed == pytest.approx(bounds_corners_enu(bounds))


def test_tileset_box_and_transform_cover_ecef_corners() -> None:
    reference = GeoReference(116.3913, 39.9075, 1200.0)
    bounds = (0.0, 176.0, 0.0, 94.0, 0.0, 47.0)
    box = local_box_from_bounds(bounds)
    center = np.array(box[:3])
    axes = np.array(box[3:]).reshape((3, 3))
    local_corners = np.array(
        [
            center + sx * axes[0] + sy * axes[1] + sz * axes[2]
            for sz in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sx in (-1.0, 1.0)
        ]
    )
    transform = np.array(reference.tileset_transform()).reshape((4, 4), order="F")
    homogeneous = np.column_stack((local_corners, np.ones(8)))
    transformed_by_tileset = (transform @ homogeneous.T).T[:, :3]

    assert transformed_by_tileset == pytest.approx(reference.transform_points(local_corners))


def test_invalid_geographic_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="longitude"):
        GeoReference(181.0, 0.0)
    with pytest.raises(ValueError, match="latitude"):
        GeoReference(0.0, -91.0)
