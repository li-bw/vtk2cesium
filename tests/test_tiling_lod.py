import numpy as np
import pytest

from vtk2cesium.lod import build_lod_pyramid, downsample_2x2x2
from vtk2cesium.tiling import (
    TileCoordinate,
    iter_fixed_tiles,
    padded_dimensions,
    required_available_levels,
    tile_grid_shape,
)


def test_tile_grid_and_required_levels() -> None:
    assert tile_grid_shape((7, 5, 3), (4, 4, 4)) == (2, 2, 1)
    assert required_available_levels((7, 5, 3), (4, 4, 4)) == 2
    assert padded_dimensions((4, 4, 4), 2) == (8, 8, 8)


def test_fixed_tiles_pad_boundary_and_preserve_validity() -> None:
    values = np.arange(3 * 5 * 7, dtype=np.float32).reshape((3, 5, 7))
    mask = np.ones(values.shape, dtype=bool)
    tiles = list(
        iter_fixed_tiles(
            values,
            mask,
            level=1,
            tile_dimensions_xyz=(4, 4, 4),
            fill_value=-1.0,
        )
    )

    assert [tile.coordinate for tile in tiles] == [
        TileCoordinate(1, 0, 0, 0),
        TileCoordinate(1, 1, 0, 0),
        TileCoordinate(1, 0, 1, 0),
        TileCoordinate(1, 1, 1, 0),
    ]
    edge = tiles[-1]
    assert edge.values.shape == (4, 4, 4)
    assert edge.valid_shape_zyx == (3, 1, 3)
    assert edge.validity.sum() == 9
    assert np.all(edge.values[3] == -1.0)


def test_downsample_averages_only_valid_values_and_handles_odd_edges() -> None:
    values = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    mask = np.ones(values.shape, dtype=bool)
    mask[0, 0, 0] = False
    result = downsample_2x2x2(values, mask, fill_value=-1.0)

    assert result.values.shape == (2, 2, 2)
    assert result.validity.all()
    expected = np.mean(values[0:2, 0:2, 0:2][mask[0:2, 0:2, 0:2]])
    assert result.values[0, 0, 0] == pytest.approx(expected)
    assert result.values[-1, -1, -1] == pytest.approx(26.0)


def test_lod_pyramid_is_coarse_to_fine() -> None:
    values = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
    mask = np.ones(values.shape, dtype=bool)
    pyramid = build_lod_pyramid(values, mask, available_levels=3)

    assert [level.values.shape for level in pyramid] == [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
    assert pyramid[0].values[0, 0, 0] == pytest.approx(values.mean())
    assert np.array_equal(pyramid[-1].values, values)


def test_invalid_tile_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        list(
            iter_fixed_tiles(
                np.zeros((4, 4, 4)),
                np.ones((4, 4, 4), dtype=bool),
                level=0,
                tile_dimensions_xyz=(2, 2, 2),
            )
        )
