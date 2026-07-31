"""Fixed-size structured voxel tiling with deterministic edge padding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, order=True)
class TileCoordinate:
    """Implicit OCTREE tile coordinate."""

    level: int
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if min(self.level, self.x, self.y, self.z) < 0:
            raise ValueError("tile coordinate components must be non-negative")
        limit = 1 << self.level
        if self.x >= limit or self.y >= limit or self.z >= limit:
            raise ValueError("tile coordinate exceeds its implicit level")


@dataclass(frozen=True)
class VoxelTile:
    """One fixed-size tile and its aligned validity mask."""

    coordinate: TileCoordinate
    values: Float32Array
    validity: BoolArray
    valid_shape_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        validity = np.ascontiguousarray(self.validity, dtype=np.bool_)
        if values.ndim != 3 or values.shape != validity.shape:
            raise ValueError("tile values and validity must have the same 3D shape")
        if any(size <= 0 for size in values.shape):
            raise ValueError("tile dimensions must be positive")
        if any(valid <= 0 or valid > size for valid, size in zip(self.valid_shape_zyx, values.shape)):
            raise ValueError("valid_shape_zyx must fit inside tile values")
        values.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)

    @property
    def dimensions_xyz(self) -> tuple[int, int, int]:
        z_size, y_size, x_size = self.values.shape
        return x_size, y_size, z_size


def tile_grid_shape(
    dimensions_xyz: tuple[int, int, int],
    tile_dimensions_xyz: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the number of tiles required along X, Y, and Z."""

    _positive_triplet(dimensions_xyz, "dimensions_xyz")
    _positive_triplet(tile_dimensions_xyz, "tile_dimensions_xyz")
    return tuple(
        math.ceil(size / tile_size)
        for size, tile_size in zip(dimensions_xyz, tile_dimensions_xyz)
    )  # type: ignore[return-value]


def required_available_levels(
    dimensions_xyz: tuple[int, int, int],
    tile_dimensions_xyz: tuple[int, int, int],
) -> int:
    """Return levels needed for one root tile and full-resolution leaf tiles."""

    grid = tile_grid_shape(dimensions_xyz, tile_dimensions_xyz)
    leaf_level = max((size - 1).bit_length() for size in grid)
    return leaf_level + 1


def padded_dimensions(
    tile_dimensions_xyz: tuple[int, int, int],
    available_levels: int,
) -> tuple[int, int, int]:
    """Return leaf capacity covered by the implicit root bounds."""

    _positive_triplet(tile_dimensions_xyz, "tile_dimensions_xyz")
    if available_levels <= 0:
        raise ValueError("available_levels must be positive")
    scale = 1 << (available_levels - 1)
    return tuple(size * scale for size in tile_dimensions_xyz)  # type: ignore[return-value]


def iter_fixed_tiles(
    values: npt.ArrayLike,
    validity: npt.ArrayLike,
    *,
    level: int,
    tile_dimensions_xyz: tuple[int, int, int],
    fill_value: float = 0.0,
) -> Iterator[VoxelTile]:
    """Yield available fixed-size tiles in deterministic Z/Y/X order."""

    source = np.asarray(values, dtype=np.float32)
    mask = np.asarray(validity, dtype=np.bool_)
    if source.ndim != 3 or source.shape != mask.shape:
        raise ValueError("values and validity must have the same 3D shape")
    _positive_triplet(tile_dimensions_xyz, "tile_dimensions_xyz")
    if not math.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    tile_x, tile_y, tile_z = tile_dimensions_xyz
    z_size, y_size, x_size = source.shape
    grid_x, grid_y, grid_z = tile_grid_shape((x_size, y_size, z_size), tile_dimensions_xyz)
    limit = 1 << level
    if max(grid_x, grid_y, grid_z) > limit:
        raise ValueError("tile grid does not fit the requested implicit level")

    for z in range(grid_z):
        for y in range(grid_y):
            for x in range(grid_x):
                x0, y0, z0 = x * tile_x, y * tile_y, z * tile_z
                x1, y1, z1 = min(x0 + tile_x, x_size), min(y0 + tile_y, y_size), min(
                    z0 + tile_z, z_size
                )
                chunk_mask = mask[z0:z1, y0:y1, x0:x1]
                if not chunk_mask.any():
                    continue
                chunk = source[z0:z1, y0:y1, x0:x1]
                padded_values = np.full((tile_z, tile_y, tile_x), fill_value, dtype=np.float32)
                padded_mask = np.zeros((tile_z, tile_y, tile_x), dtype=np.bool_)
                valid_shape = chunk.shape
                padded_values[: valid_shape[0], : valid_shape[1], : valid_shape[2]] = chunk
                padded_mask[: valid_shape[0], : valid_shape[1], : valid_shape[2]] = chunk_mask
                padded_values[~padded_mask] = fill_value
                yield VoxelTile(
                    coordinate=TileCoordinate(level, x, y, z),
                    values=padded_values,
                    validity=padded_mask,
                    valid_shape_zyx=valid_shape,
                )


def _positive_triplet(values: tuple[int, int, int], name: str) -> None:
    if len(values) != 3 or min(values) <= 0:
        raise ValueError(f"{name} must contain three positive integers")
