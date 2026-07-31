"""Atomic writer for one-tile or multi-level CesiumJS experimental voxel data sets."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from vtk2cesium.config import TilingConfig
from vtk2cesium.formats.subtree import (
    TileCoordinate,
    available_coordinates,
    build_subtree_payload,
    root_only_subtree,
)
from vtk2cesium.formats.tileset import build_voxel_tileset
from vtk2cesium.formats.voxel_gltf import encode_scalar_glb
from vtk2cesium.geo import GeoReference, local_box_from_bounds
from vtk2cesium.lod import build_lod_pyramid
from vtk2cesium.model import ScalarField, StructuredVoxelDataset
from vtk2cesium.transfer import ScalarPreprocessConfig, preprocess_scalar


class VoxelWriteError(ValueError):
    """Raised when a structured field cannot be written safely."""


def write_voxel_tileset(
    dataset: StructuredVoxelDataset,
    output: str | Path,
    *,
    field_name: str,
    georeference: GeoReference,
    preprocess: ScalarPreprocessConfig | None = None,
    tiling: TilingConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one structured field as an implicit voxel tileset.

    When ``tiling`` requests more than one level, the field is padded to a
    power-of-two ``capacity = tile_dim * 2**(levels-1)`` per axis and emitted as
    an implicit OCTREE with fixed-size tiles at every level. Otherwise the
    stage-4 single root tile is produced.

    All files are first written to a sibling temporary directory. The target is
    replaced only after every document and binary payload has been produced.
    """

    field = dataset.field(field_name)
    _require_scalar(field)
    processed = preprocess_scalar(field.values, preprocess)
    if not processed.mask.all():
        raise VoxelWriteError(
            "stage-3 FLOAT32 scalar output cannot encode a separate validity mask; "
            "choose a fill policy that keeps every value valid or wait for mask encoding"
        )
    values = processed.values
    if not np.isfinite(values).all():
        raise VoxelWriteError("encoded values must all be finite")
    output_range = processed.output_range
    if output_range is None:
        raise VoxelWriteError("encoded field has no finite values")

    destination = Path(output).expanduser().resolve()
    parent = destination.parent
    if not parent.exists():
        raise FileNotFoundError(f"output parent directory does not exist: {parent}")
    if destination.exists() and not destination.is_dir():
        raise FileNotFoundError(f"output path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {destination}")

    available_levels, tile_dimensions, capacity = _resolve_tiling(
        field.dimensions_xyz, tiling
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        if available_levels <= 1:
            _write_root_only(
                temporary,
                values=values,
                dimensions=field.dimensions_xyz,
                bounds=dataset.bounds,
                property_name=field.name,
                minimum=output_range[0],
                maximum=output_range[1],
                georeference=georeference,
            )
        else:
            _write_multilevel(
                temporary,
                values=values,
                tile_dimensions=tile_dimensions,
                capacity=capacity,
                available_levels=available_levels,
                origin=dataset.origin,
                spacing=dataset.spacing,
                bounds=dataset.bounds,
                property_name=field.name,
                minimum=output_range[0],
                maximum=output_range[1],
                georeference=georeference,
            )
        # Windows cannot always os.replace a non-empty directory, so replace the
        # target in place once the temporary payload is complete.
        if destination.exists():
            if any(destination.iterdir()):
                shutil.rmtree(destination)
            else:
                destination.rmdir()
        os.replace(temporary, destination)
        return destination / "tileset.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resolve_tiling(
    dimensions: tuple[int, int, int],
    tiling: TilingConfig | None,
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    """Return ``(available_levels, tile_dimensions, capacity)`` for the data."""

    if tiling is None or (
        tiling.available_levels is None and tiling.tile_dimensions is None
    ):
        return 1, dimensions, dimensions

    if tiling.available_levels is not None:
        levels = tiling.available_levels
        tile_dimensions = tuple(
            max(1, math.ceil(axis / (1 << (levels - 1)))) for axis in dimensions
        )
    else:
        tile_dimensions = tiling.tile_dimensions
        levels = 1 + max(
            math.ceil(math.log2(math.ceil(axis / tile)))
            for axis, tile in zip(dimensions, tile_dimensions)
        )
    capacity = tuple(tile * (1 << (levels - 1)) for tile in tile_dimensions)
    return levels, tile_dimensions, capacity


def _write_root_only(
    output: Path,
    *,
    values: np.ndarray,
    dimensions: tuple[int, int, int],
    bounds: tuple[float, float, float, float, float, float],
    property_name: str,
    minimum: float,
    maximum: float,
    georeference: GeoReference,
) -> None:
    content = output / "content"
    subtrees = output / "subtrees"
    content.mkdir(parents=True)
    subtrees.mkdir(parents=True)

    tileset = build_voxel_tileset(
        dimensions,
        property_name=property_name,
        minimum=minimum,
        maximum=maximum,
        bounding_box=local_box_from_bounds(bounds),
        transform=georeference.tileset_transform(),
    )
    _write_json(output / "tileset.json", tileset)
    _write_json(subtrees / "0.0.0.0.subtree", root_only_subtree())
    (content / "0.0.0.0.glb").write_bytes(
        encode_scalar_glb(values, property_name=property_name)
    )


def _write_multilevel(
    output: Path,
    *,
    values: np.ndarray,
    tile_dimensions: tuple[int, int, int],
    capacity: tuple[int, int, int],
    available_levels: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    bounds: tuple[float, float, float, float, float, float],
    property_name: str,
    minimum: float,
    maximum: float,
    georeference: GeoReference,
) -> None:
    content = output / "content"
    subtrees = output / "subtrees"
    content.mkdir(parents=True)
    subtrees.mkdir(parents=True)

    padded = _pad_to_capacity(values, capacity)
    mask = np.ones(padded.shape, dtype=bool)
    pyramid = build_lod_pyramid(padded, mask, available_levels=available_levels)

    tile_x, tile_y, tile_z = tile_dimensions
    coordinates: list[TileCoordinate] = []
    for level in range(available_levels):
        count = 1 << level
        for z in range(count):
            for y in range(count):
                for x in range(count):
                    block = pyramid[level].values[
                        z * tile_z : (z + 1) * tile_z,
                        y * tile_y : (y + 1) * tile_y,
                        x * tile_x : (x + 1) * tile_x,
                    ]
                    (content / f"{level}.{x}.{y}.{z}.glb").write_bytes(
                        encode_scalar_glb(block, property_name=property_name)
                    )
                    coordinates.append(TileCoordinate(level, x, y, z))

    subtree = build_subtree_payload(coordinates, subtree_levels=available_levels)
    _write_json(subtrees / "0.0.0.0.subtree", subtree.document)
    (subtrees / "0.0.0.0.bin").write_bytes(subtree.binary)

    # The transform carries no scale, so the root box must be expressed in
    # metric ENU units (origin + capacity * spacing), not voxel indices.
    ox, oy, oz = origin
    sx, sy, sz = spacing
    cx, cy, cz = capacity
    box_bounds = (
        ox,
        ox + cx * sx,
        oy,
        oy + cy * sy,
        oz,
        oz + cz * sz,
    )
    tileset = build_voxel_tileset(
        tile_dimensions,
        property_name=property_name,
        minimum=minimum,
        maximum=maximum,
        bounding_box=local_box_from_bounds(box_bounds),
        transform=georeference.tileset_transform(),
        subtree_levels=available_levels,
        available_levels=available_levels,
    )
    _write_json(output / "tileset.json", tileset)


def _pad_to_capacity(
    values: np.ndarray, capacity_xyz: tuple[int, int, int]
) -> np.ndarray:
    """Pad a (z, y, x) array with ``fill_value = 0.0`` to the capacity box."""

    cz, cy, cx = capacity_xyz[2], capacity_xyz[1], capacity_xyz[0]
    padded = np.zeros((cz, cy, cx), dtype=np.float32)
    vz, vy, vx = values.shape
    padded[:vz, :vy, :vx] = values
    return padded


def _require_scalar(field: ScalarField) -> None:
    if field.components != 1:
        raise VoxelWriteError(
            f"stage-3 writer supports one-component scalar fields; {field.name!r} "
            f"has {field.components} components"
        )


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
