"""Encode one scalar voxel tile as glTF 2.0 plus an external binary buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vtk2cesium.array_order import cesium_box_gltf_dimensions, zyx_to_cesium_buffer
from vtk2cesium.formats.glb import build_glb

VOXEL_MODE = 2_147_483_647


@dataclass(frozen=True)
class GltfTile:
    """A JSON glTF document and its external binary buffer."""

    document: dict[str, Any]
    binary: bytes


def encode_scalar_tile(
    values: np.ndarray,
    *,
    property_name: str = "density",
    buffer_uri: str | None = "density.bin",
) -> GltfTile:
    """Encode a `(z, y, x)` float array for CesiumJS 1.143 voxel loading.

    The binary buffer is flattened with X as the fastest-moving coordinate,
    followed by Y and then Z. Cesium's official BOX sample swaps the Y and Z
    dimensions in `EXT_primitive_voxels` because glTF is Y-up.
    """

    array = np.asarray(values, dtype="<f4")
    if array.ndim != 3:
        raise ValueError("values must be a three-dimensional array with shape (z, y, x)")
    if any(size <= 0 for size in array.shape):
        raise ValueError("all voxel dimensions must be positive")
    if not np.isfinite(array).all():
        raise ValueError("probe values must all be finite")
    if not property_name or not property_name.isidentifier():
        raise ValueError("property_name must be a non-empty identifier")

    z_size, y_size, x_size = map(int, array.shape)
    flattened = np.ascontiguousarray(array).ravel(order="C")
    payload = zyx_to_cesium_buffer(array, dtype="<f4")
    minimum = float(flattened.min())
    maximum = float(flattened.max())

    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "vtk2cesium/0.1"},
        "extensionsUsed": ["EXT_primitive_voxels", "EXT_structural_metadata"],
        "extensionsRequired": ["EXT_primitive_voxels", "EXT_structural_metadata"],
        "extensions": {
            "EXT_structural_metadata": {
                "schema": {
                    "classes": {
                        "voxel": {
                            "properties": {
                                property_name: {
                                    "type": "SCALAR",
                                    "componentType": "FLOAT32",
                                }
                            }
                        }
                    }
                },
                "propertyAttributes": [
                    {
                        "class": "voxel",
                        "properties": {property_name: {"attribute": "_DATA"}},
                    }
                ],
            }
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"_DATA": 0},
                        "mode": VOXEL_MODE,
                        "extensions": {
                            "EXT_primitive_voxels": {
                                "shape": 0,
                                "dimensions": list(
                                    cesium_box_gltf_dimensions((x_size, y_size, z_size))
                                ),
                            },
                            "EXT_structural_metadata": {"propertyAttributes": [0]},
                        },
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(flattened.size),
                "min": [minimum],
                "max": [maximum],
                "type": "SCALAR",
            }
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(payload)}],
        "buffers": [{"byteLength": len(payload)}],
    }
    if buffer_uri is not None:
        document["buffers"][0]["uri"] = buffer_uri
    return GltfTile(document=document, binary=payload)


def encode_scalar_glb(
    values: np.ndarray,
    *,
    property_name: str = "density",
) -> bytes:
    """Encode one finite `(z, y, x)` scalar tile as a self-contained GLB."""

    tile = encode_scalar_tile(values, property_name=property_name, buffer_uri=None)
    return build_glb(tile.document, tile.binary)
