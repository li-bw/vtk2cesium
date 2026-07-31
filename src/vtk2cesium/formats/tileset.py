"""Build a minimal CesiumJS-compatible experimental voxel tileset."""

from __future__ import annotations

from typing import Any, Sequence


VOXEL_CONTENT_EXTENSION = "3DTILES_content_voxels"


def build_voxel_tileset(
    dimensions: tuple[int, int, int],
    *,
    property_name: str,
    minimum: float,
    maximum: float,
    bounding_box: Sequence[float],
    transform: Sequence[float] | None = None,
    content_uri: str = "content/{level}.{x}.{y}.{z}.glb",
    subtree_uri: str = "subtrees/{level}.{x}.{y}.{z}.subtree",
    subtree_levels: int = 1,
    available_levels: int = 1,
) -> dict[str, Any]:
    """Build a georeference-ready implicit voxel tileset.

    ``subtree_levels`` and ``available_levels`` default to a single root tile.
    For multi-level output pass the implicit OCTREE depth so the tileset can be
    traversed by CesiumJS.
    """

    x_size, y_size, z_size = dimensions
    if min(dimensions) <= 0:
        raise ValueError("dimensions must be positive")
    if not property_name or not property_name.isidentifier():
        raise ValueError("property_name must be a non-empty identifier")
    if len(bounding_box) != 12:
        raise ValueError("bounding_box must contain 12 values")
    if transform is not None and len(transform) != 16:
        raise ValueError("transform must contain 16 column-major values")
    if subtree_levels < 1 or available_levels < 1:
        raise ValueError("implicit tiling levels must be positive")
    if subtree_levels > available_levels:
        raise ValueError("subtreeLevels must not exceed availableLevels")

    root: dict[str, Any] = {
        "boundingVolume": {"box": [float(value) for value in bounding_box]},
        "geometricError": 0.0,
        "refine": "REPLACE",
        "content": {
            "uri": content_uri,
            "extensions": {
                VOXEL_CONTENT_EXTENSION: {
                    "dimensions": [x_size, y_size, z_size],
                    "class": "voxel",
                }
            },
        },
        "implicitTiling": {
            "subdivisionScheme": "OCTREE",
            "subtreeLevels": subtree_levels,
            "availableLevels": available_levels,
            "subtrees": {"uri": subtree_uri},
        },
    }
    if transform is not None:
        root["transform"] = [float(value) for value in transform]

    return {
        "asset": {"version": "1.1", "generator": "vtk2cesium/0.1"},
        "schema": {
            "id": "vtk2cesium-voxel",
            "classes": {
                "voxel": {
                    "properties": {
                        property_name: {
                            "type": "SCALAR",
                            "componentType": "FLOAT32",
                        }
                    }
                }
            },
        },
        "statistics": {
            "classes": {
                "voxel": {
                    "count": x_size * y_size * z_size,
                    "properties": {property_name: {"min": minimum, "max": maximum}},
                }
            }
        },
        "geometricError": 0.0,
        "root": root,
        "extensionsUsed": [VOXEL_CONTENT_EXTENSION],
        "extensionsRequired": [VOXEL_CONTENT_EXTENSION],
    }


def build_probe_tileset(
    dimensions: tuple[int, int, int],
    *,
    property_name: str,
    minimum: float,
    maximum: float,
) -> dict[str, Any]:
    """Build a root-only implicit voxel tileset using a local unit box."""

    return build_voxel_tileset(
        dimensions,
        property_name=property_name,
        minimum=minimum,
        maximum=maximum,
        bounding_box=(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
        content_uri="tiles/{level}/{x}/{y}/{z}.gltf",
        subtree_uri="subtrees/{level}/{x}/{y}/{z}.json",
    )
