"""Offline validation for CesiumJS experimental voxel tilesets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vtk2cesium.formats.glb import parse_glb
from vtk2cesium.formats.subtree import TileCoordinate, available_coordinates

VOXEL_EXT = "3DTILES_content_voxels"


@dataclass(frozen=True)
class ValidationResult:
    tileset: Path
    dimensions: tuple[int, int, int]
    property_name: str
    value_count: int
    minimum: float
    maximum: float


def validate_probe(path: Path) -> ValidationResult:
    """Validate all files referenced by a root-only or multi-level tileset."""

    tileset_path = path / "tileset.json" if path.is_dir() else path
    base = tileset_path.parent
    tileset = _read_json(tileset_path)

    _require(tileset.get("asset", {}).get("version") == "1.1", "asset.version must be 1.1")
    _require(VOXEL_EXT in tileset.get("extensionsRequired", []), f"{VOXEL_EXT} must be required")
    root = _object(tileset, "root")
    content = _object(root, "content")
    voxel = _object(_object(content, "extensions"), VOXEL_EXT)
    dimensions = _dimensions(voxel.get("dimensions"), "tileset voxel dimensions")
    class_name = voxel.get("class")
    _require(isinstance(class_name, str), "voxel class must be a string")

    implicit = _object(root, "implicitTiling")
    _require(implicit.get("subdivisionScheme") == "OCTREE", "subdivisionScheme must be OCTREE")
    subtree_levels = int(implicit.get("subtreeLevels", 1))
    available_levels = int(implicit.get("availableLevels", 1))
    _require(subtree_levels >= 1 and available_levels >= 1, "implicit levels must be positive")
    _require(subtree_levels <= available_levels, "subtreeLevels must not exceed availableLevels")

    subtree_uri = _object(implicit, "subtrees")["uri"]
    subtree_path = base / _expand_uri(subtree_uri)
    subtree = _read_json(subtree_path)
    expected_gltf_dimensions = (dimensions[0], dimensions[2], dimensions[1])

    content_coordinates = _content_coordinates(subtree, subtree_path.parent, subtree_levels)
    _require(len(content_coordinates) > 0, "no available content tiles declared")
    _require(
        all(coord.level < available_levels for coord in content_coordinates),
        "content coordinate exceeds availableLevels",
    )

    overall_minimum = float("inf")
    overall_maximum = float("-inf")
    value_count = 0
    property_name: str | None = None
    for coordinate in content_coordinates:
        tile_path = base / content["uri"].format(
            level=coordinate.level, x=coordinate.x, y=coordinate.y, z=coordinate.z
        )
        tile_min, tile_max, count, name = _validate_content_glb(
            tile_path, property_name, expected_gltf_dimensions
        )
        if property_name is None:
            property_name = name
        overall_minimum = min(overall_minimum, tile_min)
        overall_maximum = max(overall_maximum, tile_max)
        value_count += count

    return ValidationResult(
        tileset=tileset_path,
        dimensions=dimensions,
        property_name=property_name or "",
        value_count=value_count,
        minimum=overall_minimum,
        maximum=overall_maximum,
    )


def _content_coordinates(
    subtree: dict[str, Any],
    subtree_dir: Path,
    subtree_levels: int,
) -> tuple[TileCoordinate, ...]:
    entry = subtree.get("contentAvailability")
    _require(entry is not None, "contentAvailability is required")
    if isinstance(entry, list):
        entry = entry[0]
    _require(isinstance(entry, dict), "contentAvailability entry must be an object")
    if "constant" in entry:
        if entry["constant"] == 1:
            return (TileCoordinate(0, 0, 0, 0),)
        return ()
    return _bitstream_coordinates(subtree, subtree_dir, subtree_levels, "contentAvailability")


def _bitstream_coordinates(
    subtree: dict[str, Any],
    subtree_dir: Path,
    subtree_levels: int,
    availability_key: str,
) -> tuple[TileCoordinate, ...]:
    entry = subtree[availability_key]
    if isinstance(entry, list):
        entry = entry[0]
    view_index = entry.get("bitstream")
    _require(isinstance(view_index, int), f"{availability_key} bitstream index missing")
    view = subtree["bufferViews"][view_index]
    buffer = subtree["buffers"][view["buffer"]]
    payload = (subtree_dir / buffer["uri"]).read_bytes()
    return available_coordinates(payload, subtree_levels=subtree_levels)


def _validate_content_glb(
    gltf_path: Path,
    property_name_hint: str | None,
    expected_dimensions: tuple[int, int, int],
) -> tuple[float, float, int, str]:
    embedded_binary: bytes | None = None
    if gltf_path.suffix.lower() == ".glb":
        parsed = parse_glb(gltf_path.read_bytes())
        gltf = parsed.document
        embedded_binary = parsed.binary
    else:
        gltf = _read_json(gltf_path)
    required = gltf.get("extensionsRequired", [])
    _require("EXT_primitive_voxels" in required, "glTF must require EXT_primitive_voxels")
    _require("EXT_structural_metadata" in required, "glTF must require EXT_structural_metadata")

    primitive = gltf["meshes"][0]["primitives"][0]
    _require(primitive.get("mode") == 2_147_483_647, "unexpected voxel primitive mode")
    primitive_voxel = _object(_object(primitive, "extensions"), "EXT_primitive_voxels")
    _require(
        tuple(primitive_voxel.get("dimensions", [])) == expected_dimensions,
        "glTF dimensions must swap Y and Z for BOX Y-up",
    )

    metadata = _object(_object(gltf, "extensions"), "EXT_structural_metadata")
    property_attribute = metadata["propertyAttributes"][0]
    properties = _object(property_attribute, "properties")
    _require(len(properties) == 1, "voxel content must contain exactly one property")
    property_name = next(iter(properties))
    if property_name_hint is not None:
        _require(property_name == property_name_hint, "property name mismatch across tiles")
    semantic = properties[property_name].get("attribute")
    accessor_index = primitive["attributes"].get(semantic)
    _require(isinstance(accessor_index, int), "property attribute accessor is missing")

    accessor = gltf["accessors"][accessor_index]
    expected_count = expected_dimensions[0] * expected_dimensions[1] * expected_dimensions[2]
    _require(accessor.get("componentType") == 5126, "voxel accessor must be FLOAT32")
    _require(accessor.get("type") == "SCALAR", "voxel accessor must be SCALAR")
    _require(accessor.get("count") == expected_count, "accessor count does not match dimensions")
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    buffer = gltf["buffers"][buffer_view["buffer"]]
    if embedded_binary is None:
        payload = (gltf_path.parent / buffer["uri"]).read_bytes()
    else:
        payload = embedded_binary
    _require(len(payload) >= buffer["byteLength"], "binary length is shorter than buffer")
    byte_offset = int(buffer_view.get("byteOffset", 0))
    byte_length = int(buffer_view["byteLength"])
    payload = payload[byte_offset : byte_offset + byte_length]
    _require(len(payload) == byte_length, "binary length does not match bufferView")
    _require(len(payload) == expected_count * 4, "binary size is not FLOAT32 count")

    values = np.frombuffer(payload, dtype="<f4")
    return float(values.min()), float(values.max()), expected_count, property_name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"referenced file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _object(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    _require(isinstance(value, dict), f"{key} must be an object")
    return value


def _dimensions(value: Any, label: str) -> tuple[int, int, int]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} must have 3 items")
    _require(all(isinstance(item, int) and item > 0 for item in value), f"{label} must be positive integers")
    return tuple(value)  # type: ignore[return-value]


def _expand_uri(template: str) -> Path:
    return Path(template.format(level=0, x=0, y=0, z=0))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = validate_probe(args.path)
    print(
        f"valid: {result.tileset} | dimensions={result.dimensions} | "
        f"property={result.property_name} | values={result.value_count} | "
        f"range=[{result.minimum}, {result.maximum}]"
    )


if __name__ == "__main__":
    main()
