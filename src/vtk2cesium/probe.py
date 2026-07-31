"""Generate a deterministic root-only CesiumJS voxel compatibility probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vtk2cesium.formats.subtree import root_only_subtree
from vtk2cesium.formats.tileset import build_probe_tileset
from vtk2cesium.formats.voxel_gltf import encode_scalar_tile


def gradient(dimensions: tuple[int, int, int]) -> np.ndarray:
    """Return a deterministic `(z, y, x)` gradient in the range [0, 1]."""

    x_size, y_size, z_size = dimensions
    if min(dimensions) <= 0:
        raise ValueError("dimensions must be positive")
    z, y, x = np.indices((z_size, y_size, x_size), dtype=np.float32)
    denominator = max((x_size - 1) + (y_size - 1) + (z_size - 1), 1)
    return (x + y + z) / denominator


def write_probe(
    output: Path,
    *,
    dimensions: tuple[int, int, int] = (4, 4, 4),
    property_name: str = "density",
) -> Path:
    """Write a complete root-only probe data set and return tileset path."""

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    values = gradient(dimensions)
    tile = encode_scalar_tile(values, property_name=property_name, buffer_uri=f"{property_name}.bin")
    tileset = build_probe_tileset(
        dimensions,
        property_name=property_name,
        minimum=float(values.min()),
        maximum=float(values.max()),
    )

    tile_directory = output / "tiles" / "0" / "0" / "0"
    subtree_directory = output / "subtrees" / "0" / "0" / "0"
    tile_directory.mkdir(parents=True, exist_ok=True)
    subtree_directory.mkdir(parents=True, exist_ok=True)

    _write_json(output / "tileset.json", tileset)
    _write_json(subtree_directory / "0.json", root_only_subtree())
    _write_json(tile_directory / "0.gltf", tile.document)
    (tile_directory / f"{property_name}.bin").write_bytes(tile.binary)
    return output / "tileset.json"


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _parse_dimensions(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("dimensions must look like 4x4x4") from error
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("dimensions must contain three positive integers")
    return parts  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dimensions", type=_parse_dimensions, default=(4, 4, 4))
    parser.add_argument("--property", default="density", dest="property_name")
    args = parser.parse_args()
    path = write_probe(args.output, dimensions=args.dimensions, property_name=args.property_name)
    print(path)


if __name__ == "__main__":
    main()
