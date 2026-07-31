import json
from pathlib import Path

import numpy as np
import pytest

from vtk2cesium.formats.glb import parse_glb
from vtk2cesium.geo import GeoReference, local_box_from_bounds
from vtk2cesium.model import ScalarAssociation, ScalarField, StructuredVoxelDataset
from vtk2cesium.transfer import NonFinitePolicy, ScalarPreprocessConfig
from vtk2cesium.validate import validate_probe
from vtk2cesium.config import TilingConfig
from vtk2cesium.writer import VoxelWriteError, write_voxel_tileset


def _dataset(values: np.ndarray) -> StructuredVoxelDataset:
    z_size, y_size, x_size = values.shape
    bounds = (10.0, 10.0 + x_size - 1, 20.0, 20.0 + y_size - 1, 30.0, 30.0 + z_size - 1)
    field = ScalarField("density", ScalarAssociation.POINT, values)
    return StructuredVoxelDataset(
        point_dimensions=(x_size, y_size, z_size),
        origin=(10.0, 20.0, 30.0),
        spacing=(1.0, 1.0, 1.0),
        bounds=bounds,
        fields={"density": field},
    )


def test_writer_outputs_complete_glb_tileset_atomically(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    dataset = _dataset(values)
    reference = GeoReference(116.3913, 39.9075, 1200.0)
    output = tmp_path / "voxel"

    tileset_path = write_voxel_tileset(
        dataset,
        output,
        field_name="density",
        georeference=reference,
    )
    result = validate_probe(tileset_path)
    tileset = json.loads(tileset_path.read_text(encoding="utf-8"))
    parsed = parse_glb((output / "content" / "0.0.0.0.glb").read_bytes())

    assert result.dimensions == (4, 3, 2)
    assert result.value_count == 24
    assert result.minimum == 0.0
    assert result.maximum == 23.0
    assert (output / "subtrees" / "0.0.0.0.subtree").exists()
    assert tileset["root"]["transform"] == list(reference.tileset_transform())
    assert tileset["root"]["boundingVolume"]["box"] == list(
        local_box_from_bounds(dataset.bounds)
    )
    assert np.array_equal(np.frombuffer(parsed.binary[:96], dtype="<f4"), np.arange(24))
    assert not list(tmp_path.glob(".voxel-*"))


def test_writer_refuses_non_empty_output_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "voxel"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        write_voxel_tileset(
            _dataset(np.zeros((2, 2, 2), dtype=np.float32)),
            output,
            field_name="density",
            georeference=GeoReference(0.0, 0.0),
        )
    assert keep.read_text(encoding="utf-8") == "keep"


def test_writer_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "voxel"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    write_voxel_tileset(
        _dataset(np.ones((2, 2, 2), dtype=np.float32)),
        output,
        field_name="density",
        georeference=GeoReference(0.0, 0.0),
        overwrite=True,
    )

    assert not (output / "old.txt").exists()
    assert (output / "tileset.json").exists()


def test_writer_rejects_mask_until_stage_supports_mask_encoding(tmp_path: Path) -> None:
    values = np.array([[[0.0, np.nan]]], dtype=np.float32)

    with pytest.raises(VoxelWriteError, match="validity mask"):
        write_voxel_tileset(
            _dataset(values),
            tmp_path / "voxel",
            field_name="density",
            georeference=GeoReference(0.0, 0.0),
        )

    tileset = write_voxel_tileset(
        _dataset(values),
        tmp_path / "filled",
        field_name="density",
        georeference=GeoReference(0.0, 0.0),
        preprocess=ScalarPreprocessConfig(
            non_finite=NonFinitePolicy.FILL,
            fill_value=0.0,
        ),
    )
    assert tileset.exists()


def test_writer_multilevel_emits_implicit_octree_pyramid(tmp_path: Path) -> None:
    values = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
    dataset = _dataset(values)
    reference = GeoReference(116.3913, 39.9075, 1200.0)
    output = tmp_path / "voxel-multilevel"

    tileset_path = write_voxel_tileset(
        dataset,
        output,
        field_name="density",
        georeference=reference,
        tiling=TilingConfig(tile_dimensions=(2, 2, 2)),
    )
    result = validate_probe(tileset_path)
    tileset = json.loads(tileset_path.read_text(encoding="utf-8"))

    implicit = tileset["root"]["implicitTiling"]
    assert implicit["subdivisionScheme"] == "OCTREE"
    assert implicit["subtreeLevels"] == 2
    assert implicit["availableLevels"] == 2
    assert tileset["root"]["content"]["extensions"]["3DTILES_content_voxels"]["dimensions"] == [
        2,
        2,
        2,
    ]

    content_dir = output / "content"
    glbs = sorted(content_dir.glob("*.glb"))
    assert [path.name for path in glbs] == [
        "0.0.0.0.glb",
        "1.0.0.0.glb",
        "1.0.0.1.glb",
        "1.0.1.0.glb",
        "1.0.1.1.glb",
        "1.1.0.0.glb",
        "1.1.0.1.glb",
        "1.1.1.0.glb",
        "1.1.1.1.glb",
    ]
    for path in glbs:
        parsed = parse_glb(path.read_bytes())
        assert parsed.document["meshes"][0]["primitives"][0]["extensions"][
            "EXT_primitive_voxels"
        ]["dimensions"] == [2, 2, 2]

    assert result.dimensions == (2, 2, 2)
    assert result.value_count == 9 * 8
    assert result.minimum == 0.0
    assert result.maximum == 63.0
    assert (output / "subtrees" / "0.0.0.0.bin").exists()
    assert not list(tmp_path.glob(".voxel-*"))


def test_writer_multilevel_box_uses_real_spacing_and_origin(tmp_path: Path) -> None:
    # Non-unit spacing and a non-zero origin: the root box must be metric
    # (origin + capacity * spacing), matching the geo transform which has no scale.
    values = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
    z_size, y_size, x_size = values.shape
    spacing = (2.0, 3.0, 4.0)
    origin = (10.0, 20.0, 30.0)
    bounds = (
        origin[0],
        origin[0] + (x_size - 1) * spacing[0],
        origin[1],
        origin[1] + (y_size - 1) * spacing[1],
        origin[2],
        origin[2] + (z_size - 1) * spacing[2],
    )
    dataset = StructuredVoxelDataset(
        point_dimensions=(x_size, y_size, z_size),
        origin=origin,
        spacing=spacing,
        bounds=bounds,
        fields={"density": ScalarField("density", ScalarAssociation.POINT, values)},
    )

    tileset_path = write_voxel_tileset(
        dataset,
        tmp_path / "voxel-scale",
        field_name="density",
        georeference=GeoReference(116.3913, 39.9075, 1200.0),
        tiling=TilingConfig(tile_dimensions=(2, 2, 2)),
    )
    tileset = json.loads(tileset_path.read_text(encoding="utf-8"))

    # tile_dim (2,2,2) + 2 levels => capacity (4,4,4); metric box = origin + 4*spacing.
    expected = local_box_from_bounds(
        (
            origin[0],
            origin[0] + 4 * spacing[0],
            origin[1],
            origin[1] + 4 * spacing[1],
            origin[2],
            origin[2] + 4 * spacing[2],
        )
    )
    assert tileset["root"]["boundingVolume"]["box"] == list(expected)
