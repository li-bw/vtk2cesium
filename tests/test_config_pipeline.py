from pathlib import Path

import pytest

from vtk2cesium.config import ConvertConfig, TilingConfig
from vtk2cesium.geo import GeoReference
from vtk2cesium.pipeline import convert_vti, inspect_vtk, validate_output


def test_config_loads_yaml_and_resolves_relative_paths(tmp_path: Path, sample_vti: Path) -> None:
    config_path = tmp_path / "convert.yaml"
    config_path.write_text(
        """input: sample.vti
output: output
field_name: temperature
association: point
georeference:
  longitude: 116.3913
  latitude: 39.9075
  height: 1200
preprocess:
  mapping: linear
  source_range: [0, 59]
""",
        encoding="utf-8",
    )
    config = ConvertConfig.from_yaml(config_path)

    assert config.input == sample_vti.resolve()
    assert config.output == (tmp_path / "output").resolve()
    assert config.preprocess.mapping.value == "linear"
    assert config.georeference.height == 1200.0


def test_config_rejects_unknown_keys_and_invalid_geo(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """input: input.vti
output: output
field_name: density
unknown: true
georeference:
  longitude: 0
  latitude: 91
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ConvertConfig.from_yaml(path)


def test_sdk_inspect_convert_and_validate(tmp_path: Path, sample_vti: Path) -> None:
    config = ConvertConfig(
        input=sample_vti,
        output=tmp_path / "tiles",
        field_name="temperature",
        association="point",
        georeference={"longitude": 116.3913, "latitude": 39.9075, "height": 1200},
        preprocess={"mapping": "linear", "source_range": (0.0, 59.0)},
    )

    inspection = inspect_vtk(sample_vti)
    result = convert_vti(config)
    validation = validate_output(result.output)

    assert inspection.point_dimensions == (3, 4, 5)
    assert result.dimensions == (3, 4, 5)
    assert result.value_count == 60
    assert result.minimum == 0.0
    assert result.maximum == 1.0
    assert validation.tileset == result.tileset
    assert result.tileset.exists()


def test_tiling_config_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        TilingConfig(available_levels=1)
    with pytest.raises(ValueError, match="three positive"):
        TilingConfig(tile_dimensions=(0, 2, 2))
    with pytest.raises(ValueError):
        TilingConfig(tile_dimensions=(2, 2))


def test_convert_config_carries_tiling_into_sdk(tmp_path: Path, sample_vti: Path) -> None:
    config = ConvertConfig(
        input=sample_vti,
        output=tmp_path / "tiles",
        field_name="temperature",
        association="point",
        georeference={"longitude": 116.3913, "latitude": 39.9075, "height": 1200},
        tiling={"available_levels": 2},
    )

    result = convert_vti(config)
    tileset = __import__("json").loads(result.tileset.read_text(encoding="utf-8"))

    assert result.dimensions == (2, 2, 3)
    assert tileset["root"]["implicitTiling"]["availableLevels"] == 2
    assert len(list(result.output.joinpath("content").glob("*.glb"))) == 9
