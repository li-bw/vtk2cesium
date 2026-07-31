import json
from pathlib import Path

from typer.testing import CliRunner

from vtk2cesium.cli import EXIT_OUTPUT, EXIT_USAGE, app
from vtk2cesium.config import ConvertConfig
from vtk2cesium.pipeline import convert_vti

runner = CliRunner()


def test_inspect_json_command(sample_vti: Path) -> None:
    result = runner.invoke(app, ["inspect", str(sample_vti), "--json"])

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["point_dimensions"] == [3, 4, 5]
    assert {field["name"] for field in document["fields"]} == {
        "temperature",
        "velocity",
        "pressure",
    }


def test_convert_and_validate_commands(tmp_path: Path, sample_vti: Path) -> None:
    output = tmp_path / "cli-output"
    convert_result = runner.invoke(
        app,
        [
            "convert",
            str(sample_vti),
            str(output),
            "--field",
            "temperature",
            "--association",
            "point",
            "--lon",
            "116.3913",
            "--lat",
            "39.9075",
            "--height",
            "1200",
            "--mapping",
            "linear",
            "--source-min",
            "0",
            "--source-max",
            "59",
            "--json",
        ],
    )
    validate_result = runner.invoke(app, ["validate", str(output), "--json"])

    assert convert_result.exit_code == 0, convert_result.output
    assert validate_result.exit_code == 0, validate_result.output
    converted = json.loads(convert_result.stdout)
    validated = json.loads(validate_result.stdout)
    assert converted["dimensions"] == [3, 4, 5]
    assert converted["value_count"] == 60
    assert validated["minimum"] == 0.0
    assert validated["maximum"] == 1.0


def test_cli_overrides_yaml_and_matches_sdk_bytes(tmp_path: Path, sample_vti: Path) -> None:
    config_path = tmp_path / "convert.yaml"
    config_path.write_text(
        f"""input: {sample_vti.as_posix()}
output: yaml-output
field_name: temperature
association: point
georeference:
  longitude: 0
  latitude: 0
  height: 0
preprocess:
  mapping: identity
""",
        encoding="utf-8",
    )
    cli_output = tmp_path / "cli-output"
    sdk_output = tmp_path / "sdk-output"
    cli_result = runner.invoke(
        app,
        [
            "convert",
            "--config",
            str(config_path),
            str(sample_vti),
            str(cli_output),
            "--lon",
            "116.3913",
            "--lat",
            "39.9075",
            "--height",
            "1200",
            "--mapping",
            "linear",
            "--source-min",
            "0",
            "--source-max",
            "59",
        ],
    )
    sdk_config = ConvertConfig(
        input=sample_vti,
        output=sdk_output,
        field_name="temperature",
        association="point",
        georeference={"longitude": 116.3913, "latitude": 39.9075, "height": 1200},
        preprocess={"mapping": "linear", "source_range": (0.0, 59.0)},
    )
    convert_vti(sdk_config)

    assert cli_result.exit_code == 0, cli_result.output
    for relative in (
        Path("tileset.json"),
        Path("subtrees/0.0.0.0.subtree"),
        Path("content/0.0.0.0.glb"),
    ):
        assert (cli_output / relative).read_bytes() == (sdk_output / relative).read_bytes()


def test_convert_reports_usage_and_output_exit_codes(tmp_path: Path, sample_vti: Path) -> None:
    missing = runner.invoke(app, ["convert"])
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    occupied = runner.invoke(
        app,
        [
            "convert",
            str(sample_vti),
            str(output),
            "--field",
            "temperature",
            "--lon",
            "0",
            "--lat",
            "0",
        ],
    )

    assert missing.exit_code == EXIT_USAGE
    assert occupied.exit_code == EXIT_OUTPUT
    assert (output / "keep.txt").exists()


def test_convert_available_levels_produces_multilevel_tileset(
    tmp_path: Path, sample_vti: Path
) -> None:
    output = tmp_path / "cli-multilevel"
    convert_result = runner.invoke(
        app,
        [
            "convert",
            str(sample_vti),
            str(output),
            "--field",
            "temperature",
            "--association",
            "point",
            "--lon",
            "116.3913",
            "--lat",
            "39.9075",
            "--height",
            "1200",
            "--available-levels",
            "2",
            "--json",
        ],
    )
    validate_result = runner.invoke(app, ["validate", str(output), "--json"])

    assert convert_result.exit_code == 0, convert_result.output
    assert validate_result.exit_code == 0, validate_result.output
    tileset = json.loads((output / "tileset.json").read_text(encoding="utf-8"))
    implicit = tileset["root"]["implicitTiling"]
    assert implicit["subtreeLevels"] == 2
    assert implicit["availableLevels"] == 2
    assert len(list((output / "content").glob("*.glb"))) == 9


def test_convert_tile_dimensions_produces_multilevel_tileset(
    tmp_path: Path, sample_vti: Path
) -> None:
    output = tmp_path / "cli-tiled"
    result = runner.invoke(
        app,
        [
            "convert",
            str(sample_vti),
            str(output),
            "--field",
            "temperature",
            "--association",
            "point",
            "--lon",
            "116.3913",
            "--lat",
            "39.9075",
            "--tile-dimensions",
            "2,2,3",
        ],
    )

    assert result.exit_code == 0, result.output
    validate = runner.invoke(app, ["validate", str(output)])
    assert validate.exit_code == 0, validate.output
    tileset = json.loads((output / "tileset.json").read_text(encoding="utf-8"))
    assert tileset["root"]["implicitTiling"]["availableLevels"] == 2
    assert len(list((output / "content").glob("*.glb"))) == 9


def test_convert_rejects_malformed_tile_dimensions(tmp_path: Path, sample_vti: Path) -> None:
    result = runner.invoke(
        app,
        [
            "convert",
            str(sample_vti),
            str(tmp_path / "bad"),
            "--field",
            "temperature",
            "--lon",
            "0",
            "--lat",
            "0",
            "--tile-dimensions",
            "2,2",
        ],
    )

    assert result.exit_code == EXIT_USAGE
    assert "tile-dimensions" in result.output.lower()
