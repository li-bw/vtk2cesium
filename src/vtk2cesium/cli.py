"""Command-line interface for vtk2Cesium inspection, conversion, and validation."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
import typer

from vtk2cesium.config import (
    ConvertConfig,
    GeoReferenceConfig,
    PreprocessConfig,
    TilingConfig,
)
from vtk2cesium.model import ScalarAssociation, StructuredVoxelDataset
from vtk2cesium.geo import GeoReference
from vtk2cesium.pipeline import convert_vti, inspect_vtk, validate_output
from vtk2cesium.readers.vti import read_vti
from vtk2cesium.transfer import NonFinitePolicy, ScalarMapping
from vtk2cesium.vector_field import build_vector_overlay, write_vector_overlay

app = typer.Typer(
    name="vtk2cesium",
    help="Convert VTK ImageData to CesiumJS experimental voxel tiles.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()
error_console = Console(stderr=True)

EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_OUTPUT = 4
EXIT_VALIDATION = 5


def _json_dump(value: Any) -> None:
    console.print_json(json.dumps(value, ensure_ascii=False, default=str))


def _fail(message: str, code: int) -> None:
    error_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)


@app.command("inspect")
def inspect_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect VTI geometry, fields, ranges, and estimated scalar memory."""

    try:
        inspection = inspect_vtk(input_path)
    except (OSError, ValueError) as error:
        _fail(str(error), EXIT_INPUT)
    fields = []
    for field in inspection.fields:
        fields.append(
            {
                "name": field.name,
                "association": field.association.value,
                "components": field.components,
                "tuples": field.tuples,
                "dtype": str(field.dtype),
                "finite_minimum": field.finite_minimum,
                "finite_maximum": field.finite_maximum,
                "non_finite_count": field.non_finite_count,
                "estimated_bytes": field.tuples * field.components * field.dtype.itemsize,
            }
        )
    document = {
        "input": str(input_path.resolve()),
        "point_dimensions": inspection.point_dimensions,
        "cell_dimensions": inspection.cell_dimensions,
        "origin": inspection.origin,
        "spacing": inspection.spacing,
        "bounds": inspection.bounds,
        "fields": fields,
    }
    if json_output:
        _json_dump(document)
        return

    console.print(f"[bold]Input:[/bold] {document['input']}")
    console.print(f"[bold]Point dimensions:[/bold] {inspection.point_dimensions}")
    console.print(f"[bold]Bounds:[/bold] {inspection.bounds}")
    table = Table("Association", "Field", "Components", "Tuples", "Type", "Finite range", "Memory")
    for field in fields:
        value_range = f"{field['finite_minimum']} .. {field['finite_maximum']}"
        table.add_row(
            field["association"],
            field["name"],
            str(field["components"]),
            str(field["tuples"]),
            field["dtype"],
            value_range,
            f"{field['estimated_bytes'] / (1024 * 1024):.2f} MiB",
        )
    console.print(table)


@app.command("convert")
def convert_command(
    input_path: Annotated[Path | None, typer.Argument()] = None,
    output_path: Annotated[Path | None, typer.Argument()] = None,
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    field_name: Annotated[str | None, typer.Option("--field")] = None,
    association: Annotated[str | None, typer.Option("--association")] = None,
    longitude: Annotated[float | None, typer.Option("--lon")] = None,
    latitude: Annotated[float | None, typer.Option("--lat")] = None,
    height: Annotated[float | None, typer.Option("--height")] = None,
    mapping: Annotated[str | None, typer.Option("--mapping")] = None,
    source_min: Annotated[float | None, typer.Option("--source-min")] = None,
    source_max: Annotated[float | None, typer.Option("--source-max")] = None,
    non_finite: Annotated[str | None, typer.Option("--non-finite")] = None,
    fill_value: Annotated[float | None, typer.Option("--fill-value")] = None,
    available_levels: Annotated[int | None, typer.Option("--available-levels")] = None,
    tile_dimensions: Annotated[str | None, typer.Option("--tile-dimensions")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Convert one VTI scalar field to a single-tile or multi-level voxel tileset.

    Pass ``--available-levels`` (>=2) to derive a uniform per-axis tile size, or
    ``--tile-dimensions`` (comma-separated x,y,z) to derive the level count. Both
    omit the default stage-4 single root tile.
    """

    try:
        config = _build_convert_config(
            config_path=config_path,
            input_path=input_path,
            output_path=output_path,
            field_name=field_name,
            association=association,
            longitude=longitude,
            latitude=latitude,
            height=height,
            mapping=mapping,
            source_min=source_min,
            source_max=source_max,
            non_finite=non_finite,
            fill_value=fill_value,
            available_levels=available_levels,
            tile_dimensions=tile_dimensions,
            overwrite=overwrite,
        )
    except (OSError, ValueError, ValidationError) as error:
        _fail(str(error), EXIT_USAGE)
    try:
        result = convert_vti(config)
    except FileExistsError as error:
        _fail(str(error), EXIT_OUTPUT)
    except (OSError, KeyError, ValueError) as error:
        _fail(str(error), EXIT_INPUT)
    if json_output:
        _json_dump(result.to_dict())
        return
    console.print(f"[green]Converted:[/green] {result.tileset}")
    console.print(
        f"dimensions={result.dimensions} values={result.value_count} "
        f"range=[{result.minimum}, {result.maximum}]"
    )


@app.command("validate")
def validate_command(
    output_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Validate tileset, subtree, GLB metadata, and FLOAT32 payload consistency."""

    try:
        result = validate_output(output_path)
    except (OSError, ValueError, KeyError, IndexError) as error:
        _fail(str(error), EXIT_VALIDATION)
    document = asdict(result)
    document["tileset"] = str(document["tileset"])
    if json_output:
        _json_dump(document)
        return
    console.print(
        f"[green]Valid:[/green] {result.tileset} | dimensions={result.dimensions} | "
        f"property={result.property_name} | values={result.value_count} | "
        f"range=[{result.minimum}, {result.maximum}]"
    )


@app.command("vector")
def vector_command(
    input_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_path: Annotated[Path, typer.Argument()],
    field_u: Annotated[str, typer.Option("--field-u")] = "u",
    field_v: Annotated[str, typer.Option("--field-v")] = "v",
    field_w: Annotated[str, typer.Option("--field-w")] = "w",
    longitude: Annotated[float, typer.Option("--lon")] = 116.3913,
    latitude: Annotated[float, typer.Option("--lat")] = 39.9075,
    height: Annotated[float, typer.Option("--height")] = 0.0,
    step: Annotated[str | None, typer.Option("--step")] = None,
    arrow_length: Annotated[float, typer.Option("--arrow-length")] = 400.0,
    streamline_count: Annotated[int, typer.Option("--streamlines")] = 150,
    streamline_steps: Annotated[int, typer.Option("--streamline-steps")] = 50,
    streamline_step_meters: Annotated[float, typer.Option("--streamline-step")] = 250.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build a decoupled CZML vector overlay (arrows + streamlines) from u/v/w.

    This is fully independent of ``convert``: it reads the original VTI velocity
    components, downsamples them, and writes ``arrows.czml`` / ``streamlines.czml``
    georeferenced with the same ENU->ECEF transform as the voxel tileset.
    """

    try:
        dataset = _load_vector_dataset(input_path, field_u, field_v, field_w)
        georeference = GeoReference(longitude=longitude, latitude=latitude, height=height)
        parsed_step = _parse_step(step)
        result = build_vector_overlay(
            dataset,
            georeference,
            u_name=field_u,
            v_name=field_v,
            w_name=field_w,
            step=parsed_step,
            arrow_length=arrow_length,
            streamline_count=streamline_count,
            streamline_steps=streamline_steps,
            streamline_step_meters=streamline_step_meters,
            seed=seed,
        )
        if output_path.exists() and any(output_path.iterdir()) and not overwrite:
            _fail(f"output directory is not empty: {output_path}", EXIT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        write_vector_overlay(result, output_path)
    except (OSError, ValueError, KeyError) as error:
        _fail(str(error), EXIT_INPUT)
    document = {
        "output": str(output_path),
        "arrow_count": result.arrow_count,
        "streamline_count": result.streamline_count,
        "speed_min": result.speed_min,
        "speed_max": result.speed_max,
    }
    if json_output:
        _json_dump(document)
        return
    console.print(f"[green]Vector overlay:[/green] {output_path}")
    console.print(
        f"arrows={result.arrow_count} streamlines={result.streamline_count} "
        f"speed=[{result.speed_min:.3f}, {result.speed_max:.3f}]"
    )


def _load_vector_dataset(
    input_path: Path, field_u: str, field_v: str, field_w: str
) -> StructuredVoxelDataset:
    """Load three velocity components into one structured dataset."""

    loaded = [
        read_vti(input_path, field_name=name, association=ScalarAssociation.POINT)
        for name in (field_u, field_v, field_w)
    ]
    base = loaded[0]
    fields = {part.field(field_name).name: part.field(field_name)
              for part, field_name in zip(loaded, (field_u, field_v, field_w))}
    return StructuredVoxelDataset(
        point_dimensions=base.point_dimensions,
        origin=base.origin,
        spacing=base.spacing,
        bounds=base.bounds,
        fields=fields,
    )


def _build_convert_config(
    *,
    config_path: Path | None,
    input_path: Path | None,
    output_path: Path | None,
    field_name: str | None,
    association: str | None,
    longitude: float | None,
    latitude: float | None,
    height: float | None,
    mapping: str | None,
    source_min: float | None,
    source_max: float | None,
    non_finite: str | None,
    fill_value: float | None,
    available_levels: int | None,
    tile_dimensions: str | None,
    overwrite: bool,
) -> ConvertConfig:
    if config_path is not None:
        base = ConvertConfig.from_yaml(config_path)
        data = base.model_dump()
    else:
        missing = [
            name
            for name, value in (
                ("INPUT", input_path),
                ("OUTPUT", output_path),
                ("--field", field_name),
                ("--lon", longitude),
                ("--lat", latitude),
            )
            if value is None
        ]
        if missing:
            raise ValueError("missing required conversion values: " + ", ".join(missing))
        data = {
            "input": input_path,
            "output": output_path,
            "field_name": field_name,
            "georeference": {"longitude": longitude, "latitude": latitude, "height": height or 0.0},
            "preprocess": {},
            "overwrite": False,
        }

    for key, value in (("input", input_path), ("output", output_path), ("field_name", field_name)):
        if value is not None:
            data[key] = value
    if association is not None:
        data["association"] = association

    geo = dict(data.get("georeference") or {})
    for key, value in (("longitude", longitude), ("latitude", latitude), ("height", height)):
        if value is not None:
            geo[key] = value
    data["georeference"] = GeoReferenceConfig.model_validate(geo).model_dump()

    preprocess = dict(data.get("preprocess") or {})
    if mapping is not None:
        preprocess["mapping"] = ScalarMapping(mapping)
    if source_min is not None or source_max is not None:
        if source_min is None or source_max is None:
            raise ValueError("--source-min and --source-max must be supplied together")
        preprocess["source_range"] = (source_min, source_max)
    if non_finite is not None:
        preprocess["non_finite"] = NonFinitePolicy(non_finite)
    if fill_value is not None:
        preprocess["fill_value"] = fill_value
    data["preprocess"] = PreprocessConfig.model_validate(preprocess).model_dump()
    if overwrite:
        data["overwrite"] = True

    tiling = _build_tiling_config(available_levels, tile_dimensions)
    if tiling is not None:
        data["tiling"] = tiling.model_dump()
    return ConvertConfig.model_validate(data)


def _build_tiling_config(
    available_levels: int | None,
    tile_dimensions: str | None,
) -> TilingConfig | None:
    if available_levels is None and tile_dimensions is None:
        return None
    parsed_dimensions = _parse_tile_dimensions(tile_dimensions)
    try:
        return TilingConfig(
            available_levels=available_levels,
            tile_dimensions=parsed_dimensions,
        )
    except ValidationError as error:
        raise ValueError(str(error)) from error


def _parse_tile_dimensions(raw: str | None) -> tuple[int, int, int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--tile-dimensions must be three comma-separated integers")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as error:
        raise ValueError("--tile-dimensions must contain integers") from error
    if min(values) <= 0:
        raise ValueError("--tile-dimensions must contain positive integers")
    return values


def _parse_step(raw: str | None) -> int | tuple[int, int, int] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) == 1:
        try:
            value = int(parts[0])
        except ValueError as error:
            raise ValueError("--step must be an integer") from error
        if value <= 0:
            raise ValueError("--step must be positive")
        return value
    if len(parts) == 3:
        try:
            values = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError("--step must contain integers") from error
        if min(values) <= 0:
            raise ValueError("--step must contain positive integers")
        return values
    raise ValueError("--step must be a single integer or three comma-separated integers")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
