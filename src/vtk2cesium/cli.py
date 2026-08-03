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
    PipelineConfig,
    PreprocessConfig,
    ReaderConfig,
    TilingConfig,
    VectorConfig,
)
from vtk2cesium.model import ScalarAssociation, StructuredVoxelDataset
from vtk2cesium.geo import GeoReference
from vtk2cesium.pipeline import convert_vti, inspect_vtk, validate_output
from vtk2cesium.readers import read_dataset
from vtk2cesium.transfer import NonFinitePolicy, ScalarMapping
from vtk2cesium.vector_field import build_vector_overlay, write_vector_overlay

VIEWER_ASSETS = ("index.html", "viewer.js", "WindParticles.js", "style.css")

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
    reference_latitude: Annotated[float | None, typer.Option("--reference-latitude", help="Reference latitude (deg) for lon->metre conversion in NetCDF/GeoTIFF.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Inspect VTI/NetCDF/GeoTIFF geometry, fields, ranges, and estimated memory."""

    try:
        inspection = inspect_vtk(
            input_path,
            reader=ReaderConfig(reference_latitude=reference_latitude) if reference_latitude is not None else None,
        )
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
    reference_latitude: Annotated[float | None, typer.Option("--reference-latitude", help="Reference latitude (deg) for lon->metre conversion.")] = None,
    x_dim: Annotated[str | None, typer.Option("--x-dim", help="Explicit NetCDF x-axis dimension name.")] = None,
    y_dim: Annotated[str | None, typer.Option("--y-dim", help="Explicit NetCDF y-axis dimension name.")] = None,
    z_dim: Annotated[str | None, typer.Option("--z-dim", help="Explicit NetCDF z-axis dimension name.")] = None,
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
            reference_latitude=reference_latitude,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
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
    input_path: Annotated[Path | None, typer.Argument()] = None,
    output_path: Annotated[Path | None, typer.Argument()] = None,
    config_path: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    field_u: Annotated[str | None, typer.Option("--field-u")] = None,
    field_v: Annotated[str | None, typer.Option("--field-v")] = None,
    field_w: Annotated[str | None, typer.Option("--field-w")] = None,
    longitude: Annotated[float | None, typer.Option("--lon")] = None,
    latitude: Annotated[float | None, typer.Option("--lat")] = None,
    height: Annotated[float | None, typer.Option("--height")] = None,
    step: Annotated[str | None, typer.Option("--step")] = None,
    arrow_length: Annotated[float | None, typer.Option("--arrow-length")] = None,
    streamline_count: Annotated[int | None, typer.Option("--streamlines")] = None,
    streamline_steps: Annotated[int | None, typer.Option("--streamline-steps")] = None,
    streamline_step_meters: Annotated[float | None, typer.Option("--streamline-step")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    emit_field: Annotated[bool | None, typer.Option("--emit-field/--no-emit-field")] = None,
    field_step: Annotated[int | None, typer.Option("--field-step")] = None,
    reference_latitude: Annotated[float | None, typer.Option("--reference-latitude", help="Reference latitude (deg) for lon->metre conversion (NetCDF/GeoTIFF).")] = None,
    x_dim: Annotated[str | None, typer.Option("--x-dim")] = None,
    y_dim: Annotated[str | None, typer.Option("--y-dim")] = None,
    z_dim: Annotated[str | None, typer.Option("--z-dim")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build a decoupled CZML vector overlay (arrows + streamlines) from u/v/w.

    This is fully independent of ``convert``: it reads the original VTI/NetCDF/GeoTIFF
    velocity components, downsamples them, and writes sharded ``arrows-<i>.czml`` /
    ``streamlines-<i>.czml`` files plus a ``vectors-manifest.json`` (each shard stays
    under common 64 KiB transport caps) georeferenced with the same ENU->ECEF transform
    as the voxel tileset.

    Pass ``--config <pipeline.yaml>`` to reuse the shared georeference and vector
    settings from the unified pipeline config; CLI options still override.
    """

    pipeline = None
    if config_path is not None:
        try:
            pipeline = PipelineConfig.from_yaml(config_path)
        except (OSError, ValueError, ValidationError) as error:
            _fail(str(error), EXIT_USAGE)

    resolved = _resolve_vector_args(
        pipeline=pipeline,
        input_path=input_path,
        output_path=output_path,
        field_u=field_u,
        field_v=field_v,
        field_w=field_w,
        longitude=longitude,
        latitude=latitude,
        height=height,
        step=step,
        arrow_length=arrow_length,
        streamline_count=streamline_count,
        streamline_steps=streamline_steps,
        streamline_step_meters=streamline_step_meters,
        seed=seed,
        emit_field=emit_field,
        field_step=field_step,
        reference_latitude=reference_latitude,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )

    document = _run_vector(**resolved, overwrite=overwrite)
    if json_output:
        _json_dump(document)
        return
    console.print(f"[green]Vector overlay:[/green] {document['output']}")
    console.print(
        f"arrows={document['arrow_count']} streamlines={document['streamline_count']} "
        f"speed=[{document['speed_min']:.3f}, {document['speed_max']:.3f}]"
    )
    if document["emit_field"]:
        console.print(
            f"[green]Velocity field:[/green] {document['output']} "
            f"(grid={document['field_dimensions']}, step={document['field_step']})"
        )


def _resolve_vector_args(
    *,
    pipeline: PipelineConfig | None,
    input_path: Path | None,
    output_path: Path | None,
    field_u: str | None,
    field_v: str | None,
    field_w: str | None,
    longitude: float | None,
    latitude: float | None,
    height: float | None,
    step: str | None,
    arrow_length: float | None,
    streamline_count: int | None,
    streamline_steps: int | None,
    streamline_step_meters: float | None,
    seed: int | None,
    emit_field: bool | None,
    field_step: int | None,
    reference_latitude: float | None = None,
    x_dim: str | None = None,
    y_dim: str | None = None,
    z_dim: str | None = None,
) -> dict:
    """Merge CLI overrides onto a pipeline config (or built-in defaults)."""

    cfg_vec: VectorConfig | None = pipeline.vector if pipeline else None
    base = dict(
        field_u="u",
        field_v="v",
        field_w="w",
        arrow_length=400.0,
        streamline_count=150,
        streamline_steps=50,
        streamline_step_meters=250.0,
        seed=0,
        emit_field=False,
        field_step=8,
    )
    if cfg_vec is not None:
        for key in base:
            base[key] = getattr(cfg_vec, key)

    def pick(cli: object, key: str) -> object:
        return cli if cli is not None else base[key]

    # Georeference: CLI wins; otherwise the shared pipeline georeference.
    if pipeline is not None:
        geo = pipeline.georeference
    else:
        geo = GeoReferenceConfig(longitude=116.3913, latitude=39.9075, height=0.0)
    resolved_geo = GeoReferenceConfig.model_validate(
        {
            "longitude": longitude if longitude is not None else geo.longitude,
            "latitude": latitude if latitude is not None else geo.latitude,
            "height": height if height is not None else geo.height,
        }
    )

    effective_input = input_path or (pipeline.input if pipeline else None)
    effective_output = output_path or (pipeline.output / "vectors" if pipeline else None)
    if effective_input is None or effective_output is None:
        _fail(
            "vector 需要 input 与 output：直接传参，或通过 --config 提供。",
            EXIT_USAGE,
        )

    # Reader hints: CLI wins; otherwise the shared pipeline vector reader.
    cli_reader = None
    if any(v is not None for v in (reference_latitude, x_dim, y_dim, z_dim)):
        cli_reader = ReaderConfig(
            reference_latitude=reference_latitude,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
        )
    reader = cli_reader or (cfg_vec.reader if cfg_vec else None)

    return {
        "input_path": effective_input,
        "output_path": effective_output,
        "georeference": GeoReference(
            longitude=resolved_geo.longitude,
            latitude=resolved_geo.latitude,
            height=resolved_geo.height,
        ),
        "field_u": str(pick(field_u, "field_u")),
        "field_v": str(pick(field_v, "field_v")),
        "field_w": str(pick(field_w, "field_w")),
        "step": step if step is not None else cfg_vec.step if cfg_vec else None,
        "arrow_length": float(pick(arrow_length, "arrow_length")),
        "streamline_count": int(pick(streamline_count, "streamline_count")),
        "streamline_steps": int(pick(streamline_steps, "streamline_steps")),
        "streamline_step_meters": float(pick(streamline_step_meters, "streamline_step_meters")),
        "seed": int(pick(seed, "seed")),
        "emit_field": bool(pick(emit_field, "emit_field")),
        "field_step": int(pick(field_step, "field_step")),
        "reader": reader,
    }


def _run_vector(
    *,
    input_path: Path,
    output_path: Path,
    georeference: GeoReference,
    field_u: str,
    field_v: str,
    field_w: str,
    step: int | tuple[int, int, int] | str | None,
    arrow_length: float,
    streamline_count: int,
    streamline_steps: int,
    streamline_step_meters: float,
    seed: int,
    overwrite: bool,
    emit_field: bool,
    field_step: int,
    reader: ReaderConfig | None = None,
) -> dict:
    """Shared vector-generation core used by both the ``vector`` and ``run`` commands."""

    try:
        dataset = _load_vector_dataset(input_path, field_u, field_v, field_w, reader=reader)
        parsed_step = _parse_step(step) if isinstance(step, str) else step
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
        field = None
        if emit_field:
            from vtk2cesium.vector_field import build_velocity_field, write_velocity_field

            field = build_velocity_field(
                dataset,
                u_name=field_u,
                v_name=field_v,
                w_name=field_w,
                field_step=field_step,
            )
            write_velocity_field(field, georeference, output_path)
    except (OSError, ValueError, KeyError) as error:
        _fail(str(error), EXIT_INPUT)
    return {
        "output": str(output_path),
        "arrow_count": result.arrow_count,
        "streamline_count": result.streamline_count,
        "speed_min": result.speed_min,
        "speed_max": result.speed_max,
        "emit_field": emit_field,
        "field_dimensions": tuple(field.dimensions) if field else None,
        "field_step": field_step,
    }


def _load_vector_dataset(
    input_path: Path, field_u: str, field_v: str, field_w: str, reader: ReaderConfig | None = None
) -> StructuredVoxelDataset:
    """Load three velocity components into one structured dataset."""

    loaded = [
        read_dataset(input_path, field_name=name, association=ScalarAssociation.POINT, reader=reader)
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
    reference_latitude: float | None = None,
    x_dim: str | None = None,
    y_dim: str | None = None,
    z_dim: str | None = None,
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
    if any(v is not None for v in (reference_latitude, x_dim, y_dim, z_dim)):
        data["reader"] = ReaderConfig(
            reference_latitude=reference_latitude,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
        ).model_dump()
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


def _deploy_viewer(output_dir: Path) -> Path:
    """Copy the standalone viewer assets into ``<output>/viewer`` and write a
    launcher ``index.html`` that opens it with the correct relative query params.

    The result is a self-contained directory: tileset/vectors at the root, viewer
    under ``viewer/``, so ``vtk2cesium serve <output>`` just works.
    """

    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "examples" / "viewer"
    if not source.is_dir():
        raise RuntimeError(f"未找到查看器资源目录: {source}")

    dest = output_dir / "viewer"
    dest.mkdir(parents=True, exist_ok=True)
    import shutil

    for asset in VIEWER_ASSETS:
        src = source / asset
        if src.exists():
            shutil.copy2(src, dest / asset)

    launcher = output_dir / "index.html"
    launcher.write_text(
        "<!DOCTYPE html>\n"
        '<html lang="zh">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta http-equiv="refresh" content="0; '
        'url=./viewer/?tileset=../tileset.json&vectors=../vectors">\n'
        "<title>VTK2Cesium 查看器</title>\n"
        "</head>\n"
        "<body>\n"
        '<p>正在打开查看器… 若未自动跳转，请<a href="./viewer/?tileset=../tileset.json&vectors=../vectors">点击这里</a>。</p>\n'
        "</body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    return dest


@app.command("run")
def run_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    no_viewer: Annotated[bool, typer.Option("--no-viewer")] = False,
    reference_latitude: Annotated[float | None, typer.Option("--reference-latitude", help="Reference latitude (deg) for lon->metre conversion (NetCDF/GeoTIFF).")] = None,
    x_dim: Annotated[str | None, typer.Option("--x-dim")] = None,
    y_dim: Annotated[str | None, typer.Option("--y-dim")] = None,
    z_dim: Annotated[str | None, typer.Option("--z-dim")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """End-to-end pipeline: read one YAML config and run convert -> vector -> deploy.

    Produces a self-contained output directory (voxel tileset + vector overlay +
    copied viewer) ready to serve with ``vtk2cesium serve``.
    """

    try:
        config = PipelineConfig.from_yaml(config_path)
    except (OSError, ValueError, ValidationError) as error:
        _fail(str(error), EXIT_USAGE)

    # CLI reader hints override the shared pipeline reader (if any).
    if any(v is not None for v in (reference_latitude, x_dim, y_dim, z_dim)):
        config = config.with_reader(
            ReaderConfig(
                reference_latitude=reference_latitude,
                x_dim=x_dim,
                y_dim=y_dim,
                z_dim=z_dim,
            )
        )

    # 1) 体素生产（与 vector 共享同一 georeference）
    convert_cfg = config.convert_config(overwrite=overwrite)
    try:
        conv = convert_vti(convert_cfg)
    except FileExistsError as error:
        _fail(str(error), EXIT_OUTPUT)
    except (OSError, KeyError, ValueError) as error:
        _fail(str(error), EXIT_INPUT)

    # 2) 矢量叠加 + 速度场（解耦，写入 <output>/vectors）
    vector_out = config.output / "vectors"
    vec = _run_vector(
        input_path=config.input,
        output_path=vector_out,
        georeference=GeoReference(**config.georeference.model_dump()),
        field_u=config.vector.field_u,
        field_v=config.vector.field_v,
        field_w=config.vector.field_w,
        step=config.vector.step,
        arrow_length=config.vector.arrow_length,
        streamline_count=config.vector.streamlines,
        streamline_steps=config.vector.streamline_steps,
        streamline_step_meters=config.vector.streamline_step,
        seed=config.vector.seed,
        overwrite=overwrite,
        emit_field=config.vector.emit_field,
        field_step=config.vector.field_step,
        reader=config.vector.reader or config.reader,
    )

    # 3) 部署自包含查看器
    viewer_dir = None
    if not no_viewer:
        try:
            viewer_dir = _deploy_viewer(config.output)
        except (OSError, RuntimeError) as error:
            console.print(
                f"[yellow]警告：[/yellow] 查看器资源部署失败，请手动拷贝 examples/viewer：{error}"
            )

    document = {
        **conv.to_dict(),
        "vectors": vec,
        "viewer": str(viewer_dir) if viewer_dir else None,
    }
    if json_output:
        _json_dump(document)
        return
    console.print(f"[green]流程完成[/green] 输出目录: {config.output}")
    console.print(f"  体素: {conv.tileset} (dimensions={conv.dimensions})")
    console.print(
        f"  矢量: {vec['arrow_count']} 箭头 / {vec['streamline_count']} 流线"
        + (f"，速度场网格={vec['field_dimensions']}" if vec["emit_field"] else "")
    )
    if viewer_dir:
        console.print(
            f"  查看器: {viewer_dir}\n"
            f"  下一步: 运行 `vtk2cesium serve {config.output}` 然后打开 http://localhost:8000/"
        )


@app.command("serve")
def serve_command(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    open_browser: Annotated[bool, typer.Option("--open")] = False,
) -> None:
    """Start a static file server (full bodies, no truncation) and print the URL.

    Use this to view a generated tileset / self-contained output directory. Press
    Ctrl+C to stop. Picks the next free port if the requested one is busy.
    """

    import functools
    import webbrowser

    import http.server
    import socketserver

    class _ThreadingHTTPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory.resolve())
    )
    httpd = None
    bound_port = None
    for candidate in range(port, port + 11):
        try:
            httpd = _ThreadingHTTPServer(("", candidate), handler)
            bound_port = candidate
            break
        except OSError:
            continue
    if httpd is None:
        _fail(f"无法在端口 {port}..{port + 10} 上绑定静态服务器", EXIT_OUTPUT)

    url = f"http://localhost:{bound_port}/"
    console.print(f"[green]Serving[/green] {directory.resolve()}")
    console.print(f"  打开: {url}  (Ctrl+C 停止)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]已停止服务器。[/yellow]")
    finally:
        httpd.server_close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
