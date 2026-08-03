"""High-level inspection, conversion, and validation SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vtk2cesium.config import ConvertConfig
from vtk2cesium.geo import GeoReference
from vtk2cesium.model import VtiInspection
from vtk2cesium.readers import inspect_dataset, read_dataset
from vtk2cesium.validate import ValidationResult, validate_probe
from vtk2cesium.writer import write_voxel_tileset


@dataclass(frozen=True)
class ConversionResult:
    """Stable summary returned after a successful conversion."""

    input: Path
    output: Path
    tileset: Path
    field_name: str
    dimensions: tuple[int, int, int]
    value_count: int
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        for key in ("input", "output", "tileset"):
            document[key] = str(document[key])
        return document


def inspect_vtk(path: str | Path, *, reader: Any = None) -> VtiInspection:
    """Inspect a supported input file (VTI/NetCDF/GeoTIFF) without selecting a field."""

    return inspect_dataset(Path(path), reader=reader)


def convert_vti(config: ConvertConfig) -> ConversionResult:
    """Convert one configured field and validate the completed output.

    Supports VTI, NetCDF, and GeoTIFF inputs through the unified reader.
    """

    dataset = read_dataset(
        config.input,
        field_name=config.field_name,
        association=config.association,
        reader=config.reader,
    )
    georeference = GeoReference(
        config.georeference.longitude,
        config.georeference.latitude,
        config.georeference.height,
    )
    tileset = write_voxel_tileset(
        dataset,
        config.output,
        field_name=config.field_name,
        georeference=georeference,
        preprocess=config.preprocess.to_domain(),
        tiling=config.tiling,
        overwrite=config.overwrite,
    )
    validation = validate_output(tileset)
    return ConversionResult(
        input=config.input.resolve(),
        output=config.output.resolve(),
        tileset=tileset,
        field_name=validation.property_name,
        dimensions=validation.dimensions,
        value_count=validation.value_count,
        minimum=validation.minimum,
        maximum=validation.maximum,
    )


def validate_output(path: str | Path) -> ValidationResult:
    """Validate a generated root-only voxel tileset."""

    return validate_probe(Path(path).expanduser().resolve())
