"""High-level inspection, conversion, and validation SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vtk2cesium.config import ConvertConfig
from vtk2cesium.geo import GeoReference
from vtk2cesium.model import VtiInspection
from vtk2cesium.readers import inspect_vti, read_vti
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


def inspect_vtk(path: str | Path) -> VtiInspection:
    """Inspect a supported VTK file without selecting a conversion field."""

    input_path = Path(path)
    if input_path.suffix.lower() != ".vti":
        raise ValueError("stage-4 inspect supports .vti input only")
    return inspect_vti(input_path)


def convert_vti(config: ConvertConfig) -> ConversionResult:
    """Convert one configured VTI field and validate the completed output."""

    dataset = read_vti(
        config.input,
        field_name=config.field_name,
        association=config.association,
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
