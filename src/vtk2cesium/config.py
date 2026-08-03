"""Validated conversion configuration with YAML loading support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from vtk2cesium.model import ScalarAssociation
from vtk2cesium.transfer import NonFinitePolicy, ScalarMapping, ScalarPreprocessConfig

SUPPORTED_INPUT_SUFFIXES = frozenset({".vti", ".nc", ".nc4", ".cdf", ".tif", ".tiff"})


class GeoReferenceConfig(BaseModel):
    """Explicit WGS84 anchor for a local ENU VTK data set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    height: float = 0.0


class ReaderConfig(BaseModel):
    """Optional, format-specific hints for the non-VTI (NetCDF/GeoTIFF) adapters.

    ``reference_latitude`` converts angular ``lon``/``lat`` coordinates (and
    geographic-CRS GeoTIFFs) to metres; ``x_dim``/``y_dim``/``z_dim`` override the
    automatic NetCDF axis detection; ``band_as_field`` keeps GeoTIFF bands as
    separate scalar fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_latitude: float | None = None
    x_dim: str | None = None
    y_dim: str | None = None
    z_dim: str | None = None
    band_as_field: bool = True


class PreprocessConfig(BaseModel):
    """Serializable scalar preprocessing options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mapping: ScalarMapping = ScalarMapping.IDENTITY
    source_range: tuple[float, float] | None = None
    clip: bool = True
    threshold: tuple[float | None, float | None] | None = None
    non_finite: NonFinitePolicy = NonFinitePolicy.MASK
    fill_value: float = 0.0
    log_base: float = 10.0
    piecewise_points: tuple[tuple[float, float], ...] = ()

    def to_domain(self) -> ScalarPreprocessConfig:
        return ScalarPreprocessConfig(**self.model_dump())


class TilingConfig(BaseModel):
    """Implicit OCTREE tiling and LOD options.

    Provide either ``available_levels`` (derive uniform per-axis tile size) or
    ``tile_dimensions`` (derive the smallest level count that fits). When both
    are omitted the writer keeps the stage-4 single root tile.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    available_levels: int | None = Field(default=None, ge=1, le=8)
    tile_dimensions: tuple[int, int, int] | None = None

    @model_validator(mode="after")
    def _check_dimensions(self) -> "TilingConfig":
        if self.tile_dimensions is not None:
            if len(self.tile_dimensions) != 3 or min(self.tile_dimensions) <= 0:
                raise ValueError("tile_dimensions must contain three positive integers")
        if self.available_levels is not None and self.available_levels < 2:
            raise ValueError("available_levels must be at least 2 for multi-level tiling")
        return self


class ConvertConfig(BaseModel):
    """Complete SDK/CLI configuration for one VTI/NetCDF/GeoTIFF conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: Path
    output: Path
    field_name: str = Field(min_length=1)
    association: ScalarAssociation | None = None
    georeference: GeoReferenceConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tiling: TilingConfig | None = None
    reader: ReaderConfig | None = None
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_paths(self) -> "ConvertConfig":
        if self.input.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(
                f"input must be one of {', '.join(sorted(SUPPORTED_INPUT_SUFFIXES))}; got {self.input.suffix}"
            )
        if self.output == self.input:
            raise ValueError("output must differ from input")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConvertConfig":
        """Load YAML and resolve relative input/output paths from its directory."""

        config_path = Path(path).expanduser().resolve()
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"configuration file does not exist: {config_path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML configuration: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("configuration root must be a mapping")
        data = dict(document)
        for key in ("input", "output"):
            if key in data:
                value = Path(data[key]).expanduser()
                if not value.is_absolute():
                    value = config_path.parent / value
                data[key] = value.resolve()
        return cls.model_validate(data)

    def with_overrides(self, **overrides: Any) -> "ConvertConfig":
        """Return a fully revalidated copy using only non-None override values."""

        data = self.model_dump()
        data.update({key: value for key, value in overrides.items() if value is not None})
        return type(self).model_validate(data)


class VectorConfig(BaseModel):
    """Velocity-vector overlay parameters; shares the voxel georeference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_u: str = "u"
    field_v: str = "v"
    field_w: str = "w"
    step: int | tuple[int, int, int] | None = None
    arrow_length: float = 400.0
    streamlines: int = 150
    streamline_steps: int = 50
    streamline_step: float = 250.0
    seed: int = 0
    emit_field: bool = True
    field_step: int = 8
    reader: ReaderConfig | None = None


class PipelineConfig(BaseModel):
    """Unified pipeline configuration reused by ``run`` and ``vector --config``.

    A single ``georeference`` drives both the voxel tileset and the decoupled
    vector overlay, eliminating the manual ``--lon/--lat/--height`` duplication
    that previously had to stay in sync across two commands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: Path
    output: Path
    georeference: GeoReferenceConfig
    field_name: str = Field(default="density", min_length=1)
    association: ScalarAssociation | None = None
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tiling: TilingConfig | None = None
    reader: ReaderConfig | None = None
    vector: VectorConfig = Field(default_factory=VectorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load YAML and resolve relative input/output paths from its directory."""

        config_path = Path(path).expanduser().resolve()
        try:
            document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"configuration file does not exist: {config_path}") from error
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML configuration: {error}") from error
        if not isinstance(document, dict):
            raise ValueError("configuration root must be a mapping")
        data = dict(document)
        for key in ("input", "output"):
            if key in data:
                value = Path(data[key]).expanduser()
                if not value.is_absolute():
                    value = config_path.parent / value
                data[key] = value.resolve()
        return cls.model_validate(data)

    def convert_config(self, *, overwrite: bool = False) -> ConvertConfig:
        """Project the convert-relevant subset into a ``ConvertConfig``."""

        return ConvertConfig(
            input=self.input,
            output=self.output,
            field_name=self.field_name,
            association=self.association,
            georeference=self.georeference,
            preprocess=self.preprocess,
            tiling=self.tiling,
            reader=self.reader,
            overwrite=overwrite,
        )

    def with_reader(self, reader: ReaderConfig) -> "PipelineConfig":
        """Return a copy with the reader hints replaced (CLI override of YAML)."""

        data = self.model_dump()
        data["reader"] = reader.model_dump()
        return type(self).model_validate(data)
