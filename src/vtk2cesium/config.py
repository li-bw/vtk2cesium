"""Validated conversion configuration with YAML loading support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from vtk2cesium.model import ScalarAssociation
from vtk2cesium.transfer import NonFinitePolicy, ScalarMapping, ScalarPreprocessConfig


class GeoReferenceConfig(BaseModel):
    """Explicit WGS84 anchor for a local ENU VTK data set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    height: float = 0.0


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
    """Complete SDK/CLI configuration for one VTI conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input: Path
    output: Path
    field_name: str = Field(min_length=1)
    association: ScalarAssociation | None = None
    georeference: GeoReferenceConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    tiling: TilingConfig | None = None
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_paths(self) -> "ConvertConfig":
        if self.input.suffix.lower() != ".vti":
            raise ValueError("input must be a .vti file")
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
