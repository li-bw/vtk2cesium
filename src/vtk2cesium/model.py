"""Validated domain models shared by VTK readers and tile encoders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
NumericArray = npt.NDArray[np.number]


class ScalarAssociation(StrEnum):
    """Location at which a VTK data array is defined."""

    POINT = "point"
    CELL = "cell"


@dataclass(frozen=True)
class ScalarFieldInfo:
    """Metadata describing one readable numeric VTK array."""

    name: str
    association: ScalarAssociation
    components: int
    tuples: int
    dtype: np.dtype
    finite_minimum: tuple[float, ...] | None = None
    finite_maximum: tuple[float, ...] | None = None
    non_finite_count: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must not be empty")
        if self.components <= 0:
            raise ValueError("field components must be positive")
        if self.tuples <= 0:
            raise ValueError("field tuples must be positive")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.non_finite_count < 0:
            raise ValueError("non_finite_count must not be negative")


@dataclass(frozen=True)
class VtiInspection:
    """Cheap structural summary used to select a scalar field before loading."""

    point_dimensions: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    bounds: tuple[float, float, float, float, float, float]
    fields: tuple[ScalarFieldInfo, ...]

    def __post_init__(self) -> None:
        if len(self.point_dimensions) != 3 or min(self.point_dimensions) <= 0:
            raise ValueError("point_dimensions must contain three positive integers")
        if len(self.origin) != 3 or len(self.spacing) != 3 or len(self.bounds) != 6:
            raise ValueError("invalid VTI geometry tuple length")
        if any(value == 0.0 for value in self.spacing):
            raise ValueError("spacing components must not be zero")

    @property
    def cell_dimensions(self) -> tuple[int, int, int]:
        return tuple(max(size - 1, 0) for size in self.point_dimensions)  # type: ignore[return-value]


@dataclass(frozen=True)
class ScalarField:
    """A numeric scalar or vector field stored as `(z, y, x, components)`."""

    name: str
    association: ScalarAssociation
    values: NumericArray

    def __post_init__(self) -> None:
        array = np.asarray(self.values)
        if not self.name:
            raise ValueError("field name must not be empty")
        if array.ndim not in (3, 4):
            raise ValueError("field values must have shape (z, y, x) or (z, y, x, components)")
        if any(size <= 0 for size in array.shape[:3]):
            raise ValueError("field spatial dimensions must be positive")
        if array.ndim == 4 and array.shape[3] <= 0:
            raise ValueError("field component count must be positive")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("field values must have a numeric dtype")
        array = np.ascontiguousarray(array)
        array.setflags(write=False)
        object.__setattr__(self, "values", array)

    @property
    def spatial_shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(size) for size in self.values.shape[:3])  # type: ignore[return-value]

    @property
    def dimensions_xyz(self) -> tuple[int, int, int]:
        z_size, y_size, x_size = self.spatial_shape_zyx
        return x_size, y_size, z_size

    @property
    def components(self) -> int:
        return 1 if self.values.ndim == 3 else int(self.values.shape[3])

    @property
    def finite_range(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        components = self.values[..., np.newaxis] if self.values.ndim == 3 else self.values
        minima: list[float] = []
        maxima: list[float] = []
        for index in range(components.shape[-1]):
            component = components[..., index]
            finite = component[np.isfinite(component)]
            if finite.size == 0:
                return None
            minima.append(float(finite.min()))
            maxima.append(float(finite.max()))
        return tuple(minima), tuple(maxima)


@dataclass(frozen=True)
class StructuredVoxelDataset:
    """A validated VTK ImageData data set with one or more aligned fields."""

    point_dimensions: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    bounds: tuple[float, float, float, float, float, float]
    fields: Mapping[str, ScalarField]

    def __post_init__(self) -> None:
        if len(self.point_dimensions) != 3 or min(self.point_dimensions) <= 0:
            raise ValueError("point_dimensions must contain three positive integers")
        if len(self.origin) != 3 or len(self.spacing) != 3 or len(self.bounds) != 6:
            raise ValueError("invalid geometry tuple length")
        if any(value == 0.0 for value in self.spacing):
            raise ValueError("spacing components must not be zero")
        if not self.fields:
            raise ValueError("dataset must contain at least one field")

        normalized: dict[str, ScalarField] = {}
        point_dimensions = self.point_dimensions
        cell_dimensions = tuple(size - 1 for size in point_dimensions)
        for key, field in self.fields.items():
            if key != field.name:
                raise ValueError("field mapping key must match field.name")
            expected = point_dimensions if field.association is ScalarAssociation.POINT else cell_dimensions
            if field.dimensions_xyz != expected:
                raise ValueError(
                    f"field {field.name!r} dimensions {field.dimensions_xyz} do not match "
                    f"{field.association.value} dimensions {expected}"
                )
            normalized[key] = field
        object.__setattr__(self, "fields", MappingProxyType(normalized))

    @property
    def cell_dimensions(self) -> tuple[int, int, int]:
        return tuple(size - 1 for size in self.point_dimensions)  # type: ignore[return-value]

    def field(self, name: str) -> ScalarField:
        try:
            return self.fields[name]
        except KeyError as error:
            available = ", ".join(sorted(self.fields))
            raise KeyError(f"unknown field {name!r}; available fields: {available}") from error
