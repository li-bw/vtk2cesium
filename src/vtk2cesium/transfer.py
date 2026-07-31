"""Scalar preprocessing and colour/opacity transfer functions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Iterable

import numpy as np
import numpy.typing as npt

NumericArray = npt.NDArray[np.number]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


class ScalarMapping(StrEnum):
    """Supported scalar value mappings."""

    IDENTITY = "identity"
    LINEAR = "linear"
    LOG = "log"
    PIECEWISE = "piecewise"


class NonFinitePolicy(StrEnum):
    """How preprocessing handles NaN and infinite values."""

    MASK = "mask"
    FILL = "fill"
    ERROR = "error"


@dataclass(frozen=True)
class ScalarPreprocessConfig:
    """Configuration for an order-preserving scalar preprocessing pass."""

    mapping: ScalarMapping = ScalarMapping.IDENTITY
    source_range: tuple[float, float] | None = None
    clip: bool = True
    threshold: tuple[float | None, float | None] | None = None
    non_finite: NonFinitePolicy = NonFinitePolicy.MASK
    fill_value: float = 0.0
    log_base: float = 10.0
    piecewise_points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", ScalarMapping(self.mapping))
        object.__setattr__(self, "non_finite", NonFinitePolicy(self.non_finite))
        if self.source_range is not None:
            _validate_range(self.source_range, "source_range")
        if self.threshold is not None:
            lower, upper = self.threshold
            if lower is not None and not math.isfinite(lower):
                raise ValueError("threshold lower bound must be finite")
            if upper is not None and not math.isfinite(upper):
                raise ValueError("threshold upper bound must be finite")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("threshold lower bound must not exceed upper bound")
        if not math.isfinite(self.fill_value):
            raise ValueError("fill_value must be finite")
        if not math.isfinite(self.log_base) or self.log_base <= 0.0 or self.log_base == 1.0:
            raise ValueError("log_base must be positive and not equal to 1")
        if self.mapping is ScalarMapping.PIECEWISE:
            _validated_piecewise_points(self.piecewise_points)


@dataclass(frozen=True)
class PreprocessedScalar:
    """Preprocessed values and an aligned validity mask."""

    values: Float32Array
    mask: BoolArray
    source_range: tuple[float, float] | None
    output_range: tuple[float, float] | None

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        mask = np.ascontiguousarray(self.mask, dtype=np.bool_)
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes")
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)


def preprocess_scalar(
    values: NumericArray,
    config: ScalarPreprocessConfig | None = None,
) -> PreprocessedScalar:
    """Preprocess scalar values without changing their shape or flattening order.

    The returned mask is true for finite values that pass the optional threshold.
    Masked values remain aligned with the input and receive ``fill_value``.
    """

    settings = config or ScalarPreprocessConfig()
    source = np.asarray(values)
    if source.ndim == 0 or not np.issubdtype(source.dtype, np.number):
        raise TypeError("values must be a non-scalar numeric NumPy array")

    working = np.asarray(source, dtype=np.float64)
    finite = np.isfinite(working)
    if settings.non_finite is NonFinitePolicy.ERROR and not finite.all():
        raise ValueError("values contain NaN or infinity")

    finite_values = working[finite]
    observed_range = _finite_range(finite_values)
    output = np.full(working.shape, settings.fill_value, dtype=np.float64)
    if finite_values.size:
        mapped = _map_finite_values(working[finite], settings, observed_range)
        output[finite] = mapped

    if settings.non_finite is NonFinitePolicy.FILL:
        mask = np.ones(working.shape, dtype=np.bool_)
    else:
        mask = finite.copy()
    if settings.threshold is not None:
        lower, upper = settings.threshold
        if lower is not None:
            mask &= working >= lower
        if upper is not None:
            mask &= working <= upper
    output[~mask] = settings.fill_value

    output_range = _finite_range(output[mask]) if mask.any() else None
    return PreprocessedScalar(
        values=output.astype(np.float32),
        mask=mask,
        source_range=observed_range,
        output_range=output_range,
    )


def _map_finite_values(
    values: npt.NDArray[np.float64],
    config: ScalarPreprocessConfig,
    observed_range: tuple[float, float] | None,
) -> npt.NDArray[np.float64]:
    if config.mapping is ScalarMapping.IDENTITY:
        return values.copy()
    if config.mapping is ScalarMapping.PIECEWISE:
        points = _validated_piecewise_points(config.piecewise_points)
        xp = np.array([point[0] for point in points], dtype=np.float64)
        fp = np.array([point[1] for point in points], dtype=np.float64)
        if config.clip:
            return np.interp(values, xp, fp)
        result = np.interp(values, xp, fp)
        result[values < xp[0]] = values[values < xp[0]]
        result[values > xp[-1]] = values[values > xp[-1]]
        return result

    source_minimum, source_maximum = config.source_range or observed_range or (0.0, 0.0)
    if source_minimum == source_maximum:
        return np.zeros_like(values)
    normalized = (values - source_minimum) / (source_maximum - source_minimum)
    if config.clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    if config.mapping is ScalarMapping.LINEAR:
        return normalized
    if config.mapping is ScalarMapping.LOG:
        if np.any(normalized < 0.0):
            raise ValueError("log mapping requires normalized values to be non-negative")
        return np.log1p((config.log_base - 1.0) * normalized) / math.log(config.log_base)
    raise AssertionError(f"unsupported mapping: {config.mapping}")


def _finite_range(values: npt.NDArray[np.float64]) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def _validate_range(value_range: tuple[float, float], name: str) -> None:
    if len(value_range) != 2 or not all(math.isfinite(value) for value in value_range):
        raise ValueError(f"{name} must contain two finite values")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{name} minimum must not exceed maximum")


def _validated_piecewise_points(
    points: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    normalized = tuple((float(x), float(y)) for x, y in points)
    if len(normalized) < 2:
        raise ValueError("piecewise mapping requires at least two control points")
    if not all(math.isfinite(value) for point in normalized for value in point):
        raise ValueError("piecewise control points must be finite")
    if any(left[0] >= right[0] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("piecewise control point inputs must be strictly increasing")
    return normalized


@dataclass(frozen=True)
class TransferPoint:
    """One scalar-to-RGBA control point."""

    value: float
    color: tuple[float, float, float]
    opacity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("transfer point value must be finite")
        if len(self.color) != 3 or not all(math.isfinite(channel) for channel in self.color):
            raise ValueError("transfer point color must contain three finite channels")
        if not all(0.0 <= channel <= 1.0 for channel in self.color):
            raise ValueError("transfer point color channels must be in [0, 1]")
        if not math.isfinite(self.opacity) or not 0.0 <= self.opacity <= 1.0:
            raise ValueError("transfer point opacity must be in [0, 1]")


@dataclass(frozen=True)
class TransferFunction:
    """Piecewise-linear scalar-to-RGBA transfer function."""

    points: tuple[TransferPoint, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) < 2:
            raise ValueError("transfer function requires at least two points")
        if any(left.value >= right.value for left, right in zip(points, points[1:])):
            raise ValueError("transfer point values must be strictly increasing")
        object.__setattr__(self, "points", points)

    def sample(self, values: NumericArray) -> Float32Array:
        """Sample RGBA values; inputs outside the domain clamp to endpoint colours."""

        source = np.asarray(values)
        if not np.issubdtype(source.dtype, np.number):
            raise TypeError("values must be numeric")
        controls = np.array([point.value for point in self.points], dtype=np.float64)
        channels = np.array(
            [(*point.color, point.opacity) for point in self.points], dtype=np.float64
        )
        flattened = np.asarray(source, dtype=np.float64).ravel(order="C")
        sampled = np.empty((flattened.size, 4), dtype=np.float64)
        for index in range(4):
            sampled[:, index] = np.interp(flattened, controls, channels[:, index])
        return np.ascontiguousarray(sampled.reshape(source.shape + (4,)), dtype=np.float32)

    def lookup_table(self, size: int = 256) -> Float32Array:
        """Build an evenly sampled RGBA lookup table over the control-point domain."""

        if size < 2:
            raise ValueError("lookup table size must be at least 2")
        samples = np.linspace(self.points[0].value, self.points[-1].value, size)
        return self.sample(samples)
