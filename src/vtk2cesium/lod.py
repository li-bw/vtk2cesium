"""Mask-aware 2x2x2 scalar LOD generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LodLevel:
    """One structured LOD array at a logical level, finest level is supplied by caller."""

    values: npt.NDArray[np.float32]
    validity: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(self.values, dtype=np.float32)
        validity = np.ascontiguousarray(self.validity, dtype=np.bool_)
        if values.ndim != 3 or values.shape != validity.shape:
            raise ValueError("LOD values and validity must have the same 3D shape")
        values.setflags(write=False)
        validity.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity", validity)


def downsample_2x2x2(
    values: npt.ArrayLike,
    validity: npt.ArrayLike,
    *,
    fill_value: float = 0.0,
) -> LodLevel:
    """Average valid samples in each 2x2x2 block, padding odd edges as invalid."""

    source = np.asarray(values, dtype=np.float32)
    mask = np.asarray(validity, dtype=np.bool_)
    if source.ndim != 3 or source.shape != mask.shape:
        raise ValueError("values and validity must have the same 3D shape")
    if not math.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    padded_shape = tuple(size + (size % 2) for size in source.shape)
    padded_values = np.full(padded_shape, fill_value, dtype=np.float32)
    padded_mask = np.zeros(padded_shape, dtype=np.bool_)
    slices = tuple(slice(0, size) for size in source.shape)
    padded_values[slices] = source
    padded_mask[slices] = mask

    z_size, y_size, x_size = padded_shape
    blocks = padded_values.reshape(z_size // 2, 2, y_size // 2, 2, x_size // 2, 2)
    block_mask = padded_mask.reshape(z_size // 2, 2, y_size // 2, 2, x_size // 2, 2)
    sums = np.where(block_mask, blocks, 0.0).sum(axis=(1, 3, 5), dtype=np.float64)
    counts = block_mask.sum(axis=(1, 3, 5))
    output_mask = counts > 0
    output = np.full(counts.shape, fill_value, dtype=np.float32)
    output[output_mask] = (sums[output_mask] / counts[output_mask]).astype(np.float32)
    return LodLevel(output, output_mask)


def build_lod_pyramid(
    values: npt.ArrayLike,
    validity: npt.ArrayLike,
    *,
    available_levels: int,
    fill_value: float = 0.0,
) -> tuple[LodLevel, ...]:
    """Return coarsest-to-finest arrays for implicit levels 0..N-1."""

    if available_levels <= 0:
        raise ValueError("available_levels must be positive")
    finest = LodLevel(np.asarray(values, dtype=np.float32), np.asarray(validity, dtype=np.bool_))
    reversed_levels = [finest]
    for _ in range(available_levels - 1):
        current = reversed_levels[-1]
        reversed_levels.append(
            downsample_2x2x2(current.values, current.validity, fill_value=fill_value)
        )
    return tuple(reversed(reversed_levels))
