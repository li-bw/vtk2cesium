"""Central conversions between VTK, NumPy, and Cesium voxel array order."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NumericArray = npt.NDArray[np.number]


def vtk_flat_to_zyx(
    values: NumericArray,
    dimensions_xyz: tuple[int, int, int],
    *,
    components: int = 1,
) -> NumericArray:
    """View a VTK tuple array as `(z, y, x[, components])`.

    VTK ImageData enumerates X fastest, then Y, then Z. A C-order NumPy
    `(z, y, x)` array has the same physical flattening order.
    """

    if len(dimensions_xyz) != 3 or min(dimensions_xyz) <= 0:
        raise ValueError("dimensions_xyz must contain three positive integers")
    if components <= 0:
        raise ValueError("components must be positive")

    array = np.asarray(values)
    x_size, y_size, z_size = dimensions_xyz
    expected = x_size * y_size * z_size * components
    if array.size != expected:
        raise ValueError(f"array has {array.size} values; expected {expected}")

    spatial_shape = (z_size, y_size, x_size)
    if components == 1:
        return array.reshape(spatial_shape, order="C")
    return array.reshape((*spatial_shape, components), order="C")


def zyx_to_cesium_buffer(values: NumericArray, *, dtype: np.dtype | str = "<f4") -> bytes:
    """Serialize `(z, y, x[, components])` values with X moving fastest."""

    array = np.asarray(values)
    if array.ndim not in (3, 4):
        raise ValueError("values must have shape (z, y, x) or (z, y, x, components)")
    if any(size <= 0 for size in array.shape[:3]):
        raise ValueError("spatial dimensions must be positive")
    target = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return target.tobytes(order="C")


def cesium_box_gltf_dimensions(dimensions_xyz: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return CesiumJS BOX glTF dimensions in its Y-up `(x, z, y)` order."""

    if len(dimensions_xyz) != 3 or min(dimensions_xyz) <= 0:
        raise ValueError("dimensions_xyz must contain three positive integers")
    x_size, y_size, z_size = dimensions_xyz
    return x_size, z_size, y_size
