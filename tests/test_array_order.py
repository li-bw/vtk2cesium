import numpy as np
import pytest

from vtk2cesium.array_order import (
    cesium_box_gltf_dimensions,
    vtk_flat_to_zyx,
    zyx_to_cesium_buffer,
)


def test_vtk_flat_to_zyx_preserves_x_fastest_order() -> None:
    flat = np.arange(24, dtype=np.float32)
    values = vtk_flat_to_zyx(flat, (2, 3, 4))

    assert values.shape == (4, 3, 2)
    assert values[0, 0, 0] == 0
    assert values[0, 0, 1] == 1
    assert values[0, 1, 0] == 2
    assert values[1, 0, 0] == 6
    assert values[-1, -1, -1] == 23


def test_multicomponent_values_keep_components_innermost() -> None:
    flat = np.arange(16, dtype=np.int16)
    values = vtk_flat_to_zyx(flat, (2, 2, 2), components=2)

    assert values.shape == (2, 2, 2, 2)
    assert np.array_equal(values[0, 0, 1], [2, 3])
    assert np.array_equal(values[-1, -1, -1], [14, 15])


def test_zyx_to_cesium_buffer_round_trips() -> None:
    values = np.arange(24, dtype=np.float64).reshape((4, 3, 2))
    payload = zyx_to_cesium_buffer(values)

    assert np.array_equal(np.frombuffer(payload, dtype="<f4"), np.arange(24, dtype=np.float32))


def test_array_size_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 8"):
        vtk_flat_to_zyx(np.arange(7), (2, 2, 2))


def test_cesium_box_dimensions_swap_y_and_z() -> None:
    assert cesium_box_gltf_dimensions((2, 3, 4)) == (2, 4, 3)
