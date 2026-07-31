import numpy as np
import pytest

from vtk2cesium.formats.glb import parse_glb
from vtk2cesium.formats.voxel_gltf import VOXEL_MODE, encode_scalar_glb, encode_scalar_tile


def test_encode_scalar_tile_uses_cesium_box_dimension_order() -> None:
    values = np.arange(24, dtype=np.float32).reshape((3, 4, 2))
    result = encode_scalar_tile(values, property_name="density")

    primitive = result.document["meshes"][0]["primitives"][0]
    extension = primitive["extensions"]["EXT_primitive_voxels"]

    assert primitive["mode"] == VOXEL_MODE
    assert extension["dimensions"] == [2, 3, 4]
    assert np.array_equal(np.frombuffer(result.binary, dtype="<f4"), np.arange(24))


def test_encode_scalar_glb_embeds_float32_payload() -> None:
    values = np.arange(24, dtype=np.float32).reshape((3, 4, 2))
    parsed = parse_glb(encode_scalar_glb(values, property_name="density"))

    assert "uri" not in parsed.document["buffers"][0]
    assert parsed.document["buffers"][0]["byteLength"] == 24 * 4
    assert np.array_equal(np.frombuffer(parsed.binary[: 24 * 4], dtype="<f4"), np.arange(24))


def test_encode_scalar_tile_rejects_non_finite_data() -> None:
    values = np.zeros((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encode_scalar_tile(values)
