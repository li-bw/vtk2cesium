from pathlib import Path

import numpy as np
import pytest

from vtk2cesium.probe import gradient, write_probe
from vtk2cesium.validate import validate_probe


def test_gradient_is_xyz_monotonic() -> None:
    values = gradient((2, 3, 4))
    assert values.shape == (4, 3, 2)
    assert np.isclose(values[0, 0, 0], 0.0)
    assert np.isclose(values[-1, -1, -1], 1.0)
    assert values[0, 0, 1] > values[0, 0, 0]
    assert values[0, 1, 0] > values[0, 0, 0]
    assert values[1, 0, 0] > values[0, 0, 0]


def test_write_and_validate_probe(tmp_path: Path) -> None:
    output = tmp_path / "probe"
    tileset = write_probe(output, dimensions=(2, 3, 4), property_name="density")
    result = validate_probe(tileset)

    assert result.dimensions == (2, 3, 4)
    assert result.value_count == 24
    assert result.property_name == "density"
    assert result.minimum == 0.0
    assert result.maximum == 1.0


def test_write_probe_refuses_non_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "probe"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        write_probe(output)
