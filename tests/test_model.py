import numpy as np
import pytest

from vtk2cesium.model import ScalarAssociation, ScalarField, StructuredVoxelDataset


def test_scalar_field_is_read_only_and_reports_finite_range() -> None:
    values = np.array([[[0.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    field = ScalarField("density", ScalarAssociation.POINT, values)

    assert field.dimensions_xyz == (2, 2, 1)
    assert field.finite_range == ((0.0,), (3.0,))
    with pytest.raises(ValueError):
        field.values[0, 0, 0] = 5.0


def test_dataset_rejects_misaligned_field() -> None:
    field = ScalarField(
        "density",
        ScalarAssociation.POINT,
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="do not match"):
        StructuredVoxelDataset(
            point_dimensions=(3, 2, 2),
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            bounds=(0.0, 2.0, 0.0, 1.0, 0.0, 1.0),
            fields={"density": field},
        )
