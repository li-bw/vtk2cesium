from pathlib import Path

import numpy as np
import pytest

from vtk2cesium.model import ScalarAssociation
from vtk2cesium.readers.vti import VtiReadError, inspect_vti, read_vti


def test_inspect_vti_reports_geometry_and_fields(sample_vti: Path) -> None:
    inspection = inspect_vti(sample_vti)

    assert inspection.point_dimensions == (3, 4, 5)
    assert inspection.cell_dimensions == (2, 3, 4)
    assert inspection.origin == (10.0, 20.0, 30.0)
    assert inspection.spacing == (2.0, 3.0, 4.0)
    assert inspection.bounds == (10.0, 14.0, 20.0, 29.0, 30.0, 46.0)
    assert [(field.association, field.name, field.components) for field in inspection.fields] == [
        (ScalarAssociation.POINT, "temperature", 1),
        (ScalarAssociation.POINT, "velocity", 3),
        (ScalarAssociation.CELL, "pressure", 1),
    ]


def test_read_active_point_scalar_preserves_corner_values(sample_vti: Path) -> None:
    dataset = read_vti(sample_vti, association="point")
    field = dataset.field("temperature")

    assert field.dimensions_xyz == (3, 4, 5)
    assert field.values.shape == (5, 4, 3)
    assert field.values[0, 0, 0] == 0.0
    assert field.values[0, 0, 1] == 1.0
    assert field.values[0, 1, 0] == 3.0
    assert field.values[1, 0, 0] == 12.0
    assert field.values[-1, -1, -1] == 59.0
    assert field.finite_range == ((0.0,), (59.0,))


def test_read_cell_scalar_uses_cell_dimensions(sample_vti: Path) -> None:
    dataset = read_vti(sample_vti, field_name="pressure")
    field = dataset.field("pressure")

    assert field.association is ScalarAssociation.CELL
    assert field.dimensions_xyz == (2, 3, 4)
    assert field.values.shape == (4, 3, 2)
    assert field.values[0, 0, 0] == 1000.0
    assert field.values[-1, -1, -1] == 1023.0


def test_read_vector_field_keeps_component_axis(sample_vti: Path) -> None:
    dataset = read_vti(sample_vti, field_name="velocity", association="point")
    values = dataset.field("velocity").values

    assert values.shape == (5, 4, 3, 3)
    assert np.array_equal(values[0, 0, 1], [1.0, 101.0, 201.0])
    assert np.array_equal(values[-1, -1, -1], [59.0, 159.0, 259.0])


def test_read_requires_explicit_name_when_point_fields_are_ambiguous(sample_vti: Path) -> None:
    import vtk

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(sample_vti))
    reader.Update()
    image = reader.GetOutput()
    image.GetPointData().SetActiveScalars(None)

    ambiguous = sample_vti.with_name("ambiguous.vti")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(ambiguous))
    writer.SetInputData(image)
    assert writer.Write() == 1

    with pytest.raises(VtiReadError, match="multiple fields"):
        read_vti(ambiguous, association="point")


def test_missing_field_lists_available_fields(sample_vti: Path) -> None:
    with pytest.raises(VtiReadError, match="point:temperature"):
        read_vti(sample_vti, field_name="missing")


def test_wrong_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.txt"
    path.write_text("not vtk", encoding="utf-8")
    with pytest.raises(VtiReadError, match="expected a .vti"):
        inspect_vti(path)
