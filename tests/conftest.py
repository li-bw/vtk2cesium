from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_vti(tmp_path: Path) -> Path:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    point_dimensions = (3, 4, 5)
    point_count = int(np.prod(point_dimensions))
    cell_dimensions = tuple(size - 1 for size in point_dimensions)
    cell_count = int(np.prod(cell_dimensions))

    image = vtk.vtkImageData()
    image.SetDimensions(*point_dimensions)
    image.SetOrigin(10.0, 20.0, 30.0)
    image.SetSpacing(2.0, 3.0, 4.0)

    point_values = np.arange(point_count, dtype=np.float32)
    point_array = numpy_to_vtk(point_values, deep=True)
    point_array.SetName("temperature")
    image.GetPointData().SetScalars(point_array)

    vector_values = np.column_stack(
        [point_values, point_values + 100.0, point_values + 200.0]
    ).astype(np.float32)
    vector_array = numpy_to_vtk(vector_values, deep=True)
    vector_array.SetName("velocity")
    image.GetPointData().AddArray(vector_array)

    cell_values = (np.arange(cell_count, dtype=np.float32) + 1000.0).astype(np.float32)
    cell_array = numpy_to_vtk(cell_values, deep=True)
    cell_array.SetName("pressure")
    image.GetCellData().SetScalars(cell_array)

    path = tmp_path / "sample.vti"
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError("failed to write VTI fixture")
    return path
