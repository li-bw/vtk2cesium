"""Read VTK XML ImageData (`.vti`) into validated NumPy domain models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vtk2cesium.array_order import vtk_flat_to_zyx
from vtk2cesium.model import (
    ScalarAssociation,
    ScalarField,
    ScalarFieldInfo,
    StructuredVoxelDataset,
    VtiInspection,
)


class VtiReadError(ValueError):
    """Raised when a VTI file cannot be interpreted safely."""


def inspect_vti(path: str | Path) -> VtiInspection:
    """Read VTI metadata and enumerate named numeric point/cell arrays."""

    image = _read_image_data(_validated_path(path))
    point_dimensions = tuple(int(value) for value in image.GetDimensions())
    fields = tuple(
        _iter_field_info(image.GetPointData(), ScalarAssociation.POINT)
    ) + tuple(_iter_field_info(image.GetCellData(), ScalarAssociation.CELL))
    return VtiInspection(
        point_dimensions=point_dimensions,
        origin=tuple(float(value) for value in image.GetOrigin()),
        spacing=tuple(float(value) for value in image.GetSpacing()),
        bounds=tuple(float(value) for value in image.GetBounds()),
        fields=fields,
    )


def read_vti(
    path: str | Path,
    *,
    field_name: str | None = None,
    association: ScalarAssociation | str | None = None,
) -> StructuredVoxelDataset:
    """Load one selected VTI numeric array into a structured voxel dataset.

    If no field name is supplied, the active scalar array is preferred. If no
    active scalar exists, a file with exactly one named numeric array is
    accepted. Ambiguous files require an explicit field name.
    """

    image = _read_image_data(_validated_path(path))
    point_dimensions = tuple(int(value) for value in image.GetDimensions())
    selected_association = _coerce_association(association)
    resolved_name, resolved_association, vtk_array = _select_array(
        image,
        field_name=field_name,
        association=selected_association,
    )

    dimensions_xyz = (
        point_dimensions
        if resolved_association is ScalarAssociation.POINT
        else tuple(size - 1 for size in point_dimensions)
    )
    if min(dimensions_xyz) <= 0:
        raise VtiReadError(
            f"{resolved_association.value} data requires positive dimensions; got {dimensions_xyz}"
        )

    from vtk.util.numpy_support import vtk_to_numpy

    numpy_values = vtk_to_numpy(vtk_array)
    components = int(vtk_array.GetNumberOfComponents())
    values = vtk_flat_to_zyx(numpy_values, dimensions_xyz, components=components)
    field = ScalarField(
        name=resolved_name,
        association=resolved_association,
        values=values,
    )
    return StructuredVoxelDataset(
        point_dimensions=point_dimensions,
        origin=tuple(float(value) for value in image.GetOrigin()),
        spacing=tuple(float(value) for value in image.GetSpacing()),
        bounds=tuple(float(value) for value in image.GetBounds()),
        fields={resolved_name: field},
    )


def _validated_path(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.exists():
        raise FileNotFoundError(f"VTI file does not exist: {value}")
    if not value.is_file():
        raise VtiReadError(f"VTI path is not a file: {value}")
    if value.suffix.lower() != ".vti":
        raise VtiReadError(f"expected a .vti file, got: {value.name}")
    return value


def _read_image_data(path: Path) -> Any:
    import vtk

    reader = vtk.vtkXMLImageDataReader()
    if not reader.CanReadFile(str(path)):
        raise VtiReadError(f"VTK cannot read this file as XML ImageData: {path}")
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    if not isinstance(output, vtk.vtkImageData):
        actual = output.GetClassName() if output is not None else "None"
        raise VtiReadError(f"expected vtkImageData, got {actual}: {path}")
    dimensions = tuple(int(value) for value in output.GetDimensions())
    if min(dimensions) <= 0:
        raise VtiReadError(f"VTI has empty point dimensions: {dimensions}")
    return output


def _iter_field_info(attributes: Any, association: ScalarAssociation) -> Iterable[ScalarFieldInfo]:
    from vtk.util.numpy_support import vtk_to_numpy

    for index in range(attributes.GetNumberOfArrays()):
        vtk_array = attributes.GetArray(index)
        if vtk_array is None:
            continue
        name = vtk_array.GetName()
        if not name:
            continue
        values = vtk_to_numpy(vtk_array)
        if not np.issubdtype(values.dtype, np.number):
            continue
        components = int(vtk_array.GetNumberOfComponents())
        component_values = values.reshape((-1, components))
        minima: list[float] = []
        maxima: list[float] = []
        non_finite_count = 0
        has_finite_values = True
        for component_index in range(components):
            component = component_values[:, component_index]
            finite_mask = np.isfinite(component)
            non_finite_count += int((~finite_mask).sum())
            finite = component[finite_mask]
            if finite.size == 0:
                has_finite_values = False
                minima.append(float("nan"))
                maxima.append(float("nan"))
            else:
                minima.append(float(finite.min()))
                maxima.append(float(finite.max()))
        yield ScalarFieldInfo(
            name=name,
            association=association,
            components=components,
            tuples=int(vtk_array.GetNumberOfTuples()),
            dtype=values.dtype,
            finite_minimum=tuple(minima) if has_finite_values else None,
            finite_maximum=tuple(maxima) if has_finite_values else None,
            non_finite_count=non_finite_count,
        )


def _coerce_association(value: ScalarAssociation | str | None) -> ScalarAssociation | None:
    if value is None or isinstance(value, ScalarAssociation):
        return value
    try:
        return ScalarAssociation(value.lower())
    except ValueError as error:
        raise VtiReadError("association must be 'point' or 'cell'") from error


def _select_array(
    image: Any,
    *,
    field_name: str | None,
    association: ScalarAssociation | None,
) -> tuple[str, ScalarAssociation, Any]:
    candidates: list[tuple[str, ScalarAssociation, Any]] = []
    locations = (
        (ScalarAssociation.POINT, image.GetPointData()),
        (ScalarAssociation.CELL, image.GetCellData()),
    )
    for current_association, attributes in locations:
        if association is not None and association is not current_association:
            continue
        for index in range(attributes.GetNumberOfArrays()):
            vtk_array = attributes.GetArray(index)
            name = vtk_array.GetName() if vtk_array is not None else None
            if vtk_array is not None and name:
                candidates.append((name, current_association, vtk_array))

    if field_name is not None:
        matches = [candidate for candidate in candidates if candidate[0] == field_name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise VtiReadError(
                f"field {field_name!r} exists in point and cell data; specify association"
            )
        available = ", ".join(f"{item[1].value}:{item[0]}" for item in candidates) or "none"
        raise VtiReadError(f"field {field_name!r} not found; available fields: {available}")

    active_candidates: list[tuple[str, ScalarAssociation, Any]] = []
    for current_association, attributes in locations:
        if association is not None and association is not current_association:
            continue
        active = attributes.GetScalars()
        if active is not None and active.GetName():
            active_candidates.append((active.GetName(), current_association, active))
    if len(active_candidates) == 1:
        return active_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise VtiReadError("VTI contains no named numeric point or cell arrays")
    available = ", ".join(f"{item[1].value}:{item[0]}" for item in candidates)
    raise VtiReadError(f"multiple fields are available; select one explicitly: {available}")
