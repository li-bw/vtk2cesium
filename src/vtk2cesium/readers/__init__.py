"""Structured-grid input readers for the VTK -> Cesium voxel pipeline.

The package exposes a single unified entry point, ``read_dataset`` (and its
``inspect_dataset`` companion), which dispatches by file extension to the
VTI, NetCDF, or GeoTIFF adapter. Every adapter emits the same contract:
a local ENU frame in metres (``origin`` is the south-west-bottom corner,
``spacing`` is the per-axis cell size) wrapped in a ``StructuredVoxelDataset``.

Format-specific hints are passed through a ``reader`` object (any object exposing
``reference_latitude``/``x_dim``/``y_dim``/``z_dim``/``band_as_field``; the
Pydantic ``ReaderConfig`` in ``vtk2cesium.config`` is the canonical one).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtk2cesium.model import StructuredVoxelDataset, VtiInspection
from vtk2cesium.readers.vti import VtiReadError, inspect_vti, read_vti

__all__ = [
    "VtiReadError",
    "inspect_vti",
    "read_vti",
    "inspect_dataset",
    "read_dataset",
]


_SUFFIX_HANDLERS = {
    ".vti": "vti",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".cdf": "netcdf",
    ".tif": "geotiff",
    ".tiff": "geotiff",
}


def _detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in _SUFFIX_HANDLERS:
        raise ValueError(
            f"unsupported input format {suffix!r}; expected one of "
            f"{', '.join(sorted(_SUFFIX_HANDLERS))}"
        )
    return _SUFFIX_HANDLERS[suffix]


def _reader_attr(reader: Any, name: str, default: Any = None) -> Any:
    return getattr(reader, name, default) if reader is not None else default


def inspect_dataset(path: str | Path, *, reader: Any = None) -> VtiInspection:
    """Enumerate fields/ranges for any supported input without selecting one."""

    fmt = _detect_format(path)
    if fmt == "vti":
        return inspect_vti(path)
    if fmt == "netcdf":
        from vtk2cesium.readers.netcdf import inspect_netcdf

        return inspect_netcdf(
            path,
            reference_latitude=_reader_attr(reader, "reference_latitude"),
            x_dim=_reader_attr(reader, "x_dim"),
            y_dim=_reader_attr(reader, "y_dim"),
            z_dim=_reader_attr(reader, "z_dim"),
        )
    from vtk2cesium.readers.geotiff import inspect_geotiff

    return inspect_geotiff(
        path,
        reference_latitude=_reader_attr(reader, "reference_latitude"),
        band_as_field=_reader_attr(reader, "band_as_field", True),
    )


def read_dataset(
    path: str | Path,
    *,
    field_name: str | None = None,
    association: Any = None,
    reader: Any = None,
) -> StructuredVoxelDataset:
    """Load one (or all) fields from any supported input into a voxel dataset."""

    fmt = _detect_format(path)
    if fmt == "vti":
        return read_vti(path, field_name=field_name, association=association)
    if fmt == "netcdf":
        from vtk2cesium.readers.netcdf import read_netcdf

        return read_netcdf(
            path,
            field_name=field_name,
            reference_latitude=_reader_attr(reader, "reference_latitude"),
            x_dim=_reader_attr(reader, "x_dim"),
            y_dim=_reader_attr(reader, "y_dim"),
            z_dim=_reader_attr(reader, "z_dim"),
        )
    from vtk2cesium.readers.geotiff import read_geotiff

    return read_geotiff(
        path,
        field_name=field_name,
        reference_latitude=_reader_attr(reader, "reference_latitude"),
        band_as_field=_reader_attr(reader, "band_as_field", True),
    )


# Adapter-specific errors, re-exported for convenient catch sites.
from vtk2cesium.readers.netcdf import NetcdfReadError  # noqa: E402
from vtk2cesium.readers.geotiff import GeotiffReadError  # noqa: E402

__all__ += ["NetcdfReadError", "GeotiffReadError"]
