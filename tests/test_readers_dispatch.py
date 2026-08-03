"""Tests for the unified reader dispatch (format routing by extension)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vtk2cesium.readers import inspect_dataset, read_dataset
from vtk2cesium.readers.netcdf import NetcdfReadError
from vtk2cesium.readers.geotiff import GeotiffReadError


def test_dispatch_unsupported_suffix_raises():
    with pytest.raises(ValueError):
        read_dataset(Path("data.xyz"))


def test_dispatch_vti_routes_without_optional_deps(sample_vti):
    # VTI path must work regardless of netCDF4/rasterio availability.
    # `pressure` is a cell field in the fixture, so select cell association.
    dataset = read_dataset(sample_vti, field_name="pressure", association="cell")
    assert dataset.point_dimensions == (3, 4, 5)
    assert "pressure" in dataset.fields


def test_inspect_vti_routes(sample_vti):
    inspection = inspect_dataset(sample_vti)
    assert inspection.point_dimensions == (3, 4, 5)


def test_dispatch_netcdf_missing_dep_error_message(tmp_path):
    # If netCDF4 is not installed, the dispatch must raise a clear error.
    try:
        import netCDF4  # noqa: F401
    except ImportError:
        nc = tmp_path / "x.nc"
        nc.write_text("not a real netcdf")
        with pytest.raises((NetcdfReadError, ValueError)):
            read_dataset(nc)


def test_dispatch_geotiff_missing_dep_error_message(tmp_path):
    try:
        import rasterio  # noqa: F401
    except ImportError:
        tif = tmp_path / "x.tif"
        tif.write_text("not a real tiff")
        with pytest.raises((GeotiffReadError, ValueError)):
            read_dataset(tif)
