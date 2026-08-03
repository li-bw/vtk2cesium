"""Tests for the NetCDF reader adapter (P1-c).

Skips gracefully when the optional ``netCDF4`` package is unavailable, and
builds a tiny in-memory NetCDF file to assert the (z, y, x) contract.
"""

from __future__ import annotations

import numpy as np
import pytest

nc4 = pytest.importorskip("netCDF4")

from vtk2cesium.model import ScalarAssociation, StructuredVoxelDataset
from vtk2cesium.readers.netcdf import NetcdfReadError, inspect_netcdf, read_netcdf


def _write_sample(tmp_path):
    path = tmp_path / "sample.nc"
    ds = nc4.Dataset(str(path), "w")
    ds.createDimension("time", 2)
    ds.createDimension("level", 3)
    ds.createDimension("lat", 4)
    ds.createDimension("lon", 5)
    lat = ds.createVariable("lat", "f8", ("lat",))
    lon = ds.createVariable("lon", "f8", ("lon",))
    level = ds.createVariable("level", "f8", ("level",))
    lat[:] = np.linspace(36.0, 36.3, 4)  # degrees
    lon[:] = np.linspace(117.0, 117.4, 5)  # degrees
    level[:] = np.array([10.0, 20.0, 30.0])
    temp = ds.createVariable("temp", "f8", ("time", "level", "lat", "lon"))
    temp[:] = np.arange(2 * 3 * 4 * 5, dtype="f8").reshape(2, 3, 4, 5)
    ds.close()
    return path


def test_read_netcdf_layout_and_units(tmp_path):
    path = _write_sample(tmp_path)
    dataset = read_netcdf(path, field_name="temp")
    assert isinstance(dataset, StructuredVoxelDataset)
    # point_dimensions = (lon=5, lat=4, level=3)
    assert dataset.point_dimensions == (5, 4, 3)
    assert dataset.fields["temp"].association is ScalarAssociation.POINT
    # values stored as (z, y, x) = (level, lat, lon)
    values = dataset.fields["temp"].values
    assert values.shape == (3, 4, 5)
    # time (first dim) was dropped -> the value at z=0,y=0,x=0 is temp[0,0,0,0] = 0
    assert values[0, 0, 0] == 0.0
    # time dropped to index 0; level 2, lat 3, lon 4 -> 0*60 + 2*20 + 3*5 + 4 = 59
    assert values[2, 3, 4] == 59.0


def test_read_netcdf_spacing_in_metres(tmp_path):
    path = _write_sample(tmp_path)
    dataset = read_netcdf(path, field_name="temp", reference_latitude=36.15)
    sx, sy, sz = dataset.spacing
    # lon spacing 0.1 deg * 111320 * cos(36.15) ; lat spacing 0.1 deg * 111320
    expected_sx = 0.1 * 111_320.0 * np.cos(np.radians(36.15))
    expected_sy = 0.1 * 111_320.0
    assert abs(sx - expected_sx) < 1e-3
    assert abs(sy - expected_sy) < 1e-3
    assert abs(sz - 10.0) < 1e-9  # level delta in metres
    assert dataset.origin == (0.0, 0.0, 0.0)


def test_inspect_netcdf_lists_fields(tmp_path):
    path = _write_sample(tmp_path)
    inspection = inspect_netcdf(path)
    names = {f.name for f in inspection.fields}
    assert "temp" in names


def test_read_netcdf_ambiguous_without_field(tmp_path):
    path = _write_sample(tmp_path)
    # only one field -> selected automatically
    dataset = read_netcdf(path)
    assert "temp" in dataset.fields


def test_read_netcdf_missing_field(tmp_path):
    path = _write_sample(tmp_path)
    with pytest.raises((NetcdfReadError, ValueError)):
        read_netcdf(path, field_name="nope")


def test_read_netcdf_bad_axis_roles(tmp_path):
    path = tmp_path / "bad.nc"
    ds = nc4.Dataset(str(path), "w")
    ds.createDimension("a", 2)
    ds.createDimension("b", 3)
    v = ds.createVariable("v", "f8", ("a", "b"))
    v[:] = np.zeros((2, 3))
    ds.close()
    with pytest.raises(NetcdfReadError):
        read_netcdf(path, field_name="v")
