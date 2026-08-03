"""Tests for the GeoTIFF reader adapter (P1-c).

Skips gracefully when the optional ``rasterio`` package is unavailable, and
builds a tiny in-memory GeoTIFF (degrees CRS) to assert the 2D (z=1) contract,
band-as-field handling, row flip (ENU y increases north), and degree->metre
spacing conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from vtk2cesium.model import ScalarAssociation, StructuredVoxelDataset
from vtk2cesium.readers.geotiff import GeotiffReadError, inspect_geotiff, read_geotiff


def _write_sample(tmp_path):
    from rasterio.transform import from_origin

    path = tmp_path / "sample.tif"
    height, width = 3, 4
    transform = from_origin(117.0, 36.3, 0.1, 0.1)  # NW corner, 0.1 deg res
    # band1: increasing east; band2: increasing north (so row flip matters)
    band1 = np.tile(np.arange(width, dtype="float64"), (height, 1))
    band2 = np.tile(np.arange(height, dtype="float64").reshape(-1, 1), (1, width))
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float64",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(band1, 1)
        dst.write(band2, 2)
        dst.set_band_description(1, "east")
        dst.set_band_description(2, "north")
    return path


def test_read_geotiff_layout_and_fields(tmp_path):
    path = _write_sample(tmp_path)
    dataset = read_geotiff(path)
    assert isinstance(dataset, StructuredVoxelDataset)
    assert dataset.point_dimensions == (4, 3, 1)  # width, height, single layer
    assert "east" in dataset.fields and "north" in dataset.fields
    assert dataset.fields["east"].association is ScalarAssociation.POINT
    # values are (z=1, y, x)
    assert dataset.fields["east"].values.shape == (1, 3, 4)


def test_read_geotiff_row_flip_north_increases(tmp_path):
    path = _write_sample(tmp_path)
    dataset = read_geotiff(path, field_name="north")
    values = dataset.fields["north"].values[0]  # (y, x)
    # band2 increases south (raster row 0 = north = 0, row 2 = south = 2). After
    # the flip, local y=0 is the SOUTH row, so the south value (2) sits at y=0
    # and the north value (0) sits at y=2 (ENU y increases north).
    assert values[0, 0] == 2.0
    assert values[2, 0] == 0.0


def test_read_geotiff_spacing_in_metres(tmp_path):
    path = _write_sample(tmp_path)
    dataset = read_geotiff(path, reference_latitude=36.15)
    sx, sy, sz = dataset.spacing
    expected_sx = 0.1 * 111_320.0 * np.cos(np.radians(36.15))
    expected_sy = 0.1 * 111_320.0
    assert abs(sx - expected_sx) < 1e-3
    assert abs(sy - expected_sy) < 1e-3
    assert sz == 1.0
    assert dataset.origin == (0.0, 0.0, 0.0)


def test_inspect_geotiff_lists_bands(tmp_path):
    path = _write_sample(tmp_path)
    inspection = inspect_geotiff(path)
    names = {f.name for f in inspection.fields}
    assert {"east", "north"} <= names


def test_read_geotiff_missing_field(tmp_path):
    path = _write_sample(tmp_path)
    with pytest.raises((GeotiffReadError, ValueError)):
        read_geotiff(path, field_name="missing")
