"""Tests for the decoupled vector overlay (arrows + streamlines)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vtk2cesium.geo import GeoReference
from vtk2cesium.model import ScalarAssociation, ScalarField, StructuredVoxelDataset
from vtk2cesium.vector_field import (
    build_vector_overlay,
    build_velocity_field,
    write_velocity_field,
    write_vector_overlay,
)


def _synthetic_dataset(
    shape_zyx=(5, 5, 5),
    *,
    u=3.0,
    v=4.0,
    w=0.0,
    origin=(0.0, 0.0, 0.0),
    spacing=(10.0, 10.0, 10.0),
):
    nz, ny, nx = shape_zyx
    values = np.empty((nz, ny, nx), dtype=np.float64)
    values[:] = 0.0
    fields = {
        "u": ScalarField(name="u", association=ScalarAssociation.POINT, values=values.copy() + u),
        "v": ScalarField(name="v", association=ScalarAssociation.POINT, values=values.copy() + v),
        "w": ScalarField(name="w", association=ScalarAssociation.POINT, values=values.copy() + w),
    }
    bounds = (
        origin[0],
        origin[0] + (nx - 1) * spacing[0],
        origin[1],
        origin[1] + (ny - 1) * spacing[1],
        origin[2],
        origin[2] + (nz - 1) * spacing[2],
    )
    return StructuredVoxelDataset(
        point_dimensions=(nx, ny, nz),
        origin=origin,
        spacing=spacing,
        bounds=bounds,
        fields=fields,
    )


def test_build_vector_overlay_arrows_align_with_transform():
    dataset = _synthetic_dataset()
    georeference = GeoReference(longitude=117.0, latitude=36.6, height=0.0)
    result = build_vector_overlay(
        dataset,
        georeference,
        u_name="u",
        v_name="v",
        w_name="w",
        step=1,
        arrow_length=20.0,
        streamline_count=0,
    )
    assert result.arrow_count == 5 * 5 * 5
    first = result.arrow_packets[0]
    assert first["id"] == "arrow-0"
    cartesian = np.asarray(first["polyline"]["positions"]["cartesian"], dtype=float).reshape(-1, 3)
    assert first["polyline"]["positions"].get("arcType") == "NONE"
    assert np.isfinite(cartesian).all()
    # Base of arrow for index (0,0,0) is the local origin; must match geo transform.
    expected = georeference.transform_points([dataset.origin])[0]
    assert np.linalg.norm(cartesian[0] - expected) < 1e-6
    # Arrow tip must be base + direction*length in ECEF (direction unit length).
    direction = np.array([3.0, 4.0, 0.0]) / 5.0
    expected_tip = georeference.transform_points(
        [tuple(np.asarray(dataset.origin) + direction * 20.0)]
    )[0]
    assert np.linalg.norm(cartesian[1] - expected_tip) < 1e-6


def test_integrate_streamline_follows_constant_field_direction():
    from vtk2cesium.vector_field import _integrate_streamline

    # Large enough domain that the 80 m line never hits a boundary.
    dataset = _synthetic_dataset(shape_zyx=(21, 21, 21), spacing=(10.0, 10.0, 10.0))
    points = _integrate_streamline(
        dataset,
        "u",
        "v",
        "w",
        start_enu=np.array([100.0, 100.0, 100.0]),
        sign=1,
        step_meters=10.0,
        steps=8,
    )
    # 8 forward steps => 8 points after the start, colinear along (0.6, 0.8, 0).
    assert len(points) == 8
    enu = np.asarray(points, dtype=float)
    direction = np.array([0.6, 0.8, 0.0])
    for index in range(len(enu) - 2):
        first = enu[index + 1] - enu[index]
        second = enu[index + 2] - enu[index + 1]
        if np.linalg.norm(first) < 1e-9 or np.linalg.norm(second) < 1e-9:
            continue
        cross = np.cross(first, second)
        assert np.linalg.norm(cross) < 1e-6
    # Endpoint direction matches the field direction (sign-agnostic).
    segment = enu[-1] - enu[0]
    segment = segment / np.linalg.norm(segment)
    assert abs(abs(float(np.dot(segment, direction))) - 1.0) < 1e-6
    total = float(np.sum(np.linalg.norm(np.diff(enu, axis=0), axis=1)))
    assert total == pytest.approx(70.0, abs=1e-6)


def test_build_vector_overlay_czml_colors_in_range():
    dataset = _synthetic_dataset(u=1.0, v=0.0, w=0.0)
    georeference = GeoReference(longitude=117.0, latitude=36.6, height=0.0)
    result = build_vector_overlay(
        dataset,
        georeference,
        u_name="u",
        v_name="v",
        w_name="w",
        step=1,
        arrow_length=20.0,
        streamline_count=0,
    )
    for packet in result.arrow_packets:
        rgba = packet["polyline"]["material"]["solidColor"]["color"]["rgba"]
        assert len(rgba) == 4
        assert all(0 <= value <= 255 for value in rgba)
    assert result.speed_min == pytest.approx(1.0)
    assert result.speed_max == pytest.approx(1.0)


def test_write_vector_overlay_emits_shards_and_manifest(tmp_path: Path):
    dataset = _synthetic_dataset()
    georeference = GeoReference(longitude=117.0, latitude=36.6, height=0.0)
    result = build_vector_overlay(
        dataset,
        georeference,
        u_name="u",
        v_name="v",
        w_name="w",
        step=2,
        arrow_length=20.0,
        streamline_count=2,
    )
    directory = write_vector_overlay(result, tmp_path / "vectors")
    manifest = json.loads((directory / "vectors-manifest.json").read_text(encoding="utf-8"))
    assert manifest["arrow_count"] == result.arrow_count
    assert manifest["streamline_count"] == result.streamline_count

    total_arrows = 0
    for name in manifest["arrows"]:
        shard = json.loads((directory / name).read_text(encoding="utf-8"))
        # CesiumJS rejects a CZML stream whose first packet is not the document object,
        # and each shard is loaded as an independent stream.
        assert shard[0]["id"] == "document"
        assert shard[0]["version"] == "1.0"
        entities = shard[1:]
        assert entities, f"{name} carries no entity packets"
        total_arrows += len(entities)
        assert all("polyline" in packet for packet in entities)
        for packet in entities:
            assert packet["id"] != "document"
            assert len(packet["polyline"]["positions"]["cartesian"]) % 3 == 0
            assert all(isinstance(c, int) for c in packet["polyline"]["positions"]["cartesian"])
    assert total_arrows == result.arrow_count

    total_lines = 0
    for name in manifest["streamlines"]:
        shard = json.loads((directory / name).read_text(encoding="utf-8"))
        assert shard[0]["id"] == "document"
        assert shard[0]["version"] == "1.0"
        entities = shard[1:]
        assert entities, f"{name} carries no entity packets"
        total_lines += len(entities)
        assert all("polyline" in packet for packet in entities)
    assert total_lines == result.streamline_count

    # Every shard must stay under the 64 KiB transport body cap.
    limit = 64 * 1024
    for name in manifest["arrows"] + manifest["streamlines"]:
        assert (directory / name).stat().st_size < limit


def test_build_velocity_field_grid_and_scaling():
    dataset = _synthetic_dataset(shape_zyx=(21, 21, 21), spacing=(10.0, 10.0, 10.0))
    georeference = GeoReference(longitude=117.0, latitude=36.6, height=0.0)
    field = build_velocity_field(
        dataset, u_name="u", v_name="v", w_name="w", field_step=4
    )
    # ceil(21/4) per axis.
    assert field.dimensions == (6, 6, 6)
    assert np.allclose(field.spacing, np.asarray(dataset.spacing) * 4)
    assert np.allclose(field.origin, dataset.origin)
    assert len(field.u) == 6 * 6 * 6
    # Synthetic field is constant (u=3, v=4, w=0) => speed 5 everywhere.
    assert field.speed_min == pytest.approx(5.0)
    assert field.speed_max == pytest.approx(5.0)


def test_write_velocity_field_shards_under_64k_and_reassembles(tmp_path: Path):
    dataset = _synthetic_dataset(shape_zyx=(21, 21, 21), spacing=(10.0, 10.0, 10.0))
    georeference = GeoReference(longitude=117.0, latitude=36.6, height=0.0)
    field = build_velocity_field(
        dataset, u_name="u", v_name="v", w_name="w", field_step=2
    )
    directory = write_velocity_field(field, georeference, tmp_path / "field")
    manifest = json.loads(
        (directory / "velocity-field-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["nx"] == field.dimensions[0]
    limit = 64 * 1024
    assert manifest["shards"]
    for name in manifest["shards"]:
        assert (directory / name).stat().st_size < limit
    # Reassemble every shard into one field and compare against the original arrays.
    nx, ny, nz = manifest["nx"], manifest["ny"], manifest["nz"]
    u = np.zeros(nx * ny * nz, dtype=np.float64)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    for name in manifest["shards"]:
        s = json.loads((directory / name).read_text(encoding="utf-8"))
        z0, zc, y0, yc = s["zStart"], s["zCount"], s["yStart"], s["yCount"]
        su, sv, sw = s["u"], s["v"], s["w"]
        ptr = 0
        for k in range(zc):
            gk = z0 + k
            for j in range(yc):
                gj = y0 + j
                for i in range(nx):
                    gi = i + gj * nx + gk * nx * ny
                    u[gi], v[gi], w[gi] = su[ptr], sv[ptr], sw[ptr]
                    ptr += 1
    assert np.allclose(u, field.u, atol=1e-2)
    assert np.allclose(v, field.v, atol=1e-2)
    assert np.allclose(w, field.w, atol=1e-2)
