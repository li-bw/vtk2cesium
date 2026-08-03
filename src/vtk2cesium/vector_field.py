"""Vector field overlay (arrow glyphs + streamlines) as a decoupled CZML layer.

This module reads the *original* VTI's 3D velocity components (``u, v, w``) in
the local ENU frame, downsamples grid points, and emits a CZML document whose
positions are ECEF cartesians produced by the same :class:`GeoReference` transform
used for the voxel tileset. It does **not** touch the voxel GLB / tileset writer,
so the single-scalar production pipeline is unaffected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from vtk2cesium.geo import GeoReference
from vtk2cesium.model import ScalarField, StructuredVoxelDataset

Float64Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VectorOverlayResult:
    """Pure computation result for a vector overlay layer."""

    arrow_packets: tuple[dict, ...]
    streamline_packets: tuple[dict, ...]
    arrow_count: int
    streamline_count: int
    speed_min: float
    speed_max: float


def _as_step(value: int | tuple[int, int, int] | None, default: int) -> tuple[int, int, int]:
    if value is None:
        return (default, default, default)
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 3:
        return tuple(int(part) for part in value)
    raise ValueError("step must be an int or a (x, y, z) tuple")


def _down_sampled_indices(dimensions_xyz: tuple[int, int, int], step: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    nx, ny, nz = dimensions_xyz
    sx, sy, sz = (max(int(s), 1) for s in step)
    indices: list[tuple[int, int, int]] = []
    for k in range(0, nz, sz):
        for j in range(0, ny, sy):
            for i in range(0, nx, sx):
                indices.append((i, j, k))
    return indices


def _sample_component(values: Float64Array, fi: float, fj: float, fk: float) -> float:
    nz, ny, nx = values.shape
    i0 = int(np.floor(fi)); j0 = int(np.floor(fj)); k0 = int(np.floor(fk))
    i1 = min(i0 + 1, nx - 1); j1 = min(j0 + 1, ny - 1); k1 = min(k0 + 1, nz - 1)
    di = fi - i0; dj = fj - j0; dk = fk - k0
    c00 = values[k0, j0, i0] * (1.0 - di) + values[k0, j0, i1] * di
    c10 = values[k0, j1, i0] * (1.0 - di) + values[k0, j1, i1] * di
    c01 = values[k1, j0, i0] * (1.0 - di) + values[k1, j0, i1] * di
    c11 = values[k1, j1, i0] * (1.0 - di) + values[k1, j1, i1] * di
    c0 = c00 * (1.0 - dj) + c10 * dj
    c1 = c01 * (1.0 - dj) + c11 * dj
    return float(c0 * (1.0 - dk) + c1 * dk)


def _vector_at_enu(
    dataset: StructuredVoxelDataset,
    u_name: str,
    v_name: str,
    w_name: str,
    position_enu: Sequence[float],
) -> Float64Array | None:
    origin = np.asarray(dataset.origin, dtype=np.float64)
    spacing = np.asarray(dataset.spacing, dtype=np.float64)
    index = (np.asarray(position_enu, dtype=np.float64) - origin) / spacing
    nx, ny, nz = dataset.point_dimensions
    if np.any(index < 0.0) or index[0] > nx - 1 or index[1] > ny - 1 or index[2] > nz - 1:
        return None
    u = _sample_component(dataset.field(u_name).values, index[0], index[1], index[2])
    v = _sample_component(dataset.field(v_name).values, index[0], index[1], index[2])
    w = _sample_component(dataset.field(w_name).values, index[0], index[1], index[2])
    return np.array([u, v, w], dtype=np.float64)


def _speed_to_rgba(speed: float, speed_min: float, speed_max: float) -> list[int]:
    if speed_max > speed_min:
        t = (speed - speed_min) / (speed_max - speed_min)
    else:
        t = 0.0
    t = min(max(t, 0.0), 1.0)
    red = int(round(255.0 * t))
    blue = int(round(255.0 * (1.0 - t)))
    green = int(round(255.0 * (1.0 - abs(2.0 * t - 1.0))))
    return [red, green, blue, 210]


def _enu_to_cartesian(points_enu: Iterable[Sequence[float]], georeference: GeoReference) -> list[float]:
    ecef = georeference.transform_points(points_enu)
    return [float(coord) for coord in np.asarray(ecef, dtype=np.float64).ravel()]


def _arrow_enu_points(base: Float64Array, vector: Float64Array, length: float) -> list[Float64Array]:
    speed = float(np.linalg.norm(vector))
    if speed < 1e-6:
        return []
    direction = vector / speed
    tip = base + direction * length
    ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(direction, ref)
    right = right / np.linalg.norm(right)
    back = -direction * (0.28 * length)
    side = right * (0.14 * length)
    barb1 = tip + back + side
    barb2 = tip + back - side
    # Single polyline: shaft -> tip -> barb1 -> tip -> barb2 (one retraced segment).
    return [base, tip, barb1, tip, barb2]


def _integrate_streamline(
    dataset: StructuredVoxelDataset,
    u_name: str,
    v_name: str,
    w_name: str,
    start_enu: Float64Array,
    sign: int,
    step_meters: float,
    steps: int,
) -> list[Float64Array]:
    points: list[Float64Array] = []
    position = np.asarray(start_enu, dtype=np.float64).copy()
    h = sign * float(step_meters)
    for _ in range(steps):
        vector = _vector_at_enu(dataset, u_name, v_name, w_name, position)
        if vector is None:
            break
        speed = float(np.linalg.norm(vector))
        if speed < 1e-6:
            break
        direction = vector / speed

        def direction_at(point: Float64Array) -> Float64Array:
            sampled = _vector_at_enu(dataset, u_name, v_name, w_name, point)
            if sampled is None:
                return np.zeros(3, dtype=np.float64)
            norm = float(np.linalg.norm(sampled))
            if norm < 1e-6:
                return np.zeros(3, dtype=np.float64)
            return sampled / norm

        k1 = direction
        k2 = direction_at(position + 0.5 * h * k1)
        k3 = direction_at(position + 0.5 * h * k2)
        k4 = direction_at(position + h * k3)
        if not np.all(np.isfinite(k2)) or not np.all(np.isfinite(k3)) or not np.all(np.isfinite(k4)):
            break
        position = position + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if _vector_at_enu(dataset, u_name, v_name, w_name, position) is None:
            break
        points.append(position.copy())
    return points


def build_vector_overlay(
    dataset: StructuredVoxelDataset,
    georeference: GeoReference,
    *,
    u_name: str,
    v_name: str,
    w_name: str,
    step: int | tuple[int, int, int] | None = None,
    arrow_length: float = 400.0,
    streamline_count: int = 150,
    streamline_steps: int = 50,
    streamline_step_meters: float = 250.0,
    seed: int = 0,
) -> VectorOverlayResult:
    """Build CZML packets for arrow glyphs and streamlines from a velocity field.

    Positions are emitted as ECEF cartesians via ``georeference.transform_points``,
    identical to the voxel tileset transform, so the overlay aligns with the voxels.
    """

    step_xyz = _as_step(step, 16)
    indices = _down_sampled_indices(dataset.point_dimensions, step_xyz)
    origin = np.asarray(dataset.origin, dtype=np.float64)
    spacing = np.asarray(dataset.spacing, dtype=np.float64)

    arrow_packets: list[dict] = []
    sampled_speeds: list[float] = []
    for counter, (i, j, k) in enumerate(indices):
        base = origin + np.array([i * spacing[0], j * spacing[1], k * spacing[2]], dtype=np.float64)
        vector = np.array(
            [
                float(dataset.field(u_name).values[k, j, i]),
                float(dataset.field(v_name).values[k, j, i]),
                float(dataset.field(w_name).values[k, j, i]),
            ],
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(vector))
        sampled_speeds.append(speed)
        enu_points = _arrow_enu_points(base, vector, arrow_length)
        if not enu_points:
            continue
        cartesian = _enu_to_cartesian(enu_points, georeference)
        arrow_packets.append(
            {
                "id": f"arrow-{counter}",
                "_speed": speed,
                "polyline": {
                    "positions": {"cartesian": cartesian, "arcType": "NONE"},
                    "width": 2,
                    "material": {
                        "solidColor": {
                            "color": {"rgba": _speed_to_rgba(speed, 0.0, 1.0)}
                        }
                    },
                },
            }
        )

    rng = np.random.default_rng(seed)
    nx, ny, nz = dataset.point_dimensions
    domain_max = origin + np.array(
        [(nx - 1) * spacing[0], (ny - 1) * spacing[1], (nz - 1) * spacing[2]], dtype=np.float64
    )
    streamline_packets: list[dict] = []
    for counter in range(streamline_count):
        seed_position = origin + rng.random(3) * (domain_max - origin)
        forward = _integrate_streamline(
            dataset, u_name, v_name, w_name, seed_position, 1, streamline_step_meters, streamline_steps
        )
        backward = _integrate_streamline(
            dataset, u_name, v_name, w_name, seed_position, -1, streamline_step_meters, streamline_steps
        )
        path: list[Float64Array] = list(reversed(backward)) + [seed_position] + forward
        if len(path) < 2:
            continue
        sampled = _vector_at_enu(dataset, u_name, v_name, w_name, seed_position)
        seed_speed = float(np.linalg.norm(sampled)) if sampled is not None else 0.0
        sampled_speeds.append(seed_speed)
        cartesian = _enu_to_cartesian(path, georeference)
        streamline_packets.append(
            {
                "id": f"streamline-{counter}",
                "_speed": seed_speed,
                "polyline": {
                    "positions": {"cartesian": cartesian, "arcType": "NONE"},
                    "width": 1.5,
                    "material": {
                        "solidColor": {
                            "color": {"rgba": _speed_to_rgba(seed_speed, 0.0, 1.0)}
                        }
                    },
                },
            }
        )

    if sampled_speeds:
        speed_min = float(min(sampled_speeds))
        speed_max = float(max(sampled_speeds))
    else:
        speed_min = 0.0
        speed_max = 1.0
    # Recolour every packet with the global speed range, then drop the temp key.
    for packet in arrow_packets:
        color = packet["polyline"]["material"]["solidColor"]["color"]["rgba"]
        color[:] = _speed_to_rgba(packet.pop("_speed"), speed_min, speed_max)
    for packet in streamline_packets:
        color = packet["polyline"]["material"]["solidColor"]["color"]["rgba"]
        color[:] = _speed_to_rgba(packet.pop("_speed"), speed_min, speed_max)

    return VectorOverlayResult(
        arrow_packets=tuple(arrow_packets),
        streamline_packets=tuple(streamline_packets),
        arrow_count=len(arrow_packets),
        streamline_count=len(streamline_packets),
        speed_min=speed_min,
        speed_max=speed_max,
    )


def _round_packet(packet: dict) -> dict:
    """Return a shallow copy with ECEF cartesian coordinates rounded to whole metres.

    The full float64 repr is ~20 chars/number; rounding to integer metres shrinks the
    serialized CZML by ~2x with sub-metre accuracy that is irrelevant at scene scale.
    """

    packet = dict(packet)
    positions = dict(packet["polyline"]["positions"])
    positions["cartesian"] = [int(round(float(coord))) for coord in positions["cartesian"]]
    polyline = dict(packet["polyline"])
    polyline["positions"] = positions
    packet["polyline"] = polyline
    return packet


def _document_packet(name: str) -> dict:
    """Return the CZML ``document`` packet every stream must start with.

    CesiumJS rejects a CZML stream whose first packet is not the document object
    ("first CZML packet is required to be the document object"), so each shard needs
    its own copy -- a shard is loaded as an independent CZML stream.
    """

    return {"id": "document", "name": name, "version": "1.0"}


def _shard_packets(packets: list[dict], name: str, max_bytes: int = 48000) -> list[list[dict]]:
    """Split packets into shards whose JSON serialization stays under ``max_bytes``.

    Every arrow/streamline is preserved (no data loss); only the on-disk file size is
    bounded so the overlay still loads through transports that cap response bodies
    (e.g. 64 KiB sandboxes / preview proxies). Each shard is prefixed with its own
    ``document`` packet so it is a valid standalone CZML stream.
    """

    document = _document_packet(name)
    document_size = len(json.dumps(document, ensure_ascii=False))
    shards: list[list[dict]] = []
    current: list[dict] = [document]
    current_size = 2 + document_size  # surrounding "[]" plus the document packet
    for packet in packets:
        size = len(json.dumps(packet, ensure_ascii=False))
        if len(current) > 1 and current_size + size > max_bytes:
            shards.append(current)
            current = [dict(document)]
            current_size = 2 + document_size
        current.append(packet)
        current_size += size
    if len(current) > 1:
        shards.append(current)
    return shards


def write_vector_overlay(
    result: VectorOverlayResult,
    output_directory: str | Path,
    max_shard_bytes: int = 48000,
) -> Path:
    """Write the overlay as shard files + a manifest into ``output_directory``.

    Instead of two potentially huge ``arrows.czml`` / ``streamlines.czml`` files, the
    packets are split into ``arrows-<i>.czml`` / ``streamlines-<i>.czml`` shards (each
    well under common 64 KiB transport caps) plus a ``vectors-manifest.json`` listing
    them. Every shard starts with its own ``document`` packet, so each one is a valid
    standalone CZML stream. The viewer loads every shard through the manifest, so
    nothing is lost and the overlay renders identically regardless of body limits.
    """

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrow_packets = [_round_packet(p) for p in result.arrow_packets]
    stream_packets = [_round_packet(p) for p in result.streamline_packets]

    arrow_shards = _shard_packets(arrow_packets, "vtk2cesium arrows", max_shard_bytes)
    stream_shards = _shard_packets(stream_packets, "vtk2cesium streamlines", max_shard_bytes)

    manifest = {
        "arrow_count": result.arrow_count,
        "streamline_count": result.streamline_count,
        "speed_min": result.speed_min,
        "speed_max": result.speed_max,
        "arrows": [],
        "streamlines": [],
    }
    for index, shard in enumerate(arrow_shards):
        name = f"arrows-{index}.czml"
        (directory / name).write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
        manifest["arrows"].append(name)
    for index, shard in enumerate(stream_shards):
        name = f"streamlines-{index}.czml"
        (directory / name).write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
        manifest["streamlines"].append(name)
    (directory / "vectors-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


# --------------------------------------------------------------------------- #
# Velocity field export (Tier-3 particle wind) — decoupled sampler data.
# --------------------------------------------------------------------------- #
#
# The particle layer (``WindParticles.js``) needs the raw u/v/w field on a
# regular grid so it can advect particles in the browser. We emit the *same*
# ENU coordinate space the arrows use (``origin`` + grid index * ``spacing``),
# so a particle at ENU ``p`` samples the field and is transformed to ECEF with
# the same ``GeoReference``. Nothing here touches ``convert`` / the voxel GLB.
#
# Files are sharded (z/y blocks) so every shard stays under common 64 KiB
# transport caps, exactly like the CZML overlay. The viewer assembles the
# shards back into one field via ``velocity-field-manifest.json``.


@dataclass(frozen=True)
class VelocityFieldResult:
    """Regular-grid velocity field for browser-side particle advection."""

    dimensions: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    u: Float64Array
    v: Float64Array
    w: Float64Array
    speed_min: float
    speed_max: float


def build_velocity_field(
    dataset: StructuredVoxelDataset,
    *,
    u_name: str,
    v_name: str,
    w_name: str,
    field_step: int = 8,
) -> VelocityFieldResult:
    """Downsample the u/v/w components onto a coarse regular grid.

    ``field_step`` is the stride through the original VTI grid; the emitted grid
    has ``ceil(n / field_step)`` cells per axis, with ``spacing`` scaled by the
    same stride. Index ``(i, j, k)`` maps to ENU ``origin + (i*dx, j*dy, k*dz)``
    — identical to the arrow base points — and the flat array order is
    ``index = i + j*nx + k*nx*ny`` (i fastest), matching C-order flattening.
    """

    nx, ny, nz = dataset.point_dimensions
    step = max(1, int(field_step))
    fx = (nx + step - 1) // step
    fy = (ny + step - 1) // step
    fz = (nz + step - 1) // step
    origin = tuple(float(coord) for coord in dataset.origin)
    spacing = (
        float(dataset.spacing[0]) * step,
        float(dataset.spacing[1]) * step,
        float(dataset.spacing[2]) * step,
    )

    u_field = dataset.field(u_name).values
    v_field = dataset.field(v_name).values
    w_field = dataset.field(w_name).values

    u = np.empty(fx * fy * fz, dtype=np.float64)
    v = np.empty(fx * fy * fz, dtype=np.float64)
    w = np.empty(fx * fy * fz, dtype=np.float64)
    speed_min = math.inf
    speed_max = -math.inf
    for ki in range(fz):
        k = min(ki * step, nz - 1)
        for jj in range(fy):
            j = min(jj * step, ny - 1)
            for ii in range(fx):
                i = min(ii * step, nx - 1)
                idx = ii + jj * fx + ki * fx * fy
                uu = float(u_field[k, j, i])
                vv = float(v_field[k, j, i])
                ww = float(w_field[k, j, i])
                u[idx] = uu
                v[idx] = vv
                w[idx] = ww
                sp = math.sqrt(uu * uu + vv * vv + ww * ww)
                if sp < speed_min:
                    speed_min = sp
                if sp > speed_max:
                    speed_max = sp
    if not math.isfinite(speed_min):
        speed_min, speed_max = 0.0, 1.0
    return VelocityFieldResult(
        dimensions=(fx, fy, fz),
        origin=origin,
        spacing=spacing,
        u=u,
        v=v,
        w=w,
        speed_min=float(speed_min),
        speed_max=float(speed_max),
    )


def _velocity_field_blocks(nx: int, ny: int, nz: int, max_bytes: int) -> list[tuple[int, int, int, int]]:
    """Return (zStart, zCount, yStart, yCount) blocks covering the grid.

    Each block is kept under ``max_bytes`` by first taking whole-y z-bands and,
    only when a single z-slab is itself too large, splitting that slab along y.
    """

    bytes_per_cell = 3 * 9  # u/v/w floats, ~9 chars each, rough upper bound
    budget = max_bytes * 0.7
    blocks: list[tuple[int, int, int, int]] = []
    z = 0
    while z < nz:
        # Largest z-band whose full-y block still fits the budget.
        max_z_full_y = budget / (ny * nx * bytes_per_cell)
        z_count = max(1, int(max_z_full_y))
        z_count = min(z_count, nz - z)
        if nx * ny * z_count * bytes_per_cell <= budget:
            blocks.append((z, z_count, 0, ny))
            z += z_count
            continue
        # This z-slab alone is too big: split along y.
        y = 0
        while y < ny:
            y_count = max(1, int(budget / (nx * z_count * bytes_per_cell)))
            y_count = min(y_count, ny - y)
            blocks.append((z, z_count, y, y_count))
            y += y_count
        z += z_count
    return blocks


def write_velocity_field(
    result: VelocityFieldResult,
    georeference: GeoReference,
    output_directory: str | Path,
    max_shard_bytes: int = 48000,
) -> Path:
    """Write the velocity field as z/y-blocked shards + a manifest.

    The viewer (``WindParticles.js``) loads every shard through
    ``velocity-field-manifest.json`` and reassembles one ``(nx*ny*nz)`` field,
    so the particle advection works regardless of the 64 KiB transport cap.
    """

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = result.dimensions
    georef = {
        "longitude": georeference.longitude,
        "latitude": georeference.latitude,
        "height": georeference.height,
    }
    blocks = _velocity_field_blocks(nx, ny, nz, max_shard_bytes)

    shard_names: list[str] = []
    for index, (z_start, z_count, y_start, y_count) in enumerate(blocks):
        u_part = _extract_block(result.u, nx, ny, nz, z_start, z_count, y_start, y_count)
        v_part = _extract_block(result.v, nx, ny, nz, z_start, z_count, y_start, y_count)
        w_part = _extract_block(result.w, nx, ny, nz, z_start, z_count, y_start, y_count)
        shard = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "zStart": z_start,
            "zCount": z_count,
            "yStart": y_start,
            "yCount": y_count,
            "origin": list(result.origin),
            "spacing": list(result.spacing),
            "georeference": georef,
            "speed_min": result.speed_min,
            "speed_max": result.speed_max,
            "u": [round(float(x), 3) for x in u_part],
            "v": [round(float(x), 3) for x in v_part],
            "w": [round(float(x), 3) for x in w_part],
        }
        name = f"velocity-field-{index}.json"
        (directory / name).write_text(json.dumps(shard, ensure_ascii=False), encoding="utf-8")
        shard_names.append(name)

    manifest = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "origin": list(result.origin),
        "spacing": list(result.spacing),
        "georeference": georef,
        "speed_min": result.speed_min,
        "speed_max": result.speed_max,
        "shards": shard_names,
    }
    (directory / "velocity-field-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def _extract_block(
    array: Float64Array,
    nx: int,
    ny: int,
    nz: int,
    z_start: int,
    z_count: int,
    y_start: int,
    y_count: int,
) -> list[float]:
    """Pull a contiguous (y, z) sub-block (full x) out of the flat field."""

    out: list[float] = []
    for k in range(z_start, z_start + z_count):
        for j in range(y_start, y_start + y_count):
            start = j * nx + k * nx * ny
            out.extend(float(x) for x in array[start : start + nx])
    return out
