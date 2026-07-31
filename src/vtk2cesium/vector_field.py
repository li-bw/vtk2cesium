"""Vector field overlay (arrow glyphs + streamlines) as a decoupled CZML layer.

This module reads the *original* VTI's 3D velocity components (``u, v, w``) in
the local ENU frame, downsamples grid points, and emits a CZML document whose
positions are ECEF cartesians produced by the same :class:`GeoReference` transform
used for the voxel tileset. It does **not** touch the voxel GLB / tileset writer,
so the single-scalar production pipeline is unaffected.
"""

from __future__ import annotations

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


def write_vector_overlay(result: VectorOverlayResult, output_directory: str | Path) -> Path:
    """Write ``arrows.czml`` and ``streamlines.czml`` into ``output_directory``."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    import json

    (directory / "arrows.czml").write_text(
        json.dumps(list(result.arrow_packets), ensure_ascii=False), encoding="utf-8"
    )
    (directory / "streamlines.czml").write_text(
        json.dumps(list(result.streamline_packets), ensure_ascii=False), encoding="utf-8"
    )
    return directory
