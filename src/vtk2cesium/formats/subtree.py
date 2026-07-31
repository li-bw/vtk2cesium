"""Implicit OCTREE subtree availability documents and bitstreams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from vtk2cesium.tiling import TileCoordinate


@dataclass(frozen=True)
class SubtreePayload:
    """A subtree JSON document and its external availability bitstream."""

    document: dict[str, Any]
    binary: bytes


def root_only_subtree() -> dict[str, Any]:
    """Return a one-level subtree where only the root tile has content."""

    return {
        "tileAvailability": {"constant": 1},
        "contentAvailability": [{"constant": 1}],
        "childSubtreeAvailability": {"constant": 0},
    }


def build_subtree_payload(
    coordinates: Iterable[TileCoordinate],
    *,
    subtree_levels: int,
    buffer_uri: str = "0.0.0.0.bin",
) -> SubtreePayload:
    """Encode tile/content availability in OCTREE Morton order, least-significant bit first."""

    if subtree_levels <= 0:
        raise ValueError("subtree_levels must be positive")
    coordinates = tuple(sorted(set(coordinates)))
    if any(coordinate.level >= subtree_levels for coordinate in coordinates):
        raise ValueError("coordinate lies outside subtree_levels")
    bit_count = sum(8**level for level in range(subtree_levels))
    payload = bytearray((bit_count + 7) // 8)
    for coordinate in coordinates:
        index = level_offset(coordinate.level) + morton_index(coordinate)
        payload[index // 8] |= 1 << (index % 8)
    available_count = len(coordinates)
    document: dict[str, Any] = {
        "buffers": [{"uri": buffer_uri, "byteLength": len(payload)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(payload)}],
        "tileAvailability": {"bitstream": 0, "availableCount": available_count},
        "contentAvailability": [{"bitstream": 0, "availableCount": available_count}],
        "childSubtreeAvailability": {"constant": 0},
    }
    return SubtreePayload(document=document, binary=bytes(payload))


def available_coordinates(payload: bytes, *, subtree_levels: int) -> tuple[TileCoordinate, ...]:
    """Decode an OCTREE availability bitstream for validation and tests."""

    result: list[TileCoordinate] = []
    for level in range(subtree_levels):
        limit = 1 << level
        offset = level_offset(level)
        for z in range(limit):
            for y in range(limit):
                for x in range(limit):
                    coordinate = TileCoordinate(level, x, y, z)
                    index = offset + morton_index(coordinate)
                    if index // 8 < len(payload) and payload[index // 8] & (1 << (index % 8)):
                        result.append(coordinate)
    return tuple(sorted(result))


def level_offset(level: int) -> int:
    """Return the breadth-first OCTREE bit offset for a level."""

    if level < 0:
        raise ValueError("level must be non-negative")
    return (8**level - 1) // 7


def morton_index(coordinate: TileCoordinate) -> int:
    """Return the 3D Morton index, interleaving X, Y, then Z bits."""

    value = 0
    for bit in range(coordinate.level):
        value |= ((coordinate.x >> bit) & 1) << (3 * bit)
        value |= ((coordinate.y >> bit) & 1) << (3 * bit + 1)
        value |= ((coordinate.z >> bit) & 1) << (3 * bit + 2)
    return value
