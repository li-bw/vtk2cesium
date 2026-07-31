from vtk2cesium.formats.subtree import (
    available_coordinates,
    build_subtree_payload,
    level_offset,
    morton_index,
)
from vtk2cesium.tiling import TileCoordinate


def test_morton_and_level_offsets_are_stable() -> None:
    assert level_offset(0) == 0
    assert level_offset(1) == 1
    assert level_offset(2) == 9
    assert morton_index(TileCoordinate(1, 1, 0, 0)) == 1
    assert morton_index(TileCoordinate(1, 0, 1, 0)) == 2
    assert morton_index(TileCoordinate(1, 0, 0, 1)) == 4


def test_subtree_bitstream_round_trips_coordinates() -> None:
    coordinates = (
        TileCoordinate(0, 0, 0, 0),
        TileCoordinate(1, 0, 0, 0),
        TileCoordinate(1, 1, 0, 0),
        TileCoordinate(1, 0, 1, 1),
    )
    subtree = build_subtree_payload(coordinates, subtree_levels=2)

    assert subtree.document["tileAvailability"]["availableCount"] == 4
    assert subtree.document["contentAvailability"][0]["availableCount"] == 4
    assert available_coordinates(subtree.binary, subtree_levels=2) == tuple(sorted(coordinates))
