from vtk2cesium.formats.subtree import root_only_subtree
from vtk2cesium.formats.tileset import build_probe_tileset, build_voxel_tileset


def test_tileset_has_required_voxel_contract() -> None:
    tileset = build_probe_tileset(
        (4, 5, 6), property_name="density", minimum=0.0, maximum=1.0
    )
    root = tileset["root"]
    voxel = root["content"]["extensions"]["3DTILES_content_voxels"]

    assert tileset["asset"]["version"] == "1.1"
    assert voxel == {"dimensions": [4, 5, 6], "class": "voxel"}
    assert root["implicitTiling"]["subdivisionScheme"] == "OCTREE"
    assert root["implicitTiling"]["availableLevels"] == 1


def test_formal_tileset_includes_transform_local_box_and_glb_template() -> None:
    transform = tuple(float(index) for index in range(16))
    box = (1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0)
    tileset = build_voxel_tileset(
        (7, 8, 9),
        property_name="temperature",
        minimum=-2.0,
        maximum=4.0,
        bounding_box=box,
        transform=transform,
    )

    root = tileset["root"]
    assert root["transform"] == list(transform)
    assert root["boundingVolume"]["box"] == list(box)
    assert root["content"]["uri"] == "content/{level}.{x}.{y}.{z}.glb"
    assert root["implicitTiling"]["subtrees"]["uri"] == (
        "subtrees/{level}.{x}.{y}.{z}.subtree"
    )
    assert tileset["statistics"]["classes"]["voxel"]["count"] == 7 * 8 * 9


def test_root_only_subtree_uses_constant_availability() -> None:
    assert root_only_subtree() == {
        "tileAvailability": {"constant": 1},
        "contentAvailability": [{"constant": 1}],
        "childSubtreeAvailability": {"constant": 0},
    }
