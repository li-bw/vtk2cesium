import struct

import pytest

from vtk2cesium.formats.glb import build_glb, parse_glb


def test_glb_round_trip_and_four_byte_alignment() -> None:
    document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": 3}]}
    payload = build_glb(document, b"\x01\x02\x03")
    parsed = parse_glb(payload)

    magic, version, length = struct.unpack_from("<4sII", payload)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(payload)
    assert len(payload) % 4 == 0
    assert parsed.document == document
    assert parsed.binary[:3] == b"\x01\x02\x03"
    assert parsed.binary[3:] == b"\x00"


def test_glb_serialization_is_deterministic() -> None:
    left = build_glb({"b": 2, "a": 1}, b"data")
    right = build_glb({"a": 1, "b": 2}, b"data")

    assert left == right


def test_parse_glb_rejects_wrong_declared_length() -> None:
    payload = bytearray(build_glb({"asset": {"version": "2.0"}}))
    struct.pack_into("<I", payload, 8, len(payload) + 4)

    with pytest.raises(ValueError, match="declared length"):
        parse_glb(payload)
