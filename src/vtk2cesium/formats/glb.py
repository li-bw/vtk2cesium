"""Deterministic glTF Binary (GLB 2.0) construction and parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from typing import Any, Mapping

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
JSON_CHUNK_TYPE = b"JSON"
BIN_CHUNK_TYPE = b"BIN\x00"
_HEADER = struct.Struct("<4sII")
_CHUNK_HEADER = struct.Struct("<I4s")


@dataclass(frozen=True)
class ParsedGlb:
    """Decoded GLB JSON document and binary chunk."""

    document: dict[str, Any]
    binary: bytes


def build_glb(document: Mapping[str, Any], binary: bytes = b"") -> bytes:
    """Build a deterministic GLB 2.0 byte sequence.

    JSON is serialized compactly with sorted keys. JSON padding uses spaces and
    BIN padding uses zero bytes as required by the glTF 2.0 container format.
    """

    normalized = dict(document)
    json_payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    json_chunk = _pad(json_payload, b" ")
    binary_chunk = _pad(bytes(binary), b"\x00") if binary else b""

    total_length = _HEADER.size + _CHUNK_HEADER.size + len(json_chunk)
    if binary:
        total_length += _CHUNK_HEADER.size + len(binary_chunk)

    parts = [
        _HEADER.pack(GLB_MAGIC, GLB_VERSION, total_length),
        _CHUNK_HEADER.pack(len(json_chunk), JSON_CHUNK_TYPE),
        json_chunk,
    ]
    if binary:
        parts.extend((_CHUNK_HEADER.pack(len(binary_chunk), BIN_CHUNK_TYPE), binary_chunk))
    return b"".join(parts)


def parse_glb(payload: bytes) -> ParsedGlb:
    """Parse and validate the structural invariants of a GLB 2.0 payload."""

    data = bytes(payload)
    if len(data) < _HEADER.size + _CHUNK_HEADER.size:
        raise ValueError("GLB payload is too short")
    magic, version, declared_length = _HEADER.unpack_from(data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("invalid GLB magic")
    if version != GLB_VERSION:
        raise ValueError(f"unsupported GLB version: {version}")
    if declared_length != len(data):
        raise ValueError("GLB declared length does not match payload length")

    offset = _HEADER.size
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        if offset + _CHUNK_HEADER.size > len(data):
            raise ValueError("truncated GLB chunk header")
        length, chunk_type = _CHUNK_HEADER.unpack_from(data, offset)
        offset += _CHUNK_HEADER.size
        end = offset + length
        if length % 4 != 0 or end > len(data):
            raise ValueError("invalid or truncated GLB chunk length")
        chunks.append((chunk_type, data[offset:end]))
        offset = end

    if not chunks or chunks[0][0] != JSON_CHUNK_TYPE:
        raise ValueError("first GLB chunk must be JSON")
    if len(chunks) > 2 or (len(chunks) == 2 and chunks[1][0] != BIN_CHUNK_TYPE):
        raise ValueError("GLB may contain one JSON chunk followed by one BIN chunk")
    try:
        document = json.loads(chunks[0][1].decode("utf-8").rstrip(" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid GLB JSON chunk") from error
    if not isinstance(document, dict):
        raise ValueError("GLB JSON root must be an object")
    binary = chunks[1][1] if len(chunks) == 2 else b""
    return ParsedGlb(document=document, binary=binary)


def _pad(payload: bytes, padding: bytes) -> bytes:
    count = (-len(payload)) % 4
    return payload + padding * count
