from __future__ import annotations

import math
from pathlib import Path


NX, NY, NZ = 16, 12, 8
OUTPUT = Path(__file__).with_name("generated-volume.vti")


def density(x: int, y: int, z: int) -> float:
    """Return a deterministic scalar field with a central blob and waves."""
    cx = (NX - 1) / 2.0
    cy = (NY - 1) / 2.0
    cz = (NZ - 1) / 2.0
    dx = (x - cx) / max(cx, 1.0)
    dy = (y - cy) / max(cy, 1.0)
    dz = (z - cz) / max(cz, 1.0)
    radius_squared = dx * dx + dy * dy + dz * dz
    blob = math.exp(-2.8 * radius_squared)
    wave = 0.12 * math.sin(x * 0.7) * math.cos(y * 0.5)
    return blob + wave


def main() -> None:
    # VTK ImageData stores x as the fastest-varying axis.
    values = [
        density(x, y, z)
        for z in range(NZ)
        for y in range(NY)
        for x in range(NX)
    ]
    payload = " ".join(f"{value:.8g}" for value in values)

    xml = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="0 {NX - 1} 0 {NY - 1} 0 {NZ - 1}" Origin="0 0 0" Spacing="1 1 1">
    <Piece Extent="0 {NX - 1} 0 {NY - 1} 0 {NZ - 1}">
      <PointData Scalars="density">
        <DataArray type="Float32" Name="density" NumberOfComponents="1" format="ascii">
          {payload}
        </DataArray>
      </PointData>
      <CellData/>
    </Piece>
  </ImageData>
</VTKFile>
'''
    OUTPUT.write_text(xml, encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT}")
    print(f"Point dimensions: {NX} x {NY} x {NZ}")
    print(f"Scalar field: density ({len(values)} Float32-compatible values)")
    print(f"Finite range: {min(values):.8g} .. {max(values):.8g}")


if __name__ == "__main__":
    main()
