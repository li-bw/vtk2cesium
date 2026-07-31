"""Generate large synthetic VTI datasets for end-to-end pipeline testing.

Two scenarios are provided, both as structured `vtkImageData` (.vti) with
point-data scalar fields that the project reader understands:

1. ``wind``  -- a wind field over Jinan Lixia District (~15 km x 15 km, 0..2 km
   above ground). Fields: ``u`` (eastward), ``v`` (northward), ``w`` (vertical),
   ``speed`` (magnitude), ``temperature`` (deg C). Use ``--field speed`` (or
   ``u`` / ``v`` / ``w`` / ``temperature``) when converting.

2. ``geo``   -- a subsurface geological body over a larger ~100 km x 100 km
   region, 0..20 km deep. Fields: ``density`` (g/cm3) with folded strata, a
   normal fault, and an igneous intrusion, plus ``porosity`` (percent).

The VTI stores only a *local* structured grid (origin + spacing). The geographic
anchor is supplied later at convert time via ``--lon --lat --height``; each
scenario prints a suggested anchor for convenience.

Binary VTK output is used so multi-million-point volumes stay compact and are
read back by the same ``vtkXMLImageDataReader`` the pipeline relies on.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
except Exception as exc:  # pragma: no cover - environment guard
    print(f"ERROR: this generator needs VTK (vtk.util.numpy_support): {exc}", file=sys.stderr)
    raise

SAMPLES_DIR = Path(__file__).resolve().parent


def _array_xyz_to_vtk(arr_xyz: np.ndarray) -> np.ndarray:
    """Convert an (nx, ny, nz) array to the flat VTK x-fastest buffer.

    A C-order ``(z, y, x)`` array flat-index matches VTK's x-fastest ordering,
    so transposing to ``(z, y, x)`` then raveling yields the correct buffer.
    """

    return np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)).ravel()


def _write_image_data(
    nx: int,
    ny: int,
    nz: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    arrays: list[tuple[str, np.ndarray]],
    output: Path,
) -> None:
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetOrigin(*origin)
    image.SetSpacing(*spacing)
    for name, arr_xyz in arrays:
        flat = _array_xyz_to_vtk(arr_xyz).astype(np.float32, copy=False)
        vtk_array = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(name)
        image.GetPointData().AddArray(vtk_array)
    image.GetPointData().SetActiveScalars(arrays[0][0])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(output))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    if not writer.Write():
        raise RuntimeError(f"VTK failed to write {output}")


def _summarize(output: Path, nx: int, ny: int, nz: int, arrays: list[tuple[str, np.ndarray]]) -> None:
    points = nx * ny * nz
    size_mb = output.stat().st_size / (1024.0 * 1024.0)
    print(f"Wrote {output}")
    print(f"  point dimensions : {nx} x {ny} x {nz}  ({points:,} points)")
    print(f"  file size        : {size_mb:.1f} MB")
    for name, arr_xyz in arrays:
        finite = arr_xyz.ravel()
        finite = finite[np.isfinite(finite)]
        print(f"  field {name:12s}: {finite.min():.4g} .. {finite.max():.4g}")


def generate_wind(
    output: Path,
    nx: int = 200,
    ny: int = 200,
    nz: int = 40,
    domain_m: float = 15000.0,
    height_m: float = 2000.0,
) -> None:
    """Atmospheric wind field over a ~15 km square centered at the local origin."""

    dx = domain_m / (nx - 1)
    dy = domain_m / (ny - 1)
    dz = height_m / (nz - 1)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rng = np.random.default_rng(20260731)

    u = (
        6.0
        + 0.0015 * Z
        + 2.0 * np.sin(2 * np.pi * X / 4000.0) * np.cos(2 * np.pi * Y / 5000.0)
        + 0.8 * np.sin(2 * np.pi * Z / 800.0)
        + 0.3 * rng.standard_normal((nx, ny, nz))
    )
    v = (
        -2.0
        + 1.5 * np.sin(2 * np.pi * Y / 6000.0)
        + 0.6 * np.cos(2 * np.pi * X / 4500.0)
        + 0.2 * rng.standard_normal((nx, ny, nz))
    )
    w = (
        0.5 * np.sin(2 * np.pi * X / 2500.0) * np.sin(2 * np.pi * Y / 2500.0)
        * np.sin(np.pi * Z / max(height_m, 1.0))
        + 0.1 * rng.standard_normal((nx, ny, nz))
    )
    speed = np.sqrt(u * u + v * v + w * w)
    temperature = (
        18.0
        - 0.0065 * Z
        + 1.5 * np.sin(2 * np.pi * X / 7000.0)
        + 1.0 * np.cos(2 * np.pi * Y / 6500.0)
        + 0.4 * rng.standard_normal((nx, ny, nz))
    )

    arrays = [
        ("speed", speed),
        ("u", u),
        ("v", v),
        ("w", w),
        ("temperature", temperature),
    ]
    _write_image_data(nx, ny, nz, (0.0, 0.0, 0.0), (dx, dy, dz), arrays, output)
    _summarize(output, nx, ny, nz, arrays)
    print("  suggested anchor : --lon 117.07 --lat 36.66 --height 0  (Jinan Lixia District)")
    print("  try               : --field speed  (or u / v / w / temperature)")


def generate_geology(
    output: Path,
    nx: int = 256,
    ny: int = 256,
    nz: int = 64,
    domain_m: float = 100000.0,
    depth_m: float = 20000.0,
) -> None:
    """Subsurface geological body over a ~100 km square, 0..20 km deep."""

    dx = domain_m / (nx - 1)
    dy = domain_m / (ny - 1)
    dz = depth_m / (nz - 1)
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = -depth_m + np.arange(nz) * dz  # local z: -depth_m (deep) .. ~0 (surface)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rng = np.random.default_rng(20260731)

    depth = -Z  # positive downward (0 at surface .. depth_m at bottom)
    # Normal fault: layers drop east of the 50 km line.
    fault_offset = 1200.0 * np.tanh((X - 50000.0) / 3000.0)
    d_eff = depth - fault_offset

    strata = 0.12 * np.sin(
        2 * np.pi * d_eff / 1500.0
        + 0.25 * np.sin(2 * np.pi * X / 25000.0)
        + 0.20 * np.cos(2 * np.pi * Y / 30000.0)
    )
    base = 2.0 + 0.045 * (d_eff / 1000.0)
    density = base + strata + 0.02 * rng.standard_normal((nx, ny, nz))

    # Igneous intrusion (high density, low porosity).
    Xc, Yc, depth_c, radius = 35000.0, 65000.0, 9000.0, 7000.0
    r = np.sqrt((X - Xc) ** 2 + (Y - Yc) ** 2 + (depth - depth_c) ** 2)
    intrusion = 0.45 * np.exp(-((r / radius) ** 2))
    density = density + intrusion
    density = np.clip(density, 1.6, 3.3)

    porosity = (
        38.0 * np.exp(-depth / 3500.0)
        + 3.0
        - 15.0 * np.exp(-((r / radius) ** 2))
        + 0.5 * rng.standard_normal((nx, ny, nz))
    )
    porosity = np.clip(porosity, 0.5, 42.0)

    arrays = [
        ("density", density),
        ("porosity", porosity),
    ]
    _write_image_data(nx, ny, nz, (0.0, 0.0, -depth_m), (dx, dy, dz), arrays, output)
    _summarize(output, nx, ny, nz, arrays)
    print("  suggested anchor : --lon 117.00 --lat 36.60 --height 0  (central Shandong)")
    print("  try               : --field density  (or porosity)")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("resolution must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="scenario", required=True)

    wind = sub.add_parser("wind", help="wind field over Jinan Lixia District")
    wind.add_argument("--output", default=str(SAMPLES_DIR / "wind_lixia.vti"))
    wind.add_argument("--nx", type=_positive_int, default=200)
    wind.add_argument("--ny", type=_positive_int, default=200)
    wind.add_argument("--nz", type=_positive_int, default=40)
    wind.add_argument("--domain", type=float, default=15000.0, help="horizontal extent (m)")
    wind.add_argument("--height", type=float, default=2000.0, help="vertical extent (m)")

    geo = sub.add_parser("geo", help="subsurface geological body over ~100 km")
    geo.add_argument("--output", default=str(SAMPLES_DIR / "geology_shandong.vti"))
    geo.add_argument("--nx", type=_positive_int, default=256)
    geo.add_argument("--ny", type=_positive_int, default=256)
    geo.add_argument("--nz", type=_positive_int, default=64)
    geo.add_argument("--domain", type=float, default=100000.0, help="horizontal extent (m)")
    geo.add_argument("--depth", type=float, default=20000.0, help="vertical depth (m)")

    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()

    if args.scenario == "wind":
        generate_wind(output, args.nx, args.ny, args.nz, args.domain, args.height)
    else:
        generate_geology(output, args.nx, args.ny, args.nz, args.domain, args.depth)


if __name__ == "__main__":
    main()
