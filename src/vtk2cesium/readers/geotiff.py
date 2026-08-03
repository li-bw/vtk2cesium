"""Read GeoTIFF rasters into validated structured voxel datasets.

A GeoTIFF is a 2D (rows x cols) grid with one or more bands. Each band becomes
a scalar field; the geotransform supplies the local ENU ``origin``/``spacing``
in metres (projected CRS uses metres directly; geographic CRS degrees are
converted with a reference latitude). Rows are flipped so ENU ``y`` increases
north. The result is a single vertical layer (``nz = 1``).

The optional ``rasterio`` package is required and imported lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vtk2cesium.model import (
    ScalarAssociation,
    ScalarField,
    ScalarFieldInfo,
    StructuredVoxelDataset,
    VtiInspection,
)
from vtk2cesium.readers._common import (
    is_degree_units,
    make_bounds,
    meters_per_degree,
)


class GeotiffReadError(ValueError):
    """Raised when a GeoTIFF file cannot be interpreted safely."""


def _import_rasterio():
    try:
        import rasterio  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise GeotiffReadError(
            "reading GeoTIFF requires the optional 'rasterio' package; "
            "install it with: pip install rasterio"
        ) from exc
    return rasterio


def _geotransform_to_local(transform, crs, *, reference_latitude: float | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Map a GDAL geotransform to local ENU origin/spacing in metres.

    The returned origin is the south-west-bottom corner (local 0,0,0); the
    pipeline georeference must point at that corner in the source CRS. Angular
    (geographic-degree) CRS resolutions are converted to metres using the
    reference latitude; projected (metre) CRS resolutions are used directly.
    """

    # rasterio/affine transform is (a, b, c, d, e, f):
    #   a = pixel width  (x resolution, easting per column)
    #   e = pixel height (y resolution, northing per row; negative when north-up)
    x_res = abs(float(transform.a))
    y_res = abs(float(transform.e))
    angular = bool(crs) and bool(getattr(crs, "is_geographic", False))
    ref_lat = reference_latitude if reference_latitude is not None else 0.0
    if angular:
        sx = x_res * meters_per_degree("x", ref_lat)
        sy = y_res * meters_per_degree("y", ref_lat)
    else:
        sx = x_res
        sy = y_res
    origin = (0.0, 0.0, 0.0)
    spacing = (float(sx), float(sy), 1.0)
    return origin, spacing


def _band_name(dataset, band_index: int) -> str:
    """Band name: its description when present, else ``band_<n>``."""

    try:
        descriptions = dataset.descriptions
        description = descriptions[band_index - 1] if descriptions else None
    except (TypeError, IndexError, AttributeError):
        description = None
    if description:
        return description
    return f"band_{band_index}"


def _make_info(name: str, values: np.ndarray) -> ScalarFieldInfo:
    components = 1 if values.ndim == 3 else int(values.shape[-1])
    flat = values.reshape(-1, components) if components > 1 else values.reshape(-1, 1)
    finite = flat[np.isfinite(flat)]
    # A 2D boolean mask flattens the result to 1-D; reshape back to (k, components).
    finite = finite.reshape(-1, components)
    if finite.size == 0:
        return ScalarFieldInfo(
            name=name,
            association=ScalarAssociation.POINT,
            components=components,
            tuples=int(np.prod(values.shape[:3])),
            dtype=values.dtype,
            finite_minimum=None,
            finite_maximum=None,
            non_finite_count=int(values.size),
        )
    minima = tuple(float(finite[:, c].min()) for c in range(components))
    maxima = tuple(float(finite[:, c].max()) for c in range(components))
    return ScalarFieldInfo(
        name=name,
        association=ScalarAssociation.POINT,
        components=components,
        tuples=int(np.prod(values.shape[:3])),
        dtype=values.dtype,
        finite_minimum=minima,
        finite_maximum=maxima,
        non_finite_count=int(values.size - finite.size),
    )


def _read_bands(path: Path, *, reference_latitude: float | None, band_as_field: bool):
    rasterio = _import_rasterio()
    with rasterio.open(str(path)) as dataset:
        transform = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        count = dataset.count
        origin, spacing = _geotransform_to_local(transform, crs, reference_latitude=reference_latitude)
        point_dimensions = (width, height, 1)
        bounds = make_bounds(origin, spacing, point_dimensions)

        fields: dict[str, ScalarField] = {}
        infos: list[ScalarFieldInfo] = []
        if band_as_field:
            for band_index in range(1, count + 1):
                array = np.asarray(dataset.read(band_index), dtype=np.float64)
                # Flip rows so ENU y increases north (model wants (z, y, x)).
                array = array[::-1, :]
                # Add a leading z axis -> (1, rows, cols) == (z, y, x)
                values = array[np.newaxis, ...]
                name = _band_name(dataset, band_index)
                fields[name] = ScalarField(name=name, association=ScalarAssociation.POINT, values=values)
                infos.append(_make_info(name, values))
        else:
            # Stack every band as a component of a single field (e.g. RGB).
            name = _band_name(dataset, 1) if count == 1 else "data"
            arrays = [
                np.asarray(dataset.read(band_index), dtype=np.float64)[::-1, :]
                for band_index in range(1, count + 1)
            ]
            stacked = np.stack(arrays, axis=-1)  # (rows, cols, bands)
            values = stacked[np.newaxis, ...]  # (1, rows, cols, bands) == (z, y, x, components)
            fields[name] = ScalarField(name=name, association=ScalarAssociation.POINT, values=values)
            infos.append(_make_info(name, values))
        return fields, infos, point_dimensions, origin, spacing, bounds, count


def inspect_geotiff(path: str | Path, *, reference_latitude: float | None = None, band_as_field: bool = True) -> VtiInspection:
    """Enumerate GeoTIFF bands as fields with finite ranges."""

    fields, infos, point_dimensions, origin, spacing, bounds, _ = _read_bands(
        Path(path), reference_latitude=reference_latitude, band_as_field=band_as_field
    )
    return VtiInspection(
        point_dimensions=point_dimensions,
        origin=origin,
        spacing=spacing,
        bounds=bounds,
        fields=tuple(infos),
    )


def read_geotiff(
    path: str | Path,
    *,
    field_name: str | None = None,
    reference_latitude: float | None = None,
    band_as_field: bool = True,
) -> StructuredVoxelDataset:
    """Load one (or all) GeoTIFF band(s) into a structured voxel dataset."""

    fields, _infos, point_dimensions, origin, spacing, bounds, count = _read_bands(
        Path(path), reference_latitude=reference_latitude, band_as_field=band_as_field
    )
    if not fields:
        raise GeotiffReadError(f"GeoTIFF has no readable bands: {path}")
    if field_name is not None:
        if field_name not in fields:
            available = ", ".join(fields) or "none"
            raise GeotiffReadError(f"field {field_name!r} not found; available: {available}")
        fields = {field_name: fields[field_name]}
    return StructuredVoxelDataset(
        point_dimensions=point_dimensions,
        origin=origin,
        spacing=spacing,
        bounds=bounds,
        fields=fields,
    )
