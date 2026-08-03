"""Read NetCDF (2D/3D) scientific grids into validated structured voxel datasets.

A NetCDF variable such as ``u(time, level, lat, lon)`` is mapped onto the
project's local ENU voxel frame:

* spatial dimensions are detected by name (``lon/x/easting``, ``lat/y/northing``,
  ``level/height/depth/z``) and any remaining non-spatial dimension (``time``)
  is dropped (first slice, or ``time_index``);
* coordinate variables give the axis positions; their deltas become the cell
  spacing in metres (angular ``lon``/``lat`` coordinates are converted to metres
  with a reference latitude);
* ``origin`` is the local (0, 0, 0) corner, anchored to the globe by the
  pipeline georeference.

The optional ``netCDF4`` package is required and imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

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
    regular_spacing,
)

X_NAMES = {"x", "lon", "longitude", "easting", "east", "xc", "lonc", "i"}
Y_NAMES = {"y", "lat", "latitude", "northing", "north", "yc", "latc", "j"}
Z_NAMES = {"z", "level", "lev", "levels", "height", "altitude", "depth", "depth_t",
           "depth_u", "depth_v", "zc", "k", "s_rho", "s_w", "sigma", "eta",
           "sigma_theta", "plev", "pressure", "isobaric", "isobaric1", "hybrid",
           "model_level", "model_levels", "nz", "zl"}
TIME_NAMES = {"time", "t", "datetime", "times", "date", "month", "day",
               "year", "hour", "timestep", "step"}

# Coordinate names that are unambiguously angular degrees even without a units
# attribute (CF convention relies on units, but many files omit them).
ANGULAR_NAMES = {"lon", "longitude", "lat", "latitude", "lonc", "latc"}


class NetcdfReadError(ValueError):
    """Raised when a NetCDF file cannot be interpreted safely."""


def _import_netcdf():
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise NetcdfReadError(
            "reading NetCDF requires the optional 'netCDF4' package; "
            "install it with: pip install netCDF4"
        ) from exc
    return nc


def _classify_dimensions(
    var_dims: Sequence[str],
    *,
    x_dim: str | None,
    y_dim: str | None,
    z_dim: str | None,
) -> dict[str, str]:
    """Assign each variable dimension a role: x, y, z, drop (time), or unknown."""

    roles: dict[str, str] = {}
    for explicit, role in ((x_dim, "x"), (y_dim, "y"), (z_dim, "z")):
        if explicit:
            roles[explicit] = role
    for dim in var_dims:
        if dim in roles:
            continue
        lower = dim.lower()
        if lower in X_NAMES:
            roles[dim] = "x"
        elif lower in Y_NAMES:
            roles[dim] = "y"
        elif lower in Z_NAMES:
            roles[dim] = "z"
        elif lower in TIME_NAMES:
            roles[dim] = "drop"
        else:
            roles[dim] = "unknown"
    return roles


def _extract_spatial(values: np.ndarray, roles: Mapping[str, str], var_dims: Sequence[str]) -> np.ndarray:
    """Slice dropped (time) dims and transpose to ``(z, y, x[, components])``.

    A single unrecognized non-time dimension is treated as a trailing component
    axis (e.g. vector components). More than one raises.
    """

    unknown_positions = [i for i, d in enumerate(var_dims) if roles[d] == "unknown"]
    if len(unknown_positions) > 1:
        bad = [var_dims[i] for i in unknown_positions]
        raise NetcdfReadError(
            f"variable has multiple unrecognized dimensions {bad}; pass explicit "
            "x_dim/y_dim/z_dim or ensure only 'time' is extra"
        )

    slicer = []
    for dim in var_dims:
        slicer.append(0 if roles[dim] == "drop" else slice(None))
    arr = values[tuple(slicer)]

    role_to_pos: dict[str, int] = {}
    position = 0
    for dim in var_dims:
        role = roles[dim]
        if role in ("x", "y", "z"):
            role_to_pos[role] = position
            position += 1
        elif role == "unknown":
            comp_pos = position
            position += 1
    if "x" not in role_to_pos or "y" not in role_to_pos:
        raise NetcdfReadError(
            f"could not locate both x and y axes in dimensions {list(var_dims)}; "
            "pass explicit x_dim/y_dim/z_dim"
        )

    order: list[int] = []
    has_z = "z" in role_to_pos
    if has_z:
        order.append(role_to_pos["z"])
    order.append(role_to_pos["y"])
    order.append(role_to_pos["x"])
    if unknown_positions:
        order.append(comp_pos)
    arr = np.transpose(arr, order)
    if not has_z:
        arr = np.expand_dims(arr, axis=0)  # single vertical layer
    return arr


def _coordinate_values(nc_dataset, dim_name: str, role: str, *, reference_latitude_deg: float) -> tuple[np.ndarray, float]:
    """Return (coordinate array, spacing-metres) for a spatial axis.

    If no coordinate variable exists the axis is treated as unit-indexed
    (spacing 1 m). Angular ``lon``/``lat`` coordinates are converted to metres.
    """

    coord = None
    if dim_name in nc_dataset.variables:
        coord = nc_dataset.variables[dim_name]
    else:
        for candidate in nc_dataset.variables.values():
            if candidate.dimensions == (dim_name,) and candidate.ndim == 1:
                coord = candidate
                break
    if coord is None:
        return np.arange(0.0), 1.0
    values = np.asarray(coord[:], dtype=float).ravel()
    units = getattr(coord, "units", None)
    angular = is_degree_units(units) or (role in ("x", "y") and dim_name.lower() in ANGULAR_NAMES)
    per_degree = meters_per_degree(role, reference_latitude_deg) if angular else 1.0
    return values, regular_spacing(values) * per_degree


def _collect_data_variables(nc_dataset) -> list[str]:
    """Named variables that expose at least x+y (optionally z) spatial dims."""

    found: list[str] = []
    for name, variable in nc_dataset.variables.items():
        if variable.ndim == 0:
            continue
        dims = list(variable.dimensions)
        roles = _classify_dimensions(dims, x_dim=None, y_dim=None, z_dim=None)
        if "x" in roles.values() and "y" in roles.values():
            found.append(name)
    return found


def _build_field(nc_dataset, var_name: str, roles: Mapping[str, str], *, reference_latitude_deg: float) -> ScalarField:
    variable = nc_dataset.variables[var_name]
    values = np.asarray(variable[:], dtype=np.float64)
    spatial = _extract_spatial(values, roles, list(variable.dimensions))
    return ScalarField(name=var_name, association=ScalarAssociation.POINT, values=spatial)


def _field_info(nc_dataset, var_name: str, roles: Mapping[str, str], *, reference_latitude_deg: float) -> ScalarFieldInfo:
    variable = nc_dataset.variables[var_name]
    values = np.asarray(variable[:], dtype=np.float64)
    spatial = _extract_spatial(values, roles, list(variable.dimensions))
    components = 1 if spatial.ndim == 3 else spatial.shape[-1]
    finite = spatial[np.isfinite(spatial)]
    if finite.size == 0:
        minima = maxima = None
        non_finite = int(spatial.size)
    else:
        flat = finite.reshape(-1, components) if components > 1 else finite.reshape(-1, 1)
        minima = tuple(float(flat[:, c].min()) for c in range(components))
        maxima = tuple(float(flat[:, c].max()) for c in range(components))
        non_finite = int(spatial.size - finite.size)
    tuples = int(np.prod(spatial.shape[:3]))
    return ScalarFieldInfo(
        name=var_name,
        association=ScalarAssociation.POINT,
        components=components,
        tuples=tuples,
        dtype=spatial.dtype,
        finite_minimum=minima,
        finite_maximum=maxima,
        non_finite_count=non_finite,
    )


def inspect_netcdf(path: str | Path, *, reference_latitude: float | None = None, x_dim: str | None = None, y_dim: str | None = None, z_dim: str | None = None) -> VtiInspection:
    """Enumerate NetCDF data variables and their finite ranges without selecting one."""

    nc = _import_netcdf()
    dataset = nc.Dataset(str(path), "r")
    try:
        data_vars = _collect_data_variables(dataset)
        if not data_vars:
            raise NetcdfReadError(f"no x/y spatial variable found in {path}")
        # Use the first recognized variable to derive grid geometry.
        roles = _classify_dimensions(
            dataset.variables[data_vars[0]].dimensions, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim
        )
        ref_lat = reference_latitude if reference_latitude is not None else _mean_latitude(dataset, roles)
        x_vals, sx = _coordinate_values(dataset, _role_dim(roles, "x"), "x", reference_latitude_deg=ref_lat)
        y_vals, sy = _coordinate_values(dataset, _role_dim(roles, "y"), "y", reference_latitude_deg=ref_lat)
        z_dim_name = _role_dim(roles, "z")
        if z_dim_name is not None:
            _, sz = _coordinate_values(dataset, z_dim_name, "z", reference_latitude_deg=ref_lat)
            nz = len(dataset.variables[z_dim_name])
        else:
            sz = 1.0
            nz = 1
        nx = x_vals.size if x_vals.size > 1 else int(_axis_len(dataset, roles, "x"))
        ny = y_vals.size if y_vals.size > 1 else int(_axis_len(dataset, roles, "y"))
        point_dimensions = (nx, ny, nz)
        origin = (0.0, 0.0, 0.0)
        spacing = (sx, sy, sz)
        bounds = make_bounds(origin, spacing, point_dimensions)
        fields = tuple(
            _field_info(dataset, name, _classify_dimensions(dataset.variables[name].dimensions, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim), reference_latitude_deg=ref_lat)
            for name in data_vars
        )
        return VtiInspection(
            point_dimensions=point_dimensions,
            origin=origin,
            spacing=spacing,
            bounds=bounds,
            fields=fields,
        )
    finally:
        dataset.close()


def read_netcdf(
    path: str | Path,
    *,
    field_name: str | None = None,
    reference_latitude: float | None = None,
    x_dim: str | None = None,
    y_dim: str | None = None,
    z_dim: str | None = None,
) -> StructuredVoxelDataset:
    """Load one (or all) NetCDF variable(s) into a structured voxel dataset."""

    nc = _import_netcdf()
    dataset = nc.Dataset(str(path), "r")
    try:
        data_vars = _collect_data_variables(dataset)
        if not data_vars:
            raise NetcdfReadError(f"no x/y spatial variable found in {path}")
        selected: list[str]
        if field_name is not None:
            if field_name not in data_vars:
                available = ", ".join(data_vars) or "none"
                raise NetcdfReadError(f"field {field_name!r} not found; available: {available}")
            selected = [field_name]
        elif len(data_vars) == 1:
            selected = [data_vars[0]]
        else:
            available = ", ".join(data_vars)
            raise NetcdfReadError(f"multiple fields available; select one explicitly: {available}")

        ref_lat = reference_latitude if reference_latitude is not None else _mean_latitude(dataset, None)
        fields: dict[str, ScalarField] = {}
        point_dimensions = None
        origin = (0.0, 0.0, 0.0)
        spacing: tuple[float, float, float] | None = None
        for name in selected:
            roles = _classify_dimensions(dataset.variables[name].dimensions, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
            field = _build_field(dataset, name, roles, reference_latitude_deg=ref_lat)
            fields[field.name] = field
            if point_dimensions is None:
                point_dimensions = field.dimensions_xyz
                x_vals, sx = _coordinate_values(dataset, _role_dim(roles, "x"), "x", reference_latitude_deg=ref_lat)
                y_vals, sy = _coordinate_values(dataset, _role_dim(roles, "y"), "y", reference_latitude_deg=ref_lat)
                z_name = _role_dim(roles, "z")
                if z_name is not None:
                    _, sz = _coordinate_values(dataset, z_name, "z", reference_latitude_deg=ref_lat)
                else:
                    sz = 1.0
                spacing = (sx, sy, sz)
        assert point_dimensions is not None and spacing is not None
        bounds = make_bounds(origin, spacing, point_dimensions)
        return StructuredVoxelDataset(
            point_dimensions=point_dimensions,
            origin=origin,
            spacing=spacing,
            bounds=bounds,
            fields=fields,
        )
    finally:
        dataset.close()


def _role_dim(roles: Mapping[str, str], role: str) -> str | None:
    for dim, r in roles.items():
        if r == role:
            return dim
    return None


def _axis_len(nc_dataset, roles: Mapping[str, str], role: str) -> int:
    dim = _role_dim(roles, role)
    if dim is None:
        return 1
    return int(nc_dataset.dimensions[dim].size)


def _mean_latitude(nc_dataset, roles: Mapping[str, str] | None) -> float:
    """Reference latitude for lon->metre conversion (mean of y coord, else 0)."""

    if roles is None:
        y_candidates = [v for v in nc_dataset.variables.values() if getattr(v, "units", None) and is_degree_units(getattr(v, "units", None)) and "lat" in v.name.lower()]
        if y_candidates:
            return float(np.mean(np.asarray(y_candidates[0][:], dtype=float)))
    else:
        y_dim = _role_dim(roles, "y")
        if y_dim is not None and y_dim in nc_dataset.variables:
            values = np.asarray(nc_dataset.variables[y_dim][:], dtype=float).ravel()
            if values.size and is_degree_units(getattr(nc_dataset.variables[y_dim], "units", None)):
                return float(np.mean(values))
    return 0.0
