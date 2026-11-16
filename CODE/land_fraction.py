"""Precompute and query a global, radius-averaged land fraction.

The global generator uses a fast spherical, area-weighted raster approximation:
source grid-cell centres inside a geodesic circle are included, and each
cell is weighted by its spherical area. Its spatial accuracy is limited by
THE INPUT land-sea mask (not the output sampling resolution).

`make_land_fraction` retains a much slower local LAEA/Shapely method for
spot-checking coastal samples against the raster approximation.

Output coordinates: latitude increasing (south to north), longitude in
[0, 360), cyclic at 360. Output is a NetCDF file with a float32 field.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

LAND_MASK_PATH = Path("/home/mbaldacchino/data/era5_land_sea_mask.nc")
LAND_FRACTION_PATH = Path("/home/mbaldacchino/data/land_fraction_r100_0.05deg.nc")
EARTH_RADIUS_KM = 6371.0088


def _cell_edges(
    x: np.ndarray, *, clip: tuple[float, float] | None = None
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError(
            "Coordinate must be a strictly increasing one-dimensional grid"
        )
    edges = np.r_[
        x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2, x[-1] + (x[-1] - x[-2]) / 2
    ]
    if clip is not None:
        edges = np.clip(edges, *clip)
    return edges


def _prepare_mask(land_mask: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize 2-D input to ascending latitude and periodic [0,360) longitude."""
    renames = {}
    if "lat" in land_mask.dims:
        renames["lat"] = "latitude"
    if "lon" in land_mask.dims:
        renames["lon"] = "longitude"
    land_mask = land_mask.rename(renames)
    if set(land_mask.dims) != {"latitude", "longitude"}:
        raise ValueError("Land mask must have only latitude and longitude dimensions")
    lat = np.asarray(land_mask.latitude.values, dtype=float)
    lon = np.asarray(land_mask.longitude.values, dtype=float) % 360.0
    raw = np.asarray(land_mask.transpose("latitude", "longitude").values, dtype=float)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat, lon = lat[lat_order], lon[lon_order]
    raw = raw[np.ix_(lat_order, lon_order)]
    # Drop duplicate longitude (e.g. both 0 and 360 in the input).
    keep = np.r_[True, np.diff(lon) > 1e-8]
    lon, raw = lon[keep], raw[:, keep]
    if len(lon) < 3 or len(lat) < 3:
        raise ValueError("Land mask grid too small")
    if not np.allclose(np.diff(lat), np.diff(lat)[0], atol=1e-5):
        raise ValueError("Expected a regular latitude grid")
    dlon = 360.0 / len(lon)
    if not np.allclose(np.diff(lon), dlon, atol=1e-5) or not np.isclose(
        (lon[0] - lon[-1]) % 360.0, dlon, atol=1e-5
    ):
        raise ValueError("Land mask must cover 360 degrees on a regular longitude grid")
    if (
        lat[0] - (lat[1] - lat[0]) / 2 > -90 + 1e-3
        or lat[-1] + (lat[-1] - lat[-2]) / 2 < 90 - 1e-3
    ):
        raise ValueError("Input land mask must cover the full latitude range")
    if np.any(np.isfinite(raw) & ((raw < -1e-6) | (raw > 1 + 1e-6))):
        raise ValueError("Land-sea mask must be in [0,1]")
    return lat, lon, np.clip(raw, 0, 1).astype(np.float32)


def _load_mask(path: str | Path, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(
                f"{variable!r} not found in {path}; fields: {list(ds.data_vars)}"
            )
        mask = ds[variable]
        for dim in list(mask.dims):
            if dim not in {"lat", "lon", "latitude", "longitude"}:
                if mask.sizes[dim] > 1 and dim not in {"time", "valid_time"}:
                    raise ValueError(f"Unexpected dimension {dim!r} in land mask")
                mask = mask.isel({dim: 0}, drop=True)
        return mask.load()


def make_land_fraction(
    land_mask: xr.DataArray, radius_km: float = 100, min_soil_coverage=None
):
    """Return slow exact-area spot-check sampler: sample(lat,lon) -> dict.

    The cells are classified by the source land-sea mask and intersected with
    a local Lambert azimuthal equal-area circle. Not intended for global raster
    generation (millions of samples would be extremely expensive).
    The legacy min_soil_coverage argument is accepted but unused: no soil data.
    """
    from pyproj import CRS, Transformer
    from shapely.geometry import Point
    from shapely import area, intersection, polygons

    lat, lon, mask = _prepare_mask(land_mask)
    late = _cell_edges(lat, clip=(-90, 90))
    lone = _cell_edges(lon)
    radius_m = float(radius_km) * 1000.0

    def sample(lat0: float, lon0: float) -> dict:
        lat0 = float(lat0)
        lon0 = float(lon0) % 360
        if not np.isfinite(lat0) or not -90 <= lat0 <= 90:
            return {"land_fraction": float("nan")}
        dlat = radius_km / 110.0 + np.max(np.diff(late))
        dlon = min(180.0, dlat / max(0.02, np.cos(np.deg2rad(lat0))))
        ilat = np.flatnonzero((late[:-1] <= lat0 + dlat) & (late[1:] >= lat0 - dlat))
        delta = (lon - lon0 + 180) % 360 - 180
        ilon = np.flatnonzero(np.abs(delta) <= dlon + np.max(np.diff(lone)))
        if not len(ilat) or not len(ilon):
            return {"land_fraction": float("nan")}
        ii, jj = np.meshgrid(ilat, ilon, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        local_lon = lon0 + (lon[jj] - lon0 + 180) % 360 - 180
        shift = local_lon - lon[jj]
        x0, x1 = lone[jj] + shift, lone[jj + 1] + shift
        y0, y1 = late[ii], late[ii + 1]
        corners_lon = np.column_stack((x0, x1, x1, x0, x0))
        corners_lat = np.column_stack((y0, y0, y1, y1, y0))
        crs = CRS.from_proj4(
            f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
        )
        trans = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xx, yy = trans.transform(corners_lon, corners_lat)
        cells = polygons(np.stack((xx, yy), axis=-1))
        disc = Point(0, 0).buffer(radius_m, quad_segs=48)
        weights = area(intersection(cells, disc))
        vals = mask[ii, jj]
        valid = np.isfinite(vals)
        covered = weights[valid].sum()
        if covered < 0.99 * disc.area:
            return {"land_fraction": float("nan")}
        return {"land_fraction": float(np.sum(weights[valid] * vals[valid]) / covered)}

    return sample


def _window_sum(
    prefix: np.ndarray,
    grid_lon: np.ndarray,
    target_lon: np.ndarray,
    halfwidth_deg: float,
) -> np.ndarray:
    """Fast cyclic sum of regular longitude cells in a window around targets."""
    if halfwidth_deg >= 180.0 - 1e-8:
        return np.full(target_lon.shape, prefix[-1], dtype=np.float64)
    left = (target_lon - halfwidth_deg) % 360
    right = (target_lon + halfwidth_deg) % 360
    a = np.searchsorted(grid_lon, left, side="left")
    b = np.searchsorted(grid_lon, right, side="right")
    return prefix[b] - prefix[a] + np.where(b < a, prefix[-1], 0.0)


def _compute_area_weighted(
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    mask: np.ndarray,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """Spherical-cap area-weighted convolution, sampled on regular output grid.

    Includes input CELL CENTRES inside the great-circle circle; source cell
    area is weighted using sin(latitude_edge). No Shapely per output point.
    """
    if radius_km <= 0 or radius_km > 3000:
        raise ValueError("radius_km must be in (0, 3000]")
    if np.nanmax(np.diff(lat_in)) > 0.3 or 360 / len(lon_in) > 0.3:
        warnings.warn(
            "Coarse input land mask may yield inaccurate small-island and coastline fractions",
            stacklevel=2,
        )
    theta = radius_km / EARTH_RADIUS_KM
    cos_theta = np.cos(theta)
    lat_edges = np.deg2rad(_cell_edges(lat_in, clip=(-90.0, 90.0)))
    area_weight = np.diff(np.sin(lat_edges))
    lat_radians = np.deg2rad(lat_in)
    sin_lat, cos_lat = np.sin(lat_radians), np.cos(lat_radians)
    out = np.empty((len(lat_out), len(lon_out)), dtype=np.float32)

    @lru_cache(maxsize=128)
    def cumulative_for_row(j: int):
        row = mask[j]
        valid = np.isfinite(row)
        land = np.where(valid, row, 0).astype(np.float64)
        return (
            np.r_[0, np.cumsum(land)],
            np.r_[0, np.cumsum(valid.astype(np.float64))],
        )

    lat_radius = np.rad2deg(theta)
    for i, phi_deg in enumerate(lat_out):
        phi = np.deg2rad(phi_deg)
        # Indices of source cell centres within latitudinal extent of disc.
        lo = np.searchsorted(lat_in, phi_deg - lat_radius, side="left")
        hi = np.searchsorted(lat_in, phi_deg + lat_radius, side="right")
        land_area = np.zeros(len(lon_out), dtype=np.float64)
        valid_area = np.zeros(len(lon_out), dtype=np.float64)
        sin_p, cos_p = np.sin(phi), np.cos(phi)
        for j in range(lo, hi):
            denom = cos_p * cos_lat[j]
            if abs(denom) < 1e-12:
                halfwidth = 180.0 if (sin_p * sin_lat[j]) >= cos_theta else -1.0
            else:
                c = (cos_theta - sin_p * sin_lat[j]) / denom
                if c >= 1.0:
                    continue
                halfwidth = 180.0 if c <= -1 else np.rad2deg(np.arccos(c))
            if halfwidth < 0:
                continue
            pref_land, pref_valid = cumulative_for_row(j)
            w = area_weight[j]
            land_area += w * _window_sum(pref_land, lon_in, lon_out, halfwidth)
            valid_area += w * _window_sum(pref_valid, lon_in, lon_out, halfwidth)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.where(valid_area > 0, land_area / valid_area, np.nan)
    # Due to floating point summation, very small excursions outside [0,1]
    return np.clip(out, 0, 1)


def generate_land_fraction(
    radius_km: float = 100,
    resolution: float = 0.05,
    land_mask_path: str | Path = LAND_MASK_PATH,
    output_path: str | Path | None = None,
    variable: str = "lsm",
    lat_bounds: tuple[float, float] = (-90, 90),
) -> xr.Dataset:
    """Generate precomputed global/regional land fractions and write a NetCDF.

    `lat_bounds` supports small tests, but operational simulation should use
    the full global (-90,90) raster. Output longitude is always global/cyclic.
    """
    if resolution <= 0 or not np.isclose(360 / resolution, round(360 / resolution)):
        raise ValueError("Resolution must evenly divide 360 degrees")
    if not (-90 <= lat_bounds[0] < lat_bounds[1] <= 90):
        raise ValueError("Invalid latitude bounds")
    mask_da = _load_mask(land_mask_path, variable)
    lat_in, lon_in, mask = _prepare_mask(mask_da)
    lat_out = np.arange(-90 + resolution / 2, 90, resolution, dtype=np.float64)
    lat_out = lat_out[(lat_out >= lat_bounds[0]) & (lat_out < lat_bounds[1])]
    lon_out = np.arange(round(360 / resolution), dtype=np.float64) * resolution
    if output_path is None:
        output_path = (
            LAND_FRACTION_PATH.parent
            / f"land_fraction_r{radius_km:g}_{resolution:g}deg.nc"
        )
    # Classic NetCDF has file-size restrictions. Choose a NetCDF4 backend
    # before an expensive generation if the output will exceed ~1.5 GB.
    engine = None
    if len(lat_out) * len(lon_out) * 4 > 1_500_000_000:
        try:
            import h5netcdf  # noqa: F401

            engine = "h5netcdf"
        except ImportError:
            try:
                import netCDF4  # noqa: F401

                engine = "netcdf4"
            except ImportError as exc:
                raise RuntimeError(
                    "Large raster needs h5netcdf or netCDF4 for NetCDF4 output; "
                    "or use a coarser resolution such as 0.05 degrees."
                ) from exc
    land_frac = _compute_area_weighted(
        lat_in, lon_in, mask, lat_out, lon_out, radius_km
    )
    ds = xr.Dataset(
        {"land_fraction": (("lat", "lon"), land_frac)},
        coords={"lat": lat_out, "lon": lon_out},
        attrs={
            "radius_km": float(radius_km),
            "output_resolution_deg": float(resolution),
            "source": str(land_mask_path),
            "method": "spherical cap, source-cell centres, area weighted",
            "longitude_convention": "0 to 360, periodic",
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path, engine=engine)  # float32, no compression dependency
    return ds


@lru_cache(maxsize=2)
def load_land_fraction(path: str | Path = LAND_FRACTION_PATH) -> xr.DataArray:
    """Load output raster once per process; subsequent queries use memory."""
    with xr.open_dataset(path) as ds:
        field = ds["land_fraction"].load()
    return field


def get_land_fraction(
    lat,
    lon,
    *,
    path: str | Path = LAND_FRACTION_PATH,
    raster: xr.DataArray | None = None,
):
    """Vectorized, cyclic bilinear interpolation. Scalars in, float out."""
    if raster is None:
        raster = load_land_fraction(str(path))
    values = raster.values
    lats = np.asarray(raster.lat.values)
    lons = np.asarray(raster.lon.values)
    if len(lats) < 2 or len(lons) < 2:
        raise ValueError("Raster needs at least 2 latitude and longitude points")
    dy, dx = lats[1] - lats[0], lons[1] - lons[0]
    lat_arr, lon_arr = np.broadcast_arrays(
        np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    )
    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    safe_lat = np.where(valid, lat_arr, lats[0])
    safe_lon = np.where(valid, lon_arr, lons[0])
    # Clip only latitude. Longitude is periodic, including dateline crossings.
    fy = (np.clip(safe_lat, lats[0], lats[-1]) - lats[0]) / dy
    i0 = np.clip(np.floor(fy).astype(int), 0, len(lats) - 2)
    ay = np.clip(fy - i0, 0, 1)
    fx = ((safe_lon % 360) - lons[0]) / dx
    fx = fx % len(lons)
    j0 = np.floor(fx).astype(int)
    j1 = (j0 + 1) % len(lons)
    ax = fx - j0
    v = (1 - ay) * ((1 - ax) * values[i0, j0] + ax * values[i0, j1]) + ay * (
        (1 - ax) * values[i0 + 1, j0] + ax * values[i0 + 1, j1]
    )
    v = np.where(valid, v, np.nan)
    return float(v) if v.ndim == 0 else v


if __name__ == "__main__":
    generate_land_fraction(radius_km=100, resolution=0.05)
