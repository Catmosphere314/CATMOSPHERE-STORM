"""Historical genesis / 1-degree OSM exposure preparation for CATMOSPHERE-STORM.

No side effects on the existing STORM / GPI-MIX simulator. In particular,
this does not replace basin-frequency or starting-point samplers.

The observational definition is the *first retained 3-hourly TS track fix*
from CODE.preprocessing.extract_data; not first tropical disturbance detection.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from CODE.geography import LAND_MASK_PATH, BINARY_RESOLUTION, is_land

BASINS = ("NA", "WP", "EP", "NI", "SI", "SP")
INDEX_TO_BASIN = dict(enumerate(("EP", "NA", "NI", "SI", "SP", "WP")))
ACTIVE_MONTHS = {
    "NA": (5, 6, 7, 8, 9, 10, 11, 12),
    "WP": (5, 6, 7, 8, 9, 10, 11, 12),
    "EP": (6, 7, 8, 9, 10, 11),
    "NI": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "SI": (1, 2, 3, 4, 5, 9, 10, 11, 12),
    "SP": (1, 2, 3, 4, 5, 6, 10, 11, 12),
}
OMEGA = 7.2921e-5
EARTH_RADIUS_KM = 6371.0088
MJD_ORIGIN = pd.Timestamp("1858-11-17")
ENV_SOURCE = {"pi": "VMAX_PI", "rh600": "RH600", "vws": "VWS", "vort850": "VORT850"}
ETA_CLIP = {
    "NA": 3.5e-5,
    "WP": 4.5e-5,
    "EP": 3.5e-5,
    "NI": 4.5e-5,
    "SI": 5.0e-5,
    "SP": 4.0e-5,
}
# Match the legacy genesis_matrix.OCEAN_FRAC_THRESHOLD = 0.2.
# This is the maximum coarse sampled LAND fraction, not minimum ocean fraction.
# The legacy simulator keeps its own full storm/track basin bounds.
MAX_GENESIS_COARSE_LAND_FRAC = 0.20

REGION_REFERENCE = {
    "NA": "subtropical",
    "WP": "scs",
    "EP": "subtropical",
    "NI": "bay_of_bengal",
    "SI": "central_south_indian_ocean",
    "SP": "central_south_pacific",
}


def BOUNDARIES_BASINS_GENESIS(idx):
    if idx == "EP":  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 35, 180, 285
    if idx == "NA":  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 35, 255, 360
    if idx == "NI":  # North Indian
        lat0, lat1, lon0, lon1 = 5, 35, 50, 100
    if idx == "SI":  # South Indian
        lat0, lat1, lon0, lon1 = -35, -5, 10, 135
    if idx == "SP":  # South Pacific
        lat0, lat1, lon0, lon1 = -35, -5, 135, 240
    if idx == "WP":  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 35, 100, 180

    return lat0, lat1, lon0, lon1


def basin_cell_centers_genesis(basin, rows, cols):
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS_GENESIS(basin)

    dlat = (lat1 - lat0) / rows
    dlon = (lon1 - lon0) / cols

    # Grid is stored north-to-south.
    lat = lat1 - (np.arange(rows) + 0.5) * dlat
    lon = lon0 + (np.arange(cols) + 0.5) * dlon

    LON, LAT = np.meshgrid(lon, lat)
    return LAT, LON


@dataclass(frozen=True)
class GenesisConfig:
    basin: str
    start_year: int = 1980
    end_year: int = 2025
    first_test_year: int = 2016
    area_unit_km2: float = 100_000.0
    reference_month_days: float = 365.2425 / 12.0

    @property
    def active_months(self):
        return ACTIVE_MONTHS[self.basin]

    @property
    def bounds(self):
        return BOUNDARIES_BASINS_GENESIS(self.basin)


def _load_dict(folder: Path, filename: str) -> dict:
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run MASTER_preprocessing.py first")
    obj = np.load(path, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a storm-indexed dict in {path}")
    return obj


def load_first_ts_genesis(data_root: Path, basins: list = None) -> pd.DataFrame:
    """One row per retained storm, indexed by original IBTrACS storm position.

    Args:
        data_root: Path to the folder containing the preprocessed TC dictionaries.
        basins: Optional list of basin names to filter the output. If None, all basins are included.

    Use TIMELIST_INTERP as numerical MJD days and BASINLIST_INTERP as the basin
    at the first TS-threshold crossing. YEARLIST_INTERP is IBTrACS `season` and
    must NOT be used as the calendar year of genesis. The first *retained* fix
    can be later than the original threshold crossing if off the 3-hour grid.
    """
    folder = Path(data_root)
    lat = _load_dict(folder, "LATLIST_INTERP.npy")
    lon = _load_dict(folder, "LONLIST_INTERP.npy")
    times = _load_dict(folder, "TIMELIST_INTERP.npy")
    basin = _load_dict(folder, "BASINLIST_INTERP.npy")
    if not (set(lat) == set(lon) == set(times) == set(basin)):
        raise AssertionError("Preprocessed TC dictionaries have different storm IDs")
    records = []
    for sid in sorted(lat):
        if not len(lat[sid]):
            continue
        if not len(lon[sid]) or not len(times[sid]) or len(basin[sid]) != 1:
            raise ValueError(f"Storm {sid}: missing first retained fix/time/basin")
        raw_basin = basin[sid][0]
        if isinstance(raw_basin, (bytes, np.bytes_)):
            raw_basin = raw_basin.decode()
        if isinstance(raw_basin, (int, float, np.integer)):
            name = INDEX_TO_BASIN.get(int(raw_basin))
        else:
            name = str(raw_basin).upper()
        if (basins is not None) and (name not in basins):
            continue  # Skip basins not requested by the caller 
        if name not in BASINS:
            raise ValueError(f"Storm {sid}: unknown genesis basin {raw_basin!r}")
        mjd = float(times[sid][0])
        # The preprocessing stores 3-hour grid timestamps, rounded to 3 decimals.
        timestamp = MJD_ORIGIN + pd.to_timedelta(mjd, unit="D")
        la = float(lat[sid][0])
        lo = float(lon[sid][0]) % 360.0
        if not (pd.notna(timestamp) and np.isfinite(la) and np.isfinite(lo)):
            raise ValueError(f"Storm {sid}: missing first-fix coordinates/date")
        records.append((int(sid), name, timestamp, la, lo, mjd))
    g = pd.DataFrame(
        records,
        columns=["storm_id", "basin", "time", "latitude", "longitude", "mjd_first_fix"],
    )
    if g.empty or g.storm_id.duplicated().any():
        raise AssertionError(
            "No usable retained genesis events, or duplicated storm IDs"
        )
    g["year"] = g.time.dt.year.astype(int)
    g["month"] = g.time.dt.month.astype(int)
    g["time_month"] = g.time.dt.to_period("M").dt.to_timestamp()
    return g


def basin_side_allowed(basin: str, lat, lon):
    """Apply exactly the NA/EP Central American separator of the repo sampler.

    Unlike _genesis_ocean_mask_1deg, this can operate at the *fine-point* level.
    """
    lat, lon = np.broadcast_arrays(
        np.asarray(lat, dtype=float), np.asarray(lon, dtype=float) % 360
    )
    if basin not in ("NA", "EP"):
        return np.ones(lat.shape, dtype=bool)
    cutoff = 16.0 - 0.571 * (lon - 268.0)
    in_overlap = (lon >= 255.0) & (lon < 285.0)
    if basin == "NA":
        return ~(in_overlap & (lat < cutoff))
    return ~(in_overlap & (lat >= cutoff))


def _metadata_for_osm(path: Path) -> dict:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; regenerate OSM geography metadata"
        )
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        meta.get("source") != "OpenStreetMap / osmdata land polygons"
        or meta.get("lakes") != "land"
    ):
        raise ValueError("Incompatible OSM land mask provenance")
    if float(meta.get("resolution_deg", -1)) != BINARY_RESOLUTION or not meta.get(
        "source_sha256"
    ):
        raise ValueError("OSM land mask resolution or snapshot digest missing")
    return meta


def spherical_1deg_area_km2(latitude):
    lat = np.asarray(latitude, dtype=float)
    return (
        EARTH_RADIUS_KM**2
        * np.deg2rad(1.0)
        * (np.sin(np.deg2rad(lat + 0.5)) - np.sin(np.deg2rad(lat - 0.5)))
    )


def build_osm_exposure(
    basin: str, mask_path: Path = LAND_MASK_PATH, *, verify_metadata: bool = True
) -> xr.Dataset:
    """Compute 1-degree water exposure from ALL 0.01-degree OSM pixels.

    Rows north->south; lon 0..360; packed input north->south and [-180,180).
    Preserve the legacy genesis_matrix coarse '<20% land' validity criterion
    (sampled at 0.1 degree), but use the narrower genesis-only boundaries
    and apply the NA/EP separator at every 0.01-degree pixel
    for exposure, so overlap-area is not counted twice. A later IPP position
    sampler must use this same within-cell admissibility convention.
    """
    if basin not in BASINS:
        raise ValueError(basin)
    path = Path(mask_path)
    meta = (
        _metadata_for_osm(path) if verify_metadata else {"source_sha256": "test-only"}
    )
    packed = np.load(path, mmap_mode="r", allow_pickle=False)
    if packed.shape != (18000, 4500) or packed.dtype != np.uint8:
        raise ValueError(
            f"Incorrect packed OSM mask geometry: {packed.shape}, {packed.dtype}"
        )
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS_GENESIS(basin)
    nlat, nlon = int(lat1 - lat0), int(lon1 - lon0)
    LAT, LON = basin_cell_centers_genesis(basin, nlat, nlon)
    ocean_frac = np.empty((nlat, nlon), dtype=np.float32)
    eligible_frac = np.empty_like(ocean_frac)
    ocean_valid = np.empty((nlat, nlon), dtype=bool)
    # The input longitude bits start at -180E (0 index) and wrap across 180E.
    index_lon = ((int(lon0) + 180) * 100 + np.arange(nlon * 100)) % 36000
    lon_subpixel = lon0 + (np.arange(nlon * 100) + 0.5) / 100
    for ilat in range(nlat):
        top_lat = lat1 - ilat
        start_row = int(round((90.0 - top_lat) * 100))
        # Unpack only the current latitude stripe (100 x 36000, ~3.6 MB).
        bits = np.unpackbits(packed[start_row : start_row + 100], axis=1)
        land = bits[:, index_lon].astype(bool)
        sea = ~land
        ocean_frac[ilat] = sea.reshape(100, nlon, 100).mean(axis=(0, 2))
        # Use the SAME floating-point sample locations and is_land() lookup as
        # genesis_matrix._genesis_ocean_mask_1deg, including coastline rounding.
        # The fractional exposure above still uses all exact 0.01-degree bits.
        coarse_lats = top_lat - (np.arange(10) + 0.5) / 10
        coarse_lons = (
            lon0 + (np.arange(nlon)[:, None] + (np.arange(10)[None, :] + 0.5) / 10)
        ).ravel()
        coarse_lat2, coarse_lon2 = np.meshgrid(coarse_lats, coarse_lons, indexing="ij")
        coarse_land_frac = (
            is_land(coarse_lat2, coarse_lon2, path=path)
            .reshape(10, nlon, 10)
            .mean(axis=(0, 2))
        )
        valid = coarse_land_frac < MAX_GENESIS_COARSE_LAND_FRAC
        valid &= basin_side_allowed(basin, LAT[ilat], LON[ilat])
        ocean_valid[ilat] = valid
        lat_subpixel = top_lat - (np.arange(100) + 0.5) / 100
        side_ok = basin_side_allowed(
            basin, lat_subpixel[:, None], lon_subpixel[None, :]
        )
        eligible_frac[ilat] = (sea & side_ok).reshape(100, nlon, 100).mean(
            axis=(0, 2)
        ) * valid
    lat_axis, lon_axis = LAT[:, 0], LON[0, :]
    cell_area = np.broadcast_to(
        spherical_1deg_area_km2(lat_axis)[:, None], (nlat, nlon)
    ).copy()
    ds = xr.Dataset(
        data_vars={
            "ocean_valid": (("latitude", "longitude"), ocean_valid),
            "raw_ocean_fraction": (("latitude", "longitude"), ocean_frac),
            "eligible_fraction": (("latitude", "longitude"), eligible_frac),
            "cell_area_km2": (("latitude", "longitude"), cell_area),
            "eligible_area_km2": (("latitude", "longitude"), cell_area * eligible_frac),
        },
        coords={"latitude": lat_axis, "longitude": lon_axis},
        attrs={
            "basin": basin,
            "osm_sha256": meta["source_sha256"],
            "mask_resolution_deg": 0.01,
            "mask_row_order": "north_to_south",
            "longitude_convention": "0_to_360",
            "coarse_ocean_rule": "0.1deg land < 0.2",
            "na_ep_split": "fine_pixel_and_cell_center",
            "lakes": "land",
            "genesis_bounds": [lat0, lat1, lon0, lon1],
            "coarse_max_land_fraction": MAX_GENESIS_COARSE_LAND_FRAC,
        },
    )
    if np.any(ds.eligible_fraction.values < 0) or np.any(
        ds.eligible_fraction.values > 1
    ):
        raise AssertionError("Invalid eligible ocean fractions")
    return ds


def audit_genesis(
    genesis: pd.DataFrame,
    exposures: dict[str, xr.Dataset],
    configs: dict[str, GenesisConfig],
    *,
    land_fn=is_land,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Assign each observed event to its *recorded basin*, never overlapping boxes.

    Return (complete one-row-per-storm audit, dict of eligible genesis events).
    Every filtered-out event receives an explicit exclusion_reason. A genuine
    eligible event is never silently discarded for absent environment values.
    """
    audit = genesis.copy()
    audit["exclusion_reason"] = ""
    audit["ilat"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    audit["ilon"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    audit["cell_lat"] = np.nan
    audit["cell_lon"] = np.nan
    if audit.storm_id.duplicated().any():
        raise AssertionError("More than one first-TS observation per storm")
    for basin, cfg in configs.items():
        idx = audit.index[audit.basin == basin]
        if not len(idx):
            continue
        g = audit.loc[idx]
        lat0, lat1, lon0, lon1 = cfg.bounds
        reason = np.full(len(g), "", dtype=object)
        years = g.year.to_numpy()
        months = g.month.to_numpy()
        period = (years >= cfg.start_year) & (years <= cfg.end_year)
        active = np.isin(months, cfg.active_months)
        bounds = (
            (g.latitude.to_numpy() >= lat0)
            & (g.latitude.to_numpy() < lat1)
            & (g.longitude.to_numpy() >= lon0)
            & (g.longitude.to_numpy() < lon1)
        )
        side = basin_side_allowed(basin, g.latitude.to_numpy(), g.longitude.to_numpy())
        land = land_fn(g.latitude.to_numpy(), g.longitude.to_numpy())
        row = np.floor(lat1 - g.latitude.to_numpy()).astype(int)
        col = np.floor(g.longitude.to_numpy() - lon0).astype(int)
        in_grid = (
            (row >= 0)
            & (row < exposures[basin].sizes["latitude"])
            & (col >= 0)
            & (col < exposures[basin].sizes["longitude"])
        )
        geom_ok = bounds & in_grid
        valid = np.zeros(len(g), dtype=bool)
        valid[geom_ok] = (
            exposures[basin]["eligible_area_km2"].values[row[geom_ok], col[geom_ok]] > 0
        )
        # Priority-ordered reasons; all excluded events remain visible in the audit.
        reason[~period] = "outside_year_range"
        reason[period & ~active] = "inactive_month"
        reason[period & active & ~bounds] = "outside_recorded_basin_box"
        reason[period & active & bounds & ~side] = "wrong_side_of_na_ep_isthmus"
        reason[period & active & bounds & side & land] = "on_osm_land"
        reason[period & active & bounds & side & ~land & ~valid] = (
            "ineligible_1deg_cell"
        )
        audit.loc[idx, "exclusion_reason"] = reason
        keep = reason == ""
        kept_indices = idx[keep]
        row_keep, col_keep = row[keep], col[keep]
        audit.loc[kept_indices, "ilat"] = row_keep
        audit.loc[kept_indices, "ilon"] = col_keep
        audit.loc[kept_indices, "cell_lat"] = exposures[basin].latitude.values[row_keep]
        audit.loc[kept_indices, "cell_lon"] = exposures[basin].longitude.values[
            col_keep
        ]
    unknown = ~audit.basin.isin(configs)
    if unknown.any():
        raise ValueError(
            f"Unexpected source basin: {audit.loc[unknown, 'basin'].unique()}"
        )
    included = audit[audit.exclusion_reason == ""]
    if included.storm_id.duplicated().any():
        raise AssertionError("A storm is attributed to more than one basin")
    by_basin = {b: included[included.basin == b].copy() for b in configs}
    assert sum(map(len, by_basin.values())) == len(included)
    return audit, by_basin


def load_basin_environment(
    env_dir: Path,
    cfg: GenesisConfig,
) -> xr.Dataset:
    """Load small 1-degree *basin* samples from monthly 0.25-degree memmaps.

    Exact interpolation at half-degree cell centres: all selected coordinates
    fall on the 0.25-degree source grid; no global (time,lat,lon) stack is made.
    """
    path = Path(env_dir)
    lat0, lat1, lon0, lon1 = cfg.bounds
    nlat, nlon = int(lat1 - lat0), int(lon1 - lon0)
    lat2, lon2 = basin_cell_centers_genesis(cfg.basin, nlat, nlon)
    lats, lons = lat2[:, 0], lon2[0, :]
    ilat = np.rint((90.0 - lats) / 0.25).astype(int)
    ilon = np.rint((lons % 360.0) / 0.25).astype(int) % 1440
    dates = [
        pd.Timestamp(y, m, 1)
        for y in range(cfg.start_year, cfg.end_year + 1)
        for m in cfg.active_months
    ]
    vals = {v: np.empty((len(dates), nlat, nlon), dtype=np.float32) for v in ENV_SOURCE}
    for k, date in enumerate(dates):
        for var, stem in ENV_SOURCE.items():
            f = path / f"{stem}_{date.year}_{date.month}.npy"
            if not f.is_file():
                raise FileNotFoundError(f"Missing historical environmental field {f}")
            a = np.load(f, mmap_mode="r", allow_pickle=False)
            if a.shape != (721, 1440):
                raise ValueError(f"{f}: expected (721,1440), found {a.shape}")
            vals[var][k] = a[np.ix_(ilat, ilon)]
    ds = xr.Dataset(
        {
            v: (("valid_time", "latitude", "longitude"), data)
            for v, data in vals.items()
        },
        coords={"valid_time": dates, "latitude": lats, "longitude": lons},
    )
    f = 2 * OMEGA * np.sin(np.deg2rad(ds.latitude))
    # Cyclonic 850-hPa absolute vorticity: positive in BOTH hemispheres.
    # Keep unclipped here, then clip by the selected basin-specific threshold.
    hemisphere = 1.0 if cfg.basin in ("NA", "WP", "EP", "NI") else -1.0
    ds["eta850"] = (hemisphere * (ds["vort850"] + f)).clip(min=0)
    ds = ds.drop_vars("vort850")
    return ds


def load_basin_w500(
    path: Path, dates: list[pd.Timestamp], lat: np.ndarray, lon: np.ndarray
) -> xr.DataArray:
    """Extract W500 from the existing monthly GRIB/NetCDF product.

    Requires one actual month per valid_time; refuses duplicate monthly records,
    steps/levels or ambiguous forecast times instead of silently choosing one.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing W500 input: {p}")
    with xr.open_dataset(p) as ds:
        if "w500" in ds:
            da = ds["w500"]
        elif "w" in ds:
            da = ds["w"]
        else:
            raise KeyError(f"W500 file {p} lacks 'w' or 'w500' variable")
        if (
            "valid_time" in da.coords
            and "valid_time" not in da.dims
            and "time" in da.dims
        ):
            if da["valid_time"].dims != ("time",):
                raise ValueError(
                    "W500 valid_time is not 1D; select GRIB steps explicitly"
                )
            da = da.swap_dims({"time": "valid_time"})
        elif "valid_time" not in da.dims and "time" in da.dims:
            da = da.rename({"time": "valid_time"})
        if not {"valid_time", "latitude", "longitude"}.issubset(da.dims):
            raise ValueError(
                f"W500 dims {da.dims}: expect valid_time,latitude,longitude"
            )
        if len(da.dims) != 3:
            raise ValueError(
                f"W500 has extra dimensions {da.dims}; select one level/step first"
            )
        da = (
            da.assign_coords(longitude=(da.longitude % 360.0))
            .sortby("longitude")
            .sortby("latitude")
        )
        t = pd.DatetimeIndex(da.valid_time.values).to_period("M").to_timestamp()
        if t.has_duplicates:
            raise ValueError("W500 file has more than one field per month")
        da = da.assign_coords(valid_time=t)
        missing = sorted(set(dates) - set(t))
        if missing:
            raise ValueError(f"Missing W500 months (first 5): {missing[:5]}")

        # Select exact bracketing source coordinates even if the W500 grid is
        # irregular/coarse; fixed-degree margins can inadvertently omit them.
        def _index_window(values, targets):
            values = np.asarray(values)
            if not np.all(np.diff(values) > 0):
                raise ValueError("W500 coordinate must be strictly increasing")
            left = max(
                0, int(np.searchsorted(values, float(np.min(targets)), side="left")) - 1
            )
            right = min(
                len(values),
                int(np.searchsorted(values, float(np.max(targets)), side="right")) + 1,
            )
            return slice(left, right)

        lat_window = _index_window(da.latitude.values, lat)
        lon_window = _index_window(da.longitude.values, lon)
        da = da.sel(valid_time=dates).isel(latitude=lat_window, longitude=lon_window)
        if da.latitude.size < 2 or da.longitude.size < 2:
            raise ValueError("W500 grid does not bracket requested basin coordinates")
        out = da.interp(latitude=lat, longitude=lon).astype("float32").load()
    if not np.isfinite(out.values).all():
        raise ValueError("Missing or out-of-bounds W500 after interpolation")
    return out


def assign_region(basin: str, lat, lon) -> str:
    """Literal bucket rules from the *six original basin notebooks*.

    Preserve original fallback classes, including WP north-east label in the
    source (which is also used in its trained soft-region formula).
    """
    if basin == "NA":
        if 18 <= lat < 31 and 260 <= lon < 282:
            return "gulf"
        if 7 <= lat < 16 and 275 <= lon < 300:
            return "south_caribbean"
        if 16 <= lat < 25 and 275 <= lon < 300:
            return "north_caribbean"
        if lat < 20 and 300 <= lon < 350:
            return "mdr"
        return "subtropical"
    if basin == "WP":
        if 0 <= lat < 30 and 100 <= lon < 120:
            return "scs"
        if 17 <= lat < 40 and 120 <= lon < 140:
            return "nwp_nw"
        if 0 <= lat < 17 and 120 <= lon < 140:
            return "nwp_sw"
        if 17 <= lat < 40 and 140 <= lon < 180:
            return "nwp_se"
        return "nwp_ne"
    if basin == "EP":
        return "mdr" if 5 <= lat < 20 and 220 <= lon < 280 else "subtropical"
    if basin == "NI":
        return "arabian_sea" if 45 <= lon < 77 and 5 <= lat < 28 else "bay_of_bengal"
    if basin == "SI":
        if -35 <= lat < 5 and 25 <= lon < 60:
            return "southwest_indian_ocean"
        if -35 <= lat < 5 and 60 <= lon < 90:
            return "central_south_indian_ocean"
        return "northwest_australia"
    if basin == "SP":
        if -31 <= lat < -5 and 150 <= lon < 170:
            return "western_south_pacific"
        if -31 <= lat < -5 and 190 <= lon < 240:
            return "eastern_south_pacific"
        return "central_south_pacific"
    raise ValueError(basin)


def build_model_table(
    env: xr.Dataset,
    exposure: xr.Dataset,
    events: pd.DataFrame,
    cfg: GenesisConfig,
    *,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct monthly cell-level Poisson response and exposure with strict joins.

    Missing predictors on eligible genesis events raise. Missing predictors on
    unobserved cells are audited and dropped; their potential exposure is not
    silently treated as observed absence. No event is replicated or lost.
    """
    fields = ["pi", "rh600", "vws", "eta850"] 
    missing_fields = sorted(set(fields) - set(env.data_vars))
    if missing_fields:
        raise KeyError(f"{cfg.basin}: missing {missing_fields}")
    base = env[fields].to_dataframe().reset_index()
    base["time_month"] = pd.to_datetime(base.pop("valid_time"))
    base["year"] = base.time_month.dt.year
    base["month"] = base.time_month.dt.month
    la0, la1, lo0, lo1 = cfg.bounds
    base["ilat"] = np.rint(la1 - base.latitude.to_numpy() - 0.5).astype(int)
    base["ilon"] = np.rint(base.longitude.to_numpy() - lo0 - 0.5).astype(int)
    if base.duplicated(["time_month", "ilat", "ilon"]).any():
        raise AssertionError("Environmental monthly grid has duplicate cells")
    masks = (
        exposure[["ocean_valid", "eligible_fraction", "eligible_area_km2"]]
        .to_dataframe()
        .reset_index()
    )
    masks["ilat"] = np.rint(la1 - masks.latitude.to_numpy() - 0.5).astype(int)
    masks["ilon"] = np.rint(masks.longitude.to_numpy() - lo0 - 0.5).astype(int)
    masks = masks.drop(columns=["latitude", "longitude"])
    base = base.merge(
        masks, on=["ilat", "ilon"], validate="many_to_one", how="left", indicator=True
    )
    if (base.pop("_merge") != "both").any():
        raise AssertionError("Missing OSM mask row for an ERA5 grid cell")
    if not events.empty:
        if (
            events.storm_id.duplicated().any()
            or events.basin.nunique() != 1
            or events.basin.iloc[0] != cfg.basin
        ):
            raise AssertionError("Event table contains duplicates or wrong basin")
        keys = ["time_month", "ilat", "ilon"]
        counts = (
            events.groupby(keys, observed=True)
            .size()
            .rename("genesis_count")
            .reset_index()
        )
        # Assert all distinct observed cells exist exactly once in the ERA5 grid.
        lookup = base[keys].merge(
            counts, on=keys, how="right", validate="one_to_one", indicator=True
        )
        if (lookup["_merge"] != "both").any():
            raise AssertionError(
                f"{cfg.basin}: eligible events outside available ERA5 months/cells"
            )
    else:
        counts = pd.DataFrame(columns=["time_month", "ilat", "ilon", "genesis_count"])
    base = base.merge(
        counts, on=["time_month", "ilat", "ilon"], how="left", validate="one_to_one"
    )
    base["genesis_count"] = base.genesis_count.fillna(0).astype(int)
    base["region"] = [
        assign_region(cfg.basin, lat, lon)
        for lat, lon in zip(base.latitude, base.longitude)
    ]
    base["exposure"] = (
        base.eligible_area_km2
        / cfg.area_unit_km2
        * (base.time_month.dt.days_in_month / cfg.reference_month_days)
    )
    if extra is not None:
        if extra.duplicated(["year", "month"]).any():
            raise AssertionError("Duplicate climate index year-months")
        base = base.merge(
            extra, on=["year", "month"], how="left", validate="many_to_one"
        )
    eligible = base.ocean_valid.fillna(False) & base.exposure.gt(0)
    missing_predictor = base[fields].isna().any(axis=1) | ~np.isfinite(
        base[fields]
    ).all(axis=1)
    if cfg.basin == "NA":
        missing_predictor |= base["rsst"].isna()
    if cfg.basin == "EP":
        missing_predictor |= base["rsst_bis"].isna()
    if cfg.basin == "WP":
        missing_predictor |= base["eio_dynamic"].isna()
    if cfg.basin == "SP":
        missing_predictor |= base["roni_ond"].isna()
    lost = base.loc[eligible & missing_predictor, "genesis_count"].sum()
    if lost:
        examples = base.loc[
            eligible & missing_predictor & base.genesis_count.gt(0),
            ["time_month", "latitude", "longitude", "genesis_count"],
        ].head(10)
        raise AssertionError(
            f"{cfg.basin}: {lost} eligible genesis events have missing predictors:\n{examples}"
        )
    base.attrs["n_rows_ineligible_geography"] = int((~eligible).sum())
    base.attrs["n_rows_missing_covariates"] = int((eligible & missing_predictor).sum())
    good = eligible & ~missing_predictor
    model = base.loc[good].copy()
    model.attrs.update(base.attrs)
    model["log_exposure"] = np.log(model.exposure)
    expected_n = len(events)
    if model.genesis_count.sum() != expected_n:
        raise AssertionError(
            f"{cfg.basin}: expected {expected_n} eligible events, table counts {model.genesis_count.sum()}"
        )
    if model.duplicated(["time_month", "ilat", "ilon"]).any():
        raise AssertionError(f"{cfg.basin}: duplicate modeling rows")
    return model


def smooth_region_weights(
    model: pd.DataFrame, exposure: xr.Dataset, cfg: GenesisConfig, sigma: float = 3.0
) -> tuple[pd.DataFrame, list[str]]:
    """Fixed geographic soft memberships on full 1-degree mask, never from test events."""
    lat, lon = np.meshgrid(
        exposure.latitude.values, exposure.longitude.values, indexing="ij"
    )
    eligible = exposure["eligible_area_km2"].values > 0
    names = sorted(
        set(
            assign_region(cfg.basin, a, b) for a, b in zip(lat[eligible], lon[eligible])
        )
    )
    if REGION_REFERENCE[cfg.basin] not in names:
        raise AssertionError(f"Missing reference region for {cfg.basin}")
    region = np.empty(lat.shape, dtype=object)
    for idx in np.ndindex(lat.shape):
        region[idx] = assign_region(cfg.basin, lat[idx], lon[idx])
    fields = {
        name: gaussian_filter(
            ((region == name) & eligible).astype(float), sigma=sigma, mode="nearest"
        )
        for name in names
    }
    total = sum(fields.values())
    if np.any((total == 0) & eligible):
        raise AssertionError("Zero Gaussian weight on an eligible cell")
    total = np.where(total > 0, total, 1.0)
    rows = {"latitude": lat.ravel(), "longitude": lon.ravel()}
    for name, f in fields.items():
        rows[f"w_{name}"] = (f / total).ravel()
    w = pd.DataFrame(rows)
    out = model.merge(
        w, on=["latitude", "longitude"], validate="many_to_one", how="left"
    )
    if out.filter(regex="^w_").isna().any().any():
        raise AssertionError("Missing geographic soft weights")
    return out, names


def standardize_train_test(
    table: pd.DataFrame, cfg: GenesisConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Transform with training-year moments only; no holdout leakage."""
    train = table.loc[table.year < cfg.first_test_year].copy()
    test = table.loc[table.year >= cfg.first_test_year].copy()
    if train.empty or test.empty:
        raise AssertionError(f"{cfg.basin}: empty training or test period")
    raw_features = (
        ["pi", "rh600", "vws", "w500"] if cfg.basin != "EP" else ["pi", "rh600", "vws"]
    )
    train["eta_clip"] = train.eta850.clip(upper=ETA_CLIP[cfg.basin])
    test["eta_clip"] = test.eta850.clip(upper=ETA_CLIP[cfg.basin])
    raw_features.append("eta_clip")
    if cfg.basin == "NA":
        raw_features.append("rsst")
    if cfg.basin == "EP":
        raw_features.append("rsst_bis")
    if cfg.basin == "WP":
        raw_features.append("eio_dynamic")
    scale = {}
    for f in raw_features:
        mu, sd = train[f].mean(), train[f].std(ddof=1)
        if not (np.isfinite(mu) and np.isfinite(sd) and sd > 0):
            raise ValueError(
                f"{cfg.basin}: invalid training moments for {f}: {mu}, {sd}"
            )
        scale[f] = (float(mu), float(sd))
        # Preserve the variable names in the selected formulas.
        col = {
            "rsst": "rsst",
            "rsst_bis": "rsst_bis",
            "eio_dynamic": "z_eio_dynamic",
        }.get(f, f"z_{f}")
        train[col] = (train[f] - mu) / sd
        test[col] = (test[f] - mu) / sd
    if cfg.basin == "NI":

        def ni_season(m):
            if m in (3, 4, 5, 6):
                return "pre_monsoon"
            if m in (7, 8, 9):
                return "monsoon"
            return "post_monsoon"

        train["ni_season"] = train.month.map(ni_season)
        test["ni_season"] = test.month.map(ni_season)
    if cfg.basin == "SP":
        # Exactly the original RONI per-SH-season de-duplication convention.
        seasons = (
            train[["season_year", "roni_ond"]].drop_duplicates("season_year").dropna()
        )
        mu, sd = seasons.roni_ond.mean(), seasons.roni_ond.std(ddof=1)
        if not (np.isfinite(sd) and sd > 0):
            raise ValueError("RONI training variance invalid")
        scale["roni_ond_by_season"] = (float(mu), float(sd))
        for t in (train, test):
            t["roni_ond_standardized"] = (t.roni_ond - mu) / sd
    return train, test, scale


def _weighted_sst_mean(da: xr.DataArray) -> float:
    if not bool(da.notnull().any()):
        raise ValueError("SST subset has no ocean observations")
    return float(
        da.weighted(np.cos(np.deg2rad(da.latitude))).mean(
            ("latitude", "longitude"), skipna=True
        )
    )


def _crop_sst(
    da: xr.DataArray, *, south: float, north: float, west: float, east: float
):
    """Monthly source is 721 x 1440 in descending-latitude, 0-360 lon order."""
    return da.sel(latitude=slice(north, south), longitude=slice(west, east))


def yearly_relative_sst(env_dir: Path, cfg: GenesisConfig) -> pd.DataFrame:
    """NA / EP retained JASO indices from original basin notebooks.

    rsst = MDR mean - tropical mean
    rsst_bis = MDR mean - area-weighted warmest-30%-of-tropics mean.
    EP uses rsst_bis, NA uses rsst. Compute full active-month SST values
    with day-weighted annual averaging exactly as in the source notebooks.
    """
    if cfg.basin not in ("NA", "EP"):
        raise ValueError("rsst indices only used by NA and EP")
    mdr = (10, 25, 280, 340) if cfg.basin == "NA" else (5, 20, 220, 270)
    records = []
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    for year in range(cfg.start_year, cfg.end_year + 1):
        for month in (7, 8, 9, 10):
            path = Path(env_dir) / f"SST_{year}_{month}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing SST month {path}")
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.shape != (721, 1440):
                raise ValueError(f"Bad SST array geometry: {path}")
            da = xr.DataArray(
                a,
                dims=("latitude", "longitude"),
                coords={"latitude": lat, "longitude": lon},
            )
            tropics = _crop_sst(da, south=-30, north=30, west=0, east=359.75)
            mdr_sst = _weighted_sst_mean(
                _crop_sst(da, south=mdr[0], north=mdr[1], west=mdr[2], east=mdr[3])
            )
            tropics_mean = _weighted_sst_mean(tropics)
            w = np.cos(np.deg2rad(tropics.latitude))
            threshold = float(
                tropics.weighted(w).quantile(
                    0.70, dim=("latitude", "longitude"), skipna=True
                )
            )
            warm_mean = _weighted_sst_mean(tropics.where(tropics >= threshold))
            records.append(
                (
                    year,
                    month,
                    calendar.monthrange(year, month)[1],
                    mdr_sst - tropics_mean,
                    mdr_sst - warm_mean,
                )
            )
    d = pd.DataFrame(records, columns=["year", "month", "days", "rsst", "rsst_bis"])
    yearly = (
        d.groupby("year")
        .apply(
            lambda x: pd.Series(
                {k: np.average(x[k], weights=x.days) for k in ("rsst", "rsst_bis")}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return yearly


def na_ep_climate_monthly(yearly_sst: pd.DataFrame, cfg: GenesisConfig) -> pd.DataFrame:
    idx = yearly_sst.set_index("year")
    if idx.index.has_duplicates:
        raise AssertionError("Duplicate SST years")
    # Monthly repetition of a *seasonal* covariate is deliberate, as in notebooks.
    return pd.DataFrame(
        [
            (y, m, float(idx.loc[y, "rsst"]), float(idx.loc[y, "rsst_bis"]))
            for y in range(cfg.start_year, cfg.end_year + 1)
            for m in cfg.active_months
        ],
        columns=["year", "month", "rsst", "rsst_bis"],
    )


def wp_eio_dynamic(sst_history_path: Path, cfg: GenesisConfig) -> pd.DataFrame:
    """WP selected EIO: eastern Indian Ocean SST monthly anomaly, 3mo trailing.

    Correct original test-year leakage: month climatology fitted on training
    years only; a given month's 3-month average uses only current/past SST.
    """
    path = Path(sst_history_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing WP SST history: {path}")
    with xr.open_dataset(path) as ds:
        if "sst" not in ds:
            raise KeyError("SST history must contain 'sst'")
        da = (
            ds["sst"]
            .assign_coords(longitude=ds.longitude % 360.0)
            .sortby("longitude")
            .sortby("latitude")
        )
        if "valid_time" not in da.dims:
            if "time" in da.dims:
                da = da.rename({"time": "valid_time"})
            else:
                raise ValueError("SST history needs valid_time or time dimension")
        da = da.sel(
            valid_time=slice(f"{cfg.start_year - 1}-01-01", f"{cfg.end_year}-12-31")
        )
        sub = da.sel(latitude=slice(-10, 22.5), longitude=slice(75, 100))
        monthly = (
            sub.weighted(np.cos(np.deg2rad(sub.latitude)))
            .mean(("latitude", "longitude"), skipna=True)
            .load()
            .to_series()
            .sort_index()
        )
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp()
    if monthly.index.has_duplicates:
        raise ValueError("Duplicate WP EIO SST months")
    monthly = monthly.asfreq("MS")
    base = monthly[
        (monthly.index.year >= cfg.start_year)
        & (monthly.index.year < cfg.first_test_year)
    ]
    if base.index.year.nunique() < 10:
        raise ValueError("Insufficient SST climatology years")
    means = base.groupby(base.index.month).mean()
    anom = monthly - monthly.index.month.map(means).to_numpy()
    trailing = anom.rolling(3, min_periods=3).mean()
    extra = (
        trailing.rename("eio_dynamic")
        .reset_index()
        .rename(columns={"index": "time_month", "valid_time": "time_month"})
    )
    extra["year"] = extra.time_month.dt.year
    extra["month"] = extra.time_month.dt.month
    return extra[["year", "month", "eio_dynamic"]]


def sp_roni_monthly(roni_csv: Path, cfg: GenesisConfig) -> pd.DataFrame:
    """Preserve selected SP model's season-year OND RONI interaction."""
    path = Path(roni_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Missing SP RONI file: {path}")
    roni = pd.read_csv(path)
    if not {"SEAS", "YR", "ANOM"}.issubset(roni.columns):
        raise ValueError("RONI CSV must provide SEAS,YR,ANOM")
    series = roni.loc[roni.SEAS.astype(str).str.upper().eq("OND")].set_index("YR")[
        "ANOM"
    ]
    if series.index.has_duplicates:
        raise ValueError("Duplicate OND RONI years")
    rows = [
        (y, m, y if m >= 9 else y - 1, series.get(y if m >= 9 else y - 1, np.nan))
        for y in range(cfg.start_year, cfg.end_year + 1)
        for m in cfg.active_months
    ]
    d = pd.DataFrame(rows, columns=["year", "month", "season_year", "roni_ond"])
    if d.roni_ond.isna().any():
        missing = d.loc[d.roni_ond.isna(), "season_year"].unique()
        raise ValueError(f"Missing OND RONI seasonal years: {missing.tolist()}")
    return d


def predict_cell_counts(fitted_model, frame: pd.DataFrame) -> pd.Series:
    """Expected genesis events per cell-month, including the Poisson exposure.

    statsmodels does NOT reuse the fitted offset when predicting on a new DataFrame.
    Omitting it inflates ~0.1-exposure 1-degree cells by about 10-fold.
    """
    if not frame.index.is_unique:
        raise ValueError("Prediction table index must be unique")
    if "log_exposure" not in frame or "exposure" not in frame:
        raise KeyError("Prediction requires both exposure and log_exposure")
    exposure = frame["exposure"].to_numpy(dtype=float)
    log_exposure = frame["log_exposure"].to_numpy(dtype=float)
    if not np.all(np.isfinite(exposure) & (exposure > 0)):
        raise ValueError("Prediction exposure must be finite and positive")
    if not np.allclose(np.log(exposure), log_exposure, atol=1e-10, rtol=1e-10):
        raise ValueError("Inconsistent log_exposure and exposure")
    result = fitted_model.predict(frame, offset=frame["log_exposure"])
    arr = np.asarray(result, dtype=float)
    if arr.shape != (len(frame),) or not np.all(np.isfinite(arr) & (arr >= 0)):
        raise ValueError("Invalid fitted Poisson predicted counts")
    return pd.Series(arr, index=frame.index, name="predicted_genesis")


def seasonal_count_diagnostics(
    fitted_model, train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Diagnostics of training calibration and held-out annual total rates.

    Model count is defined only over the basin's fitted active months, eligible
    OSM ocean cells, and months with complete predictor coverage.
    """
    result = {}
    for label, table in (("train", train), ("test", test)):
        mu = predict_cell_counts(fitted_model, table)
        summary = (
            pd.DataFrame(
                {
                    "year": table["year"],
                    "month": table["month"],
                    "observed": table["genesis_count"].astype(float),
                    "predicted": mu,
                }
            )
            .groupby(["year", "month"], as_index=False)[["observed", "predicted"]]
            .sum()
        )
        result[label] = summary
    # With an intercept and an ordinary unpenalized Poisson MLE, the total
    # predicted training count must be approximately the observed total.
    train_sum = result["train"][["observed", "predicted"]].sum()
    if abs(train_sum["observed"] - train_sum["predicted"]) > max(
        1e-3, 1e-5 * train_sum["observed"]
    ):
        raise AssertionError(
            f"Unexpected training-frequency calibration: {train_sum.to_dict()}"
        )
    return result
