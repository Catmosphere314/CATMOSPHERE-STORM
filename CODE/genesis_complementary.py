"""Scalar, environmental-year-based GLM covariates for CATMOSPHERE-STORM.

Stored raw values (NOT standardized):
  RSST_NA_<env_year>_<month>.npy
  RSST_BIS_EP_<env_year>_<month>.npy
  EIO_DYNAMIC_WP_<env_year>_<month>.npy
  RONI_OND_SP_<env_year>_<month>.npy
Each has an adjacent .json provenance record. The sampler applies the training
scaler from genesis_prepared/metadata_<basin>.joblib exactly once.
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from CODE import genesis_dataset as gd

NAMES = {
    "NA": "RSST_NA",
    "EP": "RSST_BIS_EP",
    "WP": "EIO_DYNAMIC_WP",
    "SP": "RONI_OND_SP",
}


def feature_path(root, basin, year, month):
    return (
        Path(root)
        / "complementary_features_env_years"
        / f"{NAMES[basin]}_{int(year)}_{int(month)}.npy"
    )


def write_feature(root, basin, year, month, value, *, status, input_months, definition):
    f = feature_path(root, basin, year, month)
    f.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(float(value)):
        raise ValueError(f"Non-finite {basin} complementary feature")
    np.save(f, np.array(float(value), dtype=np.float64))
    f.with_suffix(".json").write_text(
        json.dumps(
            {
                "basin": basin,
                "env_year": int(year),
                "month": int(month),
                "value_kind": "raw_before_training_standardization",
                "status": status,
                "input_months": list(input_months),
                "definition": definition,
            },
            indent=2,
        )
    )
    return f


def eio_monthly_reference(sst_history, start_year, end_exclusive):
    """WP reference climatology: historical EIO SST 10S..22.5N, 75..100E.

    Must be fitted on exactly the years preceding cfg.first_test_year, as in
    genesis_dataset.wp_eio_dynamic; preserve its absolute SST baseline.
    """
    with xr.open_dataset(sst_history) as ds:
        da = (
            ds["sst"]
            .assign_coords(longitude=ds.longitude % 360)
            .sortby("longitude")
            .sortby("latitude")
        )
        if "valid_time" not in da.dims:
            da = da.rename({"time": "valid_time"})
        da = da.sel(
            valid_time=slice(f"{start_year}-01-01", f"{end_exclusive - 1}-12-31")
        )
        sub = da.sel(latitude=slice(-10, 22.5), longitude=slice(75, 100))
        ts = (
            sub.weighted(np.cos(np.deg2rad(sub.latitude)))
            .mean(("latitude", "longitude"), skipna=True)
            .load()
            .to_series()
        )
    ts.index = pd.DatetimeIndex(ts.index).to_period("M").to_timestamp()
    if ts.index.year.nunique() < 10:
        raise ValueError("Insufficient WP reference years")
    means = ts.groupby(ts.index.month).mean()
    if len(means) != 12 or means.isna().any():
        raise ValueError("Incomplete WP monthly reference")
    return {int(month): float(means[month]) for month in range(1, 13)}


def compute_rsst_month(sst_field, basin):
    """Use the SAME regional means and weighted tropical quantile as the fit."""
    if basin not in ("NA", "EP"):
        raise ValueError(basin)
    a = np.asarray(sst_field)
    if a.shape != (721, 1440):
        raise ValueError(f"SST grid must be (721,1440), got {a.shape}")
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(1440) * 0.25
    da = xr.DataArray(
        a, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon}
    )
    mdr = (10, 25, 280, 340) if basin == "NA" else (5, 20, 220, 270)
    tropics = gd._crop_sst(da, south=-30, north=30, west=0, east=359.75)
    mdr_sst = gd._weighted_sst_mean(
        gd._crop_sst(da, south=mdr[0], north=mdr[1], west=mdr[2], east=mdr[3])
    )
    if basin == "NA":
        return mdr_sst - gd._weighted_sst_mean(tropics)
    weights = np.cos(np.deg2rad(tropics.latitude))
    threshold = float(
        tropics.weighted(weights).quantile(
            0.70, dim=("latitude", "longitude"), skipna=True
        )
    )
    return mdr_sst - gd._weighted_sst_mean(tropics.where(tropics >= threshold))


def compute_eio_sst(sst_field):
    a = np.asarray(sst_field)
    if a.shape != (721, 1440):
        raise ValueError(f"SST grid must be (721,1440), got {a.shape}")
    lats = np.linspace(90, -90, 721)
    lons = np.arange(1440) * 0.25
    ii = (lats >= -10) & (lats <= 22.5)
    jj = (lons >= 75) & (lons <= 100)
    sub = a[np.ix_(ii, jj)]
    w = np.broadcast_to(np.cos(np.deg2rad(lats[ii]))[:, None], sub.shape)
    mask = np.isfinite(sub)
    if not np.any(mask):
        raise ValueError("EIO SST all missing")
    return float(np.sum(sub[mask] * w[mask]) / np.sum(w[mask]))


def build_historical_features(
    root, *, period, basins=("NA", "EP", "WP", "SP"), first_test_year=None
):
    """Run after MASTER_climatology saved ERA5 env_yearly fields.

    Mirrors the existing notebook's *complete historical* feature definitions;
    does not require a fitted GLM for historical files.
    """
    root = Path(root)
    start, end = map(int, period)
    cutoff = int(first_test_year if first_test_year is not None else end + 1)
    for basin in basins:
        cfg = gd.GenesisConfig(
            basin, start_year=start, end_year=end, first_test_year=cutoff
        )
        if basin in ("NA", "EP"):
            seasonal = gd.yearly_relative_sst(root / "env_yearly", cfg).set_index(
                "year"
            )
            key = "rsst" if basin == "NA" else "rsst_bis"
            for year in range(start, end + 1):
                value = float(seasonal.loc[year, key])
                for month in cfg.active_months:
                    write_feature(
                        root,
                        basin,
                        year,
                        month,
                        value,
                        status="complete_JASO",
                        input_months=[7, 8, 9, 10],
                        definition="day_weighted_JASO_annual_index",
                    )
        elif basin == "WP":
            extra = gd.wp_eio_dynamic(root / "Monthly_mean_SST.nc", cfg)
            for row in extra.itertuples():
                if (
                    start <= row.year <= end
                    and row.month in cfg.active_months
                    and np.isfinite(row.eio_dynamic)
                ):
                    write_feature(
                        root,
                        basin,
                        row.year,
                        row.month,
                        row.eio_dynamic,
                        status="complete_trailing_3_month",
                        input_months=[],
                        definition="EIO_SST_anomaly_3_month_trailing",
                    )
        elif basin == "SP":
            extra = gd.sp_roni_monthly(root / "roni.csv", cfg)
            for row in extra.itertuples():
                write_feature(
                    root,
                    basin,
                    row.year,
                    row.month,
                    row.roni_ond,
                    status="observed_OND_RONI",
                    input_months=[10, 11, 12],
                    definition=f"OND_RONI_season_year_{row.season_year}",
                )
    return root / "complementary_features_env_years"


def build_forecast_features(
    root, *, init_date, lead_months, env_year, basins=("NA", "EP", "WP", "SP")
):
    """Prepare complementary scalar files for the forecast member's monthly fields.

    JASO is annual in training. When JAS is available, compute a labelled
    *partial JAS* proxy; with fewer than three months use raw training mean so
    the standardized predictor is exactly zero. No partial proxy is called JASO.

    WP trailing EIO uses this member's three consecutive months, or available
    observed base-year fields for preceding months; otherwise the train-mean
    neutral value. SP forecast RONI uses train-mean neutral until a rigorously
    comparable SEAS5 RONI implementation is available.
    """
    root = Path(root)
    base_year = int(str(init_date)[:4])
    init_month = int(str(init_date)[5:7])
    leads = [
        pd.Timestamp(base_year, init_month, 1) + pd.DateOffset(months=i)
        for i in range(int(lead_months))
    ]
    env_dir = root / "env_yearly"
    for basin in basins:
        meta_path = root / "genesis_prepared" / f"metadata_{basin}.joblib"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"Forecast complementary features require a fitted model: {meta_path}"
            )
        meta = joblib.load(meta_path)
        scalers = meta["scalers"]
        if basin in ("NA", "EP"):
            key = "rsst" if basin == "NA" else "rsst_bis"
            # Evaluate JASO separately for each calendar year, even at year boundaries.
            for year in sorted({int(x.year) for x in leads}):
                found = {}
                for mo in (7, 8, 9, 10):
                    f = env_dir / f"SST_{env_year}_{mo}.npy"
                    if f.exists() and pd.Timestamp(year, mo, 1) in leads:
                        found[mo] = compute_rsst_month(np.load(f, mmap_mode="r"), basin)
                if len(found) >= 3 and set((7, 8, 9)).issubset(found):
                    selected = [7, 8, 9, 10] if 10 in found else [7, 8, 9]
                    vals = np.asarray([found[m] for m in selected])
                    days = np.asarray(
                        [calendar.monthrange(year, m)[1] for m in selected]
                    )
                    raw = float(np.average(vals, weights=days))
                    status = (
                        "complete_JASO" if len(selected) == 4 else "partial_JAS_proxy"
                    )
                else:
                    selected = sorted(found)
                    raw = float(scalers[key][0])
                    status = "neutral_training_mean_insufficient_JAS"
                for t in leads:
                    if int(t.year) == year:
                        write_feature(
                            root,
                            basin,
                            env_year,
                            t.month,
                            raw,
                            status=status,
                            input_months=selected,
                            definition="JASO_training_index_or_explicit_partial_proxy",
                        )
        elif basin == "WP":
            clim = meta.get("eio_monthly_climatology")
            if clim is None:
                raise ValueError(
                    "WP fitted metadata missing eio_monthly_climatology; refit GLM"
                )
            lead_keys = {(int(t.year), int(t.month)) for t in leads}
            for t in leads:
                observed = []
                anomalies = []
                for step in (2, 1, 0):
                    prev = t - pd.DateOffset(months=step)
                    use_year = (
                        env_year
                        if (prev.year, prev.month) in lead_keys
                        else int(prev.year)
                    )
                    p = env_dir / f"SST_{use_year}_{prev.month}.npy"
                    if p.is_file():
                        anomalies.append(
                            compute_eio_sst(np.load(p, mmap_mode="r"))
                            - float(clim[int(prev.month)])
                        )
                        observed.append(int(prev.month))
                if len(anomalies) == 3:
                    value = float(np.mean(anomalies))
                    status = "complete_trailing_3_month"
                else:
                    value = float(scalers["eio_dynamic"][0])
                    status = "neutral_training_mean_missing_EIO_history"
                write_feature(
                    root,
                    basin,
                    env_year,
                    t.month,
                    value,
                    status=status,
                    input_months=observed,
                    definition="EIO_SST_anomaly_3_month_trailing",
                )
        elif basin == "SP":
            for t in leads:
                write_feature(
                    root,
                    basin,
                    env_year,
                    t.month,
                    float(scalers["roni_ond_by_season"][0]),
                    status="neutral_training_mean_no_forecast_RONI",
                    input_months=[],
                    definition="OND_RONI_not_reconstructed_from_SEAS5",
                )
    return root / "complementary_features_env_years"
