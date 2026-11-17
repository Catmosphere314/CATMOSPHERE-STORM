"""Decay pipeline to apply it to tropical cyclone tracks."""

import xarray as xr
import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.optimize import least_squares
import json
from pathlib import Path

from CODE.land_fraction import get_land_fraction, LAND_FRACTION_PATH
from CODE.geography import is_land, LAND_MASK_PATH, geography_provenance
from CODE.preprocessing import convert_wind_speed
from CODE.siena_utils import TS_THRESHOLD_MS


__location__ = os.getcwd()


WEIGHTS_PRESENT_FUTURE = [0.2, 0.8]  # calibration and simulator must match
DEFAULT_DECAY_PARAMETERS_PATH = (
    Path(__file__).resolve().parents[1] / "DATA" / "parameters" / "storm_l_decay.json"
)


def save_decay_parameters(parameters, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 2,
        "land_fraction_weights": WEIGHTS_PRESENT_FUTURE,
        "radius_km": 100,
        "land_fraction_raster": LAND_FRACTION_PATH.name,
        "binary_land_mask": LAND_MASK_PATH.name,
        "geography": geography_provenance(),
        "basins": parameters,
    }

    with filepath.open("w") as f:
        json.dump(payload, f, indent=2)


def load_decay_parameters(filepath):
    with Path(filepath).open("r") as f:
        payload = json.load(f)

    if payload["land_fraction_weights"] != WEIGHTS_PRESENT_FUTURE:
        raise ValueError("Land-fraction weighting differs from calibration")
    if payload.get("radius_km") != 100:
        raise ValueError("STORM-L parameters expect a 100 km land-fraction radius")
    if payload.get("land_fraction_raster") != LAND_FRACTION_PATH.name:
        raise ValueError("Calibration and simulator land-fraction raster paths differ")
    if payload.get("binary_land_mask") != LAND_MASK_PATH.name:
        raise ValueError("Calibration and simulator binary land-mask paths differ")
    if payload.get("schema_version") != 2:
        raise ValueError("Old STORM-L coefficients: refit with OSM geography")
    if payload.get("geography") != geography_provenance():
        raise ValueError(
            "Coastline snapshot or raster configuration differs from fitted coefficients"
        )

    return payload["basins"]


def _prepare_decay_df(landfall_df, max_gap_hours=6):
    """Convert landfall episodes into a DataFrame suitable for decay model calibration."""

    df = landfall_df.copy()

    # One storm may have multiple landfall episodes
    df = df.sort_values(["landfall_id", "time"]).reset_index(drop=True)

    grouped = df.groupby("landfall_id", sort=False)

    # Create destination values for each interval
    columns = [
        "time",
        "wind_ms",
        "land_fraction",
        "on_land",
        "lat",
        "lon",
    ]

    for col in columns:
        df[f"{col}_next"] = grouped[col].shift(-1)

    # Actual time interval in hours
    df["dt_hours"] = (df["time_next"] - df["time"]).dt.total_seconds() / 3600

    # Keep only valid consecutive observations
    df = df[df["dt_hours"].between(0.0, max_gap_hours, inclusive="right")].copy()

    # Current and destination land fractions
    L_t = df["land_fraction"].to_numpy(dtype=float)
    L_next = df["land_fraction_next"].to_numpy(dtype=float)

    # Require valid fractional exposure values
    valid_land = (
        np.isfinite(L_t)
        & np.isfinite(L_next)
        & (L_t >= 0)
        & (L_t <= 1)
        & (L_next >= 0)
        & (L_next <= 1)
    )

    df = df.loc[valid_land].copy()

    # Re-extract arrays after filtering
    L_t = df["land_fraction"].to_numpy(dtype=float)
    L_next = df["land_fraction_next"].to_numpy(dtype=float)

    w_t = WEIGHTS_PRESENT_FUTURE[0] * L_t
    w_next = WEIGHTS_PRESENT_FUTURE[1] * L_next

    df["Lbar"] = w_t + w_next

    # Retain transitions involving a land-covered center
    # and with positive effective land exposure
    df = df[(df["Lbar"] > 0) & (df["on_land"] | df["on_land_next"])].copy()

    # Observed wind change across the interval
    df["delta_wind"] = df["wind_ms_next"] - df["wind_ms"]

    # Complete calibration observations only
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=[
            "wind_ms",
            "wind_ms_next",
            "Lbar",
            "delta_wind",
        ]
    )

    return df.reset_index(drop=True)


def _haversine_km(lat0, lon0, lat, lon):
    p1, p2 = np.radians(lat0), np.radians(lat)
    dp = p2 - p1
    dl = np.radians((np.asarray(lon) - lon0 + 180) % 360 - 180)

    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _decode_agency(value):
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def extract_landfall_episodes(ibtracs, min_steps=2, max_hours=48):

    df1 = ibtracs.isel(quadrant=0).to_dataframe()[
        [
            "lat",
            "lon",
            "sid",
            "season",
            "number",
            "basin",
            "subbasin",
            "name",
            "iso_time",
            "wmo_wind",
            "wmo_agency",
            "wmo_pres",
            "track_type",
            "main_track_sid",
            "dist2land",
            "landfall",
            "usa_sshs",
            "usa_rmw",
        ]
    ]
    df1["wind_ms"] = [
        convert_wind_speed(w * 0.51444444, _decode_agency(a))
        for w, a in zip(df1["wmo_wind"], df1["wmo_agency"])
    ]

    df = df1.loc[df1["wind_ms"] >= TS_THRESHOLD_MS].copy()

    # Decode byte strings
    def decode_string(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore").strip()
        return str(x).strip()

    df["sid"] = df["sid"].map(decode_string)
    df["name"] = df["name"].map(decode_string)
    df["basin"] = df["basin"].map(decode_string)
    df["track_type"] = df["track_type"].map(decode_string)
    df["time"] = pd.to_datetime(
        df["iso_time"].map(decode_string),
        errors="coerce",
    )

    # Keep main storm tracks
    df = df[df["track_type"] == "main"].copy()

    df = df.dropna(subset=["time", "lat", "lon"])
    df = df.sort_values(["sid", "time"])

    segments = []

    for sid, storm in df.groupby("sid", sort=False):
        storm = storm.reset_index(drop=True)

        on_land = np.asarray(
            is_land(storm["lat"].to_numpy(), storm["lon"].to_numpy()), dtype=bool
        )

        # Indices at which storm center first moves over land
        starts = np.flatnonzero(on_land & ~np.r_[False, on_land[:-1]])

        for start in starts:
            # Need an offshore reference observation
            if start == 0:
                continue

            previous = storm.iloc[start - 1]
            first_land = storm.iloc[start]

            dt = first_land["time"] - previous["time"]

            if dt > pd.Timedelta(hours=6):
                continue

            # Minimum tropical-storm intensity at landfall
            v0 = previous["wind_ms"]

            if not np.isfinite(v0) or v0 < TS_THRESHOLD_MS:
                continue

            # Continue until the storm leaves land
            # or observations are separated by a long gap
            end = start

            while end < len(storm) and on_land[end]:
                if end > start:
                    gap = storm.loc[end, "time"] - storm.loc[end - 1, "time"]

                    if gap > pd.Timedelta(hours=6):
                        break

                end += 1

            # Include the last offshore observation
            segment = storm.iloc[start - 1 : end].copy()

            # OpenStreetMap classification, consistent with genesis and track.
            segment["on_land"] = on_land[start - 1 : end]

            # Require at least 3 measured inland winds
            if segment.loc[segment["on_land"], "wind_ms"].notna().sum() < 3:
                continue

            # Approximate crossing time as the midpoint
            # between last offshore and first inland points
            landfall_time = previous["time"] + dt / 2

            segment["hours_since_landfall"] = (
                segment["time"] - landfall_time
            ).dt.total_seconds() / 3600

            segment["landfall_id"] = f"{sid}_{start}"

            # Diagnostic: does IBTrACS confirm the crossing?
            segment["landfall_flag"] = previous["landfall"] == 0

            segments.append(segment)

    if not segments:
        return []

    landfall_df = pd.concat(segments, ignore_index=True)

    landfall_df["land_fraction"] = get_land_fraction(
        landfall_df["lat"].to_numpy(),
        landfall_df["lon"].to_numpy(),
    )

    first_land = (
        landfall_df.loc[landfall_df["on_land"].eq(True)]
        .groupby("landfall_id")["time"]
        .min()
    )

    decay_df = _prepare_decay_df(landfall_df)

    df = decay_df.sort_values(["landfall_id", "time"])

    episodes = []

    for eid, g in df.groupby("landfall_id", sort=False):
        g = g.reset_index(drop=True)

        # Require observations starting at first inland point
        if eid not in first_land.index:
            continue
        # if g.loc[0, "time"] != first_land.loc[eid]:
        #     continue
        # First transition must be offshore -> inland
        if g.loc[0, "time_next"] != first_land.loc[eid]:
            continue

        if bool(g.loc[0, "on_land"]):
            continue

        if not bool(g.loc[0, "on_land_next"]):
            continue

        # Do not bridge missing observation intervals
        gaps = np.flatnonzero(
            g["time"].to_numpy()[1:] != g["time_next"].to_numpy()[:-1]
        )

        if len(gaps):
            g = g.iloc[: gaps[0] + 1]

        g = g.loc[g["time_next"] <= g.iloc[0]["time"] + pd.Timedelta(hours=max_hours)]

        if len(g) < min_steps:
            continue

        lat0 = float(g.iloc[0]["lat"])
        lon0 = float(g.iloc[0]["lon"])

        ep = {
            "id": eid,
            "sid": g.iloc[0]["sid"],
            "basin": g.iloc[0]["basin"],
            "v0": float(g.iloc[0]["wind_ms"]),
            "y": g["wind_ms_next"].to_numpy(float),
            "L": g["Lbar"].to_numpy(float),
            "dt": g["dt_hours"].to_numpy(float),
            "d": _haversine_km(
                lat0,
                lon0,
                g["lat_next"].to_numpy(float),
                g["lon_next"].to_numpy(float),
            ),
        }

        ep["lead"] = np.cumsum(ep["dt"])

        if np.all(np.isfinite(ep["y"])):
            episodes.append(ep)

    return episodes


def predict_storm_l(v0_ms, T_l_hours, D_km, params):
    """Predict STORM-L wind intensity, returning m/s."""

    kt_to_ms = 0.5144444444
    vb = 26.7  # knots

    T = np.asarray(T_l_hours, dtype=float)
    D = np.asarray(D_km, dtype=float)

    v0 = np.asarray(v0_ms, dtype=float) / kt_to_ms

    alpha = params["alpha"]
    r = params["R"]
    c1 = params["C1"]
    d1 = params["D1"]
    t0 = params["t0_hours"]

    correction = T * (t0 - T) * (c1 * np.log(np.maximum(D, 1.0)) + d1)

    v = vb + (r * v0 - vb) * np.exp(-alpha * T) - correction

    return v * kt_to_ms


def predict_storm_landfrac(
    episode, alpha=0.095, t0=100.0, r=0.9, c1=0.0109, d1=-0.0503
):
    params = {
        "alpha": alpha,
        "R": r,
        "C1": c1,
        "D1": d1,
        "t0_hours": t0,
    }

    T_l = np.cumsum(episode["dt"] * episode["L"])

    return predict_storm_l(
        episode["v0"],
        T_l,
        episode["d"],
        params,
    )


def fit_STORM_landfrac(train: pd.DataFrame):
    """Training the STORM-L decay model to landfall episodes using least-squares optimization."""

    n_episodes = Counter(episode["sid"] for episode in train)

    def residuals(params):

        alpha = np.exp(params[0])
        r = params[1]

        # Rescaled optimization parameters
        c1 = params[2] * 0.01
        d1 = params[3] * 0.05

        return np.concatenate(
            [
                (
                    predict_storm_landfrac(
                        episode,
                        alpha=alpha,
                        r=r,
                        c1=c1,
                        d1=d1,
                    )
                    - episode["y"]
                )
                / np.sqrt(len(episode["y"]) * n_episodes[episode["sid"]])
                for episode in train
            ]
        )

    opt = least_squares(
        residuals,
        x0=[
            np.log(0.095),
            0.9,
            1.09,
            -1.006,
        ],
    )

    if not opt.success:
        raise RuntimeError(f"STORM-L fitting failed: {opt.message}")

    return (
        float(np.exp(opt.x[0])),
        float(opt.x[1]),
        float(opt.x[2] * 0.01),
        float(opt.x[3] * 0.05),
    )


def evaluate(test, predict_fn):

    y = np.concatenate([episode["y"] for episode in test])
    pred = np.concatenate([predict_fn(episode) for episode in test])

    errors = y - pred

    return {
        "RMSE": np.sqrt(np.mean(errors**2)),
        "MAE": np.mean(np.abs(errors)),
        "Bias": np.mean(errors),
        "N": len(y),
        "N_episodes": len(test),
    }


def fit_decay_model(
    episodes: list[dict], basins: list[str], *, evaluate_fn: bool = False
) -> dict[str, dict[str, float]]:
    """Fit the STORM-L decay model to landfall episodes for each basin."""
    parameters = {}
    for basin in basins:
        basin_episodes = [ep for ep in episodes if ep["basin"] == basin]

        if not basin_episodes:
            raise ValueError(f"No episodes available for basin {basin}")

        alpha, r, c1, d1 = fit_STORM_landfrac(basin_episodes)

        parameters[basin] = {
            "model": "STORM-L",
            "alpha": alpha,
            "R": r,
            "C1": c1,
            "D1": d1,
            "t0_hours": 100,
            "radius_km": 100,
        }

        if evaluate_fn:
            results = evaluate(
                basin_episodes,
                lambda ep, p=parameters[basin]: predict_storm_l(
                    ep["v0"], np.cumsum(ep["dt"] * ep["L"]), ep["d"], p
                ),
            )
            print(basin, results)
    return parameters
