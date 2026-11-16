"""Compare OSM land flags with historical IBTrACS and legacy STORM masks.

    python -m CODE.validate_geography --ibtracs IBTrACS.1980_2025v04r01.nc

Differences are diagnostics, not proof that either mask is correct.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from CODE.geography import is_land, get_land_fraction, require_geography

BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}


def decode(value):
    return (
        value.decode("utf-8").strip()
        if isinstance(value, bytes)
        else str(value).strip()
    )


def compare_ibtracs(ibtracs_file, legacy_dir="Land_ocean_masks"):
    require_geography()
    with xr.open_dataset(ibtracs_file, decode_times=False) as ds:
        track = ds.isel(quadrant=0) if "quadrant" in ds.dims else ds
        df = (
            track[["sid", "basin", "lat", "lon", "dist2land"]]
            .to_dataframe()
            .reset_index()
        )
    df = df.loc[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy()
    df["Basin"] = df["basin"].map(decode)
    df["OSM"] = is_land(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["IBTrACS"] = df["dist2land"].eq(0)
    df["Landfrac"] = get_land_fraction(df["lat"].to_numpy(), df["lon"].to_numpy())
    df["Legacy"] = np.nan
    legacy_dir = Path(legacy_dir)
    for basin, (lat0, lat1, lon0, lon1) in BOUNDS.items():
        mask_file = legacy_dir / f"Land_ocean_mask_{basin}.txt"
        if not mask_file.exists():
            continue
        mask = np.loadtxt(mask_file, dtype=float)
        selected = df["Basin"].eq(basin)
        ii = np.flatnonzero(selected.to_numpy())
        if len(ii) == 0:
            continue
        lat = df.iloc[ii]["lat"].to_numpy()
        lon = df.iloc[ii]["lon"].to_numpy() % 360
        row = np.floor((lat1 - lat) * 10).astype(int)
        col = np.floor((lon - lon0) * 10).astype(int)
        valid = (row >= 0) & (row < mask.shape[0]) & (col >= 0) & (col < mask.shape[1])
        values = np.full(len(ii), np.nan)
        values[valid] = mask[row[valid], col[valid]]
        df.iloc[ii, df.columns.get_loc("Legacy")] = values
    results = []
    for basin, g in df.groupby("Basin"):
        if basin not in BOUNDS:
            continue
        observed = g["IBTrACS"].to_numpy(dtype=bool)
        predicted = g["OSM"].to_numpy(dtype=bool)
        old = g["Legacy"].to_numpy(dtype=float)
        valid = np.isfinite(old)
        results.append(
            {
                "Basin": basin,
                "N_positions": len(g),
                "OSM_vs_IBTrACS_percent": 100 * np.mean(observed != predicted),
                "OSM_vs_legacy_percent": 100
                * np.mean(predicted[valid] != old[valid].astype(bool))
                if np.any(valid)
                else np.nan,
                "N_legacy_positions": int(valid.sum()),
                "N_OSM_inland": int(predicted.sum()),
                "N_IBTrACS_inland": int(observed.sum()),
            }
        )
    return pd.DataFrame(results).sort_values("Basin"), df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ibtracs", type=Path, required=True)
    p.add_argument("--legacy-dir", default="Land_ocean_masks")
    p.add_argument("--output", default="osm_ibtracs_mask_comparison.csv")
    args = p.parse_args()
    summary, df = compare_ibtracs(args.ibtracs, args.legacy_dir)
    print(summary.to_string(index=False))
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
