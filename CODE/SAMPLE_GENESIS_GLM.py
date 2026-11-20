"""Poisson GLM spatial/temporal genesis sampler (independent of legacy GPI).

Output matches existing TC_movement/TC_pressure inputs:
    months, longitudes, latitudes = sampler.sample_season(env_years, forecast_cfg)

Monthly cell intensities are predicted and cached per environmental month-year;
random Poisson events are re-drawn for EACH simulated synthetic year.
"""

from __future__ import annotations

import calendar
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

from CODE import genesis_dataset as gd
from CODE.genesis_complementary import feature_path

RAW_FEATURE = {"NA": "rsst", "EP": "rsst_bis", "WP": "eio_dynamic", "SP": "roni_ond"}


class GenesisGLMSampler:
    def __init__(self, basin, root, model, metadata, exposure, region_weights):
        self.basin = basin
        self.root = Path(root)
        self.model = model
        self.meta = metadata
        self.exposure = exposure
        self.regions = region_weights
        self.months = tuple(int(m) for m in metadata["active_months"])
        self.area_unit = float(metadata["area_unit_km2"])
        self.reference_days = float(metadata["reference_month_days"])
        self.lats = np.asarray(exposure.latitude.values)
        self.lons = np.asarray(exposure.longitude.values)
        self.area = np.asarray(exposure.eligible_area_km2.values)
        self.valid = self.area > 0
        self.shape = self.area.shape
        bounds = tuple(metadata["genesis_bounds"])
        if bounds != tuple(exposure.attrs["genesis_bounds"]):
            raise ValueError(f"{basin}: genesis bounds differ from fitted exposure")
        if metadata["osm_sha256"] != exposure.attrs["osm_sha256"]:
            raise ValueError(f"{basin}: OSM provenance mismatch")
        if self.shape != (int(bounds[1] - bounds[0]), int(bounds[3] - bounds[2])):
            raise ValueError(f"{basin}: unexpected 1-degree exposure shape")
        if not np.all(np.diff(self.lats) < 0) or not np.all(np.diff(self.lons) > 0):
            raise ValueError(
                "Genesis grid must be north-to-south with increasing longitude"
            )
        self.row, self.col = np.where(self.valid)
        self.cell_lat = self.lats[self.row]
        self.cell_lon = self.lons[self.col]
        self.lat0, self.lat1, self.lon0, self.lon1 = bounds
        for name, arr in self.regions.items():
            if arr.shape != self.shape:
                raise ValueError(f"Regional field {name} shape mismatch")
        self.region_labels = np.array(
            [
                gd.assign_region(basin, float(a), float(b))
                for a, b in zip(self.cell_lat, self.cell_lon)
            ]
        )

    @classmethod
    def load(cls, basin, base_dir):
        root = Path(base_dir)
        p = root / "genesis_prepared"
        model = sm.load(p / f"glm_{basin}.pkl")
        meta = joblib.load(p / f"metadata_{basin}.joblib")
        with xr.open_dataset(p / f"osm_genesis_exposure_{basin}_1deg.nc") as ds:
            exposure = ds.load()
        with np.load(p / f"regional_weights_{basin}.npz") as archive:
            regions = {key: archive[key].copy() for key in archive.files}
        # Fail if the OSM mask was changed since fitting (sampling uses this mask).
        mask_meta = gd._metadata_for_osm(gd.LAND_MASK_PATH)
        if mask_meta["source_sha256"] != meta["osm_sha256"]:
            raise ValueError(f"{basin}: current OSM mask differs from GLM fit")
        return cls(basin, root, model, meta, exposure, regions)

    def _raw_complement(self, env_year, month):
        if self.basin not in RAW_FEATURE:
            return None
        path = feature_path(self.root, self.basin, env_year, month)
        if not path.exists():
            raise FileNotFoundError(
                f"{path}: complementary genesis predictor missing; "
                "run historical/forecast feature preparation first"
            )
        json_file = path.with_suffix(".json")
        if not json_file.exists():
            raise FileNotFoundError(
                f"{json_file}: missing complementary predictor provenance"
            )
        info = json.loads(json_file.read_text())
        if info.get("value_kind") != "raw_before_training_standardization":
            raise ValueError(f"{path}: incorrect scaler convention")
        val = float(np.load(path))
        if not np.isfinite(val):
            raise ValueError(f"{path}: nonfinite predictor")
        if info.get("status", "").startswith("neutral_") or info.get(
            "status", ""
        ).startswith("partial_"):
            print(
                f"[GLM {self.basin}] {env_year}/{month}: {info['status']}", flush=True
            )
        return val

    def _global_field(self, stem, env_year, month):
        filename = self.root / "env_yearly" / f"{stem}_{env_year}_{month}.npy"
        if not filename.exists():
            raise FileNotFoundError(f"GLM needs actual year-specific field {filename}")
        a = np.load(filename, mmap_mode="r", allow_pickle=False)
        if a.shape != (721, 1440):
            raise ValueError(f"{filename}: expected 721x1440, got {a.shape}")
        ilat = np.rint((90.0 - self.lats) / 0.25).astype(int)
        ilon = np.rint((self.lons % 360) / 0.25).astype(int) % 1440
        return np.asarray(a[np.ix_(ilat, ilon)])[self.valid].astype(float)

    def predictor_frame(self, month, env_year, calendar_year=None):
        """Same columns, transformations, spatial cells and offset as fitting."""
        month, env_year = int(month), int(env_year)
        calendar_year = int(calendar_year if calendar_year is not None else env_year)
        if month not in self.months:
            raise ValueError(
                f"{self.basin}: month {month} outside fitted active months {self.months}"
            )
        row = pd.DataFrame(
            {
                "latitude": self.cell_lat,
                "longitude": self.cell_lon,
                "month": month,
                "pi": self._global_field("VMAX_PI", env_year, month),
                "rh600": self._global_field("RH600", env_year, month),
                "vws": self._global_field("VWS", env_year, month),
            }
        )
        relative = self._global_field("VORT850", env_year, month)
        f = 2.0 * gd.OMEGA * np.sin(np.deg2rad(self.cell_lat))
        hemisphere = 1.0 if self.basin in ("NA", "EP", "WP", "NI") else -1.0
        row["eta850"] = np.maximum(0.0, hemisphere * (relative + f))
        row["eta_clip"] = row.eta850.clip(upper=float(self.meta["eta_clip"]))
        raw_name = RAW_FEATURE.get(self.basin)
        if raw_name is not None:
            row[raw_name] = self._raw_complement(env_year, month)
        row["region"] = self.region_labels
        if self.basin == "NI":
            row["ni_season"] = (
                "pre_monsoon"
                if month in (3, 4, 5, 6)
                else "monsoon"
                if month in (7, 8, 9)
                else "post_monsoon"
            )
        for name, array in self.regions.items():
            row[name] = array[self.valid]
        scalers = self.meta["scalers"]
        # These names match genesis_dataset.standardize_train_test EXACTLY.
        output_names = {
            "rsst": "rsst",
            "rsst_bis": "rsst_bis",
            "eio_dynamic": "z_eio_dynamic",
        }
        for name in (
            "pi",
            "rh600",
            "vws",
            "eta_clip",
            "rsst",
            "rsst_bis",
            "eio_dynamic",
        ):
            if name in scalers:
                mean, std = map(float, scalers[name])
                row[output_names.get(name, "z_" + name)] = (row[name] - mean) / std
        if self.basin == "SP":
            mean, std = map(float, scalers["roni_ond_by_season"])
            row["roni_ond_standardized"] = (row.roni_ond - mean) / std
        days = calendar.monthrange(calendar_year, month)[1]
        row["exposure"] = (
            self.area[self.valid] / self.area_unit * (days / self.reference_days)
        )
        row["log_exposure"] = np.log(row.exposure)
        needed = list(row.columns)
        if not np.all(np.isfinite(row.select_dtypes(include=["number"]).values)):
            raise ValueError(f"{self.basin}/{env_year}/{month}: non-finite predictor")
        return row

    @lru_cache(maxsize=128)
    def expected_grid(self, month, env_year, calendar_year=None):
        """Cached expected monthly genesis counts over the *fitted* 1° grid."""
        frame = self.predictor_frame(month, env_year, calendar_year)
        mu = gd.predict_cell_counts(self.model, frame).to_numpy()
        if not np.all(np.isfinite(mu)) or np.any(mu < 0):
            raise ValueError("Invalid predicted Poisson means")
        grid = np.zeros(self.shape, dtype=np.float64)
        grid[self.valid] = mu
        return grid

    def sample_season(self, env_years, forecast_cfg=None, rng=None, months=None):
        """Draw independent Poisson counts and locations for each month.

        env_years must give an actual historical year or forecast env_year for
        EACH fitted month. No silent phase/climatology fallback is allowed.
        """
        if env_years is None:
            raise ValueError("GLM genesis requires --yearly environmental fields")
        rng = np.random if rng is None else rng
        months_out, lon_out, lat_out = [], [], []
        selected_months = (
            self.months if months is None else tuple(int(m) for m in months)
        )
        if any(m not in self.months for m in selected_months):
            raise ValueError(f"Invalid requested months for {self.basin}")
        for month in selected_months:
            env_year = env_years.get(month, env_years.get(str(month)))
            if env_year is None:
                raise ValueError(
                    f"{self.basin}: missing environmental year for month {month}"
                )
            # In forecast mode, the physical year is not the synthetic env-year ID
            # (e.g. env_year=10012); it is the forecast initialization year.
            if forecast_cfg:
                year = int(forecast_cfg["base_year"])
                init_month = int(str(forecast_cfg["init_date"])[5:7])
                # A Nov initialization can generate Jan-Apr in the NEXT year.
                if month < init_month:
                    year += 1
            else:
                year = int(env_year)
            mu = self.expected_grid(month, int(env_year), year)
            total = float(mu.sum())
            if total <= 0:
                continue
            count = int(rng.poisson(total))
            if count <= 0:
                continue
            picks = rng.choice(mu.size, size=count, p=mu.ravel() / total)
            for flat in picks:
                i, j = np.unravel_index(int(flat), mu.shape)
                lat, lon = self._sample_valid_point(i, j, rng)
                months_out.append(month)
                lon_out.append(lon)
                lat_out.append(lat)
        return months_out, lon_out, lat_out

    def _sample_valid_point(self, row, col, rng):
        """Uniform ocean-area proposal inside a 1° cell, OSM/isthmus rejection.

        Selection probability by 1° cell already includes eligible ocean area.
        Do NOT multiply or reweight the cell by ocean fraction a second time.
        """
        top = self.lat1 - row
        lo = self.lon0 + col
        sin_bot = np.sin(np.deg2rad(top - 1.0))
        sin_top = np.sin(np.deg2rad(top))
        for _ in range(10000):
            lat = np.rad2deg(np.arcsin(rng.uniform(sin_bot, sin_top)))
            lon = rng.uniform(lo, lo + 1.0)
            if gd.basin_side_allowed(self.basin, lat, lon) and not gd.is_land(lat, lon):
                return float(lat), float(lon % 360.0)
        raise RuntimeError(
            f"{self.basin}: could not sample eligible 0.01° OSM ocean in {(row, col)}"
        )
