"""Shared OpenStreetMap coastline-derived geography lookups, loaded as read-only memory maps.

Binary mask: 0.01-degree north-up, longitude [-180,180), packed MSB first.
Fraction: 0.05-degree cell-centre grid, south-up, longitude [0,360).
Inland lakes count as land (OSM coastline land polygons exclude inland-water holes).
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np

GEOGRAPHY_DIR = Path(__file__).resolve().parents[1] / "geography"
LAND_MASK_PATH = GEOGRAPHY_DIR / "osm_land_0p01_packed.npy"
LAND_FRACTION_PATH = GEOGRAPHY_DIR / "osm_landfrac_r100_0p05.npy"
BINARY_RESOLUTION = 0.01
FRACTION_RESOLUTION = 0.05


def geography_provenance():
    """Small, matching metadata for binary/fraction rasters (no raster hashing at runtime).

    Prevents accidentally mixing masks generated from different OSM snapshots.
    The ZIP SHA-256 was calculated during offline acquisition; preserve the
    cached source ZIP to reproduce that snapshot.
    """
    import json

    profiles = []
    for path in (LAND_MASK_PATH, LAND_FRACTION_PATH):
        metadata_file = path.with_suffix(".json")
        if not path.is_file() or not metadata_file.is_file():
            raise FileNotFoundError(
                f"Missing {path} or {metadata_file}; run python -m CODE.build_geography --download"
            )
        with metadata_file.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("source") != "OpenStreetMap / osmdata land polygons":
            raise ValueError(f"Unexpected land-mask source: {metadata_file}")
        if meta.get("lakes") != "land":
            raise ValueError(f"Inland-lake convention differs: {metadata_file}")
        profiles.append(meta)

    mask_meta, fraction_meta = profiles
    if mask_meta.get("resolution_deg") != BINARY_RESOLUTION:
        raise ValueError("Incorrect binary geography resolution")
    if fraction_meta.get("output_resolution_deg") != FRACTION_RESOLUTION:
        raise ValueError("Incorrect land-fraction output resolution")
    if fraction_meta.get("radius_km") != 100.0:
        raise ValueError("Expected a 100 km exposure radius")
    token = mask_meta.get("source_sha256")
    if not token or token != fraction_meta.get("source_sha256"):
        raise ValueError(
            "Binary and fractional masks come from different coastline snapshots"
        )
    return {
        "source": mask_meta["source"],
        "source_sha256": token,
        "lakes": "land",
        "binary_resolution_deg": BINARY_RESOLUTION,
        "fraction_source_resolution_deg": fraction_meta.get("source_resolution_deg"),
        "fraction_resolution_deg": FRACTION_RESOLUTION,
        "radius_km": 100.0,
    }


@lru_cache(maxsize=4)
def _open_mmap(path: str):
    # Each worker opens its own VIRTUAL mapping; the OS shares file-backed pages.
    return np.load(path, mmap_mode="r", allow_pickle=False)


def is_land(lat, lon, *, path=LAND_MASK_PATH, resolution=BINARY_RESOLUTION):
    """Classify point(s) on a OSM bit-packed coastline raster; scalar or array.

    Out-of-range/NaN input raises rather than silently classifying ocean.
    Pixel lookup uses the containing cell, not an interpolated coastline.
    """
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("Land-mask coordinates contain NaN/Inf")
    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("Latitude outside [-90,90]")
    rows = round(180.0 / resolution)
    cols = round(360.0 / resolution)
    if not (
        np.isclose(rows * resolution, 180)
        and np.isclose(cols * resolution, 360)
        and cols % 8 == 0
    ):
        raise ValueError(
            "Resolution must tile the globe and longitude count be divisible by 8"
        )
    a = _open_mmap(str(Path(path).resolve()))
    if a.shape != (rows, cols // 8) or a.dtype != np.uint8:
        raise ValueError(f"Unexpected packed mask shape/type: {a.shape}, {a.dtype}")
    row = np.clip(np.floor((90.0 - lat) / resolution).astype(np.int64), 0, rows - 1)
    col = np.floor(((lon + 180.0) % 360.0) / resolution).astype(np.int64)
    col = np.minimum(col, cols - 1)
    values = ((a[row, col // 8] >> (7 - col % 8)) & 1).astype(bool)
    return bool(values) if values.ndim == 0 else values


def get_land_fraction(
    lat, lon, *, path=LAND_FRACTION_PATH, resolution=FRACTION_RESOLUTION
):
    """Bilinear, longitude-periodic lookup of radius-100 km land fraction."""
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    good = np.isfinite(lat) & np.isfinite(lon)
    safe_lat = np.where(good, lat, -90.0 + resolution / 2)
    safe_lon = np.where(good, lon, 0.0)
    if np.any(good & ((lat < -90) | (lat > 90))):
        raise ValueError("Latitude outside [-90,90]")
    a = _open_mmap(str(Path(path).resolve()))
    rows = round(180 / resolution)
    cols = round(360 / resolution)
    if a.shape != (rows, cols) or a.dtype != np.float32:
        raise ValueError(f"Unexpected fractional mask shape/type: {a.shape}, {a.dtype}")
    # Both grids use cell centres; latitude clips at the polar rows.
    fy = np.clip((safe_lat + 90) / resolution - 0.5, 0, rows - 1)
    fx = ((safe_lon % 360.0) / resolution - 0.5) % cols
    i0 = np.minimum(np.floor(fy).astype(np.int64), rows - 2)
    j0 = np.floor(fx).astype(np.int64)
    j1 = (j0 + 1) % cols
    ay, ax = np.clip(fy - i0, 0, 1), fx - j0
    v = (1 - ay) * ((1 - ax) * a[i0, j0] + ax * a[i0, j1]) + ay * (
        (1 - ax) * a[i0 + 1, j0] + ax * a[i0 + 1, j1]
    )
    v = np.where(good, v, np.nan)
    return float(v) if v.ndim == 0 else v


def require_geography():
    """Validate both products and their common coastline snapshot."""
    return geography_provenance()
