"""Offline GSHHG L1 raster builder (NOT imported during simulation).

Download GSHHG shapefiles 2.3.7 from the SOEST project and extract
GSHHS_shp/f/GSHHS_f_L1.shp. L1 alone includes inland lakes as land.

    python -m CODE.build_geography --shapefile /path/GSHHS_f_L1.shp

Build the 0.01-degree binary and 0.025-degree source masks, then the
0.05-degree 100-km land-fraction raster. Rasterize latitude stripes so
full-resolution polygons do not require a global dense in-memory raster.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from CODE.geography import (
    GEOGRAPHY_DIR,
    LAND_MASK_PATH,
    LAND_FRACTION_PATH,
    BINARY_RESOLUTION,
    FRACTION_RESOLUTION,
)

SOURCE_MASK_PATH = GEOGRAPHY_DIR / "gshhg_f_L1_land_0p025_packed.npy"
SOURCE_RESOLUTION = 0.025


def rasterize_gshhg_l1(shapefile, output, resolution, stripe_rows=200):
    """Create bit-packed north-up global raster at specified resolution.

    Uses Fiona bbox iteration and Rasterio rasterization per latitude stripe.
    all_touched=False preserves centre-in-polygon semantics; tests should
    quantify omitted islands smaller than one source pixel.
    """
    import fiona
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    shapefile = Path(shapefile)
    if not shapefile.exists():
        raise FileNotFoundError(shapefile)
    rows, cols = int(round(180 / resolution)), int(round(360 / resolution))
    if not (
        np.isclose(rows * resolution, 180)
        and np.isclose(cols * resolution, 360)
        and cols % 8 == 0
    ):
        raise ValueError(
            "Grid must cover Earth and longitude columns be divisible by 8"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bits = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.uint8, shape=(rows, cols // 8)
    )
    try:
        with fiona.open(shapefile) as src:
            for start in range(0, rows, stripe_rows):
                stop = min(rows, start + stripe_rows)
                ymax, ymin = 90 - start * resolution, 90 - stop * resolution
                # Fiona's bbox filter avoids loading every polygon into Python.
                shapes = (
                    (feature["geometry"], 1)
                    for feature in src.filter(bbox=(-180, ymin, 180, ymax))
                    if feature["geometry"] is not None
                )
                tile = rasterize(
                    shapes,
                    out_shape=(stop - start, cols),
                    transform=from_origin(-180, ymax, resolution, resolution),
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False,
                )
                bits[start:stop] = np.packbits(tile, axis=1, bitorder="big")
                if start % (stripe_rows * 10) == 0:
                    print(f"  {output.name}: {stop}/{rows} latitude rows")
    finally:
        bits.flush()
        del bits
    metadata = {
        "source": "GSHHG",
        "version": "2.3.7",
        "layer": "GSHHS_f_L1",
        "lakes": "land",
        "resolution_deg": resolution,
        "grid": "latitude descending / longitude [-180,180)",
        "pixel_rule": "centre-in-polygon, all_touched=False",
        "bit_order": "big",
        "packed_shape": [rows, cols // 8],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return output


def unpack_source_for_fraction(path=SOURCE_MASK_PATH, resolution=SOURCE_RESOLUTION):
    """Offline only: unpack a 0.025° source into ~104 MB uint8 (no workers)."""
    packed = np.load(path, mmap_mode="r")
    rows, cols = round(180 / resolution), round(360 / resolution)
    if packed.shape != (rows, cols // 8):
        raise ValueError("Source resolution does not match packed mask")
    # The source is north-up -180..180; convolution expects south-up 0..360.
    land = np.empty((rows, cols), dtype=np.uint8)
    half = cols // 2
    for i in range(rows):
        row = np.unpackbits(packed[rows - 1 - i], bitorder="big", count=cols)
        land[i, :half] = row[half:]
        land[i, half:] = row[:half]
    return land


def generate_gshhg_land_fraction(
    source=SOURCE_MASK_PATH,
    output=LAND_FRACTION_PATH,
    source_resolution=SOURCE_RESOLUTION,
    output_resolution=FRACTION_RESOLUTION,
    radius_km=100.0,
):
    """Offline spherical-cap averaging, using the repository's reference kernel."""
    from CODE.land_fraction import _compute_area_weighted

    if radius_km != 100:
        raise ValueError("Update filename/model metadata when changing radius_km")
    land = unpack_source_for_fraction(source, source_resolution)
    ny, nx = land.shape
    lat_in = -90 + (np.arange(ny) + 0.5) * source_resolution
    lon_in = (np.arange(nx) + 0.5) * source_resolution
    lat_out = (
        -90 + (np.arange(round(180 / output_resolution)) + 0.5) * output_resolution
    )
    lon_out = (np.arange(round(360 / output_resolution)) + 0.5) * output_resolution
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=(len(lat_out), len(lon_out))
    )
    # Calculate one latitude band at a time; no 104-MB output ndarray.
    # The heavy spherical convolution is a one-time offline operation.
    output_rows_per_chunk = 64
    try:
        for start in range(0, len(lat_out), output_rows_per_chunk):
            stop = min(len(lat_out), start + output_rows_per_chunk)
            dst[start:stop] = _compute_area_weighted(
                lat_in, lon_in, land, lat_out[start:stop], lon_out, radius_km
            )
            if start % (output_rows_per_chunk * 10) == 0:
                print(f"  {output.name}: {stop}/{len(lat_out)} output latitude rows")
    finally:
        dst.flush()
        del dst
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "GSHHG 2.3.7 GSHHS_f_L1",
                "lakes": "land",
                "method": "source-cell-centre spherical-cap, area weighted",
                "radius_km": radius_km,
                "source_resolution_deg": source_resolution,
                "output_resolution_deg": output_resolution,
                "shape": [len(lat_out), len(lon_out)],
                "grid": "latitude increasing / longitude [0,360), cell centres",
            },
            indent=2,
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", required=True, type=Path)
    parser.add_argument("--stripe-rows", type=int, default=200)
    parser.add_argument("--skip-fraction", action="store_true")
    args = parser.parse_args()
    rasterize_gshhg_l1(
        args.shapefile, LAND_MASK_PATH, BINARY_RESOLUTION, args.stripe_rows
    )
    rasterize_gshhg_l1(
        args.shapefile, SOURCE_MASK_PATH, SOURCE_RESOLUTION, args.stripe_rows
    )
    if not args.skip_fraction:
        generate_gshhg_land_fraction()
    print("GSHHG geography generated in", GEOGRAPHY_DIR)


if __name__ == "__main__":
    main()
