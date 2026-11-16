"""Master to build CATMOSPHERE-STORM geography from OpenStreetMap coastline land polygons.

One-time setup (large global OSM download):
    python -m MASTER.MASTER_landmask --download

Or use a previously extracted WGS84 shapefile:
    python -m MASTER.MASTER_landmask --shapefile /path/land_polygons.shp

Source: https://osmdata.openstreetmap.de/data/land-polygons.html
WGS84 split polygons are preferred for spatial-filtered rasterization; overlapping
split polygons combine by OR and do not double-count land. Lakes count as land.

During simulation no shapefiles are opened: workers use two read-only .npy maps.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

from CODE.build_geography import (
    DEFAULT_DOWNLOAD_DIR,
    _local_source_info,
    ensure_osm_shapefile,
    rasterize_osm_land,
    SOURCE_MASK_PATH,
    SOURCE_RESOLUTION,
    generate_osm_land_fraction,
)
import numpy as np

from CODE.geography import (
    GEOGRAPHY_DIR,
    LAND_MASK_PATH,
    LAND_FRACTION_PATH,
    BINARY_RESOLUTION,
    FRACTION_RESOLUTION,
)

__location__ = os.path.realpath(os.getcwd())  # TEMP FIX?
dir_path = __location__


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build memory-mapped binary and fractional masks using OSM land polygons"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download", action="store_true", help="Get OSM split-WGS84 land polygons"
    )
    source.add_argument(
        "--shapefile", type=Path, help="Use existing land_polygons.shp (EPSG:4326)"
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument(
        "--force-download", action="store_true", help="Refresh rolling OSM source"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/extract; no rasterization",
    )
    parser.add_argument(
        "--skip-fraction",
        action="store_true",
        help="Build both binary grids, no 100 km convolution",
    )
    parser.add_argument("--stripe-rows", type=int, default=200)
    args = parser.parse_args(argv)
    if (args.force_download or args.download_only) and not args.download:
        parser.error("--force-download and --download-only require --download")
    if args.stripe_rows < 1:
        parser.error("--stripe-rows must be >= 1")

    if args.download:
        shapefile, provenance = ensure_osm_shapefile(
            args.download_dir, force_download=args.force_download
        )
    else:
        shapefile = args.shapefile
        provenance = _local_source_info(shapefile)
    if args.download_only:
        print(f"OSM coastline ready: {shapefile}")
        return

    rasterize_osm_land(
        shapefile,
        LAND_MASK_PATH,
        BINARY_RESOLUTION,
        args.stripe_rows,
        provenance=provenance,
    )
    rasterize_osm_land(
        shapefile,
        SOURCE_MASK_PATH,
        SOURCE_RESOLUTION,
        args.stripe_rows,
        provenance=provenance,
    )
    if not args.skip_fraction:
        generate_osm_land_fraction()
    print(f"OSM geography generated: {GEOGRAPHY_DIR}")
    if not args.skip_fraction:
        from CODE.geography import require_geography

        print("Geography fingerprint:", require_geography()["source_sha256"])


if __name__ == "__main__":
    main()
