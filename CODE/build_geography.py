"""Build CATMOSPHERE-STORM geography from OpenStreetMap coastline land polygons.

One-time setup (large global OSM download):
    python -m CODE.build_geography --download

Or use a previously extracted WGS84 shapefile:
    python -m CODE.build_geography --shapefile /path/land_polygons.shp

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

import numpy as np

from CODE.geography import (
    GEOGRAPHY_DIR,
    LAND_MASK_PATH,
    LAND_FRACTION_PATH,
    BINARY_RESOLUTION,
    FRACTION_RESOLUTION,
)

SOURCE_RESOLUTION = 0.025
SOURCE_MASK_PATH = GEOGRAPHY_DIR / "osm_land_0p025_packed.npy"
OSM_LAND_POLYGONS_PAGE = "https://osmdata.openstreetmap.de/data/land-polygons.html"
OSM_URL = "https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip"
ARCHIVE_NAME = "land-polygons-split-4326.zip"
DEFAULT_DOWNLOAD_DIR = GEOGRAPHY_DIR / "osm"
SHAPEFILE_BASENAME = "land_polygons"
REQUIRED_EXTENSIONS = (".shp", ".shx", ".dbf", ".prj")
OPTIONAL_EXTENSIONS = (".cpg",)
SOURCE_NAME = "OpenStreetMap / osmdata land polygons"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_shapefile(path):
    path = Path(path)
    return all(
        (path.parent / f"{path.stem}{ext}").is_file()
        and (path.parent / f"{path.stem}{ext}").stat().st_size > 0
        for ext in REQUIRED_EXTENSIONS
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        temp = Path(tmp.name)
        json.dump(value, tmp, indent=2)
    os.replace(temp, path)


def _download_archive(destination, url=OSM_URL):
    """Stream mutable upstream ZIP to temporary file and atomically replace cache.

    The publisher does not supply a pinned checksum for this rolling download.
    We therefore record the acquired ZIP SHA256 for reproducibility, NOT as
    an externally verified publisher checksum.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=".osm-coastline-",
            suffix=".part",
            delete=False,
        ) as out:
            tmp_file = Path(out.name)
            req = Request(url, headers={"User-Agent": "CATMOSPHERE-STORM/1.0"})
            print(f"Downloading OSM land polygons (large archive): {url}", flush=True)
            with urlopen(req, timeout=180) as response:
                while chunk := response.read(8 * 1024 * 1024):
                    out.write(chunk)
        with ZipFile(tmp_file) as zf:
            if not any(
                p.replace("\\", "/").endswith("/land_polygons.shp")
                or p == "land_polygons.shp"
                for p in zf.namelist()
            ):
                raise ValueError("Downloaded ZIP is not an OSM land polygons archive")
        digest = _sha256(tmp_file)
        os.replace(tmp_file, destination)
        print(f"OSM source snapshot SHA256: {digest}")
        return digest
    finally:
        if tmp_file is not None:
            tmp_file.unlink(missing_ok=True)


def _extract_shapefile(archive, destination):
    """Extract only required OSM shapefile components, streaming to disk."""
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        names = {}
        for member in zf.namelist():
            normalized = member.replace("\\", "/")
            for ext in (*REQUIRED_EXTENSIONS, *OPTIONAL_EXTENSIONS):
                if Path(normalized).name == f"{SHAPEFILE_BASENAME}{ext}":
                    if ext in names:
                        raise ValueError(f"Duplicate shapefile component in ZIP: {ext}")
                    names[ext] = member
        missing = set(REQUIRED_EXTENSIONS).difference(names)
        if missing:
            raise ValueError(f"Missing components in OSM ZIP: {sorted(missing)}")
        # Avoid excessive expansion from an unexpected archive.
        total_uncompressed = sum(zf.getinfo(name).file_size for name in names.values())
        if total_uncompressed > 25 * (1024**3):
            raise ValueError("Unexpectedly large extracted OSM shapefile")
        with tempfile.TemporaryDirectory(
            dir=destination, prefix=".osm-extract-"
        ) as tmp:
            staged = Path(tmp)
            for ext, member in names.items():
                with (
                    zf.open(member) as source,
                    (staged / f"{SHAPEFILE_BASENAME}{ext}").open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if (staged / f"{SHAPEFILE_BASENAME}{ext}").stat().st_size == 0:
                    raise ValueError(f"Empty {ext} component")
            for ext in names:
                os.replace(
                    staged / f"{SHAPEFILE_BASENAME}{ext}",
                    destination / f"{SHAPEFILE_BASENAME}{ext}",
                )
    path = destination / f"{SHAPEFILE_BASENAME}.shp"
    if not _complete_shapefile(path):
        raise RuntimeError("OSM coastline shapefile extraction incomplete")
    return path


def ensure_osm_shapefile(download_dir=DEFAULT_DOWNLOAD_DIR, *, force_download=False):
    """Reuse or obtain official split-WGS84 OSM land polygons.

    Returns (path, provenance). Archive and metadata are deliberately cached;
    use --force-download to obtain a new rolling OSM snapshot.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    shapefile = download_dir / f"{SHAPEFILE_BASENAME}.shp"
    archive = download_dir / ARCHIVE_NAME
    manifest_file = download_dir / "osm_source.json"

    if (
        not force_download
        and _complete_shapefile(shapefile)
        and manifest_file.is_file()
    ):
        info = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            info.get("source") == SOURCE_NAME
            and info.get("source_sha256")
            and archive.exists()
            and archive.stat().st_size == info.get("archive_bytes")
        ):
            print(f"Using cached OSM source: {shapefile}")
            return shapefile, info

    if force_download or not archive.is_file():
        digest = _download_archive(archive)
    else:
        digest = _sha256(archive)
        print(f"Using existing OSM ZIP (SHA256 {digest})")

    try:
        shapefile = _extract_shapefile(archive, download_dir)
    except (BadZipFile, OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid OSM source archive {archive}: {exc}") from exc

    info = {
        "source": SOURCE_NAME,
        "source_url": OSM_URL,
        "source_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "snapshot_recorded_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "land-polygons-split-4326",
        "license": "ODbL 1.0; © OpenStreetMap contributors",
    }
    _atomic_json(manifest_file, info)
    return shapefile, info


def _local_source_info(shapefile):
    shapefile = Path(shapefile)
    if not _complete_shapefile(shapefile):
        raise FileNotFoundError(f"Incomplete shapefile: {shapefile}")
    # Hash the complete SHP binary for consistent reproducibility on local inputs.
    return {
        "source": SOURCE_NAME,
        "source_url": "user-supplied local WGS84 land polygons",
        "source_sha256": _sha256(shapefile),
        "dataset": "locally supplied coastline shapefile",
        "license": "Check provenance; OSM derivatives are generally ODbL 1.0",
    }


def rasterize_osm_land(
    shapefile, output, resolution, stripe_rows=200, *, provenance=None
):
    """Rasterize OSM land polygons as an 8x compressed north-up global mask.

    OSM's 'split' land polygons may overlap; rasterize uses OR semantics
    (burn value 1) so overlapping polygons do not double-count land.
    """
    import fiona
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    shapefile = Path(shapefile)
    if not _complete_shapefile(shapefile):
        raise FileNotFoundError(shapefile)
    rows, cols = round(180 / resolution), round(360 / resolution)
    if not (
        np.isclose(rows * resolution, 180)
        and np.isclose(cols * resolution, 360)
        and cols % 8 == 0
    ):
        raise ValueError(
            "Invalid global resolution (requires divisible-by-8 longitude width)"
        )
    if stripe_rows < 1:
        raise ValueError("stripe_rows must be >= 1")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = output.with_suffix(".json")
    metadata_file.unlink(missing_ok=True)  # Partial output must not be accepted.
    staging = output.with_name(f"{output.stem}.building.npy")

    try:
        with fiona.open(shapefile) as src:
            crs = src.crs
            if crs is None or crs.to_epsg() != 4326:
                raise ValueError(f"OSM shapefile must use WGS84 EPSG:4326, got {crs}")
            bits = np.lib.format.open_memmap(
                staging, mode="w+", dtype=np.uint8, shape=(rows, cols // 8)
            )
            try:
                for start in range(0, rows, stripe_rows):
                    stop = min(start + stripe_rows, rows)
                    north, south = 90 - start * resolution, 90 - stop * resolution
                    shapes = (
                        (feat["geometry"], 1)
                        for feat in src.filter(bbox=(-180, south, 180, north))
                        if feat["geometry"] is not None
                    )
                    tile = rasterize(
                        shapes,
                        out_shape=(stop - start, cols),
                        transform=from_origin(-180, north, resolution, resolution),
                        fill=0,
                        dtype=np.uint8,
                        all_touched=False,
                    )
                    bits[start:stop] = np.packbits(tile, axis=1, bitorder="big")
                    if start % max(stripe_rows * 10, 1) == 0:
                        print(f"  {output.name}: {stop}/{rows} rows", flush=True)
            finally:
                bits.flush()
                del bits
        os.replace(staging, output)
        _atomic_json(
            metadata_file,
            {
                "source": SOURCE_NAME,
                "source_url": (provenance or {}).get("source_url", str(shapefile)),
                "source_sha256": (provenance or {}).get("source_sha256")
                or _sha256(shapefile),
                "lakes": "land",
                "resolution_deg": resolution,
                "grid": {
                    "crs": "EPSG:4326",
                    "latitude_order": "ascending",
                    "longitude_convention": "0_360",
                    "grid_registration": "cell_center",
                    "latitude_first_center": -89.975,
                    "longitude_first_center": 0.025,
                    "resolution_deg": 0.05,
                    "shape": [3600, 7200],
                    "array_axes": ["latitude", "longitude"],
                },
                "pixel_rule": "centre-in-polygon (all_touched=False)",
                "bit_order": "big",
                "packed_shape": [rows, cols // 8],
            },
        )
    finally:
        staging.unlink(missing_ok=True)
    return output


def unpack_source_for_fraction(path=SOURCE_MASK_PATH, resolution=SOURCE_RESOLUTION):
    """Offline-only transform of packed source to south-up, [0,360) uint8."""
    packed = np.load(path, mmap_mode="r")
    rows, cols = round(180 / resolution), round(360 / resolution)
    if packed.shape != (rows, cols // 8):
        raise ValueError("Source resolution does not match packed mask")
    land = np.empty((rows, cols), dtype=np.uint8)
    half = cols // 2
    for i in range(rows):
        row = np.unpackbits(packed[rows - 1 - i], bitorder="big", count=cols)
        land[i, :half] = row[half:]
        land[i, half:] = row[:half]
    return land


def generate_osm_land_fraction(
    source=SOURCE_MASK_PATH,
    output=LAND_FRACTION_PATH,
    source_resolution=SOURCE_RESOLUTION,
    output_resolution=FRACTION_RESOLUTION,
    radius_km=100.0,
):
    """Area-weighted spherical-cap average, using existing STORM-L kernel."""
    from CODE.land_fraction import _compute_area_weighted

    if radius_km != 100.0:
        raise ValueError("New radius requires new coefficient and raster contract")
    source = Path(source)
    source_meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if source_meta.get("resolution_deg") != source_resolution:
        raise ValueError("Intermediate source resolution mismatch")
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
    output.with_suffix(".json").unlink(missing_ok=True)
    staging = output.with_name(f"{output.stem}.building.npy")
    try:
        dst = np.lib.format.open_memmap(
            staging, mode="w+", dtype=np.float32, shape=(len(lat_out), len(lon_out))
        )
        try:
            for start in range(0, len(lat_out), 64):
                stop = min(len(lat_out), start + 64)
                dst[start:stop] = _compute_area_weighted(
                    lat_in, lon_in, land, lat_out[start:stop], lon_out, radius_km
                )
                if start % 640 == 0:
                    print(f"  {output.name}: {stop}/{len(lat_out)} rows", flush=True)
        finally:
            dst.flush()
            del dst
        os.replace(staging, output)
        _atomic_json(
            output.with_suffix(".json"),
            {
                "source": SOURCE_NAME,
                "source_url": source_meta["source_url"],
                "source_sha256": source_meta["source_sha256"],
                "lakes": "land",
                "method": "source-cell-centre spherical-cap, area weighted",
                "radius_km": radius_km,
                "source_resolution_deg": source_resolution,
                "output_resolution_deg": output_resolution,
                "shape": [len(lat_out), len(lon_out)],
                "grid": {
                    "crs": "EPSG:4326",
                    "latitude_order": "ascending",
                    "longitude_convention": "0_360",
                    "grid_registration": "cell_center",
                    "latitude_first_center": -89.975,
                    "longitude_first_center": 0.025,
                    "resolution_deg": 0.05,
                    "shape": [3600, 7200],
                    "array_axes": ["latitude", "longitude"]
                }
            },
        )
    finally:
        staging.unlink(missing_ok=True)
    return output


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
