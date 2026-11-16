"""Offline GSHHG L1 raster builder (NOT imported during simulation).

One-command setup: download and verify GSHHG 2.3.7, extract only the
Full-resolution Level-1 shapefile, and generate both runtime rasters:

    python -m CODE.build_geography --download

Alternatively, use an existing local shapefile:

    python -m CODE.build_geography --shapefile /path/GSHHS_f_L1.shp

Level 1 includes inland lakes as land. Latitude stripes keep memory bounded.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
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

# Pin the exact archive rather than following a mutable "latest" URL.
GSHHG_VERSION = "2.3.7"
GSHHG_ARCHIVE_NAME = f"gshhg-shp-{GSHHG_VERSION}.zip"
GSHHG_ARCHIVE_SHA256 = (
    "8dbbe7e071e77e9e75f2d639239099ebca8d5c16d6a07df8169729d49f15cf41"
)
GSHHG_URLS = (
    f"https://ftp.soest.hawaii.edu/gshhg/{GSHHG_ARCHIVE_NAME}",
    f"https://github.com/GenericMappingTools/gshhg-gmt/releases/download/"
    f"{GSHHG_VERSION}/{GSHHG_ARCHIVE_NAME}",
)
DEFAULT_DOWNLOAD_DIR = GEOGRAPHY_DIR / "gshhg"
SHAPEFILE_BASENAME = "GSHHS_f_L1"
# These are the shapefile components needed by Fiona. PRJ identifies CRS.
SHAPEFILE_EXTENSIONS = (".shp", ".shx", ".dbf", ".prj")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shapefile_complete(path):
    path = Path(path)
    return all(
        (path.parent / f"{path.stem}{ext}").is_file()
        and (path.parent / f"{path.stem}{ext}").stat().st_size > 0
        for ext in SHAPEFILE_EXTENSIONS
    )


def _download_verified_archive(
    destination, urls=GSHHG_URLS, expected_sha256=GSHHG_ARCHIVE_SHA256
):
    """Download with streaming I/O and an atomic rename after SHA-256 validation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        temp_path = None
        try:
            print(f"Downloading GSHHG {GSHHG_VERSION} from {url}", flush=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=".gshhg-download-",
                suffix=".part",
                delete=False,
            ) as temp:
                temp_path = Path(temp.name)
                request = Request(url, headers={"User-Agent": "CATMOSPHERE-STORM/1.0"})
                with urlopen(request, timeout=90) as response:
                    shutil.copyfileobj(response, temp, length=1024 * 1024)
            digest = _sha256(temp_path)
            if digest.lower() != expected_sha256.lower():
                raise ValueError(
                    f"SHA256 mismatch: {digest}; expected {expected_sha256}"
                )
            os.replace(temp_path, destination)
            print(f"Verified archive SHA-256: {digest}")
            return destination
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"  Source failed: {exc}", flush=True)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    raise RuntimeError(
        "Failed to download a valid GSHHG archive. Sources tried:\n  "
        + "\n  ".join(errors)
    )


def _extract_l1(archive, destination):
    """Extract only the L1 full-resolution shapefile; never trust ZIP paths."""
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zf:
        entries = {}
        for name in zf.namelist():
            normalized = name.replace("\\", "/")
            for ext in SHAPEFILE_EXTENSIONS:
                if normalized.endswith(f"GSHHS_shp/f/{SHAPEFILE_BASENAME}{ext}"):
                    entries[ext] = name
        missing = set(SHAPEFILE_EXTENSIONS) - set(entries)
        if missing:
            raise ValueError(
                f"Archive lacks required GSHHG L1 components: {sorted(missing)}"
            )
        with tempfile.TemporaryDirectory(
            prefix=".gshhg-extract-", dir=destination
        ) as tmp:
            tmp_path = Path(tmp)
            for ext in SHAPEFILE_EXTENSIONS:
                with (
                    zf.open(entries[ext]) as src,
                    open(tmp_path / f"{SHAPEFILE_BASENAME}{ext}", "wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
            for ext in SHAPEFILE_EXTENSIONS:
                staged = tmp_path / f"{SHAPEFILE_BASENAME}{ext}"
                if staged.stat().st_size == 0:
                    raise ValueError(f"Empty shapefile component in archive: {ext}")
            for ext in SHAPEFILE_EXTENSIONS:
                os.replace(
                    tmp_path / f"{SHAPEFILE_BASENAME}{ext}",
                    destination / f"{SHAPEFILE_BASENAME}{ext}",
                )
    path = destination / f"{SHAPEFILE_BASENAME}.shp"
    if not _shapefile_complete(path):
        raise RuntimeError(f"Incomplete shapefile extraction: {path}")
    return path


def ensure_gshhg_shapefile(download_dir=DEFAULT_DOWNLOAD_DIR, *, force_download=False):
    """Return a cached or freshly downloaded and verified GSHHG L1 shapefile.

    Existing complete extracted shapefiles are reused by default. If necessary,
    a cached ZIP is verified against the published archive SHA-256 before use.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    shapefile = download_dir / f"{SHAPEFILE_BASENAME}.shp"
    archive = download_dir / GSHHG_ARCHIVE_NAME
    if not force_download and _shapefile_complete(shapefile):
        print(f"Reusing extracted GSHHG shapefile: {shapefile}")
        return shapefile
    if not force_download and archive.is_file():
        if _sha256(archive).lower() == GSHHG_ARCHIVE_SHA256:
            print(f"Reusing SHA-256-verified GSHHG archive: {archive}")
        else:
            print("Cached GSHHG archive has an invalid checksum; downloading again.")
            _download_verified_archive(archive)
    else:
        _download_verified_archive(archive)
    try:
        shapefile = _extract_l1(archive, download_dir)
    except (BadZipFile, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not extract {archive}: {exc}") from exc
    print(f"GSHHG shapefile ready: {shapefile}")
    return shapefile


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build binary and fractional geography rasters from GSHHG Full L1."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--download",
        action="store_true",
        help="Download GSHHG 2.3.7 if needed, verify SHA-256 and extract L1",
    )
    source.add_argument(
        "--shapefile",
        type=Path,
        help="Use an existing GSHHS_f_L1.shp (with its .shx/.dbf/.prj files)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Cache for downloaded GSHHG archive and shapefile (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download and verify even if an extracted shapefile exists",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/verify/extract without building the expensive global rasters",
    )
    parser.add_argument("--stripe-rows", type=int, default=200)
    parser.add_argument("--skip-fraction", action="store_true")
    args = parser.parse_args(argv)
    if (args.force_download or args.download_only) and not args.download:
        parser.error("--force-download and --download-only require --download")
    if args.stripe_rows <= 0:
        parser.error("--stripe-rows must be a positive integer")

    shapefile = (
        ensure_gshhg_shapefile(args.download_dir, force_download=args.force_download)
        if args.download
        else args.shapefile
    )
    if not _shapefile_complete(shapefile):
        parser.error(f"Missing .shp/.shx/.dbf/.prj components for {shapefile}")
    if args.download_only:
        print(f"Verified/extracted GSHHG shapefile: {shapefile}")
        return

    rasterize_gshhg_l1(shapefile, LAND_MASK_PATH, BINARY_RESOLUTION, args.stripe_rows)
    rasterize_gshhg_l1(shapefile, SOURCE_MASK_PATH, SOURCE_RESOLUTION, args.stripe_rows)
    if not args.skip_fraction:
        generate_gshhg_land_fraction()
    print("GSHHG geography generated in", GEOGRAPHY_DIR)


if __name__ == "__main__":
    main()
