# general
from typing import Optional

import numpy as np
import rasterio

# geospatial
from geotessera import GeoTessera
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rich.progress import track

# custom
from .config import config  # Import the global config instance

TESSERA = GeoTessera()


def check_bbox(
    lat_coords: tuple, lon_coords: tuple, min_bbox_size: Optional[float] = None
):
    """
    Define a bounding box from two coordinate tuples with some basic sanity checking.

    Args:
        lat_coords: Tuple of latitude coordinates
        lon_coords: Tuple of longitude coordinates
        min_bbox_size: Override config default if provided
    """
    # Use config value if not explicitly provided
    if min_bbox_size is None:
        min_bbox_size = config.min_bbox_size

    if min_bbox_size < 0.1:
        raise ValueError("Minimum bbox size must be at least 0.1 degrees")

    lat_min, lat_max = min(lat_coords), max(lat_coords)
    lon_min, lon_max = min(lon_coords), max(lon_coords)

    # sanity checks
    if lat_min < -90 or lat_max > 90:
        raise ValueError("Latitude values must be between -90 and 90 degrees")
    if lon_min < -180 or lon_max > 180:
        raise ValueError("Longitude values must be between -180 and 180 degrees")

    # check that bounded region > min_bbox_size on both sides
    if lat_max - lat_min < min_bbox_size or lon_max - lon_min < min_bbox_size:
        raise ValueError(
            f"Bounding box too small. Minimum size is {min_bbox_size} by {min_bbox_size} degs, "
            f"but got {lat_max - lat_min:.2f} by {lon_max - lon_min:.2f} degs"
        )

    return (lat_min, lat_max), (lon_min, lon_max)


def check_tessera_tiles(
    lat_coords: tuple, lon_coords: tuple, year: Optional[int] = None
) -> list[tuple[float, float]]:
    """
    Check that the tiles are within the bounds of the Tessera tiles.

    Args:
        lat_coords: Tuple of latitude coordinates
        lon_coords: Tuple of longitude coordinates
        year: Override config year if provided
    """
    # Use config values
    target_year = year if year is not None else config.year

    # Validate and get bounding box
    (lat_min, lat_max), (lon_min, lon_max) = check_bbox(lat_coords, lon_coords)
    roi_bounds = (
        lon_min,
        lat_min,
        lon_max,
        lat_max,
    )  # (min_lon, min_lat, max_lon, max_lat)

    print(f"Searching for tiles in ROI: {roi_bounds} for year {target_year}")

    # --- Find tiles in ROI ---
    tiles_to_merge = []
    for available_year in TESSERA.get_available_years():
        TESSERA._ensure_year_loaded(available_year)

    # search tessera for embeddings within ROI
    for emb_year, lat, lon in TESSERA.list_available_embeddings():
        if emb_year != target_year:
            continue
        tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat = (
            lon,
            lat,
            lon + 0.1,
            lat + 0.1,
        )
        if (
            tile_min_lon < roi_bounds[2]
            and tile_max_lon > roi_bounds[0]
            and tile_min_lat < roi_bounds[3]
            and tile_max_lat > roi_bounds[1]
        ):
            tiles_to_merge.append((lat, lon))

    if not tiles_to_merge:
        raise ValueError(
            f"No embedding tiles found for the specified ROI in year {target_year}"
        )
    print(f"Found {len(tiles_to_merge)} tiles to merge.")
    return tiles_to_merge


def fetch_embedding_metadata(lat, lon, target_year):
    embedding = TESSERA.get_embedding(lat, lon, year=target_year)  # (H, W, C)
    landmask_path = TESSERA._fetch_landmask(lat, lon, progressbar=False)
    with rasterio.open(landmask_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_height, src_width = src.height, src.width
        src_bounds = src.bounds

    return (
        embedding,
        landmask_path,
        src_crs,
        src_transform,
        src_height,
        src_width,
        src_bounds,
    )


def reproject_tessera_tiles(tiles_to_merge):
    for t, (lat, lon) in track(
        enumerate(tiles_to_merge),
        total=len(tiles_to_merge),
        description="Processing tiles",
    ):
        print(
            f"Processing tile {t + 1} of {len(tiles_to_merge)}: ({lat:.2f}, {lon:.2f})"
        )
        try:
            (
                embedding,
                landmask_path,
                src_crs,
                src_transform,
                src_height,
                src_width,
                src_bounds,
            ) = fetch_embedding_metadata(lat, lon, config.year)
            reprojected_embedding = reproject_embedding(
                embedding,
                src_crs,
                src_transform,
                src_height,
                src_width,
                src_bounds,
                config.target_crs,
            )
            reprojected_embedding = save_reprojected_embedding_in_memory(
                reprojected_embedding,
                src_transform,
                config.target_crs,
                src_height,
                src_width,
                embedding,
            )
        except Exception as e:
            print(f"\t! WARNING: Could not process tile ({lat:.2f}, {lon:.2f}): {e}")
            continue


def reproject_embedding(
    embedding, src_crs, src_transform, src_height, src_width, src_bounds, target_crs
):
    """
    Reproject an embedding to a target CRS.
    """
    # 2. Calculate the parameters for reprojection
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, src_width, src_height, *src_bounds
    )

    # check dimensions are valid and cast to integers
    if dst_width is None or dst_height is None:
        raise ValueError("Failed to calculate target dimensions for reprojection")
    dst_width = int(dst_width)
    dst_height = int(dst_height)

    reprojected_embedding = np.empty(
        (embedding.shape[2], dst_height, dst_width), dtype=embedding.dtype
    )

    # 4. Perform the reprojection band by band
    for band_idx in range(embedding.shape[2]):
        reproject(
            source=embedding[:, :, band_idx],
            destination=reprojected_embedding[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )


def save_reprojected_embedding_in_memory(
    reprojected_embedding, dst_transform, target_crs, dst_height, dst_width, embedding
):
    """
    Save a reprojected embedding to an in-memory file.
    """
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=embedding.shape[2],
        dtype=embedding.dtype,
        crs=target_crs,
        transform=dst_transform,
    ) as dataset:
        dataset.write(reprojected_embedding)


def merge_tiles(tiles_to_merge):
    """
    Merge tiles into a single raster.
    """
    print("\nPass 2: Merging all tiles...")
    if tiles_to_merge is None:
        raise ValueError("No tiles to merge")

    # check that tiles are in memory
    if not all(isinstance(tile, MemoryFile) for tile in tiles_to_merge):
        raise ValueError("Tiles must be in memory")

    merged_array, mosaic_transform = merge(tiles_to_merge)
    embedding_mosaic = np.transpose(merged_array, (1, 2, 0))  # (H, W, C)

    for src in tiles_to_merge:
        src.close()

    print(f"Final Embedding Mosaic Shape: {embedding_mosaic.shape}")
    return embedding_mosaic
