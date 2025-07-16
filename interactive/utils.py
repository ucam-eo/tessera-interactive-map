# general
from typing import List, Optional, Tuple

import numpy as np
import rasterio

# geospatial
from geotessera import GeoTessera
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from tqdm.auto import tqdm

# custom
from .config import config  # Import the global config instance

TESSERA = GeoTessera()


def define_roi(
    lat_coords: Optional[tuple] = None,
    lon_coords: Optional[tuple] = None,
    min_bbox_size: Optional[float] = None,
):
    """
    Define region of interest using config defaults or user-provided values.

    Args:
        lat_coords: Tuple of latitude coordinates (min, max). Uses config default if None.
        lon_coords: Tuple of longitude coordinates (min, max). Uses config default if None.
        min_bbox_size: Minimum bounding box size. Uses config default if None.

    Returns:
        Tuple of validated (lat_coords, lon_coords)
    """
    if (
        lat_coords is None
        or lon_coords is None
        or None in lat_coords
        or None in lon_coords
    ):
        print(
            f"Using config defaults for lat_coords: {config.lat_coords} and lon_coords: {config.lon_coords}"
        )
        lat_coords = config.lat_coords
        lon_coords = config.lon_coords
    else:
        print(
            f"Using user-provided lat_coords: {lat_coords} and lon_coords: {lon_coords}"
        )
    lat_coords = lat_coords if lat_coords is not None else config.lat_coords
    lon_coords = lon_coords if lon_coords is not None else config.lon_coords

    min_bbox_size = min_bbox_size if min_bbox_size is not None else config.min_bbox_size

    return check_bbox(lon_coords, lat_coords, min_bbox_size)


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

    print(
        f"Bounding box defined:\n┗ ({lat_min:.2f}, {lon_min:.2f}) | ┓ ({lat_max:.2f}, {lon_max:.2f})"
    )
    return (lat_min, lat_max), (lon_min, lon_max)


class TesseraUtils:
    """Utility class for tessera-related operations with config integration."""

    def __init__(self, config_instance=None):
        """Initialize with optional config instance."""
        self.tessera = GeoTessera()
        self.config = config_instance or config

    def check_tessera_tiles(
        self, lat_coords: tuple, lon_coords: tuple, target_year: Optional[int] = None
    ) -> List[Tuple[float, float]]:
        """
        Check that the tiles are within the bounds of the Tessera tiles.

        Args:
            lat_coords: Tuple of latitude coordinates
            lon_coords: Tuple of longitude coordinates
            target_year: Override config year if provided

        Returns:
            List of (lat, lon) tuples for tiles to merge
        """
        # Use config values
        target_year = (
            target_year if target_year is not None else self.config.target_year
        )

        # Validate and get bounding box
        (lat_min, lat_max), (lon_min, lon_max) = check_bbox(lat_coords, lon_coords)
        roi_bounds = (
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        )  # (min_lon, min_lat, max_lon, max_lat)

        print(f"\nSearching for tiles in ROI: {roi_bounds} for year {target_year}")

        # --- Find tiles in ROI ---
        tiles_to_merge = []
        for available_year in self.tessera.get_available_years():
            self.tessera._ensure_year_loaded(available_year)

        # search tessera for embeddings within ROI
        for emb_year, lat, lon in self.tessera.list_available_embeddings():
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
                f"\nNo embedding tiles found for the specified ROI in year {target_year}"
            )
        print(f"\nFound {len(tiles_to_merge)} tiles to merge.")
        return tiles_to_merge

    def fetch_embedding_metadata(self, lat: float, lon: float, target_year: int):
        """Fetch embedding and associated metadata for a single tile."""
        embedding = self.tessera.get_embedding(lat, lon, year=target_year)  # (H, W, C)
        landmask_path = self.tessera._fetch_landmask(lat, lon, progressbar=False)
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

    def reproject_tessera_tiles(
        self, tiles_to_merge: List[Tuple[float, float]]
    ) -> List[MemoryFile]:
        """
        Reproject tessera tiles and return list of MemoryFile objects.

        Args:
            tiles_to_merge: List of (lat, lon) tuples

        Returns:
            List of MemoryFile objects containing reprojected tiles
        """
        reprojected_tiles = []

        for _, (lat, lon) in tqdm(
            enumerate(tiles_to_merge),
            total=len(tiles_to_merge),
            desc="Processing tiles",
        ):
            try:
                (
                    embedding,
                    landmask_path,
                    src_crs,
                    src_transform,
                    src_height,
                    src_width,
                    src_bounds,
                ) = self.fetch_embedding_metadata(lat, lon, self.config.target_year)

                reprojected_embedding, dst_transform, dst_width, dst_height = (
                    self.reproject_embedding(
                        embedding,
                        src_crs,
                        src_transform,
                        src_height,
                        src_width,
                        src_bounds,
                        self.config.target_crs,
                    )
                )

                memory_file = self.save_reprojected_embedding_in_memory(
                    reprojected_embedding,
                    dst_transform,
                    self.config.target_crs,
                    dst_height,
                    dst_width,
                    embedding,
                )
                reprojected_tiles.append(memory_file)

            except Exception as e:
                print(
                    f"\t! WARNING: Could not process tile ({lat:.2f}, {lon:.2f}): {e}"
                )
                continue

        return reprojected_tiles

    def reproject_embedding(
        self,
        embedding,
        src_crs,
        src_transform,
        src_height,
        src_width,
        src_bounds,
        target_crs,
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

        return reprojected_embedding, dst_transform, dst_width, dst_height

    def save_reprojected_embedding_in_memory(
        self,
        reprojected_embedding,
        dst_transform,
        target_crs,
        dst_height,
        dst_width,
        embedding,
    ) -> MemoryFile:
        """
        Save a reprojected embedding to an in-memory file.

        Returns:
            MemoryFile object containing the reprojected data
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

        return memfile

    def merge_tiles(self, reprojected_tiles: List[MemoryFile]):
        """
        Merge tiles into a single raster.

        Args:
            reprojected_tiles: List of MemoryFile objects

        Returns:
            Tuple of (embedding_mosaic, mosaic_transform)
        """
        print("\nMerging all tiles...")
        if not reprojected_tiles:
            raise ValueError("No tiles to merge")

        # Open all memory files for merging
        datasets = []
        try:
            for memfile in reprojected_tiles:
                datasets.append(memfile.open())

            merged_array, mosaic_transform = merge(datasets)
            embedding_mosaic = np.transpose(merged_array, (1, 2, 0))  # (H, W, C)

        finally:
            # Clean up
            for dataset in datasets:
                dataset.close()
            for memfile in reprojected_tiles:
                memfile.close()

        print(f"Shape of final embedding mosaic: {embedding_mosaic.shape}")
        return embedding_mosaic, mosaic_transform

    def process_roi_to_mosaic(
        self, lat_coords: tuple, lon_coords: tuple, target_year: Optional[int] = None
    ) -> Tuple[np.ndarray, object]:
        """
        Complete workflow: check tiles, reproject, and merge into mosaic.

        Args:
            lat_coords: Tuple of latitude coordinates
            lon_coords: Tuple of longitude coordinates
            target_year: Override config year if provided

        Returns:
            Tuple of (embedding_mosaic, mosaic_transform)
        """
        # Step 1: Find tiles in ROI
        tiles_to_merge = self.check_tessera_tiles(lat_coords, lon_coords, target_year)

        # Step 2: Reproject tiles
        reprojected_tiles = self.reproject_tessera_tiles(tiles_to_merge)

        # Step 3: Merge tiles
        embedding_mosaic, mosaic_transform = self.merge_tiles(reprojected_tiles)

        return embedding_mosaic, mosaic_transform


# # Backward compatibility functions using global tessera instance
# TESSERA = GeoTessera()


# def check_tessera_tiles(
#     lat_coords: tuple, lon_coords: tuple, target_year: Optional[int] = None
# ) -> list[tuple[float, float]]:
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.check_tessera_tiles(lat_coords, lon_coords, target_year)


# def fetch_embedding_metadata(lat, lon, target_year):
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.fetch_embedding_metadata(lat, lon, target_year)


# def reproject_tessera_tiles(tiles_to_merge):
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.reproject_tessera_tiles(tiles_to_merge)


# def reproject_embedding(
#     embedding, src_crs, src_transform, src_height, src_width, src_bounds, target_crs
# ):
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.reproject_embedding(
#         embedding, src_crs, src_transform, src_height, src_width, src_bounds, target_crs
#     )


# def save_reprojected_embedding_in_memory(
#     reprojected_embedding, dst_transform, target_crs, dst_height, dst_width, embedding
# ):
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.save_reprojected_embedding_in_memory(
#         reprojected_embedding,
#         dst_transform,
#         target_crs,
#         dst_height,
#         dst_width,
#         embedding,
#     )


# def merge_tiles(tiles_to_merge):
#     """Backward compatibility wrapper."""
#     tessera_utils = TesseraUtils()
#     return tessera_utils.merge_tiles(tiles_to_merge)


# # Create global instance for backward compatibility
# tessera_utils = TesseraUtils()
