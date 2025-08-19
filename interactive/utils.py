# general
from typing import List, Optional, Tuple

import numpy as np
import rasterio

# geospatial
import sys
from geotessera import GeoTessera
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import from_bounds
from tqdm.auto import tqdm

# custom
from .config import Config, config  # import the global config instance

TESSERA = GeoTessera()


def define_roi(
    lat_coords: Optional[tuple] = None,
    lon_coords: Optional[tuple] = None,
    min_bbox_size: Optional[float] = None,
    max_bbox_size: Optional[float] = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Define region of interest using config defaults or user-provided values.

    Args:
        lat_coords (tuple): Tuple of latitude coordinates (min, max). Uses config default if None.
        lon_coords (tuple): Tuple of longitude coordinates (min, max). Uses config default if None.
        min_bbox_size (float): Minimum bounding box size. Uses config default if None.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: Tuple of validated (lat_coords, lon_coords)
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

    if check_bbox_valid(lat_coords, lon_coords, min_bbox_size, max_bbox_size):
        return (lat_coords, lon_coords)
    else:
        raise ValueError("Invalid bounding box coordinates provided")


def check_bbox_valid(
    lat_coords: tuple,
    lon_coords: tuple,
    min_bbox_size: Optional[float] = None,
    max_bbox_size: Optional[float] = None,
    verbose: bool = True,
) -> bool:
    """
    Validate that a bounding box meets size and coordinate constraints.

    Args:
        lat_coords (tuple): Tuple of latitude coordinates
        lon_coords (tuple): Tuple of longitude coordinates
        min_bbox_size (float): Minimum bbox size in degrees. Override config default if provided.
        max_bbox_size (float): Maximum bbox size in degrees. Override config default if provided.

    Returns:
        bool: True if bounding box is valid, False otherwise
    """
    # use config defaults if not provided
    min_bbox_size = min_bbox_size or config.min_bbox_size
    max_bbox_size = max_bbox_size or config.max_bbox_size

    # extract coordinate bounds
    lat_min, lat_max = min(lat_coords), max(lat_coords)
    lon_min, lon_max = min(lon_coords), max(lon_coords)

    # validate coordinate ranges
    if not (-90 <= lat_min <= lat_max <= 90):
        print("Latitude values must be between -90 and 90 degrees")
        return False
    if not (-180 <= lon_min <= lon_max <= 180):
        print("Longitude values must be between -180 and 180 degrees")
        return False

    # calculate bbox dimensions
    lat_size = lat_max - lat_min
    lon_size = lon_max - lon_min

    # check minimum size constraint
    if lat_size < min_bbox_size or lon_size < min_bbox_size:
        print(
            f"Bounding box too small. Minimum size is {min_bbox_size}° × {min_bbox_size}°, "
            f"but got {lat_size:.2f}° × {lon_size:.2f}°"
        ) if verbose else ""
        return False

    # check maximum size constraint
    if lat_size > max_bbox_size or lon_size > max_bbox_size:
        print(
            f"Bounding box too large. Maximum size is {max_bbox_size}° × {max_bbox_size}°, "
            f"but got {lat_size:.2f}° × {lon_size:.2f}°"
        ) if verbose else ""
        return False

    print(
        f"Bounding box defined:\n"
        f"┗ ({lat_min:.2f}°, {lon_min:.2f}°) ┓ ({lat_max:.2f}°, {lon_max:.2f}°)"
    ) if verbose else ""
    return True


class TesseraUtils:
    """Utility class for tessera-related operations with config integration."""

    def __init__(self, config_instance: Optional[Config] = None):
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
        # assign config values if overwrite values not provided
        target_year = (
            target_year if target_year is not None else self.config.target_year
        )

        # validate and get bounding box
        if not check_bbox_valid(lat_coords, lon_coords):
            raise ValueError("Invalid bounding box coordinates")
        lat_min, lat_max = min(lat_coords), max(lat_coords)
        lon_min, lon_max = min(lon_coords), max(lon_coords)
        bbox = (lon_min, lat_min, lon_max, lat_max)  # (min_lon, min_lat, max_lon, max_lat)

        print(f"\nSearching for tiles in ROI: {bbox} for year {target_year}")

        # Get tiles that would be available for this bbox (without downloading)
        tiles_to_download = self.tessera.registry.load_blocks_for_region(bbox, target_year)
        tiles_to_merge = [(lat, lon) for lon, lat in tiles_to_download]

        if not tiles_to_merge:
          raise ValueError(
            f"\nNo embedding tiles found for the specified ROI in year {target_year}"
          )
        print(f"\nFound {len(tiles_to_merge)} tiles to merge.")
        return tiles_to_merge

    def fetch_embedding_metadata(self, lat: float, lon: float, target_year: int):
        """Fetch embedding and associated metadata for a single tile."""
        embedding, src_crs, src_transform = self.tessera.fetch_embedding(lon, lat, year=target_year)
        
        src_height, src_width = embedding.shape[:2]
        from rasterio.transform import array_bounds
        src_bounds = array_bounds(src_height, src_width, src_transform)

        return (
            embedding,
            None,  # landmask_path no longer needed directly
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
        embedding: np.ndarray,
        src_crs: str,
        src_transform: Affine,
        src_height: int,
        src_width: int,
        src_bounds: tuple[float, float, float, float],
        target_crs: str,
    ) -> tuple[np.ndarray, Affine, int, int]:
        """
        Reproject an embedding to a target CRS.

        Args:
            embedding (np.ndarray): Embedding array of shape (H, W, C)
            src_crs (str): Source CRS
            src_transform (Affine): Source transform
            src_height (int): Source height
            src_width (int): Source width
            src_bounds (tuple[float, float, float, float]): Source bounds
            target_crs (str): Target CRS

        Returns:
            tuple[np.ndarray, Affine, int, int]: Reprojected embedding, transform, width, height
        """
        # calculate the parameters for reprojection
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

        # perform the reprojection band by band
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
        reprojected_embedding: np.ndarray,
        dst_transform: Affine,
        target_crs: str,
        dst_height: int,
        dst_width: int,
        embedding: np.ndarray,
    ) -> MemoryFile:
        """
        Save a reprojected embedding to an in-memory file.

        Args:
            reprojected_embedding (np.ndarray): Reprojected embedding array of shape (H, W, C)
            dst_transform (Affine): Destination transform
            target_crs (str): Target CRS
            dst_height (int): Destination height
            dst_width (int): Destination width
            embedding (np.ndarray): Original embedding array of shape (H, W, C)

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

    def merge_tiles(
        self, reprojected_tiles: List[MemoryFile]
    ) -> tuple[np.ndarray, Affine]:
        """
        Merge tiles into a single raster.

        Args:
            reprojected_tiles (List[MemoryFile]): List of MemoryFile objects

        Returns:
            tuple[np.ndarray, Affine]: Tuple of (embedding_mosaic, mosaic_transform)
        """
        print("\nMerging all tiles...")
        if not reprojected_tiles:
            raise ValueError("No tiles to merge")

        # open all memory files for merging
        datasets = []
        try:
            for memfile in reprojected_tiles:
                datasets.append(memfile.open())

            merged_array, mosaic_transform = merge(datasets)
            embedding_mosaic = np.transpose(merged_array, (1, 2, 0))  # (H, W, C)

        finally:
            # clean up
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
        Complete workflow: fetch embeddings and create mosaic using GeoTessera.

        Args:
            lat_coords (tuple): Tuple of latitude coordinates
            lon_coords (tuple): Tuple of longitude coordinates
            target_year (int): Override config year if provided

        Returns:
            Tuple of (embedding_mosaic, mosaic_transform)
        """
        target_year = target_year if target_year is not None else self.config.target_year
        
        # validate and get bounding box
        if not check_bbox_valid(lat_coords, lon_coords):
            raise ValueError("Invalid bounding box coordinates")
        lat_min, lat_max = min(lat_coords), max(lat_coords)
        lon_min, lon_max = min(lon_coords), max(lon_coords)
        bbox = (lon_min, lat_min, lon_max, lat_max)  # (min_lon, min_lat, max_lon, max_lat)

        print(f"\nFetching embeddings for ROI: {bbox} for year {target_year}")
        
        def progress_callback(current, total, status=None):
            if status:
                print(f"\r{status} ({current}/{total})", end="", flush=True)
            else:
                print(f"\rProgress: {current}/{total}", end="", flush=True)
            if current == total:
                print()  # New line when complete
        
        tiles_data = self.tessera.fetch_embeddings(bbox, target_year, progress_callback)
        
        if not tiles_data:
            raise ValueError(f"No embedding tiles found for the specified ROI in year {target_year}")
        
        print(f"Fetched {len(tiles_data)} tiles. Creating mosaic...")
        tiles_to_merge = [(lat, lon) for lon, lat, _, _, _ in tiles_data]
        reprojected_tiles = self.reproject_tessera_tiles(tiles_to_merge)
        embedding_mosaic, mosaic_transform = self.merge_tiles(reprojected_tiles)
        
        return embedding_mosaic, mosaic_transform
