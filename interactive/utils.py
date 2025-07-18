# general
from typing import List, Optional, Tuple

import numpy as np
import rasterio

# geospatial
from geotessera import GeoTessera
from geotessera.registry_utils import get_all_blocks_in_range
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
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
        roi_bounds = (
            lon_min,
            lat_min,
            lon_max,
            lat_max,
        )  # (min_lon, min_lat, max_lon, max_lat)

        print(f"\nSearching for tiles in ROI: {roi_bounds} for year {target_year}")

        # find all registry blocks that intersect the ROI
        intersecting_blocks = get_all_blocks_in_range(lon_min, lon_max, lat_min, lat_max)
        print(f"ROI intersects with {len(intersecting_blocks)} registry block(s). Loading them...")

        # lazily load the registry for each intersecting block
        # this populates the internal list of available embeddings in the GeoTessera object
        for block_lon, block_lat in intersecting_blocks:
            try:
                # the _ensure_block_loaded method needs any coordinate within the block
                # the block's lower-left corner coordinate works for this
                self.tessera._ensure_block_loaded(year=target_year, lon=block_lon, lat=block_lat)
            except Exception as e:
                # this might happen if a block registry is missing on the server for the given year
                print(f"Warning: Could not load block for ({block_lon}, {block_lat}) in year {target_year}: {e}")

        print("Required registry blocks loaded.")

        # find tiles in ROI by searching the now-populated list of available embeddings
        tiles_to_merge = []
        for emb_year, lat, lon in self.tessera._available_embeddings:
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
        Complete workflow: check tiles, reproject, and merge into mosaic.

        Args:
            lat_coords (tuple): Tuple of latitude coordinates
            lon_coords (tuple): Tuple of longitude coordinates
            target_year (int): Override config year if provided

        Returns:
            Tuple of (embedding_mosaic, mosaic_transform)
        """
        # find tiles in ROI
        tiles_to_merge = self.check_tessera_tiles(lat_coords, lon_coords, target_year)

        # reproject tiles
        reprojected_tiles = self.reproject_tessera_tiles(tiles_to_merge)

        # merge tiles
        embedding_mosaic, mosaic_transform = self.merge_tiles(reprojected_tiles)

        return embedding_mosaic, mosaic_transform
