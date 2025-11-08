# general
from typing import Optional, Tuple

import numpy as np

# geospatial
from geotessera import GeoTessera

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
        self.config = config_instance or config
        self.tessera = GeoTessera(embeddings_dir=self.config.embeddings_dir)

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

        embedding_mosaic, mosaic_transform, crs = self.tessera.fetch_mosaic_for_region(
          bbox=bbox,
          year=target_year,
          target_crs=self.config.target_crs,
          auto_download=True,
          progress_callback=progress_callback
        )
        print(f"Mosaic created: shape={embedding_mosaic.shape}, crs={crs}")

        return embedding_mosaic, mosaic_transform
