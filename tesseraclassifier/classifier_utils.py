import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from rasterio.warp import Resampling, calculate_default_transform, reproject

import numpy as np
import pandas as pd
import rasterio
from geopandas import gpd
from pyproj import Transformer
from rasterio.merge import merge
from shapely.geometry import box
import tempfile


def stitch_classification_tiles(
    source_dir: str, output_path: str, max_workers: int = 20, cleanup: bool = False
):
    """
    Stitches a directory of classified tiles by reprojecting them in parallel
    and then merging the results.
    """
    print("LALAL")
    print(max_workers)
    native_files = list(Path(source_dir).glob("*.tif"))
    if not native_files:
        print("No intermediate files found to stitch.")
        return

    # 1. Determine the target CRS from the first file
    with rasterio.open(native_files[0]) as first_src:
        target_crs = first_src.crs
    print(f"Using {target_crs} as the common projection for stitching.")

    # 2. Reproject all tiles in parallel to a new temporary directory
    reprojected_dir = tempfile.mkdtemp(prefix="reprojected_")
    tasks = [(str(f), reprojected_dir, target_crs) for f in native_files]
    
    print(f"Starting parallel reprojection of {len(tasks)} tiles...")
    reprojected_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = executor.map(reproject_single_tile, tasks)
        reprojected_files = [r for r in results if r]

    if not reprojected_files:
        raise RuntimeError("Reprojection failed for all tiles.")

    # 3. Stitch the now-aligned tiles
    print(f"\nStitching {len(reprojected_files)} aligned tiles into final map...")
    src_files_to_mosaic = [rasterio.open(path) for path in reprojected_files]
    try:
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": target_crs,  # All files are merged into the CRS of the first file
            "count": 1,
            "dtype": mosaic.dtype,  # Use the dtype of the actual merged data
            "nodata": src_files_to_mosaic[0].nodata,  # Preserve the nodata value
            "compress": "lzw",
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        print(f"✅ Successfully created final classification map: {output_path}")
    finally:
        for src in src_files_to_mosaic:
            src.close()
        shutil.rmtree(reprojected_dir)
        if cleanup:
            shutil.rmtree(source_dir)


def reproject_single_tile(args: tuple):
    """
    Worker function (runs in parallel). Reprojects a single tile to the target CRS.
    """
    input_path, output_dir, target_crs = args
    try:
        with rasterio.open(input_path) as src:
            if str(src.crs) == str(target_crs):
                # If CRS already matches, just copy the file
                output_path = Path(output_dir) / Path(input_path).name
                shutil.copy(input_path, output_path)
                return str(output_path)

            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
            
            output_path = Path(output_dir) / Path(input_path).name
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
            return str(output_path)
    except Exception as e:
        # --- THIS IS THE FIX ---
        # Print the error to understand why a tile failed, then return None.
        print(f"  ! Failed to reproject {Path(input_path).name}: {e}")
        return None


# --- The Worker Function (expects 6 arguments) ---
def classify_single_tile_worker(args: tuple):
    """
    Worker function executed by each parallel process. It classifies one tile.
    """
    # This unpacks the 6 arguments passed from the main function
    tile_lat, tile_lon, year, model, output_dir, gt = args
    try:
        # Fetch and classify
        embedding_array = gt.fetch_embedding(
            lat=tile_lat, lon=tile_lon, year=year, progressbar=False
        )
        h, w, c = embedding_array.shape
        class_map = model.predict(embedding_array.reshape(-1, c)).reshape(h, w)

        # Convert to a standard, saveable data type
        class_map = class_map.astype(np.uint8)

        # Get georeferencing info
        landmask_path = gt._fetch_landmask(
            lat=tile_lat, lon=tile_lon, progressbar=False
        )
        with rasterio.open(landmask_path) as landmask_src:
            src_crs, src_transform = landmask_src.crs, landmask_src.transform

        # Save the file
        output_path = Path(output_dir) / f"classified_{tile_lat:.2f}_{tile_lon:.2f}.tif"
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=class_map.dtype,
            crs=src_crs,
            transform=src_transform,
            compress="lzw",
        ) as dst:
            dst.write(class_map, 1)

        return str(output_path)
    except Exception as e:
        print(f"  ! Worker for tile ({tile_lat:.2f}, {tile_lon:.2f}) failed: {e}")
        return None


def load_training_labels_from_json(file_path: str) -> list:
    """
    Loads labeled points from a JSON file in the new WGS84 format.

    Args:
        file_path: The path to the JSON file (e.g., 'labels.json').

    Returns:
        A list of dictionaries, with each dictionary representing a labeled point.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    raw_points = data.get("training_points", [])
    
    processed_points = []
    for item in raw_points:
        # The new format is [[lat, lon], label]
        coord = item[0]
        label = item[1]

        processed_points.append(
            {
                "lat": coord[0],
                "lon": coord[1],
                "crs": "EPSG:4326",  # The CRS is already WGS84
                "label": label,
            }
        )

    return processed_points


def load_visualization_mappings(
    json_file_path: str,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Loads the necessary color and label mappings from a JSON file for visualization.
    This version is updated for the new JSON format.
    """
    print("--- Loading Mappings for Visualization ---")

    with open(json_file_path, "r") as f:
        data = json.load(f)

    # The new format has a simpler structure for training_points
    raw_points = data.get("training_points", [])
    all_labels = [item[1] for item in raw_points]

    unique_labels = sorted(list(set(all_labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("Generated class mappings from all labels in the file.")

    class_colors = data.get("class_color_map", {})
    print(f"Loaded {len(class_colors)} color definitions.")

    print("✅ Mappings ready for visualization.")

    return label_to_id, id_to_label, class_colors


def get_utm_crs(lon: float, lat: float) -> str:
    """Calculates the appropriate UTM Zone CRS for a given lat/lon."""
    utm_band = str((int((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        return 'EPSG:326' + utm_band
    else:
        return 'EPSG:327' + utm_band
        

def roi_from_points(points: List[Dict], buffer_km: float = 1.0) -> gpd.GeoDataFrame:
    """
    Creates a buffered bounding box GeoDataFrame around a list of points.

    Args:
        points: A list of dictionaries, where each dict has 'lat' and 'lon' keys.
        buffer_km: The buffer to add around the bounding box in kilometers.

    Returns:
        A GeoDataFrame containing a single polygon feature for the buffered ROI,
        in WGS84 (EPSG:4326).
    """
    if not points:
        raise ValueError("Input points list cannot be empty.")

    # 1. Create a GeoDataFrame from the input points (assumed to be WGS84)
    points_df = pd.DataFrame(points)
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.lon, points_df.lat),
        crs="EPSG:4326"
    )

    # 2. Determine a suitable projected CRS (UTM) for accurate buffering
    # We use the centroid of all points to find the right UTM zone.
    centroid = points_gdf.unary_union.centroid
    utm_crs = get_utm_crs(centroid.x, centroid.y)
    print(f"Projecting points to a suitable local CRS: {utm_crs}")

    # 3. Reproject the points to the meter-based UTM system
    points_projected = points_gdf.to_crs(utm_crs)

    # 4. Get the bounding box of the points in the projected CRS
    min_x, min_y, max_x, max_y = points_projected.total_bounds

    # 5. Apply the buffer in meters (1 km = 1000 meters)
    buffer_m = buffer_km * 1000
    roi_bounds_projected = (
        min_x - buffer_m,
        min_y - buffer_m,
        max_x + buffer_m,
        max_y + buffer_m,
    )

    # 6. Create the buffered polygon and a new GeoDataFrame
    roi_poly_projected = box(*roi_bounds_projected)
    roi_gdf_projected = gpd.GeoDataFrame([1], geometry=[roi_poly_projected], crs=utm_crs)

    # 7. Convert the final ROI back to WGS84 for consistency
    roi_gdf_wgs84 = roi_gdf_projected.to_crs("EPSG:4326")
    
    return roi_gdf_wgs84
