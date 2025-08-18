from typing import List, Optional, Set, Tuple, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera
from pyproj import Transformer
from shapely.geometry import box, Point
from rasterio.transform import rowcol


class GeoTesseraWrapper:
    """
    A wrapper around GeoTessera to provide convenient methods for 
    extracting embeddings from labeled points and ROI-based tile fetching.
    """
    
    def __init__(self, year: int = 2024):
        """Initialize wrapper with a specific year."""
        self.gt = GeoTessera()
        self.year = year

    def fetch_tiles_for_roi(self, roi_gdf: gpd.GeoDataFrame):
        """Fetch tiles for a region of interest using the new bbox-based API."""
        if roi_gdf.crs != "EPSG:4326":
            roi_gdf = roi_gdf.to_crs("EPSG:4326")
        
        # Get bounding box from ROI
        bounds = roi_gdf.total_bounds  # [min_x, min_y, max_x, max_y]
        bbox = (bounds[0], bounds[1], bounds[2], bounds[3])  # (min_lon, min_lat, max_lon, max_lat)
        
        print(f"Fetching tiles in bbox: {bbox}")
        
        # Use new API to get tiles in bbox - returns (tile_lon, tile_lat, ...)
        tiles = self.gt.fetch_embeddings(bbox, self.year)
        tile_list = [(lat, lon) for lon, lat, _, _, _ in tiles]
        
        print(f"Found {len(tile_list)} tiles in ROI")
        return tile_list

    def extract_embeddings_for_points(
        self, labeled_points: list
    ) -> Tuple[pd.DataFrame, Set[Tuple[float, float]]]:
        """
        Extracts Tessera embeddings using batch processing for efficient tile fetching.
        Uses the new API to fetch all needed tiles at once based on point locations.
        """
        if not labeled_points:
            print("No labeled points provided.")
            return pd.DataFrame(), set()
        
        print(f"Processing {len(labeled_points)} labeled points using batch processing...")
        
        # Calculate bounding box for all points
        point_coords = [(p['lat'], p['lon']) for p in labeled_points]
        lats = [coord[0] for coord in point_coords]
        lons = [coord[1] for coord in point_coords]
        
        # Add small buffer to ensure we get all relevant tiles
        buffer = 0.05  # ~5km buffer
        bbox = (min(lons) - buffer, min(lats) - buffer, 
                max(lons) + buffer, max(lats) + buffer)
        
        print(f"Fetching tiles in bbox: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})")
        
        # Fetch all needed tiles at once using new batch API - returns (tile_lon, tile_lat, ...)
        try:
            tiles_data = self.gt.fetch_embeddings(bbox, self.year)
        except Exception as e:
            print(f"Error fetching tiles: {e}")
            return pd.DataFrame(), set()
        
        if not tiles_data:
            print("No tiles found in the specified region.")
            return pd.DataFrame(), set()
        
        print(f"Successfully fetched {len(tiles_data)} tiles")
        
        # Create spatial index of tiles with their data
        tile_cache = {}
        for tile_lon, tile_lat, embedding, crs, transform in tiles_data:
            tile_key = (tile_lat, tile_lon)
            tile_cache[tile_key] = {
                'embedding': embedding,
                'crs': crs,
                'transform': transform
            }
        
        # Process all points against the cached tiles
        training_data = []
        processed_tiles = set()
        
        for i, point in enumerate(labeled_points):
            if i % 50 == 0:  # Progress update every 50 points
                print(f"Processing point {i+1}/{len(labeled_points)}...")
            
            point_lat, point_lon = point['lat'], point['lon']
            found_match = False
            
            # Check each cached tile to find which contains this point
            for tile_key, tile_data in tile_cache.items():
                tile_lat, tile_lon = tile_key
                embedding = tile_data['embedding']
                crs = tile_data['crs']
                transform = tile_data['transform']
                
                try:
                    # Transform point coordinates to tile's CRS
                    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                    px, py = transformer.transform(point_lon, point_lat)
                    
                    # Get pixel coordinates
                    row, col = rowcol(transform, px, py)
                    
                    # Check if point is within tile bounds
                    h, w = embedding.shape[:2]
                    if 0 <= row < h and 0 <= col < w:
                        # Extract embedding vector for this point
                        embedding_vector = embedding[row, col]
                        
                        training_data.append({
                            'label': point['label'],
                            'embedding': embedding_vector,
                            'lat': point_lat,
                            'lon': point_lon,
                            'tile_lat': tile_lat,
                            'tile_lon': tile_lon
                        })
                        
                        processed_tiles.add(tile_key)
                        found_match = True
                        break  # Found the correct tile, no need to check others
                        
                except Exception as e:
                    # Skip this tile if there's an error (e.g., CRS transformation issues)
                    continue
            
            if not found_match:
                print(f"Warning: No matching tile found for point at ({point_lat:.4f}, {point_lon:.4f})")
        
        if not training_data:
            print("No embeddings were extracted.")
            return pd.DataFrame(), set()
        
        print(f"\nExtraction complete! Extracted {len(training_data)} embeddings from {len(processed_tiles)} tiles.")
        
        # Create DataFrame
        labels = [d['label'] for d in training_data]
        embeddings = np.array([d['embedding'] for d in training_data])
        
        # Create DataFrame with embedding columns
        df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(128)])
        df.insert(0, "label", labels)
        
        # Add optional metadata columns
        df['point_lat'] = [d['lat'] for d in training_data]
        df['point_lon'] = [d['lon'] for d in training_data]
        df['tile_lat'] = [d['tile_lat'] for d in training_data]
        df['tile_lon'] = [d['tile_lon'] for d in training_data]
        
        return df, processed_tiles