from typing import List, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera
from pyproj import Transformer
from shapely.geometry import box


class GeoTesseraWrapper:
    def __init__(self, year: int = 2024):
        """
        Initializes the classifier manager.
        Args:
            year: The year of Tessera embeddings to use.
        """
        self.gt = GeoTessera()
        self.year = year

    def fetch_tiles_for_roi(self, roi_gdf: gpd.GeoDataFrame):
        # Find tiles intersecting the ROI
        if roi_gdf.crs != "EPSG:4326":
            roi_gdf = roi_gdf.to_crs("EPSG:4326")
        unified_geom = roi_gdf.unary_union

        tile_list = []
        for t_year, lat, lon in self.gt.list_available_embeddings():
            if t_year != self.year:
                continue
            if unified_geom.intersects(box(lon, lat, lon + 0.1, lat + 0.1)):
                tile_list.append((lat, lon))

        return tile_list

    def extract_embeddings_for_points(
        self, labeled_points: list
    ) -> Tuple[pd.DataFrame, Set[Tuple[float, float]]]:
        """
        Extracts Tessera embeddings by definitively testing each point against
        nearby candidate tiles.
        """
        print("Fetching list of all available tiles...")
        available_tiles = [
            (lat, lon)
            for t_year, lat, lon in self.gt.list_available_embeddings()
            if t_year == self.year
        ]
        if not available_tiles:
            raise ValueError(
                f"No tiles are available at all \
                for the year {self.year}."
            )
        print(
            f"Found {len(available_tiles)} total available tiles \
            for {self.year}."
        )

        training_data = []
        embedding_cache = {}
        processed_tiles = set()  # To store the unique (lat, lon) of tiles

        for i, point in enumerate(labeled_points):
            print(
                f"\nProcessing point {i+1}/{len(labeled_points)} at \
                    ({point['lat']:.4f}, {point['lon']:.4f})..."
            )

            point_lon, point_lat = point["lon"], point["lat"]
            candidate_tiles = [
                (t_lat, t_lon)
                for t_lat, t_lon in available_tiles
                if (abs(t_lat - point_lat) < 0.2 and abs(t_lon - point_lon) < 0.2)
            ]

            if not candidate_tiles:
                print(f" > Warning: No candidate tiles found near this point.")
                continue

            found_match = False
            for tile_lat, tile_lon in candidate_tiles:
                try:
                    landmask_path = self.gt._fetch_landmask(
                        lat=tile_lat, lon=tile_lon, progressbar=False
                    )
                    with rasterio.open(landmask_path) as tile_raster:
                        h, w = tile_raster.height, tile_raster.width

                        transformer = Transformer.from_crs(
                            "EPSG:4326", tile_raster.crs, always_xy=True
                        )
                        px, py = transformer.transform(point_lon, point_lat)
                        row, col = tile_raster.index(px, py)

                        if 0 <= row < h and 0 <= col < w:
                            # If we are here, this is the one and only correct tile.
                            #print(
                            #    f"  > Match found! Point belongs to tile ({tile_lat:.2f}, {tile_lon:.2f})."
                            #)

                            tile_key = (tile_lat, tile_lon)
                            if tile_key not in embedding_cache:
                                embedding_cache[tile_key] = self.gt.fetch_embedding(
                                    lat=tile_lat, lon=tile_lon, year=self.year
                                )

                            embedding_array = embedding_cache[tile_key]
                            embedding_vector = embedding_array[row, col]

                            training_data.append(
                                {"label": point["label"], "embedding": embedding_vector}
                            )
                            processed_tiles.add(tile_key)
                            found_match = True
                            # Crucially, break the inner loop once the match is found.
                            break
                        # ---------------------------------------------

                except Exception as e:
                    # This will now only catch unexpected errors, not out-of-bounds checks.
                    print(
                        f"  ! An unexpected error occurred while testing tile ({tile_lat:.2f}, {tile_lon:.2f}): {e}"
                    )

            if not found_match:
                print(
                    f"  > Warning: Could not find a definitive matching tile for this point after checking {len(candidate_tiles)} candidates."
                )

        if not training_data:
            print("\nNo embeddings were extracted.")
            return pd.DataFrame()

        print("\nExtraction complete. Creating DataFrame.")
        labels = [d["label"] for d in training_data]
        embeddings = np.array([d["embedding"] for d in training_data])
        df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(128)])
        df.insert(0, "label", labels)
        return df, processed_tiles
