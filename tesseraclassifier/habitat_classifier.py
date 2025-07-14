import concurrent.futures
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera  # Assuming core.py is in this module
from rasterio.merge import merge
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier

from tesseraclassifier.classifier_utils import *
from tesseraclassifier.geotessera_wrapper import GeoTesseraWrapper

# The functions to load and visualize data would remain separate utilities
# from your_utils import load_training_labels_from_json, visualize_with_basemap


class HabitatClassifier:
    """
    A class to manage the entire workflow of training a habitat classifier
    and applying it to a region of interest.
    """

    def __init__(
        self,
        labels: list,
        year: int = 2024,
    ):
        """
        Initializes the classifier manager.
        Args:
            year: The year of Tessera embeddings to use.
        """
        self.gt = GeoTesseraWrapper(year)
        self.year = year
        self.model = None
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.prepare_training_data(labels)
        print(f"Classifier initialized for year {self.year}.")

    def prepare_training_data(self, labeled_points: list) -> pd.DataFrame:
        """
        Takes a list of labeled points, extracts their embeddings, and prepares
        a DataFrame for training.
        """
        print("\n--- Step 1: Preparing Training Data ---")
        # This part re-uses the robust point extraction logic we developed
        training_df, tessera_tiles = self.gt.extract_embeddings_for_points(
            labeled_points
        )

        if training_df.empty:
            raise ValueError("Could not extract any training data.")

        # Create and store the class mappings
        unique_labels = sorted(training_df["label"].unique())
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        print(f"Successfully prepared {len(training_df)} training samples.")
        print("Class mapping created:")
        print(self.label_to_id)

        self.training_df = training_df
        self.training_tiles = tessera_tiles

    def train(self, min_samples_per_class: int = 2):
        """
        Trains a RandomForestClassifier on the provided training data.
        """
        print("\n--- Step 2: Training Model ---")
        if not self.label_to_id:
            raise RuntimeError("Must run prepare_training_data before training.")

        # Clean data by removing rare classes
        class_counts = self.training_df["label"].value_counts()
        classes_to_remove = class_counts[class_counts < min_samples_per_class].index
        df_filtered = self.training_df[
            ~self.training_df["label"].isin(classes_to_remove)
        ]

        X_train = df_filtered.drop("label", axis=1)
        y_train_str = df_filtered["label"]
        y_train_int = y_train_str.map(self.label_to_id)

        print(f"Training model on {len(X_train)} samples...")
        self.model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train.values, y_train_int.values)
        print("✅ Model training complete.")

    def classify_tiles(
        self,
        tmp_dir: str,
        tile_list: Optional[List[Tuple[float, float]]] = None,
        max_workers: Optional[int] = None,
        dry_run: bool = False,
    ):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Run .train() first.")

        if not tile_list:
            raise ValueError("No tiles found to classify for the given input.")

        print(f"Found {len(tile_list)} tiles to process.")
        if dry_run:
            return

        Path(tmp_dir).mkdir(exist_ok=True, parents=True)
        tasks = [
            (lat, lon, self.year, self.model, tmp_dir, self.gt.gt)
            for lat, lon in tile_list
        ]

        print(
            f"\nStarting parallel classification using up to {max_workers or 'all'} CPU cores..."
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            results = list(executor.map(classify_single_tile_worker, tasks))

        successful_files = [r for r in results if r]
        print(
            f"\nParallel processing complete. {len(successful_files)} of {len(tasks)} tiles were successfully classified."
        )
        print(f"Intermediate classified tiles saved in: {tmp_dir}")

    def classify_region(
        self,
        tmp_dir: str,
        roi_gdf: gpd.GeoDataFrame,
        max_workers: Optional[int] = None,
        dry_run: bool = False,
    ):
        """
        Classifies a large region of interest defined by a GeoDataFrame.
        """
        tile_list = self.gt.fetch_tiles_for_roi(roi_gdf)

        if not tile_list:
            raise ValueError("No tiles found in the specified ROI.")

        self.classify_tiles(tmp_dir, tile_list, max_workers, dry_run)

    def mosaic_classification(
        self, tmp_dir: str, output_path: str, cleanup: bool = False
    ):
        stitch_classification_tiles(tmp_dir, output_path, cleanup)
