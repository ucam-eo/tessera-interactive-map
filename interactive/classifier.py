import base64
import io
from typing import Callable, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from rasterio import Affine, transform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def format_number(num: int) -> str:
    """
    Format a number with appropriate suffix (k, M, B) for readability.

    Args:
        num (int): Number to format

    Returns:
        str: Formatted number with suffix
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f}k"
    else:
        return str(num)


class EmbeddingClassifier:
    """
    A classifier that uses tessera embeddings to perform pixel-level classification
    on satellite imagery mosaics.
    """

    def __init__(self, embedding_mosaic: np.ndarray, mosaic_transform: Affine):
        """
        Initialize classifier with embedding data.

        Args:
            embedding_mosaic (np.ndarray): 3D numpy array of shape (height, width, channels) containing embeddings
            mosaic_transform (Affine): Rasterio transform for the mosaic
        """
        self.embedding_mosaic = embedding_mosaic
        self.mosaic_transform = mosaic_transform
        self.mosaic_height, self.mosaic_width, self.num_channels = (
            embedding_mosaic.shape
        )
        self.model = None
        self.model_name = None
        self.class_index_map = {}
        self.unique_class_names = []

    def prepare_training_data(
        self, training_points: list[tuple[tuple[float, float], str]]
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Prepare training data from labeled points.

        Args:
            training_points (list[tuple[tuple[float, float], str]]): ((lat, lon), class_name)

        Returns:
            tuple[np.ndarray, np.ndarray, dict]: (X_train, y_train, validation_info)
        """
        X_train, y_train = [], []
        skipped_points = []

        # create mapping from class names to integer labels
        self.unique_class_names = sorted(
            list(set(name for point, name in training_points))
        )
        self.class_index_map = {
            name: i for i, name in enumerate(self.unique_class_names)
        }

        # map training points to pixel coordinates
        for (lat, lon), class_name in training_points:
            row, col = transform.rowcol(self.mosaic_transform, lon, lat)
            if 0 <= row < self.mosaic_height and 0 <= col < self.mosaic_width:
                X_train.append(self.embedding_mosaic[row, col, :])
                y_train.append(self.class_index_map[class_name])
            else:
                skipped_points.append((lat, lon, class_name))

        validation_info = {
            "total_points": len(training_points),
            "valid_points": len(X_train),
            "skipped_points": skipped_points,
            "unique_classes": self.unique_class_names,
        }

        return np.array(X_train), np.array(y_train), validation_info

    def train_classifier(
        self, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "knn", model_params: Optional[dict] = None
    ) -> int:
        """
        Train a specified classifier.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            model_name (str): The name of the model to use ('knn', 'rf', etc.)
            model_params (dict): Optional parameters to pass to the model constructor.
        """
        model_params = model_params or {}
        self.model_name = model_name

        if model_name == 'knn':
            k = model_params.get('k', min(5, len(X_train)))
            print(f"Training k-NN classifier with k={k}...")
            self.model = KNeighborsClassifier(n_neighbors=k, weights="distance")
        
        elif model_name == 'rf':
            n_estimators = model_params.get('n_estimators', 100)
            print(f"Training Random Forest with {n_estimators} estimators...")
            self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model.fit(X_train, y_train)

        print("Model training complete.")

    def classify_mosaic(
        self, batch_size: int = 15000, progress_callback: Optional[Callable] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Classify the entire mosaic using the trained model.

        Args:
            batch_size (int): Size of batches for processing
            progress_callback (Callable): Optional callback function for progress updates

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: 
                - classification_result of shape (height, width)
                - confidence_map of shape (height, width)
        """
        if self.model is None:
            raise ValueError("Model must be trained before classification")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"The selected model '{self.model_name}' does not support probability estimates.")
        # reshape array to 2D for batch processing
        all_pixels = self.embedding_mosaic.reshape(-1, self.num_channels)
        n_pixels = all_pixels.shape[0]

        all_probabilities = np.zeros((n_pixels, len(self.unique_class_names)), dtype=np.float32)

        # process in batches with progress tracking
        total_formatted = format_number(n_pixels)
        with tqdm(
            total=n_pixels,
            desc=f"Classifying {total_formatted} pixels",
            unit="px",
            unit_scale=True,
            unit_divisor=1000,
        ) as pbar:
            for i in range(0, n_pixels, batch_size):
                end = min(i + batch_size, n_pixels)
                all_probabilities[i:end] = self.model.predict_proba(all_pixels[i:end, :])
                pbar.update(end - i)

                if progress_callback:
                    progress_callback(i + (end - i), n_pixels)

        # reshape back to image dimensions for visualization
        classification_result = np.argmax(all_probabilities, axis=1)
        confidence_map = np.max(all_probabilities, axis=1)

        # Reshape back to image dimensions
        classification_result = classification_result.reshape(self.mosaic_height, self.mosaic_width)
        confidence_map = confidence_map.reshape(self.mosaic_height, self.mosaic_width)

        # clean up variable to save memory
        del all_pixels, all_probabilities
        return classification_result, confidence_map

    def create_visualization(
        self,
        classification_result: np.ndarray,
        color_map: dict[str, str],
        confidence_map: Optional[np.ndarray] = None,
        mode: str = 'standard',
        threshold: float = 0.7
    ) -> str:
        """
        Create visualization colored by class of results.

        Args:
            confidence_map (np.ndarray): 2D array of model confidence scores (0.0 to 1.0)
            mode (str): standard, confidence_opacity, or threshold
            threshold (float): Confidence threshold for the threshold mode
        Returns:
            str: Base64-encoded PNG image data URL
        """
        # create colormap from the color mapping
        color_list = [
            color_map.get(name, "#888888") for name in self.unique_class_names
        ]
        cmap = mcolors.ListedColormap(color_list)
        norm = mcolors.Normalize(vmin=0, vmax=len(self.unique_class_names) - 1)
        colored_result_rgb = cmap(norm(classification_result))[:, :, :3]

        if mode == 'confidence_opacity' and confidence_map is not None:
            # Use confidence as the alpha channel. High confidence = opaque.
            alpha_channel = (confidence_map * 255).astype(np.uint8)
            # Add the alpha channel to the RGB image
            rgba_image = np.dstack((colored_result_rgb * 255, alpha_channel)).astype(np.uint8)

        elif mode == 'threshold' and confidence_map is not None:
            # Create a special color for uncertain pixels (e.g., grey)
            uncertain_color = np.array([0.5, 0.5, 0.5]) # Grey
            # Where confidence is low, replace the color with the uncertain color
            colored_result_rgb[confidence_map < threshold] = uncertain_color
            rgba_image = (colored_result_rgb * 255).astype(np.uint8)

        else: # Standard mode
            rgba_image = (colored_result_rgb * 255).astype(np.uint8)

        # convert to base64 PNG for saving
        buffer = io.BytesIO()
        plt.imsave(buffer, rgba_image, format="png")
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")

        return f"data:image/png;base64,{b64_data}"

    def validate_training_points(
        self,
        training_points: list[tuple[tuple[float, float], str]],
        min_points: int = 2,
        min_classes: int = 2,
    ) -> tuple[bool, str]:
        """
        Validate that training points are sufficient for classification.

        Args:
            training_points (list[tuple[tuple[float, float], str]]): List of training points
            min_points (int): Minimum number of points required
            min_classes (int): Minimum number of classes required

        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        if len(training_points) < min_points:
            return (
                False,
                f"Need at least {min_points} training points, got {len(training_points)}",
            )

        unique_classes = set(class_name for point, class_name in training_points)
        if len(unique_classes) < min_classes:
            return (
                False,
                f"Need at least {min_classes} different classes, got {len(unique_classes)}",
            )

        return True, ""

    def get_classification_stats(self, classification_result: np.ndarray) -> dict:
        """
        Get statistics about the classification results.

        Args:
            classification_result (np.ndarray): 2D array of classification labels

        Returns:
            dict: Statistics including class counts and percentages
        """
        unique_labels, counts = np.unique(classification_result, return_counts=True)
        total_pixels = classification_result.size

        stats = {}
        for label, count in zip(unique_labels, counts):
            class_name = self.unique_class_names[label]
            percentage = (count / total_pixels) * 100
            stats[class_name] = {"pixels": int(count), "percentage": float(percentage)}

        return stats
