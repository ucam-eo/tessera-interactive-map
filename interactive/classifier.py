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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class MLPClassifier(nn.Module):
    """
    A simple 2-layer MLP classifier with ReLU activation.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        """
        Initialize MLP classifier.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            hidden_dim (int, optional): Hidden layer dimension. Defaults to input_dim * 2.
        """
        super(MLPClassifier, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLPWrapper:
    """
    Wrapper class to make MLP compatible with sklearn-like interface.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: Optional[int] = None,
                 batch_size: int = 32, epochs: int = 50, learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize MLP wrapper.
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            hidden_dim (int, optional): Hidden layer dimension
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            device (str, optional): Device to use ('cpu' or 'cuda'). Auto-detected if None.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create MLP model - MLPClassifier is defined in the same module
        # Use direct reference to avoid autoreload issues
        self.model = MLPClassifier(input_dim, num_classes, hidden_dim).to(self.device)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the MLP model.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training labels of shape (n_samples,)
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy().astype(np.float32)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


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
        self.last_classification_result = None
        self.last_probabilities = None

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
        
        elif model_name == 'mlp':
            hidden_dim = model_params.get('hidden_dim', None)
            batch_size = model_params.get('batch_size', 32)
            epochs = model_params.get('epochs', 50)
            learning_rate = model_params.get('learning_rate', 0.001)
            print(f"Training MLP classifier (hidden_dim={hidden_dim}, epochs={epochs})...")
            self.model = MLPWrapper(
                input_dim=X_train.shape[1],
                num_classes=len(self.unique_class_names),
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.model.fit(X_train, y_train)

        print("Model training complete.")

    def classify_mosaic(
        self, batch_size: int = 15000, progress_callback: Optional[Callable] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Classify the entire mosaic using the trained model.

        Args:
            batch_size (int): Size of batches for processing
            progress_callback (Callable): Optional callback function for progress updates

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]]: 
                - classification_result of shape (height, width) with values 1-N
                - confidence_map of shape (height, width)
                - probabilities of shape (height, width, num_classes) or None if not available
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
        
        # Reshape probabilities to (H, W, N)
        probabilities = all_probabilities.reshape(self.mosaic_height, self.mosaic_width, len(self.unique_class_names))
        
        # Convert classification_result to 1-N (instead of 0-(N-1))
        classification_result = classification_result + 1
        
        # Store results for export
        self.last_classification_result = classification_result
        self.last_probabilities = probabilities

        # clean up variable to save memory
        del all_pixels, all_probabilities
        return classification_result, confidence_map, probabilities

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
            classification_result (np.ndarray): 2D array with values 1-N (not 0-(N-1))
            confidence_map (np.ndarray): 2D array of model confidence scores (0.0 to 1.0)
            mode (str): standard, confidence_opacity, or threshold
            threshold (float): Confidence threshold for the threshold mode
        Returns:
            str: Base64-encoded PNG image data URL
        """
        # Convert from 1-N to 0-(N-1) for visualization
        classification_result_viz = classification_result - 1
        
        # create colormap from the color mapping
        color_list = [
            color_map.get(name, "#888888") for name in self.unique_class_names
        ]
        cmap = mcolors.ListedColormap(color_list)
        norm = mcolors.Normalize(vmin=0, vmax=len(self.unique_class_names) - 1)
        colored_result_rgb = cmap(norm(classification_result_viz))[:, :, :3]

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
            classification_result (np.ndarray): 2D array of classification labels with values 1-N

        Returns:
            dict: Statistics including class counts and percentages
        """
        unique_labels, counts = np.unique(classification_result, return_counts=True)
        total_pixels = classification_result.size

        stats = {}
        for label, count in zip(unique_labels, counts):
            # Convert from 1-N to 0-(N-1) for indexing
            class_idx = int(label) - 1
            if 0 <= class_idx < len(self.unique_class_names):
                class_name = self.unique_class_names[class_idx]
                percentage = (count / total_pixels) * 100
                stats[class_name] = {"pixels": int(count), "percentage": float(percentage)}

        return stats
