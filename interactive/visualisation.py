import base64
import io
import json
from functools import partial

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from ipyleaflet import (
    CircleMarker,
    DrawControl,
    ImageOverlay,
    Map,
    Rectangle,
    TileLayer,
    LayerGroup,
)
from IPython.display import display
from ipywidgets import (
    HTML,
    Button,
    Checkbox,
    ColorPicker,
    Dropdown,
    FloatSlider,
    HBox,
    Layout,
    Output,
    Text,
    ToggleButton,
    VBox,
    IntSlider,
)
from rasterio import Affine
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio import transform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .classifier import EmbeddingClassifier
from .utils import check_bbox_valid


class InteractiveMappingTool:
    """Interactive mapping tool for labeling training points on satellite imagery."""

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        embedding_mosaic: np.ndarray,
        mosaic_transform,
    ):
        self.training_points = []
        self.markers = {}
        self.A_MARKER_WAS_JUST_REMOVED = False
        self.class_color_map = {}
        self.tab10_cmap = plt.colormaps.get_cmap("tab10")
        self.classification_layer = None
        self.sentinel_layer = None

        # arguments
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.embedding_mosaic = embedding_mosaic
        self.mosaic_transform = mosaic_transform

        # initialize embedding classifier
        self.embedding_classifier = EmbeddingClassifier(
            embedding_mosaic, mosaic_transform
        )

        # initialize tool
        self._setup_initial_classes()
        self._create_widgets()
        self.vis_bounds, self.vis_data_url = self.visualise_embedding(
            self.embedding_mosaic, self.mosaic_transform
        )
        self._create_map()
        self._setup_event_handlers()
        self._create_layout()

    def get_or_assign_color_for_class(self, class_name: str) -> str:
        """Assigns a consistent color if one doesn't exist, otherwise returns existing color.

        Args:
            class_name (str): Name of the class to get or assign a color for

        Returns:
            str: Hex color code
        """
        if class_name not in self.class_color_map:
            new_color_index = len(self.class_color_map) % 10
            self.class_color_map[class_name] = mcolors.to_hex(
                self.tab10_cmap(new_color_index)
            )
        return self.class_color_map[class_name]

    def _setup_initial_classes(self) -> None:
        """Initialize default class names and assign them colors."""
        initial_classes = ["Water", "Urban"]
        for c in initial_classes:
            self.get_or_assign_color_for_class(c)
        self.initial_classes = initial_classes

    def _create_widgets(self) -> None:
        """Create all UI widgets for the interactive mapping tool."""
        self.class_dropdown = Dropdown(
            options=self.initial_classes, value="Water", description="Class:"
        )
        self.new_class_text = Text(
            value="", placeholder="Type new class name", description="New Class:"
        )
        self.add_class_button = Button(description="Add")

        self.color_picker = ColorPicker(
            concise=False,
            description="Set Color:",
            value=self.class_color_map.get(self.class_dropdown.value, "#FFFFFF"),
            disabled=False,
        )
      
        self.vis_mode_selector = Dropdown(
            options=['Standard', 'Confidence (Opacity)', 'Uncertainty (Threshold)'],
            value='Standard', description='Vis Mode:', layout={'width': 'max-content'}
        )
        self.confidence_slider = FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.05, description='Threshold:',
            layout={'display': 'none'} # hidden until relevant
        )

        self.opacity_toggle = ToggleButton(
            value=True, description="Show Embedding", button_style="info"
        )
        self.opacity_slider = FloatSlider(
            value=0.7, min=0, max=1.0, step=0.05, description="Opacity:"
        )
        self.classify_button = Button(description="Classify")
        self.clear_pins_button = Button(description="Clear All Pins")
        self.clear_classification_button = Button(
            description="Clear Classification", disabled=True
        )
        self.model_selector = Dropdown(
            options=['kNN', 'Random Forest'],
            value='kNN',
            description='Model:',
            disabled=False,
        )
        self.filename_text = Text(
            value="labels.json", placeholder="Enter filename", description="Filename:"
        )
        self.save_button = Button(description="Save Labels", button_style="success")
        self.load_button = Button(description="Load Labels", button_style="primary")
        self.output_log = Output()

        self.legend_widget = HTML(
            value="<b>Legend:</b><br><i>Add training points to see legend.</i>"
        )

        self.basemap_layers = {
            "Esri Satellite": TileLayer(
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attribution="Esri",
                name="Esri Satellite",
            ),
            "Google Earth": TileLayer(
                url="http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}",
                attribution="Google Earth",
                name="Google",
            ),
            "Google Maps": TileLayer(
                url="http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}",
                attribution="Google Maps",
                name="Google",
            ),
        }

        self.current_basemap = self.basemap_layers["Esri Satellite"]

        basemap_options = list(self.basemap_layers.keys()) + ["Sentinel-2"]
        self.basemap_selector = Dropdown(
            options=basemap_options,
            value="Esri Satellite",
            description="Basemap:",
        )

        years = [str(y) for y in range(2018, 2025)]  # Years 2018-2024
        self.year_selector = Dropdown(
            options=years,
            value="2024",
            description="Year:",
            layout={"display": "none"},  # Initially hidden
        )

        self.kernel_size_slider = IntSlider(
            value=1,  # Default to 1x1
            min=1,
            max=9,  # up to 9x9
            step=2, # Ensures odd numbers: 1, 3, 5, 7, 9
            description='Label Kernel Size:',
            style={'description_width': 'initial'}
        )

    def _create_map(self) -> None:
        """Create the interactive map with basemap and overlay layers."""
        map_layout = Layout(height="600px", width="100%")
        self.m = Map(
            layers=(self.current_basemap,),
            center=(
                (self.min_lat + self.max_lat) / 2,
                (self.min_lon + self.max_lon) / 2,
            ),
            zoom=12,
            layout=map_layout,
        )
        self.image_overlay = ImageOverlay(
            url=self.vis_data_url,
            bounds=self.vis_bounds,
            opacity=self.opacity_slider.value if self.opacity_toggle.value else 0.7,
        )
        self.m.add(self.image_overlay)

        print("Image overlay added to map.")

    def update_legend(self) -> None:
        """Update the legend widget with current class colors."""
        if not self.class_color_map:
            self.legend_widget.value = "<b>Legend:</b> <i>No classes defined.</i>"
            return

        html = "<div style='display: flex; align-items: center; flex-wrap: wrap;'><b style='margin-right: 10px;'>Legend:</b>"

        # sort items for consistent order
        sorted_items = sorted(self.class_color_map.items(), key=lambda item: item[0])

        for class_name, color in sorted_items:
            html += f"""
                <div style='display: flex; align-items: center; margin-right: 15px;'>
                    <span style='height: 15px; width: 15px; background-color:{color}; 
                          border: 1px solid #555; display: inline-block; margin-right: 4px;'></span>
                    <span>{class_name}</span>
                </div>
            """
        html += "</div>"
        self.legend_widget.value = html

    def update_opacity(self, change: dict) -> None:
        """Update opacity of embedding and classification overlays.

        Args:
            change (dict): Widget change event containing the new value.
        """
        is_visible = self.opacity_toggle.value
        opacity_value = self.opacity_slider.value if is_visible else 0
        self.image_overlay.opacity = opacity_value

        # update classification layer opacity if it exists
        if self.classification_layer and self.classification_layer in self.m.layers:
            self.classification_layer.opacity = opacity_value

        self.opacity_slider.disabled = not is_visible

    def on_add_class_button_clicked(self, b: dict) -> None:
        """Handle click event for adding a new class.

        Args:
            b (dict): Button click event object.
        """
        new_class = self.new_class_text.value.strip()
        if new_class and new_class not in self.class_dropdown.options:
            self.class_dropdown.options += (new_class,)
            self.class_dropdown.value = new_class
            self.color_picker.value = self.get_or_assign_color_for_class(
                new_class
            )  # Assign a color and update picker
            self.new_class_text.value = ""
            self.update_legend()
            with self.output_log:
                self.output_log.clear_output()
                print(f"Added new class: '{new_class}'")

    def _update_sentinel_layer(self):
        """Creates or updates the Sentinel-2 layer with the selected year."""
        year = self.year_selector.value
        sentinel_url = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-{year}_3857/default/g/{{z}}/{{y}}/{{x}}.jpg"

        # If the sentinel layer doesn't exist yet, create it and add it to the map
        if self.sentinel_layer is None:
            self.sentinel_layer = TileLayer(
                url=sentinel_url,
                attribution="Sentinel-2 cloudless by EOx",
                name=f"Sentinel-2 ({year})",
            )
            self.m.add(self.sentinel_layer)
        # If it already exists, just update its URL (more efficient)
        else:
            self.sentinel_layer.url = sentinel_url
            self.sentinel_layer.name = f"Sentinel-2 ({year})"

    def on_year_change(self, change: dict):
        """Handler for when the year dropdown changes."""
        self._update_sentinel_layer()

    def on_basemap_change(self, change: dict) -> None:
        """Handle basemap selection change.

        Args:
            change (dict): Widget change event containing the selected basemap name.
        """
        new_basemap_name = change["new"]
        # If Sentinel-2 is selected
        if new_basemap_name == "Sentinel-2":
            # 1. Make the year selector visible
            self.year_selector.layout.display = "flex"

            # 2. Remove the old static basemap
            if self.current_basemap in self.m.layers:
                self.m.remove_layer(self.current_basemap)

            # 3. Create or update the Sentinel layer
            self._update_sentinel_layer()

        # If a static basemap is selected
        else:
            # 1. Hide the year selector
            self.year_selector.layout.display = "none"

            # 2. Remove the dynamic Sentinel layer if it exists
            if self.sentinel_layer and self.sentinel_layer in self.m.layers:
                self.m.remove_layer(self.sentinel_layer)
                self.sentinel_layer = None  # Reset it

            # 3. Add the selected static layer
            new_layer = self.basemap_layers[new_basemap_name]
            if self.current_basemap in self.m.layers:
                self.m.remove_layer(self.current_basemap)
            self.m.add_layer(new_layer)
            self.current_basemap = new_layer

    def on_class_selection_change(self, change) -> None:
        """Handle class dropdown selection change and update color picker.

        Args:
            change: Widget change event containing the selected class name.
        """
        selected_class = change.new
        color = self.get_or_assign_color_for_class(selected_class)
        self.color_picker.unobserve(self.on_color_change, names="value")
        self.color_picker.value = color
        self.color_picker.observe(self.on_color_change, names="value")

    def on_color_change(self, change) -> None:
        """Handle color picker change and update existing markers.

        Args:
            change: Widget change event containing the new color value.
        """
        new_color = change.new
        class_to_update = self.class_dropdown.value

        self.class_color_map[class_to_update] = new_color
        self.update_legend()
        for i, (point, class_name) in enumerate(self.training_points):
            if class_name == class_to_update:
                coords = point
                marker_key = tuple(coords)
                if marker_key in self.markers:
                    self.m.remove_layer(self.markers[marker_key])
                recolored_marker = CircleMarker(
                    location=coords,
                    radius=6,
                    color=new_color,
                    fill_color=new_color,
                    fill_opacity=0.8,
                    weight=1,
                )

                # attach the click-to-remove handler to the recolored marker
                recolored_marker.on_click(partial(self.remove_marker, marker_key))

                self.m.add(recolored_marker)
                self.markers[marker_key] = recolored_marker

    def remove_marker(self, marker_key: tuple, **kwargs: dict) -> None:
        """Remove a training point marker from the map and data.

        Args:
            marker_key: Tuple of coordinates for the marker to remove.
            kwargs: Additional keyword arguments.
        """
        # remove from map
        if marker_key in self.markers:
            self.m.remove_layer(self.markers[marker_key])
            del self.markers[marker_key]

        # remove from training data
        coords_to_remove = marker_key
        self.training_points = [
            p for p in self.training_points if tuple(p[0]) != coords_to_remove
        ]

        self.A_MARKER_WAS_JUST_REMOVED = True

        with self.output_log:
            self.output_log.clear_output(wait=True)
            print(
                f"Removed point at ({coords_to_remove[0]:.4f}, {coords_to_remove[1]:.4f}). Total points: {len(self.training_points)}"
            )

    def handle_map_click(self, **kwargs: dict) -> None:
        """Handle map click events to add new training points."""
        # if a marker was just deleted, this click was used
        # ignore it and reset the flag for the next click
        if self.A_MARKER_WAS_JUST_REMOVED:
            self.A_MARKER_WAS_JUST_REMOVED = False
            return

        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if coords is None:
                return

            lat, lon = coords
            selected_class = self.class_dropdown.value
            kernel_size = self.kernel_size_slider.value
            
            # Convert lat/lon to pixel row/col
            center_row, center_col = transform.rowcol(self.mosaic_transform, lon, lat)
            
            # Check if click is within bounds
            mosaic_height, mosaic_width, _ = self.embedding_mosaic.shape
            if not (0 <= center_row < mosaic_height and 0 <= center_col < mosaic_width):
                with self.output_log:
                    self.output_log.clear_output(wait=True)
                    print("Clicked outside the bounds of the embedding mosaic.")
                return

            # Calculate kernel bounds
            radius = (kernel_size - 1) // 2
            row_start = max(0, center_row - radius)
            row_end = min(mosaic_height, center_row + radius + 1)
            col_start = max(0, center_col - radius)
            col_end = min(mosaic_width, center_col + radius + 1)

            points_to_add = []
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    # Convert each pixel back to lat/lon
                    px_lon, px_lat = transform.xy(self.mosaic_transform, r, c)
                    points_to_add.append(([px_lat, px_lon], selected_class))
            
            # Add the points to the main list
            self.training_points.extend(points_to_add)            
            marker_key = tuple(coords)
            pin_color = self.get_or_assign_color_for_class(selected_class)
            marker = CircleMarker(
                location=coords,
                radius=6,
                color=pin_color,
                fill_color=pin_color,
                fill_opacity=0.8,
                weight=1,
            )

            marker.on_click(partial(self.remove_marker, marker_key))

            self.m.add(marker)
            self.markers[marker_key] = marker
            with self.output_log:
                self.output_log.clear_output(wait=True)
                print(
                    f"Added '{selected_class}' point at ({coords[0]:.4f}, {coords[1]:.4f}). Total points: {len(self.training_points)}"
                )

    def on_clear_pins_button_clicked(self, b=None) -> None:
        """Clear all training points and markers from the map.

        Args:
            b: Button click event object.
        """
        with self.output_log:
            for _, marker in self.markers.items():
                self.m.remove_layer(marker)
            self.training_points, self.markers, self.class_color_map = [], {}, {}
            self.output_log.clear_output()
            print("All pins cleared.")
            for c in self.initial_classes:
                self.get_or_assign_color_for_class(c)
            self.color_picker.value = self.get_or_assign_color_for_class(
                self.class_dropdown.value
            )
            self.update_legend()

    def on_clear_classification_clicked(self, b: dict) -> None:
        """Remove the classification overlay from the map.

        Args:
            b: Button click event object.
        """
        if self.classification_layer and self.classification_layer in self.m.layers:
            self.m.remove_layer(self.classification_layer)
            self.classification_layer = None
            self.clear_classification_button.disabled = True
            with self.output_log:
                self.output_log.clear_output()
                print("Classification layer removed.")

    def on_classify_button_clicked(self, b):
        """Perform tessera embedding-based classification and display results on map.

        Args:
            b: Button click event object.
        """
        with self.output_log:
            self.output_log.clear_output()

            # validate training points
            is_valid, error_msg = self.embedding_classifier.validate_training_points(
                self.training_points, min_points=2, min_classes=2
            )
            if not is_valid:
                print(f"ERROR: {error_msg}")
                return

            try:
                print("\nStarting classification...")

                # prepare training data from labeled points
                X_train, y_train, validation_info = (
                    self.embedding_classifier.prepare_training_data(
                        self.training_points
                    )
                )

                print(
                    f"Discovered classes for training: {validation_info['unique_classes']}"
                )
                print(
                    f"Mapping {validation_info['total_points']} training points to pixel coordinates..."
                )

                # report skipped points
                if validation_info["skipped_points"]:
                    for lat, lon, class_name in validation_info["skipped_points"]:
                        print(
                            f"\tWARNING: Skipping point for '{class_name}' at ({lat:.4f}, {lon:.4f}) as it falls outside the mosaic's bounds."
                        )

                if validation_info["valid_points"] == 0:
                    print(
                        "ERROR: None of the training points were inside the mosaic bounds."
                    )
                    return

                selected_model_display = self.model_selector.value
                model_name_map = {
                    'kNN': 'knn',
                    'Random Forest': 'rf'
                }
                model_key = model_name_map.get(selected_model_display)
                
                if not model_key:
                    print(f"Error: Invalid model selection '{selected_model_display}'")
                    return

                print(f"Training {selected_model_display} classifier on {validation_info['valid_points']} valid points...")
                
                self.embedding_classifier.train_classifier(X_train, y_train, model_name=model_key)

                # classify the entire mosaic
                print("\nClassifying pixels...")
                classification_result, confidence_map = self.embedding_classifier.classify_mosaic(
                    batch_size=15000
                )

                # Get visualization settings from the UI
                vis_mode = self.vis_mode_selector.value
                mode_map = {
                    'Standard': 'standard',
                    'Confidence (Opacity)': 'confidence_opacity',
                    'Uncertainty (Threshold)': 'threshold'
                }
                vis_mode_key = mode_map.get(vis_mode)
                confidence_threshold = self.confidence_slider.value

                # create visualization
                print("Creating visualization of the classification result...")
                classification_data_url = self.embedding_classifier.create_visualization(
                    classification_result,
                    self.class_color_map,
                    confidence_map=confidence_map,
                    mode=vis_mode_key,
                    threshold=confidence_threshold
                )

                # display results on map
                print("Displaying result on the map...")

                # remove existing classification layer if present
                if (
                    self.classification_layer
                    and self.classification_layer in self.m.layers
                ):
                    self.m.remove_layer(self.classification_layer)

                # create new ImageOverlay for the classification
                self.classification_layer = ImageOverlay(
                    url=classification_data_url,
                    bounds=self.vis_bounds,
                    opacity=0.7,
                    name="Classification",
                )
                self.m.add(self.classification_layer)

                # enable the clear button
                self.clear_classification_button.disabled = False

                # print completion message with statistics
                stats = self.embedding_classifier.get_classification_stats(
                    classification_result
                )
                print("Classification complete.")
                print(
                    f"Used {validation_info['valid_points']} training points from {len(validation_info['unique_classes'])} classes."
                )
                print("\nClassification Statistics:")
                for class_name, stat in stats.items():
                    print(
                        f"  - {class_name}: {stat['pixels']:,} pixels ({stat['percentage']:.1f}%)"
                    )

            except Exception as e:
                print(f"Error during classification: {e}")
                import traceback

                traceback.print_exc()

    def on_save_button_clicked(self, b: dict) -> None:
        """Save training points and class colors to a file.

        Args:
            b: Button click event object.
        """
        fname = self.filename_text.value
        if not fname:
            with self.output_log:
                self.output_log.clear_output()
                print("Error: Please provide a filename.")
            return

        # bundle both the points and the color map together for save state
        save_data = {
            "training_points": self.training_points,
            "class_color_map": self.class_color_map,
        }

        try:
            with open(fname, "w") as f:
                json.dump(save_data, f, indent=2)
            with self.output_log:
                self.output_log.clear_output()
                print(
                    f"Successfully saved {len(self.training_points)} points to {fname}"
                )
        except Exception as e:
            with self.output_log:
                self.output_log.clear_output()
                print(f"Error saving file: {e}")

    def on_load_button_clicked(self, b: dict) -> None:
        """Load training points and class colors from a file.

        Args:
            b: Button click event object.
        """
        fname = self.filename_text.value
        if not fname:
            with self.output_log:
                self.output_log.clear_output()
                print("Error: Please provide a filename.")
            return

        try:
            with open(fname, "r") as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            with self.output_log:
                self.output_log.clear_output()
                print(f"Error: File not found: {fname}")
            return
        except Exception as e:
            with self.output_log:
                self.output_log.clear_output()
                print(f"Error loading file: {e}")
            return

        self.on_clear_pins_button_clicked(None)

        loaded_points = loaded_data.get("training_points", [])
        loaded_colors = loaded_data.get("class_color_map", {})

        self.class_color_map.update(loaded_colors)

        # re-draw all markers on the map
        for point_data in loaded_points:
            coords, class_name = point_data

            # add class to dropdown
            if class_name not in self.class_dropdown.options:
                self.class_dropdown.options += (class_name,)
            self.training_points.append(point_data)
            pin_color = self.get_or_assign_color_for_class(class_name)
            marker = CircleMarker(
                location=coords,
                radius=6,
                color=pin_color,
                fill_color=pin_color,
                fill_opacity=0.8,
                weight=1,
            )

            marker_key = tuple(coords)
            marker.on_click(partial(self.remove_marker, marker_key))

            self.m.add(marker)
            self.markers[marker_key] = marker
        self.update_legend()
        with self.output_log:
            self.output_log.clear_output()
            print(
                f"Successfully loaded {len(self.training_points)} points from {fname}"
            )

    def _setup_event_handlers(self):
        """Setup event handlers for all UI widgets."""
        self.opacity_toggle.observe(self.update_opacity, names="value")
        self.opacity_slider.observe(self.update_opacity, names="value")
        self.add_class_button.on_click(self.on_add_class_button_clicked)
        self.class_dropdown.observe(self.on_class_selection_change, names="value")
        self.color_picker.observe(self.on_color_change, names="value")
        self.m.on_interaction(self.handle_map_click)
        self.clear_pins_button.on_click(self.on_clear_pins_button_clicked)
        self.clear_classification_button.on_click(self.on_clear_classification_clicked)
        self.basemap_selector.observe(self.on_basemap_change, names="value")
        self.year_selector.observe(self.on_year_change, names="value")
        self.save_button.on_click(self.on_save_button_clicked)
        self.load_button.on_click(self.on_load_button_clicked)
        self.classify_button.on_click(self.on_classify_button_clicked)
        def on_vis_mode_change(change):
            if change.new == 'Uncertainty (Threshold)':
                self.confidence_slider.layout.display = 'flex'
            else:
                self.confidence_slider.layout.display = 'none'
        self.vis_mode_selector.observe(on_vis_mode_change, names='value')

    def _create_layout(self):
        """Create the layout for the interactive mapping tool."""
        self.class_dropdown.layout = Layout(width='250px')
        self.kernel_size_slider.layout = Layout(width='300px')
        self.new_class_text.layout = Layout(width='250px')
        self.add_class_button.layout = Layout(width='auto')
        
        labeling_actions = HBox([self.class_dropdown, self.kernel_size_slider])
        new_class_controls = HBox([self.new_class_text, self.add_class_button])
        class_controls_left_col = VBox([labeling_actions, new_class_controls])
        class_and_color_box = HBox([class_controls_left_col, self.color_picker])
        opacity_controls = HBox([self.opacity_toggle, self.opacity_slider])
        basemap_controls = VBox([self.basemap_selector, self.year_selector])
        
        # Main controls VBox
        controls = VBox(
            [
                basemap_controls,
                class_and_color_box,
                opacity_controls,
            ],
            layout=Layout(width='auto')
        )

        top_bar = HBox(
            [controls, self.legend_widget],
            layout=Layout(
                width="100%",
                height="auto",
                justify_content="space-between",
                align_items="flex-start",
            ),
        )

        self.legend_widget.layout = Layout(
            flex="1", margin="0 0 0 20px", overflow="auto"
        )

        vis_controls = VBox(
            [self.vis_mode_selector, self.confidence_slider],
            layout=Layout(margin='0 10px 0 10px')
        )
        
        action_box = HBox(
            [self.model_selector, vis_controls, self.classify_button],
            layout=Layout(align_items='center') 
        )

        clear_buttons = HBox(
            [self.clear_pins_button, self.clear_classification_button]
        )
        
        bottom_controls_row = HBox(
            [action_box, clear_buttons],
            layout=Layout(justify_content='space-between', width='100%', margin='10px 0 0 0')
        )

        # File controls for saving/loading
        file_controls = HBox([self.filename_text, self.save_button, self.load_button])
        self.ui = VBox([top_bar, self.m, bottom_controls_row, file_controls, self.output_log])

    def display(self) -> None:
        """Display the interactive mapping tool."""
        display(self.ui)
        self.update_legend()

    def get_training_data(self) -> dict:
        """Return current training points and class colors.

        Returns:
            dict: Dictionary containing training points and class colors.
        """
        return {
            "training_points": self.training_points,
            "class_color_map": self.class_color_map,
        }

    def _create_pca_visualization(
        self,
        embedding_mosaic: np.ndarray,
        n_samples: int = 100000,
        percentiles: list[int] = [2, 98],
    ) -> np.ndarray:
        """Create PCA-based visualization of embedding mosaic.

        Args:
            embedding_mosaic: Embedding array of shape (H, W, C)
            n_samples: Number of samples to use for PCA fitting
            percentiles: Percentiles for normalization

        Returns:
            vis_mosaic: Normalized RGB visualization array
        """
        print("\nCreating PCA-based visualization...")
        mosaic_height, mosaic_width, num_channels = embedding_mosaic.shape

        # reshape to non-spatial pixels
        pixels = embedding_mosaic.reshape(-1, num_channels)
        n_sample = min(pixels.shape[0], n_samples)
        sample_indices = np.random.choice(pixels.shape[0], n_sample, replace=False)

        # fit PCA
        pca = PCA(n_components=3)
        pca.fit(pixels[sample_indices, :])
        transformed_pixels = pca.transform(pixels)
        pca_image = transformed_pixels.reshape(mosaic_height, mosaic_width, 3)

        # normalize for display
        print("Normalizing PCA components for display...")
        vis_mosaic = self._normalize_pca_channels(pca_image, percentiles)
        print("PCA visualization created.")

        return vis_mosaic

    def _normalize_pca_channels(
        self, pca_image: np.ndarray, percentiles: list[int] = [2, 98]
    ) -> np.ndarray:
        """Normalize PCA channels for display.

        Args:
            pca_image (np.ndarray): PCA-transformed image array
            percentiles (list[int]): Percentiles for clipping

        Returns:
            vis_mosaic (np.ndarray): Normalized visualization array
        """
        vis_mosaic = np.zeros_like(pca_image)
        for i in range(3):
            channel = pca_image[:, :, i]
            min_val, max_val = np.percentile(channel, percentiles)
            if max_val > min_val:
                vis_mosaic[:, :, i] = np.clip(
                    (channel - min_val) / (max_val - min_val), 0, 1
                )
        return vis_mosaic

    def _create_base64_image(self, vis_mosaic: np.ndarray) -> str:
        """Convert visualization array to base64 data URL.

        Args:
            vis_mosaic (np.ndarray): Normalized visualization array

        Returns:
            str: Base64 data URL for the image
        """
        buffer = io.BytesIO()
        plt.imsave(buffer, vis_mosaic, format="png")
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def visualise_embedding(
        self,
        embedding_mosaic: np.ndarray,
        mosaic_transform: Affine,
        n_samples: int = 100000,
        percentiles: list[int] = [2, 98],
    ) -> tuple[tuple[tuple[float, float], tuple[float, float]], str]:
        """
        Visualise an embedding mosaic using PCA.

        Args:
            embedding_mosaic: Embedding array of shape (H, W, C)
            mosaic_transform: Rasterio transform for the mosaic
            n_samples: Number of samples to use for PCA fitting
            percentiles: Percentiles for normalization

        Returns:
            tuple[tuple[tuple[float, float], tuple[float, float]], str]: (vis_bounds, vis_data_url) for map overlay
        """
        mosaic_height, mosaic_width, _ = embedding_mosaic.shape

        # calculate bounds - mosaic is in EPSG:4326, so bounds are already in lat/lon
        west, south, east, north = array_bounds(
            mosaic_height, mosaic_width, mosaic_transform
        )
        vis_bounds = ((south, west), (north, east))
        print(
            f"Bounds of displayed embedding mosaic: ┗ ({south:.2f}, {west:.2f}) | ┓ ({north:.2f}, {east:.2f})"
        )

        # create PCA visualization
        vis_mosaic = self._create_pca_visualization(
            embedding_mosaic, n_samples, percentiles
        )

        # convert to base64 data URL for map overlay
        vis_data_url = self._create_base64_image(vis_mosaic)

        return vis_bounds, vis_data_url

    def update_embedding_overlay(
        self,
        embedding_mosaic: np.ndarray,
        mosaic_transform: Affine,
        n_samples: int = 100000,
        percentiles: list[int] = [2, 98],
    ) -> None:
        """Update the map with a new embedding visualization overlay.

        Args:
            embedding_mosaic: Embedding array of shape (H, W, C)
            mosaic_transform: Rasterio transform for the mosaic
            n_samples: Number of samples to use for PCA fitting
            percentiles: Percentiles for normalization
        """
        # create visualization
        vis_bounds, vis_data_url = self.visualise_embedding(
            embedding_mosaic, mosaic_transform, n_samples, percentiles
        )

        # update the image overlay
        self.image_overlay.url = vis_data_url
        self.image_overlay.bounds = vis_bounds

        # update bounds for classification if needed
        self.vis_bounds = vis_bounds
        self.vis_data_url = vis_data_url

        # update bounding box for classification grid
        south, west = vis_bounds[0]
        north, east = vis_bounds[1]
        self.min_lat, self.max_lat = south, north
        self.min_lon, self.max_lon = west, east

        with self.output_log:
            print("Updated embedding visualization overlay")
            print(f"Bounds: ({south:.4f}, {west:.4f}) to ({north:.4f}, {east:.4f})")


class BoundingBoxSelector:
    """Interactive bounding box selector using ipyleaflet map."""

    def __init__(self):
        """Initialize the bounding box selector with a world map."""
        self.bbox_coords = None
        self.selected_rectangle = None
        self.status = None
        self.visual_rectangle = None  # track the visual rectangle layer manually (separate from the draw control)
        self.bbox_valid = False
        self.bbox_too_small = False
        self.bbox_too_large = False

        # create world map
        self.map = Map(
            center=(20, 0),  # center on world view
            zoom=2,
            layout={"width": "100%", "height": "500px"},
        )

        # create draw control for rectangles alone with improved settings
        self.draw_control = DrawControl(
            rectangle={
                "shapeOptions": {
                    "color": "#ff0000",
                    "weight": 2,
                    "fillOpacity": 0.2,
                    "fillColor": "#ff0000",
                }
            },
            polygon={},  # disable polygon
            polyline={},  # disable polyline
            circle={},  # disable circle
            marker={},  # disable marker
            circlemarker={},  # disable circle marker
            edit=False,  # disable editing to avoid confusion
            remove=False,  # disable manual removal since we handle it automatically
        )

        # add draw control to map
        self.map.add_control(self.draw_control)

        # set up event handlers
        self.draw_control.on_draw(self._on_draw)

        # create output widgets
        self.info_widget = HTML(
            value="<b>Instructions:</b> Draw a rectangle on the map to select your bounding box."
        )
        self.coords_output = Output()

        # create layout
        self.widget = VBox(
            [
                self.info_widget,
                self.map,
                self.coords_output,
            ]
        )

    def _on_draw(
        self, target: DrawControl, action: str, geo_json: dict, **kwargs
    ) -> None:
        """Handle draw events on the map.

        Args:
            target (DrawControl): the DrawControl object
            action (str): the action type (e.g., 'created', 'edited', 'deleted')
            geo_json (dict): the GeoJSON object of the drawn feature
            **kwargs: additional keyword arguments
        """
        # if a rectangle has been created, update the bounding box coordinates
        if (
            action == "created"
            and geo_json
            and geo_json.get("geometry", {}).get("type") == "Polygon"
        ):
            # use the output widget to display debug information in the notebook
            with self.coords_output:
                # remove the rectangle from DrawControl immediately to avoid visual issues
                if geo_json in target.data:
                    target.data.remove(geo_json)

                # remove previous visual rectangle if it exists
                if self.visual_rectangle is not None:
                    self.map.remove_layer(self.visual_rectangle)

                # extract coordinates from the drawn rectangle
                coords = geo_json["geometry"]["coordinates"][0]
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]

                # calculate bounds for Rectangle layer
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)

                # create a new Rectangle layer for visual display
                self.visual_rectangle = Rectangle(
                    bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                    color="#2f7d31",  # green color for selected rectangle
                    weight=3,
                    fill_opacity=0.3,
                    fill_color="#2f7d31",
                )

                # add new rectangle to map
                self.map.add_layer(self.visual_rectangle)

                # store rectangle data and coordinates
                self.selected_rectangle = geo_json
                self.bbox_coords = {
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                }

                # check if bbox is valid (using utils.check_bbox)
                if check_bbox_valid(
                    (self.bbox_coords["min_lat"], self.bbox_coords["max_lat"]),
                    (self.bbox_coords["min_lon"], self.bbox_coords["max_lon"]),
                    verbose=False,
                ):
                    self.bbox_valid = True
                    # mark presence of rectangle in status
                    self.status = "drawn"

                    # update info widget with enhanced styling
                    self.info_widget.value = f"""
                    <div style="padding: 10px; background-color: #e8f5e8; border: 1px solid #4CAF50; border-radius: 5px;">
                        <b style="color: #2E7D32;">✓ Bounding Box Selected</b><br>
                        <div style="margin-top: 8px; font-family: monospace; font-size: 0.9em;">
                            <b>Longitude:</b> {self.bbox_coords["min_lon"]:.4f} to {self.bbox_coords["max_lon"]:.4f}<br>
                            <b>Latitude:</b> {self.bbox_coords["min_lat"]:.4f} to {self.bbox_coords["max_lat"]:.4f}
                        </div>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #666;">
                            <i>Draw a new rectangle to replace this selection</i>
                        </div>
                    </div>
                    """
                else:
                    self.bbox_valid = False
                    self.visual_rectangle = (
                        None  # Reset to avoid "layer not on map" error
                    )
                    # check if bbox too large or too small
                    if self.bbox_coords["max_lat"] - self.bbox_coords["min_lat"] < 0.1:
                        self.bbox_too_small = True
                    if self.bbox_coords["max_lon"] - self.bbox_coords["min_lon"] < 0.1:
                        self.bbox_too_small = True
                    if self.bbox_coords["max_lat"] - self.bbox_coords["min_lat"] > 10:
                        self.bbox_too_large = True
                    if self.bbox_coords["max_lon"] - self.bbox_coords["min_lon"] > 10:
                        self.bbox_too_large = True

                    if self.bbox_too_small:
                        error_message = "Bounding Box Too Small"
                    elif self.bbox_too_large:
                        error_message = "Bounding Box Too Large"
                    else:
                        error_message = "Bounding Box Invalid"

                    self.info_widget.value = f"""
                    <div style="padding: 10px; background-color: #ffaca6; border: 1px solid #D30000; border-radius: 5px;">
                        <b style="color: #D30000;">x {error_message}</b><br>
                        <div style="margin-top: 8px; font-family: monospace; font-size: 0.9em;">
                            <b>Longitude:</b> {self.bbox_coords["min_lon"]:.4f} to {self.bbox_coords["max_lon"]:.4f}<br>
                            <b>Latitude:</b> {self.bbox_coords["min_lat"]:.4f} to {self.bbox_coords["max_lat"]:.4f}
                        </div>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #666;">
                            <i>Draw a new rectangle to replace this selection</i>
                        </div>
                    </div>
                    """
                    if self.visual_rectangle is not None:
                        self.map.remove_layer(self.visual_rectangle)

    def display(self):
        """Display the bounding box selector widget."""
        display(self.widget)

    def get_bbox(self):
        """Get the current bounding box coordinates.

        Returns:
            tuple: ((min_lat, max_lat), (min_lon, max_lon)) or None if no selection
        """
        if self.bbox_coords:
            return (
                (self.bbox_coords["min_lat"], self.bbox_coords["max_lat"]),
                (self.bbox_coords["min_lon"], self.bbox_coords["max_lon"]),
            )
        return None

    def get_bbox_dict(self) -> dict | None:
        """Get the current bounding box coordinates as a dictionary.

        Returns:
            dict (dict | None): Dictionary with min_lat, max_lat, min_lon, max_lon keys or None
        """
        return self.bbox_coords


class InteractiveHeightMappingTool:
    """
    UNet patch-based height mapping (regression) tool for Jupyter.

    Features (current focus):
    - Default Train/Test ROIs are created at startup (as requested).
    - ROI -> patches with configurable patch_size (default 64) and overlap% (default 20%).
    - Patch bboxes visualization on the map (capped to avoid UI freeze).
    - Train a small UNet: input = embedding patch (C=128), output = height patch (1).
    - Predict by sliding-window patches over a chosen ROI and stitching back (overlap-avg).
    - Show GT / Pred / Err overlays + export pred/err GeoTIFF.
    """

    _MAX_PATCH_RECTS_TO_DRAW = 600

    class _PatchDataset(Dataset):
        def __init__(
            self,
            embedding: np.ndarray,  # (H,W,C)
            height: np.ndarray,  # (H,W)
            coords: list[tuple[int, int]],
            patch_size: int,
            nodata: float,
            mean: np.ndarray | None,
            std: np.ndarray | None,
        ):
            self.embedding = embedding
            self.height = height
            self.coords = coords
            self.patch_size = patch_size
            self.nodata = nodata
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.coords)

        def __getitem__(self, idx):
            r0, c0 = self.coords[idx]
            ps = self.patch_size
            x = self.embedding[r0 : r0 + ps, c0 : c0 + ps, :]  # (ps,ps,C)
            y = self.height[r0 : r0 + ps, c0 : c0 + ps]  # (ps,ps)
            m = np.isfinite(y) & (y != self.nodata)

            x = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32, copy=False))
            y = torch.from_numpy(y[None, ...].astype(np.float32, copy=False))
            m = torch.from_numpy(m[None, ...].astype(np.float32, copy=False))

            if self.mean is not None and self.std is not None:
                mean = torch.from_numpy(self.mean.astype(np.float32, copy=False))[:, None, None]
                std = torch.from_numpy(self.std.astype(np.float32, copy=False))[:, None, None]
                x = (x - mean) / (std + 1e-6)

            return x, y, m

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _DoubleConv(nn.Module):
        """(conv => BN => ReLU => Dropout) * 2 (following reference script)"""

        def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class _UNetRegression(nn.Module):
        """
        UNet for dense regression (patch -> patch), following:
        /maps/zf281/btfm4rs/src/train_downstream_via_representation_borneo_patch.py
        """

        def __init__(self, in_channels: int = 128, features: list[int] | None = None, dropout: float = 0.1):
            super().__init__()
            if features is None:
                features = [128, 256, 512]

            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            ch = in_channels
            for feat in features:
                self.downs.append(InteractiveHeightMappingTool._DoubleConv(ch, feat, dropout))
                ch = feat

            self.bottleneck = InteractiveHeightMappingTool._DoubleConv(features[-1], features[-1] * 2, dropout)

            for feat in reversed(features):
                self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
                self.ups.append(InteractiveHeightMappingTool._DoubleConv(feat * 2, feat, dropout))

            self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

        def forward(self, x):
            skip_connections = []
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip = skip_connections[idx // 2]
                if x.shape != skip.shape:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
                x = torch.cat((skip, x), dim=1)
                x = self.ups[idx + 1](x)

            return self.final_conv(x)  # (B,1,H,W)

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        embedding_mosaic: np.ndarray,
        mosaic_transform: Affine,
        height_gt: np.ndarray,
        height_gt_transform: Affine,
        height_gt_crs: str | None = None,
        mosaic_crs: str = "EPSG:4326",
        height_gt_nodata: float = -9999.0,
    ):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        self.embedding_mosaic = embedding_mosaic
        self.mosaic_transform = mosaic_transform

        self.height_gt = height_gt
        self.height_gt_transform = height_gt_transform
        self.height_gt_crs = height_gt_crs
        self.mosaic_crs = mosaic_crs
        self.height_gt_nodata = height_gt_nodata

        # If GT is not already on the embedding grid, warp it onto the embedding grid.
        # This allows GT and embedding to differ in shape/CRS, as long as georeferencing is provided.
        if (
            self.height_gt.shape[:2] != self.embedding_mosaic.shape[:2]
            or self.height_gt_transform != self.mosaic_transform
        ):
            src_crs = self.height_gt_crs or self.mosaic_crs
            if src_crs is None:
                raise ValueError(
                    "GT grid differs from embedding grid but `height_gt_crs` was not provided. "
                    "Please pass `height_gt_crs=ds.crs.to_string()` when reading the GT GeoTIFF."
                )
            self.height_gt = self._warp_gt_to_embedding_grid(
                self.height_gt,
                src_transform=self.height_gt_transform,
                src_crs=src_crs,
                src_nodata=self.height_gt_nodata,
                dst_transform=self.mosaic_transform,
                dst_crs=self.mosaic_crs,
                dst_shape=self.embedding_mosaic.shape[:2],
                dst_nodata=self.height_gt_nodata,
            )
            self.height_gt_transform = self.mosaic_transform

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: nn.Module | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        # ROI state
        self.active_roi_target = "train"
        # Support multiple disjoint ROIs (append-on-draw)
        self.train_rois_geojson: list[dict] = []
        self.test_rois_geojson: list[dict] = []
        self.train_roi_layers: list[Rectangle] = []
        self.test_roi_layers: list[Rectangle] = []

        # patch state
        self.train_patch_coords: list[tuple[int, int]] = []
        self.test_patch_coords: list[tuple[int, int]] = []
        self.infer_patch_coords: list[tuple[int, int]] = []
        self.train_patch_group = LayerGroup(layers=())
        self.test_patch_group = LayerGroup(layers=())

        # overlays
        self.gt_layer = None
        self.pred_layer = None
        self.err_layer = None
        self.last_prediction = None
        self.last_err = None

        # PCA embedding overlay (reuse existing logic from InteractiveMappingTool)
        self.vis_bounds, self.vis_data_url = self._visualise_embedding_pca(
            self.embedding_mosaic, self.mosaic_transform
        )

        self._create_widgets()
        self._create_map()
        self._setup_event_handlers()
        self._create_layout()
        self._init_default_rois()

    # -----------------
    # UI
    # -----------------
    def _create_widgets(self) -> None:
        self.output_log = Output()

        self.basemap_layers = {
            "Esri Satellite": TileLayer(
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attribution="Esri",
                name="Esri Satellite",
            ),
            "Google Earth": TileLayer(
                url="http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}",
                attribution="Google Earth",
                name="Google",
            ),
            "Google Maps": TileLayer(
                url="http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}",
                attribution="Google Maps",
                name="Google",
            ),
        }
        self.current_basemap = self.basemap_layers["Esri Satellite"]
        self.basemap_selector = Dropdown(
            options=list(self.basemap_layers.keys()),
            value="Esri Satellite",
            description="Basemap:",
        )

        # ROI controls
        self.draw_train_button = ToggleButton(value=True, description="Draw Train ROI", button_style="success")
        self.draw_test_button = ToggleButton(value=False, description="Draw Test ROI", button_style="warning")
        self.undo_train_roi_button = Button(description="Undo Train ROI")
        self.undo_test_roi_button = Button(description="Undo Test ROI")
        self.clear_rois_button = Button(description="Clear ROIs")

        # Put embedding toggle+opacity on the same row as Clear ROIs (per request)
        self.show_embedding_toggle = ToggleButton(value=True, description="Show Embedding", button_style="info")
        self.embedding_opacity = FloatSlider(value=0.7, min=0.0, max=1.0, step=0.05, description="Emb Opacity:")

        # Patch settings
        self.patch_size = IntSlider(value=16, min=16, max=256, step=16, description="Patch size (px):")
        self.overlap_pct = IntSlider(value=20, min=0, max=90, step=5, description="Overlap (%):")
        self.show_patch_grid = Checkbox(value=True, description="Show patch bboxes")
        self.inference_region = Dropdown(options=["Train ROI", "Test ROI"], value="Train ROI", description="Infer on:")

        # Training
        self.epochs = IntSlider(value=5, min=1, max=200, step=1, description="Epochs:")
        self.batch_size = IntSlider(value=8, min=1, max=64, step=1, description="Batch size:")
        self.lr = FloatSlider(value=1e-3, min=1e-5, max=1e-2, step=1e-5, description="LR:", readout_format=".0e")
        self.num_workers = IntSlider(value=0, min=0, max=8, step=1, description="Workers:")
        # Reference script does not do embedding standardization; keep option but default off.
        self.normalize_embeddings = Checkbox(value=False, description="Normalize emb")
        self.train_button = Button(description="Train UNet", button_style="primary")

        self.predict_button = Button(description="Predict & Stitch", button_style="success")
        self.clear_pred_button = Button(description="Clear Pred/Err")

        # Overlays
        self.show_gt_checkbox = Checkbox(value=False, description="Show GT")
        self.show_pred_checkbox = Checkbox(value=True, description="Show Pred")
        self.show_err_checkbox = Checkbox(value=False, description="Show Err")
        self.overlay_opacity = FloatSlider(value=0.7, min=0.0, max=1.0, step=0.05, description="Overlay Opacity:")

        self.value_clip_pct = IntSlider(value=98, min=80, max=100, step=1, description="Clip pct:")
        self.err_clip_pct = IntSlider(value=98, min=80, max=100, step=1, description="Err clip pct:")

        # Export + patch count display (to the right of Export err.tif)
        self.export_pred_button = Button(description="Export pred.tif")
        self.export_err_button = Button(description="Export err.tif")
        self.export_gt_button = Button(description="Export gt_aligned.tif")
        self.patch_count_label = HTML(value="<b>Patches:</b> -")

    def _create_map(self) -> None:
        map_layout = Layout(height="600px", width="100%")
        self.m = Map(
            layers=(self.current_basemap,),
            center=((self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2),
            zoom=12,
            layout=map_layout,
        )
        self.embedding_overlay = ImageOverlay(
            url=self.vis_data_url,
            bounds=self.vis_bounds,
            opacity=float(self.embedding_opacity.value) if self.show_embedding_toggle.value else 0.0,
            name="Embedding (PCA)",
        )
        self.m.add(self.embedding_overlay)

        self.m.add(self.train_patch_group)
        self.m.add(self.test_patch_group)

        self.draw_control = DrawControl(
            rectangle={"shapeOptions": {"color": "#00aa00", "weight": 2, "fillOpacity": 0.15}},
            polygon={"shapeOptions": {"color": "#00aa00", "weight": 2, "fillOpacity": 0.15}},
            polyline={},
            circle={},
            marker={},
            circlemarker={},
            edit=False,
            remove=False,
        )
        self.m.add_control(self.draw_control)

    def _create_layout(self) -> None:
        roi_row = HBox(
            [
                self.basemap_selector,
                self.draw_train_button,
                self.draw_test_button,
                self.undo_train_roi_button,
                self.undo_test_roi_button,
                self.clear_rois_button,
                self.show_embedding_toggle,
                self.embedding_opacity,
            ],
            layout=Layout(width="100%", flex_flow="row wrap"),
        )

        patch_row = HBox(
            [self.patch_size, self.overlap_pct, self.show_patch_grid, self.inference_region],
            layout=Layout(width="100%", flex_flow="row wrap"),
        )

        train_row = HBox(
            [self.epochs, self.batch_size, self.lr, self.num_workers, self.normalize_embeddings, self.train_button],
            layout=Layout(width="100%", flex_flow="row wrap"),
        )

        infer_row = HBox(
            [self.predict_button, self.clear_pred_button, self.export_pred_button, self.export_err_button, self.export_gt_button, self.patch_count_label],
            layout=Layout(width="100%", flex_flow="row wrap"),
        )

        overlay_row = HBox(
            [
                self.show_gt_checkbox,
                self.show_pred_checkbox,
                self.show_err_checkbox,
                self.overlay_opacity,
                VBox([self.value_clip_pct, self.err_clip_pct]),
            ],
            layout=Layout(width="100%", flex_flow="row wrap"),
        )

        controls = VBox([roi_row, patch_row, train_row, infer_row, overlay_row])
        self.ui = VBox([controls, self.m, self.output_log])

    def display(self) -> None:
        display(self.ui)
        with self.output_log:
            self.output_log.clear_output()
            print("UNet height mapping tool ready.")
            print("Default Train/Test ROIs are pre-filled.")
            print("1) Adjust patch size / overlap (optional)")
            print("2) Click Train UNet")
            print("3) Click Predict & Stitch")

    # -----------------
    # Embedding PCA overlay helpers
    # -----------------
    def _visualise_embedding_pca(self, embedding_mosaic: np.ndarray, mosaic_transform: Affine):
        h, w, _ = embedding_mosaic.shape
        west, south, east, north = array_bounds(h, w, mosaic_transform)
        vis_bounds = ((south, west), (north, east))

        pixels = embedding_mosaic.reshape(-1, embedding_mosaic.shape[2])
        n_sample = min(pixels.shape[0], 100_000)
        sample_indices = np.random.choice(pixels.shape[0], n_sample, replace=False)
        pca = PCA(n_components=3)
        pca.fit(pixels[sample_indices, :])
        pca_img = pca.transform(pixels).reshape(h, w, 3)

        vis = np.zeros_like(pca_img)
        for i in range(3):
            ch = pca_img[:, :, i]
            lo, hi = np.percentile(ch, [2, 98])
            if hi > lo:
                vis[:, :, i] = np.clip((ch - lo) / (hi - lo), 0, 1)
        rgba = np.concatenate([vis, np.ones((*vis.shape[:2], 1), dtype=vis.dtype)], axis=2)
        buffer = io.BytesIO()
        plt.imsave(buffer, rgba, format="png")
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return vis_bounds, f"data:image/png;base64,{b64_data}"

    def _warp_gt_to_embedding_grid(
        self,
        src: np.ndarray,
        *,
        src_transform: Affine,
        src_crs: str,
        src_nodata: float,
        dst_transform: Affine,
        dst_crs: str,
        dst_shape: tuple[int, int],
        dst_nodata: float,
    ) -> np.ndarray:
        """Warp a single-band GT raster to the embedding grid."""
        dst = np.full(dst_shape, dst_nodata, dtype=np.float32)
        reproject(
            source=src.astype(np.float32, copy=False),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=Resampling.bilinear,
        )
        return dst

    # -----------------
    # ROI / patches
    # -----------------
    def _set_active_roi_target(self, target: str) -> None:
        self.active_roi_target = target
        if target == "train":
            self.draw_train_button.value = True
            self.draw_test_button.value = False
            color = "#00aa00"
        else:
            self.draw_train_button.value = False
            self.draw_test_button.value = True
            color = "#ff8800"
        self.draw_control.rectangle = {"shapeOptions": {"color": color, "weight": 2, "fillOpacity": 0.15}}
        self.draw_control.polygon = {"shapeOptions": {"color": color, "weight": 2, "fillOpacity": 0.15}}

    def _stride_px(self) -> int:
        ps = int(self.patch_size.value)
        ov = int(self.overlap_pct.value) / 100.0
        stride = int(round(ps * (1.0 - ov)))
        return max(1, min(ps, stride))

    def _roi_mask(self, roi_geojson: dict) -> np.ndarray:
        h, w, _ = self.embedding_mosaic.shape
        geom = roi_geojson["geometry"]
        return geometry_mask([geom], out_shape=(h, w), transform=self.mosaic_transform, invert=True, all_touched=False)

    def _rois_mask(self, rois_geojson: list[dict]) -> np.ndarray:
        """Union mask for multiple ROIs."""
        h, w, _ = self.embedding_mosaic.shape
        if not rois_geojson:
            return np.zeros((h, w), dtype=bool)
        geoms = [r["geometry"] for r in rois_geojson]
        return geometry_mask(geoms, out_shape=(h, w), transform=self.mosaic_transform, invert=True, all_touched=False)

    def _valid_gt_mask(self) -> np.ndarray:
        y = self.height_gt
        return np.isfinite(y) & (y != self.height_gt_nodata)

    def _roi_bounds_latlon(self, roi_geojson: dict) -> tuple[float, float, float, float]:
        coords = roi_geojson["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return min(lats), max(lats), min(lons), max(lons)

    def _latlon_bbox_to_pixel_bounds(self, min_lat, max_lat, min_lon, max_lon) -> tuple[int, int, int, int]:
        r0, c0 = transform.rowcol(self.mosaic_transform, min_lon, max_lat)
        r1, c1 = transform.rowcol(self.mosaic_transform, max_lon, min_lat)
        row_min, row_max = min(r0, r1), max(r0, r1)
        col_min, col_max = min(c0, c1), max(c0, c1)
        return row_min, row_max + 1, col_min, col_max + 1

    def _patch_rect_for_coord(self, r0: int, c0: int, ps: int, color: str) -> Rectangle:
        west, north = (self.mosaic_transform * (c0, r0))
        east, south = (self.mosaic_transform * (c0 + ps, r0 + ps))
        min_lon, max_lon = min(west, east), max(west, east)
        min_lat, max_lat = min(south, north), max(south, north)
        return Rectangle(
            bounds=[(min_lat, min_lon), (max_lat, max_lon)],
            color=color,
            weight=1,
            fill_opacity=0.0,
            fill_color=color,
        )

    def _set_patch_group(self, group: LayerGroup, rects: list[Rectangle]) -> None:
        group.layers = tuple(rects)

    def _refresh_patch_overlays(self) -> None:
        if not self.show_patch_grid.value:
            self._set_patch_group(self.train_patch_group, [])
            self._set_patch_group(self.test_patch_group, [])
            return

        ps = int(self.patch_size.value)
        train_rects = [self._patch_rect_for_coord(r, c, ps, "#00aa00") for (r, c) in self.train_patch_coords[: self._MAX_PATCH_RECTS_TO_DRAW]]
        test_rects = [self._patch_rect_for_coord(r, c, ps, "#ff8800") for (r, c) in self.test_patch_coords[: self._MAX_PATCH_RECTS_TO_DRAW]]
        self._set_patch_group(self.train_patch_group, train_rects)
        self._set_patch_group(self.test_patch_group, test_rects)

    def _update_patch_count_label(self) -> None:
        infer = self.inference_region.value
        infer_n = len(self.train_patch_coords) if infer == "Train ROI" else len(self.test_patch_coords)
        self.patch_count_label.value = (
            f"<b>Patches:</b> train={len(self.train_patch_coords):,} | test={len(self.test_patch_coords):,} | infer={infer_n:,}"
        )

    def _gen_patch_coords_for_roi(self, roi_geojson: dict, *, require_valid_gt: bool = False) -> list[tuple[int, int]]:
        ps = int(self.patch_size.value)
        stride = self._stride_px()
        h, w, _ = self.embedding_mosaic.shape
        roi_mask = self._roi_mask(roi_geojson)
        valid_gt = self._valid_gt_mask() if require_valid_gt else None

        min_lat, max_lat, min_lon, max_lon = self._roi_bounds_latlon(roi_geojson)
        r0, r1, c0, c1 = self._latlon_bbox_to_pixel_bounds(min_lat, max_lat, min_lon, max_lon)
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(h, r1), min(w, c1)

        # Generate start positions that COVER the ROI bbox even if bbox < patch_size.
        def starts(start: int, end: int, max_len: int) -> list[int]:
            span = end - start
            if span <= 0:
                return []
            if span <= ps:
                center = (start + end) // 2
                s0 = max(0, min(max_len - ps, center - ps // 2))
                return [s0]
            s_list = list(range(start, end - ps + 1, stride))
            last = end - ps
            if not s_list or s_list[-1] != last:
                s_list.append(last)
            # clamp
            s_list = [max(0, min(max_len - ps, s)) for s in s_list]
            # unique while preserving order
            out = []
            seen = set()
            for s in s_list:
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            return out

        row_starts = starts(r0, r1, h)
        col_starts = starts(c0, c1, w)

        coords: list[tuple[int, int]] = []
        for rr in row_starts:
            for cc in col_starts:
                cr, cc2 = rr + ps // 2, cc + ps // 2
                if cr < 0 or cr >= h or cc2 < 0 or cc2 >= w:
                    continue
                if not roi_mask[cr, cc2]:
                    continue
                if require_valid_gt:
                    # For training/metrics only: require at least some valid GT pixels
                    if not np.any(valid_gt[rr : rr + ps, cc : cc + ps]):
                        continue
                coords.append((rr, cc))
        return coords

    def _gen_patch_coords_for_rois(self, rois_geojson: list[dict], *, require_valid_gt: bool = False) -> list[tuple[int, int]]:
        """Union patch coords across multiple ROIs (de-duplicated)."""
        if not rois_geojson:
            return []
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for roi in rois_geojson:
            for rc in self._gen_patch_coords_for_roi(roi, require_valid_gt=require_valid_gt):
                if rc in seen:
                    continue
                seen.add(rc)
                out.append(rc)
        return out

    def _recompute_patches(self) -> None:
        # For UI + inference: generate patch grid regardless of GT validity (so Test ROI always shows patches).
        self.train_patch_coords = self._gen_patch_coords_for_rois(self.train_rois_geojson, require_valid_gt=False)
        self.test_patch_coords = self._gen_patch_coords_for_rois(self.test_rois_geojson, require_valid_gt=False)
        self._refresh_patch_overlays()
        self._update_patch_count_label()

        with self.output_log:
            self.output_log.clear_output(wait=True)
            stride = self._stride_px()
            print(f"Patch size: {int(self.patch_size.value)} px, overlap: {int(self.overlap_pct.value)}% (stride={stride} px)")
            if len(self.train_patch_coords) > self._MAX_PATCH_RECTS_TO_DRAW or len(self.test_patch_coords) > self._MAX_PATCH_RECTS_TO_DRAW:
                print(f"NOTE: patch bbox 可视化最多绘制 {self._MAX_PATCH_RECTS_TO_DRAW} 个，避免卡顿。")
            print(f"Train patches: {len(self.train_patch_coords):,}")
            print(f"Test patches : {len(self.test_patch_coords):,}")

    def _filter_coords_with_valid_gt(self, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Keep only patches that contain at least one valid GT pixel."""
        ps = int(self.patch_size.value)
        valid_gt = self._valid_gt_mask()
        out: list[tuple[int, int]] = []
        for (r0, c0) in coords:
            if np.any(valid_gt[r0 : r0 + ps, c0 : c0 + ps]):
                out.append((r0, c0))
        return out

    def _set_roi_from_bounds(self, target: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> None:
        geo = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_lon, max_lat],
                        [max_lon, max_lat],
                        [max_lon, min_lat],
                        [min_lon, min_lat],
                        [min_lon, max_lat],
                    ]
                ],
            },
            "properties": {},
        }

        if target == "train":
            self.train_rois_geojson.append(geo)
            layer = Rectangle(
                bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                color="#00aa00",
                weight=3,
                fill_opacity=0.1,
                fill_color="#00aa00",
            )
            self.m.add_layer(layer)
            self.train_roi_layers.append(layer)
        else:
            self.test_rois_geojson.append(geo)
            layer = Rectangle(
                bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                color="#ff8800",
                weight=3,
                fill_opacity=0.1,
                fill_color="#ff8800",
            )
            self.m.add_layer(layer)
            self.test_roi_layers.append(layer)

    def _init_default_rois(self) -> None:
        # Train ROI: left upper (47.167811,7.273987), right lower (47.163849,7.286448)
        self._set_roi_from_bounds("train", 47.163849, 47.167811, 7.273987, 7.286448)
        # Test ROI: left upper (47.171467,7.285832), right lower (47.169003,7.300424)
        self._set_roi_from_bounds("test", 47.169003, 47.171467, 7.285832, 7.300424)
        self._recompute_patches()

    def _on_draw(self, target: DrawControl, action: str, geo_json: dict, **kwargs) -> None:
        if action != "created" or not geo_json:
            return
        geom_type = geo_json.get("geometry", {}).get("type")
        if geom_type != "Polygon":
            return

        # remove the object from DrawControl data
        try:
            if geo_json in target.data:
                target.data.remove(geo_json)
        except Exception:
            pass

        coords = geo_json["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        if self.active_roi_target == "train":
            self.train_rois_geojson.append(geo_json)
            layer = Rectangle(
                bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                color="#00aa00",
                weight=3,
                fill_opacity=0.1,
                fill_color="#00aa00",
            )
            self.m.add_layer(layer)
            self.train_roi_layers.append(layer)
        else:
            self.test_rois_geojson.append(geo_json)
            layer = Rectangle(
                bounds=[(min_lat, min_lon), (max_lat, max_lon)],
                color="#ff8800",
                weight=3,
                fill_opacity=0.1,
                fill_color="#ff8800",
            )
            self.m.add_layer(layer)
            self.test_roi_layers.append(layer)

        self._recompute_patches()

    def _clear_rois(self, *_):
        self.train_rois_geojson = []
        self.test_rois_geojson = []
        for layer in list(self.train_roi_layers):
            try:
                self.m.remove_layer(layer)
            except Exception:
                pass
        for layer in list(self.test_roi_layers):
            try:
                self.m.remove_layer(layer)
            except Exception:
                pass
        self.train_roi_layers = []
        self.test_roi_layers = []
        self.train_patch_coords = []
        self.test_patch_coords = []
        self.infer_patch_coords = []
        self._set_patch_group(self.train_patch_group, [])
        self._set_patch_group(self.test_patch_group, [])
        self._update_patch_count_label()
        try:
            self.draw_control.clear()
        except Exception:
            try:
                self.draw_control.data = []
            except Exception:
                pass
        self._set_active_roi_target("train")
        with self.output_log:
            self.output_log.clear_output(wait=True)
            print("Cleared Train/Test ROIs and patch grids.")

    def _undo_last_train_roi(self, *_):
        if not self.train_rois_geojson:
            return
        self.train_rois_geojson.pop()
        if self.train_roi_layers:
            layer = self.train_roi_layers.pop()
            try:
                self.m.remove_layer(layer)
            except Exception:
                pass
        self._recompute_patches()

    def _undo_last_test_roi(self, *_):
        if not self.test_rois_geojson:
            return
        self.test_rois_geojson.pop()
        if self.test_roi_layers:
            layer = self.test_roi_layers.pop()
            try:
                self.m.remove_layer(layer)
            except Exception:
                pass
        self._recompute_patches()

    # -----------------
    # Train / Predict
    # -----------------
    def _compute_normalization(self, coords: list[tuple[int, int]], ps: int, max_pixels: int = 200_000) -> None:
        rng = np.random.default_rng(42)
        if not coords:
            self.mean = None
            self.std = None
            return
        c = self.embedding_mosaic.shape[2]
        n_patches = min(len(coords), 200)
        chosen = [coords[i] for i in rng.choice(len(coords), size=n_patches, replace=False)]
        per_patch = max(1, max_pixels // n_patches)
        samples = []
        for (r0, c0) in chosen:
            patch = self.embedding_mosaic[r0 : r0 + ps, c0 : c0 + ps, :].reshape(-1, c)
            if patch.shape[0] > per_patch:
                idx = rng.choice(patch.shape[0], size=per_patch, replace=False)
                patch = patch[idx]
            samples.append(patch)
        x = np.concatenate(samples, axis=0)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def _masked_smooth_l1(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # kept for backwards-compat; training uses masked MSE per reference script
        loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")
        loss = loss * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _masked_mse(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred/y/mask: (B,1,H,W); mask float {0,1}
        loss = (pred - y) ** 2
        loss = loss * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _regression_metrics(self, pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
        # pred/y: (B,1,H,W), mask: (B,1,H,W) float
        m = mask > 0
        if torch.count_nonzero(m) == 0:
            return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
        p = pred[m].detach().cpu().numpy()
        t = y[m].detach().cpu().numpy()
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(mse))
        return {"mse": mse, "mae": mae, "rmse": rmse}

    def _train_unet(self, *_):
        with self.output_log:
            self.output_log.clear_output(wait=True)
            # For training we require GT-valid patches
            train_coords = self._filter_coords_with_valid_gt(self.train_patch_coords) if self.train_patch_coords else []
            test_coords = self._filter_coords_with_valid_gt(self.test_patch_coords) if self.test_patch_coords else []

            if not train_coords:
                print("ERROR: No train patches with valid GT pixels. Check Train ROI / GT coverage.")
                return
            if not test_coords:
                print("WARNING: No test patches with valid GT pixels. Will train without test loss.")

            ps = int(self.patch_size.value)
            print(f"Device: {self.device}")
            print(f"Train patches (all): {len(self.train_patch_coords):,} | Test patches (all): {len(self.test_patch_coords):,}")
            print(f"Train patches (gt-valid): {len(train_coords):,} | Test patches (gt-valid): {len(test_coords):,}")
            print(f"Patch size: {ps} | overlap {int(self.overlap_pct.value)}% | stride {self._stride_px()}")

            if self.normalize_embeddings.value:
                print("Computing embedding normalization (mean/std) from train patches...")
                self._compute_normalization(train_coords, ps)
                print("Normalization ready.")
            else:
                self.mean = None
                self.std = None

            train_ds = self._PatchDataset(
                self.embedding_mosaic, self.height_gt, train_coords, ps, self.height_gt_nodata, self.mean, self.std
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=int(self.batch_size.value),
                shuffle=True,
                num_workers=int(self.num_workers.value),
                pin_memory=(self.device.type == "cuda"),
            )
            if test_coords:
                test_ds = self._PatchDataset(
                    self.embedding_mosaic, self.height_gt, test_coords, ps, self.height_gt_nodata, self.mean, self.std
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=int(self.batch_size.value),
                    shuffle=False,
                    num_workers=int(self.num_workers.value),
                    pin_memory=(self.device.type == "cuda"),
                )
            else:
                test_loader = None

            # UNet design + training strategy follows reference script
            self.net = self._UNetRegression(in_channels=self.embedding_mosaic.shape[2], features=[128, 256, 512], dropout=0.1).to(self.device)
            criterion = nn.MSELoss(reduction="none")  # per-pixel
            opt = torch.optim.Adam(self.net.parameters(), lr=float(self.lr.value), weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=10
            )

            print("\nUNET TRAINING STARTED...")
            for epoch in range(1, int(self.epochs.value) + 1):
                self.net.train()
                running = 0.0
                denom = 0
                for x, y, m in tqdm(train_loader, desc=f"Epoch {epoch}/{int(self.epochs.value)} [train]", leave=False):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    m = m.to(self.device, non_blocking=True).float()
                    opt.zero_grad(set_to_none=True)
                    pred = self.net(x)
                    # masked MSE: only valid pixels
                    loss_per_pixel = criterion(pred, y)
                    loss = (loss_per_pixel * m).sum() / m.sum().clamp(min=1.0)
                    loss.backward()
                    opt.step()
                    running += float(loss.item())
                    denom += 1
                train_loss = running / max(1, denom)

                if test_loader is not None:
                    self.net.eval()
                    t_running = 0.0
                    t_denom = 0
                    all_m = []
                    all_mae = []
                    all_rmse = []
                    with torch.no_grad():
                        for x, y, m in tqdm(test_loader, desc=f"Epoch {epoch}/{int(self.epochs.value)} [test]", leave=False):
                            x = x.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)
                            m = m.to(self.device, non_blocking=True).float()
                            pred = self.net(x)
                            loss_per_pixel = criterion(pred, y)
                            loss = (loss_per_pixel * m).sum() / m.sum().clamp(min=1.0)
                            t_running += float(loss.item())
                            t_denom += 1
                            metrics = self._regression_metrics(pred, y, m)
                            if np.isfinite(metrics["mse"]):
                                all_m.append(metrics["mse"])
                                all_mae.append(metrics["mae"])
                                all_rmse.append(metrics["rmse"])
                    test_loss = t_running / max(1, t_denom)
                    scheduler.step(test_loss)
                    lr_now = float(opt.param_groups[0]["lr"])
                    msg = f"Epoch {epoch:03d}: train_loss={train_loss:.6f} | test_loss={test_loss:.6f} | lr={lr_now:.2e}"
                    if all_rmse:
                        msg += f" | test_rmse={float(np.mean(all_rmse)):.4f} | test_mae={float(np.mean(all_mae)):.4f}"
                    print(msg)
                else:
                    scheduler.step(train_loss)
                    lr_now = float(opt.param_groups[0]["lr"])
                    print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} | lr={lr_now:.2e}")

            print("\nUNET TRAINING DONE ✓")
            print("Now click Predict & Stitch.")

    def _predict_and_stitch(self, *_):
        with self.output_log:
            self.output_log.clear_output(wait=True)
            if self.net is None:
                print("ERROR: Train UNet first.")
                return

            infer = self.inference_region.value
            rois = self.train_rois_geojson if infer == "Train ROI" else self.test_rois_geojson
            if not rois:
                print(f"ERROR: {infer} not set.")
                return

            ps = int(self.patch_size.value)
            # Inference should NOT require GT coverage (can predict anywhere inside ROI)
            coords = self._gen_patch_coords_for_rois(rois, require_valid_gt=False)
            self.infer_patch_coords = coords
            self._update_patch_count_label()
            if not coords:
                print("No inference patches found.")
                return

            print(f"Inference on: {infer} | patches={len(coords):,} | patch_size={ps} | overlap={int(self.overlap_pct.value)}%")

            h, w, _ = self.embedding_mosaic.shape
            pred_sum = np.zeros((h, w), dtype=np.float32)
            pred_w = np.zeros((h, w), dtype=np.float32)
            roi_mask = self._rois_mask(rois)

            self.net.eval()
            bs = int(self.batch_size.value)
            with torch.no_grad():
                for i in tqdm(range(0, len(coords), bs), desc="Predict [patches]", leave=False):
                    batch = coords[i : i + bs]
                    xs = []
                    for (r0, c0) in batch:
                        x = self.embedding_mosaic[r0 : r0 + ps, c0 : c0 + ps, :].transpose(2, 0, 1).astype(np.float32, copy=False)
                        if self.mean is not None and self.std is not None:
                            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
                        xs.append(x)
                    x_t = torch.from_numpy(np.stack(xs, axis=0)).to(self.device)
                    pred = self.net(x_t).cpu().numpy()[:, 0, :, :]
                    for (r0, c0), p in zip(batch, pred):
                        rr = slice(r0, r0 + ps)
                        cc = slice(c0, c0 + ps)
                        m = roi_mask[rr, cc]
                        pred_sum[rr, cc][m] += p[m]
                        pred_w[rr, cc][m] += 1.0

            pred_full = np.full((h, w), self.height_gt_nodata, dtype=np.float32)
            ok = pred_w > 0
            pred_full[ok] = pred_sum[ok] / pred_w[ok]
            self.last_prediction = pred_full

            gt_valid = self._valid_gt_mask()
            err = np.full((h, w), self.height_gt_nodata, dtype=np.float32)
            both = ok & gt_valid
            err[both] = pred_full[both] - self.height_gt[both]
            self.last_err = err

            # Print quantitative sanity-check metrics (only where GT is valid and predicted)
            if np.any(both):
                e = err[both]
                mae = float(np.mean(np.abs(e)))
                rmse = float(np.sqrt(np.mean(e ** 2)))
                bias = float(np.mean(e))
                gt_vals = self.height_gt[both]
                pred_vals = pred_full[both]
                print(f"Eval (ROI valid pixels): n={both.sum():,} | MAE={mae:.3f} | RMSE={rmse:.3f} | bias={bias:.3f}")
                print(
                    f"GT stats:  min={float(gt_vals.min()):.3f} max={float(gt_vals.max()):.3f} mean={float(gt_vals.mean()):.3f}"
                )
                print(
                    f"Pred stats:min={float(pred_vals.min()):.3f} max={float(pred_vals.max()):.3f} mean={float(pred_vals.mean()):.3f}"
                )
            else:
                print("Eval: no overlapping valid GT pixels in this ROI (GT is nodata here).")

            print("Stitching done. Updating overlays...")
            self._update_pred_err_overlays()
            print("OK: prediction/error overlays updated.")

    # -----------------
    # Overlays + export
    # -----------------
    def _array_to_rgba_data_url(self, arr: np.ndarray, *, cmap: str, vmin: float, vmax: float, nodata: float) -> str:
        from matplotlib import cm
        data = arr.astype(np.float32, copy=False)
        valid = np.isfinite(data) & (data != nodata)
        rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.float32)
        if np.any(valid):
            d = data.copy()
            d[~valid] = vmin
            d = np.clip(d, vmin, vmax)
            norm = (d - vmin) / (vmax - vmin + 1e-12)
            rgba[:, :, :] = cm.get_cmap(cmap)(norm)
            rgba[:, :, 3] = valid.astype(np.float32)
        buffer = io.BytesIO()
        plt.imsave(buffer, rgba, format="png")
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def _update_pred_err_overlays(self):
        if not self.show_gt_checkbox.value and self.gt_layer and self.gt_layer in self.m.layers:
            self.m.remove_layer(self.gt_layer)
            self.gt_layer = None
        if not self.show_pred_checkbox.value and self.pred_layer and self.pred_layer in self.m.layers:
            self.m.remove_layer(self.pred_layer)
            self.pred_layer = None
        if not self.show_err_checkbox.value and self.err_layer and self.err_layer in self.m.layers:
            self.m.remove_layer(self.err_layer)
            self.err_layer = None

        opacity = float(self.overlay_opacity.value)

        if self.show_gt_checkbox.value:
            gt = self.height_gt
            valid = self._valid_gt_mask()
            if np.any(valid):
                clip = int(self.value_clip_pct.value)
                vmax = float(np.percentile(gt[valid], clip))
                vmin = float(np.percentile(gt[valid], 100 - clip))
            else:
                vmin, vmax = 0.0, 1.0
            url = self._array_to_rgba_data_url(gt, cmap="viridis", vmin=vmin, vmax=vmax, nodata=self.height_gt_nodata)
            if self.gt_layer and self.gt_layer in self.m.layers:
                self.m.remove_layer(self.gt_layer)
            self.gt_layer = ImageOverlay(url=url, bounds=self.vis_bounds, opacity=opacity, name="GT")
            self.m.add(self.gt_layer)

        if self.show_pred_checkbox.value and self.last_prediction is not None:
            pred = self.last_prediction
            valid = np.isfinite(pred) & (pred != self.height_gt_nodata)
            if np.any(valid):
                clip = int(self.value_clip_pct.value)
                vmax = float(np.percentile(pred[valid], clip))
                vmin = float(np.percentile(pred[valid], 100 - clip))
            else:
                vmin, vmax = 0.0, 1.0
            url = self._array_to_rgba_data_url(pred, cmap="viridis", vmin=vmin, vmax=vmax, nodata=self.height_gt_nodata)
            if self.pred_layer and self.pred_layer in self.m.layers:
                self.m.remove_layer(self.pred_layer)
            self.pred_layer = ImageOverlay(url=url, bounds=self.vis_bounds, opacity=opacity, name="Pred")
            self.m.add(self.pred_layer)

        if self.show_err_checkbox.value and self.last_err is not None:
            err = self.last_err
            valid = np.isfinite(err) & (err != self.height_gt_nodata)
            if np.any(valid):
                clip = int(self.err_clip_pct.value)
                vmax = float(np.percentile(np.abs(err[valid]), clip))
                vmin = -vmax
            else:
                vmin, vmax = -1.0, 1.0
            url = self._array_to_rgba_data_url(err, cmap="coolwarm", vmin=vmin, vmax=vmax, nodata=self.height_gt_nodata)
            if self.err_layer and self.err_layer in self.m.layers:
                self.m.remove_layer(self.err_layer)
            self.err_layer = ImageOverlay(url=url, bounds=self.vis_bounds, opacity=opacity, name="Err")
            self.m.add(self.err_layer)

    def _export_singleband(self, out_fp: str, arr: np.ndarray):
        # sanitize non-finite values for GIS friendliness
        arr = arr.astype(np.float32, copy=False)
        arr = np.where(np.isfinite(arr), arr, self.height_gt_nodata).astype(np.float32, copy=False)

        with rasterio.open(
            out_fp,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=self.mosaic_transform,
            nodata=self.height_gt_nodata,
            tiled=True,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        ) as ds:
            ds.write(arr.astype(np.float32), 1)
            # Write statistics tags so QGIS doesn't show +/-Inf before computing stats
            valid = np.isfinite(arr) & (arr != self.height_gt_nodata)
            if np.any(valid):
                v = arr[valid]
                ds.update_tags(
                    1,
                    STATISTICS_MINIMUM=str(float(v.min())),
                    STATISTICS_MAXIMUM=str(float(v.max())),
                    STATISTICS_MEAN=str(float(v.mean())),
                    STATISTICS_STDDEV=str(float(v.std())),
                )
            else:
                ds.update_tags(
                    1,
                    STATISTICS_MINIMUM="0",
                    STATISTICS_MAXIMUM="0",
                    STATISTICS_MEAN="0",
                    STATISTICS_STDDEV="0",
                )
        with self.output_log:
            self.output_log.clear_output(wait=True)
            print(f"Exported {out_fp}")

    def _export_pred(self, *_):
        if self.last_prediction is None:
            with self.output_log:
                self.output_log.clear_output(wait=True)
                print("Nothing to export: run Predict & Stitch first.")
            return
        self._export_singleband("pred_height_on_embedding_grid.tif", self.last_prediction)

    def _export_err(self, *_):
        if self.last_err is None:
            with self.output_log:
                self.output_log.clear_output(wait=True)
                print("Nothing to export: run Predict & Stitch first.")
            return
        self._export_singleband("err_pred_minus_gt_on_embedding_grid.tif", self.last_err)

    def _export_gt_aligned(self, *_):
        # Export the in-memory GT (already aligned/warped to embedding grid) for easy comparison in GIS.
        self._export_singleband("gt_aligned_on_embedding_grid.tif", self.height_gt)

    def _clear_pred_err(self, *_):
        self.last_prediction = None
        self.last_err = None
        for lyr_attr in ["pred_layer", "err_layer"]:
            lyr = getattr(self, lyr_attr)
            if lyr and lyr in self.m.layers:
                self.m.remove_layer(lyr)
            setattr(self, lyr_attr, None)
        with self.output_log:
            self.output_log.clear_output(wait=True)
            print("Cleared prediction/error layers.")

    # -----------------
    # Events
    # -----------------
    def _setup_event_handlers(self):
        self.draw_control.on_draw(self._on_draw)

        self.draw_train_button.observe(lambda ch: ch["new"] and self._set_active_roi_target("train"), names="value")
        self.draw_test_button.observe(lambda ch: ch["new"] and self._set_active_roi_target("test"), names="value")
        self.undo_train_roi_button.on_click(self._undo_last_train_roi)
        self.undo_test_roi_button.on_click(self._undo_last_test_roi)
        self.clear_rois_button.on_click(self._clear_rois)

        self.basemap_selector.observe(self.on_basemap_change, names="value")

        def _on_embedding_vis(change=None):
            self.embedding_overlay.opacity = float(self.embedding_opacity.value) if self.show_embedding_toggle.value else 0.0

        self.show_embedding_toggle.observe(_on_embedding_vis, names="value")
        self.embedding_opacity.observe(_on_embedding_vis, names="value")

        # patch params update patches
        self.patch_size.observe(lambda ch: self._recompute_patches(), names="value")
        self.overlap_pct.observe(lambda ch: self._recompute_patches(), names="value")
        self.show_patch_grid.observe(lambda ch: self._refresh_patch_overlays(), names="value")
        self.inference_region.observe(lambda ch: self._update_patch_count_label(), names="value")

        # overlays
        self.show_gt_checkbox.observe(lambda ch: self._update_pred_err_overlays(), names="value")
        self.show_pred_checkbox.observe(lambda ch: self._update_pred_err_overlays(), names="value")
        self.show_err_checkbox.observe(lambda ch: self._update_pred_err_overlays(), names="value")
        self.overlay_opacity.observe(lambda ch: self._update_pred_err_overlays(), names="value")
        self.value_clip_pct.observe(lambda ch: self._update_pred_err_overlays(), names="value")
        self.err_clip_pct.observe(lambda ch: self._update_pred_err_overlays(), names="value")

        # train/predict/export
        self.train_button.on_click(self._train_unet)
        self.predict_button.on_click(self._predict_and_stitch)
        self.clear_pred_button.on_click(self._clear_pred_err)
        self.export_pred_button.on_click(self._export_pred)
        self.export_err_button.on_click(self._export_err)
        self.export_gt_button.on_click(self._export_gt_aligned)

    def on_basemap_change(self, change: dict) -> None:
        new_basemap_name = change["new"]
        new_layer = self.basemap_layers[new_basemap_name]
        if self.current_basemap in self.m.layers:
            self.m.remove_layer(self.current_basemap)
        self.m.add_layer(new_layer)
        self.current_basemap = new_layer
