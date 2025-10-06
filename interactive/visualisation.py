import base64
import io
import json
from functools import partial
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ipyleaflet import (
    CircleMarker,
    DrawControl,
    ImageOverlay,
    Map,
    Rectangle,
    TileLayer,
)
from IPython.display import display
from ipywidgets import (
    HTML,
    Button,
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
from rasterio.transform import array_bounds
from rasterio import transform
from sklearn.decomposition import PCA

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
            center=(52.2053, 0.1218), 
            zoom=13,
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

        state_file = "session/map_state.json"

        # Saving the session 
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                center = tuple(state["center"])
                zoom = state["zoom"]
            except (KeyError, ValueError):
                # fallback to Cambridge in case file is corrupted
                center, zoom = (52.2053, 0.1218), 13
        else:
            # default to Cambridge
            center, zoom = (52.2053, 0.1218), 13

            # make sure session folder exists and create the file
            os.makedirs("session", exist_ok=True)
            state = {"center": center, "zoom": zoom}
            with open(state_file, "w") as f:
                json.dump(state, f)

        # create map
        self.map = Map(
            center=center,
            zoom=zoom,
            layout={"width": "100%", "height": "500px"},
        )

        # save state on user interaction
        self.map.observe(self._save_map_state, names=["center", "zoom"])
        self._state_file = state_file
        
        self.init_draw_controls()

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

    def init_draw_controls(self):
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

                # removes the red rectangles from render by deleting the draw controls and reinstantiating them
                # not the best way to do it, but the red recs are tied to the draw controls 
                self.map.remove_layer(target)
                self.init_draw_controls()

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
                    fill_opacity=0.4,
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
    
    
    def _save_map_state(self, change=None):
        state = {
            "center": self.map.center,
            "zoom": self.map.zoom,
        }
        with open(self._state_file, "w") as f:
            json.dump(state, f)
