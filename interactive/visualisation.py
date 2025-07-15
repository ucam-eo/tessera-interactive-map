import base64
import io
import json
from functools import partial

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from ipyleaflet import CircleMarker, ImageOverlay, Map, TileLayer
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
)
from rasterio.transform import array_bounds
from sklearn.decomposition import PCA


def visualise_embedding(embedding_mosaic, mosaic_transform) -> tuple:
    """
    Visualise an embedding mosaic using PCA.
    """
    mosaic_height, mosaic_width, num_channels = embedding_mosaic.shape
    # The mosaic is now in EPSG:4326, so its bounds are already in lat/lon
    west, south, east, north = array_bounds(
        mosaic_height, mosaic_width, mosaic_transform
    )
    VIS_BOUNDS = ((south, west), (north, east))

    # --- PCA VISUALIZATION ---
    print("\nCreating PCA-based visualization...")
    pixels = embedding_mosaic.reshape(-1, num_channels)
    n_sample = min(pixels.shape[0], 100000)
    sample_indices = np.random.choice(pixels.shape[0], n_sample, replace=False)
    pca = PCA(n_components=3)
    pca.fit(pixels[sample_indices, :])
    transformed_pixels = pca.transform(pixels)
    pca_image = transformed_pixels.reshape(mosaic_height, mosaic_width, 3)
    print("Normalizing PCA components for display...")
    vis_mosaic = np.zeros_like(pca_image)
    for i in range(3):
        channel = pca_image[:, :, i]
        min_val, max_val = np.percentile(channel, [2, 98])
        if max_val > min_val:
            vis_mosaic[:, :, i] = np.clip(
                (channel - min_val) / (max_val - min_val), 0, 1
            )
    print("PCA visualization created.")

    buffer = io.BytesIO()
    plt.imsave(buffer, vis_mosaic, format="png")
    buffer.seek(0)
    b64_data = base64.b64encode(buffer.read()).decode("utf-8")
    VIS_DATA_URL = f"data:image/png;base64,{b64_data}"
    return VIS_BOUNDS, VIS_DATA_URL


# -- 3. MAPPING TOOL --


class InteractiveMappingTool:
    """Interactive mapping tool for labeling training points on satellite imagery."""

    def __init__(self, min_lat, max_lat, min_lon, max_lon, vis_data_url, vis_bounds):
        self.training_points = []
        self.markers = {}
        self.A_MARKER_WAS_JUST_REMOVED = False
        self.class_color_map = {}
        self.tab10_cmap = plt.colormaps.get_cmap("tab10")
        self.classification_layer = None

        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.vis_data_url = vis_data_url
        self.vis_bounds = vis_bounds

        # initialize the tool
        self._setup_initial_classes()
        self._create_widgets()
        self._create_map()
        self._setup_event_handlers()
        self._create_layout()

    def get_or_assign_color_for_class(self, class_name):
        """Assigns a consistent color if one doesn't exist, otherwise returns existing color."""
        if class_name not in self.class_color_map:
            new_color_index = len(self.class_color_map) % 10
            self.class_color_map[class_name] = mcolors.to_hex(
                self.tab10_cmap(new_color_index)
            )
        return self.class_color_map[class_name]

    def _setup_initial_classes(self):
        initial_classes = ["Water", "Urban"]
        for c in initial_classes:
            self.get_or_assign_color_for_class(c)
        self.initial_classes = initial_classes

    def _create_widgets(self):
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

        self.basemap_selector = Dropdown(
            options=list(self.basemap_layers.keys()),
            value="Esri Satellite",
            description="Basemap:",
        )

    def _create_map(self):
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
            opacity=self.opacity_slider.value if self.opacity_toggle.value else 0,
        )
        self.m.add(self.image_overlay)

    def update_legend(self):
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

    def update_opacity(self, change):
        is_visible = self.opacity_toggle.value
        opacity_value = self.opacity_slider.value if is_visible else 0
        self.image_overlay.opacity = opacity_value

        # Update classification layer opacity if it exists
        if self.classification_layer and self.classification_layer in self.m.layers:
            self.classification_layer.opacity = opacity_value

        self.opacity_slider.disabled = not is_visible

    def on_add_class_button_clicked(self, b):
        # CHANGE: Convert to method, use self attributes
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

    def on_basemap_change(self, change):
        # CHANGE: Convert to method, use self attributes
        new_basemap_name = change["new"]
        new_layer = self.basemap_layers[new_basemap_name]
        if self.current_basemap in self.m.layers:
            self.m.remove_layer(self.current_basemap)
        self.m.add_layer(new_layer)
        self.current_basemap = new_layer

    def on_class_selection_change(self, change):
        """When dropdown changes, update the color picker to match."""
        # CHANGE: Convert to method, use self attributes
        selected_class = change.new
        color = self.get_or_assign_color_for_class(selected_class)
        self.color_picker.unobserve(self.on_color_change, names="value")
        self.color_picker.value = color
        self.color_picker.observe(self.on_color_change, names="value")

    def on_color_change(self, change):
        """When color picker changes, update the map and redraw pins."""
        # CHANGE: Convert to method, use self attributes
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

                # Attach the click-to-remove handler to the recolored marker
                recolored_marker.on_click(partial(self.remove_marker, marker_key))

                self.m.add(recolored_marker)
                self.markers[marker_key] = recolored_marker

    def remove_marker(self, marker_key, **kwargs):
        # CHANGE: Convert to method, use self attributes
        # Remove from map
        if marker_key in self.markers:
            self.m.remove_layer(self.markers[marker_key])
            del self.markers[marker_key]

        # Remove from training data
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

    def handle_map_click(self, **kwargs):
        # CHANGE: Convert to method, use self attributes
        # If a marker was just deleted, this click was used
        # Ignore it and reset the flag for the next click
        if self.A_MARKER_WAS_JUST_REMOVED:
            self.A_MARKER_WAS_JUST_REMOVED = False
            return

        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if coords is None:
                return
            selected_class = self.class_dropdown.value

            marker_key = tuple(coords)
            if marker_key in self.markers:
                with self.output_log:
                    self.output_log.clear_output(wait=True)
                    print(
                        "A point already exists at this exact location. Click it to remove."
                    )
                return

            self.training_points.append((coords, selected_class))
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

    def on_clear_pins_button_clicked(self, b):
        # CHANGE: Convert to method, use self attributes
        with self.output_log:
            for key, marker in self.markers.items():
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

    def on_clear_classification_clicked(self, b):
        # CHANGE: Convert to method, use self attributes
        if self.classification_layer and self.classification_layer in self.m.layers:
            self.m.remove_layer(self.classification_layer)
            self.classification_layer = None
            self.clear_classification_button.disabled = True
            with self.output_log:
                self.output_log.clear_output()
                print("Classification layer removed.")

    def on_save_button_clicked(self, b):
        # CHANGE: Convert to method, use self attributes
        fname = self.filename_text.value
        if not fname:
            with self.output_log:
                self.output_log.clear_output()
                print("Error: Please provide a filename.")
            return

        # Bundle both the points and the color map together for save state
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

    def on_load_button_clicked(self, b):
        # CHANGE: Convert to method, use self attributes
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

        # Re draw all markers on the map
        for point_data in loaded_points:
            coords, class_name = point_data

            # Add class to dropdown
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
        # CHANGE: Convert event handler setup to method
        self.opacity_toggle.observe(self.update_opacity, names="value")
        self.opacity_slider.observe(self.update_opacity, names="value")
        self.add_class_button.on_click(self.on_add_class_button_clicked)
        self.class_dropdown.observe(self.on_class_selection_change, names="value")
        self.color_picker.observe(self.on_color_change, names="value")
        self.m.on_interaction(self.handle_map_click)
        self.clear_pins_button.on_click(self.on_clear_pins_button_clicked)
        self.clear_classification_button.on_click(self.on_clear_classification_clicked)
        self.basemap_selector.observe(self.on_basemap_change, names="value")
        self.save_button.on_click(self.on_save_button_clicked)
        self.load_button.on_click(self.on_load_button_clicked)

    def _create_layout(self):
        # CHANGE: Convert layout creation to method
        class_controls = HBox([self.class_dropdown, self.color_picker])
        new_class_controls = HBox([self.new_class_text, self.add_class_button])
        opacity_controls = HBox([self.opacity_toggle, self.opacity_slider])
        controls = VBox(
            [
                self.basemap_selector,
                class_controls,
                new_class_controls,
                opacity_controls,
            ]
        )

        top_bar = HBox(
            [controls, self.legend_widget],
            layout=Layout(
                width="100%",
                height="auto",
                justify_content="flex-start",
                align_items="flex-start",
            ),
        )

        self.legend_widget.layout = Layout(
            flex="1", margin="0 0 0 20px", overflow="auto"
        )

        buttons = HBox(
            [
                self.classify_button,
                self.clear_pins_button,
                self.clear_classification_button,
            ]
        )
        file_controls = HBox([self.filename_text, self.save_button, self.load_button])
        self.ui = VBox([top_bar, self.m, buttons, file_controls, self.output_log])

    def display(self):
        # CHANGE: Add method to display the UI and update legend
        display(self.ui)
        self.update_legend()

    # CHANGE: Add method to get training data (useful for external access)
    def get_training_data(self):
        """Return current training points and class colors."""
        return {
            "training_points": self.training_points,
            "class_color_map": self.class_color_map,
        }


# CHANGE: Usage would now be:
# mapping_tool = InteractiveMappingTool(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, VIS_DATA_URL, VIS_BOUNDS)
# mapping_tool.display()
