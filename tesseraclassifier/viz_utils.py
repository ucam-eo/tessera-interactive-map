from ipywidgets import Layout
import json
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from pyproj import Transformer
import pandas as pd
from typing import Dict
from tesseraclassifier import classifier_utils
import json
import base64
from io import BytesIO

import json
from typing import Dict, List, Tuple

# Assume these other imports are available for the interactive map function
import base64
from io import BytesIO
import numpy as np
import rasterio
import rasterio.warp
from PIL import Image
import ipywidgets as widgets
from ipyleaflet import (Map, TileLayer, CircleMarker, LayerGroup, 
                        FullScreenControl, LayersControl, ImageOverlay, WidgetControl)
from IPython.display import display
import geopandas as gpd
import ipywidgets as widgets
from ipyleaflet import Map, TileLayer, CircleMarker, LayerGroup, FullScreenControl, LayersControl, ImageOverlay
from IPython.display import display
import numpy as np
import rasterio
import rasterio.warp
from PIL import Image
from typing import Dict

def create_interactive_map(
    classified_raster_path: str,
    training_points: list,
    id_to_label: Dict[int, str],
    class_colors_hex: Dict[str, str]
):
    """
    Creates a correctly aligned interactive map with a legend by pre-warping 
    the raster to the Web Mercator projection before display.

    Args:
        classified_raster_path: Path to the classified GeoTIFF file.
        training_points: A list of dictionaries for the labeled points.
        id_to_label: Mapping from integer IDs back to string labels.
        class_colors_hex: Mapping from string labels to hex color codes.
    """
    # 1. Create a numeric colormap from the provided mappings
    max_id = max(id_to_label.keys()) if id_to_label else 0
    colormap = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for class_id, label in id_to_label.items():
        hex_color = class_colors_hex.get(label, "#FF00FF") # Default to magenta
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        colormap[class_id] = rgb

    # 2. Pre-warp the raster to Web Mercator (EPSG:3857)
    print("Reprojecting classified raster to Web Mercator...")
    with rasterio.open(classified_raster_path) as src:
        dst_crs = 'EPSG:3857'
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        warped_data = np.zeros((height, width), dtype=np.uint8)

        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=warped_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.nearest
        )
        
        # 3. Calculate the WGS84 (Lat/Lon) bounds of the new warped image
        warped_bounds = rasterio.warp.transform_bounds(
            dst_crs, 'EPSG:4326', *rasterio.transform.array_bounds(height, width, transform)
        )
        left, bottom, right, top = warped_bounds
        image_bounds = [[bottom, left], [top, right]]

    # 4. Convert the warped data to a colorized PNG
    print("Converting warped raster to displayable image...")
    nodata_value = src.nodata if src.nodata is not None else -1
    alpha = np.where(warped_data == nodata_value, 0, 255).astype(np.uint8)
    rgb_image_data = colormap[warped_data]
    rgba_image_data = np.dstack((rgb_image_data, alpha))

    image = Image.fromarray(rgba_image_data, 'RGBA')
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 5. Create the map and overlays
    center_lat = (image_bounds[0][0] + image_bounds[1][0]) / 2
    center_lon = (image_bounds[0][1] + image_bounds[1][1]) / 2

    map_layout = Layout(height='600px', width='100%')
    m = Map(center=(center_lat, center_lon), zoom=10, layout=map_layout)
    m.add(TileLayer(url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri Satellite'))
    m.add(LayersControl(position='topright'))
    m.add(FullScreenControl())

    classification_overlay = ImageOverlay(url=data_url, bounds=image_bounds, name="Classification")
    m.add(classification_overlay)
    
    markers = [CircleMarker(location=(p['lat'], p['lon']), radius=5, color="white", weight=1,
                            fill_color=class_colors_hex.get(p['label'], "#FF00FF"), fill_opacity=0.8)
               for p in training_points]
    points_layer = LayerGroup(layers=markers, name="Training Points")
    m.add(points_layer)

    # 6. Create Legend
    legend_html = "<h4>Classification Legend</h4>"
    legend_html += '<div style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px;">'
    for class_id, label in sorted(id_to_label.items()):
        color = class_colors_hex.get(label, "#FF00FF")
        legend_html += f'<div><i style="background:{color}; width: 15px; height: 15px; display: inline-block; border: 1px solid black; margin-right: 5px;"></i>{label}</div>'
    legend_html += "</div>"
    
    legend_widget = widgets.HTML(value=legend_html)
    legend_control = WidgetControl(widget=legend_widget, position='bottomright')
    m.add(legend_control)

    # 7. Create Opacity Control
    opacity_slider = widgets.FloatSlider(value=0.7, min=0, max=1.0, step=0.05, description='Opacity:')
    def update_opacity(change):
        classification_overlay.opacity = change['new']
    opacity_slider.observe(update_opacity, names='value')
    update_opacity({'new': opacity_slider.value})
    
    print("✅ Interactive map with legend created.")
    return widgets.VBox([m, opacity_slider])