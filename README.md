# Interactive Tessera Embedding Classifier

This repository contains a Jupyter notebook based tool for interactive, human-in-the-loop classification of geospatial data using the [Tessera foundation model](https://github.com/ucam-eo/tessera) embeddings.

The tool allows a user to define an area of interest, visualize the high-dimensional embedding data with PCA, and iteratively train a machine learning model by simply clicking on the map to label.

## Features

-   **Interactive Map Interface**: Pan and zoom on a satellite or terrain basemap.
-   **Data-Driven Visualization**: Uses PCA to create a false-color visualization of Tessera embeddings.
-   **Point-and-Click Training**: Simply click on the map to add labeled training points.
-   **Custom Classes & Colors**: Dynamically add new classes and customize their colors with a color picker.
-   **Live Classification**: Train a k-Nearest Neighbors model and classify the map with a click.
-   **Iterative Refinement**: Add more pins to correct mistakes and re-run the classification for immediate (relatively) feedback.

## Installation

This tool is designed to run in a local Python environment.

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/ucam-eo/tessera-interactive-map.git
    cd tessera-interactive-map
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    The required packages are listed in `requirements.txt`. Additionally, you need to install the `geotessera` library directly from its Git repository.

    ```bash
    pip install -r requirements.txt
    pip install git+https://github.com/ucam-eo/geotessera.git
    ```
    For jupyter lab, also run `python -m ipykernel install --user --name=tessera` to register the kernel
    
## How to Run

1.  **Open the Project**:
    Open the cloned repository folder in VS Code or Jupyter Notebook, or whatever else that runs ipynb files.

2.  **Open the Notebook**:
    Using the file browser on the left, find and open the main notebook file.

3.  **Run the Cells**:
    The notebook is organized into sequential steps. Run each cell in order from top to bottom.

    -   **Step 1: Setup**: Defines the Region of Interest (ROI) by setting latitude/longitude boundaries.
    -   **Step 2: Data Fetching & Visualization**: Downloads the required Tessera embedding tiles for your ROI, stitches them together, and uses PCA to generate the false color image. This step may take a while.
    -   **Step 3: Interactive UI**: This is the main interface. It displays the map, the embedding overlay, and all the widgets.
    -   **Step 4: Classification Logic**: This cell defines the function that is triggered when you click the "Classify" button. **You only need to run this cell once** to define the function.

    Note: run all four stages, it's necessary to run step 4 for the classification to work even if the map appears at stage 3.
    
## How to Use the Tool

Once the UI in Step 3 is displayed, you can begin your analysis:

1.  **Explore**: Use the **Basemap** dropdown and the **Opacity** slider to compare the embedding visualization with the underlying satellite/street map.
2.  **Define Classes**: Use the **Class** dropdown to select a land cover type. You can add new classes by typing in the "New Class" box and clicking "Add". Customize the color for the selected class using the **Set Color** picker.
3.  **Add Pins**: Click on the map to label features. A colored pin will appear, corresponding to the currently selected class and color.
4.  **Classify**: Once you have labeled a few points for at least two different classes, zoom into an area you're interested in and click the **Classify** button. The tool will train a model on your pins and display a colored classification map over the current viewport.
5.  **Refine**: If the classification is incorrect, add more pins to the misclassified areas (or add pins for a new class you missed) and click "Classify" again. The result will update based on your new training data.
6.  **Clear**: Use the "Clear All Pins" and "Clear Classification" buttons to reset your work.
