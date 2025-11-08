# get repo root
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

repo_root = Path(__file__).parent.parent

# define filepaths
config_fp = repo_root / "config.yaml"


@dataclass
class Config:
    """Configuration class for interactive geospatial utilities.

    All parameters can be modified at runtime or loaded from YAML config file.
    """

    # Coordinate Reference System
    target_crs: str = "EPSG:4326"  # WGS84 default

    # Spatial parameters
    min_bbox_size: float = 0.1  # minimum bounding box size in degrees
    max_bbox_size: float = 10.0  # maximum bounding box size in degrees
    lat_coords: tuple = field(default_factory=lambda: (52.15, 52.25))
    lon_coords: tuple = field(default_factory=lambda: (0.05, 0.25))

    # Temporal
    target_year: int = 2024

    # Storage
    embeddings_dir: str = "./embeddings"

    # Visualisation
    n_samples: int = 100000
    percentiles: list[float] = field(default_factory=lambda: [2, 98])
    save_format: str = "png"

    def update(self, **kwargs) -> "Config":
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update

        Returns:
            Updated Config instance (for method chaining)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"{key} not found in config")
        return self

    def save_to_yaml(self, filepath: Optional[Path] = None) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Optional path to save config. Defaults to config_fp.
        """
        filepath = filepath or config_fp

        # convert to dictionary, handling Path objects
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value

        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @classmethod
    def from_yaml(cls, filepath: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file.

        Args:
            filepath: Optional path to load config from. Defaults to config_fp.

        Returns:
            Config instance loaded from YAML
        """
        filepath = filepath or config_fp

        if not filepath.exists():  # if config file does not exist
            return cls()  # return default config

        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_bbox_params(self) -> Dict[str, float]:
        """Get bounding box related parameters."""
        return {"min_bbox_size": self.min_bbox_size}

    def get_crs_params(self) -> Dict[str, str]:
        """Get coordinate reference system parameters."""
        return {"target_crs": self.target_crs}


# create default config instance
config = Config()

# try to load from YAML if it exists
if config_fp.exists():
    config = Config.from_yaml()
