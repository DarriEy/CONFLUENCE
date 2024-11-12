import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
from datetime import datetime
import shutil

class MESHPreProcessor:
    """
    Preprocessor for the MESH model.
    Handles data preparation, PET calculation, snow module setup, and file organization.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for MESH
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        gr_setup_dir (Path): Directory for MESH setup files
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.gr_setup_dir = self.project_dir / "settings" / "MESH"
        
        # MESH-specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_gr_path = self.project_dir / 'forcing' / 'MESH_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"


    def run_preprocessing(self):
        """Run the complete GR preprocessing workflow."""
        self.logger.info("Starting GR preprocessing")
        try:
            self.create_directories()
            self.prepare_forcing_data()
            #self.create_R_script()
            self.logger.info("GR preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during GR preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for GR model setup."""
        dirs_to_create = [
            self.mesh_setup_dir,
            self.forcing_mesh_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def prepare_forcing_data():
        return

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)
    
    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))


class MESHRunner:
    """
    Runner class for the MESH model.
    Handles model execution, state management, and output processing.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for MESH model
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        
        # MESH-specific paths
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_mesh_path = self.project_dir / 'forcing' / 'MESH_input'
        self.output_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'MESH'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Model configuration

    def run_MESH(self) -> Optional[Path]:
        """
        Run the MESH model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwis
        """
        return
        
class MESHPostProcessor:
    """
    Post processor for the MESH model
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.gr_setup_dir = self.project_dir / "settings" / "MESH"
        
        # MESH specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_gr_path = self.project_dir / 'forcing' / 'MESH_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
