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

sys.path.append(str(Path(__file__).resolve().parent))
from hypeFlow import write_hype_forcing, write_hype_geo_files, write_hype_par_file, write_hype_info_filedir_files # type: ignore


class HYPEPreProcessor:
    """
    Preprocessor for the HYPE model.
    Handles data preparation, file organization, and setup.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.hype_setup_dir = self.project_dir / "settings" / "HYPE"
        
        # HYPE-specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_hype_path = self.project_dir / 'forcing' / 'HYPE_input'
        self.catchment_path = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        self.catchment_name = self.config.get('RIVER_BASINS_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"

    def run_preprocessing(self):
        """Run the complete HYPE preprocessing workflow."""
        self.logger.info("Starting HYPE preprocessing")
        try:
            self.create_directories()
            self.prepare_model_files()
            self.logger.info("HYPE preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during HYPE preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for HYPE model setup."""
        dirs_to_create = [
            self.hype_setup_dir,
            self.forcing_hype_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def prepare_model_files(self):
        """Prepare all necessary HYPE model input files."""
        try:
            # Prepare forcing data
            self.logger.info("Preparing HYPE forcing data")
            temp_var = 'airtemp'
            prec_var = 'pptrate'
            forcing_units = {
                'temperature': {
                    'in_varname': temp_var,
                    'in_units': 'celsius',
                    'out_units': 'celsius'
                },
                'precipitation': {
                    'in_varname': prec_var,
                    'in_units': 'mm/s',
                    'out_units': 'mm/day'
                }
            }
            
            geofabric_mapping = {
                'basinID': {'in_varname': 'hruId'},
                'nextDownID': {'in_varname': self.config['RIVER_NETWORK_SHP_DOWNSEGID']},
                'area': {
                    'in_varname': self.config.get('CATCHMENT_SHP_AREA'),
                    'in_units': 'm^2',
                    'out_units': 'm^2'
                                                    },
                'rivlen': {
                    'in_varname': self.config.get('RIVER_NETWORK_SHP_LENGTH'),
                    'in_units': 'm',
                    'out_units': 'm'
                }
            }

            # Write forcing data
            write_hype_forcing(
                str(self.forcing_basin_path),
                self.config.get('TIME_SHIFT', -6),
                forcing_units,
                geofabric_mapping,
                str(self.forcing_hype_path)
            )

            # Write GeoData and GeoClass files
            gistool_output = self.project_dir / 'gistool-outputs'
            subbasins_shapefile = self.project_dir / 'shapefiles' / 'extracted_subbasins.shp'
            rivers_shapefile = self.project_dir / 'shapefiles' / 'extracted_rivers.shp'

            write_hype_geo_files(
                str(gistool_output),
                str(subbasins_shapefile),
                str(rivers_shapefile),
                self.config.get('FRACTION_THRESHOLD', 0.01),
                geofabric_mapping,
                str(self.hype_setup_dir)
            )

            # Write parameter file
            write_hype_par_file(str(self.hype_setup_dir))

            # Write info and filedir files
            write_hype_info_filedir_files(
                str(self.hype_setup_dir),
                self.config.get('SPINUP_DAYS', 274)
            )

            self.logger.info("HYPE model files prepared successfully")

        except Exception as e:
            self.logger.error(f"Error preparing HYPE model files: {str(e)}")
            raise

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)


class HYPERunner:
    """
    Runner class for the HYPE model.
    Handles model execution and output processing.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'HYPE'
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_hype(self) -> Optional[Path]:
        """Run the HYPE model."""
        self.logger.info("Starting HYPE model run")
        
        try:
            # Get HYPE executable path
            hype_exe = self._get_hype_executable()
            
            # Get model setup directory
            setup_dir = self.project_dir / "settings" / "HYPE"
            
            # Construct command
            command = [str(hype_exe), str(setup_dir) + '/']
            
            # Create log directory
            log_dir = self.output_path / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / 'hype_run.log'
            
            # Run HYPE
            self.logger.info(f"Executing HYPE command: {' '.join(command)}")
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # Check if run was successful
            if result.returncode in [0, 84]:  # Code 84 is success for older versions
                self.logger.info("HYPE run completed successfully")
                return self.output_path
            else:
                self.logger.error(f"HYPE run failed with return code: {result.returncode}")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing HYPE: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during HYPE run: {str(e)}")
            raise

    def _get_hype_executable(self) -> Path:
        """Get path to HYPE executable."""
        hype_path = self.config.get('HYPE_INSTALL_PATH')
        if hype_path == 'default':
            hype_path = self.data_dir / 'installs' / 'hype' / 'bin'
        
        # Get executable name based on platform
        if sys.platform.startswith('win'):
            exe_name = 'HYPE.exe'
        else:
            exe_name = 'hype'
            
        exe_path = Path(hype_path) / exe_name
        
        if not exe_path.exists():
            raise FileNotFoundError(f"HYPE executable not found at: {exe_path}")
            
        return exe_path
        
class HYPEPostProcessor:
    """
    Post processor for the HYPE hydrological model
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.gr_setup_dir = self.project_dir / "settings" / "HYPE"
        
        # HYPE specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_gr_path = self.project_dir / 'forcing' / 'HYPE_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
