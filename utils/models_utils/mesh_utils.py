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
# import necessary libraries
import meshflow # type: ignore # version v0.1.0-dev1

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
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        
        # MESH-specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_mesh_path = self.project_dir / 'forcing' / 'MESH_input'

        self.catchment_path = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        self.catchment_name = self.config.get('RIVER_BASINS_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"

        self.rivers_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        self.rivers_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        if self.rivers_name == 'default':
            self.rivers_name = f"{self.domain_name}_riverNetwork_delineate.shp"

    def run_preprocessing(self):
        """Run the complete GR preprocessing workflow."""
        self.logger.info("Starting GR preprocessing")
        try:
            self.create_directories()
            config = self.create_json()
            self.prepare_forcing_data(config)


            self.logger.info("MESH preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during MESH preprocessing: {str(e)}")
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


    def create_json(self):
        # main work path - modify
        work_path = '/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow/'

        # using meshflow==v0.1.0-dev1
        # modify the following to match your settings
        config = {
            'riv': os.path.join(str(self.rivers_path / self.rivers_name)),
            'cat': os.path.join(str(self.catchment_path / self.catchment_name)),
            'landcover': os.path.join(f"{self.project_dir / 'attributes' / 'gistool-outputs' / 'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv'}"),
            'forcing_files': os.path.join(f"{self.project_dir / 'forcing' / 'easymore-outputs'}"),
            'forcing_vars': [ # MESH usuall needs 7 variables, list them below
            "RDRS_v2.1_P_P0_SFC",
            "RDRS_v2.1_P_HU_09944",
            "RDRS_v2.1_P_TT_09944",
            "RDRS_v2.1_P_UVC_09944",
            "RDRS_v2.1_A_PR0_SFC",
            "RDRS_v2.1_P_FB_SFC",
            "RDRS_v2.1_P_FI_SFC",
        ],
        'forcing_units': { # Enter original units for each variable listed under 'forcing_vars'
            "RDRS_v2.1_P_P0_SFC": 'millibar',
            "RDRS_v2.1_P_HU_09944": 'kg/kg',
            "RDRS_v2.1_P_TT_09944": 'celsius',
            "RDRS_v2.1_P_UVC_09944": 'knot',
            "RDRS_v2.1_A_PR0_SFC": 'm/hr',
            "RDRS_v2.1_P_FB_SFC": 'W/m^2',
            "RDRS_v2.1_P_FI_SFC": 'W/m^2',
        },
        'forcing_to_units': { # And here, the units that MESH needs to read
            "RDRS_v2.1_P_P0_SFC": 'pascal',
            "RDRS_v2.1_P_HU_09944": 'kg/kg',
            "RDRS_v2.1_P_TT_09944": 'kelvin',
            "RDRS_v2.1_P_UVC_09944": 'm/s', 
            "RDRS_v2.1_A_PR0_SFC": 'mm/s',
            "RDRS_v2.1_P_FB_SFC": 'W/m^2',
            "RDRS_v2.1_P_FI_SFC": 'W/m^2',
        },
        'main_id': 'GRU_ID', # what is the main ID of each river segment? Column name in the `cat` object
        'ds_main_id': 'DSLINKNO', # what is the downstream segment ID for each river segment? ditto.
        'landcover_classes': { # these are the classes defined for NALCMS-Landsat 2015 dataset. Is this accurate?
            1: 'Temperate or sub-polar needleleaf forest',
            2: 'Sub-polar taiga needleleaf forest',
            3: 'Tropical or sub-tropical broadleaf evergreen forest',
            4: 'Tropical or sub-tropical broadleaf deciduous forest',
            5: 'Temperate or sub-polar broadleaf deciduous forest',
            6: 'Mixed forest',
            7: 'Tropical or sub-tropical shrubland',
            8: 'Temperate or sub-polar shrubland',
            9: 'Tropical or sub-tropical grassland',
            10: 'Temperate or sub-polar grassland',
            11: 'Sub-polar or polar shrubland-lichen-moss',
            12: 'Sub-polar or polar grassland-lichen-moss',
            13: 'Sub-polar or polar barren-lichen-moss',
            14: 'Wetland',
            15: 'Cropland',
            16: 'Barren lands',
            17: 'Urban',
            18: 'Water',
            19: 'Snow and Ice',	
        },
        'ddb_vars': { # drainage database variables that MESH needs
            # FIXME: in later versions, the positions of keys and values below will be switched
            'Slope': 'ChnlSlope',
            'Length':'ChnlLength',
            'Rank': 'Rank', # Rank variable - for WATROUTE routing
            'Next': 'Next', # Next variable - for WATROUTE routing
            'landcover': 'GRU', # GRU fractions variable
            'GRU_area':'GridArea', # Sub-basin area variable
            'landcover_names': 'LandUse', # LandUse names
        },
        'ddb_units': { # units of variables in the drainage database
            'ChnlSlope': 'm/m',
            'ChnlLength': 'm',
            'Rank': 'dimensionless',
            'Next': 'dimensionless',
            'GRU': 'dimensionless',
            'GridArea': 'm^2',
            'LandUse': 'dimensionless',
        },
        'ddb_to_units': { # units of variables in the drainage database the MESH needs
            'ChnlSlope': 'm/m',
            'ChnlLength': 'm',
            'Rank': 'dimensionless',
            'Next': 'dimensionless',
            'GRU': 'dimensionless',
            'GridArea': 'm^2',
            'LandUse': 'dimensionless',
        },
        'ddb_min_values': { # minimum values in the drainage database
            'ChnlSlope': 1e-10, # in case there are 0s in the `rivers` Shapefile, we need minimums for certain variables
            'ChnlLength': 1e-3,
            'GridArea': 1e-3,
        },
        'gru_dim': 'NGRU', # change to `NGRU` for 'MESH>=r1860', keep for 'MESH<=1860', for example for r1813.
        'hru_dim': 'subbasin', # consistent in various versions, no need to change
        'outlet_value': 0, # modify depending on the outlet values specific in `ds_main_id` object
    }
        return config

    def prepare_forcing_data(self, config):

        exp = meshflow.MESHWorkflow(**config)
        exp.run()

        # create a directory for MESH setup
        self.forcing_mesh_path.mkdir(parents=True, exist_ok=True)

        # saving drainage database and forcing files
        exp.save(self.forcing_mesh_path)


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
        self.mesh_exe = self.config.get('MESH_EXE')
        self.mesh_install_path = Path(self.config['CONFLUENCE_DATA_DIR']) / 'installs' / 'MESH-DEV'

        # Model configuration


    def run_MESH(self) -> Optional[Path]:
        """
        Run the MESH model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwis
        """
        # Store current directory
        original_dir = os.getcwd()
        
            # Change to forcing directory for execution
        os.chdir(self.forcing_mesh_path)

        cmd = self._create_run_command()
        subprocess.run(cmd, check=True) 

        # Change back to original directory
        os.chdir(original_dir)
        shutil.rmtree(self.forcing_mesh_path / self.config.get('MESH_EXE', 'sa_mesh'))
        return
    
    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        mesh_exe = self.mesh_install_path / self.config.get('MESH_EXE', 'sa_mesh')
        # Copy mesh executable to forcing path
        mesh_exe_name = mesh_exe.name
        mesh_exe_dest = self.forcing_mesh_path / mesh_exe_name
        shutil.copy2(mesh_exe, mesh_exe_dest)
        
        
        cmd = [
            str(mesh_exe) 
        ]
        print(cmd)
        return cmd

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)
        
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

    def extract_streamflow(self):
        return

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)
