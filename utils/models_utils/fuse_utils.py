import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime

class FUSEPreProcessor:
    """
    Preprocessor for the FUSE (Framework for Understanding Structural Errors) model.
    Handles data preparation, PET calculation, and file setup for FUSE model runs.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        fuse_setup_dir (Path): Directory for FUSE setup files
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.fuse_setup_dir = self.project_dir / "settings" / "FUSE"
        
        # FUSE-specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        
    def run_preprocessing(self):
        """Run the complete FUSE preprocessing workflow."""
        self.logger.info("Starting FUSE preprocessing")
        try:
            self.create_directories()
            self.copy_base_settings()
            self.prepare_forcing_data()
            self.create_structure_file()
            self.create_parameter_file()
            self.logger.info("FUSE preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during FUSE preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for FUSE setup."""
        dirs_to_create = [
            self.fuse_setup_dir,
            self.forcing_fuse_path,
            self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def calculate_pet_oudin(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate potential evapotranspiration using Oudin's formula.
        
        Args:
            temp_data (xr.DataArray): Temperature data in Kelvin
            lat (float): Latitude of the catchment centroid
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        self.logger.info("Calculating PET using Oudin's formula")
        
        # Convert temperature to Celsius
        temp_C = temp_data - 273.15
        
        # Get dates for solar radiation calculation
        dates = pd.DatetimeIndex(temp_data.time.values)
        
        # Calculate day of year
        doy = dates.dayofyear
        
        # Calculate solar declination
        solar_decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
        
        # Convert latitude to radians
        lat_rad = np.deg2rad(lat)
        
        # Calculate sunset hour angle
        sunset_angle = np.arccos(-np.tan(lat_rad) * np.tan(solar_decl))
        
        # Calculate extraterrestrial radiation (Ra)
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
        Ra = (24 * 60 / np.pi) * 0.082 * dr * (
            sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
            np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)
        )
        
        # Calculate PET using Oudin's formula
        # PET = Ra * (T + 5) / 100 if T + 5 > 0, else 0
        pet = xr.where(temp_C + 5 > 0,
                      Ra * (temp_C + 5) / 100,
                      0)
        
        # Convert to proper units (mm/day) and add metadata
        pet = pet.assign_attrs({
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux'
        })
        
        return pet

    def prepare_forcing_data(self):
        """Prepare forcing data for FUSE model."""
        self.logger.info("Preparing forcing data for FUSE")
        
        # Read basin-averaged forcing data
        forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
        if not forcing_files:
            raise FileNotFoundError("No forcing files found in basin-averaged data directory")
        
        # Open and concatenate all forcing files
        ds = xr.open_mfdataset(forcing_files)
        
        # Get catchment centroid latitude for PET calculation
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lat = catchment[self.config.get('CATCHMENT_SHP_LAT')].mean()
        
        # Calculate PET
        pet = self.calculate_pet_oudin(ds['airtemp'], mean_lat)
        
        # Prepare FUSE forcing data
        fuse_forcing = xr.Dataset({
            'P': ds['pptrate'] * 86400,  # Convert from mm/s to mm/day
            'T': ds['airtemp'] - 273.15,  # Convert from K to °C
            'PET': pet
        })
        
        # Add metadata
        fuse_forcing.P.attrs = {'units': 'mm/day', 'long_name': 'Precipitation'}
        fuse_forcing.T.attrs = {'units': 'degC', 'long_name': 'Air temperature'}
        
        # Save forcing data
        output_file = self.forcing_fuse_path / f"{self.domain_name}_FUSE_forcing.nc"
        fuse_forcing.to_netcdf(output_file)
        self.logger.info(f"Forcing data prepared and saved to: {output_file}")

    def create_structure_file(self):
        """Create FUSE structure file based on configuration."""
        self.logger.info("Creating FUSE structure file")
        structure_config = {
            'SACRAMENTO': {
                'upper_tension': 1,
                'upper_free': 1,
                'lower_tension': 2,
                'lower_free': 2,
                'routing': 'gamma'
            },
            'TOPMODEL': {
                'upper_tension': 1,
                'upper_free': 0,
                'lower_tension': 0,
                'lower_free': 1,
                'routing': 'topmodel'
            },
            'PRMS': {
                'upper_tension': 1,
                'upper_free': 1,
                'lower_tension': 1,
                'lower_free': 1,
                'routing': 'prms'
            }
        }
        
        model_structure = self.config.get('FUSE_MODEL_STRUCTURE', 'SACRAMENTO')
        if model_structure not in structure_config:
            raise ValueError(f"Unsupported FUSE model structure: {model_structure}")
        
        structure = structure_config[model_structure]
        structure_file = self.fuse_setup_dir / 'structure.txt'
        
        with open(structure_file, 'w') as f:
            f.write(f"! FUSE model structure configuration for {model_structure}\n")
            for key, value in structure.items():
                f.write(f"{key:<20} {value}\n")
        
        self.logger.info(f"Structure file created: {structure_file}")

    def create_parameter_file(self):
        """Create FUSE parameter file."""
        self.logger.info("Creating FUSE parameter file")
        
        # Define default parameter ranges
        param_ranges = {
            'MAXWATER_1': (1.0, 1000.0),   # Maximum storage in upper soil layer [mm]
            'MAXWATER_2': (1.0, 1000.0),   # Maximum storage in lower soil layer [mm]
            'FRACTEN_1': (0.05, 0.95),     # Fraction of tension storage in upper layer [-]
            'FRACTEN_2': (0.05, 0.95),     # Fraction of tension storage in lower layer [-]
            'PERCFRAC': (0.05, 0.95),      # Percolation fraction [-]
            'FPRIMQB': (0.001, 0.1),       # Baseflow scaling parameter [-]
            'QB_POWR': (1.0, 10.0),        # Baseflow exponent [-]
            'K_QUICK': (0.001, 0.5),       # Quick flow routing parameter [1/day]
            'K_SLOW': (0.0001, 0.1),       # Slow flow routing parameter [1/day]
            'MAXBASE': (0.5, 10.0),        # Maximum baseflow rate [mm/day]
        }
        
        param_file = self.fuse_setup_dir / 'params.txt'
        with open(param_file, 'w') as f:
            f.write("! FUSE model parameters\n")
            f.write("! param_name    default   minimum   maximum\n")
            f.write("! ==================================\n")
            
            for param, (min_val, max_val) in param_ranges.items():
                default = (min_val + max_val) / 2  # Use middle of range as default
                f.write(f"{param:<12} {default:8.3f} {min_val:8.3f} {max_val:8.3f}\n")
        
        self.logger.info(f"Parameter file created: {param_file}")

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)

class FUSERunner:
    """
    Runner class for the FUSE (Framework for Understanding Structural Errors) model.
    Handles model execution, output processing, and file management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
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
        
        # FUSE-specific paths
        self.fuse_path = self._get_install_path()
        self.fuse_setup_dir = self.project_dir / "settings" / "FUSE"
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'
        self.output_path = self._get_output_path()

    def run_fuse(self) -> Optional[Path]:
        """
        Run the FUSE model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info("Starting FUSE model run")
        
        try:
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare run environment
            self._prepare_run_environment()
            
            # Execute FUSE
            success = self._execute_fuse()
            
            if success:
                # Process outputs
                self._process_outputs()
                self.logger.info("FUSE run completed successfully")
                return self.output_path
            else:
                self.logger.error("FUSE run failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during FUSE run: {str(e)}")
            raise

    def _get_install_path(self) -> Path:
        """Get the FUSE installation path."""
        fuse_path = self.config.get('FUSE_INSTALL_PATH')
        if fuse_path == 'default':
            return self.data_dir / 'installs' / 'fuse' / 'bin'
        return Path(fuse_path)

    def _get_output_path(self) -> Path:
        """Get the path for FUSE outputs."""
        if self.config.get('EXPERIMENT_OUTPUT_FUSE') == 'default':
            return self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        return Path(self.config.get('EXPERIMENT_OUTPUT_FUSE'))

    def _prepare_run_environment(self):
        """Prepare the environment for FUSE execution."""
        self.logger.info("Preparing FUSE run environment")
        
        # Create run directory structure
        run_dirs = {
            'input': self.output_path / 'input',
            'output': self.output_path / 'output',
            'settings': self.output_path / 'settings'
        }
        
        for dir_path in run_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files to run directory
        file_mappings = {
            self.forcing_fuse_path / f"{self.domain_name}_FUSE_forcing.nc": 
                run_dirs['input'] / 'forcing.nc',
            self.fuse_setup_dir / 'structure.txt': 
                run_dirs['settings'] / 'structure.txt',
            self.fuse_setup_dir / 'params.txt': 
                run_dirs['settings'] / 'params.txt'
        }
        
        for src, dst in file_mappings.items():
            shutil.copy2(src, dst)
            self.logger.info(f"Copied {src} to {run_dirs['settings']}")
        
        # Create FUSE control file
        self._create_control_file(run_dirs['settings'] / 'control.txt')

    def _create_control_file(self, control_file_path: Path):
        """
        Create FUSE control file with run settings.
        
        Args:
            control_file_path (Path): Path to write the control file
        """
        self.logger.info("Creating FUSE control file")
        
        # Get time period from config
        start_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_START'), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_END'), '%Y-%m-%d %H:%M')
        
        control_settings = {
            'START_TIME': start_time.strftime('%Y-%m-%d'),
            'END_TIME': end_time.strftime('%Y-%m-%d'),
            'TIME_STEP': '1',  # Daily timestep
            'OUTPUT_TIMESTEP': self.config.get('FUSE_OUTPUT_TIMESTEP', '1'),
            'SPATIAL_MODE': 'lumped' if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped' else 'distributed',
            'SNOW_MODULE': self.config.get('FUSE_SNOW_MODULE', 'temperature_index'),
            'OPTIMIZATION_METHOD': self.config.get('FUSE_OPTIMIZATION_METHOD', 'none'),
            'WARMUP_PERIOD': self.config.get('FUSE_WARMUP_DAYS', '365')
        }
        
        with open(control_file_path, 'w') as f:
            f.write("! FUSE Control File\n")
            f.write("! Generated by CONFLUENCE\n")
            f.write("! ======================\n\n")
            
            for key, value in control_settings.items():
                f.write(f"{key:<20} {value}\n")

    def _execute_fuse(self) -> bool:
        """
        Execute the FUSE model.
        
        Returns:
            bool: True if execution was successful, False otherwise
        """
        self.logger.info("Executing FUSE model")
        
        # Construct command
        fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
        control_file = self.output_path / 'settings' / 'control.txt'
        
        command = [
            str(fuse_exe),
            '-c', str(control_file),
            '-o', str(self.output_path / 'output')
        ]
        
        # Create log directory
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'fuse_run.log'
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            self.logger.info(f"FUSE execution completed with return code: {result.returncode}")
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed with error: {str(e)}")
            return False

    def _process_outputs(self):
        """Process and organize FUSE output files."""
        self.logger.info("Processing FUSE outputs")
        
        output_dir = self.output_path / 'output'
        
        # Read and process streamflow output
        q_file = output_dir / 'streamflow.nc'
        if q_file.exists():
            with xr.open_dataset(q_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.config.get('EXPERIMENT_ID')
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save processed output
                processed_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_streamflow.nc"
                ds.to_netcdf(processed_file)
                self.logger.info(f"Processed streamflow output saved to: {processed_file}")
        
        # Process state variables if they exist
        state_file = output_dir / 'states.nc'
        if state_file.exists():
            with xr.open_dataset(state_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.config.get('EXPERIMENT_ID')
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save processed output
                processed_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_states.nc"
                ds.to_netcdf(processed_file)
                self.logger.info(f"Processed state variables saved to: {processed_file}")

    def backup_run_files(self):
        """Backup important run files for reproducibility."""
        self.logger.info("Backing up run files")
        
        backup_dir = self.output_path / 'run_settings'
        backup_dir.mkdir(exist_ok=True)
        
        files_to_backup = [
            self.output_path / 'settings' / 'control.txt',
            self.output_path / 'settings' / 'structure.txt',
            self.output_path / 'settings' / 'params.txt'
        ]
        
        for file in files_to_backup:
            if file.exists():
                shutil.copy2(file, backup_dir / file.name)
                self.logger.info(f"Backed up {file.name}")