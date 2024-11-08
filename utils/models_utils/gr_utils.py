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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

class GRPreProcessor:
    """
    Preprocessor for the GR family of models (initially GR4J).
    Handles data preparation, PET calculation, snow module setup, and file organization.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for GR models
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        gr_setup_dir (Path): Directory for GR setup files
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.gr_setup_dir = self.project_dir / "settings" / "GR"
        
        # GR-specific paths
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_gr_path = self.project_dir / 'forcing' / 'GR_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')

    def run_preprocessing(self):
        """Run the complete GR preprocessing workflow."""
        self.logger.info("Starting GR preprocessing")
        try:
            self.create_directories()
            self.prepare_forcing_data()
            self.setup_cemaneige()
            self.create_parameter_file()
            self.create_spatial_weights()
            self.logger.info("GR preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during GR preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for GR model setup."""
        dirs_to_create = [
            self.gr_setup_dir,
            self.forcing_gr_path,
            self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "GR"
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
        """Prepare forcing data for GR model."""
        self.logger.info("Preparing forcing data for GR model")
        
        # Read basin-averaged forcing data
        forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
        if not forcing_files:
            raise FileNotFoundError("No forcing files found in basin-averaged data directory")
        
        # Open and concatenate all forcing files
        ds = xr.open_mfdataset(forcing_files)
        
        # Get catchment geometries
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        
        # Process each catchment/HRU
        for idx, row in catchment.iterrows():
            hru_id = row[self.config.get('CATCHMENT_SHP_HRUID')]
            lat = row[self.config.get('CATCHMENT_SHP_LAT')]
            
            # Calculate PET for this HRU
            pet = self.calculate_pet_oudin(ds['airtemp'].sel(hru=idx), lat)
            
            # Prepare GR forcing data
            gr_forcing = xr.Dataset({
                'P': ds['pptrate'].sel(hru=idx) * 86400,  # Convert from mm/s to mm/day
                'T': ds['airtemp'].sel(hru=idx) - 273.15,  # Convert from K to °C
                'PET': pet,
                'Q_obs': xr.full_like(pet, np.nan)  # Placeholder for observed discharge
            })
            
            # Add metadata
            gr_forcing.P.attrs = {'units': 'mm/day', 'long_name': 'Precipitation'}
            gr_forcing.T.attrs = {'units': 'degC', 'long_name': 'Air temperature'}
            gr_forcing.Q_obs.attrs = {'units': 'mm/day', 'long_name': 'Observed discharge'}
            
            # Save forcing data
            output_file = self.forcing_gr_path / f"{self.domain_name}_GR_forcing_hru{hru_id}.nc"
            gr_forcing.to_netcdf(output_file)
            
        self.logger.info("Forcing data preparation completed")

    def setup_cemaneige(self):
        """Setup CemaNeige snow module configuration."""
        self.logger.info("Setting up CemaNeige snow module")
        
        snow_config = {
            'use_cemaneige': self.config.get('GR_USE_CEMANEIGE', True),
            'elevation_bands': self.config.get('GR_SNOW_ELEV_BANDS', 5),
            'thermal_state_days': self.config.get('GR_SNOW_THERMAL_STATE_DAYS', 15),
            'cold_content_days': self.config.get('GR_SNOW_COLD_CONTENT_DAYS', 15)
        }
        
        # Create CemaNeige configuration file
        cemaneige_file = self.gr_setup_dir / 'cemaneige_config.txt'
        with open(cemaneige_file, 'w') as f:
            f.write("! CemaNeige Configuration\n")
            for key, value in snow_config.items():
                f.write(f"{key:<20} {value}\n")
        
        # Calculate elevation bands for each HRU if using CemaNeige
        if snow_config['use_cemaneige']:
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            elev_bands = self._calculate_elevation_bands(catchment, snow_config['elevation_bands'])
            
            # Save elevation bands
            elev_bands_file = self.gr_setup_dir / 'elevation_bands.txt'
            elev_bands.to_csv(elev_bands_file, index=False)
            
        self.logger.info("CemaNeige setup completed")

    def create_parameter_file(self):
        """Create GR4J parameter file with initial values and bounds."""
        self.logger.info("Creating GR parameter file")
        
        # Define GR4J parameters and their bounds
        gr4j_params = {
            'X1': {'default': 350.0, 'min': 100.0, 'max': 1200.0, 'description': 'Production store capacity (mm)'},
            'X2': {'default': 0.0, 'min': -5.0, 'max': 5.0, 'description': 'Groundwater exchange coefficient (mm/day)'},
            'X3': {'default': 90.0, 'min': 20.0, 'max': 300.0, 'description': 'Routing store capacity (mm)'},
            'X4': {'default': 1.7, 'min': 1.1, 'max': 2.9, 'description': 'Time base of unit hydrograph (days)'}
        }
        
        # Add CemaNeige parameters if enabled
        if self.config.get('GR_USE_CEMANEIGE', True):
            cemaneige_params = {
                'CTG': {'default': 0.0, 'min': -1.0, 'max': 1.0, 'description': 'Temperature correction factor (°C)'},
                'Kf': {'default': 2.0, 'min': 1.0, 'max': 4.0, 'description': 'Degree-day melt factor (mm/°C/day)'}
            }
            gr4j_params.update(cemaneige_params)
        
        # Save parameter file
        param_file = self.gr_setup_dir / 'parameters.txt'
        with open(param_file, 'w') as f:
            f.write("! GR4J Parameter File\n")
            f.write("! parameter   default    min       max       description\n")
            f.write("! ====================================================\n")
            
            for param, values in gr4j_params.items():
                f.write(f"{param:<12} {values['default']:<10.3f} {values['min']:<10.3f} "
                       f"{values['max']:<10.3f} ! {values['description']}\n")
        
        self.logger.info(f"Parameter file created: {param_file}")

    def create_spatial_weights(self):
        """Create spatial weights for semi-distributed setup."""
        if self.config.get('GR_SPATIAL_MODE', 'lumped') == 'semi-distributed':
            self.logger.info("Creating spatial weights for semi-distributed setup")
            
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            total_area = catchment.geometry.area.sum()
            
            # Calculate area weights
            weights = pd.DataFrame({
                'HRU_ID': catchment[self.config.get('CATCHMENT_SHP_HRUID')],
                'weight': catchment.geometry.area / total_area
            })
            
            # Save weights
            weights_file = self.gr_setup_dir / 'spatial_weights.txt'
            weights.to_csv(weights_file, index=False)
            
            self.logger.info("Spatial weights created")

    def _calculate_elevation_bands(self, catchment: gpd.GeoDataFrame, num_bands: int) -> pd.DataFrame:
        """
        Calculate elevation bands for each HRU for CemaNeige.
        
        Args:
            catchment (gpd.GeoDataFrame): Catchment geometries
            num_bands (int): Number of elevation bands
            
        Returns:
            pd.DataFrame: Elevation bands information
        """
        elevation_bands = []
        
        for idx, row in catchment.iterrows():
            hru_id = row[self.config.get('CATCHMENT_SHP_HRUID')]
            mean_elev = row['elev_mean']
            
            # Define elevation range (assuming ±500m from mean)
            elev_min = mean_elev - 500
            elev_max = mean_elev + 500
            
            # Calculate bands
            band_edges = np.linspace(elev_min, elev_max, num_bands + 1)
            band_means = (band_edges[1:] + band_edges[:-1]) / 2
            
            # Create bands data
            for band_idx, mean_elev in enumerate(band_means):
                elevation_bands.append({
                    'HRU_ID': hru_id,
                    'band_id': band_idx + 1,
                    'mean_elevation': mean_elev,
                    'lower_bound': band_edges[band_idx],
                    'upper_bound': band_edges[band_idx + 1]
                })
        
        return pd.DataFrame(elevation_bands)

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)


class GRRunner:
    """
    Runner class for the GR family of models (initially GR4J).
    Handles model execution, state management, and output processing.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for GR models
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
        
        # GR-specific paths
        self.gr_path = self._get_install_path()
        self.gr_setup_dir = self.project_dir / "settings" / "GR"
        self.forcing_gr_path = self.project_dir / 'forcing' / 'GR_input'
        self.output_path = self._get_output_path()
        
        # Model configuration
        self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')
        self.use_cemaneige = self.config.get('GR_USE_CEMANEIGE', True)
        self.use_mpi = self.config.get('GR_USE_MPI', True)
        self.num_processors = min(
            self.config.get('GR_NUM_PROCESSORS', multiprocessing.cpu_count()),
            multiprocessing.cpu_count()
        )

    def run_gr(self) -> Optional[Path]:
        """
        Run the GR model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info("Starting GR model run")
        
        try:
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare run environment
            self._prepare_run_environment()
            
            # Execute GR model
            if self.spatial_mode == 'lumped':
                success = self._execute_gr_lumped()
            else:  # semi-distributed
                success = self._execute_gr_distributed()
            
            if success:
                # Process outputs
                self._process_outputs()
                self._backup_run_files()
                self.logger.info("GR run completed successfully")
                return self.output_path
            else:
                self.logger.error("GR run failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during GR run: {str(e)}")
            raise

    def _prepare_run_environment(self):
        """Prepare the run environment for GR model execution."""
        self.logger.info("Preparing GR run environment")
        
        # Create run directory structure
        run_dirs = {
            'input': self.output_path / 'input',
            'output': self.output_path / 'output',
            'settings': self.output_path / 'settings',
            'states': self.output_path / 'states'
        }
        
        for dir_path in run_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy necessary files to run directory
        self._copy_setup_files(run_dirs)
        
        # Create control file
        self._create_control_file(run_dirs['settings'] / 'control.txt')

    def _copy_setup_files(self, run_dirs: Dict[str, Path]):
        """Copy setup files to run directory."""
        file_mappings = {
            self.gr_setup_dir / 'parameters.txt': 
                run_dirs['settings'] / 'parameters.txt'
        }
        
        # Add CemaNeige files if enabled
        if self.use_cemaneige:
            file_mappings.update({
                self.gr_setup_dir / 'cemaneige_config.txt': 
                    run_dirs['settings'] / 'cemaneige_config.txt',
                self.gr_setup_dir / 'elevation_bands.txt': 
                    run_dirs['settings'] / 'elevation_bands.txt'
            })
        
        # Add spatial weights file if in semi-distributed mode
        if self.spatial_mode == 'semi-distributed':
            file_mappings[self.gr_setup_dir / 'spatial_weights.txt'] = \
                run_dirs['settings'] / 'spatial_weights.txt'
        
        # Copy all forcing files
        for forcing_file in self.forcing_gr_path.glob('*.nc'):
            file_mappings[forcing_file] = run_dirs['input'] / forcing_file.name
        
        # Perform copy operations
        for src, dst in file_mappings.items():
            shutil.copy2(src, dst)
            self.logger.info(f"Copied {src.name} to {dst.parent}")

    def _create_control_file(self, control_file_path: Path):
        """Create control file for GR model run."""
        self.logger.info("Creating GR control file")
        
        # Get time period from config
        start_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_START'), 
                                     '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_END'), 
                                   '%Y-%m-%d %H:%M')
        
        control_settings = {
            'MODEL_VERSION': 'GR4J',
            'START_TIME': start_time.strftime('%Y-%m-%d'),
            'END_TIME': end_time.strftime('%Y-%m-%d'),
            'TIME_STEP': '1',  # Daily timestep
            'WARMUP_DAYS': self.config.get('GR_WARMUP_DAYS', '365'),
            'SPATIAL_MODE': self.spatial_mode,
            'USE_CEMANEIGE': str(self.use_cemaneige).lower(),
            'SAVE_STATES': self.config.get('GR_SAVE_STATES', 'true'),
            'STATE_SAVE_FREQUENCY': self.config.get('GR_STATE_SAVE_FREQ', '30'),
            'OUTPUT_VARIABLES': 'Q_sim,production_store,routing_store'
        }
        
        # Add snow variables if CemaNeige is enabled
        if self.use_cemaneige:
            control_settings['OUTPUT_VARIABLES'] += ',snow_store,thermal_state'
        
        with open(control_file_path, 'w') as f:
            f.write("! GR Control File\n")
            f.write("! Generated by CONFLUENCE\n")
            f.write("! ======================\n\n")
            
            for key, value in control_settings.items():
                f.write(f"{key:<20} {value}\n")

    def _execute_gr_lumped(self) -> bool:
        """Execute GR model in lumped mode."""
        self.logger.info("Executing GR model in lumped mode")
        
        gr_exe = self.gr_path / self.config.get('GR_EXE', 'gr4j.exe')
        control_file = self.output_path / 'settings' / 'control.txt'
        
        command = [
            str(gr_exe),
            '-c', str(control_file),
            '-o', str(self.output_path / 'output')
        ]
        
        return self._run_gr_command(command)

    def _execute_gr_distributed(self) -> bool:
        """Execute GR model in semi-distributed mode."""
        self.logger.info("Executing GR model in semi-distributed mode")
        
        if self.use_mpi and self.num_processors > 1:
            return self._execute_gr_parallel()
        else:
            return self._execute_gr_sequential()

    def _execute_gr_parallel(self) -> bool:
        """Execute GR model in parallel for semi-distributed mode."""
        self.logger.info(f"Running GR in parallel with {self.num_processors} processors")
        
        catchment_file = self.catchment_path / self.catchment_name
        catchment = gpd.read_file(catchment_file)
        hru_ids = catchment[self.config.get('CATCHMENT_SHP_HRUID')].tolist()
        
        # Split HRUs among processors
        chunks = np.array_split(hru_ids, self.num_processors)
        
        with ProcessPoolExecutor(max_workers=self.num_processors) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._run_gr_chunk, chunk)
                futures.append(future)
            
            success = all(future.result() for future in futures)
        
        if success:
            # Merge outputs from different processors
            self._merge_parallel_outputs()
            
        return success

    def _run_gr_chunk(self, hru_ids: List[int]) -> bool:
        """Run GR model for a chunk of HRUs."""
        for hru_id in hru_ids:
            gr_exe = self.gr_path / self.config.get('GR_EXE', 'gr4j.exe')
            control_file = self.output_path / 'settings' / 'control.txt'
            
            command = [
                str(gr_exe),
                '-c', str(control_file),
                '-h', str(hru_id),
                '-o', str(self.output_path / 'output' / f'hru_{hru_id}')
            ]
            
            if not self._run_gr_command(command):
                return False
        
        return True

    def _merge_parallel_outputs(self):
        """Merge outputs from parallel GR runs."""
        self.logger.info("Merging parallel GR outputs")
        
        output_vars = ['Q_sim', 'production_store', 'routing_store']
        if self.use_cemaneige:
            output_vars.extend(['snow_store', 'thermal_state'])
        
        # Initialize output dataset
        ds_dict = {}
        
        # Process each output variable
        for var in output_vars:
            data_list = []
            hru_ids = []
            
            # Collect data from each HRU
            for output_dir in (self.output_path / 'output').glob('hru_*'):
                hru_id = int(output_dir.name.split('_')[1])
                file_path = output_dir / f'{var}.nc'
                
                if file_path.exists():
                    with xr.open_dataset(file_path) as ds:
                        data_list.append(ds[var].values)
                        hru_ids.append(hru_id)
            
            # Create merged DataArray
            if data_list:
                data = np.stack(data_list, axis=1)
                ds_dict[var] = xr.DataArray(
                    data,
                    coords={
                        'time': data_list[0].time,
                        'hru': hru_ids
                    },
                    dims=['time', 'hru']
                )
        
        # Create merged Dataset and save
        if ds_dict:
            ds = xr.Dataset(ds_dict)
            output_file = self.output_path / 'output' / 'gr_output.nc'
            ds.to_netcdf(output_file)
            
            # Clean up individual HRU outputs
            for output_dir in (self.output_path / 'output').glob('hru_*'):
                shutil.rmtree(output_dir)

    def _run_gr_command(self, command: List[str]) -> bool:
        """Run GR command with logging."""
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'gr_run.log'
        
        try:
            with open(log_file, 'a') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GR execution failed with error: {str(e)}")
            return False

    def _process_outputs(self):
        """Process and organize GR model outputs."""
        self.logger.info("Processing GR outputs")
        
        # Process main output file
        output_file = self.output_path / 'output' / 'gr_output.nc'
        if output_file.exists():
            with xr.open_dataset(output_file) as ds:
                # Add metadata
                ds.attrs.update({
                    'model': 'GR4J',
                    'domain': self.domain_name,
                    'experiment_id': self.config.get('EXPERIMENT_ID'),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'spatial_mode': self.spatial_mode,
                    'use_cemaneige': str(self.use_cemaneige)
                })
                
                # Save processed output
                processed_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_gr_output.nc"
                ds.to_netcdf(processed_file)
                self.logger.info(f"Processed output saved to: {processed_file}")
        
        # Process state files if they exist
        self._process_state_files()

    def _process_state_files(self):
        """Process GR model state files."""
        state_dir = self.output_path / 'states'
        if not state_dir.exists():
            return
        
        state_files = list(state_dir.glob('gr_state_*.nc'))
        if not state_files:
            return
        
        # Combine state files into a single dataset
        state_data = []
        for state_file in sorted(state_files):
            with xr.open_dataset(state_file) as ds:
                state_data.append(ds)
        
        combined_states = xr.concat(state_data, dim='time')
        
        # Save combined states
        output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_gr_states.nc"
        combined_states.to_netcdf(output_file)
        self.logger.info(f"Combined state file saved to: {output_file}")

    def _backup_run_files(self):
        """Backup important run files."""