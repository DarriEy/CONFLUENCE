import os
import sys
import time
import subprocess
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
from scipy import ndimage
import csv
import itertools
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
from typing import Dict, List, Tuple, Any


sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from utils.data.variable_utils import VariableHandler # type: ignore

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
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        
    def run_preprocessing(self):
        """Run the complete FUSE preprocessing workflow."""
        self.logger.info("Starting FUSE preprocessing")
        try:
            self.create_directories()
            self.copy_base_settings()
            self.prepare_forcing_data()
            self.create_elevation_bands()  
            self.create_filemanager()
            self.logger.info("FUSE preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during FUSE preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for FUSE setup."""
        dirs_to_create = [
            self.fuse_setup_dir,
            self.forcing_fuse_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def copy_base_settings(self):
        """
        Copy FUSE base settings from the source directory to the project's settings directory.
        Updates the model ID in the decisions file name to match the experiment ID.

        This method performs the following steps:
        1. Determines the source directory for base settings
        2. Determines the destination directory for settings
        3. Creates the destination directory if it doesn't exist
        4. Copies all files from the source directory to the destination directory
        5. Renames the decisions file with the appropriate experiment ID

        Raises:
            FileNotFoundError: If the source directory or any source file is not found.
            PermissionError: If there are permission issues when creating directories or copying files.
        """
        self.logger.info("Copying FUSE base settings")
        
        base_settings_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_base_settings' / 'FUSE'
        settings_path = self._get_default_path('SETTINGS_FUSE_PATH', 'settings/FUSE')
        
        try:
            settings_path.mkdir(parents=True, exist_ok=True)
            
            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file
                
                # Handle the decisions file specially
                if 'fuse_zDecisions_' in file:
                    # Create new filename with experiment ID
                    new_filename = file.replace('902', self.config.get('EXPERIMENT_ID'))
                    dest_file = settings_path / new_filename
                    self.logger.debug(f"Renaming decisions file from {file} to {new_filename}")
                else:
                    dest_file = settings_path / file
                
                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")
            
            self.logger.info(f"FUSE base settings copied to {settings_path}")
            
        except FileNotFoundError as e:
            self.logger.error(f"Source file or directory not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission error when copying files: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error copying base settings: {e}")
            raise

    def calculate_pet_hargreaves(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hargreaves method.
        Requires mean temperature only (simplified version without min/max temps).
        
        Args:
            temp_data (xr.DataArray): Temperature data (auto-detects K or C)
            lat (float): Latitude of the catchment centroid
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        self.logger.info("Calculating PET using Hargreaves method (simplified)")
        
        # Load data if needed
        if hasattr(temp_data.data, 'compute'):
            temp_data = temp_data.load()
        
        # Check and convert temperature
        temp_mean = float(temp_data.mean())
        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}")
        
        # Auto-detect units
        if temp_mean > 100:  # Kelvin
            self.logger.info("Temperature in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Celsius
            self.logger.info("Temperature in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature has unexpected range: mean={temp_mean:.2f}")
        
        # Verify reasonable range
        temp_mean_C = float(temp_C.mean())
        self.logger.debug(f"Temperature (°C): Mean={temp_mean_C:.2f}")
        
        if temp_mean_C < -60 or temp_mean_C > 60:
            raise ValueError(f"Temperature unrealistic: {temp_mean_C:.2f}°C")
        
        # Get time information
        time_values = pd.to_datetime(temp_data.time.values)
        doy = xr.DataArray(time_values.dayofyear, coords={'time': temp_data.time}, dims=['time'])
        
        # Calculate extraterrestrial radiation (Ra)
        lat_rad = np.deg2rad(lat)
        
        # Solar declination
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)
        
        # Sunset hour angle
        cos_arg = -np.tan(lat_rad) * np.tan(solar_decl)
        cos_arg = cos_arg.clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        
        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)
        
        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr * 
            (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
            np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))
        
        self.logger.debug(f"Solar radiation Ra: Mean={float(Ra.mean()):.2f} MJ/m²/day")
        
        # Broadcast Ra if needed for multi-HRU
        if 'hru' in temp_C.dims:
            Ra = Ra.broadcast_like(temp_C)
        
        # Hargreaves formula (simplified without Tmin/Tmax)
        # PET = 0.0023 * Ra * (Tmean + 17.8) * TD^0.5
        # Without Tmin/Tmax, we use a typical diurnal range of 10°C
        # This makes it: PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(10)
        # Converting Ra from MJ/m²/day to equivalent: multiply by 0.408 to get mm/day
        
        TD = 10.0  # Assumed temperature range (°C) when min/max not available
        pet = 0.0023 * (Ra * 0.408) * (temp_C + 17.8) * np.sqrt(TD)
        
        # Ensure non-negative
        pet = xr.where(pet > 0, pet, 0.0)
        
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Hargreaves (simplified)',
            'latitude': lat,
            'note': 'Simplified version using assumed diurnal temperature range of 10°C'
        }
        
        pet_mean = float(pet.mean())
        self.logger.info(f"PET calculation complete: Mean={pet_mean:.3f} mm/day, "
                        f"Min={float(pet.min()):.3f} mm/day, Max={float(pet.max()):.3f} mm/day")
        
        return pet


    def calculate_pet_oudin(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate potential evapotranspiration using Oudin's formula.
        Handles temperature in either Kelvin or Celsius.
        
        Args:
            temp_data (xr.DataArray): Temperature data (auto-detects K or C)
            lat (float): Latitude of the catchment centroid
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        self.logger.info("Calculating PET using Oudin's formula")
        
        # Check if data needs loading (if using dask)
        if hasattr(temp_data.data, 'compute'):
            self.logger.debug("Loading temperature data from dask array...")
            temp_data = temp_data.load()
        
        # Check input temperature and determine if it's Kelvin or Celsius
        temp_mean = float(temp_data.mean())
        temp_min = float(temp_data.min())
        temp_max = float(temp_data.max())
        
        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}, Min={temp_min:.2f}, Max={temp_max:.2f}")
        
        # Auto-detect units and convert to Celsius if needed
        if temp_mean > 100:  # Likely Kelvin (>100K = -173°C, unrealistic for Earth)
            self.logger.info("Temperature appears to be in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Likely Celsius (reasonable range)
            self.logger.info("Temperature appears to be in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")
        
        # Verify final temperature is reasonable
        temp_mean_C = float(temp_C.mean())
        self.logger.debug(f"Temperature in Celsius: Mean={temp_mean_C:.2f}°C, "
                        f"Min={float(temp_C.min()):.2f}°C, Max={float(temp_C.max()):.2f}°C")
        
        if temp_mean_C < -60 or temp_mean_C > 60:
            self.logger.error(f"Temperature is unrealistic: {temp_mean_C:.2f}°C")
            raise ValueError(f"Unrealistic temperature after conversion: {temp_mean_C:.2f}°C")
        
        # Get time information
        time_values = pd.to_datetime(temp_data.time.values)
        doy = xr.DataArray(time_values.dayofyear, coords={'time': temp_data.time}, dims=['time'])
        
        # Solar calculations (vectorized)
        lat_rad = np.deg2rad(lat)
        
        # Solar declination (radians)
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)
        
        # Sunset hour angle with numerical stability
        cos_arg = -np.tan(lat_rad) * np.tan(solar_decl)
        cos_arg = cos_arg.clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        
        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)
        
        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr * 
            (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
            np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))
        
        self.logger.debug(f"Solar radiation Ra: Mean={float(Ra.mean()):.2f} MJ/m²/day")
        
        # Broadcast Ra if needed for multi-HRU
        if 'hru' in temp_C.dims:
            Ra = Ra.broadcast_like(temp_C)
        
        # Oudin's formula: PET = Ra * (T + 5) / 100 when T + 5 > 0
        pet = xr.where(temp_C + 5.0 > 0.0, Ra * (temp_C + 5.0) / 100.0, 0.0)
        
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Oudin et al. (2005)',
            'latitude': lat
        }
        
        pet_mean = float(pet.mean())
        self.logger.info(f"PET calculation complete: Mean={pet_mean:.3f} mm/day, "
                        f"Min={float(pet.min()):.3f} mm/day, Max={float(pet.max()):.3f} mm/day")
        
        if pet_mean < 0.1:
            n_valid = int((temp_C + 5.0 > 0.0).sum())
            self.logger.warning(f"Very low PET! Days with T>-5°C: {n_valid}/{len(temp_C.time)}")
        
        return pet


    def calculate_pet_hamon(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hamon's method.
        Handles temperature in either Kelvin or Celsius.
        
        Args:
            temp_data (xr.DataArray): Temperature data (auto-detects K or C)
            lat (float): Latitude of the catchment centroid
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        self.logger.info("Calculating PET using Hamon's method")
        
        # Load data if needed
        if hasattr(temp_data.data, 'compute'):
            temp_data = temp_data.load()
        
        # Check and convert temperature
        temp_mean = float(temp_data.mean())
        
        self.logger.debug(f"Input temperature: Mean={temp_mean:.2f}")
        
        # Auto-detect units
        if temp_mean > 100:  # Kelvin
            self.logger.info("Temperature in Kelvin, converting to Celsius")
            temp_C = temp_data - 273.15
        elif -100 < temp_mean < 60:  # Celsius
            self.logger.info("Temperature in Celsius, using as-is")
            temp_C = temp_data
        else:
            self.logger.error(f"Cannot determine temperature units. Mean={temp_mean:.2f}")
            raise ValueError(f"Temperature has unexpected range: mean={temp_mean:.2f}")
        
        # Get values for computation
        temp_C_vals = temp_C.values
        
        # Verify reasonable range
        temp_mean_C = np.nanmean(temp_C_vals)
        self.logger.debug(f"Temperature (°C): Mean={temp_mean_C:.2f}, "
                        f"Min={np.nanmin(temp_C_vals):.2f}, Max={np.nanmax(temp_C_vals):.2f}")
        
        if temp_mean_C < -60 or temp_mean_C > 60:
            raise ValueError(f"Temperature unrealistic: {temp_mean_C:.2f}°C")
        
        # Day of year
        dates = pd.to_datetime(temp_data.time.values)
        doy = dates.dayofyear.values
        
        # Solar calculations
        lat_rad = np.deg2rad(lat)
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)
        cos_arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        daylight_hours = 24.0 * sunset_angle / np.pi
        
        # Saturated vapor pressure (kPa)
        e_sat = 0.6108 * np.exp(17.27 * temp_C_vals / (temp_C_vals + 237.3))
        
        # Hamon PET (mm/day)
        if len(temp_C_vals.shape) > 1:
            daylight_hours = daylight_hours.reshape(-1, 1)
        
        pet_values = 0.1651 * daylight_hours * e_sat * 2.54
        pet_values = np.maximum(pet_values, 0.0)
        
        # Create DataArray
        if 'hru' in temp_data.dims:
            pet = xr.DataArray(pet_values, coords=temp_data.coords, dims=temp_data.dims)
        else:
            pet = xr.DataArray(pet_values, coords={'time': temp_data.time}, dims=['time'])
        
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'method': 'Hamon',
            'latitude': lat
        }
        
        self.logger.info(f"PET calculation complete: Mean={np.nanmean(pet_values):.3f} mm/day, "
                        f"Max={np.nanmax(pet_values):.3f} mm/day")
        
        return pet

    def generate_synthetic_hydrograph(self, ds, area_km2, mean_temp_threshold=0.0):
        """
        Generate a realistic synthetic hydrograph for snow optimization cases.
        
        Args:
            ds: xarray dataset with precipitation and temperature
            area_km2: catchment area in km2
            mean_temp_threshold: temperature threshold for snow/rain (°C)
        
        Returns:
            np.array: synthetic streamflow in mm/day
        """
        self.logger.info("Generating synthetic hydrograph for snow optimization")
        
        # Get precipitation and temperature data
        precip = ds['pr'].values  # mm/day
        temp = ds['temp'].values - 273.15  # Convert K to °C
        
        # Simple rainfall-runoff model parameters
        # These create a realistic but generic hydrograph
        runoff_coeff_rain = 0.3  # 30% of rain becomes runoff
        runoff_coeff_snow = 0.1  # 10% of snow becomes immediate runoff
        baseflow_recession = 0.95  # Daily baseflow recession coefficient
        
        # Initialize arrays
        n_days = len(precip)
        runoff = np.zeros(n_days)
        baseflow = np.zeros(n_days)
        snowpack = np.zeros(n_days)
        
        # Simple degree-day snowmelt parameters
        melt_factor = 3.0  # mm/°C/day
        
        # Initial baseflow (small constant)
        baseflow[0] = 0.5  # mm/day
        
        for i in range(n_days):
            # Determine if precipitation is rain or snow
            if temp[i] > mean_temp_threshold:
                # Rain
                rain = precip[i]
                snow = 0.0
            else:
                # Snow
                rain = 0.0
                snow = precip[i]
            
            # Snow accumulation and melt
            if i > 0:
                snowpack[i] = snowpack[i-1] + snow
            else:
                snowpack[i] = snow
                
            # Snowmelt (only if temperature > 0°C)
            if temp[i] > 0.0 and snowpack[i] > 0.0:
                melt = min(snowpack[i], melt_factor * temp[i])
                snowpack[i] -= melt
                rain += melt  # Add melt to effective rainfall
            
            # Calculate surface runoff
            surface_runoff = rain * runoff_coeff_rain + snow * runoff_coeff_snow
            
            # Update baseflow (simple recession + recharge)
            if i > 0:
                baseflow[i] = baseflow[i-1] * baseflow_recession + surface_runoff * 0.1
            else:
                baseflow[i] = surface_runoff * 0.1
            
            # Total runoff
            runoff[i] = surface_runoff + baseflow[i]
        
        # Add some realistic variability and ensure non-negative
        # Add small random component (±10% of mean)
        mean_runoff = np.mean(runoff)
        noise = np.random.normal(0, mean_runoff * 0.05, n_days)
        runoff = np.maximum(runoff + noise, 0.01)  # Ensure minimum flow
        
        # Apply a simple routing delay (moving average)
        from scipy import ndimage
        runoff_routed = ndimage.uniform_filter1d(runoff, size=3, mode='reflect')
        
        self.logger.info(f"Generated synthetic hydrograph: mean={np.mean(runoff_routed):.2f} mm/day, "
                        f"max={np.max(runoff_routed):.2f} mm/day")
        
        return runoff_routed

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped, semi-distributed, and distributed modes.
        """
        try:
            # Get spatial mode configuration
            spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
            
            self.logger.info(f"Preparing FUSE forcing data in {spatial_mode} mode")
            
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")
            
            variable_handler = VariableHandler(config=self.config, logger=self.logger, 
                                            dataset=self.config['FORCING_DATASET'], model='FUSE')
            ds = xr.open_mfdataset(forcing_files)
            ds = variable_handler.process_forcing_data(ds)
            
            # Spatial organization based on mode BEFORE resampling
            if spatial_mode == 'lumped':
                ds = self._prepare_lumped_forcing(ds)
            elif spatial_mode == 'semi_distributed':
                ds = self._prepare_semi_distributed_forcing(ds, subcatchment_dim)
            elif spatial_mode == 'distributed':
                ds = self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown FUSE spatial mode: {spatial_mode}")
            
            # Convert to daily resolution AFTER spatial organization
            self.logger.info("Resampling data to daily resolution")
            ds = ds.resample(time='D').mean()
            
            # Process temperature and precipitation
            try:
                ds['temp'] = ds['airtemp']
                ds['pr'] = ds['pptrate']
            except:
                pass
            
            # Handle streamflow observations
            obs_ds = self._load_streamflow_observations(spatial_mode)
            
            # Get PET method from config (default to 'oudin')
            pet_method = self.config.get('PET_METHOD', 'oudin').lower()
            self.logger.info(f"Using PET method: {pet_method}")
            
            # Calculate PET for the correct spatial configuration
            if spatial_mode == 'lumped':
                catchment = gpd.read_file(self.catchment_path / self.catchment_name)
                mean_lon, mean_lat = self._get_catchment_centroid(catchment)
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            else:
                # For distributed modes, calculate PET after spatial organization and resampling
                pet = self._calculate_distributed_pet(ds, spatial_mode, pet_method)
            
            # Ensure PET is also daily resolution by checking if resampling is needed
            pet_daily_check = pet.resample(time='D').mean()
            if len(pet_daily_check.time) != len(pet.time):
                self.logger.info("PET data needs resampling to daily resolution")
                pet = pet_daily_check
            else:
                self.logger.info("PET data is already at daily resolution")
            
            # Create FUSE forcing dataset
            fuse_forcing = self._create_fuse_forcing_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim)
            
            # Save forcing data
            output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
            encoding = self._get_encoding_dict(fuse_forcing)
            fuse_forcing.to_netcdf(output_file, unlimited_dims=['time'], 
                                encoding=encoding, format='NETCDF4')
            
            self.logger.info(f"FUSE forcing data saved: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise
        
            
    def _calculate_pet(self, temp_data: xr.DataArray, lat: float, method: str = 'oudin') -> xr.DataArray:
        """
        Calculate PET using the specified method.
        
        Args:
            temp_data (xr.DataArray): Temperature data
            lat (float): Latitude of the catchment centroid
            method (str): PET method ('oudin', 'hamon', or 'hargreaves')
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        method = method.lower()
        
        if method == 'oudin':
            return self.calculate_pet_oudin(temp_data, lat)
        elif method == 'hamon':
            return self.calculate_pet_hamon(temp_data, lat)
        elif method == 'hargreaves':
            return self.calculate_pet_hargreaves(temp_data, lat)
        else:
            self.logger.warning(f"Unknown PET method '{method}', defaulting to Oudin")
            return self.calculate_pet_oudin(temp_data, lat)

    def _add_forcing_variables(self, fuse_forcing, ds, pet, obs_ds, spatial_dims, n_subcatchments):
        """Add forcing variables to the dataset with proper dimension handling"""
        
        # Get the time dimension length from the coordinate system
        time_length = len(fuse_forcing.time)
        
        # Ensure all input data has the same time dimension
        self.logger.info(f"Expected time length: {time_length}")
        self.logger.info(f"ds time length: {len(ds.time)}")
        self.logger.info(f"pet time length: {len(pet.time)}")
        if obs_ds is not None:
            self.logger.info(f"obs_ds time length: {len(obs_ds.time)}")
        
        # Align all data to the common time coordinate
        common_time = fuse_forcing.time
        
        # For distributed case, make sure we're working with the right spatial dimension
        if len(spatial_dims) == 3:  # (time, spatial, 1) or (time, 1, spatial)
            if spatial_dims[1] in ['latitude', 'longitude'] and fuse_forcing.sizes[spatial_dims[1]] > 1:
                # Multiple subcatchments in this dimension
                target_shape = (time_length, n_subcatchments, 1)
            else:
                # Multiple subcatchments in the other dimension
                target_shape = (time_length, 1, n_subcatchments)
        else:
            target_shape = (time_length, n_subcatchments)
        
        # Core meteorological variables - extract only the time dimension that matches
        var_mapping = []
        
        # Handle precipitation
        if 'hru' in ds.dims and spatial_dims[1] != 'longitude':
            # Distributed data with HRU dimension
            pr_data = ds['pr'].values  # Shape: (time, hru)
            if pr_data.shape[0] != time_length:
                self.logger.warning(f"Precipitation time dimension mismatch: {pr_data.shape[0]} vs {time_length}")
                # Truncate or pad to match expected length
                if pr_data.shape[0] > time_length:
                    pr_data = pr_data[:time_length, :]
                else:
                    # Pad with the last available value
                    pad_length = time_length - pr_data.shape[0]
                    pad_values = np.repeat(pr_data[-1:, :], pad_length, axis=0)
                    pr_data = np.concatenate([pr_data, pad_values], axis=0)
        else:
            # Lumped data or single column
            pr_data = ds['pr'].values
            if pr_data.shape[0] != time_length:
                if pr_data.shape[0] > time_length:
                    pr_data = pr_data[:time_length]
                else:
                    pad_length = time_length - pr_data.shape[0]
                    pr_data = np.concatenate([pr_data, np.repeat(pr_data[-1], pad_length)])
        
        var_mapping.append(('pr', pr_data, 'precipitation', 'mm/day', 'Mean daily precipitation'))
        
        # Handle temperature
        if 'hru' in ds.dims and spatial_dims[1] != 'longitude':
            temp_data = ds['temp'].values
            if temp_data.shape[0] != time_length:
                if temp_data.shape[0] > time_length:
                    temp_data = temp_data[:time_length, :]
                else:
                    pad_length = time_length - temp_data.shape[0]
                    pad_values = np.repeat(temp_data[-1:, :], pad_length, axis=0)
                    temp_data = np.concatenate([temp_data, pad_values], axis=0)
        else:
            temp_data = ds['temp'].values
            if temp_data.shape[0] != time_length:
                if temp_data.shape[0] > time_length:
                    temp_data = temp_data[:time_length]
                else:
                    pad_length = time_length - temp_data.shape[0]
                    temp_data = np.concatenate([temp_data, np.repeat(temp_data[-1], pad_length)])
        
        var_mapping.append(('temp', temp_data, 'temperature', 'degC', 'Mean daily temperature'))
        
        # Handle PET
        pet_data = pet.values
        if pet_data.shape[0] != time_length:
            if pet_data.shape[0] > time_length:
                pet_data = pet_data[:time_length]
            else:
                pad_length = time_length - pet_data.shape[0]
                if len(pet_data.shape) > 1:
                    pad_values = np.repeat(pet_data[-1:, :], pad_length, axis=0)
                    pet_data = np.concatenate([pet_data, pad_values], axis=0)
                else:
                    pet_data = np.concatenate([pet_data, np.repeat(pet_data[-1], pad_length)])
        
        var_mapping.append(('pet', pet_data, 'pet', 'mm/day', 'Mean daily pet'))
        
        # Add streamflow observations
        if obs_ds is not None:
            obs_data = obs_ds['q_obs'].values
            if obs_data.shape[0] != time_length:
                if obs_data.shape[0] > time_length:
                    obs_data = obs_data[:time_length]
                else:
                    pad_length = time_length - obs_data.shape[0]
                    obs_data = np.concatenate([obs_data, np.repeat(obs_data[-1], pad_length)])
            var_mapping.append(('q_obs', obs_data, 'streamflow', 'mm/day', 'Mean observed daily discharge'))
        else:
            # Generate synthetic hydrograph for each subcatchment
            synthetic_q = self._generate_distributed_synthetic_hydrograph(ds, n_subcatchments, time_length)
            var_mapping.append(('q_obs', synthetic_q, 'streamflow', 'mm/day', 'Synthetic discharge for optimization'))
        
        # Add variables to dataset
        encoding = {}
        for var_name, data, _, units, long_name in var_mapping:
            # Reshape data to match spatial structure
            if len(data.shape) == 1:  # Time series only
                # Replicate for all subcatchments
                if target_shape[1] > target_shape[2]:  # More subcatchments in second dimension
                    reshaped_data = np.tile(data[:, np.newaxis, np.newaxis], (1, target_shape[1], 1))
                else:  # More subcatchments in third dimension
                    reshaped_data = np.tile(data[:, np.newaxis, np.newaxis], (1, 1, target_shape[2]))
            elif len(data.shape) == 2 and data.shape[1] == n_subcatchments:  # (time, subcatchments)
                # Already has subcatchment data
                if target_shape[1] > target_shape[2]:
                    reshaped_data = data[:, :, np.newaxis]
                else:
                    reshaped_data = data[:, np.newaxis, :]
            else:
                # Default case: replicate along subcatchment dimension
                if target_shape[1] > target_shape[2]:
                    reshaped_data = np.tile(data.reshape(-1, 1, 1), (1, target_shape[1], 1))
                else:
                    reshaped_data = np.tile(data.reshape(-1, 1, 1), (1, 1, target_shape[2]))
            
            # Verify final shape matches expected dimensions
            expected_shape = (time_length, fuse_forcing.sizes[spatial_dims[1]], fuse_forcing.sizes[spatial_dims[2]])
            if reshaped_data.shape != expected_shape:
                self.logger.error(f"Shape mismatch for {var_name}: got {reshaped_data.shape}, expected {expected_shape}")
                raise ValueError(f"Shape mismatch for {var_name}")
            
            # Handle NaN values
            if np.any(np.isnan(reshaped_data)):
                reshaped_data = np.nan_to_num(reshaped_data, nan=-9999.0)
            
            fuse_forcing[var_name] = xr.DataArray(
                reshaped_data,
                dims=spatial_dims,
                coords=fuse_forcing.coords,
                attrs={
                    'units': units,
                    'long_name': long_name
                }
            )
            
            encoding[var_name] = {
                '_FillValue': -9999.0,
                'dtype': 'float32'
            }
        
        return encoding

    def _generate_distributed_synthetic_hydrograph(self, ds, n_subcatchments, time_length):
        """Generate synthetic hydrograph for each subcatchment with correct time dimension"""
        
        # Use only the time-matched data for generating the hydrograph
        temp_data = ds['temp'].values
        pr_data = ds['pr'].values
        
        # Ensure we're using the correct time length
        if temp_data.shape[0] > time_length:
            temp_data = temp_data[:time_length]
        if pr_data.shape[0] > time_length:
            pr_data = pr_data[:time_length]
        
        # Create a temporary dataset for hydrograph generation
        temp_ds = xr.Dataset({
            'temp': (['time'], temp_data if len(temp_data.shape) == 1 else temp_data.mean(axis=1)),
            'pr': (['time'], pr_data if len(pr_data.shape) == 1 else pr_data.mean(axis=1))
        })
        
        base_hydrograph = self.generate_synthetic_hydrograph(temp_ds, area_km2=100.0)
        
        # Ensure the base hydrograph has the correct length
        if len(base_hydrograph) != time_length:
            if len(base_hydrograph) > time_length:
                base_hydrograph = base_hydrograph[:time_length]
            else:
                pad_length = time_length - len(base_hydrograph)
                base_hydrograph = np.concatenate([base_hydrograph, np.repeat(base_hydrograph[-1], pad_length)])
        
        # Create variations for different subcatchments (simple approach)
        variations = np.random.uniform(0.8, 1.2, n_subcatchments)  # ±20% variation
        distributed_q = np.outer(base_hydrograph, variations)  # (time, subcatchments)
        
        return distributed_q
    def _prepare_lumped_forcing(self, ds):
        """Prepare lumped forcing data (current implementation)"""
        return ds.mean(dim='hru') if 'hru' in ds.dims else ds

    def _prepare_semi_distributed_forcing(self, ds, subcatchment_dim):
        """Prepare semi-distributed forcing data using subcatchment IDs"""
        self.logger.info(f"Organizing subcatchments along {subcatchment_dim} dimension")
        
        # Load subcatchment information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Reorganize data by subcatchments
        if 'hru' in ds.dims:
            if ds.sizes['hru'] == n_subcatchments:
                # Data already matches subcatchments
                ds_subcat = ds
            else:
                # Need to aggregate/map to subcatchments
                ds_subcat = self._map_hrus_to_subcatchments(ds, subcatchments)
        else:
            # Replicate lumped data to all subcatchments
            ds_subcat = self._replicate_to_subcatchments(ds, n_subcatchments)
        
        return ds_subcat

    def _prepare_distributed_forcing(self, ds):
        """Prepare fully distributed forcing data"""
        self.logger.info("Preparing distributed forcing data")
        
        # Use HRU data directly if available
        if 'hru' in ds.dims:
            return ds
        else:
            # Need to create HRU-level data from catchment data
            return self._create_distributed_from_catchment(ds)

    def _load_subcatchment_data(self):
        """Load subcatchment information for semi-distributed mode"""
        # Check if delineated catchments exist (for distributed routing)
        delineated_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
        
        if delineated_path.exists():
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def _create_fuse_forcing_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim):
        """Create the final FUSE forcing dataset with proper coordinate structure"""
        
        if spatial_mode == 'lumped':
            return self._create_lumped_dataset(ds, pet, obs_ds)
        else:
            return self._create_distributed_dataset(ds, pet, obs_ds, spatial_mode, subcatchment_dim)

    def _create_distributed_dataset(self, ds, pet, obs_ds, spatial_mode, subcatchment_dim):
        """Create distributed/semi-distributed FUSE forcing dataset"""
        
        # Get spatial information
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Get reference coordinates
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self._get_catchment_centroid(catchment)
        
        # Convert time to days since 1970-01-01
        time_index = pd.date_range(start=ds.time.min().values, end=ds.time.max().values, freq='D')
        time_days = (time_index - pd.Timestamp('1970-01-01')).days.values
        
        # Create coordinate system based on subcatchment dimension choice
        if subcatchment_dim == 'latitude':
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', subcatchments.astype(float)),  # Subcatchment IDs as pseudo-lat
                'time': ('time', time_days)
            }
            spatial_dims = ('time', 'latitude', 'longitude')
        else:  # longitude
            coords = {
                'longitude': ('longitude', subcatchments.astype(float)),  # Subcatchment IDs as pseudo-lon
                'latitude': ('latitude', [mean_lat]),
                'time': ('time', time_days)
            }
            spatial_dims = ('time', 'longitude', 'latitude')
        
        # Create dataset
        fuse_forcing = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        fuse_forcing.longitude.attrs = {
            'units': 'degreesE' if subcatchment_dim != 'longitude' else 'subcatchment_id',
            'long_name': 'longitude' if subcatchment_dim != 'longitude' else 'subcatchment identifier'
        }
        fuse_forcing.latitude.attrs = {
            'units': 'degreesN' if subcatchment_dim != 'latitude' else 'subcatchment_id',
            'long_name': 'latitude' if subcatchment_dim != 'latitude' else 'subcatchment identifier'
        }
        fuse_forcing.time.attrs = {
            'units': 'days since 1970-01-01',
            'long_name': 'time'
        }
        
        # Add data variables
        self._add_forcing_variables(fuse_forcing, ds, pet, obs_ds, spatial_dims, n_subcatchments)
        
        return fuse_forcing


    def create_filemanager(self):
        """
        Create FUSE file manager file by modifying template with project-specific settings.
        """
        self.logger.info("Creating FUSE file manager file")

        # Define source and destination paths
        template_path = self.fuse_setup_dir / 'fm_catch.txt'
        
        # Define the paths to replace
        settings = {
            'SETNGS_PATH': str(self.project_dir / 'settings' / 'FUSE') + '/',
            'INPUT_PATH': str(self.project_dir / 'forcing' / 'FUSE_input') + '/',
            'OUTPUT_PATH': str(self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FUSE') + '/',
            'METRIC': self.config['OPTIMIZATION_METRIC'],
            'MAXN': str(self.config['NUMBER_OF_ITERATIONS']),
            'FMODEL_ID': self.config['EXPERIMENT_ID'],
            'M_DECISIONS': f"fuse_zDecisions_{self.config['EXPERIMENT_ID']}.txt"
        }

        # Get and format dates from config
        start_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_START'), '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(self.config.get('EXPERIMENT_TIME_END'), '%Y-%m-%d %H:%M')
        cal_start_time = datetime.strptime(self.config.get('CALIBRATION_PERIOD').split(',')[0], '%Y-%m-%d')
        cal_end_time = datetime.strptime(self.config.get('CALIBRATION_PERIOD').split(',')[1].strip(), '%Y-%m-%d')

        date_settings = {
            'date_start_sim': start_time.strftime('%Y-%m-%d'),
            'date_end_sim': end_time.strftime('%Y-%m-%d'),
            'date_start_eval': cal_start_time.strftime('%Y-%m-%d'),  # Using same dates for evaluation period
            'date_end_eval': cal_end_time.strftime('%Y-%m-%d')       # Can be modified if needed
        }

        try:
            # Read the template file
            with open(template_path, 'r') as f:
                lines = f.readlines()

            # Process each line
            modified_lines = []
            for line in lines:
                line_modified = line
                
                # Replace paths
                for path_key, new_path in settings.items():
                    if path_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_path + line[end:]
                            self.logger.debug(f"Updated {path_key} path to: {new_path}")

                # Replace dates
                for date_key, new_date in date_settings.items():
                    if date_key in line:
                        # Find the content between quotes and replace it
                        start = line.find("'") + 1
                        end = line.find("'", start)
                        if start > 0 and end > 0:
                            line_modified = line[:start] + new_date + line[end:]
                            self.logger.debug(f"Updated {date_key} to: {new_date}")

                modified_lines.append(line_modified)

            # Write the modified file
            with open(template_path, 'w') as f:
                f.writelines(modified_lines)

            self.logger.info(f"FUSE file manager created at: {template_path}")
        

        except Exception as e:
            self.logger.error(f"Error creating FUSE file manager: {str(e)}")
            raise

        

    def _get_catchment_centroid(self, catchment_gdf):
        """
        Helper function to correctly calculate catchment centroid with proper CRS handling.
        
        Args:
            catchment_gdf (gpd.GeoDataFrame): The catchment GeoDataFrame
        
        Returns:
            tuple: (longitude, latitude) of the catchment centroid
        """
        # Ensure we have the CRS information
        if catchment_gdf.crs is None:
            self.logger.warning("Catchment CRS is not defined, assuming EPSG:4326")
            catchment_gdf.set_crs(epsg=4326, inplace=True)
            
        # Convert to geographic coordinates if not already
        catchment_geo = catchment_gdf.to_crs(epsg=4326)
        
        # Get a rough center point (using bounds instead of centroid)
        bounds = catchment_geo.total_bounds  # (minx, miny, maxx, maxy)
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        
        # Calculate UTM zone from the center point
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = f"326{utm_zone:02d}" if center_lat >= 0 else f"327{utm_zone:02d}"
        
        # Project to appropriate UTM zone
        catchment_utm = catchment_geo.to_crs(f"EPSG:{epsg_code}")
        
        # Calculate centroid in UTM coordinates
        centroid_utm = catchment_utm.geometry.centroid.iloc[0]
        
        # Create a GeoDataFrame with the centroid point
        centroid_gdf = gpd.GeoDataFrame(
            geometry=[centroid_utm], 
            crs=f"EPSG:{epsg_code}"
        )
        
        # Convert back to geographic coordinates
        centroid_geo = centroid_gdf.to_crs(epsg=4326)
        
        # Extract coordinates
        lon, lat = centroid_geo.geometry.x[0], centroid_geo.geometry.y[0]
        
        self.logger.info(f"Calculated catchment centroid: {lon:.6f}°E, {lat:.6f}°N (UTM Zone {utm_zone})")
        
        return lon, lat

    def create_elevation_bands(self):
        """Create elevation bands netCDF file for FUSE input with distributed support."""
        self.logger.info("Creating elevation bands file for FUSE")

        try:
            # Check spatial mode
            spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            
            if spatial_mode == 'lumped':
                return self._create_lumped_elevation_bands()
            else:
                return self._create_distributed_elevation_bands()
                
        except Exception as e:
            self.logger.error(f"Error creating elevation bands file: {str(e)}")
            raise

    def _create_distributed_elevation_bands(self):
        """Create elevation bands for distributed mode"""
        
        # Load subcatchment information to get spatial dimensions
        subcatchments = self._load_subcatchment_data()
        n_subcatchments = len(subcatchments)
        
        # Get reference coordinates (same as used in forcing file)
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self._get_catchment_centroid(catchment)
        
        # For now, create simple elevation bands for all subcatchments
        # In future, this could use subcatchment-specific elevation data
        
        # Create dataset with distributed spatial structure
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        if subcatchment_dim == 'latitude':
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', subcatchments.astype(float)),
                'elevation_band': ('elevation_band', [1])  # Simple single band for now
            }
            spatial_dims = ['elevation_band', 'latitude', 'longitude']
        else:
            coords = {
                'longitude': ('longitude', subcatchments.astype(float)),
                'latitude': ('latitude', [mean_lat]), 
                'elevation_band': ('elevation_band', [1])
            }
            spatial_dims = ['elevation_band', 'longitude', 'latitude']
        
        ds = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'} 
        ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
        
        # Create elevation band variables for all subcatchments
        target_shape = (1, n_subcatchments, 1) if subcatchment_dim == 'latitude' else (1, 1, n_subcatchments)
        
        for var_name, value, attrs in [
            ('area_frac', 1.0, {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
            ('mean_elev', 1000.0, {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
            ('prec_frac', 1.0, {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
        ]:
            
            data = np.full(target_shape, value, dtype=np.float32)
            
            ds[var_name] = xr.DataArray(
                data,
                dims=spatial_dims,
                coords=ds.coords,
                attrs=attrs
            )
        
        # Save with proper encoding
        output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
        encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
        
        ds.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
        
        self.logger.info(f"Created distributed elevation bands file: {output_file}")
        return output_file

    def _load_streamflow_observations(self, spatial_mode):
        """
        Load streamflow observations for FUSE forcing data.
        
        Args:
            spatial_mode (str): Spatial mode ('lumped', 'semi_distributed', 'distributed')
            
        Returns:
            xr.Dataset or None: Dataset containing observed streamflow or None if not available
        """
        try:
            # Get observations file path
            obs_file_path = self.config.get('OBSERVATIONS_PATH', 'default')
            if obs_file_path == 'default':
                obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            else:
                obs_file_path = Path(obs_file_path)
            
            # Check if observations file exists
            if not obs_file_path.exists():
                self.logger.warning(f"Streamflow observations file not found: {obs_file_path}")
                self.logger.info("Will generate synthetic hydrograph for optimization")
                return None
            
            # Read observations
            self.logger.info(f"Loading streamflow observations from: {obs_file_path}")
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            
            # Resample to daily and get discharge
            if 'discharge_cms' in dfObs.columns:
                obs_streamflow = dfObs['discharge_cms'].resample('D').mean()
            elif 'discharge' in dfObs.columns: 
                obs_streamflow = dfObs['discharge'].resample('D').mean()
            else:
                available_cols = list(dfObs.columns)
                self.logger.warning(f"No discharge column found. Available columns: {available_cols}")
                return None
            
            # Get catchment area for unit conversion if needed
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            
            if basin_path.exists():
                basin_gdf = gpd.read_file(basin_path)
                area_km2 = basin_gdf['GRU_area'].sum() / 1e6  # Convert m2 to km2
            else:
                # Fallback area estimate
                catchment = gpd.read_file(self.catchment_path / self.catchment_name)
                catchment_proj = catchment.to_crs(catchment.estimate_utm_crs())
                area_km2 = catchment_proj.geometry.area.sum() / 1e6
                self.logger.warning(f"Using estimated catchment area: {area_km2:.2f} km2")
            
            # Convert from cms to mm/day for FUSE
            # Q(mm/day) = Q(cms) * 86.4 / Area(km2)
            obs_streamflow_mm = obs_streamflow * 86.4 / area_km2
            
            # Create time coordinate in days since 1970-01-01 (matching FUSE format)
            time_days = (obs_streamflow_mm.index - pd.Timestamp('1970-01-01')).days.values
            
            # Create xarray dataset
            obs_ds = xr.Dataset(
                {
                    'q_obs': xr.DataArray(
                        obs_streamflow_mm.values,
                        dims=['time'],
                        coords={'time': time_days},
                        attrs={
                            'units': 'mm/day',
                            'long_name': 'Observed daily discharge',
                            'standard_name': 'water_volume_transport_in_river_channel'
                        }
                    )
                },
                coords={
                    'time': xr.DataArray(
                        time_days,
                        dims=['time'],
                        attrs={
                            'units': 'days since 1970-01-01',
                            'long_name': 'time'
                        }
                    )
                }
            )
            
            self.logger.info(f"Loaded {len(obs_streamflow_mm)} days of streamflow observations")
            self.logger.info(f"Converted from cms to mm/day using area: {area_km2:.2f} km2")
            
            return obs_ds
            
        except Exception as e:
            self.logger.error(f"Error loading streamflow observations: {str(e)}")
            self.logger.info("Will generate synthetic hydrograph for optimization")
            return None            

    def _create_lumped_elevation_bands(self):
        """Create elevation bands for lumped mode"""
        self.logger.info("Creating lumped elevation bands file")

        try:
            # Get catchment centroid for coordinates
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self._get_catchment_centroid(catchment)
            
            # Create simple single elevation band for lumped mode
            coords = {
                'longitude': ('longitude', [mean_lon]),
                'latitude': ('latitude', [mean_lat]),
                'elevation_band': ('elevation_band', [1])
            }
            
            ds = xr.Dataset(coords=coords)
            
            # Add coordinate attributes
            ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
            ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'} 
            ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
            
            # Create elevation band variables (single band covering entire catchment)
            target_shape = (1, 1, 1)  # (elevation_band, latitude, longitude)
            spatial_dims = ['elevation_band', 'latitude', 'longitude']
            
            for var_name, value, attrs in [
                ('area_frac', 1.0, {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
                ('mean_elev', 1000.0, {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
                ('prec_frac', 1.0, {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
            ]:
                
                data = np.full(target_shape, value, dtype=np.float32)
                
                ds[var_name] = xr.DataArray(
                    data,
                    dims=spatial_dims,
                    coords=ds.coords,
                    attrs=attrs
                )
            
            # Save with proper encoding
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
            
            ds.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
            
            self.logger.info(f"Created lumped elevation bands file: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error creating lumped elevation bands: {str(e)}")
            raise

    def _calculate_distributed_pet(self, ds, spatial_mode, pet_method='oudin'):
        """
        Calculate PET for distributed/semi-distributed modes.
        
        Args:
            ds: xarray dataset with temperature data
            spatial_mode (str): Spatial mode ('semi_distributed', 'distributed')
            pet_method (str): PET calculation method
            
        Returns:
            xr.DataArray: Calculated PET data
        """
        self.logger.info(f"Calculating distributed PET for {spatial_mode} mode using {pet_method}")
        
        try:
            # Get catchment for reference latitude
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self._get_catchment_centroid(catchment)
            
            # For distributed modes, use the same latitude for all subcatchments/HRUs
            if 'hru' in ds.dims:
                # Use the mean temperature across all HRUs to calculate PET once
                temp_mean = ds['temp'].mean(dim='hru')
                pet_base = self._calculate_pet(temp_mean, mean_lat, pet_method)
                
                # Replicate the PET calculation for each HRU with correct dimension order
                n_hrus = ds.sizes['hru']
                
                # Create a list of the base PET for each HRU and concatenate along HRU dimension
                pet_list = []
                for i in range(n_hrus):
                    pet_list.append(pet_base)
                
                # Concatenate along new HRU dimension, ensuring time is first dimension
                pet = xr.concat(pet_list, dim='hru')
                # Transpose to ensure correct dimension order: (time, hru)
                pet = pet.transpose('time', 'hru')
                
                self.logger.info(f"Calculated distributed PET with shape: {pet.shape}")
            else:
                # Use lumped calculation as fallback
                pet = self._calculate_pet(ds['temp'], mean_lat, pet_method)
            
            return pet
            
        except Exception as e:
            self.logger.warning(f"Error calculating distributed PET, falling back to lumped: {str(e)}")
            # Fallback to lumped calculation
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self._get_catchment_centroid(catchment)
            return self._calculate_pet(ds['temp'], mean_lat, pet_method)

    def _get_encoding_dict(self, fuse_forcing):
        """
        Get encoding dictionary for netCDF output.
        
        Args:
            fuse_forcing: xarray Dataset to encode
            
        Returns:
            Dict: Encoding dictionary for netCDF
        """
        encoding = {}
        
        # Default encoding for coordinates
        for coord in fuse_forcing.coords:
            if coord == 'time':
                encoding[coord] = {
                    'dtype': 'float64'
                    # NOTE: 'units' should NOT be here - it belongs in attributes only
                }
            elif coord in ['longitude', 'latitude']:
                encoding[coord] = {
                    'dtype': 'float64'
                }
            else:
                encoding[coord] = {
                    'dtype': 'float32'
                }
        
        # Default encoding for data variables
        for var in fuse_forcing.data_vars:
            encoding[var] = {
                '_FillValue': -9999.0,
                'dtype': 'float32'
            }
        
        return encoding

    def _create_lumped_dataset(self, ds, pet, obs_ds):
        """
        Create lumped FUSE forcing dataset (fixed implementation).
        
        Args:
            ds: Processed forcing dataset
            pet: Calculated PET data
            obs_ds: Observed streamflow dataset (or None)
            
        Returns:
            xr.Dataset: FUSE forcing dataset for lumped mode
        """
        # Get catchment centroid for coordinates
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self._get_catchment_centroid(catchment)
        
        # Convert all time coordinates to pandas datetime for comparison
        # ds and pet should already be in datetime format
        ds_start = pd.to_datetime(ds.time.min().values)
        ds_end = pd.to_datetime(ds.time.max().values)
        pet_start = pd.to_datetime(pet.time.min().values)
        pet_end = pd.to_datetime(pet.time.max().values)
        
        # Find overlapping time period
        start_time = max(ds_start, pet_start)
        end_time = min(ds_end, pet_end)
        
        if obs_ds is not None:
            # Convert obs_ds time to datetime if it's in days since 1970-01-01
            if obs_ds.time.dtype.kind in ['i', 'u']:  # integer types
                obs_time_dt = pd.to_datetime('1970-01-01') + pd.to_timedelta(obs_ds.time.values, unit='D')
                obs_start = obs_time_dt.min()
                obs_end = obs_time_dt.max()
            else:
                obs_start = pd.to_datetime(obs_ds.time.min().values)
                obs_end = pd.to_datetime(obs_ds.time.max().values)
            
            start_time = max(start_time, obs_start)
            end_time = min(end_time, obs_end)
        
        self.logger.info(f"Aligning all data to common time period: {start_time} to {end_time}")
        
        # Create explicit time index for the overlapping period
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        
        # Align all datasets to the common time period
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        if obs_ds is not None:
            # Handle obs_ds reindexing based on its time format
            if obs_ds.time.dtype.kind in ['i', 'u']:  # integer types
                # Convert time index to days since 1970-01-01 for selection
                start_days = (pd.to_datetime(start_time) - pd.Timestamp('1970-01-01')).days
                end_days = (pd.to_datetime(end_time) - pd.Timestamp('1970-01-01')).days
                time_days_index = (time_index - pd.Timestamp('1970-01-01')).days.values
                
                # Select and reindex with integer time
                obs_ds = obs_ds.sel(time=slice(start_days, end_days))
                obs_ds = obs_ds.reindex(time=time_days_index)
            else:
                obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        # Convert time to days since 1970-01-01 for final dataset
        time_days = (time_index - pd.Timestamp('1970-01-01')).days.values
        
        # Create coordinates
        coords = {
            'longitude': ('longitude', [mean_lon]),
            'latitude': ('latitude', [mean_lat]),
            'time': ('time', time_days)
        }
        
        # Create dataset
        fuse_forcing = xr.Dataset(coords=coords)
        
        # Add coordinate attributes
        fuse_forcing.longitude.attrs = {
            'units': 'degreesE',
            'long_name': 'longitude'
        }
        fuse_forcing.latitude.attrs = {
            'units': 'degreesN',
            'long_name': 'latitude'
        }
        fuse_forcing.time.attrs = {
            'units': 'days since 1970-01-01',
            'long_name': 'time'
        }
        
        # Add forcing variables
        spatial_dims = ('time', 'latitude', 'longitude')
        
        # Core meteorological variables
        var_mapping = [
            ('pr', ds['pr'].values, 'precipitation', 'mm/day', 'Mean daily precipitation'),
            ('temp', ds['temp'].values, 'temperature', 'degC', 'Mean daily temperature'),
            ('pet', pet.values, 'pet', 'mm/day', 'Mean daily pet')
        ]
        
        # Add streamflow observations
        if obs_ds is not None:
            var_mapping.append(('q_obs', obs_ds['q_obs'].values, 'streamflow', 'mm/day', 'Mean observed daily discharge'))
        else:
            # Generate synthetic hydrograph
            synthetic_q = self.generate_synthetic_hydrograph(ds, area_km2=100.0)
            var_mapping.append(('q_obs', synthetic_q, 'streamflow', 'mm/day', 'Synthetic discharge for optimization'))
        
        # Add variables to dataset
        for var_name, data, _, units, long_name in var_mapping:
            # Reshape data to match spatial structure (time, lat, lon)
            if len(data.shape) == 1:  # Time series only
                reshaped_data = data[:, np.newaxis, np.newaxis]
            else:
                reshaped_data = data.reshape(-1, 1, 1)
            
            # Handle NaN values
            if np.any(np.isnan(reshaped_data)):
                reshaped_data = np.nan_to_num(reshaped_data, nan=-9999.0)
            
            # Verify dimensions match
            if reshaped_data.shape[0] != len(time_days):
                self.logger.error(f"Dimension mismatch for {var_name}: data has {reshaped_data.shape[0]} time points, coordinate has {len(time_days)}")
                raise ValueError(f"Dimension mismatch for {var_name}")
            
            fuse_forcing[var_name] = xr.DataArray(
                reshaped_data,
                dims=spatial_dims,
                coords=fuse_forcing.coords,
                attrs={
                    'units': units,
                    'long_name': long_name
                }
            )
        
        return fuse_forcing

    def _map_hrus_to_subcatchments(self, ds, subcatchments):
        """
        Map HRU data to subcatchments for semi-distributed mode.
        
        Args:
            ds: Dataset with HRU dimension
            subcatchments: Array of subcatchment IDs
            
        Returns:
            xr.Dataset: Dataset organized by subcatchments
        """
        self.logger.info("Mapping HRUs to subcatchments")
        
        # Simple approach: assume HRUs map directly to subcatchments
        # This could be enhanced with actual HRU-subcatchment mapping
        n_hrus = ds.sizes['hru']
        n_subcatchments = len(subcatchments)
        
        if n_hrus == n_subcatchments:
            # Direct mapping
            return ds.rename({'hru': 'subcatchment'})
        elif n_hrus > n_subcatchments:
            # Aggregate HRUs to subcatchments (simple averaging)
            hrus_per_subcat = n_hrus // n_subcatchments
            subcatchment_data = []
            
            for i in range(n_subcatchments):
                start_idx = i * hrus_per_subcat
                end_idx = start_idx + hrus_per_subcat
                if i == n_subcatchments - 1:  # Last subcatchment gets remaining HRUs
                    end_idx = n_hrus
                
                subcat_data = ds.isel(hru=slice(start_idx, end_idx)).mean(dim='hru')
                subcatchment_data.append(subcat_data)
            
            # Combine subcatchments
            ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
            ds_subcat['subcatchment'] = subcatchments
            
            return ds_subcat
        else:
            # Replicate HRU data to subcatchments
            return self._replicate_to_subcatchments(ds, n_subcatchments)

    def _replicate_to_subcatchments(self, ds, n_subcatchments):
        """
        Replicate lumped data to all subcatchments.
        
        Args:
            ds: Lumped dataset
            n_subcatchments: Number of subcatchments
            
        Returns:
            xr.Dataset: Dataset replicated to subcatchments
        """
        self.logger.info(f"Replicating data to {n_subcatchments} subcatchments")
        
        # Create subcatchment dimension
        subcatchment_data = []
        for i in range(n_subcatchments):
            subcatchment_data.append(ds)
        
        # Combine along new subcatchment dimension
        ds_subcat = xr.concat(subcatchment_data, dim='subcatchment')
        ds_subcat['subcatchment'] = range(1, n_subcatchments + 1)
        
        return ds_subcat

    def _create_distributed_from_catchment(self, ds):
        """
        Create HRU-level data from catchment data for distributed mode.
        
        Args:
            ds: Catchment-level dataset
            
        Returns:
            xr.Dataset: HRU-level dataset
        """
        self.logger.info("Creating distributed data from catchment data")
        
        # Load catchment shapefile to get number of HRUs
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        n_hrus = len(catchment)
        
        # Replicate catchment data to all HRUs
        hru_data = []
        for i in range(n_hrus):
            hru_data.append(ds)
        
        # Combine along HRU dimension
        ds_hru = xr.concat(hru_data, dim='hru')
        ds_hru['hru'] = range(1, n_hrus + 1)
        
        return ds_hru

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.

        Args:
            path_key (str): The key to look up in the config dictionary.
            default_subpath (str): The default subpath to use if the config value is 'default'.

        Returns:
            Path: The resolved path.

        Raises:
            KeyError: If the path_key is not found in the config.
        """
        try:
            path_value = self.config.get(path_key)
            if path_value == 'default' or path_value is None:
                return self.project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            self.logger.error(f"Config key '{path_key}' not found")
            raise
         
    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))

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
        self.result_dir = self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize required paths
        self.fuse_path = self._get_install_path()
        self.output_path = self._get_output_path()
        self.fuse_setup_dir = self.project_dir / "settings" / "FUSE"
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'
        
        # Spatial mode
        self.spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
        self.needs_routing = self._check_routing_requirements()

    def _convert_fuse_distributed_to_mizuroute_format(self):
        """
        Convert FUSE spatial dimensions to mizuRoute format.
        MINIMAL changes only: latitude->gru, add gruId, squeeze longitude.
        Preserves ALL original data and time coordinates unchanged.
        """
        import xarray as xr
        import numpy as np
        import shutil
        import tempfile
        import os

        experiment_id = self.config.get('EXPERIMENT_ID')
        domain = self.domain_name
        
        fuse_out_dir = self.project_dir / "simulations" / experiment_id / "FUSE"
        
        # Find FUSE output file
        target_files = [
            fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc",
            fuse_out_dir / f"{domain}_{experiment_id}_runs_best.nc"
        ]
        
        target = None
        for file_path in target_files:
            if file_path.exists():
                target = file_path
                break
        
        if target is None:
            raise FileNotFoundError(f"FUSE output not found. Tried: {[str(f) for f in target_files]}")

        self.logger.debug(f"Converting FUSE spatial dimensions: {target}")

        # Create backup
        backup_file = target.with_suffix('.backup.nc')
        if not backup_file.exists():
            shutil.copy2(target, backup_file)
            self.logger.info(f"Created backup: {backup_file}")

        # Load, modify, and immediately close the dataset
        with xr.open_dataset(target) as ds:
            self.logger.debug(f"Original dimensions: {dict(ds.sizes)}")
            
            # Step 1: Remove singleton longitude dimension if it exists
            if 'longitude' in ds.sizes and ds.sizes['longitude'] == 1:
                ds = ds.squeeze('longitude', drop=True)
                self.logger.debug("Squeezed longitude dimension")
            
            # Step 2: Rename latitude dimension to gru
            if 'latitude' in ds.sizes:
                ds = ds.rename({'latitude': 'gru'})
                self.logger.debug("Renamed latitude -> gru")
                
                # Step 3: Create gruId variable from gru coordinates
                if 'gru' in ds.coords:
                    gru_values = ds.coords['gru'].values
                    try:
                        # Try to convert to integers
                        gru_ids = gru_values.astype('int32')
                    except (ValueError, TypeError):
                        # If conversion fails, use sequential IDs
                        gru_ids = np.arange(1, len(gru_values) + 1, dtype='int32')
                        self.logger.warning(f"Using sequential GRU IDs 1-{len(gru_values)}")
                    
                    ds['gruId'] = xr.DataArray(
                        gru_ids,
                        dims=('gru',),
                        attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                    )
                    
                    self.logger.debug(f"Created gruId variable with {len(gru_ids)} GRUs")
                else:
                    raise ValueError("No gru coordinate found after renaming")
            else:
                raise ValueError("No latitude dimension found in FUSE output")
            
            self.logger.debug(f"Final dimensions: {dict(ds.sizes)}")
            
            # Load all data into memory before closing
            ds = ds.load()
        
        # Now the original file is closed, we can write to a temp file and replace
        try:
            # Make sure target file is writable
            try:
                os.chmod(target, 0o664)
            except Exception as e:
                self.logger.warning(f"Could not change file permissions: {e}")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=target.parent) as tmp_file:
                temp_path = tmp_file.name
            
            # Write the modified dataset to temp file
            ds.to_netcdf(temp_path, format='NETCDF4')
            
            # Replace original with temp file
            shutil.move(temp_path, str(target))
            self.logger.debug(f"Spatial conversion completed: {target}")
            
            # Ensure _runs_def.nc exists if we processed a different file
            def_file = fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc"
            if target != def_file and not def_file.exists():
                shutil.copy2(target, def_file)
                self.logger.info(f"Created runs_def file: {def_file}")
                
        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise

    def run_fuse(self) -> Optional[Path]:
        """Run FUSE model with distributed support"""
        self.logger.debug(f"Starting FUSE model run in {self.spatial_mode} mode")
        
        try:
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Run FUSE simulations
            success = self._execute_fuse_workflow()
            
            if success:
                # Handle routing if needed
                if self.needs_routing:
                    self._convert_fuse_distributed_to_mizuroute_format()
                    success = self._run_distributed_routing()
                
                if success:
                    self._process_outputs()
                    self.logger.debug("FUSE run completed successfully")
                    return self.output_path
                else:
                    self.logger.error("FUSE routing failed")
                    return None
            else:
                self.logger.error("FUSE simulation failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during FUSE run: {str(e)}")
            raise

    def _check_routing_requirements(self) -> bool:
        """Check if distributed routing is needed"""
        routing_integration = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
        
        if routing_integration == 'mizuRoute':
            if self.spatial_mode in ['semi_distributed', 'distributed']:
                return True
            elif self.spatial_mode == 'lumped' and self.config.get('ROUTING_DELINEATION') == 'river_network':
                return True
        
        return False

    def _execute_fuse_workflow(self) -> bool:
        """Execute the main FUSE workflow based on spatial mode"""
        
        if self.spatial_mode == 'lumped':
            # Original lumped workflow
            return self._run_lumped_fuse()
        else:
            # Distributed workflow
            return self._run_distributed_fuse()

    def _run_distributed_fuse(self) -> bool:
        """Run FUSE in distributed mode - always process the full dataset at once"""
        self.logger.debug("Running distributed FUSE workflow with full dataset")
        
        try:
            # Run FUSE once with the complete distributed forcing file
            return self._run_multidimensional_fuse()
                    
        except Exception as e:
            self.logger.error(f"Error in distributed FUSE execution: {str(e)}")
            return False

    def _run_multidimensional_fuse(self) -> bool:
        """Run FUSE once with the full distributed forcing file"""
        
        try:
            self.logger.debug("Running FUSE with complete distributed forcing dataset")
            
            # Run FUSE with the distributed forcing file (all HRUs at once)
            success = self._execute_fuse_distributed()
            
            if success:
                self.logger.debug("Distributed FUSE run completed successfully")
                return True
            else:
                self.logger.error("Distributed FUSE run failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in multidimensional FUSE execution: {str(e)}")
            return False

    def _execute_fuse_distributed(self) -> bool:
        """Execute FUSE with the complete distributed forcing file"""
        
        try:
            # Use the main file manager (points to distributed forcing file)
            fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
            control_file = self.fuse_setup_dir / 'fm_catch.txt'
            
            # Run FUSE once for the entire distributed domain
            command = [
                str(fuse_exe),
                str(control_file),
                self.domain_name,  # Use original domain name
                "run_def"  # Run with default parameters
            ]
            
            # Create log file
            log_file = self.output_path / 'fuse_distributed_run.log'
            
            self.logger.debug(f"Executing distributed FUSE: {' '.join(command)}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(self.fuse_setup_dir)
                )
            
            if result.returncode == 0:
                self.logger.debug("Distributed FUSE execution completed successfully")
                return True
            else:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing distributed FUSE: {str(e)}")
            return False


    def _create_subcatchment_settings(self, subcat_id: int, index: int) -> Path:
        """Create subcatchment-specific settings files"""
        
        try:
            # Create subcatchment-specific settings directory
            subcat_settings_dir = self.fuse_setup_dir / f"subcat_{subcat_id}"
            subcat_settings_dir.mkdir(exist_ok=True)
            
            # Copy base settings files
            base_settings_dir = self.fuse_setup_dir
            
            for file in base_settings_dir.glob("*.txt"):
                if "subcat_" not in file.name:  # Don't copy other subcatchment files
                    dest_file = subcat_settings_dir / file.name
                    shutil.copy2(file, dest_file)
            
            # Update file manager for this subcatchment
            fm_file = subcat_settings_dir / 'fm_catch.txt'
            if fm_file.exists():
                with open(fm_file, 'r') as f:
                    content = f.read()
                
                # Update paths to point to subcatchment-specific files
                content = content.replace(
                    f"{self.domain_name}_input.nc",
                    f"subcat_{subcat_id}_input.nc"
                )
                content = content.replace(
                    f"/{self.config['EXPERIMENT_ID']}/FUSE/",
                    f"/{self.config['EXPERIMENT_ID']}/FUSE/subcat_{subcat_id}/"
                )
                
                with open(fm_file, 'w') as f:
                    f.write(content)
            
            return subcat_settings_dir
            
        except Exception as e:
            self.logger.error(f"Error creating subcatchment settings for {subcat_id}: {str(e)}")
            raise

    def _execute_fuse_subcatchment(self, subcat_id: int, forcing_file: Path, settings_dir: Path) -> Optional[Path]:
        """Execute FUSE for a specific subcatchment"""
        
        try:
            # Create subcatchment output directory
            subcat_output_dir = self.output_path / f"subcat_{subcat_id}"
            subcat_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create elevation bands file for this subcatchment
            self._create_subcatchment_elevation_bands(subcat_id)
            
            # Run FUSE with subcatchment-specific settings
            fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
            control_file = settings_dir / 'fm_catch.txt'
            
            command = [
                str(fuse_exe),
                str(control_file),
                f"{self.domain_name}_subcat_{subcat_id}",
                "run_def"  # Run with default parameters for distributed mode
            ]
            
            # Create log file for this subcatchment
            log_file = subcat_output_dir / 'fuse_run.log'
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(settings_dir)
                )
            
            if result.returncode == 0:
                # Find and return the output file
                output_files = list(subcat_output_dir.glob("*_runs_best.nc"))
                if output_files:
                    return output_files[0]
                else:
                    self.logger.warning(f"No output file found for subcatchment {subcat_id}")
                    return None
            else:
                self.logger.error(f"FUSE failed for subcatchment {subcat_id} with return code {result.returncode}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing FUSE for subcatchment {subcat_id}: {str(e)}")
            return None

    def _ensure_best_output_file(self):
        """Ensure the expected 'best' output file exists by copying from 'def' output if needed"""
        
        def_file = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_runs_def.nc"
        best_file = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_runs_best.nc"
        
        if def_file.exists() and not best_file.exists():
            self.logger.info(f"Copying {def_file.name} to {best_file.name} for compatibility")
            shutil.copy2(def_file, best_file)
        
        return best_file if best_file.exists() else def_file

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment while preserving proper netCDF structure"""
        
        # Load distributed forcing data
        forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_input.nc"
        ds = xr.open_dataset(forcing_file)
        
        # Extract data for this subcatchment based on coordinate system
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        try:
            if subcatchment_dim == 'latitude':
                # Find the index of this subcatchment ID in the latitude coordinates
                lat_coords = ds.latitude.values
                try:
                    subcat_idx = list(lat_coords).index(float(subcat_id))
                except ValueError:
                    # If exact match not found, use the index directly
                    if index < len(lat_coords):
                        subcat_idx = index
                    else:
                        raise ValueError(f"Subcatchment index {index} out of range")
                
                # Extract data for this subcatchment but preserve the dimensional structure
                subcat_data = ds.isel(latitude=slice(subcat_idx, subcat_idx + 1))
                
            else:
                # Similar logic for longitude dimension
                lon_coords = ds.longitude.values
                try:
                    subcat_idx = list(lon_coords).index(float(subcat_id))
                except ValueError:
                    if index < len(lon_coords):
                        subcat_idx = index
                    else:
                        raise ValueError(f"Subcatchment index {index} out of range")
                
                subcat_data = ds.isel(longitude=slice(subcat_idx, subcat_idx + 1))
            
            # Now subcat_data should have the same dimensional structure as the original
            # but with latitude=1 (or longitude=1) instead of latitude=49
            
            # Verify the structure
            expected_dims = ['time', 'latitude', 'longitude']
            for var in ['pr', 'temp', 'pet', 'q_obs']:
                if var in subcat_data:
                    actual_dims = list(subcat_data[var].dims)
                    if actual_dims != expected_dims:
                        self.logger.error(f"Dimension mismatch for {var}: got {actual_dims}, expected {expected_dims}")
                        raise ValueError(f"Dimension structure incorrect for {var}")
            
            # Preserve all attributes
            for var in subcat_data.data_vars:
                if var in ds:
                    subcat_data[var].attrs = ds[var].attrs.copy()
            
            for coord in subcat_data.coords:
                if coord in ds.coords:
                    subcat_data[coord].attrs = ds[coord].attrs.copy()
            
            subcat_data.attrs = ds.attrs.copy()
            subcat_data.attrs['subcatchment_id'] = subcat_id
            
            # Save with proper encoding
            subcat_forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_input.nc"
            
            encoding = {}
            for var in subcat_data.data_vars:
                encoding[var] = {
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }
            
            for coord in subcat_data.coords:
                if coord == 'time':
                    encoding[coord] = {'dtype': 'float64'}
                else:
                    encoding[coord] = {'dtype': 'float64'}
            
            subcat_data.to_netcdf(
                subcat_forcing_file,
                encoding=encoding,
                format='NETCDF4',
                unlimited_dims=['time']
            )
            
            ds.close()
            subcat_data.close()
            
            self.logger.info(f"Created forcing file for subcatchment {subcat_id}: {subcat_forcing_file}")
            return subcat_forcing_file
            
        except Exception as e:
            self.logger.error(f"Error extracting forcing for subcatchment {subcat_id}: {str(e)}")
            ds.close()
            raise

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments into distributed format"""
        
        self.logger.info(f"Combining outputs from {len(outputs)} subcatchments")
        
        combined_outputs = {}
        
        # Load and combine all subcatchment outputs
        for subcat_id, output_file in outputs:
            try:
                ds = xr.open_dataset(output_file)
                
                # Store with subcatchment identifier
                for var_name in ds.data_vars:
                    if var_name not in combined_outputs:
                        combined_outputs[var_name] = {}
                    combined_outputs[var_name][subcat_id] = ds[var_name]
                
                ds.close()
                
            except Exception as e:
                self.logger.warning(f"Error loading output for subcatchment {subcat_id}: {str(e)}")
                continue
        
        # Create combined dataset and save
        if combined_outputs:
            self._create_combined_dataset(combined_outputs)

    def _create_combined_dataset(self, combined_outputs):
        """Create a combined dataset from subcatchment outputs"""
        
        try:
            self.logger.info("Creating combined dataset from subcatchment outputs")
            
            if not combined_outputs:
                self.logger.warning("No outputs to combine")
                return
            
            # Get list of subcatchment IDs and variables
            first_var = list(combined_outputs.keys())[0]
            subcatchment_ids = list(combined_outputs[first_var].keys())
            variable_names = list(combined_outputs.keys())
            
            self.logger.info(f"Combining {len(subcatchment_ids)} subcatchments with {len(variable_names)} variables")
            
            # Create the combined dataset
            combined_ds = xr.Dataset()
            
            # Add subcatchment coordinate
            combined_ds.coords['subcatchment'] = ('subcatchment', subcatchment_ids)
            
            # Process each variable
            for var_name in variable_names:
                self.logger.debug(f"Processing variable: {var_name}")
                
                # Collect data arrays for this variable from all subcatchments
                var_arrays = []
                reference_da = None
                
                for subcat_id in subcatchment_ids:
                    if subcat_id in combined_outputs[var_name]:
                        da = combined_outputs[var_name][subcat_id]
                        var_arrays.append(da)
                        if reference_da is None:
                            reference_da = da
                    else:
                        self.logger.warning(f"Missing data for variable {var_name} in subcatchment {subcat_id}")
                
                if var_arrays:
                    try:
                        # Concatenate along new subcatchment dimension
                        combined_var = xr.concat(var_arrays, dim='subcatchment')
                        
                        # Assign subcatchment coordinates
                        combined_var = combined_var.assign_coords(subcatchment=subcatchment_ids)
                        
                        # Copy attributes from reference data array
                        if reference_da is not None:
                            combined_var.attrs = reference_da.attrs.copy()
                        
                        # Add to combined dataset
                        combined_ds[var_name] = combined_var
                        
                        self.logger.debug(f"Combined {var_name} with shape: {combined_var.shape}")
                        
                    except Exception as e:
                        self.logger.error(f"Error combining variable {var_name}: {str(e)}")
                        continue
            
            # Add global attributes
            combined_ds.attrs.update({
                'model': 'FUSE',
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'n_subcatchments': len(subcatchment_ids),
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Combined FUSE distributed simulation results'
            })
            
            # Add subcatchment coordinate attributes
            combined_ds.subcatchment.attrs = {
                'long_name': 'Subcatchment identifier',
                'description': 'Unique identifier for each subcatchment in the distributed model'
            }
            
            # Save the combined dataset
            combined_file = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_distributed_results.nc"
            
            # Define encoding for better compression and compatibility
            encoding = {}
            for var_name in combined_ds.data_vars:
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True,
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }
            
            # Add coordinate encoding
            encoding['subcatchment'] = {'dtype': 'int32'}
            if 'time' in combined_ds.coords:
                encoding['time'] = {'dtype': 'float64'}
            if 'param_set' in combined_ds.coords:
                encoding['param_set'] = {'dtype': 'int32'}
            
            # Save to netCDF
            combined_ds.to_netcdf(
                combined_file,
                encoding=encoding,
                format='NETCDF4'
            )
            
            self.logger.info(f"Combined distributed results saved to: {combined_file}")
            
            # Log summary information
            self.logger.info(f"Combined dataset dimensions: {dict(combined_ds.dims)}")
            self.logger.info(f"Combined dataset variables: {list(combined_ds.data_vars.keys())}")
            
            # Also create a simplified streamflow-only file for easier analysis
            if 'q_routed' in combined_ds.data_vars:
                streamflow_file = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_streamflow_distributed.nc"
                streamflow_ds = combined_ds[['q_routed']].copy()
                streamflow_ds.to_netcdf(streamflow_file, encoding={'q_routed': encoding.get('q_routed', {})})
                self.logger.info(f"Streamflow-only file saved to: {streamflow_file}")
            
            combined_ds.close()
            
        except Exception as e:
            self.logger.error(f"Error creating combined dataset: {str(e)}")
            raise


    def _load_subcatchment_info(self):
        """Load subcatchment information for distributed mode"""
        # Check if delineated catchments exist (for distributed routing)
        delineated_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
        
        if delineated_path.exists():
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs
            catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
            catchment_name = self.config.get('CATCHMENT_SHP_NAME')
            if catchment_name == 'default':
                catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
            
            catchment = gpd.read_file(catchment_path / catchment_name)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.
        Helper method for FUSERunner class.
        """
        try:
            path_value = self.config.get(path_key)
            if path_value == 'default' or path_value is None:
                return self.project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            self.logger.error(f"Config key '{path_key}' not found")
            raise

    def _run_individual_subcatchments(self, subcatchments) -> bool:
        """Run FUSE separately for each subcatchment"""
        
        outputs = []
        
        for i, subcat_id in enumerate(subcatchments):
            self.logger.info(f"Running FUSE for subcatchment {subcat_id} ({i+1}/{len(subcatchments)})")
            
            try:
                # Extract forcing for this subcatchment
                subcat_forcing = self._extract_subcatchment_forcing(subcat_id, i)
                
                # Create subcatchment-specific settings
                subcat_settings = self._create_subcatchment_settings(subcat_id, i)
                
                # Run FUSE for this subcatchment
                subcat_output = self._execute_fuse_subcatchment(subcat_id, subcat_forcing, subcat_settings)
                
                if subcat_output:
                    outputs.append((subcat_id, subcat_output))
                else:
                    self.logger.warning(f"FUSE failed for subcatchment {subcat_id}")
                    
            except Exception as e:
                self.logger.error(f"Error running subcatchment {subcat_id}: {str(e)}")
                continue
        
        if outputs:
            # Combine outputs from all subcatchments
            self._combine_subcatchment_outputs(outputs)
            return True
        else:
            self.logger.error("No successful subcatchment runs")
            return False

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment"""
        
        # Load distributed forcing data
        forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_input.nc"
        ds = xr.open_dataset(forcing_file)
        
        # Extract data for this subcatchment based on coordinate system
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        try:
            if subcatchment_dim == 'latitude':
                # Subcatchment IDs are encoded in latitude dimension
                subcat_data = ds.sel(latitude=float(subcat_id))
            else:
                # Subcatchment IDs are encoded in longitude dimension  
                subcat_data = ds.sel(longitude=float(subcat_id))
            
            # Save subcatchment-specific forcing with the correct filename pattern
            subcat_forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_input.nc"
            subcat_data.to_netcdf(subcat_forcing_file)
            
            ds.close()
            return subcat_forcing_file
            
        except Exception as e:
            self.logger.error(f"Error extracting forcing for subcatchment {subcat_id}: {str(e)}")
            ds.close()
            raise

    def _create_subcatchment_elevation_bands(self, subcat_id: int) -> Path:
        """Create elevation bands file for a specific subcatchment"""
        
        try:
            # Source elevation bands file (the main one created during preprocessing)
            source_elev_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_elev_bands.nc"
            
            # Target elevation bands file for this subcatchment
            target_elev_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_elev_bands.nc"
            
            if source_elev_file.exists():
                # For now, copy the main elevation bands file for each subcatchment
                # In a more sophisticated implementation, you could extract subcatchment-specific elevation data
                shutil.copy2(source_elev_file, target_elev_file)
                self.logger.debug(f"Created elevation bands file for subcatchment {subcat_id}")
            else:
                self.logger.warning(f"Source elevation bands file not found: {source_elev_file}")
                # Create a simple elevation bands file as fallback
                self._create_simple_elevation_bands(target_elev_file, subcat_id)
            
            return target_elev_file
            
        except Exception as e:
            self.logger.error(f"Error creating elevation bands for subcatchment {subcat_id}: {str(e)}")
            raise

    def _create_simple_elevation_bands(self, target_file: Path, subcat_id: int):
        """Create a simple elevation bands file as fallback"""
        
        # Get catchment centroid for coordinates
        catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if catchment_name == 'default':
            catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        
        catchment = gpd.read_file(catchment_path / catchment_name)
        
        # Calculate centroid
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)
        catchment_geo = catchment.to_crs(epsg=4326)
        bounds = catchment_geo.total_bounds
        lon = (bounds[0] + bounds[2]) / 2
        lat = (bounds[1] + bounds[3]) / 2
        
        # Create simple single elevation band
        ds = xr.Dataset(
            coords={
                'longitude': ('longitude', [lon]),
                'latitude': ('latitude', [lat]),
                'elevation_band': ('elevation_band', [1])
            }
        )
        
        # Add variables (single elevation band covering entire subcatchment)
        for var_name, data, attrs in [
            ('area_frac', [1.0], {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
            ('mean_elev', [1000.0], {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
            ('prec_frac', [1.0], {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
        ]:
            ds[var_name] = xr.DataArray(
                np.array(data).reshape(1, 1, 1),
                dims=['elevation_band', 'latitude', 'longitude'],
                coords=ds.coords,
                attrs=attrs
            )
        
        # Add coordinate attributes
        ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'}
        ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
        
        # Save to file
        encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
        ds.to_netcdf(target_file, encoding=encoding)
        
        self.logger.info(f"Created simple elevation bands file for subcatchment {subcat_id}")

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments into distributed format"""
        
        self.logger.info(f"Combining outputs from {len(outputs)} subcatchments")
        
        combined_outputs = {}
        
        # Load and combine all subcatchment outputs
        for subcat_id, output_file in outputs:
            try:
                ds = xr.open_dataset(output_file)
                
                # Store with subcatchment identifier
                for var_name in ds.data_vars:
                    if var_name not in combined_outputs:
                        combined_outputs[var_name] = {}
                    combined_outputs[var_name][subcat_id] = ds[var_name]
                
                ds.close()
                
            except Exception as e:
                self.logger.warning(f"Error loading output for subcatchment {subcat_id}: {str(e)}")
                continue
        
        # Create combined dataset
        self._create_combined_dataset(combined_outputs)

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed FUSE output"""
        
        try:
            self.logger.debug("Starting mizuRoute routing for distributed FUSE")
            
            # Convert FUSE output to mizuRoute input format
            #routing_input = self._convert_fuse_to_mizuroute_format()
            
            #if not routing_input:
            #    return False
            
            # Create FUSE-specific mizuRoute control file
            from utils.models.mizuroute_utils import MizuRoutePreProcessor
            mizu_preprocessor = MizuRoutePreProcessor(self.config, self.logger)
            mizu_preprocessor.create_fuse_control_file()
            
            # Run mizuRoute
            from utils.models.mizuroute_utils import MizuRouteRunner
            mizuroute_runner = MizuRouteRunner(self.config, self.logger)
            
            # Update config for FUSE-mizuRoute integration
            self._setup_fuse_mizuroute_config()
            
            result = mizuroute_runner.run_mizuroute()
            
            if result:
                self.logger.debug("mizuRoute routing completed successfully")
                return True
            else:
                self.logger.error("mizuRoute routing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in distributed routing: {str(e)}")
            return False

    def _convert_fuse_to_mizuroute_format(self) -> bool:
        """
        Convert FUSE distributed output to the mizuRoute input format *in place*
        so it matches what the FUSE-specific mizu control file expects:
        - dims: (time, gru)
        - var:  <routing_var> = config['SETTINGS_MIZU_ROUTING_VAR']
        - id:   gruId (int)
        """
        try:
            # 1) Locate the FUSE output that the control file points to
            #    Control uses: <fname_qsim> DOMAIN_EXPERIMENT_runs_def.nc
            #    Prefer runs_def; fall back to runs_best if needed.
            out_dir = self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "FUSE"
            base = f"{self.domain_name}_{self.config['EXPERIMENT_ID']}"
            candidates = [
                out_dir / f"{base}_runs_def.nc",
                out_dir / f"{base}_runs_best.nc",
            ]
            fuse_output_file = next((p for p in candidates if p.exists()), None)
            if fuse_output_file is None:
                self.logger.error(f"FUSE output file not found. Tried: {candidates}")
                return False

            # 2) Open and convert
            with xr.open_dataset(fuse_output_file) as ds:
                mizu_ds = self._create_mizuroute_forcing_dataset(ds)

            # 3) Overwrite in place so mizuRoute reads exactly what control declares
            #    If the in-use file was runs_best, still write the converted data
            #    back to _runs_def.nc since that's what the control file names.
            write_target = out_dir / f"{base}_runs_def.nc"
            mizu_ds.to_netcdf(write_target, format="NETCDF4")
            self.logger.info(f"Converted FUSE output → mizuRoute format: {write_target}")
            return True

        except Exception as e:
            self.logger.error(f"Error converting FUSE output: {e}")
            return False


    def _create_mizuroute_forcing_dataset(self, fuse_ds: xr.Dataset) -> xr.Dataset:
        """
        Build a mizuRoute-compatible dataset from distributed FUSE output.
        - Detect which spatial coord (latitude/longitude) holds the N>1 groups.
        - Produce dims (time, gru)
        - Add integer gruId from the spatial coordinate values
        - Ensure runoff variable name matches config['SETTINGS_MIZU_ROUTING_VAR']
        """
        # --- Choose runoff variable (prefer q_routed, else sensible fallbacks)
        routing_var_name = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        candidates = [
            'q_routed', 'q_instnt', 'qsim', 'runoff',
            # fallbacks by substring
            *[v for v in fuse_ds.data_vars if v.lower().startswith("q_")],
            *[v for v in fuse_ds.data_vars if "runoff" in v.lower()],
        ]
        runoff_src = next((v for v in candidates if v in fuse_ds.data_vars), None)
        if runoff_src is None:
            raise ValueError(f"No suitable runoff variable found in FUSE output. "
                            f"Available: {list(fuse_ds.data_vars)}")

        # --- Identify spatial axis (one of latitude/longitude must have length > 1)
        lat_len = fuse_ds.dims.get('latitude', 0)
        lon_len = fuse_ds.dims.get('longitude', 0)

        if lat_len > 1 and (lon_len in (0, 1)):
            # (time, latitude, 1)
            data = fuse_ds[runoff_src].squeeze('longitude', drop=True).transpose('time', 'latitude')
            spatial_name = 'latitude'
            ids = fuse_ds[spatial_name].values
        elif lon_len > 1 and (lat_len in (0, 1)):
            # (time, 1, longitude)
            data = fuse_ds[runoff_src].squeeze('latitude', drop=True).transpose('time', 'longitude')
            spatial_name = 'longitude'
            ids = fuse_ds[spatial_name].values
        else:
            # If both >1 (unlikely for your setup) or neither, fail loudly
            raise ValueError(f"Could not infer subcatchment axis from dims: {fuse_ds.dims}")

        # --- Rename spatial dimension to 'gru'
        data = data.rename({data.dims[1]: 'gru'})

        # --- Build output dataset
        mizu = xr.Dataset()
        # copy/forward the time coordinate as-is
        mizu['time'] = fuse_ds['time']
        mizu['time'].attrs.update(fuse_ds['time'].attrs)

        # Add gruId from the spatial coordinate; cast to int32 if possible
        try:
            gid = ids.astype('int32')
        except Exception:
            gid = ids
        mizu['gru'] = xr.DataArray(range(data.sizes['gru']), dims=('gru',))
        mizu['gruId'] = xr.DataArray(gid, dims=('gru',), attrs={
            'long_name': 'ID of grouped response unit', 'units': '-'
        })

        # Ensure variable is named exactly as control expects
        if runoff_src != routing_var_name:
            data = data.rename(routing_var_name)
        mizu[routing_var_name] = data
        # Add/normalize attrs (units default to m/s unless overridden)
        units = self.config.get('SETTINGS_MIZU_ROUTING_UNITS', 'mm/d')
        mizu[routing_var_name].attrs.update({'long_name': 'FUSE runoff for mizuRoute routing',
                                            'units': units})

        # Preserve some useful globals if present
        mizu.attrs.update({k: v for k, v in fuse_ds.attrs.items()})

        return mizu


    def _setup_fuse_mizuroute_config(self):
        """Update configuration for FUSE-mizuRoute integration"""
        
        # Update input file name for mizuRoute
        self.config['EXPERIMENT_ID_TEMP'] = self.config.get('EXPERIMENT_ID')  # Backup
        
        # Set mizuRoute to look for FUSE output instead of SUMMA
        mizuroute_input_file = f"{self.config['EXPERIMENT_ID']}_fuse_runoff.nc"

    def _is_snow_optimization(self) -> bool:
        """Check if this is a snow optimization run by examining the forcing data."""
        try:
            # Check if q_obs contains only dummy values
            forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
            
            if forcing_file.exists():
                with xr.open_dataset(forcing_file) as ds:
                    if 'q_obs' in ds.variables:
                        q_obs_values = ds['q_obs'].values
                        # If all values are -9999 or very close to it, it's dummy data
                        if np.all(np.abs(q_obs_values + 9999) < 0.1):
                            return True
            
            # Also check optimization target from config
            optimization_target = self.config.get('OPTIMISATION_TARGET', 'streamflow')
            if optimization_target in ['swe', 'sca', 'snow_depth', 'snow']:
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not determine if snow optimization: {str(e)}")
            # Fall back to checking config
            optimization_target = self.config.get('OPTIMISATION_TARGET', 'streamflow')
            return optimization_target in ['swe', 'sca', 'snow_depth', 'snow']

    def _copy_default_to_best_params(self):
        """Copy default parameter file to best parameter file for snow optimization."""
        try:
            default_params = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_para_def.nc"
            best_params = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_para_sce.nc"
            
            if default_params.exists():
                import shutil
                shutil.copy2(default_params, best_params)
                self.logger.info("Copied default parameters to best parameters file for snow optimization")
            else:
                self.logger.warning("Default parameter file not found - snow optimization may fail")
                
        except Exception as e:
            self.logger.error(f"Error copying default to best parameters: {str(e)}")
            
    def _get_install_path(self) -> Path:
        """Get the FUSE installation path."""
        fuse_path = self.config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_path == 'default':
            return self.data_dir / 'installs' / 'fuse' / 'bin'
        return Path(fuse_path)

    def _get_output_path(self) -> Path:
        """Get the path for FUSE outputs."""
        if self.config.get('EXPERIMENT_OUTPUT_FUSE', 'default') == 'default':
            return self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        return Path(self.config.get('EXPERIMENT_OUTPUT_FUSE'))

    def _execute_fuse(self, mode, para_file=None) -> bool:
        """
        Execute the FUSE model.
        
        Returns:
            bool: True if execution was successful, False otherwise
        """
        self.logger.debug("Executing FUSE model")
        
        # Construct command
        fuse_fm = self.config['SETTINGS_FUSE_FILEMANAGER']
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'
            
        fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
        control_file = self.project_dir / 'settings' / 'FUSE' / fuse_fm
        
        command = [
            str(fuse_exe),
            str(control_file),
            self.config['DOMAIN_NAME'],
            mode
        ]
            # ADD THIS: Add parameter file for run_pre mode
        if mode == 'run_pre' and para_file:
            command.append(str(para_file))
        
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
            self.logger.debug(f"FUSE execution completed with return code: {result.returncode}")
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed with error: {str(e)}")
            return False

    def _process_outputs(self):
        """Process and organize FUSE output files."""
        self.logger.debug("Processing FUSE outputs")
        
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
                self.logger.debug(f"Processed streamflow output saved to: {processed_file}")
        
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


    def _run_lumped_fuse(self) -> bool:
        """Run FUSE in lumped mode using the original workflow"""
        self.logger.info("Running lumped FUSE workflow")
        
        try:
            # Check if this is a snow optimization case
            if self._is_snow_optimization():
                self.logger.info("Snow optimization detected - copying default to best parameters")
                self._copy_default_to_best_params()
            
            # Run FUSE with default parameters
            success = self._execute_fuse('run_def')
            
            if success:
                # Ensure the expected output file exists
                self._ensure_best_output_file()
                self.logger.debug("Lumped FUSE run completed successfully")
                return True
            else:
                self.logger.error("Lumped FUSE run failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in lumped FUSE execution: {str(e)}")
            return False

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

class FuseDecisionAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "plots" / "FUSE_decision_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.model_decisions_path = self.project_dir / "settings" / "FUSE" / f"fuse_zDecisions_{self.config['EXPERIMENT_ID']}.txt"

        # Initialize FuseRunner
        self.fuse_runner = FUSERunner(config, logger)

        # Get decision options from config or use defaults
        self.decision_options = self._initialize_decision_options()
        
        # Log the decision options being used
        self.logger.info("Initialized FUSE decision options:")
        for decision, options in self.decision_options.items():
            self.logger.info(f"{decision}: {options}")

        # Add storage for simulation results
        self.simulation_results = {}
        self.observed_streamflow = None
        self.area_km2 = None

    def _initialize_decision_options(self) -> Dict[str, List[str]]:
        """
        Initialize decision options from config file or use defaults.
        
        Returns:
            Dict[str, List[str]]: Dictionary of decision options
        """
        # Default decision options as fallback
        default_options = {
            'RFERR': ['additive_e', 'multiplc_e'],
            'ARCH1': ['tension1_1', 'tension2_1', 'onestate_1'],
            'ARCH2': ['tens2pll_2', 'unlimfrc_2', 'unlimpow_2', 'fixedsiz_2'],
            'QSURF': ['arno_x_vic', 'prms_varnt', 'tmdl_param'],
            'QPERC': ['perc_f2sat', 'perc_w2sat', 'perc_lower'],
            'ESOIL': ['sequential', 'rootweight'],
            'QINTF': ['intflwnone', 'intflwsome'],
            'Q_TDH': ['rout_gamma', 'no_routing'],
            'SNOWM': ['temp_index', 'no_snowmod']
        }

        # Try to get decision options from config
        config_options = self.config.get('FUSE_DECISION_OPTIONS')
        
        if config_options:
            self.logger.info("Using decision options from config file")
            
            # Validate config options
            validated_options = {}
            for decision, options in default_options.items():
                if decision in config_options:
                    # Ensure options are in list format
                    config_decision_options = config_options[decision]
                    if isinstance(config_decision_options, list):
                        validated_options[decision] = config_decision_options
                    else:
                        self.logger.warning(
                            f"Invalid options format for decision {decision} in config. "
                            f"Using defaults: {options}"
                        )
                        validated_options[decision] = options
                else:
                    self.logger.warning(
                        f"Decision {decision} not found in config. "
                        f"Using defaults: {options}"
                    )
                    validated_options[decision] = options
            
            return validated_options
        else:
            self.logger.info("No decision options found in config. Using defaults.")
            return default_options

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        """Generate all possible combinations of model decisions."""
        return list(itertools.product(*self.decision_options.values()))

    def update_model_decisions(self, combination: Tuple[str, ...]):
        """
        Update the FUSE model decisions file with a new combination.
        Only updates the decision values (first string) in lines 2-10.
        
        Args:
            combination (Tuple[str, ...]): Tuple of decision values to use
        """
        self.logger.info("Updating FUSE model decisions")
        
        try:
            with open(self.model_decisions_path, 'r') as f:
                lines = f.readlines()
            
            # The decisions are in lines 2-10 (1-based indexing)
            decision_lines = range(1, 10)  # Python uses 0-based indexing
            
            # Create a mapping of decision keys to new values
            decision_keys = list(self.decision_options.keys())
            option_map = dict(zip(decision_keys, combination))
            
            # For debugging
            self.logger.debug(f"Updating with new values: {option_map}")
            
            # Update only the first part of each decision line
            for line_idx in decision_lines:
                # Split the line into components
                line_parts = lines[line_idx].split()
                if len(line_parts) >= 2:
                    # Get the decision key (RFERR, ARCH1, etc.)
                    decision_key = line_parts[1]  # Key is the second part
                    if decision_key in option_map:
                        # Replace the first part with the new value
                        new_value = option_map[decision_key]
                        # Keep the rest of the line (key and any comments) unchanged
                        rest_of_line = ' '.join(line_parts[1:])
                        lines[line_idx] = f"{new_value:<10} {rest_of_line}\n"
                        self.logger.debug(f"Updated line {line_idx + 1}: {lines[line_idx].strip()}")
            
            # Write the updated content back to the file
            with open(self.model_decisions_path, 'w') as f:
                f.writelines(lines)
                
        except Exception as e:
            self.logger.error(f"Error updating model decisions: {str(e)}")
            raise

    def get_current_decisions(self) -> List[str]:
        """Read current decisions from the FUSE decisions file."""
        with open(self.model_decisions_path, 'r') as f:
            lines = f.readlines()
        
        # Extract the first word from lines 2-10
        decisions = []
        for line in lines[1:10]:  # Lines 2-10 in 1-based indexing
            decision = line.strip().split()[0]
            decisions.append(decision)
        
        return decisions

    def plot_hydrographs(self, results_file: Path, metric: str = 'kge'):
        """
        Plot all simulated hydrographs with top 5% highlighted, showing only the overlapping period.
        
        Args:
            results_file (Path): Path to the results CSV file
            metric (str): Metric to use for selecting top performers ('kge', 'nse', etc.)
        """
        self.logger.info(f"Creating hydrograph plot using {metric} metric")
        
        # Read results file
        results_df = pd.read_csv(results_file)
        
        # Calculate threshold for top 5%
        if metric in ['mae', 'rmse']:  # Lower is better
            threshold = results_df[metric].quantile(0.05)
            top_combinations = results_df[results_df[metric] <= threshold]
        else:  # Higher is better
            threshold = results_df[metric].quantile(0.95)
            top_combinations = results_df[results_df[metric] >= threshold]

        # Find overlapping period across all simulations and observations
        start_date = self.observed_streamflow.index.min()
        end_date = self.observed_streamflow.index.max()
        
        for sim in self.simulation_results.values():
            start_date = max(start_date, sim.index.min())
            end_date = min(end_date, sim.index.max())
        
        # Calculate y-axis limit from top 5% simulations
        max_top5 = 0
        for _, row in top_combinations.iterrows():
            combo = tuple(row[list(self.decision_options.keys())])
            if combo in self.simulation_results:
                sim = self.simulation_results[combo]
                sim_overlap = sim.loc[start_date:end_date]
                max_top5 = max(max_top5, sim_overlap.max())

        # Customize plot
        plt.title(f'Hydrograph Comparison ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})\n'
                 f'Top 5% combinations by {metric} metric highlighted', 
                 fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Streamflow (m³/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max_top5 * 1.1)  # Add 10% padding above the maximum value
        
        # Add legend
        plt.plot([], [], color='lightgray', label='All combinations')
        plt.plot([], [], color='blue', alpha=0.3, label=f'Top 5% by {metric}')
        plt.legend(fontsize=10)
        
        # Save plot
        plot_file = self.output_folder / f'hydrograph_comparison_{metric}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Hydrograph plot saved to: {plot_file}")
        
        # Create summary of top combinations
        summary_file = self.output_folder / f'top_combinations_{metric}.csv'
        top_combinations.to_csv(summary_file, index=False)
        self.logger.info(f"Top combinations saved to: {summary_file}")

    def calculate_performance_metrics(self) -> Tuple[float, float, float, float, float]:
        """Calculate performance metrics comparing simulated and observed streamflow."""
        obs_file_path = self.config.get('OBSERVATIONS_PATH')
        if obs_file_path == 'default':
            obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FUSE' / f"{self.config['DOMAIN_NAME']}_{self.config['EXPERIMENT_ID']}_runs_best.nc"

        # Read observations if not already loaded
        if self.observed_streamflow is None:
            dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
            self.observed_streamflow = dfObs['discharge_cms'].resample('d').mean()

        # Read simulations
        dfSim = xr.open_dataset(sim_file_path, decode_timedelta=True)
        dfSim = dfSim['q_routed'].isel(
                                param_set=0,
                                latitude=0,
                                longitude=0
                            )
        dfSim = dfSim.to_pandas()

        # Get area from river basins shapefile using GRU_area if not already calculated
        if self.area_km2 is None:
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Sum the GRU_area column and convert from m2 to km2
            self.area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area from GRU_area: {self.area_km2:.2f} km2")
        
        # Convert units from mm/day to cms
        # Q(cms) = Q(mm/day) * Area(km2) / 86.4
        dfSim = dfSim * self.area_km2 / 86.4

        # Store this simulation result
        current_combo = tuple(self.get_current_decisions())
        self.simulation_results[current_combo] = dfSim

        # Align timestamps and handle missing values
        dfObs = self.observed_streamflow.reindex(dfSim.index).dropna()
        dfSim = dfSim.reindex(dfObs.index).dropna()

        # Calculate metrics
        obs = dfObs.values
        sim = dfSim.values
        
        kge = get_KGE(obs, sim, transfo=1)
        kgep = get_KGEp(obs, sim, transfo=1)
        nse = get_NSE(obs, sim, transfo=1)
        mae = get_MAE(obs, sim, transfo=1)
        rmse = get_RMSE(obs, sim, transfo=1)

        return kge, kgep, nse, mae, rmse

    def run_decision_analysis(self):
        """
        Run the complete FUSE decision analysis workflow, including generating plots and analyzing results.
        
        Returns:
            Tuple[Path, Dict]: Path to results file and dictionary of best combinations
        """
        self.logger.info("Starting FUSE decision analysis")
        
        combinations = self.generate_combinations()
        self.logger.info(f"Generated {len(combinations)} decision combinations")

        optimisation_dir = self.project_dir / 'optimisation'
        optimisation_dir.mkdir(parents=True, exist_ok=True)

        master_file = optimisation_dir / f"{self.config.get('EXPERIMENT_ID')}_fuse_decisions_comparison.csv"

        # Write header to master file
        with open(master_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration'] + list(self.decision_options.keys()) + 
                          ['kge', 'kgep', 'nse', 'mae', 'rmse'])

        for i, combination in enumerate(combinations, 1):
            self.logger.info(f"Running combination {i} of {len(combinations)}")
            self.update_model_decisions(combination)
            
            try:
                # Run FUSE model
                self.fuse_runner.run_fuse()
                
                # Calculate performance metrics
                kge, kgep, nse, mae, rmse = self.calculate_performance_metrics()

                # Write results to master file
                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + [kge, kgep, nse, mae, rmse])

                self.logger.info(f"Combination {i} completed: KGE={kge:.3f}, KGEp={kgep:.3f}, "
                               f"NSE={nse:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

            except Exception as e:
                self.logger.error(f"Error in combination {i}: {str(e)}")
                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + ['erroneous combination'])

        self.logger.info("FUSE decision analysis completed")
        
        # Create hydrograph plots for different metrics
        for metric in ['kge', 'nse', 'kgep']:
            self.plot_hydrographs(master_file, metric)
        
        # Create decision impact plots
        self.plot_decision_impacts(master_file)
        
        # Analyze and save best combinations
        best_combinations = self.analyze_results(master_file)
        
        return master_file, best_combinations

    def plot_decision_impacts(self, results_file: Path):
        """Create plots showing the impact of each decision on model performance."""
        self.logger.info("Plotting FUSE decision impacts")
        
        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = list(self.decision_options.keys())

        for metric in metrics:
            plt.figure(figsize=(12, 6 * len(decisions)))
            for i, decision in enumerate(decisions, 1):
                plt.subplot(len(decisions), 1, i)
                impact = df.groupby(decision)[metric].mean().sort_values(ascending=False)
                impact.plot(kind='bar')
                plt.title(f'Impact of {decision} on {metric}')
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_folder / f'{metric}_decision_impacts.png')
            plt.close()

        self.logger.info("Decision impact plots saved")

    def analyze_results(self, results_file: Path) -> Dict[str, Dict]:
        """Analyze the results and identify the best performing combinations."""
        self.logger.info("Analyzing FUSE decision results")
        
        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = list(self.decision_options.keys())

        best_combinations = {}
        for metric in metrics:
            if metric in ['mae', 'rmse']:  # Lower is better
                best_row = df.loc[df[metric].idxmin()]
            else:  # Higher is better
                best_row = df.loc[df[metric].idxmax()]
            
            best_combinations[metric] = {
                'score': best_row[metric],
                'combination': {decision: best_row[decision] for decision in decisions}
            }

        # Save results to file
        output_file = self.project_dir / 'optimisation' / 'best_fuse_decision_combinations.txt'
        with open(output_file, 'w') as f:
            for metric, data in best_combinations.items():
                f.write(f"Best combination for {metric} (score: {data['score']:.3f}):\n")
                for decision, value in data['combination'].items():
                    f.write(f"  {decision}: {value}\n")
                f.write("\n")

        self.logger.info("FUSE decision analysis results saved")
        return best_combinations

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
        

class FUSEPostprocessor:
    """
    Postprocessor for FUSE (Framework for Understanding Structural Errors) model outputs.
    Handles extraction, processing, and saving of simulation results.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from FUSE output and save to CSV.
        Converts units from mm/day to m3/s (cms) using catchment area.
        
        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FUSE streamflow results")
            
            # Define paths
            sim_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'FUSE' / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_runs_best.nc"
            
            # Read simulation results
            ds = xr.open_dataset(sim_path)
            
            # Extract streamflow (selecting first parameter set and first grid cell)
            q_sim = ds['q_routed'].isel(
                param_set=0,
                latitude=0,
                longitude=0
            )
            
            # Convert to pandas Series
            q_sim = q_sim.to_pandas()
            
            # Get catchment area from river basins shapefile
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Calculate total area in km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
            
            # Convert units from mm/day to m3/s (cms)
            # Q(cms) = Q(mm/day) * Area(km2) / 86.4
            q_sim_cms = q_sim * area_km2 / 86.4
            
            # Create DataFrame with FUSE-prefixed column name
            results_df = pd.DataFrame({
                'FUSE_discharge_cms': q_sim_cms
            }, index=q_sim.index)
            
            # Add metadata as attributes
            results_df.attrs = {
                'model': 'FUSE',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'catchment_area_km2': area_km2,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'units': 'm3/s'
            }
            
            # Save to CSV
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
            results_df.to_csv(output_file)
            
            self.logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise
            
    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))