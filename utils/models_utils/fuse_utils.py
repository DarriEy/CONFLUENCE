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
            self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
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
        try:
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")
            
            # Debug print the forcing files found
            self.logger.info(f"Found forcing files: {[f.name for f in forcing_files]}")
            
            # Open and concatenate all forcing files
            ds = xr.open_mfdataset(forcing_files)
            
            # Average across HRUs if needed
            ds = ds.mean(dim='hru')
            
            # Convert forcing data to daily resolution
            ds = ds.resample(time='D').mean()
            
            # Load streamflow observations
            obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            
            # Read observations - explicitly rename 'datetime' column to 'time'
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'])
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_df.index = obs_df.index.tz_localize(None)
            
            # Convert to daily resolution
            obs_daily = obs_df.resample('D').mean()
            
            # Create observation dataset with explicit time dimension
            obs_ds = xr.Dataset(
                {'q_obs': ('time', obs_daily['discharge_cms'].values)},
                coords={'time': obs_daily.index.values}
            )

            # Read catchment and get centroid
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self._get_catchment_centroid(catchment)
            
            # Calculate PET using Oudin formula
            pet = self.calculate_pet_oudin(ds['airtemp'], mean_lat)

            # Find overlapping time period
            start_time = max(ds.time.min().values, obs_ds.time.min().values)
            end_time = min(ds.time.max().values, obs_ds.time.max().values)
            
            # Create explicit time index
            time_index = pd.date_range(start=start_time, end=end_time, freq='D')
            
            # Select the common time period and align to the new time index
            ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

            # Convert time to days since 1970-01-01
            time_days = (time_index - pd.Timestamp('1970-01-01')).days.values

            # Create FUSE forcing data with correct dimensions
            ds_coords = {
                'longitude': [mean_lon],
                'latitude': [mean_lat],
                'time': time_days
            }
            
            # Create the dataset with dimensions first
            fuse_forcing = xr.Dataset(
                coords={
                    'longitude': ('longitude', ds_coords['longitude']),
                    'latitude': ('latitude', ds_coords['latitude']),
                    'time': ('time', ds_coords['time'])
                }
            )

            # Add coordinate attributes (without _FillValue)
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

            # Prepare data variables
            var_mapping = [
                ('pr', ds['pptrate'].values * 86400, 'precipitation', 'mm/day', 'Mean daily precipitation'),
                ('temp', ds['airtemp'].values - 273.15, 'temperature', 'degC', 'Mean daily temperature'),
                ('pet', pet.values, 'pet', 'mm/day', 'Mean daily pet'),
                ('q_obs', obs_ds['q_obs'].values, 'streamflow', 'mm/day', 'Mean observed daily discharge')
            ]

            encoding = {}
            
            for var_name, data, _, units, long_name in var_mapping:
                if np.any(np.isnan(data)):
                    data = np.nan_to_num(data, nan=-9999.0)
                
                fuse_forcing[var_name] = xr.DataArray(
                    data.reshape(-1, 1, 1),
                    dims=['time', 'latitude', 'longitude'],
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

            # Add dimension encoding
            encoding.update({
                'longitude': {'dtype': 'float64'},
                'latitude': {'dtype': 'float64'},
                'time': {'dtype': 'float64'}
            })

            # Save forcing data
            output_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"
            fuse_forcing.to_netcdf(
                output_file, 
                unlimited_dims=['time'],
                encoding=encoding,
                format='NETCDF4'
            )

            return output_file

        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise
        
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
        
        date_settings = {
            'date_start_sim': start_time.strftime('%Y-%m-%d'),
            'date_end_sim': end_time.strftime('%Y-%m-%d'),
            'date_start_eval': start_time.strftime('%Y-%m-%d'),  # Using same dates for evaluation period
            'date_end_eval': end_time.strftime('%Y-%m-%d')       # Can be modified if needed
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
        """
        Create elevation bands netCDF file for FUSE input.
        Uses DEM to calculate elevation distribution and area fractions.
        """
        self.logger.info("Creating elevation bands file for FUSE")

        try:
            # Read DEM
            dem_path = self.project_dir / 'attributes' / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
                transform = src.transform

            # Read catchment and get centroid
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            lon, lat = self._get_catchment_centroid(catchment)

            # Mask DEM with catchment boundary
            with rasterio.open(dem_path) as src:
                out_image, out_transform = rasterio.mask.mask(src, catchment.geometry, crop=True)
                masked_dem = out_image[0]
                masked_dem = masked_dem[masked_dem != src.nodata]

            # Calculate elevation bands
            min_elev = np.floor(np.min(masked_dem))
            max_elev = np.ceil(np.max(masked_dem))
            band_size = 100  # 100m elevation bands
            num_bands = int((max_elev - min_elev) / band_size) + 1

            # Calculate area fractions and mean elevations
            area_fracs = []
            mean_elevs = []
            total_pixels = len(masked_dem)

            for i in range(num_bands):
                lower = min_elev + i * band_size
                upper = lower + band_size
                band_pixels = np.sum((masked_dem >= lower) & (masked_dem < upper))
                area_frac = band_pixels / total_pixels
                mean_elev = lower + band_size/2

                if area_frac > 0:  # Only include bands that have area
                    area_fracs.append(area_frac)
                    mean_elevs.append(mean_elev)

            # Normalize area fractions to sum to 1
            area_fracs = np.array(area_fracs) / np.sum(area_fracs)
            mean_elevs = np.array(mean_elevs)

            # Create netCDF file
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            
            # Create dataset with dimensions first
            ds = xr.Dataset(
                coords={
                    'longitude': ('longitude', [lon]),
                    'latitude': ('latitude', [lat]),
                    'elevation_band': ('elevation_band', range(1, len(area_fracs) + 1))
                }
            )
            
            # Add coordinate attributes (without _FillValue)
            ds.longitude.attrs = {
                'units': 'degreesE',
                'long_name': 'longitude'
            }
            ds.latitude.attrs = {
                'units': 'degreesN',
                'long_name': 'latitude'
            }
            ds.elevation_band.attrs = {
                'units': '-',
                'long_name': 'elevation_band'
            }

            # Define variables and their attributes
            variables = {
                'area_frac': {
                    'data': area_fracs,
                    'attrs': {
                        'units': '-',
                        'long_name': 'Fraction of the catchment covered by each elevation band'
                    }
                },
                'mean_elev': {
                    'data': mean_elevs,
                    'attrs': {
                        'units': 'm asl',
                        'long_name': 'Mid-point elevation of each elevation band'
                    }
                },
                'prec_frac': {
                    'data': area_fracs,  # Same as area_frac for now
                    'attrs': {
                        'units': '-',
                        'long_name': 'Fraction of catchment precipitation that falls on each elevation band - same as area_frac'
                    }
                }
            }

            # Create encoding dictionary
            encoding = {
                'longitude': {'dtype': 'float64'},
                'latitude': {'dtype': 'float64'},
                'elevation_band': {'dtype': 'int32'}
            }

            # Add variables to dataset
            for var_name, var_info in variables.items():
                ds[var_name] = xr.DataArray(
                    var_info['data'].reshape(-1, 1, 1),
                    dims=['elevation_band', 'latitude', 'longitude'],
                    coords=ds.coords,
                    attrs=var_info['attrs']
                )
                encoding[var_name] = {
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }

            # Save to netCDF file
            output_file = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            ds.to_netcdf(
                output_file,
                encoding=encoding,
                format='NETCDF4'
            )

            return output_file

        except Exception as e:
            self.logger.error(f"Error creating elevation bands file: {str(e)}")
            raise

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
            
            # Execute FUSE for default
            success = self._execute_fuse('run_def')
            
            # Calibrate FUSE 
            success = self._execute_fuse('calib_sce')

            # Excute FUSE for best parameters
            success = self._execute_fuse('run_best')

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

    def _execute_fuse(self, mode) -> bool:
        """
        Execute the FUSE model.
        
        Returns:
            bool: True if execution was successful, False otherwise
        """
        self.logger.info("Executing FUSE model")
        
        # Construct command
        fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
        control_file = self.project_dir / 'settings' / 'FUSE' / self.config['SETTINGS_FUSE_FILEMANAGER']
        
        command = [
            str(fuse_exe),
            str(control_file),
            self.config['DOMAIN_NAME'],
            mode
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