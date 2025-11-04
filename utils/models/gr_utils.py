from typing import Dict, Any, Optional
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
import rasterio
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
import shutil
from utils.data.variable_utils import VariableHandler

class GRPreProcessor:
    """
    Preprocessor for the GR family of models (initially GR4J).
    Handles data preparation, PET calculation, snow module setup, and file organization.
    Now supports both lumped and distributed spatial modes.
    
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
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        
        # Spatial mode configuration
        self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')

    def run_preprocessing(self):
        """Run the complete GR preprocessing workflow."""
        self.logger.info(f"Starting GR preprocessing in {self.spatial_mode} mode")
        try:
            self.create_directories()
            self.prepare_forcing_data()
            self.logger.info("GR preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during GR preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for GR model setup."""
        dirs_to_create = [
            self.gr_setup_dir,
            self.forcing_gr_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def calculate_pet_oudin(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate potential evapotranspiration using Oudin's formula.
        Optimized implementation with proper temperature handling.
        
        Args:
            temp_data (xr.DataArray): Temperature data in either Kelvin or Celsius
            lat (float): Latitude of the catchment centroid in degrees
            
        Returns:
            xr.DataArray: Calculated PET in mm/day
        """
        self.logger.info("Calculating PET using Oudin's formula (optimized)")
        
        # Check if temperature is likely in Kelvin (values > 100) and convert to Celsius if needed
        temp_max = float(temp_data.max())
        if temp_max > 100:
            self.logger.info(f"Converting temperature from Kelvin to Celsius (max value: {temp_max})")
            temp_C = temp_data - 273.15
        else:
            self.logger.info(f"Temperature appears to be in Celsius already (max value: {temp_max})")
            temp_C = temp_data
        
        # Debug temperature values
        self.logger.info(f"Temperature range: {float(temp_C.min()):.2f}°C to {float(temp_C.max()):.2f}°C")
        
        # Get dates
        dates = pd.DatetimeIndex(temp_data.time.values)
        
        # Convert latitude to radians
        lat_rad = np.deg2rad(lat)
        
        # Calculate day of year as array
        doy = np.array(dates.dayofyear)
        
        # Calculate solar declination (radians)
        solar_decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
        
        # Calculate sunset hour angle (radians)
        # Handle potential numerical issues with clipping
        cos_arg = -np.tan(lat_rad) * np.tan(solar_decl)
        # Clip to valid range for arccos
        cos_arg = np.clip(cos_arg, -1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        
        # Calculate extraterrestrial radiation (Ra) - MJ/m²/day
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
        Ra = (24 * 60 / np.pi) * 0.082 * dr * (
            sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
            np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)
        )
        
        # Debug Ra values
        self.logger.info(f"Extraterrestrial radiation range: {Ra.min():.2f} to {Ra.max():.2f} MJ/m²/day")
        
        # Broadcast Ra if needed for multi-HRU
        if 'hru' in temp_C.dims:
            Ra = Ra.broadcast_like(temp_C)
        
        # Oudin's formula: PET = Ra * (T + 5) / 100 when T + 5 > 0
        # Create a temperature adjusted array
        temp_adj = temp_C.values + 5
        
        # Apply the formula with vectorized operations
        pet_values = np.where(temp_adj > 0, Ra * temp_adj / 100, 0)
        
        # Create a new DataArray with the same dimensions and coordinates
        pet = xr.DataArray(
            pet_values,
            coords=temp_C.coords,
            dims=temp_C.dims,
            attrs={
                'units': 'mm/day',
                'long_name': 'Potential evapotranspiration (Oudin formula)',
                'standard_name': 'water_potential_evaporation_flux'
            }
        )
        
        # Check if PET has reasonable values
        pet_min, pet_max = float(pet.min()), float(pet.max())
        self.logger.info(f"PET range: {pet_min:.4f} to {pet_max:.4f} mm/day")
        
        if pet_max < 0.001:
            self.logger.warning("PET values are all near zero! This may indicate an issue with the calculation.")
        
        return pet

    def prepare_forcing_data(self):
        """
        Prepare forcing data with support for lumped and distributed modes.
        """
        try:
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")
            
            # Open and concatenate all forcing files
            ds = xr.open_mfdataset(forcing_files)
            variable_handler = VariableHandler(config=self.config, logger=self.logger, 
                                            dataset=self.config['FORCING_DATASET'], model='GR')
            
            # Process variables
            ds_variable_handler = variable_handler.process_forcing_data(ds)
            ds = ds_variable_handler
            
            # Handle spatial organization based on mode
            if self.spatial_mode == 'lumped':
                self.logger.info("Preparing lumped forcing data")
                ds = ds.mean(dim='hru') if 'hru' in ds.dims else ds
                return self._prepare_lumped_forcing(ds)
            elif self.spatial_mode == 'distributed':
                self.logger.info("Preparing distributed forcing data")
                return self._prepare_distributed_forcing(ds)
            else:
                raise ValueError(f"Unknown GR spatial mode: {self.spatial_mode}")
                
        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    def _prepare_lumped_forcing(self, ds):
        """Prepare lumped forcing data (existing implementation)"""
        # Convert forcing data to daily resolution
        ds = ds.resample(time='D').mean()
        
        try:
            ds['temp'] = ds['airtemp'] - 273.15
            ds['pr'] = ds['pptrate'] * 86400
        except:
            pass
        
        # Load streamflow observations
        obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        
        # Read observations
        obs_df = pd.read_csv(obs_path)
        obs_df['time'] = pd.to_datetime(obs_df['datetime'])
        obs_df = obs_df.drop('datetime', axis=1)
        obs_df.set_index('time', inplace=True)
        obs_df.index = obs_df.index.tz_localize(None)
        obs_daily = obs_df.resample('D').mean()
        
        # Get area from river basins shapefile
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default':
            basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
        basin_gdf = gpd.read_file(basin_path)
        
        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area from GRU_area: {area_km2:.2f} km2")
        
        # Convert units from cms to mm/day
        obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * 86.4
        
        # Create observation dataset
        obs_ds = xr.Dataset(
            {'q_obs': ('time', obs_daily['discharge_mmday'].values)},
            coords={'time': obs_daily.index.values}
        )
        
        # Read catchment and get centroid
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        mean_lon, mean_lat = self._get_catchment_centroid(catchment)
        
        # Calculate PET
        pet = self.calculate_pet_oudin(ds['temp'], mean_lat)
        
        # Find overlapping time period
        start_time = max(ds.time.min().values, obs_ds.time.min().values)
        end_time = min(ds.time.max().values, obs_ds.time.max().values)
        
        # Create explicit time index
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        
        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        # Create GR forcing data
        gr_forcing = pd.DataFrame({
            'time': time_index.strftime('%Y-%m-%d'),
            'pr': ds['pr'].values,
            'temp': ds['temp'].values,
            'pet': pet.values,
            'q_obs': obs_ds['q_obs'].values
        })
        
        # Save to CSV
        output_file = self.forcing_gr_path / f"{self.domain_name}_input.csv"
        gr_forcing.to_csv(output_file, index=False)
        
        self.logger.info(f"Lumped forcing data saved to: {output_file}")
        return output_file

    def _prepare_distributed_forcing(self, ds):
        """Prepare distributed forcing data for each HRU"""
        
        # Load catchment to get HRU information
        catchment = gpd.read_file(self.catchment_path / self.catchment_name)
        
        # Check if we have HRU dimension in forcing data
        if 'hru' not in ds.dims:
            self.logger.warning("No HRU dimension found in forcing data, creating distributed data from lumped")
            # Replicate lumped data to all HRUs
            n_hrus = len(catchment)
            ds = ds.expand_dims(hru=n_hrus)
        
        # Convert to daily resolution
        ds = ds.resample(time='D').mean()
        
        try:
            ds['temp'] = ds['airtemp'] - 273.15
            ds['pr'] = ds['pptrate'] * 86400
        except:
            pass
        
        # Load streamflow observations (at outlet)
        obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        
        if obs_path.exists():
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'])
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_df.index = obs_df.index.tz_localize(None)
            obs_daily = obs_df.resample('D').mean()
            
            # Get area for unit conversion
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * 86.4
        else:
            self.logger.warning("No streamflow observations found")
            obs_daily = None
        
        # Calculate PET for each HRU using its centroid latitude
        self.logger.info("Calculating PET for each HRU")
        
        # Ensure catchment has proper CRS
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)
        catchment_geo = catchment.to_crs(epsg=4326)
        
        # Get centroids for each HRU
        hru_centroids = catchment_geo.geometry.centroid
        hru_lats = hru_centroids.y.values
        
        # Calculate PET for each HRU
        pet_data = []
        for i, lat in enumerate(hru_lats):
            temp_hru = ds['temp'].isel(hru=i)
            pet_hru = self.calculate_pet_oudin(temp_hru, lat)
            pet_data.append(pet_hru.values)
        
        # Stack PET data
        pet_array = np.stack(pet_data, axis=1)  # shape: (time, hru)
        pet = xr.DataArray(
            pet_array,
            dims=['time', 'hru'],
            coords={'time': ds.time, 'hru': ds.hru},
            attrs={
                'units': 'mm/day',
                'long_name': 'Potential evapotranspiration (Oudin formula)',
                'standard_name': 'water_potential_evaporation_flux'
            }
        )
        
        # Find overlapping time period
        start_time = ds.time.min().values
        end_time = ds.time.max().values
        
        if obs_daily is not None:
            start_time = max(start_time, obs_daily.index.min())
            end_time = min(end_time, obs_daily.index.max())
        
        time_index = pd.date_range(start=start_time, end=end_time, freq='D')
        
        # Select and align data
        ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)
        
        if obs_daily is not None:
            obs_daily = obs_daily.reindex(time_index)
        
        # Save distributed forcing as NetCDF (one file with all HRUs)
        output_file = self.forcing_gr_path / f"{self.domain_name}_input_distributed.nc"
        
        # Create output dataset
        gr_forcing = xr.Dataset({
            'pr': ds['pr'],
            'temp': ds['temp'],
            'pet': pet
        })
        
        if obs_daily is not None:
            gr_forcing['q_obs'] = xr.DataArray(
                obs_daily['discharge_mmday'].values,
                dims=['time'],
                coords={'time': time_index}
            )
        
        # Add HRU metadata
        gr_forcing['hru_id'] = xr.DataArray(
            catchment['GRU_ID'].values if 'GRU_ID' in catchment.columns else np.arange(len(catchment)),
            dims=['hru'],
            attrs={'long_name': 'HRU identifier'}
        )
        
        gr_forcing['hru_lat'] = xr.DataArray(
            hru_lats,
            dims=['hru'],
            attrs={'long_name': 'HRU centroid latitude', 'units': 'degrees_north'}
        )
        
        # Save to NetCDF
        encoding = {var: {'zlib': True, 'complevel': 4} for var in gr_forcing.data_vars}
        gr_forcing.to_netcdf(output_file, encoding=encoding)
        
        self.logger.info(f"Distributed forcing data saved to: {output_file}")
        self.logger.info(f"Number of HRUs: {len(ds.hru)}")
        
        return output_file

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


class GRRunner:
    """
    Runner class for the GR family of models (initially GR4J).
    Handles model execution, state management, and output processing.
    Now supports both lumped and distributed spatial modes.
    
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
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        
        # GR-specific paths
        self.gr_setup_dir = self.project_dir / "settings" / "GR"
        self.forcing_gr_path = self.project_dir / 'forcing' / 'GR_input'
        self.output_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')
        self.needs_routing = self._check_routing_requirements()

    def run_gr(self) -> Optional[Path]:
        """
        Run the GR model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info(f"Starting GR model run in {self.spatial_mode} mode")
        
        try:
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Execute GR model
            if self.spatial_mode == 'lumped':
                success = self._execute_gr_lumped()
            else:  # distributed
                success = self._execute_gr_distributed()
            
            if success and self.needs_routing:
                self.logger.info("Running distributed routing with mizuRoute")
                success = self._run_distributed_routing()
            
            if success:
                self.logger.info("GR model run completed successfully")
                return self.output_path
            else:
                self.logger.error("GR model run failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during GR run: {str(e)}")
            raise

    def _check_routing_requirements(self) -> bool:
        """Check if distributed routing is needed"""
        routing_integration = self.config.get('GR_ROUTING_INTEGRATION', 'none')
        
        if routing_integration == 'mizuRoute':
            if self.spatial_mode == 'distributed':
                return True
        
        return False

    def _execute_gr_distributed(self) -> bool:
        """Execute GR4J in distributed mode - run for each HRU"""
        self.logger.info("Running distributed GR4J workflow")
        
        try:
            # Initialize R environment
            base = importr('base')
            
            # Install airGR if not already installed
            robjects.r('''
                if (!require("airGR")) {
                    install.packages("airGR", repos="https://cloud.r-project.org")
                }
            ''')
            
            # Load forcing data
            forcing_file = self.forcing_gr_path / f"{self.domain_name}_input_distributed.nc"
            ds = xr.open_dataset(forcing_file)
            
            n_hrus = len(ds.hru)
            self.logger.info(f"Running GR4J for {n_hrus} HRUs")
            
            # Load DEM for hypsometric curve (use catchment-wide for now)
            dem_path = self.project_dir / 'attributes' / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            
            with rasterio.open(dem_path) as src:
                out_image, out_transform = rasterio.mask.mask(src, catchment.geometry, crop=True)
                masked_dem = out_image[0]
                masked_dem = masked_dem[masked_dem != src.nodata]
                Hypso = np.percentile(masked_dem, np.arange(0, 101, 1))
                Zmean = np.mean(masked_dem)
            
            # Get simulation periods
            time_start = pd.to_datetime(self.config['EXPERIMENT_TIME_START'])
            time_end = pd.to_datetime(self.config['EXPERIMENT_TIME_END'])
            spinup_start = pd.to_datetime(self.config['SPINUP_PERIOD'].split(',')[0].strip()).strftime('%Y-%m-%d')
            spinup_end = pd.to_datetime(self.config['SPINUP_PERIOD'].split(',')[1].strip()).strftime('%Y-%m-%d')
            run_start = time_start.strftime('%Y-%m-%d')
            run_end = time_end.strftime('%Y-%m-%d')
            
            # Store results for each HRU
            hru_results = []
            
            # Loop through each HRU
            for hru_idx in range(n_hrus):
                hru_id = int(ds.hru_id.values[hru_idx]) if 'hru_id' in ds else hru_idx + 1
                self.logger.info(f"Processing HRU {hru_id} ({hru_idx + 1}/{n_hrus})")
                
                # Extract data for this HRU
                hru_data = ds.isel(hru=hru_idx)
                
                # Create temporary DataFrame for this HRU
                hru_df = pd.DataFrame({
                    'time': pd.to_datetime(hru_data.time.values).strftime('%Y-%m-%d'),
                    'pr': hru_data['pr'].values,
                    'temp': hru_data['temp'].values,
                    'pet': hru_data['pet'].values
                })
                
                # Save temporary CSV for R
                temp_csv = self.forcing_gr_path / f"hru_{hru_id}_temp.csv"
                hru_df.to_csv(temp_csv, index=False)
                
                # Run GR4J for this HRU
                r_script = f'''
                    library(airGR)
                    
                    # Load HRU data
                    BasinObs <- read.csv("{str(temp_csv)}")
                    
                    # Preparation of InputsModel
                    InputsModel <- CreateInputsModel(
                        FUN_MOD = RunModel_CemaNeigeGR4J,
                        DatesR = as.POSIXct(BasinObs$time),
                        Precip = BasinObs$pr,
                        PotEvap = BasinObs$pet,
                        TempMean = BasinObs$temp,
                        HypsoData = c({', '.join(map(str, Hypso))}),
                        ZInputs = {Zmean}
                    )
                    
                    # Parse dates
                    date_vector <- format(as.Date(BasinObs$time), "%Y-%m-%d")
                    
                    # Find indices
                    Ind_Warm <- which(date_vector >= "{spinup_start}" & date_vector <= "{spinup_end}")
                    Ind_Run <- which(date_vector >= "{run_start}" & date_vector <= "{run_end}")
                    
                    # Use default parameters (no calibration yet)
                    # GR4J parameters: X1, X2, X3, X4
                    # CemaNeige parameters: CTG, Kf
                    Param <- c(257.24, 1.012, 88.23, 2.208, 0.0, 3.69)  # Default parameter set
                    
                    # Preparation of RunOptions
                    RunOptions <- CreateRunOptions(
                        FUN_MOD = RunModel_CemaNeigeGR4J,
                        InputsModel = InputsModel,
                        IndPeriod_WarmUp = Ind_Warm,
                        IndPeriod_Run = Ind_Run,
                        IsHyst = TRUE
                    )
                    
                    # Run model
                    OutputsModel <- RunModel_CemaNeigeGR4J(
                        InputsModel = InputsModel,
                        RunOptions = RunOptions,
                        Param = Param
                    )
                    
                    # Extract results
                    data.frame(
                        date = format(OutputsModel$DatesR, "%Y-%m-%d"),
                        q_routed = OutputsModel$Qsim
                    )
                '''
                
                # Execute R script
                result_df = robjects.r(r_script)
                
                # Convert to pandas
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    result_df = robjects.conversion.rpy2py(result_df)
                
                result_df['date'] = pd.to_datetime(result_df['date'])
                result_df['hru_id'] = hru_id
                
                hru_results.append(result_df)
                
                # Clean up temporary file
                temp_csv.unlink()
            
            # Combine all HRU results
            self.logger.info("Combining results from all HRUs")
            combined_results = pd.concat(hru_results, ignore_index=True)
            
            # Pivot to get time x HRU structure
            results_pivot = combined_results.pivot(index='date', columns='hru_id', values='q_routed')
            
            # Convert to xarray Dataset (mizuRoute format)
            self._save_distributed_results_for_routing(results_pivot, ds)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in distributed GR4J execution: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _save_distributed_results_for_routing(self, results_df, forcing_ds):
        """
        Save distributed GR4J results in mizuRoute-compatible format.
        Mimics FUSE output structure.
        """
        self.logger.info("Saving distributed results in mizuRoute format")
        
        # Create time coordinate (days since 1970-01-01)
        time_values = results_df.index
        time_days = (time_values - pd.Timestamp('1970-01-01')).days.values
        
        # Get HRU IDs from columns
        hru_ids = results_df.columns.values.astype(int)
        n_hrus = len(hru_ids)
        
        # Create xarray Dataset with mizuRoute structure
        # Dimensions: (time, gru) - matching what mizuRoute expects
        ds_out = xr.Dataset(
            coords={
                'time': ('time', time_days),
                'gru': ('gru', np.arange(n_hrus))
            }
        )
        
        # Add gruId variable
        ds_out['gruId'] = xr.DataArray(
            hru_ids,
            dims=('gru',),
            attrs={
                'long_name': 'ID of grouped response unit',
                'units': '-'
            }
        )
        
        # Add streamflow data (in mm/day as GR4J outputs)
        routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        ds_out[routing_var] = xr.DataArray(
            results_df.values,
            dims=('time', 'gru'),
            attrs={
                'long_name': 'GR4J runoff for mizuRoute routing',
                'units': 'mm/d',
                'description': 'Runoff from distributed GR4J model'
            }
        )
        
        # Add time attributes
        ds_out.time.attrs = {
            'units': 'days since 1970-01-01',
            'calendar': 'standard',
            'long_name': 'time'
        }
        
        # Add global attributes
        ds_out.attrs = {
            'model': 'GR4J-CemaNeige',
            'spatial_mode': 'distributed',
            'domain': self.domain_name,
            'experiment_id': self.config['EXPERIMENT_ID'],
            'n_hrus': n_hrus,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Distributed GR4J simulation results for mizuRoute routing'
        }
        
        # Save to NetCDF
        output_file = self.output_path / f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_runs_def.nc"
        
        encoding = {
            'time': {'dtype': 'float64'},
            'gru': {'dtype': 'int32'},
            'gruId': {'dtype': 'int32'},
            routing_var: {'dtype': 'float32', 'zlib': True, 'complevel': 4}
        }
        
        ds_out.to_netcdf(output_file, encoding=encoding, format='NETCDF4')
        
        self.logger.info(f"Distributed results saved to: {output_file}")
        self.logger.info(f"Output dimensions: time={len(time_days)}, gru={n_hrus}")

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed GR4J output"""
        
        try:
            self.logger.info("Starting mizuRoute routing for distributed GR4J")
            
            # Create GR-specific mizuRoute control file if needed
            from utils.models.mizuroute_utils import MizuRoutePreProcessor
            mizu_preprocessor = MizuRoutePreProcessor(self.config, self.logger)
            
            # Check if we need to create a GR-specific control file
            control_file = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
            if control_file == 'default':
                control_file = 'mizuRoute_control_GR.txt'
                self.config['SETTINGS_MIZU_CONTROL_FILE'] = control_file
                mizu_preprocessor.create_gr_control_file()
            
            # Run mizuRoute
            from utils.models.mizuroute_utils import MizuRouteRunner
            mizuroute_runner = MizuRouteRunner(self.config, self.logger)
            
            # Update config for GR-mizuRoute integration
            self._setup_gr_mizuroute_config()
            
            result = mizuroute_runner.run_mizuroute()
            
            if result:
                self.logger.info("mizuRoute routing completed successfully")
                return True
            else:
                self.logger.error("mizuRoute routing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in distributed routing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _setup_gr_mizuroute_config(self):
        """Update configuration for GR-mizuRoute integration"""
        
        # Set mizuRoute to look for GR output instead of SUMMA
        self.config['MIZU_FROM_MODEL'] = 'GR'

    def _execute_gr_lumped(self):
        """Execute GR4J in lumped mode (existing implementation)"""
        try:
            # Initialize R environment
            base = importr('base')
            
            # Read DEM
            dem_path = self.project_dir / 'attributes' / 'elevation' / 'dem' / f"domain_{self.domain_name}_elv.tif"
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
                transform = src.transform

            # Read catchment and get centroid
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)

            # Mask DEM with catchment boundary
            with rasterio.open(dem_path) as src:
                out_image, out_transform = rasterio.mask.mask(src, catchment.geometry, crop=True)
                masked_dem = out_image[0]
                masked_dem = masked_dem[masked_dem != src.nodata]
                Hypso = np.percentile(masked_dem, np.arange(0, 101, 1))
                Zmean = np.mean(masked_dem)

            time_start = pd.to_datetime(self.config['EXPERIMENT_TIME_START'])
            time_end = pd.to_datetime(self.config['EXPERIMENT_TIME_END'])
            
            spinup_start = pd.to_datetime(self.config['SPINUP_PERIOD'].split(',')[0].strip()).strftime('%Y-%m-%d')
            spinup_end = pd.to_datetime(self.config['SPINUP_PERIOD'].split(',')[1].strip()).strftime('%Y-%m-%d')
            calib_start = pd.to_datetime(self.config['CALIBRATION_PERIOD'].split(',')[0].strip()).strftime('%Y-%m-%d')
            calib_end = pd.to_datetime(self.config['CALIBRATION_PERIOD'].split(',')[1].strip()).strftime('%Y-%m-%d')
            run_start = time_start.strftime('%Y-%m-%d')
            run_end = time_end.strftime('%Y-%m-%d')
            
            self.logger.info(f"Spinup period: {spinup_start} to {spinup_end}")
            self.logger.info(f"Calibration period: {calib_start} to {calib_end}")
            self.logger.info(f"Run period: {run_start} to {run_end}")

            # Install airGR if not already installed
            robjects.r('''
                if (!require("airGR")) {
                    install.packages("airGR", repos="https://cloud.r-project.org")
                }
            ''')
            
            # R script as a string with improved date handling
            r_script = f'''
                library(airGR)
                
                # Loading catchment data
                BasinObs <- read.csv("{str(self.forcing_gr_path / f"{self.domain_name}_input.csv")}")
                
                # Convert time column to POSIXct format
                BasinObs$time_posix <- as.POSIXct(BasinObs$time)
                
                # Create a safer date matching function
                find_date_indices <- function(start_date, end_date, date_vector) {{
                    date_vector_as_date <- as.Date(date_vector)
                    start_date_as_date <- as.Date(start_date)
                    end_date_as_date <- as.Date(end_date)
                    
                    indices <- which(date_vector_as_date >= start_date_as_date & 
                                    date_vector_as_date <= end_date_as_date)
                    
                    if (length(indices) < 1) {{
                        cat("WARNING: No dates found in range", start_date, "to", end_date, "\\n")
                        return(NULL)
                    }}
                    
                    return(indices)
                }}
                
                # Determine periods
                date_vector <- format(as.Date(BasinObs$time), "%Y-%m-%d")
                
                Ind_Warm <- find_date_indices("{spinup_start}", "{spinup_end}", date_vector)
                Ind_Cal <- find_date_indices("{calib_start}", "{calib_end}", date_vector)
                Ind_Run <- find_date_indices("{run_start}", "{run_end}", date_vector)
                
                # Preparation of InputsModel object
                InputsModel <- CreateInputsModel(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    DatesR = as.POSIXct(BasinObs$time),
                    Precip = BasinObs$pr,
                    PotEvap = BasinObs$pet,
                    TempMean = BasinObs$temp,
                    HypsoData = c({', '.join(map(str, Hypso))}),
                    ZInputs = {Zmean}
                )
                
                # Preparation of RunOptions object
                RunOptions <- CreateRunOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    InputsModel = InputsModel,
                    IndPeriod_WarmUp = Ind_Warm,
                    IndPeriod_Run = Ind_Cal,
                    IsHyst = TRUE
                )
                
                # Calibration criterion
                InputsCrit <- CreateInputsCrit(
                    FUN_CRIT = ErrorCrit_{self.config['OPTIMIZATION_METRIC']},
                    InputsModel = InputsModel,
                    RunOptions = RunOptions,
                    Obs = BasinObs$q_obs[Ind_Cal]
                )
                
                # Preparation of CalibOptions object
                CalibOptions <- CreateCalibOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    FUN_CALIB = Calibration_Michel,
                    IsHyst = TRUE
                )
                
                # Calibration
                OutputsCalib <- Calibration_Michel(
                    InputsModel = InputsModel,
                    RunOptions = RunOptions,
                    InputsCrit = InputsCrit,
                    CalibOptions = CalibOptions,
                    FUN_MOD = RunModel_CemaNeigeGR4J
                )

                save(OutputsCalib, file = "{str(self.output_path / 'GR_calib.Rdata')}")
                
                # Simulation
                Param <- OutputsCalib$ParamFinalR
                
                # Preparation of RunOptions for full simulation
                RunOptions <- CreateRunOptions(
                    FUN_MOD = RunModel_CemaNeigeGR4J,
                    InputsModel = InputsModel,
                    IndPeriod_Run = Ind_Run,
                    IsHyst = TRUE
                )

                OutputsModel <- RunModel_CemaNeigeGR4J(
                    InputsModel = InputsModel,
                    RunOptions = RunOptions,
                    Param = Param
                )
                
                # Create plots directory
                dir.create(dirname("{str(self.project_dir / 'plots' / 'results')}"), recursive = TRUE, showWarnings = FALSE)
                
                # Results preview
                png("{str(self.project_dir / 'plots' / 'results' / 'GRhydrology_plot.png')}", height = 900, width = 900)
                plot(OutputsModel, Qobs = BasinObs$q_obs[Ind_Run])
                dev.off()
                
                # Save results
                save(OutputsModel, file = "{str(self.output_path / 'GR_results.Rdata')}") 
                
                "GR model execution completed successfully"
            '''

            # Execute the R script
            result = robjects.r(r_script)
            self.logger.info("R script executed successfully!")
            return True
                
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """Get path from config or use default based on project directory."""
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)


class GRPostprocessor:
    """
    Postprocessor for GR (GR4J/CemaNeige) model outputs.
    Handles extraction and processing of simulation results.
    Supports both lumped and distributed modes.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from GR output and append to results CSV.
        Handles both lumped and distributed modes.
        """
        try:
            self.logger.info(f"Extracting GR streamflow results ({self.spatial_mode} mode)")
            
            if self.spatial_mode == 'lumped':
                return self._extract_lumped_streamflow()
            else:  # distributed
                return self._extract_distributed_streamflow()
                
        except Exception as e:
            self.logger.error(f"Error extracting GR streamflow: {str(e)}")
            raise

    def _extract_lumped_streamflow(self) -> Optional[Path]:
        """Extract streamflow from lumped GR4J run"""
        
        # Check for R data file
        r_results_path = self.output_path / 'GR_results.Rdata'
        if not r_results_path.exists():
            self.logger.error(f"GR results file not found at: {r_results_path}")
            return None

        # Load R data
        robjects.r(f'load("{str(r_results_path)}")')
        
        # Extract simulated streamflow
        r_script = """
        data.frame(
            date = format(OutputsModel$DatesR, "%Y-%m-%d"),
            flow = OutputsModel$Qsim
        )
        """
        
        sim_df = robjects.r(r_script)
        
        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            sim_df = robjects.conversion.rpy2py(sim_df)
        
        sim_df['date'] = pd.to_datetime(sim_df['date'])
        sim_df.set_index('date', inplace=True)
        
        # Get catchment area
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default':
            basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
        basin_gdf = gpd.read_file(basin_path)
        
        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
        
        # Convert units from mm/day to m3/s (cms)
        q_sim_cms = sim_df['flow'] * area_km2 / 86.4
        
        # Read existing results or create new
        output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
        if output_file.exists():
            results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        else:
            results_df = pd.DataFrame(index=q_sim_cms.index)
        
        # Add GR results
        results_df['GR_discharge_cms'] = q_sim_cms
        
        # Save updated results
        results_df.to_csv(output_file)
        
        self.logger.info(f"GR results appended to: {output_file}")
        return output_file

    def _extract_distributed_streamflow(self) -> Optional[Path]:
        """Extract streamflow from distributed GR4J run (after routing)"""
        
        # Check if routing was performed
        needs_routing = self.config.get('GR_ROUTING_INTEGRATION') == 'mizuRoute'
        
        if needs_routing:
            # Get routed streamflow from mizuRoute output
            mizuroute_output_dir = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'mizuRoute'
            
            # Find mizuRoute output file
            output_files = list(mizuroute_output_dir.glob(f"{self.config['EXPERIMENT_ID']}*.nc"))
            
            if not output_files:
                self.logger.error(f"No mizuRoute output files found in {mizuroute_output_dir}")
                return None
            
            # Use the first output file
            mizuroute_file = output_files[0]
            self.logger.info(f"Reading routed streamflow from: {mizuroute_file}")
            
            ds = xr.open_dataset(mizuroute_file)
            
            # Extract streamflow at outlet (typically the last reach)
            # mizuRoute typically names the variable 'IRFroutedRunoff' or similar
            streamflow_vars = ['IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff']
            streamflow_var = None
            
            for var in streamflow_vars:
                if var in ds.variables:
                    streamflow_var = var
                    break
            
            if streamflow_var is None:
                self.logger.error(f"Could not find streamflow variable in mizuRoute output. Available: {list(ds.variables)}")
                return None
            
            # Get streamflow at outlet (last segment)
            q_routed = ds[streamflow_var].isel(seg=-1)
            
            # Convert to DataFrame
            q_df = q_routed.to_dataframe(name='flow')
            q_df = q_df.reset_index()
            
            # Convert time if needed
            if 'time' in q_df.columns:
                q_df['time'] = pd.to_datetime(q_df['time'])
                q_df.set_index('time', inplace=True)
            
        else:
            # No routing - sum all HRU outputs
            gr_output = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR' / \
                        f"{self.domain_name}_{self.config['EXPERIMENT_ID']}_runs_def.nc"
            
            if not gr_output.exists():
                self.logger.error(f"GR output not found: {gr_output}")
                return None
            
            ds = xr.open_dataset(gr_output)
            
            # Sum across all GRUs
            routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            q_total = ds[routing_var].sum(dim='gru')
            
            # Convert to DataFrame
            q_df = q_total.to_dataframe(name='flow')
        
        # Convert from mm/day to m3/s
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default':
            basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
        basin_gdf = gpd.read_file(basin_path)
        
        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
        
        # Convert units
        q_cms = q_df['flow'] * area_km2 / 86.4
        
        # Save to results
        output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
        if output_file.exists():
            results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        else:
            results_df = pd.DataFrame(index=q_cms.index)
        
        results_df['GR_discharge_cms'] = q_cms
        results_df.to_csv(output_file)
        
        self.logger.info(f"Distributed GR results appended to: {output_file}")
        return output_file
            
    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))
    
    @property
    def output_path(self):
        """Get output path for backwards compatibility"""
        return self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR'
