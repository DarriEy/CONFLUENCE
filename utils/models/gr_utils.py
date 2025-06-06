from typing import Dict, Any, Optional
from pathlib import Path
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import rpy2.robjects as robjects # type: ignore
from rpy2.robjects.packages import importr # type: ignore
import rasterio # type: ignore
from rpy2.robjects import pandas2ri # type: ignore
from rpy2.robjects.conversion import localconverter # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data.variable_utils import VariableHandler # type: ignore

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
        
        # Calculate PET using Oudin's formula
        # PET = Ra * (T + 5) / 100 if T + 5 > 0, else 0
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
            # As a fallback, use a simplified temperature-based approach
            if pet_max == 0:
                self.logger.warning("Using fallback simplified PET estimation based on temperature")
                # Simple temperature-based fallback (Hamon-like)
                month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
                # Adjust for leap years - simplistic approach
                month_days = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) if dates[0].year % 4 == 0 else month_days
                
                # Create month index (0-11)
                month_idx = np.array(dates.month) - 1
                
                # Simplified temperature-based pet (positive temperature values only)
                simple_pet = np.maximum(temp_C.values * 0.15, 0)
                
                # Create new DataArray with the simple pet values
                pet = xr.DataArray(
                    simple_pet,
                    coords=temp_C.coords,
                    dims=temp_C.dims,
                    attrs={
                        'units': 'mm/day',
                        'long_name': 'Simplified Potential evapotranspiration (temperature-based)',
                        'standard_name': 'water_potential_evaporation_flux'
                    }
                )
                self.logger.info(f"Fallback PET range: {float(pet.min()):.4f} to {float(pet.max()):.4f} mm/day")
        
        return pet

    def prepare_forcing_data(self):
        try:
            # Read and process forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found in basin-averaged data directory")
            
            # Debug print the forcing files found
            #self.logger.info(f"Found forcing files: {[f.name for f in forcing_files]}")
            
            # Open and concatenate all forcing files
            ds = xr.open_mfdataset(forcing_files)
            variable_handler = VariableHandler(config=self.config, logger=self.logger, dataset=self.config['FORCING_DATASET'], model='GR')
            # Average across HRUs if needed
            ds = ds.mean(dim='hru')
            #self.logger.info(f'before variable handler {ds}')
            ds_variable_handler = variable_handler.process_forcing_data(ds)
            #self.logger.info(f'after variable handler {ds_variable_handler}')
            # Convert forcing data to daily resolution
            ds = ds_variable_handler
            ds = ds.resample(time='D').mean()
            try:
                ds['temp'] = ds['airtemp'] - 273.15
                ds['pr'] = ds['pptrate'] * 86400
            except:
                pass
                    
            # Load streamflow observations
            obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            
            # Read observations - explicitly rename 'datetime' column to 'time'
            obs_df = pd.read_csv(obs_path)
            obs_df['time'] = pd.to_datetime(obs_df['datetime'])
            obs_df = obs_df.drop('datetime', axis=1)
            obs_df.set_index('time', inplace=True)
            obs_df.index = obs_df.index.tz_localize(None)
            # Ensure the time column is properly formatted
            # Convert to daily resolution
            obs_daily = obs_df.resample('D').mean()


            # Get area from river basins shapefile using GRU_area
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{'DOMAIN_DEFINITION_METHOD'}.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Sum the GRU_area column and convert from m2 to km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area from GRU_area: {area_km2:.2f} km2")
            
            # Convert units from cms to mm/day 
            obs_daily['discharge_mmday'] = obs_daily['discharge_cms'] / area_km2 * 86.4
            
            # Create observation dataset with explicit time dimension
            obs_ds = xr.Dataset(
                {'q_obs': ('time', obs_daily['discharge_mmday'].values)},
                coords={'time': obs_daily.index.values}
            )

            # Read catchment and get centroid
            catchment = gpd.read_file(self.catchment_path / self.catchment_name)
            mean_lon, mean_lat = self._get_catchment_centroid(catchment)
            
            # Calculate PET using Oudin formula
            pet = self.calculate_pet_oudin(ds['temp'], mean_lat)

            # Find overlapping time period
            start_time = max(ds.time.min().values, obs_ds.time.min().values)
            end_time = min(ds.time.max().values, obs_ds.time.max().values)
            
            # Create explicit time index
            time_index = pd.date_range(start=start_time, end=end_time, freq='D')
            
            # Select the common time period and align to the new time index
            ds = ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            obs_ds = obs_ds.sel(time=slice(start_time, end_time)).reindex(time=time_index)
            pet = pet.sel(time=slice(start_time, end_time)).reindex(time=time_index)

            # Create GR forcing data with correct dimensions
            ds_coords = {
                'longitude': [mean_lon],
                'latitude': [mean_lat],
                'time': time_index
            }
            
            # Create the dataset with dimensions first
            gr_forcing = xr.Dataset(
                coords={
                    'longitude': ('longitude', ds_coords['longitude']),
                    'latitude': ('latitude', ds_coords['latitude']),
                    'time': ('time', ds_coords['time'])
                }
            )

            # Add coordinate attributes (without _FillValue)
            gr_forcing.longitude.attrs = {
                'units': 'degreesE',
                'long_name': 'longitude'
            }
            gr_forcing.latitude.attrs = {
                'units': 'degreesN',
                'long_name': 'latitude'
            }
            gr_forcing.time.attrs = {
                'units': 'date',
                'long_name': 'time'
            }

            # Prepare data variables
            var_mapping = [
                ('pr', ds['pr'].values, 'precipitation', 'mm/day', 'Mean daily precipitation'),
                ('temp', ds['temp'].values, 'temperature', 'degC', 'Mean daily temperature'),
                ('pet', pet.values, 'pet', 'mm/day', 'Mean daily pet'),
                ('q_obs', obs_ds['q_obs'].values, 'streamflow', 'mm/day', 'Mean observed daily discharge')
            ]

            encoding = {}
            
            for var_name, data, _, units, long_name in var_mapping:
                if np.any(np.isnan(data)):
                    data = np.nan_to_num(data, nan=-9999.0)
                
                gr_forcing[var_name] = xr.DataArray(
                    data.reshape(-1, 1, 1),
                    dims=['time', 'latitude', 'longitude'],
                    coords=gr_forcing.coords,
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
            output_file = self.forcing_gr_path / f"{self.domain_name}_input.csv"
            
            # Convert to dataframe
            gr_forcing = gr_forcing.to_dataframe()
            
            # The time is now in the index, not as a column
            # Reset the index to make time a column
            gr_forcing = gr_forcing.reset_index()
            
            # Now we can reformat the time column
            gr_forcing['time'] = pd.to_datetime(gr_forcing['time']).dt.strftime('%Y-%m-%d')
            
            # Select only the columns we need
            gr_forcing = gr_forcing[['time', 'pr', 'temp', 'pet', 'q_obs']]
            
            # Save to CSV
            gr_forcing.to_csv(output_file, index=False)
            
            return output_file

        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
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
            
            # Execute GR model
            if self.spatial_mode == 'lumped':
                self._execute_gr_lumped()
            else:  # semi-distributed
                self._execute_gr_distributed()
                
        except Exception as e:
            self.logger.error(f"Error during GR run: {str(e)}")
            raise

    def _execute_gr_lumped(self):
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
                # Remove no data values
                masked_dem = masked_dem[masked_dem != src.nodata]
                
                # Calculate percentiles (hypsometric curve)
                Hypso = np.percentile(masked_dem, np.arange(0, 101, 1))

                # Calculate mean elevation
                Zmean = np.mean(masked_dem)

            time_start = pd.to_datetime(self.config['EXPERIMENT_TIME_START'])
            time_end = pd.to_datetime(self.config['EXPERIMENT_TIME_END'])
            
            # Format dates to ensure consistency
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
                
                # Print first few rows for debugging
                cat("First few rows of BasinObs:\\n")
                print(head(BasinObs))
                
                # Convert time column to POSIXct format
                BasinObs$time_posix <- as.POSIXct(BasinObs$time)
                
                # Print date range
                cat("Data time range:", format(min(BasinObs$time_posix), "%Y-%m-%d"), "to", 
                    format(max(BasinObs$time_posix), "%Y-%m-%d"), "\\n")
                
                # Create a safer date matching function
                find_date_indices <- function(start_date, end_date, date_vector) {{
                    # Convert to Date objects
                    date_vector_as_date <- as.Date(date_vector)
                    start_date_as_date <- as.Date(start_date)
                    end_date_as_date <- as.Date(end_date)
                    
                    # Find indices where dates are within range
                    indices <- which(date_vector_as_date >= start_date_as_date & 
                                    date_vector_as_date <= end_date_as_date)
                    
                    if (length(indices) < 1) {{
                        cat("WARNING: No dates found in range", start_date, "to", end_date, "\\n")
                        return(NULL)
                    }}
                    
                    return(indices)
                }}
                
                # Determine periods
                spinup_dates <- "{spinup_start}:{spinup_end}"
                calib_dates <- "{calib_start}:{calib_end}"
                run_dates <- "{run_start}:{run_end}"
                
                cat("Looking for spinup dates:", spinup_dates, "\\n")
                cat("Looking for calibration dates:", calib_dates, "\\n")
                cat("Looking for run dates:", run_dates, "\\n")
                
                # Extract date components from the string (YYYY-MM-DD format)
                parse_date_range <- function(date_range) {{
                    parts <- strsplit(date_range, ":")[[1]]
                    list(start = parts[1], end = parts[2])
                }}
                
                spinup_range <- parse_date_range(spinup_dates)
                calib_range <- parse_date_range(calib_dates)
                run_range <- parse_date_range(run_dates)
                
                # Find indices for each period
                date_vector <- format(as.Date(BasinObs$time), "%Y-%m-%d")
                
                Ind_Warm <- find_date_indices(spinup_range$start, spinup_range$end, date_vector)
                if (is.null(Ind_Warm)) {{
                    # Fallback: use first year of data
                    first_year <- format(as.Date(BasinObs$time[1]), "%Y")
                    cat("Using fallback spinup: first year of data (", first_year, ")\\n")
                    Ind_Warm <- which(format(as.Date(BasinObs$time), "%Y") == first_year)
                }}
                
                Ind_Cal <- find_date_indices(calib_range$start, calib_range$end, date_vector)
                if (is.null(Ind_Cal)) {{
                    # Fallback: use second year if available, otherwise first 365 days
                    all_years <- unique(format(as.Date(BasinObs$time), "%Y"))
                    if (length(all_years) > 1) {{
                        second_year <- all_years[2]
                        cat("Using fallback calibration: second year of data (", second_year, ")\\n")
                        Ind_Cal <- which(format(as.Date(BasinObs$time), "%Y") == second_year)
                    }} else {{
                        cat("Using fallback calibration: first 365 days\\n")
                        Ind_Cal <- seq(1, min(365, length(BasinObs$time)))
                    }}
                }}
                
                Ind_Run <- find_date_indices(run_range$start, run_range$end, date_vector)
                if (is.null(Ind_Run)) {{
                    # Fallback: use all available data
                    cat("Using fallback run period: all available data\\n")
                    Ind_Run <- seq(1, length(BasinObs$time))
                }}
                
                # Print selected periods
                cat("Selected spinup period:", 
                    format(as.Date(BasinObs$time[Ind_Warm[1]]), "%Y-%m-%d"), "to",
                    format(as.Date(BasinObs$time[Ind_Warm[length(Ind_Warm)]]), "%Y-%m-%d"), 
                    "(", length(Ind_Warm), "days )\\n")
                    
                cat("Selected calibration period:", 
                    format(as.Date(BasinObs$time[Ind_Cal[1]]), "%Y-%m-%d"), "to",
                    format(as.Date(BasinObs$time[Ind_Cal[length(Ind_Cal)]]), "%Y-%m-%d"), 
                    "(", length(Ind_Cal), "days )\\n")
                    
                cat("Selected run period:", 
                    format(as.Date(BasinObs$time[Ind_Run[1]]), "%Y-%m-%d"), "to",
                    format(as.Date(BasinObs$time[Ind_Run[length(Ind_Run)]]), "%Y-%m-%d"), 
                    "(", length(Ind_Run), "days )\\n")
                
                # Create elevation data for CemaNeige
                
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
                
                # Calibration criterion: preparation of the InputsCrit object
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

                save(OutputsCalib, file = "{str(self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR' / 'GR_calib.Rdata')}")
                
                # Simulation
                Param <- OutputsCalib$ParamFinalR
                
                # Preparation of RunOptions object for full simulation
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
                
                # Create plots directory if it doesn't exist
                dir.create(dirname("{str(self.project_dir / 'plots' / 'results')}"), recursive = TRUE, showWarnings = FALSE)
                
                # Results preview
                png("{str(self.project_dir / 'plots' / 'results' / 'GRhydrology_plot.png')}", height = 900, width = 900)
                plot(OutputsModel, Qobs = BasinObs$q_obs[Ind_Run])
                dev.off()
                
                # Save the OutputsModel for potential further analysis
                save(OutputsModel, file = "{str(self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR' / 'GR_results.Rdata')}") 
                
                # Return success message
                "GR model execution completed successfully"
            '''
            
            self.logger.info(f"r_script:{r_script}")

            # Execute the R script
            result = robjects.r(r_script)
            self.logger.info("R script executed successfully!")
            return result
                
        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            return None
    
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
        Extract simulated streamflow from GR output and append to results CSV.
        Converts units from mm/day to m3/s (cms).
        """
        try:
            self.logger.info("Extracting GR streamflow results")
            
            # Check for R data file
            r_results_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'GR' / 'GR_results.Rdata'
            if not r_results_path.exists():
                self.logger.error(f"GR results file not found at: {r_results_path}")
                return None

            # Load R data
            robjects.r(f'load("{str(r_results_path)}")')
            
            # Extract simulated streamflow from OutputsModel
            # Use both DatesR and Qsim directly from the loaded data
            r_script = """
            data.frame(
                date = format(OutputsModel$DatesR, "%Y-%m-%d"),
                flow = OutputsModel$Qsim
            )
            """
            
            # Run R script
            sim_df = robjects.r(r_script)
            
            # Convert to pandas dataframe
            with localconverter(robjects.default_converter + pandas2ri.converter):
                sim_df = robjects.conversion.rpy2py(sim_df)
                
            # Set index to datetime
            sim_df['date'] = pd.to_datetime(sim_df['date'])
            sim_df.set_index('date', inplace=True)
            
            # Get catchment area
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
            q_sim_cms = sim_df['flow'] * area_km2 / 86.4
            
            # Read existing results file if it exists
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
            
        except Exception as e:
            self.logger.error(f"Error extracting GR streamflow: {str(e)}")
            raise
            
    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))