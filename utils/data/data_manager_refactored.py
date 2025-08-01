# In utils/dataHandling_utils/data_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import calendar
import xarray as xr
import geopandas as gpd
from rasterstats import zonal_stats # type: ignore
import rasterio # type: ignore
from rasterio.features import rasterize # type: ignore # Added for optimization
import numpy as np
import pandas as pd 
import multiprocessing as mp
import time
from utils.data.data_utils import ObservedDataProcessor, gistoolRunner, datatoolRunner # type: ignore 
from utils.data.agnosticPreProcessor import forcingResampler, geospatialStatistics # type: ignore 
from utils.data.variable_utils import VariableHandler # type: ignore 
from utils.geospatial.raster_utils import calculate_landcover_mode # type: ignore 
from utils.data.archive_utils import tar_directory # type: ignore

class DataManager:
    """
    Manages all data acquisition and preprocessing operations for CONFLUENCE.
    
    The DataManager is responsible for acquiring, processing, and preparing all the 
    data required for hydrological modeling. This includes geospatial attributes 
    (elevation, land cover, soil types), meteorological forcings, and observed 
    streamflow data. It coordinates with external data acquisition tools and ensures 
    that data is properly formatted for use by the modeling components.
    
    Key responsibilities:
    - Acquiring geospatial attributes (DEM, soil, land cover)
    - Downloading meteorological forcing data
    - Processing observed streamflow data
    - Performing model-agnostic preprocessing (basin averaging, resampling)
    - Supplementing forcing data with EM-Earth precipitation and temperature
    - Validating data availability and integrity
    - Managing input/output variable transformations
    
    The DataManager acts as a bridge between external data sources and the 
    CONFLUENCE modeling system, ensuring that all required data is available
    in the correct format before model execution begins.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        data_dir (Path): Path to the CONFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        variable_handler (VariableHandler): Handler for variable transformations
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Data Manager.
        
        Sets up the DataManager with the provided configuration and logger.
        This establishes the basic paths, identifiers, and handlers needed
        for data acquisition and processing operations.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            logger (logging.Logger): Logger instance for recording operations
            
        Raises:
            KeyError: If essential configuration values are missing
            ValueError: If configuration contains invalid values
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Initialize variable handler
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        
        # Set default optimization configuration if not present
        if 'EM_EARTH_OPTIMIZATION' not in self.config:
            self.config['EM_EARTH_OPTIMIZATION'] = {
                'USE_OPTIMIZED_REMAPPING': True,
                'USE_FLOAT32': True,
                'LOG_PERFORMANCE': True,
                'FALLBACK_ON_ERROR': True,
                'DETAILED_LOGGING': False
            }
        
    def acquire_attributes(self):
        """
        Acquire geospatial attributes including DEM, soil, and land cover data.
        
        This method coordinates the acquisition of essential geospatial attributes
        required for hydrological modeling:
        1. Digital Elevation Model (DEM) data from MERIT-Hydro
        2. Land cover data from MODIS
        3. Soil classification data from SoilGrids
        
        The data is acquired using the gistool utility, which handles downloading,
        preprocessing, and georeferencing. The method creates the necessary directory
        structure and configures the acquisition parameters based on the domain's
        bounding box.
        
        For land cover, if multiple years are specified, the method calculates the
        mode (most common value) across years to create a single representative layer.
        
        Raises:
            FileNotFoundError: If external data sources cannot be accessed
            ValueError: If coordinate bounds are invalid
            Exception: For other errors during data acquisition
        """
        self.logger.info("Starting attribute acquisition")
        
        # Create attribute directories
        dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        soilclass_dir = self.project_dir / 'attributes' / 'soilclass'
        landclass_dir = self.project_dir / 'attributes' / 'landclass'
        
        for dir_path in [dem_dir, soilclass_dir, landclass_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the gistool runner
        gr = gistoolRunner(self.config, self.logger)
        
        # Get lat and lon limits from bounding box
        bbox = self.config['BOUNDING_BOX_COORDS'].split('/')
        latlims = f"{bbox[0]},{bbox[2]}"
        lonlims = f"{bbox[1]},{bbox[3]}"
        
        try:
            # Acquire elevation data
            self._acquire_elevation_data(gr, dem_dir, latlims, lonlims)
            
            # Acquire land cover data
            self._acquire_landcover_data(gr, landclass_dir, latlims, lonlims)
            
            # Acquire soil class data
            self._acquire_soilclass_data(gr, soilclass_dir, latlims, lonlims)
            
            self.logger.info("Attribute acquisition completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during attribute acquisition: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _acquire_elevation_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        """
        Acquire elevation data using gistool.
        
        Downloads Digital Elevation Model (DEM) data from the MERIT-Hydro dataset,
        which provides global coverage of topography at a high resolution. The data
        is clipped to the domain's bounding box and saved in the specified output
        directory.
        
        Args:
            gistool_runner: Instance of gistoolRunner for executing commands
            output_dir (Path): Directory where elevation data will be saved
            lat_lims (str): Latitude limits in format "min,max"
            lon_lims (str): Longitude limits in format "min,max"
            
        Raises:
            Exception: If data acquisition fails
        """
        self.logger.info("Acquiring elevation data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='MERIT-Hydro',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='elv'
        )
        gistool_runner.execute_gistool_command(gistool_command)

    def _acquire_landcover_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        """
        Acquire land cover data using gistool.
        
        Downloads land cover classification data from the MODIS MCD12Q1.006 dataset,
        which provides global coverage of land cover types at 500m resolution. The data
        is acquired for the specified time range, clipped to the domain's bounding box,
        and saved in the output directory.
        
        If multiple years are specified, the method calculates the mode (most common
        value) across years to create a single representative land cover layer.
        
        Args:
            gistool_runner: Instance of gistoolRunner for executing commands
            output_dir (Path): Directory where land cover data will be saved
            lat_lims (str): Latitude limits in format "min,max"
            lon_lims (str): Longitude limits in format "min,max"
            
        Raises:
            Exception: If data acquisition or processing fails
        """
        self.logger.info("Acquiring land cover data")
        
        # Define time range for land cover data
        start_year = 2001
        end_year = 2020
        
        # Select MODIS dataset
        modis_var = "MCD12Q1.006"
        
        # Create gistool command
        gistool_command = gistool_runner.create_gistool_command(
            dataset='MODIS',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables=modis_var,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-01-01"
        )
        gistool_runner.execute_gistool_command(gistool_command)
        
        # Calculate mode if multiple years selected
        land_name = self.config.get('LAND_CLASS_NAME', 'default')
        if land_name == 'default':
            land_name = f"domain_{self.domain_name}_land_classes.tif"
        
        if start_year != end_year:
            input_dir = output_dir / modis_var
            output_file = output_dir / land_name
            
            self.logger.info("Calculating land cover mode across years")
            calculate_landcover_mode(input_dir, output_file, start_year, end_year, self.domain_name)
    
    def _acquire_soilclass_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        """
        Acquire soil class data using gistool.
        
        Downloads soil classification data derived from the SoilGrids dataset,
        which provides global soil information. The data is clipped to the domain's
        bounding box and saved in the specified output directory.
        
        Args:
            gistool_runner: Instance of gistoolRunner for executing commands
            output_dir (Path): Directory where soil class data will be saved
            lat_lims (str): Latitude limits in format "min,max"
            lon_lims (str): Longitude limits in format "min,max"
            
        Raises:
            Exception: If data acquisition fails
        """
        self.logger.info("Acquiring soil class data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='soil_class',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='soil_classes'
        )
        gistool_runner.execute_gistool_command(gistool_command)
    
    def acquire_forcings(self):
        """
        Acquire forcing data for the model simulation.
        
        This method downloads meteorological forcing data required for hydrological
        modeling, such as precipitation, temperature, humidity, wind, and radiation.
        The data is acquired using the datatool utility, which handles downloading
        and initial preprocessing.
        
        The method supports both point-scale and distributed simulations. For 
        point-scale simulations with user-supplied data, the acquisition is skipped.
        
        The forcing dataset and variables are specified in the configuration. If the
        variables are set to 'default', the method uses the VariableHandler to
        determine the appropriate variables for the specified dataset and model.
        
        The data is clipped to the domain's bounding box and the specified time period,
        and saved in the project's forcing directory.
        
        If SUPPLEMENT_FORCING is enabled, also acquires EM-Earth precipitation and
        temperature data for supplementing the primary forcing dataset.
        
        Returns:
            None: If data is user-supplied for point simulations
            
        Raises:
            FileNotFoundError: If external data sources cannot be accessed
            ValueError: If coordinate bounds or time period are invalid
            Exception: For other errors during data acquisition
        """
        self.logger.info("Starting forcing data acquisition")
                
        # Initialize datatool runner
        dr = datatoolRunner(self.config, self.logger)
        
        # Create output directory
        raw_data_dir = self.project_dir / 'forcing' / 'raw_data'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get lat and lon limits
        bbox = self.config['BOUNDING_BOX_COORDS'].split('/')
        latlims = f"{bbox[2]},{bbox[0]}"
        lonlims = f"{bbox[1]},{bbox[3]}"
        
        # Get variables to download
        variables = self.config.get('FORCING_VARIABLES', 'default')
        if variables == 'default':
            variables = self.variable_handler.get_dataset_variables(dataset=self.config['FORCING_DATASET'])
        
        try:
            # Create and execute datatool command
            datatool_command = dr.create_datatool_command(
                dataset=self.config['FORCING_DATASET'],
                output_dir=raw_data_dir,
                lat_lims=latlims,
                lon_lims=lonlims,
                variables=variables,
                start_date=self.config['EXPERIMENT_TIME_START'],
                end_date=self.config['EXPERIMENT_TIME_END']
            )
            dr.execute_datatool_command(datatool_command)
            
            self.logger.info("Primary forcing data acquisition completed successfully")
            
            # Acquire EM-Earth data if supplementation is enabled
            if self.config.get('SUPPLEMENT_FORCING', False):
                self.logger.info("SUPPLEMENT_FORCING enabled - acquiring EM-Earth data")
                self.acquire_em_earth_forcings()
            
        except Exception as e:
            self.logger.error(f"Error during forcing data acquisition: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def acquire_em_earth_forcings(self):
        """
        Acquire EM-Earth precipitation and temperature data for supplementing primary forcing.
        
        This method downloads and processes EM-Earth data to supplement the primary forcing
        dataset with higher quality precipitation and temperature observations. The EM-Earth
        data is clipped to the domain's bounding box and time period, then stored in a 
        separate directory for later integration during preprocessing.
        """
        self.logger.info("Starting EM-Earth forcing data acquisition")
        
        try:
            # Create output directory for EM-Earth data
            em_earth_dir = self.project_dir / 'forcing' / 'raw_data_em_earth'
            em_earth_dir.mkdir(parents=True, exist_ok=True)
            
            # Get EM-Earth data paths from configuration
            em_region = self.config.get('EM_EARTH_REGION', 'NorthAmerica')
            em_earth_prcp_dir = self.config.get('EM_EARTH_PRCP_DIR', 
                f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/prcp/{em_region}")
            em_earth_tmean_dir = self.config.get('EM_EARTH_TMEAN_DIR',
                f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/tmean/{em_region}")
            
            # Check if EM-Earth directories exist
            if not Path(em_earth_prcp_dir).exists():
                raise FileNotFoundError(f"EM-Earth precipitation directory not found: {em_earth_prcp_dir}")
            if not Path(em_earth_tmean_dir).exists():
                raise FileNotFoundError(f"EM-Earth temperature directory not found: {em_earth_tmean_dir}")
            
            # Get bounding box and check size
            bbox = self.config['BOUNDING_BOX_COORDS']  # format: lat_max/lon_min/lat_min/lon_max
            bbox_parts = bbox.split('/')
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min
            
            # Log watershed characteristics
            self.logger.info(f"Watershed bounding box: {bbox}")
            self.logger.info(f"Watershed size: {lat_range:.4f}° x {lon_range:.4f}° (~{lat_range*111:.1f}km x {lon_range*111:.1f}km)")
            
            # Check if watershed is very small
            min_bbox_size = self.config.get('EM_EARTH_MIN_BBOX_SIZE', 0.1)
            if lat_range < min_bbox_size or lon_range < min_bbox_size:
                self.logger.warning(f"Very small watershed detected. EM-Earth processing will use spatial averaging.")
                self.logger.info(f"Minimum bounding box size: {min_bbox_size}° (~{min_bbox_size*111:.1f}km)")
            
            # Parse time range
            try:
                start_date = datetime.strptime(self.config['EXPERIMENT_TIME_START'], '%Y-%m-%d %H:%M')
                end_date = datetime.strptime(self.config['EXPERIMENT_TIME_END'], '%Y-%m-%d %H:%M')
            except ValueError as e:
                raise ValueError(f"Invalid date format in configuration: {str(e)}")
            
            self.logger.info(f"Processing EM-Earth data for period: {start_date} to {end_date}")
            
            # Generate list of year-month combinations to process
            year_months = self._generate_year_month_list(start_date, end_date)
            
            if not year_months:
                raise ValueError("No valid year-month combinations found for the specified time period")
            
            self.logger.info(f"Processing {len(year_months)} month(s) of EM-Earth data")
            
            # Process each month of EM-Earth data
            processed_files = []
            failed_months = []
            
            for i, year_month in enumerate(year_months, 1):
                try:
                    self.logger.info(f"Processing month {i}/{len(year_months)}: {year_month}")
                    processed_file = self._process_em_earth_month(
                        year_month, em_earth_prcp_dir, em_earth_tmean_dir, em_earth_dir, bbox
                    )
                    if processed_file:
                        processed_files.append(processed_file)
                        self.logger.info(f"✓ Successfully processed EM-Earth data for {year_month}")
                    else:
                        failed_months.append(year_month)
                        self.logger.warning(f"✗ Failed to process EM-Earth data for {year_month}")
                        
                except Exception as e:
                    failed_months.append(year_month)
                    self.logger.warning(f"✗ Failed to process EM-Earth data for {year_month}: {str(e)}")
                    continue
            
            # Check results
            if not processed_files:
                raise ValueError("No EM-Earth data files were successfully processed")
            
            success_rate = len(processed_files) / len(year_months) * 100
            self.logger.info(f"EM-Earth forcing data acquisition completed")
            self.logger.info(f"Success rate: {success_rate:.1f}% ({len(processed_files)}/{len(year_months)} months)")
            
            if failed_months:
                self.logger.warning(f"Failed to process {len(failed_months)} months: {failed_months[:5]}{'...' if len(failed_months) > 5 else ''}")
                
                # If too many failures, consider it a critical issue
                if success_rate < 50:
                    raise ValueError(f"EM-Earth processing success rate too low ({success_rate:.1f}%). "
                                f"This may indicate the watershed is outside EM-Earth coverage or files are missing.")
            
        except Exception as e:
            self.logger.error(f"Error during EM-Earth forcing data acquisition: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _generate_year_month_list(self, start_date: datetime, end_date: datetime) -> List[str]:
        """
        Generate list of YYYYMM strings for the date range.
        
        Args:
            start_date: Start date of the experiment
            end_date: End date of the experiment
            
        Returns:
            List of YYYYMM strings covering the date range
        """
        year_months = []
        current_date = start_date.replace(day=1)  # Start from beginning of month
        
        while current_date <= end_date:
            year_month = current_date.strftime('%Y%m')
            year_months.append(year_month)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return year_months

    def _process_em_earth_month(self, year_month: str, prcp_dir: str, tmean_dir: str, 
                               output_dir: Path, bbox: str) -> Optional[Path]:
        """
        Process a single month of EM-Earth data.
        
        Args:
            year_month: Year-month string in YYYYMM format
            prcp_dir: Directory containing EM-Earth precipitation files
            tmean_dir: Directory containing EM-Earth temperature files
            output_dir: Output directory for processed files
            bbox: Bounding box string (lat_max/lon_min/lat_min/lon_max)
            
        Returns:
            Path to processed file, or None if processing failed
        """
        # Find input files for this month
        em_region = self.config.get('EM_EARTH_REGION', 'NorthAmerica')

        prcp_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"
        tmean_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"
        
        prcp_file = Path(prcp_dir) / prcp_pattern
        tmean_file = Path(tmean_dir) / tmean_pattern
        
        if not prcp_file.exists():
            self.logger.warning(f"EM-Earth precipitation file not found: {prcp_file}")
            return None
        if not tmean_file.exists():
            self.logger.warning(f"EM-Earth temperature file not found: {tmean_file}")
            return None
        
        # Define output file
        output_file = output_dir / f"watershed_subset_{year_month}.nc"
        
        # Skip if output already exists and not forcing rerun
        if output_file.exists() and not self.config.get('FORCE_RUN_ALL_STEPS', False):
            self.logger.info(f"EM-Earth file already exists, skipping: {output_file}")
            return output_file
        
        # Process the EM-Earth data
        try:
            self._process_em_earth_data(str(prcp_file), str(tmean_file), str(output_file), bbox)
            return output_file
        except Exception as e:
            self.logger.error(f"Error processing EM-Earth data for {year_month}: {str(e)}")
            return None

    def _process_em_earth_data(self, prcp_file: str, tmean_file: str, output_file: str, bbox: str):
        """
        Process EM-Earth precipitation and temperature data for a specific bounding box.
        
        This method replicates the functionality of the EM-Earth processing script,
        subsetting the data to the specified bounding box and merging precipitation
        and temperature variables into a single file.
        
        Args:
            prcp_file: Path to EM-Earth precipitation NetCDF file
            tmean_file: Path to EM-Earth temperature NetCDF file
            output_file: Path for output merged NetCDF file
            bbox: Bounding box string (lat_max/lon_min/lat_min/lon_max)
        """
        import xarray as xr
        import numpy as np
        
        # Parse bounding box
        bbox_parts = bbox.split('/')
        if len(bbox_parts) != 4:
            raise ValueError(f"Invalid bounding box format: {bbox}. Expected lat_max/lon_min/lat_min/lon_max")
        
        lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
        
        # Calculate bounding box size for diagnostics
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        self.logger.info(f"Processing EM-Earth data with bounding box: {lat_min}-{lat_max}°N, {lon_min}-{lon_max}°E")
        self.logger.info(f"Bounding box size: {lat_range:.4f}° x {lon_range:.4f}° (~{lat_range*111:.1f}km x {lon_range*111:.1f}km)")
        
        # For very small watersheds, expand the bounding box slightly to ensure we capture EM-Earth data
        min_bbox_size = 0.1  # Minimum bounding box size in degrees (~10km)
        original_bbox = (lat_min, lat_max, lon_min, lon_max)
        
        if lat_range < min_bbox_size or lon_range < min_bbox_size:
            self.logger.warning(f"Very small watershed detected (lat: {lat_range:.4f}°, lon: {lon_range:.4f}°)")
            self.logger.info(f"Expanding bounding box to minimum size of {min_bbox_size}° to capture EM-Earth data")
            
            # Expand around the center
            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2
            
            # Ensure minimum size
            lat_expand = max(0, (min_bbox_size - lat_range) / 2)
            lon_expand = max(0, (min_bbox_size - lon_range) / 2)
            
            lat_min_expanded = lat_center - min_bbox_size/2
            lat_max_expanded = lat_center + min_bbox_size/2
            lon_min_expanded = lon_center - min_bbox_size/2
            lon_max_expanded = lon_center + min_bbox_size/2
            
            #self.logger.info(f"Expanded bounding box: {lat_min_expanded:.4f}-{lat_max_expanded:.4f}°N, {lon_min_expanded:.4f}-{lon_max_expanded:.4f}°E")
            
            # Use expanded coordinates for data extraction
            lat_min_extract, lat_max_extract = lat_min_expanded, lat_max_expanded
            lon_min_extract, lon_max_extract = lon_min_expanded, lon_max_expanded
        else:
            lat_min_extract, lat_max_extract = lat_min, lat_max
            lon_min_extract, lon_max_extract = lon_min, lon_max
        
        # Open datasets
        try:
            prcp_ds = xr.open_dataset(prcp_file)
            tmean_ds = xr.open_dataset(tmean_file)
            
            # Log EM-Earth grid information
            #self.logger.info(f"EM-Earth grid - Lat range: {float(prcp_ds.lat.min()):.4f} to {float(prcp_ds.lat.max()):.4f}")
            #self.logger.info(f"EM-Earth grid - Lon range: {float(prcp_ds.lon.min()):.4f} to {float(prcp_ds.lon.max()):.4f}")
            #self.logger.info(f"EM-Earth grid - Resolution: ~{float(prcp_ds.lat.diff('lat').mean()):.4f}° lat, ~{float(prcp_ds.lon.diff('lon').mean()):.4f}° lon")
            
        except Exception as e:
            raise ValueError(f"Error opening EM-Earth files: {str(e)}")
        
        # Subset to bounding box (use expanded box for extraction)
        try:
            # Handle longitude wrapping if necessary
            if lon_min_extract > lon_max_extract:  # Crossing 180° meridian
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    ((prcp_ds.lon >= lon_min_extract) | (prcp_ds.lon <= lon_max_extract)), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    ((tmean_ds.lon >= lon_min_extract) | (tmean_ds.lon <= lon_max_extract)), drop=True
                )
            else:  # Normal case
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    (prcp_ds.lon >= lon_min_extract) & (prcp_ds.lon <= lon_max_extract), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    (prcp_ds.lon >= lon_min_extract) & (prcp_ds.lon <= lon_max_extract), drop=True
                )
            
            # Check if we have any data after subsetting
            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                # Try even larger expansion for very sparse grids
                self.logger.warning("No precipitation data found with initial expansion, trying larger expansion")
                
                # Expand to 0.2 degrees
                larger_expand = 0.2
                lat_center = (original_bbox[0] + original_bbox[1]) / 2
                lon_center = (original_bbox[2] + original_bbox[3]) / 2
                
                lat_min_large = lat_center - larger_expand
                lat_max_large = lat_center + larger_expand
                lon_min_large = lon_center - larger_expand
                lon_max_large = lon_center + larger_expand
                
                self.logger.info(f"Trying larger expansion: {lat_min_large:.4f}-{lat_max_large:.4f}°N, {lon_min_large:.4f}-{lon_max_large:.4f}°E")
                
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_large) & (prcp_ds.lat <= lat_max_large) &
                    (prcp_ds.lon >= lon_min_large) & (prcp_ds.lon <= lon_max_large), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_large) & (tmean_ds.lat <= lat_max_large) &
                    (prcp_ds.lon >= lon_min_large) & (prcp_ds.lon <= lon_max_large), drop=True
                )
            
            # Final check
            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No precipitation data found within the expanded bounding box. The watershed may be too small or outside the EM-Earth coverage area.")
            if tmean_subset.sizes.get('lat', 0) == 0 or tmean_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No temperature data found within the expanded bounding box. The watershed may be too small or outside the EM-Earth coverage area.")
            
            #self.logger.info(f"Successfully extracted EM-Earth data - Prcp grid: {prcp_subset.sizes['lat']} x {prcp_subset.sizes['lon']}")
            #self.logger.info(f"Successfully extracted EM-Earth data - Temp grid: {tmean_subset.sizes['lat']} x {tmean_subset.sizes['lon']}")
            
        except Exception as e:
            raise ValueError(f"Error subsetting EM-Earth data: {str(e)}")
        
        # If we used an expanded bounding box, spatially average to represent the original small watershed
        if (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox:
            self.logger.info("Computing spatial average over expanded area to represent the small watershed")
            
            # For small watersheds, take spatial mean over the extracted area
            # This gives us a single representative value for the watershed
            prcp_subset = prcp_subset.mean(dim=['lat', 'lon'], keep_attrs=True)
            tmean_subset = tmean_subset.mean(dim=['lat', 'lon'], keep_attrs=True)
            
            # Add dummy spatial dimensions to maintain structure
            prcp_subset = prcp_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            prcp_subset = prcp_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})
            
            tmean_subset = tmean_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            tmean_subset = tmean_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})
            
            self.logger.info("Applied spatial averaging for small watershed representation")
        
        # Merge datasets
        try:
            # Create merged dataset with all variables
            merged_ds = xr.Dataset()
            
            # Copy coordinates from precipitation dataset
            merged_ds = merged_ds.assign_coords({
                'lat': prcp_subset.lat,
                'lon': prcp_subset.lon,
                'time': prcp_subset.time
            })
            
            # Add precipitation variables
            for var in prcp_subset.data_vars:
                if 'prcp' in var:
                    merged_ds[var] = prcp_subset[var]
            
            # Add temperature variables (interpolate to precipitation grid if needed)
            for var in tmean_subset.data_vars:
                if 'tmean' in var or 'temp' in var:
                    # Interpolate temperature to precipitation grid
                    temp_interp = tmean_subset[var].interp(
                        lat=prcp_subset.lat, 
                        lon=prcp_subset.lon, 
                        method='linear'
                    )
                    merged_ds[var] = temp_interp
            
            # Add metadata (convert booleans to integers for NetCDF compatibility)
            is_small_watershed = lat_range < min_bbox_size or lon_range < min_bbox_size
            is_spatially_averaged = (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox
            
            merged_ds.attrs.update({
                'Dataset': 'EM-Earth: Ensemble Meteorological Dataset for Planet Earth',
                'Developer': 'Guoqiang Tang et al. in Center for Hydrology, Coldwater Lab, University of Saskatchewan',
                'Type': 'Deterministic station-reanalysis merged estimates',
                'original_bounding_box': '/'.join(map(str, original_bbox)),
                'extraction_bounding_box': f"{lat_min_extract}/{lon_min_extract}/{lat_max_extract}/{lon_max_extract}",
                'small_watershed_processing': int(is_small_watershed),  # Convert boolean to int
                'spatial_averaging_applied': int(is_spatially_averaged),  # Convert boolean to int
                'watershed_size_deg': f"{lat_range:.6f}x{lon_range:.6f}",
                'watershed_size_km': f"{lat_range*111:.2f}x{lon_range*111:.2f}",
                'subset_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'processing_script': 'CONFLUENCE EM-Earth subset and merge with small watershed handling',
                'merged_variables': f"prcp: {list(prcp_subset.data_vars.keys())}, tmean: {list(tmean_subset.data_vars.keys())}"
            })
            
            # Save merged dataset
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            merged_ds.to_netcdf(output_file)
            
            #self.logger.info(f"Successfully created merged EM-Earth file: {output_file}")
            #self.logger.info(f"Final spatial dimensions: {merged_ds.sizes['lat']} x {merged_ds.sizes['lon']}")
            #self.logger.info(f"Time dimension: {merged_ds.sizes['time']}")
            
        except Exception as e:
            raise ValueError(f"Error merging EM-Earth datasets: {str(e)}")
        
        finally:
            # Close datasets
            prcp_ds.close()
            tmean_ds.close()
            
    def process_observed_data(self):
        """
        Process observed streamflow data.
        
        This method handles the acquisition and preprocessing of observed streamflow
        data, which is used for model calibration and evaluation. It supports multiple
        data providers (WSC, USGS, VI, NIWA) and handles the necessary data format
        conversions and quality control.
        
        The processing includes:
        1. Downloading data from the specified provider (if enabled)
        2. Converting units if necessary
        3. Aligning the time series with the model time period
        4. Quality control and gap handling
        5. Saving the processed data in a standardized format
        
        The method delegates the actual processing to the ObservedDataProcessor class,
        which contains the provider-specific logic.
        
        Raises:
            FileNotFoundError: If required data files cannot be found
            ValueError: If data formatting or quality issues are detected
            Exception: For other errors during data processing
        """
        self.logger.info("Processing observed data")
        
        try:
            observed_data_processor = ObservedDataProcessor(self.config, self.logger)
            # Process streamflow data
            observed_data_processor.process_streamflow_data()
            
            # Process SNOTEL data
            observed_data_processor.process_snotel_data()

            # Process FLUXNET data
            observed_data_processor.process_fluxnet_data()

            # Process FLUXNET data
            observed_data_processor.process_usgs_groundwater_data()

            self.logger.info("Observed data processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during observed data processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def run_model_agnostic_preprocessing(self):
        """
        Run model-agnostic preprocessing including basin averaging and resampling.
        
        This method performs preprocessing operations that are common across different
        hydrological models. These operations convert the raw input data into formats
        that can be used by any model in the CONFLUENCE framework.
        
        The preprocessing includes:
        1. Geospatial statistics: Computing zonal statistics to aggregate geospatial
           attributes (elevation, land cover, soil types) to the hydrological response 
           units (HRUs) or basins
        2. Forcing resampling: Temporal and spatial resampling of meteorological forcing
           data to match the model's requirements, including basin averaging for lumped
           models
        3. EM-Earth integration: If SUPPLEMENT_FORCING is enabled, integrates EM-Earth
           precipitation and temperature data with the primary forcing dataset
        
        The method creates the necessary directory structure and delegates the actual
        processing to specialized classes (geospatialStatistics and forcingResampler).
        
        Raises:
            FileNotFoundError: If required input files cannot be found
            ValueError: If resampling or aggregation parameters are invalid
            Exception: For other errors during preprocessing
        """
        self.logger.info("Starting model-agnostic preprocessing")
        
        # Create required directories
        basin_averaged_data = self.project_dir / 'forcing' / 'basin_averaged_data'
        catchment_intersection_dir = self.project_dir / 'shapefiles' / 'catchment_intersection'
        
        basin_averaged_data.mkdir(parents=True, exist_ok=True)
        catchment_intersection_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run geospatial statistics
            self.logger.info("Running geospatial statistics")
            gs = geospatialStatistics(self.config, self.logger)
            gs.run_statistics()
            
            # Run forcing resampling
            self.logger.info("Running forcing resampling")
            fr = forcingResampler(self.config, self.logger)
            fr.run_resampling()
            
            # Integrate EM-Earth data if supplementation is enabled
            if self.config.get('SUPPLEMENT_FORCING', False):
                self.logger.info("SUPPLEMENT_FORCING enabled - integrating EM-Earth data")
                self._integrate_em_earth_data()
            
            # Run MAF Orchestrator if needed (currently commented out)
            # hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
            # if 'MESH' in hydrological_models or 'HYPE' in hydrological_models:
            #     from utils.data.data_utils import DataAcquisitionProcessor
            #     dap = DataAcquisitionProcessor(self.config, self.logger)
            #     dap.run_data_acquisition()

            # Archive raw forcing data to save storage space
            """
            self.logger.info("Archiving raw forcing data to save storage space")
            try:
                raw_data_dir = self.project_dir / 'forcing' / 'raw_data'
                if raw_data_dir.exists():
                    success = tar_directory(
                        raw_data_dir, 
                        "raw_forcing_data.tar.gz",
                        remove_original=True,
                        logger=self.logger
                    )
                    if success:
                        self.logger.info("Raw forcing data archived successfully")
                    else:
                        self.logger.warning("Failed to archive raw forcing data")
                
                # Also archive EM-Earth raw data if it exists
                if self.config.get('SUPPLEMENT_FORCING', False):
                    em_earth_dir = self.project_dir / 'forcing' / 'raw_data_em_earth'
                    if em_earth_dir.exists():
                        success = tar_directory(
                            em_earth_dir,
                            "raw_em_earth_data.tar.gz", 
                            remove_original=True,
                            logger=self.logger
                        )
                        if success:
                            self.logger.info("Raw EM-Earth data archived successfully")
                        else:
                            self.logger.warning("Failed to archive raw EM-Earth data")
                        
                        # Also archive remapped EM-Earth data if it exists
                        remapped_dir = self.project_dir / 'forcing' / 'em_earth_remapped'
                        if remapped_dir.exists():
                            success = tar_directory(
                                remapped_dir,
                                "remapped_em_earth_data.tar.gz",
                                remove_original=True, 
                                logger=self.logger
                            )
                            if success:
                                self.logger.info("Remapped EM-Earth data archived successfully")
                
            except Exception as e:
                self.logger.warning(f"Error during raw data archiving: {str(e)}")
            """

                # Continue execution even if archiving fails
            self.logger.info("Model-agnostic preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model-agnostic preprocessing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _integrate_em_earth_data(self):
        """
        Integrate EM-Earth precipitation and temperature data with primary forcing dataset.
        
        This method:
        1. Remaps EM-Earth data to the same grid as the primary forcing dataset
        2. Replaces precipitation and temperature variables in the basin-averaged data
        3. Ensures temporal alignment and unit consistency
        
        The integration preserves all other meteorological variables from the primary
        dataset while using EM-Earth's higher quality precipitation and temperature.
        """
        self.logger.info("Starting EM-Earth data integration")
        
        try:
            # Check if EM-Earth data exists
            em_earth_dir = self.project_dir / 'forcing' / 'raw_data_em_earth'
            if not em_earth_dir.exists():
                self.logger.warning("EM-Earth data directory not found, skipping integration")
                return
            
            # Find EM-Earth files
            em_earth_files = list(em_earth_dir.glob("watershed_subset_*.nc"))
            if not em_earth_files:
                self.logger.warning("No EM-Earth files found, skipping integration")
                return
            
            #self.logger.info(f"Found {len(em_earth_files)} EM-Earth files for integration")
            
            # Process and remap EM-Earth data
            self._remap_em_earth_to_basin_grid()
            
            # Replace precipitation and temperature in basin-averaged data
            self._replace_forcing_variables_with_em_earth()
            
            self.logger.info("EM-Earth data integration completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during EM-Earth data integration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _remap_em_earth_to_basin_grid(self):
        """
        Remap EM-Earth data to match the basin grid used by the primary forcing dataset.
        
        This method uses spatial averaging to convert the EM-Earth gridded data to
        basin-averaged values that match the structure of the primary forcing data.
        Uses parallel processing when multiple processors are available.
        """
        self.logger.info("Remapping EM-Earth data to basin grid")
        
        try:
            import xarray as xr
            import numpy as np
            
            # Get basin shapefile for remapping
            subbasins_name = self.config.get('RIVER_BASINS_NAME')
            if subbasins_name == 'default':
                subbasins_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"
            
            basin_shapefile = self.project_dir / "shapefiles/river_basins" / subbasins_name
            
            if not basin_shapefile.exists():
                raise FileNotFoundError(f"Basin shapefile not found: {basin_shapefile}")
            
            # Create output directory for remapped EM-Earth data
            remapped_dir = self.project_dir / 'forcing' / 'em_earth_remapped'
            remapped_dir.mkdir(parents=True, exist_ok=True)
            
            # Find EM-Earth files
            em_earth_dir = self.project_dir / 'forcing' / 'raw_data_em_earth'
            em_earth_files = sorted(em_earth_dir.glob("watershed_subset_*.nc"))
            
            if not em_earth_files:
                self.logger.warning("No EM-Earth files found for remapping")
                return
            
            # Filter out files that already exist (unless forcing rerun)
            files_to_process = []
            for em_file in em_earth_files:
                output_file = remapped_dir / f"remapped_{em_file.name}"
                if not output_file.exists() or self.config.get('FORCE_RUN_ALL_STEPS', False):
                    files_to_process.append(em_file)
            
            if not files_to_process:
                self.logger.info("All EM-Earth files already remapped, skipping")
                return
            
            self.logger.info(f"Found {len(files_to_process)} EM-Earth files to remap")
            
            # Check if we should use parallel processing
            num_processes = self.config.get('MPI_PROCESSES', 1)
            use_parallel = num_processes > 1 and len(files_to_process) > 1
            
            if use_parallel:
                self.logger.info(f"Using parallel processing with {num_processes} workers")
                self._remap_em_earth_files_parallel(files_to_process, basin_shapefile, remapped_dir, num_processes)
            else:
                self.logger.info("Using sequential processing")
                self._remap_em_earth_files_sequential(files_to_process, basin_shapefile, remapped_dir)
            
        except Exception as e:
            self.logger.error(f"Error remapping EM-Earth data: {str(e)}")
            raise

    def _remap_em_earth_files_sequential(self, files_to_process, basin_shapefile, remapped_dir):
        """
        Remap EM-Earth files sequentially (original method).
        
        Args:
            files_to_process: List of EM-Earth files to process
            basin_shapefile: Path to basin shapefile
            remapped_dir: Output directory for remapped files
        """
        # Process each EM-Earth file
        for i, em_file in enumerate(files_to_process, 1):
            output_file = remapped_dir / f"remapped_{em_file.name}"
            
            self.logger.info(f"Processing file {i}/{len(files_to_process)}: {em_file.name}")
            
            try:
                self._remap_single_em_earth_file(em_file, output_file, basin_shapefile)
                self.logger.info(f"✓ Successfully remapped {em_file.name}")
            except Exception as e:
                self.logger.error(f"✗ Failed to remap {em_file.name}: {str(e)}")
                continue

    def _remap_em_earth_files_parallel(self, files_to_process, basin_shapefile, remapped_dir, num_processes):
        """
        Remap EM-Earth files in parallel using multiprocessing.
        
        Args:
            files_to_process: List of EM-Earth files to process
            basin_shapefile: Path to basin shapefile
            remapped_dir: Output directory for remapped files
            num_processes: Number of parallel processes to use
        """

        
        # Limit workers to available files
        num_workers = min(num_processes, len(files_to_process))
        
        self.logger.info(f"Starting parallel EM-Earth remapping with {num_workers} workers")
        
        # Create shared arguments for workers
        worker_args = []
        for em_file in files_to_process:
            output_file = remapped_dir / f"remapped_{em_file.name}"
            worker_args.append((
                str(em_file),
                str(output_file), 
                str(basin_shapefile),
                self.config.copy()  # Pass config copy to workers
            ))
        
        # Start timer
        start_time = time.time()
        
        # Create and run process pool
        with mp.Pool(processes=num_workers) as pool:
            try:
                # Use map_async for better error handling and progress tracking
                result = pool.map_async(
                    _remap_em_earth_worker_optimized, 
                    worker_args,
                    chunksize=1
                )
                
                # Wait for completion with timeout checking
                results = []
                while True:
                    try:
                        results = result.get(timeout=30)  # Check every 30 seconds
                        break
                    except mp.TimeoutError:
                        # Log progress periodically
                        if hasattr(result, '_number_left'):
                            completed = len(worker_args) - result._number_left
                            self.logger.info(f"Parallel remapping progress: {completed}/{len(worker_args)} files completed")
                        continue
                
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {str(e)}")
                # Terminate pool
                pool.terminate()
                pool.join()
                raise
        
        # Process results
        successful = sum(results)
        failed = len(files_to_process) - successful
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Parallel EM-Earth remapping completed in {elapsed_time:.1f} seconds")
        self.logger.info(f"Success rate: {successful}/{len(files_to_process)} files ({successful/len(files_to_process)*100:.1f}%)")
        
        if failed > 0:
            self.logger.warning(f"{failed} files failed to process")
        
        if successful == 0:
            raise ValueError("No EM-Earth files were successfully remapped")

    def _remap_single_em_earth_file(self, input_file: Path, output_file: Path, basin_shapefile: Path):
        """
        OPTIMIZED: Remap a single EM-Earth file to basin-averaged values.
        Updated with pre-computed weights optimization for major performance gains.
        
        Args:
            input_file: Path to EM-Earth NetCDF file
            output_file: Path for output remapped file
            basin_shapefile: Path to basin shapefile for spatial averaging
        """
        try:
            # Performance monitoring
            start_time = time.time()
            
            if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('LOG_PERFORMANCE', False):
                self.logger.info(f"Starting optimized processing of {input_file.name}")

            # Read EM-Earth data
            em_ds = xr.open_dataset(input_file)
            
            # Read basin shapefile
            basins_gdf = gpd.read_file(basin_shapefile)
            
            # Check and reproject shapefile to WGS84 if necessary
            original_crs = basins_gdf.crs
            if basins_gdf.crs is None:
                basins_gdf = basins_gdf.set_crs('EPSG:4326')
            elif not basins_gdf.crs.equals(4326) and not basins_gdf.crs.to_string().upper() == 'EPSG:4326':
                basins_gdf = basins_gdf.to_crs('EPSG:4326')
            
            # Get basin ID column
            basin_id_col = self.config.get('RIVER_BASIN_SHP_RM_GRUID', 'GRU_ID')
            
            if basin_id_col not in basins_gdf.columns:
                raise ValueError(f"Basin ID column '{basin_id_col}' not found in shapefile")
            
            # Create output dataset structure
            basin_ids = sorted(basins_gdf[basin_id_col].unique())
            
            # Initialize output dataset
            output_ds = xr.Dataset()
            output_ds = output_ds.assign_coords({
                'time': em_ds.time,
                'hru': basin_ids
            })
            
            # Check if this is a spatially averaged small watershed (single grid point)
            is_single_point = (len(em_ds.lat) == 1 and len(em_ds.lon) == 1)
            small_watershed_flag = em_ds.attrs.get('small_watershed_processing', 0)
            spatial_averaging_flag = em_ds.attrs.get('spatial_averaging_applied', 0)
            is_single_point = is_single_point or (small_watershed_flag == 1) or (spatial_averaging_flag == 1)
            
            if is_single_point:
                # Use existing single-point logic (already fast)
                self._process_single_point_data(em_ds, output_ds, basin_ids)
                processing_method = 'single_point_replication'
            else:
                # Use optimized multi-point processing
                use_optimization = self.config.get('EM_EARTH_OPTIMIZATION', {}).get('USE_OPTIMIZED_REMAPPING', True)
                
                if use_optimization:
                    try:
                        self._process_multi_point_data_optimized(em_ds, output_ds, basins_gdf, basin_id_col, basin_ids)
                        processing_method = 'optimized_vectorized_averaging'
                    except Exception as opt_error:
                        if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('FALLBACK_ON_ERROR', True):
                            if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('DETAILED_LOGGING', False):
                                self.logger.warning(f"Optimization failed, falling back to original method: {str(opt_error)}")
                            self._process_multi_point_data_original(em_ds, output_ds, basins_gdf, basin_id_col, basin_ids)
                            processing_method = 'zonal_statistics_fallback'
                        else:
                            raise opt_error
                else:
                    # Use original method
                    self._process_multi_point_data_original(em_ds, output_ds, basins_gdf, basin_id_col, basin_ids)
                    processing_method = 'zonal_statistics'
            
            # Add metadata
            output_ds.attrs.update({
                'remapped_from': str(input_file),
                'remapping_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'remapping_method': processing_method,
                'basin_shapefile': str(basin_shapefile),
                'input_grid_size': f"{len(em_ds.lat)}x{len(em_ds.lon)}",
                'output_basins': len(basin_ids),
                'small_watershed_processing': int(is_single_point),
                'optimization_version': 'precomputed_weights_v1.0'
            })
            
            # Save remapped dataset
            output_ds.to_netcdf(output_file)
            
            # Close datasets
            em_ds.close()
            output_ds.close()
            
            # Log performance
            if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('LOG_PERFORMANCE', False):
                elapsed_time = time.time() - start_time
                self.logger.info(f"✓ Completed {input_file.name} in {elapsed_time:.2f}s using {processing_method}")
            
            return True
            
        except Exception as e:
            # Log error with file context for debugging
            self.logger.error(f"Error remapping {input_file.name}: {str(e)}")
            return False

    def _precompute_basin_pixel_mapping(self, em_ds, basins_gdf, basin_id_col):
        """
        OPTIMIZATION: Pre-compute which pixels belong to which basins.
        This is the key optimization - do this once instead of zonal_stats per timestep.
        """
        # Get spatial information
        lats = em_ds.lat.values
        lons = em_ds.lon.values
        
        # Check for valid spatial extent
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())
        
        lat_extent = lat_max - lat_min
        lon_extent = lon_max - lon_min
        
        if lat_extent == 0 or lon_extent == 0:
            raise ValueError("Zero spatial extent detected - should use single point processing")
        
        # Create rasterio transform
        transform = rasterio.transform.from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            len(lons), len(lats)
        )
        
        # Get unique basin IDs
        basin_ids = sorted(basins_gdf[basin_id_col].unique())
        
        # Create geometry-ID pairs for rasterization
        geom_value_pairs = [
            (geom, basin_id) for geom, basin_id in 
            zip(basins_gdf.geometry, basins_gdf[basin_id_col])
        ]
        
        # Rasterize all basins at once - this creates a basin ID raster
        basin_raster = rasterize(
            geom_value_pairs,
            out_shape=(len(lats), len(lons)),
            transform=transform,
            fill=0,  # Background value for pixels not in any basin
            dtype='int32',
            all_touched=True  # Include pixels that touch basin boundaries
        )
        
        # Create boolean masks for each basin
        basin_masks = {}
        for basin_id in basin_ids:
            mask = (basin_raster == basin_id)
            if mask.any():  # Only store if basin has pixels
                basin_masks[basin_id] = mask
            elif self.config.get('EM_EARTH_OPTIMIZATION', {}).get('DETAILED_LOGGING', False):
                self.logger.warning(f"Basin {basin_id} has no pixels in the grid")
        
        if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('DETAILED_LOGGING', False):
            self.logger.debug(f"Pre-computed masks for {len(basin_masks)} basins")
        
        return basin_masks, basin_ids, transform

    def _vectorized_basin_averaging(self, var_data, basin_masks, basin_ids):
        """
        OPTIMIZATION: Vectorized basin averaging - processes all time steps simultaneously.
        This is where the major speed improvement comes from.
        """
        # Get dimensions
        n_times = len(var_data.time)
        n_basins = len(basin_ids)
        
        # Use float32 if configured to save memory
        dtype = np.float32 if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('USE_FLOAT32', True) else np.float64
        
        # Initialize output array
        basin_values = np.full((n_times, n_basins), np.nan, dtype=dtype)
        
        # Convert xarray to numpy for faster processing
        data_array = var_data.values  # Shape: (time, lat, lon)
        
        # Process each basin using vectorized operations
        for i, basin_id in enumerate(basin_ids):
            if basin_id in basin_masks:
                mask = basin_masks[basin_id]
                
                # Apply mask to all time steps at once - THIS IS THE KEY OPTIMIZATION
                # Instead of looping through time steps, we process all at once
                masked_data = data_array[:, mask]  # Shape: (time, n_pixels_in_basin)
                
                # Average across pixels for all time steps simultaneously
                if masked_data.size > 0:
                    basin_values[:, i] = np.nanmean(masked_data, axis=1)
        
        return basin_values

    def _process_multi_point_data_optimized(self, em_ds, output_ds, basins_gdf, basin_id_col, basin_ids):
        """
        OPTIMIZATION: Optimized multi-point processing using vectorized operations.
        """
        try:
            # Pre-compute basin-pixel mapping (do this once per file)
            basin_masks, valid_basin_ids, transform = self._precompute_basin_pixel_mapping(
                em_ds, basins_gdf, basin_id_col
            )
            
            # Process each variable using vectorized operations
            for var_name in em_ds.data_vars:
                if var_name in ['prcp', 'prcp_corrected', 'tmean']:
                    if self.config.get('EM_EARTH_OPTIMIZATION', {}).get('DETAILED_LOGGING', False):
                        self.logger.debug(f"Processing variable {var_name} with vectorized method")
                    
                    # Get variable data
                    var_data = em_ds[var_name]  # Shape: (time, lat, lon)
                    
                    # Use vectorized basin averaging - THIS REPLACES THE TIME LOOP
                    basin_values = self._vectorized_basin_averaging(
                        var_data, basin_masks, basin_ids
                    )
                    
                    # Add to output dataset
                    output_ds[var_name] = xr.DataArray(
                        basin_values,
                        dims=['time', 'hru'],
                        coords={'time': em_ds.time, 'hru': basin_ids},
                        attrs=var_data.attrs
                    )
                    
                    # Update attributes
                    output_ds[var_name].attrs.update({
                        'spatial_processing': 'optimized_vectorized_averaging',
                        'note': 'Basin-averaged using pre-computed pixel masks'
                    })
                    
        except Exception as e:
            # If optimization fails, re-raise for fallback handling
            raise e

    def _process_single_point_data(self, em_ds, output_ds, basin_ids):
        """
        Process single-point data (extracted from existing code, already optimized).
        """
        for var_name in em_ds.data_vars:
            if var_name in ['prcp', 'prcp_corrected', 'tmean']:
                var_data = em_ds[var_name]
                
                # Extract time series from single point
                if len(var_data.dims) == 3:  # (time, lat, lon)
                    time_dim_index = var_data.dims.index('time')
                    if time_dim_index == 0:
                        time_series = var_data.values[:, 0, 0]
                    elif time_dim_index == 1:
                        time_series = var_data.values[0, :, 0]
                    elif time_dim_index == 2:
                        time_series = var_data.values[0, 0, :]
                elif len(var_data.dims) == 1:
                    time_series = var_data.values
                elif len(var_data.dims) == 2:
                    time_dim_index = var_data.dims.index('time')
                    if time_dim_index == 0:
                        time_series = var_data.values[:, 0] if var_data.shape[1] > 0 else var_data.values.flatten()
                    else:
                        time_series = var_data.values[0, :] if var_data.shape[0] > 0 else var_data.values.flatten()
                
                # Ensure it's a 1D array
                time_series = np.asarray(time_series).flatten()
                
                # Replicate for all basins
                basin_values = np.tile(
                    time_series.reshape(-1, 1),
                    (1, len(basin_ids))
                )
                
                # Add to output dataset
                output_ds[var_name] = xr.DataArray(
                    basin_values,
                    dims=['time', 'hru'],
                    coords={'time': em_ds.time, 'hru': basin_ids},
                    attrs=var_data.attrs
                )
                
                # Update attributes
                output_ds[var_name].attrs.update({
                    'spatial_processing': 'single_point_replication',
                    'note': 'Single spatially-averaged value applied to all basins'
                })

    def _process_multi_point_data_original(self, em_ds, output_ds, basins_gdf, basin_id_col, basin_ids):
        """
        Original multi-point processing method as a fallback.
        This preserves the existing functionality exactly as before.
        """
        # Original zonal statistics method from existing code
        for var_name in em_ds.data_vars:
            if var_name in ['prcp', 'prcp_corrected', 'tmean']:
                # Get variable data
                var_data = em_ds[var_name]
                
                # Create basin-averaged values for each time step
                basin_values = np.full((len(em_ds.time), len(basin_ids)), np.nan)
                
                # Create rasterio transform for zonal statistics
                lat_min, lat_max = float(em_ds.lat.min()), float(em_ds.lat.max())
                lon_min, lon_max = float(em_ds.lon.min()), float(em_ds.lon.max())
                
                lat_extent = lat_max - lat_min
                lon_extent = lon_max - lon_min
                
                if lat_extent == 0 or lon_extent == 0:
                    # Force single-point processing
                    raise ValueError("Zero spatial extent - should use single point processing")
                
                transform = rasterio.transform.from_bounds(
                    lon_min, lat_min, lon_max, lat_max,
                    len(em_ds.lon), len(em_ds.lat)
                )
                
                # Process each time step (original method)
                for t_idx, time_val in enumerate(em_ds.time):
                    # Get data for this time step
                    time_data = var_data.isel(time=t_idx)
                    
                    # Calculate zonal statistics for each basin
                    try:
                        stats = zonal_stats(
                            basins_gdf.geometry,
                            time_data.values,
                            affine=transform,
                            stats=['mean'],
                            nodata=np.nan
                        )
                    except Exception as e:
                        raise ValueError(f"Zonal statistics failed for time {t_idx}: {str(e)}")
                    
                    # Store basin-averaged values
                    for b_idx, basin_id in enumerate(basin_ids):
                        basin_row = basins_gdf[basins_gdf[basin_id_col] == basin_id].iloc[0]
                        geom_idx = basins_gdf.index[basins_gdf[basin_id_col] == basin_id].tolist()[0]
                        
                        if geom_idx < len(stats) and stats[geom_idx]['mean'] is not None:
                            basin_values[t_idx, b_idx] = stats[geom_idx]['mean']
                
                # Add to output dataset
                output_ds[var_name] = xr.DataArray(
                    basin_values,
                    dims=['time', 'hru'],
                    coords={'time': em_ds.time, 'hru': basin_ids},
                    attrs=var_data.attrs
                )
                
                # Update attributes
                output_ds[var_name].attrs.update({
                    'spatial_processing': 'zonal_statistics',
                    'note': 'Basin-averaged using traditional zonal statistics'
                })

    def _replace_forcing_variables_with_em_earth(self):
        """
        Replace precipitation and temperature variables in basin-averaged data with EM-Earth values.
        
        This method merges the remapped EM-Earth data with the existing basin-averaged
        forcing data, replacing precipitation and temperature while preserving other variables.
        """
        self.logger.info("Replacing forcing variables with EM-Earth data")
        
        try:
            import xarray as xr
            import numpy as np
            
            # Find basin-averaged forcing files
            basin_data_dir = self.project_dir / 'forcing' / 'basin_averaged_data'
            if not basin_data_dir.exists():
                raise FileNotFoundError(f"Basin-averaged data directory not found: {basin_data_dir}")
            
            # Find remapped EM-Earth files
            remapped_dir = self.project_dir / 'forcing' / 'em_earth_remapped'
            if not remapped_dir.exists():
                raise FileNotFoundError(f"Remapped EM-Earth directory not found: {remapped_dir}")
            
            # Find forcing files that need updating
            forcing_files = list(basin_data_dir.glob("*.nc"))
            em_earth_files = list(remapped_dir.glob("remapped_watershed_subset_*.nc"))
            
            if not forcing_files:
                self.logger.warning("No basin-averaged forcing files found")
                return
            
            if not em_earth_files:
                self.logger.warning("No remapped EM-Earth files found")
                return
            
            #self.logger.info(f"Found {len(forcing_files)} forcing files and {len(em_earth_files)} EM-Earth files")
            
            # Create EM-Earth lookup by time period
            em_earth_lookup = {}
            for em_file in em_earth_files:
                # Extract year-month from filename
                year_month = em_file.name.split('_')[-1].replace('.nc', '')
                em_earth_lookup[year_month] = em_file
            
            # Process each forcing file
            for forcing_file in forcing_files:
                try:
                    self._update_single_forcing_file(forcing_file, em_earth_lookup)
                except Exception as e:
                    self.logger.warning(f"Failed to update {forcing_file.name}: {str(e)}")
                    continue
            
            self.logger.info("Successfully replaced forcing variables with EM-Earth data")
            
        except Exception as e:
            self.logger.error(f"Error replacing forcing variables: {str(e)}")
            raise

    def _update_single_forcing_file(self, forcing_file: Path, em_earth_lookup: Dict[str, Path]):
        """
        Update a single forcing file with EM-Earth precipitation and temperature data.
        
        Args:
            forcing_file: Path to forcing file to update
            em_earth_lookup: Dictionary mapping year-month to EM-Earth file paths
        """
        try:
            
            # Open forcing dataset
            forcing_ds = xr.open_dataset(forcing_file)
            
            # Determine which EM-Earth files cover the time period
            start_time = forcing_ds.time.min().values
            end_time = forcing_ds.time.max().values
            
            # Convert to datetime for processing
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            
            # Find overlapping EM-Earth files
            em_datasets = []
            for year_month, em_file in em_earth_lookup.items():
                # Check if this EM-Earth file overlaps with forcing time period
                year = int(year_month[:4])
                month = int(year_month[4:])
                
                em_start = datetime(year, month, 1)
                em_end = datetime(year, month, calendar.monthrange(year, month)[1])
                
                if (em_start <= end_dt.to_pydatetime() and em_end >= start_dt.to_pydatetime()):
                    em_ds = xr.open_dataset(em_file)
                    em_datasets.append(em_ds)
            
            if not em_datasets:
                self.logger.warning(f"No matching EM-Earth data for {forcing_file.name}")
                return
            
            # Concatenate EM-Earth datasets
            em_combined = xr.concat(em_datasets, dim='time')
            
            # Align time coordinates
            em_combined = em_combined.sel(time=slice(start_time, end_time))
            
            # Replace precipitation and temperature variables
            updated_ds = forcing_ds.copy(deep=True)
            
            # FIXED: Updated variable mapping to include CONFLUENCE variable names
            variable_mapping = {
                'prcp': ['pcp', 'precipitation', 'PRCP', 'prcp', 'pptrate'],  # Added pptrate
                'prcp_corrected': ['pcp', 'precipitation', 'PRCP', 'prcp', 'pptrate'],  # Added pptrate
                'tmean': ['tmp', 'temperature', 'TEMP', 'tmean', 'tas', 'airtemp']  # Added airtemp
            }
            
            # Update variables
            for em_var, forcing_vars in variable_mapping.items():
                if em_var in em_combined.data_vars:
                    # Find corresponding variable in forcing dataset
                    for forcing_var in forcing_vars:
                        if forcing_var in updated_ds.data_vars:
                            #self.logger.info(f"Replacing {forcing_var} with EM-Earth {em_var}")
                            
                            # Interpolate EM-Earth data to forcing time grid
                            em_data_interp = em_combined[em_var].interp(time=forcing_ds.time)
                            
                            # Convert units if necessary
                            if em_var in ['prcp', 'prcp_corrected']:
                                # EM-Earth is in mm/hour, check if forcing expects different units
                                current_units = str(updated_ds[forcing_var].attrs.get('units', ''))
                                
                                if 'kg m-2 s-1' in current_units or 'kg m**-2 s**-1' in current_units:
                                    # Convert mm/hour to kg/m²/s (mm/hour / 3.6)
                                    em_data_interp = em_data_interp / 3600
                                    #self.logger.info(f"Converted precipitation from mm/hour to kg/m²/s")
                                elif 'mm/s' in current_units or 'mm s-1' in current_units:
                                    # Convert mm/hour to mm/s (mm/hour / 3600)
                                    em_data_interp = em_data_interp / 3600
                                    #self.logger.info(f"Converted precipitation from mm/hour to mm/s")
                                else:
                                    self.logger.warning(f"Unknown precipitation units: {current_units}, using mm/hour")
                            
                            elif em_var == 'tmean':
                                # EM-Earth is in Celsius, check if forcing expects Kelvin
                                current_units = str(updated_ds[forcing_var].attrs.get('units', ''))
                                
                                if 'K' in current_units and 'Celsius' not in current_units:
                                    # Convert Celsius to Kelvin
                                    em_data_interp = em_data_interp + 273.15

                            
                            # Update the forcing variable
                            updated_ds[forcing_var] = em_data_interp
                            
                            # Update attributes
                            updated_ds[forcing_var].attrs.update({
                                'source': f'EM-Earth {em_var}',
                                'replacement_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            break
            
            # Add global attributes about EM-Earth replacement
            # Convert boolean to integer for NetCDF compatibility
            updated_ds.attrs.update({
                'em_earth_replacement': 1,  # Use integer instead of boolean
                'em_earth_replacement_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'em_earth_variables_replaced': 'precipitation, temperature'
            })
            
            # Write to a temporary file first, then replace the original
            temp_file = forcing_file.with_suffix('.nc.temp')
            
            try:
                # Write updated dataset to temporary file
                updated_ds.to_netcdf(temp_file)
                
                # Close datasets to release file handles
                forcing_ds.close()
                updated_ds.close()
                for em_ds in em_datasets:
                    em_ds.close()
                
                # Remove the original file and rename temp file
                if forcing_file.exists():
                    forcing_file.unlink()  # Remove original file
                
                temp_file.rename(forcing_file)  # Rename temp file to original name
                
                self.logger.info(f"Successfully updated {forcing_file.name} with EM-Earth data")
                
            except Exception as write_error:
                # Clean up temp file if something went wrong
                if temp_file.exists():
                    temp_file.unlink()
                
                # Re-raise the error
                raise write_error
            
        except Exception as e:
            self.logger.error(f"Error updating {forcing_file.name}: {str(e)}")
            raise
    
    def validate_data_directories(self) -> bool:
        """
        Validate that required data directories exist.
        
        This method checks for the existence of essential data directories to ensure
        that the data acquisition and preprocessing steps have been completed successfully.
        It serves as a simple validation before proceeding with model-specific operations.
        
        The required directories include:
        - attributes: For geospatial attributes (elevation, soil, land cover)
        - forcing: For meteorological forcing data
        - observations: For observed streamflow data
        - shapefiles: For domain geometry definitions
        
        Returns:
            bool: True if all required directories exist, False otherwise
            
        Note:
            This method only checks for directory existence, not for the presence
            or validity of specific files within those directories.
        """
        required_dirs = [
            self.project_dir / 'attributes',
            self.project_dir / 'forcing',
            self.project_dir / 'observations',
            self.project_dir / 'shapefiles'
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.logger.warning(f"Required directory does not exist: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def get_data_status(self) -> Dict[str, Any]:
        """
        Get status of data acquisition and preprocessing.
        
        This method provides a comprehensive status report on the data acquisition
        and preprocessing operations. It checks for the existence of key directories
        and files to determine which steps have been completed successfully.
        
        The status information is useful for:
        - Diagnosing issues in the workflow
        - Determining which steps need to be rerun
        - Providing feedback to users on workflow progress
        
        Returns:
            Dict[str, Any]: Dictionary containing data status information, including:
                - project_dir: Path to the project directory
                - attributes_acquired: Whether attribute acquisition is complete
                - forcings_acquired: Whether forcing data acquisition is complete
                - forcings_preprocessed: Whether forcing preprocessing is complete
                - observed_data_processed: Whether observed data processing is complete
                - em_earth_acquired: Whether EM-Earth data has been acquired
                - em_earth_integrated: Whether EM-Earth data has been integrated
                - dem_exists: Whether DEM data exists
                - soilclass_exists: Whether soil class data exists
                - landclass_exists: Whether land cover data exists
        """
        status = {
            'project_dir': str(self.project_dir),
            'attributes_acquired': (self.project_dir / 'attributes' / 'elevation' / 'dem').exists(),
            'forcings_acquired': (self.project_dir / 'forcing' / 'raw_data').exists(),
            'forcings_preprocessed': (self.project_dir / 'forcing' / 'basin_averaged_data').exists(),
            'observed_data_processed': (self.project_dir / 'observations' / 'streamflow' / 'preprocessed').exists(),
        }
        
        # Check specific attribute files
        status['dem_exists'] = (self.project_dir / 'attributes' / 'elevation' / 'dem').exists()
        status['soilclass_exists'] = (self.project_dir / 'attributes' / 'soilclass').exists()
        status['landclass_exists'] = (self.project_dir / 'attributes' / 'landclass').exists()
        
        # Check EM-Earth status if supplementation is enabled
        if self.config.get('SUPPLEMENT_FORCING', False):
            status['em_earth_acquired'] = (self.project_dir / 'forcing' / 'raw_data_em_earth').exists()
            status['em_earth_integrated'] = (self.project_dir / 'forcing' / 'em_earth_remapped').exists()
        else:
            status['em_earth_acquired'] = False
            status['em_earth_integrated'] = False
        
        return status

# OPTIMIZED: Updated parallel worker function with optimization support
def _remap_em_earth_worker_optimized(args):
    """
    OPTIMIZED: Updated worker function for parallel EM-Earth remapping with optimization support.
    
    Args:
        args: Tuple containing (input_file_str, output_file_str, basin_shapefile_str, config)
    
    Returns:
        bool: True if successful, False otherwise
    """
    input_file_str, output_file_str, basin_shapefile_str, config = args
    
    try:
        # Create a temporary DataManager instance for the worker
        import logging
        temp_logger = logging.getLogger('confluence_worker')
        
        # Create minimal DataManager-like object
        class TempDataManager:
            def __init__(self, config, logger):
                self.config = config
                self.logger = logger
        
        temp_dm = TempDataManager(config, temp_logger)
        
        # Add all the optimized methods to the temp instance
        temp_dm._precompute_basin_pixel_mapping = DataManager._precompute_basin_pixel_mapping.__get__(temp_dm)
        temp_dm._vectorized_basin_averaging = DataManager._vectorized_basin_averaging.__get__(temp_dm)
        temp_dm._process_multi_point_data_optimized = DataManager._process_multi_point_data_optimized.__get__(temp_dm)
        temp_dm._process_single_point_data = DataManager._process_single_point_data.__get__(temp_dm)
        temp_dm._process_multi_point_data_original = DataManager._process_multi_point_data_original.__get__(temp_dm)
        temp_dm._remap_single_em_earth_file = DataManager._remap_single_em_earth_file.__get__(temp_dm)
        
        # Convert strings back to Path objects
        input_file = Path(input_file_str)
        output_file = Path(output_file_str)
        basin_shapefile = Path(basin_shapefile_str)
        
        # Call the optimized remapping function
        return temp_dm._remap_single_em_earth_file(input_file, output_file, basin_shapefile)
        
    except Exception as e:
        return False