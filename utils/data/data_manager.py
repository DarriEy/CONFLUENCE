# In utils/dataHandling_utils/data_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

from utils.data.data_utils import ObservedDataProcessor, gistoolRunner, datatoolRunner # type: ignore 
from utils.data.agnosticPreProcessor import forcingResampler, geospatialStatistics # type: ignore 
from utils.data.variable_utils import VariableHandler # type: ignore 
from utils.geospatial.raster_utils import calculate_landcover_mode # type: ignore 

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
        latlims = f"{bbox[2]},{bbox[0]}"
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
        
        Returns:
            None: If data is user-supplied for point simulations
            
        Raises:
            FileNotFoundError: If external data sources cannot be accessed
            ValueError: If coordinate bounds or time period are invalid
            Exception: For other errors during data acquisition
        """
        self.logger.info("Starting forcing data acquisition")
        
        # Check if data should be supplied rather than acquired
        if self.config.get('SPATIAL_MODE') == 'Point' and self.config.get('DATA_ACQUIRE') == 'supplied':
            self.logger.info("Spatial mode: Point simulations, data supplied")
            return None
        
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
            
            self.logger.info("Forcing data acquisition completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during forcing data acquisition: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
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
            observed_data_processor.process_streamflow_data()
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
            
            # Run MAF Orchestrator if needed (currently commented out)
            # hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
            # if 'MESH' in hydrological_models or 'HYPE' in hydrological_models:
            #     from utils.data.data_utils import DataAcquisitionProcessor
            #     dap = DataAcquisitionProcessor(self.config, self.logger)
            #     dap.run_data_acquisition()
            
            self.logger.info("Model-agnostic preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model-agnostic preprocessing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
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
        
        return status