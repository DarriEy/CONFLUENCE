# In utils/dataHandling_utils/data_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

from utils.data.data_utils import ObservedDataProcessor, gistoolRunner, datatoolRunner # type: ignore 
from utils.data.agnosticPreProcessor import forcingResampler, geospatialStatistics # type: ignore 
from utils.data.variable_utils import VariableHandler # type: ignore 
from utils.geospatial.raster_utils import calculate_landcover_mode # type: ignore 

class DataManager:
    """Manages all data acquisition and preprocessing operations."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Data Manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Initialize variable handler
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        
    def acquire_attributes(self):
        """Acquire geospatial attributes including DEM, soil, and land cover data."""
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
        """Acquire elevation data using gistool."""
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
        """Acquire land cover data using gistool."""
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
        """Acquire soil class data using gistool."""
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
        """Acquire forcing data for the model simulation."""
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
        """Process observed streamflow data."""
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
        """Run model-agnostic preprocessing including basin averaging and resampling."""
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
        
        Returns:
            True if all required directories exist, False otherwise
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
        
        Returns:
            Dictionary containing data status information
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
    