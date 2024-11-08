from pathlib import Path
import sys
from typing import Optional, Dict, Tuple
from datetime import datetime
import subprocess
import pandas as pd # type: ignore
import rasterio # type: ignore
import numpy as np # type: ignore
import shutil
from scipy import stats # type: ignore
import argparse
import geopandas as gpd # type: ignore

sys.path.append(str(Path(__file__).resolve().parent))

from utils.dataHandling_utils.data_utils import ProjectInitialisation, ObservedDataProcessor, BenchmarkPreprocessor # type: ignore  
from utils.dataHandling_utils.data_acquisition_utils import gistoolRunner # type: ignore
from utils.dataHandling_utils.data_acquisition_utils import datatoolRunner # type: ignore
from utils.dataHandling_utils.agnosticPreProcessor_util import forcingResampler, geospatialStatistics # type: ignore
from utils.dataHandling_utils.specificPreProcessor_util import SummaPreProcessor_spatial, flashPreProcessor # type: ignore
from utils.geospatial_utils.geofabric_utils import GeofabricSubsetter, GeofabricDelineator, LumpedWatershedDelineator # type: ignore
from utils.geospatial_utils.discretization_utils import DomainDiscretizer # type: ignore
from utils.models_utils.mizuroute_utils import MizuRoutePreProcessor # type: ignore
from utils.models_utils.model_utils import SummaRunner, MizuRouteRunner, FLASH # type: ignore
from utils.report_utils.reporting_utils import VisualizationReporter # type: ignore
from utils.configHandling_utils.config_utils import ConfigManager # type: ignore
from utils.configHandling_utils.logging_utils import setup_logger, get_function_logger # type: ignore
from utils.evaluation_util.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore
from utils.optimization_utils.ostrich_util import OstrichOptimizer # type: ignore

class CONFLUENCE:

    """
    CONFLUENCE: Community Optimization and Numerical Framework for Large-domain Understanding of 
    Environmental Networks and Computational Exploration

    This class serves as the main interface for the CONFLUENCE hydrological modeling platform. 
    It integrates various components for data management, model setup, optimization, 
    uncertainty analysis, forecasting, visualization, and workflow management.

    The platform is designed to facilitate comprehensive hydrological modeling and analysis
    across various scales and regions, supporting multiple models and analysis techniques.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the CONFLUENCE system
        logger (logging.Logger): Logger for the CONFLUENCE system
        data_manager (DataAcquisitionProcessor): Handles data acquisition and processing
        model_manager (ModelSetupInitializer): Manages model setup and initialization
        optimizer (OptimizationCalibrator): Handles model optimization and calibration
        uncertainty_analyzer (UncertaintyQuantifier): Performs uncertainty analysis
        forecaster (ForecastingEngine): Generates hydrological forecasts
        visualizer (VisualizationReporter): Creates visualizations and reports
        workflow_manager (WorkflowManager): Manages modeling workflows

    """

    def __init__(self, config):
        """
        Initialize the CONFLUENCE system.

        Args:
            config (Config): Configuration object containing optimization settings for the CONFLUENCE system
        """
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.config
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.setup_logging()
        self.project_initialisation = ProjectInitialisation(self.config, self.logger)

    def setup_logging(self):
        log_dir = self.project_dir / f"_workLog_{self.config.get('DOMAIN_NAME')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_general_{self.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_general', log_file)

    @get_function_logger
    def run_workflow(self):
        """
        Execute the complete CONFLUENCE modeling workflow.

        This method orchestrates the entire modeling process from project setup through 
        to analysis and visualization. It follows these key steps:
        1. Project initialization and setup
        2. Domain definition and discretization
        3. Data acquisition and preprocessing
        4. Model execution
        5. Analysis and visualization
        6. Optional: Optimization and sensitivity analysis

        Each step is executed conditionally based on output existence checks unless 
        FORCE_RUN_ALL_STEPS is True in the configuration.

        Returns:
            None

        Raises:
            ConfigurationError: If required configuration parameters are missing
            DataAcquisitionError: If data acquisition or preprocessing fails
            ModelExecutionError: If model execution fails
            AnalysisError: If post-processing analysis fails

        Example:
            >>> confluence = CONFLUENCE(config_path)
            >>> confluence.run_workflow()
        """
        self.logger.info("Starting CONFLUENCE workflow")
        
        # Check if we should force run all steps
        force_run = self.config.get('FORCE_RUN_ALL_STEPS', False)
        
        # Define the workflow steps and their output checks
        workflow_steps = [
            (self.setup_project, (self.project_dir / 'catchment').exists),
            (self.create_pourPoint, lambda: (self.project_dir / "shapefiles" / "pour_point" / f"{self.domain_name}_pourPoint.shp").exists()),
            (self.acquire_attributes, lambda: (self.project_dir / "attributes" / "elevation" / "dem" / f"domain_{self.domain_name}_elv.tif").exists()),
            (self.define_domain, lambda: (self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()),
            (self.discretize_domain, lambda: (self.project_dir / "shapefiles" / "catchment" / f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()),
            (self.acquire_forcings, lambda: (self.project_dir / "forcing" / "raw_data").exists()),
            (self.model_agnostic_pre_processing, lambda: (self.project_dir / "forcing" / "basin_averaged_data").exists()),
            (self.model_specific_pre_processing, lambda: (self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL']}").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL')}" / f"{self.config.get('EXPERIMENT_ID')}_timestep.nc").exists()),
            (self.process_observed_data, lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv").exists()),
            (self.visualise_model_output, lambda: (self.project_dir / "plots" / "results" / "streamflow_comparison.png").exists()),
            (self.run_benchmarking, lambda: (self.project_dir / "evaluation" / "benchmarking" / "benchmark_scores.csv").exists()),
            (self.calibrate_model, lambda: (self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_parallel_iteration_results.csv").exists()),
            (self.run_sensitivity_analysis, lambda: (self.project_dir / "plots" / "sensitivity_analysis" / "all_sensitivity_results.csv").exists()),
            (self.run_decision_analysis, lambda: (self.project_dir / "optimisation " / f"{self.config.get('EXPERIMENT_ID')}_model_decisions_comparison.csv2").exists()),    
        ]
        
        for step_func, check_func in workflow_steps:
            step_name = step_func.__name__
            if force_run or not check_func():
                self.logger.info(f"Running step: {step_name}")
                try:
                    step_func()
                except Exception as e:
                    self.logger.error(f"Error during {step_name}: {str(e)}")
                    raise
            else:
                self.logger.info(f"Skipping step {step_name} as output already exists")

        self.logger.info("CONFLUENCE workflow completed")

    @get_function_logger
    def setup_project(self) -> Path:
        """
        Initialize and set up the CONFLUENCE project directory structure.
        
        This method creates the necessary directory structure for a new CONFLUENCE project,
        including subdirectories for:
        - Shapefiles (catchment, pour points, river networks)
        - Observations (streamflow, meteorological data)
        - Model inputs and outputs
        - Analysis results
        - Documentation
        
        Returns:
            Path: Path to the created project directory
            
        Raises:
            ValueError: If required configuration parameters are missing
            OSError: If directory creation fails or if there are permission issues
            
        Example:
            >>> project_dir = confluence.setup_project()
            >>> print(f"Project created at: {project_dir}")
        """
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        try:
            if not self.domain_name:
                raise ValueError("Domain name not specified in configuration")
                
            if not self.data_dir.exists():
                raise OSError(f"Data directory does not exist: {self.data_dir}")
                
            project_dir = self.project_initialisation.setup_project()
            
            # Verify critical directories were created
            required_dirs = ['shapefiles', 'observations', 'documentation']
            missing_dirs = [dir for dir in required_dirs 
                        if not (project_dir / dir).exists()]
            
            if missing_dirs:
                raise OSError(f"Failed to create required directories: {missing_dirs}")
                
            self.logger.info(f"Project directory created at: {project_dir}")
            self.logger.debug("Created directory structure:")
            for path in project_dir.rglob('*'):
                if path.is_dir():
                    self.logger.debug(f"  {path.relative_to(project_dir)}/")
            
            return project_dir
            
        except OSError as e:
            self.logger.error(f"File system error during project setup: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during project setup: {str(e)}")
            raise

    @get_function_logger
    def create_pourPoint(self) -> Optional[Path]:
        """
        Create or validate the pour point shapefile for the watershed.
        
        This method either creates a new pour point shapefile using coordinates specified
        in the configuration, or validates an existing pour point shapefile provided by
        the user. The pour point represents the outlet of the watershed and is critical
        for proper watershed delineation.
        
        The pour point can be specified in two ways:
        1. By coordinates in the config (POUR_POINT_COORDS: "lat/lon")
        2. By providing a pre-existing shapefile
        
        Returns:
            Optional[Path]: Path to the created/validated pour point shapefile,
                        or None if using user-provided shapefile
            
        Raises:
            ValueError: If pour point coordinates are invalid or incorrectly formatted
            FileNotFoundError: If user-provided shapefile doesn't exist
            RuntimeError: If shapefile creation fails
            
        Example:
            >>> pour_point_path = confluence.create_pourPoint()
            >>> if pour_point_path:
            >>>     print(f"Pour point shapefile created at: {pour_point_path}")
        """
        try:
            pour_point_coords = self.config.get('POUR_POINT_COORDS', 'default')
            
            if pour_point_coords.lower() == 'default':
                self.logger.info("Using user-provided pour point shapefile")
                # Validate user-provided shapefile exists
                user_shp = self.project_dir / "shapefiles" / "pour_point" / f"{self.domain_name}_pourPoint.shp"
                if not user_shp.exists():
                    raise FileNotFoundError(
                        f"User-provided pour point shapefile not found at: {user_shp}"
                    )
                return None
                
            # Validate coordinates format
            try:
                lat, lon = map(float, pour_point_coords.split('/'))
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    raise ValueError(
                        f"Coordinates out of valid range: lat must be [-90,90], lon must be [-180,180]"
                    )
            except ValueError as e:
                raise ValueError(
                    f"Invalid pour point coordinates format: {pour_point_coords}. "
                    "Expected format: 'lat/lon' (e.g., '45.5/-122.5')"
                ) from e
                
            output_file = self.project_initialisation.create_pourPoint()
            
            if output_file and output_file.exists():
                self.logger.info(f"Pour point shapefile created successfully: {output_file}")
                return output_file
            else:
                raise RuntimeError("Pour point shapefile creation failed")
                
        except (ValueError, FileNotFoundError) as e:
            self.logger.error(f"Pour point creation error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in pour point creation: {str(e)}")
            raise

    @get_function_logger
    def acquire_attributes(self) -> Dict[str, Path]:
        """
        Acquire and process terrain, soil, and land cover attributes for the domain.
        
        This method downloads and processes three types of spatial attributes:
        1. Elevation data (DEM) from MERIT-Hydro
        2. Land cover data from MODIS (with temporal aggregation if multiple years)
        3. Soil classification data
        
        All data is clipped to the domain's bounding box coordinates and stored in the
        appropriate project subdirectories.
        
        Returns:
            Dict[str, Path]: Dictionary containing paths to processed attribute files:
                {
                    'dem': Path to processed DEM,
                    'landcover': Path to processed land cover,
                    'soilclass': Path to processed soil classification
                }
                
        Raises:
            ValueError: If bounding box coordinates are invalid
            OSError: If directories cannot be created or data cannot be saved
            RuntimeError: If data acquisition or processing fails
            
        Example:
            >>> attribute_paths = confluence.acquire_attributes()
            >>> dem_path = attribute_paths['dem']
            >>> print(f"DEM saved to: {dem_path}")
        """
        try:
            # Create and validate attribute directories
            dirs = {
                'dem': self.project_dir / 'attributes' / 'elevation' / 'dem',
                'soil': self.project_dir / 'attributes' / 'soilclass',
                'land': self.project_dir / 'attributes' / 'landclass'
            }
            
            for name, dir_path in dirs.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                if not dir_path.exists():
                    raise OSError(f"Failed to create {name} directory: {dir_path}")

            # Initialize gistool runner
            gr = gistoolRunner(self.config, self.logger)
            
            # Parse and validate bounding box
            try:
                bbox = self.config['BOUNDING_BOX_COORDS'].split('/')
                if len(bbox) != 4:
                    raise ValueError(
                        "Bounding box must have exactly 4 coordinates "
                        "(lat_max/lon_min/lat_min/lon_max)"
                    )
                
                # Convert to floats and validate ranges
                lat_max, lon_min, lat_min, lon_max = map(float, bbox)
                if not (-90 <= lat_min <= lat_max <= 90):
                    raise ValueError(f"Invalid latitude range: {lat_min} to {lat_max}")
                if not (-180 <= lon_min <= lon_max <= 180):
                    raise ValueError(f"Invalid longitude range: {lon_min} to {lon_max}")
                    
                latlims = f"{bbox[2]},{bbox[0]}"  # min_lat,max_lat
                lonlims = f"{bbox[1]},{bbox[3]}"  # min_lon,max_lon
                
            except (IndexError, ValueError) as e:
                raise ValueError(
                    "Invalid bounding box configuration. Expected format: "
                    "'lat_max/lon_min/lat_min/lon_max'"
                ) from e

            # Dictionary to store output paths
            output_paths = {}

            # Acquire elevation data
            self.logger.info("Acquiring elevation data from MERIT-Hydro")
            gistool_command_elevation = gr.create_gistool_command(
                dataset='MERIT-Hydro',
                output_dir=dirs['dem'],
                lat_lims=latlims,
                lon_lims=lonlims,
                variables='elv'
            )
            
            gr.execute_gistool_command(gistool_command_elevation)
                
            output_paths['dem'] = dirs['dem'] / f"domain_{self.domain_name}_elv.tif"

            # Acquire and process land cover data
            self.logger.info("Acquiring land cover data from MODIS")
            start_year = 2001
            end_year = 2020
            modis_var = "MCD12Q1.061"
            
            gistool_command_landcover = gr.create_gistool_command(
                dataset='MODIS',
                output_dir=dirs['land'],
                lat_lims=latlims,
                lon_lims=lonlims,
                variables=modis_var,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-01-01"
            )
            if not gr.execute_gistool_command(gistool_command_landcover):
                raise RuntimeError("Failed to acquire land cover data")

            # Calculate temporal mode for land cover if multiple years
            land_name = self.config.get('LAND_CLASS_NAME', 
                                        f"domain_{self.domain_name}_land_classes.tif")
            if start_year != end_year:
                self.logger.info(f"Calculating land cover mode for years {start_year}-{end_year}")
                input_dir = dirs['land'] / modis_var
                output_file = dirs['land'] / land_name
                try:
                    self.calculate_landcover_mode(input_dir, output_file, start_year, end_year)
                except Exception as e:
                    raise RuntimeError(f"Failed to calculate land cover mode: {str(e)}") from e
                    
            output_paths['landcover'] = dirs['land'] / land_name

            # Acquire soil classification data
            self.logger.info("Acquiring soil classification data")
            gistool_command_soilclass = gr.create_gistool_command(
                dataset='soil_class',
                output_dir=dirs['soil'],
                lat_lims=latlims,
                lon_lims=lonlims,
                variables='soil_classes'
            )
            if not gr.execute_gistool_command(gistool_command_soilclass):
                raise RuntimeError("Failed to acquire soil classification data")
                
            output_paths['soilclass'] = dirs['soil'] / 'soil_classes.tif'

            # Validate all outputs exist
            for name, path in output_paths.items():
                if not path.exists():
                    raise RuntimeError(f"Failed to find output file for {name}: {path}")
                try:
                    with rasterio.open(path) as src:
                        if src.read(1).size == 0:
                            raise RuntimeError(f"Empty raster file for {name}: {path}")
                except rasterio.errors.RasterioIOError as e:
                    raise RuntimeError(f"Invalid raster file for {name}: {path}") from e

            self.logger.info("Successfully acquired all attribute data")
            return output_paths

        except ValueError as e:
            self.logger.error(f"Configuration error in acquire_attributes: {str(e)}")
            raise
        except OSError as e:
            self.logger.error(f"File system error in acquire_attributes: {str(e)}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Processing error in acquire_attributes: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in acquire_attributes: {str(e)}")
            raise

    @get_function_logger
    def define_domain(self) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
        """
        Define the spatial domain for hydrological modeling using the specified method.
        
        This method handles three different approaches to domain definition:
        1. subset: Extract a subset from an existing geofabric
        2. lumped: Create a lumped watershed representation
        3. delineate: Delineate a new watershed from elevation data
        
        The method to use is specified by DOMAIN_DEFINITION_METHOD in the configuration.
        
        Returns:
            Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]: 
                A tuple containing (river_basins, river_network) GeoDataFrames.
                Returns (None, None) if the process fails.
        
        Raises:
            ValueError: If an invalid domain definition method is specified
            RuntimeError: If the domain definition process fails
            OSError: If required input files are missing or output cannot be saved
            
        Example:
            >>> river_basins, river_network = confluence.define_domain()
            >>> if river_basins is not None:
            >>>     print(f"Created {len(river_basins)} river basins")
        """
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        self.logger.info(f"Defining domain using method: {domain_method}")
        
        try:
            if domain_method not in ['subset', 'lumped', 'delineate']:
                raise ValueError(
                    f"Invalid domain definition method: {domain_method}. "
                    "Must be one of: subset, lumped, delineate"
                )
            
            # Create work log directory
            work_log_dir = self.data_dir / f"domain_{self.domain_name}" / "shapefiles/_workLog"
            work_log_dir.mkdir(parents=True, exist_ok=True)
            
            result = None
            if domain_method == 'subset':
                self.logger.info("Subsetting domain from existing geofabric")
                result = self.subset_geofabric(work_log_dir=work_log_dir)
                
            elif domain_method == 'lumped':
                self.logger.info("Creating lumped watershed representation")
                result = self.delineate_lumped_watershed(work_log_dir=work_log_dir)
                
            elif domain_method == 'delineate':
                self.logger.info("Delineating watershed from elevation data")
                result = self.delineate_geofabric(work_log_dir=work_log_dir)
            
            # Validate results
            if result is None or (isinstance(result, tuple) and any(r is None for r in result)):
                raise RuntimeError(f"Domain definition failed using method: {domain_method}")
                
            # If we got here, result should be a tuple of (river_basins, river_network)
            river_basins, river_network = result
            
            # Validate output files exist
            output_files = {
                'river_basins': self.project_dir / "shapefiles" / "river_basins" / 
                            f"{self.domain_name}_riverBasins_{domain_method}.shp",
                'river_network': self.project_dir / "shapefiles" / "river_network" / 
                            f"{self.domain_name}_riverNetwork_{domain_method}.shp"
            }
            
            for name, path in output_files.items():
                if not path.exists():
                    raise RuntimeError(f"Failed to create {name} shapefile at: {path}")
            
            self.logger.info(
                f"Domain definition completed successfully using {domain_method} method\n"
                f"Created {len(river_basins)} river basins and {len(river_network)} river segments"
            )
            
            return river_basins, river_network
            
        except ValueError as e:
            self.logger.error(f"Configuration error in define_domain: {str(e)}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Processing error in define_domain: {str(e)}")
            raise
        except OSError as e:
            self.logger.error(f"File system error in define_domain: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in define_domain: {str(e)}")
            raise

    @get_function_logger
    def discretize_domain(self) -> Optional[gpd.GeoDataFrame]:
        """
        Discretize the domain into Hydrologic Response Units (HRUs).
        
        This method subdivides the domain into HRUs using one of several methods:
        - elevation: Based on elevation bands
        - soilclass: Based on soil classification
        - landclass: Based on land cover types
        - radiation: Based on radiation characteristics
        - GRUs: Using Grouped Response Units
        - combined: Using multiple characteristics
        
        The discretization method is specified by DOMAIN_DISCRETIZATION in the configuration.
        
        Returns:
            Optional[gpd.GeoDataFrame]: GeoDataFrame containing the HRUs.
                Returns None if discretization fails.
                
        Raises:
            ValueError: If invalid discretization method is specified
            RuntimeError: If discretization process fails
            OSError: If required input files are missing or output cannot be saved
            
        Example:
            >>> hrus = confluence.discretize_domain()
            >>> if hrus is not None:
            >>>     print(f"Created {len(hrus)} Hydrologic Response Units")
        """
        try:
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
            discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
            
            self.logger.info(
                f"Discretizing domain using method: {discretization_method}\n"
                f"Domain definition method: {domain_method}"
            )
            
            # Initialize domain discretizer
            domain_discretizer = DomainDiscretizer(self.config, self.logger)
            
            # Perform discretization
            hru_shapefile = domain_discretizer.discretize_domain()
            
            # Handle different return types
            if isinstance(hru_shapefile, (pd.Series, pd.DataFrame)):
                if hru_shapefile.empty:
                    raise RuntimeError("Domain discretization produced empty result")
                self.logger.info("Domain discretized successfully into multiple HRU sets:")
                for index, shapefile in hru_shapefile.items():
                    self.logger.info(f"  {index}: {shapefile}")
                result = hru_shapefile
                
            elif hru_shapefile:
                self.logger.info(f"Domain discretized successfully. HRU shapefile: {hru_shapefile}")
                result = hru_shapefile
                
            else:
                raise RuntimeError("Domain discretization failed to produce output")
                
            # Validate output exists
            output_path = self.project_dir / "shapefiles" / "catchment" / \
                        f"{self.domain_name}_HRUs_{discretization_method}.shp"
            
            if not output_path.exists():
                raise RuntimeError(f"HRU shapefile not found at expected location: {output_path}")
                
            # Load and validate the shapefile
            try:
                hrus = gpd.read_file(output_path)
                if len(hrus) == 0:
                    raise RuntimeError("HRU shapefile contains no features")
                    
                required_columns = ['HRU_ID', 'GRU_ID', 'HRU_area']
                missing_columns = [col for col in required_columns if col not in hrus.columns]
                if missing_columns:
                    raise RuntimeError(f"HRU shapefile missing required columns: {missing_columns}")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to validate HRU shapefile: {str(e)}") from e
                
            self.logger.info(
                f"Domain discretization completed successfully:\n"
                f"- Created {len(hrus)} Hydrologic Response Units\n"
                f"- Total area: {hrus['HRU_area'].sum():,.2f} m²\n"
                f"- Number of GRUs: {len(hrus['GRU_ID'].unique())}"
            )
            
            return result
            
        except ValueError as e:
            self.logger.error(f"Configuration error in discretize_domain: {str(e)}")
            raise
        except RuntimeError as e:
            self.logger.error(f"Processing error in discretize_domain: {str(e)}")
            raise
        except OSError as e:
            self.logger.error(f"File system error in discretize_domain: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in discretize_domain: {str(e)}")
            raise

    @get_function_logger
    def acquire_forcings(self):
        # Initialize datatoolRunner class
        dr = datatoolRunner(self.config, self.logger)

        # Data directory
        raw_data_dir = self.project_dir / 'forcing' / 'raw_data'

        # Make sure the directory exists
        raw_data_dir.mkdir(parents = True, exist_ok = True)

        # Get lat and lon lims
        bbox = self.config['BOUNDING_BOX_COORDS'].split('/')
        latlims = f"{bbox[2]},{bbox[0]}"
        lonlims = f"{bbox[1]},{bbox[3]}"

        # Create the gistool command
        datatool_command = dr.create_datatool_command(dataset = self.config['FORCING_DATASET'], output_dir = raw_data_dir, lat_lims = latlims, lon_lims = lonlims, variables = self.config['FORCING_VARIABLES'], start_date = self.config['EXPERIMENT_TIME_START'], end_date = self.config['EXPERIMENT_TIME_END'])
        dr.execute_datatool_command(datatool_command)

    @get_function_logger
    def model_agnostic_pre_processing(self):
        # Data directoris
        raw_data_dir = self.project_dir / 'forcing' / 'raw_data'
        basin_averaged_data = self.project_dir / 'forcing' / 'basin_averaged_data'
        catchment_intersection_dir = self.project_dir / 'shapefiles' / 'catchment_intersection'

        # Make sure the new directories exists
        basin_averaged_data.mkdir(parents = True, exist_ok = True)
        catchment_intersection_dir.mkdir(parents = True, exist_ok = True)

         # Initialize geospatialStatistics class
        gs = geospatialStatistics(self.config, self.logger)

        # Run resampling
        gs.run_statistics()
        
        # Initialize forcingReampler class
        fr = forcingResampler(self.config, self.logger)

        # Run resampling
        fr.run_resampling()
       
    @get_function_logger
    def model_specific_pre_processing(self):
        # Data directoris
        model_input_dir = self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL']}_input"

        # Make sure the new directories exists
        model_input_dir.mkdir(parents = True, exist_ok = True)

        if self.config['HYDROLOGICAL_MODEL'] == 'SUMMA':
            ssp = SummaPreProcessor_spatial(self.config, self.logger)
            ssp.run_preprocessing()

            mp = MizuRoutePreProcessor(self.config,self.logger)
            mp.run_preprocessing()

    @get_function_logger
    def run_models(self):
        self.logger.info("Starting model runs")
        
        if self.config.get('HYDROLOGICAL_MODEL') == 'SUMMA':
            summa_runner = SummaRunner(self.config, self.logger)
            mizuroute_runner = MizuRouteRunner(self.config, self.logger)

            try:
                summa_runner.run_summa()
                if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                    mizuroute_runner.run_mizuroute()
                self.logger.info("SUMMA/MIZUROUTE model runs completed successfully")
            except Exception as e:
                self.logger.error(f"Error during SUMMA/MIZUROUTE model runs: {str(e)}")

        elif self.config.get('HYDROLOGICAL_MODEL') == 'FLASH':
            try:
                flash_model = FLASH(self.config, self.logger)
                flash_model.run_flash()
                self.logger.info("FLASH model run completed successfully")
            except Exception as e:
                self.logger.error(f"Error during FLASH model run: {str(e)}")

        else:
            self.logger.error(f"Unknown hydrological model: {self.config.get('HYDROLOGICAL_MODEL')}")

        self.logger.info("Model runs completed")

    @get_function_logger
    def visualise_model_output(self):
    
        # Plot streamflow comparison
        self.logger.info('Starting model output visualisation')
        visualizer = VisualizationReporter(self.config, self.logger)

        if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped':
            plot_file = visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
        else:
            visualizer.update_sim_reach_id() # Find and update the sim reach id based on the project pour point
            model_outputs = [
                (f"{self.config['HYDROLOGICAL_MODEL']}", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "mizuRoute" / f"{self.config['EXPERIMENT_ID']}.h.{self.config['FORCING_START_YEAR']}-01-01-03600.nc"))
            ]
            obs_files = [
                ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
            ]
            plot_file = visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)

        #Check if the plot was output
        if plot_file:
            self.logger.info(f"Streamflow comparison plot created: {plot_file}")
        else:
            self.logger.error("Failed to create streamflow comparison plot")

    @get_function_logger
    def process_observed_data(self):
        self.logger.info("Processing observed data")
        observed_data_processor = ObservedDataProcessor(self.config, self.logger)
        try:
            observed_data_processor.process_streamflow_data()
            self.logger.info("Observed data processing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during observed data processing: {str(e)}")
            raise

    @get_function_logger     
    def run_benchmarking(self):
        # Preprocess data for benchmarking
        preprocessor = BenchmarkPreprocessor(self.config, self.logger)
        benchmark_data = preprocessor.preprocess_benchmark_data(f"{self.config['FORCING_START_YEAR']}-01-01", f"{self.config['FORCING_END_YEAR']}-12-31")

        # Run benchmarking
        benchmarker = Benchmarker(self.config, self.logger)
        benchmark_results = benchmarker.run_benchmarking(benchmark_data, f"{self.config['FORCING_END_YEAR']}-12-31")

    @get_function_logger
    def calibrate_model(self):
        # Calibrate the model using specified method and objectives
        if self.config.get('OPTMIZATION_ALOGORITHM') == 'OSTRICH':
            self.run_ostrich_optimization()
        else:
            self.run_parallel_optimization()

    @get_function_logger
    def run_sensitivity_analysis(self):
        self.logger.info("Starting sensitivity analysis")
        sensitivity_analyzer = SensitivityAnalyzer(self.config, self.logger)
        results_file = self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_parallel_iteration_results.csv"
        
        if not results_file.exists():
            self.logger.error(f"Calibration results file not found: {results_file}")
            return
        
        if self.config.get('RUN_SENSITIVITY_ANALYSIS', True) == True:
            sensitivity_results = sensitivity_analyzer.run_sensitivity_analysis(results_file)

        sensitivity_results = sensitivity_analyzer.run_sensitivity_analysis(results_file)
        self.logger.info("Sensitivity analysis completed")
        return sensitivity_results

    @get_function_logger  
    def run_decision_analysis(self):
        self.logger.info("Starting decision analysis")
        decision_analyzer = DecisionAnalyzer(self.config, self.logger)
        
        results_file, best_combinations = decision_analyzer.run_full_analysis()
        
        self.logger.info("Decision analysis completed")
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info("Best combinations for each metric:")
        for metric, data in best_combinations.items():
            self.logger.info(f"  {metric}: score = {data['score']:.3f}")
        
        return results_file, best_combinations

    @get_function_logger
    def subset_geofabric(self):
        self.logger.info("Starting geofabric subsetting process")

        # Create GeofabricSubsetter instance
        subsetter = GeofabricSubsetter(self.config, self.logger)
        
        try:
            subset_basins, subset_rivers = subsetter.subset_geofabric()
            self.logger.info("Geofabric subsetting completed successfully")
            return subset_basins, subset_rivers
        except Exception as e:
            self.logger.error(f"Error during geofabric subsetting: {str(e)}")
            return None

    @get_function_logger
    def delineate_lumped_watershed(self):
        self.logger.info("Starting geofabric lumped delineation")
        try:
            delineator = LumpedWatershedDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_lumped_watershed()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None

    @get_function_logger
    def delineate_geofabric(self):
        self.logger.info("Starting geofabric delineation")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_geofabric()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None

    def run_parallel_optimization(self):
        
        config_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_config_files' / 'config_active.yaml'

        if shutil.which("srun"):
            run_command = "srun"
        elif shutil.which("mpirun"):
            run_command = "mpirun"


        cmd = [
            run_command,
            '-n', str(self.config.get('MPI_PROCESSES')),
            'python',
            str(Path(__file__).parent / 'utils' / 'optimization_utils' / 'parallel_parameter_estimation.py'), 
            str(config_path)
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running parallel optimization: {e}")

    def run_ostrich_optimization(self):
        optimizer = OstrichOptimizer(self.config, self.logger)
        optimizer.run_optimization()

    def calculate_landcover_mode(self, input_dir, output_file, start_year, end_year):
        # List all the geotiff files for the years we're interested in
        geotiff_files = [input_dir / f"domain_{self.config['DOMAIN_NAME']}_{year}.tif" for year in range(start_year, end_year + 1)]
        
        # Read the first file to get metadata
        with rasterio.open(geotiff_files[0]) as src:
            meta = src.meta
            shape = src.shape
        
        # Initialize an array to store all the data
        all_data = np.zeros((len(geotiff_files), *shape), dtype=np.int16)
        
        # Read all the geotiffs into the array
        for i, file in enumerate(geotiff_files):
            with rasterio.open(file) as src:
                all_data[i] = src.read(1)
        
        # Calculate the mode along the time axis
        mode_data, _ = stats.mode(all_data, axis=0)
        mode_data = mode_data.astype(np.int16).squeeze()
        
        # Update metadata for output
        meta.update(count=1, dtype='int16')
        
        # Write the result
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(mode_data, 1)
        
        print(f"Mode calculation complete. Result saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Run CONFLUENCE workflow')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file. If not provided, uses default config_active.yaml')
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
    else:
        config_path = Path(__file__).parent / '0_config_files' / 'config_active.yaml'
        if not config_path.exists():
            print(f"Error: Default configuration file not found: {config_path}")
            sys.exit(1)

    try:
        confluence = CONFLUENCE(config_path)
        confluence.run_workflow()
    except Exception as e:
        print(f"Error running CONFLUENCE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
