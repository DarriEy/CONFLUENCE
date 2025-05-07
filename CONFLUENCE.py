from pathlib import Path
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd # type: ignore
import rasterio # type: ignore
import numpy as np # type: ignore
import shutil
from scipy import stats # type: ignore
import argparse

# Import CONFLUENCE utility functions
sys.path.append(str(Path(__file__).resolve().parent))

# Data and config management utilities 
from utils.dataHandling_utils.data_utils import ProjectInitialisation, ObservedDataProcessor, BenchmarkPreprocessor, DataAcquisitionProcessor # type: ignore  
from utils.dataHandling_utils.attribute_processing_util import attributeProcessor # type: ignore
from utils.dataHandling_utils.data_acquisition_utils import gistoolRunner, datatoolRunner # type: ignore
from utils.dataHandling_utils.agnosticPreProcessor_util import forcingResampler, geospatialStatistics # type: ignore
from utils.dataHandling_utils.variable_utils import VariableHandler # type: ignore
from utils.configHandling_utils.config_utils import ConfigManager # type: ignore
from utils.configHandling_utils.logging_utils import setup_logger, get_function_logger, log_configuration # type: ignore

# Domain definition utilities
from utils.geospatial_utils.geofabric_utils import GeofabricSubsetter, GeofabricDelineator, LumpedWatershedDelineator # type: ignore
from utils.geospatial_utils.discretization_utils import DomainDiscretizer # type: ignore

# Model specific utilities
from utils.models_utils.mizuroute_utils import MizuRoutePreProcessor, MizuRouteRunner # type: ignore
from utils.models_utils.summa_utils import SUMMAPostprocessor, SummaRunner, SummaPreProcessor # type: ignore
from utils.models_utils.fuse_utils import FUSEPreProcessor, FUSERunner, FuseDecisionAnalyzer, FUSEPostprocessor # type: ignore
from utils.models_utils.gr_utils import GRPreProcessor, GRRunner, GRPostprocessor # type: ignore
from utils.models_utils.flash_utils import FLASH, FLASHPostProcessor # type: ignore
from utils.models_utils.hype_utils import HYPEPreProcessor, HYPERunner, HYPEPostProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPreProcessor, MESHRunner, MESHPostProcessor # type: ignore

# Evaluation utilities
#from utils.evaluation_util.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore

# Reporting utilities
from utils.report_utils.reporting_utils import VisualizationReporter # type: ignore
from utils.report_utils.result_vizualisation_utils import BenchmarkVizualiser, TimeseriesVisualizer # type: ignore

# Optimisation utilities
from utils.emulation_utils.random_forest_emulator import RandomForestEmulator # type: ignore
from utils.emulation_utils.single_sample_emulator import SingleSampleEmulator # type: ignore
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
        self.config_path = config
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.config
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.evaluation_dir = self.project_dir / 'evaluation'
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = self.config.get('EXPERIMENT_ID')

        self.setup_logging()
       # Log configuration file using the original config path
        log_dir = self.project_dir / f"_workLog_{self.domain_name}"
        self.config_log_file = log_configuration(config, log_dir, self.domain_name)

        self.project_initialisation = ProjectInitialisation(self.config, self.logger)
        self.reporter = VisualizationReporter(self.config, self.logger)
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')

    def setup_logging(self):
        log_dir = self.project_dir / f"_workLog_{self.config.get('DOMAIN_NAME')}"
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_general_{self.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_general', log_file)        

    @get_function_logger
    def run_workflow(self):
        self.logger.info("Starting CONFLUENCE workflow")
        
        # Check if we should force run all steps
        force_run = self.config.get('FORCE_RUN_ALL_STEPS', False)
        
        # Define the workflow steps and their output checks
        workflow_steps = [
            # Initiate project
            (self.setup_project, (self.project_dir / 'catchment').exists),
            
            # Geospatial domain definition and analysis
            (self.create_pourPoint, lambda: (self.project_dir / "shapefiles" / "pour_point" / f"{self.domain_name}_pourPoint.shp").exists()),
            (self.acquire_attributes, lambda: (self.project_dir / "attributes" / "soilclass" / f"domain_{self.domain_name}_soil_classes.tif").exists()),
            (self.define_domain, lambda: (self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()),
            (self.plot_domain, lambda: (self.project_dir / "plots" / "domain" / 'domain_map.png').exists()),
            (self.discretize_domain, lambda: (self.project_dir / "shapefiles" / "catchment" / f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()),
            (self.plot_discretised_domain, lambda: (self.project_dir / "plots" / "discretization" / f"domain_discretization_{self.config['DOMAIN_DISCRETIZATION']}.png").exists()),
            
            # Model agnostic data pre- processing
            (self.process_observed_data, lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv").exists()),
            (self.acquire_forcings, lambda: (self.project_dir / "forcing" / "raw_data").exists()),
            (self.model_agnostic_pre_processing, lambda: (self.project_dir / "forcing" / "basin_averaged_data").exists()),

            # Modesl specific processing
            (self.model_specific_pre_processing, lambda: (self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists()),
            (self.visualise_model_output, lambda: (self.project_dir / "plots" / "results" / "streamflow_comparison.png1").exists()),

            # --- Emulation and Optimization Steps ---
            (self.process_attributes, lambda: (self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv").exists()),
            (self.prepare_emulation_data, lambda: (self.project_dir / "emulation" / f"parameter_sets_{self.config.get('EXPERIMENT_ID')}.nc").exists()), 
            (self.calculate_ensemble_performance_metrics, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "ensemble_analysis" / "performance_metrics.csv").exists()),
            (self.run_random_forest_emulation, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "rf_emulation" / "optimized_parameters.csv").exists()),
            (self.run_iterative_emulation, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "final_optimized_parameters.json").exists()),
            (self.run_postprocessing, lambda: (self.project_dir / "results" / "postprocessed.csv").exists()),

            # Result analysis and optimisation
            (self.run_benchmarking, lambda: (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),
            (self.calibrate_model, lambda: (self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_parallel_iteration_results.csv").exists()),
            (self.run_decision_analysis, lambda: (self.project_dir / "optimisation " / f"{self.config.get('EXPERIMENT_ID')}_model_decisions_comparison.csv").exists()),  
            (self.run_sensitivity_analysis, lambda: (self.project_dir / "plots" / "sensitivity_analysis" / "all_sensitivity_results.csv").exists()),
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
    def setup_project(self):
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        project_dir = self.project_initialisation.setup_project()
        
        self.logger.info(f"Project directory created at: {project_dir}")
        self.logger.info(f"shapefiles directories created")
        
        return project_dir

    @get_function_logger
    def create_pourPoint(self):

                    
        if self.config.get('POUR_POINT_COORDS', 'default').lower() == 'default':
            self.logger.info("Using user-provided pour point shapefile")
            return None
        
        output_file = self.project_initialisation.create_pourPoint()
        
        if output_file:
            self.logger.info(f"Pour point shapefile created successfully: {output_file}")
        else:
            self.logger.error("Failed to create pour point shapefile")
        
        return output_file

    @get_function_logger
    def acquire_attributes(self):

        # Create attribute directories
        dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        soilclass_dir = self.project_dir / 'attributes' / 'soilclass'
        landclass_dir = self.project_dir / 'attributes' / 'landclass'

        for dir in [dem_dir, soilclass_dir, landclass_dir]: dir.mkdir(parents = True, exist_ok = True)

        # Initialize the gistool runner
        gr = gistoolRunner(self.config, self.logger)

        # Get lat and lon lims
        bbox = self.config['BOUNDING_BOX_COORDS'].split('/')
        latlims = f"{bbox[2]},{bbox[0]}"
        lonlims = f"{bbox[1]},{bbox[3]}"

        # Create the gistool command for elevation 
        gistool_command_elevation = gr.create_gistool_command(dataset = 'MERIT-Hydro', output_dir = dem_dir, lat_lims = latlims, lon_lims = lonlims, variables = 'elv')
        gr.execute_gistool_command(gistool_command_elevation)

        #Acquire landcover data, first we define which years we should acquire the data for
        start_year = 2001
        end_year = 2020

        #Select which MODIS dataset to use
        modis_var = "MCD12Q1.006"

        # Create the gistool command for landcover
        gistool_command_landcover = gr.create_gistool_command(dataset = 'MODIS', output_dir = landclass_dir, lat_lims = latlims, lon_lims = lonlims, variables = modis_var, start_date=f"{start_year}-01-01", end_date=f"{end_year}-01-01")
        gr.execute_gistool_command(gistool_command_landcover)

        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"

        # if we selected a range of years, we need to calculate the mode of the landcover
        if start_year != end_year:
            input_dir = landclass_dir / modis_var
            output_file = landclass_dir / land_name
    
            self.calculate_landcover_mode(input_dir, output_file, start_year, end_year)

        # Create the gistool command for soil classes
        gistool_command_soilclass = gr.create_gistool_command(dataset = 'soil_class', output_dir = soilclass_dir, lat_lims = latlims, lon_lims = lonlims, variables = 'soil_classes')
        gr.execute_gistool_command(gistool_command_soilclass)

    @get_function_logger
    def process_attributes(self):
        """
        Process catchment characteristic attributes.
        
        This method extracts and calculates various catchment attributes including
        soil, topography, climate, and land cover properties, saving them to a
        standardized CSV file.
        """
        if self.config.get('PROCESS_CHARACTERISTIC_ATTRIBUTES', False):
            self.logger.info("Processing catchment characteristics")
            
            try:

                # Create attribute processor instance - processing happens in __init__
                ap = attributeProcessor(self.config, self.logger)
                ap.process_attributes()
                
                # Check if output file was created
                output_file = self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv"
                if output_file.exists():
                    self.logger.info(f"Attribute processing completed successfully: {output_file}")
                else:
                    self.logger.warning("Attribute processing completed but no output file was found")
                    
            except Exception as e:
                self.logger.error(f"Error during attribute processing: {str(e)}")
                raise
        else:
            self.logger.info("Attribute processing skipped (PROCESS_CHARACTERISTIC_ATTRIBUTES=False)")


    @get_function_logger
    def define_domain(self):
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')

        # Skip domain definition if shapefile is provided
        if self.config.get('RIVER_BASINS_NAME') == 'default':        

            # Skip domain definition if in point mode
            if self.config.get('SPATIAL_MODE') == 'Point':
                self.delineate_point_buffer_shape()        
            elif domain_method == 'subset':
                self.subset_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
            elif domain_method == 'lumped':
                self.delineate_lumped_watershed(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
            elif domain_method == 'delineate':
                self.delineate_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
                if self.config.get('DELINEATE_COASTAL_WATERSHEDS'):
                    self.delineate_coastal(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")

            elif self.config.get('SPATIAL_MODE') == 'Point':
                self.logger.info("Spatial mode: Point simulations, delineation not required")
                return None
            else:
                self.logger.error(f"Unknown domain definition method: {domain_method}")
        else:
            self.logger.info('Shapefile provided, skipping domain definition')

    @get_function_logger
    def plot_domain(self):
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, no domain to plot")
            return None
        
        self.logger.info("Creating domain visualization...")
        domain_plot = self.reporter.plot_domain()
        if domain_plot:
            self.logger.info(f"Domain visualization created: {domain_plot}")
        else:
            self.logger.warning("Could not create domain visualization")

    @get_function_logger
    def discretize_domain(self):
        
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        domain_discretizer = DomainDiscretizer(self.config, self.logger)
        hru_shapefile = domain_discretizer.discretize_domain()

        if isinstance(hru_shapefile, pd.Series) or isinstance(hru_shapefile, pd.DataFrame):
            if not hru_shapefile.empty:
                self.logger.info(f"Domain discretized successfully. HRU shapefile(s):")
                for index, shapefile in hru_shapefile.items():
                    self.logger.info(f"  {index}: {shapefile}")
            else:
                self.logger.error("Domain discretization failed. No shapefiles were created.")
        elif hru_shapefile:
            self.logger.info(f"Domain discretized successfully. HRU shapefile: {hru_shapefile}")
        else:
            self.logger.error("Domain discretization failed.")

        self.logger.info(f"Domain to be defined using method {domain_method}")

    @get_function_logger
    def plot_discretised_domain(self):
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, discretisation not required")
            return None
        
        discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
        self.logger.info("Creating discretization visualization...")
        discretization_plot = self.reporter.plot_discretized_domain(discretization_method)
        if discretization_plot:
            self.logger.info(f"Discretization visualization created: {discretization_plot}")
        else:
            self.logger.warning("Could not create discretization visualization")

    @get_function_logger
    def acquire_forcings(self):
        if self.config.get('SPATIAL_MODE') == 'Point' and self.config.get('DATA_ACQUIRE') == 'supplied':
            self.logger.info("Spatial mode: Point simulations, data supplied")
            return None
        
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
        variables = self.config['FORCING_VARIABLES']
        if variables == 'default':
            variables = self.variable_handler.get_dataset_variables(dataset = self.config['FORCING_DATASET'])
        datatool_command = dr.create_datatool_command(dataset = self.config['FORCING_DATASET'], output_dir = raw_data_dir, lat_lims = latlims, lon_lims = lonlims, variables = variables, start_date = self.config['EXPERIMENT_TIME_START'], end_date = self.config['EXPERIMENT_TIME_END'])
        dr.execute_datatool_command(datatool_command)

    @get_function_logger
    def model_agnostic_pre_processing(self):
        
        # Data directories
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

        # Prepare run the MAF Orchestrator
        if 'MESH' in self.config.get('HYDROLOGICAL_MODEL').split(',') or 'HYPE' in self.config.get('HYDROLOGICAL_MODEL').split(','):
            dap = DataAcquisitionProcessor(self.config, self.logger)
            dap.run_data_acquisition()
       
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

    def model_specific_pre_processing(self):
        """Process the forcing data into model-specific formats."""
        self.logger.info("Starting model-specific preprocessing")
        
        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            try:
                # Create model input directory
                model_input_dir = self.project_dir / "forcing" / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Processing model: {model}")
                
                if model == 'SUMMA':
                    self.logger.info("Initializing SUMMA spatial preprocessor")
                    ssp = SummaPreProcessor(self.config, self.logger)
                    ssp.run_preprocessing()
                    
                    if self.config.get('SPATIAL_MODE') != 'Point' and self.config.get('SPATIAL_MODE') != 'Lumped':
                        self.logger.info("Initializing MizuRoute preprocessor")
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()

                elif model == 'GR':
                    gpp = GRPreProcessor(self.config, self.logger)
                    gpp.run_preprocessing()
                elif model == 'FUSE':
                    fpp = FUSEPreProcessor(self.config, self.logger)
                    fpp.run_preprocessing()
                elif model == 'HYPE':
                    hpp = HYPEPreProcessor(self.config, self.logger)
                    hpp.run_preprocessing()
                elif model == 'MESH':
                    mpp = MESHPreProcessor(self.config, self.logger)
                    mpp.run_preprocessing()
                else:
                    self.logger.warning(f"Unsupported model: {model}. No preprocessing performed.")
                    
            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                
        self.logger.info("Model-specific preprocessing completed")


    @get_function_logger
    def run_models(self):
        self.logger.info("Starting model runs")
        
        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            if model == 'SUMMA':
                summa_runner = SummaRunner(self.config, self.logger)
                mizuroute_runner = MizuRouteRunner(self.config, self.logger)

                try:
                    summa_runner.run_summa()
                    if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped' or self.config.get('SPATIAL_MODE') != 'Point':
                        mizuroute_runner.run_mizuroute()
                    self.logger.info("SUMMA/MIZUROUTE model runs completed successfully")
                except Exception as e:
                    self.logger.error(f"Error during SUMMA/MIZUROUTE model runs: {str(e)}")

            elif model == 'FLASH':
                try:
                    flash_model = FLASH(self.config, self.logger)
                    flash_model.run_flash()
                    self.logger.info("FLASH model run completed successfully")
                except Exception as e:
                    self.logger.error(f"Error during FLASH model run: {str(e)}")

            elif model == 'FUSE':
                try:
                    fr = FUSERunner(self.config, self.logger)
                    fr.run_fuse()
                except Exception as e:
                    self.logger.error(f"Error during FUSE model run: {str(e)}")

            elif model == 'GR':
                try:
                    gr = GRRunner(self.config, self.logger)
                    gr.run_gr()
                except Exception as e:
                    self.logger.error(f"Error during GR model run: {str(e)}")
            
            elif model == 'HYPE':
                try:
                    hr = HYPERunner(self.config, self.logger)
                    hr.run_hype()
                except Exception as e:
                    self.logger.error(f"Error during HYPE model run: {str(e)}")   

            elif model == 'MESH':
                mr = MESHRunner(self.config, self.logger)
                mr.run_MESH()   

            else:
                self.logger.error(f"Unknown hydrological model: {self.config.get('HYDROLOGICAL_MODEL')}")

        self.logger.info("Model runs completed")

    @get_function_logger
    def visualise_model_output(self):
        
        # Plot streamflow comparison
        self.logger.info('Starting model output visualisation')
        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            visualizer = VisualizationReporter(self.config, self.logger)
            if model == 'SUMMA':
                visualizer.plot_summa_outputs(self.config['EXPERIMENT_ID'])
                
                # Define observation files (used for both lumped and distributed cases)
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                ]
                
                if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped':
                    # For lumped model, use SUMMA output directly
                    summa_output_file = str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "SUMMA" / f"{self.config['EXPERIMENT_ID']}_timestep.nc")
                    model_outputs = [
                        (f"{model}", summa_output_file)
                    ]
                    
                    # Now we have model_outputs and obs_files defined
                    self.logger.info(f"Using lumped model output from {summa_output_file}")
                    plot_file = visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
                else:
                    # For distributed model, use MizuRoute output
                    visualizer.update_sim_reach_id(self.config_path) # Find and update the sim reach id based on the project pour point
                    model_outputs = [
                        (f"{model}", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "mizuRoute" / f"{self.config['EXPERIMENT_ID']}*.nc"))
                    ]
                    plot_file = visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                    
            elif model == 'FUSE':
                model_outputs = [
                    ("FUSE", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "FUSE" / f"{self.config['DOMAIN_NAME']}_{self.config['EXPERIMENT_ID']}_runs_best.nc"))
                ]
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                ]
                plot_file = visualizer.plot_fuse_streamflow_simulations_vs_observations(model_outputs, obs_files)
                
            elif model == 'GR':
                pass
                
            elif model == 'FLASH':
                pass

    @get_function_logger
    def prepare_emulation_data(self):
        """
        Run large sample emulation to generate spatially varying trial parameters.
        Creates parameter sets for ensemble modeling, runs simulations, and analyzes results.
        """
        self.logger.info("Starting large sample emulation")
        
        # Check if emulation is enabled in config (default to False)
        if not self.config.get('RUN_LARGE_SAMPLE_EMULATION', False):
            self.logger.info("Large sample emulation disabled in config. Skipping.")
            return None
        
        try:
            
            # Initialize the emulator
            emulator = SingleSampleEmulator(self.config, self.logger)
            
            # Run the emulation setup
            emulator_output = emulator.run_emulation_setup()
            
            if not emulator_output:
                self.logger.warning("Large sample emulation completed but no parameter file was generated.")
                return None
                
            self.logger.info(f"Large sample emulation completed successfully. Output: {emulator_output}")
            
            # Analyze the parameter space
            if self.config.get('EMULATION_ANALYZE_PARAMETERS', True):
                self.logger.info("Analyzing parameter space")
                param_analysis_dir = emulator.analyze_parameter_space(emulator_output)
                if param_analysis_dir:
                    self.logger.info(f"Parameter space analysis completed. Results saved to: {param_analysis_dir}")
            
            # Check if we should run the ensemble simulations
            run_ensemble = self.config.get('EMULATION_RUN_ENSEMBLE', False)
            
            if run_ensemble:
                self.logger.info("Starting ensemble simulations")
                ensemble_results = emulator.run_ensemble_simulations()
                success_count = sum(1 for _, success, _ in ensemble_results if success)
                self.logger.info(f"Ensemble simulations completed with {success_count} successes out of {len(ensemble_results)}")
                
                # Analyze ensemble results if we had successful runs
                if success_count > 0 and self.config.get('EMULATION_ANALYZE_ENSEMBLE', True):
                    self.logger.info("Starting ensemble results analysis")
                    analysis_dir = emulator.analyze_ensemble_results()
                    if analysis_dir:
                        self.logger.info(f"Ensemble analysis completed. Results saved to: {analysis_dir}")
                    else:
                        self.logger.warning("Ensemble analysis did not produce results.")
            
            return emulator_output
                
        except ImportError as e:
            self.logger.error(f"Could not import SingleSampleEmulator: {str(e)}. Ensure the module exists in utils/emulation_utils/")
            raise
        except Exception as e:
            self.logger.error(f"Error during large sample emulation: {str(e)}")
            raise

    @get_function_logger
    def run_iterative_emulation(self):
        """
        Run multiple iterations of the emulation and optimization procedure.
        
        This process:
        1. Takes the optimized parameters from RF as a baseline
        2. Generates new parameter sets around this optimum
        3. Runs simulations with these new parameter sets
        4. Adds the results to the training data
        5. Re-runs the RF optimization
        6. Continues until convergence criteria are met
        
        Returns:
            Path: Path to the final optimized parameters
        """
        self.logger.info("Starting iterative emulation refinement")
        
        # Initialize progress tracking
        max_iterations = self.config.get('EMULATION_ITERATIONS', 5)
        convergence_threshold = self.config.get('EMULATION_STOP_CRITERIA', 0.01)
        
        # Create output directory for iterative emulation
        iterative_dir = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "iterative_emulation"
        iterative_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a tracking file for performance metrics across iterations
        tracking_file = iterative_dir / "iteration_tracking.csv"
        tracking_data = []
        
        best_metric = float('-inf')
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            iteration += 1
            self.logger.info(f"Starting emulation iteration {iteration}/{max_iterations}")
            
            # Step 1: Generate new parameter sets
            if iteration == 1:
                # For first iteration, use the single-sample emulator to generate diverse samples
                self.logger.info("First iteration: running single-sample emulation setup")
                emulator = SingleSampleEmulator(self.config, self.logger)
                param_file = emulator.run_emulation_setup()
                
                # Run ensemble simulations
                self.logger.info("Running initial ensemble simulations")
                emulator.run_ensemble_simulations()
                
                # Analyze the ensemble results
                emulator.analyze_ensemble_results()
            else:
                # For subsequent iterations, use the previous RF optimum as a center
                # and generate new parameter sets around it
                self.logger.info(f"Iteration {iteration}: generating new parameter sets around previous optimum")
                
                # Create a baseline parameters JSON file from previous iteration
                prev_params_file = iterative_dir / f"optimized_params_iteration_{iteration-1}.json"
                if not prev_params_file.exists():
                    self.logger.error(f"Previous optimized parameters file not found: {prev_params_file}")
                    break
                    
                # Generate new parameter sets using the refined sampler
                sampling_dir = iterative_dir / f"iteration_{iteration}_sampling"
                sampling_dir.mkdir(parents=True, exist_ok=True)
                
                self._generate_refined_parameter_sets(
                    prev_params_file, 
                    sampling_dir, 
                    n_samples=self.config.get('EMULATION_NUM_SAMPLES', 100),
                    focus_factor=iteration * 0.2  # Gradually focus sampling as iterations progress
                )
                
                # Run the new ensemble simulations
                self.logger.info(f"Running ensemble simulations for iteration {iteration}")
                self._run_ensemble_from_directory(sampling_dir)
                
                # Analyze the ensemble results
                self._analyze_ensemble_results(sampling_dir)
            
            # Step 2: Run the RF emulator on the combined data
            self.logger.info(f"Training random forest model for iteration {iteration}")
            
            # Combine all previous data for RF training
            if iteration > 1:
                self._combine_performance_metrics(iterative_dir, iteration)
            
            # Run the RF emulator
            rf_output_dir = iterative_dir / f"iteration_{iteration}_rf"
            rf_output_dir.mkdir(parents=True, exist_ok=True)
            
            rf_emulator = RandomForestEmulator(self.config, self.logger)
            rf_emulator.emulation_output_dir = rf_output_dir  # Override output directory
            
            # Run the RF workflow
            rf_results = rf_emulator.run_workflow()
            
            if not rf_results:
                self.logger.error(f"Random forest emulation failed for iteration {iteration}")
                break
            
            # Extract the optimized parameters and predicted score
            optimized_params = rf_results.get('optimized_parameters', {})
            predicted_score = rf_results.get('predicted_score', float('-inf'))
            
            # Save the optimized parameters for this iteration
            import json
            with open(iterative_dir / f"optimized_params_iteration_{iteration}.json", 'w') as f:
                json.dump(optimized_params, f, indent=2)
            
            # Step 3: Run SUMMA with the optimized parameters
            self.logger.info(f"Running SUMMA with optimized parameters from iteration {iteration}")
            summa_success = rf_emulator.run_summa_with_optimal_parameters()
            
            if not summa_success:
                self.logger.error(f"SUMMA run failed for iteration {iteration}")
                predicted_actual_score = float('-inf')
            else:
                # Get the actual performance score
                metrics_file = rf_output_dir / "performance_metrics_rf_optimized.csv"
                if metrics_file.exists():
                    metrics_df = pd.read_csv(metrics_file)
                    actual_score = metrics_df[self.config.get('EMULATION_TARGET_METRIC', 'KGE')].iloc[0]
                else:
                    self.logger.warning(f"No performance metrics file found for iteration {iteration}")
                    actual_score = float('-inf')
                    
                predicted_actual_score = actual_score
            
            # Step 4: Track progress and check for convergence
            tracking_data.append({
                'Iteration': iteration,
                'PredictedScore': predicted_score,
                'ActualScore': predicted_actual_score,
                'Improvement': predicted_actual_score - best_metric if iteration > 1 else float('nan')
            })
            
            # Save tracking data
            tracking_df = pd.DataFrame(tracking_data)
            tracking_df.to_csv(tracking_file, index=False)
            
            # Create visualization of progress
            self._plot_iterative_progress(tracking_df, iterative_dir)
            
            # Check for convergence
            if iteration > 1:
                improvement = predicted_actual_score - best_metric
                self.logger.info(f"Iteration {iteration} improvement: {improvement:.6f}")
                
                if abs(improvement) < convergence_threshold:
                    self.logger.info(f"Convergence reached at iteration {iteration}: improvement {improvement:.6f} < threshold {convergence_threshold}")
                    converged = True
                elif improvement < 0:
                    self.logger.warning(f"Performance decreased at iteration {iteration}: {improvement:.6f}")
                    # Continue anyway, as this might be a local minimum
            
            # Update best metric
            if predicted_actual_score > best_metric:
                best_metric = predicted_actual_score
            
            self.logger.info(f"Completed iteration {iteration}, best {self.config.get('EMULATION_TARGET_METRIC', 'KGE')}: {best_metric:.6f}")
        
        # Determine the best iteration
        if tracking_data:
            best_iteration = max(range(1, len(tracking_data) + 1), 
                            key=lambda i: tracking_data[i-1]['ActualScore'])
            
            best_params_file = iterative_dir / f"optimized_params_iteration_{best_iteration}.json"
            final_params_file = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "final_optimized_parameters.json"
            
            # Copy the best parameters to the final location
            import shutil
            shutil.copy2(best_params_file, final_params_file)
            
            self.logger.info(f"Iterative emulation completed. Best result at iteration {best_iteration} with {self.config.get('EMULATION_TARGET_METRIC', 'KGE')}: {tracking_data[best_iteration-1]['ActualScore']:.6f}")
            return final_params_file
        else:
            self.logger.error("No tracking data available, iterative emulation failed")
            return None

    def _generate_refined_parameter_sets(self, baseline_params_file, output_dir, n_samples=100, focus_factor=0.2):
        """
        Generate refined parameter sets around a baseline parameter set.
        
        Args:
            baseline_params_file: Path to JSON file with baseline parameters
            output_dir: Directory to save the generated parameter sets
            n_samples: Number of parameter sets to generate
            focus_factor: How tightly to focus around the baseline (0-1)
        
        Returns:
            Path to the generated parameter sets file
        """
        self.logger.info(f"Generating {n_samples} refined parameter sets with focus factor {focus_factor}")
        
        import json
        import numpy as np
        import pandas as pd
        import netCDF4 as nc4
        from pathlib import Path
        
        # Load baseline parameters
        with open(baseline_params_file, 'r') as f:
            baseline_params = json.load(f)
        
        # Get parameter info from SUMMA parameter files
        local_param_info_path = self.project_dir / 'settings' / 'SUMMA' / 'localParamInfo.txt'
        basin_param_info_path = self.project_dir / 'settings' / 'SUMMA' / 'basinParamInfo.txt'
        
        # Load parameter bounds
        param_bounds = {}
        
        # Helper function to parse param info files
        def parse_param_info(file_path):
            bounds = {}
            if file_path.exists():
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('!'):
                            continue
                        parts = line.split('|')
                        if len(parts) >= 3:
                            param_name = parts[0].strip()
                            try:
                                max_val = float(parts[1].strip())
                                min_val = float(parts[2].strip())
                                bounds[param_name] = (min_val, max_val)
                            except:
                                pass
            return bounds
        
        # Get bounds for all parameters
        local_bounds = parse_param_info(local_param_info_path)
        basin_bounds = parse_param_info(basin_param_info_path)
        param_bounds = {**local_bounds, **basin_bounds}
        
        # Generate samples around baseline
        samples = []
        for i in range(n_samples):
            sample = {}
            for param, value in baseline_params.items():
                # Get bounds for this parameter
                if param in param_bounds:
                    min_val, max_val = param_bounds[param]
                    
                    # Calculate effective bounds (narrower based on focus_factor)
                    effective_min = max(min_val, value - (value - min_val) * (1 - focus_factor))
                    effective_max = min(max_val, value + (max_val - value) * (1 - focus_factor))
                    
                    # Generate random value from effective bounds
                    sample[param] = np.random.uniform(effective_min, effective_max)
                else:
                    # If bounds not found, use baseline value with small perturbation
                    sample[param] = value * np.random.uniform(0.9, 1.1)
            
            samples.append(sample)
        
        # Create trial parameters file
        params_df = pd.DataFrame(samples)
        
        # Create the directory structure and needed files
        param_sets_dir = output_dir / "parameter_sets"
        param_sets_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV for reference
        params_df.to_csv(param_sets_dir / "refined_parameter_sets.csv", index=False)
        
        # Get the attribute file for HRU info
        attribute_file_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attribute_file_path.exists():
            self.logger.error(f"Attribute file not found: {attribute_file_path}")
            return None
        
        # Create parameter sets for each HRU
        with nc4.Dataset(attribute_file_path, 'r') as att_ds:
            num_hru = len(att_ds.dimensions['hru'])
            hru_ids = att_ds.variables['hruId'][:]
            
            # Create NetCDF file
            output_nc = param_sets_dir / "parameter_sets.nc"
            
            with nc4.Dataset(output_nc, 'w', format='NETCDF4') as nc_out:
                # Define dimensions
                nc_out.createDimension('run', n_samples)
                nc_out.createDimension('hru', num_hru)
                
                # Create run index variable
                run_var = nc_out.createVariable('runIndex', 'i4', ('run',))
                run_var[:] = np.arange(n_samples)
                
                # Create HRU ID variable
                hru_id_var = nc_out.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                
                # Create parameter variables
                for param in params_df.columns:
                    param_var = nc_out.createVariable(param, 'f8', ('run', 'hru',))
                    
                    # For each run, apply parameter value to all HRUs
                    for run_idx in range(n_samples):
                        param_var[run_idx, :] = params_df[param].iloc[run_idx]
                    
                    # Add metadata
                    param_var.long_name = f"Parameter {param}"
                    if param in param_bounds:
                        param_var.min_value = param_bounds[param][0]
                        param_var.max_value = param_bounds[param][1]
        
        self.logger.info(f"Generated refined parameter sets saved to {output_nc}")
        return output_nc

    def _run_ensemble_from_directory(self, sampling_dir):
        """
        Run ensemble simulations from a directory with parameter sets.
        
        Args:
            sampling_dir: Directory containing parameter sets
        
        Returns:
            List of results
        """
        self.logger.info(f"Running ensemble simulations from {sampling_dir}")
        
        # Create run directories for each parameter set
        param_sets_nc = sampling_dir / "parameter_sets" / "parameter_sets.nc"
        
        if not param_sets_nc.exists():
            self.logger.error(f"Parameter sets file not found: {param_sets_nc}")
            return None
        
        # Load parameter sets
        import netCDF4 as nc4
        with nc4.Dataset(param_sets_nc, 'r') as ds:
            num_runs = len(ds.dimensions['run'])
        
        # Create ensemble run directories
        ensemble_dir = sampling_dir / "ensemble_runs"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual run directories
        import os
        import shutil
        from pathlib import Path
        
        # Paths to source files
        settings_path = self.project_dir / 'settings' / 'SUMMA'
        filemanager_path = settings_path / self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        # List of static files to copy to each run directory
        static_files = [
            'modelDecisions.txt',
            'outputControl.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL',
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL'
        ]
        
        # Create run directories and parameter files
        for run_idx in range(num_runs):
            # Create run directory
            run_dir = ensemble_dir / f"run_{run_idx:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Create settings directory in run directory
            run_settings_dir = run_dir / "settings" / "SUMMA"
            run_settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trial parameter file for this run
            with nc4.Dataset(param_sets_nc, 'r') as src_nc:
                hru_ids = src_nc['hruId'][:]
                
                trial_param_path = run_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                
                with nc4.Dataset(trial_param_path, 'w', format='NETCDF4') as dst_nc:
                    # Define dimensions
                    dst_nc.createDimension('hru', len(hru_ids))
                    
                    # Create hruId variable
                    hru_id_var = dst_nc.createVariable('hruId', 'i4', ('hru',))
                    hru_id_var[:] = hru_ids
                    
                    # Add each parameter with its values for this run
                    for param_name in src_nc.variables:
                        if param_name not in ['hruId', 'runIndex']:
                            param_var = dst_nc.createVariable(param_name, 'f8', ('hru',))
                            param_var[:] = src_nc[param_name][run_idx, :]
            
            # Copy the file manager and modify it for this run
            if filemanager_path.exists():
                with open(filemanager_path, 'r') as f:
                    fm_lines = f.readlines()
                
                # Path to the new file manager
                new_fm_path = run_settings_dir / os.path.basename(filemanager_path)
                
                # Modify relevant lines for this run
                with open(new_fm_path, 'w') as f:
                    for line in fm_lines:
                        if "outputPath" in line:
                            # Update output path
                            output_path = run_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "SUMMA" / ""
                            output_path_str = str(output_path).replace('\\', '/')  # Ensure forward slashes for SUMMA
                            f.write(f"outputPath           '{output_path_str}/' ! \n")
                        else:
                            f.write(line)
            
            # Copy static settings files
            for file_name in static_files:
                source_path = settings_path / file_name
                if source_path.exists():
                    dest_path = run_settings_dir / file_name
                    try:
                        shutil.copy2(source_path, dest_path)
                    except Exception as e:
                        self.logger.warning(f"Could not copy {file_name}: {str(e)}")
        
        # Run the ensemble simulations
        self.logger.info(f"Running {num_runs} ensemble simulations")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Get all run directories
        run_dirs = sorted(ensemble_dir.glob("run_*"))
        
        # Run in sequential mode for now
        # Could be extended to support parallel execution
        results = []
        for run_dir in run_dirs:
            run_name = run_dir.name
            self.logger.info(f"Running simulation for {run_name}")
            
            # Get fileManager path for this run
            run_settings_dir = run_dir / "settings" / "SUMMA"
            filemanager_path = run_settings_dir / os.path.basename(self.config.get('SETTINGS_SUMMA_FILEMANAGER'))
            
            if not filemanager_path.exists():
                self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                results.append((run_name, False, "FileManager not found"))
                continue
            
            # Ensure output directory exists
            output_dir = run_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Results directory
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Run SUMMA
                summa_command = f"{summa_exe} -m {filemanager_path}"
                summa_log_file = log_dir / f"{run_name}_summa.log"
                
                import subprocess
                with open(summa_log_file, 'w') as f:
                    process = subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                
                self.logger.info(f"Successfully completed SUMMA simulation for {run_name}")
                
                # Extract streamflow
                output_csv = self._extract_run_streamflow(run_dir, run_name)
                if output_csv and Path(output_csv).exists():
                    self.logger.info(f"Extracted streamflow for {run_name}")
                else:
                    self.logger.warning(f"Could not extract streamflow for {run_name}")
                
                results.append((run_name, True, None))
                
            except Exception as e:
                self.logger.error(f"Error running simulation for {run_name}: {str(e)}")
                results.append((run_name, False, str(e)))
        
        return results

    def _extract_run_streamflow(self, run_dir, run_name):
        """
        Extract streamflow results from model output for a specific run.
        
        For distributed domains, extracts from MizuRoute output.
        For lumped domains, extracts directly from SUMMA timestep output.
        
        Args:
            run_dir: Path to the run directory
            run_name: Name of the run (e.g., 'run_0001')
            
        Returns:
            Path: Path to the extracted streamflow CSV file
        """
        self.logger.info(f"Extracting streamflow results for {run_name}")
        
        import pandas as pd
        import xarray as xr
        import geopandas as gpd
        import numpy as np
        from pathlib import Path
        
        # Create results directory
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output CSV path
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        
        # Check if domain is lumped
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        # Get catchment area for conversion
        catchment_area = self._get_catchment_area()
        if catchment_area is None or catchment_area <= 0:
            self.logger.warning(f"Invalid catchment area: {catchment_area}, using 1.0 km² as default")
            catchment_area = 1.0 * 1e6  # Default 1 km² in m²
        
        #self.logger.info(f"Using catchment area of {catchment_area:.2f} m² for flow conversion")
        
        if is_lumped:
            # For lumped domains, extract directly from SUMMA output
            experiment_id = self.experiment_id
            summa_output_dir = run_dir / "simulations" / experiment_id / "SUMMA"
            
            # Look for all possible SUMMA output files
            timestep_files = list(summa_output_dir.glob(f"{experiment_id}*.nc"))
            
            if not timestep_files:
                self.logger.warning(f"No SUMMA output files found for {run_name} in {summa_output_dir}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Use the first timestep file found
            timestep_file = timestep_files[0]
            self.logger.info(f"Using SUMMA output file: {timestep_file}")
            
            try:
                # Open the NetCDF file
                with xr.open_dataset(timestep_file) as ds:
                    # Log available variables to help diagnose issues
                    self.logger.info(f"Available variables in SUMMA output: {list(ds.variables.keys())}")
                    
                    # Check for averageRoutedRunoff or other possible variables
                    runoff_var = None
                    for var_name in ['averageRoutedRunoff', 'outflow', 'basRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            runoff_var = var_name
                            break
                    
                    if runoff_var is None:
                        self.logger.warning(f"No suitable runoff variable found in {timestep_file}")
                        return self._create_empty_streamflow_file(run_dir, run_name)
                    
                    self.logger.info(f"Using {runoff_var} variable from SUMMA output")
                    
                    # Extract the runoff variable
                    runoff_data = ds[runoff_var]
                    
                    # Log the dimensions of the variable
                    self.logger.info(f"Dimensions of {runoff_var}: {runoff_data.dims}")
                    
                    # Convert to pandas series or dataframe
                    if 'gru' in runoff_data.dims:
                        self.logger.info(f"Found 'gru' dimension with size {ds.dims['gru']}")
                        # If multiple GRUs, sum them up
                        if ds.dims['gru'] > 1:
                            self.logger.info(f"Summing runoff across {ds.dims['gru']} GRUs")
                            runoff_series = runoff_data.sum(dim='gru').to_pandas()
                        else:
                            # Single GRU case
                            runoff_series = runoff_data.to_pandas()
                            if isinstance(runoff_series, pd.DataFrame):
                                runoff_series = runoff_series.iloc[:, 0]
                    else:
                        # Handle case without a gru dimension
                        runoff_series = runoff_data.to_pandas()
                        if isinstance(runoff_series, pd.DataFrame):
                            runoff_series = runoff_series.iloc[:, 0]
                    
                    # Convert from m/s to m³/s by multiplying by catchment area
                    streamflow = runoff_series * catchment_area
                    
                    # Log some statistics to debug
                    self.logger.info(f"Streamflow statistics: min={streamflow.min():.2f}, max={streamflow.max():.2f}, mean={streamflow.mean():.2f} m³/s")
                    
                    # Create dataframe and save to CSV
                    result_df = pd.DataFrame({'time': streamflow.index, 'streamflow': streamflow.values})
                    result_df.to_csv(output_csv, index=False)
                    
                    self.logger.info(f"Extracted streamflow from SUMMA output for lumped domain, saved to {output_csv}")
                    return output_csv
                    
            except Exception as e:
                self.logger.error(f"Error extracting streamflow from SUMMA output: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return self._create_empty_streamflow_file(run_dir, run_name)
        else:
            # For distributed domains, extract from MizuRoute output
            experiment_id = self.experiment_id
            mizuroute_output_dir = run_dir / "simulations" / experiment_id / "mizuRoute"
            
            # Try different pattern matching to find output files
            output_files = []
            patterns = [
                f"{experiment_id}_{run_name}*.nc",  # Try with run name in filename
                f"{experiment_id}*.nc",             # Try just experiment ID
                "*.nc"                              # Try any netCDF file as last resort
            ]
            
            for pattern in patterns:
                output_files = list(mizuroute_output_dir.glob(pattern))
                if output_files:
                    self.logger.info(f"Found MizuRoute output files with pattern: {pattern}")
                    break
            
            if not output_files:
                self.logger.warning(f"No MizuRoute output files found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Target reach ID
            sim_reach_id = self.config.get('SIM_REACH_ID')
            
            # Process each output file
            all_streamflow = []
            
            for output_file in output_files:
                try:
                    # Open the NetCDF file
                    with xr.open_dataset(output_file) as ds:
                        # Check if the reach ID exists
                        if 'reachID' in ds.variables:
                            # Find index for the specified reach ID
                            reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                            
                            if len(reach_indices) > 0:
                                # Get the index
                                reach_index = reach_indices[0]
                                
                                # Extract the time series for this reach
                                # Look for common streamflow variable names
                                for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                    if var_name in ds.variables:
                                        streamflow = ds[var_name].isel(seg=reach_index).to_pandas()
                                        
                                        # Reset index to use time as a column
                                        streamflow = streamflow.reset_index()
                                        
                                        # Rename columns for clarity
                                        streamflow = streamflow.rename(columns={var_name: 'streamflow'})
                                        
                                        all_streamflow.append(streamflow)
                                        self.logger.info(f"Extracted streamflow from variable {var_name}")
                                        break
                                
                                self.logger.info(f"Extracted streamflow for reach ID {sim_reach_id} from {output_file.name}")
                            else:
                                self.logger.warning(f"Reach ID {sim_reach_id} not found in {output_file.name}")
                        else:
                            self.logger.warning(f"No reachID variable found in {output_file.name}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {output_file.name}: {str(e)}")
            
            # Combine all streamflow data
            if all_streamflow:
                combined_streamflow = pd.concat(all_streamflow)
                combined_streamflow = combined_streamflow.sort_values('time')
                
                # Remove duplicates if any
                combined_streamflow = combined_streamflow.drop_duplicates(subset='time')
                
                # Save to CSV
                combined_streamflow.to_csv(output_csv, index=False)
                
                self.logger.info(f"Saved combined streamflow results to {output_csv}")
                
                return output_csv
            else:
                self.logger.warning(f"No streamflow data found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)

    def _get_catchment_area(self):
        """
        Get catchment area from basin shapefile.
        
        Returns:
            float: Catchment area in square meters, or None if not found
        """
        try:
            import geopandas as gpd
            
            # First try to get the basin shapefile
            river_basins_path = self.config.get('RIVER_BASINS_PATH')
            if river_basins_path == 'default':
                river_basins_path = self.project_dir / "shapefiles" / "river_basins"
            else:
                river_basins_path = Path(river_basins_path)
            
            river_basins_name = self.config.get('RIVER_BASINS_NAME')
            if river_basins_name == 'default':
                river_basins_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"
            
            basin_shapefile = river_basins_path / river_basins_name
            
            # If basin shapefile doesn't exist, try the catchment shapefile
            if not basin_shapefile.exists():
                self.logger.warning(f"River basin shapefile not found: {basin_shapefile}")
                self.logger.info("Trying to use catchment shapefile instead")
                
                catchment_path = self.config.get('CATCHMENT_PATH')
                if catchment_path == 'default':
                    catchment_path = self.project_dir / "shapefiles" / "catchment"
                else:
                    catchment_path = Path(catchment_path)
                
                catchment_name = self.config.get('CATCHMENT_SHP_NAME')
                if catchment_name == 'default':
                    catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
                    
                basin_shapefile = catchment_path / catchment_name
                
                if not basin_shapefile.exists():
                    self.logger.warning(f"Catchment shapefile not found: {basin_shapefile}")
                    return None
            
            # Open shapefile
            self.logger.info(f"Opening shapefile for area calculation: {basin_shapefile}")
            gdf = gpd.read_file(basin_shapefile)
            
            # Log the available columns
            self.logger.info(f"Available columns in shapefile: {gdf.columns.tolist()}")
            
            # Try to get area from attributes first
            area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
            
            if area_col in gdf.columns:
                # Sum all basin areas (in case of multiple basins)
                total_area = gdf[area_col].sum()
                
                # Check if the area seems reasonable
                if total_area <= 0 or total_area > 1e12:  # Suspicious if > 1 million km²
                    self.logger.warning(f"Area from attribute {area_col} seems unrealistic: {total_area} m². Calculating geometrically.")
                else:
                    self.logger.info(f"Found catchment area from attribute: {total_area} m²")
                    return total_area
            
            # If area column not found or value is suspicious, calculate area from geometry
            self.logger.info("Calculating catchment area from geometry")
            
            # Make sure CRS is in a projected system for accurate area calculation
            if gdf.crs is None:
                self.logger.warning("Shapefile has no CRS information, assuming WGS84")
                gdf.crs = "EPSG:4326"
            
            # If geographic (lat/lon), reproject to a UTM zone for accurate area calculation
            if gdf.crs.is_geographic:
                # Calculate centroid to determine appropriate UTM zone
                centroid = gdf.dissolve().centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                
                # Determine UTM zone
                utm_zone = int(((lon + 180) / 6) % 60) + 1
                north_south = 'north' if lat >= 0 else 'south'
                
                utm_crs = f"+proj=utm +zone={utm_zone} +{north_south} +datum=WGS84 +units=m +no_defs"
                self.logger.info(f"Reprojecting from {gdf.crs} to UTM zone {utm_zone} ({utm_crs})")
                
                # Reproject
                gdf = gdf.to_crs(utm_crs)
            
            # Calculate area in m²
            gdf['calc_area'] = gdf.geometry.area
            total_area = gdf['calc_area'].sum()
            
            self.logger.info(f"Calculated catchment area from geometry: {total_area} m²")
            
            # Double check if area seems reasonable
            if total_area <= 0:
                self.logger.error(f"Calculated area is non-positive: {total_area} m²")
                return None
            
            if total_area > 1e12:  # > 1 million km²
                self.logger.warning(f"Calculated area seems very large: {total_area} m² ({total_area/1e6:.2f} km²). Check units.")
            
            return total_area
            
        except Exception as e:
            self.logger.error(f"Error calculating catchment area: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _analyze_ensemble_results(self, sampling_dir):
        """
        Analyze the results of ensemble simulations.
        
        Args:
            sampling_dir: Directory containing ensemble results
        
        Returns:
            Path to analysis directory
        """
        self.logger.info(f"Analyzing ensemble results from {sampling_dir}")
        
        # Create analysis directory
        analysis_dir = sampling_dir / "ensemble_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get observed data path
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and flow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if date_col is None or flow_col is None:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Get run directories
            ensemble_dir = sampling_dir / "ensemble_runs"
            run_dirs = sorted(ensemble_dir.glob("run_*"))
            
            if not run_dirs:
                self.logger.error(f"No run directories found in {ensemble_dir}")
                return None
            
            # Process each run to extract metrics
            metrics = {}
            all_flows = pd.DataFrame(index=obs_df.index)
            all_flows['Observed'] = observed_flow
            
            for run_dir in run_dirs:
                run_id = run_dir.name
                
                # Look for streamflow results
                results_file = run_dir / "results" / f"{run_id}_streamflow.csv"
                
                if not results_file.exists():
                    self.logger.warning(f"No streamflow results found for {run_id}")
                    continue
                
                # Read the CSV file
                try:
                    run_df = pd.read_csv(results_file)
                    
                    # Get time and flow columns
                    time_col = next((col for col in run_df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                    flow_col_sim = next((col for col in run_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'streamflow' in col.lower()), None)
                    
                    if time_col is None or flow_col_sim is None:
                        self.logger.warning(f"Could not find time or flow columns in {run_id} results")
                        continue
                    
                    # Convert to datetime and set as index
                    run_df['DateTime'] = pd.to_datetime(run_df[time_col])
                    run_df.set_index('DateTime', inplace=True)
                    
                    # Add to all_flows for visualization
                    all_flows[run_id] = run_df[flow_col_sim].reindex(all_flows.index, method='nearest')
                    
                    # Calculate performance metrics against observed data
                    common_idx = obs_df.index.intersection(run_df.index)
                    if len(common_idx) < 10:
                        self.logger.warning(f"Not enough common time steps for {run_id} to calculate metrics")
                        continue
                    
                    # Calculate metrics
                    obs_aligned = observed_flow.loc[common_idx]
                    sim_aligned = run_df[flow_col_sim].loc[common_idx]
                    
                    mean_obs = obs_aligned.mean()
                    
                    # Nash-Sutcliffe Efficiency (NSE)
                    numerator = ((obs_aligned - sim_aligned) ** 2).sum()
                    denominator = ((obs_aligned - mean_obs) ** 2).sum()
                    nse = 1 - (numerator / denominator) if denominator > 0 else np.nan
                    
                    # Root Mean Square Error (RMSE)
                    rmse = np.sqrt(((obs_aligned - sim_aligned) ** 2).mean())
                    
                    # Percent Bias (PBIAS)
                    pbias = 100 * (sim_aligned.sum() - obs_aligned.sum()) / obs_aligned.sum() if obs_aligned.sum() != 0 else np.nan
                    
                    # Kling-Gupta Efficiency (KGE)
                    r = obs_aligned.corr(sim_aligned)
                    alpha = sim_aligned.std() / obs_aligned.std() if obs_aligned.std() != 0 else np.nan
                    beta = sim_aligned.mean() / mean_obs if mean_obs != 0 else np.nan
                    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
                    
                    metrics[run_id] = {
                        'KGE': kge,
                        'NSE': nse,
                        'RMSE': rmse,
                        'PBIAS': pbias
                    }
                    
                    self.logger.info(f"Calculated metrics for {run_id}: KGE={kge:.4f}, NSE={nse:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing results for {run_id}: {str(e)}")
            
            if not metrics:
                self.logger.error("No valid metrics calculated")
                return None
            
            # Convert metrics to DataFrame and save
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            metrics_df.to_csv(analysis_dir / "performance_metrics.csv")
            
            self.logger.info(f"Saved performance metrics for {len(metrics)} runs")
            
            return analysis_dir
            
        except Exception as e:
            self.logger.error(f"Error analyzing ensemble results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _combine_performance_metrics(self, iterative_dir, iteration):
        """
        Combine performance metrics from all previous iterations.
        
        Args:
            iterative_dir: Base directory for iterative emulation
            iteration: Current iteration number
        
        Returns:
            Path to combined metrics file
        """
        self.logger.info(f"Combining performance metrics from iterations 1-{iteration}")
        
        import pandas as pd
        import os
        from pathlib import Path
        
        all_metrics = []
        
        # Collect metrics from all previous iterations
        for i in range(1, iteration + 1):
            if i == 1:
                # First iteration uses the standard ensemble analysis
                metrics_file = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "ensemble_analysis" / "performance_metrics.csv"
            else:
                metrics_file = iterative_dir / f"iteration_{i-1}_sampling" / "ensemble_analysis" / "performance_metrics.csv"
            
            if metrics_file.exists():
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    
                    # Add iteration column
                    metrics_df['Iteration'] = i - 1
                    
                    all_metrics.append(metrics_df)
                    self.logger.info(f"Added {len(metrics_df)} metrics from iteration {i-1}")
                except Exception as e:
                    self.logger.warning(f"Error reading metrics from iteration {i-1}: {str(e)}")
        
        if not all_metrics:
            self.logger.warning("No metrics found from previous iterations")
            return None
        
        # Combine all metrics
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
        # Save combined metrics
        combined_file = iterative_dir / "combined_performance_metrics.csv"
        combined_metrics.to_csv(combined_file, index=False)
        
        self.logger.info(f"Combined metrics from {len(all_metrics)} iterations, total {len(combined_metrics)} samples")
        
        return combined_file

    def _plot_iterative_progress(self, tracking_df, output_dir):
        """
        Plot the progress of iterative emulation.
        
        Args:
            tracking_df: DataFrame with tracking data
            output_dir: Directory to save the plot
        """
        if tracking_df.empty:
            self.logger.warning("No tracking data to plot")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot predicted and actual scores
            plt.subplot(2, 1, 1)
            plt.plot(tracking_df['Iteration'], tracking_df['PredictedScore'], 'b-o', label='Predicted Score')
            plt.plot(tracking_df['Iteration'], tracking_df['ActualScore'], 'r-o', label='Actual Score')
            
            plt.xlabel('Iteration')
            plt.ylabel(self.config.get('EMULATION_TARGET_METRIC', 'KGE'))
            plt.title('Iterative Emulation Performance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot improvement
            plt.subplot(2, 1, 2)
            valid_improvements = tracking_df['Improvement'].dropna()
            
            if not valid_improvements.empty:
                plt.bar(valid_improvements.index, valid_improvements.values, color='green' if (valid_improvements > 0).all() else 'orange')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                plt.xlabel('Iteration')
                plt.ylabel(f'Improvement in {self.config.get("EMULATION_TARGET_METRIC", "KGE")}')
                plt.title('Performance Improvement by Iteration')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "iterative_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting iterative progress: {str(e)}")

    @get_function_logger
    def calculate_ensemble_performance_metrics(self):
        """
        Calculate performance metrics for ensemble simulations and create visualizations.
        
        This function:
        1. Processes all ensemble simulation outputs
        2. Compares them with observed streamflow
        3. Calculates performance metrics (KGE, NSE, RMSE, PBIAS)
        4. Creates visualization of all ensemble runs vs observed
        5. Saves metrics to CSV file for use by the random forest emulator
        
        Returns:
            Path: Path to performance metrics file
        """
        self.logger.info("Calculating performance metrics for ensemble simulations")
        
        # Define paths
        ensemble_dir = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "ensemble_runs"
        metrics_dir = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "ensemble_analysis"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / "performance_metrics.csv"
        
        # Check if ensemble runs exist
        if not ensemble_dir.exists() or not any(ensemble_dir.glob("run_*")):
            self.logger.error("No ensemble run directories found, cannot calculate metrics")
            return None
        
        # Get observed data
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Load observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and streamflow columns in observed data
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if date_col is None or flow_col is None:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            
            # Process each ensemble run
            run_dirs = sorted(ensemble_dir.glob("run_*"))
            metrics = {}
            all_flows = pd.DataFrame(index=obs_df.index)
            all_flows['Observed'] = obs_df[flow_col]
            
            for run_dir in run_dirs:
                run_id = run_dir.name
                self.logger.debug(f"Processing {run_id}")
                
                # Find streamflow output file
                results_file = run_dir / "results" / f"{run_id}_streamflow.csv"
                
                if not results_file.exists():
                    self.logger.warning(f"No streamflow results found for {run_id}")
                    continue
                
                # Read streamflow data
                sim_df = pd.read_csv(results_file)
                
                # Convert time column to datetime and set as index
                time_col = next((col for col in sim_df.columns if 'time' in col.lower() or 'date' in col.lower()), 'time')
                sim_df['DateTime'] = pd.to_datetime(sim_df[time_col])
                sim_df.set_index('DateTime', inplace=True)
                
                # Get streamflow column
                sim_flow_col = next((col for col in sim_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q' in col.lower()), 'streamflow')
                
                # Add to all flows dataframe
                all_flows[run_id] = sim_df[sim_flow_col].reindex(obs_df.index, method='nearest')
                
                # Calculate performance metrics
                common_idx = obs_df.index.intersection(sim_df.index)
                if len(common_idx) == 0:
                    self.logger.warning(f"No common time steps between observed and simulated data for {run_id}")
                    continue
                    
                # Extract common data
                observed = obs_df.loc[common_idx, flow_col]
                simulated = sim_df.loc[common_idx, sim_flow_col]
                
                # Calculate metrics
                metrics[run_id] = self._calculate_performance_metrics(observed, simulated)
            
            if not metrics:
                self.logger.error("No valid metrics calculated for any ensemble run")
                return None
            
            # Convert metrics to DataFrame and save
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Important: Ensure the index (run IDs) is saved as a column
            metrics_df = metrics_df.reset_index()
            metrics_df.rename(columns={'index': 'Run'}, inplace=True)
            
            # Save in both formats to be safe
            metrics_df.to_csv(metrics_file, index=False)
            metrics_df.set_index('Run').to_csv(metrics_dir / "performance_metrics_indexed.csv")
            
            self.logger.info(f"Performance metrics saved to {metrics_file}")
            
            # Create visualization
            self._plot_ensemble_comparison(all_flows, metrics_df.set_index('Run'), metrics_dir)
            
            return metrics_file
                
        except Exception as e:
            self.logger.error(f"Error calculating ensemble performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_performance_metrics(self, observed, simulated):
        """
        Calculate streamflow performance metrics.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        
        Returns:
            Dictionary of performance metrics
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaN values in either series
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan}
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed.mean()
        nse_numerator = ((observed - simulated) ** 2).sum()
        nse_denominator = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # Percent Bias (PBIAS)
        pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
        
        # Kling-Gupta Efficiency (KGE)
        r = observed.corr(simulated)  # Correlation coefficient
        alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan  # Relative variability
        beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan  # Bias ratio
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        return {
            'KGE': kge,
            'NSE': nse,
            'RMSE': rmse,
            'PBIAS': pbias
        }

    def _plot_ensemble_comparison(self, all_flows, metrics_df, output_dir):
        """
        Create visualizations comparing all ensemble runs to observed data.
        
        Args:
            all_flows: DataFrame with observed and simulated flows
            metrics_df: DataFrame with performance metrics
            output_dir: Directory to save visualizations
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.dates import DateFormatter
        
        # Create time series plot
        plt.figure(figsize=(14, 8))
        
        # Plot observed data
        plt.plot(all_flows.index, all_flows['Observed'], 'k-', linewidth=2, label='Observed')
        
        # Plot each simulation with low alpha
        for col in all_flows.columns:
            if col != 'Observed':
                plt.plot(all_flows.index, all_flows[col], 'b-', alpha=0.1, linewidth=0.5)
        
        # Format plot
        plt.xlabel('Date')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Ensemble Simulations vs. Observed Streamflow')
        plt.legend(['Observed', 'Ensemble Runs'])
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Add metrics summary
        metrics_text = (
            f"Ensemble Size: {len(metrics_df)}\n"
            f"Mean KGE: {metrics_df['KGE'].mean():.3f} (±{metrics_df['KGE'].std():.3f})\n"
            f"Mean NSE: {metrics_df['NSE'].mean():.3f} (±{metrics_df['NSE'].std():.3f})\n"
            f"Mean RMSE: {metrics_df['RMSE'].mean():.3f} (±{metrics_df['RMSE'].std():.3f})\n"
            f"Best KGE: {metrics_df['KGE'].max():.3f} (Run: {metrics_df['KGE'].idxmax()})"
        )
        
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_streamflow_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create flow duration curve plot
        plt.figure(figsize=(10, 6))
        
        # Sort observed flow
        obs_sorted = all_flows['Observed'].dropna().sort_values(ascending=False)
        obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
        
        # Plot observed FDC
        plt.plot(obs_exceed, obs_sorted, 'k-', linewidth=2, label='Observed')
        
        # Plot each simulation FDC with low alpha
        for col in all_flows.columns:
            if col != 'Observed':
                sim_sorted = all_flows[col].dropna().sort_values(ascending=False)
                sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
                plt.plot(sim_exceed, sim_sorted, 'b-', alpha=0.1, linewidth=0.5)
        
        # Format plot
        plt.xlabel('Exceedance Probability')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Flow Duration Curves')
        plt.yscale('log')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(['Observed', 'Ensemble Runs'])
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_flow_duration_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create metrics distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        metrics_list = ['KGE', 'NSE', 'RMSE', 'PBIAS']
        
        for i, metric in enumerate(metrics_list):
            ax = axes[i//2, i%2]
            ax.hist(metrics_df[metric].dropna(), bins=20, alpha=0.7)
            ax.set_title(f'{metric} Distribution')
            ax.axvline(metrics_df[metric].mean(), color='r', linestyle='-', label='Mean')
            ax.grid(True, alpha=0.3)
            
            # Add best run label
            if metric in ['KGE', 'NSE']:
                best_val = metrics_df[metric].max()
                best_run = metrics_df[metric].idxmax()
            else:
                best_val = metrics_df[metric].min()
                best_run = metrics_df[metric].idxmin()
                
            ax.text(0.05, 0.95, f'Best: {best_val:.3f}\nRun: {best_run}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    @get_function_logger
    def run_random_forest_emulation(self):
        """
        Run random forest emulation to find optimal parameters based on
        geospatial attributes and performance metrics.
        """
        self.logger.info("Starting random forest emulation")
        
        # Check if emulation is enabled in config
        if not self.config.get('RUN_RANDOM_FOREST_EMULATION', False):
            self.logger.info("Random forest emulation disabled in config. Skipping.")
            return None
        
        try:
            # Import RandomForestEmulator class

            
            # Create the emulator
            emulator = RandomForestEmulator(self.config, self.logger)
            
            # Run the emulation workflow
            results = emulator.run_workflow()
            
            if results:
                self.logger.info(f"Random forest emulation completed successfully.")
                self.logger.info(f"Optimized parameters saved to: {results['output_dir']}")
                
                # Display key metrics
                self.logger.info(f"Model R² score: {results['model_metrics']['test_score']:.4f}")
                self.logger.info(f"Predicted {emulator.target_metric}: {results['predicted_score']:.4f}")
                
                return results
            else:
                self.logger.warning("Random forest emulation did not produce results.")
                return None
                
        except ImportError as e:
            self.logger.error(f"Could not import RandomForestEmulator: {str(e)}. Ensure the module exists.")
            raise
        except Exception as e:
            self.logger.error(f"Error during random forest emulation: {str(e)}")
            raise
        
    def _get_default_path(self, config_key: str, default_suffix: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.

        Args:
            config_key (str): The key to look up in the config dictionary.
            default_suffix (str): The default subpath to use if the config value is 'default'.

        Returns:
            Path: The resolved path.
        """
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    @get_function_logger  
    def run_benchmarking(self):
        # Preprocess data for benchmarking
        preprocessor = BenchmarkPreprocessor(self.config, self.logger)
        benchmark_data = preprocessor.preprocess_benchmark_data(f"{self.config['CALIBRATION_PERIOD'].split(',')[0]}", f"{self.config['EVALUATION_PERIOD'].split(',')[1]}")

        # Run benchmarking
        benchmarker = Benchmarker(self.config, self.logger)
        benchmark_results = benchmarker.run_benchmarking()

        bv = BenchmarkVizualiser(self.config, self.logger)
        bv.visualize_benchmarks(benchmark_results)

    @get_function_logger    
    def run_postprocessing(self):
        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            if model == 'FUSE':
                fpp = FUSEPostprocessor(self.config, self.logger)
                results_file = fpp.extract_streamflow()
            elif model == 'GR':
                gpp = GRPostprocessor(self.config, self.logger)
                results_file = gpp.extract_streamflow()
            elif model == 'SUMMA':
                spp = SUMMAPostprocessor(self.config, self.logger)
                results_file = spp.extract_streamflow()
            elif model == 'FLASH':
                fpp = FLASHPostProcessor(self.config, self.logger)
                results_file = fpp.extract_streamflow()
            elif model == 'HYPE':
                hpp = HYPEPostProcessor(self.config, self.logger)
                results_file = hpp.extract_results()
            elif model == 'MESH':
                mpp = MESHPostProcessor(self.config, self.logger)
                results_file = mpp.extract_streamflow() 
            else:
                pass

        tv = TimeseriesVisualizer(self.config, self.logger)
        metrics_df = tv.create_visualizations()
            

    @get_function_logger
    def calibrate_model(self):
        # Calibrate the model using specified method and objectives
        try:
            for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
                if model == 'SUMMA':
                    if self.config.get('OPTMIZATION_ALOGORITHM') == 'OSTRICH':
                        self.run_ostrich_optimization()
                    else:
                        self.run_parallel_optimization()
                else:
                    pass
        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            return None

    @get_function_logger  
    def run_sensitivity_analysis(self):
        self.logger.info("Starting sensitivity analysis")
        try:

            for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
                if model == 'SUMMA':
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
                else:
                    pass
 
        except Exception as e:
            self.logger.error(f"Error during sensitivity analysis: {str(e)}")
            return None
                    
    @get_function_logger  
    def run_decision_analysis(self):
        self.logger.info("Starting decision analysis")

        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            if model == 'SUMMA':
                decision_analyzer = DecisionAnalyzer(self.config, self.logger)
                
                results_file, best_combinations = decision_analyzer.run_full_analysis()
                
                self.logger.info("Decision analysis completed")
                self.logger.info(f"Results saved to: {results_file}")
                self.logger.info("Best combinations for each metric:")
                for metric, data in best_combinations.items():
                    self.logger.info(f"  {metric}: score = {data['score']:.3f}")

            elif model == 'FUSE':
                FUSE_decision_analyser = FuseDecisionAnalyzer(self.config, self.logger)
                FUSE_decision_analyser.run_decision_analysis()
            
            else:
                pass

    
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
    def delineate_point_buffer_shape(self):
        self.logger.info("Starting point buffer")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('point buffer completed successfully')
            return delineator.delineate_point_buffer_shape()
        except Exception as e:
            self.logger.error(f"Error during point buffer delineation: {str(e)}")
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

    @get_function_logger
    def delineate_coastal(self):
        self.logger.info("Starting geofabric delineation")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            return delineator.delineate_coastal()
        except Exception as e:
            self.logger.error(f"Error during coastal delineation: {str(e)}")
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
