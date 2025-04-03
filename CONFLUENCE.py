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
from utils.models_utils.summa_utils import SUMMAPostprocessor, SummaRunner, SummaPreProcessor_spatial, SummaPreProcessor_point # type: ignore
from utils.models_utils.fuse_utils import FUSEPreProcessor, FUSERunner, FuseDecisionAnalyzer, FUSEPostprocessor # type: ignore
#from utils.models_utils.gr_utils import GRPreProcessor, GRRunner, GRPostprocessor # type: ignore
from utils.models_utils.flash_utils import FLASH, FLASHPostProcessor # type: ignore
#from utils.models_utils.hype_utils import HYPEPreProcessor, HYPERunner, HYPEPostProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPreProcessor, MESHRunner, MESHPostProcessor # type: ignore

# Evaluation utilities
#from utils.evaluation_util.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore
from utils.optimization_utils.ostrich_util import OstrichOptimizer # type: ignore

# Reporting utilities
from utils.report_utils.reporting_utils import VisualizationReporter # type: ignore
from utils.report_utils.result_vizualisation_utils import BenchmarkVizualiser, TimeseriesVisualizer # type: ignore

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
        self.evaluation_dir = self.project_dir / 'evaluation'
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

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
            (self.acquire_attributes, lambda: (self.project_dir / "attributes" / "elevation" / "dem" / f"domain_{self.domain_name}_elv.tif").exists()),
            (self.define_domain, lambda: (self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()),
            (self.plot_domain, lambda: (self.project_dir / "plots" / "domain" / 'domain_map.png').exists()),
            (self.discretize_domain, lambda: (self.project_dir / "shapefiles" / "catchment" / f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()),
            (self.plot_discretised_domain, lambda: (self.project_dir / "plots" / "discretization" / f"domain_discretization_{self.config['DOMAIN_DISCRETIZATION']}.png").exists()),
            
            # Model agnostic data pre- processing
            (self.process_attributes, lambda: (self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv1").exists()),
            (self.process_observed_data, lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv").exists()),
            (self.acquire_forcings, lambda: (self.project_dir / "forcing" / "raw_data").exists()),
            (self.model_agnostic_pre_processing, lambda: (self.project_dir / "forcing" / "basin_averaged_data1").exists()),

            # Modesl specific processing
            (self.model_specific_pre_processing, lambda: (self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input1").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists()),
            (self.visualise_model_output, lambda: (self.project_dir / "plots" / "results" / "streamflow_comparison.png1").exists()),
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
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, pour point not required")
            return None
                    
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
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, attribute data not required")
            return None

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
     
        # Skip domain definition if in point mode
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, domain definition not required")
            return
        
        if domain_method == 'subset':
            self.subset_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        elif domain_method == 'lumped':
            self.delineate_lumped_watershed(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        elif domain_method == 'delineate':
            self.delineate_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        elif self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, delineation not required")
            return None
        else:
            self.logger.error(f"Unknown domain definition method: {domain_method}")

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
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, discretisation not performed")
            return None
        
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
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, data acquisition not required")
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
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, data processing not required")
            return None
        
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

        # Prepare run the MAF Orchestrator
        if 'MESH' in self.config.get('HYDROLOGICAL_MODEL').split(',') or 'HYPE' in self.config.get('HYDROLOGICAL_MODEL').split(','):
            dap = DataAcquisitionProcessor(self.config, self.logger)
            dap.run_data_acquisition()
       
    @get_function_logger
    def process_observed_data(self):
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, data processing not required")
            return None
        
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
                    if self.config.get('SPATIAL_MODE') == 'Point':
                        self.logger.info("Initializing SUMMA point preprocessor")
                        ssp = SummaPreProcessor_point(self.config, self.logger)
                        ssp.run_preprocessing()
                    else:
                        self.logger.info("Initializing SUMMA spatial preprocessor")
                        ssp = SummaPreProcessor_spatial(self.config, self.logger)
                        ssp.run_preprocessing()
                        
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
                    if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
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
                if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped':
                    plot_file = visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
                else:
                    visualizer.update_sim_reach_id() # Find and update the sim reach id based on the project pour point
                    model_outputs = [
                        (f"{model}", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "mizuRoute" / f"{self.config['EXPERIMENT_ID']}*.nc"))
                    ]
                    obs_files = [
                        ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
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
