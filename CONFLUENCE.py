from pathlib import Path
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd # type: ignore
import rasterio # type: ignore
import numpy as np # type: ignore
from scipy import stats # type: ignore
import argparse

# Import CONFLUENCE utility functions
sys.path.append(str(Path(__file__).resolve().parent))

# Data and config management utilities 
from utils.dataHandling_utils.data_utils import ProjectInitialisation, ObservedDataProcessor, BenchmarkPreprocessor # type: ignore  
from utils.dataHandling_utils.data_acquisition_utils import gistoolRunner, datatoolRunner # type: ignore
from utils.dataHandling_utils.agnosticPreProcessor_util import forcingResampler, geospatialStatistics # type: ignore
from utils.dataHandling_utils.variable_utils import VariableHandler # type: ignore
from utils.configHandling_utils.config_utils import ConfigManager # type: ignore
from utils.configHandling_utils.logging_utils import setup_logger, get_function_logger, log_configuration # type: ignore

# Domain definition utilities
from utils.geospatial_utils.discretization_utils import DomainDiscretizer # type: ignore
from utils.geospatial_utils.raster_utils import calculate_landcover_mode # type: ignore
from utils.geospatial_utils.domain_utilities import DomainDelineator # type: ignore

# Model specific utilities
from utils.models_utils.mizuroute_utils import MizuRoutePreProcessor, MizuRouteRunner # type: ignore
from utils.models_utils.summa_utils import SUMMAPostprocessor, SummaRunner, SummaPreProcessor # type: ignore
from utils.models_utils.fuse_utils import FUSEPreProcessor, FUSERunner, FuseDecisionAnalyzer, FUSEPostprocessor # type: ignore
from utils.models_utils.gr_utils import GRPreProcessor, GRRunner, GRPostprocessor # type: ignore
from utils.models_utils.flash_utils import FLASH, FLASHPostProcessor # type: ignore
from utils.models_utils.hype_utils import HYPEPreProcessor, HYPERunner, HYPEPostProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPreProcessor, MESHRunner, MESHPostProcessor # type: ignore

# Reporting utilities
from utils.report_utils.reporting_utils import VisualizationReporter # type: ignore
from utils.report_utils.result_vizualisation_utils import BenchmarkVizualiser, TimeseriesVisualizer # type: ignore
from utils.report_utils.domain_visualization_utils import DomainVisualizer # type: ignore

# Optimisation and Evaluation utilities
from utils.evaluation_util.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore
from utils.optimization_utils.emulation_runner import EmulationRunner # type: ignore
from utils.optimization_utils.dds_optimizer import DDSOptimizer # type: ignore
from utils.optimization_utils.pso_optimizer import PSOOptimizer # type: ignore
from utils.optimization_utils.sce_ua_optimizer import SCEUAOptimizer # type: ignore
from utils.optimization_utils.results_utils import OptimizationResultsManager # type: ignore

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
        self.catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        self.catchment_dir.mkdir(parents=True,  exist_ok=True)

        self.setup_logging()

       # Log configuration file using the original config path
        log_dir = self.project_dir / f"_workLog_{self.domain_name}"
        self.config_log_file = log_configuration(config, log_dir, self.domain_name)

        # Initialize utility classes
        self.project_initialisation = ProjectInitialisation(self.config, self.logger)
        self.reporter = VisualizationReporter(self.config, self.logger)
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        self.domain_visualizer = DomainVisualizer(self.config, self.logger, self.reporter)
        self.domain_delineator = DomainDelineator(self.config, self.logger)
        self.optimization_results_manager = OptimizationResultsManager(
            self.project_dir, 
            self.experiment_id, 
            self.logger
        )

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
            (self.setup_project, (self.project_dir).exists),
            
            # --- Geospatial domain definition and analysis ---
            (self.create_pourPoint, lambda: (self.project_dir / "shapefiles" / "pour_point" / f"{self.domain_name}_pourPoint.shp").exists()),
            (self.acquire_attributes, lambda: (self.project_dir / "attributes" / "soilclass" / f"domain_{self.domain_name}_soil_classes.tif").exists()),
            (self.define_domain, lambda: (self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()),
            (self.discretize_domain, lambda: (self.project_dir / "shapefiles" / "catchment" / f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()),
            
            # --- Model Agnostic Data Pre- Processing ---
            (self.process_observed_data, lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv").exists()),
            (self.acquire_forcings, lambda: (self.project_dir / "forcing" / "raw_data").exists()),
            (self.model_agnostic_pre_processing, lambda: (self.project_dir / "forcing" / "basin_averaged_data1").exists()),
            (self.run_benchmarking, lambda: (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),

            # --- Model Specific Processing and Initialisation Steps ---
            (self.model_specific_pre_processing, lambda: (self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists()),

            # --- Emulation and Optimization Steps ---
            (self.calibrate_model, lambda: self.config.get('RUN_ITERATIVE_OPTIMISATION', False) and (self.project_dir / "optimisation" / f"dds_{self.config.get('EXPERIMENT_ID')}" / "best_params.csv1").exists()),
            (self.emulate_model_parameters, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "rf_emulation" / "optimized_parameters.csv1").exists()),
            (self.run_decision_analysis, lambda: (self.project_dir / "optimisation " / f"{self.config.get('EXPERIMENT_ID')}_model_decisions_comparison.csv").exists()),  
            (self.run_sensitivity_analysis, lambda: (self.project_dir / "plots" / "sensitivity_analysis" / "all_sensitivity_results.csv").exists()),
            
            # --- Result Analysis and Evaluation ---
            (self.run_postprocessing, lambda: (self.project_dir / "results" / "postprocessed.csv").exists()),
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
    
    def setup_project(self):
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        project_dir = self.project_initialisation.setup_project()
        
        self.logger.info(f"Project directory created at: {project_dir}")
        self.logger.info(f"shapefiles directories created")
        
        return project_dir

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
            
            # Use the utility function
            calculate_landcover_mode(input_dir, output_file, start_year, end_year, self.config['DOMAIN_NAME'])


        # Create the gistool command for soil classes
        gistool_command_soilclass = gr.create_gistool_command(dataset = 'soil_class', output_dir = soilclass_dir, lat_lims = latlims, lon_lims = lonlims, variables = 'soil_classes')
        gr.execute_gistool_command(gistool_command_soilclass)

    def define_domain(self):
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        self.logger.info(f"Domain definition workflow starting with: {domain_method}")
        
        # Skip domain definition if shapefile is provided
        if self.config.get('RIVER_BASINS_NAME') != 'default':
            self.logger.info('Shapefile provided, skipping domain definition')
            return
        
        # If in point mode
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.domain_delineator.delineate_point_buffer_shape()
            return
        
        # Map of domain methods to their corresponding functions
        domain_methods = {
            'subset': self.domain_delineator.subset_geofabric,
            'lumped': self.domain_delineator.delineate_lumped_watershed,
            'delineate': self.domain_delineator.delineate_geofabric
        }
        
        method_function = domain_methods.get(domain_method)
        if method_function:
            method_function()
            
            # Handle coastal watersheds if needed
            if domain_method == 'delineate' and self.config.get('DELINEATE_COASTAL_WATERSHEDS'):
                self.domain_delineator.delineate_coastal()
        else:
            self.logger.error(f"Unknown domain definition method: {domain_method}")
        
        self.domain_visualizer.plot_domain()
        self.logger.info(f"Domain definition workflow finished")
    
    def discretize_domain(self):
        try:
            self.logger.info(f"Discretizing domain using method: {self.config.get('DOMAIN_DISCRETIZATION')}")
            domain_discretizer = DomainDiscretizer(self.config, self.logger)
            hru_shapefile = domain_discretizer.discretize_domain()
            self.domain_visualizer.plot_discretized_domain()
            self.logger.info(f"Domain discretized successfully. HRU shapefile(s): {hru_shapefile}")
        except Exception as e:
            self.logger.error(f"Error during discretisation: {str(e)}")
            raise

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

        # Run the MAF Orchestrator
        #if 'MESH' in self.config.get('HYDROLOGICAL_MODEL').split(',') or 'HYPE' in self.config.get('HYDROLOGICAL_MODEL').split(','):
            #dap = DataAcquisitionProcessor(self.config, self.logger)
            #dap.run_data_acquisition()
       
    def process_observed_data(self):
        self.logger.info("Processing observed data")
        observed_data_processor = ObservedDataProcessor(self.config, self.logger)
        try:
            observed_data_processor.process_streamflow_data()
            self.logger.info("Observed data processing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during observed data processing: {str(e)}")
            raise

    def run_benchmarking(self):
        # Preprocess data for benchmarking
        preprocessor = BenchmarkPreprocessor(self.config, self.logger)
        benchmark_data = preprocessor.preprocess_benchmark_data(f"{self.config['CALIBRATION_PERIOD'].split(',')[0]}", f"{self.config['EVALUATION_PERIOD'].split(',')[1]}")

        # Run benchmarking
        benchmarker = Benchmarker(self.config, self.logger)
        benchmark_results = benchmarker.run_benchmarking()

        bv = BenchmarkVizualiser(self.config, self.logger)
        bv.visualize_benchmarks(benchmark_results)


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
                try:
                    mr = MESHRunner(self.config, self.logger)
                    mr.run_MESH()
                except Exception as e:
                    self.logger.error(f"Error during MESH model run: {str(e)}")
            else:
                self.logger.error(f"Unknown hydrological model: {model}")

        self.visualise_model_output()
        self.logger.info("Model runs completed")

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
                    
    def calibrate_model(self):
        """
        Calibrate the model using the specified optimization algorithm and objectives.
        
        Supports multiple optimization algorithms including:
        - Differential Evolution (DE)
        - Particle Swarm Optimization (PSO) 
        - Shuffled Complex Evolution (SCE-UA)
        
        Returns:
            Path: Path to calibration results file or None if calibration failed
        """
        self.logger.info("Starting model calibration")
        
        # Get the optimization algorithm from the config
        opt_algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO')
        
        try:
            for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
                if model == 'SUMMA':
                    # Create optimization directory if it doesn't exist
                    opt_dir = self.project_dir / "optimisation"
                    opt_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Choose the optimizer based on the algorithm
                    if opt_algorithm == 'PSO':
                        self.logger.info("Using Particle Swarm Optimization (PSO)")
                        optimizer = PSOOptimizer(self.config, self.logger)
                        result = optimizer.run_pso_optimization()
                    elif opt_algorithm == 'SCE-UA':
                        self.logger.info("Using Shuffled Complex Evolution (SCE-UA)")
                        optimizer = SCEUAOptimizer(self.config, self.logger)
                        result = optimizer.run_sceua_optimization()
                    elif opt_algorithm == 'DDS':
                        self.logger.info("Using Dynamically Dimensioned Search (DDS)")
                        optimizer = DDSOptimizer(self.config, self.logger)
                        result = optimizer.run_dds_optimization()
                    else:
                        self.logger.info(f"Optimisation algorithm {opt_algorithm} not supported")
                        continue
                    
                    # Save results to standard location using the results manager
                    target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
                    results_file = self.optimization_results_manager.save_optimization_results(
                        result, opt_algorithm, target_metric
                    )
                    
                    if results_file:
                        self.logger.info(f"Calibration completed successfully: {results_file}")
                        return results_file
                else:
                    self.logger.warning(f"Calibration for model {model} not yet implemented")

        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
        self.logger.warning("Calibration completed but results file not found")
        return None

    def emulate_model_parameters(self):
        """
        Emulate the model parameter response using the specified algorithm and objectives.
        
        This method delegates to the EmulationRunner class for the complete
        emulation workflow including attribute processing, ensemble simulation,
        performance metric calculation, and random forest emulation.
        
        Returns:
            Dict: Results from the emulation workflow or None if failed
        """
        emulation_runner = EmulationRunner(self.config, self.logger)
        return emulation_runner.run_emulation_workflow()

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

                    self.logger.info("Sensitivity analysis completed")
                    return sensitivity_results
                else:
                    pass
 
        except Exception as e:
            self.logger.error(f"Error during sensitivity analysis: {str(e)}")
            return None
                    
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
