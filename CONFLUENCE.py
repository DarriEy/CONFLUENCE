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
from utils.models_utils.clm_parflow_utils import CLMParFlowPreProcessor, CLMParFlowRunner, CLMParFlowPostProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPreProcessor, MESHRunner, MESHPostProcessor # type: ignore

# Evaluation utilities
from utils.evaluation_util.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore

# Reporting utilities
from utils.report_utils.reporting_utils import VisualizationReporter # type: ignore
from utils.report_utils.result_vizualisation_utils import BenchmarkVizualiser, TimeseriesVisualizer # type: ignore

# Optimisation utilities
from utils.optimization_utils.single_sample_emulator import SingleSampleEmulator, RandomForestEmulator # type: ignore
from utils.optimization_utils.dds_optimizer import DDSOptimizer # type: ignore
from utils.optimization_utils.optimizer_manager import OptimizerManager # type: ignore
from utils.optimization_utils.pso_optimizer import PSOOptimizer # type: ignore

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
            (self.setup_project, (self.project_dir).exists),
            
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
            (self.model_agnostic_pre_processing, lambda: (self.project_dir / "forcing" / "basin_averaged_data1").exists()),

            # Model specific processing
            (self.model_specific_pre_processing, lambda: (self.project_dir / "forcing" / f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists()),
            (self.visualise_model_output, lambda: (self.project_dir / "plots" / "results" / "streamflow_comparison.png").exists()),

            # --- Emulation and Optimization Steps ---
            (self.calibrate_model, lambda: self.config.get('RUN_ITERATIVE_OPTIMISATION', False) and (self.project_dir / "optimisation" / f"dds_{self.config.get('EXPERIMENT_ID')}" / "best_params.csv1").exists()),
            (self.process_attributes, lambda: (self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv").exists()),
            (self.prepare_emulation_data, lambda: (self.project_dir / "emulation" / f"parameter_sets_{self.config.get('EXPERIMENT_ID')}.nc").exists()), 
            (self.calculate_ensemble_performance_metrics, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "ensemble_analysis" / "performance_metrics.csv").exists()),
            (self.run_random_forest_emulation, lambda: (self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID') / "rf_emulation" / "optimized_parameters.csv1").exists()),

            # Result analysis and optimisation
            (self.run_benchmarking, lambda: (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),
            (self.run_postprocessing, lambda: (self.project_dir / "results" / "postprocessed.csv").exists()),
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
                        
                        # Save results to standard location
                        self._save_optimization_results_to_csv(result)
                    elif opt_algorithm == 'SCE-UA':
                        self.logger.info("Using Shuffled Complex Evolution (SCE-UA)")
                        optimizer = SCEUAOptimizer(self.config, self.logger)
                        result = optimizer.run_sceua_optimization()
                        
                        # Save results to standard location
                        self._save_optimization_results_to_csv(result)
                    else:
                        self.logger.info(f"Using parallel optimization with {opt_algorithm}")
                        self.run_parallel_optimization()
                else:
                    self.logger.warning(f"Calibration for model {model} not yet implemented")
        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
        # Return the path to the optimization results file
        results_file = self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_parallel_iteration_results.csv"
        if results_file.exists():
            self.logger.info(f"Calibration completed successfully: {results_file}")
            return results_file
        else:
            self.logger.warning("Calibration completed but results file not found")
            return None
    
    def _save_optimization_results_to_csv(self, results):
        """
        Save optimization results to a CSV file for compatibility with other parts of CONFLUENCE.
        
        Args:
            results: Dictionary with optimization results
        """
        try:
            # Get best parameters and score
            best_params = results.get('best_parameters', {})
            best_score = results.get('best_score', None)
            
            if not best_params or best_score is None:
                self.logger.warning("No valid optimization results to save")
                return
            
            # Create DataFrame from best parameters
            results_data = {}
            
            # First add iteration column
            results_data['iteration'] = [0]
            
            # Add score column with appropriate name based on the target metric
            target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
            results_data[target_metric] = [best_score]
            
            # Add parameter columns
            for param_name, values in best_params.items():
                if isinstance(values, np.ndarray) and len(values) > 1:
                    # For parameters with multiple values (like HRU-level parameters),
                    # save the mean value
                    results_data[param_name] = [float(np.mean(values))]
                else:
                    # For scalar parameters or single-value arrays
                    results_data[param_name] = [float(values[0]) if isinstance(values, np.ndarray) else float(values)]
            
            # Create DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Save to standard location
            results_file = self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_parallel_iteration_results.csv"
            results_df.to_csv(results_file, index=False)
            self.logger.info(f"Saved optimization results to {results_file}")
            
            # Also save detailed results to optimizer-specific files
            history = results.get('history', [])
            if history:
                # Extract iteration history data
                history_data = {
                    'iteration': [],
                    target_metric: [],
                }
                
                # Add parameter columns to history data
                for param_name in best_params.keys():
                    history_data[param_name] = []
                
                # Fill history data
                for h in history:
                    history_data['iteration'].append(h.get('iteration', 0))
                    history_data[target_metric].append(h.get('best_score', np.nan))
                    
                    # Add parameter values
                    if 'best_params' in h and h['best_params']:
                        for param_name, values in h['best_params'].items():
                            if param_name in history_data:
                                if isinstance(values, np.ndarray) and len(values) > 1:
                                    history_data[param_name].append(float(np.mean(values)))
                                else:
                                    val = float(values[0]) if isinstance(values, np.ndarray) else float(values)
                                    history_data[param_name].append(val)
                
                # Create DataFrame and save
                history_df = pd.DataFrame(history_data)
                history_file = self.project_dir / "optimisation" / f"{self.config.get('EXPERIMENT_ID')}_{opt_algorithm.lower()}_history.csv"
                history_df.to_csv(history_file, index=False)
                self.logger.info(f"Saved optimization history to {history_file}")
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def run_clm_parflow_optimization(self):
        """
        Run optimization for CLM-ParFlow parameters.
        
        This method would implement a parameter estimation approach for CLM-ParFlow,
        likely using a similar approach to the existing optimization methods but with
        parameters specific to CLM-ParFlow.
        """
        self.logger.info("Starting CLM-ParFlow parameter optimization")
        
        # This is a placeholder for future implementation
        self.logger.warning("CLM-ParFlow optimization not yet implemented")
        
        # Example of how this might be implemented:
        """
        # Define parameters to calibrate
        params_to_calibrate = [
            # ParFlow parameters
            {"name": "Geom.Perm.Value", "min": 0.1, "max": 10.0},
            {"name": "Phase.Saturation.VanGenuchten.Alpha.Value", "min": 0.5, "max": 3.5},
            {"name": "Phase.Saturation.VanGenuchten.N.Value", "min": 1.5, "max": 7.0},
            # CLM parameters
            {"name": "clm_ptf", "min": 1, "max": 16},  # Plant functional type
        ]
        
        # Create optimizer
        optimizer = ParamOptimizer(self.config, self.logger)
        
        # Run optimization
        best_params = optimizer.optimize(params_to_calibrate)
        
        # Run with best parameters
        runner = CLMParFlowRunner(self.config, self.logger)
        success = runner.run_with_parameters(best_params)
        
        if success:
            self.logger.info(f"CLM-ParFlow optimization completed successfully")
            return best_params
        else:
            self.logger.error("CLM-ParFlow optimization failed")
            return None
        """
        
        return None
    
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

    def define_domain(self):
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')

        # Skip domain definition if shapefile is provided
        if self.config.get('RIVER_BASINS_NAME') == 'default':        

            # Skip domain definition if in point mode
            if self.config.get('SPATIAL_MODE') == 'Point':
                self.delineate_point_buffer_shape()        
            elif domain_method == 'subset':
                self.subset_geofabric()
            elif domain_method == 'lumped':
                self.delineate_lumped_watershed()
            elif domain_method == 'delineate':
                self.delineate_geofabric()
                if self.config.get('DELINEATE_COASTAL_WATERSHEDS'):
                    self.delineate_coastal()

            elif self.config.get('SPATIAL_MODE') == 'Point':
                self.logger.info("Spatial mode: Point simulations, delineation not required")
                return None
            else:
                self.logger.error(f"Unknown domain definition method: {domain_method}")
        else:
            self.logger.info('Shapefile provided, skipping domain definition')

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

    def discretize_domain(self):
        try:
            self.logger.info(f"Discretizing domain using method: {self.config.get('DOMAIN_DISCRETIZATION')}")
            domain_discretizer = DomainDiscretizer(self.config, self.logger)
            hru_shapefile = domain_discretizer.discretize_domain()
            self.logger.info(f"Domain discretized successfully. HRU shapefile(s): {hru_shapefile}")
        except Exception as e:
            self.logger.error(f"Error during discretisation: {str(e)}")
            raise
       
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

        # Prepare run the MAF Orchestrator
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
                elif model == 'CLM_PARFLOW':
                    # Add CLM-ParFlow preprocessing
                    self.logger.info("Initializing CLM-ParFlow preprocessor")
                    cpp = CLMParFlowPreProcessor(self.config, self.logger)
                    cpp.run_preprocessing()
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

            elif model == 'CLM_PARFLOW':
                try:
                    self.logger.info("Initializing CLM-ParFlow runner")
                    clm_parflow_runner = CLMParFlowRunner(self.config, self.logger)
                    clm_parflow_runner.run_clm_parflow()
                    self.logger.info("CLM-ParFlow model run completed successfully")
                except Exception as e:
                    self.logger.error(f"Error during CLM-ParFlow model run: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())

            else:
                self.logger.error(f"Unknown hydrological model: {model}")

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
                
            elif model == 'CLM_PARFLOW':
                # Add visualization for CLM-ParFlow outputs
                self.logger.info("Visualizing CLM-ParFlow outputs")
                
                # Define observation and simulation files
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                ]
                
                # CLM-ParFlow streamflow output location
                clm_pf_streamflow = str(self.project_dir / "results" / self.config['EXPERIMENT_ID'] / "CLM_PARFLOW" / f"{self.config.get('DOMAIN_NAME')}_{self.config['EXPERIMENT_ID']}_streamflow.csv")
                
                model_outputs = [
                    ("CLM-ParFlow", clm_pf_streamflow)
                ]
                
                # Use a generic CSV-based plotting function
                # Note: This assumes we'll implement this method in the VisualizationReporter
                plot_file = visualizer.plot_csv_streamflow_simulations_vs_observations(model_outputs, obs_files)
                
                # Plot additional CLM-ParFlow specific outputs like soil moisture
                try:
                    soil_moisture_file = self.project_dir / "results" / self.config['EXPERIMENT_ID'] / "CLM_PARFLOW" / f"{self.config.get('DOMAIN_NAME')}_{self.config['EXPERIMENT_ID']}_soil_moisture.nc"
                    if soil_moisture_file.exists():
                        soil_moisture_plot = visualizer.plot_soil_moisture(soil_moisture_file)
                        self.logger.info(f"Soil moisture visualization created: {soil_moisture_plot}")
                except Exception as e:
                    self.logger.warning(f"Error creating soil moisture visualization: {str(e)}")
                    
                # Plot water balance
                try:
                    water_balance_file = self.project_dir / "results" / self.config['EXPERIMENT_ID'] / "CLM_PARFLOW" / f"{self.config.get('DOMAIN_NAME')}_{self.config['EXPERIMENT_ID']}_water_balance.csv"
                    if water_balance_file.exists():
                        water_balance_plot = visualizer.plot_water_balance(water_balance_file)
                        self.logger.info(f"Water balance visualization created: {water_balance_plot}")
                except Exception as e:
                    self.logger.warning(f"Error creating water balance visualization: {str(e)}")

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

    def calculate_ensemble_performance_metrics(self):
        """
        Calculate performance metrics for ensemble simulations with a focus on calibration and validation periods.
        
        This method:
        1. Processes all ensemble simulation outputs
        2. Compares them with observed streamflow
        3. Calculates performance metrics (KGE, NSE, RMSE, PBIAS) for calibration and validation periods
        4. Creates visualization of ensemble runs vs observed for different time periods
        
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
            
            # Set up calibration and evaluation periods from config
            calib_period = self.config.get('CALIBRATION_PERIOD', '')
            eval_period = self.config.get('EVALUATION_PERIOD', '')
            
            if calib_period and ',' in calib_period and eval_period and ',' in eval_period:
                calib_start, calib_end = [s.strip() for s in calib_period.split(',')]
                eval_start, eval_end = [s.strip() for s in eval_period.split(',')]
                
                calib_mask = (obs_df.index >= pd.Timestamp(calib_start)) & (obs_df.index <= pd.Timestamp(calib_end))
                eval_mask = (obs_df.index >= pd.Timestamp(eval_start)) & (obs_df.index <= pd.Timestamp(eval_end))
                
                # Log the period definitions
                self.logger.info(f"Using calibration period: {calib_start} to {calib_end}")
                self.logger.info(f"Using evaluation period: {eval_start} to {eval_end}")
            else:
                # If periods not defined, use time-based split (70% calibration, 30% evaluation)
                total_time = obs_df.index[-1] - obs_df.index[0]
                split_point = obs_df.index[0] + pd.Timedelta(seconds=total_time.total_seconds() * 0.7)
                
                calib_mask = obs_df.index <= split_point
                eval_mask = obs_df.index > split_point
                
                self.logger.info(f"Using time-based split for calibration (before {split_point}) and evaluation (after {split_point})")
            
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
                
                # Calculate performance metrics for calibration and evaluation periods
                metrics[run_id] = {}
                
                # Calibration period
                if calib_mask.sum() > 0:
                    common_idx = obs_df.index[calib_mask].intersection(sim_df.index)
                    if len(common_idx) > 0:
                        observed = obs_df.loc[common_idx, flow_col]
                        simulated = sim_df.loc[common_idx, sim_flow_col]
                        
                        calib_metrics = self._calculate_performance_metrics(observed, simulated)
                        for metric, value in calib_metrics.items():
                            metrics[run_id][f"Calib_{metric}"] = value
                    else:
                        self.logger.warning(f"No common time steps between observed and simulated data for {run_id} in calibration period")
                
                # Evaluation period
                if eval_mask.sum() > 0:
                    common_idx = obs_df.index[eval_mask].intersection(sim_df.index)
                    if len(common_idx) > 0:
                        observed = obs_df.loc[common_idx, flow_col]
                        simulated = sim_df.loc[common_idx, sim_flow_col]
                        
                        eval_metrics = self._calculate_performance_metrics(observed, simulated)
                        for metric, value in eval_metrics.items():
                            metrics[run_id][f"Eval_{metric}"] = value
                    else:
                        self.logger.warning(f"No common time steps between observed and simulated data for {run_id} in evaluation period")
            
            if not metrics:
                self.logger.error("No valid metrics calculated for any ensemble run")
                return None
            
            # Convert metrics to DataFrame and save
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Ensure the index (run IDs) is saved as a column
            metrics_df = metrics_df.reset_index()
            metrics_df.rename(columns={'index': 'Run'}, inplace=True)
            
            # Save metrics
            metrics_df.to_csv(metrics_file, index=False)
            self.logger.info(f"Performance metrics saved to {metrics_file}")
            
            # Create visualization
            try:
                self._plot_ensemble_comparison(all_flows, metrics_df.set_index('Run'), metrics_dir)
            except Exception as e:
                self.logger.error(f"Error creating ensemble plots: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            return metrics_file
                
        except Exception as e:
            self.logger.error(f"Error calculating ensemble performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_performance_metrics(self, observed, simulated):
        """
        Calculate streamflow performance metrics with improved handling of outliers.
        
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
            self.logger.error("No valid data points for metric calculation")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
        
        # Cap extremely high flow values (outliers) - use the 99.5th percentile
        flow_cap = observed.quantile(0.995)
        observed_capped = observed.clip(upper=flow_cap)
        simulated_capped = simulated.clip(upper=flow_cap)
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed_capped.mean()
        nse_numerator = ((observed_capped - simulated_capped) ** 2).sum()
        nse_denominator = ((observed_capped - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed_capped - simulated_capped) ** 2).mean())
        
        # Percent Bias (PBIAS)
        pbias = 100 * (simulated_capped.sum() - observed_capped.sum()) / observed_capped.sum() if observed_capped.sum() != 0 else np.nan
        
        # Kling-Gupta Efficiency (KGE)
        r = observed_capped.corr(simulated_capped)  # Correlation coefficient
        alpha = simulated_capped.std() / observed_capped.std() if observed_capped.std() != 0 else np.nan  # Relative variability
        beta = simulated_capped.mean() / mean_obs if mean_obs != 0 else np.nan  # Bias ratio
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        # Mean Absolute Error (MAE)
        mae = (observed_capped - simulated_capped).abs().mean()
        
        return {
            'KGE': kge,
            'NSE': nse,
            'RMSE': rmse,
            'PBIAS': pbias,
            'MAE': mae,
            'r': r,
            'alpha': alpha,
            'beta': beta
        }

    def _plot_ensemble_comparison(self, all_flows, metrics_df, output_dir):
        """
        Create visualizations comparing all ensemble runs to observed data with focus on training and testing periods.
        
        Args:
            all_flows: DataFrame with observed and simulated flows
            metrics_df: DataFrame with performance metrics
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.dates import DateFormatter
            import matplotlib.gridspec as gridspec
            
            # Handle outliers by capping values
            flow_cap = all_flows['Observed'].quantile(0.995) if 'Observed' in all_flows else 100  # Default cap if no observed data
            capped_flows = all_flows.clip(upper=flow_cap)
            
            # Ensure we actually have data to plot
            if capped_flows.empty:
                self.logger.warning("Empty flow data, cannot create ensemble plots")
                return
            
            # Get time information - assuming index is datetime
            date_range = capped_flows.index
            if len(date_range) == 0:
                self.logger.warning("Empty date range, cannot create ensemble plots")
                return
            
            # Get calibration and evaluation periods from config
            calib_period = self.config.get('CALIBRATION_PERIOD', '')
            eval_period = self.config.get('EVALUATION_PERIOD', '')
            
            if calib_period and ',' in calib_period and eval_period and ',' in eval_period:
                calib_start, calib_end = [s.strip() for s in calib_period.split(',')]
                eval_start, eval_end = [s.strip() for s in eval_period.split(',')]
                
                # Create masks for periods
                try:
                    calib_mask = (capped_flows.index >= pd.Timestamp(calib_start)) & (capped_flows.index <= pd.Timestamp(calib_end))
                    eval_mask = (capped_flows.index >= pd.Timestamp(eval_start)) & (capped_flows.index <= pd.Timestamp(eval_end))
                    spinup_mask = ~(calib_mask | eval_mask)
                except:
                    # If timestamps fail, use time-based split
                    spinup_end = date_range[0] + (date_range[-1] - date_range[0]) * 0.2
                    calib_end = date_range[0] + (date_range[-1] - date_range[0]) * 0.7
                    
                    spinup_mask = (date_range >= date_range[0]) & (date_range < spinup_end)
                    calib_mask = (date_range >= spinup_end) & (date_range < calib_end)
                    eval_mask = date_range >= calib_end
            else:
                # Use time-based split (20% spinup, 50% calibration, 30% evaluation)
                spinup_end = date_range[0] + (date_range[-1] - date_range[0]) * 0.2
                calib_end = date_range[0] + (date_range[-1] - date_range[0]) * 0.7
                
                spinup_mask = (date_range >= date_range[0]) & (date_range < spinup_end)
                calib_mask = (date_range >= spinup_end) & (date_range < calib_end)
                eval_mask = date_range >= calib_end
            
            # Calculate ensemble mean (excluding 'Observed' column)
            sim_cols = [col for col in capped_flows.columns if col != 'Observed' and not col.startswith('Unnamed')]
            if not sim_cols:
                self.logger.warning("No simulation columns found in data")
                return
                
            capped_flows['Ensemble Mean'] = capped_flows[sim_cols].mean(axis=1)
            
            # Create a figure with three subplots (spinup, training, testing)
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])
            
            # Create axes for each period
            ax_spinup = fig.add_subplot(gs[0])
            ax_train = fig.add_subplot(gs[1], sharex=ax_spinup)
            ax_test = fig.add_subplot(gs[2], sharex=ax_spinup)
            
            # Plot spinup period
            if spinup_mask.sum() > 0:
                spinup_flows = capped_flows[spinup_mask]
                if 'Observed' in spinup_flows:
                    ax_spinup.plot(spinup_flows.index, spinup_flows['Observed'], 'k-', linewidth=2)
                if 'Ensemble Mean' in spinup_flows:
                    ax_spinup.plot(spinup_flows.index, spinup_flows['Ensemble Mean'], 'r-', linewidth=1.5)
                    
                    # Plot each simulation with low alpha (limit to first 20 for clarity)
                    for col in sim_cols[:min(20, len(sim_cols))]:
                        ax_spinup.plot(spinup_flows.index, spinup_flows[col], 'b-', alpha=0.05, linewidth=0.5)
                    
                ax_spinup.set_title('Spinup Period', fontsize=12)
                ax_spinup.set_ylabel('Flow (m³/s)')
                ax_spinup.grid(True, alpha=0.3)
            
            # Plot calibration period
            if calib_mask.sum() > 0:
                calib_flows = capped_flows[calib_mask]
                if 'Observed' in calib_flows and 'Ensemble Mean' in calib_flows:
                    ax_train.plot(calib_flows.index, calib_flows['Observed'], 'k-', linewidth=2)
                    ax_train.plot(calib_flows.index, calib_flows['Ensemble Mean'], 'r-', linewidth=1.5)
                    
                    # Plot each simulation with low alpha (limit to first 20 for clarity)
                    for col in sim_cols[:min(20, len(sim_cols))]:
                        ax_train.plot(calib_flows.index, calib_flows[col], 'b-', alpha=0.05, linewidth=0.5)
                    
                    # Add metrics if we can calculate them
                    try:
                        train_metrics = self._calculate_period_metrics(calib_flows['Observed'], calib_flows['Ensemble Mean'])
                        metrics_text = (
                            f"Training Metrics:\n"
                            f"KGE: {train_metrics['KGE']:.3f}\n"
                            f"NSE: {train_metrics['NSE']:.3f}\n"
                            f"RMSE: {train_metrics['RMSE']:.3f}"
                        )
                        ax_train.text(0.02, 0.95, metrics_text, transform=ax_train.transAxes,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception as e:
                        self.logger.warning(f"Could not calculate training metrics: {str(e)}")
                    
                ax_train.set_title('Calibration Period', fontsize=12)
                ax_train.set_ylabel('Flow (m³/s)')
                ax_train.grid(True, alpha=0.3)
            
            # Plot evaluation period
            if eval_mask.sum() > 0:
                eval_flows = capped_flows[eval_mask]
                if 'Observed' in eval_flows and 'Ensemble Mean' in eval_flows:
                    ax_test.plot(eval_flows.index, eval_flows['Observed'], 'k-', linewidth=2)
                    ax_test.plot(eval_flows.index, eval_flows['Ensemble Mean'], 'r-', linewidth=1.5)
                    
                    # Plot each simulation with low alpha (limit to first 20 for clarity)
                    for col in sim_cols[:min(20, len(sim_cols))]:
                        ax_test.plot(eval_flows.index, eval_flows[col], 'b-', alpha=0.05, linewidth=0.5)
                    
                    # Add metrics if we can calculate them
                    try:
                        test_metrics = self._calculate_period_metrics(eval_flows['Observed'], eval_flows['Ensemble Mean'])
                        metrics_text = (
                            f"Evaluation Metrics:\n"
                            f"KGE: {test_metrics['KGE']:.3f}\n"
                            f"NSE: {test_metrics['NSE']:.3f}\n"
                            f"RMSE: {test_metrics['RMSE']:.3f}"
                        )
                        ax_test.text(0.02, 0.95, metrics_text, transform=ax_test.transAxes,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception as e:
                        self.logger.warning(f"Could not calculate evaluation metrics: {str(e)}")
                
                ax_test.set_title('Evaluation Period', fontsize=12)
                ax_test.set_xlabel('Date')
                ax_test.set_ylabel('Flow (m³/s)')
                ax_test.grid(True, alpha=0.3)
            
            # Format x-axis dates
            for ax in [ax_spinup, ax_train, ax_test]:
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add a common legend
            fig.legend(['Observed', 'Ensemble Mean', 'Ensemble Runs'], 
                    loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
            
            # Add overall ensemble information
            fig.text(0.02, 0.98, f"Ensemble Size: {len(sim_cols)} runs", fontsize=12)
            
            # Save plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the legend at the top
            plt.savefig(output_dir / "ensemble_streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create additional plots
            self._create_additional_ensemble_plots(capped_flows, sim_cols, calib_mask, eval_mask, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _calculate_period_metrics(self, observed, simulated):
        """
        Calculate metrics for a specific time period with improved robustness.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            Dictionary of performance metrics
        """
        # Clean and align data
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) < 2:
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}
        
        # Cap values at 99.5 percentile to reduce impact of outliers
        cap_val = observed.quantile(0.995)
        observed = observed.clip(upper=cap_val)
        simulated = simulated.clip(upper=cap_val)
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed.mean()
        nse_numerator = ((observed - simulated) ** 2).sum()
        nse_denominator = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # Kling-Gupta Efficiency (KGE)
        try:
            r = observed.corr(simulated)  # Correlation coefficient
            alpha = simulated.std() / observed.std() if observed.std() > 0 else np.nan  # Relative variability
            beta = simulated.mean() / mean_obs if mean_obs > 0 else np.nan  # Bias ratio
            
            if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
                kge = np.nan
            else:
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        except:
            kge = np.nan
        
        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse}


    def _create_additional_ensemble_plots(self, capped_flows, sim_cols, train_mask, test_mask, output_dir):
        """
        Create additional ensemble analysis plots (scatter, FDC, etc.) with improved error handling.
        
        Args:
            capped_flows: DataFrame with capped flow values
            sim_cols: List of simulation column names
            train_mask: Boolean mask for training period
            test_mask: Boolean mask for testing period
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.stats import linregress
            
            # Create scatter plot comparing observed vs ensemble mean for training and testing
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Training period scatter
            train_flows = capped_flows[train_mask].copy() if isinstance(train_mask, pd.Series) else pd.DataFrame()
            
            # First check if we have non-empty dataframe with both columns
            if not train_flows.empty and 'Observed' in train_flows and 'Ensemble Mean' in train_flows:
                # Filter out NaN values
                valid_mask = ~(train_flows['Observed'].isna() | train_flows['Ensemble Mean'].isna())
                if valid_mask.sum() > 0:
                    valid_train_flows = train_flows[valid_mask]
                    
                    # Convert to NumPy arrays for regression
                    x_vals = valid_train_flows['Observed'].values
                    y_vals = valid_train_flows['Ensemble Mean'].values
                    
                    # Plot scatter points
                    ax1.scatter(x_vals, y_vals, alpha=0.7, s=30, c='blue', label='Training')
                    
                    # Add 1:1 line
                    if len(x_vals) > 0:
                        max_val = max(valid_train_flows['Observed'].max(), valid_train_flows['Ensemble Mean'].max())
                        min_val = min(valid_train_flows['Observed'].min(), valid_train_flows['Ensemble Mean'].min())
                        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
                        
                        # Add regression line only if we have enough points
                        if len(x_vals) >= 2:
                            try:
                                # Simple approach to avoid shape errors
                                slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                                
                                # Only plot if we got valid results
                                if not np.isnan(slope) and not np.isnan(intercept):
                                    ax1.plot([min_val, max_val], 
                                        [slope * min_val + intercept, slope * max_val + intercept], 
                                        'r-', linewidth=1, label=f'Regression (r={r_value:.2f})')
                            except Exception as e:
                                self.logger.warning(f"Error calculating regression for training period: {str(e)}")
                    
                    ax1.set_xlabel('Observed Flow (m³/s)')
                    ax1.set_ylabel('Ensemble Mean Flow (m³/s)')
                    ax1.set_title('Training Period Scatter Plot')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
            
            # Testing period scatter
            test_flows = capped_flows[test_mask].copy() if isinstance(test_mask, pd.Series) else pd.DataFrame()
            
            # First check if we have non-empty dataframe with both columns
            if not test_flows.empty and 'Observed' in test_flows and 'Ensemble Mean' in test_flows:
                # Filter out NaN values
                valid_mask = ~(test_flows['Observed'].isna() | test_flows['Ensemble Mean'].isna())
                if valid_mask.sum() > 0:
                    valid_test_flows = test_flows[valid_mask]
                    
                    # Convert to NumPy arrays for regression
                    x_vals = valid_test_flows['Observed'].values
                    y_vals = valid_test_flows['Ensemble Mean'].values
                    
                    # Plot scatter points
                    ax2.scatter(x_vals, y_vals, alpha=0.7, s=30, c='red', label='Testing')
                    
                    # Add 1:1 line
                    if len(x_vals) > 0:
                        max_val = max(valid_test_flows['Observed'].max(), valid_test_flows['Ensemble Mean'].max())
                        min_val = min(valid_test_flows['Observed'].min(), valid_test_flows['Ensemble Mean'].min())
                        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
                        
                        # Add regression line only if we have enough points
                        if len(x_vals) >= 2:
                            try:
                                # Simple approach to avoid shape errors
                                slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                                
                                # Only plot if we got valid results
                                if not np.isnan(slope) and not np.isnan(intercept):
                                    ax2.plot([min_val, max_val], 
                                        [slope * min_val + intercept, slope * max_val + intercept], 
                                        'r-', linewidth=1, label=f'Regression (r={r_value:.2f})')
                            except Exception as e:
                                self.logger.warning(f"Error calculating regression for testing period: {str(e)}")
                    
                    ax2.set_xlabel('Observed Flow (m³/s)')
                    ax2.set_ylabel('Ensemble Mean Flow (m³/s)')
                    ax2.set_title('Testing Period Scatter Plot')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
            
            # Save the figure with equal aspect if possible
            for ax in [ax1, ax2]:
                try:
                    ax.set_aspect('equal')
                except:
                    self.logger.warning("Could not set equal aspect ratio for scatter plot")
            
            plt.tight_layout()
            plt.savefig(output_dir / "ensemble_scatter_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.error(f"Error creating additional ensemble plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


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

    def run_benchmarking(self):
        # Preprocess data for benchmarking
        preprocessor = BenchmarkPreprocessor(self.config, self.logger)
        benchmark_data = preprocessor.preprocess_benchmark_data(f"{self.config['CALIBRATION_PERIOD'].split(',')[0]}", f"{self.config['EVALUATION_PERIOD'].split(',')[1]}")

        # Run benchmarking
        benchmarker = Benchmarker(self.config, self.logger)
        benchmark_results = benchmarker.run_benchmarking()

        bv = BenchmarkVizualiser(self.config, self.logger)
        bv.visualize_benchmarks(benchmark_results)

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
            elif model == 'CLM_PARFLOW':
                # Add CLM-ParFlow postprocessing
                self.logger.info("Running CLM-ParFlow postprocessing")
                cpp = CLMParFlowPostProcessor(self.config, self.logger)
                results = cpp.process_results()
                
                if results and 'streamflow' in results:
                    results_file = results['streamflow']
                    self.logger.info(f"CLM-ParFlow streamflow extracted to: {results_file}")
                else:
                    self.logger.warning("CLM-ParFlow postprocessing did not produce streamflow results")
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

    def delineate_lumped_watershed(self):
        self.logger.info("Starting geofabric lumped delineation")
        try:
            delineator = LumpedWatershedDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_lumped_watershed()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None

    def delineate_point_buffer_shape(self):
        self.logger.info("Starting point buffer")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('point buffer completed successfully')
            return delineator.delineate_point_buffer_shape()
        except Exception as e:
            self.logger.error(f"Error during point buffer delineation: {str(e)}")
            return None

    def delineate_geofabric(self):
        self.logger.info("Starting geofabric delineation")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_geofabric()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None

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
