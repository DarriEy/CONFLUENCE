from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from pathlib import Path
from datetime import datetime
import geopandas as gpd # type: ignore
from shapely.geometry import Point # type: ignore
from mpi4py import MPI # type: ignore

sys.path.append(str(Path(__file__).resolve().parent))
from utils.data_utils import DataAcquisitionProcessor, DataPreProcessor, ProjectInitialisation # type: ignore  
from utils.model_utils import SummaRunner, MizuRouteRunner # type: ignore
from utils.optimization_utils import Optimizer # type: ignore
from utils.reporting_utils import VisualizationReporter # type: ignore
from utils.logging_utils import setup_logger, get_function_logger # type: ignore
from utils.config_utils import ConfigManager # type: ignore
from utils.geofabric_utils import GeofabricSubsetter, GeofabricDelineator, LumpedWatershedDelineator # type: ignore
from utils.discretization_utils import DomainDiscretizer # type: ignore
from utils.summa_utils import SummaPreProcessor # type: ignore
from utils.summa_spatial_utils import SummaPreProcessor_spatial # type: ignore
from utils.mizuroute_utils import MizuRoutePreProcessor # type: ignore
from utils.workflow_utils import WorkflowManager # type: ignore
from utils.forecasting_utils import ForecastingEngine # type: ignore

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
        log_dir = self.project_dir / f'_workLog_{self.config.get('DOMAIN_NAME')}'
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_general_{self.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_general', log_file)

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
    def discretize_domain(self):
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        domain_discretizer = DomainDiscretizer(self.config, self.logger)
        hru_shapefile = domain_discretizer.discretize_domain()
        if hru_shapefile:
            self.logger.info(f"Domain discretized successfully. HRU shapefile: {hru_shapefile}")
        else:
            self.logger.error("Domain discretization failed.")

        self.logger.info(f"Domain to be defined using method {domain_method}")

    @get_function_logger
    def define_domain(self):
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        
        if domain_method == 'subset':
            self.subset_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        elif domain_method == 'lumped':
            self.delineate_lumped_watershed(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        elif domain_method == 'delineate':
            self.delineate_geofabric(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"shapefiles/_workLog")
        else:
            self.logger.error(f"Unknown domain definition method: {domain_method}")

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

    @get_function_logger
    def process_input_data(self):
        self.logger.info("Starting input data processing")
        
        # Create DataAcquisitionProcessor instance
        if self.config.get('DATA_ACQUIRE') == 'HPC':
            self.logger.info('Data acquisition set to HPC')
            data_acquisition = DataAcquisitionProcessor(self.config, self.logger)
            
            # Run data acquisition
            try:
                data_acquisition.run_data_acquisition()
            except Exception as e:
                self.logger.error(f"Error during data acquisition: {str(e)}")
                raise
        
        elif self.config.get('DATA_ACQUIRE') == 'supplied':
            self.logger.info('Model input data set to supplied by user')
            data_preprocessor = DataPreProcessor(self.config, self.logger)
            data_preprocessor.process_zonal_statistics()
            
        self.logger.info("Input data processing completed")

    @get_function_logger
    def run_model_specific_preprocessing(self):
        # Run model-specific preprocessing
        if self.config.get('HYDROLOGICAL_MODEL') == 'SUMMA':
            self.run_summa_preprocessing(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"settings/_workLog")

    @get_function_logger
    def run_summa_preprocessing(self):
        self.logger.info('Running SUMMA specific input processing')
        if self.config.get('SUMMAFLOW') == 'stochastic':
            summa_preprocessor = SummaPreProcessor(self.config, self.logger)
            self.logger.info('SUMMA stochastic preprocessor initiated')
        if self.config.get('SUMMAFLOW') == 'spatial':
            summa_preprocessor = SummaPreProcessor_spatial(self.config, self.logger)
            self.logger.info('SUMMA spatial preprocessor initiated')

        summa_preprocessor.run_preprocessing(work_log_dir=self.project_dir/ f"_workLog_{self.domain_name}")
        
        mizuroute_preprocessor = MizuRoutePreProcessor(self.config, self.logger)
        mizuroute_preprocessor.run_preprocessing(work_log_dir=self.project_dir/ f"_workLog_{self.domain_name}")

    @get_function_logger
    def run_models(self):
        summa_runner = SummaRunner(self.config, self.logger)
        mizuroute_runner = MizuRouteRunner(self.config, self.logger)

        try:
            summa_runner.run_summa()
            mizuroute_runner.run_mizuroute()
            self.logger.info("Model runs completed successfully")
        except Exception as e:
            self.logger.error(f"Error during model runs: {str(e)}")

    @get_function_logger
    def visualise_model_output(self):\
        # Plot streamflow comparison
        self.logger.info('Starting model output visualisation')
        visualizer = VisualizationReporter(self.config, self.logger)
        model_outputs = [
            (f'{self.config.get('HYDROLOGICAL_MODEL')}', str(self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "mizuRoute" / f"{self.config.get('EXPERIMENT_ID')}.h.{self.config.get('FORCING_START_YEAR')}-01-01-00000.nc"))
        ]
        obs_files = [
            ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
        ]
        plot_file = visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
        if plot_file:
            self.logger.info(f"Streamflow comparison plot created: {plot_file}")
        else:
            self.logger.error("Failed to create streamflow comparison plot")

    @get_function_logger
    def run_workflow(self):
        self.logger.info("Starting CONFLUENCE workflow")
        
        # Check if we should force run all steps
        force_run = self.config.get('FORCE_RUN_ALL_STEPS', False)
        
        # Define the workflow steps and their output checks
        workflow_steps = [
            (self.setup_project, (self.project_dir / 'catchment').exists),
            (self.create_pourPoint, lambda: (self.project_dir / "shapefiles" / "pour_point" / f"{self.domain_name}_pourPoint.shp").exists()),
            (self.define_domain, lambda: (self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()),
            (self.discretize_domain, lambda: (self.project_dir / "shapefiles" / "catchment" / f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()),
            (self.process_input_data, lambda: (self.project_dir / "forcing" / "raw_data").exists()),
            (self.run_model_specific_preprocessing, lambda: (self.project_dir / "forcing" / f"{self.config.get('HYDROLOGICAL_MODEL')}_input").exists()),
            (self.run_models, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}" / f"{self.config.get('HYDROLOGICAL_MODEL')}" / f"{self.config.get('EXPERIMENT_ID')}_timestep.nc").exists()),
            (self.visualise_model_output, lambda: (self.project_dir / "plots" / "results" / "streamflow_comparison.png").exists()),
            (self.calibrate_model, lambda: (self.project_dir / "simulations" / f"{self.config.get('EXPERIMENT_ID')}_rank1" / f"{self.config.get('HYDROLOGICAL_MODEL')}" / f"{self.config.get('EXPERIMENT_ID')}_rank87_timestep.nc").exists()),
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
    def calibrate_model(self):
        # Calibrate the model using specified method and objectives
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        optimization_calibrator = Optimizer(self.config, self.logger, comm, rank)
        optimization_calibrator.run_optimization(work_log_dir=self.data_dir / f"domain_{self.domain_name}" / f"simulations/_workLog")


    def run_sensitivity_analysis(self, method, parameters):
        # Perform sensitivity analysis on model parameters
        pass

    def generate_forecast(self, forecast_horizon, ensemble_size):
        # Generate hydrological forecasts
        pass

    def analyze_results(self, analysis_type, metrics):
        # Analyze model outputs and forecasts
        pass

    def visualize_results(self, visualization_type, output_format):
        # Create visualizations of results
        pass

    def export_results(self, format, destination):
        # Export results in specified format
        pass


def main():
    config_path = Path(__file__).parent / '0_config_files'
    config_name = 'config_active.yaml'

    confluence = CONFLUENCE(config_path / config_name)
    confluence.run_workflow()
    
if __name__ == "__main__":
    main()