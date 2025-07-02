# In utils/models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any

# Model preprocessors
from utils.models.summa_utils import SummaPreProcessor # type: ignore
from utils.models.fuse_utils import FUSEPreProcessor # type: ignore
from utils.models.gr_utils import GRPreProcessor # type: ignore
from utils.models.hype_utils import HYPEPreProcessor # type: ignore
from utils.models.flash_utils import FLASH # type: ignore
from utils.models.mizuroute_utils import MizuRoutePreProcessor # type: ignore
#from utils.models.mesh_utils import MESHPreProcessor # type: ignore

# Model runners
from utils.models.summa_utils import SummaRunner # type: ignore
from utils.models.fuse_utils import FUSERunner # type: ignore
from utils.models.gr_utils import GRRunner # type: ignore
from utils.models.hype_utils import HYPERunner # type: ignore
from utils.models.flash_utils import FLASH # type: ignore
from utils.models.mizuroute_utils import MizuRouteRunner # type: ignore
#from utils.models.mesh_utils import MESHRunner # type: ignore

# Model postprocessors
from utils.models.summa_utils import SUMMAPostprocessor # type: ignore
from utils.models.fuse_utils import FUSEPostprocessor # type: ignore
from utils.models.gr_utils import GRPostprocessor # type: ignore
from utils.models.hype_utils import HYPEPostProcessor # type: ignore
from utils.models.flash_utils import FLASHPostProcessor # type: ignore
#from utils.models.mesh_utils import MESHPostProcessor # type: ignore

# Visualization
from utils.reporting.reporting_utils import VisualizationReporter # type: ignore
from utils.reporting.result_vizualisation_utils import TimeseriesVisualizer # type: ignore

# Data management
from utils.data.archive_utils import tar_directory # type: ignore

class ModelManager:
    """
    Manages all hydrological model operations within the CONFLUENCE framework.
    
    The ModelManager is responsible for coordinating the execution of hydrological
    models, including preprocessing of input data, executing model simulations,
    postprocessing results, and visualizing outputs. It provides a unified interface
    for interacting with multiple hydrological models, allowing for consistent
    execution and comparison.
    
    The ModelManager supports multiple hydrological models:
    - SUMMA: Structure for Unifying Multiple Modeling Alternatives
    - FUSE: Framework for Understanding Structural Errors
    - GR: Génie Rural models (GR4J, GR5J, etc.)
    - HYPE: HYdrological Predictions for the Environment
    - FLASH: A deep learning streamflow prediction model
    - (MESH): Modélisation Environnementale Communautaire - Surface et Hydrologie
              (currently commented out)
              
    For each model, the ModelManager provides standardized interfaces for:
    1. Preprocessing: Converting common input data to model-specific formats
    2. Execution: Running the hydrological simulations
    3. Postprocessing: Extracting and formatting simulation results
    4. Visualization: Creating standardized plots and visualizations
    
    The ModelManager dynamically selects the appropriate model components based
    on the configuration, allowing for flexible model selection and comparison.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        data_dir (Path): Path to the CONFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        experiment_id (str): ID of the current experiment
        preprocessors (Dict[str, Any]): Mapping of model names to preprocessor classes
        runners (Dict[str, Any]): Mapping of model names to runner classes
        postprocessors (Dict[str, Any]): Mapping of model names to postprocessor classes
        runner_methods (Dict[str, str]): Mapping of model names to runner method names
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Model Manager.
        
        Sets up the ModelManager with mappings between model names and their
        corresponding preprocessor, runner, and postprocessor classes. This provides
        a flexible architecture that allows for dynamic model selection based on
        the configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            logger (logging.Logger): Logger instance for recording operations
            
        Raises:
            KeyError: If essential configuration values are missing
            ImportError: If required model utility modules cannot be imported
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Define preprocessor mapping
        self.preprocessors = {
            'SUMMA': SummaPreProcessor,
            'FUSE': FUSEPreProcessor,
            'GR': GRPreProcessor,
            'HYPE': HYPEPreProcessor,
            'FLASH': None,  # FLASH doesn't have a separate preprocessor
            'MESH': None,  # MESHPreProcessor (commented out)
        }
        
        # Define runner mapping
        self.runners = {
            'SUMMA': SummaRunner,
            'FUSE': FUSERunner,
            'GR': GRRunner,
            'HYPE': HYPERunner,
            'FLASH': FLASH,
            'MESH': None,  # MESHRunner (commented out)
        }
        
        # Define postprocessor mapping
        self.postprocessors = {
            'SUMMA': SUMMAPostprocessor,
            'FUSE': FUSEPostprocessor,
            'GR': GRPostprocessor,
            'HYPE': HYPEPostProcessor,
            'FLASH': FLASHPostProcessor,
            'MESH': None,  # MESHPostProcessor (commented out)
        }
        
        # Define runner method names (since they're not standardized)
        self.runner_methods = {
            'SUMMA': 'run_summa',
            'FUSE': 'run_fuse',
            'GR': 'run_gr',
            'HYPE': 'run_hype',
            'FLASH': 'run_flash',
            'MESH': 'run_MESH',
        }
    
    def preprocess_models(self):
        """
        Process the forcing data into model-specific formats.
        
        This method converts the model-agnostic forcing data into formats required
        by each specific hydrological model. For each configured model, it:
        1. Creates the model-specific input directory
        2. Loads the appropriate preprocessor based on the model type
        3. Executes the preprocessing operations
        
        Special handling is provided for SUMMA when used with the MizuRoute
        routing model, which requires additional preprocessing for distributed
        and semi-distributed simulations.
        
        The method iterates through all models specified in the HYDROLOGICAL_MODEL
        configuration parameter, which can be a comma-separated list for multi-model
        simulations.
        
        Raises:
            FileNotFoundError: If required input files cannot be found
            ValueError: If preprocessing parameters are invalid
            Exception: For other model-specific preprocessing errors
        """
        self.logger.info("Starting model-specific preprocessing")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                # Create model input directory
                model_input_dir = self.project_dir / "forcing" / f"{model}_input"
                model_input_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"Processing model: {model}")
                
                # Get preprocessor class
                preprocessor_class = self.preprocessors.get(model)
                
                if preprocessor_class is None:
                    if model in self.preprocessors:
                        self.logger.info(f"Model {model} doesn't require preprocessing")
                    else:
                        self.logger.warning(f"Unsupported model: {model}. No preprocessing performed.")
                    continue
                
                # Run preprocessing
                preprocessor = preprocessor_class(self.config, self.logger)
                preprocessor.run_preprocessing()
                
                # Special handling for SUMMA routing (MizuRoute)
                if model == 'SUMMA':
                    spatial_mode = self.config.get('DOMAIN_DEFINITION_METHOD', '')
                    if spatial_mode not in ['point', 'lumped']:
                        self.logger.info("Initializing MizuRoute preprocessor")
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()
                        
            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        # Archive basin-averaged forcing data to save storage space
        self.logger.info("Archiving basin-averaged forcing data to save storage space")
        try:
            basin_data_dir = self.project_dir / 'forcing' / 'basin_averaged_data'
            if basin_data_dir.exists():
                success = tar_directory(
                    basin_data_dir,
                    "basin_averaged_forcing_data.tar.gz",
                    remove_original=True,
                    logger=self.logger
                )
                if success:
                    self.logger.info("Basin-averaged forcing data archived successfully")
                else:
                    self.logger.warning("Failed to archive basin-averaged forcing data")
        except Exception as e:
            self.logger.warning(f"Error during basin-averaged data archiving: {str(e)}")
            # Continue execution even if archiving fails

        self.logger.info("Model-specific preprocessing completed")
    
    def run_models(self):
        """
        Run all configured hydrological models.
        
        This method executes the simulation for each hydrological model specified
        in the configuration. For each model, it:
        1. Loads the appropriate runner class
        2. Creates a runner instance
        3. Executes the model-specific run method
        4. Handles any routing operations (e.g., MizuRoute for SUMMA)
        
        After all models have been executed, the method calls visualize_outputs()
        to generate standardized visualizations of the simulation results.
        
        The method supports multiple models through the HYDROLOGICAL_MODEL 
        configuration parameter, which can be a comma-separated list for multi-model
        simulations and comparisons.
        
        Special handling is provided for SUMMA when used with spatially distributed
        configurations, which requires the MizuRoute routing model to route runoff
        through the stream network.
        
        Raises:
            FileNotFoundError: If model executable or input files cannot be found
            RuntimeError: If model execution fails
            Exception: For other model-specific execution errors
        """
        self.logger.info("Starting model runs")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                self.logger.info(f"Running model: {model}")
                
                # Get runner class
                runner_class = self.runners.get(model)
                
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model: {model}")
                    continue
                
                # Create runner instance
                runner = runner_class(self.config, self.logger)
                
                # Get method name and run
                method_name = self.runner_methods.get(model)
                if method_name and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method not found for model: {model}")
                    continue
                
                # Special handling for SUMMA routing (MizuRoute)
                if model == 'SUMMA':
                    domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', '')
                    
                    if domain_method != 'lumped' and domain_method != 'point':
                        mizuroute_runner = MizuRouteRunner(self.config, self.logger)
                        mizuroute_runner.run_mizuroute()
                
                self.logger.info(f"{model} model run completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error during {model} model run: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # After all models run, visualize outputs
        self.visualize_outputs()
        self.logger.info("Model runs completed")
    
    def visualize_outputs(self):
        """
        Visualize model outputs.
        
        This method creates standardized visualizations of the simulation results
        for each hydrological model. It handles different visualization approaches
        based on the model type and spatial configuration (lumped vs. distributed).
        
        The visualizations include:
        - Time series plots of streamflow (simulated vs. observed)
        - Model-specific state and flux variables
        - Performance metrics
        
        For SUMMA, special handling is provided based on the domain configuration:
        - For lumped models, SUMMA output is used directly
        - For distributed models, MizuRoute output is used for streamflow
        
        The method uses the VisualizationReporter to create the plots, which are
        saved in the project's plots directory.
        
        Raises:
            FileNotFoundError: If model output or observation files cannot be found
            ValueError: If visualization parameters are invalid
            Exception: For other visualization errors
        """
        self.logger.info('Starting model output visualisation')
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            visualizer = VisualizationReporter(self.config, self.logger)
            
            if model == 'SUMMA':
                visualizer.plot_summa_outputs(self.experiment_id)
                
                # Define observation files
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                                   f"{self.domain_name}_streamflow_processed.csv"))
                ]
                
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', '')
                
                if domain_method == 'lumped':
                    # For lumped model, use SUMMA output directly
                    summa_output_file = str(self.project_dir / "simulations" / self.experiment_id / 
                                          "SUMMA" / f"{self.experiment_id}_timestep.nc")
                    model_outputs = [(model, summa_output_file)]
                    
                    self.logger.info(f"Using lumped model output from {summa_output_file}")
                    visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
                else:
                    # For distributed model, use MizuRoute output
                    visualizer.update_sim_reach_id()
                    model_outputs = [
                        (model, str(self.project_dir / "simulations" / self.experiment_id / 
                                  "mizuRoute" / f"{self.experiment_id}*.nc"))
                    ]
                    visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                    
            elif model == 'FUSE':
                model_outputs = [
                    ("FUSE", str(self.project_dir / "simulations" / self.experiment_id / 
                               "FUSE" / f"{self.domain_name}_{self.experiment_id}_runs_best.nc"))
                ]
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                                   f"{self.domain_name}_streamflow_processed.csv"))
                ]
                visualizer.plot_fuse_streamflow_simulations_vs_observations(model_outputs, obs_files)
                
            elif model in ['GR', 'FLASH', 'HYPE', 'MESH']:
                # Placeholder for other model visualizations
                self.logger.info(f"Visualization for {model} not yet implemented")
    
    def postprocess_results(self):
        """
        Post-process model results.
        
        This method extracts and processes the raw simulation outputs from each
        hydrological model into standardized formats for analysis and visualization.
        For each model, it:
        1. Loads the appropriate postprocessor class
        2. Creates a postprocessor instance
        3. Extracts streamflow or other results from model outputs
        
        After extracting results from all models, the method creates final time
        series visualizations using the TimeseriesVisualizer.
        
        The postprocessing operations typically include:
        - Extracting streamflow time series from model outputs
        - Computing performance metrics
        - Converting units if necessary
        - Storing results in standardized formats
        
        Raises:
            FileNotFoundError: If model output files cannot be found
            ValueError: If postprocessing parameters are invalid
            Exception: For other model-specific postprocessing errors
        """
        self.logger.info("Starting model post-processing")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                # Get postprocessor class
                postprocessor_class = self.postprocessors.get(model)
                
                if postprocessor_class is None:
                    self.logger.info(f"No postprocessor defined for model: {model}")
                    continue
                
                # Create postprocessor instance
                postprocessor = postprocessor_class(self.config, self.logger)
                
                # Run postprocessing
                if hasattr(postprocessor, 'extract_streamflow'):
                    results_file = postprocessor.extract_streamflow()
                elif hasattr(postprocessor, 'extract_results'):
                    results_file = postprocessor.extract_results()
                else:
                    self.logger.warning(f"No extraction method found for {model} postprocessor")
                    continue
                
                self.logger.info(f"Post-processing completed for {model}")
                
            except Exception as e:
                self.logger.error(f"Error post-processing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Create final visualizations
        try:
            tv = TimeseriesVisualizer(self.config, self.logger)
            metrics_df = tv.create_visualizations()
            self.logger.info("Time series visualizations created")
        except Exception as e:
            self.logger.error(f"Error creating time series visualizations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())