# In utils/models_utils/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, List, Optional

# Model preprocessors
from utils.models_utils.summa_utils import SummaPreProcessor # type: ignore
from utils.models_utils.fuse_utils import FUSEPreProcessor # type: ignore
from utils.models_utils.gr_utils import GRPreProcessor # type: ignore
from utils.models_utils.hype_utils import HYPEPreProcessor # type: ignore
from utils.models_utils.flash_utils import FLASH # type: ignore
from utils.models_utils.mizuroute_utils import MizuRoutePreProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPreProcessor # type: ignore

# Model runners
from utils.models_utils.summa_utils import SummaRunner # type: ignore
from utils.models_utils.fuse_utils import FUSERunner # type: ignore
from utils.models_utils.gr_utils import GRRunner # type: ignore
from utils.models_utils.hype_utils import HYPERunner # type: ignore
from utils.models_utils.flash_utils import FLASH # type: ignore
from utils.models_utils.mizuroute_utils import MizuRouteRunner # type: ignore
#from utils.models_utils.mesh_utils import MESHRunner # type: ignore

# Model postprocessors
from utils.models_utils.summa_utils import SUMMAPostprocessor # type: ignore
from utils.models_utils.fuse_utils import FUSEPostprocessor # type: ignore
from utils.models_utils.gr_utils import GRPostprocessor # type: ignore
from utils.models_utils.hype_utils import HYPEPostProcessor # type: ignore
from utils.models_utils.flash_utils import FLASHPostProcessor # type: ignore
#from utils.models_utils.mesh_utils import MESHPostProcessor # type: ignore

# Visualization
from utils.reporting.reporting_utils import VisualizationReporter # type: ignore
from utils.reporting.result_vizualisation_utils import TimeseriesVisualizer # type: ignore


class ModelManager:
    """Manages all model operations including preprocessing, running, postprocessing, and visualization."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Model Manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
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
        """Process the forcing data into model-specific formats."""
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
                    spatial_mode = self.config.get('SPATIAL_MODE', '')
                    if spatial_mode not in ['Point', 'Lumped']:
                        self.logger.info("Initializing MizuRoute preprocessor")
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()
                        
            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                
        self.logger.info("Model-specific preprocessing completed")
    
    def run_models(self):
        """Run all configured hydrological models."""
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
                    spatial_mode = self.config.get('SPATIAL_MODE', '')
                    
                    if domain_method != 'lumped' and spatial_mode != 'Point':
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
        """Visualize model outputs."""
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
                    visualizer.update_sim_reach_id(self.config)
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
        """Post-process model results."""
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