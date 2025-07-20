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
                
                # Enhanced routing logic for SUMMA
                if model == 'SUMMA':
                    routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                    domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                    
                    # Determine if we need mizuRoute
                    needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
                    
                    if needs_mizuroute:
                        self.logger.info("Initializing MizuRoute preprocessor")
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()
                        
            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        # Archive basin-averaged forcing data
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

        self.logger.info("Model-specific preprocessing completed")

    def run_models(self):
        """
        Run all configured hydrological models.
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
                
                # Enhanced routing logic for SUMMA
                if model == 'SUMMA':
                    routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                    domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                    
                    # Determine if we need mizuRoute
                    needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
                    
                    if needs_mizuroute:
                        # Handle lumped-to-distributed conversion if needed
                        if domain_method == 'lumped' and routing_delineation == 'river_network':
                            self.logger.info("Converting lumped SUMMA output for distributed routing")
                            self._convert_lumped_to_distributed_routing()
                        
                        # Run mizuRoute
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

    def _needs_mizuroute_routing(self, domain_method: str, routing_delineation: str) -> bool:
        """
        Determine if mizuRoute routing is needed based on domain and routing configuration.
        
        Args:
            domain_method (str): Domain definition method
            routing_delineation (str): Routing delineation method
            
        Returns:
            bool: True if mizuRoute routing is needed
        """
        # Original logic: distributed domain always needs routing
        if domain_method not in ['point', 'lumped']:
            return True
        
        # New logic: lumped domain with river network routing
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        
        # Point simulations never need routing
        # Lumped domain with lumped routing doesn't need mizuRoute
        return False

    def _convert_lumped_to_distributed_routing(self):
        """
        Convert lumped SUMMA output to distributed mizuRoute forcing.
        
        This method replicates the functionality from the manual script,
        broadcasting the single lumped runoff to all routing segments.
        Creates mizuRoute-compatible files with proper naming and variables.
        Only processes the timestep file for mizuRoute routing.
        """
        self.logger.info("Converting lumped SUMMA output for distributed routing")
        
        try:
            # Import required modules
            import xarray as xr
            import numpy as np
            import pandas as pd
            import netCDF4 as nc4
            import shutil
            import tempfile
            
            experiment_id = self.config.get('EXPERIMENT_ID')
            
            # Paths
            summa_output_dir = self.project_dir / "simulations" / experiment_id / "SUMMA"
            mizuroute_settings_dir = self.project_dir / "settings" / "mizuRoute"
            
            # Only process the timestep file for mizuRoute
            summa_timestep_file = summa_output_dir / f"{experiment_id}_timestep.nc"
            
            if not summa_timestep_file.exists():
                raise FileNotFoundError(f"SUMMA timestep output file not found: {summa_timestep_file}")
            
            # Load mizuRoute topology to get segment information
            topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
            
            if not topology_file.exists():
                raise FileNotFoundError(f"mizuRoute topology file not found: {topology_file}")
            
            with xr.open_dataset(topology_file) as mizuTopology:
                # Use SEGMENT IDs from topology (these are the routing elements we broadcast to)
                # This matches the original manual script approach
                seg_ids = mizuTopology['segId'].values
                n_segments = len(seg_ids)
                
                # Also get HRU info for context
                hru_ids = mizuTopology['hruId'].values if 'hruId' in mizuTopology else []
                n_hrus = len(hru_ids)
            
            self.logger.info(f"Broadcasting to {n_segments} routing segments ({n_hrus} HRUs in topology)")
            
            # Check if we actually have a distributed routing network
            if n_segments <= 1:
                self.logger.warning(f"Only {n_segments} routing segment(s) found in topology. Distributed routing may not be beneficial.")
                self.logger.warning("Consider using ROUTING_DELINEATION: lumped instead")
            
            # Get the routing variable name from config (updated defaults for new SUMMA versions)
            routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
            
            self.logger.info(f"Processing {summa_timestep_file.name}")
            
            # Load SUMMA output with time decoding disabled to avoid conversion issues
            summa_output = xr.open_dataset(summa_timestep_file, decode_times=False)
            
            try:
                # Create mizuRoute forcing dataset with proper structure
                mizuForcing = xr.Dataset()
                
                # Handle time coordinate properly - copy original time values and attributes
                original_time = summa_output['time']
                
                # Use the original time values and attributes directly
                # This preserves whatever format SUMMA created
                mizuForcing['time'] = xr.DataArray(
                    original_time.values,
                    dims=('time',),
                    attrs=dict(original_time.attrs)  # Copy all original attributes
                )
                
                # Clean up the time units if needed (remove 'T' separator)
                if 'units' in mizuForcing['time'].attrs:
                    time_units = mizuForcing['time'].attrs['units']
                    if 'T' in time_units:
                        mizuForcing['time'].attrs['units'] = time_units.replace('T', ' ')
                
                self.logger.info(f"Preserved original time coordinate: {mizuForcing['time'].attrs.get('units', 'no units')}")
                
                # Create GRU dimension using SEGMENT IDs (this is the key fix!)
                # This matches the original script: mizuTopology['segId'].values.flatten()
                mizuForcing['gru'] = xr.DataArray(
                    seg_ids, 
                    dims=('gru',),
                    attrs={'long_name': 'Index of GRU', 'units': '-'}
                )
                
                # GRU ID variable (use segment IDs as GRU IDs for routing)
                mizuForcing['gruId'] = xr.DataArray(
                    seg_ids, 
                    dims=('gru',),
                    attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                )
                
                # Copy global attributes from SUMMA output
                mizuForcing.attrs.update(summa_output.attrs)
                
                # Find the best variable to broadcast (updated for new SUMMA variable names)
                source_var = None
                available_vars = list(summa_output.variables.keys())
                
                # Check for exact match first
                if routing_var in summa_output:
                    source_var = routing_var
                    self.logger.info(f"Using configured routing variable: {routing_var}")
                else:
                    # Try fallback variables in order of preference (updated for new naming)
                    fallback_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'averageRoutedRunoff_mean', 'basin__TotalRunoff_mean']
                    for var in fallback_vars:
                        if var in summa_output:
                            source_var = var
                            self.logger.info(f"Routing variable {routing_var} not found, using: {source_var}")
                            break
                
                if source_var is None:
                    runoff_vars = [v for v in available_vars 
                                if 'runoff' in v.lower() or 'discharge' in v.lower()]
                    self.logger.error(f"No suitable runoff variable found.")
                    self.logger.error(f"Available variables: {available_vars}")
                    self.logger.error(f"Runoff-related variables: {runoff_vars}")
                    raise ValueError(f"No suitable runoff variable found. Available: {runoff_vars}")
                
                # Extract the lumped runoff (should be single value per time step)
                lumped_runoff = summa_output[source_var].values
                
                # Handle different shapes (time,) or (time, 1) or (time, n_gru)
                if len(lumped_runoff.shape) == 1:
                    # Already correct shape (time,)
                    pass
                elif len(lumped_runoff.shape) == 2:
                    if lumped_runoff.shape[1] == 1:
                        # Shape (time, 1) - flatten to (time,)
                        lumped_runoff = lumped_runoff.flatten()
                    else:
                        # Multiple GRUs - take the first one (lumped should only have 1)
                        lumped_runoff = lumped_runoff[:, 0]
                        self.logger.warning(f"Multiple GRUs found in lumped simulation, using first GRU")
                else:
                    raise ValueError(f"Unexpected runoff data shape: {lumped_runoff.shape}")
                
                # Tile to all SEGMENTS: (time,) -> (time, n_segments)
                # This is the key fix - broadcast to segments, not HRUs
                tiled_data = np.tile(lumped_runoff[:, np.newaxis], (1, n_segments))
                
                # Create the routing variable with the expected name
                mizuForcing[routing_var] = xr.DataArray(
                    tiled_data,
                    dims=('time', 'gru'),
                    attrs={
                        'long_name': 'Broadcast runoff for distributed routing',
                        'units': 'm/s'
                    }
                )
                
                self.logger.info(f"Broadcast {source_var} -> {routing_var} to {n_segments} segments")
                
                # Close the original dataset to release the file
                summa_output.close()
                
                # Backup original file before overwriting
                backup_file = summa_output_dir / f"{summa_timestep_file.stem}_original.nc"
                if not backup_file.exists():
                    shutil.copy2(summa_timestep_file, backup_file)
                    self.logger.info(f"Backed up original SUMMA output to {backup_file.name}")
                
                # Write to temporary file first, then move to final location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=summa_output_dir) as tmp_file:
                    temp_path = tmp_file.name
                
                # Save with explicit format but let xarray handle time encoding
                mizuForcing.to_netcdf(temp_path, format='NETCDF4')
                mizuForcing.close()
                
                # Move temporary file to final location
                shutil.move(temp_path, summa_timestep_file)
                
                self.logger.info(f"Created mizuRoute forcing: {summa_timestep_file}")
                
            except Exception as e:
                # Make sure to close the summa_output dataset
                summa_output.close()
                raise
            
            self.logger.info("Lumped-to-distributed conversion completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error converting lumped output: {str(e)}")
            raise

    def _fix_mizuroute_time_units(self, nc_file_path, nc4_module):
        """
        Fix time units in mizuRoute forcing file to ensure compatibility.
        
        Args:
            nc_file_path (Path): Path to the NetCDF file to fix
            nc4_module: The netCDF4 module reference
        """
        try:
            with nc4_module.Dataset(str(nc_file_path), 'r+') as ncid:
                # Access the 'units' attribute and replace 'T' with a space
                if 'time' in ncid.variables and hasattr(ncid['time'], 'units'):
                    units_attribute = ncid['time'].units
                    units_attribute = units_attribute.replace('T', ' ')
                    ncid['time'].setncattr('units', units_attribute)
            
            self.logger.debug(f"Fixed time units in {nc_file_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not fix time units in {nc_file_path}: {str(e)}")
        
    def visualize_outputs(self):
        """
        Visualize model outputs.
        """
        self.logger.info('Starting model output visualisation')
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            visualizer = VisualizationReporter(self.config, self.logger)
            
            if model == 'SUMMA':
                #visualizer.plot_summa_outputs(self.experiment_id)
                
                # Define observation files
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                                f"{self.domain_name}_streamflow_processed.csv"))
                ]
                
                # Determine output source based on routing configuration
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', '')
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                
                # Check if mizuRoute was used
                uses_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
                
                if uses_mizuroute:
                    # Use MizuRoute output for visualization
                    self.logger.info("Using mizuRoute output for visualization")
                    visualizer.update_sim_reach_id()
                    model_outputs = [
                        (model, str(self.project_dir / "simulations" / self.experiment_id / 
                                "mizuRoute" / f"{self.experiment_id}*.nc"))
                    ]
                    visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                else:
                    # Use SUMMA output directly for lumped visualization
                    self.logger.info("Using lumped SUMMA output for visualization")
                    summa_output_file = str(self.project_dir / "simulations" / self.experiment_id / 
                                        "SUMMA" / f"{self.experiment_id}_timestep.nc")
                    model_outputs = [(model, summa_output_file)]
                    visualizer.plot_lumped_streamflow_simulations_vs_observations(model_outputs, obs_files)
                        
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