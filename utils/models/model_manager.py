# In utils/models/model_manager.py

from pathlib import Path
import logging
from typing import Dict, Any
import pandas as pd 

# Model preprocessors
from utils.models.summa_utils import SummaPreProcessor # type: ignore
from utils.models.fuse_utils import FUSEPreProcessor # type: ignore
from utils.models.gr_utils import GRPreProcessor # type: ignore
from utils.models.hype_utils import HYPEPreProcessor # type: ignore
from utils.models.flash_utils import FLASH # type: ignore
from utils.models.mizuroute_utils import MizuRoutePreProcessor # type: ignore
#from utils.models.mesh_utils import MESHPreProcessor # type: ignore
from utils.models.ngen_utils import NgenPreProcessor # type: ignore

# Model runners
from utils.models.summa_utils import SummaRunner # type: ignore
from utils.models.fuse_utils import FUSERunner # type: ignore
from utils.models.gr_utils import GRRunner # type: ignore
from utils.models.hype_utils import HYPERunner # type: ignore
from utils.models.flash_utils import FLASH # type: ignore
from utils.models.mizuroute_utils import MizuRouteRunner # type: ignore
#from utils.models.mesh_utils import MESHRunner # type: ignore
from utils.models.ngen_utils import NgenRunner # type: ignore

# Model postprocessors
from utils.models.summa_utils import SUMMAPostprocessor # type: ignore
from utils.models.fuse_utils import FUSEPostprocessor # type: ignore
from utils.models.gr_utils import GRPostprocessor # type: ignore
from utils.models.hype_utils import HYPEPostProcessor # type: ignore
from utils.models.flash_utils import FLASHPostProcessor # type: ignore
#from utils.models.mesh_utils import MESHPostProcessor # type: ignore
from utils.models.ngen_utils import NgenPostprocessor # type: ignore


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
            'NGEN': NgenPreProcessor,
            'FLASH': None,  # FLASH doesn't have a separate preprocessor
            'MESH': None,  # MESHPreProcessor (commented out)
        }
        
        # Define runner mapping
        self.runners = {
            'SUMMA': SummaRunner,
            'FUSE': FUSERunner,
            'GR': GRRunner,
            'NGEN': NgenRunner,
            'HYPE': HYPERunner,
            'FLASH': FLASH,
            'MESH': None,  # MESHRunner (commented out)
        }
        
        # Define postprocessor mapping
        self.postprocessors = {
            'SUMMA': SUMMAPostprocessor,
            'FUSE': FUSEPostprocessor,
            'NGEN': NgenPostprocessor,
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
            'NGEN': 'run_model',      
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

                # Select preprocessor for this model
                preprocessor_class = None
                if hasattr(self, "preprocessors"):
                    preprocessor_class = self.preprocessors.get(model)

                if preprocessor_class is None:
                    # Models that truly don't need preprocessing (e.g., FLASH)
                    if model in ['FLASH']:
                        self.logger.info(f"Model {model} doesn't require preprocessing")
                    else:
                        self.logger.warning(f"Unsupported model: {model}. No preprocessing performed.")
                    continue

                # Run model-specific preprocessing
                preprocessor = preprocessor_class(self.config, self.logger)
                preprocessor.run_preprocessing()

                # ----- Routing preprocessing hooks -----
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)

                if model == 'SUMMA':
                    # SUMMA -> mizuRoute (existing behavior, kept)
                    if needs_mizuroute:
                        self.logger.info("Initializing mizuRoute preprocessor for SUMMA")
                        # Hint to mizu preproc which upstream model is providing qsim
                        self.config['MIZU_FROM_MODEL'] = 'SUMMA'
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()

                elif model == 'FUSE':
                    # NEW: FUSE -> mizuRoute when requested by config
                    fuse_integration = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
                    if needs_mizuroute:
                        self.logger.info("Initializing mizuRoute preprocessor for FUSE")
                        # Tell the mizu preprocessor to emit a FUSE-style control file
                        self.config['MIZU_FROM_MODEL'] = 'FUSE'
                        mp = MizuRoutePreProcessor(self.config, self.logger)
                        mp.run_preprocessing()

            except Exception as e:
                self.logger.error(f"Error preprocessing model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        # Archive basin-averaged forcing data (left disabled; keep as-is)
        '''
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
        '''

        self.logger.info("Model-specific preprocessing completed")

    def _convert_fuse_distributed_to_mizuroute_format(self):
        """
        Convert FUSE spatial dimensions to mizuRoute format.
        MINIMAL changes only: latitude->gru, add gruId, squeeze longitude.
        Preserves ALL original data and time coordinates unchanged.
        """
        import xarray as xr
        import numpy as np
        import shutil
        import tempfile
        import os

        experiment_id = self.config.get('EXPERIMENT_ID')
        domain = self.domain_name
        
        fuse_out_dir = self.project_dir / "simulations" / experiment_id / "FUSE"
        
        # Find FUSE output file
        target_files = [
            fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc",
            fuse_out_dir / f"{domain}_{experiment_id}_runs_best.nc"
        ]
        
        target = None
        for file_path in target_files:
            if file_path.exists():
                target = file_path
                break
        
        if target is None:
            raise FileNotFoundError(f"FUSE output not found. Tried: {[str(f) for f in target_files]}")

        self.logger.info(f"Converting FUSE spatial dimensions: {target}")

        # Create backup
        backup_file = target.with_suffix('.backup.nc')
        if not backup_file.exists():
            shutil.copy2(target, backup_file)
            self.logger.info(f"Created backup: {backup_file}")

        # Load, modify, and immediately close the dataset
        with xr.open_dataset(target) as ds:
            self.logger.info(f"Original dimensions: {dict(ds.sizes)}")
            
            # Step 1: Remove singleton longitude dimension if it exists
            if 'longitude' in ds.sizes and ds.sizes['longitude'] == 1:
                ds = ds.squeeze('longitude', drop=True)
                self.logger.info("Squeezed longitude dimension")
            
            # Step 2: Rename latitude dimension to gru
            if 'latitude' in ds.sizes:
                ds = ds.rename({'latitude': 'gru'})
                self.logger.info("Renamed latitude -> gru")
                
                # Step 3: Create gruId variable from gru coordinates
                if 'gru' in ds.coords:
                    gru_values = ds.coords['gru'].values
                    try:
                        # Try to convert to integers
                        gru_ids = gru_values.astype('int32')
                    except (ValueError, TypeError):
                        # If conversion fails, use sequential IDs
                        gru_ids = np.arange(1, len(gru_values) + 1, dtype='int32')
                        self.logger.warning(f"Using sequential GRU IDs 1-{len(gru_values)}")
                    
                    ds['gruId'] = xr.DataArray(
                        gru_ids,
                        dims=('gru',),
                        attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                    )
                    
                    self.logger.info(f"Created gruId variable with {len(gru_ids)} GRUs")
                else:
                    raise ValueError("No gru coordinate found after renaming")
            else:
                raise ValueError("No latitude dimension found in FUSE output")
            
            self.logger.info(f"Final dimensions: {dict(ds.sizes)}")
            
            # Load all data into memory before closing
            ds = ds.load()
        
        # Now the original file is closed, we can write to a temp file and replace
        try:
            # Make sure target file is writable
            try:
                os.chmod(target, 0o664)
            except Exception as e:
                self.logger.warning(f"Could not change file permissions: {e}")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=target.parent) as tmp_file:
                temp_path = tmp_file.name
            
            # Write the modified dataset to temp file
            ds.to_netcdf(temp_path, format='NETCDF4')
            
            # Replace original with temp file
            shutil.move(temp_path, str(target))
            self.logger.info(f"Spatial conversion completed: {target}")
            
            # Ensure _runs_def.nc exists if we processed a different file
            def_file = fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc"
            if target != def_file and not def_file.exists():
                shutil.copy2(target, def_file)
                self.logger.info(f"Created runs_def file: {def_file}")
                
        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise

    def _fix_time_attributes_netcdf4(self, file_path, time_attrs):
        """Fallback method to fix time attributes using netCDF4 directly"""
        import netCDF4 as nc4
        
        try:
            self.logger.info(f"Using netCDF4 fallback to set time attributes: {time_attrs}")
            with nc4.Dataset(file_path, 'a') as ncfile:  # 'a' for append mode
                time_var = ncfile.variables['time']
                
                # Set attributes directly
                for attr_name, attr_value in time_attrs.items():
                    time_var.setncattr(attr_name, attr_value)
                
                self.logger.info(f"Successfully set time attributes using netCDF4")
                
        except Exception as e:
            self.logger.error(f"Failed to fix time attributes with netCDF4: {e}")
            raise

    def _needs_mizuroute_routing(self, domain_method: str, routing_delineation: str) -> bool:
        """
        Enhanced version that properly handles FUSE distributed modes.
        """
        # Check for FUSE distributed modes
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        if 'FUSE' in [m.strip() for m in models]:
            fuse_spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
            fuse_routing = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
            
            if fuse_routing == 'mizuRoute':
                if fuse_spatial_mode in ['semi_distributed', 'distributed']:
                    return True
                elif fuse_spatial_mode == 'lumped' and routing_delineation == 'river_network':
                    return True
        
        # Original SUMMA logic
        if domain_method not in ['point', 'lumped']:
            return True
        
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        
        return False

    def run_models(self):
        """Enhanced run_models with FUSE distributed support"""
        self.logger.info("Starting model runs")
        
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        
        for model in models:
            model = model.strip()
            try:
                self.logger.info(f"Running model: {model}")
                runner_class = self.runners.get(model)
                if runner_class is None:
                    self.logger.error(f"Unknown hydrological model: {model}")
                    continue

                runner = runner_class(self.config, self.logger)
                method_name = self.runner_methods.get(model)
                if method_name and hasattr(runner, method_name):
                    getattr(runner, method_name)()
                else:
                    self.logger.error(f"Runner method not found for model: {model}")
                    continue

                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)

                if needs_mizuroute:
                    if model == 'FUSE':
                        try:
                            fuse_spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
                            if fuse_spatial_mode in ['semi_distributed', 'distributed']:
                                self.logger.info("Converting distributed FUSE output to mizuRoute format (gru/gruId)")
                                #self._convert_fuse_distributed_to_mizuroute_format()
                        except Exception as e:
                            self.logger.error(f"FUSE→mizuRoute distributed conversion failed: {e}")
                            raise
                        mizuroute_runner = MizuRouteRunner(self.config, self.logger)
                        mizuroute_runner.run_mizuroute()
                        self.logger.info("FUSE routing completed via mizuRoute")
                    elif model == 'SUMMA':
                        if domain_method == 'lumped' and routing_delineation == 'river_network':
                            self.logger.info("Converting lumped SUMMA output for distributed routing")
                            self._convert_lumped_to_distributed_routing()
                        mizuroute_runner = MizuRouteRunner(self.config, self.logger)
                        mizuroute_runner.run_mizuroute()

            except Exception as e:
                self.logger.error(f"Error running model {model}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def _convert_fuse_lumped_to_distributed_routing(self):
        """Convert lumped FUSE output for distributed routing - simplified version
        
        Only handles HRU and GRU dimension/variable changes.
        Leaves runoff data unchanged.
        """
        self.logger.info("Converting FUSE output HRU/GRU dimensions for distributed routing")
        
        try:
            import xarray as xr
            import numpy as np
            
            experiment_id = self.config.get('EXPERIMENT_ID')
            
            # FUSE output file path
            fuse_output_dir = self.project_dir / "simulations" / experiment_id / "FUSE"
            fuse_output_file = fuse_output_dir / f"{self.domain_name}_{experiment_id}_runs_best.nc"
            
            if not fuse_output_file.exists():
                raise FileNotFoundError(f"FUSE output file not found: {fuse_output_file}")
            
            # Load FUSE output
            fuse_output = xr.open_dataset(fuse_output_file, decode_times=False)
            
            try:
                # Check current dimensions
                self.logger.info(f"Original dimensions: {dict(fuse_output.dims)}")
                
                # Convert FUSE distributed format to mizuRoute format
                if 'latitude' in fuse_output.dims and 'longitude' in fuse_output.dims:
                    # Squeeze out longitude dimension (should be 1)
                    if fuse_output.dims['longitude'] == 1:
                        fuse_output = fuse_output.squeeze('longitude', drop=True)
                        self.logger.info("Removed longitude dimension")
                    
                    # Rename latitude dimension to gru
                    fuse_output = fuse_output.rename({'latitude': 'gru'})
                    self.logger.info("Renamed latitude dimension to gru")
                    
                    # Create gruId variable from the gru coordinate
                    if 'gru' in fuse_output.coords:
                        gru_values = fuse_output.coords['gru'].values.astype(int)
                        fuse_output['gruId'] = xr.DataArray(
                            gru_values,
                            dims=('gru',),
                            attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                        )
                        self.logger.info(f"Created gruId variable with {len(gru_values)} values")
                
                elif 'gru' not in fuse_output.dims:
                    # Fallback: add single GRU if no existing spatial dimensions
                    hru_id = 1
                    fuse_output = fuse_output.expand_dims({'gru': [hru_id]})
                    fuse_output['gruId'] = xr.DataArray(
                        [hru_id], 
                        dims=('gru',),
                        attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                    )
                    self.logger.info(f"Added single GRU dimension (ID={hru_id})")
                
                self.logger.info(f"Final dimensions: {dict(fuse_output.dims)}")
                
                # Save the modified dataset (overwrites original)
                fuse_output.to_netcdf(fuse_output_file, format='NETCDF4')
                self.logger.info("FUSE HRU/GRU conversion completed successfully")
                
            finally:
                fuse_output.close()
                
        except Exception as e:
            self.logger.error(f"Error converting FUSE HRU/GRU dimensions: {str(e)}")
            raise

    def _convert_lumped_to_distributed_routing(self):
        """
        Convert lumped SUMMA output to distributed mizuRoute forcing.
        
        This method creates a single lumped GRU that mizuRoute can map to 
        the distributed routing network. The GRU ID matches the topology HRU ID.
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
            
            # Load mizuRoute topology to get HRU information
            topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
            
            if not topology_file.exists():
                raise FileNotFoundError(f"mizuRoute topology file not found: {topology_file}")
            
            with xr.open_dataset(topology_file) as mizuTopology:
                # Get the single HRU ID from topology (not segment IDs)
                hru_id = 1 #mizuTopology['hruId'].values[0]  # Should be 1
                seg_ids = mizuTopology['segId'].values
                n_segments = len(seg_ids)
                n_hrus = len(mizuTopology['hruId'].values)
            
            self.logger.info(f"Creating single lumped GRU (ID={hru_id}) for {n_segments} routing segments ({n_hrus} HRUs in topology)")
            
            # Check if we actually have a distributed routing network
            if n_segments <= 1:
                self.logger.warning(f"Only {n_segments} routing segment(s) found in topology. Distributed routing may not be beneficial.")
                self.logger.warning("Consider using ROUTING_DELINEATION: lumped instead")
            
            # Get the routing variable name from config
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
                
                # Create single GRU using HRU ID from topology (NOT segment IDs)
                mizuForcing['gru'] = xr.DataArray(
                    [hru_id],  # Single HRU, not multiple segment IDs
                    dims=('gru',),
                    attrs={'long_name': 'Index of GRU', 'units': '-'}
                )
                
                # GRU ID variable (use HRU ID from topology)
                mizuForcing['gruId'] = xr.DataArray(
                    [hru_id],  # Single HRU ID
                    dims=('gru',),
                    attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                )
                
                # Copy global attributes from SUMMA output
                mizuForcing.attrs.update(summa_output.attrs)
                
                # Find the best variable to broadcast
                source_var = None
                available_vars = list(summa_output.variables.keys())
                
                # Check for exact match first
                if routing_var in summa_output:
                    source_var = routing_var
                    self.logger.info(f"Using configured routing variable: {routing_var}")
                else:
                    # Try fallback variables in order of preference
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
                
                # Keep as single GRU: (time,) -> (time, 1)
                # Don't tile to multiple segments - mizuRoute will handle the routing
                tiled_data = lumped_runoff[:, np.newaxis]  # Shape: (time, 1)
                
                # Create the routing variable with the expected name
                mizuForcing[routing_var] = xr.DataArray(
                    tiled_data,
                    dims=('time', 'gru'),
                    attrs={
                        'long_name': 'Lumped runoff for distributed routing',
                        'units': 'm/s'
                    }
                )
                
                self.logger.info(f"Created single lumped GRU (ID={hru_id}) with {routing_var} data")
                
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