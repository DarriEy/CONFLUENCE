# dds_optimizer.py

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import shutil
from typing import Dict, Any

class DDSOptimizer:
    """
    Dynamically Dimensioned Search (DDS) Optimizer for CONFLUENCE.
    
    This class performs parameter optimization using the DDS algorithm,
    which is particularly effective for watershed model calibration.
    DDS automatically scales the search from global to local as the
    iteration count increases.
    
    Enhanced with parameter persistence:
    - Uses existing optimized parameters as starting point for new runs
    - Saves best parameters back to default model settings for future use
    
    References:
    Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search 
    algorithm for computationally efficient watershed model calibration.
    Water Resources Research, 43(1).
    """
            
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the DDS Optimizer.
        
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
        
        # Default settings directory (where we save/load optimized parameters)
        self.default_settings_dir = self.project_dir / "settings" / "SUMMA"
        self.default_trial_params_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
        
        # Parse time periods FIRST (needed for optimization environment setup)
        calib_period = self.config.get('CALIBRATION_PERIOD', '')
        eval_period = self.config.get('EVALUATION_PERIOD', '')
        self.calibration_period = self._parse_date_range(calib_period)
        self.evaluation_period = self._parse_date_range(eval_period)
        
        # Create DDS-specific directories
        self.optimization_dir = self.project_dir / "simulations" / "run_dds"
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        
        # Create output directory for results
        self.output_dir = self.project_dir / "optimisation" / f"dds_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimization-specific directories and settings
        self._setup_optimization_environment()
        
        # Get optimization settings
        self.max_iterations = self.config.get('NUMBER_OF_ITERATIONS', 100)
        self.r_value = self.config.get('DDS_R', 0.2)  # DDS perturbation parameter
        
        # Define parameter bounds (using optimization settings paths)
        self.local_param_info_path = self.optimization_settings_dir / 'localParamInfo.txt'
        self.basin_param_info_path = self.optimization_settings_dir / 'basinParamInfo.txt'
        
        # Get parameters to calibrate
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []
        
        # Get performance metric settings
        self.target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Initialize tracking variables
        self.current_best_params = None
        self.current_best_score = None
        self.iteration_history = []
        
        # Get attribute file path (use the copied one in optimization settings)
        self.attr_file_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
        
        # Logging
        self.logger.info(f"DDS Optimizer initialized with {len(self.local_params)} local parameters and {len(self.basin_params)} basin parameters")
        self.logger.info(f"Maximum iterations: {self.max_iterations}, r value: {self.r_value}")
        self.logger.info(f"Optimization simulations will run in: {self.summa_sim_dir}")
        
        # Log optimization period
        opt_period = self._get_optimization_period_string()
        self.logger.info(f"Optimization period: {opt_period}")

    def _load_existing_optimized_parameters(self):
        """
        Load existing optimized parameters from the default settings directory.
        
        Returns:
            Dict: Dictionary with parameter values if found, None otherwise
        """
        if not self.default_trial_params_path.exists():
            self.logger.info("No existing optimized parameters found - will start from scratch")
            return None
        
        try:
            self.logger.info(f"Loading existing optimized parameters from: {self.default_trial_params_path}")
            
            # Get all parameters to extract
            all_params = self.local_params + self.basin_params
            
            with xr.open_dataset(self.default_trial_params_path) as ds:
                # Check which parameters are available
                available_params = [param for param in all_params if param in ds.variables]
                
                if not available_params:
                    self.logger.warning("No calibration parameters found in existing trialParams file")
                    return None
                
                self.logger.info(f"Found {len(available_params)} out of {len(all_params)} parameters in existing file")
                
                # Extract parameter values
                param_values = {}
                
                for param in available_params:
                    var = ds[param]
                    values = var.values
                    
                    # Ensure values is a numpy array
                    param_values[param] = np.atleast_1d(values)
                    self.logger.debug(f"Loaded {param}: {len(param_values[param])} values")
                
                self.logger.info(f"Successfully loaded existing optimized parameters for {len(param_values)} parameters")
                return param_values
                
        except Exception as e:
            self.logger.error(f"Error loading existing optimized parameters: {str(e)}")
            self.logger.error("Will proceed with parameter extraction instead")
            return None
    
    def _save_best_parameters_to_default_settings(self, best_params):
        """
        Save the best parameters to the default model settings directory.
        
        This method:
        1. Backs up the existing trialParams.nc (if it exists) to trialParams_default.nc
        2. Saves the optimized parameters as the new trialParams.nc
        
        Args:
            best_params: Dictionary with the best parameter values
        """
        self.logger.info("Saving best parameters to default model settings")
        
        try:
            # Check if default settings directory exists
            if not self.default_settings_dir.exists():
                self.logger.error(f"Default settings directory not found: {self.default_settings_dir}")
                return False
            
            # Step 1: Backup existing trialParams.nc if it exists
            if self.default_trial_params_path.exists():
                backup_path = self.default_settings_dir / "trialParams_default.nc"
                
                # If backup already exists, create a timestamped backup
                if backup_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    timestamped_backup = self.default_settings_dir / f"trialParams_backup_{timestamp}.nc"
                    shutil.copy2(backup_path, timestamped_backup)
                    self.logger.info(f"Created timestamped backup: {timestamped_backup}")
                
                # Move existing trialParams to backup
                shutil.copy2(self.default_trial_params_path, backup_path)
                self.logger.info(f"Backed up existing trialParams.nc to: {backup_path}")
            
            # Step 2: Generate new trialParams.nc with best parameters in default settings
            new_trial_params_path = self._generate_trial_params_file_in_directory(
                best_params, 
                self.default_settings_dir,
                "trialParams.nc"
            )
            
            if new_trial_params_path:
                self.logger.info(f"✅ Successfully saved optimized parameters to: {new_trial_params_path}")
                
                # Also save a copy with experiment ID for tracking
                experiment_copy = self.default_settings_dir / f"trialParams_optimized_{self.experiment_id}.nc"
                shutil.copy2(new_trial_params_path, experiment_copy)
                self.logger.info(f"Created experiment-specific copy: {experiment_copy}")
                
                return True
            else:
                self.logger.error("Failed to generate new trialParams.nc file")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving best parameters to default settings: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _generate_trial_params_file_in_directory(self, params, target_dir, filename):
        """
        Generate a trialParams.nc file with the given parameters in the specified directory.
        
        Args:
            params: Dictionary with parameter values
            target_dir: Directory where to save the file
            filename: Name of the file to create
            
        Returns:
            Path: Path to the generated trial parameters file
        """
        self.logger.debug(f"Generating trial parameters file: {target_dir / filename}")
        
        # Get attribute file path for reading HRU information
        attr_file_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
        
        if not attr_file_path.exists():
            self.logger.error(f"Attribute file not found: {attr_file_path}")
            return None
        
        try:
            # Read HRU and GRU information from the attributes file
            with xr.open_dataset(attr_file_path) as ds:
                # Check for GRU dimension
                has_gru_dim = 'gru' in ds.dims
                hru_ids = ds['hruId'].values
                
                if has_gru_dim:
                    gru_ids = ds['gruId'].values
                    self.logger.debug(f"Attribute file has {len(gru_ids)} GRUs and {len(hru_ids)} HRUs")
                else:
                    gru_ids = np.array([1])  # Default if no GRU dimension
                    self.logger.debug(f"Attribute file has no GRU dimension, using default GRU ID=1")
                
                # Create the trial parameters dataset
                trial_params_path = target_dir / filename
                
                # Routing parameters should be at GRU level
                routing_params = ['routingGammaShape', 'routingGammaScale']
                
                with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                    # Add dimensions
                    output_ds.createDimension('hru', len(hru_ids))
                    
                    # Add GRU dimension if needed for routing parameters
                    if any(param in params for param in routing_params):
                        output_ds.createDimension('gru', len(gru_ids))
                        
                        # Add GRU ID variable
                        gru_id_var = output_ds.createVariable('gruId', 'i4', ('gru',))
                        gru_id_var[:] = gru_ids
                        gru_id_var.long_name = 'Group Response Unit ID (GRU)'
                        gru_id_var.units = '-'
                        
                        # Add HRU2GRU mapping if available
                        if 'hru2gruId' in ds.variables:
                            hru2gru_var = output_ds.createVariable('hru2gruId', 'i4', ('hru',))
                            hru2gru_var[:] = ds['hru2gruId'].values
                            hru2gru_var.long_name = 'Index of GRU for each HRU'
                            hru2gru_var.units = '-'
                    
                    # Add HRU ID variable
                    hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                    hru_id_var[:] = hru_ids
                    hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                    hru_id_var.units = '-'
                    
                    # Add parameter variables
                    for param_name, param_values in params.items():
                        # Ensure param_values is a numpy array
                        param_values_array = np.asarray(param_values)
                        
                        # Check if this is a routing parameter (should be at GRU level)
                        if param_name in routing_params:
                            # These parameters should be at GRU level
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            
                            # Most likely, we have 1 value per GRU for these parameters
                            if len(param_values_array) != len(gru_ids):
                                if len(param_values_array) > len(gru_ids):
                                    # Take only what we need
                                    param_values_array = param_values_array[:len(gru_ids)]
                                else:
                                    # Expand by repetition if needed
                                    repeats = int(np.ceil(len(gru_ids) / len(param_values_array)))
                                    param_values_array = np.tile(param_values_array, repeats)[:len(gru_ids)]
                            
                            # Assign to GRU dimension
                            param_var[:] = param_values_array
                        else:
                            # Regular parameter at HRU level
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            
                            # Handle array shape issues
                            if param_values_array.ndim > 1:
                                original_shape = param_values_array.shape
                                param_values_array = param_values_array.flatten()
                                self.logger.debug(f"Flattened {param_name} from shape {original_shape} to 1D")
                            
                            # Handle array length issues
                            if len(param_values_array) != len(hru_ids):
                                if len(param_values_array) > len(hru_ids):
                                    # Truncate
                                    param_values_array = param_values_array[:len(hru_ids)]
                                else:
                                    # Expand by repetition
                                    repeats = int(np.ceil(len(hru_ids) / len(param_values_array)))
                                    param_values_array = np.tile(param_values_array, repeats)[:len(hru_ids)]
                            
                            # Assign to HRU dimension
                            param_var[:] = param_values_array
                        
                        # Add attributes
                        param_var.long_name = f"Trial value for {param_name}"
                        param_var.units = "N/A"
                    
                    # Add global attributes
                    output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE DDS Optimizer"
                    output_ds.history = f"Created on {datetime.now().isoformat()}"
                    output_ds.confluence_experiment_id = f"DDS_optimization_{self.experiment_id}"
                
                self.logger.debug(f"Trial parameters file generated: {trial_params_path}")
                return trial_params_path
                
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _parse_date_range(self, date_range_str):
        """Parse date range string from config into start and end dates."""
        if not date_range_str:
            return None, None
            
        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except:
            self.logger.warning(f"Could not parse date range: {date_range_str}")
        
        return None, None

    def _update_file_manager_for_optimization(self, is_final_run=False):
        """
        Update the fileManager.txt in the optimization settings to point to 
        optimization-specific directories and set appropriate time periods.
        
        Args:
            is_final_run (bool): If True, runs full experiment period for final evaluation.
                            If False, runs only spinup + calibration period for optimization.
        """
        file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        
        if not file_manager_path.exists():
            self.logger.warning("fileManager.txt not found in optimization settings")
            return
        
        # Parse time periods from config
        spinup_start, spinup_end = self._parse_date_range(self.config.get('SPINUP_PERIOD', ''))
        calib_start, calib_end = self.calibration_period
        experiment_start = self.config.get('EXPERIMENT_TIME_START', '')
        experiment_end = self.config.get('EXPERIMENT_TIME_END', '')
        
        # Determine simulation period based on run type
        if is_final_run:
            # Final run: use full experiment period
            sim_start = experiment_start
            sim_end = experiment_end
            run_prefix = f"run_dds_final_{self.experiment_id}"
            self.logger.info("Setting up final run with full experiment period")
        else:
            # Optimization runs: use spinup start to calibration end
            if spinup_start and calib_end:
                sim_start = spinup_start.strftime('%Y-%m-%d') + ' 01:00'
                sim_end = calib_end.strftime('%Y-%m-%d') + ' 23:00'
            elif calib_start and calib_end:
                # Fallback: just use calibration period if no spinup defined
                sim_start = calib_start.strftime('%Y-%m-%d') + ' 01:00'
                sim_end = calib_end.strftime('%Y-%m-%d') + ' 23:00'
            else:
                # Ultimate fallback: use experiment period
                sim_start = experiment_start
                sim_end = experiment_end
                self.logger.warning("Could not parse spinup/calibration periods, using full experiment period")
            
            run_prefix = f"run_dds_opt_{self.experiment_id}"
            self.logger.info(f"Setting up optimization run from {sim_start} to {sim_end}")
        
        # Read the original file manager
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        # Update paths and time periods in file manager
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        '{run_prefix}'\n")
            elif 'outputPath' in line:
                # Point to optimization SUMMA simulation directory
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                # Point to optimization settings directory  
                settings_path = str(self.optimization_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            elif 'forcingPath' in line:
                # Keep original forcing path or update if needed
                updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Write updated file manager
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
        
        period_type = "final evaluation" if is_final_run else "optimization"
        self.logger.info(f"Updated fileManager.txt for {period_type} with period {sim_start} to {sim_end}")

    def _setup_optimization_environment(self):
        """
        Set up the optimization-specific directory structure and copy settings files.
        
        This creates a dedicated environment for optimization runs, separate from 
        the baseline simulations, and copies all necessary settings files.
        """
        self.logger.info("Setting up optimization environment")
        
        # Create all necessary directories
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        self.summa_sim_dir.mkdir(parents=True, exist_ok=True)
        self.mizuroute_sim_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        (self.summa_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.mizuroute_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Source settings directory
        source_settings_dir = self.project_dir / "settings" / "SUMMA"
        
        if not source_settings_dir.exists():
            raise FileNotFoundError(f"Source SUMMA settings directory not found: {source_settings_dir}")
        
        # Copy all SUMMA settings files to optimization directory
        settings_files = [
            'fileManager.txt',
            'modelDecisions.txt', 
            'outputControl.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL', 
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL',
            'attributes.nc',
            'coldState.nc',
            'trialParams.nc',
            'forcingFileList.txt'
        ]
        
        copied_files = []
        for file_name in settings_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name
            
            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(file_name)
                    self.logger.debug(f"Copied {file_name} to optimization settings")
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")
            else:
                self.logger.debug(f"Settings file not found (optional): {file_name}")
        
        self.logger.info(f"Copied {len(copied_files)} settings files to optimization directory")
        
        # Update fileManager.txt for optimization runs (not final run)
        self._update_file_manager_for_optimization(is_final_run=False)
        
        # Copy mizuRoute settings if they exist
        self._copy_mizuroute_settings()

    def _run_final_simulation(self, best_params):
        """
        Run a final simulation with the best parameters using the full experiment period.
        
        Args:
            best_params: Dictionary with best parameter values
            
        Returns:
            Dict: Dictionary with final simulation results
        """
        self.logger.info("Running final simulation with best parameters over full experiment period")
        
        # Update file manager for final run (full experiment period)
        self._update_file_manager_for_optimization(is_final_run=True)
        
        # Generate trial parameters file with best parameters
        trial_params_path = self._generate_trial_params_file(best_params)
        if not trial_params_path:
            self.logger.warning("Could not generate trial parameters file for final run")
            return {'summa_success': False, 'mizuroute_success': False, 'metrics': None}
        
        # Create a copy of the best parameters file
        final_params_path = self.output_dir / "best_params.nc"
        try:
            shutil.copy2(trial_params_path, final_params_path)
            self.logger.info(f"Copied best parameters to: {final_params_path}")
        except Exception as e:
            self.logger.warning(f"Could not copy best parameters file: {str(e)}")
        
        # Save best parameters to CSV for easier viewing
        param_df = pd.DataFrame()
        for param_name, values in best_params.items():
            if len(values) == 1:
                # Single value (probably basin parameter)
                param_df[param_name] = [values[0]]
            else:
                # Multiple values (probably local parameter)
                # Just save the mean, min, and max
                param_df[f"{param_name}_mean"] = [np.mean(values)]
                param_df[f"{param_name}_min"] = [np.min(values)]
                param_df[f"{param_name}_max"] = [np.max(values)]
        
        param_csv_path = self.output_dir / "best_params.csv"
        param_df.to_csv(param_csv_path, index=False)
        self.logger.info(f"Saved best parameters to CSV: {param_csv_path}")
        
        # Run SUMMA with the best parameters (full period)
        summa_success = self._run_summa_simulation()
        
        # Run mizuRoute if needed
        mizuroute_success = True
        if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
            mizuroute_success = self._run_mizuroute_simulation()
        
        # Calculate final performance metrics for both calibration and evaluation periods
        final_metrics = None
        if summa_success and mizuroute_success:
            final_metrics = self._calculate_performance_metrics()
            
            if final_metrics:
                # Save metrics to CSV
                metrics_df = pd.DataFrame([final_metrics])
                metrics_csv_path = self.output_dir / "final_metrics.csv"
                metrics_df.to_csv(metrics_csv_path, index=False)
                self.logger.info(f"Saved final metrics to CSV: {metrics_csv_path}")
                
                # Log the performance for both periods
                calib_metrics = {k: v for k, v in final_metrics.items() if k.startswith('Calib_')}
                eval_metrics = {k: v for k, v in final_metrics.items() if k.startswith('Eval_')}
                
                if calib_metrics:
                    self.logger.info("Final Calibration Period Performance:")
                    for metric, value in calib_metrics.items():
                        metric_name = metric.replace('Calib_', '')
                        self.logger.info(f"  {metric_name}: {value:.4f}")
                
                if eval_metrics:
                    self.logger.info("Final Evaluation Period Performance:")
                    for metric, value in eval_metrics.items():
                        metric_name = metric.replace('Eval_', '')
                        self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Reset file manager back to optimization mode for any future runs
        self._update_file_manager_for_optimization(is_final_run=False)
        
        # Return results
        return {
            'summa_success': summa_success,
            'mizuroute_success': mizuroute_success,
            'metrics': final_metrics
        }

    def _get_optimization_period_string(self):
        """
        Get a string representation of the optimization period for logging.
        
        Returns:
            str: Description of the optimization period
        """
        spinup_start, spinup_end = self._parse_date_range(self.config.get('SPINUP_PERIOD', ''))
        calib_start, calib_end = self.calibration_period
        
        if spinup_start and calib_end:
            return f"{spinup_start.strftime('%Y-%m-%d')} to {calib_end.strftime('%Y-%m-%d')} (spinup + calibration)"
        elif calib_start and calib_end:
            return f"{calib_start.strftime('%Y-%m-%d')} to {calib_end.strftime('%Y-%m-%d')} (calibration only)"
        else:
            return "full experiment period (fallback)"

    def _copy_mizuroute_settings(self):
        """
        Copy mizuRoute settings to optimization directory if they exist.
        """
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"
        
        if not source_mizu_dir.exists():
            self.logger.debug("mizuRoute settings directory not found - skipping")
            return
        
        # Create mizuRoute settings directory
        dest_mizu_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mizuRoute settings files
        mizu_files = [
            'mizuroute.control',
            'param.nml.default', 
            'topology.nc',
            'routing_remap.nc'
        ]
        
        copied_files = []
        for file_name in mizu_files:
            source_path = source_mizu_dir / file_name
            dest_path = dest_mizu_dir / file_name
            
            if source_path.exists():
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(file_name)
                except Exception as e:
                    self.logger.warning(f"Could not copy mizuRoute file {file_name}: {str(e)}")
        
        if copied_files:
            self.logger.info(f"Copied {len(copied_files)} mizuRoute settings files")

    def run_dds_optimization(self):
        """
        Enhanced DDS optimization with parameter persistence.
        
        Enhanced to:
        1. Check for existing optimized parameters and use them as starting point
        2. Save best parameters back to default settings when complete
        
        Returns:
            Dict: Dictionary with optimization results
        """
        self.logger.info("Starting DDS optimization with parameter persistence")
        
        # Step 1: Try to load existing optimized parameters
        existing_params = self._load_existing_optimized_parameters()
        
        if existing_params:
            self.logger.info("🔄 Starting optimization from existing optimized parameters")
            initial_params = existing_params
        else:
            # Step 2: Get initial parameter values from a preliminary SUMMA run
            self.logger.info("🆕 No existing optimized parameters found, extracting from model run")
            initial_params = self.run_parameter_extraction()
        
        # Step 3: Parse parameter bounds
        param_bounds = self._parse_parameter_bounds()
        
        # Step 4: Initialize optimization variables
        self._initialize_optimization(initial_params, param_bounds)
        
        # Step 5: Run the DDS algorithm
        best_params, best_score, history = self._run_dds_algorithm()
        
        # Step 6: Create visualization of optimization progress
        self._create_optimization_plots(history)
        
        # Step 7: Run a final simulation with the best parameters
        final_result = self._run_final_simulation(best_params)
        
        # Step 8: Save best parameters to default model settings
        if best_params:
            save_success = self._save_best_parameters_to_default_settings(best_params)
            if save_success:
                self.logger.info("✅ Best parameters saved to default model settings")
                self.logger.info("🔄 Future optimization runs will start from these optimized parameters")
            else:
                self.logger.warning("⚠️ Failed to save parameters to default settings")
        
        # Step 9: Save parameter evaluation history to files
        self._save_parameter_evaluation_history()
        
        # Return results
        results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'history': history,
            'final_result': final_result,
            'output_dir': str(self.output_dir),
            'used_existing_params': existing_params is not None,
            'saved_to_default': save_success if best_params else False
        }
        
        return results
    
    def _save_parameter_evaluation_history(self):
        """
        Save all parameter evaluations to CSV files for analysis.
        
        Creates detailed files with:
        1. All parameter sets evaluated and their scores
        2. Best parameters over time
        3. Iteration statistics
        """
        self.logger.info("Saving parameter evaluation history to files")
        
        try:
            # Create detailed evaluation records from iteration history
            all_evaluations = []
            iteration_summaries = []
            
            for iter_data in self.iteration_history:
                iteration = iter_data['iteration']
                score = iter_data.get('score')
                parameters = iter_data.get('parameters', {})
                is_best = iter_data.get('is_best', False)
                duration = iter_data.get('duration_seconds', 0)
                
                # Create evaluation record
                eval_record = {
                    'iteration': iteration,
                    'score': score,
                    'is_valid': score is not None,
                    'is_new_best': is_best,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add parameter values
                for param_name, param_values in parameters.items():
                    param_values_array = np.asarray(param_values)
                    if len(param_values_array) == 1:
                        eval_record[param_name] = param_values_array[0]
                    else:
                        # For multi-value parameters, store statistics
                        eval_record[f"{param_name}_mean"] = np.mean(param_values_array)
                        eval_record[f"{param_name}_min"] = np.min(param_values_array)
                        eval_record[f"{param_name}_max"] = np.max(param_values_array)
                        eval_record[f"{param_name}_std"] = np.std(param_values_array)
                
                all_evaluations.append(eval_record)
                
                # Create iteration summary
                iter_summary = {
                    'iteration': iteration,
                    'score': score,
                    'is_new_best': is_best,
                    'best_score_so_far': iter_data.get('best_score'),
                    'duration_seconds': duration,
                    'cumulative_time': sum(h.get('duration_seconds', 0) for h in self.iteration_history[:iteration+1])
                }
                
                # Add best parameters for this iteration if it's a new best
                if is_best and parameters:
                    for param_name, param_values in parameters.items():
                        param_values_array = np.asarray(param_values)
                        if len(param_values_array) == 1:
                            iter_summary[f"best_{param_name}"] = param_values_array[0]
                        else:
                            iter_summary[f"best_{param_name}_mean"] = np.mean(param_values_array)
                
                iteration_summaries.append(iter_summary)
            
            # Save all evaluations to CSV
            if all_evaluations:
                eval_df = pd.DataFrame(all_evaluations)
                eval_csv_path = self.output_dir / "all_parameter_evaluations.csv"
                eval_df.to_csv(eval_csv_path, index=False)
                self.logger.info(f"💾 Saved {len(all_evaluations)} parameter evaluations to: {eval_csv_path}")
                
                # Also save only valid evaluations for easier analysis
                valid_eval_df = eval_df[eval_df['is_valid'] == True]
                if len(valid_eval_df) > 0:
                    valid_csv_path = self.output_dir / "valid_parameter_evaluations.csv"
                    valid_eval_df.to_csv(valid_csv_path, index=False)
                    self.logger.info(f"✅ Saved {len(valid_eval_df)} valid evaluations to: {valid_csv_path}")
            
            # Save iteration summaries to CSV
            if iteration_summaries:
                summary_df = pd.DataFrame(iteration_summaries)
                summary_csv_path = self.output_dir / "iteration_summaries.csv"
                summary_df.to_csv(summary_csv_path, index=False)
                self.logger.info(f"📊 Saved {len(iteration_summaries)} iteration summaries to: {summary_csv_path}")
            
            # Save optimization metadata
            metadata = {
                'algorithm': 'Dynamically Dimensioned Search (DDS)',
                'total_iterations': self.max_iterations,
                'r_value': self.r_value,
                'target_metric': self.target_metric,
                'parameters_calibrated': self.local_params + self.basin_params,
                'total_evaluations': len(all_evaluations),
                'experiment_id': self.experiment_id,
                'domain_name': self.domain_name,
                'optimization_period': self._get_optimization_period_string(),
                'completed_at': datetime.now().isoformat()
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_csv_path = self.output_dir / "optimization_metadata.csv"
            metadata_df.to_csv(metadata_csv_path, index=False)
            self.logger.info(f"ℹ️ Saved optimization metadata to: {metadata_csv_path}")
            
            # Create comprehensive summary statistics
            if all_evaluations:
                eval_df = pd.DataFrame(all_evaluations)
                valid_scores = eval_df[eval_df['is_valid'] == True]['score']
                
                if len(valid_scores) > 0:
                    # Calculate improvement statistics
                    improvements = eval_df[eval_df['is_new_best'] == True]
                    total_improvements = len(improvements)
                    
                    summary_stats = {
                        'total_evaluations': len(all_evaluations),
                        'valid_evaluations': len(valid_scores),
                        'failed_evaluations': len(all_evaluations) - len(valid_scores),
                        'success_rate_percent': len(valid_scores) / len(all_evaluations) * 100,
                        'total_improvements': total_improvements,
                        'improvement_rate_percent': total_improvements / len(all_evaluations) * 100,
                        'best_score': valid_scores.max(),
                        'worst_score': valid_scores.min(),
                        'mean_score': valid_scores.mean(),
                        'median_score': valid_scores.median(),
                        'std_score': valid_scores.std(),
                        'score_range': valid_scores.max() - valid_scores.min(),
                        'total_runtime_seconds': eval_df['duration_seconds'].sum(),
                        'avg_evaluation_time_seconds': eval_df['duration_seconds'].mean()
                    }
                    
                    # Add parameter exploration statistics
                    param_cols = [col for col in eval_df.columns if col not in ['iteration', 'score', 'is_valid', 'is_new_best', 'duration_seconds', 'timestamp']]
                    for param_col in param_cols:
                        if eval_df[param_col].dtype in ['float64', 'int64'] and not eval_df[param_col].isna().all():
                            param_data = eval_df[param_col].dropna()
                            summary_stats[f"{param_col}_explored_min"] = param_data.min()
                            summary_stats[f"{param_col}_explored_max"] = param_data.max()
                            summary_stats[f"{param_col}_explored_range"] = param_data.max() - param_data.min()
                            summary_stats[f"{param_col}_explored_mean"] = param_data.mean()
                            summary_stats[f"{param_col}_explored_std"] = param_data.std()
                    
                    # Save convergence statistics
                    if len(improvements) > 1:
                        improvement_iterations = improvements['iteration'].values
                        improvement_gaps = np.diff(improvement_iterations)
                        summary_stats['avg_iterations_between_improvements'] = np.mean(improvement_gaps)
                        summary_stats['max_iterations_between_improvements'] = np.max(improvement_gaps)
                        summary_stats['last_improvement_iteration'] = improvement_iterations[-1]
                        summary_stats['stagnation_iterations'] = len(all_evaluations) - improvement_iterations[-1]
                    
                    stats_df = pd.DataFrame([summary_stats])
                    stats_csv_path = self.output_dir / "optimization_statistics.csv"
                    stats_df.to_csv(stats_csv_path, index=False)
                    self.logger.info(f"📈 Saved optimization statistics to: {stats_csv_path}")
                    
                    # Log key statistics
                    self.logger.info(f"📊 Optimization Summary:")
                    self.logger.info(f"   • Success rate: {summary_stats['success_rate_percent']:.1f}% ({summary_stats['valid_evaluations']}/{summary_stats['total_evaluations']})")
                    self.logger.info(f"   • Improvements: {summary_stats['total_improvements']} ({summary_stats['improvement_rate_percent']:.1f}%)")
                    self.logger.info(f"   • Score range: {summary_stats['worst_score']:.6f} to {summary_stats['best_score']:.6f}")
                    self.logger.info(f"   • Total runtime: {summary_stats['total_runtime_seconds']:.1f} seconds")
            
            # Create a parameter bounds vs explored ranges comparison
            if hasattr(self, 'param_bounds') and all_evaluations:
                bounds_comparison = []
                eval_df = pd.DataFrame(all_evaluations)
                
                for param_name, bounds in self.param_bounds.items():
                    # Find parameter columns (could be direct or mean values)
                    param_col = param_name if param_name in eval_df.columns else f"{param_name}_mean"
                    
                    if param_col in eval_df.columns:
                        explored_data = eval_df[param_col].dropna()
                        if len(explored_data) > 0:
                            bounds_comparison.append({
                                'parameter': param_name,
                                'bound_min': bounds['min'],
                                'bound_max': bounds['max'],
                                'bound_range': bounds['max'] - bounds['min'],
                                'explored_min': explored_data.min(),
                                'explored_max': explored_data.max(),
                                'explored_range': explored_data.max() - explored_data.min(),
                                'exploration_coverage_percent': (explored_data.max() - explored_data.min()) / (bounds['max'] - bounds['min']) * 100,
                                'values_at_lower_bound': (explored_data <= bounds['min'] + 0.01 * (bounds['max'] - bounds['min'])).sum(),
                                'values_at_upper_bound': (explored_data >= bounds['max'] - 0.01 * (bounds['max'] - bounds['min'])).sum()
                            })
                
                if bounds_comparison:
                    bounds_df = pd.DataFrame(bounds_comparison)
                    bounds_csv_path = self.output_dir / "parameter_exploration_coverage.csv"
                    bounds_df.to_csv(bounds_csv_path, index=False)
                    self.logger.info(f"🎯 Saved parameter exploration coverage to: {bounds_csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving parameter evaluation history: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    # [Continue with all the existing methods from the original DDS optimizer...]
    # Including: _initialize_optimization, _run_dds_algorithm, _log_progress_summary, etc.
    
    def _initialize_optimization(self, initial_params, param_bounds):
        """
        Initialize optimization variables.
        
        Args:
            initial_params: Dictionary with initial parameter values
            param_bounds: Dictionary with parameter bounds
        """
        self.logger.info("Initializing optimization variables")
        
        # Store parameter bounds
        self.param_bounds = param_bounds
        
        # Create normalized parameter dictionary
        self.normalized_params = {}
        
        # Get HRU and GRU information from attributes file
        with xr.open_dataset(self.attr_file_path) as ds:
            num_hru = ds.sizes['hru']
            hru_ids = ds['hruId'].values
            
            if 'gru' in ds.dims:
                num_gru = ds.sizes['gru']
                gru_ids = ds['gruId'].values
                hru2gru_ids = ds['hru2gruId'].values
                
                # Create mapping from GRU ID to HRU IDs
                self.gru_to_hrus = {}
                for hru_idx in range(num_hru):
                    gru_id = hru2gru_ids[hru_idx]
                    if gru_id not in self.gru_to_hrus:
                        self.gru_to_hrus[gru_id] = []
                    self.gru_to_hrus[gru_id].append(hru_ids[hru_idx])
            else:
                num_gru = 1
                gru_ids = [1]
                self.gru_to_hrus = {1: hru_ids}
        
        # Initialize parameter arrays
        for param_name, bounds in param_bounds.items():
            min_val = bounds['min']
            max_val = bounds['max']
            
            # Check if we have initial values
            if initial_params and param_name in initial_params:
                initial_values = initial_params[param_name]
                
                # Normalize values to [0, 1] range
                normalized_values = (initial_values - min_val) / (max_val - min_val)
                
                # Clip to valid range
                normalized_values = np.clip(normalized_values, 0, 1)
            else:
                # Use random initial values in [0, 1] range
                if param_name in self.local_params:
                    normalized_values = np.random.random(num_hru)
                else:  # Basin parameter
                    normalized_values = np.random.random(num_gru)
            
            self.normalized_params[param_name] = normalized_values
        
        # Initialize tracking variables
        self.current_best_params = self._denormalize_params(self.normalized_params)
        self.current_best_score = float('-inf')  # We'll maximize the objective
        self.iteration_history = []
        
        # Run initial simulation to get baseline score
        self.logger.info("Running initial simulation")
        initial_score = self._evaluate_parameters(self.current_best_params)
        
        if initial_score is not None:
            self.current_best_score = initial_score
            
            # Record initial point
            self.iteration_history.append({
                'iteration': 0,
                'score': initial_score,
                'parameters': self.current_best_params.copy()
            })
            
            self.logger.info(f"Initial score: {initial_score:.4f}")
        else:
            self.logger.warning("Could not evaluate initial parameter set")
        
    def _run_dds_algorithm(self):
        """
        Run the DDS algorithm with enhanced logging for each iteration.
        
        Returns:
            Tuple: (best_params, best_score, history)
        """
        self.logger.info("Running DDS algorithm")
        self.logger.info("=" * 60)
        self.logger.info(f"Target metric: {self.target_metric} (higher is better)")
        self.logger.info(f"Total iterations: {self.max_iterations}")
        self.logger.info(f"DDS perturbation parameter (r): {self.r_value}")
        self.logger.info("=" * 60)
        
        # Track some statistics for logging
        improvements = 0
        no_improvement_streak = 0
        last_improvement_iteration = 0
        
        # Main DDS loop
        for i in range(1, self.max_iterations + 1):
            iteration_start_time = datetime.now()
            
            # Step 1: Create candidate solution by perturbing a subset of decision variables
            candidate_params = self._generate_candidate(i)
            
            # Step 2: Evaluate candidate solution
            candidate_score = self._evaluate_parameters(candidate_params)
            
            # Step 3: Update current best solution if candidate is better
            is_new_best = False
            if candidate_score is not None and candidate_score > self.current_best_score:
                old_best = self.current_best_score
                self.current_best_score = candidate_score
                self.current_best_params = candidate_params.copy()
                is_new_best = True
                improvements += 1
                no_improvement_streak = 0
                last_improvement_iteration = i
                
                # Calculate improvement
                if old_best > float('-inf'):
                    improvement = candidate_score - old_best
                    improvement_pct = (improvement / abs(old_best)) * 100 if old_best != 0 else float('inf')
                else:
                    improvement = candidate_score
                    improvement_pct = float('inf')
            else:
                no_improvement_streak += 1
            
            # Calculate iteration duration
            iteration_duration = datetime.now() - iteration_start_time
            
            # Create detailed log message
            if candidate_score is not None:
                score_str = f"{candidate_score:.6f}"
            else:
                score_str = "FAILED"
                candidate_score = float('-inf')  # For comparison purposes
            
            # Base log message
            log_msg = f"Iter {i:3d}/{self.max_iterations}: {self.target_metric}={score_str}"
            
            # Add new best indicator and improvement info
            if is_new_best:
                if old_best > float('-inf'):
                    improvement = candidate_score - old_best
                    improvement_pct = (improvement / abs(old_best)) * 100 if old_best != 0 else float('inf')
                    log_msg += f" ⭐ New best! (+{improvement:.6f}, +{improvement_pct:.2f}%)"
                else:
                    log_msg += f" ⭐ New best! (first valid score)"
            else:
                if self.current_best_score > float('-inf'):
                    deficit = self.current_best_score - candidate_score if candidate_score != float('-inf') else float('inf')
                    log_msg += f" (best: {self.current_best_score:.6f}, deficit: {deficit:.6f})"
                else:
                    log_msg += f" (no best yet)"
            
            # Add timing info
            log_msg += f" [{iteration_duration.total_seconds():.1f}s]"
            
            # Log the iteration
            self.logger.info(log_msg)
            
            # Record history
            self.iteration_history.append({
                'iteration': i,
                'score': candidate_score if candidate_score != float('-inf') else None,
                'parameters': candidate_params.copy(),
                'is_best': is_new_best,
                'best_score': self.current_best_score if self.current_best_score != float('-inf') else None,
                'duration_seconds': iteration_duration.total_seconds()
            })
            
            # Progress logging every 10 iterations or on improvements
            if i % 10 == 0 or is_new_best or i == self.max_iterations:
                self._log_progress_summary(i, improvements, no_improvement_streak, last_improvement_iteration)
        
        # Final summary
        self.logger.info("=" * 60)
        self.logger.info("DDS OPTIMIZATION COMPLETED")
        self.logger.info("=" * 60)
        if self.current_best_score > float('-inf'):
            self.logger.info(f"🏆 Best {self.target_metric}: {self.current_best_score:.6f}")
            self.logger.info(f"📊 Total improvements: {improvements}/{self.max_iterations} ({improvements/self.max_iterations*100:.1f}%)")
            self.logger.info(f"🎯 Last improvement: iteration {last_improvement_iteration}")
            
            # Calculate search efficiency
            if improvements > 0:
                avg_iterations_per_improvement = self.max_iterations / improvements
                self.logger.info(f"⚡ Search efficiency: {avg_iterations_per_improvement:.1f} iterations per improvement")
        else:
            self.logger.warning("❌ No valid solutions found during optimization!")
            self.logger.warning("Check model configuration, parameter bounds, and forcing data.")
        
        self.logger.info("=" * 60)
        
        return self.current_best_params, self.current_best_score, self.iteration_history

    def _log_progress_summary(self, current_iter, improvements, no_improvement_streak, last_improvement_iteration):
        """
        Log a progress summary with statistics.
        
        Args:
            current_iter: Current iteration number
            improvements: Total number of improvements so far
            no_improvement_streak: Current streak of no improvements
            last_improvement_iteration: Iteration number of last improvement
        """
        progress_pct = (current_iter / self.max_iterations) * 100
        
        summary_msg = f"📈 Progress: {progress_pct:.1f}% ({current_iter}/{self.max_iterations})"
        
        if self.current_best_score > float('-inf'):
            summary_msg += f" | Best: {self.current_best_score:.6f}"
            summary_msg += f" | Improvements: {improvements}"
            
            if no_improvement_streak > 0:
                summary_msg += f" | No improvement: {no_improvement_streak} iters"
            
            if last_improvement_iteration > 0:
                summary_msg += f" | Last improvement: iter {last_improvement_iteration}"
        else:
            summary_msg += " | No valid solutions yet"
        
        self.logger.info(summary_msg)

    def _evaluate_parameters(self, params):
        """
        Evaluate a parameter set by running SUMMA and calculating performance metrics.
        Enhanced with better error logging.
        
        Args:
            params: Dictionary with parameter values
            
        Returns:
            float: Performance metric value, or None if evaluation failed
        """
        eval_start_time = datetime.now()
        
        try:
            self.logger.debug("Evaluating parameter set")
            
            # Generate trial parameters file
            trial_params_path = self._generate_trial_params_file(params)
            if not trial_params_path:
                self.logger.debug("❌ Failed to generate trial parameters file")
                return None
            
            # Run SUMMA
            summa_success = self._run_summa_simulation()
            if not summa_success:
                self.logger.debug("❌ SUMMA simulation failed")
                return None
            
            # Run mizuRoute if needed
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                mizuroute_success = self._run_mizuroute_simulation()
                if not mizuroute_success:
                    self.logger.debug("❌ mizuRoute simulation failed")
                    return None
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            if not metrics:
                self.logger.debug("❌ Failed to calculate performance metrics")
                return None
            
            # Get target metric value
            if self.target_metric in metrics:
                score = metrics[self.target_metric]
            elif f"Calib_{self.target_metric}" in metrics:
                score = metrics[f"Calib_{self.target_metric}"]
            elif f"Eval_{self.target_metric}" in metrics:
                score = metrics[f"Eval_{self.target_metric}"]
            else:
                # Use first available metric
                score = next(iter(metrics.values()))
                self.logger.debug(f"Target metric {self.target_metric} not found, using {next(iter(metrics.keys()))} instead")
            
            # Negate the score for metrics where lower is better
            if self.target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                score = -score
            
            eval_duration = datetime.now() - eval_start_time
            self.logger.debug(f"✅ Parameter evaluation completed in {eval_duration.total_seconds():.1f}s")
            
            return score
            
        except Exception as e:
            eval_duration = datetime.now() - eval_start_time
            self.logger.debug(f"❌ Parameter evaluation failed after {eval_duration.total_seconds():.1f}s: {str(e)}")
            return None
        
    def _generate_candidate(self, iteration):
        """
        Generate a candidate solution by perturbing a subset of decision variables.
        Enhanced with debugging to ensure parameters are actually changing.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dict: Dictionary with denormalized parameter values
        """
        # Create copy of normalized parameters
        candidate_normalized = {}
        for param_name, values in self.normalized_params.items():
            candidate_normalized[param_name] = values.copy()
        
        # Calculate probability of selecting each parameter
        # This probability decreases as the iteration count increases
        num_params = len(self.normalized_params)
        selection_probability = 1.0 - np.log(iteration) / np.log(self.max_iterations)
        selection_probability = max(1.0/num_params, selection_probability)
        
        # Select parameters to perturb
        params_to_perturb = []
        for param_name in self.normalized_params.keys():
            if np.random.random() < selection_probability:
                params_to_perturb.append(param_name)
        
        # If no parameters selected, pick one randomly
        if not params_to_perturb:
            params_to_perturb = [np.random.choice(list(self.normalized_params.keys()))]
        
        # DEBUG: Log which parameters are being perturbed
        self.logger.debug(f"Iteration {iteration}: Perturbing {len(params_to_perturb)} parameters: {params_to_perturb}")
        
        # Store original values for comparison
        original_values = {}
        for param_name in params_to_perturb:
            original_values[param_name] = candidate_normalized[param_name].copy()
        
        # Perturb selected parameters
        perturbations_made = {}
        for param_name in params_to_perturb:
            values = candidate_normalized[param_name]
            
            # Apply perturbation to each value
            for i in range(len(values)):
                # Calculate standard deviation for perturbation (based on DDS paper)
                sigma = self.r_value  # DDS perturbation parameter
                
                # Store original value
                original_val = values[i]
                
                # Apply perturbation
                values[i] += sigma * np.random.normal()
                
                # Reflect back into [0, 1] range if needed
                if values[i] < 0:
                    values[i] = -values[i]
                if values[i] > 1:
                    values[i] = 2 - values[i]
                
                # Ensure it's in [0, 1] after reflection
                values[i] = np.clip(values[i], 0, 1)
                
                # Record the perturbation
                perturbation = values[i] - original_val
                if param_name not in perturbations_made:
                    perturbations_made[param_name] = []
                perturbations_made[param_name].append(perturbation)
            
            candidate_normalized[param_name] = values
        
        # DEBUG: Log the actual perturbations made
        for param_name, perturbations in perturbations_made.items():
            avg_perturbation = np.mean(np.abs(perturbations))
            self.logger.debug(f"  {param_name}: avg perturbation = {avg_perturbation:.6f}")
        
        # Denormalize parameters back to original range
        candidate_params = self._denormalize_params(candidate_normalized)
        
        # DEBUG: Compare denormalized values to see if they actually changed
        if hasattr(self, 'current_best_params') and self.current_best_params is not None:
            for param_name in candidate_params.keys():
                if param_name in self.current_best_params:
                    old_val = np.mean(self.current_best_params[param_name])
                    new_val = np.mean(candidate_params[param_name])
                    change = abs(new_val - old_val)
                    if change > 1e-10:  # Only log if there's a meaningful change
                        self.logger.debug(f"  {param_name}: {old_val:.6f} -> {new_val:.6f} (Δ={change:.6f})")
        
        return candidate_params
    
    def _denormalize_params(self, normalized_params):
        """
        Denormalize parameters from [0, 1] range to original range.
        
        Args:
            normalized_params: Dictionary with normalized parameter values
            
        Returns:
            Dict: Dictionary with denormalized parameter values
        """
        denormalized = {}
        
        for param_name, values in normalized_params.items():
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                min_val = bounds['min']
                max_val = bounds['max']
                
                # Denormalize to original range
                denormalized[param_name] = min_val + values * (max_val - min_val)
            else:
                # If bounds not found, just copy the values
                denormalized[param_name] = values.copy()
        
        return denormalized

    def _run_summa_simulation(self):
        """
        Run SUMMA with the current trial parameters using optimization-specific settings.
        
        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.debug("Running SUMMA simulation in optimization environment")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Use the optimization-specific file manager
        file_manager = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not file_manager.exists():
            self.logger.error(f"Optimization file manager not found: {file_manager}")
            return False
        
        # Create command
        summa_command = f"{summa_exe} -m {file_manager}"
        
        # Create log file in optimization logs directory
        log_file = self.summa_sim_dir / "logs" / f"summa_dds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            self.logger.debug("SUMMA simulation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"SUMMA simulation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during SUMMA simulation: {str(e)}")
            return False

    def _run_mizuroute_simulation(self):
        """
        Run mizuRoute with the current SUMMA outputs using optimization-specific settings.
        
        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.debug("Running mizuRoute simulation in optimization environment")
        
        # Get mizuRoute executable path
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Use optimization-specific control file
        control_file = self.optimization_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not control_file.exists():
            self.logger.error(f"Optimization mizuRoute control file not found: {control_file}")
            return False
        
        # Create command
        mizu_command = f"{mizu_exe} {control_file}"
        
        # Create log file in optimization logs directory
        log_file = self.mizuroute_sim_dir / "logs" / f"mizuroute_dds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(mizu_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            self.logger.debug("mizuRoute simulation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"mizuRoute simulation failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during mizuRoute simulation: {str(e)}")
            return False

    def _generate_trial_params_file(self, params):
        """
        Generate a trialParams.nc file with debugging to ensure values are written correctly.
        
        Args:
            params: Dictionary with parameter values
            
        Returns:
            Path: Path to the generated trial parameters file
        """
        self.logger.debug("Generating trial parameters file in optimization settings")
        
        # Get attribute file path for reading HRU information (use optimization copy)
        if not self.attr_file_path.exists():
            self.logger.error(f"Optimization attribute file not found: {self.attr_file_path}")
            return None
        
        return self._generate_trial_params_file_in_directory(
            params, 
            self.optimization_settings_dir, 
            self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
        )

    def run_parameter_extraction(self):
        """
        Run a preliminary SUMMA simulation to extract parameter values.
        
        This will:
        1. Create a modified outputControl.txt that outputs the calibration parameters
        2. Set up a short simulation
        3. Extract the parameter values from the timestep.nc file
        
        Returns:
            Dict: Dictionary with parameter names as keys and arrays of values as values
        """
        self.logger.info("Setting up preliminary parameter extraction run")
        
        # Create a directory for the parameter extraction run
        extract_dir = self.output_dir / "param_extraction"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Create settings directory
        extract_settings_dir = extract_dir / "settings" / "SUMMA"
        extract_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy settings files
        self._copy_static_settings_files(extract_settings_dir)
        
        # Create modified outputControl.txt with calibration parameters added
        self._create_parameter_output_control(extract_settings_dir)
        
        # Create modified fileManager.txt with shorter time period
        self._create_parameter_file_manager(extract_dir, extract_settings_dir)
        
        # Run SUMMA for the parameter extraction
        extract_result = self._run_parameter_extraction_summa(extract_dir)
        
        # Extract parameters from the result file
        if extract_result:
            return self._extract_parameters_from_results(extract_dir)
        else:
            self.logger.warning("Parameter extraction run failed, will use default parameter ranges")
            return None
    
    def _copy_static_settings_files(self, target_dir):
        """Copy static SUMMA settings files to the target directory."""
        source_dir = self.project_dir / "settings" / "SUMMA"
        static_files = [
            'modelDecisions.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL',
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL',
            'attributes.nc',
            'coldState.nc',
            'forcingFileList.txt'
        ]
        
        for file_name in static_files:
            source_path = source_dir / file_name
            if source_path.exists():
                dest_path = target_dir / file_name
                try:
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")
    
    def _create_parameter_output_control(self, settings_dir):
        """
        Create a modified outputControl.txt that outputs the calibration parameters.
        
        Args:
            settings_dir: Path to the settings directory for parameter extraction
        """
        # Read the original outputControl.txt
        orig_output_control = self.project_dir / 'settings' / 'SUMMA' / 'outputControl.txt'
        
        if not orig_output_control.exists():
            self.logger.error(f"Original outputControl.txt not found: {orig_output_control}")
            raise FileNotFoundError(f"Original outputControl.txt not found: {orig_output_control}")
        
        # Read the original file content
        with open(orig_output_control, 'r') as f:
            lines = f.readlines()
        
        # Get all parameters to calibrate
        all_params = self.local_params + self.basin_params
        
        # Create new output control file with the calibration parameters added
        output_file = settings_dir / 'outputControl.txt'
        
        with open(output_file, 'w') as f:
            # Write the original content
            for line in lines:
                f.write(line)
            
            # Add section for calibration parameters if not empty
            if all_params:
                f.write("\n! -----------------------\n")
                f.write("!  calibration parameters\n")
                f.write("! -----------------------\n")
                
                # Add each parameter with a timestep of 1 (output at every timestep)
                for param in all_params:
                    f.write(f"{param:24} | 1\n")
        
        self.logger.info(f"Created modified outputControl.txt with {len(all_params)} calibration parameters")
        return output_file
    
    def _create_parameter_file_manager(self, extract_dir, settings_dir):
        """
        Create a modified fileManager.txt with a shorter time period.
        
        Args:
            extract_dir: Path to the extraction run directory
            settings_dir: Path to the settings directory for parameter extraction
        """
        # Read the original fileManager.txt
        orig_file_manager = self.project_dir / 'settings' / 'SUMMA' / 'fileManager.txt'
        
        if not orig_file_manager.exists():
            self.logger.error(f"Original fileManager.txt not found: {orig_file_manager}")
            raise FileNotFoundError(f"Original fileManager.txt not found: {orig_file_manager}")
        
        # Read the original file content
        with open(orig_file_manager, 'r') as f:
            lines = f.readlines()
        
        # Create simulation directory
        sim_dir = extract_dir / "simulations" / "param_extract" / "SUMMA"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create new file manager
        file_manager = settings_dir / 'fileManager.txt'
        
        with open(file_manager, 'w') as f:
            for line in lines:
                # Modify simulation end time to be short (1 day after start)
                if 'simStartTime' in line:
                    start_time_str = line.split("'")[1]
                    f.write(line)  # Keep the original start time
                    
                    # Parse the start time
                    try:
                        start_parts = start_time_str.split()
                        start_date = start_parts[0]
                        
                        # Use the same start date but run for 1 day only
                        f.write(f"simEndTime           '{start_date} 23:00'\n")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error parsing start time '{start_time_str}': {str(e)}")
                elif 'simEndTime' in line:
                    # Skip the original end time as we've already written our modified one
                    continue
                elif 'outFilePrefix' in line:
                    # Change the output file prefix
                    f.write("outFilePrefix        'param_extract'\n")
                elif 'outputPath' in line:
                    # Change the output path
                    output_path = str(sim_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    f.write(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    # Change the settings path
                    settings_path = str(settings_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    f.write(f"settingsPath         '{settings_path}/'\n")
                else:
                    # Keep all other lines unchanged
                    f.write(line)
        
        self.logger.info(f"Created modified fileManager.txt for parameter extraction")
        return file_manager
    
    def _run_parameter_extraction_summa(self, extract_dir):
        """
        Run SUMMA for parameter extraction.
        
        Args:
            extract_dir: Path to the extraction run directory
            
        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.info("Running SUMMA for parameter extraction")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        file_manager = extract_dir / "settings" / "SUMMA" / 'fileManager.txt'
        
        if not file_manager.exists():
            self.logger.error(f"File manager not found: {file_manager}")
            return False
        
        # Create command
        summa_command = f"{summa_exe} -m {file_manager}"
        
        # Create log directory
        log_dir = extract_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Run SUMMA
        log_file = log_dir / "param_extract_summa.log"
        
        try:
            self.logger.info(f"Running SUMMA command: {summa_command}")
            
            with open(log_file, 'w') as f:
                process = subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            self.logger.info("SUMMA parameter extraction run completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA parameter extraction run failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during SUMMA parameter extraction: {str(e)}")
            return False
                
    def _extract_parameters_from_results(self, extract_dir):
        """
        Extract parameter values from the SUMMA results file.
        Modified to use only the top depth layer for soil parameters.
        
        Args:
            extract_dir: Path to the extraction run directory
                
        Returns:
            dict: Dictionary with parameter names as keys and arrays of values as values
        """
        self.logger.info("Extracting parameter values from SUMMA results")
        
        # Find the timestep output file
        sim_dir = extract_dir / "simulations" / "param_extract" / "SUMMA"
        timestep_files = list(sim_dir.glob("param_extract_timestep.nc"))
        
        if not timestep_files:
            self.logger.error("No timestep output file found from parameter extraction run")
            return None
        
        timestep_file = timestep_files[0]
        
        # Get all parameters to extract
        all_params = self.local_params + self.basin_params
        
        # Identify parameters that should be at GRU level
        gru_level_params = ['routingGammaShape', 'routingGammaScale', 
                            'basin__aquiferHydCond', 'basin__aquiferScaleFactor', 'basin__aquiferBaseflowExp']
        
        # Identify soil parameters that typically vary with depth
        soil_depth_params = [
            'soil_dens_intr', 'theta_sat', 'theta_res', 'vGn_n', 'k_soil', 'k_macropore',
            'specificStorage', 'theta_mp', 'theta_wp', 'vGn_alpha', 'f_impede', 'soilIceScale'
        ]
        
        try:
            # Open the file with xarray for easier handling
            with xr.open_dataset(timestep_file) as ds:
                # Check the dimensions of the output file
                dims = {dim: size for dim, size in ds.sizes.items()}
                self.logger.info(f"Parameter extraction output has dimensions: {dims}")
                
                # Check which parameters are actually in the file
                available_params = [param for param in all_params if param in ds.variables]
                
                if not available_params:
                    self.logger.warning("No calibration parameters found in the output file")
                    self.logger.debug(f"Available variables: {list(ds.variables.keys())}")
                    return None
                
                self.logger.info(f"Found {len(available_params)} out of {len(all_params)} parameters in output")
                
                # Extract the parameter values
                param_values = {}
                
                for param in available_params:
                    # Extract parameter values
                    var = ds[param]
                    
                    # Log the dimensions for debugging
                    self.logger.debug(f"Parameter {param} has dimensions: {var.dims}")
                    
                    # Handle parameters based on their level (GRU, HRU, depth)
                    if param in gru_level_params:
                        # This parameter should be at the GRU level
                        if 'gru' in var.dims:
                            # Parameter is correctly at GRU level
                            if 'time' in var.dims:
                                values = var.isel(time=0).values
                            else:
                                values = var.values
                            
                            param_values[param] = values
                            self.logger.debug(f"Extracted {param} values at GRU level: {values}")
                        elif 'hru' in var.dims:
                            # Parameter is at HRU level but should be at GRU level
                            self.logger.warning(f"Parameter {param} should be at GRU level but is at HRU level in results")
                            
                            if 'time' in var.dims:
                                values = var.isel(time=0).values
                            else:
                                values = var.values
                            
                            # For simplicity and consistency, convert to GRU level by averaging
                            # Get unique mapping from HRU to GRU if available
                            if 'hru2gruId' in ds.variables:
                                # Use proper mapping to calculate GRU-level values
                                hru2gru = ds['hru2gruId'].values
                                unique_grus = np.unique(hru2gru)
                                gru_values = np.zeros(len(unique_grus))
                                
                                for i, gru_id in enumerate(unique_grus):
                                    # Find all HRUs for this GRU
                                    hru_indices = np.where(hru2gru == gru_id)[0]
                                    if len(hru_indices) > 0:
                                        # Average the values from all HRUs in this GRU
                                        gru_values[i] = np.mean(values[hru_indices])
                                
                                param_values[param] = gru_values
                                self.logger.info(f"Mapped {param} from {len(values)} HRUs to {len(gru_values)} GRUs using hru2gruId")
                            else:
                                # No mapping available, use simple average
                                avg_value = np.mean(values)
                                param_values[param] = np.array([avg_value])
                                self.logger.info(f"Averaged {param} from {len(values)} HRUs to a single GRU value: {avg_value}")
                        else:
                            # Parameter might be scalar
                            if 'time' in var.dims:
                                values = var.isel(time=0).values
                            else:
                                values = var.values
                            
                            # Make sure it's a numpy array
                            param_values[param] = np.atleast_1d(values)
                            self.logger.debug(f"Extracted {param} as scalar value: {values}")
                    
                    elif 'depth' in var.dims and param in soil_depth_params:
                        # This is a soil parameter that varies with depth - USE TOP LAYER ONLY
                        
                        if 'time' in var.dims:
                            values = var.isel(time=0).values
                        else:
                            values = var.values
                        
                        # Extract only the top depth layer (index 0)
                        if len(values.shape) > 1:
                            # Multi-dimensional array (e.g., depth x HRU)
                            if 'hru' in var.dims:
                                # Get depth dimension index
                                depth_dim_idx = list(var.dims).index('depth')
                                if depth_dim_idx == 0:
                                    top_layer_values = values[0, :]  # First depth, all HRUs
                                else:
                                    top_layer_values = values[:, 0]  # All HRUs, first depth
                            else:
                                # Just depth dimension
                                top_layer_values = values[0] if values.ndim > 0 else values
                        else:
                            # 1D array - assume it's depth dimension
                            top_layer_values = values[0] if len(values) > 0 else values
                        
                        # Ensure we have a proper array
                        if np.isscalar(top_layer_values):
                            param_values[param] = np.array([top_layer_values])
                        else:
                            param_values[param] = np.atleast_1d(top_layer_values)
                    
                    elif 'hru' in var.dims:
                        # Standard parameter at HRU level
                        if 'time' in var.dims:
                            values = var.isel(time=0).values
                        else:
                            values = var.values
                        
                        param_values[param] = values
                        self.logger.debug(f"Extracted {param} values for {len(values)} HRUs")
                    
                    else:
                        # Parameter is scalar or has unexpected dimensions
                        if 'time' in var.dims:
                            values = var.isel(time=0).values
                        else:
                            values = var.values
                        
                        # Make sure it's a numpy array
                        param_values[param] = np.atleast_1d(values)
                        self.logger.debug(f"Extracted {param} with dimensions {var.dims}, shape: {param_values[param].shape}")
                
                # Final check for parameters that should be at GRU level
                for param in gru_level_params:
                    if param in param_values:
                        values = param_values[param]
                        if len(values) > 1 and param in ['routingGammaShape', 'routingGammaScale']:
                            self.logger.warning(f"Parameter {param} has {len(values)} values but should have one value per GRU")
                            self.logger.info(f"Using mean value for {param}: {np.mean(values)}")
                            param_values[param] = np.array([np.mean(values)])
                
                self.logger.info(f"Successfully extracted values for {len(param_values)} parameters")
                return param_values
                
        except Exception as e:
            self.logger.error(f"Error extracting parameters from results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _parse_parameter_bounds(self):
        """
        Parse parameter bounds from SUMMA parameter info files.
        
        Returns:
            Dict: Dictionary with parameter names as keys and bound dictionaries as values
        """
        self.logger.info("Parsing parameter bounds from parameter info files")
        
        bounds = {}
        
        # Parse local parameter bounds
        if self.local_params:
            local_bounds = self._parse_param_info_file(self.local_param_info_path, self.local_params)
            bounds.update(local_bounds)
        
        # Parse basin parameter bounds
        if self.basin_params:
            basin_bounds = self._parse_param_info_file(self.basin_param_info_path, self.basin_params)
            bounds.update(basin_bounds)
        
        if not bounds:
            self.logger.error("No parameter bounds found")
            raise ValueError("No parameter bounds found")
        
        self.logger.info(f"Found bounds for {len(bounds)} parameters")
        return bounds
        
    def _parse_param_info_file(self, file_path, param_names):
        """
        Parse parameter bounds from a SUMMA parameter info file.
        
        Args:
            file_path: Path to the parameter info file
            param_names: List of parameter names to extract bounds for
            
        Returns:
            Dict: Dictionary with parameter names as keys and bound dictionaries as values
        """
        bounds = {}
        
        if not file_path.exists():
            self.logger.error(f"Parameter info file not found: {file_path}")
            return bounds
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!'):
                        continue
                    
                    # Parse the line to extract parameter name and bounds
                    # Format is typically: paramName | maxValue | minValue | ...
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        param_name = parts[0]
                        if param_name in param_names:
                            try:
                                # Handle Fortran-style scientific notation (1.0d+6 -> 1.0e+6)
                                max_val_str = parts[1].replace('d', 'e').replace('D', 'e')
                                min_val_str = parts[2].replace('d', 'e').replace('D', 'e')
                                
                                max_val = float(max_val_str)
                                min_val = float(min_val_str)
                                bounds[param_name] = {'min': min_val, 'max': max_val}
                                self.logger.debug(f"Found bounds for {param_name}: min={min_val}, max={max_val}")
                            except ValueError as ve:
                                self.logger.warning(f"Could not parse bounds for parameter '{param_name}' in line: {line}. Error: {str(ve)}")
        except Exception as e:
            self.logger.error(f"Error parsing parameter info file {file_path}: {str(e)}")
        
        return bounds

    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics by comparing simulated to observed streamflow.
        FIXED: Now reads from the DDS optimization simulation directory.
        
        Returns:
            Dict: Dictionary with performance metrics
        """
        self.logger.debug("Calculating performance metrics from DDS optimization simulation")
        
        # Get observed data path
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and flow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if not date_col or not flow_col:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Get simulated data 
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # From mizuRoute output - USE DDS OPTIMIZATION DIRECTORY
                sim_dir = self.mizuroute_sim_dir  # Changed from self.project_dir / "simulations" / self.experiment_id / "mizuRoute"
                sim_files = list(sim_dir.glob("*timestep.nc"))
                
                self.logger.debug(f"Looking for mizuRoute files in: {sim_dir}")
                self.logger.debug(f"Found {len(sim_files)} mizuRoute files: {[f.name for f in sim_files]}")
                
                if not sim_files:
                    self.logger.error(f"No mizuRoute output files found in DDS optimization directory: {sim_dir}")
                    return None
                
                sim_file = sim_files[0]  # Take the most recent/first file
                self.logger.debug(f"Using mizuRoute file: {sim_file}")
                
                # Open the file with xarray
                with xr.open_dataset(sim_file) as ds:
                    # Get reach ID
                    sim_reach_id = self.config.get('SIM_REACH_ID')
                    
                    # Find the index for the reach ID
                    if 'reachID' in ds.variables:
                        reach_indices = np.where(ds['reachID'].values == int(sim_reach_id))[0]
                        
                        if len(reach_indices) > 0:
                            reach_index = reach_indices[0]
                            
                            # Try common variable names
                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                if var_name in ds.variables:
                                    simulated_flow = ds[var_name].isel(seg=reach_index).to_pandas()
                                    self.logger.debug(f"Extracted {var_name} from mizuRoute output")
                                    break
                            else:
                                self.logger.error("Could not find streamflow variable in mizuRoute output")
                                return None
                        else:
                            self.logger.error(f"Reach ID {sim_reach_id} not found in mizuRoute output")
                            return None
                    else:
                        self.logger.error("No reachID variable found in mizuRoute output")
                        return None
            else:
                # From SUMMA output - USE DDS OPTIMIZATION DIRECTORY
                sim_dir = self.summa_sim_dir  # Changed from self.project_dir / "simulations" / self.experiment_id / "SUMMA"
                
                # Look for DDS optimization output files
                sim_files = list(sim_dir.glob("run_dds_opt_*timestep.nc"))
                
                self.logger.debug(f"Looking for SUMMA files in: {sim_dir}")
                self.logger.debug(f"Found {len(sim_files)} SUMMA files: {[f.name for f in sim_files]}")
                
                if not sim_files:
                    self.logger.error(f"No SUMMA output files found in DDS optimization directory: {sim_dir}")
                    # Try to find any .nc files as fallback
                    all_nc_files = list(sim_dir.glob("*.nc"))
                    if all_nc_files:
                        self.logger.warning(f"Found {len(all_nc_files)} other .nc files: {[f.name for f in all_nc_files]}")
                        sim_files = all_nc_files
                    else:
                        return None
                
                sim_file = sim_files[-1]  # Take the most recent file
                self.logger.debug(f"Using SUMMA file: {sim_file}")
                
                # Open the file with xarray
                with xr.open_dataset(sim_file) as ds:
                    self.logger.debug(f"SUMMA output variables: {list(ds.variables.keys())}")
                    self.logger.debug(f"SUMMA output dimensions: {dict(ds.sizes)}")
                    
                    # Try to find streamflow variable
                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                # Sum across GRUs if multiple
                                simulated_flow = ds[var_name].sum(dim='gru').to_pandas()
                                self.logger.debug(f"Extracted {var_name} (summed across {ds.sizes['gru']} GRUs)")
                            else:
                                # Single GRU or no gru dimension
                                simulated_flow = ds[var_name].to_pandas()
                                if isinstance(simulated_flow, pd.DataFrame):
                                    simulated_flow = simulated_flow.iloc[:, 0]
                                self.logger.debug(f"Extracted {var_name} from single GRU")
                            break
                    else:
                        self.logger.error(f"Could not find streamflow variable in SUMMA output. Available variables: {list(ds.variables.keys())}")
                        return None
                    
                    # Get catchment area for unit conversion
                    catchment_area = self._get_catchment_area()
                    if catchment_area is not None and catchment_area > 0:
                        # Convert from m/s to m³/s
                        simulated_flow = simulated_flow * catchment_area
                        self.logger.debug(f"Converted from m/s to m³/s using catchment area: {catchment_area/1e6:.2f} km²")
            
            # Log the time ranges for debugging
            self.logger.debug(f"Observed flow time range: {observed_flow.index.min()} to {observed_flow.index.max()}")
            self.logger.debug(f"Simulated flow time range: {simulated_flow.index.min()} to {simulated_flow.index.max()}")
            
            # Calculate metrics for calibration and evaluation periods
            metrics = {}
            calib_start, calib_end = self.calibration_period
            eval_start, eval_end = self.evaluation_period
            
            # Calculate metrics for calibration period
            if calib_start and calib_end:
                calib_mask = (observed_flow.index >= calib_start) & (observed_flow.index <= calib_end)
                calib_obs = observed_flow[calib_mask]
                
                # Find common time steps
                common_idx = calib_obs.index.intersection(simulated_flow.index)
                if len(common_idx) > 0:
                    calib_obs_common = calib_obs.loc[common_idx]
                    calib_sim_common = simulated_flow.loc[common_idx]
                    
                    self.logger.debug(f"Calibration period: {len(common_idx)} common time steps")
                    
                    # Calculate metrics
                    calib_metrics = self._calculate_streamflow_metrics(calib_obs_common, calib_sim_common)
                    
                    # Add to metrics with Calib_ prefix
                    for key, value in calib_metrics.items():
                        metrics[f"Calib_{key}"] = value
                    
                    # Also add without prefix for convenience
                    metrics.update(calib_metrics)
            
            # Calculate metrics for evaluation period
            if eval_start and eval_end:
                eval_mask = (observed_flow.index >= eval_start) & (observed_flow.index <= eval_end)
                eval_obs = observed_flow[eval_mask]
                
                # Find common time steps
                common_idx = eval_obs.index.intersection(simulated_flow.index)
                if len(common_idx) > 0:
                    eval_obs_common = eval_obs.loc[common_idx]
                    eval_sim_common = simulated_flow.loc[common_idx]
                    
                    self.logger.debug(f"Evaluation period: {len(common_idx)} common time steps")
                    
                    # Calculate metrics
                    eval_metrics = self._calculate_streamflow_metrics(eval_obs_common, eval_sim_common)
                    
                    # Add to metrics with Eval_ prefix
                    for key, value in eval_metrics.items():
                        metrics[f"Eval_{key}"] = value
            
            # If no calibration or evaluation periods, calculate for the entire period
            if not metrics:
                # Find common time steps
                common_idx = observed_flow.index.intersection(simulated_flow.index)
                if len(common_idx) > 0:
                    obs_common = observed_flow.loc[common_idx]
                    sim_common = simulated_flow.loc[common_idx]
                    
                    self.logger.debug(f"Full period: {len(common_idx)} common time steps")
                    
                    # Calculate metrics
                    metrics = self._calculate_streamflow_metrics(obs_common, sim_common)
            
            # Log the calculated metrics for debugging
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    self.logger.debug(f"Calculated {metric_name}: {value:.6f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_streamflow_metrics(self, observed, simulated):
        """
        Calculate streamflow performance metrics.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        
        Returns:
            Dict: Dictionary of performance metrics
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
        
        # Mean Absolute Error (MAE)
        mae = (observed - simulated).abs().mean()
        
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
            gdf = gpd.read_file(basin_shapefile)
            
            # Try to get area from attributes first
            area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
            
            if area_col in gdf.columns:
                # Sum all basin areas (in case of multiple basins)
                total_area = gdf[area_col].sum()
                
                # Check if the area seems reasonable
                if total_area <= 0 or total_area > 1e12:  # Suspicious if > 1 million km²
                    self.logger.warning(f"Area from attribute {area_col} seems unrealistic: {total_area} m². Calculating geometrically.")
                else:
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

    def _create_optimization_plots(self, history):
        """
        Create plots showing the optimization progress.
        
        Args:
            history: List of dictionaries with optimization history
        """
        self.logger.info("Creating optimization plots")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create output directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract data from history - filter out None scores
            iterations = []
            scores = []
            valid_indices = []
            
            for i, h in enumerate(history):
                if h['score'] is not None and not np.isnan(h['score']):
                    iterations.append(h['iteration'])
                    scores.append(h['score'])
                    valid_indices.append(i)
            
            if not scores:
                self.logger.warning("No valid scores found for plotting")
                return
            
            # Convert to numpy arrays for easier handling
            iterations = np.array(iterations)
            scores = np.array(scores)
            
            # Plot optimization progress
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, scores, 'b-o', markersize=4)
            plt.xlabel('Iteration')
            plt.ylabel(f'Performance Metric ({self.target_metric})')
            plt.title('DDS Optimization Progress')
            plt.grid(True, alpha=0.3)
            
            # Add best iteration marker
            best_idx = np.argmax(scores)  # Now safe since we filtered out None values
            plt.plot(iterations[best_idx], scores[best_idx], 'ro', markersize=10, 
                    label=f'Best: {scores[best_idx]:.4f} at iteration {iterations[best_idx]}')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Optimization plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _plot_streamflow_comparison(self):
        """
        Create a plot comparing observed and simulated streamflow for the best parameter set.
        Modified to exclude spinup period from visualization.
        """
        self.logger.info("Creating streamflow comparison plot")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            import pandas as pd
            import numpy as np
            
            # Create output directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Get plot period (exclude spinup)
            calib_start, calib_end = self.calibration_period
            experiment_end = pd.Timestamp(self.config.get('EXPERIMENT_TIME_END', '').split()[0])
            
            # Use calibration start as plot start (excluding spinup)
            if calib_start:
                plot_start = calib_start
            else:
                # Fallback to experiment start if no calibration period defined
                plot_start = pd.Timestamp(self.config.get('EXPERIMENT_TIME_START', '').split()[0])
            
            plot_end = experiment_end if experiment_end else pd.Timestamp('2022-12-31')
            
            # Get observed data
            obs_path = self.config.get('OBSERVATIONS_PATH')
            if obs_path == 'default':
                obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
            
            if not obs_path.exists():
                self.logger.error(f"Observed streamflow file not found: {obs_path}")
                return
            
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and flow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if not date_col or not flow_col:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Filter observed data to plot period (exclude spinup)
            plot_mask = (observed_flow.index >= plot_start) & (observed_flow.index <= plot_end)
            observed_flow_plot = observed_flow[plot_mask]
            
            # Get simulated streamflow
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # From mizuRoute output
                sim_dir = self.project_dir / "simulations" / self.experiment_id / "mizuRoute"
                sim_files = list(sim_dir.glob("*.nc"))
                
                if not sim_files:
                    self.logger.error("No mizuRoute output files found")
                    return
                
                sim_file = sim_files[0]
                
                # Get reach ID
                sim_reach_id = self.config.get('SIM_REACH_ID')
                
                # Open file and extract streamflow
                with xr.open_dataset(sim_file) as ds:
                    if 'reachID' in ds.variables:
                        reach_indices = np.where(ds['reachID'].values == int(sim_reach_id))[0]
                        
                        if len(reach_indices) > 0:
                            reach_index = reach_indices[0]
                            
                            # Try common variable names
                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                if var_name in ds.variables:
                                    simulated_flow = ds[var_name].isel(seg=reach_index).to_pandas()
                                    break
                            else:
                                self.logger.error("Could not find streamflow variable in mizuRoute output")
                                return
                        else:
                            self.logger.error(f"Reach ID {sim_reach_id} not found in mizuRoute output")
                            return
                    else:
                        self.logger.error("No reachID variable found in mizuRoute output")
                        return
            else:
                # From SUMMA output
                sim_dir = self.project_dir / "simulations" / self.experiment_id / "SUMMA"
                sim_files = list(sim_dir.glob(f"{self.experiment_id}*.nc"))
                
                if not sim_files:
                    self.logger.error("No SUMMA output files found")
                    return
                
                sim_file = sim_files[0]
                
                # Open file and extract streamflow
                with xr.open_dataset(sim_file) as ds:
                    # Try to find streamflow variable
                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                # Sum across GRUs if multiple
                                simulated_flow = ds[var_name].sum(dim='gru').to_pandas()
                            else:
                                # Single GRU or no gru dimension
                                simulated_flow = ds[var_name].to_pandas()
                                if isinstance(simulated_flow, pd.DataFrame):
                                    simulated_flow = simulated_flow.iloc[:, 0]
                            
                            # Get catchment area
                            catchment_area = self._get_catchment_area()
                            if catchment_area is not None and catchment_area > 0:
                                # Convert from m/s to m³/s
                                simulated_flow = simulated_flow * catchment_area
                            
                            break
                    else:
                        self.logger.error("Could not find streamflow variable in SUMMA output")
                        return
            
            # Filter simulated data to plot period (exclude spinup)
            sim_plot_mask = (simulated_flow.index >= plot_start) & (simulated_flow.index <= plot_end)
            simulated_flow_plot = simulated_flow[sim_plot_mask]
            
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Plot observed and simulated flow
            plt.plot(observed_flow_plot.index, observed_flow_plot, 'b-', linewidth=1.5, label='Observed')
            plt.plot(simulated_flow_plot.index, simulated_flow_plot, 'r-', linewidth=1.5, label='Simulated (Best Parameters)')
            
            # Format plot
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title(f'Observed vs. Simulated Streamflow (Best Parameters)\nPeriod: {plot_start.strftime("%Y-%m-%d")} to {plot_end.strftime("%Y-%m-%d")} (Excludes Spinup)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Add performance metrics
            final_metrics = self._calculate_performance_metrics()
            if final_metrics:
                calib_metrics = {k: v for k, v in final_metrics.items() if k.startswith('Calib_')}
                eval_metrics = {k: v for k, v in final_metrics.items() if k.startswith('Eval_')}
                
                # Create text for metrics
                metrics_text = "Performance Metrics:\n"
                if calib_metrics:
                    metrics_text += "Calibration Period:\n"
                    for metric, value in calib_metrics.items():
                        metric_name = metric.replace('Calib_', '')
                        if not np.isnan(value):
                            metrics_text += f"  {metric_name}: {value:.4f}\n"
                if eval_metrics:
                    metrics_text += "Evaluation Period:\n"
                    for metric, value in eval_metrics.items():
                        metric_name = metric.replace('Eval_', '')
                        if not np.isnan(value):
                            metrics_text += f"  {metric_name}: {value:.4f}\n"
                if not calib_metrics and not eval_metrics:
                    # Use main metrics
                    for metric in ['KGE', 'NSE', 'RMSE', 'PBIAS']:
                        if metric in final_metrics and not np.isnan(final_metrics[metric]):
                            metrics_text += f"  {metric}: {final_metrics[metric]:.4f}\n"
                
                # Add text to plot
                plt.text(0.01, 0.98, metrics_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create a scatter plot (using plot period data)
            plt.figure(figsize=(10, 8))
            
            # Find common timepoints in plot period
            common_idx = observed_flow_plot.index.intersection(simulated_flow_plot.index)
            obs_common = observed_flow_plot.loc[common_idx]
            sim_common = simulated_flow_plot.loc[common_idx]
            
            # Create scatter plot
            plt.scatter(obs_common, sim_common, alpha=0.5, s=10)
            
            # Add 1:1 line
            max_val = max(obs_common.max(), sim_common.max())
            min_val = min(obs_common.min(), sim_common.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
            
            # Add regression line
            try:
                from scipy import stats
                if len(obs_common) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(obs_common, sim_common)
                    plt.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], 'r-',
                            label=f'Regression Line (r={r_value:.3f})')
            except ImportError:
                self.logger.warning("scipy not available for regression line")
            
            # Format plot
            plt.xlabel('Observed Streamflow (m³/s)')
            plt.ylabel('Simulated Streamflow (m³/s)')
            plt.title(f'Observed vs. Simulated Streamflow (Scatter Plot)\nPeriod: {plot_start.strftime("%Y-%m-%d")} to {plot_end.strftime("%Y-%m-%d")} (Excludes Spinup)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Try to set equal aspect ratio
            try:
                plt.axis('equal')
            except:
                pass
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create flow duration curve (using plot period data)
            plt.figure(figsize=(10, 8))
            
            # Sort flows for FDC
            obs_sorted = observed_flow_plot.dropna().sort_values(ascending=False)
            sim_sorted = simulated_flow_plot.dropna().sort_values(ascending=False)
            
            # Calculate exceedance probabilities
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
            
            # Plot FDCs
            plt.plot(obs_exceed, obs_sorted, 'b-', linewidth=1.5, label='Observed')
            plt.plot(sim_exceed, sim_sorted, 'r-', linewidth=1.5, label='Simulated (Best Parameters)')
            
            # Format plot
            plt.xlabel('Exceedance Probability')
            plt.ylabel('Streamflow (m³/s)')
            plt.title(f'Flow Duration Curves\nPeriod: {plot_start.strftime("%Y-%m-%d")} to {plot_end.strftime("%Y-%m-%d")} (Excludes Spinup)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "flow_duration_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Streamflow comparison plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating streamflow comparison plot: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())