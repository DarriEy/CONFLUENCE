#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

class DEOptimizer:
    """
    Differential Evolution (DE) Optimizer for CONFLUENCE 
    
    This class performs parameter optimization using the Differential Evolution algorithm,
    which is a population-based stochastic optimization technique.

    The soil depth calibration uses two parameters:
    - total_mult: Overall depth multiplier (0.1-5.0)
    - shape_factor: Controls depth profile shape (0.1-3.0)
      - shape_factor > 1: Deeper layers get proportionally thicker
      - shape_factor < 1: Shallower layers get proportionally thicker
      - shape_factor = 1: Uniform scaling
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the DE Optimizer.
        
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
        
        # Check if depth calibration is enabled
        self.calibrate_depth = self.config.get('CALIBRATE_DEPTH', False)
        self.logger.info(f"Soil depth calibration: {'ENABLED' if self.calibrate_depth else 'DISABLED'}")
        
        # Default settings directory (where we save/load optimized parameters)
        self.default_settings_dir = self.project_dir / "settings" / "SUMMA"
        self.default_trial_params_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
        self.default_coldstate_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
        
        # Parse time periods FIRST (needed for optimization environment setup)
        calib_period = self.config.get('CALIBRATION_PERIOD', '')
        eval_period = self.config.get('EVALUATION_PERIOD', '')
        self.calibration_period = self._parse_date_range(calib_period)
        self.evaluation_period = self._parse_date_range(eval_period)
        
        # Create DE-specific directories
        self.optimization_dir = self.project_dir / "simulations" / "run_de"
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        
        # Create output directory for results
        self.output_dir = self.project_dir / "optimisation" / f"de_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimization-specific directories and settings
        self._setup_optimization_environment()
        
        # Get optimization settings
        self.max_iterations = self.config.get('NUMBER_OF_ITERATIONS', 100)
        
        # DE specific parameters with sensible defaults
        self.F = self.config.get('DE_SCALING_FACTOR', 0.5)  # Mutation scaling factor
        self.CR = self.config.get('DE_CROSSOVER_RATE', 0.9)  # Crossover probability
        self.population_size = self.config.get('POPULATION_SIZE', None)  # Will be set based on dimensions
        
        # Define parameter bounds (using optimization settings paths)
        self.local_param_info_path = self.optimization_settings_dir / 'localParamInfo.txt'
        self.basin_param_info_path = self.optimization_settings_dir / 'basinParamInfo.txt'
        
        # Get parameters to calibrate
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []
        
        # Add depth parameters if depth calibration is enabled
        self.depth_params = []
        if self.calibrate_depth:
            self.depth_params = ['total_mult', 'shape_factor']
            self.logger.info("Added soil depth parameters: total_mult, shape_factor")
        
        # Add mizuRoute parameters if routing calibration is enabled
        self.mizuroute_params = []
        if self.config.get('CALIBRATE_MIZUROUTE', False):
            # Get mizuRoute parameters to calibrate from config
            mizuroute_params_str = self.config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            self.mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()] if mizuroute_params_str else ['velo', 'diff']
            self.logger.info(f"Added mizuRoute parameters for calibration: {self.mizuroute_params}")
        
        # Check if mizuRoute settings exist
        self.mizuroute_settings_dir = self.project_dir / "settings" / "mizuRoute"
        self.mizuroute_param_file = self.mizuroute_settings_dir / "param.nml.default"
        if self.mizuroute_params and not self.mizuroute_param_file.exists():
            self.logger.warning(f"mizuRoute parameter file not found: {self.mizuroute_param_file}")
            self.logger.warning("mizuRoute calibration may fail without this file")
        elif self.mizuroute_params:
            self.logger.info(f"mizuRoute parameter file found: {self.mizuroute_param_file}")
        
        # Get performance metric settings
        self.target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Initialize tracking variables
        self.population = None
        self.population_scores = None
        self.best_params = None
        self.best_score = float('-inf')
        self.iteration_history = []
        self.param_bounds = None  # Will be set during initialization
        
        # Get attribute file path (use the copied one in optimization settings)
        self.attr_file_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
        
        # Store original soil depths for depth calibration
        self.original_depths = None
        if self.calibrate_depth:
            self.original_depths = self._get_original_depths()
            if self.original_depths is not None:
                self.logger.info(f"Loaded original soil depths: {len(self.original_depths)} layers")
                self.logger.debug(f"Original depths (m): {self.original_depths}")
            else:
                self.logger.warning("Could not load original soil depths - depth calibration may fail")
        
        # Set population size based on number of parameters if not specified
        total_params = len(self.local_params) + len(self.basin_params) + len(self.depth_params)
        if self.population_size is None:
            self.population_size = max(15, min(4 * total_params, 50))  # 4*D, between 15-50
        
        # Logging
        self.logger.info(f"DE Optimizer initialized with {len(self.local_params)} local, {len(self.basin_params)} basin, {len(self.depth_params)} depth, and {len(self.mizuroute_params)} mizuRoute parameters")
        self.logger.info(f"Maximum generations: {self.max_iterations}, population size: {self.population_size}")
        self.logger.info(f"DE parameters: F={self.F}, CR={self.CR}")
        self.logger.info(f"Optimization simulations will run in: {self.summa_sim_dir}")
        
        # Check routing configuration
        routing_info = self._check_routing_configuration()
        self.logger.info(f"Routing configuration: {routing_info}")
        
        # Log optimization period
        opt_period = self._get_optimization_period_string()
        self.logger.info(f"Optimization period: {opt_period}")
    
    def _check_routing_configuration(self):
        """
        Check the routing configuration and log relevant information.
        
        Returns:
            str: Description of the routing configuration
        """
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
        
        needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
        
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return f"Lumped SUMMA + Distributed mizuRoute (needs conversion and routing: {needs_mizuroute})"
        elif domain_method != 'lumped':
            return f"Distributed SUMMA + mizuRoute (needs routing: {needs_mizuroute})"
        else:
            return f"Lumped SUMMA only (needs routing: {needs_mizuroute})"
    
    def _needs_mizuroute_routing(self, domain_method: str, routing_delineation: str) -> bool:
        """
        Determine if mizuRoute routing is needed based on domain and routing configuration.
        This mirrors the logic from model_manager.py.
        
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
        This mirrors the functionality from model_manager.py.
        
        This method replicates the functionality from the manual script,
        broadcasting the single lumped runoff to all routing segments.
        Creates mizuRoute-compatible files with proper naming and variables.
        Only processes the timestep file for mizuRoute routing.
        """
        self.logger.info("Converting lumped SUMMA output for distributed routing")
        
        try:
            # Import required modules
            import tempfile
            
            # Paths - use DE optimization directories
            summa_output_dir = self.summa_sim_dir
            mizuroute_settings_dir = self.optimization_dir / "settings" / "mizuRoute"
            
            # Find the most recent DE optimization timestep file
            summa_timestep_files = list(summa_output_dir.glob("run_de_opt_*timestep.nc"))
            
            if not summa_timestep_files:
                # Fallback: look for any timestep files
                summa_timestep_files = list(summa_output_dir.glob("*timestep.nc"))
            
            if not summa_timestep_files:
                raise FileNotFoundError(f"No SUMMA timestep output files found in: {summa_output_dir}")
            
            # Use the most recent file
            summa_timestep_file = max(summa_timestep_files, key=lambda x: x.stat().st_mtime)
            self.logger.debug(f"Using SUMMA timestep file: {summa_timestep_file}")
            
            # Load mizuRoute topology to get segment information
            topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
            
            if not topology_file.exists():
                raise FileNotFoundError(f"mizuRoute topology file not found: {topology_file}")
            
            with xr.open_dataset(topology_file) as mizuTopology:
                # Use SEGMENT IDs from topology (these are the routing elements we broadcast to)
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
                
                # Create GRU dimension using SEGMENT IDs
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
                
                # Tile to all SEGMENTS: (time,) -> (time, n_segments)
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
    
    def _get_original_depths(self):
        """
        Get original soil depths from coldState.nc file.
        
        Returns:
            np.ndarray: Array of original soil layer depths, or None if not found
        """
        try:
            # Try optimization settings directory first
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            # Fallback to default settings directory
            if not coldstate_path.exists():
                coldstate_path = self.default_coldstate_path
            
            if not coldstate_path.exists():
                self.logger.error(f"coldState.nc not found in either {self.optimization_settings_dir} or {self.default_settings_dir}")
                return None
            
            with nc.Dataset(coldstate_path, 'r') as ds:
                if 'mLayerDepth' not in ds.variables:
                    self.logger.error("mLayerDepth variable not found in coldState.nc")
                    return None
                
                # Get depths for the first HRU (assuming uniform depths across HRUs)
                depths = ds.variables['mLayerDepth'][:, 0].copy()
                
                self.logger.debug(f"Loaded {len(depths)} soil layers from {coldstate_path}")
                return depths
                
        except Exception as e:
            self.logger.error(f"Error reading original soil depths: {str(e)}")
            return None
    
    def _calculate_new_depths(self, total_mult, shape_factor):
        """
        Calculate new soil depths using the shape method.
        
        Args:
            total_mult: Overall depth multiplier
            shape_factor: Shape factor controlling depth profile
            
        Returns:
            np.ndarray: Array of new soil layer depths
        """
        if self.original_depths is None:
            self.logger.error("Original depths not available for depth calculation")
            return None
        
        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        # Calculate shape weights
        if shape_factor > 1:
            # Deeper layers get proportionally thicker
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            # Shallower layers get proportionally thicker
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            # Uniform scaling
            w = np.ones(n)
        
        # Normalize weights to preserve total depth scaling
        w /= w.mean()
        
        # Apply total multiplier and shape weights
        new_depths = arr * w * total_mult
        
        self.logger.debug(f"Shape calculation: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")
        self.logger.debug(f"New depths (m): {new_depths}")
        
        return new_depths
    
    def _update_soil_depths(self, new_depths):
        """
        Update the coldState.nc file with new soil depths.
        
        Args:
            new_depths: Array of new soil layer depths
        """
        try:
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            if not coldstate_path.exists():
                self.logger.error(f"coldState.nc not found in optimization settings: {coldstate_path}")
                return False
            
            # Calculate layer heights (cumulative depths)
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]
            
            # Update the file
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    self.logger.error("Required depth variables not found in coldState.nc")
                    return False
                
                # Update depths for all HRUs
                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights
            
            self.logger.debug(f"Updated soil depths in {coldstate_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating soil depths: {str(e)}")
            return False
    
    def _load_existing_optimized_parameters(self):
        """
        Load existing optimized parameters from the default settings directory.
        Enhanced to handle depth parameters.
        
        Returns:
            Dict: Dictionary with parameter values if found, None otherwise
        """
        if not self.default_trial_params_path.exists():
            self.logger.info("No existing optimized parameters found - will start from scratch")
            return None
        
        try:
            self.logger.info(f"Loading existing optimized parameters from: {self.default_trial_params_path}")
            
            # Get all parameters to extract (excluding depth parameters)
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
                
                # Add default depth parameters if depth calibration is enabled
                if self.calibrate_depth:
                    param_values['total_mult'] = np.array([1.0])  # Default: no depth scaling
                    param_values['shape_factor'] = np.array([1.0])  # Default: uniform scaling
                    self.logger.info("Added default depth parameters: total_mult=1.0, shape_factor=1.0")
                
                # Add mizuRoute parameters if routing calibration is enabled
                if self.mizuroute_params:
                    # Try to load from existing mizuRoute parameter file
                    mizuroute_values = self._load_existing_mizuroute_parameters()
                    if mizuroute_values:
                        param_values.update(mizuroute_values)
                        self.logger.info(f"Loaded existing mizuRoute parameters: {list(mizuroute_values.keys())}")
                    else:
                        # Use default values if no existing parameters
                        for param in self.mizuroute_params:
                            param_values[param] = self._get_default_mizuroute_value(param)
                        self.logger.info("Added default mizuRoute parameters")
                
                self.logger.info(f"Successfully loaded existing optimized parameters for {len(param_values)} parameters")
                return param_values
                
        except Exception as e:
            self.logger.error(f"Error loading existing optimized parameters: {str(e)}")
            self.logger.error("Will proceed with parameter extraction instead")
            return None
    
    def _save_best_parameters_to_default_settings(self, best_params):
        """
        Save the best parameters to the default model settings directory.
        Enhanced to handle depth parameters.
        
        This method:
        1. Backs up the existing trialParams.nc (if it exists) to trialParams_default.nc
        2. Saves the optimized parameters as the new trialParams.nc
        3. Updates soil depths in coldState.nc if depth calibration was used
        
        Args:
            best_params: Dictionary with the best parameter values
        """
        self.logger.info("Saving best parameters to default model settings")
        
        try:
            # Check if default settings directory exists
            if not self.default_settings_dir.exists():
                self.logger.error(f"Default settings directory not found: {self.default_settings_dir}")
                return False
            
            success = True
            
            # Step 1: Save hydraulic parameters to trialParams.nc
            hydraulic_params = {k: v for k, v in best_params.items() if k not in self.depth_params}
            
            if hydraulic_params:
                # Backup existing trialParams.nc if it exists
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
                
                # Generate new trialParams.nc with best hydraulic parameters
                new_trial_params_path = self._generate_trial_params_file_in_directory(
                    hydraulic_params, 
                    self.default_settings_dir,
                    "trialParams.nc"
                )
                
                if new_trial_params_path:
                    self.logger.info(f"✅ Successfully saved optimized hydraulic parameters to: {new_trial_params_path}")
                    
                    # Also save a copy with experiment ID for tracking
                    experiment_copy = self.default_settings_dir / f"trialParams_optimized_{self.experiment_id}.nc"
                    shutil.copy2(new_trial_params_path, experiment_copy)
                    self.logger.info(f"Created experiment-specific copy: {experiment_copy}")
                else:
                    self.logger.error("Failed to generate new trialParams.nc file")
                    success = False
            
            # Step 2: Save depth parameters to coldState.nc
            if self.calibrate_depth and 'total_mult' in best_params and 'shape_factor' in best_params:
                # Backup existing coldState.nc if it exists
                if self.default_coldstate_path.exists():
                    backup_path = self.default_settings_dir / "coldState_default.nc"
                    
                    # If backup already exists, create a timestamped backup
                    if backup_path.exists():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        timestamped_backup = self.default_settings_dir / f"coldState_backup_{timestamp}.nc"
                        shutil.copy2(backup_path, timestamped_backup)
                        self.logger.info(f"Created timestamped coldState backup: {timestamped_backup}")
                    
                    # Move existing coldState to backup
                    shutil.copy2(self.default_coldstate_path, backup_path)
                    self.logger.info(f"Backed up existing coldState.nc to: {backup_path}")
                
                # Calculate and apply new depths
                total_mult = best_params['total_mult'][0] if isinstance(best_params['total_mult'], np.ndarray) else best_params['total_mult']
                shape_factor = best_params['shape_factor'][0] if isinstance(best_params['shape_factor'], np.ndarray) else best_params['shape_factor']
                
                new_depths = self._calculate_new_depths(total_mult, shape_factor)
                if new_depths is not None:
                    # Copy the optimized coldState to default settings
                    optimized_coldstate = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
                    if optimized_coldstate.exists():
                        shutil.copy2(optimized_coldstate, self.default_coldstate_path)
                        self.logger.info(f"✅ Successfully saved optimized soil depths to: {self.default_coldstate_path}")
                        
                        # Also save a copy with experiment ID for tracking
                        experiment_copy = self.default_settings_dir / f"coldState_optimized_{self.experiment_id}.nc"
                        shutil.copy2(self.default_coldstate_path, experiment_copy)
                        self.logger.info(f"Created experiment-specific coldState copy: {experiment_copy}")
                    else:
                        self.logger.error("Optimized coldState.nc not found")
                        success = False
                else:
                    self.logger.error("Failed to calculate new soil depths")
                    success = False
            
            # Step 3: Save mizuRoute parameters if they were calibrated
            if self.mizuroute_params and any(param in best_params for param in self.mizuroute_params):
                mizuroute_success = self._save_mizuroute_parameters_to_default(best_params)
                if mizuroute_success:
                    self.logger.info("✅ Successfully saved optimized mizuRoute parameters")
                else:
                    self.logger.warning("⚠️ Failed to save mizuRoute parameters to default settings")
                    success = False
            
            return success
                
        except Exception as e:
            self.logger.error(f"Error saving best parameters to default settings: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _save_mizuroute_parameters_to_default(self, best_params):
        """
        Save optimized mizuRoute parameters to default settings directory.
        
        Args:
            best_params: Dictionary with the best parameter values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.mizuroute_param_file.exists():
                self.logger.error(f"Default mizuRoute parameter file not found: {self.mizuroute_param_file}")
                return False
            
            # Backup existing file
            backup_file = self.mizuroute_settings_dir / f"param.nml.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.mizuroute_param_file, backup_file)
            self.logger.info(f"Backed up mizuRoute parameters to: {backup_file}")
            
            # Read and update the file
            with open(self.mizuroute_param_file, 'r') as f:
                content = f.read()
            
            # Update parameter values using regex replacement
            import re
            updated_content = content
            
            for param_name in self.mizuroute_params:
                if param_name in best_params:
                    param_value = best_params[param_name]
                    
                    # Create regex pattern to find and replace parameter
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                    
                    # Format the replacement value
                    if param_name in ['tscale']:
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        replacement = rf'\g<1>{param_value:.6f}'
                    
                    # Perform the replacement
                    new_content = re.sub(pattern, replacement, updated_content)
                    
                    if new_content != updated_content:
                        updated_content = new_content
                        self.logger.info(f"Updated default {param_name} = {param_value}")
                    else:
                        self.logger.warning(f"Could not find parameter {param_name} in default mizuRoute file")
            
            # Write updated file
            with open(self.mizuroute_param_file, 'w') as f:
                f.write(updated_content)
            
            self.logger.info(f"Updated default mizuRoute parameters in: {self.mizuroute_param_file}")
            
            # Also save a copy with experiment ID for tracking
            experiment_copy = self.mizuroute_settings_dir / f"param.nml.optimized_{self.experiment_id}"
            with open(experiment_copy, 'w') as f:
                f.write(updated_content)
            self.logger.info(f"Created experiment-specific mizuRoute copy: {experiment_copy}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving mizuRoute parameters to default settings: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
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
    
    def _get_optimization_period_string(self):
        """Get a string representation of the optimization period for logging."""
        spinup_start, spinup_end = self._parse_date_range(self.config.get('SPINUP_PERIOD', ''))
        calib_start, calib_end = self.calibration_period
        
        if spinup_start and calib_end:
            return f"{spinup_start.strftime('%Y-%m-%d')} to {calib_end.strftime('%Y-%m-%d')} (spinup + calibration)"
        elif calib_start and calib_end:
            return f"{calib_start.strftime('%Y-%m-%d')} to {calib_end.strftime('%Y-%m-%d')} (calibration only)"
        else:
            return "full experiment period (fallback)"
    
    def run_de_optimization(self):
        """
        Run the DE optimization algorithm with parameter persistence and depth calibration.
        
        Enhanced to:
        1. Check for existing optimized parameters and use them as starting point
        2. Save best parameters back to default settings when complete
        3. Handle soil depth calibration if enabled
        
        Returns:
            Dict: Dictionary with optimization results
        """
        self.logger.info("Starting DE optimization with parameter persistence")
        
        if self.calibrate_depth:
            self.logger.info("🌱 Soil depth calibration is ENABLED")
            self.logger.info("📏 Will optimize soil depth profile using shape method")
        
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
        
        # Step 5: Run the DE algorithm
        best_params, best_score, history = self._run_de_algorithm()
        
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
                if self.calibrate_depth:
                    self.logger.info("🌱 Optimized soil depths saved to coldState.nc")
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
            'saved_to_default': save_success if best_params else False,
            'depth_calibration_enabled': self.calibrate_depth
        }
        
        return results
    
    def _parse_parameter_bounds(self):
        """
        Parse parameter bounds from SUMMA parameter info files.
        Enhanced to include depth parameter bounds.
        
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
        
        # Add depth parameter bounds if depth calibration is enabled
        if self.calibrate_depth:
            # Get depth bounds from config or use defaults
            total_mult_bounds = self.config.get('DEPTH_TOTAL_MULT_BOUNDS', [0.1, 5.0])
            shape_factor_bounds = self.config.get('DEPTH_SHAPE_FACTOR_BOUNDS', [0.1, 3.0])
            
            bounds['total_mult'] = {'min': total_mult_bounds[0], 'max': total_mult_bounds[1]}
            bounds['shape_factor'] = {'min': shape_factor_bounds[0], 'max': shape_factor_bounds[1]}
            
            self.logger.info(f"Added depth parameter bounds: total_mult={total_mult_bounds}, shape_factor={shape_factor_bounds}")
        
        # Add mizuRoute parameter bounds if routing calibration is enabled
        if self.mizuroute_params:
            mizuroute_bounds = self._get_mizuroute_parameter_bounds()
            bounds.update(mizuroute_bounds)
            self.logger.info(f"Added mizuRoute parameter bounds for {len(mizuroute_bounds)} parameters")
        
        if not bounds:
            self.logger.error("No parameter bounds found")
            raise ValueError("No parameter bounds found")
        
        self.logger.info(f"Found bounds for {len(bounds)} parameters")
        return bounds
    
    def _get_mizuroute_parameter_bounds(self):
        """
        Define parameter bounds for mizuRoute parameters.
        
        Returns:
            Dict: Dictionary with mizuRoute parameter bounds
        """
        # Default bounds based on physical constraints and literature
        default_bounds = {
            'velo': {'min': 0.1, 'max': 5.0},        # Channel velocity [m/s]
            'diff': {'min': 100.0, 'max': 5000.0},   # Diffusion [m²/s] 
            'mann_n': {'min': 0.01, 'max': 0.1},     # Manning's roughness [-]
            'wscale': {'min': 0.0001, 'max': 0.01},  # Width scaling [-]
            'fshape': {'min': 1.0, 'max': 5.0},      # Shape parameter [-]
            'tscale': {'min': 3600, 'max': 172800}   # Time scale [s] (1h to 48h)
        }
        
        bounds = {}
        for param in self.mizuroute_params:
            if param in default_bounds:
                # Check if user provided custom bounds
                custom_bounds_key = f'MIZUROUTE_{param.upper()}_BOUNDS'
                if custom_bounds_key in self.config:
                    custom_bounds = self.config[custom_bounds_key]
                    bounds[param] = {'min': custom_bounds[0], 'max': custom_bounds[1]}
                    self.logger.info(f"Using custom bounds for {param}: {custom_bounds}")
                else:
                    bounds[param] = default_bounds[param]
                    self.logger.info(f"Using default bounds for {param}: {default_bounds[param]}")
            else:
                self.logger.warning(f"Unknown mizuRoute parameter: {param}. Available: {list(default_bounds.keys())}")
        
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
                                default_val_str = parts[1]
                                lower_val_str   = parts[2]
                                upper_val_str   = parts[3]
                                min_val = float(lower_val_str.replace('d','e').replace('D','e'))
                                max_val = float(upper_val_str.replace('d','e').replace('D','e'))

                                bounds[param_name] = {'min': min_val, 'max': max_val}
                                self.logger.debug(f"Found bounds for {param_name}: min={min_val}, max={max_val}")
                            except ValueError as ve:
                                self.logger.warning(f"Could not parse bounds for parameter '{param_name}' in line: {line}. Error: {str(ve)}")
        except Exception as e:
            self.logger.error(f"Error parsing parameter info file {file_path}: {str(e)}")
        
        return bounds
    
    def _denormalize_individual(self, normalized_individual):
        """
        Denormalize an individual's parameters from [0,1] range to original parameter ranges.
        Enhanced to handle depth parameters.
        
        Args:
            normalized_individual: Array of normalized parameter values for one individual
            
        Returns:
            Dict: Parameter dictionary with denormalized values
        """
        params = {}
        
        # Get parameter names and bounds
        param_names = self.param_names
        param_bounds = self.param_bounds
        
        if param_bounds is None:
            self.logger.error("Parameter bounds not initialized")
            return params
        
        for i, param_name in enumerate(param_names):
            if param_name in param_bounds:
                bounds = param_bounds[param_name]
                min_val = bounds['min']
                max_val = bounds['max']
                
                # Denormalize value
                denorm_value = min_val + normalized_individual[i] * (max_val - min_val)
                
                # Special handling for depth parameters
                if param_name in self.depth_params:
                    # Depth parameters are scalar values
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.basin_params:
                    # Basin/GRU level parameter - single value
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.mizuroute_params:
                    # mizuRoute parameters are scalar values
                    params[param_name] = denorm_value
                else:
                    # Local/HRU level parameter - may need multiple values
                    # For simplicity, use the same value for all HRUs
                    with xr.open_dataset(self.attr_file_path) as ds:
                        if 'hru' in ds.dims:
                            num_hrus = ds.sizes['hru']
                            params[param_name] = np.full(num_hrus, denorm_value)
                        else:
                            params[param_name] = np.array([denorm_value])
        
        return params
    
    def _evaluate_parameters(self, params):
        """
        Evaluate a parameter set by running SUMMA and calculating performance metrics.
        Enhanced to handle depth parameter updates and routing configurations.
        
        Args:
            params: Dictionary with parameter values
            
        Returns:
            float: Performance metric value, or None if evaluation failed
        """
        eval_start_time = datetime.now()
        
        try:
            self.logger.debug("Evaluating parameter set")
            
            # Update soil depths if depth calibration is enabled
            if self.calibrate_depth and 'total_mult' in params and 'shape_factor' in params:
                total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
                shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
                
                new_depths = self._calculate_new_depths(total_mult, shape_factor)
                if new_depths is not None:
                    success = self._update_soil_depths(new_depths)
                    if not success:
                        self.logger.debug("❌ Failed to update soil depths")
                        return None
                else:
                    self.logger.debug("❌ Failed to calculate new soil depths")
                    return None
            
            # Update mizuRoute parameters if routing calibration is enabled
            if self.mizuroute_params:
                mizuroute_success = self._update_mizuroute_parameters(params)
                if not mizuroute_success:
                    self.logger.debug("❌ Failed to update mizuRoute parameters")
                    return None
            
            # Generate trial parameters file (excluding depth and mizuRoute parameters)
            hydraulic_params = {k: v for k, v in params.items() 
                              if k not in self.depth_params and k not in self.mizuroute_params}
            trial_params_path = self._generate_trial_params_file(hydraulic_params)
            if not trial_params_path:
                self.logger.debug("❌ Failed to generate trial parameters file")
                return None
            
            # Run SUMMA
            summa_success = self._run_summa_simulation()
            if not summa_success:
                self.logger.debug("❌ SUMMA simulation failed")
                return None
            
            # Check if mizuRoute routing is needed based on configuration
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
            needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
            
            # Run mizuRoute if needed
            if needs_mizuroute:
                # Handle lumped-to-distributed conversion if needed
                if domain_method == 'lumped' and routing_delineation == 'river_network':
                    self.logger.debug("Converting lumped SUMMA output for distributed routing")
                    self._convert_lumped_to_distributed_routing()
                
                # Run mizuRoute
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

    @property
    def param_names(self):
        """Get list of all parameter names including depth and mizuRoute parameters if enabled."""
        return self.local_params + self.basin_params + self.depth_params + self.mizuroute_params
    
    def _get_param_bounds(self):
        """Get parameter bounds including depth parameters if enabled."""
        if not hasattr(self, '_param_bounds'):
            self._param_bounds = self._parse_parameter_bounds()
        return self._param_bounds
    
    def run_parameter_extraction(self):
        """
        Run a preliminary SUMMA simulation to extract parameter values.
        Enhanced to handle depth parameters.
        
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
            extracted_params = self._extract_parameters_from_results(extract_dir)
            
            # Add default depth parameters if depth calibration is enabled
            if self.calibrate_depth and extracted_params:
                extracted_params['total_mult'] = np.array([1.0])  # Default: no depth scaling
                extracted_params['shape_factor'] = np.array([1.0])  # Default: uniform scaling
                self.logger.info("Added default depth parameters to extraction results")
            
            # Add mizuRoute parameters if routing calibration is enabled
            if self.mizuroute_params and extracted_params:
                for param in self.mizuroute_params:
                    extracted_params[param] = self._get_default_mizuroute_value(param)
                self.logger.info("Added default mizuRoute parameters to extraction results")
            
            return extracted_params
        else:
            self.logger.warning("Parameter extraction run failed, will use default parameter ranges")
            # Return default parameters if extraction failed
            default_params = {}
            if self.calibrate_depth:
                default_params['total_mult'] = np.array([1.0])
                default_params['shape_factor'] = np.array([1.0])
            if self.mizuroute_params:
                for param in self.mizuroute_params:
                    default_params[param] = self._get_default_mizuroute_value(param)
            return default_params if default_params else None
    
    def _setup_optimization_environment(self):
        """
        Set up the optimization-specific directory structure and copy settings files.
        Enhanced to handle depth calibration files and mizuRoute settings.
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
            'coldState.nc',  # Important for depth calibration
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
        
        # Copy and update mizuRoute settings if they exist
        self._copy_mizuroute_settings()
        
        # Update mizuRoute control file for optimization runs
        if self._needs_mizuroute_routing(
            self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
            self.config.get('ROUTING_DELINEATION', 'lumped')
        ):
            self._update_mizuroute_control_for_optimization(is_final_run=False)
    
    def _initialize_optimization(self, initial_params, param_bounds):
        """
        Initialize optimization variables for DE.
        Enhanced to handle depth parameters.
        
        Args:
            initial_params: Dictionary with initial parameter values
            param_bounds: Dictionary with parameter bounds
        """
        self.logger.info("Initializing DE optimization")
        
        # Store parameter bounds and names
        self.param_bounds = param_bounds
        param_names = self.param_names  # Use the property
        num_params = len(param_names)
        
        self.logger.info(f"Initializing population of {self.population_size} individuals with {num_params} parameters")
        if self.calibrate_depth:
            self.logger.info(f"Including {len(self.depth_params)} depth parameters: {self.depth_params}")
        
        # Initialize population in normalized parameter space [0,1]
        self.population = np.random.random((self.population_size, num_params))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Initialize with any known good parameter values if available
        if initial_params:
            # Try to use initial values for the first individual
            for i, param_name in enumerate(param_names):
                if param_name in initial_params:
                    # Get bounds
                    bounds = param_bounds[param_name]
                    min_val = bounds['min']
                    max_val = bounds['max']
                    
                    # Get initial values
                    initial_values = initial_params[param_name]
                    
                    # Use mean value if multiple values (like for HRU-level parameters)
                    if isinstance(initial_values, np.ndarray) and len(initial_values) > 1:
                        initial_value = np.mean(initial_values)
                    else:
                        initial_value = initial_values[0] if isinstance(initial_values, np.ndarray) else initial_values
                    
                    # Normalize and store
                    normalized_value = (initial_value - min_val) / (max_val - min_val)
                    # Clip to valid range
                    normalized_value = np.clip(normalized_value, 0, 1)
                    
                    # Set the first individual to use these values
                    self.population[0, i] = normalized_value
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_population()
        
        # Find best individual
        best_idx = np.nanargmax(self.population_scores)
        self.best_score = self.population_scores[best_idx]
        self.best_params = self._denormalize_individual(self.population[best_idx])
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_score:.6f}")
        
        # Record initial state
        self._record_generation(0)
    
    def _evaluate_population(self):
        """Evaluate all individuals in the population."""
        for i in range(self.population_size):
            # Skip already evaluated individuals
            if not np.isnan(self.population_scores[i]):
                continue
                
            # Denormalize parameters for this individual
            params = self._denormalize_individual(self.population[i])
            
            # Evaluate this parameter set
            score = self._evaluate_parameters(params)
            
            # Store score
            self.population_scores[i] = score if score is not None else float('-inf')
            
            # Update best if better
            if self.population_scores[i] > self.best_score:
                self.best_score = self.population_scores[i]
                self.best_params = params.copy()
    
    def _run_de_algorithm(self):
        """
        Run the DE algorithm with enhanced logging.
        
        Returns:
            Tuple: (best_params, best_score, history)
        """
        self.logger.info("Running DE algorithm")
        self.logger.info("=" * 60)
        self.logger.info(f"Target metric: {self.target_metric} (higher is better)")
        self.logger.info(f"Total generations: {self.max_iterations}")
        self.logger.info(f"Population size: {self.population_size}")
        self.logger.info(f"DE parameters: F={self.F}, CR={self.CR}")
        if self.calibrate_depth:
            self.logger.info("🌱 Soil depth calibration: ENABLED")
        self.logger.info("=" * 60)
        
        # Track statistics
        total_improvements = 0
        no_improvement_streak = 0
        last_improvement_generation = 0
        
        # Main DE loop
        for generation in range(1, self.max_iterations + 1):
            generation_start_time = datetime.now()
            
            # Create new trial population
            trial_population = np.zeros_like(self.population)
            
            # Generate trial vectors for each individual
            for i in range(self.population_size):
                trial_population[i] = self._create_trial_vector(i)
            
            # Evaluate trial population
            trial_scores = np.full(self.population_size, np.nan)
            generation_improvements = 0
            
            for i in range(self.population_size):
                # Denormalize trial parameters
                trial_params = self._denormalize_individual(trial_population[i])
                
                # Evaluate trial individual
                trial_score = self._evaluate_parameters(trial_params)
                trial_scores[i] = trial_score if trial_score is not None else float('-inf')
                
                # Selection: keep trial if better than parent
                if trial_scores[i] > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    generation_improvements += 1
                    
                    # Update global best if this is the new best
                    if trial_scores[i] > self.best_score:
                        old_best = self.best_score
                        self.best_score = trial_scores[i]
                        self.best_params = trial_params.copy()
                        
                        # Log the new best with depth info if applicable
                        log_msg = f"Gen {generation:3d}: Individual {i:2d} ⭐ NEW GLOBAL BEST! {self.target_metric}={self.best_score:.6f}"
                        if self.calibrate_depth and 'total_mult' in trial_params and 'shape_factor' in trial_params:
                            tm = trial_params['total_mult'][0] if isinstance(trial_params['total_mult'], np.ndarray) else trial_params['total_mult']
                            sf = trial_params['shape_factor'][0] if isinstance(trial_params['shape_factor'], np.ndarray) else trial_params['shape_factor']
                            log_msg += f" [depth: tm={tm:.3f}, sf={sf:.3f}]"
                        
                        if old_best > float('-inf'):
                            improvement = self.best_score - old_best
                            improvement_pct = (improvement / abs(old_best)) * 100 if old_best != 0 else float('inf')
                            log_msg += f" (+{improvement:.6f}, +{improvement_pct:.2f}%)"
                        else:
                            log_msg += " (first valid score)"
                        
                        self.logger.info(log_msg)
            
            # Track improvements
            if generation_improvements > 0:
                total_improvements += generation_improvements
                no_improvement_streak = 0
                last_improvement_generation = generation
            else:
                no_improvement_streak += 1
            
            # Calculate generation statistics
            generation_duration = datetime.now() - generation_start_time
            valid_scores = self.population_scores[self.population_scores != float('-inf')]
            
            if len(valid_scores) > 0:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                worst_score = np.min(valid_scores)
            else:
                mean_score = std_score = worst_score = float('-inf')
            
            # Log generation summary
            log_msg = f"Gen {generation:3d}/{self.max_iterations}: "
            log_msg += f"Best={self.best_score:.6f}, Mean={mean_score:.6f}, "
            log_msg += f"Improvements={generation_improvements}/{self.population_size} "
            log_msg += f"[{generation_duration.total_seconds():.1f}s]"
            
            self.logger.info(log_msg)
            
            # Record generation
            self._record_generation(generation)
            
            # Progress summary every 10 generations
            if generation % 10 == 0 or generation == self.max_iterations:
                self._log_progress_summary(generation, total_improvements, no_improvement_streak, 
                                         last_improvement_generation, mean_score, std_score)
            
            # Early termination warning
            if no_improvement_streak >= 15 and generation > 20:
                self.logger.warning(f"No improvements for {no_improvement_streak} generations. "
                                  f"Population may have converged.")
        
        # Final summary
        self._log_final_summary(total_improvements)
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _create_trial_vector(self, target_index):
        """
        Create a trial vector using DE/rand/1/bin strategy.
        
        Args:
            target_index: Index of the target individual
            
        Returns:
            ndarray: Trial vector
        """
        # Select three random individuals different from target
        candidates = list(range(self.population_size))
        candidates.remove(target_index)
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        # Mutation: V = X_r1 + F * (X_r2 - X_r3)
        mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        
        # Ensure mutant is within bounds [0, 1]
        mutant = np.clip(mutant, 0, 1)
        
        # Crossover: create trial vector
        trial = self.population[target_index].copy()
        
        # Random crossover index (ensures at least one parameter is from mutant)
        j_rand = np.random.randint(len(self.param_names))
        
        # Binomial crossover
        for j in range(len(self.param_names)):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def _record_generation(self, generation):
        """Record generation statistics."""
        valid_scores = self.population_scores[self.population_scores != float('-inf')]
        
        generation_stats = {
            'generation': generation,
            'best_score': self.best_score if self.best_score != float('-inf') else None,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores),
            'population_scores': self.population_scores.copy()
        }
        
        self.iteration_history.append(generation_stats)
    
    def _log_progress_summary(self, generation, total_improvements, no_improvement_streak, 
                            last_improvement_generation, mean_score, std_score):
        """Log progress summary with statistics."""
        progress_pct = (generation / self.max_iterations) * 100
        
        summary_msg = f"Progress: {progress_pct:.1f}% ({generation}/{self.max_iterations})"
        
        if self.best_score > float('-inf'):
            summary_msg += f" | Best: {self.best_score:.6f}"
            summary_msg += f" | Mean: {mean_score:.6f} (±{std_score:.6f})" if mean_score != float('-inf') else ""
            summary_msg += f" | Total improvements: {total_improvements}"
            
            if no_improvement_streak > 0:
                summary_msg += f" | Stagnant: {no_improvement_streak} gens"
            
            if last_improvement_generation > 0:
                summary_msg += f" | Last improvement: gen {last_improvement_generation}"
        else:
            summary_msg += " | No valid solutions yet"
        
        self.logger.info(summary_msg)
    
    def _log_final_summary(self, total_improvements):
        """Log final optimization summary."""
        self.logger.info("=" * 60)
        self.logger.info("DE OPTIMIZATION COMPLETED")
        self.logger.info("=" * 60)
        
        if self.best_score > float('-inf'):
            self.logger.info(f"🏆 Best {self.target_metric}: {self.best_score:.6f}")
            
            # Log best depth parameters if depth calibration was used
            if self.calibrate_depth and self.best_params:
                if 'total_mult' in self.best_params and 'shape_factor' in self.best_params:
                    tm = self.best_params['total_mult'][0] if isinstance(self.best_params['total_mult'], np.ndarray) else self.best_params['total_mult']
                    sf = self.best_params['shape_factor'][0] if isinstance(self.best_params['shape_factor'], np.ndarray) else self.best_params['shape_factor']
                    self.logger.info(f"🌱 Best depth parameters: total_mult={tm:.3f}, shape_factor={sf:.3f}")
                    
                    # Calculate and log depth changes
                    if self.original_depths is not None:
                        new_depths = self._calculate_new_depths(tm, sf)
                        if new_depths is not None:
                            total_orig = np.sum(self.original_depths)
                            total_new = np.sum(new_depths)
                            depth_change_pct = ((total_new - total_orig) / total_orig) * 100
                            self.logger.info(f"📏 Total depth change: {depth_change_pct:+.1f}% ({total_orig:.2f}m → {total_new:.2f}m)")
            
            total_evaluations = self.max_iterations * self.population_size
            self.logger.info(f"📊 Total improvements: {total_improvements}/{total_evaluations} "
                           f"({total_improvements/total_evaluations*100:.1f}%)")
            
            if total_improvements > 0:
                avg_evals_per_improvement = total_evaluations / total_improvements
                self.logger.info(f"⚡ Search efficiency: {avg_evals_per_improvement:.1f} evaluations per improvement")
            
            # Population diversity
            valid_scores = self.population_scores[self.population_scores != float('-inf')]
            if len(valid_scores) > 1:
                diversity = np.std(valid_scores) / np.mean(valid_scores) * 100 if np.mean(valid_scores) != 0 else 0
                self.logger.info(f"🎯 Final population diversity: {diversity:.1f}%")
                
        else:
            self.logger.warning("❌ No valid solutions found during optimization!")
            self.logger.warning("Check model configuration, parameter bounds, and forcing data.")
        
        self.logger.info("=" * 60)
    
    def _update_file_manager_for_optimization(self, is_final_run=False):
        """
        Update the fileManager.txt for DE optimization with proper file prefixes.
        
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
            run_prefix = f"run_de_final_{self.experiment_id}"
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
            
            run_prefix = f"run_de_opt_{self.experiment_id}"
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
                # Point to DE optimization SUMMA simulation directory
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                # Point to DE optimization settings directory  
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

    def _generate_trial_params_file(self, params):
        """
        Generate a trialParams.nc file with the given parameters in the optimization settings directory.
        
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
                    output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE DE Optimizer"
                    output_ds.history = f"Created on {datetime.now().isoformat()}"
                    output_ds.confluence_experiment_id = f"DE_optimization_{self.experiment_id}"
                
                self.logger.debug(f"Trial parameters file generated: {trial_params_path}")
                return trial_params_path
                
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

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
        log_file = self.summa_sim_dir / "logs" / f"summa_de_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        Enhanced with detailed debugging.
        
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
        
        # Enhanced debugging
        self.logger.debug(f"mizuRoute executable: {mizu_exe}")
        self.logger.debug(f"mizuRoute control file: {control_file}")
        self.logger.debug(f"Executable exists: {mizu_exe.exists()}")
        self.logger.debug(f"Control file exists: {control_file.exists()}")
        
        if not mizu_exe.exists():
            self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
            return False
        
        if not control_file.exists():
            self.logger.error(f"Optimization mizuRoute control file not found: {control_file}")
            return False
        
        # Check if input files exist
        expected_input_file = self.summa_sim_dir / f"run_de_opt_{self.experiment_id}_timestep.nc"
        self.logger.debug(f"Expected input file: {expected_input_file}")
        self.logger.debug(f"Input file exists: {expected_input_file.exists()}")
        
        if not expected_input_file.exists():
            self.logger.error(f"Expected SUMMA input file not found: {expected_input_file}")
            # List what files ARE in the directory
            summa_files = list(self.summa_sim_dir.glob("*.nc"))
            self.logger.error(f"Available SUMMA files: {[f.name for f in summa_files]}")
            return False
        
        # Make sure output directory exists
        self.mizuroute_sim_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"mizuRoute output directory: {self.mizuroute_sim_dir}")
        
        # Create command
        mizu_command = f"{mizu_exe} {control_file}"
        self.logger.info(f"Running mizuRoute command: {mizu_command}")
        
        # Create log file in optimization logs directory
        log_file = self.mizuroute_sim_dir / "logs" / f"mizuroute_de_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Starting mizuRoute simulation, logs: {log_file}")
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command, 
                    shell=True, 
                    stdout=f, 
                    stderr=subprocess.STDOUT, 
                    check=True,
                    cwd=str(control_file.parent)  # Run from control file directory
                )
            
            # Check if output files were created
            output_files = list(self.mizuroute_sim_dir.glob("*.nc"))
            self.logger.info(f"mizuRoute simulation completed. Created {len(output_files)} output files:")
            for f in output_files:
                self.logger.info(f"  - {f.name}")
            
            if len(output_files) == 0:
                self.logger.error("mizuRoute completed but no output files found!")
                # Read and log the mizuRoute log file for debugging
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                    self.logger.error(f"mizuRoute log content:\n{log_content}")
                except Exception as e:
                    self.logger.error(f"Could not read mizuRoute log: {str(e)}")
                return False
            
            self.logger.debug("mizuRoute simulation completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"mizuRoute simulation failed with exit code {e.returncode}")
            # Read and log the error details
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                self.logger.error(f"mizuRoute error log:\n{log_content}")
            except Exception as read_e:
                self.logger.error(f"Could not read mizuRoute log: {str(read_e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error during mizuRoute simulation: {str(e)}")
            return False

    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics by comparing simulated to observed streamflow.
        Enhanced debugging version with support for both mizuRoute and direct SUMMA output.
        """
        self.logger.debug("=== PERFORMANCE METRICS DEBUG ===")
        self.logger.debug("Calculating performance metrics from DE optimization simulation")
        
        # Get observed data path
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        self.logger.debug(f"Looking for observed data at: {obs_path}")
        self.logger.debug(f"Observed data exists: {obs_path.exists()}")
        
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            self.logger.debug(f"Observed data shape: {obs_df.shape}")
            self.logger.debug(f"Observed data columns: {obs_df.columns.tolist()}")
            
            # Identify date and flow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            self.logger.debug(f"Date column: {date_col}")
            self.logger.debug(f"Flow column: {flow_col}")
            
            if not date_col or not flow_col:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            self.logger.debug(f"Observed flow time range: {observed_flow.index.min()} to {observed_flow.index.max()}")
            self.logger.debug(f"Observed flow data points: {len(observed_flow)}")
            self.logger.debug(f"Observed flow stats: min={observed_flow.min():.2f}, max={observed_flow.max():.2f}, mean={observed_flow.mean():.2f}")
            
            # Determine which output to use based on routing configuration
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
            needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
            
            self.logger.debug(f"Domain method: {domain_method}")
            self.logger.debug(f"Routing delineation: {routing_delineation}")
            self.logger.debug(f"Needs mizuRoute: {needs_mizuroute}")
            
            if needs_mizuroute:
                # Use mizuRoute output - USE DE OPTIMIZATION DIRECTORY
                sim_dir = self.mizuroute_sim_dir  # Points to run_de/mizuRoute
                sim_files = list(sim_dir.glob("*.nc"))
                
                self.logger.debug(f"Looking for mizuRoute files in: {sim_dir}")
                self.logger.debug(f"Found {len(sim_files)} mizuRoute files: {[f.name for f in sim_files]}")
                
                if not sim_files:
                    self.logger.error(f"No mizuRoute output files found in DE optimization directory: {sim_dir}")
                    return None
                
                sim_file = sim_files[0]  # Take the first file
                self.logger.debug(f"Using mizuRoute file: {sim_file}")
                self.logger.debug(f"File size: {sim_file.stat().st_size} bytes")
                
                # Open the file with xarray and examine its structure
                try:
                    with xr.open_dataset(sim_file) as ds:
                        self.logger.debug(f"mizuRoute output variables: {list(ds.variables.keys())}")
                        self.logger.debug(f"mizuRoute output dimensions: {dict(ds.sizes)}")
                        
                        # Get reach ID
                        sim_reach_id = self.config.get('SIM_REACH_ID')
                        self.logger.debug(f"Looking for reach ID: {sim_reach_id}")
                        
                        # Check if reach variables exist
                        if 'reachID' in ds.variables:
                            reach_ids = ds['reachID'].values
                            self.logger.debug(f"Available reach IDs: {reach_ids}")
                            
                            reach_indices = np.where(reach_ids == int(sim_reach_id))[0]
                            self.logger.debug(f"Matching reach indices: {reach_indices}")
                            
                            if len(reach_indices) > 0:
                                reach_index = reach_indices[0]
                                self.logger.debug(f"Using reach index: {reach_index}")
                                
                                # Try to find streamflow variable
                                streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']
                                found_var = None
                                
                                for var_name in streamflow_vars:
                                    if var_name in ds.variables:
                                        found_var = var_name
                                        self.logger.debug(f"Found streamflow variable: {var_name}")
                                        
                                        # Get the variable info
                                        var = ds[var_name]
                                        self.logger.debug(f"Variable {var_name} dimensions: {var.dims}")
                                        self.logger.debug(f"Variable {var_name} shape: {var.shape}")
                                        
                                        # Extract simulated flow
                                        if 'seg' in var.dims:
                                            simulated_flow = var.isel(seg=reach_index).to_pandas()
                                        elif 'reachID' in var.dims:
                                            simulated_flow = var.isel(reachID=reach_index).to_pandas()
                                        else:
                                            self.logger.error(f"Cannot determine how to index variable {var_name} with dimensions {var.dims}")
                                            continue
                                        
                                        self.logger.debug(f"Extracted simulated flow shape: {simulated_flow.shape}")
                                        self.logger.debug(f"Simulated flow time range: {simulated_flow.index.min()} to {simulated_flow.index.max()}")
                                        self.logger.debug(f"Simulated flow stats: min={simulated_flow.min():.2f}, max={simulated_flow.max():.2f}, mean={simulated_flow.mean():.2f}")
                                        break
                                
                                if found_var is None:
                                    self.logger.error("Could not find streamflow variable in mizuRoute output")
                                    self.logger.error(f"Available variables: {list(ds.variables.keys())}")
                                    return None
                            else:
                                self.logger.error(f"Reach ID {sim_reach_id} not found in mizuRoute output")
                                return None
                        else:
                            self.logger.error("No reachID variable found in mizuRoute output")
                            self.logger.error(f"Available variables: {list(ds.variables.keys())}")
                            return None
                        
                except Exception as e:
                    self.logger.error(f"Error reading mizuRoute file {sim_file}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return None
            
            else:
                # Use SUMMA output directly for lumped simulations with lumped routing
                sim_dir = self.summa_sim_dir  # Points to run_de/SUMMA
                sim_files = list(sim_dir.glob("*.nc"))
                
                self.logger.debug(f"Looking for SUMMA files in: {sim_dir}")
                self.logger.debug(f"Found {len(sim_files)} SUMMA files: {[f.name for f in sim_files]}")
                
                if not sim_files:
                    self.logger.error(f"No SUMMA output files found in DE optimization directory: {sim_dir}")
                    return None
                
                # Look for timestep file (contains the averageRoutedRunoff we need)
                timestep_files = [f for f in sim_files if 'timestep' in f.name]
                if not timestep_files:
                    self.logger.error("No SUMMA timestep files found for lumped routing")
                    self.logger.error(f"Available SUMMA files: {[f.name for f in sim_files]}")
                    return None
                
                sim_file = timestep_files[0]  # Take the first timestep file
                self.logger.debug(f"Using SUMMA file: {sim_file}")
                self.logger.debug(f"File size: {sim_file.stat().st_size} bytes")
                
                # Get catchment area for unit conversion (m/s -> m³/s)
                catchment_area = self._get_catchment_area()
                if catchment_area is None:
                    self.logger.error("Could not determine catchment area for unit conversion")
                    return None
                
                self.logger.info(f"Using catchment area: {catchment_area:.2f} m² ({catchment_area/1e6:.2f} km²)")
                
                # Open the file and extract streamflow
                try:
                    with xr.open_dataset(sim_file) as ds:
                        self.logger.debug(f"SUMMA output variables: {list(ds.variables.keys())}")
                        self.logger.debug(f"SUMMA output dimensions: {dict(ds.sizes)}")
                        
                        # Try to find streamflow variable in SUMMA output
                        streamflow_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'averageRoutedRunoff_mean']
                        found_var = None
                        
                        for var_name in streamflow_vars:
                            if var_name in ds.variables:
                                found_var = var_name
                                self.logger.debug(f"Found streamflow variable: {var_name}")
                                
                                # Get the variable info
                                var = ds[var_name]
                                self.logger.debug(f"Variable {var_name} dimensions: {var.dims}")
                                self.logger.debug(f"Variable {var_name} shape: {var.shape}")
                                
                                # Extract simulated flow - for lumped case, should be single time series
                                if len(var.shape) > 1:
                                    # Multi-dimensional - take first spatial index (should be only one for lumped)
                                    if 'hru' in var.dims:
                                        simulated_flow = var.isel(hru=0).to_pandas()
                                    elif 'gru' in var.dims:
                                        simulated_flow = var.isel(gru=0).to_pandas()
                                    else:
                                        # Take first index of non-time dimension
                                        non_time_dims = [dim for dim in var.dims if dim != 'time']
                                        if non_time_dims:
                                            simulated_flow = var.isel({non_time_dims[0]: 0}).to_pandas()
                                        else:
                                            simulated_flow = var.to_pandas()
                                else:
                                    # Already 1D time series
                                    simulated_flow = var.to_pandas()
                                
                                # Convert from m/s to m³/s by multiplying by catchment area
                                simulated_flow_original = simulated_flow.copy()
                                simulated_flow = simulated_flow * catchment_area
                                
                                self.logger.debug(f"Unit conversion: {var_name} from m/s to m³/s")
                                self.logger.debug(f"Original flow stats (m/s): min={simulated_flow_original.min():.6f}, max={simulated_flow_original.max():.6f}, mean={simulated_flow_original.mean():.6f}")
                                self.logger.debug(f"Converted flow stats (m³/s): min={simulated_flow.min():.2f}, max={simulated_flow.max():.2f}, mean={simulated_flow.mean():.2f}")
                                
                                self.logger.debug(f"Extracted simulated flow shape: {simulated_flow.shape}")
                                self.logger.debug(f"Simulated flow time range: {simulated_flow.index.min()} to {simulated_flow.index.max()}")
                                break
                        
                        if found_var is None:
                            self.logger.error("Could not find streamflow variable in SUMMA output")
                            self.logger.error(f"Available variables: {list(ds.variables.keys())}")
                            return None
                            
                except Exception as e:
                    self.logger.error(f"Error reading SUMMA file {sim_file}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return None
            
            # Check if we have simulated_flow
            if 'simulated_flow' not in locals():
                self.logger.error("Failed to extract simulated flow data")
                return None
            
            # Round time index to nearest hour for better matching
            simulated_flow.index = simulated_flow.index.round('h')
            self.logger.info(f'simulated flow: {simulated_flow}')

            # Check time overlap
            common_idx = observed_flow.index.intersection(simulated_flow.index)
            self.logger.debug(f"Common time indices: {len(common_idx)}")
            
            if len(common_idx) == 0:
                self.logger.error("No common time indices between observed and simulated data!")
                self.logger.error(f"Observed time range: {observed_flow.index.min()} to {observed_flow.index.max()}")
                self.logger.error(f"Simulated time range: {simulated_flow.index.min()} to {simulated_flow.index.max()}")
                return None
            
            self.logger.debug(f"Found {len(common_idx)} common time steps for metrics calculation")
            
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

    def _update_mizuroute_parameters(self, params):
        """
        Update mizuRoute parameter file with new parameter values.
        
        Args:
            params: Dictionary with parameter values including mizuRoute parameters
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Path to mizuRoute parameter file in optimization settings
            mizuroute_settings_dir = self.optimization_dir / "settings" / "mizuRoute"
            param_file = mizuroute_settings_dir / "param.nml.default"
            
            if not param_file.exists():
                self.logger.error(f"mizuRoute parameter file not found: {param_file}")
                return False
            
            # Read the original parameter file
            with open(param_file, 'r') as f:
                content = f.read()
            
            # Update parameter values using regex replacement
            import re
            updated_content = content
            
            for param_name in self.mizuroute_params:
                if param_name in params:
                    param_value = params[param_name]
                    
                    # Create regex pattern to find and replace parameter
                    # Pattern matches: whitespace + param_name + whitespace + = + whitespace + number
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                    
                    # Format the replacement value
                    if param_name in ['tscale']:
                        # Integer parameters
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        # Float parameters with appropriate precision
                        replacement = rf'\g<1>{param_value:.6f}'
                    
                    # Perform the replacement
                    new_content = re.sub(pattern, replacement, updated_content)
                    
                    if new_content != updated_content:
                        updated_content = new_content
                        self.logger.debug(f"Updated {param_name} = {param_value}")
                    else:
                        self.logger.warning(f"Could not find parameter {param_name} in mizuRoute file")
            
            # Write updated parameter file
            with open(param_file, 'w') as f:
                f.write(updated_content)
            
            self.logger.debug(f"Updated mizuRoute parameters in {param_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating mizuRoute parameters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _update_mizuroute_control_for_optimization(self, is_final_run=False):
        """
        Update the mizuRoute control file for DE optimization with proper paths and filenames.
        
        Args:
            is_final_run (bool): If True, runs full experiment period for final evaluation.
                            If False, runs only spinup + calibration period for optimization.
        """
        mizuroute_settings_dir = self.optimization_dir / "settings" / "mizuRoute"
        control_file_path = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not control_file_path.exists():
            self.logger.warning("mizuRoute control file not found in optimization settings")
            return
        
        # Parse time periods from config
        spinup_start, spinup_end = self._parse_date_range(self.config.get('SPINUP_PERIOD', ''))
        calib_start, calib_end = self.calibration_period
        experiment_start = self.config.get('EXPERIMENT_TIME_START', '')
        experiment_end = self.config.get('EXPERIMENT_TIME_END', '')
        
        # Determine simulation period and filename prefix based on run type
        if is_final_run:
            # Final run: use full experiment period
            sim_start = experiment_start
            sim_end = experiment_end
            run_prefix = f"run_de_final_{self.experiment_id}"
            filename_prefix = f"run_de_final_{self.experiment_id}"
            self.logger.info("Setting up mizuRoute for final run with full experiment period")
        else:
            # Optimization runs: use spinup start to calibration end
            if spinup_start and calib_end:
                sim_start = spinup_start.strftime('%Y-%m-%d %H:%M')
                sim_end = calib_end.strftime('%Y-%m-%d %H:%M')
            elif calib_start and calib_end:
                # Fallback: just use calibration period if no spinup defined
                sim_start = calib_start.strftime('%Y-%m-%d %H:%M')
                sim_end = calib_end.strftime('%Y-%m-%d %H:%M')
            else:
                # Ultimate fallback: use experiment period
                sim_start = experiment_start
                sim_end = experiment_end
                self.logger.warning("Could not parse spinup/calibration periods for mizuRoute, using full experiment period")
            
            run_prefix = f"run_de_opt_{self.experiment_id}"
            filename_prefix = f"run_de_opt_{self.experiment_id}"
            self.logger.info(f"Setting up mizuRoute optimization run from {sim_start} to {sim_end}")
        
        # Read the original control file
        with open(control_file_path, 'r') as f:
            lines = f.readlines()
        
        # Update paths and settings in control file
        updated_lines = []
        for line in lines:
            if '<input_dir>' in line:
                # Point to DE optimization SUMMA simulation directory
                input_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"<input_dir>             {input_path}/    ! Folder that contains runoff data from SUMMA \n")
            elif '<output_dir>' in line:
                # Point to DE optimization mizuRoute simulation directory
                output_path = str(self.mizuroute_sim_dir).replace('\\', '/')
                updated_lines.append(f"<output_dir>            {output_path}/    ! Folder that will contain mizuRoute simulations \n")
            elif '<ancil_dir>' in line:
                # Point to DE optimization mizuRoute settings directory
                ancil_path = str(mizuroute_settings_dir).replace('\\', '/')
                updated_lines.append(f"<ancil_dir>             {ancil_path}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
            elif '<case_name>' in line:
                # Update case name for optimization runs
                updated_lines.append(f"<case_name>             {run_prefix}    ! Simulation case name. This used for output netCDF, and restart netCDF name \n")
            elif '<fname_qsim>' in line:
                # Update input filename to match SUMMA output from optimization
                updated_lines.append(f"<fname_qsim>            {filename_prefix}_timestep.nc    ! netCDF name for HM_HRU runoff \n")
            elif '<sim_start>' in line:
                # Update simulation start time
                updated_lines.append(f"<sim_start>             {sim_start}    ! Time of simulation start. format: yyyy-mm-dd or yyyy-mm-dd hh:mm:ss \n")
            elif '<sim_end>' in line:
                # Update simulation end time
                updated_lines.append(f"<sim_end>               {sim_end}    ! Time of simulation end. format: yyyy-mm-dd or yyyy-mm-dd hh:mm:ss \n")
            else:
                # Keep all other lines unchanged
                updated_lines.append(line)
        
        # Write updated control file
        with open(control_file_path, 'w') as f:
            f.writelines(updated_lines)
        
        period_type = "final evaluation" if is_final_run else "optimization"
        self.logger.info(f"Updated mizuRoute control file for {period_type} with period {sim_start} to {sim_end}")
        self.logger.info(f"mizuRoute will read from: {self.summa_sim_dir}")
        self.logger.info(f"mizuRoute will write to: {self.mizuroute_sim_dir}")
        self.logger.info(f"Input filename: {filename_prefix}_timestep.nc")

    def _load_existing_mizuroute_parameters(self):
        """
        Load existing mizuRoute parameters from parameter file.
        
        Returns:
            Dict: Dictionary with existing mizuRoute parameter values, or None if not found
        """
        try:
            if not self.mizuroute_param_file.exists():
                return None
            
            param_values = {}
            
            with open(self.mizuroute_param_file, 'r') as f:
                content = f.read()
                
                for param_name in self.mizuroute_params:
                    # Look for parameter in file using regex
                    import re
                    pattern = rf'{param_name}\s*=\s*([0-9.-]+)'
                    match = re.search(pattern, content)
                    
                    if match:
                        param_values[param_name] = float(match.group(1))
                        self.logger.debug(f"Found existing {param_name}: {param_values[param_name]}")
            
            return param_values if param_values else None
            
        except Exception as e:
            self.logger.error(f"Error loading existing mizuRoute parameters: {str(e)}")
            return None

    def _get_default_mizuroute_value(self, param_name):
        """
        Get default value for a mizuRoute parameter.
        
        Args:
            param_name: Name of the mizuRoute parameter
            
        Returns:
            float: Default parameter value
        """
        defaults = {
            'velo': 1.0,
            'diff': 1000.0,
            'mann_n': 0.025,
            'wscale': 0.001,
            'fshape': 2.5,
            'tscale': 86400
        }
        
        return defaults.get(param_name, 1.0)

    def _run_final_simulation(self, best_params):
        """
        Run a final simulation with the best parameters using the full experiment period.
        Enhanced to handle routing configurations.
        
        Args:
            best_params: Dictionary with best parameter values
            
        Returns:
            Dict: Dictionary with final simulation results
        """
        self.logger.info("Running final simulation with best parameters over full experiment period")
        
        # Update file manager for final run (full experiment period)
        self._update_file_manager_for_optimization(is_final_run=True)
        
        # Update soil depths if depth calibration was used
        if self.calibrate_depth and 'total_mult' in best_params and 'shape_factor' in best_params:
            total_mult = best_params['total_mult'][0] if isinstance(best_params['total_mult'], np.ndarray) else best_params['total_mult']
            shape_factor = best_params['shape_factor'][0] if isinstance(best_params['shape_factor'], np.ndarray) else best_params['shape_factor']
            
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is not None:
                success = self._update_soil_depths(new_depths)
                if not success:
                    self.logger.warning("Could not update soil depths for final run")
            else:
                self.logger.warning("Could not calculate new soil depths for final run")
        
        # Update mizuRoute parameters if routing calibration was used
        if self.mizuroute_params and any(param in best_params for param in self.mizuroute_params):
            mizuroute_success = self._update_mizuroute_parameters(best_params)
            if not mizuroute_success:
                self.logger.warning("Could not update mizuRoute parameters for final run")
        
        # Generate trial parameters file with best parameters (excluding depth and mizuRoute parameters)
        hydraulic_params = {k: v for k, v in best_params.items() 
                          if k not in self.depth_params and k not in self.mizuroute_params}
        trial_params_path = self._generate_trial_params_file(hydraulic_params)
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
                # Single value (probably basin parameter or depth parameter)
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
        
        # Check if mizuRoute routing is needed
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
        needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
        
        # Run mizuRoute if needed
        mizuroute_success = True
        if needs_mizuroute:
            # Handle lumped-to-distributed conversion if needed
            if domain_method == 'lumped' and routing_delineation == 'river_network':
                self.logger.info("Converting lumped SUMMA output for distributed routing in final run")
                self._convert_lumped_to_distributed_routing()
            
            # Run mizuRoute
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
            
            # Extract data from history
            generations = [h['generation'] for h in history]
            best_scores = [h['best_score'] for h in history if h['best_score'] is not None]
            
            if not best_scores:
                self.logger.warning("No valid scores found in history - skipping plots")
                return
            
            # Plot optimization progress
            plt.figure(figsize=(12, 6))
            plt.plot(generations[:len(best_scores)], best_scores, 'b-o')
            plt.xlabel('Generation')
            plt.ylabel(f'Performance Metric ({self.target_metric})')
            title = 'DE Optimization Progress'
            if self.calibrate_depth:
                title += ' (with Soil Depth Calibration)'
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Add best generation marker
            best_idx = np.nanargmax(best_scores)
            plt.plot(generations[best_idx], best_scores[best_idx], 'ro', markersize=10, 
                    label=f'Best: {best_scores[best_idx]:.4f} at generation {generations[best_idx]}')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create depth parameter evolution plot if depth calibration was used
            if self.calibrate_depth:
                self._create_depth_parameter_plots(history, plots_dir)
            
            self.logger.info("Optimization plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _create_depth_parameter_plots(self, history, plots_dir):
        """
        Create plots showing depth parameter evolution during optimization.
        
        Args:
            history: List of dictionaries with optimization history
            plots_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract depth parameters from history
            generations = []
            total_mults = []
            shape_factors = []
            
            for h in history:
                if h['best_params'] and 'total_mult' in h['best_params'] and 'shape_factor' in h['best_params']:
                    generations.append(h['generation'])
                    
                    tm = h['best_params']['total_mult']
                    sf = h['best_params']['shape_factor']
                    
                    # Handle array vs scalar values
                    tm_val = tm[0] if isinstance(tm, np.ndarray) and len(tm) > 0 else tm
                    sf_val = sf[0] if isinstance(sf, np.ndarray) and len(sf) > 0 else sf
                    
                    total_mults.append(tm_val)
                    shape_factors.append(sf_val)
            
            if not generations:
                self.logger.warning("No depth parameter data found in history")
                return
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot total multiplier evolution
            ax1.plot(generations, total_mults, 'g-o', markersize=4)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Total Depth Multiplier')
            ax1.set_title('Soil Depth Total Multiplier Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No change (1.0)')
            ax1.legend()
            
            # Plot shape factor evolution
            ax2.plot(generations, shape_factors, 'm-o', markersize=4)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Shape Factor')
            ax2.set_title('Soil Depth Shape Factor Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Uniform scaling (1.0)')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "depth_parameter_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create depth profile comparison plot
            if self.original_depths is not None and len(total_mults) > 0 and len(shape_factors) > 0:
                self._create_depth_profile_comparison_plot(
                    total_mults[-1], shape_factors[-1], plots_dir
                )
            
        except Exception as e:
            self.logger.error(f"Error creating depth parameter plots: {str(e)}")

    def _create_depth_profile_comparison_plot(self, final_total_mult, final_shape_factor, plots_dir):
        """
        Create a plot comparing original and optimized soil depth profiles.
        
        Args:
            final_total_mult: Final optimized total multiplier
            final_shape_factor: Final optimized shape factor
            plots_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            
            # Calculate new depths
            new_depths = self._calculate_new_depths(final_total_mult, final_shape_factor)
            
            if new_depths is None:
                self.logger.warning("Could not calculate new depths for profile plot")
                return
            
            # Calculate cumulative depths for plotting
            original_cumulative = np.cumsum(np.concatenate([[0], self.original_depths]))
            new_cumulative = np.cumsum(np.concatenate([[0], new_depths]))
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
            
            # Layer thickness comparison
            layer_indices = np.arange(1, len(self.original_depths) + 1)
            
            ax1.barh(layer_indices - 0.2, self.original_depths, height=0.4, 
                    label='Original', color='blue', alpha=0.7)
            ax1.barh(layer_indices + 0.2, new_depths, height=0.4, 
                    label='Optimized', color='red', alpha=0.7)
            
            ax1.set_xlabel('Layer Thickness (m)')
            ax1.set_ylabel('Soil Layer')
            ax1.set_title('Soil Layer Thickness Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()  # Top layer at top
            
            # Cumulative depth profile
            ax2.step(original_cumulative, np.arange(len(original_cumulative)), 
                    where='post', label='Original', color='blue', linewidth=2)
            ax2.step(new_cumulative, np.arange(len(new_cumulative)), 
                    where='post', label='Optimized', color='red', linewidth=2)
            
            ax2.set_xlabel('Cumulative Depth (m)')
            ax2.set_ylabel('Layer Interface')
            ax2.set_title('Cumulative Depth Profile')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()  # Surface at top
            
            # Add text with parameter values and total depth change
            total_orig = np.sum(self.original_depths)
            total_new = np.sum(new_depths)
            depth_change_pct = ((total_new - total_orig) / total_orig) * 100
            
            text_str = f'Parameters:\nTotal Mult: {final_total_mult:.3f}\nShape Factor: {final_shape_factor:.3f}\n\n'
            text_str += f'Total Depth:\nOriginal: {total_orig:.2f} m\nOptimized: {total_new:.2f} m\n'
            text_str += f'Change: {depth_change_pct:+.1f}%'
            
            ax2.text(0.02, 0.98, text_str, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "depth_profile_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Depth profile comparison plot created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating depth profile comparison plot: {str(e)}")

    def _save_parameter_evaluation_history(self):
        """
        Save all parameter evaluations to CSV files for analysis.
        Enhanced to include depth parameters.
        
        Creates detailed files with:
        1. All parameter sets evaluated and their scores
        2. Best parameters over time
        3. Population statistics over generations
        """
        self.logger.info("Saving parameter evaluation history to files")
        
        try:
            # Create detailed evaluation records
            all_evaluations = []
            generation_summaries = []
            
            for gen_idx, gen_stats in enumerate(self.iteration_history):
                generation = gen_stats['generation']
                
                # Extract population data for this generation
                if 'population_scores' in gen_stats and hasattr(self, 'population'):
                    pop_scores = gen_stats['population_scores']
                    
                    # Get parameter values for this generation's population
                    for individual_idx in range(len(pop_scores)):
                        if individual_idx < len(self.population):
                            # Denormalize this individual's parameters
                            individual_params = self._denormalize_individual(self.population[individual_idx])
                            
                            # Create evaluation record
                            eval_record = {
                                'generation': generation,
                                'individual': individual_idx,
                                'score': pop_scores[individual_idx] if not np.isnan(pop_scores[individual_idx]) else None,
                                'is_valid': not np.isnan(pop_scores[individual_idx]) if not np.isnan(pop_scores[individual_idx]) else False,
                                'is_best_in_generation': individual_idx == np.nanargmax(pop_scores) if len(pop_scores) > 0 else False,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Add parameter values
                            for param_name, param_values in individual_params.items():
                                if len(param_values) == 1:
                                    eval_record[param_name] = param_values[0]
                                else:
                                    # For multi-value parameters, store statistics
                                    eval_record[f"{param_name}_mean"] = np.mean(param_values)
                                    eval_record[f"{param_name}_min"] = np.min(param_values)
                                    eval_record[f"{param_name}_max"] = np.max(param_values)
                                    eval_record[f"{param_name}_std"] = np.std(param_values)
                            
                            all_evaluations.append(eval_record)
                
                # Create generation summary
                gen_summary = {
                    'generation': generation,
                    'best_score': gen_stats.get('best_score'),
                    'mean_score': gen_stats.get('mean_score'),
                    'std_score': gen_stats.get('std_score'),
                    'worst_score': gen_stats.get('worst_score'),
                    'valid_individuals': gen_stats.get('valid_individuals', 0),
                    'total_individuals': self.population_size
                }
                
                # Add best parameters for this generation
                if gen_stats.get('best_params'):
                    for param_name, param_values in gen_stats['best_params'].items():
                        if len(param_values) == 1:
                            gen_summary[f"best_{param_name}"] = param_values[0]
                        else:
                            gen_summary[f"best_{param_name}_mean"] = np.mean(param_values)
                
                generation_summaries.append(gen_summary)
            
            # Save all evaluations to CSV
            if all_evaluations:
                eval_df = pd.DataFrame(all_evaluations)
                eval_csv_path = self.output_dir / "all_parameter_evaluations.csv"
                eval_df.to_csv(eval_csv_path, index=False)
                self.logger.info(f"💾 Saved {len(all_evaluations)} parameter evaluations to: {eval_csv_path}")
            
            # Save generation summaries to CSV
            if generation_summaries:
                summary_df = pd.DataFrame(generation_summaries)
                summary_csv_path = self.output_dir / "generation_summaries.csv"
                summary_df.to_csv(summary_csv_path, index=False)
                self.logger.info(f"📊 Saved {len(generation_summaries)} generation summaries to: {summary_csv_path}")
            
            # Save optimization metadata
            metadata = {
                'algorithm': 'Differential Evolution',
                'total_generations': self.max_iterations,
                'population_size': self.population_size,
                'F': self.F,
                'CR': self.CR,
                'target_metric': self.target_metric,
                'parameters_calibrated': self.local_params + self.basin_params + self.depth_params + self.mizuroute_params,
                'depth_calibration_enabled': self.calibrate_depth,
                'mizuroute_calibration_enabled': bool(self.mizuroute_params),
                'routing_configuration': self._check_routing_configuration(),
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
            
            # Create a summary statistics file
            if all_evaluations:
                eval_df = pd.DataFrame(all_evaluations)
                valid_scores = eval_df[eval_df['is_valid'] == True]['score']
                
                if len(valid_scores) > 0:
                    summary_stats = {
                        'total_evaluations': len(all_evaluations),
                        'valid_evaluations': len(valid_scores),
                        'success_rate': len(valid_scores) / len(all_evaluations) * 100,
                        'best_score': valid_scores.max(),
                        'worst_score': valid_scores.min(),
                        'mean_score': valid_scores.mean(),
                        'median_score': valid_scores.median(),
                        'std_score': valid_scores.std(),
                        'score_range': valid_scores.max() - valid_scores.min()
                    }
                    
                    # Add parameter statistics
                    param_cols = [col for col in eval_df.columns if col not in ['generation', 'individual', 'score', 'is_valid', 'is_best_in_generation', 'timestamp']]
                    for param_col in param_cols:
                        if eval_df[param_col].dtype in ['float64', 'int64']:
                            summary_stats[f"{param_col}_explored_min"] = eval_df[param_col].min()
                            summary_stats[f"{param_col}_explored_max"] = eval_df[param_col].max()
                            summary_stats[f"{param_col}_explored_range"] = eval_df[param_col].max() - eval_df[param_col].min()
                    
                    stats_df = pd.DataFrame([summary_stats])
                    stats_csv_path = self.output_dir / "optimization_statistics.csv"
                    stats_df.to_csv(stats_csv_path, index=False)
                    self.logger.info(f"📈 Saved optimization statistics to: {stats_csv_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving parameter evaluation history: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

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
        
        # Get all parameters to calibrate (excluding depth parameters)
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
        
        # Get all parameters to extract (excluding depth parameters)
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
            with xr.open_dataset(timestep_file, engine = 'netcdf4') as ds:
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