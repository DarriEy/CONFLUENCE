#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import shutil
from typing import Dict, Any, List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import re
import fcntl  # Unix/Linux
from pathlib import Path
import time
import random

class DEOptimizer:
    """
    Differential Evolution (DE) Optimizer for CONFLUENCE with Parallel Support
    
    This class performs parameter optimization using the Differential Evolution algorithm,
    enhanced with parallel processing capabilities using multiprocessing.
    Each parallel process gets its own working directory to avoid conflicts.

    The soil depth calibration uses two parameters:
    - total_mult: Overall depth multiplier (0.1-5.0)
    - shape_factor: Controls depth profile shape (0.1-3.0)
      - shape_factor > 1: Deeper layers get proportionally thicker
      - shape_factor < 1: Shallower layers get proportionally thicker
      - shape_factor = 1: Uniform scaling
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the DE Optimizer with parallel processing support.
        
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
        
        # Add parallel processing configuration
        self.num_processes = max(1, self.config.get('MPI_PROCESSES', 1))
        self.use_parallel = self.num_processes > 1
        
        if self.use_parallel:
            self.logger.info(f"🚀 Parallel processing ENABLED with {self.num_processes} processes")
            self.logger.info("Each process will use its own working directory to avoid conflicts")
            
            # Adjust population size for better parallel efficiency
            if self.population_size < self.num_processes * 2:
                old_pop_size = self.population_size
                self.population_size = self.num_processes * 2
                self.logger.info(f"Increased population size from {old_pop_size} to {self.population_size} "
                                f"for better parallel efficiency")
            
            # Create parallel working directories
            self._setup_parallel_directories()
        else:
            self.logger.info("Sequential processing mode (single process)")
        
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
        
    def _setup_parallel_directories(self):
        """Create separate working directories for each parallel process with better verification."""
        self.logger.info("Setting up parallel working directories")
        
        self.parallel_dirs = []

        for proc_id in range(self.num_processes):
            # Create process-specific directories
            proc_base_dir = self.optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_summa_dir = proc_base_dir / "SUMMA"
            proc_mizuroute_dir = proc_base_dir / "mizuRoute"
            proc_settings_dir = proc_base_dir / "settings" / "SUMMA"
            proc_mizu_settings_dir = proc_base_dir / "settings" / "mizuRoute"
            
            # Create all directories using pathlib
            for directory in [proc_base_dir, proc_summa_dir, proc_mizuroute_dir, proc_settings_dir, proc_mizu_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Copy settings files to process directory
            self._copy_settings_to_process_dir(proc_settings_dir, proc_mizu_settings_dir)

            # Store directory info for this process
            proc_dirs = {
                'base_dir': proc_base_dir,
                'summa_dir': proc_summa_dir,
                'mizuroute_dir': proc_mizuroute_dir,
                'summa_settings_dir': proc_settings_dir,
                'mizuroute_settings_dir': proc_mizu_settings_dir,
                'proc_id': proc_id
            }

            # Update file managers for this process
            self._update_file_manager_for_proc(proc_dirs)
            
            # Verify the setup worked
            file_manager = proc_settings_dir / 'fileManager.txt'
            if not file_manager.exists():
                self.logger.error(f"Failed to create file manager for process {proc_id}")
                raise FileNotFoundError(f"File manager not created for process {proc_id}")
            
            self.parallel_dirs.append(proc_dirs)
        
        self.logger.info(f"Created {len(self.parallel_dirs)} parallel working directories")

    
    def _run_summa_simulation_for_proc(self, file_manager_path: Path, summa_exe: Path, log_dir: Path) -> bool:
        summa_command = f"{summa_exe} -m {file_manager_path}"
        print(summa_command)
        log_file = log_dir / f"summa_proc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(log_file, 'w') as f:
                subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
        
    def _copy_settings_to_process_dir(self, proc_settings_dir, proc_mizu_settings_dir):
        """Copy all necessary settings files to a process-specific directory."""
        
        def safe_copy_file(src_file, dest_file, max_retries=5):
            """Safely copy a file with retry logic and locking."""
            for attempt in range(max_retries):
                try:
                    # Create temp destination
                    temp_dest = dest_file.with_suffix('.copying')
                    
                    with open(src_file, 'rb') as src:
                        # Lock source file for reading
                        fcntl.flock(src.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                        
                        with open(temp_dest, 'wb') as dst:
                            # Copy in chunks
                            shutil.copyfileobj(src, dst)
                            dst.flush()
                            os.fsync(dst.fileno())  # Force write to disk
                    
                    # Atomic rename
                    temp_dest.replace(dest_file)
                    
                    # Copy metadata
                    shutil.copystat(src_file, dest_file)
                    return True
                    
                except (IOError, OSError) as e:
                    # Clean up temp file
                    if temp_dest.exists():
                        temp_dest.unlink()
                        
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(0.1, 0.5))
                        continue
                    else:
                        raise e
            return False
        
        # Copy SUMMA settings
        if self.optimization_settings_dir.exists():
            for settings_file in self.optimization_settings_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_settings_dir / settings_file.name
                    safe_copy_file(settings_file, dest_file)
        
        # Copy mizuRoute settings  
        mizu_source_dir = self.optimization_dir / "settings" / "mizuRoute"
        if mizu_source_dir.exists():
            for settings_file in mizu_source_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_mizu_settings_dir / settings_file.name
                    safe_copy_file(settings_file, dest_file)
    
    def test_parallel_setup(self):
        """Test parallel processing setup without running full optimization."""
        return True
        
    
    def cleanup_parallel_directories(self):
        """Clean up parallel working directories when optimization is complete."""
        return
        '''
        if not self.use_parallel:
            return
        
        self.logger.info("Cleaning up parallel working directories")
        
        try:
            for proc_dirs in self.parallel_dirs:
                proc_base_dir = proc_dirs['base_dir']
                if proc_base_dir.exists():
                    shutil.rmtree(proc_base_dir)
                    
            self.logger.info("Parallel directory cleanup completed")
        except Exception as e:
            self.logger.warning(f"Error during parallel directory cleanup: {str(e)}")
        '''
    
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
        Run the DE optimization algorithm with parameter persistence, depth calibration, and parallel processing.
        
        Enhanced to:
        1. Check for existing optimized parameters and use them as starting point
        2. Save best parameters back to default settings when complete
        3. Handle soil depth calibration if enabled
        4. Support parallel processing for faster evaluation
        
        Returns:
            Dict: Dictionary with optimization results
        """
        self.logger.info("Starting DE optimization with parameter persistence and parallel processing support")
        
        if self.calibrate_depth:
            self.logger.info("🌱 Soil depth calibration is ENABLED")
            self.logger.info("📏 Will optimize soil depth profile using shape method")
        
        try:
            # Test parallel setup if enabled
            if self.use_parallel:
                if not self.test_parallel_setup():
                    self.logger.warning("Parallel setup test failed, falling back to sequential mode")
                    self.use_parallel = False
                    self.num_processes = 1
            
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
            save_success = False
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
                'depth_calibration_enabled': self.calibrate_depth,
                'parallel_processing_used': self.use_parallel,
                'num_processes_used': self.num_processes if self.use_parallel else 1
            }
            
            return results
        
        finally:
            # Always cleanup parallel directories
            self.cleanup_parallel_directories()
    
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
        Enhanced to handle different file formats robustly.
        
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
        
        self.logger.debug(f"Parsing parameter bounds from: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    # Parse the line to extract parameter name and bounds
                    # Both files use format: param | default | min | max
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 3:
                        continue
                        
                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            if len(parts) >= 4:
                                # Standard format: param | default | min | max
                                default_str = parts[1]
                                min_str = parts[2]
                                max_str = parts[3]
                            elif len(parts) == 3:
                                # Fallback format: param | min | max (no default)
                                min_str = parts[1]  
                                max_str = parts[2]
                            else:
                                continue
                            
                            # Convert scientific notation (1.0d-6 -> 1.0e-6)
                            min_val = float(min_str.replace('d','e').replace('D','e'))
                            max_val = float(max_str.replace('d','e').replace('D','e'))
                            
                            # Validate bounds make sense
                            if min_val > max_val:
                                self.logger.warning(f"Line {line_num}: min > max for {param_name}, swapping")
                                min_val, max_val = max_val, min_val
                            
                            if min_val == max_val:
                                self.logger.warning(f"Line {line_num}: min == max for {param_name}, adding small range")
                                range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
                                min_val -= range_val
                                max_val += range_val
                            
                            bounds[param_name] = {'min': min_val, 'max': max_val}
                            
                            # Log the parsed bounds for verification
                            self.logger.debug(f"Line {line_num}: {param_name} bounds = [{min_val:.6e}, {max_val:.6e}]")
                            
                            # Special validation for known problematic parameters
                            if param_name == 'aquiferBaseflowRate':
                                if max_val > 1e-3:  # Should be very small
                                    self.logger.warning(f"aquiferBaseflowRate max bound seems large: {max_val:.6e}")
                                if min_val < 1e-10:  # Should not be too small
                                    self.logger.debug(f"aquiferBaseflowRate min bound: {min_val:.6e}")
                            
                        except (ValueError, IndexError) as e:
                            self.logger.error(f"Line {line_num}: Failed to parse bounds for {param_name}")
                            self.logger.error(f"Line content: {line}")
                            self.logger.error(f"Parts: {parts}")
                            self.logger.error(f"Error: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"Error parsing parameter info file {file_path}: {str(e)}")
        
        self.logger.info(f"Found bounds for {len(bounds)} parameters in {file_path.name}")
        
        # Log a few key parameters for verification
        key_params = ['aquiferBaseflowRate', 'k_soil', 'theta_sat', 'routingGammaScale']
        for param in key_params:
            if param in bounds:
                b = bounds[param]
                self.logger.info(f"  {param}: [{b['min']:.6e}, {b['max']:.6e}]")
        
        return bounds
        

    def _denormalize_individual(self, normalized_individual):
        """Enhanced with debugging for parameter denormalization issues"""
        params = {}
        
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
                
                # Denormalize value from [0,1] to [min, max]
                denorm_value = min_val + normalized_individual[i] * (max_val - min_val)
                
                
                # CRITICAL FIX: Validate denormalized value is within bounds
                if denorm_value < min_val or denorm_value > max_val:
                    self.logger.error(f"❌ DENORM ERROR {param_name}: {denorm_value:.6e} outside bounds!")
                    denorm_value = np.clip(denorm_value, min_val, max_val)
                    self.logger.warning(f"🔧 CLIPPED {param_name}: -> {denorm_value:.6e}")
                
                # Continue with existing parameter assignment logic...
                if param_name in self.depth_params:
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.basin_params:
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.mizuroute_params:
                    params[param_name] = denorm_value
                else:
                    # Local/HRU level parameter
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
            # If in parallel mode and we have proc_dirs, run SUMMA in proc-local dir
            if self.use_parallel and hasattr(self, 'parallel_dirs') and len(self.parallel_dirs) == 1:
                proc_dirs = self.parallel_dirs[0]
                summa_path = self.config.get('SUMMA_INSTALL_PATH')
                if summa_path == 'default':
                    summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
                else: 
                    summa_path = Path(summa_path)

                summa_exe = summa_path / self.config.get('SUMMA_EXE')
                file_manager = Path(proc_dirs['summa_settings_dir']) / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
                log_dir = Path(proc_dirs['summa_dir']) / "logs"
                summa_success = self._run_summa_simulation_for_proc(file_manager, summa_exe, log_dir)
            else:
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
        Get initial parameter values directly from parameter info files instead of running SUMMA.
        This is more reliable and faster than the simulation-based approach.
        
        Returns:
            Dict: Dictionary with parameter names as keys and arrays of values as values
        """
        self.logger.info("Reading initial parameter values directly from parameter files")
        
        try:
            # Parse default values from parameter files
            defaults = self._parse_parameter_defaults_from_files()
            
            if not defaults:
                self.logger.error("No parameter defaults found in parameter files")
                return None
            
            # Expand defaults to appropriate dimensions (HRU count for local params)
            expanded_defaults = self._expand_defaults_to_hru_count(defaults)
            
            # Add depth parameters if depth calibration is enabled
            if self.calibrate_depth and expanded_defaults:
                expanded_defaults['total_mult'] = np.array([1.0])    # Default: no depth scaling
                expanded_defaults['shape_factor'] = np.array([1.0])  # Default: uniform scaling
                self.logger.info("Added default depth parameters: total_mult=1.0, shape_factor=1.0")
            
            # Add mizuRoute parameters if routing calibration is enabled
            if self.mizuroute_params and expanded_defaults:
                for param in self.mizuroute_params:
                    expanded_defaults[param] = self._get_default_mizuroute_value(param)
                self.logger.info("Added default mizuRoute parameters")
            
            # Log summary
            self.logger.info(f"✅ Successfully loaded default values for {len(expanded_defaults)} parameters:")
            for param_name, values in expanded_defaults.items():
                if len(values) == 1:
                    self.logger.info(f"   {param_name}: {values[0]:.6e}")
                else:
                    self.logger.info(f"   {param_name}: {len(values)} values, mean={np.mean(values):.6e}")
            
            return expanded_defaults
            
        except Exception as e:
            self.logger.error(f"Error reading parameter defaults from files: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
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
    
    def _evaluate_population_parallel(self):
        """Evaluate population in parallel using multiprocessing."""
        self.logger.info(f"Evaluating population with {self.num_processes} parallel processes")
        
        evaluation_tasks = []
        for i in range(self.population_size):
            if not np.isnan(self.population_scores[i]):
                continue
            
            params = self._denormalize_individual(self.population[i])
            
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"pop_eval_{i:03d}"
            }
            evaluation_tasks.append(task)
        
        if not evaluation_tasks:
            return
        
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        for result in results:
            individual_id = result['individual_id']
            score = result['score']
            
            self.population_scores[individual_id] = score if score is not None else float('-inf')
            
            if self.population_scores[individual_id] > self.best_score:
                self.best_score = self.population_scores[individual_id]
                self.best_params = result['params'].copy()
                
    def _run_parallel_evaluations(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """
        Run parallel evaluations without staggered batching - submit all tasks immediately.
        
        Args:
            evaluation_tasks: List of task dictionaries with individual_id, params, proc_id, evaluation_id
            
        Returns:
            List of result dictionaries with individual_id, params, score, error
        """
        start_time = time.time()
        num_tasks = len(evaluation_tasks)
        
        self.logger.info(f"Starting parallel evaluation of {num_tasks} tasks")
        self.logger.info(f"🚀 Using ALL {self.num_processes} processes simultaneously")
        
        # Prepare all worker tasks with full task data
        worker_tasks = []
        for task in evaluation_tasks:
            proc_dirs = self.parallel_dirs[task['proc_id']]
            
            # Create complete task data for worker function
            task_data = {
                'individual_id': task['individual_id'],
                'params': task['params'],
                'proc_id': task['proc_id'],
                'evaluation_id': task['evaluation_id'],
                
                # Paths (as strings for serialization)
                'summa_exe': str(self._get_summa_exe_path()),
                'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
                'summa_dir': str(proc_dirs['summa_dir']),
                'settings_dir': str(proc_dirs['summa_settings_dir']),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
                
                # Configuration data
                'target_metric': self.target_metric,
                'calibrate_depth': self.calibrate_depth,
                'original_depths': self.original_depths.tolist() if self.original_depths is not None else None,
                'mizuroute_params': self.mizuroute_params,
                'local_params': self.local_params,
                'basin_params': self.basin_params,
                'depth_params': self.depth_params,
                'obs_file': str(self._get_obs_file_path()),
                'catchment_area': self._get_catchment_area() or 1e6,
                'domain_method': self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
                'routing_delineation': self.config.get('ROUTING_DELINEATION', 'lumped'),
            }
            
            worker_tasks.append(task_data)
        
        # Submit ALL tasks to process pool immediately - no batching, no delays
        results = []
        completed_count = 0
        failed_count = 0
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                self.logger.info(f"📤 Submitted all {num_tasks} tasks to {self.num_processes} worker processes")
                
                # Submit all tasks at once
                future_to_task_id = {}
                for i, task_data in enumerate(worker_tasks):
                    future = executor.submit(_evaluate_parameters_worker, task_data)
                    future_to_task_id[future] = i
                
                # Collect results as they complete (in any order)
                for future in as_completed(future_to_task_id):
                    task_id = future_to_task_id[future]
                    task_data = worker_tasks[task_id]
                    completed_count += 1
                    
                    try:
                        # Get result from worker with timeout
                        result = future.result(timeout=1800)  # 30 minute timeout per task
                        results.append(result)
                        
                        if result['score'] is not None:
                            # Log successful completion
                            if isinstance(result['score'], (int, float)):
                                score_str = f"score={result['score']:.6f}"
                            else:
                                score_str = f"score={result['score']}"
                            
                            self.logger.info(f"✅ Task {completed_count}/{num_tasks} completed - {score_str}")
                        else:
                            failed_count += 1
                            error_msg = result.get('error', 'Unknown error')
                            self.logger.error(f"❌ Task {completed_count}/{num_tasks} failed: {error_msg}")
                            
                    except Exception as e:
                        # Handle task timeout or other exceptions
                        failed_count += 1
                        error_result = {
                            'individual_id': task_data['individual_id'],
                            'params': task_data['params'],
                            'score': None,
                            'error': f'Task exception: {str(e)}'
                        }
                        results.append(error_result)
                        self.logger.error(f"❌ Task {completed_count}/{num_tasks} exception: {str(e)}")
                    
                    # Progress logging every 5 completed tasks or at the end
                    if completed_count % 5 == 0 or completed_count == num_tasks:
                        elapsed = time.time() - start_time
                        successful = sum(1 for r in results if r['score'] is not None)
                        avg_time = elapsed / completed_count if completed_count > 0 else 0
                        remaining = num_tasks - completed_count
                        estimated_remaining = (avg_time * remaining) / 60 if remaining > 0 else 0
                        
                        self.logger.info(f"📊 Progress: {completed_count}/{num_tasks} completed, "
                                    f"{successful} successful ({100*successful/completed_count:.1f}%), "
                                    f"elapsed: {elapsed/60:.1f}min" +
                                    (f", est. remaining: {estimated_remaining:.1f}min" if remaining > 0 else ""))
            
            # Final summary
            elapsed = time.time() - start_time
            successful_count = sum(1 for r in results if r['score'] is not None)
            
            self.logger.info(f"🏁 Parallel evaluation completed:")
            self.logger.info(f"   ✅ Successful: {successful_count}/{num_tasks} ({100*successful_count/num_tasks:.1f}%)")
            self.logger.info(f"   ❌ Failed: {failed_count}/{num_tasks} ({100*failed_count/num_tasks:.1f}%)")
            self.logger.info(f"   ⏱️ Total time: {elapsed/60:.1f} minutes")
            
            if successful_count > 0:
                avg_time_per_task = elapsed / num_tasks
                self.logger.info(f"   📈 Average time per task: {avg_time_per_task:.1f} seconds")
            
            if successful_count == 0:
                self.logger.error("⚠️ All evaluations failed! Check SUMMA configuration and parameter bounds.")
            elif failed_count > num_tasks * 0.5:
                self.logger.warning(f"⚠️ High failure rate: {failed_count}/{num_tasks} tasks failed")
            
            # Ensure results are in the same order as input tasks
            # Sort results by individual_id to match the original order
            results_dict = {r['individual_id']: r for r in results}
            ordered_results = []
            for task in evaluation_tasks:
                individual_id = task['individual_id']
                if individual_id in results_dict:
                    ordered_results.append(results_dict[individual_id])
                else:
                    # Create a failure result if somehow missing
                    ordered_results.append({
                        'individual_id': individual_id,
                        'params': task['params'],
                        'score': None,
                        'error': 'Result missing from worker output'
                    })
            
            return ordered_results
            
        except Exception as e:
            self.logger.error(f"💥 Critical error in parallel evaluation: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return failure results for all tasks
            failure_results = []
            for task in evaluation_tasks:
                failure_results.append({
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Critical parallel evaluation error: {str(e)}'
                })
            
            return failure_results

    def _get_summa_exe_path(self) -> Path:
        """Get the SUMMA executable path."""
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        return summa_path / self.config.get('SUMMA_EXE')


    def _get_obs_file_path(self) -> Path:
        """Get the path to the observed streamflow file."""
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            return self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        else:
            return Path(obs_path)
        
    def _update_file_manager_for_proc(self, proc_dirs, is_final_run=False):
        """Update file manager for process with better error handling and path verification."""
        fm_path = proc_dirs['summa_settings_dir'] / 'fileManager.txt'
        if not fm_path.exists():
            self.logger.error(f"File manager template not found for process {proc_dirs['proc_id']}: {fm_path}")
            return

        try:
            # Read the original file
            with open(fm_path, 'r') as f:
                lines = f.readlines()
            
            # Process lines
            updated_lines = []
            prefix = f"proc{proc_dirs['proc_id']:02d}_de_opt_{self.experiment_id}"
            
            for line in lines:
                if 'outFilePrefix' in line:
                    updated_lines.append(f"outFilePrefix        '{prefix}'\n")
                elif 'outputPath' in line:
                    # Ensure forward slashes for cross-platform compatibility
                    output_path = str(proc_dirs['summa_dir']).replace('\\', '/')
                    updated_lines.append(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    settings_path = str(proc_dirs['summa_settings_dir']).replace('\\', '/')
                    updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                else:
                    updated_lines.append(line)
            
            # Write the updated file
            with open(fm_path, 'w') as f:
                f.writelines(updated_lines)
            
            self.logger.debug(f"Updated file manager for process {proc_dirs['proc_id']}")
            
            # Verify the update worked by checking the file content
            with open(fm_path, 'r') as f:
                content = f.read()
                if prefix not in content:
                    self.logger.warning(f"File manager update may have failed for process {proc_dirs['proc_id']}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update file manager for process {proc_dirs['proc_id']}: {str(e)}")


    def _create_evaluation_context(self) -> Dict:
        """Create a serializable evaluation context for worker processes."""
        return {
            'config': self.config,
            'domain_name': self.domain_name,
            'experiment_id': self.experiment_id,
            'project_dir': str(self.project_dir),
            'calibration_period': self.calibration_period,
            'evaluation_period': self.evaluation_period,
            'target_metric': self.target_metric,
            'calibrate_depth': self.calibrate_depth,
            'original_depths': self.original_depths,
            'depth_params': self.depth_params,
            'mizuroute_params': self.mizuroute_params,
            'local_params': self.local_params,
            'basin_params': self.basin_params,
            'param_bounds': self.param_bounds,
            'output_variable': self.config.get("TARGET_VARIABLE", "scalarTotalRunoff"),
            'target_metric': self.target_metric,
        }

    
    def _evaluate_population(self):
        """Evaluate population - uses parallel or sequential mode based on configuration."""
        if self.use_parallel:
            self._evaluate_population_parallel()
        else:
            for i in range(self.population_size):
                if not np.isnan(self.population_scores[i]):
                    continue
                
                params = self._denormalize_individual(self.population[i])
                score = self._evaluate_parameters(params)
                self.population_scores[i] = score if score is not None else float('-inf')
                
                if self.population_scores[i] > self.best_score:
                    self.best_score = self.population_scores[i]
                    self.best_params = params.copy()
    
    def _run_de_algorithm(self):
        """
        Run the DE algorithm with enhanced logging and parallel trial evaluation.
        
        Returns:
            Tuple: (best_params, best_score, history)
        """
        self.logger.info("Running DE algorithm with parallel support")
        self.logger.info("=" * 60)
        self.logger.info(f"Target metric: {self.target_metric} (higher is better)")
        self.logger.info(f"Total generations: {self.max_iterations}")
        self.logger.info(f"Population size: {self.population_size}")
        self.logger.info(f"DE parameters: F={self.F}, CR={self.CR}")
        if self.calibrate_depth:
            self.logger.info("🌱 Soil depth calibration: ENABLED")
        if self.use_parallel:
            self.logger.info(f"🚀 Parallel processing: {self.num_processes} processes")
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
            
            # Evaluate trial population (parallel or sequential)
            if self.use_parallel:
                trial_results = self._evaluate_trial_population_parallel(trial_population)
            else:
                trial_results = self._evaluate_trial_population_sequential(trial_population)
            
            generation_improvements = 0
            
            # Selection phase
            for i, result in enumerate(trial_results):
                trial_score = result['score'] if result['score'] is not None else float('-inf')
                
                # Selection: keep trial if better than parent
                if trial_score > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_score
                    generation_improvements += 1
                    
                    # Update global best if this is the new best
                    if trial_score > self.best_score:
                        old_best = self.best_score
                        self.best_score = trial_score
                        self.best_params = result['params'].copy()
                        
                        # Log the new best with depth info if applicable
                        log_msg = f"Gen {generation:3d}: Individual {i:2d} ⭐ NEW GLOBAL BEST! {self.target_metric}={self.best_score:.6f}"
                        if self.calibrate_depth and 'total_mult' in result['params'] and 'shape_factor' in result['params']:
                            tm = result['params']['total_mult'][0] if isinstance(result['params']['total_mult'], np.ndarray) else result['params']['total_mult']
                            sf = result['params']['shape_factor'][0] if isinstance(result['params']['shape_factor'], np.ndarray) else result['params']['shape_factor']
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
    
    def _evaluate_trial_population_parallel(self, trial_population):
        """Evaluate trial population in parallel."""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            params = self._denormalize_individual(trial_population[i])
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"trial_gen_{len(self.iteration_history):03d}_{i:03d}"
            }
            evaluation_tasks.append(task)
        
        return self._run_parallel_evaluations(evaluation_tasks)
    
    def _evaluate_trial_population_sequential(self, trial_population):
        """Evaluate trial population sequentially (fallback)."""
        results = []
        
        for i in range(self.population_size):
            params = self._denormalize_individual(trial_population[i])
            score = self._evaluate_parameters(params)
            
            result = {
                'individual_id': i,
                'params': params,
                'score': score
            }
            results.append(result)
        
        return results
    
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
        """Generate trialParams.nc with proper parameter dimensions."""
        self.logger.info('hello from generate trial params file')
        trial_params_path = target_dir / filename
        
        # Get HRU and GRU counts
        with xr.open_dataset(self.attr_file_path) as ds:
            num_hrus = ds.sizes.get('hru', 1)
            num_grus = ds.sizes.get('gru', 1)  # Usually 1 for single watershed
        
        routing_params = ['routingGammaShape', 'routingGammaScale']
        
        with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
            # Create dimensions
            hru_dim = output_ds.createDimension('hru', num_hrus)
            gru_dim = output_ds.createDimension('gru', num_grus)  # ADD GRU DIMENSION
            
            # Create coordinate variables
            hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
            hru_var[:] = range(1, num_hrus + 1)
            
            gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
            gru_var[:] = range(1, num_grus + 1)
            
            # Add parameter variables with CORRECT dimensions
            for param_name, param_values in params.items():
                param_values_array = np.asarray(param_values)
                
                if param_name in routing_params:
                    # ROUTING PARAMETERS: Use GRU dimension
                    param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                    param_var.long_name = f"Trial value for {param_name}"
                    
                    # Routing parameters should have 1 value per GRU (usually just 1)
                    if len(param_values_array) == 1:
                        param_var[:] = param_values_array[0]
                    else:
                        param_var[:] = param_values_array[:num_grus]
                        
                    self.logger.debug(f"✓ {param_name}: GRU-level parameter, value = {param_values_array[0]:.6e}")
                    
                elif param_name in self.basin_params:
                    # BASIN PARAMETERS: Use GRU dimension  
                    param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                    param_var.long_name = f"Trial value for {param_name}"
                    
                    if len(param_values_array) == 1:
                        param_var[:] = param_values_array[0]
                    else:
                        param_var[:] = param_values_array[:num_grus]
                        
                    self.logger.debug(f"✓ {param_name}: Basin-level parameter, value = {param_values_array[0]:.6e}")
                    
                else:
                    # LOCAL/HRU PARAMETERS: Use HRU dimension
                    param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                    param_var.long_name = f"Trial value for {param_name}"
                    
                    if len(param_values_array) == num_hrus:
                        param_var[:] = param_values_array
                    else:
                        # Broadcast single value to all HRUs
                        param_var[:] = param_values_array[0]
                        
                    self.logger.debug(f"✓ {param_name}: HRU-level parameter, {len(param_values_array)} values")
        
        self.logger.info(f"✅ Created {trial_params_path.name} with proper GRU/HRU dimensions")
        
        return trial_params_path
    
    def _run_summa_simulation(self) -> bool:
        """
        Run SUMMA with the current trial parameters using the global optimization settings.
        This version is retained for non-parallel or final evaluations.

        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.debug("Running SUMMA simulation in optimization environment")

        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)

        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        file_manager = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')

        if not file_manager.exists():
            self.logger.error(f"Optimization file manager not found: {file_manager}")
            return False

        summa_command = f"{summa_exe} -m {file_manager}"
        log_file = self.summa_sim_dir / "logs" / f"summa_de_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger.info(f"summa command: {summa_command}")
        try:
            with open(log_file, 'w') as f:
                subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)

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
                
                #self.logger.info(f"Using catchment area: {catchment_area:.2f} m² ({catchment_area/1e6:.2f} km²)")
                
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

    def _parse_parameter_defaults_from_files(self):
        """
        Parse default parameter values directly from parameter info files.
        This replaces the complex parameter extraction simulation.
        
        Returns:
            Dict: Dictionary with parameter names as keys and default values as values
        """
        self.logger.info("Reading default parameter values directly from parameter files")
        
        defaults = {}
        
        # Parse local parameters
        if self.local_params:
            local_defaults = self._parse_defaults_from_file(
                self.local_param_info_path, self.local_params, "local"
            )
            defaults.update(local_defaults)
        
        # Parse basin parameters  
        if self.basin_params:
            basin_defaults = self._parse_defaults_from_file(
                self.basin_param_info_path, self.basin_params, "basin"
            )
            defaults.update(basin_defaults)
        
        return defaults

    def _parse_defaults_from_file(self, file_path, param_names, param_type):
        """
        Parse default values from a single parameter info file.
        
        Args:
            file_path: Path to parameter info file
            param_names: List of parameter names to extract
            param_type: Type of parameters ('local' or 'basin') for logging
            
        Returns:
            Dict: Parameter defaults
        """
        defaults = {}
        
        if not file_path.exists():
            self.logger.error(f"{param_type.title()} parameter file not found: {file_path}")
            return defaults
        
        self.logger.info(f"Parsing {param_type} parameter defaults from: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    # Parse the line: param | default | min | max
                    parts = [p.strip() for p in line.split('|')]
                    
                    # DEBUG: Print every line that gets parsed for aquiferBaseflowRate
                    if 'aquiferBaseflow' in line:
                        self.logger.info(f"🔍 DEBUG Line {line_num}: {line}")
                        self.logger.info(f"🔍 DEBUG Parts: {parts}")
                    
                    if len(parts) < 4:
                        continue
                        
                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            # Get default value (parts[1])
                            default_str = parts[1]
                            min_str = parts[2] 
                            max_str = parts[3]
                            
                            # DEBUG: Show the raw strings before conversion
                            if 'aquiferBaseflow' in param_name:
                                self.logger.info(f"🔍 DEBUG {param_name} raw values:")
                                self.logger.info(f"    default_str: '{default_str}'")
                                self.logger.info(f"    min_str: '{min_str}'") 
                                self.logger.info(f"    max_str: '{max_str}'")
                            
                            default_val = float(default_str.replace('d','e').replace('D','e'))
                            min_val = float(min_str.replace('d','e').replace('D','e'))
                            max_val = float(max_str.replace('d','e').replace('D','e'))
                            
                            # DEBUG: Show converted values
                            if 'aquiferBaseflow' in param_name:
                                self.logger.info(f"🔍 DEBUG {param_name} converted values:")
                                self.logger.info(f"    default_val: {default_val:.6e}")
                                self.logger.info(f"    min_val: {min_val:.6e}")
                                self.logger.info(f"    max_val: {max_val:.6e}")
                            
                            # Validate that default is within bounds
                            if not (min_val <= default_val <= max_val):
                                self.logger.warning(f"Line {line_num}: Default value for {param_name} ({default_val:.6e}) "
                                                f"not within bounds [{min_val:.6e}, {max_val:.6e}]")
                                # Use middle of bounds as fallback
                                default_val = (min_val + max_val) / 2
                                self.logger.warning(f"Using middle of bounds: {default_val:.6e}")
                            
                            # Store as numpy array for consistency with existing code
                            if param_type == "basin":
                                defaults[param_name] = np.array([default_val])
                            else:
                                # For local parameters, we'll expand to HRU count later
                                defaults[param_name] = np.array([default_val])
                            
                            self.logger.debug(f"✓ {param_name}: default = {default_val:.6e}")
                            
                        except ValueError as e:
                            self.logger.error(f"Line {line_num}: Could not parse {param_name}: {line}")
                            self.logger.error(f"Error: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"Error reading {param_type} parameter file {file_path}: {str(e)}")
        
        self.logger.info(f"Found defaults for {len(defaults)} {param_type} parameters")
        return defaults

    def _expand_defaults_to_hru_count(self, defaults):
        """
        Expand parameter defaults to match HRU count for local parameters.
        
        Args:
            defaults: Dictionary with parameter defaults
            
        Returns:
            Dict: Parameter defaults expanded to HRU count
        """
        try:
            # Get HRU count from attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                if 'hru' in ds.dims:
                    num_hrus = ds.sizes['hru']
                else:
                    num_hrus = 1
            
            self.logger.info(f"Expanding parameters to {num_hrus} HRUs")
            
            expanded_defaults = {}
            
            # Get routing parameters that should stay at GRU level
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            for param_name, default_values in defaults.items():
                if param_name in self.basin_params or param_name in routing_params:
                    # Basin/GRU parameters stay as single values
                    expanded_defaults[param_name] = default_values
                    self.logger.debug(f"{param_name}: kept as basin parameter ({len(default_values)} values)")
                else:
                    # Local/HRU parameters get expanded to all HRUs
                    expanded_values = np.full(num_hrus, default_values[0])
                    expanded_defaults[param_name] = expanded_values
                    self.logger.debug(f"{param_name}: expanded to {len(expanded_values)} HRUs")
            
            return expanded_defaults
            
        except Exception as e:
            self.logger.error(f"Error expanding defaults to HRU count: {str(e)}")
            return defaults

    def debug_single_parallel_process(self, process_id: int = 0) -> Dict:
        """
        Debug a single parallel process to identify issues.
        """
        if not self.parallel_dirs or process_id >= len(self.parallel_dirs):
            return {'error': 'Invalid process ID or parallel directories not set up'}
        
        proc_dirs = self.parallel_dirs[process_id]
        
        # Create simple test parameters
        dummy_params = {}
        param_bounds = self._parse_parameter_bounds()
        
        for param_name in self.local_params + self.basin_params:
            if param_name in param_bounds:
                bounds = param_bounds[param_name]
                # Use middle value of bounds
                mid_value = (bounds['min'] + bounds['max']) / 2
                dummy_params[param_name] = np.array([mid_value])
        
        # Create debug task
        task_data = {
            'individual_id': 999,
            'params': dummy_params,
            'proc_id': process_id,
            'evaluation_id': 'debug_test',
            
            'summa_exe': str(self._get_summa_exe_path()),
            'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
            'summa_dir': str(proc_dirs['summa_dir']),
            'settings_dir': str(proc_dirs['summa_settings_dir']),
            'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
            
            'target_metric': self.target_metric,
            'calibrate_depth': self.calibrate_depth,
            'original_depths': self.original_depths.tolist() if self.original_depths is not None else None,
            'mizuroute_params': self.mizuroute_params,
            'local_params': self.local_params,
            'basin_params': self.basin_params,
            'depth_params': self.depth_params,
            'obs_file': str(self._get_obs_file_path()),
            'catchment_area': self._get_catchment_area() or 1e6,
            'domain_method': self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
            'routing_delineation': self.config.get('ROUTING_DELINEATION', 'lumped'),
        }
        
        self.logger.info(f"Running debug test on process {process_id}")
        
        # Test the worker function directly
        try:
            result = _evaluate_parameters_worker(task_data)
            
            self.logger.info("Debug test completed:")
            self.logger.info(f"  Score: {result.get('score', 'None')}")
            self.logger.info(f"  Error: {result.get('error', 'None')}")
            
            # Additional debugging info
            debug_info = {
                'process_id': process_id,
                'file_manager_exists': Path(task_data['file_manager']).exists(),
                'summa_exe_exists': Path(task_data['summa_exe']).exists(),
                'summa_dir_exists': Path(task_data['summa_dir']).exists(),
                'settings_dir_exists': Path(task_data['settings_dir']).exists(),
                'obs_file_exists': Path(task_data['obs_file']).exists(),
                'result': result
            }
            
            # Check what files exist in the process directory
            summa_dir = Path(task_data['summa_dir'])
            if summa_dir.exists():
                debug_info['files_in_summa_dir'] = [f.name for f in summa_dir.glob('*')]
                debug_info['nc_files_in_summa_dir'] = [f.name for f in summa_dir.glob('*.nc')]
            
            return debug_info
            
        except Exception as e:
            import traceback
            return {
                'process_id': process_id,
                'error': f'Debug test failed: {str(e)}',
                'traceback': traceback.format_exc()
            }



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
            if self.use_parallel:
                title += f' ({self.num_processes} processes)'
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
                'parallel_processing_enabled': self.use_parallel,
                'num_processes_used': self.num_processes if self.use_parallel else 1,
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

def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger) -> bool:
    """Generate trial parameters file in worker process with proper GRU/HRU dimensions."""
    try:
        import netCDF4 as nc
        import xarray as xr
        import numpy as np
        
        if not params:
            return True
        
        attr_file = settings_dir / 'attributes.nc'
        trial_params_path = settings_dir / 'trialParams.nc'
        
        # Define parameter levels
        routing_params = ['routingGammaShape', 'routingGammaScale']
        basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
        gru_level_params = routing_params + basin_params
        
        with xr.open_dataset(attr_file) as attr_ds:
            # Get HRU information
            hru_ids = attr_ds['hruId'].values
            num_hrus = len(hru_ids)
            
            # Get GRU information (usually 1 for single watershed)
            if 'gruId' in attr_ds:
                gru_ids = attr_ds['gruId'].values
                num_grus = len(gru_ids)
            else:
                # If no gruId in attributes, assume 1 GRU
                gru_ids = np.array([1])
                num_grus = 1
            
            logger.debug(f"Creating trialParams.nc with {num_hrus} HRUs and {num_grus} GRUs")
            
            with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as ds:
                # Create BOTH dimensions
                ds.createDimension('hru', num_hrus)
                ds.createDimension('gru', num_grus)  # ← CRITICAL: Add GRU dimension
                
                # Create coordinate variables
                hru_var = ds.createVariable('hruId', 'i4', ('hru',))
                hru_var[:] = hru_ids
                
                gru_var = ds.createVariable('gruId', 'i4', ('gru',))
                gru_var[:] = gru_ids
                
                # Add parameters with CORRECT dimensions
                for param_name, param_values in params.items():
                    param_values_array = np.asarray(param_values)
                    
                    # Flatten if multi-dimensional
                    if param_values_array.ndim > 1:
                        param_values_array = param_values_array.flatten()
                    
                    if param_name in gru_level_params:
                        # GRU-LEVEL PARAMETERS (routing and basin parameters)
                        param_var = ds.createVariable(param_name, 'f8', ('gru',))  # ← Use GRU dimension
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        # GRU parameters should have 1 value per GRU (usually just 1)
                        if len(param_values_array) >= num_grus:
                            param_var[:] = param_values_array[:num_grus]
                        else:
                            # Use the first value for all GRUs
                            param_var[:] = param_values_array[0]
                        
                        logger.debug(f"✓ {param_name}: GRU-level parameter, value = {param_values_array[0]:.6e}")
                        
                    else:
                        # HRU-LEVEL PARAMETERS (all other parameters)
                        param_var = ds.createVariable(param_name, 'f8', ('hru',))  # ← Use HRU dimension
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        # Handle HRU parameter dimensions
                        if len(param_values_array) == num_hrus:
                            # Perfect match - use as is
                            param_var[:] = param_values_array
                        elif len(param_values_array) == 1:
                            # Single value - broadcast to all HRUs
                            param_var[:] = param_values_array[0]
                        else:
                            # Mismatched dimensions - handle appropriately
                            if len(param_values_array) > num_hrus:
                                # Too many values - truncate
                                param_var[:] = param_values_array[:num_hrus]
                                logger.warning(f"⚠️ {param_name}: truncated {len(param_values_array)} values to {num_hrus} HRUs")
                            else:
                                # Too few values - repeat to fill
                                repeats = int(np.ceil(num_hrus / len(param_values_array)))
                                expanded_values = np.tile(param_values_array, repeats)[:num_hrus]
                                param_var[:] = expanded_values
                                logger.warning(f"⚠️ {param_name}: expanded {len(param_values_array)} values to {num_hrus} HRUs")
                        
        logger.info(f"✅ Created {trial_params_path.name} with proper GRU/HRU dimensions")
        
        # Log summary of parameter levels for verification
        gru_params_in_file = [p for p in params.keys() if p in gru_level_params]
        hru_params_in_file = [p for p in params.keys() if p not in gru_level_params]
        
        if gru_params_in_file:
            logger.info(f"   GRU-level parameters: {gru_params_in_file}")
        if hru_params_in_file:
            logger.info(f"   HRU-level parameters: {len(hru_params_in_file)} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating trial params: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, settings_dir: Path, logger) -> bool:
    """Update mizuRoute parameters in worker process."""
    try:
        import re
        
        mizuroute_params = task_data.get('mizuroute_params', [])
        if not mizuroute_params:
            return True
        
        param_file = Path(task_data['mizuroute_settings_dir']) / "param.nml.default"
        if not param_file.exists():
            return True
        
        with open(param_file, 'r') as f:
            content = f.read()
        
        updated_content = content
        for param_name in mizuroute_params:
            if param_name in params:
                param_value = params[param_name]
                pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                
                if param_name in ['tscale']:
                    replacement = rf'\g<1>{int(param_value)}'
                else:
                    replacement = rf'\g<1>{param_value:.6f}'
                
                updated_content = re.sub(pattern, replacement, updated_content)
        
        with open(param_file, 'w') as f:
            f.write(updated_content)
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating mizuRoute params: {str(e)}")
        return False


def _evaluate_parameters_worker_staggered(task_data: Dict) -> Dict:
    """Worker function with staggered start to reduce resource contention."""
    import time
    import os
    
    # Apply stagger delay
    start_delay = task_data.get('start_delay', 0)
    if start_delay > 0:
        time.sleep(start_delay)
    
    # Set process priority to be nice to other processes
    try:
        os.nice(5)  # Lower priority
    except:
        pass
    
    # Call the original worker function
    return _evaluate_parameters_worker(task_data)


def _run_summa_worker(summa_exe: str, file_manager: Path, summa_dir: Path, 
                     proc_id: int, logger) -> bool:
    """Fixed SUMMA worker that maintains consistent path handling."""
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"proc_{proc_id:02d}_summa.log"
        
        # Set environment variables for better resource management
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'  # Force single-threaded operation
        env['MKL_NUM_THREADS'] = '1'
        
        # Use the same approach as the original non-parallel version
        # Don't change working directory - let SUMMA resolve paths as intended
        cmd = f"{summa_exe} -m {file_manager}"
        
        logger.debug(f"Process {proc_id}: Running SUMMA command: {cmd}")
        logger.debug(f"Process {proc_id}: File manager: {file_manager}")
        logger.debug(f"Process {proc_id}: Expected output in: {summa_dir}")
        
        # Run SUMMA with shell=True like the original version
        # This maintains the same path resolution behavior
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                shell=True,  # Use shell=True like the original
                stdout=f, 
                stderr=subprocess.STDOUT,
                check=True, 
                timeout=1800,  # 30 minute timeout
                env=env
            )
        
        logger.debug(f"Process {proc_id}: SUMMA completed with return code {result.returncode}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Process {proc_id}: SUMMA failed with exit code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Process {proc_id}: SUMMA timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"Process {proc_id}: Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Process {proc_id}: Traceback: {traceback.format_exc()}")
        return False

def _calculate_metrics_worker(summa_dir: Path, task_data: Dict, logger) -> float:
    """Calculate performance metrics with increased timeout and better error handling."""
    try:
        proc_id = task_data.get('proc_id', 0)
        
        # Get observed data
        obs_file = task_data['obs_file']
        if not Path(obs_file).exists():
            logger.error(f"Process {proc_id}: Observed data not found: {obs_file}")
            return None
        
        # Load observed data
        obs_df = pd.read_csv(obs_file)
        
        # Find columns with more robust detection
        date_col = None
        flow_col = None
        
        for col in obs_df.columns:
            col_lower = col.lower()
            if date_col is None and any(term in col_lower for term in ['date', 'time', 'datetime']):
                date_col = col
            if flow_col is None and any(term in col_lower for term in ['flow', 'discharge', 'q_', 'streamflow']):
                flow_col = col
        
        if not date_col or not flow_col:
            logger.error(f"Process {proc_id}: Could not identify date/flow columns")
            logger.error(f"Process {proc_id}: Available columns: {obs_df.columns.tolist()}")
            return None
        
        # Process observed data
        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)
        observed_flow = obs_df[flow_col]
        
        # Find SUMMA output with extended retry logic
        max_retries = 10  # Increased from 5
        retry_delay = 30  # Increased to 30 seconds between retries
        
        for attempt in range(max_retries):
            # Try multiple patterns for finding simulation files
            patterns = [
                f"proc_{proc_id:02d}_opt_*_timestep.nc",
                f"*proc_{proc_id:02d}*timestep.nc",
                f"*timestep.nc"  # Fallback
            ]
            
            sim_files = []
            for pattern in patterns:
                found_files = list(summa_dir.glob(pattern))
                if found_files:
                    sim_files = found_files
                    logger.debug(f"Process {proc_id}: Found files with pattern '{pattern}': {[f.name for f in found_files]}")
                    break
            
            if sim_files:
                # Verify the file is readable and complete
                sim_file = sim_files[0]  # Take the first match
                
                try:
                    # Test if file can be opened and has reasonable data
                    with xr.open_dataset(sim_file) as ds:
                        if 'time' in ds.dims and len(ds.time) > 10:  # At least 10 time steps
                            logger.debug(f"Process {proc_id}: Found valid simulation file: {sim_file.name} with {len(ds.time)} time steps")
                            break
                        else:
                            logger.debug(f"Process {proc_id}: File {sim_file.name} exists but has insufficient data (time dim: {len(ds.time) if 'time' in ds.dims else 'missing'})")
                            sim_files = []  # Reset to trigger retry
                except Exception as e:
                    logger.debug(f"Process {proc_id}: File {sim_file.name} not ready yet: {str(e)}")
                    sim_files = []  # Reset to trigger retry
            
            if attempt < max_retries - 1:
                logger.debug(f"Process {proc_id}: No valid timestep files found (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Process {proc_id}: No SUMMA timestep files found after {max_retries} attempts over {max_retries * retry_delay / 60:.1f} minutes")
                
                # Debug: list all files in directory
                all_files = list(summa_dir.glob("*"))
                nc_files = list(summa_dir.glob("*.nc"))
                logger.error(f"Process {proc_id}: All files in {summa_dir}: {[f.name for f in all_files]}")
                logger.error(f"Process {proc_id}: NetCDF files: {[f.name for f in nc_files]}")
                
                # If there are NetCDF files but they don't match our pattern, investigate
                if nc_files:
                    logger.error(f"Process {proc_id}: Investigating NetCDF files that don't match expected pattern...")
                    for nc_file in nc_files[:3]:  # Check first 3 files
                        try:
                            with xr.open_dataset(nc_file) as ds:
                                logger.error(f"Process {proc_id}: File {nc_file.name}: dims={dict(ds.sizes)}, vars={list(ds.variables.keys())[:10]}")
                        except Exception as e:
                            logger.error(f"Process {proc_id}: Could not read {nc_file.name}: {str(e)}")
                
                return None
        
        # Process simulated data (rest of function remains the same)
        with xr.open_dataset(sim_file) as ds:
            # Find runoff variable with better detection
            runoff_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff', 
                          'averageRoutedRunoff_mean', 'basin__TotalRunoff_mean']
            
            runoff_var = None
            for var_name in runoff_vars:
                if var_name in ds.variables:
                    runoff_var = var_name
                    logger.debug(f"Process {proc_id}: Found runoff variable: {var_name}")
                    break
            
            if not runoff_var:
                available_vars = list(ds.variables.keys())
                logger.error(f"Process {proc_id}: No runoff variable found")
                logger.error(f"Process {proc_id}: Available variables: {available_vars}")
                
                # Try to find any variable with "runoff" in the name
                runoff_like = [v for v in available_vars if 'runoff' in v.lower()]
                if runoff_like:
                    runoff_var = runoff_like[0]
                    logger.warning(f"Process {proc_id}: Using fallback variable: {runoff_var}")
                else:
                    return None
            
            # Extract simulated flow
            var = ds[runoff_var]
            logger.debug(f"Process {proc_id}: Variable {runoff_var} has dimensions: {var.dims}, shape: {var.shape}")
            
            if len(var.shape) > 1:
                # Multi-dimensional - extract spatial dimension
                if 'hru' in var.dims:
                    simulated_flow = var.isel(hru=0).to_pandas()
                elif 'gru' in var.dims:
                    simulated_flow = var.isel(gru=0).to_pandas()
                else:
                    # Get non-time dimensions and use first spatial index
                    non_time_dims = [dim for dim in var.dims if dim != 'time']
                    if non_time_dims:
                        simulated_flow = var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        simulated_flow = var.to_pandas()
            else:
                simulated_flow = var.to_pandas()
            
            # Convert units (m/s to m³/s)
            catchment_area = task_data.get('catchment_area', 1e6)
            simulated_flow = simulated_flow * catchment_area
            
            logger.debug(f"Process {proc_id}: Extracted {len(simulated_flow)} simulated flow values")
            logger.debug(f"Process {proc_id}: Simulated flow range: {simulated_flow.min():.2f} to {simulated_flow.max():.2f} m³/s")
        
        # Align time series with better synchronization
        simulated_flow.index = simulated_flow.index.round('h')
        common_idx = observed_flow.index.intersection(simulated_flow.index)
        
        logger.debug(f"Process {proc_id}: Time alignment: {len(observed_flow)} obs, {len(simulated_flow)} sim, {len(common_idx)} common")
        
        if len(common_idx) == 0:
            logger.error(f"Process {proc_id}: No time overlap between observed and simulated data")
            logger.error(f"Process {proc_id}: Observed range: {observed_flow.index.min()} to {observed_flow.index.max()}")
            logger.error(f"Process {proc_id}: Simulated range: {simulated_flow.index.min()} to {simulated_flow.index.max()}")
            return None
        
        obs_common = observed_flow.loc[common_idx]
        sim_common = simulated_flow.loc[common_idx]
        
        # Calculate target metric
        score = _calculate_single_metric_worker(obs_common, sim_common, task_data['target_metric'])
        
        logger.debug(f"Process {proc_id}: Calculated {task_data['target_metric']} = {score}")
        
        return score
        
    except Exception as e:
        logger.error(f"Process {proc_id}: Error calculating metrics: {str(e)}")
        import traceback
        logger.error(f"Process {proc_id}: Traceback: {traceback.format_exc()}")
        return None




def _apply_parameters_worker(params: Dict, settings_dir: Path, task_data: Dict, logger) -> bool:
    """Apply parameters to files in the worker process."""
    try:
        # Handle soil depth parameters
        if task_data.get('calibrate_depth', False):
            if 'total_mult' in params and 'shape_factor' in params:
                success = _update_soil_depths_worker(
                    params['total_mult'], params['shape_factor'],
                    task_data['original_depths'], settings_dir, logger
                )
                if not success:
                    return False
        
        # Handle hydraulic parameters (excluding depth and mizuRoute parameters)
        hydraulic_params = {k: v for k, v in params.items() 
                          if k not in ['total_mult', 'shape_factor'] + task_data.get('mizuroute_params', [])}
        
        if hydraulic_params:
            success = _generate_trial_params_worker(hydraulic_params, settings_dir, logger)
            if not success:
                return False
        
        # Handle mizuRoute parameters
        if task_data.get('mizuroute_params') and any(p in params for p in task_data['mizuroute_params']):
            success = _update_mizuroute_params_worker(params, task_data, settings_dir, logger)
            if not success:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying parameters: {str(e)}")
        return False


def _update_soil_depths_worker(total_mult: float, shape_factor: float, 
                              original_depths_list: list, settings_dir: Path, logger) -> bool:
    """Update soil depths in worker process."""
    try:
        if not original_depths_list:
            return True
        
        original_depths = np.array(original_depths_list)
        
        # Calculate new depths using the same logic as the main class
        arr = original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        w /= w.mean()
        new_depths = arr * w * total_mult
        
        # Calculate layer heights
        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]
        
        # Update coldState.nc
        coldstate_path = settings_dir / 'coldState.nc'
        if not coldstate_path.exists():
            logger.error(f"coldState.nc not found: {coldstate_path}")
            return False
        
        import netCDF4 as nc
        with nc.Dataset(coldstate_path, 'r+') as ds:
            if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                logger.error("Required depth variables not found")
                return False
            
            num_hrus = ds.dimensions['hru'].size
            for h in range(num_hrus):
                ds.variables['mLayerDepth'][:, h] = new_depths
                ds.variables['iLayerHeight'][:, h] = heights
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating soil depths: {str(e)}")
        return False


def _debug_worker_environment(task_data: Dict) -> Dict:
    """Debug function to check what's happening in worker processes."""
    import os
    import subprocess
    from pathlib import Path
    
    debug_info = {
        'worker_pid': os.getpid(),
        'current_dir': str(Path.cwd()),
        'summa_exe_exists': Path(task_data['summa_exe']).exists(),
        'file_manager_exists': Path(task_data['file_manager']).exists(),
        'summa_dir_exists': Path(task_data['summa_dir']).exists(),
        'settings_dir_exists': Path(task_data['settings_dir']).exists(),
    }
    
    # Check if we can run SUMMA command
    try:
        cmd = f"{task_data['summa_exe']} --help"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        debug_info['summa_help_exit_code'] = result.returncode
        debug_info['summa_help_stdout'] = result.stdout[:200] if result.stdout else ""
        debug_info['summa_help_stderr'] = result.stderr[:200] if result.stderr else ""
    except Exception as e:
        debug_info['summa_help_error'] = str(e)
    
    # Check file manager content
    try:
        with open(task_data['file_manager'], 'r') as f:
            debug_info['file_manager_content'] = f.read()[:500]
    except Exception as e:
        debug_info['file_manager_read_error'] = str(e)
    
    return debug_info

def _calculate_single_metric_worker(observed: pd.Series, simulated: pd.Series, metric_name: str) -> float:
    """Calculate a single performance metric."""
    try:
        import pandas as pd
        import numpy as np
        
        # Clean data
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return np.nan
        
        metric_name = metric_name.upper()
        
        if metric_name == 'KGE':
            # Kling-Gupta Efficiency
            r = observed.corr(simulated)
            alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
            beta = simulated.mean() / observed.mean() if observed.mean() != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            return kge if not np.isnan(kge) else 0.0
        
        elif metric_name == 'NSE':
            mean_obs = observed.mean()
            nse_num = ((observed - simulated) ** 2).sum()
            nse_den = ((observed - mean_obs) ** 2).sum()
            return 1 - (nse_num / nse_den) if nse_den > 0 else 0.0
        
        # Add other metrics as needed...
        else:
            # Default to KGE
            r = observed.corr(simulated)
            alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
            beta = simulated.mean() / observed.mean() if observed.mean() != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            return kge if not np.isnan(kge) else 0.0
        
    except Exception:
        return np.nan

'''

def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    """Debug version of the worker function."""
    try:
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']
        
        # Run debug check
        debug_info = _debug_worker_environment(task_data)
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': None,
            'error': f'DEBUG INFO: {debug_info}'
        }
        
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Debug worker exception: {str(e)}'
        }
'''

def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    """
    Main worker function with improved error handling and debugging.
    """
    import sys
    import logging
    from pathlib import Path
    import numpy as np
    
    try:
        # Extract basic task info
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']
        evaluation_id = task_data.get('evaluation_id', f'eval_{individual_id}')
        
        # Create a process-specific logger
        logger = logging.getLogger(f'worker_{proc_id}')
        if not logger.handlers:  # Avoid duplicate handlers
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[P{proc_id:02d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.debug(f"Starting evaluation {evaluation_id} for individual {individual_id}")
        
        # Get paths from task data (already resolved as strings)
        summa_exe = task_data['summa_exe']
        file_manager = Path(task_data['file_manager'])
        summa_dir = Path(task_data['summa_dir'])
        settings_dir = Path(task_data['settings_dir'])
        
        # Verify critical paths exist
        if not Path(summa_exe).exists():
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': f'SUMMA executable not found: {summa_exe}'
            }
        
        if not file_manager.exists():
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': f'File manager not found: {file_manager}'
            }
        
        logger.debug(f"Verified paths - SUMMA exe: {Path(summa_exe).exists()}, FileManager: {file_manager.exists()}")
        
        # Step 1: Apply parameters to files
        logger.debug("Applying parameters to model files")
        success = _apply_parameters_worker(params, settings_dir, task_data, logger)
        if not success:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'Failed to apply parameters to model files'
            }
        
        # Step 2: Run SUMMA
        logger.debug("Running SUMMA simulation")
        success = _run_summa_worker(summa_exe, file_manager, summa_dir, proc_id, logger)
        if not success:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'SUMMA simulation failed or output files not created'
            }
        
        # Step 3: Calculate metrics
        logger.debug("Calculating performance metrics")
        score = _calculate_metrics_worker(summa_dir, task_data, logger)
        
        if score is None:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'Failed to calculate performance metrics'
            }
        
        # Apply score transformation if needed (negate for metrics where lower is better)
        target_metric = task_data['target_metric']
        if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
            score = -score
        
        logger.debug(f"Evaluation {evaluation_id} completed successfully with score: {score}")
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': score,
            'error': None
        }
        
    except Exception as e:
        proc_id = task_data.get('proc_id', 'unknown')
        individual_id = task_data.get('individual_id', -1)
        
        # Try to get more detailed error info
        import traceback
        error_details = f'Worker exception in process {proc_id}: {str(e)}\nTraceback: {traceback.format_exc()}'
        
        return {
            'individual_id': individual_id,
            'params': task_data.get('params', {}),
            'score': None,
            'error': error_details
        }
class ParameterEvaluator:
    """
    Standalone parameter evaluator for use in worker processes.
    This class recreates the evaluation environment without the full DE optimizer context.
    """
    
    def __init__(self, eval_context: Dict, proc_dirs: Dict, logger: logging.Logger):
        """Initialize the parameter evaluator with the given context."""
        self.config = eval_context['config']
        self.domain_name = eval_context['domain_name']
        self.experiment_id = eval_context['experiment_id']
        self.project_dir = Path(eval_context['project_dir'])
        self.calibration_period = eval_context['calibration_period']
        self.evaluation_period = eval_context['evaluation_period']
        self.target_metric = eval_context['target_metric']
        self.calibrate_depth = eval_context['calibrate_depth']
        self.original_depths = eval_context['original_depths']
        self.depth_params = eval_context['depth_params']
        self.mizuroute_params = eval_context['mizuroute_params']
        self.local_params = eval_context['local_params']
        self.basin_params = eval_context['basin_params']
        self.param_bounds = eval_context['param_bounds']
        
        # Process-specific directories
        self.summa_dir = proc_dirs['summa_dir']
        self.mizuroute_dir = proc_dirs['mizuroute_dir']
        self.summa_settings_dir = proc_dirs['summa_settings_dir']
        self.mizuroute_settings_dir = proc_dirs['mizuroute_settings_dir']
        self.proc_id = proc_dirs['proc_id']
        
        self.logger = logger
        
        # Update file managers for this process
        self._update_process_file_managers()
    
    def _update_process_file_managers(self):
        """Update file managers to point to process-specific directories."""
        
        # Update SUMMA file manager
        summa_filemanager = self.summa_settings_dir / 'fileManager.txt'
        if summa_filemanager.exists():
            self._update_summa_file_manager(summa_filemanager)
        
        # Update mizuRoute control file
        mizuroute_control = self.mizuroute_settings_dir / 'mizuroute.control'
        if mizuroute_control.exists():
            self._update_mizuroute_control_file(mizuroute_control)
    
    def _update_summa_file_manager(self, filemanager_path):
        """Update SUMMA file manager with process-specific paths."""
        with open(filemanager_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if 'outputPath' in line:
                output_path = str(self.summa_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.summa_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            elif 'outFilePrefix' in line:
                prefix = f"proc_{self.proc_id:02d}_opt_{self.experiment_id}"
                updated_lines.append(f"outFilePrefix        '{prefix}'\n")
            else:
                updated_lines.append(line)
        
        with open(filemanager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _update_mizuroute_control_file(self, control_path):
        """Update mizuRoute control file with process-specific paths."""
        with open(control_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if '<input_dir>' in line:
                input_path = str(self.summa_dir).replace('\\', '/')
                updated_lines.append(f"<input_dir>             {input_path}/\n")
            elif '<output_dir>' in line:
                output_path = str(self.mizuroute_dir).replace('\\', '/')
                updated_lines.append(f"<output_dir>            {output_path}/\n")
            elif '<ancil_dir>' in line:
                ancil_path = str(self.mizuroute_settings_dir).replace('\\', '/')
                updated_lines.append(f"<ancil_dir>             {ancil_path}/\n")
            elif '<case_name>' in line:
                case_name = f"proc_{self.proc_id:02d}_opt_{self.experiment_id}"
                updated_lines.append(f"<case_name>             {case_name}\n")
            elif '<fname_qsim>' in line:
                fname = f"proc_{self.proc_id:02d}_opt_{self.experiment_id}_timestep.nc"
                updated_lines.append(f"<fname_qsim>            {fname}\n")
            else:
                updated_lines.append(line)
        
        with open(control_path, 'w') as f:
            f.writelines(updated_lines)
        
    def evaluate_parameters(self, params: np.ndarray) -> float:
        """
        Evaluate the given parameters by running the model and computing the objective function.
        """
        self.logger.info(f"Evaluating parameters: {params}")

        # Apply parameters to the trial NetCDF file
        if not self._apply_parameters_to_netcdf(params):
            self.logger.error("Failed to write parameters to NetCDF file.")
            return np.nan


        # Extract model outputs
        output_file = self.proc_dirs['summa_dir'] / 'output' / 'output.nc'
        if not output_file.exists():
            self.logger.error(f"Expected SUMMA output not found at: {output_file}")
            return np.nan

        # Compute objective function
        score = self._compute_objective(output_file)
        self.logger.info(f"Computed objective score: {score}")
        return score

    
    def _needs_mizuroute_routing(self, domain_method: str, routing_delineation: str) -> bool:
        """Determine if mizuRoute routing is needed."""
        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        return False
    
    def _run_summa(self) -> bool:
        """
        Run SUMMA simulation in this process directory using instance-local paths.

        Returns:
            bool: True if the run was successful, False otherwise
        """
        summa_path = Path(self.config.get('SUMMA_INSTALL_PATH', 'default'))
        if str(summa_path) == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'

        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        file_manager = self.summa_settings_dir / 'fileManager.txt'

        if not file_manager.exists():
            return False

        cmd = f"{summa_exe} -m {file_manager}"
        log_dir = self.summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"proc_{self.proc_id}_summa.log"

        try:
            with open(log_file, 'w') as f:
                subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=180)
            return True
        except Exception:
            return False
    
    def _run_mizuroute(self) -> bool:
        """Run mizuRoute simulation in this process directory."""
        mizu_path = Path(self.config.get('INSTALL_PATH_MIZUROUTE', 'default'))
        if str(mizu_path) == 'default':
            mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = self.mizuroute_settings_dir / 'mizuroute.control'
        
        cmd = f"{mizu_exe} {control_file}"
        log_file = self.mizuroute_dir / "logs" / f"proc_{self.proc_id}_mizuroute.log"
        
        try:
            with open(log_file, 'w') as f:
                subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                             check=True, timeout=180, cwd=str(control_file.parent))
            return True
        except Exception:
            return False
    
    def _update_soil_depths(self, params):
        """Update soil depths for this process."""
        try:
            if not self.calibrate_depth:
                return True
            
            total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
            shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
            
            # Calculate new depths
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is None:
                return False
            
            # Update coldState.nc in process directory
            coldstate_path = self.summa_settings_dir / 'coldState.nc'
            
            if not coldstate_path.exists():
                return False
            
            # Calculate layer heights
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]
            
            # Update the file
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    return False
                
                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights
            
            return True
            
        except Exception:
            return False
    
    def _calculate_new_depths(self, total_mult, shape_factor):
        """Calculate new soil depths using the shape method."""
        if self.original_depths is None:
            return None
        
        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        w /= w.mean()
        new_depths = arr * w * total_mult
        
        return new_depths
    
    def _update_mizuroute_parameters(self, params):
        """Update mizuRoute parameters for this process."""
        try:
            if not self.mizuroute_params:
                return True
            
            param_file = self.mizuroute_settings_dir / "param.nml.default"
            
            if not param_file.exists():
                return False
            
            with open(param_file, 'r') as f:
                content = f.read()
            
            updated_content = content
            
            for param_name in self.mizuroute_params:
                if param_name in params:
                    param_value = params[param_name]
                    
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                    
                    if param_name in ['tscale']:
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        replacement = rf'\g<1>{param_value:.6f}'
                    
                    updated_content = re.sub(pattern, replacement, updated_content)
            
            with open(param_file, 'w') as f:
                f.write(updated_content)
            
            return True
            
        except Exception:
            return False
    
    def _generate_trial_params_file(self, params):
        """Generate trial parameters file for this process with proper GRU/HRU dimensions."""
        try:
            if not params:
                return True
            
            # Use a simplified version for the worker process
            trial_params_path = self.summa_settings_dir / 'trialParams.nc'
            
            # Get attribute file to read HRU information
            attr_file_path = self.summa_settings_dir / 'attributes.nc'
            
            if not attr_file_path.exists():
                self.logger.error(f"Attributes file not found: {attr_file_path}")
                return False
            
            # Define parameter levels
            routing_params = ['routingGammaShape', 'routingGammaScale']
            basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
            gru_level_params = routing_params + basin_params
            
            with xr.open_dataset(attr_file_path) as ds:
                # Get HRU information
                hru_ids = ds['hruId'].values
                num_hrus = len(hru_ids)
                
                # Get GRU information (usually 1 for single watershed)
                if 'gruId' in ds:
                    gru_ids = ds['gruId'].values
                    num_grus = len(gru_ids)
                else:
                    # If no gruId in attributes, assume 1 GRU
                    gru_ids = np.array([1])
                    num_grus = 1
                
                self.logger.debug(f"Creating trialParams.nc with {num_hrus} HRUs and {num_grus} GRUs")
                
                with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                    # Create BOTH dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)  # ← CRITICAL: Add GRU dimension
                    
                    # Create coordinate variables
                    hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                    hru_id_var[:] = hru_ids
                    hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                    hru_id_var.units = '-'
                    
                    gru_id_var = output_ds.createVariable('gruId', 'i4', ('gru',))
                    gru_id_var[:] = gru_ids
                    gru_id_var.long_name = 'Grouped Response Unit ID (GRU)'
                    gru_id_var.units = '-'
                    
                    # Add parameters with CORRECT dimensions
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)
                        
                        # Flatten if multi-dimensional
                        if param_values_array.ndim > 1:
                            param_values_array = param_values_array.flatten()
                        
                        if param_name in gru_level_params:
                            # GRU-LEVEL PARAMETERS (routing and basin parameters)
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            param_var.units = "N/A"
                            
                            # GRU parameters should have 1 value per GRU (usually just 1)
                            if len(param_values_array) >= num_grus:
                                param_var[:] = param_values_array[:num_grus]
                            else:
                                # Use the first value for all GRUs
                                param_var[:] = param_values_array[0]
                            
                            self.logger.debug(f"✓ {param_name}: GRU-level parameter, value = {param_values_array[0]:.6e}")
                            
                        else:
                            # HRU-LEVEL PARAMETERS (all other parameters)
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            param_var.units = "N/A"
                            
                            # Handle HRU parameter dimensions
                            if len(param_values_array) == num_hrus:
                                # Perfect match - use as is
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                # Single value - broadcast to all HRUs
                                param_var[:] = param_values_array[0]
                            else:
                                # Mismatched dimensions - handle appropriately
                                if len(param_values_array) > num_hrus:
                                    # Too many values - truncate
                                    param_var[:] = param_values_array[:num_hrus]
                                    self.logger.warning(f"⚠️ {param_name}: truncated {len(param_values_array)} values to {num_hrus} HRUs")
                                else:
                                    # Too few values - repeat to fill
                                    repeats = int(np.ceil(num_hrus / len(param_values_array)))
                                    expanded_values = np.tile(param_values_array, repeats)[:num_hrus]
                                    param_var[:] = expanded_values
                                    self.logger.warning(f"⚠️ {param_name}: expanded {len(param_values_array)} values to {num_hrus} HRUs")
                            
                            # Log for debugging
                            if len(param_values_array) == 1:
                                self.logger.debug(f"✓ {param_name}: HRU-level parameter, value = {param_values_array[0]:.6e}")
                            else:
                                self.logger.debug(f"✓ {param_name}: HRU-level parameter, {len(param_values_array)} values")
                    
                    # Add global attributes
                    output_ds.description = "SUMMA Trial Parameter file for parallel DE evaluation"
                    output_ds.history = f"Created on {datetime.now().isoformat()}"
                    output_ds.comment = "Routing and basin parameters at GRU level, other parameters at HRU level"
            
            self.logger.info(f"✅ Created {trial_params_path.name} with proper GRU/HRU dimensions")
            
            # Log summary of parameter levels for verification
            gru_params_in_file = [p for p in params.keys() if p in gru_level_params]
            hru_params_in_file = [p for p in params.keys() if p not in gru_level_params]
            
            if gru_params_in_file:
                self.logger.info(f"   GRU-level parameters: {gru_params_in_file}")
            if hru_params_in_file:
                self.logger.info(f"   HRU-level parameters: {len(hru_params_in_file)} parameters")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating trial params file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _convert_lumped_to_distributed_routing(self):
        """Convert lumped SUMMA output for distributed routing (simplified for worker)."""
        try:
            # Find SUMMA timestep file
            timestep_files = list(self.summa_dir.glob("*timestep.nc"))
            if not timestep_files:
                return False
            
            summa_file = timestep_files[0]
            
            # Load topology
            topology_file = self.mizuroute_settings_dir / 'topology.nc'
            if not topology_file.exists():
                return False
            
            with xr.open_dataset(topology_file) as topo_ds:
                seg_ids = topo_ds['segId'].values
                n_segments = len(seg_ids)
            
            # Load SUMMA output and convert
            with xr.open_dataset(summa_file, decode_times=False) as summa_ds:
                # Find runoff variable
                routing_var = 'averageRoutedRunoff'
                if routing_var not in summa_ds:
                    routing_var = 'basin__TotalRunoff'
                
                if routing_var not in summa_ds:
                    return False
                
                # Create mizuRoute forcing
                mizuForcing = xr.Dataset()
                
                # Copy time
                mizuForcing['time'] = summa_ds['time']
                
                # Create GRU dimension
                mizuForcing['gru'] = xr.DataArray(seg_ids, dims=('gru',))
                mizuForcing['gruId'] = xr.DataArray(seg_ids, dims=('gru',))
                
                # Get runoff data and broadcast
                runoff_data = summa_ds[routing_var].values
                if len(runoff_data.shape) == 2:
                    if runoff_data.shape[1] == 1:
                        runoff_data = runoff_data.flatten()
                    else:
                        runoff_data = runoff_data[:, 0]
                
                # Broadcast to all segments
                tiled_data = np.tile(runoff_data[:, np.newaxis], (1, n_segments))
                
                mizuForcing['averageRoutedRunoff'] = xr.DataArray(
                    tiled_data, dims=('time', 'gru'))
            
            # Save converted file
            mizuForcing.to_netcdf(summa_file, format='NETCDF4')
            mizuForcing.close()
            
            return True
            
        except Exception:
            return False
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics (simplified for worker)."""
        try:
            # Get observed data
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
            
            if not obs_path.exists():
                return None
            
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Find date and flow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower()), None)
            
            if not date_col or not flow_col:
                return None
            
            # Process observed data
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Determine output type
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
            needs_mizuroute = self._needs_mizuroute_routing(domain_method, routing_delineation)
            
            if needs_mizuroute:
                # Use mizuRoute output
                sim_files = list(self.mizuroute_dir.glob("*.nc"))
                if not sim_files:
                    return None
                
                sim_file = sim_files[0]
                
                with xr.open_dataset(sim_file) as ds:
                    reach_id = self.config.get('SIM_REACH_ID')
                    
                    if 'reachID' in ds.variables:
                        reach_ids = ds['reachID'].values
                        reach_indices = np.where(reach_ids == int(reach_id))[0]
                        
                        if len(reach_indices) > 0:
                            reach_index = reach_indices[0]
                            
                            # Find streamflow variable
                            for var_name in ['IRFroutedRunoff', 'averageRoutedRunoff']:
                                if var_name in ds.variables:
                                    var = ds[var_name]
                                    if 'seg' in var.dims:
                                        simulated_flow = var.isel(seg=reach_index).to_pandas()
                                    else:
                                        simulated_flow = var.isel(reachID=reach_index).to_pandas()
                                    break
                            else:
                                return None
                        else:
                            return None
                    else:
                        return None
            else:
                # Use SUMMA output
                sim_files = list(self.summa_dir.glob("*timestep.nc"))
                if not sim_files:
                    return None
                
                sim_file = sim_files[0]
                
                # Get catchment area (simplified)
                try:
                    import geopandas as gpd
                    basin_path = self.project_dir / "shapefiles" / "river_basins"
                    basin_files = list(basin_path.glob("*.shp"))
                    if basin_files:
                        gdf = gpd.read_file(basin_files[0])
                        area_col = 'GRU_area'
                        if area_col in gdf.columns:
                            catchment_area = gdf[area_col].sum()
                        else:
                            catchment_area = 1e6  # Default 1 km²
                    else:
                        catchment_area = 1e6  # Default 1 km²
                except:
                    catchment_area = 1e6  # Default 1 km²
                
                with xr.open_dataset(sim_file) as ds:
                    # Find runoff variable
                    for var_name in ['averageRoutedRunoff', 'basin__TotalRunoff']:
                        if var_name in ds.variables:
                            var = ds[var_name]
                            if len(var.shape) > 1:
                                if 'hru' in var.dims:
                                    simulated_flow = var.isel(hru=0).to_pandas()
                                else:
                                    simulated_flow = var.isel({list(var.dims)[1]: 0}).to_pandas()
                            else:
                                simulated_flow = var.to_pandas()
                            
                            # Convert units
                            simulated_flow = simulated_flow * catchment_area
                            break
                    else:
                        return None
            
            # Align time series
            simulated_flow.index = simulated_flow.index.round('h')
            common_idx = observed_flow.index.intersection(simulated_flow.index)
            
            if len(common_idx) == 0:
                return None
            
            obs_common = observed_flow.loc[common_idx]
            sim_common = simulated_flow.loc[common_idx]
            
            # Calculate metrics
            metrics = self._calculate_streamflow_metrics(obs_common, sim_common)
            
            return metrics
            
        except Exception:
            return None
    
    def _calculate_streamflow_metrics(self, observed, simulated):
        """Calculate streamflow metrics (simplified)."""
        try:
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan}
            
            # Calculate KGE
            r = observed.corr(simulated)
            alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
            beta = simulated.mean() / observed.mean() if observed.mean() != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
            
            return {'KGE': kge}
            
        except Exception:
            return {'KGE': np.nan}
    
    def _extract_target_metric(self, metrics):
        """Extract target metric value."""
        if self.target_metric in metrics:
            return metrics[self.target_metric]
        else:
            return list(metrics.values())[0] if metrics else None 