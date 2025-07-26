#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CONFLUENCE Differential Evolution Optimizer - Refactored Architecture

This module provides parameter optimization for CONFLUENCE hydrological models using
Differential Evolution with support for multiple calibration targets (streamflow, snow, etc.),
parallel processing, and comprehensive parameter management.

Features:
- Multi-target calibration (streamflow, snow SWE)
- Parallel processing with process isolation
- Soil depth calibration using shape method
- mizuRoute routing parameter optimization
- Parameter persistence and restart capabilities
- Comprehensive results tracking and visualization

Usage:
    from de_optimizer import DEOptimizer
    
    optimizer = DEOptimizer(config, logger)
    results = optimizer.run_de_optimization()
"""

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
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
import time
from concurrent.futures import ProcessPoolExecutor
import re

def _evaluate_parameters_worker_safe(task_data: Dict) -> Dict:
    """Enhanced safe worker function with better error reporting and debugging"""
    import os
    import gc
    import signal
    import sys
    import time
    import random
    import traceback
    
    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set process-specific environment for isolation
    process_id = os.getpid()
    
    # Force single-threaded execution and disable problematic file locking
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    })
    
    # Add small random delay to stagger file system access
    initial_delay = random.uniform(0.1, 0.8)
    time.sleep(initial_delay)
    
    try:
        # Force garbage collection at start
        gc.collect()
        
        # Enhanced logging setup for debugging
        proc_id = task_data.get('proc_id', 0)
        individual_id = task_data.get('individual_id', -1)
        
        # Try the evaluation with basic retry for stale file handle errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    retry_delay = 2.0 * (attempt + 1) + random.uniform(0, 1)
                    time.sleep(retry_delay)
                    gc.collect()
                
                # Call the enhanced worker function
                result = _evaluate_parameters_worker_enhanced(task_data)
                
                # Force cleanup
                gc.collect()
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                error_trace = traceback.format_exc()
                
                # Check for stale file handle or similar filesystem errors
                if any(term in error_str for term in ['stale file handle', 'errno 116', 'input/output error', 'errno 5']):
                    if attempt < max_retries - 1:  # Not the last attempt
                        continue
                
                # For other errors or final attempt, return the error with full traceback
                return {
                    'individual_id': individual_id,
                    'params': task_data.get('params', {}),
                    'score': None,
                    'error': f'Worker exception (attempt {attempt + 1}): {str(e)}\nTraceback:\n{error_trace}',
                    'proc_id': proc_id,
                    'debug_info': {
                        'attempt': attempt + 1,
                        'max_retries': max_retries,
                        'process_id': process_id
                    }
                }
        
        # If we get here, all retries failed
        return {
            'individual_id': individual_id,
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Worker failed after {max_retries} attempts',
            'proc_id': proc_id
        }
        
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Critical worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
    
    finally:
        # Final cleanup
        gc.collect()


def _evaluate_parameters_worker_enhanced(task_data: Dict) -> Dict:
    """Enhanced worker function with better debugging and error handling"""
    import sys
    import logging
    from pathlib import Path
    import subprocess
    import traceback
    
    # Enhanced error collection
    debug_info = {
        'stage': 'initialization',
        'files_checked': [],
        'commands_run': [],
        'errors': []
    }
    
    try:
        # Extract task info
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']
        
        debug_info['individual_id'] = individual_id
        debug_info['proc_id'] = proc_id
        
        # Setup process logger with more detail
        logger = logging.getLogger(f'worker_{proc_id}_{individual_id}')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)  # More verbose logging
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[P{proc_id:02d}-I{individual_id:03d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting evaluation of individual {individual_id}")
        
        # Convert all paths to absolute Path objects early
        debug_info['stage'] = 'path_setup'
        
        summa_exe = Path(task_data['summa_exe']).resolve()
        file_manager = Path(task_data['file_manager']).resolve()
        summa_dir = Path(task_data['summa_dir']).resolve()
        mizuroute_dir = Path(task_data['mizuroute_dir']).resolve()
        summa_settings_dir = Path(task_data['summa_settings_dir']).resolve()
        
        # Log paths for debugging
        logger.debug(f"SUMMA exe: {summa_exe}")
        logger.debug(f"File manager: {file_manager}")
        logger.debug(f"SUMMA dir: {summa_dir}")
        logger.debug(f"Settings dir: {summa_settings_dir}")
        
        debug_info['paths'] = {
            'summa_exe': str(summa_exe),
            'file_manager': str(file_manager),
            'summa_dir': str(summa_dir),
            'summa_settings_dir': str(summa_settings_dir)
        }
        
        # Verify critical paths exist
        debug_info['stage'] = 'path_verification'
        
        critical_paths = {
            'SUMMA executable': summa_exe,
            'File manager': file_manager,
            'SUMMA directory': summa_dir,
            'Settings directory': summa_settings_dir
        }
        
        for name, path in critical_paths.items():
            debug_info['files_checked'].append(f"{name}: {path}")
            if not path.exists():
                error_msg = f'{name} not found: {path}'
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'error': error_msg,
                    'debug_info': debug_info
                }
        
        logger.info("All critical paths verified")
        
        # Apply parameters to files
        debug_info['stage'] = 'parameter_application'
        logger.info("Applying parameters to model files")
        
        if not _apply_parameters_worker_enhanced(params, task_data, summa_settings_dir, logger, debug_info):
            error_msg = 'Failed to apply parameters'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info("Parameters applied successfully")
        
        # Run SUMMA with enhanced debugging
        debug_info['stage'] = 'summa_execution'
        logger.info("Starting SUMMA execution")
        
        if not _run_summa_worker_enhanced(summa_exe, file_manager, summa_dir, logger, debug_info):
            error_msg = 'SUMMA simulation failed'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info("SUMMA execution completed successfully")
        
        # Run mizuRoute if needed
        debug_info['stage'] = 'mizuroute_check'
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        if calibration_var == 'streamflow':
            config = task_data['config']
            needs_routing = _needs_mizuroute_routing_worker(config)
            
            if needs_routing:
                debug_info['stage'] = 'mizuroute_execution'
                logger.info("Starting mizuRoute execution")
                
                if not _run_mizuroute_worker_enhanced(task_data, mizuroute_dir, logger, debug_info):
                    error_msg = 'mizuRoute simulation failed'
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': error_msg,
                        'debug_info': debug_info
                    }
                
                logger.info("mizuRoute execution completed successfully")
        
        # Calculate metrics
        debug_info['stage'] = 'metrics_calculation'
        logger.info("Calculating performance metrics")
        
        score = _calculate_metrics_worker_enhanced(task_data, summa_dir, mizuroute_dir, logger, debug_info)
        
        if score is None:
            error_msg = 'Failed to calculate metrics'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info(f"Evaluation completed successfully. Score: {score:.6f}")
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': score,
            'error': None,
            'debug_info': debug_info
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f'Worker exception at stage {debug_info.get("stage", "unknown")}: {str(e)}'
        debug_info['errors'].append(f"{error_msg}\nTraceback:\n{error_trace}")
        
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': error_msg,
            'debug_info': debug_info,
            'full_traceback': error_trace
        }

def _apply_parameters_worker_enhanced(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced parameter application with better error handling"""
    try:
        config = task_data['config']
        logger.debug(f"Applying parameters: {list(params.keys())}")
        
        # Handle soil depth parameters
        if config.get('CALIBRATE_DEPTH', False) and 'total_mult' in params and 'shape_factor' in params:
            logger.debug("Updating soil depths")
            if not _update_soil_depths_worker_enhanced(params, task_data, settings_dir, logger, debug_info):
                return False
        
        # Handle mizuRoute parameters
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
            if mizuroute_params and any(p in params for p in mizuroute_params):
                logger.debug("Updating mizuRoute parameters")
                if not _update_mizuroute_params_worker_enhanced(params, task_data, logger, debug_info):
                    return False
        
        # Generate trial parameters file
        depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()] if config.get('CALIBRATE_MIZUROUTE', False) else []
        
        hydraulic_params = {k: v for k, v in params.items() 
                          if k not in depth_params + mizuroute_params}
        
        if hydraulic_params:
            logger.debug(f"Generating trial parameters file with: {list(hydraulic_params.keys())}")
            if not _generate_trial_params_worker_enhanced(hydraulic_params, settings_dir, logger, debug_info):
                return False
        
        logger.debug("Parameter application completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error applying parameters: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

def _update_soil_depths_worker_enhanced(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced soil depth update with better error handling"""
    try:
        original_depths_list = task_data.get('original_depths')
        if not original_depths_list:
            logger.warning("No original depths provided, skipping soil depth update")
            return True
        
        original_depths = np.array(original_depths_list)
        
        total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
        shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
        
        logger.debug(f"Updating soil depths: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")
        
        # Calculate new depths
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
        
        # Calculate heights
        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]
        
        # Update coldState.nc
        coldstate_path = settings_dir / 'coldState.nc'
        if not coldstate_path.exists():
            error_msg = f"coldState.nc not found: {coldstate_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].append(f"coldState.nc: {coldstate_path}")
        
        with nc.Dataset(coldstate_path, 'r+') as ds:
            if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                error_msg = "Required depth variables not found in coldState.nc"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return False
            
            num_hrus = ds.dimensions['hru'].size
            for h in range(num_hrus):
                ds.variables['mLayerDepth'][:, h] = new_depths
                ds.variables['iLayerHeight'][:, h] = heights
        
        logger.debug("Soil depths updated successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error updating soil depths: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_mizuroute_params_worker_enhanced(params: Dict, task_data: Dict, logger, debug_info: Dict) -> bool:
    """Enhanced mizuRoute parameter update with better error handling"""
    try:
        config = task_data['config']
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        
        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        param_file = mizuroute_settings_dir / "param.nml.default"
        
        if not param_file.exists():
            logger.warning(f"mizuRoute param file not found: {param_file}")
            return True
        
        debug_info['files_checked'].append(f"mizuRoute param file: {param_file}")
        
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
                logger.debug(f"Updated {param_name} = {param_value}")
        
        with open(param_file, 'w') as f:
            f.write(updated_content)
        
        logger.debug("mizuRoute parameters updated successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error updating mizuRoute params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _generate_trial_params_worker_enhanced(params: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced trial parameters generation with better error handling and file locking"""
    import time
    import random
    import os
    
    try:
        if not params:
            logger.debug("No hydraulic parameters to write")
            return True
        
        trial_params_path = settings_dir / 'trialParams.nc'
        attr_file_path = settings_dir / 'attributes.nc'
        
        if not attr_file_path.exists():
            error_msg = f"Attributes file not found: {attr_file_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].append(f"attributes.nc: {attr_file_path}")
        
        # Add retry logic with file locking
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Create temporary file first, then move it
                temp_path = trial_params_path.with_suffix(f'.tmp_{os.getpid()}_{random.randint(1000,9999)}')
                
                logger.debug(f"Attempt {attempt + 1}: Writing trial parameters to {temp_path}")
                
                # Define parameter levels
                routing_params = ['routingGammaShape', 'routingGammaScale']
                basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
                gru_level_params = routing_params + basin_params
                
                with xr.open_dataset(attr_file_path) as ds:
                    num_hrus = ds.sizes.get('hru', 1)
                    num_grus = ds.sizes.get('gru', 1)
                    hru_ids = ds['hruId'].values if 'hruId' in ds else np.arange(1, num_hrus + 1)
                    gru_ids = ds['gruId'].values if 'gruId' in ds else np.array([1])
                
                logger.debug(f"Writing parameters for {num_hrus} HRUs, {num_grus} GRUs")
                
                # Write to temporary file with exclusive access
                with nc.Dataset(temp_path, 'w', format='NETCDF4') as output_ds:
                    # Create dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)
                    
                    # Create coordinate variables
                    hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                    hru_var[:] = hru_ids
                    
                    gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                    gru_var[:] = gru_ids
                    
                    # Add parameters
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)
                        
                        if param_values_array.ndim > 1:
                            param_values_array = param_values_array.flatten()
                        
                        if param_name in gru_level_params:
                            # GRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) >= num_grus:
                                param_var[:] = param_values_array[:num_grus]
                            else:
                                param_var[:] = param_values_array[0]
                        else:
                            # HRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) == num_hrus:
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_hrus]
                        
                        logger.debug(f"Added parameter {param_name} with shape {param_var.shape}")
                
                # Atomically move temporary file to final location
                try:
                    os.chmod(temp_path, 0o664)  # Set appropriate permissions
                    temp_path.rename(trial_params_path)
                    logger.debug(f"Trial parameters file created successfully: {trial_params_path}")
                    debug_info['files_checked'].append(f"trialParams.nc (created): {trial_params_path}")
                    return True
                except Exception as move_error:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise move_error
                
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                # Clean up temp file if it exists
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    error_msg = f"Failed to generate trial params after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return False
        
        return False
        
    except Exception as e:
        error_msg = f"Error generating trial params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_summa_worker_enhanced(summa_exe: Path, file_manager: Path, summa_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced SUMMA execution with better error handling and debugging"""
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"summa_worker_{os.getpid()}.log"
        
        # Set environment for single-threaded execution
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        })
        
        # Convert paths to strings for subprocess
        summa_exe_str = str(summa_exe)
        file_manager_str = str(file_manager)
        
        # Build command
        cmd = f"{summa_exe_str} -m {file_manager_str}"
        
        logger.info(f"Executing SUMMA command: {cmd}")
        logger.debug(f"Working directory: {summa_dir}")
        logger.debug(f"Log file: {log_file}")
        
        debug_info['commands_run'].append(f"SUMMA: {cmd}")
        debug_info['summa_log'] = str(log_file)
        
        # Verify executable permissions
        if not os.access(summa_exe, os.X_OK):
            error_msg = f"SUMMA executable is not executable: {summa_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        # Run SUMMA with explicit working directory
        with open(log_file, 'w') as f:
            f.write(f"SUMMA Execution Log\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Working Directory: {summa_dir}\n")
            f.write(f"Environment: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'unset')}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                env=env,
                cwd=str(summa_dir)  # Explicit working directory
            )
        
        # Check if output files were created
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            # Look for any .nc files
            nc_files = list(summa_dir.glob("*.nc"))
            if not nc_files:
                error_msg = f"No SUMMA output files found in {summa_dir}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                
                # Read log file for more details
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        debug_info['summa_log_content'] = log_content[-2000:]  # Last 2000 chars
                
                return False
        
        logger.info(f"SUMMA execution completed successfully. Output files: {len(timestep_files)} timestep files")
        debug_info['summa_output_files'] = [str(f) for f in timestep_files[:3]]  # First 3 files
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"SUMMA simulation failed with exit code {e.returncode}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        
        # Read log file for error details
        if 'log_file' in locals() and log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    debug_info['summa_log_content'] = log_content[-2000:]  # Last 2000 chars
            except:
                pass
                
        return False
        
    except subprocess.TimeoutExpired:
        error_msg = "SUMMA simulation timed out (30 minutes)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Error running SUMMA: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_mizuroute_worker_enhanced(task_data: Dict, mizuroute_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced mizuRoute execution with better error handling"""
    try:
        config = task_data['config']
        
        # Get mizuRoute executable
        mizu_path = config.get('INSTALL_PATH_MIZUROUTE', 'default')
        if mizu_path == 'default':
            mizu_path = Path(config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = Path(task_data['mizuroute_settings_dir']) / 'mizuroute.control'
        
        # Verify files exist
        if not mizu_exe.exists():
            error_msg = f"mizuRoute executable not found: {mizu_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
            
        if not control_file.exists():
            error_msg = f"mizuRoute control file not found: {control_file}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].extend([
            f"mizuRoute exe: {mizu_exe}",
            f"mizuRoute control: {control_file}"
        ])
        
        # Create log directory
        log_dir = mizuroute_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mizuroute_worker_{os.getpid()}.log"
        
        # Build command
        cmd = f"{mizu_exe} {control_file}"
        
        logger.info(f"Executing mizuRoute command: {cmd}")
        debug_info['commands_run'].append(f"mizuRoute: {cmd}")
        debug_info['mizuroute_log'] = str(log_file)
        
        # Run mizuRoute
        with open(log_file, 'w') as f:
            f.write(f"mizuRoute Execution Log\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Working Directory: {control_file.parent}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(control_file.parent)
            )
        
        # Check for output files
        nc_files = list(mizuroute_dir.glob("*.nc"))
        if not nc_files:
            error_msg = f"No mizuRoute output files found in {mizuroute_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        logger.info(f"mizuRoute execution completed successfully. Output files: {len(nc_files)}")
        debug_info['mizuroute_output_files'] = [str(f) for f in nc_files[:3]]
        
        return True
        
    except Exception as e:
        error_msg = f"mizuRoute execution failed: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _calculate_metrics_worker_enhanced(task_data: Dict, summa_dir: Path, mizuroute_dir: Path, logger, debug_info: Dict) -> Optional[float]:
    """Enhanced metrics calculation with better error handling"""
    try:
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        config = task_data['config']
        target_metric = task_data['target_metric']
        
        logger.debug(f"Calculating metrics for {calibration_var} using {target_metric}")
        
        # Load observed data
        if calibration_var == 'streamflow':
            obs_file = Path(task_data['project_dir']) / "observations" / "streamflow" / "preprocessed" / f"{task_data['domain_name']}_streamflow_processed.csv"
        elif calibration_var == 'snow':
            obs_file = Path(task_data['project_dir']) / "observations" / "snow" / "swe" / "processed" / f"{task_data['domain_name']}_swe_processed.csv"
        else:
            error_msg = f"Unsupported calibration variable: {calibration_var}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        if not obs_file.exists():
            error_msg = f"Observed data not found: {obs_file}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        debug_info['files_checked'].append(f"Observed data: {obs_file}")
        
        obs_df = pd.read_csv(obs_file)
        logger.debug(f"Loaded observed data: {len(obs_df)} records")
        
        # Find columns
        date_col = None
        data_col = None
        
        for col in obs_df.columns:
            col_lower = col.lower()
            if date_col is None and any(term in col_lower for term in ['date', 'time', 'datetime']):
                date_col = col
            if data_col is None:
                if calibration_var == 'streamflow' and any(term in col_lower for term in ['flow', 'discharge', 'q_']):
                    data_col = col
                elif calibration_var == 'snow' and any(term in col_lower for term in ['swe', 'snow']):
                    data_col = col
        
        if not date_col or not data_col:
            error_msg = f"Could not identify date/data columns in observed data. Columns: {list(obs_df.columns)}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        logger.debug(f"Using columns: date={date_col}, data={data_col}")
        
        # Process observed data
        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)
        observed_data = obs_df[data_col]
        
        logger.debug(f"Processed observed data: {len(observed_data)} records")
        
        # Get simulated data
        if calibration_var == 'streamflow' and _needs_mizuroute_routing_worker(config):
            # Use mizuRoute output
            sim_files = list(mizuroute_dir.glob("*.nc"))
            if not sim_files:
                error_msg = f"No mizuRoute output files found in {mizuroute_dir}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return None
            
            sim_file = sim_files[0]
            logger.debug(f"Using mizuRoute output: {sim_file}")
            simulated_data = _extract_streamflow_from_mizuroute_worker(sim_file, config, logger)
        else:
            # Use SUMMA output
            if calibration_var == 'streamflow':
                sim_files = list(summa_dir.glob("*timestep.nc"))
                if not sim_files:
                    error_msg = f"No SUMMA timestep files found in {summa_dir}"
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return None
                sim_file = sim_files[0]
                logger.debug(f"Using SUMMA timestep output: {sim_file}")
                simulated_data = _extract_streamflow_from_summa_worker(sim_file, config, logger)
            elif calibration_var == 'snow':
                sim_files = list(summa_dir.glob("*day.nc"))
                if not sim_files:
                    error_msg = f"No SUMMA daily files found in {summa_dir}"
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return None
                sim_file = sim_files[0]
                logger.debug(f"Using SUMMA daily output: {sim_file}")
                simulated_data = _extract_swe_from_summa_worker(sim_file, logger)
            else:
                error_msg = f"Unsupported calibration variable: {calibration_var}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return None
        
        if simulated_data is None or len(simulated_data) == 0:
            error_msg = "Failed to extract simulated data or no data available"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        logger.debug(f"Extracted simulated data: {len(simulated_data)} records")
        
        # Align time series
        simulated_data.index = simulated_data.index.round('h')
        common_idx = observed_data.index.intersection(simulated_data.index)
        
        if len(common_idx) == 0:
            error_msg = "No common time indices between observed and simulated data"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        obs_common = observed_data.loc[common_idx]
        sim_common = simulated_data.loc[common_idx]
        
        logger.debug(f"Common time period: {len(common_idx)} records")
        
        # Calculate metrics
        metrics = _calculate_performance_metrics_worker(obs_common, sim_common)
        
        # Extract target metric
        score = metrics.get(target_metric, metrics.get('KGE', np.nan))
        
        if np.isnan(score):
            error_msg = f"Target metric {target_metric} is NaN. Available metrics: {list(metrics.keys())}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return None
        
        # Apply negation for metrics where lower is better
        if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
            score = -score
        
        logger.debug(f"Calculated {target_metric}: {score:.6f}")
        debug_info['metrics'] = metrics
        debug_info['final_score'] = score
        
        return score
        
    except Exception as e:
        error_msg = f"Error calculating metrics: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return None


# Keep the existing helper functions but add logging parameters where needed
def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
    
    if domain_method not in ['point', 'lumped']:
        return True
    
    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True
    
    return False


def _extract_streamflow_from_mizuroute_worker(sim_file: Path, config: Dict, logger) -> Optional[pd.Series]:
    """Extract streamflow from mizuRoute output with error handling"""
    try:
        with xr.open_dataset(sim_file) as ds:
            reach_id = int(config.get('SIM_REACH_ID', 123))
            
            if 'reachID' not in ds.variables:
                logger.error("reachID variable not found in mizuRoute output")
                return None
            
            reach_ids = ds['reachID'].values
            reach_indices = np.where(reach_ids == reach_id)[0]
            
            if len(reach_indices) == 0:
                logger.error(f"Reach ID {reach_id} not found in mizuRoute output")
                return None
            
            reach_index = reach_indices[0]
            
            # Find streamflow variable
            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        return var.isel(seg=reach_index).to_pandas()
                    elif 'reachID' in var.dims:
                        return var.isel(reachID=reach_index).to_pandas()
            
            logger.error("No suitable streamflow variable found in mizuRoute output")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting mizuRoute streamflow: {str(e)}")
        return None


def _extract_streamflow_from_summa_worker(sim_file: Path, config: Dict, logger) -> Optional[pd.Series]:
    """Extract streamflow from SUMMA output with error handling"""
    try:
        with xr.open_dataset(sim_file) as ds:
            # Find streamflow variable
            for var_name in ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    
                    if len(var.shape) > 1:
                        if 'hru' in var.dims:
                            sim_data = var.isel(hru=0).to_pandas()
                        elif 'gru' in var.dims:
                            sim_data = var.isel(gru=0).to_pandas()
                        else:
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                            else:
                                sim_data = var.to_pandas()
                    else:
                        sim_data = var.to_pandas()
                    
                    # Convert units (m/s to m³/s) - use default catchment area
                    catchment_area = 1e6  # 1 km² default
                    return sim_data * catchment_area
            
            logger.error("No suitable streamflow variable found in SUMMA output")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
        return None


def _extract_swe_from_summa_worker(sim_file: Path, logger) -> Optional[pd.Series]:
    """Extract SWE from SUMMA daily output with error handling"""
    try:
        with xr.open_dataset(sim_file) as ds:
            if 'scalarSWE' not in ds.variables:
                logger.error("scalarSWE variable not found in SUMMA daily output")
                return None
            
            var = ds['scalarSWE']
            
            if len(var.shape) > 1:
                if 'hru' in var.dims:
                    return var.isel(hru=0).to_pandas()
                elif 'gru' in var.dims:
                    return var.isel(gru=0).to_pandas()
                else:
                    non_time_dims = [dim for dim in var.dims if dim != 'time']
                    if non_time_dims:
                        return var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        return var.to_pandas()
            else:
                return var.to_pandas()
                
    except Exception as e:
        logger.error(f"Error extracting SWE: {str(e)}")
        return None


def _calculate_performance_metrics_worker(observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics with error handling"""
    try:
        # Clean data
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}
        
        # Calculate metrics
        mean_obs = observed.mean()
        
        # NSE
        nse_num = ((observed - simulated) ** 2).sum()
        nse_den = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
        
        # RMSE
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # KGE
        r = observed.corr(simulated)
        alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
        beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse, 'r': r, 'alpha': alpha, 'beta': beta}
        
    except Exception:
        return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}

# ============= ABSTRACT BASE CLASSES =============

class CalibrationTarget(ABC):
    """Abstract base class for different calibration variables (streamflow, snow, etc.)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        
        # Parse time periods
        self.calibration_period = self._parse_date_range(config.get('CALIBRATION_PERIOD', ''))
        self.evaluation_period = self._parse_date_range(config.get('EVALUATION_PERIOD', ''))
    
    def calculate_metrics(self, sim_dir: Path, mizuroute_dir: Optional[Path] = None, 
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for this calibration target
        
        Args:
            sim_dir: SUMMA simulation directory
            mizuroute_dir: mizuRoute simulation directory (if needed)
            calibration_only: If True, only calculate calibration period metrics
                            If False, calculate both calibration and evaluation metrics
        """
        try:
            # Determine which simulation directory to use
            if self.needs_routing() and mizuroute_dir:
                output_dir = mizuroute_dir
            else:
                output_dir = sim_dir
            
            # Get simulation files
            sim_files = self.get_simulation_files(output_dir)
            if not sim_files:
                self.logger.error(f"No simulation files found in {output_dir}")
                return None
            
            # Extract simulated data
            sim_data = self.extract_simulated_data(sim_files)
            if sim_data is None or len(sim_data) == 0:
                self.logger.error("Failed to extract simulated data")
                return None
            
            # Load observed data
            obs_data = self._load_observed_data()
            if obs_data is None or len(obs_data) == 0:
                self.logger.error("Failed to load observed data")
                return None
            
            # Align time series and calculate metrics
            metrics = {}
            
            # Always calculate metrics for calibration period if available
            if self.calibration_period[0] and self.calibration_period[1]:
                calib_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.calibration_period, "Calib"
                )
                metrics.update(calib_metrics)
            
            # Only calculate evaluation period metrics if requested (final evaluation)
            if not calibration_only and self.evaluation_period[0] and self.evaluation_period[1]:
                eval_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.evaluation_period, "Eval"
                )
                metrics.update(eval_metrics)
            
            # If no specific periods, calculate for full overlap (fallback)
            if not metrics:
                full_metrics = self._calculate_period_metrics(obs_data, sim_data, (None, None), "")
                metrics.update(full_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {self.__class__.__name__}: {str(e)}")
            return None
    
    @abstractmethod
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get relevant simulation output files for this calibration target"""
        pass
    
    @abstractmethod
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract simulated data from output files"""
        pass
    
    @abstractmethod
    def get_observed_data_path(self) -> Path:
        """Get path to observed data file"""
        pass
    
    @abstractmethod
    def needs_routing(self) -> bool:
        """Whether this calibration target requires mizuRoute routing"""
        pass
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed data from file"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            
            # Find date and data columns
            date_col = next((col for col in obs_df.columns 
                           if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)
            
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                return None
            
            # Process data
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            
            return obs_df[data_col]
            
        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None
    
    @abstractmethod
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the data column in observed data file"""
        pass
    
    def _calculate_period_metrics(self, obs_data: pd.Series, sim_data: pd.Series, 
                                 period: Tuple, prefix: str) -> Dict[str, float]:
        """Calculate metrics for a specific time period"""
        try:
            # Filter data to period if specified
            if period[0] and period[1]:
                period_mask = (obs_data.index >= period[0]) & (obs_data.index <= period[1])
                obs_period = obs_data[period_mask]
            else:
                obs_period = obs_data
            
            # Align time series
            sim_data.index = sim_data.index.round('h')
            common_idx = obs_period.index.intersection(sim_data.index)
            
            if len(common_idx) == 0:
                self.logger.warning(f"No common time indices for {prefix} period")
                return {}
            
            obs_common = obs_period.loc[common_idx]
            sim_common = sim_data.loc[common_idx]
            
            # Calculate metrics
            base_metrics = self._calculate_performance_metrics(obs_common, sim_common)
            
            # Add prefix if specified
            if prefix:
                return {f"{prefix}_{k}": v for k, v in base_metrics.items()}
            else:
                return base_metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating period metrics: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics between observed and simulated data"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # Nash-Sutcliffe Efficiency
            mean_obs = observed.mean()
            nse_num = ((observed - simulated) ** 2).sum()
            nse_den = ((observed - mean_obs) ** 2).sum()
            nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
            
            # Root Mean Square Error
            rmse = np.sqrt(((observed - simulated) ** 2).mean())
            
            # Percent Bias
            pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
            
            # Kling-Gupta Efficiency
            r = observed.corr(simulated)
            alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
            beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
            
            # Mean Absolute Error
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
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _parse_date_range(self, date_range_str: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse date range string from config"""
        if not date_range_str:
            return None, None
        
        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except Exception as e:
            self.logger.warning(f"Could not parse date range '{date_range_str}': {str(e)}")
        
        return None, None


class StreamflowTarget(CalibrationTarget):
    """Streamflow calibration target"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA timestep files or mizuRoute output files"""
        # First try mizuRoute files if in mizuRoute directory
        if 'mizuRoute' in str(sim_dir):
            mizu_files = list(sim_dir.glob("*.nc"))
            if mizu_files:
                return mizu_files
        
        # Otherwise look for SUMMA timestep files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            # Determine if this is mizuRoute or SUMMA output
            if self._is_mizuroute_output(sim_file):
                return self._extract_mizuroute_streamflow(sim_file)
            else:
                return self._extract_summa_streamflow(sim_file)
        except Exception as e:
            self.logger.error(f"Error extracting streamflow data from {sim_file}: {str(e)}")
            raise
    
    def _is_mizuroute_output(self, sim_file: Path) -> bool:
        """Check if file is mizuRoute output based on variables"""
        try:
            with xr.open_dataset(sim_file) as ds:
                mizuroute_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'reachID']
                return any(var in ds.variables for var in mizuroute_vars)
        except:
            return False
    
    def _extract_mizuroute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from mizuRoute output"""
        with xr.open_dataset(sim_file) as ds:
            reach_id = int(self.config.get('SIM_REACH_ID'))
            
            # Find reach index
            if 'reachID' not in ds.variables:
                raise ValueError("reachID variable not found in mizuRoute output")
            
            reach_ids = ds['reachID'].values
            reach_indices = np.where(reach_ids == reach_id)[0]
            
            if len(reach_indices) == 0:
                raise ValueError(f"Reach ID {reach_id} not found in mizuRoute output")
            
            reach_index = reach_indices[0]
            
            # Find streamflow variable
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        return var.isel(seg=reach_index).to_pandas()
                    elif 'reachID' in var.dims:
                        return var.isel(reachID=reach_index).to_pandas()
            
            raise ValueError("No suitable streamflow variable found in mizuRoute output")
    
    def _extract_summa_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from SUMMA output"""
        with xr.open_dataset(sim_file) as ds:
            # Find streamflow variable
            streamflow_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
            
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    
                    # Extract data based on dimensions
                    if len(var.shape) > 1:
                        if 'hru' in var.dims:
                            sim_data = var.isel(hru=0).to_pandas()
                        elif 'gru' in var.dims:
                            sim_data = var.isel(gru=0).to_pandas()
                        else:
                            # Take first spatial index
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                            else:
                                sim_data = var.to_pandas()
                    else:
                        sim_data = var.to_pandas()
                    
                    # Convert units (m/s to m³/s) for SUMMA output
                    catchment_area = self._get_catchment_area()
                    return sim_data * catchment_area
            
            raise ValueError("No suitable streamflow variable found in SUMMA output")
    
    def _get_catchment_area(self) -> float:
        """Get catchment area for unit conversion"""
        try:
            import geopandas as gpd
            
            # Try basin shapefile first
            basin_path = self.project_dir / "shapefiles" / "river_basins"
            basin_files = list(basin_path.glob("*.shp"))
            
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:  # Reasonable area
                        return total_area
            
            # Fallback: calculate from geometry
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                if gdf.crs and gdf.crs.is_geographic:
                    # Reproject to UTM for area calculation
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                
                return gdf.geometry.area.sum()
            
        except Exception as e:
            self.logger.warning(f"Could not calculate catchment area: {str(e)}")
        
        # Default fallback
        return 1e6  # 1 km²
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data"""
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default' or not obs_path:
            return self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        return Path(obs_path)
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find streamflow data column"""
        for col in columns:
            if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        """Check if streamflow calibration needs mizuRoute routing"""
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
        
        # Distributed domain always needs routing
        if domain_method not in ['point', 'lumped']:
            return True
        
        # Lumped domain with river network routing
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        
        return False


class SnowTarget(CalibrationTarget):
    """Snow Water Equivalent (SWE) calibration target"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files for SWE"""
        return list(sim_dir.glob("*day.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract SWE data from SUMMA daily output"""
        sim_file = sim_files[0]
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if 'scalarSWE' not in ds.variables:
                    raise ValueError("scalarSWE variable not found in daily output")
                
                var = ds['scalarSWE']
                
                # Extract data based on dimensions
                if len(var.shape) > 1:
                    if 'hru' in var.dims:
                        sim_data = var.isel(hru=0).to_pandas()
                    elif 'gru' in var.dims:
                        sim_data = var.isel(gru=0).to_pandas()
                    else:
                        # Take first spatial index
                        non_time_dims = [dim for dim in var.dims if dim != 'time']
                        if non_time_dims:
                            sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                        else:
                            sim_data = var.to_pandas()
                else:
                    sim_data = var.to_pandas()
                
                # No unit conversion needed - SWE is already in mm
                return sim_data
                
        except Exception as e:
            self.logger.error(f"Error extracting SWE data from {sim_file}: {str(e)}")
            raise
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed SWE data"""
        return self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find SWE data column"""
        for col in columns:
            if any(term in col.lower() for term in ['swe', 'snow', 'water_equivalent']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        """Snow calibration never needs routing"""
        return False


# ============= PARAMETER MANAGEMENT =============

class ParameterManager:
    """Handles parameter bounds, normalization, file generation, and soil depth calculations"""
    
    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        
        # Parse parameter lists
        self.local_params = [p.strip() for p in config.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        self.basin_params = [p.strip() for p in config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        self.depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        self.mizuroute_params = []
        
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            self.mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]
        
        # Load parameter bounds
        self.param_bounds = self._parse_all_bounds()
        
        # Load original soil depths if depth calibration enabled
        self.original_depths = None
        if self.depth_params:
            self.original_depths = self._load_original_depths()
        
        # Get attribute file path
        self.attr_file_path = self.optimization_settings_dir / config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all parameter names"""
        return self.local_params + self.basin_params + self.depth_params + self.mizuroute_params
    
    def get_initial_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Get initial parameter values from existing files or defaults"""
        # Try to load existing optimized parameters
        existing_params = self._load_existing_optimized_parameters()
        if existing_params:
            self.logger.info("Loaded existing optimized parameters")
            return existing_params
        
        # Extract parameters from model files
        self.logger.info("Extracting initial parameters from default values")
        return self._extract_default_parameters()
    
    def normalize_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to normalized array [0,1]"""
        normalized = np.zeros(len(self.all_param_names))
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in params and param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                
                # Get parameter value
                param_values = params[param_name]
                if isinstance(param_values, np.ndarray) and len(param_values) > 1:
                    value = np.mean(param_values)  # Use mean for multi-value parameters
                else:
                    value = param_values[0] if isinstance(param_values, np.ndarray) else param_values
                
                # Normalize to [0,1]
                normalized[i] = (value - bounds['min']) / (bounds['max'] - bounds['min'])
        
        return np.clip(normalized, 0, 1)
    
    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized array back to parameter dictionary"""
        params = {}
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                denorm_value = bounds['min'] + normalized_array[i] * (bounds['max'] - bounds['min'])
                
                # Validate bounds
                denorm_value = np.clip(denorm_value, bounds['min'], bounds['max'])
                
                # Format based on parameter type
                if param_name in self.depth_params:
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.mizuroute_params:
                    params[param_name] = denorm_value
                elif param_name in self.basin_params:
                    params[param_name] = np.array([denorm_value])
                else:
                    # Local parameters - expand to HRU count
                    params[param_name] = self._expand_to_hru_count(denorm_value)
        
        return params
    
    def apply_parameters(self, params: Dict[str, np.ndarray]) -> bool:
        """Apply parameters to model files"""
        try:
            # Update soil depths if depth calibration enabled
            if self.depth_params and 'total_mult' in params and 'shape_factor' in params:
                if not self._update_soil_depths(params):
                    return False
            
            # Update mizuRoute parameters if enabled
            if self.mizuroute_params:
                if not self._update_mizuroute_parameters(params):
                    return False
            
            # Generate trial parameters file (excluding depth and mizuRoute parameters)
            hydraulic_params = {k: v for k, v in params.items() 
                              if k not in self.depth_params + self.mizuroute_params}
            
            if hydraulic_params:
                if not self._generate_trial_params_file(hydraulic_params):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying parameters: {str(e)}")
            return False
    
    def _parse_all_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from all parameter info files"""
        bounds = {}
        
        # Parse local parameter bounds
        if self.local_params:
            local_param_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'localParamInfo.txt'
            local_bounds = self._parse_param_info_file(local_param_file, self.local_params)
            bounds.update(local_bounds)
        
        # Parse basin parameter bounds
        if self.basin_params:
            basin_param_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'basinParamInfo.txt'
            basin_bounds = self._parse_param_info_file(basin_param_file, self.basin_params)
            bounds.update(basin_bounds)
        
        # Add depth parameter bounds
        if self.depth_params:
            bounds['total_mult'] = {'min': 0.1, 'max': 5.0}
            bounds['shape_factor'] = {'min': 0.1, 'max': 3.0}
        
        # Add mizuRoute parameter bounds
        if self.mizuroute_params:
            mizuroute_bounds = self._get_mizuroute_bounds()
            bounds.update(mizuroute_bounds)
        
        return bounds
    
    def _parse_param_info_file(self, file_path: Path, param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from a SUMMA parameter info file"""
        bounds = {}
        
        if not file_path.exists():
            self.logger.error(f"Parameter file not found: {file_path}")
            return bounds
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 4:
                        continue
                    
                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            min_val = float(parts[2].replace('d','e').replace('D','e'))
                            max_val = float(parts[3].replace('d','e').replace('D','e'))
                            
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            
                            if min_val == max_val:
                                range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
                                min_val -= range_val
                                max_val += range_val
                            
                            bounds[param_name] = {'min': min_val, 'max': max_val}
                            
                        except ValueError as e:
                            self.logger.error(f"Could not parse bounds for {param_name}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error reading parameter file {file_path}: {str(e)}")
        
        return bounds
    
    def _get_mizuroute_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for mizuRoute parameters"""
        default_bounds = {
            'velo': {'min': 0.1, 'max': 5.0},
            'diff': {'min': 100.0, 'max': 5000.0},
            'mann_n': {'min': 0.01, 'max': 0.1},
            'wscale': {'min': 0.0001, 'max': 0.01},
            'fshape': {'min': 1.0, 'max': 5.0},
            'tscale': {'min': 3600, 'max': 172800}
        }
        
        bounds = {}
        for param in self.mizuroute_params:
            if param in default_bounds:
                bounds[param] = default_bounds[param]
            else:
                self.logger.warning(f"Unknown mizuRoute parameter: {param}")
        
        return bounds
    
    def _load_existing_optimized_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Load existing optimized parameters from default settings"""
        trial_params_path = self.config.get('CONFLUENCE_DATA_DIR')
        if trial_params_path == 'default':
            return None
        
        # Implementation would check for existing trialParams.nc file
        # For brevity, returning None - full implementation would load existing params
        return None
    
    def _extract_default_parameters(self) -> Dict[str, np.ndarray]:
        """Extract default parameter values from parameter info files"""
        defaults = {}
        
        # Parse local parameters
        if self.local_params:
            local_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'localParamInfo.txt', 
                self.local_params
            )
            defaults.update(local_defaults)
        
        # Parse basin parameters
        if self.basin_params:
            basin_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'basinParamInfo.txt',
                self.basin_params
            )
            defaults.update(basin_defaults)
        
        # Add depth parameters
        if self.depth_params:
            defaults['total_mult'] = np.array([1.0])
            defaults['shape_factor'] = np.array([1.0])
        
        # Add mizuRoute parameters
        if self.mizuroute_params:
            for param in self.mizuroute_params:
                defaults[param] = self._get_default_mizuroute_value(param)
        
        # Expand to HRU count
        return self._expand_defaults_to_hru_count(defaults)
    
    def _parse_defaults_from_file(self, file_path: Path, param_names: List[str]) -> Dict[str, np.ndarray]:
        """Parse default values from parameter info file"""
        defaults = {}
        
        if not file_path.exists():
            return defaults
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        param_name = parts[0]
                        if param_name in param_names:
                            try:
                                default_val = float(parts[1].replace('d','e').replace('D','e'))
                                defaults[param_name] = np.array([default_val])
                            except ValueError:
                                continue
        except Exception as e:
            self.logger.error(f"Error parsing defaults from {file_path}: {str(e)}")
        
        return defaults
    
    def _get_default_mizuroute_value(self, param_name: str) -> float:
        """Get default value for mizuRoute parameter"""
        defaults = {
            'velo': 1.0,
            'diff': 1000.0,
            'mann_n': 0.025,
            'wscale': 0.001,
            'fshape': 2.5,
            'tscale': 86400
        }
        return defaults.get(param_name, 1.0)
    
    def _expand_defaults_to_hru_count(self, defaults: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Expand parameter defaults to match HRU count"""
        try:
            # Get HRU count from attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            
            expanded_defaults = {}
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            for param_name, values in defaults.items():
                if param_name in self.basin_params or param_name in routing_params:
                    expanded_defaults[param_name] = values
                elif param_name in self.depth_params or param_name in self.mizuroute_params:
                    expanded_defaults[param_name] = values
                else:
                    expanded_defaults[param_name] = np.full(num_hrus, values[0])
            
            return expanded_defaults
            
        except Exception as e:
            self.logger.error(f"Error expanding defaults: {str(e)}")
            return defaults
    
    def _expand_to_hru_count(self, value: float) -> np.ndarray:
        """Expand single value to HRU count"""
        try:
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            return np.full(num_hrus, value)
        except:
            return np.array([value])
    
    def _load_original_depths(self) -> Optional[np.ndarray]:
        """Load original soil depths from coldState.nc"""
        try:
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            with nc.Dataset(coldstate_path, 'r') as ds:
                if 'mLayerDepth' in ds.variables:
                    return ds.variables['mLayerDepth'][:, 0].copy()
            
        except Exception as e:
            self.logger.error(f"Error loading original depths: {str(e)}")
        
        return None
    
    def _update_soil_depths(self, params: Dict[str, np.ndarray]) -> bool:
        """Update soil depths in coldState.nc"""
        if self.original_depths is None:
            return False
        
        try:
            total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
            shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
            
            # Calculate new depths using shape method
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is None:
                return False
            
            # Calculate layer heights
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]
            
            # Update coldState.nc
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    return False
                
                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating soil depths: {str(e)}")
            return False
    
    def _calculate_new_depths(self, total_mult: float, shape_factor: float) -> Optional[np.ndarray]:
        """Calculate new soil depths using shape method"""
        if self.original_depths is None:
            return None
        
        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        # Calculate shape weights
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        # Normalize weights
        w /= w.mean()
        
        # Apply multipliers
        new_depths = arr * w * total_mult
        
        return new_depths
    
    def _update_mizuroute_parameters(self, params: Dict) -> bool:
        """Update mizuRoute parameters in param.nml.default"""
        try:
            mizuroute_settings_dir = self.optimization_settings_dir.parent / "mizuRoute"
            param_file = mizuroute_settings_dir / "param.nml.default"
            
            if not param_file.exists():
                return True  # Skip if file doesn't exist
            
            # Read file
            with open(param_file, 'r') as f:
                content = f.read()
            
            # Update parameters
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
            
            # Write updated file
            with open(param_file, 'w') as f:
                f.write(updated_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating mizuRoute parameters: {str(e)}")
            return False
    
    def _generate_trial_params_file(self, params: Dict[str, np.ndarray]) -> bool:
        """Generate trialParams.nc file with proper dimensions"""
        try:
            trial_params_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
            
            # Get HRU and GRU counts from attributes
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
                num_grus = ds.sizes.get('gru', 1)
            
            # Define parameter levels
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                # Create dimensions
                output_ds.createDimension('hru', num_hrus)
                output_ds.createDimension('gru', num_grus)
                
                # Create coordinate variables
                hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                hru_var[:] = range(1, num_hrus + 1)
                
                gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                gru_var[:] = range(1, num_grus + 1)
                
                # Add parameters
                for param_name, param_values in params.items():
                    param_values_array = np.asarray(param_values)
                    
                    if param_name in routing_params or param_name in self.basin_params:
                        # GRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_grus]
                    else:
                        # HRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == num_hrus:
                            param_var[:] = param_values_array
                        elif len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_hrus]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating trial params file: {str(e)}")
            return False


# ============= MODEL EXECUTION =============

class ModelExecutor:
    """Handles SUMMA and mizuRoute execution with routing support"""
    
    def __init__(self, config: Dict, logger: logging.Logger, calibration_target: CalibrationTarget):
        self.config = config
        self.logger = logger
        self.calibration_target = calibration_target
    
    def run_models(self, summa_dir: Path, mizuroute_dir: Path, settings_dir: Path,
                  mizuroute_settings_dir: Optional[Path] = None) -> bool:
        """Run SUMMA and mizuRoute if needed"""
        try:
            # Run SUMMA
            if not self._run_summa(settings_dir, summa_dir):
                return False
            
            # Run mizuRoute if needed
            if self.calibration_target.needs_routing():
                if mizuroute_settings_dir is None:
                    mizuroute_settings_dir = settings_dir.parent / "mizuRoute"
                
                # Handle lumped-to-distributed conversion if needed
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                
                if domain_method == 'lumped' and routing_delineation == 'river_network':
                    if not self._convert_lumped_to_distributed(summa_dir, mizuroute_settings_dir):
                        return False
                
                if not self._run_mizuroute(mizuroute_settings_dir, mizuroute_dir):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error running models: {str(e)}")
            return False
    
    def _run_summa(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run SUMMA simulation"""
        try:
            # Get SUMMA executable
            summa_path = self.config.get('SUMMA_INSTALL_PATH')
            if summa_path == 'default':
                summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
            else:
                summa_path = Path(summa_path)
            
            summa_exe = summa_path / self.config.get('SUMMA_EXE')
            file_manager = settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
            
            if not summa_exe.exists():
                self.logger.error(f"SUMMA executable not found: {summa_exe}")
                return False
            
            if not file_manager.exists():
                self.logger.error(f"File manager not found: {file_manager}")
                return False
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run SUMMA
            cmd = f"{summa_exe} -m {file_manager}"
            log_file = log_dir / f"summa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      check=True, timeout=1800)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("SUMMA simulation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False
    
    def _run_mizuroute(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run mizuRoute simulation"""
        try:
            # Get mizuRoute executable
            mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
            if mizu_path == 'default':
                mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
            else:
                mizu_path = Path(mizu_path)
            
            mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
            control_file = settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
            
            if not mizu_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
                return False
            
            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run mizuRoute
            cmd = f"{mizu_exe} {control_file}"
            log_file = log_dir / f"mizuroute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      check=True, timeout=1800, cwd=str(settings_dir))
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"mizuRoute simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute simulation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False
    
    def _convert_lumped_to_distributed(self, summa_dir: Path, mizuroute_settings_dir: Path) -> bool:
        """Convert lumped SUMMA output for distributed routing"""
        try:
            # Find SUMMA timestep file
            timestep_files = list(summa_dir.glob("*timestep.nc"))
            if not timestep_files:
                return False
            
            summa_file = timestep_files[0]
            
            # Load topology to get segment information
            topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
            if not topology_file.exists():
                return False
            
            with xr.open_dataset(topology_file) as topo_ds:
                seg_ids = topo_ds['segId'].values
                n_segments = len(seg_ids)
            
            # Load and convert SUMMA output
            with xr.open_dataset(summa_file, decode_times=False) as summa_ds:
                # Find routing variable
                routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
                if routing_var not in summa_ds:
                    routing_var = 'basin__TotalRunoff'
                
                if routing_var not in summa_ds:
                    return False
                
                # Create mizuRoute forcing dataset
                mizuForcing = xr.Dataset()
                
                # Copy time coordinate
                mizuForcing['time'] = summa_ds['time']
                
                # Create GRU dimension
                mizuForcing['gru'] = xr.DataArray(seg_ids, dims=('gru',))
                mizuForcing['gruId'] = xr.DataArray(seg_ids, dims=('gru',))
                
                # Get and broadcast runoff data
                runoff_data = summa_ds[routing_var].values
                if len(runoff_data.shape) == 2:
                    runoff_data = runoff_data[:, 0] if runoff_data.shape[1] > 0 else runoff_data.flatten()
                
                # Broadcast to all segments
                tiled_data = np.tile(runoff_data[:, np.newaxis], (1, n_segments))
                
                mizuForcing['averageRoutedRunoff'] = xr.DataArray(
                    tiled_data, dims=('time', 'gru'),
                    attrs={'long_name': 'Broadcast runoff for distributed routing', 'units': 'm/s'}
                )
                
                # Copy global attributes
                mizuForcing.attrs.update(summa_ds.attrs)
            
            # Save converted file
            mizuForcing.to_netcdf(summa_file, format='NETCDF4')
            mizuForcing.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting lumped to distributed: {str(e)}")
            return False


# ============= RESULTS MANAGEMENT =============

class ResultsManager:
    """Handles optimization results, history tracking, and visualization"""
    
    def __init__(self, config: Dict, logger: logging.Logger, output_dir: Path):
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
    
    def save_results(self, best_params: Dict, best_score: float, history: List[Dict], 
                    final_result: Optional[Dict] = None) -> bool:
        """Save optimization results to files"""
        try:
            # Save best parameters to CSV
            self._save_best_parameters_csv(best_params)
            
            # Save history to CSV
            self._save_history_csv(history)
            
            # Save metadata
            self._save_metadata(best_score, len(history), final_result)
            
            # Create visualization plots
            self._create_plots(history, best_params)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False
    
    def _save_best_parameters_csv(self, best_params: Dict) -> None:
        """Save best parameters to CSV file"""
        param_data = []
        
        for param_name, values in best_params.items():
            if isinstance(values, np.ndarray):
                if len(values) == 1:
                    param_data.append({
                        'parameter': param_name,
                        'value': values[0],
                        'type': 'scalar'
                    })
                else:
                    param_data.append({
                        'parameter': param_name,
                        'value': np.mean(values),
                        'type': 'array_mean',
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values)
                    })
            else:
                param_data.append({
                    'parameter': param_name,
                    'value': values,
                    'type': 'scalar'
                })
        
        param_df = pd.DataFrame(param_data)
        param_csv_path = self.output_dir / "best_parameters.csv"
        param_df.to_csv(param_csv_path, index=False)
        
        self.logger.info(f"Saved best parameters to: {param_csv_path}")
    
    def _save_history_csv(self, history: List[Dict]) -> None:
        """Save optimization history to CSV"""
        if not history:
            return
        
        history_data = []
        for gen_data in history:
            row = {
                'generation': gen_data.get('generation', 0),
                'best_score': gen_data.get('best_score'),
                'mean_score': gen_data.get('mean_score'),
                'std_score': gen_data.get('std_score'),
                'valid_individuals': gen_data.get('valid_individuals', 0)
            }
            
            # Add best parameters if available
            if gen_data.get('best_params'):
                for param_name, values in gen_data['best_params'].items():
                    if isinstance(values, np.ndarray):
                        row[f'best_{param_name}'] = np.mean(values) if len(values) > 1 else values[0]
                    else:
                        row[f'best_{param_name}'] = values
            
            history_data.append(row)
        
        history_df = pd.DataFrame(history_data)
        history_csv_path = self.output_dir / "optimization_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        
        self.logger.info(f"Saved optimization history to: {history_csv_path}")
    
    def _save_metadata(self, best_score: float, num_generations: int, final_result: Optional[Dict]) -> None:
        """Save optimization metadata"""
        metadata = {
            'algorithm': 'Differential Evolution',
            'domain_name': self.domain_name,
            'experiment_id': self.experiment_id,
            'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
            'target_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'best_score': best_score,
            'num_generations': num_generations,
            'population_size': self.config.get('POPULATION_SIZE', 50),
            'F': self.config.get('DE_SCALING_FACTOR', 0.5),
            'CR': self.config.get('DE_CROSSOVER_RATE', 0.9),
            'parallel_processes': self.config.get('MPI_PROCESSES', 1),
            'completed_at': datetime.now().isoformat()
        }
        
        if final_result:
            metadata.update(final_result)
        
        metadata_df = pd.DataFrame([metadata])
        metadata_csv_path = self.output_dir / "optimization_metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)
        
        self.logger.info(f"Saved metadata to: {metadata_csv_path}")
    
    def _create_plots(self, history: List[Dict], best_params: Dict) -> None:
        """Create optimization progress plots"""
        try:
            import matplotlib.pyplot as plt
            
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract progress data
            generations = [h['generation'] for h in history]
            best_scores = [h['best_score'] for h in history if h.get('best_score') is not None]
            
            if not best_scores:
                return
            
            # Progress plot
            plt.figure(figsize=(12, 6))
            plt.plot(generations[:len(best_scores)], best_scores, 'b-o', markersize=4)
            plt.xlabel('Generation')
            plt.ylabel(f"Performance ({self.config.get('OPTIMIZATION_METRIC', 'KGE')})")
            plt.title(f'DE Optimization Progress - {self.config.get("CALIBRATION_VARIABLE", "streamflow").title()} Calibration')
            plt.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = np.nanargmax(best_scores)
            plt.plot(generations[best_idx], best_scores[best_idx], 'ro', markersize=10,
                    label=f'Best: {best_scores[best_idx]:.4f} at generation {generations[best_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter evolution plots for depth parameters
            if self.config.get('CALIBRATE_DEPTH', False):
                self._create_depth_parameter_plots(history, plots_dir)
            
            self.logger.info("Created optimization plots")
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
    
    def _create_depth_parameter_plots(self, history: List[Dict], plots_dir: Path) -> None:
        """Create depth parameter evolution plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Extract depth parameters
            generations = []
            total_mults = []
            shape_factors = []
            
            for h in history:
                if h.get('best_params') and 'total_mult' in h['best_params'] and 'shape_factor' in h['best_params']:
                    generations.append(h['generation'])
                    
                    tm = h['best_params']['total_mult']
                    sf = h['best_params']['shape_factor']
                    
                    tm_val = tm[0] if isinstance(tm, np.ndarray) and len(tm) > 0 else tm
                    sf_val = sf[0] if isinstance(sf, np.ndarray) and len(sf) > 0 else sf
                    
                    total_mults.append(tm_val)
                    shape_factors.append(sf_val)
            
            if not generations:
                return
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Total multiplier plot
            ax1.plot(generations, total_mults, 'g-o', markersize=4)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Total Depth Multiplier')
            ax1.set_title('Soil Depth Total Multiplier Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No change (1.0)')
            ax1.legend()
            
            # Shape factor plot
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
            
        except Exception as e:
            self.logger.error(f"Error creating depth parameter plots: {str(e)}")


# ============= MAIN DE OPTIMIZER CLASS =============

class DEOptimizer:
    """
    Differential Evolution Optimizer for CONFLUENCE with multi-target support and parallel processing
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize DE optimizer with clean component architecture"""
        self.config = config
        self.logger = logger
        
        # Setup basic paths
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Create optimization directories

        self.optimization_dir = self.project_dir / "simulations" / "run_de"
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        self.output_dir = self.project_dir / "optimisation" / f"de_{self.experiment_id}"

        self.parameter_manager = ParameterManager(config, logger, self.optimization_settings_dir)
        self._setup_optimization_directories()
        
        # Initialize component managers
        self.calibration_target = self._create_calibration_target()
        self.model_executor = ModelExecutor(config, logger, self.calibration_target)
        self.results_manager = ResultsManager(config, logger, self.output_dir)
        
        # DE algorithm parameters
        self.population_size = self._determine_population_size()
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.F = config.get('DE_SCALING_FACTOR', 0.5)
        self.CR = config.get('DE_CROSSOVER_RATE', 0.9)
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # DE state variables
        self.population = None
        self.population_scores = None
        self.best_params = None
        self.best_score = float('-inf')
        self.iteration_history = []
        
        # Parallel processing setup
        self.use_parallel = config.get('MPI_PROCESSES', 1) > 1
        self.num_processes = max(1, config.get('MPI_PROCESSES', 1))
        
        if self.use_parallel:
            self._setup_parallel_processing()
    
    def run_de_optimization(self) -> Dict[str, Any]:
        #self.diagnose_calibration_issues()
        """Main public interface for running DE optimization"""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting DE optimization for {self.config.get('CALIBRATION_VARIABLE', 'streamflow')} calibration")
        self.logger.info(f"Target metric: {self.target_metric}")
        self.logger.info(f"Population size: {self.population_size}, Max generations: {self.max_iterations}")
        
        if self.use_parallel:
            self.logger.info(f"Parallel processing: {self.num_processes} processes")
        
        self.logger.info("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # Initialize optimization
            initial_params = self.parameter_manager.get_initial_parameters()
            if not initial_params:
                raise RuntimeError("Failed to get initial parameters")
            
            self._initialize_population(initial_params)
            
            # Run DE algorithm
            best_params, best_score, history = self._run_de_algorithm()
            
            # Run final evaluation
            final_result = self._run_final_evaluation(best_params)
            
            # Save results
            self.results_manager.save_results(best_params, best_score, history, final_result)
            
            # Save to default model settings
            self._save_to_default_settings(best_params)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 60)
            self.logger.info("DE OPTIMIZATION COMPLETED")
            self.logger.info(f"🏆 Best {self.target_metric}: {best_score:.6f}")
            self.logger.info(f"⏱️ Total time: {duration}")
            self.logger.info("=" * 60)
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'history': history,
                'final_result': final_result,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'optimization_metric': self.target_metric,
                'duration': str(duration),
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"DE optimization failed: {str(e)}")
            raise
        finally:
            self._cleanup_parallel_processing()
    
    def _setup_optimization_directories(self) -> None:
            """Setup directory structure for optimization"""
            
            # Create all directories
            for directory in [self.optimization_dir, self.summa_sim_dir, self.mizuroute_sim_dir,
                            self.optimization_settings_dir, self.output_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create log directories
            (self.summa_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
            (self.mizuroute_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            # Copy settings files
            self._copy_settings_files()
            
            # Update file managers for optimization (calibration period only)
            self._update_optimization_file_managers()
    
    def _copy_settings_files(self) -> None:
        """Copy necessary settings files to optimization directory"""
        source_settings_dir = self.project_dir / "settings" / "SUMMA"
        
        if not source_settings_dir.exists():
            raise FileNotFoundError(f"Source settings directory not found: {source_settings_dir}")
        
        # SUMMA settings files to copy
        settings_files = [
            'fileManager.txt', 'modelDecisions.txt', 'outputControl.txt',
            'localParamInfo.txt', 'basinParamInfo.txt',
            'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL',
            'attributes.nc', 'coldState.nc', 'trialParams.nc', 'forcingFileList.txt'
        ]
        
        for file_name in settings_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
        
        # Copy mizuRoute settings if they exist
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"
        
        if source_mizu_dir.exists():
            dest_mizu_dir.mkdir(parents=True, exist_ok=True)
            for mizu_file in source_mizu_dir.glob("*"):
                if mizu_file.is_file():
                    shutil.copy2(mizu_file, dest_mizu_dir / mizu_file.name)
    
    def _update_optimization_file_managers(self) -> None:
        """Update file managers for optimization runs"""
        # Update SUMMA file manager
        file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        if file_manager_path.exists():
            self._update_summa_file_manager(file_manager_path)
        
        # Update mizuRoute control file if it exists
        mizu_control_path = self.optimization_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control_path.exists():
            self._update_mizuroute_control_file(mizu_control_path)
    

    def _update_summa_file_manager(self, file_manager_path: Path, use_calibration_period: bool = True) -> None:
        """
        Update SUMMA file manager - FIXED to include spinup period
        """
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        if use_calibration_period:
            # CRITICAL FIX: Include spinup period before calibration
            calibration_period_str = self.config.get('CALIBRATION_PERIOD', '')
            spinup_period_str = self.config.get('SPINUP_PERIOD', '')
            
            if calibration_period_str and spinup_period_str:
                try:
                    # Parse spinup period
                    spinup_dates = [d.strip() for d in spinup_period_str.split(',')]
                    cal_dates = [d.strip() for d in calibration_period_str.split(',')]
                    
                    if len(spinup_dates) >= 2 and len(cal_dates) >= 2:
                        # Use spinup start, calibration end
                        spinup_start = datetime.strptime(spinup_dates[0], '%Y-%m-%d').replace(hour=1, minute=0)
                        cal_end = datetime.strptime(cal_dates[1], '%Y-%m-%d').replace(hour=23, minute=0)
                        
                        sim_start = spinup_start.strftime('%Y-%m-%d %H:%M')
                        sim_end = cal_end.strftime('%Y-%m-%d %H:%M')
                        
                        self.logger.info(f"Using spinup + calibration period: {sim_start} to {sim_end}")
                    else:
                        raise ValueError("Invalid period format")
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse spinup+calibration periods: {str(e)}")
                    # Fallback to full experiment period
                    sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                    sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
            else:
                # No spinup specified, use calibration only (but warn)
                if calibration_period_str:
                    self.logger.warning("No SPINUP_PERIOD specified - model initialization may be poor")
                    cal_dates = [d.strip() for d in calibration_period_str.split(',')]
                    if len(cal_dates) >= 2:
                        cal_start = datetime.strptime(cal_dates[0], '%Y-%m-%d').replace(hour=1, minute=0)
                        cal_end = datetime.strptime(cal_dates[1], '%Y-%m-%d').replace(hour=23, minute=0)
                        sim_start = cal_start.strftime('%Y-%m-%d %H:%M')
                        sim_end = cal_end.strftime('%Y-%m-%d %H:%M')
                    else:
                        sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                        sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
                else:
                    sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                    sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        else:
            # Use full experiment period for final evaluation
            sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
            sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
            self.logger.info(f"Using full experiment period: {sim_start} to {sim_end}")
        
        # Update file manager with proper periods
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                prefix = 'run_de_final' if not use_calibration_period else 'run_de_opt'
                updated_lines.append(f"outFilePrefix        '{prefix}_{self.experiment_id}'\n")
            elif 'outputPath' in line:
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.optimization_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _update_mizuroute_control_file(self, control_path: Path) -> None:
        """Update mizuRoute control file for optimization"""
        with open(control_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if '<input_dir>' in line:
                input_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"<input_dir>             {input_path}/\n")
            elif '<output_dir>' in line:
                output_path = str(self.mizuroute_sim_dir).replace('\\', '/')
                updated_lines.append(f"<output_dir>            {output_path}/\n")
            elif '<case_name>' in line:
                updated_lines.append(f"<case_name>             run_de_opt_{self.experiment_id}\n")
            elif '<fname_qsim>' in line:
                updated_lines.append(f"<fname_qsim>            run_de_opt_{self.experiment_id}_timestep.nc\n")
            else:
                updated_lines.append(line)
        
        with open(control_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _create_calibration_target(self) -> CalibrationTarget:
        """Factory method to create appropriate calibration target"""
        calibration_var = self.config.get('CALIBRATION_VARIABLE', 'streamflow')
        
        if calibration_var == 'streamflow':
            return StreamflowTarget(self.config, self.project_dir, self.logger)
        elif calibration_var == 'snow':
            return SnowTarget(self.config, self.project_dir, self.logger)
        else:
            raise ValueError(f"Unsupported calibration variable: {calibration_var}")
    
    def _determine_population_size(self) -> int:
        """Determine appropriate population size based on parameter count"""
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size:
            return config_pop_size
        
        # Calculate based on parameter count
        total_params = (len(self.config.get('PARAMS_TO_CALIBRATE', '').split(',')) +
                       len(self.config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',')) +
                       (2 if self.config.get('CALIBRATE_DEPTH', False) else 0) +
                       (len(self.config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',')) if self.config.get('CALIBRATE_MIZUROUTE', False) else 0))
        
        return max(15, min(4 * total_params, 50))
    
    def _setup_parallel_processing(self) -> None:
        """Setup parallel processing directories and files"""
        self.logger.info(f"Setting up parallel processing with {self.num_processes} processes")
        
        self.parallel_dirs = []
        
        for proc_id in range(self.num_processes):
            # Create process-specific directories
            proc_base_dir = self.optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_summa_dir = proc_base_dir / "SUMMA"
            proc_mizuroute_dir = proc_base_dir / "mizuRoute"
            proc_summa_settings_dir = proc_base_dir / "settings" / "SUMMA"
            proc_mizu_settings_dir = proc_base_dir / "settings" / "mizuRoute"
            
            # Create directories
            for directory in [proc_base_dir, proc_summa_dir, proc_mizuroute_dir,
                             proc_summa_settings_dir, proc_mizu_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create log directories
            (proc_summa_dir / "logs").mkdir(parents=True, exist_ok=True)
            (proc_mizuroute_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            # Copy settings files
            self._copy_settings_to_process_dir(proc_summa_settings_dir, proc_mizu_settings_dir)
            
            # Update file managers for this process
            self._update_process_file_managers(proc_id, proc_summa_dir, proc_mizuroute_dir,
                                             proc_summa_settings_dir, proc_mizu_settings_dir)
            
            # Store directory info
            self.parallel_dirs.append({
                'proc_id': proc_id,
                'base_dir': proc_base_dir,
                'summa_dir': proc_summa_dir,
                'mizuroute_dir': proc_mizuroute_dir,
                'summa_settings_dir': proc_summa_settings_dir,
                'mizuroute_settings_dir': proc_mizu_settings_dir
            })
        
        self.logger.info(f"Created {len(self.parallel_dirs)} parallel working directories")
    
    def _copy_settings_to_process_dir(self, proc_summa_settings_dir: Path, proc_mizu_settings_dir: Path) -> None:
        """Copy settings files to process-specific directory"""
        # Copy SUMMA settings
        if self.optimization_settings_dir.exists():
            for settings_file in self.optimization_settings_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_summa_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
        
        # Copy mizuRoute settings
        mizu_source_dir = self.optimization_dir / "settings" / "mizuRoute"
        if mizu_source_dir.exists():
            for settings_file in mizu_source_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_mizu_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
    
    def _update_process_file_managers(self, proc_id: int, summa_dir: Path, mizuroute_dir: Path,
                                    summa_settings_dir: Path, mizu_settings_dir: Path) -> None:
        """Update file managers for a specific process"""
        # Update SUMMA file manager
        file_manager = summa_settings_dir / 'fileManager.txt'
        if file_manager.exists():
            with open(file_manager, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if 'outFilePrefix' in line:
                    updated_lines.append(f"outFilePrefix        'proc_{proc_id:02d}_de_opt_{self.experiment_id}'\n")
                elif 'outputPath' in line:
                    output_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    settings_path = str(summa_settings_dir).replace('\\', '/')
                    updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                else:
                    updated_lines.append(line)
            
            with open(file_manager, 'w') as f:
                f.writelines(updated_lines)
        
        # Update mizuRoute control file
        control_file = mizu_settings_dir / 'mizuroute.control'
        if control_file.exists():
            with open(control_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if '<input_dir>' in line:
                    input_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"<input_dir>             {input_path}/\n")
                elif '<output_dir>' in line:
                    output_path = str(mizuroute_dir).replace('\\', '/')
                    updated_lines.append(f"<output_dir>            {output_path}/\n")
                elif '<case_name>' in line:
                    updated_lines.append(f"<case_name>             proc_{proc_id:02d}_de_opt_{self.experiment_id}\n")
                elif '<fname_qsim>' in line:
                    updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_de_opt_{self.experiment_id}_timestep.nc\n")
                else:
                    updated_lines.append(line)
            
            with open(control_file, 'w') as f:
                f.writelines(updated_lines)


    def diagnose_calibration_issues(self) -> None:
        """Diagnostic function to identify calibration issues"""
        self.logger.info("🔍 RUNNING CALIBRATION DIAGNOSTICS")
        
        # 1. Check time periods
        self.logger.info("=" * 50)
        self.logger.info("TIME PERIOD ANALYSIS")
        self.logger.info("=" * 50)
        
        exp_start = self.config.get('EXPERIMENT_TIME_START')
        exp_end = self.config.get('EXPERIMENT_TIME_END')
        cal_period = self.config.get('CALIBRATION_PERIOD', '')
        spinup_period = self.config.get('SPINUP_PERIOD', '')
        
        self.logger.info(f"Experiment period: {exp_start} to {exp_end}")
        self.logger.info(f"Calibration period: {cal_period}")
        self.logger.info(f"Spinup period: {spinup_period}")
        
        if cal_period:
            cal_dates = [d.strip() for d in cal_period.split(',')]
            if len(cal_dates) >= 2:
                cal_start = pd.Timestamp(cal_dates[0])
                cal_end = pd.Timestamp(cal_dates[1])
                cal_years = (cal_end - cal_start).days / 365.25
                self.logger.info(f"Calibration length: {cal_years:.1f} years")
                
                if cal_years < 2:
                    self.logger.warning("⚠️ Calibration period is very short (<2 years)")
                elif cal_years > 15:
                    self.logger.warning("⚠️ Calibration period is very long (>15 years)")
        
        # 2. Check observed data
        self.logger.info("=" * 50)
        self.logger.info("OBSERVED DATA ANALYSIS")
        self.logger.info("=" * 50)
        
        obs_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        if obs_file.exists():
            obs_df = pd.read_csv(obs_file)
            self.logger.info(f"Observed data file: {obs_file}")
            self.logger.info(f"Observed data shape: {obs_df.shape}")
            self.logger.info(f"Observed data columns: {list(obs_df.columns)}")
            
            # Find data column
            data_col = None
            for col in obs_df.columns:
                if any(term in col.lower() for term in ['flow', 'discharge', 'q_']):
                    data_col = col
                    break
            
            if data_col:
                obs_values = pd.to_numeric(obs_df[data_col], errors='coerce')
                self.logger.info(f"Data column: {data_col}")
                self.logger.info(f"Data range: {obs_values.min():.3f} to {obs_values.max():.3f}")
                self.logger.info(f"Data mean: {obs_values.mean():.3f}")
                self.logger.info(f"Missing values: {obs_values.isna().sum()}")
                
                # Check for calibration period data
                if cal_period:
                    date_col = None
                    for col in obs_df.columns:
                        if any(term in col.lower() for term in ['date', 'time', 'datetime']):
                            date_col = col
                            break
                    
                    if date_col:
                        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
                        cal_dates = [d.strip() for d in cal_period.split(',')]
                        if len(cal_dates) >= 2:
                            cal_start = pd.Timestamp(cal_dates[0])
                            cal_end = pd.Timestamp(cal_dates[1])
                            
                            cal_mask = (obs_df['DateTime'] >= cal_start) & (obs_df['DateTime'] <= cal_end)
                            cal_data = obs_values[cal_mask]
                            
                            self.logger.info(f"Calibration period data points: {len(cal_data)}")
                            if len(cal_data) > 0:
                                self.logger.info(f"Calibration data range: {cal_data.min():.3f} to {cal_data.max():.3f}")
                                self.logger.info(f"Calibration data mean: {cal_data.mean():.3f}")
                            else:
                                self.logger.error("❌ NO DATA AVAILABLE FOR CALIBRATION PERIOD!")
        else:
            self.logger.error(f"❌ Observed data file not found: {obs_file}")
        
        # 3. Check parameter bounds
        self.logger.info("=" * 50)
        self.logger.info("PARAMETER BOUNDS ANALYSIS")
        self.logger.info("=" * 50)
        
        for param_name, bounds in self.parameter_manager.param_bounds.items():
            min_val = bounds['min']
            max_val = bounds['max']
            range_val = max_val - min_val
            self.logger.info(f"{param_name}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f})")
            
            # Check for problematic bounds
            if min_val >= max_val:
                self.logger.error(f"❌ Invalid bounds for {param_name}: min >= max")
            elif range_val < 1e-6:
                self.logger.warning(f"⚠️ Very narrow range for {param_name}")
            elif range_val > 1e6:
                self.logger.warning(f"⚠️ Very wide range for {param_name}")
        
        # 4. Test basic SUMMA run
        self.logger.info("=" * 50)
        self.logger.info("BASIC SUMMA TEST")
        self.logger.info("=" * 50)
        
        try:
            # Run SUMMA with default parameters
            default_params = self.parameter_manager.get_initial_parameters()
            if default_params:
                self.logger.info("Testing SUMMA with default parameters...")
                
                if self.parameter_manager.apply_parameters(default_params):
                    success = self.model_executor.run_models(
                        self.summa_sim_dir,
                        self.mizuroute_sim_dir,
                        self.optimization_settings_dir
                    )
                    
                    if success:
                        self.logger.info("✅ Basic SUMMA run successful")
                        
                        # Check output files
                        summa_files = list(self.summa_sim_dir.glob("*.nc"))
                        self.logger.info(f"SUMMA output files: {len(summa_files)}")
                        
                        if summa_files:
                            # Quick check of output data
                            import xarray as xr
                            with xr.open_dataset(summa_files[0]) as ds:
                                self.logger.info(f"Output variables: {list(ds.variables.keys())}")
                                
                                # Check for streamflow variable
                                for var_name in ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']:
                                    if var_name in ds.variables:
                                        var_data = ds[var_name]
                                        if len(var_data.shape) > 1:
                                            data_vals = var_data.isel({var_data.dims[1]: 0}).values
                                        else:
                                            data_vals = var_data.values
                                        
                                        self.logger.info(f"{var_name} range: {np.nanmin(data_vals):.6f} to {np.nanmax(data_vals):.6f}")
                                        self.logger.info(f"{var_name} mean: {np.nanmean(data_vals):.6f}")
                                        self.logger.info(f"{var_name} NaN count: {np.isnan(data_vals).sum()}")
                                        break
                    else:
                        self.logger.error("❌ Basic SUMMA run failed")
                else:
                    self.logger.error("❌ Failed to apply default parameters")
            else:
                self.logger.error("❌ Could not get default parameters")
                
        except Exception as e:
            self.logger.error(f"❌ Error in basic SUMMA test: {str(e)}")
        
        self.logger.info("🔍 DIAGNOSTICS COMPLETE")

    def _cleanup_parallel_processing(self) -> None:
        """Cleanup parallel processing directories"""
        if not self.use_parallel:
            return
        
        self.logger.info("Cleaning up parallel working directories")
        # Cleanup can be optionally disabled for debugging
        cleanup_parallel = self.config.get('CLEANUP_PARALLEL_DIRS', True)
        if cleanup_parallel:
            try:
                for proc_dirs in self.parallel_dirs:
                    if proc_dirs['base_dir'].exists():
                        shutil.rmtree(proc_dirs['base_dir'])
            except Exception as e:
                self.logger.warning(f"Error during parallel cleanup: {str(e)}")
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize DE population"""
        self.logger.info("Initializing DE population")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Set first individual to initial parameters
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_population()
        
        # Find best individual
        best_idx = np.nanargmax(self.population_scores)
        if not np.isnan(self.population_scores[best_idx]):
            self.best_score = self.population_scores[best_idx]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
        
        # Record initial generation
        self._record_generation(0)
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_score:.6f}")
    
    def _run_de_algorithm(self) -> Tuple[Dict, float, List]:
        """Core DE algorithm implementation"""
        self.logger.info("Running DE algorithm")
        
        for generation in range(1, self.max_iterations + 1):
            generation_start_time = datetime.now()
            
            # Create trial population
            trial_population = self._create_trial_population()
            
            # Evaluate trial population
            trial_scores = self._evaluate_trial_population(trial_population)
            
            # Selection phase
            improvements = 0
            for i in range(self.population_size):
                if not np.isnan(trial_scores[i]) and trial_scores[i] > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    improvements += 1
                    
                    # Update global best
                    if trial_scores[i] > self.best_score:
                        self.best_score = trial_scores[i]
                        self.best_params = self.parameter_manager.denormalize_parameters(trial_population[i])
                        self.logger.info(f"Gen {generation:3d}: NEW BEST! {self.target_metric}={self.best_score:.6f}")
            
            # Record generation
            self._record_generation(generation)
            
            # Log progress
            generation_duration = datetime.now() - generation_start_time
            self.logger.info(f"Gen {generation:3d}/{self.max_iterations}: "
                           f"Best={self.best_score:.6f}, Improvements={improvements}/{self.population_size} "
                           f"[{generation_duration.total_seconds():.1f}s]")
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _create_trial_population(self) -> np.ndarray:
        """Create trial population using DE mutation and crossover"""
        trial_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Select three random individuals different from target
            candidates = list(range(self.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: V = X_r1 + F * (X_r2 - X_r3)
            mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = self.population[i].copy()
            j_rand = np.random.randint(len(self.parameter_manager.all_param_names))
            
            for j in range(len(self.parameter_manager.all_param_names)):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_population[i] = trial
        
        return trial_population
    
    def _evaluate_population(self) -> None:
        """Evaluate current population"""
        if self.use_parallel:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_sequential()
    
    def _evaluate_trial_population(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population"""
        trial_scores = np.full(self.population_size, np.nan)
        
        if self.use_parallel:
            trial_scores = self._evaluate_trial_population_parallel(trial_population)
        else:
            for i in range(self.population_size):
                trial_scores[i] = self._evaluate_individual(trial_population[i])
        
        return trial_scores
    
    def _evaluate_population_sequential(self) -> None:
        """Evaluate population sequentially"""
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                self.population_scores[i] = self._evaluate_individual(self.population[i])
    
    def _evaluate_population_parallel(self) -> None:
        """Evaluate population in parallel"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                params = self.parameter_manager.denormalize_parameters(self.population[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"pop_eval_{i:03d}"
                }
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            results = self._run_parallel_evaluations(evaluation_tasks)
            
            for result in results:
                individual_id = result['individual_id']
                score = result['score']
                self.population_scores[individual_id] = score if score is not None else float('-inf')
                
                if score is not None and score > self.best_score:
                    self.best_score = score
                    self.best_params = result['params'].copy()
    
    def _evaluate_trial_population_parallel(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population in parallel"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            params = self.parameter_manager.denormalize_parameters(trial_population[i])
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"trial_{len(self.iteration_history):03d}_{i:03d}"
            }
            evaluation_tasks.append(task)
        
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        trial_scores = np.full(self.population_size, np.nan)
        for result in results:
            individual_id = result['individual_id']
            trial_scores[individual_id] = result['score'] if result['score'] is not None else float('-inf')
        
        return trial_scores

    def _run_parallel_evaluations_enhanced(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Enhanced parallel evaluation with detailed error reporting and debugging"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import subprocess
        import tempfile
        import pickle
        import time
        import random
        import gc
        
        start_time = time.time()
        num_tasks = len(evaluation_tasks)
        
        # Track consecutive failures for recovery
        if not hasattr(self, '_consecutive_parallel_failures'):
            self._consecutive_parallel_failures = 0
        
        self.logger.info(f"Starting enhanced parallel evaluation of {num_tasks} tasks with {self.num_processes} processes")
        
        # Prepare worker tasks with absolute paths
        worker_tasks = []
        for task in evaluation_tasks:
            proc_dirs = self.parallel_dirs[task['proc_id']]
            
            # Convert all paths to absolute paths
            task_data = {
                'individual_id': task['individual_id'],
                'params': task['params'],
                'proc_id': task['proc_id'],
                'evaluation_id': task['evaluation_id'],
                
                # Absolute paths for worker
                'summa_exe': str(self._get_summa_exe_path().resolve()),
                'file_manager': str((proc_dirs['summa_settings_dir'] / 'fileManager.txt').resolve()),
                'summa_dir': str(proc_dirs['summa_dir'].resolve()),
                'mizuroute_dir': str(proc_dirs['mizuroute_dir'].resolve()),
                'summa_settings_dir': str(proc_dirs['summa_settings_dir'].resolve()),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir'].resolve()),
                
                # Configuration for worker
                'config': self.config,
                'target_metric': self.target_metric,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'domain_name': self.domain_name,
                'project_dir': str(self.project_dir.resolve()),
                'original_depths': self.parameter_manager.original_depths.tolist() if self.parameter_manager.original_depths is not None else None,
            }
            
            worker_tasks.append(task_data)
        
        # Pre-execution validation
        self.logger.info("Performing pre-execution validation...")
        validation_errors = []
        
        for i, task_data in enumerate(worker_tasks):
            # Check critical files exist
            critical_files = {
                'SUMMA executable': task_data['summa_exe'],
                'File manager': task_data['file_manager'],
                'SUMMA directory': task_data['summa_dir'],
                'Settings directory': task_data['summa_settings_dir']
            }
            
            for name, path_str in critical_files.items():
                path = Path(path_str)
                if not path.exists():
                    validation_errors.append(f"Task {i}: {name} not found: {path}")
        
        if validation_errors:
            self.logger.error(f"Pre-execution validation failed with {len(validation_errors)} errors:")
            for error in validation_errors[:10]:  # Show first 10 errors
                self.logger.error(f"  {error}")
            
            # Return failed results
            return [
                {
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Pre-execution validation failed: {validation_errors[0] if validation_errors else "Unknown error"}'
                }
                for task in evaluation_tasks
            ]
        
        self.logger.info("Pre-execution validation passed")
        
        # RECOVERY LOGIC: If we've had consecutive failures, try recovery
        if self._consecutive_parallel_failures >= 3:
            self.logger.warning(f"Detected {self._consecutive_parallel_failures} consecutive parallel failures. Attempting recovery...")
            
            # Recovery measures
            gc.collect()
            time.sleep(3.0)  # Let file system settle
            
            # Reduce concurrent processes temporarily
            effective_processes = max(4, self.num_processes // 2)
            self.logger.info(f"Reducing concurrent processes from {self.num_processes} to {effective_processes} for recovery")
            
            # Touch critical files to refresh handles
            try:
                for proc_dirs in self.parallel_dirs[:effective_processes]:
                    settings_dir = Path(proc_dirs['summa_settings_dir'])
                    for critical_file in ['fileManager.txt', 'attributes.nc', 'coldState.nc', 'trialParams.nc']:
                        file_path = settings_dir / critical_file
                        if file_path.exists():
                            file_path.touch()
                            time.sleep(0.01)  # Small delay between touches
            except Exception as e:
                self.logger.debug(f"File handle refresh failed: {str(e)}")
        else:
            effective_processes = self.num_processes
        
        # FALLBACK LOGIC: If too many consecutive failures, fall back to sequential
        if self._consecutive_parallel_failures >= 5:
            self.logger.warning("Too many consecutive parallel failures. Falling back to sequential evaluation.")
            return self._run_sequential_fallback(evaluation_tasks)
        
        results = []
        completed_count = 0
        
        try:
            # Execute with reduced concurrency and batching
            batch_size = min(effective_processes, 6)  # Smaller batches
            
            for batch_start in range(0, len(worker_tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(worker_tasks))
                batch_tasks = worker_tasks[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(worker_tasks)-1)//batch_size + 1}: tasks {batch_start}-{batch_end-1}")
                
                # Add pre-batch delay to reduce file system pressure
                if batch_start > 0:
                    time.sleep(1.0)
                
                # Execute batch with enhanced error handling
                batch_results = self._execute_batch_enhanced(batch_tasks, len(batch_tasks))
                results.extend(batch_results)
                completed_count += len(batch_tasks)
                
                # Analyze batch results for debugging
                successful_in_batch = sum(1 for r in batch_results if r.get('score') is not None)
                failed_in_batch = len(batch_tasks) - successful_in_batch
                
                self.logger.info(f"Batch completed: {successful_in_batch}/{len(batch_tasks)} successful")
                
                if failed_in_batch > 0:
                    # Log some error details
                    failed_results = [r for r in batch_results if r.get('score') is None]
                    error_summary = {}
                    
                    for result in failed_results[:5]:  # Analyze first 5 failures
                        error = result.get('error', 'Unknown error')
                        # Extract the main error type
                        if 'SUMMA execution failed' in error:
                            error_type = 'SUMMA execution failed'
                        elif 'Failed to apply parameters' in error:
                            error_type = 'Parameter application failed'
                        elif 'not found' in error:
                            error_type = 'File not found'
                        else:
                            error_type = 'Other error'
                        
                        error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    
                    self.logger.warning(f"Batch failures by type: {error_summary}")
                    
                    # Log detailed debug info for first failure
                    if failed_results:
                        first_failure = failed_results[0]
                        debug_info = first_failure.get('debug_info', {})
                        
                        self.logger.debug(f"First failure debug info:")
                        self.logger.debug(f"  Stage: {debug_info.get('stage', 'unknown')}")
                        self.logger.debug(f"  Files checked: {len(debug_info.get('files_checked', []))}")
                        self.logger.debug(f"  Commands run: {debug_info.get('commands_run', [])}")
                        
                        if debug_info.get('errors'):
                            self.logger.debug(f"  Errors: {debug_info['errors'][:2]}")  # First 2 errors
                
                # Check for stale file handle errors in this batch
                stale_errors = sum(1 for r in batch_results if r.get('error') and 'stale file handle' in str(r['error']).lower())
                if stale_errors > len(batch_tasks) * 0.5:  # More than 50% stale file handle errors
                    self.logger.warning(f"High rate of stale file handle errors in batch ({stale_errors}/{len(batch_tasks)})")
                    # Add extra recovery time
                    time.sleep(2.0)
            
            # Calculate success rate
            successful_count = sum(1 for r in results if r['score'] is not None)
            success_rate = successful_count / num_tasks if num_tasks > 0 else 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"Enhanced parallel evaluation completed: {successful_count}/{num_tasks} successful "
                        f"({100*success_rate:.1f}%) in {elapsed/60:.1f} minutes")
            
            # Update failure counter based on success rate
            if success_rate >= 0.7:  # 70% or better success
                self._consecutive_parallel_failures = 0
            else:
                self._consecutive_parallel_failures += 1
                
            # If success rate is too low, warn about potential sequential fallback
            if success_rate < 0.5:
                self.logger.warning(f"Low success rate ({100*success_rate:.1f}%). Next failure may trigger sequential fallback.")
            
            # Additional debugging for failed tasks
            if successful_count < num_tasks:
                failed_results = [r for r in results if r.get('score') is None]
                
                # Categorize failures
                failure_categories = {}
                for result in failed_results:
                    error = result.get('error', 'Unknown error')
                    debug_info = result.get('debug_info', {})
                    stage = debug_info.get('stage', 'unknown')
                    
                    category = f"{stage}_failure"
                    failure_categories[category] = failure_categories.get(category, 0) + 1
                
                self.logger.info(f"Failure analysis: {failure_categories}")
                
                # Log details of first few failures for debugging
                self.logger.debug("Sample failure details:")
                for i, result in enumerate(failed_results[:3]):
                    debug_info = result.get('debug_info', {})
                    self.logger.debug(f"  Failure {i+1}: Stage={debug_info.get('stage', 'unknown')}, "
                                    f"Individual={result.get('individual_id', -1)}")
                    if debug_info.get('summa_log_content'):
                        # Log last few lines of SUMMA log
                        log_lines = debug_info['summa_log_content'].split('\n')[-5:]
                        self.logger.debug(f"    SUMMA log (last 5 lines): {log_lines}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in enhanced parallel evaluation: {str(e)}")
            self._consecutive_parallel_failures += 1
            
            # Check if this is a stale file handle error
            if 'stale file handle' in str(e).lower() or 'errno 116' in str(e).lower():
                self.logger.error("Detected stale file handle error - file system may be overloaded")
                
            # Return failed results for all tasks
            return [
                {
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Critical parallel evaluation error: {str(e)}'
                }
                for task in evaluation_tasks
            ]


    def _execute_batch_enhanced(self, batch_tasks: List[Dict], max_workers: int) -> List[Dict]:
        """Execute a batch with enhanced error handling and debugging"""
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        import time
        
        results = []
        
        try:
            # Use a conservative timeout
            timeout_seconds = 2400  # 40 minutes per task
            
            self.logger.debug(f"Starting batch execution with {max_workers} workers, timeout={timeout_seconds}s")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_evaluate_parameters_worker_safe, task_data): task_data
                    for task_data in batch_tasks
                }
                
                for future in as_completed(future_to_task, timeout=timeout_seconds + 300):
                    task_data = future_to_task[future]
                    individual_id = task_data['individual_id']
                    
                    try:
                        result = future.result(timeout=timeout_seconds)
                        
                        # Enhanced result validation
                        if result.get('score') is not None:
                            self.logger.debug(f"Task {individual_id} completed successfully: score={result['score']:.6f}")
                        else:
                            error = result.get('error', 'Unknown error')
                            self.logger.debug(f"Task {individual_id} failed: {error[:100]}...")
                            
                            # Log debug info if available
                            debug_info = result.get('debug_info', {})
                            if debug_info.get('stage'):
                                self.logger.debug(f"  Failed at stage: {debug_info['stage']}")
                            if debug_info.get('commands_run'):
                                self.logger.debug(f"  Commands attempted: {len(debug_info['commands_run'])}")
                        
                        results.append(result)
                        
                    except TimeoutError:
                        error_msg = f'Task timeout after {timeout_seconds}s'
                        self.logger.error(f"Task {individual_id} timed out")
                        results.append({
                            'individual_id': individual_id,
                            'params': task_data['params'],
                            'score': None,
                            'error': error_msg
                        })
                        
                    except Exception as e:
                        error_msg = f'Task execution error: {str(e)}'
                        self.logger.error(f"Task {individual_id} failed: {str(e)}")
                        results.append({
                            'individual_id': individual_id,
                            'params': task_data['params'],
                            'score': None,
                            'error': error_msg
                        })
        
        except Exception as e:
            self.logger.error(f"Batch execution failed: {str(e)}")
            
            # Return failed results for all tasks in this batch
            for task_data in batch_tasks:
                results.append({
                    'individual_id': task_data['individual_id'],
                    'params': task_data['params'],
                    'score': None,
                    'error': f'Batch execution failed: {str(e)}'
                })
        
        return results


    # You would replace the existing _run_parallel_evaluations method with this:
    def _run_parallel_evaluations(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Main parallel evaluation method - now uses enhanced version"""
        return self._run_parallel_evaluations_enhanced(evaluation_tasks)


    def _run_parallel_evaluations(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Quick fix version with stale file handle recovery and fallback to sequential"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import subprocess
        import tempfile
        import pickle
        import time
        import random
        import gc
        
        start_time = time.time()
        num_tasks = len(evaluation_tasks)
        
        # Track consecutive failures for recovery
        if not hasattr(self, '_consecutive_parallel_failures'):
            self._consecutive_parallel_failures = 0
        
        self.logger.info(f"Starting parallel evaluation of {num_tasks} tasks with {self.num_processes} processes")
        
        # Prepare worker tasks
        worker_tasks = []
        for task in evaluation_tasks:
            proc_dirs = self.parallel_dirs[task['proc_id']]
            
            task_data = {
                'individual_id': task['individual_id'],
                'params': task['params'],
                'proc_id': task['proc_id'],
                'evaluation_id': task['evaluation_id'],
                
                # Paths for worker
                'summa_exe': str(self._get_summa_exe_path()),
                'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
                'summa_dir': str(proc_dirs['summa_dir']),
                'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
                'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
                
                # Configuration for worker
                'config': self.config,
                'target_metric': self.target_metric,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'domain_name': self.domain_name,
                'project_dir': str(self.project_dir),
                'original_depths': self.parameter_manager.original_depths.tolist() if self.parameter_manager.original_depths is not None else None,
            }
            
            worker_tasks.append(task_data)
        
        # RECOVERY LOGIC: If we've had consecutive failures, try recovery
        if self._consecutive_parallel_failures >= 3:
            self.logger.warning(f"Detected {self._consecutive_parallel_failures} consecutive parallel failures. Attempting recovery...")
            
            # Recovery measures
            gc.collect()
            time.sleep(3.0)  # Let file system settle
            
            # Reduce concurrent processes temporarily
            effective_processes = max(4, self.num_processes // 2)
            self.logger.info(f"Reducing concurrent processes from {self.num_processes} to {effective_processes} for recovery")
            
            # Touch critical files to refresh handles
            try:
                for proc_dirs in self.parallel_dirs[:effective_processes]:
                    settings_dir = Path(proc_dirs['summa_settings_dir'])
                    for critical_file in ['fileManager.txt', 'attributes.nc', 'coldState.nc', 'trialParams.nc']:
                        file_path = settings_dir / critical_file
                        if file_path.exists():
                            file_path.touch()
                            time.sleep(0.01)  # Small delay between touches
            except Exception as e:
                self.logger.debug(f"File handle refresh failed: {str(e)}")
        else:
            effective_processes = self.num_processes
        
        # FALLBACK LOGIC: If too many consecutive failures, fall back to sequential
        if self._consecutive_parallel_failures >= 5:
            self.logger.warning("Too many consecutive parallel failures. Falling back to sequential evaluation.")
            return self._run_sequential_fallback(evaluation_tasks)
        
        results = []
        completed_count = 0
        
        try:
            # Execute with reduced concurrency and batching
            batch_size = min(effective_processes, 6)  # Smaller batches
            
            for batch_start in range(0, len(worker_tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(worker_tasks))
                batch_tasks = worker_tasks[batch_start:batch_end]
                
                # Add pre-batch delay to reduce file system pressure
                if batch_start > 0:
                    time.sleep(1.0)
                
                # Execute batch with timeout and retry
                batch_results = self._execute_batch_safe(batch_tasks, len(batch_tasks))
                results.extend(batch_results)
                completed_count += len(batch_tasks)
                
                # Check for stale file handle errors in this batch
                stale_errors = sum(1 for r in batch_results if r.get('error') and 'stale file handle' in str(r['error']).lower())
                if stale_errors > len(batch_tasks) * 0.5:  # More than 50% stale file handle errors
                    self.logger.warning(f"High rate of stale file handle errors in batch ({stale_errors}/{len(batch_tasks)})")
                    # Add extra recovery time
                    time.sleep(2.0)
            
            # Calculate success rate
            successful_count = sum(1 for r in results if r['score'] is not None)
            success_rate = successful_count / num_tasks if num_tasks > 0 else 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"Parallel evaluation completed: {successful_count}/{num_tasks} successful "
                        f"({100*success_rate:.1f}%) in {elapsed/60:.1f} minutes")
            
            # Update failure counter based on success rate
            if success_rate >= 0.7:  # 70% or better success
                self._consecutive_parallel_failures = 0
            else:
                self._consecutive_parallel_failures += 1
                
            # If success rate is too low, warn about potential sequential fallback
            if success_rate < 0.5:
                self.logger.warning(f"Low success rate ({100*success_rate:.1f}%). Next failure may trigger sequential fallback.")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in parallel evaluation: {str(e)}")
            self._consecutive_parallel_failures += 1
            
            # Check if this is a stale file handle error
            if 'stale file handle' in str(e).lower() or 'errno 116' in str(e).lower():
                self.logger.error("Detected stale file handle error - file system may be overloaded")
                
            # Return failed results for all tasks
            return [
                {
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Critical parallel evaluation error: {str(e)}'
                }
                for task in evaluation_tasks
            ]

    def _execute_batch_safe(self, batch_tasks: List[Dict], max_workers: int) -> List[Dict]:
        """Execute a batch with enhanced error handling and recovery"""
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        import time
        
        results = []
        
        try:
            # Use a conservative timeout
            timeout_seconds = 2400  # 40 minutes per task
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_evaluate_parameters_worker_safe, task_data): task_data
                    for task_data in batch_tasks
                }
                
                for future in as_completed(future_to_task, timeout=timeout_seconds + 300):
                    task_data = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=timeout_seconds)
                        results.append(result)
                        
                    except TimeoutError:
                        self.logger.error(f"Task {task_data['individual_id']} timed out")
                        results.append({
                            'individual_id': task_data['individual_id'],
                            'params': task_data['params'],
                            'score': None,
                            'error': f'Task timeout after {timeout_seconds}s'
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Task {task_data['individual_id']} failed: {str(e)}")
                        results.append({
                            'individual_id': task_data['individual_id'],
                            'params': task_data['params'],
                            'score': None,
                            'error': f'Task execution error: {str(e)}'
                        })
        
        except Exception as e:
            self.logger.error(f"Batch execution failed: {str(e)}")
            
            # Return failed results for all tasks in this batch
            for task_data in batch_tasks:
                results.append({
                    'individual_id': task_data['individual_id'],
                    'params': task_data['params'],
                    'score': None,
                    'error': f'Batch execution failed: {str(e)}'
                })
        
        return results

    def _run_sequential_fallback(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Fallback to sequential evaluation when parallel processing fails"""
        self.logger.info("Running sequential fallback evaluation")
        
        results = []
        
        for i, task in enumerate(evaluation_tasks):
            self.logger.info(f"Sequential evaluation {i+1}/{len(evaluation_tasks)}")
            
            try:
                # Extract parameters and evaluate sequentially
                params = task['params']
                normalized_params = self.parameter_manager.normalize_parameters(params)
                score = self._evaluate_individual(normalized_params)
                
                results.append({
                    'individual_id': task['individual_id'],
                    'params': params,
                    'score': score if score != float('-inf') else None,
                    'error': None if score != float('-inf') else 'Sequential evaluation failed'
                })
                
            except Exception as e:
                self.logger.error(f"Sequential evaluation {i+1} failed: {str(e)}")
                results.append({
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Sequential evaluation error: {str(e)}'
                })
        
        # Reset parallel failure counter after successful sequential run
        successful_count = sum(1 for r in results if r['score'] is not None)
        if successful_count > 0:
            self._consecutive_parallel_failures = max(0, self._consecutive_parallel_failures - 1)
        
        return results


    
    def _evaluate_individual(self, normalized_params: np.ndarray) -> float:
        """Evaluate a single parameter set (sequential mode)"""
        try:
            # Denormalize parameters
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            
            # Apply parameters to files
            if not self.parameter_manager.apply_parameters(params):
                return float('-inf')
            
            # Run models
            if not self.model_executor.run_models(
                self.summa_sim_dir, 
                self.mizuroute_sim_dir, 
                self.optimization_settings_dir
            ):
                return float('-inf')
            
            # Calculate performance metrics
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir)
            if not metrics:
                return float('-inf')
            
            # Extract target metric
            score = self._extract_target_metric(metrics)
            
            # Apply negation for metrics where lower is better
            if self.target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                score = -score
            
            return score if score is not None and not np.isnan(score) else float('-inf')
            
        except Exception as e:
            self.logger.debug(f"Parameter evaluation failed: {str(e)}")
            return float('-inf')
    
    def _extract_target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        """Extract target metric from metrics dictionary"""
        # Try exact match
        if self.target_metric in metrics:
            return metrics[self.target_metric]
        
        # Try with calibration prefix
        calib_key = f"Calib_{self.target_metric}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try without prefix (remove Calib_ or Eval_)
        for key, value in metrics.items():
            if key.endswith(f"_{self.target_metric}"):
                return value
        
        # Fallback to first available metric
        return next(iter(metrics.values())) if metrics else None
    
    def _record_generation(self, generation: int) -> None:
        """Record generation statistics"""
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        
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
    
    def _run_final_evaluation(self, best_params: Dict) -> Optional[Dict]:
        """Run final evaluation with best parameters over full period"""
        self.logger.info("Running final evaluation with best parameters")
        
        try:
            # Update file manager for full period
            self._update_file_manager_for_final_run()
            
            # Apply best parameters
            if not self.parameter_manager.apply_parameters(best_params):
                self.logger.error("Failed to apply best parameters for final run")
                return None
            
            # Run models
            if not self.model_executor.run_models(
                self.summa_sim_dir,
                self.mizuroute_sim_dir,
                self.optimization_settings_dir
            ):
                self.logger.error("Final model run failed")
                return None
            
            # Calculate metrics for full period
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir)
            
            if metrics:
                self.logger.info("Final evaluation completed successfully")
                return {
                    'final_metrics': metrics,
                    'summa_success': True,
                    'mizuroute_success': self.calibration_target.needs_routing()
                }
            else:
                return {'summa_success': False, 'mizuroute_success': False}
                
        except Exception as e:
            self.logger.error(f"Error in final evaluation: {str(e)}")
            return None
        finally:
            # Reset file manager back to optimization mode
            self._update_summa_file_manager(self.optimization_settings_dir / 'fileManager.txt')
    
    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager to use full experiment period"""
        file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        
        if not file_manager_path.exists():
            return
        
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        # Use full experiment period
        sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        'run_de_final_{self.experiment_id}'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _save_to_default_settings(self, best_params: Dict) -> bool:
        """Save best parameters to default model settings"""
        try:
            self.logger.info("Saving best parameters to default model settings")
            
            default_settings_dir = self.project_dir / "settings" / "SUMMA"
            if not default_settings_dir.exists():
                self.logger.error("Default settings directory not found")
                return False
            
            # Backup existing files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save hydraulic parameters to trialParams.nc
            hydraulic_params = {k: v for k, v in best_params.items() 
                              if k not in ['total_mult', 'shape_factor']}
            
            if hydraulic_params:
                trial_params_path = default_settings_dir / "trialParams.nc"
                if trial_params_path.exists():
                    backup_path = default_settings_dir / f"trialParams_backup_{timestamp}.nc"
                    shutil.copy2(trial_params_path, backup_path)
                
                # Generate new trialParams.nc
                param_manager = ParameterManager(self.config, self.logger, default_settings_dir)
                if param_manager._generate_trial_params_file(hydraulic_params):
                    self.logger.info("✅ Saved optimized hydraulic parameters")
                else:
                    self.logger.error("Failed to save hydraulic parameters")
                    return False
            
            # Save soil depths to coldState.nc
            if self.config.get('CALIBRATE_DEPTH', False) and 'total_mult' in best_params:
                coldstate_path = default_settings_dir / "coldState.nc"
                if coldstate_path.exists():
                    backup_path = default_settings_dir / f"coldState_backup_{timestamp}.nc"
                    shutil.copy2(coldstate_path, backup_path)
                
                # Copy optimized coldState
                optimized_coldstate = self.optimization_settings_dir / "coldState.nc"
                if optimized_coldstate.exists():
                    shutil.copy2(optimized_coldstate, coldstate_path)
                    self.logger.info("✅ Saved optimized soil depths")
            
            # Save mizuRoute parameters
            if self.config.get('CALIBRATE_MIZUROUTE', False):
                mizu_param_file = self.project_dir / "settings" / "mizuRoute" / "param.nml.default"
                if mizu_param_file.exists():
                    backup_path = mizu_param_file.parent / f"param.nml.backup_{timestamp}"
                    shutil.copy2(mizu_param_file, backup_path)
                    
                    # Copy optimized parameters
                    optimized_mizu = self.optimization_dir / "settings" / "mizuRoute" / "param.nml.default"
                    if optimized_mizu.exists():
                        shutil.copy2(optimized_mizu, mizu_param_file)
                        self.logger.info("✅ Saved optimized mizuRoute parameters")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to default settings: {str(e)}")
            return False
    
    def _get_summa_exe_path(self) -> Path:
        """Get SUMMA executable path"""
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        return summa_path / self.config.get('SUMMA_EXE')


# ============= WORKER FUNCTIONS FOR PARALLEL PROCESSING =============

def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    """Worker function for parallel parameter evaluation"""
    import sys
    import logging
    from pathlib import Path
    import subprocess
    import tempfile
    
    try:
        # Extract task info
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']
        
        # Setup process logger
        logger = logging.getLogger(f'worker_{proc_id}')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[P{proc_id:02d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Get paths
        summa_exe = Path(task_data['summa_exe'])
        file_manager = Path(task_data['file_manager'])
        summa_dir = Path(task_data['summa_dir'])
        mizuroute_dir = Path(task_data['mizuroute_dir'])
        summa_settings_dir = Path(task_data['summa_settings_dir'])
        
        # Verify critical paths
        if not summa_exe.exists():
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
        
        # Apply parameters to files
        if not _apply_parameters_worker(params, task_data, summa_settings_dir, logger):
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'Failed to apply parameters'
            }
        
        # Run SUMMA
        if not _run_summa_worker(summa_exe, file_manager, summa_dir, logger):
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'SUMMA simulation failed'
            }
        
        # Run mizuRoute if needed
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        if calibration_var == 'streamflow':  # Only streamflow needs routing potentially
            config = task_data['config']
            needs_routing = _needs_mizuroute_routing_worker(config)
            
            if needs_routing:
                if not _run_mizuroute_worker(task_data, mizuroute_dir, logger):
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': 'mizuRoute simulation failed'
                    }
        
        # Calculate metrics
        score = _calculate_metrics_worker(task_data, summa_dir, mizuroute_dir, logger)
        
        if score is None:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': 'Failed to calculate metrics'
            }
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': score,
            'error': None
        }
        
    except Exception as e:
        import traceback
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Worker exception: {str(e)}\n{traceback.format_exc()}'
        }


def _apply_parameters_worker(params: Dict, task_data: Dict, settings_dir: Path, logger) -> bool:
    """Apply parameters in worker process"""
    try:
        config = task_data['config']
        
        # Handle soil depth parameters
        if config.get('CALIBRATE_DEPTH', False) and 'total_mult' in params and 'shape_factor' in params:
            if not _update_soil_depths_worker(params, task_data, settings_dir, logger):
                return False
        
        # Handle mizuRoute parameters
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
            if mizuroute_params and any(p in params for p in mizuroute_params):
                if not _update_mizuroute_params_worker(params, task_data, logger):
                    return False
        
        # Generate trial parameters file (excluding depth and mizuRoute parameters)
        depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()] if config.get('CALIBRATE_MIZUROUTE', False) else []
        
        hydraulic_params = {k: v for k, v in params.items() 
                          if k not in depth_params + mizuroute_params}
        
        if hydraulic_params:
            if not _generate_trial_params_worker(hydraulic_params, settings_dir, logger):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error applying parameters: {str(e)}")
        return False


def _update_soil_depths_worker(params: Dict, task_data: Dict, settings_dir: Path, logger) -> bool:
    """Update soil depths in worker process"""
    try:
        original_depths_list = task_data.get('original_depths')
        if not original_depths_list:
            return True
        
        original_depths = np.array(original_depths_list)
        
        total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
        shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
        
        # Calculate new depths
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
        
        # Calculate heights
        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]
        
        # Update coldState.nc
        coldstate_path = settings_dir / 'coldState.nc'
        if not coldstate_path.exists():
            return False
        
        with nc.Dataset(coldstate_path, 'r+') as ds:
            if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                return False
            
            num_hrus = ds.dimensions['hru'].size
            for h in range(num_hrus):
                ds.variables['mLayerDepth'][:, h] = new_depths
                ds.variables['iLayerHeight'][:, h] = heights
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating soil depths: {str(e)}")
        return False


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, logger) -> bool:
    """Update mizuRoute parameters in worker process"""
    try:
        config = task_data['config']
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        
        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        param_file = mizuroute_settings_dir / "param.nml.default"
        
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


def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger) -> bool:
    """Generate trial parameters file in worker process with file locking"""
    import time
    import random
    import os
    
    try:
        if not params:
            return True
        
        trial_params_path = settings_dir / 'trialParams.nc'
        attr_file_path = settings_dir / 'attributes.nc'
        
        if not attr_file_path.exists():
            logger.error(f"Attributes file not found: {attr_file_path}")
            return False
        
        # Add retry logic with file locking
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Create temporary file first, then move it
                temp_path = trial_params_path.with_suffix(f'.tmp_{os.getpid()}_{random.randint(1000,9999)}')
                
                # Define parameter levels
                routing_params = ['routingGammaShape', 'routingGammaScale']
                basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
                gru_level_params = routing_params + basin_params
                
                with xr.open_dataset(attr_file_path) as ds:
                    num_hrus = ds.sizes.get('hru', 1)
                    num_grus = ds.sizes.get('gru', 1)
                    hru_ids = ds['hruId'].values if 'hruId' in ds else np.arange(1, num_hrus + 1)
                    gru_ids = ds['gruId'].values if 'gruId' in ds else np.array([1])
                
                # Write to temporary file with exclusive access
                with nc.Dataset(temp_path, 'w', format='NETCDF4') as output_ds:
                    # Create dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)
                    
                    # Create coordinate variables
                    hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                    hru_var[:] = hru_ids
                    
                    gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                    gru_var[:] = gru_ids
                    
                    # Add parameters
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)
                        
                        if param_values_array.ndim > 1:
                            param_values_array = param_values_array.flatten()
                        
                        if param_name in gru_level_params:
                            # GRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) >= num_grus:
                                param_var[:] = param_values_array[:num_grus]
                            else:
                                param_var[:] = param_values_array[0]
                        else:
                            # HRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) == num_hrus:
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_hrus]
                
                # Atomically move temporary file to final location
                try:
                    os.chmod(temp_path, 0o664)  # Set appropriate permissions
                    temp_path.rename(trial_params_path)
                    return True
                except Exception as move_error:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise move_error
                
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                # Clean up temp file if it exists
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to generate trial params after {max_retries} attempts: {str(e)}")
                    return False
        
        return False
        
    except Exception as e:
        logger.error(f"Error generating trial params: {str(e)}")
        return False


def _run_summa_worker(summa_exe: Path, file_manager: Path, summa_dir: Path, logger) -> bool:
    """Run SUMMA in worker process"""
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"summa_worker.log"
        
        # Set environment for single-threaded execution
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['MKL_NUM_THREADS'] = '1'
        
        # Run SUMMA
        cmd = f"{summa_exe} -m {file_manager}"
        
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                env=env
            )
        
        return True
        
    except Exception as e:
        logger.error(f"SUMMA execution failed: {str(e)}")
        return False


def _run_mizuroute_worker(task_data: Dict, mizuroute_dir: Path, logger) -> bool:
    """Run mizuRoute in worker process"""
    try:
        config = task_data['config']
        
        # Get mizuRoute executable
        mizu_path = config.get('INSTALL_PATH_MIZUROUTE', 'default')
        if mizu_path == 'default':
            mizu_path = Path(config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = Path(task_data['mizuroute_settings_dir']) / 'mizuroute.control'
        
        if not mizu_exe.exists() or not control_file.exists():
            return False
        
        # Create log directory
        log_dir = mizuroute_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mizuroute_worker.log"
        
        # Run mizuRoute
        cmd = f"{mizu_exe} {control_file}"
        
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(control_file.parent)
            )
        
        return True
        
    except Exception as e:
        logger.error(f"mizuRoute execution failed: {str(e)}")
        return False


def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
    
    if domain_method not in ['point', 'lumped']:
        return True
    
    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True
    
    return False


def _calculate_metrics_worker(task_data: Dict, summa_dir: Path, mizuroute_dir: Path, logger) -> Optional[float]:
    """Calculate metrics in worker process - FIXED for spinup handling"""
    try:
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        config = task_data['config']
        target_metric = task_data['target_metric']
        
        # Load observed data
        if calibration_var == 'streamflow':
            obs_file = Path(task_data['project_dir']) / "observations" / "streamflow" / "preprocessed" / f"{task_data['domain_name']}_streamflow_processed.csv"
        elif calibration_var == 'snow':
            obs_file = Path(task_data['project_dir']) / "observations" / "snow" / "swe" / "processed" / f"{task_data['domain_name']}_swe_processed.csv"
        else:
            return None
        
        if not obs_file.exists():
            logger.error(f"Observed data not found: {obs_file}")
            return None
        
        obs_df = pd.read_csv(obs_file)
        
        # Find columns
        date_col = None
        data_col = None
        
        for col in obs_df.columns:
            col_lower = col.lower()
            if date_col is None and any(term in col_lower for term in ['date', 'time', 'datetime']):
                date_col = col
            if data_col is None:
                if calibration_var == 'streamflow' and any(term in col_lower for term in ['flow', 'discharge', 'q_']):
                    data_col = col
                elif calibration_var == 'snow' and any(term in col_lower for term in ['swe', 'snow']):
                    data_col = col
        
        if not date_col or not data_col:
            logger.error("Could not identify date/data columns")
            return None
        
        # Process observed data
        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
        obs_df.set_index('DateTime', inplace=True)
        observed_data = obs_df[data_col]
        
        # CRITICAL FIX: Filter observed data to CALIBRATION period only (not spinup)
        calibration_period_str = config.get('CALIBRATION_PERIOD', '')
        if calibration_period_str:
            try:
                dates = [d.strip() for d in calibration_period_str.split(',')]
                if len(dates) >= 2:
                    cal_start = pd.Timestamp(dates[0].strip())
                    cal_end = pd.Timestamp(dates[1].strip())
                    
                    # Filter observed data to calibration period ONLY
                    cal_mask = (observed_data.index >= cal_start) & (observed_data.index <= cal_end)
                    observed_data = observed_data[cal_mask]
                    
                    logger.debug(f"Filtered observed data to calibration period: {cal_start} to {cal_end}")
                    logger.debug(f"Observed data points: {len(observed_data)}")
            except Exception as e:
                logger.warning(f"Could not filter to calibration period: {str(e)}")
        
        # Get simulated data (this will be from spinup+calibration run)
        if calibration_var == 'streamflow' and _needs_mizuroute_routing_worker(config):
            sim_files = list(mizuroute_dir.glob("*.nc"))
            if not sim_files:
                logger.error("No mizuRoute output files found")
                return None
            sim_file = sim_files[0]
            simulated_data = _extract_streamflow_from_mizuroute_worker(sim_file, config, logger)
        else:
            if calibration_var == 'streamflow':
                sim_files = list(summa_dir.glob("*timestep.nc"))
                if not sim_files:
                    logger.error("No SUMMA timestep files found")
                    return None
                sim_file = sim_files[0]
                simulated_data = _extract_streamflow_from_summa_worker(sim_file, config, logger)
            elif calibration_var == 'snow':
                sim_files = list(summa_dir.glob("*day.nc"))
                if not sim_files:
                    logger.error("No SUMMA daily files found")
                    return None
                sim_file = sim_files[0]
                simulated_data = _extract_swe_from_summa_worker(sim_file, logger)
            else:
                return None
        
        if simulated_data is None or len(simulated_data) == 0:
            logger.error("Failed to extract simulated data")
            return None
        
        # CRITICAL FIX: Filter simulated data to CALIBRATION period only (exclude spinup)
        if calibration_period_str:
            try:
                dates = [d.strip() for d in calibration_period_str.split(',')]
                if len(dates) >= 2:
                    cal_start = pd.Timestamp(dates[0].strip())
                    cal_end = pd.Timestamp(dates[1].strip())
                    
                    # Filter simulated data to calibration period ONLY
                    sim_mask = (simulated_data.index >= cal_start) & (simulated_data.index <= cal_end)
                    simulated_data = simulated_data[sim_mask]
                    
                    logger.debug(f"Filtered simulated data to calibration period: {cal_start} to {cal_end}")
                    logger.debug(f"Simulated data points: {len(simulated_data)}")
            except Exception as e:
                logger.warning(f"Could not filter simulated data to calibration period: {str(e)}")
        
        # Check for reasonable data ranges
        if len(simulated_data) == 0:
            logger.error("No simulated data after calibration period filtering")
            return None
        
        # Check for NaN or infinite values
        sim_nan_count = simulated_data.isna().sum()
        if sim_nan_count > 0:
            logger.warning(f"Simulated data contains {sim_nan_count} NaN values")
        
        sim_inf_count = np.isinf(simulated_data).sum()
        if sim_inf_count > 0:
            logger.warning(f"Simulated data contains {sim_inf_count} infinite values")
        
        # Align time series
        simulated_data.index = simulated_data.index.round('h')
        common_idx = observed_data.index.intersection(simulated_data.index)
        
        if len(common_idx) == 0:
            logger.error("No common time indices between observed and simulated data")
            logger.debug(f"Observed period: {observed_data.index.min()} to {observed_data.index.max()}")
            logger.debug(f"Simulated period: {simulated_data.index.min()} to {simulated_data.index.max()}")
            return None
        
        obs_common = observed_data.loc[common_idx]
        sim_common = simulated_data.loc[common_idx]
        
        logger.debug(f"Common data points for evaluation: {len(common_idx)}")
        logger.debug(f"Observed range: {obs_common.min():.3f} to {obs_common.max():.3f}")
        logger.debug(f"Simulated range: {sim_common.min():.3f} to {sim_common.max():.3f}")
        
        # Calculate metrics
        metrics = _calculate_performance_metrics_worker(obs_common, sim_common)
        
        # Extract target metric
        score = metrics.get(target_metric, metrics.get('KGE', np.nan))
        
        # IMPORTANT: For NSE, don't apply negation! NSE should be maximized (1 is perfect)
        # Only negate metrics where lower values are better
        if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
            score = -score
        
        logger.debug(f"Calculated {target_metric}: {score}")
        
        return score if not np.isnan(score) else None
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _extract_streamflow_from_mizuroute_worker(sim_file: Path, config: Dict, logger) -> Optional[pd.Series]:
    """Extract streamflow from mizuRoute output"""
    try:
        with xr.open_dataset(sim_file) as ds:
            reach_id = int(config.get('SIM_REACH_ID', 123))
            
            if 'reachID' not in ds.variables:
                return None
            
            reach_ids = ds['reachID'].values
            reach_indices = np.where(reach_ids == reach_id)[0]
            
            if len(reach_indices) == 0:
                return None
            
            reach_index = reach_indices[0]
            
            # Find streamflow variable
            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        return var.isel(seg=reach_index).to_pandas()
                    elif 'reachID' in var.dims:
                        return var.isel(reachID=reach_index).to_pandas()
            
            return None
            
    except Exception as e:
        logger.error(f"Error extracting mizuRoute streamflow: {str(e)}")
        return None


def _extract_streamflow_from_summa_worker(sim_file: Path, config: Dict, logger) -> Optional[pd.Series]:
    """Extract streamflow from SUMMA output"""
    try:
        with xr.open_dataset(sim_file) as ds:
            # Find streamflow variable
            for var_name in ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']:
                if var_name in ds.variables:
                    var = ds[var_name]
                    
                    if len(var.shape) > 1:
                        if 'hru' in var.dims:
                            sim_data = var.isel(hru=0).to_pandas()
                        elif 'gru' in var.dims:
                            sim_data = var.isel(gru=0).to_pandas()
                        else:
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                            else:
                                sim_data = var.to_pandas()
                    else:
                        sim_data = var.to_pandas()
                    
                    # Convert units (m/s to m³/s) - use default catchment area
                    catchment_area = 1e6  # 1 km² default
                    return sim_data * catchment_area
            
            return None
            
    except Exception as e:
        logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
        return None


def _extract_swe_from_summa_worker(sim_file: Path, logger) -> Optional[pd.Series]:
    """Extract SWE from SUMMA daily output"""
    try:
        with xr.open_dataset(sim_file) as ds:
            if 'scalarSWE' not in ds.variables:
                return None
            
            var = ds['scalarSWE']
            
            if len(var.shape) > 1:
                if 'hru' in var.dims:
                    return var.isel(hru=0).to_pandas()
                elif 'gru' in var.dims:
                    return var.isel(gru=0).to_pandas()
                else:
                    non_time_dims = [dim for dim in var.dims if dim != 'time']
                    if non_time_dims:
                        return var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        return var.to_pandas()
            else:
                return var.to_pandas()
                
    except Exception as e:
        logger.error(f"Error extracting SWE: {str(e)}")
        return None


def _calculate_performance_metrics_worker(observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics"""
    try:
        # Clean data
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}
        
        # Calculate metrics
        mean_obs = observed.mean()
        
        # NSE
        nse_num = ((observed - simulated) ** 2).sum()
        nse_den = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
        
        # RMSE
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # KGE
        r = observed.corr(simulated)
        alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
        beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse}
        
    except Exception:
        return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    config_path = Path("config.yaml")  # Replace with actual config path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run optimization
    optimizer = DEOptimizer(config, logger)
    results = optimizer.run_de_optimization()
    
    print(f"Optimization completed. Best {results['optimization_metric']}: {results['best_score']:.6f}")
    print(f"Results saved to: {results['output_dir']}")