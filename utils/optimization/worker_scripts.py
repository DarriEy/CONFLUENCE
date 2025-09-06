#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker Scripts for CONFLUENCE Optimization

This module contains all the worker functions used for parallel processing
in the CONFLUENCE optimization framework. These functions are designed to
run in separate processes and handle model execution, parameter application,
and metrics calculation.
"""

import os
import sys
import time
import random
import traceback
import signal
import gc
import logging
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr


def _evaluate_parameters_worker_safe(task_data: Dict) -> Dict:
    """Safe wrapper for parameter evaluation with error handling and retries"""
    worker_seed = task_data.get('random_seed')
    if worker_seed is not None: 
        random.seed(worker_seed)
        np.random.seed(worker_seed)

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
                
                # Call the worker function
                result = _evaluate_parameters_worker(task_data)
                
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


def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    """Enhanced worker with inline metrics calculation and runtime tracking"""
    import time
    import traceback
    
    # Start timing the core evaluation
    eval_start_time = time.time()
    
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
        
        # Setup process logger
        logger = logging.getLogger(f'worker_{proc_id}_{individual_id}')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[P{proc_id:02d}-I{individual_id:03d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting evaluation of individual {individual_id}")
        
        # Check multi-objective flag
        is_multiobjective = task_data.get('multiobjective', False)
        logger.info(f"Multi-objective evaluation: {is_multiobjective}")
        
        # DETERMINE ROUTING NEEDS EARLY
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        needs_routing = False
        
        if calibration_var == 'streamflow':
            config = task_data['config']
            domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
            
            if domain_method not in ['point', 'lumped'] or (domain_method == 'lumped' and routing_delineation == 'river_network'):
                needs_routing = True
        
        logger.info(f"Needs routing: {needs_routing}")
        
        # Convert paths
        debug_info['stage'] = 'path_setup'
        summa_exe = Path(task_data['summa_exe']).resolve()
        file_manager = Path(task_data['file_manager']).resolve()
        summa_dir = Path(task_data['summa_dir']).resolve()
        mizuroute_dir = Path(task_data['mizuroute_dir']).resolve()
        summa_settings_dir = Path(task_data['summa_settings_dir']).resolve()
        
        # Verify paths
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
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None if is_multiobjective else None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }
        
        # Apply parameters
        debug_info['stage'] = 'parameter_application'
        logger.info("Applying parameters")
        if not _apply_parameters_worker(params, task_data, summa_settings_dir, logger, debug_info):
            error_msg = 'Failed to apply parameters'
            logger.error(error_msg)
            eval_runtime = time.time() - eval_start_time
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'objectives': None if is_multiobjective else None,
                'error': error_msg,
                'debug_info': debug_info,
                'runtime': eval_runtime
            }
        
        # Run SUMMA
        debug_info['stage'] = 'summa_execution'
        logger.info("Running SUMMA")
        summa_start = time.time()
        if not _run_summa_worker(summa_exe, file_manager, summa_dir, logger, debug_info):
            error_msg = 'SUMMA simulation failed'
            logger.error(error_msg)
            eval_runtime = time.time() - eval_start_time
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'objectives': None if is_multiobjective else None,
                'error': error_msg,
                'debug_info': debug_info,
                'runtime': eval_runtime
            }
        summa_runtime = time.time() - summa_start
        
        # Handle mizuRoute routing
        mizuroute_runtime = 0.0
        
        if needs_routing:
            # Check if we need lumped-to-distributed conversion
            debug_info['stage'] = 'lumped_to_distributed_conversion'
            config = task_data['config']
            domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            routing_delineation = config.get('ROUTING_DELINEATION', 'river_network')
            
            if config.get('ROUTING_DELINEATION', 'river_network') == 'river_network':
                logger.info("Converting lumped SUMMA output to distributed format")
                if not _convert_lumped_to_distributed_worker(task_data, summa_dir, logger, debug_info):
                    error_msg = 'Lumped-to-distributed conversion failed'
                    logger.error(error_msg)
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'objectives': None if is_multiobjective else None,
                        'error': error_msg,
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }
            
            debug_info['stage'] = 'mizuroute_execution'
            logger.info("Running mizuRoute")
            mizu_start = time.time()
            if not _run_mizuroute_worker(task_data, mizuroute_dir, logger, debug_info):
                error_msg = 'mizuRoute simulation failed'
                logger.error(error_msg)
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None if is_multiobjective else None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }
            mizuroute_runtime = time.time() - mizu_start
        
        # Calculate metrics using INLINE method to avoid import issues
        debug_info['stage'] = 'metrics_calculation'
        metrics_start = time.time()
        
        if is_multiobjective:
            logger.info("Starting INLINE multi-objective metrics calculation")
            
            try:
                # Use inline metrics calculation
                metrics = _calculate_metrics_inline_worker(
                    summa_dir, 
                    mizuroute_dir if needs_routing else None, 
                    task_data['config'], 
                    logger
                )
                
                if not metrics:
                    error_msg = 'Inline metrics calculation failed'
                    logger.error(error_msg)
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'objectives': None,
                        'error': error_msg,
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }
                
                logger.info(f"Inline metrics calculated: {list(metrics.keys())}")
                
                # Extract NSE and KGE
                nse_score = metrics.get('NSE') or metrics.get('Calib_NSE')
                kge_score = metrics.get('KGE') or metrics.get('Calib_KGE')
                
                logger.info(f"Extracted NSE: {nse_score}, KGE: {kge_score}")
                
                # Handle None/NaN values
                if nse_score is None or (isinstance(nse_score, float) and np.isnan(nse_score)):
                    logger.warning("NSE is None/NaN, setting to -1.0")
                    nse_score = -1.0
                
                if kge_score is None or (isinstance(kge_score, float) and np.isnan(kge_score)):
                    logger.warning("KGE is None/NaN, setting to -1.0")
                    kge_score = -1.0
                
                # Create objectives array
                objectives = [float(nse_score), float(kge_score)]
                logger.info(f"Final objectives: {objectives}")
                
                # Set score based on target metric
                target_metric = task_data.get('target_metric', 'NSE')
                if target_metric == 'NSE':
                    score = float(nse_score)
                elif target_metric == 'KGE':
                    score = float(kge_score)
                else:
                    score = float(kge_score)  # Default to KGE
                
                metrics_runtime = time.time() - metrics_start
                eval_runtime = time.time() - eval_start_time
                
                logger.info(f"Multi-objective completed. NSE: {nse_score:.6f}, KGE: {kge_score:.6f}, Score: {score:.6f}")
                logger.info(f"Runtime breakdown: Total={eval_runtime:.1f}s, SUMMA={summa_runtime:.1f}s, mizuRoute={mizuroute_runtime:.1f}s, Metrics={metrics_runtime:.1f}s")
                
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': score,
                    'objectives': objectives,
                    'error': None,
                    'debug_info': debug_info,
                    'runtime': eval_runtime,
                    'runtime_breakdown': {
                        'total': eval_runtime,
                        'summa': summa_runtime,
                        'mizuroute': mizuroute_runtime,
                        'metrics': metrics_runtime
                    }
                }
                
            except Exception as e:
                import traceback
                error_msg = f"Exception in inline multi-objective calculation: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'objectives': None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }
        
        else:
            # Single-objective evaluation
            logger.info("Single-objective evaluation using inline calculation")
            
            try:
                metrics = _calculate_metrics_inline_worker(
                    summa_dir, 
                    mizuroute_dir if needs_routing else None, 
                    task_data['config'], 
                    logger
                )
                
                if not metrics:
                    error_msg = 'Inline metrics calculation failed'
                    logger.error(error_msg)
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': error_msg,
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }
                
                # Extract target metric
                target_metric = task_data['target_metric']
                score = metrics.get(target_metric) or metrics.get(f'Calib_{target_metric}')
                
                if score is None:
                    # Try to find any metric with the target name
                    for key, value in metrics.items():
                        if target_metric in key:
                            score = value
                            break
                
                if score is None:
                    logger.error(f"Could not extract {target_metric} from metrics: {list(metrics.keys())}")
                    eval_runtime = time.time() - eval_start_time
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': f'Could not extract {target_metric}',
                        'debug_info': debug_info,
                        'runtime': eval_runtime
                    }
                
                # Apply negation for minimize metrics
                if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                    score = -score
                
                metrics_runtime = time.time() - metrics_start
                eval_runtime = time.time() - eval_start_time
                
                logger.info(f"Single-objective completed. {target_metric}: {score:.6f}")
                logger.info(f"Runtime breakdown: Total={eval_runtime:.1f}s, SUMMA={summa_runtime:.1f}s, mizuRoute={mizuroute_runtime:.1f}s, Metrics={metrics_runtime:.1f}s")
                
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': score,
                    'error': None,
                    'debug_info': debug_info,
                    'runtime': eval_runtime,
                    'runtime_breakdown': {
                        'total': eval_runtime,
                        'summa': summa_runtime,
                        'mizuroute': mizuroute_runtime,
                        'metrics': metrics_runtime
                    }
                }
                
            except Exception as e:
                import traceback
                error_msg = f"Exception in single-objective calculation: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                eval_runtime = time.time() - eval_start_time
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'error': error_msg,
                    'debug_info': debug_info,
                    'runtime': eval_runtime
                }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f'Worker exception at stage {debug_info.get("stage", "unknown")}: {str(e)}'
        debug_info['errors'].append(f"{error_msg}\nTraceback:\n{error_trace}")
        
        is_multiobjective = task_data.get('multiobjective', False)
        eval_runtime = time.time() - eval_start_time
        
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'objectives': None if is_multiobjective else None,
            'error': error_msg,
            'debug_info': debug_info,
            'full_traceback': error_trace,
            'runtime': eval_runtime
        }


def fix_summa_time_precision(input_file, output_file=None):
    """
    Round SUMMA time dimension to nearest hour to fix mizuRoute compatibility
    Fixed to handle timezone mismatch issues
    """
    import xarray as xr
    import numpy as np
    import os
    import tempfile
    import shutil
    
    print(f"Opening {input_file}")
    
    try:
        # Open without decoding times to avoid conflicts
        ds = xr.open_dataset(input_file, decode_times=False)
        
        print(f"Original time range: {ds.time.min().values} to {ds.time.max().values}")
        
        # Convert to datetime, round, then convert back
        time_vals = ds.time.values
        
        # Convert to pandas timestamps for rounding
        import pandas as pd
        
        # First convert the time values to actual timestamps
        if 'units' in ds.time.attrs:
            time_units = ds.time.attrs['units']
            print(f"Original time units: {time_units}")
            
            # Parse the reference time
            if 'since' in time_units:
                ref_time_str = time_units.split('since')[1].strip()
                ref_time = pd.Timestamp(ref_time_str)
                
                # Get the time unit (hours, days, seconds, etc.)
                unit = time_units.split()[0].lower()
                
                # Convert to timedelta and add to reference
                if unit.startswith('hour'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')
                elif unit.startswith('day'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='D')
                elif unit.startswith('second'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='s')
                elif unit.startswith('minute'):
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='min')
                else:
                    # Default to hours
                    timestamps = ref_time + pd.to_timedelta(time_vals, unit='h')
            else:
                # Fallback: assume hourly from a standard reference
                ref_time = pd.Timestamp('1990-01-01')
                timestamps = ref_time + pd.to_timedelta(time_vals, unit='H')
        else:
            # No units attribute, try to interpret as hours since 1990
            ref_time = pd.Timestamp('1990-01-01')
            timestamps = ref_time + pd.to_timedelta(time_vals, unit='H')
        
        # Round to nearest hour
        rounded_timestamps = timestamps.round('H')
        
        print(f"Rounded time range: {rounded_timestamps.min()} to {rounded_timestamps.max()}")
        
        # FIX: Ensure both timestamps are timezone-naive for consistent calculation
        ref_time_calc = pd.Timestamp('1990-01-01')
        
        # Remove timezone from rounded_timestamps if present
        if rounded_timestamps.tz is not None:
            rounded_timestamps = rounded_timestamps.tz_localize(None)
            print("Removed timezone from rounded timestamps")
        
        # Convert back to hours since reference time
        rounded_hours = (rounded_timestamps - ref_time_calc).total_seconds() / 3600.0
        
        # Create new time coordinate with cleared attributes
        new_time = xr.DataArray(
            rounded_hours,
            dims=('time',),
            attrs={}  # Start with empty attributes
        )
        
        # Set clean attributes
        new_time.attrs['units'] = 'hours since 1990-01-01 00:00:00'
        new_time.attrs['calendar'] = 'standard'
        new_time.attrs['long_name'] = 'time'
        
        # Replace time coordinate
        ds = ds.assign_coords(time=new_time)
        
        # Clean up encoding to avoid conflicts
        if 'time' in ds.encoding:
            del ds.encoding['time']
        
        # Load data into memory and close original
        ds.load()
        original_ds = ds
        ds = ds.copy()  # Create a clean copy
        original_ds.close()
        
        # Determine output path
        output_path = output_file if output_file else input_file
        print(f"Saving to {output_path}")
        
        # Write to temporary file first, then move to final location
        temp_file = None
        try:
            # Create temporary file in same directory
            temp_dir = os.path.dirname(output_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=temp_dir) as tmp:
                temp_file = tmp.name
            
            # Save to temporary file with clean encoding
            ds.to_netcdf(temp_file, format='NETCDF4')
            ds.close()
            
            # Make output file writable if overwriting
            if output_file is None and os.path.exists(input_file):
                os.chmod(input_file, 0o664)
            
            # Atomically move to final location
            shutil.move(temp_file, output_path)
            temp_file = None  # Successfully moved
            
            print("Done!")
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            raise e
            
    except Exception as e:
        print(f"Error fixing time precision: {e}")
        raise


def _convert_lumped_to_distributed_worker(task_data: Dict, summa_dir: Path, logger, debug_info: Dict) -> bool:
    """Convert lumped SUMMA output for distributed routing"""
    try:
        import xarray as xr
        import numpy as np
        import shutil
        import tempfile
        import os
        
        # Find SUMMA timestep file
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            logger.error("No SUMMA timestep files found for conversion")
            return False
        
        summa_file = timestep_files[0]
        logger.info(f"Converting SUMMA file: {summa_file}")
        
        # Load topology to get HRU information
        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        topology_file = mizuroute_settings_dir / task_data['config'].get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
        
        if not topology_file.exists():
            logger.error(f"Topology file not found: {topology_file}")
            return False
        
        with xr.open_dataset(topology_file) as topo_ds:
            # Get HRU information - use first HRU ID as lumped GRU ID
            hru_ids = topo_ds['hruId'].values
            n_hrus = len(hru_ids)
            lumped_gru_id = 1  # Use ID=1 for consistency
            
            logger.info(f"Creating single lumped GRU (ID={lumped_gru_id}) for {n_hrus} HRUs in topology")
        
        # Create backup of original file before modification
        backup_file = summa_file.with_suffix('.nc.backup')
        if not backup_file.exists():
            try:
                shutil.copy2(summa_file, backup_file)
                logger.info(f"Created backup: {backup_file.name}")
            except Exception as e:
                logger.warning(f"Could not create backup: {str(e)}")
        
        # Ensure the original file is writable
        try:
            os.chmod(summa_file, 0o664)
        except Exception as e:
            logger.warning(f"Could not change file permissions: {str(e)}")
        
        # Load and convert SUMMA output
        summa_ds = None
        try:
            # Open without decoding times to avoid conversion issues
            summa_ds = xr.open_dataset(summa_file, decode_times=False)
            
            routing_var = task_data['config'].get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
            available_vars = list(summa_ds.variables.keys())
            
            # Find the best variable to use
            source_var = None
            if routing_var in summa_ds:
                source_var = routing_var
                logger.info(f"Using configured routing variable: {routing_var}")
            else:
                # Try fallback variables
                fallback_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
                for var in fallback_vars:
                    if var in summa_ds:
                        source_var = var
                        logger.info(f"Routing variable {routing_var} not found, using: {source_var}")
                        break
            
            if source_var is None:
                logger.error(f"No suitable routing variable found in {available_vars}")
                return False
            
            # Create mizuRoute forcing dataset
            mizuForcing = xr.Dataset()
            
            # Copy time coordinate (preserve original format)
            original_time = summa_ds['time']
            mizuForcing['time'] = xr.DataArray(
                original_time.values,
                dims=('time',),
                attrs=dict(original_time.attrs)
            )
            
            # Clean up time units if needed
            if 'units' in mizuForcing['time'].attrs:
                time_units = mizuForcing['time'].attrs['units']
                if 'T' in time_units:
                    mizuForcing['time'].attrs['units'] = time_units.replace('T', ' ')
            
            # Create single GRU using lumped GRU ID
            mizuForcing['gru'] = xr.DataArray([lumped_gru_id], dims=('gru',))
            mizuForcing['gruId'] = xr.DataArray([lumped_gru_id], dims=('gru',))
            
            # Extract runoff data
            var_data = summa_ds[source_var]
            runoff_data = var_data.values
            
            # Handle different shapes
            if len(runoff_data.shape) == 2:
                if runoff_data.shape[1] > 1:
                    runoff_data = runoff_data.mean(axis=1)
                    logger.info(f"Used mean across {var_data.shape[1]} spatial elements")
                else:
                    runoff_data = runoff_data[:, 0]
            else:
                runoff_data = runoff_data.flatten()
            
            # Keep as single GRU: (time,) -> (time, 1)
            single_gru_data = runoff_data[:, np.newaxis]
            
            # Create runoff variable
            mizuForcing[routing_var] = xr.DataArray(
                single_gru_data, dims=('time', 'gru'),
                attrs={'long_name': 'Lumped runoff for distributed routing', 'units': 'm/s'}
            )
            
            # Copy global attributes
            mizuForcing.attrs.update(summa_ds.attrs)
            
            # Load data and close original
            mizuForcing.load()
            summa_ds.close()
            summa_ds = None
            
        except Exception as e:
            if summa_ds is not None:
                summa_ds.close()
            raise e
        
        # Write to temporary file first, then atomically move
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.nc', 
                dir=summa_dir,
                prefix='temp_mizu_'
            ) as tmp:
                temp_file = Path(tmp.name)
            
            # Save to temporary file
            mizuForcing.to_netcdf(temp_file, format='NETCDF4')
            mizuForcing.close()
            
            # Set permissions and move
            os.chmod(temp_file, 0o664)
            shutil.move(str(temp_file), str(summa_file))
            temp_file = None
            
            logger.info(f"Successfully converted SUMMA file: single lumped GRU for distributed routing")
            
            # CRITICAL: Now fix time precision for mizuRoute compatibility
            fix_summa_time_precision(summa_file)
            logger.info("Fixed SUMMA time precision for mizuRoute compatibility")
            
            return True
            
        except Exception as e:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            raise e
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        debug_info['errors'].append(f"Lumped-to-distributed conversion error: {str(e)}")
        return False


def _get_catchment_area_worker(config: Dict, logger) -> float:
    """Get actual catchment area for unit conversion (worker version)"""
    try:
        domain_name = config.get('DOMAIN_NAME')
        project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
        
        # Try basin shapefile first
        basin_path = project_dir / "shapefiles" / "river_basins"
        basin_files = list(basin_path.glob("*.shp"))
        
        if basin_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(basin_files[0])
                area_col = config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                
                logger.debug(f"Found basin shapefile: {basin_files[0]}")
                logger.debug(f"Looking for area column: {area_col}")
                logger.debug(f"Available columns: {list(gdf.columns)}")
                
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    logger.debug(f"Total area from column: {total_area}")
                    
                    if 0 < total_area < 1e12:  # Reasonable area check
                        logger.info(f"Using catchment area from shapefile: {total_area:.0f} m²")
                        return total_area
                
                # Fallback: calculate from geometry
                if gdf.crs and gdf.crs.is_geographic:
                    # Reproject to UTM for area calculation
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                
                geom_area = gdf.geometry.area.sum()
                logger.info(f"Using catchment area from geometry: {geom_area:.0f} m²")
                return geom_area
                
            except ImportError:
                logger.warning("geopandas not available for area calculation")
            except Exception as e:
                logger.warning(f"Error reading basin shapefile: {str(e)}")
        
        # Alternative: try catchment shapefile
        catchment_path = project_dir / "shapefiles" / "catchment"
        catchment_files = list(catchment_path.glob("*.shp"))
        
        if catchment_files:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(catchment_files[0])
                area_col = config.get('CATCHMENT_SHP_AREA', 'HRU_area')
                
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:
                        logger.info(f"Using catchment area from catchment shapefile: {total_area:.0f} m²")
                        return total_area
                        
            except Exception as e:
                logger.warning(f"Error reading catchment shapefile: {str(e)}")
        
    except Exception as e:
        logger.warning(f"Could not calculate catchment area: {str(e)}")
    
    # Fallback
    logger.warning("Using default catchment area: 1,000,000 m²")
    return 1e6


def _calculate_metrics_inline_worker(summa_dir: Path, mizuroute_dir: Path, config: Dict, logger) -> Dict:
    """Calculate metrics inline without using CalibrationTarget classes"""
    try:
        import xarray as xr
        import pandas as pd
        import numpy as np
        
        logger.info("DEBUG: Starting inline metrics calculation")
        logger.info(f"DEBUG: SUMMA dir: {summa_dir}")
        logger.info(f"DEBUG: mizuRoute dir: {mizuroute_dir}")
        logger.info(f"DEBUG: SUMMA dir exists: {summa_dir.exists()}")
        logger.info(f"DEBUG: mizuRoute dir exists: {mizuroute_dir.exists() if mizuroute_dir else 'None'}")
        
        # Priority 1: Look for mizuRoute output files first (already in m³/s)
        sim_files = []
        use_mizuroute = False
        catchment_area = None
        
        if mizuroute_dir and mizuroute_dir.exists():
            mizu_files = list(mizuroute_dir.glob("*.nc"))
            logger.info(f"DEBUG: Found {len(mizu_files)} mizuRoute .nc files")
            for f in mizu_files[:3]:  # Show first 3
                logger.info(f"DEBUG: mizuRoute file: {f.name}")
            
            if mizu_files:
                sim_files = mizu_files
                use_mizuroute = True
                logger.info("DEBUG: Using mizuRoute files (already in m³/s)")
        
        # Priority 2: If no mizuRoute files, look for SUMMA files (need m/s to m³/s conversion)
        if not sim_files:
            summa_files = list(summa_dir.glob("*timestep.nc"))
            logger.info(f"DEBUG: Found {len(summa_files)} SUMMA timestep files")
            for f in summa_files[:3]:  # Show first 3
                logger.info(f"DEBUG: SUMMA file: {f.name}")
            
            if summa_files:
                sim_files = summa_files
                use_mizuroute = False
                logger.info("DEBUG: Using SUMMA files (need m/s to m³/s conversion)")
                
                # Get the ACTUAL catchment area for unit conversion
                try:
                    catchment_area = _get_catchment_area_worker(config, logger)
                    logger.info(f"DEBUG: Catchment area for conversion: {catchment_area:.0f} m²")
                except Exception as e:
                    logger.error(f"DEBUG: Error getting catchment area: {str(e)}")
                    catchment_area = 1e6  # Default fallback
                    logger.info(f"DEBUG: Using fallback catchment area: {catchment_area:.0f} m²")
        
        # Check if we found any simulation files
        if not sim_files:
            logger.error("DEBUG: No simulation files found")
            logger.error(f"DEBUG: SUMMA dir contents: {list(summa_dir.glob('*')) if summa_dir.exists() else 'DIR_NOT_EXISTS'}")
            if mizuroute_dir and mizuroute_dir.exists():
                logger.error(f"DEBUG: mizuRoute dir contents: {list(mizuroute_dir.glob('*'))}")
            return None
        
        sim_file = sim_files[0]
        logger.info(f"DEBUG: Using simulation file: {sim_file}")
        
        # Extract simulated streamflow
        logger.info(f"DEBUG: Extracting simulated streamflow (use_mizuroute={use_mizuroute})...")
        
        try:
            with xr.open_dataset(sim_file) as ds:
                logger.info(f"DEBUG: Dataset variables: {list(ds.variables.keys())}")
                logger.info(f"DEBUG: Dataset dimensions: {dict(ds.sizes)}")
                
                if use_mizuroute:
                    # mizuRoute output - select segment with highest average runoff (outlet)
                    logger.info("DEBUG: Selecting segment with highest average runoff (outlet)")
                    
                    # Find routing variable to use for selection
                    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                    routing_var = None
                    
                    for var_name in routing_vars:
                        if var_name in ds.variables:
                            routing_var = var_name
                            logger.info(f"DEBUG: Found routing variable: {routing_var}")
                            break
                    
                    if routing_var is None:
                        logger.error("DEBUG: No routing variable found in mizuRoute output")
                        logger.error(f"DEBUG: Available variables: {list(ds.variables.keys())}")
                        return None
                    
                    var = ds[routing_var]
                    logger.info(f"DEBUG: Variable dimensions: {var.dims}")
                    logger.info(f"DEBUG: Variable shape: {var.shape}")
                    
                    try:
                        # Calculate average runoff for each segment to find outlet
                        if 'seg' in var.dims:
                            # Calculate mean runoff across time for each segment
                            segment_means = var.mean(dim='time').values
                            outlet_seg_idx = np.argmax(segment_means)
                            logger.info(f"DEBUG: Found outlet at segment index {outlet_seg_idx} with mean runoff {segment_means[outlet_seg_idx]:.3f} m³/s")
                            
                            # Extract time series for outlet segment
                            sim_data = var.isel(seg=outlet_seg_idx).to_pandas()
                            
                        elif 'reachID' in var.dims:
                            # Calculate mean runoff across time for each reach
                            reach_means = var.mean(dim='time').values
                            outlet_reach_idx = np.argmax(reach_means)
                            logger.info(f"DEBUG: Found outlet at reach index {outlet_reach_idx} with mean runoff {reach_means[outlet_reach_idx]:.3f} m³/s")
                            
                            # Extract time series for outlet reach
                            sim_data = var.isel(reachID=outlet_reach_idx).to_pandas()
                            
                        else:
                            logger.error(f"DEBUG: Unexpected dimensions for {routing_var}: {var.dims}")
                            return None
                        
                        logger.info(f"DEBUG: Extracted {routing_var} from outlet segment with {len(sim_data)} timesteps (mizuRoute - no unit conversion)")
                        logger.info(f"DEBUG: Sim data range: {sim_data.min():.3f} to {sim_data.max():.3f} m³/s")
                        
                    except Exception as e:
                        logger.error(f"DEBUG: Error extracting outlet segment from {routing_var}: {str(e)}")
                        return None
                        
                else:
                    # SUMMA output - convert from m/s to m³/s using ACTUAL area
                    summa_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
                    sim_data = None
                    
                    for var_name in summa_vars:
                        if var_name in ds.variables:
                            logger.info(f"DEBUG: Found SUMMA variable: {var_name}")
                            var = ds[var_name]
                            logger.info(f"DEBUG: Variable dimensions: {var.dims}")
                            logger.info(f"DEBUG: Variable shape: {var.shape}")
                            
                            try:
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
                                
                                # Convert units for SUMMA (m/s to m³/s) using ACTUAL catchment area
                                logger.info(f"DEBUG: Converting SUMMA units: {var_name} * {catchment_area:.0f} m²")
                                logger.info(f"DEBUG: Pre-scaling range: {sim_data.min():.6f} to {sim_data.max():.6f} m/s")
                                sim_data = sim_data * catchment_area
                                logger.info(f"DEBUG: Post-scaling range: {sim_data.min():.3f} to {sim_data.max():.3f} m³/s")
                                
                                logger.info(f"DEBUG: Extracted {var_name} with {len(sim_data)} timesteps, applied area scaling")
                                break
                            except Exception as e:
                                logger.error(f"DEBUG: Error extracting {var_name}: {str(e)}")
                                continue
                    
                    if sim_data is None:
                        logger.error(f"DEBUG: No SUMMA variable found or extracted successfully")
                        return None
        
        except Exception as e:
            logger.error(f"DEBUG: Error reading simulation file {sim_file}: {str(e)}")
            return None
        
        # Load observed data
        logger.info("DEBUG: Loading observed data...")
        
        try:
            domain_name = config.get('DOMAIN_NAME')
            project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
            obs_path = project_dir / "observations" / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"
            
            logger.info(f"DEBUG: Domain name: {domain_name}")
            logger.info(f"DEBUG: Project dir: {project_dir}")
            logger.info(f"DEBUG: Observed data path: {obs_path}")
            logger.info(f"DEBUG: Project dir exists: {project_dir.exists()}")
            logger.info(f"DEBUG: Observed data exists: {obs_path.exists()}")
            
            if not obs_path.exists():
                logger.error(f"DEBUG: Observed data not found: {obs_path}")
                if project_dir.exists():
                    logger.error(f"DEBUG: Project dir contents: {list(project_dir.glob('*'))}")
                    obs_dir = project_dir / "observations"
                    if obs_dir.exists():
                        logger.error(f"DEBUG: Observations dir contents: {list(obs_dir.glob('*'))}")
                        streamflow_dir = obs_dir / "streamflow"
                        if streamflow_dir.exists():
                            logger.error(f"DEBUG: Streamflow dir contents: {list(streamflow_dir.glob('*'))}")
                            preproc_dir = streamflow_dir / "preprocessed"
                            if preproc_dir.exists():
                                logger.error(f"DEBUG: Preprocessed dir contents: {list(preproc_dir.glob('*'))}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            logger.info(f"DEBUG: Loaded observed data with {len(obs_df)} rows and columns: {list(obs_df.columns)}")
            
            # Find date and flow columns
            date_col = None
            for col in obs_df.columns:
                if any(term in col.lower() for term in ['date', 'time', 'datetime']):
                    date_col = col
                    break
            
            flow_col = None
            for col in obs_df.columns:
                if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                    flow_col = col
                    break
            
            logger.info(f"DEBUG: Date column: {date_col}")
            logger.info(f"DEBUG: Flow column: {flow_col}")
            
            if not date_col or not flow_col:
                logger.error(f"DEBUG: Could not identify date/flow columns. Date: {date_col}, Flow: {flow_col}")
                return None
            
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            obs_data = obs_df[flow_col]
            
            logger.info(f"DEBUG: Observed data period: {obs_data.index.min()} to {obs_data.index.max()}")
            logger.info(f"DEBUG: Observed data range: {obs_data.min():.3f} to {obs_data.max():.3f}")
            
        except Exception as e:
            logger.error(f"DEBUG: Error loading observed data: {str(e)}")
            import traceback
            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            return None
        
        logger.info(f"DEBUG: Simulated data period: {sim_data.index.min()} to {sim_data.index.max()}")
        logger.info(f"DEBUG: Simulated data range: {sim_data.min():.3f} to {sim_data.max():.3f}")
        logger.info(f"DEBUG: Simulated data frequency: {sim_data.index.freq}")
        logger.info(f"DEBUG: Simulated data timezone: {sim_data.index.tz}")
        logger.info(f"DEBUG: First 5 sim timestamps: {sim_data.index[:5].tolist()}")
        
        # Filter to calibration period
        logger.info("DEBUG: Filtering to calibration period...")
        
        cal_period = config.get('CALIBRATION_PERIOD', '')
        if cal_period:
            try:
                dates = [d.strip() for d in cal_period.split(',')]
                if len(dates) >= 2:
                    start_date = pd.Timestamp(dates[0])
                    end_date = pd.Timestamp(dates[1])
                    
                    logger.info(f"DEBUG: Filtering to calibration period: {start_date} to {end_date}")
                    
                    obs_mask = (obs_data.index >= start_date) & (obs_data.index <= end_date)
                    obs_period = obs_data[obs_mask]
                    
                    # ENHANCED: Check simulation time format before rounding
                    logger.info(f"DEBUG: Sim data before rounding - first 5: {sim_data.index[:5].tolist()}")
                    logger.info(f"DEBUG: Sim data sample times: {sim_data.index[::100][:5].tolist()}")  # Every 100th
                    
                    # More careful time rounding - check if we need it
                    sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
                    logger.info(f"DEBUG: Simulation time step: {sim_time_diff}")
                    
                    # Only round if time step is roughly hourly
                    if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
                        sim_data.index = sim_data.index.round('h')
                        logger.info("DEBUG: Rounded simulation times to nearest hour")
                    else:
                        logger.info(f"DEBUG: Not rounding - time step is {sim_time_diff}")
                    
                    sim_mask = (sim_data.index >= start_date) & (sim_data.index <= end_date)
                    sim_period = sim_data[sim_mask]
                    
                    logger.info(f"DEBUG: Filtered obs points: {len(obs_period)}")
                    logger.info(f"DEBUG: Filtered sim points: {len(sim_period)}")
                    
                    # ENHANCED: Check time alignment before intersection
                    logger.info(f"DEBUG: Obs period: {obs_period.index.min()} to {obs_period.index.max()}")
                    logger.info(f"DEBUG: Sim period: {sim_period.index.min()} to {sim_period.index.max()}")
                    logger.info(f"DEBUG: Obs timezone: {obs_period.index.tz}")
                    logger.info(f"DEBUG: Sim timezone: {sim_period.index.tz}")
                    
                else:
                    logger.warning("DEBUG: Invalid calibration period format, using full data")
                    obs_period = obs_data
                    sim_period = sim_data
                    
                    # Same careful rounding for full data
                    sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
                    if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
                        sim_period.index = sim_period.index.round('h')
                        
            except Exception as e:
                logger.error(f"DEBUG: Error parsing calibration period: {str(e)}")
                obs_period = obs_data
                sim_period = sim_data
                
                # Same careful rounding for error case
                sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
                if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
                    sim_period.index = sim_period.index.round('h')
        else:
            logger.info("DEBUG: No calibration period specified, using full data")
            obs_period = obs_data
            sim_period = sim_data
            
            # Same careful rounding
            sim_time_diff = sim_data.index[1] - sim_data.index[0] if len(sim_data) > 1 else pd.Timedelta(hours=1)
            if pd.Timedelta(minutes=45) <= sim_time_diff <= pd.Timedelta(minutes=75):
                sim_period.index = sim_period.index.round('h')
        
        # ENHANCED: Detailed alignment analysis
        logger.info("DEBUG: Analyzing time alignment...")
        
        # Check for timezone mismatches and try to fix
        if obs_period.index.tz is not None and sim_period.index.tz is None:
            logger.info("DEBUG: Converting sim times to observed timezone")
            sim_period.index = sim_period.index.tz_localize(obs_period.index.tz)
        elif obs_period.index.tz is None and sim_period.index.tz is not None:
            logger.info("DEBUG: Converting obs times to simulation timezone")
            obs_period.index = obs_period.index.tz_localize(sim_period.index.tz)
        elif obs_period.index.tz != sim_period.index.tz:
            logger.info(f"DEBUG: Converting timezones - obs: {obs_period.index.tz}, sim: {sim_period.index.tz}")
            sim_period.index = sim_period.index.tz_convert(obs_period.index.tz)
        
        # Sample timestamps for alignment checking
        logger.info(f"DEBUG: Obs sample times: {obs_period.index[::max(1,len(obs_period)//5)][:5].tolist()}")
        logger.info(f"DEBUG: Sim sample times: {sim_period.index[::max(1,len(sim_period)//5)][:5].tolist()}")
        
        # Find intersection
        common_idx = obs_period.index.intersection(sim_period.index)
        logger.info(f"DEBUG: Common time indices: {len(common_idx)}")
        
        if len(common_idx) == 0:
            logger.error("DEBUG: No common time indices - checking for near matches")
            
            # Try to find near matches (within 1 hour)
            obs_times = obs_period.index
            sim_times = sim_period.index
            
            # Find closest matches
            if len(obs_times) > 0 and len(sim_times) > 0:
                time_diffs = []
                for obs_time in obs_times[:10]:  # Check first 10
                    closest_sim = sim_times[np.argmin(np.abs(sim_times - obs_time))]
                    diff = abs(closest_sim - obs_time)
                    time_diffs.append(diff)
                    logger.error(f"DEBUG: Obs {obs_time} closest to Sim {closest_sim} (diff: {diff})")
                
                min_diff = min(time_diffs) if time_diffs else pd.Timedelta(days=1)
                logger.error(f"DEBUG: Minimum time difference: {min_diff}")
                
                # If differences are small, try approximate matching
                if min_diff <= pd.Timedelta(hours=1):
                    logger.info("DEBUG: Attempting approximate time matching (±30 min)")
                    
                    # Create aligned series by finding nearest neighbors
                    aligned_obs = []
                    aligned_sim = []
                    aligned_times = []
                    
                    for obs_time in obs_times:
                        # Find closest sim time within 30 minutes
                        time_diffs = np.abs(sim_times - obs_time)
                        min_diff_idx = np.argmin(time_diffs)
                        min_diff = time_diffs[min_diff_idx]
                        
                        if min_diff <= pd.Timedelta(minutes=30):
                            aligned_obs.append(obs_period.loc[obs_time])
                            aligned_sim.append(sim_period.iloc[min_diff_idx])
                            aligned_times.append(obs_time)
                    
                    if len(aligned_obs) > 0:
                        obs_common = pd.Series(aligned_obs, index=aligned_times)
                        sim_common = pd.Series(aligned_sim, index=aligned_times)
                        logger.info(f"DEBUG: Approximate matching found {len(aligned_obs)} pairs")
                    else:
                        logger.error("DEBUG: No approximate matches found")
                        return None
                else:
                    logger.error("DEBUG: Time differences too large for alignment")
                    return None
            else:
                logger.error("DEBUG: Empty time series")
                return None
        else:
            # Normal intersection worked
            obs_common = pd.to_numeric(obs_period.loc[common_idx], errors='coerce')
            sim_common = pd.to_numeric(sim_period.loc[common_idx], errors='coerce')
            logger.info(f"DEBUG: Successfully aligned {len(common_idx)} time points")
        
        # Remove invalid data
        logger.info("DEBUG: Cleaning data...")
        
        logger.info(f"DEBUG: Before cleaning - obs: {len(obs_common)}, sim: {len(sim_common)}")
        logger.info(f"DEBUG: Obs NaN count: {obs_common.isna().sum()}")
        logger.info(f"DEBUG: Sim NaN count: {sim_common.isna().sum()}")
        logger.info(f"DEBUG: Obs <= 0 count: {(obs_common <= 0).sum()}")
        logger.info(f"DEBUG: Sim <= 0 count: {(sim_common <= 0).sum()}")
        
        valid = ~(obs_common.isna() | sim_common.isna() | (obs_common <= 0) | (sim_common <= 0))
        obs_valid = obs_common[valid]
        sim_valid = sim_common[valid]
        
        logger.info(f"DEBUG: Valid data points after cleaning: {len(obs_valid)}")
        
        if len(obs_valid) < 10:
            logger.error(f"DEBUG: Insufficient valid data points: {len(obs_valid)}")
            return None
        
        logger.info(f"DEBUG: Final obs range: {obs_valid.min():.3f} to {obs_valid.max():.3f}")
        logger.info(f"DEBUG: Final sim range: {sim_valid.min():.3f} to {sim_valid.max():.3f}")
        
        # Calculate metrics
        logger.info("DEBUG: Calculating metrics...")
        
        try:
            # Calculate NSE
            mean_obs = obs_valid.mean()
            nse_num = ((obs_valid - sim_valid) ** 2).sum()
            nse_den = ((obs_valid - mean_obs) ** 2).sum()
            nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
            
            # Calculate KGE
            r = obs_valid.corr(sim_valid)
            alpha = sim_valid.std() / obs_valid.std() if obs_valid.std() != 0 else np.nan
            beta = sim_valid.mean() / mean_obs if mean_obs != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            # Calculate additional metrics
            rmse = np.sqrt(((obs_valid - sim_valid) ** 2).mean())
            mae = (obs_valid - sim_valid).abs().mean()
            pbias = 100 * (sim_valid.sum() - obs_valid.sum()) / obs_valid.sum() if obs_valid.sum() != 0 else np.nan
            
            logger.info(f"DEBUG: Calculated metrics - NSE: {nse:.6f}, KGE: {kge:.6f}, RMSE: {rmse:.6f}")
            logger.info(f"DEBUG: Additional metrics - r: {r:.6f}, alpha: {alpha:.6f}, beta: {beta:.6f}")
            
            result = {
                'Calib_NSE': nse,
                'Calib_KGE': kge,
                'Calib_RMSE': rmse,
                'Calib_MAE': mae,
                'Calib_PBIAS': pbias,
                'Calib_r': r,
                'Calib_alpha': alpha,
                'Calib_beta': beta,
                'NSE': nse,
                'KGE': kge,
                'RMSE': rmse,
                'MAE': mae,
                'PBIAS': pbias
            }
            
            logger.info("DEBUG: Metrics calculation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"DEBUG: Error in metrics calculation: {str(e)}")
            import traceback
            logger.error(f"DEBUG: Metrics traceback: {traceback.format_exc()}")
            return None
        
    except ImportError as e:
        logger.error(f"DEBUG: Import error: {str(e)}")
        return None
    except FileNotFoundError as e:
        logger.error(f"DEBUG: File not found error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"DEBUG: Error in inline metrics calculation: {str(e)}")
        import traceback
        logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return None


def _apply_parameters_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Apply parameters consistently with sequential approach"""
    try:
        config = task_data['config']
        logger.debug(f"Applying parameters: {list(params.keys())} (consistent method)")
        
        # Parse parameter lists EXACTLY as ParameterManager does
        local_params = [p.strip() for p in config.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        basin_params = [p.strip() for p in config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        mizuroute_params = []
        
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]
        
        # 1. Handle soil depth parameters
        if depth_params and 'total_mult' in params and 'shape_factor' in params:
            logger.debug("Updating soil depths (consistent)")
            if not _update_soil_depths_worker(params, task_data, settings_dir, logger, debug_info):
                return False
        
        # 2. Handle mizuRoute parameters
        if mizuroute_params and any(p in params for p in mizuroute_params):
            logger.debug("Updating mizuRoute parameters (consistent)")
            if not _update_mizuroute_params_worker(params, task_data, logger, debug_info):
                return False
        
        # 3. Generate trial parameters file (same exclusion logic as ParameterManager)
        hydrological_params = {k: v for k, v in params.items() 
                          if k not in depth_params + mizuroute_params}
        
        if hydrological_params:
            logger.debug(f"Generating trial parameters file with: {list(hydrological_params.keys())} (consistent)")
            if not _generate_trial_params_worker(hydrological_params, settings_dir, logger, debug_info):
                return False
        
        logger.debug("Parameter application completed successfully (consistent)")
        return True
        
    except Exception as e:
        error_msg = f"Error applying parameters (consistent): {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_soil_depths_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
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


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, logger, debug_info: Dict) -> bool:
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


def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced trial parameters generation with better error handling and file locking"""
    import time
    import os
    
    try:
        if not params:
            logger.debug("No hydrological parameters to write")
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


def _run_summa_worker(summa_exe: Path, file_manager: Path, summa_dir: Path, logger, debug_info: Dict) -> bool:
    """SUMMA execution without log files to save disk space"""
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
        
        debug_info['commands_run'].append(f"SUMMA: {cmd}")
        
        # Verify executable permissions
        if not os.access(summa_exe, os.X_OK):
            error_msg = f"SUMMA executable is not executable: {summa_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        # Run SUMMA 
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                env=env,
                cwd=str(summa_dir)
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
                return False
        
        logger.info(f"SUMMA execution completed successfully. Output files: {len(timestep_files)} timestep files")
        debug_info['summa_output_files'] = [str(f) for f in timestep_files[:3]]  # First 3 files
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"SUMMA simulation failed with exit code {e.returncode}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False
        
    except subprocess.TimeoutExpired:
        error_msg = "SUMMA simulation timed out (120 minutes)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Error running SUMMA: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_mizuroute_worker(task_data: Dict, mizuroute_dir: Path, logger, debug_info: Dict) -> bool:
    """Updated mizuRoute worker with fixed time precision handling"""
    try:
        # Verify SUMMA output exists first
        summa_dir = Path(task_data['summa_dir'])
        expected_files = list(summa_dir.glob("*timestep.nc"))
        
        if not expected_files:
            error_msg = f"No SUMMA timestep files found for mizuRoute input: {summa_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False

        # Fix SUMMA time precision with better error handling
        try:
            logger.info("Fixing SUMMA time precision for mizuRoute compatibility")
            fix_summa_time_precision(expected_files[0])
            logger.info("SUMMA time precision fixed successfully")
        except Exception as e:
            error_msg = f"Failed to fix SUMMA time precision: {str(e)}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
 
        logger.info(f"Found {len(expected_files)} SUMMA output files for mizuRoute")
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


def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
    
    if domain_method not in ['point', 'lumped']:
        return True
    
    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True
    
    return False


def _run_dds_instance_worker(worker_data: Dict) -> Dict:
    """
    Worker function that runs a complete DDS instance
    This runs in a separate process
    """
    import numpy as np
    import logging
    import os
    from pathlib import Path
    
    try:
        dds_task = worker_data['dds_task']
        start_id = dds_task['start_id']
        max_iterations = dds_task['max_iterations']
        dds_r = dds_task['dds_r']
        starting_solution = dds_task['starting_solution']
        
        # Set up process-specific random seed
        np.random.seed(dds_task['random_seed'])
        
        # Set up logger
        logger = logging.getLogger(f'dds_worker_{start_id}')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[DDS-{start_id:02d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting DDS instance {start_id} with {max_iterations} iterations")
        
        # Initialize DDS state
        current_solution = starting_solution.copy()
        param_count = len(current_solution)
        
        # Evaluate initial solution
        current_score = _evaluate_single_solution_worker(current_solution, worker_data, logger)
        
        best_solution = current_solution.copy()
        best_score = current_score
        best_params = _denormalize_params_worker(best_solution, worker_data)
        
        history = []
        total_evaluations = 1
        
        # Record initial state
        history.append({
            'generation': 0,
            'best_score': best_score,
            'current_score': current_score,
            'best_params': best_params
        })
        
        # DDS main loop
        for iteration in range(1, max_iterations + 1):
            # Calculate selection probability
            prob_select = 1.0 - np.log(iteration) / np.log(max_iterations)
            prob_select = max(prob_select, 1.0 / param_count)
            
            # Create trial solution
            trial_solution = current_solution.copy()
            
            # Select variables to perturb
            variables_to_perturb = np.random.random(param_count) < prob_select
            if not np.any(variables_to_perturb):
                random_idx = np.random.randint(0, param_count)
                variables_to_perturb[random_idx] = True
            
            # Apply perturbations
            for i in range(param_count):
                if variables_to_perturb[i]:
                    perturbation = np.random.normal(0, dds_r)
                    trial_solution[i] = current_solution[i] + perturbation
                    
                    # Reflect at bounds
                    if trial_solution[i] < 0:
                        trial_solution[i] = -trial_solution[i]
                    elif trial_solution[i] > 1:
                        trial_solution[i] = 2.0 - trial_solution[i]
                    
                    trial_solution[i] = np.clip(trial_solution[i], 0, 1)
            
            # Evaluate trial solution
            trial_score = _evaluate_single_solution_worker(trial_solution, worker_data, logger)
            total_evaluations += 1
            
            # Selection (greedy)
            improvement = False
            if trial_score > current_score:
                current_solution = trial_solution.copy()
                current_score = trial_score
                improvement = True
                
                if trial_score > best_score:
                    best_solution = trial_solution.copy()
                    best_score = trial_score
                    best_params = _denormalize_params_worker(best_solution, worker_data)
                    logger.info(f"Iter {iteration}: NEW BEST! Score={best_score:.6f}")
            
            # Record iteration
            history.append({
                'generation': iteration,
                'best_score': best_score,
                'current_score': current_score,
                'trial_score': trial_score,
                'improvement': improvement,
                'num_variables_perturbed': np.sum(variables_to_perturb),
                'best_params': best_params
            })
        
        logger.info(f"DDS instance {start_id} completed: Best={best_score:.6f}, Evaluations={total_evaluations}")
        
        return {
            'start_id': start_id,
            'best_score': best_score,
            'best_params': best_params,
            'best_solution': best_solution,
            'history': history,
            'total_evaluations': total_evaluations,
            'final_current_score': current_score
        }
        
    except Exception as e:
        import traceback
        return {
            'start_id': worker_data['dds_task']['start_id'],
            'best_score': None,
            'error': f'DDS worker exception: {str(e)}\n{traceback.format_exc()}'
        }


def _evaluate_single_solution_worker(solution: np.ndarray, worker_data: Dict, logger) -> float:
    """Evaluate a single solution in the worker process"""
    try:
        # Denormalize parameters
        params = _denormalize_params_worker(solution, worker_data)
        
        # Create evaluation task
        task_data = {
            'individual_id': 0,
            'params': params,
            'config': worker_data['config'],
            'target_metric': worker_data['target_metric'],
            'calibration_variable': worker_data['calibration_variable'],
            'domain_name': worker_data['domain_name'],
            'project_dir': worker_data['project_dir'],
            'original_depths': worker_data['original_depths'],
            'summa_exe': worker_data['summa_exe'],
            'file_manager': worker_data['file_manager'],
            'summa_dir': worker_data['summa_dir'],
            'mizuroute_dir': worker_data['mizuroute_dir'],
            'summa_settings_dir': worker_data['summa_settings_dir'],
            'mizuroute_settings_dir': worker_data['mizuroute_settings_dir'],
            'proc_id': 0
        }
        
        # Use existing worker function
        result = _evaluate_parameters_worker_safe(task_data)
        
        score = result.get('score')
        if score is None:
            logger.warning(f"Evaluation failed: {result.get('error', 'Unknown error')}")
            return float('-inf')
        
        return score
        
    except Exception as e:
        logger.error(f"Error evaluating solution: {str(e)}")
        return float('-inf')


def _denormalize_params_worker(normalized_solution: np.ndarray, worker_data: Dict) -> Dict:
    """Denormalize parameters in worker process"""
    param_bounds = worker_data['param_bounds']
    param_names = worker_data['all_param_names']
    
    params = {}
    for i, param_name in enumerate(param_names):
        if param_name in param_bounds:
            bounds = param_bounds[param_name]
            value = bounds['min'] + normalized_solution[i] * (bounds['max'] - bounds['min'])
            params[param_name] = np.array([value])
    
    return params