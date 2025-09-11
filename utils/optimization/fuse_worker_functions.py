#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUSE Worker Functions

Worker functions for FUSE model calibration that can be called from 
the main iterative optimizer or used in parallel processing.
"""

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from pathlib import Path
import subprocess
import logging
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import sys

def _apply_fuse_parameters_worker(config: Dict[str, Any], params: Dict[str, float]) -> bool:
    """
    Worker function to apply FUSE parameters to the parameter NetCDF file.
    
    Args:
        config: Configuration dictionary
        params: Dictionary of parameter names and values
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Construct parameter file path
        domain_name = config.get('DOMAIN_NAME')
        experiment_id = config.get('EXPERIMENT_ID')
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        
        param_file_path = (data_dir / f"domain_{domain_name}" / 'simulations' / 
                          experiment_id / 'FUSE' / f"{domain_name}_{experiment_id}_para_def.nc")
        
        if not param_file_path.exists():
            return False
        
        # Update parameters in NetCDF file
        with nc.Dataset(param_file_path, 'r+') as ds:
            for param_name, value in params.items():
                if param_name in ds.variables:
                    ds.variables[param_name][0] = value
                else:
                    print(f"Warning: Parameter {param_name} not found in NetCDF file")
                    return False
        
        return True
        
    except Exception as e:
        print(f"Error applying FUSE parameters: {str(e)}")
        return False

def _run_fuse_worker(config: Dict[str, Any], mode: str = 'run_def') -> bool:
    """
    Worker function to execute FUSE model.
    
    Args:
        config: Configuration dictionary
        mode: FUSE execution mode ('run_def', 'calib_sce', 'run_best')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get FUSE executable path
        fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_install == 'default':
            data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
            fuse_exe = data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
        else:
            fuse_exe = Path(fuse_install) / 'fuse.exe'
        
        # Get file manager path
        domain_name = config.get('DOMAIN_NAME')
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        filemanager_path = (data_dir / f"domain_{domain_name}" / 'settings' / 
                           'FUSE' / 'fm_catch.txt')
        
        if not fuse_exe.exists():
            print(f"FUSE executable not found: {fuse_exe}")
            return False
            
        if not filemanager_path.exists():
            print(f"FUSE file manager not found: {filemanager_path}")
            return False
        
        # Execute FUSE
        cmd = [str(fuse_exe), str(filemanager_path), mode]
        
        # Change to FUSE settings directory for execution
        settings_dir = filemanager_path.parent
        
        result = subprocess.run(
            cmd,
            cwd=settings_dir,
            capture_output=True,
            text=True,
            timeout=config.get('FUSE_TIMEOUT', 300)  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"FUSE execution failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FUSE execution timed out")
        return False
    except Exception as e:
        print(f"Error running FUSE: {str(e)}")
        return False

def _calculate_fuse_metrics_worker(config: Dict[str, Any], metric: str = 'KGE') -> float:
    """
    Worker function to calculate performance metrics for FUSE output.
    
    Args:
        config: Configuration dictionary
        metric: Metric to calculate ('KGE', 'NSE', 'RMSE', 'MAE')
        
    Returns:
        float: Calculated metric value, or -999 if error
    """
    try:
        # Import evaluation functions
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from utils.evaluation.calculate_sim_stats import get_KGE, get_NSE, get_RMSE, get_MAE
        
        # Get paths
        domain_name = config.get('DOMAIN_NAME')
        experiment_id = config.get('EXPERIMENT_ID')
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{domain_name}"
        
        # Read observed streamflow
        obs_file_path = config.get('OBSERVATIONS_PATH')
        if obs_file_path == 'default':
            obs_file_path = (project_dir / 'observations' / 'streamflow' / 'preprocessed' / 
                           f"{domain_name}_streamflow_processed.csv")
        else:
            obs_file_path = Path(obs_file_path)
        
        if not obs_file_path.exists():
            print(f"Observation file not found: {obs_file_path}")
            return -999.0
        
        # Read observations
        dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        observed_streamflow = dfObs['discharge_cms'].resample('d').mean()
        
        # Read FUSE simulation output
        sim_file_path = (project_dir / 'simulations' / experiment_id / 'FUSE' / 
                        f"{domain_name}_{experiment_id}_runs_def.nc")
        
        if not sim_file_path.exists():
            print(f"Simulation file not found: {sim_file_path}")
            return -999.0
        
        # Read simulations
        with xr.open_dataset(sim_file_path) as ds:
            # Get routed runoff (assuming this is the main output)
            if 'q_routed' in ds.variables:
                simulated = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0)
            elif 'q_instnt' in ds.variables:
                simulated = ds['q_instnt'].isel(param_set=0, latitude=0, longitude=0)
            else:
                print("No runoff variable found in FUSE output")
                return -999.0
            
            simulated_streamflow = simulated.to_pandas()
        
        # Get catchment area for unit conversion
        area_km2 = _get_catchment_area_for_fuse(config, project_dir)
        
        # Convert FUSE output from mm/day to cms
        # Q(cms) = Q(mm/day) * Area(km2) / 86.4
        simulated_streamflow = simulated_streamflow * area_km2 / 86.4
        
        # Align time series
        common_index = observed_streamflow.index.intersection(simulated_streamflow.index)
        if len(common_index) == 0:
            print("No overlapping time period between observations and simulations")
            return -999.0
        
        obs_aligned = observed_streamflow.loc[common_index].dropna()
        sim_aligned = simulated_streamflow.loc[common_index].dropna()
        
        # Further align after dropping NaNs
        common_index = obs_aligned.index.intersection(sim_aligned.index)
        obs_values = obs_aligned.loc[common_index].values
        sim_values = sim_aligned.loc[common_index].values
        
        if len(obs_values) == 0:
            print("No valid data points for metric calculation")
            return -999.0
        
        # Calculate metric
        if metric.upper() == 'KGE':
            return get_KGE(obs_values, sim_values, transfo=1)
        elif metric.upper() == 'NSE':
            return get_NSE(obs_values, sim_values, transfo=1)
        elif metric.upper() == 'RMSE':
            return get_RMSE(obs_values, sim_values, transfo=1)
        elif metric.upper() == 'MAE':
            return get_MAE(obs_values, sim_values, transfo=1)
        else:
            print(f"Unknown metric: {metric}")
            return -999.0
            
    except Exception as e:
        print(f"Error calculating FUSE metrics: {str(e)}")
        return -999.0

def _get_catchment_area_for_fuse(config: Dict[str, Any], project_dir: Path) -> float:
    """
    Helper function to get catchment area for FUSE unit conversion.
    
    Args:
        config: Configuration dictionary
        project_dir: Project directory path
        
    Returns:
        float: Catchment area in km2
    """
    try:
        import geopandas as gpd
        
        # Get catchment shapefile path
        catchment_path = config.get('CATCHMENT_PATH')
        catchment_name = config.get('CATCHMENT_SHP_NAME')
        
        if catchment_path == 'default':
            catchment_path = project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)
        
        if catchment_name == 'default':
            domain_name = config.get('DOMAIN_NAME')
            catchment_name = f"{domain_name}_HRUs_{config['DOMAIN_DISCRETIZATION']}.shp"
        
        catchment_file = catchment_path / catchment_name
        
        if not catchment_file.exists():
            print(f"Catchment file not found: {catchment_file}")
            return 1000.0  # Default value
        
        # Read catchment and calculate area
        gdf = gpd.read_file(catchment_file)
        
        # Check if GRU_area column exists
        if 'GRU_area' in gdf.columns:
            total_area_m2 = gdf['GRU_area'].sum()
            return total_area_m2 / 1e6  # Convert to km2
        else:
            # Calculate area from geometry
            if gdf.crs and not gdf.crs.is_geographic:
                total_area_m2 = gdf.geometry.area.sum()
            else:
                # Project to appropriate UTM zone for area calculation
                gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                total_area_m2 = gdf_utm.geometry.area.sum()
            
            return total_area_m2 / 1e6  # Convert to km2
            
    except Exception as e:
        print(f"Error calculating catchment area: {str(e)}")
        return 1000.0  # Default fallback value

def _evaluate_fuse_parameters_worker(config: Dict[str, Any], 
                                   normalized_params: np.ndarray,
                                   param_names: List[str],
                                   param_bounds: Dict[str, Dict[str, float]]) -> float:
    """
    Complete worker function to evaluate a FUSE parameter set.
    
    Args:
        config: Configuration dictionary
        normalized_params: Normalized parameter values [0,1]
        param_names: List of parameter names
        param_bounds: Parameter bounds dictionary
        
    Returns:
        float: Fitness value (metric score)
    """
    try:
        # Denormalize parameters
        params = {}
        for i, param_name in enumerate(param_names):
            bounds = param_bounds[param_name]
            value = (normalized_params[i] * (bounds['max'] - bounds['min']) + 
                    bounds['min'])
            params[param_name] = float(value)
        
        # Apply parameters to FUSE
        if not _apply_fuse_parameters_worker(config, params):
            return -999.0
        
        # Run FUSE model
        if not _run_fuse_worker(config, 'run_def'):
            return -999.0
        
        # Calculate metrics
        metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        fitness = _calculate_fuse_metrics_worker(config, metric)
        
        return fitness
        
    except Exception as e:
        print(f"Error in FUSE parameter evaluation: {str(e)}")
        return -999.0