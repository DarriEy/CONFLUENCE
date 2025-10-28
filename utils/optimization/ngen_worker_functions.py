#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NGEN Worker Functions for Parallel Calibration

These functions are used by MPI workers to evaluate NGEN parameter sets in parallel.
Similar structure to SUMMA worker functions but adapted for NGEN.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import shutil


def ngen_worker_function(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel NGEN parameter evaluation
    
    This function is called by each MPI worker to:
    1. Update NGEN configuration with new parameters
    2. Run NGEN model
    3. Calculate performance metrics
    4. Return results
    
    Args:
        task: Dictionary containing:
            - individual_id: Unique identifier for this evaluation
            - params: Parameter dictionary to evaluate
            - config: Configuration dictionary
            - optimization_metric: Target metric (e.g., 'KGE')
            
    Returns:
        Dictionary with evaluation results:
            - individual_id: Task identifier
            - params: Parameters that were evaluated
            - score: Optimization metric value
            - metrics: All calculated metrics
            - success: Boolean indicating if evaluation succeeded
    """
    try:
        individual_id = task.get('individual_id', -1)
        params = task.get('params', {})
        config = task.get('config', {})
        target_metric = task.get('optimization_metric', 'KGE')
        
        # Setup paths
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        domain_name = config.get('DOMAIN_NAME')
        experiment_id = config.get('EXPERIMENT_ID')
        
        project_dir = data_dir / f"domain_{domain_name}"
        ngen_settings_dir = project_dir / 'settings' / 'ngen'
        ngen_output_dir = project_dir / 'simulations' / experiment_id / 'ngen'
        
        # Update NGEN configuration files with new parameters
        if not _update_ngen_configs(params, ngen_settings_dir, config):
            return {
                'individual_id': individual_id,
                'params': params,
                'score': np.nan,
                'success': False,
                'error': 'Failed to update configs'
            }
        
        # Run NGEN model
        output_dir = _run_ngen_model(config, ngen_settings_dir, ngen_output_dir)
        if not output_dir:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': np.nan,
                'success': False,
                'error': 'NGEN run failed'
            }
        
        # Calculate metrics
        metrics = _calculate_ngen_metrics(config, ngen_output_dir)
        if not metrics or target_metric not in metrics:
            return {
                'individual_id': individual_id,
                'params': params,
                'score': np.nan,
                'metrics': metrics,
                'success': False,
                'error': f'Metric {target_metric} not found'
            }
        
        score = metrics[target_metric]
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': score,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        return {
            'individual_id': task.get('individual_id', -1),
            'params': task.get('params', {}),
            'score': np.nan,
            'success': False,
            'error': str(e)
        }


def _update_ngen_configs(params: Dict[str, float], ngen_settings_dir: Path, 
                         config: Dict) -> bool:
    """
    Update NGEN BMI configuration files with new parameter values
    
    Args:
        params: Parameter dictionary
        ngen_settings_dir: Path to NGEN settings directory
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_dir = ngen_settings_dir / 'model_configs'
        
        if not config_dir.exists():
            return False
        
        # Determine which modules to update based on parameter names
        modules_to_calibrate = config.get('NGEN_MODULES_TO_CALIBRATE', 'NOAH').split(',')
        modules_to_calibrate = [m.strip() for m in modules_to_calibrate]
        
        for module in modules_to_calibrate:
            # Filter parameters for this module
            module_params = {k: v for k, v in params.items() 
                           if k.startswith(module.lower()) or k in ['rain_snow_thresh', 'ZREF']}
            
            if not module_params:
                continue
            
            # Find and update config files for this module
            pattern = f"*{module.lower()}*.json"
            config_files = list(config_dir.glob(pattern))
            
            for config_file in config_files:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update parameters in the config
                # Handle different possible config structures
                if 'config' in config_data:
                    config_section = config_data['config']
                elif 'parameters' in config_data:
                    config_section = config_data['parameters']
                else:
                    config_section = config_data
                
                # Update each parameter
                for param_name, param_value in module_params.items():
                    # Remove module prefix if present (e.g., noah_refdk -> refdk)
                    param_key = param_name.replace(f'{module.lower()}_', '')
                    config_section[param_key] = float(param_value)
                
                # Write updated config
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error updating NGEN configs: {str(e)}")
        return False


def _run_ngen_model(config: Dict, ngen_settings_dir: Path, 
                    ngen_output_dir: Path) -> Optional[Path]:
    """
    Run NGEN model
    
    Args:
        config: Configuration dictionary
        ngen_settings_dir: Path to NGEN settings
        ngen_output_dir: Path to output directory
        
    Returns:
        Output directory path if successful, None otherwise
    """
    try:
        # Get NGEN executable path
        ngen_install_path = Path(config.get('NGEN_INSTALL_PATH', 'default'))
        if ngen_install_path == Path('default'):
            data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
            ngen_install_path = data_dir / 'installs' / 'ngen'
        
        ngen_exe = ngen_install_path / 'build' / 'ngen'
        
        if not ngen_exe.exists():
            print(f"NGEN executable not found: {ngen_exe}")
            return None
        
        # Get input files
        domain_name = config.get('DOMAIN_NAME')
        catchment_file = ngen_settings_dir / f"{domain_name}_catchments.gpkg"
        nexus_file = ngen_settings_dir / "nexus.geojson"
        realization_file = ngen_settings_dir / "realization_config.json"
        
        if not all([catchment_file.exists(), nexus_file.exists(), realization_file.exists()]):
            print("Required NGEN input files not found")
            return None
        
        # Create output directory
        ngen_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            str(ngen_exe),
            str(catchment_file), "all",
            str(nexus_file), "all",
            str(realization_file)
        ]
        
        # Run NGEN (LD_LIBRARY_PATH should already be set in environment)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ngen_output_dir),
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"NGEN failed: {result.stderr}")
            return None
        
        return ngen_output_dir
        
    except subprocess.TimeoutExpired:
        print("NGEN execution timed out")
        return None
    except Exception as e:
        print(f"Error running NGEN: {str(e)}")
        return None


def _calculate_ngen_metrics(config: Dict, ngen_output_dir: Path) -> Optional[Dict[str, float]]:
    """
    Calculate performance metrics from NGEN output
    
    Args:
        config: Configuration dictionary
        ngen_output_dir: Path to NGEN output directory
        
    Returns:
        Dictionary of metrics or None if calculation fails
    """
    try:
        import pandas as pd
        
        # Find output files
        nexus_outputs = list(ngen_output_dir.glob("nex-*_output.csv"))
        
        if not nexus_outputs:
            print("No NGEN output files found")
            return None
        
        # Read simulated streamflow from nexus output
        nexus_output_file = nexus_outputs[0]  # Use first nexus (typically outlet)
        sim_df = pd.read_csv(nexus_output_file)
        
        # Get observed streamflow
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        domain_name = config.get('DOMAIN_NAME')
        obs_file = (data_dir / f"domain_{domain_name}" / 'observations' / 'streamflow' / 
                   'preprocessed' / f"{domain_name}_streamflow_processed.csv")
        
        if not obs_file.exists():
            print(f"Observed data not found: {obs_file}")
            return None
        
        obs_df = pd.read_csv(obs_file, parse_dates=['datetime'])
        
        # Merge simulated and observed
        sim_df['datetime'] = pd.to_datetime(sim_df['time'])
        merged = pd.merge(
            obs_df[['datetime', 'discharge_cms']],
            sim_df[['datetime', 'flow']],
            on='datetime',
            how='inner'
        )
        
        if len(merged) == 0:
            print("No overlapping data between simulated and observed")
            return None
        
        observed = merged['discharge_cms'].values
        simulated = merged['flow'].values
        
        # Calculate metrics
        metrics = calculate_performance_metrics(observed, simulated)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None


def calculate_performance_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
    """
    Calculate hydrological performance metrics
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        Dictionary with KGE, NSE, RMSE, MAE, PBIAS
    """
    # Remove NaN values
    valid = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[valid]
    simulated = simulated[valid]
    
    if len(observed) == 0:
        return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'PBIAS': np.nan}
    
    # Nash-Sutcliffe Efficiency
    mean_obs = observed.mean()
    nse_num = ((observed - simulated) ** 2).sum()
    nse_den = ((observed - mean_obs) ** 2).sum()
    nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
    
    # Root Mean Square Error
    rmse = np.sqrt(((observed - simulated) ** 2).mean())
    
    # Mean Absolute Error
    mae = np.abs(observed - simulated).mean()
    
    # Percent Bias
    pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
    
    # Kling-Gupta Efficiency
    r = np.corrcoef(observed, simulated)[0, 1] if len(observed) > 1 else np.nan
    std_obs = observed.std()
    std_sim = simulated.std()
    mean_sim = simulated.mean()
    
    alpha = std_sim / std_obs if std_obs != 0 else np.nan
    beta = mean_sim / mean_obs if mean_obs != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
    
    return {
        'KGE': kge,
        'NSE': nse,
        'RMSE': rmse,
        'MAE': mae,
        'PBIAS': pbias,
        'r': r,
        'alpha': alpha,
        'beta': beta,
        'n_obs': len(observed)
    }


def evaluate_ngen_parameters(params: Dict[str, float], config: Dict, 
                             optimization_metric: str = 'KGE') -> float:
    """
    Simplified function to evaluate a single parameter set
    
    Args:
        params: Parameter dictionary
        config: Configuration dictionary
        optimization_metric: Target metric name
        
    Returns:
        Metric value (fitness score)
    """
    task = {
        'individual_id': 0,
        'params': params,
        'config': config,
        'optimization_metric': optimization_metric
    }
    
    result = ngen_worker_function(task)
    return result.get('score', np.nan)
