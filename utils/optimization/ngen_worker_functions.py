#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Worker Functions

Worker functions for ngen model calibration that can be called from 
the main iterative optimizer or used in parallel processing.

Author: CONFLUENCE Development Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple, Optional

# Add CONFLUENCE root directory to path
# File is at: CONFLUENCE/utils/optimization/ngen_worker_functions.py
# We need:  CONFLUENCE/ (so we can import utils.model_utils.ngen_utils)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _apply_ngen_parameters_worker(config: Dict[str, Any], params: Dict[str, float]) -> bool:
    """
    Worker function to apply ngen parameters to configuration files.
    
    Args:
        config: Configuration dictionary
        params: Dictionary of parameter names and values (format: "MODULE.param")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import json
        
        domain_name = config.get('DOMAIN_NAME')
        experiment_id = config.get('EXPERIMENT_ID')
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        
        ngen_setup_dir = data_dir / f"domain_{domain_name}" / 'settings' / 'ngen'
        
        # Group parameters by module
        module_params = {}
        for param_name, value in params.items():
            if '.' in param_name:
                module, param = param_name.split('.', 1)
                if module not in module_params:
                    module_params[module] = {}
                module_params[module][param] = value
        
        # Update each module's config file
        for module, module_param_dict in module_params.items():
            config_file = ngen_setup_dir / module / f"{module.lower()}_config.json"
            
            if not config_file.exists():
                print(f"Warning: Config file not found: {config_file}")
                continue
            
            # Read existing config
            with open(config_file, 'r') as f:
                cfg = json.load(f)
            
            # Update parameters
            for param, value in module_param_dict.items():
                if param in cfg:
                    cfg[param] = value
                else:
                    print(f"Warning: Parameter {param} not found in {module} config")
            
            # Write updated config
            with open(config_file, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error applying ngen parameters: {str(e)}")
        return False


def _run_ngen_worker(config: Dict[str, Any]) -> bool:
    """
    Worker function to execute ngen model.
    Supports both serial and parallel modes.
    
    Args:
        config: Configuration dictionary. In parallel mode, includes:
                - '_proc_ngen_dir': Process-specific ngen directory
                - '_proc_settings_dir': Process-specific settings directory
                - '_proc_id': Process ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    import logging
    import traceback
    
    # Get the main confluence logger if available
    logger = logging.getLogger('confluence')
    
    try:
        # Check if this is a parallel process
        if '_proc_ngen_dir' in config:
            # Parallel mode - use process-specific directories
            proc_id = config.get('_proc_id', 0)
            logger.debug(f"Running ngen in parallel mode (proc {proc_id})")
            
            # Create a modified config for this process
            # The NgenRunner will use these paths if they're set
            parallel_config = config.copy()
            parallel_config['_ngen_output_dir'] = config['_proc_ngen_dir']
            parallel_config['_ngen_settings_dir'] = config['_proc_settings_dir']
        else:
            # Serial mode - use standard configuration
            logger.debug("Running ngen in serial mode")
            parallel_config = config
            proc_id = 0
        
        # Import NgenRunner from ngen_utils
        from utils.models.ngen_utils import NgenRunner
        
        domain_name = parallel_config.get('DOMAIN_NAME')
        experiment_id = parallel_config.get('EXPERIMENT_ID')
        
        # Initialize runner with modified config
        runner = NgenRunner(parallel_config, logger)
        
        # Run ngen
        success = runner.run_model(experiment_id)
        
        return success
        
    except FileNotFoundError as e:
        logger.error(f"Required ngen input file not found (proc {proc_id if '_proc_ngen_dir' in config else 0}): {str(e)}")
        logger.error("Make sure ngen preprocessing has been run to generate required files:")
        logger.error("  - catchment geopackage")
        logger.error("  - nexus geojson")
        logger.error("  - realization config json")
        return False
        
    except Exception as e:
        logger.error(f"Error running ngen (proc {proc_id if '_proc_ngen_dir' in config else 0}): {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False


def _calculate_ngen_metrics_worker(config: Dict[str, Any], metric: str = 'KGE') -> float:
    """
    Worker function to calculate ngen performance metrics.
    
    Args:
        config: Configuration dictionary
        metric: Metric to calculate ('KGE', 'NSE', 'MAE', 'RMSE')
        
    Returns:
        float: Metric value (or -999.0 for error/invalid)
    """
    try:
        from ngen_calibration_targets import NgenStreamflowTarget
        import logging
        
        domain_name = config.get('DOMAIN_NAME')
        experiment_id = config.get('EXPERIMENT_ID')
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{domain_name}"
        
        # Create minimal logger
        logger = logging.getLogger(f'ngen_metrics_{experiment_id}')
        logger.setLevel(logging.WARNING)
        
        # Create calibration target
        target = NgenStreamflowTarget(config, project_dir, logger)
        
        # Calculate metrics
        metrics = target.calculate_metrics(experiment_id)
        
        # Return requested metric
        metric_upper = metric.upper()
        if metric_upper in metrics:
            return float(metrics[metric_upper])
        else:
            print(f"Warning: Metric {metric} not found in results")
            return -999.0
        
    except Exception as e:
        print(f"Error calculating ngen metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return -999.0


def _evaluate_ngen_parameters_worker(
    config: Dict[str, Any],
    normalized_params: np.ndarray,
    param_names: List[str],
    param_bounds: Dict[str, Dict[str, float]]
) -> float:
    """
    Worker function to evaluate a parameter set for ngen.
    
    This function:
    1. Denormalizes parameters
    2. Applies them to ngen config files
    3. Runs ngen model
    4. Calculates performance metrics
    
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
        
        # Apply parameters to ngen
        if not _apply_ngen_parameters_worker(config, params):
            return -999.0
        
        # Run ngen model
        if not _run_ngen_worker(config):
            return -999.0
        
        # Calculate metrics
        metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        fitness = _calculate_ngen_metrics_worker(config, metric)
        
        return fitness
        
    except Exception as e:
        print(f"Error in ngen parameter evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return -999.0


def _extract_ngen_streamflow_worker(config: Dict[str, Any], experiment_id: str) -> Optional[Path]:
    """
    Worker function to extract streamflow from ngen outputs.
    
    Args:
        config: Configuration dictionary
        experiment_id: Experiment identifier
        
    Returns:
        Path to extracted streamflow file, or None if failed
    """
    try:
        from utils.models.ngen_utils import NgenPostprocessor
        import logging
        
        # Create minimal logger
        logger = logging.getLogger(f'ngen_extract_{experiment_id}')
        logger.setLevel(logging.WARNING)
        
        # Initialize postprocessor
        postprocessor = NgenPostprocessor(config, logger)
        
        # Extract streamflow
        output_file = postprocessor.extract_streamflow(experiment_id)
        
        return output_file
        
    except Exception as e:
        print(f"Error extracting ngen streamflow: {str(e)}")
        return None