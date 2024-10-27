import os
import csv
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

def read_param_bounds(param_file, params_to_calibrate):
    """Read parameter bounds from the parameter file."""
    bounds = {}
    with open(param_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4 and parts[0].strip() in params_to_calibrate:
                param = parts[0].strip()
                lower = float(parts[2].strip().replace('d', 'e'))
                upper = float(parts[3].strip().replace('d', 'e'))
                bounds[param] = (lower, upper)
    return bounds

def update_param_files(local_param_values, basin_param_values, local_param_keys, basin_param_keys, local_param_file, basin_param_file, local_bounds_dict, basin_bounds_dict):
    def update_file(param_values, param_keys, file_path, bounds_dict):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        param_map = dict(zip(param_keys, param_values))
        
        for i, line in enumerate(lines):
            parts = line.split('|')
            if len(parts) >= 4 and parts[0].strip() in param_map:
                param = parts[0].strip()
                value = param_map[param]
                lower_bound, upper_bound = bounds_dict[param]
                
                # Ensure the value is within bounds
                value = max(lower_bound, min(upper_bound, value))
                
                # Format the value with 'd' for scientific notation
                value_str = f"{value:.6e}".replace('e', 'd')
                # Keep the parameter name unchanged
                lines[i] = f"{parts[0]}| {value_str} |{parts[2]}|{parts[3]}"

        with open(file_path, 'w') as f:
            f.writelines(lines)

    update_file(local_param_values, local_param_keys, local_param_file, local_bounds_dict)
    update_file(basin_param_values, basin_param_keys, basin_param_file, basin_bounds_dict)

def read_summa_error(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                return lines[-3].strip()  # Return the last 3 lines, stripped of leading/trailing whitespace
            else:
                return "SUMMA log file is empty."
    except Exception as e:
        return f"Error reading SUMMA log file: {str(e)}"
    
def write_iteration_results(file_path, iteration, params, metrics, mode='a'):
    calib_metrics = metrics['calib']
    eval_metrics = metrics['eval']
    fieldnames = ['Iteration'] + list(params.keys()) + \
                    [f'Calib_{k}' for k in calib_metrics.keys()] + \
                    [f'Eval_{k}' for k in eval_metrics.keys()]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if mode == 'w':
            writer.writeheader()
        
        row = {'Iteration': iteration, **params, 
                **{f'Calib_{k}': v for k, v in calib_metrics.items()},
                **{f'Eval_{k}': v for k, v in eval_metrics.items()}}
        writer.writerow(row)