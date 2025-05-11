#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import shutil
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple, Union
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer

class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) Optimizer for CONFLUENCE.
    
    This class performs parameter optimization using the PSO algorithm,
    which is a population-based stochastic optimization technique inspired by 
    social behavior of bird flocking or fish schooling.
    
    The implementation uses the pyswarms library, which provides different PSO variants.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the PSO Optimizer.
        
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
        
        # Create output directory
        self.output_dir = self.project_dir / "optimisation" / f"pso_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimization settings
        self.max_iterations = self.config.get('NUMBER_OF_ITERATIONS', 100)
        
        # PSO specific parameters
        self.swarm_size = self.config.get('SWRMSIZE', 40)  # Number of particles
        self.cognitive_param = self.config.get('PSO_COGNITIVE_PARAM', 1.5)  # c1 parameter (personal best influence)
        self.social_param = self.config.get('PSO_SOCIAL_PARAM', 1.5)  # c2 parameter (global best influence)
        self.inertia_weight = self.config.get('PSO_INERTIA_WEIGHT', 0.7)  # w parameter (inertia weight)
        self.inertia_reduction = self.config.get('PSO_INERTIA_REDUCTION_RATE', 0.99)  # Rate at which inertia reduces over iterations
        
        # Define parameter bounds
        self.local_param_info_path = self.project_dir / 'settings' / 'SUMMA' / 'localParamInfo.txt'
        self.basin_param_info_path = self.project_dir / 'settings' / 'SUMMA' / 'basinParamInfo.txt'
        
        # Get parameters to calibrate
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []
        
        # Get performance metric settings
        self.target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Time period information for evaluation
        calib_period = self.config.get('CALIBRATION_PERIOD', '')
        eval_period = self.config.get('EVALUATION_PERIOD', '')
        self.calibration_period = self._parse_date_range(calib_period)
        self.evaluation_period = self._parse_date_range(eval_period)
        
        # Get attribute file path
        self.attr_file_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
        
        # Tracking optimization progress
        self.best_params = None
        self.best_score = float('-inf')  # We'll maximize the objective
        self.iteration_history = []
        
        # Logging
        self.logger.info(f"PSO Optimizer initialized with {len(self.local_params)} local parameters and {len(self.basin_params)} basin parameters")
        self.logger.info(f"Maximum iterations: {self.max_iterations}, swarm size: {self.swarm_size}")
    
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
                dims = {dim: size for dim, size in ds.dims.items()}
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
                        # This is a soil parameter that varies with depth
                        self.logger.info(f"Parameter {param} varies with soil depth ({dims.get('depth', 0)} layers)")
                        
                        if 'time' in var.dims:
                            values = var.isel(time=0).values
                        else:
                            values = var.values
                        
                        # For hydraulic conductivity parameters, consider using harmonic mean
                        if param in ['k_soil', 'k_macropore']:
                            # For hydraulic conductivity, harmonic mean is more appropriate
                            # but we need to handle possible zeros or negative values
                            if np.all(values > 0):
                                from scipy import stats
                                mean_value = stats.hmean(values)
                                self.logger.info(f"Using harmonic mean for {param}: {mean_value}")
                                param_values[param] = np.array([mean_value])
                            else:
                                # Fall back to arithmetic mean if there are non-positive values
                                mean_value = np.mean(values)
                                self.logger.info(f"Using arithmetic mean for {param} (has non-positive values): {mean_value}")
                                param_values[param] = np.array([mean_value])
                        else:
                            # For other soil parameters, use arithmetic mean
                            mean_value = np.mean(values)
                            self.logger.info(f"Using arithmetic mean for soil parameter {param}: {mean_value}")
                            param_values[param] = np.array([mean_value])
                    
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
    
    def _generate_trial_params_file(self, params):
        """
        Generate a trialParams.nc file with the given parameters.
        
        Args:
            params: Dictionary with parameter values
            
        Returns:
            Path: Path to the generated trial parameters file
        """
        self.logger.debug("Generating trial parameters file")
        
        # Get attribute file path for reading HRU information
        if not self.attr_file_path.exists():
            self.logger.error(f"Attribute file not found: {self.attr_file_path}")
            return None
        
        try:
            # Read HRU and GRU information from the attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                # Check for GRU dimension
                has_gru_dim = 'gru' in ds.dims
                hru_ids = ds['hruId'].values
                
                if has_gru_dim:
                    gru_ids = ds['gruId'].values
                    self.logger.info(f"Attribute file has {len(gru_ids)} GRUs and {len(hru_ids)} HRUs")
                else:
                    gru_ids = np.array([1])  # Default if no GRU dimension
                    self.logger.info(f"Attribute file has no GRU dimension, using default GRU ID=1")
                
                # Create the trial parameters dataset
                trial_params_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                
                # Routing parameters should be at GRU level
                routing_params = ['routingGammaShape', 'routingGammaScale']
                basin_params = ['basin__aquiferHydCond', 'basin__aquiferScaleFactor', 'basin__aquiferBaseflowExp']
                gru_level_params = routing_params + basin_params
                
                with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                    # Add dimensions
                    output_ds.createDimension('hru', len(hru_ids))
                    
                    # Add GRU dimension if needed for routing parameters
                    if any(param in params for param in gru_level_params):
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
                        if param_name in gru_level_params:
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
                            #self.logger.info(f"Added {param_name} at GRU level with values: {param_values_array}")
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
                    output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE PSO Optimizer"
                    output_ds.history = f"Created on {datetime.now().isoformat()}"
                    output_ds.confluence_experiment_id = self.experiment_id
                
                self.logger.info(f"Trial parameters file generated: {trial_params_path}")
                return trial_params_path
                
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None 
            
    def _denormalize_individual(self, normalized_individual):
        """
        Denormalize an individual's parameters from [0,1] range to original parameter ranges.
        
        Args:
            normalized_individual: Array of normalized parameter values for one individual
            
        Returns:
            Dict: Parameter dictionary with denormalized values
        """
        params = {}
        
        for i, param_name in enumerate(self.param_names):
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                min_val = bounds['min']
                max_val = bounds['max']
                
                # Denormalize value
                denorm_value = min_val + normalized_individual[i] * (max_val - min_val)
                
                # Special handling for GRU and HRU level parameters
                if param_name in self.basin_params:
                    # Basin/GRU level parameter - single value
                    params[param_name] = np.array([denorm_value])
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
        
        Args:
            params: Dictionary with parameter values
            
        Returns:
            float: Performance metric value, or None if evaluation failed
        """
        self.logger.debug("Evaluating parameter set")
        
        # Generate trial parameters file
        trial_params_path = self._generate_trial_params_file(params)
        if not trial_params_path:
            self.logger.warning("Could not generate trial parameters file")
            return None
        
        # Run SUMMA
        summa_success = self._run_summa_simulation()
        if not summa_success:
            self.logger.warning("SUMMA simulation failed")
            return None
        
        # Run mizuRoute if needed
        if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
            mizuroute_success = self._run_mizuroute_simulation()
            if not mizuroute_success:
                self.logger.warning("mizuRoute simulation failed")
                return None
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        if not metrics:
            self.logger.warning("Could not calculate performance metrics")
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
            self.logger.warning(f"Target metric {self.target_metric} not found, using {next(iter(metrics.keys()))} instead")
        
        # Negate the score for metrics where lower is better
        if self.target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
            score = -score
        
        return score
        
    def _run_summa_simulation(self):
        """
        Run SUMMA with the current trial parameters.
        
        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.debug("Running SUMMA simulation")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        file_manager = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not file_manager.exists():
            self.logger.error(f"File manager not found: {file_manager}")
            return False
        
        # Create simulation directory
        sim_dir = self.project_dir / "simulations" / self.experiment_id / "SUMMA"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = sim_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create command
        summa_command = f"{summa_exe} -m {file_manager}"
        
        # Run SUMMA
        log_file = log_dir / f"summa_pso_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        Run mizuRoute with the current SUMMA outputs.
        
        Returns:
            bool: True if the run was successful, False otherwise
        """
        self.logger.debug("Running mizuRoute simulation")
        
        # Get mizuRoute executable path
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = self.project_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not control_file.exists():
            self.logger.error(f"Control file not found: {control_file}")
            return False
        
        # Create simulation directory
        sim_dir = self.project_dir / "simulations" / self.experiment_id / "mizuRoute"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = sim_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create command
        mizu_command = f"{mizu_exe} {control_file}"
        
        # Run mizuRoute
        log_file = log_dir / f"mizuroute_pso_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        
    def _calculate_performance_metrics(self):
        """
        Calculate performance metrics by comparing simulated to observed streamflow.
        
        Returns:
            Dict: Dictionary with performance metrics
        """
        self.logger.debug("Calculating performance metrics")
        
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
                # From mizuRoute output
                sim_dir = self.project_dir / "simulations" / self.experiment_id / "mizuRoute"
                sim_files = list(sim_dir.glob("*.nc"))
                
                if not sim_files:
                    self.logger.error("No mizuRoute output files found")
                    return None
                
                sim_file = sim_files[0]
                
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
                # From SUMMA output
                sim_dir = self.project_dir / "simulations" / self.experiment_id / "SUMMA"
                sim_files = list(sim_dir.glob(f"{self.experiment_id}*.nc"))
                
                if not sim_files:
                    self.logger.error("No SUMMA output files found")
                    return None
                
                sim_file = sim_files[0]
                
                # Open the file with xarray
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
                            break
                    else:
                        self.logger.error("Could not find streamflow variable in SUMMA output")
                        return None
                    
                    # Get catchment area
                    catchment_area = self._get_catchment_area()
                    if catchment_area is not None and catchment_area > 0:
                        # Convert from m/s to m³/s
                        simulated_flow = simulated_flow * catchment_area
            
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
                    
                    # Calculate metrics
                    metrics = self._calculate_streamflow_metrics(obs_common, sim_common)
            
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
                    self.logger.info(f"Found catchment area from attribute: {total_area} m²")
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

    def run_pso_optimization(self):
        """
        Run the PSO optimization algorithm.
        
        Retur:
            Dict: Dictionary with optimization results
        """
        self.logger.info("Starting PSO optimization")
        
        # Step 1: Get initial parameter values from a preliminary SUMMA run
        initial_params = self.run_parameter_extraction()
        
        # Step 2: Parse parameter bounds
        param_bounds = self._parse_parameter_bounds()
        
        # Step 3: Set up PSO optimizer
        best_params, best_score, history = self._run_pso_algorithm(initial_params, param_bounds)
        
        # Step 4: Create visualization of optimization progress
        self._create_optimization_plots(history)
        
        # Step 5: Run a final simulation with the best parameters
        final_result = self._run_final_simulation(best_params)
        
        # Return results
        results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'history': history,
            'final_result': final_result,
            'output_dir': str(self.output_dir)
        }
        
        return results

    def _run_pso_algorithm(self, initial_params, param_bounds):
        """
        Run the PSO algorithm using pyswarms.
        
        Args:
            initial_params: Dictionary with initial parameter values (optional)
            param_bounds: Dictionary with parameter bounds
            
        Returns:
            Tuple: (best_params, best_score, history)
        """
        self.logger.info("Setting up PSO algorithm")
        
        # Store parameter names in order
        self.param_names = list(param_bounds.keys())
        self.param_bounds = param_bounds
        
        # Extract lower and upper bounds as arrays for pyswarms
        lb = np.array([param_bounds[param]['min'] for param in self.param_names])
        ub = np.array([param_bounds[param]['max'] for param in self.param_names])
        
        # Normalize bounds for PSO (to [0, 1] range)
        bounds = (np.zeros(len(self.param_names)), np.ones(len(self.param_names)))
        
        # Create optimizer options
        options = {
            'c1': self.cognitive_param,  # cognitive parameter
            'c2': self.social_param,     # social parameter
            'w': self.inertia_weight,    # inertia parameter
            'k': 3,                      # number of neighbors to consider for local topology
            'p': 2                       # Minkowski p-norm parameter for distance computation
        }
        
        # Initialize global best PSO optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.swarm_size,
            dimensions=len(self.param_names),
            options=options,
            bounds=bounds,
            init_pos=None,  # Random initialization
            ftol=-float('inf'),  # No function tolerance (run for max iterations)
            ftol_iter=1
        )
        
        # Set up a custom callback wrapper and iteration counter
        iteration_history = []
        current_iteration = [0]  # Use list to make it mutable in inner scope
        
        # Create wrapped objective function that does the tracking internally
        def wrapped_objective_function(x):
            n_particles = x.shape[0]
            scores = np.zeros(n_particles)
            
            for i in range(n_particles):
                # Denormalize parameters
                params = self._denormalize_individual(x[i])
                
                # Evaluate parameters
                score = self._evaluate_parameters(params)
                
                # Store score (negative for minimization, PSO minimizes by default)
                scores[i] = -score if score is not None else float('inf')
                
                # Update best if better
                if score is not None and score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
            
            # After evaluating all particles in this iteration, record the history
            if hasattr(optimizer, 'swarm'):
                # Create history entry
                history_entry = {
                    'iteration': current_iteration[0],
                    'best_score': -optimizer.swarm.best_cost if hasattr(optimizer.swarm, 'best_cost') else self.best_score,
                    'best_params': self.best_params,
                    'swarm_positions': x.copy(),
                    'swarm_scores': -scores.copy(),  # Negate back to original score
                    'mean_score': -np.nanmean(scores),
                    'std_score': np.nanstd(scores)
                }
                
                # Add to history
                iteration_history.append(history_entry)
                
                # Log progress
                self.logger.info(f"Iteration {current_iteration[0]}: Best={history_entry['best_score']:.4f}, "
                                f"Mean={history_entry['mean_score']:.4f}, Std={history_entry['std_score']:.4f}")
                
                # Check if we need to update inertia weight
                if current_iteration[0] > 0:
                    optimizer.options['w'] *= self.inertia_reduction
                    self.logger.debug(f"Updated inertia weight to {optimizer.options['w']:.4f}")
                
                # Increment iteration counter for next call
                current_iteration[0] += 1
            
            return scores
        
        # Run optimization without passing callback
        self.logger.info(f"Running PSO optimization with {self.swarm_size} particles for {self.max_iterations} iterations")
        best_score, best_pos = optimizer.optimize(
            wrapped_objective_function,
            iters=self.max_iterations,
            verbose=True
        )
        
        # Denormalize best position
        best_params = self._denormalize_individual(best_pos)
        
        # Negate best score (PSO minimizes, we maximize)
        best_score = -best_score
        
        self.logger.info(f"PSO optimization completed with best score: {best_score:.4f}")
        
        return best_params, best_score, iteration_history
    
    def _run_final_simulation(self, best_params):
        """
        Run a final simulation with the best parameters.
        
        Args:
            best_params: Dictionary with best parameter values
            
        Returns:
            Dict: Dictionary with final simulation results
        """
        self.logger.info("Running final simulation with best parameters")
        
        # Create a copy of the best parameters file
        final_params_path = self.output_dir / "best_params.nc"
        trial_params_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
        
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
        
        # Run SUMMA with the best parameters
        summa_success = self._run_summa_simulation()
        
        # Run mizuRoute if needed
        mizuroute_success = True
        if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
            mizuroute_success = self._run_mizuroute_simulation()
        
        # Calculate final performance
        final_metrics = None
        if summa_success and mizuroute_success:
            final_metrics = self._calculate_performance_metrics()
            
            if final_metrics:
                # Save metrics to CSV
                metrics_df = pd.DataFrame([final_metrics])
                metrics_csv_path = self.output_dir / "final_metrics.csv"
                metrics_df.to_csv(metrics_csv_path, index=False)
                self.logger.info(f"Saved final metrics to CSV: {metrics_csv_path}")
        
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
            # Create output directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract data from history
            iterations = [h['iteration'] for h in history]
            best_scores = [h['best_score'] for h in history]
            mean_scores = [h['mean_score'] for h in history]
            std_scores = [h['std_score'] for h in history]
            
            # Plot optimization progress
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, best_scores, 'b-', label='Best Score')
            plt.plot(iterations, mean_scores, 'g-', label='Mean Score')
            plt.fill_between(iterations, 
                            [m - s for m, s in zip(mean_scores, std_scores)],
                            [m + s for m, s in zip(mean_scores, std_scores)],
                            alpha=0.2, color='green')
            
            plt.xlabel('Iteration')
            plt.ylabel(f'Performance Metric ({self.target_metric})')
            plt.title('PSO Optimization Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot parameter evolution
            # First, get best parameter values at each iteration
            param_values = {}
            for param_name in self.param_names:
                param_values[param_name] = []
            
            for h in history:
                if h['best_params'] is not None:
                    for param_name in self.param_names:
                        if param_name in h['best_params']:
                            values = h['best_params'][param_name]
                            # Use mean if there are multiple values
                            if isinstance(values, np.ndarray) and len(values) > 1:
                                param_values[param_name].append(np.mean(values))
                            else:
                                value = values[0] if isinstance(values, np.ndarray) else values
                                param_values[param_name].append(value)
                        else:
                            param_values[param_name].append(np.nan)
            
            # Create subplots for each parameter
            n_params = len(self.param_names)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 4 * n_rows))
            
            for i, param_name in enumerate(self.param_names):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.plot(iterations, param_values[param_name], 'b-o')
                
                # Add bounds
                if param_name in self.param_bounds:
                    bounds = self.param_bounds[param_name]
                    plt.axhline(bounds['min'], color='r', linestyle='--', alpha=0.5, label='Min bound')
                    plt.axhline(bounds['max'], color='g', linestyle='--', alpha=0.5, label='Max bound')
                
                plt.xlabel('Iteration')
                plt.ylabel(f'{param_name} Value')
                plt.title(f'{param_name} Evolution')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "parameter_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot streamflow comparison for best parameters
            self._plot_streamflow_comparison()
            
            # Try to create swarm visualization if feasible
            if len(self.param_names) == 2:
                # Create meshgrid for plotting contours
                from scipy.interpolate import griddata
                
                # Get final swarm positions and scores
                final_positions = history[-1]['swarm_positions']
                final_scores = history[-1]['swarm_scores']
                
                # Create grid for contour plot
                grid_size = 50
                x = np.linspace(0, 1, grid_size)
                y = np.linspace(0, 1, grid_size)
                X, Y = np.meshgrid(x, y)
                
                # Interpolate scores onto grid
                points = final_positions
                values = final_scores
                Z = griddata(points, values, (X, Y), method='cubic', fill_value=np.nanmin(values))
                
                # Plot contour
                plt.figure(figsize=(10, 8))
                plt.contourf(X, Y, Z, 15, cmap='viridis')
                plt.colorbar(label=f'Performance Metric ({self.target_metric})')
                
                # Mark best position
                best_pos = history[-1]['swarm_positions'][np.argmax(history[-1]['swarm_scores'])]
                plt.scatter(best_pos[0], best_pos[1], color='r', marker='*', s=200, label='Best Position')
                
                # Plot swarm
                plt.scatter(final_positions[:, 0], final_positions[:, 1], color='white', alpha=0.7, label='Particles')
                
                # Labels
                plt.xlabel(f'{self.param_names[0]} (normalized)')
                plt.ylabel(f'{self.param_names[1]} (normalized)')
                plt.title('PSO Search Space (Final Iteration)')
                plt.legend()
                
                # Save plot
                plt.tight_layout()
                plt.savefig(plots_dir / "pso_search_space.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot particle score distribution at different iterations
            # Select a few iterations to plot (first, middle, last)
            selected_iterations = [0]  # Always include initial swarm
            if len(iterations) > 1:
                middle_iteration = iterations[len(iterations) // 2]
                selected_iterations.append(middle_iteration)
                selected_iterations.append(iterations[-1])  # Add final iteration
            
            # Create score distribution plots
            plt.figure(figsize=(15, 5))
            for i, iteration in enumerate(selected_iterations):
                plt.subplot(1, len(selected_iterations), i + 1)
                h = history[iteration]
                plt.hist(h['swarm_scores'], bins=15, alpha=0.7)
                plt.axvline(h['best_score'], color='r', linestyle='-', label=f'Best: {h["best_score"]:.4f}')
                plt.axvline(h['mean_score'], color='g', linestyle='--', label=f'Mean: {h["mean_score"]:.4f}')
                plt.xlabel(f'Performance Metric ({self.target_metric})')
                plt.ylabel('Frequency')
                plt.title(f'Score Distribution at Iteration {iteration}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "swarm_score_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Optimization plots created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _plot_streamflow_comparison(self):
        """
        Create a plot comparing observed and simulated streamflow for the best parameter set.
        """
        self.logger.info("Creating streamflow comparison plot")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Create output directory
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
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
                            if 'gru' in ds[var_name].dims and ds.dims['gru'] > 1:
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
            
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Plot observed and simulated flow
            plt.plot(observed_flow.index, observed_flow, 'b-', linewidth=1.5, label='Observed')
            plt.plot(simulated_flow.index, simulated_flow, 'r-', linewidth=1.5, label='Simulated (Best Parameters)')
            
            # Format plot
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow (Best Parameters)')
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
                        metrics_text += f"  {metric_name}: {value:.4f}\n"
                if eval_metrics:
                    metrics_text += "Evaluation Period:\n"
                    for metric, value in eval_metrics.items():
                        metric_name = metric.replace('Eval_', '')
                        metrics_text += f"  {metric_name}: {value:.4f}\n"
                if not calib_metrics and not eval_metrics:
                    # Use main metrics
                    for metric in ['KGE', 'NSE', 'RMSE', 'PBIAS']:
                        if metric in final_metrics:
                            metrics_text += f"  {metric}: {final_metrics[metric]:.4f}\n"
                
                # Add text to plot
                plt.text(0.01, 0.98, metrics_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save plot
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create a scatter plot
            plt.figure(figsize=(10, 8))
            
            # Find common timepoints
            common_idx = observed_flow.index.intersection(simulated_flow.index)
            obs_common = observed_flow.loc[common_idx]
            sim_common = simulated_flow.loc[common_idx]
            
            # Create scatter plot
            plt.scatter(obs_common, sim_common, alpha=0.5)
            
            # Add 1:1 line
            max_val = max(obs_common.max(), sim_common.max())
            min_val = min(obs_common.min(), sim_common.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
            
            # Add regression line
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(obs_common, sim_common)
            plt.plot([min_val, max_val], [slope * min_val + intercept, slope * max_val + intercept], 'r-',
                    label=f'Regression Line (r={r_value:.3f})')
            
            # Format plot
            plt.xlabel('Observed Streamflow (m³/s)')
            plt.ylabel('Simulated Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow (Scatter Plot)')
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
            
            # Create flow duration curve
            plt.figure(figsize=(10, 8))
            
            # Sort flows for FDC
            obs_sorted = observed_flow.sort_values(ascending=False)
            sim_sorted = simulated_flow.sort_values(ascending=False)
            
            # Calculate exceedance probabilities
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
            
            # Plot FDCs
            plt.plot(obs_exceed, obs_sorted, 'b-', linewidth=1.5, label='Observed')
            plt.plot(sim_exceed, sim_sorted, 'r-', linewidth=1.5, label='Simulated (Best Parameters)')
            
            # Format plot
            plt.xlabel('Exceedance Probability')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Flow Duration Curves')
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