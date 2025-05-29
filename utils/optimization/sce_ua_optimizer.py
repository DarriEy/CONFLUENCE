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

class SCEUAOptimizer:
    """
    Shuffled Complex Evolution (SCE-UA) Optimizer for CONFLUENCE.
    
    This class performs parameter optimization using the SCE-UA algorithm,
    which is a global optimization method particularly effective for watershed model calibration.
    
    References:
    Duan, Q., Sorooshian, S., & Gupta, V. (1992). Effective and efficient global optimization 
    for conceptual rainfall-runoff models. Water resources research, 28(4), 1015-1031.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the SCE-UA Optimizer.
        
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
        self.output_dir = self.project_dir / "optimisation" / f"sceua_{self.experiment_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimization settings
        self.max_iterations = self.config.get('NUMBER_OF_ITERATIONS', 1000)
        
        # SCE-UA specific parameters
        self.population_size = self.config.get('POPULATION_SIZE', 100)  # p
        self.num_complexes = self.config.get('NUMBER_OF_COMPLEXES', 2)  # q
        self.points_per_complex = None  # m (calculated based on population size and complexes)
        self.points_per_subcomplex = self.config.get('POINTS_PER_SUBCOMPLEX', 5)  # s
        self.num_evolution_steps = self.config.get('NUMBER_OF_EVOLUTION_STEPS', 20)  # alpha
        self.evolution_stagnation = self.config.get('EVOLUTION_STAGNATION', 5)  # beta
        self.pct_change_threshold = self.config.get('PERCENT_CHANGE_THRESHOLD', 0.01)  # 1%
        
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
        
        # Logging
        self.logger.info(f"SCE-UA Optimizer initialized with {len(self.local_params)} local parameters and {len(self.basin_params)} basin parameters")
        self.logger.info(f"Maximum iterations: {self.max_iterations}, population size: {self.population_size}, complexes: {self.num_complexes}")
    
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
                    output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE SCE-UA Optimizer"
                    output_ds.history = f"Created on {datetime.now().isoformat()}"
                    output_ds.confluence_experiment_id = self.experiment_id
                
                #self.logger.info(f"Trial parameters file generated: {trial_params_path}")
                return trial_params_path
                
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def _initialize_optimization(self, initial_params, param_bounds):
        """
        Initialize optimization variables for SCE-UA.
        
        Args:
            initial_params: Dictionary with initial parameter values
            param_bounds: Dictionary with parameter bounds
        """
        self.logger.info("Initializing SCE-UA optimization")
        
        # Store parameter bounds
        self.param_bounds = param_bounds
        
        # Get number of parameters to optimize
        num_params = len(param_bounds)
        
        # Calculate points per complex based on formula from Duan et al. (1992)
        # m = 2*n + 1, where n is the number of parameters
        self.points_per_complex = 2 * num_params + 1
        
        # Adjust population size to be a multiple of number of complexes
        self.population_size = self.num_complexes * self.points_per_complex
        self.logger.info(f"Adjusted population size to {self.population_size} (complexes={self.num_complexes}, points per complex={self.points_per_complex})")
        
        # Initialize population in normalized parameter space [0,1]
        self.normalized_population = np.random.random((self.population_size, num_params))
        
        # Store parameter names in order used for population matrix
        self.param_names = list(param_bounds.keys())
        
        # Initialize with any known good parameter values if available
        if initial_params:
            # Try to use initial values for the first individual
            for i, param_name in enumerate(self.param_names):
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
                    self.normalized_population[0, i] = normalized_value
        
        # Initialize tracking variables
        self.population_scores = np.full(self.population_size, np.nan)
        self.best_params = None
        self.best_score = float('-inf')  # We'll maximize the objective
        self.iteration_history = []
        
        # Initialize the first population
        self.logger.info(f"Initializing population of {self.population_size} individuals with {num_params} parameters")
        
        # Run initial evaluation of the population
        self._evaluate_population()
        
        # Record initial state
        self._record_iteration(0)
        
    def _evaluate_population(self):
        """
        Evaluate all individuals in the population.
        
        This runs the model for each parameter set and calculates its performance.
        """
        self.logger.info(f"Evaluating population of {self.population_size} individuals")
        
        for i in range(self.population_size):
            # Skip already evaluated individuals
            if not np.isnan(self.population_scores[i]):
                continue
                
            # Denormalize parameters for this individual
            params = self._denormalize_individual(self.normalized_population[i, :])
            
            # Evaluate this parameter set
            score = self._evaluate_parameters(params)
            
            # Store score
            self.population_scores[i] = score if score is not None else float('-inf')
            
            # Update best if better
            if self.population_scores[i] > self.best_score:
                self.best_score = self.population_scores[i]
                self.best_params = params.copy()
                self.logger.info(f"New best score: {self.best_score:.4f}")
        
        self.logger.info(f"Population evaluation complete. Best score: {self.best_score:.4f}")
    
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
    
    def _normalize_params_dict(self, params_dict):
        """
        Normalize a parameters dictionary to [0,1] range.
        
        Args:
            params_dict: Dictionary with parameter values
            
        Returns:
            ndarray: Normalized parameter array
        """
        normalized = np.zeros(len(self.param_names))
        
        for i, param_name in enumerate(self.param_names):
            if param_name in params_dict and param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                min_val = bounds['min']
                max_val = bounds['max']
                
                # Get value (use mean if it's an array)
                if isinstance(params_dict[param_name], np.ndarray) and len(params_dict[param_name]) > 1:
                    value = np.mean(params_dict[param_name])
                else:
                    value = params_dict[param_name][0] if isinstance(params_dict[param_name], np.ndarray) else params_dict[param_name]
                
                # Normalize
                normalized[i] = (value - min_val) / (max_val - min_val)
                # Clip to valid range
                normalized[i] = np.clip(normalized[i], 0, 1)
            else:
                # Default to random value if parameter not found
                normalized[i] = np.random.random()
        
        return normalized

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
        log_file = log_dir / f"summa_sceua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
        log_file = log_dir / f"mizuroute_sceua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
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
                    #self.logger.info(f"Found catchment area from attribute: {total_area} m²")
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

    def _record_iteration(self, iteration):
        """
        Record the current state of the optimization in the history.
        
        Args:
            iteration: Current iteration number
        """
        # Sort population by score
        sorted_indices = np.argsort(-self.population_scores)  # Descending order
        
        # Create entry for history
        history_entry = {
            'iteration': iteration,
            'best_score': self.best_score,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'population_scores': self.population_scores.copy(),
            'mean_score': np.nanmean(self.population_scores),
            'min_score': np.nanmin(self.population_scores),
            'max_score': np.nanmax(self.population_scores),
            'std_score': np.nanstd(self.population_scores),
        }
        
        # Add to history
        self.iteration_history.append(history_entry)
        
        # Print progress
        self.logger.info(f"Iteration {iteration}: Best={self.best_score:.4f}, Mean={history_entry['mean_score']:.4f}, Std={history_entry['std_score']:.4f}")
    
    def run_sceua_optimization(self):
        """
        Run the SCE-UA optimization algorithm.
        
        Returns:
            Dict: Dictionary with optimization results
        """
        self.logger.info("Starting SCE-UA optimization")
        
        # Step 1: Get initial parameter values from a preliminary SUMMA run
        initial_params = self.run_parameter_extraction()
        
        # Step 2: Parse parameter bounds
        param_bounds = self._parse_parameter_bounds()
        
        # Step 3: Initialize optimization variables
        self._initialize_optimization(initial_params, param_bounds)
        
        # Step 4: Run the SCE-UA algorithm
        best_params, best_score, history = self._run_sceua_algorithm()
        
        # Step 5: Create visualization of optimization progress
        self._create_optimization_plots(history)
        
        # Step 6: Run a final simulation with the best parameters
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
    
    def _run_sceua_algorithm(self):
        """
        Run the SCE-UA algorithm.
        
        Returns:
            Tuple: (best_params, best_score, history)
        """
        self.logger.info("Running SCE-UA algorithm")
        
        # Get number of parameters
        num_params = len(self.param_names)
        
        # Initialize counters
        num_iterations = 0
        num_evaluations = self.population_size  # Initial population already evaluated
        
        # Initialize convergence variables
        converged = False
        stagnant_count = 0
        prev_best_score = self.best_score
        
        # Main optimization loop
        while (num_iterations < self.max_iterations) and not converged:
            num_iterations += 1
            self.logger.info(f"Starting iteration {num_iterations}/{self.max_iterations}")
            
            # Sort population by score (descending order)
            sorted_indices = np.argsort(-self.population_scores)
            sorted_population = self.normalized_population[sorted_indices]
            sorted_scores = self.population_scores[sorted_indices]
            
            # Partition into complexes
            complexes = []
            complex_scores = []
            
            for k in range(self.num_complexes):
                # Select points for this complex
                complex_indices = np.arange(k, self.population_size, self.num_complexes)
                complex_k = sorted_population[complex_indices]
                complex_k_scores = sorted_scores[complex_indices]
                
                complexes.append(complex_k)
                complex_scores.append(complex_k_scores)
            
            # Evolve each complex
            for k in range(self.num_complexes):
                self.logger.debug(f"Evolving complex {k+1}/{self.num_complexes}")
                
                # Get this complex
                complex_k = complexes[k].copy()
                complex_k_scores = complex_scores[k].copy()
                
                # Perform competitive complex evolution
                complex_k, complex_k_scores, num_eval = self._evolve_complex(
                    complex_k, complex_k_scores, num_iterations
                )
                
                # Update complex
                complexes[k] = complex_k
                complex_scores[k] = complex_k_scores
                
                # Update total evaluations
                num_evaluations += num_eval
            
            # Merge complexes back into population
            new_population = np.zeros_like(self.normalized_population)
            new_scores = np.zeros_like(self.population_scores)
            
            idx = 0
            for k in range(self.num_complexes):
                for i in range(self.points_per_complex):
                    new_population[idx] = complexes[k][i]
                    new_scores[idx] = complex_scores[k][i]
                    idx += 1
            
            # Update population
            self.normalized_population = new_population
            self.population_scores = new_scores
            
            # Update best solution
            best_idx = np.argmax(new_scores)
            current_best_score = new_scores[best_idx]
            
            if current_best_score > self.best_score:
                self.best_score = current_best_score
                self.best_params = self._denormalize_individual(new_population[best_idx])
                self.logger.info(f"New best score at iteration {num_iterations}: {self.best_score:.4f}")
            
            # Record this iteration
            self._record_iteration(num_iterations)
            
            # Check for convergence
            pct_improvement = (self.best_score - prev_best_score) / abs(prev_best_score) if prev_best_score != 0 else float('inf')
            
            if abs(pct_improvement) < self.pct_change_threshold:
                stagnant_count += 1
                self.logger.info(f"Improvement ({pct_improvement:.6f}) below threshold. Stagnant count: {stagnant_count}/{self.evolution_stagnation}")
            else:
                stagnant_count = 0
                self.logger.info(f"Improvement: {pct_improvement:.6f}")
            
            # Update previous best score
            prev_best_score = self.best_score
            
            # Check if we've stagnated enough to converge
            if stagnant_count >= self.evolution_stagnation:
                self.logger.info(f"Converged after {num_iterations} iterations (stagnation threshold reached)")
                converged = True
            
            # Log progress
            if num_iterations % 10 == 0 or num_iterations == self.max_iterations or converged:
                self.logger.info(f"Completed {num_iterations}/{self.max_iterations} iterations. Best score: {self.best_score:.4f}")
        
        self.logger.info(f"SCE-UA optimization completed. Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _evolve_complex(self, complex_points, complex_scores, iteration):
        """
        Evolve a complex using the Competitive Complex Evolution (CCE) algorithm.
        
        Args:
            complex_points: The points in the complex
            complex_scores: The scores of the points
            iteration: Current iteration number
            
        Returns:
            Tuple: (evolved_complex, evolved_scores, num_evaluations)
        """
        # Get number of parameters
        num_params = complex_points.shape[1]
        
        # Number of evaluations performed
        num_evaluations = 0
        
        # Perform evolution steps
        for evolution_step in range(self.num_evolution_steps):
            # Create subcomplex - parents are selected according to a trapezoidal probability distribution
            subcomplex_indices = self._select_subcomplex_indices(self.points_per_complex, self.points_per_subcomplex)
            subcomplex = complex_points[subcomplex_indices]
            subcomplex_scores = complex_scores[subcomplex_indices]
            
            # Sort subcomplex by score (descending order)
            sorted_indices = np.argsort(-subcomplex_scores)
            subcomplex = subcomplex[sorted_indices]
            subcomplex_scores = subcomplex_scores[sorted_indices]
            
            # Select worst point in subcomplex for reflection
            worst_point = subcomplex[-1].copy()
            worst_score = subcomplex_scores[-1]
            
            # Compute centroid of all points except worst
            centroid = np.mean(subcomplex[:-1], axis=0)
            
            # Reflection step
            reflection_point = 2.0 * centroid - worst_point
            
            # Make sure reflection_point is within [0,1] bounds
            reflection_point = np.clip(reflection_point, 0, 1)
            
            # Evaluate reflection point
            reflection_params = self._denormalize_individual(reflection_point)
            reflection_score = self._evaluate_parameters(reflection_params)
            num_evaluations += 1
            
            # If reflection is better than worst, accept it
            if reflection_score is not None and reflection_score > worst_score:
                # Replace worst point with reflection
                subcomplex[-1] = reflection_point
                subcomplex_scores[-1] = reflection_score
            else:
                # Contraction step - try a point between centroid and worst
                contraction_point = (centroid + worst_point) / 2.0
                
                # Evaluate contraction point
                contraction_params = self._denormalize_individual(contraction_point)
                contraction_score = self._evaluate_parameters(contraction_params)
                num_evaluations += 1
                
                # If contraction is better than worst, accept it
                if contraction_score is not None and contraction_score > worst_score:
                    # Replace worst point with contraction
                    subcomplex[-1] = contraction_point
                    subcomplex_scores[-1] = contraction_score
                else:
                    # Random point step - generate a random point
                    random_point = np.random.random(num_params)
                    
                    # Evaluate random point
                    random_params = self._denormalize_individual(random_point)
                    random_score = self._evaluate_parameters(random_params)
                    num_evaluations += 1
                    
                    # Replace worst point with random point regardless of score
                    subcomplex[-1] = random_point
                    subcomplex_scores[-1] = random_score if random_score is not None else float('-inf')
            
            # Replace points in original complex with evolved subcomplex
            complex_points[subcomplex_indices] = subcomplex
            complex_scores[subcomplex_indices] = subcomplex_scores
            
            # Sort complex by score after each evolution step
            sorted_indices = np.argsort(-complex_scores)
            complex_points = complex_points[sorted_indices]
            complex_scores = complex_scores[sorted_indices]
        
        return complex_points, complex_scores, num_evaluations
    
    def _select_subcomplex_indices(self, complex_size, subcomplex_size):
        """
        Select indices for the subcomplex according to a trapezoidal probability distribution.
        
        Args:
            complex_size: Size of the complex
            subcomplex_size: Size of the subcomplex
            
        Returns:
            ndarray: Indices of selected points
        """
        # Create probability vector for selection (trapezoidal distribution)
        p = np.zeros(complex_size)
        for i in range(complex_size):
            p[i] = (2.0 * (complex_size - i)) / (complex_size * (complex_size + 1))
        
        # Select subcomplex_size indices without replacement
        indices = np.random.choice(complex_size, size=subcomplex_size, replace=False, p=p)
        
        return indices
    
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
            import matplotlib.pyplot as plt
            
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
            plt.title('SCE-UA Optimization Progress')
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
            
            # Plot population distribution at selected iterations
            # Select a few iterations to plot
            selected_iterations = [0]  # Always include initial population
            if len(iterations) > 1:
                # Add middle and final iterations
                mid_iteration = iterations[len(iterations) // 2]
                last_iteration = iterations[-1]
                selected_iterations.extend([mid_iteration, last_iteration])
            
            # Create population distribution plots
            for it in selected_iterations:
                h = history[iterations.index(it)]
                if 'population_scores' in h:
                    plt.figure(figsize=(10, 6))
                    plt.hist(h['population_scores'], bins=20, alpha=0.7)
                    plt.axvline(h['best_score'], color='r', linestyle='-', 
                                label=f'Best: {h["best_score"]:.4f}')
                    plt.axvline(h['mean_score'], color='g', linestyle='--', 
                                label=f'Mean: {h["mean_score"]:.4f}')
                    plt.xlabel(f'Performance Metric ({self.target_metric})')
                    plt.ylabel('Frequency')
                    plt.title(f'Population Score Distribution at Iteration {it}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"population_distribution_iter_{it}.png", dpi=300, bbox_inches='tight')
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