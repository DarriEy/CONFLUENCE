"""
Large Sample Emulator for CONFLUENCE.

This module provides functionality to generate spatially varying parameter sets
for large sample hydrological model emulation, sensitivity analysis, and ensemble modeling.
"""

import os
import re
import numpy as np # type: ignore
import netCDF4 as nc4 # type: ignore
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from utils.configHandling_utils.logging_utils import get_function_logger # type: ignore


class LargeSampleEmulator:
    """
    Handles the setup for large sample emulation, primarily by generating
    spatially varying trial parameter files based on random sampling within
    defined bounds.
        """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_path = self.project_dir / 'settings' / 'SUMMA'
        self.emulator_output_dir = self.project_dir / "emulation" / self.config.get('EXPERIMENT_ID')
        self.emulator_output_dir.mkdir(parents=True, exist_ok=True)

        self.local_param_info_path = self.settings_path / 'localParamInfo.txt'
        self.basin_param_info_path = self.settings_path / 'basinParamInfo.txt'
        self.attribute_file_path = self.settings_path / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        self.trial_param_output_path = self.emulator_output_dir / f"trialParams_emulator_{self.config.get('EXPERIMENT_ID')}.nc"

        # Get parameters to vary from config, handle potential None or empty string
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params_to_emulate = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params_to_emulate = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []

        # Get sampling settings
        self.num_samples = self.config.get('EMULATION_NUM_SAMPLES', 100)
        self.random_seed = self.config.get('EMULATION_SEED', 42)
        self.sampling_method = self.config.get('EMULATION_SAMPLING_METHOD', 'uniform').lower()

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        self.logger.info(f"Large Sample Emulator initialized with:")
        self.logger.info(f"  Local parameters: {self.local_params_to_emulate}")
        self.logger.info(f"  Basin parameters: {self.basin_params_to_emulate}")
        self.logger.info(f"  Samples: {self.num_samples}")
        self.logger.info(f"  Sampling method: {self.sampling_method}")
        self.logger.info(f"  Random seed: {self.random_seed}")

    @get_function_logger
    def generate_spatially_varying_trial_params(self, local_bounds: Dict, basin_bounds: Dict):
        """Generates the trialParamFile.nc with spatially varying parameters."""
        self.logger.info(f"Generating spatially varying trial parameters file at: {self.trial_param_output_path}")

        if not self.attribute_file_path.exists():
            raise FileNotFoundError(f"Attribute file not found, needed for HRU/GRU info: {self.attribute_file_path}")

        # Read HRU and GRU information from the attributes file
        with nc4.Dataset(self.attribute_file_path, 'r') as att_ds:
            if 'hru' not in att_ds.dimensions:
                raise ValueError(f"Dimension 'hru' not found in attributes file: {self.attribute_file_path}")
            num_hru = len(att_ds.dimensions['hru'])
            hru_ids = att_ds.variables['hruId'][:]
            hru2gru_ids = att_ds.variables['hru2gruId'][:]

            # Need gruId variable and dimension for basin params
            if self.basin_params_to_emulate:
                if 'gru' not in att_ds.dimensions or 'gruId' not in att_ds.variables:
                    raise ValueError(f"Dimension 'gru' or variable 'gruId' not found in attributes file, needed for basin parameters: {self.attribute_file_path}")
                num_gru = len(att_ds.dimensions['gru'])
                gru_ids = att_ds.variables['gruId'][:]
                # Create a mapping from gru_id to its index in the gru dimension
                gru_id_to_index = {gid: idx for idx, gid in enumerate(gru_ids)}
            else:
                num_gru = 0 # Not needed if no basin params

        self.logger.info(f"Found {num_hru} HRUs and {num_gru} GRUs.")

        # Create the output NetCDF file
        with nc4.Dataset(self.trial_param_output_path, 'w', format='NETCDF4') as tp_ds:
            # Define dimensions
            tp_ds.createDimension('hru', num_hru)

            # --- Create hruId variable (essential for matching) ---
            hru_id_var = tp_ds.createVariable('hruId', 'i4', ('hru',))
            hru_id_var[:] = hru_ids
            hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
            hru_id_var.units = '-'

            # --- Generate and write Local (HRU-level) parameters ---
            for param_name, bounds in local_bounds.items():
                self.logger.debug(f"Generating random values for HRU parameter: {param_name}")
                min_val, max_val = bounds['min'], bounds['max']
                
                # Generate random values based on sampling method
                random_values = self._generate_random_values(min_val, max_val, num_hru, param_name)

                # Create variable in NetCDF
                param_var = tp_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False) # Use f8 = float64
                param_var[:] = random_values
                param_var.long_name = f"Trial value for {param_name}"
                param_var.units = "N/A" # Usually unitless in trialParam, units applied later via tables

            # --- Generate and write Basin (GRU-level) parameters ---
            for param_name, bounds in basin_bounds.items():
                self.logger.debug(f"Generating random values for GRU parameter: {param_name}")
                min_val, max_val = bounds['min'], bounds['max']
                
                # Generate random values for GRUs based on sampling method
                gru_random_values = self._generate_random_values(min_val, max_val, num_gru, param_name)

                # Map GRU values to HRUs
                hru_mapped_values = np.zeros(num_hru, dtype=np.float64)
                for i in range(num_hru):
                    hru_gru_id = hru2gru_ids[i] # Get the GRU ID for this HRU
                    gru_index = gru_id_to_index.get(hru_gru_id) # Find the index corresponding to this GRU ID
                    if gru_index is not None:
                        hru_mapped_values[i] = gru_random_values[gru_index]
                    else:
                        self.logger.warning(f"Could not find index for GRU ID {hru_gru_id} associated with HRU {hru_ids[i]}. Setting parameter {param_name} to 0 for this HRU.")
                        hru_mapped_values[i] = 0.0 # Or handle as an error, or use default?

                # Create variable in NetCDF (dimension is still 'hru')
                param_var = tp_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                param_var[:] = hru_mapped_values
                param_var.long_name = f"Trial value for {param_name} (GRU-based)"
                param_var.units = "N/A"

            # Add global attributes
            tp_ds.description = "SUMMA Trial Parameter file with spatially varying values for large sample emulation"
            tp_ds.history = f"Created on {datetime.now().isoformat()} by CONFLUENCE LargeSampleEmulator"
            tp_ds.confluence_experiment_id = self.config.get('EXPERIMENT_ID')
            tp_ds.sampling_method = self.sampling_method
            tp_ds.random_seed = self.random_seed
            tp_ds.generated_samples = self.num_samples

        self.logger.info(f"Finished writing emulator trial parameter file: {self.trial_param_output_path}")

    def _generate_random_values(self, min_val: float, max_val: float, num_values: int, param_name: str) -> np.ndarray:
        """
        Generate random parameter values using the specified sampling method.
        
        Args:
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            num_values: Number of values to generate
            param_name: Name of parameter (for logging)
            
        Returns:
            np.ndarray: Array of random values within the specified bounds
        """
        try:
            if self.sampling_method == 'uniform':
                # Simple uniform random sampling
                return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
                
            elif self.sampling_method == 'lhs':
                # Latin Hypercube Sampling (if scipy is available)
                try:
                    from scipy.stats import qmc # type: ignore
                    
                    # Create a normalized LHS sampler in [0, 1]
                    sampler = qmc.LatinHypercube(d=1, seed=self.random_seed)
                    samples = sampler.random(n=num_values)
                    
                    # Scale to the parameter range
                    scaled_samples = qmc.scale(samples, [min_val], [max_val]).flatten()
                    return scaled_samples.astype(np.float64)
                    
                except ImportError:
                    self.logger.warning(f"Could not import scipy.stats.qmc for Latin Hypercube Sampling. Falling back to uniform sampling for {param_name}.")
                    return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
                    
            elif self.sampling_method == 'sobol':
                # Sobol sequence (if scipy is available)
                try:
                    from scipy.stats import qmc # type: ignore
                    
                    # Create a Sobol sequence generator
                    sampler = qmc.Sobol(d=1, scramble=True, seed=self.random_seed)
                    samples = sampler.random(n=num_values)
                    
                    # Scale to the parameter range
                    scaled_samples = qmc.scale(samples, [min_val], [max_val]).flatten()
                    return scaled_samples.astype(np.float64)
                    
                except ImportError:
                    self.logger.warning(f"Could not import scipy.stats.qmc for Sobol sequence. Falling back to uniform sampling for {param_name}.")
                    return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
            
            else:
                self.logger.warning(f"Unknown sampling method '{self.sampling_method}'. Falling back to uniform sampling for {param_name}.")
                return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
                
        except Exception as e:
            self.logger.error(f"Error generating random values for {param_name}: {str(e)}")
            self.logger.warning(f"Using uniform sampling as fallback for {param_name}.")
            return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
