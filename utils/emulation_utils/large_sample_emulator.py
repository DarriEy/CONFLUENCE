"""
Large Sample Emulator for CONFLUENCE with support for multiple ensemble runs.

This enhanced version creates multiple parameter sets in separate run directories
for ensemble simulations and sensitivity analysis.
"""

import os
import re
import numpy as np # type: ignore
import netCDF4 as nc4 # type: ignore
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from utils.configHandling_utils.logging_utils import get_function_logger # type: ignore

class LargeSampleEmulator:
    """
    Handles the setup for large sample emulation, primarily by generating
    spatially varying trial parameter files for ensemble simulations.
    
    This enhanced version supports creating multiple run directories with
    different parameter sets for ensemble modeling and sensitivity analysis.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_path = self.project_dir / 'settings' / 'SUMMA'
        
        # Base directory for all emulation outputs
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        self.emulator_output_dir = self.project_dir / "emulation" / self.experiment_id
        self.emulator_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for ensemble runs
        self.ensemble_dir = self.emulator_output_dir / "ensemble_runs"
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        self.local_param_info_path = self.settings_path / 'localParamInfo.txt'
        self.basin_param_info_path = self.settings_path / 'basinParamInfo.txt'
        self.attribute_file_path = self.settings_path / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        self.filemanager_path = self.settings_path / self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        # Summary file (will contain all parameter sets)
        self.summary_file_path = self.emulator_output_dir / f"trialParams_summary_{self.experiment_id}.nc"
        
        # Get parameters to vary from config, handle potential None or empty string
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params_to_emulate = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params_to_emulate = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []

        # Get sampling settings
        self.num_samples = self.config.get('EMULATION_NUM_SAMPLES', 100)
        self.random_seed = self.config.get('EMULATION_SEED', 42)
        self.sampling_method = self.config.get('EMULATION_SAMPLING_METHOD', 'uniform').lower()
        self.create_individual_runs = self.config.get('EMULATION_CREATE_INDIVIDUAL_RUNS', True)

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        self.logger.info(f"Large Sample Emulator initialized with:")
        self.logger.info(f"  Local parameters: {self.local_params_to_emulate}")
        self.logger.info(f"  Basin parameters: {self.basin_params_to_emulate}")
        self.logger.info(f"  Samples: {self.num_samples}")
        self.logger.info(f"  Sampling method: {self.sampling_method}")
        self.logger.info(f"  Random seed: {self.random_seed}")
        self.logger.info(f"  Creating individual run directories: {self.create_individual_runs}")

    @get_function_logger
    def run_emulation_setup(self):
        """Orchestrates the setup for large sample emulation with multiple runs."""
        self.logger.info("Starting Large Sample Emulation setup")
        try:
            # Parse parameter bounds
            local_bounds = self._parse_param_info(self.local_param_info_path, self.local_params_to_emulate)
            basin_bounds = self._parse_param_info(self.basin_param_info_path, self.basin_params_to_emulate)

            if not local_bounds and not basin_bounds:
                 self.logger.warning("No parameters specified or found for emulation. Skipping trial parameter file generation.")
                 return None # Indicate nothing was generated

            # Generate the summary file with all parameter sets
            self.generate_parameter_summary(local_bounds, basin_bounds)
            
            # Create individual run directories if requested
            if self.create_individual_runs:
                self.create_ensemble_run_directories(local_bounds, basin_bounds)
                self.logger.info(f"Successfully created {self.num_samples} ensemble run directories in {self.ensemble_dir}")
            
            self.logger.info(f"Large sample emulation setup completed successfully")
            return self.summary_file_path

        except FileNotFoundError as e:
            self.logger.error(f"Required file not found during emulation setup: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during large sample emulation setup: {str(e)}")
            raise

    def _parse_param_info(self, file_path: Path, param_names: list) -> Dict[str, Dict[str, float]]:
        """Parses min/max bounds from SUMMA parameter info files."""
        bounds = {}
        if not param_names:
            return bounds # Return empty if no parameters requested

        self.logger.info(f"Parsing parameter bounds from: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Parameter info file not found: {file_path}")

        found_params = set()
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!'):
                        continue

                    # Use regex for more robust parsing, allowing for variable whitespace
                    match = re.match(r"^\s*(\w+)\s*\|\s*([\d\.\-eE]+)\s*\|\s*([\d\.\-eE]+)\s*\|.*$", line)
                    if match:
                        param_name, max_val_str, min_val_str = match.groups()
                        if param_name in param_names:
                            try:
                                min_val = float(min_val_str)
                                max_val = float(max_val_str)
                                if min_val >= max_val:
                                    self.logger.warning(f"Parameter '{param_name}' in {file_path} has min >= max ({min_val} >= {max_val}). Check file.")
                                bounds[param_name] = {'min': min_val, 'max': max_val}
                                found_params.add(param_name)
                                self.logger.debug(f"Found bounds for {param_name}: min={min_val}, max={max_val}")
                            except ValueError:
                                self.logger.warning(f"Could not parse bounds for parameter '{param_name}' in line: {line}")
                    else:
                         # Log lines that don't match the expected format (excluding comments/empty)
                         self.logger.debug(f"Skipping non-parameter line or unexpected format in {file_path}: {line}")

            # Check if all requested parameters were found
            missing_params = set(param_names) - found_params
            if missing_params:
                self.logger.warning(f"Could not find bounds for the following parameters in {file_path}: {', '.join(missing_params)}")

        except Exception as e:
            self.logger.error(f"Error reading or parsing {file_path}: {str(e)}")
            raise
        return bounds

    @get_function_logger
    def generate_parameter_summary(self, local_bounds: Dict, basin_bounds: Dict):
        """Generates a summary file with all parameter sets."""
        self.logger.info(f"Generating parameter summary file at: {self.summary_file_path}")

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


        # Store parameter values for each sample (run) and each parameter
        # Dictionary to hold all random values for each parameter across all samples
        all_param_values = {}
        
        # Generate all parameter values for all samples upfront
        for param_name, bounds in local_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            # For each parameter, store values for all HRUs across all samples
            # Shape: [num_samples, num_hru]
            all_param_values[param_name] = np.array([
                self._generate_random_values(min_val, max_val, num_hru, param_name) 
                for _ in range(self.num_samples)
            ])
        
        # Similarly for basin parameters
        for param_name, bounds in basin_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            # Generate GRU values for all samples
            gru_values = np.array([
                self._generate_random_values(min_val, max_val, num_gru, param_name)
                for _ in range(self.num_samples)
            ])
            
            # Map GRU values to HRUs for each sample
            # Shape: [num_samples, num_hru]
            hru_mapped_values = np.zeros((self.num_samples, num_hru), dtype=np.float64)
            
            for sample_idx in range(self.num_samples):
                for hru_idx in range(num_hru):
                    hru_gru_id = hru2gru_ids[hru_idx]
                    gru_index = gru_id_to_index.get(hru_gru_id)
                    if gru_index is not None:
                        hru_mapped_values[sample_idx, hru_idx] = gru_values[sample_idx, gru_index]
                    else:
                        self.logger.warning(f"Could not find index for GRU ID {hru_gru_id} associated with HRU {hru_ids[hru_idx]}. Setting parameter {param_name} to 0 for this HRU.")
                        hru_mapped_values[sample_idx, hru_idx] = 0.0
            
            all_param_values[param_name] = hru_mapped_values

        # Create the summary NetCDF file
        with nc4.Dataset(self.summary_file_path, 'w', format='NETCDF4') as summary_ds:
            # Define dimensions
            summary_ds.createDimension('hru', num_hru)
            summary_ds.createDimension('run', self.num_samples)

            # Create hruId variable
            hru_id_var = summary_ds.createVariable('hruId', 'i4', ('hru',))
            hru_id_var[:] = hru_ids
            hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
            hru_id_var.units = '-'
            
            # Create run index variable (helpful for referencing)
            run_var = summary_ds.createVariable('runIndex', 'i4', ('run',))
            run_var[:] = np.arange(self.num_samples)
            run_var.long_name = 'Run/Sample Index'
            run_var.units = '-'

            # Create variables for each parameter
            # Store all values across all samples: [run, hru]
            for param_name, values in all_param_values.items():
                param_var = summary_ds.createVariable(param_name, 'f8', ('run', 'hru',), fill_value=False)
                param_var[:] = values
                param_var.long_name = f"Trial values for {param_name} across all runs"
                
                # Determine if it's a local or basin parameter
                is_local = param_name in local_bounds
                param_var.parameter_type = "local" if is_local else "basin"
                param_var.units = "N/A"

            # Add global attributes
            summary_ds.description = "SUMMA Trial Parameter summary file for multiple ensemble runs"
            summary_ds.history = f"Created on {datetime.now().isoformat()} by CONFLUENCE LargeSampleEmulator"
            summary_ds.confluence_experiment_id = self.experiment_id
            summary_ds.sampling_method = self.sampling_method
            summary_ds.random_seed = self.random_seed
            summary_ds.num_samples = self.num_samples

        self.logger.info(f"Successfully generated parameter summary file: {self.summary_file_path}")
        return self.summary_file_path, all_param_values

    @get_function_logger
    def create_ensemble_run_directories(self, local_bounds: Dict, basin_bounds: Dict):
        """
        Creates individual run directories with unique parameter sets.
        
        This allows for running multiple model instances with different parameter sets.
        """
        self.logger.info(f"Creating {self.num_samples} ensemble run directories in {self.ensemble_dir}")
        
        # First, generate all parameter values using the summary function
        summary_file, all_param_values = self.generate_parameter_summary(local_bounds, basin_bounds)
        
        # Read HRU information from the attributes file
        with nc4.Dataset(self.attribute_file_path, 'r') as att_ds:
            hru_ids = att_ds.variables['hruId'][:]
        
        # Create run directories and parameter files
        for run_idx in range(self.num_samples):
            # Create run directory
            run_dir = self.ensemble_dir / f"run_{run_idx:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Create settings directory in run directory
            run_settings_dir = run_dir / "settings" / "SUMMA"
            run_settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trial parameter file for this run
            trial_param_path = run_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
            
            # Extract this run's parameter values from the summary
            with nc4.Dataset(trial_param_path, 'w', format='NETCDF4') as tp_ds:
                # Define dimensions
                tp_ds.createDimension('hru', len(hru_ids))
                
                # Create hruId variable
                hru_id_var = tp_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Add each parameter with its values for this run
                for param_name, values in all_param_values.items():
                    param_var = tp_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                    param_var[:] = values[run_idx]  # Get values for this run
                    param_var.long_name = f"Trial value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                tp_ds.description = f"SUMMA Trial Parameter file for run {run_idx:04d}"
                tp_ds.history = f"Created on {datetime.now().isoformat()} by CONFLUENCE LargeSampleEmulator"
                tp_ds.confluence_experiment_id = self.experiment_id
                tp_ds.run_index = run_idx
            
            # Copy the file manager and modify it for this run
            self._create_run_file_manager(run_idx, run_dir, run_settings_dir)
            
            # Copy other necessary settings files (static files)
            self._copy_static_settings_files(run_settings_dir)
            
            self.logger.debug(f"Created run directory and parameter file for run_{run_idx:04d}")
        
        self.logger.info(f"Successfully created {self.num_samples} ensemble run directories")
        return self.ensemble_dir

    def _create_run_file_manager(self, run_idx: int, run_dir: Path, run_settings_dir: Path):
        """
        Create a modified fileManager for a specific run directory.
        
        Args:
            run_idx: Index of the current run
            run_dir: Base directory for this run
            run_settings_dir: Settings directory for this run
        """
        if not self.filemanager_path.exists():
            self.logger.warning(f"Original fileManager not found at {self.filemanager_path}, skipping creation for run_{run_idx:04d}")
            return
        
        # Read the original file manager
        with open(self.filemanager_path, 'r') as f:
            fm_lines = f.readlines()
        
        # Path to the new file manager
        new_fm_path = run_settings_dir / os.path.basename(self.filemanager_path)
        
        # Modify relevant lines for this run
        modified_lines = []
        for line in fm_lines:
            if "outFilePrefix" in line:
                # Update output file prefix to include run index
                prefix_parts = line.split("'")
                if len(prefix_parts) >= 3:
                    original_prefix = prefix_parts[1]
                    new_prefix = f"{original_prefix}_run{run_idx:04d}"
                    modified_line = line.replace(f"'{original_prefix}'", f"'{new_prefix}'")
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)  # Keep unchanged if format unexpected
            elif "outputPath" in line:
                # Update output path to use this run's directory
                output_path = run_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                output_path_str = str(output_path).replace('\\', '/')  # Ensure forward slashes for SUMMA
                modified_line = f"outputPath           '{output_path_str}/' ! \n"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)  # Keep other lines unchanged
        
        # Write the modified file manager
        with open(new_fm_path, 'w') as f:
            f.writelines(modified_lines)
        
        self.logger.debug(f"Created modified fileManager for run_{run_idx:04d}")
        return new_fm_path

    def _copy_static_settings_files(self, run_settings_dir: Path):
        """
        Copy static SUMMA settings files to the run directory.
        
        Args:
            run_settings_dir: Target settings directory for this run
        """
        # Files to copy (could be made configurable)
        static_files = [
            'modelDecisions.txt',
            'outputControl.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL',
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL'
        ]
        
        # Copy each file if it exists
        for file_name in static_files:
            source_path = self.settings_path / file_name
            if source_path.exists():
                dest_path = run_settings_dir / file_name
                try:
                    shutil.copy2(source_path, dest_path)
                    self.logger.debug(f"Copied {file_name} to run settings directory")
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")

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
            
    @get_function_logger
    def run_ensemble_simulations(self):
        """
        Run SUMMA for each ensemble member (each run directory).
        
        This launches multiple SUMMA instances, one for each parameter set.
        Depending on computational resources, this can be sequential or parallel.
        """
        self.logger.info("Starting ensemble simulations")
        
        # Check if the ensemble directories exist
        if not self.ensemble_dir.exists() or not any(self.ensemble_dir.glob("run_*")):
            self.logger.error("Ensemble run directories not found. Run create_ensemble_run_directories first.")
            return None
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Run SUMMA for each ensemble member
        results = []
        for run_dir in sorted(self.ensemble_dir.glob("run_*")):
            run_name = run_dir.name
            self.logger.info(f"Running simulation for {run_name}")
            
            # Get fileManager path for this run
            run_settings_dir = run_dir / "settings" / "SUMMA"
            filemanager_path = run_settings_dir / os.path.basename(self.filemanager_path)
            
            if not filemanager_path.exists():
                self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                continue
            
            # Ensure output directory exists
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run SUMMA
            command = f"{summa_exe} -m {filemanager_path}"
            log_file = log_dir / f"{run_name}_summa.log"
            
            try:
                import subprocess
                with open(log_file, 'w') as f:
                    process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                
                self.logger.info(f"Successfully completed simulation for {run_name}")
                results.append((run_name, True, None))
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error running SUMMA for {run_name}: {str(e)}")
                results.append((run_name, False, str(e)))
                
            except Exception as e:
                self.logger.error(f"Unexpected error running {run_name}: {str(e)}")
                results.append((run_name, False, str(e)))
        
        # Summarize results
        successes = sum(1 for _, success, _ in results if success)
        self.logger.info(f"Completed {successes} out of {len(results)} ensemble simulations")
        
        return results