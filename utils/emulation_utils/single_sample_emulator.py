"""
Emulator for CONFLUENCE with support for multiple ensemble runs.

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

class SingleSampleEmulator:
    """
    Handles the setup for single sample emulation, primarily by generating
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

            # Store all parameter sets in a consolidated file
            consolidated_file = self.store_parameter_sets(local_bounds, basin_bounds)
            self.logger.info(f"All parameter sets stored in consolidated file: {consolidated_file}")

            # Generate the summary file with all parameter sets
            self.generate_parameter_summary(local_bounds, basin_bounds)
            
            # Create individual run directories if requested
            if self.create_individual_runs:
                self.create_ensemble_run_directories(local_bounds, basin_bounds)
                self.logger.info(f"Successfully created {self.num_samples} ensemble run directories in {self.ensemble_dir}")
            
            self.logger.info(f"Large sample emulation setup completed successfully")
            return consolidated_file

        except FileNotFoundError as e:
            self.logger.error(f"Required file not found during emulation setup: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during large sample emulation setup: {str(e)}")
            raise

    @get_function_logger
    def analyze_parameter_space(self, consolidated_file_path=None):
        """
        Analyze the parameter space of the ensemble simulations.
        
        Creates visualizations and statistics for the parameter space coverage,
        including correlation analysis, parameter distributions, and coverage metrics.
        
        Args:
            consolidated_file_path (Path, optional): Path to the consolidated parameter file.
                If None, uses the default path.
        
        Returns:
            Path: Path to the analysis results directory
        """
        self.logger.info("Starting parameter space analysis")
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats
            
            # Set default path if not provided
            if consolidated_file_path is None:
                consolidated_file_path = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
            
            if not consolidated_file_path.exists():
                self.logger.error(f"Consolidated parameter file not found: {consolidated_file_path}")
                return None
            
            # Create analysis directory
            analysis_dir = self.emulator_output_dir / "parameter_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Read the parameter file
            with nc4.Dataset(consolidated_file_path, 'r') as nc_file:
                # Get dimensions
                num_runs = len(nc_file.dimensions['run'])
                num_hru = len(nc_file.dimensions['hru'])
                
                # Get parameter names (excluding dimensions and special variables)
                param_names = [var for var in nc_file.variables if var not in ['hruId', 'runIndex']]
                
                # Extract parameter values
                param_data = {}
                param_types = {}
                param_bounds = {}
                
                for param in param_names:
                    var = nc_file.variables[param]
                    # Get mean parameter value for each run (averaging across HRUs)
                    param_data[param] = np.mean(var[:], axis=1)  # Shape: [num_runs]
                    
                    # Get parameter type and bounds if available
                    if hasattr(var, 'parameter_type'):
                        param_types[param] = var.parameter_type
                    else:
                        param_types[param] = "unknown"
                    
                    if hasattr(var, 'min_value') and hasattr(var, 'max_value'):
                        param_bounds[param] = (var.min_value, var.max_value)
                    else:
                        param_bounds[param] = (np.min(param_data[param]), np.max(param_data[param]))
            
            # Create a DataFrame for easier analysis
            df = pd.DataFrame(param_data)
            
            # Save parameter data to CSV
            df.to_csv(analysis_dir / "parameter_values.csv")
            
            # Calculate basic statistics
            stats_df = pd.DataFrame({
                'Mean': df.mean(),
                'Std Dev': df.std(),
                'Min': df.min(),
                'Max': df.max(),
                'Median': df.median(),
                'Range': df.max() - df.min()
            })
            
            # Add parameter types and normalized range
            stats_df['Type'] = pd.Series(param_types)
            stats_df['NormalizedRange'] = stats_df['Range'] / (
                pd.Series({p: bounds[1] - bounds[0] for p, bounds in param_bounds.items()})
            )
            
            # Save statistics
            stats_df.to_csv(analysis_dir / "parameter_statistics.csv")
            
            # Create correlation matrix
            corr_matrix = df.corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Parameter Correlation Matrix')
            plt.tight_layout()
            plt.savefig(analysis_dir / "parameter_correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot parameter distributions
            if len(param_names) <= 20:  # Only create individual plots if parameters are not too many
                for param in param_names:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot histogram
                    sns.histplot(df[param], kde=True)
                    
                    # Add bounds if available
                    if param in param_bounds:
                        min_val, max_val = param_bounds[param]
                        plt.axvline(x=min_val, color='r', linestyle='--', label=f'Min: {min_val:.4f}')
                        plt.axvline(x=max_val, color='g', linestyle='--', label=f'Max: {max_val:.4f}')
                    
                    # Add statistics
                    mean_val = df[param].mean()
                    std_val = df[param].std()
                    plt.axvline(x=mean_val, color='k', linestyle='-', label=f'Mean: {mean_val:.4f}')
                    
                    # Add title and labels
                    plt.title(f'Distribution of {param} ({param_types.get(param, "unknown")})')
                    plt.xlabel('Parameter Value')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save plot
                    plt.savefig(analysis_dir / f"param_dist_{param}.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Create pairplot for parameter pairs (limit to most important parameters if there are many)
            if len(param_names) <= 10:
                # Create pairplot for all parameters
                plt.figure(figsize=(15, 15))
                sns.pairplot(df, diag_kind='kde')
                plt.suptitle('Parameter Pairwise Relationships', y=1.02)
                plt.savefig(analysis_dir / "parameter_pairplot.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Select top parameters based on range utilization
                top_params = stats_df.sort_values('NormalizedRange', ascending=False).head(8).index.tolist()
                
                plt.figure(figsize=(15, 15))
                sns.pairplot(df[top_params], diag_kind='kde')
                plt.suptitle('Parameter Pairwise Relationships (Top 8 Parameters)', y=1.02)
                plt.savefig(analysis_dir / "parameter_pairplot_top.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Calculate parameter space coverage metrics
            coverage_metrics = {
                'Sampling Method': self.sampling_method,
                'Number of Samples': num_runs,
                'Number of Parameters': len(param_names),
                'Average Range Utilization': stats_df['NormalizedRange'].mean(),
                'Min Range Utilization': stats_df['NormalizedRange'].min(),
                'Max Range Utilization': stats_df['NormalizedRange'].max(),
                'Average Correlation': np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]).mean(),
                'Max Correlation': np.max(np.abs(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]))
            }
            
            # Save coverage metrics
            with open(analysis_dir / "parameter_coverage_metrics.txt", 'w') as f:
                f.write("Parameter Space Coverage Metrics\n")
                f.write("================================\n\n")
                for metric, value in coverage_metrics.items():
                    f.write(f"{metric}: {value}\n")
            
            # Create summary report
            with open(analysis_dir / "parameter_analysis_report.txt", 'w') as f:
                f.write(f"Parameter Space Analysis Report\n")
                f.write(f"=============================\n\n")
                f.write(f"Experiment ID: {self.experiment_id}\n")
                f.write(f"Ensemble Size: {num_runs}\n")
                f.write(f"Number of Parameters: {len(param_names)}\n")
                f.write(f"Parameter Types: {sorted(set(param_types.values()))}\n\n")
                
                f.write(f"Parameter Space Coverage:\n")
                for metric, value in coverage_metrics.items():
                    f.write(f"  {metric}: {value}\n")
                
                f.write(f"\nParameters with Highest Range Utilization:\n")
                top_range = stats_df.sort_values('NormalizedRange', ascending=False).head(5)
                for param, row in top_range.iterrows():
                    f.write(f"  {param} ({row['Type']}): {row['NormalizedRange']:.4f}\n")
                
                f.write(f"\nParameters with Lowest Range Utilization:\n")
                bottom_range = stats_df.sort_values('NormalizedRange').head(5)
                for param, row in bottom_range.iterrows():
                    f.write(f"  {param} ({row['Type']}): {row['NormalizedRange']:.4f}\n")
                
                f.write(f"\nHighest Parameter Correlations:\n")
                # Get the upper triangle of the correlation matrix
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                # Find the top absolute correlations
                top_corr = upper.unstack().dropna().abs().sort_values(ascending=False).head(10)
                for (param1, param2), corr in top_corr.items():
                    f.write(f"  {param1} and {param2}: {upper.loc[param1, param2]:.4f}\n")
                
                f.write(f"\nFiles Generated:\n")
                f.write(f"  - parameter_values.csv: Raw parameter values for all runs\n")
                f.write(f"  - parameter_statistics.csv: Statistical metrics for each parameter\n")
                f.write(f"  - parameter_correlation_matrix.png: Visualization of parameter correlations\n")
                f.write(f"  - parameter_pairplot.png: Pairwise relationships between parameters\n")
                f.write(f"  - parameter_coverage_metrics.txt: Metrics for parameter space coverage\n")
                
                f.write(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"Parameter space analysis completed. Results saved to {analysis_dir}")
            return analysis_dir
            
        except ImportError as e:
            self.logger.warning(f"Could not complete parameter space analysis due to missing library: {str(e)}")
            self.logger.warning("Make sure pandas, numpy, matplotlib, seaborn, and scipy are available.")
            return None
        except Exception as e:
            self.logger.error(f"Error during parameter space analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

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
                    new_prefix = f"{original_prefix}"  # _run{run_idx:04d}
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
            elif "settingsPath" in line:
                # Update settings path to point to the run-specific settings directory
                settings_path_str = str(run_settings_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                modified_line = f"settingsPath         '{settings_path_str}/'\n"
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
            'TBL_VEGPARM.TBL',
            'attributes.nc',
            'coldState.nc',
            'forcingFileList.txt'
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
        Run SUMMA and MizuRoute for each ensemble member (each run directory).
        
        This launches multiple SUMMA instances, one for each parameter set,
        followed by MizuRoute for streamflow routing, and extracts results
        to run-specific results directories.
        
        Returns:
            List: Results of the ensemble simulations including success/failure status
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
        
        # Get all run directories
        run_dirs = sorted(self.ensemble_dir.glob("run_*"))
        
        # Check if we should run in parallel
        run_parallel = self.config.get('EMULATION_PARALLEL_ENSEMBLE', False)
        max_parallel_jobs = self.config.get('EMULATION_MAX_PARALLEL_JOBS', 10)
        
        if run_parallel and len(run_dirs) > 1:
            # Run in parallel mode
            return self._run_parallel_ensemble(summa_exe, run_dirs, max_parallel_jobs)
        else:
            # Run in sequential mode
            return self._run_sequential_ensemble(summa_exe, run_dirs)

    def get_results(self):
        """
        Generate a summary of the ensemble results.
        
        Returns:
            Dict: A dictionary containing summarized ensemble results
        """
        self.logger.info("Gathering ensemble results")
        
        # Check if the ensemble directories exist
        if not self.ensemble_dir.exists() or not any(self.ensemble_dir.glob("run_*")):
            self.logger.error("Ensemble run directories not found.")
            return {"error": "No ensemble run directories found"}
        
        # Get all run directories
        run_dirs = sorted(self.ensemble_dir.glob("run_*"))
        
        # Count successful runs
        successful_runs = 0
        failed_runs = 0
        
        for run_dir in run_dirs:
            # Check if SUMMA output exists
            summa_output = run_dir / "simulations" / self.experiment_id / "SUMMA" / f"{self.experiment_id}*.nc"
            summa_success = len(list(run_dir.glob(str(summa_output)))) > 0
            
            # Check if MizuRoute output exists
            mizu_output = run_dir / "simulations" / self.experiment_id / "mizuRoute" / "*.nc"
            mizu_success = len(list(run_dir.glob(str(mizu_output)))) > 0
            
            # Check if extracted results exist
            results_file = run_dir / "results" / f"{run_dir.name}_streamflow.csv" 
            results_success = results_file.exists()
            
            if results_success or (summa_success and not self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped'):
                successful_runs += 1
            else:
                failed_runs += 1
        
        # Get performance metrics if available
        metrics_file = self.emulator_output_dir / "ensemble_analysis" / "performance_metrics_summary.csv"
        metrics = None
        
        if metrics_file.exists():
            try:
                import pandas as pd
                metrics = pd.read_csv(metrics_file, index_col=0).to_dict()
            except Exception as e:
                self.logger.error(f"Error reading performance metrics: {str(e)}")
        
        # Create summary
        summary = {
            "experiment_id": self.experiment_id,
            "total_runs": len(run_dirs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "parameter_sets_file": str(self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"),
            "analysis_dir": str(self.emulator_output_dir / "ensemble_analysis"),
            "metrics": metrics,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary

    def _run_sequential_ensemble(self, summa_exe, run_dirs):
        """Run ensemble simulations one after another, including post-processing."""
        self.logger.info(f"Running {len(run_dirs)} ensemble simulations sequentially")
        
        # Get MizuRoute executable path
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Check if we're in lumped domain mode
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        results = []
        for run_dir in run_dirs:
            run_name = run_dir.name
            self.logger.info(f"Running simulation for {run_name}")
            
            # Get fileManager path for this run
            run_settings_dir = run_dir / "settings" / "SUMMA"
            filemanager_path = run_settings_dir / os.path.basename(self.filemanager_path)
            
            if not filemanager_path.exists():
                self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                results.append((run_name, False, "FileManager not found"))
                continue
            
            # Ensure output directory exists
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Results directory
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a debugging log for this run
            debug_log_path = results_dir / f"{run_name}_debug.log"
            with open(debug_log_path, 'w') as debug_log:
                debug_log.write(f"Debug log for {run_name}\n")
                debug_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                debug_log.write(f"SUMMA executable: {summa_exe}\n")
                debug_log.write(f"SUMMA fileManager: {filemanager_path}\n")
                debug_log.write(f"Domain definition method: {self.config.get('DOMAIN_DEFINITION_METHOD')}\n")
                debug_log.write(f"Is lumped domain: {is_lumped}\n\n")
                
                try:
                    # 1. Run SUMMA
                    self.logger.info(f"Running SUMMA for {run_name}")
                    debug_log.write("\n--- Running SUMMA ---\n")
                    summa_command = f"{summa_exe} -m {filemanager_path}"
                    debug_log.write(f"SUMMA command: {summa_command}\n")
                    
                    summa_log_file = log_dir / f"{run_name}_summa.log"
                    debug_log.write(f"SUMMA log file: {summa_log_file}\n")
                    
                    import subprocess
                    with open(summa_log_file, 'w') as f:
                        process = subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                    
                    self.logger.info(f"Successfully completed SUMMA simulation for {run_name}")
                    debug_log.write("SUMMA simulation completed successfully\n")
                    
                    # Check if SUMMA output files were created
                    summa_output_files = list(output_dir.glob(f"{self.experiment_id}*.nc"))
                    debug_log.write(f"SUMMA output files found: {len(summa_output_files)}\n")
                    for file in summa_output_files:
                        debug_log.write(f"  - {file.name}\n")
                    
                    # 2. Run MizuRoute if configured to do so and NOT in lumped mode
                    run_mizuroute = (not is_lumped and 
                                    'SUMMA' in self.config.get('HYDROLOGICAL_MODEL', '').split(','))
                    
                    debug_log.write(f"\nRun MizuRoute? {run_mizuroute}\n")
                    debug_log.write(f"Domain definition method: {self.config.get('DOMAIN_DEFINITION_METHOD')}\n")
                    debug_log.write(f"Hydrological model: {self.config.get('HYDROLOGICAL_MODEL', '')}\n")
                    
                    if run_mizuroute:
                        self.logger.info(f"Running MizuRoute for {run_name}")
                        debug_log.write("\n--- Running MizuRoute ---\n")
                        
                        # Set up MizuRoute settings for this run
                        mizu_settings_dir = run_dir / "settings" / "mizuRoute"
                        debug_log.write(f"MizuRoute settings directory: {mizu_settings_dir}\n")
                        
                        # Create MizuRoute settings if they don't exist
                        if not mizu_settings_dir.exists():
                            debug_log.write("Creating MizuRoute settings...\n")
                            self._setup_run_mizuroute_settings(run_dir, run_name)
                            debug_log.write("MizuRoute settings created\n")
                        
                        # Get control file for MizuRoute
                        mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                        debug_log.write(f"MizuRoute control file: {mizu_control_file}\n")
                        
                        if not mizu_control_file.exists():
                            error_msg = f"MizuRoute control file not found: {mizu_control_file}"
                            self.logger.warning(error_msg)
                            debug_log.write(f"ERROR: {error_msg}\n")
                            debug_log.write("Skipping MizuRoute, will try to extract from SUMMA output directly\n")
                        else:
                            # Write contents of control file for debugging
                            debug_log.write("\nMizuRoute control file contents:\n")
                            with open(mizu_control_file, 'r') as f:
                                debug_log.write(f.read())
                            debug_log.write("\n")
                            
                            # Check if SUMMA timestep file exists and log details
                            mizu_input_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
                            timestep_file = mizu_input_dir / f"{self.experiment_id}_timestep.nc"
                            debug_log.write(f"Looking for SUMMA timestep file at: {timestep_file}\n")
                            
                            if timestep_file.exists():
                                debug_log.write(f"SUMMA timestep file found: {timestep_file}\n")
                                # Try to get basic info about the file
                                try:
                                    import xarray as xr
                                    with xr.open_dataset(timestep_file) as ds:
                                        debug_log.write(f"Timestep file dimensions: {dict(ds.dims)}\n")
                                        debug_log.write(f"Timestep file variables: {list(ds.variables)}\n")
                                        if 'time' in ds.dims:
                                            debug_log.write(f"Time dimension size: {ds.dims['time']}\n")
                                            debug_log.write(f"First time: {ds.time.values[0]}\n")
                                            debug_log.write(f"Last time: {ds.time.values[-1]}\n")
                                except Exception as e:
                                    debug_log.write(f"Error examining timestep file: {str(e)}\n")
                                
                                # Run MizuRoute
                                mizu_command = f"{mizu_exe} {mizu_control_file}"
                                debug_log.write(f"MizuRoute command: {mizu_command}\n")
                                
                                mizu_log_file = log_dir / f"{run_name}_mizuroute.log"
                                debug_log.write(f"MizuRoute log file: {mizu_log_file}\n")
                                
                                try:
                                    with open(mizu_log_file, 'w') as f:
                                        process = subprocess.run(mizu_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                                    
                                    self.logger.info(f"Successfully completed MizuRoute simulation for {run_name}")
                                    debug_log.write("MizuRoute simulation completed successfully\n")
                                    
                                    # Check if MizuRoute output files were created
                                    mizu_output_dir = run_dir / "simulations" / self.experiment_id / "mizuRoute"
                                    mizu_output_files = list(mizu_output_dir.glob("*.nc"))
                                    debug_log.write(f"MizuRoute output files found: {len(mizu_output_files)}\n")
                                    for file in mizu_output_files:
                                        debug_log.write(f"  - {file.name}\n")
                                
                                except subprocess.CalledProcessError as e:
                                    error_msg = f"MizuRoute execution failed: {str(e)}"
                                    self.logger.error(error_msg)
                                    debug_log.write(f"ERROR: {error_msg}\n")
                                    debug_log.write("Attempting to extract any available results despite MizuRoute failure\n")
                            else:
                                error_msg = f"SUMMA timestep file not found: {timestep_file}"
                                debug_log.write(f"ERROR: {error_msg}\n")
                    
                    # 3. Extract streamflow - do this for BOTH lumped and distributed domains
                    debug_log.write("\n--- Extracting streamflow ---\n")
                    debug_log.write(f"Calling _extract_run_streamflow for {run_name}\n")
                    
                    output_csv = self._extract_run_streamflow(run_dir, run_name)
                    
                    if output_csv and Path(output_csv).exists():
                        debug_log.write(f"Streamflow extraction successful: {output_csv}\n")
                        # Check if the file has actual data
                        try:
                            import pandas as pd
                            df = pd.read_csv(output_csv)
                            if len(df) > 0:
                                debug_log.write(f"Streamflow file contains {len(df)} rows of data\n")
                            else:
                                debug_log.write("WARNING: Streamflow file is empty\n")
                        except Exception as e:
                            debug_log.write(f"Error checking streamflow file: {str(e)}\n")
                    else:
                        debug_log.write("Streamflow extraction failed or produced no file\n")
                        # Create an empty streamflow file if one doesn't exist
                        self._create_empty_streamflow_file(run_dir, run_name)
                    
                    # Record success regardless of MizuRoute (we consider run successful if SUMMA completes)
                    results.append((run_name, True, None))
                    
                except subprocess.CalledProcessError as e:
                    error_msg = f"Error running simulation: {str(e)}"
                    self.logger.error(error_msg)
                    debug_log.write(f"ERROR: {error_msg}\n")
                    results.append((run_name, False, error_msg))
                    
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    self.logger.error(error_msg)
                    debug_log.write(f"ERROR: {error_msg}\n")
                    import traceback
                    tb = traceback.format_exc()
                    debug_log.write(f"Traceback:\n{tb}\n")
                    results.append((run_name, False, error_msg))
                
                finally:
                    # Record completion time
                    debug_log.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    # Create streamflow file if it doesn't exist yet
                    if not (run_dir / "results" / f"{run_name}_streamflow.csv").exists():
                        self._create_empty_streamflow_file(run_dir, run_name)
        
        # Summarize results
        successes = sum(1 for _, success, _ in results if success)
        self.logger.info(f"Completed {successes} out of {len(results)} sequential ensemble simulations")
        
        return results

    def _create_empty_streamflow_file(self, run_dir, run_name):
        """Create an empty streamflow file to ensure existence for later analysis"""
        import pandas as pd
        
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create an empty streamflow file with appropriate columns
        empty_df = pd.DataFrame(columns=['time', 'streamflow'])
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        empty_df.to_csv(output_csv, index=False)
        
        self.logger.warning(f"Created empty streamflow file at {output_csv}")
        return output_csv

    def _setup_run_mizuroute_settings(self, run_dir, run_name):
        """
        Setup MizuRoute settings for a specific run.
        
        This method copies the base MizuRoute settings to the run directory
        and modifies them as needed for the specific run.
        """
        self.logger.info(f"Setting up MizuRoute for {run_name}")
        
        # Source MizuRoute settings directory
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        
        if not source_mizu_dir.exists():
            self.logger.error(f"Source MizuRoute settings directory not found: {source_mizu_dir}")
            raise FileNotFoundError(f"Source MizuRoute settings directory not found: {source_mizu_dir}")
        
        # Destination MizuRoute settings directory
        dest_mizu_dir = run_dir / "settings" / "mizuRoute"
        dest_mizu_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from source to destination
        import shutil
        for file in os.listdir(source_mizu_dir):
            source_file = source_mizu_dir / file
            if source_file.is_file():
                shutil.copy2(source_file, dest_mizu_dir / file)
        
        # Modify the control file for this run
        self._update_mizuroute_control_file(dest_mizu_dir, run_dir, run_name)
        
        self.logger.info(f"MizuRoute settings setup completed for {run_name}")

    def _update_mizuroute_control_file(self, mizu_settings_dir, run_dir, run_name):
        """
        Update the MizuRoute control file for a specific run.
        
        Args:
            mizu_settings_dir: Path to MizuRoute settings directory
            run_dir: Path to the run directory
            run_name: Name of the run (e.g., 'run_0001')
        """
        control_file_path = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not control_file_path.exists():
            self.logger.error(f"MizuRoute control file not found: {control_file_path}")
            raise FileNotFoundError(f"MizuRoute control file not found: {control_file_path}")
        
        # Read the original control file
        with open(control_file_path, 'r') as f:
            control_lines = f.readlines()
        
        # Modify paths in the control file
        modified_lines = []
        for line in control_lines:
            # Update input directory to point to this run's SUMMA output
            if '<input_dir>' in line:
                summa_output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                summa_output_dir.mkdir(parents=True, exist_ok=True)
                modified_line = f"<input_dir>             {summa_output_dir}/    ! Folder that contains runoff data from SUMMA \n"
                modified_lines.append(modified_line)
            # Update ancil_dir to point to this run's mizuRoute settings directory
            elif '<ancil_dir>' in line:
                modified_line = f"<ancil_dir>             {mizu_settings_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n"
                modified_lines.append(modified_line)            
            # Update output directory to point to this run's MizuRoute output
            elif '<output_dir>' in line:
                mizuroute_output_dir = run_dir / "simulations" / self.experiment_id / "mizuRoute" / ""
                mizuroute_output_dir.mkdir(parents=True, exist_ok=True)
                modified_line = f"<output_dir>            {mizuroute_output_dir}/    ! Folder that will contain mizuRoute simulations \n"
                modified_lines.append(modified_line)
            # Update case name to include run identifier
            elif '<case_name>' in line:
                # Extract the experiment ID from the config
                experiment_id = self.config.get('EXPERIMENT_ID')
                # Create a new case name with run identifier
                new_case = f"{experiment_id}_{run_name}"
                modified_line = f"<case_name>             {new_case}    ! Simulation case name with run identifier \n"
                modified_lines.append(modified_line)
            # Update input file name to match SUMMA output naming
            elif '<fname_qsim>' in line:
                # The SUMMA output file will be named like: experiment_id_runID_timestep.nc
                # We need to match this naming convention
                experiment_id = self.config.get('EXPERIMENT_ID')
                run_id_format = run_name.lower()  # Convert to lowercase to match file naming
                modified_line = f"<fname_qsim>            {experiment_id}_timestep.nc    ! netCDF name for HM_HRU runoff \n"
                modified_lines.append(modified_line)
            # Keep simulation start time as is from the main configuration
            elif '<sim_start>' in line:
                sim_start = self.config.get('EXPERIMENT_TIME_START')
                # Ensure proper formatting
                if ' ' not in sim_start:
                    sim_start = f"{sim_start} 00:00"
                modified_line = f"<sim_start>             {sim_start}    ! Time of simulation start. format: yyyy-mm-dd hh:mm:ss \n"
                modified_lines.append(modified_line)
            # Keep simulation end time as is from the main configuration
            elif '<sim_end>' in line:
                sim_end = self.config.get('EXPERIMENT_TIME_END')
                # Ensure proper formatting
                if ' ' not in sim_end:
                    sim_end = f"{sim_end} 23:59"
                modified_line = f"<sim_end>               {sim_end}    ! Time of simulation end. format: yyyy-mm-dd hh:mm:ss \n"
                modified_lines.append(modified_line)
            # Set calendar to standard for compatibility
            elif '<calendar>' in line:
                modified_line = f"<calendar>              standard    ! Calendar of the nc file \n"
                modified_lines.append(modified_line)
            # Ensure routing variable is set correctly
            elif '<vname_qsim>' in line:
                routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
                modified_line = f"<vname_qsim>            {routing_var}    ! Variable name for HM_HRU runoff \n"
                modified_lines.append(modified_line)
            # Set time steps consistently
            elif '<dt_qsim>' in line:
                time_step = self.config.get('SETTINGS_MIZU_ROUTING_DT', '3600')
                modified_line = f"<dt_qsim>               {time_step}    ! Time interval of input runoff in seconds \n"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)  # Keep other lines unchanged
        
        # Write the modified control file
        with open(control_file_path, 'w') as f:
            f.writelines(modified_lines)
        
        self.logger.info(f"Updated MizuRoute control file for {run_name}")

    def _extract_run_streamflow(self, run_dir, run_name):
        """
        Extract streamflow results from model output for a specific run.
        
        For distributed domains, extracts from MizuRoute output.
        For lumped domains, extracts directly from SUMMA timestep output.
        
        Args:
            run_dir: Path to the run directory
            run_name: Name of the run (e.g., 'run_0001')
            
        Returns:
            Path: Path to the extracted streamflow CSV file
        """
        self.logger.info(f"Extracting streamflow results for {run_name}")
        
        import pandas as pd
        import xarray as xr
        import geopandas as gpd
        import numpy as np
        from pathlib import Path
        
        # Create results directory
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output CSV path
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        
        # Check if domain is lumped
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        # Get catchment area for conversion
        catchment_area = self._get_catchment_area()
        if catchment_area is None or catchment_area <= 0:
            self.logger.warning(f"Invalid catchment area: {catchment_area}, using 1.0 km² as default")
            catchment_area = 1.0 * 1e6  # Default 1 km² in m²
        
        self.logger.info(f"Using catchment area of {catchment_area:.2f} m² for flow conversion")
        
        if is_lumped:
            # For lumped domains, extract directly from SUMMA output
            experiment_id = self.experiment_id
            summa_output_dir = run_dir / "simulations" / experiment_id / "SUMMA"
            
            # Look for all possible SUMMA output files
            timestep_files = list(summa_output_dir.glob(f"{experiment_id}*.nc"))
            
            if not timestep_files:
                self.logger.warning(f"No SUMMA output files found for {run_name} in {summa_output_dir}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Use the first timestep file found
            timestep_file = timestep_files[0]
            self.logger.info(f"Using SUMMA output file: {timestep_file}")
            
            try:
                # Open the NetCDF file
                with xr.open_dataset(timestep_file) as ds:
                    # Log available variables to help diagnose issues
                    self.logger.info(f"Available variables in SUMMA output: {list(ds.variables.keys())}")
                    
                    # Check for averageRoutedRunoff or other possible variables
                    runoff_var = None
                    for var_name in ['averageRoutedRunoff', 'outflow', 'basRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            runoff_var = var_name
                            break
                    
                    if runoff_var is None:
                        self.logger.warning(f"No suitable runoff variable found in {timestep_file}")
                        return self._create_empty_streamflow_file(run_dir, run_name)
                    
                    self.logger.info(f"Using {runoff_var} variable from SUMMA output")
                    
                    # Extract the runoff variable
                    runoff_data = ds[runoff_var]
                    
                    # Log the dimensions of the variable
                    self.logger.info(f"Dimensions of {runoff_var}: {runoff_data.dims}")
                    
                    # Convert to pandas series or dataframe
                    if 'gru' in runoff_data.dims:
                        self.logger.info(f"Found 'gru' dimension with size {ds.dims['gru']}")
                        # If multiple GRUs, sum them up
                        if ds.dims['gru'] > 1:
                            self.logger.info(f"Summing runoff across {ds.dims['gru']} GRUs")
                            runoff_series = runoff_data.sum(dim='gru').to_pandas()
                        else:
                            # Single GRU case
                            runoff_series = runoff_data.to_pandas()
                            if isinstance(runoff_series, pd.DataFrame):
                                runoff_series = runoff_series.iloc[:, 0]
                    else:
                        # Handle case without a gru dimension
                        runoff_series = runoff_data.to_pandas()
                        if isinstance(runoff_series, pd.DataFrame):
                            runoff_series = runoff_series.iloc[:, 0]
                    
                    # Convert from m/s to m³/s by multiplying by catchment area
                    streamflow = runoff_series * catchment_area
                    
                    # Log some statistics to debug
                    self.logger.info(f"Streamflow statistics: min={streamflow.min():.2f}, max={streamflow.max():.2f}, mean={streamflow.mean():.2f} m³/s")
                    
                    # Create dataframe and save to CSV
                    result_df = pd.DataFrame({'time': streamflow.index, 'streamflow': streamflow.values})
                    result_df.to_csv(output_csv, index=False)
                    
                    self.logger.info(f"Extracted streamflow from SUMMA output for lumped domain, saved to {output_csv}")
                    return output_csv
                    
            except Exception as e:
                self.logger.error(f"Error extracting streamflow from SUMMA output: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return self._create_empty_streamflow_file(run_dir, run_name)
        else:
            # For distributed domains, extract from MizuRoute output
            experiment_id = self.experiment_id
            mizuroute_output_dir = run_dir / "simulations" / experiment_id / "mizuRoute"
            
            # Try different pattern matching to find output files
            output_files = []
            patterns = [
                f"{experiment_id}_{run_name}*.nc",  # Try with run name in filename
                f"{experiment_id}*.nc",             # Try just experiment ID
                "*.nc"                              # Try any netCDF file as last resort
            ]
            
            for pattern in patterns:
                output_files = list(mizuroute_output_dir.glob(pattern))
                if output_files:
                    self.logger.info(f"Found MizuRoute output files with pattern: {pattern}")
                    break
            
            if not output_files:
                self.logger.warning(f"No MizuRoute output files found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Target reach ID
            sim_reach_id = self.config.get('SIM_REACH_ID')
            
            # Process each output file
            all_streamflow = []
            
            for output_file in output_files:
                try:
                    # Open the NetCDF file
                    with xr.open_dataset(output_file) as ds:
                        # Check if the reach ID exists
                        if 'reachID' in ds.variables:
                            # Find index for the specified reach ID
                            reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                            
                            if len(reach_indices) > 0:
                                # Get the index
                                reach_index = reach_indices[0]
                                
                                # Extract the time series for this reach
                                # Look for common streamflow variable names
                                for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                    if var_name in ds.variables:
                                        streamflow = ds[var_name].isel(seg=reach_index).to_pandas()
                                        
                                        # Reset index to use time as a column
                                        streamflow = streamflow.reset_index()
                                        
                                        # Rename columns for clarity
                                        streamflow = streamflow.rename(columns={var_name: 'streamflow'})
                                        
                                        all_streamflow.append(streamflow)
                                        self.logger.info(f"Extracted streamflow from variable {var_name}")
                                        break
                                
                                self.logger.info(f"Extracted streamflow for reach ID {sim_reach_id} from {output_file.name}")
                            else:
                                self.logger.warning(f"Reach ID {sim_reach_id} not found in {output_file.name}")
                        else:
                            self.logger.warning(f"No reachID variable found in {output_file.name}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {output_file.name}: {str(e)}")
            
            # Combine all streamflow data
            if all_streamflow:
                combined_streamflow = pd.concat(all_streamflow)
                combined_streamflow = combined_streamflow.sort_values('time')
                
                # Remove duplicates if any
                combined_streamflow = combined_streamflow.drop_duplicates(subset='time')
                
                # Save to CSV
                combined_streamflow.to_csv(output_csv, index=False)
                
                self.logger.info(f"Saved combined streamflow results to {output_csv}")
                
                return output_csv
            else:
                self.logger.warning(f"No streamflow data found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)

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
            self.logger.info(f"Opening shapefile for area calculation: {basin_shapefile}")
            gdf = gpd.read_file(basin_shapefile)
            
            # Log the available columns
            self.logger.info(f"Available columns in shapefile: {gdf.columns.tolist()}")
            
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
            return None

    @get_function_logger
    def run_job(self, job_info):
        """
        Run a single simulation job, handling both lumped and distributed domains.
        
        Args:
            job_info: Dictionary containing all information needed for the job
            
        Returns:
            tuple: (run_name, success_flag, error_message)
        """
        run_name = job_info['run_name']
        run_dir = job_info['run_dir']
        is_lumped = job_info.get('is_lumped', False)
        
        # Initialize debug log
        with open(job_info['debug_log_file'], 'w') as debug_log:
            debug_log.write(f"Debug log for {run_name}\n")
            debug_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            debug_log.write(f"SUMMA executable: {job_info['summa_exe']}\n")
            debug_log.write(f"SUMMA fileManager: {job_info['filemanager_path']}\n")
            debug_log.write(f"Domain definition method: {job_info['domain_method']}\n")
            debug_log.write(f"Is lumped domain: {is_lumped}\n\n")
            
            try:
                # 1. Run SUMMA
                self.logger.info(f"Running SUMMA for {run_name}")
                debug_log.write("\n--- Running SUMMA ---\n")
                debug_log.write(f"SUMMA command: {job_info['summa_command']}\n")
                
                import subprocess
                with open(job_info['summa_log_file'], 'w') as f:
                    process = subprocess.run(job_info['summa_command'], shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                
                self.logger.info(f"Successfully completed SUMMA simulation for {run_name}")
                debug_log.write("SUMMA simulation completed successfully\n")
                
                # Check if SUMMA output files were created
                output_dir = run_dir / "simulations" / job_info['experiment_id'] / "SUMMA"
                summa_output_files = list(output_dir.glob(f"{job_info['experiment_id']}*.nc"))
                debug_log.write(f"SUMMA output files found: {len(summa_output_files)}\n")
                for file in summa_output_files:
                    debug_log.write(f"  - {file.name}\n")
                
                # 2. Run MizuRoute if configured and NOT in lumped mode
                run_mizuroute = job_info['run_mizuroute'] and not is_lumped
                
                debug_log.write(f"\nRun MizuRoute? {run_mizuroute}\n")
                
                if run_mizuroute and job_info['mizu_command']:
                    self.logger.info(f"Running MizuRoute for {run_name}")
                    debug_log.write("\n--- Running MizuRoute ---\n")
                    debug_log.write(f"MizuRoute command: {job_info['mizu_command']}\n")
                    
                    # Check if SUMMA timestep file exists
                    timestep_file = output_dir / f"{job_info['experiment_id']}_timestep.nc"
                    debug_log.write(f"Looking for SUMMA timestep file at: {timestep_file}\n")
                    
                    if timestep_file.exists():
                        debug_log.write(f"SUMMA timestep file found: {timestep_file}\n")
                        # Try to get basic info about the file
                        try:
                            import xarray as xr
                            with xr.open_dataset(timestep_file) as ds:
                                debug_log.write(f"Timestep file dimensions: {dict(ds.dims)}\n")
                                debug_log.write(f"Timestep file variables: {list(ds.variables)}\n")
                                if 'time' in ds.dims:
                                    debug_log.write(f"Time dimension size: {ds.dims['time']}\n")
                                    debug_log.write(f"First time: {ds.time.values[0]}\n")
                                    debug_log.write(f"Last time: {ds.time.values[-1]}\n")
                        except Exception as e:
                            debug_log.write(f"Error examining timestep file: {str(e)}\n")
                            
                        try:
                            with open(job_info['mizu_log_file'], 'w') as f:
                                process = subprocess.run(job_info['mizu_command'], shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                            
                            self.logger.info(f"Successfully completed MizuRoute simulation for {run_name}")
                            debug_log.write("MizuRoute simulation completed successfully\n")
                            
                            # Check for MizuRoute output
                            mizu_output_dir = run_dir / "simulations" / job_info['experiment_id'] / "mizuRoute"
                            mizu_files = list(mizu_output_dir.glob("*.nc"))
                            debug_log.write(f"MizuRoute output files found: {len(mizu_files)}\n")
                            for file in mizu_files:
                                debug_log.write(f"  - {file.name}\n")
                        except subprocess.CalledProcessError as e:
                            error_msg = f"MizuRoute execution failed: {str(e)}"
                            self.logger.error(error_msg)
                            debug_log.write(f"ERROR: {error_msg}\n")
                            debug_log.write("Will continue with extracting results from SUMMA\n")
                    else:
                        error_msg = f"SUMMA timestep file not found: {timestep_file}"
                        debug_log.write(f"ERROR: {error_msg}\n")
                        debug_log.write("Will continue with extracting results from any available output\n")
                
                # 3. Extract streamflow - always do this for both lumped and distributed domains
                debug_log.write("\n--- Extracting streamflow ---\n")
                debug_log.write(f"Calling _extract_run_streamflow for {run_name}\n")
                
                output_csv = self._extract_run_streamflow(run_dir, run_name)
                
                if output_csv and Path(output_csv).exists():
                    debug_log.write(f"Streamflow extraction successful: {output_csv}\n")
                    # Check if the file has actual data
                    try:
                        import pandas as pd
                        df = pd.read_csv(output_csv)
                        debug_log.write(f"Streamflow file contains {len(df)} rows of data\n")
                    except Exception as e:
                        debug_log.write(f"Error checking streamflow file: {str(e)}\n")
                else:
                    debug_log.write("Streamflow extraction failed or produced no file\n")
                    # Create empty streamflow file
                    self._create_empty_streamflow_file(run_dir, run_name)
                
                debug_log.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                return (run_name, True, None)
            
            except subprocess.CalledProcessError as e:
                error_msg = f"Command failed for {run_name}: {str(e)}"
                self.logger.error(error_msg)
                debug_log.write(f"ERROR: {error_msg}\n")
                # Create empty streamflow file
                self._create_empty_streamflow_file(run_dir, run_name)
                debug_log.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                return (run_name, False, str(e))
            
            except Exception as e:
                error_msg = f"Unexpected error running {run_name}: {str(e)}"
                self.logger.error(error_msg)
                debug_log.write(f"ERROR: {error_msg}\n")
                import traceback
                tb = traceback.format_exc()
                debug_log.write(f"Traceback:\n{tb}\n")
                # Create empty streamflow file
                self._create_empty_streamflow_file(run_dir, run_name)
                debug_log.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                return (run_name, False, str(e))
        
    def _run_parallel_ensemble(self, summa_exe, run_dirs, max_jobs):
        """Run ensemble simulations in parallel, handling both lumped and distributed domains."""
        self.logger.info(f"Running {len(run_dirs)} ensemble simulations in parallel (max jobs: {max_jobs})")
        
        # Check if we're in lumped domain mode
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        
        try:
            import concurrent.futures
            import subprocess
            
            # Prepare job info for each run
            jobs = []
            for run_dir in run_dirs:
                run_name = run_dir.name
                
                # Get fileManager path for this run
                run_settings_dir = run_dir / "settings" / "SUMMA"
                filemanager_path = run_settings_dir / os.path.basename(self.filemanager_path)
                
                if not filemanager_path.exists():
                    self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                    continue
                
                # Ensure output directory exists
                output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Results directory
                results_dir = run_dir / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                
                # Log directory
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup MizuRoute for this run (even for lumped domains, we'll decide later whether to run it)
                mizu_settings_dir = run_dir / "settings" / "mizuRoute"
                if not mizu_settings_dir.exists() and not is_lumped:
                    try:
                        self._setup_run_mizuroute_settings(run_dir, run_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to set up MizuRoute for {run_name}: {str(e)}")
                
                # Get MizuRoute executable path and control file
                mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
                if mizu_path == 'default':
                    mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
                else:
                    mizu_path = Path(mizu_path)
                mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
                mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                
                # Check if MizuRoute should be configured to run
                # Note: actual running will depend on domain type
                run_mizuroute = ('SUMMA' in self.config.get('HYDROLOGICAL_MODEL', '').split(',') and
                                mizu_control_file.exists())
                
                # SUMMA command
                summa_command = f"{summa_exe} -m {filemanager_path}"
                summa_log_file = log_dir / f"{run_name}_summa.log"
                
                # MizuRoute command
                if run_mizuroute:
                    mizu_command = f"{mizu_exe} {mizu_control_file}"
                    mizu_log_file = log_dir / f"{run_name}_mizuroute.log"
                else:
                    mizu_command = None
                    mizu_log_file = None
                
                # Debug log file
                debug_log_file = results_dir / f"{run_name}_debug.log"
                
                # Add to jobs list with enhanced info
                job_info = {
                    'run_name': run_name,
                    'run_dir': run_dir,
                    'summa_exe': summa_exe,
                    'summa_command': summa_command,
                    'summa_log_file': summa_log_file,
                    'mizu_command': mizu_command,
                    'mizu_log_file': mizu_log_file,
                    'run_mizuroute': run_mizuroute,
                    'debug_log_file': debug_log_file,
                    'filemanager_path': filemanager_path,
                    'is_lumped': is_lumped,
                    'domain_method': domain_method,
                    'experiment_id': self.experiment_id
                }
                jobs.append(job_info)
            
            # Run jobs in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as executor:
                future_to_job = {executor.submit(self.run_job, job): job['run_name'] for job in jobs}
                
                for future in concurrent.futures.as_completed(future_to_job):
                    job_name = future_to_job[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result[1]:  # If success
                            self.logger.info(f"Successfully completed simulation for {job_name}")
                        else:
                            self.logger.warning(f"Failed simulation for {job_name}: {result[2]}")
                    except Exception as e:
                        self.logger.error(f"Exception processing result for {job_name}: {str(e)}")
                        results.append((job_name, False, str(e)))
            
            # Summarize results
            successes = sum(1 for _, success, _ in results if success)
            self.logger.info(f"Completed {successes} out of {len(results)} parallel ensemble simulations")
            
            return results
            
        except ImportError:
            self.logger.warning("concurrent.futures module not available. Falling back to sequential execution.")
            return self._run_sequential_ensemble(summa_exe, run_dirs)
        
        except Exception as e:
            self.logger.error(f"Error in parallel ensemble execution: {str(e)}. Falling back to sequential execution.")
            return self._run_sequential_ensemble(summa_exe, run_dirs)

    @get_function_logger
    def store_parameter_sets(self, local_bounds: Dict, basin_bounds: Dict):
        """
        Store all generated parameter sets in a consolidated file.
        
        This adds a 'run' dimension to the parameter file, allowing all parameter
        sets to be stored in a single NetCDF file for easier analysis and reference.
        
        Args:
            local_bounds: Dictionary of local parameter bounds
            basin_bounds: Dictionary of basin parameter bounds
            
        Returns:
            Path: Path to the consolidated parameter file
        """
        self.logger.info("Storing all parameter sets in a consolidated file")
        
        # Define path for the consolidated parameter file
        consolidated_file_path = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
        
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
        
        # Generate parameter values for all runs at once
        all_param_values = {}
        
        # For local parameters
        for param_name, bounds in local_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            # Shape: [num_samples, num_hru]
            all_param_values[param_name] = np.array([
                self._generate_random_values(min_val, max_val, num_hru, param_name)
                for _ in range(self.num_samples)
            ])
        
        # For basin parameters
        for param_name, bounds in basin_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            # First generate values for GRUs
            gru_values = np.array([
                self._generate_random_values(min_val, max_val, num_gru, param_name)
                for _ in range(self.num_samples)
            ])
            
            # Then map to HRUs
            hru_mapped_values = np.zeros((self.num_samples, num_hru), dtype=np.float64)
            
            for sample_idx in range(self.num_samples):
                for hru_idx in range(num_hru):
                    hru_gru_id = hru2gru_ids[hru_idx]
                    gru_index = gru_id_to_index.get(hru_gru_id)
                    if gru_index is not None:
                        hru_mapped_values[sample_idx, hru_idx] = gru_values[sample_idx, gru_index]
                    else:
                        hru_mapped_values[sample_idx, hru_idx] = 0.0
            
            all_param_values[param_name] = hru_mapped_values
        
        # Create the consolidated NetCDF file
        with nc4.Dataset(consolidated_file_path, 'w', format='NETCDF4') as nc_file:
            # Define dimensions
            nc_file.createDimension('hru', num_hru)
            nc_file.createDimension('run', self.num_samples)
            
            # Create hruId variable
            hru_id_var = nc_file.createVariable('hruId', 'i4', ('hru',))
            hru_id_var[:] = hru_ids
            hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
            hru_id_var.units = '-'
            
            # Create run index variable
            run_var = nc_file.createVariable('runIndex', 'i4', ('run',))
            run_var[:] = np.arange(self.num_samples)
            run_var.long_name = 'Run/Sample Index'
            run_var.units = '-'
            
            # Add parameter variables with 'run' and 'hru' dimensions
            for param_name, values in all_param_values.items():
                param_var = nc_file.createVariable(param_name, 'f8', ('run', 'hru',), fill_value=False)
                param_var[:] = values
                
                # Determine if it's a local or basin parameter
                is_local = param_name in local_bounds
                if is_local:
                    param_var.parameter_type = "local"
                    if param_name in local_bounds:
                        param_var.min_value = local_bounds[param_name]['min']
                        param_var.max_value = local_bounds[param_name]['max']
                else:
                    param_var.parameter_type = "basin"
                    if param_name in basin_bounds:
                        param_var.min_value = basin_bounds[param_name]['min']
                        param_var.max_value = basin_bounds[param_name]['max']
                
                param_var.long_name = f"Parameter values for {param_name} across all runs"
                param_var.units = "N/A"
            
            # Add global attributes
            nc_file.description = "Consolidated parameter sets for ensemble simulations"
            nc_file.history = f"Created on {datetime.now().isoformat()} by CONFLUENCE LargeSampleEmulator"
            nc_file.confluence_experiment_id = self.experiment_id
            nc_file.sampling_method = self.sampling_method
            nc_file.random_seed = self.random_seed
            nc_file.num_samples = self.num_samples
        
        self.logger.info(f"Successfully stored all parameter sets in: {consolidated_file_path}")
        return consolidated_file_path

    @get_function_logger
    def analyze_ensemble_results(self):
        """
        Analyze the results of ensemble simulations, focusing on streamflow.
        
        This method:
        1. Reads streamflow output from each ensemble member
        2. Computes basic performance metrics
        3. Creates a visualization comparing simulated vs. observed streamflow
        4. Saves the visualization and metrics to file
        
        Returns:
            Path: Path to the analysis results directory
        """
        self.logger.info("Starting ensemble results analysis")
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import datetime as dt
        
        # Create analysis directory
        analysis_dir = self.emulator_output_dir / "ensemble_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load observed data
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            obs_path = Path(obs_path)
            
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Load observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and streamflow columns in observed data
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if date_col is None or flow_col is None:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Get all run directories
            run_dirs = sorted(self.ensemble_dir.glob("run_*"))
            if not run_dirs:
                self.logger.error("No ensemble run directories found.")
                return None
            
            # Process each run directory to extract results
            metrics = {}
            all_flows = pd.DataFrame(index=obs_df.index)
            all_flows['Observed'] = observed_flow
            
            # Print debug info on the first few observed flow values
            self.logger.debug(f"First 5 observed flow values: {observed_flow.head()}")
            
            # Loop through each run directory
            for run_dir in run_dirs:
                run_id = run_dir.name
                
                # Look for streamflow output file
                results_file = run_dir / "results" / f"{run_id}_streamflow.csv"
                
                if not results_file.exists():
                    # Try to extract from the source files if results don't exist
                    results_file = self._extract_run_streamflow(run_dir, run_id)
                    if not results_file or not Path(results_file).exists():
                        self.logger.warning(f"No streamflow results found for {run_id}")
                        continue
                
                # Read the CSV file
                try:
                    run_df = pd.read_csv(results_file)
                    
                    # Check if file has data
                    if len(run_df) == 0:
                        self.logger.warning(f"Empty streamflow file for {run_id}")
                        continue
                    
                    # Get time column
                    time_col = next((col for col in run_df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                    if time_col is None:
                        self.logger.warning(f"No time column found in {run_id} results")
                        continue
                    
                    # Get flow column
                    flow_col_sim = next((col for col in run_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'streamflow' in col.lower()), None)
                    if flow_col_sim is None:
                        self.logger.warning(f"No flow column found in {run_id} results. Columns: {run_df.columns.tolist()}")
                        continue
                    
                    # Print debug info on first few values
                    self.logger.debug(f"{run_id} first 5 flow values: {run_df[flow_col_sim].head()}")
                    self.logger.debug(f"{run_id} flow stats: mean={run_df[flow_col_sim].mean():.4f}, min={run_df[flow_col_sim].min():.4f}, max={run_df[flow_col_sim].max():.4f}")
                    
                    # Convert to datetime and set as index
                    run_df['DateTime'] = pd.to_datetime(run_df[time_col])
                    run_df.set_index('DateTime', inplace=True)
                    
                    # Add to all_flows dataframe for visualization
                    # Use reindex to align with observed data timepoints
                    all_flows[run_id] = run_df[flow_col_sim].reindex(all_flows.index, method='nearest')
                    
                    # Calculate performance metrics against observed data
                    # Find common time period
                    common_idx = obs_df.index.intersection(run_df.index)
                    if len(common_idx) < 5:  # Need at least a few points for metrics
                        self.logger.warning(f"Insufficient common time period between {run_id} and observed data")
                        continue
                    
                    # Extract aligned data for metric calculation
                    obs_aligned = observed_flow.loc[common_idx]
                    sim_aligned = run_df[flow_col_sim].loc[common_idx]
                    
                    # Calculate metrics - just KGE for simplicity
                    mean_obs = obs_aligned.mean()
                    
                    # Correlation coefficient
                    try:
                        r = np.corrcoef(sim_aligned, obs_aligned)[0, 1]
                    except:
                        r = np.nan
                        
                    # Relative variability
                    alpha = sim_aligned.std() / obs_aligned.std() if obs_aligned.std() != 0 else np.nan
                    
                    # Bias ratio
                    beta = sim_aligned.mean() / mean_obs if mean_obs != 0 else np.nan
                    
                    # KGE
                    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
                    
                    # Root Mean Square Error (RMSE)
                    rmse = np.sqrt(((obs_aligned - sim_aligned) ** 2).mean())
                    
                    # Nash-Sutcliffe Efficiency (NSE)
                    numerator = ((obs_aligned - sim_aligned) ** 2).sum()
                    denominator = ((obs_aligned - mean_obs) ** 2).sum()
                    nse = 1 - (numerator / denominator) if denominator > 0 else np.nan
                    
                    # Percent Bias (PBIAS)
                    pbias = 100 * (sim_aligned.sum() - obs_aligned.sum()) / obs_aligned.sum() if obs_aligned.sum() != 0 else np.nan
                    
                    metrics[run_id] = {
                        'KGE': kge,
                        'NSE': nse,
                        'RMSE': rmse,
                        'PBIAS': pbias
                    }
                    
                    self.logger.info(f"Processed {run_id}: KGE={kge:.4f}, NSE={nse:.4f}, RMSE={rmse:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing results for {run_id}: {str(e)}")
                    import traceback
                    self.logger.warning(traceback.format_exc())
            
            # Check if we found any valid results
            if len(all_flows.columns) <= 1:  # Only the 'Observed' column
                self.logger.error("No valid streamflow data found in any ensemble run")
                return None
            
            # Convert metrics to DataFrame and save
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            metrics_df.to_csv(analysis_dir / "performance_metrics.csv")
            self.logger.info(f"Saved performance metrics for {len(metrics)} runs")
            
            # Create main visualization plot
            plt.figure(figsize=(16, 8))
            
            # Plot individual simulations with low opacity
            for col in all_flows.columns:
                if col != 'Observed':
                    plt.plot(all_flows.index, all_flows[col], 'b-', alpha=0.1, linewidth=0.5)
            
            # Calculate ensemble mean
            simulation_cols = [col for col in all_flows.columns if col != 'Observed']
            if simulation_cols:
                ensemble_mean = all_flows[simulation_cols].mean(axis=1)
                plt.plot(all_flows.index, ensemble_mean, 'r-', linewidth=2, label='Ensemble Mean')
            
            # Plot observed data
            plt.plot(all_flows.index, all_flows['Observed'], 'g-', linewidth=2, label='Observed')
            
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title(f'Ensemble Streamflow Simulations ({len(metrics)} Runs)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save the plot
            plt.savefig(analysis_dir / "ensemble_streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Ensemble analysis completed. Results saved to {analysis_dir}")
            return analysis_dir
                
        except Exception as e:
            self.logger.error(f"Error during ensemble results analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None