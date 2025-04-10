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
        Can run in sequential or parallel mode depending on configuration.
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
        
        # Check if we should run in parallel
        run_parallel = self.config.get('EMULATION_PARALLEL_ENSEMBLE', False)
        max_parallel_jobs = self.config.get('EMULATION_MAX_PARALLEL_JOBS', 10)
        
        # Get all run directories
        run_dirs = sorted(self.ensemble_dir.glob("run_*"))
        
        if run_parallel and len(run_dirs) > 1:
            # Run in parallel mode
            return self._run_parallel_ensemble(summa_exe, run_dirs, max_parallel_jobs)
        else:
            # Run in sequential mode
            return self._run_sequential_ensemble(summa_exe, run_dirs)

    def _run_sequential_ensemble(self, summa_exe, run_dirs):
        """Run ensemble simulations one after another."""
        self.logger.info(f"Running {len(run_dirs)} ensemble simulations sequentially")
        
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
        self.logger.info(f"Completed {successes} out of {len(results)} sequential ensemble simulations")
        
        return results

    def _run_parallel_ensemble(self, summa_exe, run_dirs, max_jobs):
        """Run ensemble simulations in parallel."""
        self.logger.info(f"Running {len(run_dirs)} ensemble simulations in parallel (max jobs: {max_jobs})")
        
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
                
                # Log directory
                log_dir = output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Run SUMMA command
                command = f"{summa_exe} -m {filemanager_path}"
                log_file = log_dir / f"{run_name}_summa.log"
                
                jobs.append((run_name, command, log_file))
            
            # Function to run a single job
            def run_job(job_info):
                run_name, command, log_file = job_info
                try:
                    with open(log_file, 'w') as f:
                        process = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                    return (run_name, True, None)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Error running SUMMA for {run_name}: {str(e)}")
                    return (run_name, False, str(e))
                except Exception as e:
                    self.logger.error(f"Unexpected error running {run_name}: {str(e)}")
                    return (run_name, False, str(e))
            
            # Run jobs in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as executor:
                future_to_job = {executor.submit(run_job, job): job[0] for job in jobs}
                
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
        Analyze the results of ensemble simulations.
        
        This method:
        1. Reads output from each ensemble member
        2. Computes statistics (mean, std dev, min, max, percentiles)
        3. Creates visualizations
        4. Generates a summary report
        
        Returns:
            Path: Path to the analysis results directory
        """
        self.logger.info("Starting ensemble results analysis")
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import datetime as dt
        import os
        import glob
        
        # Create analysis directory
        analysis_dir = self.emulator_output_dir / "ensemble_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all run directories
        run_dirs = sorted(self.ensemble_dir.glob("run_*"))
        if not run_dirs:
            self.logger.error("No ensemble run directories found.")
            return None
        
        # Collect streamflow outputs from all runs
        streamflow_data = {}
        run_ids = []
        
        for run_dir in run_dirs:
            run_id = run_dir.name
            run_ids.append(run_id)
            
            # Find SUMMA/MizuRoute output files
            summa_output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            netcdf_files = list(summa_output_dir.glob("*.nc"))
            
            if not netcdf_files:
                self.logger.warning(f"No NetCDF output files found for {run_id}")
                continue
            
            # Try to find streamflow data in the output files
            streamflow_found = False
            
            for nc_file in netcdf_files:
                try:
                    with nc4.Dataset(nc_file, 'r') as ds:
                        # Check for common streamflow variable names
                        streamflow_vars = ['IRFroutedRunoff', 'averageRoutedRunoff', 'scalarTotalRunoff', 'discharge']
                        var_name = None
                        
                        for sv in streamflow_vars:
                            if sv in ds.variables:
                                var_name = sv
                                break
                        
                        if var_name is None:
                            # Try to find any variable that might be streamflow
                            for var in ds.variables:
                                if 'runoff' in var.lower() or 'discharge' in var.lower() or 'flow' in var.lower():
                                    var_name = var
                                    break
                        
                        if var_name is not None:
                            # Get time and streamflow data
                            time_var = ds.variables['time']
                            streamflow_var = ds.variables[var_name]
                            
                            # Convert time to datetime
                            if hasattr(time_var, 'units'):
                                time_units = time_var.units
                                try:
                                    import netCDF4 as nc4
                                    dates = nc4.num2date(time_var[:], time_units)
                                except:
                                    # If netCDF4 date conversion fails, try a simple approach
                                    if 'days since' in time_units:
                                        base_date_str = time_units.split('days since ')[1].split()[0]
                                        base_date = dt.datetime.strptime(base_date_str, '%Y-%m-%d')
                                        dates = [base_date + dt.timedelta(days=float(d)) for d in time_var[:]]
                                    elif 'hours since' in time_units:
                                        base_date_str = time_units.split('hours since ')[1].split()[0]
                                        base_date = dt.datetime.strptime(base_date_str, '%Y-%m-%d')
                                        dates = [base_date + dt.timedelta(hours=float(d)) for d in time_var[:]]
                                    else:
                                        # Try to use time values as is (may need additional processing)
                                        dates = [dt.datetime.fromtimestamp(t) for t in time_var[:]]
                            else:
                                # If no time units, use sequential days starting from a default date
                                dates = [dt.datetime(2000, 1, 1) + dt.timedelta(days=i) for i in range(len(time_var[:]))]
                            
                            # Extract streamflow data
                            if len(streamflow_var.shape) == 1:
                                # Single time series
                                flow_data = streamflow_var[:]
                            elif len(streamflow_var.shape) == 2:
                                # Multiple locations - use first one (typically for the outlet)
                                flow_data = streamflow_var[:, 0]
                            else:
                                # More complex structure - try to find the appropriate dimension
                                self.logger.warning(f"Complex variable structure for {var_name} in {nc_file}")
                                flow_data = streamflow_var[:].flatten()
                            
                            # Create DataFrame for this run
                            df = pd.DataFrame({'DateTime': dates, 'Streamflow': flow_data})
                            df.set_index('DateTime', inplace=True)
                            
                            # Store in dictionary
                            streamflow_data[run_id] = df
                            streamflow_found = True
                            break
                
                except Exception as e:
                    self.logger.warning(f"Error processing {nc_file} for {run_id}: {str(e)}")
                    continue
            
            if not streamflow_found:
                self.logger.warning(f"No streamflow data found in any output file for {run_id}")
        
        # Check if we found any streamflow data
        if not streamflow_data:
            self.logger.error("No streamflow data found in any ensemble run.")
            return None
        
        self.logger.info(f"Found streamflow data for {len(streamflow_data)} out of {len(run_dirs)} runs")
        
        # Combine all streamflow data into a single DataFrame
        all_flows = pd.DataFrame()
        for run_id, df in streamflow_data.items():
            # Ensure consistent time steps by resampling if needed
            if all_flows.empty:
                all_flows = df.copy()
                all_flows.rename(columns={'Streamflow': run_id}, inplace=True)
            else:
                # Resample to match index if needed
                if not df.index.equals(all_flows.index):
                    try:
                        # Try daily resampling first
                        df_resampled = df.resample('D').mean()
                        all_flows_resampled = all_flows.resample('D').mean()
                        common_index = df_resampled.index.intersection(all_flows_resampled.index)
                        
                        if not common_index.empty:
                            # Use the common dates
                            all_flows = all_flows_resampled.loc[common_index]
                            df = df_resampled.loc[common_index]
                        else:
                            # If no common dates, try interpolation
                            self.logger.warning(f"No common dates found for {run_id}, trying interpolation")
                            df = df.reindex(all_flows.index, method='nearest')
                    except Exception as e:
                        self.logger.warning(f"Error resampling data for {run_id}: {str(e)}")
                        continue
                
                # Add this run's data
                all_flows[run_id] = df['Streamflow']
        
        # Save the combined flow data
        all_flows.to_csv(analysis_dir / "all_ensemble_streamflows.csv")
        
        # Calculate statistics
        flow_mean = all_flows.mean(axis=1)
        flow_std = all_flows.std(axis=1)
        flow_min = all_flows.min(axis=1)
        flow_max = all_flows.max(axis=1)
        flow_median = all_flows.median(axis=1)
        flow_q05 = all_flows.quantile(0.05, axis=1)
        flow_q25 = all_flows.quantile(0.25, axis=1)
        flow_q75 = all_flows.quantile(0.75, axis=1)
        flow_q95 = all_flows.quantile(0.95, axis=1)
        
        # Create a stats DataFrame
        stats_df = pd.DataFrame({
            'Mean': flow_mean,
            'Std Dev': flow_std,
            'Min': flow_min,
            'Max': flow_max,
            'Median': flow_median,
            'Q05': flow_q05,
            'Q25': flow_q25,
            'Q75': flow_q75,
            'Q95': flow_q95
        })
        
        # Save statistics
        stats_df.to_csv(analysis_dir / "ensemble_statistics.csv")
        
        # Create visualization
        plt.figure(figsize=(16, 10))
        
        # Plot the ensemble range (min to max)
        plt.fill_between(flow_mean.index, flow_min, flow_max, alpha=0.2, color='lightblue', label='Min-Max Range')
        
        # Plot the interquartile range (25th to 75th percentile)
        plt.fill_between(flow_mean.index, flow_q25, flow_q75, alpha=0.4, color='blue', label='Interquartile Range')
        
        # Plot the 90% confidence interval (5th to 95th percentile)
        plt.fill_between(flow_mean.index, flow_q05, flow_q95, alpha=0.3, color='skyblue', label='90% Confidence Interval')
        
        # Plot the mean and median
        plt.plot(flow_mean.index, flow_mean, 'r-', linewidth=2, label='Mean')
        plt.plot(flow_median.index, flow_median, 'k--', linewidth=1.5, label='Median')
        
        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Streamflow')
        plt.title(f'Ensemble Streamflow Statistics ({len(streamflow_data)} Runs)')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.savefig(analysis_dir / "ensemble_streamflow_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary report
        with open(analysis_dir / "ensemble_analysis_report.txt", 'w') as f:
            f.write(f"Ensemble Analysis Report\n")
            f.write(f"=======================\n\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Ensemble Size: {len(run_dirs)}\n")
            f.write(f"Successful Runs: {len(streamflow_data)}\n\n")
            
            f.write(f"Time Period: {all_flows.index.min()} to {all_flows.index.max()}\n")
            f.write(f"Number of Time Steps: {len(all_flows)}\n\n")
            
            f.write(f"Streamflow Statistics (over all runs):\n")
            f.write(f"  Overall Mean: {flow_mean.mean():.2f}\n")
            f.write(f"  Overall Median: {flow_median.mean():.2f}\n")
            f.write(f"  Maximum Value: {flow_max.max():.2f}\n")
            f.write(f"  Minimum Value: {flow_min.min():.2f}\n")
            f.write(f"  Average Std Dev: {flow_std.mean():.2f}\n\n")
            
            f.write(f"Files Generated:\n")
            f.write(f"  - all_ensemble_streamflows.csv: Combined streamflow time series for all runs\n")
            f.write(f"  - ensemble_statistics.csv: Statistical metrics for each time step\n")
            f.write(f"  - ensemble_streamflow_statistics.png: Visualization of ensemble statistics\n\n")
            
            f.write(f"Generated on: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self.logger.info(f"Ensemble analysis completed. Results saved to {analysis_dir}")
        return analysis_dir