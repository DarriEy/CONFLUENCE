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

    def run_emulation_setup(self):
        """Orchestrates the setup for large sample emulation with multiple runs."""
        self.logger.info("Starting Large Sample Emulation setup")
        try:
            # STEP 1: Run a preliminary SUMMA simulation to extract current parameter values
            extracted_params = self.run_parameter_extraction_run()
            
            # STEP 2: Parse parameter bounds from parameter info files
            local_bounds = self._parse_param_info(self.local_param_info_path, self.local_params_to_emulate)
            basin_bounds = self._parse_param_info(self.basin_param_info_path, self.basin_params_to_emulate)

            if not local_bounds and not basin_bounds:
                self.logger.warning("No parameters specified or found for emulation. Skipping trial parameter file generation.")
                return None # Indicate nothing was generated

            # STEP 3: Store all parameter sets in a consolidated file
            consolidated_file = self.store_parameter_sets(local_bounds, basin_bounds, extracted_params)
            self.logger.info(f"All parameter sets stored in consolidated file: {consolidated_file}")

            # STEP 4: Generate the summary file with all parameter sets
            self.generate_parameter_summary(local_bounds, basin_bounds, extracted_params)
            
            # STEP 5: Create individual run directories if requested
            if self.create_individual_runs:
                self.create_ensemble_run_directories(local_bounds, basin_bounds, extracted_params)
                self.logger.info(f"Successfully created {self.num_samples} ensemble run directories in {self.ensemble_dir}")
            
            self.logger.info(f"Large sample emulation setup completed successfully")
            return consolidated_file

        except FileNotFoundError as e:
            self.logger.error(f"Required file not found during emulation setup: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during large sample emulation setup: {str(e)}")
            raise

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

    def generate_parameter_summary(self, local_bounds: Dict, basin_bounds: Dict, extracted_params=None):
        """
        Generates a summary file with all parameter sets.
        
        Args:
            local_bounds: Dictionary of local parameter bounds
            basin_bounds: Dictionary of basin parameter bounds
            extracted_params: Optional dictionary with parameter names as keys and extracted values as values
        
        Returns:
            Tuple: Path to summary file and dictionary of parameter values
        """
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

        # Dictionary to hold all parameter values across all samples
        all_param_values = {}
        
        # Generate all parameter values for all samples upfront
        for param_name, bounds in local_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            
            # Check if we have extracted values for this parameter
            extracted_values = None
            if extracted_params is not None and param_name in extracted_params:
                extracted_values = extracted_params[param_name]
                
                # Convert to numpy array if it's not already
                if not isinstance(extracted_values, np.ndarray):
                    extracted_values = np.array(extracted_values)
                
                # Ensure it's a 1D array to avoid inhomogeneous shape errors
                extracted_values = np.atleast_1d(extracted_values.flatten())
                
                self.logger.info(f"Using extracted values for {param_name} with shape {extracted_values.shape}")
            
            # Create an array to hold all parameter values for this parameter
            # Shape: [num_samples, num_hru]
            param_values = np.zeros((self.num_samples, num_hru), dtype=np.float64)
            
            for i in range(self.num_samples):
                if i == 0 and extracted_values is not None:
                    # Use extracted values directly for the first sample
                    if len(extracted_values) != num_hru:
                        self.logger.warning(f"Extracted values for {param_name} have length {len(extracted_values)}, expected {num_hru}. Will use the values but they may be misaligned.")
                    
                    # Ensure we have the right number of values
                    if len(extracted_values) >= num_hru:
                        # Use the first num_hru values
                        param_values[i, :] = extracted_values[:num_hru]
                    else:
                        # Repeat the values to fill num_hru
                        repeats = int(np.ceil(num_hru / len(extracted_values)))
                        param_values[i, :] = np.tile(extracted_values, repeats)[:num_hru]
                        
                    self.logger.info(f"Using extracted values for {param_name} in first sample")
                else:
                    # Generate random values for other samples
                    sample_values = self._generate_random_values(
                        min_val, max_val, num_hru, param_name, 
                        extracted_values
                    )
                    
                    # Ensure sample_values is the right shape
                    if isinstance(sample_values, np.ndarray) and len(sample_values) == num_hru:
                        param_values[i, :] = sample_values
                    else:
                        self.logger.warning(f"Generated values for {param_name} have unexpected shape, using uniform sampling instead")
                        param_values[i, :] = np.random.uniform(min_val, max_val, num_hru)
            
            all_param_values[param_name] = param_values
        
        # Similarly for basin parameters
        for param_name, bounds in basin_bounds.items():
            min_val, max_val = bounds['min'], bounds['max']
            
            # Check if we have extracted values for this parameter
            extracted_values = None
            if extracted_params is not None and param_name in extracted_params:
                extracted_values = extracted_params[param_name]
                
                # Convert to numpy array if it's not already
                if not isinstance(extracted_values, np.ndarray):
                    extracted_values = np.array(extracted_values)
                
                # Ensure it's a 1D array
                extracted_values = np.atleast_1d(extracted_values.flatten())
                
                self.logger.info(f"Using extracted values for basin parameter {param_name} with shape {extracted_values.shape}")
            
            # Generate GRU values for all samples
            # Shape: [num_samples, num_gru]
            gru_values = np.zeros((self.num_samples, num_gru), dtype=np.float64)
            
            for i in range(self.num_samples):
                if i == 0 and extracted_values is not None:
                    # Use extracted values directly for the first sample
                    if len(extracted_values) != num_gru:
                        self.logger.warning(f"Extracted values for {param_name} have length {len(extracted_values)}, expected {num_gru}. Will use the values but they may be misaligned.")
                    
                    # Ensure we have the right number of values
                    if len(extracted_values) >= num_gru:
                        # Use the first num_gru values
                        gru_values[i, :] = extracted_values[:num_gru]
                    else:
                        # Repeat the values to fill num_gru
                        repeats = int(np.ceil(num_gru / len(extracted_values)))
                        gru_values[i, :] = np.tile(extracted_values, repeats)[:num_gru]
                        
                    self.logger.info(f"Using extracted values for {param_name} in first sample")
                else:
                    # Generate random values for other samples
                    sample_values = self._generate_random_values(
                        min_val, max_val, num_gru, param_name, 
                        extracted_values
                    )
                    
                    # Ensure sample_values is the right shape
                    if isinstance(sample_values, np.ndarray) and len(sample_values) == num_gru:
                        gru_values[i, :] = sample_values
                    else:
                        self.logger.warning(f"Generated values for {param_name} have unexpected shape, using uniform sampling instead")
                        gru_values[i, :] = np.random.uniform(min_val, max_val, num_gru)
            
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
                
                # Add bounds as attributes if available
                if is_local and param_name in local_bounds:
                    param_var.min_value = local_bounds[param_name]['min']
                    param_var.max_value = local_bounds[param_name]['max']
                elif not is_local and param_name in basin_bounds:
                    param_var.min_value = basin_bounds[param_name]['min']
                    param_var.max_value = basin_bounds[param_name]['max']

            # Add global attributes
            summary_ds.description = "SUMMA Trial Parameter summary file for multiple ensemble runs"
            summary_ds.history = f"Created on {datetime.now().isoformat()} by CONFLUENCE LargeSampleEmulator"
            summary_ds.confluence_experiment_id = self.experiment_id
            summary_ds.sampling_method = self.sampling_method
            summary_ds.random_seed = self.random_seed
            summary_ds.num_samples = self.num_samples
            
            if extracted_params is not None:
                summary_ds.using_extracted_parameters = "true"
                summary_ds.extraction_date = datetime.now().isoformat()

        self.logger.info(f"Successfully generated parameter summary file: {self.summary_file_path}")
        return self.summary_file_path, all_param_values

    def create_ensemble_run_directories(self, local_bounds, basin_bounds, extracted_params=None):
        """
        Creates individual run directories with unique parameter sets.
        
        This allows for running multiple model instances with different parameter sets.
        
        Args:
            local_bounds: Dictionary of local parameter bounds
            basin_bounds: Dictionary of basin parameter bounds
            extracted_params: Optional dictionary of extracted parameter values
        """
        self.logger.info(f"Creating {self.num_samples} ensemble run directories in {self.ensemble_dir}")
        
        # First, generate all parameter values using the summary function
        summary_file, all_param_values = self.generate_parameter_summary(local_bounds, basin_bounds, extracted_params)
        
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

    def _generate_random_values(self, min_val, max_val, num_values, param_name, extracted_values=None):
        """
        Generate random parameter values using the specified sampling method.
        
        Args:
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            num_values: Number of values to generate
            param_name: Name of parameter (for logging)
            extracted_values: Optional array of extracted parameter values to use as baseline
                
        Returns:
            np.ndarray: Array of random values within the specified bounds
        """
        try:
            # If we have extracted values and this is the first sample, use them directly
            if extracted_values is not None and len(extracted_values) == num_values:
                return extracted_values.astype(np.float64)
            
            # Determine central tendency for parameter distribution
            if extracted_values is not None:
                # Use the median of extracted values if available, but replicate to match num_values
                if len(extracted_values) == 1 and num_values > 1:
                    # Single value for all HRUs/GRUs
                    center_val = extracted_values[0]
                    center_values = np.full(num_values, center_val)
                elif len(extracted_values) == num_values:
                    # Use extracted values directly as centers
                    center_values = extracted_values
                else:
                    # Mismatch in array sizes - use mean of extracted values
                    center_val = np.mean(extracted_values)
                    center_values = np.full(num_values, center_val)
                    self.logger.warning(f"Mismatch in array sizes for {param_name}. Using mean value {center_val} for all elements.")
            else:
                # No extracted values - use the midpoint of min/max range
                center_val = (min_val + max_val) / 2
                center_values = np.full(num_values, center_val)
            
            # Now generate values around these centers
            if self.sampling_method == 'uniform':
                # For uniform, we'll sample within the full min-max range
                return np.random.uniform(min_val, max_val, num_values).astype(np.float64)
                    
            elif self.sampling_method == 'lhs':
                # Latin Hypercube Sampling with parameter-specific center points
                try:
                    from scipy.stats import qmc
                    
                    # Create a normalized LHS sampler in [0, 1]
                    sampler = qmc.LatinHypercube(d=1, seed=self.random_seed)
                    samples_unit = sampler.random(n=num_values)
                    
                    # Adjust sampling to allow for clustering near the extracted/center values
                    # For each HRU/GRU, scale sample to match its specific center value
                    scaled_samples = np.zeros(num_values, dtype=np.float64)
                    
                    for i in range(num_values):
                        # Get the center value for this location
                        center = center_values[i]
                        
                        # Determine which half of the range we're in
                        if samples_unit[i] < 0.5:
                            # Sample from min_val to center
                            sample_range = center - min_val
                            # Rescale from [0, 0.5] to [0, 1]
                            unit_sample = samples_unit[i] * 2
                            # Generate value
                            scaled_samples[i] = min_val + unit_sample * sample_range
                        else:
                            # Sample from center to max_val
                            sample_range = max_val - center
                            # Rescale from [0.5, 1] to [0, 1]
                            unit_sample = (samples_unit[i] - 0.5) * 2
                            # Generate value
                            scaled_samples[i] = center + unit_sample * sample_range
                    
                    return scaled_samples.astype(np.float64)
                    
                except ImportError:
                    self.logger.warning(f"Could not import scipy.stats.qmc for Latin Hypercube Sampling. Falling back to normal sampling for {param_name}.")
                    # Fall back to normal distribution around center values with bounds
                    return self._generate_normal_bounded_values(min_val, max_val, center_values, num_values)
                    
            elif self.sampling_method == 'sobol':
                # Sobol sequence with parameter-specific center points
                try:
                    from scipy.stats import qmc
                    
                    # Create a Sobol sequence generator
                    sampler = qmc.Sobol(d=1, scramble=True, seed=self.random_seed)
                    samples_unit = sampler.random(n=num_values)
                    
                    # Adjust sampling to allow for clustering near the extracted/center values
                    # Similar approach as with LHS
                    scaled_samples = np.zeros(num_values, dtype=np.float64)
                    
                    for i in range(num_values):
                        center = center_values[i]
                        
                        if samples_unit[i] < 0.5:
                            sample_range = center - min_val
                            unit_sample = samples_unit[i] * 2
                            scaled_samples[i] = min_val + unit_sample * sample_range
                        else:
                            sample_range = max_val - center
                            unit_sample = (samples_unit[i] - 0.5) * 2
                            scaled_samples[i] = center + unit_sample * sample_range
                    
                    return scaled_samples.astype(np.float64)
                    
                except ImportError:
                    self.logger.warning(f"Could not import scipy.stats.qmc for Sobol sequence. Falling back to normal sampling for {param_name}.")
                    # Fall back to normal distribution around center values with bounds
                    return self._generate_normal_bounded_values(min_val, max_val, center_values, num_values)
            
            else:
                self.logger.warning(f"Unknown sampling method '{self.sampling_method}'. Falling back to normal sampling for {param_name}.")
                # Fall back to normal distribution around center values with bounds
                return self._generate_normal_bounded_values(min_val, max_val, center_values, num_values)
                
        except Exception as e:
            self.logger.error(f"Error generating random values for {param_name}: {str(e)}")
            self.logger.warning(f"Using uniform sampling as fallback for {param_name}.")
            return np.random.uniform(min_val, max_val, num_values).astype(np.float64)

    def _generate_normal_bounded_values(self, min_val, max_val, center_values, num_values):
        """
        Generate values from a normal distribution centered around each value in center_values,
        but bounded within min_val and max_val.
        
        Args:
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            center_values: Array of center values for the distribution
            num_values: Number of values to generate
            
        Returns:
            np.ndarray: Array of bounded normally distributed values
        """
        range_size = max_val - min_val
        # Use 1/6th of range size as standard deviation
        std_dev = range_size / 6
        
        # Generate values from normal distribution
        values = np.random.normal(center_values, std_dev, size=num_values)
        
        # Bound values within min_val and max_val
        values = np.clip(values, min_val, max_val)
        
        return values.astype(np.float64)
        
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

    def _create_parameter_output_control(self, settings_dir):
        """
        Create a modified outputControl.txt that outputs the calibration parameters.
        
        Args:
            settings_dir: Path to the settings directory for parameter extraction
        """
        # Read the original outputControl.txt
        orig_output_control = self.settings_path / 'outputControl.txt'
        
        if not orig_output_control.exists():
            self.logger.error(f"Original outputControl.txt not found: {orig_output_control}")
            raise FileNotFoundError(f"Original outputControl.txt not found: {orig_output_control}")
        
        # Read the original file content
        with open(orig_output_control, 'r') as f:
            lines = f.readlines()
        
        # Get all parameters to calibrate
        all_params = self.local_params_to_emulate + self.basin_params_to_emulate
        
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
        orig_file_manager = self.settings_path / 'fileManager.txt'
        
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
                    import datetime
                    try:
                        start_parts = start_time_str.split()
                        start_date = start_parts[0]
                        
                        # Use the same start date but run for 1 day only
                        # This keeps the same start_time which is important for initialization
                        f.write(f"simEndTime           '{start_date} 23:00'\n")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error parsing start time '{start_time_str}': {str(e)}")
                        # If there's an error parsing, keep the original end time
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
            import subprocess
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
        all_params = self.local_params_to_emulate + self.basin_params_to_emulate
        
        try:
            import xarray as xr
            
            # Open the file
            with xr.open_dataset(timestep_file) as ds:
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
                    # Extract parameter values - handle different dimensions
                    var = ds[param]
                    
                    # The parameter might be per HRU or per GRU
                    # We need to identify the right dimension and extract
                    if 'hru' in var.dims:
                        # Parameter per HRU
                        param_values[param] = var.values
                        self.logger.debug(f"Extracted {param} values for {len(param_values[param])} HRUs")
                    elif 'gru' in var.dims:
                        # Parameter per GRU - need to map to HRUs
                        param_values[param] = var.values
                        self.logger.debug(f"Extracted {param} values for {len(param_values[param])} GRUs")
                    else:
                        # Parameter is scalar or has unexpected dimensions
                        self.logger.warning(f"Parameter {param} has unexpected dimensions: {var.dims}")
                        continue
                
                self.logger.info(f"Successfully extracted values for {len(param_values)} parameters")
                return param_values
                
        except Exception as e:
            self.logger.error(f"Error extracting parameters from results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def run_parameter_extraction_run(self):
        """
        Sets up and runs a preliminary SUMMA simulation to extract parameter values.
        
        This will:
        1. Create a modified outputControl.txt that outputs the calibration parameters
        2. Set up a short simulation using the existing configuration
        3. Extract the parameter values from the resulting timestep.nc file
        4. Return the extracted parameter values
        
        Returns:
            dict: Dictionary with parameter names as keys and arrays of values as values
        """
        self.logger.info("Setting up preliminary parameter extraction run")
        
        # Create a directory for the parameter extraction run
        extract_dir = self.emulator_output_dir / "param_extraction"
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
                        self.logger.info(f"Found 'gru' dimension with size {ds.sizes['gru']}")
                        # If multiple GRUs, sum them up
                        if ds.sizes['gru'] > 1:
                            self.logger.info(f"Summing runoff across {ds.sizes['gru']} GRUs")
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

    def store_parameter_sets(self, local_bounds, basin_bounds, extracted_params=None):
        """
        Store all generated parameter sets in a consolidated file.
        
        Args:
            local_bounds: Dictionary of local parameter bounds
            basin_bounds: Dictionary of basin parameter bounds
            extracted_params: Optional dictionary of extracted parameter values
                
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

    def analyze_ensemble_results(self):
        """
        Analyze the results of ensemble simulations, focusing on streamflow.
        
        This method:
        1. Reads streamflow output from each ensemble member
        2. Computes basic performance metrics for training and test periods separately
        3. Creates visualizations comparing simulated vs. observed streamflow
        4. Saves the visualization and metrics to file
        
        Returns:
            Path: Path to the analysis results directory
        """
        self.logger.info("Starting ensemble results analysis")
        
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import matplotlib.gridspec as gridspec
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
            
            # Define training and testing periods based on config
            calib_period = self.config.get('CALIBRATION_PERIOD', '')
            eval_period = self.config.get('EVALUATION_PERIOD', '')
            
            if calib_period and ',' in calib_period and eval_period and ',' in eval_period:
                # Split periods as defined in config
                calib_start, calib_end = [pd.Timestamp(s.strip()) for s in calib_period.split(',')]
                eval_start, eval_end = [pd.Timestamp(s.strip()) for s in eval_period.split(',')]
                
                calib_mask = (obs_df.index >= calib_start) & (obs_df.index <= calib_end)
                eval_mask = (obs_df.index >= eval_start) & (obs_df.index <= eval_end)
                
                self.logger.info(f"Using calibration period: {calib_start} to {calib_end}")
                self.logger.info(f"Using evaluation period: {eval_start} to {eval_end}")
            else:
                # Create default periods: 60% calibration, 40% evaluation
                sorted_dates = sorted(obs_df.index)
                split_idx = int(len(sorted_dates) * 0.6)
                split_date = sorted_dates[split_idx]
                
                calib_mask = obs_df.index <= split_date
                eval_mask = obs_df.index > split_date
                
                self.logger.info(f"Using default period split at {split_date}")
            
            # Get all run directories
            run_dirs = sorted(self.ensemble_dir.glob("run_*"))
            if not run_dirs:
                self.logger.error("No ensemble run directories found.")
                return None
            
            # Process each run directory to extract results
            calib_metrics = {}
            eval_metrics = {}
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
                    
                    # Convert to datetime and set as index
                    run_df['DateTime'] = pd.to_datetime(run_df[time_col])
                    run_df.set_index('DateTime', inplace=True)
                    
                    # Add to all_flows dataframe for visualization
                    # Use reindex to align with observed data timepoints
                    all_flows[run_id] = run_df[flow_col_sim].reindex(all_flows.index, method='nearest')
                    
                    # Calculate performance metrics for CALIBRATION period
                    # Find common time period within calibration mask
                    calib_obs = observed_flow[calib_mask]
                    common_idx_calib = calib_obs.index.intersection(run_df.index)
                    
                    if len(common_idx_calib) > 5:  # Need at least a few points for metrics
                        # Extract aligned data for metric calculation
                        obs_calib = observed_flow.loc[common_idx_calib]
                        sim_calib = run_df[flow_col_sim].loc[common_idx_calib]
                        
                        # Calculate metrics
                        calib_metrics[run_id] = self._calculate_metrics(obs_calib, sim_calib)
                        self.logger.info(f"Processed {run_id}: KGE={calib_metrics[run_id]['KGE']:.4f}, NSE={calib_metrics[run_id]['NSE']:.4f}, RMSE={calib_metrics[run_id]['RMSE']:.4f}")
                    else:
                        self.logger.warning(f"No common time steps between observed and simulated data for {run_id} in calibration period")
                    
                    # Calculate performance metrics for EVALUATION period
                    # Find common time period within evaluation mask
                    eval_obs = observed_flow[eval_mask]
                    common_idx_eval = eval_obs.index.intersection(run_df.index)
                    
                    if len(common_idx_eval) > 5:  # Need at least a few points for metrics
                        # Extract aligned data for metric calculation
                        obs_eval = observed_flow.loc[common_idx_eval]
                        sim_eval = run_df[flow_col_sim].loc[common_idx_eval]
                        
                        # Calculate metrics
                        eval_metrics[run_id] = self._calculate_metrics(obs_eval, sim_eval)
                        self.logger.debug(f"Evaluation metrics for {run_id}: KGE={eval_metrics[run_id]['KGE']:.4f}, NSE={eval_metrics[run_id]['NSE']:.4f}")
                    else:
                        self.logger.warning(f"No common time steps between observed and simulated data for {run_id} in evaluation period")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing results for {run_id}: {str(e)}")
                    import traceback
                    self.logger.warning(traceback.format_exc())
            
            # Check if we found any valid results
            if len(all_flows.columns) <= 1:  # Only the 'Observed' column
                self.logger.error("No valid streamflow data found in any ensemble run")
                return None
            
            # Convert metrics to DataFrame and save
            if calib_metrics:
                calib_df = pd.DataFrame.from_dict(calib_metrics, orient='index')
                calib_df['Period'] = 'Calibration'
                calib_df.to_csv(analysis_dir / "calibration_metrics.csv")
                self.logger.info(f"Saved calibration metrics for {len(calib_metrics)} runs")
            
            if eval_metrics:
                eval_df = pd.DataFrame.from_dict(eval_metrics, orient='index')
                eval_df['Period'] = 'Evaluation'
                eval_df.to_csv(analysis_dir / "evaluation_metrics.csv")
                self.logger.info(f"Saved evaluation metrics for {len(eval_metrics)} runs")
            
            # Combine calibration and evaluation metrics
            combined_metrics = {}
            for run_id in set(calib_metrics.keys()).union(eval_metrics.keys()):
                combined_metrics[run_id] = {}
                
                # Add calibration metrics with prefix
                if run_id in calib_metrics:
                    for metric, value in calib_metrics[run_id].items():
                        combined_metrics[run_id][f'Calib_{metric}'] = value
                
                # Add evaluation metrics with prefix
                if run_id in eval_metrics:
                    for metric, value in eval_metrics[run_id].items():
                        combined_metrics[run_id][f'Eval_{metric}'] = value
            
            # Save combined metrics
            combined_df = pd.DataFrame.from_dict(combined_metrics, orient='index')
            combined_df = combined_df.reset_index()
            combined_df.rename(columns={'index': 'Run'}, inplace=True)
            combined_df.to_csv(analysis_dir / "performance_metrics.csv", index=False)
            self.logger.info(f"Saved combined performance metrics for {len(combined_metrics)} runs")
            
            # Create period-specific visualization plots
            self._create_period_plots(all_flows, calib_mask, eval_mask, calib_metrics, eval_metrics, analysis_dir)
            
            self.logger.info(f"Ensemble analysis completed. Results saved to {analysis_dir}")
            return analysis_dir
                
        except Exception as e:
            self.logger.error(f"Error during ensemble results analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_metrics(self, observed, simulated):
        """
        Calculate performance metrics between observed and simulated streamflow.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
            
        Returns:
            Dictionary of performance metrics
        """
        import numpy as np
        
        # Handle common issues
        observed = observed.astype(float)
        simulated = simulated.astype(float)
        
        # Drop NaN values
        valid = ~(np.isnan(observed) | np.isnan(simulated))
        observed = observed[valid]
        simulated = simulated[valid]
        
        # Apply capping to reduce impact of extreme outliers (use 99.5th percentile)
        obs_cap = np.percentile(observed, 99.5)
        observed = np.minimum(observed, obs_cap)
        simulated = np.minimum(simulated, obs_cap)
        
        mean_obs = observed.mean()
        
        # Correlation coefficient
        try:
            r = np.corrcoef(simulated, observed)[0, 1]
        except:
            r = np.nan
        
        # Relative variability
        alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
        
        # Bias ratio
        beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan
        
        # KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # Nash-Sutcliffe Efficiency (NSE)
        numerator = ((observed - simulated) ** 2).sum()
        denominator = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (numerator / denominator) if denominator > 0 else np.nan
        
        # Percent Bias (PBIAS)
        pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
        
        # Mean Absolute Error (MAE)
        mae = np.abs(observed - simulated).mean()
        
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

    def _create_period_plots(self, all_flows, calib_mask, eval_mask, calib_metrics, eval_metrics, analysis_dir):
        """
        Create separate plots for calibration and evaluation periods.
        
        Args:
            all_flows: DataFrame with all observed and simulated flows
            calib_mask: Boolean mask for calibration period
            eval_mask: Boolean mask for evaluation period
            calib_metrics: Dictionary with calibration metrics for each run
            eval_metrics: Dictionary with evaluation metrics for each run
            analysis_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.dates import DateFormatter
        import numpy as np
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        
        # Plots for calibration and evaluation periods
        ax_calib = fig.add_subplot(gs[0])
        ax_eval = fig.add_subplot(gs[1], sharex=ax_calib)
        
        # Get simulation columns
        sim_cols = [col for col in all_flows.columns if col != 'Observed']
        
        # Calculate ensemble mean
        all_flows['Ensemble Mean'] = all_flows[sim_cols].mean(axis=1)
        
        # Plot calibration period
        calib_flows = all_flows[calib_mask]
        if not calib_flows.empty:
            # Plot individual simulations
            for col in sim_cols:
                ax_calib.plot(calib_flows.index, calib_flows[col], 'b-', alpha=0.05, linewidth=0.5)
            
            # Plot ensemble mean
            ax_calib.plot(calib_flows.index, calib_flows['Ensemble Mean'], 'r-', linewidth=1.5, label='Ensemble Mean')
            
            # Plot observed
            ax_calib.plot(calib_flows.index, calib_flows['Observed'], 'g-', linewidth=2, label='Observed')
            
            # Add metrics as text
            if calib_metrics:
                # Find best run by KGE
                best_run = max(calib_metrics, key=lambda x: calib_metrics[x]['KGE'])
                best_metrics = calib_metrics[best_run]
                ensemble_kge = calib_metrics.get('Ensemble Mean', {}).get('KGE', np.nan)
                
                metrics_text = (
                    f"Calibration Metrics:\n"
                    f"Ensemble Mean KGE: {np.nanmean([m['KGE'] for m in calib_metrics.values()]):.3f}\n"
                    f"Best Run: {best_run} (KGE: {best_metrics['KGE']:.3f})\n"
                    f"Mean NSE: {np.nanmean([m['NSE'] for m in calib_metrics.values()]):.3f}"
                )
                
                ax_calib.text(0.02, 0.95, metrics_text, transform=ax_calib.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax_calib.set_title('Calibration Period', fontsize=14)
            ax_calib.set_ylabel('Streamflow (m³/s)')
            ax_calib.grid(True, alpha=0.3)
            ax_calib.legend(loc='upper right')
        
        # Plot evaluation period
        eval_flows = all_flows[eval_mask]
        if not eval_flows.empty:
            # Plot individual simulations
            for col in sim_cols:
                ax_eval.plot(eval_flows.index, eval_flows[col], 'b-', alpha=0.05, linewidth=0.5)
            
            # Plot ensemble mean
            ax_eval.plot(eval_flows.index, eval_flows['Ensemble Mean'], 'r-', linewidth=1.5, label='Ensemble Mean')
            
            # Plot observed
            ax_eval.plot(eval_flows.index, eval_flows['Observed'], 'g-', linewidth=2, label='Observed')
            
            # Add metrics as text
            if eval_metrics:
                # Find best run by KGE
                best_run = max(eval_metrics, key=lambda x: eval_metrics[x]['KGE'])
                best_metrics = eval_metrics[best_run]
                
                metrics_text = (
                    f"Evaluation Metrics:\n"
                    f"Ensemble Mean KGE: {np.nanmean([m['KGE'] for m in eval_metrics.values()]):.3f}\n"
                    f"Best Run: {best_run} (KGE: {best_metrics['KGE']:.3f})\n"
                    f"Mean NSE: {np.nanmean([m['NSE'] for m in eval_metrics.values()]):.3f}"
                )
                
                ax_eval.text(0.02, 0.95, metrics_text, transform=ax_eval.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax_eval.set_title('Evaluation Period', fontsize=14)
            ax_eval.set_xlabel('Date')
            ax_eval.set_ylabel('Streamflow (m³/s)')
            ax_eval.grid(True, alpha=0.3)
            ax_eval.legend(loc='upper right')
        
        # Format x-axis dates
        for ax in [ax_calib, ax_eval]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add overall title
        plt.suptitle(f'Ensemble Streamflow Simulations ({len(sim_cols)} Runs)', fontsize=16)
        
        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        plt.savefig(analysis_dir / "ensemble_streamflow_by_period.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional visualization: comparison scatter plots
        self._create_scatter_plots(all_flows, calib_mask, eval_mask, calib_metrics, eval_metrics, analysis_dir)
        
        # Create flow duration curves by period
        self._create_flow_duration_curves(all_flows, calib_mask, eval_mask, analysis_dir)

    def _create_scatter_plots(self, all_flows, calib_mask, eval_mask, calib_metrics, eval_metrics, analysis_dir):
        """Create scatter plots comparing observed vs. ensemble mean streamflow for each period."""
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import linregress
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calibration period scatter plot
        calib_flows = all_flows[calib_mask]
        if not calib_flows.empty and 'Ensemble Mean' in calib_flows.columns:
            # Filter out NaN values
            valid_mask = ~(calib_flows['Observed'].isna() | calib_flows['Ensemble Mean'].isna())
            if valid_mask.sum() > 0:
                valid_calib = calib_flows[valid_mask]
                
                # Get arrays for regression
                x = valid_calib['Observed'].values
                y = valid_calib['Ensemble Mean'].values
                
                # Plot scatter
                ax1.scatter(x, y, alpha=0.6, color='blue', label='Calibration')
                
                # Add 1:1 line
                max_val = max(x.max(), y.max())
                min_val = min(x.min(), y.min())
                ax1.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
                
                # Add regression line if possible
                if len(x) >= 2:
                    try:
                        slope, intercept, r_value, _, _ = linregress(x, y)
                        if not np.isnan(slope) and not np.isnan(intercept):
                            ax1.plot([min_val, max_val], 
                                    [slope * min_val + intercept, slope * max_val + intercept], 
                                    'r-', label=f'Regression (r={r_value:.2f})')
                    except Exception as e:
                        self.logger.warning(f"Error calculating regression for calibration scatter: {str(e)}")
                
                # Add metrics text
                if calib_metrics:
                    metrics_text = (
                        f"Calibration Metrics:\n"
                        f"KGE: {np.nanmean([m['KGE'] for m in calib_metrics.values()]):.3f}\n"
                        f"NSE: {np.nanmean([m['NSE'] for m in calib_metrics.values()]):.3f}\n"
                        f"RMSE: {np.nanmean([m['RMSE'] for m in calib_metrics.values()]):.3f}"
                    )
                    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_xlabel('Observed Streamflow (m³/s)')
        ax1.set_ylabel('Simulated Streamflow (m³/s)')
        ax1.set_title('Calibration Period')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Evaluation period scatter plot
        eval_flows = all_flows[eval_mask]
        if not eval_flows.empty and 'Ensemble Mean' in eval_flows.columns:
            # Filter out NaN values
            valid_mask = ~(eval_flows['Observed'].isna() | eval_flows['Ensemble Mean'].isna())
            if valid_mask.sum() > 0:
                valid_eval = eval_flows[valid_mask]
                
                # Get arrays for regression
                x = valid_eval['Observed'].values
                y = valid_eval['Ensemble Mean'].values
                
                # Plot scatter
                ax2.scatter(x, y, alpha=0.6, color='red', label='Evaluation')
                
                # Add 1:1 line
                max_val = max(x.max(), y.max())
                min_val = min(x.min(), y.min())
                ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
                
                # Add regression line if possible
                if len(x) >= 2:
                    try:
                        slope, intercept, r_value, _, _ = linregress(x, y)
                        if not np.isnan(slope) and not np.isnan(intercept):
                            ax2.plot([min_val, max_val], 
                                    [slope * min_val + intercept, slope * max_val + intercept], 
                                    'r-', label=f'Regression (r={r_value:.2f})')
                    except Exception as e:
                        self.logger.warning(f"Error calculating regression for evaluation scatter: {str(e)}")
                
                # Add metrics text
                if eval_metrics:
                    metrics_text = (
                        f"Evaluation Metrics:\n"
                        f"KGE: {np.nanmean([m['KGE'] for m in eval_metrics.values()]):.3f}\n"
                        f"NSE: {np.nanmean([m['NSE'] for m in eval_metrics.values()]):.3f}\n"
                        f"RMSE: {np.nanmean([m['RMSE'] for m in eval_metrics.values()]):.3f}"
                    )
                    ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('Observed Streamflow (m³/s)')
        ax2.set_ylabel('Simulated Streamflow (m³/s)')
        ax2.set_title('Evaluation Period')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Try to set equal aspect ratio and axes limits
        for ax in [ax1, ax2]:
            try:
                ax.set_aspect('equal')
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(analysis_dir / "ensemble_scatter_by_period.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_flow_duration_curves(self, all_flows, calib_mask, eval_mask, analysis_dir):
        """Create flow duration curves for calibration and evaluation periods."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get simulation columns
        sim_cols = [col for col in all_flows.columns if col not in ['Observed', 'Ensemble Mean']]
        
        # Calibration period FDC
        calib_flows = all_flows[calib_mask]
        if not calib_flows.empty:
            # Calculate FDC for observed data
            obs_sorted = calib_flows['Observed'].dropna().sort_values(ascending=False)
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            
            # Plot observed FDC
            ax1.plot(obs_exceed, obs_sorted, 'g-', linewidth=2, label='Observed')
            
            # Calculate and plot FDC for ensemble mean
            if 'Ensemble Mean' in calib_flows.columns:
                mean_sorted = calib_flows['Ensemble Mean'].dropna().sort_values(ascending=False)
                mean_exceed = np.arange(1, len(mean_sorted) + 1) / (len(mean_sorted) + 1)
                ax1.plot(mean_exceed, mean_sorted, 'r-', linewidth=1.5, label='Ensemble Mean')
            
            # Plot individual model FDCs (limit to first 20 for clarity)
            for col in sim_cols[:min(20, len(sim_cols))]:
                sim_sorted = calib_flows[col].dropna().sort_values(ascending=False)
                sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
                ax1.plot(sim_exceed, sim_sorted, 'b-', alpha=0.05, linewidth=0.5)
        
        ax1.set_xlabel('Exceedance Probability')
        ax1.set_ylabel('Streamflow (m³/s)')
        ax1.set_title('Calibration Period Flow Duration Curve')
        ax1.set_yscale('log')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend()
        
        # Evaluation period FDC
        eval_flows = all_flows[eval_mask]
        if not eval_flows.empty:
            # Calculate FDC for observed data
            obs_sorted = eval_flows['Observed'].dropna().sort_values(ascending=False)
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            
            # Plot observed FDC
            ax2.plot(obs_exceed, obs_sorted, 'g-', linewidth=2, label='Observed')
            
            # Calculate and plot FDC for ensemble mean
            if 'Ensemble Mean' in eval_flows.columns:
                mean_sorted = eval_flows['Ensemble Mean'].dropna().sort_values(ascending=False)
                mean_exceed = np.arange(1, len(mean_sorted) + 1) / (len(mean_sorted) + 1)
                ax2.plot(mean_exceed, mean_sorted, 'r-', linewidth=1.5, label='Ensemble Mean')
            
            # Plot individual model FDCs (limit to first 20 for clarity)
            for col in sim_cols[:min(20, len(sim_cols))]:
                sim_sorted = eval_flows[col].dropna().sort_values(ascending=False)
                sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
                ax2.plot(sim_exceed, sim_sorted, 'b-', alpha=0.05, linewidth=0.5)
        
        ax2.set_xlabel('Exceedance Probability')
        ax2.set_ylabel('Streamflow (m³/s)')
        ax2.set_title('Evaluation Period Flow Duration Curve')
        ax2.set_yscale('log')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
        
        # Use same y-axis limits for both plots if possible
        try:
            y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(analysis_dir / "flow_duration_curves_by_period.png", dpi=300, bbox_inches='tight')
        plt.close()


import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import xarray as xr
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import netCDF4 as nc 

class RandomForestEmulator:
    """
    Random Forest Emulator for CONFLUENCE.
    
    This class uses parameter sets and performance metrics to train a random forest model
    that can predict performance metrics based on parameter values. It then optimizes 
    the parameter values to maximize performance and runs SUMMA with the optimal parameter set.
    
    Optionally, attribute data can be used to improve predictions.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Random Forest Emulator.
        
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
        
        # Paths to required data
        self.attributes_path = self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv"
        self.emulator_dir = self.project_dir / "emulation" / self.experiment_id
        self.parameter_sets_path = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
        self.performance_metrics_path = self.emulator_dir / "ensemble_analysis" / "performance_metrics.csv"
        
        # Output paths
        self.emulation_output_dir = self.emulator_dir / "rf_emulation"
        self.emulation_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.target_metric = self.config.get('EMULATION_TARGET_METRIC', 'KGE')
        self.n_estimators = self.config.get('EMULATION_RF_N_ESTIMATORS', 100)
        self.max_depth = self.config.get('EMULATION_RF_MAX_DEPTH', None)
        self.random_state = self.config.get('EMULATION_SEED', 42)
        
        # New flag to control attribute data usage
        self.use_attributes = self.config.get('EMULATION_USE_ATTRIBUTES', False)
        
        # Optimization settings
        self.n_optimization_runs = self.config.get('EMULATION_N_OPTIMIZATION_RUNS', 10)
        
        # Model and scaler objects
        self.model = None
        self.X_scaler = None
        self.param_mins = {}
        self.param_maxs = {}
        self.param_cols = []
        self.attribute_cols = []
        
        # Time period information from config for evaluation
        self.calibration_period = self._parse_date_range(self.config.get('CALIBRATION_PERIOD', ''))
        self.evaluation_period = self._parse_date_range(self.config.get('EVALUATION_PERIOD', ''))
        
        # Check if all required files exist
        self._check_required_files()
    
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
            
    def _check_required_files(self):
        """Check if all required input files exist."""
        self.logger.info("Checking required input files")
        
        files_to_check = [
            (self.parameter_sets_path, "Parameter sets"),
            (self.performance_metrics_path, "Performance metrics")
        ]
        
        # Only check for attributes file if we're using attributes
        if self.use_attributes:
            files_to_check.append((self.attributes_path, "Attribute data"))
        
        missing_files = []
        for file_path, description in files_to_check:
            if not file_path.exists():
                self.logger.error(f"{description} file not found: {file_path}")
                missing_files.append((file_path, description))
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {[desc for _, desc in missing_files]}")
            
        self.logger.info("All required input files found")
                        
    def load_and_prepare_data(self):
        """
        Load and prepare the data for training the random forest model
        with improved handling of calibration and evaluation metrics.
        
        Returns:
            Tuple of X and y arrays for model training
        """
        self.logger.info("Loading and preparing data for random forest training")
        
        # Load parameter sets
        parameter_data = self._load_parameter_sets()
        self.logger.info(f"Loaded parameter data with shape: {parameter_data.shape}")
        
        # Load performance metrics
        try:
            performance_df = pd.read_csv(self.performance_metrics_path)
            self.logger.info(f"Loaded performance metrics with shape: {performance_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading performance metrics: {str(e)}")
            raise
        
        # Print debugging info about the data
        self.logger.info(f"Parameter data columns: {parameter_data.columns.tolist()}")
        self.logger.info(f"Performance metrics columns: {performance_df.columns.tolist()}")
        
        # Special handling for performance metrics file
        if 'Run' in performance_df.columns:
            performance_df.set_index('Run', inplace=True)
        elif 'Unnamed: 0' in performance_df.columns:
            performance_df.set_index('Unnamed: 0', inplace=True)
        
        # Check for calibration and evaluation metrics
        if 'Calib_KGE' in performance_df.columns:
            # Use calibration metrics preferentially
            self.target_metric = 'Calib_KGE'
            self.logger.info(f"Using calibration metrics for training with target: {self.target_metric}")
        elif 'Eval_KGE' in performance_df.columns:
            # Fall back to evaluation metrics
            self.target_metric = 'Eval_KGE'
            self.logger.info(f"Using evaluation metrics for training with target: {self.target_metric}")
        elif 'KGE' in performance_df.columns:
            # Fall back to original metrics
            self.target_metric = 'KGE'
            self.logger.info(f"Using original metrics for training with target: {self.target_metric}")
        else:
            # Check for any KGE-containing column
            kge_cols = [col for col in performance_df.columns if 'KGE' in col]
            if kge_cols:
                self.target_metric = kge_cols[0]
                self.logger.info(f"Using {self.target_metric} as target metric")
            else:
                # Try other metrics
                available_metrics = []
                for prefix in ['Calib_', 'Eval_', '']:
                    for metric in ['NSE', 'RMSE', 'PBIAS']:
                        full_metric = f"{prefix}{metric}"
                        if full_metric in performance_df.columns:
                            available_metrics.append(full_metric)
                
                if available_metrics:
                    self.target_metric = available_metrics[0]
                    self.logger.info(f"No KGE metric found, using {self.target_metric} instead")
                else:
                    raise ValueError(f"No valid performance metrics found in {performance_df.columns}")
        
        # Filter out NaN values in the target metric
        valid_metrics = ~performance_df[self.target_metric].isna()
        self.logger.info(f"Found {valid_metrics.sum()} valid entries for {self.target_metric} out of {len(performance_df)}")
        
        if valid_metrics.sum() == 0:
            raise ValueError(f"No valid data for target metric {self.target_metric}")
        
        performance_df = performance_df[valid_metrics]
        
        # If we have run IDs in the index, filter parameter data
        if not performance_df.index.name and isinstance(performance_df.index, pd.Index) and len(performance_df.index) > 0:
            # Extract run numbers (assuming run_XXXX format)
            if isinstance(performance_df.index[0], str) and 'run_' in performance_df.index[0]:
                # Extract numeric part if index is like "run_0001"
                run_nums = []
                for idx in performance_df.index:
                    try:
                        run_nums.append(int(idx.split('_')[1]))
                    except (IndexError, ValueError):
                        run_nums.append(None)
                
                # Filter parameter data to matching runs
                valid_runs = [i for i, num in enumerate(run_nums) if num is not None and num < len(parameter_data)]
                if valid_runs:
                    performance_df = performance_df.iloc[valid_runs]
                    parameter_data = parameter_data.iloc[[run_nums[i] for i in valid_runs if run_nums[i] is not None]]
                    self.logger.info(f"Filtered to {len(valid_runs)} matching runs")
        
        # Ensure both datasets have the same number of rows
        min_rows = min(len(parameter_data), len(performance_df))
        if min_rows == 0:
            raise ValueError("No matching data between parameters and performance metrics")
        
        parameter_data = parameter_data.iloc[:min_rows]
        performance_df = performance_df.iloc[:min_rows]
        
        self.logger.info(f"After alignment: parameter_data shape: {parameter_data.shape}, performance_df shape: {performance_df.shape}")
        
        # Merge parameters and performance metrics
        merged_data = pd.concat([parameter_data.reset_index(drop=True), 
                                performance_df.reset_index()], axis=1)
        
        self.logger.info(f"Merged parameter and performance data with shape: {merged_data.shape}")
        
        # Fill missing values in numeric columns with means
        numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            missing_count = merged_data[col].isna().sum()
            if missing_count > 0:
                col_mean = merged_data[col].mean()
                merged_data[col].fillna(col_mean, inplace=True)
                self.logger.info(f"Filled {missing_count} missing values in {col} with mean: {col_mean}")
        
        # Final check for any remaining missing values
        missing_values = merged_data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            self.logger.warning(f"Still have columns with missing values: {columns_with_missing}")
            # For remaining missing values, use forward fill then backward fill
            merged_data = merged_data.ffill().bfill()
            
        # Final check - ensure we have enough data
        if merged_data.shape[0] < 5:  # Arbitrary minimum for RF model
            raise ValueError(f"Not enough data for training: only {merged_data.shape[0]} samples after preprocessing")
        
        self.logger.info(f"Final data shape after preprocessing: {merged_data.shape}")
        
        # Split into features and target
        X, y, _ = self._split_features_target(merged_data)
        
        # Save record of which columns are parameters
        self.param_cols = [col for col in X.columns if col.startswith('param_')]
        
        # Store parameter mins and maxs for optimization bounds
        for param in self.param_cols:
            self.param_mins[param] = X[param].min()
            self.param_maxs[param] = X[param].max()
        
        # Scale features with StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.X_scaler = scaler
        
        return X_scaled, y
    
    def _load_parameter_sets(self):
        """
        Load parameter sets from NetCDF file and convert to DataFrame.
        
        Returns:
            DataFrame with parameter values
        """
        # Load parameter sets from NetCDF
        try:
            ds = xr.open_dataset(self.parameter_sets_path)
            
            # Get all parameter variables (excluding dimensions and special variables)
            param_vars = [var for var in ds.variables if var not in ['hruId', 'runIndex', 'run', 'hru']]
            
            # Initialize empty DataFrame
            param_data = pd.DataFrame(index=ds.runIndex.values)
            
            # For each parameter, calculate the mean value across all HRUs for each run
            for param in param_vars:
                param_data[f"param_{param}"] = ds[param].mean(dim='hru').values
            
            return param_data
            
        except Exception as e:
            self.logger.error(f"Error loading parameter sets: {str(e)}")
            raise
                
    def _split_features_target(self, merged_data):
        """
        Split merged data into features (X) and target (y), excluding other metrics.
        
        Args:
            merged_data: Merged DataFrame with all data
        
        Returns:
            Tuple of (X, y) DataFrames
        """
        # Ensure the target metric exists
        if self.target_metric not in merged_data.columns:
            self.logger.error(f"Target metric {self.target_metric} not found in data columns: {merged_data.columns.tolist()}")
            raise ValueError(f"Target metric {self.target_metric} not found in data")
        
        # Identify parameters (columns starting with 'param_')
        param_cols = [col for col in merged_data.columns if col.startswith('param_')]
        
        # List of all potential metric prefixes and suffixes
        prefixes = ['Calib_', 'Eval_', '']
        suffixes = ['KGE', 'NSE', 'RMSE', 'PBIAS', 'MAE', 'r', 'alpha', 'beta']
        
        # Generate all possible metric column names
        metric_cols = [f"{prefix}{suffix}" for prefix in prefixes for suffix in suffixes]
        
        # Also exclude any ID columns
        excluded_cols = metric_cols + ['Run', 'Unnamed: 0', 'id', 'hru_id', 'gru_id', 'basin_id']
        
        # Create feature set (parameters only)
        X = merged_data[param_cols].copy()
        
        # For target, determine how to handle based on the metric
        if self.target_metric.endswith('KGE') or self.target_metric.endswith('NSE'):
            y = merged_data[self.target_metric]  # Higher is better
        elif self.target_metric.endswith('RMSE'):
            y = -merged_data[self.target_metric]  # Lower is better, so negate
        elif self.target_metric.endswith('PBIAS'):
            y = -merged_data[self.target_metric].abs()  # Minimize absolute bias
        else:
            y = merged_data[self.target_metric]
        
        return X, y, merged_data

    def _scale_features(self, X, merged_data=None):
        """
        Scale features using StandardScaler and handle categorical features.
        
        Args:
            X: Feature DataFrame (numeric features only)
            merged_data: Full merged DataFrame including categorical features
        
        Returns:
            Tuple of (scaled_X, scaler)
        """
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import numpy as np
        
        if X.empty or X.shape[0] == 0:
            raise ValueError("Cannot scale empty feature set")
        
        # First, handle the categorical (non-numeric) columns if present
        if hasattr(self, 'non_numeric_cols') and self.non_numeric_cols and merged_data is not None:
            self.logger.info(f"One-hot encoding {len(self.non_numeric_cols)} categorical features")
            
            # Create a one-hot encoder for categorical columns
            # We'll use pandas get_dummies for simplicity
            categorical_data = merged_data[self.non_numeric_cols].fillna('missing')
            
            # Convert all columns to string to ensure proper encoding
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].astype(str)
            
            # One-hot encode
            one_hot = pd.get_dummies(categorical_data, prefix=self.non_numeric_cols, drop_first=True)
            self.logger.info(f"One-hot encoding created {one_hot.shape[1]} binary features")
            
            # Scale numeric features
            scaler = StandardScaler()
            X_scaled_numeric = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Combine scaled numeric and one-hot encoded categorical features
            X_scaled = pd.concat([X_scaled_numeric, one_hot], axis=1)
            
            # Remember the one-hot encoded columns for optimization
            self.one_hot_cols = one_hot.columns.tolist()
        else:
            # Just scale numeric features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, scaler

    def train_random_forest(self, X, y, spinup_size=1/3, test_size=1/3):
        """
        Train a random forest model using the prepared data with a three-way split.
        
        Args:
            X: Scaled feature data
            y: Target values
            spinup_size: Proportion of data to use for spinup
            test_size: Proportion of data to use for testing
        
        Returns:
            Trained model and evaluation metrics
        """
        self.logger.info("Training random forest model with three-way split")
        
        # Calculate sizes for a three-way split
        n_samples = len(X)
        n_spinup = int(n_samples * spinup_size)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_spinup - n_test
        
        # Create indices for each portion
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        # Split indices into three groups
        spinup_indices = indices[:n_spinup]
        train_indices = indices[n_spinup:n_spinup+n_train]
        test_indices = indices[n_spinup+n_train:]
        
        # Create the splits
        X_spinup, y_spinup = X.iloc[spinup_indices], y.iloc[spinup_indices]
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        self.logger.info(f"Data split: {n_spinup} spinup, {n_train} training, {n_test} testing samples")
        
        # Initialize and train the model (using only the training data)
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate the model on all three splits
        spinup_score = model.score(X_spinup, y_spinup)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.logger.info(f"Random forest model trained with R² scores:")
        self.logger.info(f"  Spinup: {spinup_score:.4f}")
        self.logger.info(f"  Training: {train_score:.4f}")
        self.logger.info(f"  Testing: {test_score:.4f}")
        
        # Calculate feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        
        # Log top 10 most important features
        self.logger.info("Top 10 most important features:")
        for feature, importance in importances[:10].items():
            self.logger.info(f"  {feature}: {importance:.4f}")
        
        # Save feature importances to CSV
        importances_df = importances.reset_index()
        importances_df.columns = ['Feature', 'Importance']
        importances_df.to_csv(self.emulation_output_dir / "feature_importances.csv", index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        importances[:20].plot(kind='bar')
        plt.title('Feature Importances (Top 20)')
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "feature_importances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store the model for later use
        self.model = model
        
        # Save the indices for each split to enable consistent evaluation
        self.spinup_indices = spinup_indices
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        return model, {
            'spinup_score': spinup_score, 
            'train_score': train_score, 
            'test_score': test_score
        }
    
    def optimize_parameters_global(self, fixed_attributes=None):
        """Use differential evolution for global optimization"""
        from scipy.optimize import differential_evolution
        
        # Define objective function
        def objective(params_array):
            input_data = {param: val for param, val in zip(self.param_cols, params_array)}
            X_pred = pd.DataFrame([input_data])
            
            # Ensure all columns from training are present
            for col in self.model.feature_names_in_:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            # Ensure correct column order
            X_pred = X_pred[self.model.feature_names_in_]
            
            # Return negative predicted score (for minimization)
            return -self.model.predict(X_pred)[0]
        
        # Define bounds
        bounds = [(self.param_mins[param], self.param_maxs[param]) for param in self.param_cols]
        
        # Run differential evolution
        result = differential_evolution(
            objective, 
            bounds,
            popsize=15,  # Increase population size
            mutation=(0.5, 1.0),  # Allow larger mutations
            recombination=0.7,
            strategy='best1bin',
            maxiter=100,  # More iterations
            polish=True   # Local optimization at the end
        )
        
        # Convert results to parameter dictionary
        optimized_params = {}
        for i, param in enumerate(self.param_cols):
            param_name = param.replace('param_', '')
            optimized_params[param_name] = result.x[i]
            self.logger.info(f"Optimized {param_name}: {result.x[i]:.4f}")
        
        # Save to CSV
        params_df = pd.DataFrame([optimized_params])
        params_df.to_csv(self.emulation_output_dir / "optimized_parameters.csv", index=False)
        
        return optimized_params, -result.fun

    def optimize_parameters(self, fixed_attributes=None):
        """
        Optimize parameters to maximize the predicted performance metric.
        
        Args:
            fixed_attributes: Dictionary of fixed attribute values to use (optional)
        
        Returns:
            Dictionary of optimized parameter values and predicted score
        """
        self.logger.info("Starting parameter optimization")
        
        # Check if model exists
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest() first.")
        
        # Define objective function (negative because we want to maximize)
        def objective(params_array):
            # Create a DataFrame with just the parameter values
            input_data = {}
            
            # Add parameters
            for i, param in enumerate(self.param_cols):
                input_data[param] = params_array[i]
            
            # Convert to DataFrame with the exact same columns as used in training
            X_pred = pd.DataFrame([input_data])
            
            # Make sure all columns from training are present in prediction
            for col in self.model.feature_names_in_:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            # Make sure columns are in the same order as training
            X_pred = X_pred[self.model.feature_names_in_]
            
            # Predict the metric and return negative (for minimization)
            return -self.model.predict(X_pred)[0]
        
        # Define bounds for parameters (from min/max values observed in training data)
        bounds = [(self.param_mins[param], self.param_maxs[param]) for param in self.param_cols]
        
        # Run optimization multiple times with different starting points
        best_score = float('-inf')
        best_params = None
        
        for i in range(self.n_optimization_runs):
            self.logger.info(f"Optimization run {i+1}/{self.n_optimization_runs}")
            
            # Generate random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
            
            # Run optimization
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Check if better than current best
            if -result.fun > best_score:
                best_score = -result.fun
                best_params = result.x
                if i > 0:  # Don't log for first run as it's always the "best" so far
                    self.logger.info(f"New best score: {best_score:.4f}")
        
        # Convert best parameters to dictionary
        optimized_params = {}
        for i, param in enumerate(self.param_cols):
            # Remove 'param_' prefix from parameter name
            param_name = param.replace('param_', '')
            optimized_params[param_name] = best_params[i]
            self.logger.info(f"Optimized {param_name}: {best_params[i]:.4f}")
        
        # Save optimized parameters to CSV
        params_df = pd.DataFrame([optimized_params])
        params_df.to_csv(self.emulation_output_dir / "optimized_parameters.csv", index=False)
        
        # Generate and save trialParams.nc file
        self._generate_trial_params_file(optimized_params)
        
        self.logger.info(f"Parameter optimization completed with predicted score: {best_score:.4f}")
        
        return optimized_params, best_score
        
    def _generate_trial_params_file(self, optimized_params):
        """
        Generate a trialParams.nc file with the optimized parameters.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
        
        Returns:
            Path: Path to the generated file
        """
        self.logger.info("Generating trialParams.nc file with optimized parameters")
        
        # Get attribute file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        try:
            # Read HRU information from the attributes file
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                
                # Create the trial parameters dataset
                output_ds = nc.Dataset(self.emulation_output_dir / "trialParams.nc", 'w', format='NETCDF4')
                
                # Add dimensions
                output_ds.createDimension('hru', len(hru_ids))
                
                # Add HRU ID variable
                hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Add parameter variables
                for param_name, param_value in optimized_params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Create variable and assign values
                    param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                    param_var[:] = param_array
                    param_var.long_name = f"Trial value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE RandomForestEmulator"
                output_ds.history = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.confluence_experiment_id = self.experiment_id
                
                # Close the file explicitly
                output_ds.close()
                
                # Copy to SUMMA settings directory for use in model runs
                summa_settings_dir = self.project_dir / "settings" / "SUMMA"
                target_path = summa_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                
                import shutil
                shutil.copy2(self.emulation_output_dir / "trialParams.nc", target_path)
                self.logger.info(f"Copied trial parameters to SUMMA settings directory: {target_path}")
                
                return target_path
                    
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_summa_with_optimal_parameters(self):
        """
        Run SUMMA with the optimized parameters.
        
        Returns:
            True if SUMMA run was successful, False otherwise
        """
        self.logger.info("Running SUMMA with optimized parameters")
        
        # Define paths
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE', 'summa.exe')
        filemanager_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        # Check if files exist
        if not summa_exe.exists():
            self.logger.error(f"SUMMA executable not found: {summa_exe}")
            return False
        
        if not filemanager_path.exists():
            self.logger.error(f"File manager not found: {filemanager_path}")
            return False
        
        # Create output directory
        output_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "SUMMA"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify fileManager to use the optimized output directory
        modified_filemanager = self._modify_filemanager(filemanager_path, output_dir)
        
        # If we couldn't modify the fileManager, use the original
        if modified_filemanager is None:
            self.logger.warning(f"Could not modify fileManager, using original: {filemanager_path}")
            modified_filemanager = filemanager_path
        
        # Run SUMMA
        summa_command = f"{summa_exe} -m {modified_filemanager}"
        self.logger.info(f"SUMMA command: {summa_command}")
        
        try:
            # Run SUMMA and capture output
            log_file = log_dir / "summa_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    summa_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"SUMMA run completed successfully, log saved to: {log_file}")
            
            # Run mizuRoute if needed
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                self._run_mizuroute(output_dir)
            
            # Run evaluation
            self._evaluate_summa_output()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False
    
    def _modify_filemanager(self, filemanager_path, output_dir):
        """
        Modify fileManager to use the optimized output directory.
        
        Args:
            filemanager_path: Path to original fileManager
            output_dir: New output directory
        
        Returns:
            Path to modified fileManager
        """
        try:
            with open(filemanager_path, 'r') as f:
                lines = f.readlines()
            
            # Create new fileManager in the emulation output directory
            new_filemanager = self.emulation_output_dir / "fileManager_rf_optimized.txt"
            
            with open(new_filemanager, 'w') as f:
                for line in lines:
                    if "outputPath" in line:
                        # Update output path
                        output_path_str = str(output_dir).replace('\\', '/')
                        f.write(f"outputPath           '{output_path_str}/' ! \n")
                    elif "outFilePrefix" in line:
                        # Update output file prefix
                        prefix_parts = line.split("'")
                        if len(prefix_parts) >= 3:
                            original_prefix = prefix_parts[1]
                            new_prefix = f"{self.experiment_id}_rf_optimized"
                            modified_line = line.replace(f"'{original_prefix}'", f"'{new_prefix}'")
                            f.write(modified_line)
                        else:
                            f.write(line)
                    else:
                        f.write(line)
            
            return new_filemanager
        
        except Exception as e:
            self.logger.error(f"Error modifying fileManager: {str(e)}")
            return None
    
    def _run_mizuroute(self, summa_output_dir):
        """
        Run mizuRoute with the SUMMA outputs.
        
        Args:
            summa_output_dir: Directory containing SUMMA outputs
        """
        self.logger.info("Running mizuRoute with SUMMA outputs")
        
        # Define paths
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Get mizuRoute settings
        mizu_settings_dir = self.project_dir / "settings" / "mizuRoute"
        mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        # Check if files exist
        if not mizu_exe.exists():
            self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
            return False
        
        if not mizu_control_file.exists():
            self.logger.error(f"mizuRoute control file not found: {mizu_control_file}")
            return False
        
        # Create output directory
        mizu_output_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "mizuRoute"
        mizu_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = mizu_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify mizuRoute control file
        modified_control = self._modify_mizuroute_control(
            mizu_control_file,
            summa_output_dir,
            mizu_settings_dir,
            mizu_output_dir
        )
        
        # If we couldn't modify the control file, use the original
        if modified_control is None:
            self.logger.warning(f"Could not modify mizuRoute control file, using original: {mizu_control_file}")
            modified_control = mizu_control_file
        
        # Run mizuRoute
        mizu_command = f"{mizu_exe} {modified_control}"
        self.logger.info(f"mizuRoute command: {mizu_command}")
        
        try:
            # Run mizuRoute and capture output
            log_file = log_dir / "mizuroute_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"mizuRoute run completed successfully, log saved to: {log_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False
    
    def _modify_mizuroute_control(self, control_file_path, summa_output_dir, mizu_settings_dir, mizu_output_dir):
        """
        Modify mizuRoute control file to use the optimized outputs.
        
        Args:
            control_file_path: Path to original control file
            summa_output_dir: Directory containing SUMMA outputs
            mizu_settings_dir: Directory containing mizuRoute settings
            mizu_output_dir: Directory for mizuRoute outputs
        
        Returns:
            Path to modified control file
        """
        try:
            with open(control_file_path, 'r') as f:
                lines = f.readlines()
            
            # Create new control file in the emulation output directory
            new_control_file = self.emulation_output_dir / "mizuroute_rf_optimized.control"
            
            with open(new_control_file, 'w') as f:
                for line in lines:
                    if '<input_dir>' in line:
                        # Update input directory (SUMMA output)
                        summa_dir_str = str(summa_output_dir).replace('\\', '/')
                        f.write(f"<input_dir>             {summa_dir_str}/    ! Folder that contains runoff data from SUMMA \n")
                    elif '<ancil_dir>' in line:
                        # Update ancillary directory
                        mizu_settings_str = str(mizu_settings_dir).replace('\\', '/')
                        f.write(f"<ancil_dir>             {mizu_settings_str}/    ! Folder that contains ancillary data \n")
                    elif '<output_dir>' in line:
                        # Update output directory
                        mizu_output_str = str(mizu_output_dir).replace('\\', '/')
                        f.write(f"<output_dir>            {mizu_output_str}/    ! Folder that will contain mizuRoute simulations \n")
                    elif '<case_name>' in line:
                        # Update case name
                        f.write(f"<case_name>             {self.experiment_id}_rf_optimized    ! Simulation case name \n")
                    elif '<fname_qsim>' in line:
                        # Update input file name
                        f.write(f"<fname_qsim>            {self.experiment_id}_rf_optimized_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                    else:
                        f.write(line)
            
            return new_control_file
        
        except Exception as e:
            self.logger.error(f"Error modifying mizuRoute control file: {str(e)}")
            return None
        
    def _evaluate_summa_output(self, output_dir=None):
        """
        Evaluate SUMMA and mizuRoute outputs by comparing to observed data.
        Only evaluates the calibration and evaluation periods defined in config.
        
        Args:
            output_dir: Optional directory where model output is stored. If None,
                    uses the default simulation directory.
        
        Returns:
            Dict: Dictionary with performance metrics, or None if evaluation failed
        """
        self.logger.info("Evaluating model outputs")
        
        # Define paths based on output_dir
        if output_dir is None:
            # Use default simulation directories
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # For distributed models, use mizuRoute output
                sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}" / "mizuRoute"
                sim_pattern = "*.nc"  # All netCDF files
            else:
                # For lumped models, use SUMMA output directly
                sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}" / "SUMMA"
                sim_pattern = f"{self.experiment_id}*.nc"  # SUMMA output files
        else:
            # Use specified output directory
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # For distributed models, use mizuRoute output
                sim_dir = output_dir / "simulations" / self.experiment_id / "mizuRoute"
                sim_pattern = "*.nc"  # All netCDF files
            else:
                # For lumped models, use SUMMA output directly
                sim_dir = output_dir / "simulations" / self.experiment_id / "SUMMA"
                sim_pattern = "*rf_iter*.nc"  # SUMMA iteration output files
        
        # Get observed data path
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            obs_path = Path(obs_path)
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Read observed data - first check column names
            with open(obs_path, 'r') as f:
                header_line = f.readline().strip()
            
            header_cols = header_line.split(',')
            self.logger.debug(f"Observed data columns: {header_cols}")
            
            # Identify date column
            date_col = next((col for col in header_cols if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data. Available columns: {header_cols}")
                return None
            
            # Identify flow column 
            flow_col = next((col for col in header_cols if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data. Available columns: {header_cols}")
                return None
            
            # Now read the CSV with known column names
            obs_df = pd.read_csv(obs_path, parse_dates=[date_col])
            
            # Set the date column as index
            obs_df.set_index(date_col, inplace=True)
            observed_flow = obs_df[flow_col]
                        
            # Extract simulated streamflow
            sim_files = list(sim_dir.glob(sim_pattern))

            if not sim_files:
                # If we didn't find any files matching the pattern, try specifically looking for timestep files
                timestep_pattern = "*timestep.nc"
                sim_files = list(sim_dir.glob(timestep_pattern))
                
                if sim_files:
                    self.logger.info(f"Found timestep files using pattern: {timestep_pattern}")
                else:
                    # If still no files, try a more general pattern
                    sim_files = list(sim_dir.glob("*.nc"))
                    if sim_files:
                        self.logger.info(f"Found some NetCDF files using general pattern: *.nc")
                    else:
                        self.logger.warning(f"No simulation files found in {sim_dir}")
                        return None
            
            if not sim_files:
                self.logger.warning(f"No simulation files found matching pattern {sim_pattern} in {sim_dir}")
                return None
            
            # Use the first file found
            sim_file = sim_files[1]
            self.logger.info(f"Using simulation file: {sim_file}")
            
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # From mizuRoute output
                with xr.open_dataset(sim_file) as ds:
                    # Find the index for the reach ID
                    sim_reach_id = self.config.get('SIM_REACH_ID')
                    if 'reachID' in ds.variables:
                        reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                        
                        if len(reach_indices) > 0:
                            reach_index = reach_indices[0]
                            
                            # Extract streamflow for this reach
                            # Try common variable names
                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                if var_name in ds.variables:
                                    sim_flow = ds[var_name].isel(seg=reach_index).to_pandas()
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
                with xr.open_dataset(sim_file) as ds:
                    # Try to find streamflow variable
                    streamflow_var = None
                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            streamflow_var = var_name
                            break
                    
                    if streamflow_var is None:
                        self.logger.error("Could not find streamflow variable in SUMMA output")
                        return None
                    
                    # Extract the variable
                    if 'gru' in ds[streamflow_var].dims and ds.sizes['gru'] > 1:
                        # Sum across GRUs if multiple
                        runoff_series = ds[streamflow_var].sum(dim='gru').to_pandas()
                    else:
                        # Single GRU or no gru dimension
                        runoff_series = ds[streamflow_var].to_pandas()
                        # Handle case if it's a DataFrame
                        if isinstance(runoff_series, pd.DataFrame):
                            runoff_series = runoff_series.iloc[:, 0]
                    
                    # Get catchment area to convert from m/s to m³/s
                    catchment_area = self._get_catchment_area()
                    if catchment_area is None or catchment_area <= 0:
                        self.logger.warning("Missing or invalid catchment area. Using default 1.0 km²")
                        catchment_area = 1.0 * 1e6  # 1 km² in m²
                    
                    # Convert from m/s to m³/s
                    sim_flow = runoff_series * catchment_area
            
            # Align time index format for observed and simulated data
            obs_aligned = observed_flow.copy()
            sim_aligned = sim_flow.copy()
            
            # Filter data to calibration and evaluation periods if defined
            calib_start, calib_end = self.calibration_period
            eval_start, eval_end = self.evaluation_period
            
            # Create DataFrames to hold results for different periods
            calib_df = pd.DataFrame()
            eval_df = pd.DataFrame()
            
            # Calculate metrics for calibration period if defined
            metrics = {}
            
            if calib_start and calib_end:
                # Filter to calibration period
                calib_mask = (obs_aligned.index >= calib_start) & (obs_aligned.index <= calib_end)
                calib_obs = obs_aligned[calib_mask]
                
                # Find corresponding simulated values
                common_calib_idx = calib_obs.index.intersection(sim_aligned.index)
                if len(common_calib_idx) > 0:
                    calib_df['Observed'] = calib_obs.loc[common_calib_idx]
                    calib_df['Simulated'] = sim_aligned.loc[common_calib_idx]
                    
                    # Calculate metrics for calibration period
                    calib_metrics = self._calculate_streamflow_metrics(calib_df['Observed'], calib_df['Simulated'])
                    
                    self.logger.info(f"Calibration period ({calib_start} to {calib_end}) metrics:")
                    for metric, value in calib_metrics.items():
                        self.logger.info(f"  {metric}: {value:.4f}")
                        # Add to overall metrics dictionary
                        metrics[f"Calib_{metric}"] = value
                    
                    # Add without prefix for easier access
                    metrics.update(calib_metrics)
                else:
                    self.logger.warning("No common data points in calibration period")
            
            if eval_start and eval_end:
                # Filter to evaluation period
                eval_mask = (obs_aligned.index >= eval_start) & (obs_aligned.index <= eval_end)
                eval_obs = obs_aligned[eval_mask]
                
                # Find corresponding simulated values
                common_eval_idx = eval_obs.index.intersection(sim_aligned.index)
                if len(common_eval_idx) > 0:
                    eval_df['Observed'] = eval_obs.loc[common_eval_idx]
                    eval_df['Simulated'] = sim_aligned.loc[common_eval_idx]
                    
                    # Calculate metrics for evaluation period
                    eval_metrics = self._calculate_streamflow_metrics(eval_df['Observed'], eval_df['Simulated'])
                    
                    self.logger.info(f"Evaluation period ({eval_start} to {eval_end}) metrics:")
                    for metric, value in eval_metrics.items():
                        self.logger.info(f"  {metric}: {value:.4f}")
                        # Add to overall metrics dictionary
                        metrics[f"Eval_{metric}"] = value
                else:
                    self.logger.warning("No common data points in evaluation period")
            
            # Calculate metrics for the entire period if calibration and evaluation not specified
            if not metrics:
                # Get the time period where both datasets have data
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) == 0:
                    self.logger.error("No common time period between observed and simulated flows")
                    return None
                
                # Align the datasets
                obs_all = obs_aligned.loc[common_idx]
                sim_all = sim_aligned.loc[common_idx]
                
                # Calculate performance metrics
                metrics = self._calculate_streamflow_metrics(obs_all, sim_all)
                
                self.logger.info(f"Full period metrics:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            # Create plots directory if we're using a specific output directory
            if output_dir is not None:
                plots_dir = output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Create visualization plots for this output
                self._create_streamflow_plots(obs_aligned, sim_aligned, calib_df, eval_df, metrics, plots_dir)
            
            # Save metrics to a file if using a specific output directory
            if output_dir is not None:
                metrics_file = output_dir / "metrics.csv"
                pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
                self.logger.info(f"Saved metrics to: {metrics_file}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model outputs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _create_streamflow_plots(self, obs_aligned, sim_aligned, calib_df, eval_df, metrics, plots_dir):
        """
        Create visualization plots for streamflow comparison.
        
        Args:
            obs_aligned: Observed streamflow series
            sim_aligned: Simulated streamflow series
            calib_df: DataFrame with calibration period data
            eval_df: DataFrame with evaluation period data
            metrics: Dictionary with performance metrics
            plots_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            import pandas as pd
            import numpy as np
            
            # 1. Create time series comparison plot
            plt.figure(figsize=(15, 8))
            
            # Add shaded backgrounds for calibration and evaluation periods
            if not calib_df.empty:
                calib_start = calib_df.index.min()
                calib_end = calib_df.index.max()
                plt.axvspan(calib_start, calib_end, color='lightblue', alpha=0.3, label='Calibration Period')
                
                # Plot calibration data
                plt.plot(calib_df.index, calib_df['Observed'], 'g-', linewidth=1.5, label='Observed (Calibration)')
                plt.plot(calib_df.index, calib_df['Simulated'], 'b-', linewidth=1.5, label='Simulated (Calibration)')
            
            if not eval_df.empty:
                eval_start = eval_df.index.min()
                eval_end = eval_df.index.max()
                plt.axvspan(eval_start, eval_end, color='lightsalmon', alpha=0.3, label='Evaluation Period')
                
                # Plot evaluation data
                plt.plot(eval_df.index, eval_df['Observed'], 'g--', linewidth=1.5, label='Observed (Evaluation)')
                plt.plot(eval_df.index, eval_df['Simulated'], 'b--', linewidth=1.5, label='Simulated (Evaluation)')
            
            # If no calibration or evaluation data, plot full period
            if calib_df.empty and eval_df.empty:
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) > 0:
                    plt.plot(common_idx, obs_aligned.loc[common_idx], 'g-', linewidth=1.5, label='Observed')
                    plt.plot(common_idx, sim_aligned.loc[common_idx], 'b-', linewidth=1.5, label='Simulated')
            
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Add metrics as text if available
            if metrics:
                # Format metrics text
                metrics_text = "Performance Metrics:\n"
                if any(k.startswith('Calib_') for k in metrics):
                    metrics_text += "Calibration:\n"
                    for k in ['KGE', 'NSE', 'RMSE']:
                        calib_key = f"Calib_{k}"
                        if calib_key in metrics:
                            metrics_text += f"  {k}: {metrics[calib_key]:.3f}\n"
                
                if any(k.startswith('Eval_') for k in metrics):
                    metrics_text += "Evaluation:\n"
                    for k in ['KGE', 'NSE', 'RMSE']:
                        eval_key = f"Eval_{k}"
                        if eval_key in metrics:
                            metrics_text += f"  {k}: {metrics[eval_key]:.3f}\n"
                
                if not any(k.startswith('Calib_') for k in metrics) and not any(k.startswith('Eval_') for k in metrics):
                    for k in ['KGE', 'NSE', 'RMSE']:
                        if k in metrics:
                            metrics_text += f"  {k}: {metrics[k]:.3f}\n"
                
                # Add text to plot
                plt.text(0.01, 0.99, metrics_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Combine data for scatter plot
            scatter_df = pd.DataFrame()
            if not calib_df.empty:
                calib_df_copy = calib_df.copy()
                calib_df_copy['Period'] = 'Calibration'
                scatter_df = pd.concat([scatter_df, calib_df_copy])
            
            if not eval_df.empty:
                eval_df_copy = eval_df.copy()
                eval_df_copy['Period'] = 'Evaluation'
                scatter_df = pd.concat([scatter_df, eval_df_copy])
            
            # If no calibration or evaluation data, use all data
            if scatter_df.empty:
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) > 0:
                    scatter_df = pd.DataFrame({
                        'Observed': obs_aligned.loc[common_idx],
                        'Simulated': sim_aligned.loc[common_idx],
                        'Period': 'All'
                    })
            
            # Create scatter plot with different colors for calibration and evaluation
            if 'Period' in scatter_df.columns:
                for period, color in [('Calibration', 'blue'), ('Evaluation', 'red'), ('All', 'purple')]:
                    period_df = scatter_df[scatter_df['Period'] == period]
                    if not period_df.empty:
                        plt.scatter(period_df['Observed'], period_df['Simulated'], 
                                    alpha=0.5, color=color, label=period)
            else:
                plt.scatter(scatter_df['Observed'], scatter_df['Simulated'], alpha=0.5)
            
            # Add 1:1 line
            max_val = scatter_df[['Observed', 'Simulated']].max().max()
            min_val = scatter_df[['Observed', 'Simulated']].min().min()
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 Line')
            
            # Add regression line
            from scipy import stats
            if len(scatter_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    scatter_df['Observed'], scatter_df['Simulated'])
                plt.plot([min_val, max_val], 
                        [slope * min_val + intercept, slope * max_val + intercept], 
                        'r-', label=f'Regression (r={r_value:.3f})')
            
            plt.xlabel('Observed Streamflow (m³/s)')
            plt.ylabel('Simulated Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow (Scatter Plot)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Try to set equal aspect ratio
            try:
                plt.axis('equal')
            except:
                pass
            
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Create flow duration curve
            plt.figure(figsize=(10, 8))
            
            # Find common time period for observed and simulated data
            common_idx = obs_aligned.index.intersection(sim_aligned.index)
            if len(common_idx) > 0:
                # Extract aligned data
                obs_common = obs_aligned.loc[common_idx]
                sim_common = sim_aligned.loc[common_idx]
                
                # Calculate flow duration curves
                obs_sorted = np.sort(obs_common.values)[::-1]  # Descending order
                sim_sorted = np.sort(sim_common.values)[::-1]  # Descending order
                
                # Calculate exceedance probabilities
                n_obs = len(obs_sorted)
                n_sim = len(sim_sorted)
                exc_prob_obs = np.arange(1, n_obs + 1) / (n_obs + 1)
                exc_prob_sim = np.arange(1, n_sim + 1) / (n_sim + 1)
                
                # Plot FDCs
                plt.plot(exc_prob_obs, obs_sorted, 'g-', linewidth=1.5, label='Observed')
                plt.plot(exc_prob_sim, sim_sorted, 'b-', linewidth=1.5, label='Simulated')
                
                plt.xlabel('Exceedance Probability')
                plt.ylabel('Streamflow (m³/s)')
                plt.title('Flow Duration Curves')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')  # Log scale for better visualization
                
                plt.tight_layout()
                plt.savefig(plots_dir / "flow_duration_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Created streamflow plots in {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating streamflow plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _plot_flow_duration_curve(self, calib_df, eval_df, obs_all, sim_all):
        """
        Create flow duration curve visualization.
        
        Args:
            calib_df: DataFrame with observed and simulated data for calibration period
            eval_df: DataFrame with observed and simulated data for evaluation period
            obs_all: Series with all observed data
            sim_all: Series with all simulated data
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 8))
        
        # Function to calculate and plot FDC
        def plot_fdc(obs, sim, label_prefix, color_obs, color_sim, linestyle='-'):
            if obs.empty or sim.empty:
                return
                
            # Sort flows in descending order
            obs_sorted = obs.sort_values(ascending=False)
            sim_sorted = sim.sort_values(ascending=False)
            
            # Calculate exceedance probability
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
            
            # Plot the FDCs
            plt.plot(obs_exceed, obs_sorted, color=color_obs, linestyle=linestyle, 
                     label=f'Observed ({label_prefix})')
            plt.plot(sim_exceed, sim_sorted, color=color_sim, linestyle=linestyle, 
                     label=f'Simulated ({label_prefix})')
        
        # Plot FDCs for calibration and evaluation periods if available
        if not calib_df.empty:
            plot_fdc(calib_df['Observed'], calib_df['Simulated'], 'Calibration', 'g', 'b')
        
        if not eval_df.empty:
            plot_fdc(eval_df['Observed'], eval_df['Simulated'], 'Evaluation', 'g', 'b', linestyle='--')
        
        # If no calibration or evaluation data, use all data
        if calib_df.empty and eval_df.empty:
            common_idx = obs_all.index.intersection(sim_all.index)
            if len(common_idx) > 0:
                plot_fdc(obs_all.loc[common_idx], sim_all.loc[common_idx], 'All', 'g', 'b')
        
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Exceedance Probability')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Flow Duration Curves')
        plt.grid(True, which='both', alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "flow_duration_curve_rf_optimized.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_streamflow_metrics(self, observed, simulated):
        """
        Calculate performance metrics for streamflow comparison.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        
        Returns:
            Dictionary of performance metrics
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
            return None

    def _generate_new_ensemble(self, optimized_params, iteration):
        """
        Generate a new parameter ensemble centered around the optimized parameters
        with progressively narrower sampling distribution.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration: Current iteration number (used to narrow sampling range)
        
        Returns:
            Dictionary of parameter sets for the new ensemble
        """
        import numpy as np
        import pandas as pd
        import netCDF4 as nc
        
        self.logger.info("Generating new parameter ensemble centered around optimized parameters")
        
        # Get attributes file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        # Determine how much to narrow the sampling range based on iteration
        # Start with 50% and decrease by 10% each iteration (50%, 40%, 30%, etc.)
        # but never go below 10%
        range_reduction = max(0.8 - (0.05 * iteration), 0.1)
        self.logger.info(f"Using sampling range reduction factor of {range_reduction:.2f}")
        
        try:
            # Read HRU information
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                num_hru = len(hru_ids)
                
                # Create directory for the new ensemble
                ensemble_dir = self.emulation_output_dir / f"ensemble_iteration_{iteration+1}"
                ensemble_dir.mkdir(parents=True, exist_ok=True)
                
                # Number of parameter sets to generate
                num_sets = self.config.get('EMULATION_ENSEMBLE_SIZE', 100)
                
                # Create parameter sets
                param_sets = []
                for i in range(num_sets):
                    if i == 0:
                        # First set is optimized parameters
                        param_set = optimized_params.copy()
                    elif i < 0.2 * num_sets:
                        # 20% of sets with wide exploration (random)
                        param_set = {}
                        for param_name, _ in optimized_params.items():
                            param_col = f"param_{param_name}"
                            min_val = self.param_mins.get(param_col, 0)
                            max_val = self.param_maxs.get(param_col, 1)
                            param_set[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        # Rest with focused exploration around optimal
                        param_set = {}
                        for param_name, optimal_value in optimized_params.items():
                            param_col = f"param_{param_name}"
                            min_val = self.param_mins.get(param_col, 0)
                            max_val = self.param_maxs.get(param_col, 1)
                            
                            param_range = max_val - min_val
                            narrowed_range = param_range * range_reduction
                            
                            new_min = max(min_val, optimal_value - (narrowed_range / 2))
                            new_max = min(max_val, optimal_value + (narrowed_range / 2))
                            
                            param_set[param_name] = np.random.uniform(new_min, new_max)
                    
                    param_sets.append(param_set)
                
                # Create parameter files for each set
                for i, param_set in enumerate(param_sets):
                    # Create a directory for this run
                    run_dir = ensemble_dir / f"run_{i:04d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create settings directory
                    run_settings_dir = run_dir / "settings" / "SUMMA"
                    run_settings_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create trial parameter file
                    trial_param_path = run_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                    
                    # Create NetCDF file
                    with nc.Dataset(trial_param_path, 'w', format='NETCDF4') as tp_ds:
                        # Define dimensions
                        tp_ds.createDimension('hru', num_hru)
                        
                        # Create hruId variable
                        hru_id_var = tp_ds.createVariable('hruId', 'i4', ('hru',))
                        hru_id_var[:] = hru_ids
                        hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                        hru_id_var.units = '-'
                        
                        # Add each parameter with its values for this run
                        for param_name, value in param_set.items():
                            param_var = tp_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                            # Set all HRUs to the same parameter value
                            param_var[:] = np.full(num_hru, value)
                            param_var.long_name = f"Trial value for {param_name}"
                            param_var.units = "N/A"
                        
                        # Add global attributes
                        tp_ds.description = f"SUMMA Trial Parameter file for iteration {iteration+1}, run {i:04d}"
                        tp_ds.history = f"Created on {pd.Timestamp.now().isoformat()} by CONFLUENCE RandomForestEmulator"
                        tp_ds.confluence_experiment_id = self.experiment_id
                        tp_ds.emulation_iteration = iteration + 1
                    
                    # Copy necessary settings files
                    self._copy_static_settings_files(run_settings_dir)
                    
                    # Create and modify the file manager
                    self._create_run_file_manager(i, run_dir, run_settings_dir, iteration+1)
                
                # Also save a consolidated parameter file with all parameter sets
                consolidated_file = ensemble_dir / f"parameter_sets_iteration_{iteration+1}.csv"
                pd.DataFrame(param_sets).to_csv(consolidated_file, index=False)
                
                self.logger.info(f"Generated {num_sets} parameter sets for iteration {iteration+1}")
                return param_sets
                
        except Exception as e:
            self.logger.error(f"Error generating new parameter ensemble: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _copy_static_settings_files(self, run_settings_dir):
        """
        Copy static SUMMA settings files to the run directory.
        
        Args:
            run_settings_dir: Target settings directory for this run
        """
        # Files to copy
        import shutil
        from pathlib import Path
        
        source_dir = self.project_dir / "settings" / "SUMMA"
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
            source_path = source_dir / file_name
            if source_path.exists():
                dest_path = run_settings_dir / file_name
                try:
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")

    def _create_run_file_manager(self, run_idx, run_dir, run_settings_dir, iteration):
        """
        Create a modified fileManager for a specific run.
        
        Args:
            run_idx: Index of the current run
            run_dir: Base directory for this run
            run_settings_dir: Settings directory for this run
            iteration: Current iteration number
        """
        import os
        
        source_file_manager = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not source_file_manager.exists():
            self.logger.warning(f"Original fileManager not found at {source_file_manager}")
            return
        
        # Read the original file manager
        with open(source_file_manager, 'r') as f:
            fm_lines = f.readlines()
        
        # Path to the new file manager
        new_fm_path = run_settings_dir / os.path.basename(source_file_manager)
        
        # Modify relevant lines for this run
        modified_lines = []
        for line in fm_lines:
            if "outFilePrefix" in line:
                # Update output file prefix to include iteration and run index
                prefix_parts = line.split("'")
                if len(prefix_parts) >= 3:
                    original_prefix = prefix_parts[1]
                    new_prefix = f"{self.experiment_id}_iter{iteration}_run{run_idx:04d}"
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
        
        return new_fm_path

    def _run_ensemble_simulations(self, param_sets, iteration):
        """
        Run SUMMA and MizuRoute for each parameter set in the ensemble.
        
        Args:
            param_sets: List of parameter set dictionaries
            iteration: Current iteration number
        
        Returns:
            Dictionary with simulation results and performance metrics
        """
        import concurrent.futures
        import subprocess
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        self.logger.info(f"Running ensemble simulations for iteration {iteration+1}")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Get ensemble directory
        ensemble_dir = self.emulation_output_dir / f"ensemble_iteration_{iteration+1}"
        
        # Get mizuRoute settings if needed
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        if not is_lumped:
            mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
            if mizu_path == 'default':
                mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
            else:
                mizu_path = Path(mizu_path)
            
            mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Check if we should run in parallel
        run_parallel = self.config.get('EMULATION_PARALLEL_ENSEMBLE', False)
        max_workers = self.config.get('EMULATION_MAX_WORKERS', min(16, len(param_sets)))
        
        # Function to run a single parameter set
        def run_one_set(run_idx):
            run_name = f"run_{run_idx:04d}"
            run_dir = ensemble_dir / run_name
            
            # Get fileManager path
            filemanager_path = run_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
            
            if not filemanager_path.exists():
                self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                return {
                    'run_id': run_idx,
                    'success': False,
                    'error': "FileManager not found"
                }
            
            # Ensure output directories exist
            summa_output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            summa_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = summa_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create results directory
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run SUMMA
            summa_command = f"{summa_exe} -m {filemanager_path}"
            summa_log_file = log_dir / f"{run_name}_summa.log"
            
            try:
                with open(summa_log_file, 'w') as f:
                    subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                
                self.logger.debug(f"SUMMA completed for {run_name}")
                summa_success = True
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"SUMMA failed for {run_name}: {str(e)}")
                summa_success = False
            
            # Run mizuRoute if needed and SUMMA succeeded
            mizu_success = True
            if not is_lumped and summa_success:
                mizu_output_dir = run_dir / "simulations" / self.experiment_id / "mizuRoute"
                mizu_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup mizuRoute
                mizu_settings_dir = run_dir / "settings" / "mizuRoute"
                if not mizu_settings_dir.exists():
                    mizu_settings_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy and modify mizuRoute control file
                    source_control_file = self.project_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                    
                    if source_control_file.exists():
                        mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                        
                        # Copy and modify the control file
                        with open(source_control_file, 'r') as f:
                            lines = f.readlines()
                        
                        with open(mizu_control_file, 'w') as f:
                            for line in lines:
                                if '<input_dir>' in line:
                                    # Update input directory (SUMMA output)
                                    f.write(f"<input_dir>             {summa_output_dir}/    ! Folder with runoff data from SUMMA \n")
                                elif '<ancil_dir>' in line:
                                    # Update ancillary directory
                                    source_ancil_dir = self.project_dir / "settings" / "mizuRoute"
                                    f.write(f"<ancil_dir>             {source_ancil_dir}/    ! Folder with ancillary data \n")
                                elif '<output_dir>' in line:
                                    # Update output directory
                                    f.write(f"<output_dir>            {mizu_output_dir}/    ! Folder for mizuRoute simulations \n")
                                elif '<case_name>' in line:
                                    # Update case name
                                    f.write(f"<case_name>             {self.experiment_id}_iter{iteration+1}_run{run_idx:04d}    ! Simulation case name \n")
                                elif '<fname_qsim>' in line:
                                    # Update input file name
                                    f.write(f"<fname_qsim>            {self.experiment_id}_iter{iteration+1}_run{run_idx:04d}_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                                else:
                                    f.write(line)
                        
                        # Run mizuRoute
                        mizu_command = f"{mizu_exe} {mizu_control_file}"
                        mizu_log_file = log_dir / f"{run_name}_mizuroute.log"
                        
                        try:
                            with open(mizu_log_file, 'w') as f:
                                subprocess.run(mizu_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                            
                            self.logger.debug(f"mizuRoute completed for {run_name}")
                            mizu_success = True
                        except subprocess.CalledProcessError as e:
                            self.logger.warning(f"mizuRoute failed for {run_name}: {str(e)}")
                            mizu_success = False
                    else:
                        self.logger.warning(f"mizuRoute control file not found: {source_control_file}")
                        mizu_success = False
                else:
                    self.logger.warning(f"mizuRoute settings directory not found for {run_name}")
                    mizu_success = False
            
            # Extract streamflow data
            streamflow_file = None
            if summa_success:
                streamflow_file = self._extract_run_streamflow(run_dir, run_name)
            
            # Calculate performance metrics
            metrics = None
            if streamflow_file and Path(streamflow_file).exists():
                metrics = self._calculate_performance_for_run(streamflow_file)
            
            # Return results
            return {
                'run_id': run_idx,
                'success': summa_success and mizu_success,
                'summa_success': summa_success,
                'mizu_success': mizu_success if not is_lumped else None,
                'streamflow_file': str(streamflow_file) if streamflow_file else None,
                'metrics': metrics,
                'parameters': param_sets[run_idx]
            }
        
        # Run simulations (in parallel or sequentially)
        results = []
        
        if run_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_one_set, i): i for i in range(len(param_sets))}
                
                for future in concurrent.futures.as_completed(futures):
                    run_idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Log progress periodically
                        if len(results) % 10 == 0 or len(results) == len(param_sets):
                            success_count = sum(1 for r in results if r['success'])
                            self.logger.info(f"Completed {len(results)}/{len(param_sets)} runs ({success_count} successful)")
                    except Exception as e:
                        self.logger.error(f"Error running set {run_idx}: {str(e)}")
                        results.append({
                            'run_id': run_idx,
                            'success': False,
                            'error': str(e),
                            'parameters': param_sets[run_idx]
                        })
        else:
            for i in range(len(param_sets)):
                try:
                    result = run_one_set(i)
                    results.append(result)
                    
                    # Log progress periodically
                    if (i+1) % 10 == 0 or (i+1) == len(param_sets):
                        success_count = sum(1 for r in results if r['success'])
                        self.logger.info(f"Completed {i+1}/{len(param_sets)} runs ({success_count} successful)")
                except Exception as e:
                    self.logger.error(f"Error running set {i}: {str(e)}")
                    results.append({
                        'run_id': i,
                        'success': False,
                        'error': str(e),
                        'parameters': param_sets[i]
                    })
        
        # Create a summary of results
        success_count = sum(1 for r in results if r['success'])
        self.logger.info(f"Ensemble simulation completed: {success_count}/{len(param_sets)} successful runs")
        
        # Save a summary CSV with parameters and performance metrics
        summary_data = []
        for result in results:
            result_data = {'run_id': result['run_id'], 'success': result['success']}
            
            # Add parameters
            for param_name, value in result['parameters'].items():
                result_data[f"param_{param_name}"] = value
            
            # Add metrics if available
            if result['metrics']:
                for metric_name, value in result['metrics'].items():
                    result_data[metric_name] = value
            
            summary_data.append(result_data)
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_file = ensemble_dir / f"ensemble_results_iteration_{iteration+1}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Ensemble results summary saved to {summary_file}")
        
        return results

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
            timestep_files = list(summa_output_dir.glob("*timestep.nc"))
            
            if not timestep_files:
                self.logger.warning(f"No SUMMA output files found for {run_name} in {summa_output_dir}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Use the first timestep file found
            timestep_file = timestep_files[0]
            self.logger.info(f"Using SUMMA output file: {timestep_file}")
            
            try:
                # Open the NetCDF file
                with xr.open_dataset(timestep_file) as ds:
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
                    
                    # Convert to pandas series or dataframe
                    if 'gru' in runoff_data.dims:
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

    def _create_empty_streamflow_file(self, run_dir, run_name):
        """
        Create an empty streamflow file to ensure existence for later analysis.
        
        Args:
            run_dir: Path to the run directory
            run_name: Name of the run
            
        Returns:
            Path: Path to the created file
        """
        import pandas as pd
        
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create an empty streamflow file with appropriate columns
        empty_df = pd.DataFrame(columns=['time', 'streamflow'])
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        empty_df.to_csv(output_csv, index=False)
        
        self.logger.warning(f"Created empty streamflow file at {output_csv}")
        return output_csv

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
            return None
        
    def _prepare_new_data(self, ensemble_results):
        """
        Prepare new data from ensemble results for training.
        
        Args:
            ensemble_results: List of result dictionaries from ensemble simulations
        
        Returns:
            Tuple of X and y DataFrames for model training
        """
        import pandas as pd
        import numpy as np
        
        self.logger.info("Preparing new data from ensemble results")
        
        # Create DataFrame for parameter values (X)
        param_data = []
        for result in ensemble_results:
            if result['success'] and result['metrics']:
                param_dict = {f"param_{k}": v for k, v in result['parameters'].items()}
                param_data.append(param_dict)
        
        if not param_data:
            self.logger.warning("No successful runs with metrics found")
            return pd.DataFrame(), pd.Series()
        
        X = pd.DataFrame(param_data)
        
        # Create Series for target metric (y)
        target_values = []
        for result in ensemble_results:
            if result['success'] and result['metrics']:
                if self.target_metric in result['metrics']:
                    target_values.append(result['metrics'][self.target_metric])
                else:
                    # If target metric not found, use KGE or similar
                    kge_key = next((k for k in result['metrics'] if 'KGE' in k), None)
                    if kge_key:
                        target_values.append(result['metrics'][kge_key])
                    else:
                        # Use first metric as fallback
                        target_values.append(next(iter(result['metrics'].values())))
        
        y = pd.Series(target_values)
        
        # Scale features with same scaler as original data
        if self.X_scaler is not None:
            # Ensure all columns from training are present in new data
            for col in self.X_scaler.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            
            # Ensure columns are in the same order as training
            X = X[self.X_scaler.feature_names_in_]
            
            # Scale the features
            X_scaled = pd.DataFrame(
                self.X_scaler.transform(X),
                columns=X.columns
            )
        else:
            # If no scaler yet, just return the raw data
            X_scaled = X
        
        self.logger.info(f"Prepared {len(X_scaled)} new data points with {X_scaled.shape[1]} features")
        return X_scaled, y

    def _run_with_best_parameters(self, best_params):
        """
        Run SUMMA with the best parameters from all iterations.
        
        Args:
            best_params: Dictionary of best parameter values
        
        Returns:
            True if SUMMA run was successful, False otherwise
        """
        self.logger.info("Running final model with best parameters from all iterations")
        
        # Create output directory for final run
        final_output_dir = self.emulation_output_dir / "final_optimized_run"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate trial parameters file with best parameters
        self._generate_trial_params_file(best_params, final_output_dir)
        
        # Run SUMMA
        summa_success = self.run_summa_with_optimal_parameters(final_output_dir)
        
        if summa_success:
            self.logger.info("Final optimized run completed successfully")
        else:
            self.logger.error("Final optimized run failed")
        
        return summa_success


    def _create_iteration_evolution_plots(self, iteration_results):
        """
        Create plots showing the evolution of parameters and performance across iterations.
        
        Args:
            iteration_results: List of dictionaries with results from each iteration
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        self.logger.info("Creating iteration evolution plots")
        
        # Extract data from iteration results
        iterations = [result['iteration'] for result in iteration_results]
        predicted_scores = [result['predicted_score'] for result in iteration_results]
        actual_scores = [result.get('actual_score', np.nan) for result in iteration_results]
        
        # Performance evolution plot
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, predicted_scores, 'b-o', label='Predicted Score')
        
        # Plot actual scores if available
        valid_actuals = [(i, s) for i, s in zip(iterations, actual_scores) if not np.isnan(s)]
        if valid_actuals:
            plt.plot([i for i, _ in valid_actuals], [s for _, s in valid_actuals], 'r-o', label='Actual Score')
        
        plt.xlabel('Iteration')
        plt.ylabel(f'Performance Metric ({self.target_metric})')
        plt.title(f'Performance Evolution Across Iterations')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(iterations)
        
        # Add annotations with values
        for i, (pred, act) in enumerate(zip(predicted_scores, actual_scores)):
            plt.annotate(f"{pred:.3f}", (iterations[i], pred), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
            
            if not np.isnan(act):
                plt.annotate(f"{act:.3f}", (iterations[i], act), textcoords="offset points", 
                            xytext=(0, -15), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "performance_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter evolution plot (separate plot for each parameter)
        all_params = set()
        for result in iteration_results:
            all_params.update(result['optimized_parameters'].keys())
        
        # Create a DataFrame with parameter values for each iteration
        param_df = pd.DataFrame(index=iterations)
        for param in all_params:
            param_df[param] = [result['optimized_parameters'].get(param, np.nan) for result in iteration_results]
        
        # Plot each parameter evolution
        plt.figure(figsize=(15, 10))
        num_params = len(all_params)
        nrows = (num_params + 2) // 3  # Calculate rows needed for 3 columns
        ncols = min(3, num_params)
        
        for i, param in enumerate(sorted(all_params)):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.plot(iterations, param_df[param], 'g-o')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Evolution of {param}')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
            
            # Add parameter values as annotations
            for j, val in enumerate(param_df[param]):
                if not np.isnan(val):
                    ax.annotate(f"{val:.4f}", (iterations[j], val), textcoords="offset points", 
                            xytext=(0, 5), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "parameter_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined streamflow plot for all iterations
        self._create_combined_streamflow_plot(iteration_results)
        
        self.logger.info("Iteration evolution plots created")

    def _create_combined_streamflow_plot(self, iteration_results):
        """
        Create a combined plot showing the streamflow for each iteration,
        skipping the first 10% of data to avoid showing spin-up period.
        
        Args:
            iteration_results: List of dictionaries with results from each iteration
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from matplotlib.dates import DateFormatter
        
        self.logger.info("Creating combined streamflow plot (skipping spin-up period)")
        
        # Read observed data
        obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return
        
        try:
            # Load observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date column
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data")
                return
            
            # Identify flow column
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data")
                return
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Calculate the spin-up period (first 10% of data)
            total_days = (observed_flow.index.max() - observed_flow.index.min()).days
            spinup_days = int(total_days * 0.1)  # 10% of total period
            spinup_end_date = observed_flow.index.min() + pd.Timedelta(days=spinup_days)
            
            self.logger.info(f"Skipping spin-up period: {observed_flow.index.min()} to {spinup_end_date} ({spinup_days} days)")
            
            # Filter out spin-up period
            observed_flow = observed_flow[observed_flow.index >= spinup_end_date]
            
            # Get simulated streamflow for each iteration
            sim_flows = []
            for result in iteration_results:
                iter_num = result['iteration']
                iter_dir = self.emulation_output_dir / f"iteration_{iter_num}"
                
                # Look for streamflow files
                if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                    # Use mizuRoute output
                    mizu_dir = iter_dir / "simulations" / self.experiment_id / "mizuRoute"
                    if mizu_dir.exists():
                        import xarray as xr
                        
                        # Find NetCDF files
                        nc_files = list(mizu_dir.glob("*.nc"))
                        if nc_files:
                            # Get the first file
                            mizu_file = nc_files[0]
                            
                            # Get reach ID
                            sim_reach_id = self.config.get('SIM_REACH_ID')
                            
                            try:
                                with xr.open_dataset(mizu_file) as ds:
                                    # Find the index for the reach ID
                                    if 'reachID' in ds.variables:
                                        reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                                        
                                        if len(reach_indices) > 0:
                                            reach_index = reach_indices[0]
                                            
                                            # Try common variable names
                                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                                if var_name in ds.variables:
                                                    flow_series = ds[var_name].isel(seg=reach_index).to_pandas()
                                                    # Filter out spin-up period
                                                    flow_series = flow_series[flow_series.index >= spinup_end_date]
                                                    sim_flows.append((iter_num, flow_series))
                                                    break
                            except Exception as e:
                                self.logger.warning(f"Error reading mizuRoute output for iteration {iter_num}: {str(e)}")
                else:
                    # Use SUMMA output directly
                    summa_dir = iter_dir / "simulations" / self.experiment_id / "SUMMA"
                    if summa_dir.exists():
                        import xarray as xr
                        
                        # Find NetCDF files
                        nc_files = list(summa_dir.glob(f"{self.experiment_id}*timestep.nc"))
                        if nc_files:
                            # Get the first file
                            summa_file = nc_files[0]
                            
                            try:
                                with xr.open_dataset(summa_file) as ds:
                                    # Try to find streamflow variable
                                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                                        if var_name in ds.variables:
                                            if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                                # Sum across GRUs if multiple
                                                flow_series = ds[var_name].sum(dim='gru').to_pandas()
                                            else:
                                                # Single GRU or no gru dimension
                                                flow_series = ds[var_name].to_pandas()
                                                # Handle case if it's a DataFrame
                                                if isinstance(flow_series, pd.DataFrame):
                                                    flow_series = flow_series.iloc[:, 0]
                                            
                                            # Get catchment area
                                            catchment_area = self._get_catchment_area()
                                            if catchment_area and catchment_area > 0:
                                                # Convert from m/s to m³/s
                                                flow_series = flow_series * catchment_area
                                            
                                            # Filter out spin-up period
                                            flow_series = flow_series[flow_series.index >= spinup_end_date]
                                            sim_flows.append((iter_num, flow_series))
                                            break
                            except Exception as e:
                                self.logger.warning(f"Error reading SUMMA output for iteration {iter_num}: {str(e)}")
            
            if not sim_flows:
                self.logger.warning("No simulation outputs found to plot")
                return
            
            # Create plot
            plt.figure(figsize=(16, 10))
            
            # Plot observed flow
            plt.plot(observed_flow.index, observed_flow, 'k-', linewidth=2, label='Observed')
            
            # Plot simulated flows for each iteration
            colors = plt.cm.viridis(np.linspace(0, 1, len(sim_flows)))
            for i, (iter_num, flow_series) in enumerate(sim_flows):
                plt.plot(flow_series.index, flow_series, '-', color=colors[i], linewidth=1.5, 
                        label=f'Iteration {iter_num}')
            
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Streamflow Comparison Across Iterations (Spin-up Period Excluded)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(self.emulation_output_dir / "streamflow_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a DataFrame with all flows for further analysis
            all_flows = pd.DataFrame(index=observed_flow.index)
            all_flows['Observed'] = observed_flow
            
            for iter_num, flow_series in sim_flows:
                # Align index with observed data
                reindexed_flow = flow_series.reindex(observed_flow.index, method='nearest')
                all_flows[f'Iteration_{iter_num}'] = reindexed_flow
            
            # Add a column for the final optimized run if available
            final_dir = self.emulation_output_dir / "final_optimized_run"
            if final_dir.exists():
                final_flow = self._get_final_optimized_flow(final_dir)
                if final_flow is not None:
                    # Filter out spin-up period
                    final_flow = final_flow[final_flow.index >= spinup_end_date]
                    # Align index with observed data
                    reindexed_final = final_flow.reindex(observed_flow.index, method='nearest')
                    all_flows['Final_Optimized'] = reindexed_final
            
            # Save flow data for potential further analysis
            all_flows.to_csv(self.emulation_output_dir / "all_streamflows.csv")
            
            self.logger.info("Combined streamflow plot created (spin-up period excluded)")
            
        except Exception as e:
            self.logger.error(f"Error creating combined streamflow plot: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _get_final_optimized_flow(self, final_dir, exclude_spinup=True):
        """
        Extract the streamflow from the final optimized run.
        
        Args:
            final_dir: Directory for the final optimized run
            exclude_spinup: Whether to exclude the spin-up period
        
        Returns:
            Series with the streamflow data, or None if not found
        """
        try:
            # Calculate spin-up period end date if needed
            spinup_end_date = None
            if exclude_spinup:
                # Get observed data to determine the date range
                obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
                if obs_path.exists():
                    obs_df = pd.read_csv(obs_path)
                    date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
                    if date_col:
                        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
                        total_days = (obs_df['DateTime'].max() - obs_df['DateTime'].min()).days
                        spinup_days = int(total_days * 0.1)  # 10% of total period
                        spinup_end_date = obs_df['DateTime'].min() + pd.Timedelta(days=spinup_days)
            
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # Use mizuRoute output
                mizu_dir = final_dir / "simulations" / self.experiment_id / "mizuRoute"
                if mizu_dir.exists():
                    import xarray as xr
                    
                    # Find NetCDF files
                    nc_files = list(mizu_dir.glob("*.nc"))
                    if nc_files:
                        # Get the first file
                        mizu_file = nc_files[0]
                        
                        # Get reach ID
                        sim_reach_id = self.config.get('SIM_REACH_ID')
                        
                        with xr.open_dataset(mizu_file) as ds:
                            # Find the index for the reach ID
                            if 'reachID' in ds.variables:
                                reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                                
                                if len(reach_indices) > 0:
                                    reach_index = reach_indices[0]
                                    
                                    # Try common variable names
                                    for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                        if var_name in ds.variables:
                                            flow_series = ds[var_name].isel(seg=reach_index).to_pandas()
                                            # Filter out spin-up period if requested
                                            if exclude_spinup and spinup_end_date is not None:
                                                flow_series = flow_series[flow_series.index >= spinup_end_date]
                                            return flow_series
            else:
                # Use SUMMA output directly
                summa_dir = final_dir / "simulations" / self.experiment_id / "SUMMA"
                if summa_dir.exists():
                    import xarray as xr
                    
                    # Find NetCDF files
                    nc_files = list(summa_dir.glob(f"{self.experiment_id}*timestep.nc"))
                    if nc_files:
                        # Get the first file
                        summa_file = nc_files[0]
                        
                        with xr.open_dataset(summa_file) as ds:
                            # Try to find streamflow variable
                            for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                                if var_name in ds.variables:
                                    if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                        # Sum across GRUs if multiple
                                        flow_series = ds[var_name].sum(dim='gru').to_pandas()
                                    else:
                                        # Single GRU or no gru dimension
                                        flow_series = ds[var_name].to_pandas()
                                        # Handle case if it's a DataFrame
                                        if isinstance(flow_series, pd.DataFrame):
                                            flow_series = flow_series.iloc[:, 0]
                                    
                                    # Get catchment area
                                    catchment_area = self._get_catchment_area()
                                    if catchment_area and catchment_area > 0:
                                        # Convert from m/s to m³/s
                                        flow_series = flow_series * catchment_area
                                    
                                    # Filter out spin-up period if requested
                                    if exclude_spinup and spinup_end_date is not None:
                                        flow_series = flow_series[flow_series.index >= spinup_end_date]
                                        
                                    return flow_series
            
            return None
        except Exception as e:
            self.logger.warning(f"Error getting final optimized flow: {str(e)}")
            return None

    def _calculate_performance_for_run(self, streamflow_file):
        """
        Calculate performance metrics for a run by comparing to observed data.
        
        Args:
            streamflow_file: Path to the streamflow CSV file
        
        Returns:
            Dictionary of performance metrics
        """
        import pandas as pd
        import numpy as np
        
        # Get observed data path
        obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date column
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data")
                return None
            
            # Identify flow column
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Read simulated data
            sim_df = pd.read_csv(streamflow_file)
            
            # Identify time column
            time_col = next((col for col in sim_df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
            if not time_col:
                self.logger.warning(f"No time column found in simulated data")
                return None
            
            # Identify flow column
            sim_flow_col = next((col for col in sim_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'streamflow' in col.lower()), None)
            if not sim_flow_col:
                self.logger.warning(f"No flow column found in simulated data")
                return None
            
            # Convert time column to datetime and set as index
            sim_df['DateTime'] = pd.to_datetime(sim_df[time_col])
            sim_df.set_index('DateTime', inplace=True)
            simulated_flow = sim_df[sim_flow_col]
            
            # Filter data to calibration and evaluation periods if defined
            calib_start, calib_end = self.calibration_period
            eval_start, eval_end = self.evaluation_period
            
            # Create dictionaries to hold metrics
            calib_metrics = {}
            eval_metrics = {}
            all_metrics = {}
            
            # Calculate metrics for calibration period if defined
            if calib_start and calib_end:
                calib_mask = (observed_flow.index >= calib_start) & (observed_flow.index <= calib_end)
                calib_obs = observed_flow[calib_mask]
                
                # Find corresponding simulated values
                common_calib_idx = calib_obs.index.intersection(simulated_flow.index)
                if len(common_calib_idx) > 0:
                    calib_obs_aligned = calib_obs.loc[common_calib_idx]
                    calib_sim_aligned = simulated_flow.loc[common_calib_idx]
                    
                    # Calculate metrics
                    calib_metrics = self._calculate_streamflow_metrics(calib_obs_aligned, calib_sim_aligned)
                    for key, value in calib_metrics.items():
                        all_metrics[f"Calib_{key}"] = value
                    
                    # Always include KGE for convenience
                    if "Calib_KGE" in all_metrics:
                        all_metrics["KGE"] = all_metrics["Calib_KGE"]
            
            # Calculate metrics for evaluation period if defined
            if eval_start and eval_end:
                eval_mask = (observed_flow.index >= eval_start) & (observed_flow.index <= eval_end)
                eval_obs = observed_flow[eval_mask]
                
                # Find corresponding simulated values
                common_eval_idx = eval_obs.index.intersection(simulated_flow.index)
                if len(common_eval_idx) > 0:
                    eval_obs_aligned = eval_obs.loc[common_eval_idx]
                    eval_sim_aligned = simulated_flow.loc[common_eval_idx]
                    
                    # Calculate metrics
                    eval_metrics = self._calculate_streamflow_metrics(eval_obs_aligned, eval_sim_aligned)
                    for key, value in eval_metrics.items():
                        all_metrics[f"Eval_{key}"] = value
                    
                    # If no calibration metrics, use evaluation KGE
                    if "KGE" not in all_metrics and "Eval_KGE" in all_metrics:
                        all_metrics["KGE"] = all_metrics["Eval_KGE"]
            
            # If no calibration or evaluation periods defined, use all data
            if not all_metrics:
                # Find common time period
                common_idx = observed_flow.index.intersection(simulated_flow.index)
                if len(common_idx) > 0:
                    obs_aligned = observed_flow.loc[common_idx]
                    sim_aligned = simulated_flow.loc[common_idx]
                    
                    # Calculate metrics
                    all_metrics = self._calculate_streamflow_metrics(obs_aligned, sim_aligned)
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _run_summa_for_iteration(self, optimized_params, iteration_output_dir):
        """
        Run SUMMA with optimized parameters for a specific iteration.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            bool: True if SUMMA run was successful, False otherwise
        """
        self.logger.info(f"Running SUMMA with optimized parameters in {iteration_output_dir}")
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all necessary SUMMA settings files first
        source_settings_dir = self.project_dir / "settings" / "SUMMA"
        self._copy_settings_files(source_settings_dir, settings_dir)
        
        # Generate trial parameters file specifically for this iteration
        trial_params_path = self._generate_trial_params_file_for_iteration(optimized_params, iteration_output_dir)
        if not trial_params_path:
            self.logger.error("Could not generate trial parameters file for iteration")
            return False
        
        # Define paths
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Create a modified fileManager for this iteration
        modified_filemanager = self._create_iteration_file_manager(iteration_output_dir)
        if not modified_filemanager:
            self.logger.error("Could not create modified fileManager for iteration")
            return False
        
        # Create output directory
        sim_dir = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = sim_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Run SUMMA
        summa_command = f"{summa_exe} -m {modified_filemanager}"
        log_file = log_dir / "summa_rf_optimized.log"
        
        try:
            # Run SUMMA and capture output
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    summa_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"SUMMA run completed successfully for iteration, log saved to: {log_file}")
            
            # Run mizuRoute if needed
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                self._run_mizuroute_for_iteration(iteration_output_dir)
            
            # Evaluate output for this iteration
            metrics = self._evaluate_summa_output(iteration_output_dir)  # Pass the output directory
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running SUMMA for iteration: {str(e)}")
            return False

    def _copy_settings_files(self, source_dir, target_dir):
        """
        Copy all necessary settings files from source to target directory.
        
        Args:
            source_dir: Source directory with original settings files
            target_dir: Target directory to copy files to
        """
        self.logger.info(f"Copying settings files from {source_dir} to {target_dir}")
        
        # Files to copy
        files_to_copy = [
            'outputControl.txt',
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
        
        for file_name in files_to_copy:
            source_file = source_dir / file_name
            target_file = target_dir / file_name
            
            if source_file.exists():
                try:
                    import shutil
                    shutil.copy2(source_file, target_file)
                    self.logger.debug(f"Copied {file_name} to {target_dir}")
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")
            else:
                self.logger.warning(f"Source file {source_file} not found")
        
    def _generate_trial_params_file_for_iteration(self, optimized_params, iteration_output_dir):
        """
        Generate a trialParams.nc file with the optimized parameters for a specific iteration.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            Path: Path to the generated file
        """
        self.logger.info(f"Generating trial parameters file for iteration in {iteration_output_dir}")
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Get attribute file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        try:
            # Read HRU information from the attributes file
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                
                # Create the trial parameters dataset
                output_path = settings_dir / "trialParams.nc"
                output_ds = nc.Dataset(output_path, 'w', format='NETCDF4')
                
                # Add dimensions
                output_ds.createDimension('hru', len(hru_ids))
                
                # Add HRU ID variable
                hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Add parameter variables
                for param_name, param_value in optimized_params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Create variable
                    param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                    param_var[:] = param_array
                    param_var.long_name = f"Trial value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE RandomForestEmulator"
                output_ds.history = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.confluence_experiment_id = self.experiment_id
                
                # Close the file explicitly
                output_ds.close()
                
                self.logger.info(f"Generated trial parameters file: {output_path}")
                return output_path
                    
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _create_iteration_file_manager(self, iteration_output_dir):
        """
        Create a modified fileManager for a specific iteration.
        
        Args:
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            Path: Path to the modified fileManager
        """
        source_file_manager = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not source_file_manager.exists():
            self.logger.error(f"Original fileManager not found at {source_file_manager}")
            return None
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the new file manager
        new_fm_path = settings_dir / os.path.basename(source_file_manager)
        
        try:
            # Read the original file manager
            with open(source_file_manager, 'r') as f:
                lines = f.readlines()
            
            # Modify relevant lines
            modified_lines = []
            for line in lines:
                if "outFilePrefix" in line:
                    # Update output file prefix
                    modified_line = f"outFilePrefix        '{self.experiment_id}_rf_iter'\n"
                    modified_lines.append(modified_line)
                elif "outputPath" in line:
                    # Update output path
                    output_path = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                    output_path_str = str(output_path).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    modified_line = f"outputPath           '{output_path_str}/'\n"
                    modified_lines.append(modified_line)
                elif "settingsPath" in line:
                    # Update settings path
                    settings_path_str = str(settings_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    modified_line = f"settingsPath         '{settings_path_str}/'\n"
                    modified_lines.append(modified_line)
                elif "trialParamFile" in line:
                    # Use the iteration-specific trial params file
                    modified_line = f"trialParamFile       'trialParams.nc'\n"
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            
            # Write the modified file manager
            with open(new_fm_path, 'w') as f:
                f.writelines(modified_lines)
                
            self.logger.info(f"Created modified fileManager for iteration: {new_fm_path}")
            return new_fm_path
            
        except Exception as e:
            self.logger.error(f"Error creating modified fileManager for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _run_mizuroute_for_iteration(self, iteration_output_dir):
        """
        Run mizuRoute with the SUMMA outputs for a specific iteration.
        
        Args:
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            bool: True if mizuRoute run was successful, False otherwise
        """
        self.logger.info(f"Running mizuRoute for iteration in {iteration_output_dir}")
        
        # Get mizuRoute executable path
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Create settings directory
        mizu_settings_dir = iteration_output_dir / "settings" / "mizuRoute"
        mizu_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and modify mizuRoute control file
        source_control_file = self.project_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not source_control_file.exists():
            self.logger.error(f"Source mizuRoute control file not found: {source_control_file}")
            return False
        
        # Modified control file path
        control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        try:
            # Read the original control file
            with open(source_control_file, 'r') as f:
                lines = f.readlines()
            
            # Create output directory
            mizu_output_dir = iteration_output_dir / "simulations" / self.experiment_id / "mizuRoute"
            mizu_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = mizu_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Modify the control file
            with open(control_file, 'w') as f:
                for line in lines:
                    if '<input_dir>' in line:
                        # Update input directory (SUMMA output)
                        summa_output_dir = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                        summa_output_str = str(summa_output_dir).replace('\\', '/')
                        f.write(f"<input_dir>             {summa_output_str}/    ! Folder that contains runoff data from SUMMA \n")
                    elif '<ancil_dir>' in line:
                        # Use the original ancillary directory
                        source_ancil_dir = self.project_dir / "settings" / "mizuRoute"
                        source_ancil_str = str(source_ancil_dir).replace('\\', '/')
                        f.write(f"<ancil_dir>             {source_ancil_str}/    ! Folder that contains ancillary data \n")
                    elif '<output_dir>' in line:
                        # Update output directory
                        mizu_output_str = str(mizu_output_dir).replace('\\', '/')
                        f.write(f"<output_dir>            {mizu_output_str}/    ! Folder that will contain mizuRoute simulations \n")
                    elif '<case_name>' in line:
                        # Update case name
                        f.write(f"<case_name>             {self.experiment_id}_rf_iter    ! Simulation case name \n")
                    elif '<fname_qsim>' in line:
                        # Update input file name
                        f.write(f"<fname_qsim>            {self.experiment_id}_rf_iter_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                    else:
                        f.write(line)
            
            # Run mizuRoute
            mizu_command = f"{mizu_exe} {control_file}"
            log_file = log_dir / "mizuroute_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"mizuRoute run completed successfully for iteration, log saved to: {log_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running mizuRoute for iteration: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error setting up or running mizuRoute for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def run_workflow(self):
        """
        Run the complete random forest emulation workflow with iterative refinement:
        1. Load and prepare initial data
        2. For each iteration:
        a. Train the random forest model with three-way split
        b. Optimize parameters
        c. Generate new parameter ensemble centered around optimized parameters
        d. Run models with new ensemble and calculate performance metrics
        e. Append new results to the dataset
        3. Run final SUMMA with best optimized parameters
        4. Evaluate and visualize results across iterations
        
        Returns:
            Dictionary with workflow results
        """
        self.logger.info("Starting random forest emulation workflow with iterative refinement")
        
        # Get iteration settings from config
        max_iterations = self.config.get('EMULATION_ITERATIONS', 3)
        stop_criteria = self.config.get('EMULATION_STOP_CRITERIA', 0.01)
        
        # Track best score and parameters across iterations
        best_score = float('-inf')
        best_params = None
        iteration_results = []
        
        try:
            # Step 1: Load and prepare initial data
            X, y = self.load_and_prepare_data()
            self.logger.info(f"Initial data prepared with {X.shape[1]} features and {len(y)} samples")
            
            # Initialize tracking for iteration metrics
            iteration_metrics = []
            
            # Iterate until max iterations or convergence
            for iteration in range(max_iterations):
                self.logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
                
                # Step 2a: Train the random forest model
                model, metrics = self.train_random_forest(X, y)
                self.logger.info(f"Model trained with R² scores - Test: {metrics['test_score']:.4f}")
                
                # Step 2b: Optimize parameters
                optimized_params, predicted_score = self.optimize_parameters()
                self.logger.info(f"Parameters optimized with predicted score: {predicted_score:.4f}")
                
                # Save this iteration's results
                iteration_result = {
                    'iteration': iteration + 1,
                    'model_metrics': metrics,
                    'optimized_parameters': optimized_params,
                    'predicted_score': predicted_score
                }
                iteration_results.append(iteration_result)
                
                # Run SUMMA with optimized parameters for this iteration
                iteration_output_dir = self.emulation_output_dir / f"iteration_{iteration+1}"
                iteration_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a modified version of run_summa_with_optimal_parameters for this iteration
                summa_success = self._run_summa_for_iteration(optimized_params, iteration_output_dir)
                
                if summa_success:
                    # Get actual performance from this run
                    actual_metrics = self._evaluate_summa_output(iteration_output_dir)
                    if actual_metrics:
                        actual_score = actual_metrics.get(self.target_metric, float('-inf'))
                        iteration_result['actual_score'] = actual_score
                        
                        # Track if this is the best result so far
                        if actual_score > best_score:
                            best_score = actual_score
                            best_params = optimized_params.copy()
                            self.logger.info(f"New best score: {best_score:.4f} in iteration {iteration+1}")
                    
                # Check for convergence
                if iteration > 0:
                    prev_score = iteration_results[iteration-1].get('actual_score', 
                                                            iteration_results[iteration-1]['predicted_score'])
                    current_score = iteration_result.get('actual_score', predicted_score)
                    improvement = abs(current_score - prev_score)
                    
                    self.logger.info(f"Improvement from previous iteration: {improvement:.4f}")
                    if improvement < stop_criteria:
                        self.logger.info(f"Stopping due to convergence (improvement < {stop_criteria})")
                        break
                
                # Step 2c: Generate new parameter ensemble centered around optimized parameters
                if iteration < max_iterations - 1:  # Skip on last iteration
                    self.logger.info("Generating new parameter ensemble for next iteration")
                    new_param_sets = self._generate_new_ensemble(optimized_params, iteration)
                    
                    # Step 2d: Run models with new ensemble
                    self.logger.info("Running models with new parameter ensemble")
                    new_results = self._run_ensemble_simulations(new_param_sets, iteration)
                    
                    # Step 2e: Append new results to dataset for next iteration
                    X_new, y_new = self._prepare_new_data(new_results)
                    X = pd.concat([X, X_new], ignore_index=True)
                    y = pd.concat([y, y_new], ignore_index=True)
                    self.logger.info(f"Dataset expanded to {len(y)} samples")
            
            # Use the best parameters from all iterations for final run
            if best_params:
                self.logger.info(f"Running final model with best parameters (score: {best_score:.4f})")
                final_success = self.run_summa_with_optimal_parameters()
            else:
                self.logger.info("Using parameters from final iteration for final model")
                final_success = True  # Already ran in last iteration
            
            # Generate summary visualizations across iterations
            self._create_iteration_evolution_plots(iteration_results)
            
            # Return results
            results = {
                'iterations': iteration_results,
                'best_score': best_score,
                'best_parameters': best_params,
                'final_success': final_success,
                'output_dir': str(self.emulation_output_dir)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in random forest emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from rasterstats import zonal_stats
import rasterio
from osgeo import gdal

from scipy.stats import skew, kurtosis, circmean, circstd
from shapely.geometry import box
from rasterio.mask import mask

class attributeProcessor:
    """
    Simple attribute processor that calculates elevation, slope, and aspect 
    statistics from an existing DEM using GDAL.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the attribute processor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.logger.info(f'data dir: {self.data_dir}')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Create or access directories
        self.dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        self.slope_dir = self.project_dir / 'attributes' / 'elevation' / 'slope'
        self.aspect_dir = self.project_dir / 'attributes' / 'elevation' / 'aspect'
        
        # Create directories if they don't exist
        self.slope_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()
        
        # Initialize results dictionary
        self.results = {}
    
    def _get_catchment_path(self) -> Path:
        """Get the path to the catchment shapefile."""
        catchment_path = self.config.get('CATCHMENT_PATH')
        self.logger.info(f'catchment path: {catchment_path}')
        
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        self.logger.info(f'catchment name: {catchment_name}')
        

        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)
        
        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name
        
        return catchment_path / catchment_file
    
    def find_dem_file(self) -> Path:
        """Find the DEM file in the elevation/dem directory."""
        dem_files = list(self.dem_dir.glob("*.tif"))
        
        if not dem_files:
            self.logger.error(f"No DEM files found in {self.dem_dir}")
            raise FileNotFoundError(f"No DEM files found in {self.dem_dir}")
        
        # Use the first found DEM file
        return dem_files[0]
    
    def generate_slope_and_aspect(self, dem_file: Path) -> Dict[str, Path]:
        """Generate slope and aspect rasters from the DEM using GDAL."""
        self.logger.info(f"Generating slope and aspect from DEM: {dem_file}")
        
        # Create output file paths
        slope_file = self.slope_dir / f"{dem_file.stem}_slope.tif"
        aspect_file = self.aspect_dir / f"{dem_file.stem}_aspect.tif"
        
        try:
            # Prepare the slope options
            slope_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                slopeFormat='degree',
                alg='Horn'
            )
            
            # Generate slope
            self.logger.info("Calculating slope...")
            gdal.DEMProcessing(
                str(slope_file),
                str(dem_file),
                'slope',
                options=slope_options
            )
            
            # Prepare the aspect options
            aspect_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                alg='Horn',
                zeroForFlat=True
            )
            
            # Generate aspect
            self.logger.info("Calculating aspect...")
            gdal.DEMProcessing(
                str(aspect_file),
                str(dem_file),
                'aspect',
                options=aspect_options
            )
            
            self.logger.info(f"Slope saved to: {slope_file}")
            self.logger.info(f"Aspect saved to: {aspect_file}")
            
            return {
                'dem': dem_file,
                'slope': slope_file,
                'aspect': aspect_file
            }
        
        except Exception as e:
            self.logger.error(f"Error generating slope and aspect: {str(e)}")
            raise
    
    def calculate_statistics(self, raster_file: Path, attribute_name: str) -> Dict[str, float]:
        """Calculate zonal statistics for a raster."""
        self.logger.info(f"Calculating statistics for {attribute_name} from {raster_file}")
        
        # Define statistics to calculate
        stats = ['min', 'mean', 'max', 'std']
        
        # Special handling for aspect (circular statistics)
        if attribute_name == 'aspect':
            def calc_circmean(x):
                """Calculate circular mean of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular mean, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circmean(rad))
                return result
            
            def calc_circstd(x):
                """Calculate circular standard deviation of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular std, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circstd(rad))
                return result
            
            # Add custom circular statistics
            stats = ['min', 'max']
            custom_stats = {
                'circmean': calc_circmean
            }
            
            try:
                from scipy.stats import circstd
                custom_stats['circstd'] = calc_circstd
            except ImportError:
                self.logger.warning("scipy.stats.circstd not available, skipping circular std")
            
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats,
                add_stats=custom_stats, 
                all_touched=True
            )
        else:
            # Regular statistics for elevation and slope
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats, 
                all_touched=True
            )
        
        # Format the results
        results = {}
        if zonal_out:
            for i, zonal_result in enumerate(zonal_out):
                # Use HRU ID as key if available, otherwise use index
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                if not is_lumped:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    hru_id = catchment.iloc[i][hru_id_field]
                    key_prefix = f"HRU_{hru_id}_"
                else:
                    key_prefix = ""
                
                # Add each statistic to results
                for stat, value in zonal_result.items():
                    if value is not None:
                        results[f"{key_prefix}{attribute_name}_{stat}"] = value
        
        return results

    def process_attributes(self) -> pd.DataFrame:
        """
        Process  catchment attributes from  available data sources,
        creating an integrated dataset of hydrologically relevant basin characteristics.
        
        Returns:
            pd.DataFrame: Complete dataframe of catchment attributes
        """
        self.logger.info("Starting attribute processing")
        
        try:
            # Initialize results dictionary
            all_results = {}
            
            # Process base elevation attributes (DEM, slope, aspect)
            self.logger.info("Processing elevation attributes")
            elevation_results = self._process_elevation_attributes()
            all_results.update(elevation_results)
            
            # Process soil properties
            self.logger.info("Processing soil attributes")
            soil_results = self._process_soil_attributes()
            all_results.update(soil_results)
            
            # Process geological attributes
            self.logger.info("Processing geological attributes")
            geo_results = self._process_geological_attributes()
            all_results.update(geo_results)
            
            # Enhance geological attributes with derived hydraulic properties
            geo_enhanced = self.enhance_hydrogeological_attributes(geo_results)
            all_results.update(geo_enhanced)
            
            # Process land cover attributes
            self.logger.info("Processing land cover attributes")
            lc_results = self._process_landcover_attributes()
            all_results.update(lc_results)
            
            # Process vegetation attributes (LAI, forest height)
            self.logger.info("Processing vegetation attributes")
            veg_results = self._process_forest_height()
            all_results.update(veg_results)
            
            # Process composite land cover metrics
            self.logger.info("Processing composite land cover metrics")
            self.results = all_results  # Temporarily set results for composite metrics calculation
            composite_lc = self.calculate_composite_landcover_metrics()
            all_results.update(composite_lc)
            
            # Process climate attributes
            self.logger.info("Processing climate attributes")
            climate_results = self._process_climate_attributes()
            all_results.update(climate_results)
            
            # Process advanced seasonality metrics
            self.logger.info("Processing seasonality metrics")
            seasonality_results = self.calculate_seasonality_metrics()
            all_results.update(seasonality_results)
            
            # Process hydrological attributes
            self.logger.info("Processing hydrological attributes")
            hydro_results = self._process_hydrological_attributes()
            all_results.update(hydro_results)
            
            # Process enhanced river network analysis
            self.logger.info("Processing enhanced river network")
            river_results = self.enhance_river_network_analysis()
            all_results.update(river_results)
            
            # Process baseflow attributes
            self.logger.info("Processing baseflow attributes")
            baseflow_results = self.calculate_baseflow_attributes()
            all_results.update(baseflow_results)
            
            # Process streamflow signatures
            self.logger.info("Processing streamflow signatures")
            signature_results = self.calculate_streamflow_signatures()
            all_results.update(signature_results)
            
            # Process water balance
            self.logger.info("Processing water balance")
            wb_results = self.calculate_water_balance()
            all_results.update(wb_results)
            
            # Create final dataframe
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, create a single row
                df = pd.DataFrame([all_results])
                df['basin_id'] = self.domain_name
                df = df.set_index('basin_id')
            else:
                # For distributed catchment, create multi-level DataFrame with HRUs
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                hru_ids = catchment[hru_id_field].values
                
                # Reorganize results by HRU
                hru_results = []
                for i, hru_id in enumerate(hru_ids):
                    result = {'hru_id': hru_id}
                    
                    for key, value in all_results.items():
                        if key.startswith(f"HRU_{hru_id}_"):
                            # Remove HRU prefix
                            clean_key = key.replace(f"HRU_{hru_id}_", "")
                            result[clean_key] = value
                        elif not any(k.startswith("HRU_") for k in all_results.keys()):
                            # If there are no HRU-specific results, use the same values for all HRUs
                            result[key] = value
                    
                    hru_results.append(result)
                    
                df = pd.DataFrame(hru_results)
                df['basin_id'] = self.domain_name
                df = df.set_index(['basin_id', 'hru_id'])
            
            # Save the  attributes to different formats
            output_dir = self.project_dir / 'attributes'
            
            # Use prefix to distinguish from basic attribute files
            csv_file = output_dir / f"{self.domain_name}_attributes.csv"
            df.to_csv(csv_file)
            
            # Save as Parquet
            try:
                parquet_file = output_dir / f"{self.domain_name}_attributes.parquet"
                df.to_parquet(parquet_file)
                self.logger.info(f" attributes saved as Parquet: {parquet_file}")
            except ImportError:
                self.logger.warning("pyarrow not installed, skipping Parquet output")
            
            # Save as pickle
            pickle_file = output_dir / f"{self.domain_name}_attributes.pkl"
            df.to_pickle(pickle_file)
            
            self.logger.info(f" attributes saved to {csv_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in attribute processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Try to return whatever we processed so far
            if 'all_results' in locals() and all_results:
                try:
                    return pd.DataFrame([all_results])
                except:
                    pass
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()


    def _process_geological_attributes(self) -> Dict[str, Any]:
        """
        Process geological attributes including bedrock permeability, porosity, and lithology.
        Extracts data from GLHYMPS and other geological datasets.
        
        Returns:
            Dict[str, Any]: Dictionary of geological attributes
        """
        results = {}
        
        # Process GLHYMPS data for permeability and porosity
        glhymps_results = self._process_glhymps_data()
        results.update(glhymps_results)
        
        # Process additional lithology data if available
        litho_results = self._process_lithology_data()
        results.update(litho_results)
        
        return results

    def _process_glhymps_data(self) -> Dict[str, Any]:
        """
        Process GLHYMPS (Global Hydrogeology MaPS) data for permeability and porosity.
        Optimized for performance with spatial filtering and simplified geometries.
        
        Returns:
            Dict[str, Any]: Dictionary of hydrogeological attributes
        """
        results = {}
        
        # Define path to GLHYMPS data
        glhymps_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/glhymps/raw/glhymps.shp")
        
        # Check if GLHYMPS file exists
        if not glhymps_path.exists():
            self.logger.warning(f"GLHYMPS file not found: {glhymps_path}")
            return results
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'glhymps'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glhymps_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached GLHYMPS results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached GLHYMPS results: {str(e)}")
        
        self.logger.info("Processing GLHYMPS data for hydrogeological attributes")
        
        try:
            # Get catchment bounding box first to filter GLHYMPS data
            catchment = gpd.read_file(self.catchment_path)
            catchment_bbox = catchment.total_bounds
            
            # Add buffer to bounding box (e.g., 1% of width/height)
            bbox_width = catchment_bbox[2] - catchment_bbox[0]
            bbox_height = catchment_bbox[3] - catchment_bbox[1]
            buffer_x = bbox_width * 0.01
            buffer_y = bbox_height * 0.01
            
            # Create expanded bbox
            expanded_bbox = (
                catchment_bbox[0] - buffer_x,
                catchment_bbox[1] - buffer_y,
                catchment_bbox[2] + buffer_x,
                catchment_bbox[3] + buffer_y
            )
            
            # Load only GLHYMPS data within the bounding box and only necessary columns
            self.logger.info("Loading GLHYMPS data within catchment bounding box")
            glhymps = gpd.read_file(
                str(glhymps_path), 
                bbox=expanded_bbox,
                # Only read necessary columns if they exist
                columns=['geometry', 'porosity', 'logK_Ice', 'Porosity', 'Permeabi_1']
            )
            
            if glhymps.empty:
                self.logger.warning("No GLHYMPS data within catchment bounding box")
                return results
            
            # Rename columns if they exist with shortened names
            if ('Porosity' in glhymps.columns) and ('Permeabi_1' in glhymps.columns):
                glhymps.rename(columns={'Porosity': 'porosity', 'Permeabi_1': 'logK_Ice'}, inplace=True)
            
            # Check for missing required columns
            required_columns = ['porosity', 'logK_Ice']
            missing_columns = [col for col in required_columns if col not in glhymps.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns in GLHYMPS: {', '.join(missing_columns)}")
                return results
            
            # Simplify geometries for faster processing
            self.logger.info("Simplifying geometries for faster processing")
            simplify_tolerance = 0.001  # Adjust based on your data units
            glhymps['geometry'] = glhymps.geometry.simplify(simplify_tolerance)
            catchment['geometry'] = catchment.geometry.simplify(simplify_tolerance)
            
            # Calculate area in equal area projection for weighting
            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
            glhymps['New_area_m2'] = glhymps.to_crs(equal_area_crs).area
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                try:
                    # Ensure CRS match
                    if glhymps.crs != catchment.crs:
                        glhymps = glhymps.to_crs(catchment.crs)
                    
                    self.logger.info("Intersecting GLHYMPS with catchment boundary")
                    # Intersect GLHYMPS with catchment
                    intersection = gpd.overlay(glhymps, catchment, how='intersection')
                    
                    if not intersection.empty:
                        # Convert to equal area for proper area calculation
                        intersection['New_area_m2'] = intersection.to_crs(equal_area_crs).area
                        total_area = intersection['New_area_m2'].sum()
                        
                        if total_area > 0:
                            # Calculate area-weighted averages for porosity and permeability
                            # Porosity
                            porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            porosity_variance = ((intersection['New_area_m2'] * 
                                                (intersection['porosity'] - porosity_mean)**2).sum() / 
                                                total_area)
                            porosity_std = np.sqrt(porosity_variance)
                            
                            # Get min and max values
                            porosity_min = intersection['porosity'].min()
                            porosity_max = intersection['porosity'].max()
                            
                            # Add porosity results
                            results["geology.porosity_mean"] = porosity_mean
                            results["geology.porosity_std"] = porosity_std
                            results["geology.porosity_min"] = porosity_min
                            results["geology.porosity_max"] = porosity_max
                            
                            # Permeability (log10 values)
                            logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            logk_variance = ((intersection['New_area_m2'] * 
                                            (intersection['logK_Ice'] - logk_mean)**2).sum() / 
                                            total_area)
                            logk_std = np.sqrt(logk_variance)
                            
                            # Get min and max values
                            logk_min = intersection['logK_Ice'].min()
                            logk_max = intersection['logK_Ice'].max()
                            
                            # Add permeability results
                            results["geology.log_permeability_mean"] = logk_mean
                            results["geology.log_permeability_std"] = logk_std
                            results["geology.log_permeability_min"] = logk_min
                            results["geology.log_permeability_max"] = logk_max
                            
                            # Calculate derived hydraulic properties
                            
                            # Convert log permeability to hydraulic conductivity
                            # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                            hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                            results["geology.hydraulic_conductivity_m_per_s"] = hyd_cond
                            
                            # Calculate transmissivity assuming 100m aquifer thickness
                            # T = K * b where b is aquifer thickness
                            transmissivity = hyd_cond * 100  # m²/s
                            results["geology.transmissivity_m2_per_s"] = transmissivity
                    else:
                        self.logger.warning("No intersection between GLHYMPS and catchment boundary")
                except Exception as e:
                    self.logger.error(f"Error processing GLHYMPS for lumped catchment: {str(e)}")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Ensure CRS match
                if glhymps.crs != catchment.crs:
                    glhymps = glhymps.to_crs(catchment.crs)
                
                # Create spatial index for GLHYMPS
                self.logger.info("Creating spatial index for faster intersections")
                import rtree
                spatial_index = rtree.index.Index()
                for idx, geom in enumerate(glhymps.geometry):
                    spatial_index.insert(idx, geom.bounds)
                
                # Process in chunks for better performance
                chunk_size = 10  # Adjust based on your data
                for i in range(0, len(catchment), chunk_size):
                    self.logger.info(f"Processing HRUs {i} to {min(i+chunk_size, len(catchment))-1}")
                    hru_chunk = catchment.iloc[i:min(i+chunk_size, len(catchment))]
                    
                    for j, hru in hru_chunk.iterrows():
                        try:
                            hru_id = hru[hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Create a GeoDataFrame with just this HRU
                            hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                            
                            # Use spatial index to filter candidates
                            hru_bounds = hru.geometry.bounds
                            potential_matches_idx = list(spatial_index.intersection(hru_bounds))
                            
                            if not potential_matches_idx:
                                self.logger.debug(f"No potential GLHYMPS matches for HRU {hru_id}")
                                continue
                            
                            # Subset GLHYMPS using potential matches
                            glhymps_subset = glhymps.iloc[potential_matches_idx]
                            
                            # Intersect with GLHYMPS
                            intersection = gpd.overlay(glhymps_subset, hru_gdf, how='intersection')
                            
                            if not intersection.empty:
                                # Convert to equal area for proper area calculation
                                intersection['New_area_m2'] = intersection.to_crs(equal_area_crs).area
                                total_area = intersection['New_area_m2'].sum()
                                
                                if total_area > 0:
                                    # Calculate area-weighted averages for porosity and permeability
                                    # Porosity
                                    porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    porosity_variance = ((intersection['New_area_m2'] * 
                                                        (intersection['porosity'] - porosity_mean)**2).sum() / 
                                                        total_area)
                                    porosity_std = np.sqrt(porosity_variance)
                                    
                                    # Get min and max values
                                    porosity_min = intersection['porosity'].min()
                                    porosity_max = intersection['porosity'].max()
                                    
                                    # Add porosity results
                                    results[f"{prefix}geology.porosity_mean"] = porosity_mean
                                    results[f"{prefix}geology.porosity_std"] = porosity_std
                                    results[f"{prefix}geology.porosity_min"] = porosity_min
                                    results[f"{prefix}geology.porosity_max"] = porosity_max
                                    
                                    # Permeability (log10 values)
                                    logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    logk_variance = ((intersection['New_area_m2'] * 
                                                    (intersection['logK_Ice'] - logk_mean)**2).sum() / 
                                                    total_area)
                                    logk_std = np.sqrt(logk_variance)
                                    
                                    # Get min and max values
                                    logk_min = intersection['logK_Ice'].min()
                                    logk_max = intersection['logK_Ice'].max()
                                    
                                    # Add permeability results
                                    results[f"{prefix}geology.log_permeability_mean"] = logk_mean
                                    results[f"{prefix}geology.log_permeability_std"] = logk_std
                                    results[f"{prefix}geology.log_permeability_min"] = logk_min
                                    results[f"{prefix}geology.log_permeability_max"] = logk_max
                                    
                                    # Calculate derived hydraulic properties
                                    
                                    # Convert log permeability to hydraulic conductivity
                                    # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                                    hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                                    results[f"{prefix}geology.hydraulic_conductivity_m_per_s"] = hyd_cond
                                    
                                    # Calculate transmissivity assuming 100m aquifer thickness
                                    # T = K * b where b is aquifer thickness
                                    transmissivity = hyd_cond * 100  # m²/s
                                    results[f"{prefix}geology.transmissivity_m2_per_s"] = transmissivity
                            else:
                                self.logger.debug(f"No intersection between GLHYMPS and HRU {hru_id}")
                        except Exception as e:
                            self.logger.error(f"Error processing GLHYMPS for HRU {hru_id}: {str(e)}")
        
            # Cache results
            try:
                self.logger.info(f"Caching GLHYMPS results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching GLHYMPS results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing GLHYMPS data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_lithology_data(self) -> Dict[str, Any]:
        """
        Process lithology classification data from geological maps.
        
        Returns:
            Dict[str, Any]: Dictionary of lithology attributes
        """
        results = {}
        
        # Define path to geological map data
        # This could be GMNA (Geological Map of North America) or similar dataset
        geo_map_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/geology/raw")
        
        # Check if geological map directory exists
        if not geo_map_path.exists():
            self.logger.warning(f"Geological map directory not found: {geo_map_path}")
            return results
        
        self.logger.info("Processing lithology data from geological maps")
        
        try:
            # Search for potential geological shapefiles
            shapefile_patterns = ["*geologic*.shp", "*lithology*.shp", "*bedrock*.shp", "*rock*.shp"]
            geo_files = []
            
            for pattern in shapefile_patterns:
                geo_files.extend(list(geo_map_path.glob(pattern)))
            
            if not geo_files:
                self.logger.warning(f"No geological map files found in {geo_map_path}")
                return results
            
            # Use the first matching file
            geo_file = geo_files[0]
            self.logger.info(f"Using geological map file: {geo_file}")
            
            # Load geological map shapefile
            geo_map = gpd.read_file(str(geo_file))
            
            # Identify potential lithology classification columns
            litho_columns = []
            for col in geo_map.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['lith', 'rock', 'geol', 'form', 'unit']):
                    litho_columns.append(col)
            
            if not litho_columns:
                self.logger.warning(f"No lithology classification columns found in {geo_file}")
                return results
            
            # Use the first matching column
            litho_column = litho_columns[0]
            self.logger.info(f"Using lithology classification column: {litho_column}")
            
            # Define rock type categories for classification
            rock_categories = {
                'igneous': ['igneous', 'volcanic', 'plutonic', 'basalt', 'granite', 'diorite', 'gabbro', 'rhyolite', 'andesite'],
                'metamorphic': ['metamorphic', 'gneiss', 'schist', 'quartzite', 'marble', 'amphibolite', 'slate', 'phyllite'],
                'sedimentary': ['sedimentary', 'limestone', 'sandstone', 'shale', 'conglomerate', 'dolomite', 'chalk', 'siltstone'],
                'unconsolidated': ['alluvium', 'colluvium', 'sand', 'gravel', 'clay', 'silt', 'soil', 'till', 'glacial']
            }
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                catchment = gpd.read_file(self.catchment_path)
                
                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)
                
                # Intersect geological map with catchment
                intersection = gpd.overlay(geo_map, catchment, how='intersection')
                
                if not intersection.empty:
                    # Convert to equal area for proper area calculation
                    equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                    intersection_equal_area = intersection.to_crs(equal_area_crs)
                    
                    # Calculate area for each lithology class
                    intersection_equal_area['area_km2'] = intersection_equal_area.geometry.area / 10**6  # m² to km²
                    
                    # Calculate lithology class distribution
                    litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                    total_area = litho_areas.sum()
                    
                    if total_area > 0:
                        # Calculate fraction for each lithology class
                        for litho_type, area in litho_areas.items():
                            if litho_type and str(litho_type).strip():  # Skip empty values
                                fraction = area / total_area
                                # Clean name for attribute key
                                clean_name = self._clean_attribute_name(str(litho_type))
                                results[f"geology.{clean_name}_fraction"] = fraction
                        
                        # Identify dominant lithology class
                        dominant_litho = litho_areas.idxmax()
                        if dominant_litho and str(dominant_litho).strip():
                            results["geology.dominant_lithology"] = str(dominant_litho)
                            results["geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area
                        
                        # Calculate lithology diversity (Shannon entropy)
                        shannon_entropy = 0
                        for _, area in litho_areas.items():
                            if area > 0:
                                p = area / total_area
                                shannon_entropy -= p * np.log(p)
                        results["geology.lithology_diversity"] = shannon_entropy
                    
                    # Classify by broad rock types
                    litho_by_category = {category: 0 for category in rock_categories}
                    
                    # Iterate through each geological unit
                    for _, unit in intersection_equal_area.iterrows():
                        area = unit['area_km2']
                        unit_desc = str(unit.get(litho_column, '')).lower()
                        
                        # Classify the unit into a category based on keywords
                        classified = False
                        for category, keywords in rock_categories.items():
                            if any(keyword in unit_desc for keyword in keywords):
                                litho_by_category[category] += area
                                classified = True
                                break
                    
                    # Calculate percentages for rock categories
                    category_total = sum(litho_by_category.values())
                    if category_total > 0:
                        for category, area in litho_by_category.items():
                            results[f"geology.{category}_fraction"] = area / category_total
                        
                        # Identify dominant category
                        dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                        results["geology.dominant_category"] = dominant_category
                        results["geology.dominant_category_fraction"] = litho_by_category[dominant_category] / category_total
                else:
                    self.logger.warning("No intersection between geological map and catchment boundary")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)
                
                for i, hru in catchment.iterrows():
                    try:
                        hru_id = hru[hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with geological map
                        intersection = gpd.overlay(geo_map, hru_gdf, how='intersection')
                        
                        if not intersection.empty:
                            # Convert to equal area for proper area calculation
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            intersection_equal_area = intersection.to_crs(equal_area_crs)
                            
                            # Calculate area for each lithology class
                            intersection_equal_area['area_km2'] = intersection_equal_area.geometry.area / 10**6  # m² to km²
                            
                            # Calculate lithology class distribution
                            litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                            total_area = litho_areas.sum()
                            
                            if total_area > 0:
                                # Calculate fraction for each lithology class
                                for litho_type, area in litho_areas.items():
                                    if litho_type and str(litho_type).strip():  # Skip empty values
                                        fraction = area / total_area
                                        # Clean name for attribute key
                                        clean_name = self._clean_attribute_name(str(litho_type))
                                        results[f"{prefix}geology.{clean_name}_fraction"] = fraction
                                
                                # Identify dominant lithology class
                                dominant_litho = litho_areas.idxmax()
                                if dominant_litho and str(dominant_litho).strip():
                                    results[f"{prefix}geology.dominant_lithology"] = str(dominant_litho)
                                    results[f"{prefix}geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area
                                
                                # Calculate lithology diversity (Shannon entropy)
                                shannon_entropy = 0
                                for _, area in litho_areas.items():
                                    if area > 0:
                                        p = area / total_area
                                        shannon_entropy -= p * np.log(p)
                                results[f"{prefix}geology.lithology_diversity"] = shannon_entropy
                            
                            # Classify by broad rock types
                            litho_by_category = {category: 0 for category in rock_categories}
                            
                            # Iterate through each geological unit
                            for _, unit in intersection_equal_area.iterrows():
                                area = unit['area_km2']
                                unit_desc = str(unit.get(litho_column, '')).lower()
                                
                                # Classify the unit into a category based on keywords
                                classified = False
                                for category, keywords in rock_categories.items():
                                    if any(keyword in unit_desc for keyword in keywords):
                                        litho_by_category[category] += area
                                        classified = True
                                        break
                            
                            # Calculate percentages for rock categories
                            category_total = sum(litho_by_category.values())
                            if category_total > 0:
                                for category, area in litho_by_category.items():
                                    results[f"{prefix}geology.{category}_fraction"] = area / category_total
                                
                                # Identify dominant category
                                dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                                results[f"{prefix}geology.dominant_category"] = dominant_category
                        else:
                            self.logger.debug(f"No intersection between geological map and HRU {hru_id}")
                    except Exception as e:
                        self.logger.error(f"Error processing lithology for HRU {hru_id}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing lithology data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _clean_attribute_name(self, name: str) -> str:
        """
        Clean a string to be usable as an attribute name.
        
        Args:
            name: Original string
            
        Returns:
            String usable as an attribute name
        """
        if not name:
            return "unknown"
        
        # Convert to string and strip whitespace
        name_str = str(name).strip()
        if not name_str:
            return "unknown"
        
        # Replace spaces and special characters
        cleaned = name_str.lower()
        cleaned = cleaned.replace(' ', '_')
        cleaned = cleaned.replace('-', '_')
        cleaned = cleaned.replace('.', '_')
        cleaned = cleaned.replace('(', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace('/', '_')
        cleaned = cleaned.replace('\\', '_')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('%', 'percent')
        
        # Remove any remaining non-alphanumeric characters
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
        
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'x' + cleaned
        
        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        
        # Handle empty result
        if not cleaned:
            return "unknown"
        
        return cleaned

    def _process_structural_features(self, geologic_map_path: Path) -> Dict[str, Any]:
        """
        Process structural geological features like faults and folds.
        
        Args:
            geologic_map_path: Path to geological map data
            
        Returns:
            Dictionary of structural attributes
        """
        results = {}
        
        # Check for Faults.shp in the GMNA_SHAPES directory
        gmna_dir = geologic_map_path / "GMNA_SHAPES"
        if not gmna_dir.exists():
            self.logger.warning(f"GMNA_SHAPES directory not found at {gmna_dir}")
            return results
        
        # Files to process and their descriptions
        structural_files = [
            ("Faults.shp", "faults"),
            ("Geologic_contacts.shp", "contacts"),
            ("Volcanoes.shp", "volcanoes")
        ]
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            catchment_area_km2 = catchment.to_crs('ESRI:102008').area.sum() / 10**6
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'geology' / 'structural'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Parse bounding box coordinates for clipping
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
            bbox = None
            if len(bbox_coords) == 4:
                bbox = (
                    float(bbox_coords[1]),  # lon_min
                    float(bbox_coords[2]),  # lat_min
                    float(bbox_coords[3]),  # lon_max
                    float(bbox_coords[0])   # lat_max
                )
            
            # Process each structural feature
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            for file_name, feature_type in structural_files:
                feature_file = gmna_dir / file_name
                if not feature_file.exists():
                    self.logger.info(f"{feature_type.capitalize()} file not found: {feature_file}")
                    continue
                
                self.logger.info(f"Processing {feature_type} data from: {feature_file}")
                
                # Create a clipped version if it doesn't exist
                clipped_file = output_dir / f"{self.domain_name}_{feature_type}.gpkg"
                
                if not clipped_file.exists() and bbox:
                    self.logger.info(f"Clipping {feature_type} to catchment bounding box")
                    try:
                        # Read features within the bounding box
                        features = gpd.read_file(feature_file, bbox=bbox)
                        
                        # Save the clipped features
                        if not features.empty:
                            self.logger.info(f"Saving {len(features)} {feature_type} to {clipped_file}")
                            features.to_file(clipped_file, driver="GPKG")
                        else:
                            self.logger.info(f"No {feature_type} found within bounding box")
                    except Exception as e:
                        self.logger.error(f"Error reading {feature_type}: {str(e)}")
                        continue
                
                # Process the features
                if clipped_file.exists():
                    try:
                        # Read the clipped features
                        features = gpd.read_file(clipped_file)
                        
                        if features.empty:
                            self.logger.info(f"No {feature_type} found in the study area")
                            continue
                        
                        # Ensure CRS match
                        if features.crs != catchment.crs:
                            features = features.to_crs(catchment.crs)
                        
                        if is_lumped:
                            # For lumped catchment - intersect with entire catchment
                            if feature_type in ['faults', 'contacts']:  # Line features
                                # Intersect with catchment
                                features_in_catchment = gpd.clip(features, catchment)
                                
                                if not features_in_catchment.empty:
                                    # Convert to equal area projection for length calculations
                                    features_equal_area = features_in_catchment.to_crs('ESRI:102008')
                                    
                                    # Calculate total length and density
                                    total_length_km = features_equal_area.length.sum() / 1000  # m to km
                                    density = total_length_km / catchment_area_km2 if catchment_area_km2 > 0 else 0
                                    
                                    # Add to results
                                    results[f"geology.{feature_type}_count"] = len(features_in_catchment)
                                    results[f"geology.{feature_type}_length_km"] = total_length_km
                                    results[f"geology.{feature_type}_density_km_per_km2"] = density
                                else:
                                    results[f"geology.{feature_type}_count"] = 0
                                    results[f"geology.{feature_type}_length_km"] = 0
                                    results[f"geology.{feature_type}_density_km_per_km2"] = 0
                            
                            elif feature_type == 'volcanoes':  # Point features
                                # Count volcanoes in catchment
                                volcanoes_in_catchment = gpd.clip(features, catchment)
                                volcano_count = len(volcanoes_in_catchment)
                                volcano_density = volcano_count / catchment_area_km2 if catchment_area_km2 > 0 else 0
                                
                                results["geology.volcano_count"] = volcano_count
                                results["geology.volcano_density_per_km2"] = volcano_density
                        else:
                            # For distributed catchment - process each HRU
                            for idx, hru in catchment.iterrows():
                                hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                                prefix = f"HRU_{hru_id}_"
                                
                                # Create a GeoDataFrame with just this HRU
                                hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                                hru_area_km2 = hru_gdf.to_crs('ESRI:102008').area.sum() / 10**6
                                
                                if feature_type in ['faults', 'contacts']:  # Line features
                                    # Clip features to HRU
                                    try:
                                        features_in_hru = gpd.clip(features, hru_gdf)
                                        
                                        if not features_in_hru.empty:
                                            # Calculate length and density
                                            features_equal_area = features_in_hru.to_crs('ESRI:102008')
                                            total_length_km = features_equal_area.length.sum() / 1000
                                            density = total_length_km / hru_area_km2 if hru_area_km2 > 0 else 0
                                            
                                            results[f"{prefix}geology.{feature_type}_count"] = len(features_in_hru)
                                            results[f"{prefix}geology.{feature_type}_length_km"] = total_length_km
                                            results[f"{prefix}geology.{feature_type}_density_km_per_km2"] = density
                                        else:
                                            results[f"{prefix}geology.{feature_type}_count"] = 0
                                            results[f"{prefix}geology.{feature_type}_length_km"] = 0
                                            results[f"{prefix}geology.{feature_type}_density_km_per_km2"] = 0
                                    except Exception as e:
                                        self.logger.error(f"Error processing {feature_type} for HRU {hru_id}: {str(e)}")
                                
                                elif feature_type == 'volcanoes':  # Point features
                                    try:
                                        volcanoes_in_hru = gpd.clip(features, hru_gdf)
                                        volcano_count = len(volcanoes_in_hru)
                                        volcano_density = volcano_count / hru_area_km2 if hru_area_km2 > 0 else 0
                                        
                                        results[f"{prefix}geology.volcano_count"] = volcano_count
                                        results[f"{prefix}geology.volcano_density_per_km2"] = volcano_density
                                    except Exception as e:
                                        self.logger.error(f"Error processing volcanoes for HRU {hru_id}: {str(e)}")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {feature_type}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing structural features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _process_hydrological_attributes(self) -> Dict[str, Any]:
        """
        Process hydrological data including lakes and river networks.
        
        Sources:
        - HydroLAKES: Global database of lakes
        - MERIT Basins: River network and basin characteristics
        """
        results = {}
        
        # Find and check paths for hydrological data
        hydrolakes_path = self._get_data_path('ATTRIBUTES_HYDROLAKES_PATH', 'hydrolakes')
        merit_basins_path = self._get_data_path('ATTRIBUTES_MERIT_BASINS_PATH', 'merit_basins')
        
        self.logger.info(f"HydroLAKES path: {hydrolakes_path}")
        self.logger.info(f"MERIT Basins path: {merit_basins_path}")
        
        # Process lakes data
        lakes_results = self._process_lakes_data(hydrolakes_path)
        results.update(lakes_results)
        
        # Process river network data
        river_results = self._process_river_network_data(merit_basins_path)
        results.update(river_results)
        
        return results

    def calculate_seasonality_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced seasonality metrics including sine curve fitting.
        
        Returns:
            Dict[str, Any]: Dictionary of seasonality metrics
        """
        results = {}
        
        # Load temperature and precipitation data
        temp_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_temperature.csv"
        precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
        
        if not temp_path.exists() or not precip_path.exists():
            self.logger.warning("Temperature or precipitation data not found for seasonality calculation")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            from scipy.optimize import curve_fit
            
            # Define sine fitting functions
            def sine_function_temp(x, mean, delta, phase, period=1):
                return mean + delta * np.sin(2*np.pi*(x-phase)/period)
            
            def sine_function_prec(x, mean, delta, phase, period=1):
                result = mean * (1 + delta * np.sin(2*np.pi*(x-phase)/period))
                return np.where(result < 0, 0, result)
            
            # Read data
            temp_df = pd.read_csv(temp_path, parse_dates=['date'])
            precip_df = pd.read_csv(precip_path, parse_dates=['date'])
            
            # Resample to daily if needed
            temp_daily = temp_df.set_index('date').resample('D').mean()
            precip_daily = precip_df.set_index('date').resample('D').mean()
            
            # Ensure same time period
            common_idx = temp_daily.index.intersection(precip_daily.index)
            temp_daily = temp_daily.loc[common_idx]
            precip_daily = precip_daily.loc[common_idx]
            
            # Calculate day of year as fraction of year for x-axis
            day_of_year = np.array([(d.dayofyear - 1) / 365 for d in temp_daily.index])
            
            # Fit temperature sine curve
            temp_values = temp_daily['temperature'].values
            try:
                t_mean = float(np.mean(temp_values))
                t_delta = float(np.max(temp_values) - np.min(temp_values))
                t_phase = 0.5  # Initial guess for phase
                t_initial_guess = [t_mean, t_delta, t_phase]
                t_pars, _ = curve_fit(sine_function_temp, day_of_year, temp_values, p0=t_initial_guess)
                
                results["climate.temperature_mean"] = t_pars[0]
                results["climate.temperature_amplitude"] = t_pars[1]
                results["climate.temperature_phase"] = t_pars[2]
            except Exception as e:
                self.logger.warning(f"Error fitting temperature sine curve: {str(e)}")
            
            # Fit precipitation sine curve
            precip_values = precip_daily['precipitation'].values
            try:
                p_mean = float(np.mean(precip_values))
                p_delta = float(np.max(precip_values) - np.min(precip_values)) / p_mean if p_mean > 0 else 0
                p_phase = 0.5  # Initial guess for phase
                p_initial_guess = [p_mean, p_delta, p_phase]
                p_pars, _ = curve_fit(sine_function_prec, day_of_year, precip_values, p0=p_initial_guess)
                
                results["climate.precipitation_mean"] = p_pars[0]
                results["climate.precipitation_amplitude"] = p_pars[1]
                results["climate.precipitation_phase"] = p_pars[2]
                
                # Calculate Woods (2009) seasonality index
                if hasattr(t_pars, '__len__') and hasattr(p_pars, '__len__') and len(t_pars) >= 3 and len(p_pars) >= 3:
                    seasonality = p_pars[1] * np.sign(t_pars[1]) * np.cos(2*np.pi*(p_pars[2]-t_pars[2])/1)
                    results["climate.seasonality_index"] = seasonality
            except Exception as e:
                self.logger.warning(f"Error fitting precipitation sine curve: {str(e)}")
            
            # Calculate additional seasonality metrics
            
            # Walsh & Lawler (1981) seasonality index for precipitation
            try:
                # Resample to monthly
                monthly_precip = precip_df.set_index('date').resample('M').sum()
                
                # Group by month and calculate multi-year average for each month
                monthly_avg = monthly_precip.groupby(monthly_precip.index.month).mean()
                
                if len(monthly_avg) == 12:
                    annual_sum = monthly_avg.sum().iloc[0]
                    monthly_diff = np.abs(monthly_avg.values - annual_sum/12)
                    walsh_index = np.sum(monthly_diff) / annual_sum
                    
                    results["climate.precipitation_walsh_seasonality"] = walsh_index
                    
                    # Interpret the Walsh & Lawler index
                    if walsh_index < 0.19:
                        results["climate.seasonality_regime"] = "very_equable"
                    elif walsh_index < 0.40:
                        results["climate.seasonality_regime"] = "equable_but_with_a_definite_wetter_season"
                    elif walsh_index < 0.60:
                        results["climate.seasonality_regime"] = "rather_seasonal"
                    elif walsh_index < 0.80:
                        results["climate.seasonality_regime"] = "seasonal"
                    elif walsh_index < 0.99:
                        results["climate.seasonality_regime"] = "markedly_seasonal_with_a_long_drier_season"
                    elif walsh_index < 1.19:
                        results["climate.seasonality_regime"] = "most_rain_in_3_months_or_less"
                    else:
                        results["climate.seasonality_regime"] = "extreme_seasonality"
            except Exception as e:
                self.logger.warning(f"Error calculating Walsh & Lawler seasonality index: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonality metrics: {str(e)}")
            return results

    def calculate_water_balance(self) -> Dict[str, Any]:
        """
        Calculate water balance components and hydrological indices.
        
        Returns:
            Dict[str, Any]: Dictionary of water balance metrics
        """
        results = {}
        
        # Look for required data
        precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
        pet_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_pet.csv"
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not precip_path.exists() or not streamflow_path.exists():
            self.logger.warning("Cannot calculate water balance: missing precipitation or streamflow data")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            
            # Read data
            precip_df = pd.read_csv(precip_path, parse_dates=['date'])
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            
            # Read PET if available, otherwise use None
            if pet_path.exists():
                pet_df = pd.read_csv(pet_path, parse_dates=['date'])
            else:
                pet_df = None
            
            # Set index and align data
            precip_df.set_index('date', inplace=True)
            streamflow_df.set_index('date', inplace=True)
            if pet_df is not None:
                pet_df.set_index('date', inplace=True)
            
            # Define common time period
            common_period = precip_df.index.intersection(streamflow_df.index)
            if pet_df is not None:
                common_period = common_period.intersection(pet_df.index)
            
            if len(common_period) == 0:
                self.logger.warning("No common time period between precipitation and streamflow data")
                return results
            
            # Subset data to common period
            precip = precip_df.loc[common_period, 'precipitation'].copy()
            streamflow = streamflow_df.loc[common_period, 'flow'].copy()
            if pet_df is not None:
                pet = pet_df.loc[common_period, 'pet'].copy()
            else:
                pet = None
            
            # Calculate water balance components
            
            # 1. Runoff ratio (Q/P)
            annual_precip = precip.resample('YS').sum()
            annual_streamflow = streamflow.resample('YS').sum()
            
            # Align years
            common_years = annual_precip.index.intersection(annual_streamflow.index)
            
            if len(common_years) > 0:
                annual_runoff_ratio = annual_streamflow.loc[common_years] / annual_precip.loc[common_years]
                results["hydrology.annual_runoff_ratio_mean"] = annual_runoff_ratio.mean()
                results["hydrology.annual_runoff_ratio_std"] = annual_runoff_ratio.std()
                results["hydrology.annual_runoff_ratio_cv"] = annual_runoff_ratio.std() / annual_runoff_ratio.mean() if annual_runoff_ratio.mean() > 0 else np.nan
            
            # 2. Estimate actual evapotranspiration (AET) from water balance
            # AET = P - Q - dS/dt (ignoring storage changes)
            annual_aet_estimate = annual_precip.loc[common_years] - annual_streamflow.loc[common_years]
            results["hydrology.annual_aet_estimate_mean"] = annual_aet_estimate.mean()
            results["hydrology.annual_aet_estimate_std"] = annual_aet_estimate.std()
            
            # 3. Aridity index and Budyko analysis
            if pet_df is not None:
                annual_pet = pet.resample('YS').sum()
                common_years_pet = common_years.intersection(annual_pet.index)
                
                if len(common_years_pet) > 0:
                    # Aridity index (PET/P)
                    aridity_index = annual_pet.loc[common_years_pet] / annual_precip.loc[common_years_pet]
                    results["hydrology.aridity_index_mean"] = aridity_index.mean()
                    results["hydrology.aridity_index_std"] = aridity_index.std()
                    
                    # Evaporative index (AET/P)
                    evaporative_index = annual_aet_estimate.loc[common_years_pet] / annual_precip.loc[common_years_pet]
                    results["hydrology.evaporative_index_mean"] = evaporative_index.mean()
                    
                    # Calculate Budyko curve parameters (Fu equation: ET/P = 1 + PET/P - [1 + (PET/P)^w]^(1/w))
                    # We'll use optimization to find the w parameter
                    
                    from scipy.optimize import minimize
                    
                    def fu_equation(w, pet_p):
                        return 1 + pet_p - (1 + pet_p**w)**(1/w)
                    
                    def objective(w, pet_p, et_p):
                        predicted = fu_equation(w, pet_p)
                        return np.mean((predicted - et_p)**2)
                    
                    # Filter out invalid values
                    valid = (aridity_index > 0) & (evaporative_index > 0) & (~np.isnan(aridity_index)) & (~np.isnan(evaporative_index))
                    
                    if valid.any():
                        pet_p_values = aridity_index[valid].values
                        et_p_values = evaporative_index[valid].values
                        
                        # Find optimal w parameter
                        result = minimize(objective, 2.0, args=(pet_p_values, et_p_values))
                        w_parameter = result.x[0]
                        
                        results["hydrology.budyko_parameter_w"] = w_parameter
                        
                        # Calculate Budyko-predicted ET/P
                        budyko_predicted = fu_equation(w_parameter, aridity_index.mean())
                        results["hydrology.budyko_predicted_et_ratio"] = budyko_predicted
                        
                        # Calculate residual (actual - predicted)
                        results["hydrology.budyko_residual"] = evaporative_index.mean() - budyko_predicted
            
            # 4. Flow duration curve metrics
            flow_sorted = np.sort(streamflow.dropna().values)[::-1]  # sort in descending order
            if len(flow_sorted) > 0:
                total_flow = np.sum(flow_sorted)
                
                # Calculate normalized flow duration curve
                n = len(flow_sorted)
                exceedance_prob = np.arange(1, n+1) / (n+1)  # exceedance probability
                
                # Calculate common FDC metrics
                q1 = np.interp(0.01, exceedance_prob, flow_sorted)
                q5 = np.interp(0.05, exceedance_prob, flow_sorted)
                q10 = np.interp(0.1, exceedance_prob, flow_sorted)
                q50 = np.interp(0.5, exceedance_prob, flow_sorted)
                q85 = np.interp(0.85, exceedance_prob, flow_sorted)
                q95 = np.interp(0.95, exceedance_prob, flow_sorted)
                q99 = np.interp(0.99, exceedance_prob, flow_sorted)
                
                results["hydrology.fdc_q1"] = q1
                results["hydrology.fdc_q5"] = q5
                results["hydrology.fdc_q10"] = q10
                results["hydrology.fdc_q50"] = q50
                results["hydrology.fdc_q85"] = q85
                results["hydrology.fdc_q95"] = q95
                results["hydrology.fdc_q99"] = q99
                
                # Calculate high flow (Q10/Q50) and low flow (Q50/Q90) indices
                results["hydrology.high_flow_index"] = q10 / q50 if q50 > 0 else np.nan
                results["hydrology.low_flow_index"] = q50 / q95 if q95 > 0 else np.nan
                
                # Calculate flow variability indices
                results["hydrology.flow_duration_index"] = (q10 - q95) / q50 if q50 > 0 else np.nan
                
                # Baseflow index (Q90/Q50)
                results["hydrology.baseflow_index_fdc"] = q95 / q50 if q50 > 0 else np.nan
            
            # 5. Flow seasonality and timing
            monthly_flow = streamflow.resample('M').sum()
            monthly_precip = precip.resample('M').sum()
            
            # Calculate monthly flow contribution
            annual_flow_by_month = monthly_flow.groupby(monthly_flow.index.month).mean()
            annual_flow_total = annual_flow_by_month.sum()
            
            if annual_flow_total > 0:
                monthly_flow_fraction = annual_flow_by_month / annual_flow_total
                
                # Find month with maximum flow
                max_flow_month = monthly_flow_fraction.idxmax()
                results["hydrology.peak_flow_month"] = max_flow_month
                
                # Calculate seasonality indices
                
                # Colwell's predictability index components
                # M = seasonality, C = contingency, P = predictability = M + C
                # Ranges from 0 (unpredictable) to 1 (perfectly predictable)
                
                # Use normalized monthly values for calculations
                monthly_flow_norm = monthly_flow / monthly_flow.mean() if monthly_flow.mean() > 0 else monthly_flow
                
                # Seasonality index (Walsh & Lawler)
                if len(monthly_flow_fraction) == 12:
                    monthly_diff = np.abs(monthly_flow_fraction - 1/12)
                    seasonality_index = np.sum(monthly_diff) * 0.5  # Scale to 0-1
                    results["hydrology.flow_seasonality_index"] = seasonality_index
                
                # Calculate monthly runoff ratios
                monthly_precip_avg = monthly_precip.groupby(monthly_precip.index.month).mean()
                monthly_flow_avg = monthly_flow.groupby(monthly_flow.index.month).mean()
                
                # Calculate monthly runoff ratios
                common_months = monthly_flow_avg.index.intersection(monthly_precip_avg.index)
                
                if len(common_months) > 0:
                    monthly_runoff_ratio = monthly_flow_avg.loc[common_months] / monthly_precip_avg.loc[common_months]
                    
                    # Store min, max, and range of monthly runoff ratios
                    results["hydrology.monthly_runoff_ratio_min"] = monthly_runoff_ratio.min()
                    results["hydrology.monthly_runoff_ratio_max"] = monthly_runoff_ratio.max()
                    results["hydrology.monthly_runoff_ratio_range"] = monthly_runoff_ratio.max() - monthly_runoff_ratio.min()
                    
                    # Find month with maximum and minimum runoff ratio
                    results["hydrology.max_runoff_ratio_month"] = monthly_runoff_ratio.idxmax()
                    results["hydrology.min_runoff_ratio_month"] = monthly_runoff_ratio.idxmin()
                    
                    # Calculate correlation between monthly precipitation and runoff
                    corr = np.corrcoef(monthly_precip_avg.loc[common_months], monthly_flow_avg.loc[common_months])[0, 1]
                    results["hydrology.precip_flow_correlation"] = corr
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating water balance: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results


    def calculate_composite_landcover_metrics(self) -> Dict[str, Any]:
        """
        Calculate composite land cover metrics combining multiple land cover datasets
        and extracting ecologically meaningful indicators.
        
        Returns:
            Dict[str, Any]: Dictionary of composite land cover metrics
        """
        results = {}
        
        # Collect existing land cover results from individual datasets
        lc_datasets = {
            "glclu2019": "landcover.forest_fraction",
            "modis": "landcover.evergreen_needleleaf_fraction",
            "esa": "landcover.forest_broadleaf_evergreen_fraction"
        }
        
        # Define ecological groupings that span datasets
        ecological_groups = {
            "forest": ["forest", "tree", "woodland", "evergreen", "deciduous", "needleleaf", "broadleaf"],
            "grassland": ["grass", "savanna", "rangeland", "pasture"],
            "wetland": ["wetland", "marsh", "bog", "fen", "swamp"],
            "cropland": ["crop", "agriculture", "cultivated"],
            "urban": ["urban", "built", "city", "development"],
            "water": ["water", "lake", "river", "reservoir"],
            "barren": ["barren", "desert", "rock", "sand"]
        }
        
        try:
            # Collect all landcover attributes from results
            all_attributes = [key for key in self.results.keys() if key.startswith("landcover.")]
            
            # Group by datasets
            datasets_found = {}
            for attr in all_attributes:
                # Extract dataset name from the attribute
                parts = attr.split(".")
                if len(parts) > 1:
                    attribute = parts[1]
                    # Try to identify which dataset this comes from
                    for dataset_name, marker in lc_datasets.items():
                        if marker in attr:
                            if dataset_name not in datasets_found:
                                datasets_found[dataset_name] = []
                            datasets_found[dataset_name].append(attr)
            
            # If we found at least one dataset, calculate composite metrics
            if datasets_found:
                # First, create ecological groupings for each dataset
                dataset_eco_groups = {}
                
                for dataset, attributes in datasets_found.items():
                    dataset_eco_groups[dataset] = {group: [] for group in ecological_groups}
                    
                    # Assign attributes to ecological groups
                    for attr in attributes:
                        for group, keywords in ecological_groups.items():
                            if any(keyword in attr.lower() for keyword in keywords):
                                dataset_eco_groups[dataset][group].append(attr)
                
                # Calculate composite metrics for each ecological group
                for group, keywords in ecological_groups.items():
                    # Collect values across datasets
                    group_values = []
                    
                    for dataset, eco_groups in dataset_eco_groups.items():
                        if eco_groups[group]:
                            # Get average value for this group in this dataset
                            group_attrs = eco_groups[group]
                            attr_values = [self.results.get(attr, 0) for attr in group_attrs]
                            if attr_values:
                                group_values.append(np.mean(attr_values))
                    
                    # Calculate composite metric if we have values
                    if group_values:
                        results[f"landcover.composite_{group}_fraction"] = np.mean(group_values)
                
                # Calculate landscape diversity from composite fractions
                # Shannon diversity index
                fractions = [v for k, v in results.items() if k.startswith("landcover.composite_") and k.endswith("_fraction")]
                if fractions:
                    # Normalize fractions to sum to 1
                    total = sum(fractions)
                    if total > 0:
                        normalized = [f/total for f in fractions]
                        shannon_index = -sum(p * np.log(p) for p in normalized if p > 0)
                        results["landcover.composite_shannon_diversity"] = shannon_index
                        
                        # Simpson diversity index (1-D)
                        simpson_index = 1 - sum(p**2 for p in normalized)
                        results["landcover.composite_simpson_diversity"] = simpson_index
                
                # Calculate anthropogenic influence
                if "landcover.composite_urban_fraction" in results and "landcover.composite_cropland_fraction" in results:
                    anthropogenic = results["landcover.composite_urban_fraction"] + results["landcover.composite_cropland_fraction"]
                    results["landcover.anthropogenic_influence"] = anthropogenic
                
                # Calculate natural vegetation cover
                if "landcover.composite_forest_fraction" in results and "landcover.composite_grassland_fraction" in results and "landcover.composite_wetland_fraction" in results:
                    natural = results["landcover.composite_forest_fraction"] + results["landcover.composite_grassland_fraction"] + results["landcover.composite_wetland_fraction"]
                    results["landcover.natural_vegetation_cover"] = natural
            
            # Analyze vegetation dynamics if we have LAI data
            lai_months = [key for key in self.results.keys() if key.startswith("vegetation.lai_month")]
            if lai_months:
                # Extract monthly values
                monthly_lai = {}
                for key in lai_months:
                    if "_mean" in key:
                        # Extract month number
                        try:
                            month = int(key.split("month")[1][:2])
                            monthly_lai[month] = self.results[key]
                        except (ValueError, IndexError):
                            pass
                
                if len(monthly_lai) >= 6:  # Need at least half the year
                    # Sort by month
                    months = sorted(monthly_lai.keys())
                    lai_values = [monthly_lai[m] for m in months]
                    
                    # Calculate seasonal amplitude
                    lai_min = min(lai_values)
                    lai_max = max(lai_values)
                    results["vegetation.seasonal_lai_amplitude"] = lai_max - lai_min
                    
                    # Calculate growing season characteristics
                    if lai_min < lai_max:
                        # Define growing season threshold (e.g., 50% of range above minimum)
                        threshold = lai_min + 0.5 * (lai_max - lai_min)
                        
                        # Find months above threshold
                        growing_months = [m for m in months if monthly_lai[m] >= threshold]
                        
                        if growing_months:
                            results["vegetation.growing_season_length_months"] = len(growing_months)
                            results["vegetation.growing_season_start_month"] = min(growing_months)
                            results["vegetation.growing_season_end_month"] = max(growing_months)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error calculating composite land cover metrics: {str(e)}")
            return results

    def enhance_river_network_analysis(self) -> Dict[str, Any]:
        """
        Enhanced analysis of river network including stream order, drainage density,
        and river morphology metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of enhanced river network attributes
        """
        results = {}
        
        # Get path to river network shapefile
        river_dir = self.project_dir / 'shapefiles' / 'river_network'
        river_files = list(river_dir.glob("*.shp"))
        
        if not river_files:
            self.logger.warning("No river network shapefile found")
            return results
        
        river_file = river_files[0]
        
        try:
            import geopandas as gpd
            import numpy as np
            from scipy.stats import skew, kurtosis
            
            # Read river network and catchment shapefiles
            river_network = gpd.read_file(river_file)
            catchment = gpd.read_file(self.catchment_path)
            
            # Ensure CRS match
            if river_network.crs != catchment.crs:
                river_network = river_network.to_crs(catchment.crs)
            
            # Calculate catchment area in km²
            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
            catchment_area_km2 = catchment.to_crs(equal_area_crs).area.sum() / 10**6
            
            # Basic river network statistics
            results["hydrology.stream_segment_count"] = len(river_network)
            
            # Calculate total stream length
            if 'Length' in river_network.columns:
                length_column = 'Length'
            elif 'length' in river_network.columns:
                length_column = 'length'
            elif any(col.lower() == 'length' for col in river_network.columns):
                length_column = next(col for col in river_network.columns if col.lower() == 'length')
            else:
                # Calculate length using geometry
                river_network_equal_area = river_network.to_crs(equal_area_crs)
                river_network_equal_area['length_m'] = river_network_equal_area.geometry.length
                length_column = 'length_m'
            
            # Total stream length in km
            total_length_km = river_network[length_column].sum() / 1000 if length_column != 'length_m' else river_network[length_column].sum() / 1000
            results["hydrology.stream_total_length_km"] = total_length_km
            
            # Drainage density (km/km²)
            results["hydrology.drainage_density"] = total_length_km / catchment_area_km2
            
            # Stream order statistics if available
            stream_order_column = None
            for possible_column in ['str_order', 'order', 'strahler', 'STRAHLER']:
                if possible_column in river_network.columns:
                    stream_order_column = possible_column
                    break
            
            if stream_order_column:
                stream_orders = river_network[stream_order_column].astype(float)
                results["hydrology.stream_order_min"] = stream_orders.min()
                results["hydrology.stream_order_max"] = stream_orders.max()
                results["hydrology.stream_order_mean"] = stream_orders.mean()
                
                # Stream order frequency and length by order
                for order in range(1, int(stream_orders.max()) + 1):
                    order_segments = river_network[river_network[stream_order_column] == order]
                    if not order_segments.empty:
                        # Number of segments for this order
                        results[f"hydrology.stream_order_{order}_count"] = len(order_segments)
                        
                        # Length for this order in km
                        order_length_km = order_segments[length_column].sum() / 1000 if length_column != 'length_m' else order_segments[length_column].sum() / 1000
                        results[f"hydrology.stream_order_{order}_length_km"] = order_length_km
                        
                        # Percentage of total length
                        if total_length_km > 0:
                            results[f"hydrology.stream_order_{order}_percent"] = (order_length_km / total_length_km) * 100
                
                # Calculate bifurcation ratio (ratio of number of streams of a given order to number of streams of next higher order)
                for order in range(1, int(stream_orders.max())):
                    count_current = len(river_network[river_network[stream_order_column] == order])
                    count_next = len(river_network[river_network[stream_order_column] == order + 1])
                    
                    if count_next > 0:
                        bifurcation_ratio = count_current / count_next
                        results[f"hydrology.bifurcation_ratio_order_{order}_{order+1}"] = bifurcation_ratio
                
                # Average bifurcation ratio across all orders
                bifurcation_ratios = []
                for order in range(1, int(stream_orders.max())):
                    count_current = len(river_network[river_network[stream_order_column] == order])
                    count_next = len(river_network[river_network[stream_order_column] == order + 1])
                    
                    if count_next > 0:
                        bifurcation_ratios.append(count_current / count_next)
                
                if bifurcation_ratios:
                    results["hydrology.mean_bifurcation_ratio"] = np.mean(bifurcation_ratios)
            
            # Slope statistics if available
            slope_column = None
            for possible_column in ['slope', 'SLOPE', 'Slope']:
                if possible_column in river_network.columns:
                    slope_column = possible_column
                    break
            
            if slope_column:
                slopes = river_network[slope_column].astype(float)
                results["hydrology.stream_slope_min"] = slopes.min()
                results["hydrology.stream_slope_max"] = slopes.max()
                results["hydrology.stream_slope_mean"] = slopes.mean()
                results["hydrology.stream_slope_median"] = slopes.median()
                results["hydrology.stream_slope_std"] = slopes.std()
            
            # Calculate sinuosity if we have both actual length and straight-line distance
            sinuosity_column = None
            for possible_column in ['sinuosity', 'SINUOSITY', 'Sinuosity']:
                if possible_column in river_network.columns:
                    sinuosity_column = possible_column
                    break
            
            if sinuosity_column:
                sinuosity = river_network[sinuosity_column].astype(float)
                results["hydrology.stream_sinuosity_min"] = sinuosity.min()
                results["hydrology.stream_sinuosity_max"] = sinuosity.max()
                results["hydrology.stream_sinuosity_mean"] = sinuosity.mean()
                results["hydrology.stream_sinuosity_median"] = sinuosity.median()
            else:
                # Try to calculate sinuosity using geometry
                try:
                    # Create a new column with the straight-line distance between endpoints
                    def calculate_sinuosity(geometry):
                        try:
                            coords = list(geometry.coords)
                            if len(coords) >= 2:
                                from shapely.geometry import LineString
                                straight_line = LineString([coords[0], coords[-1]])
                                return geometry.length / straight_line.length
                            return 1.0
                        except:
                            return 1.0
                    
                    river_network['calc_sinuosity'] = river_network.geometry.apply(calculate_sinuosity)
                    
                    results["hydrology.stream_sinuosity_min"] = river_network['calc_sinuosity'].min()
                    results["hydrology.stream_sinuosity_max"] = river_network['calc_sinuosity'].max()
                    results["hydrology.stream_sinuosity_mean"] = river_network['calc_sinuosity'].mean()
                    results["hydrology.stream_sinuosity_median"] = river_network['calc_sinuosity'].median()
                except Exception as e:
                    self.logger.warning(f"Could not calculate sinuosity: {str(e)}")
            
            # Calculate headwater density
            headwater_column = None
            if stream_order_column:
                # Headwaters are typically first-order streams
                headwaters = river_network[river_network[stream_order_column] == 1]
                results["hydrology.headwater_count"] = len(headwaters)
                results["hydrology.headwater_density_per_km2"] = len(headwaters) / catchment_area_km2
            
            # If we have a pour point, calculate:
            # 1. Distance to farthest headwater (longest flow path)
            # 2. Main stem length
            # 3. Basin elongation ratio
            pour_point_file = self.project_dir / 'shapefiles' / 'pour_point' / f"{self.domain_name}_pourPoint.shp"
            
            if pour_point_file.exists():
                pour_point = gpd.read_file(pour_point_file)
                
                # Ensure CRS match
                if pour_point.crs != river_network.crs:
                    pour_point = pour_point.to_crs(river_network.crs)
                
                # Try to find the main stem by tracing upstream from the pour point
                # This would require river network connectivity information
                # Placeholder for the logic - depends on how your river network topology is structured
                
                # Estimate basin shape metrics
                # Basin length is approximately the longest flow path
                if "hydrology.longest_flowpath_km" in results:
                    basin_length = results["hydrology.longest_flowpath_km"]
                else:
                    # Estimate basin length using catchment geometry
                    # Calculate longest intra-polygon distance as rough approximation
                    from shapely.geometry import MultiPoint
                    hull = catchment.unary_union.convex_hull
                    if hasattr(hull, 'exterior'):
                        points = list(hull.exterior.coords)
                        if len(points) > 1:
                            # Find approximate longest axis through convex hull
                            point_combinations = [(points[i], points[j]) 
                                                for i in range(len(points)) 
                                                for j in range(i+1, len(points))]
                            distances = [np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
                                        for p1, p2 in point_combinations]
                            basin_length = max(distances) / 1000  # convert to km
                            results["hydrology.basin_length_km"] = basin_length
                
                # Calculate basin shape metrics
                if "hydrology.basin_length_km" in results and catchment_area_km2 > 0:
                    # Basin elongation ratio (Schumm, 1956)
                    # Re = diameter of circle with same area as basin / basin length
                    diameter_equivalent_circle = 2 * np.sqrt(catchment_area_km2 / np.pi)
                    elongation_ratio = diameter_equivalent_circle / results["hydrology.basin_length_km"]
                    results["hydrology.basin_elongation_ratio"] = elongation_ratio
                    
                    # Basin form factor (Horton, 1932)
                    # Form factor = Area / (Basin length)²
                    form_factor = catchment_area_km2 / (results["hydrology.basin_length_km"]**2)
                    results["hydrology.basin_form_factor"] = form_factor
                    
                    # Basin circularity ratio
                    # Circularity = 4πA / P² where A is area and P is perimeter
                    perimeter_km = catchment.to_crs(equal_area_crs).geometry.length.sum() / 1000
                    circularity_ratio = (4 * np.pi * catchment_area_km2) / (perimeter_km**2)
                    results["hydrology.basin_circularity_ratio"] = circularity_ratio
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in enhanced river network analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results

    def enhance_hydrogeological_attributes(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing geological attributes with derived hydraulic properties.
        
        Args:
            current_results: Current geological attributes
            
        Returns:
            Dict[str, Any]: Enhanced geological attributes
        """
        results = dict(current_results)  # Create a copy to avoid modifying the original
        
        try:
            # Check if we have log permeability values
            if "geology.log_permeability_mean" in results:
                log_k = results["geology.log_permeability_mean"]
                
                # Convert log permeability to hydraulic conductivity in m/s
                # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                hydraulic_conductivity = 10**(log_k) * (10**7)
                results["geology.hydraulic_conductivity_m_per_s"] = hydraulic_conductivity
                
                # Convert to more common units (cm/hr, m/day)
                results["geology.hydraulic_conductivity_cm_per_hr"] = hydraulic_conductivity * 3600 * 100
                results["geology.hydraulic_conductivity_m_per_day"] = hydraulic_conductivity * 86400
                
                # Calculate transmissivity assuming typical aquifer thickness values
                # Shallow aquifer (10m)
                results["geology.transmissivity_shallow_m2_per_day"] = hydraulic_conductivity * 10 * 86400
                
                # Deep aquifer (100m)
                results["geology.transmissivity_deep_m2_per_day"] = hydraulic_conductivity * 100 * 86400
                
                # Calculate hydraulic diffusivity if we have porosity data
                if "geology.porosity_mean" in results:
                    porosity = results["geology.porosity_mean"]
                    if porosity > 0:
                        specific_storage = 1e-6  # Typical value for confined aquifer (1/m)
                        storativity = specific_storage * 100  # For 100m thick aquifer
                        
                        # Calculate specific yield (typically 0.1-0.3 of porosity)
                        specific_yield = 0.2 * porosity
                        results["geology.specific_yield"] = specific_yield
                        
                        # Hydraulic diffusivity (m²/s)
                        diffusivity = hydraulic_conductivity / storativity
                        results["geology.hydraulic_diffusivity_m2_per_s"] = diffusivity
                        
                        # Groundwater response time (days) for 1km travel distance
                        if diffusivity > 0:
                            response_time = (1000**2) / (diffusivity * 86400)  # days
                            results["geology.groundwater_response_time_days"] = response_time
            
            # Derive aquifer properties from soil depth if available
            if "soil.regolith_thickness_mean" in results or "soil.sedimentary_thickness_mean" in results:
                # Use available thickness data or default to a typical value
                regolith_thickness = results.get("soil.regolith_thickness_mean", 0)
                sedimentary_thickness = results.get("soil.sedimentary_thickness_mean", 0)
                
                # Estimate total aquifer thickness
                aquifer_thickness = max(regolith_thickness, sedimentary_thickness)
                if aquifer_thickness == 0:
                    aquifer_thickness = 50  # Default value if no data available
                
                results["geology.estimated_aquifer_thickness_m"] = aquifer_thickness
                
                # If we have hydraulic conductivity, calculate transmissivity
                if "geology.hydraulic_conductivity_m_per_s" in results:
                    transmissivity = results["geology.hydraulic_conductivity_m_per_s"] * aquifer_thickness
                    results["geology.transmissivity_m2_per_s"] = transmissivity
            
            # Extract bedrock depth information from soil data if available
            if "soil.regolith_thickness_mean" in results:
                results["geology.bedrock_depth_m"] = results["soil.regolith_thickness_mean"]
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error enhancing hydrogeological attributes: {str(e)}")
            return current_results  # Return the original results if there was an error

    def calculate_streamflow_signatures(self) -> Dict[str, Any]:
        """
        Calculate streamflow signatures including durations, timing, and distributions.
        
        Returns:
            Dict[str, Any]: Dictionary of streamflow signature attributes
        """
        results = {}
        
        # Load streamflow data
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not streamflow_path.exists():
            self.logger.warning(f"Streamflow data not found: {streamflow_path}")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            from scipy.stats import skew, kurtosis
            
            # Read streamflow data
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            streamflow_df.set_index('date', inplace=True)
            
            # Calculate water year
            streamflow_df['water_year'] = streamflow_df.index.year.where(
                streamflow_df.index.month < 10, 
                streamflow_df.index.year + 1
            )
            
            # Basic statistics
            results["hydrology.daily_discharge_mean"] = streamflow_df['flow'].mean()
            results["hydrology.daily_discharge_std"] = streamflow_df['flow'].std()
            
            # Calculate flow percentiles
            for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                results[f"hydrology.q{percentile}"] = np.percentile(streamflow_df['flow'].dropna(), percentile)
            
            # Calculate flow duration curve slope
            flow_sorted = np.sort(streamflow_df['flow'].dropna().values)
            if len(flow_sorted) > 0:
                # Replace zeros with a small value to enable log calculation
                flow_sorted[flow_sorted == 0] = 0.001 * results["hydrology.daily_discharge_mean"]
                
                log_flow = np.log(flow_sorted)
                q33_index = int(0.33 * len(log_flow))
                q66_index = int(0.66 * len(log_flow))
                
                if q66_index > q33_index:
                    fdc_slope = (log_flow[q66_index] - log_flow[q33_index]) / (0.66 - 0.33)
                    results["hydrology.fdc_slope"] = fdc_slope
            
            # Calculate high flow and low flow durations
            mean_flow = streamflow_df['flow'].mean()
            median_flow = streamflow_df['flow'].median()
            
            # High flow condition (Q > 9 * median)
            high_flow_threshold = 9 * median_flow
            high_flow = streamflow_df['flow'] > high_flow_threshold
            
            # Low flow condition (Q < 0.2 * mean)
            low_flow_threshold = 0.2 * mean_flow
            low_flow = streamflow_df['flow'] < low_flow_threshold
            
            # No flow condition
            no_flow = streamflow_df['flow'] <= 0
            
            # Calculate durations
            for condition, name in [(high_flow, 'high_flow'), (low_flow, 'low_flow'), (no_flow, 'no_flow')]:
                # Find durations of consecutive True values
                transitions = np.diff(np.concatenate([[False], condition, [False]]).astype(int))
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0]
                durations = ends - starts
                
                if len(durations) > 0:
                    results[f"hydrology.{name}_duration_mean"] = durations.mean()
                    results[f"hydrology.{name}_duration_median"] = np.median(durations)
                    results[f"hydrology.{name}_duration_max"] = durations.max()
                    results[f"hydrology.{name}_count_per_year"] = len(durations) / streamflow_df['water_year'].nunique()
                    results[f"hydrology.{name}_duration_skew"] = skew(durations) if len(durations) > 2 else np.nan
                    results[f"hydrology.{name}_duration_kurtosis"] = kurtosis(durations) if len(durations) > 2 else np.nan
            
            # Calculate half-flow date
            # Group by water year and find the date when 50% of annual flow occurs
            half_flow_dates = []
            
            for year, year_data in streamflow_df.groupby('water_year'):
                if year_data['flow'].sum() > 0:  # Avoid years with no flow
                    # Calculate cumulative flow for the year
                    year_data = year_data.sort_index()  # Ensure chronological order
                    year_data['cum_flow'] = year_data['flow'].cumsum()
                    year_data['frac_flow'] = year_data['cum_flow'] / year_data['flow'].sum()
                    
                    # Find the first date when fractional flow exceeds 0.5
                    half_flow_idx = year_data['frac_flow'].gt(0.5).idxmax() if any(year_data['frac_flow'] > 0.5) else None
                    
                    if half_flow_idx:
                        half_flow_dates.append(half_flow_idx.dayofyear)
            
            if half_flow_dates:
                results["hydrology.half_flow_day_mean"] = np.mean(half_flow_dates)
                results["hydrology.half_flow_day_std"] = np.std(half_flow_dates)
            
            # Calculate runoff ratio if precipitation data is available
            precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
            
            if precip_path.exists():
                precip_df = pd.read_csv(precip_path, parse_dates=['date'])
                precip_df.set_index('date', inplace=True)
                
                # Ensure same time period for both datasets
                common_period = streamflow_df.index.intersection(precip_df.index)
                if len(common_period) > 0:
                    streamflow_subset = streamflow_df.loc[common_period]
                    precip_subset = precip_df.loc[common_period]
                    
                    # Calculate runoff ratio (Q/P)
                    total_flow = streamflow_subset['flow'].sum()
                    total_precip = precip_subset['precipitation'].sum()
                    
                    if total_precip > 0:
                        runoff_ratio = total_flow / total_precip
                        results["hydrology.runoff_ratio"] = runoff_ratio
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating streamflow signatures: {str(e)}")
            return results

    def calculate_baseflow_attributes(self) -> Dict[str, Any]:
        """
        Process baseflow separation and calculate baseflow index using the Eckhardt method.
        
        Returns:
            Dict[str, Any]: Dictionary of baseflow attributes
        """
        results = {}
        
        # Load streamflow data
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not streamflow_path.exists():
            self.logger.warning(f"Streamflow data not found: {streamflow_path}")
            return results
        
        try:
            # Read streamflow data
            import pandas as pd
            import baseflow
            
            # Load streamflow data with pandas
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            
            # Apply baseflow separation using the Eckhardt method
            rec = baseflow.separation(streamflow_df, method='Eckhardt')
            
            # Calculate baseflow index (BFI)
            bfi = rec['Eckhardt']['q_bf'].mean() / rec['Eckhardt']['q_obs'].mean()
            
            results["hydrology.baseflow_index"] = bfi
            results["hydrology.baseflow_recession_constant"] = baseflow.calculate_recession_constant(streamflow_df)
            
            # Calculate seasonal baseflow indices
            seasons = {'winter': [12, 1, 2], 'spring': [3, 4, 5], 
                    'summer': [6, 7, 8], 'autumn': [9, 10, 11]}
            
            for season_name, months in seasons.items():
                seasonal_df = rec['Eckhardt'][rec['Eckhardt'].index.month.isin(months)]
                if not seasonal_df.empty:
                    seasonal_bfi = seasonal_df['q_bf'].mean() / seasonal_df['q_obs'].mean()
                    results[f"hydrology.baseflow_index_{season_name}"] = seasonal_bfi
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error calculating baseflow attributes: {str(e)}")
            return results

    def _process_lakes_data(self, hydrolakes_path: Path) -> Dict[str, Any]:
        """
        Process lake data from HydroLAKES.
        
        Args:
            hydrolakes_path: Path to HydroLAKES data
            
        Returns:
            Dictionary of lake attributes
        """
        results = {}
        
        # Look for HydroLAKES shapefile
        lakes_files = list(hydrolakes_path.glob("**/HydroLAKES_polys*.shp"))
        
        if not lakes_files:
            self.logger.warning(f"No HydroLAKES files found in {hydrolakes_path}")
            return results
        
        lakes_file = lakes_files[0]
        self.logger.info(f"Processing lake data from: {lakes_file}")
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'hydrology' / 'lakes'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a clipped version of HydroLAKES if it doesn't exist
            clipped_lakes_file = output_dir / f"{self.domain_name}_hydrolakes.gpkg"
            
            if not clipped_lakes_file.exists():
                self.logger.info(f"Clipping HydroLAKES to catchment bounding box")
                
                # Parse bounding box coordinates
                bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
                if len(bbox_coords) == 4:
                    bbox = (
                        float(bbox_coords[1]),  # lon_min
                        float(bbox_coords[2]),  # lat_min
                        float(bbox_coords[3]),  # lon_max
                        float(bbox_coords[0])   # lat_max
                    )
                    
                    # Create a spatial filter for the bbox (to avoid loading the entire dataset)
                    bbox_geom = box(*bbox)
                    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
                    
                    # Read lakes within the bounding box
                    self.logger.info(f"Reading lakes within bounding box")
                    lakes = gpd.read_file(lakes_file, bbox=bbox)
                    
                    # Save the clipped lakes
                    if not lakes.empty:
                        self.logger.info(f"Saving {len(lakes)} lakes to {clipped_lakes_file}")
                        lakes.to_file(clipped_lakes_file, driver="GPKG")
                    else:
                        self.logger.warning("No lakes found within bounding box")
                else:
                    self.logger.error("Invalid bounding box coordinates format")
            
            # Process lakes for the catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if clipped_lakes_file.exists():
                # Read the clipped lakes
                lakes = gpd.read_file(clipped_lakes_file)
                
                if lakes.empty:
                    self.logger.info("No lakes found in the study area")
                    return results
                
                # Ensure CRS match
                if lakes.crs != catchment.crs:
                    lakes = lakes.to_crs(catchment.crs)
                
                # Intersect lakes with catchment
                if is_lumped:
                    # For lumped catchment - intersect with entire catchment
                    lakes_in_catchment = gpd.overlay(lakes, catchment, how='intersection')
                    
                    # Count lakes and calculate area
                    if not lakes_in_catchment.empty:
                        # Convert to equal area projection for area calculations
                        equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                        lakes_equal_area = lakes_in_catchment.to_crs(equal_area_crs)
                        catchment_equal_area = catchment.to_crs(equal_area_crs)
                        
                        # Calculate catchment area in km²
                        catchment_area_km2 = catchment_equal_area.area.sum() / 10**6
                        
                        # Calculate lake attributes
                        num_lakes = len(lakes_in_catchment)
                        lake_area_km2 = lakes_equal_area.area.sum() / 10**6
                        lake_area_percent = (lake_area_km2 / catchment_area_km2) * 100 if catchment_area_km2 > 0 else 0
                        lake_density = num_lakes / catchment_area_km2 if catchment_area_km2 > 0 else 0
                        
                        # Add to results
                        results["hydrology.lake_count"] = num_lakes
                        results["hydrology.lake_area_km2"] = lake_area_km2
                        results["hydrology.lake_area_percent"] = lake_area_percent
                        results["hydrology.lake_density"] = lake_density  # lakes per km²
                        
                        # Calculate statistics for lake size
                        if num_lakes > 0:
                            lakes_equal_area['area_km2'] = lakes_equal_area.area / 10**6
                            results["hydrology.lake_area_min_km2"] = lakes_equal_area['area_km2'].min()
                            results["hydrology.lake_area_max_km2"] = lakes_equal_area['area_km2'].max()
                            results["hydrology.lake_area_mean_km2"] = lakes_equal_area['area_km2'].mean()
                            results["hydrology.lake_area_median_km2"] = lakes_equal_area['area_km2'].median()
                        
                        # Get volume and residence time if available
                        if 'Vol_total' in lakes_in_catchment.columns:
                            # Total volume in km³
                            total_volume_km3 = lakes_in_catchment['Vol_total'].sum() / 10**9
                            results["hydrology.lake_volume_km3"] = total_volume_km3
                        
                        if 'Depth_avg' in lakes_in_catchment.columns:
                            # Area-weighted average depth
                            weighted_depth = np.average(
                                lakes_in_catchment['Depth_avg'], 
                                weights=lakes_equal_area.area
                            )
                            results["hydrology.lake_depth_avg_m"] = weighted_depth
                        
                        if 'Res_time' in lakes_in_catchment.columns:
                            # Area-weighted average residence time
                            weighted_res_time = np.average(
                                lakes_in_catchment['Res_time'], 
                                weights=lakes_equal_area.area
                            )
                            results["hydrology.lake_residence_time_days"] = weighted_res_time
                        
                        # Largest lake attributes
                        if num_lakes > 0:
                            largest_lake_idx = lakes_equal_area['area_km2'].idxmax()
                            largest_lake = lakes_in_catchment.iloc[largest_lake_idx]
                            
                            results["hydrology.largest_lake_area_km2"] = lakes_equal_area['area_km2'].max()
                            
                            if 'Lake_name' in largest_lake:
                                results["hydrology.largest_lake_name"] = largest_lake['Lake_name']
                            
                            if 'Vol_total' in largest_lake:
                                results["hydrology.largest_lake_volume_km3"] = largest_lake['Vol_total'] / 10**9
                            
                            if 'Depth_avg' in largest_lake:
                                results["hydrology.largest_lake_depth_avg_m"] = largest_lake['Depth_avg']
                            
                            if 'Shore_len' in largest_lake:
                                results["hydrology.largest_lake_shoreline_km"] = largest_lake['Shore_len']
                            
                            if 'Res_time' in largest_lake:
                                results["hydrology.largest_lake_residence_time_days"] = largest_lake['Res_time']
                    else:
                        self.logger.info("No lakes found within catchment boundary")
                        results["hydrology.lake_count"] = 0
                        results["hydrology.lake_area_km2"] = 0
                        results["hydrology.lake_area_percent"] = 0
                        results["hydrology.lake_density"] = 0
                else:
                    # For distributed catchment - process each HRU
                    catchment_results = []
                    
                    for idx, hru in catchment.iterrows():
                        hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with lakes
                        lakes_in_hru = gpd.overlay(lakes, hru_gdf, how='intersection')
                        
                        if not lakes_in_hru.empty:
                            # Convert to equal area projection for area calculations
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            lakes_equal_area = lakes_in_hru.to_crs(equal_area_crs)
                            hru_equal_area = hru_gdf.to_crs(equal_area_crs)
                            
                            # Calculate HRU area in km²
                            hru_area_km2 = hru_equal_area.area.sum() / 10**6
                            
                            # Calculate lake attributes for this HRU
                            num_lakes = len(lakes_in_hru)
                            lake_area_km2 = lakes_equal_area.area.sum() / 10**6
                            lake_area_percent = (lake_area_km2 / hru_area_km2) * 100 if hru_area_km2 > 0 else 0
                            lake_density = num_lakes / hru_area_km2 if hru_area_km2 > 0 else 0
                            
                            # Add to results
                            results[f"{prefix}hydrology.lake_count"] = num_lakes
                            results[f"{prefix}hydrology.lake_area_km2"] = lake_area_km2
                            results[f"{prefix}hydrology.lake_area_percent"] = lake_area_percent
                            results[f"{prefix}hydrology.lake_density"] = lake_density
                            
                            # Calculate statistics for lake size if we have more than one lake
                            if num_lakes > 0:
                                lakes_equal_area['area_km2'] = lakes_equal_area.area / 10**6
                                results[f"{prefix}hydrology.lake_area_min_km2"] = lakes_equal_area['area_km2'].min()
                                results[f"{prefix}hydrology.lake_area_max_km2"] = lakes_equal_area['area_km2'].max()
                                results[f"{prefix}hydrology.lake_area_mean_km2"] = lakes_equal_area['area_km2'].mean()
                                
                                # Add largest lake attributes for this HRU
                                largest_lake_idx = lakes_equal_area['area_km2'].idxmax()
                                largest_lake = lakes_in_hru.iloc[largest_lake_idx]
                                
                                if 'Vol_total' in largest_lake:
                                    results[f"{prefix}hydrology.largest_lake_volume_km3"] = largest_lake['Vol_total'] / 10**9
                                
                                if 'Depth_avg' in largest_lake:
                                    results[f"{prefix}hydrology.largest_lake_depth_avg_m"] = largest_lake['Depth_avg']
                        else:
                            # No lakes in this HRU
                            results[f"{prefix}hydrology.lake_count"] = 0
                            results[f"{prefix}hydrology.lake_area_km2"] = 0
                            results[f"{prefix}hydrology.lake_area_percent"] = 0
                            results[f"{prefix}hydrology.lake_density"] = 0
            else:
                self.logger.warning(f"Clipped lakes file not found: {clipped_lakes_file}")
        
        except Exception as e:
            self.logger.error(f"Error processing lake data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_river_network_data(self, merit_basins_path: Path) -> Dict[str, Any]:
        """
        Process river network data from MERIT Basins.
        
        Args:
            merit_basins_path: Path to MERIT Basins data
            
        Returns:
            Dictionary of river network attributes
        """
        results = {}
        
        # Look for MERIT Basins river network shapefile
        rivers_dir = merit_basins_path / 'MERIT_Hydro_modified_North_America_shapes' / 'rivers'
        basins_dir = merit_basins_path / 'MERIT_Hydro_modified_North_America_shapes' / 'basins'
        
        if not rivers_dir.exists() or not basins_dir.exists():
            self.logger.warning(f"MERIT Basins directories not found: {rivers_dir} or {basins_dir}")
            return results
        
        # Find river network shapefile
        river_files = list(rivers_dir.glob("**/riv_*.shp"))
        basin_files = list(basins_dir.glob("**/cat_*.shp"))
        
        if not river_files or not basin_files:
            self.logger.warning(f"MERIT Basins shapefiles not found in {rivers_dir} or {basins_dir}")
            return results
        
        river_file = river_files[0]
        basin_file = basin_files[0]
        
        self.logger.info(f"Processing river network data from: {river_file}")
        self.logger.info(f"Processing basin data from: {basin_file}")
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'hydrology' / 'rivers'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a clipped version of river network if it doesn't exist
            clipped_river_file = output_dir / f"{self.domain_name}_merit_rivers.gpkg"
            clipped_basin_file = output_dir / f"{self.domain_name}_merit_basins.gpkg"
            
            # Process river network data
            if not clipped_river_file.exists() or not clipped_basin_file.exists():
                self.logger.info(f"Clipping MERIT Basins data to catchment bounding box")
                
                # Parse bounding box coordinates
                bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
                if len(bbox_coords) == 4:
                    bbox = (
                        float(bbox_coords[1]),  # lon_min
                        float(bbox_coords[2]),  # lat_min
                        float(bbox_coords[3]),  # lon_max
                        float(bbox_coords[0])   # lat_max
                    )
                    
                    # Create a spatial filter for the bbox
                    bbox_geom = box(*bbox)
                    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
                    
                    # Read rivers within the bounding box
                    self.logger.info(f"Reading rivers within bounding box")
                    rivers = gpd.read_file(river_file, bbox=bbox)
                    
                    # Read basins within the bounding box
                    self.logger.info(f"Reading basins within bounding box")
                    basins = gpd.read_file(basin_file, bbox=bbox)
                    
                    # Save the clipped data
                    if not rivers.empty:
                        self.logger.info(f"Saving {len(rivers)} river segments to {clipped_river_file}")
                        rivers.to_file(clipped_river_file, driver="GPKG")
                    else:
                        self.logger.warning("No river segments found within bounding box")
                    
                    if not basins.empty:
                        self.logger.info(f"Saving {len(basins)} basins to {clipped_basin_file}")
                        basins.to_file(clipped_basin_file, driver="GPKG")
                    else:
                        self.logger.warning("No basins found within bounding box")
                else:
                    self.logger.error("Invalid bounding box coordinates format")
            
            # Process river network for the catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if clipped_river_file.exists() and clipped_basin_file.exists():
                # Read the clipped data
                rivers = gpd.read_file(clipped_river_file)
                basins = gpd.read_file(clipped_basin_file)
                
                if rivers.empty:
                    self.logger.info("No rivers found in the study area")
                    return results
                
                # Ensure CRS match
                if rivers.crs != catchment.crs:
                    rivers = rivers.to_crs(catchment.crs)
                
                if basins.crs != catchment.crs:
                    basins = basins.to_crs(catchment.crs)
                
                # Intersect river network with catchment
                if is_lumped:
                    # For lumped catchment - intersect with entire catchment
                    rivers_in_catchment = gpd.overlay(rivers, catchment, how='intersection')
                    
                    if not rivers_in_catchment.empty:
                        # Convert to equal area projection for length calculations
                        equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                        rivers_equal_area = rivers_in_catchment.to_crs(equal_area_crs)
                        catchment_equal_area = catchment.to_crs(equal_area_crs)
                        
                        # Calculate catchment area in km²
                        catchment_area_km2 = catchment_equal_area.area.sum() / 10**6
                        
                        # Calculate river network attributes
                        num_segments = len(rivers_in_catchment)
                        total_length_km = rivers_equal_area.length.sum() / 1000  # Convert m to km
                        river_density = total_length_km / catchment_area_km2 if catchment_area_km2 > 0 else 0
                        
                        # Add to results
                        results["hydrology.river_segment_count"] = num_segments
                        results["hydrology.river_length_km"] = total_length_km
                        results["hydrology.river_density_km_per_km2"] = river_density
                        
                        # Calculate stream order statistics if available
                        if 'str_order' in rivers_in_catchment.columns:
                            results["hydrology.stream_order_min"] = int(rivers_in_catchment['str_order'].min())
                            results["hydrology.stream_order_max"] = int(rivers_in_catchment['str_order'].max())
                            
                            # Calculate frequency of each stream order
                            for order in range(1, int(rivers_in_catchment['str_order'].max()) + 1):
                                order_segments = rivers_in_catchment[rivers_in_catchment['str_order'] == order]
                                if not order_segments.empty:
                                    order_length_km = order_segments.to_crs(equal_area_crs).length.sum() / 1000
                                    results[f"hydrology.stream_order_{order}_length_km"] = order_length_km
                                    results[f"hydrology.stream_order_{order}_segments"] = len(order_segments)
                        
                        # Calculate slope statistics if available
                        if 'slope' in rivers_in_catchment.columns:
                            # Length-weighted average slope
                            weighted_slope = np.average(
                                rivers_in_catchment['slope'], 
                                weights=rivers_equal_area.length
                            )
                            results["hydrology.river_slope_mean"] = weighted_slope
                            results["hydrology.river_slope_min"] = rivers_in_catchment['slope'].min()
                            results["hydrology.river_slope_max"] = rivers_in_catchment['slope'].max()
                        
                        # Calculate sinuosity if available
                        if 'sinuosity' in rivers_in_catchment.columns:
                            # Length-weighted average sinuosity
                            weighted_sinuosity = np.average(
                                rivers_in_catchment['sinuosity'], 
                                weights=rivers_equal_area.length
                            )
                            results["hydrology.river_sinuosity_mean"] = weighted_sinuosity
                    else:
                        self.logger.info("No river segments found within catchment boundary")
                        results["hydrology.river_segment_count"] = 0
                        results["hydrology.river_length_km"] = 0
                        results["hydrology.river_density_km_per_km2"] = 0
                else:
                    # For distributed catchment - process each HRU
                    for idx, hru in catchment.iterrows():
                        hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with rivers
                        rivers_in_hru = gpd.overlay(rivers, hru_gdf, how='intersection')
                        
                        if not rivers_in_hru.empty:
                            # Convert to equal area projection for length calculations
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            rivers_equal_area = rivers_in_hru.to_crs(equal_area_crs)
                            hru_equal_area = hru_gdf.to_crs(equal_area_crs)
                            
                            # Calculate HRU area in km²
                            hru_area_km2 = hru_equal_area.area.sum() / 10**6
                            
                            # Calculate river network attributes for this HRU
                            num_segments = len(rivers_in_hru)
                            total_length_km = rivers_equal_area.length.sum() / 1000
                            river_density = total_length_km / hru_area_km2 if hru_area_km2 > 0 else 0
                            
                            # Add to results
                            results[f"{prefix}hydrology.river_segment_count"] = num_segments
                            results[f"{prefix}hydrology.river_length_km"] = total_length_km
                            results[f"{prefix}hydrology.river_density_km_per_km2"] = river_density
                            
                            # Calculate stream order statistics if available
                            if 'str_order' in rivers_in_hru.columns:
                                results[f"{prefix}hydrology.stream_order_min"] = int(rivers_in_hru['str_order'].min())
                                results[f"{prefix}hydrology.stream_order_max"] = int(rivers_in_hru['str_order'].max())
                            
                            # Calculate slope statistics if available
                            if 'slope' in rivers_in_hru.columns:
                                # Length-weighted average slope
                                weighted_slope = np.average(
                                    rivers_in_hru['slope'], 
                                    weights=rivers_equal_area.length
                                )
                                results[f"{prefix}hydrology.river_slope_mean"] = weighted_slope
                        else:
                            # No rivers in this HRU
                            results[f"{prefix}hydrology.river_segment_count"] = 0
                            results[f"{prefix}hydrology.river_length_km"] = 0
                            results[f"{prefix}hydrology.river_density_km_per_km2"] = 0
            else:
                self.logger.warning(f"Clipped river network files not found")
        
        except Exception as e:
            self.logger.error(f"Error processing river network data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _process_climate_attributes(self) -> Dict[str, Any]:
        """
        Process climate data including temperature, precipitation, potential evapotranspiration,
        and derived climate indices.
        
        Sources:
        - WorldClim raw data: Monthly temperature, precipitation, radiation, wind, vapor pressure
        - WorldClim derived: PET, moisture index, climate indices, snow
        """
        results = {}
        
        # Find and check paths for climate data
        worldclim_path = self._get_data_path('ATTRIBUTES_WORLDCLIM_PATH', 'worldclim')
        
        self.logger.info(f"WorldClim path: {worldclim_path}")
        
        # Process raw climate variables
        raw_climate = self._process_raw_climate_variables(worldclim_path)
        results.update(raw_climate)
        
        # Process derived climate indices
        derived_climate = self._process_derived_climate_indices(worldclim_path)
        results.update(derived_climate)
        
        return results

    def _process_raw_climate_variables(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process raw climate variables from WorldClim.
        
        Args:
            worldclim_path: Path to WorldClim data
            
        Returns:
            Dictionary of climate attributes
        """
        results = {}
        output_dir = self.project_dir / 'attributes' / 'climate'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the raw climate variables to process
        raw_variables = {
            'prec': {'unit': 'mm/month', 'description': 'Precipitation'},
            'tavg': {'unit': '°C', 'description': 'Average Temperature'},
            'tmax': {'unit': '°C', 'description': 'Maximum Temperature'},
            'tmin': {'unit': '°C', 'description': 'Minimum Temperature'},
            'srad': {'unit': 'kJ/m²/day', 'description': 'Solar Radiation'},
            'wind': {'unit': 'm/s', 'description': 'Wind Speed'},
            'vapr': {'unit': 'kPa', 'description': 'Vapor Pressure'}
        }
        
        # Process each climate variable
        for var, var_info in raw_variables.items():
            var_path = worldclim_path / 'raw' / var
            
            if not var_path.exists():
                self.logger.warning(f"WorldClim {var} directory not found: {var_path}")
                continue
            
            self.logger.info(f"Processing WorldClim {var_info['description']} ({var})")
            
            # Find all monthly files for this variable
            monthly_files = sorted(var_path.glob(f"wc2.1_30s_{var}_*.tif"))
            
            if not monthly_files:
                self.logger.warning(f"No {var} files found in {var_path}")
                continue
            
            # Process monthly data
            monthly_values = []
            monthly_attributes = {}
            
            for month_file in monthly_files:
                # Extract month number from filename (assuming format wc2.1_30s_var_MM.tif)
                month_str = os.path.basename(month_file).split('_')[-1].split('.')[0]
                try:
                    month = int(month_str)
                except ValueError:
                    self.logger.warning(f"Could not extract month from filename: {month_file}")
                    continue
                
                # Clip to catchment if not already done
                month_name = os.path.basename(month_file)
                clipped_file = output_dir / f"{self.domain_name}_{month_name}"
                
                if not clipped_file.exists():
                    self.logger.info(f"Clipping {var} for month {month} to catchment")
                    self._clip_raster_to_bbox(month_file, clipped_file)
                
                # Calculate zonal statistics for this month
                if clipped_file.exists():
                    try:
                        stats = ['mean', 'min', 'max', 'std']
                        zonal_out = zonal_stats(
                            str(self.catchment_path),
                            str(clipped_file),
                            stats=stats,
                            all_touched=True
                        )
                        
                        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                        
                        if is_lumped:
                            # For lumped catchment, add monthly values
                            if zonal_out and len(zonal_out) > 0:
                                monthly_values.append(zonal_out[0].get('mean', np.nan))
                                
                                for stat in stats:
                                    if stat in zonal_out[0]:
                                        monthly_attributes[f"climate.{var}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                        else:
                            # For distributed catchment
                            catchment = gpd.read_file(self.catchment_path)
                            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                            
                            # Process each HRU
                            for i, zonal_result in enumerate(zonal_out):
                                if i < len(catchment):
                                    hru_id = catchment.iloc[i][hru_id_field]
                                    prefix = f"HRU_{hru_id}_"
                                    
                                    for stat in stats:
                                        if stat in zonal_result:
                                            monthly_attributes[f"{prefix}climate.{var}_m{month:02d}_{stat}"] = zonal_result[stat]
                                    
                                    # Keep track of monthly values for each HRU for annual calculations
                                    if i >= len(monthly_values):
                                        monthly_values.append([])
                                    if len(monthly_values[i]) < month:
                                        # Fill with NaNs if we skipped months
                                        monthly_values[i].extend([np.nan] * (month - len(monthly_values[i]) - 1))
                                        monthly_values[i].append(zonal_result.get('mean', np.nan))
                                    else:
                                        monthly_values[i][month-1] = zonal_result.get('mean', np.nan)
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {var} for month {month}: {str(e)}")
            
            # Add monthly attributes to results
            results.update(monthly_attributes)
            
            # Calculate annual statistics
            if monthly_values:
                if is_lumped:
                    # For lumped catchment
                    monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
                    if monthly_values_clean:
                        # Annual mean
                        annual_mean = np.mean(monthly_values_clean)
                        results[f"climate.{var}_annual_mean"] = annual_mean
                        
                        # Annual range (max - min)
                        annual_range = np.max(monthly_values_clean) - np.min(monthly_values_clean)
                        results[f"climate.{var}_annual_range"] = annual_range
                        
                        # Seasonality metrics
                        if len(monthly_values_clean) >= 12:
                            # Calculate seasonality index (standard deviation / mean)
                            if annual_mean > 0:
                                seasonality_index = np.std(monthly_values_clean) / annual_mean
                                results[f"climate.{var}_seasonality_index"] = seasonality_index
                            
                            # For precipitation: calculate precipitation seasonality
                            if var == 'prec':
                                # Walsh & Lawler (1981) seasonality index
                                # 1/P * sum(|pi - P/12|) where P is annual precip, pi is monthly precip
                                annual_total = np.sum(monthly_values_clean)
                                if annual_total > 0:
                                    monthly_diff = [abs(p - annual_total/12) for p in monthly_values_clean]
                                    walsh_index = sum(monthly_diff) / annual_total
                                    results["climate.prec_walsh_seasonality_index"] = walsh_index
                                
                                # Markham Seasonality Index - vector-based measure
                                # Each month is treated as a vector with magnitude=precip, direction=month angle
                                if len(monthly_values_clean) == 12:
                                    month_angles = np.arange(0, 360, 30) * np.pi / 180  # in radians
                                    x_sum = sum(p * np.sin(a) for p, a in zip(monthly_values_clean, month_angles))
                                    y_sum = sum(p * np.cos(a) for p, a in zip(monthly_values_clean, month_angles))
                                    
                                    # Markham concentration index (0-1, higher = more seasonal)
                                    vector_magnitude = np.sqrt(x_sum**2 + y_sum**2)
                                    markham_index = vector_magnitude / annual_total
                                    results["climate.prec_markham_seasonality_index"] = markham_index
                                    
                                    # Markham seasonality angle (direction of concentration)
                                    seasonality_angle = np.arctan2(x_sum, y_sum) * 180 / np.pi
                                    if seasonality_angle < 0:
                                        seasonality_angle += 360
                                    
                                    # Convert to month (1-12)
                                    seasonality_month = round(seasonality_angle / 30) % 12
                                    if seasonality_month == 0:
                                        seasonality_month = 12
                                    
                                    results["climate.prec_seasonality_month"] = seasonality_month
                else:
                    # For distributed catchment
                    for i, hru_values in enumerate(monthly_values):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            hru_values_clean = [v for v in hru_values if not np.isnan(v)]
                            if hru_values_clean:
                                # Annual mean
                                annual_mean = np.mean(hru_values_clean)
                                results[f"{prefix}climate.{var}_annual_mean"] = annual_mean
                                
                                # Annual range
                                annual_range = np.max(hru_values_clean) - np.min(hru_values_clean)
                                results[f"{prefix}climate.{var}_annual_range"] = annual_range
                                
                                # Seasonality metrics
                                if len(hru_values_clean) >= 12:
                                    # Calculate seasonality index
                                    if annual_mean > 0:
                                        seasonality_index = np.std(hru_values_clean) / annual_mean
                                        results[f"{prefix}climate.{var}_seasonality_index"] = seasonality_index
                                    
                                    # For precipitation: calculate precipitation seasonality
                                    if var == 'prec':
                                        # Walsh & Lawler seasonality index
                                        annual_total = np.sum(hru_values_clean)
                                        if annual_total > 0:
                                            monthly_diff = [abs(p - annual_total/12) for p in hru_values_clean]
                                            walsh_index = sum(monthly_diff) / annual_total
                                            results[f"{prefix}climate.prec_walsh_seasonality_index"] = walsh_index
            
            # Calculate derived attributes based on variable
            if var == 'prec' and 'climate.prec_annual_mean' in results:
                # Define aridity zones based on annual precipitation
                aridity_zones = {
                    'hyperarid': (0, 100),
                    'arid': (100, 400),
                    'semiarid': (400, 600),
                    'subhumid': (600, 1000),
                    'humid': (1000, 2000),
                    'superhumid': (2000, float('inf'))
                }
                
                if is_lumped:
                    annual_precip = results['climate.prec_annual_mean']
                    for zone, (lower, upper) in aridity_zones.items():
                        if lower <= annual_precip < upper:
                            results['climate.aridity_zone_precip'] = zone
                            break
                else:
                    for i in range(len(catchment)):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        key = f"{prefix}climate.prec_annual_mean"
                        
                        if key in results:
                            annual_precip = results[key]
                            for zone, (lower, upper) in aridity_zones.items():
                                if lower <= annual_precip < upper:
                                    results[f"{prefix}climate.aridity_zone_precip"] = zone
                                    break
        
        return results

    def _process_derived_climate_indices(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process derived climate indices from WorldClim.
        
        Args:
            worldclim_path: Path to WorldClim data
            
        Returns:
            Dictionary of climate indices
        """
        results = {}
        output_dir = self.project_dir / 'attributes' / 'climate' / 'derived'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for derived data directories
        derived_path = worldclim_path / 'derived'
        if not derived_path.exists():
            self.logger.warning(f"WorldClim derived data directory not found: {derived_path}")
            return results
        
        # Define the derived products to process
        derived_products = [
            {
                'name': 'pet',
                'path': derived_path / 'pet',
                'pattern': 'wc2.1_30s_pet_*.tif',
                'description': 'Potential Evapotranspiration',
                'unit': 'mm/day'
            },
            {
                'name': 'moisture_index',
                'path': derived_path / 'moisture_index',
                'pattern': 'wc2.1_30s_moisture_index_*.tif',
                'description': 'Moisture Index',
                'unit': '-'
            },
            {
                'name': 'snow',
                'path': derived_path / 'snow',
                'pattern': 'wc2.1_30s_snow_*.tif',
                'description': 'Snow',
                'unit': 'mm/month'
            },
            {
                'name': 'climate_index_im',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_im.tif',
                'description': 'Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_imr',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_imr.tif',
                'description': 'Relative Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_fs',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_fs.tif',
                'description': 'Fraction of Precipitation as Snow',
                'unit': '-'
            }
        ]
        
        # Process each derived product
        for product in derived_products:
            product_path = product['path']
            
            if not product_path.exists():
                self.logger.warning(f"WorldClim {product['description']} directory not found: {product_path}")
                continue
            
            self.logger.info(f"Processing WorldClim {product['description']} ({product['name']})")
            
            # Find files for this product
            product_files = sorted(product_path.glob(product['pattern']))
            
            if not product_files:
                self.logger.warning(f"No {product['name']} files found in {product_path}")
                continue
            
            # Process monthly data for products with monthly files
            if len(product_files) > 1 and any('_01.' in file.name for file in product_files):
                # This is a monthly product
                monthly_values = []
                monthly_attributes = {}
                
                for month_file in product_files:
                    # Extract month number
                    if '_' in month_file.name and '.' in month_file.name:
                        month_part = month_file.name.split('_')[-1].split('.')[0]
                        try:
                            month = int(month_part)
                        except ValueError:
                            self.logger.warning(f"Could not extract month from filename: {month_file}")
                            continue
                    
                        # Clip to catchment if not already done
                        month_name = os.path.basename(month_file)
                        clipped_file = output_dir / f"{self.domain_name}_{month_name}"
                        
                        if not clipped_file.exists():
                            self.logger.info(f"Clipping {product['name']} for month {month} to catchment")
                            self._clip_raster_to_bbox(month_file, clipped_file)
                        
                        # Calculate zonal statistics for this month
                        if clipped_file.exists():
                            try:
                                stats = ['mean', 'min', 'max', 'std']
                                zonal_out = zonal_stats(
                                    str(self.catchment_path),
                                    str(clipped_file),
                                    stats=stats,
                                    all_touched=True
                                )
                                
                                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                                
                                if is_lumped:
                                    # For lumped catchment
                                    if zonal_out and len(zonal_out) > 0:
                                        monthly_values.append(zonal_out[0].get('mean', np.nan))
                                        
                                        for stat in stats:
                                            if stat in zonal_out[0]:
                                                monthly_attributes[f"climate.{product['name']}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                                else:
                                    # For distributed catchment
                                    catchment = gpd.read_file(self.catchment_path)
                                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                                    
                                    # Process each HRU
                                    for i, zonal_result in enumerate(zonal_out):
                                        if i < len(catchment):
                                            hru_id = catchment.iloc[i][hru_id_field]
                                            prefix = f"HRU_{hru_id}_"
                                            
                                            for stat in stats:
                                                if stat in zonal_result:
                                                    monthly_attributes[f"{prefix}climate.{product['name']}_m{month:02d}_{stat}"] = zonal_result[stat]
                            
                            except Exception as e:
                                self.logger.error(f"Error processing {product['name']} for month {month}: {str(e)}")
                
                # Add monthly attributes to results
                results.update(monthly_attributes)
                
                # Calculate annual statistics for monthly products
                if monthly_values and is_lumped:
                    monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
                    if monthly_values_clean:
                        # Annual statistics
                        results[f"climate.{product['name']}_annual_mean"] = np.mean(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_min"] = np.min(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_max"] = np.max(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_range"] = np.max(monthly_values_clean) - np.min(monthly_values_clean)
                        
                        # Calculate seasonality index for applicable variables
                        if product['name'] in ['pet', 'snow']:
                            annual_mean = results[f"climate.{product['name']}_annual_mean"]
                            if annual_mean > 0:
                                seasonality_index = np.std(monthly_values_clean) / annual_mean
                                results[f"climate.{product['name']}_seasonality_index"] = seasonality_index
            else:
                # This is an annual/static product (like climate indices)
                for product_file in product_files:
                    # Clip to catchment if not already done
                    file_name = os.path.basename(product_file)
                    clipped_file = output_dir / f"{self.domain_name}_{file_name}"
                    
                    if not clipped_file.exists():
                        self.logger.info(f"Clipping {product['name']} to catchment")
                        self._clip_raster_to_bbox(product_file, clipped_file)
                    
                    # Calculate zonal statistics
                    if clipped_file.exists():
                        try:
                            stats = ['mean', 'min', 'max', 'std']
                            zonal_out = zonal_stats(
                                str(self.catchment_path),
                                str(clipped_file),
                                stats=stats,
                                all_touched=True
                            )
                            
                            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                            
                            if is_lumped:
                                # For lumped catchment
                                if zonal_out and len(zonal_out) > 0:
                                    for stat in stats:
                                        if stat in zonal_out[0]:
                                            results[f"climate.{product['name']}_{stat}"] = zonal_out[0][stat]
                            else:
                                # For distributed catchment
                                catchment = gpd.read_file(self.catchment_path)
                                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                                
                                # Process each HRU
                                for i, zonal_result in enumerate(zonal_out):
                                    if i < len(catchment):
                                        hru_id = catchment.iloc[i][hru_id_field]
                                        prefix = f"HRU_{hru_id}_"
                                        
                                        for stat in stats:
                                            if stat in zonal_result:
                                                results[f"{prefix}climate.{product['name']}_{stat}"] = zonal_result[stat]
                        
                        except Exception as e:
                            self.logger.error(f"Error processing {product['name']}: {str(e)}")
        
        # Calculate derived climate indicators if we have the necessary data
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        if is_lumped:
            # Aridity index (PET/P) - if we have both annual precipitation and PET
            if "climate.prec_annual_mean" in results and "climate.pet_annual_mean" in results:
                precip = results["climate.prec_annual_mean"]
                pet = results["climate.pet_annual_mean"]
                
                if precip > 0:
                    aridity_index = pet / precip
                    results["climate.aridity_index"] = aridity_index
                    
                    # Classify aridity zone based on UNEP criteria
                    if aridity_index < 0.03:
                        results["climate.aridity_zone"] = "hyperhumid"
                    elif aridity_index < 0.2:
                        results["climate.aridity_zone"] = "humid"
                    elif aridity_index < 0.5:
                        results["climate.aridity_zone"] = "subhumid"
                    elif aridity_index < 0.65:
                        results["climate.aridity_zone"] = "dry_subhumid"
                    elif aridity_index < 1.0:
                        results["climate.aridity_zone"] = "semiarid"
                    elif aridity_index < 3.0:
                        results["climate.aridity_zone"] = "arid"
                    else:
                        results["climate.aridity_zone"] = "hyperarid"
            
            # Snow fraction - total annual snow / total annual precipitation
            if "climate.snow_annual_mean" in results and "climate.prec_annual_mean" in results:
                snow = results["climate.snow_annual_mean"] * 12  # Convert mean to annual total
                precip = results["climate.prec_annual_mean"] * 12
                
                if precip > 0:
                    snow_fraction = snow / precip
                    results["climate.snow_fraction"] = snow_fraction
        else:
            # For distributed catchment, calculate indices for each HRU
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # Aridity index
                prec_key = f"{prefix}climate.prec_annual_mean"
                pet_key = f"{prefix}climate.pet_annual_mean"
                
                if prec_key in results and pet_key in results:
                    precip = results[prec_key]
                    pet = results[pet_key]
                    
                    if precip > 0:
                        aridity_index = pet / precip
                        results[f"{prefix}climate.aridity_index"] = aridity_index
                        
                        # Classify aridity zone
                        if aridity_index < 0.03:
                            results[f"{prefix}climate.aridity_zone"] = "hyperhumid"
                        elif aridity_index < 0.2:
                            results[f"{prefix}climate.aridity_zone"] = "humid"
                        elif aridity_index < 0.5:
                            results[f"{prefix}climate.aridity_zone"] = "subhumid"
                        elif aridity_index < 0.65:
                            results[f"{prefix}climate.aridity_zone"] = "dry_subhumid"
                        elif aridity_index < 1.0:
                            results[f"{prefix}climate.aridity_zone"] = "semiarid"
                        elif aridity_index < 3.0:
                            results[f"{prefix}climate.aridity_zone"] = "arid"
                        else:
                            results[f"{prefix}climate.aridity_zone"] = "hyperarid"
                
                # Snow fraction
                snow_key = f"{prefix}climate.snow_annual_mean"
                if snow_key in results and prec_key in results:
                    snow = results[snow_key] * 12
                    precip = results[prec_key] * 12
                    
                    if precip > 0:
                        snow_fraction = snow / precip
                        results[f"{prefix}climate.snow_fraction"] = snow_fraction
        
        return results


    def _process_landcover_vegetation_attributes(self) -> Dict[str, Any]:
        """
        Process land cover and vegetation attributes from multiple datasets,
        including GLCLU2019, MODIS LAI, and others.
        
        Returns:
            Dict[str, Any]: Dictionary of land cover and vegetation attributes
        """
        results = {}
        
        # Process GLCLU2019 land cover classification
        glclu_results = self._process_glclu2019_landcover()
        results.update(glclu_results)
        
        # Process LAI (Leaf Area Index) data
        lai_results = self._process_lai_data()
        results.update(lai_results)
        
        # Process forest height data
        forest_results = self._process_forest_height()
        results.update(forest_results)
        
        return results

    def _process_glclu2019_landcover(self) -> Dict[str, Any]:
        """
        Process land cover data from the GLCLU2019 dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of land cover attributes
        """
        results = {}
        
        # Define path to GLCLU2019 data
        glclu_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/glclu2019/raw")
        main_tif = glclu_path / "glclu2019_map.tif"
        
        # Check if file exists
        if not main_tif.exists():
            self.logger.warning(f"GLCLU2019 file not found: {main_tif}")
            return results
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'landcover'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glclu2019_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached GLCLU2019 results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached GLCLU2019 results: {str(e)}")
        
        self.logger.info("Processing GLCLU2019 land cover data")
        
        try:
            # Define the land cover classes from GLCLU2019
            landcover_classes = {
                1: 'true_desert',
                2: 'semi_arid',
                3: 'dense_short_vegetation',
                4: 'open_tree_cover',
                5: 'dense_tree_cover',
                6: 'tree_cover_gain',
                7: 'tree_cover_loss',
                8: 'salt_pan',
                9: 'wetland_sparse_vegetation',
                10: 'wetland_dense_short_vegetation',
                11: 'wetland_open_tree_cover',
                12: 'wetland_dense_tree_cover',
                13: 'wetland_tree_cover_gain',
                14: 'wetland_tree_cover_loss',
                15: 'ice',
                16: 'water',
                17: 'cropland',
                18: 'built_up',
                19: 'ocean',
                20: 'no_data'
            }
            
            # Define broader categories for aggregation
            category_mapping = {
                'forest': [4, 5, 6, 11, 12, 13],  # Tree cover classes
                'wetland': [9, 10, 11, 12, 13, 14],  # Wetland classes
                'barren': [1, 2, 8],  # Desert and semi-arid classes
                'water': [15, 16, 19],  # Water and ice classes
                'agricultural': [17],  # Cropland
                'urban': [18],  # Built-up areas
                'other_vegetation': [3]  # Other vegetation
            }
            
            # Calculate zonal statistics (categorical)
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(main_tif), 
                categorical=True, 
                nodata=20,  # 'no_data' class
                all_touched=True
            )
            
            # Check if we have valid results
            if not zonal_out:
                self.logger.warning("No valid zonal statistics for GLCLU2019")
                return results
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out and len(zonal_out) > 0:
                    # Calculate total pixels (excluding 'no_data')
                    valid_pixels = sum(count for class_id, count in zonal_out[0].items() 
                                    if class_id is not None and class_id != 20)
                    
                    if valid_pixels > 0:
                        # Calculate fractions for each land cover class
                        for class_id, class_name in landcover_classes.items():
                            if class_id == 20:  # Skip 'no_data'
                                continue
                            
                            pixel_count = zonal_out[0].get(class_id, 0)
                            fraction = pixel_count / valid_pixels if valid_pixels > 0 else 0
                            results[f"landcover.{class_name}_fraction"] = fraction
                        
                        # Calculate broader category fractions
                        for category, class_ids in category_mapping.items():
                            category_count = sum(zonal_out[0].get(class_id, 0) for class_id in class_ids)
                            fraction = category_count / valid_pixels if valid_pixels > 0 else 0
                            results[f"landcover.{category}_fraction"] = fraction
                        
                        # Find dominant land cover class
                        dominant_class_id = max(
                            ((class_id, count) for class_id, count in zonal_out[0].items() 
                            if class_id is not None and class_id != 20),
                            key=lambda x: x[1],
                            default=(None, 0)
                        )[0]
                        
                        if dominant_class_id is not None:
                            results["landcover.dominant_class"] = landcover_classes[dominant_class_id]
                            results["landcover.dominant_class_id"] = dominant_class_id
                            results["landcover.dominant_fraction"] = zonal_out[0][dominant_class_id] / valid_pixels
                        
                        # Calculate land cover diversity (Shannon entropy)
                        shannon_entropy = 0
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None and class_id != 20 and count > 0:
                                p = count / valid_pixels
                                shannon_entropy -= p * np.log(p)
                        results["landcover.diversity_index"] = shannon_entropy
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate total pixels (excluding 'no_data')
                        valid_pixels = sum(count for class_id, count in zonal_result.items() 
                                        if class_id is not None and class_id != 20)
                        
                        if valid_pixels > 0:
                            # Calculate fractions for each land cover class
                            for class_id, class_name in landcover_classes.items():
                                if class_id == 20:  # Skip 'no_data'
                                    continue
                                
                                pixel_count = zonal_result.get(class_id, 0)
                                fraction = pixel_count / valid_pixels if valid_pixels > 0 else 0
                                results[f"{prefix}landcover.{class_name}_fraction"] = fraction
                            
                            # Calculate broader category fractions
                            for category, class_ids in category_mapping.items():
                                category_count = sum(zonal_result.get(class_id, 0) for class_id in class_ids)
                                fraction = category_count / valid_pixels if valid_pixels > 0 else 0
                                results[f"{prefix}landcover.{category}_fraction"] = fraction
                            
                            # Find dominant land cover class
                            dominant_class_id = max(
                                ((class_id, count) for class_id, count in zonal_result.items() 
                                if class_id is not None and class_id != 20),
                                key=lambda x: x[1],
                                default=(None, 0)
                            )[0]
                            
                            if dominant_class_id is not None:
                                results[f"{prefix}landcover.dominant_class"] = landcover_classes[dominant_class_id]
                                results[f"{prefix}landcover.dominant_class_id"] = dominant_class_id
                                results[f"{prefix}landcover.dominant_fraction"] = zonal_result[dominant_class_id] / valid_pixels
                            
                            # Calculate land cover diversity (Shannon entropy)
                            shannon_entropy = 0
                            for class_id, count in zonal_result.items():
                                if class_id is not None and class_id != 20 and count > 0:
                                    p = count / valid_pixels
                                    shannon_entropy -= p * np.log(p)
                            results[f"{prefix}landcover.diversity_index"] = shannon_entropy
            
            # Try to read additional information from the Excel legend if available
            legend_file = glclu_path / "legend_0.xlsx"
            if legend_file.exists():
                try:
                    import pandas as pd
                    legend_df = pd.read_excel(legend_file)
                    self.logger.info(f"Loaded GLCLU2019 legend with {len(legend_df)} entries")
                    # Add any additional processing based on legend information
                except Exception as e:
                    self.logger.warning(f"Error reading GLCLU2019 legend: {str(e)}")
            
            # Cache results
            try:
                self.logger.info(f"Caching GLCLU2019 results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching GLCLU2019 results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing GLCLU2019 data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_lai_data(self) -> Dict[str, Any]:
        """
        Process Leaf Area Index (LAI) data from MODIS.
        
        Returns:
            Dict[str, Any]: Dictionary of LAI attributes
        """
        results = {}
        
        # Define path to LAI data
        lai_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/lai/monthly_average_2013_2023")
        
        # Check if we should use water-masked LAI (preferred) or non-water-masked
        use_water_mask = self.config.get('USE_WATER_MASKED_LAI', True)
        
        if use_water_mask:
            lai_folder = lai_path / 'monthly_lai_with_water_mask'
        else:
            lai_folder = lai_path / 'monthly_lai_no_water_mask'
        
        # Check if folder exists
        if not lai_folder.exists():
            self.logger.warning(f"LAI folder not found: {lai_folder}")
            # Try alternative folder if preferred one doesn't exist
            if use_water_mask:
                lai_folder = lai_path / 'monthly_lai_no_water_mask'
            else:
                lai_folder = lai_path / 'monthly_lai_with_water_mask'
            
            if not lai_folder.exists():
                self.logger.warning(f"Alternative LAI folder not found: {lai_folder}")
                return results
            
            self.logger.info(f"Using alternative LAI folder: {lai_folder}")
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'lai'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_lai_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached LAI results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached LAI results: {str(e)}")
        
        self.logger.info("Processing LAI data")
        
        try:
            # Find monthly LAI files
            lai_files = sorted(lai_folder.glob("*.tif"))
            
            if not lai_files:
                self.logger.warning(f"No LAI files found in {lai_folder}")
                return results
            
            # Process each monthly LAI file
            monthly_lai_values = []  # For seasonal analysis
            
            for lai_file in lai_files:
                # Extract month from filename (e.g., "2013_2023_01_MOD_Grid_MOD15A2H_Lai_500m.tif")
                filename = lai_file.name
                if '_' in filename:
                    try:
                        month = int(filename.split('_')[2])
                    except (IndexError, ValueError):
                        self.logger.warning(f"Could not extract month from LAI filename: {filename}")
                        continue
                else:
                    self.logger.warning(f"Unexpected LAI filename format: {filename}")
                    continue
                
                self.logger.info(f"Processing LAI for month {month}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(lai_file), 
                    stats=['mean', 'min', 'max', 'std', 'median', 'count'], 
                    nodata=255,  # Common MODIS no-data value
                    all_touched=True
                )
                
                # Check if we have valid results
                if not zonal_out:
                    self.logger.warning(f"No valid zonal statistics for LAI month {month}")
                    continue
                
                # Get scale and offset
                scale, offset = self._read_scale_and_offset(lai_file)
                if scale is None:
                    scale = 0.1  # Default scale for MODIS LAI (convert to actual LAI values)
                if offset is None:
                    offset = 0
                
                # Check if we're dealing with lumped or distributed catchment
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0 and 'mean' in zonal_out[0]:
                        # Apply scale and offset
                        for stat in ['mean', 'min', 'max', 'std', 'median']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                value = zonal_out[0][stat] * scale + offset
                                results[f"vegetation.lai_month{month:02d}_{stat}"] = value
                        
                        # Store mean LAI for seasonal analysis
                        if 'mean' in zonal_out[0] and zonal_out[0]['mean'] is not None:
                            monthly_lai_values.append((month, zonal_out[0]['mean'] * scale + offset))
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Apply scale and offset
                            for stat in ['mean', 'min', 'max', 'std', 'median']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat] * scale + offset
                                    results[f"{prefix}vegetation.lai_month{month:02d}_{stat}"] = value
            
            # Calculate seasonal metrics from monthly values
            if is_lumped and monthly_lai_values:
                # Sort by month for correct seasonal analysis
                monthly_lai_values.sort(key=lambda x: x[0])
                
                # Extract LAI values
                lai_values = [v[1] for v in monthly_lai_values if not np.isnan(v[1])]
                
                if lai_values:
                    # Annual LAI statistics
                    results["vegetation.lai_annual_min"] = min(lai_values)
                    results["vegetation.lai_annual_mean"] = sum(lai_values) / len(lai_values)
                    results["vegetation.lai_annual_max"] = max(lai_values)
                    results["vegetation.lai_annual_range"] = max(lai_values) - min(lai_values)
                    
                    # Seasonal amplitude (difference between max and min)
                    results["vegetation.lai_seasonal_amplitude"] = results["vegetation.lai_annual_range"]
                    
                    # Calculate seasonality index (coefficient of variation)
                    if results["vegetation.lai_annual_mean"] > 0:
                        lai_std = np.std(lai_values)
                        results["vegetation.lai_seasonality_index"] = lai_std / results["vegetation.lai_annual_mean"]
                    
                    # Growing season metrics
                    # Define growing season as months with LAI >= 50% of annual range above minimum
                    threshold = results["vegetation.lai_annual_min"] + 0.5 * results["vegetation.lai_annual_range"]
                    growing_season_months = [month for month, lai in monthly_lai_values if lai >= threshold]
                    
                    if growing_season_months:
                        results["vegetation.growing_season_months"] = len(growing_season_months)
                        results["vegetation.growing_season_start"] = min(growing_season_months)
                        results["vegetation.growing_season_end"] = max(growing_season_months)
                    
                    # Phenology timing
                    max_month = [m for m, l in monthly_lai_values if l == results["vegetation.lai_annual_max"]]
                    min_month = [m for m, l in monthly_lai_values if l == results["vegetation.lai_annual_min"]]
                    
                    if max_month:
                        results["vegetation.lai_peak_month"] = max_month[0]
                    if min_month:
                        results["vegetation.lai_min_month"] = min_month[0]
                    
                    # Vegetation density classification
                    if results["vegetation.lai_annual_max"] < 1:
                        results["vegetation.density_class"] = "sparse"
                    elif results["vegetation.lai_annual_max"] < 3:
                        results["vegetation.density_class"] = "moderate"
                    else:
                        results["vegetation.density_class"] = "dense"
            
            # Cache results
            try:
                self.logger.info(f"Caching LAI results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching LAI results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing LAI data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_forest_height(self) -> Dict[str, Any]:
        """
        Process forest height data.
        
        Returns:
            Dict[str, Any]: Dictionary of forest height attributes
        """
        results = {}
        
        # Define path to forest height data
        forest_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/forest_height/raw")
        
        # Check for 2000 and 2020 forest height files
        forest_2000_file = forest_dir / "forest_height_2000.tif"
        forest_2020_file = forest_dir / "forest_height_2020.tif"
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'forest_height'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_forest_height_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached forest height results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached forest height results: {str(e)}")
        
        self.logger.info("Processing forest height data")
        
        try:
            years = []
            
            # Check which years are available
            if forest_2000_file.exists():
                years.append(("2000", forest_2000_file))
            
            if forest_2020_file.exists():
                years.append(("2020", forest_2020_file))
            
            if not years:
                self.logger.warning("No forest height files found")
                return results
            
            # Process each year's forest height data
            for year_str, forest_file in years:
                self.logger.info(f"Processing forest height for year {year_str}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(forest_file), 
                    stats=['mean', 'min', 'max', 'std', 'median', 'count'], 
                    nodata=-9999,  # Common no-data value for forest height
                    all_touched=True
                )
                
                # Check if we have valid results
                if not zonal_out:
                    self.logger.warning(f"No valid zonal statistics for forest height year {year_str}")
                    continue
                
                # Get scale and offset
                scale, offset = self._read_scale_and_offset(forest_file)
                if scale is None:
                    scale = 1  # Default scale
                if offset is None:
                    offset = 0
                
                # Check if we're dealing with lumped or distributed catchment
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0 and 'mean' in zonal_out[0]:
                        # Apply scale and offset to basic statistics
                        for stat in ['mean', 'min', 'max', 'std', 'median']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                value = zonal_out[0][stat] * scale + offset
                                results[f"forest.height_{year_str}_{stat}"] = value
                        
                        # Calculate forest coverage
                        if 'count' in zonal_out[0] and zonal_out[0]['count'] is not None:
                            # Try to count total pixels in catchment
                            with rasterio.open(str(forest_file)) as src:
                                catchment = gpd.read_file(self.catchment_path)
                                geom = catchment.geometry.values[0]
                                if catchment.crs != src.crs:
                                    # Transform catchment to raster CRS
                                    catchment = catchment.to_crs(src.crs)
                                    geom = catchment.geometry.values[0]
                                
                                # Get the mask of pixels within the catchment
                                mask, transform = rasterio.mask.mask(src, [geom], crop=False, nodata=0)
                                total_pixels = np.sum(mask[0] != 0)
                                
                                if total_pixels > 0:
                                    # Calculate forest coverage percentage (non-zero forest height pixels)
                                    forest_pixels = zonal_out[0]['count']
                                    forest_coverage = (forest_pixels / total_pixels) * 100
                                    results[f"forest.coverage_{year_str}_percent"] = forest_coverage
                        
                        # Calculate height distribution metrics
                        if forest_file.exists():
                            with rasterio.open(str(forest_file)) as src:
                                # Mask the raster to the catchment
                                catchment = gpd.read_file(self.catchment_path)
                                if catchment.crs != src.crs:
                                    catchment = catchment.to_crs(src.crs)
                                
                                # Get masked data
                                masked_data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=-9999)
                                
                                # Flatten and filter to valid values
                                valid_data = masked_data[masked_data != -9999]
                                
                                if len(valid_data) > 10:  # Only calculate if we have enough data points
                                    # Calculate additional statistics
                                    results[f"forest.height_{year_str}_skew"] = float(skew(valid_data))
                                    results[f"forest.height_{year_str}_kurtosis"] = float(kurtosis(valid_data))
                                    
                                    # Calculate percentiles
                                    percentiles = [5, 10, 25, 50, 75, 90, 95]
                                    for p in percentiles:
                                        results[f"forest.height_{year_str}_p{p}"] = float(np.percentile(valid_data, p))
                                    
                                    # Calculate canopy density metrics
                                    # Assume forest height threshold of 5m for "canopy"
                                    canopy_threshold = 5.0
                                    canopy_pixels = np.sum(valid_data >= canopy_threshold)
                                    results[f"forest.canopy_density_{year_str}"] = float(canopy_pixels / len(valid_data))
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Apply scale and offset to basic statistics
                            for stat in ['mean', 'min', 'max', 'std', 'median']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat] * scale + offset
                                    results[f"{prefix}forest.height_{year_str}_{stat}"] = value
                            
                            # Calculate forest coverage for each HRU
                            if 'count' in zonal_result and zonal_result['count'] is not None:
                                # Try to count total pixels in this HRU
                                with rasterio.open(str(forest_file)) as src:
                                    hru_gdf = catchment.iloc[i:i+1]
                                    if hru_gdf.crs != src.crs:
                                        hru_gdf = hru_gdf.to_crs(src.crs)
                                    
                                    # Get the mask of pixels within the HRU
                                    mask, transform = rasterio.mask.mask(src, hru_gdf.geometry, crop=False, nodata=0)
                                    total_pixels = np.sum(mask[0] != 0)
                                    
                                    if total_pixels > 0:
                                        # Calculate forest coverage percentage (non-zero forest height pixels)
                                        forest_pixels = zonal_result['count']
                                        forest_coverage = (forest_pixels / total_pixels) * 100
                                        results[f"{prefix}forest.coverage_{year_str}_percent"] = forest_coverage
            
            # Cache results
            try:
                self.logger.info(f"Caching forest height results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching forest height results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing forest height data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_irrigation_attributes(self) -> Dict[str, Any]:
        """
        Process irrigation data including irrigated area percentages.
        
        Sources:
        - LGRIP30: Global Landsat-based rainfed and irrigated cropland map at 30m resolution
        
        The LGRIP30 dataset classifies areas into:
        0: Non-cropland
        1: Rainfed cropland
        2: Irrigated cropland
        3: Paddy cropland (flooded irrigation, primarily rice)
        """
        results = {}
        
        # Find and check paths for irrigation data
        lgrip_path = self._get_data_path('ATTRIBUTES_LGRIP_PATH', 'lgrip30')
        
        self.logger.info(f"LGRIP path: {lgrip_path}")
        
        # Look for the LGRIP30 agriculture raster
        lgrip_file = lgrip_path / 'raw' / 'lgrip30_agriculture.tif'
        
        if not lgrip_file.exists():
            self.logger.warning(f"LGRIP30 agriculture file not found: {lgrip_file}")
            return results
        
        self.logger.info(f"Processing irrigation data from: {lgrip_file}")
        
        # Create output directory for clipped data
        output_dir = self.project_dir / 'attributes' / 'irrigation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clip irrigation data to catchment if not already done
        clipped_file = output_dir / f"{self.domain_name}_lgrip30_agriculture.tif"
        
        if not clipped_file.exists():
            self.logger.info(f"Clipping irrigation data to catchment bounding box")
            self._clip_raster_to_bbox(lgrip_file, clipped_file)
        
        # Process the clipped irrigation data
        if clipped_file.exists():
            try:
                # LGRIP30 is a categorical raster with these classes:
                lgrip_classes = {
                    0: 'non_cropland',
                    1: 'rainfed_cropland',
                    2: 'irrigated_cropland',
                    3: 'paddy_cropland'
                }
                
                # Calculate zonal statistics (categorical)
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(clipped_file),
                    categorical=True,
                    all_touched=True
                )
                
                # Process results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0:
                        # Calculate total pixel count (area)
                        total_pixels = sum(count for class_id, count in zonal_out[0].items() 
                                        if class_id is not None)
                        
                        if total_pixels > 0:
                            # Calculate class fractions and add to results
                            for class_id, class_name in lgrip_classes.items():
                                pixel_count = zonal_out[0].get(class_id, 0)
                                fraction = pixel_count / total_pixels
                                results[f"irrigation.{class_name}_fraction"] = fraction
                            
                            # Calculate combined cropland area (all types)
                            cropland_pixels = sum(zonal_out[0].get(i, 0) for i in [1, 2, 3])
                            cropland_fraction = cropland_pixels / total_pixels
                            results["irrigation.cropland_fraction"] = cropland_fraction
                            
                            # Calculate combined irrigated area (irrigated + paddy)
                            irrigated_pixels = sum(zonal_out[0].get(i, 0) for i in [2, 3])
                            irrigated_fraction = irrigated_pixels / total_pixels
                            results["irrigation.irrigated_fraction"] = irrigated_fraction
                            
                            # Calculate irrigation intensity (irrigated area / total cropland)
                            if cropland_pixels > 0:
                                irrigation_intensity = irrigated_pixels / cropland_pixels
                                results["irrigation.irrigation_intensity"] = irrigation_intensity
                            
                            # Calculate dominant agricultural type
                            ag_classes = {
                                1: 'rainfed_cropland',
                                2: 'irrigated_cropland',
                                3: 'paddy_cropland'
                            }
                            
                            # Find the dominant agricultural class
                            dominant_class = max(
                                ag_classes.keys(),
                                key=lambda k: zonal_out[0].get(k, 0),
                                default=None
                            )
                            
                            if dominant_class and zonal_out[0].get(dominant_class, 0) > 0:
                                results["irrigation.dominant_agriculture"] = ag_classes[dominant_class]
                                dom_fraction = zonal_out[0].get(dominant_class, 0) / total_pixels
                                results["irrigation.dominant_agriculture_fraction"] = dom_fraction
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    # Process each HRU
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Calculate total pixel count (area) for this HRU
                            total_pixels = sum(count for class_id, count in zonal_result.items() 
                                            if class_id is not None)
                            
                            if total_pixels > 0:
                                # Calculate class fractions and add to results
                                for class_id, class_name in lgrip_classes.items():
                                    pixel_count = zonal_result.get(class_id, 0)
                                    fraction = pixel_count / total_pixels
                                    results[f"{prefix}irrigation.{class_name}_fraction"] = fraction
                                
                                # Calculate combined cropland area (all types)
                                cropland_pixels = sum(zonal_result.get(i, 0) for i in [1, 2, 3])
                                cropland_fraction = cropland_pixels / total_pixels
                                results[f"{prefix}irrigation.cropland_fraction"] = cropland_fraction
                                
                                # Calculate combined irrigated area (irrigated + paddy)
                                irrigated_pixels = sum(zonal_result.get(i, 0) for i in [2, 3])
                                irrigated_fraction = irrigated_pixels / total_pixels
                                results[f"{prefix}irrigation.irrigated_fraction"] = irrigated_fraction
                                
                                # Calculate irrigation intensity
                                if cropland_pixels > 0:
                                    irrigation_intensity = irrigated_pixels / cropland_pixels
                                    results[f"{prefix}irrigation.irrigation_intensity"] = irrigation_intensity
                                
                                # Calculate dominant agricultural type
                                ag_classes = {
                                    1: 'rainfed_cropland',
                                    2: 'irrigated_cropland',
                                    3: 'paddy_cropland'
                                }
                                
                                # Find the dominant agricultural class
                                dominant_class = max(
                                    ag_classes.keys(),
                                    key=lambda k: zonal_result.get(k, 0),
                                    default=None
                                )
                                
                                if dominant_class and zonal_result.get(dominant_class, 0) > 0:
                                    results[f"{prefix}irrigation.dominant_agriculture"] = ag_classes[dominant_class]
                                    dom_fraction = zonal_result.get(dominant_class, 0) / total_pixels
                                    results[f"{prefix}irrigation.dominant_agriculture_fraction"] = dom_fraction
                
                # If we have any irrigation, add a simple binary flag
                if is_lumped:
                    has_irrigation = results.get("irrigation.irrigated_fraction", 0) > 0.01  # >1% irrigated
                    results["irrigation.has_significant_irrigation"] = has_irrigation
                else:
                    for i in range(len(catchment)):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        irr_key = f"{prefix}irrigation.irrigated_fraction"
                        
                        if irr_key in results:
                            has_irrigation = results[irr_key] > 0.01  # >1% irrigated
                            results[f"{prefix}irrigation.has_significant_irrigation"] = has_irrigation
                
            except Exception as e:
                self.logger.error(f"Error processing irrigation data: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.warning(f"Clipped irrigation file not found or could not be created: {clipped_file}")
        
        return results


    def _get_data_path(self, config_key: str, default_subfolder: str) -> Path:
        """
        Get the path to a data source, checking multiple possible locations.
        
        Args:
            config_key: Configuration key to check for a specified path
            default_subfolder: Default subfolder to use within attribute data directories
            
        Returns:
            Path to the data source
        """
        # Check if path is explicitly specified in config
        path = self.config.get(config_key)
        
        if path and path != 'default':
            return Path(path)
        
        # Check the attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_path = Path(attr_data_dir) / default_subfolder
            if attr_path.exists():
                return attr_path
        
        # Check the specific North America data directory
        na_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial") / default_subfolder
        if na_path.exists():
            return na_path
        
        # Fall back to project data directory
        proj_path = self.data_dir / 'geospatial-data' / default_subfolder
        return proj_path


    def _process_forest_height_attributes(self) -> Dict[str, Any]:
        """
        Acquire, clip, and process forest height data for the catchment.
        """
        results = {}
        
        # Create forest height directory
        forest_dir = self.project_dir / 'attributes' / 'forest_height'
        forest_dir.mkdir(parents=True, exist_ok=True)
        
        # Find source forest height data
        forest_source = self._find_forest_height_source()
        
        if not forest_source:
            self.logger.warning("No forest height data found")
            return results
        
        # Clip forest height data to bounding box if not already done
        clipped_forest = forest_dir / f"{self.domain_name}_forest_height.tif"
        
        if not clipped_forest.exists():
            self.logger.info(f"Clipping forest height data to catchment bounding box")
            self._clip_raster_to_bbox(forest_source, clipped_forest)
        
        # Process the clipped forest height data
        if clipped_forest.exists():
            self.logger.info(f"Processing forest height statistics from {clipped_forest}")
            
            # Calculate zonal statistics
            stats = ['min', 'mean', 'max', 'std', 'median', 'count']
            
            try:
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(clipped_forest),
                    stats=stats,
                    all_touched=True,
                    nodata=-9999  # Common no-data value for forest height datasets
                )
                
                # Process results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0:
                        for stat, value in zonal_out[0].items():
                            if value is not None:
                                results[f"forest.height_{stat}"] = value
                        
                        # Calculate forest coverage
                        if 'count' in zonal_out[0] and zonal_out[0]['count'] is not None:
                            with rasterio.open(str(clipped_forest)) as src:
                                # Calculate total number of pixels in the catchment
                                total_pixels = self._count_pixels_in_catchment(src)
                                
                                if total_pixels > 0:
                                    # Count non-zero forest height pixels
                                    forest_pixels = zonal_out[0]['count']
                                    # Calculate forest coverage percentage
                                    forest_coverage = (forest_pixels / total_pixels) * 100
                                    results["forest.coverage_percent"] = forest_coverage
                        
                        # Calculate forest height distribution metrics
                        self._add_forest_distribution_metrics(clipped_forest, results)
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    # Process each HRU
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            for stat, value in zonal_result.items():
                                if value is not None:
                                    results[f"{prefix}forest.height_{stat}"] = value
                            
                            # Calculate forest coverage for this HRU
                            if 'count' in zonal_result and zonal_result['count'] is not None:
                                with rasterio.open(str(clipped_forest)) as src:
                                    # Calculate total number of pixels in this HRU
                                    total_pixels = self._count_pixels_in_hru(src, catchment.iloc[i:i+1])
                                    
                                    if total_pixels > 0:
                                        # Count non-zero forest height pixels
                                        forest_pixels = zonal_result['count']
                                        # Calculate forest coverage percentage
                                        forest_coverage = (forest_pixels / total_pixels) * 100
                                        results[f"{prefix}forest.coverage_percent"] = forest_coverage
                            
                            # Calculate forest height distribution metrics for this HRU
                            self._add_forest_distribution_metrics(
                                clipped_forest,
                                results,
                                hru_geometry=catchment.iloc[i:i+1],
                                prefix=prefix
                            )
            
            except Exception as e:
                self.logger.error(f"Error processing forest height statistics: {str(e)}")
        
        return results

    def _find_forest_height_source(self) -> Optional[Path]:
        """
        Find forest height data in the data directory or global datasets.
        Returns the path to the forest height raster if found, None otherwise.
        """
        # Check config for forest height data path - use the correct config key
        forest_path = self.config.get('ATTRIBUTES_FOREST_HEIGHT_PATH')
        
        if forest_path and forest_path != 'default':
            forest_path = Path(forest_path)
            if forest_path.exists():
                self.logger.info(f"Using forest height data from config: {forest_path}")
                return forest_path
        
        # Use the base attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_data_dir = Path(attr_data_dir)
            
            # Check for forest_height directory within the attribute data directory
            forest_dir = attr_data_dir / 'forest_height' / 'raw'
            if forest_dir.exists():
                # Prefer the 2020 data as it's more recent
                if (forest_dir / "forest_height_2020.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2020.tif'}")
                    return forest_dir / "forest_height_2020.tif"
                # Fall back to 2000 data if 2020 isn't available
                elif (forest_dir / "forest_height_2000.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2000.tif'}")
                    return forest_dir / "forest_height_2000.tif"
                
                # Look for any TIF file in the directory
                forest_files = list(forest_dir.glob("*.tif"))
                if forest_files:
                    self.logger.info(f"Using forest height data from attribute directory: {forest_files[0]}")
                    return forest_files[0]
        
        # Fall back to project data directory if attribute data dir doesn't have the data
        global_forest_dir = self.data_dir / 'geospatial-data' / 'forest_height'
        if global_forest_dir.exists():
            forest_files = list(global_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from global directory: {forest_files[0]}")
                return forest_files[0]
        
        # Try the hard-coded path as a last resort
        specific_forest_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/forest_height/raw")
        if specific_forest_dir.exists():
            forest_files = list(specific_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from specific directory: {forest_files[0]}")
                return forest_files[0]
        
        self.logger.warning("No forest height data found")
        return None

    def _clip_raster_to_bbox(self, source_raster: Path, output_raster: Path) -> None:
        """
        Clip a raster to the catchment's bounding box.
        """
        try:
            # Parse bounding box coordinates
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
            if len(bbox_coords) == 4:
                bbox = [
                    float(bbox_coords[1]),  # lon_min
                    float(bbox_coords[2]),  # lat_min
                    float(bbox_coords[3]),  # lon_max
                    float(bbox_coords[0])   # lat_max
                ]
            else:
                raise ValueError("Invalid bounding box coordinates format")
            
            # Ensure output directory exists
            output_raster.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a bounding box geometry
            geom = box(*bbox)
            
            # Create a GeoDataFrame with the bounding box
            bbox_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            
            # Clip the raster
            with rasterio.open(str(source_raster)) as src:
                # Check if coordinate reference systems match
                if src.crs != bbox_gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(src.crs)
                
                # Get the geometry in the raster's CRS
                geoms = [geom for geom in bbox_gdf.geometry]
                
                # Clip the raster
                out_image, out_transform = mask(src, geoms, crop=True)
                
                # Copy the metadata
                out_meta = src.meta.copy()
                
                # Update metadata for clipped raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # Write the clipped raster
                with rasterio.open(str(output_raster), "w", **out_meta) as dest:
                    dest.write(out_image)
            
            self.logger.info(f"Clipped raster saved to {output_raster}")
            
        except Exception as e:
            self.logger.error(f"Error clipping raster: {str(e)}")
            raise

    def _count_pixels_in_catchment(self, raster_src) -> int:
        """
        Count the total number of pixels within the catchment boundary.
        """
        # Read the catchment shapefile
        catchment = gpd.read_file(str(self.catchment_path))
        
        # Ensure the shapefile is in the same CRS as the raster
        if catchment.crs != raster_src.crs:
            catchment = catchment.to_crs(raster_src.crs)
        
        # Create a mask of the catchment area
        mask, transform = rasterio.mask.mask(raster_src, catchment.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the catchment)
        return np.sum(mask[0] != 0)

    def _count_pixels_in_hru(self, raster_src, hru_geometry) -> int:
        """
        Count the total number of pixels within a specific HRU boundary.
        """
        # Ensure the HRU geometry is in the same CRS as the raster
        if hru_geometry.crs != raster_src.crs:
            hru_geometry = hru_geometry.to_crs(raster_src.crs)
        
        # Create a mask of the HRU area
        mask, transform = rasterio.mask.mask(raster_src, hru_geometry.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the HRU)
        return np.sum(mask[0] != 0)

    def _add_forest_distribution_metrics(self, forest_raster: Path, results: Dict[str, Any], 
                                        hru_geometry=None, prefix="") -> None:
        """
        Calculate forest height distribution metrics like skewness, kurtosis, and percentiles.
        """
        try:
            with rasterio.open(str(forest_raster)) as src:
                if hru_geometry is not None:
                    # For a specific HRU
                    if hru_geometry.crs != src.crs:
                        hru_geometry = hru_geometry.to_crs(src.crs)
                    
                    # Mask the raster to the HRU
                    masked_data, _ = rasterio.mask.mask(src, hru_geometry.geometry, crop=True, nodata=-9999)
                else:
                    # For the entire catchment
                    catchment = gpd.read_file(str(self.catchment_path))
                    
                    if catchment.crs != src.crs:
                        catchment = catchment.to_crs(src.crs)
                    
                    # Mask the raster to the catchment
                    masked_data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=-9999)
                
                # Flatten the data and exclude no-data values
                valid_data = masked_data[masked_data != -9999]
                
                # Only calculate statistics if we have enough valid data points
                if len(valid_data) > 10:
                    # Calculate additional statistics
                    results[f"{prefix}forest.height_skew"] = float(skew(valid_data))
                    results[f"{prefix}forest.height_kurtosis"] = float(kurtosis(valid_data))
                    
                    # Calculate percentiles
                    percentiles = [5, 10, 25, 50, 75, 90, 95]
                    for p in percentiles:
                        results[f"{prefix}forest.height_p{p}"] = float(np.percentile(valid_data, p))
                    
                    # Calculate canopy density metrics
                    # Assume forest height threshold of 5m for "canopy"
                    canopy_threshold = 5.0
                    canopy_pixels = np.sum(valid_data >= canopy_threshold)
                    if len(valid_data) > 0:
                        results[f"{prefix}forest.canopy_density"] = float(canopy_pixels / len(valid_data))
        
        except Exception as e:
            self.logger.error(f"Error calculating forest distribution metrics: {str(e)}")

    def _process_soil_attributes(self) -> Dict[str, Any]:
        """
        Process soil properties including texture, hydraulic properties, and soil depth.
        Extracts data from SOILGRIDS and Pelletier datasets.
        
        Returns:
            Dict[str, Any]: Dictionary of soil attributes
        """
        results = {}
        
        # Process soil texture from SOILGRIDS
        texture_results = self._process_soilgrids_texture()
        results.update(texture_results)
        
        # Process soil hydraulic properties derived from texture classes
        hydraulic_results = self._derive_hydraulic_properties(texture_results)
        results.update(hydraulic_results)
        
        # Process soil depth attributes from Pelletier dataset
        depth_results = self._process_pelletier_soil_depth()
        results.update(depth_results)
        
        return results

    def _process_soilgrids_texture(self) -> Dict[str, Any]:
        """
        Process soil texture data from SOILGRIDS.
        
        Returns:
            Dict[str, Any]: Dictionary of soil texture attributes
        """
        results = {}
        
        # Define paths to SOILGRIDS data
        soilgrids_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/soilgrids/raw")
        
        # Define soil components and depths to process
        components = ['clay', 'sand', 'silt']
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        stats = ['mean', 'min', 'max', 'std']
        
        # Process each soil component at each depth
        for component in components:
            for depth in depths:
                # Define file path for mean values
                tif_path = soilgrids_dir / component / f"{component}_{depth}_mean.tif"
                
                # Skip if file doesn't exist
                if not tif_path.exists():
                    self.logger.warning(f"Soil component file not found: {tif_path}")
                    continue
                
                self.logger.info(f"Processing soil {component} at depth {depth}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(tif_path), 
                    stats=stats, 
                    all_touched=True
                )
                
                # Get scale and offset for proper unit conversion
                scale, offset = self._read_scale_and_offset(tif_path)
                
                # Update results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    for stat in stats:
                        if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                            value = zonal_out[0][stat]
                            if value is not None:
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"soil.{component}_{depth}_{stat}"] = value
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            for stat in stats:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat]
                                    # Apply scale and offset if they exist
                                    if scale is not None:
                                        value = value * scale
                                    if offset is not None:
                                        value = value + offset
                                    results[f"{prefix}soil.{component}_{depth}_{stat}"] = value
        
        # Calculate USDA soil texture class percentages
        self._calculate_usda_texture_classes(results)
        
        return results

    def _derive_hydraulic_properties(self, texture_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive hydraulic properties from soil texture using pedotransfer functions.
        
        Args:
            texture_results: Dictionary containing soil texture information
            
        Returns:
            Dict[str, Any]: Dictionary of derived hydraulic properties
        """
        results = {}
        
        # Define pedotransfer functions for common hydraulic properties
        # These are based on the relationships in Saxton and Rawls (2006)
        
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
        
        if is_lumped:
            # For lumped catchment
            # Calculate weighted average properties across all depths
            depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
            depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
            total_depth = sum(depth_weights)
            
            # Get weighted average of sand, clay, and silt content across all depths
            avg_sand = 0
            avg_clay = 0
            avg_silt = 0
            
            for depth, weight in zip(depths, depth_weights):
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"
                
                if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                    avg_sand += texture_results[sand_key] * weight / total_depth
                    avg_clay += texture_results[clay_key] * weight / total_depth
                    avg_silt += texture_results[silt_key] * weight / total_depth
            
            # Convert from g/kg to fraction (divide by 10)
            sand_fraction = avg_sand / 1000
            clay_fraction = avg_clay / 1000 
            silt_fraction = avg_silt / 1000
            
            # Calculate hydraulic properties using pedotransfer functions
            
            # Porosity (saturated water content)
            porosity = 0.46 - 0.0026 * avg_clay
            results["soil.porosity"] = porosity
            
            # Field capacity (-33 kPa matric potential)
            field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
            results["soil.field_capacity"] = field_capacity
            
            # Wilting point (-1500 kPa matric potential)
            wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
            results["soil.wilting_point"] = wilting_point
            
            # Available water capacity
            results["soil.available_water_capacity"] = field_capacity - wilting_point
            
            # Saturated hydraulic conductivity (mm/h)
            ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
            results["soil.ksat"] = ksat
            
        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # Calculate weighted average properties across all depths
                depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
                depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
                total_depth = sum(depth_weights)
                
                # Get weighted average of sand, clay, and silt content across all depths
                avg_sand = 0
                avg_clay = 0
                avg_silt = 0
                
                for depth, weight in zip(depths, depth_weights):
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"
                    
                    if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                        avg_sand += texture_results[sand_key] * weight / total_depth
                        avg_clay += texture_results[clay_key] * weight / total_depth
                        avg_silt += texture_results[silt_key] * weight / total_depth
                
                # Convert from g/kg to fraction (divide by 10)
                sand_fraction = avg_sand / 1000
                clay_fraction = avg_clay / 1000
                silt_fraction = avg_silt / 1000
                
                # Calculate hydraulic properties using pedotransfer functions
                
                # Porosity (saturated water content)
                porosity = 0.46 - 0.0026 * avg_clay
                results[f"{prefix}soil.porosity"] = porosity
                
                # Field capacity (-33 kPa matric potential)
                field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
                results[f"{prefix}soil.field_capacity"] = field_capacity
                
                # Wilting point (-1500 kPa matric potential)
                wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
                results[f"{prefix}soil.wilting_point"] = wilting_point
                
                # Available water capacity
                results[f"{prefix}soil.available_water_capacity"] = field_capacity - wilting_point
                
                # Saturated hydraulic conductivity (mm/h)
                ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
                results[f"{prefix}soil.ksat"] = ksat
        
        return results

    def _calculate_usda_texture_classes(self, results: Dict[str, Any]) -> None:
        """
        Calculate USDA soil texture classes based on sand, silt, and clay percentages.
        Updates the results dictionary in place.
        
        Args:
            results: Dictionary to update with texture class information
        """
        # USDA soil texture classes
        usda_classes = {
            'clay': (0, 45, 0, 40, 40, 100),  # Sand%, Silt%, Clay% ranges
            'silty_clay': (0, 20, 40, 60, 40, 60),
            'sandy_clay': (45, 65, 0, 20, 35, 55),
            'clay_loam': (20, 45, 15, 53, 27, 40),
            'silty_clay_loam': (0, 20, 40, 73, 27, 40),
            'sandy_clay_loam': (45, 80, 0, 28, 20, 35),
            'loam': (23, 52, 28, 50, 7, 27),
            'silty_loam': (0, 50, 50, 88, 0, 27),
            'sandy_loam': (50, 80, 0, 50, 0, 20),
            'silt': (0, 20, 80, 100, 0, 12),
            'loamy_sand': (70, 90, 0, 30, 0, 15),
            'sand': (85, 100, 0, 15, 0, 10)
        }
        
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
        
        # Depths to analyze
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        
        if is_lumped:
            # For each depth, determine texture class percentage
            for depth in depths:
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"
                
                if sand_key in results and clay_key in results and silt_key in results:
                    # Get values and convert to percentages
                    sand_val = results[sand_key] / 10  # g/kg to %
                    clay_val = results[clay_key] / 10  # g/kg to %
                    silt_val = results[silt_key] / 10  # g/kg to %
                    
                    # Normalize to ensure they sum to 100%
                    total = sand_val + clay_val + silt_val
                    if total > 0:
                        sand_pct = (sand_val / total) * 100
                        clay_pct = (clay_val / total) * 100
                        silt_pct = (silt_val / total) * 100
                        
                        # Determine soil texture class
                        for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                            if (sand_min <= sand_pct <= sand_max and
                                silt_min <= silt_pct <= silt_max and
                                clay_min <= clay_pct <= clay_max):
                                
                                # Add texture class to results
                                results[f"soil.texture_class_{depth}"] = texture_class
                                break
        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # For each depth, determine texture class percentage
                for depth in depths:
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"
                    
                    if sand_key in results and clay_key in results and silt_key in results:
                        # Get values and convert to percentages
                        sand_val = results[sand_key] / 10  # g/kg to %
                        clay_val = results[clay_key] / 10  # g/kg to %
                        silt_val = results[silt_key] / 10  # g/kg to %
                        
                        # Normalize to ensure they sum to 100%
                        total = sand_val + clay_val + silt_val
                        if total > 0:
                            sand_pct = (sand_val / total) * 100
                            clay_pct = (clay_val / total) * 100
                            silt_pct = (silt_val / total) * 100
                            
                            # Determine soil texture class
                            for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                                if (sand_min <= sand_pct <= sand_max and
                                    silt_min <= silt_pct <= silt_max and
                                    clay_min <= clay_pct <= clay_max):
                                    
                                    # Add texture class to results
                                    results[f"{prefix}soil.texture_class_{depth}"] = texture_class
                                    break

    def _process_pelletier_soil_depth(self) -> Dict[str, Any]:
        """
        Process soil depth attributes from Pelletier dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of soil depth attributes
        """
        results = {}
        
        # Define path to Pelletier data
        pelletier_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/pelletier/raw")
        
        # Define files to process
        pelletier_files = {
            "upland_hill-slope_regolith_thickness.tif": "regolith_thickness",
            "upland_hill-slope_soil_thickness.tif": "soil_thickness",
            "upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif": "sedimentary_thickness",
            "average_soil_and_sedimentary-deposit_thickness.tif": "average_thickness"
        }
        
        stats = ['mean', 'min', 'max', 'std']
        
        for file_name, attribute in pelletier_files.items():
            # Define file path
            tif_path = pelletier_dir / file_name
            
            # Skip if file doesn't exist
            if not tif_path.exists():
                self.logger.warning(f"Pelletier file not found: {tif_path}")
                continue
            
            self.logger.info(f"Processing Pelletier {attribute}")
            
            # Check and set no-data value if needed
            self._check_and_set_nodata_value(tif_path)
            
            # Calculate zonal statistics
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(tif_path), 
                stats=stats, 
                all_touched=True
            )
            
            # Some tifs may have no data because the variable doesn't exist in the area
            zonal_out = self._check_zonal_stats_outcomes(zonal_out, new_val=0)
            
            # Get scale and offset
            scale, offset = self._read_scale_and_offset(tif_path)
            
            # Update results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                for stat in stats:
                    if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                        value = zonal_out[0][stat]
                        if value is not None:
                            # Apply scale and offset if they exist
                            if scale is not None:
                                value = value * scale
                            if offset is not None:
                                value = value + offset
                            results[f"soil.{attribute}_{stat}"] = value
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        for stat in stats:
                            if stat in zonal_result and zonal_result[stat] is not None:
                                value = zonal_result[stat]
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"{prefix}soil.{attribute}_{stat}"] = value
        
        return results

    def _read_scale_and_offset(self, geotiff_path: Path) -> Tuple[Optional[float], Optional[float]]:
        """
        Read scale and offset from a GeoTIFF file.
        
        Args:
            geotiff_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of scale and offset values, potentially None if not set
        """
        try:
            dataset = gdal.Open(str(geotiff_path))
            if dataset is None:
                self.logger.warning(f"Could not open GeoTIFF: {geotiff_path}")
                return None, None
            
            # Get the scale and offset values
            scale = dataset.GetRasterBand(1).GetScale()
            offset = dataset.GetRasterBand(1).GetOffset()
            
            # Close the dataset
            dataset = None
            
            return scale, offset
        except Exception as e:
            self.logger.error(f"Error reading scale and offset: {str(e)}")
            return None, None

    def _check_and_set_nodata_value(self, tif: Path, nodata: int = 255) -> None:
        """
        Check and set no-data value for a GeoTIFF if not already set.
        
        Args:
            tif: Path to the GeoTIFF file
            nodata: No-data value to set
        """
        try:
            # Open the dataset
            ds = gdal.Open(str(tif), gdal.GA_Update)
            if ds is None:
                self.logger.warning(f"Could not open GeoTIFF for no-data setting: {tif}")
                return
            
            # Get the current no-data value
            band = ds.GetRasterBand(1)
            current_nodata = band.GetNoDataValue()
            
            # If no no-data value is set but we need one
            if current_nodata is None:
                # Check if the maximum value in the dataset is the no-data value
                data = band.ReadAsArray()
                if data.max() == nodata:
                    # Set the no-data value
                    band.SetNoDataValue(nodata)
                    self.logger.info(f"Set no-data value to {nodata} for {tif}")
                
            # Close the dataset
            ds = None
        except Exception as e:
            self.logger.error(f"Error checking and setting no-data value: {str(e)}")

    def _check_zonal_stats_outcomes(self, zonal_out: List[Dict], new_val: Union[float, int] = np.nan) -> List[Dict]:
        """
        Check for None values in zonal statistics results and replace with specified value.
        
        Args:
            zonal_out: List of dictionaries with zonal statistics results
            new_val: Value to replace None with
            
        Returns:
            Updated zonal statistics results
        """
        for i in range(len(zonal_out)):
            for key, val in zonal_out[i].items():
                if val is None:
                    zonal_out[i][key] = new_val
                    self.logger.debug(f"Replaced None in {key} with {new_val}")
        
        return zonal_out

    def _process_elevation_attributes(self) -> Dict[str, float]:
        """Process elevation, slope, and aspect attributes."""
        results = {}
        
        try:
            # Find the DEM file
            dem_file = self.find_dem_file()
            
            # Generate slope and aspect
            raster_files = self.generate_slope_and_aspect(dem_file)
            
            # Calculate statistics for each raster
            for attribute_name, raster_file in raster_files.items():
                stats = self.calculate_statistics(raster_file, attribute_name)
                
                # Add statistics with proper prefixes
                for stat_name, value in stats.items():
                    if "." in stat_name:  # If stat_name already has hierarchical structure
                        results[stat_name] = value
                    else:
                        # Extract any HRU prefix if present
                        if stat_name.startswith("HRU_"):
                            hru_part = stat_name.split("_", 2)[0] + "_" + stat_name.split("_", 2)[1] + "_"
                            clean_stat = stat_name.replace(hru_part, "")
                            prefix = hru_part
                        else:
                            clean_stat = stat_name
                            prefix = ""
                        
                        # Remove attribute prefix if present in the stat name
                        clean_stat = clean_stat.replace(f"{attribute_name}_", "")
                        results[f"{prefix}{attribute_name}.{clean_stat}"] = value
        
        except Exception as e:
            self.logger.error(f"Error processing elevation attributes: {str(e)}")
        
        return results

    def _process_landcover_attributes(self) -> Dict[str, Any]:
        """Process land cover attributes including dominant land cover type."""
        results = {}
        
        # Find the land cover raster
        landclass_dir = self.project_dir / 'attributes' / 'landclass'
        if not landclass_dir.exists():
            self.logger.warning(f"Land cover directory not found: {landclass_dir}")
            return results
        
        # Search for land cover raster files
        landcover_files = list(landclass_dir.glob("*.tif"))
        if not landcover_files:
            self.logger.warning(f"No land cover raster files found in {landclass_dir}")
            return results
        
        # Use the first land cover file found
        landcover_file = landcover_files[0]
        self.logger.info(f"Processing land cover raster: {landcover_file}")
        
        try:
            # Get land cover classes and class names if available
            lc_classes = self._get_landcover_classes(landcover_file)
            
            # Calculate zonal statistics with categorical=True to get class counts
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(landcover_file), 
                categorical=True, 
                all_touched=True
            )
            
            # Process results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out:
                    # Calculate class fractions
                    total_pixels = sum(count for count in zonal_out[0].values() if count is not None)
                    
                    if total_pixels > 0:
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None:  # Skip None values
                                # Get class name if available, otherwise use class ID
                                class_name = lc_classes.get(class_id, f"class_{class_id}")
                                fraction = (count / total_pixels) if count is not None else 0
                                results[f"landcover.{class_name}_fraction"] = fraction
                        
                        # Find dominant land cover class
                        dominant_class_id = max(zonal_out[0].items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                        dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                        results["landcover.dominant_class"] = dominant_class_name
                        results["landcover.dominant_class_id"] = dominant_class_id
                        results["landcover.dominant_fraction"] = (zonal_out[0][dominant_class_id] / total_pixels) if zonal_out[0][dominant_class_id] is not None else 0
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Process each HRU
                for i, zonal_result in enumerate(zonal_out):
                    try:
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate class fractions for this HRU
                        total_pixels = sum(count for count in zonal_result.values() if count is not None)
                        
                        if total_pixels > 0:
                            for class_id, count in zonal_result.items():
                                if class_id is not None:  # Skip None values
                                    class_name = lc_classes.get(class_id, f"class_{class_id}")
                                    fraction = (count / total_pixels) if count is not None else 0
                                    results[f"{prefix}landcover.{class_name}_fraction"] = fraction
                            
                            # Find dominant land cover class for this HRU
                            dominant_class_id = max(zonal_result.items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                            dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                            results[f"{prefix}landcover.dominant_class"] = dominant_class_name
                            results[f"{prefix}landcover.dominant_class_id"] = dominant_class_id
                            results[f"{prefix}landcover.dominant_fraction"] = (zonal_result[dominant_class_id] / total_pixels) if zonal_result[dominant_class_id] is not None else 0
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {i}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing land cover attributes: {str(e)}")
        
        return results

    def _get_landcover_classes(self, landcover_file: Path) -> Dict[int, str]:
        """
        Get land cover class definitions.
        Attempts to determine class names from metadata or uses a default mapping.
        """
        # Default MODIS IGBP land cover classes
        modis_classes = {
            1: 'evergreen_needleleaf',
            2: 'evergreen_broadleaf',
            3: 'deciduous_needleleaf',
            4: 'deciduous_broadleaf',
            5: 'mixed_forest',
            6: 'closed_shrubland',
            7: 'open_shrubland',
            8: 'woody_savanna',
            9: 'savanna',
            10: 'grassland',
            11: 'wetland',
            12: 'cropland',
            13: 'urban',
            14: 'crop_natural_mosaic',
            15: 'snow_ice',
            16: 'barren',
            17: 'water'
        }
        
        # Try to read class definitions from raster metadata or config
        try:
            with rasterio.open(str(landcover_file)) as src:
                # Check if there's land cover class metadata
                if 'land_cover_classes' in src.tags():
                    # Parse metadata
                    class_info = src.tags()['land_cover_classes']
                    # Implement parsing logic based on the format
                    return {}  # Return parsed classes
        except Exception as e:
            self.logger.warning(f"Could not read land cover class metadata: {str(e)}")
        
        # Check if land cover type is specified in config
        lc_type = self.config.get('LAND_COVER_TYPE', '').lower()
        
        if 'modis' in lc_type or 'igbp' in lc_type:
            return modis_classes
        elif 'esa' in lc_type or 'cci' in lc_type:
            # ESA CCI land cover classes
            return {
                10: 'cropland_rainfed',
                20: 'cropland_irrigated',
                30: 'cropland_mosaic',
                40: 'forest_broadleaf_evergreen',
                50: 'forest_broadleaf_deciduous',
                60: 'forest_needleleaf_evergreen',
                70: 'forest_needleleaf_deciduous',
                80: 'forest_mixed',
                90: 'shrubland_mosaic',
                100: 'grassland',
                110: 'wetland',
                120: 'shrubland',
                130: 'grassland_sparse',
                140: 'lichens_mosses',
                150: 'sparse_vegetation',
                160: 'forest_flooded_freshwater',
                170: 'forest_flooded_saline',
                180: 'shrubland_flooded',
                190: 'urban',
                200: 'bare_areas',
                210: 'water',
                220: 'snow_ice'
            }
        
        # Default to generic class names
        return {i: f"class_{i}" for i in range(1, 30)}
    

# In utils/optimization_utils/ensemble_performance_analyzer.py

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import logging
from typing import Dict, Tuple, Optional, List

class EnsemblePerformanceAnalyzer:
    """
    Analyzes performance metrics for ensemble simulations and creates visualizations.
    
    This class handles:
    1. Processing ensemble simulation outputs
    2. Comparing simulations with observed streamflow
    3. Calculating performance metrics (KGE, NSE, RMSE, PBIAS)
    4. Creating visualizations of ensemble runs vs observed data
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the ensemble performance analyzer.
        
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
    
    def calculate_ensemble_performance_metrics(self) -> Optional[Path]:
        """
        Calculate performance metrics for ensemble simulations with focus on calibration and validation periods.
        
        Returns:
            Path: Path to performance metrics file or None if calculation failed
        """
        self.logger.info("Calculating performance metrics for ensemble simulations")
        
        # Define paths
        ensemble_dir = self.project_dir / "emulation" / self.experiment_id / "ensemble_runs"
        metrics_dir = self.project_dir / "emulation" / self.experiment_id / "ensemble_analysis"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / "performance_metrics.csv"
        
        # Check if ensemble runs exist
        if not ensemble_dir.exists() or not any(ensemble_dir.glob("run_*")):
            self.logger.error("No ensemble run directories found, cannot calculate metrics")
            return None
        
        # Get observed data
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.error(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Load observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date and streamflow columns
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if date_col is None or flow_col is None:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            
            # Set up calibration and evaluation periods
            calib_mask, eval_mask = self._get_period_masks(obs_df.index)
            
            # Process each ensemble run
            run_dirs = sorted(ensemble_dir.glob("run_*"))
            metrics = {}
            all_flows = pd.DataFrame(index=obs_df.index)
            all_flows['Observed'] = obs_df[flow_col]
            
            for run_dir in run_dirs:
                run_id = run_dir.name
                self.logger.debug(f"Processing {run_id}")
                
                # Find streamflow output file
                results_file = run_dir / "results" / f"{run_id}_streamflow.csv"
                
                if not results_file.exists():
                    self.logger.warning(f"No streamflow results found for {run_id}")
                    continue
                
                # Read and process simulation data
                sim_df = pd.read_csv(results_file)
                time_col = next((col for col in sim_df.columns if 'time' in col.lower() or 'date' in col.lower()), 'time')
                sim_df['DateTime'] = pd.to_datetime(sim_df[time_col])
                sim_df.set_index('DateTime', inplace=True)
                
                sim_flow_col = next((col for col in sim_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q' in col.lower()), 'streamflow')
                
                # Add to all flows dataframe
                all_flows[run_id] = sim_df[sim_flow_col].reindex(obs_df.index, method='nearest')
                
                # Calculate metrics for different periods
                metrics[run_id] = self._calculate_run_metrics(obs_df, sim_df, flow_col, sim_flow_col, calib_mask, eval_mask)
            
            if not metrics:
                self.logger.error("No valid metrics calculated for any ensemble run")
                return None
            
            # Convert metrics to DataFrame and save
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            metrics_df = metrics_df.reset_index()
            metrics_df.rename(columns={'index': 'Run'}, inplace=True)
            
            metrics_df.to_csv(metrics_file, index=False)
            self.logger.info(f"Performance metrics saved to {metrics_file}")
            
            # Create visualizations
            try:
                self._plot_ensemble_comparison(all_flows, metrics_df.set_index('Run'), metrics_dir)
            except Exception as e:
                self.logger.error(f"Error creating ensemble plots: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            return metrics_file
                
        except Exception as e:
            self.logger.error(f"Error calculating ensemble performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _get_period_masks(self, date_index: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
        """
        Get boolean masks for calibration and evaluation periods.
        
        Args:
            date_index: DateTimeIndex for creating masks
            
        Returns:
            Tuple of calibration mask and evaluation mask
        """
        calib_period = self.config.get('CALIBRATION_PERIOD', '')
        eval_period = self.config.get('EVALUATION_PERIOD', '')
        
        if calib_period and ',' in calib_period and eval_period and ',' in eval_period:
            calib_start, calib_end = [s.strip() for s in calib_period.split(',')]
            eval_start, eval_end = [s.strip() for s in eval_period.split(',')]
            
            calib_mask = (date_index >= pd.Timestamp(calib_start)) & (date_index <= pd.Timestamp(calib_end))
            eval_mask = (date_index >= pd.Timestamp(eval_start)) & (date_index <= pd.Timestamp(eval_end))
            
            self.logger.info(f"Using calibration period: {calib_start} to {calib_end}")
            self.logger.info(f"Using evaluation period: {eval_start} to {eval_end}")
        else:
            # Use time-based split (70% calibration, 30% evaluation)
            total_time = date_index[-1] - date_index[0]
            split_point = date_index[0] + pd.Timedelta(seconds=total_time.total_seconds() * 0.7)
            
            calib_mask = date_index <= split_point
            eval_mask = date_index > split_point
            
            self.logger.info(f"Using time-based split for calibration (before {split_point}) and evaluation (after {split_point})")
        
        return calib_mask, eval_mask
    
    def _calculate_run_metrics(self, obs_df: pd.DataFrame, sim_df: pd.DataFrame, 
                             flow_col: str, sim_flow_col: str, 
                             calib_mask: pd.Series, eval_mask: pd.Series) -> Dict:
        """
        Calculate metrics for a single run for different periods.
        
        Args:
            obs_df: Observed data DataFrame
            sim_df: Simulation data DataFrame  
            flow_col: Column name for observed flow
            sim_flow_col: Column name for simulated flow
            calib_mask: Boolean mask for calibration period
            eval_mask: Boolean mask for evaluation period
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Calibration period
        if calib_mask.sum() > 0:
            common_idx = obs_df.index[calib_mask].intersection(sim_df.index)
            if len(common_idx) > 0:
                observed = obs_df.loc[common_idx, flow_col]
                simulated = sim_df.loc[common_idx, sim_flow_col]
                
                calib_metrics = self._calculate_performance_metrics(observed, simulated)
                for metric, value in calib_metrics.items():
                    metrics[f"Calib_{metric}"] = value
        
        # Evaluation period
        if eval_mask.sum() > 0:
            common_idx = obs_df.index[eval_mask].intersection(sim_df.index)
            if len(common_idx) > 0:
                observed = obs_df.loc[common_idx, flow_col]
                simulated = sim_df.loc[common_idx, sim_flow_col]
                
                eval_metrics = self._calculate_performance_metrics(observed, simulated)
                for metric, value in eval_metrics.items():
                    metrics[f"Eval_{metric}"] = value
        
        return metrics
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict:
        """
        Calculate streamflow performance metrics with improved handling of outliers.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        
        Returns:
            Dictionary of performance metrics
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
        
        # Cap extremely high flow values (outliers) - use the 99.5th percentile
        flow_cap = observed.quantile(0.995)
        observed_capped = observed.clip(upper=flow_cap)
        simulated_capped = simulated.clip(upper=flow_cap)
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed_capped.mean()
        nse_numerator = ((observed_capped - simulated_capped) ** 2).sum()
        nse_denominator = ((observed_capped - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed_capped - simulated_capped) ** 2).mean())
        
        # Percent Bias (PBIAS)
        pbias = 100 * (simulated_capped.sum() - observed_capped.sum()) / observed_capped.sum() if observed_capped.sum() != 0 else np.nan
        
        # Kling-Gupta Efficiency (KGE)
        r = observed_capped.corr(simulated_capped)  # Correlation coefficient
        alpha = simulated_capped.std() / observed_capped.std() if observed_capped.std() != 0 else np.nan  # Relative variability
        beta = simulated_capped.mean() / mean_obs if mean_obs != 0 else np.nan  # Bias ratio
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        # Mean Absolute Error (MAE)
        mae = (observed_capped - simulated_capped).abs().mean()
        
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
    
    def _plot_ensemble_comparison(self, all_flows: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path):
        """
        Create visualizations comparing all ensemble runs to observed data.
        
        Args:
            all_flows: DataFrame with observed and simulated flows
            metrics_df: DataFrame with performance metrics
            output_dir: Directory to save visualizations
        """
        try:
            # Handle outliers by capping values
            flow_cap = all_flows['Observed'].quantile(0.995) if 'Observed' in all_flows else 100
            capped_flows = all_flows.clip(upper=flow_cap)
            
            # Ensure we have data to plot
            if capped_flows.empty:
                self.logger.warning("Empty flow data, cannot create ensemble plots")
                return
            
            # Get time information
            date_range = capped_flows.index
            if len(date_range) == 0:
                self.logger.warning("Empty date range, cannot create ensemble plots")
                return
            
            # Get period masks
            calib_mask, eval_mask = self._get_period_masks(date_range)
            
            # Calculate spinup mask (everything before calibration)
            spinup_end = date_range[calib_mask][0] if calib_mask.sum() > 0 else date_range[0]
            spinup_mask = date_range < spinup_end
            
            # Calculate ensemble mean (excluding 'Observed' column)
            sim_cols = [col for col in capped_flows.columns if col != 'Observed' and not col.startswith('Unnamed')]
            if not sim_cols:
                self.logger.warning("No simulation columns found in data")
                return
                
            capped_flows['Ensemble Mean'] = capped_flows[sim_cols].mean(axis=1)
            
            # Create a figure with three subplots
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])
            
            # Create axes for each period
            ax_spinup = fig.add_subplot(gs[0])
            ax_train = fig.add_subplot(gs[1], sharex=ax_spinup)
            ax_test = fig.add_subplot(gs[2], sharex=ax_spinup)
            
            # Plot spinup period
            self._plot_period(ax_spinup, capped_flows, spinup_mask, sim_cols, "Spinup Period")
            
            # Plot calibration period
            self._plot_period(ax_train, capped_flows, calib_mask, sim_cols, "Calibration Period")
            
            # Plot evaluation period
            self._plot_period(ax_test, capped_flows, eval_mask, sim_cols, "Evaluation Period")
            
            # Format x-axis dates
            for ax in [ax_spinup, ax_train, ax_test]:
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add common legend
            fig.legend(['Observed', 'Ensemble Mean', 'Ensemble Runs'], 
                    loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
            
            # Add overall ensemble information
            fig.text(0.02, 0.98, f"Ensemble Size: {len(sim_cols)} runs", fontsize=12)
            
            # Save plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(output_dir / "ensemble_streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create additional plots
            self._create_additional_ensemble_plots(capped_flows, sim_cols, calib_mask, eval_mask, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _plot_period(self, ax: plt.Axes, flows: pd.DataFrame, mask: pd.Series, 
                    sim_cols: List[str], title: str):
        """
        Plot a specific time period with ensemble results.
        
        Args:
            ax: Matplotlib axes to plot on
            flows: DataFrame with flow data
            mask: Boolean mask for the period
            sim_cols: List of simulation column names
            title: Title for the plot
        """
        if mask.sum() > 0:
            period_flows = flows[mask]
            if 'Observed' in period_flows:
                ax.plot(period_flows.index, period_flows['Observed'], 'k-', linewidth=2)
            if 'Ensemble Mean' in period_flows:
                ax.plot(period_flows.index, period_flows['Ensemble Mean'], 'r-', linewidth=1.5)
                
                # Plot individual simulations (limit to first 20 for clarity)
                for col in sim_cols[:min(20, len(sim_cols))]:
                    ax.plot(period_flows.index, period_flows[col], 'b-', alpha=0.05, linewidth=0.5)
                
                # Add metrics if this is calibration or evaluation period
                if 'Calib' in title or 'Eval' in title:
                    try:
                        metrics = self._calculate_period_metrics(period_flows['Observed'], period_flows['Ensemble Mean'])
                        metrics_text = (
                            f"{title.split()[0]} Metrics:\n"
                            f"KGE: {metrics['KGE']:.3f}\n"
                            f"NSE: {metrics['NSE']:.3f}\n"
                            f"RMSE: {metrics['RMSE']:.3f}"
                        )
                        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception as e:
                        self.logger.warning(f"Could not calculate metrics for {title}: {str(e)}")
            
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Flow (m³/s)')
            ax.grid(True, alpha=0.3)
    
    def _calculate_period_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict:
        """
        Calculate metrics for a specific time period with improved robustness.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            Dictionary of performance metrics
        """
        # Clean and align data
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) < 2:
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan}
        
        # Cap values at 99.5 percentile to reduce impact of outliers
        cap_val = observed.quantile(0.995)
        observed = observed.clip(upper=cap_val)
        simulated = simulated.clip(upper=cap_val)
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed.mean()
        nse_numerator = ((observed - simulated) ** 2).sum()
        nse_denominator = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # Kling-Gupta Efficiency (KGE)
        try:
            r = observed.corr(simulated)  # Correlation coefficient
            alpha = simulated.std() / observed.std() if observed.std() > 0 else np.nan  # Relative variability
            beta = simulated.mean() / mean_obs if mean_obs > 0 else np.nan  # Bias ratio
            
            if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
                kge = np.nan
            else:
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        except:
            kge = np.nan
        
        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse}
    
    def _create_additional_ensemble_plots(self, capped_flows: pd.DataFrame, sim_cols: List[str], 
                                       train_mask: pd.Series, test_mask: pd.Series, output_dir: Path):
        """
        Create additional ensemble analysis plots (scatter, FDC, etc.) with improved error handling.
        
        Args:
            capped_flows: DataFrame with capped flow values
            sim_cols: List of simulation column names
            train_mask: Boolean mask for training period
            test_mask: Boolean mask for testing period
            output_dir: Directory to save plots
        """
        try:
            # Create scatter plot comparing observed vs ensemble mean for training and testing
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Training period scatter
            self._create_scatter_plot(ax1, capped_flows, train_mask, "Training Period", "blue")
            
            # Testing period scatter
            self._create_scatter_plot(ax2, capped_flows, test_mask, "Testing Period", "red")
            
            # Save the figure
            for ax in [ax1, ax2]:
                try:
                    ax.set_aspect('equal')
                except:
                    self.logger.warning("Could not set equal aspect ratio for scatter plot")
            
            plt.tight_layout()
            plt.savefig(output_dir / "ensemble_scatter_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating additional ensemble plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _create_scatter_plot(self, ax: plt.Axes, flows: pd.DataFrame, mask: pd.Series, 
                           title: str, color: str):
        """
        Create a scatter plot for a specific period.
        
        Args:
            ax: Matplotlib axes to plot on
            flows: DataFrame with flow data
            mask: Boolean mask for the period
            title: Title for the plot
            color: Color for the scatter points
        """
        period_flows = flows[mask].copy() if isinstance(mask, pd.Series) else pd.DataFrame()
        
        if not period_flows.empty and 'Observed' in period_flows and 'Ensemble Mean' in period_flows:
            # Filter out NaN values
            valid_mask = ~(period_flows['Observed'].isna() | period_flows['Ensemble Mean'].isna())
            if valid_mask.sum() > 0:
                valid_flows = period_flows[valid_mask]
                
                # Convert to NumPy arrays for regression
                x_vals = valid_flows['Observed'].values
                y_vals = valid_flows['Ensemble Mean'].values
                
                # Plot scatter points
                ax.scatter(x_vals, y_vals, alpha=0.7, s=30, c=color, label=title.split()[0])
                
                # Add 1:1 line
                if len(x_vals) > 0:
                    max_val = max(valid_flows['Observed'].max(), valid_flows['Ensemble Mean'].max())
                    min_val = min(valid_flows['Observed'].min(), valid_flows['Ensemble Mean'].min())
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
                    
                    # Add regression line only if we have enough points
                    if len(x_vals) >= 2:
                        try:
                            slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                            
                            if not np.isnan(slope) and not np.isnan(intercept):
                                ax.plot([min_val, max_val], 
                                    [slope * min_val + intercept, slope * max_val + intercept], 
                                    'r-', linewidth=1, label=f'Regression (r={r_value:.2f})')
                        except Exception as e:
                            self.logger.warning(f"Error calculating regression for {title}: {str(e)}")
                
                ax.set_xlabel('Observed Flow (m³/s)')
                ax.set_ylabel('Ensemble Mean Flow (m³/s)')
                ax.set_title(f'{title} Scatter Plot')
                ax.grid(True, alpha=0.3)
                ax.legend()