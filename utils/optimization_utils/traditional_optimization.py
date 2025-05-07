"""
Traditional Optimization utility for CONFLUENCE.

This module provides optimization algorithms for hydrological model calibration without
using emulation approaches. It focuses on direct model calibration using techniques like
Dynamically Dimensioned Search (DDS).
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

class TraditionalOptimizer:
    """
    Traditional optimization class for CONFLUENCE.
    
    This class implements direct optimization algorithms for hydrological models
    without using emulation-based approaches. The primary focus is on the
    Dynamically Dimensioned Search (DDS) algorithm, which is particularly
    effective for watershed model calibration.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the Traditional Optimizer.
        
        Args:
            config: Configuration dictionary from CONFLUENCE
            logger: Logger instance from CONFLUENCE
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Create optimization output directory
        self.optimization_output_dir = self.project_dir / "optimisation" / f"{self.experiment_id}_traditional"
        self.optimization_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get parameters to optimize from config
        local_params_str = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params_str = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        self.local_params = [p.strip() for p in local_params_str.split(',') if p.strip()] if local_params_str else []
        self.basin_params = [p.strip() for p in basin_params_str.split(',') if p.strip()] if basin_params_str else []
        
        self.params_to_optimize = self.local_params + self.basin_params
        self.param_bounds = {}
        
        # Optimization settings
        self.max_iterations = self.config.get('NUMBER_OF_ITERATIONS', 100)
        self.target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
        self.random_seed = self.config.get('EMULATION_SEED', 42)
        self.r_value = self.config.get('DDS_R', 0.2)  # Neighborhood perturbation size parameter
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Path to observed streamflow data
        self.observed_data_path = self.config.get('OBSERVATIONS_PATH')
        if self.observed_data_path == 'default':
            self.observed_data_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            self.observed_data_path = Path(self.observed_data_path)
            
        # Paths for model runs
        self.summa_settings_dir = self.project_dir / "settings" / "SUMMA"
        self.summa_install_dir = self.config.get('SUMMA_INSTALL_PATH')
        if self.summa_install_dir == 'default':
            self.summa_install_dir = self.data_dir / 'installs/summa/bin/'
        self.summa_exec = self.summa_install_dir / self.config.get('SUMMA_EXE', "summa.exe")
        
        # Load parameter bounds
        self._load_parameter_bounds()
        
        self.logger.info(f"Traditional Optimizer initialized with {len(self.params_to_optimize)} parameters")
        self.logger.info(f"Optimization target: {self.target_metric}, Max iterations: {self.max_iterations}")
        
    def _load_parameter_bounds(self):
        """
        Load parameter bounds from SUMMA parameter files.
        
        This method reads the local and basin parameter information files to extract
        the minimum and maximum bounds for each parameter to be optimized.
        """
        # Load local parameter bounds
        local_param_file = self.summa_settings_dir / 'localParamInfo.txt'
        self._parse_param_file(local_param_file, self.local_params)
        
        # Load basin parameter bounds
        basin_param_file = self.summa_settings_dir / 'basinParamInfo.txt'
        self._parse_param_file(basin_param_file, self.basin_params)
        
        # Make sure all parameters to optimize have bounds
        missing_params = [p for p in self.params_to_optimize if p not in self.param_bounds]
        if missing_params:
            self.logger.warning(f"Could not find bounds for parameters: {missing_params}")
            raise ValueError(f"Missing parameter bounds for: {missing_params}")
            
        # Log the parameter bounds
        self.logger.info("Parameter bounds loaded:")
        for param, bounds in self.param_bounds.items():
            self.logger.info(f"  {param}: {bounds}")
            
    def _parse_param_file(self, file_path: Path, param_list: List[str]):
        """
        Parse parameter bounds from SUMMA parameter files.
        
        Args:
            file_path: Path to parameter file
            param_list: List of parameters to extract bounds for
        """
        if not file_path.exists():
            self.logger.warning(f"Parameter file not found: {file_path}")
            return
            
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!'):
                        continue
                        
                    parts = line.split('|')
                    if len(parts) >= 3:
                        param_name = parts[0].strip()
                        if param_name in param_list:
                            try:
                                max_val = float(parts[1].strip())
                                min_val = float(parts[2].strip())
                                # Note the order: bounds are stored as [min, max]
                                self.param_bounds[param_name] = (min_val, max_val)
                                self.logger.debug(f"Loaded bounds for {param_name}: [{min_val}, {max_val}]")
                            except ValueError:
                                self.logger.warning(f"Invalid bounds for parameter {param_name}: {parts[1:3]}")
        except Exception as e:
            self.logger.error(f"Error updating mizuRoute control file: {str(e)}")
            return False
            
    def _extract_streamflow(self, eval_dir: Path) -> pd.DataFrame:
        """
        Extract streamflow from model output files.
        
        Args:
            eval_dir: Directory containing model outputs
            
        Returns:
            pd.DataFrame: DataFrame with datetime index and streamflow column
        """
        # Check if domain is lumped
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        # Get catchment area for conversion
        catchment_area = self._get_catchment_area()
        if catchment_area is None or catchment_area <= 0:
            self.logger.warning(f"Invalid catchment area: {catchment_area}, using 1.0 km² as default")
            catchment_area = 1.0 * 1e6  # Default 1 km² in m²
            
        if is_lumped:
            # For lumped domains, extract directly from SUMMA output
            summa_output_dir = eval_dir / "simulations" / self.experiment_id / "SUMMA"
            
            # Look for output files
            timestep_files = list(summa_output_dir.glob(f"{self.experiment_id}_optimization*.nc"))
            
            if not timestep_files:
                self.logger.warning(f"No SUMMA output files found in {summa_output_dir}")
                return pd.DataFrame()
                
            # Use the first timestep file found
            timestep_file = timestep_files[0]
            
            try:
                # Open the NetCDF file
                with xr.open_dataset(timestep_file) as ds:
                    # Check for various runoff variable names
                    runoff_var = None
                    for var_name in ['averageRoutedRunoff', 'outflow', 'basRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            runoff_var = var_name
                            break
                            
                    if runoff_var is None:
                        self.logger.warning(f"No suitable runoff variable found in {timestep_file}")
                        return pd.DataFrame()
                        
                    # Extract the runoff variable
                    runoff_data = ds[runoff_var]
                    
                    # Handle different dimensions
                    if 'gru' in runoff_data.dims:
                        # If multiple GRUs, sum them up
                        if ds.dims['gru'] > 1:
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
                    
                    # Create dataframe
                    result_df = pd.DataFrame({'streamflow': streamflow})
                    
                    return result_df
                    
            except Exception as e:
                self.logger.error(f"Error extracting streamflow from SUMMA output: {str(e)}")
                return pd.DataFrame()
        else:
            # For distributed domains, extract from MizuRoute output
            mizuroute_output_dir = eval_dir / "simulations" / self.experiment_id / "mizuRoute"
            
            # Look for output files
            output_files = list(mizuroute_output_dir.glob("*.nc"))
            
            if not output_files:
                self.logger.warning(f"No MizuRoute output files found in {mizuroute_output_dir}")
                return pd.DataFrame()
                
            # Target reach ID
            sim_reach_id = self.config.get('SIM_REACH_ID')
            
            # Process each output file
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
                                        
                                        # Create dataframe
                                        if isinstance(streamflow, pd.Series):
                                            result_df = pd.DataFrame({'streamflow': streamflow})
                                            return result_df
                                        else:
                                            self.logger.warning(f"Unexpected format for streamflow data: {type(streamflow)}")
                                            return pd.DataFrame()
                            else:
                                self.logger.warning(f"Reach ID {sim_reach_id} not found in {output_file.name}")
                        else:
                            self.logger.warning(f"No reachID variable found in {output_file.name}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {output_file.name}: {str(e)}")
                    
            # If we get here, no streamflow data was found
            return pd.DataFrame()
            
    def _load_observed_flow(self) -> pd.DataFrame:
        """
        Load observed streamflow data.
        
        Returns:
            pd.DataFrame: DataFrame with datetime index and streamflow column
        """
        if not self.observed_data_path.exists():
            self.logger.error(f"Observed streamflow file not found: {self.observed_data_path}")
            return pd.DataFrame()
            
        try:
            # Read observed data
            obs_df = pd.read_csv(self.observed_data_path)
            
            # Identify date and flow columns in observed data
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            
            if date_col is None or flow_col is None:
                self.logger.error(f"Could not identify date or flow columns in observed data: {obs_df.columns.tolist()}")
                return pd.DataFrame()
                
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df = obs_df.set_index('DateTime')
            
            # Extract the flow series
            obs_flow = pd.DataFrame({'streamflow': obs_df[flow_col]})
            
            return obs_flow
            
        except Exception as e:
            self.logger.error(f"Error loading observed flow data: {str(e)}")
            return pd.DataFrame()
            
    def _calculate_objective(self, sim_flow: pd.DataFrame, obs_flow: pd.DataFrame) -> float:
        """
        Calculate objective function value based on simulated and observed streamflow.
        
        Args:
            sim_flow: DataFrame with simulated streamflow
            obs_flow: DataFrame with observed streamflow
            
        Returns:
            float: Objective function value
        """
        # Get common time period
        common_idx = obs_flow.index.intersection(sim_flow.index)
        
        if len(common_idx) == 0:
            self.logger.warning("No common time steps between observed and simulated flows")
            return -1.0  # Poor value for KGE
            
        # Extract aligned data
        obs_aligned = obs_flow.loc[common_idx, 'streamflow']
        sim_aligned = sim_flow.loc[common_idx, 'streamflow']
        
        # Calculate metric based on target metric
        if self.target_metric == 'KGE':
            return self._calculate_kge(obs_aligned, sim_aligned)
        elif self.target_metric == 'NSE':
            return self._calculate_nse(obs_aligned, sim_aligned)
        elif self.target_metric == 'RMSE':
            # For RMSE, lower is better, but we want to maximize, so we use negative RMSE
            return -self._calculate_rmse(obs_aligned, sim_aligned)
        elif self.target_metric == 'PBIAS':
            # For PBIAS, we want to minimize the absolute value
            return -abs(self._calculate_pbias(obs_aligned, sim_aligned))
        else:
            self.logger.warning(f"Unknown metric: {self.target_metric}, using KGE instead")
            return self._calculate_kge(obs_aligned, sim_aligned)
            
    def _calculate_kge(self, observed: pd.Series, simulated: pd.Series) -> float:
        """
        Calculate Kling-Gupta Efficiency.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            float: KGE value
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaNs
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return -1.0  # Poor value for KGE
            
        # Calculate components
        mean_obs = observed.mean()
        mean_sim = simulated.mean()
        
        # Correlation coefficient
        r = observed.corr(simulated)
        
        # Relative variability
        std_obs = observed.std()
        std_sim = simulated.std()
        if std_obs == 0:
            alpha = 0  # Worst case for alpha component
        else:
            alpha = std_sim / std_obs
            
        # Bias ratio
        if mean_obs == 0:
            beta = 0  # Worst case for beta component
        else:
            beta = mean_sim / mean_obs
            
        # Calculate KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        return kge
        
    def _calculate_nse(self, observed: pd.Series, simulated: pd.Series) -> float:
        """
        Calculate Nash-Sutcliffe Efficiency.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            float: NSE value
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaNs
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return -1.0  # Poor value for NSE
            
        # Calculate NSE
        mean_obs = observed.mean()
        numerator = ((observed - simulated) ** 2).sum()
        denominator = ((observed - mean_obs) ** 2).sum()
        
        if denominator == 0:
            return -1.0  # Poor value for NSE
            
        nse = 1 - (numerator / denominator)
        
        return nse
        
    def _calculate_rmse(self, observed: pd.Series, simulated: pd.Series) -> float:
        """
        Calculate Root Mean Square Error.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            float: RMSE value
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaNs
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            return np.inf  # Worst case for RMSE
            
        # Calculate RMSE
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        return rmse
        
    def _calculate_pbias(self, observed: pd.Series, simulated: pd.Series) -> float:
        """
        Calculate Percent Bias.
        
        Args:
            observed: Series of observed values
            simulated: Series of simulated values
            
        Returns:
            float: PBIAS value
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaNs
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0 or observed.sum() == 0:
            return np.inf  # Worst case for PBIAS
            
        # Calculate PBIAS
        pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum()
        
        return pbias
        
    def _get_catchment_area(self) -> float:
        """
        Get catchment area from basin shapefile.
        
        Returns:
            float: Catchment area in square meters
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
                    return 1.0 * 1e6  # Default 1 km² in m²
            
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
                return 1.0 * 1e6  # Default 1 km² in m²
            
            if total_area > 1e12:  # > 1 million km²
                self.logger.warning(f"Calculated area seems very large: {total_area} m² ({total_area/1e6:.2f} km²). Check units.")
            
            return total_area
            
        except Exception as e:
            self.logger.error(f"Error calculating catchment area: {str(e)}")
            return 1.0 * 1e6  # Default 1 km² in m²
            
    def _plot_optimization_progress(self, results_df: pd.DataFrame, output_dir: Path):
        """
        Create visualization of optimization progress.
        
        Args:
            results_df: DataFrame with optimization results
            output_dir: Directory to save plots
        """
        try:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot objective function values
            axes[0].plot(results_df['Iteration'], results_df['Objective'], 'b.-')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel(f'Objective Function ({self.target_metric})')
            axes[0].set_title('Optimization Progress')
            axes[0].grid(True, alpha=0.3)
            
            # Calculate best objective value at each iteration (cumulative maximum for maximization)
            best_vals = results_df['Objective'].cummax()
            axes[0].plot(results_df['Iteration'], best_vals, 'r-', alpha=0.7, label='Best Value')
            axes[0].legend()
            
            # Plot parameter trajectories
            # Select a subset of parameters if there are many
            param_cols = [col for col in results_df.columns if col not in ['Iteration', 'Objective']]
            max_params_to_plot = 10
            if len(param_cols) > max_params_to_plot:
                # Select the parameters with the highest range of values (most variation)
                param_ranges = {col: results_df[col].max() - results_df[col].min() for col in param_cols}
                sorted_params = sorted(param_ranges.items(), key=lambda x: x[1], reverse=True)
                param_cols = [p[0] for p in sorted_params[:max_params_to_plot]]
                
            # Normalize parameter values for better visualization
            normalized_df = results_df.copy()
            for col in param_cols:
                min_val, max_val = self.param_bounds[col]
                normalized_df[col] = (results_df[col] - min_val) / (max_val - min_val)
                
            # Plot normalized parameter values
            for col in param_cols:
                axes[1].plot(normalized_df['Iteration'], normalized_df[col], '.-', label=col)
                
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Normalized Parameter Value')
            axes[1].set_title('Parameter Trajectories')
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
            
            # Add legend outside the plot
            axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a waterfall plot showing improvement by iteration
            improvements = results_df['Objective'].diff()
            improvements[0] = results_df['Objective'][0]  # First iteration is absolute
            
            # Create figure
            plt.figure(figsize=(12, 6))
            plt.bar(results_df['Iteration'], improvements, color=np.where(improvements >= 0, 'g', 'r'))
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.xlabel('Iteration')
            plt.ylabel(f'Improvement in {self.target_metric}')
            plt.title('Objective Function Improvement by Iteration')
            plt.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "optimization_improvements.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating optimization plots: {str(e)}")
            
    def _run_final_simulation(self, best_params: Dict[str, float], output_dir: Path):
        """
        Run a final simulation with the best parameters and save results.
        
        Args:
            best_params: Dictionary of best parameter values
            output_dir: Directory to save results
        """
        try:
            # Create final simulation directory
            final_sim_dir = output_dir / "final_simulation"
            final_sim_dir.mkdir(parents=True, exist_ok=True)
            
            # Create trial parameter file
            self._create_trial_params(best_params, final_sim_dir)
            
            # Create modified filemanager
            filemanager_path = self._create_filemanager(final_sim_dir)
            
            # Run SUMMA
            summa_success = self._run_summa(filemanager_path, final_sim_dir)
            if not summa_success:
                self.logger.warning("Final SUMMA run failed")
                return
                
            # Run mizuRoute if needed
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            if not is_lumped:
                mizuroute_success = self._run_mizuroute(final_sim_dir)
                if not mizuroute_success:
                    self.logger.warning("Final mizuRoute run failed")
                    
            # Extract streamflow from model output
            sim_flow = self._extract_streamflow(final_sim_dir)
            if sim_flow is None or sim_flow.empty:
                self.logger.warning("Failed to extract streamflow from final simulation")
                return
                
            # Save streamflow to CSV
            sim_flow.to_csv(output_dir / "final_simulated_streamflow.csv")
            
            # Load observed data
            obs_flow = self._load_observed_flow()
            if obs_flow is None or obs_flow.empty:
                self.logger.warning("Failed to load observed flow data for final comparison")
                return
                
            # Calculate performance metrics
            metrics = self._calculate_all_metrics(sim_flow, obs_flow)
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(output_dir / "final_performance_metrics.csv", index=False)
            
            # Create comparison plot
            self._plot_final_comparison(sim_flow, obs_flow, metrics, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error in final simulation: {str(e)}")
            
    def _calculate_all_metrics(self, sim_flow: pd.DataFrame, obs_flow: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            sim_flow: DataFrame with simulated streamflow
            obs_flow: DataFrame with observed streamflow
            
        Returns:
            Dict[str, float]: Dictionary with metrics
        """
        # Get common time period
        common_idx = obs_flow.index.intersection(sim_flow.index)
        
        if len(common_idx) == 0:
            return {'KGE': -1.0, 'NSE': -1.0, 'RMSE': np.inf, 'PBIAS': np.inf}
            
        # Extract aligned data
        obs_aligned = obs_flow.loc[common_idx, 'streamflow']
        sim_aligned = sim_flow.loc[common_idx, 'streamflow']
        
        # Calculate all metrics
        kge = self._calculate_kge(obs_aligned, sim_aligned)
        nse = self._calculate_nse(obs_aligned, sim_aligned)
        rmse = self._calculate_rmse(obs_aligned, sim_aligned)
        pbias = self._calculate_pbias(obs_aligned, sim_aligned)
        
        return {'KGE': kge, 'NSE': nse, 'RMSE': rmse, 'PBIAS': pbias}
        
    def _plot_final_comparison(self, sim_flow: pd.DataFrame, obs_flow: pd.DataFrame, 
                               metrics: Dict[str, float], output_dir: Path):
        """
        Create comparison plot between simulated and observed streamflow.
        
        Args:
            sim_flow: DataFrame with simulated streamflow
            obs_flow: DataFrame with observed streamflow
            metrics: Dictionary with performance metrics
            output_dir: Directory to save plots
        """
        try:
            # Get common time period
            common_idx = obs_flow.index.intersection(sim_flow.index)
            
            if len(common_idx) == 0:
                self.logger.warning("No common time steps for comparison plot")
                return
                
            # Extract aligned data
            obs_aligned = obs_flow.loc[common_idx, 'streamflow']
            sim_aligned = sim_flow.loc[common_idx, 'streamflow']
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot timeseries
            axes[0].plot(obs_aligned.index, obs_aligned, 'b-', label='Observed')
            axes[0].plot(sim_aligned.index, sim_aligned, 'r-', label='Simulated (DDS)')
            
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Streamflow (m³/s)')
            axes[0].set_title('Streamflow Comparison - Traditional Optimization (DDS)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Add metrics text
            metrics_text = (
                f"KGE: {metrics['KGE']:.4f}\n"
                f"NSE: {metrics['NSE']:.4f}\n"
                f"RMSE: {metrics['RMSE']:.4f}\n"
                f"PBIAS: {metrics['PBIAS']:.4f}%"
            )
            
            axes[0].text(0.02, 0.95, metrics_text, transform=axes[0].transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot scatter
            axes[1].scatter(obs_aligned, sim_aligned, alpha=0.5)
            
            # Add 1:1 line
            max_val = max(obs_aligned.max(), sim_aligned.max())
            min_val = min(obs_aligned.min(), sim_aligned.min())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
            
            axes[1].set_xlabel('Observed Streamflow (m³/s)')
            axes[1].set_ylabel('Simulated Streamflow (m³/s)')
            axes[1].set_title('Scatter Plot')
            axes[1].grid(True, alpha=0.3)
            
            # Add metrics text to scatter plot
            axes[1].text(0.02, 0.95, metrics_text, transform=axes[1].transAxes,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "final_streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a flow duration curve plot
            self._plot_flow_duration_curve(obs_aligned, sim_aligned, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating comparison plot: {str(e)}")
            
    def _plot_flow_duration_curve(self, observed: pd.Series, simulated: pd.Series, output_dir: Path):
        """
        Create a flow duration curve plot.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
            output_dir: Directory to save plot
        """
        try:
            # Make sure both series are numeric
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            # Drop NaNs
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                self.logger.warning("No valid data for flow duration curve")
                return
                
            # Sort values in descending order
            obs_sorted = observed.sort_values(ascending=False)
            sim_sorted = simulated.sort_values(ascending=False)
            
            # Calculate exceedance probabilities
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot flow duration curves
            plt.plot(obs_exceed, obs_sorted, 'b-', linewidth=2, label='Observed')
            plt.plot(sim_exceed, sim_sorted, 'r-', linewidth=2, label='Simulated (DDS)')
            
            # Set log scale for y-axis
            plt.yscale('log')
            
            # Add labels and legend
            plt.xlabel('Exceedance Probability')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Flow Duration Curves')
            plt.grid(True, which='both', alpha=0.3)
            plt.legend()
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "flow_duration_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating flow duration curve: {str(e)}")
            
    def run_comparison_with_rf_emulation(self):
        """
        Run a comparison between traditional DDS optimization and RF emulation.
        
        This method runs both optimization approaches and compares their results,
        providing visualizations and metrics to evaluate their relative performance.
        
        Returns:
            Dict: Dictionary with comparison results
        """
        self.logger.info("Starting comparison between DDS and RF emulation")
        
        # Create comparison directory
        comparison_dir = self.project_dir / "optimisation" / f"{self.experiment_id}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Run traditional DDS optimization
        self.logger.info("Running traditional DDS optimization")
        dds_start_time = time.time()
        dds_best_params, dds_best_value = self.optimize_with_dds()
        dds_elapsed_time = time.time() - dds_start_time
        
        self.logger.info(f"DDS optimization completed in {dds_elapsed_time:.2f} seconds")
        self.logger.info(f"DDS best {self.target_metric}: {dds_best_value:.4f}")
        
        # 2. Run RF emulation (using RandomForestEmulator)
        self.logger.info("Running RF emulation optimization")
        rf_start_time = time.time()
        
        try:
            # Import RandomForestEmulator class
            from utils.emulation_utils.random_forest_emulator import RandomForestEmulator
            
            # Initialize RF emulator
            rf_emulator = RandomForestEmulator(self.config, self.logger)
            
            # Run RF emulation workflow
            rf_results = rf_emulator.run_workflow()
            
            if rf_results:
                rf_best_params = rf_results['optimized_parameters']
                rf_best_value = rf_results['predicted_score']
                rf_elapsed_time = time.time() - rf_start_time
                
                self.logger.info(f"RF emulation completed in {rf_elapsed_time:.2f} seconds")
                self.logger.info(f"RF best {self.target_metric}: {rf_best_value:.4f}")
                
                # 3. Compare results
                comparison_results = self._compare_optimization_results(
                    dds_best_params, dds_best_value, dds_elapsed_time,
                    rf_best_params, rf_best_value, rf_elapsed_time,
                    comparison_dir
                )
                
                return comparison_results
            else:
                self.logger.error("RF emulation failed to produce results")
                return None
                
        except ImportError:
            self.logger.error("Could not import RandomForestEmulator class")
            return None
        except Exception as e:
            self.logger.error(f"Error during RF emulation: {str(e)}")
            return None
    
    def _compare_optimization_results(self, 
                                     dds_params: Dict[str, float], dds_value: float, dds_time: float,
                                     rf_params: Dict[str, float], rf_value: float, rf_time: float,
                                     output_dir: Path) -> Dict:
        """
        Compare results from DDS and RF emulation optimization.
        
        Args:
            dds_params: Best parameters from DDS
            dds_value: Best objective value from DDS
            dds_time: Time taken by DDS optimization
            rf_params: Best parameters from RF emulation
            rf_value: Best objective value from RF emulation
            rf_time: Time taken by RF emulation
            output_dir: Directory to save comparison results
            
        Returns:
            Dict: Dictionary with comparison results
        """
        try:
            # Create a dataframe with parameter values
            comparison_df = pd.DataFrame({
                'Parameter': list(dds_params.keys()),
                'DDS': list(dds_params.values()),
                'RF': [rf_params.get(param, np.nan) for param in dds_params.keys()]
            })
            
            # Calculate parameter differences
            comparison_df['Difference'] = comparison_df['DDS'] - comparison_df['RF']
            comparison_df['Difference %'] = 100 * comparison_df['Difference'] / comparison_df['DDS']
            
            # Save comparison dataframe
            comparison_df.to_csv(output_dir / "parameter_comparison.csv", index=False)
            
            # Create performance comparison
            performance_df = pd.DataFrame({
                'Method': ['DDS', 'RF Emulation'],
                self.target_metric: [dds_value, rf_value],
                'Time (s)': [dds_time, rf_time],
                'Speedup': [1.0, dds_time / rf_time]
            })
            
            # Save performance comparison
            performance_df.to_csv(output_dir / "performance_comparison.csv", index=False)
            
            # Create parameter comparison plot
            self._plot_parameter_comparison(comparison_df, output_dir)
            
            # Create performance metrics plot
            self._plot_performance_comparison(performance_df, output_dir)
            
            # Run model with both parameter sets and compare results
            # (Optional - this could be computationally expensive)
            self._compare_streamflow_results(dds_params, rf_params, output_dir)
            
            # Return summary of comparison
            return {
                'DDS': {
                    'best_value': dds_value,
                    'time': dds_time,
                    'params': dds_params
                },
                'RF': {
                    'best_value': rf_value,
                    'time': rf_time,
                    'params': rf_params
                },
                'comparison': {
                    'param_diff_max': comparison_df['Difference'].abs().max(),
                    'param_diff_mean': comparison_df['Difference'].abs().mean(),
                    'value_diff': dds_value - rf_value,
                    'value_diff_pct': 100 * (dds_value - rf_value) / dds_value,
                    'speedup': dds_time / rf_time
                },
                'output_dir': str(output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing optimization results: {str(e)}")
            return None
            
    def _plot_parameter_comparison(self, comparison_df: pd.DataFrame, output_dir: Path):
        """
        Create a comparison plot of parameter values from both methods.
        
        Args:
            comparison_df: DataFrame with parameter comparisons
            output_dir: Directory to save plot
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Number of parameters
            n_params = len(comparison_df)
            
            # Set up bar positions
            ind = np.arange(n_params)
            width = 0.35
            
            # Normalize parameter values for better visualization
            normalized_df = comparison_df.copy()
            for param in comparison_df['Parameter']:
                if param in self.param_bounds:
                    min_val, max_val = self.param_bounds[param]
                    param_range = max_val - min_val
                    
                    # Normalize DDS
                    normalized_df.loc[normalized_df['Parameter'] == param, 'DDS'] = \
                        (comparison_df.loc[comparison_df['Parameter'] == param, 'DDS'] - min_val) / param_range
                    
                    # Normalize RF
                    normalized_df.loc[normalized_df['Parameter'] == param, 'RF'] = \
                        (comparison_df.loc[comparison_df['Parameter'] == param, 'RF'] - min_val) / param_range
            
            # Plot bars
            plt.bar(ind - width/2, normalized_df['DDS'], width, label='DDS')
            plt.bar(ind + width/2, normalized_df['RF'], width, label='RF Emulation')
            
            # Add labels and legend
            plt.xlabel('Parameter')
            plt.ylabel('Normalized Parameter Value')
            plt.title('Parameter Comparison - DDS vs RF Emulation')
            plt.xticks(ind, normalized_df['Parameter'], rotation=45, ha='right')
            plt.legend()
            
            # Add grid
            plt.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create parameter difference plot
            plt.figure(figsize=(12, 8))
            
            # Plot parameter differences
            plt.bar(ind, comparison_df['Difference %'], width)
            
            # Add labels
            plt.xlabel('Parameter')
            plt.ylabel('Difference (%)')
            plt.title('Parameter Difference (DDS - RF)')
            plt.xticks(ind, comparison_df['Parameter'], rotation=45, ha='right')
            
            # Add reference line at zero
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add grid
            plt.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "parameter_difference.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating parameter comparison plot: {str(e)}")
            
    def _plot_performance_comparison(self, performance_df: pd.DataFrame, output_dir: Path):
        """
        Create a comparison plot of performance metrics from both methods.
        
        Args:
            performance_df: DataFrame with performance comparisons
            output_dir: Directory to save plot
        """
        try:
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot objective values
            axes[0].bar(['DDS', 'RF Emulation'], performance_df[self.target_metric])
            axes[0].set_ylabel(self.target_metric)
            axes[0].set_title(f'{self.target_metric} Comparison')
            axes[0].grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df[self.target_metric]):
                axes[0].text(i, v + 0.01, f"{v:.4f}", ha='center')
            
            # Plot computation time
            axes[1].bar(['DDS', 'RF Emulation'], performance_df['Time (s)'])
            axes[1].set_ylabel('Time (s)')
            axes[1].set_title('Computation Time Comparison')
            axes[1].grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df['Time (s)']):
                axes[1].text(i, v + 0.01, f"{v:.2f}s", ha='center')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create speedup plot
            plt.figure(figsize=(8, 6))
            
            # Plot speedup
            plt.bar(['DDS', 'RF Emulation'], performance_df['Speedup'])
            plt.ylabel('Relative Speed (1 = baseline)')
            plt.title('Computational Efficiency Comparison')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df['Speedup']):
                plt.text(i, v + 0.01, f"{v:.2f}x", ha='center')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_dir / "speedup_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison plot: {str(e)}")
            
    def _compare_streamflow_results(self, dds_params: Dict[str, float], rf_params: Dict[str, float], output_dir: Path):
        """
        Compare streamflow results from both parameter sets.
        
        Args:
            dds_params: Best parameters from DDS
            rf_params: Best parameters from RF emulation
            output_dir: Directory to save comparison
        """
        try:
            # Create directories for each method
            dds_dir = output_dir / "dds_validation"
            rf_dir = output_dir / "rf_validation"
            dds_dir.mkdir(parents=True, exist_ok=True)
            rf_dir.mkdir(parents=True, exist_ok=True)
            
            # Run model with DDS parameters
            self._run_final_simulation(dds_params, dds_dir)
            
            # Run model with RF parameters
            self._run_final_simulation(rf_params, rf_dir)
            
            # Extract streamflow results
            dds_flow = pd.read_csv(dds_dir / "final_simulated_streamflow.csv", index_col=0, parse_dates=True)
            rf_flow = pd.read_csv(rf_dir / "final_simulated_streamflow.csv", index_col=0, parse_dates=True)
            
            # Load observed data
            obs_flow = self._load_observed_flow()
            
            if dds_flow.empty or rf_flow.empty or obs_flow.empty:
                self.logger.warning("Missing data for streamflow comparison")
                return
                
            # Get common time period
            common_idx = obs_flow.index.intersection(dds_flow.index.intersection(rf_flow.index))
            
            if len(common_idx) == 0:
                self.logger.warning("No common time steps for streamflow comparison")
                return
                
            # Extract aligned data
            obs_aligned = obs_flow.loc[common_idx, 'streamflow']
            dds_aligned = dds_flow.loc[common_idx, 'streamflow']
            rf_aligned = rf_flow.loc[common_idx, 'streamflow']
            
            # Calculate performance metrics
            dds_metrics = self._calculate_all_metrics(dds_flow, obs_flow)
            rf_metrics = self._calculate_all_metrics(rf_flow, obs_flow)
            
            # Create comparison plot
            plt.figure(figsize=(14, 8))
            
            # Plot streamflow time series
            plt.plot(obs_aligned.index, obs_aligned, 'k-', linewidth=2, label='Observed')
            plt.plot(dds_aligned.index, dds_aligned, 'b-', linewidth=1.5, label='DDS')
            plt.plot(rf_aligned.index, rf_aligned, 'r-', linewidth=1.5, label='RF Emulation')
            
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Streamflow Comparison - DDS vs RF Emulation')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add metrics text
            metrics_text = (
                f"DDS - KGE: {dds_metrics['KGE']:.4f}, NSE: {dds_metrics['NSE']:.4f}, RMSE: {dds_metrics['RMSE']:.4f}\n"
                f"RF  - KGE: {rf_metrics['KGE']:.4f}, NSE: {rf_metrics['NSE']:.4f}, RMSE: {rf_metrics['RMSE']:.4f}"
            )
            
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for metrics text at bottom
            plt.savefig(output_dir / "streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save metrics comparison to CSV
            metrics_df = pd.DataFrame([
                {'Method': 'DDS', **dds_metrics},
                {'Method': 'RF Emulation', **rf_metrics}
            ])
            metrics_df.to_csv(output_dir / "metrics_comparison.csv", index=False)
            
        except Exception as e:
            self.logger.error(f"Error comparing streamflow results: {str(e)}")
            raise
            
    def optimize_with_dds(self):
        """
        Optimize parameters using the Dynamically Dimensioned Search (DDS) algorithm.
        
        The DDS algorithm is a single-solution-based heuristic global search algorithm
        designed for computationally expensive optimization problems like watershed model
        calibration. It dynamically adjusts the search dimension as the optimization
        progresses.
        
        Returns:
            Tuple[Dict[str, float], float]: Optimized parameters and final objective value
        """
        self.logger.info("Starting optimization with DDS algorithm")
        
        # Create output directory for this optimization run
        run_dir = self.optimization_output_dir / f"run_{self.experiment_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parameters - start with midpoint of bounds
        initial_params = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            initial_params[param] = min_val + (max_val - min_val) / 2.0
            
        # Create results dataframe to track progress
        results_df = pd.DataFrame(columns=['Iteration', 'Objective'] + list(self.param_bounds.keys()))
        
        # Generate initial parameter set
        current_params = initial_params.copy()
        
        # Run model with initial parameters to get baseline performance
        current_obj_value = self._evaluate_parameters(current_params)
        best_params = current_params.copy()
        best_obj_value = current_obj_value
        
        # Store initial result
        result_row = {'Iteration': 0, 'Objective': current_obj_value}
        result_row.update(current_params)
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
        
        # Main DDS loop
        start_time = time.time()
        try:
            for i in range(1, self.max_iterations + 1):
                # Calculate probability of inclusion based on iteration
                p_i = 1.0 - np.log(i) / np.log(self.max_iterations)
                
                # Dynamically select dimensions to perturb
                dims_to_perturb = []
                for param in self.params_to_optimize:
                    if np.random.random() < p_i:
                        dims_to_perturb.append(param)
                        
                # If no dimensions selected, pick one randomly
                if not dims_to_perturb:
                    dims_to_perturb = [np.random.choice(self.params_to_optimize)]
                    
                # Create new candidate by perturbing selected dimensions
                new_params = current_params.copy()
                
                for param in dims_to_perturb:
                    min_val, max_val = self.param_bounds[param]
                    param_range = max_val - min_val
                    
                    # Neighborhood perturbation
                    sigma = self.r_value * param_range
                    perturbation = np.random.normal(0, sigma)
                    
                    # Apply perturbation
                    new_val = current_params[param] + perturbation
                    
                    # Enforce bounds
                    new_val = max(min_val, min(max_val, new_val))
                    new_params[param] = new_val
                    
                # Evaluate new parameter set
                new_obj_value = self._evaluate_parameters(new_params)
                
                # Update current solution (for DDS, we keep the better solution)
                if new_obj_value > current_obj_value:  # Assuming higher is better for KGE
                    current_params = new_params.copy()
                    current_obj_value = new_obj_value
                    
                    # Update best solution if needed
                    if current_obj_value > best_obj_value:
                        best_params = current_params.copy()
                        best_obj_value = current_obj_value
                        
                # Store result
                result_row = {'Iteration': i, 'Objective': new_obj_value}
                result_row.update(new_params)
                results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
                
                # Log progress
                elapsed_time = time.time() - start_time
                if i % 10 == 0 or i == 1:
                    self.logger.info(f"Iteration {i}/{self.max_iterations}, Best {self.target_metric}: {best_obj_value:.4f}, "
                                     f"Current: {new_obj_value:.4f}, Elapsed time: {elapsed_time:.2f} s")
                    
                # Save progress to file
                results_df.to_csv(run_dir / "dds_optimization_progress.csv", index=False)
                
        except KeyboardInterrupt:
            self.logger.warning("Optimization interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise
            
        # Final logging and saving
        elapsed_time = time.time() - start_time
        self.logger.info(f"DDS Optimization completed in {elapsed_time:.2f} seconds")
        self.logger.info(f"Best {self.target_metric}: {best_obj_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Save final results
        results_df.to_csv(run_dir / "dds_optimization_results.csv", index=False)
        with open(run_dir / "dds_best_parameters.txt", 'w') as f:
            f.write(f"Best {self.target_metric}: {best_obj_value:.6f}\n")
            f.write("Best parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value:.6f}\n")
                
        # Create visualization
        self._plot_optimization_progress(results_df, run_dir)
        
        # Run final simulation with best parameters and save outputs
        self._run_final_simulation(best_params, run_dir)
        
        return best_params, best_obj_value
        
    def _evaluate_parameters(self, params: Dict[str, float]) -> float:
        """
        Evaluate a parameter set by running the hydrological model.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            float: Objective function value (KGE or other metric)
        """
        # Create temporary directory for this evaluation
        eval_dir = self.optimization_output_dir / "evaluations" / f"eval_{int(time.time())}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create trial parameter file
        trial_params_path = self._create_trial_params(params, eval_dir)
        
        # Create modified fileManager
        filemanager_path = self._create_filemanager(eval_dir)
        
        # Run SUMMA
        summa_success = self._run_summa(filemanager_path, eval_dir)
        if not summa_success:
            self.logger.warning("SUMMA run failed, returning poor objective value")
            return -1.0  # Return a poor value for KGE
            
        # Run mizuRoute if needed (for distributed domains)
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        if not is_lumped:
            mizuroute_success = self._run_mizuroute(eval_dir)
            if not mizuroute_success:
                self.logger.warning("MizuRoute run failed, returning poor objective value")
                return -1.0
                
        # Extract streamflow from model output
        sim_flow = self._extract_streamflow(eval_dir)
        if sim_flow is None or sim_flow.empty:
            self.logger.warning("Failed to extract streamflow from model output")
            return -1.0
            
        # Load observed data
        obs_flow = self._load_observed_flow()
        if obs_flow is None or obs_flow.empty:
            self.logger.warning("Failed to load observed flow data")
            return -1.0
            
        # Calculate objective function
        obj_value = self._calculate_objective(sim_flow, obs_flow)
        
        # Clean up evaluation directory to save space (optional)
        # import shutil
        # shutil.rmtree(eval_dir)
        
        return obj_value
        
    def _create_trial_params(self, params: Dict[str, float], eval_dir: Path) -> Path:
        """
        Create a trial parameters file for SUMMA with the given parameter values.
        
        Args:
            params: Dictionary of parameter values
            eval_dir: Directory to place the trial parameters file
            
        Returns:
            Path: Path to the created trial parameters file
        """
        # Create settings directory
        settings_dir = eval_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the trial parameters file
        trial_params_path = settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
        
        # Path to attributes file for reading HRU information
        attr_file = self.summa_settings_dir / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
        
        if not attr_file.exists():
            self.logger.error(f"Attributes file not found: {attr_file}")
            raise FileNotFoundError(f"Attributes file not found: {attr_file}")
            
        try:
            # Read HRU information from attributes file
            with xr.open_dataset(attr_file) as ds:
                hru_ids = ds['hruId'].values
                
                # Create trial parameters dataset
                output_ds = xr.Dataset()
                
                # Add dimensions
                output_ds['hru'] = ('hru', np.arange(len(hru_ids)))
                
                # Add HRU ID variable
                output_ds['hruId'] = ('hru', hru_ids)
                output_ds['hruId'].attrs['long_name'] = 'Hydrologic Response Unit ID (HRU)'
                output_ds['hruId'].attrs['units'] = '-'
                
                # Add parameter variables
                for param_name, param_value in params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Add to dataset
                    output_ds[param_name] = ('hru', param_array)
                    output_ds[param_name].attrs['long_name'] = f"Trial value for {param_name}"
                    output_ds[param_name].attrs['units'] = "N/A"
                
                # Add global attributes
                output_ds.attrs['description'] = "SUMMA Trial Parameter file generated by CONFLUENCE Traditional Optimizer"
                output_ds.attrs['history'] = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.attrs['confluence_experiment_id'] = self.experiment_id
                
                # Save to file
                output_ds.to_netcdf(trial_params_path)
                
            # Copy other necessary files
            self._copy_static_settings_files(settings_dir)
            
            return trial_params_path
            
        except Exception as e:
            self.logger.error(f"Error creating trial parameters file: {str(e)}")
            raise
            
    def _copy_static_settings_files(self, settings_dir: Path):
        """
        Copy static settings files from the original settings directory.
        
        Args:
            settings_dir: Target directory for settings files
        """
        # Files to copy
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
            source_path = self.summa_settings_dir / file_name
            if source_path.exists():
                dest_path = settings_dir / file_name
                try:
                    import shutil
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")
                    
    def _create_filemanager(self, eval_dir: Path) -> Path:
        """
        Create a modified filemanager for this evaluation.
        
        Args:
            eval_dir: Directory for this evaluation
            
        Returns:
            Path: Path to the modified filemanager
        """
        # Path to original filemanager
        original_fm_path = self.summa_settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not original_fm_path.exists():
            self.logger.error(f"Original filemanager not found: {original_fm_path}")
            raise FileNotFoundError(f"Original filemanager not found: {original_fm_path}")
            
        # Path to new filemanager
        settings_dir = eval_dir / "settings" / "SUMMA"
        new_fm_path = settings_dir / os.path.basename(original_fm_path)
        
        try:
            # Read original filemanager
            with open(original_fm_path, 'r') as f:
                lines = f.readlines()
                
            # Create simulations directory
            output_dir = eval_dir / "simulations" / self.experiment_id / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Modify filemanager
            with open(new_fm_path, 'w') as f:
                for line in lines:
                    if "settingsPath" in line:
                        # Update settings path
                        settings_path_str = str(settings_dir).replace('\\', '/')
                        f.write(f"settingsPath         '{settings_path_str}/'\n")
                    elif "outputPath" in line:
                        # Update output path
                        output_path_str = str(output_dir).replace('\\', '/')
                        f.write(f"outputPath           '{output_path_str}/'\n")
                    elif "outFilePrefix" in line:
                        # Update output file prefix
                        prefix_parts = line.split("'")
                        if len(prefix_parts) >= 3:
                            original_prefix = prefix_parts[1]
                            new_prefix = f"{self.experiment_id}_optimization"
                            new_line = line.replace(f"'{original_prefix}'", f"'{new_prefix}'")
                            f.write(new_line)
                        else:
                            f.write(line)
                    else:
                        f.write(line)
                        
            return new_fm_path
            
        except Exception as e:
            self.logger.error(f"Error creating filemanager: {str(e)}")
            raise
            
    def _run_summa(self, filemanager_path: Path, eval_dir: Path) -> bool:
        """
        Run SUMMA with the given filemanager.
        
        Args:
            filemanager_path: Path to the filemanager file
            eval_dir: Directory for this evaluation
            
        Returns:
            bool: True if SUMMA run was successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = eval_dir / "simulations" / self.experiment_id / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "summa.log"
            
            # Run SUMMA
            summa_command = f"{self.summa_exec} -m {filemanager_path}"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    summa_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
                
            # Check if output files were created
            output_files = list(output_dir.glob(f"{self.experiment_id}_optimization*.nc"))
            if not output_files:
                self.logger.warning(f"No output files found after SUMMA run in {output_dir}")
                return False
                
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error running SUMMA: {str(e)}")
            return False
            
    def _run_mizuroute(self, eval_dir: Path) -> bool:
        """
        Run mizuRoute for distributed domains.
        
        Args:
            eval_dir: Directory for this evaluation
            
        Returns:
            bool: True if mizuRoute run was successful, False otherwise
        """
        # Only run mizuRoute for distributed domains
        if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped':
            return True
            
        # Get mizuRoute settings and executable paths
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Setup mizuRoute settings
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        mizu_settings_dir = eval_dir / "settings" / "mizuRoute"
        mizu_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mizuRoute output directory
        mizu_output_dir = eval_dir / "simulations" / self.experiment_id / "mizuRoute"
        mizu_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mizuRoute settings files
        for file in os.listdir(source_mizu_dir):
            source_file = source_mizu_dir / file
            if source_file.is_file():
                import shutil
                shutil.copy2(source_file, mizu_settings_dir / file)
                
        # Create modified control file
        control_file_path = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        if not self._update_mizuroute_control_file(control_file_path, eval_dir):
            self.logger.error("Failed to update mizuRoute control file")
            return False
            
        try:
            # Run mizuRoute
            log_file = mizu_output_dir / "mizuroute.log"
            mizu_command = f"{mizu_exe} {control_file_path}"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
                
            # Check if output files were created
            output_files = list(mizu_output_dir.glob("*.nc"))
            if not output_files:
                self.logger.warning(f"No output files found after mizuRoute run in {mizu_output_dir}")
                return False
                
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error running mizuRoute: {str(e)}")
            return False
            
    def _update_mizuroute_control_file(self, control_file_path: Path, eval_dir: Path) -> bool:
        """
        Update mizuRoute control file for this evaluation.
        
        Args:
            control_file_path: Path to the mizuRoute control file
            eval_dir: Directory for this evaluation
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        if not control_file_path.exists():
            self.logger.error(f"mizuRoute control file not found: {control_file_path}")
            return False
            
        try:
            # Read the control file
            with open(control_file_path, 'r') as f:
                lines = f.readlines()
                
            # Get relevant paths
            summa_output_dir = eval_dir / "simulations" / self.experiment_id / "SUMMA"
            mizu_settings_dir = eval_dir / "settings" / "mizuRoute"
            mizu_output_dir = eval_dir / "simulations" / self.experiment_id / "mizuRoute"
            
            # Update the control file
            with open(control_file_path, 'w') as f:
                for line in lines:
                    if '<input_dir>' in line:
                        # Update input directory
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
                        f.write(f"<case_name>             {self.experiment_id}_optimization    ! Simulation case name \n")
                    elif '<fname_qsim>' in line:
                        # Update input file name
                        f.write(f"<fname_qsim>            {self.experiment_id}_optimization_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                    else:
                        f.write(line)
                        
            return True
            
        except Exception as e:
            self.logger.error('Hi')
            raise