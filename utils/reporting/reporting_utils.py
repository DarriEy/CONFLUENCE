import sys
import xarray as xr # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.dates as mdates # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
from matplotlib.colors import LinearSegmentedColormap, ListedColormap # type: ignore
import traceback
import rasterio # type: ignore
from matplotlib import gridspec
from matplotlib.lines import Line2D # type: ignore
from matplotlib.patches import Patch # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore


class VisualizationReporter:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"


    def plot_lumped_streamflow_simulations_vs_observations(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]], show_calib_eval_periods: bool = False, spinup_percent: float = 10.0):
        """
        Create a streamflow comparison visualization for lumped watershed models
        where the streamflow is directly taken from the SUMMA output as averageRoutedRunoff.
        
        Args:
            model_outputs: List of tuples (model_name, model_file_path)
            obs_files: List of tuples (observation_name, observation_file_path)
            show_calib_eval_periods: Whether to show calibration and evaluation period statistics
            spinup_percent: Percentage of data period to skip at the beginning (default: 10%)
            
        Returns:
            Path to the saved plot
        """
        try:
            plot_folder = self.project_dir / "plots" / "results"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'streamflow_comparison.png'

            # Read observation data
            obs_data = []
            for obs_name, obs_file in obs_files:
                try:
                    df = pd.read_csv(obs_file, parse_dates=['datetime'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df = df['discharge_cms'].resample('h').mean()
                    obs_data.append((obs_name, df))
                except Exception as e:
                    self.logger.warning(f"Could not read observation file {obs_file}: {str(e)}")
                    continue

            if not obs_data:
                self.logger.error("No observation data could be loaded")
                return None

            # Read simulation data from SUMMA output (averageRoutedRunoff)
            sim_data = []
            for sim_name, sim_file in model_outputs:
                try:
                    ds = xr.open_dataset(sim_file)
                    # Extract the averageRoutedRunoff variable
                    if 'averageRoutedRunoff' in ds:
                        runoff_series = ds['averageRoutedRunoff'].to_pandas()
                        
                        # Handle different xarray/pandas output formats
                        if isinstance(runoff_series, pd.DataFrame):
                            # If it's a DataFrame with gru as columns
                            if 'gru' in ds['averageRoutedRunoff'].dims:
                                # If multiple GRUs exist, use the first one for now
                                runoff_series = runoff_series.iloc[:, 0]
                            else:
                                # Other case - take the first column
                                runoff_series = runoff_series.iloc[:, 0]
                        
                        # Get catchment area from the river basin shapefile
                        basin_shapefile = self.config.get('RIVER_BASINS_NAME')
                        if basin_shapefile == 'default':
                            basin_shapefile = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
                        basin_path = self.project_dir / "shapefiles" / "river_basins" / basin_shapefile
                        
                        try:
                            basin_gdf = gpd.read_file(basin_path)
                            area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                            # Area is expected to be in m²
                            area_m2 = basin_gdf[area_col].sum() 
                            
                            # Convert from m/s to m³/s (cms) by multiplying by area in m²
                            runoff_series = runoff_series * area_m2
                            
                            self.logger.info(f"Converted runoff from m/s to m³/s using basin area: {area_m2} m²")
                        except Exception as e:
                            self.logger.warning(f"Error getting basin area: {str(e)}. Using raw values.")
                        
                        sim_data.append((sim_name, runoff_series))
                    else:
                        self.logger.error(f"No 'averageRoutedRunoff' variable found in {sim_file}")
                        continue
                            
                except Exception as e:
                    self.logger.warning(f"Could not read simulation file {sim_file}: {str(e)}")
                    continue

            if not sim_data:
                self.logger.error("No simulation data could be loaded")
                return None

            # Determine time range based on available data
            start_date = max([data.index.min() for _, data in sim_data + obs_data])
            end_date = min([data.index.max() for _, data in sim_data + obs_data])
            
            # Calculate spinup period duration
            total_days = (end_date - start_date).days
            spinup_days = int(total_days * spinup_percent / 100)
            spinup_end_date = start_date + pd.Timedelta(days=spinup_days)
            
            self.logger.info(f"Skipping first {spinup_days} days ({spinup_percent}% of total period) as spinup")
            self.logger.info(f"Using data from {spinup_end_date} to {end_date}")

            # Filter data to common time range, skipping spinup period
            sim_data = [(name, data.loc[spinup_end_date:end_date]) for name, data in sim_data]
            obs_data = [(name, data.loc[spinup_end_date:end_date]) for name, data in obs_data]

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.suptitle("Lumped Watershed Streamflow Comparison", fontsize=16, fontweight='bold')

            # Plot time series
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            linestyles = ['--', '-.', ':']

            # Plot observations
            for obs_name, obs in obs_data:
                ax1.plot(obs.index, obs, label=f'Observed ({obs_name})', color='black', linewidth=2.5)

            # Plot simulations
            for i, (sim_name, sim) in enumerate(sim_data):
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                ax1.plot(sim.index, sim, label=f'Simulated ({sim_name})', 
                        color=color, linestyle=linestyle, linewidth=1.5)

            # Add metrics calculation
            for i, (sim_name, sim) in enumerate(sim_data):
                # Align observation and simulation data
                df_obs = obs_data[0][1]
                df_sim = sim
                
                # Create merged dataframe with common timestamps
                aligned_data = pd.merge(df_obs, df_sim, 
                                    left_index=True, right_index=True, how='inner')
                
                # Calculate metrics
                if not aligned_data.empty:
                    metrics = self.calculate_metrics(aligned_data.iloc[:, 0].values, 
                                                aligned_data.iloc[:, 1].values)
                    
                    # Format metrics text
                    metric_text = f"{sim_name} Metrics:\n"
                    metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
                    
                    # Add text to plot
                    ax1.text(0.02, 0.98 - 0.15 * i, metric_text,
                            transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax1.set_title(f'Streamflow Comparison (after {spinup_percent}% spinup period)', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.set_facecolor('#f0f0f0')

            # Format x-axis
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Plot exceedance frequency
            try:
                for obs_name, obs in obs_data:
                    self.plot_exceedance(ax2, obs.values, f'Observed ({obs_name})', 
                                    color='black', linewidth=2.5)

                for i, (sim_name, sim) in enumerate(sim_data):
                    color = colors[i % len(colors)]
                    linestyle = linestyles[i % len(linestyles)]
                    self.plot_exceedance(ax2, sim.values, f'Simulated ({sim_name})', 
                                    color=color, linestyle=linestyle, linewidth=1.5)
            except Exception as e:
                self.logger.warning(f'Could not plot exceedance frequency: {str(e)}')

            ax2.set_xlabel('Exceedance Probability', fontsize=12)
            ax2.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, which='both', linestyle=':', alpha=0.6)
            ax2.set_facecolor('#f0f0f0')

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            # Save the plot
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_lumped_streamflow_simulations_vs_observations: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def visualise_model_output(self, config_path):
        
        # Plot streamflow comparison
        self.logger.info('Starting model output visualisation')
        for model in self.config.get('HYDROLOGICAL_MODEL').split(','):
            visualizer = VisualizationReporter(self.config, self.logger)

            if model == 'SUMMA':
                visualizer.plot_summa_outputs(self.config['EXPERIMENT_ID'])
                
                # Check if using lumped domain definition
                if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped':
                    # For lumped model, get streamflow from SUMMA's averageRoutedRunoff
                    self.logger.info("Using lumped model output from SUMMA averageRoutedRunoff")
                    
                    # Define model_outputs and obs_files
                    summa_output_file = str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "SUMMA" / f"{self.config['EXPERIMENT_ID']}_timestep.nc")
                    model_outputs = [(f"{model}", summa_output_file)]
                    
                    obs_files = [
                        ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                    ]
                    
                    try:
                        # First convert SUMMA output to a format compatible with the visualization function
                        import xarray as xr
                        import pandas as pd
                        import numpy as np
                        import os
                        
                        # Read SUMMA output
                        ds = xr.open_dataset(summa_output_file)
                        
                        # Extract averageRoutedRunoff
                        runoff = ds['averageRoutedRunoff'].to_pandas()
                        
                        # If it's a DataFrame, convert to Series (depends on xarray version)
                        if isinstance(runoff, pd.DataFrame):
                            runoff = runoff.iloc[:, 0]
                            
                        # Need to get basin area to convert from m/s to m³/s (cms)
                        basin_name = self.config.get('RIVER_BASINS_NAME', 'default')
                        if basin_name == 'default':
                            basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
                        
                        # Using the helper function for getting the file path
                        basin_path = self.project_dir / "shapefiles" / "river_basins" / basin_name
                        if not basin_path.exists() and hasattr(self, '_get_file_path'):
                            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
                        
                        import geopandas as gpd
                        basin_gdf = gpd.read_file(basin_path)
                        area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                        area_m2 = basin_gdf[area_col].sum()
                        
                        # Convert from m/s to m³/s (cms)
                        runoff = runoff * area_m2
                        
                        # Make a netCDF file that mimics mizuRoute output format
                        modified_file = str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "SUMMA" / f"{self.config['EXPERIMENT_ID']}_modified_for_viz.nc")
                        
                        # Create a new dataset with the expected structure
                        new_ds = xr.Dataset(
                            data_vars={
                                "IRFroutedRunoff": (["time", "seg"], runoff.values[:, np.newaxis]),
                                "reachID": (["seg"], np.array([1]))
                            },
                            coords={
                                "time": runoff.index,
                                "seg": np.array([0])
                            }
                        )
                        new_ds.to_netcdf(modified_file)
                        
                        # Update model_outputs to use the modified file
                        model_outputs = [(model, modified_file)]
                        
                        # Now call the usual visualization function
                        plot_file = visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                        
                    except Exception as e:
                        self.logger.error(f"Error preparing lumped SUMMA output for visualization: {str(e)}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        
                else:
                    # For distributed model, use mizuRoute output as usual
                    # Only attempt to update SIM_REACH_ID if config_path is a string path
                    if isinstance(config_path, str):
                        visualizer.update_sim_reach_id(config_path)  # Find and update the sim reach id based on the project pour point
                    else:
                        # Just update in-memory config without updating file
                        visualizer.update_sim_reach_id()
                    model_outputs = [
                        (f"{model}", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "mizuRoute" / f"{self.config['EXPERIMENT_ID']}*.nc"))
                    ]
                    obs_files = [
                        ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                    ]
                    plot_file = visualizer.plot_streamflow_simulations_vs_observations(model_outputs, obs_files)
                    
            elif model == 'FUSE':
                model_outputs = [
                    ("FUSE", str(self.project_dir / "simulations" / self.config['EXPERIMENT_ID'] / "FUSE" / f"{self.config['DOMAIN_NAME']}_{self.config['EXPERIMENT_ID']}_runs_best.nc"))
                ]
                obs_files = [
                    ('Observed', str(self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"))
                ]
                plot_file = visualizer.plot_fuse_streamflow_simulations_vs_observations(model_outputs, obs_files)

            elif model == 'GR':
                pass

            elif model == 'FLASH':
                pass


    def plot_streamflow_simulations_vs_observations(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]], show_calib_eval_periods: bool = False):
        try:
            plot_folder = self.project_dir / "plots" / "results"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'streamflow_comparison.png'

            # Read simulation data first
            sim_data = []
            for sim_name, sim_file in model_outputs:
                try:
                    ds = xr.open_mfdataset(sim_file, engine='netcdf4')
                    basinID = int(self.config.get('SIM_REACH_ID'))  
                    segment_index = ds['reachID'].values == basinID
                    ds = ds.sel(seg=ds['seg'][segment_index])
                    df = ds['IRFroutedRunoff'].to_dataframe().reset_index()
                    df.set_index('time', inplace=True)
                    sim_data.append((sim_name, df))
                except Exception as e:
                    self.logger.warning(f"Could not read simulation file {sim_file}: {str(e)}")
                    continue

            if not sim_data:
                self.logger.error("No simulation data could be loaded")
                return None

            # Try to read observation data
            obs_data = []
            for obs_name, obs_file in obs_files:
                try:
                    df = pd.read_csv(obs_file, parse_dates=['datetime'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df = df['discharge_cms'].resample('h').mean()
                    obs_data.append((obs_name, df))
                except Exception as e:
                    self.logger.warning(f"Could not read observation file {obs_file}: {str(e)}")
                    continue

            # Determine time range based on simulations
            start_date = max([data.index.min() for _, data in sim_data]) + pd.Timedelta(100, unit="d")
            end_date = min([data.index.max() for _, data in sim_data])

            # Filter simulation data to common time range
            sim_data = [(name, data.loc[start_date:end_date]) for name, data in sim_data]

            # Filter observation data if it exists
            if obs_data:
                obs_data = [(name, data.loc[start_date:end_date]) for name, data in obs_data]

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.suptitle("Streamflow Comparison and Flow Duration Curve", fontsize=16, fontweight='bold')

            # Plot time series
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            linestyles = ['--', '-.', ':']

            # Plot observations if available
            if obs_data:
                for obs_name, obs in obs_data:
                    ax1.plot(obs.index, obs, label=f'Observed ({obs_name})', color='black', linewidth=2.5)

            # Plot simulations
            for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                ax1.plot(sim.index, sim['IRFroutedRunoff'], label=f'Simulated ({sim_name})', 
                        color=color, linestyle=linestyle, linewidth=1.5)

            # Add metrics if observations are available
            if obs_data and show_calib_eval_periods:
                self._add_calibration_evaluation_metrics(ax1, obs_data, sim_data)
            elif obs_data:
                self._add_overall_metrics(ax1, obs_data, sim_data)

            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax1.set_title('Streamflow Comparison', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.set_facecolor('#f0f0f0')

            # Format x-axis
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Plot exceedance frequency
            try:
                if obs_data:
                    for obs_name, obs in obs_data:
                        self.plot_exceedance(ax2, obs.values, f'Observed ({obs_name})', color='black', linewidth=2.5)

                for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                    self.plot_exceedance(ax2, sim['IRFroutedRunoff'].values, f'Simulated ({sim_name})', 
                                    color=color, linestyle=linestyle, linewidth=1.5)
            except Exception as e:
                self.logger.warning(f'Could not plot exceedance frequency: {str(e)}')

            ax2.set_xlabel('Exceedance Probability', fontsize=12)
            ax2.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, which='both', linestyle=':', alpha=0.6)
            ax2.set_facecolor('#f0f0f0')

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            # Save the plot
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_streamflow_simulations_vs_observations: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _add_calibration_evaluation_metrics(self, ax, obs_data, sim_data):
        """Helper method to add calibration and evaluation period metrics to the plot"""
        calib_start = pd.Timestamp('2011-01-01')
        calib_end = pd.Timestamp('2014-12-31')
        eval_start = pd.Timestamp('2015-01-01')
        eval_end = pd.Timestamp('2018-12-31')

        for i, (sim_name, sim) in enumerate(sim_data):
            aligned_calib = pd.merge(obs_data[0][1].loc[calib_start:calib_end], 
                                    sim.loc[calib_start:calib_end, 'IRFroutedRunoff'], 
                                    left_index=True, right_index=True, how='inner')
            calib_metrics = self.calculate_metrics(aligned_calib.iloc[:, 0].values, aligned_calib.iloc[:, 1].values)
            
            aligned_eval = pd.merge(obs_data[0][1].loc[eval_start:eval_end], 
                                    sim.loc[eval_start:eval_end, 'IRFroutedRunoff'], 
                                    left_index=True, right_index=True, how='inner')
            eval_metrics = self.calculate_metrics(aligned_eval.iloc[:, 0].values, aligned_eval.iloc[:, 1].values)
            
            metric_text = f"{sim_name} Metrics:\nCalibration (2011-2014):\n"
            metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in calib_metrics.items()])
            metric_text += "\n\nEvaluation (2015-2018):\n"
            metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in eval_metrics.items()])
            
            ax.text(0.02, 0.98 - 0.35 * i, metric_text,
                    transform=ax.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

        ax.axvspan(calib_start, calib_end, alpha=0.2, color='gray', label='Calibration Period')
        ax.axvspan(eval_start, eval_end, alpha=0.2, color='lightblue', label='Evaluation Period')

    def _add_overall_metrics(self, ax, obs_data, sim_data):
        """Helper method to add overall metrics to the plot"""
        for i, (sim_name, sim) in enumerate(sim_data):
            sim.index = sim.index.round(freq='h')
            aligned_data = pd.merge(obs_data[0][1], sim['IRFroutedRunoff'], 
                                    left_index=True, right_index=True, how='inner')
            metrics = self.calculate_metrics(aligned_data.iloc[:, 0].values, aligned_data.iloc[:, 1].values)
            
            metric_text = f"{sim_name} Metrics:\n"
            metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            
            ax.text(0.02, 0.98 - 0.15 * i, metric_text,
                    transform=ax.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
        
    def plot_snow_simulations_vs_observations(self, model_outputs: List[List[str]]):
        try:
            # Read and prepare data
            snow_obs = pd.read_csv(Path(self.config.snow_processed_path) / self.config.snow_processed_name)
            snow_obs['datetime'] = pd.to_datetime(snow_obs['datetime'], errors='coerce')
            station_gdf = gpd.read_file(Path(self.config.snow_station_shapefile_path) / self.config.snow_station_shapefile_name)
            snow_obs['station_id'] = snow_obs['station_id'].astype(str)
            merged_data = pd.merge(snow_obs, station_gdf, on='station_id')
            model_data = [(name, xr.open_dataset(file)) for name, file in model_outputs]

            # Filter observations to simulation period
            sim_start = pd.Timestamp(min(ds.time.min().values for _, ds in model_data)) + pd.Timedelta(50, unit="d")
            sim_end = pd.Timestamp(max(ds.time.max().values for _, ds in model_data))
            merged_data = merged_data[(merged_data['datetime'] >= sim_start) & (merged_data['datetime'] <= sim_end)]

            unique_hrus = merged_data['HRU_ID'].unique()
            n_hrus = len(unique_hrus)

            # Set up the main figure
            fig_all = plt.figure(figsize=(20, 10*n_hrus + 8))
            gs = GridSpec(n_hrus + 1, 1, height_ratios=[10]*n_hrus + [2])
            fig_all.suptitle("Snow Water Equivalent Comparison", fontsize=16, fontweight='bold')

            # Color palette for simulations
            sim_colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))

            # Color palette for observations (using a different colormap for more contrast)
            obs_colors = plt.cm.Set1(np.linspace(0, 1, len(station_gdf)))

            results = {'plot_file': '', 'individual_plots': [], 'metrics': {}}

            for idx, hru_id in enumerate(unique_hrus):
                hru_data = merged_data[merged_data['HRU_ID'] == hru_id]

                # Create individual figure for each HRU
                fig_individual = plt.figure(figsize=(15, 8))
                ax_individual = fig_individual.add_subplot(111)

                # Plot on both the main figure and the individual figure
                for ax in [fig_all.add_subplot(gs[idx]), ax_individual]:
                    # Plot simulations
                    for i, (sim_name, model_output) in enumerate(model_data):
                        hru_sim = model_output['scalarSWE'].sel(hru=hru_id).to_dataframe()
                        ax.plot(hru_sim.index, hru_sim['scalarSWE'], label=f'Simulated ({sim_name})', 
                                color=sim_colors[i], linewidth=2, linestyle=['--', '-.', ':'][i % 3])

                    # Plot observations
                    stations = hru_data['station_id'].unique()
                    for j, station in enumerate(stations):
                        station_data = hru_data[hru_data['station_id'] == station].sort_values('datetime')
                        
                        if self.is_continuous_data(station_data['datetime']):
                            # Plot as a line if data is continuous
                            ax.plot(station_data['datetime'], station_data['snw'], 
                                    label=f'Observed (Station {station})', 
                                    color=obs_colors[j], alpha=0.7, linewidth=2)
                        else:
                            # Plot as scatter if data is not continuous
                            ax.scatter(station_data['datetime'], station_data['snw'], 
                                    label=f'Observed (Station {station})', 
                                    color=obs_colors[j], alpha=0.7, s=50, edgecolors='black')

                    ax.set_title(f'Snow Water Equivalent - HRU {hru_id}', fontsize=14, fontweight='bold')
                    ax.set_ylabel('SWE (mm)', fontsize=12)
                    ax.set_xlabel('Date', fontsize=12)
                    
                    # Move legend inside the plot
                    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.99, 0.99), 
                              ncol=1, borderaxespad=0, frameon=True, fancybox=True, shadow=True)
                    
                    ax.grid(True, linestyle=':', alpha=0.6)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    ax.set_facecolor('#f0f0f0')  # Light gray background

                    # Format x-axis to show years
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

                    # Calculate and display metrics
                    for i, (sim_name, model_output) in enumerate(model_data):
                        sim_data = model_output['scalarSWE'].sel(hru=hru_id).to_dataframe()['scalarSWE']
                        aligned_data = pd.merge(hru_data.set_index('datetime')['snw'], sim_data, left_index=True, right_index=True, how='outer')
                        metrics = self.calculate_metrics_snow(aligned_data['snw'].values, aligned_data['scalarSWE'].values)
                        metric_text = f"{sim_name} Metrics:\n"
                        metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items() if not np.isnan(v)])
                        ax.text(0.02, 0.98 - 0.15 * i, metric_text, transform=ax.transAxes, 
                                verticalalignment='top', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

                # Save individual plot
                plot_folder = self.project_dir / "plots"
                plot_folder.mkdir(parents=True, exist_ok=True)
                individual_plot_filename = plot_folder / f'snow_comparison_HRU_{hru_id}.png'
                fig_individual.tight_layout()
                fig_individual.savefig(individual_plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig_individual)
                results['individual_plots'].append(str(individual_plot_filename))

                # Calculate metrics
                hru_metrics = self.calculate_all_metrics(hru_data, model_data, hru_id)
                results['metrics'][hru_id] = hru_metrics

            # Add overall metrics table to the main figure
            ax_table = fig_all.add_subplot(gs[-1])
            ax_table.axis('off')
            overall_metrics = self.calculate_overall_metrics(results['metrics'])
            self.create_metrics_table(ax_table, overall_metrics)

            fig_all.tight_layout()
            plt.subplots_adjust(top=0.95)  # Adjust for main title
            
            # Save the main plot
            plot_filename = plot_folder / 'snow_comparison_all_hrus.png'
            fig_all.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_all)

            results['plot_file'] = str(plot_filename)
            return results

        except Exception as e:
            print(f"Error in plot_snow_simulations_vs_observations: {str(e)}")
            return {}

    # Add these helper methods to the VisualizationReporter class
    def is_continuous_data(self, dates, threshold_days=20):
        date_diffs = dates.diff().dt.total_seconds() / (24 * 3600)  # Convert to days
        return (date_diffs.fillna(0) <= threshold_days).all()

    def calculate_metrics_snow(self, obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
        # Ensure obs and sim have the same length
        min_length = min(len(obs), len(sim))
        obs = obs[:min_length]
        sim = sim[:min_length]
        
        # Remove NaN values
        mask = ~np.isnan(obs) & ~np.isnan(sim)
        obs = obs[mask]
        sim = sim[mask]
        
        if len(obs) == 0:
            return {metric: np.nan for metric in ['RMSE', 'KGE', 'KGEp', 'NSE', 'MAE', 'KGEnp']}
        
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, transfo=1)
        }

    def calculate_all_metrics(self, hru_data, model_data, hru_id):
        metrics = {}
        for station in hru_data['station_id'].unique():
            station_data = hru_data[hru_data['station_id'] == station].set_index('datetime')['snw']
            for sim_name, model_output in model_data:
                sim_data = model_output['scalarSWE'].sel(hru=hru_id).to_dataframe()['scalarSWE']
                
                # Align the data and fill missing values with NaN
                aligned_data = pd.merge(station_data, sim_data, left_index=True, right_index=True, how='outer')
                metrics[f"HRU {hru_id} - {station} vs {sim_name}"] = self.calculate_metrics_snow(aligned_data['snw'].values, aligned_data['scalarSWE'].values)
        return metrics

    def calculate_overall_metrics(self, hru_metrics):
        overall_metrics = {}
        for hru, metrics in hru_metrics.items():
            for comparison, values in metrics.items():
                if comparison not in overall_metrics:
                    overall_metrics[comparison] = {metric: [] for metric in values.keys()}
                for metric, value in values.items():
                    overall_metrics[comparison][metric].append(value)
        
        for comparison in overall_metrics:
            for metric in overall_metrics[comparison]:
                overall_metrics[comparison][metric] = np.mean(overall_metrics[comparison][metric])
        
        return overall_metrics

    def create_metrics_table(self, ax, metrics):
        cell_text = []
        rows = []
        for comparison, values in metrics.items():
            rows.append(comparison)
            cell_text.append([f"{value:.3f}" for value in values.values()])
        
        columns = list(next(iter(metrics.values())).keys())
        
        # Create a custom colormap for the table
        cmap = LinearSegmentedColormap.from_list("", ["white", "lightblue"])
        
        # Create cell colors with the correct number of columns
        num_rows = len(cell_text)
        num_cols = len(columns)
        cell_colors = cmap(np.linspace(0, 0.5, num_rows * num_cols)).reshape(num_rows, num_cols, 4)
        
        table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, 
                         cellLoc='center', loc='center', cellColours=cell_colors)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
        
        return table

    def calculate_metrics(self, obs: np.ndarray, sim: np.ndarray) -> dict:
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, transfo=1)
        }

    def plot_exceedance(self, ax, data, label, color=None, linestyle='-', linewidth=1):
        sorted_data = np.sort(data)[::-1]
        exceedance = np.arange(1, len(data) + 1) / len(data)
        ax.plot(exceedance, sorted_data, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        ax.set_xscale('log')
        ax.set_yscale('log')

    def update_sim_reach_id(self, config_path=None):
        """
        Update the SIM_REACH_ID in both the config object and YAML file by finding the 
        nearest river segment to the pour point.
        
        Args:
            config_path: Either a path to the config file or None. If None, will only update the in-memory config.
        """
        try:
            # Load the pour point shapefile
            pour_point_name = self.config.get('POUR_POINT_SHP_NAME')
            if pour_point_name == 'default':
                pour_point_name = f"{self.config.get('DOMAIN_NAME')}_pourPoint.shp"
            
            pour_point_path = self._get_file_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', pour_point_name)
            
            pour_point_gdf = gpd.read_file(pour_point_path)

            # Load the river network shapefile
            river_network_name = self.config.get('RIVER_NETWORK_SHP_NAME')
            if river_network_name == 'default':
                river_network_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_delineate.shp"

            river_network_path = self._get_file_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', river_network_name)
            river_network_gdf = gpd.read_file(river_network_path)

            # Ensure both GeoDataFrames have the same CRS
            if pour_point_gdf.crs != river_network_gdf.crs:
                pour_point_gdf = pour_point_gdf.to_crs(river_network_gdf.crs)

            # Get the pour point coordinates
            pour_point = pour_point_gdf.geometry.iloc[0]

            # Convert to a projected CRS for accurate distance calculation
            # Use UTM zone based on the data's centroid
            center_lon = pour_point.centroid.x
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_crs = f"EPSG:326{utm_zone if pour_point.centroid.y >= 0 else utm_zone+30}"

            # Reproject both geometries to UTM
            pour_point_utm = pour_point_gdf.to_crs(utm_crs).geometry.iloc[0]
            river_network_utm = river_network_gdf.to_crs(utm_crs)

            # Find the nearest stream segment to the pour point
            nearest_segment = river_network_utm.iloc[river_network_utm.distance(pour_point_utm).idxmin()]

            # Get the ID of the nearest segment
            reach_id = nearest_segment[self.config.get('RIVER_NETWORK_SHP_SEGID')]

            # Update the config object
            self.config['SIM_REACH_ID'] = reach_id
            
            # Update the YAML config file if a file path was provided
            if config_path is not None and isinstance(config_path, (str, Path)):
                config_file_path = Path(config_path)
                
                if not config_file_path.exists():
                    self.logger.error(f"Config file not found at {config_file_path}")
                    return None

                # Read the current config file
                with open(config_file_path, 'r') as f:
                    config_lines = f.readlines()

                # Find and update the SIM_REACH_ID line
                for i, line in enumerate(config_lines):
                    if line.strip().startswith('SIM_REACH_ID:'):
                        config_lines[i] = f"SIM_REACH_ID: {reach_id}                                              # River reach ID used for streamflow evaluation and optimization\n"
                        break
                else:
                    # If SIM_REACH_ID line not found, add it in the Simulation settings section
                    for i, line in enumerate(config_lines):
                        if line.strip().startswith('## Simulation settings'):
                            config_lines.insert(i + 1, f"SIM_REACH_ID: {reach_id}                                              # River reach ID used for streamflow evaluation and optimization\n")
                            break

                # Write the updated config back to file
                with open(config_file_path, 'w') as f:
                    f.writelines(config_lines)

                self.logger.info(f"Updated SIM_REACH_ID to {reach_id} in both config object and file: {config_file_path}")
            else:
                self.logger.info(f"Updated SIM_REACH_ID to {reach_id} in config object only (no file path provided or invalid path type)")
            
            return reach_id

        except Exception as e:
            self.logger.error(f"Error updating SIM_REACH_ID: {str(e)}")
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            return None

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
        
    def plot_domain(self):
        """
        Creates a map visualization of the delineated domain with a basemap.
        Plots catchment boundary, river network (if not lumped), and pour point on top of a contextily basemap.
        """
        try:
            # Create plot folder if it doesn't exist
            plot_folder = self.project_dir / "plots" / "domain"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'domain_map.png'
            
            # Load the shapefiles
            # Catchment
            catchment_name = self.config.get('RIVER_BASINS_NAME')
            if catchment_name == 'default':
                catchment_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"
            catchment_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', catchment_name)
            catchment_gdf = gpd.read_file(catchment_path)
            
            # Check if we need to load river network (not needed for lumped domains)
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
            load_river_network = domain_method not in ['lumped', 'point']
            
            # River network - only load if not lumped
            if load_river_network:
                river_name = self.config.get('RIVER_NETWORK_SHP_NAME')
                if river_name == 'default':
                    river_name = f"{self.config.get('DOMAIN_NAME')}_riverNetwork_delineate.shp"
                river_path = self._get_file_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', river_name)
                river_gdf = gpd.read_file(river_path)
            
            # Pour point
            pour_point_name = self.config.get('POUR_POINT_SHP_NAME')
            if pour_point_name == 'default':
                pour_point_name = f"{self.config.get('DOMAIN_NAME')}_pourPoint.shp"
            pour_point_path = self._get_file_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', pour_point_name)
            pour_point_gdf = gpd.read_file(pour_point_path)
            
            # Reproject to Web Mercator for contextily basemap
            catchment_gdf = catchment_gdf.to_crs(epsg=3857)
            if load_river_network:
                river_gdf = river_gdf.to_crs(epsg=3857)
            pour_point_gdf = pour_point_gdf.to_crs(epsg=3857)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Reproject everything to Web Mercator for consistency with basemap
            catchment_gdf_web = catchment_gdf.to_crs(epsg=3857)
            if load_river_network:
                river_gdf_web = river_gdf.to_crs(epsg=3857)
            pour_point_gdf_web = pour_point_gdf.to_crs(epsg=3857)
            
            # Calculate buffer for extent (10% of the catchment bounds)
            bounds = catchment_gdf_web.total_bounds
            buffer_x = (bounds[2] - bounds[0]) * 0.1
            buffer_y = (bounds[3] - bounds[1]) * 0.1
            plot_bounds = [
                bounds[0] - buffer_x,
                bounds[1] - buffer_y,
                bounds[2] + buffer_x,
                bounds[3] + buffer_y
            ]
            
            # Set map extent with buffer
            ax.set_xlim([plot_bounds[0], plot_bounds[2]])
            ax.set_ylim([plot_bounds[1], plot_bounds[3]])
            
            # Add the global basemap
            #self.plot_with_global_basemap(ax, plot_bounds)
            
            # Plot the catchment boundary with a clean style
            catchment_gdf_web.boundary.plot(
                ax=ax,
                linewidth=2,
                color='#2c3e50',
                label='Catchment Boundary',
                zorder=2
            )
            
            # Plot the river network only if not lumped
            if load_river_network:
                # Plot the river network with line width based on stream order
                if 'StreamOrde' in river_gdf_web.columns:
                    min_order = river_gdf_web['StreamOrde'].min()
                    max_order = river_gdf_web['StreamOrde'].max()
                    river_gdf_web['line_width'] = river_gdf_web['StreamOrde'].apply(
                        lambda x: 0.5 + 2 * (x - min_order) / (max_order - min_order)
                    )
                    
                    for idx, row in river_gdf_web.iterrows():
                        ax.plot(
                            row.geometry.xy[0],
                            row.geometry.xy[1],
                            color='#3498db',
                            linewidth=row['line_width'],
                            zorder=3,
                            alpha=0.8
                        )
                else:
                    river_gdf_web.plot(
                        ax=ax,
                        color='#3498db',
                        linewidth=1,
                        label='River Network',
                        zorder=3,
                        alpha=0.8
                    )
            
            # Plot the pour point with a prominent style
            pour_point_gdf_web.plot(
                ax=ax,
                color='#e74c3c',
                marker='*',
                markersize=200,
                label='Pour Point',
                zorder=4
            )
            
            self._add_north_arrow(ax)
            
            # Add title
            plt.title(f'Delineated Domain: {self.config.get("DOMAIN_NAME")}', 
                    fontsize=16, pad=20, fontweight='bold')
            
            # Create custom legend with larger symbols
            legend_elements = [
                Line2D([0], [0], color='#2c3e50', linewidth=2, label='Catchment Boundary'),
                Line2D([0], [0], color='#e74c3c', marker='*', label='Pour Point',
                    markersize=15, linewidth=0)
            ]
            
            # Add river network to legend only if it exists
            if load_river_network:
                legend_elements.insert(1, Line2D([0], [0], color='#3498db', linewidth=2, label='River Network'))
            
            ax.legend(handles=legend_elements, loc='upper right', 
                    frameon=True, facecolor='white', framealpha=0.9)
            
            # Remove axes
            ax.set_axis_off()
            
            # Add domain information box
            self._add_domain_info_box(ax, catchment_gdf_web)
            
            # Save the plot
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            self.logger.info(f"Domain map saved to {plot_filename}")
            return str(plot_filename)
            
        except Exception as e:
            self.logger.error(f"Error in plot_domain: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _add_domain_info_box(self, ax, catchment_gdf):
        """Add a information box with domain statistics"""
        # Calculate total area in km²
        area_km2 = catchment_gdf.geometry.area.sum() / 1e6
        
        # Create info text
        info_text = f"Domain Statistics:\n"
        info_text += f"Area: {area_km2:.1f} km²"
        
        # Add elevation info if available
        if 'elev_mean' in catchment_gdf.columns:
            min_elev = catchment_gdf['elev_mean'].min()
            max_elev = catchment_gdf['elev_mean'].max()
            info_text += f"\nElevation Range: {min_elev:.0f} - {max_elev:.0f} m"

        # Position text box in lower left corner
        ax.text(0.02, 0.02, info_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10),
                fontsize=10,
                verticalalignment='bottom')

    def _add_north_arrow(self, ax):
        """Add a north arrow to the map"""
        # Calculate arrow position (upper left corner)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Position in upper left with some padding
        arrow_x = xmin + (xmax - xmin) * 0.05
        arrow_y_top = ymax - (ymax - ymin) * 0.05
        arrow_y_bottom = arrow_y_top - (ymax - ymin) * 0.05
        
        # Create north arrow
        ax.annotate('N', 
                    xy=(arrow_x, arrow_y_top), 
                    xytext=(arrow_x, arrow_y_bottom),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', 
                    va='center', 
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=5))

    def _add_info_box(self, ax, catchment_gdf):
        """Add an information box with catchment statistics"""
        # Calculate catchment area in km²
        area_km2 = catchment_gdf.geometry.area.sum() / 1e6  # Convert from m² to km²
        
        # Create info text
        info_text = f"Catchment Statistics:\n"
        info_text += f"Area: {area_km2:.1f} km²\n"
        
        # Add elevation info if available
        if 'Elevation' in catchment_gdf.columns:
            min_elev = catchment_gdf['Elevation'].min()
            max_elev = catchment_gdf['Elevation'].max()
            info_text += f"Elevation Range: {min_elev:.0f} - {max_elev:.0f} m"

        # Position text box in lower left corner
        ax.text(0.02, 0.02, info_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10),
                fontsize=10,
                verticalalignment='bottom')
   
    def plot_with_global_basemap(self, ax, bounds):
        """
        Adds the global basemap to a matplotlib axis.
        
        Args:
            ax: Matplotlib axis
            bounds: [xmin, ymin, xmax, ymax] in EPSG:3857
        """
        try:
            
            basemap_path = Path(self.config['BASEMAP_PATH'])
            
            if basemap_path is None:
                self.logger.warning("Global basemap not found in any standard location. Plotting without basemap.")
                return
                
            self.logger.info(f"Using global basemap from {basemap_path}")
            
            with rasterio.open(basemap_path) as src:
                # Calculate window to read based on bounds
                window = rasterio.windows.from_bounds(
                    bounds[0], bounds[1],
                    bounds[2], bounds[3],
                    src.transform
                )
                
                # Calculate output shape maintaining aspect ratio
                aspect_ratio = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])
                target_width = 2048  # Adjust this for performance/quality balance
                target_height = int(target_width / aspect_ratio)
                
                # Read data
                data = src.read(
                    window=window,
                    out_shape=(3, target_height, target_width)
                )
                
                # Plot the basemap
                ax.imshow(
                    np.transpose(data, (1, 2, 0)),
                    extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                    interpolation='bilinear',
                    zorder=-1
                )
                
                self.logger.info("Global basemap added successfully")
                
        except Exception as e:
            self.logger.warning(f"Error adding global basemap: {str(e)}")
            self.logger.warning("Continuing without basemap")

        
    def plot_discretized_domain(self, discretization_method: str) -> str:
        """
        Creates a map visualization of the discretized domain with a clean, professional style.
        
        Args:
            discretization_method (str): Method used for discretization (e.g., 'elevation', 'landclass')
        
        Returns:
            str: Path to the saved plot
        """
        try:
            # Create plot folder if it doesn't exist
            plot_folder = self.project_dir / "plots" / "discretization"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / f'domain_discretization_{discretization_method}.png'

            # Load cathcment shapefile
            catchment_name = self.config.get('CATCHMENT_SHP_NAME')
            if catchment_name == 'default':
                catchment_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{discretization_method}.shp"
            catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', catchment_name)
            hru_gdf = gpd.read_file(catchment_path)

            # Check if we need to load river network (not needed for lumped domains)
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
            load_river_network = domain_method != 'lumped'

            # River network - only load if not lumped
            if load_river_network:
                river_name = self.config.get('RIVER_NETWORK_SHP_NAME')
                if river_name == 'default':
                    river_name = f"{self.config.get('DOMAIN_NAME')}_riverNetwork_delineate.shp"
                river_path = self._get_file_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network', river_name)
                river_gdf = gpd.read_file(river_path)

            # Load pour point for context
            pour_point_name = self.config.get('POUR_POINT_SHP_NAME')
            if pour_point_name == 'default':
                pour_point_name = f"{self.config.get('DOMAIN_NAME')}_pourPoint.shp"
            pour_point_path = self._get_file_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', pour_point_name)
            pour_point_gdf = gpd.read_file(pour_point_path)

            # Create figure and axis with high DPI for better quality
            fig, ax = plt.subplots(figsize=(15, 15), dpi=300)

            # Reproject everything to Web Mercator
            hru_gdf_web = hru_gdf.to_crs(epsg=3857)
            if load_river_network:
                river_gdf_web = river_gdf.to_crs(epsg=3857)
            pour_point_gdf_web = pour_point_gdf.to_crs(epsg=3857)

            # Calculate bounds with buffer
            bounds = hru_gdf_web.total_bounds
            buffer_x = (bounds[2] - bounds[0]) * 0.1
            buffer_y = (bounds[3] - bounds[1]) * 0.1
            plot_bounds = [
                bounds[0] - buffer_x,
                bounds[1] - buffer_y,
                bounds[2] + buffer_x,
                bounds[3] + buffer_y
            ]

            # Set map extent
            ax.set_xlim([plot_bounds[0], plot_bounds[2]])
            ax.set_ylim([plot_bounds[1], plot_bounds[3]])

            # Get the column name and setup legend title
            class_mappings = {
                'elevation': {'col': 'elevClass', 'title': 'Elevation Classes', 'cm': 'terrain'},
                'soilclass': {'col': 'soilClass', 'title': 'Soil Classes', 'cm': 'Set3'},
                'landclass': {'col': 'landClass', 'title': 'Land Use Classes', 'cm': 'Set2'},
                'radiation': {'col': 'radiationClass', 'title': 'Radiation Classes', 'cm': 'YlOrRd'},
                'default': {'col': 'HRU_ID', 'title': 'HRU Classes', 'cm': 'tab20'}
            }

            # Get mapping for the discretization method, defaulting to HRU if not found
            mapping = class_mappings.get(discretization_method.lower(), class_mappings['default'])
            class_col = mapping['col']
            legend_title = mapping['title']

            # Get unique classes and determine number of colors needed
            unique_classes = sorted(hru_gdf[class_col].unique())
            n_classes = len(unique_classes)

            # Create color map based on number of classes
            if discretization_method.lower() == 'elevation':
                # Calculate elevation ranges for labels
                elev_range = hru_gdf['elev_mean'].agg(['min', 'max'])
                min_elev = int(elev_range['min'])
                max_elev = int(elev_range['max'])
                band_size = int(self.config.get('ELEVATION_BAND_SIZE', 400))
                
                # Create labels for each class
                legend_labels = []
                for cls in unique_classes:
                    lower = min_elev + ((cls-1) * band_size)
                    upper = min_elev + (cls * band_size)
                    if upper > max_elev:
                        upper = max_elev
                    legend_labels.append(f'{lower}-{upper}m')
            else:
                legend_labels = [f'Class {i}' for i in unique_classes]

            # Get colors from the specified colormap
            base_cmap = plt.get_cmap(mapping['cm'])

            # Handle cases where we need more colors than the base colormap provides
            if n_classes > base_cmap.N:
                # Use multiple colormaps and combine them
                additional_cmaps = ['Set3', 'Set2', 'Set1', 'Paired', 'tab20']
                all_colors = []
                
                # First, get colors from the base colormap
                all_colors.extend([base_cmap(i) for i in np.linspace(0, 1, base_cmap.N)])
                
                # Then add colors from additional colormaps until we have enough
                for cmap_name in additional_cmaps:
                    if len(all_colors) >= n_classes:
                        break
                    cmap = plt.get_cmap(cmap_name)
                    all_colors.extend([cmap(i) for i in np.linspace(0, 1, cmap.N)])
                
                # Ensure we have exactly n_classes colors by interpolating if necessary
                if len(all_colors) < n_classes:
                    colors_array = np.array(all_colors)
                    indices = np.linspace(0, len(colors_array) - 1, n_classes)
                    colors = [colors_array[int(i)] if i < len(colors_array) else colors_array[-1] for i in indices]
                else:
                    colors = all_colors[:n_classes]
            else:
                colors = [base_cmap(i) for i in np.linspace(0, 1, n_classes)]

            # Create custom colormap and normalization
            cmap = ListedColormap(colors)
            norm = plt.Normalize(vmin=min(unique_classes)-0.5, vmax=max(unique_classes)+0.5)

            hru_plot = hru_gdf_web.plot(
                column=class_col,
                ax=ax,
                cmap=cmap,
                norm=norm,
                alpha=0.7,
                legend=False
            )

            # Create custom legend
            legend_elements = [Patch(facecolor=colors[i],
                                alpha=0.7,
                                label=legend_labels[i]) 
                            for i in range(n_classes)]

            # Add legend with scrollbar if too many classes
            if n_classes > 20:
                # Create a second figure for the legend
                legend_fig = plt.figure(figsize=(2, 6))
                legend_ax = legend_fig.add_axes([0, 0, 1, 1])
                legend_ax.set_axis_off()
                
                # Create scrollable legend
                legend = legend_ax.legend(handles=legend_elements,
                                        title=legend_title,
                                        loc='center',
                                        bbox_to_anchor=(0.5, 0.5),
                                        frameon=True,
                                        fancybox=True,
                                        shadow=True,
                                        ncol=1,
                                        mode="expand")
                
                # Save separate legend file
                legend_filename = plot_filename.parent / f'{plot_filename.stem}_legend.png'
                legend_fig.savefig(legend_filename, bbox_inches='tight', dpi=300)
                plt.close(legend_fig)
                
                # Add note about separate legend file
                ax.text(0.98, 0.02, 'See separate legend file',
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
            else:
                # Add legend to main plot if not too many classes
                ax.legend(handles=legend_elements,
                        title=legend_title,
                        loc='center left',
                        bbox_to_anchor=(1, 0.5),
                        frameon=True,
                        fancybox=True,
                        shadow=True)

            # Plot HRU boundaries
            hru_gdf_web.boundary.plot(
                ax=ax,
                linewidth=0.5,
                color='#2c3e50',
                alpha=0.5
            )

            # Plot river network only if not lumped
            if load_river_network:
                if 'StreamOrde' in river_gdf_web.columns:
                    min_order = river_gdf_web['StreamOrde'].min()
                    max_order = river_gdf_web['StreamOrde'].max()
                    river_gdf_web['line_width'] = river_gdf_web['StreamOrde'].apply(
                        lambda x: 0.5 + 2 * (x - min_order) / (max_order - min_order)
                    )
                    for idx, row in river_gdf_web.iterrows():
                        ax.plot(
                            row.geometry.xy[0],
                            row.geometry.xy[1],
                            color='#3498db',
                            linewidth=row['line_width'],
                            zorder=3,
                            alpha=0.8
                        )
                else:
                    river_gdf_web.plot(
                        ax=ax,
                        color='#3498db',
                        linewidth=1,
                        label='River Network',
                        zorder=3,
                        alpha=0.8
                    )

            # Plot pour point
            pour_point_gdf_web.plot(
                ax=ax,
                color='#e74c3c',
                marker='*',
                markersize=200,
                label='Pour Point',
                zorder=4
            )

            # Add north arrow
            self._add_north_arrow(ax)

            # Add title
            plt.title(f'Domain Discretization: {discretization_method.title()}', 
                    fontsize=16, pad=20, fontweight='bold')

            # Add info box with statistics
            self._add_discretization_info_box(ax, hru_gdf_web)

            # Remove axes
            ax.set_axis_off()

            # Save the plot
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            self.logger.info(f"Discretization map saved to {plot_filename}")
            return str(plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_discretized_domain: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _add_discretization_info_box(self, ax, hru_gdf):
        """Add an information box with discretization statistics"""
        # Calculate statistics
        n_hrus = len(hru_gdf)
        total_area = hru_gdf.geometry.area.sum() / 1e6  # Convert to km²
        mean_area = total_area / n_hrus
        
        # Create info text
        info_text = f"Discretization Statistics:\n"
        info_text += f"Number of HRUs: {n_hrus}\n"
        info_text += f"Total Area: {total_area:.1f} km²\n"
        info_text += f"Mean HRU Area: {mean_area:.1f} km²"

        # Add elevation info if available
        if 'elev_mean' in hru_gdf.columns:
            min_elev = hru_gdf['elev_mean'].min()
            max_elev = hru_gdf['elev_mean'].max()
            info_text += f"\nElevation Range: {min_elev:.0f} - {max_elev:.0f} m"

        # Position text box in lower left corner
        ax.text(0.02, 0.02, info_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10),
                fontsize=10,
                verticalalignment='bottom')
        
    def plot_fuse_streamflow_simulations_vs_observations(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]) -> Optional[Path]:
        """
        Plot FUSE simulated streamflow against observations with calibration/evaluation period shading.
        Only plots the overlapping time period between observations and simulations.
        Converts FUSE output from mm/day to cms using GRU areas.
        """
        self.logger.info("Creating FUSE streamflow comparison plot")
        
        try:
            # Create single figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Get observation data and time range
            obs_data = []
            obs_start = None
            obs_end = None
            for obs_name, obs_file in obs_files:
                obs_df = pd.read_csv(obs_file, parse_dates=['datetime'])
                obs_df.set_index('datetime', inplace=True)
                obs_data.append((obs_name, obs_df))
                
                # Update overall observation time range
                if obs_start is None or obs_df.index.min() > obs_start:
                    obs_start = obs_df.index.min()
                if obs_end is None or obs_df.index.max() < obs_end:
                    obs_end = obs_df.index.max()
            
            # Get simulation data and time range
            sim_data = []
            sim_start = None
            sim_end = None
            
            for model_name, output_file in model_outputs:
                if model_name.upper() == 'FUSE':
                    with xr.open_dataset(output_file) as ds:
                        self.logger.info(f"Opened FUSE output file: {output_file}")
                        
                        try:
                            # Get q_routed data
                            sim_flow = ds['q_routed'].isel(
                                param_set=0,
                                latitude=0,
                                longitude=0
                            )
                            
                            # Convert to pandas series with datetime index
                            sim_series = sim_flow.to_pandas()
                            
                            # Get area from river basins shapefile
                            basin_name = self.config.get('RIVER_BASINS_NAME')
                            if basin_name == 'default':
                                basin_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins_delineate.shp"
                            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
                            basin_gdf = gpd.read_file(basin_path)
                            
                            # Convert units from mm/day to cms
                            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
                            sim_series = sim_series * area_km2 / 86.4
                            
                            sim_data.append((model_name, sim_series))
                            
                            # Update overall simulation time range
                            if sim_start is None or sim_series.index.min() > sim_start:
                                sim_start = sim_series.index.min()
                            if sim_end is None or sim_series.index.max() < sim_end:
                                sim_end = sim_series.index.max()
                                
                        except Exception as e:
                            self.logger.error(f"Error processing FUSE output: {str(e)}")
                            self.logger.error(f"Traceback: {traceback.format_exc()}")
                            continue
            
            # Determine overlapping time period
            start_date = max(obs_start, sim_start)
            end_date = min(obs_end, sim_end)
            
            self.logger.info(f"Plotting overlapping period: {start_date} to {end_date}")
            
            # Plot simulations for overlapping period
            for model_name, sim_series in sim_data:
                sim_flow = sim_series.loc[start_date:end_date]
                ax.plot(sim_flow.index, sim_flow.values, '-', 
                       label='FUSE', linewidth=1.5, color='#1f77b4')
                
            # Plot observations for overlapping period
            for obs_name, obs_df in obs_data:
                obs_flow = obs_df['discharge_cms'].loc[start_date:end_date]
                ax.plot(obs_flow.index, obs_flow, 'k-', label=f'Observed', linewidth=1.5)
            
            # Get calibration and evaluation periods
            cal_start = pd.Timestamp(self.config.get('CALIBRATION_PERIOD').split(',')[0].strip())
            cal_end = pd.Timestamp(self.config.get('CALIBRATION_PERIOD').split(',')[1].strip())
            eval_start = pd.Timestamp(self.config.get('EVALUATION_PERIOD').split(',')[0].strip())
            eval_end = pd.Timestamp(self.config.get('EVALUATION_PERIOD').split(',')[1].strip())
            
            # Add shaded regions only within the overlapping period
            cal_start = max(cal_start, start_date)
            cal_end = min(cal_end, end_date)
            eval_start = max(eval_start, start_date)
            eval_end = min(eval_end, end_date)
            
            #ax.axvspan(cal_start, cal_end, alpha=0.2, color='gray', label='Calibration Period')
            #ax.axvspan(eval_start, eval_end, alpha=0.2, color='lightblue', label='Evaluation Period')
            
            # Rest of the plotting code remains the same
            ax.set_title('FUSE Streamflow Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Streamflow (cms)', fontsize=12)
            
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.project_dir / "plots" / "results" / f"{self.config.get('EXPERIMENT_ID')}_FUSE_streamflow_comparison.png"
            plot_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"FUSE streamflow comparison plot saved to {plot_file}")
            return plot_file
            
        except Exception as e:
            self.logger.error(f"Error creating FUSE streamflow comparison plot: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
        """
        Creates visualization plots for each variable in the SUMMA output file.
        Each plot contains a map of the average value and a time series.
        
        Args:
            experiment_id (str): ID of the experiment to plot
            
        Returns:
            Dict[str, str]: Dictionary mapping variable names to their plot file paths
        """
        try:
            # Construct path to SUMMA output file
            summa_output_file = self.project_dir / "simulations" / experiment_id / "SUMMA" / f"{experiment_id}_day.nc"
            
            if not summa_output_file.exists():
                self.logger.error(f"SUMMA output file not found: {summa_output_file}")
                return {}
                
            # Create output directory
            plot_folder = self.project_dir / "plots" / "summa_outputs" / experiment_id
            plot_folder.mkdir(parents=True, exist_ok=True)
            
            # Load SUMMA output data
            ds = xr.open_dataset(summa_output_file)
            
            # Load HRU shapefile for spatial plotting
            hru_name = self.config.get('CATCHMENT_SHP_NAME')
            if hru_name == 'default':
                hru_name = f"{self.config.get('DOMAIN_NAME')}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"
            hru_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', hru_name)
            hru_gdf = gpd.read_file(hru_path)
            
            # Dictionary to store plot paths
            plot_files = {}
            
            # Skip these variables as they're typically dimensions or metadata
            skip_vars = {'hru', 'time', 'gru', 'dateId', 'latitude', 'longitude', 'hruId', 'gruId'}
            
            # Plot each variable
            for var_name in ds.data_vars:
                if var_name in skip_vars:
                    continue
                    
                try:
                    self.logger.info(f"Plotting variable: {var_name}")
                    
                    # Check if variable has time dimension
                    if 'time' not in ds[var_name].dims:
                        self.logger.info(f"Skipping {var_name} - no time dimension")
                        continue
                    
                    # Create figure with two subplots with adjusted heights
                    fig = plt.figure(figsize=(12, 12))
                    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
                    ax1 = fig.add_subplot(gs[0])
                    ax2 = fig.add_subplot(gs[1])
                    fig.suptitle(f'SUMMA Output: {var_name}', fontsize=16, fontweight='bold')
                    
                    # Calculate temporal mean for spatial plot using chunks
                    var_mean = ds[var_name].chunk({'time': -1, 'hru': 100}).mean(dim='time').compute()
                    
                    # Create GeoDataFrame with mean values
                    plot_gdf = hru_gdf.copy()
                    plot_gdf['value'] = var_mean.values
                    
                    # Reproject to Web Mercator for consistent plotting
                    plot_gdf = plot_gdf.to_crs(epsg=3857)
                    
                    # Calculate bounds with buffer
                    bounds = plot_gdf.total_bounds
                    buffer_x = (bounds[2] - bounds[0]) * 0.1
                    buffer_y = (bounds[3] - bounds[1]) * 0.1
                    
                    # Set map extent
                    ax1.set_xlim([bounds[0] - buffer_x, bounds[2] + buffer_x])
                    ax1.set_ylim([bounds[1] - buffer_y, bounds[3] + buffer_y])
                    
                    # Plot spatial distribution with simplified geometries
                    plot_gdf.geometry = plot_gdf.geometry.simplify(tolerance=100)
                    vmin, vmax = np.percentile(var_mean.values, [2, 98])
                    plot = plot_gdf.plot(column='value', 
                                ax=ax1,
                                legend=True,
                                legend_kwds={
                                    'label': f'Mean {var_name}',
                                    'orientation': 'horizontal',
                                    'fraction': 0.03,  # Make colorbar smaller
                                    'pad': 0.01,  # Reduce padding
                                    'aspect': 40  # Make colorbar thinner
                                },
                                vmin=vmin,
                                vmax=vmax,
                                cmap='RdYlBu')  # Changed colormap to Red-Yellow-Blue
                    
                    # Add HRU boundaries with simplified geometries
                    plot_gdf.boundary.plot(ax=ax1, color='black', linewidth=0.3, alpha=0.3)
                    
                    # Add title and remove axes
                    ax1.set_title(f'Spatial Distribution of Mean {var_name}', fontsize=12)
                    ax1.set_axis_off()
                    
                    self._add_north_arrow(ax1)
                    
                    # Plot time series using chunked computation
                    time_series = ds[var_name].chunk({'time': -1, 'hru': 100})
                    
                    # Calculate appropriate resampling frequency
                    n_points = len(time_series.time)
                    if n_points > 10000:
                        freq = 'W'
                    elif n_points > 1000:
                        freq = 'D'
                    else:
                        freq = None
                    
                    if freq:
                        time_series = time_series.resample(time=freq).mean()
                    
                    # Only compute and plot the mean across HRUs
                    mean_ts = time_series.mean(dim='hru').compute()
                    
                    # Further reduce points if still too many
                    if len(mean_ts) > 1000:
                        step = len(mean_ts) // 1000
                        mean_ts = mean_ts[::step]
                    
                    # Plot only the mean line
                    ax2.plot(mean_ts.time, mean_ts, 
                            color='#1f77b4',  # Use a nice blue color
                            linewidth=1.5,
                            label='Mean across HRUs')
                    
                    # Customize time series plot
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.set_ylabel(f'{var_name} {ds[var_name].units if hasattr(ds[var_name], "units") else ""}', 
                                fontsize=12)
                    ax2.set_title(f'Time Series of {var_name}', fontsize=12)
                    ax2.grid(True, linestyle=':', alpha=0.6)
                    
                    # Format x-axis to show years
                    ax2.xaxis.set_major_locator(mdates.YearLocator())
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    
                    # Add some basic statistics
                    stats_text = (f"Statistics:\n"
                                f"Mean: {float(mean_ts.mean()):.2f}\n"
                                f"Min: {float(mean_ts.min()):.2f}\n"
                                f"Max: {float(mean_ts.max()):.2f}")
                    ax2.text(0.02, 0.98, stats_text,
                            transform=ax2.transAxes,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Add title and remove axes
                    ax1.set_title(f'Spatial Distribution of Mean {var_name}', fontsize=12)
                    ax1.set_axis_off()
                    
                    self._add_north_arrow(ax1)
                
                    
                    # Customize time series plot
                    ax2.set_xlabel('Date', fontsize=12)
                    ax2.set_ylabel(f'{var_name} {ds[var_name].units if hasattr(ds[var_name], "units") else ""}', 
                                fontsize=12)
                    ax2.set_title(f'Time Series of {var_name}', fontsize=12)
                    ax2.grid(True, linestyle=':', alpha=0.6)
                    ax2.legend()
                    
                    # Format x-axis to show years
                    ax2.xaxis.set_major_locator(mdates.YearLocator())
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


                    ax2.text(0.02, 0.98, stats_text,
                            transform=ax2.transAxes,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Adjust layout and save
                    plt.tight_layout()
                    plot_file = plot_folder / f'{var_name}.png'
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plot_files[var_name] = str(plot_file)
                    
                except Exception as e:
                    self.logger.warning(f"Error plotting variable {var_name}: {str(e)}")
                    continue
            
            ds.close()
            return plot_files
            
        except Exception as e:
            self.logger.error(f"Error in plot_summa_outputs: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}