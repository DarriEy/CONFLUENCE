import sys
import xarray as xr # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.dates as mdates # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore
from utils.logging_utils import get_function_logger # type: ignore

class VisualizationReporter:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

    def plot_streamflow_simulations_vs_observations(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]):
        try:
            # Read observation data
            obs_data = []
            for obs_name, obs_file in obs_files:
                df = pd.read_csv(obs_file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                df = df['discharge_cms'].resample('h').mean()
                obs_data.append((obs_name, df))

            # Read simulation data
            sim_data = []
            for sim_name, sim_file in model_outputs:
                ds = xr.open_dataset(sim_file, engine='netcdf4')
                basinID = int(self.config.get('SIM_REACH_ID'))  
                segment_index = ds['reachID'].values == basinID
                ds = ds.sel(seg=ds['seg'][segment_index])
                df = ds['IRFroutedRunoff'].to_dataframe().reset_index()
                df.set_index('time', inplace=True)
                sim_data.append((sim_name, df))

            print(sim_data)
            # Determine common time range
            start_date = max([data.index.min() for _, data in obs_data + sim_data]) + pd.Timedelta(50, unit="d")
            end_date = min([data.index.max() for _, data in obs_data + sim_data])

            # Filter data to common time range
            obs_data = [(name, data.loc[start_date:end_date]) for name, data in obs_data]
            sim_data = [(name, data.loc[start_date:end_date]) for name, data in sim_data]

            # Define calibration and evaluation periods
            calib_start = pd.Timestamp('2011-01-01')
            calib_end = pd.Timestamp('2014-12-31')
            eval_start = pd.Timestamp('2015-01-01')
            eval_end = pd.Timestamp('2018-12-31')

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.suptitle("Streamflow Comparison and Flow Duration Curve", fontsize=16, fontweight='bold')

            # Plot time series
            for obs_name, obs in obs_data:
                ax1.plot(obs.index, obs, label=f'Observed ({obs_name})', color='black', linewidth=2.5)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            linestyles = ['--', '-.', ':']
            for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                ax1.plot(sim.index, sim['IRFroutedRunoff'], label=f'Simulated ({sim_name})', 
                         color=color, linestyle=linestyle, linewidth=1.5)
                
                # Calculate and display metrics for calibration and evaluation periods
                calib_metrics = self.calculate_metrics(
                    obs_data[0][1].loc[calib_start:calib_end].values,
                    sim.loc[calib_start:calib_end, 'IRFroutedRunoff'].values
                )
                eval_metrics = self.calculate_metrics(
                    obs_data[0][1].loc[eval_start:eval_end].values,
                    sim.loc[eval_start:eval_end, 'IRFroutedRunoff'].values
                )
                
                metric_text = f"{sim_name} Metrics:\nCalibration (2011-2014):\n"
                metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in calib_metrics.items()])
                metric_text += "\n\nEvaluation (2016-2018):\n"
                metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in eval_metrics.items()])
                
                # Add semi-transparent background to text
                ax1.text(0.02, 0.98 - 0.35 * sim_data.index((sim_name, sim)), metric_text,
                         transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

            # Add shaded areas for calibration and evaluation periods
            ax1.axvspan(calib_start, calib_end, alpha=0.2, color='gray', label='Calibration Period')
            ax1.axvspan(eval_start, eval_end, alpha=0.2, color='lightblue', label='Evaluation Period')

            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax1.set_title('Streamflow Comparison', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.set_facecolor('#f0f0f0')  # Light gray background

            # Format x-axis to show years
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Plot exceedance frequency
            for obs_name, obs in obs_data:
                self.plot_exceedance(ax2, obs.values, f'Observed ({obs_name})', color='black', linewidth=2.5)

            for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                self.plot_exceedance(ax2, sim['IRFroutedRunoff'].values, f'Simulated ({sim_name})', 
                                color=color, linestyle=linestyle, linewidth=1.5)

            ax2.set_xlabel('Exceedance Probability', fontsize=12)
            ax2.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, which='both', linestyle=':', alpha=0.6)
            ax2.set_facecolor('#f0f0f0')  # Light gray background

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Adjust for main title

            # Save the plot
            plot_folder = self.project_dir / "plots" / "results"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'streamflow_comparison.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_filename)

        except Exception as e:
            print(f"Error in plot_streamflow_simulations_vs_observations: {str(e)}")
            return None
        
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

    def update_sim_reach_id(self):
        try:
            # Load the pour point shapefile
            pour_point_path = self._get_file_path('POUR_POINT_SHP_PATH', 'shapefiles/pour_point', self.config.get('POUR_POINT_SHP_NAME'))
            pour_point_gdf = gpd.read_file(pour_point_path)

            # Load the river network shapefile
            river_network_path = self._get_file_path('RIVER_NETWORK_PATH', 'shapefiles/river_network', self.config.get('RIVER_NETWORK_SHP_NAME'))
            river_network_gdf = gpd.read_file(river_network_path)

            # Ensure both GeoDataFrames have the same CRS
            if pour_point_gdf.crs != river_network_gdf.crs:
                pour_point_gdf = pour_point_gdf.to_crs(river_network_gdf.crs)

            # Get the pour point coordinates
            pour_point = pour_point_gdf.geometry.iloc[0]

            # Find the nearest stream segment to the pour point
            nearest_segment = river_network_gdf.iloc[river_network_gdf.distance(pour_point).idxmin()]

            # Get the ID of the nearest segment
            reach_id = nearest_segment[self.config.get('RIVER_NETWORK_SHP_SEGID', 'COMID')]  # Use 'COMID' as default if not specified

            # Update the config
            self.config['SIM_REACH_ID'] = str(reach_id)
            self.logger.info(f"Updated SIM_REACH_ID to {reach_id}")

            return reach_id

        except Exception as e:
            self.logger.error(f"Error updating SIM_REACH_ID: {str(e)}")
            return None

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
    
    @get_function_logger
    def plot_lumped_streamflow_simulations_vs_observations(self, model_outputs: List[Tuple[str, str]], obs_files: List[Tuple[str, str]]):
        try:
            # Read observation data
            obs_data = []
            for obs_name, obs_file in obs_files:
                df = pd.read_csv(obs_file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                df = df['discharge_cms'].resample('h').mean()
                obs_data.append((obs_name, df))

            # Read simulation data
            sim_data = []
            for sim_name, sim_file in model_outputs:
                ds = xr.open_dataset(sim_file, engine='netcdf4')
                df = ds['averageRoutedRunoff'].to_dataframe().reset_index()
                df.set_index('time', inplace=True)
                sim_data.append((sim_name, df))

            # Determine common time range
            start_date = max([data.index.min() for _, data in obs_data + sim_data]) + pd.Timedelta(50, unit="d")
            end_date = min([data.index.max() for _, data in obs_data + sim_data])

            # Filter data to common time range
            obs_data = [(name, data.loc[start_date:end_date]) for name, data in obs_data]
            sim_data = [(name, data.loc[start_date:end_date]) for name, data in sim_data]

            # Define calibration and evaluation periods
            calib_start = pd.Timestamp('2011-01-01')
            calib_end = pd.Timestamp('2014-12-31')
            eval_start = pd.Timestamp('2015-01-01')
            eval_end = pd.Timestamp('2018-12-31')

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.suptitle("Lumped Watershed Streamflow Comparison and Flow Duration Curve", fontsize=16, fontweight='bold')

            # Plot time series
            for obs_name, obs in obs_data:
                ax1.plot(obs.index, obs, label=f'Observed ({obs_name})', color='black', linewidth=2.5)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            linestyles = ['--', '-.', ':']
            for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                ax1.plot(sim.index, sim['averageRoutedRunoff'], label=f'Simulated ({sim_name})', 
                         color=color, linestyle=linestyle, linewidth=1.5)
                
                # Calculate and display metrics for calibration and evaluation periods
                calib_metrics = self.calculate_metrics(
                    obs_data[0][1].loc[calib_start:calib_end].values,
                    sim.loc[calib_start:calib_end, 'averageRoutedRunoff'].values
                )
                eval_metrics = self.calculate_metrics(
                    obs_data[0][1].loc[eval_start:eval_end].values,
                    sim.loc[eval_start:eval_end, 'averageRoutedRunoff'].values
                )
                
                metric_text = f"{sim_name} Metrics:\nCalibration (2011-2014):\n"
                metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in calib_metrics.items()])
                metric_text += "\n\nEvaluation (2015-2018):\n"
                metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in eval_metrics.items()])
                
                # Add semi-transparent background to text
                ax1.text(0.02, 0.98 - 0.35 * sim_data.index((sim_name, sim)), metric_text,
                         transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

            # Add shaded areas for calibration and evaluation periods
            ax1.axvspan(calib_start, calib_end, alpha=0.2, color='gray', label='Calibration Period')
            ax1.axvspan(eval_start, eval_end, alpha=0.2, color='lightblue', label='Evaluation Period')

            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax1.set_title('Lumped Watershed Streamflow Comparison', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.set_facecolor('#f0f0f0')  # Light gray background

            # Format x-axis to show years
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Plot exceedance frequency
            for obs_name, obs in obs_data:
                self.plot_exceedance(ax2, obs.values, f'Observed ({obs_name})', color='black', linewidth=2.5)

            for (sim_name, sim), color, linestyle in zip(sim_data, colors, linestyles):
                self.plot_exceedance(ax2, sim['averageRoutedRunoff'].values, f'Simulated ({sim_name})', 
                                color=color, linestyle=linestyle, linewidth=1.5)

            ax2.set_xlabel('Exceedance Probability', fontsize=12)
            ax2.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, which='both', linestyle=':', alpha=0.6)
            ax2.set_facecolor('#f0f0f0')  # Light gray background

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Adjust for main title

            # Save the plot
            plot_folder = self.project_dir / "plots" / "results"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'lumped_streamflow_comparison.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_filename)

        except Exception as e:
            self.logger.error(f"Error in plot_lumped_streamflow_simulations_vs_observations: {str(e)}")
            return None