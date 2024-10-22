# results_utils.py
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
import seaborn as sns # type: ignore
import plotly.graph_objects as go # type: ignore
import math

class resultMapper:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

class timeseriesVizualiser:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

    def visualise_streamflow(self, streamflow_results: str, streamflow_obs: str, show_calib_eval_periods: bool = False):
        """
        Visualize streamflow simulation results against observations
        
        Parameters:
        -----------
        streamflow_results : str
            Path to the netCDF file containing simulation results
        streamflow_obs : str
            Path to the CSV file containing streamflow observations
        show_calib_eval_periods : bool, optional
            Whether to show calibration and evaluation periods in the plot
            
        Returns:
        --------
        str
            Path to the saved plot
        """
        try:
            plot_folder = self.project_dir / "plots" / "results"
            plot_folder.mkdir(parents=True, exist_ok=True)
            plot_filename = plot_folder / 'streamflow_comparison.png'

            # Read observation data
            obs_df = pd.read_csv(streamflow_obs, parse_dates=['datetime'])
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            obs_df.set_index('datetime', inplace=True)
            obs_df = obs_df['discharge_cms'].resample('h').mean()

            # Read simulation data
            ds = xr.open_dataset(streamflow_results, engine='netcdf4')
            basinID = int(self.config.get('SIM_REACH_ID'))  
            segment_index = ds['reachID'].values == basinID
            ds = ds.sel(seg=ds['seg'][segment_index])
            sim_df = ds['IRFroutedRunoff'].to_dataframe().reset_index()
            sim_df.set_index('time', inplace=True)

            # Determine common time range
            start_date = max(obs_df.index.min(), sim_df.index.min()) + pd.Timedelta(50, unit="d")
            end_date = min(obs_df.index.max(), sim_df.index.max())

            # Filter data to common time range
            obs_df = obs_df.loc[start_date:end_date]
            sim_df = sim_df.loc[start_date:end_date]

            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
            fig.suptitle("Streamflow Comparison and Flow Duration Curve", fontsize=16, fontweight='bold')

            # Plot time series
            ax1.plot(obs_df.index, obs_df, label='Observed', color='black', linewidth=2.5)
            ax1.plot(sim_df.index, sim_df['IRFroutedRunoff'], label='Simulated', 
                    color='#1f77b4', linestyle='--', linewidth=1.5)

            if show_calib_eval_periods:
                # Add calibration and evaluation periods
                calib_start = pd.Timestamp('2011-01-01')
                calib_end = pd.Timestamp('2014-12-31')
                eval_start = pd.Timestamp('2015-01-01')
                eval_end = pd.Timestamp('2018-12-31')

                # Calculate metrics for both periods
                for period, start_date, end_date in [
                    ('Calibration', calib_start, calib_end),
                    ('Evaluation', eval_start, eval_end)
                ]:
                    period_obs = obs_df.loc[start_date:end_date]
                    period_sim = sim_df.loc[start_date:end_date, 'IRFroutedRunoff']
                    metrics = self._calculate_metrics(period_obs.values, period_sim.values)
                    
                    metric_text = f"{period} Period ({start_date.year}-{end_date.year}):\n"
                    metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
                    
                    y_pos = 0.98 if period == 'Calibration' else 0.7
                    ax1.text(0.02, y_pos, metric_text, transform=ax1.transAxes, 
                            verticalalignment='top', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

                # Add shaded areas
                ax1.axvspan(calib_start, calib_end, alpha=0.2, color='gray', label='Calibration Period')
                ax1.axvspan(eval_start, eval_end, alpha=0.2, color='lightblue', label='Evaluation Period')
            
            else:
                # Calculate and display metrics for the entire period
                metrics = self._calculate_metrics(obs_df.values, sim_df['IRFroutedRunoff'].values)
                metric_text = "Overall Metrics:\n"
                metric_text += "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
                ax1.text(0.02, 0.98, metric_text, transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

            # Format axes
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax1.set_title('Streamflow Comparison', fontsize=14)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.set_facecolor('#f0f0f0')
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Plot flow duration curve
            self._plot_exceedance(ax2, obs_df.values, 'Observed', color='black', linewidth=2.5)
            self._plot_exceedance(ax2, sim_df['IRFroutedRunoff'].values, 'Simulated', 
                               color='#1f77b4', linestyle='--', linewidth=1.5)

            ax2.set_xlabel('Exceedance Probability', fontsize=12)
            ax2.set_ylabel('Streamflow (m³/s)', fontsize=12)
            ax2.set_title('Flow Duration Curve', fontsize=14)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, which='both', linestyle=':', alpha=0.6)
            ax2.set_facecolor('#f0f0f0')

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return str(plot_filename)

        except Exception as e:
            self.logger.error(f"Error in visualise_streamflow: {str(e)}")
            return None

    def _calculate_metrics(self, obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, transfo=1)
        }

    def _plot_exceedance(self, ax, data, label, color=None, linestyle='-', linewidth=1):
        """Plot flow duration curve"""
        sorted_data = np.sort(data)[::-1]
        exceedance = np.arange(1, len(data) + 1) / len(data)
        ax.plot(exceedance, sorted_data, label=label, color=color, 
                linestyle=linestyle, linewidth=linewidth)
        ax.set_xscale('log')
        ax.set_yscale('log')