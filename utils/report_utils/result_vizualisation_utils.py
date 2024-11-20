# results_utils.py
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
import seaborn as sns # type: ignore
import plotly.graph_objects as go # type: ignore
import math
import matplotlib.gridspec as gridspec # type: ignore
from matplotlib.colors import LinearSegmentedColormap # type: ignore
import matplotlib.dates as mdates # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.evaluation_util.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore

class resultMapper:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

class TimeseriesVisualizer:
    """Visualizes simulated and observed streamflow timeseries."""
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.plots_dir = self.project_dir / "plots" / "results"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Extend the colors list to match the number of possible models
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Define model names and ensure they match your data columns
        self.model_names = ['FUSE', 'GR', 'FLASH', 'HYPE', 'SUMMA']  # Update this list to match your actual model names

    def read_results(self) -> pd.DataFrame:
        """Read simulation results and observed streamflow."""
        try:
            # Read simulation results
            results_file = self.project_dir / "results" / f"{self.config['EXPERIMENT_ID']}_results.csv"
            if not results_file.exists():
                raise FileNotFoundError(f"Results file not found: {results_file}")
            
            # Read the CSV file and handle the time column
            sim_df = pd.read_csv(results_file)
            print(f'{sim_df}')
            
            # Check if the DataFrame is empty
            if sim_df.empty:
                self.logger.error("Results file is empty")
                raise ValueError("Results file contains no data")
            
            # Convert time column to datetime and set as index
            if 'time' in sim_df.columns:
                sim_df['time'] = pd.to_datetime(sim_df['time'])
                sim_df.set_index('time', inplace=True)
            elif 'datetime' in sim_df.columns:
                sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
                sim_df.set_index('datetime', inplace=True)
            elif 'Unnamed: 0' in sim_df.columns:
                # Handle case where dates are in 'Unnamed: 0' column
                sim_df['datetime'] = pd.to_datetime(sim_df['Unnamed: 0'])
                sim_df.set_index('datetime', inplace=True)
                sim_df.drop('Unnamed: 0', axis=1, errors='ignore')  # Remove the column if it wasn't used as index
            elif sim_df.index.name is None and isinstance(sim_df.index, pd.Index):
                # Try to convert the unnamed index to datetime
                try:
                    sim_df.index = pd.to_datetime(sim_df.index)
                except (ValueError, TypeError):
                    raise ValueError("Index cannot be converted to datetime format")
            else:
                raise ValueError("No time or datetime column found in results file")
            
            print(f'{sim_df}')
            # Check if the DataFrame is empty
            if sim_df.empty:
                self.logger.error("Results file is empty")
                raise ValueError("Results file contains no data")
            
            # Round to nearest day
            #sim_df.set_index('time', inplace=True)
            print(f'{sim_df}')
               
            # Read observations
            obs_file_path = self.config.get('OBSERVATIONS_PATH')
            if obs_file_path == 'default':
                obs_file_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                raise FileNotFoundError(f"Observations file not found: {obs_file_path}")

            obs_df = pd.read_csv(obs_file_path)
            
            # Check if observations DataFrame is empty
            if obs_df.empty:
                self.logger.error("Observations file is empty")
                raise ValueError("Observations file contains no data")
            
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'], format='mixed', dayfirst=True)
            obs_df.set_index('datetime', inplace=True)
            obs_df = obs_df['discharge_cms'].resample('D').mean()
            
            # Combine into single dataframe
            results_df = sim_df.copy()
            print(f'{results_df}')
            print(f'{obs_df}')
            results_df['Observed'] = obs_df
            # Check if we have any data after combining
            if results_df.empty:
                raise ValueError("No data available after combining simulations and observations")
            
            # Skip first year (spin-up period)
            if len(results_df.index) > 0:  # Only if we have data
                first_year = results_df.index[0].year
                start_date = pd.Timestamp(year=first_year + 1, month=1, day=1)
                results_df = results_df[results_df.index >= start_date]
                
                if results_df.empty:
                    raise ValueError("No data available after removing spin-up period")
            
            self.logger.info(f"Data period: {results_df.index[0]} to {results_df.index[-1]}")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Error reading results: {str(e)}")
            raise

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics for each model."""
        metrics = []
        obs = df['Observed'].values
        
        # Only process models that actually exist in the data
        available_models = [model for model in self.model_names 
                           if f"{model}_discharge_cms" in df.columns]
        
        for model in available_models:
            col = f"{model}_discharge_cms"
            if col in df.columns:
                sim = df[col].values
                # Make sure to align and handle missing values
                valid = ~(np.isnan(obs) | np.isnan(sim))
                if valid.sum() > 0:
                    kge = get_KGE(obs[valid], sim[valid], transfo=1)
                    kgep = get_KGEp(obs[valid], sim[valid], transfo=1)
                    nse = get_NSE(obs[valid], sim[valid], transfo=1)
                    mae = get_MAE(obs[valid], sim[valid], transfo=1)
                    rmse = get_RMSE(obs[valid], sim[valid], transfo=1)
                    
                    metrics.append({
                        'Model': model,
                        'KGE': kge,
                        'KGEp': kgep,
                        'NSE': nse,
                        'MAE': mae,
                        'RMSE': rmse
                    })
        
        return pd.DataFrame(metrics)

    def calculate_flow_duration_curve(self, flows: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate flow duration curve from a series of flows."""
        sorted_flows = np.sort(flows.dropna().values)[::-1]
        exceedance_prob = np.arange(1, len(sorted_flows) + 1) / (len(sorted_flows) + 1) * 100
        return exceedance_prob, sorted_flows

    def plot_timeseries(self, df: pd.DataFrame):
        """Create timeseries plot of all available model results."""
        plt.style.use('default')
        
        # Create single figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Find overlapping time period
        available_models = [model for model in self.model_names 
                          if f"{model}_discharge_cms" in df.columns]
        
        # Start with observed data's valid range
        mask = ~df['Observed'].isna()
        
        # Add model data validity to mask
        for model in available_models:
            col = f"{model}_discharge_cms"
            mask &= ~df[col].isna()
        
        # Apply mask to get overlapping period
        df_overlap = df[mask]
        
        if df_overlap.empty:
            self.logger.warning("No overlapping time period found between models and observations")
            return
            
        # Skip first year for spinup
        if len(df_overlap.index) > 0:
            first_year = df_overlap.index[0].year
            start_date = pd.Timestamp(year=first_year + 1, month=1, day=1)
            df_overlap = df_overlap[df_overlap.index >= start_date]
            
            if df_overlap.empty:
                self.logger.warning("No data available after removing spinup period")
                return
        
        # Calculate metrics for the overlap period
        metrics_df = self.calculate_metrics(df_overlap)
        
        # Plot each model result with KGE in legend first (behind observed data)
        for i, model in enumerate(available_models):
            col = f"{model}_discharge_cms"
            color_idx = i % len(self.colors)
            
            # Get KGE value for this model
            kge = metrics_df.loc[metrics_df['Model'] == model, 'KGE'].values[0]
            
            # Create label with KGE value
            label = f'{model} (KGE: {kge:.3f})'
            
            df_overlap[col].plot(ax=ax, label=label, alpha=0.4, linewidth=1, color=self.colors[color_idx])
        
        # Plot observed data last (on top) with enhanced visibility
        df_overlap['Observed'].plot(ax=ax, 
                                  color='black', 
                                  label='Observed', 
                                  alpha=1.0,  # Full opacity
                                  linewidth=1.5,  # Slightly thicker line
                                  linestyle='-',  # Solid line
                                  zorder=5)  # Ensure it's on top
        
        # Customize plot
        ax.set_ylabel('Discharge (m³/s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Adjusted to prevent overlap
        ax.set_title(f'Streamflow Comparison - {self.domain_name}\n(excluding first year for spinup)')
        ax.set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.plots_dir / f'{self.config["EXPERIMENT_ID"]}_timeseries_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_diagnostics(self, df: pd.DataFrame):
        """Create diagnostic plots including scatter plots and flow duration curves."""
        # Get models from config
        config_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        available_models = [model.strip() for model in config_models 
                          if f"{model.strip()}_discharge_cms" in df.columns]
        
        if not available_models:
            self.logger.warning("No models found in the data")
            return
            
        n_models = len(available_models)
        
        # Adjust figure size based on number of models
        fig_height = max(5, min(4 * n_models, 20))  # Cap maximum height at 20
        plt.style.use('default')
        fig = plt.figure(figsize=(15, fig_height))
        
        # Create grid for subplots
        gs = plt.GridSpec(n_models, 2, figure=fig)
        
        # Start with observed data's valid range
        mask = ~df['Observed'].isna()
        
        # Add model data validity to mask
        for model in available_models:
            col = f"{model}_discharge_cms"
            mask &= ~df[col].isna()
        
        # Apply mask to get overlapping period
        df_overlap = df[mask]
        
        if df_overlap.empty:
            self.logger.warning("No overlapping time period found between models and observations")
            return
            
        # Skip first year for spinup
        if len(df_overlap.index) > 0:
            first_year = df_overlap.index[0].year
            start_date = pd.Timestamp(year=first_year + 1, month=1, day=1)
            df_overlap = df_overlap[df_overlap.index >= start_date]
            
            if df_overlap.empty:
                self.logger.warning("No data available after removing spinup period")
                return
        
        # Calculate FDC for observed flows
        exc_prob_obs, flows_obs = self.calculate_flow_duration_curve(df_overlap['Observed'])
        
        # Plot each model
        metrics_df = self.calculate_metrics(df_overlap)  # Use overlapping period for metrics
        
        for i, model in enumerate(available_models):
            col = f"{model}_discharge_cms"
            
            # Scatter plot
            ax_scatter = fig.add_subplot(gs[i, 0])
            color_idx = i % len(self.colors)
            
            # Create scatter plot with hexbin for dense datasets
            if len(df_overlap) > 1000:
                hb = ax_scatter.hexbin(df_overlap['Observed'], df_overlap[col], 
                                     gridsize=30, cmap='YlOrRd', 
                                     bins='log')
                fig.colorbar(hb, ax=ax_scatter, label='Count')
            else:
                ax_scatter.scatter(df_overlap['Observed'], df_overlap[col], 
                                 alpha=0.5, s=10, color=self.colors[color_idx])
            
            # Add 1:1 line
            max_val = max(df_overlap['Observed'].max(), df_overlap[col].max())
            min_val = min(df_overlap['Observed'].min(), df_overlap[col].min())
            lims = [min_val, max_val]
            ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
            
            # Add metrics
            model_metrics = metrics_df[metrics_df['Model'] == model].iloc[0]
            metrics_text = (f'KGE: {model_metrics["KGE"]:.3f}\n'
                          f'NSE: {model_metrics["NSE"]:.3f}\n'
                          f'RMSE: {model_metrics["RMSE"]:.3f}')
            ax_scatter.text(0.05, 0.95, metrics_text, 
                          transform=ax_scatter.transAxes,
                          verticalalignment='top', 
                          bbox=dict(boxstyle='round', 
                                  facecolor='white', 
                                  alpha=0.8))
            
            # Format scatter plot
            ax_scatter.set_xlabel('Observed (m³/s)')
            ax_scatter.set_ylabel('Simulated (m³/s)')
            ax_scatter.set_title(f'{model} - Scatter Plot')
            ax_scatter.grid(True, alpha=0.3)
            
            # Flow Duration Curve
            ax_fdc = fig.add_subplot(gs[i, 1])
            exc_prob_sim, flows_sim = self.calculate_flow_duration_curve(df_overlap[col])
            
            ax_fdc.plot(exc_prob_obs, flows_obs, 'k-', label='Observed', alpha=0.6)
            ax_fdc.plot(exc_prob_sim, flows_sim, '-', 
                       color=self.colors[color_idx], label=model, alpha=0.6)
            
            # Format FDC plot
            ax_fdc.set_yscale('log')
            ax_fdc.set_xlabel('Exceedance Probability (%)')
            ax_fdc.set_ylabel('Discharge (m³/s)')
            ax_fdc.set_title(f'{model} - Flow Duration Curve')
            ax_fdc.grid(True, alpha=0.3)
            ax_fdc.legend()
        
        # Add overall title
        fig.suptitle(f'Diagnostic Plots - {self.domain_name}', 
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f'{self.config["EXPERIMENT_ID"]}_diagnostic_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to: {plot_path}")

    def create_visualizations(self):
        """Create all visualizations."""
        try:
            # Read data
            df = self.read_results()
            print(df)
            # Calculate metrics
            metrics_df = self.calculate_metrics(df)
            metrics_df.to_csv(self.plots_dir / f'{self.config["EXPERIMENT_ID"]}_performance_metrics.csv', index=False)
            
            # Create plots
            self.plot_timeseries(df)
            self.plot_diagnostics(df)
            
            self.logger.info(f"Visualizations saved to: {self.plots_dir}")
            
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
            raise

class BenchmarkVizualiser:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        
        # Define metrics and their properties
        self.metrics = {
            'nse': {
                'name': 'NSE',
                'description': 'Nash-Sutcliffe Efficiency',
                'range': [-np.inf, 1]
            },
            'kge': {
                'name': 'KGE',
                'description': 'Kling-Gupta Efficiency',
                'range': [-np.inf, 1]
            },
            'rmse': {
                'name': 'RMSE',
                'description': 'Root Mean Square Error',
                'range': [0, np.inf]
            }
        }
        
        # Refined benchmark groups with clearer organization
        self.benchmark_groups = {
            'Time-invariant': {
                'benchmarks': [
                    'bm_mean_flow',
                    'bm_median_flow',
                    'bm_annual_mean_flow',
                    'bm_annual_median_flow'
                ],
                'color': '#1f77b4',
                'style_cycle': ['-', '--', '-.', ':'],
                'description': 'Constant and annual benchmarks'
            },
            'Time-variant': {
                'benchmarks': [
                    'bm_monthly_mean_flow',
                    'bm_monthly_median_flow',
                    'bm_daily_mean_flow',
                    'bm_daily_median_flow'
                ],
                'color': '#2ca02c',
                'style_cycle': ['-', '--', '-.', ':'],
                'description': 'Monthly and daily benchmarks'
            },
            'Rainfall-Runoff': {
                'benchmarks': [
                    'bm_rainfall_runoff_ratio_to_all',
                    'bm_rainfall_runoff_ratio_to_annual',
                    'bm_rainfall_runoff_ratio_to_monthly',
                    'bm_rainfall_runoff_ratio_to_daily',
                    'bm_rainfall_runoff_ratio_to_timestep',
                    'bm_monthly_rainfall_runoff_ratio_to_monthly',
                    'bm_monthly_rainfall_runoff_ratio_to_daily',
                    'bm_monthly_rainfall_runoff_ratio_to_timestep'
                ],
                'color': '#ff7f0e',
                'style_cycle': ['-', '--', '-.', ':', '-', '--', '-.', ':'],
                'description': 'Precipitation-based benchmarks'
            },
            'Advanced': {
                'benchmarks': [
                    'bm_scaled_precipitation_benchmark',
                    'bm_adjusted_precipitation_benchmark',
                    'bm_adjusted_smoothed_precipitation_benchmark'
                ],
                'color': '#9467bd',
                'style_cycle': ['-', '--', '-.'],
                'description': 'Schaefli & Gupta benchmarks'
            }
        }


    def visualize_benchmarks(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Create three separate benchmark visualization figures."""
        try:
            # Create plot directory
            plot_folder = self.project_dir / "plots" / "benchmarks"
            plot_folder.mkdir(parents=True, exist_ok=True)
            
            # Extract and prepare data
            flows = benchmark_results['benchmark_flows']
            scores = pd.DataFrame(benchmark_results['scores'])
            flows, scores = self.prepare_benchmark_data(flows, scores)
            
            # Generate three different figures
            plot_paths = []
            
            # Figure 1: Benchmark Comparison by Group
            plot_paths.append(self._create_group_comparison_figure(flows, scores, plot_folder))
            
            # Figure 2: Statistical Summary and Performance Metrics
            plot_paths.append(self._create_statistics_figure(flows, scores, plot_folder))
            
            # Figure 3: Envelope Analysis
            plot_paths.append(self._create_envelope_figure(flows, scores, plot_folder))
            
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"Error in visualize_benchmarks: {str(e)}")
            raise

    def load_observed_data(self) -> pd.DataFrame:
        """Load and process observed streamflow data."""
        try:
            obs_path = (self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                       f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv")
            
            if not obs_path.exists():
                self.logger.warning(f"Observed data file not found: {obs_path}")
                return pd.DataFrame()
                
            obs_df = pd.read_csv(obs_path, parse_dates=['datetime'])
            obs_df.set_index('datetime', inplace=True)
            
            # Ensure we have the discharge column
            if 'discharge_cms' not in obs_df.columns:
                self.logger.warning("No 'discharge_cms' column found in observed data")
                return pd.DataFrame()
                
            return obs_df[['discharge_cms']].copy()
            
        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return pd.DataFrame()

    def prepare_benchmark_data(self, flows: pd.DataFrame, scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare benchmark data for visualization."""
        # Add observed data
        obs_df = self.load_observed_data()
        if not obs_df.empty:
            self.logger.info("Successfully loaded observed data")
            # Ensure the observed data matches the benchmark timeframe
            obs_df = obs_df.reindex(flows.index)
            flows['observed'] = obs_df['discharge_cms']
        else:
            self.logger.warning("No observed data available")
        
        # Process scores and ensure all benchmarks are prefixed with 'bm_'
        scores = scores.copy()
        scores.index = scores['benchmarks'].apply(lambda x: f"bm_{x}" if not x.startswith('bm_') else x)
        
        # Log available benchmarks
        self.logger.info(f"Available benchmarks in flows: {flows.columns.tolist()}")
        self.logger.info(f"Available benchmarks in scores: {scores.index.tolist()}")
        
        return flows, scores

    
    def _plot_group_stats(self, ax: plt.Axes, scores: pd.DataFrame, benchmarks: List[str]) -> None:
        """Plot performance statistics table with refined professional formatting."""
        # Remove existing axis elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Create table data starting with headers
        table_data = [['Benchmark', 'Metric', 'Cal', 'Val']]
        
        # Add data for each benchmark
        for benchmark in benchmarks:
            if benchmark in scores.index:
                # Simplified benchmark names
                name = (benchmark.replace('bm_', '')
                                .replace('rainfall_runoff_ratio_to_', 'RR ratio: ')
                                .replace('monthly_rainfall_runoff_ratio_to_', 'monthly RR: ')
                                .replace('adjusted_smoothed_precipitation_benchmark', 'adjusted smoothed precip.')
                                .replace('adjusted_precipitation_benchmark', 'adjusted precip.')
                                .replace('scaled_precipitation_benchmark', 'scaled precip.')
                                .replace('_', ' '))
                
                # Format the values with proper handling of NaN
                nse_cal = f"{scores.loc[benchmark, 'nse_cal']:.3f}" if not pd.isna(scores.loc[benchmark, 'nse_cal']) else 'nan'
                nse_val = f"{scores.loc[benchmark, 'nse_val']:.3f}" if not pd.isna(scores.loc[benchmark, 'nse_val']) else 'nan'
                kge_cal = f"{scores.loc[benchmark, 'kge_cal']:.3f}" if not pd.isna(scores.loc[benchmark, 'kge_cal']) else 'nan'
                kge_val = f"{scores.loc[benchmark, 'kge_val']:.3f}" if not pd.isna(scores.loc[benchmark, 'kge_val']) else 'nan'
                rmse_cal = f"{scores.loc[benchmark, 'rmse_cal']:.1f}" if not pd.isna(scores.loc[benchmark, 'rmse_cal']) else 'nan'
                rmse_val = f"{scores.loc[benchmark, 'rmse_val']:.1f}" if not pd.isna(scores.loc[benchmark, 'rmse_val']) else 'nan'
                
                metrics_data = [
                    [name, 'NSE', nse_cal, nse_val],
                    ['', 'KGE', kge_cal, kge_val],
                    ['', 'RMSE', rmse_cal, rmse_val]
                ]
                table_data.extend(metrics_data)

        # Create table
        table = ax.table(cellText=table_data,
                        loc='center',
                        cellLoc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)  
        
        # Format header row
        for i in range(len(table_data[0])):
            cell = table[0, i]
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f8f9fa')
        
        # Set column widths
        col_widths = [0.46, 0.18, 0.18, 0.18]  
        for i, width in enumerate(col_widths):
            for row in range(len(table_data)):
                cell = table[row, i]
                cell.set_width(width)
                
                # Text alignment
                if i == 0 and row > 0:  # Benchmark names
                    cell._text.set_horizontalalignment('left')
                else:  # Metrics and values
                    cell._text.set_horizontalalignment('center')
                
                # Cell styling
                cell.set_height(0.12)  
                cell.PAD = 0.05 
                
                # Set edges
                edges = 'BTRL' if row == 0 else ''  
                cell.visible_edges = edges
        
        # Add subtle horizontal lines between benchmarks
        for row in range(1, len(table_data), 3):
            for col in range(len(table_data[0])):
                cell = table[row, col]
                cell.visible_edges = 'T'
                cell.set_edgecolor('#dddddd')

        # Add title with more padding
        ax.set_title('Performance Metrics', pad=15, fontsize=10)
        
        # Move table up slightly
        table.set_transform(ax.transAxes)
        table._bbox = [0, -0.05, 1, 1]  # Adjust vertical position


    def _create_group_comparison_figure(self, flows: pd.DataFrame, scores: pd.DataFrame, plot_folder: Path) -> str:
        """Create improved figure showing benchmark comparisons by group."""
        n_groups = len(self.benchmark_groups)
        fig = plt.figure(figsize=(15, 4 * n_groups))
        
        # Create grid with extra space for title
        gs = gridspec.GridSpec(n_groups + 1, 2, height_ratios=[0.2] + [1] * n_groups, width_ratios=[3, 1])
        
        # Add main title
        fig.text(0.5, 0.98, 'Benchmark Performance Analysis', 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Add domain info
        fig.text(0.02, 0.98, f'Domain: {self.config.get("DOMAIN_NAME")}', 
                ha='left', va='top', fontsize=10, style='italic')
        
        # Add metric descriptions
        metric_desc = '\n'.join([f"{m['name']}: {m['description']}" 
                               for m in self.metrics.values()])
        fig.text(0.98, 0.98, metric_desc, 
                ha='right', va='top', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        for idx, (group_name, group_info) in enumerate(self.benchmark_groups.items()):
            # Time series plot
            ax_main = fig.add_subplot(gs[idx + 1, 0])
            
            # Plot observed data first if available
            if 'observed' in flows.columns:
                ax_main.plot(flows.index, flows['observed'],
                           label='Observed', color='black', linewidth=1.5, zorder=10)
            
            # Plot benchmarks for this group
            valid_benchmarks = [b for b in group_info['benchmarks'] if b in flows.columns]
            for benchmark, style in zip(valid_benchmarks, group_info['style_cycle']):
                ax_main.plot(flows.index, flows[benchmark],
                           label=benchmark.replace('bm_', '').replace('_', ' '),
                           color=group_info['color'],
                           linestyle=style,
                           alpha=0.7)
            
            # Enhance subplot formatting
            ax_main.set_ylabel('Streamflow (m³/s)')
            title = f'{group_name} Benchmarks\n{group_info["description"]}'
            ax_main.set_title(title, pad=10, fontsize=12)
            ax_main.grid(True, alpha=0.3)
            
            # Improved legend
            leg = ax_main.legend(bbox_to_anchor=(1.01, 1), loc='upper left', 
                               fontsize=8, frameon=True)
            leg.get_frame().set_alpha(0.9)
            
            # Format x-axis
            ax_main.xaxis.set_major_locator(mdates.YearLocator())
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            # Only show x-label for bottom plot
            if idx == len(self.benchmark_groups) - 1:
                ax_main.set_xlabel('Date')
            
            # Stats panel with table
            ax_stats = fig.add_subplot(gs[idx + 1, 1])
            self._plot_group_stats(ax_stats, scores, valid_benchmarks)
            ax_stats.set_title('Performance Metrics' if idx == 0 else '')
        
        plt.tight_layout()
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.95, right=0.85, hspace=0.4)
        
        # Save plot
        plot_path = plot_folder / f"{self.config.get('DOMAIN_NAME')}_benchmark_groups.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


    def _create_statistics_figure(self, flows: pd.DataFrame, scores: pd.DataFrame, plot_folder: Path) -> str:
        """Create figure showing statistical summaries and performance metrics."""
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2)
        
        # Performance metrics heatmap
        ax_heatmap = fig.add_subplot(gs[0, :])
        self._plot_metrics_heatmap(ax_heatmap, scores)
        
        # Statistical summaries
        ax_stats1 = fig.add_subplot(gs[1, 0])
        self._plot_statistical_summary(ax_stats1, flows, 'mean_std')
        
        ax_stats2 = fig.add_subplot(gs[1, 1])
        self._plot_statistical_summary(ax_stats2, flows, 'percentiles')
        
        plt.tight_layout()
        plot_path = plot_folder / f"{self.config.get('DOMAIN_NAME')}_benchmark_statistics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)

    def _create_envelope_figure(self, flows: pd.DataFrame, scores: pd.DataFrame, plot_folder: Path) -> str:
        """Create envelope figure using top 5 benchmarks by KGE."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Get top 5 benchmarks by KGE
        kge_scores = scores['kge_cal'].sort_values(ascending=False)
        top_benchmarks = kge_scores.head(5).index
        
        # Plot time series with envelope
        if 'observed' in flows.columns:
            ax1.plot(flows.index, flows['observed'],
                    label='Observed', color='black', linewidth=1.5, zorder=10)
        
        # Calculate envelope from top benchmarks
        benchmark_data = flows[[b for b in top_benchmarks if b in flows.columns]]
        min_flow = benchmark_data.min(axis=1)
        max_flow = benchmark_data.max(axis=1)
        
        # Plot envelope
        ax1.fill_between(flows.index, min_flow, max_flow,
                        alpha=0.3, color='#2196f3',
                        label='Top 5 Benchmark Range')
        
        # Format time series plot
        ax1.set_ylabel('Streamflow (m³/s)')
        ax1.set_title('Benchmark Envelope (Top 5 by KGE)', pad=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Plot FDC with envelope
        def calculate_fdc(data):
            sorted_data = np.sort(data.dropna())[::-1]
            exceedance = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            return exceedance, sorted_data
        
        # Plot observed FDC
        if 'observed' in flows.columns:
            exc_obs, obs = calculate_fdc(flows['observed'])
            ax2.plot(exc_obs, obs, label='Observed', color='black', linewidth=1.5)
        
        # Calculate and plot FDC envelope
        all_fdcs = []
        for benchmark in top_benchmarks:
            if benchmark in flows.columns:
                exc, values = calculate_fdc(flows[benchmark])
                # Interpolate to common exceedance points
                exceedance = np.linspace(0, 1, num=1000)
                interpolated = np.interp(exceedance, exc, values)
                all_fdcs.append(interpolated)
        
        all_fdcs = np.array(all_fdcs)
        min_fdc = np.min(all_fdcs, axis=0)
        max_fdc = np.max(all_fdcs, axis=0)
        
        ax2.fill_between(exceedance, min_fdc, max_fdc,
                        alpha=0.3, color='#2196f3',
                        label='Top 5 Benchmark Range')
        
        # Format FDC plot
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Exceedance Probability')
        ax2.set_ylabel('Streamflow (m³/s)')
        ax2.set_title('Flow Duration Curves', pad=10)
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
        
        # Add annotation with top benchmarks
        benchmark_text = "Top 5 Benchmarks by KGE:\n" + "\n".join([
            f"{i+1}. {b.replace('bm_', '').replace('_', ' ')} (KGE: {scores.loc[b, 'kge_cal']:.3f})"
            for i, b in enumerate(top_benchmarks)
        ])
        fig.text(0.02, 0.02, benchmark_text, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = plot_folder / f"{self.config.get('DOMAIN_NAME')}_benchmark_envelopes.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


    def _plot_metrics_heatmap(self, ax: plt.Axes, scores: pd.DataFrame) -> None:
        """Plot performance metrics heatmap with enhanced styling."""
        metrics_data = []
        for group_name, group_info in self.benchmark_groups.items():
            for benchmark in group_info['benchmarks']:
                if benchmark in scores.index:
                    row = scores.loc[benchmark]
                    metrics_data.append({
                        'Benchmark': benchmark.replace('bm_', '').replace('_', ' '),
                        'Group': group_name,
                        **{m.upper(): row[f'{m}_cal'] for m in self.metrics}
                    })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.set_index(['Group', 'Benchmark'])[
            [m.upper() for m in self.metrics]]
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f',
                   cmap='RdYlBu_r', center=0, ax=ax,
                   cbar_kws={'label': 'Score'})
        ax.set_title('Performance Metrics')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_statistical_summary(self, ax: plt.Axes, flows: pd.DataFrame,
                                summary_type: str) -> None:
        """Plot statistical summary with different options."""
        if summary_type == 'mean_std':
            stats = []
            for group_name, group_info in self.benchmark_groups.items():
                for benchmark in group_info['benchmarks']:
                    if benchmark in flows.columns:
                        data = flows[benchmark].dropna()
                        stats.append({
                            'Benchmark': benchmark.replace('bm_', '').replace('_', ' '),
                            'Mean': data.mean(),
                            'Std': data.std(),
                            'CV': data.std() / data.mean()
                        })
            
            stats_df = pd.DataFrame(stats)
            stats_df.plot(x='Benchmark', y=['Mean', 'Std', 'CV'],
                         kind='bar', ax=ax, rot=45)
            ax.set_title('Basic Statistics')
            
        elif summary_type == 'percentiles':
            stats = []
            for group_name, group_info in self.benchmark_groups.items():
                for benchmark in group_info['benchmarks']:
                    if benchmark in flows.columns:
                        data = flows[benchmark].dropna()
                        stats.append({
                            'Benchmark': benchmark.replace('bm_', '').replace('_', ' '),
                            'Q25': data.quantile(0.25),
                            'Q50': data.quantile(0.50),
                            'Q75': data.quantile(0.75)
                        })
            
            stats_df = pd.DataFrame(stats)
            stats_df.plot(x='Benchmark', y=['Q25', 'Q50', 'Q75'],
                         kind='bar', ax=ax, rot=45)
            ax.set_title('Flow Percentiles')
        
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')