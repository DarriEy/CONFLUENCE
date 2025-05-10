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