import pandas as pd # type: ignore
import numpy as np # type: ignore
import sys
import csv
import subprocess
from hydrobm.calculate import calc_bm # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path 
from pyviscous import viscous # type: ignore
from SALib.analyze import sobol, rbd_fast # type: ignore
from SALib.sample import sobol as sobol_sample # type: ignore
from scipy.stats import spearmanr  # type: ignore
from tqdm import tqdm # type: ignore
import itertools
from typing import Dict, List, Tuple, Any
import xarray as xr # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from utils.model_utils import SummaRunner, MizuRouteRunner # type: ignore

class SensitivityAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "plots" / "sensitivity_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def read_calibration_results(self, file_path):
        df = pd.read_csv(file_path)
        return df.dropna()

    def preprocess_data(self, samples, metric='RMSE'):
        samples_unique = samples.drop_duplicates(subset=[col for col in samples.columns if col != 'Iteration'])
        return samples_unique

    def perform_sensitivity_analysis(self, samples, metric='Calib_KGEnp', min_samples=60):
        self.logger.info(f"Performing sensitivity analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'Calib_RMSE', 'Calib_KGE', 'Calib_KGEp', 'Calib_NSE', 'Calib_MAE']]
        
        if len(samples) < min_samples:
            self.logger.warning(f"Insufficient data for reliable sensitivity analysis. Have {len(samples)} samples, recommend at least {min_samples}.")
            return pd.Series([-999] * len(parameter_columns), index=parameter_columns)

        x = samples[parameter_columns].values
        y = samples[metric].values.reshape(-1, 1)
        
        sensitivities = []
        
        for i, param in tqdm(enumerate(parameter_columns), total=len(parameter_columns), desc="Calculating sensitivities"):
            try:
                try:
                    sensitivity_result = viscous(x, y, i, sensType='total')
                except ValueError:
                    sensitivity_result = viscous(x, y, i, sensType='single')
                
                if isinstance(sensitivity_result, tuple):
                    sensitivity = sensitivity_result[0]
                else:
                    sensitivity = sensitivity_result
                
                sensitivities.append(sensitivity)
                self.logger.info(f"Successfully calculated sensitivity for {param}")
            except Exception as e:
                self.logger.error(f"Error in sensitivity analysis for parameter {param}: {str(e)}")
                sensitivities.append(-999)
        
        self.logger.info("Sensitivity analysis completed")
        return pd.Series(sensitivities, index=parameter_columns)

    def perform_sobol_analysis(self, samples, metric='RMSE'):
        self.logger.info(f"Performing Sobol analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
        
        problem = {
            'num_vars': len(parameter_columns),
            'names': parameter_columns,
            'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
        }
        
        param_values = sobol_sample.sample(problem, 1024)
        
        Y = np.zeros(param_values.shape[0])
        for i in range(param_values.shape[0]):
            interpolated_values = []
            for j, col in enumerate(parameter_columns):
                interpolated_values.append(np.interp(param_values[i, j], 
                                                     samples[col].sort_values().values, 
                                                     samples[metric].values[samples[col].argsort()]))
            Y[i] = np.mean(interpolated_values)
        
        Si = sobol.analyze(problem, Y)

        self.logger.info("Sobol analysis completed")
        return pd.Series(Si['ST'], index=parameter_columns)

    def perform_rbd_fast_analysis(self, samples, metric='RMSE'):
        self.logger.info(f"Performing RBD-FAST analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
        
        problem = {
            'num_vars': len(parameter_columns),
            'names': parameter_columns,
            'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
        }
        
        X = samples[parameter_columns].values
        Y = samples[metric].values
        
        rbd_results = rbd_fast.analyze(problem, X, Y)
        self.logger.info("RBD-FAST analysis completed")
        return pd.Series(rbd_results['S1'], index=parameter_columns)

    def perform_correlation_analysis(self, samples, metric='RMSE'):
        self.logger.info(f"Performing correlation analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
        correlations = []
        for param in parameter_columns:
            corr, _ = spearmanr(samples[param], samples[metric])
            correlations.append(abs(corr))  # Use absolute value for sensitivity
        self.logger.info("Correlation analysis completed") 
        return pd.Series(correlations, index=parameter_columns)

    def plot_sensitivity(self, sensitivity, output_file):
        plt.figure(figsize=(10, 6))
        sensitivity.plot(kind='bar')
        plt.title("Parameter Sensitivity Analysis")
        plt.xlabel("Parameters")
        plt.ylabel("Sensitivity")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def plot_sensitivity_comparison(self, all_results, output_file):
        plt.figure(figsize=(12, 8))
        all_results.plot(kind='bar')
        plt.title("Sensitivity Analysis Comparison")
        plt.xlabel("Parameters")
        plt.ylabel("Sensitivity")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def run_sensitivity_analysis(self, results_file):
        self.logger.info("Starting sensitivity analysis")
        
        results = self.read_calibration_results(results_file)
        self.logger.info(f"Read {len(results)} calibration results")
        
        if len(results) < 10:
            self.logger.error("Error: Not enough data points for sensitivity analysis.")
            return
        
        results_preprocessed = self.preprocess_data(results, metric='Calib_RMSE')
        self.logger.info("Data preprocessing completed")

        methods = {
            'pyViscous': self.perform_sensitivity_analysis,
            'Sobol': self.perform_sobol_analysis,
            'RBD-FAST': self.perform_rbd_fast_analysis,
            'Correlation': self.perform_correlation_analysis
        }

        all_results = {}
        for name, method in methods.items():
            sensitivity = method(results_preprocessed, metric='Calib_RMSE')
            all_results[name] = sensitivity
            sensitivity.to_csv(self.output_folder / f'{name.lower()}_sensitivity.csv')
            self.plot_sensitivity(sensitivity, self.output_folder / f'{name.lower()}_sensitivity.png')
            self.logger.info(f"Saved {name} sensitivity results and plot")

        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(self.output_folder / 'all_sensitivity_results.csv')
        self.plot_sensitivity_comparison(comparison_df, self.output_folder / 'sensitivity_comparison.png')
        self.logger.info("Saved comparison of all sensitivity results")

        self.logger.info("Sensitivity analysis completed successfully")
        return comparison_df
    
class DecisionAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_folder = self.project_dir / "plots" / "decision_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.model_decisions_path = self.project_dir / "settings" / "SUMMA" / "modelDecisions.txt"

        # Initialize SummaRunner and MizuRouteRunner
        self.summa_runner = SummaRunner(config, logger)
        self.mizuroute_runner = MizuRouteRunner(config, logger)

        # Get decision options from config
        self.decision_options = self.config.get('DECISION_OPTIONS', {})

    def generate_combinations(self) -> List[Tuple[str, ...]]:
        return list(itertools.product(*self.decision_options.values()))

    def update_model_decisions(self, combination: Tuple[str, ...]):
        decision_keys = list(self.decision_options.keys())
        with open(self.model_decisions_path, 'r') as f:
            lines = f.readlines()
        
        option_map = dict(zip(decision_keys, combination))
        
        for i, line in enumerate(lines):
            for option, value in option_map.items():
                if line.strip().startswith(option):
                    lines[i] = f"{option.ljust(30)} {value.ljust(15)} ! {line.split('!')[-1].strip()}\n"
        
        with open(self.model_decisions_path, 'w') as f:
            f.writelines(lines)

    def run_models(self):
        self.summa_runner.run_summa()
        self.mizuroute_runner.run_mizuroute()

    def calculate_performance_metrics(self) -> Tuple[float, float, float, float, float]:
        obs_file_path = Path(self.config.get('OBSERVATIONS_PATH'))
        sim_reach_ID = self.config.get('SIM_REACH_ID')


        if self.config.get('SIMULATIONS_PATH') == 'default':
            sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute' / f'{self.config.get('EXPERIMENT_ID')}.h.{self.config.get('FORCING_START_YEAR')}-01-01-03600.nc'
        else:
            sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))

        dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        dfObs = dfObs['discharge_cms'].resample('h').mean()

        dfSim = xr.open_dataset(sim_file_path, engine='netcdf4')  
        segment_index = dfSim['reachID'].values == int(sim_reach_ID)
        dfSim = dfSim.sel(seg=segment_index) 
        dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
        dfSim.set_index('time', inplace=True)

        dfObs = dfObs.reindex(dfSim.index).dropna()
        dfSim = dfSim.reindex(dfObs.index).dropna()

        obs = dfObs.values
        sim = dfSim['IRFroutedRunoff'].values
        kge = get_KGE(obs, sim, transfo=1)
        kgep = get_KGEp(obs, sim, transfo=1)
        nse = get_NSE(obs, sim, transfo=1)
        mae = get_MAE(obs, sim, transfo=1)
        rmse = get_RMSE(obs, sim, transfo=1)

        return kge, kgep, nse, mae, rmse

    def run_decision_analysis(self):
        self.logger.info("Starting decision analysis")
        
        combinations = self.generate_combinations()

        master_file = self.project_dir / 'optimisation' / f"{self.config.get('EXPERIMENT_ID')}_model_decisions_comparison.csv"
        with open(master_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration'] + list(self.decision_options.keys()) + ['kge', 'kgep', 'nse', 'mae', 'rmse'])

        for i, combination in enumerate(combinations, 1):
            self.logger.info(f"Running combination {i} of {len(combinations)}")
            self.update_model_decisions(combination)
            
            try:
                self.run_models()
                kge, kgep, nse, mae, rmse = self.calculate_performance_metrics()

                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + [kge, kgep, nse, mae, rmse])

                self.logger.info(f"Combination {i} completed: KGE={kge:.3f}, KGEp={kgep:.3f}, NSE={nse:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

            except Exception as e:
                self.logger.error(f"Error in combination {i}: {str(e)}")
                with open(master_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i] + list(combination) + ['erroneous combination'])

        self.logger.info("Decision analysis completed")
        return master_file

    def plot_decision_impacts(self, results_file: Path):
        self.logger.info("Plotting decision impacts")
        
        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = [col for col in df.columns if col not in ['Iteration'] + metrics]

        for metric in metrics:
            plt.figure(figsize=(12, 6 * len(decisions)))
            for i, decision in enumerate(decisions, 1):
                plt.subplot(len(decisions), 1, i)
                impact = df.groupby(decision)[metric].mean().sort_values(ascending=False)
                impact.plot(kind='bar')
                plt.title(f'Impact of {decision} on {metric}')
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_folder / f'{metric}_decision_impacts.png')
            plt.close()

        self.logger.info("Decision impact plots saved")

    def analyze_results(self, results_file: Path):
        self.logger.info("Analyzing decision results")
        
        df = pd.read_csv(results_file)
        metrics = ['kge', 'kgep', 'nse', 'mae', 'rmse']
        decisions = [col for col in df.columns if col not in ['Iteration'] + metrics]

        best_combinations = {}
        for metric in metrics:
            if metric in ['mae', 'rmse']:
                best_row = df.loc[df[metric].idxmin()]
            else:
                best_row = df.loc[df[metric].idxmax()]
            
            best_combinations[metric] = {
                'score': best_row[metric],
                'combination': {decision: best_row[decision] for decision in decisions}
            }

        with open(self.project_dir / 'optimisation' / 'best_decision_combinations.txt', 'w') as f:
            for metric, data in best_combinations.items():
                f.write(f"Best combination for {metric} (score: {data['score']:.3f}):\n")
                for decision, value in data['combination'].items():
                    f.write(f"  {decision}: {value}\n")
                f.write("\n")

        self.logger.info("Decision analysis results saved")
        return best_combinations

    def run_full_analysis(self):
        results_file = self.run_decision_analysis()
        self.plot_decision_impacts(results_file)
        best_combinations = self.analyze_results(results_file)
        return results_file, best_combinations
    
class Benchmarker:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

    def run_benchmarking(self, input_data: pd.DataFrame, cal_end_date: str) -> Dict[str, Any]:
        """
        Run hydrobm benchmarking on the input data.
        
        Args:
            input_data (pd.DataFrame): DataFrame with date, temperature, streamflow, and precipitation.
            cal_end_date (str): End date for the calibration period (YYYY-MM-DD).
        
        Returns:
            Dict[str, Any]: Dictionary containing benchmark flows and scores.
        """
        self.logger.info("Starting hydrobm benchmarking")

        # Prepare data for hydrobm
        data = input_data.to_xarray()

        # Create calibration and validation masks
        cal_mask = data['index'].values < np.datetime64(cal_end_date)
        val_mask = ~cal_mask

        self.logger.info(f'input data: {data}')

        # Define benchmarks and metrics
        benchmarks = [
            # Streamflow benchmarks
            "mean_flow",
            "median_flow",
            "annual_mean_flow",
            "annual_median_flow",
            "monthly_mean_flow",
            "monthly_median_flow",
            "daily_mean_flow",
            "daily_median_flow",

            # Long-term rainfall-runoff ratio benchmarks
            "rainfall_runoff_ratio_to_all",
            "rainfall_runoff_ratio_to_annual",
            "rainfall_runoff_ratio_to_monthly",
            "rainfall_runoff_ratio_to_daily",
            "rainfall_runoff_ratio_to_timestep",

            # Short-term rainfall-runoff ratio benchmarks
            "monthly_rainfall_runoff_ratio_to_monthly",
            "monthly_rainfall_runoff_ratio_to_daily",
            "monthly_rainfall_runoff_ratio_to_timestep",

            # Schaefli & Gupta (2007) benchmarks
            "scaled_precipitation_benchmark",  # equivalent to "rainfall_runoff_ratio_to_daily"
            "adjusted_precipitation_benchmark",
            "adjusted_smoothed_precipitation_benchmark",
            ]
        
        metrics = ['nse', 'kge', 'mse', 'rmse']
        self.logger.info(f'Running benchmarks {benchmarks}')

        # Run hydrobm
        benchmark_flows, scores = calc_bm(
            data,
            cal_mask,
            val_mask=val_mask,
            precipitation="precipitation",
            streamflow="streamflow",
            benchmarks=benchmarks,
            metrics=metrics,
            calc_snowmelt=True,
            temperature="temperature",
            snowmelt_threshold=273.0,
            snowmelt_rate=3.0,
        )

        self.logger.info('Finished running benchmarks')

        # Save results
        self._save_results(benchmark_flows, scores)

        return {"benchmark_flows": benchmark_flows, "scores": scores}

    def _save_results(self, benchmark_flows: pd.DataFrame, scores: Dict[str, Any]):
        # Save benchmark flows
        flows_path = self.project_dir / 'evaluation' / "benchmark_flows.csv"
        benchmark_flows.to_csv(flows_path)
        self.logger.info(f"Benchmark flows saved to {flows_path}")

        # Save scores
        scores_df = pd.DataFrame(scores)
        scores_path = self.project_dir / 'evaluation' 
        scores_name = "benchmark_scores.csv"
        scores_df.to_csv(scores_path / scores_name)
        self.logger.info(f"Benchmark scores saved to {scores_path}")