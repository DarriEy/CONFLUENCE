import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path 
from pyviscous import viscous # type: ignore
from SALib.analyze import sobol, delta, rbd_fast # type: ignore
from SALib.sample import sobol as sobol_sample # type: ignore
from scipy.stats import spearmanr 
from tqdm import tqdm # type: ignore
from scipy import stats

class SensitivityAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_folder = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "plots" / "sensitivity_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def read_calibration_results(self, file_path):
        df = pd.read_csv(file_path)
        return df.dropna()

    def preprocess_data(self, samples, metric='RMSE'):
        samples_unique = samples.drop_duplicates(subset=[col for col in samples.columns if col != 'Iteration'])
        return samples_unique

    def perform_sensitivity_analysis(self, samples, metric='RMSE', min_samples=60):
        self.logger.info(f"Performing sensitivity analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
        
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

    def perform_delta_analysis(self, samples, metric='RMSE'):
        self.logger.info(f"Performing delta analysis using {metric} metric")
        parameter_columns = [col for col in samples.columns if col not in ['Iteration', 'RMSE', 'KGE', 'KGEp', 'NSE', 'MAE']]
        
        problem = {
            'num_vars': len(parameter_columns),
            'names': parameter_columns,
            'bounds': [[samples[col].min(), samples[col].max()] for col in parameter_columns]
        }
        
        X = samples[parameter_columns].values
        Y = samples[metric].values
        
        delta_results = delta.analyze(problem, X, Y)
        self.logger.info("Delta analysis completed")
        return pd.Series(delta_results['delta'], index=parameter_columns)

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
        
        results_preprocessed = self.preprocess_data(results, metric='RMSE')
        self.logger.info("Data preprocessing completed")

        methods = {
            'pyViscous': self.perform_sensitivity_analysis,
            'Sobol': self.perform_sobol_analysis,
            'Delta': self.perform_delta_analysis,
            'RBD-FAST': self.perform_rbd_fast_analysis,
            'Correlation': self.perform_correlation_analysis
        }

        all_results = {}
        for name, method in methods.items():
            sensitivity = method(results_preprocessed, metric='RMSE')
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
        self.output_folder = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "plots" / "decision_analysis"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def read_decision_results(self, file_path):
        return pd.read_csv(file_path)

    def analyze_decisions(self, data):
        self.logger.info("Starting decision analysis")
        
        # Identify decision columns and metric columns
        decision_columns = [col for col in data.columns if col not in ['Iteration', 'kge', 'kgep', 'nse', 'mae', 'rmse']]
        metric_columns = ['kge', 'kgep', 'nse', 'mae', 'rmse']

        results = {}

        for metric in metric_columns:
            metric_results = {}
            for decision in decision_columns:
                unique_options = data[decision].unique()
                if len(unique_options) > 1:
                    option_results = {option: data[data[decision] == option][metric].mean() for option in unique_options}
                    metric_results[decision] = option_results
            results[metric] = metric_results

        self.logger.info("Decision analysis completed")
        return results

    def plot_decision_impacts(self, results):
        self.logger.info("Plotting decision impacts")
        
        for metric, decisions in results.items():
            plt.figure(figsize=(12, 6 * len(decisions)))
            for i, (decision, options) in enumerate(decisions.items(), 1):
                plt.subplot(len(decisions), 1, i)
                options_sorted = sorted(options.items(), key=lambda x: x[1], reverse=True)
                labels, values = zip(*options_sorted)
                plt.bar(labels, values)
                plt.title(f'Impact of {decision} on {metric}')
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_folder / f'{metric}_decision_impacts.png')
            plt.close()

        self.logger.info("Decision impact plots saved")

    def perform_statistical_tests(self, data):
        self.logger.info("Performing statistical tests on decisions")
        
        decision_columns = [col for col in data.columns if col not in ['Iteration', 'kge', 'kgep', 'nse', 'mae', 'rmse']]
        metric_columns = ['kge', 'kgep', 'nse', 'mae', 'rmse']

        results = {}

        for metric in metric_columns:
            metric_results = {}
            for decision in decision_columns:
                unique_options = data[decision].unique()
                if len(unique_options) == 2:
                    # Perform t-test for two options
                    option1, option2 = unique_options
                    group1 = data[data[decision] == option1][metric]
                    group2 = data[data[decision] == option2][metric]
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    metric_results[decision] = {'test': 't-test', 't_statistic': t_stat, 'p_value': p_value}
                elif len(unique_options) > 2:
                    # Perform one-way ANOVA for more than two options
                    groups = [group[metric].values for name, group in data.groupby(decision)]
                    f_stat, p_value = stats.f_oneway(*groups)
                    metric_results[decision] = {'test': 'ANOVA', 'f_statistic': f_stat, 'p_value': p_value}
            
            results[metric] = metric_results

        self.logger.info("Statistical tests completed")
        return results

    def plot_statistical_results(self, results):
        self.logger.info("Plotting statistical test results")

        for metric, decisions in results.items():
            plt.figure(figsize=(10, 6))
            decisions_sorted = sorted(decisions.items(), key=lambda x: x[1]['p_value'])
            labels, p_values = zip(*[(decision, data['p_value']) for decision, data in decisions_sorted])
            
            plt.bar(labels, p_values)
            plt.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
            plt.title(f'Statistical Significance of Decisions on {metric}')
            plt.ylabel('p-value')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_folder / f'{metric}_statistical_significance.png')
            plt.close()

        self.logger.info("Statistical test result plots saved")

    def run_decision_analysis(self, results_file):
        self.logger.info("Starting decision analysis")
        
        data = self.read_decision_results(results_file)
        self.logger.info(f"Read {len(data)} decision results")

        if len(data) < 2:
            self.logger.error("Error: Not enough data points for decision analysis.")
            return

        impact_results = self.analyze_decisions(data)
        self.plot_decision_impacts(impact_results)

        statistical_results = self.perform_statistical_tests(data)
        self.plot_statistical_results(statistical_results)

        # Save results to CSV
        impact_df = pd.DataFrame(impact_results)
        impact_df.to_csv(self.output_folder / 'decision_impact_results.csv')

        statistical_df = pd.DataFrame({(metric, decision): data 
                                       for metric, decisions in statistical_results.items() 
                                       for decision, data in decisions.items()})
        statistical_df.to_csv(self.output_folder / 'decision_statistical_results.csv')

        self.logger.info("Decision analysis completed successfully")
        return impact_results, statistical_results