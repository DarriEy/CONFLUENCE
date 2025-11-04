import pandas as pd # type: ignore
import sys
import csv
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path 
import itertools
from typing import List, Tuple
import xarray as xr # type: ignore

from utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from utils.models.summa_utils import SummaRunner # type: ignore
from utils.models.mizuroute_utils import MizuRouteRunner # type: ignore

    
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
        self.decision_options = self.config.get('SUMMA_DECISION_OPTIONS', {})

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
        obs_file_path = self.config.get('OBSERVATIONS_PATH')
        if obs_file_path == 'default':
            obs_file_path = self.project_dir / 'observations'/ 'streamflow' / 'preprocessed' / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            obs_file_path = Path(obs_file_path)

        sim_reach_ID = self.config.get('SIM_REACH_ID')

        if self.config.get('SIMULATIONS_PATH') == 'default':
            sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute' / f"{self.config['EXPERIMENT_ID']}.h.{self.config.get('EXPERIMENT_TIME_START').split('-')[0]}-01-01-03600.nc"
        else:
            sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))

        dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        dfObs = dfObs['discharge_cms'].resample('h').mean()

        dfSim = xr.open_dataset(sim_file_path, engine='netcdf4')  
        segment_index = dfSim['reachID'].values == int(sim_reach_ID)
        dfSim = dfSim.sel(seg=segment_index) 
        dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
        dfSim.set_index('time', inplace=True)
        dfSim.index = dfSim.index.round(freq='h')

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

        optimisation_dir = self.project_dir / 'optimisation'
        optimisation_dir.mkdir(parents=True, exist_ok=True)

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