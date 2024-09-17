import subprocess
from pathlib import Path
import sys
import json
import os
import xarray as xr # type: ignore
import pandas as pd # type: ignore
import csv
from typing import List, Tuple, Dict, Any, Optional
import logging
import numpy as np # type: ignore
import shutil
from mpi4py import MPI # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore
from utils.logging_utils import setup_logger # type: ignore
from utils.optimization_config import Config # type: ignore


def run_summa(summa_path, summa_exe, filemanager_path, log_path, log_name, local_rank):
    """
    Run SUMMA model.

    Parameters:
    summa_path (str): Path to SUMMA executable directory
    summa_exe (str): Name of SUMMA executable
    filemanager_path (Path): Path to SUMMA file manager
    log_path (Path): Path to store log files
    log_name (str): Name of the log file
    local_rank (int): Rank of the current process

    Returns:
    int: Return code of the SUMMA process
    """
    # Ensure log directory exists
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Construct SUMMA command
    summa_command = [str(Path(summa_path) / summa_exe), "-m", str(filemanager_path)]

    # Run SUMMA
    with open(log_path / log_name, 'w') as log_file:
        summa_result = subprocess.run(summa_command, check=False, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    return summa_result.returncode

def run_mizuroute(mizuroute_path, mizuroute_exe, control_path, log_path, log_name, local_rank):
    """
    Run mizuRoute model.

    Parameters:
    mizuroute_path (str): Path to mizuRoute executable directory
    mizuroute_exe (str): Name of mizuRoute executable
    control_path (Path): Path to mizuRoute control file
    log_path (Path): Path to store log files
    log_name (str): Name of the log file
    local_rank (int): Rank of the current process

    Returns:
    int: Return code of the mizuRoute process
    """
    # Ensure log directory exists
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Construct mizuRoute command
    mizuroute_command = [str(Path(mizuroute_path) / mizuroute_exe), str(control_path)]

    # Run mizuRoute
    with open(log_path / log_name, 'w') as log_file:
        mizuroute_result = subprocess.run(mizuroute_command, check=False, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    return mizuroute_result.returncode



def prepare_model_run(root_path, domain_name, experiment_id, rank_experiment_id, local_rank, 
                      local_param_values, basin_param_values, params_to_calibrate, basin_params_to_calibrate,
                      local_bounds_dict, basin_bounds_dict, filemanager_name, mizu_control_file):
    """
    Prepare the model run by setting up directories and updating configuration files.

    Parameters:
    root_path (str): Root path of the project
    domain_name (str): Name of the domain
    experiment_id (str): ID of the experiment
    rank_experiment_id (str): Rank-specific experiment ID
    local_rank (int): Rank of the current process
    local_param_values (list): Local parameter values
    basin_param_values (list): Basin parameter values
    params_to_calibrate (list): Names of local parameters to calibrate
    basin_params_to_calibrate (list): Names of basin parameters to calibrate
    local_bounds_dict (dict): Dictionary of local parameter bounds
    basin_bounds_dict (dict): Dictionary of basin parameter bounds
    filemanager_name (str): Name of the SUMMA file manager file
    mizu_control_file (str): Name of the mizuRoute control file

    Returns:
    tuple: Paths to rank-specific directories and settings
    """
    rank_specific_path = create_rank_specific_directory(root_path, domain_name, rank_experiment_id, local_rank, "SUMMA")
    mizuroute_rank_specific_path = create_rank_specific_directory(root_path, domain_name, rank_experiment_id, local_rank, "mizuRoute")
    
    # Copy SUMMA and mizuRoute settings to rank-specific directory
    summa_source_settings_path = Path(root_path) / f"domain_{domain_name}" / "settings" / "SUMMA"
    summa_destination_settings_path = rank_specific_path / "run_settings"
    mizuroute_source_settings_path = Path(root_path) / f"domain_{domain_name}" / "settings" / "mizuRoute"
    mizuroute_destination_settings_path = mizuroute_rank_specific_path / "run_settings"
    copy_mizuroute_settings(mizuroute_source_settings_path, mizuroute_destination_settings_path)
    copy_summa_settings(summa_source_settings_path, summa_destination_settings_path)

    # Update parameter files
    update_param_files(local_param_values, basin_param_values, 
                       params_to_calibrate, basin_params_to_calibrate, 
                       summa_destination_settings_path / "localParamInfo.txt", 
                       summa_destination_settings_path / "basinParamInfo.txt",
                       local_bounds_dict, basin_bounds_dict)
    
    # Update file manager
    update_file_manager(summa_destination_settings_path / filemanager_name, rank_experiment_id, experiment_id)
    update_mizu_control_file(mizuroute_destination_settings_path / mizu_control_file, rank_experiment_id, experiment_id)

    return rank_specific_path, mizuroute_rank_specific_path, summa_destination_settings_path, mizuroute_destination_settings_path


def write_failed_run_info(root_path, domain_name, experiment_id, local_rank, attempt, params, summa_log_path, summa_log_name):
    """
    Write information about a failed SUMMA run to a special file with a unique, sequentially numbered name.

    Parameters:
    root_path (str): Root path of the project
    domain_name (str): Name of the domain
    experiment_id (str): ID of the experiment
    local_rank (int): Rank of the current process
    attempt (int): Attempt number of the failed run
    params (dict): Dictionary of parameter names and values
    summa_log_path (Path): Path to the SUMMA log file
    summa_log_name (str): Name of the SUMMA log file
    """
    failed_runs_dir = Path(root_path) / f"domain_{domain_name}" / "optimisation" / "failed_runs"
    failed_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the current highest number for failed run files
    existing_files = [f for f in os.listdir(failed_runs_dir) if f.startswith(f"{experiment_id}_rank{local_rank}_attempt{attempt}_failed_run")]
    current_highest = 0
    for file in existing_files:
        try:
            num = int(file.split('_')[-1].split('.')[0])
            current_highest = max(current_highest, num)
        except ValueError:
            pass
    
    # Create a new filename with the next number
    new_file_number = current_highest + 1
    failed_run_file = failed_runs_dir / f"{experiment_id}_rank{local_rank}_attempt{attempt}_failed_run_{new_file_number:03d}.txt"
    
    with open(failed_run_file, 'w') as f:
        f.write(f"Failed run information for {experiment_id}, rank {local_rank}, attempt {attempt}\n\n")
        f.write("Parameter values:\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\nSUMMA log:\n")
        with open(summa_log_path / summa_log_name, 'r') as log_file:
            f.write(log_file.read())

def evaluate_model(mizuroute_output_path, calib_period, eval_period, sim_reach_ID, obs_file_path):
    """
    Evaluate the model by comparing simulations to observations.

    Parameters:
    mizuroute_rank_specific_path (Path): Path to mizuRoute output
    calib_period (tuple): Start and end dates for calibration period
    eval_period (tuple): Start and end dates for evaluation period
    sim_reach_ID (str): ID of the simulated reach to evaluate
    obs_file_path (str): Path to the observation file

    Returns:
    tuple: Calibration and evaluation metrics
    """

    # Open the mizuRoute output file
    #sim_file_path = str(mizuroute_rank_specific_path) + '/*.nc'
    dfSim = xr.open_dataset(mizuroute_output_path)         
    segment_index = dfSim['reachID'].values == int(sim_reach_ID)
    dfSim = dfSim.sel(seg=segment_index) 
    dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
    dfSim.set_index('time', inplace=True)

    dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
    dfObs = dfObs['discharge_cms'].resample('h').mean()

    def calculate_metrics(obs, sim):
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, tranfo = 1)
        }

    calib_start, calib_end = calib_period
    calib_obs = dfObs.loc[calib_start:calib_end]
    calib_sim = dfSim.loc[calib_start:calib_end]
    calib_metrics = calculate_metrics(calib_obs.values, calib_sim['IRFroutedRunoff'].values)

    eval_start, eval_end = eval_period
    eval_obs = dfObs.loc[eval_start:eval_end]
    eval_sim = dfSim.loc[eval_start:eval_end]
    eval_metrics = calculate_metrics(eval_obs.values, eval_sim['IRFroutedRunoff'].values)

    return calib_metrics, eval_metrics

def create_rank_specific_directory(root_path, domain_name, rank_experiment_id, rank, model):
    rank_specific_path = Path(root_path) / f"domain_{domain_name}" / "simulations" / rank_experiment_id / model
    
    # Create directories
    rank_specific_path.mkdir(parents=True, exist_ok=True)
    (rank_specific_path / "run_settings").mkdir(exist_ok=True)
    (rank_specific_path / f"{model}_logs").mkdir(exist_ok=True)

    return rank_specific_path

def copy_summa_settings(source_settings_path, destination_settings_path):
    for file in ['attributes.nc', 'trialParams.nc', 'forcingFileList.txt', 'outputControl.txt', 'modelDecisions.txt', 'localParamInfo.txt', 'basinParamInfo.txt', 'fileManager.txt','coldState.nc', 'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL']:
        shutil.copy(source_settings_path / file, destination_settings_path / file)

def copy_mizuroute_settings(source_settings_path, destination_settings_path):
    for file in ['mizuroute.control', 'param.nml.default', 'topology.nc']:
        shutil.copy(source_settings_path / file, destination_settings_path / file)

def update_file_manager(file_path, rank_experiment_id, experiment_id):
    base_settings = "/settings/SUMMA/' !"
    rank_settings = f"/simulations/{rank_experiment_id}/SUMMA/run_settings/' !"
    with open(file_path, 'r') as f: 
        content = f.read()
    
    content = content.replace(experiment_id, rank_experiment_id)
    content = content.replace(base_settings, rank_settings)

    with open(file_path, 'w') as f:
        f.write(content)

def update_mizu_control_file(file_path, rank_experiment_id, experiment_id):
    base_settings = "/settings/mizuRoute/"
    rank_settings = f"/simulations/{rank_experiment_id}/mizuRoute/run_settings/"
    with open(file_path, 'r') as f:
        content = f.read()
    
    content = content.replace(experiment_id, rank_experiment_id)
    content = content.replace(base_settings, rank_settings)

    with open(file_path, 'w') as f:
        f.write(content)

def create_iteration_results_file(experiment_id, root_path, domain_name, all_params):
    """
    Create a new iteration results file with appropriate headers.
    """
    iteration_results_dir = Path(root_path) / f"domain_{domain_name}" / "optimisation"
    iteration_results_dir.mkdir(parents=True, exist_ok=True)
    iteration_results_file = iteration_results_dir / f"{experiment_id}_parallel_iteration_results.csv"
    
    # We'll create the file, but not write the header yet
    # The header will be written in the first call to write_iteration_results
    iteration_results_file.touch()
    
    return str(iteration_results_file)

def write_iteration_results(file_path, iteration, param_names, rank_params, calib_metrics, eval_metrics):
    """
    Write iteration results to the CSV file.
    """
    logger = logging.getLogger('write_iteration_results')
    logger.info(f"Writing results for iteration {iteration}")
    logger.info(f"Calib metrics: {calib_metrics}")
    logger.info(f"Eval metrics: {eval_metrics}")

    try:
        all_calib_keys = set().union(*[m for m in calib_metrics if m])
        all_eval_keys = set().union(*[m for m in eval_metrics if m])
        
        fieldnames = ['Iteration', 'Rank'] + param_names + \
                     [f'Calib_{k}' for k in sorted(all_calib_keys)] + \
                     [f'Eval_{k}' for k in sorted(all_eval_keys)]
        
        file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for rank, (params, calib, eval) in enumerate(zip(rank_params, calib_metrics, eval_metrics)):
                row = {'Iteration': iteration, 'Rank': rank}
                row.update(dict(zip(param_names, params)))
                row.update({f'Calib_{k}': calib.get(k, np.nan) for k in sorted(all_calib_keys)})
                row.update({f'Eval_{k}': eval.get(k, np.nan) for k in sorted(all_eval_keys)})
                writer.writerow(row)
        
        logger.info(f"Results written successfully for iteration {iteration}")
    except Exception as e:
        logger.error(f"Error writing results for iteration {iteration}: {str(e)}", exc_info=True)
        raise

def read_param_bounds(param_file, params_to_calibrate):
    """Read parameter bounds from the parameter file."""
    bounds = {}
    with open(param_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4 and parts[0].strip() in params_to_calibrate:
                param = parts[0].strip()
                lower = float(parts[2].strip().replace('d', 'e'))
                upper = float(parts[3].strip().replace('d', 'e'))
                bounds[param] = (lower, upper)
    return bounds

def update_param_files(local_param_values, basin_param_values, local_param_keys, basin_param_keys, local_param_file, basin_param_file, local_bounds_dict, basin_bounds_dict):
    def update_file(param_values, param_keys, file_path, bounds_dict):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        param_map = dict(zip(param_keys, param_values))
        
        for i, line in enumerate(lines):
            parts = line.split('|')
            if len(parts) >= 4 and parts[0].strip() in param_map:
                param = parts[0].strip()
                value = param_map[param]
                lower_bound, upper_bound = bounds_dict[param]
                
                # Ensure the value is within bounds
                value = max(lower_bound, min(upper_bound, value))
                
                # Format the value with 'd' for scientific notation
                value_str = f"{value:.6e}".replace('e', 'd')
                # Keep the parameter name unchanged
                lines[i] = f"{parts[0]}| {value_str} |{parts[2]}|{parts[3]}"

        with open(file_path, 'w') as f:
            f.writelines(lines)

    update_file(local_param_values, local_param_keys, local_param_file, local_bounds_dict)
    update_file(basin_param_values, basin_param_keys, basin_param_file, basin_bounds_dict)

def are_params_valid(params: List[float], param_bounds: List[Tuple[float, float]]) -> bool:
    return all(lower <= value <= upper for value, (lower, upper) in zip(params, param_bounds))


def read_summa_error(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                return lines[-3].strip()  # Return the last 3 lines, stripped of leading/trailing whitespace
            else:
                return "SUMMA log file is empty."
    except Exception as e:
        return f"Error reading SUMMA log file: {str(e)}"

class ModelRunner:
    """
    Initialize the ModelRunner.
    
    Args:
        config (Config): Configuration object containing model settings.
        comm (MPI.Comm): MPI communicator for parallel processing.
        rank (int): Rank of the current process.
    """
    def __init__(self, config: Config, comm: MPI.Comm, rank: int):
        self.config = config
        self.comm = comm
        self.rank = rank

        log_dir = Path(config.root_path) / f'domain_{config.domain_name}' / f'_workLog_{config.domain_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_optimization_{config.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_optimization', log_file)

    def run_models(self, local_param_values: list, basin_param_values: list) -> Optional[Path]:
        """
        Run SUMMA and mizuRoute models with given parameter values.
        
        Args:
            local_param_values (list): Values for local parameters.
            basin_param_values (list): Values for basin parameters.
        
        Returns:
            Optional[Path]: Path to mizuRoute output if successful, None otherwise.
        """
        max_retries = 50
        original_local_params = local_param_values.copy()
        original_basin_params = basin_param_values.copy()

        for attempt in range(max_retries):
            try:
                # Generate new parameter values for retry attempts
                if attempt > 0:
                    local_param_values = self.generate_new_params(original_local_params, self.config.local_bounds)
                    basin_param_values = self.generate_new_params(original_basin_params, self.config.basin_bounds)
                    self.logger.info(f"Retry {attempt} with new parameters: local={local_param_values}, basin={basin_param_values}")

                # Prepare model run directories and settings
                rank_experiment_id = f"{self.config.experiment_id}_rank{self.rank}"
                rank_specific_path, mizuroute_rank_specific_path, summa_destination_settings_path, mizuroute_destination_settings_path = prepare_model_run(
                    self.config.root_path, self.config.domain_name, self.config.experiment_id, rank_experiment_id, self.rank,
                    local_param_values, basin_param_values, self.config.params_to_calibrate, self.config.basin_params_to_calibrate,
                    self.config.local_bounds_dict, self.config.basin_bounds_dict, self.config.filemanager_name, self.config.mizu_control_file
                )

                # Update parameter files with new values
                update_param_files(
                    local_param_values, 
                    basin_param_values, 
                    self.config.params_to_calibrate, 
                    self.config.basin_params_to_calibrate,
                    summa_destination_settings_path / "localParamInfo.txt",
                    summa_destination_settings_path / "basinParamInfo.txt",
                    self.config.local_bounds_dict,
                    self.config.basin_bounds_dict
                )
                
                self.logger.info(f"Rank {self.rank} starting model run attempt {attempt + 1}")
                self.logger.info(f"Rank {self.rank} prepared model run. SUMMA path: {rank_specific_path}, mizuRoute path: {mizuroute_rank_specific_path}")
                
                # Run SUMMA model
                summa_success = self.run_summa(rank_specific_path, summa_destination_settings_path, attempt, local_param_values, basin_param_values)
                if not summa_success:
                    continue
                
                # Run mizuRoute model
                mizuroute_success = self.run_mizuroute(mizuroute_rank_specific_path, mizuroute_destination_settings_path, attempt)
                if not mizuroute_success:
                    continue
                
                self.logger.info(f"Rank {self.rank} completed model run successfully")
                return mizuroute_rank_specific_path

            except Exception as e:
                self.logger.error(f"Error in model run for rank {self.rank}, attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Max retries reached for rank {self.rank}. Returning None.")
                    return None

        return None

    def generate_new_params(self, original_params: list, bounds: list) -> list:
        """
        Generate new parameter values within bounds, but different from original values.
        
        Args:
            original_params (list): Original parameter values.
            bounds (list): List of (lower, upper) bounds for each parameter.
        
        Returns:
            list: New parameter values.
        """
        new_params = []
        for param, (lower, upper) in zip(original_params, bounds):
            while True:
                new_value = np.random.uniform(lower, upper)
                if abs(new_value - param) > (upper - lower) * 0.01:  # Ensure at least 1% difference
                    new_params.append(new_value)
                    break
        return new_params

    def run_summa(self, rank_specific_path: Path, summa_destination_settings_path: Path, 
                  attempt: int, local_param_values: list, basin_param_values: list) -> bool:
        """
        Run the SUMMA model.
        
        Args:
            rank_specific_path (Path): Path specific to this rank's run.
            summa_destination_settings_path (Path): Path to SUMMA settings.
            attempt (int): Current attempt number.
            local_param_values (list): Local parameter values.
            basin_param_values (list): Basin parameter values.
        
        Returns:
            bool: True if SUMMA run was successful, False otherwise.
        """
        summa_path = Path(f'{self.config.root_path}/installs/summa/bin/')
        summa_exe = 'summa.exe' #read_from_control(Path('../../0_control_files/control_active.txt'), 'exe_name_summa')
        
        filemanager_path = summa_destination_settings_path / self.config.filemanager_name
        summa_log_path = rank_specific_path / "SUMMA_logs"
        summa_log_name = f"summa_log_attempt{attempt}.txt"
        
        summa_return_code = run_summa(summa_path, summa_exe, filemanager_path, summa_log_path, summa_log_name, self.rank)
        
        if summa_return_code != 0:
            all_param_values = dict(zip(self.config.params_to_calibrate, local_param_values))
            all_param_values.update(dict(zip(self.config.basin_params_to_calibrate, basin_param_values)))
            write_failed_run_info(self.config.root_path, self.config.domain_name, self.config.experiment_id, self.rank, attempt, 
                                    all_param_values, summa_log_path, summa_log_name)
            self.logger.error(f"SUMMA run failed with return code: {summa_return_code}")
            return False
        
        self.logger.info(f"SUMMA run completed successfully for rank {self.rank}")
        return True

    def run_mizuroute(self, mizuroute_rank_specific_path: Path, mizuroute_destination_settings_path: Path, attempt: int) -> bool:
        
        """
        Run the mizuRoute model.
        
        Args:
            mizuroute_rank_specific_path (Path): Path specific to this rank's mizuRoute run.
            mizuroute_destination_settings_path (Path): Path to mizuRoute settings.
            attempt (int): Current attempt number.
        
        Returns:
            bool: True if mizuRoute run was successful, False otherwise.
        """

        mizuroute_path = Path(f'{self.config.root_path}/installs/mizuRoute/route/bin/')
        mizuroute_exe = 'mizuroute.exe' #read_from_control(Path('../../0_control_files/control_active.txt'), 'exe_name_mizuroute')
        mizuroute_control_path = mizuroute_destination_settings_path / self.config.mizu_control_file
        mizuroute_log_path = mizuroute_rank_specific_path / "mizuroute_logs"
        mizuroute_log_name = f"mizuroute_log_attempt{attempt}.txt"
        
        mizuroute_return_code = run_mizuroute(mizuroute_path, mizuroute_exe, mizuroute_control_path, mizuroute_log_path, mizuroute_log_name, self.rank)
        
        if mizuroute_return_code != 0:
            self.logger.error(f"mizuRoute run failed with return code: {mizuroute_return_code}")
            return False
        
        self.logger.info(f"mizuRoute run completed successfully for rank {self.rank}")
        return True

    def split_params(self, params: List[float]) -> Tuple[List[float], List[float]]:
        """
        Split the input parameters into local and basin parameters.

        Args:
            params (List[float]): List of all parameters.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing two lists:
                                             (local parameters, basin parameters)
        """
        n_local = len(self.config.params_to_calibrate)
        return params[:n_local], params[n_local:]

    def run_MESH(self, MESH_rank_specific_path: Path, MESH_destination_settings_path: Path, attempt: int) -> bool:
        """
        Run the MESH model.
        
        Args:
            MESH_rank_specific_path (Path): Path specific to this rank's MESH run.
            MESH_destination_settings_path (Path): Path to MESH settings.
            attempt (int): Current attempt number.
        
        Returns:
            bool: True if MESH run was successful, False otherwise.
        """
        # Placeholder for MESH running
        # Implementation to be added later
        return {}
    
class ModelEvaluator:
    """
    A class to evaluate the results of SUMMA and mizuRoute model runs.

    This class handles the evaluation of model outputs, calculation of performance metrics,
    and provides methods for additional evaluations such as geospatial and land attribute assessments.

    Attributes:
        config (Config): Configuration object containing evaluation settings.
        logger (logging.Logger): Logger for this class.
    """
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the ModelEvaluator.

        Args:
            config (Config): Configuration object containing evaluation settings.
            logger (logging.Logger): Logger for this class.
        """
        self.config = config
        self.logger = logger

    def evaluate(self, mizuroute_rank_specific_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Evaluate the model outputs from a specific run.

        This method processes the mizuRoute output, calculates performance metrics for both
        calibration and evaluation periods, and returns the results.

        Args:
            mizuroute_rank_specific_path (Path): Path to the mizuRoute output directory.
            SUMMA_rank_specific_path (Path): Path to the SUMMA output directory.
        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]: 
                A tuple containing two dictionaries (calibration metrics, evaluation metrics).
                Returns (None, None) if evaluation fails.
        """
        try:
            self.logger.info(f"Starting evaluation. mizuRoute path: {mizuroute_rank_specific_path}")
            
            # List files in the directory
            files = os.listdir(mizuroute_rank_specific_path)
            self.logger.info(f"Files in mizuRoute directory: {files}")
            
            # Find the appropriate output file 
            output_file = next((f for f in files if f.endswith('.nc')), None)
            
            if output_file is None:
                self.logger.error(f"No .nc file found in {mizuroute_rank_specific_path}")
                return None, None
            
            output_file_path = mizuroute_rank_specific_path / output_file
            self.logger.info(f"Evaluating file: {output_file_path}")
            
            calib_metrics, eval_metrics = self._calculate_performance_metrics(
                output_file_path, 
                self.config.calib_period, 
                self.config.eval_period, 
                self.config.sim_reach_ID, 
                self.config.obs_file_path
            )

            # Derive SUMMA rank-specific path from mizuRoute path
            #parent_dir = mizuroute_rank_specific_path.parent
            #summa_rank_specific_path = parent_dir / "SUMMA"
            #summa_output_file = str(summa_rank_specific_path) + '/*_day.nc'

            #summa_output = xr.open_mfdataset(summa_output_file)
            #land_attribute_metrics = self.evaluate_land_attributes(summa_output)
            #print(land_attribute_metrics)

            # Evaluate geospatial attributes (MODIS NDSI)
            #geospatial_metrics = self.evaluate_geospatial(summa_output)
            #print(geospatial_metrics)

            return calib_metrics, eval_metrics

        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
            return None, None

    def _calculate_performance_metrics(self, 
                                       output_file_path: Path, 
                                       calib_period: Tuple[np.datetime64, np.datetime64],
                                       eval_period: Tuple[np.datetime64, np.datetime64],
                                       sim_reach_ID: str,
                                       obs_file_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate performance metrics for calibration and evaluation periods.

        Args:
            output_file_path (Path): Path to the model output file.
            calib_period (Tuple[np.datetime64, np.datetime64]): Start and end dates for calibration period.
            eval_period (Tuple[np.datetime64, np.datetime64]): Start and end dates for evaluation period.
            sim_reach_ID (str): ID of the simulated reach to evaluate.
            obs_file_path (Path): Path to the observation file.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: Calibration and evaluation metrics.
        """
        # Read simulation data
        sim_data = xr.open_dataset(output_file_path)
        segment_index = sim_data['reachID'].values == int(sim_reach_ID)
        sim_df = sim_data.sel(seg=segment_index)['IRFroutedRunoff'].to_dataframe().reset_index()
        sim_df.set_index('time', inplace=True)

        # Read observation data
        obs_df = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
        obs_df = obs_df['discharge_cms'].resample('h').mean()

        def calculate_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
            return {
                'RMSE': get_RMSE(obs, sim, transfo=1),
                'KGE': get_KGE(obs, sim, transfo=1),
                'KGEp': get_KGEp(obs, sim, transfo=1),
                'NSE': get_NSE(obs, sim, transfo=1),
                'KGEnp': get_KGEnp(obs, sim, transfo=1),
                'MAE': get_MAE(obs, sim, transfo=1)
            }

        # Calculate metrics for calibration period
        calib_start, calib_end = calib_period
        calib_obs = obs_df.loc[calib_start:calib_end]
        calib_sim = sim_df.loc[calib_start:calib_end]
        calib_metrics = calculate_metrics(calib_obs.values, calib_sim['IRFroutedRunoff'].values)

        # Calculate metrics for evaluation period
        eval_start, eval_end = eval_period
        eval_obs = obs_df.loc[eval_start:eval_end]
        eval_sim = sim_df.loc[eval_start:eval_end]
        eval_metrics = calculate_metrics(eval_obs.values, eval_sim['IRFroutedRunoff'].values)

        return calib_metrics, eval_metrics

    def evaluate_geospatial(self, model_output: xr.Dataset) -> Dict[str, Any]:
        """
        Perform geospatial evaluation of model results using preprocessed MODIS NDSI data.

        Args:
            model_output (xr.Dataset): Model output dataset.

        Returns:
            Dict[str, Any]: Geospatial evaluation metrics.
        """
        try:
            # Load preprocessed MODIS data
            preprocessed_file = self.config.snow_processed_path / f"{self.config.domain_name}_preprocessed_modis_data.csv"
            modis_df = pd.read_csv(preprocessed_file, parse_dates=['date'])
            print(f"Loaded preprocessed MODIS data. Shape: {modis_df.shape}")

            # Get simulated SCA from model output
            sim_sca = model_output.scalarGroundSnowFraction.to_dataframe().reset_index()
            sim_sca = sim_sca.rename(columns={'scalarGroundSnowFraction': 'sim_sca'})

            # Merge observed and simulated data
            merged_df = pd.merge(modis_df, sim_sca, on=['date', 'hru_id'], how='inner')
            print(f"Merged data shape: {merged_df.shape}")

            # Calculate metrics for calibration and evaluation periods
            calib_metrics = self._calculate_geospatial_metrics(merged_df, self.config.calib_period)
            eval_metrics = self._calculate_geospatial_metrics(merged_df, self.config.eval_period)

            print(f"Calibration metrics: {calib_metrics}")
            print(f"Evaluation metrics: {eval_metrics}")

            return {
                'calib_metrics': calib_metrics,
                'eval_metrics': eval_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in evaluate_geospatial: {str(e)}", exc_info=True)
            return {}

    def _calculate_geospatial_metrics(self, df: pd.DataFrame, period: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate performance metrics for geospatial data within a specified period."""
        period_df = df[(df['date'] >= period[0]) & (df['date'] <= period[1])]
        
        return {
            'RMSE': get_RMSE(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1),
            'KGE': get_KGE(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1),
            'KGEp': get_KGEp(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1),
            'NSE': get_NSE(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1),
            'MAE': get_MAE(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1),
            'KGEnp': get_KGEnp(period_df['obs_sca'].values, period_df['sim_sca'].values, transfo=1)
        }

    def evaluate_land_attributes(self, model_output: xr.Dataset) -> Dict[str, Any]:
        """
        Evaluate model performance with respect to land attributes (SWE).

        Args:
            model_output (xr.Dataset): Model output dataset.

        Returns:
            Dict[str, Any]: Land attribute evaluation metrics.
        """
        try:
            # Read the snow observation data
            snow_obs_path = self.config.snow_processed_path / self.config.snow_processed_name
            snow_obs = pd.read_csv(snow_obs_path, parse_dates=['datetime'])

            # Read the snow station shapefile
            station_shp_path = self.config.snow_station_shapefile_path / self.config.snow_station_shapefile_name
            station_gdf = gpd.read_file(station_shp_path)

            # Merge observation data with station information
            merged_data = pd.merge(snow_obs, station_gdf, on='station_id')

            # Group by HRU and calculate average observed SWE
            hru_avg_swe = merged_data.groupby(['HRU_ID', 'datetime'])['snw'].mean().reset_index()

            # Initialize dictionaries to store metrics
            calib_metrics = {}
            eval_metrics = {}

            # Iterate through HRUs with observations
            for hru_id in hru_avg_swe['HRU_ID'].unique():
                hru_obs = hru_avg_swe[hru_avg_swe['HRU_ID'] == hru_id].set_index('datetime')
                hru_sim = model_output['scalarSWE'].sel(hru=hru_id).to_dataframe()

                # Align observation and simulation data
                aligned_data = pd.merge(hru_obs, hru_sim, left_index=True, right_index=True, how='inner')

                # Calculate metrics for calibration period
                calib_data = aligned_data[(aligned_data.index >= self.config.calib_period[0]) & 
                                          (aligned_data.index <= self.config.calib_period[1])]
                calib_metrics[hru_id] = self._calculate_metrics(calib_data['snw'].values, calib_data['scalarSWE'].values)

                # Calculate metrics for evaluation period
                eval_data = aligned_data[(aligned_data.index >= self.config.eval_period[0]) & 
                                         (aligned_data.index <= self.config.eval_period[1])]
                eval_metrics[hru_id] = self._calculate_metrics(eval_data['snw'].values, eval_data['scalarSWE'].values)

            # Calculate average metrics across all HRUs
            avg_calib_metrics = self._average_metrics(calib_metrics)
            avg_eval_metrics = self._average_metrics(eval_metrics)

            return {
                'calib_metrics': avg_calib_metrics,
                'eval_metrics': avg_eval_metrics,
                #'hru_calib_metrics': calib_metrics,
                #'hru_eval_metrics': eval_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in evaluate_land_attributes: {str(e)}", exc_info=True)
            return {}

    def _calculate_metrics(self, obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics for observed and simulated data."""
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1),
            'KGEnp': get_KGEnp(obs, sim, transfo=1)
        }

    def _average_metrics(self, metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics across all HRUs."""
        avg_metrics = {}
        for metric in ['RMSE', 'KGE', 'KGEp', 'NSE', 'MAE', 'KGEnp']:
            avg_metrics[metric] = np.mean([hru_metrics[metric] for hru_metrics in metrics.values()])
        return avg_metrics