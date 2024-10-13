import sys
import os
from pathlib import Path
from typing import Dict
import traceback
import logging
import subprocess
import xarray as xr # type: ignore
import numpy as np # type: ignore
import datetime
import pandas as pd # type: ignore
import tempfile
import shutil
import time
import glob


# Set up logging
rank = sys.argv[1] if len(sys.argv) > 1 else "unknown"
log_file = Path(f'run_trial_rank{rank}.log')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ])

logger = logging.getLogger(__name__)
sys.path.append(str(Path('/Users/darrieythorsson/compHydro/code/CONFLUENCE').resolve()))

from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore   
from utils.config_utils import ConfigManager # type: ignore

def prepare_summa_job_script(config, rank):
    script_content = f"""#!/bin/bash

#SBATCH --cpus-per-task={config.get('SETTINGS_SUMMA_CPUS_PER_TASK')}
#SBATCH --time={config.get('SETTINGS_SUMMA_TIME_LIMIT')}
#SBATCH --mem={config.get('SETTINGS_SUMMA_MEM')}
#SBATCH --job-name=SUMMA_Rank{rank}
#SBATCH --output=SUMMA_Rank{rank}_%j.out

module load gcc/9.3.0
module load netcdf-fortran
module load openblas
module load caf

gru_max={config.get('SETTINGS_SUMMA_GRU_COUNT')}
gru_count={config.get('SETTINGS_SUMMA_GRU_PER_JOB')}

offset=$SLURM_ARRAY_TASK_ID
gru_start=$(( 1 + gru_count*offset ))
check=$(( $gru_start + $gru_count ))

if [ $check -gt $gru_max ]
then
  gru_count=$((gru_max - gru_start + 1))
fi

summa_command="{config.get('SETTINGS_SUMMA_PARALLEL_EXE')} -g $gru_start $gru_count -m {config.get('SETTINGS_SUMMA_FILEMANAGER')} --caf.scheduler.max-threads=$SLURM_CPUS_PER_TASK"
$summa_command > {config.get('CONFLUENCE_DATA_DIR')}/domain_{config.get('DOMAIN_NAME')}/simulations/{config.get('EXPERIMENT_ID')}/SUMMA/summa_log_rank{rank}_$SLURM_ARRAY_TASK_ID.txt
"""
    
    script_file = Path(f'summa_job_rank{rank}.sh')
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    return script_file

def submit_summa_array_job(config, rank):
    job_script = prepare_summa_job_script(config, rank)
    result = subprocess.run(['sbatch', '--array=0-3', str(job_script)], capture_output=True, text=True)
    job_id = parse_job_id(result.stdout)
    logger.info(f"Submitted SUMMA array job for rank {rank} with job ID: {job_id}")
    return job_id

def parse_job_id(stdout):
    # Extract job ID from SLURM output
    for line in stdout.split('\n'):
        if "Submitted batch job" in line:
            return line.split()[-1]
    return None

def wait_for_summa_completion(job_id):
    logger.info(f"Waiting for SUMMA job {job_id} to complete...")
    while True:
        result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        if job_id not in result.stdout:
            break
        time.sleep(60)  # Check every minute
    logger.info(f"SUMMA job {job_id} completed")

def post_process_summa_output(config, rank):
    logger.info("Post-processing SUMMA output")

    # Use existing paths from the config
    summa_output_path = get_rank_specific_path(Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}", 
                                               config.get('EXPERIMENT_ID'), "SUMMA", rank) / "output"
    mizuRoute_input_path = get_rank_specific_path(Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}", 
                                                  config.get('EXPERIMENT_ID'), "mizuRoute", rank) / "input"

    src_dir = Path(summa_output_path)
    src_pat = f"{config.get('EXPERIMENT_ID')}_day.nc"
    src_var = "averageRoutedRunoff"  # Ensure this is the correct variable name
    des_dir = Path(mizuRoute_input_path)
    des_fil = f"{config.get('EXPERIMENT_ID')}_{{}}_.nc"

    # Extract start and end years from config.calib_period and config.eval_period
    calib_period = config.get('CALIBRATION_PERIOD').split(',')
    eval_period = config.get('EVALUATION_PERIOD').split(',')
    start_year = min(calib_period[0].year, eval_period[0].year)
    end_year = max(calib_period[1].year, eval_period[1].year)

    # Ensure the output path exists
    des_dir.mkdir(parents=True, exist_ok=True)

    # Get the names of all inputs
    src_files = glob.glob(str(src_dir / src_pat))
    src_files.sort()

    def make_new_file(year):
        output_file = des_dir / des_fil.format(year)
        if output_file.exists():
            logger.info(f'File for {year} already exists. Skipping.')
            return

        logger.info(f'Starting on file for {year}.')

        new_ds = None

        for src_file in src_files:
            logger.debug(f'Processing {src_file}')

            with xr.open_dataset(src_file) as ds:
                ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                ds = ds[[src_var]]

                if new_ds is None:
                    new_ds = ds
                else:
                    new_ds = xr.merge([new_ds, ds])

        if new_ds is not None:
            new_ds.to_netcdf(output_file)
            logger.info(f'Saved file for {year}')
        else:
            logger.warning(f'No data found for {year}')

    for year in range(start_year, end_year + 1):
        make_new_file(year)

    logger.info("SUMMA output post-processing completed")

def run_summa(config, rank):
    if config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA'):
        job_id = submit_summa_array_job(config, rank)
        wait_for_summa_completion(job_id)
        post_process_summa_output(config)
    else:
        # Existing serial SUMMA run logic
        summa_exe = f"{config['CONFLUENCE_DATA_DIR']}/installs/summa/bin/{config['SUMMA_EXE']}"
        file_manager = get_summa_settings_path(config, rank) / config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        log_path = Path.cwd() / "SUMMA_logs"
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"summa_run_rank{rank}_{timestamp}.log"
        
        summa_command = [str(summa_exe), "-m", str(file_manager)]
        
        logger.info(f"Running SUMMA for rank {rank} with command: {' '.join(summa_command)}")
        logger.info(f"SUMMA log will be written to: {log_path / log_name}")
        
        try:
            with open(log_path / log_name, 'w') as log_file:
                result = subprocess.run(summa_command, check=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            
            logger.info(f"SUMMA run completed successfully for rank {rank}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"SUMMA run failed for rank {rank} with return code: {e.returncode}")
            logger.error(f"Please check the log file for details: {log_path / log_name}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while running SUMMA for rank {rank}: {str(e)}")
            return False

def run_mizuroute(config, rank):
    mizuroute_exe = f"{config['CONFLUENCE_DATA_DIR']}/installs/mizuroute/route/bin/{config['EXE_NAME_MIZUROUTE']}"
    control_file = get_mizuroute_settings_path(config, rank) / config.get('SETTINGS_MIZU_CONTROL_FILE')
    cmd = f"{mizuroute_exe} {control_file}"
    
    logger.info(f"Running mizuRoute for rank {rank} with command: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return result.returncode == 0

def get_summa_settings_path(config, rank):
    rank_specific_path = get_rank_specific_path(Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}", 
                                                config.get('EXPERIMENT_ID'), "SUMMA", rank)
    return rank_specific_path / "run_settings"

def get_mizuroute_settings_path(config, rank):
    rank_specific_path = get_rank_specific_path(Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}", 
                                                config.get('EXPERIMENT_ID'), "mizuRoute", rank)
    return rank_specific_path / "run_settings"

def get_rank_specific_path(base_path: Path, experiment_id: str, model: str, rank: int) -> Path:
    experiment_id = experiment_id.split('_rank')[0]
    return base_path / "simulations" / f"{experiment_id}_rank{rank}" / model

def calculate_objective_function(config, rank):
    logger.info("Calculating objective functions")

    try:
        calib_metrics, eval_metrics = evaluate_model(config, rank)
        
        objective_values = []
        for metric in config.get('OPTIMIZATION_METRICS').split(','):
            if metric in calib_metrics:
                value = calib_metrics[metric]
                
                # For metrics where higher values are better, we negate the value for minimization
                if metric in ['KGE', 'KGEp', 'KGEnp', 'NSE']:
                    value = -value
                
                objective_values.append(float(value))  # Convert to float to ensure it's not a numpy type
                logger.info(f"Objective function ({metric}) value: {value}")
            else:
                logger.error(f"Optimization metric {metric} not found in calibration metrics")
                objective_values.append(float('inf'))
        
        logger.info(f"All calibration metrics: {calib_metrics}")
        return objective_values 
    
    except Exception as e:
        logger.error(f"Error calculating objective functions: {str(e)}")
        return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))

def round_to_hour(dt):
    return dt.round('h')

def evaluate_model(config, rank):
    logger.info(f"Evaluating model performance for rank {rank}")

    try:
        # Get the mizuRoute output file
        mizuroute_output_path = get_mizuroute_output_path(config, rank)
        
        # Open the mizuRoute output file
        sim_data = xr.open_dataset(mizuroute_output_path)
        
        # Get the simulated streamflow for the specified reach
        segment_index = sim_data['reachID'].values == int(config.get('SIM_REACH_ID'))
        sim_df = sim_data.sel(seg=segment_index)['IRFroutedRunoff'].to_dataframe().reset_index()

        # Round the time index to the nearest hour
        sim_df['time'] = sim_df['time'].apply(round_to_hour)
        sim_df.set_index('time', inplace=True)

        # Read observation data
        obs_df = pd.read_csv(config.get('OBSERVATIONS_PATH'), index_col='datetime', parse_dates=True)
        obs_df = obs_df['discharge_cms'].resample('h').mean()

        logger.info(obs_df, 'observations')
        logger.info(sim_df, 'simulations')

        # Calculate metrics for calibration period
        calib_period = config.get('CALIBRATION_PERIOD').split(',')
        eval_period = config.get('EVALUATION_PERIOD').split(',')

        # Ensure indexes match
        common_index = obs_df.index.intersection(sim_df.index)
        if len(common_index) != len(obs_df) or len(common_index) != len(sim_df):
            logger.warning("Observation and simulation indexes don't fully match. Using only common dates.")
            obs_df = obs_df.loc[common_index]
            sim_df = sim_df.loc[common_index]

        calib_metrics = calculate_metrics(
            obs_df.loc[calib_period[0]:calib_period[1]],
            sim_df.loc[calib_period[0]:calib_period[1]]['IRFroutedRunoff']
        )
        
        logger.info(f'Calib_metrics {calib_metrics}')

        # Calculate metrics for evaluation period
        eval_metrics = calculate_metrics(
            obs_df.loc[eval_period[0]:eval_period[1]],
            sim_df.loc[eval_period[0]:eval_period[1]]['IRFroutedRunoff']
        )

        logger.info(f"Calibration metrics: {calib_metrics}")
        logger.info(f"Evaluation metrics: {eval_metrics}")

        return calib_metrics, eval_metrics

    except Exception as e:
        logger.error(f"Error evaluating model for rank {rank}: {str(e)}")
        return None, None


def calculate_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:


    return {
        'RMSE': get_RMSE(obs, sim, transfo=1),
        'KGE': get_KGE(obs, sim, transfo=1),
        'KGEp': get_KGEp(obs, sim, transfo=1),
        'NSE': get_NSE(obs, sim, transfo=1),
        'MAE': get_MAE(obs, sim, transfo=1),
        'KGEnp': get_KGEnp(obs, sim, transfo=1)
    }

def get_mizuroute_output_path(config, rank):
    mizuroute_settings_path = get_mizuroute_settings_path(config, rank)
    output_dir = mizuroute_settings_path.parent
    output_file = output_dir / f"{config.get('EXPERIMENT_ID')}_rank{rank}.h.*.nc"  # Adjust the file pattern as needed
    matching_files = list(output_dir.glob(output_file.name))
    
    if not matching_files:
        raise FileNotFoundError(f"No matching mizuRoute output file found for rank {rank}: {output_file}")
    
    return str(matching_files[0])

def update_trial_params(config, rank):
    logger.info(f"Updating trial parameters for rank {rank}")

    # Read multipliers
    multipliers_file = Path(os.getcwd()) / f'multipliers.txt'
    try:
        with open(multipliers_file, 'r') as f:
            multipliers = [float(line.strip()) for line in f]
    except FileNotFoundError:
        logger.error(f"Multipliers file not found: {multipliers_file}")
        return False
    except ValueError:
        logger.error("Invalid data in multipliers file")
        return False

    all_params = config.get('PARAMS_TO_CALIBRATE').split(',') + config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')
    if len(multipliers) != len(all_params):
        logger.info(f'All params: {all_params}, multipliers: {multipliers}')
        logger.error("Mismatch between number of multipliers and parameters to calibrate")
        return False


    summa_settings_path = get_summa_settings_path(config, rank)
    trial_param_file = summa_settings_path / config.get('SETTINGS_SUMMA_TRIALPARAMS')
    trial_param_file_priori = trial_param_file.with_name(f"{trial_param_file.stem}.priori.nc")

    try:
        # Load data from both files
        with xr.open_dataset(trial_param_file_priori) as src, xr.open_dataset(trial_param_file) as dst:
            src_data = src.load()
            dst_data = dst.load()

        # Update parameters
        for param, multiplier in zip(all_params, multipliers):
            if param in src_data.variables and param in dst_data.variables:
                old_value = dst_data[param][:].mean().item()
                new_value = src_data[param][:] * multiplier
                
                # Ensure the new value is within valid range
                param_min = float(src_data[param].attrs.get('valid_min', -np.inf))
                param_max = float(src_data[param].attrs.get('valid_max', np.inf))
                new_value = np.clip(new_value, param_min, param_max)
                
                dst_data[param][:] = new_value
                new_value = dst_data[param][:].mean().item()
                logger.info(f"Updated parameter {param}: old={old_value:.6e}, new={new_value:.6e}, multiplier={multiplier:.6e}")
            else:
                logger.warning(f"Parameter {param} not found in trial parameter file or priori file")

        # Write to a temporary file and replace the original
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            dst_data.to_netcdf(tmp_file.name)
        
        shutil.move(tmp_file.name, trial_param_file)

        logger.info("Trial parameters updated successfully")
        return True

    except Exception as e:
        logger.error(f"Error updating trial parameters: {str(e)}")
        return False

def run_model_and_calculate_objective(config, rank):
    logger.info(f"Starting run_model_and_calculate_objective for rank {rank}")
    
    try:
        # Update trial parameters
        update_success = update_trial_params(config, rank)
        if not update_success:
            logger.error(f"Failed to update trial parameters for rank {rank}")
            return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))

        # Run SUMMA
        summa_success = run_summa(config, rank)
        if not summa_success:
            logger.error(f"SUMMA run failed for rank {rank}")
            return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))

        # Run mizuRoute
        mizuroute_success = run_mizuroute(config, rank)
        if not mizuroute_success:
            logger.error(f"mizuRoute run failed for rank {rank}")
            return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))

        # Calculate objective function
        objective_values = calculate_objective_function(config, rank)

        # Check for invalid values
        if isinstance(objective_values, list):
            if any(np.isinf(value) or np.isnan(value) for value in objective_values):
                logger.error(f"Invalid objective value(s) calculated for rank {rank}")
                return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))
        else:
            if np.isinf(objective_values) or np.isnan(objective_values):
                logger.error(f"Invalid objective value calculated for rank {rank}")
                return float('inf')

        logger.info(f"Objective function value(s) for rank {rank}: {objective_values}")
        return objective_values
    except Exception as e:
        logger.error(f"Error in run_model_and_calculate_objective for rank {rank}: {str(e)}")
        logger.error(traceback.format_exc())
        return [float('inf')] * len(config.get('OPTIMIZATION_METRICS').split(','))

def main():

    logger.info('executing main')
    config_path = Path('/Users/darrieythorsson/compHydro/code/CONFLUENCE/0_config_files')
    config_name = 'config_active.yaml'
    config_manager = ConfigManager(config_path / config_name)
    config = config_manager.config


    if len(sys.argv) < 2:
        logger.error("Usage: python run_trial.py <rank>")
        sys.exit(1)

    rank = int(sys.argv[1])
    logger.info(f"Starting run_trial.py for rank {rank}")

    #ontrol_file_path = Path('/Users/darrieythorsson/compHydro/code/CWARHM/0_control_files/control_active.txt')
    #config = Config.from_control_file(control_file_path)
    logger.info("Config initialized")

    logger.info("Running model and calculating objectives...")
    objective_values = run_model_and_calculate_objective(config, rank)

    # Write to the file that Ostrich expects
    output_file = Path('ostrich_objective.txt')
    with open(output_file, 'w') as f:
        f.write("\n".join(map(str, objective_values)) + "\n")
    logger.info(f"Objective value(s) {objective_values} written to: {output_file}")

    logger.info(f"Trial completed for rank {rank}")


if __name__ == "__main__":
    main()
