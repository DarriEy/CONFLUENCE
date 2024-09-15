# In parallel_parameter_estimation.py

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path
from mpi4py import MPI # type: ignore
import yaml # type: ignore

@dataclass
class Config:
    root_path: Path
    domain_name: str
    experiment_id: str
    params_to_calibrate: List[str]
    basin_params_to_calibrate: List[str]
    obs_file_path: Path
    sim_reach_ID: str
    filemanager_name: str
    optimization_metrics: List[str]
    pop_size: int
    mizu_control_file: str
    moo_num_iter: int
    nsga2_n_gen: int
    nsga2_n_obj: int
    optimization_metric: str
    algorithm: str
    num_iter: int
    calib_period: Tuple[datetime, datetime]
    eval_period: Tuple[datetime, datetime]
    local_bounds_dict: Dict[str, Tuple[float, float]]
    basin_bounds_dict: Dict[str, Tuple[float, float]]
    local_bounds: List[Tuple[float, float]]
    basin_bounds: List[Tuple[float, float]]
    all_bounds: List[Tuple[float, float]]
    all_params: List[str]
    poplsize: int
    swrmsize: int
    ngsize: int
    dds_r: float
    diagnostic_frequency: int
    # Add any other necessary attributes

def read_from_confluence_config(file_path: Path, setting: str) -> Any:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get(setting)

def parse_time_period(period_str: str) -> Tuple[datetime, datetime]:
    start, end = [datetime.strptime(date.strip(), '%Y-%m-%d') for date in period_str.split(',')]
    return start, end

def initialize_config(rank: int, comm: MPI.Comm) -> Config:
    confluence_config_path = Path('/path/to/confluence/config_active.yaml')  # Update this path

    if rank == 0:
        root_path = Path(read_from_confluence_config(confluence_config_path, 'CONFLUENCE_DATA_DIR'))
        domain_name = read_from_confluence_config(confluence_config_path, 'DOMAIN_NAME')
        experiment_id = read_from_confluence_config(confluence_config_path, 'EXPERIMENT_ID')
        params_to_calibrate = read_from_confluence_config(confluence_config_path, 'PARAMS_TO_CALIBRATE').split(',')
        basin_params_to_calibrate = read_from_confluence_config(confluence_config_path, 'BASIN_PARAMS_TO_CALIBRATE').split(',')
        obs_file_path = Path(read_from_confluence_config(confluence_config_path, 'STREAMFLOW_PROCESSED_PATH')) / read_from_confluence_config(confluence_config_path, 'STREAMFLOW_PROCESSED_NAME')
        sim_reach_ID = read_from_confluence_config(confluence_config_path, 'SIM_REACH_ID')
        filemanager_name = read_from_confluence_config(confluence_config_path, 'SETTINGS_SUMMA_FILEMANAGER')
        optimization_metrics = read_from_confluence_config(confluence_config_path, 'MOO_OPTIMIZATION_METRICS').split(',')
        pop_size = int(read_from_confluence_config(confluence_config_path, 'POPULATION_SIZE'))
        mizu_control_file = read_from_confluence_config(confluence_config_path, 'SETTINGS_MIZU_CONTROL_FILE')
        moo_num_iter = int(read_from_confluence_config(confluence_config_path, 'MOO_NUM_ITER'))
        nsga2_n_gen = int(read_from_confluence_config(confluence_config_path, 'NSGA2_N_GEN'))
        nsga2_n_obj = int(read_from_confluence_config(confluence_config_path, 'NSGA2_N_OBJ'))
        optimization_metric = read_from_confluence_config(confluence_config_path, 'OPTIMIZATION_METRIC')
        algorithm = read_from_confluence_config(confluence_config_path, 'OPTMIZATION_ALOGORITHM')
        num_iter = int(read_from_confluence_config(confluence_config_path, 'NUMBER_OF_ITERATIONS'))
        calib_period_str = read_from_confluence_config(confluence_config_path, 'CALIBRATION_PERIOD')
        eval_period_str = read_from_confluence_config(confluence_config_path, 'EVALUATION_PERIOD')
        calib_period = parse_time_period(calib_period_str)
        eval_period = parse_time_period(eval_period_str)

        # Add logic to read and process local_bounds_dict and basin_bounds_dict
        # You may need to implement a function to read these from the CONFLUENCE config or other files

        poplsize = int(read_from_confluence_config(confluence_config_path, 'POPLSIZE'))
        swrmsize = int(read_from_confluence_config(confluence_config_path, 'SWRMSIZE'))
        ngsize = int(read_from_confluence_config(confluence_config_path, 'NGSIZE'))
        dds_r = float(read_from_confluence_config(confluence_config_path, 'DDS_R'))
        diagnostic_frequency = int(read_from_confluence_config(confluence_config_path, 'DIAGNOSTIC_FREQUENCY'))

        # Process bounds and params
        local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
        basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
        all_bounds = local_bounds + basin_bounds
        all_params = params_to_calibrate + basin_params_to_calibrate

    else:
        root_path = None
        domain_name = None
        experiment_id = None
        params_to_calibrate = None
        basin_params_to_calibrate = None
        obs_file_path = None
        sim_reach_ID = None
        filemanager_name = None
        optimization_metrics = None
        pop_size = None
        mizu_control_file = None
        moo_num_iter = None
        nsga2_n_gen = None
        nsga2_n_obj = None
        optimization_metric = None
        algorithm = None
        num_iter = None
        calib_period = None
        eval_period = None
        local_bounds_dict = None
        basin_bounds_dict = None
        local_bounds = None
        basin_bounds = None
        all_bounds = None
        all_params = None
        poplsize = None
        swrmsize = None
        ngsize = None
        dds_r = None
        diagnostic_frequency = None

    config = Config(
        root_path=comm.bcast(root_path, root=0),
        domain_name=comm.bcast(domain_name, root=0),
        experiment_id=comm.bcast(experiment_id, root=0),
        params_to_calibrate=comm.bcast(params_to_calibrate, root=0),
        basin_params_to_calibrate=comm.bcast(basin_params_to_calibrate, root=0),
        obs_file_path=comm.bcast(obs_file_path, root=0),
        sim_reach_ID=comm.bcast(sim_reach_ID, root=0),
        filemanager_name=comm.bcast(filemanager_name, root=0),
        optimization_metrics=comm.bcast(optimization_metrics, root=0),
        pop_size=comm.bcast(pop_size, root=0),
        mizu_control_file=comm.bcast(mizu_control_file, root=0),
        moo_num_iter=comm.bcast(moo_num_iter, root=0),
        nsga2_n_gen=comm.bcast(nsga2_n_gen, root=0),
        nsga2_n_obj=comm.bcast(nsga2_n_obj, root=0),
        optimization_metric=comm.bcast(optimization_metric, root=0),
        algorithm=comm.bcast(algorithm, root=0),
        num_iter=comm.bcast(num_iter, root=0),
        calib_period=comm.bcast(calib_period, root=0),
        eval_period=comm.bcast(eval_period, root=0),
        local_bounds_dict=comm.bcast(local_bounds_dict, root=0),
        basin_bounds_dict=comm.bcast(basin_bounds_dict, root=0),
        local_bounds=comm.bcast(local_bounds, root=0),
        basin_bounds=comm.bcast(basin_bounds, root=0),
        all_bounds=comm.bcast(all_bounds, root=0),
        all_params=comm.bcast(all_params, root=0),
        poplsize=comm.bcast(poplsize, root=0),
        swrmsize=comm.bcast(swrmsize, root=0),
        ngsize=comm.bcast(ngsize, root=0),
        dds_r=comm.bcast(dds_r, root=0),
        diagnostic_frequency=comm.bcast(diagnostic_frequency, root=0),
    )

    return config