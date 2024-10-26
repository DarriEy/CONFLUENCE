from mpi4py import MPI # type: ignore
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np # type: ignore
from datetime import datetime
from typing import Union
import sys
from collections import deque
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.logging_utils import setup_logger # type: ignore
from utils.optimisation_utils import run_nsga2, run_nsga3, run_moead, run_smsemoa, run_mopso, get_algorithm_kwargs, run_de, run_dds, run_basin_hopping, run_pso, run_sce_ua, run_borg_moea # type: ignore
from utils.results_utils import Results # type: ignore
from utils.parallel_utils import Worker # type: ignore
from utils.ostrich_util import OstrichOptimizer # type: ignore
from utils.optimization_config import initialize_config # type: ignore

class Optimizer:

    """
    A class to manage the optimization process for hydrological model calibration.

    This class handles the coordination of workers, parameter distribution, and result collection
    for both single-objective and multi-objective optimization algorithms.

    Attributes:
        config (Config): Configuration object containing optimization settings.
        comm (MPI.Comm): MPI communicator for parallel processing.
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        logger (logging.Logger): Logger for this optimizer.
        iteration_count (int): Counter for optimization iterations.
        iteration_results_file (Optional[str]): Path to the file storing iteration results.
    """

    def __init__(self, config: Dict[str, Any], comm: MPI.Comm, rank: int):
        """
        Initialize the Optimizer.

        Args:
            config (Config): Configuration object containing optimization settings.
            comm (MPI.Comm): MPI communicator for parallel processing.
            rank (int): Rank of the current process.
        """
        self.config = config
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()

        log_dir = Path(config.root_path) / f'domain_{config.domain_name}' / f'_workLog_{config.domain_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_optimization_{config.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_optimization', log_file)

        self.iteration_count = 0
        self.results = Results(config, self.logger)
        
        if self.rank == 0:
            self.results.create_iteration_results_file()
        
        self.results.iteration_results_file = self.comm.bcast(self.results.iteration_results_file, root=0)

    def run_optimization(self) -> Union[Tuple[List[float], float], Tuple[List[List[float]], List[List[float]]]]:
        """
        Run the optimization process.

        This method determines the type of optimization (single or multi-objective)
        based on the configured algorithm and executes the appropriate optimization routine.

        Returns:
            Union[Tuple[List[float], float], Tuple[List[List[float]], List[List[float]]]]:
                For single-objective optimization: (best_params, best_value)
                For multi-objective optimization: (best_params_list, best_values_list)
        """
        if self.rank == 0:
            self.logger.info(f"Starting optimization with {self.config.algorithm} algorithm")
        
        

        if self.config.algorithm in ["DE", "PSO", "SCE-UA", "Basin-hopping", "DDS"]:
            result = self.run_single_objective_optimization()
        elif self.config.algorithm in ["NSGA-II", "NSGA-III", "MOEA/D", "SMS-EMOA", "Borg-MOEA", "MOPSO"]:
            result = self.run_multi_objective_optimization()
        elif self.config.algorithm == "OSTRICH":
            result = self.run_ostrich_optimization()
        else:
            raise ValueError("Invalid algorithm choice")

        if self.rank == 0:
            self.logger.info("Optimization completed. Generating final diagnostics.")
            self.results.generate_final_diagnostics()
            self.results.save_final_results()
            self.logger.info(f"Final results and diagnostics saved to: {self.results.output_folder}")

        return result

    def parallel_objective_function(self, all_params: np.ndarray) -> np.ndarray:
        self.logger.info(f"parallel_objective_function called with {len(all_params)} parameter sets")
        num_workers = self.size - 1
        all_params = np.atleast_2d(all_params)
        task_queue = deque(enumerate(all_params))
        results = [None] * len(all_params)
        active_workers = set()
        requests = []

        # Initial distribution of tasks
        for i in range(1, min(num_workers + 1, len(all_params) + 1)):
            if task_queue:
                idx, params = task_queue.popleft()
                self.logger.info(f"Master sending initial parameters to worker {i}")
                req = self.comm.isend((idx, params.tolist()), dest=i)
                requests.append(req)
                active_workers.add(i)

        while active_workers or task_queue:
            if active_workers:
                status = MPI.Status()
                worker_result = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                worker_rank = status.Get_source()
                self.logger.info(f"Master received result from worker {worker_rank}")
                
                if worker_result is not None and isinstance(worker_result, tuple):
                    idx, params, result = worker_result
                    if 'calib_metrics' in result:
                        obj_values = [-result['calib_metrics'].get(metric, float('-inf')) if metric in ['KGE', 'KGEp', 'KGEnp', 'NSE'] else result['calib_metrics'].get(metric, float('inf')) for metric in self.config.optimization_metrics]
                        results[idx] = obj_values
                        if self.rank == 0:
                            self.results.process_iteration_results(params, result)
                    else:
                        results[idx] = [float('inf')] * len(self.config.optimization_metrics)
                else:
                    results[idx] = [float('inf')] * len(self.config.optimization_metrics)

                # Assign new task to the worker that just finished
                if task_queue:
                    idx, params = task_queue.popleft()
                    self.logger.info(f"Master sending new parameters to worker {worker_rank}")
                    req = self.comm.isend((idx, params.tolist()), dest=worker_rank)
                    requests.append(req)
                else:
                    active_workers.remove(worker_rank)

        # Wait for all requests to complete
        MPI.Request.waitall(requests)

        self.logger.info(f"Master completed parallel evaluation")
        return np.array(results)

    def run_single_objective_optimization(self) -> Tuple[List[float], float]:
        """
        Run single-objective optimization.

        This method sets up and executes the chosen single-objective optimization algorithm.

        Returns:
            Tuple[List[float], float]: Best parameters and their corresponding objective value.
        """
        algorithm_funcs = {
            "DE": run_de,
            "PSO": run_pso,
            "SCE-UA": run_sce_ua,
            "Basin-hopping": run_basin_hopping,
            "DDS": run_dds
        }
        
        algorithm_func = algorithm_funcs.get(self.config.algorithm)
        if algorithm_func is None:
            raise ValueError(f"Unknown single-objective optimization algorithm: {self.config.algorithm}")

        kwargs = get_algorithm_kwargs(self.config, self.size)
        self.logger.info(f"{self.config.algorithm} parameters: {kwargs}")
        
        best_params, best_value = algorithm_func(
            self.parallel_objective_function,
            self.config.all_bounds,
            **kwargs
        )

        self.logger.info(f"Optimization completed. Best params: {best_params}")
        self.logger.info(f"Best objective value: {best_value}")

        return best_params, best_value
        
    def run_multi_objective_optimization(self) -> Tuple[List[List[float]], List[List[float]]]:
        n_obj = len(self.config.optimization_metrics)
        
        algorithm_funcs = {
            "NSGA-II": run_nsga2,
            "NSGA-III": run_nsga3,
            "MOEA/D": run_moead,
            "SMS-EMOA": run_smsemoa,
            "Borg-MOEA": run_borg_moea,
            "MOPSO": run_mopso
        }
        
        kwargs = get_algorithm_kwargs(self.config, self.size)
        self.logger.info(f"algorithm parameters: {kwargs}")
        algorithm_func = algorithm_funcs.get(self.config.algorithm)

        if algorithm_func is None:
            raise ValueError(f"Unknown multi-objective optimization algorithm: {self.config.algorithm}")
        
        best_params, best_values = algorithm_func(
            self.parallel_objective_function,
            self.config.all_bounds,
            n_obj,
            **kwargs
        )
        
        if self.rank == 0:
            self.logger.info(f"Multi-objective optimization completed")
            self.logger.info(f"Number of Pareto optimal solutions: {len(best_params)}")
            self.results.save_pareto_optimal_solutions(best_params, best_values)
        
        return best_params, best_values

    def run_ostrich_optimization(self) -> Tuple[List[float], float]:
        ostrich_optimizer = OstrichOptimizer(self.config, self.comm, self.rank)
        ostrich_optimizer.prepare_ostrich_files()
        ostrich_optimizer.run_ostrich()
        return ostrich_optimizer.parse_ostrich_results()

def main() -> None:
    """
    Main function to run the parallel parameter estimation process.

    This function initializes the MPI environment, sets up logging, and orchestrates 
    the optimization process across multiple processes. It handles both the master process 
    (rank 0) which manages the optimization, and worker processes which perform model evaluations.

    The function performs the following steps:
    1. Initialize MPI environment and get process rank and size
    2. Initialize configuration and set up logging
    3. If master process (rank 0):
       - Set up and run the optimization
       - Log optimization results
       - Write results to file
    4. If worker process:
       - Set up worker and run in listening mode for tasks
    5. Ensure all processes finish before exiting

    Raises:
        Exception: Any unhandled exception during the optimization process
    """
    parser = argparse.ArgumentParser(description="Run parallel parameter estimation for CONFLUENCE")
    parser.add_argument("config_path", type=str, help="Path to the CONFLUENCE configuration file")
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()

    confluence_config_path = Path(args.config_path)
    #confluence_config_path = Path('/Users/darrieythorsson/compHydro/code/CONFLUENCE/0_config_files/config_active.yaml')  # Update this path
    config = initialize_config(rank, comm,confluence_config_path)

    log_dir = Path(config.root_path) /f'domain_{config.domain_name}' / f'_workLog_{config.domain_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'confluence_optimization_{config.domain_name}_{current_time}.log'
    logger = setup_logger('confluence_optimization', log_file)
    
    logger.info(f"Process {rank} initialized")

    optimizer = Optimizer(config, comm, rank)

    if rank == 0:
        optimizer = Optimizer(config, comm, rank)
        start_time = datetime.now()

        logger.info(f"Starting optimization with {config.algorithm} algorithm")
        logger.info(f"Optimization metric: {config.optimization_metric}")
        logger.info(f"Number of iterations: {config.num_iter}")
        logger.info(f"Calibration period: {config.calib_period[0].date()} to {config.calib_period[1].date()}")
        logger.info(f"Evaluation period: {config.eval_period[0].date()} to {config.eval_period[1].date()}")

        try:
            best_params, best_value = optimizer.run_optimization()
            
            logger.info(f"Optimization completed")
            logger.info(f"Best {config.optimization_metric} value: {best_value if config.optimization_metric in ['RMSE', 'MAE'] else -best_value}")
            logger.info("Best parameters:")
            for param, value in zip(config.all_params, best_params):
                logger.info(f"{param}: {value:.6e}")

            # Run final evaluation with best parameters
            final_result = optimizer.parallel_objective_function([best_params])[0]
            
            logger.info("Final performance metrics:")
            logger.info(f"Calibration: {final_result['calib_metrics']}")
            logger.info(f"Evaluation: {final_result['eval_metrics']}")

            # Write results to file
            optimizer.results.write_optimization_results(best_params, final_result)

            end_time = datetime.now()
            logger.info(f"Optimization ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
            logger.info(f"Total optimization time: {end_time - start_time}")

        except Exception as e:
            logger.error(f"Error occurred during optimization: {str(e)}", exc_info=True)
        finally:
            end_time = datetime.now()
            logger.info(f"Optimization ended on {end_time.strftime('%Y/%m/%d %H:%M:%S')}")
            logger.info(f"Total optimization time: {end_time - start_time}")
            
            # Send termination signal to workers
            for i in range(1, size):
                comm.send(None, dest=i)
    else:
        worker = Worker(config, comm, rank)
        worker.run()

    logger.info(f"Process {rank} finishing")
    comm.Barrier()  # Ensure all processes finish before exiting

if __name__ == "__main__":
    main()