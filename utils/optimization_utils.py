from mpi4py import MPI # type: ignore
import numpy as np # type: ignore
from typing import List, Tuple, Union, Any, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_utils import get_function_logger # type: ignore
from utils.results_utils import Results # type: ignore
from utils.optimisation_algorithms import ( # type: ignore
    run_de, run_pso, run_sce_ua, run_basin_hopping, run_dds,
    run_nsga2, run_nsga3, run_moead, run_smsemoa, run_mopso, run_borg_moea,
    get_algorithm_kwargs
)
from utils.calibration_utils import read_param_bounds # type: ignore
from utils.ostrich_util import OstrichOptimizer # type: ignore

class Optimizer:
    def __init__(self, config: Dict[str, Any], logger: Any, comm: MPI.Comm, rank: int):
        self.config = config
        self.comm = comm
        self.rank = rank
        self.size = comm.Get_size()
        self.logger = logger
        self.iteration_count = 0
        self.results = Results(self.config, self.logger)
        
        if self.rank == 0:
            self.results.create_iteration_results_file()
        
        self.results.iteration_results_file = self.comm.bcast(self.results.iteration_results_file, root=0)

        #Get the parameters to calibrate and their bounds
        local_parameters_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f'domain_{self.config.get('DOMAIN_NAME')}' / 'settings/summa/localParamInfo.txt'
        basin_parameters_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f'domain_{self.config.get('DOMAIN_NAME')}' / 'settings/summa/basinParamInfo.txt'
        params_to_calibrate = self.config.get('PARAMS_TO_CALIBRATE').split(',')
        basin_params_to_calibrate = self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')


        local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
        basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
        local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
        basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
        self.all_bounds = local_bounds + basin_bounds


    @get_function_logger
    def run_optimization(self) -> Union[Tuple[List[float], float], Tuple[List[List[float]], List[List[float]]]]:
        self.logger.info('Running optimization')
        algorithm = self.config.get('OPTMIZATION_ALOGORITHM')
        if self.rank == 0:
            self.logger.info(f"Starting optimization with {algorithm} algorithm")
        
        if algorithm in ["DE", "PSO", "SCE-UA", "Basin-hopping", "DDS"]:
            result = self.run_single_objective_optimization()
        elif algorithm in ["NSGA-II", "NSGA-III", "MOEA/D", "SMS-EMOA", "Borg-MOEA", "MOPSO"]:
            result = self.run_multi_objective_optimization()
        elif algorithm == "OSTRICH":
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
        chunk_size = max(1, len(all_params) // num_workers)
        results = []

        all_params = np.atleast_2d(all_params)  # Ensure 2D array

        for i in range(0, len(all_params), chunk_size):
            chunk = all_params[i:i+chunk_size]
            self.logger.info(f"Processing chunk of size {len(chunk)}")

            # Send tasks to workers
            for j, params in enumerate(chunk):
                worker_rank = (i + j) % num_workers + 1
                self.logger.info(f"Master sending parameters to worker {worker_rank}")
                self.comm.send(params.tolist(), dest=worker_rank)

            # Collect results from workers
            chunk_results = []
            for j in range(len(chunk)):
                worker_rank = (i + j) % num_workers + 1
                self.logger.info(f"Master waiting for result from worker {worker_rank}")
                response = self.comm.recv(source=worker_rank)
                self.logger.info(f"Master received result from worker {worker_rank}")
                if response is not None and isinstance(response, tuple):
                    params, result = response
                    if 'calib_metrics' in result:
                        obj_values = [-result['calib_metrics'].get(metric, float('-inf')) if metric in ['KGE', 'KGEp', 'KGEnp', 'NSE'] else result['calib_metrics'].get(metric, float('inf')) for metric in self.config.optimization_metrics]
                        chunk_results.append(obj_values)
                        if self.rank == 0:
                            self.results.process_iteration_results(params, result)
                    else:
                        chunk_results.append([float('inf')] * len(self.config.get('MOO_OPTIMIZATION_METRICS').split(',')))
                else:
                    chunk_results.append([float('inf')] * len(self.config.get('MOO_OPTIMIZATION_METRICS').split(',')))
            
            results.extend(chunk_results)

        self.logger.info(f"Master completed parallel evaluation")
        return np.array(results)

    def run_single_objective_optimization(self) -> Tuple[List[float], float]:
        algorithm_funcs = {
            "DE": run_de,
            "PSO": run_pso,
            "SCE-UA": run_sce_ua,
            "Basin-hopping": run_basin_hopping,
            "DDS": run_dds
        }
        
        algorithm_func = algorithm_funcs.get(self.config.get('OPTMIZATION_ALOGORITHM'))
        if algorithm_func is None:
            raise ValueError(f"Unknown single-objective optimization algorithm: {self.config.get('OPTMIZATION_ALOGORITHM')}")

        kwargs = get_algorithm_kwargs(self.config, self.size)
        self.logger.info(f"{self.config.get('OPTMIZATION_ALOGORITHM')} parameters: {kwargs}")
        
        best_params, best_value = algorithm_func(
            self.parallel_objective_function,
            self.all_bounds,
            **kwargs
        )

        self.logger.info(f"Optimization completed. Best params: {best_params}")
        self.logger.info(f"Best objective value: {best_value}")

        return best_params, best_value

    def run_multi_objective_optimization(self) -> Tuple[List[List[float]], List[List[float]]]:
        optimization_metrics = self.config.get('MOO_OPTIMIZATION_METRICS').split(',')
        algorithm = self.config.get('OPTMIZATION_ALOGORITHM')
        n_obj = len(optimization_metrics)
        
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
        algorithm_func = algorithm_funcs.get(algorithm)

        if algorithm_func is None:
            raise ValueError(f"Unknown multi-objective optimization algorithm: {algorithm}")
        
        best_params, best_values = algorithm_func(
            self.parallel_objective_function,
            self.all_bounds,
            n_obj,
            **kwargs
        )
        
        if self.rank == 0:
            self.logger.info(f"Multi-objective optimization completed")
            self.logger.info(f"Number of Pareto optimal solutions: {len(best_params)}")
            self.results.save_pareto_optimal_solutions(best_params, best_values)
        
        return best_params, best_values

    def run_ostrich_optimization(self) -> Tuple[List[float], float]:
        optimizer = OstrichOptimizer(self.config, self.logger)
        optimizer.run_optimization()