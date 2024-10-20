from mpi4py import MPI # type: ignore
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.opt_model_utils import ModelRunner, ModelEvaluator # type: ignore
from utils.logging_utils import setup_logger # type: ignore
from utils.optimisation_utils import calculate_objective_value # type: ignore 
from datetime import datetime

class Worker:
    """
    A class to manage individual worker processes in the parallel optimization setup.

    This class handles the execution of model runs, parameter evaluation, and communication
    with the master process. It uses the ModelRunner and ModelEvaluator classes to perform
    its tasks.

    Attributes:
        config (Config): Configuration object containing worker settings.
        comm (MPI.Comm): MPI communicator for parallel processing.
        rank (int): Rank of the current worker process.
        model_runner (ModelRunner): Instance of ModelRunner for executing model runs.
        model_evaluator (ModelEvaluator): Instance of ModelEvaluator for evaluating model outputs.
        logger (logging.Logger): Logger for this worker.
    """

    def __init__(self, config: 'Config', comm: MPI.Comm, rank: int): # type: ignore
        """
        Initialize the Worker.

        Args:
            config (Config): Configuration object containing worker settings.
            comm (MPI.Comm): MPI communicator for parallel processing.
            rank (int): Rank of the current worker process.
        """
        self.config = config
        self.comm = comm
        self.rank = rank
        self.model_runner = ModelRunner(config, comm, rank)
        
        log_dir = Path(config.root_path) / f'domain_{config.domain_name}' / f'_workLog_{config.domain_name}'
        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'confluence_optimization_{config.domain_name}_{current_time}.log'
        self.logger = setup_logger('confluence_optimization', log_file)


        self.model_evaluator = ModelEvaluator(config, self.logger)

    def run(self) -> None:
        self.logger.info(f"Worker {self.rank} started and waiting for tasks")
        while True:
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG)
            if task is None:
                self.logger.info(f"Worker {self.rank} received termination signal")
                break
            idx, params = task
            self.logger.info(f"Worker {self.rank} received parameters for index {idx}")
            result = self.evaluate_params(params)
            self.logger.info(f"Worker {self.rank} completed evaluation. Sending results back to master")
            self.comm.send((idx, params, result), dest=0)
        self.logger.info(f"Worker {self.rank} shutting down")

    def evaluate_params(self, params):
        """
        Evaluate a set of parameters by running the model and calculating metrics.

        Args:
            params (List[float]): List of parameter values to evaluate.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results, including
                            objective value, calibration metrics, and evaluation metrics.
        """
        self.logger.info(f"Worker {self.rank} starting parameter evaluation")
        try:
            #self.logger.info(f"Worker {self.rank} received params: {params}")
            model_output_path = self.run_models(params)
            if model_output_path is None:
                return {'params': params, 'calib_metrics': None, 'eval_metrics': None, 'objective': float('inf')}
            
            return self.evaluate_results(params, model_output_path)
        except Exception as e:
            self.logger.error(f"Worker {self.rank} error in evaluate_params: {str(e)}", exc_info=True)
            return {'params': params, 'calib_metrics': None, 'eval_metrics': None, 'objective': float('inf')}

    def run_models(self, params):
        """
        Run the hydrological models with the given parameters.

        Args:
            params (List[float]): List of parameter values to use in the model run.

        Returns:
            Optional[Path]: Path to the model output if successful, None otherwise.
        """
        local_params, basin_params = self.model_runner.split_params(params)
        #self.logger.info(f"Worker {self.rank} split params: local={local_params}, basin={basin_params}")
        model_output_path = self.model_runner.run_models(local_params, basin_params)
        if model_output_path is None:
            self.logger.warning(f"Worker {self.rank} model run failed")
            return None
        self.logger.info(f"Worker {self.rank} model run completed. Output path: {model_output_path}")
        return model_output_path

    def evaluate_results(self, params, model_output_path):
        """
        Evaluate the results of the model run.

        Args:
            params (List[float]): The parameters used in the model run.
            model_output_path (Path): Path to the model output files.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation results.
        """
        calib_metrics, eval_metrics = self.model_evaluator.evaluate(model_output_path)
        if calib_metrics is None or eval_metrics is None:
            self.logger.warning(f"Worker {self.rank} evaluation failed")
            return {'params': params, 'calib_metrics': None, 'eval_metrics': None, 'objective': float('inf')}
        
        objective_value = calculate_objective_value(calib_metrics, self.config.optimization_metric)
        self.logger.info(f"Worker {self.rank} evaluation completed. Objective value: {objective_value}")
        
        return {
            'params': params,
            'objective': objective_value,
            'calib_metrics': calib_metrics,
            'eval_metrics': eval_metrics
        }