from mpi4py import MPI # type: ignore
from utils.model_utils import ModelRunner, ModelEvaluator # type: ignore
from utils.logging_utils import get_logger # type: ignore
from utils.optimisation_utils import calculate_objective_value # type: ignore 

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
        self.model_evaluator = ModelEvaluator(config, get_logger(f'ModelEvaluator_{rank}', config.root_path, config.domain_name, config.experiment_id))
        self.logger = get_logger(f'Worker_{rank}', config.root_path, config.domain_name, config.experiment_id)

    def run(self) -> None:
        """
        Run the worker process.

        This method enters a loop to receive tasks from the master process,
        execute them, and send results back. It continues until a termination
        signal is received.
        """
        self.logger.info(f"Worker {self.rank} started and waiting for tasks")
        while True:
            try:
                #self.logger.info(f"Worker {self.rank} waiting for next task")
                task = self.comm.recv(source=0, tag=MPI.ANY_TAG)
                #self.logger.info(f"Worker {self.rank} received task: {task}")
                if task is None:
                    self.logger.info(f"Worker {self.rank} received termination signal")
                    break
                params = task
                #self.logger.info(f"Worker {self.rank} received parameters: {params}")
                result = self.evaluate_params(params)
                self.logger.info(f"Worker {self.rank} completed evaluation. Sending results back to master")
                self.comm.send((params, result), dest=0)  # Send both params and result back to master
            except Exception as e:
                self.logger.error(f"Worker {self.rank} encountered an error: {str(e)}", exc_info=True)
                self.comm.send(None, dest=0)  # Send None to indicate an error
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