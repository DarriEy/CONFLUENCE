# optimization_utils.py
from abc import ABC, abstractmethod
from scipy.optimize import differential_evolution, basinhopping # type: ignore
import spotpy # type: ignore
import numpy as np # type: ignore
from spotpy.algorithms import sceua # type: ignore
from typing import Callable, List, Tuple, Dict, Any, Optional
from pymoo.algorithms.moo.nsga2 import NSGA2 # type: ignore
from pymoo.algorithms.moo.nsga3 import NSGA3 # type: ignore
from pymoo.algorithms.moo.moead import MOEAD # type: ignore
from pymoo.algorithms.moo.sms import SMSEMOA # type: ignore
from pymoo.core.problem import Problem # type: ignore
from pymoo.core.algorithm import Algorithm # type: ignore
from pymoo.core.population import Population # type: ignore
from pymoo.operators.crossover.sbx import SBX # type: ignore
from pymoo.operators.mutation.pm import PolynomialMutation # type: ignore
from pymoo.operators.sampling.rnd import FloatRandomSampling # type: ignore
from pymoo.optimize import minimize # type: ignore
from pymoo.util.ref_dirs import get_reference_directions # type: ignore
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting # type: ignore
from pymoo.operators.crossover.sbx import SBX # type: ignore
from pymoo.algorithms.soo.nonconvex.de import DE # type: ignore
from pymoo.algorithms.soo.nonconvex.pso import PSO # type: ignore
from mpi4py import MPI # type: ignore
import logging


class OptimizationAlgorithm(ABC):
    @abstractmethod
    def optimize(self, objective_func: Callable, bounds: List[Tuple[float, float]], **kwargs) -> Tuple[np.ndarray, float]:
        pass

class DifferentialEvolution(OptimizationAlgorithm):
    def optimize(self, objective_func, bounds, **kwargs):
        
        def parallel_objective(X):
            return objective_func(X)

        result = differential_evolution(
            parallel_objective, 
            bounds, 
            updating='deferred',
            workers=1,  # We're handling parallelization ourselves
            **kwargs
        )
        return result.x, result.fun
    
class ParticleSwarmOptimization(OptimizationAlgorithm):
    def optimize(self, objective_func, bounds, **kwargs):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        pop_size = kwargs.get('popsize', 100)
        max_iter = kwargs.get('maxiter', 1000)
        w = kwargs.get('w', 0.5)
        c1 = kwargs.get('c1', 1.5)
        c2 = kwargs.get('c2', 1.5)

        def init_population(pop_size, bounds):
            return np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(pop_size, len(bounds))
            )

        if rank == 0:
            population = init_population(pop_size, bounds)
            velocities = np.zeros_like(population)
            personal_best_pos = population.copy()
        else:
            population = None
            velocities = None
            personal_best_pos = None

        # Scatter population across all ranks
        local_pop_size = pop_size // size
        local_population = np.zeros((local_pop_size, len(bounds)))
        comm.Scatter(population, local_population, root=0)

        local_velocities = np.zeros_like(local_population)
        comm.Scatter(velocities, local_velocities, root=0)

        local_personal_best_pos = local_population.copy()
        comm.Scatter(personal_best_pos, local_personal_best_pos, root=0)

        # Evaluate initial fitness
        local_fitness = objective_func(local_population)
        local_personal_best_fitness = local_fitness.copy()

        # Initialize global best
        if rank == 0:
            global_best_index = np.argmin(local_fitness)
            global_best_pos = local_population[global_best_index].copy()
            global_best_fitness = local_fitness[global_best_index]
        else:
            global_best_pos = np.zeros(len(bounds))
            global_best_fitness = np.inf

        # Broadcast initial global best to all ranks
        comm.Bcast(global_best_pos, root=0)
        global_best_fitness = comm.bcast(global_best_fitness, root=0)

        for _ in range(max_iter):
            r1, r2 = np.random.rand(2, local_pop_size, len(bounds))
            
            local_velocities = (w * local_velocities + 
                                c1 * r1 * (local_personal_best_pos - local_population) +
                                c2 * r2 * (global_best_pos - local_population))
            
            local_population += local_velocities
            
            # Clip to bounds
            np.clip(local_population, [b[0] for b in bounds], [b[1] for b in bounds], out=local_population)
            
            local_fitness = objective_func(local_population)
            
            # Update personal best
            improved = local_fitness < local_personal_best_fitness
            local_personal_best_pos[improved] = local_population[improved]
            local_personal_best_fitness[improved] = local_fitness[improved]

            # Update global best
            local_best_index = np.argmin(local_fitness)
            local_best_pos = local_population[local_best_index]
            local_best_fitness = local_fitness[local_best_index]

            all_best_fitness = comm.gather(local_best_fitness, root=0)
            all_best_pos = comm.gather(local_best_pos, root=0)

            if rank == 0:
                global_best_index = np.argmin(all_best_fitness)
                if all_best_fitness[global_best_index] < global_best_fitness:
                    global_best_pos = all_best_pos[global_best_index]
                    global_best_fitness = all_best_fitness[global_best_index]

            # Broadcast updated global best to all ranks
            comm.Bcast(global_best_pos, root=0)
            global_best_fitness = comm.bcast(global_best_fitness, root=0)

        # Gather final results
        final_population = comm.gather(local_population, root=0)
        final_fitness = comm.gather(local_fitness, root=0)

        if rank == 0:
            final_population = np.concatenate(final_population)
            final_fitness = np.concatenate(final_fitness)
            best_index = np.argmin(final_fitness)
            best_params = final_population[best_index]
            best_fitness = final_fitness[best_index]
        else:
            best_params = None
            best_fitness = None

        # Broadcast final results to all ranks
        best_params = comm.bcast(best_params, root=0)
        best_fitness = comm.bcast(best_fitness, root=0)

        return best_params, best_fitness

class ShuffledComplexEvolution(OptimizationAlgorithm):
    def optimize(self, objective_func, bounds, optimizer=None, **kwargs):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        class SpotpySetup(object):
            def __init__(self, objective_func, bounds):
                self.params = [spotpy.parameter.Uniform(f'x{i}', low, high) 
                               for i, (low, high) in enumerate(bounds)]
                self.objective_func = objective_func
                self.logger = logging.getLogger('SpotpySetup')

            def parameters(self):
                return spotpy.parameter.generate(self.params)

            def simulation(self, vector):
                self.logger.info(f"SpotpySetup simulation called with vector: {vector}")
                return vector  # Return the vector directly

            def evaluation(self):
                self.logger.info("SpotpySetup evaluation called")
                return [0]  # Dummy evaluation

            def objectivefunction(self, simulation, evaluation):
                self.logger.info(f"SpotpySetup objectivefunction called with simulation: {simulation}")
                
                # Check if simulation is already a list/array of parameter values
                if isinstance(simulation, (list, np.ndarray)):
                    X = np.array(simulation)
                else:
                    # If it's not, try to extract parameters as before
                    try:
                        X = np.array([simulation[0][f'x{i}'] for i in range(len(self.params))])
                    except (TypeError, IndexError):
                        # If that fails, assume simulation is a single parameter set
                        X = np.array(simulation)
                    
                # Ensure X is 2D for consistency with parallel_objective_function
                if X.ndim == 1:
                    X = X.reshape(1, -1)
                
                result = self.objective_func(X)[0]
                self.logger.info(f"SpotpySetup objectivefunction result: {result}")
                return result

        setup = SpotpySetup(objective_func, bounds)
        logger = logging.getLogger('ShuffledComplexEvolution')
        
        # Set SCE-UA parameters
        repetitions = kwargs.get('repetitions', 1000)
        ngs = kwargs.get('ngs', 2)
        npg = kwargs.get('npg', 2 * len(bounds) + 1)
        
        if rank == 0:
            logger.info(f"Starting SCE-UA optimization with repetitions={repetitions}, ngs={ngs}, npg={npg}")
            sampler = sceua(setup, dbname='SCEUA', dbformat='ram', parallel='seq')
            
            # Initialize parameters
            population = [sampler.parameter()['random'] for _ in range(ngs * npg)]
            
            # Send initial tasks to workers
            for i in range(1, min(size, len(population) + 1)):
                logger.info(f"Sending initial task to worker {i}")
                comm.send(population[i-1], dest=i, tag=0)
            
            next_param_index = size - 1
            results = []
            
            while len(results) < repetitions:
                status = MPI.Status()
                response = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                worker_rank = status.Get_source()
                logger.info(f"Received result from worker {worker_rank}: {response}")
                
                params, result = response
                results.append(result)  # Now result is already a dictionary with all necessary information
                
                # Log iteration results using the Optimizer instance
                #if optimizer:
                #    optimizer.log_iteration_results(params, result)
                
                if next_param_index < len(population):
                    # Send next parameter set to the worker
                    logger.info(f"Sending next parameter set to worker {worker_rank}")
                    comm.send(population[next_param_index], dest=worker_rank, tag=0)
                    next_param_index += 1
                else:
                    # Generate a new parameter set using SCE-UA logic
                    new_params = sampler.sample(repetitions=1)[0]
                    logger.info(f"Generated new parameter set: {new_params}")
                    comm.send(new_params, dest=worker_rank, tag=0)
            
            # Send termination signal to workers
            for i in range(1, size):
                comm.send(None, dest=i, tag=1)
            
            # Process results
            best_result = min(results, key=lambda x: x['objective'])
            best_params = best_result['params']
            best_value = best_result['objective']
            logger.info(f"Optimization complete. Best params: {best_params}, Best value: {best_value}")
            
            return best_params, best_value  # Return as a tuple
            
        else:
            # Worker processes
            while True:
                status = MPI.Status()
                params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if status.Get_tag() == 1:  # Termination signal
                    break
                logger.info(f"Worker {rank} received params: {params}")
                result = objective_func(np.array(params).reshape(1, -1))
                logger.info(f"Worker {rank} computed result: {result}")
                comm.send((params, result), dest=0, tag=0)
            
            best_params, best_value = None, None

        # Broadcast results to all ranks
        best_params = comm.bcast(best_params, root=0)
        best_value = comm.bcast(best_value, root=0)

        return best_params, best_value  # Return as a tuple

class BasinHopping(OptimizationAlgorithm):
    def optimize(self, objective_func, bounds, **kwargs):
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        
        def parallel_objective(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return objective_func(x)[0]  # Assuming objective_func can handle batches
        
        result = basinhopping(parallel_objective, x0, minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds}, **kwargs)
        return result.x, result.fun

class DynamicallyDimensionedSearch(OptimizationAlgorithm):
    def optimize(self, objective_func, bounds, **kwargs):
        def perturb_dds(x, r, bounds, iter, maxiter):
            x_new = x.copy()
            d = len(x)
            n_perturb = np.random.binomial(d, 1 - np.log(iter) / np.log(maxiter))
            perturb_indices = np.random.choice(d, n_perturb, replace=False)
            for i in perturb_indices:
                x_new[i] = x[i] + r * (bounds[i][1] - bounds[i][0]) * np.random.normal()
                x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
            return x_new

        x_best = np.array([np.random.uniform(low, high) for low, high in bounds])
        f_best = objective_func(x_best.reshape(1, -1))[0]
        r = kwargs.get('r', 0.2)
        maxiter = kwargs.get('maxiter', 1000)

        for i in range(maxiter):
            x_new = perturb_dds(x_best, r, bounds, i+1, maxiter)
            f_new = objective_func(x_new.reshape(1, -1))[0]
            if f_new < f_best:
                x_best, f_best = x_new, f_new

        return x_best, f_best

def get_optimization_algorithm(algorithm_name: str) -> OptimizationAlgorithm:
    algorithms = {
        "DE": DifferentialEvolution(),
        "PSO": ParticleSwarmOptimization(),
        "SCE-UA": ShuffledComplexEvolution(),
        "Basin-hopping": BasinHopping(),
        "DDS": DynamicallyDimensionedSearch(),
    }
    return algorithms.get(algorithm_name)

def run_nsga2(objective_func: Callable[[np.ndarray], np.ndarray],
              bounds: List[Tuple[float, float]],
              n_obj: int,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the NSGA-II multi-objective optimization algorithm.

    This function sets up and executes the NSGA-II algorithm using the provided objective function,
    parameter bounds, and number of objectives. It uses a parallel problem formulation to enable
    distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 2D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives to optimize.
        **kwargs: Additional keyword arguments for the NSGA-II algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The best parameter sets found (2D array, solutions x parameters)
            - The corresponding objective values (2D array, solutions x objectives)

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_nsga2')
    logger.info(f"NSGA-II parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_gen={kwargs.get('n_gen')}, n_offsprings={kwargs.get('n_offsprings')}")
    
    if not bounds or n_obj < 1:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = NSGA2(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen')), verbose=True)
    return res.X, res.F

class ParallelProblem(Problem):
    """
    A problem class for parallel evaluation in multi-objective optimization.

    This class wraps the objective function to enable parallel evaluation of solutions
    in the context of multi-objective optimization algorithms.

    Attributes:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to be minimized.
        logger (logging.Logger): Logger for this class.

    """

    def __init__(self, objective_func: Callable[[np.ndarray], np.ndarray],
                 n_var: int, n_obj: int, xl: List[float], xu: List[float]):
        """
        Initialize the ParallelProblem.

        Args:
            objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to be minimized.
            n_var (int): Number of decision variables.
            n_obj (int): Number of objectives.
            xl (List[float]): List of lower bounds for each decision variable.
            xu (List[float]): List of upper bounds for each decision variable.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.objective_func = objective_func
        self.logger = logging.getLogger('ParallelProblem')

    def _evaluate(self, X: np.ndarray, out: Dict[str, np.ndarray], *args: Any, **kwargs: Any) -> None:
        """
        Evaluate the objective function for a batch of solutions.

        This method is called by the optimization algorithm to evaluate a population of solutions.
        It logs the number of solutions being evaluated and calls the objective function.

        Args:
            X (np.ndarray): A 2D array of decision variables (population_size x n_var).
            out (Dict[str, np.ndarray]): A dictionary to store the output values.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        self.logger.info(f"Evaluating {len(X)} solutions")
        F = self.objective_func(X)
        out["F"] = np.array(F)

def run_nsga3(objective_func: Callable[[np.ndarray], np.ndarray],
              bounds: List[Tuple[float, float]],
              n_obj: int,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the NSGA-III multi-objective optimization algorithm.

    This function sets up and executes the NSGA-III algorithm using the provided objective function,
    parameter bounds, and number of objectives. It uses a parallel problem formulation to enable
    distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 2D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives to optimize.
        **kwargs: Additional keyword arguments for the NSGA-III algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The best parameter sets found (2D array, solutions x parameters)
            - The corresponding objective values (2D array, solutions x objectives)

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_nsga3')
    logger.info(f"NSGA-III parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_gen={kwargs.get('n_gen')}, n_offsprings={kwargs.get('n_offsprings')}")
    
    if not bounds or n_obj < 1:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    algorithm = NSGA3(ref_dirs=ref_dirs, **kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen')), verbose=True)
    return res.X, res.F

def run_moead(objective_func: Callable[[np.ndarray], np.ndarray],
              bounds: List[Tuple[float, float]],
              n_obj: int,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) optimization algorithm.

    This function sets up and executes the MOEA/D algorithm using the provided objective function,
    parameter bounds, and number of objectives. It uses a parallel problem formulation to enable
    distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 2D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives to optimize.
        **kwargs: Additional keyword arguments for the MOEA/D algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The best parameter sets found (2D array, solutions x parameters)
            - The corresponding objective values (2D array, solutions x objectives)

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_moead')
    logger.info(f"MOEA/D parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_gen={kwargs.get('n_gen')}, n_neighbors={kwargs.get('n_neighbors')}")
    
    if not bounds or n_obj < 1:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    algorithm = MOEAD(ref_dirs=ref_dirs, **kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen')), verbose=True)
    return res.X, res.F

def run_smsemoa(objective_func: Callable[[np.ndarray], np.ndarray],
                bounds: List[Tuple[float, float]],
                n_obj: int,
                **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm) optimization algorithm.

    This function sets up and executes the SMS-EMOA algorithm using the provided objective function,
    parameter bounds, and number of objectives. It uses a parallel problem formulation to enable
    distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 2D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives to optimize.
        **kwargs: Additional keyword arguments for the SMS-EMOA algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The best parameter sets found (2D array, solutions x parameters)
            - The corresponding objective values (2D array, solutions x objectives)

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_smsemoa')
    logger.info(f"SMS-EMOA parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_gen={kwargs.get('n_gen')}")
    
    if not bounds or n_obj < 1:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = SMSEMOA(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen')), verbose=True)
    return res.X, res.F

def run_mopso(objective_func: Callable[[np.ndarray], np.ndarray],
              bounds: List[Tuple[float, float]],
              n_obj: int,
              **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the MOPSO (Multi-Objective Particle Swarm Optimization) algorithm.

    This function sets up and executes the MOPSO algorithm using the provided objective function,
    parameter bounds, and number of objectives. It uses a parallel problem formulation to enable
    distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 2D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives to optimize.
        **kwargs: Additional keyword arguments for the MOPSO algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The best parameter sets found (2D array, solutions x parameters)
            - The corresponding objective values (2D array, solutions x objectives)

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_mopso')
    logger.info(f"MOPSO parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_gen={kwargs.get('n_gen')}, w={kwargs.get('w')}, c1={kwargs.get('c1')}, c2={kwargs.get('c2')}")
    
    if not bounds or n_obj < 1:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = MOPSO(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen')), verbose=True)
    return res.X, res.F

class BorgMOEA(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = kwargs.get('pop_size', 100)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.sbx_prob = kwargs.get('sbx_prob', 0.8)
        self.sbx_eta = kwargs.get('sbx_eta', 15)
        self.pm_prob = kwargs.get('pm_prob', 1.0)
        self.pm_eta = kwargs.get('pm_eta', 20)
        
        self.sampling = FloatRandomSampling()
        self.crossover = SBX(prob=self.sbx_prob, eta=self.sbx_eta)
        self.mutation = PolynomialMutation(prob=self.pm_prob, eta=self.pm_eta)

    def _initialize_infill(self):
        pop = self.sampling(self.problem, self.pop_size)
        return pop

    def _infill(self):
        # Select parents
        parents = self.pop.random(self.n_offsprings)

        # Apply crossover and mutation
        offspring = self.crossover(self.problem, parents)
        offspring = self.mutation(self.problem, offspring)

        return offspring

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.infill()

        # Evaluate the infill solutions
        self.evaluator.eval(self.problem, infills)

        # Merge the infills with the population
        self.pop = Population.merge(self.pop, infills)

        # Update the epsilon-box dominance
        self.eps = self.calculate_epsilon(self.pop)
        E = self.epsilon_dominance(self.pop.get("F"))

        # Do the non-dominated sorting
        fronts = NonDominatedSorting().do(E, sort_by_objectives=True)

        # Fill the new population
        new_pop = Population()
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop = Population.merge(new_pop, self.pop[front])
            else:
                I = self.survival.do(self.problem, self.pop[front], n_survive=self.pop_size - len(new_pop))
                new_pop = Population.merge(new_pop, self.pop[front[I]])
                break

        self.pop = new_pop

    def calculate_epsilon(self, pop):
        return np.std(pop.get("F"), axis=0) * self.epsilon

    def epsilon_dominance(self, F):
        eps = self.eps
        N = F.shape[0]
        M = F.shape[1]

        # Calculate the boxed objectives
        boxed = (F / eps).astype(int) * eps

        # Calculate the epsilon box dominance
        E = np.zeros((N, N))
        for i in range(N):
            E[i, :] = np.all(boxed[i] <= boxed, axis=1) & np.any(boxed[i] < boxed, axis=1)

        return E

class MOPSO(Algorithm):
    """
    Implementation of the Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.

    This class implements the MOPSO algorithm for multi-objective optimization problems.

    Attributes:
        pop_size (int): Population size.
        w (float): Inertia weight.
        c1 (float): Cognitive learning factor.
        c2 (float): Social learning factor.
        sampling (FloatRandomSampling): Sampling method for initial population.
        velocity (np.ndarray): Particle velocities.
        personal_best (Population): Personal best solutions for each particle.
        global_best (Population): Global best solutions.
    """

    def __init__(self, pop_size: int = 100, w: float = 0.4, c1: float = 2.0, c2: float = 2.0, **kwargs: Any):
        """
        Initialize the MOPSO algorithm.

        Args:
            pop_size (int): Population size.
            w (float): Inertia weight.
            c1 (float): Cognitive learning factor.
            c2 (float): Social learning factor.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.sampling = FloatRandomSampling()
        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.logger = logging.getLogger('MOPSO')

    def _initialize_infill(self) -> Population:
        """
        Initialize the population and related attributes.

        Returns:
            Population: The initial population.
        """
        pop = self.sampling(self.problem, self.pop_size)
        self.velocity = np.zeros((self.pop_size, self.problem.n_var))
        self.personal_best = pop.copy()
        return pop

    def _infill(self) -> Population:
        """
        Perform the infill procedure to generate new solutions.

        Returns:
            Population: The new population after applying MOPSO operators.
        """
        # Update velocity and position
        r1, r2 = np.random.rand(2, self.pop_size, self.problem.n_var)
        
        self.velocity = (self.w * self.velocity + 
                         self.c1 * r1 * (self.personal_best.get("X") - self.pop.get("X")) + 
                         self.c2 * r2 * (self.global_best[np.random.randint(len(self.global_best))].get("X") - self.pop.get("X")))
        
        X = self.pop.get("X") + self.velocity
        
        # Ensure the particles stay within bounds
        X = np.clip(X, self.problem.xl, self.problem.xu)
        
        return Population.new(X=X)

    def _advance(self, infills: Population = None, **kwargs: Any) -> None:
        """
        Advance the algorithm by one step.

        Args:
            infills (Population): The new solutions to be incorporated.
            **kwargs: Additional keyword arguments.
        """
        if infills is None:
            infills = self.infill()

        # Evaluate the infill solutions
        self.evaluator.eval(self.problem, infills)

        # Update personal best
        mask = (infills.get("F") <= self.personal_best.get("F")).all(axis=1)
        self.personal_best[mask] = infills[mask]

        # Update global best
        self.pop = Population.merge(self.pop, infills)
        non_dominated = NonDominatedSorting().do(self.pop.get("F"), only_non_dominated_front=True)
        self.global_best = self.pop[non_dominated]

        # Replace the population
        self.pop = infills

    def _set_optimum(self, X: np.ndarray, F: np.ndarray) -> None:
        """
        Set the optimum solutions found by the algorithm.

        Args:
            X (np.ndarray): Decision variables of the optimal solutions.
            F (np.ndarray): Objective values of the optimal solutions.
        """
        super()._set_optimum(X, F)
        self.logger.info(f"Set optimum with {len(X)} solutions")


def get_algorithm_kwargs(config, size: int) -> Dict[str, Any]:
    """
    Get the keyword arguments for the optimization algorithm.

    This method prepares algorithm-specific parameters based on the chosen optimization algorithm.

    Args:
        config: Configuration object containing optimization settings.
        size: Total number of processes.

    Returns:
        Dict[str, Any]: Dictionary of keyword arguments for the algorithm.
    """
    num_workers = size - 1
    population_size = max(config.get('POPULATION_SIZE'), num_workers * 2)  # Ensure at least 2 evaluations per worker
    algorithm = config.get('OPTMIZATION_ALOGORITHM')
    num_iter = config.get('NUMBER_OF_ITERATIONS')
    kwargs: Dict[str, Any] = {}

    if algorithm == "DE":
        kwargs.update({
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
            "strategy": 'best1bin',
            "tol": 0.01,
            "atol": 0,
        })
    elif algorithm == "PSO":
        kwargs.update({
            "swarmsize": population_size,
            "omega": 0.5,
            "phip": 0.5,
            "phig": 0.5,
            "maxiter": num_iter,
            "minstep": 1e-8,
            "minfunc": 1e-8,
        })
    elif algorithm == "SCE-UA":
            kwargs.update({
                "repetitions": num_iter,
                "npg": 2 * len(config.all_bounds) + 1,  # Number of points in each complex
                "kstop": 10,
                "peps": 0.001,
                "pcento": 0.1,
        })
    elif algorithm == "Basin-hopping":
        kwargs.update({
            "niter": num_iter,
            "T": 1.0,
            "stepsize": 0.5,
            "interval": 50,
            "niter_success": None,
        })
    elif algorithm == "DDS":
        kwargs.update({
            "r": config.dds_r,
            "maxiter": num_iter,
        })
    elif algorithm in ["NSGA-II", "NSGA-III"]:
        kwargs.update({
            "maxiter": num_iter,
            "pop_size": population_size,
            "n_offsprings": population_size,
            "eliminate_duplicates": True,
            "crossover": SBX(prob=0.9, eta=15),
            "mutation": PolynomialMutation(eta=20),
        })
    elif algorithm == "MOEA/D":
        kwargs.update({
            "maxiter": num_iter,
            "n_neighbors": 15,
            "prob_neighbor_mating": 0.7,
            "crossover": SBX(prob=0.9, eta=15),
            "mutation": PolynomialMutation(eta=20),
        })
    elif algorithm == "SMS-EMOA":
        kwargs.update({
            "maxiter": num_iter,
            "pop_size": population_size,
            "eliminate_duplicates": True,
            "crossover": SBX(prob=0.9, eta=15),
            "mutation": PolynomialMutation(eta=20),
        })
    elif algorithm == "Borg-MOEA":
        kwargs.update({
            "n_gen": num_iter,
            "pop_size": population_size,
            "epsilon": 0.01,
            "sbx_prob": 0.8,
            "sbx_eta": 15,
            "pm_prob": 1.0,
            "pm_eta": 20,
            "eliminate_duplicates": True,
        })
    elif algorithm == "MOPSO":
        kwargs.update({
            "maxiter": num_iter,
            "pop_size": population_size,
            "w": 0.4,
            "c1": 2.0,
            "c2": 2.0,
            "eliminate_duplicates": True,
        })

    return kwargs

def calculate_objective_value(calib_metrics: Dict[str, float], optimization_metric: str) -> Optional[float]:
    """
    Calculate the objective value based on the calibration metrics.

    Args:
        calib_metrics (Dict[str, float]): Dictionary of calibration metrics.
        optimization_metric (str): The metric to use for optimization.

    Returns:
        Optional[float]: The calculated objective value, or None if the metric is not found.
    """
    if optimization_metric not in calib_metrics:
        return None

    value = calib_metrics[optimization_metric]
    return -value if optimization_metric in ['KGE', 'KGEp', 'KGEnp', 'NSE'] else value

class CustomDE(DE):
    """
    Custom implementation of the Differential Evolution algorithm.

    This class extends the pymoo DE implementation with additional logging and customization options.

    Attributes:
        All attributes from DE class
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the CustomDE algorithm.

        Args:
            **kwargs: Keyword arguments to be passed to the DE constructor.
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger('CustomDE')

    def _set_optimum(self, X: np.ndarray, F: np.ndarray) -> None:
        """
        Set the optimum solutions found by the algorithm.

        Args:
            X (np.ndarray): Decision variables of the optimal solutions.
            F (np.ndarray): Objective values of the optimal solutions.
        """
        super()._set_optimum()
        self.logger.info(f"Set optimum with {len(self.opt)} solutions")

def run_de(objective_func: Callable[[np.ndarray], np.ndarray],
           bounds: List[Tuple[float, float]],
           **kwargs: Any) -> Tuple[np.ndarray, float]:
    """
    Run the Differential Evolution (DE) optimization algorithm.

    This function sets up and executes the DE algorithm using the provided objective function
    and parameter bounds. It uses a parallel problem formulation to enable distributed evaluation of solutions.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
            It should take a 2D numpy array of parameter sets and return a 1D numpy array of objective values.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        **kwargs: Additional keyword arguments for the DE algorithm.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing:
            - The best parameter set found (1D array)
            - The corresponding objective value (float)

    Raises:
        ValueError: If the provided bounds are invalid.
    """
    logger = logging.getLogger('run_de')
    logger.info(f"DE parameters: pop_size={kwargs.get('pop_size')}, "
                f"F={kwargs.get('F')}, CR={kwargs.get('CR')}")
    
    if not bounds:
        raise ValueError("Invalid bounds")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=1, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = CustomDE(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen', 100)), verbose=True)
    return res.X, res.F[0]

class CustomPSO(PSO):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.logger = logging.getLogger('CustomPSO')

    def _set_optimum(self) -> None:
        super()._set_optimum()
        self.logger.info(f"Set optimum with {len(self.opt)} solutions")


def run_pso(objective_func: Callable[[np.ndarray], np.ndarray],
            bounds: List[Tuple[float, float]],
            **kwargs: Any) -> Tuple[np.ndarray, float]:
    """
    Run the Particle Swarm Optimization (PSO) algorithm.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        **kwargs: Additional keyword arguments for the PSO algorithm.

    Returns:
        Tuple[np.ndarray, float]: Best parameters and corresponding objective value.

    Raises:
        ValueError: If the provided bounds are invalid.
    """
    logger = logging.getLogger('run_pso')
    logger.info(f"PSO parameters: pop_size={kwargs.get('pop_size')}, "
                f"w={kwargs.get('w')}, c1={kwargs.get('c1')}, c2={kwargs.get('c2')}")
    
    if not bounds:
        raise ValueError("Invalid bounds")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=1, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = CustomPSO(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen', 100)), verbose=True)
    return res.X, res.F[0]

class CustomSCEUA(Algorithm):
    def __init__(self, pop_size=100, n_complexes=2, n_evolution_steps=5, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.n_complexes = n_complexes
        self.n_evolution_steps = n_evolution_steps
        self.logger = logging.getLogger('CustomSCEUA')

    def _initialize_infill(self):
        pop = Population.new(X=np.random.rand(self.pop_size, self.problem.n_var))
        return pop

    def _infill(self):
        # Implement the SCE-UA algorithm logic here
        # This is a simplified version and may need to be expanded for full functionality
        new_pop = []
        for _ in range(self.n_complexes):
            complex = self.pop[np.random.choice(len(self.pop), size=self.pop_size // self.n_complexes, replace=False)]
            for _ in range(self.n_evolution_steps):
                parents = complex[np.argsort(complex.get("F")[:, 0])[:3]]
                child = self.create_child(parents)
                new_pop.append(child)
        return Population.new(X=np.array([ind.X for ind in new_pop]))

    def create_child(self, parents):
        # Implement the child creation logic here
        # This is a simplified version
        alpha = np.random.rand(3)
        alpha = alpha / np.sum(alpha)
        child = np.sum([alpha[i] * parents[i].X for i in range(3)], axis=0)
        return Population.new(X=child)

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.infill()
        self.evaluator.eval(self.problem, infills)
        self.pop = Population.merge(self.pop, infills)
        self.pop = self.pop[np.argsort(self.pop.get("F")[:, 0])[:self.pop_size]]

    def _set_optimum(self, X: np.ndarray, F: np.ndarray) -> None:
        super()._set_optimum(X, F)
        self.logger.info(f"Set optimum with {len(X)} solutions")

def run_sce_ua(objective_func: Callable[[np.ndarray], np.ndarray],
               bounds: List[Tuple[float, float]],
               **kwargs: Any) -> Tuple[np.ndarray, float]:
    """
    Run the Shuffled Complex Evolution (SCE-UA) algorithm.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        **kwargs: Additional keyword arguments for the SCE-UA algorithm.

    Returns:
        Tuple[np.ndarray, float]: Best parameters and corresponding objective value.

    Raises:
        ValueError: If the provided bounds are invalid.
    """
    logger = logging.getLogger('run_sce_ua')
    logger.info(f"SCE-UA parameters: pop_size={kwargs.get('pop_size')}, "
                f"n_complexes={kwargs.get('n_complexes')}")
    
    if not bounds:
        raise ValueError("Invalid bounds")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=1, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = CustomSCEUA(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen', 100)), verbose=True)
    return res.X, res.F[0]

def run_basin_hopping(objective_func: Callable[[np.ndarray], np.ndarray],
                      bounds: List[Tuple[float, float]],
                      **kwargs: Any) -> Tuple[np.ndarray, float]:
    """
    Run the Basin-hopping algorithm.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        **kwargs: Additional keyword arguments for the Basin-hopping algorithm.

    Returns:
        Tuple[np.ndarray, float]: Best parameters and corresponding objective value.

    Raises:
        ValueError: If the provided bounds are invalid.
    """
    logger = logging.getLogger('run_basin_hopping')
    logger.info(f"Basin-hopping parameters: niter={kwargs.get('niter')}, "
                f"T={kwargs.get('T')}, stepsize={kwargs.get('stepsize')}")
    
    if not bounds:
        raise ValueError("Invalid bounds")

    def wrapper_func(x):
        return objective_func(np.array([x]))[0]

    x0 = np.mean(bounds, axis=1)
    res = basinhopping(wrapper_func, x0, niter=kwargs.get('niter', 100),
                       T=kwargs.get('T', 1.0), stepsize=kwargs.get('stepsize', 0.5),
                       minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds})
    
    return res.x, res.fun

def run_dds(objective_func: Callable[[np.ndarray], np.ndarray],
            bounds: List[Tuple[float, float]],
            **kwargs: Any) -> Tuple[np.ndarray, float]:
    """
    Run the Dynamically Dimensioned Search (DDS) algorithm.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        **kwargs: Additional keyword arguments for the DDS algorithm.

    Returns:
        Tuple[np.ndarray, float]: Best parameters and corresponding objective value.

    Raises:
        ValueError: If the provided bounds are invalid.
    """
    logger = logging.getLogger('run_dds')
    logger.info(f"DDS parameters: max_iter={kwargs.get('max_iter')}, "
                f"r={kwargs.get('r')}")
    
    if not bounds:
        raise ValueError("Invalid bounds")

    def perturb_dds(x, r, bounds, iter, max_iter):
        x_new = x.copy()
        d = len(x)
        n_perturb = np.random.binomial(d, 1 - np.log(iter) / np.log(max_iter))
        perturb_indices = np.random.choice(d, n_perturb, replace=False)
        for i in perturb_indices:
            x_new[i] = x[i] + r * (bounds[i][1] - bounds[i][0]) * np.random.normal()
            x_new[i] = np.clip(x_new[i], bounds[i][0], bounds[i][1])
        return x_new

    x_best = np.array([np.random.uniform(low, high) for low, high in bounds])
    f_best = objective_func(x_best.reshape(1, -1))[0]
    r = kwargs.get('r', 0.2)
    max_iter = kwargs.get('max_iter', 1000)

    for i in range(max_iter):
        x_new = perturb_dds(x_best, r, bounds, i+1, max_iter)
        f_new = objective_func(x_new.reshape(1, -1))[0]
        if f_new < f_best:
            x_best, f_best = x_new, f_new

    return x_best, f_best

class CustomBorgMOEA(Algorithm):
    def __init__(self, 
                 pop_size: int = 100,
                 epsilon: float = 0.01,
                 sbx_prob: float = 0.8,
                 sbx_eta: float = 15,
                 pm_prob: float = 1.0,
                 pm_eta: float = 20,
                 **kwargs: Any):
        """
        Initialize the CustomBorgMOEA algorithm.

        Args:
            pop_size (int): Population size.
            epsilon (float): Epsilon value for epsilon-dominance archive.
            sbx_prob (float): Simulated Binary Crossover probability.
            sbx_eta (float): Simulated Binary Crossover distribution index.
            pm_prob (float): Polynomial Mutation probability.
            pm_eta (float): Polynomial Mutation distribution index.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.epsilon = epsilon
        
        self.sampling = FloatRandomSampling()
        self.crossover = SBX(prob=sbx_prob, eta=sbx_eta)
        self.mutation = PolynomialMutation(prob=pm_prob, eta=pm_eta)
        
        self.logger = logging.getLogger('CustomBorgMOEA')

    def _initialize_infill(self):
        pop = self.sampling(self.problem, self.pop_size)
        return pop

    def _infill(self):
        # Select parents
        parents = self.pop.random(self.n_offsprings)

        # Apply crossover and mutation
        offspring = self.crossover(self.problem, parents)
        offspring = self.mutation(self.problem, offspring)

        return offspring

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.infill()

        # Evaluate the infill solutions
        self.evaluator.eval(self.problem, infills)

        # Merge the infills with the population
        self.pop = Population.merge(self.pop, infills)

        # Update the epsilon-box dominance
        self.eps = self.calculate_epsilon(self.pop)
        E = self.epsilon_dominance(self.pop.get("F"))

        # Do the non-dominated sorting
        fronts = NonDominatedSorting().do(E, sort_by_objectives=True)

        # Fill the new population
        new_pop = Population()
        for front in fronts:
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop = Population.merge(new_pop, self.pop[front])
            else:
                I = self.survival.do(self.problem, self.pop[front], n_survive=self.pop_size - len(new_pop))
                new_pop = Population.merge(new_pop, self.pop[front[I]])
                break

        self.pop = new_pop

    def calculate_epsilon(self, pop):
        return np.std(pop.get("F"), axis=0) * self.epsilon

    def epsilon_dominance(self, F):
        eps = self.eps
        N = F.shape[0]
        M = F.shape[1]

        # Calculate the boxed objectives
        boxed = (F / eps).astype(int) * eps

        # Calculate the epsilon box dominance
        E = np.zeros((N, N))
        for i in range(N):
            E[i, :] = np.all(boxed[i] <= boxed, axis=1) & np.any(boxed[i] < boxed, axis=1)

        return E

    def _set_optimum(self, X: np.ndarray, F: np.ndarray) -> None:
        super()._set_optimum(X, F)
        self.logger.info(f"Set optimum with {len(X)} solutions")

def run_borg_moea(objective_func: Callable[[np.ndarray], np.ndarray],
                  bounds: List[Tuple[float, float]],
                  n_obj: int,
                  **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the Borg-MOEA algorithm.

    Args:
        objective_func (Callable[[np.ndarray], np.ndarray]): The objective function to minimize.
        bounds (List[Tuple[float, float]]): List of (lower, upper) bounds for each parameter.
        n_obj (int): Number of objectives.
        **kwargs: Additional keyword arguments for the Borg-MOEA algorithm.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pareto optimal solutions and their objective values.

    Raises:
        ValueError: If the provided bounds or n_obj are invalid.
    """
    logger = logging.getLogger('run_borg_moea')
    logger.info(f"Borg-MOEA parameters: pop_size={kwargs.get('pop_size')}, "
                f"epsilon={kwargs.get('epsilon')}, n_gen={kwargs.get('n_gen')}")
    
    if not bounds or n_obj < 2:
        raise ValueError("Invalid bounds or number of objectives")

    problem = ParallelProblem(objective_func, n_var=len(bounds), n_obj=n_obj, 
                              xl=[b[0] for b in bounds], xu=[b[1] for b in bounds])
    
    algorithm = CustomBorgMOEA(**kwargs)
    
    res = minimize(problem, algorithm, ('n_gen', kwargs.get('n_gen', 100)), verbose=True)
    return res.X, res.F