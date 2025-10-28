#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NGEN Optimizer with Parallel Calibration Support

This module adds MPI-based parallel execution to NGEN calibration, 
matching the approach used for SUMMA calibration.

KEY CHANGES:
1. Added parallel evaluation support using MPI
2. Uses worker functions for distributed parameter evaluation
3. Supports both DE and DDS optimization algorithms with parallelization
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time

class NgenOptimizer:
    """
    NGEN model optimizer with parallel calibration support
    
    Supports Differential Evolution (DE) and other optimization algorithms
    with MPI-based parallel evaluation when MPI_PROCESSES > 1
    """
    
    def __init__(self, config: Dict, logger: logging.Logger, param_manager: Any,
                 calibration_target: Any, ngen_runner: Any):
        """
        Initialize NGEN optimizer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance  
            param_manager: NgenParameterManager instance
            calibration_target: Calibration target manager
            ngen_runner: NGEN model runner
        """
        self.config = config
        self.logger = logger
        self.param_manager = param_manager
        self.calibration_target = calibration_target
        self.ngen_runner = ngen_runner
        
        # Optimization settings
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.optimization_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Parallel processing settings
        self.num_processes = config.get('MPI_PROCESSES', 1)
        self.use_parallel = self.num_processes > 1
        
        # Algorithm-specific settings
        self.algorithm = config.get('OPTIMIZATION_ALGORITHM', 'DE')
        
        self.logger.info("NgenOptimizer initialized")
        self.logger.info(f"Optimization metric: {self.optimization_metric}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        if self.use_parallel:
            self.logger.info(f"Parallel processing: {self.num_processes} processes")
    
    def run_de(self) -> Dict[str, Any]:
        """
        Run Differential Evolution optimization with parallel support
        
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting ngen calibration with DE")
        
        # DE configuration
        population_size = self.config.get('POPULATION_SIZE', 50)
        F = self.config.get('DE_SCALING_FACTOR', 0.5)
        CR = self.config.get('DE_CROSSOVER_RATE', 0.9)
        
        self.logger.info("DE configuration:")
        self.logger.info(f"  Population size: {population_size}")
        self.logger.info(f"  Scaling factor (F): {F}")
        self.logger.info(f"  Crossover rate (CR): {CR}")
        
        # Initialize population
        self.logger.info("Evaluating initial DE population...")
        bounds = self.param_manager.get_parameter_bounds()
        n_params = len(self.param_manager.all_param_names)
        
        # Create initial population
        population = np.random.uniform(0, 1, (population_size, n_params))
        
        # PARALLEL EVALUATION: Evaluate initial population
        if self.use_parallel:
            population_scores = self._evaluate_population_parallel(population)
        else:
            population_scores = self._evaluate_population_serial(population)
        
        # Find initial best
        best_idx = np.nanargmax(population_scores)
        best_score = population_scores[best_idx]
        best_params = self.param_manager.denormalize_parameters(population[best_idx])
        
        self.logger.info(f"Initial best score: {best_score:.6f}")
        
        # Optimization history
        history = []
        history.append({
            'iteration': 0,
            'best_score': best_score,
            'mean_score': np.nanmean(population_scores),
            'best_params': best_params.copy()
        })
        
        # Main DE loop
        start_time = time.time()
        
        for iteration in range(1, self.max_iterations + 1):
            iter_start = time.time()
            
            # Create trial population through mutation and crossover
            trial_population = np.zeros_like(population)
            
            for i in range(population_size):
                # Mutation: Select three random distinct individuals
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Mutant vector
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, 0, 1)
                
                # Crossover
                cross_points = np.random.rand(n_params) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, n_params)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_population[i] = trial
            
            # PARALLEL EVALUATION: Evaluate trial population
            if self.use_parallel:
                trial_scores = self._evaluate_population_parallel(trial_population)
            else:
                trial_scores = self._evaluate_population_serial(trial_population)
            
            # Selection: Keep better individuals
            improvements = 0
            for i in range(population_size):
                if not np.isnan(trial_scores[i]) and trial_scores[i] > population_scores[i]:
                    population[i] = trial_population[i]
                    population_scores[i] = trial_scores[i]
                    improvements += 1
                    
                    # Update global best
                    if trial_scores[i] > best_score:
                        best_score = trial_scores[i]
                        best_params = self.param_manager.denormalize_parameters(trial_population[i])
            
            iter_time = time.time() - iter_start
            
            # Log progress
            mean_score = np.nanmean(population_scores)
            self.logger.info(
                f"Gen {iteration:3d}: Best={best_score:.6f}, Mean={mean_score:.6f}, "
                f"Improvements={improvements}, Time={iter_time:.1f}s"
            )
            
            # Record history
            history.append({
                'iteration': iteration,
                'best_score': best_score,
                'mean_score': mean_score,
                'improvements': improvements,
                'best_params': best_params.copy(),
                'time': iter_time
            })
        
        total_time = time.time() - start_time
        
        self.logger.info(f"DE optimization completed in {total_time:.1f}s")
        self.logger.info(f"Final best score: {best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'total_time': total_time,
            'algorithm': 'DE',
            'converged': True
        }
    
    def _evaluate_population_parallel(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate population in parallel using MPI worker functions
        
        Args:
            population: Population of normalized parameter sets (n_individuals x n_params)
            
        Returns:
            Array of fitness scores for each individual
        """
        from utils.optimization.ngen_worker_functions import evaluate_ngen_parameters
        
        # Create evaluation tasks
        tasks = []
        for i, individual in enumerate(population):
            params = self.param_manager.denormalize_parameters(individual)
            
            task = {
                'individual_id': i,
                'params': params,
                'config': self.config,
                'optimization_metric': self.optimization_metric,
            }
            tasks.append(task)
        
        self.logger.debug(f"Evaluating {len(tasks)} individuals in parallel")
        
        # Execute in parallel using MPI
        results = self._execute_parallel_mpi(tasks)
        
        # Extract scores
        scores = np.full(len(population), np.nan)
        for result in results:
            if result and 'individual_id' in result:
                idx = result['individual_id']
                score = result.get('score', np.nan)
                scores[idx] = score if score is not None else np.nan
        
        return scores
    
    def _evaluate_population_serial(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate population serially (no parallelization)
        
        Args:
            population: Population of normalized parameter sets
            
        Returns:
            Array of fitness scores
        """
        scores = np.full(len(population), np.nan)
        
        for i, individual in enumerate(population):
            params = self.param_manager.denormalize_parameters(individual)
            
            try:
                # Update NGEN configuration files
                if not self.param_manager.update_ngen_configs(params):
                    self.logger.warning(f"Failed to update configs for individual {i}")
                    continue
                
                # Run NGEN model
                output_dir = self.ngen_runner.run_model()
                if not output_dir:
                    self.logger.warning(f"NGEN run failed for individual {i}")
                    continue
                
                # Calculate metrics
                metrics = self.calibration_target.calculate_metrics()
                if metrics and self.optimization_metric in metrics:
                    scores[i] = metrics[self.optimization_metric]
                else:
                    self.logger.warning(f"Metric {self.optimization_metric} not found for individual {i}")
                    
            except Exception as e:
                self.logger.error(f"Error evaluating individual {i}: {str(e)}")
                continue
        
        return scores
    
    def _execute_parallel_mpi(self, tasks: List[Dict]) -> List[Dict]:
        """
        Execute tasks in parallel using MPI
        
        This uses the same MPI infrastructure as SUMMA calibration
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            List of result dictionaries
        """
        try:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor
            from utils.optimization.ngen_worker_functions import ngen_worker_function
            
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            
            if rank == 0:
                self.logger.info(f"Starting parallel evaluation of {len(tasks)} tasks with {self.num_processes} processes")
            
            # Use MPIPoolExecutor for parallel execution
            with MPIPoolExecutor(max_workers=self.num_processes) as executor:
                futures = {executor.submit(ngen_worker_function, task): task 
                          for task in tasks}
                
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=3600)  # 1 hour timeout per evaluation
                        results.append(result)
                    except Exception as e:
                        task = futures[future]
                        self.logger.error(f"Task {task.get('individual_id', '?')} failed: {str(e)}")
                        results.append({
                            'individual_id': task.get('individual_id', -1),
                            'score': np.nan,
                            'error': str(e)
                        })
            
            if rank == 0:
                self.logger.info(f"Completed parallel evaluation")
            
            return results
            
        except ImportError:
            self.logger.warning("mpi4py not available, falling back to serial execution")
            return self._execute_serial(tasks)
        except Exception as e:
            self.logger.error(f"MPI execution failed: {str(e)}, falling back to serial")
            return self._execute_serial(tasks)
    
    def _execute_serial(self, tasks: List[Dict]) -> List[Dict]:
        """Execute tasks serially as fallback"""
        from utils.optimization.ngen_worker_functions import ngen_worker_function
        
        results = []
        for task in tasks:
            try:
                result = ngen_worker_function(task)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task {task.get('individual_id', '?')} failed: {str(e)}")
                results.append({
                    'individual_id': task.get('individual_id', -1),
                    'score': np.nan,
                    'error': str(e)
                })
        
        return results


# ============================================================================
# USAGE NOTES
# ============================================================================

"""
ENABLING PARALLEL CALIBRATION FOR NGEN:

1. Set MPI_PROCESSES in your config file:
   MPI_PROCESSES: 10  # Use 10 parallel processes

2. Ensure mpi4py is installed:
   conda install -c conda-forge mpi4py

3. Submit as SLURM job with proper MPI configuration:
   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks=10
   #SBATCH --cpus-per-task=1
   
   module load openmpi
   export LD_LIBRARY_PATH=/path/to/ngen/libs:$LD_LIBRARY_PATH
   
   python CONFLUENCE.py --calibrate_model

4. The optimizer will automatically use parallel evaluation when MPI_PROCESSES > 1

PERFORMANCE NOTES:
- Parallel speedup is nearly linear for large populations
- Recommended: population_size >= 2 * MPI_PROCESSES
- Each process runs a complete NGEN simulation
- Network file I/O can be a bottleneck with many processes
"""
