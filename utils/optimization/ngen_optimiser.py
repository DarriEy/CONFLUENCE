#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Optimizer

Handles ngen model calibration using various optimization algorithms:
- DDS (Dynamically Dimensioned Search)
- PSO (Particle Swarm Optimization)
- SCE-UA (Shuffled Complex Evolution - University of Arizona)

Integrates with the existing SYMFLUENCE optimization framework.

Author: SYMFLUENCE Development Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import traceback
import torch
import torch.optim as optim
from tqdm import tqdm
import shutil

from ngen_parameter_manager import NgenParameterManager
from ngen_calibration_targets import NgenCalibrationTarget, NgenStreamflowTarget
from ngen_worker_functions import (
    _apply_ngen_parameters_worker,
    _run_ngen_worker,
    _calculate_ngen_metrics_worker,
    _evaluate_ngen_parameters_worker
)


class NgenOptimizer:
    """Main ngen optimization class that coordinates calibration"""
    
    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        """
        Initialize ngen optimizer.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
            optimization_settings_dir: Path to optimization settings directory
        """
        self.config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Initialize core components
        self.param_manager = NgenParameterManager(config, logger, optimization_settings_dir)
        self.calibration_target = self._create_calibration_target()
        
        # Optimization settings
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.optimization_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # ngen execution settings
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        self.ngen_sim_dir = project_dir / 'simulations' / self.experiment_id / 'ngen'
        self.ngen_setup_dir = project_dir / 'settings' / 'ngen'
        
        # Results tracking
        self.results_dir = Path(config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}" / "optimisation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.best_params = None
        self.best_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf
        self.iteration_results = []
        
        # Track failures for error reporting
        self.consecutive_failures = 0
        self.max_error_reports = 3  # Only report detailed errors for first 3 failures
        
        # Parallel processing setup (similar to SUMMA calibration)
        self.use_parallel = config.get('MPI_PROCESSES', 1) > 1
        self.num_processes = max(1, config.get('MPI_PROCESSES', 1))
        self.parallel_dirs = []
        self._consecutive_parallel_failures = 0
        
        if self.use_parallel:
            self.logger.info(f"Setting up parallel processing with {self.num_processes} MPI processes")
            self._setup_parallel_processing()
        
        self.logger.info("NgenOptimizer initialized")
        self.logger.info(f"Optimization metric: {self.optimization_metric}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        if self.use_parallel:
            self.logger.info(f"Parallel mode: {self.num_processes} processes")
        else:
            self.logger.info("Serial mode: 1 process")
    
    def _create_calibration_target(self) -> Any:
        """Factory method to create appropriate ngen calibration target"""
        optimization_target = self.config.get('OPTIMISATION_TARGET', 'streamflow')
        
        # Get project directory for calibration target
        data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        
        if optimization_target in ['streamflow', 'discharge', 'flow']:
            return NgenStreamflowTarget(self.config, project_dir, self.logger)
        # Future: Add snow, ET, etc. targets here
        else:
            self.logger.warning(f"Unknown target {optimization_target}, defaulting to streamflow")
            return NgenStreamflowTarget(self.config, project_dir, self.logger)
    
    def _setup_parallel_processing(self) -> None:
        """Setup parallel processing directories and files (similar to SUMMA calibration)"""
        self.logger.info(f"Setting up parallel processing with {self.num_processes} processes")
        
        data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        optimization_dir = project_dir / "simulations" / self.experiment_id
        
        for proc_id in range(self.num_processes):
            # Create process-specific directories (parallel_proc_00, parallel_proc_01, etc.)
            proc_base_dir = optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_ngen_dir = proc_base_dir / "ngen"
            proc_settings_dir = proc_base_dir / "settings" / "ngen"
            
            # Create directories
            for directory in [proc_base_dir, proc_ngen_dir, proc_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create log directories
            (proc_ngen_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            # Copy settings files to process directory
            self._copy_settings_to_process_dir(proc_settings_dir)
            
            # Store directory info for later use
            proc_dirs = {
                'base_dir': proc_base_dir,
                'ngen_dir': proc_ngen_dir,
                'settings_dir': proc_settings_dir,
                'proc_id': proc_id
            }
            self.parallel_dirs.append(proc_dirs)
            
            self.logger.debug(f"Created parallel directory for process {proc_id}: {proc_base_dir}")
        
        self.logger.info(f"Parallel processing setup complete: {len(self.parallel_dirs)} process directories created")
    
    def _copy_settings_to_process_dir(self, proc_settings_dir: Path) -> None:
        """Copy ngen settings files to a process-specific directory"""
        source_settings_dir = self.ngen_setup_dir
        
        if not source_settings_dir.exists():
            self.logger.warning(f"Source settings directory not found: {source_settings_dir}")
            return
        
        # Copy all ngen configuration files (realization config, catchment config, routing config, etc.)
        for item in source_settings_dir.iterdir():
            if item.is_file():
                dest_file = proc_settings_dir / item.name
                shutil.copy2(item, dest_file)
                self.logger.debug(f"Copied {item.name} to process settings directory")
            elif item.is_dir():
                # Copy directories recursively (e.g., for forcing data links)
                dest_dir = proc_settings_dir / item.name
                if not dest_dir.exists():
                    shutil.copytree(item, dest_dir)
                    self.logger.debug(f"Copied directory {item.name} to process settings directory")
    
    def _cleanup_parallel_processing(self) -> None:
        """Clean up parallel processing resources"""
        if self.use_parallel:
            self.logger.info("Cleaning up parallel processing directories")
            # Optional: Could remove parallel directories if desired
            # For now, we keep them for debugging purposes
    
    def _evaluate_parameters(self, params: Dict[str, float], proc_id: int = 0) -> float:
        """
        Evaluate a parameter set.
        
        Args:
            params: Dictionary of parameters
            proc_id: Process ID for parallel processing (default 0 for serial mode)
            
        Returns:
            Fitness value
        """
        try:
            # Get appropriate directories based on parallel/serial mode
            if self.use_parallel and proc_id < len(self.parallel_dirs):
                proc_info = self.parallel_dirs[proc_id]
                settings_dir = proc_info['settings_dir']
                ngen_sim_dir = proc_info['ngen_dir']
                
                # Create a modified config for this process
                proc_config = self.config.copy()
                proc_config['_proc_ngen_dir'] = str(ngen_sim_dir)
                proc_config['_proc_settings_dir'] = str(settings_dir)
                proc_config['_proc_id'] = proc_id
                
                # Use process-specific parameter manager
                param_manager = NgenParameterManager(proc_config, self.logger, settings_dir)
            else:
                # Serial mode or proc_id 0 - use main directories
                settings_dir = self.optimization_settings_dir
                ngen_sim_dir = self.ngen_sim_dir
                proc_config = self.config
                param_manager = self.param_manager
            
            # Apply parameters
            if not param_manager.update_config_files(params):
                if self.consecutive_failures < self.max_error_reports:
                    self.logger.error(f"Failed to update ngen config files (proc {proc_id})")
                self.consecutive_failures += 1
                return -999.0
            
            # Run ngen (errors are now logged in the worker function)
            if not _run_ngen_worker(proc_config):
                if self.consecutive_failures < self.max_error_reports:
                    self.logger.error(f"Failed to run ngen model (proc {proc_id}, see detailed error above)")
                elif self.consecutive_failures == self.max_error_reports:
                    self.logger.error(f"Suppressing further error messages after {self.max_error_reports} failures...")
                self.consecutive_failures += 1
                return -999.0
            
            # Calculate metrics using process-specific output if in parallel mode
            if self.use_parallel and proc_id > 0:
                # For parallel mode, metrics calculation needs to use process-specific output
                # You may need to modify calibration_target.calculate_metrics to accept output_dir
                metrics = self.calibration_target.calculate_metrics(self.experiment_id, output_dir=ngen_sim_dir)
            else:
                metrics = self.calibration_target.calculate_metrics(self.experiment_id)
            
            metric_name = self.optimization_metric.upper()
            if metric_name in metrics:
                fitness = metrics[metric_name]
                # Reset failure counter on success
                if self.consecutive_failures > 0:
                    self.logger.info(f"Model run successful after {self.consecutive_failures} failures")
                    self.consecutive_failures = 0
                self.logger.info(f"Fitness (proc {proc_id}): {metric_name}={fitness:.4f}")
                return fitness
            else:
                self.logger.error(f"Metric {metric_name} not in results")
                return -999.0
                
        except Exception as e:
            self.logger.error(f"Error evaluating parameters (proc {proc_id}): {e}")
            if self.consecutive_failures < self.max_error_reports:
                traceback.print_exc()
            return -999.0
    
    def run_dds(self) -> Path:
        """
        Run Dynamically Dimensioned Search (DDS) for ngen.
        
        Returns:
            Path to results file
        """
        self.logger.info("Starting ngen calibration with DDS")
        
        # DDS parameters
        r = self.config.get('DDS_R', 0.2)  # Perturbation size parameter
        
        # Get parameter info
        param_names = self.param_manager.all_param_names
        param_bounds = self.param_manager.get_parameter_bounds()
        n_params = len(param_names)
        
        self.logger.info(f"DDS parameters: r={r}, n_params={n_params}")
        
        # Initialize with default parameters (middle of bounds)
        current_params = self.param_manager.get_default_parameters()
        current_fitness = self._evaluate_parameters(current_params)
        
        # Track best
        best_params = current_params.copy()
        best_fitness = current_fitness
        
        self.logger.info(f"Initial fitness: {self.optimization_metric}={current_fitness:.4f}")
        
        # Store initial result
        self.iteration_results.append({
            'iteration': 0,
            'fitness': current_fitness,
            'parameters': current_params
        })
        
        # DDS iterations
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"=== DDS Iteration {iteration}/{self.max_iterations} ===")
            
            # Calculate probability of including each parameter
            prob_include = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            
            # Generate trial parameters
            trial_params = current_params.copy()
            
            # Select parameters to perturb
            include_param = np.random.random(n_params) < prob_include
            
            # Ensure at least one parameter is perturbed
            if not np.any(include_param):
                include_param[np.random.randint(n_params)] = True
            
            # Perturb selected parameters
            for i, (param_name, include) in enumerate(zip(param_names, include_param)):
                if include:
                    bounds = param_bounds[param_name]
                    current_value = current_params[param_name]
                    param_range = bounds['max'] - bounds['min']
                    
                    # Generate perturbation
                    perturbation = r * param_range * np.random.randn()
                    new_value = current_value + perturbation
                    
                    # Reflect at boundaries
                    while new_value < bounds['min'] or new_value > bounds['max']:
                        if new_value < bounds['min']:
                            new_value = bounds['min'] + (bounds['min'] - new_value)
                        if new_value > bounds['max']:
                            new_value = bounds['max'] - (new_value - bounds['max'])
                    
                    trial_params[param_name] = new_value
            
            # Evaluate trial parameters
            trial_fitness = self._evaluate_parameters(trial_params)
            
            # Store result
            self.iteration_results.append({
                'iteration': iteration,
                'fitness': trial_fitness,
                'parameters': trial_params
            })
            
            # Check if better than current
            is_better = (trial_fitness > current_fitness if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                        else trial_fitness < current_fitness)
            
            if is_better:
                current_params = trial_params.copy()
                current_fitness = trial_fitness
                self.logger.info(f"✓ Improved! New {self.optimization_metric}={current_fitness:.4f}")
                
                # Check if better than best
                is_best = (trial_fitness > best_fitness if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                          else trial_fitness < best_fitness)
                
                if is_best:
                    best_params = trial_params.copy()
                    best_fitness = trial_fitness
                    self.logger.info(f"★ New best! {self.optimization_metric}={best_fitness:.4f}")
            else:
                self.logger.info(f"No improvement. Current {self.optimization_metric}={current_fitness:.4f}")
        
        # Store best results
        self.best_params = best_params
        self.best_fitness = best_fitness
        
        return self._save_results('DDS')
    
    def run_pso(self) -> Path:
        """
        Run Particle Swarm Optimization (PSO) for ngen.
        
        Returns:
            Path to results file
        """
        self.logger.info("Starting ngen calibration with PSO")
        
        # PSO parameters
        n_particles = self.config.get('NUMBER_OF_PARTICLES', 20)
        w = self.config.get('PSO_INERTIA_WEIGHT', 0.7)
        c1 = self.config.get('PSO_COGNITIVE_WEIGHT', 1.5)
        c2 = self.config.get('PSO_SOCIAL_WEIGHT', 1.5)
        
        # Get parameter info
        param_names = self.param_manager.all_param_names
        param_bounds = self.param_manager.get_parameter_bounds()
        n_params = len(param_names)
        
        self.logger.info(f"PSO parameters: n_particles={n_particles}, w={w}, c1={c1}, c2={c2}")
        
        # Initialize particles (normalized [0,1])
        particles = np.random.random((n_particles, n_params))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_params))
        
        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_fitness = np.full(n_particles, -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf)
        
        global_best_position = None
        global_best_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf
        
        # Evaluate initial particles
        self.logger.info("Evaluating initial particle swarm...")
        for i, particle in enumerate(particles):
            params = self.param_manager.denormalize_parameters(particle)
            fitness = self._evaluate_parameters(params)
            
            personal_best_fitness[i] = fitness
            
            is_better = (fitness > global_best_fitness if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                        else fitness < global_best_fitness)
            
            if is_better:
                global_best_fitness = fitness
                global_best_position = particle.copy()
                self.best_fitness = fitness
                self.best_params = params
            
            self.logger.info(f"Particle {i}: {self.optimization_metric}={fitness:.4f}")
            
            self.iteration_results.append({
                'iteration': 0,
                'particle': i,
                'fitness': fitness,
                'parameters': params
            })
        
        # PSO iterations
        for iteration in range(self.max_iterations):
            self.logger.info(f"=== PSO Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Evaluate all particles
            for i in range(n_particles):
                try:
                    # Denormalize and evaluate
                    params = self.param_manager.denormalize_parameters(particles[i])
                    fitness = self._evaluate_parameters(params)
                    
                    # Update personal best
                    is_better_personal = (fitness > personal_best_fitness[i] if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                                        else fitness < personal_best_fitness[i])
                    
                    if is_better_personal:
                        personal_best_fitness[i] = fitness
                        personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    is_better_global = (fitness > global_best_fitness if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                                      else fitness < global_best_fitness)
                    
                    if is_better_global:
                        global_best_fitness = fitness
                        global_best_position = particles[i].copy()
                        self.best_fitness = fitness
                        self.best_params = params
                    
                    self.logger.info(f"Particle {i}: {self.optimization_metric}={fitness:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating particle {i}: {str(e)}")
                    fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf
                
                # Store results
                self.iteration_results.append({
                    'iteration': iteration + 1,
                    'particle': i,
                    'fitness': fitness,
                    'parameters': self.param_manager.denormalize_parameters(particles[i])
                })
            
            # Update velocities and positions
            if global_best_position is not None:
                for i in range(n_particles):
                    r1, r2 = np.random.random(n_params), np.random.random(n_params)
                    
                    velocities[i] = (w * velocities[i] + 
                                   c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                   c2 * r2 * (global_best_position - particles[i]))
                    
                    particles[i] += velocities[i]
                    particles[i] = np.clip(particles[i], 0, 1)
            
            self.logger.info(f"Iteration {iteration + 1} complete. Best {self.optimization_metric}: {global_best_fitness:.4f}")
        
        return self._save_results('PSO')
    
    def run_sce(self) -> Path:
        """
        Run Shuffled Complex Evolution - University of Arizona (SCE-UA) for ngen.
        
        Returns:
            Path to results file
        """
        self.logger.info("Starting ngen calibration with SCE-UA")
        
        # SCE-UA parameters
        self.num_complexes = self.config.get('NUMBER_OF_COMPLEXES', 2)
        self.points_per_subcomplex = self.config.get('POINTS_PER_SUBCOMPLEX', 5)
        self.num_evolution_steps = self.config.get('NUMBER_OF_EVOLUTION_STEPS', 20)
        self.evolution_stagnation = self.config.get('EVOLUTION_STAGNATION', 5)
        self.percent_change_threshold = self.config.get('PERCENT_CHANGE_THRESHOLD', 0.01)
        
        # Calculate points per complex (SCE-UA rule: 2n+1 where n is number of parameters)
        param_count = len(self.param_manager.all_param_names)
        recommended_points = 2 * param_count + 1
        min_points = self.points_per_subcomplex * 2  # At least 2 subcomplexes
        self.points_per_complex = max(recommended_points, min_points)
        
        # Calculate total population size
        self.population_size = self.num_complexes * self.points_per_complex
        
        self.logger.info(f"SCE-UA parameters:")
        self.logger.info(f"  Number of complexes: {self.num_complexes}")
        self.logger.info(f"  Points per complex: {self.points_per_complex}")
        self.logger.info(f"  Total population: {self.population_size}")
        
        # Initialize population (normalized [0,1])
        population = np.random.random((self.population_size, param_count))
        fitness_values = np.full(self.population_size, -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        for i in range(self.population_size):
            params = self.param_manager.denormalize_parameters(population[i])
            fitness = self._evaluate_parameters(params)
            fitness_values[i] = fitness
            
            if i == 0 or ((fitness > self.best_fitness) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] 
                         else (fitness < self.best_fitness)):
                self.best_fitness = fitness
                self.best_params = params
            
            self.iteration_results.append({
                'iteration': 0,
                'individual': i,
                'fitness': fitness,
                'parameters': params
            })
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Evaluated {i+1}/{self.population_size} individuals")
        
        # SCE-UA evolution
        stagnation_count = 0
        previous_best = self.best_fitness
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"=== SCE-UA Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Sort population by fitness
            if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']:
                sorted_indices = np.argsort(fitness_values)[::-1]  # Descending
            else:
                sorted_indices = np.argsort(fitness_values)  # Ascending
            
            population = population[sorted_indices]
            fitness_values = fitness_values[sorted_indices]
            
            # Partition into complexes
            complexes = []
            for c in range(self.num_complexes):
                complex_indices = list(range(c, self.population_size, self.num_complexes))
                complex_pop = population[complex_indices]
                complex_fit = fitness_values[complex_indices]
                complexes.append((complex_pop, complex_fit))
            
            # Evolve each complex
            for c, (complex_pop, complex_fit) in enumerate(complexes):
                # Evolve complex using Competitive Complex Evolution (CCE)
                evolved_pop, evolved_fit = self._evolve_complex_sce(complex_pop, complex_fit)
                
                # Update population
                complex_indices = list(range(c, self.population_size, self.num_complexes))
                population[complex_indices] = evolved_pop
                fitness_values[complex_indices] = evolved_fit
            
            # Update best
            best_idx = 0 if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.argmin(fitness_values)
            current_best = fitness_values[best_idx]
            
            if ((current_best > self.best_fitness) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                else (current_best < self.best_fitness)):
                self.best_fitness = current_best
                self.best_params = self.param_manager.denormalize_parameters(population[best_idx])
                self.logger.info(f"★ New best! {self.optimization_metric}={self.best_fitness:.4f}")
            
            # Check for stagnation
            percent_change = abs((current_best - previous_best) / (previous_best + 1e-10)) * 100
            if percent_change < self.percent_change_threshold:
                stagnation_count += 1
                self.logger.info(f"Stagnation count: {stagnation_count}/{self.evolution_stagnation}")
                
                if stagnation_count >= self.evolution_stagnation:
                    self.logger.info(f"Evolution stagnated. Stopping at iteration {iteration + 1}")
                    break
            else:
                stagnation_count = 0
            
            previous_best = current_best
            
            self.logger.info(f"Best {self.optimization_metric}: {self.best_fitness:.4f}")
        
        return self._save_results('SCE-UA')
    
    def _evolve_complex_sce(self, complex_pop: np.ndarray, complex_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve a complex using Competitive Complex Evolution (CCE)"""
        
        n_points = len(complex_pop)
        param_count = complex_pop.shape[1]
        
        for _ in range(self.num_evolution_steps):
            # Select a subcomplex (random sample)
            indices = np.random.choice(n_points, self.points_per_subcomplex, replace=False)
            subcomplex_pop = complex_pop[indices]
            subcomplex_fit = complex_fit[indices]
            
            # Sort subcomplex by fitness
            if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']:
                sorted_idx = np.argsort(subcomplex_fit)[::-1]
            else:
                sorted_idx = np.argsort(subcomplex_fit)
            
            subcomplex_pop = subcomplex_pop[sorted_idx]
            subcomplex_fit = subcomplex_fit[sorted_idx]
            
            # Generate offspring using reflection
            centroid = np.mean(subcomplex_pop[:-1], axis=0)  # Exclude worst point
            worst = subcomplex_pop[-1]
            
            # Reflection
            offspring = centroid + (centroid - worst)
            offspring = np.clip(offspring, 0, 1)
            
            # Evaluate offspring
            params = self.param_manager.denormalize_parameters(offspring)
            offspring_fit = self._evaluate_parameters(params)
            
            # Replace worst if offspring is better
            is_better = ((offspring_fit > subcomplex_fit[-1]) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                        else (offspring_fit < subcomplex_fit[-1]))
            
            if is_better:
                subcomplex_pop[-1] = offspring
                subcomplex_fit[-1] = offspring_fit
            
            # Update complex
            complex_pop[indices] = subcomplex_pop
            complex_fit[indices] = subcomplex_fit
        
        return complex_pop, complex_fit
    
    def run_de(self) -> Path:
        """
        Run Differential Evolution (DE) for ngen.
        
        Returns:
            Path to results file
        """
        self.logger.info("Starting ngen calibration with DE")
        
        # DE parameters
        self.population_size = self.config.get('POPULATION_SIZE', self._determine_de_population_size())
        self.F = self.config.get('DE_SCALING_FACTOR', 0.5)  # Scaling factor
        self.CR = self.config.get('DE_CROSSOVER_RATE', 0.9)  # Crossover rate
        
        self.logger.info(f"DE configuration:")
        self.logger.info(f"  Population size: {self.population_size}")
        self.logger.info(f"  Scaling factor (F): {self.F}")
        self.logger.info(f"  Crossover rate (CR): {self.CR}")
        
        # Get parameter info
        param_names = self.param_manager.all_param_names
        n_params = len(param_names)
        
        # Initialize population (normalized [0,1])
        population = np.random.random((self.population_size, n_params))
        fitness_values = np.full(self.population_size, -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial DE population...")
        for i in range(self.population_size):
            params = self.param_manager.denormalize_parameters(population[i])
            fitness = self._evaluate_parameters(params)
            fitness_values[i] = fitness
            
            if i == 0 or ((fitness > self.best_fitness) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] 
                         else (fitness < self.best_fitness)):
                self.best_fitness = fitness
                self.best_params = params
            
            self.iteration_results.append({
                'iteration': 0,
                'individual': i,
                'fitness': fitness,
                'parameters': params
            })
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Evaluated {i+1}/{self.population_size} individuals")
        
        # DE evolution
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"=== DE Iteration {iteration}/{self.max_iterations} ===")
            
            for i in range(self.population_size):
                # Select three random individuals (different from i)
                candidates = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                
                # Mutation: v = a + F * (b - c)
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, 0, 1)
                
                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.random(n_params) < self.CR
                # Ensure at least one parameter is from mutant
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(n_params)] = True
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Evaluate trial
                trial_params = self.param_manager.denormalize_parameters(trial)
                trial_fitness = self._evaluate_parameters(trial_params)
                
                # Selection
                is_better = ((trial_fitness > fitness_values[i]) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                            else (trial_fitness < fitness_values[i]))
                
                if is_better:
                    population[i] = trial
                    fitness_values[i] = trial_fitness
                    self.logger.info(f"Individual {i}: Improved! {self.optimization_metric}={trial_fitness:.4f}")
                    
                    # Update global best
                    is_best = ((trial_fitness > self.best_fitness) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']
                              else (trial_fitness < self.best_fitness))
                    
                    if is_best:
                        self.best_fitness = trial_fitness
                        self.best_params = trial_params
                        self.logger.info(f"★ New best! {self.optimization_metric}={self.best_fitness:.4f}")
                
                # Store result
                self.iteration_results.append({
                    'iteration': iteration,
                    'individual': i,
                    'fitness': trial_fitness if is_better else fitness_values[i],
                    'parameters': trial_params if is_better else self.param_manager.denormalize_parameters(population[i])
                })
            
            best_idx = np.nanargmax(fitness_values) if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.nanargmin(fitness_values)
            self.logger.info(f"Iteration {iteration} complete. Best {self.optimization_metric}: {fitness_values[best_idx]:.4f}")
        
        return self._save_results('DE')
    
    def _determine_de_population_size(self) -> int:
        """Determine appropriate population size for DE based on parameter count"""
        param_count = len(self.param_manager.all_param_names)
        # DE rule of thumb: 5-10 times the number of parameters
        return max(15, min(6 * param_count, 60))
    
    def run_nsga2(self) -> Path:
        """
        Run NSGA-II (Non-dominated Sorting Genetic Algorithm II) for ngen.
        Multi-objective optimization.
        
        Returns:
            Path to results file
        """
        self.logger.info("Starting ngen calibration with NSGA-II")
        
        # NSGA-II parameters
        self.population_size = self.config.get('NSGA2_POPULATION_SIZE', self._determine_nsga2_population_size())
        self.crossover_rate = self.config.get('NSGA2_CROSSOVER_RATE', 0.9)
        self.mutation_rate = self.config.get('NSGA2_MUTATION_RATE', 0.1)
        self.eta_c = self.config.get('NSGA2_ETA_C', 20)  # Crossover distribution index
        self.eta_m = self.config.get('NSGA2_ETA_M', 20)  # Mutation distribution index
        
        # Multi-objective setup
        self.objectives = self.config.get('NSGA2_OBJECTIVES', ['NSE', 'KGE'])
        self.num_objectives = len(self.objectives)
        
        self.logger.info(f"NSGA-II configuration:")
        self.logger.info(f"  Population size: {self.population_size}")
        self.logger.info(f"  Crossover rate: {self.crossover_rate}")
        self.logger.info(f"  Mutation rate: {self.mutation_rate}")
        self.logger.info(f"  Objectives: {', '.join(self.objectives)}")
        
        # Get parameter info
        param_names = self.param_manager.all_param_names
        n_params = len(param_names)
        
        # Initialize population
        population = np.random.random((self.population_size, n_params))
        objectives_matrix = np.full((self.population_size, self.num_objectives), np.nan)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial NSGA-II population...")
        for i in range(self.population_size):
            params = self.param_manager.denormalize_parameters(population[i])
            metrics = self.calibration_target.calculate_metrics(self.experiment_id)
            
            # Extract objectives
            for j, obj in enumerate(self.objectives):
                objectives_matrix[i, j] = metrics.get(obj.upper(), -999.0)
            
            self.iteration_results.append({
                'iteration': 0,
                'individual': i,
                'objectives': objectives_matrix[i],
                'parameters': params
            })
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Evaluated {i+1}/{self.population_size} individuals")
        
        # NSGA-II evolution
        for generation in range(1, self.max_iterations + 1):
            self.logger.info(f"=== NSGA-II Generation {generation}/{self.max_iterations} ===")
            
            # Generate offspring
            offspring = self._generate_nsga2_offspring(population)
            offspring_objectives = np.full((self.population_size, self.num_objectives), np.nan)
            
            # Evaluate offspring
            for i in range(len(offspring)):
                params = self.param_manager.denormalize_parameters(offspring[i])
                metrics = self.calibration_target.calculate_metrics(self.experiment_id)
                
                for j, obj in enumerate(self.objectives):
                    offspring_objectives[i, j] = metrics.get(obj.upper(), -999.0)
            
            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([objectives_matrix, offspring_objectives])
            
            # Environmental selection (non-dominated sorting + crowding distance)
            selected_indices = self._environmental_selection_nsga2(combined_pop, combined_obj)
            
            # Update population
            population = combined_pop[selected_indices]
            objectives_matrix = combined_obj[selected_indices]
            
            # Log progress
            pareto_front = self._get_pareto_front(objectives_matrix)
            front_size = len(pareto_front)
            avg_objectives = np.nanmean(objectives_matrix, axis=0)
            
            obj_str = ", ".join([f"Avg_{obj}={avg:.4f}" for obj, avg in zip(self.objectives, avg_objectives)])
            self.logger.info(f"Generation {generation} complete. Pareto front size: {front_size}, {obj_str}")
            
            # Store results
            for i in range(len(population)):
                params = self.param_manager.denormalize_parameters(population[i])
                self.iteration_results.append({
                    'iteration': generation,
                    'individual': i,
                    'objectives': objectives_matrix[i],
                    'parameters': params
                })
        
        # Store final Pareto front
        pareto_indices = self._get_pareto_front(objectives_matrix)
        self.pareto_front = {
            'solutions': population[pareto_indices],
            'objectives': objectives_matrix[pareto_indices],
            'parameters': [self.param_manager.denormalize_parameters(sol) for sol in population[pareto_indices]]
        }
        
        # Select a representative solution (knee point)
        knee_idx = self._find_knee_point(objectives_matrix[pareto_indices])
        self.best_params = self.pareto_front['parameters'][knee_idx]
        self.best_fitness = objectives_matrix[pareto_indices][knee_idx][0]  # Use first objective
        
        self.logger.info(f"NSGA-II completed. Pareto front size: {len(pareto_indices)}")
        
        return self._save_results('NSGA-II')
    
    def _determine_nsga2_population_size(self) -> int:
        """Determine appropriate population size for NSGA-II"""
        param_count = len(self.param_manager.all_param_names)
        return max(20, min(8 * param_count, 100))
    
    def _generate_nsga2_offspring(self, population: np.ndarray) -> np.ndarray:
        """Generate offspring using SBX crossover and polynomial mutation"""
        offspring = []
        pop_size, n_params = population.shape
        
        for _ in range(pop_size):
            # Tournament selection
            parent1 = population[np.random.randint(pop_size)]
            parent2 = population[np.random.randint(pop_size)]
            
            # Simulated Binary Crossover (SBX)
            if np.random.random() < self.crossover_rate:
                u = np.random.random(n_params)
                beta = np.where(u <= 0.5,
                               (2 * u) ** (1.0 / (self.eta_c + 1)),
                               (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1)))
                
                child = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            else:
                child = parent1.copy()
            
            # Polynomial mutation
            for i in range(n_params):
                if np.random.random() < self.mutation_rate:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (self.eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (self.eta_m + 1))
                    
                    child[i] += delta
            
            child = np.clip(child, 0, 1)
            offspring.append(child)
        
        return np.array(offspring)
    
    def _environmental_selection_nsga2(self, population: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        """Environmental selection using non-dominated sorting and crowding distance"""
        pop_size = len(population) // 2  # Select half
        
        # Non-dominated sorting
        fronts = self._fast_non_dominated_sort(objectives)
        
        # Select individuals
        selected = []
        front_idx = 0
        
        while len(selected) < pop_size and front_idx < len(fronts):
            front = fronts[front_idx]
            
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                # Calculate crowding distance for this front
                crowding = self._crowding_distance(objectives[front])
                # Sort by crowding distance (descending)
                sorted_indices = np.argsort(crowding)[::-1]
                needed = pop_size - len(selected)
                selected.extend([front[i] for i in sorted_indices[:needed]])
            
            front_idx += 1
        
        return np.array(selected)
    
    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting algorithm"""
        pop_size = len(objectives)
        domination_count = np.zeros(pop_size)
        dominated_solutions = [[] for _ in range(pop_size)]
        fronts = [[]]
        
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove last empty front
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)
    
    def _crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for a set of solutions"""
        pop_size, num_obj = objectives.shape
        crowding = np.zeros(pop_size)
        
        for m in range(num_obj):
            sorted_indices = np.argsort(objectives[:, m])
            crowding[sorted_indices[0]] = np.inf
            crowding[sorted_indices[-1]] = np.inf
            
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, pop_size - 1):
                    crowding[sorted_indices[i]] += (
                        (objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m]) / obj_range
                    )
        
        return crowding
    
    def _get_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """Get indices of solutions in Pareto front"""
        fronts = self._fast_non_dominated_sort(objectives)
        return np.array(fronts[0]) if fronts else np.array([])
    
    def _find_knee_point(self, objectives: np.ndarray) -> int:
        """Find knee point in Pareto front (representative solution)"""
        if len(objectives) == 1:
            return 0
        
        # Normalize objectives
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        normalized = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
        
        # Find solution closest to ideal point (1, 1, ...)
        distances = np.linalg.norm(normalized - 1, axis=1)
        return np.argmin(distances)
    
    def run_adam(self, steps: int = 100, lr: float = 0.01) -> Path:
        """
        Run Adam gradient-based optimization for ngen using finite differences.
        
        Args:
            steps: Number of optimization steps
            lr: Learning rate
            
        Returns:
            Path to results file
        """
        return self._run_differentiable_optimization('ADAM', steps, lr)
    
    def run_lbfgs(self, steps: int = 50, lr: float = 0.1) -> Path:
        """
        Run L-BFGS gradient-based optimization for ngen using finite differences.
        
        Args:
            steps: Number of optimization steps
            lr: Learning rate
            
        Returns:
            Path to results file
        """
        return self._run_differentiable_optimization('LBFGS', steps, lr)
    
    def _run_differentiable_optimization(self, optimizer_choice: str, steps: int, lr: float) -> Path:
        """
        Core differentiable optimization implementation using finite differences.
        
        Args:
            optimizer_choice: 'ADAM' or 'LBFGS'
            steps: Number of optimization steps
            lr: Learning rate
            
        Returns:
            Path to results file
        """
        self.logger.info(f"Starting ngen differentiable optimization with {optimizer_choice}")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Initialize parameters (normalized [0,1])
        initial_params = self.param_manager.get_default_parameters()
        x0 = self.param_manager.normalize_parameters(initial_params)
        x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)
        
        # Setup optimizer
        if optimizer_choice == 'ADAM':
            optimizer = optim.Adam([x], lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else 'min',
                                                             factor=0.5, patience=10)
        elif optimizer_choice == 'LBFGS':
            optimizer = optim.LBFGS([x], lr=lr, max_iter=20, line_search_fn='strong_wolfe')
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_choice}")
        
        # Optimization state
        best_loss = float('-inf') if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else float('inf')
        best_x = x.detach().clone()
        loss_history = []
        
        # Finite difference parameters
        epsilon = self.config.get('FD_EPSILON', 1e-4)
        
        # Optimization loop
        if optimizer_choice == 'ADAM':
            progress_bar = tqdm(range(1, steps + 1), desc=f"ngen {optimizer_choice}")
            
            for step in progress_bar:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
                
                # Compute loss and gradients using finite differences
                current_loss, gradients = self._compute_fd_gradients(x, epsilon)
                
                # Manual gradient setting
                x.grad = gradients
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
                
                optimizer.step()
                
                # Track progress
                loss_history.append(current_loss)
                is_better = (current_loss > best_loss if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] 
                            else current_loss < best_loss)
                
                if is_better:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                
                # Update learning rate
                if scheduler:
                    scheduler.step(current_loss)
                
                # Progress bar update
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.6f}',
                    'Best': f'{best_loss:.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Store result
                params = self.param_manager.denormalize_parameters(x.detach().cpu().numpy())
                self.iteration_results.append({
                    'iteration': step,
                    'fitness': current_loss,
                    'parameters': params
                })
                
                if step % 10 == 0:
                    self.logger.info(f"Step {step}: {self.optimization_metric}={current_loss:.6f}, Best={best_loss:.6f}")
        
        elif optimizer_choice == 'LBFGS':
            for step in range(1, steps + 1):
                def closure():
                    optimizer.zero_grad()
                    loss, grad = self._compute_fd_gradients(x, epsilon)
                    x.grad = grad
                    return loss
                
                loss = optimizer.step(closure)
                
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
                
                current_loss = float(loss)
                loss_history.append(current_loss)
                
                is_better = (current_loss > best_loss if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] 
                            else current_loss < best_loss)
                
                if is_better:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                
                params = self.param_manager.denormalize_parameters(x.detach().cpu().numpy())
                self.iteration_results.append({
                    'iteration': step,
                    'fitness': current_loss,
                    'parameters': params
                })
                
                if step % 5 == 0:
                    self.logger.info(f"Step {step}: {self.optimization_metric}={current_loss:.6f}, Best={best_loss:.6f}")
        
        # Store best results
        self.best_params = self.param_manager.denormalize_parameters(best_x.detach().cpu().numpy())
        self.best_fitness = best_loss
        
        self.logger.info(f"{optimizer_choice} optimization completed. Best {self.optimization_metric}: {best_loss:.6f}")
        
        return self._save_results(f'{optimizer_choice}')
    
    def _compute_fd_gradients(self, x: torch.Tensor, epsilon: float) -> Tuple[float, torch.Tensor]:
        """
        Compute gradients using finite differences.
        
        Args:
            x: Current parameter tensor (normalized)
            epsilon: Finite difference step size
            
        Returns:
            Tuple of (loss, gradients)
        """
        x_np = x.detach().cpu().numpy()
        n_params = len(x_np)
        
        # Evaluate at current point
        params = self.param_manager.denormalize_parameters(x_np)
        current_loss = self._evaluate_parameters(params)
        
        # Compute finite difference gradients
        gradients = np.zeros(n_params)
        
        for i in range(n_params):
            # Forward step
            x_forward = x_np.copy()
            x_forward[i] += epsilon
            x_forward = np.clip(x_forward, 0, 1)
            
            params_forward = self.param_manager.denormalize_parameters(x_forward)
            loss_forward = self._evaluate_parameters(params_forward)
            
            # Central difference
            x_backward = x_np.copy()
            x_backward[i] -= epsilon
            x_backward = np.clip(x_backward, 0, 1)
            
            params_backward = self.param_manager.denormalize_parameters(x_backward)
            loss_backward = self._evaluate_parameters(params_backward)
            
            # Gradient
            gradients[i] = (loss_forward - loss_backward) / (2 * epsilon)
        
        # Convert to torch tensor with appropriate sign for maximization/minimization
        grad_tensor = torch.tensor(gradients, dtype=torch.float32, device=x.device)
        
        # Flip sign for maximization metrics
        if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE']:
            grad_tensor = -grad_tensor  # Gradient ascent
        
        return current_loss, grad_tensor
    
    def _save_results(self, algorithm: str) -> Path:
        """Save optimization results to file"""
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.iteration_results)
            
            # Handle NSGA-II objectives differently
            if algorithm == 'NSGA-II' and 'objectives' in results_df.columns:
                # Expand objectives
                obj_cols = pd.DataFrame(results_df['objectives'].tolist(), 
                                       columns=[f'obj_{obj}' for obj in self.objectives])
                results_df = pd.concat([results_df.drop('objectives', axis=1), obj_cols], axis=1)
            
            # Expand parameter columns
            param_cols = pd.DataFrame(results_df['parameters'].tolist())
            param_cols.columns = [f'param_{col}' for col in param_cols.columns]
            results_df = pd.concat([results_df.drop('parameters', axis=1), param_cols], axis=1)
            
            # Save to CSV
            results_file = self.results_dir / f"ngen_{algorithm}_{self.experiment_id}_results.csv"
            results_df.to_csv(results_file, index=False)
            
            self.logger.info(f"Results saved to: {results_file}")
            
            # Save best parameters or Pareto front
            if algorithm == 'NSGA-II' and hasattr(self, 'pareto_front') and self.pareto_front:
                # Save entire Pareto front
                pareto_file = self.results_dir / f"ngen_{algorithm}_{self.experiment_id}_pareto_front.csv"
                
                pareto_data = []
                for i, (sol, obj, params) in enumerate(zip(
                    self.pareto_front['solutions'],
                    self.pareto_front['objectives'],
                    self.pareto_front['parameters']
                )):
                    row = {'solution_id': i}
                    for j, obj_name in enumerate(self.objectives):
                        row[f'obj_{obj_name}'] = obj[j]
                    row.update(params)
                    pareto_data.append(row)
                
                pareto_df = pd.DataFrame(pareto_data)
                pareto_df.to_csv(pareto_file, index=False)
                self.logger.info(f"Pareto front saved to: {pareto_file}")
                
                # Also save representative (knee point) solution
                best_params_file = self.results_dir / f"ngen_{algorithm}_{self.experiment_id}_best_params.json"
                import json
                with open(best_params_file, 'w') as f:
                    json.dump({
                        'algorithm': algorithm,
                        'representative_solution': 'knee_point',
                        'objectives': {obj: float(val) for obj, val in zip(self.objectives, self.pareto_front['objectives'][self._find_knee_point(self.pareto_front['objectives'])])},
                        'parameters': {k: float(v) for k, v in self.best_params.items()}
                    }, f, indent=2)
                
                self.logger.info(f"Representative solution saved to: {best_params_file}")
                self.logger.info(f"Pareto front size: {len(self.pareto_front['solutions'])}")
            
            else:
                # Single-objective: save best parameters
                best_params_file = self.results_dir / f"ngen_{algorithm}_{self.experiment_id}_best_params.json"
                import json
                with open(best_params_file, 'w') as f:
                    json.dump({
                        'algorithm': algorithm,
                        'best_fitness': float(self.best_fitness),
                        'metric': self.optimization_metric,
                        'parameters': {k: float(v) for k, v in self.best_params.items()}
                    }, f, indent=2)
                
                self.logger.info(f"Best parameters saved to: {best_params_file}")
                self.logger.info(f"Final best {self.optimization_metric}: {self.best_fitness:.4f}")
            
            return results_file
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            traceback.print_exc()
            return None