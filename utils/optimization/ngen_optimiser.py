#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Optimizer

Handles ngen model calibration using various optimization algorithms:
- DDS (Dynamically Dimensioned Search)
- PSO (Particle Swarm Optimization)
- SCE-UA (Shuffled Complex Evolution - University of Arizona)

Integrates with the existing CONFLUENCE optimization framework.

Author: CONFLUENCE Development Team
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import traceback

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
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        self.ngen_sim_dir = project_dir / 'simulations' / self.experiment_id / 'ngen'
        self.ngen_setup_dir = project_dir / 'settings' / 'ngen'
        
        # Results tracking
        self.results_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}" / "optimisation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.best_params = None
        self.best_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'KGEp', 'NSE'] else np.inf
        self.iteration_results = []
        
        self.logger.info("NgenOptimizer initialized")
        self.logger.info(f"Optimization metric: {self.optimization_metric}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
    
    def _create_calibration_target(self) -> Any:
        """Factory method to create appropriate ngen calibration target"""
        optimization_target = self.config.get('OPTIMISATION_TARGET', 'streamflow')
        
        # Get project directory for calibration target
        data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        
        if optimization_target in ['streamflow', 'discharge', 'flow']:
            return NgenStreamflowTarget(self.config, project_dir, self.logger)
        # Future: Add snow, ET, etc. targets here
        else:
            self.logger.warning(f"Unknown target {optimization_target}, defaulting to streamflow")
            return NgenStreamflowTarget(self.config, project_dir, self.logger)
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> float:
        """
        Evaluate a parameter set.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Fitness value
        """
        try:
            # Apply parameters
            if not self.param_manager.update_config_files(params):
                self.logger.error("Failed to update ngen config files")
                return -999.0
            
            # Run ngen
            if not _run_ngen_worker(self.config):
                self.logger.error("Failed to run ngen model")
                return -999.0
            
            # Calculate metrics
            metrics = self.calibration_target.calculate_metrics(self.experiment_id)
            
            metric_name = self.optimization_metric.upper()
            if metric_name in metrics:
                fitness = metrics[metric_name]
                self.logger.info(f"Fitness: {metric_name}={fitness:.4f}")
                return fitness
            else:
                self.logger.error(f"Metric {metric_name} not in results")
                return -999.0
                
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
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
    
    def _save_results(self, algorithm: str) -> Path:
        """Save optimization results to file"""
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.iteration_results)
            
            # Expand parameter columns
            param_cols = pd.DataFrame(results_df['parameters'].tolist())
            param_cols.columns = [f'param_{col}' for col in param_cols.columns]
            results_df = pd.concat([results_df.drop('parameters', axis=1), param_cols], axis=1)
            
            # Save to CSV
            results_file = self.results_dir / f"ngen_{algorithm}_{self.experiment_id}_results.csv"
            results_df.to_csv(results_file, index=False)
            
            self.logger.info(f"Results saved to: {results_file}")
            
            # Save best parameters separately
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
