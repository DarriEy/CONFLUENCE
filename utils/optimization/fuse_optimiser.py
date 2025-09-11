#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUSE Optimizer

Handles FUSE model calibration using various optimization algorithms.
Integrates with the existing CONFLUENCE optimization framework.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import time
import traceback


import torch
import torch.optim as optim
from tqdm import tqdm
import logging

from fuse_parameter_manager import FUSEParameterManager
from fuse_calibration_targets import FUSECalibrationTarget, FUSEStreamflowTarget, FUSESnowTarget


class FUSEOptimizer:
    """Main FUSE optimization class that coordinates calibration"""
    
    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Initialize core components
        self.param_manager = FUSEParameterManager(config, logger, optimization_settings_dir)
        self.calibration_target = self._create_calibration_target()
        
        # Optimization settings
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.optimization_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # FUSE execution settings
        self.fuse_exe_path = self._get_fuse_executable_path()
        data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        self.fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
        self.fuse_setup_dir = project_dir / 'settings' / 'FUSE'
        
        # Results tracking
        self.results_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}" / "optimisation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.best_params = None
        self.best_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
        self.iteration_results = []
    
    def _get_fuse_executable_path(self) -> Path:
        """Get path to FUSE executable"""
        fuse_install = self.config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_install == 'default':
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            return data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
        return Path(fuse_install) / 'fuse.exe'
    
    def _create_calibration_target(self) -> Any:
        """Factory method to create appropriate FUSE calibration target"""
        optimization_target = self.config.get('OPTIMISATION_TARGET', 'streamflow')
        
        # Get project directory for calibration target
        data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        project_dir = data_dir / f"domain_{self.domain_name}"
        
        if optimization_target in ['swe', 'sca', 'snow_depth']:
            return FUSESnowTarget(self.config, project_dir, self.logger)
        else:
            return FUSEStreamflowTarget(self.config, project_dir, self.logger)
    
    def run_pso(self) -> Path:
        """Run Particle Swarm Optimization for FUSE"""
        self.logger.info("Starting FUSE calibration with PSO")
        
        # PSO parameters
        n_particles = self.config.get('PSO_PARTICLES', 20)
        w = self.config.get('PSO_INERTIA', 0.9)
        c1 = self.config.get('PSO_COGNITIVE', 2.0)
        c2 = self.config.get('PSO_SOCIAL', 2.0)
        
        # Get parameter bounds and initial values
        param_bounds = self.param_manager.get_parameter_bounds()
        n_params = len(self.param_manager.all_param_names)
        
        # Initialize swarm
        particles = np.random.uniform(0, 1, (n_particles, n_params))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_params))
        
        personal_best_positions = particles.copy()
        personal_best_fitness = np.full(n_particles, -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf)
        
        global_best_position = None
        global_best_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
        
        # Main PSO loop
        for iteration in range(self.max_iterations):
            self.logger.info(f"PSO Iteration {iteration + 1}/{self.max_iterations}")
            
            # Evaluate all particles
            for i, particle in enumerate(particles):
                try:
                    fitness = self._evaluate_particle(particle, iteration, i)
                    
                    # Update personal best
                    if self._is_better_fitness(fitness, personal_best_fitness[i]):
                        personal_best_fitness[i] = fitness
                        personal_best_positions[i] = particle.copy()
                    
                    # Update global best
                    if self._is_better_fitness(fitness, global_best_fitness):
                        global_best_fitness = fitness
                        global_best_position = particle.copy()
                        self.best_fitness = fitness
                        self.best_params = self.param_manager.denormalize_parameters(particle)
                    
                    self.logger.info(f"Particle {i}: {self.optimization_metric}={fitness:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating particle {i}: {str(e)}")
                    fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
                
                # Store results
                self.iteration_results.append({
                    'iteration': iteration + 1,
                    'particle': i,
                    'fitness': fitness,
                    'parameters': self.param_manager.denormalize_parameters(particle)
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
        """Run Shuffled Complex Evolution - University of Arizona for FUSE"""
        self.logger.info("Starting FUSE calibration with SCE-UA")
        
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
        
        self.logger.info(f"SCE-UA configuration:")
        self.logger.info(f"  Complexes: {self.num_complexes}")
        self.logger.info(f"  Points per complex: {self.points_per_complex}")
        self.logger.info(f"  Total population: {self.population_size}")
        self.logger.info(f"  Points per subcomplex: {self.points_per_subcomplex}")
        self.logger.info(f"  Evolution steps: {self.num_evolution_steps}")
        
        # Initialize SCE-UA state variables
        self.population = None
        self.population_scores = None
        self.complexes = None
        self.stagnation_counter = 0
        
        # Initialize population and complexes
        self._initialize_sceua_population()
        
        # Run SCE-UA iterations
        return self._run_sceua_algorithm()

    def _initialize_sceua_population(self) -> None:
        """Initialize SCE-UA population and organize into complexes"""
        self.logger.info(f"Initializing SCE-UA population with {self.population_size} points")
        
        param_count = len(self.param_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.uniform(0, 1, (self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Set first individual to current best or default parameters if available
        try:
            initial_params = self.param_manager.get_initial_parameters()
            if initial_params:
                initial_normalized = self.param_manager.normalize_parameters(initial_params)
                self.population[0] = np.clip(initial_normalized, 0, 1)
        except Exception:
            self.logger.warning("Could not set initial parameters, using random start")
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_sceua_population()
        
        # Sort population by fitness (best first)
        self._sort_sceua_population()
        
        # Initialize complexes
        self._initialize_sceua_complexes()
        
        # Find best individual and update best_fitness
        if not np.isnan(self.population_scores[0]):
            if self._is_better_fitness(self.population_scores[0], self.best_fitness):
                self.best_fitness = self.population_scores[0]
                self.best_params = self.param_manager.denormalize_parameters(self.population[0])
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_fitness:.4f}")

    def _evaluate_sceua_population(self) -> None:
        """Evaluate the entire SCE-UA population"""
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                try:
                    score = self._evaluate_particle(self.population[i], 0, i)
                    self.population_scores[i] = score if score is not None else float('-inf')
                    
                    # Update best fitness if this score is better
                    if score is not None and self._is_better_fitness(score, self.best_fitness):
                        self.best_fitness = score
                        self.best_params = self.param_manager.denormalize_parameters(self.population[i])
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating individual {i}: {str(e)}")
                    self.population_scores[i] = float('-inf')

    def _sort_sceua_population(self) -> None:
        """Sort population by fitness scores (best first)"""
        # Get sort indices (descending order for maximization metrics)
        if self.optimization_metric.upper() in ['KGE', 'NSE', 'KGEP']:
            sort_indices = np.argsort(self.population_scores)[::-1]  # Descending
        else:  # RMSE, MAE (lower is better)
            sort_indices = np.argsort(self.population_scores)  # Ascending
        
        # Sort both population and scores
        self.population = self.population[sort_indices]
        self.population_scores = self.population_scores[sort_indices]

    def _initialize_sceua_complexes(self) -> None:
        """Initialize complexes by distributing population members"""
        self.complexes = []
        for i in range(self.num_complexes):
            complex_indices = list(range(i, self.population_size, self.num_complexes))[:self.points_per_complex]
            self.complexes.append(complex_indices)

    def _run_sceua_algorithm(self) -> Path:
        """Core SCE-UA algorithm implementation"""
        self.logger.info("Starting SCE-UA evolution")
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"SCE-UA Iteration {iteration}/{self.max_iterations}")
            
            # Collect all trial solutions from all complexes
            all_trial_solutions = []
            complex_trial_info = []
            
            for complex_id in range(self.num_complexes):
                complex_trials = self._generate_complex_trials(complex_id)
                
                for trial_info in complex_trials:
                    trial_info['global_trial_id'] = len(all_trial_solutions)
                    all_trial_solutions.append(trial_info['trial_solution'])
                    complex_trial_info.append(trial_info)
            
            # Evaluate all trial solutions
            trial_scores = self._evaluate_trial_solutions(all_trial_solutions, iteration)
            
            # Apply results back to complexes
            total_improvements = 0
            for trial_info in complex_trial_info:
                global_trial_id = trial_info['global_trial_id']
                trial_score = trial_scores[global_trial_id]
                
                if self._apply_trial_to_complex(trial_info, trial_score):
                    total_improvements += 1
            
            # Shuffle complexes (redistribute points among complexes)
            self._shuffle_complexes()
            
            # Sort entire population
            self._sort_sceua_population()
            
            # Update global best
            if self._is_better_fitness(self.population_scores[0], self.best_fitness):
                self.best_fitness = self.population_scores[0]
                self.best_params = self.param_manager.denormalize_parameters(self.population[0])
                self.logger.info(f"Iteration {iteration}: NEW BEST! {self.optimization_metric}={self.best_fitness:.4f}")
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Store iteration results
            self.iteration_results.append({
                'iteration': iteration,
                'particle': 0,  # Representative for the iteration
                'fitness': self.best_fitness,
                'parameters': self.best_params,
                'improvements': total_improvements,
                'stagnation_counter': self.stagnation_counter
            })
            
            self.logger.info(f"Iteration {iteration} complete. Best {self.optimization_metric}: {self.best_fitness:.4f}, "
                            f"Improvements: {total_improvements}, Stagnation: {self.stagnation_counter}")
            
            # Check convergence
            if self.stagnation_counter >= self.evolution_stagnation:
                self.logger.info(f"Convergence reached after {iteration} iterations (stagnation: {self.stagnation_counter})")
                break
        
        return self._save_results('SCE-UA')

    def _generate_complex_trials(self, complex_id: int) -> List[Dict]:
        """Generate trial solutions for a complex using competitive complex evolution"""
        complex_indices = self.complexes[complex_id]
        complex_population = self.population[complex_indices]
        complex_scores = self.population_scores[complex_indices]
        
        trials = []
        
        for step in range(self.num_evolution_steps):
            # Create subcomplex by selecting points
            subcomplex_size = self.points_per_subcomplex
            
            # Select points using triangular probability (favor better solutions)
            probs = np.arange(len(complex_indices), 0, -1)
            probs = probs / np.sum(probs)
            
            selected_indices = np.random.choice(
                len(complex_indices), 
                size=min(subcomplex_size, len(complex_indices)), 
                replace=False, 
                p=probs
            )
            
            subcomplex_indices = [complex_indices[i] for i in selected_indices]
            subcomplex_population = complex_population[selected_indices]
            
            # Generate new trial point using simplex reflection
            trial_solution = self._generate_simplex_trial(subcomplex_population)
            
            # Find worst point in subcomplex for potential replacement
            if self.optimization_metric.upper() in ['KGE', 'NSE', 'KGEP']:
                worst_idx = selected_indices[np.argmin(complex_scores[selected_indices])]
            else:
                worst_idx = selected_indices[np.argmax(complex_scores[selected_indices])]
            
            trial_info = {
                'trial_solution': trial_solution,
                'complex_id': complex_id,
                'subcomplex_indices': subcomplex_indices,
                'worst_global_idx': complex_indices[worst_idx],
                'step': step
            }
            
            trials.append(trial_info)
        
        return trials

    def _generate_simplex_trial(self, subcomplex_population: np.ndarray) -> np.ndarray:
        """Generate trial solution using simplex reflection method"""
        if len(subcomplex_population) < 2:
            return np.random.uniform(0, 1, subcomplex_population.shape[1])
        
        # Calculate centroid of all but worst point
        if self.optimization_metric.upper() in ['KGE', 'NSE', 'KGEP']:
            # For maximization metrics, worst is minimum score
            best_points = subcomplex_population[:-1]  # All but last (worst)
        else:
            # For minimization metrics, worst is maximum score  
            best_points = subcomplex_population[:-1]  # All but last (worst)
        
        centroid = np.mean(best_points, axis=0)
        worst_point = subcomplex_population[-1]
        
        # Reflection with random step size
        alpha = 2.0 * np.random.random()  # Random reflection coefficient
        trial_solution = centroid + alpha * (centroid - worst_point)
        
        # Ensure bounds
        trial_solution = np.clip(trial_solution, 0, 1)
        
        # If trial is outside bounds or identical to existing point, use random mutation
        if np.any(trial_solution <= 0) or np.any(trial_solution >= 1):
            # Random mutation around centroid
            mutation_strength = 0.1
            trial_solution = centroid + np.random.normal(0, mutation_strength, len(centroid))
            trial_solution = np.clip(trial_solution, 0, 1)
        
        return trial_solution

    def _evaluate_trial_solutions(self, trial_solutions: List[np.ndarray], iteration: int) -> List[float]:
        """Evaluate list of trial solutions"""
        trial_scores = []
        
        for i, trial_solution in enumerate(trial_solutions):
            try:
                score = self._evaluate_particle(trial_solution, iteration, i)
                trial_scores.append(score if score is not None else float('-inf'))
            except Exception as e:
                self.logger.error(f"Error evaluating trial solution {i}: {str(e)}")
                trial_scores.append(float('-inf'))
        
        return trial_scores

    def _apply_trial_to_complex(self, trial_info: Dict, trial_score: float) -> bool:
        """Apply trial solution to complex if it improves the worst point"""
        worst_global_idx = trial_info['worst_global_idx']
        current_worst_score = self.population_scores[worst_global_idx]
        
        # Check if trial is better than worst
        is_improvement = self._is_better_fitness(trial_score, current_worst_score)
        
        if is_improvement:
            # Replace worst point with trial solution
            self.population[worst_global_idx] = trial_info['trial_solution']
            self.population_scores[worst_global_idx] = trial_score
            return True
        
        return False

    def _shuffle_complexes(self) -> None:
        """Shuffle complexes by redistributing population members"""
        # Sort population first
        self._sort_sceua_population()
        
        # Redistribute points among complexes
        for i in range(self.num_complexes):
            complex_indices = list(range(i, self.population_size, self.num_complexes))[:self.points_per_complex]
            self.complexes[i] = complex_indices


    def run_de(self) -> Path:
        """Run Differential Evolution for FUSE"""
        self.logger.info("Starting FUSE calibration with DE")
        
        # DE parameters
        self.population_size = self.config.get('POPULATION_SIZE', self._determine_de_population_size())
        self.F = self.config.get('DE_SCALING_FACTOR', 0.5)  # Scaling factor
        self.CR = self.config.get('DE_CROSSOVER_RATE', 0.9)  # Crossover rate
        
        self.logger.info(f"DE configuration:")
        self.logger.info(f"  Population size: {self.population_size}")
        self.logger.info(f"  Scaling factor (F): {self.F}")
        self.logger.info(f"  Crossover rate (CR): {self.CR}")
        
        # Initialize DE state variables
        self.population = None
        self.population_scores = None
        
        # Initialize population
        self._initialize_de_population()
        
        # Run DE iterations
        return self._run_de_algorithm()

    # Add these methods to the FUSEOptimizer class:

    def run_adam(self, steps: int = 100, lr: float = 0.01, initial_params: Optional[Dict[str, float]] = None) -> Path:
        """Run Adam gradient-based optimization for FUSE"""
        return self._run_differentiable_optimization('ADAM', steps, lr, initial_params)

    def run_lbfgs(self, steps: int = 50, lr: float = 0.1, initial_params: Optional[Dict[str, float]] = None) -> Path:
        """Run L-BFGS gradient-based optimization for FUSE"""
        return self._run_differentiable_optimization('LBFGS', steps, lr, initial_params)

    def run_differentiable(self, optimizer: str = 'ADAM', steps: int = 100, lr: float = 0.01, 
                        initial_params: Optional[Dict[str, float]] = None) -> Path:
        """Run differentiable optimization with specified optimizer"""
        return self._run_differentiable_optimization(optimizer.upper(), steps, lr, initial_params)

    def _run_differentiable_optimization(self, optimizer_choice: str, steps: int, lr: float, 
                                    initial_params: Optional[Dict[str, float]]) -> Path:
        """Core differentiable optimization implementation"""
        self.logger.info(f"Starting FUSE differentiable optimization with {optimizer_choice}")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Initialize parameters
        if initial_params:
            x0 = self.param_manager.normalize_parameters(initial_params)
        else:
            try:
                initial_params = self.param_manager.get_initial_parameters()
                x0 = self.param_manager.normalize_parameters(initial_params) if initial_params else None
            except:
                x0 = None
        
        if x0 is None:
            x0 = np.full(len(self.param_manager.all_param_names), 0.5)
        
        # Convert to tensor
        x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)
        
        # Setup optimizer
        if optimizer_choice == 'ADAM':
            # Adaptive learning rate for Adam
            initial_lr = max(lr * 5, 0.001)
            optimizer = optim.Adam([x], lr=initial_lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min' if self.optimization_metric.upper() in ['RMSE', 'MAE'] else 'max',
                factor=0.7, patience=3, min_lr=lr * 0.1, verbose=False
            )
            self.logger.info(f"Using Adam with initial LR: {initial_lr:.6f}")
            
        elif optimizer_choice == 'LBFGS':
            optimizer = optim.LBFGS(
                [x], lr=lr, max_iter=10, max_eval=15, tolerance_grad=1e-7,
                tolerance_change=1e-9, history_size=10, line_search_fn='strong_wolfe'
            )
            scheduler = None
            self.logger.info("Using L-BFGS with strong Wolfe line search")
            
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_choice}'. Use 'ADAM' or 'LBFGS'.")
        
        # Optimization state
        best_loss = float('inf') if self.optimization_metric.upper() in ['RMSE', 'MAE'] else float('-inf')
        best_x = x.detach().clone()
        loss_history = []
        function_evals = 0
        stagnation_counter = 0
        
        start_time = time.time()
        
        # Optimization loop
        if optimizer_choice == 'ADAM':
            progress_bar = tqdm(range(1, steps + 1), desc="FUSE Differentiable (Adam)")
            
            for step in progress_bar:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
                
                # Forward pass with gradient computation
                current_loss, gradients = self._fuse_autograd_step(x)
                function_evals += 1
                
                # Manual gradient setting (since we computed them externally)
                x.grad = gradients
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([x], max_norm=1.0)
                
                optimizer.step()
                
                # Track progress
                loss_history.append(current_loss)
                is_better = (current_loss < best_loss if self.optimization_metric.upper() in ['RMSE', 'MAE'] 
                            else current_loss > best_loss)
                
                if is_better:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # Update learning rate
                if scheduler:
                    scheduler.step(current_loss)
                
                # Progress bar update
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.6f}',
                    'Best': f'{best_loss:.6f}',
                    'Evals': f'{function_evals}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # Detailed logging
                if step % 5 == 0 or step == 1:
                    self.logger.info(f"Step {step:4d}: {self.optimization_metric}={current_loss:.6f}, "
                                f"Best={best_loss:.6f}, Function evals: {function_evals}")
                
                # Store iteration results
                self.iteration_results.append({
                    'iteration': step,
                    'particle': 0,
                    'fitness': current_loss,
                    'parameters': self.param_manager.denormalize_parameters(x.detach().cpu().numpy()),
                    'gradient_norm': float(torch.norm(x.grad).cpu()),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        elif optimizer_choice == 'LBFGS':
            self.logger.info("Starting L-BFGS optimization")
            
            for step in range(1, steps + 1):
                def closure():
                    optimizer.zero_grad()
                    loss, gradients = self._fuse_autograd_step(x)
                    x.grad = gradients
                    return loss
                
                try:
                    loss = optimizer.step(closure)
                    function_evals += 1
                    
                    current_loss = float(loss)
                    is_better = (current_loss < best_loss if self.optimization_metric.upper() in ['RMSE', 'MAE'] 
                                else current_loss > best_loss)
                    
                    if is_better:
                        best_loss = current_loss
                        best_x = x.detach().clone()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    
                    # Logging
                    if step % 5 == 0 or step == 1:
                        self.logger.info(f"L-BFGS Step {step:4d}: {self.optimization_metric}={current_loss:.6f}, "
                                    f"Best={best_loss:.6f}")
                    
                    # Store iteration results
                    self.iteration_results.append({
                        'iteration': step,
                        'particle': 0,
                        'fitness': current_loss,
                        'parameters': self.param_manager.denormalize_parameters(x.detach().cpu().numpy()),
                        'gradient_norm': float(torch.norm(x.grad).cpu()) if x.grad is not None else 0.0
                    })
                    
                except RuntimeError as e:
                    self.logger.warning(f"L-BFGS error at step {step}: {e}")
                    # Fallback to gradient descent step
                    current_loss, gradients = self._fuse_autograd_step(x)
                    x.grad = gradients
                    with torch.no_grad():
                        x -= lr * 0.1 * gradients  # Small gradient step
                        x.clamp_(0.0, 1.0)
        
        # Finalization
        optimization_time = time.time() - start_time
        final_params = self.param_manager.denormalize_parameters(best_x.clamp(0.0, 1.0).detach().cpu().numpy())
        
        # Update best results for compatibility
        self.best_fitness = best_loss
        self.best_params = final_params
        
        self.logger.info(f"FUSE differentiable optimization completed in {optimization_time:.1f}s")
        self.logger.info(f"Best {self.optimization_metric}: {best_loss:.6f}")
        self.logger.info(f"Total function evaluations: {function_evals}")
        
        return self._save_results(f'{optimizer_choice}_DIFF')

    def _fuse_autograd_step(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Single forward+backward pass for FUSE optimization
        
        Returns:
            Tuple of (loss_value, gradients_tensor)
        """
        try:
            # Convert to numpy parameters
            x_np = x.detach().cpu().numpy()
            params = self.param_manager.denormalize_parameters(x_np)
            
            # Update constraints and run FUSE
            success = self.update_constraint_defaults(params)
            if not success:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            success = self._run_fuse_model_with_constraints()
            if not success:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            metrics, jacobian = self._calculate_metrics_with_jacobian()
            if metrics is None:
                return self._get_worst_loss(), torch.zeros_like(x)

            fitness = self._extract_target_metric(metrics)
            if fitness is None or np.isnan(fitness):
                return self._get_worst_loss(), torch.zeros_like(x)

            if jacobian is None:
                # NEW: fall back to FD gradients so ADAM can still work
                gradients = self._finite_difference_gradients(x)
            else:
                gradients = self._jacobian_to_gradients(jacobian, x.device)

            return fitness, gradients

            
        except Exception as e:
            self.logger.error(f"Error in autograd step: {str(e)}")
            return self._get_worst_loss(), torch.zeros_like(x)

    def _get_worst_loss(self) -> float:
        """Get worst possible loss value for the optimization metric"""
        if self.optimization_metric.upper() in ['KGE', 'NSE']:
            return -1.0  # Worst for maximization
        else:
            return 1e6   # Worst for minimization (RMSE, MAE)

    def _calculate_metrics_with_jacobian(self) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        Calculate metrics and extract jacobian from FUSE outputs
        
        Returns:
            Tuple of (metrics_dict, jacobian_array)
        """
        try:
            # Calculate metrics using calibration target
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            metrics = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            if not metrics:
                return None, None
            
            # Extract jacobian from FUSE output file
            jacobian = self._extract_fuse_jacobian(fuse_sim_dir)
            
            return metrics, jacobian
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics with jacobian: {str(e)}")
            return None, None

    def _extract_fuse_jacobian(self, fuse_sim_dir: Path) -> Optional[np.ndarray]:
        """
        Extract numerical jacobian from FUSE output NetCDF file
        
        The jacobian represents d(outputs)/d(parameters) and has shape:
        [n_time_steps, n_param_sets, n_spatial, n_outputs, n_parameters]
        
        For optimization, we typically want the jacobian of streamflow (q_instnt or q_routed)
        with respect to parameters, aggregated over time.
        """
        try:
            # Find the most recent FUSE output file
            output_files = list(fuse_sim_dir.glob("*_runs_def.nc"))
            if not output_files:
                self.logger.error("No FUSE output files found")
                return None
            
            output_file = sorted(output_files)[-1]  # Most recent
            
            with xr.open_dataset(output_file) as ds:
                # Check if jacobian data exists
                if 'numjacobian' not in ds.data_vars:
                    self.logger.error("numjacobian not found in FUSE output")
                    return None
                
                # Extract jacobian - shape should be (time, param_set, lat, lon, n_params)
                jacobian_raw = ds['numjacobian'].values
                
                if jacobian_raw.size == 0:
                    self.logger.warning("Jacobian array is empty")
                    return None
                
                # For streamflow optimization, we typically want q_instnt or q_routed gradients
                # The jacobian in FUSE output represents d(q)/d(params)
                
                # Aggregate over time (could use mean, sum, or focus on specific period)
                if len(jacobian_raw.shape) >= 3:
                    # Take mean over time dimension (index 0)
                    jacobian_aggregated = np.nanmean(jacobian_raw, axis=0)
                    
                    # If there are spatial dimensions, aggregate those too
                    while len(jacobian_aggregated.shape) > 2:
                        jacobian_aggregated = np.nanmean(jacobian_aggregated, axis=1)
                    
                    # Final shape should be (param_set, n_parameters)
                    # For single parameter set optimization, take first set
                    if len(jacobian_aggregated.shape) == 2:
                        jacobian = jacobian_aggregated[0, :]  # Shape: (n_parameters,)
                    else:
                        jacobian = jacobian_aggregated  # Already 1D
                    
                    # Handle NaN values
                    jacobian = np.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Ensure correct length
                    expected_length = len(self.param_manager.all_param_names)
                    if len(jacobian) != expected_length:
                        self.logger.warning(f"Jacobian length ({len(jacobian)}) doesn't match "
                                        f"parameter count ({expected_length})")
                        # Pad or truncate as needed
                        if len(jacobian) < expected_length:
                            jacobian = np.pad(jacobian, (0, expected_length - len(jacobian)))
                        else:
                            jacobian = jacobian[:expected_length]
                    
                    return jacobian
                else:
                    self.logger.error(f"Unexpected jacobian shape: {jacobian_raw.shape}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error extracting FUSE jacobian: {str(e)}")
            return None

    def _jacobian_to_gradients(self, jacobian: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Convert FUSE jacobian to PyTorch gradients for the optimization metric
        
        The jacobian from FUSE represents d(model_output)/d(parameters).
        We need d(optimization_metric)/d(parameters).
        
        For simple cases where the metric is directly the model output (like streamflow),
        this is straightforward. For complex metrics like KGE or NSE, we need to
        account for the chain rule.
        """
        try:
            # For now, assume the jacobian directly represents gradients of our metric
            # This works for cases where we're optimizing streamflow directly
            
            # The sign depends on whether we're maximizing or minimizing
            if self.optimization_metric.upper() in ['KGE', 'NSE']:
                # Maximizing - use negative gradients (for minimization in optimizer)
                gradients = -torch.tensor(jacobian, dtype=torch.float32, device=device)
            else:
                # Minimizing (RMSE, MAE) - use positive gradients
                gradients = torch.tensor(jacobian, dtype=torch.float32, device=device)
            
            # Handle invalid gradients
            gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gradient clipping for stability
            gradient_norm = torch.norm(gradients)
            if gradient_norm > 1.0:
                gradients = gradients / gradient_norm
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Error converting jacobian to gradients: {str(e)}")
            return torch.zeros(len(self.param_manager.all_param_names), device=device)

    def _finite_difference_gradients(self, x: torch.Tensor, step_size: float = 1e-4) -> torch.Tensor:
        """
        Fallback method to compute gradients using finite differences
        when jacobian extraction fails
        """
        try:
            x_np = x.detach().cpu().numpy()
            n_params = len(x_np)
            gradients = np.zeros(n_params)
            
            # Get baseline evaluation
            params_base = self.param_manager.denormalize_parameters(x_np)
            self.update_constraint_defaults(params_base)
            self._run_fuse_model_with_constraints()
            
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            metrics_base = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            fitness_base = self._extract_target_metric(metrics_base) if metrics_base else 0.0
            
            # Compute finite differences
            for i in range(n_params):
                x_pert = x_np.copy()
                x_pert[i] = np.clip(x_pert[i] + step_size, 0.0, 1.0)
                
                params_pert = self.param_manager.denormalize_parameters(x_pert)
                self.update_constraint_defaults(params_pert)
                success = self._run_fuse_model_with_constraints()
                
                if success:
                    metrics_pert = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
                    fitness_pert = self._extract_target_metric(metrics_pert) if metrics_pert else fitness_base
                    gradients[i] = (fitness_pert - fitness_base) / step_size
                else:
                    gradients[i] = 0.0
            
            # Convert to tensor with appropriate sign
            if self.optimization_metric.upper() in ['KGE', 'NSE']:
                gradients = -gradients  # Negative for maximization
            
            return torch.tensor(gradients, dtype=torch.float32, device=x.device)
            
        except Exception as e:
            self.logger.error(f"Error in finite difference gradients: {str(e)}")
            return torch.zeros_like(x)

    # Multi-objective differentiable optimization

    def run_differentiable_multiobjective(self, optimizer: str = 'ADAM', steps: int = 100, 
                                        lr: float = 0.01, objectives: List[str] = None,
                                        weights: List[float] = None,
                                        initial_params: Optional[Dict[str, float]] = None) -> Path:
        """
        Run multi-objective differentiable optimization using weighted sum approach
        
        Args:
            optimizer: 'ADAM' or 'LBFGS'
            steps: Number of optimization steps
            lr: Learning rate
            objectives: List of objective names (e.g., ['NSE', 'KGE'])
            weights: Weights for each objective (must sum to 1.0)
            initial_params: Initial parameter values
        """
        objectives = objectives or ['NSE', 'KGE']
        weights = weights or [0.5, 0.5]
        
        if len(objectives) != len(weights):
            raise ValueError("Number of objectives must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.logger.info(f"Starting multi-objective differentiable optimization")
        self.logger.info(f"Objectives: {objectives}")
        self.logger.info(f"Weights: {weights}")
        
        # Store multi-objective settings
        self._multi_objectives = objectives
        self._multi_weights = weights
        
        # Run optimization with modified evaluation
        return self._run_differentiable_optimization(optimizer.upper(), steps, lr, initial_params)

    def _fuse_autograd_step_multiobjective(self, x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Multi-objective version of autograd step using weighted sum"""
        try:
            x_np = x.detach().cpu().numpy()
            params = self.param_manager.denormalize_parameters(x_np)
            
            # Update constraints and run FUSE
            success = self.update_constraint_defaults(params)
            if not success:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            success = self._run_fuse_model_with_constraints()
            if not success:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            # Calculate all metrics
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            metrics = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            if not metrics:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            # Extract jacobian
            jacobian = self._extract_fuse_jacobian(fuse_sim_dir)
            if jacobian is None:
                return self._get_worst_loss(), torch.zeros_like(x)
            
            # Calculate weighted objective
            weighted_fitness = 0.0
            for obj_name, weight in zip(self._multi_objectives, self._multi_weights):
                obj_value = self._extract_target_metric_by_name(metrics, obj_name)
                if obj_value is not None and not np.isnan(obj_value):
                    weighted_fitness += weight * obj_value
            
            # For gradients, use the same jacobian (simplified approach)
            # In practice, you'd want separate jacobians for each objective
            gradients = self._jacobian_to_gradients(jacobian, x.device)
            
            return weighted_fitness, gradients
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective autograd step: {str(e)}")
            return self._get_worst_loss(), torch.zeros_like(x)

    def run_nsga2(self) -> Path:
        """Run NSGA-II (Non-dominated Sorting Genetic Algorithm II) for FUSE"""
        self.logger.info("Starting FUSE calibration with NSGA-II")
        
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
        
        # Initialize NSGA-II state variables
        self.population = None
        self.population_objectives = None  # Shape: (pop_size, num_objectives)
        self.population_ranks = None
        self.population_crowding_distances = None
        self.pareto_front = None
        
        # Initialize population
        self._initialize_nsga2_population()
        
        # Run NSGA-II iterations
        return self._run_nsga2_algorithm()

    def _determine_nsga2_population_size(self) -> int:
        """Determine appropriate population size for NSGA-II"""
        param_count = len(self.param_manager.all_param_names)
        # NSGA-II typically needs larger populations for good diversity
        config_size = self.config.get('POPULATION_SIZE')
        if config_size:
            return max(config_size, 30)  # Minimum for NSGA-II
        return max(50, min(8 * param_count, 100))

    def _initialize_nsga2_population(self) -> None:
        """Initialize NSGA-II population"""
        self.logger.info(f"Initializing NSGA-II population with {self.population_size} individuals")
        
        param_count = len(self.param_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.uniform(0, 1, (self.population_size, param_count))
        self.population_objectives = np.full((self.population_size, self.num_objectives), np.nan)
        
        # Set first individual to initial parameters if available
        try:
            initial_params = self.param_manager.get_initial_parameters()
            if initial_params:
                initial_normalized = self.param_manager.normalize_parameters(initial_params)
                self.population[0] = np.clip(initial_normalized, 0, 1)
        except Exception:
            self.logger.warning("Could not set initial parameters, using random start")
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population for multi-objective optimization...")
        self._evaluate_nsga2_population()
        
        # Perform initial non-dominated sorting and crowding distance calculation
        self._perform_nsga2_selection()
        
        # Update representative solution for compatibility
        self._update_nsga2_representative_solution()
        
        front_1_size = np.sum(self.population_ranks == 0)
        self.logger.info(f"Initial population evaluated. Pareto front size: {front_1_size}")

    def _evaluate_nsga2_population(self) -> None:
        """Evaluate the entire NSGA-II population for multiple objectives"""
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                try:
                    objectives = self._evaluate_particle_multiobjective(self.population[i], 0, i)
                    if objectives is not None:
                        self.population_objectives[i] = objectives
                    else:
                        # Set to worst possible values
                        self.population_objectives[i] = [-1.0] * self.num_objectives
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating individual {i}: {str(e)}")
                    self.population_objectives[i] = [-1.0] * self.num_objectives

    def _evaluate_particle_multiobjective(self, normalized_params: np.ndarray, iteration: int, particle_id: int) -> Optional[List[float]]:
        """Evaluate a particle for multiple objectives"""
        try:
            # Denormalize parameters
            params = self.param_manager.denormalize_parameters(normalized_params)
            
            # Update constraints file
            success = self.update_constraint_defaults(params)
            if not success:
                return [-1.0] * self.num_objectives
            
            # Run FUSE model
            success = self._run_fuse_model_with_constraints()
            if not success:
                return [-1.0] * self.num_objectives
            
            # Calculate metrics using calibration target
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            # Calculate all metrics
            metrics = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            if not metrics:
                return [-1.0] * self.num_objectives
            
            # Extract objectives
            objectives = []
            for obj_name in self.objectives:
                obj_value = self._extract_target_metric_by_name(metrics, obj_name)
                if obj_value is None or np.isnan(obj_value):
                    obj_value = -1.0
                objectives.append(obj_value)
            
            return objectives
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters for multi-objective: {str(e)}")
            return [-1.0] * self.num_objectives

    def _extract_target_metric_by_name(self, metrics: Dict[str, float], metric_name: str) -> Optional[float]:
        """Extract specific metric by name from metrics dictionary"""
        # Try exact match
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Try with calibration prefix
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try without prefix
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"):
                return value
        
        # Try case-insensitive match
        for key, value in metrics.items():
            if key.upper() == metric_name.upper():
                return value
        
        self.logger.warning(f"Metric {metric_name} not found in calculated metrics: {list(metrics.keys())}")
        return None

    def _run_nsga2_algorithm(self) -> Path:
        """Core NSGA-II algorithm implementation"""
        self.logger.info("Starting NSGA-II evolution")
        
        for generation in range(1, self.max_iterations + 1):
            self.logger.info(f"NSGA-II Generation {generation}/{self.max_iterations}")
            
            # Generate offspring population
            offspring = self._generate_nsga2_offspring()
            
            # Evaluate offspring
            offspring_objectives = self._evaluate_nsga2_offspring(offspring)
            
            # Combine parent and offspring populations
            combined_population = np.vstack([self.population, offspring])
            combined_objectives = np.vstack([self.population_objectives, offspring_objectives])
            
            # Environmental selection using NSGA-II
            selected_indices = self._environmental_selection(combined_population, combined_objectives)
            
            # Update population
            self.population = combined_population[selected_indices]
            self.population_objectives = combined_objectives[selected_indices]
            
            # Perform non-dominated sorting and crowding distance
            self._perform_nsga2_selection()
            
            # Update representative solution
            self._update_nsga2_representative_solution()
            
            # Record generation statistics
            self._record_nsga2_generation(generation)
            
            # Log progress
            front_1_size = np.sum(self.population_ranks == 0)
            avg_objectives = np.nanmean(self.population_objectives, axis=0)
            
            obj_str = ", ".join([f"Avg_{obj}={avg_obj:.4f}" for obj, avg_obj in zip(self.objectives, avg_objectives)])
            self.logger.info(f"Generation {generation} complete. Pareto front size: {front_1_size}, {obj_str}")
        
        # Extract final Pareto front
        pareto_indices = np.where(self.population_ranks == 0)[0]
        self.pareto_front = {
            'solutions': self.population[pareto_indices],
            'objectives': self.population_objectives[pareto_indices],
            'parameters': [self.param_manager.denormalize_parameters(sol) for sol in self.population[pareto_indices]]
        }
        
        self.logger.info(f"NSGA-II optimization completed. Final Pareto front size: {len(pareto_indices)}")
        
        return self._save_nsga2_results()

    def _generate_nsga2_offspring(self) -> np.ndarray:
        """Generate offspring using tournament selection, crossover, and mutation"""
        offspring = np.zeros_like(self.population)
        
        for i in range(0, self.population_size, 2):
            # Tournament selection for parents
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._simulated_binary_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)
            
            offspring[i] = child1
            if i + 1 < self.population_size:
                offspring[i + 1] = child2
        
        return offspring

    def _tournament_selection(self, tournament_size: int = 2) -> int:
        """Tournament selection based on dominance and crowding distance"""
        candidates = np.random.choice(self.population_size, tournament_size, replace=False)
        
        best_idx = candidates[0]
        for i in range(1, len(candidates)):
            candidate_idx = candidates[i]
            
            # Compare based on rank (lower is better)
            if self.population_ranks[candidate_idx] < self.population_ranks[best_idx]:
                best_idx = candidate_idx
            elif self.population_ranks[candidate_idx] == self.population_ranks[best_idx]:
                # Same rank, compare crowding distance (higher is better)
                if self.population_crowding_distances[candidate_idx] > self.population_crowding_distances[best_idx]:
                    best_idx = candidate_idx
        
        return best_idx

    def _simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    # Calculate beta
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2.0 * rand) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (self.eta_c + 1.0))
                    
                    # Calculate children
                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Ensure bounds
                    child1[i] = np.clip(child1[i], 0, 1)
                    child2[i] = np.clip(child2[i], 0, 1)
        
        return child1, child2

    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                rand = np.random.random()
                if rand < 0.5:
                    delta = (2.0 * rand) ** (1.0 / (self.eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (self.eta_m + 1.0))
                
                mutated[i] = individual[i] + delta
                mutated[i] = np.clip(mutated[i], 0, 1)
        
        return mutated

    def _evaluate_nsga2_offspring(self, offspring: np.ndarray) -> np.ndarray:
        """Evaluate offspring population"""
        offspring_objectives = np.full((len(offspring), self.num_objectives), np.nan)
        
        for i in range(len(offspring)):
            try:
                objectives = self._evaluate_particle_multiobjective(offspring[i], 0, i)
                if objectives is not None:
                    offspring_objectives[i] = objectives
                else:
                    offspring_objectives[i] = [-1.0] * self.num_objectives
            except Exception as e:
                self.logger.error(f"Error evaluating offspring {i}: {str(e)}")
                offspring_objectives[i] = [-1.0] * self.num_objectives
        
        return offspring_objectives

    def _environmental_selection(self, combined_population: np.ndarray, combined_objectives: np.ndarray) -> np.ndarray:
        """Environmental selection using NSGA-II principles"""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(combined_objectives)
        
        # Calculate crowding distances for each front
        crowding_distances = np.zeros(len(combined_population))
        for front in fronts:
            if len(front) > 0:
                distances = self._calculate_crowding_distance(combined_objectives[front])
                crowding_distances[front] = distances
        
        # Select individuals
        selected_indices = []
        current_front = 0
        
        while len(selected_indices) + len(fronts[current_front]) <= self.population_size:
            selected_indices.extend(fronts[current_front])
            current_front += 1
            if current_front >= len(fronts):
                break
        
        # If we need to select some individuals from the current front
        if len(selected_indices) < self.population_size and current_front < len(fronts):
            remaining_slots = self.population_size - len(selected_indices)
            current_front_indices = fronts[current_front]
            current_front_distances = crowding_distances[current_front_indices]
            
            # Sort by crowding distance (descending)
            sorted_indices = np.argsort(current_front_distances)[::-1]
            selected_from_current = [current_front_indices[i] for i in sorted_indices[:remaining_slots]]
            selected_indices.extend(selected_from_current)
        
        return np.array(selected_indices)

    def _non_dominated_sorting(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting"""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
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
        
        return fronts[:-1]  # Remove empty last front

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        # For maximization objectives (NSE, KGE should be maximized)
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _calculate_crowding_distance(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for a set of objectives"""
        n = len(objectives)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        
        for m in range(self.num_objectives):
            # Sort by objective m
            sorted_indices = np.argsort(objectives[:, m])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Calculate range
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m]
                    ) / obj_range
        
        return distances

    def _perform_nsga2_selection(self) -> None:
        """Perform non-dominated sorting and crowding distance calculation"""
        fronts = self._non_dominated_sorting(self.population_objectives)
        
        # Assign ranks
        self.population_ranks = np.zeros(self.population_size, dtype=int)
        for rank, front in enumerate(fronts):
            self.population_ranks[front] = rank
        
        # Calculate crowding distances
        self.population_crowding_distances = np.zeros(self.population_size)
        for front in fronts:
            if len(front) > 0:
                distances = self._calculate_crowding_distance(self.population_objectives[front])
                self.population_crowding_distances[front] = distances

    def _update_nsga2_representative_solution(self) -> None:
        """Update representative solution for compatibility with single-objective interface"""
        # Find best solution based on a composite metric
        front_1_indices = np.where(self.population_ranks == 0)[0]
        
        if len(front_1_indices) > 0:
            # Use a composite metric (e.g., sum of normalized objectives)
            front_1_objectives = self.population_objectives[front_1_indices]
            
            # Normalize objectives to [0,1] range (assuming NSE and KGE are in [-1,1])
            composite_scores = np.mean(front_1_objectives, axis=1)  # Simple average
            best_idx_in_front = np.argmax(composite_scores)
            best_idx = front_1_indices[best_idx_in_front]
            
            self.best_params = self.param_manager.denormalize_parameters(self.population[best_idx])
            self.best_fitness = composite_scores[best_idx_in_front]
        else:
            # Fallback if no valid solutions
            self.best_params = None
            self.best_fitness = -1.0

    def _record_nsga2_generation(self, generation: int) -> None:
        """Record NSGA-II generation statistics"""
        # Calculate statistics for each objective
        objective_stats = {}
        for i, obj_name in enumerate(self.objectives):
            valid_values = self.population_objectives[:, i][~np.isnan(self.population_objectives[:, i])]
            objective_stats[f'{obj_name}_mean'] = np.mean(valid_values) if len(valid_values) > 0 else np.nan
            objective_stats[f'{obj_name}_std'] = np.std(valid_values) if len(valid_values) > 0 else np.nan
            objective_stats[f'{obj_name}_best'] = np.max(valid_values) if len(valid_values) > 0 else np.nan
        
        # Count individuals in each front
        front_counts = {}
        for rank in range(int(np.max(self.population_ranks)) + 1):
            front_counts[f'front_{rank}'] = np.sum(self.population_ranks == rank)
        
        # Store results
        generation_result = {
            'iteration': generation,
            'algorithm': 'NSGA-II',
            'fitness': self.best_fitness,
            'parameters': self.best_params,
            **objective_stats,
            **front_counts,
            'pareto_front_size': front_counts.get('front_0', 0)
        }
        
        self.iteration_results.append(generation_result)

    def _save_nsga2_results(self) -> Path:
        """Save NSGA-II optimization results"""
        # Save generation results
        results_file = self.results_dir / f"{self.experiment_id}_fuse_nsga2_results.csv"
        results_df = pd.DataFrame(self.iteration_results)
        
        # Expand parameters into separate columns if available
        if len(self.iteration_results) > 0 and 'parameters' in self.iteration_results[0]:
            param_df = pd.json_normalize(results_df['parameters'])
            param_df.columns = [f'param_{col}' for col in param_df.columns]
            results_df = pd.concat([results_df.drop('parameters', axis=1), param_df], axis=1)
        
        results_df.to_csv(results_file, index=False)
        
        # Save Pareto front
        if self.pareto_front is not None:
            pareto_file = self.results_dir / f"{self.experiment_id}_fuse_nsga2_pareto_front.csv"
            
            # Create DataFrame with solutions and objectives
            pareto_data = []
            for i, (solution, objectives, parameters) in enumerate(zip(
                self.pareto_front['solutions'],
                self.pareto_front['objectives'], 
                self.pareto_front['parameters']
            )):
                row = {'solution_id': i}
                
                # Add objectives
                for j, obj_name in enumerate(self.objectives):
                    row[obj_name] = objectives[j]
                
                # Add parameters
                for param_name, param_value in parameters.items():
                    row[f'param_{param_name}'] = param_value
                
                pareto_data.append(row)
            
            pareto_df = pd.DataFrame(pareto_data)
            pareto_df.to_csv(pareto_file, index=False)
            
            self.logger.info(f"Pareto front saved to {pareto_file}")
        
        # Save best representative solution
        best_params_file = self.results_dir / f"{self.experiment_id}_fuse_nsga2_best_representative.csv"
        if self.best_params:
            best_df = pd.DataFrame([self.best_params])
            best_df['best_fitness'] = self.best_fitness
            best_df['algorithm'] = 'NSGA-II'
            best_df.to_csv(best_params_file, index=False)
        
        self.logger.info(f"NSGA-II results saved to {results_file}")
        self.logger.info(f"Best representative solution saved to {best_params_file}")
        self.logger.info(f"Final Pareto front size: {len(self.pareto_front['solutions']) if self.pareto_front else 0}")
        
        return results_file


    def _determine_de_population_size(self) -> int:
        """Determine appropriate population size for DE based on parameter count"""
        param_count = len(self.param_manager.all_param_names)
        # DE rule of thumb: 5-10 times the number of parameters
        return max(15, min(6 * param_count, 60))

    def _initialize_de_population(self) -> None:
        """Initialize DE population"""
        self.logger.info(f"Initializing DE population with {self.population_size} individuals")
        
        param_count = len(self.param_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.uniform(0, 1, (self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Set first individual to current best or default parameters if available
        try:
            initial_params = self.param_manager.get_initial_parameters()
            if initial_params:
                initial_normalized = self.param_manager.normalize_parameters(initial_params)
                self.population[0] = np.clip(initial_normalized, 0, 1)
        except Exception:
            self.logger.warning("Could not set initial parameters, using random start")
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_de_population()
        
        # Find best individual
        best_idx = np.nanargmax(self.population_scores)
        if not np.isnan(self.population_scores[best_idx]):
            if self._is_better_fitness(self.population_scores[best_idx], self.best_fitness):
                self.best_fitness = self.population_scores[best_idx]
                self.best_params = self.param_manager.denormalize_parameters(self.population[best_idx])
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_fitness:.4f}")

    def _evaluate_de_population(self) -> None:
        """Evaluate the entire DE population"""
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                try:
                    score = self._evaluate_particle(self.population[i], 0, i)
                    self.population_scores[i] = score if score is not None else float('-inf')
                    
                    # Update best fitness if this score is better
                    if score is not None and self._is_better_fitness(score, self.best_fitness):
                        self.best_fitness = score
                        self.best_params = self.param_manager.denormalize_parameters(self.population[i])
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating individual {i}: {str(e)}")
                    self.population_scores[i] = float('-inf')

    def _run_de_algorithm(self) -> Path:
        """Core DE algorithm implementation"""
        self.logger.info("Starting DE evolution")
        
        for generation in range(1, self.max_iterations + 1):
            self.logger.info(f"DE Generation {generation}/{self.max_iterations}")
            
            # Create trial population using DE mutation and crossover
            trial_population = self._create_de_trial_population()
            
            # Evaluate trial population
            trial_scores = self._evaluate_de_trial_population(trial_population)
            
            # Selection phase: keep better solutions
            improvements = 0
            for i in range(self.population_size):
                if (not np.isnan(trial_scores[i]) and 
                    self._is_better_fitness(trial_scores[i], self.population_scores[i])):
                    
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    improvements += 1
                    
                    # Update global best
                    if self._is_better_fitness(trial_scores[i], self.best_fitness):
                        self.best_fitness = trial_scores[i]
                        self.best_params = self.param_manager.denormalize_parameters(trial_population[i])
                        self.logger.info(f"Generation {generation}: NEW BEST! {self.optimization_metric}={self.best_fitness:.4f}")
            
            # Store iteration results
            self.iteration_results.append({
                'iteration': generation,
                'particle': 0,  # Representative for the iteration
                'fitness': self.best_fitness,
                'parameters': self.best_params,
                'improvements': improvements,
                'population_size': self.population_size
            })
            
            self.logger.info(f"Generation {generation} complete. Best {self.optimization_metric}: {self.best_fitness:.4f}, "
                            f"Improvements: {improvements}/{self.population_size}")
        
        return self._save_results('DE')

    def _create_de_trial_population(self) -> np.ndarray:
        """Create trial population using DE mutation and crossover"""
        trial_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Select three random individuals different from target
            candidates = list(range(self.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: V = X_r1 + F * (X_r2 - X_r3)
            mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = self.population[i].copy()
            j_rand = np.random.randint(len(self.param_manager.all_param_names))
            
            for j in range(len(self.param_manager.all_param_names)):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_population[i] = trial
        
        return trial_population

    def _evaluate_de_trial_population(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population"""
        trial_scores = np.full(self.population_size, np.nan)
        
        for i in range(self.population_size):
            try:
                score = self._evaluate_particle(trial_population[i], 0, i)
                trial_scores[i] = score if score is not None else float('-inf')
            except Exception as e:
                self.logger.error(f"Error evaluating trial individual {i}: {str(e)}")
                trial_scores[i] = float('-inf')
        
        return trial_scores

    def run_dds(self) -> Path:
        """Run Dynamically Dimensioned Search for FUSE"""
        self.logger.info("Starting FUSE calibration with DDS")
        
        # DDS parameters
        r = self.config.get('DDS_R', 0.2)  # Perturbation factor
        
        # Initialize with current best or random
        current_solution = np.random.uniform(0, 1, len(self.param_manager.all_param_names))
        current_fitness = self._evaluate_particle(current_solution, 0, 0)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        self.best_fitness = best_fitness
        self.best_params = self.param_manager.denormalize_parameters(best_solution)
        
        # Main DDS loop
        for iteration in range(1, self.max_iterations):
            self.logger.info(f"DDS Iteration {iteration + 1}/{self.max_iterations}")
            
            # Determine number of parameters to perturb
            pn = 1.0 - np.log(iteration + 1) / np.log(self.max_iterations)
            num_perturb = max(1, int(pn * len(self.param_manager.all_param_names)))
            
            # Select parameters to perturb
            perturb_indices = np.random.choice(
                len(self.param_manager.all_param_names), 
                num_perturb, 
                replace=False
            )
            
            # Create new solution
            new_solution = current_solution.copy()
            for idx in perturb_indices:
                new_solution[idx] = current_solution[idx] + np.random.normal(0, r)
                new_solution[idx] = np.clip(new_solution[idx], 0, 1)
            
            # Evaluate new solution
            try:
                new_fitness = self._evaluate_particle(new_solution, iteration, 0)
                
                # Accept if better
                if self._is_better_fitness(new_fitness, current_fitness):
                    current_solution = new_solution
                    current_fitness = new_fitness
                    
                    if self._is_better_fitness(new_fitness, best_fitness):
                        best_solution = new_solution.copy()
                        best_fitness = new_fitness
                        self.best_fitness = best_fitness
                        self.best_params = self.param_manager.denormalize_parameters(best_solution)
                
                self.logger.info(f"Iteration {iteration + 1}: {self.optimization_metric}={new_fitness:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in DDS iteration {iteration + 1}: {str(e)}")
                new_fitness = -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # Store results
            self.iteration_results.append({
                'iteration': iteration + 1,
                'particle': 0,
                'fitness': new_fitness,
                'parameters': self.param_manager.denormalize_parameters(new_solution)
            })
        
        return self._save_results('DDS')



    def _create_parameter_file_for_iteration(self, params: Dict[str, float], iteration: int) -> Path:
        # Use shorter directory name
        iter_dir = self.param_manager.fuse_sim_dir / f"i{iteration:03d}"  # "i001" instead of "iter_0001"
        iter_dir.mkdir(exist_ok=True)
        
        param_file = iter_dir / f"p{iteration:03d}.nc"  # "p001.nc" instead of "para_iter_0001.nc"

        
        # Copy structure from para_def.nc and update values
        with xr.open_dataset(self.param_manager.para_def_path) as template:  # Use param_manager's path
            ds = template.copy()
            
            # Update with new parameter values (use only first parameter set)
            for param_name, param_value in params.items():
                if param_name in ds.variables:
                    ds[param_name][0] = param_value
        
        # Save the modified parameter file
        ds.to_netcdf(param_file)
        
        # VERIFY the file was created and is readable
        if not param_file.exists():
            self.logger.error(f"Failed to create parameter file: {param_file}")
            return None
        
        try:
            # Test that the file can be opened
            with xr.open_dataset(param_file) as test_ds:
                self.logger.debug(f"Parameter file created successfully: {param_file}")
        except Exception as e:
            self.logger.error(f"Parameter file corrupted: {param_file}, error: {e}")
            return None
        
        return param_file

    def _create_parameter_list_file(self, param_file: Path, iteration: int) -> Path:
        """Create text file listing parameter NetCDF files"""
        
        list_file = param_file.parent / f"para_list_{iteration:04d}.txt"
        
        # Debug: log the paths being used
        self.logger.debug(f"Creating list file: {list_file}")
        self.logger.debug(f"Parameter file path: {param_file}")
        
        # Ensure parameter file exists before adding to list
        if not param_file.exists():
            self.logger.error(f"Parameter file does not exist: {param_file}")
            return None
        
        # Write with explicit UTF-8 encoding and newline
        with open(list_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"{param_file.resolve().as_posix()}\n")
        
        # Debug: read back and verify content
        with open(list_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.logger.debug(f"List file content: '{content.strip()}'")
        
        return list_file

    def _run_fuse_model(self, params: Dict[str, float], iteration: int) -> bool:
        """Execute FUSE model with run_pre mode"""
        try:
            # Create parameter file and list file
            param_file = self._create_parameter_file_for_iteration(params, iteration)
            list_file = self._create_parameter_list_file(param_file, iteration)
            
            from utils.models.fuse_utils import FUSERunner
            fuse_runner = FUSERunner(self.config, self.logger)
            
            # Use run_pre with parameter list file
            success = fuse_runner._execute_fuse('run_pre', list_file)
            return bool(success)
            
        except Exception as e:
            self.logger.error(f"Error running FUSE model: {str(e)}")
            return False

    def update_constraint_defaults(self, params: Dict[str, float]) -> bool:
        constraint_file = self.param_manager.fuse_setup_dir / 'fuse_zConstraints_snow.txt'
        self.logger.debug(f"Updating constraints file: {constraint_file}")
        
        try:
            with open(constraint_file, 'r') as f:
                lines = f.readlines()
            
            updates_made = 0
            for i, line in enumerate(lines):
                if line.startswith('T 0') or line.startswith('F 0'):
                    parts = line.split()
                    if len(parts) >= 14:  # Need at least 14 parts
                        param_name = parts[13]  # FIXED: Parameter name is at index 13, not 12
                        
                        if param_name in params:
                            old_value = float(parts[2])
                            new_value = params[param_name]
                            
                            # DEBUG: Log the change
                            self.logger.debug(f"  {param_name}: {old_value:.3f} -> {new_value:.3f}")
                            
                            # CRITICAL: Use exact FORTRAN fixed-width format
                            safe = new_value
                            if param_name.lower() in {"route_shape","route_gamma_shape","rout_shape","alpha_route","nroute"}:
                                safe = max(new_value, 1e-3)
                            if param_name.lower() in {"route_scale","route_vel","krout","tau_route"}:
                                safe = max(new_value, 1e-6)
                            formatted_value = f"{safe:9.3f}"
                            
                            # Replace the exact character positions
                            start_pos = 4  # After "T 0 "
                            end_pos = start_pos + 9  # 9 characters for F9.3
                            
                            # Replace in the original line preserving exact positions
                            new_line = line[:start_pos] + formatted_value + line[end_pos:]
                            lines[i] = new_line
                            updates_made += 1
            
            # Write back
            with open(constraint_file, 'w') as f:
                f.writelines(lines)
            
            self.logger.debug(f"Made {updates_made} parameter updates to constraints file")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating constraints file: {str(e)}")
            return False

    def _evaluate_particle(self, normalized_params: np.ndarray, iteration: int, particle_id: int) -> float:
        try:
            # Denormalize parameters
            params = self.param_manager.denormalize_parameters(normalized_params)
            # DEBUG: Log parameter values for this iteration
            self.logger.debug(f"Iteration {iteration} parameters:")
            for param_name, param_value in params.items():
                self.logger.debug(f"  {param_name}: {param_value:.6f}")
            
            # Validate parameters
            #if not self.param_manager.validate_parameters(params):
            #    return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # FIXED: Update constraints file instead of parameter files
            self.logger.debug(f"About to update constraints file for iteration {iteration}")
            success = self.update_constraint_defaults(params)
            self.logger.debug(f"Constraints update result: {success}")
                        
            # Run FUSE model

            success = self._run_fuse_model_with_constraints()
            if not success:
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            self.logger.debug(f"fuse run: {success}")

            # Calculate metrics using calibration target
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            self.logger.debug(f"about to calculate metrics")
            # Calculate metrics
            metrics = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            self.logger.debug(f"metrics calculated")

            if not metrics:
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # Extract target metric
            fitness = self._extract_target_metric(metrics)
            
            if fitness is None or np.isnan(fitness):
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {str(e)}")
            return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf

    def _run_fuse_model_with_constraints(self) -> bool:
        """Execute FUSE model using run_def with updated constraints"""
        try:
            from utils.models.fuse_utils import FUSERunner
            fuse_runner = FUSERunner(self.config, self.logger)
            
            # Use run_def - it will read the updated constraints
            success = fuse_runner._execute_fuse('run_def')
            return bool(success)
            
        except Exception as e:
            self.logger.error(f"Error running FUSE model: {str(e)}")
        return False

    def _extract_target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        """Extract target metric from metrics dictionary"""
        # Try exact match
        if self.optimization_metric in metrics:
            return metrics[self.optimization_metric]
        
        # Try with calibration prefix
        calib_key = f"Calib_{self.optimization_metric}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try without prefix
        for key, value in metrics.items():
            if key.endswith(f"_{self.optimization_metric}"):
                return value
        
        # Fallback to first available metric
        return next(iter(metrics.values())) if metrics else None
    
    def _is_better_fitness(self, new_fitness: float, current_fitness: float) -> bool:
        """Check if new fitness is better than current"""
        if self.optimization_metric.upper() in ['KGE', 'NSE', 'KGEP']:
            return new_fitness > current_fitness
        else:  # RMSE, MAE, etc. (lower is better)
            return new_fitness < current_fitness
    
    def _save_results(self, algorithm: str) -> Path:
        """Save optimization results to file"""
        results_file = self.results_dir / f"{self.experiment_id}_fuse_{algorithm.lower()}_results.csv"
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.iteration_results)
        
        # Expand parameters into separate columns
        if len(self.iteration_results) > 0:
            param_df = pd.json_normalize(results_df['parameters'])
            param_df.columns = [f'param_{col}' for col in param_df.columns]
            results_df = pd.concat([results_df.drop('parameters', axis=1), param_df], axis=1)
        
        # Save to CSV
        results_df.to_csv(results_file, index=False)
        
        # Save best parameters separately
        best_params_file = self.results_dir / f"{self.experiment_id}_fuse_best_parameters.csv"
        if self.best_params:
            best_df = pd.DataFrame([self.best_params])
            best_df['best_fitness'] = self.best_fitness
            best_df['algorithm'] = algorithm
            best_df.to_csv(best_params_file, index=False)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Best parameters saved to {best_params_file}")
        self.logger.info(f"Best {self.optimization_metric}: {self.best_fitness:.4f}")
        
        return results_file