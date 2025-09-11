#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUSE Optimizer

Handles FUSE model calibration using various optimization algorithms.
Integrates with the existing CONFLUENCE optimization framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import time
import traceback

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
        
    def _run_fuse_model(self) -> bool:
        """Execute FUSE model run - FIXED VERSION"""
        try:
            from utils.models.fuse_utils import FUSERunner
            
            fuse_runner = FUSERunner(self.config, self.logger)
            
            # FIXED: Use 'run_def' mode during calibration iterations
            # This ensures FUSE reads from para_def.nc which we're updating
            success = fuse_runner._execute_fuse('run_best')
            return bool(success)
            
        except Exception as e:
            self.logger.error(f"Error running FUSE model: {str(e)}")
            return False

    def _evaluate_particle(self, normalized_params: np.ndarray, iteration: int, particle_id: int) -> float:
        """Evaluate a single parameter set - FIXED VERSION"""
        try:
            # Denormalize parameters
            params = self.param_manager.denormalize_parameters(normalized_params)
            
            # Validate parameters
            if not self.param_manager.validate_parameters(params):
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # FIXED: Update parameter file (use_best_file=False for calibration)
            if not self.param_manager.update_parameter_file(params, use_best_file=False):
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # FIXED: Add a small delay to ensure file write is complete
            import time
            time.sleep(0.1)  # 100ms delay to ensure file sync
            
            # Run FUSE model
            success = self._run_fuse_model()
            if not success:
                return -np.inf if self.optimization_metric.upper() in ['KGE', 'NSE'] else np.inf
            
            # Calculate metrics using calibration target
            data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
            project_dir = data_dir / f"domain_{self.domain_name}"
            fuse_sim_dir = project_dir / 'simulations' / self.experiment_id / 'FUSE'
            
            # Calculate metrics
            metrics = self.calibration_target.calculate_metrics(fuse_sim_dir, None)
            
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