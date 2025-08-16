#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CONFLUENCE DDS Optimizer - Clean Implementation

Implements Dynamically Dimensioned Search (DDS) optimization algorithm
without external dependencies beyond standard scientific Python libraries.
Uses joblib for efficient parallelization.

This implementation avoids dependency conflicts by using only:
- numpy, scipy, pandas (standard scientific stack)
- joblib (for parallelization)
- Standard library modules

No SPOTPY or other specialized optimization libraries required.

Algorithm Reference:
    Tolson, B.A. and C.A. Shoemaker (2007), Dynamically dimensioned search 
    algorithm for computationally efficient watershed model calibration, 
    Water Resources Research, 43, W01413, doi:10.1029/2005WR004723.

Key Features:
- Clean DDS implementation following the original paper
- joblib parallelization (much better than multiprocessing)
- No dependency conflicts
- Seamless CONFLUENCE integration

Usage:
    from dds_optimizer import DDSOptimizer
    
    optimizer = DDSOptimizer(config, logger)
    results = optimizer.run_optimization()
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import shutil
from typing import Dict, Any, List, Tuple, Optional, Callable
import re
import warnings
from abc import ABC, abstractmethod
import tempfile
import time

# Parallel processing with joblib (only external dependency)
from joblib import Parallel, delayed
import joblib

# Import the same helper classes from the DE optimizer
from utils.optimization.de_optimizer_scipy import (
    ParameterManager, 
    CalibrationTarget, 
    StreamflowTarget, 
    SnowTarget, 
    ResultsManager
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DDSOptimizer:
    """
    CONFLUENCE DDS Optimizer - Clean Implementation
    
    Implements the DDS algorithm with joblib parallelization and
    no external optimization library dependencies.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize DDS optimizer"""
        self.config = config
        self.logger = logger
        
        # Setup paths
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Optimization directories
        self.optimization_dir = self.project_dir / "simulations" / "run_dds_clean"
        self.output_dir = self.project_dir / "optimisation" / f"dds_clean_{self.experiment_id}"
        self.settings_dir = self.optimization_dir / "settings" / "SUMMA"
        self.summa_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_dir = self.optimization_dir / "mizuRoute"
        
        # Create directories
        for directory in [self.optimization_dir, self.output_dir, self.settings_dir, 
                         self.summa_dir, self.mizuroute_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers (using existing from DE optimizer)
        self.parameter_manager = ParameterManager(config, logger, self.settings_dir)
        self.calibration_target = self._create_calibration_target()
        self.results_manager = ResultsManager(config, logger, self.output_dir)
        
        # Copy and setup model settings
        self._setup_model_settings()
        
        # DDS parameters
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.num_workers = config.get('MPI_PROCESSES', 1)
        self.dds_r = config.get('DDS_R', 0.2)  # DDS perturbation parameter
        
        # Random seed for reproducibility
        self.random_seed = config.get('RANDOM_SEED', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Progress tracking
        self.iteration_count = 0
        self.evaluation_count = 0
        self.best_score = float('-inf')
        self.best_params = None
        self.history = []
        
        # Performance tracking
        self.failed_evaluations = 0
        self.successful_evaluations = 0
        
        # Parallel processing setup
        self.use_parallel = self.num_workers > 1
        if self.use_parallel:
            self.logger.info(f"Using joblib with {self.num_workers} workers (threading backend)")
    
    def run_optimization(self) -> Dict[str, Any]:
        """Main DDS optimization interface"""
        self.logger.info("=" * 60)
        self.logger.info(f"CONFLUENCE DDS Optimization (Clean Implementation)")
        self.logger.info(f"Domain: {self.domain_name}")
        self.logger.info(f"Target: {self.config.get('CALIBRATION_VARIABLE', 'streamflow')} ({self.target_metric})")
        self.logger.info(f"Parameters: {len(self.parameter_manager.all_param_names)}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        self.logger.info(f"DDS r parameter: {self.dds_r}")
        if self.use_parallel:
            self.logger.info(f"Parallel workers: {self.num_workers} (joblib)")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Initialize DDS
            bounds = self._get_bounds()
            n_params = len(bounds)
            
            # Get initial solution
            x_best = self._get_initial_solution()
            if x_best is None:
                # Generate random initial solution
                x_best = np.random.rand(n_params)
                self.logger.info("Using random initial solution")
            else:
                self.logger.info("Using initial solution from model defaults")
            
            # Evaluate initial solution
            score_best = self._evaluate_solution(x_best)
            
            if score_best == float('-inf'):
                raise RuntimeError("Initial solution evaluation failed")
            
            self.best_score = score_best
            self.best_params = self.parameter_manager.denormalize_parameters(x_best)
            
            self.logger.info(f"Initial solution: {self.target_metric} = {score_best:.6f}")
            
            # Record initial point in history
            self.history.append({
                'iteration': 0,
                'evaluations': self.evaluation_count,
                'best_score': self.best_score,
                'current_score': score_best,
                'improvement': True,
                'success_rate': 100.0
            })
            
            # Main DDS optimization loop
            if self.use_parallel:
                self._run_parallel_dds(x_best, score_best, bounds, n_params)
            else:
                self._run_sequential_dds(x_best, score_best, bounds, n_params)
            
            # Run final evaluation with full time period
            final_metrics = self._run_final_evaluation()
            
            # Save all results
            self._save_results(final_metrics)
            
            # Apply best parameters to default model settings
            self._apply_to_default_settings()
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            success_rate = (self.successful_evaluations / 
                          (self.successful_evaluations + self.failed_evaluations) * 100 
                          if (self.successful_evaluations + self.failed_evaluations) > 0 else 0)
            
            self.logger.info("=" * 60)
            self.logger.info("DDS OPTIMIZATION COMPLETED")
            self.logger.info(f"🏆 Best {self.target_metric}: {self.best_score:.6f}")
            self.logger.info(f"⏱️ Duration: {duration}")
            self.logger.info(f"🔢 Evaluations: {self.evaluation_count}")
            self.logger.info(f"✅ Success rate: {success_rate:.1f}%")
            self.logger.info("=" * 60)
            
            return {
                'best_parameters': self.best_params,
                'best_score': self.best_score,
                'optimization_metric': self.target_metric,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'duration': str(duration),
                'function_evaluations': self.evaluation_count,
                'iterations': self.iteration_count,
                'success': True,
                'success_rate': success_rate,
                'dds_r': self.dds_r,
                'final_metrics': final_metrics,
                'history': self.history,
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"DDS optimization failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _run_sequential_dds(self, x_best: np.ndarray, score_best: float, 
                           bounds: List[Tuple[float, float]], n_params: int):
        """Run DDS optimization sequentially"""
        self.logger.info("Running sequential DDS optimization...")
        
        for iteration in range(1, self.max_iterations + 1):
            self.iteration_count = iteration
            
            # Generate new solution using DDS perturbation
            x_new = self._dds_perturbation(x_best, bounds, iteration, n_params)
            
            # Evaluate new solution
            score_new = self._evaluate_solution(x_new)
            
            # Check for improvement
            improved = False
            if score_new > score_best:
                x_best = x_new.copy()
                score_best = score_new
                self.best_score = score_best
                self.best_params = self.parameter_manager.denormalize_parameters(x_best)
                improved = True
                
                self.logger.info(f"Iter {iteration}: NEW BEST {self.target_metric} = {score_best:.6f}")
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'evaluations': self.evaluation_count,
                'best_score': self.best_score,
                'current_score': score_new,
                'improvement': improved,
                'success_rate': (self.successful_evaluations / self.evaluation_count * 100 
                               if self.evaluation_count > 0 else 0)
            })
            
            # Periodic progress updates
            if iteration % 10 == 0:
                success_rate = self.successful_evaluations / self.evaluation_count * 100
                improvements = sum(1 for h in self.history if h.get('improvement', False))
                self.logger.info(f"Iter {iteration}/{self.max_iterations}: "
                               f"Best={self.best_score:.6f}, "
                               f"Improvements={improvements}, "
                               f"Success={success_rate:.1f}%")
    
    def _run_parallel_dds(self, x_best: np.ndarray, score_best: float, 
                         bounds: List[Tuple[float, float]], n_params: int):
        """Run DDS optimization with parallel batch evaluation"""
        self.logger.info("Running parallel DDS optimization...")
        
        # Calculate batch size for efficient parallel processing
        batch_size = min(self.num_workers * 2, max(5, self.max_iterations // 10))
        
        for batch_start in range(1, self.max_iterations + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, self.max_iterations)
            current_batch_size = batch_end - batch_start + 1
            
            # Generate batch of candidate solutions using DDS
            candidates = []
            for i in range(current_batch_size):
                iteration = batch_start + i
                x_new = self._dds_perturbation(x_best, bounds, iteration, n_params)
                candidates.append((iteration, x_new.copy()))
            
            # Evaluate candidates in parallel using joblib
            self.logger.debug(f"Evaluating batch {batch_start}-{batch_end} with {len(candidates)} candidates")
            
            # Use joblib for parallel evaluation
            with joblib.parallel_backend('threading', n_jobs=self.num_workers):
                results = Parallel()(
                    delayed(self._evaluate_candidate)(iteration, x_candidate) 
                    for iteration, x_candidate in candidates
                )
            
            # Process results and update best solution
            batch_improvements = 0
            for iteration, x_candidate, score in results:
                self.iteration_count = iteration
                
                # Check for improvement
                improved = False
                if score > score_best:
                    x_best = x_candidate.copy()
                    score_best = score
                    self.best_score = score_best
                    self.best_params = self.parameter_manager.denormalize_parameters(x_best)
                    improved = True
                    batch_improvements += 1
                    
                    self.logger.info(f"Iter {iteration}: NEW BEST {self.target_metric} = {score_best:.6f}")
                
                # Record history
                self.history.append({
                    'iteration': iteration,
                    'evaluations': self.evaluation_count,
                    'best_score': self.best_score,
                    'current_score': score,
                    'improvement': improved,
                    'success_rate': (self.successful_evaluations / self.evaluation_count * 100 
                                   if self.evaluation_count > 0 else 0)
                })
            
            # Batch progress update
            success_rate = self.successful_evaluations / self.evaluation_count * 100
            total_improvements = sum(1 for h in self.history if h.get('improvement', False))
            self.logger.info(f"Batch {batch_start}-{batch_end}: "
                           f"Best={self.best_score:.6f}, "
                           f"BatchImprovements={batch_improvements}, "
                           f"TotalImprovements={total_improvements}, "
                           f"Success={success_rate:.1f}%")
    
    def _evaluate_candidate(self, iteration: int, x_candidate: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """Evaluate a single candidate (used for parallel processing)"""
        score = self._evaluate_solution(x_candidate)
        return iteration, x_candidate, score
    
    def _dds_perturbation(self, x_current: np.ndarray, bounds: List[Tuple[float, float]], 
                         iteration: int, n_params: int) -> np.ndarray:
        """
        Generate new solution using DDS perturbation strategy
        
        This implements the core DDS algorithm from Tolson & Shoemaker (2007):
        1. Calculate probability of selecting each parameter for perturbation
        2. Randomly select parameters based on this probability  
        3. Perturb selected parameters using neighborhood-based sampling
        4. Handle boundary constraints through reflection
        
        Args:
            x_current: Current solution in normalized space [0,1]
            bounds: Parameter bounds (should be [(0,1), (0,1), ...])
            iteration: Current iteration number (used for probability calculation)
            n_params: Number of parameters
            
        Returns:
            New perturbed solution
        """
        x_new = x_current.copy()
        
        # Calculate probability of selecting each parameter for perturbation
        # This probability decreases as iterations progress, focusing the search
        if iteration <= n_params:
            # Early iterations: higher probability to explore broadly
            prob_select = 1.0 - (np.log(iteration) / np.log(n_params))
        else:
            # Later iterations: focus on fewer parameters
            prob_select = 1.0 / n_params
        
        # Ensure probability is reasonable
        prob_select = max(prob_select, 1.0 / n_params)
        prob_select = min(prob_select, 1.0)
        
        # Randomly select parameters for perturbation
        selected_params = np.random.rand(n_params) < prob_select
        
        # Ensure at least one parameter is selected (critical for DDS)
        if not np.any(selected_params):
            selected_params[np.random.randint(n_params)] = True
        
        # Perturb selected parameters
        for i in range(n_params):
            if selected_params[i]:
                current_val = x_current[i]
                
                # Calculate neighborhood size based on DDS r parameter
                # Use distance to nearest boundary to avoid constraint violations
                lower_bound, upper_bound = bounds[i]
                
                # Calculate distances to boundaries
                lower_dist = current_val - lower_bound
                upper_dist = upper_bound - current_val
                
                # Neighborhood size is fraction of smaller boundary distance
                neighborhood_size = self.dds_r * min(lower_dist, upper_dist)
                
                # Generate perturbation using normal distribution
                if neighborhood_size > 0:
                    # Standard normal perturbation scaled by neighborhood size
                    perturbation = np.random.normal(0, neighborhood_size)
                    x_new[i] = current_val + perturbation
                    
                    # Handle boundary violations using reflection
                    x_new[i] = self._reflect_at_boundaries(x_new[i], lower_bound, upper_bound)
                else:
                    # If at boundary, try small random perturbation in feasible direction
                    if current_val <= lower_bound + 1e-10:
                        x_new[i] = lower_bound + np.random.rand() * self.dds_r * (upper_bound - lower_bound)
                    elif current_val >= upper_bound - 1e-10:
                        x_new[i] = upper_bound - np.random.rand() * self.dds_r * (upper_bound - lower_bound)
        
        return x_new
    
    def _reflect_at_boundaries(self, value: float, lower: float, upper: float) -> float:
        """
        Handle boundary violations using reflection
        
        When a parameter goes outside its bounds, reflect it back into
        the feasible space. This helps maintain diversity while respecting constraints.
        """
        # Handle multiple reflections if needed
        while value < lower or value > upper:
            if value < lower:
                value = lower + (lower - value)
            elif value > upper:
                value = upper - (value - upper)
            
            # Prevent infinite loops with very small ranges
            if abs(upper - lower) < 1e-10:
                value = (lower + upper) / 2
                break
        
        return value
    
    def _evaluate_solution(self, x: np.ndarray) -> float:
        """
        Evaluate a solution (parameter set)
        
        Args:
            x: Normalized parameter vector [0,1]
            
        Returns:
            Objective value (higher is better, float('-inf') for failed runs)
        """
        self.evaluation_count += 1
        
        try:
            # Convert normalized parameters to physical parameters
            params = self.parameter_manager.denormalize_parameters(x)
            
            # Apply parameters and run models
            score = self._evaluate_parameters(params)
            
            # Handle success/failure tracking
            if score == float('-inf') or np.isnan(score):
                self.failed_evaluations += 1
                return float('-inf')
            else:
                self.successful_evaluations += 1
                
                # Occasional detailed progress updates
                if self.evaluation_count % 50 == 0:
                    success_rate = self.successful_evaluations / self.evaluation_count * 100
                    self.logger.debug(f"Eval {self.evaluation_count}: Score={score:.6f}, "
                                    f"Best={self.best_score:.6f}, Success={success_rate:.1f}%")
                
                return score
                
        except Exception as e:
            self.failed_evaluations += 1
            self.logger.debug(f"Evaluation {self.evaluation_count} failed: {str(e)}")
            return float('-inf')
    
    def _evaluate_parameters(self, params: Dict[str, np.ndarray]) -> float:
        """
        Evaluate a parameter set by running the hydrological model
        
        This method:
        1. Applies parameters to model configuration files
        2. Runs SUMMA hydrological model
        3. Runs mizuRoute routing model if needed
        4. Calculates performance metrics
        5. Returns the target metric value
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Performance score (float('-inf') for failed runs)
        """
        try:
            # Debug: Log parameter values for first few evaluations
            if self.evaluation_count <= 5:
                self.logger.info(f"Eval {self.evaluation_count} - Applying parameters:")
                for param_name, values in list(params.items())[:3]:  # Show first 3 parameters
                    if isinstance(values, np.ndarray):
                        if len(values) == 1:
                            self.logger.info(f"  {param_name}: {values[0]:.6f}")
                        else:
                            self.logger.info(f"  {param_name}: mean={np.mean(values):.6f}, range=[{np.min(values):.6f}, {np.max(values):.6f}]")
                    else:
                        self.logger.info(f"  {param_name}: {values:.6f}")
            
            # Apply parameters to model files
            param_apply_success = self.parameter_manager.apply_parameters(params)
            
            # CRITICAL FIX: Ensure the ParameterManager writes to the correct filename
            if param_apply_success:
                trial_params_file = self.settings_dir / "trialParams.nc"
                current_time = datetime.now().timestamp()
                
                # Check if the correct file was updated recently
                if trial_params_file.exists():
                    file_time = trial_params_file.stat().st_mtime
                    seconds_since_modified = current_time - file_time
                    
                    if seconds_since_modified < 5:  # Updated within 5 seconds
                        if self.evaluation_count <= 5:
                            self.logger.info(f"Eval {self.evaluation_count} - ✅ trialParams.nc correctly updated")
                    else:
                        # File not updated - look for process-suffixed files
                        process_files = list(self.settings_dir.glob("trialParams_proc*.nc"))
                        if process_files:
                            # Find the most recently created process file
                            newest_process_file = max(process_files, key=lambda x: x.stat().st_mtime)
                            
                            # Copy to correct location
                            shutil.copy2(newest_process_file, trial_params_file)
                            
                            # Clean up ALL process files (they accumulate over runs)
                            for pf in process_files:
                                try:
                                    pf.unlink()
                                except:
                                    pass  # Ignore cleanup errors
                            
                            if self.evaluation_count <= 5:
                                self.logger.info(f"Eval {self.evaluation_count} - 🔧 Fixed: copied {newest_process_file.name} to trialParams.nc (cleaned up {len(process_files)} process files)")
                        else:
                            self.logger.error(f"Eval {self.evaluation_count} - ❌ No parameter files found!")
                            param_apply_success = False
                else:
                    # trialParams.nc doesn't exist - look for process files
                    process_files = list(self.settings_dir.glob("trialParams_proc*.nc"))
                    if process_files:
                        newest_process_file = max(process_files, key=lambda x: x.stat().st_mtime)
                        shutil.copy2(newest_process_file, trial_params_file)
                        
                        # Clean up process files
                        for pf in process_files:
                            try:
                                pf.unlink()
                            except:
                                pass
                        
                        if self.evaluation_count <= 5:
                            self.logger.info(f"Eval {self.evaluation_count} - 🔧 Created trialParams.nc from {newest_process_file.name}")
                    else:
                        self.logger.error(f"Eval {self.evaluation_count} - ❌ trialParams.nc missing and no process files found!")
                        param_apply_success = False
            
            if not param_apply_success:
                self.logger.debug("Failed to apply parameters")
                return float('-inf')
            
            # Run SUMMA simulation
            summa_success = self._run_summa()
            if self.evaluation_count <= 5:
                self.logger.info(f"Eval {self.evaluation_count} - SUMMA execution: {'SUCCESS' if summa_success else 'FAILED'}")
                
                # Check if trialParams.nc was updated
                trial_params_file = self.settings_dir / "trialParams.nc"
                if trial_params_file.exists():
                    import os
                    file_time = os.path.getmtime(trial_params_file)
                    self.logger.info(f"Eval {self.evaluation_count} - trialParams.nc last modified: {datetime.fromtimestamp(file_time)}")
                else:
                    self.logger.warning(f"Eval {self.evaluation_count} - trialParams.nc does not exist!")
            
            if not summa_success:
                self.logger.debug("SUMMA simulation failed")
                return float('-inf')
            
            # Run mizuRoute routing if needed
            if self.calibration_target.needs_routing():
                if not self._run_mizuroute():
                    self.logger.debug("mizuRoute simulation failed")
                    return float('-inf')
            
            # Calculate performance metrics (calibration period only during optimization)
            metrics = self.calibration_target.calculate_metrics(
                self.summa_dir, 
                self.mizuroute_dir, 
                calibration_only=True  # CRITICAL: Only evaluate calibration period
            )
            
            # CRITICAL DEBUG: Check what's happening with time period filtering
            if self.evaluation_count <= 3:
                self.logger.info(f"Eval {self.evaluation_count} - DEBUGGING DATA ALIGNMENT:")
                
                # Check simulation output file timestamps
                output_files = list(self.summa_dir.glob("*timestep.nc"))
                if output_files:
                    newest_file = output_files[0]
                    try:
                        import xarray as xr
                        with xr.open_dataset(newest_file) as ds:
                            if 'time' in ds.dims:
                                sim_start = str(ds.time.values[0])[:19]
                                sim_end = str(ds.time.values[-1])[:19]
                                sim_count = len(ds.time)
                                self.logger.info(f"  📊 Simulation data: {sim_start} to {sim_end} ({sim_count} timesteps)")
                                
                                # Check what period should be used for calibration
                                calib_start = pd.Timestamp(self.config.get('CALIBRATION_PERIOD', '').split(',')[0].strip())
                                calib_end = pd.Timestamp(self.config.get('CALIBRATION_PERIOD', '').split(',')[1].strip())
                                self.logger.info(f"  🎯 Calibration period: {calib_start} to {calib_end}")
                                
                                # Count how many simulation timesteps fall in calibration period
                                sim_times = pd.to_datetime(ds.time.values)
                                calib_mask = (sim_times >= calib_start) & (sim_times <= calib_end)
                                calib_sim_count = calib_mask.sum()
                                self.logger.info(f"  ✂️ Simulation timesteps in calibration period: {calib_sim_count}")
                                
                                # Show spinup period that should be excluded
                                spinup_start = pd.Timestamp(self.config.get('SPINUP_PERIOD', '').split(',')[0].strip())
                                spinup_end = pd.Timestamp(self.config.get('SPINUP_PERIOD', '').split(',')[1].strip())
                                spinup_mask = (sim_times >= spinup_start) & (sim_times <= spinup_end)
                                spinup_sim_count = spinup_mask.sum()
                                self.logger.info(f"  🚫 Simulation timesteps in SPINUP (should be excluded): {spinup_sim_count}")
                                
                    except Exception as e:
                        self.logger.error(f"  ❌ Error reading simulation timestamps: {str(e)}")
                
                # Check observed data 
                try:
                    obs_path = self.calibration_target.get_observed_data_path()
                    if obs_path.exists():
                        obs_df = pd.read_csv(obs_path)
                        self.logger.info(f"  📈 Observed data file: {obs_path.name} ({len(obs_df)} records)")
                        
                        # Find date column
                        date_col = next((col for col in obs_df.columns 
                                       if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)
                        if date_col:
                            obs_dates = pd.to_datetime(obs_df[date_col])
                            obs_start = obs_dates.min()
                            obs_end = obs_dates.max()
                            self.logger.info(f"  📈 Observed data: {obs_start} to {obs_end}")
                            
                            # Check overlap with calibration period
                            calib_obs_mask = (obs_dates >= calib_start) & (obs_dates <= calib_end)
                            calib_obs_count = calib_obs_mask.sum()
                            self.logger.info(f"  🎯 Observed data in calibration period: {calib_obs_count}")
                        
                except Exception as e:
                    self.logger.error(f"  ❌ Error reading observed data: {str(e)}")
            
            # Debug: Check what output files exist
            if self.evaluation_count <= 5:
                output_files = list(self.summa_dir.glob("*.nc"))
                self.logger.info(f"Eval {self.evaluation_count} - SUMMA output files: {[f.name for f in output_files]}")
                if output_files:
                    # Check the timestamp of the newest output file
                    newest_file = max(output_files, key=lambda x: x.stat().st_mtime)
                    file_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
                    self.logger.info(f"Eval {self.evaluation_count} - Newest output file: {newest_file.name} at {file_time}")
            
            if not metrics:
                self.logger.debug("No metrics calculated")
                return float('-inf')
            
            # Debug: Log the metrics for the first few evaluations
            if self.evaluation_count <= 5:
                self.logger.info(f"Eval {self.evaluation_count} metrics: {metrics}")
                calibration_period = self.config.get('CALIBRATION_PERIOD', '')
                spinup_period = self.config.get('SPINUP_PERIOD', '')
                self.logger.info(f"Config - Calibration: {calibration_period}, Spinup: {spinup_period}")
            
            # Extract target metric value
            score = self._extract_target_metric(metrics)
            
            # Debug: Log score extraction
            if self.evaluation_count <= 5:
                self.logger.info(f"Eval {self.evaluation_count} - Target metric '{self.target_metric}': {score}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Parameter evaluation failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return float('-inf')
    
    def _extract_target_metric(self, metrics: Dict[str, float]) -> float:
        """Extract target metric from results dictionary"""
        # Try direct match first
        if self.target_metric in metrics:
            return metrics[self.target_metric]
        
        # Try with calibration prefix
        calib_key = f"Calib_{self.target_metric}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try suffix match (handles different naming conventions)
        for key, value in metrics.items():
            if key.endswith(f"_{self.target_metric}"):
                return value
        
        # Fallback to first available metric
        self.logger.warning(f"Target metric '{self.target_metric}' not found, using first available metric")
        return next(iter(metrics.values())) if metrics else float('-inf')
    
    def _run_summa(self) -> bool:
        """Run SUMMA hydrological model simulation"""
        try:
            # Get SUMMA executable path
            summa_install_path = self.config.get('SUMMA_INSTALL_PATH', 'default')
            if summa_install_path == 'default':
                summa_install_path = self.data_dir / 'installs' / 'summa' / 'bin'
            
            summa_exe = Path(summa_install_path) / self.config.get('SUMMA_EXE', 'summa.exe')
            file_manager = self.settings_dir / 'fileManager.txt'
            
            if not summa_exe.exists():
                self.logger.error(f"SUMMA executable not found: {summa_exe}")
                return False
            
            # Prepare command
            cmd = f"{summa_exe} -m {file_manager}"
            
            # Create logs directory
            log_dir = self.summa_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"summa_{self.evaluation_count:06d}.log"
            
            # Set environment for single-threaded execution (important for parallel workers)
            env = os.environ.copy()
            env.update({
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1'
            })
            
            # Run SUMMA with timeout
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                    check=True, timeout=1800, env=env, cwd=str(self.settings_dir)
                )
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.debug("SUMMA simulation timed out")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.debug(f"SUMMA failed with exit code {e.returncode}")
            return False
        except Exception as e:
            self.logger.debug(f"SUMMA execution error: {str(e)}")
            return False
    
    def _run_mizuroute(self) -> bool:
        """Run mizuRoute routing model simulation"""
        try:
            # Get mizuRoute executable path
            mizu_install_path = self.config.get('INSTALL_PATH_MIZUROUTE', 'default')
            if mizu_install_path == 'default':
                mizu_install_path = self.data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin'
            
            mizu_exe = Path(mizu_install_path) / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
            mizu_settings_dir = self.optimization_dir / 'settings' / 'mizuRoute'
            control_file = mizu_settings_dir / 'mizuroute.control'
            
            if not mizu_exe.exists() or not control_file.exists():
                return False
            
            # Prepare command
            cmd = f"{mizu_exe} {control_file}"
            
            # Create logs directory
            log_dir = self.mizuroute_dir / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"mizuroute_{self.evaluation_count:06d}.log"
            
            # Run mizuRoute with timeout
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                    check=True, timeout=1800, cwd=str(mizu_settings_dir)
                )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"mizuRoute execution error: {str(e)}")
            return False
    
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization (normalized space [0,1])"""
        return [(0.0, 1.0) for _ in self.parameter_manager.all_param_names]
    
    def _get_initial_solution(self) -> Optional[np.ndarray]:
        """Get initial parameter guess in normalized space"""
        try:
            initial_params = self.parameter_manager.get_initial_parameters()
            if initial_params:
                return self.parameter_manager.normalize_parameters(initial_params)
        except Exception as e:
            self.logger.debug(f"Could not get initial parameter guess: {str(e)}")
        return None
    
    def _run_final_evaluation(self) -> Optional[Dict]:
        """Run final evaluation with best parameters over full time period"""
        if self.best_params is None:
            return None
        
        self.logger.info("Running final evaluation with best parameters (full period)")
        
        try:
            # Update file manager for full experiment period
            self._update_file_manager_for_full_period()
            
            # Apply best parameters
            if not self.parameter_manager.apply_parameters(self.best_params):
                return None
            
            # Run models with best parameters
            if not self._run_summa():
                return None
            
            if self.calibration_target.needs_routing() and not self._run_mizuroute():
                return None
            
            # Calculate metrics for full period (both calibration and evaluation)
            final_metrics = self.calibration_target.calculate_metrics(
                self.summa_dir, 
                self.mizuroute_dir, 
                calibration_only=False  # Full period evaluation
            )
            
            if final_metrics:
                self.logger.info("Final evaluation completed successfully")
                
                # Log key metrics for different periods
                for metric_name in ['KGE', 'NSE', 'RMSE']:
                    for period in ['Calib', 'Eval']:
                        key = f"{period}_{metric_name}"
                        if key in final_metrics:
                            self.logger.info(f"Final {key}: {final_metrics[key]:.4f}")
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"Final evaluation failed: {str(e)}")
            return None
        finally:
            # Reset file manager back to calibration period
            self._update_file_manager_for_optimization()
    
    def _save_results(self, final_metrics: Optional[Dict]) -> None:
        """Save optimization results to files"""
        if self.best_params is None:
            return
        
        try:
            # Save best parameters
            param_data = []
            for param_name, values in self.best_params.items():
                if isinstance(values, np.ndarray):
                    if len(values) == 1:
                        param_data.append({
                            'parameter': param_name,
                            'value': float(values[0]),
                            'type': 'scalar'
                        })
                    else:
                        param_data.append({
                            'parameter': param_name,
                            'mean_value': float(np.mean(values)),
                            'min_value': float(np.min(values)),
                            'max_value': float(np.max(values)),
                            'std_value': float(np.std(values)),
                            'type': 'array'
                        })
                else:
                    param_data.append({
                        'parameter': param_name,
                        'value': float(values),
                        'type': 'scalar'
                    })
            
            param_df = pd.DataFrame(param_data)
            param_df.to_csv(self.output_dir / "best_parameters.csv", index=False)
            
            # Save optimization history
            if self.history:
                history_df = pd.DataFrame(self.history)
                history_df.to_csv(self.output_dir / "optimization_history.csv", index=False)
            
            # Save final metrics if available
            if final_metrics:
                metrics_df = pd.DataFrame([final_metrics])
                metrics_df.to_csv(self.output_dir / "final_metrics.csv", index=False)
            
            # Save optimization summary
            improvements = sum(1 for h in self.history if h.get('improvement', False))
            summary = {
                'domain_name': self.domain_name,
                'experiment_id': self.experiment_id,
                'algorithm': 'DDS_Clean',
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'target_metric': self.target_metric,
                'dds_r': self.dds_r,
                'best_score': self.best_score,
                'total_evaluations': self.evaluation_count,
                'successful_evaluations': self.successful_evaluations,
                'failed_evaluations': self.failed_evaluations,
                'success_rate': (self.successful_evaluations / self.evaluation_count * 100 
                               if self.evaluation_count > 0 else 0),
                'total_improvements': improvements,
                'improvement_rate': (improvements / self.iteration_count * 100 
                                   if self.iteration_count > 0 else 0),
                'parallel_workers': self.num_workers,
                'random_seed': self.random_seed,
                'completed_at': datetime.now().isoformat()
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(self.output_dir / "optimization_summary.csv", index=False)
            
            self.logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def _apply_to_default_settings(self) -> None:
        """Apply optimized parameters to default model settings"""
        if self.best_params is None:
            return
        
        try:
            self.logger.info("Applying optimized parameters to default model settings")
            
            default_settings_dir = self.project_dir / "settings" / "SUMMA"
            if not default_settings_dir.exists():
                self.logger.error("Default settings directory not found")
                return
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Apply hydraulic parameters (all except depth parameters)
            hydraulic_params = {k: v for k, v in self.best_params.items() 
                              if k not in ['total_mult', 'shape_factor']}
            
            if hydraulic_params:
                # Backup and update trialParams.nc
                trial_params_path = default_settings_dir / "trialParams.nc"
                if trial_params_path.exists():
                    backup_path = default_settings_dir / f"trialParams_backup_{timestamp}.nc"
                    shutil.copy2(trial_params_path, backup_path)
                
                # Copy optimized version
                optimized_trial = self.settings_dir / "trialParams.nc"
                if optimized_trial.exists():
                    shutil.copy2(optimized_trial, trial_params_path)
                    self.logger.info("✅ Applied optimized hydraulic parameters")
            
            # Apply soil depth parameters if calibration enabled
            if self.config.get('CALIBRATE_DEPTH', False) and 'total_mult' in self.best_params:
                coldstate_path = default_settings_dir / "coldState.nc"
                if coldstate_path.exists():
                    backup_path = default_settings_dir / f"coldState_backup_{timestamp}.nc"
                    shutil.copy2(coldstate_path, backup_path)
                
                # Copy optimized version
                optimized_coldstate = self.settings_dir / "coldState.nc"
                if optimized_coldstate.exists():
                    shutil.copy2(optimized_coldstate, coldstate_path)
                    self.logger.info("✅ Applied optimized soil depths")
            
            # Apply mizuRoute parameters if calibration enabled
            if self.config.get('CALIBRATE_MIZUROUTE', False):
                default_mizu_dir = self.project_dir / "settings" / "mizuRoute"
                mizu_param_file = default_mizu_dir / "param.nml.default"
                
                if mizu_param_file.exists():
                    backup_path = default_mizu_dir / f"param.nml.backup_{timestamp}"
                    shutil.copy2(mizu_param_file, backup_path)
                    
                    optimized_mizu = self.optimization_dir / "settings" / "mizuRoute" / "param.nml.default"
                    if optimized_mizu.exists():
                        shutil.copy2(optimized_mizu, mizu_param_file)
                        self.logger.info("✅ Applied optimized mizuRoute parameters")
            
        except Exception as e:
            self.logger.error(f"Error applying optimized parameters to default settings: {str(e)}")
    
    def _setup_model_settings(self) -> None:
        """Setup model configuration files for optimization"""
        # Copy SUMMA settings from project directory
        source_summa_dir = self.project_dir / "settings" / "SUMMA"
        if source_summa_dir.exists():
            for settings_file in source_summa_dir.glob("*"):
                if settings_file.is_file():
                    shutil.copy2(settings_file, self.settings_dir / settings_file.name)
        
        # Copy mizuRoute settings if they exist
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"
        
        if source_mizu_dir.exists():
            dest_mizu_dir.mkdir(parents=True, exist_ok=True)
            for mizu_file in source_mizu_dir.glob("*"):
                if mizu_file.is_file():
                    shutil.copy2(mizu_file, dest_mizu_dir / mizu_file.name)
        
        # Update configuration files for optimization
        self._update_file_manager_for_optimization()
        if dest_mizu_dir.exists():
            self._update_mizuroute_control_file(dest_mizu_dir)
    
    def _update_file_manager_for_optimization(self) -> None:
        """Update SUMMA file manager for optimization runs (calibration period)"""
        file_manager_path = self.settings_dir / 'fileManager.txt'
        if not file_manager_path.exists():
            self.logger.warning(f"File manager not found: {file_manager_path}")
            return
        
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        # Determine simulation period (include spinup if specified)
        spinup_period = self.config.get('SPINUP_PERIOD', '')
        calibration_period = self.config.get('CALIBRATION_PERIOD', '')
        
        self.logger.info(f"Setting up optimization time periods:")
        self.logger.info(f"  Spinup period: {spinup_period}")
        self.logger.info(f"  Calibration period: {calibration_period}")
        
        if spinup_period and calibration_period:
            try:
                spinup_dates = [d.strip() for d in spinup_period.split(',')]
                cal_dates = [d.strip() for d in calibration_period.split(',')]
                
                if len(spinup_dates) >= 2 and len(cal_dates) >= 2:
                    # Include spinup for model initialization but evaluate only calibration
                    sim_start = f"{spinup_dates[0]} 01:00"
                    sim_end = f"{cal_dates[1]} 23:00"
                    self.logger.info(f"  Simulation will run from {sim_start} to {sim_end}")
                    self.logger.info(f"  But evaluation will only use calibration period: {cal_dates[0]} to {cal_dates[1]}")
                else:
                    raise ValueError("Invalid period format")
            except Exception as e:
                self.logger.warning(f"Could not parse time periods: {e}")
                # Fallback to full experiment period
                sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
                self.logger.info(f"  Using fallback period: {sim_start} to {sim_end}")
        else:
            sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
            sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
            self.logger.info(f"  Using experiment period: {sim_start} to {sim_end}")
        
        # Update file manager lines
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        'dds_clean_opt_{self.experiment_id}'\n")
            elif 'outputPath' in line:
                output_path = str(self.summa_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
        
        self.logger.info(f"Updated file manager for optimization: {file_manager_path}")
    
    def _update_file_manager_for_full_period(self) -> None:
        """Update file manager for final evaluation (full experiment period)"""
        file_manager_path = self.settings_dir / 'fileManager.txt'
        if not file_manager_path.exists():
            return
        
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        'dds_clean_final_{self.experiment_id}'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _update_mizuroute_control_file(self, mizu_settings_dir: Path) -> None:
        """Update mizuRoute control file for optimization"""
        control_path = mizu_settings_dir / "mizuroute.control"
        if not control_path.exists():
            return
        
        with open(control_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if '<input_dir>' in line:
                input_path = str(self.summa_dir).replace('\\', '/')
                updated_lines.append(f"<input_dir>             {input_path}/\n")
            elif '<output_dir>' in line:
                output_path = str(self.mizuroute_dir).replace('\\', '/')
                updated_lines.append(f"<output_dir>            {output_path}/\n")
            elif '<case_name>' in line:
                updated_lines.append(f"<case_name>             dds_clean_opt_{self.experiment_id}\n")
            elif '<fname_qsim>' in line:
                updated_lines.append(f"<fname_qsim>            dds_clean_opt_{self.experiment_id}_timestep.nc\n")
            else:
                updated_lines.append(line)
        
        with open(control_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _create_calibration_target(self) -> 'CalibrationTarget':
        """Factory method to create appropriate calibration target"""
        calibration_var = self.config.get('CALIBRATION_VARIABLE', 'streamflow')
        
        if calibration_var == 'streamflow':
            return StreamflowTarget(self.config, self.project_dir, self.logger)
        elif calibration_var == 'snow':
            return SnowTarget(self.config, self.project_dir, self.logger)
        else:
            raise ValueError(f"Unsupported calibration variable: {calibration_var}")


# Add the run method for consistency with optimization_manager.py
DDSOptimizer.run_dds_optimization = lambda self: self.run_optimization()


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run optimization
    optimizer = DDSOptimizer(config, logger)
    results = optimizer.run_optimization()
    
    print(f"\n🎉 DDS optimization completed!")
    print(f"Best {results['optimization_metric']}: {results['best_score']:.6f}")
    print(f"Duration: {results['duration']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Results: {results['output_dir']}")