#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Differentiable Parameter Emulation (DPE) for SUMMA – Enhanced Config-driven Multipath Optimizer

This module provides multiple gradient-based optimization strategies for CONFLUENCE hydrological model calibration:

1. EMULATOR: Pure neural network surrogate with PyTorch backpropagation
2. FD: Direct finite-difference optimization on SUMMA
3. SUMMA_AUTODIFF: End-to-end autodiff through SUMMA (requires Sundials sensitivities)
4. SUMMA_AUTODIFF_FD: Autodiff wrapper with finite-difference Jacobian approximation

Key Features:
- Automatic training data caching and reuse
- Comprehensive performance monitoring and timing
- Validation convergence tracking
- Progress reporting with detailed logging
- Robust error handling and fallback mechanisms
- Integration with CONFLUENCE's existing optimization infrastructure

Configuration Options (config.yaml):
  EMULATOR_SETTING: "EMULATOR" | "FD" | "SUMMA_AUTODIFF" | "SUMMA_AUTODIFF_FD"
  
  # Neural Network Configuration
  DPE_HIDDEN_DIMS: [256, 128, 64]
  DPE_TRAINING_SAMPLES: 500
  DPE_VALIDATION_SAMPLES: 100
  DPE_EPOCHS: 300
  DPE_LEARNING_RATE: 1e-3
  DPE_OPTIMIZATION_LR: 1e-2
  DPE_OPTIMIZATION_STEPS: 200
  
  # Autodiff Configuration
  DPE_USE_NN_HEAD: true
  DPE_USE_SUNDIALS: true
  DPE_AUTODIFF_STEPS: 200
  DPE_AUTODIFF_LR: 1e-2
  
  # Finite Difference Configuration
  DPE_FD_STEP: 1e-3
  DPE_GD_STEP_SIZE: 1e-1
  
  # Data Management
  DPE_TRAINING_CACHE: "default"  # Uses domain_dir/emulation/training_data/
  DPE_FORCE_RETRAIN: false       # Force regeneration of training data

Author: CONFLUENCE Development Team
License: MIT
Version: 2.0.0
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

# Reuse iterative optimizer backend for SUMMA IO/parallelism
from iterative_optimizer import DEOptimizer  # type: ignore


# ==============================
# Configuration Classes
# ==============================

@dataclass
class EmulatorConfig:
    """
    Configuration class for neural network emulator training and optimization.
    
    Attributes
    ----------
    hidden_dims : List[int]
        Architecture of hidden layers in the neural network
    dropout : float
        Dropout rate for regularization
    activation : str
        Activation function ('relu', 'swish')
    n_training_samples : int
        Number of samples for training the emulator
    n_validation_samples : int
        Number of samples for validation
    batch_size : int
        Training batch size
    learning_rate : float
        Learning rate for emulator training
    n_epochs : int
        Maximum number of training epochs
    early_stopping_patience : int
        Early stopping patience (epochs without improvement)
    optimization_lr : float
        Learning rate for parameter optimization
    optimization_steps : int
        Number of optimization steps
    objective_weights : Dict[str, float]
        Weights for combining multiple objectives
    """
    hidden_dims: List[int] = None
    dropout: float = 0.1
    activation: str = 'relu'
    n_training_samples: int = 1000
    pretrain_nn_head: bool = False
    n_validation_samples: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 500
    early_stopping_patience: int = 50
    optimization_lr: float = 1e-2
    optimization_steps: int = 100
    objective_weights: Dict[str, float] = None
    iterate_enabled: bool = False
    iterate_max_iterations: int = 5
    iterate_samples_per_cycle: int = 100
    iterate_sampling_radius: float = 0.1
    iterate_convergence_tol: float = 1e-4
    iterate_min_improvement: float = 1e-6
    iterate_sampling_method: str = 'gaussian'

    def __post_init__(self):
        """
        Ensures correct types for numeric fields after initialization.
        This prevents TypeErrors when values are loaded from YAML as strings.
        """
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64, 32]
        if self.objective_weights is None:
            # Maximize KGE/NSE: use negative weights to minimize their negatives
            self.objective_weights = {'KGE': -1.0, 'NSE': -1.0, 'RMSE': 0.1, 'PBIAS': 0.05}
        
        # --- FIX: Ensure all numeric parameters are cast to the correct type ---
        self.dropout = float(self.dropout)
        self.n_training_samples = int(self.n_training_samples)
        self.n_validation_samples = int(self.n_validation_samples)
        self.batch_size = int(self.batch_size)
        self.learning_rate = float(self.learning_rate)
        self.n_epochs = int(self.n_epochs)
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.optimization_lr = float(self.optimization_lr)
        self.optimization_steps = int(self.optimization_steps)
        self.iterate_max_iterations = int(self.iterate_max_iterations)
        self.iterate_samples_per_cycle = int(self.iterate_samples_per_cycle)
        self.iterate_sampling_radius = float(self.iterate_sampling_radius)
        self.iterate_convergence_tol = float(self.iterate_convergence_tol)
        self.iterate_min_improvement = float(self.iterate_min_improvement)

# ==============================
# Neural Network Architecture
# ==============================

class ParameterEmulator(nn.Module):
    """
    Neural network emulator for SUMMA parameter-to-objective mapping.
    
    This network learns to approximate the complex relationship between
    hydrological parameters and model performance objectives, enabling
    fast gradient-based optimization.
    
    Parameters
    ----------
    n_parameters : int
        Number of input parameters
    n_objectives : int
        Number of output objectives
    config : EmulatorConfig
        Configuration for network architecture and training
    """
    
    def __init__(self, n_parameters: int, n_objectives: int, config: EmulatorConfig):
        super().__init__()
        
        # Build network architecture
        layers = []
        input_dim = n_parameters
        
        for hidden in config.hidden_dims:
            layers += [
                nn.Linear(input_dim, hidden),
                nn.SiLU() if config.activation == 'swish' else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.BatchNorm1d(hidden)
            ]
            input_dim = hidden
        
        # Output layer (no activation - raw objective values)
        layers += [nn.Linear(input_dim, n_objectives)]
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialize network weights using Xavier normal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


# ==============================
# SUMMA Interface Integration
# ==============================

class SummaInterface:
    """
    Interface to SUMMA hydrological model via CONFLUENCE's iterative optimizer backend.
    
    This class provides a clean interface for parameter manipulation, model execution,
    and result extraction while leveraging CONFLUENCE's existing parallel processing
    and parameter management infrastructure.
    
    Parameters
    ----------
    confluence_config : Dict
        CONFLUENCE configuration dictionary
    logger : logging.Logger
        Logger instance for tracking operations
    """
    
    def __init__(self, confluence_config: Dict, logger: logging.Logger):
        self.config = confluence_config  # Add this line
        self.logger = logger
        
        # Initialize backend component
        self.backend = DEOptimizer(confluence_config, self.logger)
        self.param_manager = self.backend.parameter_manager
        self.model_executor = self.backend.model_executor
        self.calib_target = self.backend.calibration_target
        self.param_names = self.param_manager.all_param_names
        
        self.logger.info(f"Initialized SUMMA interface with {len(self.param_names)} parameters")

    def normalize_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to normalized array [0,1]."""
        return self.param_manager.normalize_parameters(params)

    def denormalize_parameters(self, x_norm: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized array to parameter dictionary."""
        return self.param_manager.denormalize_parameters(x_norm)

    def run_simulations_batch(self, param_samples: List[np.ndarray], objective_names) -> List[Optional[Dict]]:
        """
        Run batch simulations for multiple parameter sets with proper multi-objective setup.
        
        Parameters
        ----------
        param_samples : List[np.ndarray]
            List of normalized parameter arrays
            
        Returns
        -------
        List[Optional[Dict]]
            List of metric dictionaries (None for failed simulations)
        """
        tasks = []
        for i, x in enumerate(param_samples):
            params = self.denormalize_parameters(x)
            
            # Get parallel directory info (similar to NSGA-II)
            proc_dirs = self.backend.parallel_dirs[i % len(self.backend.parallel_dirs)]
            
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.backend.num_processes,
                'evaluation_id': f"dpe_batch_{i:04d}",
                'multiobjective': True,  # CRITICAL: Enable multi-objective mode
                'objective_names': objective_names, 
                
                # Required fields for worker (copied from NSGA-II implementation)
                'target_metric': self.backend.target_metric,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'config': self.config,
                'domain_name': self.backend.domain_name,
                'project_dir': str(self.backend.project_dir),
                'original_depths': self.backend.parameter_manager.original_depths.tolist() if self.backend.parameter_manager.original_depths is not None else None,
                
                # Paths for worker
                'summa_exe': str(self.backend._get_summa_exe_path()),
                'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
                'summa_dir': str(proc_dirs['summa_dir']),
                'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
                'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir'])
            }
            tasks.append(task)
        
        self.logger.info(f"Running batch simulation for {len(tasks)} parameter sets (multi-objective mode)")
        results = self.backend._run_parallel_evaluations(tasks)
        
        # Convert results to metrics format
        output = [None] * len(param_samples)
        for result in results:
            if result and 'individual_id' in result:
                idx = result['individual_id']
                if idx < len(output):
                    # Extract metrics properly - prioritize full metrics
                    if 'metrics' in result and result['metrics']:
                        output[idx] = result['metrics']
                    elif 'objectives' in result and result['objectives'] is not None:
                        # Convert objectives array to metrics dictionary
                        objectives_array = result['objectives']
                        # Map to the actual objective names we're tracking
                        if len(objectives_array) >= len(objective_names):
                            metrics = {}
                            for i, obj_name in enumerate(objective_names):
                                if i < len(objectives_array):
                                    metrics[obj_name] = float(objectives_array[i])
                                    metrics[f'Calib_{obj_name}'] = float(objectives_array[i])
                            output[idx] = metrics
                        else:
                            self.logger.warning(f"Objectives array length ({len(objectives_array)}) < expected ({len(objective_names)})")
                    elif 'score' in result and result['score'] is not None:
                        # Fallback: create metrics from single score
                        score = float(result['score'])
                        # Map to all objective names with the same value (not ideal but better than nothing)
                        metrics = {}
                        for obj_name in objective_names:
                            metrics[obj_name] = score
                            metrics[f'Calib_{obj_name}'] = score
                        output[idx] = metrics
                        self.logger.warning(f"Only single score available, mapping to all objectives: {score}")
        
        success_rate = sum(1 for r in output if r is not None) / len(output)
        self.logger.info(f"Batch simulation completed. Success rate: {success_rate:.2%}")
        
        return output
        
# ==============================
# PyTorch Autograd Integration
# ==============================

class SummaAutogradOp(torch.autograd.Function):
    """
    Custom PyTorch autograd function to integrate SUMMA into computational graphs.
    
    This function enables end-to-end gradient computation through SUMMA,
    supporting both analytical sensitivities (via Sundials) and numerical
    finite-difference approximations.
    """
    
    @staticmethod
    def forward(ctx, x_norm: torch.Tensor, dpe_self, use_sundials: bool, fd_step: float):
        """
        Forward pass: evaluate SUMMA objectives.
        
        Parameters
        ----------
        ctx : torch.autograd.Function.ctx
            Context for saving information for backward pass
        x_norm : torch.Tensor
            Normalized parameters [0,1]
        dpe_self : DifferentiableParameterOptimizer
            Reference to optimizer instance
        use_sundials : bool
            Whether to use Sundials sensitivities for gradients
        fd_step : float
            Step size for finite difference approximation
            
        Returns
        -------
        torch.Tensor
            Objective values
        """
        # Convert to numpy and evaluate
        x_np = x_norm.detach().cpu().numpy().astype(float)
        objectives_dict = dpe_self._evaluate_objectives_real(x_np)
        
        # Convert to tensor
        obj_values = []
        for obj_name in dpe_self.objective_names:
            value = objectives_dict.get(obj_name, objectives_dict.get(f"Calib_{obj_name}", 0.0))
            obj_values.append(value)
        
        obj_tensor = torch.tensor(obj_values, dtype=torch.float32, device=x_norm.device)
        
        # Prepare for backward pass
        if use_sundials:
            try:
                jacobian_np = dpe_self._summa_objective_jacobian(x_np)
                jacobian = torch.tensor(jacobian_np, dtype=torch.float32, device=x_norm.device)
                ctx.save_for_backward(jacobian)
                ctx.jacobian_ready = True
                ctx.use_sundials = True
            except NotImplementedError:
                # Fall back to finite differences
                ctx.save_for_backward(x_norm.detach().clone())
                ctx.jacobian_ready = False
                ctx.use_sundials = False
                ctx.fd_step = float(fd_step)
                ctx.dpe_ref = dpe_self
        else:
            ctx.save_for_backward(x_norm.detach().clone())
            ctx.jacobian_ready = False
            ctx.use_sundials = False
            ctx.fd_step = float(fd_step)
            ctx.dpe_ref = dpe_self
        
        return obj_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute parameter gradients.
        
        Parameters
        ----------
        ctx : torch.autograd.Function.ctx
            Context from forward pass
        grad_output : torch.Tensor
            Gradients from downstream operations
            
        Returns
        -------
        Tuple[torch.Tensor, None, None, None]
            Gradients w.r.t. input parameters
        """
        if getattr(ctx, 'jacobian_ready', False):
            # Use precomputed Jacobian (Sundials)
            (jacobian,) = ctx.saved_tensors  # [n_objectives, n_parameters]
        else:
            # Compute finite difference Jacobian
            (x_saved,) = ctx.saved_tensors
            dpe = ctx.dpe_ref
            jacobian_np = dpe._finite_difference_jacobian(
                x_saved.detach().cpu().numpy(), 
                step=ctx.fd_step
            )
            jacobian = torch.tensor(jacobian_np, dtype=torch.float32, device=grad_output.device)
        
        # Chain rule: grad_x = grad_output^T @ jacobian
        grad_x = grad_output.unsqueeze(0).matmul(jacobian).squeeze(0)
        
        return grad_x, None, None, None


class ObjectiveHead(nn.Module):
    """
    Small neural network head for post-processing SUMMA objectives.
    
    This optional component can learn residual corrections to SUMMA
    objectives, potentially improving optimization performance.
    
    Parameters
    ----------
    n_params : int
        Number of input parameters
    n_objs : int
        Number of objectives
    hidden : int
        Hidden layer size
    """
    
    def __init__(self, n_params: int, n_objs: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_objs + n_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_objs)
        )
    
    def forward(self, objectives: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: add learned residuals to objectives.
        
        Parameters
        ----------
        objectives : torch.Tensor
            Raw SUMMA objectives
        x_norm : torch.Tensor
            Normalized parameters
            
        Returns
        -------
        torch.Tensor
            Adjusted objectives
        """
        combined_input = torch.cat([objectives, x_norm], dim=-1)
        residuals = self.net(combined_input)
        return objectives + residuals


# ==============================
# Main Optimizer Class
# ==============================

class DifferentiableParameterOptimizer:
    """
    Differentiable Parameter Optimizer for CONFLUENCE.
    
    This class provides multiple gradient-based optimization strategies for
    hydrological model calibration, including pure emulation, finite differences,
    and end-to-end autodiff through SUMMA.
    
    The optimizer automatically handles training data generation, caching,
    neural network training, and parameter optimization with comprehensive
    logging and performance monitoring.
    
    Parameters
    ----------
    confluence_config : Dict
        CONFLUENCE configuration dictionary
    domain_name : str
        Name of the modeling domain
    emulator_config : EmulatorConfig, optional
        Configuration for neural network training and optimization
        
    Examples
    --------
    >>> config = yaml.safe_load(open('config.yaml'))
    >>> dpe = DifferentiableParameterOptimizer(config, "bow_at_banff")
    >>> optimized_params = dpe.run_from_config()
    """
    
    def __init__(self, confluence_config: Dict, domain_name: str, emulator_config: EmulatorConfig = None):
        self.confluence_config = confluence_config
        self.config = confluence_config
        self.domain_name = domain_name
        self.emulator_config = emulator_config or EmulatorConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"{__name__}.DPE")
        self.backend = DEOptimizer(confluence_config, self.logger)
        
        # Initialize SUMMA interface
        self.summa = SummaInterface(confluence_config, self.logger)
        self.param_names = self.summa.param_names
        self.objective_names = sorted(self.emulator_config.objective_weights.keys())
        self.n_parameters = len(self.param_names)
        self.n_objectives = len(self.objective_names)
        
        # Initialize neural network
        self.emulator = ParameterEmulator(self.n_parameters, self.n_objectives, self.emulator_config)
        
        # Data storage
        self.training_data = {'parameters': [], 'objectives': []}
        self.validation_data = {'parameters': [], 'objectives': []}
        
        # Setup cache directory
        self._setup_cache_directory()
        
        # Performance tracking
        self.timing_results = {}
        
        self.logger.info(f"Initialized DPE for domain '{domain_name}' with {self.n_parameters} parameters")
        self.logger.info(f"Target objectives: {self.objective_names}")

    def _setup_cache_directory(self):
        """Setup default cache directory structure."""
        data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        domain_dir = data_dir / f"domain_{self.domain_name}"
        self.cache_dir = domain_dir / "emulation" / "training_data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Cache directory: {self.cache_dir}")

    # ==============================
    # Utility Methods
    # ==============================

    def _generate_samples_around_optimum(self, optimum_params: np.ndarray, 
                                    n_samples: int, 
                                    radius: float,
                                    method: str = 'gaussian') -> np.ndarray:
        """
        Generate parameter samples around current optimum for iterative refinement.
        
        Parameters
        ----------
        optimum_params : np.ndarray
            Current optimum in normalized space [0,1]
        n_samples : int
            Number of samples to generate
        radius : float
            Sampling radius in normalized space
        method : str
            Sampling method ('gaussian', 'uniform', 'adaptive')
            
        Returns
        -------
        np.ndarray
            Array of parameter samples [n_samples, n_parameters]
        """
        samples = []
        
        for _ in range(n_samples):
            if method == 'gaussian':
                # Gaussian sampling around optimum
                perturbation = np.random.normal(0, radius, self.n_parameters)
                sample = optimum_params + perturbation
                
            elif method == 'uniform':
                # Uniform sampling in hypercube around optimum
                perturbation = np.random.uniform(-radius, radius, self.n_parameters)
                sample = optimum_params + perturbation
                
            elif method == 'adaptive':
                # Adaptive sampling with parameter-specific radii
                # Use smaller radius for parameters that have converged
                param_sensitivity = np.ones(self.n_parameters) * radius
                perturbation = np.random.normal(0, param_sensitivity)
                sample = optimum_params + perturbation
                
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
            # Clip to valid parameter space [0,1]
            sample = np.clip(sample, 0.0, 1.0)
            samples.append(sample)
        
        return np.array(samples)

    def _check_iterative_convergence(self, 
                                    old_optimum: np.ndarray, 
                                    new_optimum: np.ndarray,
                                    old_loss: float,
                                    new_loss: float) -> Tuple[bool, str]:
        """
        Check convergence criteria for iterative optimization.
        
        Parameters
        ----------
        old_optimum : np.ndarray
            Previous optimum parameters
        new_optimum : np.ndarray  
            Current optimum parameters
        old_loss : float
            Previous best loss
        new_loss : float
            Current best loss
            
        Returns
        -------
        Tuple[bool, str]
            (converged, reason)
        """
        # Parameter change convergence
        param_change = np.linalg.norm(new_optimum - old_optimum)
        if param_change < self.emulator_config.iterate_convergence_tol:
            return True, f"Parameter convergence (change: {param_change:.6f})"
        
        # Objective improvement convergence
        improvement = old_loss - new_loss  # Positive = improvement
        if abs(improvement) < self.emulator_config.iterate_min_improvement:
            return True, f"Objective convergence (improvement: {improvement:.6f})"
        
        return False, f"Not converged (param_change: {param_change:.6f}, improvement: {improvement:.6f})"

    def _update_emulator_config_for_iteration(self, iteration: int):
        """Update emulator config for iterative training (e.g., reduce epochs)."""
        if iteration > 1:
            # Reduce training epochs for subsequent iterations since we're refining
            self.emulator_config.n_epochs = min(self.emulator_config.n_epochs, 150)
            self.emulator_config.early_stopping_patience = min(self.emulator_config.early_stopping_patience, 25)
            self.logger.info(f"Iteration {iteration}: Reduced training epochs to {self.emulator_config.n_epochs}")

    def optimize_parameters_iterative(self, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Iterative emulator optimization with active learning around optima.
        
        This method implements an iterative refinement approach:
        1. Train emulator on initial data
        2. Optimize to find current best parameters
        3. Generate new samples around the optimum
        4. Add new data and retrain emulator
        5. Repeat until convergence
        
        Parameters
        ----------
        initial_params : Dict[str, float], optional
            Initial parameter values
            
        Returns
        -------
        Dict[str, float]
            Final optimized parameters
        """
        if not self.emulator_config.iterate_enabled:
            return self.optimize_parameters(initial_params)
        
        self.logger.info("Starting iterative emulator optimization...")
        self.logger.info(f"Max iterations: {self.emulator_config.iterate_max_iterations}")
        self.logger.info(f"Samples per cycle: {self.emulator_config.iterate_samples_per_cycle}")
        self.logger.info(f"Sampling radius: {self.emulator_config.iterate_sampling_radius}")
        
        start_time = time.time()
        
        # Initialize with first optimization
        current_optimum_params = self.optimize_parameters(initial_params)
        current_optimum_norm = self.summa.normalize_parameters(current_optimum_params)
        
        # Evaluate current optimum to get baseline loss
        current_objectives = self._evaluate_objectives_real(current_optimum_norm)
        current_loss = self._composite_loss_from_objectives(current_objectives)
        
        self.logger.info(f"Initial optimization complete. Loss: {current_loss:.6f}")
        
        # Store iteration history
        iteration_history = [{
            'iteration': 0,
            'loss': current_loss,
            'parameters': current_optimum_params.copy(),
            'training_samples': len(self.training_data['parameters']),
            'validation_samples': len(self.validation_data['parameters'])
        }]
        
        # Iterative refinement loop
        for iteration in range(1, self.emulator_config.iterate_max_iterations + 1):
            iteration_start = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ITERATIVE REFINEMENT - CYCLE {iteration}/{self.emulator_config.iterate_max_iterations}")
            self.logger.info(f"{'='*60}")
            
            # 1. Generate new samples around current optimum
            self.logger.info(f"Generating {self.emulator_config.iterate_samples_per_cycle} samples around optimum...")
            new_param_samples = self._generate_samples_around_optimum(
                current_optimum_norm,
                self.emulator_config.iterate_samples_per_cycle,
                self.emulator_config.iterate_sampling_radius,
                self.emulator_config.iterate_sampling_method
            )
            
            # 2. Evaluate new samples with SUMMA
            self.logger.info("Evaluating new samples with SUMMA...")
            new_objective_results = self.summa.run_simulations_batch(new_param_samples, self.objective_names)
            
            # 3. Filter valid results and add to training data
            valid_new_params, valid_new_objectives = [], []
            for params, objectives in zip(new_param_samples, new_objective_results):
                if objectives and self._is_valid_objective(objectives):
                    valid_new_params.append(params.tolist())
                    
                    # Extract objective values in consistent order
                    obj_values = []
                    for obj_name in self.objective_names:
                        value = objectives.get(obj_name, objectives.get(f"Calib_{obj_name}", 0.0))
                        obj_values.append(value)
                    valid_new_objectives.append(obj_values)
            
            success_rate = len(valid_new_params) / len(new_param_samples)
            self.logger.info(f"New sample success rate: {success_rate:.2%} ({len(valid_new_params)}/{len(new_param_samples)})")
            
            if len(valid_new_params) == 0:
                self.logger.warning("No valid new samples obtained. Stopping iterations.")
                break
            
            # 4. Add new data to existing training data
            self.training_data['parameters'].extend(valid_new_params)
            self.training_data['objectives'].extend(valid_new_objectives)
            
            # Also add some to validation set (20% of new samples)
            n_val_add = max(1, len(valid_new_params) // 5)
            self.validation_data['parameters'].extend(valid_new_params[:n_val_add])
            self.validation_data['objectives'].extend(valid_new_objectives[:n_val_add])
            
            total_training = len(self.training_data['parameters'])
            total_validation = len(self.validation_data['parameters'])
            self.logger.info(f"Updated training data: {total_training} samples")
            self.logger.info(f"Updated validation data: {total_validation} samples")
            
            # 5. Update emulator config for iteration
            self._update_emulator_config_for_iteration(iteration)
            
            # 6. Retrain emulator on expanded dataset
            self.logger.info("Retraining emulator on expanded dataset...")
            self.train_emulator()
            
            # 7. Re-optimize parameters
            self.logger.info("Re-optimizing parameters with updated emulator...")
            new_optimum_params = self.optimize_parameters(initial_params)
            new_optimum_norm = self.summa.normalize_parameters(new_optimum_params)
            
            # 8. Evaluate new optimum
            new_objectives = self._evaluate_objectives_real(new_optimum_norm)
            new_loss = self._composite_loss_from_objectives(new_objectives)
            
            iteration_time = time.time() - iteration_start
            
            # 9. Check convergence
            converged, convergence_reason = self._check_iterative_convergence(
                current_optimum_norm, new_optimum_norm, current_loss, new_loss
            )
            
            # Log iteration results
            improvement = current_loss - new_loss
            self.logger.info(f"Iteration {iteration} completed in {iteration_time:.1f}s")
            self.logger.info(f"Previous loss: {current_loss:.6f}")
            self.logger.info(f"New loss: {new_loss:.6f}")
            self.logger.info(f"Improvement: {improvement:.6f}")
            self.logger.info(f"Convergence check: {convergence_reason}")
            
            # Store iteration data
            iteration_history.append({
                'iteration': iteration,
                'loss': new_loss,
                'improvement': improvement,
                'parameters': new_optimum_params.copy(),
                'training_samples': total_training,
                'validation_samples': total_validation,
                'convergence_check': convergence_reason,
                'iteration_time': iteration_time
            })
            
            # Update current optimum
            current_optimum_params = new_optimum_params
            current_optimum_norm = new_optimum_norm
            current_loss = new_loss
            
            # 10. Check for convergence
            if converged:
                self.logger.info(f"Converged after {iteration} iterations: {convergence_reason}")
                break
            
            # Adaptive radius adjustment
            if improvement > 0:
                # Good improvement - keep radius
                self.logger.info("Improvement found - maintaining sampling radius")
            else:
                # No improvement - reduce radius for next iteration
                self.emulator_config.iterate_sampling_radius *= 0.8
                self.logger.info(f"No improvement - reducing sampling radius to {self.emulator_config.iterate_sampling_radius:.4f}")
        
        total_time = time.time() - start_time
        self.timing_results['iterative_emulator_optimization'] = total_time
        
        # Save iteration history
        history_path = self.cache_dir / f"iterative_history_{self.domain_name}.json"
        with open(history_path, 'w') as f:
            json.dump(iteration_history, f, indent=2, default=str)
        
        self.logger.info(f"\nIterative emulator optimization completed in {total_time:.1f}s")
        self.logger.info(f"Final loss: {current_loss:.6f}")
        self.logger.info(f"Total iterations: {len(iteration_history) - 1}")
        self.logger.info(f"Final training samples: {len(self.training_data['parameters'])}")
        
        return current_optimum_params



    def _latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """Generate Latin Hypercube samples for parameter space exploration."""
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.n_parameters, seed=42)
        return sampler.random(n=n_samples)

    def _is_valid_objective(self, objectives: Dict[str, float]) -> bool:
        """Check if objective dictionary contains valid (non-NaN, finite) values."""
        for obj_name in self.objective_names:
            value = objectives.get(obj_name, objectives.get(f"Calib_{obj_name}"))
            if value is None or np.isnan(value) or np.isinf(value):
                return False
        return True

    def _composite_loss_from_objectives(self, objectives: Dict[str, float]) -> float:
        """Compute weighted composite loss from objective dictionary."""
        total_loss = 0.0
        for obj_name in self.objective_names:
            value = objectives.get(obj_name, objectives.get(f"Calib_{obj_name}"))
            if value is None or np.isnan(value) or np.isinf(value):
                return 1e6  # Large penalty for invalid objectives
            
            weight = self.emulator_config.objective_weights.get(obj_name, 0.0)
            total_loss += weight * float(value)
        
        return float(total_loss)

    def _evaluate_objectives_real(self, x_norm: np.ndarray) -> Dict[str, float]:
        """Evaluate objectives using real SUMMA simulation with proper multi-objective setup."""
        params = self.summa.denormalize_parameters(x_norm)
        
        # Get parallel directory info for single evaluation
        proc_dirs = self.backend.parallel_dirs[0]  # Use first available directory
        
        task = {
            'individual_id': 0,
            'params': params,
            'proc_id': 0,
            'evaluation_id': 'dpe_eval_single',
            'multiobjective': True,  # CRITICAL: Enable multi-objective mode
            'objective_names': self.objective_names, 
            
            # Required fields for worker (matching NSGA-II implementation)
            'target_metric': self.backend.target_metric,
            'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
            'config': self.config,
            'domain_name': self.backend.domain_name,
            'project_dir': str(self.backend.project_dir),
            'original_depths': self.backend.parameter_manager.original_depths.tolist() if self.backend.parameter_manager.original_depths is not None else None,
            
            # Paths for worker
            'summa_exe': str(self.backend._get_summa_exe_path()),
            'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
            'summa_dir': str(proc_dirs['summa_dir']),
            'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
            'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
            'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir'])
        }
        
        results = self.summa.backend._run_parallel_evaluations([task])

      
        if not results or not results[0]:
            self.logger.warning("No results returned from SUMMA evaluation")
            return {}
        
        result = results[0]
        #self.logger.info(f"Raw SUMMA objectives: {result}")

        # Handle both metrics format and objectives format
        if 'metrics' in result and result['metrics']:
            return result['metrics']
        elif 'objectives' in result and result['objectives'] is not None:
            # Convert objectives array back to metrics dictionary
            objectives_array = result['objectives']
            if len(objectives_array) >= len(self.objective_names):
                metrics = {}
                for i, obj_name in enumerate(self.objective_names):
                    if i < len(objectives_array):
                        metrics[obj_name] = float(objectives_array[i])
                        metrics[f'Calib_{obj_name}'] = float(objectives_array[i])
                return metrics
            else:
                self.logger.warning(f"Objectives array length ({len(objectives_array)}) < expected ({len(self.objective_names)})")
        
        # Fallback: if we only have a score, try to map it to primary objective
        score = result.get('score')
        if score is not None:
            self.logger.warning(f"Only score available, mapping to primary objective: {score}")
            # Map to all objectives (not ideal but maintains consistency)
            metrics = {}
            for obj_name in self.objective_names:
                metrics[obj_name] = float(score)
                metrics[f'Calib_{obj_name}'] = float(score)
            return metrics
        
        self.logger.error(f"No valid objectives found in result: {result.keys()}")
        return {}

    # ==============================
    # Data Generation and Caching
    # ==============================

    def generate_training_data(self, cache_path: Optional[Path] = None, force: bool = False):
        """
        Generate training and validation data for emulator.
        
        Parameters
        ----------
        cache_path : Path, optional
            Path to cache file. If None, uses default cache directory.
        force : bool
            Force regeneration even if cached data exists
        """
        # Determine cache path
        if cache_path is None:
            cache_path = self.cache_dir / f"training_data_{self.domain_name}.json"
        
        # Check for existing cache
        if cache_path.exists() and not force:
            self.logger.info(f"Loading cached training data from {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                self.training_data = cached_data['training']
                self.validation_data = cached_data['validation']
                
                n_train = len(self.training_data['parameters'])
                n_val = len(self.validation_data['parameters'])
                self.logger.info(f"Loaded {n_train} training + {n_val} validation samples from cache")
                return
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Regenerating data.")
        
        # Generate new data
        self.logger.info("Generating training data...")
        start_time = time.time()
        
        total_samples = self.emulator_config.n_training_samples + self.emulator_config.n_validation_samples
        parameter_samples = self._latin_hypercube_sampling(total_samples)
        
        self.logger.info(f"Running {total_samples} SUMMA simulations...")
        objective_results = self.summa.run_simulations_batch(parameter_samples, self.objective_names)        
        
        # Filter valid results
        valid_parameters, valid_objectives = [], []
        for params, objectives in zip(parameter_samples, objective_results):
            if objectives and self._is_valid_objective(objectives):
                valid_parameters.append(params.tolist())
                
                # Extract objective values in consistent order
                obj_values = []
                for obj_name in self.objective_names:
                    value = objectives.get(obj_name, objectives.get(f"Calib_{obj_name}", 0.0))
                    obj_values.append(value)
                valid_objectives.append(obj_values)
        
        if len(valid_parameters) < self.emulator_config.n_training_samples:
            raise RuntimeError(
                f"Insufficient valid samples ({len(valid_parameters)}) for training "
                f"({self.emulator_config.n_training_samples} required)"
            )
        
        # Split into training and validation
        n_train = self.emulator_config.n_training_samples
        self.training_data = {
            'parameters': valid_parameters[:n_train],
            'objectives': valid_objectives[:n_train]
        }
        self.validation_data = {
            'parameters': valid_parameters[n_train:],
            'objectives': valid_objectives[n_train:]
        }
        
        # Save cache
        cache_data = {
            'training': self.training_data,
            'validation': self.validation_data,
            'generated_at': datetime.now().isoformat(),
            'config_snapshot': {
                'n_training_samples': self.emulator_config.n_training_samples,
                'n_validation_samples': self.emulator_config.n_validation_samples,
                'objective_names': self.objective_names,
                'parameter_names': self.param_names
            }
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        generation_time = time.time() - start_time
        success_rate = len(valid_parameters) / total_samples
        
        self.logger.info(f"Data generation completed in {generation_time:.1f}s")
        self.logger.info(f"Success rate: {success_rate:.2%}")
        self.logger.info(f"Training samples: {len(self.training_data['parameters'])}")
        self.logger.info(f"Validation samples: {len(self.validation_data['parameters'])}")
        self.logger.info(f"Cache saved to: {cache_path}")

    # ==============================
    # Neural Network Training
    # ==============================

    def train_emulator(self):
        """
        Train the neural network emulator on generated data.
        
        Includes progress tracking, early stopping, and comprehensive logging.
        """
        if not self.training_data['parameters']:
            raise ValueError("No training data available. Call generate_training_data() first.")
        
        self.logger.info("Starting emulator training...")
        start_time = time.time()
        
        # Prepare data
        X_train = torch.tensor(np.array(self.training_data['parameters']), dtype=torch.float32)
        y_train = torch.tensor(np.array(self.training_data['objectives']), dtype=torch.float32)
        X_val = torch.tensor(np.array(self.validation_data['parameters']), dtype=torch.float32)
        y_val = torch.tensor(np.array(self.validation_data['objectives']), dtype=torch.float32)
        
        self.logger.info(f"Training data shape: {X_train.shape} -> {y_train.shape}")
        self.logger.info(f"Validation data shape: {X_val.shape} -> {y_val.shape}")
        
        # Setup training
        optimizer = optim.Adam(self.emulator.parameters(), lr=self.emulator_config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        # Training loop with progress bar
        progress_bar = tqdm(range(self.emulator_config.n_epochs), desc="Training emulator")
        
        for epoch in progress_bar:
            # Training phase
            self.emulator.train()
            epoch_train_loss = 0.0
            n_batches = 0
            
            # Shuffle training data
            perm = torch.randperm(X_train.shape[0])
            
            for i in range(0, X_train.shape[0], self.emulator_config.batch_size):
                batch_indices = perm[i:i + self.emulator_config.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                optimizer.zero_grad()
                predictions = self.emulator(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_train_loss / n_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.emulator.eval()
            with torch.no_grad():
                val_predictions = self.emulator(X_val)
                val_loss = criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = self.cache_dir / f"best_emulator_{self.domain_name}.pt"
                torch.save(self.emulator.state_dict(), model_path)
            else:
                patience_counter += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Train Loss': f'{avg_train_loss:.6f}',
                'Val Loss': f'{val_loss:.6f}',
                'Best Val': f'{best_val_loss:.6f}',
                'Patience': f'{patience_counter}/{self.emulator_config.early_stopping_patience}'
            })
            
            # Detailed logging every 50 epochs
            if epoch % 50 == 0 or epoch == self.emulator_config.n_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch:4d}: Train={avg_train_loss:.6f}, "
                    f"Val={val_loss:.6f}, Best={best_val_loss:.6f}"
                )
            
            # Early stopping
            if patience_counter >= self.emulator_config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model_path = self.cache_dir / f"best_emulator_{self.domain_name}.pt"
        if model_path.exists():
            self.emulator.load_state_dict(torch.load(model_path))
            self.logger.info(f"Loaded best model from {model_path}")
        
        training_time = time.time() - start_time
        self.timing_results['emulator_training'] = training_time
        
        self.logger.info(f"Emulator training completed in {training_time:.1f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save training history
        history_path = self.cache_dir / f"training_history_{self.domain_name}.json"
        history_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'training_time': training_time
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

    # ==============================
    # Optimization Methods
    # ==============================

    def _compute_weighted_loss_tensor(self, objective_tensor: torch.Tensor) -> torch.Tensor:
        """Compute weighted composite loss from objective tensor."""
        loss = torch.tensor(0.0, dtype=torch.float32, device=objective_tensor.device)
        
        for i, obj_name in enumerate(self.objective_names):
            weight = self.emulator_config.objective_weights.get(obj_name, 0.0)
            loss = loss + weight * objective_tensor[i]
        
        return loss

    def optimize_parameters(self, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize parameters using pure neural network emulator.
        
        Parameters
        ----------
        initial_params : Dict[str, float], optional
            Initial parameter values. If None, starts from center of parameter space.
            
        Returns
        -------
        Dict[str, float]
            Optimized parameters
        """
        self.logger.info("Starting emulator-based optimization...")
        start_time = time.time()
        
        # Initialize parameters
        if initial_params:
            x0 = self.summa.normalize_parameters(initial_params)
        else:
            x0 = np.full(self.n_parameters, 0.5)
        
        x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([x], lr=self.emulator_config.optimization_lr)
        
        best_loss = float('inf')
        best_x = x.detach().clone()
        loss_history = []
        
        self.emulator.eval()
        
        progress_bar = tqdm(range(self.emulator_config.optimization_steps), desc="Optimizing")
        
        for step in progress_bar:
            optimizer.zero_grad()
            
            # Clamp parameters to valid range
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
            
            # Forward pass
            objectives = self.emulator(x.unsqueeze(0)).squeeze(0)
            loss = self._compute_weighted_loss_tensor(objectives)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track best
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_x = x.detach().clone()
            
            progress_bar.set_postfix({'Loss': f'{current_loss:.6f}', 'Best': f'{best_loss:.6f}'})
        
        optimization_time = time.time() - start_time
        self.timing_results['emulator_optimization'] = optimization_time
        
        # Convert back to parameter dictionary
        best_params = self.summa.denormalize_parameters(best_x.clamp(0.0, 1.0).numpy())
        
        self.logger.info(f"Emulator optimization completed in {optimization_time:.1f}s")
        self.logger.info(f"Best loss: {best_loss:.6f}")
        
        return best_params

    def _finite_difference_jacobian(self, x: np.ndarray, step: float = 1e-3) -> np.ndarray:
        """
        Compute finite difference Jacobian of objectives w.r.t. parameters.
        
        Parameters
        ----------
        x : np.ndarray
            Parameter values (normalized)
        step : float
            Step size for finite differences
            
        Returns
        -------
        np.ndarray
            Jacobian matrix [n_objectives, n_parameters]
        """
        n_params = self.n_parameters
        n_objs = self.n_objectives
        jacobian = np.zeros((n_objs, n_params), dtype=float)
        
        # Evaluate at base point
        base_objectives = self._evaluate_objectives_real(x)
        base_vector = np.array([
            base_objectives.get(obj_name, base_objectives.get(f"Calib_{obj_name}", 0.0))
            for obj_name in self.objective_names
        ])
        
        # Prepare perturbation points
        forward_points, backward_points, difference_types = [], [], []
        
        for i in range(n_params):
            x_forward, x_backward = x.copy(), x.copy()
            
            # Determine step type based on bounds
            if x[i] + step <= 1.0 and x[i] - step >= 0.0:
                # Central difference
                x_forward[i] += step
                x_backward[i] -= step
                forward_points.append(x_forward)
                backward_points.append(x_backward)
                difference_types.append('central')
            elif x[i] + step <= 1.0:
                # Forward difference
                x_forward[i] += step
                forward_points.append(x_forward)
                backward_points.append(None)
                difference_types.append('forward')
            elif x[i] - step >= 0.0:
                # Backward difference
                x_backward[i] -= step
                forward_points.append(None)
                backward_points.append(x_backward)
                difference_types.append('backward')
            else:
                # No valid step (shouldn't happen with reasonable step size)
                forward_points.append(None)
                backward_points.append(None)
                difference_types.append('zero')
        
        # Batch evaluate perturbation points
        batch_points = []
        point_mapping = []
        
        for i, (x_f, x_b) in enumerate(zip(forward_points, backward_points)):
            if x_f is not None:
                batch_points.append(x_f)
                point_mapping.append(('forward', i))
            if x_b is not None:
                batch_points.append(x_b)
                point_mapping.append(('backward', i))
        
        if batch_points:
            batch_results = self.summa.run_simulations_batch(batch_points, self.objective_names)
        else:
            batch_results = []
        
        # Convert results to objective vectors
        batch_vectors = []
        for result in batch_results:
            if result is None:
                batch_vectors.append(np.full(n_objs, 1e6))  # Large penalty for failed runs
            else:
                vector = np.array([
                    result.get(obj_name, result.get(f"Calib_{obj_name}", 0.0))
                    for obj_name in self.objective_names
                ])
                batch_vectors.append(vector)
        
        # Map results back to forward/backward arrays
        forward_vectors = [None] * n_params
        backward_vectors = [None] * n_params
        
        result_idx = 0
        for point_type, param_idx in point_mapping:
            if point_type == 'forward':
                forward_vectors[param_idx] = batch_vectors[result_idx]
            else:
                backward_vectors[param_idx] = batch_vectors[result_idx]
            result_idx += 1
        
        # Compute finite differences
        for i, diff_type in enumerate(difference_types):
            if diff_type == 'central':
                jacobian[:, i] = (forward_vectors[i] - backward_vectors[i]) / (2 * step)
            elif diff_type == 'forward':
                jacobian[:, i] = (forward_vectors[i] - base_vector) / step
            elif diff_type == 'backward':
                jacobian[:, i] = (base_vector - backward_vectors[i]) / step
            else:
                jacobian[:, i] = 0.0  # No gradient information available
        
        return jacobian

    def optimize_with_fd(self, 
                        initial_params: Optional[Dict[str, float]] = None,
                        step_size: float = 1e-1,
                        fd_step: float = 1e-3,
                        max_iters: int = 50,
                        tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Optimize parameters using finite difference gradients on real SUMMA.
        
        Parameters
        ----------
        initial_params : Dict[str, float], optional
            Initial parameter values
        step_size : float
            Gradient descent step size
        fd_step : float
            Finite difference step size
        max_iters : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns
        -------
        Dict[str, float]
            Optimized parameters
        """
        self.logger.info("Starting finite-difference optimization...")
        start_time = time.time()
        
        # Initialize
        if initial_params:
            x = self.summa.normalize_parameters(initial_params)
        else:
            x = np.full(self.n_parameters, 0.5)
        
        # Initial evaluation
        base_objectives = self._evaluate_objectives_real(x)
        current_loss = self._composite_loss_from_objectives(base_objectives)
        
        self.logger.info(f"Initial loss: {current_loss:.6f}")
        
        # Optimization loop
        for iteration in range(1, max_iters + 1):
            self.logger.info(f"FD iteration {iteration}/{max_iters}")
            
            # Compute gradient via finite differences
            jacobian = self._finite_difference_jacobian(x, step=fd_step)
            
            # Chain rule for scalar loss: dL/dx = weights^T @ jacobian
            weights = np.array([
                self.emulator_config.objective_weights.get(obj_name, 0.0)
                for obj_name in self.objective_names
            ])
            gradient = weights @ jacobian  # [n_parameters]
            
            # Gradient descent step
            x_new = np.clip(x - step_size * gradient, 0.0, 1.0)
            
            # Evaluate new point
            new_objectives = self._evaluate_objectives_real(x_new)
            new_loss = self._composite_loss_from_objectives(new_objectives)
            
            self.logger.info(f"  Loss: {new_loss:.6f} (prev: {current_loss:.6f})")
            
            # Accept/reject step
            if new_loss < current_loss:
                x = x_new
                current_loss = new_loss
                self.logger.info(f"  Step accepted")
            else:
                step_size *= 0.5  # Reduce step size
                self.logger.info(f"  Step rejected, reducing step size to {step_size:.6f}")
                
                if step_size < 1e-5:
                    self.logger.info("Step size too small, terminating")
                    break
            
            # Convergence check
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < tolerance:
                self.logger.info(f"Converged (gradient norm: {gradient_norm:.6f})")
                break
        
        optimization_time = time.time() - start_time
        self.timing_results['fd_optimization'] = optimization_time
        
        final_params = self.summa.denormalize_parameters(x)
        
        self.logger.info(f"FD optimization completed in {optimization_time:.1f}s")
        self.logger.info(f"Final loss: {current_loss:.6f}")
        
        return final_params

    def _summa_objective_jacobian(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Compute analytical Jacobian using SUMMA's Sundials sensitivities.
        
        NOTE: This method requires integration with SUMMA's Sundials-based
        sensitivity analysis. Implementation pending availability from SUMMA team.
        
        Parameters
        ----------
        x_norm : np.ndarray
            Normalized parameters [0,1]
            
        Returns
        -------
        np.ndarray
            Jacobian matrix [n_objectives, n_parameters]
            
        Raises
        ------
        NotImplementedError
            Until Sundials integration is available
        """
        raise NotImplementedError(
            "Sundials sensitivity integration pending. "
            "Contact SUMMA development team for implementation status."
        )

    def _create_pretrained_head(self) -> ObjectiveHead:
        """Create ObjectiveHead initialized with emulator knowledge."""
        self.logger.info("Creating pre-trained ObjectiveHead from emulator...")
        
        # Create head network
        head = ObjectiveHead(self.n_parameters, self.n_objectives)
        
        # Extract useful layers from trained emulator
        emulator_layers = list(self.emulator.network.children())[:-1]  # All except output
        
        # Initialize head's network with emulator's learned representations
        # This is a simplified transfer - you might want more sophisticated mapping
        if len(emulator_layers) >= 2:
            # Use emulator's first hidden layer weights for head initialization
            with torch.no_grad():
                first_emulator_layer = emulator_layers[0]
                if hasattr(head.net[0], 'weight') and hasattr(first_emulator_layer, 'weight'):
                    # Map parameter portion of head input to emulator's learned representations
                    param_dim = self.n_parameters
                    emulator_input_dim = first_emulator_layer.weight.shape[1]
                    if param_dim <= emulator_input_dim:
                        head.net[0].weight[:, -param_dim:] = first_emulator_layer.weight[:64, :param_dim]
        
        self.logger.info("ObjectiveHead pre-trained with emulator knowledge")
        return head

    def optimize_with_summa_autodiff(self,
                                use_sundials: bool = True,
                                fd_step: float = 1e-3,
                                steps: int = 200,
                                lr: float = 1e-2,
                                use_nn_head: bool = True,
                                initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optimize parameters using SUMMA autodiff integration with adaptive learning rate.
        
        This method enables end-to-end gradient computation through SUMMA,
        with an optional neural network head for objective post-processing.
        It supports both 'ADAM' and 'LBFGS' optimizers, configurable via
        the 'DPE_OPTIMIZER' setting in the confluence_config.
        
        Parameters
        ----------
        use_sundials : bool
            Whether to use Sundials analytical sensitivities
        fd_step : float
            Finite difference step size (fallback)
        steps : int
            Number of optimization steps
        lr : float
            Learning rate for Adam, or initial step size for L-BFGS
        use_nn_head : bool
            Whether to use neural network head
        initial_params : Dict[str, float], optional
            Initial parameter values
            
        Returns
        -------
        Dict[str, float]
            Optimized parameters
        """
        optimizer_choice = self.confluence_config.get('DPE_OPTIMIZER', 'ADAM').upper()
        
        self.logger.info(f"Starting SUMMA autodiff optimization (Sundials: {use_sundials})")
        self.logger.info(f"Using optimizer: {optimizer_choice}")
        start_time = time.time()
        
        # Check Sundials availability
        if use_sundials:
            try:
                test_x = np.full(self.n_parameters, 0.5)
                _ = self._summa_objective_jacobian(test_x)
            except NotImplementedError:
                self.logger.warning("Sundials sensitivities not available. Falling back to finite differences.")
                use_sundials = False
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        
        # Initialize parameters
        if initial_params:
            x0 = self.summa.normalize_parameters(initial_params)
        else:
            x0 = np.full(self.n_parameters, 0.5)
        
        x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)
        
        # Setup neural network head with optional pretraining
        objective_head = None
        if use_nn_head:
            if hasattr(self.emulator_config, 'pretrain_nn_head') and self.emulator_config.pretrain_nn_head:
                self.generate_training_data()
                self.train_emulator()
                objective_head = self._create_pretrained_head()
            else:
                objective_head = ObjectiveHead(self.n_parameters, self.n_objectives)
            
            objective_head = objective_head.to(device)
            pretrained = hasattr(self.emulator_config, 'pretrain_nn_head') and self.emulator_config.pretrain_nn_head
            self.logger.info(f"Using neural network head (pretrained: {pretrained})")
        
        # Setup optimizer based on configuration
        optimization_params = [x]
        if objective_head:
            optimization_params.extend(list(objective_head.parameters()))
        
        if optimizer_choice == 'ADAM':
            # ENHANCED: Adaptive learning rate for Adam
            initial_lr = max(lr * 5, 0.001)  # Start 5x higher, minimum 0.001
            optimizer = optim.Adam(optimization_params, lr=initial_lr)
            
            # ENHANCED: Add learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',           # Minimize loss
                factor=0.7,           # Reduce by 30% when plateauing
                patience=3,           # Wait 3 steps before reducing
                min_lr=lr * 0.1,      # Don't go below 10% of original lr
                verbose=False         # We'll log manually
            )
            self.logger.info(f"Using adaptive ADAM with initial LR: {initial_lr:.6f}")
            
        elif optimizer_choice == 'LBFGS':
            optimizer = optim.LBFGS(optimization_params, lr=lr)
            scheduler = None
        else:
            raise ValueError(f"Unknown optimizer '{optimizer_choice}'. Use 'ADAM' or 'LBFGS'.")

        best_loss = float('inf')
        best_x = x.detach().clone()
        loss_history = []
        
        # ENHANCED: Progress tracking for adaptive features (Adam only)
        if optimizer_choice == 'ADAM':
            recent_losses = []
            stagnation_counter = 0
        
        # ==================================
        # Optimization Loop
        # ==================================
        if optimizer_choice == 'ADAM':
            progress_bar = tqdm(range(1, steps + 1), desc="SUMMA Autodiff (Adam)")
            for step in progress_bar:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)
                
                objectives_raw = SummaAutogradOp.apply(x, self, use_sundials, float(fd_step))
                objectives_adjusted = objective_head(objectives_raw, x) if objective_head else objectives_raw
                loss = self._compute_weighted_loss_tensor(objectives_adjusted)

                loss.backward()
                
                # ENHANCED: Gradient clipping to prevent explosive updates
                torch.nn.utils.clip_grad_norm_(optimization_params, max_norm=1.0)
                
                optimizer.step()
                
                # ENHANCED: Track progress for adaptive features
                current_loss = loss.item()
                loss_history.append(current_loss)
                recent_losses.append(current_loss)
                
                # Keep only last 5 losses for trend analysis
                if len(recent_losses) > 5:
                    recent_losses.pop(0)
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                # ENHANCED: Update learning rate based on progress
                scheduler.step(current_loss)
                current_lr = optimizer.param_groups[0]['lr']
                
                # ENHANCED: Detect stagnation and boost learning ratesing adaptive ADAM with initial L
                if stagnation_counter >= 5 and len(recent_losses) >= 3:
                    recent_improvement = recent_losses[0] - recent_losses[-1]
                    if recent_improvement < 1e-4:  # Very small improvement
                        # Temporary learning rate boost
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = min(param_group['lr'] * 2.0, initial_lr)
                        self.logger.info(f"Step {step}: Stagnation detected, boosting LR to {optimizer.param_groups[0]['lr']:.6f}")
                        stagnation_counter = 0
                
                # Enhanced logging and progress bar update
                obj_values = objectives_raw.detach().cpu().numpy()
                postfix_dict = {
                    'Loss': f'{current_loss:.6f}', 
                    'Best': f'{best_loss:.6f}',
                    'LR': f'{current_lr:.6f}',
                    'KGE': f'{obj_values[0]:.3f}',  # Shorter format
                    'NSE': f'{obj_values[1]:.3f}'   # Shorter format  
                }
                if len(self.objective_names) > 0: postfix_dict[self.objective_names[0]] = f'{obj_values[0]:.4f}'
                if len(self.objective_names) > 1: postfix_dict[self.objective_names[1]] = f'{obj_values[1]:.4f}'
                progress_bar.set_postfix(postfix_dict)
                
                if step % 5 == 0 or step == 1:
                    obj_str = ", ".join([f"{name}={val:.4f}" for name, val in zip(self.objective_names, obj_values)])
                    self.logger.info(f"Step {step:4d}: Loss={current_loss:.6f}, Best={best_loss:.6f}, LR={current_lr:.6f}")
                    self.logger.info(f"          Objectives: [ {obj_str} ]")

        elif optimizer_choice == 'LBFGS':
            progress_bar = tqdm(range(1, steps + 1), desc="SUMMA Autodiff (L-BFGS)")
            
            # This variable will be updated by the closure to capture objectives
            last_eval = {'objectives_raw': None}

            for step in progress_bar:
                def closure():
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        x.clamp_(0.0, 1.0)
                    
                    objectives_raw = SummaAutogradOp.apply(x, self, use_sundials, float(fd_step))
                    
                    # Capture the objectives for logging outside the closure
                    last_eval['objectives_raw'] = objectives_raw.detach().clone()
                    
                    objectives_adjusted = objective_head(objectives_raw, x) if objective_head else objectives_raw
                    loss = self._compute_weighted_loss_tensor(objectives_adjusted)
                    
                    loss.backward()
                    return loss  # The closure MUST return only the scalar loss

                # optimizer.step() calls the closure and returns the final loss
                loss = optimizer.step(closure)
                
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_x = x.detach().clone()

                # Logging and progress bar update using the captured objectives
                objectives_raw = last_eval['objectives_raw']
                if objectives_raw is not None:
                    obj_values = objectives_raw.cpu().numpy()
                    postfix_dict = {'Loss': f'{current_loss:.6f}', 'Best': f'{best_loss:.6f}'}
                    if len(self.objective_names) > 0: postfix_dict[self.objective_names[0]] = f'{obj_values[0]:.4f}'
                    if len(self.objective_names) > 1: postfix_dict[self.objective_names[1]] = f'{obj_values[1]:.4f}'
                    progress_bar.set_postfix(postfix_dict)

                    if step % 5 == 0 or step == 1:
                        obj_str = ", ".join([f"{name}={val:.4f}" for name, val in zip(self.objective_names, obj_values)])
                        self.logger.info(f"Step {step:4d}: Loss={current_loss:.6f}, Best={best_loss:.6f}")
                        self.logger.info(f"          Objectives: [ {obj_str} ]")

        # ==================================
        # Finalization (common to both optimizers)
        # ==================================
        optimization_time = time.time() - start_time
        self.timing_results['autodiff_optimization'] = optimization_time
        
        final_params = self.summa.denormalize_parameters(best_x.clamp(0.0, 1.0).detach().cpu().numpy())
        
        self.logger.info(f"SUMMA autodiff optimization completed in {optimization_time:.1f}s")
        self.logger.info(f"Best loss: {best_loss:.6f}")
        if optimizer_choice == 'ADAM':
            self.logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        return final_params

    # ==============================
    # Main Orchestration
    # ==============================

    def run_from_config(self) -> Dict[str, float]:
        """
        Run optimization based on configuration settings.
        
        Returns
        -------
        Dict[str, float]
            Optimized parameters
        """
        mode = self.confluence_config.get('EMULATOR_SETTING', 'EMULATOR').upper()
        
        self.logger.info("=" * 80)
        self.logger.info("DIFFERENTIABLE PARAMETER OPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Domain: {self.domain_name}")
        self.logger.info(f"Parameters: {self.n_parameters}")
        self.logger.info(f"Objectives: {self.objective_names}")
        self.logger.info("=" * 80)
        
        total_start_time = time.time()
        
        # Extract configuration parameters
        fd_step = float(self.confluence_config.get('DPE_FD_STEP', 1e-3))
        gd_step = float(self.confluence_config.get('DPE_GD_STEP_SIZE', 1e-1))
        autodiff_steps = int(self.confluence_config.get('DPE_AUTODIFF_STEPS', 200))
        autodiff_lr = float(self.confluence_config.get('DPE_AUTODIFF_LR', 1e-2))
        use_nn_head = bool(self.confluence_config.get('DPE_USE_NN_HEAD', True))
        use_sundials = bool(self.confluence_config.get('DPE_USE_SUNDIALS', True))
        force_retrain = bool(self.confluence_config.get('DPE_FORCE_RETRAIN', False))
        
        # Extract new configuration
        pretrain_head = bool(self.confluence_config.get('DPE_PRETRAIN_NN_HEAD', False))
    
        # Update emulator config
        self.emulator_config.pretrain_nn_head = pretrain_head

        # Determine cache path
        cache_setting = self.confluence_config.get("DPE_TRAINING_CACHE", "default")
        if cache_setting == "default" or not cache_setting:
            cache_path = None  # Uses default directory
        else:
            cache_path = Path(cache_setting)
     
        # Extract iterative configuration
        iterate_enabled = bool(self.confluence_config.get('DPE_EMULATOR_ITERATE', False))
        iterate_max_iterations = int(self.confluence_config.get('DPE_ITERATE_MAX_ITERATIONS', 5))
        iterate_samples_per_cycle = int(self.confluence_config.get('DPE_ITERATE_SAMPLES_PER_CYCLE', 100))
        iterate_sampling_radius = float(self.confluence_config.get('DPE_ITERATE_SAMPLING_RADIUS', 0.1))
        iterate_convergence_tol = float(self.confluence_config.get('DPE_ITERATE_CONVERGENCE_TOL', 1e-4))
        iterate_min_improvement = float(self.confluence_config.get('DPE_ITERATE_MIN_IMPROVEMENT', 1e-6))
        iterate_sampling_method = self.confluence_config.get('DPE_ITERATE_SAMPLING_METHOD', 'gaussian')
        
        # Update emulator config with iterative settings
        self.emulator_config.iterate_enabled = iterate_enabled
        self.emulator_config.iterate_max_iterations = iterate_max_iterations
        self.emulator_config.iterate_samples_per_cycle = iterate_samples_per_cycle
        self.emulator_config.iterate_sampling_radius = iterate_sampling_radius
        self.emulator_config.iterate_convergence_tol = iterate_convergence_tol
        self.emulator_config.iterate_min_improvement = iterate_min_improvement
        self.emulator_config.iterate_sampling_method = iterate_sampling_method
        
        try:
            if mode == 'EMULATOR':
                if iterate_enabled:
                    self.logger.info("Running ITERATIVE EMULATOR mode: Active learning with refinement")
                else:
                    self.logger.info("Running EMULATOR mode: Standard NN surrogate optimization")
                
                # Generate/load initial training data
                self.generate_training_data(cache_path=cache_path, force=force_retrain)
                
                # Train initial emulator
                self.train_emulator()
                
                # Optimize using emulator (iterative or standard)
                optimized_params = self.optimize_parameters_iterative()
                
            elif mode == 'FD':
                self.logger.info("Running FD mode: Finite difference optimization")
                
                optimized_params = self.optimize_with_fd(
                    step_size=gd_step,
                    fd_step=fd_step,
                    max_iters=self.emulator_config.optimization_steps
                )
                
            elif mode == 'SUMMA_AUTODIFF':
                self.logger.info("Running SUMMA_AUTODIFF mode: End-to-end autodiff")
                
                optimized_params = self.optimize_with_summa_autodiff(
                    use_sundials=use_sundials,
                    fd_step=fd_step,
                    steps=autodiff_steps,
                    lr=autodiff_lr,
                    use_nn_head=use_nn_head
                )
                
            elif mode == 'SUMMA_AUTODIFF_FD':
                self.logger.info("Running SUMMA_AUTODIFF_FD mode: Autodiff with FD Jacobian")
                
                optimized_params = self.optimize_with_summa_autodiff(
                    use_sundials=False,  # Force finite differences
                    fd_step=fd_step,
                    steps=autodiff_steps,
                    lr=autodiff_lr,
                    use_nn_head=use_nn_head
                )
                
            else:
                raise ValueError(f"Unknown EMULATOR_SETTING: {mode}")
            
            # Record total time
            total_time = time.time() - total_start_time
            self.timing_results['total_optimization'] = total_time
            
            # Final validation
            self.logger.info("Running final validation...")
            validation_start = time.time()
            validation_metrics = self.validate_optimization(optimized_params)
            validation_time = time.time() - validation_start
            self.timing_results['final_validation'] = validation_time
            
            # Log final results
            self._log_optimization_summary(mode, optimized_params, validation_metrics)
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise

    def validate_optimization(self, optimized_params: Dict[str, float]) -> Dict[str, float]:
        """
        Validate optimized parameters by running final SUMMA simulation.
        
        Parameters
        ----------
        optimized_params : Dict[str, float]
            Optimized parameter values
            
        Returns
        -------
        Dict[str, float]
            Validation metrics
        """
        task = {
            'individual_id': 0,
            'params': optimized_params,
            'proc_id': 0,
            'evaluation_id': 'dpe_final_validation',
            'multiobjective': True,  
            'objective_names': self.objective_names 
        }
        
        results = self.summa.backend._run_parallel_evaluations([task])
        actual_metrics = results[0].get('metrics') if results else {}
        
        # Log validation results for convergence monitoring
        if actual_metrics:
            self.logger.info("VALIDATION CONVERGENCE CHECK:")
            for obj_name in self.objective_names:
                actual_value = actual_metrics.get(obj_name, actual_metrics.get(f"Calib_{obj_name}", "N/A"))
                self.logger.info(f"  {obj_name}: {actual_value}")
        
        return actual_metrics or {}

    def _log_optimization_summary(self, mode: str, params: Dict[str, float], validation: Dict[str, float]):
        """Log comprehensive optimization summary."""
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"Domain: {self.domain_name}")
        
        # Timing results
        self.logger.info("\nTIMING RESULTS:")
        for phase, duration in self.timing_results.items():
            self.logger.info(f"  {phase}: {duration:.1f}s")
        
        # Validation metrics
        if validation:
            self.logger.info("\nFINAL VALIDATION METRICS:")
            for obj_name in self.objective_names:
                value = validation.get(obj_name, validation.get(f"Calib_{obj_name}", "N/A"))
                self.logger.info(f"  {obj_name}: {value}")
        
        # Optimized parameters (first few)
        self.logger.info("\nOPTIMIZED PARAMETERS (sample):")
        param_items = list(params.items())[:5]  # Show first 5 parameters
        for name, value in param_items:
            if isinstance(value, np.ndarray):
                self.logger.info(f"  {name}: {value[0]:.6f} (first element)")
            else:
                self.logger.info(f"  {name}: {value:.6f}")
        
        if len(params) > 5:
            self.logger.info(f"  ... and {len(params) - 5} more parameters")
        
        self.logger.info("=" * 80)

    def save_results(self, optimized_params: Dict[str, float], results_dir: Path):
        """
        Save optimization results to specified directory.
        
        Parameters
        ----------
        optimized_params : Dict[str, float]
            Optimized parameter values
        results_dir : Path
            Output directory
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save optimized parameters
        params_serializable = {}
        for key, value in optimized_params.items():
            if isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist()
            else:
                params_serializable[key] = value
        
        params_file = results_dir / 'optimized_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(params_serializable, f, indent=2)
        
        # Save emulator model if available
        if hasattr(self, 'emulator'):
            model_file = results_dir / 'emulator_model.pt'
            torch.save(self.emulator.state_dict(), model_file)
        
        # Save metadata
        metadata = {
            'domain_name': self.domain_name,
            'parameter_names': self.param_names,
            'objective_names': self.objective_names,
            'optimization_mode': self.confluence_config.get('EMULATOR_SETTING', 'EMULATOR'),
            'timing_results': self.timing_results,
            'training_data_sizes': {
                'training': len(self.training_data.get('parameters', [])),
                'validation': len(self.validation_data.get('parameters', []))
            },
            'generated_at': datetime.now().isoformat()
        }
        
        metadata_file = results_dir / 'optimization_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_dir}")


# ==============================
# Command Line Interface
# ==============================

def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CONFLUENCE Differentiable Parameter Optimization")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--domain', type=str, 
                       help='Domain name (overrides config)')
    parser.add_argument('--mode', type=str, 
                       choices=['EMULATOR', 'FD', 'SUMMA_AUTODIFF', 'SUMMA_AUTODIFF_FD'],
                       help='Optimization mode (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Error: Configuration file not found: {config_path}')
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.domain:
        config['DOMAIN_NAME'] = args.domain
    if args.mode:
        config['EMULATOR_SETTING'] = args.mode
    
    # Extract domain name
    domain_name = config.get('DOMAIN_NAME', 'test_domain')
    
    # Create emulator configuration from CONFLUENCE config
    emulator_config = EmulatorConfig(
        hidden_dims=config.get('DPE_HIDDEN_DIMS', [256, 128, 64]),
        n_training_samples=config.get('DPE_TRAINING_SAMPLES', 500),
        n_validation_samples=config.get('DPE_VALIDATION_SAMPLES', 100),
        n_epochs=config.get('DPE_EPOCHS', 300),
        learning_rate=config.get('DPE_LEARNING_RATE', 1e-3),
        optimization_lr=config.get('DPE_OPTIMIZATION_LR', 1e-2),
        optimization_steps=config.get('DPE_OPTIMIZATION_STEPS', 200),
    )
    
    try:
        # Initialize optimizer
        dpe = DifferentiableParameterOptimizer(config, domain_name, emulator_config)
        
        # Run optimization
        optimized_params = dpe.run_from_config()
        
        # Validate results
        validation_metrics = dpe.validate_optimization(optimized_params)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        mode = config.get('EMULATOR_SETTING', 'EMULATOR').lower()
        output_dir = Path(f"results_dpe_{mode}_{domain_name}_{timestamp}")
        dpe.save_results(optimized_params, output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("✅ DIFFERENTIABLE PARAMETER OPTIMIZATION COMPLETED")
        print("="*80)
        print(f"Domain: {domain_name}")
        print(f"Mode: {config.get('EMULATOR_SETTING', 'EMULATOR')}")
        print(f"Output: {output_dir}")
        print(f"Total time: {dpe.timing_results.get('total_optimization', 0):.1f}s")
        print("="*80)
        
        # Print results as JSON for easy parsing
        result_summary = {
            'optimized_parameters': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                   for k, v in optimized_params.items()},
            'validation_metrics': validation_metrics,
            'timing_results': dpe.timing_results,
            'output_directory': str(output_dir)
        }
        
        print("\nDETAILED RESULTS:")
        print(json.dumps(result_summary, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())