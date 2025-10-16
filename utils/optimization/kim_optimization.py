#!/usr/bin/env python3
"""
KIM (Knowledge-Informed Mapping) Optimizer for CONFLUENCE

This module integrates the KIM toolkit with CONFLUENCE's optimization framework,
providing knowledge-informed deep learning for hydrological model calibration
using mutual information-based sensitivity analysis and inverse mapping.

The KIM optimizer implements a two-step approach:
1. Mutual Information analysis to identify sensitive parameters and responses
2. Deep learning inverse mapping to directly estimate parameters from observations

Key features:
- Uses fewer model realizations (few hundred vs thousands for traditional methods)
- Knowledge-guided parameter selection based on sensitivity analysis
- Direct parameter estimation from model responses
- Integration with CONFLUENCE's existing model infrastructure

References:
- Jiang, P., Shuai, P., Sun, A., et al. (2023). Knowledge-informed deep learning 
  for hydrological model calibration: an application to Coal Creek Watershed in Colorado. 
  Hydrol. Earth Syst. Sci., 27, 2621-2643.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings

try:
    import kim_jax as kim
    KIM_AVAILABLE = True
except ImportError:
    KIM_AVAILABLE = False
    warnings.warn("KIM (kim-jax) not available. Install with: pip install kim-jax")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. Install with: pip install jax")

from utils.optimization.iterative_optimizer import BaseOptimizer


class KIMOptimizer(BaseOptimizer):
    """
    Knowledge-Informed Mapping (KIM) Optimizer for CONFLUENCE
    
    This optimizer uses the KIM toolkit to perform hydrological model calibration
    through a knowledge-informed deep learning approach that combines mutual
    information-based sensitivity analysis with inverse neural network mapping.
    
    The KIM approach offers several advantages:
    - Reduced computational requirements (hundreds vs thousands of model runs)
    - Knowledge-guided parameter selection
    - Direct parameter estimation from observations
    - Handles nonlinear parameter-response relationships effectively
    
    Workflow:
    1. Preliminary MI analysis (~50 runs) to screen parameters
    2. Full MI analysis (~200-400 runs) for sensitivity mapping  
    3. Train inverse neural network mapping
    4. Estimate parameters directly from observations
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize KIM optimizer
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing KIM-specific parameters
        logger : logging.Logger
            Logger instance for tracking operations
        """
        super().__init__(config, logger)
        
        if not KIM_AVAILABLE:
            raise ImportError("KIM (kim-jax) package not available. Install with: pip install kim-jax")
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX package not available. Install with: pip install jax")
        
        # KIM-specific configuration parameters
        self.preliminary_runs = config.get('KIM_PRELIMINARY_RUNS', 50)
        self.full_analysis_runs = config.get('KIM_FULL_ANALYSIS_RUNS', 400)
        self.sensitivity_threshold = config.get('KIM_SENSITIVITY_THRESHOLD', 0.05)  # 5%
        self.mi_bins = config.get('KIM_MI_BINS', 10)
        self.bootstrap_samples = config.get('KIM_BOOTSTRAP_SAMPLES', 100)
        self.significance_level = config.get('KIM_SIGNIFICANCE_LEVEL', 0.95)
        
        # Neural network parameters
        self.nn_epochs = config.get('KIM_NN_EPOCHS', 1000)
        self.nn_batch_size = config.get('KIM_NN_BATCH_SIZE', 32)
        self.nn_learning_rate = config.get('KIM_NN_LEARNING_RATE', 0.001)
        self.nn_hidden_layers = config.get('KIM_NN_HIDDEN_LAYERS', 3)
        self.train_val_test_split = config.get('KIM_TRAIN_VAL_TEST_SPLIT', [0.75, 0.125, 0.125])
        
        # Observation error for uncertainty analysis
        self.observation_error = config.get('KIM_OBSERVATION_ERROR', 0.05)  # 5%
        self.uncertainty_samples = config.get('KIM_UNCERTAINTY_SAMPLES', 100)
        
        # KIM state variables
        self.sensitive_params = None
        self.mi_results = None
        self.inverse_mappings = {}
        self.training_data = None
        self.kim_results_dir = None
        
        # Initialize results directory
        self._setup_kim_directories()
        
        self.logger.info("KIM Optimizer initialized")
        self.logger.info(f"Preliminary runs: {self.preliminary_runs}")
        self.logger.info(f"Full analysis runs: {self.full_analysis_runs}")
        
    def get_algorithm_name(self) -> str:
        """Return algorithm identifier"""
        return "KIM"
    
    def _setup_kim_directories(self) -> None:
        """Set up directories for KIM-specific outputs"""
        project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.kim_results_dir = project_dir / "results" / "optimization" / "kim"
        self.kim_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.kim_results_dir / "sensitivity_analysis").mkdir(exist_ok=True)
        (self.kim_results_dir / "inverse_mappings").mkdir(exist_ok=True)
        (self.kim_results_dir / "parameter_estimation").mkdir(exist_ok=True)
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """
        Run the complete KIM optimization algorithm
        
        Returns
        -------
        Tuple[Dict, float, List]
            Best parameters, best score, and iteration history
        """
        self.logger.info("Starting KIM optimization algorithm")
        
        try:
            # Step 1: Preliminary sensitivity analysis to screen parameters
            self.logger.info("Step 1: Preliminary sensitivity analysis")
            preliminary_results = self._run_preliminary_sensitivity()
            
            # Step 2: Full sensitivity analysis on screened parameters
            self.logger.info("Step 2: Full sensitivity analysis")
            full_mi_results = self._run_full_sensitivity_analysis()
            
            # Step 3: Develop inverse neural network mappings
            self.logger.info("Step 3: Training inverse neural network mappings")
            self._train_inverse_mappings()
            
            # Step 4: Estimate parameters from observations
            self.logger.info("Step 4: Parameter estimation from observations")
            best_params, best_score = self._estimate_parameters_from_observations()
            
            # Step 5: Uncertainty analysis (optional)
            if self.uncertainty_samples > 0:
                self.logger.info("Step 5: Uncertainty analysis")
                uncertainty_results = self._run_uncertainty_analysis()
            
            # Save final results
            self._save_kim_results(best_params, best_score)
            
            iteration_history = self._compile_iteration_history()
            
            self.logger.info("KIM optimization completed successfully")
            return best_params, best_score, iteration_history
            
        except Exception as e:
            self.logger.error(f"KIM optimization failed: {str(e)}")
            raise
    
    def _run_preliminary_sensitivity(self) -> Dict[str, Any]:
        """
        Run preliminary mutual information analysis to screen parameters
        
        Returns
        -------
        Dict[str, Any]
            Preliminary sensitivity results
        """
        self.logger.info(f"Running preliminary sensitivity analysis with {self.preliminary_runs} model runs")
        
        # Generate parameter samples using existing parameter manager
        param_samples = self._generate_parameter_samples(self.preliminary_runs)
        
        # Run model simulations
        model_results = self._run_model_ensemble(param_samples, run_type="preliminary")
        
        # Calculate mutual information
        preliminary_mi = self._calculate_mutual_information(
            param_samples, model_results, run_type="preliminary"
        )
        
        # Screen parameters based on sensitivity threshold
        sensitive_params = self._screen_sensitive_parameters(preliminary_mi)
        
        self.logger.info(f"Preliminary screening identified {len(sensitive_params)} sensitive parameters")
        return {
            'mi_results': preliminary_mi,
            'sensitive_params': sensitive_params,
            'param_samples': param_samples,
            'model_results': model_results
        }
    
    def _run_full_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Run full mutual information analysis on screened parameters
        
        Returns
        -------
        Dict[str, Any]
            Full sensitivity analysis results
        """
        self.logger.info(f"Running full sensitivity analysis with {self.full_analysis_runs} model runs")
        
        # Generate larger parameter sample focusing on sensitive parameters
        param_samples = self._generate_parameter_samples(
            self.full_analysis_runs, 
            focus_params=self.sensitive_params
        )
        
        # Run model ensemble
        model_results = self._run_model_ensemble(param_samples, run_type="full")
        
        # Calculate detailed mutual information
        full_mi_results = self._calculate_mutual_information(
            param_samples, model_results, run_type="full"
        )
        
        # Store for use in inverse mapping
        self.mi_results = full_mi_results
        self.training_data = {
            'param_samples': param_samples,
            'model_results': model_results
        }
        
        # Save MI results
        self._save_sensitivity_results(full_mi_results)
        
        return full_mi_results
    
    def _generate_parameter_samples(self, n_samples: int, focus_params: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate parameter samples using CONFLUENCE's parameter manager
        
        Parameters
        ----------
        n_samples : int
            Number of parameter samples to generate
        focus_params : Optional[List[str]]
            Parameters to focus sampling on (for full analysis)
            
        Returns
        -------
        np.ndarray
            Normalized parameter samples [n_samples, n_params]
        """
        param_count = len(self.parameter_manager.all_param_names)
        
        # Use Sobol sequence for better coverage (following KIM paper approach)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=param_count, scramble=True)
            samples = sampler.random(n_samples)
        except ImportError:
            self.logger.warning("scipy.stats.qmc not available, using random sampling")
            samples = np.random.random((n_samples, param_count))
        
        return samples
    
    def _run_model_ensemble(self, param_samples: np.ndarray, run_type: str) -> Dict[str, np.ndarray]:
        """
        Run ensemble of model simulations using CONFLUENCE's model executor
        
        Parameters
        ----------
        param_samples : np.ndarray
            Normalized parameter samples
        run_type : str
            Type of run ('preliminary' or 'full')
            
        Returns
        -------
        Dict[str, np.ndarray]
            Model results with time series data for each run
        """
        n_samples = param_samples.shape[0]
        self.logger.info(f"Running {n_samples} model simulations for {run_type} analysis")
        
        # Initialize results storage
        model_results = {
            'streamflow': [],
            'timestamps': [],
            'success_mask': []
        }
        
        # Run simulations (using parallel processing if available)
        if self.use_parallel:
            results = self._run_ensemble_parallel(param_samples)
        else:
            results = self._run_ensemble_sequential(param_samples)
        
        # Process results
        for i, result in enumerate(results):
            if result is not None and result.get('success', False):
                # Extract time series data
                sim_data = result.get('sim_streamflow', [])
                timestamps = result.get('timestamps', [])
                
                model_results['streamflow'].append(sim_data)
                model_results['timestamps'].append(timestamps)
                model_results['success_mask'].append(True)
            else:
                # Handle failed runs
                model_results['streamflow'].append([])
                model_results['timestamps'].append([])
                model_results['success_mask'].append(False)
        
        # Convert to arrays
        model_results['success_mask'] = np.array(model_results['success_mask'])
        successful_runs = np.sum(model_results['success_mask'])
        
        self.logger.info(f"Completed {successful_runs}/{n_samples} successful model runs")
        
        if successful_runs < n_samples * 0.5:
            self.logger.warning(f"Low success rate: {successful_runs}/{n_samples} runs succeeded")
        
        return model_results
    
    def _run_ensemble_sequential(self, param_samples: np.ndarray) -> List[Dict]:
        """Run ensemble sequentially"""
        results = []
        for i, normalized_params in enumerate(param_samples):
            # Denormalize parameters
            params_dict = self.parameter_manager.denormalize_parameters(normalized_params)
            
            # Run single model simulation
            result = self._evaluate_single_run(params_dict, run_id=i)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Completed {i + 1}/{len(param_samples)} runs")
        
        return results
    
    def _run_ensemble_parallel(self, param_samples: np.ndarray) -> List[Dict]:
        """Run ensemble in parallel using existing parallel infrastructure"""
        evaluation_tasks = []
        
        for i, normalized_params in enumerate(param_samples):
            params_dict = self.parameter_manager.denormalize_parameters(normalized_params)
            task = {
                'individual_id': i,
                'params': params_dict,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"kim_{i:04d}"
            }
            evaluation_tasks.append(task)
        
        # Use parent class parallel evaluation method
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        # Convert to expected format
        model_results = []
        for result in results:
            if result.get('score') is not None:
                # Extract additional simulation data if available
                model_result = {
                    'success': True,
                    'sim_streamflow': result.get('sim_data', {}).get('streamflow', []),
                    'timestamps': result.get('sim_data', {}).get('timestamps', []),
                    'score': result['score']
                }
            else:
                model_result = {'success': False}
            
            model_results.append(model_result)
        
        return model_results
    
    def _evaluate_single_run(self, params_dict: Dict[str, float], run_id: int) -> Dict:
        """
        Evaluate single model run and extract time series data
        
        Parameters
        ----------
        params_dict : Dict[str, float]
            Parameter dictionary
        run_id : int
            Run identifier
            
        Returns
        -------
        Dict
            Run results including time series data
        """
        try:
            # Use existing model executor from parent class
            result = self.model_executor.run_model(params_dict)
            
            if result.get('success', False):
                # Extract time series data for MI calculation
                sim_data = result.get('sim_data', {})
                
                return {
                    'success': True,
                    'sim_streamflow': sim_data.get('streamflow', []),
                    'timestamps': sim_data.get('timestamps', []),
                    'score': result.get('score', 0.0)
                }
            else:
                return {'success': False}
                
        except Exception as e:
            self.logger.warning(f"Run {run_id} failed: {str(e)}")
            return {'success': False}
    
    def _calculate_mutual_information(self, param_samples: np.ndarray, 
                                    model_results: Dict[str, Any],
                                    run_type: str) -> Dict[str, Any]:
        """
        Calculate mutual information between parameters and model responses
        
        This implements the KIM mutual information analysis approach,
        calculating MI between each parameter and model response at each time step.
        
        Parameters
        ----------
        param_samples : np.ndarray
            Parameter samples used for model runs
        model_results : Dict[str, Any]
            Model simulation results
        run_type : str
            Type of analysis ('preliminary' or 'full')
            
        Returns
        -------
        Dict[str, Any]
            Mutual information results
        """
        self.logger.info("Calculating mutual information between parameters and responses")
        
        # Filter successful runs
        success_mask = model_results['success_mask']
        valid_param_samples = param_samples[success_mask]
        valid_streamflow = [model_results['streamflow'][i] for i in range(len(success_mask)) if success_mask[i]]
        
        n_successful = len(valid_streamflow)
        if n_successful < 20:
            raise ValueError(f"Too few successful runs ({n_successful}) for MI analysis")
        
        # Convert streamflow data to matrix format [n_runs, n_timesteps]
        max_timesteps = max(len(flow) for flow in valid_streamflow if len(flow) > 0)
        
        streamflow_matrix = np.full((n_successful, max_timesteps), np.nan)
        for i, flow_series in enumerate(valid_streamflow):
            if len(flow_series) > 0:
                streamflow_matrix[i, :len(flow_series)] = flow_series
        
        # Calculate MI for each parameter and timestep
        n_params = valid_param_samples.shape[1]
        param_names = self.parameter_manager.all_param_names
        
        mi_matrix = np.zeros((n_params, max_timesteps))
        mi_significance = np.zeros((n_params, max_timesteps), dtype=bool)
        
        for param_idx in range(n_params):
            param_values = valid_param_samples[:, param_idx]
            
            for time_idx in range(max_timesteps):
                response_values = streamflow_matrix[:, time_idx]
                
                # Skip timesteps with too many NaN values
                valid_mask = ~np.isnan(response_values)
                if np.sum(valid_mask) < 15:  # Need minimum data for MI
                    continue
                
                # Calculate MI using binning approach (following KIM methodology)
                mi_value = self._calculate_mi_binning(
                    param_values[valid_mask], 
                    response_values[valid_mask]
                )
                
                # Statistical significance test
                is_significant = self._test_mi_significance(
                    param_values[valid_mask], 
                    response_values[valid_mask], 
                    mi_value
                )
                
                mi_matrix[param_idx, time_idx] = mi_value if is_significant else 0.0
                mi_significance[param_idx, time_idx] = is_significant
        
        # Calculate summary statistics
        param_mi_summary = {}
        for param_idx, param_name in enumerate(param_names):
            param_mi_values = mi_matrix[param_idx, :]
            nonzero_mi = param_mi_values[param_mi_values > 0]
            
            param_mi_summary[param_name] = {
                'mean_mi': np.mean(param_mi_values),
                'max_mi': np.max(param_mi_values),
                'significant_fraction': np.mean(mi_significance[param_idx, :]),
                'significant_timesteps': np.sum(mi_significance[param_idx, :]),
                'total_timesteps': max_timesteps
            }
        
        mi_results = {
            'mi_matrix': mi_matrix,
            'mi_significance': mi_significance,
            'param_summary': param_mi_summary,
            'param_names': param_names,
            'n_successful_runs': n_successful,
            'n_timesteps': max_timesteps,
            'run_type': run_type
        }
        
        self.logger.info("Mutual information analysis completed")
        return mi_results
    
    def _calculate_mi_binning(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate mutual information using fixed binning method
        Following KIM paper methodology
        """
        # Use fixed number of bins as in KIM paper
        n_bins = self.mi_bins
        
        # Create bins
        x_bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
        y_bins = np.linspace(np.min(y), np.max(y), n_bins + 1)
        
        # Digitize data
        x_binned = np.digitize(x, x_bins) - 1
        y_binned = np.digitize(y, y_bins) - 1
        
        # Ensure values are within valid range
        x_binned = np.clip(x_binned, 0, n_bins - 1)
        y_binned = np.clip(y_binned, 0, n_bins - 1)
        
        # Calculate joint and marginal histograms
        joint_hist, _, _ = np.histogram2d(x_binned, y_binned, bins=[n_bins, n_bins])
        
        # Convert to probabilities
        joint_hist = joint_hist / np.sum(joint_hist)
        x_hist = np.sum(joint_hist, axis=1)
        y_hist = np.sum(joint_hist, axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_hist[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                    mi += joint_hist[i, j] * np.log(joint_hist[i, j] / (x_hist[i] * y_hist[j]))
        
        return mi
    
    def _test_mi_significance(self, x: np.ndarray, y: np.ndarray, mi_value: float) -> bool:
        """
        Test statistical significance of mutual information using bootstrap
        Following KIM paper methodology
        """
        if len(x) < 10:
            return False
        
        # Bootstrap test
        n_bootstrap = min(self.bootstrap_samples, 50)  # Limit for computational efficiency
        bootstrap_mi = []
        
        for _ in range(n_bootstrap):
            # Shuffle y to break dependence
            y_shuffled = np.random.permutation(y)
            bootstrap_mi_val = self._calculate_mi_binning(x, y_shuffled)
            bootstrap_mi.append(bootstrap_mi_val)
        
        # Calculate significance threshold
        threshold = np.percentile(bootstrap_mi, self.significance_level * 100)
        
        return mi_value > threshold
    
    def _screen_sensitive_parameters(self, mi_results: Dict[str, Any]) -> List[str]:
        """
        Screen parameters based on sensitivity threshold
        Following KIM paper methodology
        """
        param_summary = mi_results['param_summary']
        sensitive_params = []
        
        for param_name, summary in param_summary.items():
            # Use the same criterion as KIM paper: 
            # parameters with >5% of timesteps showing significant MI
            if summary['significant_fraction'] > self.sensitivity_threshold:
                sensitive_params.append(param_name)
        
        self.sensitive_params = sensitive_params
        
        self.logger.info("Parameter sensitivity screening results:")
        for param_name in self.parameter_manager.all_param_names:
            summary = param_summary[param_name]
            status = "SELECTED" if param_name in sensitive_params else "EXCLUDED"
            self.logger.info(
                f"  {param_name}: {summary['significant_fraction']:.3f} significant fraction - {status}"
            )
        
        return sensitive_params
    
    def _train_inverse_mappings(self) -> None:
        """
        Train inverse neural network mappings for parameter estimation
        Following KIM knowledge-informed approach
        """
        if self.mi_results is None or self.training_data is None:
            raise ValueError("Must run sensitivity analysis before training inverse mappings")
        
        self.logger.info("Training inverse neural network mappings")
        
        # Prepare training data
        param_samples = self.training_data['param_samples']
        model_results = self.training_data['model_results']
        
        # Filter successful runs
        success_mask = model_results['success_mask']
        valid_param_samples = param_samples[success_mask]
        valid_streamflow = [model_results['streamflow'][i] for i in range(len(success_mask)) if success_mask[i]]
        
        # Convert streamflow to matrix
        max_timesteps = max(len(flow) for flow in valid_streamflow if len(flow) > 0)
        streamflow_matrix = np.full((len(valid_streamflow), max_timesteps), np.nan)
        
        for i, flow_series in enumerate(valid_streamflow):
            if len(flow_series) > 0:
                streamflow_matrix[i, :len(flow_series)] = flow_series
        
        # Train separate inverse mapping for each sensitive parameter
        for param_idx, param_name in enumerate(self.parameter_manager.all_param_names):
            if param_name not in self.sensitive_params:
                continue
            
            self.logger.info(f"Training inverse mapping for parameter: {param_name}")
            
            # Select informative responses for this parameter based on MI
            mi_values = self.mi_results['mi_matrix'][param_idx, :]
            significant_mask = mi_values > 0
            
            if not np.any(significant_mask):
                self.logger.warning(f"No significant responses found for {param_name}")
                continue
            
            # Create training dataset
            X = streamflow_matrix[:, significant_mask]  # Input: significant responses
            y = valid_param_samples[:, param_idx]       # Output: parameter values
            
            # Remove samples with NaN values
            valid_samples = ~np.any(np.isnan(X), axis=1)
            X = X[valid_samples]
            y = y[valid_samples]
            
            if len(X) < 20:
                self.logger.warning(f"Insufficient data for {param_name}: {len(X)} samples")
                continue
            
            # Train neural network mapping
            inverse_mapping = self._train_single_inverse_mapping(X, y, param_name)
            self.inverse_mappings[param_name] = inverse_mapping
        
        self.logger.info(f"Trained {len(self.inverse_mappings)} inverse mappings")
    
    def _train_single_inverse_mapping(self, X: np.ndarray, y: np.ndarray, param_name: str) -> Dict[str, Any]:
        """
        Train a single inverse neural network mapping
        
        This creates a multilayer perceptron that maps from informative model responses
        to the parameter value, following the KIM methodology.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1-self.train_val_test_split[0], random_state=42
        )
        
        val_size = self.train_val_test_split[1] / (self.train_val_test_split[1] + self.train_val_test_split[2])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_size, random_state=42
        )
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
        
        # Determine network architecture (following KIM paper)
        input_size = X_train_scaled.shape[1]
        output_size = 1
        
        # Use arithmetic sequence for hidden layer sizes
        hidden_layer_sizes = []
        for i in range(self.nn_hidden_layers):
            layer_size = int(input_size - i * (input_size - output_size) / (self.nn_hidden_layers + 1))
            hidden_layer_sizes.append(max(layer_size, output_size))
        
        # Train neural network
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            batch_size=min(self.nn_batch_size, len(X_train_scaled)),
            learning_rate_init=self.nn_learning_rate,
            max_iter=self.nn_epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42
        )
        
        mlp.fit(X_train_scaled, y_train_scaled)
        
        # Evaluate performance
        y_pred_scaled = mlp.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.logger.info(f"  {param_name} - MSE: {mse:.6f}, R²: {r2:.6f}")
        
        return {
            'model': mlp,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'input_features': X.shape[1],
            'performance': {'mse': mse, 'r2': r2},
            'training_size': len(X_train)
        }
    
    def _estimate_parameters_from_observations(self) -> Tuple[Dict[str, float], float]:
        """
        Estimate parameters from observations using trained inverse mappings
        
        Returns
        -------
        Tuple[Dict[str, float], float]
            Estimated parameters and model performance score
        """
        self.logger.info("Estimating parameters from observations")
        
        # Load observations using existing calibration target
        observations = self._load_observations()
        
        if observations is None or len(observations) == 0:
            raise ValueError("No observations available for parameter estimation")
        
        # Estimate each parameter using its inverse mapping
        estimated_params = {}
        
        for param_name, mapping in self.inverse_mappings.items():
            try:
                # Extract relevant observation data based on MI results
                param_idx = self.parameter_manager.all_param_names.index(param_name)
                mi_values = self.mi_results['mi_matrix'][param_idx, :]
                significant_mask = mi_values > 0
                
                if not np.any(significant_mask):
                    continue
                
                # Select relevant observations
                obs_data = observations[:len(significant_mask)][significant_mask]
                
                # Handle missing data
                if np.any(np.isnan(obs_data)):
                    self.logger.warning(f"Missing observation data for {param_name}")
                    continue
                
                # Scale observations
                obs_scaled = mapping['scaler_X'].transform(obs_data.reshape(1, -1))
                
                # Predict parameter value
                param_scaled = mapping['model'].predict(obs_scaled)
                param_value = mapping['scaler_y'].inverse_transform(param_scaled.reshape(-1, 1))[0, 0]
                
                # Ensure parameter is within bounds
                param_value = np.clip(param_value, 0, 1)  # Normalized space
                
                estimated_params[param_name] = param_value
                
            except Exception as e:
                self.logger.warning(f"Failed to estimate {param_name}: {str(e)}")
        
        # Convert to full parameter dictionary
        full_params = self._create_full_parameter_dict(estimated_params)
        
        # Evaluate estimated parameters
        model_score = self._evaluate_estimated_parameters(full_params)
        
        self.logger.info(f"Parameter estimation completed. Model score: {model_score:.6f}")
        
        return full_params, model_score
    
    def _load_observations(self) -> Optional[np.ndarray]:
        """Load observation data for parameter estimation"""
        try:
            # Use existing calibration target to load observations
            obs_data = self.calibration_target.get_target_values()
            
            if obs_data is not None:
                # Convert to numpy array if needed
                if isinstance(obs_data, (list, pd.Series)):
                    obs_data = np.array(obs_data)
                
                return obs_data
            
        except Exception as e:
            self.logger.error(f"Failed to load observations: {str(e)}")
        
        return None
    
    def _create_full_parameter_dict(self, estimated_params: Dict[str, float]) -> Dict[str, float]:
        """
        Create full parameter dictionary with defaults for non-estimated parameters
        """
        # Get initial parameters as defaults
        initial_params = self.parameter_manager.get_initial_parameters()
        full_params = {}
        
        for param_name in self.parameter_manager.all_param_names:
            if param_name in estimated_params:
                # Use estimated value (convert from normalized space)
                normalized_value = estimated_params[param_name]
                full_params[param_name] = normalized_value
            else:
                # Use initial/default value
                if initial_params and param_name in initial_params:
                    # Normalize the initial parameter value
                    param_dict = {param_name: initial_params[param_name]}
                    normalized = self.parameter_manager.normalize_parameters(param_dict)
                    full_params[param_name] = normalized
                else:
                    # Use middle of parameter range as default
                    full_params[param_name] = 0.5
        
        # Convert to denormalized parameter dictionary
        normalized_array = np.array([full_params[name] for name in self.parameter_manager.all_param_names])
        denormalized_params = self.parameter_manager.denormalize_parameters(normalized_array)
        
        return denormalized_params
    
    def _evaluate_estimated_parameters(self, params_dict: Dict[str, float]) -> float:
        """Evaluate estimated parameters using model"""
        try:
            result = self._evaluate_individual_dict(params_dict)
            return result if result is not None else float('-inf')
        except Exception as e:
            self.logger.warning(f"Failed to evaluate estimated parameters: {str(e)}")
            return float('-inf')
    
    def _run_uncertainty_analysis(self) -> Dict[str, Any]:
        """
        Run uncertainty analysis by adding observation errors
        Following KIM methodology
        """
        self.logger.info("Running uncertainty analysis with observation errors")
        
        observations = self._load_observations()
        if observations is None:
            return {}
        
        # Generate noisy observations
        uncertainty_results = {
            'parameter_estimates': [],
            'model_scores': []
        }
        
        for i in range(self.uncertainty_samples):
            # Add noise to observations
            noise = np.random.normal(0, self.observation_error * np.std(observations), len(observations))
            noisy_obs = observations + noise
            
            # Estimate parameters with noisy observations
            try:
                estimated_params = self._estimate_parameters_with_observations(noisy_obs)
                model_score = self._evaluate_estimated_parameters(estimated_params)
                
                uncertainty_results['parameter_estimates'].append(estimated_params)
                uncertainty_results['model_scores'].append(model_score)
                
            except Exception as e:
                self.logger.debug(f"Uncertainty sample {i} failed: {str(e)}")
        
        # Calculate uncertainty statistics
        if uncertainty_results['parameter_estimates']:
            uncertainty_stats = self._calculate_uncertainty_statistics(uncertainty_results)
            uncertainty_results.update(uncertainty_stats)
        
        self.logger.info(f"Uncertainty analysis completed with {len(uncertainty_results['parameter_estimates'])} successful samples")
        
        return uncertainty_results
    
    def _estimate_parameters_with_observations(self, observations: np.ndarray) -> Dict[str, float]:
        """Estimate parameters with given observations (for uncertainty analysis)"""
        estimated_params = {}
        
        for param_name, mapping in self.inverse_mappings.items():
            try:
                param_idx = self.parameter_manager.all_param_names.index(param_name)
                mi_values = self.mi_results['mi_matrix'][param_idx, :]
                significant_mask = mi_values > 0
                
                if not np.any(significant_mask):
                    continue
                
                obs_data = observations[:len(significant_mask)][significant_mask]
                
                if np.any(np.isnan(obs_data)):
                    continue
                
                obs_scaled = mapping['scaler_X'].transform(obs_data.reshape(1, -1))
                param_scaled = mapping['model'].predict(obs_scaled)
                param_value = mapping['scaler_y'].inverse_transform(param_scaled.reshape(-1, 1))[0, 0]
                param_value = np.clip(param_value, 0, 1)
                
                estimated_params[param_name] = param_value
                
            except Exception as e:
                continue
        
        return self._create_full_parameter_dict(estimated_params)
    
    def _calculate_uncertainty_statistics(self, uncertainty_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate uncertainty statistics from ensemble results"""
        param_estimates = uncertainty_results['parameter_estimates']
        
        if not param_estimates:
            return {}
        
        # Extract parameter arrays
        param_arrays = {}
        for param_name in param_estimates[0].keys():
            values = [est[param_name] for est in param_estimates]
            param_arrays[param_name] = np.array(values)
        
        # Calculate statistics
        uncertainty_stats = {}
        for param_name, values in param_arrays.items():
            uncertainty_stats[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            }
        
        return {'uncertainty_statistics': uncertainty_stats}
    
    def _save_sensitivity_results(self, mi_results: Dict[str, Any]) -> None:
        """Save mutual information analysis results"""
        results_file = self.kim_results_dir / "sensitivity_analysis" / "mi_results.npz"
        
        np.savez(results_file, 
                mi_matrix=mi_results['mi_matrix'],
                mi_significance=mi_results['mi_significance'],
                param_names=mi_results['param_names'])
        
        # Save summary as CSV
        summary_df = pd.DataFrame(mi_results['param_summary']).T
        summary_file = self.kim_results_dir / "sensitivity_analysis" / "parameter_summary.csv"
        summary_df.to_csv(summary_file)
        
        self.logger.info(f"Sensitivity results saved to {results_file}")
    
    def _save_kim_results(self, best_params: Dict[str, float], best_score: float) -> None:
        """Save final KIM optimization results"""
        results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'sensitive_parameters': self.sensitive_params,
            'n_inverse_mappings': len(self.inverse_mappings),
            'algorithm': 'KIM',
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = self.kim_results_dir / "kim_optimization_results.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"KIM results saved to {results_file}")
    
    def _compile_iteration_history(self) -> List[Dict[str, Any]]:
        """Compile iteration history for compatibility with other optimizers"""
        history = [
            {
                'iteration': 0,
                'algorithm': 'KIM',
                'phase': 'Preliminary Sensitivity Analysis',
                'n_runs': self.preliminary_runs,
                'best_score': None
            },
            {
                'iteration': 1,
                'algorithm': 'KIM',
                'phase': 'Full Sensitivity Analysis', 
                'n_runs': self.full_analysis_runs,
                'best_score': None
            },
            {
                'iteration': 2,
                'algorithm': 'KIM',
                'phase': 'Inverse Mapping Training',
                'n_mappings': len(self.inverse_mappings),
                'best_score': None
            },
            {
                'iteration': 3,
                'algorithm': 'KIM',
                'phase': 'Parameter Estimation',
                'best_score': self.best_score if hasattr(self, 'best_score') else None
            }
        ]
        
        return history


# Integration function for OptimizationManager
def create_kim_optimizer(config: Dict[str, Any], logger: logging.Logger) -> KIMOptimizer:
    """
    Factory function to create KIM optimizer instance
    
    Parameters
    ----------
    config : Dict[str, Any]
        CONFLUENCE configuration dictionary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    KIMOptimizer
        Configured KIM optimizer instance
    """
    return KIMOptimizer(config, logger)