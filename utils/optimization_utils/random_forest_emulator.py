import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import xarray as xr
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import netCDF4 as nc 

class RandomForestEmulator:
    """
    Random Forest Emulator for CONFLUENCE.
    
    This class uses parameter sets and performance metrics to train a random forest model
    that can predict performance metrics based on parameter values. It then optimizes 
    the parameter values to maximize performance and runs SUMMA with the optimal parameter set.
    
    Optionally, attribute data can be used to improve predictions.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Random Forest Emulator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Paths to required data
        self.attributes_path = self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv"
        self.emulator_dir = self.project_dir / "emulation" / self.experiment_id
        self.parameter_sets_path = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
        self.performance_metrics_path = self.emulator_dir / "ensemble_analysis" / "performance_metrics.csv"
        
        # Output paths
        self.emulation_output_dir = self.emulator_dir / "rf_emulation"
        self.emulation_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.target_metric = self.config.get('EMULATION_TARGET_METRIC', 'KGE')
        self.n_estimators = self.config.get('EMULATION_RF_N_ESTIMATORS', 100)
        self.max_depth = self.config.get('EMULATION_RF_MAX_DEPTH', None)
        self.random_state = self.config.get('EMULATION_SEED', 42)
        
        # New flag to control attribute data usage
        self.use_attributes = self.config.get('EMULATION_USE_ATTRIBUTES', False)
        
        # Optimization settings
        self.n_optimization_runs = self.config.get('EMULATION_N_OPTIMIZATION_RUNS', 10)
        
        # Model and scaler objects
        self.model = None
        self.X_scaler = None
        self.param_mins = {}
        self.param_maxs = {}
        self.param_cols = []
        self.attribute_cols = []
        
        # Time period information from config for evaluation
        self.calibration_period = self._parse_date_range(self.config.get('CALIBRATION_PERIOD', ''))
        self.evaluation_period = self._parse_date_range(self.config.get('EVALUATION_PERIOD', ''))
        
        # Check if all required files exist
        self._check_required_files()
    
    def _parse_date_range(self, date_range_str):
        """Parse date range string from config into start and end dates."""
        if not date_range_str:
            return None, None
            
        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except:
            self.logger.warning(f"Could not parse date range: {date_range_str}")
        
        return None, None
            
    def _check_required_files(self):
        """Check if all required input files exist."""
        self.logger.info("Checking required input files")
        
        files_to_check = [
            (self.parameter_sets_path, "Parameter sets"),
            (self.performance_metrics_path, "Performance metrics")
        ]
        
        # Only check for attributes file if we're using attributes
        if self.use_attributes:
            files_to_check.append((self.attributes_path, "Attribute data"))
        
        missing_files = []
        for file_path, description in files_to_check:
            if not file_path.exists():
                self.logger.error(f"{description} file not found: {file_path}")
                missing_files.append((file_path, description))
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {[desc for _, desc in missing_files]}")
            
        self.logger.info("All required input files found")
                        
    def load_and_prepare_data(self):
        """
        Load and prepare the data for training the random forest model
        with improved handling of calibration and evaluation metrics.
        
        Returns:
            Tuple of X and y arrays for model training
        """
        self.logger.info("Loading and preparing data for random forest training")
        
        # Load parameter sets
        parameter_data = self._load_parameter_sets()
        self.logger.info(f"Loaded parameter data with shape: {parameter_data.shape}")
        
        # Load performance metrics
        try:
            performance_df = pd.read_csv(self.performance_metrics_path)
            self.logger.info(f"Loaded performance metrics with shape: {performance_df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading performance metrics: {str(e)}")
            raise
        
        # Print debugging info about the data
        self.logger.info(f"Parameter data columns: {parameter_data.columns.tolist()}")
        self.logger.info(f"Performance metrics columns: {performance_df.columns.tolist()}")
        
        # Special handling for performance metrics file
        if 'Run' in performance_df.columns:
            performance_df.set_index('Run', inplace=True)
        elif 'Unnamed: 0' in performance_df.columns:
            performance_df.set_index('Unnamed: 0', inplace=True)
        
        # Check for calibration and evaluation metrics
        if 'Calib_KGE' in performance_df.columns:
            # Use calibration metrics preferentially
            self.target_metric = 'Calib_KGE'
            self.logger.info(f"Using calibration metrics for training with target: {self.target_metric}")
        elif 'Eval_KGE' in performance_df.columns:
            # Fall back to evaluation metrics
            self.target_metric = 'Eval_KGE'
            self.logger.info(f"Using evaluation metrics for training with target: {self.target_metric}")
        elif 'KGE' in performance_df.columns:
            # Fall back to original metrics
            self.target_metric = 'KGE'
            self.logger.info(f"Using original metrics for training with target: {self.target_metric}")
        else:
            # Check for any KGE-containing column
            kge_cols = [col for col in performance_df.columns if 'KGE' in col]
            if kge_cols:
                self.target_metric = kge_cols[0]
                self.logger.info(f"Using {self.target_metric} as target metric")
            else:
                # Try other metrics
                available_metrics = []
                for prefix in ['Calib_', 'Eval_', '']:
                    for metric in ['NSE', 'RMSE', 'PBIAS']:
                        full_metric = f"{prefix}{metric}"
                        if full_metric in performance_df.columns:
                            available_metrics.append(full_metric)
                
                if available_metrics:
                    self.target_metric = available_metrics[0]
                    self.logger.info(f"No KGE metric found, using {self.target_metric} instead")
                else:
                    raise ValueError(f"No valid performance metrics found in {performance_df.columns}")
        
        # Filter out NaN values in the target metric
        valid_metrics = ~performance_df[self.target_metric].isna()
        self.logger.info(f"Found {valid_metrics.sum()} valid entries for {self.target_metric} out of {len(performance_df)}")
        
        if valid_metrics.sum() == 0:
            raise ValueError(f"No valid data for target metric {self.target_metric}")
        
        performance_df = performance_df[valid_metrics]
        
        # If we have run IDs in the index, filter parameter data
        if not performance_df.index.name and isinstance(performance_df.index, pd.Index) and len(performance_df.index) > 0:
            # Extract run numbers (assuming run_XXXX format)
            if isinstance(performance_df.index[0], str) and 'run_' in performance_df.index[0]:
                # Extract numeric part if index is like "run_0001"
                run_nums = []
                for idx in performance_df.index:
                    try:
                        run_nums.append(int(idx.split('_')[1]))
                    except (IndexError, ValueError):
                        run_nums.append(None)
                
                # Filter parameter data to matching runs
                valid_runs = [i for i, num in enumerate(run_nums) if num is not None and num < len(parameter_data)]
                if valid_runs:
                    performance_df = performance_df.iloc[valid_runs]
                    parameter_data = parameter_data.iloc[[run_nums[i] for i in valid_runs if run_nums[i] is not None]]
                    self.logger.info(f"Filtered to {len(valid_runs)} matching runs")
        
        # Ensure both datasets have the same number of rows
        min_rows = min(len(parameter_data), len(performance_df))
        if min_rows == 0:
            raise ValueError("No matching data between parameters and performance metrics")
        
        parameter_data = parameter_data.iloc[:min_rows]
        performance_df = performance_df.iloc[:min_rows]
        
        self.logger.info(f"After alignment: parameter_data shape: {parameter_data.shape}, performance_df shape: {performance_df.shape}")
        
        # Merge parameters and performance metrics
        merged_data = pd.concat([parameter_data.reset_index(drop=True), 
                                performance_df.reset_index()], axis=1)
        
        self.logger.info(f"Merged parameter and performance data with shape: {merged_data.shape}")
        
        # Fill missing values in numeric columns with means
        numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            missing_count = merged_data[col].isna().sum()
            if missing_count > 0:
                col_mean = merged_data[col].mean()
                merged_data[col].fillna(col_mean, inplace=True)
                self.logger.info(f"Filled {missing_count} missing values in {col} with mean: {col_mean}")
        
        # Final check for any remaining missing values
        missing_values = merged_data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            self.logger.warning(f"Still have columns with missing values: {columns_with_missing}")
            # For remaining missing values, use forward fill then backward fill
            merged_data = merged_data.ffill().bfill()
            
        # Final check - ensure we have enough data
        if merged_data.shape[0] < 5:  # Arbitrary minimum for RF model
            raise ValueError(f"Not enough data for training: only {merged_data.shape[0]} samples after preprocessing")
        
        self.logger.info(f"Final data shape after preprocessing: {merged_data.shape}")
        
        # Split into features and target
        X, y, _ = self._split_features_target(merged_data)
        
        # Save record of which columns are parameters
        self.param_cols = [col for col in X.columns if col.startswith('param_')]
        
        # Store parameter mins and maxs for optimization bounds
        for param in self.param_cols:
            self.param_mins[param] = X[param].min()
            self.param_maxs[param] = X[param].max()
        
        # Scale features with StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.X_scaler = scaler
        
        return X_scaled, y
    
    def _load_parameter_sets(self):
        """
        Load parameter sets from NetCDF file and convert to DataFrame.
        
        Returns:
            DataFrame with parameter values
        """
        # Load parameter sets from NetCDF
        try:
            ds = xr.open_dataset(self.parameter_sets_path)
            
            # Get all parameter variables (excluding dimensions and special variables)
            param_vars = [var for var in ds.variables if var not in ['hruId', 'runIndex', 'run', 'hru']]
            
            # Initialize empty DataFrame
            param_data = pd.DataFrame(index=ds.runIndex.values)
            
            # For each parameter, calculate the mean value across all HRUs for each run
            for param in param_vars:
                param_data[f"param_{param}"] = ds[param].mean(dim='hru').values
            
            return param_data
            
        except Exception as e:
            self.logger.error(f"Error loading parameter sets: {str(e)}")
            raise
                
    def _split_features_target(self, merged_data):
        """
        Split merged data into features (X) and target (y), excluding other metrics.
        
        Args:
            merged_data: Merged DataFrame with all data
        
        Returns:
            Tuple of (X, y) DataFrames
        """
        # Ensure the target metric exists
        if self.target_metric not in merged_data.columns:
            self.logger.error(f"Target metric {self.target_metric} not found in data columns: {merged_data.columns.tolist()}")
            raise ValueError(f"Target metric {self.target_metric} not found in data")
        
        # Identify parameters (columns starting with 'param_')
        param_cols = [col for col in merged_data.columns if col.startswith('param_')]
        
        # List of all potential metric prefixes and suffixes
        prefixes = ['Calib_', 'Eval_', '']
        suffixes = ['KGE', 'NSE', 'RMSE', 'PBIAS', 'MAE', 'r', 'alpha', 'beta']
        
        # Generate all possible metric column names
        metric_cols = [f"{prefix}{suffix}" for prefix in prefixes for suffix in suffixes]
        
        # Also exclude any ID columns
        excluded_cols = metric_cols + ['Run', 'Unnamed: 0', 'id', 'hru_id', 'gru_id', 'basin_id']
        
        # Create feature set (parameters only)
        X = merged_data[param_cols].copy()
        
        # For target, determine how to handle based on the metric
        if self.target_metric.endswith('KGE') or self.target_metric.endswith('NSE'):
            y = merged_data[self.target_metric]  # Higher is better
        elif self.target_metric.endswith('RMSE'):
            y = -merged_data[self.target_metric]  # Lower is better, so negate
        elif self.target_metric.endswith('PBIAS'):
            y = -merged_data[self.target_metric].abs()  # Minimize absolute bias
        else:
            y = merged_data[self.target_metric]
        
        return X, y, merged_data

    def _scale_features(self, X, merged_data=None):
        """
        Scale features using StandardScaler and handle categorical features.
        
        Args:
            X: Feature DataFrame (numeric features only)
            merged_data: Full merged DataFrame including categorical features
        
        Returns:
            Tuple of (scaled_X, scaler)
        """
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import numpy as np
        
        if X.empty or X.shape[0] == 0:
            raise ValueError("Cannot scale empty feature set")
        
        # First, handle the categorical (non-numeric) columns if present
        if hasattr(self, 'non_numeric_cols') and self.non_numeric_cols and merged_data is not None:
            self.logger.info(f"One-hot encoding {len(self.non_numeric_cols)} categorical features")
            
            # Create a one-hot encoder for categorical columns
            # We'll use pandas get_dummies for simplicity
            categorical_data = merged_data[self.non_numeric_cols].fillna('missing')
            
            # Convert all columns to string to ensure proper encoding
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].astype(str)
            
            # One-hot encode
            one_hot = pd.get_dummies(categorical_data, prefix=self.non_numeric_cols, drop_first=True)
            self.logger.info(f"One-hot encoding created {one_hot.shape[1]} binary features")
            
            # Scale numeric features
            scaler = StandardScaler()
            X_scaled_numeric = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Combine scaled numeric and one-hot encoded categorical features
            X_scaled = pd.concat([X_scaled_numeric, one_hot], axis=1)
            
            # Remember the one-hot encoded columns for optimization
            self.one_hot_cols = one_hot.columns.tolist()
        else:
            # Just scale numeric features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled, scaler

    def train_random_forest(self, X, y, spinup_size=1/3, test_size=1/3):
        """
        Train a random forest model using the prepared data with a three-way split.
        
        Args:
            X: Scaled feature data
            y: Target values
            spinup_size: Proportion of data to use for spinup
            test_size: Proportion of data to use for testing
        
        Returns:
            Trained model and evaluation metrics
        """
        self.logger.info("Training random forest model with three-way split")
        
        # Calculate sizes for a three-way split
        n_samples = len(X)
        n_spinup = int(n_samples * spinup_size)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_spinup - n_test
        
        # Create indices for each portion
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        # Split indices into three groups
        spinup_indices = indices[:n_spinup]
        train_indices = indices[n_spinup:n_spinup+n_train]
        test_indices = indices[n_spinup+n_train:]
        
        # Create the splits
        X_spinup, y_spinup = X.iloc[spinup_indices], y.iloc[spinup_indices]
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        self.logger.info(f"Data split: {n_spinup} spinup, {n_train} training, {n_test} testing samples")
        
        # Initialize and train the model (using only the training data)
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate the model on all three splits
        spinup_score = model.score(X_spinup, y_spinup)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.logger.info(f"Random forest model trained with R² scores:")
        self.logger.info(f"  Spinup: {spinup_score:.4f}")
        self.logger.info(f"  Training: {train_score:.4f}")
        self.logger.info(f"  Testing: {test_score:.4f}")
        
        # Calculate feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        
        # Log top 10 most important features
        self.logger.info("Top 10 most important features:")
        for feature, importance in importances[:10].items():
            self.logger.info(f"  {feature}: {importance:.4f}")
        
        # Save feature importances to CSV
        importances_df = importances.reset_index()
        importances_df.columns = ['Feature', 'Importance']
        importances_df.to_csv(self.emulation_output_dir / "feature_importances.csv", index=False)
        
        # Create feature importance plot
        plt.figure(figsize=(12, 8))
        importances[:20].plot(kind='bar')
        plt.title('Feature Importances (Top 20)')
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "feature_importances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store the model for later use
        self.model = model
        
        # Save the indices for each split to enable consistent evaluation
        self.spinup_indices = spinup_indices
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        return model, {
            'spinup_score': spinup_score, 
            'train_score': train_score, 
            'test_score': test_score
        }
    
    def optimize_parameters_global(self, fixed_attributes=None):
        """Use differential evolution for global optimization"""
        from scipy.optimize import differential_evolution
        
        # Define objective function
        def objective(params_array):
            input_data = {param: val for param, val in zip(self.param_cols, params_array)}
            X_pred = pd.DataFrame([input_data])
            
            # Ensure all columns from training are present
            for col in self.model.feature_names_in_:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            # Ensure correct column order
            X_pred = X_pred[self.model.feature_names_in_]
            
            # Return negative predicted score (for minimization)
            return -self.model.predict(X_pred)[0]
        
        # Define bounds
        bounds = [(self.param_mins[param], self.param_maxs[param]) for param in self.param_cols]
        
        # Run differential evolution
        result = differential_evolution(
            objective, 
            bounds,
            popsize=15,  # Increase population size
            mutation=(0.5, 1.0),  # Allow larger mutations
            recombination=0.7,
            strategy='best1bin',
            maxiter=100,  # More iterations
            polish=True   # Local optimization at the end
        )
        
        # Convert results to parameter dictionary
        optimized_params = {}
        for i, param in enumerate(self.param_cols):
            param_name = param.replace('param_', '')
            optimized_params[param_name] = result.x[i]
            self.logger.info(f"Optimized {param_name}: {result.x[i]:.4f}")
        
        # Save to CSV
        params_df = pd.DataFrame([optimized_params])
        params_df.to_csv(self.emulation_output_dir / "optimized_parameters.csv", index=False)
        
        return optimized_params, -result.fun

    def optimize_parameters(self, fixed_attributes=None):
        """
        Optimize parameters to maximize the predicted performance metric.
        
        Args:
            fixed_attributes: Dictionary of fixed attribute values to use (optional)
        
        Returns:
            Dictionary of optimized parameter values and predicted score
        """
        self.logger.info("Starting parameter optimization")
        
        # Check if model exists
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest() first.")
        
        # Define objective function (negative because we want to maximize)
        def objective(params_array):
            # Create a DataFrame with just the parameter values
            input_data = {}
            
            # Add parameters
            for i, param in enumerate(self.param_cols):
                input_data[param] = params_array[i]
            
            # Convert to DataFrame with the exact same columns as used in training
            X_pred = pd.DataFrame([input_data])
            
            # Make sure all columns from training are present in prediction
            for col in self.model.feature_names_in_:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            # Make sure columns are in the same order as training
            X_pred = X_pred[self.model.feature_names_in_]
            
            # Predict the metric and return negative (for minimization)
            return -self.model.predict(X_pred)[0]
        
        # Define bounds for parameters (from min/max values observed in training data)
        bounds = [(self.param_mins[param], self.param_maxs[param]) for param in self.param_cols]
        
        # Run optimization multiple times with different starting points
        best_score = float('-inf')
        best_params = None
        
        for i in range(self.n_optimization_runs):
            self.logger.info(f"Optimization run {i+1}/{self.n_optimization_runs}")
            
            # Generate random starting point
            x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
            
            # Run optimization
            result = minimize(
                objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Check if better than current best
            if -result.fun > best_score:
                best_score = -result.fun
                best_params = result.x
                if i > 0:  # Don't log for first run as it's always the "best" so far
                    self.logger.info(f"New best score: {best_score:.4f}")
        
        # Convert best parameters to dictionary
        optimized_params = {}
        for i, param in enumerate(self.param_cols):
            # Remove 'param_' prefix from parameter name
            param_name = param.replace('param_', '')
            optimized_params[param_name] = best_params[i]
            self.logger.info(f"Optimized {param_name}: {best_params[i]:.4f}")
        
        # Save optimized parameters to CSV
        params_df = pd.DataFrame([optimized_params])
        params_df.to_csv(self.emulation_output_dir / "optimized_parameters.csv", index=False)
        
        # Generate and save trialParams.nc file
        self._generate_trial_params_file(optimized_params)
        
        self.logger.info(f"Parameter optimization completed with predicted score: {best_score:.4f}")
        
        return optimized_params, best_score
        
    def _generate_trial_params_file(self, optimized_params):
        """
        Generate a trialParams.nc file with the optimized parameters.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
        
        Returns:
            Path: Path to the generated file
        """
        self.logger.info("Generating trialParams.nc file with optimized parameters")
        
        # Get attribute file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        try:
            # Read HRU information from the attributes file
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                
                # Create the trial parameters dataset
                output_ds = nc.Dataset(self.emulation_output_dir / "trialParams.nc", 'w', format='NETCDF4')
                
                # Add dimensions
                output_ds.createDimension('hru', len(hru_ids))
                
                # Add HRU ID variable
                hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Add parameter variables
                for param_name, param_value in optimized_params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Create variable and assign values
                    param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                    param_var[:] = param_array
                    param_var.long_name = f"Trial value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE RandomForestEmulator"
                output_ds.history = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.confluence_experiment_id = self.experiment_id
                
                # Close the file explicitly
                output_ds.close()
                
                # Copy to SUMMA settings directory for use in model runs
                summa_settings_dir = self.project_dir / "settings" / "SUMMA"
                target_path = summa_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                
                import shutil
                shutil.copy2(self.emulation_output_dir / "trialParams.nc", target_path)
                self.logger.info(f"Copied trial parameters to SUMMA settings directory: {target_path}")
                
                return target_path
                    
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_summa_with_optimal_parameters(self):
        """
        Run SUMMA with the optimized parameters.
        
        Returns:
            True if SUMMA run was successful, False otherwise
        """
        self.logger.info("Running SUMMA with optimized parameters")
        
        # Define paths
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE', 'summa.exe')
        filemanager_path = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        # Check if files exist
        if not summa_exe.exists():
            self.logger.error(f"SUMMA executable not found: {summa_exe}")
            return False
        
        if not filemanager_path.exists():
            self.logger.error(f"File manager not found: {filemanager_path}")
            return False
        
        # Create output directory
        output_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "SUMMA"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify fileManager to use the optimized output directory
        modified_filemanager = self._modify_filemanager(filemanager_path, output_dir)
        
        # If we couldn't modify the fileManager, use the original
        if modified_filemanager is None:
            self.logger.warning(f"Could not modify fileManager, using original: {filemanager_path}")
            modified_filemanager = filemanager_path
        
        # Run SUMMA
        summa_command = f"{summa_exe} -m {modified_filemanager}"
        self.logger.info(f"SUMMA command: {summa_command}")
        
        try:
            # Run SUMMA and capture output
            log_file = log_dir / "summa_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    summa_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"SUMMA run completed successfully, log saved to: {log_file}")
            
            # Run mizuRoute if needed
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                self._run_mizuroute(output_dir)
            
            # Run evaluation
            self._evaluate_summa_output()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False
    
    def _modify_filemanager(self, filemanager_path, output_dir):
        """
        Modify fileManager to use the optimized output directory.
        
        Args:
            filemanager_path: Path to original fileManager
            output_dir: New output directory
        
        Returns:
            Path to modified fileManager
        """
        try:
            with open(filemanager_path, 'r') as f:
                lines = f.readlines()
            
            # Create new fileManager in the emulation output directory
            new_filemanager = self.emulation_output_dir / "fileManager_rf_optimized.txt"
            
            with open(new_filemanager, 'w') as f:
                for line in lines:
                    if "outputPath" in line:
                        # Update output path
                        output_path_str = str(output_dir).replace('\\', '/')
                        f.write(f"outputPath           '{output_path_str}/' ! \n")
                    elif "outFilePrefix" in line:
                        # Update output file prefix
                        prefix_parts = line.split("'")
                        if len(prefix_parts) >= 3:
                            original_prefix = prefix_parts[1]
                            new_prefix = f"{self.experiment_id}_rf_optimized"
                            modified_line = line.replace(f"'{original_prefix}'", f"'{new_prefix}'")
                            f.write(modified_line)
                        else:
                            f.write(line)
                    else:
                        f.write(line)
            
            return new_filemanager
        
        except Exception as e:
            self.logger.error(f"Error modifying fileManager: {str(e)}")
            return None
    
    def _run_mizuroute(self, summa_output_dir):
        """
        Run mizuRoute with the SUMMA outputs.
        
        Args:
            summa_output_dir: Directory containing SUMMA outputs
        """
        self.logger.info("Running mizuRoute with SUMMA outputs")
        
        # Define paths
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Get mizuRoute settings
        mizu_settings_dir = self.project_dir / "settings" / "mizuRoute"
        mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        # Check if files exist
        if not mizu_exe.exists():
            self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
            return False
        
        if not mizu_control_file.exists():
            self.logger.error(f"mizuRoute control file not found: {mizu_control_file}")
            return False
        
        # Create output directory
        mizu_output_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "mizuRoute"
        mizu_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = mizu_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify mizuRoute control file
        modified_control = self._modify_mizuroute_control(
            mizu_control_file,
            summa_output_dir,
            mizu_settings_dir,
            mizu_output_dir
        )
        
        # If we couldn't modify the control file, use the original
        if modified_control is None:
            self.logger.warning(f"Could not modify mizuRoute control file, using original: {mizu_control_file}")
            modified_control = mizu_control_file
        
        # Run mizuRoute
        mizu_command = f"{mizu_exe} {modified_control}"
        self.logger.info(f"mizuRoute command: {mizu_command}")
        
        try:
            # Run mizuRoute and capture output
            log_file = log_dir / "mizuroute_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"mizuRoute run completed successfully, log saved to: {log_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False
    
    def _modify_mizuroute_control(self, control_file_path, summa_output_dir, mizu_settings_dir, mizu_output_dir):
        """
        Modify mizuRoute control file to use the optimized outputs.
        
        Args:
            control_file_path: Path to original control file
            summa_output_dir: Directory containing SUMMA outputs
            mizu_settings_dir: Directory containing mizuRoute settings
            mizu_output_dir: Directory for mizuRoute outputs
        
        Returns:
            Path to modified control file
        """
        try:
            with open(control_file_path, 'r') as f:
                lines = f.readlines()
            
            # Create new control file in the emulation output directory
            new_control_file = self.emulation_output_dir / "mizuroute_rf_optimized.control"
            
            with open(new_control_file, 'w') as f:
                for line in lines:
                    if '<input_dir>' in line:
                        # Update input directory (SUMMA output)
                        summa_dir_str = str(summa_output_dir).replace('\\', '/')
                        f.write(f"<input_dir>             {summa_dir_str}/    ! Folder that contains runoff data from SUMMA \n")
                    elif '<ancil_dir>' in line:
                        # Update ancillary directory
                        mizu_settings_str = str(mizu_settings_dir).replace('\\', '/')
                        f.write(f"<ancil_dir>             {mizu_settings_str}/    ! Folder that contains ancillary data \n")
                    elif '<output_dir>' in line:
                        # Update output directory
                        mizu_output_str = str(mizu_output_dir).replace('\\', '/')
                        f.write(f"<output_dir>            {mizu_output_str}/    ! Folder that will contain mizuRoute simulations \n")
                    elif '<case_name>' in line:
                        # Update case name
                        f.write(f"<case_name>             {self.experiment_id}_rf_optimized    ! Simulation case name \n")
                    elif '<fname_qsim>' in line:
                        # Update input file name
                        f.write(f"<fname_qsim>            {self.experiment_id}_rf_optimized_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                    else:
                        f.write(line)
            
            return new_control_file
        
        except Exception as e:
            self.logger.error(f"Error modifying mizuRoute control file: {str(e)}")
            return None
        
    def _evaluate_summa_output(self, output_dir=None):
        """
        Evaluate SUMMA and mizuRoute outputs by comparing to observed data.
        Only evaluates the calibration and evaluation periods defined in config.
        
        Args:
            output_dir: Optional directory where model output is stored. If None,
                    uses the default simulation directory.
        
        Returns:
            Dict: Dictionary with performance metrics, or None if evaluation failed
        """
        self.logger.info("Evaluating model outputs")
        
        # Define paths based on output_dir
        if output_dir is None:
            # Use default simulation directories
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # For distributed models, use mizuRoute output
                sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}" / "mizuRoute"
                sim_pattern = "*.nc"  # All netCDF files
            else:
                # For lumped models, use SUMMA output directly
                sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}" / "SUMMA"
                sim_pattern = f"{self.experiment_id}*.nc"  # SUMMA output files
        else:
            # Use specified output directory
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # For distributed models, use mizuRoute output
                sim_dir = output_dir / "simulations" / self.experiment_id / "mizuRoute"
                sim_pattern = "*.nc"  # All netCDF files
            else:
                # For lumped models, use SUMMA output directly
                sim_dir = output_dir / "simulations" / self.experiment_id / "SUMMA"
                sim_pattern = "*rf_iter*.nc"  # SUMMA iteration output files
        
        # Get observed data path
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default':
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        else:
            obs_path = Path(obs_path)
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Read observed data - first check column names
            with open(obs_path, 'r') as f:
                header_line = f.readline().strip()
            
            header_cols = header_line.split(',')
            self.logger.debug(f"Observed data columns: {header_cols}")
            
            # Identify date column
            date_col = next((col for col in header_cols if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data. Available columns: {header_cols}")
                return None
            
            # Identify flow column 
            flow_col = next((col for col in header_cols if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data. Available columns: {header_cols}")
                return None
            
            # Now read the CSV with known column names
            obs_df = pd.read_csv(obs_path, parse_dates=[date_col])
            
            # Set the date column as index
            obs_df.set_index(date_col, inplace=True)
            observed_flow = obs_df[flow_col]
                        
            # Extract simulated streamflow
            sim_files = list(sim_dir.glob(sim_pattern))

            if not sim_files:
                # If we didn't find any files matching the pattern, try specifically looking for timestep files
                timestep_pattern = "*timestep.nc"
                sim_files = list(sim_dir.glob(timestep_pattern))
                
                if sim_files:
                    self.logger.info(f"Found timestep files using pattern: {timestep_pattern}")
                else:
                    # If still no files, try a more general pattern
                    sim_files = list(sim_dir.glob("*.nc"))
                    if sim_files:
                        self.logger.info(f"Found some NetCDF files using general pattern: *.nc")
                    else:
                        self.logger.warning(f"No simulation files found in {sim_dir}")
                        return None
            
            if not sim_files:
                self.logger.warning(f"No simulation files found matching pattern {sim_pattern} in {sim_dir}")
                return None
            
            # Use the first file found
            sim_file = sim_files[1]
            self.logger.info(f"Using simulation file: {sim_file}")
            
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # From mizuRoute output
                with xr.open_dataset(sim_file) as ds:
                    # Find the index for the reach ID
                    sim_reach_id = self.config.get('SIM_REACH_ID')
                    if 'reachID' in ds.variables:
                        reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                        
                        if len(reach_indices) > 0:
                            reach_index = reach_indices[0]
                            
                            # Extract streamflow for this reach
                            # Try common variable names
                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                if var_name in ds.variables:
                                    sim_flow = ds[var_name].isel(seg=reach_index).to_pandas()
                                    break
                            else:
                                self.logger.error("Could not find streamflow variable in mizuRoute output")
                                return None
                        else:
                            self.logger.error(f"Reach ID {sim_reach_id} not found in mizuRoute output")
                            return None
                    else:
                        self.logger.error("No reachID variable found in mizuRoute output")
                        return None
            else:
                # From SUMMA output
                with xr.open_dataset(sim_file) as ds:
                    # Try to find streamflow variable
                    streamflow_var = None
                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            streamflow_var = var_name
                            break
                    
                    if streamflow_var is None:
                        self.logger.error("Could not find streamflow variable in SUMMA output")
                        return None
                    
                    # Extract the variable
                    if 'gru' in ds[streamflow_var].dims and ds.sizes['gru'] > 1:
                        # Sum across GRUs if multiple
                        runoff_series = ds[streamflow_var].sum(dim='gru').to_pandas()
                    else:
                        # Single GRU or no gru dimension
                        runoff_series = ds[streamflow_var].to_pandas()
                        # Handle case if it's a DataFrame
                        if isinstance(runoff_series, pd.DataFrame):
                            runoff_series = runoff_series.iloc[:, 0]
                    
                    # Get catchment area to convert from m/s to m³/s
                    catchment_area = self._get_catchment_area()
                    if catchment_area is None or catchment_area <= 0:
                        self.logger.warning("Missing or invalid catchment area. Using default 1.0 km²")
                        catchment_area = 1.0 * 1e6  # 1 km² in m²
                    
                    # Convert from m/s to m³/s
                    sim_flow = runoff_series * catchment_area
            
            # Align time index format for observed and simulated data
            obs_aligned = observed_flow.copy()
            sim_aligned = sim_flow.copy()
            
            # Filter data to calibration and evaluation periods if defined
            calib_start, calib_end = self.calibration_period
            eval_start, eval_end = self.evaluation_period
            
            # Create DataFrames to hold results for different periods
            calib_df = pd.DataFrame()
            eval_df = pd.DataFrame()
            
            # Calculate metrics for calibration period if defined
            metrics = {}
            
            if calib_start and calib_end:
                # Filter to calibration period
                calib_mask = (obs_aligned.index >= calib_start) & (obs_aligned.index <= calib_end)
                calib_obs = obs_aligned[calib_mask]
                
                # Find corresponding simulated values
                common_calib_idx = calib_obs.index.intersection(sim_aligned.index)
                if len(common_calib_idx) > 0:
                    calib_df['Observed'] = calib_obs.loc[common_calib_idx]
                    calib_df['Simulated'] = sim_aligned.loc[common_calib_idx]
                    
                    # Calculate metrics for calibration period
                    calib_metrics = self._calculate_streamflow_metrics(calib_df['Observed'], calib_df['Simulated'])
                    
                    self.logger.info(f"Calibration period ({calib_start} to {calib_end}) metrics:")
                    for metric, value in calib_metrics.items():
                        self.logger.info(f"  {metric}: {value:.4f}")
                        # Add to overall metrics dictionary
                        metrics[f"Calib_{metric}"] = value
                    
                    # Add without prefix for easier access
                    metrics.update(calib_metrics)
                else:
                    self.logger.warning("No common data points in calibration period")
            
            if eval_start and eval_end:
                # Filter to evaluation period
                eval_mask = (obs_aligned.index >= eval_start) & (obs_aligned.index <= eval_end)
                eval_obs = obs_aligned[eval_mask]
                
                # Find corresponding simulated values
                common_eval_idx = eval_obs.index.intersection(sim_aligned.index)
                if len(common_eval_idx) > 0:
                    eval_df['Observed'] = eval_obs.loc[common_eval_idx]
                    eval_df['Simulated'] = sim_aligned.loc[common_eval_idx]
                    
                    # Calculate metrics for evaluation period
                    eval_metrics = self._calculate_streamflow_metrics(eval_df['Observed'], eval_df['Simulated'])
                    
                    self.logger.info(f"Evaluation period ({eval_start} to {eval_end}) metrics:")
                    for metric, value in eval_metrics.items():
                        self.logger.info(f"  {metric}: {value:.4f}")
                        # Add to overall metrics dictionary
                        metrics[f"Eval_{metric}"] = value
                else:
                    self.logger.warning("No common data points in evaluation period")
            
            # Calculate metrics for the entire period if calibration and evaluation not specified
            if not metrics:
                # Get the time period where both datasets have data
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) == 0:
                    self.logger.error("No common time period between observed and simulated flows")
                    return None
                
                # Align the datasets
                obs_all = obs_aligned.loc[common_idx]
                sim_all = sim_aligned.loc[common_idx]
                
                # Calculate performance metrics
                metrics = self._calculate_streamflow_metrics(obs_all, sim_all)
                
                self.logger.info(f"Full period metrics:")
                for metric, value in metrics.items():
                    self.logger.info(f"  {metric}: {value:.4f}")
            
            # Create plots directory if we're using a specific output directory
            if output_dir is not None:
                plots_dir = output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Create visualization plots for this output
                self._create_streamflow_plots(obs_aligned, sim_aligned, calib_df, eval_df, metrics, plots_dir)
            
            # Save metrics to a file if using a specific output directory
            if output_dir is not None:
                metrics_file = output_dir / "metrics.csv"
                pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
                self.logger.info(f"Saved metrics to: {metrics_file}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model outputs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _create_streamflow_plots(self, obs_aligned, sim_aligned, calib_df, eval_df, metrics, plots_dir):
        """
        Create visualization plots for streamflow comparison.
        
        Args:
            obs_aligned: Observed streamflow series
            sim_aligned: Simulated streamflow series
            calib_df: DataFrame with calibration period data
            eval_df: DataFrame with evaluation period data
            metrics: Dictionary with performance metrics
            plots_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            import pandas as pd
            import numpy as np
            
            # 1. Create time series comparison plot
            plt.figure(figsize=(15, 8))
            
            # Add shaded backgrounds for calibration and evaluation periods
            if not calib_df.empty:
                calib_start = calib_df.index.min()
                calib_end = calib_df.index.max()
                plt.axvspan(calib_start, calib_end, color='lightblue', alpha=0.3, label='Calibration Period')
                
                # Plot calibration data
                plt.plot(calib_df.index, calib_df['Observed'], 'g-', linewidth=1.5, label='Observed (Calibration)')
                plt.plot(calib_df.index, calib_df['Simulated'], 'b-', linewidth=1.5, label='Simulated (Calibration)')
            
            if not eval_df.empty:
                eval_start = eval_df.index.min()
                eval_end = eval_df.index.max()
                plt.axvspan(eval_start, eval_end, color='lightsalmon', alpha=0.3, label='Evaluation Period')
                
                # Plot evaluation data
                plt.plot(eval_df.index, eval_df['Observed'], 'g--', linewidth=1.5, label='Observed (Evaluation)')
                plt.plot(eval_df.index, eval_df['Simulated'], 'b--', linewidth=1.5, label='Simulated (Evaluation)')
            
            # If no calibration or evaluation data, plot full period
            if calib_df.empty and eval_df.empty:
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) > 0:
                    plt.plot(common_idx, obs_aligned.loc[common_idx], 'g-', linewidth=1.5, label='Observed')
                    plt.plot(common_idx, sim_aligned.loc[common_idx], 'b-', linewidth=1.5, label='Simulated')
            
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Add metrics as text if available
            if metrics:
                # Format metrics text
                metrics_text = "Performance Metrics:\n"
                if any(k.startswith('Calib_') for k in metrics):
                    metrics_text += "Calibration:\n"
                    for k in ['KGE', 'NSE', 'RMSE']:
                        calib_key = f"Calib_{k}"
                        if calib_key in metrics:
                            metrics_text += f"  {k}: {metrics[calib_key]:.3f}\n"
                
                if any(k.startswith('Eval_') for k in metrics):
                    metrics_text += "Evaluation:\n"
                    for k in ['KGE', 'NSE', 'RMSE']:
                        eval_key = f"Eval_{k}"
                        if eval_key in metrics:
                            metrics_text += f"  {k}: {metrics[eval_key]:.3f}\n"
                
                if not any(k.startswith('Calib_') for k in metrics) and not any(k.startswith('Eval_') for k in metrics):
                    for k in ['KGE', 'NSE', 'RMSE']:
                        if k in metrics:
                            metrics_text += f"  {k}: {metrics[k]:.3f}\n"
                
                # Add text to plot
                plt.text(0.01, 0.99, metrics_text, transform=plt.gca().transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Combine data for scatter plot
            scatter_df = pd.DataFrame()
            if not calib_df.empty:
                calib_df_copy = calib_df.copy()
                calib_df_copy['Period'] = 'Calibration'
                scatter_df = pd.concat([scatter_df, calib_df_copy])
            
            if not eval_df.empty:
                eval_df_copy = eval_df.copy()
                eval_df_copy['Period'] = 'Evaluation'
                scatter_df = pd.concat([scatter_df, eval_df_copy])
            
            # If no calibration or evaluation data, use all data
            if scatter_df.empty:
                common_idx = obs_aligned.index.intersection(sim_aligned.index)
                if len(common_idx) > 0:
                    scatter_df = pd.DataFrame({
                        'Observed': obs_aligned.loc[common_idx],
                        'Simulated': sim_aligned.loc[common_idx],
                        'Period': 'All'
                    })
            
            # Create scatter plot with different colors for calibration and evaluation
            if 'Period' in scatter_df.columns:
                for period, color in [('Calibration', 'blue'), ('Evaluation', 'red'), ('All', 'purple')]:
                    period_df = scatter_df[scatter_df['Period'] == period]
                    if not period_df.empty:
                        plt.scatter(period_df['Observed'], period_df['Simulated'], 
                                    alpha=0.5, color=color, label=period)
            else:
                plt.scatter(scatter_df['Observed'], scatter_df['Simulated'], alpha=0.5)
            
            # Add 1:1 line
            max_val = scatter_df[['Observed', 'Simulated']].max().max()
            min_val = scatter_df[['Observed', 'Simulated']].min().min()
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='1:1 Line')
            
            # Add regression line
            from scipy import stats
            if len(scatter_df) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    scatter_df['Observed'], scatter_df['Simulated'])
                plt.plot([min_val, max_val], 
                        [slope * min_val + intercept, slope * max_val + intercept], 
                        'r-', label=f'Regression (r={r_value:.3f})')
            
            plt.xlabel('Observed Streamflow (m³/s)')
            plt.ylabel('Simulated Streamflow (m³/s)')
            plt.title('Observed vs. Simulated Streamflow (Scatter Plot)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Try to set equal aspect ratio
            try:
                plt.axis('equal')
            except:
                pass
            
            plt.tight_layout()
            plt.savefig(plots_dir / "streamflow_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Create flow duration curve
            plt.figure(figsize=(10, 8))
            
            # Find common time period for observed and simulated data
            common_idx = obs_aligned.index.intersection(sim_aligned.index)
            if len(common_idx) > 0:
                # Extract aligned data
                obs_common = obs_aligned.loc[common_idx]
                sim_common = sim_aligned.loc[common_idx]
                
                # Calculate flow duration curves
                obs_sorted = np.sort(obs_common.values)[::-1]  # Descending order
                sim_sorted = np.sort(sim_common.values)[::-1]  # Descending order
                
                # Calculate exceedance probabilities
                n_obs = len(obs_sorted)
                n_sim = len(sim_sorted)
                exc_prob_obs = np.arange(1, n_obs + 1) / (n_obs + 1)
                exc_prob_sim = np.arange(1, n_sim + 1) / (n_sim + 1)
                
                # Plot FDCs
                plt.plot(exc_prob_obs, obs_sorted, 'g-', linewidth=1.5, label='Observed')
                plt.plot(exc_prob_sim, sim_sorted, 'b-', linewidth=1.5, label='Simulated')
                
                plt.xlabel('Exceedance Probability')
                plt.ylabel('Streamflow (m³/s)')
                plt.title('Flow Duration Curves')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.yscale('log')  # Log scale for better visualization
                
                plt.tight_layout()
                plt.savefig(plots_dir / "flow_duration_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Created streamflow plots in {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating streamflow plots: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _plot_flow_duration_curve(self, calib_df, eval_df, obs_all, sim_all):
        """
        Create flow duration curve visualization.
        
        Args:
            calib_df: DataFrame with observed and simulated data for calibration period
            eval_df: DataFrame with observed and simulated data for evaluation period
            obs_all: Series with all observed data
            sim_all: Series with all simulated data
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 8))
        
        # Function to calculate and plot FDC
        def plot_fdc(obs, sim, label_prefix, color_obs, color_sim, linestyle='-'):
            if obs.empty or sim.empty:
                return
                
            # Sort flows in descending order
            obs_sorted = obs.sort_values(ascending=False)
            sim_sorted = sim.sort_values(ascending=False)
            
            # Calculate exceedance probability
            obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
            sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
            
            # Plot the FDCs
            plt.plot(obs_exceed, obs_sorted, color=color_obs, linestyle=linestyle, 
                     label=f'Observed ({label_prefix})')
            plt.plot(sim_exceed, sim_sorted, color=color_sim, linestyle=linestyle, 
                     label=f'Simulated ({label_prefix})')
        
        # Plot FDCs for calibration and evaluation periods if available
        if not calib_df.empty:
            plot_fdc(calib_df['Observed'], calib_df['Simulated'], 'Calibration', 'g', 'b')
        
        if not eval_df.empty:
            plot_fdc(eval_df['Observed'], eval_df['Simulated'], 'Evaluation', 'g', 'b', linestyle='--')
        
        # If no calibration or evaluation data, use all data
        if calib_df.empty and eval_df.empty:
            common_idx = obs_all.index.intersection(sim_all.index)
            if len(common_idx) > 0:
                plot_fdc(obs_all.loc[common_idx], sim_all.loc[common_idx], 'All', 'g', 'b')
        
        # Set log scale for y-axis
        plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Exceedance Probability')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Flow Duration Curves')
        plt.grid(True, which='both', alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "flow_duration_curve_rf_optimized.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_streamflow_metrics(self, observed, simulated):
        """
        Calculate performance metrics for streamflow comparison.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        
        Returns:
            Dictionary of performance metrics
        """
        # Make sure both series are numeric
        observed = pd.to_numeric(observed, errors='coerce')
        simulated = pd.to_numeric(simulated, errors='coerce')
        
        # Drop NaN values in either series
        valid = ~(observed.isna() | simulated.isna())
        observed = observed[valid]
        simulated = simulated[valid]
        
        if len(observed) == 0:
            self.logger.error("No valid data points for metric calculation")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
        
        # Nash-Sutcliffe Efficiency (NSE)
        mean_obs = observed.mean()
        nse_numerator = ((observed - simulated) ** 2).sum()
        nse_denominator = ((observed - mean_obs) ** 2).sum()
        nse = 1 - (nse_numerator / nse_denominator) if nse_denominator > 0 else np.nan
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(((observed - simulated) ** 2).mean())
        
        # Percent Bias (PBIAS)
        pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
        
        # Kling-Gupta Efficiency (KGE)
        r = observed.corr(simulated)  # Correlation coefficient
        alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan  # Relative variability
        beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan  # Bias ratio
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
        
        # Mean Absolute Error (MAE)
        mae = (observed - simulated).abs().mean()
        
        return {
            'KGE': kge,
            'NSE': nse,
            'RMSE': rmse,
            'PBIAS': pbias,
            'MAE': mae,
            'r': r,
            'alpha': alpha,
            'beta': beta
        }
    
    def _get_catchment_area(self):
        """
        Get catchment area from basin shapefile.
        
        Returns:
            float: Catchment area in square meters, or None if not found
        """
        try:
            import geopandas as gpd
            
            # First try to get the basin shapefile
            river_basins_path = self.config.get('RIVER_BASINS_PATH')
            if river_basins_path == 'default':
                river_basins_path = self.project_dir / "shapefiles" / "river_basins"
            else:
                river_basins_path = Path(river_basins_path)
            
            river_basins_name = self.config.get('RIVER_BASINS_NAME')
            if river_basins_name == 'default':
                river_basins_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"
            
            basin_shapefile = river_basins_path / river_basins_name
            
            # If basin shapefile doesn't exist, try the catchment shapefile
            if not basin_shapefile.exists():
                self.logger.warning(f"River basin shapefile not found: {basin_shapefile}")
                self.logger.info("Trying to use catchment shapefile instead")
                
                catchment_path = self.config.get('CATCHMENT_PATH')
                if catchment_path == 'default':
                    catchment_path = self.project_dir / "shapefiles" / "catchment"
                else:
                    catchment_path = Path(catchment_path)
                
                catchment_name = self.config.get('CATCHMENT_SHP_NAME')
                if catchment_name == 'default':
                    catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
                    
                basin_shapefile = catchment_path / catchment_name
                
                if not basin_shapefile.exists():
                    self.logger.warning(f"Catchment shapefile not found: {basin_shapefile}")
                    return None
            
            # Open shapefile
            gdf = gpd.read_file(basin_shapefile)
            
            # Try to get area from attributes first
            area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
            
            if area_col in gdf.columns:
                # Sum all basin areas (in case of multiple basins)
                total_area = gdf[area_col].sum()
                
                # Check if the area seems reasonable
                if total_area <= 0 or total_area > 1e12:  # Suspicious if > 1 million km²
                    self.logger.warning(f"Area from attribute {area_col} seems unrealistic: {total_area} m². Calculating geometrically.")
                else:
                    self.logger.info(f"Found catchment area from attribute: {total_area} m²")
                    return total_area
            
            # If area column not found or value is suspicious, calculate area from geometry
            self.logger.info("Calculating catchment area from geometry")
            
            # Make sure CRS is in a projected system for accurate area calculation
            if gdf.crs is None:
                self.logger.warning("Shapefile has no CRS information, assuming WGS84")
                gdf.crs = "EPSG:4326"
            
            # If geographic (lat/lon), reproject to a UTM zone for accurate area calculation
            if gdf.crs.is_geographic:
                # Calculate centroid to determine appropriate UTM zone
                centroid = gdf.dissolve().centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                
                # Determine UTM zone
                utm_zone = int(((lon + 180) / 6) % 60) + 1
                north_south = 'north' if lat >= 0 else 'south'
                
                utm_crs = f"+proj=utm +zone={utm_zone} +{north_south} +datum=WGS84 +units=m +no_defs"
                self.logger.info(f"Reprojecting from {gdf.crs} to UTM zone {utm_zone} ({utm_crs})")
                
                # Reproject
                gdf = gdf.to_crs(utm_crs)
            
            # Calculate area in m²
            gdf['calc_area'] = gdf.geometry.area
            total_area = gdf['calc_area'].sum()
            
            self.logger.info(f"Calculated catchment area from geometry: {total_area} m²")
            
            # Double check if area seems reasonable
            if total_area <= 0:
                self.logger.error(f"Calculated area is non-positive: {total_area} m²")
                return None
            
            if total_area > 1e12:  # > 1 million km²
                self.logger.warning(f"Calculated area seems very large: {total_area} m² ({total_area/1e6:.2f} km²). Check units.")
            
            return total_area
            
        except Exception as e:
            self.logger.error(f"Error calculating catchment area: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _generate_new_ensemble(self, optimized_params, iteration):
        """
        Generate a new parameter ensemble centered around the optimized parameters
        with progressively narrower sampling distribution.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration: Current iteration number (used to narrow sampling range)
        
        Returns:
            Dictionary of parameter sets for the new ensemble
        """
        import numpy as np
        import pandas as pd
        import netCDF4 as nc
        
        self.logger.info("Generating new parameter ensemble centered around optimized parameters")
        
        # Get attributes file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        # Determine how much to narrow the sampling range based on iteration
        # Start with 50% and decrease by 10% each iteration (50%, 40%, 30%, etc.)
        # but never go below 10%
        range_reduction = max(0.8 - (0.05 * iteration), 0.1)
        self.logger.info(f"Using sampling range reduction factor of {range_reduction:.2f}")
        
        try:
            # Read HRU information
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                num_hru = len(hru_ids)
                
                # Create directory for the new ensemble
                ensemble_dir = self.emulation_output_dir / f"ensemble_iteration_{iteration+1}"
                ensemble_dir.mkdir(parents=True, exist_ok=True)
                
                # Number of parameter sets to generate
                num_sets = self.config.get('EMULATION_ENSEMBLE_SIZE', 100)
                
                # Create parameter sets
                param_sets = []
                for i in range(num_sets):
                    if i == 0:
                        # First set is optimized parameters
                        param_set = optimized_params.copy()
                    elif i < 0.2 * num_sets:
                        # 20% of sets with wide exploration (random)
                        param_set = {}
                        for param_name, _ in optimized_params.items():
                            param_col = f"param_{param_name}"
                            min_val = self.param_mins.get(param_col, 0)
                            max_val = self.param_maxs.get(param_col, 1)
                            param_set[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        # Rest with focused exploration around optimal
                        param_set = {}
                        for param_name, optimal_value in optimized_params.items():
                            param_col = f"param_{param_name}"
                            min_val = self.param_mins.get(param_col, 0)
                            max_val = self.param_maxs.get(param_col, 1)
                            
                            param_range = max_val - min_val
                            narrowed_range = param_range * range_reduction
                            
                            new_min = max(min_val, optimal_value - (narrowed_range / 2))
                            new_max = min(max_val, optimal_value + (narrowed_range / 2))
                            
                            param_set[param_name] = np.random.uniform(new_min, new_max)
                    
                    param_sets.append(param_set)
                
                # Create parameter files for each set
                for i, param_set in enumerate(param_sets):
                    # Create a directory for this run
                    run_dir = ensemble_dir / f"run_{i:04d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create settings directory
                    run_settings_dir = run_dir / "settings" / "SUMMA"
                    run_settings_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create trial parameter file
                    trial_param_path = run_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                    
                    # Create NetCDF file
                    with nc.Dataset(trial_param_path, 'w', format='NETCDF4') as tp_ds:
                        # Define dimensions
                        tp_ds.createDimension('hru', num_hru)
                        
                        # Create hruId variable
                        hru_id_var = tp_ds.createVariable('hruId', 'i4', ('hru',))
                        hru_id_var[:] = hru_ids
                        hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                        hru_id_var.units = '-'
                        
                        # Add each parameter with its values for this run
                        for param_name, value in param_set.items():
                            param_var = tp_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                            # Set all HRUs to the same parameter value
                            param_var[:] = np.full(num_hru, value)
                            param_var.long_name = f"Trial value for {param_name}"
                            param_var.units = "N/A"
                        
                        # Add global attributes
                        tp_ds.description = f"SUMMA Trial Parameter file for iteration {iteration+1}, run {i:04d}"
                        tp_ds.history = f"Created on {pd.Timestamp.now().isoformat()} by CONFLUENCE RandomForestEmulator"
                        tp_ds.confluence_experiment_id = self.experiment_id
                        tp_ds.emulation_iteration = iteration + 1
                    
                    # Copy necessary settings files
                    self._copy_static_settings_files(run_settings_dir)
                    
                    # Create and modify the file manager
                    self._create_run_file_manager(i, run_dir, run_settings_dir, iteration+1)
                
                # Also save a consolidated parameter file with all parameter sets
                consolidated_file = ensemble_dir / f"parameter_sets_iteration_{iteration+1}.csv"
                pd.DataFrame(param_sets).to_csv(consolidated_file, index=False)
                
                self.logger.info(f"Generated {num_sets} parameter sets for iteration {iteration+1}")
                return param_sets
                
        except Exception as e:
            self.logger.error(f"Error generating new parameter ensemble: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _copy_static_settings_files(self, run_settings_dir):
        """
        Copy static SUMMA settings files to the run directory.
        
        Args:
            run_settings_dir: Target settings directory for this run
        """
        # Files to copy
        import shutil
        from pathlib import Path
        
        source_dir = self.project_dir / "settings" / "SUMMA"
        static_files = [
            'modelDecisions.txt',
            'outputControl.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL',
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL',
            'attributes.nc',
            'coldState.nc',
            'forcingFileList.txt'
        ]
        
        # Copy each file if it exists
        for file_name in static_files:
            source_path = source_dir / file_name
            if source_path.exists():
                dest_path = run_settings_dir / file_name
                try:
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")

    def _create_run_file_manager(self, run_idx, run_dir, run_settings_dir, iteration):
        """
        Create a modified fileManager for a specific run.
        
        Args:
            run_idx: Index of the current run
            run_dir: Base directory for this run
            run_settings_dir: Settings directory for this run
            iteration: Current iteration number
        """
        import os
        
        source_file_manager = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not source_file_manager.exists():
            self.logger.warning(f"Original fileManager not found at {source_file_manager}")
            return
        
        # Read the original file manager
        with open(source_file_manager, 'r') as f:
            fm_lines = f.readlines()
        
        # Path to the new file manager
        new_fm_path = run_settings_dir / os.path.basename(source_file_manager)
        
        # Modify relevant lines for this run
        modified_lines = []
        for line in fm_lines:
            if "outFilePrefix" in line:
                # Update output file prefix to include iteration and run index
                prefix_parts = line.split("'")
                if len(prefix_parts) >= 3:
                    original_prefix = prefix_parts[1]
                    new_prefix = f"{self.experiment_id}_iter{iteration}_run{run_idx:04d}"
                    modified_line = line.replace(f"'{original_prefix}'", f"'{new_prefix}'")
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)  # Keep unchanged if format unexpected
            elif "outputPath" in line:
                # Update output path to use this run's directory
                output_path = run_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                output_path_str = str(output_path).replace('\\', '/')  # Ensure forward slashes for SUMMA
                modified_line = f"outputPath           '{output_path_str}/' ! \n"
                modified_lines.append(modified_line)
            elif "settingsPath" in line:
                # Update settings path to point to the run-specific settings directory
                settings_path_str = str(run_settings_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                modified_line = f"settingsPath         '{settings_path_str}/'\n"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)  # Keep other lines unchanged
        
        # Write the modified file manager
        with open(new_fm_path, 'w') as f:
            f.writelines(modified_lines)
        
        return new_fm_path

    def _run_ensemble_simulations(self, param_sets, iteration):
        """
        Run SUMMA and MizuRoute for each parameter set in the ensemble.
        
        Args:
            param_sets: List of parameter set dictionaries
            iteration: Current iteration number
        
        Returns:
            Dictionary with simulation results and performance metrics
        """
        import concurrent.futures
        import subprocess
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        self.logger.info(f"Running ensemble simulations for iteration {iteration+1}")
        
        # Get SUMMA executable path
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Get ensemble directory
        ensemble_dir = self.emulation_output_dir / f"ensemble_iteration_{iteration+1}"
        
        # Get mizuRoute settings if needed
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        if not is_lumped:
            mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
            if mizu_path == 'default':
                mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
            else:
                mizu_path = Path(mizu_path)
            
            mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Check if we should run in parallel
        run_parallel = self.config.get('EMULATION_PARALLEL_ENSEMBLE', False)
        max_workers = self.config.get('EMULATION_MAX_WORKERS', min(16, len(param_sets)))
        
        # Function to run a single parameter set
        def run_one_set(run_idx):
            run_name = f"run_{run_idx:04d}"
            run_dir = ensemble_dir / run_name
            
            # Get fileManager path
            filemanager_path = run_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
            
            if not filemanager_path.exists():
                self.logger.warning(f"FileManager not found for {run_name}: {filemanager_path}")
                return {
                    'run_id': run_idx,
                    'success': False,
                    'error': "FileManager not found"
                }
            
            # Ensure output directories exist
            summa_output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            summa_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = summa_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create results directory
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run SUMMA
            summa_command = f"{summa_exe} -m {filemanager_path}"
            summa_log_file = log_dir / f"{run_name}_summa.log"
            
            try:
                with open(summa_log_file, 'w') as f:
                    subprocess.run(summa_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                
                self.logger.debug(f"SUMMA completed for {run_name}")
                summa_success = True
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"SUMMA failed for {run_name}: {str(e)}")
                summa_success = False
            
            # Run mizuRoute if needed and SUMMA succeeded
            mizu_success = True
            if not is_lumped and summa_success:
                mizu_output_dir = run_dir / "simulations" / self.experiment_id / "mizuRoute"
                mizu_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup mizuRoute
                mizu_settings_dir = run_dir / "settings" / "mizuRoute"
                if not mizu_settings_dir.exists():
                    mizu_settings_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy and modify mizuRoute control file
                    source_control_file = self.project_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                    
                    if source_control_file.exists():
                        mizu_control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
                        
                        # Copy and modify the control file
                        with open(source_control_file, 'r') as f:
                            lines = f.readlines()
                        
                        with open(mizu_control_file, 'w') as f:
                            for line in lines:
                                if '<input_dir>' in line:
                                    # Update input directory (SUMMA output)
                                    f.write(f"<input_dir>             {summa_output_dir}/    ! Folder with runoff data from SUMMA \n")
                                elif '<ancil_dir>' in line:
                                    # Update ancillary directory
                                    source_ancil_dir = self.project_dir / "settings" / "mizuRoute"
                                    f.write(f"<ancil_dir>             {source_ancil_dir}/    ! Folder with ancillary data \n")
                                elif '<output_dir>' in line:
                                    # Update output directory
                                    f.write(f"<output_dir>            {mizu_output_dir}/    ! Folder for mizuRoute simulations \n")
                                elif '<case_name>' in line:
                                    # Update case name
                                    f.write(f"<case_name>             {self.experiment_id}_iter{iteration+1}_run{run_idx:04d}    ! Simulation case name \n")
                                elif '<fname_qsim>' in line:
                                    # Update input file name
                                    f.write(f"<fname_qsim>            {self.experiment_id}_iter{iteration+1}_run{run_idx:04d}_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                                else:
                                    f.write(line)
                        
                        # Run mizuRoute
                        mizu_command = f"{mizu_exe} {mizu_control_file}"
                        mizu_log_file = log_dir / f"{run_name}_mizuroute.log"
                        
                        try:
                            with open(mizu_log_file, 'w') as f:
                                subprocess.run(mizu_command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
                            
                            self.logger.debug(f"mizuRoute completed for {run_name}")
                            mizu_success = True
                        except subprocess.CalledProcessError as e:
                            self.logger.warning(f"mizuRoute failed for {run_name}: {str(e)}")
                            mizu_success = False
                    else:
                        self.logger.warning(f"mizuRoute control file not found: {source_control_file}")
                        mizu_success = False
                else:
                    self.logger.warning(f"mizuRoute settings directory not found for {run_name}")
                    mizu_success = False
            
            # Extract streamflow data
            streamflow_file = None
            if summa_success:
                streamflow_file = self._extract_run_streamflow(run_dir, run_name)
            
            # Calculate performance metrics
            metrics = None
            if streamflow_file and Path(streamflow_file).exists():
                metrics = self._calculate_performance_for_run(streamflow_file)
            
            # Return results
            return {
                'run_id': run_idx,
                'success': summa_success and mizu_success,
                'summa_success': summa_success,
                'mizu_success': mizu_success if not is_lumped else None,
                'streamflow_file': str(streamflow_file) if streamflow_file else None,
                'metrics': metrics,
                'parameters': param_sets[run_idx]
            }
        
        # Run simulations (in parallel or sequentially)
        results = []
        
        if run_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_one_set, i): i for i in range(len(param_sets))}
                
                for future in concurrent.futures.as_completed(futures):
                    run_idx = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Log progress periodically
                        if len(results) % 10 == 0 or len(results) == len(param_sets):
                            success_count = sum(1 for r in results if r['success'])
                            self.logger.info(f"Completed {len(results)}/{len(param_sets)} runs ({success_count} successful)")
                    except Exception as e:
                        self.logger.error(f"Error running set {run_idx}: {str(e)}")
                        results.append({
                            'run_id': run_idx,
                            'success': False,
                            'error': str(e),
                            'parameters': param_sets[run_idx]
                        })
        else:
            for i in range(len(param_sets)):
                try:
                    result = run_one_set(i)
                    results.append(result)
                    
                    # Log progress periodically
                    if (i+1) % 10 == 0 or (i+1) == len(param_sets):
                        success_count = sum(1 for r in results if r['success'])
                        self.logger.info(f"Completed {i+1}/{len(param_sets)} runs ({success_count} successful)")
                except Exception as e:
                    self.logger.error(f"Error running set {i}: {str(e)}")
                    results.append({
                        'run_id': i,
                        'success': False,
                        'error': str(e),
                        'parameters': param_sets[i]
                    })
        
        # Create a summary of results
        success_count = sum(1 for r in results if r['success'])
        self.logger.info(f"Ensemble simulation completed: {success_count}/{len(param_sets)} successful runs")
        
        # Save a summary CSV with parameters and performance metrics
        summary_data = []
        for result in results:
            result_data = {'run_id': result['run_id'], 'success': result['success']}
            
            # Add parameters
            for param_name, value in result['parameters'].items():
                result_data[f"param_{param_name}"] = value
            
            # Add metrics if available
            if result['metrics']:
                for metric_name, value in result['metrics'].items():
                    result_data[metric_name] = value
            
            summary_data.append(result_data)
        
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_file = ensemble_dir / f"ensemble_results_iteration_{iteration+1}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Ensemble results summary saved to {summary_file}")
        
        return results

    def _extract_run_streamflow(self, run_dir, run_name):
        """
        Extract streamflow results from model output for a specific run.
        
        For distributed domains, extracts from MizuRoute output.
        For lumped domains, extracts directly from SUMMA timestep output.
        
        Args:
            run_dir: Path to the run directory
            run_name: Name of the run (e.g., 'run_0001')
            
        Returns:
            Path: Path to the extracted streamflow CSV file
        """
        self.logger.info(f"Extracting streamflow results for {run_name}")
        
        import pandas as pd
        import xarray as xr
        import numpy as np
        from pathlib import Path
        
        # Create results directory
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output CSV path
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        
        # Check if domain is lumped
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        # Get catchment area for conversion
        catchment_area = self._get_catchment_area()
        if catchment_area is None or catchment_area <= 0:
            self.logger.warning(f"Invalid catchment area: {catchment_area}, using 1.0 km² as default")
            catchment_area = 1.0 * 1e6  # Default 1 km² in m²
        
        self.logger.info(f"Using catchment area of {catchment_area:.2f} m² for flow conversion")
        
        if is_lumped:
            # For lumped domains, extract directly from SUMMA output
            experiment_id = self.experiment_id
            summa_output_dir = run_dir / "simulations" / experiment_id / "SUMMA"
            
            # Look for all possible SUMMA output files
            timestep_files = list(summa_output_dir.glob("*timestep.nc"))
            
            if not timestep_files:
                self.logger.warning(f"No SUMMA output files found for {run_name} in {summa_output_dir}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Use the first timestep file found
            timestep_file = timestep_files[0]
            self.logger.info(f"Using SUMMA output file: {timestep_file}")
            
            try:
                # Open the NetCDF file
                with xr.open_dataset(timestep_file) as ds:
                    # Check for averageRoutedRunoff or other possible variables
                    runoff_var = None
                    for var_name in ['averageRoutedRunoff', 'outflow', 'basRunoff', 'totalRunoff']:
                        if var_name in ds.variables:
                            runoff_var = var_name
                            break
                    
                    if runoff_var is None:
                        self.logger.warning(f"No suitable runoff variable found in {timestep_file}")
                        return self._create_empty_streamflow_file(run_dir, run_name)
                    
                    self.logger.info(f"Using {runoff_var} variable from SUMMA output")
                    
                    # Extract the runoff variable
                    runoff_data = ds[runoff_var]
                    
                    # Convert to pandas series or dataframe
                    if 'gru' in runoff_data.dims:
                        # If multiple GRUs, sum them up
                        if ds.dims['gru'] > 1:
                            self.logger.info(f"Summing runoff across {ds.dims['gru']} GRUs")
                            runoff_series = runoff_data.sum(dim='gru').to_pandas()
                        else:
                            # Single GRU case
                            runoff_series = runoff_data.to_pandas()
                            if isinstance(runoff_series, pd.DataFrame):
                                runoff_series = runoff_series.iloc[:, 0]
                    else:
                        # Handle case without a gru dimension
                        runoff_series = runoff_data.to_pandas()
                        if isinstance(runoff_series, pd.DataFrame):
                            runoff_series = runoff_series.iloc[:, 0]
                    
                    # Convert from m/s to m³/s by multiplying by catchment area
                    streamflow = runoff_series * catchment_area
                    
                    # Create dataframe and save to CSV
                    result_df = pd.DataFrame({'time': streamflow.index, 'streamflow': streamflow.values})
                    result_df.to_csv(output_csv, index=False)
                    
                    self.logger.info(f"Extracted streamflow from SUMMA output for lumped domain, saved to {output_csv}")
                    return output_csv
                    
            except Exception as e:
                self.logger.error(f"Error extracting streamflow from SUMMA output: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return self._create_empty_streamflow_file(run_dir, run_name)
        else:
            # For distributed domains, extract from MizuRoute output
            experiment_id = self.experiment_id
            mizuroute_output_dir = run_dir / "simulations" / experiment_id / "mizuRoute"
            
            # Try different pattern matching to find output files
            output_files = []
            patterns = [
                f"{experiment_id}*.nc",             # Try just experiment ID
                "*.nc"                              # Try any netCDF file as last resort
            ]
            
            for pattern in patterns:
                output_files = list(mizuroute_output_dir.glob(pattern))
                if output_files:
                    self.logger.info(f"Found MizuRoute output files with pattern: {pattern}")
                    break
            
            if not output_files:
                self.logger.warning(f"No MizuRoute output files found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)
            
            # Target reach ID
            sim_reach_id = self.config.get('SIM_REACH_ID')
            
            # Process each output file
            all_streamflow = []
            
            for output_file in output_files:
                try:
                    # Open the NetCDF file
                    with xr.open_dataset(output_file) as ds:
                        # Check if the reach ID exists
                        if 'reachID' in ds.variables:
                            # Find index for the specified reach ID
                            reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                            
                            if len(reach_indices) > 0:
                                # Get the index
                                reach_index = reach_indices[0]
                                
                                # Extract the time series for this reach
                                # Look for common streamflow variable names
                                for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                    if var_name in ds.variables:
                                        streamflow = ds[var_name].isel(seg=reach_index).to_pandas()
                                        
                                        # Reset index to use time as a column
                                        streamflow = streamflow.reset_index()
                                        
                                        # Rename columns for clarity
                                        streamflow = streamflow.rename(columns={var_name: 'streamflow'})
                                        
                                        all_streamflow.append(streamflow)
                                        self.logger.info(f"Extracted streamflow from variable {var_name}")
                                        break
                                
                                self.logger.info(f"Extracted streamflow for reach ID {sim_reach_id} from {output_file.name}")
                            else:
                                self.logger.warning(f"Reach ID {sim_reach_id} not found in {output_file.name}")
                        else:
                            self.logger.warning(f"No reachID variable found in {output_file.name}")
                            
                except Exception as e:
                    self.logger.error(f"Error processing {output_file.name}: {str(e)}")
            
            # Combine all streamflow data
            if all_streamflow:
                combined_streamflow = pd.concat(all_streamflow)
                combined_streamflow = combined_streamflow.sort_values('time')
                
                # Remove duplicates if any
                combined_streamflow = combined_streamflow.drop_duplicates(subset='time')
                
                # Save to CSV
                combined_streamflow.to_csv(output_csv, index=False)
                
                self.logger.info(f"Saved combined streamflow results to {output_csv}")
                
                return output_csv
            else:
                self.logger.warning(f"No streamflow data found for {run_name}")
                return self._create_empty_streamflow_file(run_dir, run_name)

    def _create_empty_streamflow_file(self, run_dir, run_name):
        """
        Create an empty streamflow file to ensure existence for later analysis.
        
        Args:
            run_dir: Path to the run directory
            run_name: Name of the run
            
        Returns:
            Path: Path to the created file
        """
        import pandas as pd
        
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create an empty streamflow file with appropriate columns
        empty_df = pd.DataFrame(columns=['time', 'streamflow'])
        output_csv = results_dir / f"{run_name}_streamflow.csv"
        empty_df.to_csv(output_csv, index=False)
        
        self.logger.warning(f"Created empty streamflow file at {output_csv}")
        return output_csv

    def _get_catchment_area(self):
        """
        Get catchment area from basin shapefile.
        
        Returns:
            float: Catchment area in square meters, or None if not found
        """
        try:
            import geopandas as gpd
            
            # First try to get the basin shapefile
            river_basins_path = self.config.get('RIVER_BASINS_PATH')
            if river_basins_path == 'default':
                river_basins_path = self.project_dir / "shapefiles" / "river_basins"
            else:
                river_basins_path = Path(river_basins_path)
            
            river_basins_name = self.config.get('RIVER_BASINS_NAME')
            if river_basins_name == 'default':
                river_basins_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"
            
            basin_shapefile = river_basins_path / river_basins_name
            
            # If basin shapefile doesn't exist, try the catchment shapefile
            if not basin_shapefile.exists():
                self.logger.warning(f"River basin shapefile not found: {basin_shapefile}")
                self.logger.info("Trying to use catchment shapefile instead")
                
                catchment_path = self.config.get('CATCHMENT_PATH')
                if catchment_path == 'default':
                    catchment_path = self.project_dir / "shapefiles" / "catchment"
                else:
                    catchment_path = Path(catchment_path)
                
                catchment_name = self.config.get('CATCHMENT_SHP_NAME')
                if catchment_name == 'default':
                    catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
                    
                basin_shapefile = catchment_path / catchment_name
                
                if not basin_shapefile.exists():
                    self.logger.warning(f"Catchment shapefile not found: {basin_shapefile}")
                    return None
            
            # Open shapefile
            gdf = gpd.read_file(basin_shapefile)
            
            # Try to get area from attributes first
            area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
            
            if area_col in gdf.columns:
                # Sum all basin areas (in case of multiple basins)
                total_area = gdf[area_col].sum()
                
                # Check if the area seems reasonable
                if total_area <= 0 or total_area > 1e12:  # Suspicious if > 1 million km²
                    self.logger.warning(f"Area from attribute {area_col} seems unrealistic: {total_area} m². Calculating geometrically.")
                else:
                    self.logger.info(f"Found catchment area from attribute: {total_area} m²")
                    return total_area
            
            # If area column not found or value is suspicious, calculate area from geometry
            self.logger.info("Calculating catchment area from geometry")
            
            # Make sure CRS is in a projected system for accurate area calculation
            if gdf.crs is None:
                self.logger.warning("Shapefile has no CRS information, assuming WGS84")
                gdf.crs = "EPSG:4326"
            
            # If geographic (lat/lon), reproject to a UTM zone for accurate area calculation
            if gdf.crs.is_geographic:
                # Calculate centroid to determine appropriate UTM zone
                centroid = gdf.dissolve().centroid.iloc[0]
                lon, lat = centroid.x, centroid.y
                
                # Determine UTM zone
                utm_zone = int(((lon + 180) / 6) % 60) + 1
                north_south = 'north' if lat >= 0 else 'south'
                
                utm_crs = f"+proj=utm +zone={utm_zone} +{north_south} +datum=WGS84 +units=m +no_defs"
                self.logger.info(f"Reprojecting from {gdf.crs} to UTM zone {utm_zone} ({utm_crs})")
                
                # Reproject
                gdf = gdf.to_crs(utm_crs)
            
            # Calculate area in m²
            gdf['calc_area'] = gdf.geometry.area
            total_area = gdf['calc_area'].sum()
            
            self.logger.info(f"Calculated catchment area from geometry: {total_area} m²")
            
            # Double check if area seems reasonable
            if total_area <= 0:
                self.logger.error(f"Calculated area is non-positive: {total_area} m²")
                return None
            
            if total_area > 1e12:  # > 1 million km²
                self.logger.warning(f"Calculated area seems very large: {total_area} m² ({total_area/1e6:.2f} km²). Check units.")
            
            return total_area
            
        except Exception as e:
            self.logger.error(f"Error calculating catchment area: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def _prepare_new_data(self, ensemble_results):
        """
        Prepare new data from ensemble results for training.
        
        Args:
            ensemble_results: List of result dictionaries from ensemble simulations
        
        Returns:
            Tuple of X and y DataFrames for model training
        """
        import pandas as pd
        import numpy as np
        
        self.logger.info("Preparing new data from ensemble results")
        
        # Create DataFrame for parameter values (X)
        param_data = []
        for result in ensemble_results:
            if result['success'] and result['metrics']:
                param_dict = {f"param_{k}": v for k, v in result['parameters'].items()}
                param_data.append(param_dict)
        
        if not param_data:
            self.logger.warning("No successful runs with metrics found")
            return pd.DataFrame(), pd.Series()
        
        X = pd.DataFrame(param_data)
        
        # Create Series for target metric (y)
        target_values = []
        for result in ensemble_results:
            if result['success'] and result['metrics']:
                if self.target_metric in result['metrics']:
                    target_values.append(result['metrics'][self.target_metric])
                else:
                    # If target metric not found, use KGE or similar
                    kge_key = next((k for k in result['metrics'] if 'KGE' in k), None)
                    if kge_key:
                        target_values.append(result['metrics'][kge_key])
                    else:
                        # Use first metric as fallback
                        target_values.append(next(iter(result['metrics'].values())))
        
        y = pd.Series(target_values)
        
        # Scale features with same scaler as original data
        if self.X_scaler is not None:
            # Ensure all columns from training are present in new data
            for col in self.X_scaler.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            
            # Ensure columns are in the same order as training
            X = X[self.X_scaler.feature_names_in_]
            
            # Scale the features
            X_scaled = pd.DataFrame(
                self.X_scaler.transform(X),
                columns=X.columns
            )
        else:
            # If no scaler yet, just return the raw data
            X_scaled = X
        
        self.logger.info(f"Prepared {len(X_scaled)} new data points with {X_scaled.shape[1]} features")
        return X_scaled, y

    def _run_with_best_parameters(self, best_params):
        """
        Run SUMMA with the best parameters from all iterations.
        
        Args:
            best_params: Dictionary of best parameter values
        
        Returns:
            True if SUMMA run was successful, False otherwise
        """
        self.logger.info("Running final model with best parameters from all iterations")
        
        # Create output directory for final run
        final_output_dir = self.emulation_output_dir / "final_optimized_run"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate trial parameters file with best parameters
        self._generate_trial_params_file(best_params, final_output_dir)
        
        # Run SUMMA
        summa_success = self.run_summa_with_optimal_parameters(final_output_dir)
        
        if summa_success:
            self.logger.info("Final optimized run completed successfully")
        else:
            self.logger.error("Final optimized run failed")
        
        return summa_success


    def _create_iteration_evolution_plots(self, iteration_results):
        """
        Create plots showing the evolution of parameters and performance across iterations.
        
        Args:
            iteration_results: List of dictionaries with results from each iteration
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        self.logger.info("Creating iteration evolution plots")
        
        # Extract data from iteration results
        iterations = [result['iteration'] for result in iteration_results]
        predicted_scores = [result['predicted_score'] for result in iteration_results]
        actual_scores = [result.get('actual_score', np.nan) for result in iteration_results]
        
        # Performance evolution plot
        plt.figure(figsize=(12, 6))
        plt.plot(iterations, predicted_scores, 'b-o', label='Predicted Score')
        
        # Plot actual scores if available
        valid_actuals = [(i, s) for i, s in zip(iterations, actual_scores) if not np.isnan(s)]
        if valid_actuals:
            plt.plot([i for i, _ in valid_actuals], [s for _, s in valid_actuals], 'r-o', label='Actual Score')
        
        plt.xlabel('Iteration')
        plt.ylabel(f'Performance Metric ({self.target_metric})')
        plt.title(f'Performance Evolution Across Iterations')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(iterations)
        
        # Add annotations with values
        for i, (pred, act) in enumerate(zip(predicted_scores, actual_scores)):
            plt.annotate(f"{pred:.3f}", (iterations[i], pred), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
            
            if not np.isnan(act):
                plt.annotate(f"{act:.3f}", (iterations[i], act), textcoords="offset points", 
                            xytext=(0, -15), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "performance_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter evolution plot (separate plot for each parameter)
        all_params = set()
        for result in iteration_results:
            all_params.update(result['optimized_parameters'].keys())
        
        # Create a DataFrame with parameter values for each iteration
        param_df = pd.DataFrame(index=iterations)
        for param in all_params:
            param_df[param] = [result['optimized_parameters'].get(param, np.nan) for result in iteration_results]
        
        # Plot each parameter evolution
        plt.figure(figsize=(15, 10))
        num_params = len(all_params)
        nrows = (num_params + 2) // 3  # Calculate rows needed for 3 columns
        ncols = min(3, num_params)
        
        for i, param in enumerate(sorted(all_params)):
            ax = plt.subplot(nrows, ncols, i+1)
            ax.plot(iterations, param_df[param], 'g-o')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title(f'Evolution of {param}')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(iterations)
            
            # Add parameter values as annotations
            for j, val in enumerate(param_df[param]):
                if not np.isnan(val):
                    ax.annotate(f"{val:.4f}", (iterations[j], val), textcoords="offset points", 
                            xytext=(0, 5), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "parameter_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined streamflow plot for all iterations
        self._create_combined_streamflow_plot(iteration_results)
        
        self.logger.info("Iteration evolution plots created")

    def _create_combined_streamflow_plot(self, iteration_results):
        """
        Create a combined plot showing the streamflow for each iteration,
        skipping the first 10% of data to avoid showing spin-up period.
        
        Args:
            iteration_results: List of dictionaries with results from each iteration
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from matplotlib.dates import DateFormatter
        
        self.logger.info("Creating combined streamflow plot (skipping spin-up period)")
        
        # Read observed data
        obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return
        
        try:
            # Load observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date column
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data")
                return
            
            # Identify flow column
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data")
                return
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Calculate the spin-up period (first 10% of data)
            total_days = (observed_flow.index.max() - observed_flow.index.min()).days
            spinup_days = int(total_days * 0.1)  # 10% of total period
            spinup_end_date = observed_flow.index.min() + pd.Timedelta(days=spinup_days)
            
            self.logger.info(f"Skipping spin-up period: {observed_flow.index.min()} to {spinup_end_date} ({spinup_days} days)")
            
            # Filter out spin-up period
            observed_flow = observed_flow[observed_flow.index >= spinup_end_date]
            
            # Get simulated streamflow for each iteration
            sim_flows = []
            for result in iteration_results:
                iter_num = result['iteration']
                iter_dir = self.emulation_output_dir / f"iteration_{iter_num}"
                
                # Look for streamflow files
                if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                    # Use mizuRoute output
                    mizu_dir = iter_dir / "simulations" / self.experiment_id / "mizuRoute"
                    if mizu_dir.exists():
                        import xarray as xr
                        
                        # Find NetCDF files
                        nc_files = list(mizu_dir.glob("*.nc"))
                        if nc_files:
                            # Get the first file
                            mizu_file = nc_files[0]
                            
                            # Get reach ID
                            sim_reach_id = self.config.get('SIM_REACH_ID')
                            
                            try:
                                with xr.open_dataset(mizu_file) as ds:
                                    # Find the index for the reach ID
                                    if 'reachID' in ds.variables:
                                        reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                                        
                                        if len(reach_indices) > 0:
                                            reach_index = reach_indices[0]
                                            
                                            # Try common variable names
                                            for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                                if var_name in ds.variables:
                                                    flow_series = ds[var_name].isel(seg=reach_index).to_pandas()
                                                    # Filter out spin-up period
                                                    flow_series = flow_series[flow_series.index >= spinup_end_date]
                                                    sim_flows.append((iter_num, flow_series))
                                                    break
                            except Exception as e:
                                self.logger.warning(f"Error reading mizuRoute output for iteration {iter_num}: {str(e)}")
                else:
                    # Use SUMMA output directly
                    summa_dir = iter_dir / "simulations" / self.experiment_id / "SUMMA"
                    if summa_dir.exists():
                        import xarray as xr
                        
                        # Find NetCDF files
                        nc_files = list(summa_dir.glob(f"{self.experiment_id}*timestep.nc"))
                        if nc_files:
                            # Get the first file
                            summa_file = nc_files[0]
                            
                            try:
                                with xr.open_dataset(summa_file) as ds:
                                    # Try to find streamflow variable
                                    for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                                        if var_name in ds.variables:
                                            if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                                # Sum across GRUs if multiple
                                                flow_series = ds[var_name].sum(dim='gru').to_pandas()
                                            else:
                                                # Single GRU or no gru dimension
                                                flow_series = ds[var_name].to_pandas()
                                                # Handle case if it's a DataFrame
                                                if isinstance(flow_series, pd.DataFrame):
                                                    flow_series = flow_series.iloc[:, 0]
                                            
                                            # Get catchment area
                                            catchment_area = self._get_catchment_area()
                                            if catchment_area and catchment_area > 0:
                                                # Convert from m/s to m³/s
                                                flow_series = flow_series * catchment_area
                                            
                                            # Filter out spin-up period
                                            flow_series = flow_series[flow_series.index >= spinup_end_date]
                                            sim_flows.append((iter_num, flow_series))
                                            break
                            except Exception as e:
                                self.logger.warning(f"Error reading SUMMA output for iteration {iter_num}: {str(e)}")
            
            if not sim_flows:
                self.logger.warning("No simulation outputs found to plot")
                return
            
            # Create plot
            plt.figure(figsize=(16, 10))
            
            # Plot observed flow
            plt.plot(observed_flow.index, observed_flow, 'k-', linewidth=2, label='Observed')
            
            # Plot simulated flows for each iteration
            colors = plt.cm.viridis(np.linspace(0, 1, len(sim_flows)))
            for i, (iter_num, flow_series) in enumerate(sim_flows):
                plt.plot(flow_series.index, flow_series, '-', color=colors[i], linewidth=1.5, 
                        label=f'Iteration {iter_num}')
            
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title('Streamflow Comparison Across Iterations (Spin-up Period Excluded)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(self.emulation_output_dir / "streamflow_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create a DataFrame with all flows for further analysis
            all_flows = pd.DataFrame(index=observed_flow.index)
            all_flows['Observed'] = observed_flow
            
            for iter_num, flow_series in sim_flows:
                # Align index with observed data
                reindexed_flow = flow_series.reindex(observed_flow.index, method='nearest')
                all_flows[f'Iteration_{iter_num}'] = reindexed_flow
            
            # Add a column for the final optimized run if available
            final_dir = self.emulation_output_dir / "final_optimized_run"
            if final_dir.exists():
                final_flow = self._get_final_optimized_flow(final_dir)
                if final_flow is not None:
                    # Filter out spin-up period
                    final_flow = final_flow[final_flow.index >= spinup_end_date]
                    # Align index with observed data
                    reindexed_final = final_flow.reindex(observed_flow.index, method='nearest')
                    all_flows['Final_Optimized'] = reindexed_final
            
            # Save flow data for potential further analysis
            all_flows.to_csv(self.emulation_output_dir / "all_streamflows.csv")
            
            self.logger.info("Combined streamflow plot created (spin-up period excluded)")
            
        except Exception as e:
            self.logger.error(f"Error creating combined streamflow plot: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _get_final_optimized_flow(self, final_dir, exclude_spinup=True):
        """
        Extract the streamflow from the final optimized run.
        
        Args:
            final_dir: Directory for the final optimized run
            exclude_spinup: Whether to exclude the spin-up period
        
        Returns:
            Series with the streamflow data, or None if not found
        """
        try:
            # Calculate spin-up period end date if needed
            spinup_end_date = None
            if exclude_spinup:
                # Get observed data to determine the date range
                obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
                if obs_path.exists():
                    obs_df = pd.read_csv(obs_path)
                    date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
                    if date_col:
                        obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
                        total_days = (obs_df['DateTime'].max() - obs_df['DateTime'].min()).days
                        spinup_days = int(total_days * 0.1)  # 10% of total period
                        spinup_end_date = obs_df['DateTime'].min() + pd.Timedelta(days=spinup_days)
            
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                # Use mizuRoute output
                mizu_dir = final_dir / "simulations" / self.experiment_id / "mizuRoute"
                if mizu_dir.exists():
                    import xarray as xr
                    
                    # Find NetCDF files
                    nc_files = list(mizu_dir.glob("*.nc"))
                    if nc_files:
                        # Get the first file
                        mizu_file = nc_files[0]
                        
                        # Get reach ID
                        sim_reach_id = self.config.get('SIM_REACH_ID')
                        
                        with xr.open_dataset(mizu_file) as ds:
                            # Find the index for the reach ID
                            if 'reachID' in ds.variables:
                                reach_indices = (ds['reachID'].values == int(sim_reach_id)).nonzero()[0]
                                
                                if len(reach_indices) > 0:
                                    reach_index = reach_indices[0]
                                    
                                    # Try common variable names
                                    for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff', 'instRunoff']:
                                        if var_name in ds.variables:
                                            flow_series = ds[var_name].isel(seg=reach_index).to_pandas()
                                            # Filter out spin-up period if requested
                                            if exclude_spinup and spinup_end_date is not None:
                                                flow_series = flow_series[flow_series.index >= spinup_end_date]
                                            return flow_series
            else:
                # Use SUMMA output directly
                summa_dir = final_dir / "simulations" / self.experiment_id / "SUMMA"
                if summa_dir.exists():
                    import xarray as xr
                    
                    # Find NetCDF files
                    nc_files = list(summa_dir.glob(f"{self.experiment_id}*timestep.nc"))
                    if nc_files:
                        # Get the first file
                        summa_file = nc_files[0]
                        
                        with xr.open_dataset(summa_file) as ds:
                            # Try to find streamflow variable
                            for var_name in ['outflow', 'basRunoff', 'averageRoutedRunoff', 'totalRunoff']:
                                if var_name in ds.variables:
                                    if 'gru' in ds[var_name].dims and ds.sizes['gru'] > 1:
                                        # Sum across GRUs if multiple
                                        flow_series = ds[var_name].sum(dim='gru').to_pandas()
                                    else:
                                        # Single GRU or no gru dimension
                                        flow_series = ds[var_name].to_pandas()
                                        # Handle case if it's a DataFrame
                                        if isinstance(flow_series, pd.DataFrame):
                                            flow_series = flow_series.iloc[:, 0]
                                    
                                    # Get catchment area
                                    catchment_area = self._get_catchment_area()
                                    if catchment_area and catchment_area > 0:
                                        # Convert from m/s to m³/s
                                        flow_series = flow_series * catchment_area
                                    
                                    # Filter out spin-up period if requested
                                    if exclude_spinup and spinup_end_date is not None:
                                        flow_series = flow_series[flow_series.index >= spinup_end_date]
                                        
                                    return flow_series
            
            return None
        except Exception as e:
            self.logger.warning(f"Error getting final optimized flow: {str(e)}")
            return None

    def _calculate_performance_for_run(self, streamflow_file):
        """
        Calculate performance metrics for a run by comparing to observed data.
        
        Args:
            streamflow_file: Path to the streamflow CSV file
        
        Returns:
            Dictionary of performance metrics
        """
        import pandas as pd
        import numpy as np
        
        # Get observed data path
        obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return None
        
        try:
            # Read observed data
            obs_df = pd.read_csv(obs_path)
            
            # Identify date column
            date_col = next((col for col in obs_df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data")
                return None
            
            # Identify flow column
            flow_col = next((col for col in obs_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data")
                return None
            
            # Convert date column to datetime and set as index
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Read simulated data
            sim_df = pd.read_csv(streamflow_file)
            
            # Identify time column
            time_col = next((col for col in sim_df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
            if not time_col:
                self.logger.warning(f"No time column found in simulated data")
                return None
            
            # Identify flow column
            sim_flow_col = next((col for col in sim_df.columns if 'flow' in col.lower() or 'discharge' in col.lower() or 'streamflow' in col.lower()), None)
            if not sim_flow_col:
                self.logger.warning(f"No flow column found in simulated data")
                return None
            
            # Convert time column to datetime and set as index
            sim_df['DateTime'] = pd.to_datetime(sim_df[time_col])
            sim_df.set_index('DateTime', inplace=True)
            simulated_flow = sim_df[sim_flow_col]
            
            # Filter data to calibration and evaluation periods if defined
            calib_start, calib_end = self.calibration_period
            eval_start, eval_end = self.evaluation_period
            
            # Create dictionaries to hold metrics
            calib_metrics = {}
            eval_metrics = {}
            all_metrics = {}
            
            # Calculate metrics for calibration period if defined
            if calib_start and calib_end:
                calib_mask = (observed_flow.index >= calib_start) & (observed_flow.index <= calib_end)
                calib_obs = observed_flow[calib_mask]
                
                # Find corresponding simulated values
                common_calib_idx = calib_obs.index.intersection(simulated_flow.index)
                if len(common_calib_idx) > 0:
                    calib_obs_aligned = calib_obs.loc[common_calib_idx]
                    calib_sim_aligned = simulated_flow.loc[common_calib_idx]
                    
                    # Calculate metrics
                    calib_metrics = self._calculate_streamflow_metrics(calib_obs_aligned, calib_sim_aligned)
                    for key, value in calib_metrics.items():
                        all_metrics[f"Calib_{key}"] = value
                    
                    # Always include KGE for convenience
                    if "Calib_KGE" in all_metrics:
                        all_metrics["KGE"] = all_metrics["Calib_KGE"]
            
            # Calculate metrics for evaluation period if defined
            if eval_start and eval_end:
                eval_mask = (observed_flow.index >= eval_start) & (observed_flow.index <= eval_end)
                eval_obs = observed_flow[eval_mask]
                
                # Find corresponding simulated values
                common_eval_idx = eval_obs.index.intersection(simulated_flow.index)
                if len(common_eval_idx) > 0:
                    eval_obs_aligned = eval_obs.loc[common_eval_idx]
                    eval_sim_aligned = simulated_flow.loc[common_eval_idx]
                    
                    # Calculate metrics
                    eval_metrics = self._calculate_streamflow_metrics(eval_obs_aligned, eval_sim_aligned)
                    for key, value in eval_metrics.items():
                        all_metrics[f"Eval_{key}"] = value
                    
                    # If no calibration metrics, use evaluation KGE
                    if "KGE" not in all_metrics and "Eval_KGE" in all_metrics:
                        all_metrics["KGE"] = all_metrics["Eval_KGE"]
            
            # If no calibration or evaluation periods defined, use all data
            if not all_metrics:
                # Find common time period
                common_idx = observed_flow.index.intersection(simulated_flow.index)
                if len(common_idx) > 0:
                    obs_aligned = observed_flow.loc[common_idx]
                    sim_aligned = simulated_flow.loc[common_idx]
                    
                    # Calculate metrics
                    all_metrics = self._calculate_streamflow_metrics(obs_aligned, sim_aligned)
            
            return all_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _run_summa_for_iteration(self, optimized_params, iteration_output_dir):
        """
        Run SUMMA with optimized parameters for a specific iteration.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            bool: True if SUMMA run was successful, False otherwise
        """
        self.logger.info(f"Running SUMMA with optimized parameters in {iteration_output_dir}")
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all necessary SUMMA settings files first
        source_settings_dir = self.project_dir / "settings" / "SUMMA"
        self._copy_settings_files(source_settings_dir, settings_dir)
        
        # Generate trial parameters file specifically for this iteration
        trial_params_path = self._generate_trial_params_file_for_iteration(optimized_params, iteration_output_dir)
        if not trial_params_path:
            self.logger.error("Could not generate trial parameters file for iteration")
            return False
        
        # Define paths
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.data_dir / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
        
        summa_exe = summa_path / self.config.get('SUMMA_EXE')
        
        # Create a modified fileManager for this iteration
        modified_filemanager = self._create_iteration_file_manager(iteration_output_dir)
        if not modified_filemanager:
            self.logger.error("Could not create modified fileManager for iteration")
            return False
        
        # Create output directory
        sim_dir = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA"
        sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log directory
        log_dir = sim_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Run SUMMA
        summa_command = f"{summa_exe} -m {modified_filemanager}"
        log_file = log_dir / "summa_rf_optimized.log"
        
        try:
            # Run SUMMA and capture output
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    summa_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"SUMMA run completed successfully for iteration, log saved to: {log_file}")
            
            # Run mizuRoute if needed
            if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
                self._run_mizuroute_for_iteration(iteration_output_dir)
            
            # Evaluate output for this iteration
            metrics = self._evaluate_summa_output(iteration_output_dir)  # Pass the output directory
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running SUMMA for iteration: {str(e)}")
            return False

    def _copy_settings_files(self, source_dir, target_dir):
        """
        Copy all necessary settings files from source to target directory.
        
        Args:
            source_dir: Source directory with original settings files
            target_dir: Target directory to copy files to
        """
        self.logger.info(f"Copying settings files from {source_dir} to {target_dir}")
        
        # Files to copy
        files_to_copy = [
            'outputControl.txt',
            'modelDecisions.txt',
            'localParamInfo.txt',
            'basinParamInfo.txt',
            'TBL_GENPARM.TBL',
            'TBL_MPTABLE.TBL',
            'TBL_SOILPARM.TBL',
            'TBL_VEGPARM.TBL',
            'attributes.nc',
            'coldState.nc',
            'forcingFileList.txt'
        ]
        
        for file_name in files_to_copy:
            source_file = source_dir / file_name
            target_file = target_dir / file_name
            
            if source_file.exists():
                try:
                    import shutil
                    shutil.copy2(source_file, target_file)
                    self.logger.debug(f"Copied {file_name} to {target_dir}")
                except Exception as e:
                    self.logger.warning(f"Could not copy {file_name}: {str(e)}")
            else:
                self.logger.warning(f"Source file {source_file} not found")
        
    def _generate_trial_params_file_for_iteration(self, optimized_params, iteration_output_dir):
        """
        Generate a trialParams.nc file with the optimized parameters for a specific iteration.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            Path: Path to the generated file
        """
        self.logger.info(f"Generating trial parameters file for iteration in {iteration_output_dir}")
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Get attribute file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return None
        
        try:
            # Read HRU information from the attributes file
            with nc.Dataset(attr_file, 'r') as ds:
                hru_ids = ds.variables['hruId'][:]
                
                # Create the trial parameters dataset
                output_path = settings_dir / "trialParams.nc"
                output_ds = nc.Dataset(output_path, 'w', format='NETCDF4')
                
                # Add dimensions
                output_ds.createDimension('hru', len(hru_ids))
                
                # Add HRU ID variable
                hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Add parameter variables
                for param_name, param_value in optimized_params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Create variable
                    param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=False)
                    param_var[:] = param_array
                    param_var.long_name = f"Trial value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                output_ds.description = "SUMMA Trial Parameter file generated by CONFLUENCE RandomForestEmulator"
                output_ds.history = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.confluence_experiment_id = self.experiment_id
                
                # Close the file explicitly
                output_ds.close()
                
                self.logger.info(f"Generated trial parameters file: {output_path}")
                return output_path
                    
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _create_iteration_file_manager(self, iteration_output_dir):
        """
        Create a modified fileManager for a specific iteration.
        
        Args:
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            Path: Path to the modified fileManager
        """
        source_file_manager = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        
        if not source_file_manager.exists():
            self.logger.error(f"Original fileManager not found at {source_file_manager}")
            return None
        
        # Create settings directory
        settings_dir = iteration_output_dir / "settings" / "SUMMA"
        settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Path to the new file manager
        new_fm_path = settings_dir / os.path.basename(source_file_manager)
        
        try:
            # Read the original file manager
            with open(source_file_manager, 'r') as f:
                lines = f.readlines()
            
            # Modify relevant lines
            modified_lines = []
            for line in lines:
                if "outFilePrefix" in line:
                    # Update output file prefix
                    modified_line = f"outFilePrefix        '{self.experiment_id}_rf_iter'\n"
                    modified_lines.append(modified_line)
                elif "outputPath" in line:
                    # Update output path
                    output_path = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                    output_path_str = str(output_path).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    modified_line = f"outputPath           '{output_path_str}/'\n"
                    modified_lines.append(modified_line)
                elif "settingsPath" in line:
                    # Update settings path
                    settings_path_str = str(settings_dir).replace('\\', '/')  # Ensure forward slashes for SUMMA
                    modified_line = f"settingsPath         '{settings_path_str}/'\n"
                    modified_lines.append(modified_line)
                elif "trialParamFile" in line:
                    # Use the iteration-specific trial params file
                    modified_line = f"trialParamFile       'trialParams.nc'\n"
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            
            # Write the modified file manager
            with open(new_fm_path, 'w') as f:
                f.writelines(modified_lines)
                
            self.logger.info(f"Created modified fileManager for iteration: {new_fm_path}")
            return new_fm_path
            
        except Exception as e:
            self.logger.error(f"Error creating modified fileManager for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _run_mizuroute_for_iteration(self, iteration_output_dir):
        """
        Run mizuRoute with the SUMMA outputs for a specific iteration.
        
        Args:
            iteration_output_dir: Directory for this iteration's output
            
        Returns:
            bool: True if mizuRoute run was successful, False otherwise
        """
        self.logger.info(f"Running mizuRoute for iteration in {iteration_output_dir}")
        
        # Get mizuRoute executable path
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        if mizu_path == 'default':
            mizu_path = self.data_dir / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        
        # Create settings directory
        mizu_settings_dir = iteration_output_dir / "settings" / "mizuRoute"
        mizu_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and modify mizuRoute control file
        source_control_file = self.project_dir / "settings" / "mizuRoute" / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        if not source_control_file.exists():
            self.logger.error(f"Source mizuRoute control file not found: {source_control_file}")
            return False
        
        # Modified control file path
        control_file = mizu_settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
        
        try:
            # Read the original control file
            with open(source_control_file, 'r') as f:
                lines = f.readlines()
            
            # Create output directory
            mizu_output_dir = iteration_output_dir / "simulations" / self.experiment_id / "mizuRoute"
            mizu_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_dir = mizu_output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Modify the control file
            with open(control_file, 'w') as f:
                for line in lines:
                    if '<input_dir>' in line:
                        # Update input directory (SUMMA output)
                        summa_output_dir = iteration_output_dir / "simulations" / self.experiment_id / "SUMMA" / ""
                        summa_output_str = str(summa_output_dir).replace('\\', '/')
                        f.write(f"<input_dir>             {summa_output_str}/    ! Folder that contains runoff data from SUMMA \n")
                    elif '<ancil_dir>' in line:
                        # Use the original ancillary directory
                        source_ancil_dir = self.project_dir / "settings" / "mizuRoute"
                        source_ancil_str = str(source_ancil_dir).replace('\\', '/')
                        f.write(f"<ancil_dir>             {source_ancil_str}/    ! Folder that contains ancillary data \n")
                    elif '<output_dir>' in line:
                        # Update output directory
                        mizu_output_str = str(mizu_output_dir).replace('\\', '/')
                        f.write(f"<output_dir>            {mizu_output_str}/    ! Folder that will contain mizuRoute simulations \n")
                    elif '<case_name>' in line:
                        # Update case name
                        f.write(f"<case_name>             {self.experiment_id}_rf_iter    ! Simulation case name \n")
                    elif '<fname_qsim>' in line:
                        # Update input file name
                        f.write(f"<fname_qsim>            {self.experiment_id}_rf_iter_timestep.nc    ! netCDF name for HM_HRU runoff \n")
                    else:
                        f.write(line)
            
            # Run mizuRoute
            mizu_command = f"{mizu_exe} {control_file}"
            log_file = log_dir / "mizuroute_rf_optimized.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    mizu_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True
                )
            
            self.logger.info(f"mizuRoute run completed successfully for iteration, log saved to: {log_file}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running mizuRoute for iteration: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error setting up or running mizuRoute for iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def run_workflow(self):
        """
        Run the complete random forest emulation workflow with iterative refinement:
        1. Load and prepare initial data
        2. For each iteration:
        a. Train the random forest model with three-way split
        b. Optimize parameters
        c. Generate new parameter ensemble centered around optimized parameters
        d. Run models with new ensemble and calculate performance metrics
        e. Append new results to the dataset
        3. Run final SUMMA with best optimized parameters
        4. Evaluate and visualize results across iterations
        
        Returns:
            Dictionary with workflow results
        """
        self.logger.info("Starting random forest emulation workflow with iterative refinement")
        
        # Get iteration settings from config
        max_iterations = self.config.get('EMULATION_ITERATIONS', 3)
        stop_criteria = self.config.get('EMULATION_STOP_CRITERIA', 0.01)
        
        # Track best score and parameters across iterations
        best_score = float('-inf')
        best_params = None
        iteration_results = []
        
        try:
            # Step 1: Load and prepare initial data
            X, y = self.load_and_prepare_data()
            self.logger.info(f"Initial data prepared with {X.shape[1]} features and {len(y)} samples")
            
            # Initialize tracking for iteration metrics
            iteration_metrics = []
            
            # Iterate until max iterations or convergence
            for iteration in range(max_iterations):
                self.logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
                
                # Step 2a: Train the random forest model
                model, metrics = self.train_random_forest(X, y)
                self.logger.info(f"Model trained with R² scores - Test: {metrics['test_score']:.4f}")
                
                # Step 2b: Optimize parameters
                optimized_params, predicted_score = self.optimize_parameters()
                self.logger.info(f"Parameters optimized with predicted score: {predicted_score:.4f}")
                
                # Save this iteration's results
                iteration_result = {
                    'iteration': iteration + 1,
                    'model_metrics': metrics,
                    'optimized_parameters': optimized_params,
                    'predicted_score': predicted_score
                }
                iteration_results.append(iteration_result)
                
                # Run SUMMA with optimized parameters for this iteration
                iteration_output_dir = self.emulation_output_dir / f"iteration_{iteration+1}"
                iteration_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a modified version of run_summa_with_optimal_parameters for this iteration
                summa_success = self._run_summa_for_iteration(optimized_params, iteration_output_dir)
                
                if summa_success:
                    # Get actual performance from this run
                    actual_metrics = self._evaluate_summa_output(iteration_output_dir)
                    if actual_metrics:
                        actual_score = actual_metrics.get(self.target_metric, float('-inf'))
                        iteration_result['actual_score'] = actual_score
                        
                        # Track if this is the best result so far
                        if actual_score > best_score:
                            best_score = actual_score
                            best_params = optimized_params.copy()
                            self.logger.info(f"New best score: {best_score:.4f} in iteration {iteration+1}")
                    
                # Check for convergence
                if iteration > 0:
                    prev_score = iteration_results[iteration-1].get('actual_score', 
                                                            iteration_results[iteration-1]['predicted_score'])
                    current_score = iteration_result.get('actual_score', predicted_score)
                    improvement = abs(current_score - prev_score)
                    
                    self.logger.info(f"Improvement from previous iteration: {improvement:.4f}")
                    if improvement < stop_criteria:
                        self.logger.info(f"Stopping due to convergence (improvement < {stop_criteria})")
                        break
                
                # Step 2c: Generate new parameter ensemble centered around optimized parameters
                if iteration < max_iterations - 1:  # Skip on last iteration
                    self.logger.info("Generating new parameter ensemble for next iteration")
                    new_param_sets = self._generate_new_ensemble(optimized_params, iteration)
                    
                    # Step 2d: Run models with new ensemble
                    self.logger.info("Running models with new parameter ensemble")
                    new_results = self._run_ensemble_simulations(new_param_sets, iteration)
                    
                    # Step 2e: Append new results to dataset for next iteration
                    X_new, y_new = self._prepare_new_data(new_results)
                    X = pd.concat([X, X_new], ignore_index=True)
                    y = pd.concat([y, y_new], ignore_index=True)
                    self.logger.info(f"Dataset expanded to {len(y)} samples")
            
            # Use the best parameters from all iterations for final run
            if best_params:
                self.logger.info(f"Running final model with best parameters (score: {best_score:.4f})")
                final_success = self.run_summa_with_optimal_parameters()
            else:
                self.logger.info("Using parameters from final iteration for final model")
                final_success = True  # Already ran in last iteration
            
            # Generate summary visualizations across iterations
            self._create_iteration_evolution_plots(iteration_results)
            
            # Return results
            results = {
                'iterations': iteration_results,
                'best_score': best_score,
                'best_parameters': best_params,
                'final_success': final_success,
                'output_dir': str(self.emulation_output_dir)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in random forest emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise