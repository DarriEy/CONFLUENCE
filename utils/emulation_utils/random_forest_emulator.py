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

class RandomForestEmulator:
    """
    Random Forest Emulator for CONFLUENCE.
    
    This class uses the attribute data, parameter sets, and performance metrics
    to train a random forest model that can predict performance metrics based
    on attribute and parameter values. It then optimizes the parameter values
    to maximize performance and runs SUMMA with the optimal parameter set.
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
        
        # Optimization settings
        self.n_optimization_runs = self.config.get('EMULATION_N_OPTIMIZATION_RUNS', 10)
        
        # Model and scaler objects
        self.model = None
        self.X_scaler = None
        self.param_mins = {}
        self.param_maxs = {}
        self.param_cols = []
        self.attribute_cols = []
        
        # Check if all required files exist
        self._check_required_files()
        
    def _check_required_files(self):
        """Check if all required input files exist."""
        self.logger.info("Checking required input files")
        
        files_to_check = [
            (self.attributes_path, "Attribute data"),
            (self.parameter_sets_path, "Parameter sets"),
            (self.performance_metrics_path, "Performance metrics")
        ]
        
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
        Load and prepare the data for training the random forest model.
        
        Returns:
            Tuple of X and y arrays for model training
        """
        self.logger.info("Loading and preparing data for random forest training")
        
        # Load attributes data
        attributes_df = pd.read_csv(self.attributes_path)
        self.logger.info(f"Loaded attributes data with shape: {attributes_df.shape}")
        
        # Load parameter sets
        parameter_data = self._load_parameter_sets()
        self.logger.info(f"Loaded parameter data with shape: {parameter_data.shape}")
        
        # Load performance metrics
        performance_df = pd.read_csv(self.performance_metrics_path)
        self.logger.info(f"Loaded performance metrics with shape: {performance_df.shape}")
        
        # Print debugging info about the data
        self.logger.info(f"Parameter data columns: {parameter_data.columns.tolist()}")
        self.logger.info(f"Performance metrics columns: {performance_df.columns.tolist()}")
        
        # Special handling for performance metrics file
        # If it has an unnamed index column, use it as the index
        if 'Unnamed: 0' in performance_df.columns:
            run_ids = performance_df['Unnamed: 0'].tolist()
            self.logger.info(f"Using 'Unnamed: 0' column as index with values: {run_ids[:5]}...")
            performance_df = performance_df.set_index('Unnamed: 0')
        else:
            performance_df = performance_df.reset_index(drop=True)
        
        # Handle the target metric
        if self.target_metric not in performance_df.columns:
            available_metrics = [col for col in performance_df.columns 
                                if col in ['KGE', 'NSE', 'RMSE', 'PBIAS']]
            if available_metrics:
                self.target_metric = available_metrics[0]
                self.logger.warning(f"Target metric not found, using {self.target_metric} instead")
            else:
                raise ValueError(f"No valid performance metrics found in {performance_df.columns}")
        
        # Filter out NaN values in the target metric
        valid_metrics = ~performance_df[self.target_metric].isna()
        self.logger.info(f"Found {valid_metrics.sum()} valid entries for {self.target_metric} out of {len(performance_df)}")
        
        if valid_metrics.sum() == 0:
            raise ValueError(f"No valid data for target metric {self.target_metric}")
        
        performance_df = performance_df[valid_metrics]
        
        # If we have run IDs in the index, make sure we only use parameters for those runs
        if isinstance(performance_df.index, pd.Index) and performance_df.index.name is not None:
            # Extract run numbers (assuming run_XXXX format)
            if performance_df.index.dtype == 'object' and isinstance(performance_df.index[0], str):
                # Extract numeric part if index is like "run_0001"
                run_nums = []
                for idx in performance_df.index:
                    if 'run_' in idx:
                        try:
                            run_nums.append(int(idx.split('_')[1]))
                        except (IndexError, ValueError):
                            run_nums.append(None)
                    else:
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
                                performance_df.reset_index(drop=True)], axis=1)
        
        self.logger.info(f"Merged parameter and performance data with shape: {merged_data.shape}")
        
        # Add attribute data
        if len(attributes_df) > 0:
            # Clean up attribute data - remove problematic columns
            # First, detect columns with missing or NaN values
            attr_na_cols = attributes_df.columns[attributes_df.isna().any()].tolist()
            if attr_na_cols:
                self.logger.info(f"Dropping attribute columns with NaN values: {attr_na_cols}")
                attributes_df = attributes_df.drop(columns=attr_na_cols)
            
            # For lumped catchment, just use the first row
            attr_row = attributes_df.iloc[0:1]
            self.logger.info(f"Using attribute row with shape: {attr_row.shape}")
            
            # Create repeated attribute rows
            attr_data = pd.DataFrame(
                np.repeat(attr_row.values, len(merged_data), axis=0),
                columns=attr_row.columns
            )
            
            # Exclude the first column if it's an ID column
            if attr_data.shape[1] > 1 and attr_data.columns[0].lower() in ['id', 'hru_id', 'gru_id', 'basin_id']:
                attr_data = attr_data.iloc[:, 1:]
                
            # Combine with merged data
            merged_data = pd.concat([merged_data, attr_data.reset_index(drop=True)], axis=1)
            self.logger.info(f"After adding attributes, merged data shape: {merged_data.shape}")
        
        # Handle missing values 
        # First, check for columns with ALL missing values - drop these completely
        na_counts = merged_data.isna().sum()
        all_na_cols = na_counts[na_counts == len(merged_data)].index.tolist()
        if all_na_cols:
            self.logger.warning(f"Dropping columns with all missing values: {all_na_cols}")
            merged_data = merged_data.drop(columns=all_na_cols)
        
        # For remaining missing values, fill numeric columns with means
        numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            missing_count = merged_data[col].isna().sum()
            if missing_count > 0:
                col_mean = merged_data[col].mean()
                merged_data[col].fillna(col_mean, inplace=True)
                self.logger.info(f"Filled {missing_count} missing values in {col} with mean: {col_mean}")
        
        # For non-numeric columns with missing values, drop the column if > 50% missing, otherwise fill with mode
        non_numeric_cols = [col for col in merged_data.columns if col not in numeric_cols]
        problem_cols = []
        
        for col in non_numeric_cols:
            missing_count = merged_data[col].isna().sum()
            if missing_count > 0:
                missing_percent = missing_count / len(merged_data) * 100
                if missing_percent > 50:
                    problem_cols.append(col)
                    self.logger.warning(f"Column {col} has {missing_percent:.1f}% missing values - will be dropped")
                else:
                    # Try to get the mode, handle empty mode result
                    mode_result = merged_data[col].mode()
                    if not mode_result.empty:
                        mode_val = mode_result.iloc[0]
                        merged_data[col].fillna(mode_val, inplace=True)
                        self.logger.info(f"Filled {missing_count} missing values in {col} with mode: {mode_val}")
                    else:
                        problem_cols.append(col)
                        self.logger.warning(f"Could not find mode for {col} - will be dropped")
        
        # Drop problematic columns
        if problem_cols:
            self.logger.warning(f"Dropping problematic columns: {problem_cols}")
            merged_data = merged_data.drop(columns=problem_cols)
        
        # Final check for any remaining missing values
        missing_values = merged_data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        if not columns_with_missing.empty:
            self.logger.warning(f"Still have columns with missing values after processing: {columns_with_missing}")
            # Drop these columns too rather than dropping rows
            self.logger.warning(f"Dropping columns with remaining missing values")
            merged_data = merged_data.drop(columns=columns_with_missing.index.tolist())
        
        # Final check - ensure we have enough data
        if merged_data.shape[0] < 5:  # Arbitrary minimum for RF model
            raise ValueError(f"Not enough data for training: only {merged_data.shape[0]} samples after preprocessing")
        
        self.logger.info(f"Final data shape after preprocessing: {merged_data.shape}")
        
        # Split into features and target, also pass merged_data for categorical encoding
        X, y, merged_data_full = self._split_features_target(merged_data)
        
        # Save record of which columns are parameters vs attributes
        self.param_cols = [col for col in X.columns if col.startswith('param_')]
        self.attribute_cols = [col for col in X.columns if not col.startswith('param_')]
        
        self.logger.info(f"Feature split: {len(self.param_cols)} parameters, {len(self.attribute_cols)} attributes")
        
        # Store parameter mins and maxs for optimization bounds
        for param in self.param_cols:
            self.param_mins[param] = X[param].min()
            self.param_maxs[param] = X[param].max()
        
        # Scale the features, passing merged_data for categorical encoding
        X_scaled, scaler = self._scale_features(X, merged_data_full)
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
        
    def _merge_datasets(self, attributes_df, parameter_data, performance_df):
        """
        Merge attributes, parameters, and performance metrics.
        
        Args:
            attributes_df: DataFrame with attribute data
            parameter_data: DataFrame with parameter values
            performance_df: DataFrame with performance metrics
        
        Returns:
            Merged DataFrame
        """
        # Simple concatenation since indices are now aligned
        merged = pd.concat([parameter_data, performance_df], axis=1)
        self.logger.info(f"Merged parameters and metrics with shape: {merged.shape}")
        
        # If attributes are per GRU, we need to aggregate them
        if 'hru_id' in attributes_df.columns or 'HRU_ID' in attributes_df.columns:
            self.logger.info("Attributes appear to be per GRU, aggregating to catchment level")
            
            # Identify the HRU ID column
            hru_id_col = 'hru_id' if 'hru_id' in attributes_df.columns else 'HRU_ID'
            
            # Aggregate attributes (mean across all HRUs)
            attributes_agg = attributes_df.drop(columns=[hru_id_col]).mean().to_frame().T
            
            # Replicate the attributes for each row in merged data
            attributes_rep = pd.DataFrame(
                np.repeat(attributes_agg.values, len(merged), axis=0),
                columns=attributes_agg.columns,
                index=merged.index
            )
            
            # Combine with merged data
            result = pd.concat([merged, attributes_rep], axis=1)
        else:
            # Attributes are already at catchment level - use first row
            if len(attributes_df) > 0:
                # Take first row of attributes and repeat for all runs
                attributes_row = attributes_df.iloc[0:1]
                attributes_rep = pd.DataFrame(
                    np.repeat(attributes_row.values, len(merged), axis=0),
                    columns=attributes_row.columns,
                    index=merged.index
                )
                # Drop the first column which is typically the ID
                if attributes_rep.shape[1] > 1:
                    attributes_rep = attributes_rep.iloc[:, 1:]
                
                # Combine with merged data
                result = pd.concat([merged, attributes_rep], axis=1)
            else:
                self.logger.warning("Empty attributes DataFrame, skipping attributes")
                result = merged
        
        self.logger.info(f"Final merged dataset has shape: {result.shape}")
        return result
        
    def _split_features_target(self, merged_data):
        """
        Split merged data into features (X) and target (y).
        
        Args:
            merged_data: Merged DataFrame with all data
        
        Returns:
            Tuple of (X, y) DataFrames
        """
        # Check if target metric exists
        if self.target_metric not in merged_data.columns:
            available_metrics = [col for col in merged_data.columns 
                                if col in ['KGE', 'NSE', 'RMSE', 'PBIAS']]
            if available_metrics:
                self.target_metric = available_metrics[0]
                self.logger.warning(f"Target metric not found, using {self.target_metric} instead")
            else:
                raise ValueError(f"Target metric {self.target_metric} not found in data and no alternatives available")
        
        # Identify features - parameters and attributes
        param_cols = [col for col in merged_data.columns if col.startswith('param_')]
        metric_cols = ['KGE', 'NSE', 'RMSE', 'PBIAS']
        attribute_cols = [col for col in merged_data.columns 
                        if col not in param_cols + metric_cols 
                        and col != 'Unnamed: 0']  # Exclude index column if present
        
        self.logger.info(f"Feature columns breakdown: {len(param_cols)} parameters, {len(attribute_cols)} attributes")
        
        # Now identify which columns are categorical (non-numeric)
        # We'll need to handle these differently during scaling
        non_numeric_cols = merged_data[attribute_cols].select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            self.logger.info(f"Found {len(non_numeric_cols)} non-numeric attribute columns that will be one-hot encoded")
            self.logger.debug(f"Non-numeric columns: {non_numeric_cols[:10]}...")
            
            # Drop non-numeric columns for now - we'll one-hot encode them in _scale_features
            # We store them for later use
            self.non_numeric_cols = non_numeric_cols
            attribute_cols = [col for col in attribute_cols if col not in non_numeric_cols]
        else:
            self.non_numeric_cols = []
        
        # Create feature set (parameters + numeric attributes)
        X = merged_data[param_cols + attribute_cols]
        
        # For target, we want to maximize KGE and NSE, but minimize RMSE and PBIAS (absolute value)
        if self.target_metric in ['KGE', 'NSE']:
            y = merged_data[self.target_metric]
        elif self.target_metric == 'RMSE':
            y = -merged_data[self.target_metric]  # Negate to convert to maximization problem
        elif self.target_metric == 'PBIAS':
            y = -merged_data[self.target_metric].abs()  # Negate absolute value for maximization
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

    def train_random_forest(self, X, y, test_size=0.2):
        """
        Train a random forest model using the prepared data.
        
        Args:
            X: Scaled feature data
            y: Target values
            test_size: Proportion of data to use for testing
        
        Returns:
            Trained model and evaluation metrics
        """
        self.logger.info("Training random forest model")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Initialize and train the model
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate the model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.logger.info(f"Random forest model trained with R² score on training: {train_score:.4f}, testing: {test_score:.4f}")
        
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
        
        return model, {'train_score': train_score, 'test_score': test_score}
    
    def optimize_parameters(self, fixed_attributes):
        """
        Optimize parameters to maximize the predicted performance metric.
        
        Args:
            fixed_attributes: Dictionary of fixed attribute values to use
        
        Returns:
            Dictionary of optimized parameter values
        """
        self.logger.info("Starting parameter optimization")
        
        # Check if model exists
        if self.model is None:
            raise ValueError("Model not trained. Call train_random_forest() first.")
        
        # Check if X_scaler exists
        if self.X_scaler is None:
            raise ValueError("Feature scaler not found. Call load_and_prepare_data() first.")
        
        # Make sure all attribute columns are in fixed_attributes
        missing_attrs = set(self.attribute_cols) - set(fixed_attributes.keys())
        if missing_attrs:
            self.logger.warning(f"Missing attribute values: {missing_attrs}. Using mean values.")
            
            # Fill in missing attributes with means from the training data
            X_means = pd.DataFrame(
                self.X_scaler.inverse_transform(np.zeros((1, len(self.X_scaler.mean_)))),
                columns=self.param_cols + self.attribute_cols
            )
            
            # Update fixed_attributes with mean values for missing attributes
            for attr in missing_attrs:
                fixed_attributes[attr] = X_means[attr].iloc[0]
        
        # Define objective function (negative because we want to maximize)
        def objective(params_array):
            # Create a full input vector with parameters and fixed attributes
            input_data = {}
            
            # Add parameters
            for i, param in enumerate(self.param_cols):
                input_data[param] = params_array[i]
            
            # Add fixed attributes
            for attr, value in fixed_attributes.items():
                input_data[attr] = value
            
            # Convert to DataFrame with correct columns order
            X_pred = pd.DataFrame([input_data])[self.param_cols + self.attribute_cols]
            
            # Scale the input
            X_pred_scaled = self.X_scaler.transform(X_pred)
            
            # Predict the metric and return negative (for minimization)
            return -self.model.predict(X_pred_scaled)[0]
        
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
        
        self.logger.info(f"Parameter optimization completed with predicted {self.target_metric}: {best_score:.4f}")
        
        return optimized_params, best_score
    
    def _generate_trial_params_file(self, optimized_params):
        """
        Generate a trialParams.nc file with the optimized parameters.
        
        Args:
            optimized_params: Dictionary of optimized parameter values
        """
        self.logger.info("Generating trialParams.nc file with optimized parameters")
        
        # Get attribute file path for reading HRU information
        attr_file = self.project_dir / "settings" / "SUMMA" / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        if not attr_file.exists():
            self.logger.error(f"Attribute file not found: {attr_file}")
            return
        
        try:
            # Read HRU information from the attributes file
            with xr.open_dataset(attr_file) as ds:
                hru_ids = ds['hruId'].values
                hru2gru_ids = ds['hru2gruId'].values
                
                # Create the trial parameters dataset
                output_ds = xr.Dataset()
                
                # Add dimensions
                output_ds['hru'] = ('hru', np.arange(len(hru_ids)))
                
                # Add HRU ID variable
                output_ds['hruId'] = ('hru', hru_ids)
                output_ds['hruId'].attrs['long_name'] = 'Hydrologic Response Unit ID (HRU)'
                output_ds['hruId'].attrs['units'] = '-'
                
                # Add parameter variables
                for param_name, param_value in optimized_params.items():
                    # Create array with the parameter value for each HRU
                    param_array = np.full(len(hru_ids), param_value)
                    
                    # Add to dataset
                    output_ds[param_name] = ('hru', param_array)
                    output_ds[param_name].attrs['long_name'] = f"Trial value for {param_name}"
                    output_ds[param_name].attrs['units'] = "N/A"
                
                # Add global attributes
                output_ds.attrs['description'] = "SUMMA Trial Parameter file generated by CONFLUENCE RandomForestEmulator"
                output_ds.attrs['history'] = f"Created on {pd.Timestamp.now().isoformat()}"
                output_ds.attrs['confluence_experiment_id'] = self.experiment_id
                
                # Save to file
                output_path = self.emulation_output_dir / "trialParams.nc"
                output_ds.to_netcdf(output_path)
                
                self.logger.info(f"Trial parameters file saved to: {output_path}")
                
                # Copy to SUMMA settings directory for use in model runs
                summa_settings_dir = self.project_dir / "settings" / "SUMMA"
                target_path = summa_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
                
                import shutil
                shutil.copy2(output_path, target_path)
                self.logger.info(f"Copied trial parameters to SUMMA settings directory: {target_path}")
                
                return target_path
                
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
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
            
    def metrics_to_string(self, metrics):
        """Format metrics dictionary as a string for plot titles."""
        return f"KGE: {metrics['KGE']:.3f}, NSE: {metrics['NSE']:.3f}, RMSE: {metrics['RMSE']:.3f} m³/s"

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
            #self.logger.info(f"Opening shapefile for area calculation: {basin_shapefile}")
            gdf = gpd.read_file(basin_shapefile)
            
            # Log the available columns
            #self.logger.info(f"Available columns in shapefile: {gdf.columns.tolist()}")
            
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

    def _evaluate_summa_output(self):
        """
        Evaluate SUMMA and mizuRoute outputs by comparing to observed data.
        """
        self.logger.info("Evaluating model outputs")
        
        # Define paths
        if self.config.get('DOMAIN_DEFINITION_METHOD') != 'lumped':
            # For distributed models, use mizuRoute output
            sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "mizuRoute"
            sim_files = list(sim_dir.glob("*.nc"))
            
            if not sim_files:
                self.logger.warning("No mizuRoute output files found for evaluation")
                return
            
            sim_file = sim_files[0]
        else:
            # For lumped models, use SUMMA output directly
            sim_dir = self.project_dir / "simulations" / f"{self.experiment_id}_rf_optimized" / "SUMMA"
            sim_files = list(sim_dir.glob(f"{self.experiment_id}_rf_optimized*.nc"))
            
            if not sim_files:
                self.logger.warning("No SUMMA output files found for evaluation")
                return
            
            sim_file = sim_files[0]
        
        # Get observed data path
        obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not obs_path.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_path}")
            return
        
        try:
            import pandas as pd
            import xarray as xr
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            # Read observed data - first check column names
            with open(obs_path, 'r') as f:
                header_line = f.readline().strip()
            
            header_cols = header_line.split(',')
            self.logger.info(f"Observed data columns: {header_cols}")
            
            # Identify date column
            date_col = next((col for col in header_cols if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower()), None)
            if not date_col:
                self.logger.warning(f"No date column found in observed data. Available columns: {header_cols}")
                return
            
            # Identify flow column 
            flow_col = next((col for col in header_cols if 'flow' in col.lower() or 'discharge' in col.lower() or 'q_' in col.lower()), None)
            if not flow_col:
                self.logger.warning(f"No flow column found in observed data. Available columns: {header_cols}")
                return
            
            # Now read the CSV with known column names
            obs_df = pd.read_csv(obs_path, parse_dates=[date_col])
            
            # Set the date column as index
            obs_df.set_index(date_col, inplace=True)
            observed_flow = obs_df[flow_col]
            
            # Extract simulated streamflow
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
                                return
                        else:
                            self.logger.error(f"Reach ID {sim_reach_id} not found in mizuRoute output")
                            return
                    else:
                        self.logger.error("No reachID variable found in mizuRoute output")
                        return
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
                        return
                    
                    # Extract the variable
                    if 'gru' in ds[streamflow_var].dims and ds.dims['gru'] > 1:
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
            
            # Get the time period where both datasets have data
            common_idx = obs_df.index.intersection(sim_flow.index)
            if len(common_idx) == 0:
                self.logger.error("No common time period between observed and simulated flows")
                return
            
            # Align the datasets
            obs_aligned = observed_flow.loc[common_idx]
            sim_aligned = sim_flow.loc[common_idx]
            
            # Calculate performance metrics
            metrics = self._calculate_streamflow_metrics(obs_aligned, sim_aligned)
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(self.emulation_output_dir / "performance_metrics_rf_optimized.csv", index=False)
            
            # Create visualization
            plt.figure(figsize=(16, 10))
            
            # Plot time series
            plt.subplot(2, 1, 1)
            plt.plot(obs_aligned.index, obs_aligned, 'g-', linewidth=1.5, label='Observed')
            plt.plot(sim_aligned.index, sim_aligned, 'b-', linewidth=1.5, label='Simulated (RF Optimized)')
            
            # Add labels and legend
            plt.xlabel('Date')
            plt.ylabel('Streamflow (m³/s)')
            plt.title(f'RF-Optimized Streamflow Comparison\n{self.metrics_to_string(metrics)}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Plot scatter
            plt.subplot(2, 1, 2)
            plt.scatter(obs_aligned, sim_aligned, alpha=0.5)
            
            # Add 1:1 line
            max_val = max(obs_aligned.max(), sim_aligned.max())
            min_val = min(obs_aligned.min(), sim_aligned.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
            
            plt.xlabel('Observed Streamflow (m³/s)')
            plt.ylabel('Simulated Streamflow (m³/s)')
            plt.title('Scatter Plot')
            plt.grid(True, alpha=0.3)
            
            # Equal aspect ratio
            plt.axis('equal')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(self.emulation_output_dir / "observed_vs_simulated_rf_optimized.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Evaluation complete, results saved to: {self.emulation_output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error evaluating model outputs: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

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
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan}
        
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
    
    def _plot_observed_vs_simulated(self, observed, simulated, metrics):
        """
        Create visualization of observed vs. simulated streamflow.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
            metrics: Dictionary of performance metrics
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot time series
        plt.subplot(2, 1, 1)
        plt.plot(observed.index, observed, 'b-', label='Observed', linewidth=1)
        plt.plot(simulated.index, simulated, 'r-', label='Simulated (RF Optimized)', linewidth=1)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Observed vs. Simulated Streamflow')
        
        # Add metrics to plot
        metric_text = (
            f"KGE: {metrics['KGE']:.3f}\n"
            f"NSE: {metrics['NSE']:.3f}\n"
            f"RMSE: {metrics['RMSE']:.3f} m³/s\n"
            f"PBIAS: {metrics['PBIAS']:.1f}%"
        )
        plt.text(
            0.02, 0.98, metric_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot scatter
        plt.subplot(2, 1, 2)
        plt.scatter(observed, simulated, alpha=0.5)
        
        # Add 1:1 line
        max_val = max(observed.max(), simulated.max())
        min_val = min(observed.min(), simulated.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        
        plt.xlabel('Observed Streamflow (m³/s)')
        plt.ylabel('Simulated Streamflow (m³/s)')
        plt.title('Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        plt.axis('equal')
        
        # Add metrics to plot
        plt.text(
            0.02, 0.98, metric_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.emulation_output_dir / "observed_vs_simulated.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create flow duration curve
        self._plot_flow_duration_curve(observed, simulated)
    
    def _plot_flow_duration_curve(self, observed, simulated):
        """
        Create flow duration curve visualization.
        
        Args:
            observed: Series of observed streamflow values
            simulated: Series of simulated streamflow values
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Sort flows in descending order
        obs_sorted = observed.sort_values(ascending=False)
        sim_sorted = simulated.sort_values(ascending=False)
        
        # Calculate exceedance probability
        obs_exceed = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
        sim_exceed = np.arange(1, len(sim_sorted) + 1) / (len(sim_sorted) + 1)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot the FDCs
        plt.plot(obs_exceed, obs_sorted, 'b-', label='Observed')
        plt.plot(sim_exceed, sim_sorted, 'r-', label='Simulated (RF Optimized)')
        
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
        plt.savefig(self.emulation_output_dir / "flow_duration_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_workflow(self):
        """
        Run the complete random forest emulation workflow:
        1. Load and prepare data
        2. Train the random forest model
        3. Optimize parameters
        4. Run SUMMA with optimized parameters
        5. Evaluate results
        
        Returns:
            Dictionary with workflow results
        """
        self.logger.info("Starting random forest emulation workflow")
        
        try:
            # Step 1: Load and prepare data
            X, y = self.load_and_prepare_data()
            self.logger.info(f"Data prepared with {X.shape[1]} features and {len(y)} samples")
            
            # Step 2: Train the random forest model
            model, metrics = self.train_random_forest(X, y)
            self.logger.info(f"Model trained with R² score: {metrics['test_score']:.4f}")
            
            # Step 3: Optimize parameters
            # Get fixed attribute values (from mean of training data)
            X_means = pd.DataFrame(
                self.X_scaler.inverse_transform(np.zeros((1, len(self.X_scaler.mean_)))),
                columns=X.columns
            )
            
            fixed_attributes = {attr: X_means[attr].iloc[0] for attr in self.attribute_cols}
            
            optimized_params, predicted_score = self.optimize_parameters(fixed_attributes)
            self.logger.info(f"Parameters optimized with predicted {self.target_metric}: {predicted_score:.4f}")
            
            # Step 4: Run SUMMA with optimized parameters
            summa_success = self.run_summa_with_optimal_parameters()
            self.logger.info(f"SUMMA run {'successful' if summa_success else 'failed'}")
            
            # Return results
            results = {
                'model_metrics': metrics,
                'optimized_parameters': optimized_params,
                'predicted_score': predicted_score,
                'summa_success': summa_success,
                'output_dir': str(self.emulation_output_dir)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in random forest emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
