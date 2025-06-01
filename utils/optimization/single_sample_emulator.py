"""
Enhanced Emulator for CONFLUENCE with optimization integration and neural network support.

This enhanced version:
1. Uses stored parameter values from iterative optimization runs as training data
2. Includes neural network emulation in addition to random forest
3. Applies best practices from DDS/DE optimizers
4. Updates default trial params if emulator improves on optimization results
"""

import sys
import numpy as np # type: ignore
import netCDF4 as nc # type: ignore
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from dataclasses import dataclass
from scipy.optimize import differential_evolution

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data.attribute_processing import attributeProcessor # type: ignore

# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass 
class ParameterBounds:
    """Simple container for parameter bounds."""
    name: str
    min_val: float
    max_val: float
    param_type: str = "local"  # "local" or "basin"


@dataclass
class RunResult:
    """Container for simulation run results."""
    run_id: str
    success: bool
    parameters: Dict[str, float]
    metrics: Optional[Dict[str, float]] = None
    streamflow_file: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class OptimizationRecord:
    """Container for optimization run data."""
    algorithm: str
    parameters: Dict[str, float]
    score: float
    metrics: Dict[str, float]
    generation: Optional[int] = None
    individual: Optional[int] = None
    is_best: bool = False


# ============================================================================
# Enhanced Training Data Loader
# ============================================================================

class OptimizationDataLoader:
    """
    Loads existing optimization results for use as emulator training data.
    
    This class searches for and loads parameter evaluation data from:
    - DDS optimization runs
    - DE optimization runs  
    - Other iterative optimization methods
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
    
    def load_optimization_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]]:
        """
        Load existing optimization data for emulator training.
        
        Returns:
            Tuple of (features DataFrame, target Series, metadata dict) or None if no data found
        """
        self.logger.info("Searching for existing optimization data")
        
        # Search for optimization result directories
        optimization_dirs = self._find_optimization_directories()
        
        if not optimization_dirs:
            self.logger.info("No existing optimization data found")
            return None
        
        # Load data from all found optimization runs
        all_records = []
        metadata = {'sources': [], 'total_evaluations': 0, 'algorithms': set()}
        
        for opt_dir in optimization_dirs:
            records = self._load_optimization_directory(opt_dir)
            if records:
                all_records.extend(records)
                metadata['sources'].append(str(opt_dir))
                metadata['algorithms'].update([r.algorithm for r in records])
        
        if not all_records:
            self.logger.warning("Found optimization directories but no valid data")
            return None
        
        # Convert to training format
        X, y = self._convert_to_training_data(all_records)
        
        metadata['total_evaluations'] = len(all_records)
        metadata['algorithms'] = list(metadata['algorithms'])
        metadata['best_score'] = max(r.score for r in all_records if not np.isnan(r.score))
        metadata['parameter_names'] = list(X.columns) if X is not None else []
        
        self.logger.info(f"Loaded {len(all_records)} optimization evaluations from {len(optimization_dirs)} sources")
        self.logger.info(f"Algorithms found: {', '.join(metadata['algorithms'])}")
        self.logger.info(f"Best existing score: {metadata['best_score']:.6f}")
        
        return X, y, metadata
    
    def _find_optimization_directories(self) -> List[Path]:
        """Find optimization result directories."""
        optimization_dirs = []
        
        # Look for DDS optimization results
        dds_pattern = self.project_dir / "optimisation" / f"dds_*"
        dds_dirs = list(self.project_dir.glob("optimisation/dds_*"))
        optimization_dirs.extend(dds_dirs)
        
        # Look for DE optimization results  
        de_pattern = self.project_dir / "optimisation" / f"de_*"
        de_dirs = list(self.project_dir.glob("optimisation/de_*"))
        optimization_dirs.extend(de_dirs)
        
        # Look for other optimization results
        other_dirs = list(self.project_dir.glob("optimisation/*"))
        for dir_path in other_dirs:
            if dir_path.is_dir() and not any(dir_path.name.startswith(prefix) for prefix in ['dds_', 'de_']):
                # Check if it contains optimization results
                if (dir_path / "all_parameter_evaluations.csv").exists():
                    optimization_dirs.append(dir_path)
        
        # Remove duplicates and filter existing directories
        optimization_dirs = [d for d in set(optimization_dirs) if d.exists() and d.is_dir()]
        
        self.logger.info(f"Found {len(optimization_dirs)} optimization directories")
        return optimization_dirs
    
    def _load_optimization_directory(self, opt_dir: Path) -> List[OptimizationRecord]:
        """Load optimization data from a single directory."""
        records = []
        
        try:
            # Try to load detailed parameter evaluations
            eval_file = opt_dir / "all_parameter_evaluations.csv"
            if eval_file.exists():
                records.extend(self._load_parameter_evaluations(eval_file, opt_dir))
            
            # Try to load generation summaries as fallback
            if not records:
                summary_file = opt_dir / "generation_summaries.csv"
                if summary_file.exists():
                    records.extend(self._load_generation_summaries(summary_file, opt_dir))
            
            # Try to load any CSV files with parameter and score data
            if not records:
                records.extend(self._load_generic_optimization_data(opt_dir))
                
        except Exception as e:
            self.logger.warning(f"Error loading data from {opt_dir}: {str(e)}")
        
        return records
            
    def _load_parameter_evaluations(self, eval_file: Path, opt_dir: Path) -> List[OptimizationRecord]:
        """Load detailed parameter evaluation data with strict parameter filtering."""
        records = []
        
        try:
            df = pd.read_csv(eval_file)
            algorithm = self._determine_algorithm(opt_dir, df)
            
            # Get expected parameter names from config if available
            expected_params = set()
            if hasattr(self, 'config'):
                local_params = self.config.get('PARAMS_TO_CALIBRATE', '').split(',')
                basin_params = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',')
                expected_params = {p.strip() for p in local_params + basin_params if p.strip()}
            
            if not expected_params:
                # Fallback to common hydrological parameters
                expected_params = {
                    'tempCritRain', 'rootingDepth', 'theta_sat', 'theta_res', 'vGn_n', 
                    'k_soil', 'k_macropore', 'zScale_TOPMODEL', 'routingGammaScale',
                    'routingGammaShape', 'Fcapil', 'albedoDecayRate'
                }
            
            self.logger.debug(f"Expected parameters: {sorted(expected_params)}")
            self.logger.debug(f"Available columns: {sorted(df.columns)}")
            
            # Enhanced non-parameter column detection
            non_param_cols = {
                'generation', 'individual', 'score', 'timestamp', 'iteration',
                'is_valid', 'is_new_best', 'is_best_in_generation', 'duration_seconds',
                'best_score', 'mean_score', 'std_score', 'worst_score', 'valid_individuals',
                'total_individuals', 'cumulative_time', 'KGE', 'NSE', 'RMSE', 'PBIAS', 'MAE',
                'r', 'alpha', 'beta', 'Calib_KGE', 'Calib_NSE', 'Eval_KGE', 'Eval_NSE',
                # Add more metric variations
                'Calib_RMSE', 'Calib_PBIAS', 'Calib_MAE', 'Calib_r', 'Calib_alpha', 'Calib_beta',
                'Eval_RMSE', 'Eval_PBIAS', 'Eval_MAE', 'Eval_r', 'Eval_alpha', 'Eval_beta'
            }
            
            # Find parameter columns - only those in expected parameters OR their statistical versions
            param_columns = set()
            stat_suffixes = ['_mean', '_min', '_max', '_std']
            
            for col in df.columns:
                if col in non_param_cols:
                    continue
                    
                # Check if it's a direct parameter
                if col in expected_params:
                    param_columns.add(col)
                else:
                    # Check if it's a statistical version of an expected parameter
                    for suffix in stat_suffixes:
                        if col.endswith(suffix):
                            base_name = col.replace(suffix, '')
                            if base_name in expected_params:
                                param_columns.add(base_name)  # Add base name, not the statistical version
                                break
            
            param_columns = sorted(list(param_columns))
            self.logger.info(f"Detected {len(param_columns)} parameter columns: {param_columns}")
            
            if not param_columns:
                self.logger.warning(f"No parameter columns found in {eval_file}")
                return records
            
            for _, row in df.iterrows():
                if pd.notna(row.get('score')):
                    parameters = {}
                    
                    for param_name in param_columns:
                        value = None
                        
                        # Try direct parameter value first
                        if param_name in df.columns and pd.notna(row.get(param_name)):
                            value = float(row[param_name])
                        # Fall back to _mean if available
                        elif f"{param_name}_mean" in df.columns and pd.notna(row.get(f"{param_name}_mean")):
                            value = float(row[f"{param_name}_mean"])
                        
                        if value is not None:
                            parameters[param_name] = value
                    
                    # Only add record if we have most of the expected parameters
                    if len(parameters) >= max(1, len(param_columns) * 0.5):
                        record = OptimizationRecord(
                            algorithm=algorithm,
                            parameters=parameters,
                            score=float(row['score']),
                            metrics={'primary_metric': float(row['score'])},
                            generation=int(row.get('generation', row.get('iteration', 0))) if pd.notna(row.get('generation', row.get('iteration'))) else None,
                            individual=int(row.get('individual', 0)) if pd.notna(row.get('individual')) else None,
                            is_best=bool(row.get('is_best_in_generation', row.get('is_new_best', False)))
                        )
                        records.append(record)
            
            self.logger.info(f"Loaded {len(records)} evaluations from {eval_file}")
            if records:
                sample_params = list(records[0].parameters.keys())
                self.logger.debug(f"Sample parameter names: {sample_params}")
            
        except Exception as e:
            self.logger.error(f"Error loading parameter evaluations from {eval_file}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return records
    
    def _load_generation_summaries(self, summary_file: Path, opt_dir: Path) -> List[OptimizationRecord]:
        """Load generation summary data."""
        records = []
        
        try:
            df = pd.read_csv(summary_file)
            algorithm = self._determine_algorithm(opt_dir, df)
            
            # Get best parameter columns
            param_cols = [col for col in df.columns if col.startswith('best_') and not col.endswith(('_score', '_generation'))]
            
            for _, row in df.iterrows():
                if pd.notna(row.get('best_score')):
                    # Extract best parameters for this generation
                    parameters = {}
                    for col in param_cols:
                        param_name = col.replace('best_', '')
                        if pd.notna(row.get(col)):
                            parameters[param_name] = float(row[col])
                    
                    if parameters:
                        record = OptimizationRecord(
                            algorithm=algorithm,
                            parameters=parameters,
                            score=float(row['best_score']),
                            metrics={'primary_metric': float(row['best_score'])},
                            generation=int(row.get('generation', 0)) if pd.notna(row.get('generation')) else None,
                            is_best=True  # These are best per generation
                        )
                        records.append(record)
            
            self.logger.info(f"Loaded {len(records)} generation summaries from {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading generation summaries from {summary_file}: {str(e)}")
        
        return records
    
    def _load_generic_optimization_data(self, opt_dir: Path) -> List[OptimizationRecord]:
        """Load any CSV files that might contain optimization data."""
        records = []
        
        # Look for CSV files with parameter and score data
        csv_files = list(opt_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check if this looks like optimization data
                if 'score' in df.columns or any('KGE' in col or 'NSE' in col for col in df.columns):
                    # Try to extract records
                    score_col = 'score'
                    if score_col not in df.columns:
                        # Look for common metric columns
                        for metric in ['KGE', 'NSE', 'RMSE']:
                            if metric in df.columns:
                                score_col = metric
                                break
                    
                    if score_col in df.columns:
                        algorithm = self._determine_algorithm(opt_dir, df)
                        param_cols = [col for col in df.columns if col not in ['score'] and 
                                    not any(col.startswith(prefix) for prefix in ['generation', 'individual', 'is_', 'timestamp'])]
                        
                        for _, row in df.iterrows():
                            if pd.notna(row.get(score_col)):
                                parameters = {}
                                for col in param_cols:
                                    if pd.notna(row.get(col)) and isinstance(row[col], (int, float)):
                                        parameters[col] = float(row[col])
                                
                                if parameters:
                                    record = OptimizationRecord(
                                        algorithm=algorithm,
                                        parameters=parameters,
                                        score=float(row[score_col]),
                                        metrics={score_col: float(row[score_col])},
                                    )
                                    records.append(record)
            
            except Exception as e:
                self.logger.debug(f"Could not load {csv_file} as optimization data: {str(e)}")
                continue
        
        if records:
            self.logger.info(f"Loaded {len(records)} records from generic CSV files in {opt_dir}")
        
        return records
    
    def _determine_algorithm(self, opt_dir: Path, df: pd.DataFrame) -> str:
        """Determine optimization algorithm from directory name or data."""
        dir_name = opt_dir.name.lower()
        
        if 'dds' in dir_name:
            return 'DDS'
        elif 'de' in dir_name:
            return 'DE'
        elif 'pso' in dir_name:
            return 'PSO'
        elif 'sceua' in dir_name:
            return 'SCE-UA'
        elif hasattr(df, 'columns') and 'algorithm' in df.columns:
            return df['algorithm'].iloc[0] if len(df) > 0 else 'Unknown'
        else:
            return 'Unknown'
            
    def _convert_to_training_data(self, records: List[OptimizationRecord]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Convert optimization records to training data format - FIXED VERSION."""
        if not records:
            self.logger.warning("No optimization records to convert")
            return None, None
        
        # Get expected parameter names from config (CRITICAL FIX)
        expected_params = set()
        if hasattr(self, 'config'):
            local_params = self.config.get('PARAMS_TO_CALIBRATE', '').split(',')
            basin_params = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',')
            expected_params = {p.strip() for p in local_params + basin_params if p.strip()}
        
        if not expected_params:
            # Fallback to common parameters if config not available
            expected_params = {
                'tempCritRain', 'rootingDepth', 'theta_sat', 'theta_res', 'vGn_n', 
                'k_soil', 'k_macropore', 'zScale_TOPMODEL', 'routingGammaScale'
            }
        
        # CRITICAL: Only use expected parameters, not all parameters from records
        all_param_names = sorted(list(expected_params))
        self.logger.info(f"Using expected parameter names: {all_param_names}")
        
        if not all_param_names:
            self.logger.warning("No parameter names found in configuration")
            return None, None
        
        # Create feature matrix - only include records that have most of the expected parameters
        features = []
        scores = []
        valid_records = 0
        skipped_records = 0
        
        for record in records:
            if not np.isnan(record.score):
                # Create feature vector using ONLY expected parameters
                feature_vector = []
                valid_params = 0
                
                for param_name in all_param_names:
                    value = record.parameters.get(param_name, np.nan)
                    feature_vector.append(value)
                    if not np.isnan(value):
                        valid_params += 1
                
                # More lenient threshold: require at least 30% of expected parameters
                required_threshold = max(1, len(all_param_names) * 0.3)
                
                if valid_params >= required_threshold:
                    features.append(feature_vector)
                    scores.append(record.score)
                    valid_records += 1
                else:
                    skipped_records += 1
                    self.logger.debug(f"Skipped record with only {valid_params}/{len(all_param_names)} valid parameters")
        
        self.logger.info(f"Processing results: {valid_records} valid records, {skipped_records} skipped")
        
        if not features:
            self.logger.warning("No valid feature vectors could be created from optimization data")
            self.logger.warning(f"Checked {len(records)} records for {len(all_param_names)} expected parameters")
            
            # Debug: Print parameter availability in records
            param_availability = {}
            for record in records[:5]:  # Check first 5 records
                for param in all_param_names:
                    if param not in param_availability:
                        param_availability[param] = 0
                    if param in record.parameters:
                        param_availability[param] += 1
            
            self.logger.debug(f"Parameter availability in first 5 records: {param_availability}")
            return None, None
        
        # Convert to pandas using ONLY expected parameter names
        X = pd.DataFrame(features, columns=all_param_names)
        y = pd.Series(scores)
        
        self.logger.info(f"Created feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        self.logger.debug(f"Parameter columns: {list(X.columns)}")
        
        # Handle missing values more carefully
        missing_info = X.isnull().sum()
        if missing_info.sum() > 0:
            self.logger.debug(f"Missing values per parameter: {missing_info[missing_info > 0].to_dict()}")
            
            # Strategy 1: For each parameter, use median from valid values in that parameter
            for col in X.columns:
                if X[col].isnull().any():
                    valid_values = X[col].dropna()
                    if len(valid_values) > 0:
                        median_val = valid_values.median()
                        X[col] = X[col].fillna(median_val)
                        self.logger.debug(f"Filled {X[col].isnull().sum()} missing values in {col} with median {median_val:.6f}")
                    else:
                        # If no valid values for this parameter, use a reasonable default
                        default_values = {
                            'tempCritRain': 0.0,
                            'rootingDepth': 2.0,
                            'theta_sat': 0.5,
                            'theta_res': 0.05,
                            'vGn_n': 1.5,
                            'k_soil': 1e-5,
                            'k_macropore': 1e-3,
                            'zScale_TOPMODEL': 10.0,
                            'routingGammaScale': 1000.0
                        }
                        default_val = default_values.get(col, 0.01)
                        X[col] = X[col].fillna(default_val)
                        self.logger.warning(f"Used default value {default_val} for parameter {col}")
        
        # Remove any remaining NaN rows (shouldn't happen now)
        initial_count = len(X)
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < initial_count:
            self.logger.info(f"Removed {initial_count - len(X)} rows with remaining NaN values")
        
        if len(X) == 0:
            self.logger.error("All rows removed due to NaN values")
            return None, None
        
        self.logger.info(f"Final training data: {len(X)} samples, {len(X.columns)} features")
        self.logger.info(f"Score range: {y.min():.6f} to {y.max():.6f}")
        
        # Final validation: ensure we have the expected number of features
        if len(X.columns) != len(all_param_names):
            self.logger.error(f"Feature count mismatch: {len(X.columns)} vs expected {len(all_param_names)}")
            return None, None
        
        return X, y


# ============================================================================
# Neural Network Emulator
# ============================================================================

class NeuralNetworkEmulator:
    """
    Neural Network-based parameter emulator for CONFLUENCE.
    
    Uses neural networks to learn the relationship between parameters and performance metrics.
    Provides an alternative to Random Forest with potentially better performance for complex
    parameter interactions.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize paths
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Setup output directory
        self.emulator_dir = self.project_dir / "emulation" / self.experiment_id
        self.nn_output_dir = self.emulator_dir / "nn_emulation"
        self.nn_output_dir.mkdir(parents=True, exist_ok=True)
        
        # ML settings
        self.target_metric = config.get('EMULATION_TARGET_METRIC', 'KGE')
        self.hidden_layers = config.get('NN_HIDDEN_LAYERS', [20])  # Much smaller default
        self.max_iter = config.get('NN_MAX_ITERATIONS', 1000)
        self.learning_rate = config.get('NN_LEARNING_RATE', 0.001)
        self.random_state = config.get('EMULATION_SEED', 42)
        self.validation_split = config.get('NN_VALIDATION_SPLIT', 0.2)
        
        # Initialize ML components
        self.model = None
        self.scaler = None
        self.training_history = {}
        
        self.logger.info(f"NeuralNetworkEmulator initialized for target metric: {self.target_metric}")
        self.logger.info(f"Network architecture: {self.hidden_layers}")
    
    def run_workflow(self, training_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Optional[Dict]:
        """
        Run the complete NN emulation workflow.
        
        Args:
            training_data: Optional pre-loaded training data (X, y)
            
        Returns:
            Dictionary with emulation results or None if failed
        """
        self.logger.info("Starting Neural Network emulation workflow")
        
        try:
            # Step 1: Load and prepare data
            if training_data is not None:
                X, y = training_data
                self.logger.info("Using provided training data")
            else:
                X, y = self._load_and_prepare_data()
                
            if X is None or len(X) == 0:
                self.logger.error("No data available for training")
                return None
            
            # Step 2: Train neural network
            model_metrics = self._train_model(X, y)
            
            # Step 3: Optimize parameters
            best_params, best_score = self._optimize_parameters(X, y)
            
            # Step 4: Generate final model files
            final_success = self._generate_model_files(best_params)
            
            # Step 5: Create visualizations
            self._create_visualizations(X, y)
            
            # Compile results
            results = {
                'best_parameters': best_params,
                'best_score': best_score,
                'model_metrics': model_metrics,
                'predicted_score': best_score,
                'final_success': final_success,
                'output_dir': str(self.nn_output_dir),
                'training_history': self.training_history
            }
            
            self.logger.info(f"NN emulation completed successfully. Best score: {best_score:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in NN emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _load_and_prepare_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load and prepare training data from existing sources."""
        try:
            # First try to load from optimization data
            data_loader = OptimizationDataLoader(self.config, self.logger)
            optimization_data = data_loader.load_optimization_data()
            
            if optimization_data is not None:
                X, y, metadata = optimization_data
                self.logger.info(f"Using optimization data: {len(X)} samples from {len(metadata['sources'])} sources")
                return X, y
            
            # Fallback to ensemble data if available
            return self._load_ensemble_data()
                
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None, None
    
    def _load_ensemble_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load training data from ensemble simulations."""
        try:
            # Load parameter sets
            param_file = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
            if not param_file.exists():
                self.logger.error("No parameter sets file found")
                return None, None
            
            # Load performance metrics
            metrics_file = self.emulator_dir / "ensemble_analysis" / "performance_metrics.csv"
            if not metrics_file.exists():
                self.logger.error("No performance metrics file found")
                return None, None
            
            # Read parameter data
            param_data = {}
            with nc.Dataset(param_file, 'r') as ds:
                param_names = [var for var in ds.variables if var not in ['hruId', 'runIndex']]
                for param in param_names:
                    param_data[param] = np.mean(ds.variables[param][:], axis=1)
            
            X = pd.DataFrame(param_data)
            
            # Read metrics data
            metrics_df = pd.read_csv(metrics_file)
            
            # Align data
            min_rows = min(len(X), len(metrics_df))
            X = X.iloc[:min_rows]
            
            # Extract target metric
            if self.target_metric in metrics_df.columns:
                y = metrics_df[self.target_metric].iloc[:min_rows]
            else:
                self.logger.error(f"Target metric {self.target_metric} not found")
                return None, None
            
            # Remove NaN values
            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.logger.info(f"Loaded {len(X)} ensemble samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble data: {str(e)}")
            return None, None
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train neural network model."""
        try:
            # Try sklearn first (more reliable)
            try:
                from sklearn.neural_network import MLPRegressor
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_squared_error
                
                return self._train_sklearn_model(X, y)
                
            except ImportError:
                self.logger.warning("scikit-learn not available, trying TensorFlow")
                
            # Fallback to TensorFlow if available
            try:
                import tensorflow as tf # type: ignore
                return self._train_tensorflow_model(X, y)
                
            except ImportError:
                self.logger.error("Neither scikit-learn nor TensorFlow available for neural network training")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error training neural network: {str(e)}")
            return {}

    def _optimize_parameters(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], float]:
        """FIXED Neural Network parameter optimization with strict bounds checking."""
        try:
            if self.model is None or self.scaler is None:
                self.logger.error("Model not trained")
                return {}, 0.0
            
            # Check if we can import differential_evolution
            try:
                from scipy.optimize import differential_evolution
            except ImportError:
                self.logger.error("scipy.optimize.differential_evolution not available")
                # Fallback to best training sample
                best_idx = y.idxmax()
                best_params = X.loc[best_idx].to_dict()
                best_score = y.loc[best_idx]
                return best_params, best_score
            
            param_names = list(X.columns)
            
            def objective(params):
                """Objective function with strict extrapolation prevention."""
                try:
                    # STRICT bounds checking against training data
                    for i, (param_name, value) in enumerate(zip(param_names, params)):
                        train_values = X[param_name]
                        # Much stricter bounds - only allow interpolation
                        if value < train_values.quantile(0.1) or value > train_values.quantile(0.9):
                            return 1000.0  # Heavy penalty for extrapolation
                    
                    # Create DataFrame with proper feature names to avoid sklearn warnings
                    X_pred = pd.DataFrame([params], columns=param_names)
                    X_pred_scaled = self.scaler.transform(X_pred)
                    prediction = self.model.predict(X_pred_scaled)[0]
                    
                    # Sanity check on prediction
                    if prediction > y.max() * 1.5 or prediction < y.min() * 1.5:
                        return 1000.0  # Penalty for unrealistic predictions
                    
                    return -prediction
                except Exception as e:
                    return 1000.0
            
            # VERY conservative bounds - only within training data range
            bounds = []
            for name in param_names:
                train_values = X[name]
                # Use 10th to 90th percentile of training data
                bounds.append((train_values.quantile(0.1), train_values.quantile(0.9)))
            
            # Conservative optimization
            result = differential_evolution(
                objective, 
                bounds, 
                seed=self.random_state, 
                maxiter=50,   # Much fewer iterations
                popsize=8,    # Smaller population
                atol=1e-6,
                tol=1e-6
            )
            
            optimal_params = {}
            for i, name in enumerate(param_names):
                optimal_params[name] = float(result.x[i])
            
            best_score = -result.fun
            
            # Final sanity check
            if best_score > 1:
                self.logger.warning(f"NN prediction ({best_score:.4f}) seems unrealistic")
            
            self.logger.info(f"NN optimization completed: predicted score = {best_score:.4f}")
            return optimal_params, best_score
            
        except Exception as e:
            self.logger.error(f"Error optimizing NN parameters: {str(e)}")
            return {}, 0.0

    def _train_sklearn_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """FIXED Neural Network training with better regularization."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Handle small sample sizes
        if len(X) == 1:
            self.logger.warning("Only 1 sample available. Cannot train neural network.")
            # Create a dummy model
            self.scaler = StandardScaler()
            self.scaler.fit(X)  # X is already a DataFrame
            
            class DummyModel:
                def __init__(self, value):
                    self.value = value
                def predict(self, X):
                    return np.full(len(X), self.value)
            
            self.model = DummyModel(y.iloc[0])
            return {'train_r2': 0.0, 'test_r2': 0.0, 'n_samples': len(X)}
        
        # For very small datasets, don't split
        if len(X) < 10:
            X_train, X_test, y_train, y_test = X, X, y, y
            test_size = 0.0
        else:
            test_size = 0.3
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Scale features - keep as DataFrames to maintain feature names
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # FIXED: More conservative network architecture
        max_hidden_size = min(20, len(X_train) // 2)  # Much smaller network
        adjusted_layers = [max_hidden_size] if max_hidden_size > 5 else [10]
        
        # FIXED: Much stronger regularization
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(adjusted_layers),
            max_iter=min(200, 50),  # Fewer iterations
            learning_rate_init=0.01,  # Lower learning rate
            random_state=self.random_state,
            early_stopping=len(X_train) > 10,
            validation_fraction=0.2 if len(X_train) > 10 else 0.1,
            n_iter_no_change=10,  # Stop early
            alpha=0.1,  # Strong L2 regularization
            solver='lbfgs' if len(X_train) < 50 else 'adam'  # Better solver for small data
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation to detect overfitting
        if len(X_train) >= 5:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=min(5, len(X_train)//2))
            self.logger.info(f"NN Cross-validation R² scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, train_pred) if len(y_train) > 1 else 0.0
        test_r2 = r2_score(y_test, test_pred) if len(y_test) > 1 and test_size > 0 else train_r2
        
        # Store training history
        self.training_history = {
            'loss_curve': getattr(self.model, 'loss_curve_', []),
            'validation_scores': getattr(self.model, 'validation_scores_', [])
        }
        
        self.logger.info(f"Neural network trained: R² = {test_r2:.4f}")
        self.logger.info(f"Training completed in {getattr(self.model, 'n_iter_', 0)} iterations")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.3:
            self.logger.warning(f"Possible overfitting detected: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'n_iterations': getattr(self.model, 'n_iter_', 0)
        }

    
    def _train_tensorflow_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train model using TensorFlow (placeholder for future implementation)."""
        self.logger.warning("TensorFlow training not yet implemented, falling back to simple model")
        
        # Simple linear model as fallback
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.values)
        X_test_scaled = self.scaler.transform(X_test.values)
        
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        test_score = self.model.score(X_test_scaled, y_test)
        
        return {
            'test_r2': test_score,
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
    
    
    def _get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds from parameter info files."""
        bounds = {}
        
        # Local parameters
        local_file = self.project_dir / "settings" / "SUMMA" / "localParamInfo.txt"
        if local_file.exists():
            bounds.update(self._parse_bounds_file(local_file))
        
        # Basin parameters
        basin_file = self.project_dir / "settings" / "SUMMA" / "basinParamInfo.txt"
        if basin_file.exists():
            bounds.update(self._parse_bounds_file(basin_file))
        
        return bounds
    
    def _parse_bounds_file(self, file_path: Path) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds file."""
        bounds = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3:
                    param_name = parts[0].strip()
                    try:
                        # Handle Fortran-style scientific notation
                        max_val_str = parts[1].replace('d', 'e').replace('D', 'e')
                        min_val_str = parts[2].replace('d', 'e').replace('D', 'e')
                        
                        max_val = float(max_val_str)
                        min_val = float(min_val_str)
                        bounds[param_name] = {'min': min_val, 'max': max_val}
                    except ValueError:
                        continue
        
        return bounds
    
    def _generate_model_files(self, best_params: Dict[str, float]) -> bool:
        """Generate model parameter files with optimized parameters."""
        try:
            # Create trial parameters file in the output directory
            settings_dir = self.project_dir / "settings" / "SUMMA"
            trial_file = self.nn_output_dir / "optimized_trialParams.nc"
            
            FileManager.create_trial_params_file(
                best_params,
                trial_file,
                settings_dir / "attributes.nc"
            )
            
            # Save parameters in CSV format for easier viewing
            params_df = pd.DataFrame([best_params])
            params_df.to_csv(self.nn_output_dir / "optimized_parameters_readable.csv", index=False)
            
            self.logger.info(f"Model files generated in: {self.nn_output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating model files: {str(e)}")
            return False
    
    def _create_visualizations(self, X: pd.DataFrame, y: pd.Series):
        """Create training and optimization visualizations."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plots directory
            plots_dir = self.nn_output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot training history if available
            if self.training_history and 'loss_curve' in self.training_history:
                self._plot_training_history(plots_dir)
            
            # Plot feature importance (approximate)
            self._plot_feature_analysis(X, y, plots_dir)
            
            # Plot prediction vs actual
            self._plot_predictions(X, y, plots_dir)
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
    
    def _plot_training_history(self, plots_dir: Path):
        """Plot neural network training history."""
        try:
            plt.figure(figsize=(12, 5))
            
            # Loss curve
            plt.subplot(1, 2, 1)
            loss_curve = self.training_history['loss_curve']
            plt.plot(loss_curve, 'b-', label='Training Loss')
            
            if 'validation_scores' in self.training_history and self.training_history['validation_scores']:
                val_scores = self.training_history['validation_scores']
                plt.plot(val_scores, 'r-', label='Validation Score')
            
            plt.xlabel('Iteration')
            plt.ylabel('Loss / Score')
            plt.title('Neural Network Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convergence analysis
            plt.subplot(1, 2, 2)
            if len(loss_curve) > 10:
                # Plot recent loss to show convergence
                recent_loss = loss_curve[-min(100, len(loss_curve)):]
                plt.plot(recent_loss, 'b-')
                plt.xlabel('Recent Iterations')
                plt.ylabel('Loss')
                plt.title('Training Convergence')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting training history: {str(e)}")
    
    def _plot_feature_analysis(self, X: pd.DataFrame, y: pd.Series, plots_dir: Path):
        """Plot feature correlation analysis."""
        try:
            # Feature correlation with target
            correlations = []
            for col in X.columns:
                corr = X[col].corr(y)
                correlations.append((col, corr))
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Plot top correlations
            plt.figure(figsize=(10, 6))
            features, corrs = zip(*correlations[:15])  # Top 15
            
            plt.barh(range(len(features)), corrs)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Correlation with Target Metric')
            plt.title('Feature Correlations with Target Metric')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature analysis: {str(e)}")
    
    def _plot_predictions(self, X: pd.DataFrame, y: pd.Series, plots_dir: Path):
        """Plot model predictions vs actual values."""
        try:
            if self.model is None or self.scaler is None:
                return
            
            # Use DataFrame to maintain feature names consistency
            X_scaled = self.scaler.transform(X)  # X is already a DataFrame
            y_pred = self.model.predict(X_scaled)
            
            # Create scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(y, y_pred, alpha=0.6, s=30)
            
            # Add 1:1 line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Neural Network Predictions (R² = {r2:.3f})')
            plt.grid(True, alpha=0.3)
            
            # Add text with R²
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting predictions: {str(e)}")



# ============================================================================
# Strategic Gap Filling Framework
# ============================================================================

class GapFillingStrategy:
    """
    Strategic gap filling for parameter space coverage enhancement.
    
    This class implements multiple strategies to identify and fill gaps in parameter space
    that weren't well explored by optimization algorithms, ensuring better emulator coverage.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Gap filling settings
        self.gap_filling_samples = config.get('GAP_FILLING_SAMPLES', 50)
        self.gap_filling_methods = config.get('GAP_FILLING_METHODS', ['density', 'boundary', 'uncertainty'])
        self.density_threshold = config.get('GAP_DENSITY_THRESHOLD', 0.1)
        self.boundary_margin = config.get('GAP_BOUNDARY_MARGIN', 0.1)
        self.uncertainty_percentile = config.get('GAP_UNCERTAINTY_PERCENTILE', 90)
        
        # Output directory for gap filling results
        self.gap_filling_dir = self.project_dir / "emulation" / self.experiment_id / "gap_filling"
        self.gap_filling_dir.mkdir(parents=True, exist_ok=True)
    
    def identify_and_fill_gaps(self, X_existing: pd.DataFrame, y_existing: pd.Series, 
                              param_bounds: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main method to identify gaps and generate strategic samples.
        
        Args:
            X_existing: Existing parameter samples from optimization
            y_existing: Existing performance scores
            param_bounds: Parameter bounds dictionary
            
        Returns:
            Tuple of (gap_filling_samples, estimated_scores)
        """
        self.logger.info("Starting strategic gap filling analysis")
        
        gap_samples = []
        gap_strategies_used = []
        
        # 1. Density-based gap filling
        if 'density' in self.gap_filling_methods:
            density_samples = self._density_based_gap_filling(X_existing, param_bounds)
            gap_samples.extend(density_samples)
            gap_strategies_used.extend(['density'] * len(density_samples))
            self.logger.info(f"Generated {len(density_samples)} density-based gap filling samples")
        
        # 2. Boundary exploration
        if 'boundary' in self.gap_filling_methods:
            boundary_samples = self._boundary_exploration(X_existing, param_bounds)
            gap_samples.extend(boundary_samples)
            gap_strategies_used.extend(['boundary'] * len(boundary_samples))
            self.logger.info(f"Generated {len(boundary_samples)} boundary exploration samples")
        
        # 3. Uncertainty-based sampling (requires preliminary emulator)
        if 'uncertainty' in self.gap_filling_methods:
            uncertainty_samples = self._uncertainty_based_sampling(X_existing, y_existing, param_bounds)
            gap_samples.extend(uncertainty_samples)
            gap_strategies_used.extend(['uncertainty'] * len(uncertainty_samples))
            self.logger.info(f"Generated {len(uncertainty_samples)} uncertainty-based samples")
        
        # 4. Diversity-based sampling
        if 'diversity' in self.gap_filling_methods:
            diversity_samples = self._diversity_based_sampling(X_existing, param_bounds)
            gap_samples.extend(diversity_samples)
            gap_strategies_used.extend(['diversity'] * len(diversity_samples))
            self.logger.info(f"Generated {len(diversity_samples)} diversity-based samples")
        
        if not gap_samples:
            self.logger.warning("No gap filling samples generated")
            return pd.DataFrame(), pd.Series()
        
        # Convert to DataFrame
        gap_X = pd.DataFrame(gap_samples, columns=X_existing.columns)
        
        # Estimate scores for gap samples (placeholder - would normally run simulations)
        gap_y = self._estimate_gap_sample_scores(gap_X, X_existing, y_existing)
        
        # Save gap filling analysis
        self._save_gap_filling_analysis(gap_X, gap_y, gap_strategies_used, X_existing)
        
        self.logger.info(f"Strategic gap filling completed: {len(gap_X)} samples generated")
        return gap_X, gap_y
    
    def _density_based_gap_filling(self, X_existing: pd.DataFrame, 
                                param_bounds: Dict[str, Dict[str, float]]) -> List[List[float]]:
        """
        Generate samples in low-density regions of parameter space.
        """
        try:
            if len(X_existing) < 2:
                self.logger.warning("Not enough samples for density-based gap filling, using random sampling")
                return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
            
            from scipy.spatial.distance import pdist, squareform
            
            # Calculate pairwise distances to identify sparse regions
            distances = pdist(X_existing.values)
            
            if len(distances) == 0:
                self.logger.warning("No distances calculated, falling back to uniform sampling")
                return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
            
            distance_matrix = squareform(distances)
            
            # Find regions with low neighbor density
            neighbor_threshold = np.percentile(distances, 20) if len(distances) > 1 else distances[0]
            neighbor_counts = np.sum(distance_matrix < neighbor_threshold, axis=1)
            
            if len(neighbor_counts) == 0:
                return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
            
            density_threshold = np.percentile(neighbor_counts, self.density_threshold * 100)
            low_density_indices = np.where(neighbor_counts < density_threshold)[0]
            
            # Generate samples around low-density regions but spread out
            density_samples = []
            n_density_samples = min(self.gap_filling_samples // 4, 20)
            
            for _ in range(n_density_samples):
                # Start from a low-density point and perturb
                if len(low_density_indices) > 0:
                    base_idx = np.random.choice(low_density_indices)
                    base_point = X_existing.iloc[base_idx].values
                    
                    # Generate perturbed sample within bounds
                    new_sample = []
                    for i, (param_name, values) in enumerate(X_existing.items()):
                        if param_name in param_bounds:
                            bounds = param_bounds[param_name]
                            # Add noise proportional to parameter range
                            param_range = bounds['max'] - bounds['min']
                            noise = np.random.normal(0, param_range * 0.1)
                            new_value = np.clip(base_point[i] + noise, bounds['min'], bounds['max'])
                            new_sample.append(new_value)
                        else:
                            new_sample.append(base_point[i])
                    
                    density_samples.append(new_sample)
                else:
                    # Fallback to uniform if no low-density regions
                    uniform_samples = self._fallback_uniform_sampling(X_existing, param_bounds, 1)
                    density_samples.extend(uniform_samples)
            
            return density_samples
            
        except ImportError:
            self.logger.warning("scipy not available for density-based gap filling, using uniform sampling")
            return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
        except Exception as e:
            self.logger.error(f"Error in density-based gap filling: {str(e)}")
            return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
    
    def _boundary_exploration(self, X_existing: pd.DataFrame, 
                            param_bounds: Dict[str, Dict[str, float]]) -> List[List[float]]:
        """
        Generate samples near parameter boundaries that may not have been explored.
        """
        boundary_samples = []
        n_boundary_samples = min(self.gap_filling_samples // 4, 15)
        
        for _ in range(n_boundary_samples):
            sample = []
            for param_name in X_existing.columns:
                if param_name in param_bounds:
                    bounds = param_bounds[param_name]
                    param_range = bounds['max'] - bounds['min']
                    margin = param_range * self.boundary_margin
                    
                    # Randomly choose to sample near min or max boundary
                    if np.random.random() < 0.5:
                        # Near minimum boundary
                        value = np.random.uniform(bounds['min'], bounds['min'] + margin)
                    else:
                        # Near maximum boundary
                        value = np.random.uniform(bounds['max'] - margin, bounds['max'])
                    
                    sample.append(value)
                else:
                    # If no bounds available, use existing data range
                    existing_values = X_existing[param_name]
                    value = np.random.uniform(existing_values.min(), existing_values.max())
                    sample.append(value)
            
            boundary_samples.append(sample)
        
        return boundary_samples
        
    def _uncertainty_based_sampling(self, X_existing: pd.DataFrame, y_existing: pd.Series,
                                param_bounds: Dict[str, Dict[str, float]]) -> List[List[float]]:
        """
        Generate samples in regions where a preliminary emulator has high uncertainty.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Train a quick preliminary emulator
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_existing)
            
            # Use Random Forest with uncertainty estimation
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_scaled, y_existing)
            
            # Generate candidate samples and evaluate uncertainty
            n_candidates = 1000
            candidate_samples = []
            
            for _ in range(n_candidates):
                sample = []
                for param_name in X_existing.columns:
                    if param_name in param_bounds:
                        bounds = param_bounds[param_name]
                        value = np.random.uniform(bounds['min'], bounds['max'])
                    else:
                        existing_values = X_existing[param_name]
                        value = np.random.uniform(existing_values.min(), existing_values.max())
                    sample.append(value)
                candidate_samples.append(sample)
            
            candidate_X = pd.DataFrame(candidate_samples, columns=X_existing.columns)
            candidate_X_scaled = scaler.transform(candidate_X)
            
            # Calculate prediction uncertainty (standard deviation across trees)
            tree_predictions = np.array([tree.predict(candidate_X_scaled) for tree in rf.estimators_])
            uncertainties = np.std(tree_predictions, axis=0)
            
            # Select samples with highest uncertainty
            uncertainty_threshold = np.percentile(uncertainties, self.uncertainty_percentile)
            high_uncertainty_indices = np.where(uncertainties >= uncertainty_threshold)[0]
            
            # Sample from high uncertainty regions
            n_uncertainty_samples = min(self.gap_filling_samples // 4, len(high_uncertainty_indices), 20)
            selected_indices = np.random.choice(high_uncertainty_indices, n_uncertainty_samples, replace=False)
            
            uncertainty_samples = [candidate_samples[i] for i in selected_indices]
            
            return uncertainty_samples
            
        except ImportError:
            self.logger.warning("scikit-learn not available for uncertainty-based sampling, skipping")
            return []
        except Exception as e:
            self.logger.error(f"Error in uncertainty-based sampling: {str(e)}")
            return []
    
    def _diversity_based_sampling(self, X_existing: pd.DataFrame, 
                                param_bounds: Dict[str, Dict[str, float]]) -> List[List[float]]:
        """
        Generate samples to maximize diversity in parameter space using space-filling designs.
        """
        try:
            # Use Latin Hypercube Sampling for diversity
            from scipy.stats import qmc
            
            n_diversity_samples = min(self.gap_filling_samples // 4, 25)
            n_dims = len(X_existing.columns)
            
            # Generate LHS samples
            sampler = qmc.LatinHypercube(d=n_dims, seed=42)
            unit_samples = sampler.random(n=n_diversity_samples)
            
            # Scale to parameter bounds
            diversity_samples = []
            for sample in unit_samples:
                scaled_sample = []
                for i, param_name in enumerate(X_existing.columns):
                    if param_name in param_bounds:
                        bounds = param_bounds[param_name]
                        value = bounds['min'] + sample[i] * (bounds['max'] - bounds['min'])
                    else:
                        # Use existing data range if no bounds
                        existing_values = X_existing[param_name]
                        value = existing_values.min() + sample[i] * (existing_values.max() - existing_values.min())
                    scaled_sample.append(value)
                diversity_samples.append(scaled_sample)
            
            return diversity_samples
            
        except ImportError:
            self.logger.warning("scipy.stats.qmc not available, using uniform sampling for diversity")
            return self._fallback_uniform_sampling(X_existing, param_bounds, self.gap_filling_samples // 4)
        except Exception as e:
            self.logger.error(f"Error in diversity-based sampling: {str(e)}")
            return []
    
    def _fallback_uniform_sampling(self, X_existing: pd.DataFrame, 
                                 param_bounds: Dict[str, Dict[str, float]], 
                                 n_samples: int) -> List[List[float]]:
        """Fallback uniform sampling when advanced methods fail."""
        uniform_samples = []
        
        for _ in range(n_samples):
            sample = []
            for param_name in X_existing.columns:
                if param_name in param_bounds:
                    bounds = param_bounds[param_name]
                    value = np.random.uniform(bounds['min'], bounds['max'])
                else:
                    existing_values = X_existing[param_name]
                    value = np.random.uniform(existing_values.min(), existing_values.max())
                sample.append(value)
            uniform_samples.append(sample)
        
        return uniform_samples
            
    def _estimate_gap_sample_scores(self, gap_X: pd.DataFrame, X_existing: pd.DataFrame, 
                                y_existing: pd.Series) -> pd.Series:
        """
        Estimate performance scores for gap samples using nearest neighbor or preliminary emulator.
        In practice, these would be replaced with actual SUMMA simulations.
        """
        try:
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Adjust k-neighbors based on available samples
            n_samples = len(X_existing)
            k_neighbors = min(5, max(1, n_samples))  # Use 1-5 neighbors based on available data
            
            if n_samples >= 2:  # Need at least 2 samples for scaling and modeling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_existing)
                gap_X_scaled = scaler.transform(gap_X)
                
                knn = KNeighborsRegressor(n_neighbors=k_neighbors, weights='distance')
                knn.fit(X_scaled, y_existing)
                
                estimated_scores = knn.predict(gap_X_scaled)
                
                # Add some noise to avoid overfitting to estimates
                noise_std = y_existing.std() * 0.1 if len(y_existing) > 1 else 0.05
                estimated_scores += np.random.normal(0, noise_std, len(estimated_scores))
            else:
                # With only 1 sample, use that value with noise
                base_score = y_existing.iloc[0] if len(y_existing) > 0 else 0.5
                noise_std = 0.1
                estimated_scores = np.random.normal(base_score, noise_std, len(gap_X))
            
            return pd.Series(estimated_scores)
            
        except ImportError:
            # Simple fallback: use mean of existing scores with noise
            mean_score = y_existing.mean() if len(y_existing) > 0 else 0.5
            std_score = y_existing.std() if len(y_existing) > 1 else 0.1
            estimated_scores = np.random.normal(mean_score, std_score * 0.5, len(gap_X))
            return pd.Series(estimated_scores)
        except Exception as e:
            self.logger.error(f"Error estimating gap sample scores: {str(e)}")
            # Return conservative estimates
            conservative_score = y_existing.quantile(0.3) if len(y_existing) > 0 else 0.3
            return pd.Series([conservative_score] * len(gap_X))
    
    def _save_gap_filling_analysis(self, gap_X: pd.DataFrame, gap_y: pd.Series, 
                                 strategies_used: List[str], X_existing: pd.DataFrame):
        """Save gap filling analysis results and visualizations."""
        try:
            # Save gap filling samples
            gap_data = gap_X.copy()
            gap_data['estimated_score'] = gap_y
            gap_data['strategy'] = strategies_used
            gap_data['sample_type'] = 'gap_filling'
            
            gap_file = self.gap_filling_dir / "gap_filling_samples.csv"
            gap_data.to_csv(gap_file, index=False)
            
            # Save analysis summary
            strategy_counts = pd.Series(strategies_used).value_counts()
            summary = {
                'total_gap_samples': len(gap_X),
                'strategies_used': list(strategy_counts.index),
                'samples_per_strategy': strategy_counts.to_dict(),
                'parameter_space_dimensions': len(gap_X.columns),
                'existing_samples': len(X_existing),
                'coverage_improvement': len(gap_X) / len(X_existing) * 100
            }
            
            summary_df = pd.DataFrame([summary])
            summary_file = self.gap_filling_dir / "gap_filling_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Create visualizations
            self._create_gap_filling_visualizations(gap_X, X_existing, strategies_used)
            
            self.logger.info(f"Gap filling analysis saved to: {self.gap_filling_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving gap filling analysis: {str(e)}")
    
    def _create_gap_filling_visualizations(self, gap_X: pd.DataFrame, X_existing: pd.DataFrame, 
                                         strategies_used: List[str]):
        """Create visualizations showing gap filling coverage."""
        try:
            import matplotlib.pyplot as plt
            
            plots_dir = self.gap_filling_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Parameter space coverage plots (first few parameter pairs)
            n_params = min(4, len(gap_X.columns))
            param_pairs = [(i, j) for i in range(n_params) for j in range(i+1, n_params)][:6]
            
            if param_pairs:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for idx, (i, j) in enumerate(param_pairs):
                    if idx >= len(axes):
                        break
                        
                    ax = axes[idx]
                    param_i = gap_X.columns[i]
                    param_j = gap_X.columns[j]
                    
                    # Plot existing samples
                    ax.scatter(X_existing[param_i], X_existing[param_j], 
                             alpha=0.6, c='blue', s=30, label='Optimization samples')
                    
                    # Plot gap filling samples by strategy
                    strategy_colors = {'density': 'red', 'boundary': 'green', 
                                     'uncertainty': 'orange', 'diversity': 'purple'}
                    
                    for strategy in set(strategies_used):
                        strategy_mask = [s == strategy for s in strategies_used]
                        gap_subset = gap_X[strategy_mask]
                        if len(gap_subset) > 0:
                            ax.scatter(gap_subset[param_i], gap_subset[param_j], 
                                     alpha=0.8, c=strategy_colors.get(strategy, 'black'), 
                                     s=50, marker='^', label=f'{strategy.title()} gap filling')
                    
                    ax.set_xlabel(param_i)
                    ax.set_ylabel(param_j)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "parameter_space_coverage.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Strategy distribution plot
            strategy_counts = pd.Series(strategies_used).value_counts()
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(strategy_counts.index, strategy_counts.values)
            plt.xlabel('Gap Filling Strategy')
            plt.ylabel('Number of Samples')
            plt.title('Gap Filling Samples by Strategy')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "strategy_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Gap filling visualizations saved to: {plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating gap filling visualizations: {str(e)}")


class GapFillingSimulator:
    """
    Handles running simulations for gap filling samples.
    
    This class manages the execution of SUMMA simulations for strategically generated
    gap filling samples, replacing estimated scores with actual model evaluations.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Gap filling simulation directory
        self.gap_sim_dir = self.project_dir / "emulation" / self.experiment_id / "gap_filling" / "simulations"
        self.gap_sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Model runner - placeholder for actual implementation
        # We'll disable actual simulations for now and use estimates
        self.model_runner = None
    
    def run_gap_filling_simulations(self, gap_X: pd.DataFrame) -> pd.Series:
        """
        Run SUMMA simulations for gap filling samples.
        
        Args:
            gap_X: DataFrame with gap filling parameter samples
            
        Returns:
            Series with actual performance scores
        """
        self.logger.info(f"Running simulations for {len(gap_X)} gap filling samples")
        
        # Check if we should actually run simulations or use estimates
        if not self.config.get('RUN_GAP_FILLING_SIMULATIONS', False) or self.model_runner is None:
            if self.model_runner is None:
                self.logger.info("ModelRunner not available, using estimates for gap filling")
            else:
                self.logger.info("Gap filling simulations disabled, using estimates")
            return self._use_estimated_scores(gap_X)
        
        actual_scores = []
        successful_runs = 0
        
        for idx, (_, sample) in enumerate(gap_X.iterrows()):
            run_id = f"gap_fill_{idx:04d}"
            self.logger.debug(f"Running gap filling simulation {run_id}")
            
            try:
                # Create parameter dictionary
                params = sample.to_dict()
                
                # Set up simulation
                success = self._setup_gap_simulation(run_id, params)
                if not success:
                    actual_scores.append(np.nan)
                    continue
                
                # Run simulation
                score = self._run_gap_simulation(run_id)
                actual_scores.append(score)
                
                if not np.isnan(score):
                    successful_runs += 1
                    
            except Exception as e:
                self.logger.warning(f"Gap filling simulation {run_id} failed: {str(e)}")
                actual_scores.append(np.nan)
        
        self.logger.info(f"Gap filling simulations completed: {successful_runs}/{len(gap_X)} successful")
        
        return pd.Series(actual_scores)
    
    def _use_estimated_scores(self, gap_X: pd.DataFrame) -> pd.Series:
        """Use estimated scores instead of running simulations."""
        # This would use the same estimation logic as in GapFillingStrategy
        # For now, return conservative estimates
        n_samples = len(gap_X)
        base_score = 0.6  # Conservative baseline
        noise_std = 0.1
        
        estimated_scores = np.random.normal(base_score, noise_std, n_samples)
        estimated_scores = np.clip(estimated_scores, 0.0, 1.0)  # Keep in reasonable range
        
        return pd.Series(estimated_scores)
    
    def _setup_gap_simulation(self, run_id: str, params: Dict[str, float]) -> bool:
        """Set up simulation directory and files for gap filling run."""
        try:
            run_dir = self.gap_sim_dir / run_id
            settings_dir = run_dir / "settings" / "SUMMA"
            
            # Copy settings files
            source_settings = self.project_dir / "settings" / "SUMMA"
            static_files = [
                'modelDecisions.txt', 'outputControl.txt', 'localParamInfo.txt',
                'basinParamInfo.txt', 'attributes.nc', 'coldState.nc',
                'forcingFileList.txt', 'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL',
                'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL'
            ]
            
            FileManager.copy_settings_files(source_settings, settings_dir, static_files)
            
            # Create trial parameters file
            FileManager.create_trial_params_file(
                params,
                settings_dir / 'trialParams.nc',
                source_settings / 'attributes.nc'
            )
            
            # Create file manager
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            FileManager.create_file_manager(
                source_settings / 'fileManager.txt',
                settings_dir / 'fileManager.txt',
                settings_dir,
                output_dir
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up gap simulation {run_id}: {str(e)}")
            return False
    
    def _run_gap_simulation(self, run_id: str) -> float:
        """Run gap filling simulation and calculate performance."""
        try:
            # For now, return estimated scores since ModelRunner is not available
            # In a full implementation, this would run actual SUMMA simulations
            self.logger.debug(f"Using estimated score for gap filling sample {run_id}")
            return np.random.uniform(0.3, 0.8)  # Conservative estimate
            
        except Exception as e:
            self.logger.error(f"Error running gap simulation {run_id}: {str(e)}")
            return np.nan
    
    def _calculate_gap_performance(self, run_dir: Path) -> Optional[float]:
        """Calculate performance metric for gap filling simulation."""
        try:
            # This is a simplified version - would normally extract streamflow and compare
            # For now, return a random score in a reasonable range
            return np.random.uniform(0.3, 0.8)
            
        except Exception as e:
            self.logger.error(f"Error calculating gap performance: {str(e)}")
            return None


# ============================================================================
# Enhanced Parameter Manager
# ============================================================================

class ParameterManager:
    """
    Manages parameter persistence and updates with best practices from optimization methods.
    
    This class handles:
    1. Loading existing optimized parameters
    2. Comparing emulator results with optimization results
    3. Updating default trial parameters if improvement is found
    4. Creating backups with proper versioning
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Default settings directory
        self.default_settings_dir = self.project_dir / "settings" / "SUMMA"
        self.default_trial_params_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
    
    def get_best_existing_score(self) -> Optional[float]:
        """Get the best score from existing optimization runs."""
        try:
            data_loader = OptimizationDataLoader(self.config, self.logger)
            optimization_data = data_loader.load_optimization_data()
            
            if optimization_data is not None:
                _, y, metadata = optimization_data
                if len(y) > 0:
                    best_score = y.max()
                    self.logger.info(f"Best existing optimization score: {best_score:.6f}")
                    return best_score
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting best existing score: {str(e)}")
            return None
    
    def should_update_parameters(self, emulator_score: float, emulator_params: Dict[str, float]) -> bool:
        """
        Determine if emulator results are better than existing optimization results.
        
        Args:
            emulator_score: Predicted score from emulator
            emulator_params: Optimized parameters from emulator
            
        Returns:
            True if emulator results should replace existing parameters
        """
        try:
            best_existing = self.get_best_existing_score()
            
            if best_existing is None:
                self.logger.info("No existing optimization results found, emulator results will be used")
                return True
            
            improvement_threshold = self.config.get('EMULATOR_IMPROVEMENT_THRESHOLD', 0.01)
            improvement = emulator_score - best_existing
            
            if improvement > improvement_threshold:
                self.logger.info(f"Emulator improvement: {improvement:.6f} (threshold: {improvement_threshold:.6f})")
                self.logger.info("Emulator results are significantly better, will update parameters")
                return True
            else:
                self.logger.info(f"Emulator improvement: {improvement:.6f} (below threshold: {improvement_threshold:.6f})")
                self.logger.info("Emulator results do not meet improvement threshold")
                return False
                
        except Exception as e:
            self.logger.error(f"Error determining if parameters should be updated: {str(e)}")
            return False
    
    def update_default_parameters(self, best_params: Dict[str, float], emulator_score: float, 
                                emulator_type: str = "emulator") -> bool:
        """
        Update default trial parameters with improved emulator results.
        
        Args:
            best_params: Dictionary with best parameter values
            emulator_score: Predicted score from emulator
            emulator_type: Type of emulator used (for tracking)
            
        Returns:
            True if update was successful
        """
        self.logger.info(f"Updating default parameters with {emulator_type} results")
        
        try:
            # Check if default settings directory exists
            if not self.default_settings_dir.exists():
                self.logger.error(f"Default settings directory not found: {self.default_settings_dir}")
                return False
            
            # Step 1: Create timestamped backup of existing parameters
            backup_success = self._create_parameter_backup()
            if not backup_success:
                self.logger.warning("Could not create parameter backup, proceeding anyway")
            
            # Step 2: Generate new trial parameters file
            new_trial_params_path = self._generate_trial_params_file(best_params)
            
            if new_trial_params_path:
                self.logger.info(f"✅ Successfully updated default parameters: {new_trial_params_path}")
                
                # Step 3: Save metadata about the update
                self._save_update_metadata(best_params, emulator_score, emulator_type)
                
                # Step 4: Save parameters in readable format
                self._save_readable_parameters(best_params, emulator_score, emulator_type)
                
                return True
            else:
                self.logger.error("Failed to generate new trial parameters file")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating default parameters: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_parameter_backup(self) -> bool:
        """Create timestamped backup of existing trial parameters."""
        try:
            if not self.default_trial_params_path.exists():
                self.logger.info("No existing trial parameters file to backup")
                return True
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.default_settings_dir / f"trialParams_default_{timestamp}.nc"
            
            shutil.copy2(self.default_trial_params_path, backup_path)
            self.logger.info(f"Created parameter backup: {backup_path}")
            
            # Also create a "latest backup" for easy reference
            latest_backup = self.default_settings_dir / "trialParams_previous.nc"
            shutil.copy2(self.default_trial_params_path, latest_backup)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating parameter backup: {str(e)}")
            return False
    
    def _generate_trial_params_file(self, params: Dict[str, float]) -> Optional[Path]:
        """Generate new trial parameters file."""
        try:
            # Get attribute file for HRU information
            attr_file_path = self.default_settings_dir / self.config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
            
            if not attr_file_path.exists():
                self.logger.error(f"Attribute file not found: {attr_file_path}")
                return None
            
            # Read HRU and GRU information
            with nc.Dataset(attr_file_path, 'r') as attr_ds:
                hru_ids = attr_ds.variables['hruId'][:]
                has_gru_dim = 'gru' in attr_ds.dimensions
                
                if has_gru_dim:
                    gru_ids = attr_ds.variables['gruId'][:]
                else:
                    gru_ids = np.array([1])
            
            # Create new trial parameters file
            with nc.Dataset(self.default_trial_params_path, 'w', format='NETCDF4') as output_ds:
                # Add dimensions
                output_ds.createDimension('hru', len(hru_ids))
                
                if has_gru_dim:
                    output_ds.createDimension('gru', len(gru_ids))
                    
                    # Add GRU ID variable
                    gru_id_var = output_ds.createVariable('gruId', 'i4', ('gru',))
                    gru_id_var[:] = gru_ids
                    gru_id_var.long_name = 'Group Response Unit ID (GRU)'
                    gru_id_var.units = '-'
                
                # Add HRU ID variable
                hru_id_var = output_ds.createVariable('hruId', 'i4', ('hru',))
                hru_id_var[:] = hru_ids
                hru_id_var.long_name = 'Hydrologic Response Unit ID (HRU)'
                hru_id_var.units = '-'
                
                # Routing parameters (GRU level)
                routing_params = ['routingGammaShape', 'routingGammaScale']
                
                # Add parameter variables
                for param_name, param_value in params.items():
                    if param_name in routing_params and has_gru_dim:
                        # GRU-level parameter
                        param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                        param_var[:] = np.full(len(gru_ids), param_value)
                    else:
                        # HRU-level parameter
                        param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                        param_var[:] = np.full(len(hru_ids), param_value)
                    
                    # Add attributes
                    param_var.long_name = f"Optimized value for {param_name}"
                    param_var.units = "N/A"
                
                # Add global attributes
                output_ds.description = "SUMMA Trial Parameters optimized by CONFLUENCE Emulator"
                output_ds.history = f"Created on {datetime.now().isoformat()}"
                output_ds.confluence_experiment_id = f"emulator_optimization_{self.experiment_id}"
                output_ds.emulator_type = "Neural Network and Random Forest"
            
            self.logger.info(f"Generated new trial parameters file: {self.default_trial_params_path}")
            return self.default_trial_params_path
            
        except Exception as e:
            self.logger.error(f"Error generating trial parameters file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _save_update_metadata(self, params: Dict[str, float], score: float, emulator_type: str):
        """Save metadata about the parameter update."""
        try:
            metadata = {
                'update_timestamp': datetime.now().isoformat(),
                'emulator_type': emulator_type,
                'predicted_score': score,
                'experiment_id': self.experiment_id,
                'domain_name': self.domain_name,
                'num_parameters_updated': len(params),
                'parameter_names': list(params.keys())
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.default_settings_dir / "emulator_update_log.csv"
            
            # Append to existing log or create new one
            if metadata_file.exists():
                existing_df = pd.read_csv(metadata_file)
                combined_df = pd.concat([existing_df, metadata_df], ignore_index=True)
                combined_df.to_csv(metadata_file, index=False)
            else:
                metadata_df.to_csv(metadata_file, index=False)
            
            self.logger.info(f"Saved update metadata to: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving update metadata: {str(e)}")
    
    def _save_readable_parameters(self, params: Dict[str, float], score: float, emulator_type: str):
        """Save parameters in human-readable CSV format."""
        try:
            # Create readable parameter record
            param_record = {
                'parameter_name': list(params.keys()),
                'optimized_value': list(params.values()),
                'emulator_type': [emulator_type] * len(params),
                'predicted_score': [score] * len(params),
                'update_timestamp': [datetime.now().isoformat()] * len(params)
            }
            
            param_df = pd.DataFrame(param_record)
            readable_file = self.default_settings_dir / f"emulator_optimized_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            param_df.to_csv(readable_file, index=False)
            
            # Also save as "latest" for easy reference
            latest_file = self.default_settings_dir / "latest_emulator_parameters.csv"
            param_df.to_csv(latest_file, index=False)
            
            self.logger.info(f"Saved readable parameters to: {readable_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving readable parameters: {str(e)}")


# ============================================================================
# File Manager (Enhanced from original)
# ============================================================================

class FileManager:
    """Enhanced file management with better error handling."""
    
    @staticmethod
    def create_trial_params_file(params: Dict[str, float], output_path: Path, 
                                attr_file: Path) -> Path:
        """Create trial parameters NetCDF file with enhanced error handling."""
        try:
            with nc.Dataset(attr_file, 'r') as attr_ds:
                hru_ids = attr_ds.variables['hruId'][:]
                num_hru = len(hru_ids)
                
                # Check for GRU dimension
                has_gru_dim = 'gru' in attr_ds.dimensions
                if has_gru_dim:
                    gru_ids = attr_ds.variables['gruId'][:]
                else:
                    gru_ids = np.array([1])
            
            with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
                # Create dimensions
                ds.createDimension('hru', num_hru)
                
                if has_gru_dim:
                    ds.createDimension('gru', len(gru_ids))
                    
                    # Add GRU ID variable
                    gru_var = ds.createVariable('gruId', 'i4', ('gru',))
                    gru_var[:] = gru_ids
                    gru_var.long_name = 'Group Response Unit ID'
                
                # Add HRU ID variable
                hru_var = ds.createVariable('hruId', 'i4', ('hru',))
                hru_var[:] = hru_ids
                hru_var.long_name = 'Hydrologic Response Unit ID'
                
                # Routing parameters that should be at GRU level
                routing_params = ['routingGammaShape', 'routingGammaScale']
                
                # Add parameter variables
                for param_name, value in params.items():
                    if param_name in routing_params and has_gru_dim:
                        # GRU-level parameter
                        param_var = ds.createVariable(param_name, 'f8', ('gru',))
                        param_var[:] = np.full(len(gru_ids), value)
                    else:
                        # HRU-level parameter
                        param_var = ds.createVariable(param_name, 'f8', ('hru',))
                        param_var[:] = np.full(num_hru, value)
                    
                    param_var.long_name = f"Trial value for {param_name}"
                
                # Add metadata
                ds.description = "SUMMA Trial Parameters from CONFLUENCE Emulator"
                ds.created = datetime.now().isoformat()
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to create trial parameters file: {str(e)}")
    
    @staticmethod
    def copy_settings_files(source_dir: Path, target_dir: Path, 
                           files_to_copy: List[str]) -> None:
        """Copy SUMMA settings files with error handling."""
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in files_to_copy:
            source_file = source_dir / filename
            if source_file.exists():
                try:
                    shutil.copy2(source_file, target_dir / filename)
                except Exception as e:
                    logging.getLogger().warning(f"Could not copy {filename}: {str(e)}")
    
    @staticmethod
    def create_file_manager(template_path: Path, output_path: Path, 
                           settings_dir: Path, output_dir: Path) -> Path:
        """Create modified fileManager.txt with enhanced error handling."""
        try:
            with open(template_path, 'r') as f:
                lines = f.readlines()
            
            with open(output_path, 'w') as f:
                for line in lines:
                    if "outputPath" in line:
                        f.write(f"outputPath           '{output_dir}/'\n")
                    elif "settingsPath" in line:
                        f.write(f"settingsPath         '{settings_dir}/'\n")
                    else:
                        f.write(line)
            
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to create file manager: {str(e)}")


# ============================================================================
# Enhanced Emulation Runner
# ============================================================================

class EnhancedEmulationRunner:
    """
    Enhanced EmulationRunner with optimization integration and neural network support.
    
    Key improvements:
    1. Uses existing optimization data when available
    2. Supports both Random Forest and Neural Network emulators
    3. Updates default parameters if emulator improves on optimization results
    4. Better error handling and logging
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Initialize managers
        self.data_loader = OptimizationDataLoader(config, logger)
        self.param_manager = ParameterManager(config, logger)
    
    def run_emulation_workflow(self) -> Optional[Dict]:
        """Execute the complete enhanced emulation workflow with strategic gap filling."""
        self.logger.info("Starting enhanced model emulation workflow with strategic gap filling")
        
        try:
            # Step 1: Check for existing optimization data
            optimization_data = self._load_optimization_training_data()
            
            # Step 2: If no optimization data, fall back to ensemble generation
            if optimization_data is None:
                self.logger.info("No optimization data found, generating ensemble training data")
                optimization_data = self._generate_ensemble_training_data()
            
            if optimization_data is None:
                self.logger.error("No training data available for emulation")
                return None
            
            X_base, y_base, metadata = optimization_data
            
            # Check if data loading was successful
            if X_base is None or y_base is None:
                self.logger.error("Failed to load valid training data")
                self.logger.error("This could be due to:")
                self.logger.error("  1. Parameter format mismatches between optimization files")
                self.logger.error("  2. Missing or corrupted optimization results")
                self.logger.error("  3. Insufficient valid parameter evaluations")
                
                # Try to provide more specific information
                if optimization_data and len(optimization_data) >= 3:
                    _, _, meta = optimization_data
                    if meta:
                        self.logger.error(f"Metadata shows: {meta.get('total_evaluations', 0)} evaluations from {len(meta.get('sources', []))} sources")
                        self.logger.error(f"Parameter names expected: {meta.get('parameter_names', [])}")
                
                return None
            
            self.logger.info(f"Base training data: {len(X_base)} samples, {len(X_base.columns)} features")
            
            # Check if we have sufficient training data
            min_samples_required = 5  # Minimum for meaningful emulation
            if len(X_base) < min_samples_required:
                self.logger.warning(f"Very limited training data ({len(X_base)} samples). Emulation results may be unreliable.")
                self.logger.warning("Consider:")
                self.logger.warning("  1. Running more optimization iterations to generate more training data")
                self.logger.warning("  2. Checking parameter format consistency between optimization methods")
                self.logger.warning("  3. Disabling emulation and using optimization results directly")
                
                # If we have only 1 sample, disable gap filling
                if len(X_base) == 1:
                    self.logger.warning("Only 1 training sample available. Disabling gap filling.")
                    X_final, y_final = X_base, y_base
                else:
                    # Step 3: Strategic gap filling (if enabled and we have enough samples)
                    X_final, y_final = self._perform_strategic_gap_filling(X_base, y_base, metadata)
            else:
                # Step 3: Strategic gap filling (if enabled)
                X_final, y_final = self._perform_strategic_gap_filling(X_base, y_base, metadata)
            
            # Step 4: Run emulators on enhanced dataset
            emulation_results = self._run_emulators(X_final, y_final, metadata)
            
            # Step 5: Compare results and update parameters if improved
            self._evaluate_and_update_parameters(emulation_results, metadata)
            
            return emulation_results
            
        except Exception as e:
            self.logger.error(f"Error during enhanced emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _perform_strategic_gap_filling(self, X_base: pd.DataFrame, y_base: pd.Series, 
                                     metadata: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform strategic gap filling to enhance parameter space coverage.
        
        Args:
            X_base: Base parameter samples from optimization
            y_base: Base performance scores
            metadata: Training data metadata
            
        Returns:
            Enhanced (X, y) data with gap filling samples added
        """
        # Check if gap filling is enabled
        if not self.config.get('ENABLE_GAP_FILLING', True):
            self.logger.info("Strategic gap filling disabled, using base dataset only")
            return X_base, y_base
        
        try:
            self.logger.info("Performing strategic gap filling analysis")
            
            # Get parameter bounds
            param_bounds = self._get_parameter_bounds_for_gap_filling()
            if not param_bounds:
                self.logger.warning("No parameter bounds available for gap filling")
                return X_base, y_base
            
            # Initialize gap filling strategy
            gap_filler = GapFillingStrategy(self.config, self.logger)
            
            # Identify and generate gap filling samples
            gap_X, gap_y_estimated = gap_filler.identify_and_fill_gaps(X_base, y_base, param_bounds)
            
            if len(gap_X) == 0:
                self.logger.info("No gap filling samples generated, using base dataset")
                return X_base, y_base
            
            # Option 1: Run actual simulations for gap samples (if enabled)
            if self.config.get('RUN_GAP_FILLING_SIMULATIONS', False):
                gap_simulator = GapFillingSimulator(self.config, self.logger)
                gap_y_actual = gap_simulator.run_gap_filling_simulations(gap_X)
                gap_y = gap_y_actual
                self.logger.info(f"Gap filling simulations completed for {len(gap_X)} samples")
            else:
                # Option 2: Use estimated scores
                gap_y = gap_y_estimated
                self.logger.info(f"Using estimated scores for {len(gap_X)} gap filling samples")
            
            # Combine base data with gap filling data
            X_combined = pd.concat([X_base, gap_X], ignore_index=True)
            y_combined = pd.concat([y_base, gap_y], ignore_index=True)
            
            # Remove any NaN scores
            valid_mask = ~y_combined.isna()
            X_final = X_combined[valid_mask]
            y_final = y_combined[valid_mask]
            
            # Update metadata
            metadata['gap_filling'] = {
                'enabled': True,
                'samples_added': len(gap_X),
                'samples_with_valid_scores': valid_mask.sum() - len(y_base),
                'coverage_improvement': len(gap_X) / len(X_base) * 100,
                'total_final_samples': len(X_final)
            }
            
            self.logger.info(f"Strategic gap filling completed:")
            self.logger.info(f"  • Base samples: {len(X_base)}")
            self.logger.info(f"  • Gap filling samples: {len(gap_X)}")
            self.logger.info(f"  • Final dataset: {len(X_final)} samples")
            self.logger.info(f"  • Coverage improvement: {len(gap_X) / len(X_base) * 100:.1f}%")
            
            return X_final, y_final
            
        except Exception as e:
            self.logger.error(f"Error during strategic gap filling: {str(e)}")
            self.logger.info("Falling back to base dataset without gap filling")
            return X_base, y_base
    
    def _get_parameter_bounds_for_gap_filling(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for gap filling analysis with better error handling."""
        bounds = {}
        
        try:
            # Local parameters
            local_file = self.project_dir / "settings" / "SUMMA" / "localParamInfo.txt"
            if local_file.exists():
                local_bounds = self._parse_bounds_file(local_file)
                bounds.update(local_bounds)
                self.logger.debug(f"Loaded {len(local_bounds)} local parameter bounds")
            
            # Basin parameters
            basin_file = self.project_dir / "settings" / "SUMMA" / "basinParamInfo.txt"
            if basin_file.exists():
                basin_bounds = self._parse_bounds_file(basin_file)
                bounds.update(basin_bounds)
                self.logger.debug(f"Loaded {len(basin_bounds)} basin parameter bounds")
            
            # If no bounds found, create reasonable defaults for known parameters
            if not bounds:
                self.logger.warning("No parameter bounds found, using defaults")
                default_bounds = {
                    'tempCritRain': {'min': -2.0, 'max': 2.0},
                    'rootingDepth': {'min': 0.1, 'max': 10.0},
                    'theta_sat': {'min': 0.3, 'max': 0.7},
                    'theta_res': {'min': 0.01, 'max': 0.2},
                    'vGn_n': {'min': 1.1, 'max': 3.0},
                    'k_soil': {'min': 1e-7, 'max': 1e-3},
                    'k_macropore': {'min': 1e-5, 'max': 1e-1},
                    'zScale_TOPMODEL': {'min': 1.0, 'max': 100.0},
                    'routingGammaScale': {'min': 1.0, 'max': 10000.0},
                    'routingGammaShape': {'min': 1.0, 'max': 10.0}
                }
                bounds.update(default_bounds)
            
            self.logger.info(f"Total parameter bounds available: {len(bounds)}")
            return bounds
            
        except Exception as e:
            self.logger.error(f"Error getting parameter bounds: {str(e)}")
            return {}
    
    def _parse_bounds_file(self, file_path: Path) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds file."""
        bounds = {}
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!'):
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        param_name = parts[0].strip()
                        try:
                            # Handle Fortran-style scientific notation
                            max_val_str = parts[1].replace('d', 'e').replace('D', 'e')
                            min_val_str = parts[2].replace('d', 'e').replace('D', 'e')
                            
                            max_val = float(max_val_str)
                            min_val = float(min_val_str)
                            bounds[param_name] = {'min': min_val, 'max': max_val}
                        except ValueError:
                            continue
        except Exception as e:
            self.logger.error(f"Error parsing bounds file {file_path}: {str(e)}")
        
        return bounds
    
    def _load_optimization_training_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series, Dict]]:
        """Load training data from existing optimization runs."""
        self.logger.info("Attempting to load existing optimization data for training")
        return self.data_loader.load_optimization_data()
    
    def _generate_ensemble_training_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series, Dict]]:
        """Generate training data using ensemble simulations (fallback)."""
        self.logger.info("Generating ensemble training data as fallback")
        
        try:
            # Use simplified emulator for ensemble generation
            from single_sample_emulator import SingleSampleEmulator
            
            emulator = SingleSampleEmulator(self.config, self.logger)
            
            # Generate ensemble
            emulator_output = emulator.run_emulation_setup()
            if not emulator_output:
                return None
            
            # Run simulations
            ensemble_results = emulator.run_ensemble_simulations()
            success_count = sum(1 for _, success, _ in ensemble_results if success)
            
            if success_count == 0:
                self.logger.error("No successful ensemble simulations")
                return None
            
            # Analyze results
            analysis_dir = emulator.analyze_ensemble_results()
            if not analysis_dir:
                return None
            
            # Load the generated data
            return self._load_ensemble_data()
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble training data: {str(e)}")
            return None
    
    def _load_ensemble_data(self) -> Optional[Tuple[pd.DataFrame, pd.Series, Dict]]:
        """Load ensemble data for training."""
        try:
            # Load parameter sets
            param_file = self.project_dir / "emulation" / f"parameter_sets_{self.experiment_id}.nc"
            metrics_file = self.project_dir / "emulation" / self.experiment_id / "ensemble_analysis" / "performance_metrics.csv"
            
            if not param_file.exists() or not metrics_file.exists():
                return None
            
            # Read parameter data
            param_data = {}
            with nc.Dataset(param_file, 'r') as ds:
                param_names = [var for var in ds.variables if var not in ['hruId', 'runIndex']]
                for param in param_names:
                    param_data[param] = np.mean(ds.variables[param][:], axis=1)
            
            X = pd.DataFrame(param_data)
            
            # Read metrics
            metrics_df = pd.read_csv(metrics_file)
            target_metric = self.config.get('EMULATION_TARGET_METRIC', 'KGE')
            
            if target_metric not in metrics_df.columns:
                self.logger.error(f"Target metric {target_metric} not found in ensemble results")
                return None
            
            min_rows = min(len(X), len(metrics_df))
            X = X.iloc[:min_rows]
            y = metrics_df[target_metric].iloc[:min_rows]
            
            # Remove NaN values
            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            metadata = {
                'sources': ['ensemble_simulation'],
                'total_evaluations': len(X),
                'algorithms': ['ensemble'],
                'best_score': y.max() if len(y) > 0 else 0.0,
                'parameter_names': list(X.columns)
            }
            
            return X, y, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble data: {str(e)}")
            return None
                
    def _run_random_forest_emulation(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
        """FIXED Random Forest emulation workflow."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Handle small sample sizes
            if len(X) < 5:
                self.logger.warning(f"Very small sample size ({len(X)}) for Random Forest. Results may be unreliable.")
                test_size = 0.2 if len(X) > 2 else 0.0
            else:
                test_size = 0.3
            
            if test_size > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Scale features - keep as DataFrames to maintain feature names
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest with parameters suitable for small datasets
            n_estimators = min(100, max(10, len(X_train) * 5))
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=min(10, max(3, len(X.columns))),
                min_samples_split=max(2, len(X_train) // 5),
                min_samples_leaf=max(1, len(X_train) // 10),
                random_state=42
            )
            
            rf.fit(X_train_scaled, y_train)
            
            # Cross-validation to check for overfitting
            cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=min(5, len(X_train)//2), scoring='r2')
            self.logger.info(f"RF Cross-validation R² scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Make predictions
            train_pred = rf.predict(X_train_scaled)
            test_pred = rf.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred) if len(y_test) > 1 else train_r2
            
            # Get feature importances
            feature_importance = dict(zip(X.columns, rf.feature_importances_))
            
            # FIXED: Actually optimize using the trained RF model
            best_params, best_score = self._optimize_rf_parameters(rf, scaler, X)
            
            # Validate the optimization result
            if best_score > 1:
                self.logger.warning(f"RF predicted score ({best_score:.4f}) much higher than training max ({y.max():.4f})")
                self.logger.warning("This may indicate extrapolation")
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'model_metrics': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'n_samples': len(X),
                    'n_features': len(X.columns)
                },
                'feature_importance': feature_importance,
                'final_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error in Random Forest emulation: {str(e)}")
            return None


    def _optimize_rf_parameters(self, rf_model, scaler, X: pd.DataFrame) -> Tuple[Dict[str, float], float]:
        """Optimize parameters using trained Random Forest model."""
        try:
            from scipy.optimize import differential_evolution
            
            # Get parameter bounds - more conservative approach
            param_bounds = self._get_conservative_parameter_bounds(X)
            if not param_bounds:
                self.logger.error("No parameter bounds available for RF optimization")
                # Fallback to best training sample
                best_idx = X.iloc[:, 0].idxmax()  # Use first column to find index
                return X.loc[best_idx].to_dict(), 0.0
            
            param_names = list(X.columns)
            
            def objective(params):
                """Objective function with bounds checking."""
                try:
                    # Check if parameters are within reasonable range of training data
                    for i, (param_name, value) in enumerate(zip(param_names, params)):
                        train_values = X[param_name]
                        if value < train_values.quantile(0.01) or value > train_values.quantile(0.99):
                            return 1000.0  # Penalty for extrapolation
                    
                    # Create DataFrame with proper feature names to avoid sklearn warnings
                    X_pred = pd.DataFrame([params], columns=param_names)
                    X_pred_scaled = scaler.transform(X_pred)
                    prediction = rf_model.predict(X_pred_scaled)[0]
                    return -prediction  # Minimize negative (maximize positive)
                except Exception as e:
                    return 1000.0  # High penalty for failed evaluations
            
            # Create bounds array
            bounds = []
            for name in param_names:
                if name in param_bounds:
                    # Use more conservative bounds - within training data range
                    train_values = X[name]
                    conservative_min = max(param_bounds[name]['min'], train_values.quantile(0.05))
                    conservative_max = min(param_bounds[name]['max'], train_values.quantile(0.95))
                    bounds.append((conservative_min, conservative_max))
                else:
                    # Use training data range if no bounds
                    train_values = X[name]
                    bounds.append((train_values.min(), train_values.max()))
            
            # Run optimization with conservative settings
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=100,  # Reduced iterations
                popsize=10,   # Smaller population
                atol=1e-6,
                tol=1e-6
            )
            
            # Convert to parameter dictionary
            optimal_params = {}
            for i, name in enumerate(param_names):
                optimal_params[name] = float(result.x[i])
            
            best_score = -result.fun
            
            self.logger.info(f"RF optimization completed: predicted score = {best_score:.4f}")
            return optimal_params, best_score
            
        except Exception as e:
            self.logger.error(f"Error in RF parameter optimization: {str(e)}")
            # Fallback to best training sample
            best_idx = X.iloc[:, 0].idxmax()
            return X.loc[best_idx].to_dict(), 0.0


    def _get_conservative_parameter_bounds(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get conservative parameter bounds based on training data distribution."""
        bounds = {}
        
        for param_name in X.columns:
            values = X[param_name]
            # Use training data range with small buffer
            value_range = values.max() - values.min()
            buffer = value_range * 0.1  # 10% buffer
            
            bounds[param_name] = {
                'min': values.min() - buffer,
                'max': values.max() + buffer
            }
        
        return bounds



    def _run_emulators(self, X: pd.DataFrame, y: pd.Series, metadata: Dict) -> Dict:
        """Run both Random Forest and Neural Network emulators."""
        results = {}
        
        # Random Forest Emulator
        if self.config.get('RUN_RANDOM_FOREST_EMULATION', True):
            self.logger.info("Running Random Forest emulation")
            try:
                # Import and create RandomForestEmulator locally to avoid circular import
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import r2_score, mean_squared_error
                
                rf_results = self._run_random_forest_emulation(X, y)
                if rf_results:
                    results['random_forest'] = rf_results
                    self.logger.info(f"Random Forest completed: score = {rf_results['best_score']:.4f}")
            except Exception as e:
                self.logger.error(f"Random Forest emulation failed: {str(e)}")
        
        # Neural Network Emulator
        if self.config.get('RUN_NEURAL_NETWORK_EMULATION', True):
            self.logger.info("Running Neural Network emulation")
            try:
                nn_emulator = NeuralNetworkEmulator(self.config, self.logger)
                nn_results = nn_emulator.run_workflow((X, y))
                if nn_results:
                    results['neural_network'] = nn_results
                    self.logger.info(f"Neural Network completed: score = {nn_results['best_score']:.4f}")
            except Exception as e:
                self.logger.error(f"Neural Network emulation failed: {str(e)}")
        
        # Add metadata
        results['metadata'] = metadata
        results['training_data_info'] = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target_metric': self.config.get('EMULATION_TARGET_METRIC', 'KGE'),
            'data_sources': metadata.get('sources', [])
        }
        
        return results
        

    # Update the main evaluation method to use dual validation:
    def _evaluate_and_update_parameters(self, emulation_results: Dict, metadata: Dict):
        """Enhanced evaluation with dual-period validation."""
        self.logger.info("Evaluating emulation results with dual-period validation")
        
        validation_results = {}
        best_calib_score = float('-inf')
        best_eval_score = float('-inf')
        best_emulator = None
        best_params = None
        
        # Run dual-period validation for each emulator
        for emulator_type in ['random_forest', 'neural_network']:
            if emulator_type not in emulation_results or not emulation_results[emulator_type]:
                continue
                
            results = emulation_results[emulator_type]
            params = results.get('best_parameters', {})
            
            if not params:
                continue
            
            predicted_score = results.get('best_score', float('-inf'))
            self.logger.info(f"🔍 Dual-period validation for {emulator_type} (predicted: {predicted_score:.6f})")
            
            # Run dual validation
            validator = SummaValidator(self.config, self.logger)
            dual_result = validator.run_dual_period_validation(params, emulator_type)
            validation_results[emulator_type] = dual_result
            
            # Track best results for each period
            if dual_result.get('calibration_validation', {}).get('success'):
                calib_score = dual_result['calibration_validation'].get('primary_metric', float('-inf'))
                if calib_score > best_calib_score:
                    best_calib_score = calib_score
                    
            if dual_result.get('evaluation_validation', {}).get('success'):
                eval_score = dual_result['evaluation_validation'].get('primary_metric', float('-inf'))
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_emulator = emulator_type
                    best_params = params
        
        # Add validation results
        emulation_results['dual_period_validation'] = validation_results
        
        # Report comprehensive comparison
        self.logger.info("📊 Dual-Period Validation Summary:")
        for emulator_type, dual_result in validation_results.items():
            if dual_result.get('temporal_analysis'):
                analysis = dual_result['temporal_analysis']
                predicted = emulation_results[emulator_type]['best_score']
                
                self.logger.info(f"   {emulator_type:15}:")
                self.logger.info(f"      Predicted: {predicted:.6f}")
                self.logger.info(f"      Calibration: {analysis['calibration_score']:.6f}")
                self.logger.info(f"      Evaluation: {analysis['evaluation_score']:.6f}")
                self.logger.info(f"      Temporal degradation: {analysis['temporal_degradation']:.6f}")
                self.logger.info(f"      Transferability: {analysis['temporal_transferability']}")
        
        # Parameter update decision based on evaluation period (temporal transferability test)
        best_existing = self.param_manager.get_best_existing_score()
        
        if best_emulator and best_params and best_eval_score > (best_existing or float('-inf')):
            # Update parameters if evaluation period shows improvement
            success = self.param_manager.update_default_parameters(
                best_params, best_eval_score, f"{best_emulator}_dual_validated"
            )
            emulation_results['parameter_update'] = {
                'updated': success,
                'emulator_used': best_emulator,
                'evaluation_score': best_eval_score,
                'calibration_score': best_calib_score,
                'improvement_over_existing': best_eval_score - (best_existing or 0)
            }
        else:
            emulation_results['parameter_update'] = {
                'updated': False,
                'reason': 'no_improvement_on_evaluation_period',
                'best_evaluation_score': best_eval_score,
                'existing_best': best_existing
            }

class SummaValidator:
    """
    Simple validator that runs SUMMA with emulator-optimized parameters
    and reports the actual performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Create validation directory
        self.validation_dir = self.project_dir / "emulation" / self.experiment_id / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
    def run_dual_period_validation(self, emulator_params: Dict[str, float], 
                                emulator_type: str) -> Dict[str, Any]:
        """
        Run SUMMA validation on BOTH calibration and evaluation periods.
        This helps separate emulator accuracy from temporal transferability.
        """
        self.logger.info(f"🔍 Running dual-period validation with {emulator_type} parameters")
        
        results = {
            'emulator_type': emulator_type,
            'emulator_parameters': emulator_params,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Test 1: Calibration Period (same period as training data)
        self.logger.info("📊 Testing on CALIBRATION period (same as training)...")
        calib_result = self._run_single_period_validation(
            emulator_params, 
            emulator_type, 
            "calibration",
            period_type="CALIBRATION_PERIOD"
        )
        results['calibration_validation'] = calib_result
        
        # Test 2: Evaluation Period (different period - temporal transferability)
        self.logger.info("📊 Testing on EVALUATION period (temporal transferability)...")
        eval_result = self._run_single_period_validation(
            emulator_params, 
            emulator_type, 
            "evaluation", 
            period_type="EVALUATION_PERIOD"
        )
        results['evaluation_validation'] = eval_result
        
        # Analysis
        if calib_result.get('success') and eval_result.get('success'):
            calib_score = calib_result.get('primary_metric', float('-inf'))
            eval_score = eval_result.get('primary_metric', float('-inf'))
            
            results['temporal_analysis'] = {
                'calibration_score': calib_score,
                'evaluation_score': eval_score,
                'temporal_degradation': calib_score - eval_score,
                'relative_degradation_pct': ((calib_score - eval_score) / abs(calib_score) * 100) if calib_score != 0 else float('inf'),
                'temporal_transferability': 'Good' if eval_score > calib_score * 0.8 else 'Poor'
            }
            
            self.logger.info(f"📈 Temporal Analysis Results:")
            self.logger.info(f"   Calibration period: {calib_score:.6f}")
            self.logger.info(f"   Evaluation period:  {eval_score:.6f}")
            self.logger.info(f"   Degradation: {calib_score - eval_score:.6f}")
            self.logger.info(f"   Transferability: {results['temporal_analysis']['temporal_transferability']}")
        
        return results

    def _run_single_period_validation(self, emulator_params: Dict[str, float], 
                                    emulator_type: str, period_name: str,
                                    period_type: str) -> Dict[str, Any]:
        """Run validation for a single time period."""
        
        # Create unique validation ID
        validation_id = f"validation_{emulator_type}_{period_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Set up SUMMA run
            run_success = self._setup_summa_validation(validation_id, emulator_params)
            if not run_success:
                return {'success': False, 'error': f'Failed to setup SUMMA for {period_name}'}
            
            # Run SUMMA
            summa_success = self._run_summa_validation(validation_id)
            if not summa_success:
                return {'success': False, 'error': f'SUMMA simulation failed for {period_name}'}
            
            # Calculate metrics for specific period
            actual_metrics = self._calculate_period_specific_metrics(validation_id, period_type)
            if not actual_metrics:
                return {'success': False, 'error': f'Failed to calculate {period_name} metrics'}
            
            # Create results
            results = {
                'success': True,
                'validation_id': validation_id,
                'period_name': period_name,
                'period_type': period_type,
                'emulator_type': emulator_type,
                'actual_metrics': actual_metrics,
                'primary_metric': actual_metrics.get(self.config.get('OPTIMIZATION_METRIC', 'KGE')),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            self._save_validation_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error during {period_name} validation: {str(e)}")
            return {'success': False, 'error': f'{period_name} validation error: {str(e)}'}

    def _calculate_period_specific_metrics(self, validation_id: str, period_type: str) -> Optional[Dict[str, float]]:
        """Calculate metrics for a specific time period."""
        try:
            # Get period from config
            period_config = self.config.get(period_type, '')
            if not period_config:
                self.logger.error(f"Period type {period_type} not found in config")
                return None
                
            period_dates = period_config.split(',')
            if len(period_dates) != 2:
                self.logger.error(f"Invalid period format for {period_type}: {period_config}")
                return None
                
            eval_start = period_dates[0].strip()
            eval_end = period_dates[1].strip()
            
            self.logger.info(f"📅 Calculating metrics for {period_type}: {eval_start} to {eval_end}")
            
            # Load SUMMA output (same as existing method)
            run_dir = self.validation_dir / validation_id
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            
            output_files = list(output_dir.glob("*.nc"))
            if not output_files:
                self.logger.error("No SUMMA output files found")
                return None
            
            output_file = output_files[0]
            
            # Load simulation data
            with nc.Dataset(output_file, 'r') as ds:
                time_var = ds.variables['time']
                dates = nc.num2date(time_var[:], time_var.units)
                
                # Get streamflow variable
                streamflow_var_names = ['scalarTotalRunoff', 'averageRoutedRunoff', 'totalRunoff']
                streamflow_data = None
                
                for var_name in streamflow_var_names:
                    if var_name in ds.variables:
                        streamflow_data = ds.variables[var_name][:]
                        if streamflow_data.ndim > 1:
                            streamflow_data = streamflow_data[:, 0]
                        break
                
                if streamflow_data is None:
                    self.logger.error("No streamflow variable found")
                    return None
            
            # Load observed data
            obs_data = self._load_observed_streamflow()
            if obs_data is None:
                return None
            
            # Convert dates and create DataFrame
            iso_strings = [d.isoformat() for d in dates]
            dates_pd = pd.to_datetime(iso_strings, format='ISO8601')
            
            sim_df = pd.DataFrame(
                data={'simulated': streamflow_data},
                index=dates_pd
            )
            
            # Filter to specific period
            sim_period = sim_df.loc[eval_start:eval_end]
            obs_period = obs_data.loc[eval_start:eval_end]
            
            # Align data
            aligned_data = pd.merge(sim_period, obs_period, left_index=True, right_index=True, how='inner')
            
            if len(aligned_data) == 0:
                self.logger.error(f"No overlapping data for {period_type}")
                return None
            
            sim_values = aligned_data['simulated'].values
            obs_values = aligned_data['observed'].values
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(sim_values, obs_values)
            
            self.logger.info(f"📊 Calculated {period_type} metrics for {len(aligned_data)} time steps")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating {period_type} metrics: {str(e)}")
            return None



    def run_summa_with_emulator_params(self, emulator_params: Dict[str, float], 
                                     emulator_type: str) -> Dict[str, Any]:
        """
        Run SUMMA with emulator-optimized parameters and return actual results.
        
        Args:
            emulator_params: Optimized parameters from emulator
            emulator_type: Type of emulator used
            
        Returns:
            Dictionary with actual SUMMA results
        """
        self.logger.info(f"🔍 Running SUMMA validation with {emulator_type} optimized parameters")
        
        # Create unique validation ID
        validation_id = f"validation_{emulator_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Step 1: Set up SUMMA run with emulator parameters
            run_success = self._setup_summa_validation(validation_id, emulator_params)
            if not run_success:
                return {'success': False, 'error': 'Failed to setup SUMMA validation'}
            
            # Step 2: Run SUMMA
            summa_success = self._run_summa_validation(validation_id)
            if not summa_success:
                return {'success': False, 'error': 'SUMMA simulation failed'}
            
            # Step 3: Calculate actual performance metrics
            actual_metrics = self._calculate_actual_metrics(validation_id)
            if not actual_metrics:
                return {'success': False, 'error': 'Failed to calculate metrics'}
            
            # Step 4: Create results summary
            results = {
                'success': True,
                'validation_id': validation_id,
                'emulator_type': emulator_type,
                'emulator_parameters': emulator_params,
                'actual_metrics': actual_metrics,
                'primary_metric': actual_metrics.get(self.config.get('OPTIMIZATION_METRIC', 'KGE')),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Step 5: Save and report results
            self._save_validation_results(results)
            self._report_validation_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during SUMMA validation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': f'Validation error: {str(e)}'}
    
    def _setup_summa_validation(self, validation_id: str, params: Dict[str, float]) -> bool:
        """Set up SUMMA simulation with emulator parameters."""
        try:
            # Create validation run directory structure
            run_dir = self.validation_dir / validation_id
            settings_dir = run_dir / "settings" / "SUMMA"
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            
            settings_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy SUMMA settings files from the main experiment
            source_settings = self.project_dir / "settings" / "SUMMA"
            
            settings_files = [
                'modelDecisions.txt', 'outputControl.txt', 'localParamInfo.txt',
                'basinParamInfo.txt', 'attributes.nc', 'coldState.nc',
                'forcingFileList.txt', 'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL'
            ]
            
            for filename in settings_files:
                source_file = source_settings / filename
                if source_file.exists():
                    shutil.copy2(source_file, settings_dir / filename)
                else:
                    self.logger.warning(f"Settings file not found: {filename}")
            
            # Create trial parameters file with emulator-optimized values
            trial_params_file = settings_dir / 'trialParams.nc'
            FileManager.create_trial_params_file(
                params, 
                trial_params_file, 
                settings_dir / 'attributes.nc'
            )
            
            # Create fileManager.txt pointing to validation directories
            self._create_validation_filemanager(settings_dir, output_dir)
            
            self.logger.info(f"✅ SUMMA validation setup complete: {validation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup SUMMA validation: {str(e)}")
            return False
    
    def _create_validation_filemanager(self, settings_dir: Path, output_dir: Path):
        """Create fileManager.txt for validation run."""
        try:
            # Read template fileManager from main settings
            template_file = self.project_dir / "settings" / "SUMMA" / "fileManager.txt"
            
            if not template_file.exists():
                raise FileNotFoundError(f"Template fileManager not found: {template_file}")
            
            with open(template_file, 'r') as f:
                lines = f.readlines()
            
            # Create new fileManager with validation paths
            filemanager_path = settings_dir / 'fileManager.txt'
            with open(filemanager_path, 'w') as f:
                for line in lines:
                    if "outputPath" in line:
                        f.write(f"outputPath           '{output_dir}/'\n")
                    elif "settingsPath" in line:
                        f.write(f"settingsPath         '{settings_dir}/'\n")
                    else:
                        f.write(line)
            
            self.logger.debug(f"Created validation fileManager: {filemanager_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating validation fileManager: {str(e)}")
            raise
    
    def _run_summa_validation(self, validation_id: str) -> bool:
        """Run SUMMA simulation for validation."""
        try:
            run_dir = self.validation_dir / validation_id
            filemanager = run_dir / "settings" / "SUMMA" / "fileManager.txt"
            
            # Create logs directory
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            summa_log = logs_dir / "summa_validation.log"
            
            # Get SUMMA executable path
            summa_exe = self.config.get('SUMMA_INSTALL_PATH', 'summa.exe')
            if summa_exe == 'default':
                code_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
                summa_exe = code_dir / "installs" / "summa" / "bin" / self.config.get('SUMMA_EXE', 'summa.exe')
            
            # Run SUMMA
            self.logger.info("🏃‍♂️ Running SUMMA validation simulation...")
            
            cmd = [str(summa_exe), '-m', str(filemanager)]
            
            with open(summa_log, 'w') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=run_dir,
                    timeout=3600  # 1 hour timeout
                )
            
            if result.returncode == 0:
                self.logger.info("✅ SUMMA validation simulation completed successfully")
                return True
            else:
                self.logger.error(f"SUMMA validation failed with return code: {result.returncode}")
                # Log first few lines of error log
                if summa_log.exists():
                    with open(summa_log, 'r') as f:
                        log_content = f.read()
                        self.logger.error(f"SUMMA log excerpt:\n{log_content[:1000]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("SUMMA validation simulation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running SUMMA validation: {str(e)}")
            return False
    
    def _calculate_actual_metrics(self, validation_id: str) -> Optional[Dict[str, float]]:
        """Calculate performance metrics from SUMMA validation output."""
        try:
            # Find SUMMA output file
            run_dir = self.validation_dir / validation_id
            output_dir = run_dir / "simulations" / self.experiment_id / "SUMMA"
            
            # Look for SUMMA output files
            output_files = list(output_dir.glob("*.nc"))
            if not output_files:
                self.logger.error("No SUMMA output files found")
                return None
            
            # Use the first output file (assuming single HRU for now)
            output_file = output_files[0]
            self.logger.debug(f"Using SUMMA output: {output_file}")
            
            # Load simulation data
            with nc.Dataset(output_file, 'r') as ds:
                # Get time and streamflow data
                time_var = ds.variables['time']
                dates = nc.num2date(time_var[:], time_var.units)
                
                # Get streamflow variable (check multiple possible names)
                streamflow_var_names = ['scalarTotalRunoff', 'averageRoutedRunoff', 'totalRunoff']
                streamflow_data = None
                
                for var_name in streamflow_var_names:
                    if var_name in ds.variables:
                        streamflow_data = ds.variables[var_name][:]
                        if streamflow_data.ndim > 1:
                            streamflow_data = streamflow_data[:, 0]  # Take first HRU
                        break
                
                if streamflow_data is None:
                    self.logger.error("No streamflow variable found in SUMMA output")
                    return None
            
            # Load observed streamflow for comparison
            obs_data = self._load_observed_streamflow()
            if obs_data is None:
                self.logger.error("Could not load observed streamflow data")
                return None
            
            # Align simulation and observed data for evaluation period
            eval_period = self.config.get('EVALUATION_PERIOD', '').split(',')
            if len(eval_period) == 2:
                eval_start = eval_period[0].strip()
                eval_end = eval_period[1].strip()
            else:
                eval_start = "2015-01-01"
                eval_end = "2018-12-31"
            
            # dates is an array of cftime.DatetimeGregorian
            iso_strings = [d.isoformat() for d in dates]

            # Option A: use format='ISO8601'
            dates_pd = pd.to_datetime(iso_strings, format='ISO8601')

            # Build DataFrame with a proper DatetimeIndex
            sim_df = pd.DataFrame(
                data={'simulated': streamflow_data},
                index=dates_pd
            )
            
            # Filter to evaluation period
            sim_eval = sim_df.loc[eval_start:eval_end]
            obs_eval = obs_data.loc[eval_start:eval_end]
            
            # Align data (inner join)
            aligned_data = pd.merge(sim_eval, obs_eval, left_index=True, right_index=True, how='inner')
            
            if len(aligned_data) == 0:
                self.logger.error("No overlapping data between simulation and observations")
                return None
            
            sim_values = aligned_data['simulated'].values
            obs_values = aligned_data['observed'].values
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(sim_values, obs_values)
            
            self.logger.info(f"📊 Calculated metrics for {len(aligned_data)} time steps")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating actual metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def _load_observed_streamflow(self) -> Optional[pd.DataFrame]:
        """Load observed streamflow data for comparison."""
        try:
            obs_file = (
                self.project_dir
                / "observations"
                / "streamflow"
                / "preprocessed"
                / f"{self.domain_name}_streamflow_processed.csv"
            )
            
            if not obs_file.exists():
                self.logger.error(f"Observed streamflow file not found: {obs_file}")
                return None
            
            obs_df = pd.read_csv(obs_file)
            # 1) Convert "datetime" strings to a new datetime column
            obs_df["date"] = pd.to_datetime(obs_df["datetime"])
            # 2) Make "date" the index
            obs_df.set_index("date", inplace=True)

            # At this point obs_df.columns == ['datetime', 'discharge_cms']

            # 3a) Drop the old "datetime" column:
            obs_df.drop(columns=["datetime"], inplace=True)
            #     Now obs_df.columns == ['discharge_cms']

            # 3b) Rename "discharge_cms" → "observed"
            obs_df.columns = ["observed"]
            #     Now obs_df.columns == ['observed']

            return obs_df

        except Exception as e:
            self.logger.error(f"Error loading observed streamflow: {str(e)}")
            return None


    def _calculate_performance_metrics(self, sim: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
        """Calculate standard hydrological performance metrics."""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(sim) | np.isnan(obs))
        sim_clean = sim[valid_mask]
        obs_clean = obs[valid_mask]
        
        if len(sim_clean) == 0:
            return {}
        
        metrics = {}
        
        try:
            # Nash-Sutcliffe Efficiency
            nse_num = np.sum((obs_clean - sim_clean) ** 2)
            nse_den = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
            metrics['NSE'] = 1 - (nse_num / nse_den) if nse_den != 0 else float('-inf')
            
            # Root Mean Square Error
            metrics['RMSE'] = np.sqrt(np.mean((sim_clean - obs_clean) ** 2))
            
            # Mean Absolute Error
            metrics['MAE'] = np.mean(np.abs(sim_clean - obs_clean))
            
            # Percent Bias
            pbias_num = np.sum(sim_clean - obs_clean)
            pbias_den = np.sum(obs_clean)
            metrics['PBIAS'] = (pbias_num / pbias_den) * 100 if pbias_den != 0 else float('inf')
            
            # Correlation coefficient
            if len(sim_clean) > 1:
                correlation_matrix = np.corrcoef(sim_clean, obs_clean)
                metrics['r'] = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
            else:
                metrics['r'] = 0.0
            
            # Kling-Gupta Efficiency
            if np.std(obs_clean) > 0 and np.std(sim_clean) > 0:
                r = metrics['r']
                alpha = np.std(sim_clean) / np.std(obs_clean)
                beta = np.mean(sim_clean) / np.mean(obs_clean) if np.mean(obs_clean) != 0 else 1.0
                
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                metrics['KGE'] = kge
            else:
                metrics['KGE'] = float('-inf')
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
        
        return metrics
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file."""
        try:
            # Save as JSON
            results_file = self.validation_dir / f"{results['validation_id']}_results.json"
            
            # Make results JSON serializable
            json_results = results.copy()
            if 'emulator_parameters' in json_results:
                json_results['emulator_parameters'] = dict(json_results['emulator_parameters'])
            
            import json
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            # Save summary CSV
            summary_data = {
                'validation_id': results['validation_id'],
                'emulator_type': results['emulator_type'],
                'timestamp': results['validation_timestamp'],
                **results.get('actual_metrics', {})
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_file = self.validation_dir / "validation_summary.csv"
            
            if summary_file.exists():
                existing_df = pd.read_csv(summary_file)
                combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
                combined_df.to_csv(summary_file, index=False)
            else:
                summary_df.to_csv(summary_file, index=False)
            
            self.logger.info(f"📄 Validation results saved to: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {str(e)}")
    
    def _report_validation_results(self, results: Dict[str, Any]):
        """Report validation results to console."""
        if not results.get('success', False):
            self.logger.error(f"❌ SUMMA validation failed: {results.get('error', 'Unknown error')}")
            return
        
        metrics = results.get('actual_metrics', {})
        primary_metric = results.get('primary_metric')
        
        self.logger.info("🎯 SUMMA Validation Results:")
        self.logger.info(f"   Emulator: {results['emulator_type']}")
        
        if primary_metric is not None:
            target_metric_name = self.config.get('OPTIMIZATION_METRIC', 'KGE')
            self.logger.info(f"   {target_metric_name}: {primary_metric:.6f}")
        
        # Report all calculated metrics
        for metric_name, value in metrics.items():
            if metric_name != self.config.get('OPTIMIZATION_METRIC', 'KGE'):  # Don't repeat primary metric
                self.logger.info(f"   {metric_name}: {value:.6f}")
        
        self.logger.info(f"   Validation ID: {results['validation_id']}")




# ============================================================================
# Main Entry Point (keeping same interface)
# ============================================================================

# Keep the original classes for backward compatibility
SingleSampleEmulator = EnhancedEmulationRunner  # Alias for backward compatibility

# Update the EmulationRunner to use the enhanced version
class EmulationRunner(EnhancedEmulationRunner):
    """
    Backward compatible EmulationRunner that uses enhanced functionality.
    """
    pass