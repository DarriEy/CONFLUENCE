# In utils/optimization_utils/results_utils.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional


class OptimizationResultsManager:
    """Manages saving and loading of optimization results."""
    
    def __init__(self, project_dir: Path, experiment_id: str, logger: logging.Logger):
        self.project_dir = project_dir
        self.experiment_id = experiment_id
        self.logger = logger
        self.opt_dir = project_dir / "optimisation"
        self.opt_dir.mkdir(parents=True, exist_ok=True)
    
    def save_optimization_results(self, results: Dict[str, Any], algorithm: str, target_metric: str = 'KGE') -> Optional[Path]:
        """
        Save optimization results to a CSV file for compatibility with other parts of CONFLUENCE.
        
        Args:
            results: Dictionary with optimization results
            algorithm: Name of the optimization algorithm used
            target_metric: Target optimization metric (default: 'KGE')
        
        Returns:
            Path to the saved results file or None if save failed
        """
        try:
            # Get best parameters and score
            best_params = results.get('best_parameters', {})
            best_score = results.get('best_score', None)
            
            if not best_params or best_score is None:
                self.logger.warning("No valid optimization results to save")
                return None
            
            # Create DataFrame from best parameters
            results_data = {}
            
            # First add iteration column
            results_data['iteration'] = [0]
            
            # Add score column with appropriate name based on the target metric
            results_data[target_metric] = [best_score]
            
            # Add parameter columns
            for param_name, values in best_params.items():
                if isinstance(values, np.ndarray) and len(values) > 1:
                    # For parameters with multiple values (like HRU-level parameters),
                    # save the mean value
                    results_data[param_name] = [float(np.mean(values))]
                else:
                    # For scalar parameters or single-value arrays
                    results_data[param_name] = [float(values[0]) if isinstance(values, np.ndarray) else float(values)]
            
            # Create DataFrame
            results_df = pd.DataFrame(results_data)
            
            # Save to standard location
            results_file = self.opt_dir / f"{self.experiment_id}_parallel_iteration_results.csv"
            results_df.to_csv(results_file, index=False)
            self.logger.info(f"Saved optimization results to {results_file}")
            
            # Also save detailed results to optimizer-specific files
            self._save_optimization_history(results.get('history', []), algorithm, target_metric, best_params)
            
            return results_file
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _save_optimization_history(self, history: list, algorithm: str, target_metric: str, best_params: Dict[str, Any]):
        """
        Save optimization history to algorithm-specific file.
        
        Args:
            history: List of optimization history
            algorithm: Name of the optimization algorithm
            target_metric: Target optimization metric
            best_params: Best parameters from optimization
        """
        if not history:
            return
        
        try:
            # Extract iteration history data
            history_data = {
                'iteration': [],
                target_metric: [],
            }
            
            # Add parameter columns to history data
            for param_name in best_params.keys():
                history_data[param_name] = []
            
            # Fill history data
            for h in history:
                history_data['iteration'].append(h.get('iteration', 0))
                history_data[target_metric].append(h.get('best_score', np.nan))
                
                # Add parameter values
                if 'best_params' in h and h['best_params']:
                    for param_name, values in h['best_params'].items():
                        if param_name in history_data:
                            if isinstance(values, np.ndarray) and len(values) > 1:
                                history_data[param_name].append(float(np.mean(values)))
                            else:
                                val = float(values[0]) if isinstance(values, np.ndarray) else float(values)
                                history_data[param_name].append(val)
            
            # Create DataFrame and save
            history_df = pd.DataFrame(history_data)
            history_file = self.opt_dir / f"{self.experiment_id}_{algorithm.lower()}_history.csv"
            history_df.to_csv(history_file, index=False)
            self.logger.info(f"Saved optimization history to {history_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization history: {str(e)}")
    
    def load_optimization_results(self, filename: str = None) -> pd.DataFrame:
        """
        Load optimization results from CSV file.
        
        Args:
            filename: Name of the file to load. If None, uses default naming.
        
        Returns:
            DataFrame with optimization results
        """
        if filename is None:
            filename = f"{self.experiment_id}_parallel_iteration_results.csv"
        
        results_file = self.opt_dir / filename
        
        if not results_file.exists():
            raise FileNotFoundError(f"Optimization results file not found: {results_file}")
        
        return pd.read_csv(results_file)