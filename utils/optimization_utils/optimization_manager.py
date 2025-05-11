# In utils/optimization_utils/optimization_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union

# Optimizers
from utils.optimization_utils.dds_optimizer import DDSOptimizer
from utils.optimization_utils.pso_optimizer import PSOOptimizer
from utils.optimization_utils.sce_ua_optimizer import SCEUAOptimizer

# Results management
from utils.optimization_utils.results_utils import OptimizationResultsManager

# Emulation
from utils.optimization_utils.emulation_runner import EmulationRunner


class OptimizationManager:
    """Manages all optimization operations including calibration and emulation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Optimization Manager.
        
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
        
        # Initialize results manager
        self.results_manager = OptimizationResultsManager(
            self.project_dir,
            self.experiment_id,
            self.logger
        )
        
        # Initialize emulation runner
        self.emulation_runner = EmulationRunner(self.config, self.logger)
        
        # Define optimizer mapping
        self.optimizers = {
            'PSO': PSOOptimizer,
            'SCE-UA': SCEUAOptimizer,
            'DDS': DDSOptimizer
        }
        
        # Define optimizer run method names
        self.optimizer_methods = {
            'PSO': 'run_pso_optimization',
            'SCE-UA': 'run_sceua_optimization',
            'DDS': 'run_dds_optimization'
        }
    
    def calibrate_model(self) -> Optional[Path]:
        """
        Calibrate the model using the specified optimization algorithm.
        
        Returns:
            Path to calibration results file or None if calibration failed
        """
        self.logger.info("Starting model calibration")
        
        # Check if iterative optimization is enabled
        if not self.config.get('RUN_ITERATIVE_OPTIMISATION', False):
            self.logger.info("Iterative optimization is disabled in configuration")
            return None
        
        # Get the optimization algorithm from config
        opt_algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO')
        
        try:
            hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
            
            for model in hydrological_models:
                model = model.strip()
                
                if model == 'SUMMA':
                    return self._calibrate_summa(opt_algorithm)
                else:
                    self.logger.warning(f"Calibration for model {model} not yet implemented")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error during model calibration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _calibrate_summa(self, algorithm: str) -> Optional[Path]:
        """
        Calibrate SUMMA model using specified algorithm.
        
        Args:
            algorithm: Optimization algorithm to use
            
        Returns:
            Path to results file or None if failed
        """
        # Create optimization directory if it doesn't exist
        opt_dir = self.project_dir / "optimisation"
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Get optimizer class
        optimizer_class = self.optimizers.get(algorithm)
        
        if optimizer_class is None:
            self.logger.error(f"Optimisation algorithm {algorithm} not supported")
            return None
        
        # Get optimizer method name
        method_name = self.optimizer_methods.get(algorithm)
        
        if method_name is None:
            self.logger.error(f"No run method defined for optimizer {algorithm}")
            return None
        
        try:
            # Initialize optimizer
            self.logger.info(f"Using {algorithm} optimization")
            optimizer = optimizer_class(self.config, self.logger)
            
            # Run optimization
            result = getattr(optimizer, method_name)()
            
            # Save results to standard location
            target_metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
            results_file = self.results_manager.save_optimization_results(
                result, algorithm, target_metric
            )
            
            if results_file:
                self.logger.info(f"Calibration completed successfully: {results_file}")
                return results_file
            else:
                self.logger.warning("Calibration completed but results file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during {algorithm} optimization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_emulation(self) -> Optional[Dict]:
        """
        Run model parameter emulation workflow.
        
        Returns:
            Dictionary with emulation results or None if failed
        """
        self.logger.info("Starting model parameter emulation")
        
        try:
            # Run the complete emulation workflow
            results = self.emulation_runner.run_emulation_workflow()
            
            if results:
                self.logger.info("Model parameter emulation completed successfully")
            else:
                self.logger.warning("Model parameter emulation did not produce results")
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error during model parameter emulation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get status of optimization operations.
        
        Returns:
            Dictionary containing optimization status information
        """
        status = {
            'iterative_optimization_enabled': self.config.get('RUN_ITERATIVE_OPTIMISATION', False),
            'optimization_algorithm': self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'PSO'),
            'optimization_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'optimization_dir': str(self.project_dir / "optimisation"),
            'results_exist': False,
            'emulation_enabled': self.config.get('RUN_LARGE_SAMPLE_EMULATION', False),
            'rf_emulation_enabled': self.config.get('RUN_RANDOM_FOREST_EMULATION', False)
        }
        
        # Check for optimization results
        results_file = self.project_dir / "optimisation" / f"{self.experiment_id}_parallel_iteration_results.csv"
        status['results_exist'] = results_file.exists()
        
        # Check for emulation outputs
        emulation_dir = self.project_dir / "emulation" / self.experiment_id
        if emulation_dir.exists():
            status['emulation_parameter_sets'] = (emulation_dir / "parameter_sets.nc").exists()
            status['ensemble_runs_exist'] = (emulation_dir / "ensemble_runs").exists()
            status['performance_metrics_exist'] = (emulation_dir / "ensemble_analysis" / "performance_metrics.csv").exists()
            status['rf_emulation_complete'] = (emulation_dir / "rf_emulation" / "optimized_parameters.csv").exists()
        
        return status
    
    def validate_optimization_configuration(self) -> Dict[str, bool]:
        """
        Validate optimization configuration settings.
        
        Returns:
            Dictionary indicating validation results
        """
        validation = {
            'algorithm_valid': False,
            'model_supported': False,
            'parameters_defined': False,
            'metric_valid': False
        }
        
        # Check algorithm
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', '')
        validation['algorithm_valid'] = algorithm in self.optimizers
        
        # Check model support
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        validation['model_supported'] = 'SUMMA' in [m.strip() for m in models]
        
        # Check parameters to calibrate
        local_params = self.config.get('PARAMS_TO_CALIBRATE', '')
        basin_params = self.config.get('BASIN_PARAMS_TO_CALIBRATE', '')
        validation['parameters_defined'] = bool(local_params or basin_params)
        
        # Check metric
        valid_metrics = ['KGE', 'NSE', 'RMSE', 'MAE', 'KGEp']
        metric = self.config.get('OPTIMIZATION_METRIC', '')
        validation['metric_valid'] = metric in valid_metrics
        
        return validation
    
    def get_available_optimizers(self) -> Dict[str, str]:
        """
        Get list of available optimization algorithms.
        
        Returns:
            Dictionary mapping algorithm names to descriptions
        """
        return {
            'PSO': 'Particle Swarm Optimization',
            'SCE-UA': 'Shuffled Complex Evolution',
            'DDS': 'Dynamically Dimensioned Search'
        }
    
    def load_optimization_results(self, filename: str = None) -> Optional[Dict]:
        """
        Load optimization results from file.
        
        Args:
            filename: Name of results file to load
            
        Returns:
            Dictionary with optimization results or None if failed
        """
        try:
            results_df = self.results_manager.load_optimization_results(filename)
            
            # Convert DataFrame to dictionary format
            results = {
                'parameters': results_df.to_dict(orient='records'),
                'best_iteration': results_df.iloc[0].to_dict(),
                'columns': results_df.columns.tolist()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading optimization results: {str(e)}")
            return None