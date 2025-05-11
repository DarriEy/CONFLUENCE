# In utils/analysis_utils/analysis_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List

from utils.evaluation.evaluation_utils import SensitivityAnalyzer, DecisionAnalyzer, Benchmarker # type: ignore
from utils.data.data_utils import BenchmarkPreprocessor # type: ignore
from utils.reporting.result_vizualisation_utils import BenchmarkVizualiser # type: ignore
from utils.models.fuse_utils import FuseDecisionAnalyzer # type: ignore



class AnalysisManager:
    """Manages all analysis operations including benchmarking, sensitivity, and decision analysis."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Analysis Manager.
        
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
        
    def run_benchmarking(self) -> Optional[Path]:
        """
        Run benchmarking analysis.
        
        Returns:
            Path to benchmark results file or None if failed
        """
        self.logger.info("Starting benchmarking analysis")
        
        try:
            # Preprocess data for benchmarking
            preprocessor = BenchmarkPreprocessor(self.config, self.logger)
            
            # Extract calibration and evaluation periods
            calib_start = self.config['CALIBRATION_PERIOD'].split(',')[0].strip()
            eval_end = self.config['EVALUATION_PERIOD'].split(',')[1].strip()
            
            benchmark_data = preprocessor.preprocess_benchmark_data(calib_start, eval_end)
            
            # Run benchmarking
            benchmarker = Benchmarker(self.config, self.logger)
            benchmark_results = benchmarker.run_benchmarking()
            
            # Visualize benchmark results
            visualizer = BenchmarkVizualiser(self.config, self.logger)
            visualizer.visualize_benchmarks(benchmark_results)
            
            # Return path to benchmark results
            benchmark_file = self.project_dir / "evaluation" / "benchmark_scores.csv"
            if benchmark_file.exists():
                self.logger.info(f"Benchmarking completed successfully: {benchmark_file}")
                return benchmark_file
            else:
                self.logger.warning("Benchmarking completed but results file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during benchmarking: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_sensitivity_analysis(self) -> Optional[Dict]:
        """
        Run sensitivity analysis for supported models.
        
        Returns:
            Dictionary of sensitivity results or None if failed
        """
        self.logger.info("Starting sensitivity analysis")
        
        # Check if sensitivity analysis is enabled
        if not self.config.get('RUN_SENSITIVITY_ANALYSIS', True):
            self.logger.info("Sensitivity analysis is disabled in configuration")
            return None
        
        sensitivity_results = {}
        
        try:
            hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
            
            for model in hydrological_models:
                model = model.strip()
                
                if model == 'SUMMA':
                    sensitivity_results[model] = self._run_summa_sensitivity_analysis()
                else:
                    self.logger.info(f"Sensitivity analysis not implemented for model: {model}")
                    
            return sensitivity_results if sensitivity_results else None
            
        except Exception as e:
            self.logger.error(f"Error during sensitivity analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_summa_sensitivity_analysis(self) -> Optional[Dict]:
        """Run sensitivity analysis for SUMMA model."""
        self.logger.info("Running SUMMA sensitivity analysis")
        
        sensitivity_analyzer = SensitivityAnalyzer(self.config, self.logger)
        results_file = self.project_dir / "optimisation" / f"{self.experiment_id}_parallel_iteration_results.csv"
        
        if not results_file.exists():
            self.logger.error(f"Calibration results file not found: {results_file}")
            return None
        
        return sensitivity_analyzer.run_sensitivity_analysis(results_file)
    
    def run_decision_analysis(self) -> Optional[Dict]:
        """
        Run decision analysis for supported models.
        
        Returns:
            Dictionary containing decision analysis results or None if failed
        """
        self.logger.info("Starting decision analysis")
        
        # Check if decision analysis is enabled
        if not self.config.get('RUN_DECISION_ANALYSIS', True):
            self.logger.info("Decision analysis is disabled in configuration")
            return None
        
        decision_results = {}
        
        try:
            hydrological_models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
            
            for model in hydrological_models:
                model = model.strip()
                
                if model == 'SUMMA':
                    decision_results[model] = self._run_summa_decision_analysis()
                elif model == 'FUSE':
                    decision_results[model] = self._run_fuse_decision_analysis()
                else:
                    self.logger.info(f"Decision analysis not implemented for model: {model}")
                    
            return decision_results if decision_results else None
            
        except Exception as e:
            self.logger.error(f"Error during decision analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_summa_decision_analysis(self) -> Dict:
        """Run decision analysis for SUMMA model."""
        self.logger.info("Running SUMMA decision analysis")
        
        decision_analyzer = DecisionAnalyzer(self.config, self.logger)
        results_file, best_combinations = decision_analyzer.run_full_analysis()
        
        self.logger.info("SUMMA decision analysis completed")
        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info("Best combinations for each metric:")
        for metric, data in best_combinations.items():
            self.logger.info(f"  {metric}: score = {data['score']:.3f}")
        
        return {
            'results_file': results_file,
            'best_combinations': best_combinations
        }
    
    def _run_fuse_decision_analysis(self) -> Dict:
        """Run decision analysis for FUSE model."""
        self.logger.info("Running FUSE decision analysis")
        
        fuse_analyzer = FuseDecisionAnalyzer(self.config, self.logger)
        results = fuse_analyzer.run_decision_analysis()
        
        self.logger.info("FUSE decision analysis completed")
        return results
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get status of various analyses.
        
        Returns:
            Dictionary containing analysis status information
        """
        status = {
            'benchmarking_complete': (self.project_dir / "evaluation" / "benchmark_scores.csv").exists(),
            'sensitivity_analysis_available': self.config.get('RUN_SENSITIVITY_ANALYSIS', True),
            'decision_analysis_available': self.config.get('RUN_DECISION_ANALYSIS', True),
            'optimization_results_exist': (self.project_dir / "optimisation" / f"{self.experiment_id}_parallel_iteration_results.csv").exists(),
        }
        
        # Check for analysis outputs
        if (self.project_dir / "plots" / "sensitivity_analysis").exists():
            status['sensitivity_plots_exist'] = True
        
        if (self.project_dir / "optimisation").exists():
            status['decision_analysis_results_exist'] = any(
                file.name.endswith('_model_decisions_comparison.csv')
                for file in (self.project_dir / "optimisation").glob('*.csv')
            )
        
        return status
    
    def validate_analysis_requirements(self) -> Dict[str, bool]:
        """
        Validate that requirements are met for running analyses.
        
        Returns:
            Dictionary indicating which analyses can be run
        """
        requirements = {
            'benchmarking': True,  # Benchmarking has minimal requirements
            'sensitivity_analysis': False,
            'decision_analysis': False
        }
        
        # Check for optimization results (required for sensitivity analysis)
        optimization_results = self.project_dir / "optimisation" / f"{self.experiment_id}_parallel_iteration_results.csv"
        if optimization_results.exists():
            requirements['sensitivity_analysis'] = True
        
        # Check for model outputs (required for decision analysis)
        simulation_dir = self.project_dir / "simulations" / self.experiment_id
        if simulation_dir.exists():
            requirements['decision_analysis'] = True
        
        # Check for processed observations (required for all analyses)
        obs_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        if not obs_file.exists():
            self.logger.warning("Processed observations not found - all analyses may fail")
            requirements = {key: False for key in requirements}
        
        return requirements