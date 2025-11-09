# In utils/analysis_utils/analysis_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional

from utils.evaluation.decision_analysis import DecisionAnalyzer # type: ignore
from utils.evaluation.sensitivity_analysis import SensitivityAnalyzer # type: ignore
from utils.evaluation.benchmarking import Benchmarker, BenchmarkPreprocessor # type: ignore
from utils.reporting.result_vizualisation_utils import BenchmarkVizualiser # type: ignore
from utils.models.fuse_utils import FuseDecisionAnalyzer # type: ignore


class AnalysisManager:
    """
    Manages all analysis operations including benchmarking, sensitivity, and decision analysis.
    
    The AnalysisManager is responsible for coordinating various analyses that evaluate 
    hydrological model performance, parameter sensitivity, and model structure decisions. 
    These analyses provide critical insights for understanding model behavior, improving 
    model configurations, and quantifying uncertainty.
    
    Key responsibilities:
    - Benchmarking: Comparing model performance against simple reference models
    - Sensitivity Analysis: Evaluating parameter importance and uncertainty
    - Decision Analysis: Assessing the impact of model structure choices
    - Result Visualization: Creating plots and visualizations of analysis results
    
    The AnalysisManager works with multiple hydrological models, providing consistent
    interfaces for analysis operations across different model types. It integrates
    with other SYMFLUENCE components to ensure that analyses use the correct input data
    and that results are properly stored and visualized.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        data_dir (Path): Path to the SYMFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        experiment_id (str): ID of the current experiment
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Analysis Manager.
        
        Sets up the AnalysisManager with the necessary configuration and paths
        for coordinating various analysis operations. This establishes the foundation
        for benchmarking, sensitivity analysis, and decision analysis.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            logger (logging.Logger): Logger instance for recording operations
            
        Raises:
            KeyError: If essential configuration values are missing
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
    def run_benchmarking(self) -> Optional[Path]:
        """
        Run benchmarking analysis to evaluate model performance against reference models.
        
        Benchmarking compares the performance of sophisticated hydrological models
        against simple reference models (e.g., mean flow, seasonality model) to
        quantify the value added by the model's complexity. This process includes:
        
        1. Preprocessing observed data for the benchmark period
        2. Running simple benchmark models (e.g., mean, seasonality, persistence)
        3. Computing performance metrics for each benchmark
        4. Visualizing benchmark results for comparison
        
        Benchmarking provides a baseline for evaluating model performance and helps
        identify the minimum acceptable performance for a given watershed.
        
        Returns:
            Optional[Path]: Path to benchmark results file or None if benchmarking failed
            
        Raises:
            FileNotFoundError: If required observation data is missing
            ValueError: If date ranges are invalid
            Exception: For other errors during benchmarking
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
        Run sensitivity analysis to evaluate parameter importance and uncertainty.
        
        Sensitivity analysis quantifies how model parameters influence simulation
        results and performance metrics. This analysis helps:
        
        1. Identify which parameters have the most significant impact on model performance
        2. Quantify parameter uncertainty and its effect on predictions
        3. Guide model simplification by identifying insensitive parameters
        4. Inform calibration strategies by focusing on sensitive parameters
        
        The method iterates through configured hydrological models, running
        model-specific sensitivity analyses where supported. Currently,
        detailed sensitivity analysis is implemented for the SUMMA model.
        
        Returns:
            Optional[Dict]: Dictionary mapping model names to sensitivity results,
                          or None if the analysis was disabled or failed
                          
        Raises:
            FileNotFoundError: If required optimization results are missing
            Exception: For other errors during sensitivity analysis
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
        """
        Run sensitivity analysis for SUMMA model.
        
        This internal method handles the specifics of sensitivity analysis for the
        SUMMA hydrological model. It uses parameter sets from previous optimization
        results to assess parameter sensitivity through variance-based methods,
        parameter identifiability, and correlation analysis.
        
        The analysis requires optimization results from a prior calibration run,
        which contain multiple parameter sets and their corresponding performance metrics.
        
        Returns:
            Optional[Dict]: Dictionary containing sensitivity analysis results,
                          or None if the required files are missing
                          
        Raises:
            FileNotFoundError: If optimization results file is not found
            Exception: For other errors during analysis
        """
        self.logger.info("Running SUMMA sensitivity analysis")
        
        sensitivity_analyzer = SensitivityAnalyzer(self.config, self.logger)
        results_file = self.project_dir / "optimisation" / f"{self.experiment_id}_parallel_iteration_results.csv"
        
        if not results_file.exists():
            self.logger.error(f"Calibration results file not found: {results_file}")
            return None
        
        return sensitivity_analyzer.run_sensitivity_analysis(results_file)
    
    def run_decision_analysis(self) -> Optional[Dict]:
        """
        Run decision analysis to assess the impact of model structure choices.
        
        Decision analysis evaluates how different model structure decisions
        (e.g., process representations, numerical schemes) affect model performance.
        This analysis helps:
        
        1. Identify optimal combinations of model structure decisions
        2. Quantify the sensitivity of model performance to structure choices
        3. Guide model development by focusing on impactful process representations
        4. Provide insights into process understanding for a specific watershed
        
        The method iterates through configured hydrological models, running
        model-specific decision analyses where supported. Currently,
        decision analysis is implemented for SUMMA and FUSE models.
        
        Returns:
            Optional[Dict]: Dictionary mapping model names to decision analysis results,
                          or None if the analysis was disabled or failed
                          
        Raises:
            FileNotFoundError: If required model output files are missing
            ValueError: If model structure options are invalid
            Exception: For other errors during decision analysis
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
        """
        Run decision analysis for SUMMA model.
        
        This internal method handles the specifics of decision analysis for the
        SUMMA hydrological model. It evaluates different combinations of process
        representations defined in the DECISION_OPTIONS configuration.
        
        The analysis includes:
        1. Running SUMMA with different combinations of process representations
        2. Evaluating performance for each combination
        3. Identifying optimal combinations for different metrics
        4. Visualizing the impact of different decisions
        
        Returns:
            Dict: Dictionary containing decision analysis results, including:
                - results_file: Path to the CSV file with detailed results
                - best_combinations: Dictionary of best combinations for each metric
                
        Raises:
            Exception: For errors during SUMMA decision analysis
        """
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
        """
        Run decision analysis for FUSE model.
        
        This internal method handles the specifics of decision analysis for the
        FUSE hydrological model. FUSE is specifically designed for exploring model
        structure decisions, offering a flexible framework for combining different
        process representations.
        
        The analysis includes:
        1. Running FUSE with different structural configurations
        2. Evaluating performance for each configuration
        3. Identifying optimal model structures
        4. Visualizing the impact of different structural choices
        
        Returns:
            Dict: Dictionary containing FUSE decision analysis results
                
        Raises:
            Exception: For errors during FUSE decision analysis
        """
        self.logger.info("Running FUSE decision analysis")
        
        fuse_analyzer = FuseDecisionAnalyzer(self.config, self.logger)
        results = fuse_analyzer.run_decision_analysis()
        
        self.logger.info("FUSE decision analysis completed")
        return results
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get status of various analyses.
        
        This method provides a comprehensive status report on the analysis operations.
        It checks for the existence of key files and directories to determine which
        analyses have been completed successfully and which are available to run.
        
        The status information includes:
        - Whether benchmarking has been completed
        - Whether sensitivity analysis is available and its results exist
        - Whether decision analysis is available and its results exist
        - Whether optimization results (required for some analyses) exist
        
        This information is useful for tracking progress, diagnosing issues,
        and providing feedback to users.
        
        Returns:
            Dict[str, Any]: Dictionary containing analysis status information,
                          including flags for completed analyses and available results
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
        
        This method checks whether the necessary files and data are available
        to run each type of analysis. It verifies:
        
        1. For benchmarking: Existence of processed observation data
        2. For sensitivity analysis: Existence of optimization results
        3. For decision analysis: Existence of model simulation outputs
        
        These validations help prevent runtime errors by ensuring that analyses
        only run when their prerequisites are met.
        
        Returns:
            Dict[str, bool]: Dictionary indicating which analyses can be run:
                          - benchmarking: Whether benchmarking can be run
                          - sensitivity_analysis: Whether sensitivity analysis can be run
                          - decision_analysis: Whether decision analysis can be run
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