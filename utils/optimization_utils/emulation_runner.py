# In utils/emulation_utils/emulation_runner.py

from pathlib import Path
import logging
from typing import Dict, Optional

from utils.dataHandling_utils.attribute_processing_util import attributeProcessor
from utils.optimization_utils.single_sample_emulator import SingleSampleEmulator, RandomForestEmulator
from utils.optimization_utils.ensemble_performance_analyzer import EnsemblePerformanceAnalyzer


class EmulationRunner:
    """
    Manages the complete emulation workflow for hydrological models.
    
    This class orchestrates the entire emulation process including:
    1. Processing catchment attributes
    2. Preparing emulation data and running ensemble simulations
    3. Calculating performance metrics for ensemble runs
    4. Running random forest emulation for parameter optimization
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the emulation runner.
        
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
    
    def run_emulation_workflow(self) -> Optional[Dict]:
        """
        Execute the complete emulation workflow.
        
        Returns:
            Dict: Results from the random forest emulation or None if workflow failed
        """
        self.logger.info("Starting model emulation workflow")
        
        try:
            # Step 1: Process attributes
            self.process_attributes()
            
            # Step 2: Prepare emulation data and run ensemble
            emulation_output = self.prepare_emulation_data()
            
            # Step 3: Calculate ensemble performance metrics
            metrics_file = self.calculate_ensemble_performance_metrics()
            
            # Step 4: Run random forest emulation
            rf_results = self.run_random_forest_emulation()
            
            self.logger.info("Model emulation workflow completed successfully")
            return rf_results
            
        except Exception as e:
            self.logger.error(f"Error during emulation workflow: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def process_attributes(self):
        """
        Process catchment characteristic attributes.
        
        This method extracts and calculates various catchment attributes including
        soil, topography, climate, and land cover properties, saving them to a
        standardized CSV file.
        """
        if self.config.get('PROCESS_CHARACTERISTIC_ATTRIBUTES', False):
            self.logger.info("Processing catchment characteristics")
            
            try:
                # Create attribute processor instance
                ap = attributeProcessor(self.config, self.logger)
                ap.process_attributes()
                
                # Check if output file was created
                output_file = self.project_dir / "attributes" / f"{self.domain_name}_attributes.csv"
                if output_file.exists():
                    self.logger.info(f"Attribute processing completed successfully: {output_file}")
                else:
                    self.logger.warning("Attribute processing completed but no output file was found")
                    
            except Exception as e:
                self.logger.error(f"Error during attribute processing: {str(e)}")
                raise
        else:
            self.logger.info("Attribute processing skipped (PROCESS_CHARACTERISTIC_ATTRIBUTES=False)")
    
    def prepare_emulation_data(self) -> Optional[Path]:
        """
        Run large sample emulation to generate spatially varying trial parameters.
        Creates parameter sets for ensemble modeling, runs simulations, and analyzes results.
        
        Returns:
            Path: Path to emulation output file or None if failed
        """
        self.logger.info("Starting large sample emulation")
        
        # Check if emulation is enabled in config
        if not self.config.get('RUN_LARGE_SAMPLE_EMULATION', False):
            self.logger.info("Large sample emulation disabled in config. Skipping.")
            return None
        
        try:
            # Initialize the emulator
            emulator = SingleSampleEmulator(self.config, self.logger)
            
            # Run the emulation setup
            emulator_output = emulator.run_emulation_setup()
            
            if not emulator_output:
                self.logger.warning("Large sample emulation completed but no parameter file was generated.")
                return None
                
            self.logger.info(f"Large sample emulation completed successfully. Output: {emulator_output}")
            
            # Analyze the parameter space
            if self.config.get('EMULATION_ANALYZE_PARAMETERS', True):
                self.logger.info("Analyzing parameter space")
                param_analysis_dir = emulator.analyze_parameter_space(emulator_output)
                if param_analysis_dir:
                    self.logger.info(f"Parameter space analysis completed. Results saved to: {param_analysis_dir}")
            
            # Check if we should run the ensemble simulations
            run_ensemble = self.config.get('EMULATION_RUN_ENSEMBLE', False)
            
            if run_ensemble:
                self.logger.info("Starting ensemble simulations")
                ensemble_results = emulator.run_ensemble_simulations()
                success_count = sum(1 for _, success, _ in ensemble_results if success)
                self.logger.info(f"Ensemble simulations completed with {success_count} successes out of {len(ensemble_results)}")
                
                # Analyze ensemble results if we had successful runs
                if success_count > 0 and self.config.get('EMULATION_ANALYZE_ENSEMBLE', True):
                    self.logger.info("Starting ensemble results analysis")
                    analysis_dir = emulator.analyze_ensemble_results()
                    if analysis_dir:
                        self.logger.info(f"Ensemble analysis completed. Results saved to: {analysis_dir}")
                    else:
                        self.logger.warning("Ensemble analysis did not produce results.")
            
            return emulator_output
                
        except ImportError as e:
            self.logger.error(f"Could not import SingleSampleEmulator: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during large sample emulation: {str(e)}")
            raise
    
    def calculate_ensemble_performance_metrics(self) -> Optional[Path]:
        """
        Calculate performance metrics for ensemble simulations and create visualizations.
        
        Returns:
            Path: Path to performance metrics file or None if calculation failed
        """
        self.logger.info("Calculating ensemble performance metrics")
        
        try:
            analyzer = EnsemblePerformanceAnalyzer(self.config, self.logger)
            metrics_file = analyzer.calculate_ensemble_performance_metrics()
            
            if metrics_file:
                self.logger.info(f"Ensemble performance metrics calculated successfully: {metrics_file}")
            else:
                self.logger.warning("Ensemble performance metrics calculation did not produce output")
                
            return metrics_file
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble performance metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def run_random_forest_emulation(self) -> Optional[Dict]:
        """
        Run random forest emulation to find optimal parameters based on
        geospatial attributes and performance metrics.
        
        Returns:
            Dict: Random forest emulation results or None if emulation failed
        """
        self.logger.info("Starting random forest emulation")
        
        # Check if emulation is enabled in config
        if not self.config.get('RUN_RANDOM_FOREST_EMULATION', False):
            self.logger.info("Random forest emulation disabled in config. Skipping.")
            return None
        
        try:
            # Create the emulator
            emulator = RandomForestEmulator(self.config, self.logger)
            
            # Run the emulation workflow
            results = emulator.run_workflow()
            
            if results:
                self.logger.info(f"Random forest emulation completed successfully.")
                self.logger.info(f"Optimized parameters saved to: {results['output_dir']}")
                
                # Display key metrics
                self.logger.info(f"Model R² score: {results['model_metrics']['test_score']:.4f}")
                self.logger.info(f"Predicted {emulator.target_metric}: {results['predicted_score']:.4f}")
                
                return results
            else:
                self.logger.warning("Random forest emulation did not produce results.")
                return None
                
        except ImportError as e:
            self.logger.error(f"Could not import RandomForestEmulator: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error during random forest emulation: {str(e)}")
            raise