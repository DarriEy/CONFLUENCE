# In utils/workflow_utils/workflow_orchestrator.py

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Callable
from datetime import datetime


class WorkflowOrchestrator:
    """Orchestrates the CONFLUENCE workflow execution."""
    
    def __init__(self, managers: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the workflow orchestrator.
        
        Args:
            managers: Dictionary of manager instances
            config: Configuration dictionary
            logger: Logger instance
        """
        self.managers = managers
        self.config = config
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}"
    
    def define_workflow_steps(self) -> List[Tuple[Callable, Callable]]:
        """
        Define the workflow steps and their output checks.
        
        Returns:
            List of tuples containing (function, check_function) pairs
        """
        return [
            # --- Project Initialization ---
            (
                self.managers['project'].setup_project,
                lambda: self.project_dir.exists()
            ),
            
            # --- Geospatial Domain Definition and Analysis ---
            (
                self.managers['project'].create_pour_point,
                lambda: (self.project_dir / "shapefiles" / "pour_point" / 
                        f"{self.domain_name}_pourPoint.shp").exists()
            ),
            (
                self.managers['data'].acquire_attributes,
                lambda: (self.project_dir / "attributes" / "soilclass" / 
                        f"domain_{self.domain_name}_soil_classes.tif").exists()
            ),
            (
                self.managers['domain'].define_domain,
                lambda: (self.project_dir / "shapefiles" / "river_basins" / 
                        f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists()
            ),
            (
                self.managers['domain'].discretize_domain,
                lambda: (self.project_dir / "shapefiles" / "catchment" / 
                        f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp").exists()
            ),
            
            # --- Model Agnostic Data Pre-Processing ---
            (
                self.managers['data'].process_observed_data,
                lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                        f"{self.domain_name}_streamflow_processed.csv").exists()
            ),
            (
                self.managers['data'].acquire_forcings,
                lambda: (self.project_dir / "forcing" / "raw_data").exists()
            ),
            (
                self.managers['data'].run_model_agnostic_preprocessing,
                lambda: (self.project_dir / "forcing" / "basin_averaged_data").exists()
            ),
            (
                self.managers['analysis'].run_benchmarking,
                lambda: (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()
            ),
            
            # --- Model Specific Processing and Initialization ---
            (
                self.managers['model'].preprocess_models,
                lambda: (self.project_dir / "forcing" / 
                        f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input").exists()
            ),
            (
                self.managers['model'].run_models,
                lambda: (self.project_dir / "simulations" / self.experiment_id / 
                        f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists()
            ),
            
            # --- Optimization and Emulation Steps ---
            (
                self.managers['optimization'].calibrate_model,
                lambda: (self.config.get('RUN_ITERATIVE_OPTIMISATION', False) and 
                        (self.project_dir / "optimisation" / 
                         f"{self.experiment_id}_parallel_iteration_results.csv").exists())
            ),
            (
                self.managers['optimization'].run_emulation,
                lambda: ((self.config.get('RUN_LARGE_SAMPLE_EMULATION', False) or 
                         self.config.get('RUN_RANDOM_FOREST_EMULATION', False)) and
                        (self.project_dir / "emulation" / self.experiment_id / 
                         "rf_emulation" / "optimized_parameters.csv").exists())
            ),
            
            # --- Analysis Steps ---
            (
                self.managers['analysis'].run_decision_analysis,
                lambda: (self.config.get('RUN_DECISION_ANALYSIS', False) and
                        (self.project_dir / "optimisation" / 
                         f"{self.experiment_id}_model_decisions_comparison.csv").exists())
            ),
            (
                self.managers['analysis'].run_sensitivity_analysis,
                lambda: (self.config.get('RUN_SENSITIVITY_ANALYSIS', False) and
                        (self.project_dir / "plots" / "sensitivity_analysis" / 
                         "all_sensitivity_results.csv").exists())
            ),
            
            # --- Result Analysis and Evaluation ---
            (
                self.managers['model'].postprocess_results,
                lambda: (self.project_dir / "results" / "postprocessed.csv").exists()
            ),
        ]
    
    def run_workflow(self):
        """Execute the CONFLUENCE workflow."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting CONFLUENCE workflow for domain: {self.domain_name}")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
        
        # Get force run setting
        force_run = self.config.get('FORCE_RUN_ALL_STEPS', False)
        if force_run:
            self.logger.warning("FORCE_RUN_ALL_STEPS is enabled - all steps will be executed")
        
        # Get workflow steps
        workflow_steps = self.define_workflow_steps()
        total_steps = len(workflow_steps)
        
        # Track execution status
        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0
        
        # Execute workflow
        for idx, (step_func, check_func) in enumerate(workflow_steps, 1):
            step_name = step_func.__name__
            self.logger.info(f"\n{'=' * 40}")
            self.logger.info(f"Step {idx}/{total_steps}: {step_name}")
            self.logger.info(f"{'=' * 40}")
            
            try:
                if force_run or not check_func():
                    self.logger.info(f"Executing: {step_name}")
                    start_time = datetime.now()
                    
                    step_func()
                    
                    end_time = datetime.now()
                    duration = end_time - start_time
                    
                    self.logger.info(f"✓ Completed: {step_name} (Duration: {duration})")
                    completed_steps += 1
                else:
                    self.logger.info(f"→ Skipping: {step_name} (Output already exists)")
                    skipped_steps += 1
                    
            except Exception as e:
                self.logger.error(f"✗ Failed: {step_name}")
                self.logger.error(f"Error: {str(e)}")
                failed_steps += 1
                
                # Decide whether to continue or stop
                if self.config.get('STOP_ON_ERROR', True):
                    self.logger.error("Workflow stopped due to error (STOP_ON_ERROR=True)")
                    raise
                else:
                    self.logger.warning("Continuing workflow despite error (STOP_ON_ERROR=False)")
        
        # Summary report
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CONFLUENCE Workflow Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Completed: {completed_steps}")
        self.logger.info(f"Skipped: {skipped_steps}")
        self.logger.info(f"Failed: {failed_steps}")
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if failed_steps > 0:
            self.logger.warning(f"Workflow completed with {failed_steps} failures")
        else:
            self.logger.info("Workflow completed successfully")
        self.logger.info("=" * 60)
    
    def validate_workflow_prerequisites(self) -> bool:
        """
        Validate that all prerequisites are met before running workflow.
        
        Returns:
            True if all prerequisites are met, False otherwise
        """
        valid = True
        
        # Check configuration validity
        required_config = [
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'CONFLUENCE_DATA_DIR',
            'HYDROLOGICAL_MODEL',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION'
        ]
        
        for key in required_config:
            if not self.config.get(key):
                self.logger.error(f"Required configuration missing: {key}")
                valid = False
        
        # Check manager initialization
        required_managers = ['project', 'domain', 'data', 'model', 'analysis', 'optimization']
        for manager_name in required_managers:
            if manager_name not in self.managers:
                self.logger.error(f"Required manager not initialized: {manager_name}")
                valid = False
        
        return valid
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get the current status of the workflow.
        
        Returns:
            Dictionary containing workflow status information
        """
        workflow_steps = self.define_workflow_steps()
        
        status = {
            'total_steps': len(workflow_steps),
            'completed_steps': 0,
            'pending_steps': 0,
            'step_details': []
        }
        
        for step_func, check_func in workflow_steps:
            step_name = step_func.__name__
            is_complete = check_func()
            
            if is_complete:
                status['completed_steps'] += 1
            else:
                status['pending_steps'] += 1
            
            status['step_details'].append({
                'name': step_name,
                'complete': is_complete
            })
        
        return status