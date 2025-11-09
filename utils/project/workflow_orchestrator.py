# In utils/project/workflow_orchestrator.py

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Callable
from datetime import datetime


class WorkflowOrchestrator:
    """
    Orchestrates the SYMFLUENCE workflow execution and manages the step sequence.
    
    The WorkflowOrchestrator is responsible for defining, coordinating, and executing
    the complete SYMFLUENCE modeling workflow. It integrates the various manager 
    components into a coherent sequence of operations, handling dependencies between
    steps, tracking progress, and providing status information.
    
    Key responsibilities:
    - Defining the sequence of workflow steps and their validation checks
    - Coordinating execution across different manager components
    - Handling execution flow (skipping completed steps, stopping on errors)
    - Providing status information and execution reports
    - Validating prerequisites before workflow execution
    
    This class represents the "conductor" of the SYMFLUENCE system, ensuring that
    each component performs its tasks in the correct order and with the necessary
    inputs from previous steps.
    
    Attributes:
        managers (Dict[str, Any]): Dictionary of manager instances
        config (Dict[str, Any]): Configuration dictionary
        logger (logging.Logger): Logger instance
        domain_name (str): Name of the hydrological domain
        experiment_id (str): ID of the current experiment
        project_dir (Path): Path to the project directory
        logging_manager: Reference to logging manager for enhanced formatting
    """
    
    def __init__(self, managers: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger, logging_manager=None):
        """
        Initialize the workflow orchestrator.
        
        Sets up the orchestrator with references to all manager components, the 
        configuration, and the logger. This creates the central coordination point
        for the entire SYMFLUENCE workflow.
        
        Args:
            managers (Dict[str, Any]): Dictionary of manager instances for each 
                                      functional area (project, domain, data, etc.)
            config (Dict[str, Any]): Configuration dictionary with all settings
            logger (logging.Logger): Logger instance for recording operations
            logging_manager: Reference to LoggingManager for enhanced formatting
            
        Raises:
            KeyError: If essential configuration values are missing
        """
        self.managers = managers
        self.config = config
        self.logger = logger
        self.logging_manager = logging_manager
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Support both old (CONFLUENCE) and new (SYMFLUENCE) config keys
        data_dir = config.get('SYMFLUENCE_DATA_DIR') or config.get('CONFLUENCE_DATA_DIR')
        if not data_dir:
            raise KeyError("Neither SYMFLUENCE_DATA_DIR nor CONFLUENCE_DATA_DIR found in config")
        
        self.project_dir = Path(data_dir) / f"domain_{self.domain_name}"
    
    def define_workflow_steps(self) -> List[Tuple[Callable, Callable, str]]:
        """
        Define the workflow steps with their output validation checks and descriptions.
        
        This method establishes the complete sequence of operations for a SYMFLUENCE
        modeling workflow. Each step consists of:
        1. A function to execute (typically a method from a manager component)
        2. A check function that verifies the step's output exists
        3. A human-readable description of what the step does
        
        The workflow follows a logical progression through five main phases:
        1. Project initialization
        2. Geospatial domain definition and analysis
        3. Model-agnostic data preprocessing
        4. Model-specific processing and execution
        5. Optimization, emulation, and analysis
        
        Returns:
            List[Tuple[Callable, Callable, str]]: List of (function, check_function, description) tuples
        """

        # Get configured analyses
        analyses = self.config.get('ANALYSES', [])
        optimisations = self.config.get('OPTIMISATION_METHODS', []) 

        return [
            # --- Project Initialization ---
            (
                self.managers['project'].setup_project,
                lambda: (self.project_dir / 'shapefiles').exists(),
                "Setting up project structure and directories"
            ),
            
            # --- Geospatial Domain Definition and Analysis ---
            (
                self.managers['project'].create_pour_point,
                lambda: (self.project_dir / "shapefiles" / "pour_point" / 
                        f"{self.domain_name}_pourPoint.shp").exists(),
                "Creating watershed pour point"
            ),
            (
                self.managers['data'].acquire_attributes,
                lambda: (self.project_dir / "attributes" / "soilclass" / 
                        f"domain_{self.domain_name}_soil_classes.tif").exists(),
                "Acquiring geospatial attributes and data"
            ),
            (
                self.managers['domain'].define_domain,
                lambda: (self.project_dir / "shapefiles" / "river_basins" / 
                        f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp").exists(),
                "Defining hydrological domain boundaries"
            ),
            (
                self.managers['domain'].discretize_domain,
                lambda: (self.project_dir / "shapefiles" / "catchment" / 
                        f"{self.domain_name}_HRUs_{str(self.config['DOMAIN_DISCRETIZATION']).replace(',','_')}.shp").exists(),
                "Discretizing domain into hydrological response units"
            ),
            
            # --- Model-Agnostic Data Preprocessing ---
            (
                self.managers['data'].process_observed_data,
                lambda: (self.project_dir / "observations" / "streamflow" / 
                        f"{self.domain_name}_streamflow_processed.csv").exists(),
                "Processing observed streamflow data"
            ),
            (
                self.managers['data'].acquire_forcings,
                lambda: (self.project_dir / "forcing" / 
                        f"{self.domain_name}_forcing.nc").exists(),
                "Acquiring meteorological forcing data"
            ),
            (
                self.managers['data'].run_model_agnostic_preprocessing,
                lambda: (self.project_dir / "domain_data" / 
                        f"{self.domain_name}_attributes.nc").exists(),
                "Running model-agnostic data preprocessing"
            ),
            
            # --- Model-Specific Preprocessing and Execution ---
            (
                self.managers['model'].preprocess_models,
                lambda: any((self.project_dir / "settings").glob(f"*_{self.config.get('HYDROLOGICAL_MODEL', 'SUMMA')}*")),
                "Preprocessing model-specific input files"
            ),
            (
                self.managers['model'].run_models,
                lambda: (self.project_dir / "simulations" / 
                        f"{self.experiment_id}_{self.config.get('HYDROLOGICAL_MODEL', 'SUMMA')}_output.nc").exists(),
                "Running hydrological model simulation"
            ),
            (
                self.managers['model'].postprocess_results,
                lambda: (self.project_dir / "simulations" / 
                        f"{self.experiment_id}_postprocessed.nc").exists(),
                "Post-processing simulation results"
            ),
            
            # --- Optimization and Emulation Steps ---
            (
                self.managers['optimization'].calibrate_model,
                lambda: ('optimization' in optimisations and 
                        (self.project_dir / "optimisation" / 
                        f"{self.experiment_id}_parallel_iteration_results.csv").exists()),
                "Calibrating model parameters"
            ),
            
            (
                self.managers['optimization'].run_emulation,
                lambda: ('emulation' in optimisations and
                        (self.project_dir / "emulation" / self.experiment_id / 
                        "rf_emulation" / "optimized_parameters.csv").exists()),
                "Running parameter emulation analysis"
            ),
            
            # --- Analysis Steps ---
            (
                self.managers['analysis'].run_benchmarking,
                lambda: ('benchmarking' in analyses and
                        (self.project_dir / "evaluation" / "benchmark_scores.csv").exists()),
                "Running model benchmarking analysis"
            ),
            
            (
                self.managers['analysis'].run_decision_analysis,
                lambda: ('decision' in analyses and
                        (self.project_dir / "optimisation" / 
                        f"{self.experiment_id}_model_decisions_comparison.csv").exists()),
                "Analyzing modeling decisions impact"
            ),
            
            (
                self.managers['analysis'].run_sensitivity_analysis,
                lambda: ('sensitivity' in analyses and
                        (self.project_dir / "plots" / "sensitivity_analysis" / 
                        "all_sensitivity_results.csv").exists()),
                "Running parameter sensitivity analysis"
            ),
            
        ]
    
    def run_workflow(self, force_run: bool = False):
        """
        Run the complete workflow according to the defined steps.
        
        This method executes each step in the workflow sequence, handling:
        - Conditional execution based on existing outputs
        - Error handling with configurable stop-on-error behavior
        - Progress tracking and timing information
        - Comprehensive logging of each operation
        
        The workflow can be configured to:
        - Skip steps that have already been completed (default)
        - Force re-execution of all steps (force_run=True)
        - Continue or stop on errors (based on STOP_ON_ERROR config)
        
        Args:
            force_run (bool): If True, forces execution of all steps even if outputs exist.
                            If False (default), skips steps with existing outputs.
        
        Raises:
            Exception: If a step fails and STOP_ON_ERROR is True in configuration
            
        Note:
            The method provides detailed logging throughout execution, including:
            - Step headers with progress indicators
            - Execution timing for each step
            - Clear success/skip/failure indicators
            - Final summary statistics
        """
        # Check prerequisites
        if not self.validate_workflow_prerequisites():
            raise ValueError("Workflow prerequisites not met")
        
        # Log workflow start
        start_time = datetime.now()
        
        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("=" * 60)
        self.logger.info("SYMFLUENCE WORKFLOW EXECUTION")
        self.logger.info(f"Domain: {self.domain_name}")
        self.logger.info(f"Experiment: {self.experiment_id}")
        self.logger.info("=" * 60)
        
        # Get workflow steps
        workflow_steps = self.define_workflow_steps()
        total_steps = len(workflow_steps)
        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0
        
        # Execute each step
        for idx, (step_func, check_func, description) in enumerate(workflow_steps, 1):
            step_name = step_func.__name__
            
            # FIXED: Use log_step_header() instead of non-existent format_step_header()
            if self.logging_manager:
                self.logging_manager.log_step_header(idx, total_steps, step_name, description)
            else:
                self.logger.info(f"\nStep {idx}/{total_steps}: {step_name}")
                self.logger.info(f"{description}")
                self.logger.info("=" * 40)
            
            try:
                if force_run or not check_func():
                    step_start_time = datetime.now()
                    self.logger.info(f"Executing: {description}")
                    
                    step_func()
                    
                    step_end_time = datetime.now()
                    duration = (step_end_time - step_start_time).total_seconds()
                    
                    # FIXED: Use log_completion() instead of non-existent format_step_completion()
                    if self.logging_manager:
                        self.logging_manager.log_completion(
                            success=True, 
                            message=description, 
                            duration=duration
                        )
                    else:
                        self.logger.info(f"✓ Completed: {step_name} (Duration: {duration:.2f}s)")
                    
                    completed_steps += 1
                else:
                    # Log skip
                    if self.logging_manager:
                        self.logging_manager.log_substep(f"Skipping: {description} (Output already exists)")
                    else:
                        self.logger.info(f"→ Skipping: {step_name} (Output already exists)")
                    
                    skipped_steps += 1
                    
            except Exception as e:
                # Log failure
                if self.logging_manager:
                    self.logging_manager.log_completion(
                        success=False,
                        message=f"{description}: {str(e)}"
                    )
                else:
                    self.logger.error(f"✗ Failed: {step_name}")
                    self.logger.error(f"Error: {str(e)}")
                
                failed_steps += 1
                
                # Decide whether to continue or stop
                if self.config.get('STOP_ON_ERROR', True):
                    self.logger.error("Workflow stopped due to error (STOP_ON_ERROR=True)")
                    raise
                else:
                    self.logger.warning("Continuing despite error (STOP_ON_ERROR=False)")
        
        # Summary report
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        # FIXED: Use direct logging instead of non-existent format_section_header()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 60)
        
        self.logger.info(f"Total execution time: {total_duration}")
        self.logger.info(f"Steps completed: {completed_steps}/{total_steps}")
        self.logger.info(f"Steps skipped: {skipped_steps}")
        
        if failed_steps > 0:
            self.logger.warning(f"Steps failed: {failed_steps}")
            self.logger.warning("Workflow completed with errors")
        else:
            self.logger.info("✓ Workflow completed successfully")
        
        self.logger.info("═" * 60)
    
    def validate_workflow_prerequisites(self) -> bool:
        """
        Validate that all prerequisites are met before running the workflow.
        
        This method performs a series of checks to ensure that the workflow can be
        executed successfully:
        1. Verifies that all required configuration parameters are present
        2. Confirms that all manager components have been properly initialized
        
        These validations help prevent runtime errors by catching configuration or
        initialization issues before workflow execution begins.
        
        Returns:
            bool: True if all prerequisites are met, False otherwise
            
        Note:
            This method logs detailed information about any validation failures,
            making it useful for diagnosing configuration problems.
        """
        valid = True
        
        # Check configuration validity (support both old and new config keys)
        required_config = [
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'HYDROLOGICAL_MODEL',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION'
        ]
        
        for key in required_config:
            if not self.config.get(key):
                self.logger.error(f"Required configuration missing: {key}")
                valid = False
        
        # Check for data directory (either old or new name)
        if not (self.config.get('SYMFLUENCE_DATA_DIR') or self.config.get('CONFLUENCE_DATA_DIR')):
            self.logger.error("Required configuration missing: SYMFLUENCE_DATA_DIR (or CONFLUENCE_DATA_DIR)")
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
        Get the current status of the workflow execution.
        
        This method examines each step in the workflow to determine whether it has
        been completed, using the same output validation checks used during execution.
        It provides a comprehensive view of workflow progress, including which steps
        are complete and which are pending.
        
        The status information is useful for:
        - Monitoring long-running workflows
        - Generating progress reports
        - Diagnosing execution issues
        - Providing feedback to users
        
        Returns:
            Dict[str, Any]: Dictionary containing workflow status information, including:
                - total_steps: Total number of workflow steps
                - completed_steps: Number of completed steps
                - pending_steps: Number of pending steps
                - step_details: List of dictionaries with details for each step
                  (name and completion status)
        """
        workflow_steps = self.define_workflow_steps()
        
        status = {
            'total_steps': len(workflow_steps),
            'completed_steps': 0,
            'pending_steps': 0,
            'step_details': []
        }
        
        for step_func, check_func, description in workflow_steps:
            step_name = step_func.__name__
            is_complete = check_func()
            
            if is_complete:
                status['completed_steps'] += 1
            else:
                status['pending_steps'] += 1
            
            status['step_details'].append({
                'name': step_name,
                'description': description,
                'complete': is_complete
            })
        
        return status