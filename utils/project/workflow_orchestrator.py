# In utils/workflow_utils/workflow_orchestrator.py

from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Callable
from datetime import datetime


class WorkflowOrchestrator:
    """
    Orchestrates the CONFLUENCE workflow execution and manages the step sequence.
    
    The WorkflowOrchestrator is responsible for defining, coordinating, and executing
    the complete CONFLUENCE modeling workflow. It integrates the various manager 
    components into a coherent sequence of operations, handling dependencies between
    steps, tracking progress, and providing status information.
    
    Key responsibilities:
    - Defining the sequence of workflow steps and their validation checks
    - Coordinating execution across different manager components
    - Handling execution flow (skipping completed steps, stopping on errors)
    - Providing status information and execution reports
    - Validating prerequisites before workflow execution
    
    This class represents the "conductor" of the CONFLUENCE system, ensuring that
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
        for the entire CONFLUENCE workflow.
        
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
        self.project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.domain_name}"
    
    def define_workflow_steps(self) -> List[Tuple[Callable, Callable, str]]:
        """
        Define the workflow steps with their output validation checks and descriptions.
        
        This method establishes the complete sequence of operations for a CONFLUENCE
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
                lambda: (self.project_dir / 'shapefiles_').exists(),
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
            
            # --- Model Agnostic Data Pre-Processing ---
            (
                self.managers['data'].process_observed_data,
                lambda: (self.project_dir / "observations" / "streamflow" / "preprocessed" / 
                        f"{self.domain_name}_streamflow_processed.csv").exists(),
                "Processing observational data"
            ),
            (
                self.managers['data'].acquire_forcings,
                lambda: (self.project_dir / "forcing" / "raw_data").exists() or (self.project_dir / "forcing" / "raw_data.tar.gz").exists(),
                "Acquiring meteorological forcing data"
            ),
            (
                self.managers['data'].run_model_agnostic_preprocessing,
                lambda: (self.project_dir / "forcing" / "basin_averaged_data").exists() or (self.project_dir / "forcing" / "basin_averaged_data.tar.gz").exists(),
                "Processing forcing data for modeling"
            ),
            
            # --- Model Specific Processing and Initialization ---
            (
                self.managers['model'].preprocess_models,
                lambda: (self.project_dir / "forcing" / 
                        f"{self.config['HYDROLOGICAL_MODEL'].split(',')[0]}_input").exists(),
                "Preparing model-specific input files"
            ),
            (
                self.managers['model'].run_models,
                lambda: (self.project_dir / "simulations" / self.experiment_id / 
                        f"{self.config.get('HYDROLOGICAL_MODEL').split(',')[0]}").exists(),
                "Running hydrological model simulations"
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
            
            # --- Result Analysis and Evaluation ---
            (
                self.managers['model'].postprocess_results,
                lambda: (self.project_dir / "results" / "postprocessed.csv").exists(),
                "Post-processing simulation results"
            ),
        ]
    
    def run_workflow(self) -> None:
        """
        Execute the complete CONFLUENCE workflow with enhanced logging.
        
        This method represents the main execution point of the CONFLUENCE system,
        running through all workflow steps in sequence. For each step, it:
        1. Checks if the step has already been completed (unless forced to rerun)
        2. Executes the step if necessary
        3. Tracks execution time and status
        4. Handles any errors according to configuration settings
        
        The method provides detailed logging throughout the execution process and
        generates a summary report upon completion.
        """
        # Log workflow header
        if self.logging_manager:
            header = self.logging_manager.format_section_header(
                f"CONFLUENCE WORKFLOW │ {self.domain_name} │ {self.experiment_id}"
            )
            self.logger.info(header)
        else:
            self.logger.info("=" * 60)
            self.logger.info(f"Starting CONFLUENCE workflow for domain: {self.domain_name}")
            self.logger.info(f"Experiment ID: {self.experiment_id}")
            self.logger.info("=" * 60)
        
        # Log start time and configuration
        start_time = datetime.now()
        self.logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get force run setting
        force_run = self.config.get('FORCE_RUN_ALL_STEPS', False)
        if force_run:
            self.logger.info("⚠ FORCE_RUN_ALL_STEPS enabled - all steps will execute")
        
        # Get workflow steps
        workflow_steps = self.define_workflow_steps()
        total_steps = len(workflow_steps)
        
        # Track execution status
        completed_steps = 0
        skipped_steps = 0
        failed_steps = 0
        
        # Execute workflow
        for idx, (step_func, check_func, description) in enumerate(workflow_steps, 1):
            step_name = step_func.__name__
            
            # Log step header
            if self.logging_manager:
                step_header = self.logging_manager.format_step_header(idx, total_steps, description)
                self.logger.info(step_header)
            else:
                self.logger.info(f"\n{'=' * 40}")
                self.logger.info(f"Step {idx}/{total_steps}: {step_name}")
                self.logger.info(f"{'=' * 40}")
            
            try:
                if force_run or not check_func():
                    step_start_time = datetime.now()
                    self.logger.info(f"Executing: {description}")
                    
                    step_func()
                    
                    step_end_time = datetime.now()
                    duration = step_end_time - step_start_time
                    
                    # Log completion
                    if self.logging_manager:
                        completion_msg = self.logging_manager.format_step_completion(
                            description, str(duration), "✓"
                        )
                        self.logger.info(completion_msg)
                    else:
                        self.logger.info(f"✓ Completed: {step_name} (Duration: {duration})")
                    
                    completed_steps += 1
                else:
                    # Log skip
                    if self.logging_manager:
                        skip_msg = self.logging_manager.format_step_completion(
                            description, "already exists", "→"
                        )
                        self.logger.info(skip_msg)
                    else:
                        self.logger.info(f"→ Skipping: {step_name} (Output already exists)")
                    
                    skipped_steps += 1
                    
            except Exception as e:
                # Log failure
                if self.logging_manager:
                    error_msg = self.logging_manager.format_step_completion(
                        description, f"ERROR: {str(e)}", "✗"
                    )
                    self.logger.error(error_msg)
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
        
        if self.logging_manager:
            summary_header = self.logging_manager.format_section_header("WORKFLOW SUMMARY")
            self.logger.info(summary_header)
        else:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("CONFLUENCE Workflow Summary")
            self.logger.info("=" * 60)
        
        self.logger.info(f"Total execution time: {total_duration}")
        self.logger.info(f"Steps completed: {completed_steps}/{total_steps}")
        self.logger.info(f"Steps skipped: {skipped_steps}")
        
        if failed_steps > 0:
            self.logger.warning(f"Steps failed: {failed_steps}")
            self.logger.warning("Workflow completed with errors")
        else:
            self.logger.info("✓ Workflow completed successfully")
        
        if self.logging_manager:
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