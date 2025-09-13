#!/usr/bin/env python3
"""
CONFLUENCE - Community Optimization Nexus for Leveraging
Understanding of Environmental Networks in Computational Exploration

Enhanced main entry point for the CONFLUENCE hydrological modeling platform.
CONFLUENCE provides an integrated framework for:
- Hydrological model setup and configuration
- Multi-model comparison and evaluation  
- Parameter optimization and calibration
- Workflow management
- Individual workflow step execution
- Pour point-based domain setup

Usage:
    # Run complete workflow
    python CONFLUENCE.py [--config CONFIG_PATH]
    
    # Run individual workflow steps
    python CONFLUENCE.py --calibrate_model --run_benchmarking
    
    # Set up for a specific pour point
    python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate
    
    # Check status
    python CONFLUENCE.py --status

For more information, see the documentation at:
    https://github.com/DarriEy/CONFLUENCE

License: MIT License
Version: 1.0.0
"""

from pathlib import Path
import sys
import yaml
from datetime import datetime
from typing import Dict, Any, List
import warnings

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import CONFLUENCE components
from utils.project.project_manager import ProjectManager
from utils.project.workflow_orchestrator import WorkflowOrchestrator
from utils.project.logging_manager import LoggingManager
from utils.data.data_manager import DataManager
from utils.geospatial.domain_manager import DomainManager
from utils.models.model_manager import ModelManager
from utils.evaluation.analysis_manager import AnalysisManager
from utils.optimization.optimization_manager import OptimizationManager

# Import the new CLI argument manager
from utils.cli.cli_argument_manager import CLIArgumentManager

# Suppress pyogrio field width warnings (non-fatal - data is still written)
warnings.filterwarnings('ignore', 
                       message='.*not successfully written.*field width.*',
                       category=RuntimeWarning,
                       module='pyogrio.raw')

class CONFLUENCE:
    """
    Enhanced CONFLUENCE main class with comprehensive CLI support.
    
    This class serves as the central coordinator for all CONFLUENCE operations,
    now with enhanced CLI capabilities including individual step execution,
    pour point setup, and comprehensive workflow management.
    """
    
    def __init__(self, config_path: Path, config_overrides: Dict[str, Any] = None, debug_mode: bool = False):
        """
        Initialize the CONFLUENCE system with configuration and CLI options.
        
        Args:
            config_path: Path to the configuration file
            config_overrides: Dictionary of configuration overrides from CLI
            debug_mode: Whether to enable debug mode
        """
        self.config_path = config_path
        self.debug_mode = debug_mode
        self.config_overrides = config_overrides or {}
        
        # Load and merge configuration
        self.config = self._load_and_merge_config()
        
        # Initialize logging
        self.logging_manager = LoggingManager(self.config, debug_mode=debug_mode)
        self.logger = self.logging_manager.logger
        
        # Initialize managers
        self.managers = self._initialize_managers()
        
        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            self.managers, self.config, self.logger
        )
        
        self.logger.info(f"CONFLUENCE initialized with config: {config_path}")
        if self.config_overrides:
            self.logger.info(f"Configuration overrides applied: {list(self.config_overrides.keys())}")
    
    def _load_and_merge_config(self) -> Dict[str, Any]:
        """Load configuration file and apply CLI overrides."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        if self.config_overrides:
            config.update(self.config_overrides)
        
        return config
    
    def _initialize_managers(self) -> Dict[str, Any]:
        """Initialize all manager components."""
        try:
            return {
                'project': ProjectManager(self.config, self.logger),
                'domain': DomainManager(self.config, self.logger),
                'data': DataManager(self.config, self.logger),
                'model': ModelManager(self.config, self.logger),
                'analysis': AnalysisManager(self.config, self.logger),
                'optimization': OptimizationManager(self.config, self.logger),
            }
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize managers: {str(e)}")
            raise RuntimeError(f"Manager initialization failed: {str(e)}")
    
    def run_workflow(self) -> None:
        """Execute the complete CONFLUENCE workflow."""
        try:
            start_time = datetime.now()
            self.logger.info("Starting complete CONFLUENCE workflow execution")
            
            # Execute the full workflow
            self.workflow_orchestrator.run_workflow()
            
            # Create run summary
            end_time = datetime.now()
            status = self.workflow_orchestrator.get_workflow_status()
            self.logging_manager.create_run_summary(start_time, end_time, status)
            
            self.logger.info("Complete CONFLUENCE workflow execution completed")
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise RuntimeError(f"Workflow execution error: {str(e)}")
        
    def run_individual_steps(self, step_names: List[str]) -> None:
        """
        Execute specific workflow steps.
        
        Args:
            step_names: List of step names to execute
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting individual step execution: {', '.join(step_names)}")
            
            # Get the complete workflow definition
            workflow_steps = self.workflow_orchestrator.define_workflow_steps()
            
            # Create a mapping from function names to workflow steps
            step_name_to_function = {}
            
            for step_item in workflow_steps:
                try:
                    # Handle variable tuple lengths safely
                    if isinstance(step_item, (tuple, list)) and len(step_item) >= 2:
                        step_func = step_item[0]
                        check_func = step_item[1] 
                        # Ignore any additional elements beyond the first 2
                        
                        if hasattr(step_func, '__name__'):
                            function_name = step_func.__name__
                            step_name_to_function[function_name] = (step_func, check_func)
                            self.logger.debug(f"Mapped workflow function: {function_name}")
                        else:
                            self.logger.warning(f"Step function does not have __name__ attribute: {step_func}")
                    else:
                        self.logger.warning(f"Unexpected step format: {step_item} (type: {type(step_item)}, length: {len(step_item) if hasattr(step_item, '__len__') else 'unknown'})")
                        
                except Exception as e:
                    self.logger.error(f"Error processing workflow step {step_item}: {str(e)}")
                    continue
            
            # Create mapping from CLI step names to actual function names
            cli_to_function_map = {
                'setup_project': 'setup_project',
                'create_pour_point': 'create_pour_point',
                'acquire_attributes': 'acquire_attributes',
                'define_domain': 'define_domain',
                'discretize_domain': 'discretize_domain',
                'setup_model': 'preprocess_models',  # Maps to actual function name
                'run_model': 'run_models',           # Fixed: maps to plural function name
                'calibrate_model': 'calibrate_model', # Direct mapping
                'run_emulation': 'run_emulation',
                'run_benchmarking': 'run_benchmarking',
                'run_decision_analysis': 'run_decision_analysis',
                'run_sensitivity_analysis': 'run_sensitivity_analysis',
                'postprocess_results': 'postprocess_results',
                # Additional mappings for data processing steps
                'process_observed_data': 'process_observed_data',
                'acquire_forcings': 'acquire_forcings',
                'run_model_agnostic_preprocessing': 'run_model_agnostic_preprocessing',
            }
            
            self.logger.info(f"Available workflow functions: {list(step_name_to_function.keys())}")
            
            # Execute requested steps
            for step_name in step_names:
                if step_name in cli_to_function_map:
                    function_name = cli_to_function_map[step_name]
                    
                    if function_name in step_name_to_function:
                        step_function, check_function = step_name_to_function[function_name]
                        
                        self.logger.info(f"Executing step: {step_name} -> {function_name}")
                        
                        # FORCE RUN: Skip completion checks for individual steps
                        # When someone explicitly requests a step, they want it to run
                        self.logger.info(f"Individual step mode: forcing execution of {step_name} (skipping completion check)")
                        
                        # Execute the step
                        try:
                            step_function()
                            self.logger.info(f"Step {step_name} completed successfully")
                        except Exception as e:
                            self.logger.error(f"Step {step_name} failed: {str(e)}")
                            if self.config.get('STOP_ON_ERROR', True):
                                raise
                    else:
                        self.logger.warning(f"Function {function_name} not found in workflow for step {step_name}")
                        self.logger.info(f"Available functions: {list(step_name_to_function.keys())}")
                        
                        # Try to find close matches
                        possible_matches = [name for name in step_name_to_function.keys() 
                                        if step_name.replace('_', '') in name.replace('_', '') or
                                            name.replace('_', '') in step_name.replace('_', '')]
                        if possible_matches:
                            self.logger.info(f"Possible matches: {possible_matches}")
                else:
                    self.logger.warning(f"Unknown step: {step_name}")
                    self.logger.info(f"Available steps: {', '.join(cli_to_function_map.keys())}")
            
            end_time = datetime.now()
            self.logger.info(f"Individual step execution completed in {end_time - start_time}")
            
        except Exception as e:
            self.logger.error(f"Individual step execution failed: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Step execution error: {str(e)}")
    
    def setup_pour_point_workflow(self, coordinates: str, domain_def_method: str) -> None:
        """
        Set up and run a workflow for a specific pour point.
        
        Args:
            coordinates: Pour point coordinates in "lat/lon" format
            domain_def_method: Domain definition method to use
        """
        try:
            self.logger.info(f"Setting up pour point workflow for {coordinates}")
            self.logger.info(f"Using domain definition method: {domain_def_method}")
            
            # The configuration has already been updated with pour point coordinates
            # Run the key steps for pour point setup
            essential_steps = [
                'setup_project',
                'create_pour_point',
                'define_domain',
                'discretize_domain'
            ]
            
            self.logger.info("Running essential steps for pour point setup")
            self.run_individual_steps(essential_steps)
            
            self.logger.info("Pour point workflow setup completed")
            self.logger.info("You can now run additional steps like --acquire_attributes, --setup_model, etc.")
            
        except Exception as e:
            self.logger.error(f"Pour point workflow setup failed: {str(e)}")
            raise RuntimeError(f"Pour point setup error: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the CONFLUENCE system."""
        return {
            'config_valid': bool(self.config),
            'managers_initialized': all(self.managers.values()),
            'workflow_status': self.workflow_orchestrator.get_workflow_status(),
            'domain': self.config.get('DOMAIN_NAME'),
            'experiment': self.config.get('EXPERIMENT_ID'),
            'config_path': str(self.config_path),
            'debug_mode': self.debug_mode
        }


# Fix for the execute_confluence_plan function in CONFLUENCE.py
# Replace the existing execute_confluence_plan function with this corrected version:

def execute_confluence_plan(plan: Dict[str, Any], config_path: Path) -> None:
    """
    Execute CONFLUENCE based on the CLI execution plan.
    
    Args:
        plan: Execution plan from CLI argument manager
        config_path: Path to configuration file
    """
    mode = plan['mode']
    config_overrides = plan.get('config_overrides', {})
    settings = plan.get('settings', {})
    
    cli_manager = CLIArgumentManager()
    
    # Handle binary management operations FIRST
    if mode == 'binary_management':
        binary_ops = plan.get('binary_operations', {})
        
        # For binary operations, we may or may not need a CONFLUENCE instance
        confluence_instance = None
        try:
            confluence_instance = CONFLUENCE(
                config_path=config_path,
                config_overrides=config_overrides,
                debug_mode=settings.get('debug', False)
            )
        except Exception as e:
            print(f"⚠️  Could not initialize CONFLUENCE instance: {str(e)}")
            print("   Proceeding with binary operations without CONFLUENCE instance...")
        
        if binary_ops.get('validate_binaries', False):
            cli_manager.validate_binaries(confluence_instance)
        
        if binary_ops.get('get_executables') is not None:
            specific_tools = binary_ops.get('get_executables')
            force_install = binary_ops.get('force_install', False)
            dry_run = settings.get('dry_run', False)
            
            # If empty list, install all tools
            if isinstance(specific_tools, list) and len(specific_tools) == 0:
                specific_tools = None
            
            cli_manager.get_executables(
                specific_tools=specific_tools,
                confluence_instance=confluence_instance,
                force=force_install,
                dry_run=dry_run
            )
        
        print("✅ Binary management completed successfully")
        return

    # Handle management operations
    elif mode == 'management':
        ops = plan['management_operations']
        
        # Operations that don't need CONFLUENCE instance
        if ops['list_templates']:
            cli_manager.list_templates()
        
        if ops['update_config']:
            cli_manager.update_config(ops['update_config'])
        
        if ops['validate_environment']:
            cli_manager.validate_environment()
        
        # Operations that need CONFLUENCE instance
        confluence = None
        if ops['workflow_status'] or ops['resume_from'] or ops['clean']:
            try:
                confluence = CONFLUENCE(
                    config_path=config_path,
                    config_overrides=config_overrides,
                    debug_mode=settings.get('debug', False)
                )
            except Exception as e:
                print(f"⚠️  Could not initialize CONFLUENCE: {str(e)}")
                if ops['workflow_status'] or ops['resume_from']:
                    print("Cannot perform workflow operations without valid configuration")
                    return
        
        if ops['workflow_status']:
            status = cli_manager.get_detailed_workflow_status(confluence)
            cli_manager.print_detailed_workflow_status(status)
        
        if ops['resume_from']:
            steps = cli_manager.resume_workflow_from_step(ops['resume_from'], confluence)
            if steps:
                confirm = input(f"\n🚀 Execute {len(steps)} steps? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    confluence.run_individual_steps(steps)
                else:
                    print("❌ Workflow resume cancelled")
        
        if ops['clean']:
            cli_manager.clean_workflow_files(
                clean_level=ops['clean_level'],
                confluence_instance=confluence,
                dry_run=settings.get('dry_run', False)
            )
        
        return
    
    # Handle status-only operations
    elif mode == 'status_only':
        try:
            confluence = CONFLUENCE(
                config_path=config_path,
                config_overrides=config_overrides,
                debug_mode=settings.get('debug', False)
            )
            cli_manager.print_status_information(confluence, plan['status_operations'])
        except:
            print("⚠️  Could not initialize CONFLUENCE for status check")
            cli_manager.print_status_information(None, plan['status_operations'])
        return
    
    # Initialize CONFLUENCE for workflow operations
    confluence = CONFLUENCE(
        config_path=config_path,
        config_overrides=config_overrides,
        debug_mode=settings.get('debug', False)
    )
    
    # Handle dry run
    if settings.get('dry_run'):
        print_dry_run_plan(plan, confluence)
        return
    
    # Execute based on mode
    if mode == 'pour_point_setup':
        pour_point_info = plan['pour_point']
        confluence.setup_pour_point_workflow(
            pour_point_info['coordinates'],
            pour_point_info['domain_definition_method']
        )
    elif mode == 'individual_steps':
        confluence.run_individual_steps(plan['steps'])
    elif mode == 'workflow':
        confluence.run_workflow()
    else:
        raise ValueError(f"Unknown execution mode: {mode}")


def print_dry_run_plan(plan: Dict[str, Any], confluence) -> None:
    """Print what would be executed in a dry run."""
    print("\n🔍 Dry Run - Execution Plan:")
    print("=" * 40)
    print(f"Mode: {plan['mode']}")
    
    if plan['mode'] == 'pour_point_setup':
        pour_point = plan['pour_point']
        print(f"Pour Point: {pour_point['coordinates']}")
        print(f"Domain Definition: {pour_point['domain_definition_method']}")
        print("Steps that would be executed:")
        for step in ['setup_project', 'create_pour_point', 'define_domain', 'discretize_domain']:
            print(f"  - {step}")
    
    elif plan['mode'] == 'individual_steps':
        print("Individual steps that would be executed:")
        for step in plan['steps']:
            print(f"  - {step}")
    
    elif plan['mode'] == 'workflow':
        print("Complete workflow would be executed")
    
    if plan.get('config_overrides'):
        print("\nConfiguration overrides:")
        for key, value in plan['config_overrides'].items():
            print(f"  {key}: {value}")
    
    print("\nSettings:")
    for key, value in plan['settings'].items():
        print(f"  {key}: {value}")
    
    print("\nTo execute this plan, remove the --dry_run flag")


def main() -> None:
    """
    Enhanced main entry point for CONFLUENCE with comprehensive CLI support.
    """
    try:
        # Initialize CLI argument manager
        cli_manager = CLIArgumentManager()
        
        # Parse arguments
        args = cli_manager.parse_arguments()
        
        # Validate arguments
        is_valid, errors = cli_manager.validate_arguments(args)
        if not is_valid:
            print("❌ Argument validation errors:")
            for error in errors:
                print(f"   {error}")
            sys.exit(1)
        
        # Get execution plan
        plan = cli_manager.get_execution_plan(args)
        
        # Handle pour point setup FIRST (before config loading)
        if plan['mode'] == 'pour_point_setup':
            pour_point_info = plan['pour_point']
            
            # Create config file using CLI manager
            setup_result = cli_manager.setup_pour_point_workflow(
                coordinates=pour_point_info['coordinates'],
                domain_def_method=pour_point_info['domain_definition_method'],
                domain_name=pour_point_info['domain_name'],
                bounding_box_coords=pour_point_info.get('bounding_box_coords'),
                confluence_code_dir=None  # Let it auto-detect
            )
            
            # Use the newly created config file
            config_path = Path(setup_result['config_file'])
        else:
            # Use the specified config file for other modes
            config_path = Path(args.config)
        
        # Display startup message
        if not plan.get('settings', {}).get('dry_run') and plan['mode'] != 'status_only':
            debug_message = " (DEBUG MODE)" if args.debug else ""
            print(f"🚀 Starting CONFLUENCE with configuration: {config_path}{debug_message}")
            if plan['mode'] != 'workflow':
                print(f"   Execution mode: {plan['mode']}")
        
        # Execute the plan
        execute_confluence_plan(plan, config_path)
        
        # Success message
        if plan['mode'] != 'status_only' and not plan.get('settings', {}).get('dry_run'):
            print("✅ CONFLUENCE completed successfully")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"❌ Error running CONFLUENCE: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()