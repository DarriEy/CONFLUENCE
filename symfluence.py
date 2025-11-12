#!/usr/bin/env python3
"""
SYMFLUENCE - SYnergistic Modelling Framework for Linking and Unifying
Earth-system Nexii for Computational Exploration

Enhanced main entry point for the SYMFLUENCE hydrological modeling platform.
SYMFLUENCE provides an integrated framework for:
- Hydrological model setup and configuration
- Multi-model comparison and evaluation  
- Parameter optimization and calibration
- Workflow management
- Individual workflow step execution
- Pour point-based domain setup
- SLURM job submission for HPC environments

Usage:
    # Run complete workflow
    python symfluence.py [--config CONFIG_PATH]
    
    # Run individual workflow steps
    python symfluence.py --calibrate_model --run_benchmarking
    
    # Set up for a specific pour point
    python symfluence.py --pour_point 51.1722/-115.5717 --domain_def delineate
    
    # Submit as SLURM job
    python symfluence.py --calibrate_model --submit_job --job_account ees250064
    
    # Check status
    python symfluence.py --status

For more information, see the documentation at:
    https://github.com/DarriEy/symfluence

License: GPL-3.0 License
"""
try:
    from symfluence_version import __version__
except Exception:
    __version__ = "0+unknown"


from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Any, List
import sys

# Import SYMFLUENCE components
from utils.project.project_manager import ProjectManager
from utils.project.workflow_orchestrator import WorkflowOrchestrator
from utils.project.logging_manager import LoggingManager
from utils.data.data_manager import DataManager
from utils.geospatial.domain_manager import DomainManager
from utils.models.model_manager import ModelManager
from utils.evaluation.analysis_manager import AnalysisManager
from utils.optimization.optimization_manager import OptimizationManager
from utils.cli.cli_argument_manager import CLIArgumentManager


class SYMFLUENCE:
    """
    Enhanced SYMFLUENCE main class with comprehensive CLI support.
    
    This class serves as the central coordinator for all SYMFLUENCE operations,
    with enhanced CLI capabilities including individual step execution,
    pour point setup, SLURM job submission, and comprehensive workflow management.
    """
    
    def __init__(self, config_path: Path, config_overrides: Dict[str, Any] = None, debug_mode: bool = False):
        """
        Initialize the SYMFLUENCE system with configuration and CLI options.
        
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

        self.logger.info(f"SYMFLUENCE initialized with config: {config_path}")
        if self.config_overrides:
            self.logger.info(f"Configuration overrides applied: {list(self.config_overrides.keys())}")


        # Initialize managers
        self.managers = self._initialize_managers()
        
        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            self.managers, self.config, self.logger, self.logging_manager
        )
        
    
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
        """Execute the complete SYMFLUENCE workflow (CLI wrapper)."""
        start = datetime.now()
        steps_completed = []
        errors = []
        warns = []

        try:
            self.logger.info("Starting complete SYMFLUENCE workflow execution")

            # Run the workflow; if your orchestrator exposes steps executed, collect them
            self.workflow_orchestrator.run_workflow()
            steps_completed = getattr(self.workflow_orchestrator, "steps_executed", []) or []

            status = getattr(self.workflow_orchestrator, "get_workflow_status", lambda: "completed")()
            self.logger.info("Complete SYMFLUENCE workflow execution completed")

        except Exception as e:
            status = "failed"
            errors.append({"where": "run_workflow", "error": str(e)})
            self.logger.error(f"Workflow execution failed: {e}")
            # re-raise after summary so the CI can fail meaningfully if needed
            raise
        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            # Call with the expected signature:
            self.logging_manager.create_run_summary(
                steps_completed=steps_completed,
                errors=errors,
                warnings=warns,
                execution_time=elapsed_s,
                status=status,
            )
        
    def run_individual_steps(self, step_names: List[str]) -> None:
        """Execute specific workflow steps (CLI wrapper)."""
        start = datetime.now()
        steps_completed = []
        errors = []
        warns = []

        # Resolve workflow functions once
        workflow_steps = self.workflow_orchestrator.define_workflow_steps()
        name_to_fn = {}
        for item in workflow_steps:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                fn, _check = item[0], item[1]
                if hasattr(fn, "__name__"):
                    name_to_fn[fn.__name__] = fn

        cli_to_function = {
            "setup_project": "setup_project",
            "create_pour_point": "create_pour_point",
            "acquire_attributes": "acquire_attributes",
            "define_domain": "define_domain",
            "discretize_domain": "discretize_domain",
            "setup_model": "preprocess_models",
            "run_model": "run_models",
            "calibrate_model": "calibrate_model",
            "run_emulation": "run_emulation",
            "run_benchmarking": "run_benchmarking",
            "run_decision_analysis": "run_decision_analysis",
            "run_sensitivity_analysis": "run_sensitivity_analysis",
            "postprocess_results": "postprocess_results",
            "process_observed_data": "process_observed_data",
            "acquire_forcings": "acquire_forcings",
            "model_agnostic_preprocessing": "run_model_agnostic_preprocessing",
            "model_specific_preprocessing": "preprocess_models",
        }

        status = "completed"
        try:
            self.logger.info(f"Starting individual step execution: {', '.join(step_names)}")

            for cli_name in step_names:
                impl = cli_to_function.get(cli_name)
                if not impl:
                    self.logger.warning(f"Step '{cli_name}' not recognized; skipping")
                    continue

                fn = name_to_fn.get(impl)
                if not fn:
                    self.logger.warning(f"Function '{impl}' not found; skipping step '{cli_name}'")
                    continue

                self.logger.info(f"Executing step: {cli_name} -> {impl}")
                # Force execution; skip completion checks in CLI wrapper
                try:
                    fn()
                    steps_completed.append(cli_name)
                    self.logger.info(f"✓ Completed step: {cli_name}")
                except Exception as e:
                    status = "partial" if steps_completed else "failed"
                    errors.append({"step": cli_name, "error": str(e)})
                    self.logger.error(f"Step '{cli_name}' failed: {e}")
                    # You can either continue or re-raise; the wrapper flag decides
                    raise
        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            self.logging_manager.create_run_summary(
                steps_completed=steps_completed,
                errors=errors,
                warnings=warns,
                execution_time=elapsed_s,
                status=status,
            )


def main():
    """
    Main entry point for SYMFLUENCE with comprehensive CLI support.
    """
    try:
        # Initialize CLI argument manager
        cli_manager = CLIArgumentManager()
        
        # Parse arguments
        args = cli_manager.parse_arguments()
        
        # Validate arguments
        is_valid, errors = cli_manager.validate_arguments(args)
        if not is_valid:
            print("Argument validation errors:")
            for error in errors:
                print(f"   {error}")
            sys.exit(1)
        
        # Get execution plan
        plan = cli_manager.get_execution_plan(args)
        
        if plan.get('mode') == 'example_notebook':
            ex_id = plan['example_notebook']
            rc = cli_manager.launch_example_notebook(ex_id)
            sys.exit(rc)

        # Handle pour point setup FIRST (before config loading)
        if plan['mode'] == 'pour_point_setup':
            pour_point_info = plan['pour_point']
            
            # Create config file using CLI manager
            setup_result = cli_manager.setup_pour_point_workflow(
                coordinates=pour_point_info['coordinates'],
                domain_def_method=pour_point_info['domain_definition_method'],
                domain_name=pour_point_info['domain_name'],
                bounding_box_coords=pour_point_info.get('bounding_box_coords'),
                symfluence_code_dir=None  # Let it auto-detect
            )
            
            # Use the newly created config file
            config_path = Path(setup_result['config_file'])
            
        # Handle SLURM job submission for pour point setup
        elif plan['mode'] == 'slurm_job' and plan.get('job_mode') == 'pour_point_setup':
            pour_point_info = plan['pour_point']
            
            # Create config file using CLI manager (same as above)
            setup_result = cli_manager.setup_pour_point_workflow(
                coordinates=pour_point_info['coordinates'],
                domain_def_method=pour_point_info['domain_definition_method'],
                domain_name=pour_point_info['domain_name'],
                bounding_box_coords=pour_point_info.get('bounding_box_coords'),
                symfluence_code_dir=None
            )
            
            # Use the newly created config file for the SLURM job
            config_path = Path(setup_result['config_file'])
            
        else:
            # Use the specified config file for other modes
            config_path = Path(args.config)
        
        # Display startup information (updated for SYMFLUENCE)
        if not (plan.get('mode') in ['status_only', 'management', 'binary_management']):
            print(f"\n{'='*70}")
            print(f"SYMFLUENCE - SYnergistic Modelling Framework")
            print(f"for Linking and Unifying Earth-system Nexii")
            print(f"for Computational Exploration")
            print(f"{'='*70}")
            print(f"Version: {__version__}")
            print(f"Config: {config_path}")
            if plan['settings'].get('debug'):
                print(f"Mode: DEBUG")
            print(f"{'='*70}\n")
        
        # Handle binary management operations
        if plan['mode'] == 'binary_management':
            binary_ops = plan['binary_operations']
            
            if binary_ops.get('validate_binaries'):
                # Try new signature first (no args), then instance-based, then legacy positional
                try:
                    result = cli_manager.validate_binaries()
                except TypeError:
                    try:
                        sym_inst = SYMFLUENCE(
                            config_path,
                            debug_mode=plan['settings'].get('debug', False)
                        )
                        result = cli_manager.validate_binaries(symfluence_instance=sym_inst)
                    except TypeError:
                        # Last resort: legacy positional config_path
                        result = cli_manager.validate_binaries(config_path)
                sys.exit(0 if (result is True) else 1)
            
            if binary_ops.get('get_executables') is not None:
                symfluence_instance = SYMFLUENCE(config_path, debug_mode=plan['settings'].get('debug', False))
                specific_tools = binary_ops['get_executables'] if binary_ops['get_executables'] else None
                result = cli_manager.get_executables(
                    specific_tools=specific_tools,
                    symfluence_instance=symfluence_instance,
                    force=binary_ops.get('force_install', False),
                    dry_run=plan['settings'].get('dry_run', False)
                )
                sys.exit(0 if result.get('successful') else 1)
        
        # Handle management operations
        if plan['mode'] == 'management':
            mgmt_ops = plan['management_operations']
            
            if mgmt_ops['list_templates']:
                cli_manager.list_config_templates()
                sys.exit(0)
            
            if mgmt_ops['update_config']:
                cli_manager.update_config(mgmt_ops['update_config'])
                sys.exit(0)
            
            if mgmt_ops['validate_environment']:
                cli_manager.validate_environment()
                sys.exit(0)
            
            if mgmt_ops['workflow_status'] or mgmt_ops['resume_from'] or mgmt_ops['clean']:
                print("Management operations not yet fully implemented")
                sys.exit(1)
        
        # Handle status-only operations
        if plan['mode'] == 'status_only':
            status_ops = plan['status_operations']
            
            if status_ops['list_steps']:
                cli_manager.list_workflow_steps()
                sys.exit(0)
            
            if status_ops['show_status'] or status_ops['validate_config']:
                print("Status operations require a SYMFLUENCE instance")
                # These would be implemented with a minimal SYMFLUENCE instance
                sys.exit(0)
        
        # Initialize SYMFLUENCE instance for workflow execution
        config_overrides = plan.get('config_overrides', {})
        symfluence = SYMFLUENCE(
            config_path,
            config_overrides=config_overrides,
            debug_mode=plan['settings'].get('debug', False)
        )
        
        # Handle SLURM job submission
        if plan['mode'] == 'slurm_job':
            success = cli_manager.handle_slurm_submission(plan, config_path, symfluence)
            sys.exit(0 if success else 1)
        
        # Execute workflow based on mode
        if plan['mode'] == 'individual_steps':
            if plan['settings'].get('dry_run'):
                print("\n🔍 DRY RUN - Steps that would be executed:")
                for step in plan['steps']:
                    print(f"   • {step}")
                print("\nNo actual execution performed.\n")
            else:
                symfluence.run_individual_steps(plan['steps'])
        else:
            # Full workflow execution
            if plan['settings'].get('dry_run'):
                print("\n🔍 DRY RUN - Complete workflow would be executed")
                print("No actual execution performed.\n")
            else:
                symfluence.run_workflow()
        
        print(f"\n{'='*70}")
        print("SYMFLUENCE execution completed successfully")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
