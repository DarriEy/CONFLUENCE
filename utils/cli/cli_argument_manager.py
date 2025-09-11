#!/usr/bin/env python3
"""
CONFLUENCE CLI Argument Manager Utility

Save this file as: utils/cli/cli_argument_manager.py

This utility provides comprehensive command-line interface functionality for the CONFLUENCE
hydrological modeling platform. It handles argument parsing, validation, and workflow
step execution control.

Features:
- Individual workflow step execution
- Pour point coordinate setup  
- Flexible configuration management
- Debug and logging controls
- Workflow validation and status reporting

Usage:
    from utils.cli.cli_argument_manager import CLIArgumentManager
    
    cli_manager = CLIArgumentManager()
    args = cli_manager.parse_arguments()
    plan = cli_manager.get_execution_plan(args)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml


class CLIArgumentManager:
    """
    Manages command-line arguments and workflow execution options for CONFLUENCE.
    
    This class provides a comprehensive CLI interface that allows users to:
    - Run individual workflow steps
    - Set up pour point configurations
    - Control workflow execution behavior
    - Manage configuration and debugging options
    
    The argument manager integrates with the existing CONFLUENCE workflow orchestrator
    to provide granular control over workflow execution.
    """
    
    def __init__(self):
        """Initialize the CLI argument manager."""
        self.parser = None
        self.workflow_steps = self._define_workflow_steps()
        self.domain_definition_methods = ['lumped', 'point', 'subset', 'delineate']
        self._setup_parser()
        
    def _define_workflow_steps(self) -> Dict[str, Dict[str, str]]:
        """
        Define available workflow steps that can be run individually.
        
        Returns:
            Dictionary mapping step names to their descriptions and manager methods
        """
        return {
            'setup_project': {
                'description': 'Initialize project directory structure and shapefiles',
                'manager': 'project',
                'method': 'setup_project',
                'function_name': 'setup_project'
            },
            'create_pour_point': {
                'description': 'Create pour point shapefile from coordinates',
                'manager': 'project', 
                'method': 'create_pour_point',
                'function_name': 'create_pour_point'
            },
            'acquire_attributes': {
                'description': 'Download and process geospatial attributes (soil, land class, etc.)',
                'manager': 'data',
                'method': 'acquire_attributes',
                'function_name': 'acquire_attributes'
            },
            'define_domain': {
                'description': 'Define hydrological domain boundaries and river basins',
                'manager': 'domain',
                'method': 'define_domain',
                'function_name': 'define_domain'
            },
            'discretize_domain': {
                'description': 'Discretize domain into HRUs or other modeling units',
                'manager': 'domain',
                'method': 'discretize_domain',
                'function_name': 'discretize_domain'
            },
            'setup_model': {
                'description': 'Setup model-specific input files and configuration',
                'manager': 'model',
                'method': 'setup_model',
                'function_name': 'setup_model'
            },
            'run_model': {
                'description': 'Execute the hydrological model simulation',
                'manager': 'model',
                'method': 'run_model',
                'function_name': 'run_model'
            },
            'calibrate_model': {
                'description': 'Run model calibration and parameter optimization',
                'manager': 'optimization',
                'method': 'run_calibration',
                'function_name': 'run_calibration'  # Note: actual function name differs from CLI name
            },
            'run_emulation': {
                'description': 'Run emulation-based optimization if configured',
                'manager': 'optimization',
                'method': 'run_emulation',
                'function_name': 'run_emulation'
            },
            'run_benchmarking': {
                'description': 'Run benchmarking analysis against observations',
                'manager': 'analysis',
                'method': 'run_benchmarking',
                'function_name': 'run_benchmarking'
            },
            'run_decision_analysis': {
                'description': 'Run decision analysis for model comparison',
                'manager': 'analysis',
                'method': 'run_decision_analysis',
                'function_name': 'run_decision_analysis'
            },
            'run_sensitivity_analysis': {
                'description': 'Run sensitivity analysis on model parameters',
                'manager': 'analysis',
                'method': 'run_sensitivity_analysis',
                'function_name': 'run_sensitivity_analysis'
            },
            'postprocess_results': {
                'description': 'Postprocess and finalize model results',
                'manager': 'model',
                'method': 'postprocess_results',
                'function_name': 'postprocess_results'
            }
        }

    def validate_step_availability(self, confluence_instance, requested_steps: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate that requested steps are available in the current workflow.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            requested_steps: List of step names requested by user
            
        Returns:
            Tuple of (available_steps, unavailable_steps)
        """
        available_steps = []
        unavailable_steps = []
        
        if hasattr(confluence_instance, 'workflow_orchestrator'):
            # Get actual workflow steps
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            actual_functions = {step_func.__name__ for step_func, _ in workflow_steps}
            
            # Check each requested step
            for step_name in requested_steps:
                if step_name in self.workflow_steps:
                    function_name = self.workflow_steps[step_name].get('function_name', step_name)
                    if function_name in actual_functions:
                        available_steps.append(step_name)
                    else:
                        unavailable_steps.append(step_name)
                else:
                    unavailable_steps.append(step_name)
        else:
            # Fallback - assume all defined steps are available
            for step_name in requested_steps:
                if step_name in self.workflow_steps:
                    available_steps.append(step_name)
                else:
                    unavailable_steps.append(step_name)
        
        return available_steps, unavailable_steps
    
    def _setup_parser(self) -> None:
        """Set up the argument parser with all CLI options."""
        self.parser = argparse.ArgumentParser(
            description='CONFLUENCE - Community Optimization Nexus for Leveraging Understanding of Environmental Networks in Computational Exploration',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )
        
        # Configuration options
        config_group = self.parser.add_argument_group('Configuration Options')
        config_group.add_argument(
            '--config', 
            type=str,
            default='./0_config_files/config_active.yaml',
            help='Path to YAML configuration file (default: ./0_config_files/config_active.yaml)'
        )
        config_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output and detailed logging'
        )
        config_group.add_argument(
            '--version',
            action='version',
            version='CONFLUENCE 1.0.0'
        )
        
        # Workflow execution options
        workflow_group = self.parser.add_argument_group('Workflow Execution')
        workflow_group.add_argument(
            '--run_workflow',
            action='store_true',
            help='Run the complete CONFLUENCE workflow (default behavior if no individual steps specified)'
        )
        workflow_group.add_argument(
            '--force_rerun',
            action='store_true',
            help='Force rerun of all steps, overwriting existing outputs'
        )
        workflow_group.add_argument(
            '--stop_on_error',
            action='store_true',
            default=True,
            help='Stop workflow execution on first error (default: True)'
        )
        workflow_group.add_argument(
            '--continue_on_error',
            action='store_true',
            help='Continue workflow execution even if errors occur'
        )
        
        # Individual workflow steps
        steps_group = self.parser.add_argument_group('Individual Workflow Steps')
        for step_name, step_info in self.workflow_steps.items():
            steps_group.add_argument(
                f'--{step_name}',
                action='store_true',
                help=step_info['description']
            )
        
        # Pour point setup
        pourpoint_group = self.parser.add_argument_group('Pour Point Setup')
        pourpoint_group.add_argument(
            '--pour_point',
            type=str,
            metavar='LAT/LON',
            help='Set up CONFLUENCE for a pour point coordinate (format: lat/lon, e.g., 51.1722/-115.5717)'
        )
        pourpoint_group.add_argument(
            '--domain_def',
            type=str,
            choices=self.domain_definition_methods,
            help=f'Domain definition method when using --pour_point. Options: {", ".join(self.domain_definition_methods)}'
        )
        pourpoint_group.add_argument(
            '--domain_name',
            type=str,
            help='Override domain name in configuration'
        )
        pourpoint_group.add_argument(
            '--experiment_id',
            type=str,
            help='Override experiment ID in configuration'
        )
        
        # Analysis and status options
        status_group = self.parser.add_argument_group('Status and Analysis')
        status_group.add_argument(
            '--status',
            action='store_true',
            help='Show current workflow status and exit'
        )
        status_group.add_argument(
            '--list_steps',
            action='store_true',
            help='List all available workflow steps and exit'
        )
        status_group.add_argument(
            '--validate_config',
            action='store_true',
            help='Validate configuration file and exit'
        )
        status_group.add_argument(
            '--dry_run',
            action='store_true',
            help='Show what would be executed without actually running'
        )
    
    def _get_examples_text(self) -> str:
        """Generate examples text for help output."""
        return """
Examples:
  # Run complete workflow with default configuration
  python CONFLUENCE.py
  
  # Run complete workflow with custom configuration
  python CONFLUENCE.py --config /path/to/config.yaml
  
  # Run only model calibration step
  python CONFLUENCE.py --calibrate_model
  
  # Run multiple specific steps
  python CONFLUENCE.py --setup_project --create_pour_point --define_domain
  
  # Set up for a specific pour point with delineation
  python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate
  
  # Run workflow with custom domain name
  python CONFLUENCE.py --domain_name "MyWatershed" --experiment_id "test_run"
  
  # Check workflow status
  python CONFLUENCE.py --status
  
  # Validate configuration
  python CONFLUENCE.py --validate_config
  
  # Run with debug output and force rerun
  python CONFLUENCE.py --debug --force_rerun
  
  # Dry run to see what would be executed
  python CONFLUENCE.py --dry_run

For more information, visit: https://github.com/DarriEy/CONFLUENCE
        """
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Args:
            args: Optional list of arguments to parse (for testing)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    
    def validate_arguments(self, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """
        Validate parsed arguments for logical consistency.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check pour point format
        if args.pour_point:
            if not self._validate_coordinates(args.pour_point):
                errors.append(f"Invalid pour point format: {args.pour_point}. Expected format: lat/lon (e.g., 51.1722/-115.5717)")
            
            if not args.domain_def:
                errors.append("--domain_def is required when using --pour_point")
        
        # Check conflicting options
        if args.stop_on_error and args.continue_on_error:
            errors.append("Cannot specify both --stop_on_error and --continue_on_error")
        
        # Check configuration file exists
        config_path = Path(args.config)
        if not config_path.exists():
            errors.append(f"Configuration file not found: {config_path}")
        
        return len(errors) == 0, errors
    
    def _validate_coordinates(self, coord_string: str) -> bool:
        """
        Validate coordinate string format.
        
        Args:
            coord_string: Coordinate string in format "lat/lon"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = coord_string.split('/')
            if len(parts) != 2:
                return False
            
            lat, lon = float(parts[0]), float(parts[1])
            
            # Basic range validation
            if not (-90 <= lat <= 90):
                return False
            if not (-180 <= lon <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
    
    def get_execution_plan(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Determine what should be executed based on parsed arguments.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Dictionary describing the execution plan
        """
        plan = {
            'mode': 'workflow',  # 'workflow', 'individual_steps', 'pour_point_setup', 'status_only'
            'steps': [],
            'config_overrides': {},
            'settings': {
                'force_rerun': args.force_rerun,
                'stop_on_error': args.stop_on_error and not args.continue_on_error,
                'debug': args.debug,
                'dry_run': args.dry_run
            }
        }
        
        # Handle status-only operations
        if args.status or args.list_steps or args.validate_config:
            plan['mode'] = 'status_only'
            plan['status_operations'] = {
                'show_status': args.status,
                'list_steps': args.list_steps,
                'validate_config': args.validate_config
            }
            return plan
        
        # Handle pour point setup
        if args.pour_point:
            plan['mode'] = 'pour_point_setup'
            plan['pour_point'] = {
                'coordinates': args.pour_point,
                'domain_definition_method': args.domain_def
            }
        
        # Check for individual step execution
        individual_steps = []
        for step_name in self.workflow_steps.keys():
            if getattr(args, step_name, False):
                individual_steps.append(step_name)
        
        if individual_steps:
            plan['mode'] = 'individual_steps'
            plan['steps'] = individual_steps
        elif not args.pour_point and not args.run_workflow:
            # Default to full workflow if no specific steps requested
            plan['mode'] = 'workflow'
        
        # Handle configuration overrides
        if args.domain_name:
            plan['config_overrides']['DOMAIN_NAME'] = args.domain_name
        
        if args.experiment_id:
            plan['config_overrides']['EXPERIMENT_ID'] = args.experiment_id
        
        if args.pour_point:
            plan['config_overrides']['POUR_POINT_COORDS'] = args.pour_point
            plan['config_overrides']['DOMAIN_DEFINITION_METHOD'] = args.domain_def
        
        if args.force_rerun:
            plan['config_overrides']['FORCE_RUN_ALL_STEPS'] = True
        
        return plan
    
    def apply_config_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides from CLI arguments.
        
        Args:
            config: Original configuration dictionary
            overrides: Override values from CLI
            
        Returns:
            Updated configuration dictionary
        """
        updated_config = config.copy()
        updated_config.update(overrides)
        return updated_config
    
    def print_status_information(self, confluence_instance, operations: Dict[str, bool]) -> None:
        """
        Print various status information based on requested operations.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            operations: Dictionary of status operations to perform
        """
        if operations.get('list_steps'):
            self._print_workflow_steps()
        
        if operations.get('validate_config'):
            self._print_config_validation(confluence_instance)
        
        if operations.get('show_status'):
            self._print_workflow_status(confluence_instance)
    
    def _print_workflow_steps(self) -> None:
        """Print all available workflow steps."""
        print("\n📋 Available Workflow Steps:")
        print("=" * 50)
        for step_name, step_info in self.workflow_steps.items():
            print(f"--{step_name:<25} {step_info['description']}")
        print("\n💡 Use these flags to run individual steps, e.g.:")
        print("  python CONFLUENCE.py --setup_project --create_pour_point")
        print()
    
    def _print_config_validation(self, confluence_instance) -> None:
        """Print configuration validation results."""
        print("\n🔍 Configuration Validation:")
        print("=" * 30)
        
        # This would integrate with the CONFLUENCE validation methods
        if hasattr(confluence_instance, 'workflow_orchestrator'):
            is_valid = confluence_instance.workflow_orchestrator.validate_workflow_prerequisites()
            if is_valid:
                print("✅ Configuration is valid")
            else:
                print("❌ Configuration validation failed")
                print("Check logs for detailed error information")
        else:
            print("⚠️  Configuration validation not available")
    
    def _print_workflow_status(self, confluence_instance) -> None:
        """Print current workflow status."""
        print("\n📊 Workflow Status:")
        print("=" * 20)
        
        if hasattr(confluence_instance, 'get_status'):
            status = confluence_instance.get_status()
            print(f"🏞️  Domain: {status.get('domain', 'Unknown')}")
            print(f"🧪 Experiment: {status.get('experiment', 'Unknown')}")
            print(f"⚙️  Config Valid: {'✅' if status.get('config_valid', False) else '❌'}")
            print(f"🔧 Managers Initialized: {'✅' if status.get('managers_initialized', False) else '❌'}")
            print(f"📁 Config Path: {status.get('config_path', 'Unknown')}")
            print(f"🐛 Debug Mode: {'✅' if status.get('debug_mode', False) else '❌'}")
            
            if 'workflow_status' in status:
                workflow_status = status['workflow_status']
                print(f"🔄 Workflow Status: {workflow_status}")
        else:
            print("⚠️  Status information not available")
    
    def setup_pour_point_workflow(self, coordinates: str, domain_def_method: str) -> Dict[str, Any]:
        """
        Create a configuration setup for pour point workflow.
        
        This is a placeholder implementation that would be expanded based on
        specific requirements for pour point setup workflows.
        
        Args:
            coordinates: Pour point coordinates in "lat/lon" format
            domain_def_method: Domain definition method to use
            
        Returns:
            Dictionary with pour point workflow configuration
        """
        # Placeholder implementation - would be expanded based on requirements
        print(f"\n🎯 Setting up pour point workflow:")
        print(f"   📍 Coordinates: {coordinates}")
        print(f"   🗺️  Domain Definition Method: {domain_def_method}")
        print(f"   🔧 This will configure CONFLUENCE for the specified pour point location")
        print(f"   📝 [PLACEHOLDER - Implementation needed based on specific requirements]")
        
        # Return configuration for pour point setup
        return {
            'coordinates': coordinates,
            'domain_definition_method': domain_def_method,
            'setup_steps': [
                'setup_project',
                'create_pour_point', 
                'define_domain',
                'discretize_domain'
            ]
        }


def create_cli_manager() -> CLIArgumentManager:
    """
    Factory function to create a CLI argument manager instance.
    
    Returns:
        Configured CLIArgumentManager instance
    """
    return CLIArgumentManager()


# Example usage and testing
if __name__ == "__main__":
    # This allows the utility to be tested independently
    cli_manager = CLIArgumentManager()
    
    # Example: parse some test arguments
    test_args = ['--calibrate_model', '--debug']
    args = cli_manager.parse_arguments(test_args)
    
    # Get execution plan
    plan = cli_manager.get_execution_plan(args)
    
    print("Test execution plan:")
    print(f"Mode: {plan['mode']}")
    print(f"Steps: {plan['steps']}")
    print(f"Settings: {plan['settings']}")