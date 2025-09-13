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
            'process_observed_data': {
                'description': 'Process observational data (streamflow, etc.)',
                'manager': 'data',
                'method': 'process_observed_data',
                'function_name': 'process_observed_data'
            },
            'acquire_forcings': {
                'description': 'Acquire meteorological forcing data',
                'manager': 'data',
                'method': 'acquire_forcings',
                'function_name': 'acquire_forcings'
            },
            'model_agnostic_preprocessing': {
                'description': 'Run model-agnostic preprocessing of forcing and attribute data',
                'manager': 'data',
                'method': 'model_agnostic_preprocessing',
                'function_name': 'model_agnostic_preprocessing'
            },
            'model_specific_preprocessing': {
                'description': 'Setup model-specific input files and configuration',
                'manager': 'model',
                'method': 'model_specific_preprocessing',
                'function_name': 'model_specific_preprocessing'
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

    def resume_workflow_from_step(self, step_name: str, confluence_instance) -> List[str]:
        """
        Determine which steps to execute when resuming from a specific step.
        
        Args:
            step_name: Name of the step to resume from
            confluence_instance: CONFLUENCE system instance
            
        Returns:
            List of step names to execute
        """
        print(f"\n🔄 Resuming Workflow from: {step_name}")
        print("=" * 50)
        
        try:
            # Get workflow definition
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            
            # Create mapping from function names to steps
            step_functions = {}
            step_order = []
            
            for i, (step_func, check_func, description) in enumerate(workflow_steps):
                func_name = step_func.__name__
                step_functions[func_name] = {
                    'function': step_func,
                    'check': check_func,
                    'description': description,
                    'order': i
                }
                step_order.append(func_name)
            
            # Map CLI step names to function names
            cli_to_function_map = {
                'setup_project': 'setup_project',
                'create_pour_point': 'create_pour_point',
                'acquire_attributes': 'acquire_attributes',
                'define_domain': 'define_domain',
                'discretize_domain': 'discretize_domain',
                'process_observed_data': 'process_observed_data',
                'acquire_forcings': 'acquire_forcings',
                'run_model_agnostic_preprocessing': 'run_model_agnostic_preprocessing',
                'setup_model': 'preprocess_models',
                'run_model': 'run_models',
                'calibrate_model': 'calibrate_model',
                'run_emulation': 'run_emulation',
                'run_benchmarking': 'run_benchmarking',
                'run_decision_analysis': 'run_decision_analysis',
                'run_sensitivity_analysis': 'run_sensitivity_analysis',
                'postprocess_results': 'postprocess_results'
            }
            
            # Find the function name for the CLI step
            function_name = cli_to_function_map.get(step_name, step_name)
            
            if function_name not in step_functions:
                print(f"❌ Step '{step_name}' not found in workflow")
                print(f"Available steps: {', '.join(cli_to_function_map.keys())}")
                return []
            
            # Find the starting position
            start_index = step_functions[function_name]['order']
            
            # Get steps to execute (from start_index onwards)
            steps_to_execute = []
            skipped_steps = []
            
            for i, func_name in enumerate(step_order):
                cli_name = None
                # Find CLI name for this function
                for cli_step, func_step in cli_to_function_map.items():
                    if func_step == func_name:
                        cli_name = cli_step
                        break
                
                if i < start_index:
                    # Check if earlier steps are completed
                    step_info = step_functions[func_name]
                    try:
                        is_completed = step_info['check']() if callable(step_info['check']) else False
                        if is_completed:
                            skipped_steps.append({
                                'name': cli_name or func_name,
                                'status': 'completed',
                                'description': step_info['description']
                            })
                        else:
                            skipped_steps.append({
                                'name': cli_name or func_name,
                                'status': 'missing_dependencies',
                                'description': step_info['description']
                            })
                    except:
                        skipped_steps.append({
                            'name': cli_name or func_name,
                            'status': 'unknown',
                            'description': step_info['description']
                        })
                else:
                    if cli_name:
                        steps_to_execute.append(cli_name)
            
            # Print resume plan
            print(f"📍 Starting from step {start_index + 1}: {step_name}")
            print(f"🎯 Will execute {len(steps_to_execute)} steps")
            
            if skipped_steps:
                print(f"\n⏭️  Skipped Steps (executed previously):")
                for i, step in enumerate(skipped_steps, 1):
                    status_icon = {
                        'completed': '✅',
                        'missing_dependencies': '⚠️ ',
                        'unknown': '❓'
                    }.get(step['status'], '❓')
                    
                    print(f"   {i:2d}. {status_icon} {step['name']} - {step['description']}")
                    
                    if step['status'] == 'missing_dependencies':
                        print(f"       ⚠️  Warning: Dependencies may be missing")
            
            if steps_to_execute:
                print(f"\n🚀 Steps to Execute:")
                for i, step in enumerate(steps_to_execute, start_index + 1):
                    step_description = step_functions.get(
                        cli_to_function_map.get(step, step), {}
                    ).get('description', step)
                    print(f"   {i:2d}. ⏳ {step} - {step_description}")
            
            # Check for potential issues
            warnings = []
            missing_deps = [s for s in skipped_steps if s['status'] == 'missing_dependencies']
            if missing_deps:
                warnings.append(f"⚠️  {len(missing_deps)} earlier steps appear incomplete")
            
            if warnings:
                print(f"\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"   {warning}")
                print(f"   Consider running --workflow_status to check dependencies")
            
            print(f"\n💡 Resume command: python CONFLUENCE.py {' --'.join([''] + steps_to_execute)}")
            
            return steps_to_execute
            
        except Exception as e:
            print(f"❌ Error planning workflow resume: {str(e)}")
            return []

    def clean_workflow_files(self, clean_level: str = 'intermediate', confluence_instance = None, 
                            dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean workflow files and directories based on the specified level.
        
        Args:
            clean_level: Level of cleaning ('intermediate', 'outputs', 'all')
            confluence_instance: CONFLUENCE system instance
            dry_run: If True, show what would be cleaned without actually deleting
            
        Returns:
            Dictionary with cleaning results
        """
        print(f"\n🧹 Cleaning Workflow Files (Level: {clean_level})")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN - No files will be deleted")
            print("-" * 30)
        
        cleaning_results = {
            'level': clean_level,
            'dry_run': dry_run,
            'files_removed': [],
            'directories_removed': [],
            'errors': [],
            'total_size_freed': 0,
            'summary': {}
        }
        
        if not confluence_instance:
            print("❌ Cannot clean files without CONFLUENCE instance")
            return cleaning_results
        
        try:
            project_dir = confluence_instance.workflow_orchestrator.project_dir
            
            if not project_dir.exists():
                print(f"📁 Project directory not found: {project_dir}")
                return cleaning_results
            
            print(f"📁 Project directory: {project_dir}")
            
            # Define what to clean at each level
            clean_targets = self._get_clean_targets(project_dir, clean_level)
            
            # Show what will be cleaned
            print(f"\n🎯 Targets for cleaning:")
            for category, paths in clean_targets.items():
                if paths:
                    print(f"   📂 {category.title()}:")
                    for path in paths:
                        if path.exists():
                            size = self._get_path_size(path)
                            size_str = f" ({size:.1f} MB)" if size > 0 else ""
                            print(f"      🗂️  {path.relative_to(project_dir)}{size_str}")
                        else:
                            print(f"      ❓ {path.relative_to(project_dir)} (not found)")
            
            if not any(clean_targets.values()):
                print("✨ No files found to clean!")
                return cleaning_results
            
            # Confirm deletion (unless dry run)
            if not dry_run:
                total_size = sum(self._get_path_size(path) 
                            for paths in clean_targets.values() 
                            for path in paths if path.exists())
                
                print(f"\n💾 Total space to be freed: {total_size:.1f} MB")
                confirm = input("⚠️  Proceed with deletion? (y/N): ").strip().lower()
                
                if confirm not in ['y', 'yes']:
                    print("❌ Cleaning cancelled by user")
                    return cleaning_results
            
            # Perform cleaning
            print(f"\n🧹 {'Would clean' if dry_run else 'Cleaning'} files...")
            
            for category, paths in clean_targets.items():
                category_results = {
                    'files_removed': 0,
                    'directories_removed': 0,
                    'size_freed': 0,
                    'errors': 0
                }
                
                for path in paths:
                    if not path.exists():
                        continue
                    
                    try:
                        size = self._get_path_size(path)
                        
                        if not dry_run:
                            if path.is_file():
                                path.unlink()
                                cleaning_results['files_removed'].append(str(path))
                                category_results['files_removed'] += 1
                            elif path.is_dir():
                                import shutil
                                shutil.rmtree(path)
                                cleaning_results['directories_removed'].append(str(path))
                                category_results['directories_removed'] += 1
                        else:
                            # Dry run - just record what would be removed
                            if path.is_file():
                                cleaning_results['files_removed'].append(str(path))
                                category_results['files_removed'] += 1
                            elif path.is_dir():
                                cleaning_results['directories_removed'].append(str(path))
                                category_results['directories_removed'] += 1
                        
                        category_results['size_freed'] += size
                        cleaning_results['total_size_freed'] += size
                        
                        action = "Would remove" if dry_run else "Removed"
                        print(f"   {'🗑️ ' if not dry_run else '👁️ '} {action}: {path.relative_to(project_dir)} ({size:.1f} MB)")
                        
                    except Exception as e:
                        error_msg = f"Error cleaning {path}: {str(e)}"
                        cleaning_results['errors'].append(error_msg)
                        category_results['errors'] += 1
                        print(f"   ❌ {error_msg}")
                
                cleaning_results['summary'][category] = category_results
            
            # Print summary
            total_files = len(cleaning_results['files_removed'])
            total_dirs = len(cleaning_results['directories_removed'])
            total_size = cleaning_results['total_size_freed']
            total_errors = len(cleaning_results['errors'])
            
            print(f"\n📊 Cleaning Summary:")
            action = "Would clean" if dry_run else "Cleaned"
            print(f"   {action}: {total_files} files, {total_dirs} directories")
            print(f"   💾 Space {'that would be' if dry_run else ''} freed: {total_size:.1f} MB")
            
            if total_errors > 0:
                print(f"   ❌ Errors: {total_errors}")
            
            if not dry_run:
                print("   ✅ Cleaning completed successfully!")
            else:
                print("   👁️  Dry run completed - use without --dry_run to actually clean")
            
        except Exception as e:
            error_msg = f"Error during cleaning: {str(e)}"
            cleaning_results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
        
        return cleaning_results

    def _get_clean_targets(self, project_dir: Path, clean_level: str) -> Dict[str, List[Path]]:
        """Get list of files/directories to clean based on level."""
        targets = {
            'intermediate': [],
            'logs': [],
            'outputs': [],
            'all': []
        }
        
        # Intermediate files (temporary processing files)
        intermediate_patterns = [
            'tmp*', '*.tmp', '*.temp', 
            '*_temp*', '*_intermediate*',
            '*/cache/*', '*/temp/*'
        ]
        
        # Log files
        log_patterns = [
            '*.log', 'logs/*', '*_log.txt',
            'debug_*', 'error_*'
        ]
        
        # Output files (results that can be regenerated)
        output_dirs = [
            'simulations/*/tmp*',
            'forcing/*/intermediate*', 
            'optimisation/*/iteration_*',
            'plots/*/temp*'
        ]
        
        if clean_level in ['intermediate', 'all']:
            # Find intermediate files
            for pattern in intermediate_patterns:
                targets['intermediate'].extend(project_dir.glob(pattern))
                targets['intermediate'].extend(project_dir.rglob(pattern))
        
        if clean_level in ['intermediate', 'outputs', 'all']:
            # Find log files
            for pattern in log_patterns:
                targets['logs'].extend(project_dir.glob(pattern))
                targets['logs'].extend(project_dir.rglob(pattern))
        
        if clean_level in ['outputs', 'all']:
            # Find output directories
            for pattern in output_dirs:
                targets['outputs'].extend(project_dir.glob(pattern))
        
        if clean_level == 'all':
            # Add specific directories for complete cleaning
            all_clean_dirs = [
                project_dir / 'simulations' / 'temp',
                project_dir / 'optimisation' / 'temp',
                project_dir / 'forcing' / 'temp',
                project_dir / 'plots' / 'temp'
            ]
            targets['all'].extend([d for d in all_clean_dirs if d.exists()])
        
        # Remove duplicates and sort
        for key in targets:
            targets[key] = sorted(list(set(targets[key])))
        
        return targets

    def _get_path_size(self, path: Path) -> float:
        """Get size of file or directory in MB."""
        if not path.exists():
            return 0.0
        
        try:
            if path.is_file():
                return path.stat().st_size / (1024 * 1024)  # Convert to MB
            elif path.is_dir():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
        
        return 0.0

    def get_detailed_workflow_status(self, confluence_instance = None) -> Dict[str, Any]:
        """
        Get detailed workflow status with step completion, file checks, and timestamps.
        
        Args:
            confluence_instance: CONFLUENCE system instance
            
        Returns:
            Dictionary with detailed status information
        """
        from datetime import datetime
        import os
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'steps': {},
            'summary': {
                'total_steps': 0,
                'completed': 0,
                'failed': 0,
                'pending': 0
            },
            'config_validation': {},
            'system_info': {}
        }
        
        if not confluence_instance:
            return status
        
        try:
            # Get workflow definition
            workflow_steps = confluence_instance.workflow_orchestrator.define_workflow_steps()
            status['summary']['total_steps'] = len(workflow_steps)
            
            project_dir = confluence_instance.workflow_orchestrator.project_dir
            
            # Check each workflow step
            for i, (step_func, check_func, description) in enumerate(workflow_steps, 1):
                step_name = step_func.__name__
                
                step_status = {
                    'name': step_name,
                    'description': description,
                    'order': i,
                    'status': 'pending',
                    'files_exist': False,
                    'file_checks': {},
                    'timestamps': {},
                    'size_info': {}
                }
                
                # Check if step outputs exist
                try:
                    files_exist = check_func() if callable(check_func) else False
                    step_status['files_exist'] = files_exist
                    
                    if files_exist:
                        step_status['status'] = 'completed'
                        status['summary']['completed'] += 1
                    else:
                        step_status['status'] = 'pending'
                        status['summary']['pending'] += 1
                        
                except Exception as e:
                    step_status['status'] = 'failed'
                    step_status['error'] = str(e)
                    status['summary']['failed'] += 1
                
                # Get detailed file information for completed steps
                if step_status['files_exist']:
                    step_status.update(self._get_step_file_details(step_name, project_dir))
                
                status['steps'][step_name] = step_status
            
            # Config validation
            status['config_validation'] = self._validate_config_status(confluence_instance)
            
            # System info
            status['system_info'] = self._get_system_info()
            
        except Exception as e:
            status['error'] = f"Error getting workflow status: {str(e)}"
        
        return status

    def _get_step_file_details(self, step_name: str, project_dir: Path) -> Dict[str, Any]:
        """Get detailed file information for a workflow step."""
        file_details = {
            'file_checks': {},
            'timestamps': {},
            'size_info': {}
        }
        
        # Define expected output locations for each step
        step_outputs = {
            'setup_project': [project_dir / 'shapefiles'],
            'create_pour_point': [project_dir / 'shapefiles' / 'pour_point'],
            'acquire_attributes': [project_dir / 'attributes'],
            'define_domain': [project_dir / 'shapefiles' / 'catchment'],
            'discretize_domain': [project_dir / 'shapefiles' / 'river_basins'],
            'process_observed_data': [project_dir / 'observations' / 'streamflow' / 'preprocessed'],
            'acquire_forcings': [project_dir / 'forcing' / 'raw_data'],
            'run_model_agnostic_preprocessing': [project_dir / 'forcing' / 'basin_averaged_data'],
            'preprocess_models': [project_dir / 'forcing'],
            'run_models': [project_dir / 'simulations'],
            'calibrate_model': [project_dir / 'optimisation'],
            'postprocess_results': [project_dir / 'results']
        }
        
        expected_files = step_outputs.get(step_name, [])
        
        for file_path in expected_files:
            if file_path.exists():
                file_details['file_checks'][str(file_path)] = True
                
                # Get timestamp
                mtime = file_path.stat().st_mtime
                file_details['timestamps'][str(file_path)] = datetime.fromtimestamp(mtime).isoformat()
                
                # Get size info
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_details['size_info'][str(file_path)] = f"{size / (1024**2):.2f} MB"
                elif file_path.is_dir():
                    # Count files in directory
                    file_count = len(list(file_path.rglob('*')))
                    file_details['size_info'][str(file_path)] = f"{file_count} files"
            else:
                file_details['file_checks'][str(file_path)] = False
        
        return file_details

    def _validate_config_status(self, confluence_instance) -> Dict[str, Any]:
        """Validate configuration status."""
        config_status = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'key_settings': {}
        }
        
        try:
            config = confluence_instance.config
            
            # Check required settings
            required_settings = [
                'DOMAIN_NAME', 'EXPERIMENT_ID', 'CONFLUENCE_DATA_DIR', 
                'EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END'
            ]
            
            for setting in required_settings:
                if setting in config and config[setting]:
                    config_status['key_settings'][setting] = config[setting]
                else:
                    config_status['errors'].append(f"Missing required setting: {setting}")
                    config_status['valid'] = False
            
            # Check paths exist
            if 'CONFLUENCE_DATA_DIR' in config:
                data_dir = Path(config['CONFLUENCE_DATA_DIR'])
                if not data_dir.exists():
                    config_status['warnings'].append(f"Data directory does not exist: {data_dir}")
            
            # Check date formats
            for date_field in ['EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END']:
                if date_field in config:
                    try:
                        # Try to parse the date
                        datetime.fromisoformat(str(config[date_field]).replace(' ', 'T'))
                    except:
                        config_status['warnings'].append(f"Invalid date format in {date_field}")
            
        except Exception as e:
            config_status['valid'] = False
            config_status['errors'].append(f"Config validation error: {str(e)}")
        
        return config_status

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        system_info = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'working_directory': str(Path.cwd()),
            'user': os.getenv('USER', 'unknown')
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info['memory_usage'] = f"{memory.percent:.1f}%"
            system_info['available_memory_gb'] = f"{memory.available / (1024**3):.1f}"
        except ImportError:
            pass
        
        return system_info

    def print_detailed_workflow_status(self, status: Dict[str, Any]) -> None:
        """Print detailed workflow status in a user-friendly format."""
        print("\n📊 Detailed Workflow Status")
        print("=" * 60)
        
        # Summary
        summary = status['summary']
        total = summary['total_steps']
        completed = summary['completed']
        pending = summary['pending']
        failed = summary['failed']
        
        progress = (completed / total * 100) if total > 0 else 0
        
        print(f"🎯 Progress: {completed}/{total} steps ({progress:.1f}%)")
        print(f"✅ Completed: {completed}")
        print(f"⏳ Pending: {pending}")
        print(f"❌ Failed: {failed}")
        print(f"🕐 Status as of: {status['timestamp']}")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * completed // total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📈 [{bar}] {progress:.1f}%")
        
        # Step details
        print(f"\n📋 Step Details:")
        print("-" * 60)
        
        for step_name, step_info in status.get('steps', {}).items():
            order = step_info.get('order', 0)
            step_status = step_info.get('status', 'unknown')
            description = step_info.get('description', step_name)
            
            # Status icon
            if step_status == 'completed':
                icon = '✅'
            elif step_status == 'failed':
                icon = '❌'
            else:
                icon = '⏳'
            
            print(f"{order:2d}. {icon} {step_name}")
            print(f"    📝 {description}")
            print(f"    📊 Status: {step_status}")
            
            # File checks
            if step_info.get('files_exist'):
                print(f"    📁 Output files: Present")
                
                # Show timestamps for recent files
                timestamps = step_info.get('timestamps', {})
                if timestamps:
                    latest_file = max(timestamps.items(), key=lambda x: x[1])
                    print(f"    🕐 Latest: {latest_file[1]}")
            else:
                print(f"    📁 Output files: Missing")
            
            # Size info
            size_info = step_info.get('size_info', {})
            if size_info:
                for path, size in size_info.items():
                    file_name = Path(path).name
                    print(f"    📏 {file_name}: {size}")
            
            print()
        
        # Config validation
        config_status = status.get('config_validation', {})
        print(f"\n⚙️ Configuration Status:")
        if config_status.get('valid', False):
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has issues")
            
            errors = config_status.get('errors', [])
            warnings = config_status.get('warnings', [])
            
            for error in errors:
                print(f"   ❌ Error: {error}")
            for warning in warnings:
                print(f"   ⚠️  Warning: {warning}")
        
        # Key settings
        key_settings = config_status.get('key_settings', {})
        if key_settings:
            print("\n🔑 Key Settings:")
            for key, value in key_settings.items():
                print(f"   {key}: {value}")
        
        # System info
        system_info = status.get('system_info', {})
        if system_info:
            print(f"\n💻 System Information:")
            for key, value in system_info.items():
                print(f"   {key}: {value}")


    def list_templates(self) -> None:
        """List all available configuration templates."""
        print("\n📋 Available Configuration Templates:")
        print("=" * 50)
        
        # Look for templates in common locations
        template_locations = [
            Path("./0_config_files"),
            Path("../0_config_files"),
            Path("../../0_config_files"),
        ]
        
        templates_found = []
        
        for location in template_locations:
            if location.exists():
                # Look for template files
                for template_file in location.glob("config_*.yaml"):
                    if template_file.name not in [f.name for f in templates_found]:
                        templates_found.append(template_file)
                
                # Also look for the main template
                main_template = location / "config_template.yaml"
                if main_template.exists():
                    templates_found.insert(0, main_template)
        
        if not templates_found:
            print("❌ No templates found in standard locations")
            print("   Searched:", [str(loc) for loc in template_locations])
            return
        
        for i, template in enumerate(templates_found, 1):
            try:
                # Load template to get metadata
                with open(template, 'r') as f:
                    config = yaml.safe_load(f)
                
                domain_name = config.get('DOMAIN_NAME', 'Unknown')
                model_type = config.get('HYDROLOGICAL_MODEL', 'Unknown')
                description = self._get_template_description(template, config)
                
                print(f"{i}. 📄 {template.name}")
                print(f"   📍 Path: {template}")
                print(f"   🏞️  Domain: {domain_name}")
                print(f"   🔧 Model: {model_type}")
                print(f"   📝 Description: {description}")
                print()
                
            except Exception as e:
                print(f"{i}. ❌ {template.name} (Error reading: {str(e)})")
        
        print("💡 Usage:")
        print("   python CONFLUENCE.py --config path/to/template.yaml")
        print("   python CONFLUENCE.py --pour_point LAT/LON --domain_def METHOD --domain_name NAME")

    def _get_template_description(self, template_path: Path, config: Dict[str, Any]) -> str:
        """Extract description from template file or generate one."""
        # Check if there's a description in the config
        if 'DESCRIPTION' in config:
            return config['DESCRIPTION']
        
        # Generate description based on filename and contents
        filename = template_path.stem
        if 'template' in filename.lower():
            return "Base template for new projects"
        elif 'bow' in filename.lower():
            return "Bow River watershed configuration"
        elif 'example' in filename.lower():
            return "Example configuration"
        else:
            domain = config.get('DOMAIN_NAME', 'custom')
            return f"Configuration for {domain} domain"

    def update_config(self, config_file: str, updates: Dict[str, Any] = None) -> None:
        """
        Update an existing configuration file with new settings.
        
        Args:
            config_file: Path to configuration file to update
            updates: Dictionary of updates to apply
        """
        import yaml
        
        config_path = Path(config_file)
        
        print(f"\n🔧 Updating Configuration: {config_path}")
        print("=" * 50)
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return
        
        try:
            # Load existing config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"📄 Loaded configuration: {config_path}")
            print(f"🏞️  Current domain: {config.get('DOMAIN_NAME', 'Unknown')}")
            print(f"🧪 Current experiment: {config.get('EXPERIMENT_ID', 'Unknown')}")
            
            # Apply updates if provided
            if updates:
                for key, value in updates.items():
                    old_value = config.get(key, 'Not set')
                    config[key] = value
                    print(f"🔄 Updated {key}: {old_value} → {value}")
            
            # Interactive updates (if no updates provided)
            if not updates:
                print("\n💡 Interactive update mode. Press Enter to keep current values.")
                
                # Key settings to potentially update
                update_fields = [
                    ('DOMAIN_NAME', 'Domain name'),
                    ('EXPERIMENT_ID', 'Experiment ID'),
                    ('EXPERIMENT_TIME_START', 'Start time'),
                    ('EXPERIMENT_TIME_END', 'End time'),
                    ('HYDROLOGICAL_MODEL', 'Hydrological model'),
                    ('DOMAIN_DEFINITION_METHOD', 'Domain definition method')
                ]
                
                for field, description in update_fields:
                    current = config.get(field, '')
                    prompt = f"  {description} [{current}]: "
                    new_value = input(prompt).strip()
                    if new_value:
                        config[field] = new_value
                        print(f"    ✅ Updated: {new_value}")
            
            # Create backup
            backup_path = config_path.with_suffix('.yaml.backup')
            import shutil
            shutil.copy2(config_path, backup_path)
            print(f"💾 Backup created: {backup_path}")
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"✅ Configuration updated successfully!")
            print(f"📁 Updated file: {config_path}")
            
        except Exception as e:
            print(f"❌ Error updating configuration: {str(e)}")

    def validate_environment(self) -> None:
        """Validate system environment and dependencies."""
        print("\n🔍 Environment Validation:")
        print("=" * 40)
        
        # Check Python version
        import sys
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major >= 3 and python_version.minor >= 8:
            print("   ✅ Python version is compatible")
        else:
            print("   ❌ Python 3.8+ required")
        
        # Check required packages
        required_packages = [
            'numpy', 'pandas', 'geopandas', 'rasterio', 'shapely', 
            'yaml', 'pathlib', 'datetime', 'logging'
        ]
        
        print(f"\n📦 Package Dependencies:")
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        # Check CONFLUENCE structure
        print(f"\n📁 CONFLUENCE Structure:")
        
        required_dirs = [
            'utils', 'utils/project', 'utils/data', 'utils/geospatial',
            'utils/models', 'utils/evaluation', 'utils/optimization', 'utils/cli'
        ]
        
        for directory in required_dirs:
            dir_path = Path(directory)
            if dir_path.exists():
                print(f"   ✅ {directory}/")
            else:
                print(f"   ❌ {directory}/ (missing)")
        
        # Check config files
        print(f"\n⚙️  Configuration Files:")
        config_files = ['0_config_files/config_template.yaml']
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                print(f"   ✅ {config_file}")
            else:
                print(f"   ❌ {config_file} (missing)")
        
        # System resources
        print(f"\n💻 System Resources:")
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            print(f"   💾 RAM: {memory.available // (1024**3):.1f} GB available / {memory.total // (1024**3):.1f} GB total")
            print(f"   💿 Disk: {disk.free // (1024**3):.1f} GB available / {disk.total // (1024**3):.1f} GB total")
            
            if memory.available > 4 * (1024**3):  # 4GB
                print("   ✅ Sufficient memory available")
            else:
                print("   ⚠️  Low memory - consider closing other applications")
                
        except ImportError:
            print("   ⚠️  psutil not available - cannot check system resources")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        if missing_packages:
            print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
            print(f"   💡 Install with: pip install {' '.join(missing_packages)}")
        else:
            print("   ✅ All dependencies satisfied")
        
        print("   🚀 CONFLUENCE is ready to run!")

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
        
        # NEW: Config Management
        config_mgmt_group = self.parser.add_argument_group('Configuration Management')
        config_mgmt_group.add_argument(
            '--list_templates',
            action='store_true',
            help='List all available configuration templates'
        )
        config_mgmt_group.add_argument(
            '--update_config',
            type=str,
            metavar='CONFIG_FILE',
            help='Update an existing configuration file with new settings'
        )
        config_mgmt_group.add_argument(
            '--validate_environment',
            action='store_true',
            help='Validate system environment and dependencies'
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
        
        # NEW: Workflow Management
        workflow_mgmt_group = self.parser.add_argument_group('Workflow Management')
        workflow_mgmt_group.add_argument(
            '--workflow_status',
            action='store_true',
            help='Show detailed workflow status with step completion and file checks'
        )
        workflow_mgmt_group.add_argument(
            '--resume_from',
            type=str,
            metavar='STEP_NAME',
            help='Resume workflow execution from a specific step'
        )
        workflow_mgmt_group.add_argument(
            '--clean',
            action='store_true',
            help='Clean intermediate files and outputs'
        )
        workflow_mgmt_group.add_argument(
            '--clean_level',
            type=str,
            choices=['intermediate', 'outputs', 'all'],
            default='intermediate',
            help='Level of cleaning: intermediate files only, outputs, or all (default: intermediate)'
        )
        
        # Individual workflow steps (existing)
        steps_group = self.parser.add_argument_group('Individual Workflow Steps')
        for step_name, step_info in self.workflow_steps.items():
            steps_group.add_argument(
                f'--{step_name}',
                action='store_true',
                help=step_info['description']
            )
        
        # Pour point setup (existing)
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
            help='Domain name when using --pour_point (required)'
        )
        pourpoint_group.add_argument(
            '--experiment_id',
            type=str,
            help='Override experiment ID in configuration'
        )
        pourpoint_group.add_argument(
            '--bounding_box_coords',
            type=str,
            metavar='LAT_MAX/LON_MIN/LAT_MIN/LON_MAX',
            help='Bounding box coordinates (format: lat_max/lon_min/lat_min/lon_max, e.g., 51.76/-116.55/50.95/-115.5). Default: 1 degree buffer around pour point'
        )
        
        # Analysis and status options (updated)
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
# Basic workflow execution
python CONFLUENCE.py
python CONFLUENCE.py --config /path/to/config.yaml

# Individual workflow steps
python CONFLUENCE.py --calibrate_model
python CONFLUENCE.py --setup_project --create_pour_point --define_domain

# Pour point setup
python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "MyWatershed"
python CONFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "Test" --bounding_box_coords 52.0/-116.0/51.0/-115.0

# Configuration management
python CONFLUENCE.py --list_templates
python CONFLUENCE.py --update_config my_config.yaml
python CONFLUENCE.py --validate_environment

# Workflow management
python CONFLUENCE.py --workflow_status
python CONFLUENCE.py --resume_from define_domain
python CONFLUENCE.py --clean --clean_level intermediate
python CONFLUENCE.py --clean --clean_level all --dry_run

# Status and validation
python CONFLUENCE.py --status
python CONFLUENCE.py --list_steps
python CONFLUENCE.py --validate_config

# Advanced options
python CONFLUENCE.py --debug --force_rerun
python CONFLUENCE.py --dry_run
python CONFLUENCE.py --continue_on_error

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
    

    def _validate_bounding_box(self, bbox_string: str) -> bool:
        """
        Validate bounding box coordinate string format.
        
        Args:
            bbox_string: Bounding box string in format "lat_max/lon_min/lat_min/lon_max"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = bbox_string.split('/')
            if len(parts) != 4:
                return False
            
            lat_max, lon_min, lat_min, lon_max = map(float, parts)
            
            # Basic range and logic validation
            if not (-90 <= lat_min <= lat_max <= 90):
                return False
            if not (-180 <= lon_min <= lon_max <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
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
            'mode': 'workflow',  # 'workflow', 'individual_steps', 'pour_point_setup', 'status_only', 'management'
            'steps': [],
            'config_overrides': {},
            'settings': {
                'force_rerun': args.force_rerun,
                'stop_on_error': args.stop_on_error and not args.continue_on_error,
                'debug': args.debug,
                'dry_run': args.dry_run
            }
        }
        
        # Handle management operations (config and workflow management)
        if (args.list_templates or args.update_config or args.validate_environment or 
            args.workflow_status or args.resume_from or args.clean):
            plan['mode'] = 'management'
            plan['management_operations'] = {
                'list_templates': args.list_templates,
                'update_config': args.update_config,
                'validate_environment': args.validate_environment,
                'workflow_status': args.workflow_status,
                'resume_from': args.resume_from,
                'clean': args.clean,
                'clean_level': getattr(args, 'clean_level', 'intermediate')
            }
            return plan
        
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
                'domain_definition_method': args.domain_def,
                'domain_name': args.domain_name,
                'bounding_box_coords': getattr(args, 'bounding_box_coords', None)
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
            
            # Add bounding box to overrides if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords:
                plan['config_overrides']['BOUNDING_BOX_COORDS'] = args.bounding_box_coords
        
        if args.force_rerun:
            plan['config_overrides']['FORCE_RUN_ALL_STEPS'] = True
        
        return plan

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
                
            if not args.domain_name:
                errors.append("--domain_name is required when using --pour_point")
            
            # Validate bounding box if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords and not self._validate_bounding_box(args.bounding_box_coords):
                errors.append(f"Invalid bounding box format: {args.bounding_box_coords}. Expected format: lat_max/lon_min/lat_min/lon_max")
        
        # Validate resume_from step name
        if args.resume_from:
            if args.resume_from not in self.workflow_steps:
                errors.append(f"Invalid step name for --resume_from: {args.resume_from}. Available steps: {', '.join(self.workflow_steps.keys())}")
        
        # Validate update_config file exists
        if args.update_config:
            config_path = Path(args.update_config)
            if not config_path.exists():
                errors.append(f"Configuration file not found for --update_config: {config_path}")
        
        # Check conflicting options
        if args.stop_on_error and args.continue_on_error:
            errors.append("Cannot specify both --stop_on_error and --continue_on_error")
        
        # Check configuration file exists (only if not doing pour point setup or management operations)
        management_ops = any([
            args.list_templates, args.update_config, args.validate_environment,
            args.workflow_status, args.resume_from, args.clean
        ])
        
        if not args.pour_point and not management_ops:
            config_path = Path(args.config)
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_path}")
        
        return len(errors) == 0, errors
    
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
        
    def setup_pour_point_workflow(self, coordinates: str, domain_def_method: str, domain_name: str, 
                                bounding_box_coords: Optional[str] = None, 
                                confluence_code_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a configuration setup for pour point workflow.
        
        This method:
        1. Loads the config template from CONFLUENCE_CODE_DIR/0_config_files/config_template.yaml
        2. Updates key settings (pour point, domain name, domain definition method, bounding box)
        3. Saves as config_{domain_name}.yaml
        4. Returns configuration details for workflow execution
        
        Args:
            coordinates: Pour point coordinates in "lat/lon" format
            domain_def_method: Domain definition method to use
            domain_name: Name for the domain/watershed
            bounding_box_coords: Optional bounding box, defaults to 1-degree buffer around pour point
            confluence_code_dir: Path to CONFLUENCE code directory
            
        Returns:
            Dictionary with pour point workflow configuration including 'config_file' path
        """
        import yaml
        from pathlib import Path
        
        try:
            print(f"\n🎯 Setting up pour point workflow:")
            print(f"   📍 Coordinates: {coordinates}")
            print(f"   🗺️  Domain Definition Method: {domain_def_method}")
            print(f"   🏞️  Domain Name: {domain_name}")
            
            # Parse coordinates for bounding box calculation
            lat, lon = map(float, coordinates.split('/'))
            
            # Calculate bounding box if not provided (1-degree buffer)
            if not bounding_box_coords:
                lat_max = lat + 0.5
                lat_min = lat - 0.5
                lon_max = lon + 0.5
                lon_min = lon - 0.5
                bounding_box_coords = f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"
                print(f"   📦 Auto-calculated bounding box (1° buffer): {bounding_box_coords}")
            else:
                print(f"   📦 User-provided bounding box: {bounding_box_coords}")
            
            # Determine template path
            template_path = None
            
            # Try multiple locations for the template
            possible_locations = [
                Path("./0_config_files/config_template.yaml"),
                Path("../0_config_files/config_template.yaml"), 
                Path("../../0_config_files/config_template.yaml"),
            ]
            
            # If confluence_code_dir provided, try that first
            if confluence_code_dir:
                possible_locations.insert(0, Path(confluence_code_dir) / "0_config_files" / "config_template.yaml")
            
            # Try to detect CONFLUENCE_CODE_DIR from environment or relative paths
            for location in possible_locations:
                if location.exists():
                    template_path = location
                    break
            
            if not template_path:
                raise FileNotFoundError(
                    f"Config template not found. Tried locations: {[str(p) for p in possible_locations]}\n"
                    f"Please ensure you're running from the CONFLUENCE directory or specify --config with a template path."
                )
            
            print(f"   📄 Loading template from: {template_path}")
            
            # Load template config
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update config with pour point settings
            config['DOMAIN_NAME'] = domain_name
            config['POUR_POINT_COORDS'] = coordinates
            config['DOMAIN_DEFINITION_METHOD'] = domain_def_method
            config['BOUNDING_BOX_COORDS'] = bounding_box_coords
            
            # Set some sensible defaults for pour point workflows
            if 'EXPERIMENT_ID' not in config or config['EXPERIMENT_ID'] == 'run_1':
                config['EXPERIMENT_ID'] = 'pour_point_setup'
            
            # Save new config file
            output_config_path = Path(f"0_config_files/config_{domain_name}.yaml")
            with open(output_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"   💾 Created config file: {output_config_path}")
            print(f"   ✅ Pour point workflow setup complete!")
            print(f"\n🚀 Next steps:")
            print(f"   1. Review the generated config file: {output_config_path}")
            print(f"   2. CONFLUENCE will now use this config to run the pour point workflow")
            print(f"   3. Essential steps (setup_project, create_pour_point, define_domain, discretize_domain) will be executed")
            
            return {
                'config_file': str(output_config_path.resolve()),  # Return absolute path
                'coordinates': coordinates,
                'domain_name': domain_name,
                'domain_definition_method': domain_def_method,
                'bounding_box_coords': bounding_box_coords,
                'template_used': str(template_path),
                'setup_steps': [
                    'setup_project',
                    'create_pour_point', 
                    'define_domain',
                    'discretize_domain'
                ]
            }
            
        except Exception as e:
            print(f"❌ Error setting up pour point workflow: {str(e)}")
            raise

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