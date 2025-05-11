#!/usr/bin/env python3
"""
CONFLUENCE - Community Optimization and Numerical Framework for Large-domain 
Understanding of Environmental Networks and Computational Exploration

This is the main entry point for the CONFLUENCE hydrological modeling platform.
CONFLUENCE provides an integrated framework for:
- Hydrological model setup and configuration
- Multi-model comparison and evaluation
- Parameter optimization and calibration
- Uncertainty quantification
- Model emulation and machine learning integration
- Comprehensive workflow management

Usage:
    python CONFLUENCE.py [--config CONFIG_PATH]
    
    If no config path is provided, defaults to './0_config_files/config_active.yaml'

Example:
    python CONFLUENCE.py --config /path/to/my_config.yaml

For more information, see the documentation at:
    https://github.com/your-org/CONFLUENCE

Author: CONFLUENCE Development Team
License: MIT License
Version: 1.0.0
"""

from pathlib import Path
import sys
import argparse
from datetime import datetime
from typing import Optional

# Add the parent directory to the path to enable imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import CONFLUENCE components
from utils.project.project_manager import ProjectManager # type: ignore
from utils.project.workflow_orchestrator import WorkflowOrchestrator # type: ignore
from utils.config.config_utils import ConfigManager # type: ignore
from utils.config.logging_manager import LoggingManager # type: ignore
from utils.dataHandling_utils.data_manager import DataManager # type: ignore
from utils.geospatial.domain_manager import DomainManager # type: ignore
from utils.models.model_manager import ModelManager # type: ignore
from utils.evaluation.analysis_manager import AnalysisManager # type: ignore
from utils.optimization.optimization_manager import OptimizationManager # type: ignore


class CONFLUENCE:
    """
    CONFLUENCE main class that orchestrates the hydrological modeling workflow.
    
    This class serves as the central coordinator for all CONFLUENCE operations,
    initializing the various manager components and orchestrating the workflow
    execution. It follows a modular architecture where each major functionality
    is handled by a dedicated manager class.
    
    Architecture:
        - ConfigManager: Handles configuration loading and validation
        - LoggingManager: Manages all logging operations
        - ProjectManager: Manages project structure and initialization
        - DomainManager: Handles spatial domain definition and discretization
        - DataManager: Manages data acquisition and preprocessing
        - ModelManager: Handles model-specific operations
        - AnalysisManager: Manages analysis and evaluation tasks
        - OptimizationManager: Handles calibration and emulation
        - WorkflowOrchestrator: Coordinates the execution workflow
    
    Attributes:
        config_manager (ConfigManager): Configuration management instance
        config (dict): Loaded configuration dictionary
        logging_manager (LoggingManager): Logging management instance
        logger (logging.Logger): Main logger instance
        managers (dict): Dictionary of all manager instances
        workflow_orchestrator (WorkflowOrchestrator): Workflow coordination instance
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize the CONFLUENCE system with a configuration file.
        
        This method sets up all the necessary components for CONFLUENCE operation:
        1. Loads and validates the configuration
        2. Initializes the logging system
        3. Creates all manager instances
        4. Sets up the workflow orchestrator
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
            RuntimeError: If initialization of any component fails
        """
        try:
            # Initialize configuration management
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
            
            # Initialize logging system
            self.logging_manager = LoggingManager(self.config)
            self.logger = self.logging_manager.logger
            
            self.logger.info("Initializing CONFLUENCE system")
            self.logger.info(f"Configuration loaded from: {config_path}")
            
            # Initialize all manager components
            self.managers = self._initialize_managers()
            
            # Initialize workflow orchestrator
            self.workflow_orchestrator = WorkflowOrchestrator(
                self.managers, 
                self.config, 
                self.logger
            )
            
            # Validate system initialization
            if not self.workflow_orchestrator.validate_workflow_prerequisites():
                raise RuntimeError("Workflow prerequisites validation failed")
            
            self.logger.info("CONFLUENCE system initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize CONFLUENCE: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise
    
    def _initialize_managers(self) -> dict:
        """
        Initialize all manager components.
        
        This method creates instances of all manager classes that handle
        different aspects of the CONFLUENCE workflow.
        
        Returns:
            Dictionary mapping manager names to their instances
            
        Raises:
            RuntimeError: If any manager fails to initialize
        """
        try:
            return {
                'project': ProjectManager(self.config, self.logger),
                'domain': DomainManager(self.config, self.logger),
                'data': DataManager(self.config, self.logger),
                'model': ModelManager(self.config, self.logger),
                'analysis': AnalysisManager(self.config, self.logger),
                'optimization': OptimizationManager(self.config, self.logger)
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize managers: {str(e)}")
            raise RuntimeError(f"Manager initialization failed: {str(e)}")
    
    def run_workflow(self) -> None:
        """
        Execute the complete CONFLUENCE workflow.
        
        This method delegates to the workflow orchestrator to run the complete
        modeling workflow as defined in the configuration. The workflow includes:
        - Project setup
        - Domain definition and discretization
        - Data acquisition and preprocessing
        - Model execution
        - Analysis and optimization
        - Results postprocessing
        
        The workflow execution is logged and can be monitored through the
        log files created in the project directory.
        
        Raises:
            RuntimeError: If the workflow execution fails
        """
        try:
            start_time = datetime.now()
            self.logger.info("Starting CONFLUENCE workflow execution")
            
            # Execute the workflow
            self.workflow_orchestrator.run_workflow()
            
            # Create run summary
            end_time = datetime.now()
            status = self.workflow_orchestrator.get_workflow_status()
            self.logging_manager.create_run_summary(start_time, end_time, status)
            
            self.logger.info("CONFLUENCE workflow execution completed")
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise RuntimeError(f"Workflow execution error: {str(e)}")
    
    def get_status(self) -> dict:
        """
        Get the current status of the CONFLUENCE system.
        
        Returns:
            Dictionary containing status information for all components
        """
        return {
            'config_valid': bool(self.config),
            'managers_initialized': all(self.managers.values()),
            'workflow_status': self.workflow_orchestrator.get_workflow_status(),
            'domain': self.config.get('DOMAIN_NAME'),
            'experiment': self.config.get('EXPERIMENT_ID')
        }

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='CONFLUENCE - Hydrological Modeling Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python CONFLUENCE.py
  
  # Run with custom configuration
  python CONFLUENCE.py --config /path/to/config.yaml
  
  # Show version information
  python CONFLUENCE.py --version
  
For more information, visit: https://github.com/your-org/CONFLUENCE
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to YAML configuration file (default: ./0_config_files/config_active.yaml)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CONFLUENCE 1.0.0'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for CONFLUENCE application.
    
    This function:
    1. Parses command line arguments
    2. Determines the configuration file path
    3. Validates the configuration file exists
    4. Initializes and runs the CONFLUENCE system
    5. Handles any errors that occur
    
    Exit codes:
        0: Success
        1: Configuration file not found
        2: Runtime error during execution
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Determine configuration file path
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = Path(__file__).parent / '0_config_files' / 'config_active.yaml'
        
        # Validate configuration file exists
        if not config_path.exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Initialize and run CONFLUENCE
        print(f"Starting CONFLUENCE with configuration: {config_path}")
        confluence = CONFLUENCE(config_path)
        confluence.run_workflow()
        
        print("CONFLUENCE completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"Error running CONFLUENCE: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()