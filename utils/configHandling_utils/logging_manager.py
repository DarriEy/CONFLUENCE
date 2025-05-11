# In utils/configHandling_utils/logging_manager.py

from pathlib import Path
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
import yaml


class LoggingManager:
    """Manages logging configuration and setup for CONFLUENCE."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logging manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.log_dir = self.project_dir / f"_workLog_{self.domain_name}"
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up main logger
        self.logger = self.setup_logging()
        
        # Log configuration
        self.config_log_file = self.log_configuration()
    
    def setup_logging(self, log_level: Optional[str] = None) -> logging.Logger:
        """
        Set up logging configuration with console and file handlers.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Configured logger instance
        """
        # Get log level from config or parameter
        if log_level is None:
            log_level = self.config.get('LOG_LEVEL', 'INFO')
        
        # Create timestamp for log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'confluence_general_{self.domain_name}_{current_time}.log'
        
        # Create logger
        logger = logging.getLogger('confluence_general')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler with simple formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Only INFO and above to console
        console_handler.setFormatter(simple_formatter)
        
        # Add custom filter for console to reduce verbosity
        console_handler.addFilter(self.ConsoleFilter())
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log startup information
        logger.info("=" * 60)
        logger.info(f"CONFLUENCE Logging Initialized")
        logger.info(f"Domain: {self.domain_name}")
        logger.info(f"Experiment ID: {self.config.get('EXPERIMENT_ID', 'N/A')}")
        logger.info(f"Log Level: {log_level}")
        logger.info(f"Log File: {log_file}")
        logger.info("=" * 60)
        
        return logger
    
    def log_configuration(self) -> Path:
        """
        Log the configuration file to the log directory.
        
        Returns:
            Path to the logged configuration file
        """
        # Create timestamp for config log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine config format and extension
        config_format = self.config.get('CONFIG_FORMAT', 'yaml')
        extension = 'yaml' if config_format == 'yaml' else 'json'
        
        # Create config log file path
        config_log_file = self.log_dir / f'config_{self.domain_name}_{timestamp}.{extension}'
        
        try:
            # Log configuration in appropriate format
            with open(config_log_file, 'w') as f:
                if config_format == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(self.config, f, indent=4)
            
            self.logger.info(f"Configuration logged to: {config_log_file}")
            
            # Also create a 'latest' symlink for easy access
            latest_link = self.log_dir / f'config_latest.{extension}'
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(config_log_file.name)
            
        except Exception as e:
            self.logger.error(f"Failed to log configuration: {str(e)}")
            raise
        
        return config_log_file
    
    def setup_module_logger(self, module_name: str, log_level: Optional[str] = None) -> logging.Logger:
        """
        Set up a logger for a specific module.
        
        Args:
            module_name: Name of the module
            log_level: Logging level for this module
            
        Returns:
            Configured logger for the module
        """
        # Create module-specific log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'{module_name}_{self.domain_name}_{current_time}.log'
        
        # Get log level
        if log_level is None:
            log_level = self.config.get(f'{module_name.upper()}_LOG_LEVEL', 
                                      self.config.get('LOG_LEVEL', 'INFO'))
        
        # Create logger
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(file_handler)
        
        # Also log to main log file
        logger.propagate = True
        
        return logger
    
    def create_run_summary(self, start_time: datetime, end_time: datetime, 
                          status: Dict[str, Any]) -> Path:
        """
        Create a summary file for the run.
        
        Args:
            start_time: Workflow start time
            end_time: Workflow end time
            status: Workflow status dictionary
            
        Returns:
            Path to the summary file
        """
        summary_file = self.log_dir / f'run_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        summary = {
            'domain_name': self.domain_name,
            'experiment_id': self.config.get('EXPERIMENT_ID'),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': str(end_time - start_time),
            'status': status,
            'configuration': {
                'hydrological_model': self.config.get('HYDROLOGICAL_MODEL'),
                'domain_definition_method': self.config.get('DOMAIN_DEFINITION_METHOD'),
                'optimization_algorithm': self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM'),
                'force_run_all': self.config.get('FORCE_RUN_ALL_STEPS', False)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.logger.info(f"Run summary created: {summary_file}")
        return summary_file
    
    def archive_logs(self, archive_name: Optional[str] = None) -> Path:
        """
        Archive all logs for the current run.
        
        Args:
            archive_name: Name for the archive file
            
        Returns:
            Path to the created archive
        """
        import shutil
        
        if archive_name is None:
            archive_name = f'logs_{self.domain_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        archive_path = self.log_dir / 'archives'
        archive_path.mkdir(exist_ok=True)
        
        # Create archive
        archive_file = shutil.make_archive(
            str(archive_path / archive_name),
            'zip',
            self.log_dir,
            '.'
        )
        
        self.logger.info(f"Logs archived to: {archive_file}")
        return Path(archive_file)
    
    def get_log_file_path(self, log_type: str = 'general') -> Optional[Path]:
        """
        Get the path to a specific log file.
        
        Args:
            log_type: Type of log file ('general', 'config', etc.)
            
        Returns:
            Path to the log file if it exists
        """
        if log_type == 'general':
            # Find the most recent general log
            log_files = sorted(self.log_dir.glob(f'confluence_general_{self.domain_name}_*.log'))
            return log_files[-1] if log_files else None
        elif log_type == 'config':
            return self.config_log_file
        else:
            # Find module-specific log
            log_files = sorted(self.log_dir.glob(f'{log_type}_{self.domain_name}_*.log'))
            return log_files[-1] if log_files else None
    
    class ConsoleFilter(logging.Filter):
        """Filter to reduce console output verbosity."""
        
        def filter(self, record):
            """Filter out verbose debug messages from console."""
            # Filter out debug messages from specific modules
            if record.levelno <= logging.DEBUG:
                if record.module in ['matplotlib', 'urllib3', 'requests']:
                    return False
            return True