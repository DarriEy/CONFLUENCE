# In utils/config/logging_manager.py

from pathlib import Path
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
import yaml


class LoggingManager:
    """
    Manages logging configuration and setup for the CONFLUENCE framework.
    
    The LoggingManager is responsible for establishing a structured and comprehensive
    logging system for CONFLUENCE operations. It creates and configures loggers that
    provide detailed tracking of workflow execution, error conditions, and system
    status. The logging system supports both console output for interactive feedback
    and file-based logging for detailed analysis and troubleshooting.
    
    Key responsibilities:
    - Setting up the main logger with appropriate handlers and formatters
    - Creating and managing module-specific loggers
    - Capturing and preserving the configuration used for each run
    - Creating run summaries with execution statistics
    - Providing archiving capabilities for log persistence
    
    The logging system uses a hierarchical approach with different levels of detail
    for console (less verbose) and file (more verbose) outputs. It also supports
    module-specific logging configurations to allow fine-grained control over 
    log verbosity.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary
        data_dir (Path): Path to the CONFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        log_dir (Path): Path to the logging directory
        logger (logging.Logger): Main logger instance
        config_log_file (Path): Path to the logged configuration file
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logging manager.
        
        This method sets up the logging directory structure and initializes the main
        logger for CONFLUENCE. It also captures and preserves the configuration
        used for the current run.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            
        Raises:
            KeyError: If essential configuration values are missing
            PermissionError: If log directories cannot be created due to permissions
            OSError: If other file system operations fail
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
        
        This method creates and configures the main logger for CONFLUENCE with
        two handlers:
        1. A file handler that captures detailed log information for debugging
           and records all log levels (DEBUG and above)
        2. A console handler that provides less verbose output for interactive
           feedback, showing only important messages (INFO and above)
        
        The log file is named with a timestamp to ensure uniqueness across runs.
        
        Args:
            log_level (Optional[str]): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                                      If None, uses the level from configuration
            
        Returns:
            logging.Logger: Configured logger instance ready for use
            
        Raises:
            ValueError: If an invalid log level is specified
            PermissionError: If log file cannot be created due to permissions
        """
        # Get log level from config or parameter
        if log_level is None:
            log_level = self.config.get('LOG_LEVEL', 'INFO')
        
        # Create timestamp for log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'confluence_general_{self.domain_name}_{current_time}.log'
        
        # Disable the root logger to prevent interference
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create main logger
        logger = logging.getLogger('confluence')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Enhanced console formatter for cleaner output
        simple_formatter = self.EnhancedFormatter(
            '%(asctime)s │ %(message)s',
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
    
    def format_step_header(self, step_num: int, total_steps: int, step_name: str) -> str:
        """
        Create a formatted header for workflow steps.
        
        Args:
            step_num: Current step number
            total_steps: Total number of steps
            step_name: Name of the current step
            
        Returns:
            Formatted header string
        """
        progress = f"[{step_num:2d}/{total_steps:2d}]"
        separator = "─" * 50
        return f"\n┌{separator}┐\n│ {progress} {step_name:<40} │\n└{separator}┘"
    
    def format_step_completion(self, step_name: str, duration: str, status: str = "✓") -> str:
        """
        Create a formatted completion message for workflow steps.
        
        Args:
            step_name: Name of the completed step
            duration: Time taken for the step
            status: Status symbol (✓, ✗, →)
            
        Returns:
            Formatted completion message
        """
        return f"{status} {step_name} │ {duration}"
    
    def format_section_header(self, section_name: str) -> str:
        """
        Create a formatted section header.
        
        Args:
            section_name: Name of the section
            
        Returns:
            Formatted section header
        """
        separator = "═" * 60
        return f"\n{separator}\n║ {section_name:<56} ║\n{separator}"
    
    def log_configuration(self) -> Path:
        """
        Log the configuration file to the log directory.
        
        This method preserves the configuration used for the current run by
        saving it to a file in the log directory. This is critical for
        reproducibility and debugging, as it captures the exact settings
        used for each experiment.
        
        The configuration can be saved in either YAML or JSON format, depending
        on the CONFIG_FORMAT setting. Additionally, a 'latest' symlink is created
        for easy access to the most recent configuration.
        
        Returns:
            Path: Path to the logged configuration file
            
        Raises:
            PermissionError: If the configuration file cannot be written
            OSError: If the symlink cannot be created
            Exception: For other errors during configuration logging
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
        Set up a logger for a specific module with dedicated log file.
        
        This method creates a module-specific logger that allows for fine-grained
        control over logging verbosity on a per-module basis. Each module logger
        writes to its own log file while also propagating messages to the main
        logger for consolidated logging.
        
        Module-specific log levels can be specified in the configuration using
        the pattern MODULE_NAME_LOG_LEVEL (e.g., DOMAIN_LOG_LEVEL). If not specified,
        the method falls back to the global log level.
        
        Args:
            module_name (str): Name of the module (e.g., 'domain', 'data', 'model')
            log_level (Optional[str]): Logging level for this module
                                      If None, uses module-specific or global level
            
        Returns:
            logging.Logger: Configured logger for the module
            
        Raises:
            ValueError: If an invalid log level is specified
            PermissionError: If log file cannot be created due to permissions
        """
        # Create module-specific log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'{module_name}_{self.domain_name}_{current_time}.log'
        
        # Get log level
        if log_level is None:
            log_level = self.config.get(f'{module_name.upper()}_LOG_LEVEL', 
                                      self.config.get('LOG_LEVEL', 'INFO'))
        
        # Create module logger as child of main logger
        logger_name = f'confluence.{module_name}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for module-specific logs
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(file_handler)
        
        # Propagate to main confluence logger (not root)
        logger.propagate = True
        
        return logger
    
    def create_run_summary(self, start_time: datetime, end_time: datetime, 
                          status: Dict[str, Any]) -> Path:
        """
        Create a summary file for the run with execution statistics.
        
        This method generates a JSON file containing a summary of the workflow
        execution, including timing information, configuration settings, and
        execution status. This summary is valuable for tracking experiment
        history, performance analysis, and troubleshooting.
        
        The summary includes:
        - Domain and experiment identifiers
        - Start and end times
        - Execution duration
        - Workflow status (completion of steps)
        - Key configuration settings
        
        Args:
            start_time (datetime): Workflow start time
            end_time (datetime): Workflow end time
            status (Dict[str, Any]): Workflow status dictionary with execution details
            
        Returns:
            Path: Path to the created summary file
            
        Raises:
            PermissionError: If the summary file cannot be written
            TypeError: If the status dictionary contains non-serializable objects
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
        Archive all logs for the current run into a compressed file.
        
        This method collects all log files for the current run and compresses
        them into a ZIP archive. This is useful for preserving logs long-term,
        sharing logs for troubleshooting, or cleaning up log directories while
        retaining the information.
        
        If no archive name is provided, one is generated using the domain name
        and current timestamp.
        
        Args:
            archive_name (Optional[str]): Name for the archive file
                                         If None, a name is generated automatically
            
        Returns:
            Path: Path to the created archive file
            
        Raises:
            PermissionError: If the archive cannot be created
            ImportError: If the shutil module is not available
            Exception: For other errors during archiving
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
        Get the path to a specific log file type.
        
        This method locates and returns the path to a specific type of log file,
        such as the general log, configuration log, or module-specific logs.
        For types other than 'config', it returns the most recent log file
        of that type.
        
        Args:
            log_type (str): Type of log file to retrieve:
                          - 'general': Main CONFLUENCE log
                          - 'config': Configuration log
                          - Any module name: Module-specific log
            
        Returns:
            Optional[Path]: Path to the requested log file if it exists, None otherwise
            
        Raises:
            ValueError: If an invalid log_type is specified
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
    
    class EnhancedFormatter(logging.Formatter):
        """
        Enhanced formatter for console output with better visual formatting.
        
        This formatter provides clean, visually appealing output for different
        types of log messages, including special formatting for step headers,
        completions, and progress indicators.
        """
        
        # ANSI color codes for different log levels
        COLORS = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            """Format log record with enhanced visual styling."""
            # Get the basic formatted message
            formatted = super().format(record)
            
            # Add color coding for log levels (if terminal supports it)
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                level_color = self.COLORS.get(record.levelname, '')
                if level_color:
                    # Color the level indicator
                    level_indicator = f"{level_color}●{self.RESET}"
                else:
                    level_indicator = "●"
            else:
                level_indicator = "●"
            
            # Special formatting for different message types
            message = record.getMessage()
            
            # Step headers (messages starting with step numbers or containing "Step")
            if "Step " in message and "/" in message:
                return f"\n{formatted}"
            
            # Completion messages (messages with checkmarks or duration info)
            elif any(symbol in message for symbol in ["✓", "✗", "→", "Duration:"]):
                return f"{formatted}"
            
            # Section headers (messages with lots of = or -)
            elif message.count("=") > 10 or message.count("─") > 10:
                return f"\n{formatted}"
            
            # Regular messages
            else:
                return f"{record.asctime} {level_indicator} {message}"
    
    class ConsoleFilter(logging.Filter):
        """
        Filter to reduce console output verbosity by excluding noisy debug messages.
        
        This filter prevents debug-level messages from certain verbose modules
        (e.g., matplotlib, urllib3, requests) from appearing in the console output.
        This helps maintain a cleaner console interface while still capturing
        all messages in the log files.
        
        The filter is applied only to the console handler, not to file handlers,
        ensuring that complete logging information is still preserved.
        """
        
        def filter(self, record):
            """
            Filter log records based on level and module.
            
            This method determines whether a log record should be displayed in
            the console output. It filters out DEBUG-level messages from modules
            known to be verbose.
            
            Args:
                record: Log record to be evaluated
                
            Returns:
                bool: True if the record should be displayed, False otherwise
            """
            # Filter out debug messages from specific modules
            if record.levelno <= logging.DEBUG:
                if record.module in ['matplotlib', 'urllib3', 'requests']:
                    return False
            return True