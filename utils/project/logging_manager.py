# In utils/project/logging_manager.py

from pathlib import Path
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import yaml
import warnings
import os

# Suppress pyogrio field width warnings (non-fatal - data is still written)
warnings.filterwarnings('ignore', 
                       message='.*not successfully written.*field width.*',
                       category=RuntimeWarning,
                       module='pyogrio.raw')

class LoggingManager:
    """
    Manages logging configuration and setup for the SYMFLUENCE framework.
    
    The LoggingManager is responsible for establishing a structured and comprehensive
    logging system for SYMFLUENCE operations. It creates and configures loggers that
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
        debug_mode (bool): Whether debug mode is enabled
        data_dir (Path): Path to the SYMFLUENCE data directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the project directory
        log_dir (Path): Path to the logging directory
        logger (logging.Logger): Main logger instance
        config_log_file (Path): Path to the logged configuration file
    """
    
    def __init__(self, config: Dict[str, Any], debug_mode: bool = False):
        """
        Initialize the logging manager.
        
        This method sets up the logging directory structure and initializes the main
        logger for SYMFLUENCE. It also captures and preserves the configuration
        used for the current run.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing all settings
            debug_mode (bool): Whether to enable debug mode for detailed console output
            
        Raises:
            KeyError: If essential configuration values are missing
            PermissionError: If log directories cannot be created due to permissions
            OSError: If other file system operations fail
        """
        self.config = config
        self.debug_mode = debug_mode
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.log_dir = self.project_dir / f"_workLog_{self.domain_name}"
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up main logger with debug mode consideration
        self.logger = self.setup_logging()
        
        # Log debug mode status
        if self.debug_mode:
            self.logger.debug("DEBUG MODE ENABLED - Detailed console output active")
        
        # Log configuration
        self.config_log_file = self.log_configuration()
    
    def setup_logging(self, log_level: Optional[str] = None) -> logging.Logger:
        """
        Set up logging configuration with console and file handlers.
        
        This method creates and configures the main logger for SYMFLUENCE with
        two handlers:
        1. A file handler that captures detailed log information for debugging
           and records all log levels (DEBUG and above)
        2. A console handler that provides output based on debug mode:
           - Normal mode: INFO and above (clean output)
           - Debug mode: DEBUG and above (detailed output)
        
        The log file is named with a timestamp to ensure uniqueness across runs.
        
        Args:
            log_level (Optional[str]): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
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
        log_file = self.log_dir / f'symfluence_general_{self.domain_name}_{current_time}.log'
        
        # Disable the root logger to prevent interference
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Create main logger
        logger = logging.getLogger('symfluence')
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
        
        # Console formatter - different based on debug mode
        if self.debug_mode:
            # Detailed console formatter for debug mode
            console_formatter = self.EnhancedFormatter(
                '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            # Clean console formatter for normal mode
            console_formatter = self.EnhancedFormatter(
                '%(asctime)s ● %(message)s',
                datefmt='%H:%M:%S'
            )
        
        # File handler with detailed formatting (always logs everything)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler - level depends on debug mode
        console_handler = logging.StreamHandler(sys.stdout)
        if self.debug_mode:
            console_handler.setLevel(logging.DEBUG)  # Show debug messages in debug mode
        else:
            console_handler.setLevel(logging.INFO)   # Only INFO and above in normal mode
        console_handler.setFormatter(console_formatter)
        
        # Add custom filter for console to reduce verbosity (only in normal mode)
        if not self.debug_mode:
            console_handler.addFilter(self.ConsoleFilter())
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log startup information
        logger.info("=" * 60)
        logger.info("SYMFLUENCE Logging Initialized")
        if self.debug_mode:
            logger.info("DEBUG MODE: Enabled")
        logger.info(f"Domain: {self.domain_name}")
        logger.info(f"Experiment ID: {self.config.get('EXPERIMENT_ID', 'N/A')}")
        logger.info(f"Log Level: {log_level}")
        logger.info(f"Log File: {log_file}")
        logger.info("=" * 60)
        
        return logger
    
    def log_configuration(self) -> Path:
        """
        Log the current configuration to a YAML file for reproducibility.
        
        This method captures the complete configuration used for the current run
        and saves it to a timestamped YAML file. This is essential for
        reproducibility and debugging, as it preserves the exact settings used
        for each model run.
        
        The configuration is saved with masked sensitive information (such as
        passwords or API keys, though none are currently expected in SYMFLUENCE).
        
        Returns:
            Path: Path to the saved configuration file
            
        Raises:
            PermissionError: If configuration file cannot be written
            IOError: If file writing operations fail
        """
        # Create timestamp for config file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_file = self.log_dir / f'config_{self.domain_name}_{current_time}.yaml'
        
        # Create a copy and mask sensitive information
        config_to_log = self.config.copy()
        
        # Mask any potential sensitive fields (add as needed)
        sensitive_fields = ['PASSWORD', 'API_KEY', 'SECRET']
        for field in sensitive_fields:
            if field in config_to_log:
                config_to_log[field] = '***MASKED***'
        
        # Add metadata
        metadata = {
            'logged_at': current_time,
            'symfluence_version': '1.0.0',  # Update with actual version
            'debug_mode': self.debug_mode,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        config_to_log['_metadata'] = metadata
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(config_to_log, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Configuration logged to: {config_file}")
        return config_file
    
    def create_module_logger(self, module_name: str, 
                           log_level: Optional[str] = None) -> logging.Logger:
        """
        Create a logger for a specific module with its own file handler.
        
        This method creates module-specific loggers that inherit from the main
        SYMFLUENCE logger but have their own log files. This allows for
        module-specific debugging and analysis without cluttering the main log.
        
        Module loggers use the same formatters as the main logger but can have
        different log levels if needed.
        
        Args:
            module_name (str): Name of the module requiring a logger
            log_level (Optional[str]): Logging level for this specific module.
                                      If None, inherits from main logger
            
        Returns:
            logging.Logger: Configured module-specific logger
            
        Raises:
            ValueError: If module_name is empty or contains invalid characters
            PermissionError: If module log file cannot be created
        """
        # Create module logger as child of main logger
        module_logger = logging.getLogger(f'symfluence.{module_name}')
        
        # Set level
        if log_level:
            module_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Don't add handlers if they already exist (avoid duplicates)
        if module_logger.handlers:
            return module_logger
        
        # Create timestamp for log file
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'{module_name}_{self.domain_name}_{current_time}.log'
        
        # Create file handler for module
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Use same detailed formatter as main logger
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to module logger
        module_logger.addHandler(file_handler)
        
        # Log creation
        module_logger.info(f"Module logger initialized for: {module_name}")
        
        return module_logger
    
    def log_step_header(self, step_number: int, total_steps: int, 
                       step_name: str, description: str = "") -> None:
        """
        Log a formatted header for a workflow step.
        
        This method creates visually distinct headers for workflow steps to improve
        log readability. It uses box drawing characters to create a clear visual
        separation between different workflow stages.
        
        Args:
            step_number (int): Current step number
            total_steps (int): Total number of steps in the workflow
            step_name (str): Name of the current step
            description (str): Optional detailed description of the step
            
        Raises:
            ValueError: If step_number > total_steps or if values are negative
        """
        # Create the step header with box drawing characters
        header = f"""
┌{'─' * 58}┐
│ Step {step_number}/{total_steps}: {step_name:<46} │
│ {description:<56} │
└{'─' * 58}┘"""
        
        # Log without timestamp (formatted message)
        for line in header.strip().split('\n'):
            self.logger.info(line)
    
    def log_substep(self, message: str, level: str = 'INFO') -> None:
        """
        Log a substep or progress message with consistent formatting.
        
        This method provides consistent formatting for substep messages within
        a larger workflow step. It uses arrow indicators to show progression
        and maintains visual consistency throughout the logs.
        
        Args:
            message (str): The message to log
            level (str): Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
            
        Raises:
            ValueError: If an invalid logging level is specified
        """
        formatted_message = f"  → {message}"
        log_method = getattr(self.logger, level.lower())
        log_method(formatted_message)
    
    def log_completion(self, success: bool = True, message: str = "", 
                      duration: Optional[float] = None) -> None:
        """
        Log the completion of a step or operation.
        
        This method logs the completion status of an operation with visual
        indicators for success or failure. It can also include execution time
        if provided.
        
        Args:
            success (bool): Whether the operation completed successfully
            message (str): Optional completion message
            duration (Optional[float]): Execution time in seconds
            
        Raises:
            ValueError: If duration is negative
        """
        status_symbol = "✓" if success else "✗"
        status_text = "Complete" if success else "Failed"
        
        completion_msg = f"  {status_symbol} {status_text}"
        if message:
            completion_msg += f": {message}"
        if duration is not None:
            completion_msg += f" (Duration: {duration:.2f}s)"
        
        if success:
            self.logger.info(completion_msg)
        else:
            self.logger.error(completion_msg)
    
    def create_run_summary(self, steps_completed: List[str], 
                          errors: List[Dict[str, Any]],
                          warnings: List[str],
                          execution_time: float,
                          status: str = 'completed') -> Path:
        """
        Create a summary JSON file for the entire run.
        
        This method creates a comprehensive summary of a SYMFLUENCE run, including
        all completed steps, any errors or warnings encountered, and overall
        execution statistics. The summary is saved as a JSON file for easy
        parsing and analysis.
        
        Args:
            steps_completed (List[str]): List of successfully completed step names
            errors (List[Dict[str, Any]]): List of error dictionaries with details
            warnings (List[str]): List of warning messages
            execution_time (float): Total execution time in seconds
            status (str): Overall run status ('completed', 'failed', 'partial')
            
        Returns:
            Path: Path to the created summary file
            
        Raises:
            IOError: If summary file cannot be written
            ValueError: If execution_time is negative
        """
        summary_file = self.log_dir / f'run_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'domain': self.domain_name,
            'experiment_id': self.config.get('EXPERIMENT_ID', 'N/A'),
            'execution_time_seconds': execution_time,
            'steps_completed': steps_completed,
            'total_steps_completed': len(steps_completed),
            'errors': errors,
            'total_errors': len(errors),
            'warnings': warnings,
            'total_warnings': len(warnings),
            'debug_mode': self.debug_mode,
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
            archive_name (Optional[str]): Name for the archive file.
                                         If None, a name is generated automatically
            
        Returns:
            Path: Path to the created archive file
            
        Raises:
            PermissionError: If the archive cannot be created
            ImportError: If the shutil module is not available
            IOError: For other errors during archiving
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
                          - 'general': Main SYMFLUENCE log
                          - 'config': Configuration log
                          - Any module name: Module-specific log
            
        Returns:
            Optional[Path]: Path to the requested log file if it exists, None otherwise
            
        Raises:
            ValueError: If an invalid log_type is specified
        """
        if log_type == 'general':
            # Find the most recent general log
            log_files = sorted(self.log_dir.glob(f'symfluence_general_{self.domain_name}_*.log'))
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
        
        # ANSI color codes for different log levels (with cross-platform check)
        use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.name != 'nt'
        
        if use_colors:
            COLORS = {
                'DEBUG': '\033[36m',    # Cyan
                'INFO': '\033[32m',     # Green
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',    # Red
                'CRITICAL': '\033[35m', # Magenta
            }
            RESET = '\033[0m'
        else:
            COLORS = {}
            RESET = ''
        
        def format(self, record: logging.LogRecord) -> str:
            """
            Format log record with enhanced visual styling.
            
            Args:
                record (logging.LogRecord): The log record to format
                
            Returns:
                str: Formatted log message
            """
            # Get the basic formatted message
            message = record.getMessage()
            
            # Special formatting for different message types
            
            # Step headers (messages starting with "Step X/Y:" or containing step patterns)
            if (message.startswith("Step ") and "/" in message and ":" in message) or \
               (message.startswith("┌") or message.startswith("│") or message.startswith("└")):
                # Return step headers without timestamp or bullets
                return message
            
            # Section separators (messages with lots of = signs)
            elif message.count("=") > 10:
                # Return section separators without timestamp or bullets
                return message
            
            # Completion messages (messages with status symbols)
            elif any(symbol in message for symbol in ["✓", "✗", "→", "Duration:"]):
                # Format timestamp but use the original formatted string approach
                formatted = super().format(record)
                return formatted
            
            # Debug messages (only in debug mode)
            elif record.levelname == 'DEBUG':
                # Use the detailed debug format from super().format()
                formatted = super().format(record)
                return formatted
            
            # Regular info messages
            else:
                # Add color coding for log levels (if terminal supports it)
                if self.use_colors:
                    level_color = self.COLORS.get(record.levelname, '')
                    if level_color and record.levelname == 'INFO':
                        level_indicator = f"{level_color}●{self.RESET}"
                    else:
                        level_indicator = "●"
                else:
                    level_indicator = "●"
                
                # Format: "HH:MM:SS ● message"
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
        
        def filter(self, record: logging.LogRecord) -> bool:
            """
            Filter log records based on level and module.
            
            This method determines whether a log record should be displayed in
            the console output. It filters out DEBUG-level messages from modules
            known to be verbose.
            
            Args:
                record (logging.LogRecord): Log record to be evaluated
                
            Returns:
                bool: True if the record should be displayed, False otherwise
            """
            # Filter out debug messages from specific modules
            if record.levelno <= logging.DEBUG:
                if record.module in ['matplotlib', 'urllib3', 'requests']:
                    return False
            return True