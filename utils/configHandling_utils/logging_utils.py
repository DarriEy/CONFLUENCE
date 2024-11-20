import logging
from pathlib import Path
from functools import wraps
import inspect
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_function_logger(func):
    """Decorator to create and return a logger for a specific function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if work_log_dir is provided in kwargs
        work_log_dir = kwargs.pop('work_log_dir', None)
        
        if work_log_dir is None:
            # Fallback to default behavior if work_log_dir is not provided
            if args and hasattr(args[0], 'data_dir'):
                data_dir = args[0].data_dir
            else:
                data_dir = Path('/path/to/CONFLUENCE_data')
            
            domain_name = args[0].domain_name if args and hasattr(args[0], 'domain_name') else 'default_domain'
            work_log_dir = data_dir / f"domain_{domain_name}" / f"_workLog_{domain_name}"

        # Ensure the output directory exists
        work_log_dir = Path(work_log_dir)
        work_log_dir.mkdir(parents=True, exist_ok=True)

        # Setup the logger for this function
        log_file = work_log_dir / f"{func.__name__}.log"
        logger = setup_logger(func.__name__, log_file)

        # Log the function's source code
        source_code = inspect.getsource(func)
        logger.info(f"Function source code:\n{source_code}\n{'='*50}\n")

        # Add the logger to the function's globals so it can be accessed within the function
        func.__globals__['logger'] = logger

        return func(*args, **kwargs)
    return wrapper

def log_exception(logger, exc):
    """Function to log exceptions"""
    logger.exception(f"An exception occurred: {exc}")

def log_configuration(config_path: str | Path, log_dir: Path, domain_name: str):
    """Log configuration file with timestamp for version control.
    
    Args:
        config_path (str | Path): Path to original config file (as string or Path)
        log_dir (Path): Directory to save logged config
        domain_name (str): Name of the domain for file naming
    
    Returns:
        Path: Path to the logged config file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_log_file = log_dir / f"config_{domain_name}_{timestamp}.yaml"
    
    # Convert input path to Path object if it's a string
    config_path = Path(config_path)
    
    # Copy the config file with timestamp
    config_log_file.write_text(config_path.read_text())
            
    return config_log_file