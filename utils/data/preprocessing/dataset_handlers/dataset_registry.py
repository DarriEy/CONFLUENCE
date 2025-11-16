"""
Dataset Registry for SYMFLUENCE

This module provides a registry system for dataset handlers. It allows the main
preprocessor to dynamically load the appropriate dataset handler based on the
forcing dataset specified in the configuration.
"""

from typing import Dict, Type
from pathlib import Path


class DatasetRegistry:
    """
    Registry for dataset handlers.
    
    This class maintains a mapping of dataset names to their handler classes
    and provides a factory method to instantiate the correct handler.
    """
    
    _handlers: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, dataset_name: str):
        """
        Decorator to register a dataset handler.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'rdrs', 'era5')
            
        Returns:
            Decorator function
            
        Example:
            @DatasetRegistry.register('rdrs')
            class RDRSHandler(BaseDatasetHandler):
                ...
        """
        def decorator(handler_class):
            cls._handlers[dataset_name.lower()] = handler_class
            return handler_class
        return decorator
    
    @classmethod
    def get_handler(cls, dataset_name: str, config: Dict, logger, project_dir: Path):
        """
        Get an instance of the appropriate dataset handler.
        
        Args:
            dataset_name: Name of the dataset from config
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Path to project directory
            
        Returns:
            Instance of the appropriate dataset handler
            
        Raises:
            ValueError: If the dataset name is not registered
        """
        dataset_name_lower = dataset_name.lower()
        
        if dataset_name_lower not in cls._handlers:
            available = ', '.join(cls._handlers.keys())
            raise ValueError(
                f"Unknown forcing dataset: '{dataset_name}'. "
                f"Available datasets: {available}"
            )
        
        handler_class = cls._handlers[dataset_name_lower]
        return handler_class(config, logger, project_dir)
    
    @classmethod
    def list_datasets(cls) -> list:
        """
        Get a list of all registered dataset names.
        
        Returns:
            List of registered dataset names
        """
        return list(cls._handlers.keys())
    
    @classmethod
    def is_registered(cls, dataset_name: str) -> bool:
        """
        Check if a dataset is registered.
        
        Args:
            dataset_name: Name of the dataset to check
            
        Returns:
            True if registered, False otherwise
        """
        return dataset_name.lower() in cls._handlers
