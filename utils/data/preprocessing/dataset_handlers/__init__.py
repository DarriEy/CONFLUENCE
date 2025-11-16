"""
Dataset Handlers for SYMFLUENCE

This package provides dataset-specific handlers for different forcing datasets.
Each handler encapsulates all dataset-specific logic including variable mappings,
unit conversions, grid specifications, and shapefile creation.

Available Handlers:
    - RDRSHandler: Regional Deterministic Reforecast System
    - CASRHandler: Canadian Arctic System Reanalysis
    - ERA5Handler: ECMWF Reanalysis v5
    - CARRAHandler: Copernicus Arctic Regional Reanalysis

Usage:
    from dataset_handlers import DatasetRegistry
    
    # Get the appropriate handler for a dataset
    handler = DatasetRegistry.get_handler('era5', config, logger, project_dir)
    
    # Use the handler
    handler.merge_forcings(...)
    handler.create_shapefile(...)
"""

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry

# Import all handlers to register them
from .rdrs_utils import RDRSHandler
from .casr_utils import CASRHandler
from .era5_utils import ERA5Handler
from .carra_utils import CARRAHandler
from .aorc_utils import AORCHandler

__all__ = [
    'BaseDatasetHandler',
    'DatasetRegistry',
    'RDRSHandler',
    'CASRHandler',
    'ERA5Handler',
    'CARRAHandler',
]

__version__ = '1.0.0'
