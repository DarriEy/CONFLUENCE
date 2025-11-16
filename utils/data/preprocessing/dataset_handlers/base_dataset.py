"""
Base Dataset Handler for SYMFLUENCE

This module provides the abstract base class that all dataset-specific handlers must implement.
Each dataset handler encapsulates all dataset-specific logic including variable mappings,
unit conversions, grid specifications, and processing functions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xarray as xr
import geopandas as gpd


class BaseDatasetHandler(ABC):
    """
    Abstract base class for dataset-specific handlers.
    
    All dataset handlers must inherit from this class and implement the required methods.
    This ensures a consistent interface for the main preprocessor to interact with
    different forcing datasets.
    """
    
    def __init__(self, config: Dict, logger, project_dir: Path):
        """
        Initialize the dataset handler.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
            project_dir: Path to the project directory
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        self.domain_name = config['DOMAIN_NAME']
    
    @abstractmethod
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Return the mapping from dataset-specific variable names to standard names.
        
        Returns:
            Dictionary mapping original variable names to standard names
            Example: {'RDRS_v2.1_P_TT_09944': 'airtemp', ...}
        """
        pass
    
    @abstractmethod
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process a dataset by applying variable renaming and unit conversions.
        
        Args:
            ds: Input xarray Dataset
            
        Returns:
            Processed xarray Dataset with standardized variables and units
        """
        pass
    
    @abstractmethod
    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        Return the names of latitude and longitude coordinates in the dataset.
        
        Returns:
            Tuple of (latitude_name, longitude_name)
            Example: ('latitude', 'longitude') for ERA5
                     ('lat', 'lon') for RDRS/CASR
        """
        pass
    
    @abstractmethod
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path, 
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create a shapefile representing the forcing dataset grid.
        
        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to merged forcing data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics
            
        Returns:
            Path to the created shapefile
        """
        pass
    
    @abstractmethod
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Merge forcing data files into monthly files.
        
        Args:
            raw_forcing_path: Path to raw forcing data
            merged_forcing_path: Path where merged files should be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        pass
    
    @abstractmethod
    def needs_merging(self) -> bool:
        """
        Check if this dataset requires merging of raw files.
        
        Returns:
            True if merging is needed, False otherwise
        """
        pass
    
    def get_file_pattern(self) -> str:
        """
        Get the file naming pattern for this dataset.
        
        Returns:
            File pattern string (e.g., "domain_{domain_name}_*.nc")
        """
        return f"domain_{self.domain_name}_*.nc"
    
    def get_merged_file_pattern(self, year: int, month: int) -> str:
        """
        Get the pattern for merged monthly files.
        
        Args:
            year: Year
            month: Month
            
        Returns:
            File pattern for merged files
        """
        dataset_name = self.__class__.__name__.replace('Handler', '').upper()
        return f"{dataset_name}_monthly_{year}{month:02d}.nc"
    
    def setup_time_encoding(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Set up standard time encoding for the dataset.
        
        Args:
            ds: Dataset to modify
            
        Returns:
            Dataset with standardized time encoding
        """
        ds['time'].encoding['units'] = 'hours since 1900-01-01'
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds
    
    def add_metadata(self, ds: xr.Dataset, description: str) -> xr.Dataset:
        """
        Add standard metadata to the dataset.
        
        Args:
            ds: Dataset to modify
            description: Description of the data processing
            
        Returns:
            Dataset with added metadata
        """
        import time
        ds.attrs.update({
            'History': f'Created {time.ctime(time.time())}',
            'Language': 'Written using Python',
            'Reason': description
        })
        return ds
    
    def clean_variable_attributes(self, ds: xr.Dataset, missing_value: float = -999) -> xr.Dataset:
        """
        Clean up variable attributes and set missing value.
        
        Args:
            ds: Dataset to clean
            missing_value: Value to use for missing data
            
        Returns:
            Dataset with cleaned attributes
        """
        for var in ds.data_vars:
            # Remove conflicting attributes
            if 'missing_value' in ds[var].attrs:
                del ds[var].attrs['missing_value']
            if '_FillValue' in ds[var].attrs:
                del ds[var].attrs['_FillValue']
            # Set new missing value
            ds[var].attrs['missing_value'] = missing_value
        return ds
