"""
Cloud Data Utilities for SYMFLUENCE
====================================

Direct access to cloud-hosted forcing datasets (AORC, ERA5, etc.) via S3/Zarr
without requiring intermediate file downloads or hydrofabric preprocessing.

Activated via: DATA_ACCESS: cloud

Supported datasets:
- AORC: Analysis of Record for Calibration (CONUS, 1km, hourly, 1979-present)
- ERA5: ECMWF Reanalysis (Global, 31km, hourly, 1940-present) [planned]
- EM-Earth: Ensemble Meteorological Dataset (Global, 11km, 1950-2019) [planned]

Author: SYMFLUENCE Development Team
Date: 2025-01-14
"""

import xarray as xr
import s3fs
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class CloudForcingDownloader:
    """
    Download forcing data directly from cloud storage (AWS S3) using Zarr/NetCDF.
    
    This class provides methods to access cloud-optimized forcing datasets
    without requiring local file downloads or hydrofabric preprocessing.
    """
    
    def __init__(self, config: Dict, logger):
        """
        Initialize the CloudForcingDownloader.
        
        Parameters
        ----------
        config : dict
            SYMFLUENCE configuration dictionary
        logger : logging.Logger
            Logger instance for tracking progress
        """
        self.config = config
        self.logger = logger
        
        # Parse bounding box
        self.bbox = self._parse_bbox(config['BOUNDING_BOX_COORDS'])
        
        # Parse time period
        self.start_date = pd.to_datetime(config['EXPERIMENT_TIME_START'])
        self.end_date = pd.to_datetime(config['EXPERIMENT_TIME_END'])
        
        # Get forcing dataset
        self.dataset_name = config.get('FORCING_DATASET', '').upper()
        
        # Initialize S3 filesystem
        self.fs = s3fs.S3FileSystem(anon=True)
        
    def _parse_bbox(self, bbox_string: str) -> Dict[str, float]:
        """
        Parse bounding box string into dictionary.
        
        Parameters
        ----------
        bbox_string : str
            Bounding box in format 'lat_max/lon_max/lat_min/lon_min'
            
        Returns
        -------
        dict
            Dictionary with keys: lat_min, lat_max, lon_min, lon_max
        """
        coords = bbox_string.split('/')
        return {
            'lat_min': float(coords[2]),
            'lat_max': float(coords[0]),
            'lon_min': float(coords[3]),
            'lon_max': float(coords[1])
        }
    
    def download_forcing_data(self, output_dir: Path) -> Path:
        """
        Download forcing data based on configured dataset.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded forcing data
            
        Returns
        -------
        Path
            Path to the downloaded forcing file
            
        Raises
        ------
        ValueError
            If dataset is not supported for cloud access
        """
        self.logger.info(f"Starting cloud data download for {self.dataset_name}")
        
        if self.dataset_name == 'AORC':
            return self._download_aorc(output_dir)
        elif self.dataset_name == 'ERA5':
            return self._download_era5(output_dir)
        elif self.dataset_name == 'EM-EARTH':
            return self._download_emearth(output_dir)
        else:
            raise ValueError(
                f"Dataset '{self.dataset_name}' is not supported for cloud access. "
                f"Supported datasets: AORC, ERA5, EM-EARTH"
            )
    
    def _download_aorc(self, output_dir: Path) -> Path:
        """
        Download AORC data from S3 Zarr store.
        
        AORC bucket structure:
        s3://noaa-nws-aorc-v1-1-1km/YEAR.zarr
        
        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data
            
        Returns
        -------
        Path
            Path to the saved NetCDF file
        """
        self.logger.info("Downloading AORC data from S3")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")
        
        # Determine which years we need
        years = range(self.start_date.year, self.end_date.year + 1)
        
        datasets = []
        for year in years:
            self.logger.info(f"  Processing year {year}...")
            
            try:
                # Open Zarr store for this year
                zarr_path = f'noaa-nws-aorc-v1-1-1km/{year}.zarr'
                store = s3fs.S3Map(zarr_path, s3=self.fs)
                ds = xr.open_zarr(store)
                
                # Subset by bounding box and time
                ds_subset = ds.sel(
                    latitude=slice(self.bbox['lat_min'], self.bbox['lat_max']),
                    longitude=slice(self.bbox['lon_min'], self.bbox['lon_max'])
                )
                
                # Filter by time range for this year
                year_start = max(self.start_date, pd.Timestamp(f'{year}-01-01'))
                year_end = min(self.end_date, pd.Timestamp(f'{year}-12-31 23:59:59'))
                ds_subset = ds_subset.sel(time=slice(year_start, year_end))
                
                if len(ds_subset.time) > 0:
                    datasets.append(ds_subset)
                    self.logger.info(f"    ✓ Extracted {len(ds_subset.time)} timesteps")
                
            except Exception as e:
                self.logger.error(f"    ✗ Error processing year {year}: {str(e)}")
                raise
        
        if not datasets:
            raise ValueError("No data extracted for the specified time period")
        
        # Combine all years
        self.logger.info("Combining data across years...")
        ds_combined = xr.concat(datasets, dim='time')
        
        # Log data summary
        self.logger.info(f"Data extraction summary:")
        self.logger.info(f"  Dimensions: {dict(ds_combined.dims)}")
        self.logger.info(f"  Variables: {list(ds_combined.data_vars)}")
        self.logger.info(f"  Time steps: {len(ds_combined.time)}")
        self.logger.info(f"  Grid size: {len(ds_combined.latitude)} x {len(ds_combined.longitude)}")
        
        # Add metadata
        ds_combined.attrs['source'] = 'NOAA AORC v1.1'
        ds_combined.attrs['source_url'] = 's3://noaa-nws-aorc-v1-1-1km'
        ds_combined.attrs['downloaded_by'] = 'SYMFLUENCE cloud_data_utils'
        ds_combined.attrs['download_date'] = pd.Timestamp.now().isoformat()
        ds_combined.attrs['bbox'] = str(self.bbox)
        
        # Save to NetCDF
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        output_file = output_dir / f'{domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc'
        
        self.logger.info(f"Saving data to: {output_file}")
        ds_combined.to_netcdf(output_file)
        
        self.logger.info(f"✓ AORC data download complete: {output_file}")
        return output_file
    
    def _download_era5(self, output_dir: Path) -> Path:
        """
        Download ERA5 data from S3.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data
            
        Returns
        -------
        Path
            Path to the saved NetCDF file
            
        Raises
        ------
        NotImplementedError
            ERA5 download not yet implemented
        """
        raise NotImplementedError(
            "ERA5 cloud download is not yet implemented. "
            "Please use CLOUD_DATA_ACCESS: False and conventional download methods."
        )
    
    def _download_emearth(self, output_dir: Path) -> Path:
        """
        Download EM-Earth data from S3.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data
            
        Returns
        -------
        Path
            Path to the saved NetCDF file
            
        Raises
        ------
        NotImplementedError
            EM-Earth download not yet implemented
        """
        raise NotImplementedError(
            "EM-Earth cloud download is not yet implemented. "
            "Please use CLOUD_DATA_ACCESS: False and conventional download methods."
        )


def get_aorc_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from AORC variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of AORC variables to standard names
    """
    return {
        'APCP_surface': 'pptrate',           # Precipitation rate [kg/m2 or mm]
        'TMP_2maboveground': 'airtemp',      # Air temperature [K]
        'SPFH_2maboveground': 'spechum',     # Specific humidity [kg/kg]
        'PRES_surface': 'airpres',           # Surface pressure [Pa]
        'DLWRF_surface': 'LWRadAtm',         # Downward longwave radiation [W/m2]
        'DSWRF_surface': 'SWRadAtm',         # Downward shortwave radiation [W/m2]
        'UGRD_10maboveground': 'wind_u',     # U-component of wind [m/s]
        'VGRD_10maboveground': 'wind_v'      # V-component of wind [m/s]
    }


def check_cloud_access_availability(dataset_name: str, logger) -> bool:
    """
    Check if a dataset is available for cloud access.
    
    Parameters
    ----------
    dataset_name : str
        Name of the forcing dataset
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    bool
        True if dataset supports cloud access
    """
    supported_datasets = ['AORC']
    
    if dataset_name.upper() in supported_datasets:
        logger.info(f"✓ {dataset_name} supports cloud data access")
        return True
    else:
        logger.warning(
            f"✗ {dataset_name} does not support cloud access. "
            f"Supported datasets: {', '.join(supported_datasets)}"
        )
        return False