"""
Cloud Data Utilities for SYMFLUENCE
====================================

Direct access to cloud-hosted forcing datasets via S3/Zarr/GCS
without requiring intermediate file downloads or hydrofabric preprocessing.

Activated via: DATA_ACCESS: cloud

Supported datasets:
- AORC: Analysis of Record for Calibration (CONUS, 1km, hourly, 1979-present)
  Storage: AWS S3 (s3://noaa-nws-aorc-v1-1-1km) - Zarr format
  
- ERA5: ECMWF Reanalysis (Global, 31km, hourly, 1940-present)
  Storage: Google Cloud (gs://gcp-public-data-arco-era5) - ARCO Zarr format
  
- EM-Earth: Ensemble Meteorological Dataset (Global, 11km, 1950-2019) [planned]

Requirements:
- s3fs: For AWS S3 access (AORC)
- gcsfs: For Google Cloud Storage access (ERA5)

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
    Download forcing data directly from cloud storage (AWS S3, Google Cloud Storage).
    
    This class provides methods to access cloud-optimized forcing datasets
    without requiring local file downloads or hydrofabric preprocessing.
    
    Supported storage:
    - AWS S3: AORC (s3fs)
    - Google Cloud Storage: ERA5 ARCO (gcsfs)
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
        Download ERA5 data from Google Cloud ARCO-ERA5.
        
        ARCO-ERA5 (Analysis-Ready, Cloud Optimized ERA5) provides:
        - Unified Zarr stores with all variables
        - Bucket: gcp-public-data-arco-era5
        - Location: us-central1 (Iowa)
        
        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data
            
        Returns
        -------
        Path
            Path to the saved NetCDF file
        """
        self.logger.info("Downloading ERA5 data from Google Cloud ARCO-ERA5")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")
        
        try:
            # Use gcsfs for Google Cloud Storage access
            import gcsfs
            
            # Initialize Google Cloud Storage filesystem (anonymous access)
            gcs = gcsfs.GCSFileSystem(token='anon')
            
            # ARCO-ERA5 analysis-ready store contains all surface variables
            # Structure: gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3
            zarr_store = 'gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
            
            self.logger.info(f"Opening ARCO-ERA5 Zarr store: {zarr_store}")
            
            # Open the Zarr store using gcsfs mapper
            mapper = gcs.get_mapper(zarr_store)
            ds = xr.open_zarr(mapper, consolidated=True)
            
            self.logger.info(f"Successfully opened Zarr store")
            self.logger.info(f"  Available variables: {list(ds.data_vars)}")
            
            # Subset by bounding box
            # ARCO-ERA5 uses standard longitude (0 to 360) and latitude
            lon_min = self.bbox['lon_min'] if self.bbox['lon_min'] >= 0 else self.bbox['lon_min'] + 360
            lon_max = self.bbox['lon_max'] if self.bbox['lon_max'] >= 0 else self.bbox['lon_max'] + 360
            
            self.logger.info(f"Subsetting spatial domain...")
            ds_subset = ds.sel(
                latitude=slice(self.bbox['lat_max'], self.bbox['lat_min']),  # ERA5 latitude is descending
                longitude=slice(lon_min, lon_max)
            )
            
            # Subset by time
            self.logger.info(f"Subsetting temporal domain...")
            ds_subset = ds_subset.sel(time=slice(self.start_date, self.end_date))
            
            # Select only the variables we need for hydrological modeling
            required_vars = [
                't2m',          # 2m temperature [K]
                'u10',          # 10m U wind component [m/s]
                'v10',          # 10m V wind component [m/s]
                'sp',           # Surface pressure [Pa]
                'd2m',          # 2m dewpoint temperature [K]
                'tp',           # Total precipitation [m]
                'ssrd',         # Surface solar radiation downwards [J/m2]
                'strd',         # Surface thermal radiation downwards [J/m2]
            ]
            
            # Filter to only variables that exist in the dataset
            available_vars = [var for var in required_vars if var in ds_subset.data_vars]
            
            if not available_vars:
                raise ValueError("None of the required variables found in ARCO-ERA5 dataset")
            
            ds_subset = ds_subset[available_vars]
            
            self.logger.info(f"Selected variables: {available_vars}")
            
            # Load the data (this triggers the actual download from GCS)
            self.logger.info("Loading data from cloud storage (this may take a few minutes)...")
            ds_final = ds_subset.load()
            
            # Log data summary
            self.logger.info(f"ERA5 data extraction summary:")
            self.logger.info(f"  Dimensions: {dict(ds_final.dims)}")
            self.logger.info(f"  Variables: {list(ds_final.data_vars)}")
            self.logger.info(f"  Time steps: {len(ds_final.time)}")
            self.logger.info(f"  Grid size: {len(ds_final.latitude)} x {len(ds_final.longitude)}")
            
            # Add metadata
            ds_final.attrs['source'] = 'ARCO-ERA5 (Google Cloud)'
            ds_final.attrs['source_url'] = 'gs://gcp-public-data-arco-era5'
            ds_final.attrs['downloaded_by'] = 'SYMFLUENCE cloud_data_utils'
            ds_final.attrs['download_date'] = pd.Timestamp.now().isoformat()
            ds_final.attrs['bbox'] = str(self.bbox)
            
            # Save to NetCDF
            output_dir.mkdir(parents=True, exist_ok=True)
            domain_name = self.config.get('DOMAIN_NAME', 'domain')
            output_file = output_dir / f'{domain_name}_ERA5_{self.start_date.year}-{self.end_date.year}.nc'
            
            self.logger.info(f"Saving ERA5 data to: {output_file}")
            ds_final.to_netcdf(output_file)
            
            self.logger.info(f"✓ ERA5 data download complete: {output_file}")
            return output_file
            
        except ImportError:
            raise ImportError(
                "gcsfs package is required for ERA5 cloud access. "
                "Install with: pip install gcsfs"
            )
        except Exception as e:
            self.logger.error(f"Error downloading ERA5 data: {str(e)}")
            raise

    
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


def get_era5_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from ERA5 variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of ERA5 variables to standard names
    """
    return {
        't2m': 'airtemp',      # 2m temperature [K]
        'u10': 'wind_u',       # 10m U wind component [m/s]
        'v10': 'wind_v',       # 10m V wind component [m/s]
        'sp': 'airpres',       # Surface pressure [Pa]
        'd2m': 'dewpoint',     # 2m dewpoint temperature [K]
        'q': 'spechum',        # Specific humidity [kg/kg]
        'tp': 'pptrate',       # Total precipitation [m]
        'ssrd': 'SWRadAtm',    # Surface solar radiation downwards [J/m2]
        'strd': 'LWRadAtm',    # Surface thermal radiation downwards [J/m2]
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
    supported_datasets = ['AORC', 'ERA5']
    
    if dataset_name.upper() in supported_datasets:
        logger.info(f"✓ {dataset_name} supports cloud data access")
        return True
    else:
        logger.warning(
            f"✗ {dataset_name} does not support cloud access. "
            f"Supported datasets: {', '.join(supported_datasets)}"
        )
        return False