import os
from pathlib import Path
import easymore # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import shutil
from rasterio.mask import mask # type: ignore
from shapely.geometry import Polygon # type: ignore
import rasterstats # type: ignore
from pyproj import CRS, Transformer # type: ignore
import pyproj # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore
import multiprocessing as mp
import time
import uuid

class forcingResampler:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.domain_name = self.config['DOMAIN_NAME']
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{str(self.config['DOMAIN_DISCRETIZATION']).replace(',','_')}.shp"
        self.forcing_dataset = self.config.get('FORCING_DATASET').lower()
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/raw_data')
        
        # Merge forcings for both RDRS and CASR
        if self.forcing_dataset in ['rdrs', 'casr']:
            self.merge_forcings()
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
            
    def _get_default_path(self, path_key, default_subpath):
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)

    def run_resampling(self):
        self.logger.info("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.info("Forcing data resampling process completed")

    def merge_forcings(self):
        """
        Merge forcing data files into monthly files for RDRS and CASR datasets.

        This method performs the following steps:
        1. Determine the year range for processing
        2. Create output directories
        3. Process each year and month, merging daily files into monthly files
        4. Apply dataset-specific variable renaming and unit conversions
        5. Save the merged and processed data to netCDF files

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If there are issues with data processing or merging.
            IOError: If there are issues writing output files.
        """
        if self.forcing_dataset == 'rdrs':
            self.logger.info("Starting to merge RDRS forcing data")
            self._merge_rdrs_forcings()
        elif self.forcing_dataset == 'casr':
            self.logger.info("Starting to merge CASR forcing data")
            self._merge_casr_forcings()
        else:
            self.logger.warning(f"Merging not implemented for dataset: {self.forcing_dataset}")

    def _merge_rdrs_forcings(self):
        """Merge RDRS forcing data files into monthly files."""
        years = [
                    self.config.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
        years = range(int(years[0])-1, int(years[1]) + 1)
        
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        
        self.merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'
        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
        
        variable_mapping = {
            'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
            'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
            'RDRS_v2.1_A_PR0_SFC': 'pptrate',
            'RDRS_v2.1_P_P0_SFC': 'airpres',
            'RDRS_v2.1_P_TT_09944': 'airtemp',
            'RDRS_v2.1_P_HU_09944': 'spechum',
            'RDRS_v2.1_P_UVC_09944': 'windspd',
            'RDRS_v2.1_P_TT_1.5m': 'airtemp',
            'RDRS_v2.1_P_HU_1.5m': 'spechum',
            'RDRS_v2.1_P_UVC_10m': 'windspd',
            'RDRS_v2.1_P_UUC_10m': 'windspd_u',
            'RDRS_v2.1_P_VVC_10m': 'windspd_v',
        }

        def process_rdrs_data(ds):
            existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
            ds = ds.rename(existing_vars)
            
            if 'airpres' in ds:
                ds['airpres'] = ds['airpres'] * 100
                ds['airpres'].attrs.update({'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'})
            
            if 'airtemp' in ds:
                ds['airtemp'] = ds['airtemp'] + 273.15
                ds['airtemp'].attrs.update({'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'})
            
            if 'pptrate' in ds:
                ds['pptrate'] = ds['pptrate'] / 3600 * 1000
                ds['pptrate'].attrs.update({'units': 'm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'})
            
            if 'windspd' in ds:
                ds['windspd'] = ds['windspd'] * 0.514444
                ds['windspd'].attrs.update({'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'})
            
            if 'LWRadAtm' in ds:
                ds['LWRadAtm'].attrs.update({'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'})
            
            if 'SWRadAtm' in ds:
                ds['SWRadAtm'].attrs.update({'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'})
            
            if 'spechum' in ds:
                ds['spechum'].attrs.update({'long_name': 'specific humidity', 'standard_name': 'specific_humidity'})
            
            return ds

        for year in years:
            self.logger.info(f"Processing RDRS year {year}")
            year_folder = raw_forcing_path / str(year)

            for month in range(1, 13):
                self.logger.info(f"Processing RDRS {year}-{month:02d}")
                
                daily_files = sorted(year_folder.glob(file_name_pattern.replace('*', f'{year}{month:02d}*')))         

                if not daily_files:
                    self.logger.warning(f"No RDRS files found for {year}-{month:02d}")
                    continue
                
                datasets = []
                for file in daily_files:
                    try:
                        ds = xr.open_dataset(file)
                        datasets.append(ds)
                    except Exception as e:
                        self.logger.error(f"Error opening RDRS file {file}: {str(e)}")

                if not datasets:
                    self.logger.warning(f"No valid RDRS datasets for {year}-{month:02d}")
                    continue

                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = process_rdrs_data(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:
                        self.logger.error(f"Error processing RDRS dataset: {str(e)}")

                if not processed_datasets:
                    self.logger.warning(f"No processed RDRS datasets for {year}-{month:02d}")
                    continue

                monthly_data = xr.concat(processed_datasets, dim="time")
                monthly_data = monthly_data.sortby("time")

                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

                monthly_data = monthly_data.sel(time=slice(start_time, end_time))

                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times, method='nearest')

                monthly_data['time'].encoding['units'] = 'hours since 1900-01-01'
                monthly_data['time'].encoding['calendar'] = 'gregorian'

                monthly_data.attrs.update({
                    'History': f'Created {time.ctime(time.time())}',
                    'Language': 'Written using Python',
                    'Reason': 'RDRS data aggregated to monthly files and variables renamed for SUMMA compatibility'
                })

                for var in monthly_data.data_vars:
                    # Remove any existing missing_value or _FillValue attributes that might conflict
                    if 'missing_value' in monthly_data[var].attrs:
                        del monthly_data[var].attrs['missing_value']
                    if '_FillValue' in monthly_data[var].attrs:
                        del monthly_data[var].attrs['_FillValue']
                    # Set the new missing_value
                    monthly_data[var].attrs['missing_value'] = -999

                output_file = self.merged_forcing_path / f"RDRS_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)

                for ds in datasets:
                    ds.close()

        self.logger.info("RDRS forcing data merging completed")

    def _merge_casr_forcings(self):
        """Merge CASR forcing data files into monthly files."""
        years = [
                    self.config.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
        years = range(int(years[0])-1, int(years[1]) + 1)
        
        # CASR files are all in the same directory, not organized by year
        self.merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'
        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        self.merged_forcing_path.mkdir(parents=True, exist_ok=True)

        # Get all CASR files in the raw_data directory
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        all_casr_files = sorted(raw_forcing_path.glob(file_name_pattern))
        
        if not all_casr_files:
            self.logger.warning(f"No CASR files found in {raw_forcing_path}")
            return

        self.logger.info(f"Found {len(all_casr_files)} CASR files in {raw_forcing_path}")

        for year in years:
            self.logger.info(f"Processing CASR year {year}")

            for month in range(1, 13):
                self.logger.info(f"Processing CASR {year}-{month:02d}")
                
                # Look for daily CASR files for this month - files are all in raw_data directory
                # Pattern: domain_name_YYYYMMDD*.nc
                month_pattern = f"domain_{self.domain_name}_{year}{month:02d}"
                daily_files = sorted([f for f in all_casr_files if month_pattern in f.name])

                if not daily_files:
                    self.logger.warning(f"No CASR files found for {year}-{month:02d}")
                    continue
                
                self.logger.info(f"Found {len(daily_files)} CASR files for {year}-{month:02d}")
                
                datasets = []
                for file in daily_files:
                    try:
                        ds = xr.open_dataset(file)
                        ds = ds.drop_duplicates(dim='time')
                        datasets.append(ds)
                    except Exception as e:
                        self.logger.error(f"Error opening CASR file {file}: {str(e)}")

                if not datasets:
                    self.logger.warning(f"No valid CASR datasets for {year}-{month:02d}")
                    continue

                # Process each dataset using the existing CASR processing method
                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = self.process_casr_data(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:
                        self.logger.error(f"Error processing CASR dataset: {str(e)}")

                if not processed_datasets:
                    self.logger.warning(f"No processed CASR datasets for {year}-{month:02d}")
                    continue

                # Concatenate daily files into monthly data
                monthly_data = xr.concat(processed_datasets, dim="time")
                monthly_data = monthly_data.sortby("time")

                # Set up time range for the month
                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

                # Filter data to the expected time range
                monthly_data = monthly_data.sel(time=slice(start_time, end_time))

                # Ensure complete hourly time series
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times, method='nearest')

                # Set time encoding
                monthly_data['time'].encoding['units'] = 'hours since 1900-01-01'
                monthly_data['time'].encoding['calendar'] = 'gregorian'

                # Add metadata
                monthly_data.attrs.update({
                    'History': f'Created {time.ctime(time.time())}',
                    'Language': 'Written using Python',
                    'Reason': 'CASR data aggregated to monthly files and variables renamed for SUMMA compatibility'
                })

                # Aggressively clean up variable attributes and encoding to avoid conflicts
                for var_name in monthly_data.data_vars:
                    var = monthly_data[var_name]
                    # Clear all existing attributes that might cause encoding conflicts
                    var.attrs.clear()
                    var.encoding.clear()
                    
                    # Set clean attributes based on variable name
                    if var_name == 'airpres':
                        var.attrs = {'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure', 'missing_value': -999}
                    elif var_name == 'airtemp':
                        var.attrs = {'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature', 'missing_value': -999}
                    elif var_name == 'pptrate':
                        var.attrs = {'units': 'm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate', 'missing_value': -999}
                    elif var_name == 'windspd':
                        var.attrs = {'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed', 'missing_value': -999}
                    elif var_name == 'windspd_u':
                        var.attrs = {'units': 'm s-1', 'long_name': 'eastward wind', 'standard_name': 'eastward_wind', 'missing_value': -999}
                    elif var_name == 'windspd_v':
                        var.attrs = {'units': 'm s-1', 'long_name': 'northward wind', 'standard_name': 'northward_wind', 'missing_value': -999}
                    elif var_name == 'LWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air', 'missing_value': -999}
                    elif var_name == 'SWRadAtm':
                        var.attrs = {'units': 'W m-2', 'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air', 'missing_value': -999}
                    elif var_name == 'spechum':
                        var.attrs = {'units': 'kg kg-1', 'long_name': 'specific humidity', 'standard_name': 'specific_humidity', 'missing_value': -999}
                    else:
                        # For any other variables, just set missing_value
                        var.attrs = {'missing_value': -999}

                # Save monthly file
                output_file = self.merged_forcing_path / f"CASR_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                self.logger.info(f"Saved CASR monthly file: {output_file}")

                # Close datasets to free memory
                for ds in datasets:
                    ds.close()

        self.logger.info("CASR forcing data merging completed")

    def create_shapefile(self):
        """Create forcing shapefile with check for existing files"""
        self.logger.info(f"Creating {self.forcing_dataset.upper()} shapefile")
        
        # Check if shapefile already exists
        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        if output_shapefile.exists():
            try:
                # Verify the shapefile is valid
                gdf = gpd.read_file(output_shapefile)
                expected_columns = [self.config.get('FORCING_SHAPE_LAT_NAME'), 
                                    self.config.get('FORCING_SHAPE_LON_NAME'), 
                                    'ID', 'elev_m']
                
                if all(col in gdf.columns for col in expected_columns) and len(gdf) > 0:
                    self.logger.info(f"Forcing shapefile already exists: {output_shapefile}. Skipping creation.")
                    return output_shapefile
                else:
                    self.logger.info(f"Existing forcing shapefile missing expected columns. Recreating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing forcing shapefile: {str(e)}. Recreating.")
        
        # Create appropriate shapefile based on forcing dataset
        if self.forcing_dataset == 'rdrs':
            return self._create_rdrs_shapefile()
        elif self.forcing_dataset.lower() == 'casr':
            return self._create_casr_shapefile()
        elif self.forcing_dataset.lower() == 'era5':
            return self._create_era5_shapefile()
        elif self.forcing_dataset == 'carra':
            return self._create_carra_shapefile()
        else:
            self.logger.error(f"Unsupported forcing dataset: {self.forcing_dataset}")
            raise ValueError(f"Unsupported forcing dataset: {self.forcing_dataset}")

    def process_casr_data(self, ds):
        """
        Process CASR dataset by renaming variables and applying unit conversions.
        
        Args:
            ds: xarray Dataset containing CASR data
            
        Returns:
            xarray Dataset with renamed variables and corrected units
        """
        # CASR variable mapping to standard names
        casr_variable_mapping = {
            # Temperature (choose between analysis A_ or forecast P_ - prefer analysis)
            'CaSR_v3.1_A_TT_1.5m': 'airtemp',  # Analysis air temperature at 1.5m
            'CaSR_v3.1_P_TT_1.5m': 'airtemp',  # Forecast air temperature at 1.5m (fallback)
            
            # Precipitation 
            'CaSR_v3.1_A_PR0_SFC': 'pptrate',  # Analysis precipitation (1h)
            'CaSR_v3.1_P_PR0_SFC': 'pptrate',  # Forecast precipitation (1h) (fallback)
            
            # Pressure
            'CaSR_v3.1_P_P0_SFC': 'airpres',   # Surface pressure
            
            # Humidity
            'CaSR_v3.1_P_HU_1.5m': 'spechum',  # Specific humidity at 1.5m
            
            # Wind speed
            'CaSR_v3.1_P_UVC_10m': 'windspd',  # Wind speed at 10m
            
            # Wind components
            'CaSR_v3.1_P_UUC_10m': 'windspd_u',  # U-component corrected
            'CaSR_v3.1_P_VVC_10m': 'windspd_v',  # V-component corrected
            
            # Radiation
            'CaSR_v3.1_P_FB_SFC': 'SWRadAtm',   # Downward solar flux
            'CaSR_v3.1_P_FI_SFC': 'LWRadAtm',   # Surface incoming infrared flux
        }
        
        # Rename variables that exist in the dataset
        existing_vars = {old: new for old, new in casr_variable_mapping.items() if old in ds.variables}
        ds = ds.rename(existing_vars)
        
        # Apply unit conversions for CASR data
        if 'airpres' in ds:
            # CASR pressure is in mb, convert to Pa
            ds['airpres'] = ds['airpres'] * 100
            # Clean up attributes to avoid conflicts
            ds['airpres'].attrs = {}
            ds['airpres'].attrs.update({'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'})
        
        if 'airtemp' in ds:
            # CASR temperature is in deg_C, convert to K
            ds['airtemp'] = ds['airtemp'] + 273.15
            # Clean up attributes to avoid conflicts
            ds['airtemp'].attrs = {}
            ds['airtemp'].attrs.update({'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'})
        
        if 'pptrate' in ds:
            # CASR precipitation is in m (per hour), convert to mm/s
            ds['pptrate'] = ds['pptrate'] * 1000 / 3600
            # Clean up attributes to avoid conflicts
            ds['pptrate'].attrs = {}
            ds['pptrate'].attrs.update({'units': 'm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'})
        
        if 'windspd' in ds:
            # CASR wind speed is in kts (knots), convert to m/s
            ds['windspd'] = ds['windspd'] * 0.514444
            # Clean up attributes to avoid conflicts
            ds['windspd'].attrs = {}
            ds['windspd'].attrs.update({'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'})
        
        if 'windspd_u' in ds:
            # Convert U-component from kts to m/s
            ds['windspd_u'] = ds['windspd_u'] * 0.514444
            # Clean up attributes to avoid conflicts
            ds['windspd_u'].attrs = {}
            ds['windspd_u'].attrs.update({'units': 'm s-1', 'long_name': 'eastward wind', 'standard_name': 'eastward_wind'})
        
        if 'windspd_v' in ds:
            # Convert V-component from kts to m/s  
            ds['windspd_v'] = ds['windspd_v'] * 0.514444
            # Clean up attributes to avoid conflicts
            ds['windspd_v'].attrs = {}
            ds['windspd_v'].attrs.update({'units': 'm s-1', 'long_name': 'northward wind', 'standard_name': 'northward_wind'})
        
        # Radiation variables are already in W m**-2, just update attributes
        if 'LWRadAtm' in ds:
            # Clean up attributes to avoid conflicts
            ds['LWRadAtm'].attrs = {}
            ds['LWRadAtm'].attrs.update({'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'})
        
        if 'SWRadAtm' in ds:
            # Clean up attributes to avoid conflicts
            ds['SWRadAtm'].attrs = {}
            ds['SWRadAtm'].attrs.update({'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'})
        
        if 'spechum' in ds:
            # Clean up attributes to avoid conflicts
            ds['spechum'].attrs = {}
            ds['spechum'].attrs.update({'long_name': 'specific humidity', 'standard_name': 'specific_humidity'})
        
        return ds

    def _create_casr_shapefile(self):
        """Create CASR shapefile with output file checking"""
        self.logger.info("Creating CASR grid shapefile")
        
        # Define output shapefile path
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find a CASR file to get grid information
            casr_files = list(self.merged_forcing_path.glob('*.nc'))
            if not casr_files:
                raise FileNotFoundError("No CASR files found")
            casr_file = casr_files[0]
            
            self.logger.info(f"Using CASR file for grid: {casr_file}")

            # Read CASR data - similar structure to RDRS
            with xr.open_dataset(casr_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values

            self.logger.info(f"CASR dimensions: rlat={rlat.shape}, rlon={rlon.shape}")
            
            geometries, ids, lats, lons = [], [], [], []
            
            # Create grid cells in batches to manage memory
            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size
            
            self.logger.info(f"Creating CASR grid cells in {num_batches} batches")
            
            cell_count = 0
            for i in range(len(rlat)):
                for j in range(len(rlon)):
                    # Create grid cell corners similar to RDRS
                    rlat_corners = [rlat[i], rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i]]
                    rlon_corners = [rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j]]
                    lat_corners = [lat[i,j], lat[i,j+1] if j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j] if i+1 < len(rlat) else lat[i,j]]
                    lon_corners = [lon[i,j], lon[i,j+1] if j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j] if i+1 < len(rlat) else lon[i,j]]
                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * len(rlon) + j)
                    lats.append(lat[i,j])
                    lons.append(lon[i,j])
                    
                    cell_count += 1
                    if cell_count % batch_size == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} CASR grid cells")

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): lons,
            }, crs='EPSG:4326')
            
            # Add elevation data in batches
            gdf['elev_m'] = -9999  # Default value
            
            batch_size = 50
            num_batches = (len(gdf) + batch_size - 1) // batch_size
            
            self.logger.info(f"Calculating elevations in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gdf))
                
                self.logger.info(f"Processing elevation batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                
                try:
                    # Get batch of geometries
                    batch_gdf = gdf.iloc[start_idx:end_idx]
                    
                    # Calculate zonal statistics for this batch
                    zs = rasterstats.zonal_stats(batch_gdf, str(self.dem_path), stats=['mean'])
                    
                    # Update elevation values in the main GeoDataFrame
                    for i, item in enumerate(zs):
                        idx = start_idx + i
                        gdf.loc[idx, 'elev_m'] = item['mean'] if item['mean'] is not None else -9999
                except Exception as e:
                    self.logger.warning(f"Error calculating elevations for batch {batch_idx+1}: {str(e)}")
                    # Continue with next batch

            # Remove rows with invalid elevation values if requested
            if self.config.get('REMOVE_INVALID_ELEVATION_CELLS', False):
                valid_count = len(gdf)
                gdf = gdf[gdf['elev_m'] != -9999].copy()
                removed_count = valid_count - len(gdf)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} cells with invalid elevation values")

            # Save the shapefile
            output_shapefile.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"CASR shapefile created and saved to {output_shapefile}")
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_casr_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _create_era5_shapefile(self):
        """Create ERA5 shapefile with output file checking"""
        self.logger.info("Creating ERA5 shapefile")
        
        # Define output shapefile path
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find an .nc file in the forcing path
            forcing_files = list(self.merged_forcing_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No ERA5 forcing files found")
            
            forcing_file = forcing_files[0]
            self.logger.info(f"Using ERA5 file: {forcing_file}")

            # Set the dimension variable names
            source_name_lat = "latitude"
            source_name_lon = "longitude"

            # Open the file and get the dimensions and spatial extent of the domain
            try:
                with xr.open_dataset(forcing_file) as src:
                    lat = src[source_name_lat].values
                    lon = src[source_name_lon].values
                
                self.logger.info(f"ERA5 dimensions: lat={lat.shape}, lon={lon.shape}")
            except Exception as e:
                self.logger.error(f"Error reading ERA5 dimensions: {str(e)}")
                raise

            # Find the spacing
            try:
                half_dlat = abs(lat[1] - lat[0])/2 if len(lat) > 1 else 0.125  # Default to 0.25/2 if only one value
                half_dlon = abs(lon[1] - lon[0])/2 if len(lon) > 1 else 0.125  # Default to 0.25/2 if only one value
                
                self.logger.info(f"ERA5 grid spacings: half_dlat={half_dlat}, half_dlon={half_dlon}")
            except Exception as e:
                self.logger.error(f"Error calculating grid spacings: {str(e)}")
                raise

            # Create lists to store the data
            geometries = []
            ids = []
            lats = []
            lons = []

            # Create grid cells
            try:
                self.logger.info("Creating grid cell geometries")
                if len(lat) == 1:
                    self.logger.info("Single latitude value detected, creating row of grid cells")
                    for i, center_lon in enumerate(lon):
                        center_lat = lat[0]
                        vertices = [
                            [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                            [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                            [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                            [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                            [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                        ]
                        geometries.append(Polygon(vertices))
                        ids.append(i)
                        lats.append(float(center_lat))
                        lons.append(float(center_lon))
                else:
                    self.logger.info("Multiple latitude values, creating grid")
                    for i, center_lon in enumerate(lon):
                        for j, center_lat in enumerate(lat):
                            vertices = [
                                [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                                [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                                [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                                [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                                [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                            ]
                            geometries.append(Polygon(vertices))
                            ids.append(i * len(lat) + j)
                            lats.append(float(center_lat))
                            lons.append(float(center_lon))
                
                self.logger.info(f"Created {len(geometries)} grid cell geometries")
            except Exception as e:
                self.logger.error(f"Error creating grid cell geometries: {str(e)}")
                raise

            # Create the GeoDataFrame
            try:
                self.logger.info("Creating GeoDataFrame")
                gdf = gpd.GeoDataFrame({
                    'geometry': geometries,
                    'ID': ids,
                    self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                    self.config.get('FORCING_SHAPE_LON_NAME'): lons,
                }, crs='EPSG:4326')
                
                self.logger.info(f"GeoDataFrame created with {len(gdf)} rows")
            except Exception as e:
                self.logger.error(f"Error creating GeoDataFrame: {str(e)}")
                raise

            # Calculate zonal statistics
            try:
                self.logger.info(f"Calculating zonal statistics with DEM: {self.dem_path}")
                if not Path(self.dem_path).exists():
                    self.logger.error(f"DEM file not found: {self.dem_path}")
                    raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
                    
                # To avoid memory issues, use a sample-based approach
                sample_size = min(100, len(gdf))
                self.logger.info(f"Using sample of {sample_size} grid cells for zonal statistics")
                
                # Process in batches to avoid memory issues
                batch_size = 20
                num_batches = (len(gdf) + batch_size - 1) // batch_size
                
                # Initialize elevation column with default value
                gdf['elev_m'] = -9999
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(gdf))
                    
                    self.logger.info(f"Processing elevation batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                    
                    try:
                        # Get batch of geometries
                        batch_gdf = gdf.iloc[start_idx:end_idx]
                        
                        # Calculate zonal statistics for this batch
                        zs = rasterstats.zonal_stats(batch_gdf, str(self.dem_path), stats=['mean'])
                        
                        # Update elevation values in the main GeoDataFrame
                        for i, item in enumerate(zs):
                            idx = start_idx + i
                            gdf.loc[idx, 'elev_m'] = item['mean'] if item['mean'] is not None else -9999
                    except Exception as e:
                        self.logger.warning(f"Error calculating elevations for batch {batch_idx+1}: {str(e)}")
                        # Continue with next batch
                
                self.logger.info(f"Elevation calculation complete")
            except Exception as e:
                self.logger.error(f"Error calculating zonal statistics: {str(e)}")
                # Continue without elevation data rather than failing completely
                gdf['elev_m'] = -9999
                self.logger.warning("Using default elevation values due to zonal statistics error")

            # Save the shapefile
            try:
                self.logger.info(f"Saving shapefile to: {output_shapefile}")
                gdf.to_file(output_shapefile)
                self.logger.info(f"ERA5 shapefile saved successfully to {output_shapefile}")
                return output_shapefile
            except Exception as e:
                self.logger.error(f"Error saving shapefile: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                
        except Exception as e:
            self.logger.error(f"Error in create_era5_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _create_rdrs_shapefile(self):
        """Create RDRS shapefile with output file checking"""
        # Define output shapefile path
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            forcing_file = next((f for f in os.listdir(self.merged_forcing_path) if f.endswith('.nc') and f.startswith('RDRS_monthly_')), None)
            if not forcing_file:
                self.logger.error("No RDRS monthly file found")
                return None

            with xr.open_dataset(self.merged_forcing_path / forcing_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values

            self.logger.info(f"RDRS dimensions: rlat={rlat.shape}, rlon={rlon.shape}")
            
            geometries, ids, lats, lons = [], [], [], []
            
            # Create grid cells in batches to manage memory
            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size
            
            self.logger.info(f"Creating RDRS grid cells in {num_batches} batches")
            
            cell_count = 0
            for i in range(len(rlat)):
                for j in range(len(rlon)):
                    rlat_corners = [rlat[i], rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i]]
                    rlon_corners = [rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j]]
                    lat_corners = [lat[i,j], lat[i,j+1] if j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j] if i+1 < len(rlat) else lat[i,j]]
                    lon_corners = [lon[i,j], lon[i,j+1] if j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j] if i+1 < len(rlat) else lon[i,j]]
                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * len(rlon) + j)
                    lats.append(lat[i,j])
                    lons.append(lon[i,j])
                    
                    cell_count += 1
                    if cell_count % batch_size == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} RDRS grid cells")

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): lons,
            }, crs='EPSG:4326')
            
            # Add elevation data in batches
            gdf['elev_m'] = -9999  # Default value
            
            batch_size = 50
            num_batches = (len(gdf) + batch_size - 1) // batch_size
            
            self.logger.info(f"Calculating elevations in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gdf))
                
                self.logger.info(f"Processing elevation batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                
                try:
                    # Get batch of geometries
                    batch_gdf = gdf.iloc[start_idx:end_idx]
                    
                    # Calculate zonal statistics for this batch
                    zs = rasterstats.zonal_stats(batch_gdf, str(self.dem_path), stats=['mean'])
                    
                    # Update elevation values in the main GeoDataFrame
                    for i, item in enumerate(zs):
                        idx = start_idx + i
                        gdf.loc[idx, 'elev_m'] = item['mean'] if item['mean'] is not None else -9999
                except Exception as e:
                    self.logger.warning(f"Error calculating elevations for batch {batch_idx+1}: {str(e)}")
                    # Continue with next batch

            # Remove rows with invalid elevation values if requested
            if self.config.get('REMOVE_INVALID_ELEVATION_CELLS', False):
                valid_count = len(gdf)
                gdf = gdf[gdf['elev_m'] != -9999].copy()
                removed_count = valid_count - len(gdf)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} cells with invalid elevation values")

            # Save the shapefile
            output_shapefile.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"RDRS shapefile created and saved to {output_shapefile}")
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_rdrs_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _create_carra_shapefile(self):
        """Create CARRA shapefile with output file checking"""
        self.logger.info("Creating CARRA grid shapefile")
        
        # Define output shapefile path
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find a processed CARRA file
            carra_files = list(self.merged_forcing_path.glob('*.nc'))
            if not carra_files:
                raise FileNotFoundError("No processed CARRA files found")
            carra_file = carra_files[0]

            # Read CARRA data
            with xr.open_dataset(carra_file) as ds:
                lats = ds.latitude.values
                lons = ds.longitude.values

            self.logger.info(f"CARRA dimensions: {lats.shape}")
            
            # Define CARRA projection
            carra_proj = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs')
            wgs84 = pyproj.CRS('EPSG:4326')

            transformer = Transformer.from_crs(carra_proj, wgs84, always_xy=True)
            transformer_to_carra = Transformer.from_crs(wgs84, carra_proj, always_xy=True)

            # Create geometries in memory first to avoid file I/O in the loop
            self.logger.info("Creating CARRA grid cell geometries")
            
            geometries = []
            ids = []
            center_lats = []
            center_lons = []
            
            # Process in batches
            batch_size = 100
            total_cells = len(lons)
            num_batches = (total_cells + batch_size - 1) // batch_size
            
            self.logger.info(f"Creating {total_cells} CARRA grid cells in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(lons))
                
                self.logger.info(f"Processing grid cell batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                
                for i in range(start_idx, end_idx):
                    # Convert lat/lon to CARRA coordinates
                    x, y = transformer_to_carra.transform(lons[i], lats[i])
                    
                    # Define grid cell (assuming 2.5 km resolution)
                    half_dx = 1250  # meters
                    half_dy = 1250  # meters
                    
                    vertices = [
                        (x - half_dx, y - half_dy),
                        (x - half_dx, y + half_dy),
                        (x + half_dx, y + half_dy),
                        (x + half_dx, y - half_dy),
                        (x - half_dx, y - half_dy)
                    ]
                    
                    # Convert vertices back to lat/lon
                    lat_lon_vertices = [transformer.transform(vx, vy) for vx, vy in vertices]
                    
                    geometries.append(Polygon(lat_lon_vertices))
                    ids.append(i)
                    
                    center_lon, center_lat = transformer.transform(x, y)
                    center_lats.append(center_lat)
                    center_lons.append(center_lon)
            
            # Create GeoDataFrame
            self.logger.info("Creating GeoDataFrame")
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): center_lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): center_lons,
            }, crs='EPSG:4326')
            
            # Add elevation data in batches
            self.logger.info("Calculating elevation values")
            gdf['elev_m'] = -9999  # Default value
            
            batch_size = 50
            num_batches = (len(gdf) + batch_size - 1) // batch_size
            
            self.logger.info(f"Calculating elevations in {num_batches} batches")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(gdf))
                
                self.logger.info(f"Processing elevation batch {batch_idx+1}/{num_batches} ({start_idx} to {end_idx-1})")
                
                try:
                    # Get batch of geometries
                    batch_gdf = gdf.iloc[start_idx:end_idx]
                    
                    # Calculate zonal statistics for this batch
                    zs = rasterstats.zonal_stats(batch_gdf, str(self.dem_path), stats=['mean'])
                    
                    # Update elevation values in the main GeoDataFrame
                    for i, item in enumerate(zs):
                        idx = start_idx + i
                        gdf.loc[idx, 'elev_m'] = item['mean'] if item['mean'] is not None else -9999
                except Exception as e:
                    self.logger.warning(f"Error calculating elevations for batch {batch_idx+1}: {str(e)}")
                    # Continue with next batch

            # Save the shapefile
            self.logger.info(f"Saving CARRA shapefile to {output_shapefile}")
            gdf.to_file(output_shapefile)
            self.logger.info(f"CARRA grid shapefile created and saved to {output_shapefile}")
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_carra_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def remap_forcing(self):
        self.logger.info("Starting forcing remapping process")
        self._create_parallelized_weighted_forcing()
        self.logger.info("Forcing remapping process completed")

    def _determine_output_filename(self, input_file):
        """
        Determine the expected output filename for a given input file.
        This handles different forcing datasets with their specific naming patterns.
        
        Args:
            input_file (Path): Input forcing file path
        
        Returns:
            Path: Expected output file path
        """
        # Extract base information
        domain_name = self.config['DOMAIN_NAME']
        forcing_dataset = self.config['FORCING_DATASET']
        input_stem = input_file.stem
        
        # Handle RDRS and CASR specific naming patterns
        if forcing_dataset.lower() == 'rdrs':
            # For files like "RDRS_monthly_198001.nc", output should be "Canada_RDRS_remapped_RDRS_monthly_198001.nc"
            if input_stem.startswith('RDRS_monthly_'):
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
            else:
                # General fallback for other RDRS files
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        elif forcing_dataset.lower() == 'casr':
            # For files like "CASR_monthly_198001.nc", output should be "Canada_CASR_remapped_CASR_monthly_198001.nc"
            if input_stem.startswith('CASR_monthly_'):
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
            else:
                # General fallback for other CASR files
                output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        else:
            # General pattern for other datasets (ERA5, CARRA, etc.)
            output_filename = f"{domain_name}_{forcing_dataset}_remapped_{input_stem}.nc"
        
        return self.forcing_basin_path / output_filename

    def _create_parallelized_weighted_forcing(self):
        """Create weighted forcing files in parallel with improved file checking and memory management"""
        self.logger.info("Creating weighted forcing files in parallel")
        
        # Create output directories if they don't exist
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of forcing files
        forcing_path = self.merged_forcing_path
        forcing_files = sorted([f for f in forcing_path.glob('*.nc')])
        
        if not forcing_files:
            self.logger.warning("No forcing files found to process")
            return
        
        self.logger.info(f"Found {len(forcing_files)} forcing files to process")
        
        # Get number of CPUs to use
        num_cpus = min(int(self.config.get('MPI_PROCESSES', mp.cpu_count())), mp.cpu_count())
        if num_cpus <= 0:
            num_cpus = max(1, mp.cpu_count() // 2)  # Use half available CPUs as default to reduce memory pressure
        
        self.logger.info(f"Using {num_cpus} CPUs for parallel processing")
        
        # Filter out already processed files with more detailed naming check
        remaining_files = []
        already_processed = 0
        
        for file in forcing_files:
            # Determine expected output filename pattern more precisely
            output_file = self._determine_output_filename(file)
            
            if output_file.exists():
                # Check if file is valid (not corrupted/empty)
                try:
                    file_size = output_file.stat().st_size
                    if file_size > 1000:  # Basic size check to ensure file isn't corrupted
                        self.logger.debug(f"Skipping already processed file: {file.name} -> {output_file.name}")
                        already_processed += 1
                        continue
                    else:
                        self.logger.warning(f"Found potentially corrupted output file {output_file} (size: {file_size} bytes). Will reprocess.")
                except Exception as e:
                    self.logger.warning(f"Error checking output file {output_file}: {str(e)}. Will reprocess.")
            
            remaining_files.append(file)
        
        self.logger.info(f"Found {already_processed} already processed files")
        self.logger.info(f"Found {len(remaining_files)} files that need processing")
        
        if not remaining_files:
            self.logger.info("All files have already been processed, nothing to do")
            return
        
        # Process in smaller batches to avoid memory issues
        batch_size = min(10, len(remaining_files))  # Process 10 files at a time or fewer if there are fewer files
        total_batches = (len(remaining_files) + batch_size - 1) // batch_size
        
        self.logger.info(f"Processing {total_batches} batches of up to {batch_size} files each")
        
        success_count = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(remaining_files))
            batch_files = remaining_files[start_idx:end_idx]
            
            self.logger.info(f"Processing batch {batch_num+1}/{total_batches} with {len(batch_files)} files")
            
            # Process files in parallel using Pool
            with mp.Pool(processes=num_cpus) as pool:
                # Create a list of (file, worker_id) tuples to distribute work
                worker_assignments = [(file, i % num_cpus) for i, file in enumerate(batch_files)]
                
                # Map the processing function to each file with its worker ID
                results = pool.starmap(
                    self._process_forcing_file, 
                    worker_assignments
                )
            
            # Count successes in this batch
            batch_success = sum(1 for r in results if r)
            success_count += batch_success
            
            self.logger.info(f"Batch {batch_num+1}/{total_batches} complete: {batch_success}/{len(batch_files)} successful")
            
            # Optional: Force garbage collection between batches
            import gc
            gc.collect()
        
        # Report final results
        self.logger.info(f"Processing complete: {success_count} files processed successfully out of {len(remaining_files)}")
        self.logger.info(f"Total files processed or skipped: {success_count + already_processed} out of {len(forcing_files)}")

    def _process_forcing_file(self, file, worker_id):
        """Process a single forcing file for parallel execution with robust output checking"""
        try:
            start_time = time.time()
            
            # Check output file first before doing any processing
            output_file = self._determine_output_filename(file)
            
            if output_file.exists():
                try:
                    file_size = output_file.stat().st_size
                    if file_size > 1000:  # Basic size check to ensure file isn't corrupted
                        self.logger.info(f"Worker {worker_id}: Skipping already processed file {file.name}")
                        return True
                except Exception:
                    # If we can't check the file, we'll reprocess it
                    pass
            
            self.logger.info(f"Worker {worker_id}: Processing file {file.name}")
            
            # For CASR and RDRS, files are already processed during merging, no need for additional processing
            file_to_process = file
            
            # Define the output directory and remapped file name based on your configuration
            intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
            intersect_path.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique temp directory for this process
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = self.project_dir / 'forcing' / f'temp_easymore_{unique_id}_{worker_id}'
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup easymore configuration
            esmr = easymore.Easymore()
            
            esmr.author_name = 'SUMMA public workflow scripts'
            esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
            esmr.case_name = f"{self.config['DOMAIN_NAME']}_{self.config['FORCING_DATASET']}"
            
            esmr.source_shp = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self.config['FORCING_DATASET']}.shp"
            esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
            esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')
            
            esmr.target_shp = self.catchment_path / self.catchment_name
            esmr.target_shp_ID = self.config.get('CATCHMENT_SHP_HRUID')
            esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
            esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')
            
            # Set coordinate variable names based on forcing dataset
            if self.forcing_dataset in ['rdrs', 'casr']:
                var_lat = 'lat' 
                var_lon = 'lon'
            else:  # era5, carra, etc.
                var_lat = 'latitude'
                var_lon = 'longitude'
            
            esmr.source_nc = str(file_to_process)
            esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
            esmr.var_lat = var_lat
            esmr.var_lon = var_lon
            esmr.var_time = 'time'
            
            esmr.temp_dir = str(temp_dir) + '/'
            esmr.output_dir = str(self.forcing_basin_path) + '/'
            
            esmr.remapped_dim_id = 'hru'
            esmr.remapped_var_id = 'hruId'
            esmr.format_list = ['f4']
            esmr.fill_value_list = ['-9999']
            
            esmr.save_csv = False
            esmr.remap_csv = ''
            esmr.sort_ID = False
            
            # Only create the remap file if it doesn't exist yet
            remap_file = f"{esmr.case_name}_remapping.nc"
            if not (intersect_path / remap_file).exists():
                try:
                    esmr.nc_remapper()
                    
                    # Move the remap file to the intersection path
                    if os.path.exists(os.path.join(esmr.temp_dir, remap_file)):
                        os.rename(os.path.join(esmr.temp_dir, remap_file), intersect_path / remap_file)
                        
                    # Move the shapefile files
                    for shp_file in Path(esmr.temp_dir).glob(f"{esmr.case_name}_intersected_shapefile.*"):
                        os.rename(shp_file, intersect_path / shp_file.name)
                except Exception as e:
                    self.logger.error(f"Worker {worker_id}: Error in creating remap file: {str(e)}")
            else:
                # Use existing remap file
                esmr.remap_csv = str(intersect_path / remap_file)
                esmr.nc_remapper()
            
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.warning(f"Worker {worker_id}: Failed to clean up temp files: {str(e)}")
                
            # Verify output file exists
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size > 1000:  # Basic size check
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Worker {worker_id}: Successfully processed {file.name} in {elapsed_time:.2f} seconds")
                    return True
                else:
                    self.logger.error(f"Worker {worker_id}: Output file {output_file} exists but may be corrupted (size: {file_size} bytes)")
                    return False
            else:
                self.logger.error(f"Worker {worker_id}: Expected output file {output_file} was not created")
                # Let's check if the file might exist with a different naming convention
                possible_files = list(self.forcing_basin_path.glob(f"*{file.stem}*"))
                if possible_files:
                    self.logger.info(f"Worker {worker_id}: Found possible matching files: {[f.name for f in possible_files]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: Error processing {file.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


class geospatialStatistics:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{str(self.config['DOMAIN_DISCRETIZATION']).replace(',','_')}.shp"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass')

    def _get_file_path(self, file_type, file_def_path):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        """Calculate elevation statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing elevation statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing elevation statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating elevation statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        nodata_value = self.get_nodata_value(self.dem_path)

        # Open the DEM and calculate statistics
        try:
            with rasterio.open(self.dem_path) as src:
                affine = src.transform
                dem_data = src.read(1)

            stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
            result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
            
            if 'elev_mean' in catchment_gdf.columns:
                self.logger.info("Updating existing 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']
            else:
                self.logger.info("Adding new 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']

            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Elevation statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating elevation statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_soil_stats(self):
        """Calculate soil statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one USGS soil class column
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing soil statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing soil statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name
        
        try:
            nodata_value = self.get_nodata_value(soil_raster)

            with rasterio.open(soil_raster) as src:
                affine = src.transform
                soil_data = src.read(1)

            stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Soil statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating soil statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_land_stats(self):
        """Calculate land statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one IGBP land class column
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing land statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing land statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"
        land_raster = self.land_path / land_name
        
        try:
            nodata_value = self.get_nodata_value(land_raster)

            with rasterio.open(land_raster) as src:
                affine = src.transform
                land_data = src.read(1)

            stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Land statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating land statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def run_statistics(self):
        """Run all geospatial statistics with checks for existing outputs"""
        self.logger.info("Starting geospatial statistics calculation")
        
        # Count how many steps we're skipping
        skipped = 0
        total = 3  # Total number of statistics operations
        
        # Check soil stats
        intersect_soil_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'

        soil_output_file = intersect_soil_path / intersect_soil_name
        
        if soil_output_file.exists():
            try:
                gdf = gpd.read_file(soil_output_file)
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics already calculated: {soil_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 1:
            self.calculate_soil_stats()
        
        # Check land stats
        intersect_land_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_land_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_land_name == 'default':
            intersect_land_name = 'catchment_with_landclass.shp'

        land_output_file = intersect_land_path / intersect_land_name
        
        if land_output_file.exists():
            try:
                gdf = gpd.read_file(land_output_file)
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics already calculated: {land_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 2:
            self.calculate_land_stats()
        
        # Check elevation stats
        intersect_dem_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_dem_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_dem_name == 'default':
            intersect_dem_name = 'catchment_with_dem.shp'

        dem_output_file = intersect_dem_path / intersect_dem_name
        
        if dem_output_file.exists():
            try:
                gdf = gpd.read_file(dem_output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics already calculated: {dem_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 3:
            self.calculate_elevation_stats()
        
        self.logger.info(f"Geospatial statistics completed: {skipped}/{total} steps skipped, {total-skipped}/{total} steps executed")

class geospatialStatistics:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{str(self.config['DOMAIN_DISCRETIZATION']).replace(',','_')}.shp"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass')

    def _get_file_path(self, file_type, file_def_path):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        """Calculate elevation statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing elevation statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing elevation statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating elevation statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        nodata_value = self.get_nodata_value(self.dem_path)

        # Open the DEM and calculate statistics
        try:
            with rasterio.open(self.dem_path) as src:
                affine = src.transform
                dem_data = src.read(1)

            stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
            result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
            
            if 'elev_mean' in catchment_gdf.columns:
                self.logger.info("Updating existing 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']
            else:
                self.logger.info("Adding new 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']

            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Elevation statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating elevation statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_soil_stats(self):
        """Calculate soil statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one USGS soil class column
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing soil statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing soil statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name
        
        try:
            nodata_value = self.get_nodata_value(soil_raster)

            with rasterio.open(soil_raster) as src:
                affine = src.transform
                soil_data = src.read(1)

            stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Soil statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating soil statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_land_stats(self):
        """Calculate land statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one IGBP land class column
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing land statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing land statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"
        land_raster = self.land_path / land_name
        
        try:
            nodata_value = self.get_nodata_value(land_raster)

            with rasterio.open(land_raster) as src:
                affine = src.transform
                land_data = src.read(1)

            stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Land statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating land statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def run_statistics(self):
        """Run all geospatial statistics with checks for existing outputs"""
        self.logger.info("Starting geospatial statistics calculation")
        
        # Count how many steps we're skipping
        skipped = 0
        total = 3  # Total number of statistics operations
        
        # Check soil stats
        intersect_soil_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'

        soil_output_file = intersect_soil_path / intersect_soil_name
        
        if soil_output_file.exists():
            try:
                gdf = gpd.read_file(soil_output_file)
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics already calculated: {soil_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 1:
            self.calculate_soil_stats()
        
        # Check land stats
        intersect_land_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_land_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_land_name == 'default':
            intersect_land_name = 'catchment_with_landclass.shp'

        land_output_file = intersect_land_path / intersect_land_name
        
        if land_output_file.exists():
            try:
                gdf = gpd.read_file(land_output_file)
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics already calculated: {land_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 2:
            self.calculate_land_stats()
        
        # Check elevation stats
        intersect_dem_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_dem_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_dem_name == 'default':
            intersect_dem_name = 'catchment_with_dem.shp'

        dem_output_file = intersect_dem_path / intersect_dem_name
        
        if dem_output_file.exists():
            try:
                gdf = gpd.read_file(dem_output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics already calculated: {dem_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 3:
            self.calculate_elevation_stats()
        
        self.logger.info(f"Geospatial statistics completed: {skipped}/{total} steps skipped, {total-skipped}/{total} steps executed")

class geospatialStatistics:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{str(self.config['DOMAIN_DISCRETIZATION']).replace(',','_')}.shp"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass')

    def _get_file_path(self, file_type, file_def_path):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        """Calculate elevation statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing elevation statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing elevation statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating elevation statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        nodata_value = self.get_nodata_value(self.dem_path)

        # Open the DEM and calculate statistics
        try:
            with rasterio.open(self.dem_path) as src:
                affine = src.transform
                dem_data = src.read(1)

            stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
            result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
            
            if 'elev_mean' in catchment_gdf.columns:
                self.logger.info("Updating existing 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']
            else:
                self.logger.info("Adding new 'elev_mean' column")
                catchment_gdf['elev_mean'] = result_df['elev_mean_new']

            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Elevation statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating elevation statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_soil_stats(self):
        """Calculate soil statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one USGS soil class column
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing soil statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing soil statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name
        
        try:
            nodata_value = self.get_nodata_value(soil_raster)

            with rasterio.open(soil_raster) as src:
                affine = src.transform
                soil_data = src.read(1)

            stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Soil statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating soil statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def calculate_land_stats(self):
        """Calculate land statistics with output file checking"""
        # Get the output path and check if the file already exists
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'

        output_file = intersect_path / intersect_name
        
        # Check if output already exists
        if output_file.exists():
            try:
                # Verify the file is valid
                gdf = gpd.read_file(output_file)
                # Check for at least one IGBP land class column
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics file already exists: {output_file}. Skipping calculation.")
                    return
                else:
                    self.logger.info(f"Existing land statistics file {output_file} does not contain expected data. Recalculating.")
            except Exception as e:
                self.logger.warning(f"Error checking existing land statistics file: {str(e)}. Recalculating.")
        
        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"
        land_raster = self.land_path / land_name
        
        try:
            nodata_value = self.get_nodata_value(land_raster)

            with rasterio.open(land_raster) as src:
                affine = src.transform
                land_data = src.read(1)

            stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats).fillna(0)

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            for col in result_df.columns:
                if col != 'count':
                    result_df[col] = result_df[col].astype(int)

            catchment_gdf = catchment_gdf.join(result_df)
            
            # Create output directory and save the file
            intersect_path.mkdir(parents=True, exist_ok=True)
            catchment_gdf.to_file(output_file)
            
            self.logger.info(f"Land statistics saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error calculating land statistics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def run_statistics(self):
        """Run all geospatial statistics with checks for existing outputs"""
        self.logger.info("Starting geospatial statistics calculation")
        
        # Count how many steps we're skipping
        skipped = 0
        total = 3  # Total number of statistics operations
        
        # Check soil stats
        intersect_soil_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'

        soil_output_file = intersect_soil_path / intersect_soil_name
        
        if soil_output_file.exists():
            try:
                gdf = gpd.read_file(soil_output_file)
                usgs_cols = [col for col in gdf.columns if col.startswith('USGS_')]
                if len(usgs_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Soil statistics already calculated: {soil_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 1:
            self.calculate_soil_stats()
        
        # Check land stats
        intersect_land_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_land_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_land_name == 'default':
            intersect_land_name = 'catchment_with_landclass.shp'

        land_output_file = intersect_land_path / intersect_land_name
        
        if land_output_file.exists():
            try:
                gdf = gpd.read_file(land_output_file)
                igbp_cols = [col for col in gdf.columns if col.startswith('IGBP_')]
                if len(igbp_cols) > 0 and len(gdf) > 0:
                    self.logger.info(f"Land statistics already calculated: {land_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 2:
            self.calculate_land_stats()
        
        # Check elevation stats
        intersect_dem_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_dem_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_dem_name == 'default':
            intersect_dem_name = 'catchment_with_dem.shp'

        dem_output_file = intersect_dem_path / intersect_dem_name
        
        if dem_output_file.exists():
            try:
                gdf = gpd.read_file(dem_output_file)
                if 'elev_mean' in gdf.columns and len(gdf) > 0:
                    self.logger.info(f"Elevation statistics already calculated: {dem_output_file}")
                    skipped += 1
            except Exception:
                pass
        
        if skipped < 3:
            self.calculate_elevation_stats()
        
        self.logger.info(f"Geospatial statistics completed: {skipped}/{total} steps skipped, {total-skipped}/{total} steps executed")