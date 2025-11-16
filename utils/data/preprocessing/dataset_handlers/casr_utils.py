"""
CASR Dataset Handler for SYMFLUENCE

This module provides the CASR (Canadian Arctic System Reanalysis) specific implementation
for forcing data processing. It handles CASR variable mappings, unit conversions,
grid structure, and shapefile creation.
"""

from pathlib import Path
from typing import Dict, Tuple
import time
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('casr')
class CASRHandler(BaseDatasetHandler):
    """Handler for CASR (Canadian Arctic System Reanalysis) dataset."""
    
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        CASR variable name mapping to standard names.
        
        Returns:
            Dictionary mapping CASR variable names to standard names
        """
        return {
            # Temperature (prefer analysis A_ over forecast P_)
            'CaSR_v3.1_A_TT_1.5m': 'airtemp',
            'CaSR_v3.1_P_TT_1.5m': 'airtemp',
            
            # Precipitation
            'CaSR_v3.1_A_PR0_SFC': 'pptrate',
            'CaSR_v3.1_P_PR0_SFC': 'pptrate',
            
            # Pressure
            'CaSR_v3.1_P_P0_SFC': 'airpres',
            
            # Humidity
            'CaSR_v3.1_P_HU_1.5m': 'spechum',
            
            # Wind speed
            'CaSR_v3.1_P_UVC_10m': 'windspd',
            
            # Wind components
            'CaSR_v3.1_P_UUC_10m': 'windspd_u',
            'CaSR_v3.1_P_VVC_10m': 'windspd_v',
            
            # Radiation
            'CaSR_v3.1_P_FB_SFC': 'SWRadAtm',
            'CaSR_v3.1_P_FI_SFC': 'LWRadAtm',
        }
    
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CASR dataset with variable renaming and unit conversions.
        
        Unit conversions applied:
        - airpres: mb -> Pa (multiply by 100)
        - airtemp: °C -> K (add 273.15)
        - pptrate: m/hr -> m/s (multiply by 1000, divide by 3600)
        - windspd: knots -> m/s (multiply by 0.514444)
        
        Args:
            ds: Input CASR dataset
            
        Returns:
            Processed dataset with standardized variables and units
        """
        # Rename variables
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
        ds = ds.rename(existing_vars)
        
        # Apply unit conversions with clean attributes
        if 'airpres' in ds:
            ds['airpres'] = ds['airpres'] * 100
            ds['airpres'].attrs = {}
            ds['airpres'].attrs.update({
                'units': 'Pa', 
                'long_name': 'air pressure', 
                'standard_name': 'air_pressure'
            })
        
        if 'airtemp' in ds:
            ds['airtemp'] = ds['airtemp'] + 273.15
            ds['airtemp'].attrs = {}
            ds['airtemp'].attrs.update({
                'units': 'K', 
                'long_name': 'air temperature', 
                'standard_name': 'air_temperature'
            })
        
        if 'pptrate' in ds:
            ds['pptrate'] = ds['pptrate'] * 1000 / 3600
            ds['pptrate'].attrs = {}
            ds['pptrate'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'precipitation rate', 
                'standard_name': 'precipitation_rate'
            })
        
        if 'windspd' in ds:
            ds['windspd'] = ds['windspd'] * 0.514444
            ds['windspd'].attrs = {}
            ds['windspd'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'wind speed', 
                'standard_name': 'wind_speed'
            })
        
        if 'windspd_u' in ds:
            ds['windspd_u'] = ds['windspd_u'] * 0.514444
            ds['windspd_u'].attrs = {}
            ds['windspd_u'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'eastward wind', 
                'standard_name': 'eastward_wind'
            })
        
        if 'windspd_v' in ds:
            ds['windspd_v'] = ds['windspd_v'] * 0.514444
            ds['windspd_v'].attrs = {}
            ds['windspd_v'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'northward wind', 
                'standard_name': 'northward_wind'
            })
        
        # Radiation variables are already in W m**-2, just update attributes
        if 'LWRadAtm' in ds:
            ds['LWRadAtm'].attrs = {}
            ds['LWRadAtm'].attrs.update({
                'long_name': 'downward longwave radiation at the surface', 
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            })
        
        if 'SWRadAtm' in ds:
            ds['SWRadAtm'].attrs = {}
            ds['SWRadAtm'].attrs.update({
                'long_name': 'downward shortwave radiation at the surface', 
                'standard_name': 'surface_downwelling_shortwave_flux_in_air'
            })
        
        if 'spechum' in ds:
            ds['spechum'].attrs = {}
            ds['spechum'].attrs.update({
                'long_name': 'specific humidity', 
                'standard_name': 'specific_humidity'
            })
        
        return ds
    
    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CASR uses rotated pole coordinates with auxiliary lat/lon.
        
        Returns:
            Tuple of ('lat', 'lon') for auxiliary coordinates
        """
        return ('lat', 'lon')
    
    def needs_merging(self) -> bool:
        """CASR requires merging of daily files into monthly files."""
        return True
    
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Merge CASR forcing data files into monthly files.
        
        Note: CASR files are all in the same directory, not organized by year.
        
        Args:
            raw_forcing_path: Path to raw CASR data (all files in one directory)
            merged_forcing_path: Path where merged monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info("Starting to merge CASR forcing data")
        
        years = range(start_year - 1, end_year + 1)
        merged_forcing_path.mkdir(parents=True, exist_ok=True)
        
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
                
                # Look for daily CASR files for this month
                month_pattern = f"domain_{self.domain_name}_{year}{month:02d}"
                daily_files = sorted([f for f in all_casr_files if month_pattern in f.name])
                
                if not daily_files:
                    self.logger.warning(f"No CASR files found for {year}-{month:02d}")
                    continue
                
                self.logger.info(f"Found {len(daily_files)} CASR files for {year}-{month:02d}")
                
                # Load datasets
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
                
                # Process each dataset
                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = self.process_dataset(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:
                        self.logger.error(f"Error processing CASR dataset: {str(e)}")
                
                if not processed_datasets:
                    self.logger.warning(f"No processed CASR datasets for {year}-{month:02d}")
                    continue
                
                # Concatenate into monthly data
                monthly_data = xr.concat(processed_datasets, dim="time")
                monthly_data = monthly_data.sortby("time")
                
                # Set up time range
                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)
                
                monthly_data = monthly_data.sel(time=slice(start_time, end_time))
                
                # Ensure complete hourly time series
                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times, method='nearest')
                
                # Set time encoding
                monthly_data = self.setup_time_encoding(monthly_data)
                
                # Add metadata
                monthly_data = self.add_metadata(
                    monthly_data, 
                    'CASR data aggregated to monthly files and variables renamed for SUMMA compatibility'
                )
                
                # Aggressively clean up variable attributes and encoding
                for var_name in monthly_data.data_vars:
                    var = monthly_data[var_name]
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
                        var.attrs = {'missing_value': -999}
                
                # Save monthly file
                output_file = merged_forcing_path / f"CASR_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                self.logger.info(f"Saved CASR monthly file: {output_file}")
                
                # Clean up
                for ds in datasets:
                    ds.close()
        
        self.logger.info("CASR forcing data merging completed")
    
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create CASR grid shapefile.
        
        CASR uses a rotated pole grid similar to RDRS.
        
        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to merged CASR data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics
            
        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating CASR grid shapefile")
        
        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find a CASR file to get grid information
            casr_files = list(merged_forcing_path.glob('*.nc'))
            if not casr_files:
                raise FileNotFoundError("No CASR files found")
            casr_file = casr_files[0]
            
            self.logger.info(f"Using CASR file for grid: {casr_file}")
            
            # Read CASR data - similar structure to RDRS
            with xr.open_dataset(casr_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values
            
            self.logger.info(f"CASR dimensions: rlat={rlat.shape}, rlon={rlon.shape}")
            
            # Create grid cells
            geometries, ids, lats, lons = [], [], [], []
            
            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size
            
            self.logger.info(f"Creating CASR grid cells in {num_batches} batches")
            
            cell_count = 0
            for i in range(len(rlat)):
                for j in range(len(rlon)):
                    # Create grid cell corners
                    rlat_corners = [
                        rlat[i], rlat[i], 
                        rlat[i+1] if i+1 < len(rlat) else rlat[i], 
                        rlat[i+1] if i+1 < len(rlat) else rlat[i]
                    ]
                    rlon_corners = [
                        rlon[j], 
                        rlon[j+1] if j+1 < len(rlon) else rlon[j], 
                        rlon[j+1] if j+1 < len(rlon) else rlon[j], 
                        rlon[j]
                    ]
                    
                    # Get actual lat/lon corners
                    lat_corners = [
                        lat[i,j], 
                        lat[i, j+1] if j+1 < len(rlon) else lat[i,j],
                        lat[i+1, j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j],
                        lat[i+1, j] if i+1 < len(rlat) else lat[i,j]
                    ]
                    lon_corners = [
                        lon[i,j], 
                        lon[i, j+1] if j+1 < len(rlon) else lon[i,j],
                        lon[i+1, j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j],
                        lon[i+1, j] if i+1 < len(rlat) else lon[i,j]
                    ]
                    
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
            
            # Calculate elevation
            self.logger.info("Calculating elevation values using safe method")
            elevations = elevation_calculator(gdf, dem_path, batch_size=50)
            gdf['elev_m'] = elevations
            
            # Remove invalid elevation cells if requested
            if self.config.get('REMOVE_INVALID_ELEVATION_CELLS', False):
                valid_count = len(gdf)
                gdf = gdf[gdf['elev_m'] != -9999].copy()
                removed_count = valid_count - len(gdf)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} cells with invalid elevation values")
            
            # Save shapefile
            output_shapefile.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_shapefile)
            self.logger.info(f"CASR shapefile created and saved to {output_shapefile}")
            
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_casr_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
