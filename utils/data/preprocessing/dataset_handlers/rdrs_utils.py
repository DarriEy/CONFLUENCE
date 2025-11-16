"""
RDRS Dataset Handler for SYMFLUENCE

This module provides the RDRS-specific implementation for forcing data processing.
It handles RDRS variable mappings, unit conversions, grid structure, and shapefile creation.
"""

from pathlib import Path
from typing import Dict, Tuple
import os
import time
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('rdrs')
class RDRSHandler(BaseDatasetHandler):
    """Handler for RDRS (Regional Deterministic Reforecast System) dataset."""
    
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        RDRS variable name mapping to standard names.
        
        Returns:
            Dictionary mapping RDRS variable names to standard names
        """
        return {
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
    
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process RDRS dataset with variable renaming and unit conversions.
        
        Unit conversions applied:
        - airpres: mb -> Pa (multiply by 100)
        - airtemp: °C -> K (add 273.15)
        - pptrate: mm/hr -> m/s (divide by 3600)
        - windspd: knots -> m/s (multiply by 0.514444)
        
        Args:
            ds: Input RDRS dataset
            
        Returns:
            Processed dataset with standardized variables and units
        """
        # Rename variables
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
        ds = ds.rename(existing_vars)
        
        # Apply unit conversions
        if 'airpres' in ds:
            ds['airpres'] = ds['airpres'] * 100
            ds['airpres'].attrs.update({
                'units': 'Pa', 
                'long_name': 'air pressure', 
                'standard_name': 'air_pressure'
            })
        
        if 'airtemp' in ds:
            ds['airtemp'] = ds['airtemp'] + 273.15
            ds['airtemp'].attrs.update({
                'units': 'K', 
                'long_name': 'air temperature', 
                'standard_name': 'air_temperature'
            })
        
        if 'pptrate' in ds:
            ds['pptrate'] = ds['pptrate'] / 3600
            ds['pptrate'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'precipitation rate', 
                'standard_name': 'precipitation_rate'
            })
        
        if 'windspd' in ds:
            ds['windspd'] = ds['windspd'] * 0.514444
            ds['windspd'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'wind speed', 
                'standard_name': 'wind_speed'
            })
        
        if 'LWRadAtm' in ds:
            ds['LWRadAtm'].attrs.update({
                'long_name': 'downward longwave radiation at the surface', 
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            })
        
        if 'SWRadAtm' in ds:
            ds['SWRadAtm'].attrs.update({
                'long_name': 'downward shortwave radiation at the surface', 
                'standard_name': 'surface_downwelling_shortwave_flux_in_air'
            })
        
        if 'spechum' in ds:
            ds['spechum'].attrs.update({
                'long_name': 'specific humidity', 
                'standard_name': 'specific_humidity'
            })
        
        return ds
    
    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        RDRS uses rotated pole coordinates with auxiliary lat/lon.
        
        Returns:
            Tuple of ('lat', 'lon') for auxiliary coordinates
        """
        return ('lat', 'lon')
    
    def needs_merging(self) -> bool:
        """RDRS requires merging of daily files into monthly files."""
        return True
    
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        Merge RDRS forcing data files into monthly files.
        
        Args:
            raw_forcing_path: Path to raw RDRS data organized by year
            merged_forcing_path: Path where merged monthly files will be saved
            start_year: Start year for processing
            end_year: End year for processing
        """
        self.logger.info("Starting to merge RDRS forcing data")
        
        years = range(start_year - 1, end_year + 1)
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        
        merged_forcing_path.mkdir(parents=True, exist_ok=True)
        
        for year in years:
            self.logger.info(f"Processing RDRS year {year}")
            year_folder = raw_forcing_path / str(year)
            
            if not year_folder.exists():
                self.logger.warning(f"Year folder not found: {year_folder}")
                continue
            
            for month in range(1, 13):
                self.logger.info(f"Processing RDRS {year}-{month:02d}")
                
                # Find daily files for this month
                daily_files = sorted(year_folder.glob(
                    file_name_pattern.replace('*', f'{year}{month:02d}*')
                ))
                
                if not daily_files:
                    self.logger.warning(f"No RDRS files found for {year}-{month:02d}")
                    continue
                
                # Load datasets
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
                
                # Process each dataset
                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = self.process_dataset(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:
                        self.logger.error(f"Error processing RDRS dataset: {str(e)}")
                
                if not processed_datasets:
                    self.logger.warning(f"No processed RDRS datasets for {year}-{month:02d}")
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
                
                # Set time encoding and metadata
                monthly_data = self.setup_time_encoding(monthly_data)
                monthly_data = self.add_metadata(
                    monthly_data, 
                    'RDRS data aggregated to monthly files and variables renamed for SUMMA compatibility'
                )
                monthly_data = self.clean_variable_attributes(monthly_data)
                
                # Save monthly file
                output_file = merged_forcing_path / f"RDRS_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)
                
                # Clean up
                for ds in datasets:
                    ds.close()
        
        self.logger.info("RDRS forcing data merging completed")
    
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create RDRS grid shapefile.
        
        RDRS uses a rotated pole grid with auxiliary lat/lon coordinates.
        
        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to merged RDRS data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics
            
        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating RDRS grid shapefile")
        
        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find an RDRS file to get grid information
            forcing_file = next(
                (f for f in os.listdir(merged_forcing_path) 
                 if f.endswith('.nc') and f.startswith('RDRS_monthly_')), 
                None
            )
            
            if not forcing_file:
                self.logger.error("No RDRS monthly file found")
                raise FileNotFoundError("No RDRS monthly file found")
            
            # Read grid information
            with xr.open_dataset(merged_forcing_path / forcing_file) as ds:
                rlat, rlon = ds.rlat.values, ds.rlon.values
                lat, lon = ds.lat.values, ds.lon.values
            
            self.logger.info(f"RDRS dimensions: rlat={rlat.shape}, rlon={rlon.shape}")
            
            # Create grid cells
            geometries, ids, lats, lons = [], [], [], []
            
            batch_size = 100
            total_cells = len(rlat) * len(rlon)
            num_batches = (total_cells + batch_size - 1) // batch_size
            
            self.logger.info(f"Creating RDRS grid cells in {num_batches} batches")
            
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
                        self.logger.info(f"Created {cell_count}/{total_cells} RDRS grid cells")
            
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
            self.logger.info(f"RDRS shapefile created and saved to {output_shapefile}")
            
            return output_shapefile
            
        except Exception as e:
            self.logger.error(f"Error in create_rdrs_shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
