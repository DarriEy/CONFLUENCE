"""
ERA5 Dataset Handler for SYMFLUENCE

This module provides the ERA5-specific implementation for forcing data processing.
ERA5 uses regular lat/lon grids and typically doesn't require merging operations.
"""

from pathlib import Path
from typing import Dict, Tuple
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('era5')
class ERA5Handler(BaseDatasetHandler):
    """Handler for ERA5 (ECMWF Reanalysis v5) dataset."""
    
    def get_variable_mapping(self) -> Dict[str, str]:
        """
        ERA5 variable name mapping to standard names.
        
        ERA5 data is typically already downloaded with standard names,
        but this allows for flexibility in case custom naming is used.
        
        Returns:
            Dictionary mapping ERA5 variable names to standard names
        """
        return {
            # ERA5 typically uses standard names already
            't2m': 'airtemp',
            'tp': 'pptrate',
            'sp': 'airpres',
            'q': 'spechum',
            'u10': 'windspd_u',
            'v10': 'windspd_v',
            'ws10': 'windspd',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
        }
    
    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process ERA5 dataset with variable renaming if needed.
        
        ERA5 data typically comes in standard units, but this method
        handles any necessary conversions.
        
        Args:
            ds: Input ERA5 dataset
            
        Returns:
            Processed dataset with standardized variables
        """
        # ERA5 data is typically already in correct format
        # but we can apply mapping if needed
        variable_mapping = self.get_variable_mapping()
        existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
        
        if existing_vars:
            ds = ds.rename(existing_vars)
        
        # ERA5 typically has correct units, but ensure attributes are set
        if 'airpres' in ds:
            if 'units' not in ds['airpres'].attrs or ds['airpres'].attrs['units'] != 'Pa':
                ds['airpres'].attrs.update({
                    'units': 'Pa', 
                    'long_name': 'air pressure', 
                    'standard_name': 'air_pressure'
                })
        
        if 'airtemp' in ds:
            if 'units' not in ds['airtemp'].attrs or ds['airtemp'].attrs['units'] != 'K':
                ds['airtemp'].attrs.update({
                    'units': 'K', 
                    'long_name': 'air temperature', 
                    'standard_name': 'air_temperature'
                })
        
        if 'pptrate' in ds:
            ds['pptrate'].attrs.update({
                'units': 'm s-1', 
                'long_name': 'precipitation rate', 
                'standard_name': 'precipitation_rate'
            })
        
        if 'windspd' in ds:
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
        ERA5 uses standard latitude/longitude coordinates.
        
        Returns:
            Tuple of ('latitude', 'longitude')
        """
        return ('latitude', 'longitude')
    
    def needs_merging(self) -> bool:
        """ERA5 data typically doesn't require merging."""
        return False
    
    def merge_forcings(self, raw_forcing_path: Path, merged_forcing_path: Path,
                      start_year: int, end_year: int) -> None:
        """
        ERA5 typically doesn't require merging.
        
        This method is a no-op for ERA5 but is required by the interface.
        """
        self.logger.info("ERA5 data does not require merging. Skipping merge step.")
        pass
    
    def create_shapefile(self, shapefile_path: Path, merged_forcing_path: Path,
                        dem_path: Path, elevation_calculator) -> Path:
        """
        Create ERA5 grid shapefile.
        
        ERA5 uses a regular lat/lon grid (typically 0.25° resolution).
        
        Args:
            shapefile_path: Directory where shapefile should be saved
            merged_forcing_path: Path to ERA5 data
            dem_path: Path to DEM for elevation calculation
            elevation_calculator: Function to calculate elevation statistics
            
        Returns:
            Path to the created shapefile
        """
        self.logger.info("Creating ERA5 shapefile")
        
        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        
        try:
            # Find an .nc file in the forcing path
            forcing_files = list(merged_forcing_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No ERA5 forcing files found")
            
            forcing_file = forcing_files[0]
            self.logger.info(f"Using ERA5 file: {forcing_file}")
            
            # Set the dimension variable names
            source_name_lat = "latitude"
            source_name_lon = "longitude"
            
            # Open the file and get the dimensions
            try:
                with xr.open_dataset(forcing_file) as src:
                    lat = src[source_name_lat].values
                    lon = src[source_name_lon].values
                
                self.logger.info(f"ERA5 dimensions: lat={lat.shape}, lon={lon.shape}")
            except Exception as e:
                self.logger.error(f"Error reading ERA5 dimensions: {str(e)}")
                raise
            
            # Find the grid spacing
            try:
                half_dlat = abs(lat[1] - lat[0])/2 if len(lat) > 1 else 0.125
                half_dlon = abs(lon[1] - lon[0])/2 if len(lon) > 1 else 0.125
                
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
            
            # Calculate elevation using the safe method
            try:
                self.logger.info("Calculating elevation values using safe method")
                
                if not Path(dem_path).exists():
                    self.logger.error(f"DEM file not found: {dem_path}")
                    raise FileNotFoundError(f"DEM file not found: {dem_path}")
                
                elevations = elevation_calculator(gdf, dem_path, batch_size=20)
                gdf['elev_m'] = elevations
                
                self.logger.info(f"Elevation calculation complete")
            except Exception as e:
                self.logger.error(f"Error calculating elevation: {str(e)}")
                # Continue without elevation data rather than failing completely
                gdf['elev_m'] = -9999
                self.logger.warning("Using default elevation values due to calculation error")
            
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
