from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
from typing import Dict, Any, List, Optional
from rasterstats import zonal_stats
import rasterio
from osgeo import gdal

from scipy.stats import skew, kurtosis, circmean, circstd
from shapely.geometry import box
from rasterio.mask import mask

class attributeProcessor:
    """
    Simple attribute processor that calculates elevation, slope, and aspect 
    statistics from an existing DEM using GDAL.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the attribute processor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.logger.info(f'data dir: {self.data_dir}')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Create or access directories
        self.dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        self.slope_dir = self.project_dir / 'attributes' / 'elevation' / 'slope'
        self.aspect_dir = self.project_dir / 'attributes' / 'elevation' / 'aspect'
        
        # Create directories if they don't exist
        self.slope_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()
        
        # Initialize results dictionary
        self.results = {}
    
    def _get_catchment_path(self) -> Path:
        """Get the path to the catchment shapefile."""
        catchment_path = self.config.get('CATCHMENT_PATH')
        self.logger.info(f'catchment path: {catchment_path}')
        
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        self.logger.info(f'catchment name: {catchment_name}')
        

        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)
        
        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name
        
        return catchment_path / catchment_file
    
    def find_dem_file(self) -> Path:
        """Find the DEM file in the elevation/dem directory."""
        dem_files = list(self.dem_dir.glob("*.tif"))
        
        if not dem_files:
            self.logger.error(f"No DEM files found in {self.dem_dir}")
            raise FileNotFoundError(f"No DEM files found in {self.dem_dir}")
        
        # Use the first found DEM file
        return dem_files[0]
    
    def generate_slope_and_aspect(self, dem_file: Path) -> Dict[str, Path]:
        """Generate slope and aspect rasters from the DEM using GDAL."""
        self.logger.info(f"Generating slope and aspect from DEM: {dem_file}")
        
        # Create output file paths
        slope_file = self.slope_dir / f"{dem_file.stem}_slope.tif"
        aspect_file = self.aspect_dir / f"{dem_file.stem}_aspect.tif"
        
        try:
            # Prepare the slope options
            slope_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                slopeFormat='degree',
                alg='Horn'
            )
            
            # Generate slope
            self.logger.info("Calculating slope...")
            gdal.DEMProcessing(
                str(slope_file),
                str(dem_file),
                'slope',
                options=slope_options
            )
            
            # Prepare the aspect options
            aspect_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                alg='Horn',
                zeroForFlat=True
            )
            
            # Generate aspect
            self.logger.info("Calculating aspect...")
            gdal.DEMProcessing(
                str(aspect_file),
                str(dem_file),
                'aspect',
                options=aspect_options
            )
            
            self.logger.info(f"Slope saved to: {slope_file}")
            self.logger.info(f"Aspect saved to: {aspect_file}")
            
            return {
                'dem': dem_file,
                'slope': slope_file,
                'aspect': aspect_file
            }
        
        except Exception as e:
            self.logger.error(f"Error generating slope and aspect: {str(e)}")
            raise
    
    def calculate_statistics(self, raster_file: Path, attribute_name: str) -> Dict[str, float]:
        """Calculate zonal statistics for a raster."""
        self.logger.info(f"Calculating statistics for {attribute_name} from {raster_file}")
        
        # Define statistics to calculate
        stats = ['min', 'mean', 'max', 'std']
        
        # Special handling for aspect (circular statistics)
        if attribute_name == 'aspect':
            def calc_circmean(x):
                """Calculate circular mean of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular mean, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circmean(rad))
                return result
            
            def calc_circstd(x):
                """Calculate circular standard deviation of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular std, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circstd(rad))
                return result
            
            # Add custom circular statistics
            stats = ['min', 'max']
            custom_stats = {
                'circmean': calc_circmean
            }
            
            try:
                from scipy.stats import circstd
                custom_stats['circstd'] = calc_circstd
            except ImportError:
                self.logger.warning("scipy.stats.circstd not available, skipping circular std")
            
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats,
                add_stats=custom_stats, 
                all_touched=True
            )
        else:
            # Regular statistics for elevation and slope
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats, 
                all_touched=True
            )
        
        # Format the results
        results = {}
        if zonal_out:
            for i, zonal_result in enumerate(zonal_out):
                # Use HRU ID as key if available, otherwise use index
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                if not is_lumped:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    hru_id = catchment.iloc[i][hru_id_field]
                    key_prefix = f"HRU_{hru_id}_"
                else:
                    key_prefix = ""
                
                # Add each statistic to results
                for stat, value in zonal_result.items():
                    if value is not None:
                        results[f"{key_prefix}{attribute_name}_{stat}"] = value
        
        return results

    def process_attributes(self) -> pd.DataFrame:
        """
        Process all available catchment attributes from multiple datasets.
        """
        try:
            # Dictionary to store all results
            all_results = {}
            
            # Process elevation attributes
            self.logger.info("Processing elevation attributes...")
            dem_results = self._process_elevation_attributes()
            all_results.update(dem_results)
            
            # Process land cover attributes
            self.logger.info("Processing land cover attributes...")
            landcover_results = self._process_landcover_attributes()
            all_results.update(landcover_results)
            
            # Process soil attributes
            self.logger.info("Processing soil attributes...")
            soil_results = self._process_soil_attributes()
            all_results.update(soil_results)
            
            # Process forest height attributes
            self.logger.info("Processing forest height attributes...")
            forest_results = self._process_forest_height_attributes()
            all_results.update(forest_results)

            # Process climate attributes
            self.logger.info("Processing climate attributes...")
            climate_results = self._process_climate_attributes()
            all_results.update(climate_results)

            # Process geological attributes
            self.logger.info("Processing geological attributes...")
            #geological_results = self._process_geological_attributes()
            #all_results.update(geological_results)
            
            # Process hydrological attributes
            self.logger.info("Processing hydrological attributes...")
            hydrological_results = self._process_hydrological_attributes()
            all_results.update(hydrological_results)
            
            # Process vegetation attributes
            self.logger.info("Processing vegetation attributes...")
            vegetation_results = self._process_vegetation_attributes()
            all_results.update(vegetation_results)
            
            # Process irrigation attributes
            self.logger.info("Processing irrigation attributes...")
            irrigation_results = self._process_irrigation_attributes()
            all_results.update(irrigation_results)
            
            # Create a structured DataFrame
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchments, create a single row
                df = pd.DataFrame([all_results])
                df['basin_id'] = self.domain_name
                df = df.set_index('basin_id')
            else:
                # For distributed catchments, create multi-level DataFrame with HRUs
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                hru_ids = catchment[hru_id_field].values
                
                # Reorganize results by HRU
                hru_results = []
                for i, hru_id in enumerate(hru_ids):
                    result = {'hru_id': hru_id}
                    
                    for key, value in all_results.items():
                        if key.startswith(f"HRU_{hru_id}_"):
                            # Remove HRU prefix
                            clean_key = key.replace(f"HRU_{hru_id}_", "")
                            result[clean_key] = value
                        elif not any(k.startswith("HRU_") for k in all_results.keys()):
                            # If there are no HRU-specific results, use the same values for all HRUs
                            result[key] = value
                    
                    hru_results.append(result)
                    
                df = pd.DataFrame(hru_results)
                df['basin_id'] = self.domain_name
                df = df.set_index(['basin_id', 'hru_id'])
            
            # Save to different formats
            output_dir = self.project_dir / 'attributes'
            
            # Use generic attribute file names
            csv_file = output_dir / f"{self.domain_name}_attributes.csv"
            df.to_csv(csv_file)
            
            # Save as Parquet
            try:
                parquet_file = output_dir / f"{self.domain_name}_attributes.parquet"
                df.to_parquet(parquet_file)
                self.logger.info(f"Attributes saved as Parquet: {parquet_file}")
            except ImportError:
                self.logger.warning("pyarrow not installed, skipping Parquet output")
            
            # Save as pickle
            pickle_file = output_dir / f"{self.domain_name}_attributes.pkl"
            df.to_pickle(pickle_file)
            
            self.logger.info(f"Attributes saved to {csv_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing attributes: {str(e)}")
            raise

    def _process_geological_attributes(self) -> Dict[str, Any]:
        """
        Process geological data including rock types and aquifer properties.
        
        Sources:
        - geologic_map: Rock types and lithology
        - glhymps: Groundwater permeability and porosity
        """
        results = {}
        
        # Find and check paths for geological data
        geologic_map_path = self._get_data_path('ATTRIBUTES_GEOLOGIC_MAP_PATH', 'geologic_map')
        glhymps_path = self._get_data_path('ATTRIBUTES_GLHYMPS_PATH', 'glhymps')
        
        self.logger.info(f"Geologic map path: {geologic_map_path}")
        self.logger.info(f"GLHYMPS path: {glhymps_path}")
        
        # Process geological map data (lithology)
        geo_results = self._process_lithology_data(geologic_map_path)
        results.update(geo_results)
        
        # Process groundwater properties
        glhymps_results = self._process_glhymps_data(glhymps_path)
        results.update(glhymps_results)
        
        return results

    def _process_lithology_data(self, geologic_map_path: Path) -> Dict[str, Any]:
        """
        Process lithology data from geological map shapefiles.
        
        Args:
            geologic_map_path: Path to geological map data
            
        Returns:
            Dictionary of lithology attributes
        """
        results = {}
        
        # Look for the geology shapefile directory
        shapefile_dir = geologic_map_path / "GMNA_SHAPES"
        if not shapefile_dir.exists():
            self.logger.warning(f"Geology shapefile directory not found at {shapefile_dir}")
            return results
        
        # Check for geologic units shapefile
        geologic_units_file = shapefile_dir / "Geologic_units.shp"
        if not geologic_units_file.exists():
            self.logger.warning(f"Geologic units shapefile not found: {geologic_units_file}")
            return results
        
        self.logger.info(f"Processing lithology data from: {geologic_units_file}")
        
        try:
            # Read the geologic units shapefile using geopandas
            geology = gpd.read_file(geologic_units_file, engine="fiona")

            # Identify invalid geometries (optional, for debugging)
            invalid_geoms = geology[~geology.is_valid]
            if not invalid_geoms.empty:
                self.logger.warning("Found invalid geometries; attempting to repair them.")

            # Repair invalid geometries using buffer(0)
            geology['geometry'] = geology['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

            
            # Read the catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            
            # Ensure both have the same CRS
            if geology.crs != catchment.crs:
                geology = geology.to_crs(catchment.crs)
            
            # Intersect geologic units with catchment to get area of each unit within catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, intersect with entire catchment
                intersect = gpd.overlay(geology, catchment, how='intersection')
                
                # Calculate area in square meters
                intersect['area'] = intersect.geometry.area
                
                # Get total area
                total_area = intersect['area'].sum()
                
                if total_area > 0:
                    # Group by lithology type and calculate percentage
                    if 'LITHOLOGY' in intersect.columns:
                        lith_column = 'LITHOLOGY'
                    elif 'LITH' in intersect.columns:
                        lith_column = 'LITH'
                    elif 'ROCKTYPE' in intersect.columns:
                        lith_column = 'ROCKTYPE'
                    elif 'ROCKTYPE1' in intersect.columns:
                        lith_column = 'ROCKTYPE1'
                    elif 'UNIT_NAME' in intersect.columns:
                        lith_column = 'UNIT_NAME'
                    elif 'TYPE' in intersect.columns:
                        lith_column = 'TYPE'
                    else:
                        # If we can't find a lithology column, use the first column with string data
                        string_columns = [col for col in intersect.columns 
                                        if intersect[col].dtype == 'object' and col != 'geometry']
                        lith_column = string_columns[0] if string_columns else None
                    
                    # Calculate percentage for each lithology type
                    if lith_column:
                        lith_areas = intersect.groupby(lith_column)['area'].sum()

                        # Check if lith_areas is empty
                        if lith_areas.empty:
                            self.logger.warning("No lithology areas found after intersection. Skipping dominant lithology calculation.")
                        else:
                            dominant_lith = lith_areas.idxmax()
                            clean_dominant = self._clean_attribute_name(dominant_lith)
                            results["geology.dominant_lithology"] = dominant_lith
                            results["geology.dominant_lithology_clean"] = clean_dominant
                            results["geology.dominant_lithology_fraction"] = lith_areas[dominant_lith] / total_area

                        
                        # Convert to fractions
                        lith_fractions = lith_areas / total_area
                        
                        # Add to results
                        for lith_type, fraction in lith_fractions.items():
                            # Clean up the lithology name for use as an attribute name
                            clean_lith = self._clean_attribute_name(lith_type)
                            results[f"geology.{clean_lith}_fraction"] = fraction
                        
                        # Find dominant lithology
                        dominant_lith = lith_areas.idxmax()
                        clean_dominant = self._clean_attribute_name(dominant_lith)
                        results["geology.dominant_lithology"] = dominant_lith
                        results["geology.dominant_lithology_clean"] = clean_dominant
                        results["geology.dominant_lithology_fraction"] = lith_fractions[dominant_lith]
                        
                        # Calculate lithological diversity (Shannon entropy)
                        shannon_entropy = 0
                        for lith_type, fraction in lith_fractions.items():
                            if fraction > 0:
                                shannon_entropy -= fraction * np.log(fraction)
                        
                        results["geology.lithology_diversity"] = shannon_entropy
                        
                        # Get age information if available
                        age_column = None
                        for col in ['AGE', 'ERA', 'PERIOD', 'EPOCH', 'FORMATION']:
                            if col in intersect.columns:
                                age_column = col
                                break
                        
                        if age_column:
                            # Calculate percentage for each age
                            age_areas = intersect.groupby(age_column)['area'].sum()
                            age_fractions = age_areas / total_area
                            
                            # Add to results
                            for age, fraction in age_fractions.items():
                                if age and str(age).strip():  # Check if age is not empty
                                    clean_age = self._clean_attribute_name(str(age))
                                    results[f"geology.age_{clean_age}_fraction"] = fraction
                            
                            # Find dominant age
                            dominant_age = age_areas.idxmax()
                            if dominant_age and str(dominant_age).strip():
                                results["geology.dominant_age"] = str(dominant_age)
                                results["geology.dominant_age_fraction"] = age_fractions[dominant_age]
            else:
                # For distributed catchment, process each HRU
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i, hru in catchment.iterrows():
                    hru_id = hru[hru_id_field]
                    prefix = f"HRU_{hru_id}_"
                    
                    # Create a GeoDataFrame with just this HRU
                    hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                    
                    # Intersect with geology
                    intersect = gpd.overlay(geology, hru_gdf, how='intersection')
                    
                    # Calculate area
                    intersect['area'] = intersect.geometry.area
                    total_area = intersect['area'].sum()
                    
                    if total_area > 0:
                        # Identify lithology column
                        if 'LITHOLOGY' in intersect.columns:
                            lith_column = 'LITHOLOGY'
                        elif 'LITH' in intersect.columns:
                            lith_column = 'LITH'
                        elif 'ROCKTYPE' in intersect.columns:
                            lith_column = 'ROCKTYPE'
                        elif 'ROCKTYPE1' in intersect.columns:
                            lith_column = 'ROCKTYPE1'
                        elif 'UNIT_NAME' in intersect.columns:
                            lith_column = 'UNIT_NAME'
                        elif 'TYPE' in intersect.columns:
                            lith_column = 'TYPE'
                        else:
                            # If we can't find a lithology column, use the first column with string data
                            string_columns = [col for col in intersect.columns 
                                            if intersect[col].dtype == 'object' and col != 'geometry']
                            lith_column = string_columns[0] if string_columns else None
                        
                        if lith_column:
                            # Calculate percentage for each lithology type
                            lith_areas = intersect.groupby(lith_column)['area'].sum()
                            lith_fractions = lith_areas / total_area
                            
                            # Add to results
                            for lith_type, fraction in lith_fractions.items():
                                clean_lith = self._clean_attribute_name(lith_type)
                                results[f"{prefix}geology.{clean_lith}_fraction"] = fraction
                            
                            # Find dominant lithology
                            dominant_lith = lith_areas.idxmax()
                            clean_dominant = self._clean_attribute_name(dominant_lith)
                            results[f"{prefix}geology.dominant_lithology"] = dominant_lith
                            results[f"{prefix}geology.dominant_lithology_clean"] = clean_dominant
                            results[f"{prefix}geology.dominant_lithology_fraction"] = lith_fractions[dominant_lith]
                            
                            # Calculate lithological diversity
                            shannon_entropy = 0
                            for lith_type, fraction in lith_fractions.items():
                                if fraction > 0:
                                    shannon_entropy -= fraction * np.log(fraction)
                            
                            results[f"{prefix}geology.lithology_diversity"] = shannon_entropy
                            
                            # Get age information if available
                            age_column = None
                            for col in ['AGE', 'ERA', 'PERIOD', 'EPOCH', 'FORMATION']:
                                if col in intersect.columns:
                                    age_column = col
                                    break
                            
                            if age_column:
                                # Calculate percentage for each age
                                age_areas = intersect.groupby(age_column)['area'].sum()
                                age_fractions = age_areas / total_area
                                
                                # Add to results
                                for age, fraction in age_fractions.items():
                                    if age and str(age).strip():  # Check if age is not empty
                                        clean_age = self._clean_attribute_name(str(age))
                                        results[f"{prefix}geology.age_{clean_age}_fraction"] = fraction
                                
                                # Find dominant age
                                dominant_age = age_areas.idxmax()
                                if dominant_age and str(dominant_age).strip():
                                    results[f"{prefix}geology.dominant_age"] = str(dominant_age)
                                    results[f"{prefix}geology.dominant_age_fraction"] = age_fractions[dominant_age]
        
        except Exception as e:
            self.logger.error(f"Error processing lithology data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _clean_attribute_name(self, name: str) -> str:
        """
        Clean up a string to be usable as an attribute name.
        
        Args:
            name: Original string
            
        Returns:
            String usable as an attribute name
        """
        if not name:
            return "unknown"
        
        # Convert to string and strip whitespace
        name_str = str(name).strip()
        if not name_str:
            return "unknown"
        
        # Replace spaces and special characters
        cleaned = name_str.lower()
        cleaned = cleaned.replace(' ', '_')
        cleaned = cleaned.replace('-', '_')
        cleaned = cleaned.replace('.', '_')
        cleaned = cleaned.replace('(', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace('/', '_')
        cleaned = cleaned.replace('\\', '_')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('%', 'percent')
        
        # Remove any remaining non-alphanumeric characters
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
        
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'x' + cleaned
        
        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        
        # Handle empty result
        if not cleaned:
            return "unknown"
        
        return cleaned

    def _process_glhymps_data(self, glhymps_path: Path) -> Dict[str, Any]:
        """
        Process GLHYMPS groundwater permeability and porosity data.
        
        Args:
            glhymps_path: Path to GLHYMPS data
            
        Returns:
            Dictionary of groundwater attributes
        """
        results = {}
        
        # Check if GLHYMPS data is in vector format
        glhymps_shp = glhymps_path / 'raw' / "glhymps.shp"
        
        if glhymps_shp.exists():
            self.logger.info(f"Processing GLHYMPS data from shapefile: {glhymps_shp}")
            
            try:
                # Read the GLHYMPS shapefile
                glhymps = gpd.read_file(glhymps_shp)
                
                # Read the catchment shapefile
                catchment = gpd.read_file(self.catchment_path)
                
                # Ensure both have the same CRS
                if glhymps.crs != catchment.crs:
                    glhymps = glhymps.to_crs(catchment.crs)
                
                # Look for permeability and porosity columns
                perm_cols = [col for col in glhymps.columns if 'PERM' in col.upper()]
                por_cols = [col for col in glhymps.columns if 'PORO' in col.upper()]
                
                perm_col = perm_cols[0] if perm_cols else None
                por_col = por_cols[0] if por_cols else None
                
                if perm_col or por_col:
                    # Intersect GLHYMPS with catchment
                    is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                    
                    if is_lumped:
                        # For lumped catchment
                        intersect = gpd.overlay(glhymps, catchment, how='intersection')
                        
                        # Calculate area in square meters
                        intersect['area'] = intersect.geometry.area
                        total_area = intersect['area'].sum()
                        
                        if total_area > 0:
                            # Process permeability
                            if perm_col:
                                # Calculate area-weighted average permeability
                                weighted_perm = (intersect[perm_col] * intersect['area']).sum() / total_area
                                results["geology.permeability_mean"] = weighted_perm
                                
                                # Calculate hydraulic conductivity (if perm is in log10 m^2)
                                if weighted_perm < 0:  # Likely in log10 units
                                    perm_m2 = 10**weighted_perm
                                    hyd_cond = perm_m2 * 1000 * 9.81 / 0.001  # Convert to m/s
                                    results["geology.hydraulic_conductivity"] = hyd_cond
                            
                            # Process porosity
                            if por_col:
                                # Calculate area-weighted average porosity
                                weighted_por = (intersect[por_col] * intersect['area']).sum() / total_area
                                results["geology.porosity_mean"] = weighted_por
                            
                            # Calculate transmissivity if we have hydraulic conductivity
                            if "geology.hydraulic_conductivity" in results:
                                # Assume aquifer thickness of 100 m
                                results["geology.transmissivity"] = results["geology.hydraulic_conductivity"] * 100
                    else:
                        # For distributed catchment
                        hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                        
                        for i, hru in catchment.iterrows():
                            hru_id = hru[hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Create a GeoDataFrame with just this HRU
                            hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                            
                            # Intersect with GLHYMPS
                            intersect = gpd.overlay(glhymps, hru_gdf, how='intersection')
                            
                            # Calculate area
                            intersect['area'] = intersect.geometry.area
                            total_area = intersect['area'].sum()
                            
                            if total_area > 0:
                                # Process permeability
                                if perm_col:
                                    # Calculate area-weighted average permeability
                                    weighted_perm = (intersect[perm_col] * intersect['area']).sum() / total_area
                                    results[f"{prefix}geology.permeability_mean"] = weighted_perm
                                    
                                    # Calculate hydraulic conductivity
                                    if weighted_perm < 0:  # Likely in log10 units
                                        perm_m2 = 10**weighted_perm
                                        hyd_cond = perm_m2 * 1000 * 9.81 / 0.001  # Convert to m/s
                                        results[f"{prefix}geology.hydraulic_conductivity"] = hyd_cond
                                
                                # Process porosity
                                if por_col:
                                    # Calculate area-weighted average porosity
                                    weighted_por = (intersect[por_col] * intersect['area']).sum() / total_area
                                    results[f"{prefix}geology.porosity_mean"] = weighted_por
                                
                                # Calculate transmissivity
                                hyd_cond_key = f"{prefix}geology.hydraulic_conductivity"
                                if hyd_cond_key in results:
                                    results[f"{prefix}geology.transmissivity"] = results[hyd_cond_key] * 100
            
            except Exception as e:
                self.logger.error(f"Error processing GLHYMPS shapefile: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.warning(f"GLHYMPS shapefile not found at {glhymps_shp}")
            # Look for any other GLHYMPS shapefiles
            glhymps_files = list(glhymps_path.glob("*.shp"))
            if glhymps_files:
                self.logger.info(f"Found alternative GLHYMPS shapefile: {glhymps_files[0]}")
                # You could process this file instead
        
        return results

    def _process_hydrological_attributes(self) -> Dict[str, Any]:
        """
        Process hydrological data including lakes and rivers.
        
        Sources:
        - hydrolakes: Lake polygons to determine lake area, count, coverage and density.
        - merit_basins: River vectors to calculate total river length and river density.
        
        Returns:
            Dictionary of hydrological attributes.
        """
        results = {}
        
        try:
            # ---------------------------
            # Process Hydrolakes
            # ---------------------------
            lakes_path = self._get_data_path('ATTRIBUTES_HYDROLAKES_PATH', 'hydrolakes')
            # Assume lake data are stored in a "raw" subfolder
            lakes_shp = lakes_path / "raw" / "HydroLAKES_polys_v10_NorthAmerica.shp"
            
            if lakes_shp.exists():
                self.logger.info(f"Processing Hydrolakes from: {lakes_shp}")
                lakes = gpd.read_file(lakes_shp)
                catchment = gpd.read_file(self.catchment_path)
                
                # Ensure both layers share the same CRS.
                if lakes.crs != catchment.crs:
                    lakes = lakes.to_crs(catchment.crs)
                
                # Intersect lake polygons with catchment
                lakes_intersect = gpd.overlay(lakes, catchment, how="intersection")
                if not lakes_intersect.empty:
                    # Calculate area for each intersected lake feature.
                    lakes_intersect['area'] = lakes_intersect.geometry.area
                    total_lake_area = lakes_intersect['area'].sum()
                    catchment_area = catchment.geometry.area.sum()
                    lake_count = len(lakes_intersect)
                    
                    results["hydro.lakes_total_area"] = total_lake_area
                    results["hydro.lakes_count"] = lake_count
                    results["hydro.lakes_coverage_fraction"] = total_lake_area / catchment_area if catchment_area > 0 else np.nan
                    
                    # Compute catchment area in km2 for lake density calculation
                    catchment_area_km2 = catchment_area / 1e6
                    results["hydro.lakes_density"] = lake_count / catchment_area_km2 if catchment_area_km2 > 0 else np.nan
                    results["hydro.lakes_avg_size"] = total_lake_area / lake_count if lake_count > 0 else np.nan
                else:
                    self.logger.warning("No hydrolake features intersect the catchment.")
            else:
                self.logger.warning(f"Hydrolakes shapefile not found at: {lakes_shp}")
            
            # ---------------------------
            # Process Rivers from Merit Basins
            # ---------------------------
            merit_path = self._get_data_path('ATTRIBUTES_MERIT_BASINS_PATH', 'merit_basins')
            # Navigate into the rivers subfolder within the MERIT Hydro shapes directory.
            rivers_dir = merit_path / "MERIT_Hydro_modified_North_America_shapes" / "rivers"
            river_shps = list(rivers_dir.glob("*.shp"))
            
            if river_shps:
                river_shp = river_shps[0]
                self.logger.info(f"Processing rivers from: {river_shp}")
                rivers = gpd.read_file(river_shp)
                catchment = gpd.read_file(self.catchment_path)
                
                # Ensure the rivers are in the same CRS as the catchment.
                if rivers.crs != catchment.crs:
                    rivers = rivers.to_crs(catchment.crs)
                
                # Intersect the river features with the catchment.
                rivers_intersect = gpd.overlay(rivers, catchment, how="intersection")
                if not rivers_intersect.empty:
                    # Calculate the length of each river segment (assuming the CRS is in meters)
                    rivers_intersect['length'] = rivers_intersect.geometry.length
                    total_river_length = rivers_intersect['length'].sum()
                    # Convert total river length to kilometers.
                    total_river_length_km = total_river_length / 1000.0
                    results["hydro.rivers_total_length_km"] = total_river_length_km
                    
                    # Use the catchment area already computed (if not, compute it here again)
                    catchment_area = catchment.geometry.area.sum()
                    catchment_area_km2 = catchment_area / 1e6
                    results["hydro.rivers_density"] = total_river_length_km / catchment_area_km2 if catchment_area_km2 > 0 else np.nan
                else:
                    self.logger.warning("No river features intersect the catchment.")
            else:
                self.logger.warning(f"No river shapefile found in directory: {rivers_dir}")
            
        except Exception as e:
            self.logger.error(f"Error processing hydrological attributes: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _process_climate_attributes(self) -> Dict[str, Any]:
        """
        Process climate data including temperature, precipitation,
        and related derived indices using WorldClim datasets.
        
        Expected structure in the WorldClim data directory (e.g., from _get_data_path):
        - raw/tavg/      : monthly average temperature rasters
        - raw/tmin/      : monthly minimum temperature rasters
        - raw/tmax/      : monthly maximum temperature rasters
        - raw/prec/      : monthly precipitation rasters
        - derived/pet/   : monthly PET (potential evapotranspiration) rasters
        
        For temperature variables, the annual aggregate is the mean of the monthly means,
        whereas for precipitation and PET the annual total is computed.
        
        Also computes an aridity index defined as: (annual PET) / (annual Precipitation).
        
        Returns:
            Dictionary of computed climate attributes.
        """
        results = {}
        
        # Get the WorldClim path using config.
        wc_path = self._get_data_path('ATTRIBUTES_WORLDCLIM_PATH', 'worldclim')
        self.logger.info(f"WorldClim path: {wc_path}")
        
        # Decide whether to process the catchment as lumped or distributed.
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        catchment_shape = str(self.catchment_path)
        
        # Helper to process a set of monthly files for a given variable.
        # folder_relative: relative path under the WorldClim folder (e.g., "raw/tavg")
        # variable: a short string identifier (e.g., "tavg", "prec", "pet")
        # For temperature variables we aggregate by averaging (and take min, max over months, etc.),
        # For accumulative variables (precipitation, PET) we can also sum.
        def process_monthly_rasters(variable: str, folder_relative: str, sum_values: bool = False) -> Dict[str, Any]:
            var_results = {}
            folder = wc_path / folder_relative
            self.logger.info(f"Processing {variable} data from folder: {folder}")
            # List and sort files (assuming file names include _01.tif, _02.tif, etc.)
            raster_files = sorted(folder.glob("*.tif"))
            if not raster_files:
                self.logger.warning(f"No raster files found in {folder} for variable {variable}")
                return var_results
            
            monthly_means = []
            monthly_mins = []
            monthly_maxs = []
            monthly_stds = []
            monthly_sums = []
            
            for i, rf in enumerate(raster_files, start=1):
                zs = zonal_stats(
                    catchment_shape,
                    str(rf),
                    stats=['min', 'mean', 'max', 'std', 'sum'],
                    all_touched=True
                )
                # Here we assume a lumped catchment so we look at zs[0].
                # (For distributed, you might want to loop over HRUs.)
                if zs and len(zs) > 0:
                    stat = zs[0]
                    monthly_means.append(stat.get('mean', np.nan))
                    monthly_mins.append(stat.get('min', np.nan))
                    monthly_maxs.append(stat.get('max', np.nan))
                    monthly_stds.append(stat.get('std', np.nan))
                    monthly_sums.append(stat.get('sum', np.nan))
                else:
                    monthly_means.append(np.nan)
                    monthly_mins.append(np.nan)
                    monthly_maxs.append(np.nan)
                    monthly_stds.append(np.nan)
                    monthly_sums.append(np.nan)
                # Save individual monthly statistics for each file.
                month_str = f"{i:02d}"
                var_results[f"{variable}_month_{month_str}_mean"] = monthly_means[-1]
                var_results[f"{variable}_month_{month_str}_min"] = monthly_mins[-1]
                var_results[f"{variable}_month_{month_str}_max"] = monthly_maxs[-1]
                var_results[f"{variable}_month_{month_str}_std"] = monthly_stds[-1]
                var_results[f"{variable}_month_{month_str}_sum"] = monthly_sums[-1]
            
            # Aggregate across months.
            # For temperature, we take the mean over months as annual average;
            # also record the overall min, max, and variability (std) of the monthly means.
            if monthly_means:
                annual_avg = np.nanmean(monthly_means)
                annual_min = np.nanmin(monthly_mins)
                annual_max = np.nanmax(monthly_maxs)
                annual_std = np.nanstd(monthly_means)
            else:
                annual_avg = annual_min = annual_max = annual_std = np.nan
            
            # For accumulative data (precipitation and PET) sum the monthly totals.
            if sum_values:
                annual_total = np.nansum(monthly_sums)
            else:
                annual_total = np.nan
            
            var_results[f"{variable}_annual_avg"] = annual_avg
            var_results[f"{variable}_annual_min"] = annual_min
            var_results[f"{variable}_annual_max"] = annual_max
            var_results[f"{variable}_annual_std"] = annual_std
            if sum_values:
                var_results[f"{variable}_annual_total"] = annual_total
            
            return var_results

        # Process temperature (average) from raw/tavg.
        tavg_results = process_monthly_rasters("tavg", "raw/tavg", sum_values=False)
        results.update(tavg_results)

        # Process temperature minimum and maximum.
        tmin_results = process_monthly_rasters("tmin", "raw/tmin", sum_values=False)
        tmax_results = process_monthly_rasters("tmax", "raw/tmax", sum_values=False)
        results.update(tmin_results)
        results.update(tmax_results)

        # Process precipitation from raw/prec.
        # For precipitation it makes more sense to sum the monthly values to get an annual total.
        prec_results = process_monthly_rasters("prec", "raw/prec", sum_values=True)
        results.update(prec_results)

        # Process PET from derived/pet (if available).
        pet_folder = wc_path / "derived" / "pet"
        if pet_folder.exists():
            pet_results = process_monthly_rasters("pet", "derived/pet", sum_values=True)
            results.update(pet_results)
        else:
            self.logger.warning("PET data folder not found in derived data.")

        # Compute derived indices.
        # Example: Aridity index = (annual PET) / (annual Precipitation)
        if ("pet_annual_total" in results and "prec_annual_total" in results and 
            results["prec_annual_total"] not in (None, 0, np.nan)):
            results["aridity_index"] = results["pet_annual_total"] / results["prec_annual_total"]
        else:
            results["aridity_index"] = np.nan

        # Mean annual temperature based on tavg.
        results["mean_annual_temperature"] = results.get("tavg_annual_avg", np.nan)

        # You can add additional derived metrics here as needed.
        
        return results


    def _process_vegetation_attributes(self) -> Dict[str, Any]:
        """
        Process vegetation data using LAI datasets.
        
        This function extracts monthly LAI statistics from the LAI
        dataset (assumed to be stored as monthly average composites)
        and computes aggregated vegetation attributes for the catchment.
        
        Returns:
            Dictionary with vegetation attributes such as:
            - vegetation.lai_mean: Overall mean LAI (averaged over all months)
            - vegetation.lai_min: Minimum monthly LAI (across the period)
            - vegetation.lai_max: Maximum monthly LAI (across the period)
            - vegetation.lai_std: Average of the monthly LAI standard deviations
            - vegetation.lai_amplitude: Difference between max and min monthly means
            - vegetation.lai_cv: Coefficient of variation (std/mean) of monthly means
            - vegetation.lai_monthly_means: List of monthly mean LAI values
        """
        results = {}
        try:
            # Get the base LAI data directory using the helper function.
            lai_base = self._get_data_path("ATTRIBUTES_LAI_PATH", "lai")
            
            # Define the target subdirectory for the monthly averaged LAI data.
            # You could change this to "monthly_lai_no_water_mask" if preferred.
            target_dir = lai_base / "monthly_average_2013_2023" / "monthly_lai_with_water_mask"
            if not target_dir.exists():
                self.logger.warning(f"LAI target directory not found: {target_dir}")
                return results

            # List all TIF files from the target directory and sort them
            lai_files = sorted(target_dir.glob("*.tif"))
            if not lai_files:
                self.logger.warning(f"No LAI raster files found in {target_dir}")
                return results

            self.logger.info(f"Processing vegetation LAI attributes from {len(lai_files)} files in {target_dir}")

            # Lists to store zonal statistics for each monthly composite
            monthly_means = []
            monthly_mins = []
            monthly_maxs = []
            monthly_stds = []

            # Process each LAI file (assumes a lumped catchment)
            for lai_file in lai_files:
                self.logger.info(f"Processing LAI file: {lai_file}")
                # Calculate zonal statistics over the catchment (self.catchment_path)
                stats = zonal_stats(
                    str(self.catchment_path),
                    str(lai_file),
                    stats=["mean", "min", "max", "std"],
                    all_touched=True
                )
                if stats and len(stats) > 0:
                    stat = stats[0]  # For a lumped catchment, use the first (and only) result
                    # Append the statistics if they are valid
                    if stat.get("mean") is not None:
                        monthly_means.append(stat["mean"])
                    if stat.get("min") is not None:
                        monthly_mins.append(stat["min"])
                    if stat.get("max") is not None:
                        monthly_maxs.append(stat["max"])
                    if stat.get("std") is not None:
                        monthly_stds.append(stat["std"])

            # If we have computed monthly statistics, aggregate them to produce overall vegetation attributes.
            if monthly_means:
                overall_mean = np.mean(monthly_means)
                overall_std = np.std(monthly_means)
                amplitude = np.max(monthly_means) - np.min(monthly_means)
                cv = overall_std / overall_mean if overall_mean != 0 else np.nan

                results["vegetation.lai_mean"] = float(overall_mean)
                results["vegetation.lai_mean_std"] = float(overall_std)
                results["vegetation.lai_amplitude"] = float(amplitude)
                results["vegetation.lai_cv"] = float(cv)
                # Also, record the min and max values across months
                results["vegetation.lai_min"] = float(np.min(monthly_mins)) if monthly_mins else None
                results["vegetation.lai_max"] = float(np.max(monthly_maxs)) if monthly_maxs else None
                # And average the per-month standard deviations
                results["vegetation.lai_std"] = float(np.mean(monthly_stds)) if monthly_stds else None
                
                # Optionally, store the individual monthly mean values (this may help with further analysis)
                results["vegetation.lai_monthly_means"] = monthly_means

        except Exception as e:
            self.logger.error(f"Error processing vegetation LAI attributes: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

        return results


    def _process_irrigation_attributes(self) -> Dict[str, Any]:
        """
        Process irrigation data from LGRIP30 dataset (global irrigation patterns).
        
        This function uses the LGRIP30 agriculture raster (lgrip30_agriculture.tif)
        to compute the fraction of the catchment that is irrigated. It assumes that
        the raster is a classification where a pixel value of 1 indicates irrigated
        areas (and other values—for example, 0—represent non-irrigated areas).
        
        Returns:
            Dictionary with irrigation attributes, including:
            - irrigation.agriculture_fraction: Fraction of the catchment area that is irrigated.
            - irrigation.agriculture_area: Total irrigated area in square meters.
        """
        results = {}
        try:
            # Get the LGRIP30 data directory (e.g., /work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/lgrip30)
            lgrip_path = self._get_data_path('ATTRIBUTES_LGRIP_PATH', 'lgrip30')
            # Construct the expected path to the agriculture raster in the raw subfolder.
            agri_file = lgrip_path / "raw" / "lgrip30_agriculture.tif"
            
            if not agri_file.exists():
                self.logger.warning(f"Irrigation raster not found at: {agri_file}")
                return results
            
            self.logger.info(f"Processing irrigation data from: {agri_file}")
            
            # Use zonal_stats to get categorical counts over the catchment.
            # We assume that the catchment is stored in self.catchment_path.
            zstats = zonal_stats(
                str(self.catchment_path),
                str(agri_file),
                categorical=True,
                all_touched=True
            )
            
            if not zstats or len(zstats) == 0:
                self.logger.warning("Zonal statistics for irrigation returned an empty result.")
                return results
            
            # Assuming a lumped catchment (single polygon), get the first result.
            cat_stats = zstats[0]
            total_pixels = sum(cat_stats.values())
            
            # In this example we assume value 1 in the raster represents irrigated area.
            irrigated_pixels = cat_stats.get(1, 0)
            fraction = irrigated_pixels / total_pixels if total_pixels > 0 else np.nan
            results["irrigation.agriculture_fraction"] = fraction
            
            # To compute the total irrigated area (in m²), determine the area of one pixel.
            with rasterio.open(str(agri_file)) as src:
                transform = src.transform
                # Typically, transform.a is the pixel width and transform.e is the pixel height.
                # Use the absolute values to compute area.
                pixel_area = abs(transform.a * transform.e)
            
            irrigated_area = irrigated_pixels * pixel_area
            results["irrigation.agriculture_area"] = irrigated_area
            
        except Exception as e:
            self.logger.error(f"Error processing irrigation attributes: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _get_data_path(self, config_key: str, default_subfolder: str) -> Path:
        """
        Get the path to a data source, checking multiple possible locations.
        
        Args:
            config_key: Configuration key to check for a specified path
            default_subfolder: Default subfolder to use within attribute data directories
            
        Returns:
            Path to the data source
        """
        # Check if path is explicitly specified in config
        path = self.config.get(config_key)
        
        if path and path != 'default':
            return Path(path)
        
        # Check the attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_path = Path(attr_data_dir) / default_subfolder
            if attr_path.exists():
                return attr_path
        
        # Check the specific North America data directory
        na_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial") / default_subfolder
        if na_path.exists():
            return na_path
        
        # Fall back to project data directory
        proj_path = self.data_dir / 'geospatial-data' / default_subfolder
        return proj_path


    def _process_forest_height_attributes(self) -> Dict[str, Any]:
        """
        Acquire, clip, and process forest height data for the catchment.
        """
        results = {}
        
        # Create forest height directory
        forest_dir = self.project_dir / 'attributes' / 'forest_height'
        forest_dir.mkdir(parents=True, exist_ok=True)
        
        # Find source forest height data
        forest_source = self._find_forest_height_source()
        
        if not forest_source:
            self.logger.warning("No forest height data found")
            return results
        
        # Clip forest height data to bounding box if not already done
        clipped_forest = forest_dir / f"{self.domain_name}_forest_height.tif"
        
        if not clipped_forest.exists():
            self.logger.info(f"Clipping forest height data to catchment bounding box")
            self._clip_raster_to_bbox(forest_source, clipped_forest)
        
        # Process the clipped forest height data
        if clipped_forest.exists():
            self.logger.info(f"Processing forest height statistics from {clipped_forest}")
            
            # Calculate zonal statistics
            stats = ['min', 'mean', 'max', 'std', 'median', 'count']
            
            try:
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(clipped_forest),
                    stats=stats,
                    all_touched=True,
                    nodata=-9999  # Common no-data value for forest height datasets
                )
                
                # Process results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0:
                        for stat, value in zonal_out[0].items():
                            if value is not None:
                                results[f"forest.height_{stat}"] = value
                        
                        # Calculate forest coverage
                        if 'count' in zonal_out[0] and zonal_out[0]['count'] is not None:
                            with rasterio.open(str(clipped_forest)) as src:
                                # Calculate total number of pixels in the catchment
                                total_pixels = self._count_pixels_in_catchment(src)
                                
                                if total_pixels > 0:
                                    # Count non-zero forest height pixels
                                    forest_pixels = zonal_out[0]['count']
                                    # Calculate forest coverage percentage
                                    forest_coverage = (forest_pixels / total_pixels) * 100
                                    results["forest.coverage_percent"] = forest_coverage
                        
                        # Calculate forest height distribution metrics
                        self._add_forest_distribution_metrics(clipped_forest, results)
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    # Process each HRU
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            for stat, value in zonal_result.items():
                                if value is not None:
                                    results[f"{prefix}forest.height_{stat}"] = value
                            
                            # Calculate forest coverage for this HRU
                            if 'count' in zonal_result and zonal_result['count'] is not None:
                                with rasterio.open(str(clipped_forest)) as src:
                                    # Calculate total number of pixels in this HRU
                                    total_pixels = self._count_pixels_in_hru(src, catchment.iloc[i:i+1])
                                    
                                    if total_pixels > 0:
                                        # Count non-zero forest height pixels
                                        forest_pixels = zonal_result['count']
                                        # Calculate forest coverage percentage
                                        forest_coverage = (forest_pixels / total_pixels) * 100
                                        results[f"{prefix}forest.coverage_percent"] = forest_coverage
                            
                            # Calculate forest height distribution metrics for this HRU
                            self._add_forest_distribution_metrics(
                                clipped_forest,
                                results,
                                hru_geometry=catchment.iloc[i:i+1],
                                prefix=prefix
                            )
            
            except Exception as e:
                self.logger.error(f"Error processing forest height statistics: {str(e)}")
        
        return results

    def _find_forest_height_source(self) -> Optional[Path]:
        """
        Find forest height data in the data directory or global datasets.
        Returns the path to the forest height raster if found, None otherwise.
        """
        # Check config for forest height data path - use the correct config key
        forest_path = self.config.get('ATTRIBUTES_FOREST_HEIGHT_PATH')
        
        if forest_path and forest_path != 'default':
            forest_path = Path(forest_path)
            if forest_path.exists():
                self.logger.info(f"Using forest height data from config: {forest_path}")
                return forest_path
        
        # Use the base attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_data_dir = Path(attr_data_dir)
            
            # Check for forest_height directory within the attribute data directory
            forest_dir = attr_data_dir / 'forest_height' / 'raw'
            if forest_dir.exists():
                # Prefer the 2020 data as it's more recent
                if (forest_dir / "forest_height_2020.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2020.tif'}")
                    return forest_dir / "forest_height_2020.tif"
                # Fall back to 2000 data if 2020 isn't available
                elif (forest_dir / "forest_height_2000.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2000.tif'}")
                    return forest_dir / "forest_height_2000.tif"
                
                # Look for any TIF file in the directory
                forest_files = list(forest_dir.glob("*.tif"))
                if forest_files:
                    self.logger.info(f"Using forest height data from attribute directory: {forest_files[0]}")
                    return forest_files[0]
        
        # Fall back to project data directory if attribute data dir doesn't have the data
        global_forest_dir = self.data_dir / 'geospatial-data' / 'forest_height'
        if global_forest_dir.exists():
            forest_files = list(global_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from global directory: {forest_files[0]}")
                return forest_files[0]
        
        # Try the hard-coded path as a last resort
        specific_forest_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/forest_height/raw")
        if specific_forest_dir.exists():
            forest_files = list(specific_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from specific directory: {forest_files[0]}")
                return forest_files[0]
        
        self.logger.warning("No forest height data found")
        return None

    def _clip_raster_to_bbox(self, source_raster: Path, output_raster: Path) -> None:
        """
        Clip a raster to the catchment's bounding box.
        """
        try:
            # Parse bounding box coordinates
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
            if len(bbox_coords) == 4:
                bbox = [
                    float(bbox_coords[1]),  # lon_min
                    float(bbox_coords[2]),  # lat_min
                    float(bbox_coords[3]),  # lon_max
                    float(bbox_coords[0])   # lat_max
                ]
            else:
                raise ValueError("Invalid bounding box coordinates format")
            
            # Ensure output directory exists
            output_raster.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a bounding box geometry
            geom = box(*bbox)
            
            # Create a GeoDataFrame with the bounding box
            bbox_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            
            # Clip the raster
            with rasterio.open(str(source_raster)) as src:
                # Check if coordinate reference systems match
                if src.crs != bbox_gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(src.crs)
                
                # Get the geometry in the raster's CRS
                geoms = [geom for geom in bbox_gdf.geometry]
                
                # Clip the raster
                out_image, out_transform = mask(src, geoms, crop=True)
                
                # Copy the metadata
                out_meta = src.meta.copy()
                
                # Update metadata for clipped raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # Write the clipped raster
                with rasterio.open(str(output_raster), "w", **out_meta) as dest:
                    dest.write(out_image)
            
            self.logger.info(f"Clipped raster saved to {output_raster}")
            
        except Exception as e:
            self.logger.error(f"Error clipping raster: {str(e)}")
            raise

    def _count_pixels_in_catchment(self, raster_src) -> int:
        """
        Count the total number of pixels within the catchment boundary.
        """
        # Read the catchment shapefile
        catchment = gpd.read_file(str(self.catchment_path))
        
        # Ensure the shapefile is in the same CRS as the raster
        if catchment.crs != raster_src.crs:
            catchment = catchment.to_crs(raster_src.crs)
        
        # Create a mask of the catchment area
        mask, transform = rasterio.mask.mask(raster_src, catchment.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the catchment)
        return np.sum(mask[0] != 0)

    def _count_pixels_in_hru(self, raster_src, hru_geometry) -> int:
        """
        Count the total number of pixels within a specific HRU boundary.
        """
        # Ensure the HRU geometry is in the same CRS as the raster
        if hru_geometry.crs != raster_src.crs:
            hru_geometry = hru_geometry.to_crs(raster_src.crs)
        
        # Create a mask of the HRU area
        mask, transform = rasterio.mask.mask(raster_src, hru_geometry.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the HRU)
        return np.sum(mask[0] != 0)

    def _add_forest_distribution_metrics(self, forest_raster: Path, results: Dict[str, Any], 
                                        hru_geometry=None, prefix="") -> None:
        """
        Calculate forest height distribution metrics like skewness, kurtosis, and percentiles.
        """
        try:
            with rasterio.open(str(forest_raster)) as src:
                if hru_geometry is not None:
                    # For a specific HRU
                    if hru_geometry.crs != src.crs:
                        hru_geometry = hru_geometry.to_crs(src.crs)
                    
                    # Mask the raster to the HRU
                    masked_data, _ = rasterio.mask.mask(src, hru_geometry.geometry, crop=True, nodata=-9999)
                else:
                    # For the entire catchment
                    catchment = gpd.read_file(str(self.catchment_path))
                    
                    if catchment.crs != src.crs:
                        catchment = catchment.to_crs(src.crs)
                    
                    # Mask the raster to the catchment
                    masked_data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=-9999)
                
                # Flatten the data and exclude no-data values
                valid_data = masked_data[masked_data != -9999]
                
                # Only calculate statistics if we have enough valid data points
                if len(valid_data) > 10:
                    # Calculate additional statistics
                    results[f"{prefix}forest.height_skew"] = float(skew(valid_data))
                    results[f"{prefix}forest.height_kurtosis"] = float(kurtosis(valid_data))
                    
                    # Calculate percentiles
                    percentiles = [5, 10, 25, 50, 75, 90, 95]
                    for p in percentiles:
                        results[f"{prefix}forest.height_p{p}"] = float(np.percentile(valid_data, p))
                    
                    # Calculate canopy density metrics
                    # Assume forest height threshold of 5m for "canopy"
                    canopy_threshold = 5.0
                    canopy_pixels = np.sum(valid_data >= canopy_threshold)
                    if len(valid_data) > 0:
                        results[f"{prefix}forest.canopy_density"] = float(canopy_pixels / len(valid_data))
        
        except Exception as e:
            self.logger.error(f"Error calculating forest distribution metrics: {str(e)}")

    def _process_soil_attributes(self) -> Dict[str, Any]:
        """Process soil attributes including dominant soil class."""
        results = {}
        
        # Find the soil class raster
        soilclass_dir = self.project_dir / 'attributes' / 'soilclass'
        if not soilclass_dir.exists():
            self.logger.warning(f"Soil class directory not found: {soilclass_dir}")
            return results
        
        # Search for soil class raster files
        soilclass_files = list(soilclass_dir.glob("*.tif"))
        if not soilclass_files:
            self.logger.warning(f"No soil class raster files found in {soilclass_dir}")
            return results
        
        # Use the first soil class file found
        soilclass_file = soilclass_files[0]
        self.logger.info(f"Processing soil class raster: {soilclass_file}")
        
        try:
            # Get soil class names if available
            soil_classes = self._get_soil_classes(soilclass_file)
            
            # Calculate zonal statistics with categorical=True to get class counts
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(soilclass_file), 
                categorical=True, 
                all_touched=True
            )
            
            # Process results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out:
                    # Calculate class fractions
                    total_pixels = sum(count for count in zonal_out[0].values() if count is not None)
                    
                    if total_pixels > 0:
                        # Add soil class fractions
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None:  # Skip None values
                                # Get class name if available, otherwise use class ID
                                class_name = soil_classes.get(class_id, f"class_{class_id}")
                                fraction = (count / total_pixels) if count is not None else 0
                                results[f"soil.{class_name}_fraction"] = fraction
                        
                        # Find dominant soil class
                        dominant_class_id = max(zonal_out[0].items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                        dominant_class_name = soil_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                        results["soil.dominant_class"] = dominant_class_name
                        results["soil.dominant_class_id"] = dominant_class_id
                        results["soil.dominant_fraction"] = (zonal_out[0][dominant_class_id] / total_pixels) if zonal_out[0][dominant_class_id] is not None else 0
                        
                        # Calculate soil diversity index (Shannon entropy)
                        # This measures the diversity of soil classes in the catchment
                        shannon_entropy = 0
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None and count is not None and count > 0:
                                p = count / total_pixels
                                shannon_entropy -= p * np.log(p)
                        
                        results["soil.diversity_index"] = shannon_entropy
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Process each HRU
                for i, zonal_result in enumerate(zonal_out):
                    try:
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate class fractions for this HRU
                        total_pixels = sum(count for count in zonal_result.values() if count is not None)
                        
                        if total_pixels > 0:
                            # Add soil class fractions
                            for class_id, count in zonal_result.items():
                                if class_id is not None:  # Skip None values
                                    class_name = soil_classes.get(class_id, f"class_{class_id}")
                                    fraction = (count / total_pixels) if count is not None else 0
                                    results[f"{prefix}soil.{class_name}_fraction"] = fraction
                            
                            # Find dominant soil class for this HRU
                            dominant_class_id = max(zonal_result.items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                            dominant_class_name = soil_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                            results[f"{prefix}soil.dominant_class"] = dominant_class_name
                            results[f"{prefix}soil.dominant_class_id"] = dominant_class_id
                            results[f"{prefix}soil.dominant_fraction"] = (zonal_result[dominant_class_id] / total_pixels) if zonal_result[dominant_class_id] is not None else 0
                            
                            # Calculate soil diversity for this HRU
                            shannon_entropy = 0
                            for class_id, count in zonal_result.items():
                                if class_id is not None and count is not None and count > 0:
                                    p = count / total_pixels
                                    shannon_entropy -= p * np.log(p)
                            
                            results[f"{prefix}soil.diversity_index"] = shannon_entropy
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {i}: {str(e)}")
            
            # Calculate additional soil properties if available
            self._calculate_soil_properties(soilclass_file, soil_classes, results, is_lumped)
        
        except Exception as e:
            self.logger.error(f"Error processing soil attributes: {str(e)}")
        
        return results

    def _get_soil_classes(self, soilclass_file: Path) -> Dict[int, str]:
        """
        Get soil class definitions.
        Attempts to determine class names from metadata or uses a default mapping.
        """
        # USDA soil texture classes
        usda_classes = {
            1: 'clay',
            2: 'silty_clay',
            3: 'sandy_clay',
            4: 'clay_loam',
            5: 'silty_clay_loam',
            6: 'sandy_clay_loam',
            7: 'loam',
            8: 'silty_loam',
            9: 'sandy_loam',
            10: 'silt',
            11: 'loamy_sand',
            12: 'sand'
        }
        
        # Try to read class definitions from raster metadata or config
        try:
            with rasterio.open(str(soilclass_file)) as src:
                # Check if there's soil class metadata
                if 'soil_classes' in src.tags():
                    # Parse metadata
                    class_info = src.tags()['soil_classes']
                    # Implement parsing logic based on the format
                    return {}  # Return parsed classes
        except Exception as e:
            self.logger.warning(f"Could not read soil class metadata: {str(e)}")
        
        # Check if soil class type is specified in config
        soil_type = self.config.get('SOIL_CLASS_TYPE', '').lower()
        
        if 'usda' in soil_type:
            return usda_classes
        elif 'fao' in soil_type:
            # FAO soil classes
            return {
                1: 'acrisols',
                2: 'alisols',
                3: 'andosols',
                4: 'arenosols',
                5: 'calcisols',
                6: 'cambisols',
                7: 'chernozems',
                8: 'ferralsols',
                9: 'fluvisols',
                10: 'gleysols',
                11: 'gypsisols',
                12: 'histosols',
                13: 'kastanozems',
                14: 'leptosols',
                15: 'luvisols',
                16: 'nitisols',
                17: 'phaeozems',
                18: 'planosols',
                19: 'plinthosols',
                20: 'podzols',
                21: 'regosols',
                22: 'solonchaks',
                23: 'solonetz',
                24: 'stagnosols',
                25: 'umbrisols',
                26: 'vertisols'
            }
        
        # Default to USDA classes as they're commonly used
        return usda_classes

    def _calculate_soil_properties(self, soilclass_file: Path, soil_classes: Dict[int, str], 
                                results: Dict[str, Any], is_lumped: bool) -> None:
        """
        Calculate derived soil properties based on soil classification.
        For example, estimate hydraulic parameters based on soil texture classes.
        """
        # USDA soil texture class to typical hydraulic parameters mapping
        # Values for: porosity, field capacity, wilting point, saturated hydraulic conductivity (mm/h)
        soil_params = {
            'clay': {'porosity': 0.50, 'field_capacity': 0.40, 'wilting_point': 0.27, 'ksat': 1.0},
            'silty_clay': {'porosity': 0.51, 'field_capacity': 0.39, 'wilting_point': 0.25, 'ksat': 1.5},
            'sandy_clay': {'porosity': 0.43, 'field_capacity': 0.36, 'wilting_point': 0.24, 'ksat': 3.0},
            'clay_loam': {'porosity': 0.47, 'field_capacity': 0.34, 'wilting_point': 0.22, 'ksat': 5.0},
            'silty_clay_loam': {'porosity': 0.49, 'field_capacity': 0.38, 'wilting_point': 0.22, 'ksat': 3.0},
            'sandy_clay_loam': {'porosity': 0.42, 'field_capacity': 0.28, 'wilting_point': 0.18, 'ksat': 10.0},
            'loam': {'porosity': 0.46, 'field_capacity': 0.28, 'wilting_point': 0.14, 'ksat': 20.0},
            'silty_loam': {'porosity': 0.48, 'field_capacity': 0.30, 'wilting_point': 0.15, 'ksat': 12.0},
            'sandy_loam': {'porosity': 0.44, 'field_capacity': 0.18, 'wilting_point': 0.08, 'ksat': 40.0},
            'silt': {'porosity': 0.50, 'field_capacity': 0.28, 'wilting_point': 0.10, 'ksat': 25.0},
            'loamy_sand': {'porosity': 0.42, 'field_capacity': 0.12, 'wilting_point': 0.05, 'ksat': 100.0},
            'sand': {'porosity': 0.40, 'field_capacity': 0.10, 'wilting_point': 0.04, 'ksat': 200.0}
        }
        
        # Only calculate these if we're using USDA soil texture classes
        soil_type = self.config.get('SOIL_CLASS_TYPE', '').lower()
        if not ('usda' in soil_type or all(class_name in soil_params for class_name in soil_classes.values())):
            self.logger.info("Skipping derived soil properties - non-USDA soil classification")
            return
        
        try:
            # Calculate area-weighted average soil properties for lumped catchment
            if is_lumped:
                # Extract dominant class and its fraction
                dominant_class = results.get("soil.dominant_class")
                
                if dominant_class and dominant_class in soil_params:
                    # Add dominant soil properties
                    for param, value in soil_params[dominant_class].items():
                        results[f"soil.dominant_{param}"] = value
                
                # Calculate weighted average properties across all soil classes
                weighted_props = {'porosity': 0, 'field_capacity': 0, 'wilting_point': 0, 'ksat': 0}
                total_fraction = 0
                
                for class_id, class_name in soil_classes.items():
                    fraction_key = f"soil.{class_name}_fraction"
                    if fraction_key in results and class_name in soil_params:
                        fraction = results[fraction_key]
                        if fraction > 0:
                            total_fraction += fraction
                            for param, value in soil_params[class_name].items():
                                weighted_props[param] += value * fraction
                
                # Normalize by total fraction (in case some classes don't have parameters)
                if total_fraction > 0:
                    for param, weighted_sum in weighted_props.items():
                        results[f"soil.avg_{param}"] = weighted_sum / total_fraction
            else:
                # For distributed catchment, calculate properties for each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Process each HRU
                for i, hru_id in enumerate(catchment[hru_id_field].values):
                    prefix = f"HRU_{hru_id}_"
                    
                    # Extract dominant class for this HRU
                    dominant_class = results.get(f"{prefix}soil.dominant_class")
                    
                    if dominant_class and dominant_class in soil_params:
                        # Add dominant soil properties
                        for param, value in soil_params[dominant_class].items():
                            results[f"{prefix}soil.dominant_{param}"] = value
                    
                    # Calculate weighted average properties across all soil classes for this HRU
                    weighted_props = {'porosity': 0, 'field_capacity': 0, 'wilting_point': 0, 'ksat': 0}
                    total_fraction = 0
                    
                    for class_id, class_name in soil_classes.items():
                        fraction_key = f"{prefix}soil.{class_name}_fraction"
                        if fraction_key in results and class_name in soil_params:
                            fraction = results[fraction_key]
                            if fraction > 0:
                                total_fraction += fraction
                                for param, value in soil_params[class_name].items():
                                    weighted_props[param] += value * fraction
                    
                    # Normalize by total fraction
                    if total_fraction > 0:
                        for param, weighted_sum in weighted_props.items():
                            results[f"{prefix}soil.avg_{param}"] = weighted_sum / total_fraction
        
        except Exception as e:
            self.logger.error(f"Error calculating soil properties: {str(e)}")

    def _process_elevation_attributes(self) -> Dict[str, float]:
        """Process elevation, slope, and aspect attributes."""
        results = {}
        
        try:
            # Find the DEM file
            dem_file = self.find_dem_file()
            
            # Generate slope and aspect
            raster_files = self.generate_slope_and_aspect(dem_file)
            
            # Calculate statistics for each raster
            for attribute_name, raster_file in raster_files.items():
                stats = self.calculate_statistics(raster_file, attribute_name)
                
                # Add statistics with proper prefixes
                for stat_name, value in stats.items():
                    if "." in stat_name:  # If stat_name already has hierarchical structure
                        results[stat_name] = value
                    else:
                        # Extract any HRU prefix if present
                        if stat_name.startswith("HRU_"):
                            hru_part = stat_name.split("_", 2)[0] + "_" + stat_name.split("_", 2)[1] + "_"
                            clean_stat = stat_name.replace(hru_part, "")
                            prefix = hru_part
                        else:
                            clean_stat = stat_name
                            prefix = ""
                        
                        # Remove attribute prefix if present in the stat name
                        clean_stat = clean_stat.replace(f"{attribute_name}_", "")
                        results[f"{prefix}{attribute_name}.{clean_stat}"] = value
        
        except Exception as e:
            self.logger.error(f"Error processing elevation attributes: {str(e)}")
        
        return results

    def _process_landcover_attributes(self) -> Dict[str, Any]:
        """Process land cover attributes including dominant land cover type."""
        results = {}
        
        # Find the land cover raster
        landclass_dir = self.project_dir / 'attributes' / 'landclass'
        if not landclass_dir.exists():
            self.logger.warning(f"Land cover directory not found: {landclass_dir}")
            return results
        
        # Search for land cover raster files
        landcover_files = list(landclass_dir.glob("*.tif"))
        if not landcover_files:
            self.logger.warning(f"No land cover raster files found in {landclass_dir}")
            return results
        
        # Use the first land cover file found
        landcover_file = landcover_files[0]
        self.logger.info(f"Processing land cover raster: {landcover_file}")
        
        try:
            # Get land cover classes and class names if available
            lc_classes = self._get_landcover_classes(landcover_file)
            
            # Calculate zonal statistics with categorical=True to get class counts
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(landcover_file), 
                categorical=True, 
                all_touched=True
            )
            
            # Process results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out:
                    # Calculate class fractions
                    total_pixels = sum(count for count in zonal_out[0].values() if count is not None)
                    
                    if total_pixels > 0:
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None:  # Skip None values
                                # Get class name if available, otherwise use class ID
                                class_name = lc_classes.get(class_id, f"class_{class_id}")
                                fraction = (count / total_pixels) if count is not None else 0
                                results[f"landcover.{class_name}_fraction"] = fraction
                        
                        # Find dominant land cover class
                        dominant_class_id = max(zonal_out[0].items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                        dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                        results["landcover.dominant_class"] = dominant_class_name
                        results["landcover.dominant_class_id"] = dominant_class_id
                        results["landcover.dominant_fraction"] = (zonal_out[0][dominant_class_id] / total_pixels) if zonal_out[0][dominant_class_id] is not None else 0
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Process each HRU
                for i, zonal_result in enumerate(zonal_out):
                    try:
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate class fractions for this HRU
                        total_pixels = sum(count for count in zonal_result.values() if count is not None)
                        
                        if total_pixels > 0:
                            for class_id, count in zonal_result.items():
                                if class_id is not None:  # Skip None values
                                    class_name = lc_classes.get(class_id, f"class_{class_id}")
                                    fraction = (count / total_pixels) if count is not None else 0
                                    results[f"{prefix}landcover.{class_name}_fraction"] = fraction
                            
                            # Find dominant land cover class for this HRU
                            dominant_class_id = max(zonal_result.items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                            dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                            results[f"{prefix}landcover.dominant_class"] = dominant_class_name
                            results[f"{prefix}landcover.dominant_class_id"] = dominant_class_id
                            results[f"{prefix}landcover.dominant_fraction"] = (zonal_result[dominant_class_id] / total_pixels) if zonal_result[dominant_class_id] is not None else 0
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {i}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing land cover attributes: {str(e)}")
        
        return results

    def _get_landcover_classes(self, landcover_file: Path) -> Dict[int, str]:
        """
        Get land cover class definitions.
        Attempts to determine class names from metadata or uses a default mapping.
        """
        # Default MODIS IGBP land cover classes
        modis_classes = {
            1: 'evergreen_needleleaf',
            2: 'evergreen_broadleaf',
            3: 'deciduous_needleleaf',
            4: 'deciduous_broadleaf',
            5: 'mixed_forest',
            6: 'closed_shrubland',
            7: 'open_shrubland',
            8: 'woody_savanna',
            9: 'savanna',
            10: 'grassland',
            11: 'wetland',
            12: 'cropland',
            13: 'urban',
            14: 'crop_natural_mosaic',
            15: 'snow_ice',
            16: 'barren',
            17: 'water'
        }
        
        # Try to read class definitions from raster metadata or config
        try:
            with rasterio.open(str(landcover_file)) as src:
                # Check if there's land cover class metadata
                if 'land_cover_classes' in src.tags():
                    # Parse metadata
                    class_info = src.tags()['land_cover_classes']
                    # Implement parsing logic based on the format
                    return {}  # Return parsed classes
        except Exception as e:
            self.logger.warning(f"Could not read land cover class metadata: {str(e)}")
        
        # Check if land cover type is specified in config
        lc_type = self.config.get('LAND_COVER_TYPE', '').lower()
        
        if 'modis' in lc_type or 'igbp' in lc_type:
            return modis_classes
        elif 'esa' in lc_type or 'cci' in lc_type:
            # ESA CCI land cover classes
            return {
                10: 'cropland_rainfed',
                20: 'cropland_irrigated',
                30: 'cropland_mosaic',
                40: 'forest_broadleaf_evergreen',
                50: 'forest_broadleaf_deciduous',
                60: 'forest_needleleaf_evergreen',
                70: 'forest_needleleaf_deciduous',
                80: 'forest_mixed',
                90: 'shrubland_mosaic',
                100: 'grassland',
                110: 'wetland',
                120: 'shrubland',
                130: 'grassland_sparse',
                140: 'lichens_mosses',
                150: 'sparse_vegetation',
                160: 'forest_flooded_freshwater',
                170: 'forest_flooded_saline',
                180: 'shrubland_flooded',
                190: 'urban',
                200: 'bare_areas',
                210: 'water',
                220: 'snow_ice'
            }
        
        # Default to generic class names
        return {i: f"class_{i}" for i in range(1, 30)}