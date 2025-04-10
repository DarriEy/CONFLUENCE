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
        Process DEM derivatives, land cover, soil, and forest height attributes.
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