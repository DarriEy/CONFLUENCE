import geopandas as gpd # type: ignore
import numpy as np # type: ignore
from typing import List, Dict, Any, Optional, Tuple
import rasterio # type: ignore
from rasterio.mask import mask # type: ignore
from shapely.geometry import Polygon, MultiPolygon, shape # type: ignore
from shapely.ops import unary_union # type: ignore
import matplotlib.pyplot as plt # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import pvlib # type: ignore
import pandas as pd # type: ignore
from pyproj import CRS # type: ignore
import rasterstats # type: ignore
import time

class DomainDiscretizer:
    """
    A class for discretizing a domain into Hydrologic Response Units (HRUs).

    This class provides methods for various types of domain discretization,
    including elevation-based, soil class-based, land class-based, and
    radiation-based discretization. HRUs are allowed to be MultiPolygons,
    meaning spatially disconnected areas with the same attributes are 
    grouped into single HRUs.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        logger: Logger object for logging information and errors.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        self.catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
        self.catchment_dir.mkdir(parents=True, exist_ok=True)
        delineation_method = self.config.get('DOMAIN_DEFINITION_METHOD')

        if delineation_method == 'delineate':
            self.delineation_suffix = 'delineate'
        elif delineation_method == 'lumped':
            self.delineation_suffix = 'lumped'
        elif delineation_method == 'subset':
            self.delineation_suffix = f"subset_{self.config['GEOFABRIC_TYPE']}"

    def sort_catchment_shape(self):
        """
        Sort the catchment shapefile based on GRU and HRU IDs.

        This method performs the following steps:
        1. Loads the catchment shapefile
        2. Sorts the shapefile based on GRU and HRU IDs
        3. Saves the sorted shapefile back to the original location

        The method uses GRU and HRU ID column names specified in the configuration.

        Raises:
            FileNotFoundError: If the catchment shapefile is not found.
            ValueError: If the required ID columns are not present in the shapefile.
        """
        self.logger.info("Sorting catchment shape")

        self.catchment_path = self.config.get('CATCHMENT_PATH')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
            # Handle comma-separated attributes for output filename
            if ',' in discretization_method:
                method_suffix = discretization_method.replace(',', '_')
            else:
                method_suffix = discretization_method
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{method_suffix}.shp"
        self.gruId = self.config.get('CATCHMENT_SHP_GRUID')
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')

        if self.catchment_path == 'default':
            self.catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            self.catchment_path = Path(self.catchment_path)

        catchment_file = self.catchment_path / self.catchment_name
        
        try:
            # Open the shape
            shp = gpd.read_file(catchment_file)
            
            # Check if required columns exist
            if self.gruId not in shp.columns or self.hruId not in shp.columns:
                raise ValueError(f"Required columns {self.gruId} and/or {self.hruId} not found in shapefile")
            
            # Sort
            shp = shp.sort_values(by=[self.gruId, self.hruId])
            
            # Save
            shp.to_file(catchment_file)
            
            self.logger.info(f"Catchment shape sorted and saved to {catchment_file}")
        except FileNotFoundError:
            self.logger.error(f"Catchment shapefile not found at {catchment_file}")
            raise
        except ValueError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Error sorting catchment shape: {str(e)}")
            raise

    def discretize_domain(self) -> Optional[Path]:
        """
        Discretize the domain based on the method specified in the configuration.
        If CATCHMENT_SHP_NAME is provided and not 'default', it uses the provided shapefile instead.
        Supports both single attributes and comma-separated multiple attributes.
        """
        start_time = time.time()
        
        # Check if a custom catchment shapefile is provided
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if catchment_name != 'default':
            self.logger.info(f"Using provided catchment shapefile: {catchment_name}")
            self.logger.info("Skipping discretization steps")
            
            # Just sort the existing shapefile
            self.logger.info("Sorting provided catchment shape")
            shp = self.sort_catchment_shape()
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Catchment processing completed in {elapsed_time:.2f} seconds")
            return shp
        
        # Parse discretization method to check for multiple attributes
        discretization_config = self.config.get('DOMAIN_DISCRETIZATION')
        attributes = [attr.strip() for attr in discretization_config.split(',')]
        
        self.logger.info(f"Starting domain discretization using attributes: {attributes}")

        # Handle single vs multiple attributes
        if len(attributes) == 1:
            # Single attribute - use existing logic
            discretization_method = attributes[0].lower()
            method_map = {
                'grus': self._use_grus_as_hrus,
                'elevation': self._discretize_by_elevation,
                'aspect': self._discretize_by_aspect,
                'soilclass': self._discretize_by_soil_class,
                'landclass': self._discretize_by_land_class,
                'radiation': self._discretize_by_radiation
            }

            if discretization_method not in method_map:
                self.logger.error(f"Invalid discretization method: {discretization_method}")
                raise ValueError(f"Invalid discretization method: {discretization_method}")

            self.logger.info("Step 1/2: Running single attribute discretization method")
            method_map[discretization_method]()
        else:
            # Multiple attributes - use combined discretization
            self.logger.info("Step 1/2: Running combined attributes discretization method")
            self._discretize_combined(attributes)
        
        self.logger.info("Step 2/2: Sorting catchment shape")
        shp = self.sort_catchment_shape()

        elapsed_time = time.time() - start_time
        self.logger.info(f"Domain discretization completed in {elapsed_time:.2f} seconds")
        return shp

    def _discretize_combined(self, attributes: List[str]):
        """
        Discretize the domain based on a combination of geospatial attributes.
        
        Args:
            attributes: List of attribute names to combine (e.g., ['elevation', 'landclass'])
        """
        self.logger.info(f"Starting combined discretization with attributes: {attributes}")
        
        # Get GRU shapefile
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               self.config.get('RIVER_BASINS_NAME'))
        
        # Generate output filename
        method_suffix = '_'.join(attributes)
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", 
                                              f"{self.domain_name}_HRUs_{method_suffix}.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", 
                                         f"{self.domain_name}_HRUs_{method_suffix}.png")
        
        # Get raster paths and thresholds for each attribute
        raster_info = self._get_raster_info_for_attributes(attributes)
        
        # Read GRU data
        gru_gdf = self._read_shapefile(gru_shapefile)
        
        # Create combined HRUs
        hru_gdf = self._create_combined_attribute_hrus(gru_gdf, raster_info, attributes)
        
        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Combined attribute HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            # Create plot with combined attributes
            plot_column = f"combined_{method_suffix}"
            self._plot_hrus(hru_gdf, output_plot, plot_column, f'Combined {method_suffix.replace("_", " + ")} HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _get_raster_info_for_attributes(self, attributes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get raster paths and classification information for each attribute.
        
        Args:
            attributes: List of attribute names
            
        Returns:
            Dictionary containing raster path and classification info for each attribute
        """
        raster_info = {}
        
        for attr in attributes:
            attr_lower = attr.lower()
            
            if attr_lower == 'elevation':
                dem_name = self.config['DEM_NAME']
                if dem_name == "default":
                    dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                
                raster_path = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                band_size = float(self.config.get('ELEVATION_BAND_SIZE'))
                
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'continuous',
                    'band_size': band_size,
                    'class_name': 'elevClass'
                }
                
            elif attr_lower == 'soilclass':
                raster_path = self._get_file_path("SOIL_CLASS_PATH", "attributes/soilclass/", 
                                                f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif")
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'discrete',
                    'class_name': 'soilClass'
                }
                
            elif attr_lower == 'landclass':
                raster_path = self._get_file_path("LAND_CLASS_PATH", "attributes/landclass", 
                                                f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif")
                raster_info[attr] = {
                    'path': raster_path,
                    'type': 'discrete',
                    'class_name': 'landClass'
                }
                
            elif attr_lower == 'radiation':
                radiation_raster = self._get_file_path("RADIATION_PATH", "attributes/radiation", 
                                                     "annual_radiation.tif")
                
                # Calculate radiation if it doesn't exist
                if not radiation_raster.exists():
                    self.logger.info("Annual radiation raster not found. Calculating radiation...")
                    dem_name = self.config['DEM_NAME']
                    if dem_name == "default":
                        dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                    dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                    radiation_raster = self._calculate_annual_radiation(dem_raster, radiation_raster)
                    if radiation_raster is None:
                        raise ValueError("Failed to calculate annual radiation")
                
                radiation_class_number = int(self.config.get('RADIATION_CLASS_NUMBER'))
                
                raster_info[attr] = {
                    'path': radiation_raster,
                    'type': 'continuous',
                    'band_size': radiation_class_number,
                    'class_name': 'radiationClass'
                }
            elif attr_lower == 'aspect':
                aspect_raster = self._get_file_path("ASPECT_PATH", "attributes/aspect", "aspect.tif")
                
                # Calculate aspect if it doesn't exist
                if not aspect_raster.exists():
                    self.logger.info("Aspect raster not found. Calculating aspect...")
                    dem_name = self.config['DEM_NAME']
                    if dem_name == "default":
                        dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
                    dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
                    aspect_raster = self._calculate_aspect(dem_raster, aspect_raster)
                    if aspect_raster is None:
                        raise ValueError("Failed to calculate aspect")
                
                raster_info[attr] = {
                    'path': aspect_raster,
                    'type': 'discrete',
                    'class_name': 'aspectClass'
                }

            else:
                raise ValueError(f"Unsupported attribute for discretization: {attr}")
        
        return raster_info

    def _create_combined_attribute_hrus(self, gru_gdf: gpd.GeoDataFrame, 
                                       raster_info: Dict[str, Dict[str, Any]], 
                                       attributes: List[str]) -> gpd.GeoDataFrame:
        """
        Create HRUs based on unique combinations of multiple attributes within each GRU.
        
        Args:
            gru_gdf: GeoDataFrame containing GRU data
            raster_info: Dictionary containing raster information for each attribute
            attributes: List of attribute names
            
        Returns:
            GeoDataFrame containing combined attribute HRUs
        """
        self.logger.info(f"Creating combined attribute HRUs within {len(gru_gdf)} GRUs")
        
        all_hrus = []
        hru_id_counter = 1
        
        # Process each GRU individually
        for gru_idx, gru_row in gru_gdf.iterrows():
            self.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")
            
            gru_geometry = gru_row.geometry
            gru_id = gru_row.get('GRU_ID', gru_idx + 1)
            
            # Extract all raster data for this GRU
            raster_data = {}
            common_transform = None
            common_shape = None
            
            for attr in attributes:
                attr_info = raster_info[attr]
                raster_path = attr_info['path']
                
                try:
                    with rasterio.open(raster_path) as src:
                        out_image, out_transform = mask(src, [gru_geometry], crop=True, 
                                                       all_touched=True, filled=False)
                        out_image = out_image[0]
                        nodata_value = src.nodata
                        
                        # Store raster data and metadata
                        raster_data[attr] = {
                            'data': out_image,
                            'nodata': nodata_value,
                            'info': attr_info
                        }
                        
                        # Set common transform and shape from first raster
                        if common_transform is None:
                            common_transform = out_transform
                            common_shape = out_image.shape
                        
                except Exception as e:
                    self.logger.warning(f"Could not extract {attr} raster data for GRU {gru_id}: {str(e)}")
                    continue
            
            if not raster_data:
                self.logger.warning(f"No valid raster data found for GRU {gru_id}")
                continue
            
            # Create combined valid mask (pixels that are valid in all rasters)
            combined_valid_mask = np.ones(common_shape, dtype=bool)
            for attr, data in raster_data.items():
                raster_array = data['data']
                nodata_value = data['nodata']
                
                if nodata_value is not None:
                    valid_mask = raster_array != nodata_value
                else:
                    valid_mask = ~np.isnan(raster_array) if raster_array.dtype == np.float64 else np.ones_like(raster_array, dtype=bool)
                
                combined_valid_mask &= valid_mask
            
            if not np.any(combined_valid_mask):
                self.logger.warning(f"No valid pixels found in GRU {gru_id}")
                continue
            
            # Classify each attribute and find unique combinations
            classified_data = {}
            for attr in attributes:
                data_info = raster_data[attr]
                raster_array = data_info['data']
                attr_info = data_info['info']
                
                if attr_info['type'] == 'continuous':
                    # Classify continuous data into bands
                    classified_data[attr] = self._classify_continuous_data(
                        raster_array, combined_valid_mask, attr_info['band_size']
                    )
                else:
                    # Use discrete values directly
                    classified_data[attr] = raster_array
            
            # Find unique combinations of classified values
            unique_combinations = self._find_unique_combinations(classified_data, combined_valid_mask)
            
            # Create HRUs for each unique combination
            gru_hrus = self._create_hrus_from_combinations(
                unique_combinations, classified_data, combined_valid_mask, 
                common_transform, gru_geometry, gru_row, hru_id_counter, attributes
            )
            
            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)
        
        self.logger.info(f"Created {len(all_hrus)} combined attribute HRUs across all GRUs")
        return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)

    def _classify_continuous_data(self, raster_array: np.ndarray, valid_mask: np.ndarray, 
                                 band_size: float) -> np.ndarray:
        """
        Classify continuous raster data into discrete bands.
        
        Args:
            raster_array: Input raster array
            valid_mask: Boolean mask for valid pixels
            band_size: Size of bands (for elevation) or number of classes (for radiation)
            
        Returns:
            Classified array with discrete class values
        """
        valid_data = raster_array[valid_mask]
        
        if len(valid_data) == 0:
            return raster_array.copy()
        
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        # Create classification based on band_size
        if isinstance(band_size, int) and band_size < 50:  # Assume it's number of classes for radiation
            # Use quantile-based classification
            quantiles = np.linspace(0, 1, band_size + 1)
            thresholds = np.quantile(valid_data, quantiles)
        else:
            # Use fixed band size for elevation
            thresholds = np.arange(data_min, data_max + band_size, band_size)
            if thresholds[-1] < data_max:
                thresholds = np.append(thresholds, thresholds[-1] + band_size)
        
        # Classify the data
        classified = np.zeros_like(raster_array, dtype=int)
        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i:i+2]
            if i == len(thresholds) - 2:  # Last band
                mask = valid_mask & (raster_array >= lower) & (raster_array <= upper)
            else:
                mask = valid_mask & (raster_array >= lower) & (raster_array < upper)
            classified[mask] = i + 1
        
        return classified

    def _find_unique_combinations(self, classified_data: Dict[str, np.ndarray], 
                                 valid_mask: np.ndarray) -> List[Tuple]:
        """
        Find unique combinations of classified values across all attributes.
        
        Args:
            classified_data: Dictionary of classified raster arrays for each attribute
            valid_mask: Boolean mask for valid pixels
            
        Returns:
            List of unique value combinations
        """
        # Stack all classified arrays
        stacked_data = []
        for attr in sorted(classified_data.keys()):
            stacked_data.append(classified_data[attr][valid_mask])
        
        # Find unique combinations
        combined_array = np.column_stack(stacked_data)
        unique_combinations = [tuple(row) for row in np.unique(combined_array, axis=0)]
        
        return unique_combinations

    def _create_hrus_from_combinations(self, unique_combinations: List[Tuple], 
                                      classified_data: Dict[str, np.ndarray],
                                      valid_mask: np.ndarray, transform: Any,
                                      gru_geometry: Any, gru_row: Any, 
                                      start_hru_id: int, attributes: List[str]) -> List[Dict]:
        """
        Create HRUs for each unique combination of attribute values.
        
        Args:
            unique_combinations: List of unique value combinations
            classified_data: Dictionary of classified raster arrays
            valid_mask: Boolean mask for valid pixels
            transform: Raster transform
            gru_geometry: GRU geometry
            gru_row: GRU data row
            start_hru_id: Starting HRU ID
            attributes: List of attribute names
            
        Returns:
            List of HRU dictionaries
        """
        hrus = []
        current_hru_id = start_hru_id
        
        for combination in unique_combinations:
            # Create mask for this combination
            combination_mask = valid_mask.copy()
            
            for i, attr in enumerate(sorted(classified_data.keys())):
                attr_value = combination[i]
                combination_mask &= (classified_data[attr] == attr_value)
            
            if not np.any(combination_mask):
                continue
            
            # Create HRU from this combination
            hru = self._create_hru_from_combination_mask(
                combination_mask, transform, classified_data, gru_geometry,
                gru_row, current_hru_id, attributes, combination
            )
            
            if hru:
                hrus.append(hru)
                current_hru_id += 1
        
        return hrus

    def _create_hru_from_combination_mask(self, combination_mask: np.ndarray, transform: Any,
                                         classified_data: Dict[str, np.ndarray], 
                                         gru_geometry: Any, gru_row: Any, hru_id: int,
                                         attributes: List[str], combination: Tuple) -> Optional[Dict]:
        """
        Create a single HRU from a combination mask.
        
        Args:
            combination_mask: Boolean mask for the combination
            transform: Raster transform
            classified_data: Dictionary of classified raster arrays
            gru_geometry: GRU geometry
            gru_row: GRU data row
            hru_id: HRU ID
            attributes: List of attribute names
            combination: Tuple of attribute values for this combination
            
        Returns:
            Dictionary representing the HRU or None if creation fails
        """
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                combination_mask.astype(np.uint8), 
                mask=combination_mask, 
                transform=transform,
                connectivity=4
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                try:
                    geom = shape(shp)
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        polygons.append(geom)
                except Exception:
                    continue
            
            if not polygons:
                return None
            
            # Create final geometry
            if len(polygons) == 1:
                final_geometry = polygons[0]
            else:
                final_geometry = MultiPolygon(polygons)
            
            # Clean the geometry
            if not final_geometry.is_valid:
                final_geometry = final_geometry.buffer(0)
            
            if final_geometry.is_empty or not final_geometry.is_valid:
                return None
            
            # Ensure it's within the GRU boundary
            clipped_geometry = final_geometry.intersection(gru_geometry)
            
            if clipped_geometry.is_empty or not clipped_geometry.is_valid:
                return None
            
            # Create HRU data with combination attributes
            hru_data = {
                'geometry': clipped_geometry,
                'GRU_ID': gru_row.get('GRU_ID', gru_row.name),
                'HRU_ID': hru_id,
                'hru_type': f'combined_{"_".join(attributes)}'
            }
            
            # Add individual attribute values
            for i, attr in enumerate(sorted(attributes)):
                attr_name = attr.lower()
                if attr_name == 'elevation':
                    hru_data['elevClass'] = combination[i]
                elif attr_name == 'soilclass':
                    hru_data['soilClass'] = combination[i]
                elif attr_name == 'landclass':
                    hru_data['landClass'] = combination[i]
                elif attr_name == 'radiation':
                    hru_data['radiationClass'] = combination[i]
            
            # Add combined attribute identifier
            combined_id = '_'.join([str(val) for val in combination])
            combined_name = f"combined_{'_'.join(attributes)}"
            hru_data[combined_name] = combined_id
            
            # Copy relevant GRU attributes (excluding geometry)
            for col in gru_row.index:
                if col not in ['geometry', 'GRU_ID'] and col not in hru_data:
                    hru_data[col] = gru_row[col]
            
            return hru_data
            
        except Exception as e:
            self.logger.warning(f"Error creating HRU for combination {combination}: {str(e)}")
            return None

    def _use_grus_as_hrus(self):
        """
        Use Grouped Response Units (GRUs) as Hydrologic Response Units (HRUs) without further discretization.

        Returns:
            Path: Path to the output HRU shapefile.
        """
        self.logger.info(f"config domain name {self.config.get('DOMAIN_NAME')}")
        if self.config.get('RIVER_BASINS_NAME') == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp")
        
            if self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
                gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_with_coastal.shp")
        
            elif self.config.get('DOMAIN_DEFINITION_METHOD') == "point":
                gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_point.shp")
            
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))
        
        hru_output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_GRUs.shp")

        gru_gdf = self._read_shapefile(gru_shapefile)
        gru_gdf['HRU_ID'] = range(1, len(gru_gdf) + 1)
        gru_gdf['hru_type'] = 'GRU'

        # Calculate mean elevation for each HRU with proper CRS handling
        self.logger.info("Calculating mean elevation for each HRU")
        
        # Get CRS information
        with rasterio.open(self.dem_path) as src:
            dem_crs = src.crs
            self.logger.info(f"DEM CRS: {dem_crs}")
        
        shapefile_crs = gru_gdf.crs
        self.logger.info(f"Shapefile CRS: {shapefile_crs}")
        
        # Check if CRS match
        if dem_crs != shapefile_crs:
            self.logger.info(f"CRS mismatch detected. Reprojecting shapefile from {shapefile_crs} to {dem_crs}")
            gru_gdf_projected = gru_gdf.to_crs(dem_crs)
        else:
            self.logger.info("CRS match - no reprojection needed")
            gru_gdf_projected = gru_gdf.copy()
        
        # Use rasterstats with the raster file path directly (more efficient and handles CRS properly)
        try:
            zs = rasterstats.zonal_stats(
                gru_gdf_projected.geometry, 
                str(self.dem_path),  # Use file path instead of array
                stats=['mean'],
                nodata=-9999  # Explicit nodata value
            )
            gru_gdf['elev_mean'] = [item['mean'] if item['mean'] is not None else -9999 for item in zs]
            self.logger.info(f"Successfully calculated elevation statistics for {len(gru_gdf)} HRUs")
            
        except Exception as e:
            self.logger.error(f"Error calculating zonal statistics: {str(e)}")
            # Fallback: set all elevation means to -9999
            gru_gdf['elev_mean'] = -9999
            self.logger.warning("Setting all elevation means to -9999 due to calculation error")
        
        # Calculate centroids in projected CRS for accuracy
        # Project to UTM for accurate centroid calculation if not already in UTM
        try:
            if gru_gdf.crs.is_geographic:
                utm_crs = gru_gdf.estimate_utm_crs()
                gru_gdf_utm = gru_gdf.to_crs(utm_crs)
            else:
                # Already in projected coordinate system
                gru_gdf_utm = gru_gdf.copy()
                utm_crs = gru_gdf.crs
            
            centroids_utm = gru_gdf_utm.geometry.centroid
            centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
            
            gru_gdf['center_lon'] = centroids_wgs84.x
            gru_gdf['center_lat'] = centroids_wgs84.y
            
            self.logger.info(f"Calculated centroids in WGS84: lat range {centroids_wgs84.y.min():.6f} to {centroids_wgs84.y.max():.6f}, lon range {centroids_wgs84.x.min():.6f} to {centroids_wgs84.x.max():.6f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating centroids: {str(e)}")
            # Fallback: try to use existing center_lat/center_lon if they exist and look reasonable
            if 'center_lat' in gru_gdf.columns and 'center_lon' in gru_gdf.columns:
                # Check if existing values look like actual lat/lon (rough check)
                if (gru_gdf['center_lat'].between(-90, 90).all() and 
                    gru_gdf['center_lon'].between(-180, 180).all()):
                    self.logger.info("Using existing center_lat/center_lon coordinates")
                else:
                    self.logger.warning("Existing center_lat/center_lon appear to be in projected coordinates, setting to default values")
                    gru_gdf['center_lat'] = 0.0
                    gru_gdf['center_lon'] = 0.0
            else:
                gru_gdf['center_lat'] = 0.0
                gru_gdf['center_lon'] = 0.0
        
        if 'COMID' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['COMID']
        elif 'fid' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['fid']

        gru_gdf['HRU_area'] = gru_gdf['GRU_area']
        gru_gdf['HRU_ID'] = gru_gdf['GRU_ID']        

        gru_gdf.to_file(hru_output_shapefile)
        self.logger.info(f"GRUs saved as HRUs to {hru_output_shapefile}")

        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_as_GRUs.png")
        self._plot_hrus(gru_gdf, output_plot, 'HRU_ID', 'GRUs = HRUs')
        return hru_output_shapefile

    def _discretize_by_elevation(self):
        """
        Discretize the domain based on elevation within each GRU.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins__with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))
        
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_elevation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_elevation.png")

        elevation_band_size = float(self.config.get('ELEVATION_BAND_SIZE'))
        gru_gdf, elevation_thresholds = self._read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, dem_raster, elevation_thresholds, 'elevClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Elevation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'elevClass', 'Elevation-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_soil_class(self):
        """
        Discretize the domain based on soil classifications using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        soil_raster = self._get_file_path("SOIL_CLASS_PATH", "attributes/soilclass/", f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_soilclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_soilclass.png")

        gru_gdf, soil_classes = self._read_and_prepare_data(gru_shapefile, soil_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, soil_raster, soil_classes, 'soilClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Soil-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'soilClass', 'Soil-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_land_class(self):
        """
        Discretize the domain based on land cover classifications using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins__with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))

        land_raster = self._get_file_path("LAND_CLASS_PATH","attributes/landclass", f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_landclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_landclass.png")

        gru_gdf, land_classes = self._read_and_prepare_data(gru_shapefile, land_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, land_raster, land_classes, 'landClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'landClass', 'Land-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_aspect(self):
        """
        Discretize the domain based on aspect (slope direction) using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        elif self.config.get('DELINEATE_COASTAL_WATERSHEDS') == True:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               f"{self.domain_name}_riverBasins_with_coastal.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", 
                                               self.config.get('RIVER_BASINS_NAME'))

        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        aspect_raster = self._get_file_path("ASPECT_PATH", "attributes/aspect", "aspect.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", 
                                              f"{self.domain_name}_HRUs_aspect.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", 
                                         f"{self.domain_name}_HRUs_aspect.png")

        aspect_class_number = int(self.config.get('ASPECT_CLASS_NUMBER', 8))

        if not aspect_raster.exists():
            self.logger.info("Aspect raster not found. Calculating aspect...")
            aspect_raster = self._calculate_aspect(dem_raster, aspect_raster)
            if aspect_raster is None:
                raise ValueError("Failed to calculate aspect")

        gru_gdf, aspect_classes = self._read_and_prepare_data(gru_shapefile, aspect_raster)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, aspect_raster, aspect_classes, 'aspectClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Aspect-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'aspectClass', 'Aspect-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _calculate_aspect(self, dem_raster: Path, aspect_raster: Path) -> Path:
            """
            Calculate aspect (slope direction) from DEM and classify into directional classes.
            
            Args:
                dem_raster: Path to the DEM raster
                aspect_raster: Path where the aspect raster will be saved
                
            Returns:
                Path to the created aspect raster
            """
            self.logger.info(f"Calculating aspect from DEM: {dem_raster}")
            
            try:
                with rasterio.open(dem_raster) as src:
                    dem = src.read(1)
                    transform = src.transform
                    crs = src.crs
                    nodata = src.nodata
                
                # Calculate gradients
                dy, dx = np.gradient(dem.astype(float))
                
                # Calculate aspect in radians, then convert to degrees
                aspect_rad = np.arctan2(-dx, dy)  # Note the negative sign for dx
                aspect_deg = np.degrees(aspect_rad)
                
                # Convert to compass bearing (0-360 degrees, 0 = North)
                aspect_deg = (90 - aspect_deg) % 360
                
                # Handle flat areas (where both dx and dy are near zero)
                slope_magnitude = np.sqrt(dx*dx + dy*dy)
                flat_threshold = 1e-6  # Adjust as needed
                flat_mask = slope_magnitude < flat_threshold
                
                # Classify aspect into directional classes
                aspect_class_number = int(self.config.get('ASPECT_CLASS_NUMBER', 8))
                classified_aspect = self._classify_aspect_into_classes(aspect_deg, flat_mask, aspect_class_number)
                
                # Handle nodata values from original DEM
                if nodata is not None:
                    dem_nodata_mask = dem == nodata
                    classified_aspect[dem_nodata_mask] = -9999
                
                # Save the classified aspect raster
                aspect_raster.parent.mkdir(parents=True, exist_ok=True)
                
                with rasterio.open(aspect_raster, 'w', driver='GTiff',
                                height=classified_aspect.shape[0], width=classified_aspect.shape[1],
                                count=1, dtype=classified_aspect.dtype,
                                crs=crs, transform=transform, nodata=-9999) as dst:
                    dst.write(classified_aspect, 1)
                
                self.logger.info(f"Aspect raster saved to: {aspect_raster}")
                self.logger.info(f"Aspect classes: {np.unique(classified_aspect[classified_aspect != -9999])}")
                return aspect_raster
            
            except Exception as e:
                self.logger.error(f"Error calculating aspect: {str(e)}", exc_info=True)
                return None

    def _classify_aspect_into_classes(self, aspect_deg: np.ndarray, flat_mask: np.ndarray, 
                                    num_classes: int) -> np.ndarray:
        """
        Classify aspect degrees into directional classes.
        
        Args:
            aspect_deg: Aspect in degrees (0-360)
            flat_mask: Boolean mask for flat areas
            num_classes: Number of aspect classes to create
            
        Returns:
            Classified aspect array
        """
        classified = np.zeros_like(aspect_deg, dtype=int)
        
        if num_classes == 8:
            # Standard 8-direction classification
            # N, NE, E, SE, S, SW, W, NW
            bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
            labels = [1, 2, 3, 4, 5, 6, 7, 8, 1]  # Last one wraps to North
            
            for i in range(len(bins) - 1):
                if i == len(bins) - 2:  # Last bin (337.5 to 360)
                    mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
                else:
                    mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
                classified[mask] = labels[i]
                
        elif num_classes == 4:
            # 4-direction classification (N, E, S, W)
            bins = [0, 45, 135, 225, 315, 360]
            labels = [1, 2, 3, 4, 1]  # N, E, S, W, N
            
            for i in range(len(bins) - 1):
                if i == len(bins) - 2:  # Last bin
                    mask = (aspect_deg >= bins[i]) & (aspect_deg <= bins[i+1])
                else:
                    mask = (aspect_deg >= bins[i]) & (aspect_deg < bins[i+1])
                classified[mask] = labels[i]
        
        else:
            # Custom number of classes - divide 360 degrees evenly
            class_width = 360.0 / num_classes
            for i in range(num_classes):
                lower = i * class_width
                upper = (i + 1) * class_width
                
                if i == num_classes - 1:  # Last class includes 360
                    mask = (aspect_deg >= lower) & (aspect_deg <= upper)
                else:
                    mask = (aspect_deg >= lower) & (aspect_deg < upper)
                classified[mask] = i + 1
        
        # Set flat areas to a special class (0)
        classified[flat_mask] = 0
        
        # Set areas that don't fall into any class to -9999 (shouldn't happen but safety)
        classified[classified == 0] = 0  # Keep flat areas as 0
        
        return classified

    def _discretize_by_radiation(self):
        """
        Discretize the domain based on radiation properties using MultiPolygon HRUs.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", dem_name)
        radiation_raster = self._get_file_path("RADIATION_PATH", "attributes/radiation", "annual_radiation.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_radiation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_radiation.png")

        radiation_class_number = int(self.config.get('RADIATION_CLASS_NUMBER'))

        if not radiation_raster.exists():
            self.logger.info("Annual radiation raster not found. Calculating radiation...")
            radiation_raster = self._calculate_annual_radiation(dem_raster, radiation_raster)
            if radiation_raster is None:
                raise ValueError("Failed to calculate annual radiation")

        gru_gdf, radiation_thresholds = self._read_and_prepare_data(gru_shapefile, radiation_raster, radiation_class_number)
        hru_gdf = self._create_multipolygon_hrus(gru_gdf, radiation_raster, radiation_thresholds, 'radiationClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Radiation-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'radiationClass', 'Radiation-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _calculate_annual_radiation(self, dem_raster: Path, radiation_raster: Path) -> Path:
        self.logger.info(f"Calculating annual radiation from DEM: {dem_raster}")
        
        try:
            with rasterio.open(dem_raster) as src:
                dem = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            # Calculate slope and aspect
            dy, dx = np.gradient(dem)
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            aspect = np.arctan2(-dx, dy)
            
            # Create a DatetimeIndex for the entire year (daily)
            times = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')
            
            # Create location object
            location = pvlib.location.Location(latitude=center_lat, longitude=center_lon, altitude=np.mean(dem))
            
            # Calculate solar position
            solar_position = location.get_solarposition(times=times)
            
            # Calculate clear sky radiation
            clearsky = location.get_clearsky(times=times)
            
            # Initialize the radiation array
            radiation = np.zeros_like(dem)
            
            self.logger.info("Calculating radiation for each pixel...")
            for i in range(dem.shape[0]):
                for j in range(dem.shape[1]):
                    surface_tilt = np.degrees(slope[i, j])
                    surface_azimuth = np.degrees(aspect[i, j])
                    
                    total_irrad = pvlib.irradiance.get_total_irradiance(
                        surface_tilt, surface_azimuth,
                        solar_position['apparent_zenith'], solar_position['azimuth'],
                        clearsky['dni'], clearsky['ghi'], clearsky['dhi']
                    )
                    
                    radiation[i, j] = total_irrad['poa_global'].sum()
            
            # Save the radiation raster
            radiation_raster.parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(radiation_raster, 'w', driver='GTiff',
                            height=radiation.shape[0], width=radiation.shape[1],
                            count=1, dtype=radiation.dtype,
                            crs=crs, transform=transform) as dst:
                dst.write(radiation, 1)
            
            self.logger.info(f"Radiation raster saved to: {radiation_raster}")
            return radiation_raster
        
        except Exception as e:
            self.logger.error(f"Error calculating annual radiation: {str(e)}", exc_info=True)
            return None

    def _read_and_prepare_data(self, shapefile_path, raster_path, band_size=None):
        """
        Read and prepare data with chunking for large rasters.
        
        Args:
            shapefile_path: Path to the GRU shapefile
            raster_path: Path to the raster file
            band_size: Optional band size for discretization
            
        Returns:
            tuple: (gru_gdf, thresholds) where:
                - gru_gdf is the GeoDataFrame containing GRU data
                - thresholds are the class boundaries for discretization
        """
        # Read the GRU shapefile
        gru_gdf = self._read_shapefile(shapefile_path)
        
        # Process raster in chunks
        CHUNK_SIZE = 1024  # Adjust based on available memory
        valid_data = []
        
        with rasterio.open(raster_path) as src:
            height = src.height
            width = src.width
            nodata = src.nodata
            
            self.logger.info(f"Raster info: {width}x{height} pixels, nodata={nodata}")
            
            for y in range(0, height, CHUNK_SIZE):
                for x in range(0, width, CHUNK_SIZE):
                    window = rasterio.windows.Window(x, y, 
                        min(CHUNK_SIZE, width - x),
                        min(CHUNK_SIZE, height - y))
                    chunk = src.read(1, window=window)
                    
                    # Filter out nodata values
                    if nodata is not None:
                        valid_chunk = chunk[chunk != nodata]
                    else:
                        valid_chunk = chunk[~np.isnan(chunk)] if chunk.dtype == np.float64 else chunk
                    
                    if len(valid_chunk) > 0:
                        valid_data.extend(valid_chunk.flatten())
        
        if len(valid_data) == 0:
            raise ValueError("No valid data found in raster")
        
        valid_data = np.array(valid_data)
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        self.logger.info(f"Valid data range: {data_min:.2f} to {data_max:.2f}")
        self.logger.info(f"Total valid pixels: {len(valid_data)}")
        
        # Calculate thresholds based on the data
        if band_size is not None:
            # For elevation-based or radiation-based discretization
            # Ensure thresholds cover the full data range
            min_val = data_min
            max_val = data_max
            
            # Create bands that fully cover the data range
            thresholds = np.arange(min_val, max_val + band_size, band_size)
            
            # Ensure the last threshold covers the maximum value
            if thresholds[-1] < max_val:
                thresholds = np.append(thresholds, thresholds[-1] + band_size)
            
            self.logger.info(f"Created {len(thresholds)-1} bands with size {band_size}")
            self.logger.info(f"Threshold range: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
        else:
            # For soil or land class-based discretization
            thresholds = np.unique(valid_data)
            self.logger.info(f"Found {len(thresholds)} unique classes: {thresholds}")
        
        return gru_gdf, thresholds

    def _create_multipolygon_hrus(self, gru_gdf, raster_path, thresholds, attribute_name):
        """
        Create HRUs by discretizing each GRU based on raster values within it.
        Each unique raster value within a GRU becomes an HRU (Polygon or MultiPolygon).
        
        Args:
            gru_gdf: GeoDataFrame containing GRU data
            raster_path: Path to the classification raster
            thresholds: Array of threshold values for classification
            attribute_name: Name of the attribute column
            
        Returns:
            GeoDataFrame containing HRUs
        """
        self.logger.info(f"Creating HRUs within {len(gru_gdf)} GRUs based on {attribute_name}")
        
        all_hrus = []
        hru_id_counter = 1
        
        # Process each GRU individually
        for gru_idx, gru_row in gru_gdf.iterrows():
            self.logger.info(f"Processing GRU {gru_idx + 1}/{len(gru_gdf)}")
            
            gru_geometry = gru_row.geometry
            gru_id = gru_row.get('GRU_ID', gru_idx + 1)
            
            # Extract raster data within this GRU
            with rasterio.open(raster_path) as src:
                try:
                    # Mask the raster to this GRU's geometry
                    out_image, out_transform = mask(src, [gru_geometry], crop=True, all_touched=True, filled=False)
                    out_image = out_image[0]
                    nodata_value = src.nodata
                except Exception as e:
                    self.logger.warning(f"Could not extract raster data for GRU {gru_id}: {str(e)}")
                    continue
            
            # Create mask for valid pixels
            if nodata_value is not None:
                valid_mask = out_image != nodata_value
            else:
                valid_mask = ~np.isnan(out_image) if out_image.dtype == np.float64 else np.ones_like(out_image, dtype=bool)
            
            if not np.any(valid_mask):
                self.logger.warning(f"No valid pixels found in GRU {gru_id}")
                continue
            
            # Find unique values within this GRU
            valid_values = out_image[valid_mask]
            
            if attribute_name in ['elevClass', 'radiationClass']:
                # For continuous data, classify into bands
                gru_hrus = self._create_hrus_from_bands(
                    out_image, valid_mask, out_transform, thresholds, 
                    attribute_name, gru_geometry, gru_row, hru_id_counter
                )
            else:
                # For discrete classes, use unique values
                unique_values = np.unique(valid_values)
                gru_hrus = self._create_hrus_from_classes(
                    out_image, valid_mask, out_transform, unique_values,
                    attribute_name, gru_geometry, gru_row, hru_id_counter
                )
            
            all_hrus.extend(gru_hrus)
            hru_id_counter += len(gru_hrus)
        
        self.logger.info(f"Created {len(all_hrus)} HRUs across all GRUs")
        return gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)

    def _create_hrus_from_bands(self, raster_data, valid_mask, transform, thresholds, 
                               attribute_name, gru_geometry, gru_row, start_hru_id):
        """Create HRUs from elevation/radiation bands within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id
        
        for i in range(len(thresholds) - 1):
            lower, upper = thresholds[i:i+2]
            
            # Make the last band inclusive of the upper bound
            if i == len(thresholds) - 2:  # Last band
                class_mask = valid_mask & (raster_data >= lower) & (raster_data <= upper)
            else:
                class_mask = valid_mask & (raster_data >= lower) & (raster_data < upper)
            
            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask, transform, raster_data, gru_geometry,
                    gru_row, current_hru_id, attribute_name, i + 1
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1
        
        return hrus

    def _create_hrus_from_classes(self, raster_data, valid_mask, transform, unique_values,
                                 attribute_name, gru_geometry, gru_row, start_hru_id):
        """Create HRUs from discrete classes within a single GRU."""
        hrus = []
        current_hru_id = start_hru_id
        
        for class_value in unique_values:
            class_mask = valid_mask & (raster_data == class_value)
            
            if np.any(class_mask):
                hru = self._create_hru_from_mask(
                    class_mask, transform, raster_data, gru_geometry,
                    gru_row, current_hru_id, attribute_name, class_value
                )
                if hru:
                    hrus.append(hru)
                    current_hru_id += 1
        
        return hrus

    def _create_hru_from_mask(self, class_mask, transform, raster_data, gru_geometry,
                             gru_row, hru_id, attribute_name, class_value):
        """Create a single HRU from a class mask within a GRU."""
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                class_mask.astype(np.uint8), 
                mask=class_mask, 
                transform=transform,
                connectivity=4
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                try:
                    geom = shape(shp)
                    if geom.is_valid and not geom.is_empty and geom.area > 0:
                        polygons.append(geom)
                except Exception:
                    continue
            
            if not polygons:
                return None
            
            # Create final geometry (naturally Polygon or MultiPolygon)
            if len(polygons) == 1:
                final_geometry = polygons[0]
            else:
                # This naturally creates a MultiPolygon if there are disconnected areas
                final_geometry = MultiPolygon(polygons)
            
            # Clean the geometry
            if not final_geometry.is_valid:
                final_geometry = final_geometry.buffer(0)
            
            if final_geometry.is_empty or not final_geometry.is_valid:
                return None
            
            # Ensure it's within the GRU boundary
            clipped_geometry = final_geometry.intersection(gru_geometry)
            
            if clipped_geometry.is_empty or not clipped_geometry.is_valid:
                return None
            
            # Calculate average attribute value
            avg_value = np.mean(raster_data[class_mask]) if np.any(class_mask) else class_value
            
            # Create HRU data
            hru_data = {
                'geometry': clipped_geometry,
                'GRU_ID': gru_row.get('GRU_ID', gru_row.name),
                'HRU_ID': hru_id,
                attribute_name: class_value,
                f'avg_{attribute_name.lower()}': avg_value,
                'hru_type': f'{attribute_name}_within_gru'
            }
            
            # Copy relevant GRU attributes (excluding geometry)
            for col in gru_row.index:
                if col not in ['geometry', 'GRU_ID'] and col not in hru_data:
                    hru_data[col] = gru_row[col]
            
            return hru_data
            
        except Exception as e:
            self.logger.warning(f"Error creating HRU for class {class_value} in GRU: {str(e)}")
            return None

    def _create_single_multipolygon_hru(self, class_mask, out_transform, domain_boundary, 
                                       class_value, out_image, attribute_name, gru_gdf):
        """
        Create a single MultiPolygon HRU from a class mask.
        
        Args:
            class_mask: Boolean mask for the class
            out_transform: Raster transform
            domain_boundary: Boundary of the domain
            class_value: Value of the class
            out_image: Original raster data
            attribute_name: Name of the attribute
            gru_gdf: Original GRU GeoDataFrame
            
        Returns:
            Dictionary representing the HRU
        """
        try:
            # Extract shapes from the mask
            shapes = list(rasterio.features.shapes(
                class_mask.astype(np.uint8), 
                mask=class_mask, 
                transform=out_transform,
                connectivity=8
            ))
            
            if not shapes:
                return None
            
            # Create polygons from shapes
            polygons = []
            for shp, _ in shapes:
                geom = shape(shp)
                if geom.is_valid and not geom.is_empty:
                    # Intersect with domain boundary to ensure it's within the domain
                    intersected = geom.intersection(domain_boundary)
                    if not intersected.is_empty:
                        if isinstance(intersected, (Polygon, MultiPolygon)):
                            polygons.append(intersected)
            
            if not polygons:
                return None
            
            # Create a single MultiPolygon from all polygons
            if len(polygons) == 1:
                multipolygon = polygons[0]
            else:
                multipolygon = MultiPolygon(polygons)
            
            # Clean the geometry
            multipolygon = multipolygon.buffer(0)  # Fix any topology issues
            
            if multipolygon.is_empty or not multipolygon.is_valid:
                return None
            
            # Calculate average attribute value
            avg_value = np.mean(out_image[class_mask])
            
            # Get a representative GRU for metadata (use the first one)
            representative_gru = gru_gdf.iloc[0]
            
            return {
                'geometry': multipolygon,
                'GRU_ID': 1,  # Single domain-wide unit
                attribute_name: class_value,
                f'avg_{attribute_name.lower()}': avg_value,
                'HRU_ID': class_value,  # Use class value as HRU ID
                'hru_type': f'{attribute_name}_multipolygon'
            }
            
        except Exception as e:
            self.logger.warning(f"Error creating MultiPolygon HRU for class {class_value}: {str(e)}")
            return None

    def _clean_and_prepare_hru_gdf(self, hru_gdf):
        """
        Clean and prepare the HRU GeoDataFrame for output.
        """
        # Ensure all geometries are valid
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(self._clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        
        # Final check: ensure only Polygon or MultiPolygon geometries
        valid_rows = []
        for idx, row in hru_gdf.iterrows():
            geom = row['geometry']
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid and not geom.is_empty:
                valid_rows.append(row)
            else:
                self.logger.warning(f"Removing HRU {idx} with invalid geometry type: {type(geom)}")
        
        if not valid_rows:
            self.logger.error("No valid HRUs after final geometry validation")
            return gpd.GeoDataFrame(columns=hru_gdf.columns, crs=hru_gdf.crs)
        
        hru_gdf = gpd.GeoDataFrame(valid_rows, crs=hru_gdf.crs)
        
        self.logger.info(f"Retained {len(hru_gdf)} HRUs after geometry validation")
        
        # Calculate areas and centroids
        self.logger.info("Calculating HRU areas and centroids")
        
        # Project to UTM for accurate area calculation
        utm_crs = hru_gdf.estimate_utm_crs()
        hru_gdf_utm = hru_gdf.to_crs(utm_crs)
        hru_gdf_utm['HRU_area'] = hru_gdf_utm.geometry.area
        
        # Calculate centroids (use representative point for MultiPolygons)
        centroids_utm = hru_gdf_utm.geometry.representative_point()
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
        
        hru_gdf_utm['center_lon'] = centroids_wgs84.x
        hru_gdf_utm['center_lat'] = centroids_wgs84.y
        
        # Convert back to original CRS
        hru_gdf = hru_gdf_utm.to_crs(hru_gdf.crs)
        
        # Calculate mean elevation for each HRU with proper CRS handling
        self.logger.info("Calculating mean elevation for each HRU")
        try:
            # Get CRS information
            with rasterio.open(self.dem_path) as src:
                dem_crs = src.crs
            
            shapefile_crs = hru_gdf.crs
            
            # Check if CRS match
            if dem_crs != shapefile_crs:
                self.logger.info(f"CRS mismatch detected. Reprojecting HRUs from {shapefile_crs} to {dem_crs}")
                hru_gdf_projected = hru_gdf.to_crs(dem_crs)
            else:
                hru_gdf_projected = hru_gdf.copy()
            
            # Use rasterstats with the raster file path directly (more efficient and handles CRS properly)
            zs = rasterstats.zonal_stats(
                hru_gdf_projected.geometry, 
                str(self.dem_path),  # Use file path instead of array
                stats=['mean'],
                nodata=-9999  # Explicit nodata value
            )
            hru_gdf['elev_mean'] = [item['mean'] if item['mean'] is not None else -9999 for item in zs]
            
        except Exception as e:
            self.logger.error(f"Error calculating mean elevation: {str(e)}")
            hru_gdf['elev_mean'] = -9999

        # Ensure HRU_ID is sequential if not already set properly
        if 'HRU_ID' in hru_gdf.columns:
            # Reset HRU_ID to be sequential
            hru_gdf = hru_gdf.sort_values(['GRU_ID', 'HRU_ID'])
            hru_gdf['HRU_ID'] = range(1, len(hru_gdf) + 1)
        else:
            hru_gdf['HRU_ID'] = range(1, len(hru_gdf) + 1)
        
        return hru_gdf

    def _clean_geometries(self, geometry):
        """Clean and validate geometries, ensuring only Polygon or MultiPolygon."""
        if geometry is None or geometry.is_empty:
            return None
        
        try:
            # Handle GeometryCollection - extract only Polygons
            from shapely.geometry import GeometryCollection
            if isinstance(geometry, GeometryCollection):
                polygons = []
                for geom in geometry.geoms:
                    if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                        polygons.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        for poly in geom.geoms:
                            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                                polygons.append(poly)
                
                if not polygons:
                    return None
                elif len(polygons) == 1:
                    geometry = polygons[0]
                else:
                    geometry = MultiPolygon(polygons)
            
            # Ensure we have a valid Polygon or MultiPolygon
            if not isinstance(geometry, (Polygon, MultiPolygon)):
                return None
            
            # Fix invalid geometries
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
                
                # Check again after buffer
                if not isinstance(geometry, (Polygon, MultiPolygon)):
                    return None
            
            return geometry if geometry.is_valid and not geometry.is_empty else None
            
        except Exception as e:
            self.logger.debug(f"Error cleaning geometry: {str(e)}")
            return None

    def _plot_hrus(self, hru_gdf, output_file, class_column, title):
        """Plot HRUs with appropriate coloring."""
        fig, ax = plt.subplots(figsize=(15, 15))

        try:
            if class_column == 'radiationClass':
                # Use the average radiation value for plotting
                if 'avg_radiationclass' in hru_gdf.columns:
                    hru_gdf.plot(column='avg_radiationclass', cmap='viridis', legend=True, ax=ax)
                else:
                    hru_gdf.plot(column=class_column, cmap='viridis', legend=True, ax=ax)
            elif 'combined_' in class_column:
                # For combined attributes, use qualitative colormap
                hru_gdf.plot(column=class_column, cmap='tab20', legend=False, ax=ax)
            else:
                # Use a qualitative colormap for other class types
                hru_gdf.plot(column=class_column, cmap='tab20', legend=True, ax=ax)
                
        except Exception as e:
            self.logger.warning(f"Error plotting with column {class_column}: {str(e)}")
            # Fallback: plot without legend
            hru_gdf.plot(ax=ax, alpha=0.7)
        
        ax.set_title(title)
        plt.axis('off')
        plt.tight_layout()
        
        # Create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"HRU plot saved to {output_file}")

    def _read_shapefile(self, shapefile_path):
        """
        Read a shapefile and return it as a GeoDataFrame.

        Args:
            shapefile_path (str or Path): Path to the shapefile.

        Returns:
            gpd.GeoDataFrame: The shapefile content as a GeoDataFrame.
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs is None:
                self.logger.warning(f"CRS is not defined for {shapefile_path}. Setting to EPSG:4326.")
                gdf = gdf.set_crs("EPSG:4326")
            return gdf
        except Exception as e:
            self.logger.error(f"Error reading shapefile {shapefile_path}: {str(e)}")
            raise

    def _get_file_path(self, file_type, file_def_path, file_name):
        """
        Construct file paths based on configuration.

        Args:
            file_type (str): Type of the file (used as a key in config).
            file_def_path (str): Default path relative to project directory.
            file_name (str): Name of the file.

        Returns:
            Path: Constructed file path.
        """
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))