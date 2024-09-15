import geopandas as gpd # type: ignore
import numpy as np # type: ignore
from typing import List, Dict, Any, Optional
import rasterio # type: ignore
from rasterio.mask import mask # type: ignore
from shapely.geometry import Polygon, MultiPolygon, LineString, shape # type: ignore
from shapely.ops import unary_union # type: ignore
import matplotlib.pyplot as plt # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import pvlib # type: ignore
import pandas as pd # type: ignore
from pyproj import CRS # type: ignore
import rasterstats # type: ignore

class DomainDiscretizer:
    """
    A class for discretizing a domain into Hydrologic Response Units (HRUs).

    This class provides methods for various types of domain discretization,
    including elevation-based, soil class-based, land class-based, and
    radiation-based discretization.

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
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        self.dem_path = self._get_file_path("DEM_PATH", "attributes/elevation/dem", "elevation.tif")

        delineation_method = self.config.get('DOMAIN_DEFINITION_METHOD')

        if delineation_method == 'delineate':
            self.delineation_suffix = 'delineate'
        elif delineation_method == 'lumped':
            self.delineation_suffix = 'lumped'
        elif delineation_method == 'subset':
            self.delineation_suffix = f'subset_{self.config.get('GEOFABRIC_TYPE')}'
        
    def sort_catchment_shape(self):
        self.logger.info("Sorting catchment shape")
        
        self.catchment_path = self.config.get('CATCHMENT_PATH')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        
        if self.catchment_path == 'default':
            self.catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            self.catchment_path = Path(self.catchment_path)
        
        gru_id = self.config.get('CATCHMENT_SHP_GRUID')
        hru_id = self.config.get('CATCHMENT_SHP_HRUID')
        
        # Open the shape
        shp = gpd.read_file(self.catchment_path / self.catchment_name)
        
        # Sort
        shp = shp.sort_values(by=[gru_id, hru_id])
        
        # Save
        shp.to_file(self.catchment_path / self.catchment_name)
        
        self.logger.info(f"Catchment shape sorted and saved to {self.catchment_path / self.catchment_name}")

    def discretize_domain(self) -> Optional[Path]:
        """
        Discretize the domain based on the method specified in the configuration.

        Returns:
            Optional[Path]: Path to the output shapefile, or None if discretization fails.
        """
        discretization_method = self.config.get('DOMAIN_DISCRETIZATION').lower()
        self.logger.info(f"Starting domain discretization using method: {discretization_method}")

        method_map = {
            'grus': self._use_grus_as_hrus,
            'elevation': self._discretize_by_elevation,
            'soilclass': self._discretize_by_soil_class,
            'landclass': self._discretize_by_land_class,
            'radiation': self._discretize_by_radiation,
            'combined': self._discretize_combined
        }

        if discretization_method not in method_map:
            self.logger.error(f"Invalid discretization method: {discretization_method}")
            raise ValueError(f"Invalid discretization method: {discretization_method}")

        method_map[discretization_method]()
        self.sort_catchment_shape()
        

    def _use_grus_as_hrus(self):
        """
        Use Grouped Response Units (GRUs) as Hydrologic Response Units (HRUs) without further discretization.

        Returns:
            Path: Path to the output HRU shapefile.
        """
        self.logger.info(f"config domain name {self.config.get('DOMAIN_NAME')}")
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        hru_output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_GRUs.shp")

        gru_gdf = self._read_shapefile(gru_shapefile)
        gru_gdf['HRU_ID'] = range(1, len(gru_gdf) + 1)
        gru_gdf = gru_gdf.to_crs('epsg:3763')
        gru_gdf['HRU_area'] = gru_gdf.geometry.area 
        gru_gdf = gru_gdf.to_crs('epsg:4326')
        gru_gdf['hru_type'] = 'GRU'

        # Calculate mean elevation for each HRU
        self.logger.info("Calculating mean elevation for each HRU")
        with rasterio.open(self.dem_path) as src:
            dem_data = src.read(1)
            dem_affine = src.transform

        # Use rasterstats to calculate zonal statistics
        zs = rasterstats.zonal_stats(gru_gdf.geometry, dem_data, affine=dem_affine, stats=['mean'])
        gru_gdf['elev_mean'] = [item['mean'] for item in zs]
        
        centroids_utm = gru_gdf.geometry.centroid
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
        
        gru_gdf['center_lon'] = centroids_wgs84.x
        gru_gdf['center_lat'] = centroids_wgs84.y
        
        if 'COMID' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['COMID']
        elif 'fid' in gru_gdf.columns:
            gru_gdf['GRU_ID'] = gru_gdf['fid']

        gru_gdf['HRU_area'] = gru_gdf['GRU_area']
        #gru_gdf['HRU_ID'] = gru_gdf['GRU_ID']        

        gru_gdf.to_file(hru_output_shapefile)
        self.logger.info(f"GRUs saved as HRUs to {hru_output_shapefile}")

        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_as_GRUs.png")
        self._plot_hrus(gru_gdf, output_plot, 'HRU_ID', 'GRUs = HRUs')
        return hru_output_shapefile

    def _discretize_by_elevation(self):
        """
        Discretize the domain based on elevation.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """

        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))

        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", "elevation.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_elevation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_elevation.png")

        elevation_band_size = float(self.config.get('ELEVATION_BAND_SIZE'))
        min_hru_size = float(self.config.get('MIN_HRU_SIZE'))
        gru_gdf, elevation_thresholds = self._read_and_prepare_data(gru_shapefile, dem_raster, elevation_band_size)
        hru_gdf = self._process_hrus(gru_gdf, dem_raster, elevation_thresholds, 'elevClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._merge_small_hrus(hru_gdf, min_hru_size, 'elevClass')
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
        Discretize the domain based on soil classifications.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        soil_raster = self._get_file_path("SOIL_CLASS_PATH", "attributes/soil_class/", "soil_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_soilclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_soilclass.png")

        min_hru_size = float(self.config.get('MIN_HRU_SIZE'))

        gru_gdf, soil_classes = self._read_and_prepare_data(gru_shapefile, soil_raster)
        hru_gdf = self._process_hrus(gru_gdf, soil_raster, soil_classes, 'soilClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._merge_small_hrus(hru_gdf, min_hru_size, 'soilClass')
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
        Discretize the domain based on land cover classifications.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self.config.get('RIVER_BASINS_NAME')
        if gru_shapefile == 'default':
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        else:
            gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", self.config.get('RIVER_BASINS_NAME'))

        land_raster = self._get_file_path("LAND_CLASS_PATH","attributes/land_class", "land_classes.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_landclass.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_landclass.png")

        min_hru_size = float(self.config.get('MIN_HRU_SIZE'))

        gru_gdf, land_classes = self._read_and_prepare_data(gru_shapefile, land_raster)
        hru_gdf = self._process_hrus(gru_gdf, land_raster, land_classes, 'landClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._merge_small_hrus(hru_gdf, min_hru_size, 'landClass')
            hru_gdf = self._clean_and_prepare_hru_gdf(hru_gdf)
            
            hru_gdf.to_file(output_shapefile)
            self.logger.info(f"Land-based HRU Shapefile created with {len(hru_gdf)} HRUs and saved to {output_shapefile}")

            self._plot_hrus(hru_gdf, output_plot, 'landClass', 'Land-based HRUs')
            return output_shapefile
        else:
            self.logger.error("No valid HRUs were created. Check your input data and parameters.")
            return None

    def _discretize_by_radiation(self):
        """
        Discretize the domain based on radiation properties.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        gru_shapefile = self._get_file_path("RIVER_BASINS_PATH", "shapefiles/river_basins", f"{self.domain_name}_riverBasins_{self.delineation_suffix}.shp")
        dem_raster = self._get_file_path("DEM_PATH", "attributes/elevation/dem", "elevation.tif")
        radiation_raster = self._get_file_path("RADIATION_PATH", "attributes/radiation", "annual_radiation.tif")
        output_shapefile = self._get_file_path("CATCHMENT_PATH", "shapefiles/catchment", f"{self.domain_name}_HRUs_radiation.shp")
        output_plot = self._get_file_path("CATCHMENT_PLOT_DIR", "plots/catchment", f"{self.domain_name}_HRUs_radiation.png")

        min_hru_size = float(self.config.get('MIN_HRU_SIZE'))
        radiation_class_number = int(self.config.get('RADIATION_CLASS_NUMBER'))

        if not radiation_raster.exists():
            self.logger.info("Annual radiation raster not found. Calculating radiation...")
            radiation_raster = self._calculate_annual_radiation(dem_raster, radiation_raster)
            if radiation_raster is None:
                raise ValueError("Failed to calculate annual radiation")

        gru_gdf, radiation_thresholds = self._read_and_prepare_data(gru_shapefile, radiation_raster, radiation_class_number)
        hru_gdf = self._process_hrus(gru_gdf, radiation_raster, radiation_thresholds, 'radiationClass')

        if hru_gdf is not None and not hru_gdf.empty:
            hru_gdf = self._merge_small_hrus(hru_gdf, min_hru_size, 'radiationClass')
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

    def _discretize_combined(self):
        """
        Discretize the domain based on a combination of geospatial attributes.

        Returns:
            Optional[Path]: Path to the output HRU shapefile, or None if discretization fails.
        """
        # Implementation for combined discretization
        pass

    def _read_and_prepare_data(self, shapefile_path, raster_path, band_size=None):
        """
        Read and prepare data for discretization.

        Args:
            shapefile_path (Path): Path to the GRU shapefile.
            raster_path (Path): Path to the raster file.
            band_size (Optional[float]): Size of bands for discretization.

        Returns:
            Tuple[gpd.GeoDataFrame, np.ndarray]: Prepared GeoDataFrame and thresholds.
        """
        gru_gdf = self._read_shapefile(shapefile_path)
        if 'gruId' not in gru_gdf.columns:
            gru_gdf['gruId'] = gru_gdf.index.astype(str)

        gru_gdf['geometry'] = gru_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, gru_gdf.geometry, crop=False)
            out_image = out_image[0]  # Get the first band
            valid_data = out_image[out_image != src.nodata]
            
            if band_size:
                min_value = np.floor(np.min(valid_data))
                max_value = np.ceil(np.max(valid_data))
                thresholds = np.arange(min_value, max_value + band_size, band_size)
            else:
                thresholds = np.unique(valid_data)
        
        return gru_gdf, thresholds

    def _process_hrus(self, gru_gdf, raster_path, thresholds, attribute_name):
        """
        Process HRUs based on the given raster and thresholds.

        Args:
            gru_gdf (gpd.GeoDataFrame): GeoDataFrame of GRUs.
            raster_path (Path): Path to the raster file.
            thresholds (np.ndarray): Thresholds for discretization.
            attribute_name (str): Name of the attribute for classification.

        Returns:
            gpd.GeoDataFrame: Processed HRU GeoDataFrame.
        """
        all_hrus = []
        num_cores = max(1, multiprocessing.cpu_count() // 2)
        
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            future_to_row = {executor.submit(self._create_hrus, row, raster_path, thresholds, attribute_name): row for _, row in gru_gdf.iterrows()}
            for future in as_completed(future_to_row):
                all_hrus.extend(future.result())

        hru_gdf = gpd.GeoDataFrame(all_hrus, crs=gru_gdf.crs)
        return self._postprocess_hrus(hru_gdf)

    def _create_hrus(self, row, raster_path: Path, thresholds: np.ndarray, attribute_name: str) -> List[Dict[str, Any]]:
        """
        Create HRUs for a single GRU based on the given raster and thresholds.

        Args:
            row: A single row from the GRU GeoDataFrame.
            raster_path (Path): Path to the raster file.
            thresholds (np.ndarray): Thresholds for discretization.
            attribute_name (str): Name of the attribute for classification.

        Returns:
            List[Dict[str, Any]]: List of HRU dictionaries.
        """
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [row.geometry], crop=False, all_touched=False)
            out_image = out_image[0]
            
            gru_attributes = row.drop('geometry').to_dict()
            
            hrus = []
            for i in range(len(thresholds) - 1):
                lower, upper = thresholds[i:i+2]
                class_mask = (out_image >= lower) & (out_image < upper)
                if np.any(class_mask):
                    shapes = rasterio.features.shapes(class_mask.astype(np.uint8), mask=class_mask, transform=out_transform)
                    class_polys = [shape(shp) for shp, _ in shapes]
                    
                    if class_polys:
                        merged_poly = unary_union(class_polys).intersection(row.geometry)
                        if not merged_poly.is_empty:
                            if isinstance(merged_poly, (Polygon, MultiPolygon)):
                                geoms = [merged_poly] if isinstance(merged_poly, Polygon) else merged_poly.geoms
                            elif isinstance(merged_poly, LineString):
                                # Skip LineString geometries
                                self.logger.warning(f"Skipping LineString geometry in GRU {row['gruId']}")
                                continue
                            else:
                                # For other geometry types, try to buffer to create a polygon
                                buffered = merged_poly.buffer(0.0000001)  # Small buffer
                                if isinstance(buffered, (Polygon, MultiPolygon)):
                                    geoms = [buffered] if isinstance(buffered, Polygon) else buffered.geoms
                                else:
                                    self.logger.warning(f"Skipping invalid geometry in GRU {row['gruId']}")
                                    continue

                            for geom in geoms:
                                if isinstance(geom, Polygon):
                                    hrus.append({
                                        'geometry': geom,
                                        'gruNo': row.name,
                                        'gruId': row['gruId'],
                                        attribute_name: i + 1,
                                        f'avg_{attribute_name.lower()}': np.mean(out_image[class_mask]),
                                        **gru_attributes
                                    })
            
            if not hrus:
                # If no valid HRUs were created, use the entire GRU as a single HRU
                self.logger.warning(f"No valid HRUs created for GRU {row['gruId']}. Using entire GRU as single HRU.")
                hrus.append({
                    'geometry': row.geometry,
                    'gruNo': row.name,
                    'gruId': row['gruId'],
                    attribute_name: -9999,  # No-data value
                    f'avg_{attribute_name.lower()}': np.mean(out_image),
                    **gru_attributes
                })
            
            return hrus

    def _merge_small_hrus(self, hru_gdf, min_hru_size, class_column):
        self.logger.info(f"Merging small HRUs (minimum size: {min_hru_size} km²)")
        hru_gdf.set_crs(epsg=4326, inplace=True)
        utm_crs = hru_gdf.estimate_utm_crs()
        hru_gdf_utm = hru_gdf.to_crs(utm_crs)
        
        hru_gdf_utm['geometry'] = hru_gdf_utm['geometry'].apply(self._clean_geometries)
        hru_gdf_utm = hru_gdf_utm[hru_gdf_utm['geometry'].notnull()]
        
        original_boundary = unary_union(hru_gdf_utm.geometry)
        
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000  # Convert to km²
        hru_gdf_utm = hru_gdf_utm.sort_values('area')
        
        merged_count = 0
        while True:
            small_hrus = hru_gdf_utm[hru_gdf_utm['area'] < min_hru_size]
            if len(small_hrus) == 0:
                break
            
            progress = False
            for idx, small_hru in small_hrus.iterrows():
                try:
                    small_hru_geom = self._clean_geometries(small_hru.geometry)
                    if small_hru_geom is None:
                        hru_gdf_utm = hru_gdf_utm.drop(idx)
                        continue

                    neighbors = self._find_neighbors(small_hru_geom, hru_gdf_utm, idx)
                    if len(neighbors) > 0:
                        largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                        largest_neighbor_geom = self._clean_geometries(largest_neighbor.geometry)
                        if largest_neighbor_geom is None:
                            continue

                        merged_geometry = unary_union([small_hru_geom, largest_neighbor_geom])
                        merged_geometry = self._simplify_geometry(merged_geometry)
                        
                        if merged_geometry and merged_geometry.is_valid:
                            hru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                            hru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                            hru_gdf_utm = hru_gdf_utm.drop(idx)
                            merged_count += 1
                            progress = True
                    else:
                        distances = hru_gdf_utm.geometry.distance(small_hru_geom)
                        nearest_hru = hru_gdf_utm.loc[distances.idxmin()]
                        if nearest_hru.name != idx:
                            nearest_hru_geom = self._clean_geometries(nearest_hru.geometry)
                            if nearest_hru_geom is None:
                                continue
                            
                            merged_geometry = unary_union([small_hru_geom, nearest_hru_geom])
                            merged_geometry = self._simplify_geometry(merged_geometry)
                            
                            if merged_geometry and merged_geometry.is_valid:
                                hru_gdf_utm.at[nearest_hru.name, 'geometry'] = merged_geometry
                                hru_gdf_utm.at[nearest_hru.name, 'area'] = merged_geometry.area / 1_000_000
                                hru_gdf_utm = hru_gdf_utm.drop(idx)
                                merged_count += 1
                                progress = True
                except Exception as e:
                    self.logger.error(f"Error merging HRU {idx}: {str(e)}")
            
            if not progress:
                break
            
            hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
            hru_gdf_utm = hru_gdf_utm.sort_values('area')
        
        # Ensure complete coverage
        current_coverage = unary_union(hru_gdf_utm.geometry)
        gaps = original_boundary.difference(current_coverage)
        if not gaps.is_empty:
            if gaps.geom_type == 'MultiPolygon':
                gap_geoms = list(gaps.geoms)
            else:
                gap_geoms = [gaps]
            
            for gap in gap_geoms:
                if gap.area > 0:
                    nearest_hru = hru_gdf_utm.geometry.distance(gap.centroid).idxmin()
                    merged_geom = self._clean_geometries(unary_union([hru_gdf_utm.at[nearest_hru, 'geometry'], gap]))
                    if merged_geom and merged_geom.is_valid:
                        hru_gdf_utm.at[nearest_hru, 'geometry'] = merged_geom
                        hru_gdf_utm.at[nearest_hru, 'area'] = merged_geom.area / 1_000_000
        
        hru_gdf_utm = hru_gdf_utm.reset_index(drop=True)
        hru_gdf_utm['HRU_ID'] = range(1, len(hru_gdf_utm) + 1)
        hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'].astype(str) + '_' + hru_gdf_utm[class_column].astype(str)
        hru_gdf_utm['area'] = hru_gdf_utm.geometry.area / 1_000_000
        
        hru_gdf_merged = hru_gdf_utm.to_crs(hru_gdf.crs)
        
        self.logger.info(f"Merged {merged_count} small HRUs. Final HRU count: {len(hru_gdf_merged)}")
        return hru_gdf_merged

    def _find_neighbors(self, geometry, gdf, current_index, buffer_distance=1e-6):
        try:
            if geometry is None or not geometry.is_valid:
                return gpd.GeoDataFrame()
            
            simplified_geom = self._simplify_geometry(geometry)
            buffered = simplified_geom.buffer(buffer_distance)
            
            possible_matches_index = list(gdf.sindex.intersection(buffered.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.geometry.is_valid & possible_matches.geometry.intersects(buffered)]
            
            return precise_matches[precise_matches.index != current_index]
        except Exception as e:
            self.logger.error(f"Error finding neighbors for HRU {current_index}: {str(e)}")
            return gpd.GeoDataFrame()

    def _simplify_geometry(self, geom, tolerance=1e-8):
        if geom is None or geom.is_empty:
            return None
        try:
            if geom.geom_type == 'MultiPolygon':
                parts = [part for part in geom.geoms if part.is_valid and not part.is_empty]
                if len(parts) == 0:
                    return None
                return MultiPolygon(parts).simplify(tolerance, preserve_topology=True)
            return geom.simplify(tolerance, preserve_topology=True)
        except Exception:
            return None

    def _clean_and_prepare_hru_gdf(self, hru_gdf):
        # Ensure all geometries are valid polygons
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(self._clean_geometries)
        hru_gdf = hru_gdf[hru_gdf['geometry'].notnull()]
        hru_gdf['HRU_ID'] = range(1, len(hru_gdf) + 1)

        # Calculate mean elevation for each HRU
        self.logger.info("Calculating mean elevation for each HRU")
        try:
            with rasterio.open(self.dem_path) as src:
                dem_data = src.read(1)
                dem_affine = src.transform

            # Use rasterstats to calculate zonal statistics
            zs = rasterstats.zonal_stats(hru_gdf.geometry, dem_data, affine=dem_affine, stats=['mean'])
            hru_gdf['elev_mean'] = [item['mean'] for item in zs]

            self.logger.info("Mean elevation calculation completed")
        except Exception as e:
            self.logger.error(f"Error calculating mean elevation: {str(e)}")
            # If elevation calculation fails, set a default value
            hru_gdf['elev_mean'] = -9999

        # Remove any columns that might cause issues with shapefiles
        columns_to_keep = ['GRU_ID', 'HRU_ID', 'geometry', 'HRU_area', 'center_lon', 'center_lat', 'elev_mean']
        class_columns = [col for col in hru_gdf.columns if col.endswith('Class')]
        columns_to_keep.extend(class_columns)
        hru_gdf = hru_gdf[columns_to_keep]

        # Final check for valid polygons
        valid_polygons = []
        for idx, row in hru_gdf.iterrows():
            geom = row['geometry']
            if isinstance(geom, (Polygon, MultiPolygon)) and geom.is_valid:
                valid_polygons.append(row)
            else:
                self.logger.warning(f"Removing invalid geometry for HRU {idx}")
        
        hru_gdf = gpd.GeoDataFrame(valid_polygons, crs=hru_gdf.crs)

        # Convert MultiPolygons to Polygons where possible
        hru_gdf['geometry'] = hru_gdf['geometry'].apply(self._to_polygon)

        return hru_gdf

    def _clean_geometries(self, geometry):
        if geometry is None or geometry.is_empty:
            return None
        try:
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.geom_type == 'MultiPolygon':
                geometry = geometry.buffer(0).buffer(0)
            return geometry if geometry.is_valid and not geometry.is_empty else None
        except Exception:
            return None

    def _postprocess_hrus(self, hru_gdf):
        """
        Perform final processing on the HRU GeoDataFrame.

        Args:
            hru_gdf (gpd.GeoDataFrame): The HRU GeoDataFrame to process.

        Returns:
            gpd.GeoDataFrame: The processed HRU GeoDataFrame.
        """
        # Project to UTM for area calculation
        utm_crs = hru_gdf.estimate_utm_crs()
        hru_gdf_utm = hru_gdf.to_crs(utm_crs)
        hru_gdf_utm['HRU_area'] = hru_gdf_utm.geometry.area 
        hru_gdf_utm['HRU_area'] = hru_gdf_utm['HRU_area'].astype('float64')

        if 'avg_elevation' in hru_gdf_utm.columns:
            hru_gdf_utm['avg_elevation'] = hru_gdf_utm['avg_elevation'].astype('float64')

        # Determine the class column name
        #class_column = next((col for col in ['elevClass', 'soilClass', 'landClass', 'radiationClass'] if col in hru_gdf_utm.columns), None)
        #if class_column:
        #    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'].astype(str) + '_' + hru_gdf_utm[class_column].astype(str)
        #else:
        #    hru_gdf_utm['hruId'] = hru_gdf_utm['gruId'].astype(str) + '_' + hru_gdf_utm['hruNo'].astype(str)
        
        centroids_utm = hru_gdf_utm.geometry.centroid
        centroids_wgs84 = centroids_utm.to_crs(CRS.from_epsg(4326))
        
        hru_gdf_utm['center_lon'] = centroids_wgs84.x
        hru_gdf_utm['center_lat'] = centroids_wgs84.y
        
        # Convert back to the original CRS
        hru_gdf = hru_gdf_utm.to_crs(hru_gdf.crs)
        
        return hru_gdf

    def _to_polygon(self, geom):
        if isinstance(geom, MultiPolygon):
            if len(geom.geoms) == 1:
                return geom.geoms[0]
            else:
                return geom.buffer(0)
        return geom

    def _plot_hrus(self, hru_gdf, output_file, class_column, title):
        fig, ax = plt.subplots(figsize=(15, 15))

        if class_column == 'radiationClass':
            # Use a sequential colormap for radiation
            hru_gdf.plot(column='avg_radiation', cmap='viridis', legend=True, ax=ax)
        else:
            # Use a qualitative colormap for other class types
            hru_gdf.plot(column=class_column, cmap='tab20', legend=True, ax=ax)
        
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