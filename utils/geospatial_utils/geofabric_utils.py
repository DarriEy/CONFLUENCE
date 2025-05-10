"""
geofabric_utils.py

This module provides utilities for geofabric delineation and processing in the CONFLUENCE system.
It includes classes for geofabric delineation, subsetting, and lumped watershed delineation.

Classes:
    - GeofabricDelineator: Handles the delineation of geofabrics using TauDEM.
    - GeofabricSubsetter: Subsets geofabric data based on pour points and upstream basins.
    - LumpedWatershedDelineator: Delineates lumped watersheds using TauDEM.

Each class provides methods for processing geospatial data, running external commands,
and managing file operations related to geofabric analysis.
"""
import os
import geopandas as gpd # type: ignore
import networkx as nx # type: ignore
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import subprocess
from osgeo import gdal, ogr # type: ignore
import shutil
from functools import wraps
import sys
import glob
#from pysheds.grid import Grid # type: ignore
import rasterio # type: ignore
import numpy as np # type: ignore
from shapely.geometry import Polygon # type: ignore
import multiprocessing  
from shapely.ops import unary_union # type: ignore
import time
import shapely # type: ignore
from scipy import ndimage # type: ignore
import pandas as pd # type: ignore

sys.path.append(str(Path(__file__).resolve().parent))
from utils.configHandling_utils.logging_utils import setup_logger, get_function_logger # type: ignore

class GeofabricDelineator:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.mpi_processes = self.config.get('MPI_PROCESSES', multiprocessing.cpu_count())
        self.interim_dir = self.project_dir / "taudem-interim-files" / "d8"
        self.dem_path = self._get_dem_path()
        
        # Set up TauDEM directory (don't add to PATH)
        taudem_dir = self.config.get('TAUDEM_DIR')
        if taudem_dir == 'default':
            taudem_dir = str(self.data_dir / 'installs' / 'TauDEM' / 'bin')
        self.taudem_dir = Path(taudem_dir) if taudem_dir else None
        
        self.max_retries = self.config.get('MAX_RETRIES', 3)
        self.retry_delay = self.config.get('RETRY_DELAY', 5)
        self.min_gru_size = self.config.get('MIN_GRU_SIZE', 5.0)  # Default 1 km²

    def _get_dem_path(self) -> Path:
        dem_path = self.config.get('DEM_PATH')
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        if dem_path == 'default':
            return self.project_dir / 'attributes' / 'elevation' / 'dem' / dem_name
        return Path(dem_path)

    def _set_taudem_path(self):
        if taudem_dir == 'default':
            taudem_dir = str(self.data_dir / 'installs' / 'TauDEM' / 'bin')
        os.environ['PATH'] = f"{os.environ['PATH']}:{taudem_dir}"

    def run_command(self, command: str):
        """
        Run a shell command and log any errors.
        
        Args:
            command (str): The command to run.
            
        Raises:
            subprocess.CalledProcessError: If the command fails.
        """
        try:
            # Set up environment with GDAL library path
            env = os.environ.copy()
            
            # Check if GDAL is already loaded in current environment
            gdal_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
            
            # If GDAL library path is specified in config, use it
            if self.config.get('GDAL_LIB_PATH'):
                gdal_path = self.config.get('GDAL_LIB_PATH')
                env['LD_LIBRARY_PATH'] = f"{gdal_path}:{gdal_lib_path}"
            
            result = subprocess.run(command, check=True, shell=True, env=env, 
                                capture_output=True, text=True)
            self.logger.info(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing command: {command}")
            self.logger.error(f"Error details: {e.stderr}")
            raise

    @get_function_logger
    def delineate_geofabric(self) -> Tuple[Optional[Path], Optional[Path]]:
        try:
            self.logger.info(f"Starting geofabric delineation for {self.domain_name}")
            self._validate_inputs()
            self.create_directories()
            self.pour_point_path = self._get_pour_point_path()
            self.run_taudem_steps(self.dem_path, self.pour_point_path)
            self.run_gdal_processing()
            river_network_path, river_basins_path = self.subset_upstream_geofabric()

            self.cleanup()

            self.logger.info(f"Geofabric delineation completed for {self.domain_name}")
            return river_network_path, river_basins_path
        except Exception as e:
            self.logger.error(f"Error in geofabric delineation: {str(e)}")
            self.cleanup()
            raise

    def _validate_inputs(self):
        if not self.dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
        # Add more input validations as needed

    def create_directories(self):
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created interim directory: {self.interim_dir}")

    def _get_pour_point_path(self) -> Path:
        pour_point_path = self.config.get('POUR_POINT_SHP_PATH')
        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(pour_point_path)
        
        if self.config['POUR_POINT_SHP_NAME'] == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"
        
        if not pour_point_path.exists():
            raise FileNotFoundError(f"Pour point file not found: {pour_point_path}")
        
        return pour_point_path

    def run_command(self, command: str, retry: bool = True) -> None:
        def get_run_command():
            if shutil.which("srun"):
                return "srun"
            elif shutil.which("mpirun"):
                return "mpirun"
            else:
                return None

        run_cmd = get_run_command()

        for attempt in range(self.max_retries if retry else 1):
            try:
                if run_cmd:
                    full_command = f"{run_cmd} {command}"
                else:
                    full_command = command

                self.logger.info(f"Running command: {full_command}")
                result = subprocess.run(full_command, check=True, shell=True, capture_output=True, text=True)
                self.logger.info(f"Command output: {result.stdout}")
                return
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error executing command: {full_command}")
                self.logger.error(f"Error details: {e.stderr}")
                if attempt < self.max_retries - 1 and retry:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                elif run_cmd:
                    self.logger.info(f"Trying without {run_cmd}...")
                    run_cmd = None  # Try without srun/mpirun on the next attempt
                else:
                    raise

    def run_taudem_steps(self, dem_path: Path, pour_point_path: Path):
        threshold = self.config.get('STREAM_THRESHOLD')
        max_distance = self.config.get('MOVE_OUTLETS_MAX_DISTANCE', 200)

        # Check if TauDEM directory exists
        if not self.taudem_dir or not self.taudem_dir.exists():
            self.logger.error(f"TauDEM directory not found: {self.taudem_dir}")
            raise RuntimeError("TauDEM directory not available")

        # Always load GDAL module before TauDEM commands
        module_prefix = "module load gdal/3.9.2 && "
        self.logger.info("Loading GDAL module: gdal/3.9.2")

        steps = [
            f"{module_prefix}{self.taudem_dir}/pitremove -z {dem_path} -fel {self.interim_dir}/elv-fel.tif -v",
            f"{module_prefix}{self.taudem_dir}/d8flowdir -fel {self.interim_dir}/elv-fel.tif -sd8 {self.interim_dir}/elv-sd8.tif -p {self.interim_dir}/elv-fdir.tif",
            f"{module_prefix}{self.taudem_dir}/aread8 -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -nc",
            f"{module_prefix}{self.taudem_dir}/gridnet -p {self.interim_dir}/elv-fdir.tif -plen {self.interim_dir}/elv-plen.tif -tlen {self.interim_dir}/elv-tlen.tif -gord {self.interim_dir}/elv-gord.tif",
            f"{module_prefix}{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -thresh {threshold}",
            f"{module_prefix}{self.taudem_dir}/moveoutletstostrm -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",
            f"{module_prefix}{self.taudem_dir}/streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif"
        ]

        for step in steps:
            self.run_command(f"-n {self.mpi_processes} {step}")
            self.logger.info(f"Completed TauDEM step: {step}")

    def _clean_geometries(self, geometry):

        """Clean and validate geometry."""
        if geometry is None or not geometry.is_valid:
            return None
        try:
            return geometry.buffer(0)
        except:
            return None

    def _simplify_geometry(self, geometry, tolerance=1):
        """Simplify geometry while preserving topology."""
        try:
            return geometry.simplify(tolerance, preserve_topology=True)
        except:
            return geometry
            
    @get_function_logger
    def delineate_coastal(self, work_log_dir=None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate coastal watersheds that drain directly to the ocean.
        
        This method:
        1. Creates a land mask from the DEM
        2. Finds the difference between the land mask and existing watersheds
        3. Divides the coastal strip into individual watersheds by extending the boundaries
        of adjacent inland watersheds
        
        Args:
            work_log_dir (Path, optional): Directory for logging. Defaults to None.
            
        Returns:
            Tuple[Optional[Path], Optional[Path]]: Paths to the updated river_network and river_basins shapefiles.
        """
        try:
            self.logger.info(f"Starting coastal watershed delineation for {self.domain_name}")
            
            # Get paths to existing delineated river basins and network
            river_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_delineate.shp"
            river_network_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_delineate.shp"
            
            if not river_basins_path.exists() or not river_network_path.exists():
                self.logger.error("River basins or network files not found. Run delineate_geofabric first.")
                return None, None
            
            # Load existing delineation
            river_basins = self.load_geopandas(river_basins_path)
            river_network = self.load_geopandas(river_network_path)
            
            # Create interim directory for coastal delineation
            coastal_interim_dir = self.project_dir / "taudem-interim-files" / "coastal"
            coastal_interim_dir.mkdir(parents=True, exist_ok=True)
            
            # ---------- STEP 1: IDENTIFY COASTAL AREAS ---------- #
            
            # Create a land polygon from the DEM (areas with elevation > 0)
            land_polygon = self._create_land_polygon_from_dem()
            if land_polygon is None or land_polygon.empty:
                self.logger.error("Failed to create land polygon from DEM.")
                return river_network_path, river_basins_path
                
            # Create a single polygon from all existing watersheds
            try:
                watersheds_polygon = gpd.GeoDataFrame(
                    geometry=[river_basins.unary_union], 
                    crs=river_basins.crs
                )
            except Exception as e:
                self.logger.error(f"Error creating watersheds polygon: {str(e)}")
                return river_network_path, river_basins_path
                
            # Find land areas not covered by existing watersheds
            try:
                # Make sure CRS matches
                if land_polygon.crs != watersheds_polygon.crs:
                    land_polygon = land_polygon.to_crs(watersheds_polygon.crs)
                    
                # Find the difference between land and watersheds
                coastal_strip = gpd.overlay(land_polygon, watersheds_polygon, how='difference')
                
                if coastal_strip.empty:
                    self.logger.info("No coastal areas found outside existing watersheds.")
                    return river_network_path, river_basins_path
                    
            except Exception as e:
                self.logger.error(f"Error finding coastal strip: {str(e)}")
                return river_network_path, river_basins_path
                
            # ---------- STEP 2: DIVIDE COASTAL STRIP INTO INDIVIDUAL WATERSHEDS ---------- #
            
            # First, create Voronoi polygons for each river basin
            try:
                # Get centroids of each river basin
                river_basins_centroids = river_basins.copy()
                river_basins_centroids.geometry = river_basins.geometry.centroid
                
                # Buffer centroids to avoid potential issues with geopandas Voronoi
                river_basins_centroids.geometry = river_basins_centroids.geometry.buffer(0.000001)
                
                # Create Voronoi polygons
                voronoi_gdf = self._create_voronoi_tessellation(river_basins_centroids)
                
                if voronoi_gdf is None or voronoi_gdf.empty:
                    self.logger.warning("Failed to create Voronoi tessellation. Using alternative approach.")
                    # Use watershed boundaries to create extended lines to the coast
                    coastal_watersheds = self._divide_coastal_strip_by_extending_boundaries(coastal_strip, river_basins)
                else:
                    # Intersect Voronoi polygons with coastal strip to create coastal watersheds
                    voronoi_gdf = gpd.GeoDataFrame(
                        geometry=voronoi_gdf.geometry,
                        crs=river_basins.crs
                    )
                    coastal_watersheds = gpd.overlay(coastal_strip, voronoi_gdf, how='intersection')
            except Exception as e:
                self.logger.error(f"Error dividing coastal strip: {str(e)}")
                # Fallback to simpler approach: use a buffer method
                coastal_watersheds = self._divide_coastal_strip_by_buffer_method(coastal_strip, river_basins)
            
            # If we still don't have valid coastal watersheds, use a simpler approach
            if coastal_watersheds is None or coastal_watersheds.empty:
                self.logger.warning("Failed to create divided coastal watersheds. Using buffer method.")
                coastal_watersheds = self._divide_coastal_strip_by_buffer_method(coastal_strip, river_basins)
                
            if coastal_watersheds is None or coastal_watersheds.empty:
                self.logger.info("No valid coastal watersheds created.")
                return river_network_path, river_basins_path
            
            # ---------- STEP 3: PROCESS AND CLEAN COASTAL WATERSHEDS ---------- #
            
            # Calculate areas and filter small fragments
            utm_crs = coastal_watersheds.estimate_utm_crs()
            coastal_watersheds_utm = coastal_watersheds.to_crs(utm_crs)
            
            # Calculate area
            coastal_watersheds_utm['area_km2'] = coastal_watersheds_utm.geometry.area / 1_000_000
            
            # Remove tiny fragments
            min_coastal_area = 0.1  # 0.1 km²
            coastal_watersheds_utm = coastal_watersheds_utm[coastal_watersheds_utm['area_km2'] > min_coastal_area]
            
            if coastal_watersheds_utm.empty:
                self.logger.info("No significant coastal watersheds after size filtering.")
                return river_network_path, river_basins_path
            
            # Add required attributes
            max_gru_id = river_basins['GRU_ID'].max() if 'GRU_ID' in river_basins.columns else 0
            coastal_watersheds_utm = coastal_watersheds_utm.reset_index(drop=True)
            coastal_watersheds_utm['GRU_ID'] = range(max_gru_id + 1, max_gru_id + 1 + len(coastal_watersheds_utm))
            coastal_watersheds_utm['gru_to_seg'] = 0  # No river segment
            coastal_watersheds_utm['GRU_area'] = coastal_watersheds_utm.geometry.area
            coastal_watersheds_utm['is_coastal'] = True
            
            # Convert back to original CRS
            coastal_watersheds = coastal_watersheds_utm.to_crs(river_basins.crs)
            
            # ---------- STEP 4: MERGE WITH EXISTING WATERSHEDS ---------- #
            
            # Add coastal attribute to existing basins
            river_basins['is_coastal'] = False
            
            # Ensure required columns exist
            required_cols = ['GRU_ID', 'gru_to_seg', 'GRU_area', 'is_coastal', 'geometry']
            for col in required_cols:
                if col not in coastal_watersheds.columns and col != 'geometry':
                    coastal_watersheds[col] = None if col == 'is_coastal' else 0
                    
            # Get only needed columns
            coastal_cols = [col for col in coastal_watersheds.columns if col in required_cols or col == 'geometry']
            
            # Merge with existing river basins
            combined_basins = pd.concat([
                river_basins, 
                coastal_watersheds[coastal_cols]
            ])
            
            # Save combined results
            combined_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_with_coastal.shp"
            combined_basins.to_file(combined_basins_path)
            
            self.logger.info(f"Added {len(coastal_watersheds)} coastal watersheds to the delineation.")
            self.logger.info(f"Combined river basins saved to: {combined_basins_path}")
            
            # Cleanup if requested
            if self.config.get('CLEANUP_INTERMEDIATE_FILES', True):
                shutil.rmtree(coastal_interim_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up coastal interim files: {coastal_interim_dir}")
            
            return river_network_path, combined_basins_path
            
        except Exception as e:
            self.logger.error(f"Error in coastal watershed delineation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return river_network_path, river_basins_path

    def _create_land_polygon_from_dem(self) -> gpd.GeoDataFrame:
        """Create a polygon representing land areas based on the DEM."""
        try:
            # Open the DEM
            with rasterio.open(str(self.dem_path)) as src:
                # Read DEM data
                dem_data = src.read(1)
                nodata_value = src.nodata
                transform = src.transform
                crs = src.crs
                
                # Create binary mask where elevation > 0
                land_mask = np.where(
                    (dem_data > 0) & (dem_data != nodata_value),
                    1,  # Land
                    0   # Ocean or nodata
                ).astype(np.uint8)
                
                # Use rasterio features to extract land polygons
                shapes = rasterio.features.shapes(
                    land_mask,
                    mask=land_mask == 1,
                    transform=transform
                )
                
                # Convert shapes to shapely geometries
                land_polygons = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]
                
                if not land_polygons:
                    self.logger.warning("No land areas detected in DEM.")
                    return None
                    
                # Create a GeoDataFrame with dissolved geometry
                land_gdf = gpd.GeoDataFrame(
                    geometry=[shapely.ops.unary_union(land_polygons)],
                    crs=crs
                )
                
                self.logger.info(f"Created land polygon from DEM with {len(land_polygons)} original features.")
                return land_gdf
                
        except Exception as e:
            self.logger.error(f"Error creating land polygon from DEM: {str(e)}")
            return None

    def _create_voronoi_tessellation(self, points_gdf):
        """
        Create Voronoi polygons from point data.
        
        Args:
            points_gdf (GeoDataFrame): Points to create Voronoi diagram from
            
        Returns:
            GeoDataFrame: Voronoi polygons
        """
        try:
            from scipy.spatial import Voronoi
            import numpy as np
            
            # Extract points coordinates - ensure we're getting actual points
            # This is where the previous error was occurring
            coords = []
            for geom in points_gdf.geometry:
                # Get centroid if it's not already a point
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    point = geom.centroid
                else:
                    point = geom
                    
                coords.append((point.x, point.y))
                
            coords = np.array(coords)
            
            if len(coords) < 4:
                self.logger.warning("Not enough points for Voronoi tessellation (need at least 4).")
                return None
            
            # Create Voronoi diagram
            vor = Voronoi(coords)
            
            # Create polygons from Voronoi regions
            regions = []
            for region in vor.regions:
                if not -1 in region and len(region) > 0:  # Valid regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) > 2:  # Valid polygon needs at least 3 points
                        regions.append(shapely.geometry.Polygon(polygon))
            
            # Create GeoDataFrame
            voronoi_gdf = gpd.GeoDataFrame(geometry=regions, crs=points_gdf.crs)
            
            # Create a convex hull around the points to limit Voronoi extent
            convex_hull = points_gdf.unary_union.convex_hull
            
            # Use a large buffer around the convex hull
            buffer_distance = 0.1  # ~10km in decimal degrees
            extended_hull = shapely.geometry.Polygon(convex_hull).buffer(buffer_distance)
            
            # Clip Voronoi polygons to the extended hull
            voronoi_gdf.geometry = [geom.intersection(extended_hull) for geom in voronoi_gdf.geometry]
            
            # Filter out empty geometries
            voronoi_gdf = voronoi_gdf[~voronoi_gdf.geometry.is_empty]
            
            self.logger.info(f"Created {len(voronoi_gdf)} Voronoi polygons.")
            return voronoi_gdf
            
        except Exception as e:
            self.logger.error(f"Error creating Voronoi tessellation: {str(e)}")
            return None
        
    def _divide_coastal_strip_by_extending_boundaries(self, coastal_strip, river_basins):
        """
        Divide coastal strip by extending the external boundaries of river basins.
        
        Args:
            coastal_strip (GeoDataFrame): The coastal areas to divide
            river_basins (GeoDataFrame): Existing river basins
            
        Returns:
            GeoDataFrame: Divided coastal watersheds
        """
        try:
            # Get the exterior boundaries of each basin
            basin_boundaries = []
            for idx, basin in river_basins.iterrows():
                # Get the exterior of the basin
                if basin.geometry.geom_type == 'Polygon':
                    boundary = basin.geometry.exterior
                elif basin.geometry.geom_type == 'MultiPolygon':
                    # Get the longest boundary for multipolygons
                    boundary = max([poly.exterior for poly in basin.geometry.geoms], 
                                key=lambda x: x.length)
                else:
                    continue
                    
                basin_boundaries.append({
                    'boundary': boundary,
                    'gru_id': basin['GRU_ID']
                })
            
            # Create a convex hull around all basins and extend it outward
            convex_hull = river_basins.unary_union.convex_hull
            ext_distance = 0.1  # ~10km in decimal degrees
            extended_hull = shapely.geometry.Polygon(convex_hull).buffer(ext_distance)
            
            # For each basin boundary, extend the lines to the extended hull
            extended_lines = []
            for basin_boundary in basin_boundaries:
                if isinstance(basin_boundary['boundary'], shapely.geometry.LineString):
                    # Only consider lines at the edge of the basin collection
                    coords = list(basin_boundary['boundary'].coords)
                    if len(coords) < 2:
                        continue
                        
                    # Sample points along the boundary for extending
                    num_points = max(5, len(coords) // 5)  # Sample fewer points for efficiency
                    sample_indices = np.linspace(0, len(coords)-1, num_points).astype(int)
                    
                    for i in sample_indices:
                        if i+1 >= len(coords):
                            continue
                            
                        p1 = shapely.geometry.Point(coords[i])
                        p2 = shapely.geometry.Point(coords[i+1])
                        
                        # Create a line segment
                        line = shapely.geometry.LineString([p1, p2])
                        
                        # Only extend lines that are at the edge (not touching other basins)
                        touches_other_basins = any(
                            line.intersects(other['boundary']) 
                            for other in basin_boundaries 
                            if other['gru_id'] != basin_boundary['gru_id']
                        )
                        
                        if not touches_other_basins:
                            # Calculate direction vector
                            dx = p2.x - p1.x
                            dy = p2.y - p1.y
                            length = (dx**2 + dy**2)**0.5
                            
                            if length > 0:
                                dx /= length
                                dy /= length
                                
                                # Create extended line that reaches the hull
                                factor = 1.0  # Extension factor
                                extended_p1 = shapely.geometry.Point(p1.x - dx * factor, p1.y - dy * factor)
                                extended_p2 = shapely.geometry.Point(p2.x + dx * factor, p2.y + dy * factor)
                                extended_line = shapely.geometry.LineString([extended_p1, extended_p2])
                                
                                # Clip the line to the extended hull
                                if extended_line.intersects(extended_hull.boundary):
                                    extended_line = extended_line.intersection(extended_hull)
                                    extended_lines.append(extended_line)
            
            # Combine extended lines with basin boundaries
            all_lines = extended_lines + [
                basin_boundary['boundary'] for basin_boundary in basin_boundaries
            ]
            
            # Create polygons from lines - use a modified approach
            coastal_geom = coastal_strip.geometry.unary_union
            divided_coastal = []
            
            # Use a buffer-based approach to divide the coastal strip
            for idx, basin in river_basins.iterrows():
                try:
                    # Create a buffer around the basin
                    buffer_dist = 0.01  # About 1km in decimal degrees
                    buffer = basin.geometry.buffer(buffer_dist)
                    
                    # Intersect with coastal strip
                    coastal_part = buffer.intersection(coastal_geom)
                    
                    if not coastal_part.is_empty:
                        if coastal_part.geom_type == 'GeometryCollection':
                            # Extract polygons from collection
                            for geom in coastal_part.geoms:
                                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                    divided_coastal.append({
                                        'geometry': geom,
                                        'basin_id': basin['GRU_ID']
                                    })
                        elif coastal_part.geom_type in ['Polygon', 'MultiPolygon']:
                            divided_coastal.append({
                                'geometry': coastal_part,
                                'basin_id': basin['GRU_ID']
                            })
                except Exception as e:
                    self.logger.warning(f"Error processing basin {basin['GRU_ID']}: {str(e)}")
            
            # Create GeoDataFrame from divided coastal watersheds
            if divided_coastal:
                coastal_watersheds = gpd.GeoDataFrame(
                    {
                        'geometry': [item['geometry'] for item in divided_coastal],
                        'parent_basin': [item['basin_id'] for item in divided_coastal]
                    },
                    crs=river_basins.crs
                )
                return coastal_watersheds
            else:
                self.logger.warning("No coastal watersheds created by boundary extension method.")
                return None
            
        except Exception as e:
            self.logger.error(f"Error dividing coastal strip by extending boundaries: {str(e)}")
            return None

    def _divide_coastal_strip_by_buffer_method(self, coastal_strip, river_basins):
        """
        Divide the coastal strip using a buffer-based method.
        
        This is a more robust fallback method that uses buffers around each basin
        to claim portions of the coastal strip.
        
        Args:
            coastal_strip (GeoDataFrame): Coastal areas to divide
            river_basins (GeoDataFrame): Existing river basins
            
        Returns:
            GeoDataFrame: Divided coastal watersheds
        """
        try:
            coastal_geom = coastal_strip.geometry.unary_union
            
            # Create an empty list to store divided coastal watersheds
            divided_coastal = []
            
            # Create multiple buffer sizes for more uniform coverage
            buffer_sizes = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]  # In decimal degrees
            remaining_coastal = coastal_geom
            
            for buffer_size in buffer_sizes:
                if remaining_coastal.is_empty:
                    break
                    
                # Process each basin to claim its portion of the coastal strip
                for idx, basin in river_basins.iterrows():
                    try:
                        # Create a buffer around the basin with gradient size
                        buffer = basin.geometry.buffer(buffer_size)
                        
                        # Intersect with remaining coastal strip
                        claimed_area = buffer.intersection(remaining_coastal)
                        
                        if not claimed_area.is_empty:
                            # Add to divided coastal watersheds
                            divided_coastal.append({
                                'geometry': claimed_area,
                                'basin_id': basin['GRU_ID']
                            })
                            
                            # Remove claimed area from remaining coastal strip
                            remaining_coastal = remaining_coastal.difference(claimed_area)
                    except Exception as e:
                        self.logger.warning(f"Error processing basin {basin['GRU_ID']} with buffer {buffer_size}: {str(e)}")
            
            # Handle any remaining coastal strip by assigning to nearest basin
            if not remaining_coastal.is_empty:
                self.logger.info("Assigning remaining coastal areas to nearest basins.")
                
                # Convert to GeoDataFrame for easier processing
                remaining_gdf = gpd.GeoDataFrame(
                    geometry=[remaining_coastal],
                    crs=river_basins.crs
                )
                
                # Explode to get individual polygons
                try:
                    # For newer geopandas versions
                    remaining_gdf = remaining_gdf.explode(index_parts=True).reset_index(drop=True)
                except:
                    # For older geopandas versions
                    remaining_gdf = remaining_gdf.explode().reset_index(drop=True)
                
                # For each remaining polygon, find the nearest basin
                for idx, row in remaining_gdf.iterrows():
                    nearest_basin = None
                    min_distance = float('inf')
                    
                    for basin_idx, basin in river_basins.iterrows():
                        distance = row.geometry.distance(basin.geometry)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_basin = basin['GRU_ID']
                    
                    if nearest_basin is not None:
                        divided_coastal.append({
                            'geometry': row.geometry,
                            'basin_id': nearest_basin
                        })
            
            # Create GeoDataFrame from divided coastal watersheds
            if divided_coastal:
                coastal_watersheds = gpd.GeoDataFrame(
                    {
                        'geometry': [item['geometry'] for item in divided_coastal],
                        'parent_basin': [item['basin_id'] for item in divided_coastal]
                    },
                    crs=river_basins.crs
                )
                
                # Dissolve by parent_basin to merge adjacent pieces
                coastal_watersheds = coastal_watersheds.dissolve(by='parent_basin').reset_index()
                
                return coastal_watersheds
            else:
                self.logger.warning("No coastal watersheds created by buffer method.")
                return None
            
        except Exception as e:
            self.logger.error(f"Error dividing coastal strip by buffer method: {str(e)}")
            return None

    @get_function_logger
    def delineate_point_buffer_shape(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Create a small square buffer around the pour point for point-scale simulations.
        
        This method creates a simple square buffer with 0.01 degree (~1km) around the pour point
        specified in the configuration. It saves the buffer as shapefiles in both the river_basins
        and catchment directories to satisfy CONFLUENCE's requirements.
        
        Returns:
            Tuple[Optional[Path], Optional[Path]]: Paths to the created river_basins and catchment shapefiles
        """
        try:
            self.logger.info(f"Creating point buffer shape for point-scale simulation at {self.domain_name}")
            
            # Get pour point coordinates
            pour_point_coords = self.config.get('POUR_POINT_COORDS', '').split('/')
            if len(pour_point_coords) != 2:
                self.logger.error(f"Invalid pour point coordinates: {self.config.get('POUR_POINT_COORDS')}")
                return None, None
            
            # Convert to floats
            try:
                lat, lon = float(pour_point_coords[0]), float(pour_point_coords[1])
            except ValueError:
                self.logger.error(f"Invalid pour point coordinates format: {self.config.get('POUR_POINT_COORDS')}")
                return None, None
            
            # Define buffer distance (0.01 degrees, approximately 1km at the equator)
            buffer_dist = self.config.get('POINT_BUFFER_DISTANCE')
            
            # Create a square buffer around the point
            min_lon = lon - buffer_dist
            max_lon = lon + buffer_dist
            min_lat = lat - buffer_dist
            max_lat = lat + buffer_dist
            
            # Create polygon geometry
            polygon = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat)
            ])
            
            # Create GeoDataFrame with the polygon
            gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')
            
            # Add required attributes
            gdf['GRU_ID'] = 1
            gdf['gru_to_seg'] = 1
            
            # Convert to UTM for area calculation
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf['GRU_area'] = gdf_utm.geometry.area
            
            # Create a simple point feature at the pour point for river network
            point = shapely.geometry.Point(lon, lat)
            river_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
            river_gdf['LINKNO'] = 1
            river_gdf['DSLINKNO'] = 0
            river_gdf['Length'] = 0
            river_gdf['Slope'] = 0
            river_gdf['GRU_ID'] = 1
            
            # Create directories if they don't exist
            river_basins_dir = self.project_dir / "shapefiles" / "river_basins"
            catchment_dir = self.project_dir / "shapefiles" / "catchment"
            river_network_dir = self.project_dir / "shapefiles" / "river_network"
            
            river_basins_dir.mkdir(parents=True, exist_ok=True)
            catchment_dir.mkdir(parents=True, exist_ok=True)
            river_network_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output paths
            river_basins_path = river_basins_dir / f"{self.domain_name}_riverBasins_point.shp"
            catchment_path = catchment_dir / f"{self.domain_name}_HRUs_point.shp"
            river_network_path = river_network_dir / f"{self.domain_name}_riverNetwork_point.shp"
            
            # Save shapefiles
            gdf.to_file(river_basins_path)
            gdf.to_file(catchment_path)
            river_gdf.to_file(river_network_path)
            
            self.logger.info(f"Point buffer shapefiles created successfully at:")
            self.logger.info(f"  - River basins: {river_basins_path}")
            self.logger.info(f"  - Catchment: {catchment_path}")
            self.logger.info(f"  - River network: {river_network_path}")
            
            return river_network_path, river_basins_path
            
        except Exception as e:
            self.logger.error(f"Error creating point buffer shape: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def _find_neighbors(self, geometry, gdf, exclude_idx):
        """Find neighboring GRUs that share a boundary."""
        return gdf[
            (gdf.index != exclude_idx) & 
            (gdf.geometry.boundary.intersects(geometry.boundary))
        ]

    def _merge_small_grus(self, gru_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Merge GRUs smaller than the minimum size threshold with their neighbors.
        Optimized version with spatial indexing and vectorized operations.
        """
        self.logger.info(f"Starting GRU merging process (minimum size: {self.min_gru_size} km²)")
        initial_count = len(gru_gdf)
        
        # Ensure CRS is geographic and convert to UTM for area calculations
        gru_gdf.set_crs(epsg=4326, inplace=True)
        utm_crs = gru_gdf.estimate_utm_crs()
        gru_gdf_utm = gru_gdf.to_crs(utm_crs)
        
        # Clean geometries (vectorized)
        gru_gdf_utm['geometry'] = gru_gdf_utm['geometry'].apply(self._clean_geometries)
        gru_gdf_utm = gru_gdf_utm[gru_gdf_utm['geometry'].notnull()]
        
        # Store original boundary
        original_boundary = unary_union(gru_gdf_utm.geometry)
        
        # Calculate areas in km² (vectorized)
        gru_gdf_utm['area'] = gru_gdf_utm.geometry.area / 1_000_000
        
        # Create spatial index for faster neighbor finding
        spatial_index = gru_gdf_utm.sindex
        
        merged_count = 0
        while True:
            small_grus = gru_gdf_utm[gru_gdf_utm['area'] < self.min_gru_size]
            if len(small_grus) == 0:
                break
                
            # Process multiple small GRUs in parallel
            small_grus_to_merge = small_grus.head(100)  # Process in batches
            if len(small_grus_to_merge) == 0:
                break
                
            for idx, small_gru in small_grus_to_merge.iterrows():
                try:
                    small_gru_geom = self._clean_geometries(small_gru.geometry)
                    if small_gru_geom is None:
                        gru_gdf_utm = gru_gdf_utm.drop(idx)
                        continue
                    
                    # Use spatial index to find potential neighbors
                    possible_matches_idx = list(spatial_index.intersection(small_gru_geom.bounds))
                    possible_matches = gru_gdf_utm.iloc[possible_matches_idx]
                    
                    # Filter actual neighbors
                    neighbors = possible_matches[
                        (possible_matches.index != idx) & 
                        (possible_matches.geometry.boundary.intersects(small_gru_geom.boundary))
                    ]
                    
                    if len(neighbors) > 0:
                        largest_neighbor = neighbors.loc[neighbors['area'].idxmax()]
                        merged_geometry = unary_union([small_gru_geom, largest_neighbor.geometry])
                        merged_geometry = self._simplify_geometry(merged_geometry)
                        
                        if merged_geometry and merged_geometry.is_valid:
                            gru_gdf_utm.at[largest_neighbor.name, 'geometry'] = merged_geometry
                            gru_gdf_utm.at[largest_neighbor.name, 'area'] = merged_geometry.area / 1_000_000
                            gru_gdf_utm = gru_gdf_utm.drop(idx)
                            merged_count += 1
                            
                except Exception as e:
                    self.logger.error(f"Error merging GRU {idx}: {str(e)}")
            
            # Update spatial index after batch processing
            spatial_index = gru_gdf_utm.sindex
        
        # Handle gaps (vectorized where possible)
        current_coverage = unary_union(gru_gdf_utm.geometry)
        gaps = original_boundary.difference(current_coverage)
        if not gaps.is_empty:
            gap_geoms = list(gaps.geoms) if gaps.geom_type == 'MultiPolygon' else [gaps]
            
            for gap in gap_geoms:
                if gap.area > 0:
                    # Use spatial index to find nearest GRU
                    possible_matches_idx = list(spatial_index.nearest(gap.bounds))
                    nearest_gru = possible_matches_idx[0]
                    merged_geom = self._clean_geometries(unary_union([gru_gdf_utm.iloc[nearest_gru].geometry, gap]))
                    if merged_geom and merged_geom.is_valid:
                        gru_gdf_utm.iloc[nearest_gru, gru_gdf_utm.columns.get_loc('geometry')] = merged_geom
                        gru_gdf_utm.iloc[nearest_gru, gru_gdf_utm.columns.get_loc('area')] = merged_geom.area / 1_000_000
        
        # Reset index and update IDs (vectorized)
        gru_gdf_utm = gru_gdf_utm.reset_index(drop=True)
        gru_gdf_utm['GRU_ID'] = range(1, len(gru_gdf_utm) + 1)
        gru_gdf_utm['gru_to_seg'] = gru_gdf_utm['GRU_ID']
        
        # Convert back to original CRS
        gru_gdf_merged = gru_gdf_utm.to_crs(gru_gdf.crs)
        
        self.logger.info(f"GRU merging statistics:")
        self.logger.info(f"- Initial GRUs: {initial_count}")
        self.logger.info(f"- Merged {merged_count} small GRUs")
        self.logger.info(f"- Final GRUs: {len(gru_gdf_merged)}")
        self.logger.info(f"- Reduction: {((initial_count - len(gru_gdf_merged)) / initial_count) * 100:.1f}%")
        
        return gru_gdf_merged

    def run_gdal_processing(self):
        """Convert watershed raster to polygon shapefile"""
        # Ensure output directory exists
        output_dir = self.interim_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_raster = str(self.interim_dir / "elv-watersheds.tif")
        output_shapefile = str(self.interim_dir / "basin-watersheds.shp")
        
        try:
            # First attempt: Using gdal.Polygonize directly
            src_ds = gdal.Open(input_raster)
            if src_ds is None:
                raise RuntimeError(f"Could not open input raster: {input_raster}")
                
            srcband = src_ds.GetRasterBand(1)
            
            # Create output shapefile
            drv = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(output_shapefile):
                drv.DeleteDataSource(output_shapefile)
                
            dst_ds = drv.CreateDataSource(output_shapefile)
            if dst_ds is None:
                raise RuntimeError(f"Could not create output shapefile: {output_shapefile}")
                
            dst_layer = dst_ds.CreateLayer("watersheds", srs=None)
            if dst_layer is None:
                raise RuntimeError("Could not create output layer")
                
            # Add field for raster value
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            dst_layer.CreateField(fd)
            
            # Run polygonize
            gdal.Polygonize(srcband, srcband.GetMaskBand(), dst_layer, 0)
            
            # Cleanup
            dst_ds = None
            src_ds = None
            
            self.logger.info("Completed GDAL polygonization using direct method")
            
        except Exception as e:
            self.logger.warning(f"Direct polygonization failed: {str(e)}, trying command line method...")
            try:
                # Second attempt: Using command line tool without MPI
                command = f"gdal_polygonize.py -f 'ESRI Shapefile' {input_raster} {output_shapefile}"
                subprocess.run(command, shell=True, check=True)
                self.logger.info("Completed GDAL polygonization using command line method")
                
            except Exception as e:
                self.logger.error(f"All polygonization attempts failed: {str(e)}")
                raise

    def subset_upstream_geofabric(self) -> Tuple[Optional[Path], Optional[Path]]:
        try:
            basins_path = self.interim_dir / "basin-watersheds.shp"
            rivers_path = self.interim_dir / "basin-streams.shp"

            pour_point = self.load_geopandas(self.pour_point_path)
            basins = self.load_geopandas(basins_path)
            rivers = self.load_geopandas(rivers_path)
            
            self._process_geofabric(basins, rivers)
            
            subset_basins_path, subset_rivers_path = self._get_output_paths()
            
            if self.config.get('DELINEATE_BY_POURPOINT', True):
                basins, rivers, pour_point = self.ensure_crs_consistency(basins, rivers, pour_point)
                downstream_basin_id = self.find_basin_for_pour_point(pour_point, basins)
                river_graph = self.build_river_graph(rivers)
                upstream_basin_ids = self.find_upstream_basins(downstream_basin_id, river_graph)
                subset_basins = basins[basins['GRU_ID'].isin(upstream_basin_ids)].copy()
                subset_rivers = rivers[rivers['GRU_ID'].isin(upstream_basin_ids)].copy()
            else:
                subset_basins, subset_rivers = basins, rivers
            
            subset_basins = self._merge_small_grus(subset_basins)

            self._save_geofabric(subset_basins, subset_rivers, subset_basins_path, subset_rivers_path)
            return subset_rivers_path, subset_basins_path

        except Exception as e:
            self.logger.error(f"Error during geofabric subsetting: {str(e)}")
            return None, None

    def _process_geofabric(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame):
        basins['GRU_ID'] = basins['DN']
        rivers['GRU_ID'] = rivers['LINKNO']
        utm_crs = basins.estimate_utm_crs()
        basins_utm = basins.to_crs(utm_crs)
        basins['GRU_area'] = basins_utm.geometry.area 
        basins['gru_to_seg'] = basins['GRU_ID']
        basins = basins.drop(columns=['DN'])
        
        # Check for duplicate GRU_IDs
        original_count = len(basins)
        duplicated_ids = basins['GRU_ID'].duplicated(keep=False)
        duplicate_count = duplicated_ids.sum()
        
        if duplicate_count > 0:
            # Log information about duplicates
            if hasattr(self, 'logger'):
                self.logger.info(f"Found {duplicate_count} rows with duplicate GRU_ID values")
                
            # Keep only the largest area for each GRU_ID
            # Sort by GRU_ID and GRU_area (descending), then drop duplicates keeping the first occurrence
            basins = basins.sort_values(['GRU_ID', 'GRU_area'], ascending=[True, False])
            basins = basins.drop_duplicates(subset=['GRU_ID'], keep='first')
            
            # Log information about the removal of duplicates
            if hasattr(self, 'logger'):
                self.logger.info(f"Removed {duplicate_count - (duplicated_ids.sum() - duplicate_count)} duplicate GRU_IDs, keeping the largest area for each")
                self.logger.info(f"Remaining GRUs: {len(basins)}")
        
        return basins, rivers

    def _get_output_paths(self) -> Tuple[Path, Path]:
        subset_basins_path = self.config.get('OUTPUT_BASINS_PATH')
        subset_rivers_path = self.config.get('OUTPUT_RIVERS_PATH')
        
        if subset_basins_path == 'default':
            subset_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_delineate.shp"
        else:
            subset_basins_path = Path(self.config['OUTPUT_BASINS_PATH'])

        if subset_rivers_path == 'default':
            subset_rivers_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_delineate.shp"
        else:
            subset_rivers_path = Path(self.config['OUTPUT_RIVERS_PATH'])

        return subset_basins_path, subset_rivers_path

    def _save_geofabric(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, basins_path: Path, rivers_path: Path):
        """Save geofabric files with corrected geometries."""
        basins_path.parent.mkdir(parents=True, exist_ok=True)
        rivers_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fix polygon winding order
        basins['geometry'] = basins['geometry'].apply(lambda geom: self._fix_polygon_winding(geom))
        
        basins, rivers = self._process_geofabric(basins, rivers)

        # Save files
        basins.to_file(basins_path)
        rivers.to_file(rivers_path)
        self.logger.info(f"Subset basins shapefile saved to: {basins_path}")
        self.logger.info(f"Subset rivers shapefile saved to: {rivers_path}")

    def _fix_polygon_winding(self, geometry):
        """Ensure correct winding order for polygon geometries."""
        if geometry is None:
            return None
            
        try:
            # First try the new Shapely 2.0+ method
            if geometry.geom_type == 'Polygon':
                return geometry.orient(1.0)
            elif geometry.geom_type == 'MultiPolygon':
                return geometry.__class__([geom.orient(1.0) for geom in geometry.geoms])
        except AttributeError:
            # Fallback for older Shapely versions
            if geometry.geom_type == 'Polygon':
                # Make exterior ring counterclockwise
                if not geometry.exterior.is_ccw:
                    geometry = shapely.geometry.Polygon(
                        list(geometry.exterior.coords)[::-1],
                        [list(interior.coords)[::-1] for interior in geometry.interiors]
                    )
            elif geometry.geom_type == 'MultiPolygon':
                # Fix each polygon in the multipolygon
                polygons = []
                for poly in geometry.geoms:
                    if not poly.exterior.is_ccw:
                        poly = shapely.geometry.Polygon(
                            list(poly.exterior.coords)[::-1],
                            [list(interior.coords)[::-1] for interior in poly.interiors]
                        )
                    polygons.append(poly)
                geometry = shapely.geometry.MultiPolygon(polygons)
        
        return geometry

    def load_geopandas(self, path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            self.logger.warning(f"CRS is not defined for {path}. Setting to EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def ensure_crs_consistency(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, pour_point: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        target_crs = basins.crs or rivers.crs or pour_point.crs or "EPSG:4326"
        self.logger.info(f"Ensuring CRS consistency. Target CRS: {target_crs}")
        
        return (basins.to_crs(target_crs), rivers.to_crs(target_crs), pour_point.to_crs(target_crs))

    def find_basin_for_pour_point(self, pour_point: gpd.GeoDataFrame, basins: gpd.GeoDataFrame) -> Any:
        containing_basin = gpd.sjoin(pour_point, basins, how='left', predicate='within')
        if containing_basin.empty:
            self.logger.error("No basin contains the given pour point.")
            raise ValueError("No basin contains the given pour point.")
        return containing_basin.iloc[0]['GRU_ID']

    def build_river_graph(self, rivers: gpd.GeoDataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        for _, row in rivers.iterrows():
            current_basin = row['GRU_ID']
            for up_col in ['USLINKNO1', 'USLINKNO2']:
                upstream_basin = row[up_col]
                if upstream_basin != -9999:  # Assuming -9999 is the default value for no upstream link
                    G.add_edge(upstream_basin, current_basin)
        return G

    def find_upstream_basins(self, basin_id: Any, G: nx.DiGraph) -> set:
        if G.has_node(basin_id):
            upstream_basins = nx.ancestors(G, basin_id)
            upstream_basins.add(basin_id)
        else:
            self.logger.warning(f"Basin ID {basin_id} not found in the river network.")
            upstream_basins = set()
        return upstream_basins

    def cleanup(self):
        if self.config.get('CLEANUP_INTERMEDIATE_FILES', True):
            shutil.rmtree(self.interim_dir.parent, ignore_errors=True)
            self.logger.info(f"Cleaned up intermediate files: {self.interim_dir.parent}")

class GeofabricSubsetter:
    """
    Subsets geofabric data based on pour points and upstream basins.

    This class provides methods for loading, processing, and subsetting geofabric data
    for different hydrofabric types (MERIT, TDX, NWS).

    Attributes:
        config (Dict[str, Any]): Configuration settings for the subsetter.
        logger (logging.Logger): Logger for the subsetter.
        data_dir (Path): Directory for data storage.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
        hydrofabric_types (Dict[str, Dict[str, Union[str, List[str], int]]]): Configuration for different hydrofabric types.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):

        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.hydrofabric_types = {
            'MERIT': {
                'basin_id_col': 'COMID',
                'river_id_col': 'COMID',
                'upstream_cols': ['up1', 'up2', 'up3'],
                'upstream_default': -9999
            },
            'TDX': {
                'basin_id_col': 'streamID',
                'river_id_col': 'LINKNO',
                'upstream_cols': ['USLINKNO1', 'USLINKNO2'],
                'upstream_default': -9999
            },
            'NWS': {
                'basin_id_col': 'COMID',
                'river_id_col': 'COMID',
                'upstream_cols': ['toCOMID'],
                'upstream_default': 0
            }
        }

    def subset_geofabric(self):
        """
        Subset the geofabric based on the configuration settings.

        Returns:
            Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]: Subset basins and rivers GeoDataFrames.
        """
        hydrofabric_type = self.config.get('GEOFABRIC_TYPE').upper()
        if hydrofabric_type not in self.hydrofabric_types:
            self.logger.error(f"Unknown hydrofabric type: {hydrofabric_type}")
            return None

        fabric_config = self.hydrofabric_types[hydrofabric_type]

        # Load data
        basins = self.load_geopandas(self.config['SOURCE_GEOFABRIC_BASINS_PATH'])
        rivers = self.load_geopandas(self.config['SOURCE_GEOFABRIC_RIVERS_PATH'])
        
        if self.config['POUR_POINT_SHP_PATH'] == 'default':
            pourPoint_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pourPoint_path = Path(self.config['POUR_POINT_SHP_PATH'])

        if self.config['POUR_POINT_SHP_NAME'] == "default":
            pourPoint_name = f"{self.domain_name}_pourPoint.shp"    

        pour_point = self.load_geopandas(pourPoint_path / pourPoint_name)

        # Ensure CRS consistency
        basins, rivers, pour_point = self.ensure_crs_consistency(basins, rivers, pour_point)

        # Find downstream basin
        downstream_basin_id = self.find_basin_for_pour_point(pour_point, basins, fabric_config['basin_id_col'])

        # Build river network and find upstream basins
        river_graph = self.build_river_graph(rivers, fabric_config)
        upstream_basin_ids = self.find_upstream_basins(downstream_basin_id, river_graph)

        # Subset basins and rivers
        subset_basins = basins[basins[fabric_config['basin_id_col']].isin(upstream_basin_ids)].copy()
        subset_rivers = rivers[rivers[fabric_config['river_id_col']].isin(upstream_basin_ids)].copy()

        # Add CONFLUENCE specific columns dependiing on fabric

        if self.config.get('GEOFABRIC_TYPE') == 'NWS':
            subset_basins['GRU_ID'] = subset_basins['COMID']
            subset_basins['gru_to_seg'] = subset_basins['COMID']
            subset_basins = subset_basins.to_crs('epsg:3763')
            subset_basins['GRU_area'] = subset_basins.geometry.area 
            subset_basins = subset_basins.to_crs('epsg:4326')
            subset_rivers['LINKNO'] = subset_rivers['COMID']
            subset_rivers['DSLINKNO'] = subset_rivers['toCOMID']

        elif self.config.get('GEOFABRIC_TYPE') == 'TDX':
            subset_basins['GRU_ID'] = subset_basins['fid']
            subset_basins['gru_to_seg'] = subset_basins['streamID']
            subset_basins = subset_basins.to_crs('epsg:3763')
            subset_basins['GRU_area'] = subset_basins.geometry.area 
            subset_basins = subset_basins.to_crs('epsg:4326')


        elif self.config.get('GEOFABRIC_TYPE') == 'Merit':
            subset_basins['GRU_ID'] = subset_basins['COMID']
            subset_basins['gru_to_seg'] = subset_basins['COMID']
            subset_basins = subset_basins.to_crs('epsg:3763')
            subset_basins['GRU_area'] = subset_basins.geometry.area 
            subset_basins = subset_basins.to_crs('epsg:4326')
            subset_rivers['LINKNO'] = subset_rivers['COMID']
            subset_rivers['DSLINKNO'] = subset_rivers['NextDownID']
            subset_rivers = subset_rivers.to_crs('epsg:3763')
            subset_rivers['Length'] = subset_rivers.geometry.length 
            subset_rivers = subset_rivers.to_crs('epsg:4326')
            subset_rivers.rename(columns={'slope':'Slope'}, inplace = True)

        # Save subsets
        self.save_geofabric(subset_basins, subset_rivers)

        return subset_basins, subset_rivers

    def load_geopandas(self, path: str) -> gpd.GeoDataFrame:
        """
        Load a shapefile into a GeoDataFrame.

        Args:
            path (Union[str, Path]): Path to the shapefile.

        Returns:
            gpd.GeoDataFrame: Loaded GeoDataFrame.
        """
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            self.logger.warning(f"CRS is not defined for {path}. Setting to EPSG:4326.")
            gdf = gdf.set_crs("EPSG:4326")
        return gdf

    def ensure_crs_consistency(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame, pour_point: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Ensure CRS consistency across all GeoDataFrames.

        Args:
            basins (gpd.GeoDataFrame): Basins GeoDataFrame.
            rivers (gpd.GeoDataFrame): Rivers GeoDataFrame.
            pour_point (gpd.GeoDataFrame): Pour point GeoDataFrame.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: CRS-consistent GeoDataFrames.
        """
        target_crs = basins.crs or rivers.crs or pour_point.crs or "EPSG:4326"
        self.logger.info(f"Ensuring CRS consistency. Target CRS: {target_crs}")
        
        if basins.crs != target_crs:
            basins = basins.to_crs(target_crs)
        if rivers.crs != target_crs:
            rivers = rivers.to_crs(target_crs)
        if pour_point.crs != target_crs:
            pour_point = pour_point.to_crs(target_crs)
        
        return basins, rivers, pour_point

    def find_basin_for_pour_point(self, pour_point: gpd.GeoDataFrame, basins: gpd.GeoDataFrame, id_col: str) -> Any:
        """
        Find the basin containing the pour point.

        Args:
            pour_point (gpd.GeoDataFrame): Pour point GeoDataFrame.
            basins (gpd.GeoDataFrame): Basins GeoDataFrame.
            id_col (str): Name of the basin ID column.

        Returns:
            Any: ID of the basin containing the pour point.

        Raises:
            ValueError: If no basin contains the pour point.
        """
        containing_basin = gpd.sjoin(pour_point, basins, how='left', predicate='within')
        if containing_basin.empty:
            raise ValueError("No basin contains the given pour point.")
        return containing_basin.iloc[0][id_col]

    def build_river_graph(self, rivers: gpd.GeoDataFrame, fabric_config: Dict[str, Any]) -> nx.DiGraph:
        """
        Build a directed graph representing the river network.

        Args:
            rivers (gpd.GeoDataFrame): Rivers GeoDataFrame.
            fabric_config (Dict[str, Any]): Configuration for the specific hydrofabric type.

        Returns:
            nx.DiGraph: Directed graph of the river network.
        """
        G = nx.DiGraph()
        for _, row in rivers.iterrows():
            current_basin = row[fabric_config['river_id_col']]
            for up_col in fabric_config['upstream_cols']:
                upstream_basin = row[up_col]
                if upstream_basin != fabric_config['upstream_default']:
                    if fabric_config['upstream_cols'] == ['toCOMID']:  # NWS case
                        G.add_edge(current_basin, upstream_basin)
                    else:
                        G.add_edge(upstream_basin, current_basin)
        return G

    def find_upstream_basins(self, basin_id: Any, G: nx.DiGraph) -> set:
        """
        Find all upstream basins for a given basin ID.

        Args:
            basin_id (Any): ID of the basin to find upstream basins for.
            G (nx.DiGraph): Directed graph of the river network.

        Returns:
            set: Set of upstream basin IDs, including the given basin ID.
        """
        if G.has_node(basin_id):
            upstream_basins = nx.ancestors(G, basin_id)
            upstream_basins.add(basin_id)
        else:
            self.logger.warning(f"Basin ID {basin_id} not found in the river network.")
            upstream_basins = set()
        return upstream_basins

    def save_geofabric(self, subset_basins: gpd.GeoDataFrame, subset_rivers: gpd.GeoDataFrame):
        """
        Save the subset geofabric (basins and rivers) to shapefiles.

        Args:
            subset_basins (gpd.GeoDataFrame): Subset of basins to save.
            subset_rivers (gpd.GeoDataFrame): Subset of rivers to save.
        """
        if self.config['OUTPUT_BASINS_PATH'] == 'default':
            output_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_subset_{self.config.get('GEOFABRIC_TYPE')}.shp"
        else:
            output_basins_path = Path(self.config['OUTPUT_BASINS_PATH'])

        if self.config['OUTPUT_RIVERS_PATH'] == 'default':
            output_rivers_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_subset_{self.config.get('GEOFABRIC_TYPE')}.shp"
        else:
            output_rivers_path = Path(self.config['OUTPUT_RIVERS_PATH'])

        output_basins_path.parent.mkdir(parents=True, exist_ok=True)
        output_rivers_path.parent.mkdir(parents=True, exist_ok=True)

        subset_basins.to_file(output_basins_path)
        subset_rivers.to_file(output_rivers_path)

        self.logger.info(f"Subset basins shapefile saved to: {output_basins_path}")
        self.logger.info(f"Subset rivers shapefile saved to: {output_rivers_path}")

class LumpedWatershedDelineator:
    """
    Delineates lumped watersheds using TauDEM.

    This class provides methods for running TauDEM commands to delineate a lumped watershed
    based on a DEM and pour point.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the delineator.
        logger (logging.Logger): Logger for the delineator.
        data_dir (Path): Directory for data storage.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
        output_dir (Path): Directory for output files.
        mpi_processes (int): Number of MPI processes to use.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_dir = self.project_dir / "shapefiles/tempdir"
        self.mpi_processes = self.config.get('MPI_PROCESSES', 4)
        self.delineation_method = self.config.get('LUMPED_WATERSHED_METHOD', 'pysheds')
        self.dem_path = self.config.get('DEM_PATH')

        # Set up TauDEM directory (don't add to PATH)
        taudem_dir = self.config.get('TAUDEM_DIR')
        if taudem_dir == 'default':
            taudem_dir = str(self.data_dir / 'installs' / 'TauDEM' / 'bin')
        self.taudem_dir = Path(taudem_dir) if taudem_dir else None

        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        if self.dem_path == 'default':
            self.dem_path = self.project_dir / 'attributes' / 'elevation' / 'dem' / dem_name
        else:
            self.dem_path = Path(self.dem_path)

    def run_command(self, command: str, retry: bool = True) -> None:
        def get_run_command():
            if shutil.which("srun"):
                return "srun"
            elif shutil.which("mpirun"):
                return "mpirun"
            else:
                return None

        run_cmd = get_run_command()

        for attempt in range(self.max_retries if retry else 1):
            try:
                if run_cmd:
                    full_command = f"{run_cmd} {command}"
                else:
                    full_command = command

                self.logger.info(f"Running command: {full_command}")
                result = subprocess.run(full_command, check=True, shell=True, capture_output=True, text=True)
                self.logger.info(f"Command output: {result.stdout}")
                return
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error executing command: {full_command}")
                self.logger.error(f"Error details: {e.stderr}")
                if attempt < self.max_retries - 1 and retry:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                elif run_cmd:
                    self.logger.info(f"Trying without {run_cmd}...")
                    run_cmd = None  # Try without srun/mpirun on the next attempt
                else:
                    raise

    @get_function_logger
    def delineate_lumped_watershed(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Delineate a lumped watershed using either TauDEM or pysheds.
        
        Returns:
            Tuple[Optional[Path], Optional[Path]]: Paths to the delineated river network and river basins shapefiles
        """
        self.logger.info(f"Starting lumped watershed delineation for {self.domain_name}")
        
        # Get pour point path
        pour_point_path = self.config.get('POUR_POINT_SHP_PATH')
        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(self.config['POUR_POINT_SHP_PATH'])
            
        if self.config['POUR_POINT_SHP_NAME'] == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"
        
        self.pour_point_path = pour_point_path
        
        # Define output paths
        river_basins_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_lumped.shp"
        river_network_path = self.project_dir / "shapefiles" / "river_network" / f"{self.domain_name}_riverNetwork_lumped.shp"
        
        # Create directories if they don't exist
        river_basins_path.parent.mkdir(parents=True, exist_ok=True)
        river_network_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Delineate watershed based on selected method
        if self.delineation_method.lower() == 'pysheds':
            watershed_path = self.delineate_with_pysheds()
        else:  # default to TauDEM
            watershed_path = self.delineate_with_taudem()
        

        # Create river network shapefile from pour point
        self.create_river_network(self.pour_point_path, river_network_path)
        
        # Ensure required fields are present in both shapefiles
        self.ensure_required_fields(river_basins_path, river_network_path)
        
        return river_network_path, river_basins_path

    def create_river_network(self, pour_point_path: Path, river_network_path: Path) -> None:
        """
        Create a simple river network shapefile based on the pour point.
        
        Args:
            pour_point_path (Path): Path to the pour point shapefile
            river_network_path (Path): Path to save the river network shapefile
        """
        try:
            # Load pour point
            pour_point_gdf = gpd.read_file(pour_point_path)
            
            # Create river network from pour point
            river_network = pour_point_gdf.copy()
            
            # Add required fields for river network
            river_network['LINKNO'] = 1
            river_network['DSLINKNO'] = 0  # Outlet has no downstream link
            river_network['Length'] = 100.0  # Placeholder length in meters
            river_network['Slope'] = 0.01   # Placeholder slope
            river_network['GRU_ID'] = 1
            
            # Save river network shapefile
            river_network.to_file(river_network_path)
            self.logger.info(f"Created river network shapefile at: {river_network_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating river network: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def ensure_required_fields(self, river_basins_path: Path, river_network_path: Path) -> None:
        """
        Ensure that all required fields are present in both shapefiles.
        
        Args:
            river_basins_path (Path): Path to the river basins shapefile
            river_network_path (Path): Path to the river network shapefile
        """
        try:
            # Load and check river basins
            basins_gdf = gpd.read_file(river_basins_path)
            
            # Add required fields for basins if missing
            required_basin_fields = {
                'GRU_ID': 1,
                'gru_to_seg': 1
            }
            
            # Calculate area in square meters
            if 'GRU_area' not in basins_gdf.columns:
                utm_crs = basins_gdf.estimate_utm_crs()
                basins_utm = basins_gdf.to_crs(utm_crs)
                basins_gdf['GRU_area'] = basins_utm.geometry.area
                # Convert back to original CRS
                basins_gdf = basins_gdf.to_crs(basins_gdf.crs)
            
            # Add any missing fields
            for field, default_value in required_basin_fields.items():
                if field not in basins_gdf.columns:
                    basins_gdf[field] = default_value
            
            # Save updated basin shapefile
            basins_gdf.to_file(river_basins_path)
            self.logger.info(f"Updated river basins shapefile with required fields at: {river_basins_path}")
            
            # Load and check river network if it exists
            if river_network_path.exists():
                network_gdf = gpd.read_file(river_network_path)
                
                # Add required fields for network if missing
                required_network_fields = {
                    'LINKNO': 1,
                    'DSLINKNO': 0,
                    'Length': 100.0,
                    'Slope': 0.01,
                    'GRU_ID': 1
                }
                
                # Add any missing fields
                for field, default_value in required_network_fields.items():
                    if field not in network_gdf.columns:
                        network_gdf[field] = default_value
                
                # Save updated network shapefile
                network_gdf.to_file(river_network_path)
                self.logger.info(f"Updated river network shapefile with required fields at: {river_network_path}")
        
        except Exception as e:
            self.logger.error(f"Error ensuring required fields: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def delineate_with_pysheds(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using pysheds.
        
        Returns:
            Optional[Path]: Path to the delineated watershed shapefile, or None if delineation fails.
        """
        try:
            from pysheds.grid import Grid  # Import here to handle potential missing dependency
            
            self.logger.info(f"Delineating watershed using pysheds for {self.domain_name}")
            
            # Initialize grid from raster
            grid = Grid.from_raster(str(self.dem_path))
            
            # Read the DEM
            dem = grid.read_raster(str(self.dem_path))
            
            # Read the pour point
            pour_point = gpd.read_file(self.pour_point_path)
            pour_point = pour_point.to_crs(grid.crs)
            x, y = pour_point.geometry.iloc[0].coords[0]
            
            # Condition DEM
            pit_filled_dem = grid.fill_pits(dem)
            flooded_dem = grid.fill_depressions(pit_filled_dem)
            inflated_dem = grid.resolve_flats(flooded_dem)
            
            # Compute flow direction
            fdir = grid.flowdir(inflated_dem)
            
            # Delineate the catchment
            catch = grid.catchment(x, y, fdir, xytype='coordinate')
            
            # Create a binary mask of the catchment
            mask = np.where(catch, 1, 0).astype(np.uint8)
            
            # Convert the mask to a polygon
            shapes = rasterio.features.shapes(mask, transform=grid.affine)
            polygons = [Polygon(shape[0]['coordinates'][0]) for shape in shapes if shape[1] == 1]
            
            if not polygons:
                self.logger.error("No watershed polygon generated.")
                return None
                
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=grid.crs)
            gdf = gdf.dissolve()  # Merge all polygons into one
            gdf = gdf.reset_index(drop=True)
            
            # Add required attributes
            gdf['GRU_ID'] = 1
            gdf['gru_to_seg'] = 1
            
            # Calculate area in square meters
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf['GRU_area'] = gdf_utm.geometry.area
            
            # Convert back to geographic coordinates
            gdf = gdf.to_crs('EPSG:4326')
            
            # Save the watershed shapefile
            watershed_shp_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_lumped.shp"
            watershed_shp_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(watershed_shp_path)
            
            self.logger.info(f"Watershed shapefile created at: {watershed_shp_path}")
            return watershed_shp_path
            
        except ImportError:
            self.logger.error("pysheds not installed. Please install it with 'pip install pysheds'")
            return None
        except Exception as e:
            self.logger.error(f"Error during pysheds watershed delineation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def delineate_with_taudem(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using TauDEM.
        
        Returns:
            Optional[Path]: Path to the delineated watershed shapefile, or None if delineation fails.
        """
        try:
            # Check if TauDEM directory exists
            if not self.taudem_dir or not self.taudem_dir.exists():
                self.logger.error(f"TauDEM directory not found: {self.taudem_dir}")
                raise RuntimeError("TauDEM directory not available")
                
            if not self.pour_point_path.is_file():
                self.logger.error(f"Pour point file not found: {self.pour_point_path}")
                return None
                
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine the correct MPI command
            def get_run_command():
                if shutil.which("srun"):
                    return "srun"
                elif shutil.which("mpirun"):
                    return "mpirun"
                else:
                    return ""  # Empty string for no MPI launcher
                    
            mpi_cmd = get_run_command()
            if mpi_cmd:
                mpi_prefix = f"{mpi_cmd} -n {self.mpi_processes} "
            else:
                mpi_prefix = ""
            
            # Always load GDAL module before TauDEM commands
            module_prefix = "module load gdal/3.9.2 && "
            self.logger.info("Loading GDAL module: gdal/3.9.2")
            
            # TauDEM processing steps with absolute paths
            steps = [
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/pitremove -z {self.dem_path} -fel {self.output_dir}/fel.tif",
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/d8flowdir -fel {self.output_dir}/fel.tif -p {self.output_dir}/p.tif -sd8 {self.output_dir}/sd8.tif",
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/aread8 -p {self.output_dir}/p.tif -ad8 {self.output_dir}/ad8.tif",
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/threshold -ssa {self.output_dir}/ad8.tif -src {self.output_dir}/src.tif -thresh 100",
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/moveoutletstostrm -p {self.output_dir}/p.tif -src {self.output_dir}/src.tif -o {self.pour_point_path} -om {self.output_dir}/om.shp",
                f"{module_prefix}{mpi_prefix}{self.taudem_dir}/gagewatershed -p {self.output_dir}/p.tif -o {self.output_dir}/om.shp -gw {self.output_dir}/watershed.tif -id {self.output_dir}/watershed_id.txt"
            ]
            
            for step in steps:
                self.logger.info(f"Running TauDEM command: {step}")
                self.run_command(step)
                self.logger.info(f"Completed TauDEM step")
                
            # Convert the watershed raster to polygon
            watershed_shp_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_lumped.shp"
            watershed_shp_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.raster_to_polygon(self.output_dir / "watershed.tif", watershed_shp_path)
            
            # Add required attributes if they don't exist
            watershed_gdf = gpd.read_file(watershed_shp_path)
            
            if 'GRU_ID' not in watershed_gdf.columns:
                watershed_gdf['GRU_ID'] = 1
                
            if 'gru_to_seg' not in watershed_gdf.columns:
                watershed_gdf['gru_to_seg'] = 1
                
            # Calculate area in square meters if it doesn't exist
            if 'GRU_area' not in watershed_gdf.columns:
                utm_crs = watershed_gdf.estimate_utm_crs()
                watershed_utm = watershed_gdf.to_crs(utm_crs)
                watershed_gdf['GRU_area'] = watershed_utm.geometry.area
                watershed_gdf = watershed_gdf.to_crs('EPSG:4326')
                
            # Save updated watershed shapefile
            watershed_gdf.to_file(watershed_shp_path)
            
            self.logger.info(f"Updated watershed shapefile at: {watershed_shp_path}")
            
            # Clean up temporary files if requested
            if self.config.get('CLEANUP_INTERMEDIATE_FILES', True):
                shutil.rmtree(self.output_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up intermediate files: {self.output_dir}")
                
            return watershed_shp_path
            
        except Exception as e:
            self.logger.error(f"Error during TauDEM watershed delineation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def raster_to_polygon(self, raster_path: Path, output_shp_path: Path):
        """
        Convert a raster to a polygon shapefile.

        Args:
            raster_path (Path): Path to the input raster file.
            output_shp_path (Path): Path to save the output shapefile.

        Raises:
            ValueError: If no polygon with ID = 1 is found in the watershed shapefile.
        """
        gdal.UseExceptions()
        ogr.UseExceptions()

        # Open the raster
        raster = gdal.Open(str(raster_path))
        band = raster.GetRasterBand(1)

        # Create a temporary shapefile
        temp_shp_path = output_shp_path.with_name(output_shp_path.stem + "_temp.shp")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        temp_ds = driver.CreateDataSource(str(temp_shp_path))
        temp_layer = temp_ds.CreateLayer("watershed", srs=None)

        # Add a field to the layer
        field_def = ogr.FieldDefn("ID", ogr.OFTInteger)
        temp_layer.CreateField(field_def)

        # Polygonize the raster
        gdal.Polygonize(band, None, temp_layer, 0, [], callback=None)

        # Close the temporary datasource
        temp_ds = None
        raster = None

        # Read the temporary shapefile with geopandas
        gdf = gpd.read_file(temp_shp_path)

        # Filter to keep only the shape with ID = 1
        filtered_gdf = gdf[gdf['ID'] == 1]
        filtered_gdf = filtered_gdf.set_crs('epsg:4326')

        if filtered_gdf.empty:
            self.logger.error("No polygon with ID = 1 found in the watershed shapefile.")
            raise ValueError("No polygon with ID = 1 found in the watershed shapefile.")

        # Save the filtered GeoDataFrame to the final shapefile
        filtered_gdf.to_file(output_shp_path)

        # Remove all temporary files
        temp_files = glob.glob(str(temp_shp_path.with_suffix(".*")))
        for temp_file in temp_files:
            Path(temp_file).unlink()
            self.logger.info(f"Removed temporary file: {temp_file}")

        self.logger.info(f"Filtered watershed shapefile created at: {output_shp_path}")