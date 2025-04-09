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
        self._set_taudem_path()
        self.max_retries = self.config.get('MAX_RETRIES', 3)
        self.retry_delay = self.config.get('RETRY_DELAY', 5)
        self.min_gru_size = self.config.get('MIN_GRU_SIZE', 5.0)  # Default 1 km²
        #self.pour_point_path = self.project_dir / 'shapefiles' / 'pour_point' / f"{self.config['DOMAIN_NAME']}_pourPoint.shp"

    def _get_dem_path(self) -> Path:
        dem_path = self.config.get('DEM_PATH')
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        if dem_path == 'default':
            return self.project_dir / 'attributes' / 'elevation' / 'dem' / dem_name
        return Path(dem_path)

    def _set_taudem_path(self):
        taudem_dir = self.config['TAUDEM_DIR']
        os.environ['PATH'] = f"{os.environ['PATH']}:{taudem_dir}"

    def run_command(self, command: str, retry: bool = True) -> None:
        for attempt in range(self.max_retries if retry else 1):
            try:
                self.logger.info(f"Running command: {command}")
                result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
                self.logger.info(f"Command output: {result.stdout}")
                return
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error executing command: {command}")
                self.logger.error(f"Error details: {e.stderr}")
                if attempt < self.max_retries - 1 and retry:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
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

        steps = [
            f"pitremove -z {dem_path} -fel {self.interim_dir}/elv-fel.tif -v",
            f"d8flowdir -fel {self.interim_dir}/elv-fel.tif -sd8 {self.interim_dir}/elv-sd8.tif -p {self.interim_dir}/elv-fdir.tif",
            f"aread8 -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -nc",
            f"gridnet -p {self.interim_dir}/elv-fdir.tif -plen {self.interim_dir}/elv-plen.tif -tlen {self.interim_dir}/elv-tlen.tif -gord {self.interim_dir}/elv-gord.tif",
            f"threshold -ssa {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -thresh {threshold}",
            f"moveoutletstostrm -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",
            f"streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif"
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
            coastal_watersheds_utm['gru_