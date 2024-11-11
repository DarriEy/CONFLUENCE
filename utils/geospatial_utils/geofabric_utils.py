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
import time

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

    def run_gdal_processing(self):
        command = f"gdal_polygonize.py {self.interim_dir}/elv-watersheds.tif {self.interim_dir}/basin-watersheds.shp"
        self.run_command(command)
        self.logger.info("Completed GDAL polygonization")

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
        basins_path.parent.mkdir(parents=True, exist_ok=True)
        rivers_path.parent.mkdir(parents=True, exist_ok=True)
        basins.to_file(basins_path)
        rivers.to_file(rivers_path)
        self.logger.info(f"Subset basins shapefile saved to: {basins_path}")
        self.logger.info(f"Subset rivers shapefile saved to: {rivers_path}")

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

        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        if self.dem_path == 'default':
            self.dem_path = self.project_dir / 'attributes' / 'elevation' / 'dem' / dem_name
        else:
            self.dem_path = Path(self.dem_path)
    
    def delineate_with_pysheds(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using pysheds.

        Returns:
            Optional[Path]: Path to the delineated watershed shapefile, or None if delineation fails.
        """

        pour_point_path = self.config.get('POUR_POINT_SHP_PATH')

        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(self.config['POUR_POINT_SHP_PATH'])

        if self.config['POUR_POINT_SHP_NAME'] == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"

        self.pour_point_path = pour_point_path

        try:
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

            gdf['GRU_ID'] = 1
            gdf['gru_to_seg'] = 1
            gdf = gdf.to_crs('epsg:3763')
            gdf['GRU_area'] = gdf.geometry.area 
            gdf = gdf.to_crs('epsg:4326')

            # Save the watershed shapefile
            watershed_shp_path = self.project_dir / "shapefiles/river_basins" / f"{self.domain_name}_riverBasins_lumped.shp"
            gdf.to_file(watershed_shp_path)

            self.logger.info(f"Lumped watershed delineation completed using pysheds for {self.domain_name}")
            return watershed_shp_path

        except Exception as e:
            self.logger.error(f"Error during pysheds watershed delineation: {str(e)}")
            return None

    def run_command(self, command: str):
        """
        Run a shell command and log any errors.

        Args:
            command (str): The command to run.

        Raises:
            subprocess.CalledProcessError: If the command fails.
        """
        try:
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing command: {command}")
            self.logger.error(f"Error details: {str(e)}")
            raise

    @get_function_logger
    def delineate_lumped_watershed(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using either TauDEM or pysheds.

        Returns:
            Optional[Path]: Path to the delineated watershed shapefile, or None if delineation fails.
        """
        self.logger.info(f"Starting lumped watershed delineation for {self.domain_name}")

        if self.delineation_method.lower() == 'pysheds':
            return self.delineate_with_pysheds()
        else:  # default to TauDEM
            return self.delineate_with_taudem()

    @get_function_logger
    def delineate_with_taudem(self) -> Optional[Path]:
        """
        Delineate a lumped watershed using TauDEM.

        Returns:
            Optional[Path]: Path to the delineated watershed shapefile, or None if delineation fails.
        """
        self.logger.info(f"Starting lumped watershed delineation for {self.domain_name}")
            
        pour_point_path = self.config.get('POUR_POINT_SHP_PATH')

        if pour_point_path == 'default':
            pour_point_path = self.project_dir / "shapefiles" / "pour_point"
        else:
            pour_point_path = Path(self.config['POUR_POINT_SHP_PATH'])

        if self.config['POUR_POINT_SHP_NAME'] == "default":
            pour_point_path = pour_point_path / f"{self.domain_name}_pourPoint.shp"

        self.pour_point_path = pour_point_path

        if not pour_point_path.is_file():
            self.logger.error(f"Pour point file not found: {pour_point_path}")
            return None

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TauDEM processing steps for lumped watershed delineation
        steps = [
            f"mpirun -n {self.mpi_processes} pitremove -z {self.dem_path} -fel {self.output_dir}/fel.tif",
            f"mpirun -n {self.mpi_processes} d8flowdir -fel {self.output_dir}/fel.tif -p {self.output_dir}/p.tif -sd8 {self.output_dir}/sd8.tif",
            f"mpirun -n {self.mpi_processes} aread8 -p {self.output_dir}/p.tif -ad8 {self.output_dir}/ad8.tif",
            f"mpirun -n {self.mpi_processes} threshold -ssa {self.output_dir}/ad8.tif -src {self.output_dir}/src.tif -thresh 100",
            f"mpirun -n {self.mpi_processes} moveoutletstostrm -p {self.output_dir}/p.tif -src {self.output_dir}/src.tif -o {pour_point_path} -om {self.output_dir}/om.shp",
            f"mpirun -n {self.mpi_processes} gagewatershed -p {self.output_dir}/p.tif -o {self.output_dir}/om.shp -gw {self.output_dir}/watershed.tif -id {self.output_dir}/watershed_id.txt"
        ]

        for step in steps:
            self.run_command(step)
            self.logger.info(f"Completed TauDEM step: {step}")

        # Convert the watershed raster to polygon
        watershed_shp_path = self.project_dir / "shapefiles/river_basins" / f"{self.domain_name}_riverBasins_lumped.shp"
        self.raster_to_polygon(self.output_dir / "watershed.tif", watershed_shp_path)


        self.logger.info(f"Lumped watershed delineation completed for {self.domain_name}")

        shutil.rmtree(self.output_dir, ignore_errors=True)

        return watershed_shp_path

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