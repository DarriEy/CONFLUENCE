import os
import sys
import pandas as pd # type: ignore
import netCDF4 as nc4 # type: ignore
import geopandas as gpd # type: ignore
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from typing import Dict, Any
import easymore as esmr # type: ignore
import subprocess
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent))

class MizuRoutePreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.mizuroute_setup_dir = self.project_dir / "settings" / "mizuRoute"

    def run_preprocessing(self):
        self.logger.debug("Starting mizuRoute spatial preprocessing")
        self.copy_base_settings()
        self.create_network_topology_file()
        self.logger.info(f"Should we remap?: {self.config.get('SETTINGS_MIZU_NEEDS_REMAP')}")
        if self.config.get('SETTINGS_MIZU_NEEDS_REMAP'):
            self.remap_summa_catchments_to_routing()

        # NEW: choose control writer based on source model
        if self.config.get('MIZU_FROM_MODEL') == 'FUSE' or self.config.get('FUSE_ROUTING_INTEGRATION', 'mizuRoute') == 'mizuRoute':
            self.create_fuse_control_file()
        else:
            self.create_control_file()

        self.logger.info("mizuRoute spatial preprocessing completed")


    def copy_base_settings(self):
        self.logger.info("Copying mizuRoute base settings")
        base_settings_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_base_settings' / 'mizuRoute'
        self.mizuroute_setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.mizuroute_setup_dir / file)
        self.logger.info("mizuRoute base settings copied")

    def create_area_weighted_remap_file(self):
        """Create remapping file with area-based weights from delineated catchments"""
        self.logger.info("Creating area-weighted remapping file")
        
        # Load topology to get HRU information
        topology_file = self.mizuroute_setup_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY')
        with xr.open_dataset(topology_file) as topo:
            hru_ids = topo['hruId'].values
        
        n_hrus = len(hru_ids)
        
        # Use the weights stored during topology creation
        if hasattr(self, 'subcatchment_weights') and hasattr(self, 'subcatchment_gru_ids'):
            weights = self.subcatchment_weights
            gru_ids = self.subcatchment_gru_ids
        else:
            # Fallback: load from delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.config['DOMAIN_NAME']}_catchment_delineated.shp"
            shp_catchments = gpd.read_file(catchment_path)
            weights = shp_catchments['avg_subbas'].values
            gru_ids = shp_catchments['GRU_ID'].values.astype(int)
        
        remap_name = self.config.get('SETTINGS_MIZU_REMAP')
        
        with nc4.Dataset(self.mizuroute_setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Area-weighted remapping for lumped to distributed routing')
            
            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU
            
            # Create variables
            # RN_hruId: The routing HRU IDs (from delineated catchments)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = gru_ids
            rn_hru.long_name = 'River network HRU ID'
            
            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'
            
            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'
            
            # weight: Area-based weights from delineated catchments
            weight_var = ncid.createVariable('weight', 'f8', ('data',))
            weight_var[:] = weights
            weight_var.long_name = 'Areal weights based on delineated subcatchment areas'
        
        self.logger.info(f"Area-weighted remapping file created with {n_hrus} HRUs")
        self.logger.info(f"Weight range: {weights.min():.4f} to {weights.max():.4f}")
        self.logger.info(f"Weight sum: {weights.sum():.4f}")

    def _check_if_headwater_basin(self, shp_river):
        """
        Check if this is a headwater basin with None/invalid river network data.
        
        Args:
            shp_river: GeoDataFrame of river network
            
        Returns:
            bool: True if this appears to be a headwater basin with invalid network data
        """
        # Check for critical None values in key columns
        seg_id_col = self.config.get('RIVER_NETWORK_SHP_SEGID')
        downseg_id_col = self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')
        
        if seg_id_col in shp_river.columns and downseg_id_col in shp_river.columns:
            # Check if all segment IDs are None/null
            seg_ids_null = shp_river[seg_id_col].isna().all()
            downseg_ids_null = shp_river[downseg_id_col].isna().all()
            
            if seg_ids_null and downseg_ids_null:
                self.logger.info("Detected headwater basin: all river network IDs are None/null")
                return True
                
            # Also check for string 'None' values (sometimes shapefiles store None as string)
            if shp_river[seg_id_col].dtype == 'object':
                seg_ids_none_str = (shp_river[seg_id_col] == 'None').all()
                downseg_ids_none_str = (shp_river[downseg_id_col] == 'None').all()
                
                if seg_ids_none_str and downseg_ids_none_str:
                    self.logger.info("Detected headwater basin: all river network IDs are 'None' strings")
                    return True
        
        return False

    def _create_synthetic_river_network(self, shp_river, hru_ids):
        """
        Create a synthetic single-segment river network for headwater basins.
        
        Args:
            shp_river: Original GeoDataFrame (with None values)
            hru_ids: Array of HRU IDs from delineated catchments
            
        Returns:
            GeoDataFrame: Modified river network with synthetic single segment
        """
        self.logger.info("Creating synthetic river network for headwater basin")
        
        # Use the first HRU ID as the segment ID (should be reasonable identifier)
        synthetic_seg_id = int(hru_ids[0]) if len(hru_ids) > 0 else 1
        
        # Create synthetic values for the single segment
        synthetic_data = {
            self.config.get('RIVER_NETWORK_SHP_SEGID'): synthetic_seg_id,
            self.config.get('RIVER_NETWORK_SHP_DOWNSEGID'): 0,  # Outlet (downstream ID = 0)
            self.config.get('RIVER_NETWORK_SHP_LENGTH'): 1000.0,  # Default 1 km length
            self.config.get('RIVER_NETWORK_SHP_SLOPE'): 0.001,  # Default 0.1% slope
        }
        
        # Get the geometry column name (usually 'geometry')
        geom_col = shp_river.geometry.name
        
        # Create a simple point geometry at the centroid of the original (if it exists)
        if not shp_river.empty and shp_river.geometry.iloc[0] is not None:
            # Use the centroid of the first geometry
            synthetic_geom = shp_river.geometry.iloc[0].centroid
        else:
            # Create a default point geometry (this won't be used for actual routing)
            from shapely.geometry import Point
            synthetic_geom = Point(0, 0)
        
        synthetic_data[geom_col] = synthetic_geom
        
        # Create new GeoDataFrame with single row
        synthetic_gdf = gpd.GeoDataFrame([synthetic_data], crs=shp_river.crs)
        
        self.logger.info(f"Created synthetic river network: segment ID {synthetic_seg_id} (outlet)")
        
        return synthetic_gdf

    def create_network_topology_file(self):
        self.logger.info("Creating network topology file")
        
        river_network_path = self.config.get('RIVER_NETWORK_SHP_PATH')
        river_network_name = self.config.get('RIVER_NETWORK_SHP_NAME')

        if river_network_name == 'default':
            river_network_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_{self.config.get('DOMAIN_DEFINITION_METHOD','delineate')}.shp"
        
        if river_network_path == 'default':
            river_network_path = self.project_dir / 'shapefiles/river_network'
        else:
            river_network_path = Path(river_network_path)

        river_basin_path = self.config.get('RIVER_BASINS_PATH')
        river_basin_name = self.config.get('RIVER_BASINS_NAME')

        if river_basin_name == 'default':
            river_basin_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"

        if river_basin_path == 'default':
            river_basin_path = self.project_dir / 'shapefiles/river_basins'
        else:
            river_basin_path = Path(river_basin_path)        

        topology_name = self.config.get('SETTINGS_MIZU_TOPOLOGY')
        
        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)
        
        # Check if this is lumped domain with distributed routing
        is_lumped_to_distributed = (
            self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped' and 
            self.config.get('ROUTING_DELINEATION', 'river_network') == 'river_network'
        )
        
        if is_lumped_to_distributed:
            self.logger.info("Using delineated catchments for lumped-to-distributed routing")
            
            # Load the delineated catchments shapefile
            catchment_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.config['DOMAIN_NAME']}_catchment_delineated.shp"
            if not catchment_path.exists():
                raise FileNotFoundError(f"Delineated catchment shapefile not found: {catchment_path}")
            
            shp_catchments = gpd.read_file(catchment_path)
            self.logger.info(f"Loaded {len(shp_catchments)} delineated subcatchments")
            
            # Extract HRU data from delineated catchments
            hru_ids = shp_catchments['GRU_ID'].values.astype(int)
            
            # Check if we have a headwater basin (None values in river network)
            if self._check_if_headwater_basin(shp_river):
                # Create synthetic river network for headwater basin
                shp_river = self._create_synthetic_river_network(shp_river, hru_ids)
            
            # Use the delineated catchments as HRUs
            num_seg = len(shp_river)
            num_hru = len(shp_catchments)
            
            hru_to_seg_ids = shp_catchments['GRU_ID'].values.astype(int)  # Each GRU drains to segment with same ID
            
            # Convert fractional areas to actual areas (multiply by total basin area)
            total_basin_area = shp_basin[self.config.get('RIVER_BASIN_SHP_AREA')].sum()
            hru_areas = shp_catchments['avg_subbas'].values * total_basin_area
            
            # Store fractional areas for remapping
            self.subcatchment_weights = shp_catchments['avg_subbas'].values
            self.subcatchment_gru_ids = hru_ids
            
            self.logger.info(f"Created {num_hru} HRUs from delineated catchments")
            self.logger.info(f"Weight range: {self.subcatchment_weights.min():.4f} to {self.subcatchment_weights.max():.4f}")
            
        else:
            # Original logic for non-lumped cases
            closest_segment_id = self._find_closest_segment_to_pour_point(shp_river)
            
            if len(shp_basin) == 1:  # For lumped case
                shp_basin.loc[0, self.config.get('RIVER_BASIN_SHP_HRU_TO_SEG')] = closest_segment_id
                self.logger.info(f"Set single HRU to drain to closest segment: {closest_segment_id}")
            
            num_seg = len(shp_river)
            num_hru = len(shp_basin)
            
            hru_ids = shp_basin[self.config.get('RIVER_BASIN_SHP_RM_GRUID')].values.astype(int)
            hru_to_seg_ids = shp_basin[self.config.get('RIVER_BASIN_SHP_HRU_TO_SEG')].values.astype(int)
            hru_areas = shp_basin[self.config.get('RIVER_BASIN_SHP_AREA')].values.astype(float)
        
        # Ensure minimum segment length - now safe from None values
        length_col = self.config.get('RIVER_NETWORK_SHP_LENGTH')
        if length_col in shp_river.columns:
            # Convert None/null values to 0 first, then set minimum
            shp_river[length_col] = shp_river[length_col].fillna(0)
            shp_river.loc[shp_river[length_col] == 0, length_col] = 1
        
        # Ensure slope column has valid values
        slope_col = self.config.get('RIVER_NETWORK_SHP_SLOPE')
        if slope_col in shp_river.columns:
            shp_river[slope_col] = shp_river[slope_col].fillna(0.001)  # Default slope
            shp_river.loc[shp_river[slope_col] == 0, slope_col] = 0.001
        
        # Enforce outlets if specified
        if self.config.get('SETTINGS_MIZU_MAKE_OUTLET') != 'n/a':
            river_outlet_ids = [int(id) for id in self.config.get('SETTINGS_MIZU_MAKE_OUTLET').split(',')]
            seg_id_col = self.config.get('RIVER_NETWORK_SHP_SEGID')
            downseg_id_col = self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')
            
            for outlet_id in river_outlet_ids:
                if outlet_id in shp_river[seg_id_col].values:
                    shp_river.loc[shp_river[seg_id_col] == outlet_id, downseg_id_col] = 0
                else:
                    self.logger.warning(f"Outlet ID {outlet_id} not found in river network")
        
        # Create the netCDF file
        with nc4.Dataset(self.mizuroute_setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_seg, num_hru)
            
            # Create segment variables (now safe from None values)
            self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')].values.astype(int), 'Unique ID of each stream segment', '-')
            self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')].values.astype(int), 'ID of the downstream segment', '-')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_SLOPE')].values.astype(float), 'Segment slope', '-')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_LENGTH')].values.astype(float), 'Segment length', 'm')
            
            # Create HRU variables (using our computed values)
            self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', hru_ids, 'Unique hru ID', '-')
            self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', hru_to_seg_ids, 'ID of the stream segment to which the HRU discharges', '-')
            self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', hru_areas, 'HRU area', 'm^2')
        
        self.logger.info(f"Network topology file created at {self.mizuroute_setup_dir / topology_name}")
        
    def _find_closest_segment_to_pour_point(self, shp_river):
        """
        Find the river segment closest to the pour point.
        
        Args:
            shp_river: GeoDataFrame of river network
            
        Returns:
            int: Segment ID of closest segment to pour point
        """
        from pathlib import Path
        import numpy as np
        
        # Find pour point shapefile
        pour_point_dir = self.project_dir / 'shapefiles' / 'pour_point'
        pour_point_files = list(pour_point_dir.glob('*.shp'))
        
        if not pour_point_files:
            self.logger.error(f"No pour point shapefiles found in {pour_point_dir}")
            # Fallback: use outlet segment (downSegId == 0)
            outlet_mask = shp_river[self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.config.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                # Last resort: use first segment
                fallback_seg = shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg
        
        # Load first pour point file
        pour_point_file = pour_point_files[0]
        self.logger.info(f"Loading pour point from {pour_point_file}")
        
        try:
            shp_pour_point = gpd.read_file(pour_point_file)
            
            # Ensure both are in the same CRS
            if shp_river.crs != shp_pour_point.crs:
                shp_pour_point = shp_pour_point.to_crs(shp_river.crs)
            
            # Get pour point coordinates (assume first/only point)
            pour_point_geom = shp_pour_point.geometry.iloc[0]
            
            # Calculate distances from pour point to all river segments
            distances = shp_river.geometry.distance(pour_point_geom)
            
            # Find closest segment
            closest_idx = distances.idxmin()
            closest_segment_id = shp_river.loc[closest_idx, self.config.get('RIVER_NETWORK_SHP_SEGID')]
            
            self.logger.info(f"Closest segment to pour point: {closest_segment_id} (distance: {distances.iloc[closest_idx]:.1f} units)")
            
            return closest_segment_id
            
        except Exception as e:
            self.logger.error(f"Error finding closest segment: {str(e)}")
            # Fallback to outlet segment
            outlet_mask = shp_river[self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')] == 0
            if outlet_mask.any():
                outlet_seg = shp_river.loc[outlet_mask, self.config.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using outlet segment as fallback: {outlet_seg}")
                return outlet_seg
            else:
                fallback_seg = shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')].iloc[0]
                self.logger.warning(f"Using first segment as fallback: {fallback_seg}")
                return fallback_seg

    def create_equal_weight_remap_file(self):
        """Create remapping file with equal weights for all segments"""
        self.logger.info("Creating equal-weight remapping file")
        
        # Load topology to get segment information
        topology_file = self.mizuroute_setup_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY')
        with xr.open_dataset(topology_file) as topo:
            seg_ids = topo['segId'].values
            hru_ids = topo['hruId'].values  # Now we have multiple HRUs
        
        n_segments = len(seg_ids)
        n_hrus = len(hru_ids)
        equal_weight = 1.0 / n_hrus  # Equal weight for each HRU
        
        remap_name = self.config.get('SETTINGS_MIZU_REMAP')
        
        with nc4.Dataset(self.mizuroute_setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            # Set attributes
            ncid.setncattr('Author', "Created by SUMMA workflow scripts")
            ncid.setncattr('Purpose', 'Equal-weight remapping for lumped to distributed routing')
            
            # Create dimensions
            ncid.createDimension('hru', n_hrus)  # One entry per HRU
            ncid.createDimension('data', n_hrus)  # One data entry per HRU
            
            # Create variables
            # RN_hruId: The routing HRU IDs (1, 2, 3, ..., n_hrus)
            rn_hru = ncid.createVariable('RN_hruId', 'i4', ('hru',))
            rn_hru[:] = hru_ids
            rn_hru.long_name = 'River network HRU ID'
            
            # nOverlaps: Each HRU gets input from 1 SUMMA GRU
            noverlaps = ncid.createVariable('nOverlaps', 'i4', ('hru',))
            noverlaps[:] = [1] * n_hrus  # Each HRU has 1 overlap (with SUMMA GRU 1)
            noverlaps.long_name = 'Number of overlapping HM_HRUs for each RN_HRU'
            
            # HM_hruId: The SUMMA GRU ID (1) for each entry
            hm_hru = ncid.createVariable('HM_hruId', 'i4', ('data',))
            hm_hru[:] = [1] * n_hrus  # All entries point to SUMMA GRU 1
            hm_hru.long_name = 'ID of overlapping HM_HRUs'
            
            # weight: Equal weights for all HRUs
            weights = ncid.createVariable('weight', 'f8', ('data',))
            weights[:] = [equal_weight] * n_hrus
            weights.long_name = f'Equal areal weights ({equal_weight:.4f}) for all HRUs'
        
        self.logger.info(f"Equal-weight remapping file created with {n_hrus} HRUs, weight = {equal_weight:.4f}")

    def remap_summa_catchments_to_routing(self):
        self.logger.info("Remapping SUMMA catchments to routing catchments")
        if self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped' and self.config.get('ROUTING_DELINEATION') == 'river_network':
            self.logger.info("Area-weighted mapping for SUMMA catchments to routing catchments")
            self.create_area_weighted_remap_file()  # Changed from create_equal_weight_remap_file
            return

        hm_catchment_path = Path(self.config.get('CATCHMENT_PATH'))
        hm_catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if hm_catchment_name == 'default':
            hm_catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        rm_catchment_path = Path(self.config.get('RIVER_BASINS_PATH'))
        rm_catchment_name = self.config.get('RIVER_BASINS_NAME')
        
        intersect_path = Path(self.config.get('INTERSECT_ROUTING_PATH'))
        intersect_name = self.config.get('INTERSECT_ROUTING_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_routing_basins.shp'
        
        if intersect_path == 'default':
            intersect_path = self.project_dir / 'shapefiles/catchment_intersection' 
        else:
            intersect_path = Path(intersect_path)

        remap_name = self.config.get('SETTINGS_MIZU_REMAP')
        
        if hm_catchment_path == 'default':
            hm_catchment_path = self.project_dir / 'shapefiles/catchment' 
        else:
            hm_catchment_path = Path(hm_catchment_path)
            
        if rm_catchment_path == 'default':
            rm_catchment_path = self.project_dir / 'shapefiles/catchment' 
        else:
            rm_catchment_path = Path(rm_catchment_path)

        # Load shapefiles
        hm_shape = gpd.read_file(hm_catchment_path / hm_catchment_name)
        rm_shape = gpd.read_file(rm_catchment_path / rm_catchment_name)
        
        # Create intersection
        esmr_caller = esmr()
        hm_shape = hm_shape.to_crs('EPSG:6933')
        rm_shape = rm_shape.to_crs('EPSG:6933')
        intersected_shape = esmr.intersection_shp(esmr_caller, rm_shape, hm_shape)
        intersected_shape = intersected_shape.to_crs('EPSG:4326')
        intersected_shape.to_file(intersect_path / intersect_name)
        
        # Process variables for remapping file
        self._process_remap_variables(intersected_shape)
        
        # Create remapping netCDF file
        self._create_remap_file(intersected_shape, remap_name)
        
        self.logger.info(f"Remapping file created at {self.mizuroute_setup_dir / remap_name}")

    def create_control_file(self):
        self.logger.debug("Creating mizuRoute control file")
        
        control_name = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
        
        with open(self.mizuroute_setup_dir / control_name, 'w') as cf:
            self._write_control_file_header(cf)
            self._write_control_file_directories(cf)
            self._write_control_file_parameters(cf)
            self._write_control_file_simulation_controls(cf)
            self._write_control_file_topology(cf)
            self._write_control_file_runoff(cf)
            self._write_control_file_remapping(cf)
            self._write_control_file_miscellaneous(cf)
        
        self.logger.debug(f"mizuRoute control file created at {self.mizuroute_setup_dir / control_name}")

    def _set_topology_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a river network .nc file for mizuRoute routing')

    def _create_topology_dimensions(self, ncid, num_seg, num_hru):
        ncid.createDimension('seg', num_seg)
        ncid.createDimension('hru', num_hru)

    def create_fuse_control_file(self):
        """Create mizuRoute control file specifically for FUSE input"""
        self.logger.debug("Creating mizuRoute control file for FUSE")
        
        control_name = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
        
        with open(self.mizuroute_setup_dir / control_name, 'w') as cf:
            self._write_control_file_header(cf)
            self._write_fuse_control_file_directories(cf)  # FUSE-specific directories
            self._write_control_file_parameters(cf)
            self._write_control_file_simulation_controls(cf)
            self._write_control_file_topology(cf)
            self._write_fuse_control_file_runoff(cf)  # FUSE-specific runoff settings
            self._write_control_file_remapping(cf)
            self._write_control_file_miscellaneous(cf)

    def _write_fuse_control_file_directories(self, cf):
        """Write FUSE-specific directory paths for mizuRoute control"""
        experiment_output_fuse = self.config.get('EXPERIMENT_OUTPUT_FUSE')
        experiment_output_mizuroute = self.config.get('EXPERIMENT_OUTPUT_MIZUROUTE')

        if experiment_output_fuse == 'default':
            experiment_output_fuse = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'FUSE'
        else:
            experiment_output_fuse = Path(experiment_output_fuse)

        if experiment_output_mizuroute == 'default':
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.mizuroute_setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_fuse}/    ! Folder that contains runoff data from FUSE \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")


    def _write_fuse_control_file_runoff(self, cf):
        """Write FUSE-specific runoff file settings"""
        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {self.config.get('DOMAIN_NAME')}_{self.config.get('EXPERIMENT_ID')}_runs_def.nc    ! netCDF name for FUSE runoff \n")
        cf.write(f"<vname_qsim>            {self.config.get('SETTINGS_MIZU_ROUTING_VAR')}    ! Variable name for FUSE runoff \n")
        cf.write(f"<units_qsim>            {self.config.get('SETTINGS_MIZU_ROUTING_UNITS')}    ! Units of input runoff \n")
        cf.write(f"<dt_qsim>               {self.config.get('SETTINGS_MIZU_ROUTING_DT')}    ! Time interval of input runoff in seconds \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        cf.write("<dname_hruid>           gru     ! Dimension name for HM_HRU ID \n")
        cf.write("<vname_hruid>           gruId   ! Variable name for HM_HRU ID \n")
        cf.write("<calendar>              standard    ! Calendar of the nc file \n")


    def _write_control_file_simulation_controls(self, cf):
        """Enhanced simulation control writing with proper time handling"""
        # Get simulation dates from config
        sim_start = self.config.get('EXPERIMENT_TIME_START')
        sim_end = self.config.get('EXPERIMENT_TIME_END')
        
        # Ensure dates are in proper format
        from datetime import datetime
        if isinstance(sim_start, str) and len(sim_start) == 10:  # YYYY-MM-DD format
            sim_start = f"{sim_start} 00:00"
        if isinstance(sim_end, str) and len(sim_end) == 10:  # YYYY-MM-DD format
            sim_end = f"{sim_end} 23:00"
        
        cf.write("!\n! --- DEFINE SIMULATION CONTROLS \n")
        cf.write(f"<case_name>             {self.config.get('EXPERIMENT_ID')}    ! Simulation case name \n")
        cf.write(f"<sim_start>             {sim_start}    ! Time of simulation start \n")
        cf.write(f"<sim_end>               {sim_end}    ! Time of simulation end \n")
        cf.write(f"<route_opt>             {self.config.get('SETTINGS_MIZU_OUTPUT_VARS')}    ! Option for routing schemes \n")
        cf.write(f"<newFileFrequency>      {self.config.get('SETTINGS_MIZU_OUTPUT_FREQ')}    ! Frequency for new output files \n")

    def _create_topology_variables(self, ncid, shp_river, shp_basin):
        self._create_and_fill_nc_var(ncid, 'segId', 'int', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')].values.astype(int), 'Unique ID of each stream segment', '-')
        self._create_and_fill_nc_var(ncid, 'downSegId', 'int', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')].values.astype(int), 'ID of the downstream segment', '-')
        self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_SLOPE')].values.astype(float), 'Segment slope', '-')
        self._create_and_fill_nc_var(ncid, 'length', 'f8', 'seg', shp_river[self.config.get('RIVER_NETWORK_SHP_LENGTH')].values.astype(float), 'Segment length', 'm')
        self._create_and_fill_nc_var(ncid, 'hruId', 'int', 'hru', shp_basin[self.config.get('RIVER_BASIN_SHP_RM_GRUID')].values.astype(int), 'Unique hru ID', '-')
        self._create_and_fill_nc_var(ncid, 'hruToSegId', 'int', 'hru', shp_basin[self.config.get('RIVER_BASIN_SHP_HRU_TO_SEG')].values.astype(int), 'ID of the stream segment to which the HRU discharges', '-')
        self._create_and_fill_nc_var(ncid, 'area', 'f8', 'hru', shp_basin[self.config.get('RIVER_BASIN_SHP_AREA')].values.astype(float), 'HRU area', 'm^2')

    def _process_remap_variables(self, intersected_shape):
        int_rm_id = f"S_1_{self.config.get('RIVER_BASIN_SHP_RM_HRUID')}"
        int_hm_id = f"S_2_{self.config.get('CATCHMENT_SHP_GRUID')}"
        int_weight = 'AP1N'
        
        intersected_shape = intersected_shape.sort_values(by=[int_rm_id, int_hm_id])
        
        self.nc_rnhruid = intersected_shape.groupby(int_rm_id).agg({int_rm_id: pd.unique}).values.astype(int)
        self.nc_noverlaps = intersected_shape.groupby(int_rm_id).agg({int_hm_id: 'count'}).values.astype(int)
        
        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_hm_id: list}).values.tolist()
        self.nc_hmgruid = [item for sublist in multi_nested_list for item in sublist[0]]
        
        multi_nested_list = intersected_shape.groupby(int_rm_id).agg({int_weight: list}).values.tolist()
        self.nc_weight = [item for sublist in multi_nested_list for item in sublist[0]]

    def _create_remap_file(self, intersected_shape, remap_name):
        num_hru = len(intersected_shape[f"S_1_{self.config.get('RIVER_BASIN_SHP_RM_HRUID')}"].unique())
        num_data = len(intersected_shape)
        
        with nc4.Dataset(self.mizuroute_setup_dir / remap_name, 'w', format='NETCDF4') as ncid:
            self._set_remap_attributes(ncid)
            self._create_remap_dimensions(ncid, num_hru, num_data)
            self._create_remap_variables(ncid)

    def _set_remap_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a remapping .nc file for mizuRoute routing')

    def _create_remap_dimensions(self, ncid, num_hru, num_data):
        ncid.createDimension('hru', num_hru)
        ncid.createDimension('data', num_data)

    def _create_remap_variables(self, ncid):
        self._create_and_fill_nc_var(ncid, 'RN_hruId', 'int', 'hru', self.nc_rnhruid, 'River network HRU ID', '-')
        self._create_and_fill_nc_var(ncid, 'nOverlaps', 'int', 'hru', self.nc_noverlaps, 'Number of overlapping HM_HRUs for each RN_HRU', '-')
        self._create_and_fill_nc_var(ncid, 'HM_hruId', 'int', 'data', self.nc_hmgruid, 'ID of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')
        self._create_and_fill_nc_var(ncid, 'weight', 'f8', 'data', self.nc_weight, 'Areal weight of overlapping HM_HRUs. Note that SUMMA calls these GRUs', '-')

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, fill_data, long_name, units):
        ncvar = ncid.createVariable(var_name, var_type, (dim,))
        ncvar[:] = fill_data
        ncvar.long_name = long_name
        ncvar.units = units

    def _write_control_file_header(self, cf):
        cf.write("! mizuRoute control file generated by SUMMA public workflow scripts \n")

    def _write_control_file_directories(self, cf):
        experiment_output_summa = self.config.get('EXPERIMENT_OUTPUT_SUMMA')
        experiment_output_mizuroute = self.config.get('EXPERIMENT_OUTPUT_SUMMA')

        if experiment_output_summa == 'default':
            experiment_output_summa = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'SUMMA'
        else:
            experiment_output_summa = Path(experiment_output_summa)

        if experiment_output_mizuroute == 'default':
            experiment_output_mizuroute = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'mizuRoute'
        else:
            experiment_output_mizuroute = Path(experiment_output_mizuroute)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.mizuroute_setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {experiment_output_summa}/    ! Folder that contains runoff data from SUMMA \n")
        cf.write(f"<output_dir>            {experiment_output_mizuroute}/    ! Folder that will contain mizuRoute simulations \n")

    def _write_control_file_parameters(self, cf):
        cf.write("!\n! --- NAMELIST FILENAME \n")
        cf.write(f"<param_nml>             {self.config.get('SETTINGS_MIZU_PARAMETERS')}    ! Spatially constant parameter namelist (should be stored in the ancil_dir) \n")


    def _write_control_file_topology(self, cf):
        cf.write("!\n! --- DEFINE TOPOLOGY FILE \n")
        cf.write(f"<fname_ntopOld>         {self.config.get('SETTINGS_MIZU_TOPOLOGY')}    ! Name of input netCDF for River Network \n")
        cf.write("<dname_sseg>            seg    ! Dimension name for reach in river network netCDF \n")
        cf.write("<dname_nhru>            hru    ! Dimension name for RN_HRU in river network netCDF \n")
        cf.write("<seg_outlet>            -9999    ! Outlet reach ID at which to stop routing (i.e. use subset of full network). -9999 to use full network \n")
        cf.write("<varname_area>          area    ! Name of variable holding hru area \n")
        cf.write("<varname_length>        length    ! Name of variable holding segment length \n")
        cf.write("<varname_slope>         slope    ! Name of variable holding segment slope \n")
        cf.write("<varname_HRUid>         hruId    ! Name of variable holding HRU id \n")
        cf.write("<varname_hruSegId>      hruToSegId    ! Name of variable holding the stream segment below each HRU \n")
        cf.write("<varname_segId>         segId    ! Name of variable holding the ID of each stream segment \n")
        cf.write("<varname_downSegId>     downSegId    ! Name of variable holding the ID of the next downstream segment \n")

    def _write_control_file_remapping(self, cf):
        cf.write("!\n! --- DEFINE RUNOFF MAPPING FILE \n")
        remap_flag = self.config.get('SETTINGS_MIZU_NEEDS_REMAP', '') == True
        cf.write(f"<is_remap>              {'T' if remap_flag else 'F'}    ! Logical to indicate runoff needs to be remapped to RN_HRU. T or F \n")
        
        if remap_flag:
            cf.write(f"<fname_remap>           {self.config.get('SETTINGS_MIZU_REMAP')}    ! netCDF name of runoff remapping \n")
            cf.write("<vname_hruid_in_remap>  RN_hruId    ! Variable name for RN_HRUs \n")
            cf.write("<vname_weight>          weight    ! Variable name for areal weights of overlapping HM_HRUs \n")
            cf.write("<vname_qhruid>          HM_hruId    ! Variable name for HM_HRU ID \n")
            cf.write("<vname_num_qhru>        nOverlaps    ! Variable name for a numbers of overlapping HM_HRUs with RN_HRUs \n")
            cf.write("<dname_hru_remap>       hru    ! Dimension name for HM_HRU \n")
            cf.write("<dname_data_remap>      data    ! Dimension name for data \n")

    def _write_control_file_miscellaneous(self, cf):
        cf.write("!\n! --- MISCELLANEOUS \n")
        cf.write(f"<doesBasinRoute>        {self.config.get('SETTINGS_MIZU_WITHIN_BASIN')}    ! Hillslope routing options. 0 -> no (already routed by SUMMA), 1 -> use IRF \n")

    def _get_default_time(self, time_key, default_year):
        time_value = self.config.get(time_key)
        if time_value == 'default':
            raw_time = [
                    self.config.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
            year = raw_time[0] if default_year == 'start' else raw_time[1]
            return f"{year}-{'01-01 00:00' if default_year == 'start' else '12-31 23:00'}"
        return time_value

    def _pad_string(self, string, pad_to=20):
        return f"{string:{pad_to}}"
    

class MizuRouteRunner:
    """
    A class to run the mizuRoute model.

    This class handles the execution of the mizuRoute model, including setting up paths,
    running the model, and managing log files.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def fix_time_precision(self):
        """
        Fix model output time precision by rounding to nearest hour.
        This fixes compatibility issues with mizuRoute time matching.
        Now supports both SUMMA and FUSE outputs with proper time format detection.
        """
        # Determine which model's output to process
        models = self.config.get('HYDROLOGICAL_MODEL', '').split(',')
        active_models = [m.strip() for m in models]
        
        # For FUSE, check if it has already converted its output
        if 'FUSE' in active_models:
            self.logger.debug("Fixing FUSE time precision for mizuRoute compatibility")
            experiment_output_dir = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'FUSE'
            runoff_filename = f"{self.config.get('DOMAIN_NAME')}_{self.config.get('EXPERIMENT_ID')}_runs_def.nc"
        else:
            self.logger.info("Fixing SUMMA time precision for mizuRoute compatibility")
            experiment_output_summa = self.config.get('EXPERIMENT_OUTPUT_SUMMA')
            if experiment_output_summa == 'default':
                experiment_output_dir = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'SUMMA'
            else:
                experiment_output_dir = Path(experiment_output_summa)
            runoff_filename = f"{self.config.get('EXPERIMENT_ID')}_timestep.nc"
        
        runoff_filepath = experiment_output_dir / runoff_filename
        
        if not runoff_filepath.exists():
            self.logger.error(f"Model output file not found: {runoff_filepath}")
            return
        
        try:
            import xarray as xr
            import os
            
            self.logger.debug(f"Processing {runoff_filepath}")
            
            # Open dataset and examine time format
            ds = xr.open_dataset(runoff_filepath, decode_times=False)
            
            # Detect the time format by examining attributes and values
            time_attrs = ds.time.attrs
            time_values = ds.time.values
            
            self.logger.debug(f"Time units: {time_attrs.get('units', 'No units specified')}")
            self.logger.debug(f"Time range: {time_values.min()} to {time_values.max()}")
            
            # Check if time precision fix is needed and determine format
            needs_fix = False
            time_format_detected = None
            
            if 'units' in time_attrs:
                units_str = time_attrs['units'].lower()
                
                if 'since 1990-01-01' in units_str:
                    # SUMMA-style format: seconds since 1990-01-01
                    time_format_detected = 'summa_seconds_1990'
                    first_time = pd.to_datetime(time_values[0], unit='s', origin='1990-01-01')
                    rounded_time = first_time.round('h')
                    needs_fix = (first_time != rounded_time)
                    
                elif 'since' in units_str:
                    # Other reference time format - extract the reference
                    import re
                    since_match = re.search(r'since\s+([0-9-]+(?:\s+[0-9:]+)?)', units_str)
                    if since_match:
                        ref_time_str = since_match.group(1).strip()
                        time_format_detected = f'generic_since_{ref_time_str}'
                        
                        # Determine the unit (seconds, hours, days)
                        if 'second' in units_str:
                            first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                            time_unit = 's'
                        elif 'hour' in units_str:
                            first_time = pd.to_datetime(time_values[0], unit='h', origin=ref_time_str)
                            time_unit = 'h'
                        elif 'day' in units_str:
                            first_time = pd.to_datetime(time_values[0], unit='D', origin=ref_time_str)
                            time_unit = 'D'
                        else:
                            # Default to seconds
                            first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                            time_unit = 's'
                        
                        rounded_time = first_time.round('h')
                        needs_fix = (first_time != rounded_time)
                        
                else:
                    # No 'since' found, might be already in datetime format
                    time_format_detected = 'unknown'
                    try:
                        # Try to interpret as datetime directly
                        ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                        first_time = pd.Timestamp(ds_decoded.time.values[0])
                        rounded_time = first_time.round('h')
                        needs_fix = (first_time != rounded_time)
                        ds_decoded.close()
                        time_format_detected = 'datetime64'
                    except:
                        self.logger.warning("Could not determine time format - skipping time precision fix")
                        ds.close()
                        return
            else:
                # No units attribute - try to decode directly
                try:
                    ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                    first_time = pd.Timestamp(ds_decoded.time.values[0])
                    rounded_time = first_time.round('h')
                    needs_fix = (first_time != rounded_time)
                    ds_decoded.close()
                    time_format_detected = 'datetime64'
                except:
                    self.logger.warning("No time units and cannot decode times - skipping time precision fix")
                    ds.close()
                    return
            
            self.logger.debug(f"Detected time format: {time_format_detected}")
            self.logger.debug(f"Needs time precision fix: {needs_fix}")
            
            if not needs_fix:
                self.logger.debug("Time precision is already correct")
                ds.close()
                return
            
            # Apply the appropriate fix based on detected format
            if time_format_detected == 'summa_seconds_1990':
                # Original SUMMA logic
                self.logger.info("Applying SUMMA-style time precision fix")
                time_stamps = pd.to_datetime(time_values, unit='s', origin='1990-01-01')
                rounded_stamps = time_stamps.round('h')
                reference = pd.Timestamp('1990-01-01')
                rounded_seconds = (rounded_stamps - reference).total_seconds().values
                
                ds = ds.assign_coords(time=rounded_seconds)
                ds.time.attrs.clear()
                ds.time.attrs['units'] = 'seconds since 1990-01-01 00:00:00'
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'
                
            elif time_format_detected.startswith('generic_since_'):
                # Generic 'since' format
                self.logger.info(f"Applying generic time precision fix for format: {time_format_detected}")
                ref_time_str = time_format_detected.split('generic_since_')[1]
                
                time_stamps = pd.to_datetime(time_values, unit=time_unit, origin=ref_time_str)
                rounded_stamps = time_stamps.round('h')
                reference = pd.Timestamp(ref_time_str)
                
                if time_unit == 's':
                    rounded_values = (rounded_stamps - reference).total_seconds().values
                elif time_unit == 'h':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(hours=1)
                elif time_unit == 'D':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(days=1)
                
                ds = ds.assign_coords(time=rounded_values)
                ds.time.attrs.clear()
                ds.time.attrs['units'] = f"{time_unit} since {ref_time_str}"
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'
                
            elif time_format_detected == 'datetime64':
                # Already in datetime format, just round
                self.logger.info("Applying datetime64 time precision fix")
                ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                time_stamps = pd.to_datetime(ds_decoded.time.values)
                rounded_stamps = time_stamps.round('h')
                
                # Keep original format but with rounded times
                ds = ds_decoded.assign_coords(time=rounded_stamps)
                ds_decoded.close()
            
            # Save the corrected file
            ds.load()
            ds.close()
            
            os.chmod(runoff_filepath, 0o664)
            ds.to_netcdf(runoff_filepath, format='NETCDF4')
            self.logger.info("Time precision fixed successfully")
            
        except Exception as e:
            self.logger.error(f"Error fixing time precision: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.debug("Starting mizuRoute run")
        self.fix_time_precision()
        # Set up paths and filenames
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        
        if mizu_path == 'default':
            mizu_path = self.root_path / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)

        mizu_exe = self.config.get('EXE_NAME_MIZUROUTE')
        settings_path = self._get_config_path('SETTINGS_MIZU_PATH', 'settings/mizuRoute/')
        control_file = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        mizu_log_path = self._get_config_path('EXPERIMENT_LOG_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/mizuRoute_logs/")
        mizu_log_name = "mizuRoute_log.txt"
        
        mizu_out_path = self._get_config_path('EXPERIMENT_OUTPUT_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            backup_path = mizu_out_path / "run_settings"
            self._backup_settings(settings_path, backup_path)

        # Run mizuRoute
        os.makedirs(mizu_log_path, exist_ok=True)
        mizu_command = f"{mizu_path / mizu_exe} {settings_path / control_file}"
        self.logger.debug(f'Running mizuRoute with comman: {mizu_command}')
        try:
            with open(mizu_log_path / mizu_log_name, 'w') as log_file:
                subprocess.run(mizu_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.debug("mizuRoute run completed successfully")
            return mizu_out_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"mizuRoute run failed with error: {e}")
            raise

    def _get_config_path(self, config_key: str, default_suffix: str) -> Path:
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    def _backup_settings(self, source_path: Path, backup_path: Path):
        backup_path.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -R {source_path}/. {backup_path}")
        self.logger.info(f"Settings backed up to {backup_path}")