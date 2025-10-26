"""
NextGen (ngen) Framework Utilities for CONFLUENCE

This module provides preprocessing, execution, and postprocessing utilities
for the NOAA NextGen Water Resources Modeling Framework within CONFLUENCE.

Classes:
    NgenPreProcessor: Handles spatial preprocessing and configuration generation
    NgenRunner: Manages model execution
    NgenPostprocessor: Processes model outputs

Author: CONFLUENCE Development Team
Date: 2025
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from shutil import copyfile
import netCDF4 as nc4

sys.path.append(str(Path(__file__).resolve().parent.parent))


class NgenPreProcessor:
    """
    Preprocessor for NextGen Framework.
    
    Handles conversion of CONFLUENCE data to ngen-compatible formats including:
    - Catchment geometry (geopackage)
    - Nexus points (GeoJSON)
    - Forcing data (NetCDF)
    - Model configurations (CFE, PET, NOAH-OWP)
    - Realization configuration (JSON)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the NextGen preprocessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
        """
        self.config = config
        self.logger = logger
        
        # Directories
        self.project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}"
        self.ngen_setup_dir = self.project_dir / "settings" / "ngen"
        self.forcing_dir = self.project_dir / "forcing" / "ngen_input"
        
        # Shapefiles
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')
        if self.catchment_name == 'default':
            self.catchment_name = f"{config.get('DOMAIN_NAME')}_HRUs_{config.get('DOMAIN_DISCRETIZATION', 'GRUs')}.shp"
        
        self.river_network_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        self.river_network_name = config.get('RIVER_NETWORK_SHP_NAME', 'default')
        if self.river_network_name == 'default':
            self.river_network_name = f"{config.get('DOMAIN_NAME')}_riverNetwork_delineate.shp"
        
        # Forcing
        self.basin_forcing_dir = self.project_dir / "forcing" / "basin_averaged_data"
        
        # IDs
        self.hru_id_col = config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
        self.domain_name = config.get('DOMAIN_NAME')
        
    def _get_default_path(self, key: str, default_subpath: str) -> Path:
        """Get path from config or use default."""
        path_value = self.config.get(key, 'default')
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)
    
    def _copy_noah_parameter_tables(self):
        """
        Copy Noah-OWP parameter tables from base settings to domain settings.
        
        Copies GENPARM.TBL, MPTABLE.TBL, and SOILPARM.TBL from:
            CONFLUENCE_CODE_DIR/0_base_settings/NOAH/parameters/
        To:
            domain_dir/settings/ngen/NOAH/parameters/
        """
        self.logger.info("Copying Noah-OWP parameter tables")
        
        # Get path to CONFLUENCE code directory (parent of CONFLUENCE_DATA_DIR)
        confluence_data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        confluence_code_dir = confluence_data_dir.parent / 'CONFLUENCE'
        
        # Source directory for Noah parameter tables
        source_param_dir = confluence_code_dir / '0_base_settings' / 'NOAH' / 'parameters'
        
        # Destination directory
        dest_param_dir = self.ngen_setup_dir / 'NOAH' / 'parameters'
        
        # Parameter table files to copy
        param_files = ['GENPARM.TBL', 'MPTABLE.TBL', 'SOILPARM.TBL']
        
        for param_file in param_files:
            source_file = source_param_dir / param_file
            dest_file = dest_param_dir / param_file
            
            if source_file.exists():
                copyfile(source_file, dest_file)
                self.logger.info(f"Copied {param_file} to {dest_param_dir}")
            else:
                self.logger.warning(f"Parameter file not found: {source_file}")

    
    def run_preprocessing(self):
        """
        Execute complete ngen preprocessing workflow.
        
        Steps:
        1. Create ngen directory structure
        2. Copy Noah-OWP parameter tables from base settings
        3. Generate nexus GeoJSON from river network
        4. Create catchment geopackage
        5. Prepare forcing data in ngen format
        6. Generate model-specific configs (CFE, PET, NOAH)
        7. Generate realization config JSON
        """
        self.logger.info("Starting NextGen preprocessing")
        
        # Create directory structure
        self.ngen_setup_dir.mkdir(parents=True, exist_ok=True)
        self.forcing_dir.mkdir(parents=True, exist_ok=True)
        (self.ngen_setup_dir / "CFE").mkdir(exist_ok=True)
        (self.ngen_setup_dir / "PET").mkdir(exist_ok=True)
        (self.ngen_setup_dir / "NOAH").mkdir(exist_ok=True)
        (self.ngen_setup_dir / "NOAH" / "parameters").mkdir(exist_ok=True)
        
        # Copy Noah-OWP parameter tables from base settings
        self._copy_noah_parameter_tables()
        
        # Generate spatial data
        nexus_file = self.create_nexus_geojson()
        catchment_file = self.create_catchment_geopackage()
        
        # Prepare forcing
        forcing_file = self.prepare_forcing_data()
        
        # Generate model configs
        self.generate_model_configs()
        
        # Generate realization config
        self.generate_realization_config(catchment_file, nexus_file, forcing_file)
        
        self.logger.info("NextGen preprocessing completed")
        
    def create_nexus_geojson(self) -> Path:
        """
        Create nexus GeoJSON from river network topology.
        
        Nexus points represent junctions and outlets in the stream network.
        Each catchment flows to a nexus point.
        
        Returns:
            Path to nexus GeoJSON file
        """
        self.logger.info("Creating nexus GeoJSON")
        
        # Load river network
        river_network_file = self.river_network_path / self.river_network_name
        if not river_network_file.exists():
            self.logger.warning(f"River network not found: {river_network_file}")
            # For lumped catchments, create a single outlet nexus
            return self._create_simple_nexus()
        
        river_gdf = gpd.read_file(river_network_file)
        
        # Get segment ID columns
        seg_id_col = self.config.get('RIVER_NETWORK_SHP_SEGID', 'LINKNO')
        downstream_col = self.config.get('RIVER_NETWORK_SHP_DOWNSEGID', 'DSLINKNO')
        
        # Create nexus points at segment endpoints
        nexus_features = []
        
        for idx, row in river_gdf.iterrows():
            seg_id = row[seg_id_col]
            downstream_id = row[downstream_col]
            
            # Get endpoint of segment
            geom = row.geometry
            if geom.geom_type == 'LineString':
                endpoint = geom.coords[-1]  # Last point
            else:
                # For Point geometries (lumped case)
                endpoint = (geom.x, geom.y)
            
            # Create nexus ID
            nexus_id = f"nex-{int(seg_id)}"
            
            if downstream_id == 0 or pd.isna(downstream_id):
                nexus_type = "poi"
                toid = ""                                  # terminal outlet
            else:
                nexus_type = "nexus"
                toid = f"wb-{int(downstream_id)}"          # nexus -> downstream catchment


            
            feature = {
                "type": "Feature",
                "id": nexus_id,
                "properties": {
                    "toid": toid,
                    "hl_id": None,
                    "hl_uri": "NA",
                    "type": nexus_type
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": list(endpoint)
                }
            }
            nexus_features.append(feature)
        
        # Create GeoJSON
        nexus_geojson = {
            "type": "FeatureCollection",
            "name": "nexus",
            "xy_coordinate_resolution": 1e-06,
            "features": nexus_features
        }
        
        # Save to file
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump(nexus_geojson, f, indent=2)
        
        self.logger.info(f"Created nexus file with {len(nexus_features)} nexus points: {nexus_file}")
        return nexus_file
    
    def _create_simple_nexus(self) -> Path:
        """Create a simple single-nexus for lumped catchments."""
        self.logger.info("Creating simple outlet nexus for lumped catchment")
        
        # Load catchment to get centroid
        catchment_file = self.catchment_path / self.catchment_name
        catchment_gdf = gpd.read_file(catchment_file)
        
        # Get catchment centroid (in WGS84)
        catchment_wgs84 = catchment_gdf.to_crs("EPSG:4326")
        centroid = catchment_wgs84.geometry.centroid.iloc[0]
        
        # Get catchment ID
        catchment_id = str(catchment_gdf[self.hru_id_col].iloc[0])
        
        nexus_geojson = {
            "type": "FeatureCollection",
            "name": "nexus",
            "xy_coordinate_resolution": 1e-06,
            "features": [{
                "type": "Feature",
                "id": f"nex-{catchment_id}",
                "properties": {
                    "toid": "",  # Terminal outlet - empty toid breaks the cycle
                    "hl_id": None,
                    "hl_uri": "NA",
                    "type": "poi"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [centroid.x, centroid.y]
                }
            }]
        }
        
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        with open(nexus_file, 'w') as f:
            json.dump(nexus_geojson, f, indent=2)
        
        self.logger.info(f"Created simple nexus file: {nexus_file}")
        return nexus_file
    
    def create_catchment_geopackage(self) -> Path:
        """
        Create ngen-compatible geopackage from CONFLUENCE catchment shapefile.
        
        The geopackage must contain a 'divides' layer with required attributes:
        - divide_id: Catchment identifier
        - toid: ID of downstream catchment (or nexus)
        - areasqkm: Catchment area
        - geometry: Polygon geometry
        
        Returns:
            Path to catchment geopackage
        """
        self.logger.info("Creating catchment geopackage")
        
        # Load catchment shapefile
        catchment_file = self.catchment_path / self.catchment_name
        catchment_gdf = gpd.read_file(catchment_file)
        
        # Create divides layer
        divides_gdf = catchment_gdf.copy()
        
        # Map to ngen schema
        divides_gdf['divide_id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'cat-{x}')
        divides_gdf['id'] = divides_gdf[self.hru_id_col].apply(lambda x: f'wb-{x}')  # Waterbody ID
        
        # Determine downstream connections
        # For lumped catchment, connect to corresponding nexus
        divides_gdf['toid'] = divides_gdf[self.hru_id_col].apply(lambda x: f'nex-{x}')
        
        # Add type
        divides_gdf['type'] = 'network'  # Changed from 'land' to 'network'
        
        # Calculate area in km²
        if 'areasqkm' not in divides_gdf.columns:
            # Convert to equal-area projection for area calculation
            utm_crs = divides_gdf.estimate_utm_crs()
            divides_utm = divides_gdf.to_crs(utm_crs)
            divides_gdf['areasqkm'] = divides_utm.geometry.area / 1e6
        
        # Select required columns
        required_cols = ['divide_id', 'toid', 'type', 'id', 'areasqkm', 'geometry']
        optional_cols = ['ds_id', 'lengthkm', 'tot_drainage_areasqkm', 'has_flowline']
        
        # Add optional columns with defaults if missing
        for col in optional_cols:
            if col not in divides_gdf.columns:
                if col == 'ds_id':
                    divides_gdf[col] = 0.0
                elif col == 'lengthkm':
                    divides_gdf[col] = 0.0
                elif col == 'tot_drainage_areasqkm':
                    divides_gdf[col] = divides_gdf['areasqkm']
                elif col == 'has_flowline':
                    divides_gdf[col] = False
        
        output_cols = required_cols + [c for c in optional_cols if c in divides_gdf.columns]
        divides_gdf = divides_gdf[output_cols]
        
        # Ensure proper CRS (NAD83 Conus Albers - EPSG:5070)
        if divides_gdf.crs != "EPSG:5070":
            divides_gdf = divides_gdf.to_crs("EPSG:5070")
        
        # Remove the column since the index will carry 'id'
        divides_gdf = divides_gdf.drop(columns=['id'])
        
        # Set index to the divide_id for proper feature identification
        divides_gdf.index = divides_gdf['divide_id']
        divides_gdf.index.name = 'id'
        
        # Save as geopackage
        gpkg_file = self.ngen_setup_dir / f"{self.domain_name}_catchments.gpkg"
        divides_gdf.to_file(gpkg_file, layer='divides', driver='GPKG')
        
        self.logger.info(f"Created catchment geopackage with {len(divides_gdf)} catchments: {gpkg_file}")
        return gpkg_file
    
    def prepare_forcing_data(self) -> Path:
        """
        Convert CONFLUENCE basin-averaged ERA5 forcing to ngen format.
        
        Processes:
        1. Load all monthly forcing files
        2. Merge across time
        3. Map variable names (ERA5 → ngen)
        4. Reorganize dimensions (hru, time) → (catchment-id, time)
        5. Add catchment IDs
        
        Returns:
            Path to ngen forcing NetCDF file
        """
        self.logger.info("Preparing forcing data for ngen")
        
        # Load catchment IDs - must match divide_id format in geopackage (cat-X)
        catchment_file = self.catchment_path / self.catchment_name
        catchment_gdf = gpd.read_file(catchment_file)
        catchment_ids = [f"cat-{x}" for x in catchment_gdf[self.hru_id_col].astype(str).tolist()]
        n_catchments = len(catchment_ids)
        
        self.logger.info(f"Processing forcing for {n_catchments} catchments")
        
        # Get forcing files
        forcing_files = sorted(self.basin_forcing_dir.glob("*.nc"))
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {self.basin_forcing_dir}")
        
        self.logger.info(f"Found {len(forcing_files)} forcing files")
        
        # Open all files and concatenate
        datasets = []
        for f in forcing_files:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        
        # Concatenate along time dimension
        forcing_data = xr.concat(datasets, dim='time')
        
        # Get time bounds from config - use defaults if not specified or set to 'default'
        sim_start = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        
        # Handle 'default' string
        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'
        
        # Always subset to simulation period
        start_time = pd.to_datetime(sim_start)
        end_time = pd.to_datetime(sim_end)
        
        # Convert forcing time to datetime if needed
        time_values = pd.to_datetime(forcing_data.time.values)
        forcing_data['time'] = time_values
        
        # Select time slice for simulation period
        forcing_data = forcing_data.sel(time=slice(start_time, end_time))
            
        self.logger.info(f"Forcing time range: {forcing_data.time.values[0]} to {forcing_data.time.values[-1]}")
        
        # Create ngen-formatted dataset
        ngen_ds = self._create_ngen_forcing_dataset(forcing_data, catchment_ids)
        
        # Save to file with NETCDF4 format (supports native string type)
        output_file = self.forcing_dir / "forcing.nc"
        ngen_ds.to_netcdf(output_file, format='NETCDF4')
        
        # Close datasets
        forcing_data.close()
        for ds in datasets:
            ds.close()
        
        self.logger.info(f"Created ngen forcing file: {output_file}")
        return output_file
    
    def _create_ngen_forcing_dataset(self, forcing_data: xr.Dataset, catchment_ids: List[str]) -> xr.Dataset:
        """
        Create ngen-formatted forcing dataset with proper variable mapping.
        
        Args:
            forcing_data: Source forcing dataset (ERA5)
            catchment_ids: List of catchment identifiers
            
        Returns:
            ngen-formatted xarray Dataset
        """
        n_catchments = len(catchment_ids)
        n_times = len(forcing_data.time)
        
        # Convert time to nanoseconds since epoch (ngen format - matching working example)
        time_values = forcing_data.time.values
        
        # Ensure we have datetime64 objects
        if not np.issubdtype(time_values.dtype, np.datetime64):
            # Try to decode using xarray's built-in time decoding
            if 'units' in forcing_data.time.attrs:
                time_values = xr.decode_cf(forcing_data).time.values
            else:
                # Fallback: assume hours since 1900-01-01 (common ERA5 format)
                time_values = pd.to_datetime(time_values, unit='h', origin='1900-01-01').values
        
        # Verify we have datetime64 now
        if not np.issubdtype(time_values.dtype, np.datetime64):
            raise ValueError(f"Could not convert time to datetime64, got dtype: {time_values.dtype}")
        
        # Convert to nanoseconds since 1970-01-01 (Unix epoch)
        time_ns = ((time_values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'ns')).astype(np.int64)
        
        # Sanity check - values should be positive and reasonable (between 1970 and 2100)
        min_ns = 0  # 1970-01-01
        max_ns = 4102444800 * 1e9  # 2100-01-01 in nanoseconds
        if np.any(time_ns < min_ns) or np.any(time_ns > max_ns):
            raise ValueError(f"Time values out of reasonable range. Got min={time_ns.min()}, max={time_ns.max()}. "
                           f"Expected between {min_ns} and {max_ns}")
        
        # Create coordinate arrays
        catchment_coord = np.arange(n_catchments)
        time_coord = np.arange(n_times)  # Simple indices for time dimension
        
        # Initialize dataset with dimensions matching working example
        ngen_ds = xr.Dataset(
            coords={
                'catchment-id': ('catchment-id', catchment_coord),
                'time': ('time', time_coord),
                'str_dim': ('str_dim', np.array([1]))  # Required dimension for ngen
            }
        )
        
        # Add catchment IDs as native NetCDF4 string type
        ngen_ds['ids'] = xr.DataArray(
            np.array(catchment_ids, dtype=object),
            dims=['catchment-id'],
            attrs={'long_name': 'catchment identifiers'}
        )
        
        # Add Time variable (capital T) with nanoseconds for each catchment-time pair
        # Replicate time values for each catchment
        time_data = np.tile(time_ns, (n_catchments, 1)).astype(np.float64)
        ngen_ds['Time'] = xr.DataArray(
            time_data,
            dims=['catchment-id', 'time'],
            attrs={'units': 'ns'}
        )
        
        # Map and add forcing variables
        # ERA5 → ngen variable mapping
        var_mapping = {
            'pptrate': 'precip_rate',
            'airtemp': 'TMP_2maboveground',
            'spechum': 'SPFH_2maboveground',
            'airpres': 'PRES_surface',
            'SWRadAtm': 'DSWRF_surface',
            'LWRadAtm': 'DLWRF_surface'
        }
        
        # Add variables as float32 to match ngen requirements
        for era5_var, ngen_var in var_mapping.items():
            if era5_var in forcing_data:
                # Get data and transpose to (catchment, time)
                data = forcing_data[era5_var].values.T  # (time, hru) → (hru, time)
                
                # Replicate for multiple catchments if needed
                if data.shape[0] == 1 and n_catchments > 1:
                    data = np.tile(data, (n_catchments, 1))
                
                ngen_ds[ngen_var] = xr.DataArray(
                    data.astype(np.float32),  # Ensure float32
                    dims=['catchment-id', 'time']
                )
        
        # Handle wind components
        # ERA5 provides windspd; ngen needs UGRD and VGRD
        # Approximate: assume wind from west, so UGRD = windspd, VGRD = 0
        if 'windspd' in forcing_data:
            windspd_data = forcing_data['windspd'].values.T
            
            if windspd_data.shape[0] == 1 and n_catchments > 1:
                windspd_data = np.tile(windspd_data, (n_catchments, 1))
            
            # Approximate split (could be improved with actual U/V components)
            ngen_ds['UGRD_10maboveground'] = xr.DataArray(
                (windspd_data * 0.707).astype(np.float32),  # Ensure float32
                dims=['catchment-id', 'time']
            )
            ngen_ds['VGRD_10maboveground'] = xr.DataArray(
                (windspd_data * 0.707).astype(np.float32),  # Ensure float32
                dims=['catchment-id', 'time']
            )
        
        return ngen_ds
    
    def generate_model_configs(self):
        """
        Generate model-specific configuration files for each catchment.
        
        Creates:
        - CFE (Conceptual Functional Equivalent) configs
        - PET (Potential Evapotranspiration) configs  
        - NOAH-OWP (Noah-Owens-Pries) configs
        """
        self.logger.info("Generating model configuration files")
        
        # Load catchment data for parameters
        catchment_file = self.catchment_path / self.catchment_name
        catchment_gdf = gpd.read_file(catchment_file)
        
        # Store the CRS for use in config generation
        self.catchment_crs = catchment_gdf.crs
        
        for idx, catchment in catchment_gdf.iterrows():
            cat_id = str(catchment[self.hru_id_col])
            
            # Generate configs
            self._generate_cfe_config(cat_id, catchment)
            self._generate_pet_config(cat_id, catchment)
            self._generate_noah_config(cat_id, catchment)
        
        self.logger.info(f"Generated configs for {len(catchment_gdf)} catchments")
    
    def _generate_cfe_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """Generate CFE model configuration file."""
        
        # Get catchment-specific parameters (or use defaults)
        # In a full implementation, these would come from soil/vegetation data
        config_text = f"""forcing_file=BMI
surface_partitioning_scheme=Schaake
soil_params.depth=2.0[m]
soil_params.b=5.0[]
soil_params.satdk=5.0e-06[m s-1]
soil_params.satpsi=0.141[m]
soil_params.slop=0.03[m/m]
soil_params.smcmax=0.439[m/m]
soil_params.wltsmc=0.047[m/m]
soil_params.expon=1.0[]
soil_params.expon_secondary=1.0[]
refkdt=1.0
max_gw_storage=0.0129[m]
Cgw=1.8e-05[m h-1]
expon=7.0[]
gw_storage=0.35[m/m]
alpha_fc=0.33
soil_storage=0.35[m/m]
K_nash=0.03[]
K_lf=0.01[]
nash_storage=0.0,0.0
num_timesteps=1
verbosity=1
DEBUG=0
giuh_ordinates=0.65,0.35
"""
        
        config_file = self.ngen_setup_dir / "CFE" / f"cat-{catchment_id}_bmi_config_cfe_pass.txt"
        with open(config_file, 'w') as f:
            f.write(config_text)
    
    def _generate_pet_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """Generate PET model configuration file."""
        
        # Get catchment centroid for lat/lon
        centroid = catchment_row.geometry.centroid
        
        # Convert to WGS84 if needed
        if self.catchment_crs != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid
        
        config_text = f"""forcing_file=BMI
wind_speed_measurement_height_m=10.0
humidity_measurement_height_m=2.0
vegetation_height_m=0.12
zero_plane_displacement_height_m=0.0003
momentum_transfer_roughness_length_m=0.0
heat_transfer_roughness_length_m=0.0
surface_longwave_emissivity=1.0
surface_shortwave_albedo=0.23
latitude_degrees={centroid.y}
longitude_degrees={centroid.x}
site_elevation_m=100.0
time_step_size_s=3600
num_timesteps=1
"""
        
        config_file = self.ngen_setup_dir / "PET" / f"cat-{catchment_id}_pet_config.txt"
        with open(config_file, 'w') as f:
            f.write(config_text)
    
    def _generate_noah_config(self, catchment_id: str, catchment_row: gpd.GeoSeries):
        """
        Generate NOAH-OWP model configuration file (.input file).
        
        Creates a Fortran namelist file with all required sections:
        - timing: simulation period and file paths
        - parameters: paths to parameter tables
        - location: catchment lat/lon and terrain
        - forcing: forcing data settings
        - model_options: Noah-OWP model configuration
        - structure: soil/snow/vegetation structure
        - initial_values: initial soil moisture, snow, water table
        """
        # Get catchment centroid for lat/lon
        centroid = catchment_row.geometry.centroid
        
        # Convert to WGS84 if needed
        if self.catchment_crs != "EPSG:4326":
            geom_wgs84 = gpd.GeoSeries([catchment_row.geometry], crs=self.catchment_crs)
            geom_wgs84 = geom_wgs84.to_crs("EPSG:4326")
            centroid = geom_wgs84.iloc[0].centroid
        
        # Get simulation timing from config
        start_time = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        end_time = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        
        # Convert to Noah-OWP format (YYYYMMDDhhmm)
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        start_str = start_dt.strftime('%Y%m%d%H%M')
        end_str = end_dt.strftime('%Y%m%d%H%M')
        
        # Absolute path to parameters directory
        # Noah-OWP requires absolute path since ngen runs from its build directory
        param_dir = str((self.ngen_setup_dir / "NOAH" / "parameters").resolve()) + "/"
        
        # Create Noah-OWP configuration file
        config_text = f"""&timing
  dt                 = 3600.0
  startdate          = "{start_str}"
  enddate            = "{end_str}"
  forcing_filename   = "BMI"
  output_filename    = "out_cat-{catchment_id}.csv"
/

&parameters
  parameter_dir      = "{param_dir}"
  general_table      = "GENPARM.TBL"
  soil_table         = "SOILPARM.TBL"
  noahowp_table      = "MPTABLE.TBL"
  soil_class_name    = "STAS"
  veg_class_name     = "MODIFIED_IGBP_MODIS_NOAH"
/

&location
  lat                = {centroid.y}
  lon                = {centroid.x}
  terrain_slope      = 0.0
  azimuth            = 0.0
/

&forcing
  ZREF               = 10.0
  rain_snow_thresh   = 1.0
/

&model_options
  precip_phase_option               = 1
  snow_albedo_option                = 1
  dynamic_veg_option                = 4
  runoff_option                     = 3
  drainage_option                   = 8
  frozen_soil_option                = 1
  dynamic_vic_option                = 1
  radiative_transfer_option         = 3
  sfc_drag_coeff_option             = 1
  canopy_stom_resist_option         = 1
  crop_model_option                 = 0
  snowsoil_temp_time_option         = 3
  soil_temp_boundary_option         = 2
  supercooled_water_option          = 1
  stomatal_resistance_option        = 1
  evap_srfc_resistance_option       = 4
  subsurface_option                 = 2
/

&structure
 isltyp           = 3
 nsoil            = 4
 nsnow            = 3
 nveg             = 20
 vegtyp           = 10
 croptype         = 0
 sfctyp           = 1
 soilcolor        = 4
/

&initial_values
 dzsnso    =  0.0,  0.0,  0.0,  0.1,  0.3,  0.6,  1.0
 sice      =  0.0,  0.0,  0.0,  0.0
 sh2o      =  0.3,  0.3,  0.3,  0.3
 zwt       =  -2.0
/
"""
        
        config_file = self.ngen_setup_dir / "NOAH" / f"cat-{catchment_id}.input"
        with open(config_file, 'w') as f:
            f.write(config_text)

    
    def generate_realization_config(self, catchment_file: Path, nexus_file: Path, forcing_file: Path):
        """
        Generate ngen realization configuration JSON.
        
        This is the main configuration file that ngen uses to:
        - Define model formulations for each catchment
        - Specify input/output connections
        - Configure model parameters
        - Link to forcing data
        """
        self.logger.info("Generating realization configuration")
        
        # Get absolute paths
        forcing_abs_path = str(forcing_file.resolve())
        
        # Model configuration directories
        cfe_config_base = str((self.ngen_setup_dir / "CFE").resolve())
        pet_config_base = str((self.ngen_setup_dir / "PET").resolve())
        noah_config_base = str((self.ngen_setup_dir / "NOAH").resolve())
        
        # Get simulation time bounds - handle 'default' string
        sim_start = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        
        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'
        
        # Ensure time strings have seconds (some configs may omit them)
        # Convert to datetime and back to ensure proper format
        sim_start = pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M:%S')
        sim_end = pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create realization config
        config = {
            "global": {
                "formulations": [{
                    "name": "bmi_multi",
                    "params": {
                        "model_type_name": "bmi_multi_noahowp_cfe",
                        "init_config": "",
                        "allow_exceed_end_time": True,
                        "main_output_variable": "Q_OUT",
                        "modules": [
                            {
                                "name": "bmi_c++",
                                "params": {
                                    "model_type_name": "bmi_c++_sloth",
                                    "library_file": "./extern/sloth/cmake_build/libslothmodel.so",
                                    "init_config": "/dev/null",
                                    "allow_exceed_end_time": True,
                                    "main_output_variable": "z",
                                    "uses_forcing_file": False,
                                    "model_params": {
                                        "sloth_ice_fraction_schaake(1,double,m,node)": 0.0,
                                        "sloth_ice_fraction_xinanjiang(1,double,1,node)": 0.0,
                                        "sloth_smp(1,double,1,node)": 0.0
                                    }
                                }
                            },
                            {
                                "name": "bmi_c",
                                "params": {
                                    "model_type_name": "PET",
                                    "library_file": "./extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi.so",
                                    "forcing_file": "",
                                    "init_config": f"{pet_config_base}/{{{{id}}}}_pet_config.txt",
                                    "allow_exceed_end_time": True,
                                    "main_output_variable": "water_potential_evaporation_flux",
                                    "registration_function": "register_bmi_pet",
                                    "uses_forcing_file": False
                                }
                            },
                            {
                                "name": "bmi_fortran",
                                "params": {
                                    "model_type_name": "bmi_fortran_noahowp",
                                    "library_file": "./extern/noah-owp-modular/cmake_build/libsurfacebmi.so",
                                    "forcing_file": "",
                                    "init_config": f"{noah_config_base}/{{{{id}}}}.input",
                                    "allow_exceed_end_time": True,
                                    "main_output_variable": "QINSUR",
                                    "uses_forcing_file": False,
                                    "variables_names_map": {
                                        "PRCPNONC": "atmosphere_water__liquid_equivalent_precipitation_rate",
                                        "Q2": "atmosphere_air_water~vapor__relative_saturation",
                                        "SFCTMP": "land_surface_air__temperature",
                                        "UU": "land_surface_wind__x_component_of_velocity",
                                        "VV": "land_surface_wind__y_component_of_velocity",
                                        "LWDN": "land_surface_radiation~incoming~longwave__energy_flux",
                                        "SOLDN": "land_surface_radiation~incoming~shortwave__energy_flux",
                                        "SFCPRS": "land_surface_air__pressure"
                                    }
                                }
                            },
                            {
                                "name": "bmi_c",
                                "params": {
                                    "model_type_name": "bmi_c_cfe",
                                    "library_file": "./extern/cfe/cmake_build/libcfebmi.so",
                                    "forcing_file": "",
                                    "init_config": f"{cfe_config_base}/{{{{id}}}}_bmi_config_cfe_pass.txt",
                                    "allow_exceed_end_time": True,
                                    "main_output_variable": "Q_OUT",
                                    "registration_function": "register_bmi_cfe",
                                    "variables_names_map": {
                                        "atmosphere_water__liquid_equivalent_precipitation_rate": "QINSUR",
                                        "water_potential_evaporation_flux": "water_potential_evaporation_flux",
                                        "ice_fraction_schaake": "sloth_ice_fraction_schaake",
                                        "ice_fraction_xinanjiang": "sloth_ice_fraction_xinanjiang",
                                        "soil_moisture_profile": "sloth_smp"
                                    },
                                    "uses_forcing_file": False
                                }
                            }
                        ],
                        "uses_forcing_file": False
                    }
                }],
                "forcing": {
                    "path": forcing_abs_path,
                    "provider": "NetCDF"
                }
            },
            "time": {
                "start_time": sim_start,
                "end_time": sim_end,
                "output_interval": 3600
            }
        }
        
        # Save configuration
        config_file = self.ngen_setup_dir / "realization_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Created realization config: {config_file}")


class NgenRunner:
    """
    Runner for NextGen Framework simulations.
    
    Handles execution of ngen with proper paths and error handling.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the NextGen runner.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
        """
        self.config = config
        self.logger = logger
        
        self.project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}"
        self.ngen_setup_dir = self.project_dir / "settings" / "ngen"
        
        # Get ngen installation path
        ngen_install_path = config.get('NGEN_INSTALL_PATH', 'default')
        if ngen_install_path == 'default':
            self.ngen_exe = Path(config.get('CONFLUENCE_DATA_DIR')).parent / 'installs' / 'ngen' / 'build' / 'ngen'
        else:
            self.ngen_exe = Path(ngen_install_path) / 'ngen'
    
    def run_model(self):
        """
        Execute NextGen model simulation.
        
        Runs ngen with the prepared catchment, nexus, forcing, and configuration files.
        """
        self.logger.info("Starting NextGen model run")
        
        # Get experiment info
        experiment_id = self.config.get('EXPERIMENT_ID', 'default_run')
        output_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths for ngen execution
        catchment_file = self.ngen_setup_dir / f"{self.config.get('DOMAIN_NAME')}_catchments.gpkg"
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        realization_file = self.ngen_setup_dir / "realization_config.json"
        
        # Verify files exist
        for file in [catchment_file, nexus_file, realization_file]:
            if not file.exists():
                raise FileNotFoundError(f"Required file not found: {file}")
        
        # Setup environment with library paths
        env = os.environ.copy()
        
        # Get ngen conda environment path from config or use default
        ngen_conda_env = self.config.get('NGEN_CONDA_ENV', 'ngen_py310')
        ngen_conda_path = Path.home() / '.conda' / 'envs' / ngen_conda_env / 'lib'
        
        # Get current conda environment lib path
        current_conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'confluence')
        current_conda_path = Path.home() / '.conda' / 'envs' / current_conda_env / 'lib'
        
        # Build LD_LIBRARY_PATH with conda environments and existing path
        # Note: Load system modules (netcdf-c, netcdf-cxx4, udunits) before running CONFLUENCE
        lib_paths = [
            str(current_conda_path),  # Current environment first
            str(ngen_conda_path),     # ngen environment second
        ]
        
        # Add existing LD_LIBRARY_PATH if present (includes system module paths)
        existing_ld_path = env.get('LD_LIBRARY_PATH', '')
        if existing_ld_path:
            lib_paths.append(existing_ld_path)
        
        env['LD_LIBRARY_PATH'] = ':'.join(lib_paths)
        
        self.logger.info(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
        
        # Build ngen command
        ngen_cmd = [
            str(self.ngen_exe),
            str(catchment_file),
            "all",
            str(nexus_file),
            "all",
            str(realization_file)
        ]
        
        self.logger.info(f"Running command: {' '.join(ngen_cmd)}")
        
        # Run ngen
        log_file = output_dir / "ngen_log.txt"
        try:
            with open(log_file, 'w') as log:
                result = subprocess.run(
                    ngen_cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=self.ngen_exe.parent,  # Run from ngen build directory (needed for relative library paths)
                    env=env  # Use modified environment with library paths
                )
            
            # Move outputs from build directory to output directory
            self._move_ngen_outputs(self.ngen_exe.parent, output_dir)
            
            self.logger.info("NextGen model run completed successfully")
            return output_dir
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"NextGen model run failed with error code {e.returncode}")
            self.logger.error(f"Check log file: {log_file}")
            raise
    
    def _move_ngen_outputs(self, build_dir: Path, output_dir: Path):
        """
        Move ngen output files from build directory to output directory.
        
        ngen writes outputs to its working directory, so we need to move them
        to the proper experiment output directory.
        
        Args:
            build_dir: ngen build directory where outputs are written
            output_dir: Target output directory for this experiment
        """
        import shutil
        
        # Common ngen output patterns
        output_patterns = [
            'cat-*.csv',      # Catchment outputs
            'nex-*.csv',      # Nexus outputs  
            '*.parquet',      # Parquet outputs
            'cfe_output_*.txt',  # CFE specific outputs
            'noah_output_*.txt', # Noah specific outputs
        ]
        
        moved_files = []
        for pattern in output_patterns:
            for file in build_dir.glob(pattern):
                dest = output_dir / file.name
                shutil.move(str(file), str(dest))
                moved_files.append(file.name)
        
        if moved_files:
            self.logger.info(f"Moved {len(moved_files)} output files to {output_dir}")
            for f in moved_files[:10]:  # Log first 10
                self.logger.info(f"  - {f}")
            if len(moved_files) > 10:
                self.logger.info(f"  ... and {len(moved_files) - 10} more")
        else:
            self.logger.warning(f"No output files found in {build_dir}. Check if model ran correctly.")



class NgenPostprocessor:
    """
    Postprocessor for NextGen Framework outputs.
    
    Handles extraction and analysis of simulation results.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the NextGen postprocessor.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
        """
        self.config = config
        self.logger = logger
        
        self.project_dir = Path(config.get('CONFLUENCE_DATA_DIR')) / f"domain_{config.get('DOMAIN_NAME')}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_streamflow(self, experiment_id: str = None) -> Optional[Path]:
        """
        Extract streamflow from ngen nexus outputs.
        
        Args:
            experiment_id: Experiment identifier (default: from config)
            
        Returns:
            Path to extracted streamflow CSV file
        """
        self.logger.info("Extracting streamflow from ngen outputs")
        
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'run_1')
        
        # Get output directory
        output_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        
        # Find nexus output files
        nexus_files = list(output_dir.glob('nex-*_output.csv'))
        
        if not nexus_files:
            self.logger.error(f"No nexus output files found in {output_dir}")
            return None
        
        self.logger.info(f"Found {len(nexus_files)} nexus output file(s)")
        
        # Read and process each nexus file
        all_streamflow = []
        for nexus_file in nexus_files:
            nexus_id = nexus_file.stem.replace('_output', '')
            
            try:
                # Read nexus output
                df = pd.read_csv(nexus_file)
                
                # Check for flow column (common names)
                flow_col = None
                for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                    if col_name in df.columns:
                        flow_col = col_name
                        break
                
                if flow_col is None:
                    self.logger.warning(f"No flow column found in {nexus_file}. Columns: {df.columns.tolist()}")
                    continue
                
                # Extract time and flow
                if 'Time' in df.columns:
                    time = pd.to_datetime(df['Time'], unit='ns')
                elif 'time' in df.columns:
                    time = pd.to_datetime(df['time'])
                else:
                    self.logger.warning(f"No time column found in {nexus_file}")
                    continue
                
                # Create streamflow dataframe
                streamflow_df = pd.DataFrame({
                    'datetime': time,
                    'streamflow_cms': df[flow_col],
                    'nexus_id': nexus_id
                })
                
                all_streamflow.append(streamflow_df)
                
            except Exception as e:
                self.logger.error(f"Error processing {nexus_file}: {e}")
                continue
        
        if not all_streamflow:
            self.logger.error("No streamflow data could be extracted")
            return None
        
        # Combine all nexus outputs
        combined_streamflow = pd.concat(all_streamflow, ignore_index=True)
        
        # Save to results directory
        output_file = self.results_dir / f"ngen_streamflow_{experiment_id}.csv"
        combined_streamflow.to_csv(output_file, index=False)
        
        self.logger.info(f"Extracted streamflow saved to: {output_file}")
        self.logger.info(f"Total timesteps: {len(combined_streamflow)}")
        
        return output_file
    
    def plot_streamflow(self, experiment_id: str = None, observed_file: Path = None) -> Optional[Path]:
        """
        Create streamflow plots comparing simulated and observed (if available).
        
        Args:
            experiment_id: Experiment identifier
            observed_file: Path to observed streamflow CSV file
            
        Returns:
            Path to plot file
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        self.logger.info("Creating streamflow plots")
        
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'run_1')
        
        # Get streamflow file
        streamflow_file = self.results_dir / f"ngen_streamflow_{experiment_id}.csv"
        
        if not streamflow_file.exists():
            self.logger.info("Streamflow file not found, extracting first...")
            streamflow_file = self.extract_streamflow(experiment_id)
            if streamflow_file is None:
                return None
        
        # Read simulated streamflow
        sim_df = pd.read_csv(streamflow_file)
        sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Full time series
        ax1 = axes[0]
        ax1.plot(sim_df['datetime'], sim_df['streamflow_cms'], 
                label='NGEN Simulated', color='blue', linewidth=0.8)
        
        # Add observed if available
        if observed_file and Path(observed_file).exists():
            obs_df = pd.read_csv(observed_file)
            obs_df['datetime'] = pd.to_datetime(obs_df['datetime'])
            ax1.plot(obs_df['datetime'], obs_df['streamflow_cms'], 
                    label='Observed', color='red', linewidth=0.8, alpha=0.7)
            
            # Calculate metrics
            merged = pd.merge(sim_df, obs_df, on='datetime', suffixes=('_sim', '_obs'))
            nse = self._calculate_nse(merged['streamflow_cms_obs'], merged['streamflow_cms_sim'])
            kge = self._calculate_kge(merged['streamflow_cms_obs'], merged['streamflow_cms_sim'])
            
            ax1.text(0.02, 0.98, f'NSE: {nse:.3f}\nKGE: {kge:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_ylabel('Streamflow (cms)', fontsize=12)
        ax1.set_title(f'NGEN Streamflow - {experiment_id}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot 2: Flow duration curve
        ax2 = axes[1]
        sorted_flow = np.sort(sim_df['streamflow_cms'].values)[::-1]
        exceedance = np.arange(1, len(sorted_flow) + 1) / len(sorted_flow) * 100
        ax2.semilogy(exceedance, sorted_flow, label='NGEN Simulated', color='blue', linewidth=1.5)
        
        if observed_file and Path(observed_file).exists():
            sorted_obs = np.sort(obs_df['streamflow_cms'].values)[::-1]
            exceedance_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs) * 100
            ax2.semilogy(exceedance_obs, sorted_obs, label='Observed', 
                        color='red', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Exceedance Probability (%)', fontsize=12)
        ax2.set_ylabel('Streamflow (cms)', fontsize=12)
        ax2.set_title('Flow Duration Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"ngen_streamflow_plot_{experiment_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Streamflow plot saved to: {plot_file}")
        
        return plot_file
    
    def _calculate_nse(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]
        
        if len(obs) == 0:
            return np.nan
        
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)
    
    def _calculate_kge(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]
        
        if len(obs) == 0:
            return np.nan
        
        # Calculate components
        r = np.corrcoef(obs, sim)[0, 1]  # Correlation
        alpha = np.std(sim) / np.std(obs)  # Variability ratio
        beta = np.mean(sim) / np.mean(obs)  # Bias ratio
        
        # Calculate KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        
        return kge