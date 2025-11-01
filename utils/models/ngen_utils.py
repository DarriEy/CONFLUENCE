"""
NextGen (ngen) Framework Utilities for CONFLUENCE

This module provides preprocessing, execution, and postprocessing utilities
for the NOAA NextGen Water Resources Modeling Framework within CONFLUENCE.

Classes:
    NgenPreProcessor: Handles spatial preprocessing and configuration generation
    NgenRunner: Manages model execution
    NgenPostprocessor: Processes model outputs
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
        
        Note: These are generic CONUS tables. For best results:
        - Verify soil_class_name matches your region
        - Consider region-specific MPTABLE.TBL
        - Check vegetation types in your domain
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

        Steps:
        1) load monthly forcing files
        2) merge across time
        3) subset to simulation window
        4) map variables and reshape to (catchment, time)
        5) write a clean NETCDF4 file
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

        # Open/concat by time
        forcing_data = xr.open_mfdataset(
            [str(p) for p in forcing_files],
            combine="by_coords",
            parallel=False,
            decode_times=True,
            engine="netcdf4",
        )

        # Simulation window (handle 'default' in config)
        sim_start = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        sim_end   = self.config.get('EXPERIMENT_TIME_END',   '2000-12-31 23:00:00')
        if sim_start == 'default':
            sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default':
            sim_end = '2000-12-31 23:00:00'

        start_time = pd.to_datetime(sim_start)
        end_time   = pd.to_datetime(sim_end)

        # Ensure datetime64[ns]
        forcing_data = forcing_data.assign_coords(
            time=pd.to_datetime(forcing_data.time.values)
        )

        # Subset
        forcing_data = forcing_data.sel(time=slice(start_time, end_time))

        self.logger.info(
            f"Forcing time range: {forcing_data.time.values[0]} to {forcing_data.time.values[-1]}"
        )

        # Create ngen-formatted dataset
        ngen_ds = self._create_ngen_forcing_dataset(forcing_data, catchment_ids)

        # Output path
        output_file = self.forcing_dir / "forcing.nc"

        # Build encoding dynamically only for variables that exist
        encoding: Dict[str, Dict[str, Any]] = {}

        # Downcast float data vars + set fill
        for name, da in ngen_ds.data_vars.items():
            if np.issubdtype(da.dtype, np.floating):
                encoding[name] = {'dtype': 'float32', '_FillValue': np.nan}

        # Let xarray handle CF time encoding; no manual dtype needed for 'time'
        # (If you insist on integer epoch, you could set dtype=int64, but CF is safer.)

        ngen_ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
        # Sanity checks the NetCDF provider depends on
        try:
            with nc4.Dataset(output_file, mode='r') as ds:
                # Allow either 'feature_id' (preferred) or 'catchment-id'
                has_feature = 'feature_id' in ds.dimensions
                has_catchment = 'catchment-id' in ds.dimensions
                assert has_feature or has_catchment, "Missing dim 'feature_id' (or 'catchment-id')"

                feat_dim = 'feature_id' if has_feature else 'catchment-id'

                # time dim
                assert 'time' in ds.dimensions, "Missing dim 'time'"

                # ids variable must exist and be 1-D over the feature dim
                assert 'ids' in ds.variables, "Missing variable 'ids'"
                ids_var = ds.variables['ids']
                assert ids_var.dimensions == (feat_dim,), f"'ids' must be 1-D over '{feat_dim}'"

                # some core vars present (either canonical or AORC aliases)
                present = set(ds.variables.keys())
                must_have_any = [
                    'precip_rate', 'PRCPNONC'
                ]
                assert any(v in present for v in must_have_any), \
                    "Missing precipitation variable (need 'precip_rate' or 'PRCPNONC')"

            self.logger.info(f"Forcing file passes basic NGen checks (dims/ids via '{feat_dim}').")
        except Exception as e:
            self.logger.error(f"Forcing validation failed: {e}")
            raise

    def _create_ngen_forcing_dataset(
        self, forcing_data: xr.Dataset, catchment_ids: List[str]
    ) -> xr.Dataset:
        """
        Build an NGen-ready forcing file that satisfies both:
        - NGen NetCDF provider's canonical names (incl. precip_rate), and
        - AORC-style names expected by Noah/PET/CFE examples.
        Output dims: ('feature_id','time'); string coord 'ids' on 'feature_id'.
        """
        # ---------- prep ----------
        n_cats = len(catchment_ids)
        time_vals = forcing_data.time.values.astype('datetime64[ns]')

        # utility to expand (time,) → (feature_id,time)
        def expand_to_features(arr_1d_time: np.ndarray) -> np.ndarray:
            return np.tile(arr_1d_time[None, :], (n_cats, 1))

        # start empty ds with the final dims we want
        ds = xr.Dataset()
        ds = ds.assign_coords(
            feature_id=("feature_id", np.array(catchment_ids, dtype=object)),
            time=time_vals,
        )
        # expose ids as both coord and data var for maximum compatibility
        ds["ids"] = xr.DataArray(ds["feature_id"].values, dims=("feature_id",))

        # ---------- core mappings you already have ----------
        # Source → canonical NetCDF names used by the provider
        # (we'll also write AORC aliases below)
        mapping = {
            "pptrate": "precip_rate",           # kg m-2 s-1 (mm s-1)
            "t2m": "TMP_2maboveground",         # K
            "d2m": "DPT_2maboveground",         # K  (we'll still compute Q2 from d2m)
            "sp": "PRES_surface",               # Pa
            "dswrf": "DSWRF_surface",           # W m-2
            "dlwrf": "DLWRF_surface",           # W m-2
            "u10": "UGRD_10maboveground",       # m s-1
            "v10": "VGRD_10maboveground",       # m s-1
            "windspd": None,                    # optional fallback handled below
        }

        # write canonical variables (duplicated over features)
        for src, tgt in mapping.items():
            if tgt is None or (src not in forcing_data):
                continue
            arr = forcing_data[src].values
            # handle (time,) or (1,time) inputs
            if arr.ndim == 1:
                arr2d = expand_to_features(arr)
            elif arr.ndim == 2 and 1 in arr.shape:
                arr2d = expand_to_features(np.squeeze(arr))
            else:
                # assume already aligned; if (time,feat) flip
                arr2d = arr.T if arr.shape[0] == forcing_data.dims.get("time", len(time_vals)) else arr
            ds[tgt] = xr.DataArray(arr2d.astype(np.float32), dims=("feature_id", "time"))

        # wind fallback if no u10/v10 but windspd exists
        if ("UGRD_10maboveground" not in ds) and ("VGRD_10maboveground" not in ds) and ("windspd" in forcing_data):
            self.logger.warning("Using wind speed approximation for U/V components. For better accuracy, consider ERA5 u10/v10.")
            w = forcing_data["windspd"].values
            if w.ndim == 1:
                base = expand_to_features(w / np.sqrt(2.0))
            elif w.ndim == 2 and 1 in w.shape:
                base = expand_to_features(np.squeeze(w) / np.sqrt(2.0))
            else:
                base = (w.T if w.shape[0] == forcing_data.dims.get("time", len(time_vals)) else w) / np.sqrt(2.0)
            ds["UGRD_10maboveground"] = xr.DataArray(base.astype(np.float32), dims=("feature_id", "time"))
            ds["VGRD_10maboveground"] = xr.DataArray(base.astype(np.float32), dims=("feature_id", "time"))

        # ---------- compute Q2 (specific humidity) from t2m, d2m, pressure ----------
        if ("TMP_2maboveground" in ds) and ("DPT_2maboveground" in ds) and ("PRES_surface" in ds):
            q2 = self._compute_q2_specific_humidity(
                ds["TMP_2maboveground"].values,
                ds["DPT_2maboveground"].values,
                ds["PRES_surface"].values,
            )
            ds["Q2"] = xr.DataArray(q2, dims=("feature_id", "time"))
        else:
            # If we can’t compute Q2, still proceed; Noah/PET may derive internally,
            # but having Q2 avoids provider lookups failing on some builds.
            self.logger.info("Q2 not computed (need t2m, d2m, and surface pressure). Proceeding without it.")

        # ---------- AORC aliases (duplicate variables so both naming schemes exist) ----------
        # Precipitation
        if "precip_rate" in ds:
            ds["PRCPNONC"] = ds["precip_rate"]  # duplicate

        # Temperature
        if "TMP_2maboveground" in ds:
            ds["SFCTMP"] = ds["TMP_2maboveground"]

        # Pressure
        if "PRES_surface" in ds:
            ds["SFCPRS"] = ds["PRES_surface"]

        # Shortwave/Longwave
        if "DSWRF_surface" in ds:
            ds["SOLDN"] = ds["DSWRF_surface"]
        if "DLWRF_surface" in ds:
            ds["LWDN"] = ds["DLWRF_surface"]

        # Wind components
        if "UGRD_10maboveground" in ds:
            ds["UU"] = ds["UGRD_10maboveground"]
        if "VGRD_10maboveground" in ds:
            ds["VV"] = ds["VGRD_10maboveground"]

        # ---------- final touch ----------
        # Put a convenience alias coord many tools expect
        ds = ds.assign_coords(**{"catchment-id": ("feature_id", ds["feature_id"].values)})

        return ds



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
        """
        Generate CFE model configuration file.
        
        NOTE: This uses generic default parameters. For better results:
        - Extract parameters from NWM hydrofabric attributes
        - Use soil texture-based parameters
        - Customize GIUH ordinates based on catchment size
        """
        
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
        
        # IMPROVED: Validate the generated config
        if not self._validate_cfe_config(config_file):
            self.logger.warning(f"Generated CFE config may be incomplete: {config_file}")

    def _compute_q2_specific_humidity(self, t2m_K: np.ndarray, d2m_K: np.ndarray, pres_Pa: np.ndarray) -> np.ndarray:
        """
        Compute specific humidity (Q2, kg/kg) at 2m from air temp (t2m, K),
        dewpoint (d2m, K) and surface pressure (Pa).
        Formula: q = 0.622 * e / (p - 0.378 * e), where e is vapor pressure (Pa).
        e from dewpoint (Tetens, in Pa): e = 611.2 * exp(17.67 * Td_C / (Td_C + 243.5))
        """
        Td_C = d2m_K - 273.15
        # Tetens (Pa)
        e = 611.2 * np.exp(17.67 * Td_C / (Td_C + 243.5))
        p = pres_Pa
        # guard against weird/zero pressure
        p = np.where(p <= 100.0, 101325.0, p)
        q = 0.622 * e / (p - 0.378 * e)
        return q.astype(np.float32)

    def _validate_cfe_config(self, config_file: Path) -> bool:
        """
        Validate CFE config file has required parameters.
        
        NEW METHOD: Adds validation to catch errors early.
        """
        required_params = [
            'forcing_file', 'soil_params.smcmax', 'Cgw', 
            'max_gw_storage', 'K_nash', 'K_lf', 'giuh_ordinates'
        ]
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            missing_params = []
            for param in required_params:
                if param not in content:
                    missing_params.append(param)
            
            if missing_params:
                self.logger.error(f"CFE config missing parameters: {', '.join(missing_params)}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating CFE config: {e}")
            return False
    
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


    def _get_ngen_root(self) -> Path:
        """
        Default NGen root where we search for BMI .so files:
        <CONFLUENCE_data>/installs/ngen
        If NGEN_INSTALL_PATH is set, use that instead.
        """
        ngen_install_path = self.config.get('NGEN_INSTALL_PATH', 'default')
        if ngen_install_path and ngen_install_path != 'default':
            return Path(ngen_install_path)
        # Default
        confluence_data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        return confluence_data_dir.parent / 'installs' / 'ngen'

    def _find_bmi_library(self, name_patterns: List[str]) -> Optional[str]:
        """
        Search common build locations under the NGen root for a .so
        matching any of the provided glob patterns. Env vars can override.
        - CFE: NGEN_CFE_LIB
        - NOAH: NGEN_NOAH_LIB
        - PET: NGEN_PET_LIB
        """
        # env override
        env_overrides = {
            'cfe': os.environ.get('NGEN_CFE_LIB'),
            'noah': os.environ.get('NGEN_NOAH_LIB'),
            'pet': os.environ.get('NGEN_PET_LIB'),
        }
        # If any override matches requested patterns, use it
        for v in env_overrides.values():
            if v and any(Path(v).name.endswith(pat.replace('*', '')) or True for pat in name_patterns):
                if Path(v).exists():
                    return str(Path(v).resolve())

        root = self._get_ngen_root()
        # Places we commonly see built libs
        candidate_dirs = [
            root,
            root / 'build',
            root / 'extern',
            # typical cmake build dirs
            *[p for p in (root / 'extern').rglob('*') if p.is_dir() and any(s in p.name.lower() for s in ('build', 'cmake', 'lib'))]
        ]

        # Unique + existing
        seen = set()
        dirs = []
        for d in candidate_dirs:
            if d.exists():
                rp = d.resolve()
                if rp not in seen:
                    seen.add(rp)
                    dirs.append(rp)

        # Search breadth-first; return first match
        for d in dirs:
            for pat in name_patterns:
                for f in d.rglob(pat):
                    if f.is_file() and f.suffix in ('.so', '.dylib'):
                        return str(f.resolve())

        return None



    def generate_realization_config(self, catchment_file: Path, nexus_file: Path, forcing_file: Path):
        """
        Generate ngen realization configuration JSON.
        Auto-detect BMI library .so paths under CONFLUENCE_data/installs/ngen (or NGEN_INSTALL_PATH).
        Env var overrides:
        NGEN_NOAH_LIB, NGEN_PET_LIB, NGEN_CFE_LIB
        """
        self.logger.info("Generating realization configuration")

        forcing_abs_path = str(forcing_file.resolve())
        cfe_config_base = str((self.ngen_setup_dir / "CFE").resolve())
        pet_config_base = str((self.ngen_setup_dir / "PET").resolve())
        noah_config_base = str((self.ngen_setup_dir / "NOAH").resolve())

        # Time bounds
        sim_start = self.config.get('EXPERIMENT_TIME_START', '2000-01-01 00:00:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2000-12-31 23:00:00')
        if sim_start == 'default': sim_start = '2000-01-01 00:00:00'
        if sim_end == 'default': sim_end = '2000-12-31 23:00:00'
        sim_start = pd.to_datetime(sim_start).strftime('%Y-%m-%d %H:%M:%S')
        sim_end = pd.to_datetime(sim_end).strftime('%Y-%m-%d %H:%M:%S')

        # ---- Auto-detect BMI libraries ----
        # Try several common names per component
        noah_patterns = [
            '*noah*owp*modular*.*so', '*surface*bmi*.so', '*libnoah*.so',
            '*noah*owp*modular*.*dylib', '*surface*bmi*.dylib', '*libnoah*.dylib'
        ]
        pet_patterns = [
            '*evapo*transp*.*so', '*evapotranspiration*.*so', '*pet*bmi*.so',
            '*evapo*transp*.*dylib', '*evapotranspiration*.*dylib', '*pet*bmi*.dylib'
        ]
        cfe_patterns = [
            '*cfe*bmi*.so', '*libcfe*.so', '*bmi_cfe*.so',
            '*cfe*bmi*.dylib', '*libcfe*.dylib', '*bmi_cfe*.dylib'
        ]

        noah_lib = self._find_bmi_library(noah_patterns)
        pet_lib = self._find_bmi_library(pet_patterns)
        cfe_lib = self._find_bmi_library(cfe_patterns)

        # Log & warn if any unresolved
        if noah_lib: self.logger.info(f"NOAH BMI library: {noah_lib}")
        else:        self.logger.warning("NOAH BMI library not found. Set NGEN_NOAH_LIB or ensure build outputs are present.")

        if pet_lib:  self.logger.info(f"PET BMI library: {pet_lib}")
        else:        self.logger.warning("PET BMI library not found. Set NGEN_PET_LIB or ensure build outputs are present.")

        if cfe_lib:  self.logger.info(f"CFE BMI library: {cfe_lib}")
        else:        self.logger.warning("CFE BMI library not found. Set NGEN_CFE_LIB or ensure build outputs are present.")

        # Build modules config; leave empty string if missing (NGen will then error with a clear dlopen message)
        modules = [
            {
                "name": "bmi_fortran",
                "params": {
                    "name": "bmi_fortran",
                    "model_type_name": "NoahOWP",
                    "library_file": noah_lib or "",
                    "forcing_file": "",
                    "init_config": f"{noah_config_base}/{{{{id}}}}.input",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "QINSUR",
                    "registration_function": "register_bmi_noahowp",
                    "variables_names_map": {
                        "PRCPNONC": "atmosphere_water__liquid_equivalent_precipitation_rate",
                        "Q2": "atmosphere_air_water~vapor__specific_humidity",
                        "SFCTMP": "land_surface_air__temperature",
                        "UU": "land_surface_wind__x_component_of_velocity",
                        "VV": "land_surface_wind__y_component_of_velocity",
                        "LWDN": "land_surface_radiation~incoming~longwave__energy_flux",
                        "SOLDN": "land_surface_radiation~incoming~shortwave__energy_flux",
                        "SFCPRS": "land_surface_air__pressure"
                    },
                    "uses_forcing_file": False
                }
            },
            {
                "name": "bmi_c++",
                "params": {
                    "name": "bmi_c++",
                    "model_type_name": "EVAPOTRANSPIRATION",
                    "library_file": pet_lib or "",
                    "forcing_file": "",
                    "init_config": f"{pet_config_base}/{{{{id}}}}_pet_config.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "water_potential_evaporation_flux",
                    "registration_function": "register_bmi_pet",
                    "uses_forcing_file": False
                }
            },
            {
                "name": "bmi_c",
                "params": {
                    "name": "bmi_c",
                    "model_type_name": "CFE",
                    "library_file": cfe_lib or "",
                    "forcing_file": "",
                    "init_config": f"{cfe_config_base}/{{{{id}}}}_bmi_config_cfe_pass.txt",
                    "allow_exceed_end_time": True,
                    "main_output_variable": "Q_OUT",
                    "registration_function": "register_bmi_cfe",
                    "variables_names_map": {
                        "atmosphere_water__liquid_equivalent_precipitation_rate": "QINSUR",
                        "water_potential_evaporation_flux": "water_potential_evaporation_flux"
                    },
                    "uses_forcing_file": False
                }
            }
        ]

        config = {
            "global": {
                "formulations": [{
                    "name": "bmi_multi",
                    "params": {
                        "model_type_name": "bmi_multi_noahowp_cfe",
                        "init_config": "",
                        "allow_exceed_end_time": True,
                        "main_output_variable": "Q_OUT",
                        "modules": modules,
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
    
    def run_model(self, experiment_id: str = None):
        """
        Execute NextGen model simulation.
        
        Args:
            experiment_id: Optional experiment identifier. If None, uses config value.
        
        Runs ngen with the prepared catchment, nexus, forcing, and configuration files.
        """
        self.logger.debug("Starting NextGen model run")
        
        # Get experiment info
        if experiment_id is None:
            experiment_id = self.config.get('EXPERIMENT_ID', 'default_run')
        output_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths for ngen execution
        catchment_file = self.ngen_setup_dir / f"{self.config.get('DOMAIN_NAME')}_catchments.gpkg"
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        realization_file = self.ngen_setup_dir / "realization_config.json"
        
        # Verify files exist - IMPROVED with better error messages
        missing_files = []
        for file in [catchment_file, nexus_file, realization_file]:
            if not file.exists():
                missing_files.append(str(file))
        
        if missing_files:
            error_msg = f"Required ngen input files not found:\n"
            for f in missing_files:
                error_msg += f"  - {f}\n"
            error_msg += "\nPlease run ngen preprocessing first."
            raise FileNotFoundError(error_msg)
        
        # Setup environment with library paths
        env = os.environ.copy()
        
        # Build ngen command
        ngen_cmd = [
            str(self.ngen_exe),
            str(catchment_file),
            "all",
            str(nexus_file),
            "all",
            str(realization_file)
        ]
        
        self.logger.debug(f"Running command: {' '.join(ngen_cmd)}")
        self.logger.debug(f"Working directory: {self.ngen_exe.parent}")
        
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
            
            self.logger.debug("NextGen model run completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            # IMPROVED error handling
            self.logger.error(f"NextGen model run failed with error code {e.returncode}")
            self.logger.error(f"Command: {' '.join(ngen_cmd)}")
            self.logger.error(f"Working directory: {self.ngen_exe.parent}")
            self.logger.error(f"Check log file for details: {log_file}")
            
            # Try to extract useful error info from log
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        
                    # Look for common error patterns
                    if "No such file" in log_content or "cannot open" in log_content:
                        self.logger.error("NGEN couldn't find a required file")
                    if "Failed to initialize" in log_content:
                        self.logger.error("BMI module initialization failed")
                    if "forcing" in log_content.lower():
                        self.logger.error("Possible forcing data issue")
                        
                    # Show last 20 lines of log
                    log_lines = log_content.split('\n')
                    self.logger.error("Last 20 lines of ngen log:")
                    for line in log_lines[-20:]:
                        self.logger.error(f"  {line}")
                        
                except Exception as log_error:
                    self.logger.error(f"Couldn't read log file: {log_error}")
            
            return False
    
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
            self.logger.debug(f"Moved {len(moved_files)} output files to {output_dir}")
            for f in moved_files[:10]:  # Log first 10
                self.logger.debug(f"  - {f}")
            if len(moved_files) > 10:
                self.logger.debug(f"  ... and {len(moved_files) - 10} more")
        else:
            self.logger.warning(f"No output files found in {build_dir}. Check if model ran correctly.")
            self.logger.warning(f"Expected patterns: {', '.join(output_patterns)}")


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