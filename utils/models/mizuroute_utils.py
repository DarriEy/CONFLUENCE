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

sys.path.append(str(Path(__file__).resolve().parent.parent))

class MizuRoutePreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.mizuroute_setup_dir = self.project_dir / "settings" / "mizuRoute"

    def run_preprocessing(self):
        self.logger.info("Starting mizuRoute spatial preprocessing")
        
        self.copy_base_settings()
        self.create_network_topology_file()
        if self.config.get('SETTINGS_MIZUE_NEEDS_REMAP', '') == 'yes':
            self.remap_summa_catchments_to_routing()
        self.create_control_file()
        
        self.logger.info("mizuRoute spatial preprocessing completed")

    def copy_base_settings(self):
        self.logger.info("Copying mizuRoute base settings")
        base_settings_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_base_settings' / 'mizuRoute'
        self.mizuroute_setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.mizuroute_setup_dir / file)
        self.logger.info("mizuRoute base settings copied")

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
        
        num_seg = len(shp_river)
        num_hru = len(shp_basin)
        
        # Ensure minimum segment length
        shp_river.loc[shp_river[self.config.get('RIVER_NETWORK_SHP_LENGTH')] == 0, self.config.get('RIVER_NETWORK_SHP_LENGTH')] = 1
        
        # Enforce outlets if specified
        if self.config.get('SETTINGS_MIZU_MAKE_OUTLET') != 'n/a':
            river_outlet_ids = [int(id) for id in self.config.get('SETTINGS_MIZU_MAKE_OUTLET').split(',')]
            for outlet_id in river_outlet_ids:
                if outlet_id in shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')].values:
                    shp_river.loc[shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')] == outlet_id, self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')] = 0
                else:
                    self.logger.warning(f"Outlet ID {outlet_id} not found in river network")
        
        # Create the netCDF file
        with nc4.Dataset(self.mizuroute_setup_dir / topology_name, 'w', format='NETCDF4') as ncid:
            self._set_topology_attributes(ncid)
            self._create_topology_dimensions(ncid, num_seg, num_hru)
            self._create_topology_variables(ncid, shp_river, shp_basin)
        
        self.logger.info(f"Network topology file created at {self.mizuroute_setup_dir / topology_name}")

    def remap_summa_catchments_to_routing(self):
        self.logger.info("Remapping SUMMA catchments to routing catchments")

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
        self.logger.info("Creating mizuRoute control file")
        
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
        
        self.logger.info(f"mizuRoute control file created at {self.mizuroute_setup_dir / control_name}")

    def _set_topology_attributes(self, ncid):
        now = datetime.now()
        ncid.setncattr('Author', "Created by SUMMA workflow scripts")
        ncid.setncattr('History', f'Created {now.strftime("%Y/%m/%d %H:%M:%S")}')
        ncid.setncattr('Purpose', 'Create a river network .nc file for mizuRoute routing')

    def _create_topology_dimensions(self, ncid, num_seg, num_hru):
        ncid.createDimension('seg', num_seg)
        ncid.createDimension('hru', num_hru)

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

    def _write_control_file_simulation_controls(self, cf):
        self.sim_start = self.config.get('EXPERIMENT_TIME_START')
        self.sim_end = self.config.get('EXPERIMENT_TIME_END')

        if self.sim_start == 'default' or self.sim_end == 'default':
            raw_time = [
                    self.config.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
            self.sim_start = f"{raw_time[0]}-01-01 00:00" if self.sim_start == 'default' else self.sim_start
            self.sim_end = f"{raw_time[1]}-12-31 23:00" if self.sim_end == 'default' else self.sim_end

        cf.write("!\n! --- DEFINE SIMULATION CONTROLS \n")
        cf.write(f"<case_name>             {self.config.get('EXPERIMENT_ID')}    ! Simulation case name. This used for output netCDF, and restart netCDF name \n")
        cf.write(f"<sim_start>             {self.sim_start}    ! Time of simulation start. format: yyyy-mm-dd or yyyy-mm-dd hh:mm:ss \n")
        cf.write(f"<sim_end>               {self.sim_end}    ! Time of simulation end. format: yyyy-mm-dd or yyyy-mm-dd hh:mm:ss \n")
        cf.write(f"<route_opt>             {self.config.get('SETTINGS_MIZU_OUTPUT_VARS')}    ! Option for routing schemes. 0: both; 1: IRF; 2: KWT. Saves no data if not specified \n")
        cf.write(f"<newFileFrequency>      {self.config.get('SETTINGS_MIZU_OUTPUT_FREQ')}    ! Frequency for new output files (single, day, month, or annual) \n")

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

    def _write_control_file_runoff(self, cf):
        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {self.config.get('EXPERIMENT_ID')}_timestep.nc    ! netCDF name for HM_HRU runoff \n")
        cf.write(f"<vname_qsim>            {self.config.get('SETTINGS_MIZU_ROUTING_VAR')}    ! Variable name for HM_HRU runoff \n")
        cf.write(f"<units_qsim>            {self.config.get('SETTINGS_MIZU_ROUTING_UNITS')}    ! Units of input runoff. e.g., mm/s \n")
        cf.write(f"<dt_qsim>               {self.config.get('SETTINGS_MIZU_ROUTING_DT')}    ! Time interval of input runoff in seconds, e.g., 86400 sec for daily step \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        cf.write("<dname_hruid>           gru     ! Dimension name for HM_HRU ID \n")
        cf.write("<vname_hruid>           gruId   ! Variable name for HM_HRU ID \n")
        cf.write("<calendar>              standard    ! Calendar of the nc file if not provided in the time variable of the nc file \n")

    def _write_control_file_remapping(self, cf):
        cf.write("!\n! --- DEFINE RUNOFF MAPPING FILE \n")
        remap_flag = self.config.get('river_basin_needs_remap', '').lower() == 'yes'
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

    def fix_summa_time_precision(self):
        """
        Fix SUMMA output time precision by rounding to nearest hour.
        This fixes compatibility issues with mizuRoute time matching.
        """
        self.logger.info("Fixing SUMMA time precision for mizuRoute compatibility")
        
        # Get SUMMA output path
        experiment_output_summa = self.config.get('EXPERIMENT_OUTPUT_SUMMA')
        if experiment_output_summa == 'default':
            experiment_output_summa = self.project_dir / f"simulations/{self.config['EXPERIMENT_ID']}" / 'SUMMA'
        else:
            experiment_output_summa = Path(experiment_output_summa)
        
        # Get the specific runoff file
        runoff_filename = f"{self.config.get('EXPERIMENT_ID')}_timestep.nc"
        runoff_filepath = experiment_output_summa / runoff_filename
        
        if not runoff_filepath.exists():
            self.logger.error(f"SUMMA output file not found: {runoff_filepath}")
            return
        
        try:
            import xarray as xr
            import os
            
            self.logger.info(f"Processing {runoff_filepath}")
            ds = xr.open_dataset(runoff_filepath)
            
            # Check if time fixing is needed
            first_time = ds.time.values[0]
            rounded_time = pd.Timestamp(first_time).round('H')
            
            if pd.Timestamp(first_time) != rounded_time:
                self.logger.info("Time precision issue detected, rounding to nearest hour")
                
                # Round time to nearest hour
                ds['time'] = ds.time.dt.round('H')
                
                # Remove existing time attributes that might conflict
                attrs_to_remove = ['units', 'calendar', 'long_name']
                for attr in attrs_to_remove:
                    if attr in ds.time.attrs:
                        del ds.time.attrs[attr]
                
                # Fix time encoding to remove timezone info
                ds.time.attrs['units'] = 'hours since 1981-01-01 00:00:00'
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'
                ds.time.encoding['units'] = 'hours since 1981-01-01 00:00:00'
                ds.time.encoding['calendar'] = 'standard'
                if 'dtype' in ds.time.encoding:
                    del ds.time.encoding['dtype']
                
                # Load data into memory and close file
                ds.load()
                ds.close()
                
                # Make file writable and overwrite original
                os.chmod(runoff_filepath, 0o664)
                ds.to_netcdf(runoff_filepath)
                self.logger.info("SUMMA time precision fixed")
            else:
                self.logger.info("SUMMA time precision is already correct")
                ds.close()
                
        except Exception as e:
            self.logger.error(f"Error fixing SUMMA time precision: {e}")
            raise

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting mizuRoute run")
        self.fix_summa_time_precision()
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
        
        try:
            with open(mizu_log_path / mizu_log_name, 'w') as log_file:
                subprocess.run(mizu_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info("mizuRoute run completed successfully")
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