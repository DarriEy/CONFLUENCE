"""
This module provides a complete, standalone utility for integrating the t-route
routing model within the SYMFLUENCE framework. It handles all necessary 
preprocessing, configuration, data preparation, and execution steps.

Classes:
    TRoutePreProcessor: Handles the creation of t-route specific input files,
                        including the NWM-standard network topology NetCDF and
                        the YAML configuration file.
    TRouteRunner:       Manages the execution of the t-route model, including
                        a crucial data preparation step to rename the runoff
                        variable to the required 'q_lateral'.
"""
import os
import sys
import yaml
import netCDF4 as nc4
import geopandas as gpd
import xarray as xr
from pathlib import Path
from shutil import copyfile
from datetime import datetime
from typing import Dict, Any
import subprocess

class TRoutePreProcessor:
    """
    A standalone preprocessor for t-route within the SYMFLUENCE framework.

    This class creates all necessary input and configuration files for a 
    t-route run without any dependency on other routing model utilities.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.troute_setup_dir = self.project_dir / "settings" / "troute"

    def run_preprocessing(self):
        """Main entry point for running all t-route preprocessing steps."""
        self.logger.info("--- Starting t-route Preprocessing ---")
        self.copy_base_settings()
        self.create_troute_topology_file()
        self.create_troute_yaml_config()
        self.logger.info("--- t-route Preprocessing Completed Successfully ---")

    def copy_base_settings(self):
        """Copies base settings for t-route from the 0_base_settings directory."""
        self.logger.info("Copying t-route base settings...")
        base_settings_path = Path(self.config.get('SYMFLUENCE_CODE_DIR')) / '0_base_settings' / 'troute'
        self.troute_setup_dir.mkdir(parents=True, exist_ok=True)
        
        if not base_settings_path.exists():
            self.logger.warning(f"Base settings directory not found at {base_settings_path}. Skipping copy.")
            return

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.troute_setup_dir / file)
        self.logger.info("t-route base settings copied.")

    def create_troute_topology_file(self):
        """
        Creates the NetCDF network topology file using t-route's expected NWM variable names.
        """
        self.logger.info("Creating t-route specific network topology file...")

        # Define paths using SYMFLUENCE conventions
        river_network_path = self.project_dir / 'shapefiles/river_network'
        river_network_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_{self.config.get('DOMAIN_DEFINITION_METHOD','delineate')}.shp"
        river_basin_path = self.project_dir / 'shapefiles/river_basins'
        river_basin_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        topology_name = self.config.get('SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc')
        topology_filepath = self.troute_setup_dir / topology_name

        # Load shapefiles
        shp_river = gpd.read_file(river_network_path / river_network_name)
        shp_basin = gpd.read_file(river_basin_path / river_basin_name)

        with nc4.Dataset(topology_filepath, 'w', format='NETCDF4') as ncid:
            # Set global attributes
            ncid.setncattr('Author', "Created by SYMFLUENCE workflow for t-route")
            ncid.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            ncid.setncattr('Conventions', "CF-1.6")

            # Create dimensions based on t-route standards
            ncid.createDimension('link', len(shp_river))
            ncid.createDimension('nhru', len(shp_basin))
            ncid.createDimension('gages', None) # Unlimited dimension for gages

            # Map SYMFLUENCE shapefile columns to t-route's required variable names
            self._create_and_fill_nc_var(ncid, 'comid', 'i4', 'link', shp_river[self.config.get('RIVER_NETWORK_SHP_SEGID')], 'Unique segment ID')
            self._create_and_fill_nc_var(ncid, 'to_node', 'i4', 'link', shp_river[self.config.get('RIVER_NETWORK_SHP_DOWNSEGID')], 'Downstream segment ID')
            self._create_and_fill_nc_var(ncid, 'length', 'f8', 'link', shp_river[self.config.get('RIVER_NETWORK_SHP_LENGTH')], 'Segment length', 'meters')
            self._create_and_fill_nc_var(ncid, 'slope', 'f8', 'link', shp_river[self.config.get('RIVER_NETWORK_SHP_SLOPE')], 'Segment slope', 'm/m')
            self._create_and_fill_nc_var(ncid, 'link_id_hru', 'i4', 'nhru', shp_basin[self.config.get('RIVER_BASIN_SHP_HRU_TO_SEG')], 'Segment ID for HRU discharge')
            self._create_and_fill_nc_var(ncid, 'hru_area_m2', 'f8', 'nhru', shp_basin[self.config.get('RIVER_BASIN_SHP_AREA')], 'HRU area', 'm^2')

            # Add required placeholder variables with sensible defaults
            shp_river['lat'] = shp_river.geometry.centroid.y
            shp_river['lon'] = shp_river.geometry.centroid.x
            self._create_and_fill_nc_var(ncid, 'lat', 'f8', 'link', shp_river['lat'], 'Latitude of segment midpoint', 'degrees_north')
            self._create_and_fill_nc_var(ncid, 'lon', 'f8', 'link', shp_river['lon'], 'Longitude of segment midpoint', 'degrees_east')
            self._create_and_fill_nc_var(ncid, 'alt', 'f8', 'link', [0.0] * len(shp_river), 'Mean elevation of segment', 'meters')
            self._create_and_fill_nc_var(ncid, 'from_node', 'i4', 'link', [0] * len(shp_river), 'Upstream node ID')
            self._create_and_fill_nc_var(ncid, 'n', 'f8', 'link', [0.035] * len(shp_river), 'Mannings roughness coefficient')

        self.logger.info(f"t-route topology file created at {topology_filepath}")

    def create_troute_yaml_config(self):
        """Creates the t-route YAML configuration file from SYMFLUENCE config settings."""
        self.logger.info("Creating t-route YAML configuration file...")

        # Determine paths and parameters from config
        source_model = self.config.get('TROUTE_FROM_MODEL', 'SUMMA').upper()
        experiment_id = self.config.get('EXPERIMENT_ID')
        input_dir = self.project_dir / f"simulations/{experiment_id}" / source_model
        output_dir = self.project_dir / f"simulations/{experiment_id}" / 'troute'
        topology_name = self.config.get('SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc')

        # Calculate nts (Number of Timesteps)
        start_dt = datetime.fromisoformat(self.config.get('EXPERIMENT_TIME_START'))
        end_dt = datetime.fromisoformat(self.config.get('EXPERIMENT_TIME_END'))
        time_step_seconds = int(self.config.get('SETTINGS_TROUTE_DT_SECONDS', 3600))
        total_seconds = (end_dt - start_dt).total_seconds() + time_step_seconds
        nts = int(total_seconds / time_step_seconds)

        # Build configuration dictionary matching t-route's schema
        config_dict = {
            'log_parameters': {'showtiming': True, 'log_level': 'DEBUG'},
            'network_topology_parameters': {'supernetwork_parameters': {'geo_file_path': str(self.troute_setup_dir / topology_name)}},
            'compute_parameters': {
                'restart_parameters': {'start_datetime': self.config.get('EXPERIMENT_TIME_START')},
                'forcing_parameters': {
                    'nts': nts,
                    'qlat_input_folder': str(input_dir),
                    'qlat_file_pattern_filter': f"{experiment_id}_timestep.nc"
                }
            },
            'output_parameters': {'stream_output': {'stream_output_directory': str(output_dir)}}
        }

        # Write dictionary to YAML file
        yaml_filename = self.config.get('SETTINGS_TROUTE_CONFIG_FILE', 'troute_config.yml')
        yaml_filepath = self.troute_setup_dir / yaml_filename
        with open(yaml_filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        self.logger.info(f"t-route YAML config written to {yaml_filepath}")

    def _create_and_fill_nc_var(self, ncid, var_name, var_type, dim, data, long_name, units='-'):
        """Helper to create and fill a NetCDF variable."""
        var = ncid.createVariable(var_name, var_type, (dim,))
        var[:] = data.values if hasattr(data, 'values') else data
        var.long_name = long_name
        var.units = units


class TRouteRunner:
    """A standalone runner for the t-route model."""
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

    def run_troute(self):
        """
        Prepares runoff data and executes t-route as a Python module.
        """
        self.logger.info("--- Starting t-route Run ---")
        
        # 1. Prepare runoff file by renaming the variable
        self._prepare_runoff_file()
        
        # 2. Set up paths for execution
        settings_path = self.project_dir / 'settings' / 'troute'
        config_file = self.config.get('SETTINGS_TROUTE_CONFIG_FILE', 'troute_config.yml')
        config_filepath = settings_path / config_file
        experiment_id = self.config.get('EXPERIMENT_ID')
        troute_out_path = self.project_dir / f"simulations/{experiment_id}/troute"
        log_path = troute_out_path / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / "troute_run.log"

        # 3. Construct and run the command
        command = f"python -m nwm_routing {config_filepath}"
        self.logger.info(f'Executing t-route command: {command}')

        try:
            with open(log_file_path, 'w') as log_file:
                subprocess.run(command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info(f"t-route run completed successfully. Log file available at: {log_file_path}")
            self.logger.info("--- t-route Run Finished ---")
            return troute_out_path
        except subprocess.CalledProcessError:
            self.logger.error(f"t-route run failed. See full log for details: {log_file_path}")
            if log_file_path.exists():
                with open(log_file_path, 'r') as f:
                    self.logger.error("--- t-route Log Output ---\n" + f.read())
            raise

    def _prepare_runoff_file(self):
        """
        Loads the hydrological model output and renames the runoff variable
        to 'q_lateral' as required by t-route.
        """
        self.logger.info("Preparing runoff file for t-route...")

        source_model = self.config.get('TROUTE_FROM_MODEL', 'SUMMA').upper()
        experiment_id = self.config.get('EXPERIMENT_ID')
        runoff_filepath = self.project_dir / f"simulations/{experiment_id}/{source_model}/{experiment_id}_timestep.nc"

        if not runoff_filepath.exists():
            self.logger.error(f"Runoff file not found at {runoff_filepath}.")
            raise FileNotFoundError(f"Runoff file not found: {runoff_filepath}")

        # Fetch the original runoff variable name from config
        original_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
        
        self.logger.debug(f"Checking for variable '{original_var}' in {runoff_filepath}")

        ds = xr.open_dataset(runoff_filepath)
        if original_var in ds.data_vars:
            self.logger.info(f"Found '{original_var}', renaming to 'q_lateral'.")
            ds = ds.rename({original_var: 'q_lateral'})
            ds.to_netcdf(runoff_filepath, 'w', format='NETCDF4')
            self.logger.info("Runoff variable successfully renamed.")
        elif 'q_lateral' in ds.data_vars:
            self.logger.info("Runoff variable is already named 'q_lateral'. No action needed.")
        else:
            ds.close()
            self.logger.error(f"Expected runoff variable '{original_var}' not found in {runoff_filepath}.")
            raise ValueError(f"Runoff variable not found in {runoff_filepath}")
        ds.close()