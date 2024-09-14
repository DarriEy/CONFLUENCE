import os
import sys
from shutil import rmtree, copyfile
import glob
import easymore # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import netCDF4 as nc4 # type: ignore
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from shutil import copyfile
import rasterstats # type: ignore
from pyproj import CRS, Transformer # type: ignore
from shapely.geometry import Polygon # type: ignore
import time
import tempfile
import shutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging_utils import get_function_logger # type: ignore

class SummaPreProcessor_spatial:
    def __init__(self, config: Dict[str, Any], logger: Any):
        
        """

        Initialize the SummaPreProcessor_spatial class.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing setup parameters.
            logger (Any): Logger object for recording processing information.

        Attributes:
            config (Dict[str, Any]): Stored configuration dictionary.
            logger (Any): Stored logger object.
            project_dir (Path): Path to the project directory.
            summa_setup_dir (Path): Path to the SUMMA setup directory.
            hruId (str): HRU identifier from config.
            gruId (str): GRU identifier from config.
            domain_name (str): Name of the domain being processed.
            merged_forcing_path (Path): Path to merged forcing data.
            shapefile_path (Path): Path to shapefiles.
            dem_path (Path): Path to DEM file.
            forcing_basin_path (Path): Path to basin-averaged forcing data.
            forcing_summa_path (Path): Path to SUMMA input forcing data.
            catchment_path (Path): Path to catchment shapefile.
            catchment_name (str): Name of the catchment shapefile.
            forcing_dataset (str): Name of the forcing dataset.
            data_step (int): Time step size for forcing data.
            settings_path (Path): Path to SUMMA settings.
            coldstate_name (str): Name of the cold state file.
            parameter_name (str): Name of the trial parameters file.
            attribute_name (str): Name of the attributes file.
            forcing_measurement_height (float): Measurement height for forcing data.

        """

        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.summa_setup_dir = self.project_dir / "settings" / "SUMMA"
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')
        self.gruId = self.config.get('CATCHMENT_SHP_GRUID')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        self.dem_path = self._get_default_path('DEM_PATH', f'attributes/elevation/dem/{self.config.get('DEM_NAME')}')
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_summa_path = self.project_dir / 'forcing' / 'SUMMA_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        self.forcing_dataset = self.config.get('FORCING_DATASET').lower()
        self.data_step = int(self.config.get('FORCING_TIME_STEP_SIZE'))
        self.settings_path = self.project_dir / 'settings/SUMMA'
        self.coldstate_name = self.config.get('SETTINGS_SUMMA_COLDSTATE')
        self.parameter_name = self.config.get('SETTINGS_SUMMA_TRIALPARAMS')
        self.attribute_name = self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        self.forcing_measurement_height = float(self.config.get('FORCING_MEASUREMENT_HEIGHT'))
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_data')


    @get_function_logger
    def run_preprocessing(self):
        """
        Run the complete SUMMA spatial preprocessing workflow.

        This method orchestrates the entire preprocessing pipeline, including:
        1. Sorting the catchment shape
        2. Processing forcing data
        3. Copying base settings
        4. Creating the various SUMMA configuration files

        The method uses a function logger decorator for detailed logging.

        Raises:
            Exception: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting SUMMA spatial preprocessing")
        
        try:
            self.sort_catchment_shape(work_log_dir=self.project_dir / f"shapefiles/_workLog")
            self.process_forcing_data(work_log_dir=self.project_dir / f"forcing/_workLog")
            self.copy_base_settings(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")
            self.create_file_manager(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")
            self.create_forcing_file_list(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")
            self.create_initial_conditions(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")
            self.create_trial_parameters(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")
            self.create_attributes_file(work_log_dir=self.project_dir / f"settings/SUMMA/_workLog")

            self.logger.info("SUMMA spatial preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during SUMMA spatial preprocessing: {str(e)}")
            raise

    @get_function_logger
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

    @get_function_logger
    def process_forcing_data(self):

        """
        Process the forcing data for SUMMA.

        This method orchestrates the following steps:
        1. Merge forcings if the dataset is RDRS
        2. Create a shapefile for the forcing data
        3. Remap the forcing data
        4. Apply lapse rate correction if configured
        5. Apply timestep to the forcing data

        Each step is conditionally executed based on configuration settings.

        Raises:
            Exception: If any step in the forcing data processing fails.
        """

        self.logger.info("Starting forcing data processing")

        try:
            if self.config.get('FORCING_DATASET') == 'RDRS':
                self.merge_forcings(work_log_dir=self.project_dir / f"forcing/_workLog")
                self.logger.info("Forcings merged successfully")

            #self.convert_units_and_vars()
            self.create_shapefile(work_log_dir=self.project_dir / f"shapefiles/_workLog")
            self.logger.info("Shapefile created successfully")

            self.remap_forcing(work_log_dir=self.project_dir / f"forcing/_workLog")
            self.logger.info("Forcing data remapped successfully")

            if self.config.get('APPLY_LAPSE_RATE', False):
                self.apply_lapse_rate(work_log_dir=self.project_dir / f"forcing/_workLog")
                self.logger.info("Lapse rate correction applied successfully")

            self.apply_timestep(work_log_dir=self.project_dir / f"forcing/_workLog")
            self.logger.info("Timestep applied to forcing data successfully")

            self.logger.info("Forcing data processing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during forcing data processing: {str(e)}")
            raise

    @get_function_logger
    def copy_base_settings(self):
        """
        Copy SUMMA base settings from the source directory to the project's settings directory.

        This method performs the following steps:
        1. Determines the source directory for base settings
        2. Determines the destination directory for settings
        3. Creates the destination directory if it doesn't exist
        4. Copies all files from the source directory to the destination directory

        Raises:
            FileNotFoundError: If the source directory or any source file is not found.
            PermissionError: If there are permission issues when creating directories or copying files.
        """
        self.logger.info("Copying SUMMA base settings")
        
        base_settings_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_base_settings' / 'SUMMA'
        settings_path = self._get_default_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA')
        
        try:
            settings_path.mkdir(parents=True, exist_ok=True)
            
            for file in os.listdir(base_settings_path):
                source_file = base_settings_path / file
                dest_file = settings_path / file
                copyfile(source_file, dest_file)
                self.logger.debug(f"Copied {source_file} to {dest_file}")
            
            self.logger.info(f"SUMMA base settings copied to {settings_path}")
        except FileNotFoundError as e:
            self.logger.error(f"Source file or directory not found: {e}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission error when copying files: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error copying base settings: {e}")
            raise

    @get_function_logger
    def create_file_manager(self):
        """
        Create the SUMMA file manager configuration file.

        This method generates a file manager configuration for SUMMA, including:
        - Control version
        - Simulation start and end times
        - Output file prefix
        - Paths for various settings and data files

        The method uses configuration values and default paths where appropriate.

        Raises:
            ValueError: If required configuration values are missing or invalid.
            IOError: If there's an error writing the file manager configuration.
        """
        self.logger.info("Creating SUMMA file manager")

        try:
            experiment_id = self.config.get('EXPERIMENT_ID')
            if not experiment_id:
                raise ValueError("EXPERIMENT_ID is missing from configuration")

            self.sim_start, self.sim_end = self._get_simulation_times()

            filemanager_name = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
            if not filemanager_name:
                raise ValueError("SETTINGS_SUMMA_FILEMANAGER is missing from configuration")

            filemanager_path = self.summa_setup_dir / filemanager_name

            with open(filemanager_path, 'w') as fm:
                fm.write(f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n")
                fm.write(f"simStartTime         '{self.sim_start}'\n")
                fm.write(f"simEndTime           '{self.sim_end}'\n")
                fm.write(f"tmZoneInfo           'utcTime'\n")
                fm.write(f"outFilePrefix        '{experiment_id}'\n")
                fm.write(f"settingsPath         '{self._get_default_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA')}/'\n")
                fm.write(f"forcingPath          '{self._get_default_path('FORCING_SUMMA_PATH', 'forcing/SUMMA_input')}/'\n")
                fm.write(f"outputPath           '{self.project_dir / 'simulations' / experiment_id / 'SUMMA'}/'\n")

                fm.write(f"initConditionFile    '{self.config.get('SETTINGS_SUMMA_COLDSTATE')}'\n")
                fm.write(f"attributeFile        '{self.config.get('SETTINGS_SUMMA_ATTRIBUTES')}'\n")
                fm.write(f"trialParamFile       '{self.config.get('SETTINGS_SUMMA_TRIALPARAMS')}'\n")
                fm.write(f"forcingListFile      '{self.config.get('SETTINGS_SUMMA_FORCING_LIST')}'\n")
                fm.write(f"decisionsFile        'modelDecisions.txt'\n")
                fm.write(f"outputControlFile    'outputControl.txt'\n")
                fm.write(f"globalHruParamFile   'localParamInfo.txt'\n")
                fm.write(f"globalGruParamFile   'basinParamInfo.txt'\n")
                fm.write(f"vegTableFile         'TBL_VEGPARM.TBL'\n")
                fm.write(f"soilTableFile        'TBL_SOILPARM.TBL'\n")
                fm.write(f"generalTableFile     'TBL_GENPARM.TBL'\n")
                fm.write(f"noahmpTableFile      'TBL_MPTABLE.TBL'\n")

            self.logger.info(f"SUMMA file manager created at {filemanager_path}")

        except ValueError as ve:
            self.logger.error(f"Configuration error: {str(ve)}")
            raise
        except IOError as io_err:
            self.logger.error(f"Error writing file manager configuration: {str(io_err)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in create_file_manager: {str(e)}")
            raise

    def convert_units_and_vars(self):
        self.logger.info(f"Starting in-place variable renaming and unit conversion for {self.forcing_dataset} data")

        variable_mapping = {
            'RDRS': {
                'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
                'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
                'RDRS_v2.1_A_PR0_SFC': 'pptrate',
                'RDRS_v2.1_P_P0_SFC': 'airpres',
                'RDRS_v2.1_P_TT_1.5m': 'airtemp',
                'RDRS_v2.1_P_HU_1.5m': 'spechum',
                'RDRS_v2.1_P_UVC_10m': 'windspd',
            },
            'ERA5': {
                't2m': 'airtemp',
                'd2m': 'dewpoint',
                'sp': 'airpres',
                'tp': 'pptrate',
                'ssrd': 'SWRadAtm',
                'strd': 'LWRadAtm',
                'u10': 'windspd_u',
                'v10': 'windspd_v'
            },
            'CARRA': {
                '2t': 'airtemp',
                'sp': 'airpres',
                '2sh': 'spechum',
                'ssr': 'SWRadAtm',
                'strd': 'LWRadAtm',
                'tp': 'pptrate',
                '10u': 'windspd_u',
                '10v': 'windspd_v'
            }
        }

        forcing_files = sorted(self.merged_forcing_path.glob(f"{self.config.get('FORCING_DATASET')}_*.nc"))

        for file in forcing_files:
            self.logger.info(f"Processing {file.name}")
            try:
                with xr.open_dataset(file) as ds:
                    # Rename variables
                    ds = ds.rename(variable_mapping[self.forcing_dataset.upper()])

                    # Apply unit conversions
                    if self.forcing_dataset == 'rdrs':
                        ds['airpres'] = ds['airpres'] * 100  # Convert from mb to Pa
                        ds['airtemp'] = ds['airtemp'] + 273.15  # Convert from deg_C to K
                        ds['pptrate'] = ds['pptrate'] / 3600 * 1000  # Convert from m/hour to mm/s
                        ds['windspd'] = ds['windspd'] * 0.514444  # Convert from knots to m/s
                    elif self.forcing_dataset == 'era5':
                        ds['airtemp'] = ds['airtemp']  # Already in K
                        ds['airpres'] = ds['airpres']  # Already in Pa
                        ds['pptrate'] = ds['pptrate'] * 1000 / 3600  # Convert from m/hour to mm/s
                        ds['windspd'] = np.sqrt(ds['windspd_u']**2 + ds['windspd_v']**2)
                    elif self.forcing_dataset == 'carra':
                        # Add CARRA-specific conversions if needed
                        pass

                    # Update attributes
                    ds['airpres'].attrs.update({'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'})
                    ds['airtemp'].attrs.update({'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'})
                    ds['pptrate'].attrs.update({'units': 'mm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'})
                    ds['windspd'].attrs.update({'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'})
                    ds['LWRadAtm'].attrs.update({'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'})
                    ds['SWRadAtm'].attrs.update({'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'})
                    if 'spechum' in ds:
                        ds['spechum'].attrs.update({'long_name': 'specific humidity', 'standard_name': 'specific_humidity'})

                    # Add a global attribute to track the conversion
                    ds.attrs['converted'] = 'True'
                    ds.attrs['conversion_time'] = str(pd.Timestamp.now())

                    # Save the converted file back to the original location
                    ds.to_netcdf(file)
                    self.logger.info(f"Converted file saved back to {file}")

            except Exception as e:
                self.logger.error(f"Error processing file {file}: {str(e)}")

        self.logger.info(f"Completed in-place variable renaming and unit conversion for {self.forcing_dataset} data")

    @get_function_logger
    def remap_forcing(self):
        """
        Remap forcing data to the catchment shapefile using EASYMORE.

        This method performs the following steps:
        1. Create one weighted forcing file
        2. Create all weighted forcing files

        The remapping process uses the EASYMORE library to perform area-weighted
        remapping from the forcing grid to the catchment polygons.

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If there are issues with EASYMORE configuration or execution.
            IOError: If there are issues writing output files.
        """
        self.logger.info("Starting forcing remapping process")

        # Step 1: Create one weighted forcing file
        self._create_one_weighted_forcing_file()

        # Step 2: Create all weighted forcing files
        self._create_all_weighted_forcing_files()

        self.logger.info("Forcing remapping process completed")

    def _create_one_weighted_forcing_file(self):
        self.logger.info("Creating one weighted forcing file")

        # Initialize EASYMORE object
        esmr = easymore.easymore()

        esmr.author_name = 'SUMMA public workflow scripts'
        esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
        esmr.case_name = self.config.get('DOMAIN_NAME')

        # Set up source and target shapefiles
        esmr.source_shp = self.project_dir / 'shapefiles' / 'forcing' / f'forcing_{self.config.get('FORCING_DATASET')}.shp'
        esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
        esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')

        esmr.target_shp = self.catchment_path / self.catchment_name
        esmr.target_shp_ID = self.config.get('CATCHMENT_SHP_HRUID')
        esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
        esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')

        # Set up source netcdf file
        forcing_files = sorted([f for f in self.merged_forcing_path.glob('*.nc')])

        if self.config.get('FORCING_DATASET') == 'RDRS':
            var_lat = 'lat'
            var_lon = 'lon'
        else:
            var_lat = 'latitude'
            var_lon = 'longitude'

        esmr.source_nc = str(forcing_files[0])
        esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
        esmr.var_lat = var_lat 
        esmr.var_lon = var_lon
        esmr.var_time = 'time'

        # Set up temporary and output directories
        esmr.temp_dir = str(self.project_dir / 'forcing' / 'temp_easymore') + '/'
        esmr.output_dir = str(self.forcing_basin_path) + '/'

        # Set up netcdf settings
        esmr.remapped_dim_id = 'hru'
        esmr.remapped_var_id = 'hruId'
        esmr.format_list = ['f4']
        esmr.fill_value_list = ['-9999']

        esmr.save_csv = False
        esmr.remap_csv = ''
        esmr.sort_ID = False

        # Run EASYMORE
        esmr.nc_remapper()

        # Move files to prescribed locations
        remap_file = f"{esmr.case_name}_remapping.csv"
        self.intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        self.intersect_path.mkdir(parents=True, exist_ok=True)
        copyfile(os.path.join(esmr.temp_dir, remap_file), self.intersect_path / remap_file)

        for file in glob.glob(os.path.join(esmr.temp_dir, f"{esmr.case_name}_intersected_shapefile.*")):
            copyfile(file, self.intersect_path / os.path.basename(file))

        # Remove temporary directory
        rmtree(esmr.temp_dir, ignore_errors=True)

        self.logger.info("One weighted forcing file created")

    def _create_all_weighted_forcing_files(self):
        self.logger.info("Creating all weighted forcing files")

        if self.config.get('FORCING_DATASET') == 'RDRS':
            var_lat = 'lat'
            var_lon = 'lon'
        else:
            var_lat = 'latitude'
            var_lon = 'longitude'

        # Initialize EASYMORE object
        esmr = easymore.easymore()

        esmr.author_name = 'SUMMA public workflow scripts'
        esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
        esmr.case_name = f'{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}'

        esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
        esmr.var_lat = var_lat
        esmr.var_lon = var_lon
        esmr.var_time = 'time'

        esmr.temp_dir = ''
        esmr.output_dir = str(self.forcing_basin_path) + '/'

        esmr.remapped_dim_id = 'hru'
        esmr.remapped_var_id = 'hruId'
        esmr.format_list = ['f4']
        esmr.fill_value_list = ['-9999']

        esmr.save_csv = False
        esmr.remap_csv = str(self.intersect_path / f"{self.config.get('DOMAIN_NAME')}_remapping.csv")
        esmr.sort_ID = False
        esmr.overwrite_existing_remap = False

        # Process remaining forcing files
        forcing_files = sorted([f for f in self.merged_forcing_path.glob('*.nc')])
        for file in forcing_files[1:]:
            esmr.source_nc = str(file)
            esmr.nc_remapper()

        self.logger.info("All weighted forcing files created")

    @get_function_logger
    def apply_lapse_rate(self):
        """
        Apply temperature lapse rate corrections to the forcing data.

        This method performs the following steps:
        1. Load area-weighted information for each basin
        2. Calculate lapse rate corrections for each HRU
        3. Apply lapse rate corrections to temperature data in each forcing file
        4. Save the corrected forcing data

        The lapse rate is applied based on the elevation difference between the forcing data
        grid cells and the mean elevation of each HRU.

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If there are issues with data processing or lapse rate application.
            IOError: If there are issues reading or writing data files.
        """
        self.logger.info("Starting to apply temperature lapse rate and add data step")

        # Find intersection file
        intersect_name = f"{self.domain_name}_intersected_shapefile.csv"
        
        # Get forcing files
        forcing_files = [f for f in os.listdir(self.forcing_basin_path) if f.startswith(f"{self.domain_name}_{self.config.get('FORCING_DATASET')}") and f.endswith('.nc')]
        forcing_files.sort()

        # Prepare output directory
        self.forcing_summa_path.mkdir(parents=True, exist_ok=True)

        # Load area-weighted information for each basin
        topo_data = pd.read_csv(self.intersect_path / intersect_name)

        # Specify column names
        gru_id = f'S_1_{self.gruId}'
        hru_id = f'S_1_{self.hruId}'
        forcing_id = 'S_2_ID'
        catchment_elev = 'S_1_elev_mean'
        forcing_elev = 'S_2_elev_m'
        weights = 'weight'

        # Define lapse rate
        lapse_rate = float(self.config.get('LAPSE_RATE'))  # [K m-1]
        # Calculate weighted lapse values for each HRU
        topo_data['lapse_values'] = topo_data[weights] * lapse_rate * (topo_data[forcing_elev] - topo_data[catchment_elev])

        # Find total lapse value per basin
        if gru_id == hru_id:
            lapse_values = topo_data.groupby([hru_id]).lapse_values.sum().reset_index()
        else:
            lapse_values = topo_data.groupby([gru_id, hru_id]).lapse_values.sum().reset_index()

        # Sort and set hruID as the index variable
        lapse_values = lapse_values.sort_values(hru_id).set_index(hru_id)

        # Process each forcing file
        for file in forcing_files:
            self.logger.info(f"Processing {file}")
            output_file = self.forcing_summa_path / file
            if output_file.exists():
                self.logger.info(f"{file} already exists ... skipping")
                continue

            with xr.open_dataset(self.forcing_basin_path / file) as dat:
                # Temperature lapse rates
                lapse_values_sorted = lapse_values['lapse_values'].loc[dat['hruId'].values]
                addThis = xr.DataArray(np.tile(lapse_values_sorted.values, (len(dat['time']), 1)), dims=('time', 'hru'))

                # Get air temperature attributes
                tmp_units = dat['airtemp'].units

                # Apply lapse rate correction
                dat['airtemp'] = dat['airtemp'] + addThis
                dat.airtemp.attrs['units'] = tmp_units
                # Save to file in new location
                dat.to_netcdf(output_file)

        self.logger.info(f"Completed processing of {self.forcing_dataset.upper()} forcing files with temperature lapsing")

    @get_function_logger
    def create_shapefile(self):
        """
        Create a shapefile for the RDRS forcing data.

        This method performs the following steps:
        1. Find the first RDRS monthly file
        2. Extract latitude, longitude, and other relevant information
        3. Create polygons for each grid cell
        4. Calculate zonal statistics (mean elevation) for each grid cell
        5. Create a GeoDataFrame with the extracted information
        6. Save the GeoDataFrame as a shapefile

        Raises:
            FileNotFoundError: If no RDRS monthly file or DEM file is found.
            ValueError: If there are issues with data extraction or processing.
            IOError: If there are issues writing the shapefile.
        """
        
        

        if self.config.get('FORCING_DATASET') == 'RDRS':
            self.logger.info("Starting to create RDRS shapefile")

            # Find the first monthly file
            forcing_file = next((f for f in os.listdir(self.merged_forcing_path) if f.endswith('.nc') and f.startswith('RDRS_monthly_')), None)
            
            if not forcing_file:
                self.logger.error("No RDRS monthly file found")
                return

            # Read the forcing file
            with xr.open_dataset(self.merged_forcing_path / forcing_file) as ds:
                rlat = ds.rlat.values
                rlon = ds.rlon.values
                lat = ds.lat.values
                lon = ds.lon.values

            # Create lists to store the data
            geometries = []
            ids = []
            lats = []
            lons = []

            for i in range(len(rlat)):
                for j in range(len(rlon)):
                    # Get the corners of the grid cell in rotated coordinates
                    rlat_corners = [rlat[i], rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i]]
                    rlon_corners = [rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j]]
                    
                    # Convert rotated coordinates to lat/lon
                    lat_corners = [lat[i,j], lat[i,j+1] if j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j], 
                                lat[i+1,j] if i+1 < len(rlat) else lat[i,j]]
                    lon_corners = [lon[i,j], lon[i,j+1] if j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j], 
                                lon[i+1,j] if i+1 < len(rlat) else lon[i,j]]
                    
                    # Create polygon
                    poly = Polygon(zip(lon_corners, lat_corners))
                    
                    # Append to lists
                    geometries.append(poly)
                    ids.append(i * len(rlon) + j)
                    lats.append(lat[i,j])
                    lons.append(lon[i,j])

            # Create the GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): lons,
            }, crs='EPSG:4326')

            # Calculate zonal statistics (mean elevation) for each grid cell
            zs = rasterstats.zonal_stats(gdf, str(self.dem_path), stats=['mean'])

            # Add mean elevation to the GeoDataFrame
            gdf['elev_m'] = [item['mean'] for item in zs]

            # Drop columns that are on the edge and don't have elevation data
            gdf.dropna(subset=['elev_m'], inplace=True)

            # Save the shapefile
            self.shapefile_path.mkdir(parents=True, exist_ok=True)
            output_shapefile = self.shapefile_path / f'forcing_{self.config.get('FORCING_DATASET')}.shp'
            gdf.to_file(output_shapefile)

            self.logger.info(f"RDRS shapefile created and saved to {output_shapefile}")

        elif self.config.get('FORCING_DATASET') == 'ERA5':
            self.logger.info("Creating ERA5 shapefile")

            # Find an .nc file in the forcing path
            forcing_files = list(self.merged_forcing_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No ERA5 forcing files found")
            forcing_file = forcing_files[0]

            # Set the dimension variable names
            source_name_lat = "latitude"
            source_name_lon = "longitude"

            # Open the file and get the dimensions and spatial extent of the domain
            with xr.open_dataset(forcing_file) as src:
                lat = src[source_name_lat].values
                lon = src[source_name_lon].values

            # Find the spacing
            half_dlat = abs(lat[1] - lat[0])/2
            half_dlon = abs(lon[1] - lon[0])/2

            # Create lists to store the data
            geometries = []
            ids = []
            lats = []
            lons = []

            for i, center_lon in enumerate(lon):
                for j, center_lat in enumerate(lat):
                    vertices = [
                        [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                        [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                        [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                        [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                        [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                    ]
                    geometries.append(Polygon(vertices))
                    ids.append(i * len(lat) + j)
                    lats.append(float(center_lat))
                    lons.append(float(center_lon))

            # Create the GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): lons,
            }, crs='EPSG:4326')

            # Calculate zonal statistics (mean elevation) for each grid cell
            zs = rasterstats.zonal_stats(gdf, str(self.dem_path), stats=['mean'])

            # Add mean elevation to the GeoDataFrame
            gdf['elev_m'] = [item['mean'] for item in zs]

            # Drop columns that are on the edge and don't have elevation data
            gdf.dropna(subset=['elev_m'], inplace=True)

            # Save the shapefile
            self.shapefile_path.mkdir(parents=True, exist_ok=True)
            output_shapefile = self.shapefile_path / f'forcing_{self.config.get('FORCING_DATASET')}.shp'
            gdf.to_file(output_shapefile)

            self.logger.info(f"ERA5 shapefile created and saved to {output_shapefile}")

    @get_function_logger
    def apply_timestep(self):
        """
        Add data step (time step) information to the forcing files.

        This method performs the following steps:
        1. Identify all forcing files in the SUMMA input directory
        2. For each file, add a 'data_step' variable with the configured time step
        3. Save the modified files, replacing the originals

        The data step is added as a new variable to ensure SUMMA can correctly
        interpret the temporal resolution of the forcing data.

        Raises:
            FileNotFoundError: If no forcing files are found.
            ValueError: If there are issues with data processing or time step application.
            IOError: If there are issues reading or writing data files.
        """
        self.logger.info("Starting to apply data step to forcing files")

        forcing_files = [f for f in self.forcing_summa_path.glob(f"{self.domain_name}_{self.config.get('FORCING_DATASET')}*.nc")]

        for file in forcing_files:
            self.logger.info(f"Processing {file}")
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name

            # Process the file using the temporary path
            with xr.open_dataset(file) as dat:
                dat['data_step'] = self.data_step
                dat.data_step.attrs['long_name'] = 'data step length in seconds'
                dat.data_step.attrs['units'] = 's'
                dat.to_netcdf(temp_path)
                
            # Try to replace the original file with the temporary file
            shutil.move(temp_path, file)
            self.logger.info(f"Successfully processed {file.name} using a temporary file")


        self.logger.info("Completed adding data step to forcing files")

    @get_function_logger
    def merge_forcings(self):
        """
        Merge RDRS forcing data files into monthly files.

        This method performs the following steps:
        1. Determine the year range for processing
        2. Create output directories
        3. Process each year and month, merging daily files into monthly files
        4. Apply variable renaming and unit conversions
        5. Save the merged and processed data to netCDF files

        Raises:
            FileNotFoundError: If required input files are missing.
            ValueError: If there are issues with data processing or merging.
            IOError: If there are issues writing output files.
        """
        self.logger.info("Starting to merge RDRS forcing data")

        
        years = [self.config.get('FORCING_START_YEAR'),self.config.get('FORCING_END_YEAR')]
        years = range(int(years[0])-1, int(years[1]) + 1)
        
        file_name_pattern = f"domain_{self.domain_name}_*.nc"
        
        self.merged_forcing_path = self.project_dir / 'forcing' / 'merged_path'
        raw_forcing_path = self.project_dir / 'forcing/raw_data/'
        self.merged_forcing_path.mkdir(parents=True, exist_ok=True)
        
        variable_mapping = {
            'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
            'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
            'RDRS_v2.1_A_PR0_SFC': 'pptrate',
            'RDRS_v2.1_P_P0_SFC': 'airpres',
            'RDRS_v2.1_P_TT_09944': 'airtemp',
            'RDRS_v2.1_P_HU_09944': 'spechum',
            'RDRS_v2.1_P_UVC_09944': 'windspd',
            'RDRS_v2.1_P_UUC_10m': 'windspd_u',
            'RDRS_v2.1_P_VVC_10m': 'windspd_v',
        }

        def process_rdrs_data(ds):
            existing_vars = {old: new for old, new in variable_mapping.items() if old in ds.variables}
            ds = ds.rename(existing_vars)
            
            if 'airpres' in ds:
                ds['airpres'] = ds['airpres'] * 100
                ds['airpres'].attrs.update({'units': 'Pa', 'long_name': 'air pressure', 'standard_name': 'air_pressure'})
            
            if 'airtemp' in ds:
                ds['airtemp'] = ds['airtemp'] + 273.15
                ds['airtemp'].attrs.update({'units': 'K', 'long_name': 'air temperature', 'standard_name': 'air_temperature'})
            
            if 'pptrate' in ds:
                ds['pptrate'] = ds['pptrate'] / 3600 * 1000
                ds['pptrate'].attrs.update({'units': 'm s-1', 'long_name': 'precipitation rate', 'standard_name': 'precipitation_rate'})
            
            if 'windspd' in ds:
                ds['windspd'] = ds['windspd'] * 0.514444
                ds['windspd'].attrs.update({'units': 'm s-1', 'long_name': 'wind speed', 'standard_name': 'wind_speed'})
            
            if 'LWRadAtm' in ds:
                ds['LWRadAtm'].attrs.update({'long_name': 'downward longwave radiation at the surface', 'standard_name': 'surface_downwelling_longwave_flux_in_air'})
            
            if 'SWRadAtm' in ds:
                ds['SWRadAtm'].attrs.update({'long_name': 'downward shortwave radiation at the surface', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'})
            
            if 'spechum' in ds:
                ds['spechum'].attrs.update({'long_name': 'specific humidity', 'standard_name': 'specific_humidity'})
            
            return ds

        for year in years:
            self.logger.info(f"Processing year {year}")
            year_folder = raw_forcing_path / str(year)

            for month in range(1, 13):
                self.logger.info(f"Processing {year}-{month:02d}")
                
                daily_files = sorted(year_folder.glob(file_name_pattern.replace('*', f'{year}{month:02d}*')))         

                if not daily_files:
                    self.logger.warning(f"No files found for {year}-{month:02d}")
                    continue
                
                datasets = []
                for file in daily_files:
                    try:
                        ds = xr.open_dataset(file)
                        datasets.append(ds)
                    except Exception as e:
                        self.logger.error(f"Error opening file {file}: {str(e)}")

                if not datasets:
                    self.logger.warning(f"No valid datasets for {year}-{month:02d}")
                    continue

                processed_datasets = []
                for ds in datasets:
                    try:
                        processed_ds = process_rdrs_data(ds)
                        processed_datasets.append(processed_ds)
                    except Exception as e:
                        self.logger.error(f"Error processing dataset: {str(e)}")

                if not processed_datasets:
                    self.logger.warning(f"No processed datasets for {year}-{month:02d}")
                    continue

                monthly_data = xr.concat(processed_datasets, dim="time")
                monthly_data = monthly_data.sortby("time")

                start_time = pd.Timestamp(year, month, 1)
                if month == 12:
                    end_time = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=1)
                else:
                    end_time = pd.Timestamp(year, month + 1, 1) - pd.Timedelta(hours=1)

                monthly_data = monthly_data.sel(time=slice(start_time, end_time))

                expected_times = pd.date_range(start=start_time, end=end_time, freq='h')
                monthly_data = monthly_data.reindex(time=expected_times, method='nearest')

                monthly_data['time'].encoding['units'] = 'hours since 1900-01-01'
                monthly_data['time'].encoding['calendar'] = 'gregorian'

                monthly_data.attrs.update({
                    'History': f'Created {time.ctime(time.time())}',
                    'Language': 'Written using Python',
                    'Reason': 'RDRS data aggregated to monthly files and variables renamed for SUMMA compatibility'
                })

                for var in monthly_data.data_vars:
                    monthly_data[var].attrs['missing_value'] = -999

                output_file = self.merged_forcing_path / f"RDRS_monthly_{year}{month:02d}.nc"
                monthly_data.to_netcdf(output_file)

                for ds in datasets:
                    ds.close()

        self.logger.info("RDRS forcing data merging completed")

    @get_function_logger
    def create_forcing_file_list(self):
        """
        Create a list of forcing files for SUMMA.

        This method performs the following steps:
        1. Determine the forcing dataset from the configuration
        2. Find all relevant forcing files in the SUMMA input directory
        3. Sort the files to ensure chronological order
        4. Write the sorted file list to a text file

        The resulting file list is used by SUMMA to locate and read the forcing data.

        Raises:
            FileNotFoundError: If no forcing files are found.
            IOError: If there are issues writing the file list.
            ValueError: If an unsupported forcing dataset is specified.
        """
        self.logger.info("Creating forcing file list")
        forcing_dataset = self.config.get('FORCING_DATASET')
        domain_name = self.config.get('DOMAIN_NAME')
        forcing_path = self.project_dir / 'forcing/SUMMA_input'
        file_list_path = self.summa_setup_dir / self.config.get('SETTINGS_SUMMA_FORCING_LIST')

        if forcing_dataset == 'CARRA':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_{forcing_dataset}") and f.endswith('.nc')]
        elif forcing_dataset == 'ERA5':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_{forcing_dataset}") and f.endswith('.nc')]
        elif forcing_dataset == 'RDRS':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{domain_name}_{forcing_dataset}") and f.endswith('.nc')]
        else:
            self.logger.error(f"Unsupported forcing dataset: {forcing_dataset}")
            raise ValueError(f"Unsupported forcing dataset: {forcing_dataset}")

        forcing_files.sort()

        with open(file_list_path, 'w') as f:
            for file in forcing_files:
                f.write(f"{file}\n")

        self.logger.info(f"Forcing file list created at {file_list_path}")

    @get_function_logger
    def create_initial_conditions(self):
        """
        Create the initial conditions (cold state) file for SUMMA.

        This method performs the following steps:
        1. Define the dimensions and variables for the cold state file
        2. Set default values for all state variables
        3. Create the netCDF file with the defined structure and values
        4. Ensure consistency with the forcing data (e.g., number of HRUs)

        The resulting file provides SUMMA with a starting point for model simulations.

        Raises:
            FileNotFoundError: If required input files (e.g., forcing file template) are not found.
            IOError: If there are issues creating or writing to the cold state file.
            ValueError: If there are inconsistencies between the cold state and forcing data.
        """
        self.logger.info("Creating initial conditions (cold state) file")
        self.logger.info("Creating initial conditions (cold state) file")

        # Find a forcing file to use as a template for hruId order
        forcing_files = list(self.forcing_summa_path.glob('*.nc'))
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        # Get the sorting order from the forcing file
        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)

        num_hru = len(forcing_hruIds)

        # Define the dimensions and fill values
        nSoil = 8
        nSnow = 0
        midSoil = 8
        midToto = 8
        ifcToto = midToto + 1
        scalarv = 1

        mLayerDepth = np.asarray([0.025, 0.075, 0.15, 0.25, 0.5, 0.5, 1, 1.5])
        iLayerHeight = np.asarray([0, 0.025, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 4])

        # States
        states = {
            'scalarCanopyIce': 0,
            'scalarCanopyLiq': 0,
            'scalarSnowDepth': 0,
            'scalarSWE': 0,
            'scalarSfcMeltPond': 0,
            'scalarAquiferStorage': 1.0,
            'scalarSnowAlbedo': 0,
            'scalarCanairTemp': 283.16,
            'scalarCanopyTemp': 283.16,
            'mLayerTemp': 283.16,
            'mLayerVolFracIce': 0,
            'mLayerVolFracLiq': 0.2,
            'mLayerMatricHead': -1.0
        }

        coldstate_path = self.settings_path / self.coldstate_name

        def create_and_fill_nc_var(nc, newVarName, newVarVal, fillDim1, fillDim2, newVarDim, newVarType, fillVal):
            if newVarName in ['iLayerHeight', 'mLayerDepth']:
                fillWithThis = np.full((fillDim1, fillDim2), newVarVal).transpose()
            else:
                fillWithThis = np.full((fillDim1, fillDim2), newVarVal)
            
            ncvar = nc.createVariable(newVarName, newVarType, (newVarDim, 'hru'), fill_value=fillVal)
            ncvar[:] = fillWithThis

        with nc4.Dataset(coldstate_path, "w", format="NETCDF4") as cs:
            # Set attributes
            cs.setncattr('Author', "Created by SUMMA workflow scripts")
            cs.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            cs.setncattr('Purpose', 'Create a cold state .nc file for initial SUMMA runs')

            # Define dimensions
            cs.createDimension('hru', num_hru)
            cs.createDimension('midSoil', midSoil)
            cs.createDimension('midToto', midToto)
            cs.createDimension('ifcToto', ifcToto)
            cs.createDimension('scalarv', scalarv)

            # Create variables
            var = cs.createVariable('hruId', 'i4', 'hru', fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of hydrological response unit (HRU)')
            var[:] = forcing_hruIds

            create_and_fill_nc_var(cs, 'dt_init', self.data_step, 1, num_hru, 'scalarv', 'f8', False)
            create_and_fill_nc_var(cs, 'nSoil', nSoil, 1, num_hru, 'scalarv', 'i4', False)
            create_and_fill_nc_var(cs, 'nSnow', nSnow, 1, num_hru, 'scalarv', 'i4', False)

            for var_name, var_value in states.items():
                if var_name.startswith('mLayer'):
                    create_and_fill_nc_var(cs, var_name, var_value, midToto, num_hru, 'midToto', 'f8', False)
                else:
                    create_and_fill_nc_var(cs, var_name, var_value, 1, num_hru, 'scalarv', 'f8', False)

            create_and_fill_nc_var(cs, 'iLayerHeight', iLayerHeight, num_hru, ifcToto, 'ifcToto', 'f8', False)
            create_and_fill_nc_var(cs, 'mLayerDepth', mLayerDepth, num_hru, midToto, 'midToto', 'f8', False)

        self.logger.info(f"Initial conditions file created at: {coldstate_path}")

    @get_function_logger
    def create_trial_parameters(self):
        """
        Create the trial parameters file for SUMMA.

        This method performs the following steps:
        1. Read trial parameter configurations from the main configuration
        2. Find a forcing file to use as a template for HRU order
        3. Create a netCDF file with the trial parameters
        4. Set the parameters for each HRU based on the configuration

        The resulting file provides SUMMA with parameter values to use in simulations.

        Raises:
            FileNotFoundError: If required input files (e.g., forcing file template) are not found.
            IOError: If there are issues creating or writing to the trial parameters file.
            ValueError: If there are inconsistencies in the parameter configurations.
        """
        self.logger.info("Creating trial parameters file")

        # Find a forcing file to use as a template for hruId order
        forcing_files = list(self.forcing_summa_path.glob('*.nc'))
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        # Get the sorting order from the forcing file
        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)

        num_hru = len(forcing_hruIds)

        # Read trial parameters from configuration
        num_tp = int(self.config.get('SETTINGS_SUMMA_TRIALPARAM_N', 0))
        all_tp = {}
        for i in range(num_tp):
            par_and_val = self.config.get(f'SETTINGS_SUMMA_TRIALPARAM_{i+1}')
            if par_and_val:
                arr = par_and_val.split(',')
                if len(arr) > 2:
                    val = np.array(arr[1:], dtype=np.float32)
                else:
                    val = float(arr[1])
                all_tp[arr[0]] = val

        parameter_path = self.settings_path / self.parameter_name

        with nc4.Dataset(parameter_path, "w", format="NETCDF4") as tp:
            # Set attributes
            tp.setncattr('Author', "Created by SUMMA workflow scripts")
            tp.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            tp.setncattr('Purpose', 'Create a trial parameter .nc file for initial SUMMA runs')

            # Define dimensions
            tp.createDimension('hru', num_hru)

            # Create hruId variable
            var = tp.createVariable('hruId', 'i4', 'hru', fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of hydrological response unit (HRU)')
            var[:] = forcing_hruIds

            # Create variables for specified trial parameters
            for var, val in all_tp.items():
                tp_var = tp.createVariable(var, 'f8', 'hru', fill_value=False)
                tp_var[:] = val

        self.logger.info(f"Trial parameters file created at: {parameter_path}")

    @get_function_logger
    def create_attributes_file(self):
        """
        Create the attributes file for SUMMA.

        This method performs the following steps:
        1. Load the catchment shapefile
        2. Get HRU order from a forcing file
        3. Create a netCDF file with HRU attributes
        4. Set attribute values for each HRU
        5. Insert soil class, land class, and elevation data
        6. Optionally set up HRU connectivity

        The resulting file provides SUMMA with essential information about each HRU.

        Raises:
            FileNotFoundError: If required input files are not found.
            IOError: If there are issues creating or writing to the attributes file.
            ValueError: If there are inconsistencies in the attribute data.
        """
        self.logger.info("Creating attributes file")

        # Load the catchment shapefile
        shp = gpd.read_file(self.catchment_path / self.catchment_name)

        # Get HRU order from a forcing file
        forcing_files = list(self.forcing_summa_path.glob('*.nc'))
        if not forcing_files:
            self.logger.error("No forcing files found in the SUMMA input directory")
            return
        forcing_file = forcing_files[0]

        with xr.open_dataset(forcing_file) as forc:
            forcing_hruIds = forc['hruId'].values.astype(int)

        # Sort shapefile based on forcing HRU order
        shp = shp.set_index(self.config.get('CATCHMENT_SHP_HRUID'))
        shp.index = shp.index.astype(int)
        shp = shp.loc[forcing_hruIds].reset_index()

        # Get number of GRUs and HRUs
        hru_ids = pd.unique(shp[self.config.get('CATCHMENT_SHP_HRUID')].values)
        gru_ids = pd.unique(shp[self.config.get('CATCHMENT_SHP_GRUID')].values)
        num_hru = len(hru_ids)
        num_gru = len(gru_ids)

        attribute_path = self.settings_path / self.attribute_name

        with nc4.Dataset(attribute_path, "w", format="NETCDF4") as att:
            # Set attributes
            att.setncattr('Author', "Created by SUMMA workflow scripts")
            att.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

            # Define dimensions
            att.createDimension('hru', num_hru)
            att.createDimension('gru', num_gru)

            # Define variables
            variables = {
                'hruId': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of hydrological response unit (HRU)'},
                'gruId': {'dtype': 'i4', 'dims': 'gru', 'units': '-', 'long_name': 'Index of grouped response unit (GRU)'},
                'hru2gruId': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of GRU to which the HRU belongs'},
                'downHRUindex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index of downslope HRU (0 = basin outlet)'},
                'longitude': {'dtype': 'f8', 'dims': 'hru', 'units': 'Decimal degree east', 'long_name': 'Longitude of HRU''s centroid'},
                'latitude': {'dtype': 'f8', 'dims': 'hru', 'units': 'Decimal degree north', 'long_name': 'Latitude of HRU''s centroid'},
                'elevation': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Mean HRU elevation'},
                'HRUarea': {'dtype': 'f8', 'dims': 'hru', 'units': 'm^2', 'long_name': 'Area of HRU'},
                'tan_slope': {'dtype': 'f8', 'dims': 'hru', 'units': 'm m-1', 'long_name': 'Average tangent slope of HRU'},
                'contourLength': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Contour length of HRU'},
                'slopeTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining slope'},
                'soilTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining soil type'},
                'vegTypeIndex': {'dtype': 'i4', 'dims': 'hru', 'units': '-', 'long_name': 'Index defining vegetation type'},
                'mHeight': {'dtype': 'f8', 'dims': 'hru', 'units': 'm', 'long_name': 'Measurement height above bare ground'},
            }

            for var_name, var_attrs in variables.items():
                var = att.createVariable(var_name, var_attrs['dtype'], var_attrs['dims'], fill_value=False)
                var.setncattr('units', var_attrs['units'])
                var.setncattr('long_name', var_attrs['long_name'])

            # Fill GRU variable
            att['gruId'][:] = gru_ids

            # Fill HRU variables
            for idx in range(num_hru):
                att['hruId'][idx] = shp.iloc[idx][self.config.get('CATCHMENT_SHP_HRUID')]
                att['HRUarea'][idx] = shp.iloc[idx][self.config.get('CATCHMENT_SHP_AREA')]
                att['latitude'][idx] = shp.iloc[idx][self.config.get('CATCHMENT_SHP_LAT')]
                att['longitude'][idx] = shp.iloc[idx][self.config.get('CATCHMENT_SHP_LON')]
                att['hru2gruId'][idx] = shp.iloc[idx][self.config.get('CATCHMENT_SHP_GRUID')]

                # Constants and placeholders
                att['tan_slope'][idx] = 0.1
                att['contourLength'][idx] = 30
                att['slopeTypeIndex'][idx] = 1
                att['mHeight'][idx] = self.forcing_measurement_height
                att['downHRUindex'][idx] = 0
                att['elevation'][idx] = -999
                att['soilTypeIndex'][idx] = -999
                att['vegTypeIndex'][idx] = -999

                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Processed {idx + 1} out of {num_hru} HRUs")

        self.logger.info(f"Attributes file created at: {attribute_path}")
        
        self.insert_soil_class(attribute_path)
        self.insert_land_class(attribute_path)
        self.insert_elevation(attribute_path)

    def insert_soil_class(self, attribute_file):
        """Insert soil class data into the attributes file."""
        self.logger.info("Inserting soil class into attributes file")
        if self.config.get('DATA_ACQUIRE') == 'HPC':
            self._insert_soil_class_from_supplied(attribute_file)
        
            gistool_output = self.project_dir / "attributes/soil_class"
            soil_stats = pd.read_csv(gistool_output / f"domain_{self.config.get('DOMAIN_NAME')}_stats_soil_classes.csv")

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    hru_id = att['hruId'][idx]
                    soil_row = soil_stats[soil_stats[self.hruId] == hru_id]
                    if not soil_row.empty:
                        soil_class = soil_row['majority'].values[0]
                        att['soilTypeIndex'][idx] = soil_class
                        self.logger.info(f"Set soil class for HRU {hru_id} to {soil_class}")
                    else:
                        self.logger.warning(f"No soil data found for HRU {hru_id}")

        elif self.config.get('DATA_ACQUIRE') == 'supplied':
            """Insert soil class data from supplied intersection file."""
            intersect_path = self._get_default_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
            intersect_name = self.config.get('INTERSECT_SOIL_NAME')
            intersect_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')

            shp = gpd.read_file(intersect_path / intersect_name)

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]
                    shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)
                    
                    tmp_hist = []
                    for j in range(13):
                        col_name = f'USGS_{j}'
                        if col_name in shp.columns:
                            tmp_hist.append(shp[col_name][shp_mask].values[0])
                        else:
                            tmp_hist.append(0)
                    
                    tmp_hist[0] = -1
                    tmp_sc = np.argmax(np.asarray(tmp_hist))
                    
                    if shp[f'USGS_{tmp_sc}'][shp_mask].values != tmp_hist[tmp_sc]:
                        self.logger.warning(f'Index and mode soil class do not match at hru_id {attribute_hru}')
                        tmp_sc = -999
                    
                    self.logger.info(f'Replacing soil class {att['soilTypeIndex'][idx]} with {tmp_sc} at HRU {attribute_hru}')
                    att['soilTypeIndex'][idx] = tmp_sc

    def insert_land_class(self, attribute_file):
        """Insert land class data into the attributes file."""
        self.logger.info("Inserting land class into attributes file")
        if self.config.get('DATA_ACQUIRE') == 'HPC':
            gistool_output = self.project_dir / "attributes/land_class"
            land_stats = pd.read_csv(gistool_output / f"domain_{self.config.get('DOMAIN_NAME')}_stats_NA_NALCMS_landcover_2020_30m.csv")

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    hru_id = att['hruId'][idx]
                    land_row = land_stats[land_stats[self.hruId] == hru_id]
                    if not land_row.empty:
                        land_class = land_row['majority'].values[0]
                        att['vegTypeIndex'][idx] = land_class
                        self.logger.info(f"Set land class for HRU {hru_id} to {land_class}")
                    else:
                        self.logger.warning(f"No land data found for HRU {hru_id}")
        
        elif self.config.get('DATA_ACQUIRE') == 'supplied':
            """Insert land class data from supplied intersection file."""
            intersect_path = self._get_default_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
            intersect_name = self.config.get('INTERSECT_LAND_NAME')
            intersect_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')

            shp = gpd.read_file(intersect_path / intersect_name)

            is_water = 0

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]
                    shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)
                    
                    tmp_hist = []
                    for j in range(1, 18):
                        col_name = f'IGBP_{j}'
                        if col_name in shp.columns:
                            tmp_hist.append(shp[col_name][shp_mask].values[0])
                        else:
                            tmp_hist.append(0)
                    
                    tmp_lc = np.argmax(np.asarray(tmp_hist)) + 1
                    
                    if shp[f'IGBP_{tmp_lc}'][shp_mask].values != tmp_hist[tmp_lc - 1]:
                        self.logger.warning(f'Index and mode land class do not match at hru_id {attribute_hru}')
                        tmp_lc = -999
                    
                    if tmp_lc == 17:
                        if any(val > 0 for val in tmp_hist[0:-1]):  # HRU is mostly water but other land classes are present
                            tmp_lc = np.argmax(np.asarray(tmp_hist[0:-1])) + 1  # select 2nd-most common class
                        else:
                            is_water += 1  # HRU is exclusively water
                    
                    self.logger.info(f'Replacing land class {att['vegTypeIndex'][idx]} with {tmp_lc} at HRU {attribute_hru}')
                    att['vegTypeIndex'][idx] = tmp_lc

                self.logger.info(f'{is_water} HRUs were identified as containing only open water. Note that SUMMA skips hydrologic calculations for such HRUs.')


    def insert_elevation(self, attribute_file):
        """Insert elevation data into the attributes file."""
        self.logger.info("Inserting elevation into attributes file")
        if self.config.get('DATA_ACQUIRE') == 'HPC':
            gistool_output = self.project_dir / "attributes/elevation"
            elev_stats = pd.read_csv(gistool_output / f"domain_{self.config.get('DOMAIN_NAME')}_stats_elv.csv")

            do_downHRUindex = self.config.get('SETTINGS_SUMMA_CONNECT_HRUS') == 'yes'

            with nc4.Dataset(attribute_file, "r+") as att:
                gru_data = {}
                for idx in range(len(att['hruId'])):
                    hru_id = att['hruId'][idx]
                    gru_id = att['hru2gruId'][idx]
                    elev_row = elev_stats[elev_stats[self.hruId] == hru_id]
                    if not elev_row.empty:
                        elevation = elev_row['mean'].values[0]
                        att['elevation'][idx] = elevation
                        self.logger.info(f"Set elevation for HRU {hru_id} to {elevation}")

                        if do_downHRUindex:
                            if gru_id not in gru_data:
                                gru_data[gru_id] = []
                            gru_data[gru_id].append((hru_id, elevation))
                    else:
                        self.logger.warning(f"No elevation data found for HRU {hru_id}")

                if do_downHRUindex:
                    self._set_downHRUindex(att, gru_data)
       
        elif self.config.get('DATA_ACQUIRE') == 'supplied':
            """Insert elevation data from supplied intersection file."""
            intersect_path = self._get_default_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
            intersect_name = self.config.get('INTERSECT_DEM_NAME')
            intersect_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')
            elev_column ='elev_mean'

            shp = gpd.read_file(intersect_path / intersect_name)

            do_downHRUindex = self.config.get('SETTINGS_SUMMA_CONNECT_HRUS') == 'yes'

            with nc4.Dataset(attribute_file, "r+") as att:
                gru_data = {}
                for idx in range(len(att['hruId'])):
                    hru_id = att['hruId'][idx]
                    gru_id = att['hru2gruId'][idx]
                    shp_mask = (shp[intersect_hruId_var].astype(int) == hru_id)
                    
                    if any(shp_mask):
                        elevation = shp[elev_column][shp_mask].values[0]
                        att['elevation'][idx] = elevation
                        self.logger.info(f"Set elevation for HRU {hru_id} to {elevation}")

                        if do_downHRUindex:
                            if gru_id not in gru_data:
                                gru_data[gru_id] = []
                            gru_data[gru_id].append((hru_id, elevation))
                    else:
                        self.logger.warning(f"No elevation data found for HRU {hru_id}")

                if do_downHRUindex:
                    self._set_downHRUindex(att, gru_data)

    def _set_downHRUindex(self, att, gru_data):
        """Set the downHRUindex based on elevation data."""
        for gru_id, hru_list in gru_data.items():
            sorted_hrus = sorted(hru_list, key=lambda x: x[1], reverse=True)
            for i, (hru_id, _) in enumerate(sorted_hrus):
                idx = np.where(att['hruId'][:] == hru_id)[0][0]
                if i == len(sorted_hrus) - 1:
                    att['downHRUindex'][idx] = 0  # outlet
                else:
                    att['downHRUindex'][idx] = sorted_hrus[i+1][0]
                self.logger.info(f"Set downHRUindex for HRU {hru_id} to {att['downHRUindex'][idx]}")

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.

        Args:
            path_key (str): The key to look up in the config dictionary.
            default_subpath (str): The default subpath to use if the config value is 'default'.

        Returns:
            Path: The resolved path.

        Raises:
            KeyError: If the path_key is not found in the config.
        """
        try:
            path_value = self.config.get(path_key)
            if path_value == 'default' or path_value is None:
                return self.project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            self.logger.error(f"Config key '{path_key}' not found")
            raise
    
    def _get_simulation_times(self) -> tuple[str, str]:
        """
        Get the simulation start and end times from config or calculate defaults.

        Returns:
            tuple[str, str]: A tuple containing the simulation start and end times.

        Raises:
            ValueError: If the time format in the configuration is invalid.
        """
        sim_start = self.config.get('EXPERIMENT_TIME_START')
        sim_end = self.config.get('EXPERIMENT_TIME_END')

        if sim_start == 'default' or sim_end == 'default':
            start_year = self.config.get('FORCING_START_YEAR')
            end_year = self.config.get('FORCING_END_YEAR')
            if not start_year or not end_year:
                raise ValueError("FORCING_START_YEAR or FORCING_END_YEAR is missing from configuration")
            sim_start = f"{start_year}-01-01 00:00" if sim_start == 'default' else sim_start
            sim_end = f"{end_year}-12-31 23:00" if sim_end == 'default' else sim_end

        # Validate time format
        try:
            datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            datetime.strptime(sim_end, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError("Invalid time format in configuration. Expected 'YYYY-MM-DD HH:MM'")

        return sim_start, sim_end

    