import os
import sys
from pathlib import Path
import xarray as xr # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
from typing import Dict, Any, Optional
import subprocess
import netCDF4 as nc4 # type: ignore
from shutil import copyfile
from pathlib import Path
from datetime import datetime
from shutil import copyfile
import rasterstats # type: ignore
from shapely.geometry import Polygon # type: ignore
import shutil
import rasterio # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))


class SummaPreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        
        """

        Initialize the SummaPreProcessor_spatial class.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing setup parameters.
            logger (Any): Logger object for recording processing information.
        """

        # Config and Logger
        self.config = config
        self.logger = logger

        # Directories and paths
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.summa_setup_dir = self.project_dir / "settings" / "SUMMA"
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        self.settings_path = self.project_dir / 'settings/SUMMA'
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"
        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")

        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_data')
        self.intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_basin_path.mkdir(parents=True, exist_ok=True)
        self.forcing_summa_path = self.project_dir / 'forcing' / 'SUMMA_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.river_network_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        if self.river_network_name == 'default':
            self.river_network_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_delineate.shp"

        self.river_network_path = self._get_default_path('RIVER_NETWORK_SHP_PATH', 'shapefiles/river_network')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        # Handles and variables
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')
        self.gruId = self.config.get('CATCHMENT_SHP_GRUID')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.forcing_dataset = self.config.get('FORCING_DATASET').lower()
        self.data_step = int(self.config.get('FORCING_TIME_STEP_SIZE'))
        self.coldstate_name = self.config.get('SETTINGS_SUMMA_COLDSTATE')
        self.parameter_name = self.config.get('SETTINGS_SUMMA_TRIALPARAMS')
        self.attribute_name = self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        self.forcing_measurement_height = float(self.config.get('FORCING_MEASUREMENT_HEIGHT'))

    def run_preprocessing(self):
        """
        Run the complete SUMMA spatial preprocessing workflow.

        This method orchestrates the  preprocessing pipeline.

        Raises:
            Exception: If any step in the preprocessing pipeline fails.
        """
        self.logger.info("Starting SUMMA spatial preprocessing")
        
        try:
            self.apply_datastep_and_lapse_rate()
            self.copy_base_settings()
            self.create_file_manager()
            self.create_forcing_file_list()
            self.create_initial_conditions()
            self.create_trial_parameters()
            self.create_attributes_file()

            self.logger.info("SUMMA spatial preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during SUMMA spatial preprocessing: {str(e)}")
            raise

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

    def apply_datastep_and_lapse_rate(self):
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
        intersect_base = f"{self.domain_name}_{self.config.get('FORCING_DATASET')}_intersected_shapefile"
        intersect_csv = self.intersect_path / f"{intersect_base}.csv"
        intersect_shp = self.intersect_path / f"{intersect_base}.shp"

        # If CSV doesn't exist but shapefile does, convert shapefile to CSV
        if not intersect_csv.exists() and intersect_shp.exists():
            self.logger.info(f"Converting {intersect_shp} to CSV format")
            try:
                shp_df = gpd.read_file(intersect_shp)
                shp_df['weight'] = shp_df['AP1']
                shp_df.to_csv(intersect_csv, index=False)
                self.logger.info(f"Successfully created {intersect_csv}")
            except Exception as e:
                self.logger.error(f"Failed to convert shapefile to CSV: {str(e)}")
                raise
        elif not intersect_csv.exists() and not intersect_shp.exists():
            raise FileNotFoundError(f"Neither {intersect_csv} nor {intersect_shp} exist")

        # Continue with existing code using the CSV file
        topo_data = pd.read_csv(intersect_csv)

        # Get forcing files
        forcing_files = [f for f in os.listdir(self.forcing_basin_path) if f.startswith(f"{self.domain_name}_{self.config.get('FORCING_DATASET')}") and f.endswith('.nc')]
        forcing_files.sort()
        self.logger.info(f"forcing files: {forcing_files}")

        # Prepare output directory
        self.forcing_summa_path.mkdir(parents=True, exist_ok=True)

        # Specify column names
        gru_id = f'S_1_{self.gruId}'
        hru_id = f'S_1_{self.hruId}'
        forcing_id = 'S_2_ID'
        catchment_elev = 'S_1_elev_m'
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
            output_file = self.forcing_summa_path / file

            with xr.open_dataset(self.forcing_basin_path / file) as dat:
                # Find which HRU IDs exist in the forcing data but not in the lapse values
                valid_hru_mask = np.isin(dat['hruId'].values, lapse_values.index)
                
                # Log the HRUs that will be removed
                if not np.all(valid_hru_mask):
                    missing_hrus = dat['hruId'].values[~valid_hru_mask]
                    self.logger.warning(f"Removing {len(missing_hrus)} HRU IDs that don't have lapse values")
                    self.logger.debug(f"Removed HRU IDs: {missing_hrus}")
                    
                    # Filter the dataset to only include valid HRUs
                    dat = dat.sel(hru=valid_hru_mask)
                    
                    if len(dat.hru) == 0:
                        self.logger.error("No valid HRUs remain after filtering. Cannot proceed.")
                        raise ValueError("No valid HRUs found after filtering out those without lapse values")
                
                # Apply datastep
                dat['data_step'] = self.data_step
                dat.data_step.attrs['long_name'] = 'data step length in seconds'
                dat.data_step.attrs['units'] = 's'

                if 'pptrate' in dat:
                    # Keep the values unchanged but update the unit attribute
                    dat.pptrate.attrs['units'] = 'mm/s'
                    dat.pptrate.attrs['long_name'] = 'Mean total precipitation rate'

                if self.config.get('APPLY_LAPSE_RATE') == True:
                    # Get lapse values for the HRUs
                    lapse_values_sorted = lapse_values['lapse_values'].loc[dat['hruId'].values]
                    addThis = xr.DataArray(np.tile(lapse_values_sorted.values, (len(dat['time']), 1)), 
                                        dims=('time', 'hru'))

                    # Get air temperature attributes
                    tmp_units = dat['airtemp'].units
                    
                    # Apply lapse rate correction
                    dat['airtemp'] = dat['airtemp'] + addThis
                    dat.airtemp.attrs['units'] = tmp_units

                # Save to file in new location
                dat.to_netcdf(output_file)

        self.logger.info(f"Completed processing of {self.forcing_dataset.upper()} forcing files with temperature lapsing")

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
            'scalarAquiferStorage': 0.4,
            'scalarSnowAlbedo': 0,
            'scalarCanairTemp': 283.16,
            'scalarCanopyTemp': 283.16,
            'mLayerTemp': 283.16,
            'mLayerVolFracIce': 0,
            'mLayerVolFracLiq': 0.4,
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

        # Setup example trial parameter file initialisation
        num_tp = 1
        all_tp = {}
        for i in range(num_tp):
            par_and_val = 'maxstep,900'
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

            if self.config.get('SETTINGS_SUMMA_TRIALPARAM_N') != 0:
                for var, val in all_tp.items():
                    tp_var = tp.createVariable(var, 'f8', 'hru', fill_value=False)
                    tp_var[:] = val

        self.logger.info(f"Trial parameters file created at: {parameter_path}")

    def calculate_slope_and_contour(self, shp, dem_path):
        """
        Calculate average slope and contour length for each HRU using vectorized operations.
        
        Args:
            shp (gpd.GeoDataFrame): GeoDataFrame containing HRU polygons
            dem_path (Path): Path to the DEM file
        
        Returns:
            dict: Dictionary with HRU IDs as keys and tuples (slope, contour_length) as values
        """
        self.logger.info("Calculating slope and contour length for each HRU")
        
        # Calculate contour lengths using vectorized operation
        contour_lengths = np.sqrt(shp.geometry.area)
        
        # Read DEM once
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            transform = src.transform
            
            # Calculate dx and dy for the entire DEM once
            cell_size_x = transform[0]
            cell_size_y = -transform[4]  # Negative because Y increases downward in pixel space
            
            # Calculate gradients for entire DEM once
            dy, dx = np.gradient(dem, cell_size_y, cell_size_x)
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            
            # Use zonal_stats to get mean slope for all HRUs at once
            mean_slopes = rasterstats.zonal_stats(
                shp.geometry,
                slope,
                affine=transform,
                stats=['mean'],
                nodata=np.nan
            )
        
        # Create results dictionary using vectorized operations
        results = {}
        for idx, row in shp.iterrows():
            hru_id = row[self.config.get('CATCHMENT_SHP_HRUID')]
            avg_slope = mean_slopes[idx]['mean']
            
            if avg_slope is None or np.isnan(avg_slope):
                self.logger.warning(f"No valid slope data found for HRU {hru_id}")
                results[hru_id] = (0.1, 30)  # Default values
            else:
                results[hru_id] = (avg_slope, contour_lengths[idx])
        
        return results

    def calculate_contour_length(self, hru_dem, hru_geometry, downstream_geometry, transform, hru_id):
        """
        Calculate the length of intersection between an HRU and its downstream neighbor.
        
        Args:
            hru_dem (numpy.ndarray): DEM data for the HRU
            hru_geometry (shapely.geometry): Geometry of the current HRU
            downstream_geometry (shapely.geometry): Geometry of the downstream HRU
            transform (affine.Affine): Transform for converting pixel to geographic coordinates
            hru_id (int): ID of the current HRU
        
        Returns:
            float: Length of the intersection between the HRU and its downstream neighbor
        """
        # If there's no downstream HRU (outlet), use the HRU's minimum perimeter length
        if downstream_geometry is None:
            min_dimension = min(hru_geometry.bounds[2] - hru_geometry.bounds[0], 
                            hru_geometry.bounds[3] - hru_geometry.bounds[1])
            self.logger.info(f"HRU {hru_id} is an outlet. Using minimum dimension: {min_dimension}")
            return min_dimension

        # Find the intersection between current and downstream HRUs
        intersection = hru_geometry.intersection(downstream_geometry)
        
        if intersection.is_empty:
            self.logger.warning(f"No intersection found between HRU {hru_id} and its downstream HRU")
            # Use minimum perimeter length as a fallback
            min_dimension = min(hru_geometry.bounds[2] - hru_geometry.bounds[0], 
                            hru_geometry.bounds[3] - hru_geometry.bounds[1])
            return min_dimension
        
        # Calculate the length of the intersection
        contour_length = intersection.length
        
        self.logger.info(f"Calculated contour length {contour_length:.2f} m for HRU {hru_id}")
        return contour_length


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
        
        # Calculate slope and contour length
        #slope_contour = self.calculate_slope_and_contour(shp, self.dem_path)

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

                # Set slope and contour length
                #hru_id = shp.iloc[idx][self.config.get('CATCHMENT_SHP_HRUID')]
                #slope, contour_length = slope_contour.get(hru_id, (0.1, 30))  # Use default values if not found
                att['tan_slope'][idx] = 0.1 #np.tan(slope)  # Convert slope to tan(slope)
                att['contourLength'][idx] = 100 #contour_length

                att['slopeTypeIndex'][idx] = 1
                att['mHeight'][idx] = self.forcing_measurement_height
                att['downHRUindex'][idx] = 0
                att['elevation'][idx] = -999
                att['soilTypeIndex'][idx] = -999
                att['vegTypeIndex'][idx] = -999

                #if (idx + 1) % 100 == 0:
                #    self.logger.info(f"Processed {idx + 1} out of {num_hru} HRUs")

        self.logger.info(f"Attributes file created at: {attribute_path}")

        self.insert_land_class(attribute_path)
        self.insert_soil_class(attribute_path)
        self.insert_elevation(attribute_path)

    def insert_soil_class(self, attribute_file):
        """Insert soil class data into the attributes file."""
        self.logger.info("Inserting soil class into attributes file")

        intersect_path = self._get_default_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_soilclass.shp'
        intersect_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')

        try:
            shp = gpd.read_file(intersect_path / intersect_name)

            # Check and create missing USGS_X columns
            for i in range(13):
                col_name = f'USGS_{i}'
                if col_name not in shp.columns:
                    shp[col_name] = 0  # Add the missing column and initialize with 0

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]
                    shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)
                    
                    # Check if there are any matching rows for this HRU
                    if not any(shp_mask):
                        self.logger.warning(f"No soil class data found for HRU {attribute_hru}, using default class")
                        att['soilTypeIndex'][idx] = 6  # Use a default value (6 = loam)
                        continue
                    
                    tmp_hist = []
                    for j in range(13):
                        col_name = f'USGS_{j}'
                        tmp_hist.append(shp[col_name][shp_mask].values[0])
                    
                    tmp_hist[0] = -1  # Set USGS_0 to -1 to avoid selecting it
                    tmp_sc = np.argmax(np.asarray(tmp_hist))
                    
                    if shp[f'USGS_{tmp_sc}'][shp_mask].values[0] != tmp_hist[tmp_sc]:
                        self.logger.warning(f'Index and mode soil class do not match at hru_id {attribute_hru}')
                        tmp_sc = 6  # Use a default value (6 = loam) instead of -999
                    
                    # Ensure soil type index is positive (SUMMA requires this)
                    if tmp_sc <= 0:
                        self.logger.warning(f"Invalid soil class {tmp_sc} for HRU {attribute_hru}, using default class")
                        tmp_sc = 6  # Use a default value (6 = loam)
                    
                    att['soilTypeIndex'][idx] = tmp_sc
        
        except Exception as e:
            self.logger.error(f"Error inserting soil class: {str(e)}")
            # If the process fails, set all soil types to a default value
            with nc4.Dataset(attribute_file, "r+") as att:
                self.logger.warning("Setting all soil types to default value (6 = loam)")
                att['soilTypeIndex'][:] = 6  # Set all to loam as fallback

    def insert_land_class(self, attribute_file):
        """Insert land class data into the attributes file."""
        self.logger.info("Inserting land class into attributes file")

        intersect_path = self._get_default_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'
        intersect_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')

        try:
            shp = gpd.read_file(intersect_path / intersect_name)

            # Check and create missing IGBP_X columns
            for i in range(1, 18):
                col_name = f'IGBP_{i}'
                if col_name not in shp.columns:
                    shp[col_name] = 0  # Add the missing column and initialize with 0

            is_water = 0

            with nc4.Dataset(attribute_file, "r+") as att:
                for idx in range(len(att['hruId'])):
                    attribute_hru = att['hruId'][idx]
                    shp_mask = (shp[intersect_hruId_var].astype(int) == attribute_hru)
                    
                    # Check if there are any matching rows for this HRU
                    if not any(shp_mask):
                        self.logger.warning(f"No land class data found for HRU {attribute_hru}, using default class")
                        att['vegTypeIndex'][idx] = 1  # Use a default value (1 = Evergreen Needleleaf)
                        continue
                    
                    tmp_hist = []
                    for j in range(1, 18):
                        col_name = f'IGBP_{j}'
                        tmp_hist.append(shp[col_name][shp_mask].values[0])
                    
                    tmp_lc = np.argmax(np.asarray(tmp_hist)) + 1
                    
                    if shp[f'IGBP_{tmp_lc}'][shp_mask].values[0] != tmp_hist[tmp_lc - 1]:
                        self.logger.warning(f'Index and mode land class do not match at hru_id {attribute_hru}')
                        tmp_lc = 1  # Use a default value (1 = Evergreen Needleleaf) instead of -999
                    
                    if tmp_lc == 17:
                        if any(val > 0 for val in tmp_hist[0:-1]):  # HRU is mostly water but other land classes are present
                            tmp_lc = np.argmax(np.asarray(tmp_hist[0:-1])) + 1  # select 2nd-most common class
                        else:
                            is_water += 1  # HRU is exclusively water
                    
                    # Ensure vegetation type index is positive (SUMMA requires this)
                    if tmp_lc <= 0:
                        self.logger.warning(f"Invalid vegetation class {tmp_lc} for HRU {attribute_hru}, using default class")
                        tmp_lc = 1  # Use a default value (1 = Evergreen Needleleaf)
                    
                    att['vegTypeIndex'][idx] = tmp_lc

                self.logger.info(f"{is_water} HRUs were identified as containing only open water. Note that SUMMA skips hydrologic calculations for such HRUs.")
        
        except Exception as e:
            self.logger.error(f"Error inserting land class: {str(e)}")
            # If the process fails, set all vegetation types to a default value
            with nc4.Dataset(attribute_file, "r+") as att:
                self.logger.warning("Setting all vegetation types to default value (1 = Evergreen Needleleaf)")
                att['vegTypeIndex'][:] = 1  # Set all to Evergreen Needleleaf as fallback


    def insert_elevation(self, attribute_file):
        """Insert elevation data into the attributes file."""
        self.logger.info("Inserting elevation into attributes file")

        intersect_path = self._get_default_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_dem.shp'

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
            start_year = self.config.get('EXPERIMENT_TIME_START').split('-')[0]
            end_year = self.config.get('EXPERIMENT_TIME_END').split('-')[0]
            if not start_year or not end_year:
                raise ValueError("EXPERIMENT_TIME_START or EXPERIMENT_TIME_END is missing from configuration")
            sim_start = f"{start_year}-01-01 01:00" if sim_start == 'default' else sim_start
            sim_end = f"{end_year}-12-31 22:00" if sim_end == 'default' else sim_end

        # Validate time format
        try:
            datetime.strptime(sim_start, "%Y-%m-%d %H:%M")
            datetime.strptime(sim_end, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError("Invalid time format in configuration. Expected 'YYYY-MM-DD HH:MM'")

        return sim_start, sim_end


class SUMMAPostprocessor:
    """
    Postprocessor for SUMMA model outputs via MizuRoute routing.
    Handles extraction and processing of simulation results.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_streamflow(self) -> Optional[Path]:
        """Extract streamflow from MizuRoute outputs for spatial mode."""
        self.logger.info("Extracting SUMMA/MizuRoute streamflow results")
        try:
            self.logger.info("Extracting SUMMA/MizuRoute streamflow results")
            
            # Get simulation output path
            if self.config.get('SIMULATIONS_PATH') == 'default':
                # Parse the start time and extract the date portion
                start_date = self.config['EXPERIMENT_TIME_START'].split()[0]  # Gets '2011-01-01' from '2011-01-01 01:00'
                sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute' / f"{self.config['EXPERIMENT_ID']}.h.{start_date}-03600.nc"
            else:
                sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))
                
            if not sim_file_path.exists():
                self.logger.error(f"SUMMA/MizuRoute output file not found at: {sim_file_path}")
                return None
                
            # Get simulation reach ID
            sim_reach_ID = self.config.get('SIM_REACH_ID')
            
            # Read simulation data
            ds = xr.open_dataset(sim_file_path, engine='netcdf4')
            
            # Extract data for the specific reach
            segment_index = ds['reachID'].values == int(sim_reach_ID)
            sim_df = ds.sel(seg=segment_index)
            q_sim = sim_df['IRFroutedRunoff'].to_dataframe().reset_index()
            q_sim.set_index('time', inplace=True)
            q_sim.index = q_sim.index.round(freq='h')
            
            # Convert from hourly to daily average
            q_sim_daily = q_sim['IRFroutedRunoff'].resample('D').mean()
            
            # Read existing results file if it exists
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
            if output_file.exists():
                results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
            else:
                results_df = pd.DataFrame(index=q_sim_daily.index)
            
            # Add SUMMA results
            results_df['SUMMA_discharge_cms'] = q_sim_daily
            
            # Save updated results
            results_df.to_csv(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
            raise


class SummaRunner:
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
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

    def run_summa(self):
        """
        Run the SUMMA model.
        
        This method selects the appropriate run mode (parallel, serial, or point)
        based on configuration settings and executes the SUMMA model accordingly.
        """
        #if self.config.get('SPATIAL_MODE') == 'Point':
            #self.run_summa_point()
        if self.config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False):
            self.run_summa_parallel()
        else:
            self.run_summa_serial()

    def run_summa_point(self):
        """
        Run SUMMA in point simulation mode.
        
        This method executes SUMMA for multiple point simulations, based on the
        file manager list created during preprocessing. It handles both the
        initial condition runs and the main simulation runs.
        """
        self.logger.info("Starting SUMMA point simulations")
        
        # Set up paths
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
            
        summa_exe = self.config.get('SUMMA_EXE')
        setting_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA_point/')
        
        # Run all sites from the file manager lists
        fm_ic_list_path = setting_path / 'list_fileManager_IC.txt'
        fm_list_path = setting_path / 'list_fileManager.txt'
        
        # Check if file manager lists exist
        if not fm_ic_list_path.exists() or not fm_list_path.exists():
            self.logger.error(f"File manager lists not found at {setting_path}")
            raise FileNotFoundError(f"Required file manager lists not found at {setting_path}")
        
        # Read file manager lists
        with open(fm_ic_list_path, 'r') as f:
            fm_ic_list = [line.strip() for line in f if line.strip()]
        
        with open(fm_list_path, 'r') as f:
            fm_list = [line.strip() for line in f if line.strip()]
        
        if len(fm_ic_list) != len(fm_list):
            self.logger.warning(f"Mismatch in file manager list lengths: {len(fm_ic_list)} IC files vs {len(fm_list)} main files")

        # Create output directory
        experiment_id = self.config.get('EXPERIMENT_ID')
        main_output_path = self.project_dir / 'simulations' / experiment_id / 'SUMMA_point'
        main_output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each site
        for i, (ic_fm, main_fm) in enumerate(zip(fm_ic_list, fm_list)):
            site_name = os.path.basename(ic_fm).split('_')[1]  # Extract site name from file manager name
            self.logger.info(f"Processing site {i+1}/{len(fm_list)}: {site_name}")
            
            # Create site-specific output directory
            site_output_path = main_output_path / site_name
            site_output_path.mkdir(parents=True, exist_ok=True)
            
            # Create log directory
            log_path = site_output_path / "logs"
            log_path.mkdir(parents=True, exist_ok=True)
            
            # Run initial conditions (IC) simulation
            self.logger.info(f"Running initial conditions simulation for {site_name}")
            ic_command = f"{str(summa_path / summa_exe)} -m {ic_fm} -r e"
            
            try:
                with open(log_path / f"{site_name}_IC.log", 'w') as log_file:
                    subprocess.run(ic_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                
                # Find the restart file (newest file with 'restart' in name)
                site_setting_path = Path(os.path.dirname(ic_fm))
                site_output_files = list(site_output_path.glob("*restart*"))
                
                if not site_output_files:
                    self.logger.error(f"No restart file found for {site_name}")
                    continue
                
                # Sort by modification time and get the most recent
                restart_file = sorted(site_output_files, key=os.path.getmtime)[-1]
                
                # Copy to warm_state.nc in settings directory
                shutil.copy(restart_file, site_setting_path / "warm_state.nc")
                self.logger.info(f"Copied restart file to warm state for {site_name}")
                
                # Run main simulation
                self.logger.info(f"Running main simulation for {site_name}")
                main_command = f"{str(summa_path / summa_exe)} -m {main_fm}"
                
                with open(log_path / f"{site_name}_main.log", 'w') as log_file:
                    subprocess.run(main_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                
                self.logger.info(f"Completed simulation for {site_name}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"SUMMA run failed for {site_name} with error code {e.returncode}")
                self.logger.error(f"Command that failed: {e.cmd}")
            except Exception as e:
                self.logger.error(f"Error processing site {site_name}: {str(e)}")
        
        self.logger.info(f"Completed all SUMMA point simulations ({len(fm_list)} sites)")
        return main_output_path
    
    def run_summa_parallel(self):
        """
        Run SUMMA in parallel using SLURM array jobs.
        This method handles GRU-based parallelization using SLURM's job array capability.
        """
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Set up paths and filenames
        summa_path = self.config.get('SETTINGS_SUMMA_PARALLEL_PATH')
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)

        summa_exe = self.config.get('SETTINGS_SUMMA_PARALLEL_EXE')
        settings_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self._get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Create output and log directories if they don't exist
        summa_log_path.mkdir(parents=True, exist_ok=True)
        summa_out_path.mkdir(parents=True, exist_ok=True)

        # Get total GRU count from catchment shapefile
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name
        
        # Read shapefile and count unique GRU_IDs
        try:
            gdf = gpd.read_file(subbasins_shapefile)
            total_grus = len(gdf[self.config.get('CATCHMENT_SHP_GRUID')].unique())
            self.logger.info(f"Counted {total_grus} unique GRUs from shapefile: {subbasins_shapefile}")
        except Exception as e:
            self.logger.error(f"Error counting GRUs from shapefile: {str(e)}")
            raise RuntimeError(f"Failed to count GRUs from shapefile {subbasins_shapefile}: {str(e)}")

        # Logically estimate GRUs per job based on total GRU count
        grus_per_job = self._estimate_grus_per_job(total_grus)
        self.logger.info(f"Estimated optimal GRUs per job: {grus_per_job} for {total_grus} total GRUs")

        # Calculate number of array jobs needed (minimum 1)
        n_array_jobs = max(1, -(-total_grus // grus_per_job))  # Ceiling division
        
        self.logger.info(f"Will launch {n_array_jobs} parallel jobs with {grus_per_job} GRUs per job")
        
        # Create SLURM script
        slurm_script = self._create_slurm_script(
            summa_path=summa_path,
            summa_exe=summa_exe,
            settings_path=settings_path,
            filemanager=filemanager,
            summa_log_path=summa_log_path,
            summa_out_path=summa_out_path,
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            n_array_jobs=n_array_jobs - 1  # SLURM arrays are 0-based
        )
        
        # Write SLURM script
        script_path = self.project_dir / 'run_summa_parallel.sh'
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        
        # Make script executable
        import os
        os.chmod(script_path, 0o755)
        
        # Submit job
        try:
            import subprocess
            import shutil
            
            # Check if sbatch exists in the path
            if not shutil.which("sbatch"):
                self.logger.error("SLURM 'sbatch' command not found. Is SLURM installed on this system?")
                raise RuntimeError("SLURM 'sbatch' command not found")
            
            # Log the full command being executed
            cmd = f"sbatch {script_path}"
            self.logger.info(f"Executing command: {cmd}")
            
            # Run the command
            process = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            job_id = process.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM array job with ID: {job_id}")
            
            # Backup settings if required
            if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
                backup_path = summa_out_path / "run_settings"
                self._backup_settings(settings_path, backup_path)
            
            # Check if we should monitor the job
            monitor_job = self.config.get('MONITOR_SLURM_JOB', True)
            if monitor_job:
                import time
                
                self.logger.info(f"Monitoring SLURM job {job_id}")
                
                # Wait for SLURM job to complete
                wait_time = 0
                max_wait_time = 3600  # 1 hour
                check_interval = 60  # 1 minute
                
                while wait_time < max_wait_time:
                    try:
                        result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)
                        
                        # If result only contains header, job is no longer in queue
                        if result.stdout.count('\n') <= 1:
                            self.logger.info(f"Job {job_id} no longer in queue, checking status")
                            
                            # Check if job completed successfully
                            sacct_cmd = f"sacct -j {job_id} -o State -n | head -1"
                            state_result = subprocess.run(sacct_cmd, shell=True, capture_output=True, text=True)
                            state = state_result.stdout.strip()
                            
                            if "COMPLETED" in state:
                                self.logger.info(f"Job {job_id} completed successfully")
                                break
                            elif "FAILED" in state or "CANCELLED" in state or "TIMEOUT" in state:
                                self.logger.error(f"Job {job_id} ended with status: {state}")
                                raise RuntimeError(f"SLURM job {job_id} failed with status: {state}")
                            else:
                                self.logger.warning(f"Job {job_id} has unknown status: {state}")
                                break
                        else:
                            pending_count = result.stdout.count("PENDING")
                            running_count = result.stdout.count("RUNNING")
                            self.logger.info(f"Job {job_id} status: {running_count} running, {pending_count} pending")
                    except subprocess.SubprocessError as e:
                        self.logger.warning(f"Error checking job status: {str(e)}")
                    
                    # Wait before checking again
                    time.sleep(check_interval)
                    wait_time += check_interval
                
                if wait_time >= max_wait_time:
                    self.logger.warning(f"Maximum wait time exceeded for job {job_id}. Continuing without waiting for completion.")
            
            self.logger.info("SUMMA parallel run completed or continuing in background")
            return self.merge_parallel_outputs()
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing sbatch command: {str(e)}")
            self.logger.error(f"Command output: {e.stdout}")
            self.logger.error(f"Command error: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job. Error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in parallel SUMMA workflow: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _estimate_grus_per_job(self, total_grus: int) -> int:
        """
        Estimate the optimal number of GRUs per job based on total GRU count.
        
        This function balances computational efficiency with queue management by:
        - Keeping the number of parallel jobs reasonable (not too many small jobs)
        - Ensuring each job has enough work to be worthwhile
        - Adapting to different domain sizes
        
        Args:
            total_grus (int): Total number of GRUs in the domain
            
        Returns:
            int: Optimal number of GRUs to process per job
        """
        # Define optimization parameters
        min_jobs = 10          # Minimum number of jobs to split into
        max_jobs = 500         # Maximum number of jobs to prevent queue flooding
        min_grus_per_job = 1   # Minimum GRUs per job
        ideal_grus_per_job = 50  # Ideal number of GRUs per job for efficiency
        
        # For very small domains, process all GRUs in fewer jobs
        if total_grus <= min_jobs:
            return 1
        
        # For small to medium domains, aim for the ideal GRUs per job
        if total_grus <= ideal_grus_per_job * min_jobs:
            return max(min_grus_per_job, total_grus // min_jobs)
        
        # For larger domains, balance between ideal number and not exceeding max jobs
        ideal_jobs = total_grus // ideal_grus_per_job
        
        if ideal_jobs <= max_jobs:
            # We can use the ideal number
            grus_per_job = ideal_grus_per_job
        else:
            # Need to increase GRUs per job to stay under max_jobs limit
            grus_per_job = -(-total_grus // max_jobs)  # Ceiling division
        
        # Additional consideration for very large domains
        # If we have more than 10,000 GRUs, we might want to increase GRUs per job
        # to reduce overhead and improve efficiency
        if total_grus > 10000:
            # Scale up based on domain size
            scale_factor = min(3.0, total_grus / 10000)
            grus_per_job = int(grus_per_job * scale_factor)
        
        # Ensure we don't exceed total GRUs
        grus_per_job = min(grus_per_job, total_grus)
        
        self.logger.debug(f"GRU estimation details: total_grus={total_grus}, "
                        f"ideal_jobs={ideal_jobs}, grus_per_job={grus_per_job}")
        
        return grus_per_job

    def _create_slurm_script(self, summa_path: Path, summa_exe: str, settings_path: Path, 
                            filemanager: str, summa_log_path: Path, summa_out_path: Path,
                            total_grus: int, grus_per_job: int, n_array_jobs: int) -> str:
        """
        Create a SLURM batch script for running SUMMA in parallel.
        
        Args:
            summa_path (Path): Path to SUMMA executable directory
            summa_exe (str): Name of SUMMA executable
            settings_path (Path): Path to SUMMA settings directory
            filemanager (str): Name of SUMMA file manager
            summa_log_path (Path): Path for SUMMA log files
            summa_out_path (Path): Path for SUMMA output files
            total_grus (int): Total number of GRUs to process
            grus_per_job (int): Number of GRUs to process per job
            n_array_jobs (int): Number of array jobs (0-based maximum index)
            
        Returns:
            str: Content of the SLURM batch script
        """
                
        # Create the script
        script = f"""#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00'
#SBATCH --mem=4G
#SBATCH --job-name=Summa-{self.config.get('DOMAIN_NAME')}
#SBATCH --output={summa_log_path}/summa_%A_%a.out
#SBATCH --error={summa_log_path}/summa_%A_%a.err
#SBATCH --array=0-{n_array_jobs}

# Print job info for debugging
echo "Starting SUMMA parallel job at $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"

# Create required directories
mkdir -p {summa_out_path}
mkdir -p {summa_log_path}

# Calculate GRU range for this job
gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))
gru_end=$(( gru_start + {grus_per_job} - 1 ))

# Ensure we don't exceed total GRUs
if [ $gru_end -gt {total_grus} ]; then
    gru_end={total_grus}
fi

echo "Processing GRUs $gru_start to $gru_end"

# Check if SUMMA executable exists
if [ ! -f "{summa_path}/{summa_exe}" ]; then
    echo "ERROR: SUMMA executable not found at {summa_path}/{summa_exe}"
    exit 1
fi

# Check if filemanager exists
if [ ! -f "{settings_path}/{filemanager}" ]; then
    echo "ERROR: File manager not found at {settings_path}/{filemanager}"
    exit 1
fi

# Process each GRU in the range
for gru in $(seq $gru_start $gru_end); do
    echo "Starting GRU $gru"
    
    # Run SUMMA
    {summa_path}/{summa_exe} -g $gru 1 -m {settings_path}/{filemanager}
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "SUMMA failed for GRU $gru with exit code $exit_code"
        exit 1
    fi
    
    echo "Completed GRU $gru"
done

echo "Completed all GRUs for this job at $(date)"
"""
        return script

    def run_summa_serial(self):
        """
        Run the SUMMA model.

        This method sets up the necessary paths, executes the SUMMA model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting SUMMA run")

        # Set up paths and filenames
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
            
        summa_exe = self.config.get('SUMMA_EXE')
        settings_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self._get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_log_name = "summa_log.txt"
        
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            backup_path = summa_out_path / "run_settings"
            self._backup_settings(settings_path, backup_path)

        # Run SUMMA
        os.makedirs(summa_log_path, exist_ok=True)
        summa_command = f"{str(summa_path / summa_exe)} -m {str(settings_path / filemanager)}"
        
        try:
            with open(summa_log_path / summa_log_name, 'w') as log_file:
                subprocess.run(summa_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info("SUMMA run completed successfully")
            return summa_out_path
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA run failed with error: {e}")
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

    def merge_parallel_outputs(self):
        """
        Merge parallel SUMMA outputs into two MizuRoute-readable files:
        one for timestep data and one for daily data.
        This function is called after parallel SUMMA execution completes.
        Preserves all variables from the original SUMMA output.
        """
        self.logger.info("Starting to merge parallel SUMMA outputs")
        
        # Get experiment settings
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Define output files
            timestep_output = mizu_in_path / f"{experiment_id}_timestep.nc"
            daily_output = mizu_in_path / f"{experiment_id}_day.nc"
            
            # Source file patterns
            timestep_pattern = f"{experiment_id}_*_timestep.nc"
            daily_pattern = f"{experiment_id}_*_day.nc"
            
            def process_and_merge_files(file_pattern, output_file):
                self.logger.info(f"Processing files matching {file_pattern}")
                input_files = list(summa_out_path.glob(file_pattern))
                input_files.sort()
                
                if not input_files:
                    self.logger.warning(f"No files found matching pattern: {file_pattern}")
                    return
                
                merged_ds = None
                for src_file in input_files:
                    try:
                        ds = xr.open_dataset(src_file)
                        
                        # Convert time to seconds since reference date
                        reference_date = pd.Timestamp('1990-01-01')
                        time_values = pd.to_datetime(ds.time.values)
                        seconds_since_ref = (time_values - reference_date).total_seconds()
                        
                        # Replace the time coordinate with seconds since reference
                        ds = ds.assign_coords(time=seconds_since_ref)
                        
                        # Set time attributes
                        ds.time.attrs = {
                            'units': 'seconds since 1990-1-1 0:0:0.0 -0:00',
                            'calendar': 'standard',
                            'long_name': 'time since time reference (instant)'
                        }
                        
                        # Merge with existing data
                        if merged_ds is None:
                            merged_ds = ds
                        else:
                            merged_ds = xr.merge([merged_ds, ds])
                        
                        ds.close()
                        
                    except Exception as e:
                        self.logger.error(f"Error processing file {src_file}: {str(e)}")
                        continue
                
                # Save merged data
                if merged_ds is not None:
                    # Create encoding dict for all variables
                    encoding = {
                        'time': {
                            'dtype': 'double',
                            '_FillValue': None
                        }
                    }
                    
                    # Add encoding for all other variables
                    for var in merged_ds.data_vars:
                        encoding[var] = {'_FillValue': None}
                    
                    # Preserve the original attributes
                    if 'summaVersion' in merged_ds.attrs:
                        global_attrs = merged_ds.attrs
                    else:
                        global_attrs = {
                            'summaVersion': '',
                            'buildTime': '',
                            'gitBranch': '',
                            'gitHash': '',
                        }
                    
                    # Update merged dataset attributes
                    merged_ds.attrs.update(global_attrs)
                    
                    # Save to netCDF
                    merged_ds.to_netcdf(
                        output_file,
                        encoding=encoding,
                        unlimited_dims=['time'],
                        format='NETCDF4'
                    )
                    self.logger.info(f"Successfully created merged file: {output_file}")
                    merged_ds.close()
            
            # Process both timestep and daily files
            process_and_merge_files(timestep_pattern, timestep_output)
            process_and_merge_files(daily_pattern, daily_output)
            
            self.logger.info("SUMMA output merging completed successfully")
            return mizu_in_path
            
        except Exception as e:
            self.logger.error(f"Error merging SUMMA outputs: {str(e)}")
            raise

