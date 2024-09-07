import os
import sys
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import netCDF4 as nc4 # type: ignore
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from shutil import copyfile
import shapefile # type: ignore
import rasterio # type: ignore
import rasterstats # type: ignore
from pyproj import CRS, Transformer # type: ignore
import random

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logging_utils import get_function_logger # type: ignore

class SummaPreProcessor_spatial:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.summa_setup_dir = self.project_dir / "settings" / "SUMMA"
        self.hruId = self.config.get('CATCHMENT_SHP_HRUID')

    @get_function_logger
    def run_preprocessing(self):
        self.logger.info("Starting SUMMA spatial preprocessing")
        
        #self.sort_catchment_shape()
        self.copy_base_settings()
        self.create_file_manager()
        self.process_forcing_data()
        self.create_forcing_file_list()
        self.create_initial_conditions()
        self.create_trial_parameters()
        self.create_attributes_file()

        self.logger.info("SUMMA spatial preprocessing completed")


    def sort_catchment_shape(self):
        self.logger.info("Sorting catchment shape")
        
        self.catchment_path = self.config.get('CATCHMENT_PATH')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        
        if self.catchment_path == 'default':
            self.catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            self.catchment_path = Path(self.catchment_path)
        
        gru_id = self.config.get('CATCHMENT_SHP_GRUID')
        hru_id = self.config.get('CATCHMENT_SHP_HRUID')
        
        # Open the shape
        shp = gpd.read_file(self.catchment_path / self.catchment_name)
        
        # Sort
        shp = shp.sort_values(by=[gru_id, hru_id])
        
        # Save
        shp.to_file(self.catchment_path / self.catchment_name)
        
        self.logger.info(f"Catchment shape sorted and saved to {self.catchment_path / self.catchment_name}")

    def copy_base_settings(self):
        self.logger.info("Copying SUMMA base settings")
        
        base_settings_path = Path(self.config.get('CONFLUENCE_CODE_DIR')) / '0_base_settings' / 'SUMMA'
        
        settings_path = self.config.get('SETTINGS_SUMMA_PATH')
        if settings_path == 'default':
            settings_path = self.project_dir / 'settings' / 'SUMMA'
        else:
            settings_path = Path(settings_path)
        
        settings_path.mkdir(parents=True, exist_ok=True)
        
        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, settings_path / file)
        
        self.logger.info(f"SUMMA base settings copied to {settings_path}")

    def create_file_manager(self):
        self.logger.info("Creating SUMMA file manager")
        experiment_id = self.config.get('EXPERIMENT_ID')
        self.sim_start = self.config.get('EXPERIMENT_TIME_START')
        self.sim_end = self.config.get('EXPERIMENT_TIME_END')

        if self.sim_start == 'default' or self.sim_end == 'default':
            raw_time = [self.config.get('FORCING_START_YEAR'),self.config.get('FORCING_END_YEAR')]
            self.sim_start = f"{raw_time[0]}-01-01 00:00" if self.sim_start == 'default' else self.sim_start
            self.sim_end = f"{raw_time[1]}-12-31 23:00" if self.sim_end == 'default' else self.sim_end

        filemanager_name = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        filemanager_path = self.summa_setup_dir / filemanager_name

        with open(filemanager_path, 'w') as fm:
            fm.write(f"controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'\n")
            fm.write(f"simStartTime         '{self.sim_start}'\n")
            fm.write(f"simEndTime           '{self.sim_end}'\n")
            fm.write(f"tmZoneInfo           'utcTime'\n")
            fm.write(f"outFilePrefix        '{experiment_id}'\n")
            fm.write(f"settingsPath         '{self.summa_setup_dir}/'\n")
            fm.write(f"forcingPath          '{self.project_dir / 'forcing/SUMMA_input'}/'\n")
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

    def process_forcing_data(self):
        self.logger.info("Processing forcing data")
        forcing_dataset = self.config.get('FORCING_DATASET').lower()
        input_path = self.project_dir / 'forcing/basin_averaged_data'
        output_path = self.project_dir / 'forcing/SUMMA_input'
        output_path.mkdir(parents=True, exist_ok=True)

        # Get the date range from the config
        start_date = pd.to_datetime(self.sim_start)
        end_date = pd.to_datetime(self.sim_end)

        # Get the variable mapping for the dataset
        var_mapping = forcing_dataset_mapping().get(forcing_dataset, {})

        # Process each day within the date range
        current_date = start_date
        while current_date <= end_date:
            self.logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")

            # Find the file for the current day
            input_file = list(input_path.glob(f"remapped_remapped_domain_{self.config.get('DOMAIN_NAME')}_{current_date.strftime('%Y%m%d')}*.nc"))
            
            if not input_file:
                self.logger.warning(f"No file found for {current_date.strftime('%Y-%m-%d')}")
                current_date += timedelta(days=1)
                continue

            input_file = input_file[0]  # Take the first file if multiple exist

            # Open the dataset
            ds = xr.open_dataset(input_file)

            # Rename variables according to the mapping
            ds = ds.rename(var_mapping)

            # Apply unit conversions
            ds = self.apply_unit_conversions(ds, forcing_dataset)

            # Set attributes
            ds.attrs['History'] = f'Processed on {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}'
            ds.attrs['Author'] = "Created by SUMMA workflow scripts"
            ds.attrs['Purpose'] = f'{forcing_dataset.upper()} data processed for SUMMA compatibility'

            # Set missing value attribute for all variables
            for var in ds.data_vars:
                ds[var].attrs['missing_value'] = -999

            # Define output filename
            output_file = output_path / f"{forcing_dataset}_processed_{current_date.strftime('%Y%m%d')}.nc"

            # Save the processed data
            ds.to_netcdf(output_file)

            self.logger.info(f"Saved processed data to {output_file}")

            # Move to the next day
            current_date += timedelta(days=1)

        self.logger.info("Forcing data processing completed")

    def create_forcing_file_list(self):
        self.logger.info("Creating forcing file list")
        forcing_dataset = self.config.get('FORCING_DATASET').lower()
        domain_name = self.config.get('DOMAIN_NAME')
        forcing_path = self.project_dir / 'forcing/SUMMA_input'
        file_list_path = self.summa_setup_dir / self.config.get('SETTINGS_SUMMA_FORCING_LIST')

        if forcing_dataset == 'carra':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{forcing_dataset}") and f.endswith('.nc')]
        elif forcing_dataset == 'era5':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{forcing_dataset}") and f.endswith('.nc')]
        elif forcing_dataset == 'rdrs':
            forcing_files = [f for f in os.listdir(forcing_path) if f.startswith(f"{forcing_dataset}") and f.endswith('.nc')]
        else:
            self.logger.error(f"Unsupported forcing dataset: {forcing_dataset}")
            raise ValueError(f"Unsupported forcing dataset: {forcing_dataset}")

        forcing_files.sort()

        with open(file_list_path, 'w') as f:
            for file in forcing_files:
                f.write(f"{file}\n")

        self.logger.info(f"Forcing file list created at {file_list_path}")

    def create_initial_conditions(self):
        self.logger.info("Creating initial conditions file")
        coldstate_path = self.summa_setup_dir
        coldstate_name = self.config.get('SETTINGS_SUMMA_COLDSTATE')
        forcing_path = self.project_dir / 'forcing/SUMMA_input'
        
        # Get HRU IDs from a forcing file
        forcing_files = os.listdir(forcing_path)
        forcing_name = forcing_files[0]
        forc = xr.open_dataset(forcing_path / forcing_name)
        forcing_hruIds = forc['hruId'].values.astype(int)
        
        # Define dimensions and fill values
        num_hru = len(forcing_hruIds)
        nSoil = 8
        nSnow = 0
        midSoil = 8
        midToto = 8
        ifcToto = midToto + 1
        dt_init = float(self.config.get('FORCING_TIME_STEP_SIZE'))
        mLayerDepth = np.asarray([0.025, 0.075, 0.15, 0.25, 0.5, 0.5, 1, 1.5])
        iLayerHeight = np.asarray([0, 0.025, 0.1, 0.25, 0.5, 1, 1.5, 2.5, 4])

        # Define states
        scalarCanopyIce = 0
        scalarCanopyLiq = 0
        scalarSnowDepth = 0
        scalarSWE = 0
        scalarSfcMeltPond = 0
        scalarAquiferStorage = 1.0
        scalarSnowAlbedo = 0
        scalarCanairTemp = 283.16
        scalarCanopyTemp = 283.16
        mLayerTemp = 283.16
        mLayerVolFracIce = 0
        mLayerVolFracLiq = 0.2
        mLayerMatricHead = -1.0

        # Create the initial conditions file
        with nc4.Dataset(coldstate_path / coldstate_name, "w", format="NETCDF4") as cs:
            # Set attributes
            cs.setncattr('Author', "Created by SUMMA workflow scripts")
            cs.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            cs.setncattr('Purpose', 'Create a cold state .nc file for initial SUMMA runs')

            # Define dimensions
            cs.createDimension('hru', num_hru)
            cs.createDimension('midSoil', midSoil)
            cs.createDimension('midToto', midToto)
            cs.createDimension('ifcToto', ifcToto)

            # Create variables
            self.create_variable(cs, 'hruId', 'i4', 'hru', forcing_hruIds)
            self.create_variable(cs, 'dt_init', 'f8', 'hru', dt_init)
            self.create_variable(cs, 'nSoil', 'i4', 'hru', nSoil)
            self.create_variable(cs, 'nSnow', 'i4', 'hru', nSnow)
            self.create_variable(cs, 'nLayers', 'i4', 'hru', nSoil + nSnow)
            
            # States
            self.create_variable(cs, 'scalarCanopyIce', 'f8', 'hru', scalarCanopyIce)
            self.create_variable(cs, 'scalarCanopyLiq', 'f8', 'hru', scalarCanopyLiq)
            self.create_variable(cs, 'scalarSnowDepth', 'f8', 'hru', scalarSnowDepth)
            self.create_variable(cs, 'scalarSWE', 'f8', 'hru', scalarSWE)
            self.create_variable(cs, 'scalarSfcMeltPond', 'f8', 'hru', scalarSfcMeltPond)
            self.create_variable(cs, 'scalarAquiferStorage', 'f8', 'hru', scalarAquiferStorage)
            self.create_variable(cs, 'scalarSnowAlbedo', 'f8', 'hru', scalarSnowAlbedo)
            self.create_variable(cs, 'scalarCanairTemp', 'f8', 'hru', scalarCanairTemp)
            self.create_variable(cs, 'scalarCanopyTemp', 'f8', 'hru', scalarCanopyTemp)
            
            # Layer variables
            self.create_variable(cs, 'mLayerTemp', 'f8', ('hru', 'midToto'), mLayerTemp)
            self.create_variable(cs, 'mLayerVolFracIce', 'f8', ('hru', 'midToto'), mLayerVolFracIce)
            self.create_variable(cs, 'mLayerVolFracLiq', 'f8', ('hru', 'midToto'), mLayerVolFracLiq)
            self.create_variable(cs, 'mLayerMatricHead', 'f8', ('hru', 'midSoil'), mLayerMatricHead)
            
            # Layer dimensions
            self.create_variable(cs, 'iLayerHeight', 'f8', ('hru', 'ifcToto'), iLayerHeight)
            self.create_variable(cs, 'mLayerDepth', 'f8', ('hru', 'midToto'), mLayerDepth)

        self.logger.info(f"Initial conditions file created at {coldstate_path / coldstate_name}")

    def create_variable(self, nc_file, var_name, var_type, dimensions, values):
        var = nc_file.createVariable(var_name, var_type, dimensions)
        if isinstance(values, (int, float)):
            if len(dimensions) == 1:
                var[:] = np.full(var.shape, values)
            elif len(dimensions) == 2:
                var[:] = np.full(var.shape, values)
        elif isinstance(values, np.ndarray) and len(dimensions) == 2:
            var[:] = np.tile(values, (var.shape[0], 1))
        else:
            var[:] = values
        return var

    def create_trial_parameters(self):
        self.logger.info("Creating trial parameters file")
        # Implementation based on 1_create_trialParams.py
        parameter_path = self.summa_setup_dir
        parameter_name = self.config.get('SETTINGS_SUMMA_TRIALPARAMS')
        forcing_path = self.project_dir / 'forcing/SUMMA_input'
        
        # Get HRU IDs from a forcing file
        forcing_files = os.listdir(forcing_path)
        forcing_name = forcing_files[0]
        forc = xr.open_dataset(forcing_path / forcing_name)
        forcing_hruIds = forc['hruId'].values.astype(int)

        # Create the trial parameter file
        with nc4.Dataset(parameter_path / parameter_name, "w", format="NETCDF4") as tp:
            # Set attributes
            tp.setncattr('Author', "Created by SUMMA workflow scripts")
            tp.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')
            tp.setncattr('Purpose', 'Create a trial parameter .nc file for initial SUMMA runs')

            # Define dimensions
            tp.createDimension('hru', len(forcing_hruIds))

            # Create hruId variable
            self.create_variable(tp, 'hruId', 'i4', 'hru', forcing_hruIds)

            # Add any specified trial parameters
            num_tp = int(self.config.get('SETTINGS_SUMMA_TRIALPARAM_N', 0))
            for ii in range(num_tp):
                par_and_val = self.config.get(f'SETTINGS_SUMMA_TRIALPARAM_1{ii+1}')
                if par_and_val:
                    arr = par_and_val.split(',')
                    var_name = arr[0]
                    values = np.array(arr[1:], dtype=np.float32)
                    self.create_variable(tp, var_name, 'f8', 'hru', values)

        self.logger.info(f"Trial parameters file created at {parameter_path / parameter_name}")

    def create_attributes_file(self):
        self.logger.info("Creating attributes file")
        attribute_path = self.summa_setup_dir
        attribute_name = self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        
        # Load catchment shapefile
        catchment_path = self.config.get('CATCHMENT_SHP_PATH', 'default')
        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles/catchment'
        else:
            catchment_path = Path(catchment_path)

        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        shp = gpd.read_file(catchment_path / catchment_name)
        
        # Get HRU IDs from a forcing file
        forcing_path = self.project_dir / 'forcing/SUMMA_input'
        forcing_files = os.listdir(forcing_path)
        forcing_name = forcing_files[0]
        forc = xr.open_dataset(forcing_path / forcing_name)
        forcing_hruIds = forc['hruId'].values.astype(int)
        print(forcing_hruIds)

        # Sort shapefile based on forcing HRU order
        catchment_hruId_var = self.config.get('CATCHMENT_SHP_HRUID')
        shp = shp.set_index(catchment_hruId_var)
        shp.index = shp.index.astype(int)
        shp = shp.loc[forcing_hruIds]
        shp = shp.reset_index()

        # Get GRU and HRU information
        catchment_gruId_var = self.config.get('CATCHMENT_SHP_GRUID')
        hru_ids = pd.unique(shp[catchment_hruId_var].values)
        gru_ids = pd.unique(shp[catchment_gruId_var].values)
        num_hru = len(hru_ids)
        num_gru = len(gru_ids)

        # Create the attributes file
        with nc4.Dataset(attribute_path / attribute_name, "w", format="NETCDF4") as att:
            # Set attributes
            att.setncattr('Author', "Created by SUMMA workflow scripts")
            att.setncattr('History', f'Created {datetime.now().strftime("%Y/%m/%d %H:%M:%S")}')

            # Define dimensions
            att.createDimension('hru', num_hru)
            att.createDimension('gru', num_gru)

            # Create variables
            var = att.createVariable('hruId', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of hydrological response unit (HRU)')

            var = att.createVariable('gruId', 'i4', ('gru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of grouped response unit (GRU)')

            var = att.createVariable('hru2gruId', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of GRU to which the HRU belongs')

            var = att.createVariable('downHRUindex', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index of downslope HRU (0 = basin outlet)')

            var = att.createVariable('longitude', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'Decimal degree east')
            var.setncattr('long_name', "Longitude of HRU's centroid")

            var = att.createVariable('latitude', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'Decimal degree north')
            var.setncattr('long_name', "Latitude of HRU's centroid")

            var = att.createVariable('elevation', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'm')
            var.setncattr('long_name', 'Mean HRU elevation')

            var = att.createVariable('HRUarea', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'm^2')
            var.setncattr('long_name', 'Area of HRU')

            var = att.createVariable('tan_slope', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'm m-1')
            var.setncattr('long_name', 'Average tangent slope of HRU')

            var = att.createVariable('contourLength', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'm')
            var.setncattr('long_name', 'Contour length of HRU')

            var = att.createVariable('slopeTypeIndex', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index defining slope')

            var = att.createVariable('soilTypeIndex', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index defining soil type')

            var = att.createVariable('vegTypeIndex', 'i4', ('hru',), fill_value=False)
            var.setncattr('units', '-')
            var.setncattr('long_name', 'Index defining vegetation type')

            var = att.createVariable('mHeight', 'f8', ('hru',), fill_value=False)
            var.setncattr('units', 'm')
            var.setncattr('long_name', 'Measurement height above bare ground')

            # Fill GRU variable
            att['gruId'][:] = gru_ids

            # Fill HRU variables
            catchment_area_var = self.config.get('CATCHMENT_SHP_AREA')
            catchment_lat_var = self.config.get('CATCHMENT_SHP_LAT')
            catchment_lon_var = self.config.get('CATCHMENT_SHP_LON')
            forcing_measurement_height = float(self.config.get('FORCING_MEASUREMENT_HEIGHT'))

            for idx in range(num_hru):
                att['hruId'][idx] = shp.iloc[idx][catchment_hruId_var]
                att['HRUarea'][idx] = shp.iloc[idx][catchment_area_var]
                att['latitude'][idx] = shp.iloc[idx][catchment_lat_var]
                att['longitude'][idx] = shp.iloc[idx][catchment_lon_var]
                att['hru2gruId'][idx] = shp.iloc[idx][catchment_gruId_var]
                
                att['tan_slope'][idx] = 0.1  # Only used in qbaseTopmodel modelling decision
                att['contourLength'][idx] = 30  # Only used in qbaseTopmodel modelling decision
                att['slopeTypeIndex'][idx] = 1  # Needs to be set but not used
                att['mHeight'][idx] = forcing_measurement_height
                att['downHRUindex'][idx] = 0  # All HRUs modeled as independent columns
                
                att['elevation'][idx] = -999  # Placeholder
                att['soilTypeIndex'][idx] = -999  # Placeholder
                att['vegTypeIndex'][idx] = -999  # Placeholder

                if (idx + 1) % 10 == 0:
                    self.logger.info(f"{idx + 1} out of {num_hru} HRUs completed.")

        self.logger.info(f"Attributes file created at {attribute_path / attribute_name}")
        
        self.insert_soil_class(attribute_path / attribute_name)
        self.insert_land_class(attribute_path / attribute_name)
        self.insert_elevation(attribute_path / attribute_name)


        self.logger.info(f"Attributes file created at {attribute_path / attribute_name}")



    def insert_soil_class(self, attribute_file):
        self.logger.info("Inserting soil class into attributes file")
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

    def insert_land_class(self, attribute_file):
        self.logger.info("Inserting land class into attributes file")
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

    def insert_elevation(self, attribute_file):
        self.logger.info("Inserting elevation into attributes file")
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
                for gru_id, hru_list in gru_data.items():
                    sorted_hrus = sorted(hru_list, key=lambda x: x[1], reverse=True)
                    for i, (hru_id, _) in enumerate(sorted_hrus):
                        idx = np.where(att['hruId'][:] == hru_id)[0][0]
                        if i == len(sorted_hrus) - 1:
                            att['downHRUindex'][idx] = 0  # outlet
                        else:
                            att['downHRUindex'][idx] = sorted_hrus[i+1][0]
                        self.logger.info(f"Set downHRUindex for HRU {hru_id} to {att['downHRUindex'][idx]}")


    def apply_unit_conversions(self, ds, forcing_dataset):
        if forcing_dataset == 'rdrs':
            ds['pptrate'] = ds['pptrate'] / 3600 * 1000  # Convert from m/hour to mm/s
            ds['airpres'] = ds['airpres'] * 100  # Convert from mb to Pa
            ds['airtemp'] = ds['airtemp'] + 273.15  # Convert from C to K
            ds['windspd'] = ds['windspd'] * 0.514444  # Convert from knots to m/s
        elif forcing_dataset == 'era5':
            ds['pptrate'] = ds['pptrate'] * 1000 / 3600  # Convert from m/hour to mm/s
        elif forcing_dataset == 'carra':
            # CARRA data is already in the correct units
            pass
        return ds
    

def forcing_dataset_mapping():
    return {
        'rdrs': {
            'RDRS_v2.1_P_FI_SFC': 'LWRadAtm',
            'RDRS_v2.1_P_FB_SFC': 'SWRadAtm',
            'RDRS_v2.1_A_PR0_SFC': 'pptrate',
            'RDRS_v2.1_P_P0_SFC': 'airpres',
            'RDRS_v2.1_P_TT_09944': 'airtemp',
            'RDRS_v2.1_P_HU_09944': 'spechum',
            'RDRS_v2.1_P_UVC_09944': 'windspd'
        },
        'era5': {
            't2m': 'airtemp',
            'd2m': 'dewpoint',
            'sp': 'airpres',
            'tp': 'pptrate',
            'ssrd': 'SWRadAtm',
            'strd': 'LWRadAtm',
            'u10': 'windspd_u',
            'v10': 'windspd_v'
        },
        'carra': {
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

