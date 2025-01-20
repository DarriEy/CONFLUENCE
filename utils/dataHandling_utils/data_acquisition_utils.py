import os
import sys
import time
from typing import Dict, Any
import tempfile
from pathlib import Path
import subprocess
import cdsapi # type: ignore
import calendar
import netCDF4 as nc4 # type: ignore
import numpy as np # type: ignore
from datetime import datetime
import requests # type: ignore
import shutil
import tarfile
from netrc import netrc
from hs_restclient import HydroShare, HydroShareAuthBasic # type: ignore
from osgeo import gdal # type: ignore
import urllib.request
import ssl

class gistoolRunner:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self.config.get('TOOL_CACHE')
        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'

        #Import required CONFLUENCE utils
        sys.path.append(str(self.code_dir))
        from utils.configHandling_utils.config_utils import get_default_path # type: ignore

        #Get the path to the directory containing the gistool script
        self.gistool_path = get_default_path(self.config, self.data_dir, self.config['GISTOOL_PATH'], 'installs/gistool', self.logger)
    
    def create_gistool_command(self, dataset, output_dir, lat_lims, lon_lims, variables, start_date=None, end_date=None):
        dataset_dir = dataset
        if dataset == 'soil_class':
            dataset_dir = 'soil_classes'
 
        gistool_command = [
            f"{self.gistool_path}/extract-gis.sh",
            f"--dataset={dataset}",
            f"--dataset-dir={self.config['GISTOOL_DATASET_ROOT']}{dataset_dir}",
            f"--output-dir={output_dir}",
            f"--lat-lims={lat_lims}",
            f"--lon-lims={lon_lims}",
            f"--variable={variables}",
            f"--prefix=domain_{self.domain_name}_",
            f"--lib-path={self.config['GISTOOL_LIB_PATH']}"
            "--submit-job",
            "--print-geotiff=true",
            f"--cache={self.tool_cache}",
            f"--account={self.config['TOOL_ACCOUNT']}"
        ] 
        
        if start_date and end_date:
            gistool_command.extend([
                f"--start-date={start_date}",
                f"--end-date={end_date}"
            ])

        print(gistool_command)
        return gistool_command  # This line ensures the function always returns the command list
    
    def execute_gistool_command(self, gistool_command):
        
        #Run the gistool command
        try:
            subprocess.run(gistool_command, check=True)
            self.logger.info("gistool completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running gistool: {e}")
            raise
        self.logger.info("Geospatial data acquisition process completed")

class datatoolRunner:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self.config.get('TOOL_CACHE')
        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'

        #Import required CONFLUENCE utils
        sys.path.append(str(self.code_dir))
        from utils.configHandling_utils.config_utils import get_default_path # type: ignore

        #Get the path to the directory containing the gistool script
        self.datatool_path = get_default_path(self.config, self.data_dir, self.config['DATATOOL_PATH'], 'installs/datatool', self.logger)
    
    def create_datatool_command(self, dataset, output_dir, start_date, end_date, lat_lims, lon_lims, variables):
        dataset_dir = dataset
        if dataset == "ERA5":
            dataset_dir = 'era5'
        elif dataset == "RDRS":
            dataset_dir = 'rdrsv2.1'

        datatool_command = [
        f"{self.datatool_path}/extract-dataset.sh",
        f"--dataset={dataset}",
        f"--dataset-dir={self.config['DATATOOL_DATASET_ROOT']}{dataset_dir}",
        f"--output-dir={output_dir}",
        f"--start-date={start_date}",
        f"--end-date={end_date}",
        f"--lat-lims={lat_lims}",
        f"--lon-lims={lon_lims}",
        f"--variable={variables}",
        f"--prefix=domain_{self.domain_name}_",
        #"--submit-job",
        f"--cache={self.tool_cache}",
        f"--account={self.config['TOOL_ACCOUNT']}"
        ] 

        return datatool_command
    
    def execute_datatool_command(self, datatool_command):
        
        #Run the datatool command
        try:
            subprocess.run(datatool_command, check=True)
            self.logger.info("datatool completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running datatool: {e}")
            raise
        self.logger.info("Meteorological data acquisition process completed")

class carraDownloader:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        #Import required CONFLUENCE utils
        sys.path.append(str(self.code_dir))


    def download_carra_data(self, year, month, output_file):
        ''' Function to download monthly batch of CARRA data'''
        if output_file.exists():
            print(f"File already exists for {year}-{month}, skipping download.")
            return True

        c = cdsapi.Client()
        
        try:
            c.retrieve(
                'reanalysis-carra-single-levels',
                {
                    'domain': 'west_domain',
                    'level_type': 'surface_or_atmosphere',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_specific_humidity',
                        '2m_temperature', 'surface_net_solar_radiation', 'thermal_surface_radiation_downwards',
                        'surface_pressure', 'total_precipitation',
                    ],
                    'product_type': 'forecast',
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                    ],
                    'year': year,
                    'month': month,
                    'day': [f"{i:02d}" for i in range(1, 32)],
                    'format': 'grib',
                    'leadtime_hour': '1',
                },
                output_file)
            print(f"Download complete for {year}-{month}")
            return True
        except Exception as e:
            print(f"Download failed for {year}-{month}: {str(e)}")
            return False
        
    def run_download(self, year_start, year_end):
        years = range(year_start,year_end)
        months = [f"{i:02d}" for i in range(1, 13)]
        carra_folder = self.project_dir / 'forcing' / 'raw_data'


        for year in years:
            for month in months:
                raw_file = carra_folder / f'carra_raw_{year}_{month}.grib'
                
                
                # Check if raw file exists
                if raw_file.exists():
                    print(f"Raw file exists for {year}-{month}, skipping download.")
                else:
                    # Download raw file if it doesn't exist
                    if not self.download_carra_data(year, month, raw_file):
                        print(f"Failed to download data for {year}-{month}, skipping.")
                        continue


class era5Downloader:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

    def download_era5_pressurelevelData(self, year, month, output_file):
        self.logger.info(f"Downloading ERA5 pressure level data for {year}-{month}")
        
        c = cdsapi.Client()
        
        date = f"{year}-{month:02d}-01/to/{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        
        try:
            c.retrieve(
                'reanalysis-era5-complete',
                {
                    'class': 'ea',
                    'expver': '1',
                    'stream': 'oper',
                    'type': 'an',
                    'levtype': 'ml',
                    'levelist': '137',
                    'param': '130/131/132/133',
                    'date': date,
                    'time': '00/to/23/by/1',
                    'area': self.config.get('BOUNDING_BOX_COORDS'),
                    'grid': '0.25/0.25',
                    'format': 'netcdf',
                },
                output_file
            )
            self.logger.info(f"Successfully downloaded ERA5 pressure level data for {year}-{month}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download ERA5 pressure level data for {year}-{month}: {str(e)}")
            return False

    def download_era5_surfacelevelData(self, year, month, output_file):
        self.logger.info(f"Downloading ERA5 surface level data for {year}-{month}")
        
        c = cdsapi.Client()
        
        date = f"{year}-{month:02d}-01/{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'mean_surface_downward_long_wave_radiation_flux',
                        'mean_surface_downward_short_wave_radiation_flux',
                        'mean_total_precipitation_rate',
                        'surface_pressure',
                    ],
                    'date': date,
                    'time': '00/to/23/by/1',
                    'area': self.config.get('BOUNDING_BOX_COORDS'),
                    'grid': '0.25/0.25',
                    'format'  : 'netcdf'
                },
                output_file
            )
            self.logger.info(f"Successfully downloaded ERA5 surface level data for {year}-{month}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to download ERA5 surface level data for {year}-{month}: {str(e)}")
            return False

    def run_download(self, year_start, year_end):
        years = range(year_start, year_end + 1)
        months = range(1, 13)
        era5_folder = self.project_dir / 'forcing' / 'raw_data'
        era5_folder.mkdir(parents=True, exist_ok=True)

        for year in years:
            for month in months:
                pressure_file = era5_folder / f'ERA5_pressureLevel137_{year}{month:02d}.nc'
                surface_file = era5_folder / f'ERA5_surface_{year}{month:02d}.nc'

                if not pressure_file.exists():
                    if not self.download_era5_pressurelevelData(year, month, pressure_file):
                        self.logger.warning(f"Failed to download pressure level data for {year}-{month}, skipping.")
                        continue
                else:
                    self.logger.info(f"Pressure level file exists for {year}-{month}, skipping download.")

                if not surface_file.exists():
                    if not self.download_era5_surfacelevelData(year, month, surface_file):
                        self.logger.warning(f"Failed to download surface level data for {year}-{month}, skipping.")
                        continue
                else:
                    self.logger.info(f"Surface level file exists for {year}-{month}, skipping download.")

        self.logger.info("ERA5 download process completed")
        
        # Now merge the pressureLevel and surfaceLevel data
        self.era5Merger()
        self.logger.info("ERA5 data merging process completed")

    def era5Merger(self):
        self.logger.info("Starting ERA5 surface and pressure level merger")
        raw_time = [
                    self.config.get('EXPERIMENT_TIME_START').split('-')[0],  # Get year from full datetime
                    self.config.get('EXPERIMENT_TIME_END').split('-')[0]
                ]
        for year in range(raw_time[0], raw_time[1] + 1):
            for month in range(1, 13):
                data_pres = f'ERA5_pressureLevel137_{year}{month:02d}.nc'
                data_surf = f'ERA5_surface_{year}{month:02d}.nc'
                data_dest = f'ERA5_merged_{year}{month:02d}.nc'

                source_path = self.project_dir / 'forcing' / 'raw_data'
                dest_path = self.project_dir / 'forcing' / 'merged_data'
                dest_path.mkdir(parents=True, exist_ok=True)

                self.logger.info(f"Merging {data_surf} and {data_pres}")

                try:
                    with nc4.Dataset(source_path / data_pres) as src1, \
                         nc4.Dataset(source_path / data_surf) as src2, \
                         nc4.Dataset(dest_path / data_dest, "w") as dest:

                        # Transfer dimensions and coordinates
                        for name, dimension in src2.dimensions.items():
                            dest.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

                        for name in ['longitude', 'latitude', 'time']:
                            dest.createVariable(name, src2[name].datatype, src2[name].dimensions)
                            dest[name][:] = src2[name][:]
                            dest[name].setncatts(src2[name].__dict__)

                        # Process and transfer variables
                        var_mapping = {
                            'sp': 'airpres',
                            'msdwlwrf': 'LWRadAtm',
                            'msdwswrf': 'SWRadAtm',
                            'mtpr': 'pptrate',
                            't': 'airtemp',
                            'q': 'spechum'
                        }

                        for src_name, dest_name in var_mapping.items():
                            src = src2 if src_name in src2.variables else src1
                            data = src[src_name][:]
                            if src_name != 't':  # Apply non-negativity constraint except for temperature
                                data[data < 0] = 0
                            
                            dest.createVariable(dest_name, 'f4', ('time', 'latitude', 'longitude'))
                            dest[dest_name][:] = data
                            for attr in ['units', 'long_name', 'standard_name']:
                                if attr in src[src_name].ncattrs():
                                    dest[dest_name].setncattr(attr, src[src_name].getncattr(attr))
                            dest[dest_name].setncattr('missing_value', -999)

                        # Calculate and store wind speed
                        u = src1['u'][:]
                        v = src1['v'][:]
                        wind_speed = np.sqrt(u**2 + v**2)
                        dest.createVariable('windspd', 'f4', ('time', 'latitude', 'longitude'))
                        dest['windspd'][:] = wind_speed
                        dest['windspd'].setncattr('units', 'm s-1')
                        dest['windspd'].setncattr('long_name', 'wind speed at the measurement height')
                        dest['windspd'].setncattr('standard_name', 'wind_speed')
                        dest['windspd'].setncattr('missing_value', -999)

                        # Set some general attributes
                        dest.setncattr('History', f'Created {datetime.now()}')
                        dest.setncattr('Description', 'ERA5 surface and pressure level data merged for SUMMA')

                    self.logger.info(f"Successfully merged {data_surf} and {data_pres} into {data_dest}")

                except Exception as e:
                    self.logger.error(f"Error merging files for {year}-{month:02d}: {str(e)}")

        self.logger.info("ERA5 surface and pressure level merger completed")

class meritDownloader:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # MERIT specific attributes
        self.merit_url = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0.1/"
        self.merit_template = "elv_{}{}.tar"
        self.merit_path = self.project_dir / 'parameters' / 'dem' / '1_MERIT_raw_data'


    def get_download_area(self):
        coordinates = self.config.get('BOUNDING_BOX_COORDS').split('/')
        domain_min_lon = float(coordinates[1])
        domain_max_lon = float(coordinates[3])
        domain_min_lat = float(coordinates[2])
        domain_max_lat = float(coordinates[0])

        lon_edges = np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])
        lat_edges = np.array([-60, -30, 0, 30, 60, 90])

        dl_lon_all = np.array(['w180', 'w150', 'w120', 'w090', 'w060', 'w030', 'e000', 'e030', 'e060', 'e090', 'e120', 'e150'])
        dl_lat_all = np.array(['s60', 's30', 'n00', 'n30', 'n60'])

        dl_lons = dl_lon_all[(domain_min_lon < lon_edges[1:]) & (domain_max_lon > lon_edges[:-1])]
        dl_lats = dl_lat_all[(domain_min_lat < lat_edges[1:]) & (domain_max_lat > lat_edges[:-1])]

        return dl_lons, dl_lats

    def download_merit_data(self):
        self.merit_path.mkdir(parents=True, exist_ok=True)
        dl_lons, dl_lats = self.get_download_area()

        merit_login = {}
        with open(os.path.expanduser("~/.merit")) as file:
            for line in file:
                key, val = line.split(':')
                merit_login[key] = val.strip()

        usr, pwd = merit_login['name'], merit_login['pass']

        # Create a password manager
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self.merit_url, usr, pwd)
        auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

        # Create an SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Create and install the opener
        opener = urllib.request.build_opener(auth_handler, urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)

        retries_max = 10

        for dl_lon in dl_lons:
            for dl_lat in dl_lats:
                if (dl_lat == 'n00' and dl_lon == 'w150') or \
                   (dl_lat == 's60' and dl_lon == 'w150') or \
                   (dl_lat == 's60' and dl_lon == 'w120'):
                    continue

                file_url = f"{self.merit_url}{self.merit_template.format(dl_lat, dl_lon)}"
                file_name = file_url.split('/')[-1].strip()
                file_path = self.merit_path / file_name

                if file_path.is_file() and file_path.stat().st_size > 0:
                    self.logger.info(f"File {file_name} already exists and is not empty. Skipping.")
                    continue

                self.logger.info(f"Attempting to download {file_name}")

                retries_cur = 1
                while retries_cur <= retries_max:
                    try:
                        with urllib.request.urlopen(file_url, timeout=30) as response, open(file_path, 'wb') as out_file:
                            shutil.copyfileobj(response, out_file)

                        # Verify file size after download
                        actual_size = file_path.stat().st_size
                        if actual_size == 0:
                            self.logger.error(f"Downloaded file {file_name} is empty. Deleting and retrying.")
                            file_path.unlink()
                            raise Exception("Empty file downloaded")

                        self.logger.info(f"Successfully downloaded {file_name}")
                        break

                    except urllib.error.URLError as e:
                        self.logger.error(f"Error downloading {file_name} on try {retries_cur}: {str(e)}")
                        retries_cur += 1
                        if retries_cur > retries_max:
                            self.logger.error(f"Max retries reached for {file_name}. Skipping.")
                        else:
                            time.sleep(10)  # Wait before retrying
                    except Exception as e:
                        self.logger.error(f"Unexpected error downloading {file_name}: {str(e)}")
                        if file_path.exists():
                            file_path.unlink()
                        retries_cur += 1
                        if retries_cur > retries_max:
                            self.logger.error(f"Max retries reached for {file_name}. Skipping.")
                        else:
                            time.sleep(10)  # Wait before retrying

        self.logger.info("MERIT data download completed")

    def unpack_and_clean(self):
        self.logger.info("Unpacking downloaded tar files")
        for tar_file in self.merit_path.glob('*.tar'):
            with tarfile.open(tar_file, 'r') as tar:
                tar.extractall(path=self.merit_path)
            os.remove(tar_file)
            self.logger.info(f"Unpacked and removed {tar_file}")
        self.logger.info("Unpacking completed")

    def merge_files(self):
        # Implementation for merging files (if needed)
        pass

    def subset_by_bbox(self):
        # Implementation for subsetting by bounding box (if needed)
        pass

    def run_download(self):
        self.download_merit_data()
        self.unpack_and_clean()
        self.merge_files()
        self.subset_by_bbox()

class soilgridsDownloader:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        self.download_ID = "1361509511e44adfba814f6950c6e742"
        self.soil_raw_path = self.project_dir / 'parameters' / 'soilclass' / '1_soil_classes_global'
        self.soil_domain_path = self.project_dir / 'parameters' / 'soilclass' / '2_soil_classes_domain'
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        self.soil_domain_name = self.config.get('SOIL_CLASS_NAME', soil_name)

    def download_soilgrids_data(self):
        self.soil_raw_path.mkdir(parents=True, exist_ok=True)
        
        hydroshare_login = {}
        with open(os.path.expanduser("~/.hydroshare")) as file:
            for line in file:
                key, val = line.split(':')
                hydroshare_login[key] = val.strip()

        usr, pwd = hydroshare_login['name'], hydroshare_login['pass']

        auth = HydroShareAuthBasic(username=usr, password=pwd)
        hs = HydroShare(auth=auth)

        try:
            out = hs.getResourceFile(f"www.hydroshare.org/{self.download_ID}", "usda_mode_soilclass_250m_ll.tif", destination=self.soil_raw_path)
            self.logger.info(f"Successfully downloaded SOILGRIDS data: {out}")
        except Exception as e:
            self.logger.error(f"Error downloading SOILGRIDS data: {str(e)}")

    def extract_domain(self):
        self.soil_domain_path.mkdir(parents=True, exist_ok=True)

        coordinates = self.config.get('BOUNDING_BOX_COORDS').split('/')
        bbox = (float(coordinates[1]), float(coordinates[0]), float(coordinates[3]), float(coordinates[2]))

        soil_raw_files = list(self.soil_raw_path.glob('*.tif'))
        if not soil_raw_files:
            self.logger.error("No .tif files found in the raw soil data directory")
            return

        old_tif = str(soil_raw_files[0])
        new_tif = str(self.soil_domain_path / self.soil_domain_name)

        try:
            translate_options = gdal.TranslateOptions(format='GTiff', projWin=bbox, creationOptions=['COMPRESS=DEFLATE'])
            ds = gdal.Translate(new_tif, old_tif, options=translate_options)
            ds = None  # Close the dataset
            self.logger.info(f"Successfully extracted domain to {new_tif}")
        except Exception as e:
            self.logger.error(f"Error extracting domain: {str(e)}")

    def cleanup(self):
        try:
            shutil.rmtree(self.soil_raw_path)
            self.logger.info(f"Removed raw soil data directory: {self.soil_raw_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning up raw soil data: {str(e)}")

    def process_soilgrids_data(self):
        self.download_soilgrids_data()
        self.extract_domain()
        self.cleanup()
        self.logger.info("SOILGRIDS data processing completed")

class modisDownloader:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        self.links_path = self._get_links_path()
        self.links_file = self.config.get('PARAMETER_LAND_LIST_NAME')
        self.modis_raw_path = self._get_modis_raw_path()
        self.modis_vrt1_path = self._get_modis_vrt1_path()
        self.modis_vrt2_path = self._get_modis_vrt2_path()
        self.modis_vrt3_path = self._get_modis_vrt3_path()
        self.modis_vrt4_path = self._get_modis_vrt4_path()
        
        self.auth_url = 'urs.earthdata.nasa.gov'
        self.netrc_folder = os.path.expanduser("~/.netrc")

    def _get_links_path(self):
        links_path = self.config.get('PARAMETER_LAND_LIST_PATH')
        return Path('./') if links_path == 'default' else Path(links_path)

    def _get_modis_raw_path(self):
        modis_path = self.config.get('PARAMETER_LAND_RAW_PATH')
        return self.project_dir / 'parameters' / 'landclass' / '1_MODIS_raw_data' if modis_path == 'default' else Path(modis_path)

    def _get_modis_vrt1_path(self):
        vrt1_path = self.config.get('PARAMETER_LAND_VRT1_PATH')
        return self.project_dir / 'parameters' / 'landclass' / '2_vrt_native_crs' if vrt1_path == 'default' else Path(vrt1_path)

    def _get_modis_vrt2_path(self):
        vrt2_path = self.config.get('PARAMETER_LAND_VRT2_PATH')
        return self.project_dir / 'parameters' / 'landclass' / '3_vrt_epsg_4326' if vrt2_path == 'default' else Path(vrt2_path)

    def _get_auth_credentials(self):
        try:
            auth = netrc(self.netrc_folder).authenticators(self.auth_url)
            if auth:
                return auth[0], auth[2]  # username and password
            else:
                self.logger.error("No credentials found for urs.earthdata.nasa.gov in .netrc file")
                return None, None
        except Exception as e:
            self.logger.error(f"Error reading .netrc file: {str(e)}")
            return None, None

    def download_modis_data(self):
        self.modis_path.mkdir(parents=True, exist_ok=True)

        usr, pwd = self._get_auth_credentials()
        if not usr or not pwd:
            self.logger.error("Unable to retrieve authentication credentials. Aborting download.")
            return

        try:
            with open(self.links_path / self.links_file, 'r') as f:
                file_list = f.readlines()
        except Exception as e:
            self.logger.error(f"Error reading file list: {str(e)}")
            return

        retries_max = 100

        for file_url in file_list:
            file_name = file_url.split('/')[-1].strip()
            file_path = self.modis_path / file_name

            if file_path.is_file():
                self.logger.info(f"File {file_name} already exists. Skipping.")
                continue

            retries_cur = 1
            while retries_cur <= retries_max:
                try:
                    with requests.get(file_url.strip(), verify=True, stream=True, auth=(usr, pwd)) as response:
                        response.raise_for_status()
                        response.raw.decode_content = True
                        with open(file_path, 'wb') as data:
                            shutil.copyfileobj(response.raw, data)
                    
                    self.logger.info(f"Successfully downloaded: {file_name}")
                    time.sleep(3)  # sleep to avoid overwhelming the server
                    break
                except requests.RequestException as e:
                    self.logger.error(f"Error downloading {file_name} on try {retries_cur}: {str(e)}")
                    retries_cur += 1
                    if retries_cur > retries_max:
                        self.logger.error(f"Max retries reached for {file_name}. Skipping.")
                    time.sleep(10)  # wait before retrying

    def _get_modis_vrt3_path(self):
        vrt3_path = self.config.get('PARAMETER_LAND_VRT3_PATH')
        return self.project_dir / 'parameters' / 'landclass' / '4_domain_vrt_epsg_4326' if vrt3_path == 'default' else Path(vrt3_path)

    def _get_modis_vrt4_path(self):
        vrt4_path = self.config.get('PARAMETER_LAND_VRT4_PATH')
        return self.project_dir / 'parameters' / 'landclass' / '5_multiband_domain_vrt_epsg_4326' if vrt4_path == 'default' else Path(vrt4_path)

    def make_vrt_per_year(self):
        self.logger.info("Creating VRT files per year")
        self.modis_vrt1_path.mkdir(parents=True, exist_ok=True)
        filelists_path = self.modis_vrt1_path / "filelists"
        filelists_path.mkdir(exist_ok=True)

        for year in range(2001, 2019):
            out_txt = filelists_path / f"MCD12Q1_filelist_{year}.txt"
            out_vrt = self.modis_vrt1_path / f"MCD12Q1_{year}.vrt"

            # Create file list
            with open(out_txt, 'w') as f:
                for file in self.modis_raw_path.glob(f"MCD12Q1.A{year}*"):
                    f.write(str(file) + "\n")

            # Create VRT
            subprocess.run([
                "gdalbuildvrt", str(out_vrt), 
                "-input_file_list", str(out_txt), 
                "-sd", "1", "-resolution", "highest"
            ], check=True)

        self.logger.info("VRT files created successfully")

    def reproject_vrt(self):
        self.logger.info("Reprojecting VRT files to EPSG:4326")
        self.modis_vrt2_path.mkdir(parents=True, exist_ok=True)

        for vrt_file in self.modis_vrt1_path.glob("MCD*.vrt"):
            output_file = self.modis_vrt2_path / vrt_file.name
            subprocess.run([
                "gdalwarp", "-of", "VRT", 
                "-t_srs", "EPSG:4326", 
                str(vrt_file), str(output_file)
            ], check=True)

        self.logger.info("VRT files reprojected successfully")

    def create_multiband_vrt(self):
        self.logger.info("Creating multiband VRT")
        self.modis_vrt4_path.mkdir(parents=True, exist_ok=True)

        output_vrt = self.modis_vrt4_path / "domain_MCD12Q1_2001_2018.vrt"
        
        # Create a temporary file list
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            for vrt_file in sorted(self.modis_vrt3_path.glob("*.vrt")):
                temp_file.write(f"{vrt_file}\n")
            temp_file_name = temp_file.name

        # Create the multiband VRT
        subprocess.run([
            "gdalbuildvrt", "-separate",
            "-input_file_list", temp_file_name,
            str(output_vrt),
            "-resolution", "highest"
        ], check=True)

        # Remove the temporary file
        os.unlink(temp_file_name)

        self.logger.info(f"Multiband VRT created: {output_vrt}")

    def run_modis_workflow(self):
        self.logger.info("Starting MODIS data download and processing")
        self.download_modis_data()
        self.make_vrt_per_year()
        self.reproject_vrt()
        self.create_multiband_vrt()
        self.logger.info("MODIS data download and processing completed")