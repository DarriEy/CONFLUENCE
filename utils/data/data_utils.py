import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore
from shapely.geometry import Point # type: ignore
import csv
from datetime import datetime
import xarray as xr # type: ignore
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data.variable_utils import VariableHandler # type: ignore

class DataAcquisitionProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')
        

    def prepare_maf_json(self) -> Path:
        """Prepare the JSON file for the Model Agnostic Framework."""

        met_path = str(self.root_path / "installs/datatool/" / "extract-dataset.sh")
        gis_path = str(self.root_path / "installs/gistool/" / "extract-gis.sh")
        easymore_client = str(self.config.get('EASYMORE_CLIENT'))

        subbasins_name = self.config.get('RIVER_BASINS_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp"

        tool_cache = self.config.get('TOOL_CACHE')
        if tool_cache == 'default':
            tool_cache = '$HOME/cache_dir/'

        variables = self.config['FORCING_VARIABLES']
        if variables == 'default':
            variables = self.variable_handler.get_dataset_variables(dataset = self.config['FORCING_DATASET'])

        maf_config = {
            "exec": {
                "met": met_path,
                "gis": gis_path,
                "remap": easymore_client
            },
            "args": {
                "met": [{
                    "dataset": self.config.get('FORCING_DATASET'),
                    "dataset-dir": str(Path(self.config.get('DATATOOL_DATASET_ROOT')) / "era5/"),
                    "variable": variables,
                    "output-dir": str(self.project_dir / "forcing/datatool-outputs"),
                    "start-date": f"{self.config.get('EXPERIMENT_TIME_START')}",
                    "end-date": f"{self.config.get('EXPERIMENT_TIME_END')}",
                    "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "prefix": f"domain_{self.domain_name}_",
                    "cache": tool_cache,
                    "account": self.config.get('TOOL_ACCOUNT'),
                    "_flags": [
                        #"submit-job",
                        #"parsable"
                    ]
                }],
                "gis": [
                    {
                        "dataset": "MODIS",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MODIS"),
                        "variable": "MCD12Q1.061",
                        "start-date": "2001-01-01",
                        "end-date": "2020-01-01",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["frac", "majority", "coords"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "soil_classes"),
                        "variable": "soil_classes",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["majority"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job"]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MERIT-Hydro"),
                        "variable": "elv,hnd",
                        "output-dir": str(self.project_dir / "attributes/gistool-outputs"),
                        "shape-file": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                        "print-geotiff": "true",
                        "stat": ["min", "max", "mean", "median"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": tool_cache,
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                        "_flags": ["include-na", "parsable"]#, "submit-job",]
                    }
                ],
                "remap": [{
                    "case-name": "remapped",
                    "cache": tool_cache,
                    "shapefile": str(self.project_dir / "shapefiles/river_basins" / subbasins_name),
                    "shapefile-id": self.config.get('RIVER_BASIN_SHP_RM_GRUID'),
                    "source-nc": str(self.project_dir / "forcing/datatool-outputs/**/*.nc*"),
                    "variable-lon": "lon",
                    "variable-lat": "lat",
                    "variable": variables,
                    "remapped-var-id": "hruId",
                    "remapped-dim-id": "hru",
                    "output-dir": str(self.project_dir / "forcing/easymore-outputs/") + '/',
                    "job-conf": self.config.get('EASYMORE_JOB_CONF'),
                    #"_flags": ["submit-job"]
                }]
            },
            "order": {
                "met": 1,
                "gis": -1,
                "remap": 2
            }
        }

        # Save the JSON file
        json_path = self.project_dir / "forcing/maf_config.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(maf_config, f, indent=2)

        self.logger.info(f"MAF configuration JSON saved to: {json_path}")
        return json_path

    def run_data_acquisition(self):
        """Run the data acquisition process using MAF."""
        json_path = self.prepare_maf_json()
        self.logger.info("Starting data acquisition process")


        maf_script = self.root_path / "installs/MAF/02_model_agnostic_component/model-agnostic.sh"
        
        #Run the MAF script
        try:
            subprocess.run([str(maf_script), str(json_path)], check=True)
            self.logger.info("Model Agnostic Framework completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running Model Agnostic Framework: {e}")
            raise
        self.logger.info("Data acquisition process completed")
    
    def _get_file_path(self, file_type, file_def_path, file_name):
        """
        Construct file paths based on configuration.

        Args:
            file_type (str): Type of the file (used as a key in config).
            file_def_path (str): Default path relative to project directory.
            file_name (str): Name of the file.

        Returns:
            Path: Constructed file path.
        """
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))

class DataCleanupProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def cleanup_and_checks(self):
        """Perform cleanup and checks on the MAF output."""
        self.logger.info("Performing cleanup and checks on MAF output")
        
        # Define paths
        path_soil_type = self.project_dir / f'attributes/soil_class/domain_{self.domain_name}_stats_soil_classes.csv'
        path_landcover_type = self.project_dir / f'attributes/land_class/domain_{self.domain_name}_stats_NA_NALCMS_landcover_2020_30m.csv'
        path_elevation_mean = self.project_dir / f'attributes/elevation/domain_{self.domain_name}_stats_elv.csv'

        # Read files
        soil_type = pd.read_csv(path_soil_type)
        landcover_type = pd.read_csv(path_landcover_type)
        elevation_mean = pd.read_csv(path_elevation_mean)

        # Sort by COMID
        soil_type = soil_type.sort_values(by='COMID').reset_index(drop=True)
        landcover_type = landcover_type.sort_values(by='COMID').reset_index(drop=True)
        elevation_mean = elevation_mean.sort_values(by='COMID').reset_index(drop=True)

        # Check if COMIDs are the same across all files
        if not (len(soil_type) == len(landcover_type) == len(elevation_mean) and
                (soil_type['COMID'] == landcover_type['COMID']).all() and
                (landcover_type['COMID'] == elevation_mean['COMID']).all()):
            raise ValueError("COMIDs are not consistent across soil, landcover, and elevation files")

        # Process soil type
        majority_value = soil_type['majority'].replace(0, np.nan).mode().iloc[0]
        soil_type['majority'] = soil_type['majority'].replace(0, majority_value).fillna(majority_value)
        if self.config.get('UNIFY_SOIL', False):
            soil_type['majority'] = majority_value

        # Process landcover
        min_land_fraction = self.config.get('MINIMUM_LAND_FRACTION', 0.01)
        for col in landcover_type.columns:
            if col.startswith('frac_'):
                landcover_type[col] = landcover_type[col].apply(lambda x: 0 if x < min_land_fraction else x)
        
        for index, row in landcover_type.iterrows():
            frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
            row_sum = row[frac_columns].sum()
            if row_sum > 0:
                for col in frac_columns:
                    landcover_type.at[index, col] /= row_sum

        num_land_cover = self.config.get('NUM_LAND_COVER', 20)
        missing_columns = [f"frac_{i}" for i in range(1, num_land_cover+1) if f"frac_{i}" not in landcover_type.columns]
        for col in missing_columns:
            landcover_type[col] = 0

        frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
        frac_columns.sort(key=lambda x: int(x.split('_')[1]))
        sorted_columns = [col for col in landcover_type.columns if col not in frac_columns] + frac_columns
        landcover_type = landcover_type.reindex(columns=sorted_columns)

        for col in frac_columns:
            if landcover_type.loc[0, col] < 0.00001:
                landcover_type.loc[0, col] = 0.00001

        # Process elevation
        elevation_mean['mean'].fillna(0, inplace=True)

        # Save modified files
        soil_type.to_csv(self.project_dir / 'attributes/soil_class/modified_domain_stats_soil_classes.csv', index=False)
        landcover_type.to_csv(self.project_dir / 'attributes/land_class/modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv', index=False)
        elevation_mean.to_csv(self.project_dir / 'attributes/elevation/modified_domain_stats_elv.csv', index=False)

        self.logger.info("Cleanup and checks completed")

    

class DataPreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        self.logger.info("Calculating elevation statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)

        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        dem_path = self._get_file_path('DEM_PATH', 'attributes/elevation/dem', dem_name)
        dem_name = self.config.get('INTERSECT_DEM_NAME')
        if dem_name == 'default':
            dem_name = 'catchment_with_dem.shp'
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem', dem_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(intersect_path), exist_ok=True)

        catchment_gdf = gpd.read_file(catchment_path)
        nodata_value = self.get_nodata_value(dem_path)

        with rasterio.open(dem_path) as src:
            affine = src.transform
            dem_data = src.read(1)

        stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
        result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
        
        if 'elev_mean' in catchment_gdf.columns:
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']
        else:
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']

        result_df = result_df.drop(columns=['elev_mean_new'])
        catchment_gdf.to_file(intersect_path)

    def calculate_soil_stats(self):
        self.logger.info("Calculating soil statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass/', soil_name)
        intersect_soil_name = self.config.get('INTERSECT_SOIL_NAME')
        if intersect_soil_name == 'default':
            intersect_soil_name = 'catchment_with_soilclass.shp'
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids', intersect_soil_name)
        self.logger.info(f'processing landclasses: {soil_path}')

        if not intersect_path.exists() or self.config.get('FORCE_RUN_ALL_STEPS') == True:
            intersect_path.parent.mkdir(parents=True, exist_ok=True)

            catchment_gdf = gpd.read_file(catchment_path)
            nodata_value = self.get_nodata_value(soil_path)

            with rasterio.open(soil_path) as src:
                affine = src.transform
                soil_data = src.read(1)

            stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats)
            
            # Find the most common soil class (excluding 'count' column)
            soil_columns = [col for col in result_df.columns if col != 'count']
            most_common_soil = result_df[soil_columns].sum().idxmax()
            
            # Fill NaN values with the most common soil class (fallback in case very small HRUs)
            if result_df.isna().any().any():
                self.logger.warning("NaN values found in soil statistics. Filling with most common soil class. Please check HRU's size or use higher resolution land class raster")
                result_df = result_df.fillna({col: (0 if col == 'count' else most_common_soil) for col in result_df.columns})            

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'USGS_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            result_df = result_df.astype({col: int for col in result_df.columns if col != 'count'})

            # Merge with original GeoDataFrame
            for col in result_df.columns:
                if col != 'count':
                    catchment_gdf[col] = result_df[col]

            try:
                catchment_gdf.to_file(intersect_path)
                self.logger.info(f"Soil statistics calculated and saved to {intersect_path}")
            except Exception as e:
                self.logger.error(f"Failed to save file: {e}")
                raise

    def calculate_land_stats(self):
        self.logger.info("Calculating land statistics")
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"
        land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass/', land_name)
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        if intersect_name == 'default':
            intersect_name = 'catchment_with_landclass.shp'
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass', intersect_name)
        self.logger.info(f'processing landclasses: {land_path}')

        if not intersect_path.exists() or self.config.get('FORCE_RUN_ALL_STEPS') == True:
            intersect_path.parent.mkdir(parents=True, exist_ok=True)

            catchment_gdf = gpd.read_file(catchment_path)
            nodata_value = self.get_nodata_value(land_path)

            with rasterio.open(land_path) as src:
                affine = src.transform
                land_data = src.read(1)

            stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
            result_df = pd.DataFrame(stats)
            
            # Find the most common land class (excluding 'count' column)
            land_columns = [col for col in result_df.columns if col != 'count']
            most_common_land = result_df[land_columns].sum().idxmax()
            
            # Fill NaN values with the most common land class (fallback in case very small HRUs)
            if result_df.isna().any().any():
                self.logger.warning("NaN values found in land statistics. Filling with most common land class. Please check HRU's size or use higher resolution land class raster")
                result_df = result_df.fillna({col: (0 if col == 'count' else most_common_land) for col in result_df.columns})

            def rename_column(x):
                if x == 'count':
                    return x
                try:
                    return f'IGBP_{int(float(x))}'
                except ValueError:
                    return x

            result_df = result_df.rename(columns=rename_column)
            result_df = result_df.astype({col: int for col in result_df.columns if col != 'count'})

            # Merge with original GeoDataFrame
            for col in result_df.columns:
                if col != 'count':
                    catchment_gdf[col] = result_df[col]

            try:
                catchment_gdf.to_file(intersect_path)
                self.logger.info(f"Land statistics calculated and saved to {intersect_path}")
            except Exception as e:
                self.logger.error(f"Failed to save file: {e}")
                raise

    def process_zonal_statistics(self):
        self.calculate_elevation_stats()
        self.calculate_soil_stats()
        self.calculate_land_stats()
        self.logger.info("All zonal statistics processed")

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))
        
class ObservedDataProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.forcing_time_step_size = int(self.config.get('FORCING_TIME_STEP_SIZE'))
        self.data_provider = self.config.get('STREAMFLOW_DATA_PROVIDER', 'USGS').upper()

        self.streamflow_raw_path = self._get_file_path('STREAMFLOW_RAW_PATH', 'observations/streamflow/raw_data', '')
        self.streamflow_processed_path = self._get_file_path('STREAMFLOW_PROCESSED_PATH', 'observations/streamflow/preprocessed', '')
        self.streamflow_raw_name = self.config.get('STREAMFLOW_RAW_NAME')

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_resample_freq(self):
        if self.forcing_time_step_size == 3600:
            return 'h'
        if self.forcing_time_step_size == 10800:
            return 'h'
        elif self.forcing_time_step_size == 86400:
            return 'D'
        else:
            return f'{self.forcing_time_step_size}s'

    def process_streamflow_data(self):
        try:
            if self.data_provider == 'USGS':
                if self.config.get('DOWNLOAD_USGS_DATA') == True:
                    self._download_and_process_usgs_data()
                else:
                    self._process_usgs_data()

            elif self.data_provider == 'WSC':
                if self.config.get('DOWNLOAD_WSC_DATA') == True:
                    self._extract_and_process_hydat_data()
                else:
                    self._process_wsc_data()
            elif self.data_provider == 'VI':
                self._process_vi_data()
            else:
                self.logger.error(f"Unsupported streamflow data provider: {self.data_provider}")
                raise ValueError(f"Unsupported streamflow data provider: {self.data_provider}")
        except Exception as e:
            self.logger.error(f'Issue in streamflow data preprocessing: {e}')

    def _extract_and_process_hydat_data(self):
        """
        Process Water Survey of Canada (WSC) streamflow data by fetching it directly from the HYDAT SQLite database.
        
        This function fetches discharge data for the specified WSC station,
        processes it, and resamples it to the configured time step.
        """
        import sqlite3
        import pandas as pd
        from datetime import datetime, timedelta
        from pathlib import Path
        
        self.logger.info("Processing WSC streamflow data from HYDAT database")
        
        # Get configuration parameters
        station_id = self.config.get('STATION_ID')
        hydat_path = self.config.get('HYDAT_PATH')
        if hydat_path == 'default':
            hydat_path = f"{str(self.project_dir.parent().parent() / 'geospatial-data' / 'hydat')}/Hydat.sqlite3"
        
        # Check if HYDAT_PATH exists
        if not hydat_path or not Path(hydat_path).exists():
            self.logger.error(f"HYDAT database not found at: {hydat_path}")
            raise FileNotFoundError(f"HYDAT database not found at: {hydat_path}")
        
        # Parse and format the start date properly
        start_date_raw = self.config.get('EXPERIMENT_TIME_START')
        try:
            # Try to parse the date string with various formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    parsed_date = datetime.strptime(start_date_raw, fmt)
                    start_date = parsed_date.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                # If none of the formats match, use a default format
                self.logger.warning(f"Could not parse start date: {start_date_raw}. Using first 10 characters as YYYY-MM-DD.")
                start_date = start_date_raw[:10]
        except Exception as e:
            self.logger.warning(f"Error parsing start date: {e}. Using default date format.")
            start_date = start_date_raw[:10]
        
        # Parse the date components for SQL queries
        try:
            start_year = int(start_date.split('-')[0])
            end_year = datetime.now().year
            self.logger.info(f"Querying data from year {start_year} to {end_year}")
        except Exception as e:
            self.logger.warning(f"Error parsing date components: {e}. Using default range.")
            start_year = 1900
            end_year = datetime.now().year
        
        # Log the station and date range
        self.logger.info(f"Retrieving discharge data for WSC station {station_id} from HYDAT database")
        self.logger.info(f"Database path: {hydat_path}")
        self.logger.info(f"Time period: {start_year} to {end_year}")
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(hydat_path)
            
            # First, check if the station exists in the database
            station_query = "SELECT * FROM STATIONS WHERE STATION_NUMBER = ?"
            station_df = pd.read_sql_query(station_query, conn, params=(station_id,))
            
            if station_df.empty:
                self.logger.error(f"Station {station_id} not found in HYDAT database")
                raise ValueError(f"Station {station_id} not found in HYDAT database")
            
            self.logger.info(f"Found station {station_id} in HYDAT database")
            if 'STATION_NAME' in station_df.columns:
                self.logger.info(f"Station name: {station_df['STATION_NAME'].iloc[0]}")
            
            # Query for daily discharge data
            # HYDAT stores discharge data in DLY_FLOWS table
            # The column names are like FLOW1, FLOW2, ... FLOW31 for each day of the month
            query = """
            SELECT * FROM DLY_FLOWS 
            WHERE STATION_NUMBER = ? 
            AND YEAR >= ? AND YEAR <= ?
            ORDER BY YEAR, MONTH
            """
            
            self.logger.info(f"Executing SQL query for daily flows...")
            dly_flow_df = pd.read_sql_query(query, conn, params=(station_id, start_year, end_year))
            
            if dly_flow_df.empty:
                self.logger.error(f"No flow data found for station {station_id} in the specified date range")
                raise ValueError(f"No flow data found for station {station_id} in the specified date range")
            
            self.logger.info(f"Retrieved {len(dly_flow_df)} monthly records from HYDAT")
            
            # Now we need to reshape the data from the HYDAT format to a time series
            # HYDAT stores each month as a row, with columns FLOW1, FLOW2, ... FLOW31
            
            # Create an empty list to store the time series data
            time_series_data = []
            
            # Process each row (each row is a month of data)
            for _, row in dly_flow_df.iterrows():
                year = row['YEAR']
                month = row['MONTH']
                
                # Days in the month (accounting for leap years)
                days_in_month = 31  # Default max
                if month == 2:  # February
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):  # Leap year
                        days_in_month = 29
                    else:
                        days_in_month = 28
                elif month in [4, 6, 9, 11]:  # April, June, September, November
                    days_in_month = 30
                
                # Extract flow values for each day and create a date
                for day in range(1, days_in_month + 1):
                    flow_col = f'FLOW{day}'
                    if flow_col in row and not pd.isna(row[flow_col]):
                        date = f"{year}-{month:02d}-{day:02d}"
                        flow = row[flow_col]
                        
                        # Check for data flags - HYDAT has flags for data quality
                        symbol_col = f'SYMBOL{day}'
                        symbol = row.get(symbol_col, '')
                        
                        # Skip values with certain flags if needed
                        # E.g., 'E' for Estimate, 'A' for Partial Day, etc.
                        # Uncomment if you want to filter based on symbols
                        # if symbol in ['B', 'D', 'E']:
                        #     continue
                        
                        time_series_data.append({'date': date, 'flow': flow, 'symbol': symbol})
            
            # Convert to DataFrame
            df = pd.DataFrame(time_series_data)
            
            if df.empty:
                self.logger.error("No valid flow data found after processing")
                raise ValueError("No valid flow data found after processing")
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Sort index to ensure chronological order
            df.sort_index(inplace=True)
            
            # Filter to the exact date range we want
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
            
            # Check if we have data after filtering
            if df.empty:
                self.logger.error("No data available after filtering to the specified date range")
                raise ValueError("No data available for the specified date range")
            
            self.logger.info(f"Processed {len(df)} daily flow records")
            
            # Create the discharge_cms column (HYDAT data is in m³/s)
            df['discharge_cms'] = df['flow']
            
            # Basic statistics for logging
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Min flow: {df['discharge_cms'].min()} m³/s")
            self.logger.info(f"Max flow: {df['discharge_cms'].max()} m³/s")
            self.logger.info(f"Mean flow: {df['discharge_cms'].mean()} m³/s")
            
            # Call the resampling and saving function
            self._resample_and_save(df['discharge_cms'])
            
            self.logger.info(f"Successfully processed WSC data for station {station_id}")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error processing WSC data from HYDAT: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
    def _process_vi_data(self):
        self.logger.info("Processing VI (Iceland) streamflow data")
        vi_data = pd.read_csv(self.streamflow_raw_path / self.streamflow_raw_name, 
                              sep=';', 
                              header=None, 
                              names=['YYYY', 'MM', 'DD', 'qobs', 'qc_flag'],
                              parse_dates={'datetime': ['YYYY', 'MM', 'DD']},
                              na_values='',
                              skiprows = 1)

        vi_data['discharge_cms'] = pd.to_numeric(vi_data['qobs'], errors='coerce')
        vi_data.set_index('datetime', inplace=True)

        # Filter out data with qc_flag values indicating unreliable measurements
        # Adjust this based on the specific qc_flag values that indicate reliable data
        #reliable_data = vi_data[vi_data['qc_flag'] <= 100]

        self._resample_and_save(vi_data['discharge_cms'])

    def _download_and_process_usgs_data(self):
        """
        Process USGS streamflow data by fetching it directly from USGS API.
        
        This function fetches discharge data for the specified USGS station,
        converts it from cubic feet per second (cfs) to cubic meters per second (cms),
        and resamples it to the configured time step.
        """
        import requests
        import io
        from datetime import datetime

        self.logger.info("Processing USGS streamflow data directly from API")
        
        # Get configuration parameters
        station_id = self.config.get('STATION_ID')
        
        # Parse and format the start date properly
        start_date_raw = self.config.get('EXPERIMENT_TIME_START')
        try:
            # Try to parse the date string with various formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    parsed_date = datetime.strptime(start_date_raw, fmt)
                    start_date = parsed_date.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                # If none of the formats match, use a default format
                self.logger.warning(f"Could not parse start date: {start_date_raw}. Using first 10 characters as YYYY-MM-DD.")
                start_date = start_date_raw[:10]
        except Exception as e:
            self.logger.warning(f"Error parsing start date: {e}. Using default date format.")
            start_date = start_date_raw[:10]
        
        # Format end date as YYYY-MM-DD
        end_date = datetime.now().strftime("%Y-%m-%d")
        parameter_cd = "00060"  # Discharge parameter code (cubic feet per second)
        
        # Conversion factor from cubic feet per second (cfs) to cubic meters per second (cms)
        CFS_TO_CMS = 0.0283168
        
        self.logger.info(f"Retrieving discharge data for station {station_id}")
        self.logger.info(f"Time period: {start_date} to {end_date}")
        self.logger.info(f"Converting from cfs to cms using factor: {CFS_TO_CMS}")
        
        # Construct the URL for tab-delimited data - ensure no spaces in the URL
        url = f"https://waterservices.usgs.gov/nwis/iv/?site={station_id}&format=rdb&parameterCd={parameter_cd}&startDT={start_date}&endDT={end_date}"
        
        # Log the formatted dates
        self.logger.info(f"Using formatted start date: {start_date}")
        self.logger.info(f"Using formatted end date: {end_date}")
        
        try:
            self.logger.info(f"Fetching data from: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # The RDB format has comment lines starting with #
            lines = response.text.split('\n')
            
            # Find the header line (it's after the comments and has field names)
            header_line = None
            for i, line in enumerate(lines):
                if not line.startswith('#') and '\t' in line:
                    header_line = i
                    break
            
            if header_line is None:
                self.logger.error("Could not find header line in the response")
                return
            
            # Skip the header line and the line after (which contains format info)
            data_start = header_line + 2
            
            # Create a data string with just the header and data rows
            data_str = '\n'.join([lines[header_line]] + lines[data_start:])
            
            # Parse the tab-delimited data
            df = pd.read_csv(io.StringIO(data_str), sep='\t', comment='#')
            
            # Find the discharge column (usually contains the parameter code)
            discharge_cols = [col for col in df.columns if parameter_cd in col]
            datetime_col = None
            
            # Find the datetime column (usually named 'datetime')
            datetime_candidates = ['datetime', 'date_time', 'dateTime']
            for col in df.columns:
                if col.lower() in [c.lower() for c in datetime_candidates]:
                    datetime_col = col
                    break
            
            if not discharge_cols:
                self.logger.error(f"Could not find column with parameter code {parameter_cd}")
                # Try to guess based on typical column names
                value_cols = [col for col in df.columns if 'value' in col.lower()]
                if value_cols:
                    discharge_cols = [value_cols[0]]
                    self.logger.info(f"Using column {discharge_cols[0]} as discharge values")
                else:
                    raise ValueError(f"Could not identify discharge column in USGS data")
            
            if not datetime_col:
                self.logger.error("Could not find datetime column")
                # Try to guess based on column data
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].str.contains('-').any():
                        datetime_col = col
                        self.logger.info(f"Using column {datetime_col} as datetime")
                        break
                if not datetime_col:
                    raise ValueError("Could not identify datetime column in USGS data")
            
            discharge_col = discharge_cols[0]
            self.logger.info(f"Using discharge column: {discharge_col}")
            self.logger.info(f"Using datetime column: {datetime_col}")
            
            # Keep only the necessary columns
            df_clean = df[[datetime_col, discharge_col]].copy()
            
            # Convert datetime column to datetime type
            df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col])
            
            # Convert discharge values to numeric, forcing errors to NaN
            df_clean[discharge_col] = pd.to_numeric(df_clean[discharge_col], errors='coerce')
            
            # Drop rows with NaN discharge values
            na_count = df_clean[discharge_col].isna().sum()
            if na_count > 0:
                self.logger.warning(f"Dropping {na_count} rows with non-numeric discharge values")
                df_clean = df_clean.dropna(subset=[discharge_col])
            
            # Create a new column with the discharge in cubic meters per second (cms)
            df_clean['discharge_cms'] = df_clean[discharge_col] * CFS_TO_CMS
            
            # Set datetime as index
            df_clean.set_index(datetime_col, inplace=True)
            
            # Call the resampling and saving function
            self._resample_and_save(df_clean['discharge_cms'])
            
            self.logger.info(f"Successfully processed USGS data for station {station_id}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from USGS API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing USGS data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _process_usgs_data(self):
        self.logger.info("Processing USGS streamflow data")
        usgs_data = pd.read_csv(self.streamflow_raw_path / self.streamflow_raw_name, 
                                comment='#', sep='\t', 
                                skiprows=[6],
                                parse_dates=['datetime'],
                                date_format='%Y-%m-%d %H:%M',
                                low_memory=False 
                                )

        usgs_data = usgs_data.loc[1:]
        usgs_data['discharge_cfs'] = pd.to_numeric(usgs_data[usgs_data.columns[4]], errors='coerce')
        usgs_data['discharge_cms'] = usgs_data['discharge_cfs'] * 0.028316847
        usgs_data['datetime'] = pd.to_datetime(usgs_data['datetime'], format='%Y-%m-%d %H:%M', errors='coerce')
        usgs_data = usgs_data.dropna(subset=['datetime'])
        usgs_data.set_index('datetime', inplace=True)

        self._resample_and_save(usgs_data['discharge_cms'])

    def _process_wsc_data(self):
        self.logger.info("Processing WSC streamflow data")
        wsc_data = pd.read_csv(self.streamflow_raw_path / self.streamflow_raw_name, 
                               comment='#', 
                               low_memory=False)

        wsc_data['ISO 8601 UTC'] = pd.to_datetime(wsc_data['ISO 8601 UTC'], format='ISO8601')
        wsc_data.set_index('ISO 8601 UTC', inplace=True)
        wsc_data.index = wsc_data.index.tz_convert('America/Edmonton').tz_localize(None)
        wsc_data['discharge_cms'] = pd.to_numeric(wsc_data['Value'], errors='coerce')

        self._resample_and_save(wsc_data['discharge_cms'])

    def _resample_and_save(self, data):
        resample_freq = self.get_resample_freq()
        resampled_data = data.resample(resample_freq).mean()
        resampled_data = resampled_data.interpolate(method='time', limit=24)
        #resampled_data = resampled_data.dropna()

        output_file = self.streamflow_processed_path / f'{self.domain_name}_streamflow_processed.csv'
        data_to_write = [('datetime', 'discharge_cms')] + list(resampled_data.items())
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in data_to_write:
                if isinstance(row[0], datetime):
                    formatted_datetime = row[0].strftime('%Y-%m-%d %H:%M:%S')
                    csv_writer.writerow([formatted_datetime, row[1]])
                else:
                    csv_writer.writerow(row)

        self.logger.info(f"Processed streamflow data saved to: {output_file}")
        self.logger.info(f"Total rows in processed data: {len(resampled_data)}")
        self.logger.info(f"Number of non-null values: {resampled_data.count()}")
        self.logger.info(f"Number of null values: {resampled_data.isnull().sum()}")


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
        from utils.config.config_utils import get_default_path # type: ignore

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
            #"--submit-job",
            "--print-geotiff=true",
            f"--cache={self.tool_cache}_{self.domain_name}",
            f"--account={self.config['TOOL_ACCOUNT']}"
        ] 
        
        if start_date and end_date:
            gistool_command.extend([
                f"--start-date={start_date}",
                f"--end-date={end_date}"
            ])

        return gistool_command  
    
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
        from utils.config.config_utils import get_default_path # type: ignore

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
        f"--submit-job",
        f"--cache={self.tool_cache}",
        f"--cluster={self.config['CLUSTER_JSON']}",
        ] 

        return datatool_command

    def execute_datatool_command(self, datatool_command):
        try:
            # Submit the array job
            result = subprocess.run(datatool_command, check=True, capture_output=True, text=True)
            self.logger.info("datatool job submitted successfully.")
            
            # Extract job ID from the output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    break
            
            if not job_id:
                raise RuntimeError("Could not extract job ID from submission output")
            
            # Wait for all array jobs to complete
            while True:
                check_cmd = ['squeue', '-j', job_id, '-h']
                status_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if not status_result.stdout.strip():
                    break
                time.sleep(30)
            
            self.logger.info("All datatool array jobs completed.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running datatool: {e}")
            raise
        self.logger.info("Meteorological data acquisition process completed")

