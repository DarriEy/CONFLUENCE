import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore


class DataAcquisitionProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"
        

    def prepare_maf_json(self) -> Path:
        """Prepare the JSON file for the Model Agnostic Framework."""

        met_path = str(self.root_path / "installs/datatool/" / "extract-dataset.sh")
        gis_path = str(self.root_path / "installs/gistool/" / "extract-gis.sh")
        easymore_client = str(self.config.get('EASYMORE_CLIENT'))

        maf_config = {
            "exec": {
                "met": met_path,
                "gis": gis_path,
                "remap": easymore_client
            },
            "args": {
                "met": [{
                    "dataset": self.config.get('FORCING_DATASET'),
                    "dataset-dir": str(Path(self.config.get('DATATOOL_DATASET_ROOT')) / "rdrsv2.1/"),
                    "variable": self.config.get('FORCING_VARIABLES'),
                    "output-dir": str(self.project_dir / "forcing/raw_data"),
                    "start-date": f"{self.config.get('FORCING_START_YEAR')}-01-01T13:00:00",
                    "end-date": f"{self.config.get('FORCING_END_YEAR')}-12-31T12:00:00",
                    "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                    "prefix": f"domain_{self.domain_name}_",
                    "cache": self.config.get('DATATOOL_CACHE'),
                    "account": self.config.get('TOOL_ACCOUNT'),
                    "_flags": [
                        "submit-job",
                        "parsable"
                    ]
                }],
                "gis": [
                    {
                        "dataset": "landsat",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "Landsat"),
                        "variable": "land-cover",
                        "start-date": self.config.get('LANDCOVER_YEAR'),
                        "end-date": self.config.get('LANDCOVER_YEAR'),
                        "output-dir": str(self.project_dir / "attributes/land_class"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["frac", "majority", "coords"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('CATCHMENT_SHP_HRUID'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "soil_classes"),
                        "variable": "soil_classes",
                        "output-dir": str(self.project_dir / "attributes/soil_class"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["majority"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('CATCHMENT_SHP_HRUID'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MERIT-Hydro"),
                        "variable": "elv,hnd",
                        "output-dir": str(self.project_dir / "attributes/elevation"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["min", "max", "mean", "median"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('TOOL_ACCOUNT'),
                        "fid": self.config.get('CATCHMENT_SHP_HRUID'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    }
                ],
                "remap": [{
                    "case-name": "remapped",
                    "cache": self.config.get('EASYMORE_CACHE'),
                    "shapefile": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                    "shapefile-id": self.config.get('CATCHMENT_SHP_HRUID'),
                    "source-nc": str(self.project_dir / "forcing/raw_data/**/*.nc*"),
                    "variable-lon": "lon",
                    "variable-lat": "lat",
                    "variable": self.config.get('FORCING_VARIABLES', [
                        "RDRS_v2.1_P_P0_SFC",
                        "RDRS_v2.1_P_HU_09944",
                        "RDRS_v2.1_P_TT_09944",
                        "RDRS_v2.1_P_UVC_09944",
                        "RDRS_v2.1_A_PR0_SFC",
                        "RDRS_v2.1_P_FB_SFC",
                        "RDRS_v2.1_P_FI_SFC"
                    ]),
                    "remapped-var-id": "hruId",
                    "remapped-dim-id": "hru",
                    "output-dir": str(self.project_dir / "forcing/basin_averaged_data/") + '/',
                    "job-conf": str(Path(self.config.get('CONFLUENCE_DATA_DIR')) / self.config.get('EASYMORE_JOB_CONF')),
                    "_flags": ["submit-job"]
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
        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', self.config.get('CATCHMENT_SHP_NAME'))
        dem_path = self._get_file_path('DEM_PATH', 'attributes/elevation/dem', self.config.get('DEM_NAME'))
        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem', self.config.get('INTERSECT_DEM_NAME'))

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
        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', self.config.get('CATCHMENT_SHP_NAME'))
        soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass/', self.config.get('SOIL_CLASS_NAME'))
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids', self.config.get('INTERSECT_SOIL_NAME'))

        intersect_path.parent.mkdir(parents=True, exist_ok=True)

        catchment_gdf = gpd.read_file(catchment_path)
        nodata_value = self.get_nodata_value(soil_path)

        with rasterio.open(soil_path) as src:
            affine = src.transform
            soil_data = src.read(1)

        stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
        result_df = pd.DataFrame(stats).fillna(0)
        
        print(result_df.head(11))  # Print first 11 rows for debugging

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
        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', self.config.get('CATCHMENT_SHP_NAME'))
        land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass/', self.config.get('LAND_CLASS_NAME'))
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass', self.config.get('INTERSECT_LAND_NAME'))

        intersect_path.parent.mkdir(parents=True, exist_ok=True)

        catchment_gdf = gpd.read_file(catchment_path)
        nodata_value = self.get_nodata_value(land_path)

        with rasterio.open(land_path) as src:
            affine = src.transform
            land_data = src.read(1)

        stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], categorical=True, nodata=nodata_value)
        result_df = pd.DataFrame(stats).fillna(0)
        
        print(result_df.head(11))  # Print first 11 rows for debugging
        print(result_df.dtypes)  # Print data types of columns

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