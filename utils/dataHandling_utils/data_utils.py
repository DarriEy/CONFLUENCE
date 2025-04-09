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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.dataHandling_utils.variable_utils import VariableHandler # type: ignore

class ProjectInitialisation:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"


    def setup_project(self):
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        shapefile_dir = self.project_dir / "shapefiles"
        shapefile_dir.mkdir(parents=True, exist_ok=True)
        pourPoint_dir = shapefile_dir / "pour_point"
        pourPoint_dir.mkdir(parents=True, exist_ok=True)
        catchment_dir = shapefile_dir / "catchment"
        catchment_dir.mkdir(parents=True, exist_ok=True)
        riverNetwork_dir = shapefile_dir / "river_network"
        riverNetwork_dir.mkdir(parents=True, exist_ok=True)
        riverBasins_dir = shapefile_dir / "river_basins"
        riverBasins_dir.mkdir(parents=True, exist_ok=True)
        Q_observations_dir = self.project_dir / 'observations' / 'streamflow' / 'raw_data'
        Q_observations_dir.mkdir(parents=True, exist_ok=True)
        documentation_dir = self.project_dir / "documentation"
        documentation_dir.mkdir(parents=True, exist_ok=True)
        attributes_dir = self.project_dir / 'attributes'
        attributes_dir.mkdir(parents=True, exist_ok=True)

        # If in point mode, update bounding box coordinates
        if self.config.get('SPATIAL_MODE') == 'Point':
            self._update_bounding_box_for_point_mode()

        return self.project_dir

    def _update_bounding_box_for_point_mode(self):
        """
        Update the bounding box coordinates in the config to create a 0.02 degree buffer
        around the pour point for point-scale simulations.
        """
        try:
            # Get pour point coordinates
            pour_point_coords = self.config.get('POUR_POINT_COORDS', '')
            if not pour_point_coords or pour_point_coords.lower() == 'default':
                self.logger.warning("Pour point coordinates not specified, cannot update bounding box for point mode")
                return
            
            # Parse coordinates
            lat, lon = map(float, pour_point_coords.split('/'))
            
            # Define buffer distance (0.01 degree in each direction for a total of 0.02 degrees)
            buffer_dist = 0.01
            
            # Create a square buffer around the point
            min_lon = round(lon - buffer_dist, 4)
            max_lon = round(lon + buffer_dist, 4)
            min_lat = round(lat - buffer_dist, 4)
            max_lat = round(lat + buffer_dist, 4)
            
            # Format the new bounding box string
            new_bbox = f"{max_lat}/{min_lon}/{min_lat}/{max_lon}"
            
            self.logger.info(f"Updating bounding box for point-scale simulation to: {new_bbox}")
            
            # Update the configuration in memory
            self.config['BOUNDING_BOX_COORDS'] = new_bbox
            
            # Update the active config file
            self._update_active_config_file('BOUNDING_BOX_COORDS', new_bbox)
            
        except Exception as e:
            self.logger.error(f"Error updating bounding box for point mode: {str(e)}")

    def _update_active_config_file(self, key, value):
        """
        Update a specific key in the active configuration file.
        
        Args:
            key (str): The configuration key to update
            value (str): The new value to set
        """
        try:
            # Get the path to the active config file
            if 'CONFLUENCE_CODE_DIR' in self.config:
                config_path = Path(self.config['CONFLUENCE_CODE_DIR']) / '0_config_files' / 'config_active.yaml'
            else:
                self.logger.warning("CONFLUENCE_CODE_DIR not specified, cannot update config file")
                return
            
            if not config_path.exists():
                self.logger.warning(f"Active config file not found at {config_path}")
                return
            
            # Read the current config file
            with open(config_path, 'r') as f:
                lines = f.readlines()
            
            # Update the specific key
            updated = False
            for i, line in enumerate(lines):
                # Look for the key at the beginning of the line
                if line.strip().startswith(f"{key}:"):
                    # Replace the line with the updated value
                    lines[i] = f"{key}: {value}  # Updated for point-scale simulation\n"
                    updated = True
                    break
            
            if not updated:
                self.logger.warning(f"Could not find {key} in config file to update")
                return
            
            # Write the updated config back to the file
            with open(config_path, 'w') as f:
                f.writelines(lines)
            
            self.logger.info(f"Updated {key} in active config file: {config_path}")
        
        except Exception as e:
            self.logger.error(f"Error updating config file: {str(e)}")

    def create_pourPoint(self):
        if self.config.get('POUR_POINT_COORDS', 'default').lower() == 'default':
            return None
        
        try:
            lat, lon = map(float, self.config['POUR_POINT_COORDS'].split('/'))
            point = Point(lon, lat)
            gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
            
            if self.config.get('POUR_POINT_SHP_PATH') == 'default':
                output_path = self.project_dir / "shapefiles" / "pour_point"
            else:
                output_path = Path(self.config['POUR_POINT_SHP_PATH'])
            
            pour_point_shp_name = self.config.get('POUR_POINT_SHP_NAME')
            if pour_point_shp_name == 'default':
                pour_point_shp_name = f"{self.domain_name}_pourPoint.shp"
            
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / pour_point_shp_name
            
            gdf.to_file(output_file)
            return output_file
        except ValueError:
            self.logger.error("Invalid pour point coordinates format. Expected 'lat,lon'.")
        except Exception as e:
            self.logger.error(f"Error creating pour point shapefile: {str(e)}")
        
        return None


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
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment', subbasins_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass/', soil_name)
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids', self.config.get('INTERSECT_SOIL_NAME'))
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
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass', self.config.get('INTERSECT_LAND_NAME'))
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
                if self.config.get('DOWNLOAD_USGS_DATA') == True:
                    self._download_and_process_wsc_data()
                else:
                    self._process_wsc_data()
            elif self.data_provider == 'VI':
                self._process_vi_data()
            else:
                self.logger.error(f"Unsupported streamflow data provider: {self.data_provider}")
                raise ValueError(f"Unsupported streamflow data provider: {self.data_provider}")
        except Exception as e:
            self.logger.error(f'Issue in streamflow data preprocessing: {e}')

    def _download_and_process_wsc_data(self):
        """
        Process Water Survey of Canada (WSC) streamflow data by fetching it directly from current WSC endpoints.
        
        This function fetches discharge data for the specified WSC station,
        processes it, and resamples it to the configured time step.
        """
        import requests
        import io
        import pandas as pd
        from datetime import datetime, timedelta
        
        self.logger.info("Processing WSC streamflow data directly from API")
        
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
        
        # Log the station and date range
        self.logger.info(f"Retrieving discharge data for WSC station {station_id}")
        self.logger.info(f"Time period: {start_date} to {end_date}")
        
        # Current URL patterns for WSC data
        # Format without periods, e.g., 05BB001 -> 05bb001
        station_id_lower = station_id.lower()
        
        # Define provinces with their codes
        provinces = {
            "AB": "alberta",
            "BC": "british-columbia",
            "MB": "manitoba",
            "NB": "new-brunswick",
            "NL": "newfoundland-and-labrador",
            "NS": "nova-scotia",
            "NT": "northwest-territories",
            "NU": "nunavut",
            "ON": "ontario",
            "PE": "prince-edward-island",
            "QC": "quebec",
            "SK": "saskatchewan",
            "YT": "yukon"
        }
        
        # Determine the province from the station ID
        # WSC station IDs are structured where first two digits indicate the province
        province_code = station_id[:2]
        province_map = {
            "01": "NL", "02": "NS", "03": "NB", "04": "QC", "05": "QC",
            "06": "ON", "07": "ON", "08": "MB", "09": "SK", "10": "SK",
            "11": "AB", "12": "BC", "13": "YT", "14": "NT", "15": "NU"
        }
        
        province = None
        if province_code in province_map:
            province = provinces.get(province_map[province_code])
        
        if province is None:
            self.logger.warning(f"Could not determine province from station ID {station_id}. Will try all provinces.")
        else:
            self.logger.info(f"Determined province as {province} based on station ID {station_id}")
        
        # URLs to try - organized by priority
        urls_to_try = []
        
        # 1. Current Water Office real-time data - try specific province if available
        if province:
            urls_to_try.append(f"https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline?stations={station_id}&parameters=47")
            urls_to_try.append(f"https://wateroffice.ec.gc.ca/report/real_time_e.html?stn={station_id}")
        
        # 2. Historical data - try both CSV and HTML
        urls_to_try.append(f"https://wateroffice.ec.gc.ca/report/historical_e.html?stn={station_id}")
        urls_to_try.append(f"https://wateroffice.ec.gc.ca/services/historical/inline?stations={station_id}&parameters=47&start_date={start_date}&end_date={end_date}")
        
        # 3. Add HYDAT links (National Water Data Archive)
        urls_to_try.append(f"https://wateroffice.ec.gc.ca/report/data_e.html?type=h2oArc&stn={station_id}")
        
        # 4. Try directly with the WSC Data Mart endpoints
        urls_to_try.append(f"https://dd.weather.gc.ca/hydrometric/csv/daily/{province}/{station_id}_daily_mean.csv" if province else None)
        urls_to_try.append(f"https://dd.weather.gc.ca/hydrometric/csv/daily/all/{station_id}_daily_mean.csv")
        
        # Filter out None values
        urls_to_try = [url for url in urls_to_try if url]
        
        # Try each URL until we get a successful response
        df = None
        response = None
        successful_url = None
        html_content = False
        
        for url in urls_to_try:
            self.logger.info(f"Trying to fetch data from: {url}")
            try:
                response = requests.get(url, timeout=30)
                self.logger.info(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    # Check if response is HTML
                    if response.text.strip().lower().startswith("<!doctype html>") or "<html" in response.text.lower():
                        self.logger.warning(f"Response from {url} is HTML, not CSV data")
                        html_content = True
                        continue
                    
                    # Check if we got valid data
                    if len(response.text) > 100:  # Arbitrary minimal content length
                        self.logger.info(f"Successfully retrieved data from {url}")
                        successful_url = url
                        break
                    else:
                        self.logger.warning(f"Response from {url} appears too short: {len(response.text)} bytes")
                else:
                    self.logger.warning(f"Failed to retrieve data from {url}: Status code {response.status_code}")
            except Exception as e:
                self.logger.warning(f"Error fetching data from {url}: {e}")
        
        # If we couldn't get data from direct URLs, try using a manual download approach
        if not successful_url and not html_content:
            self.logger.warning("Trying alternative approach - using the WSC data download form")
            
            # For now, we'll direct the user to manually download the data
            self.logger.error(
                "Could not automatically retrieve WSC data. Please manually download data from "
                f"https://wateroffice.ec.gc.ca/report/historical_e.html?stn={station_id} and "
                f"place it in {self.streamflow_raw_path}/{self.streamflow_raw_name}"
            )
            raise ValueError(
                f"Could not retrieve WSC data for station {station_id}. "
                "Please download data manually and specify the path in the configuration."
            )
        
        # Handle HTML response (provide instructions)
        if html_content and not successful_url:
            self.logger.error(
                "Received HTML instead of data. WSC now requires interactive steps to download data. "
                f"Please manually download data from https://wateroffice.ec.gc.ca/search/historical_e.html "
                f"for station {station_id} and place it in {self.streamflow_raw_path}/{self.streamflow_raw_name}"
            )
            raise ValueError(
                f"WSC data for station {station_id} requires manual download. "
                "Please follow the instructions in the log."
            )
        
        # If we have a successful URL and response, process the data
        if successful_url and response:
            try:
                data_str = response.text
                
                # Save the raw data for reference
                raw_data_path = self.streamflow_raw_path / f"{station_id}_raw_data.csv"
                raw_data_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(raw_data_path, 'w') as f:
                    f.write(data_str)
                
                self.logger.info(f"Saved raw data to {raw_data_path}")
                
                # Try to parse the CSV data
                try:
                    # First try with default pandas CSV parsing
                    df = pd.read_csv(io.StringIO(data_str))
                    
                    # Check if we actually got data or just headers
                    if len(df) <= 1:
                        raise ValueError("No data rows found")
                    
                except Exception as first_error:
                    self.logger.warning(f"Initial parsing failed: {first_error}")
                    
                    # Try alternative parsing approaches
                    try:
                        # Try to detect the format by looking at the first few lines
                        lines = data_str.splitlines()
                        self.logger.debug(f"First few lines of data: {lines[:3]}")
                        
                        # Check if there are headers or comments
                        if lines and any(line.startswith('#') for line in lines[:10]):
                            # Skip comment lines
                            skip_rows = 0
                            for line in lines:
                                if line.startswith('#'):
                                    skip_rows += 1
                                else:
                                    break
                            
                            self.logger.info(f"Skipping {skip_rows} comment rows")
                            df = pd.read_csv(io.StringIO(data_str), skiprows=skip_rows)
                        else:
                            # Try with different delimiters
                            for sep in [',', ';', '\t', '|']:
                                try:
                                    df = pd.read_csv(io.StringIO(data_str), sep=sep)
                                    if len(df.columns) > 1:  # If we got multiple columns, it worked
                                        self.logger.info(f"Successfully parsed with delimiter: '{sep}'")
                                        break
                                except:
                                    continue
                    
                    except Exception as second_error:
                        self.logger.error(f"All parsing attempts failed: {second_error}")
                        raise ValueError(f"Could not parse data from {successful_url}")
                
                # Display column info for debugging
                self.logger.info(f"Columns in the data: {df.columns.tolist()}")
                
                # Try to identify the date and discharge columns
                date_col = None
                discharge_col = None
                
                # Common patterns for date columns
                date_patterns = ['date', 'time', 'datetime', 'day']
                # Common patterns for discharge columns
                discharge_patterns = ['flow', 'discharge', 'value', 'mean', 'q', 'debit']
                
                # Find date column
                for col in df.columns:
                    if any(pattern in str(col).lower() for pattern in date_patterns):
                        date_col = col
                        break
                
                # If no date column found, use the first column if it looks like a date
                if date_col is None and len(df.columns) > 0:
                    # Try to convert the first column to datetime
                    try:
                        pd.to_datetime(df.iloc[:, 0])
                        date_col = df.columns[0]
                        self.logger.info(f"Using first column as date column: {date_col}")
                    except:
                        pass
                
                # Find discharge column
                for col in df.columns:
                    if any(pattern in str(col).lower() for pattern in discharge_patterns):
                        discharge_col = col
                        break
                
                # If no discharge column found, try to identify a numeric column that's not the date
                if discharge_col is None:
                    for col in df.columns:
                        if col != date_col:
                            try:
                                # Try to convert a sample to float
                                sample = df[col].iloc[:10].astype(float)
                                if not sample.isna().all():
                                    discharge_col = col
                                    self.logger.info(f"Using column {col} as discharge column based on numeric content")
                                    break
                            except:
                                continue
                
                if date_col is None or discharge_col is None:
                    self.logger.error(f"Could not identify date and discharge columns in data")
                    self.logger.debug(f"Available columns: {df.columns.tolist()}")
                    raise ValueError("Could not identify date and discharge columns in WSC data")
                
                self.logger.info(f"Using date column: {date_col}")
                self.logger.info(f"Using discharge column: {discharge_col}")
                
                # Convert date column to datetime if it's not already
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Drop rows with invalid dates
                invalid_dates = df[date_col].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(f"Dropping {invalid_dates} rows with invalid dates")
                    df = df.dropna(subset=[date_col])
                
                # Convert discharge values to numeric
                df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
                
                # Drop rows with NaN discharge values
                na_count = df[discharge_col].isna().sum()
                if na_count > 0:
                    self.logger.warning(f"Dropping {na_count} rows with non-numeric discharge values")
                    df = df.dropna(subset=[discharge_col])
                
                # Set the date column as index
                df.set_index(date_col, inplace=True)
                
                # Create the discharge_cms column (WSC data is already in m³/s)
                df['discharge_cms'] = df[discharge_col]
                
                # Filter the data to our date range
                try:
                    start_datetime = pd.to_datetime(start_date)
                    # Add one day to end_date to include that day in the result
                    end_datetime = pd.to_datetime(end_date) + timedelta(days=1)
                    df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
                    self.logger.info(f"Filtered data to date range: {start_date} to {end_date}")
                    self.logger.info(f"Records after date filtering: {len(df)}")
                except Exception as e:
                    self.logger.warning(f"Error filtering by date range: {e}")
                
                # Check if we have data
                if df.empty:
                    self.logger.error("No data available after filtering")
                    raise ValueError("No data available for the specified date range")
                
                self.logger.info(f"Total records after processing: {len(df)}")
                
                # Call the resampling and saving function
                self._resample_and_save(df['discharge_cms'])
                
                self.logger.info(f"Successfully processed WSC data for station {station_id}")
                
            except Exception as e:
                self.logger.error(f"Error processing WSC data: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
        else:
            self.logger.error("Could not retrieve WSC data from any of the tried URLs")
            raise ValueError("Could not retrieve WSC data. Please check the station ID or try again later.")

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

class BenchmarkPreprocessor:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"

    def preprocess_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Preprocess data for hydrobm benchmarking.
        """
        self.logger.info("Starting benchmark data preprocessing")

        # Load and process data
        streamflow_data = self._load_streamflow_data()
        forcing_data = self._load_forcing_data()
        merged_data = self._merge_data(streamflow_data, forcing_data)
        
        # Aggregate to daily timestep using a single resample pass
        daily_data = self._process_to_daily(merged_data)
        
        # Filter data for the experiment run period
        filtered_data = daily_data.loc[start_date:end_date]
        
        # Validate and save data
        self._validate_data(filtered_data)
        output_path = self.project_dir / 'evaluation'
        output_path.mkdir(exist_ok=True)
        filtered_data.to_csv(output_path / "benchmark_input_data.csv")
        
        return filtered_data

    def _process_to_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to daily values with correct units using a single resample."""
        daily_data = data.resample('D').agg({
            'temperature': 'mean',
            'streamflow': 'mean',
            'precipitation': 'sum'
        })
        return daily_data

    def _validate_data(self, data: pd.DataFrame):
        """Validate data ranges and consistency."""
        missing = data.isnull().sum()
        if missing.any():
            self.logger.warning(f"Missing values detected:\n{missing}")
        
        if (data['temperature'] < 200).any() or (data['temperature'] > 330).any():
            self.logger.warning("Temperature values outside physical range (200-330 K)")
        
        if (data['streamflow'] < 0).any():
            self.logger.warning("Negative streamflow values detected")
        
        if (data['precipitation'] < 0).any():
            self.logger.warning("Negative precipitation values detected")
            
        if (data['precipitation'] > 1000).any():
            self.logger.warning("Extremely high precipitation values (>1000 mm/day) detected")

        self.logger.info(f"Data statistics:\n{data.describe()}")

    def _load_streamflow_data(self) -> pd.DataFrame:
        """Load and basic process streamflow data."""
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
        data = pd.read_csv(streamflow_path, parse_dates=['datetime'], index_col='datetime')
        # Ensure the index is sorted for a faster merge later on
        data.sort_index(inplace=True)
        return data.rename(columns={'discharge_cms': 'streamflow'})

    def _load_forcing_data(self) -> pd.DataFrame:
        """Load and process forcing data, returning hourly dataframe."""
        forcing_path = self.project_dir / "forcing" / "basin_averaged_data"
        # Use open_mfdataset to load all netCDF files at once efficiently
        combined_ds = xr.open_mfdataset(list(forcing_path.glob("*.nc")), combine='by_coords')
        
        # Average across HRUs
        averaged_ds = combined_ds.mean(dim='hru')
        
        # Convert precipitation to mm/day (assuming input is in m/s)
        precip_data = averaged_ds['pptrate'] * 3600
        
        # Create DataFrame directly using to_series for better integration
        forcing_df = pd.DataFrame({
            'temperature': averaged_ds['airtemp'].to_series(),
            'precipitation': precip_data.to_series()
        })
        forcing_df.sort_index(inplace=True)
        return forcing_df

    def _merge_data(self, streamflow_data: pd.DataFrame, forcing_data: pd.DataFrame) -> pd.DataFrame:
        """Merge streamflow and forcing data on timestamps using concatenation for efficiency."""
        merged_data = pd.concat([streamflow_data, forcing_data], axis=1, join='inner')
        
        # Verify data completeness
        expected_records = len(pd.date_range(merged_data.index.min(), 
                                              merged_data.index.max(), 
                                              freq='h'))
        if len(merged_data) != expected_records:
            self.logger.warning(f"Data gaps detected. Expected {expected_records} records, got {len(merged_data)}")
        
        return merged_data
