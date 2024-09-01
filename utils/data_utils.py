import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore
import numpy as np # type: ignore

class DataAcquisitionProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def prepare_maf_json(self) -> Path:
        """Prepare the JSON file for the Model Agnostic Framework."""
        maf_config = {
            "exec": {
                "met": str(self.root_path / self.config.get('DATATOOL_PATH', "installs/datatool/extract-dataset.sh")),
                "gis": str(self.root_path / self.config.get('GISTOOL_PATH', "installs/gistool/extract-gis.sh")),
                "remap": self.config.get('EASYMORE_PATH', "easymore cli")
            },
            "args": {
                "met": [{
                    "dataset": self.config.get('FORCING_DATASET'),
                    "dataset-dir": str(Path(self.config.get('DATATOOL_DATASET_ROOT')) / "rdrsv2.1/"),
                    "variable": self.config.get('FORCING_VARIABLES', [
                        "RDRS_v2.1_P_P0_SFC",
                        "RDRS_v2.1_P_HU_09944",
                        "RDRS_v2.1_P_TT_09944",
                        "RDRS_v2.1_P_UVC_09944",
                        "RDRS_v2.1_A_PR0_SFC",
                        "RDRS_v2.1_P_FB_SFC",
                        "RDRS_v2.1_P_FI_SFC"
                    ]),
                    "output-dir": str(self.project_dir / "forcing/1_raw_data"),
                    "start-date": f"{self.config.get('FORCING_START_YEAR')}-01-01T13:00:00",
                    "end-date": f"{self.config.get('FORCING_END_YEAR')}-12-31T12:00:00",
                    "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                    "prefix": f"domain_{self.domain_name}_",
                    "cache": self.config.get('DATATOOL_CACHE'),
                    "account": self.config.get('DATATOOL_ACCOUNT'),
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
                        "start-date": self.config.get('LANDCOVER_YEAR', "2020"),
                        "end-date": self.config.get('LANDCOVER_YEAR', "2020"),
                        "output-dir": str(self.project_dir / "parameters/landclass"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["frac", "majority", "coords"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('GISTOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_HRUID'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    },
                    {
                        "dataset": "soil_class",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "soil_classes"),
                        "variable": "soil_classes",
                        "output-dir": str(self.project_dir / "parameters/soilclass"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["majority"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('GISTOOL_ACCOUNT'),
                        "fid": self.config.get('RIVER_BASIN_SHP_RM_HRUID'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    },
                    {
                        "dataset": "merit-hydro",
                        "dataset-dir": str(Path(self.config.get('GISTOOL_DATASET_ROOT')) / "MERIT-Hydro"),
                        "variable": "elv,hnd",
                        "output-dir": str(self.project_dir / "parameters/dem"),
                        "shape-file": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                        "print-geotiff": "true",
                        "stat": ["min", "max", "mean", "median"],
                        "lib-path": self.config.get('GISTOOL_LIB_PATH'),
                        "cache": self.config.get('GISTOOL_CACHE'),
                        "prefix": f"domain_{self.domain_name}_",
                        "account": self.config.get('GISTOOL_ACCOUNT'),
                        "_flags": ["include-na", "submit-job", "parsable"]
                    }
                ],
                "remap": [{
                    "case-name": "remapped",
                    "cache": self.config.get('EASYMORE_CACHE'),
                    "shapefile": str(self.project_dir / "shapefiles/catchment" / self.config.get('CATCHMENT_SHP_NAME')),
                    "shapefile-id": self.config.get('RIVER_BASIN_SHP_RM_HRUID'),
                    "source-nc": str(self.project_dir / "forcing/1_raw_data/**/*.nc*"),
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
                    "remapped-var-id": self.config.get('RIVER_BASIN_SHP_RM_HRUID'),
                    "remapped-dim-id": self.config.get('RIVER_BASIN_SHP_RM_HRUID'),
                    "output-dir": str(self.project_dir / "forcing/3_basin_averaged_data/"),
                    "job-conf": str(self.root_path / self.config.get('EASYMORE_JOB_CONF', "installs/MAF/02_model_agnostic_component/easymore-job.slurm")),
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
        json_path = self.project_dir / "maf_config.json"
        with open(json_path, 'w') as f:
            json.dump(maf_config, f, indent=2)

        self.logger.info(f"MAF configuration JSON saved to: {json_path}")
        return json_path

    def run_data_acquisition(self):
        """Run the data acquisition process using MAF."""
        maf_json_path = self.prepare_maf_json()
        self.logger.info("Starting data acquisition process")

        # Run MAF components
        components = ['met', 'gis', 'remap']
        for component in components:
            self.logger.info(f"Running MAF component: {component}")
            command = f"maf run {component} --config {maf_json_path}"
            try:
                subprocess.run(command, shell=True, check=True)
                self.logger.info(f"Successfully completed MAF component: {component}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error running MAF component {component}: {str(e)}")
                raise

        self.logger.info("Data acquisition process completed")

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
        path_soil_type = self.project_dir / f'parameters/soilclass/domain_{self.domain_name}_stats_soil_classes.csv'
        path_landcover_type = self.project_dir / f'parameters/landclass/domain_{self.domain_name}_stats_NA_NALCMS_landcover_2020_30m.csv'
        path_elevation_mean = self.project_dir / f'parameters/dem/domain_{self.domain_name}_stats_elv.csv'

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
        soil_type.to_csv(self.project_dir / 'parameters/soilclass/modified_domain_stats_soil_classes.csv', index=False)
        landcover_type.to_csv(self.project_dir / 'parameters/landclass/modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv', index=False)
        elevation_mean.to_csv(self.project_dir / 'parameters/dem/modified_domain_stats_elv.csv', index=False)

        self.logger.info("Cleanup and checks completed")

class DataPreProcessor:
    def subset_hydrofabric(self, data_sources):
        # Acquire data from various sources
        pass