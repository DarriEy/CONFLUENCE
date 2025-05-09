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
