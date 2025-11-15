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
import time

from utils.data.utilities.variable_utils import VariableHandler # type: ignore


class gistoolRunner:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('SYMFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self.config.get('TOOL_CACHE')
        
        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'

        #Get the path to the directory containing the gistool script
        self.gistool_path = self.config['GISTOOL_PATH']
        if self.gistool_path == 'default':
            self.gistool_path = Path(self.config['SYMFLUENCE_DATA_DIR']) / 'installs/gistool'
        else: 
            self.gistool_path = self.config['GISTOOL_PATH']
    
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
            #f"--lib-path={self.config['GISTOOL_LIB_PATH']}"
            #"--submit-job",
            "--print-geotiff=true",
            f"--cache={self.tool_cache}_{self.domain_name}",
            f"--cluster={self.config['CLUSTER_JSON']}"
        ] 
        
        self.logger.info(f'gistool command: {gistool_command}')
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
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('SYMFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.tool_cache = self.config.get('TOOL_CACHE')
        if self.tool_cache == 'default':
            self.tool_cache = '$HOME/cache_dir/'
        
        #Get the path to the directory containing the datatool script
        self.datatool_path = self.config['DATATOOL_PATH']
        if self.datatool_path == 'default':
            self.datatool_path = Path(self.config['SYMFLUENCE_DATA_DIR']) / 'installs/datatool'
        else: 
            self.datatool_path = self.config['DATATOOL_PATH']
    
    def create_datatool_command(self, dataset, output_dir, start_date, end_date, lat_lims, lon_lims, variables):
        dataset_dir = dataset
        if dataset == "ERA5":
            dataset_dir = 'era5'
        elif dataset == "RDRS":
            dataset_dir = 'rdrsv2.1'
        elif dataset == "CASR":
            dataset_dir = 'casrv3.1'
            dataset = 'casr'

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
        """
        Execute the datatool command and wait for the job to complete in the queue.
        
        This simplified implementation focuses on tracking the specific job ID
        until it's no longer present in the Slurm queue.
        """
        try:
            # Submit the array job
            self.logger.info(f"Submitting datatool job")
            result = subprocess.run(datatool_command, check=True, capture_output=True, text=True)
            self.logger.info("datatool job submitted successfully.")
            
            # Extract job ID from the output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    try:
                        job_id = line.split()[-1].strip()
                        break
                    except (IndexError, ValueError):
                        pass
            
            if not job_id:
                self.logger.error("Could not extract job ID from submission output")
                self.logger.debug(f"Submission output: {result.stdout}")
                raise RuntimeError("Could not extract job ID from submission output")
            
            self.logger.info(f"Monitoring job ID: {job_id}")
            
            # Wait for job to no longer be in the queue
            wait_time = 30  # seconds between checks
            max_checks = 1000  # Maximum number of checks
            check_count = 0
            
            while check_count < max_checks:
                # Check if job is still in the queue
                check_cmd = ['squeue', '-j', job_id, '-h']
                status_result = subprocess.run(check_cmd, capture_output=True, text=True)
                
                # If no output, the job is no longer in the queue
                if not status_result.stdout.strip():
                    self.logger.info(f"Job {job_id} is no longer in the queue, assuming completed.")
                    # Wait an additional minute to allow for any file system operations to complete
                    break
                
                self.logger.info(f"Job {job_id} still in queue. Waiting {wait_time} seconds. Check {check_count+1}/{max_checks}")
                time.sleep(wait_time)
                check_count += 1
            
            if check_count >= max_checks:
                self.logger.warning(f"Reached maximum checks ({max_checks}) for job {job_id}, but continuing anyway")
            
            self.logger.info("datatool job monitoring completed.")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running datatool: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"Command error output: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during datatool execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
        self.logger.info("Meteorological data acquisition process completed")