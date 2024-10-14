import sys
from typing import Dict, Any
from pathlib import Path
import subprocess

class gistoolRunner:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        #Import required CONFLUENCE utils
        sys.path.append(str(self.code_dir))
        from utils.configHandling_utils.config_utils import get_default_path # type: ignore

        #Get the path to the directory containing the gistool script
        self.gistool_path = get_default_path(self.config, self.code_dir, self.config['GISTOOL_PATH'], 'installs/gistool', self.logger)
    
    def create_gistool_command(self, dataset, output_dir, lat_lims, lon_lims, variables, start_date=None, end_date=None):
        gistool_command = [
            f"{self.gistool_path}/extract-gis.sh",
            f"--dataset={dataset}",
            f"--dataset-dir={self.config['GISTOOL_DATASET_ROOT']}{dataset}",
            f"--output-dir={output_dir}",
            f"--lat-lims={lat_lims}",
            f"--lon-lims={lon_lims}",
            f"--variable={variables}",
            f"--prefix=domain_{self.domain_name}",
            "--submit-job",
            "--print-geotiff=true",
            f"--cache={self.config['TOOL_CACHE']}",
            f"--account={self.config['TOOL_ACCOUNT']}"
        ] 
        
        if start_date and end_date:
            gistool_command.extend([
                f"--start-date={start_date}",
                f"--end-date={end_date}"
            ])

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

        #Import required CONFLUENCE utils
        sys.path.append(str(self.code_dir))
        from utils.configHandling_utils.config_utils import get_default_path # type: ignore

        #Get the path to the directory containing the gistool script
        self.datatool_path = get_default_path(self.config, self.code_dir, self.config['DATATOOL_PATH'], '/installs/gistool', self.logger)
    
    def create_datatool_command(self, dataset, output_dir, start_date, end_date, lat_lims, lon_lims, variables):
        datatool_command = [
        f"{self.datatool_path}/extract-dataset.sh",
        f"--dataset={dataset}",
        f"--dataset-dir={self.config['DATATOOL_DATASET_ROOT']}{dataset}",
        f"--output-dir={output_dir}",
        f"--start-date={start_date}",
        f"--end-date={end_date}",
        f"--lat-lims={lat_lims}",
        f"--lon-lims={lon_lims}",
        f"--variable={variables}",
        f"--prefix=domain_{self.domain_name}",
        "--submit-job",
        f"--cache={self.config['TOOL_CACHE']}",
        f"--account={self.config['TOOL_ACCOUNT']}"
        ] 
        return datatool_command
    
    def execute_datatool_command(self, datatool_command):
        
        #Run the gistool command
        try:
            subprocess.run(datatool_command, check=True)
            self.logger.info("datatool completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running datatool: {e}")
            raise
        self.logger.info("Meteorological data acquisition process completed")