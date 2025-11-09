from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd # type: ignore
import numpy as np
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import subprocess
import sys
import os
import cdo # type: ignore


import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from utils.models.hypeFlow import write_hype_forcing, write_hype_geo_files, write_hype_par_file, write_hype_info_filedir_files # type: ignore


class HYPEPreProcessor:
    """
    HYPE (HYdrological Predictions for the Environment) preprocessor for SYMFLUENCE.
    Handles preparation of HYPE model inputs using SYMFLUENCE's data structure.
    
    Attributes:
        config (Dict[str, Any]): SYMFLUENCE configuration settings
        logger (logging.Logger): Logger for the preprocessing workflow
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE preprocessor with SYMFLUENCE config."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Define HYPE-specific paths
        self.hype_setup_dir = self.project_dir / "settings" / "HYPE"
        self.hype_setup_dir.mkdir(parents=True, exist_ok=True)
        self.gistool_output = f"{str(self.project_dir / 'attributes' / 'gistool-outputs')}/"
        self.easymore_output = f"{str(self.project_dir / 'forcing' / 'easymore-outputs')}/"
        self.hype_setup_dir = f"{str(self.project_dir / 'settings' / 'HYPE')}/"
        self.hype_results_dir = self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "HYPE"
        self.hype_results_dir.mkdir(parents=True, exist_ok=True)
        self.hype_results_dir = f"{str(self.hype_results_dir)}/"
        self.cache_path = self.project_dir / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        # Initialize time parameters
        self.timeshift = self.config.get('HYPE_TIMESHIFT')  
        self.spinup_days = self.config.get('HYPE_SPINUP_DAYS')  
        self.frac_threshold = self.config.get('HYPE_FRAC_THRESHOLD') # fraction to exclude landcover with coverage less than this value
        
        # inputs
        self.output_path = self.hype_setup_dir

        self.forcing_units= {
            # required variable # name of var in input data, units in input data, required units for HYPE
            'temperature': {'in_varname':'RDRS_v2.1_P_TT_09944', 'in_units':'celsius', 'out_units': 'celsius'},
            'precipitation': {'in_varname':'RDRS_v2.1_A_PR0_SFC','in_units':'m/hr', 'out_units': 'mm/day'},
        }
        
        #mapping geofabric fields to model names
        self.geofabric_mapping ={
            'basinID': {'in_varname':self.config['RIVER_BASIN_SHP_RM_GRUID']},
            'nextDownID': {'in_varname': self.config['RIVER_NETWORK_SHP_DOWNSEGID']},
            'area': {'in_varname':self.config['RIVER_BASIN_SHP_AREA'], 'in_units':'m^2', 'out_units':'m^2'},
            'rivlen': {'in_varname':self.config['RIVER_NETWORK_SHP_LENGTH'], 'in_units':'m', 'out_units':'m'}
        }

        # domain subbasins and rivers
        self.subbasins_shapefile = str(self.project_dir / 'shapefiles' / 'river_basins' / f'{self.domain_name}_riverBasins_delineate.shp')
        self.rivers_shapefile = str(self.project_dir / 'shapefiles' / 'river_network' / f'{self.domain_name}_riverNetwork_delineate.shp')

    def run_preprocessing(self):
        """Execute complete HYPE preprocessing workflow."""
        self.logger.info("Starting HYPE preprocessing")
        
        try:

            # Write forcing files
            write_hype_forcing(self.easymore_output, self.timeshift, self.forcing_units, self.geofabric_mapping, self.output_path, f"{self.cache_path}/")
            
            # Write geographic data files
            write_hype_geo_files(self.gistool_output, self.subbasins_shapefile, self.rivers_shapefile, self.frac_threshold, self.geofabric_mapping, self.output_path)

            # Write parameter file
            write_hype_par_file(self.output_path)
            
            # Write info and file directory files
            write_hype_info_filedir_files(self.output_path, self.spinup_days, self.hype_results_dir)
            
            self.logger.info("HYPE preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during HYPE preprocessing: {str(e)}")
            raise

class HYPERunner:
    """
    Runner class for the HYPE model within SYMFLUENCE.
    Handles model execution and run-time management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE runner."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Set up HYPE paths
        self.hype_dir = self._get_hype_path()
        self.setup_dir = self.project_dir / "settings" / "HYPE"
        self.output_dir = self._get_output_path()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_hype(self) -> Optional[Path]:
        """
        Run the HYPE model simulation.
        
        Returns:
            Optional[Path]: Path to output directory if successful, None otherwise
        """
        self.logger.info("Starting HYPE model run")
        
        try:
            
            # Create run command
            cmd = self._create_run_command()
            
            # Set up logging
            log_file = self._setup_logging()
            
            # Execute HYPE
            self.logger.info(f"Executing command: {' '.join(map(str, cmd))}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.setup_dir
                )
            
            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                self.logger.info("HYPE simulation completed successfully")
                return self.output_dir
            else:
                self.logger.error("HYPE simulation failed")
                self._analyze_log_file(log_file)
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"HYPE execution failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error running HYPE: {str(e)}")
            raise

    def _get_hype_path(self) -> Path:
        """Get HYPE installation path."""
        hype_path = self.config.get('HYPE_INSTALL_PATH')
        if hype_path == 'default' or hype_path is None:
            return Path(self.config.get('SYMFLUENCE_DATA_DIR')) / 'installs' / 'hype'
        return Path(hype_path)

    def _get_output_path(self) -> Path:
        """Get path for HYPE outputs."""
        if self.config.get('EXPERIMENT_OUTPUT_HYPE') == 'default':
            return (self.project_dir / "simulations" / 
                   self.config.get('EXPERIMENT_ID') / "HYPE")
        return Path(self.config.get('EXPERIMENT_OUTPUT_HYPE'))


    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        hype_exe = self.hype_dir / self.config.get('HYPE_EXE', 'hype')
        
        cmd = [
            str(hype_exe),
            str(self.setup_dir) + '/'  # HYPE requires trailing slash
        ]
        print(cmd)
        return cmd

    def _setup_logging(self) -> Path:
        """Set up HYPE run logging."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f'hype_run_{current_time}.log'

    def _verify_outputs(self) -> bool:
        """Verify HYPE output files exist."""
        required_outputs = [
            'timeCOUT.txt',  # Computed discharge
            'timeEVAP.txt',  # Evaporation
            'timeSNOW.txt'   # Snow water equivalent
        ]
        
        missing_files = []
        for output in required_outputs:
            if not (self.output_dir / output).exists():
                missing_files.append(output)
        
        if missing_files:
            self.logger.error(f"Missing HYPE output files: {', '.join(missing_files)}")
            return False
        return True


class HYPEPostProcessor:
    """
    Postprocessor for HYPE model outputs within SYMFLUENCE.
    Handles output extraction, processing, and analysis.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE postprocessor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Setup paths
        self.sim_dir = (self.project_dir / "simulations" / 
                       self.config.get('EXPERIMENT_ID') / "HYPE")
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_results(self) -> Dict[str, Path]:
        """
        Extract and process all HYPE results.
        
        Returns:
            Dict[str, Path]: Paths to processed result files
        """
        self.logger.info("Extracting HYPE results")
        
        try:
            
            # Process streamflow
            self.extract_streamflow()
            self.logger.info("Streamflow extracted successfully")

            self.plot_streamflow_comparison()
            self.logger.info("Streamflow comparison plot created successfully")
            
        except Exception as e:
            self.logger.error(f"Error extracting HYPE results: {str(e)}")
            raise

    def extract_streamflow(self) -> Optional[Path]:
        try:
            self.logger.info("Processing HYPE streamflow results for outlet")
            
            # Log input file path
            cout_path = self.sim_dir / "timeCOUT.txt"
            self.logger.info(f"Reading HYPE output from: {cout_path}")
            
            # Read computed discharge
            cout = pd.read_csv(cout_path, sep='\t', skiprows=lambda x: x == 0, parse_dates=['DATE'])
            self.logger.info(f"Initial cout DataFrame shape: {cout.shape}")
            self.logger.info(f"Cout columns: {cout.columns.tolist()}")
            
            # Set DATE as index
            cout.set_index('DATE', inplace=True)
            self.logger.info(f"Index type: {type(cout.index)}")
            
            # Get outlet ID and log it
            outlet_id = str(self.config['SIM_REACH_ID'])
            self.logger.info(f"Processing outlet ID: {outlet_id}")
            
            # Create results DataFrame
            if outlet_id not in cout.columns:
                self.logger.error(f"Outlet ID {outlet_id} not found in columns: {cout.columns.tolist()}")
                raise KeyError(f"Outlet ID {outlet_id} not found in HYPE output")
                
            results = pd.DataFrame(cout[outlet_id])
            results.columns = ['HYPE_discharge_cms']
            self.logger.info(f"Results DataFrame shape: {results.shape}")
            self.logger.info(f"Results columns: {results.columns.tolist()}")
            
            # Save individual streamflow results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow.csv"
            self.logger.info(f"Saving individual results to: {output_file}")
            results.to_csv(output_file)
            
            # Append to experiment results file
            results_file = self.project_dir / "results" / f"{self.config['EXPERIMENT_ID']}_results.csv"
            self.logger.info(f"Appending to combined results file: {results_file}")
            
            cms_data = results.copy()
            
            if results_file.exists():
                self.logger.info("Reading existing results file")
                existing_results = pd.read_csv(results_file, index_col=0, parse_dates=True)
                self.logger.info(f"Existing results shape: {existing_results.shape}")
                combined_results = pd.concat([existing_results, cms_data], axis=1)
            else:
                self.logger.info("No existing results file found, creating new")
                combined_results = cms_data
                
            self.logger.info(f"Final combined results shape: {combined_results.shape}")
            combined_results.to_csv(results_file)
                        
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            self.logger.exception("Full traceback:")
            return None
        
    def plot_streamflow_comparison(self) -> Optional[Path]:
        try:
            self.logger.info("Creating streamflow comparison plot")
            
            # Read simulated streamflow
            sim_path = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow.csv"
            self.logger.info(f"Reading simulated streamflow from: {sim_path}")
            
            # Add explicit time parsing
            sim_flow = pd.read_csv(sim_path)
            self.logger.info("Original sim_flow columns: " + str(sim_flow.columns.tolist()))
            
            # Convert the first column to datetime index
            time_col = sim_flow.columns[0]  # Get the name of the first column
            self.logger.info(f"Converting time column: {time_col}")
            sim_flow[time_col] = pd.to_datetime(sim_flow[time_col])
            sim_flow.set_index(time_col, inplace=True)
            
            self.logger.info(f"Simulated flow DataFrame shape: {sim_flow.shape}")
            self.logger.info(f"Simulated flow columns: {sim_flow.columns.tolist()}")
            self.logger.info(f"Simulated flow index type: {type(sim_flow.index)}")
            
            # Read observed streamflow
            obs_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
            self.logger.info(f"Reading observed streamflow from: {obs_path}")
            
            # Add explicit datetime parsing for observed data
            obs_flow = pd.read_csv(obs_path)
            obs_flow['datetime'] = pd.to_datetime(obs_flow['datetime'])
            obs_flow.set_index('datetime', inplace=True)
            
            self.logger.info(f"Observed flow DataFrame shape: {obs_flow.shape}")
            self.logger.info(f"Observed flow columns: {obs_flow.columns.tolist()}")
            self.logger.info(f"Observed flow index type: {type(obs_flow.index)}")
            
            # Get outlet ID
            outlet_id = str(self.config['SIM_REACH_ID'])
            self.logger.info(f"Processing outlet ID: {outlet_id}")
            
            sim_col = 'HYPE_discharge_cms'
            self.logger.info(f"Looking for simulation column: {sim_col}")
            
            if sim_col not in sim_flow.columns:
                self.logger.error(f"Column {sim_col} not found in simulated flow columns: {sim_flow.columns.tolist()}")
                raise KeyError(f"Column {sim_col} not found in simulated flow data")
            
            if 'discharge_cms' not in obs_flow.columns:
                self.logger.error(f"Column 'discharge_cms' not found in observed flow columns: {obs_flow.columns.tolist()}")
                raise KeyError("Column 'discharge_cms' not found in observed flow data")
            
            # Create figure
            plt.figure(figsize=(12, 6))
            plt.plot(sim_flow.index, sim_flow[sim_col], label='Simulated', color='blue', alpha=0.7)
            plt.plot(obs_flow.index, obs_flow['discharge_cms'], label='Observed', color='red', alpha=0.7)
            
            plt.title(f'Streamflow Comparison - {self.domain_name}\nOutlet ID: {outlet_id}')
            plt.xlabel('Date')
            plt.ylabel('Discharge (m³/s)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Ensure the plots directory exists
            plot_dir = self.project_dir / "plots" / "results"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            plot_path = plot_dir / f"{self.config['EXPERIMENT_ID']}_HYPE_streamflow_comparison.png"
            self.logger.info(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating streamflow comparison plot: {str(e)}")
            self.logger.exception("Full traceback:")
            return None
        
        