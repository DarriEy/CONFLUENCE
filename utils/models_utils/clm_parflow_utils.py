from pathlib import Path
import os
import sys
import shutil
import subprocess
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import yaml
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Import required CONFLUENCE utilities
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.configHandling_utils.logging_utils import get_function_logger
from utils.dataHandling_utils.variable_utils import VariableHandler


class CLMParFlowPreProcessor:
    """
    Class to handle preprocessing data for CLM-ParFlow simulations.
    
    This class prepares input files and parameters needed to run a coupled
    CLM-ParFlow simulation as part of the CONFLUENCE framework.
    
    Attributes:
        config (Dict): Configuration dictionary from CONFLUENCE
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the domain
        experiment_id (str): ID of the current experiment
        clm_parflow_input_dir (Path): Directory for CLM-ParFlow input files
        geospatial_dir (Path): Directory containing geospatial data
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the CLM-ParFlow preprocessor.
        
        Args:
            config (Dict): Configuration dictionary
            logger (logging.Logger): Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Set up paths
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Create CLM-ParFlow specific directories
        self.clm_parflow_input_dir = self.project_dir / "forcing" / "CLM_PARFLOW_input"
        self.clm_parflow_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize variable handler for mapping variables
        self.variable_handler = VariableHandler(self.config, self.logger, 
                                               self.config.get('FORCING_DATASET', 'ERA5'), 
                                               'CLM_PARFLOW')

    @get_function_logger
    def run_preprocessing(self):
        """
        Execute the preprocessing workflow for CLM-ParFlow.
        
        This method:
        1. Prepares the domain setup (terrain, subsurface properties)
        2. Processes meteorological forcing data
        3. Creates initial and boundary conditions
        4. Generates CLM and ParFlow input files
        5. Sets up the run directory and scripts
        
        Returns:
            bool: True if preprocessing completes successfully, False otherwise
        """
        self.logger.info("Starting CLM-ParFlow preprocessing")
        
        try:
            # 1. Setup domain parameters
            self.prepare_domain_setup()
            
            # 2. Process meteorological forcing data
            self.process_meteorological_forcing()
            
            # 3. Create initial and boundary conditions
            self.create_initial_conditions()
            self.create_boundary_conditions()
            
            # 4. Generate CLM input files
            self.generate_clm_input_files()
            
            # 5. Generate ParFlow input files
            self.generate_parflow_input_files()
            
            # 6. Create run directory and scripts
            self.setup_run_environment()
            
            self.logger.info("CLM-ParFlow preprocessing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during CLM-ParFlow preprocessing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    @get_function_logger
    def prepare_domain_setup(self):
        """
        Prepare the domain setup for CLM-ParFlow simulation.
        
        This includes:
        - Processing DEM data
        - Generating slope and aspect files
        - Creating soil property maps
        - Setting up land cover parameters
        """
        self.logger.info("Preparing domain setup for CLM-ParFlow")
        
        # Get DEM data
        dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        dem_file = list(dem_dir.glob("*.tif"))[0] if list(dem_dir.glob("*.tif")) else None
        
        if not dem_file:
            self.logger.error("No DEM file found. Cannot prepare domain setup.")
            return False
        
        # Define output files
        domain_dir = self.clm_parflow_input_dir / 'domain'
        domain_dir.mkdir(exist_ok=True)
        
        # Process DEM for ParFlow (adjust resolution, extent, etc.)
        self._process_dem_for_parflow(dem_file, domain_dir)
        
        # Extract soil properties for subsurface parameterization
        soil_dir = self.project_dir / 'attributes' / 'soilclass'
        soil_file = list(soil_dir.glob("*.tif"))[0] if list(soil_dir.glob("*.tif")) else None
        
        if soil_file:
            self._process_soil_for_parflow(soil_file, domain_dir)
        else:
            self.logger.warning("No soil classification file found. Using default soil parameters.")
            self._create_default_soil_params(domain_dir)
        
        # Process land cover for CLM parameters
        land_dir = self.project_dir / 'attributes' / 'landclass'
        land_file = list(land_dir.glob("*.tif"))[0] if list(land_dir.glob("*.tif")) else None
        
        if land_file:
            self._process_landcover_for_clm(land_file, domain_dir)
        else:
            self.logger.warning("No land cover file found. Using default land cover parameters.")
            self._create_default_landcover_params(domain_dir)
        
        return True

    def _process_dem_for_parflow(self, dem_file: Path, output_dir: Path):
        """
        Process DEM data for ParFlow simulation.
        
        Args:
            dem_file (Path): Path to the DEM file
            output_dir (Path): Directory to save processed files
        """
        self.logger.info(f"Processing DEM file {dem_file} for ParFlow")
        
        # In a real implementation, this would:
        # 1. Load the DEM using rasterio or similar
        # 2. Resample to the required resolution
        # 3. Calculate slopes for ParFlow
        # 4. Generate indicator files for ParFlow domain
        # 5. Save as ParFlow-compatible formats
        
        # For now, we'll create placeholder files
        output_files = {
            'slope_x': output_dir / 'slope_x.pfb',
            'slope_y': output_dir / 'slope_y.pfb',
            'mask': output_dir / 'mask.pfb',
            'dem': output_dir / 'dem.pfb'
        }
        
        # In a real implementation, create actual PFB files
        for name, file_path in output_files.items():
            with open(file_path, 'w') as f:
                f.write(f"# Placeholder for {name}")
                
        self.logger.info(f"Created ParFlow domain files in {output_dir}")

    def _process_soil_for_parflow(self, soil_file: Path, output_dir: Path):
        """
        Process soil classification data for ParFlow simulation.
        
        Args:
            soil_file (Path): Path to the soil classification file
            output_dir (Path): Directory to save processed files
        """
        self.logger.info(f"Processing soil file {soil_file} for ParFlow")
        
        # In a real implementation, this would:
        # 1. Load soil classification raster
        # 2. Map soil classes to hydraulic properties
        # 3. Generate 3D subsurface property fields
        # 4. Save in ParFlow-compatible format
        
        # For now, create placeholder files
        output_files = {
            'perm_x': output_dir / 'perm_x.pfb',
            'perm_y': output_dir / 'perm_y.pfb',
            'perm_z': output_dir / 'perm_z.pfb',
            'porosity': output_dir / 'porosity.pfb',
            'specific_storage': output_dir / 'specific_storage.pfb',
            'van_genuchten_alpha': output_dir / 'van_genuchten_alpha.pfb',
            'van_genuchten_n': output_dir / 'van_genuchten_n.pfb'
        }
        
        # Create placeholder files
        for name, file_path in output_files.items():
            with open(file_path, 'w') as f:
                f.write(f"# Placeholder for {name}")
                
        self.logger.info(f"Created ParFlow soil parameter files in {output_dir}")

    def _create_default_soil_params(self, output_dir: Path):
        """
        Create default soil parameter files when no soil data is available.
        
        Args:
            output_dir (Path): Directory to save default parameter files
        """
        # Similar to _process_soil_for_parflow but with default values
        self._process_soil_for_parflow(Path("default"), output_dir)

    def _process_landcover_for_clm(self, landcover_file: Path, output_dir: Path):
        """
        Process land cover data for CLM.
        
        Args:
            landcover_file (Path): Path to land cover classification file
            output_dir (Path): Directory to save processed files
        """
        self.logger.info(f"Processing land cover file {landcover_file} for CLM")
        
        # In a real implementation, this would:
        # 1. Load land cover classification
        # 2. Map to CLM plant functional types
        # 3. Generate parameter files for vegetation properties
        # 4. Create CLM surface dataset
        
        # For now, create placeholder files
        clm_surface_file = output_dir / 'clm_surface_data.nc'
        
        # Create a simple netCDF file as placeholder
        try:
            # Create a simple dataset with minimal attributes
            ds = nc.Dataset(clm_surface_file, 'w')
            
            # Add dimensions
            ds.createDimension('x', 10)
            ds.createDimension('y', 10)
            
            # Add variables
            pft = ds.createVariable('PFT', 'i4', ('y', 'x'))
            pft[:] = 1  # Default to bare soil
            
            # Add metadata
            ds.title = 'CLM Surface Data (Placeholder)'
            ds.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            ds.close()
            self.logger.info(f"Created CLM surface data file: {clm_surface_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating CLM surface data file: {str(e)}")
            # Create empty file as fallback
            with open(clm_surface_file, 'w') as f:
                f.write("# Placeholder for CLM surface data")

    def _create_default_landcover_params(self, output_dir: Path):
        """
        Create default land cover parameter files when no land cover data is available.
        
        Args:
            output_dir (Path): Directory to save default parameter files
        """
        # Similar to _process_landcover_for_clm but with default values
        self._process_landcover_for_clm(Path("default"), output_dir)

    @get_function_logger
    def process_meteorological_forcing(self):
        """
        Process meteorological forcing data for CLM-ParFlow.
        
        This method:
        1. Extracts required variables from forcing data
        2. Converts to CLM-compatible format
        3. Creates meteorological forcing files
        """
        self.logger.info("Processing meteorological forcing data for CLM-ParFlow")
        
        # Define source and destination paths
        forcing_dir = self.project_dir / 'forcing' / 'basin_averaged_data'
        clm_forcing_dir = self.clm_parflow_input_dir / 'forcing'
        clm_forcing_dir.mkdir(exist_ok=True)
        
        # Check if forcing data exists
        forcing_files = list(forcing_dir.glob("*.nc"))
        if not forcing_files:
            self.logger.error("No forcing data found. Cannot process meteorological forcing.")
            return False
        
        # In a real implementation, this would:
        # 1. Load forcing data
        # 2. Convert variables to CLM requirements
        # 3. Regrid to CLM domain
        # 4. Split into required time periods
        # 5. Save in CLM-compatible format
        
        # For now, create a placeholder forcing file
        try:
            # Create sample forcing data
            start_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_START').split()[0], '%Y-%m-%d')
            end_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_END').split()[0], '%Y-%m-%d')
            
            # Create dates range
            dates = [start_date + timedelta(hours=i) for i in range(0, int((end_date - start_date).total_seconds() // 3600) + 24)]
            
            # Create a simple netCDF forcing file
            output_file = clm_forcing_dir / 'clm_forcing.nc'
            ds = nc.Dataset(output_file, 'w')
            
            # Add dimensions
            ds.createDimension('time', len(dates))
            
            # Create time variable
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = f"hours since {start_date.strftime('%Y-%m-%d %H:%M:%S')}"
            time_var.calendar = 'gregorian'
            time_var[:] = range(len(dates))
            
            # Create forcing variables (CLM requires these)
            variables = [
                ('FSDS', 'f4', 'Solar radiation downward (W/m2)'),
                ('FLDS', 'f4', 'Longwave radiation downward (W/m2)'),
                ('APCP', 'f4', 'Precipitation (mm/s)'),
                ('TBOT', 'f4', 'Air temperature (K)'),
                ('WIND', 'f4', 'Wind speed (m/s)'),
                ('PSRF', 'f4', 'Surface pressure (Pa)'),
                ('QBOT', 'f4', 'Specific humidity (kg/kg)')
            ]
            
            # Add each variable with dummy data
            for var_name, var_type, var_desc in variables:
                var = ds.createVariable(var_name, var_type, ('time',))
                var.description = var_desc
                # Set dummy values
                if var_name == 'TBOT':
                    var[:] = 273.15 + 15  # 15°C in Kelvin
                elif var_name == 'PSRF':
                    var[:] = 101325  # Standard atmospheric pressure
                elif var_name == 'QBOT':
                    var[:] = 0.008  # Typical specific humidity
                else:
                    var[:] = 10.0  # Placeholder value
            
            # Add metadata
            ds.title = 'CLM Forcing Data (Placeholder)'
            ds.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            ds.close()
            self.logger.info(f"Created CLM forcing data file: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating CLM forcing data: {str(e)}")
            return False
        
        return True

    @get_function_logger
    def create_initial_conditions(self):
        """
        Create initial condition files for CLM and ParFlow.
        """
        self.logger.info("Creating initial conditions for CLM-ParFlow")
        
        # Define output directory
        ic_dir = self.clm_parflow_input_dir / 'initial_conditions'
        ic_dir.mkdir(exist_ok=True)
        
        # Create ParFlow initial pressure file (placeholder)
        pf_ic_file = ic_dir / 'press.init.pfb'
        with open(pf_ic_file, 'w') as f:
            f.write("# Placeholder for ParFlow initial pressure")
        
        # Create CLM initial conditions
        clm_ic_file = ic_dir / 'clm_restart.nc'
        try:
            # Create minimal restart file
            ds = nc.Dataset(clm_ic_file, 'w')
            ds.title = 'CLM Restart File (Placeholder)'
            ds.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ds.close()
        except Exception as e:
            self.logger.error(f"Error creating CLM restart file: {str(e)}")
            with open(clm_ic_file, 'w') as f:
                f.write("# Placeholder for CLM restart file")
        
        self.logger.info(f"Created initial condition files in {ic_dir}")
        return True

    @get_function_logger
    def create_boundary_conditions(self):
        """
        Create boundary condition files for ParFlow.
        """
        self.logger.info("Creating boundary conditions for ParFlow")
        
        # Define output directory
        bc_dir = self.clm_parflow_input_dir / 'boundary_conditions'
        bc_dir.mkdir(exist_ok=True)
        
        # Create boundary condition files (placeholders)
        bc_files = ['bc_pressure_west.pfb', 'bc_pressure_east.pfb', 
                   'bc_pressure_south.pfb', 'bc_pressure_north.pfb',
                   'bc_pressure_bottom.pfb']
        
        for bc_file in bc_files:
            with open(bc_dir / bc_file, 'w') as f:
                f.write(f"# Placeholder for boundary condition: {bc_file}")
        
        self.logger.info(f"Created boundary condition files in {bc_dir}")
        return True

    @get_function_logger
    def generate_clm_input_files(self):
        """
        Generate CLM input parameter files.
        """
        self.logger.info("Generating CLM input files")
        
        # Define output directory
        clm_dir = self.clm_parflow_input_dir / 'clm_input'
        clm_dir.mkdir(exist_ok=True)
        
        # Create drv_clmin.dat file (CLM input parameters)
        clm_in_file = clm_dir / 'drv_clmin.dat'
        clm_in_content = """
# CLM driver input (Placeholder)
infile             clm                        # CLM main input parameter file name
clm_ic             ../initial_conditions/clm_restart.nc
metforce           ../forcing/clm_forcing.nc  # CLM forcing data file
metfile            narr                      # Type of forcing data
outfile            clm                        # CLM output file name
clm_output         ../output/clm_output.nc    # CLM output NetCDF filename
startcode          1                          # 1 = restart file, 0 = initial run
"""
        
        with open(clm_in_file, 'w') as f:
            f.write(clm_in_content)
        
        # Create a CLM namelist file
        clm_namelist_file = clm_dir / 'clm_config.nc'
        try:
            # Create a simple netCDF file with CLM configuration
            ds = nc.Dataset(clm_namelist_file, 'w')
            ds.title = 'CLM Configuration (Placeholder)'
            ds.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ds.close()
        except Exception as e:
            self.logger.error(f"Error creating CLM namelist: {str(e)}")
            with open(clm_namelist_file, 'w') as f:
                f.write("# Placeholder for CLM configuration")
        
        self.logger.info(f"Created CLM input files in {clm_dir}")
        return True

    @get_function_logger
    def generate_parflow_input_files(self):
        """
        Generate ParFlow input script file.
        """
        self.logger.info("Generating ParFlow input files")
        
        # Define output directory
        pf_dir = self.clm_parflow_input_dir / 'parflow_input'
        pf_dir.mkdir(exist_ok=True)
        
        # Create a ParFlow input script (TCL format)
        pf_script_file = pf_dir / 'run.tcl'
        
        # Basic ParFlow script content (placeholder)
        pf_script_content = """
# ParFlow input script (Placeholder)

# Import the ParFlow TCL package
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

# Project name
pfset FileVersion 4

# Basic domain setup
pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

# Computational grid
pfset ComputationalGrid.Lower.X 0.0
pfset ComputationalGrid.Lower.Y 0.0
pfset ComputationalGrid.Lower.Z 0.0
pfset ComputationalGrid.NX 10
pfset ComputationalGrid.NY 10
pfset ComputationalGrid.NZ 5

# Domain size [m]
pfset ComputationalGrid.DX 100.0
pfset ComputationalGrid.DY 100.0
pfset ComputationalGrid.DZ 2.0

# Time setup
pfset TimingInfo.StartCount 0
pfset TimingInfo.StartTime 0.0
pfset TimingInfo.DumpInterval 1.0
pfset TimingInfo.StopTime 24.0

# Time step [h]
pfset TimeStep.Type Constant
pfset TimeStep.Value 1.0

# Clm coupling
pfset Solver.CLM.MetForcing ../forcing/clm_forcing.nc
pfset Solver.CLM.MetFileName narr
pfset Solver.CLM.IstepStart 1
pfset Solver.CLM.ReuseCount 1
pfset Solver.CLM.WriteLogs False
pfset Solver.CLM.DailyRstDir False
pfset Solver.CLM.SingleFile True
pfset Solver.CLM.CLMDumpInterval 1
"""
        
        with open(pf_script_file, 'w') as f:
            f.write(pf_script_content)
        
        self.logger.info(f"Created ParFlow input files in {pf_dir}")
        return True

    @get_function_logger
    def setup_run_environment(self):
        """
        Create the run directory structure and scripts for CLM-ParFlow.
        """
        self.logger.info("Setting up CLM-ParFlow run environment")
        
        # Define run directory
        run_dir = self.project_dir / "simulations" / self.experiment_id / "CLM_PARFLOW"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create necessary subdirectories
        for subdir in ['output', 'logs', 'scripts']:
            (run_dir / subdir).mkdir(exist_ok=True)
        
        # Create symbolic links to input directories
        for input_dir in ['domain', 'forcing', 'initial_conditions', 
                         'boundary_conditions', 'clm_input', 'parflow_input']:
            src = self.clm_parflow_input_dir / input_dir
            dst = run_dir / input_dir
            
            # Remove existing link/directory if it exists
            if dst.exists():
                if dst.is_symlink():
                    dst.unlink()
                elif dst.is_dir():
                    shutil.rmtree(dst)
            
            # Create symbolic link
            os.symlink(src, dst, target_is_directory=True)
        
        # Create run script
        run_script_file = run_dir / 'scripts' / 'run_clm_parflow.sh'
        run_script_content = """#!/bin/bash
#
# Run script for CLM-ParFlow simulation
#

# Load modules (adjust for your system)
# module load parflow
# module load clm

# Set environment variables
export PARFLOW_DIR=/path/to/parflow
export PARFLOW_BIN=$PARFLOW_DIR/bin

# Run directory
RUN_DIR=$(pwd)/..

# Run simulation
cd $RUN_DIR
tclsh parflow_input/run.tcl

echo "CLM-ParFlow simulation completed"
"""
        
        with open(run_script_file, 'w') as f:
            f.write(run_script_content)
        
        # Make script executable
        run_script_file.chmod(0o755)
        
        self.logger.info(f"Created CLM-ParFlow run environment in {run_dir}")
        return True


class CLMParFlowRunner:
    """
    Class to handle running CLM-ParFlow simulations.
    
    This class manages the execution of coupled CLM-ParFlow simulations
    through the CONFLUENCE framework.
    
    Attributes:
        config (Dict): Configuration dictionary from CONFLUENCE
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the domain
        experiment_id (str): ID of the current experiment
        run_dir (Path): Directory for running the simulation
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the CLM-ParFlow runner.
        
        Args:
            config (Dict): Configuration dictionary
            logger (logging.Logger): Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Set up paths
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Define run directory
        self.run_dir = self.project_dir / "simulations" / self.experiment_id / "CLM_PARFLOW"

    @get_function_logger
    def run_clm_parflow(self):
        """
        Execute a CLM-ParFlow simulation.
        
        This method:
        1. Validates the run environment
        2. Configures the simulation parameters
        3. Executes the CLM-ParFlow coupled model
        4. Monitors the simulation progress
        
        Returns:
            bool: True if simulation completes successfully, False otherwise
        """
        self.logger.info("Starting CLM-ParFlow simulation")
        
        # Check if run directory exists
        if not self.run_dir.exists():
            self.logger.error(f"Run directory {self.run_dir} does not exist. Run preprocessing first.")
            return False
        
        # Validate input files
        if not self._validate_run_environment():
            self.logger.error("Run environment validation failed. Cannot run simulation.")
            return False
        
        # Create a log file for the simulation
        log_dir = self.run_dir / "logs"
        log_file = log_dir / f"clm_parflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Run the simulation
        try:
            # Change to run directory
            original_dir = os.getcwd()
            os.chdir(self.run_dir)
            
            # Execute the run script
            run_script = self.run_dir / "scripts" / "run_clm_parflow.sh"
            
            # Check if the system supports MPI and how many processes to use
            mpi_processes = self.config.get('MPI_PROCESSES', 1)
            use_mpi = mpi_processes > 1
            
            if use_mpi:
                # Construct MPI command
                if shutil.which("mpirun"):
                    cmd = ["mpirun", "-np", str(mpi_processes), "bash", str(run_script)]
                elif shutil.which("srun"):
                    cmd = ["srun", "-n", str(mpi_processes), "bash", str(run_script)]
                else:
                    self.logger.warning("MPI requested but mpirun/srun not found. Running without MPI.")
                    cmd = ["bash", str(run_script)]
            else:
                cmd = ["bash", str(run_script)]
            
            # Execute command
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            with open(log_file, 'w') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                self.logger.info(f"CLM-ParFlow simulation started with PID {process.pid}")
                
                # Wait for process to complete
                process.wait()
                
                if process.returncode == 0:
                    self.logger.info("CLM-ParFlow simulation completed successfully")
                    success = True
                else:
                    self.logger.error(f"CLM-ParFlow simulation failed with return code {process.returncode}")
                    success = False
            
            # Change back to original directory
            os.chdir(original_dir)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error running CLM-ParFlow simulation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Try to change back to original directory
            try:
                os.chdir(original_dir)
            except:
                pass
                
            return False

    def _validate_run_environment(self):
        """
        Validate that all required files and directories exist.
        
        Returns:
            bool: True if the environment is valid, False otherwise
        """
        # Check for key directories
        required_dirs = ['domain', 'forcing', 'initial_conditions', 
                        'boundary_conditions', 'clm_input', 'parflow_input', 
                        'output', 'logs', 'scripts']
        
        for dir_name in required_dirs:
            dir_path = self.run_dir / dir_name
            if not dir_path.exists():
                self.logger.error(f"Required directory {dir_path} does not exist")
                return False
        
        # Check for key files
        required_files = [
            'parflow_input/run.tcl', 
            'clm_input/drv_clmin.dat', 
            'scripts/run_clm_parflow.sh'
        ]
        
        for file_path in required_files:
            full_path = self.run_dir / file_path
            if not full_path.exists():
                self.logger.error(f"Required file {full_path} does not exist")
                return False
        
        # Additional validation could check ParFlow/CLM installation
        # For now, we'll just check if the commands exist
        parflow_found = shutil.which("parflow") is not None
        tclsh_found = shutil.which("tclsh") is not None
        
        if not parflow_found:
            self.logger.warning("ParFlow executable not found in PATH. Simulation may fail.")
        
        if not tclsh_found:
            self.logger.warning("tclsh executable not found in PATH. Simulation may fail.")
        
        return True

    @get_function_logger
    def run_with_parameters(self, parameters: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        Run CLM-ParFlow with a specific set of parameters.
        
        This method is used for calibration runs where parameters need to be modified.
        
        Args:
            parameters (Dict[str, Any]): Dictionary of parameter names and values
            output_dir (Optional[Path]): Optional directory to save outputs
            
        Returns:
            bool: True if simulation completes successfully, False otherwise
        """
        self.logger.info("Running CLM-ParFlow with modified parameters")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.run_dir / "output" / f"param_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modify ParFlow TCL script with new parameters
        try:
            # Read original script
            script_path = self.run_dir / "parflow_input" / "run.tcl"
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            # Create modified script
            modified_script_path = output_dir / "run_modified.tcl"
            
            # Apply parameter modifications
            modified_content = script_content
            for param_name, param_value in parameters.items():
                # Find the parameter in the script
                if isinstance(param_value, (int, float)):
                    # Format value appropriately for TCL
                    param_str = f"{param_value}"
                else:
                    # Assume string value
                    param_str = f'"{param_value}"'
                
                # Look for this parameter pattern in TCL script
                param_pattern = f"pfset {param_name}"
                
                # Check if parameter exists in script
                if param_pattern in modified_content:
                    # Create replacement line with new value
                    # Find the full line with this parameter
                    import re
                    pattern = re.compile(f"pfset {param_name}.*")
                    modified_content = pattern.sub(f"pfset {param_name} {param_str}", modified_content)
                else:
                    # Add parameter to end of script
                    modified_content += f"\npfset {param_name} {param_str}"
            
            # Write modified script
            with open(modified_script_path, 'w') as f:
                f.write(modified_content)
            
            # Use the modified script for this run
            original_script = script_path
            script_path.rename(script_path.with_suffix('.tcl.bak'))
            shutil.copy(modified_script_path, script_path)
            
            # Run simulation
            success = self.run_clm_parflow()
            
            # Restore original script
            if script_path.with_suffix('.tcl.bak').exists():
                script_path.unlink()
                script_path.with_suffix('.tcl.bak').rename(script_path)
            
            # Copy output files to output directory
            if success:
                # Copy all output files
                for file in (self.run_dir / "output").glob("*"):
                    if file.is_file():
                        shutil.copy(file, output_dir)
                
                self.logger.info(f"Saved parameter run outputs to {output_dir}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in parameter run: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Attempt to restore original script if it exists
            try:
                if script_path.with_suffix('.tcl.bak').exists():
                    script_path.unlink()
                    script_path.with_suffix('.tcl.bak').rename(script_path)
            except:
                pass
                
            return False


class CLMParFlowPostProcessor:
    """
    Class to handle postprocessing of CLM-ParFlow simulation results.
    
    This class extracts and processes outputs from CLM-ParFlow simulations
    to prepare them for analysis and visualization in the CONFLUENCE framework.
    
    Attributes:
        config (Dict): Configuration dictionary from CONFLUENCE
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the domain
        experiment_id (str): ID of the current experiment
        simulation_dir (Path): Directory containing simulation results
        output_dir (Path): Directory for processed outputs
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the CLM-ParFlow postprocessor.
        
        Args:
            config (Dict): Configuration dictionary
            logger (logging.Logger): Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Set up paths
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Define simulation and output directories
        self.simulation_dir = self.project_dir / "simulations" / self.experiment_id / "CLM_PARFLOW"
        self.output_dir = self.project_dir / "results" / self.experiment_id / "CLM_PARFLOW"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @get_function_logger
    def extract_streamflow(self):
        """
        Extract and process streamflow results from CLM-ParFlow simulation.
        
        This method:
        1. Locates the ParFlow and CLM output files
        2. Extracts streamflow data 
        3. Converts to standard format for comparison with observations
        
        Returns:
            Path: Path to the processed streamflow file
        """
        self.logger.info("Extracting streamflow results from CLM-ParFlow simulation")
        
        # Check if simulation directory exists
        if not self.simulation_dir.exists() or not (self.simulation_dir / "output").exists():
            self.logger.error(f"Simulation directory {self.simulation_dir} or output subdirectory does not exist")
            return None
        
        # Look for ParFlow output files
        pf_output_files = list((self.simulation_dir / "output").glob("*.pfb"))
        clm_output_file = self.simulation_dir / "output" / "clm_output.nc"
        
        if not pf_output_files and not clm_output_file.exists():
            self.logger.error("No CLM-ParFlow output files found")
            return None
        
        # In a real implementation, this would:
        # 1. Extract subsurface flow data from ParFlow outputs
        # 2. Calculate surface runoff from CLM outputs
        # 3. Combine to get total streamflow at outlet
        
        # For now, create a placeholder streamflow file
        output_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_streamflow.csv"
        
        try:
            # Create a dataframe for streamflow
            start_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_START').split()[0], '%Y-%m-%d')
            end_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_END').split()[0], '%Y-%m-%d')
            
            # Create hourly timestamps
            timestamps = [start_date + timedelta(hours=i) for i in range(0, int((end_date - start_date).total_seconds() // 3600) + 24)]
            
            # Create dataframe with timestamps and placeholder streamflow values
            df = pd.DataFrame({
                'DateTime': timestamps,
                'streamflow': np.random.uniform(1, 10, size=len(timestamps))  # Placeholder random values
            })
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved streamflow data to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    @get_function_logger
    def extract_water_balance(self):
        """
        Extract water balance components from CLM-ParFlow output.
        
        This method processes outputs to calculate water balance components:
        - Precipitation
        - Evapotranspiration
        - Runoff
        - Change in storage
        
        Returns:
            Path: Path to the processed water balance file
        """
        self.logger.info("Extracting water balance from CLM-ParFlow simulation")
        
        # Check if CLM output file exists
        clm_output_file = self.simulation_dir / "output" / "clm_output.nc"
        if not clm_output_file.exists():
            self.logger.error(f"CLM output file {clm_output_file} does not exist")
            return None
        
        # In a real implementation, this would:
        # 1. Extract precipitation, ET from CLM outputs
        # 2. Calculate runoff and storage changes from ParFlow
        # 3. Compile water balance components
        
        # For now, create a placeholder water balance file
        output_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_water_balance.csv"
        
        try:
            # Create a dataframe for water balance
            start_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_START').split()[0], '%Y-%m-%d')
            end_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_END').split()[0], '%Y-%m-%d')
            
            # Create daily timestamps
            timestamps = [start_date + timedelta(days=i) for i in range(0, (end_date - start_date).days + 1)]
            
            # Create dataframe with timestamps and placeholder values
            df = pd.DataFrame({
                'Date': timestamps,
                'Precipitation_mm': np.random.uniform(0, 5, size=len(timestamps)),
                'Evapotranspiration_mm': np.random.uniform(0, 3, size=len(timestamps)),
                'Runoff_mm': np.random.uniform(0, 2, size=len(timestamps)),
                'GW_Recharge_mm': np.random.uniform(0, 1, size=len(timestamps)),
                'Storage_Change_mm': np.random.uniform(-1, 1, size=len(timestamps))
            })
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved water balance data to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting water balance data: {str(e)}")
            return None

    @get_function_logger
    def extract_soil_moisture(self):
        """
        Extract soil moisture profiles from CLM-ParFlow output.
        
        This method processes soil moisture outputs at different depths
        and locations in the domain.
        
        Returns:
            Path: Path to the processed soil moisture file
        """
        self.logger.info("Extracting soil moisture from CLM-ParFlow simulation")
        
        # In a real implementation, this would:
        # 1. Extract 3D soil moisture fields from ParFlow
        # 2. Process into profiles or maps
        # 3. Save in analyzable format
        
        # For now, create a placeholder soil moisture file
        output_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_soil_moisture.nc"
        
        try:
            # Create a simple netCDF file with soil moisture data
            ds = nc.Dataset(output_file, 'w')
            
            # Add dimensions
            start_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_START').split()[0], '%Y-%m-%d')
            end_date = datetime.strptime(self.config.get('EXPERIMENT_TIME_END').split()[0], '%Y-%m-%d')
            n_days = (end_date - start_date).days + 1
            
            ds.createDimension('time', n_days)
            ds.createDimension('depth', 5)  # 5 soil layers
            ds.createDimension('x', 10)
            ds.createDimension('y', 10)
            
            # Create variables
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = f"days since {start_date.strftime('%Y-%m-%d')}"
            time_var[:] = range(n_days)
            
            depth_var = ds.createVariable('depth', 'f4', ('depth',))
            depth_var.units = 'm'
            depth_var[:] = [0.1, 0.3, 0.6, 1.0, 2.0]  # Example depths
            
            x_var = ds.createVariable('x', 'f4', ('x',))
            x_var.units = 'm'
            x_var[:] = range(10)
            
            y_var = ds.createVariable('y', 'f4', ('y',))
            y_var.units = 'm'
            y_var[:] = range(10)
            
            # Create soil moisture variable
            sm_var = ds.createVariable('soil_moisture', 'f4', ('time', 'depth', 'y', 'x'))
            sm_var.units = 'm³/m³'
            sm_var.long_name = 'Volumetric soil moisture'
            
            # Fill with random data between 0.1 and 0.4
            sm_data = np.random.uniform(0.1, 0.4, size=(n_days, 5, 10, 10))
            sm_var[:] = sm_data
            
            # Add metadata
            ds.title = f'CLM-ParFlow Soil Moisture Results for {self.domain_name}'
            ds.created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            ds.close()
            self.logger.info(f"Saved soil moisture data to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting soil moisture data: {str(e)}")
            return None

    @get_function_logger
    def process_results(self):
        """
        Process all CLM-ParFlow simulation results.
        
        This method extracts and processes all relevant outputs from
        the CLM-ParFlow simulation for further analysis.
        
        Returns:
            Dict[str, Path]: Dictionary of processed output file paths by type
        """
        self.logger.info("Processing all CLM-ParFlow simulation results")
        
        results = {}
        
        # Extract streamflow
        streamflow_file = self.extract_streamflow()
        if streamflow_file:
            results['streamflow'] = streamflow_file
        
        # Extract water balance
        water_balance_file = self.extract_water_balance()
        if water_balance_file:
            results['water_balance'] = water_balance_file
        
        # Extract soil moisture
        soil_moisture_file = self.extract_soil_moisture()
        if soil_moisture_file:
            results['soil_moisture'] = soil_moisture_file
        
        # Create summary report
        summary_file = self._create_summary_report(results)
        if summary_file:
            results['summary'] = summary_file
        
        return results

    def _create_summary_report(self, result_files: Dict[str, Path]):
        """
        Create a summary report of simulation results.
        
        Args:
            result_files (Dict[str, Path]): Dictionary of result file paths
            
        Returns:
            Path: Path to summary report file
        """
        summary_file = self.output_dir / f"{self.domain_name}_{self.experiment_id}_summary.txt"
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"CLM-ParFlow Simulation Summary\n")
                f.write(f"==============================\n\n")
                f.write(f"Domain: {self.domain_name}\n")
                f.write(f"Experiment: {self.experiment_id}\n")
                f.write(f"Simulation Period: {self.config.get('EXPERIMENT_TIME_START')} to {self.config.get('EXPERIMENT_TIME_END')}\n\n")
                
                f.write("Processed Results:\n")
                for result_type, file_path in result_files.items():
                    f.write(f"- {result_type}: {file_path}\n")
                
                # In a real implementation, would summarize key metrics
                f.write("\nKey Metrics (Placeholder):\n")
                f.write("- Average Streamflow: 5.0 m³/s\n")
                f.write("- Total Precipitation: 500 mm\n")
                f.write("- Total Evapotranspiration: 300 mm\n")
                f.write("- Average Soil Moisture: 0.25 m³/m³\n")
            
            return summary_file
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {str(e)}")
            return None