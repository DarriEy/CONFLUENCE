import os
import sys
from pathlib import Path
import xarray as xr # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
from typing import Dict, Any, Optional
import subprocess
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.configHandling_utils.logging_utils import get_function_logger # type: ignore
from utils.models_utils.summaflow import ( # type: ignore
    write_summa_forcing,
    write_summa_attribute,
    write_summa_paramtrial,
    write_summa_initial_conditions,
    write_summa_filemanager,
    copy_summa_static_files
)

class SummaPreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.summa_setup_dir = self.project_dir / "settings" / "summa_setup"
        
        # Add these new attributes
        self.geofabric_mapping = self.config.get('GEOFABRIC_MAPPING', {})
        self.landcover_mapping = self.config.get('LANDCOVER_MAPPING', {})
        self.soil_mapping = self.config.get('SOIL_MAPPING', {})
        self.write_mizuroute_domain = self.config.get('WRITE_MIZUROUTE_DOMAIN', False)

    @get_function_logger
    def run_preprocessing(self):
        self.logger.info("Starting SUMMA preprocessing")
        
        self.summa_setup_dir.mkdir(parents=True, exist_ok=True)
        
        # Write SUMMA attribute file
        attr = self.write_summa_attribute()
        
        # Write SUMMA forcing file
        forcing = self.write_summa_forcing(attr)
        
        # Write SUMMA parameter trial file
        self.write_summa_paramtrial(attr)
        
        # Write SUMMA initial conditions file
        self.write_summa_initial_conditions(attr)
        
        # Write SUMMA file manager
        self.write_summa_filemanager(forcing)
        
        # Copy SUMMA static files
        self.copy_summa_static_files()
        
        self.logger.info("SUMMA preprocessing completed")

    def write_summa_attribute(self):
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name

        rivers_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        if rivers_name == 'default':
            rivers_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_delineate.shp"

        rivers_shapefile = self.project_dir / "shapefiles" / "river_network" / rivers_name
        gistool_output = self.project_dir / "attributes"
        
        return write_summa_attribute(
            self.summa_setup_dir,
            subbasins_shapefile,
            rivers_shapefile,
            gistool_output,
            self.config.get('MINIMUM_LAND_FRACTION'),
            self.config.get('HRU_DISCRETIZATION'),
            self.geofabric_mapping,
            self.landcover_mapping,
            self.soil_mapping,
            self.write_mizuroute_domain
        )

    def write_summa_forcing(self, attr):
        easymore_output = self.project_dir / "forcing" / "basin_averaged_data"
        timeshift = self.config.get('FORCING_TIMESHIFT', 0)
        forcing_units = self.config.get('FORCING_UNITS', {})
        return write_summa_forcing(self.summa_setup_dir, timeshift, forcing_units, easymore_output, attr, self.geofabric_mapping)

    def write_summa_paramtrial(self, attr):
        write_summa_paramtrial(attr, self.summa_setup_dir)

    def write_summa_initial_conditions(self, attr):
        write_summa_initial_conditions(attr, self.config.get('SOIL_LAYER_DEPTH'), self.summa_setup_dir)

    def write_summa_filemanager(self, forcing):
        write_summa_filemanager(self.summa_setup_dir, forcing)

    def copy_summa_static_files(self):
        copy_summa_static_files(self.summa_setup_dir)

class SUMMAPostprocessor:
    """
    Postprocessor for SUMMA model outputs via MizuRoute routing.
    Handles extraction and processing of simulation results.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_streamflow(self) -> Optional[Path]:
        try:
            self.logger.info("Extracting SUMMA/MizuRoute streamflow results")
            
            # Get simulation output path
            if self.config.get('SIMULATIONS_PATH') == 'default':
                # Parse the start time and extract the date portion
                start_date = self.config['EXPERIMENT_TIME_START'].split()[0]  # Gets '2011-01-01' from '2011-01-01 01:00'
                sim_file_path = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute' / f"{self.config['EXPERIMENT_ID']}.h.{start_date}-03600.nc"
            else:
                sim_file_path = Path(self.config.get('SIMULATIONS_PATH'))
                
            if not sim_file_path.exists():
                self.logger.error(f"SUMMA/MizuRoute output file not found at: {sim_file_path}")
                return None
                
            # Get simulation reach ID
            sim_reach_ID = self.config.get('SIM_REACH_ID')
            
            # Read simulation data
            ds = xr.open_dataset(sim_file_path, engine='netcdf4')
            
            # Extract data for the specific reach
            segment_index = ds['reachID'].values == int(sim_reach_ID)
            sim_df = ds.sel(seg=segment_index)
            q_sim = sim_df['IRFroutedRunoff'].to_dataframe().reset_index()
            q_sim.set_index('time', inplace=True)
            q_sim.index = q_sim.index.round(freq='h')
            
            # Convert from hourly to daily average
            q_sim_daily = q_sim['IRFroutedRunoff'].resample('D').mean()
            
            # Read existing results file if it exists
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
            if output_file.exists():
                results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
            else:
                results_df = pd.DataFrame(index=q_sim_daily.index)
            
            # Add SUMMA results
            results_df['SUMMA_discharge_cms'] = q_sim_daily
            
            # Save updated results
            results_df.to_csv(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting SUMMA streamflow: {str(e)}")
            raise


class SummaRunner:
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
    running the model, and managing log files.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def run_summa(self):
        """
        Run the SUMMA model either in parallel or serial mode based on configuration.
        """
        if self.config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA', False):
            self.run_summa_parallel()
        else:
            self.run_summa_serial()
        
    def run_summa_parallel(self):
        """
        Run SUMMA in parallel using SLURM array jobs.
        This method handles GRU-based parallelization using SLURM's job array capability.
        """
        self.logger.info("Starting parallel SUMMA run with SLURM")

        # Set up paths and filenames
        summa_path = self.config.get('SETTINGS_SUMMA_PARALLEL_PATH')
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)

        summa_exe = self.config.get('SETTINGS_SUMMA_PARALLEL_EXE')
        settings_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self._get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Get and validate GRU count
        total_grus = self.config.get('SETTINGS_SUMMA_GRU_COUNT')
        if total_grus == 'default':
            # Get catchment shapefile path
            subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
            if subbasins_name == 'default':
                subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
            subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name
            
            # Read shapefile and count unique GRU_IDs
            gdf = gpd.read_file(subbasins_shapefile)
            total_grus = len(gdf['GRU_ID'].unique())
            self.logger.info(f"Counted {total_grus} unique GRUs from shapefile")

        # Get and validate GRUs per job
        grus_per_job = self.config.get('SETTINGS_SUMMA_GRU_PER_JOB')
        if grus_per_job == 'default':
            if total_grus > 500:
                # Divide GRUs among 500 jobs (rounded up to ensure all GRUs are covered)
                grus_per_job = -(-total_grus // 500)  # Ceiling division
                self.logger.info(f"Setting GRUs per job to {grus_per_job} to distribute {total_grus} GRUs across ~500 jobs")
            else:
                grus_per_job = 1
                self.logger.info("Setting default of 1 GRU per job")

        # Calculate number of array jobs needed
        n_array_jobs = -(-total_grus // grus_per_job)  # Ceiling division
        
        # Create SLURM script
        slurm_script = self._create_slurm_script(
            summa_path=summa_path,
            summa_exe=summa_exe,
            settings_path=settings_path,
            filemanager=filemanager,
            summa_log_path=summa_log_path,
            summa_out_path=summa_out_path,
            total_grus=total_grus,
            grus_per_job=grus_per_job,
            n_array_jobs=n_array_jobs - 1  # SLURM arrays are 0-based
        )
        
        # Write SLURM script
        script_path = self.project_dir / 'run_summa_parallel.sh'
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        script_path.chmod(0o755)  # Make executable
        
        # Submit job
        try:
            
            cmd = f"sbatch {script_path}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM array job with ID: {job_id}")
            
            # Backup settings if required
            if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
                backup_path = summa_out_path / "run_settings"
                self._backup_settings(settings_path, backup_path)
                
            # Wait for SLURM job to complete
            while True:
                result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)
                if result.stdout.count('\n') <= 1:  # Only header line remains
                    break
                time.sleep(60)  # Check every minute
            
            self.logger.info("SUMMA parallel run completed, starting output merge")
            
            return self.merge_parallel_outputs()
            
        except Exception as e:
            self.logger.error(f"Error in parallel SUMMA workflow: {str(e)}")
            raise

    def _create_slurm_script(self, summa_path: Path, summa_exe: str, settings_path: Path, 
                            filemanager: str, summa_log_path: Path, summa_out_path: Path,
                            total_grus: int, grus_per_job: int, n_array_jobs: int) -> str:
        
        script = f"""#!/bin/bash
#SBATCH --cpus-per-task={self.config.get('SETTINGS_SUMMA_CPUS_PER_TASK')}
#SBATCH --time={self.config.get('SETTINGS_SUMMA_TIME_LIMIT')}
#SBATCH --mem={self.config.get('SETTINGS_SUMMA_MEM')}
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --job-name=Summa-Parallel
#SBATCH --output={summa_log_path}/summa_%A_%a.out
#SBATCH --error={summa_log_path}/summa_%A_%a.err
#SBATCH --array=0-{n_array_jobs}

# Create required directories
mkdir -p {summa_out_path}
mkdir -p {summa_log_path}

# Calculate GRU range for this job
gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))
gru_end=$(( gru_start + {grus_per_job} - 1 ))

# Ensure we don't exceed total GRUs
if [ $gru_end -gt {total_grus} ]; then
    gru_end={total_grus}
fi

echo "Processing GRUs $gru_start to $gru_end"

# Process each GRU in the range
for gru in $(seq $gru_start $gru_end); do
    echo "Starting GRU $gru"
    
    # Run SUMMA
    {summa_path}/{summa_exe} -g $gru 1 -m {settings_path}/{filemanager}
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "SUMMA failed for GRU $gru with exit code $exit_code"
        exit 1
    fi
    
    echo "Completed GRU $gru"
done

echo "Completed all GRUs for this job"
"""
        return script

    def run_summa_serial(self):
        """
        Run the SUMMA model.

        This method sets up the necessary paths, executes the SUMMA model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting SUMMA run")

        # Set up paths and filenames
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
            
        summa_exe = self.config.get('SUMMA_EXE')
        settings_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self._get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_log_name = "summa_log.txt"
        
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            backup_path = summa_out_path / "run_settings"
            self._backup_settings(settings_path, backup_path)

        # Run SUMMA
        os.makedirs(summa_log_path, exist_ok=True)
        summa_command = f"{str(summa_path / summa_exe)} -m {str(settings_path / filemanager)}"
        
        try:
            with open(summa_log_path / summa_log_name, 'w') as log_file:
                subprocess.run(summa_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info("SUMMA run completed successfully")
            return summa_out_path
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA run failed with error: {e}")
            raise

    def _get_config_path(self, config_key: str, default_suffix: str) -> Path:
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    def _backup_settings(self, source_path: Path, backup_path: Path):
        backup_path.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -R {source_path}/. {backup_path}")
        self.logger.info(f"Settings backed up to {backup_path}")

    def merge_parallel_outputs(self):
        """
        Merge parallel SUMMA outputs into MizuRoute-readable files.
        Handles both daily and timestep outputs, then cleans up individual GRU files.
        """
        self.logger.info("Starting to merge parallel SUMMA outputs")
        
        # Get experiment settings
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")
        mizu_in_path.mkdir(parents=True, exist_ok=True)
        
        # Get time settings
        start_year = int(self.config.get('EXPERIMENT_TIME_START').split('-')[0])
        end_year = int(self.config.get('EXPERIMENT_TIME_END').split('-')[0])
        
        # Define patterns for both daily and timestep files
        file_patterns = {
            'day': {'src': f"{experiment_id}_G*_day.nc", 'dest': f"{experiment_id}_{{year}}_day.nc"},
            'timestep': {'src': f"{experiment_id}_G*_timestep.nc", 'dest': f"{experiment_id}_{{year}}_timestep.nc"}
        }
        
        # Define the variable we want to extract for MizuRoute
        src_variable = 'averageRoutedRunoff'
        
        try:
            # Process each file type (day and timestep)
            for file_type, patterns in file_patterns.items():
                self.logger.info(f"Processing {file_type} files")
                
                # Process each year
                for year in range(start_year, end_year + 1):
                    self.logger.info(f"Processing year {year} for {file_type} files")
                    
                    # Check if output already exists
                    output_file = mizu_in_path / patterns['dest'].format(year=year)
                    if output_file.exists():
                        self.logger.info(f"Output file for {year} ({file_type}) already exists, skipping")
                        continue
                    
                    # Get all source files for this year
                    src_files = list(summa_out_path.glob(patterns['src']))
                    src_files.sort()
                    
                    if not src_files:
                        self.logger.warning(f"No source files found matching pattern: {patterns['src']}")
                        continue
                    
                    # Initialize merged dataset
                    merged_ds = None
                    
                    # Process each GRU file
                    for src_file in src_files:
                        try:
                            # Open dataset and select the year
                            ds = xr.open_dataset(src_file)
                            ds = ds.sel(time=str(year))
                            
                            # Keep only the variable needed for MizuRoute
                            for var in list(ds.data_vars):
                                if var != src_variable:
                                    ds = ds.drop_vars(var)
                            
                            # Merge with existing data
                            if merged_ds is None:
                                merged_ds = ds
                            else:
                                merged_ds = xr.merge([merged_ds, ds])
                            
                            ds.close()
                            
                        except Exception as e:
                            self.logger.error(f"Error processing file {src_file}: {str(e)}")
                            continue
                    
                    # Save merged data
                    if merged_ds is not None:
                        merged_ds.to_netcdf(output_file)
                        self.logger.info(f"Successfully created merged file for {year} ({file_type}): {output_file}")
                        merged_ds.close()
            
            # Cleanup: remove individual GRU files after successful merging
            self.logger.info("Starting cleanup of individual GRU files")
            patterns_to_remove = [
                f"{experiment_id}_G*_day.nc",
                f"{experiment_id}_G*_timestep.nc"
            ]
            
            files_removed = 0
            for pattern in patterns_to_remove:
                for file_path in summa_out_path.glob(pattern):
                    try:
                        file_path.unlink()
                        files_removed += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove file {file_path}: {str(e)}")
            
            self.logger.info(f"Cleanup completed. Removed {files_removed} individual GRU files")
            return mizu_in_path
            
        except Exception as e:
            self.logger.error(f"Error in SUMMA output processing: {str(e)}")
            raise

