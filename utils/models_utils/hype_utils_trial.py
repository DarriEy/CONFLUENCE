from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd # type: ignore
import numpy as np # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
import yaml # type: ignore
import subprocess
import shutil

class HYPEPreProcessor:
    """
    Preprocessor for the HYPE (HYdrological Predictions for the Environment) model.
    Handles data preparation and file setup for HYPE model runs.
    
    This implementation focuses on:
    - Hourly timestep simulation
    - Streamflow modeling (without water quality)
    - Text file inputs/outputs
    - Basic configuration subset
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for HYPE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize the HYPE preprocessor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # HYPE-specific paths
        self.hype_setup_dir = self.project_dir / "settings" / "HYPE"
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_hype_path = self.project_dir / 'forcing' / 'HYPE_input'
        
        # Define paths for HYPE input files
        self.geo_data_path = self.hype_setup_dir / 'GeoData.txt'
        self.geo_class_path = self.hype_setup_dir / 'GeoClass.txt'
        self.par_path = self.hype_setup_dir / 'par.txt'
        self.info_path = self.hype_setup_dir / 'info.txt'

        self.land_intersect_path = self._get_default_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        self.soil_intersect_path = self._get_default_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        
    def run_preprocessing(self):
        """Run the complete HYPE preprocessing workflow."""
        self.logger.info("Starting HYPE preprocessing")
        try:
            self.create_directories()
            self.prepare_forcing_data()
            self.create_geo_data()
            self.create_geo_class()
            self.create_par_file()
            self.create_info_file()
            self.logger.info("HYPE preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during HYPE preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for HYPE setup."""
        dirs_to_create = [
            self.hype_setup_dir,
            self.forcing_hype_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def prepare_forcing_data(self):
        """
        Prepare forcing data files (Pobs.txt and Tobs.txt) from basin-averaged NetCDF data.
        Handles conversion to HYPE's required format for hourly timesteps.
        """
        self.logger.info("Preparing HYPE forcing data")
        
        try:
            # Read basin-averaged forcing data
            forcing_files = sorted(self.forcing_basin_path.glob('*.nc'))
            if not forcing_files:
                raise FileNotFoundError("No forcing files found")
            
            ds = xr.open_mfdataset(forcing_files)
            
            # Extract precipitation and temperature
            precip = ds['pptrate'] * 3600  # Convert to mm/hr
            temp = ds['airtemp'] - 273.15  # Convert to Celsius
            
            # Create Pobs.txt
            self._create_forcing_file(precip, 'Pobs.txt', 'prec')
            
            # Create Tobs.txt
            self._create_forcing_file(temp, 'Tobs.txt', 'temp')
            
            self.logger.info("Forcing data preparation completed")
            
        except Exception as e:
            self.logger.error(f"Error preparing forcing data: {str(e)}")
            raise

    def _create_forcing_file(self, data: xr.DataArray, filename: str, var_name: str):
        """Helper method to create HYPE forcing files."""
        output_path = self.forcing_hype_path / filename
        
        # Format timestamps for HYPE (YYYY-MM-DD HH:MM)
        times = pd.to_datetime(data.time.values).strftime('%Y-%m-%d %H:%M')
        
        # Create header
        header = f"date {' '.join(map(str, range(1, len(data.hru) + 1)))}"
        
        # Create DataFrame
        df = pd.DataFrame(data.values, index=times, columns=range(1, len(data.hru) + 1))
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(header + '\n')
            df.to_csv(f, sep=' ', float_format='%.3f')
        
        self.logger.info(f"Created forcing file: {filename}")

    def create_geo_class(self):
        """
        Create GeoClass.txt file defining characteristics of SLC classes.
        Uses soil and land class data from intersection files.
        """
        self.logger.info("Creating GeoClass.txt")
        
        try:
            # Read soil class intersection data            
            soil_intersect = gpd.read_file(self.soil_intersect_path / self.config.get('INTERSECT_SOIL_NAME'))

            # Read land class intersection data
            land_intersect = gpd.read_file(self.land_intersect_path / self.config.get('INTERSECT_LAND_NAME'))


            # Get unique soil and land classes
            soil_classes = set()
            for j in range(13):  # USGS soil classes 0-12
                col_name = f'USGS_{j}'
                if col_name in soil_intersect.columns:
                    mask = soil_intersect[col_name] > 0
                    if any(mask):
                        soil_classes.add(j)
            
            land_classes = set()
            for j in range(1, 18):  # IGBP land classes 1-17
                col_name = f'IGBP_{j}'
                if col_name in land_intersect.columns:
                    mask = land_intersect[col_name] > 0
                    if any(mask):
                        if j != 17:  # Skip pure water class (17)
                            land_classes.add(j)
            
            # Create SLC combinations
            slc_data = []
            slc_counter = 1
            
            for soil_type in sorted(soil_classes):
                for land_use in sorted(land_classes):
                    # Map IGBP classes to simplified HYPE classes
                    # Here's a simple mapping - adjust based on your needs:
                    # 1-5: Forest, 6-7: Short vegetation, 8-11: Cropland, 
                    # 12: Urban, 13: Snow/Ice, 14-16: Sparse/Barren
                    if land_use <= 5:
                        hype_land = 2  # Forest
                    elif land_use <= 11:
                        hype_land = 1  # Open land
                    else:
                        hype_land = 3  # Other
                    
                    slc_data.append({
                        'slc_no': slc_counter,
                        'landuse': hype_land,
                        'soiltype': soil_type,
                        'crop': 0,  # Not used for streamflow only
                        'tiledepth': 0,  # No tile drainage
                        'streamdepth': 1.0,  # Default stream depth
                        'soillayers': 3,  # Using 3 soil layers
                        'soildepth1': 0.2,  # Top layer depth
                        'soildepth2': 0.5,  # Middle layer depth
                        'soildepth3': 1.0   # Bottom layer depth
                    })
                    slc_counter += 1
            
            # Create DataFrame
            geo_class = pd.DataFrame(slc_data)
            
            # Write file with header comments
            with open(self.geo_class_path, 'w') as f:
                f.write("!! HYPE GeoClass file - Generated from soil and land class data\n")
                f.write("!! Soil types: USGS classification\n")
                f.write("!! Land use: 1=Open, 2=Forest, 3=Other\n")
                geo_class.to_csv(f, sep='\t', index=False)
            
            self.logger.info(f"Created GeoClass.txt with {len(slc_data)} SLC classes")
            
        except Exception as e:
            self.logger.error(f"Error creating GeoClass.txt: {str(e)}")
            raise

    def create_geo_data(self):
        """
        Create GeoData.txt file containing subbasin characteristics.
        Uses elevation data and calculates SLC fractions from intersection files.
        """
        self.logger.info("Creating GeoData.txt")
        
        try:
            # Read catchment data
            catchment = gpd.read_file(self.project_dir / 'shapefiles' / 'catchment' / 
                                    f"{self.domain_name}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp")
            
            # Read river network data
            river_network = gpd.read_file(self.project_dir / 'shapefiles' / 'river_network' / 
                                        f"{self.domain_name}_riverNetwork_delineate.shp")
            
            # Read elevation intersection data
            elev_intersect_path = self._get_default_path('INTERSECT_DEM_PATH', 
                                                        'shapefiles/catchment_intersection/with_dem')
            elev_data = gpd.read_file(elev_intersect_path / self.config.get('INTERSECT_DEM_NAME'))
            
            # Read soil and land class data
            soil_intersect = gpd.read_file(self.soil_intersect_path / self.config.get('INTERSECT_SOIL_NAME'))
            land_intersect = gpd.read_file(self.land_intersect_path / self.config.get('INTERSECT_LAND_NAME'))
            
            # Initialize DataFrame with basic attributes
            geo_data = pd.DataFrame()
            geo_data['subid'] = catchment[self.config['RIVER_BASIN_SHP_RM_GRUID']]
            geo_data['area'] = catchment[self.config['RIVER_BASIN_SHP_AREA']]
            
            # Create mapping between catchment GRU_ID and river segment ID using gru_to_seg
            gru_to_seg_mapping = catchment.set_index(self.config['RIVER_BASIN_SHP_RM_GRUID'])[self.config['RIVER_BASIN_SHP_HRU_TO_SEG']]
            
            # Create mapping of segment IDs to downstream segment IDs
            downstream_mapping = river_network.set_index(self.config['RIVER_NETWORK_SHP_SEGID'])[self.config['RIVER_NETWORK_SHP_DOWNSEGID']]
            
            # Map downstream IDs using the two mappings
            geo_data['maindown'] = geo_data['subid'].map(gru_to_seg_mapping).map(downstream_mapping)
            
            # Get river length from river network if available, otherwise estimate from catchment area
            if 'rivlen' in catchment.columns:
                geo_data['rivlen'] = catchment['rivlen']
            else:
                # Map river lengths from river network using gru_to_seg mapping
                river_lengths = river_network.set_index(self.config['RIVER_NETWORK_SHP_SEGID'])[self.config['RIVER_NETWORK_SHP_LENGTH']]
                geo_data['rivlen'] = geo_data['subid'].map(gru_to_seg_mapping).map(river_lengths)
                
                # For any missing values, estimate using catchment area
                missing_lengths = geo_data['rivlen'].isna()
                if any(missing_lengths):
                    self.logger.warning("Estimating missing river lengths from catchment area")
                    geo_data.loc[missing_lengths, 'rivlen'] = np.sqrt(catchment.loc[missing_lengths, self.config['RIVER_BASIN_SHP_AREA']])
            
            # Add mean elevation from intersection data
            for idx, row in geo_data.iterrows():
                hru_id = row['subid']
                
                # Get elevation
                elev_mask = elev_data[self.config['RIVER_BASIN_SHP_RM_GRUID']].astype(int) == hru_id
                if any(elev_mask):
                    geo_data.loc[idx, 'elevation'] = elev_data['elev_mean'][elev_mask].values[0]
                else:
                    self.logger.warning(f"No elevation data found for subbasin {hru_id}")
                    geo_data.loc[idx, 'elevation'] = 0
                
                # Calculate SLC fractions for each soil-landuse combination
                slc_fractions = self._calculate_slc_fractions(hru_id, soil_intersect, land_intersect)
                for slc_num, fraction in slc_fractions.items():
                    col_name = f'slc_{slc_num}'
                    geo_data.loc[idx, col_name] = fraction
            
            # Add slope information if available from river network, otherwise use catchment data or default
            if self.config['RIVER_NETWORK_SHP_SLOPE'] in river_network.columns:
                slope_mapping = river_network.set_index(self.config['RIVER_NETWORK_SHP_SEGID'])[self.config['RIVER_NETWORK_SHP_SLOPE']]
                geo_data['slope'] = geo_data['subid'].map(gru_to_seg_mapping).map(slope_mapping)
            elif 'slope_mean' in catchment.columns:
                geo_data['slope'] = catchment['slope_mean']
            else:
                self.logger.warning("No slope data available, using default value of 0.01")
                geo_data['slope'] = 0.01
            
            # Replace any remaining NaN values with appropriate defaults
            geo_data['maindown'] = geo_data['maindown'].fillna(-1)  # -1 indicates outlet
            geo_data['rivlen'] = geo_data['rivlen'].fillna(0)
            geo_data['slope'] = geo_data['slope'].fillna(0.01)
            
            # Save to file
            geo_data.to_csv(self.geo_data_path, sep='\t', index=False, float_format='%.6f')
            
            self.logger.info("Created GeoData.txt")
            
        except Exception as e:
            self.logger.error(f"Error creating GeoData.txt: {str(e)}")
            raise
    
    def _calculate_slc_fractions(self, hru_id: int, soil_intersect: gpd.GeoDataFrame, 
                               land_intersect: gpd.GeoDataFrame) -> Dict[int, float]:
        """
        Calculate SLC fractions for a given HRU based on soil and land class intersections.
        
        Args:
            hru_id: HRU identifier
            soil_intersect: GeoDataFrame with soil class intersections
            land_intersect: GeoDataFrame with land class intersections
        
        Returns:
            Dictionary mapping SLC numbers to their fractional areas
        """
        try:
            # Get masks for this HRU
            soil_mask = soil_intersect[self.config['CATCHMENT_SHP_HRUID']].astype(int) == hru_id
            land_mask = land_intersect[self.config['CATCHMENT_SHP_HRUID']].astype(int) == hru_id
            
            # Get soil fractions
            soil_fractions = {}
            for j in range(13):  # USGS soil classes
                col_name = f'USGS_{j}'
                if col_name in soil_intersect.columns:
                    if any(soil_mask):
                        soil_fractions[j] = soil_intersect[col_name][soil_mask].values[0]
            
            # Get land use fractions
            land_fractions = {}
            for j in range(1, 18):  # IGBP land classes
                col_name = f'IGBP_{j}'
                if col_name in land_intersect.columns:
                    if any(land_mask):
                        if j != 17:  # Skip water class
                            land_fractions[j] = land_intersect[col_name][land_mask].values[0]
            
            # Calculate combined SLC fractions
            slc_fractions = {}
            slc_counter = 1
            
            total_fraction = 0
            for soil_type, soil_frac in soil_fractions.items():
                for land_use, land_frac in land_fractions.items():
                    combined_fraction = soil_frac * land_frac
                    if combined_fraction > 0:
                        slc_fractions[slc_counter] = combined_fraction
                        total_fraction += combined_fraction
                    slc_counter += 1
            
            # Normalize fractions if they don't sum to 1
            if total_fraction > 0:
                for slc in slc_fractions:
                    slc_fractions[slc] /= total_fraction
            
            return slc_fractions
            
        except Exception as e:
            self.logger.error(f"Error calculating SLC fractions for HRU {hru_id}: {str(e)}")
            raise

    def create_par_file(self):
        """
        Create par.txt file containing model parameters.
        Focuses on essential parameters for streamflow simulation.
        """
        self.logger.info("Creating par.txt")
        
        try:
            # Create dictionary of parameters with comments
            parameters = {
                "!! Soil parameters": None,
                "wcfc": [0.3, 0.3, 0.4],  # Field capacity
                "wcwp": [0.1, 0.1, 0.15], # Wilting point
                "wcep": [0.4, 0.4, 0.45], # Effective porosity
                "!! Runoff parameters": None,
                "rrcs1": [0.3, 0.3, 0.2], # Recession coefficients
                "rrcs2": [0.1, 0.1, 0.05],
                "!! Snow parameters": None,
                "ttmp": [0.0, 0.0],       # Threshold temperature
                "cmlt": [2.0, 3.0],       # Melting parameter
                "!! River parameters": None,
                "rivvel": [1.0],          # River velocity
                "damp": [0.5]             # Damping parameter
            }
            
            # Write to file
            with open(self.par_path, 'w') as f:
                for key, values in parameters.items():
                    if values is None:  # Comment line
                        f.write(f"{key}\n")
                    else:
                        values_str = ' '.join(map(str, values))
                        f.write(f"{key} {values_str}\n")
            
            self.logger.info("Created par.txt")
            
        except Exception as e:
            self.logger.error(f"Error creating par.txt: {str(e)}")
            raise

    def create_info_file(self):
        """
        Create info.txt file containing simulation settings.
        Configures HYPE for hourly streamflow simulation.
        """
        self.logger.info("Creating info.txt")
        
        try:
            # Get simulation period from config
            start_time = datetime.strptime(self.config['EXPERIMENT_TIME_START'], '%Y-%m-%d %H:%M')
            end_time = datetime.strptime(self.config['EXPERIMENT_TIME_END'], '%Y-%m-%d %H:%M')
            
            # Create info file content
            info_content = [
                "!! HYPE info file - Basic setup for hourly streamflow simulation",
                f"modeldir {self.hype_setup_dir}",
                f"forcingdir {self.forcing_hype_path}",
                f"resultdir {self.project_dir}/simulations/{self.config['EXPERIMENT_ID']}/HYPE",
                f"bdate {start_time.strftime('%Y-%m-%d %H:%M')}",
                f"edate {end_time.strftime('%Y-%m-%d %H:%M')}",
                "steplength 1h",
                "basinoutput variable cout rout",
                "mapoutput variable cout",
                "timeoutput variable cout",
                "criterion NSE",
                "submodel N"  # Not using submodel functionality
            ]
            
            # Write to file
            with open(self.info_path, 'w') as f:
                f.write('\n'.join(info_content))
            
            self.logger.info("Created info.txt")
            
        except Exception as e:
            self.logger.error(f"Error creating info.txt: {str(e)}")
            raise

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.

        Args:
            path_key (str): The key to look up in the config dictionary.
            default_subpath (str): The default subpath to use if the config value is 'default'.

        Returns:
            Path: The resolved path.

        Raises:
            KeyError: If the path_key is not found in the config.
        """
        try:
            path_value = self.config.get(path_key)
            if path_value == 'default' or path_value is None:
                return self.project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            self.logger.error(f"Config key '{path_key}' not found")
            raise


class HYPERunner:
    """
    Runner class for the HYPE (HYdrological Predictions for the Environment) model.
    Handles model execution, output processing, and file management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for HYPE
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize the HYPE runner."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # HYPE-specific paths
        self.result_dir = self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "HYPE"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Get executable path
        self.hype_path = self._get_install_path()

    def run_hype(self) -> Optional[Path]:
        """
        Run the HYPE model.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info("Starting HYPE model run")
        
        try:
            # Create output directory
            self.output_path = self._get_output_path()
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Backup settings
            self._backup_settings()
            
            # Execute HYPE
            success = self._execute_hype()
            
            if success:
                # Process outputs
                self._process_outputs()
                self.logger.info("HYPE run completed successfully")
                return self.output_path
            else:
                self.logger.error("HYPE run failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during HYPE run: {str(e)}")
            raise

    def _get_install_path(self) -> Path:
        """Get the HYPE installation path."""
        hype_path = self.config.get('HYPE_INSTALL_PATH')
        if hype_path == 'default':
            return Path(self.config.get('CONFLUENCE_CODE_DIR')) / 'installs' / 'hype' / 'bin'
        return Path(hype_path)

    def _get_output_path(self) -> Path:
        """Get the path for HYPE outputs."""
        return (self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "HYPE" 
                if self.config.get('EXPERIMENT_OUTPUT_HYPE') == 'default' 
                else Path(self.config.get('EXPERIMENT_OUTPUT_HYPE')))

    def _backup_settings(self):
        """Backup important HYPE settings files for reproducibility."""
        self.logger.info("Backing up HYPE settings files")
        
        # Create backup directory
        backup_dir = self.output_path / '_settings_backup'
        backup_dir.mkdir(exist_ok=True)
        
        # List of files to backup
        settings_dir = self.project_dir / "settings" / "HYPE"
        files_to_backup = [
            'GeoData.txt',
            'GeoClass.txt',
            'par.txt',
            'info.txt'
        ]
        
        # Copy files
        for filename in files_to_backup:
            source = settings_dir / filename
            if source.exists():
                shutil.copy2(source, backup_dir / filename)
                self.logger.debug(f"Backed up {filename}")
            else:
                self.logger.warning(f"Could not find {filename} for backup")

    def _execute_hype(self) -> bool:
        """
        Execute the HYPE model.
        
        Returns:
            bool: True if execution was successful, False otherwise
        """
        self.logger.info("Executing HYPE model")
        
        try:
            # Construct command
            hype_exe = self.hype_path / self.config.get('HYPE_EXE', 'hype.exe')
            model_path = self.project_dir / 'settings' / 'HYPE'
            
            # HYPE expects the path to end with a slash
            model_path_str = str(model_path) + '/'
            
            command = [
                str(hype_exe),
                model_path_str
            ]
            
            # Create log directory
            log_dir = self.output_path / 'logs'
            log_dir.mkdir(exist_ok=True)
            
            # Run HYPE with log file
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f'hype_run_{current_time}.log'
            
            self.logger.info(f"Running command: {' '.join(map(str, command))}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            # Check HYPE return code (0 means success)
            success = result.returncode == 0
            
            # Also check if output files were created
            main_output = self.output_path / f"timeCOUT.txt"
            if not main_output.exists():
                self.logger.error("HYPE output file timeCOUT.txt not found")
                success = False
            
            self.logger.info(f"HYPE execution {'completed successfully' if success else 'failed'}")
            
            # Process any error messages from the log
            if not success:
                self._analyze_log_file(log_file)
            
            return success
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"HYPE execution failed with error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error executing HYPE: {str(e)}")
            return False

    def _analyze_log_file(self, log_file: Path):
        """Analyze HYPE log file for error messages."""
        self.logger.info("Analyzing HYPE log file for errors")
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # Look for common error patterns
            error_patterns = [
                "ERROR",
                "Error",
                "error",
                "failed",
                "Failed",
                "FAILED"
            ]
            
            for pattern in error_patterns:
                if pattern in log_content:
                    # Get the line containing the error
                    error_line = next(line for line in log_content.split('\n') if pattern in line)
                    self.logger.error(f"Found error in log: {error_line}")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing log file: {str(e)}")

    def _process_outputs(self):
        """Process and organize HYPE output files."""
        self.logger.info("Processing HYPE outputs")
        
        try:
            # Process time series outputs
            self._process_time_output()
            
            # Process basin outputs
            self._process_basin_output()
            
            # Process map outputs
            self._process_map_output()
            
            # Process performance criteria
            self._process_criteria()
            
            self.logger.info("HYPE output processing completed")
            
        except Exception as e:
            self.logger.error(f"Error processing HYPE outputs: {str(e)}")
            raise

    def _process_time_output(self):
        """Process HYPE time series output files."""
        self.logger.info("Processing time series outputs")
        
        # Find all timeCOUT.txt files
        time_files = list(self.output_path.glob("time*.txt"))
        
        for file in time_files:
            try:
                # Read the file
                df = pd.read_csv(file, sep='\s+', index_col='DATE', parse_dates=True)
                
                # Add metadata
                df.attrs = {
                    'model': 'HYPE',
                    'domain': self.domain_name,
                    'experiment_id': self.config.get('EXPERIMENT_ID'),
                    'variable': file.stem[4:],  # Remove 'time' prefix
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save as CSV with metadata
                output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_{file.stem}.csv"
                
                # Save data
                df.to_csv(output_file)
                
                # Save metadata separately
                meta_file = output_file.with_suffix('.meta')
                with open(meta_file, 'w') as f:
                    for key, value in df.attrs.items():
                        f.write(f"{key}: {value}\n")
                
                self.logger.debug(f"Processed {file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file.name}: {str(e)}")

    def _process_basin_output(self):
        """Process HYPE basin output files."""
        self.logger.info("Processing basin outputs")
        
        # Find all basin output files (numeric filenames)
        basin_files = [f for f in self.output_path.glob("*.txt") 
                      if f.stem.isdigit()]
        
        for file in basin_files:
            try:
                # Read the file
                df = pd.read_csv(file, sep='\s+', index_col='DATE', parse_dates=True)
                
                # Add metadata
                df.attrs = {
                    'model': 'HYPE',
                    'domain': self.domain_name,
                    'experiment_id': self.config.get('EXPERIMENT_ID'),
                    'subbasin': file.stem,
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save as CSV with metadata
                output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_{file.stem}_basin.csv"
                
                # Save data
                df.to_csv(output_file)
                
                # Save metadata separately
                meta_file = output_file.with_suffix('.meta')
                with open(meta_file, 'w') as f:
                    for key, value in df.attrs.items():
                        f.write(f"{key}: {value}\n")
                
                self.logger.debug(f"Processed {file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file.name}: {str(e)}")

    def _process_map_output(self):
        """Process HYPE map output files."""
        self.logger.info("Processing map outputs")
        
        # Find all map output files
        map_files = list(self.output_path.glob("map*.txt"))
        
        for file in map_files:
            try:
                # Read the file
                df = pd.read_csv(file, sep=',')
                
                # Add metadata
                df.attrs = {
                    'model': 'HYPE',
                    'domain': self.domain_name,
                    'experiment_id': self.config.get('EXPERIMENT_ID'),
                    'variable': file.stem[3:],  # Remove 'map' prefix
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save as CSV with metadata
                output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_{file.stem}.csv"
                
                # Save data
                df.to_csv(output_file, index=False)
                
                # Save metadata separately
                meta_file = output_file.with_suffix('.meta')
                with open(meta_file, 'w') as f:
                    for key, value in df.attrs.items():
                        f.write(f"{key}: {value}\n")
                
                self.logger.debug(f"Processed {file.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {file.name}: {str(e)}")

    def _process_criteria(self):
        """Process HYPE performance criteria output files."""
        self.logger.info("Processing performance criteria")
        
        try:
            # Process subass files (subbasin assessment)
            subass_files = list(self.output_path.glob("subass*.txt"))
            for file in subass_files:
                df = pd.read_csv(file, sep='\s+')
                output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_{file.stem}_criteria.csv"
                df.to_csv(output_file, index=False)
            
            # Process simass file (simulation assessment)
            simass_file = self.output_path / "simass.txt"
            if simass_file.exists():
                df = pd.read_csv(simass_file, sep='\s+')
                output_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_simulation_criteria.csv"
                df.to_csv(output_file, index=False)
            
            self.logger.info("Performance criteria processing completed")
            
        except Exception as e:
            self.logger.error(f"Error processing criteria files: {str(e)}")

    




class HYPEPostProcessor:
    """
    Postprocessor for HYPE (HYdrological Predictions for the Environment) model outputs.
    Handles output extraction, processing, analysis and unit conversion.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for HYPE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
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
        """
        Extract simulated streamflow from HYPE output and process it.
        Handles unit conversions and data organization.
        
        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting HYPE streamflow results")
            
            # Define paths
            sim_dir = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'HYPE'
            
            # Process simulated streamflow
            cout_file = sim_dir / "timeCOUT.txt"  # Simulated discharge
            rout_file = sim_dir / "timeROUT.txt"  # Observed discharge
            
            # Read simulated discharge
            cout = pd.read_csv(cout_file, sep='\s+', parse_dates=['DATE'], index_col='DATE')
            
            # Read observed discharge if available
            rout = None
            if rout_file.exists():
                rout = pd.read_csv(rout_file, sep='\s+', parse_dates=['DATE'], index_col='DATE')
            
            # Get catchment area from river basins shapefile
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_delineate.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Calculate total area in km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
            
            # Create results DataFrame
            results = pd.DataFrame(index=cout.index)
            
            # Process each subbasin
            for subid in cout.columns:
                # Convert units from m3/s to mm/day if needed
                # Q(mm/day) = Q(m3/s) * 86.4 / area(km2)
                q_sim = cout[subid]
                q_sim_mm = q_sim * 86.4 / area_km2
                
                # Add to results with prefixes for identification
                results[f'HYPE_discharge_cms_{subid}'] = q_sim
                results[f'HYPE_discharge_mmday_{subid}'] = q_sim_mm
                
                # Add observed data if available
                if rout is not None and subid in rout.columns:
                    q_obs = rout[subid]
                    q_obs_mm = q_obs * 86.4 / area_km2
                    results[f'Obs_discharge_cms_{subid}'] = q_obs
                    results[f'Obs_discharge_mmday_{subid}'] = q_obs_mm
            
            # Add metadata as attributes
            results.attrs = {
                'model': 'HYPE',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'catchment_area_km2': area_km2,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_start': results.index.min().strftime('%Y-%m-%d %H:%M'),
                'simulation_end': results.index.max().strftime('%Y-%m-%d %H:%M'),
                'timestep': 'hourly' if 'H' in str(results.index.freq) else 'daily'
            }
            
            # Save to CSV
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow_results.csv"
            results.to_csv(output_file)
            
            # Save metadata separately
            meta_file = output_file.with_suffix('.meta')
            with open(meta_file, 'w') as f:
                for key, value in results.attrs.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise

    def analyze_performance(self) -> Optional[Path]:
        """
        Analyze model performance using HYPE's criteria output files.
        Combines and processes various performance metrics.
        
        Returns:
            Optional[Path]: Path to the performance analysis file
        """
        try:
            self.logger.info("Analyzing HYPE model performance")
            
            sim_dir = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'HYPE'
            
            # Process subbasin criteria
            subass_files = list(sim_dir.glob("subass*.txt"))
            subbasin_criteria = []
            
            for file in subass_files:
                df = pd.read_csv(file, sep='\s+')
                if not df.empty:
                    subbasin_criteria.append(df)
            
            # Process simulation criteria
            simass_file = sim_dir / "simass.txt"
            if simass_file.exists():
                sim_criteria = pd.read_csv(simass_file, sep='\s+')
            else:
                sim_criteria = pd.DataFrame()
            
            # Combine results
            analysis = {
                'simulation_criteria': sim_criteria,
                'subbasin_criteria': pd.concat(subbasin_criteria) if subbasin_criteria else pd.DataFrame()
            }
            
            # Format and save results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_performance_analysis.txt"
            with open(output_file, 'w') as f:
                # Write header
                f.write(f"HYPE Model Performance Analysis\n")
                f.write(f"Domain: {self.domain_name}\n")
                f.write(f"Experiment: {self.config['EXPERIMENT_ID']}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write simulation criteria
                f.write("Overall Simulation Criteria:\n")
                f.write("-" * 30 + "\n")
                if not sim_criteria.empty:
                    for col in sim_criteria.columns:
                        f.write(f"{col}: {sim_criteria[col].iloc[0]:.4f}\n")
                else:
                    f.write("No simulation criteria available\n")
                f.write("\n")
                
                # Write subbasin criteria summary
                f.write("Subbasin Criteria Summary:\n")
                f.write("-" * 30 + "\n")
                if not analysis['subbasin_criteria'].empty:
                    summary = analysis['subbasin_criteria'].describe()
                    f.write(summary.to_string())
                else:
                    f.write("No subbasin criteria available\n")
                
                self.logger.info(f"Performance analysis saved to: {output_file}")
                
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            raise

    def extract_water_balance(self) -> Optional[Path]:
        """
        Extract and process water balance components from HYPE output.
        Handles precipitation, evaporation, runoff, and storage terms.
        
        Returns:
            Optional[Path]: Path to the water balance file
        """
        try:
            self.logger.info("Extracting water balance components")
            
            sim_dir = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'HYPE'
            
            # List of water balance components to process
            components = {
                'PREC': 'timePREC.txt',  # Precipitation
                'EVAP': 'timeEVAP.txt',  # Evaporation
                'CRUN': 'timeCRUN.txt',  # Local runoff
                'COUT': 'timeCOUT.txt',  # Outlet discharge
            }
            
            # Read each component
            wb_data = {}
            for comp, filename in components.items():
                file_path = sim_dir / filename
                if file_path.exists():
                    df = pd.read_csv(file_path, sep='\s+', parse_dates=['DATE'], index_col='DATE')
                    wb_data[comp] = df
                else:
                    self.logger.warning(f"Water balance component file not found: {filename}")
            
            if not wb_data:
                self.logger.error("No water balance components found")
                return None
            
            # Process water balance
            results = pd.DataFrame(index=next(iter(wb_data.values())).index)
            
            # Add components to results
            for comp, df in wb_data.items():
                for col in df.columns:
                    results[f'{comp}_{col}'] = df[col]
            
            # Calculate residual if possible
            if all(k in wb_data for k in ['PREC', 'EVAP', 'COUT']):
                for col in wb_data['COUT'].columns:
                    # Extract matching columns
                    P = wb_data['PREC'][col] if col in wb_data['PREC'].columns else 0
                    E = wb_data['EVAP'][col] if col in wb_data['EVAP'].columns else 0
                    Q = wb_data['COUT'][col]
                    
                    # Calculate residual (P - E - Q)
                    results[f'RESD_{col}'] = P - E - Q
            
            # Add metadata
            results.attrs = {
                'model': 'HYPE',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'components': list(wb_data.keys()),
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_water_balance.csv"
            results.to_csv(output_file)
            
            # Save metadata
            meta_file = output_file.with_suffix('.meta')
            with open(meta_file, 'w') as f:
                for key, value in results.attrs.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Water balance saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting water balance: {str(e)}")
            raise

    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to resolve file paths."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        return Path(self.config.get(file_type))