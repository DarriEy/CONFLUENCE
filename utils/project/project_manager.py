# In utils/project/project_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional
import geopandas as gpd
from shapely.geometry import Point

class ProjectManager:
    """
    Manages project-level operations including directory structure and initialization.
    
    The ProjectManager is responsible for creating and managing the project directory
    structure, handling pour point creation, and maintaining project metadata. It serves
    as the foundation for all other CONFLUENCE components by establishing the physical
    file organization that the workflow depends on.
    
    Key responsibilities:
    - Creating the project directory structure
    - Generating pour point shapefiles from coordinates
    - Handling special cases like point-scale simulations
    - Validating project structure integrity
    - Providing project metadata to other components
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing project settings
        logger (logging.Logger): Logger instance for recording operations
        data_dir (Path): Path to the CONFLUENCE data directory
        code_dir (Path): Path to the CONFLUENCE code directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the specific project directory for this domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Project Manager.
        
        Creates a ProjectManager instance with the provided configuration and logger.
        This sets up the basic paths and identifiers needed for project management.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing project settings
            logger (logging.Logger): Logger instance for recording operations
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('CONFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
    
    def setup_project(self) -> Path:
        """
        Set up the project directory structure.
        
        Creates the main project directory and all required subdirectories based on 
        a predefined structure. This structure includes directories for:
        - Shapefiles (pour point, catchment, river network, river basins)
        - Observations and streamflow data
        - Documentation
        - Attribute data
        
        For point-scale simulations, this method also updates the bounding box coordinates.
        
        Returns:
            Path: Path to the created project directory
            
        Raises:
            OSError: If directory creation fails due to permission or disk space issues
        """
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        directories = {
            'shapefiles': ['pour_point', 'catchment', 'river_network', 'river_basins'],
            'observations/streamflow': ['raw_data'],
            'documentation': [],
            'attributes': []
        }
        
        for main_dir, subdirs in directories.items():
            main_path = self.project_dir / main_dir
            main_path.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (main_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # If in point mode, update bounding box coordinates
        #if self.config.get('SPATIAL_MODE') == 'Point':
        #    self._update_bounding_box_for_point_mode()
        
        self.logger.info(f"Project directory created at: {self.project_dir}")
        return self.project_dir
    
    def create_pour_point(self) -> Optional[Path]:
        """
        Create pour point shapefile from coordinates if needed.
        
        If pour point coordinates are specified in the configuration, this method
        creates a GeoDataFrame with a single point geometry and saves it as a
        shapefile at the appropriate location. If 'default' is specified in the
        configuration, it assumes a user-provided pour point shapefile exists.
        
        Returns:
            Optional[Path]: Path to the created pour point shapefile if successful,
                           None if using a user-provided shapefile or if creation fails
                           
        Raises:
            ValueError: If the pour point coordinates are in an invalid format
            Exception: For other errors during shapefile creation
        """
        if self.config.get('POUR_POINT_COORDS', 'default').lower() == 'default':
            self.logger.info("Using user-provided pour point shapefile")
            return None
        
        try:
            lat, lon = map(float, self.config['POUR_POINT_COORDS'].split('/'))
            point = Point(lon, lat)
            gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
            
            output_path = self.project_dir / "shapefiles" / "pour_point"
            if self.config.get('POUR_POINT_SHP_PATH') != 'default':
                output_path = Path(self.config['POUR_POINT_SHP_PATH'])
            
            pour_point_shp_name = f"{self.domain_name}_pourPoint.shp"
            if self.config.get('POUR_POINT_SHP_NAME') != 'default':
                pour_point_shp_name = self.config['POUR_POINT_SHP_NAME']
            
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / pour_point_shp_name
            
            gdf.to_file(output_file)
            self.logger.info(f"Pour point shapefile created successfully: {output_file}")
            return output_file
            
        except ValueError:
            self.logger.error("Invalid pour point coordinates format. Expected 'lat/lon'.")
        except Exception as e:
            self.logger.error(f"Error creating pour point shapefile: {str(e)}")
        
        return None
    
    def _update_bounding_box_for_point_mode(self):
        """
        Update the bounding box coordinates for point-scale simulations.
        
        For point-scale simulations, this method creates a small bounding box around 
        the pour point coordinates by adding a buffer distance in all directions.
        It then updates both the runtime configuration and the active configuration file.
        
        The buffer distance is controlled by the POINT_BUFFER_DISTANCE configuration
        parameter, with a default of 0.001 degrees (approximately 100m at the equator).
        
        Raises:
            ValueError: If pour point coordinates are incorrectly formatted
            Exception: For other errors during bounding box update
        """
        try:
            pour_point_coords = self.config.get('POUR_POINT_COORDS', '')
            if not pour_point_coords or pour_point_coords.lower() == 'default':
                self.logger.warning("Pour point coordinates not specified, cannot update bounding box for point mode")
                return
            
            lat, lon = map(float, pour_point_coords.split('/'))
            buffer_dist = self.config.get('POINT_BUFFER_DISTANCE', 0.001)
            
            min_lon = round(lon - buffer_dist, 4)
            max_lon = round(lon + buffer_dist, 4)
            min_lat = round(lat - buffer_dist, 4)
            max_lat = round(lat + buffer_dist, 4)
            
            new_bbox = f"{max_lat}/{min_lon}/{min_lat}/{max_lon}"
            
            self.logger.info(f"Updating bounding box for point-scale simulation to: {new_bbox}")
            
            self.config['BOUNDING_BOX_COORDS'] = new_bbox
            self._update_active_config_file('BOUNDING_BOX_COORDS', new_bbox)
            
        except Exception as e:
            self.logger.error(f"Error updating bounding box for point mode: {str(e)}")
    
    def _update_active_config_file(self, key: str, value: str):
        """
        Update a specific key in the active configuration file.
        
        This method modifies the active configuration file on disk to reflect runtime
        changes to the configuration. It searches for the specific key and replaces
        its value, preserving the rest of the file.
        
        Args:
            key (str): The configuration key to update
            value (str): The new value to set for the key
            
        Raises:
            FileNotFoundError: If the active configuration file cannot be found
            PermissionError: If the file cannot be written due to permission issues
            Exception: For other errors during file update
        """
        try:
            if 'CONFLUENCE_CODE_DIR' not in self.config:
                self.logger.warning("CONFLUENCE_CODE_DIR not specified, cannot update config file")
                return
            
            config_path = Path(self.config['CONFLUENCE_CODE_DIR']) / '0_config_files' / 'config_active.yaml'
            
            if not config_path.exists():
                self.logger.warning(f"Active config file not found at {config_path}")
                return
            
            with open(config_path, 'r') as f:
                lines = f.readlines()
            
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}:"):
                    lines[i] = f"{key}: {value}  # Updated for point-scale simulation\n"
                    updated = True
                    break
            
            if not updated:
                self.logger.warning(f"Could not find {key} in config file to update")
                return
            
            with open(config_path, 'w') as f:
                f.writelines(lines)
            
            self.logger.info(f"Updated {key} in active config file: {config_path}")
        
        except Exception as e:
            self.logger.error(f"Error updating config file: {str(e)}")
    
    def validate_project_structure(self) -> bool:
        """
        Validate that the project structure is properly set up.
        
        This method checks that all required directories exist in the project
        structure. It's used to verify that the project was initialized correctly
        before proceeding with other workflow steps.
        
        The required directories include:
        - Main project directory
        - Shapefiles directory
        - Attributes directory 
        - Forcing directory
        - Simulations directory
        - Evaluation directory
        - Plots directory
        - Optimisation directory
        
        Returns:
            bool: True if all required directories exist, False otherwise
        """
        required_dirs = [
            self.project_dir,
            self.project_dir / 'shapefiles',
            self.project_dir / 'attributes',
            self.project_dir / 'forcing',
            self.project_dir / 'simulations',
            self.project_dir / 'evaluation',
            self.project_dir / 'plots',
            self.project_dir / 'optimisation'
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.logger.warning(f"Required directory missing: {dir_path}")
                all_exist = False
        
        return all_exist
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get information about the project configuration.
        
        This method collects key project metadata into a dictionary, which can be
        used for reporting, logging, or providing status information to other
        components of the CONFLUENCE system.
        
        The returned information includes:
        - Domain name
        - Experiment ID
        - Project directory path
        - Data directory path
        - Pour point coordinates
        - Structure validation status
        
        Returns:
            Dict[str, Any]: Dictionary containing project information
        """
        info = {
            'domain_name': self.domain_name,
            'experiment_id': self.config.get('EXPERIMENT_ID'),
            'project_dir': str(self.project_dir),
            'data_dir': str(self.data_dir),
            'pour_point_coords': self.config.get('POUR_POINT_COORDS'),
            'structure_valid': self.validate_project_structure()
        }
        
        return info