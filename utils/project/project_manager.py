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
    as the foundation for all other SYMFLUENCE components by establishing the physical
    file organization that the workflow depends on.
    
    Key responsibilities:
    - Creating the project directory structure
    - Generating pour point shapefiles from coordinates
    - Validating project structure integrity
    - Providing project metadata to other components
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing project settings
        logger (logging.Logger): Logger instance for recording operations
        data_dir (Path): Path to the SYMFLUENCE data directory
        code_dir (Path): Path to the SYMFLUENCE code directory
        domain_name (str): Name of the hydrological domain
        project_dir (Path): Path to the specific project directory for this domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the ProjectManager.
        
        Creates a ProjectManager instance with the provided configuration and logger.
        Sets up the basic paths and identifiers needed for project management.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing project settings
            logger (logging.Logger): Logger instance for recording operations
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.code_dir = Path(self.config.get('SYMFLUENCE_CODE_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
    
    def setup_project(self) -> Path:
        """
        Set up the project directory structure.
        
        Creates the main project directory and all required subdirectories based on 
        a predefined structure. This structure includes directories for:
        - Shapefiles (pour point, catchment, river network, river basins)
        - Attribute data
        
        Returns:
            Path: Path to the created project directory
            
        Raises:
            OSError: If directory creation fails due to permission or disk space issues
        """
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        # Create main project directory
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        # Define directory structure
        directories = {
            'shapefiles': ['pour_point', 'catchment', 'river_network', 'river_basins'],
            'attributes': []
        }
        
        # Create directory structure
        for main_dir, subdirs in directories.items():
            main_path = self.project_dir / main_dir
            main_path.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (main_path / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Project directory created at: {self.project_dir}")
        return self.project_dir
    
    def create_pour_point(self) -> Optional[Path]:
        """
        Create pour point shapefile from coordinates if specified.
        
        If pour point coordinates are specified in the configuration, creates
        a GeoDataFrame with a single point geometry and saves it as a shapefile
        at the appropriate location. If 'default' is specified, assumes a 
        user-provided pour point shapefile exists.
        
        Returns:
            Optional[Path]: Path to the created pour point shapefile if successful,
                          None if using a user-provided shapefile or if creation fails
                           
        Raises:
            ValueError: If the pour point coordinates are in an invalid format
            Exception: For other errors during shapefile creation
        """
        # Check if using user-provided shapefile
        if self.config.get('POUR_POINT_COORDS', 'default').lower() == 'default':
            self.logger.info("Using user-provided pour point shapefile")
            return None
        
        try:
            # Parse coordinates
            lat, lon = map(float, self.config['POUR_POINT_COORDS'].split('/'))
            point = Point(lon, lat)  # Note: Point takes (lon, lat) order
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
            
            # Determine output path
            output_path = self.project_dir / "shapefiles" / "pour_point"
            if self.config.get('POUR_POINT_SHP_PATH', 'default') != 'default':
                output_path = Path(self.config['POUR_POINT_SHP_PATH'])
            
            # Determine shapefile name
            pour_point_shp_name = f"{self.domain_name}_pourPoint.shp"
            if self.config.get('POUR_POINT_SHP_NAME', 'default') != 'default':
                pour_point_shp_name = self.config['POUR_POINT_SHP_NAME']
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / pour_point_shp_name
            
            # Save shapefile
            gdf.to_file(output_file)
            self.logger.info(f"Pour point shapefile created successfully: {output_file}")
            return output_file
            
        except ValueError as e:
            self.logger.error("Invalid pour point coordinates format. Expected 'lat/lon'.")
            return None
        except Exception as e:
            self.logger.error(f"Error creating pour point shapefile: {str(e)}")
            return None
    
    def get_project_info(self) -> Dict[str, Any]:
        """
        Get information about the project configuration.
        
        Collects key project metadata into a dictionary for reporting,
        logging, or providing status information to other components.
        
        The returned information includes:
        - Domain name
        - Experiment ID
        - Project directory path
        - Data directory path
        - Pour point coordinates
        
        Returns:
            Dict[str, Any]: Dictionary containing project information
        """
        info = {
            'domain_name': self.domain_name,
            'experiment_id': self.config.get('EXPERIMENT_ID'),
            'project_dir': str(self.project_dir),
            'data_dir': str(self.data_dir),
            'pour_point_coords': self.config.get('POUR_POINT_COORDS')
        }
        
        return info