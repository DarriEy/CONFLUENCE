# In utils/geospatial_utils/domain_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, Tuple

from utils.geospatial_utils.domain_utilities import DomainDelineator # type: ignore
from utils.geospatial_utils.discretization_utils import DomainDiscretizer # type: ignore
from utils.reporting.domain_visualization_utils import DomainVisualizer # type: ignore
from utils.reporting.reporting_utils import VisualizationReporter # type: ignore

class DomainManager:
    """Manages all domain-related operations including definition, discretization, and visualization."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Domain Manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Initialize visualization reporter for domain visualizer
        self.reporter = VisualizationReporter(self.config, self.logger)
        
        # Initialize utility classes
        self.domain_delineator = DomainDelineator(self.config, self.logger)
        self.domain_visualizer = DomainVisualizer(self.config, self.logger, self.reporter)
        self.domain_discretizer = None  # Initialized when needed
    
    def define_domain(self) -> Optional[Union[Path, Tuple[Path, Path]]]:
        """
        Define the domain using the configured method.
        
        Returns:
            Path to the created domain shapefile(s) or None if failed
        """
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        self.logger.info(f"Domain definition workflow starting with: {domain_method}")
        
        # Skip domain definition if shapefile is provided
        if self.config.get('RIVER_BASINS_NAME') != 'default':
            self.logger.info('Shapefile provided, skipping domain definition')
            return None
        
        # Handle point mode
        if self.config.get('SPATIAL_MODE') == 'Point':
            result = self.domain_delineator.delineate_point_buffer_shape()
            return result
        
        # Map of domain methods to their corresponding functions
        domain_methods = {
            'subset': self.domain_delineator.subset_geofabric,
            'lumped': self.domain_delineator.delineate_lumped_watershed,
            'delineate': self.domain_delineator.delineate_geofabric
        }
        
        # Execute the appropriate domain method
        method_function = domain_methods.get(domain_method)
        if method_function:
            result = method_function()
            
            # Handle coastal watersheds if needed
            if domain_method == 'delineate' and self.config.get('DELINEATE_COASTAL_WATERSHEDS'):
                coastal_result = self.domain_delineator.delineate_coastal()
                self.logger.info(f"Coastal delineation completed: {coastal_result}")
                
            self.logger.info(f"Domain definition completed using method: {domain_method}")
        else:
            self.logger.error(f"Unknown domain definition method: {domain_method}")
            result = None
        
        # Visualize the domain after definition
        self.visualize_domain()
        
        self.logger.info(f"Domain definition workflow finished")
        return result
    
    def discretize_domain(self) -> Optional[Union[Path, dict]]:
        """
        Discretize the domain into HRUs or GRUs.
        
        Returns:
            Path to HRU shapefile(s) or dictionary of paths for multiple discretizations
        """
        try:
            discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
            self.logger.info(f"Discretizing domain using method: {discretization_method}")
            
            # Initialize discretizer if not already done
            if self.domain_discretizer is None:
                self.domain_discretizer = DomainDiscretizer(self.config, self.logger)
            
            # Perform discretization
            hru_shapefile = self.domain_discretizer.discretize_domain()
            
            # Log results based on return type
            if isinstance(hru_shapefile, dict):
                self.logger.info(f"Domain discretized successfully. HRU shapefiles:")
                for key, path in hru_shapefile.items():
                    self.logger.info(f"  {key}: {path}")
            elif hru_shapefile:
                self.logger.info(f"Domain discretized successfully. HRU shapefile: {hru_shapefile}")
            else:
                self.logger.error("Domain discretization failed - no shapefile created")
                return None
            
            # Visualize the discretized domain
            self.visualize_discretized_domain()
            
            return hru_shapefile
            
        except Exception as e:
            self.logger.error(f"Error during domain discretization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def visualize_domain(self) -> Optional[Path]:
        """
        Create visualization of the domain.
        
        Returns:
            Path to the created plot or None if failed
        """
        try:
            plot_path = self.domain_visualizer.plot_domain()
            return plot_path
        except Exception as e:
            self.logger.error(f"Error visualizing domain: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def visualize_discretized_domain(self) -> Optional[Path]:
        """
        Create visualization of the discretized domain.
        
        Returns:
            Path to the created plot or None if failed
        """
        try:
            plot_path = self.domain_visualizer.plot_discretized_domain()
            return plot_path
        except Exception as e:
            self.logger.error(f"Error visualizing discretized domain: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about the current domain configuration.
        
        Returns:
            Dictionary containing domain information
        """
        info = {
            'domain_name': self.domain_name,
            'domain_method': self.config.get('DOMAIN_DEFINITION_METHOD'),
            'spatial_mode': self.config.get('SPATIAL_MODE'),
            'discretization_method': self.config.get('DOMAIN_DISCRETIZATION'),
            'pour_point_coords': self.config.get('POUR_POINT_COORDS'),
            'bounding_box': self.config.get('BOUNDING_BOX_COORDS'),
            'project_dir': str(self.project_dir),
        }
        
        # Add shapefile paths if they exist
        river_basins_path = self.project_dir / "shapefiles" / "river_basins"
        catchment_path = self.project_dir / "shapefiles" / "catchment"
        
        if river_basins_path.exists():
            info['river_basins_path'] = str(river_basins_path)
        
        if catchment_path.exists():
            info['catchment_path'] = str(catchment_path)
        
        return info
    
    def validate_domain_configuration(self) -> bool:
        """
        Validate the domain configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        */
        """
        required_settings = [
            'DOMAIN_NAME',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION',
            'BOUNDING_BOX_COORDS'
        ]
        
        # Check required settings
        for setting in required_settings:
            if not self.config.get(setting):
                self.logger.error(f"Required domain setting missing: {setting}")
                return False
        
        # Validate domain definition method
        valid_methods = ['subset', 'lumped', 'delineate']
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        if domain_method not in valid_methods:
            self.logger.error(f"Invalid domain definition method: {domain_method}. Must be one of {valid_methods}")
            return False
        
        # Validate spatial mode if present
        if self.config.get('SPATIAL_MODE'):
            valid_modes = ['Point', 'Lumped', 'Distributed']
            spatial_mode = self.config.get('SPATIAL_MODE')
            if spatial_mode not in valid_modes:
                self.logger.error(f"Invalid spatial mode: {spatial_mode}. Must be one of {valid_modes}")
                return False
        
        # Validate bounding box format
        bbox = self.config.get('BOUNDING_BOX_COORDS', '')
        bbox_parts = bbox.split('/')
        if len(bbox_parts) != 4:
            self.logger.error(f"Invalid bounding box format: {bbox}. Expected format: lat_max/lon_min/lat_min/lon_max")
            return False
        
        try:
            # Check if values are valid floats
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
            
            # Basic validation of coordinates
            if lat_max <= lat_min:
                self.logger.error(f"Invalid bounding box: lat_max ({lat_max}) must be greater than lat_min ({lat_min})")
                return False
            if lon_max <= lon_min:
                self.logger.error(f"Invalid bounding box: lon_max ({lon_max}) must be greater than lon_min ({lon_min})")
                return False
                
        except ValueError:
            self.logger.error(f"Invalid bounding box values: {bbox}. All values must be numeric.")
            return False
        
        self.logger.info("Domain configuration validation passed")
        return True