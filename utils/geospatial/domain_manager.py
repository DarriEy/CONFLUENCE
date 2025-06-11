# In utils/geospatial/domain_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, Tuple
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

from utils.geospatial.discretization_utils import DomainDiscretizer # type: ignore
from utils.reporting.reporting_utils import VisualizationReporter # type: ignore
from utils.geospatial.geofabric_utils import GeofabricSubsetter, GeofabricDelineator, LumpedWatershedDelineator # type: ignore


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
        self.delineator = GeofabricDelineator(self.config, self.logger)
        self.lumped_delineator = LumpedWatershedDelineator(self.config, self.logger)
        self.subsetter = GeofabricSubsetter(self.config, self.logger)
        
        self.domain_discretizer = None  # Initialized when needed
        
        # Create point domain shapefile if method is 'point'
        if self.config.get('DOMAIN_DEFINITION_METHOD') == 'point':
            self.create_point_domain_shapefile()
    
    def create_point_domain_shapefile(self) -> Optional[Path]:
        """
        Create a square basin shapefile from bounding box coordinates for point modelling.
        
        This method creates a rectangular polygon from the BOUNDING_BOX_COORDS and saves it
        as a shapefile for point-based modelling approaches.
        
        Returns:
            Path to the created shapefile or None if failed
        """
        try:
            self.logger.info("Creating point domain shapefile from bounding box coordinates")
            
            # Parse bounding box coordinates
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS', '')
            if not bbox_coords:
                self.logger.error("BOUNDING_BOX_COORDS not found in configuration")
                return None
            
            # Parse coordinates: lat_max/lon_min/lat_min/lon_max
            try:
                lat_max, lon_min, lat_min, lon_max = map(float, bbox_coords.split('/'))
            except ValueError:
                self.logger.error(f"Invalid bounding box format: {bbox_coords}. Expected format: lat_max/lon_min/lat_min/lon_max")
                return None
            
            # Create rectangular polygon from bounding box
            # Coordinates in (lon, lat) order for shapely
            coords = [
                (lon_min, lat_min),  # Bottom-left
                (lon_max, lat_min),  # Bottom-right
                (lon_max, lat_max),  # Top-right
                (lon_min, lat_max),  # Top-left
                (lon_min, lat_min)   # Close polygon
            ]
            
            polygon = Polygon(coords)
            
            # Calculate area in square degrees (approximate)
            area_deg2 = polygon.area
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                'GRU_ID': [1],
                'GRU_area': [area_deg2],  # Area in square degrees
                'basin_name': [self.domain_name],
                'domain_method': ['point']
            }, geometry=[polygon], crs='EPSG:4326')
            
            # Create output directory
            output_dir = self.project_dir / "shapefiles" / "river_basins"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output path
            output_path = output_dir / f"{self.domain_name}_riverBasins_point.shp"
            
            # Save shapefile
            gdf.to_file(output_path)
            
            self.logger.info(f"Point domain shapefile created successfully: {output_path}")
            self.logger.info(f"Bounding box: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}")
            self.logger.info(f"Area: {area_deg2:.6f} square degrees")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating point domain shapefile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
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
        
        # Handle point mode - shapefile already created in __init__
        if domain_method == 'point':
            output_path = self.project_dir / "shapefiles" / "river_basins" / f"{self.domain_name}_riverBasins_point.shp"
            if output_path.exists():
                self.logger.info(f"Point domain shapefile already exists: {output_path}")
                result = output_path
            else:
                self.logger.warning("Point domain shapefile not found, creating it now")
                result = self.create_point_domain_shapefile()
        else:
            # Map of domain methods to their corresponding functions
            domain_methods = {
                'subset': self.subsetter.subset_geofabric,
                'lumped': self.lumped_delineator.delineate_lumped_watershed,
                'delineate': self.delineator.delineate_geofabric
            }
            
            # Execute the appropriate domain method
            method_function = domain_methods.get(domain_method)
            if method_function:
                result = method_function()
                
                # Handle coastal watersheds if needed
                if domain_method == 'delineate' and self.config.get('DELINEATE_COASTAL_WATERSHEDS'):
                    coastal_result = self.delineator.delineate_coastal()
                    self.logger.info(f"Coastal delineation completed: {coastal_result}")
            else:
                self.logger.error(f"Unknown domain definition method: {domain_method}")
                result = None
        
        if result:
            self.logger.info(f"Domain definition completed using method: {domain_method}")
        
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
            plot_path = self.reporter.plot_domain()
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
            discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
            self.logger.info("Creating discretization visualization...")
            plot_path = self.reporter.plot_discretized_domain(discretization_method)
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
            'spatial_mode': self.config.get('DOMAIN_DEFINITION_METHOD'),
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
        valid_methods = ['subset', 'lumped', 'delineate', 'point']  # Added 'point' to valid methods
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD')
        if domain_method not in valid_methods:
            self.logger.error(f"Invalid domain definition method: {domain_method}. Must be one of {valid_methods}")
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