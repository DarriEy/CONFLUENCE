# In utils/geospatial_utils/domain_utilities.py

from pathlib import Path
import logging
from typing import Tuple, Optional
from utils.geospatial_utils.geofabric_utils import GeofabricSubsetter, GeofabricDelineator, LumpedWatershedDelineator


class DomainDelineator:
    """Handles various domain delineation methods."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def subset_geofabric(self) -> Optional[Tuple]:
        """Subset the geofabric."""
        self.logger.info("Starting geofabric subsetting process")
        
        subsetter = GeofabricSubsetter(self.config, self.logger)
        
        try:
            subset_basins, subset_rivers = subsetter.subset_geofabric()
            self.logger.info("Geofabric subsetting completed successfully")
            return subset_basins, subset_rivers
        except Exception as e:
            self.logger.error(f"Error during geofabric subsetting: {str(e)}")
            return None
    
    def delineate_lumped_watershed(self) -> Optional[Path]:
        """Delineate a lumped watershed."""
        self.logger.info("Starting geofabric lumped delineation")
        try:
            delineator = LumpedWatershedDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_lumped_watershed()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None
    
    def delineate_point_buffer_shape(self) -> Optional[Path]:
        """Create a buffer shape around a point."""
        self.logger.info("Starting point buffer")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('point buffer completed successfully')
            return delineator.delineate_point_buffer_shape()
        except Exception as e:
            self.logger.error(f"Error during point buffer delineation: {str(e)}")
            return None
    
    def delineate_geofabric(self) -> Optional[Path]:
        """Delineate the geofabric."""
        self.logger.info("Starting geofabric delineation")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            self.logger.info('Geofabric delineation completed successfully')
            return delineator.delineate_geofabric()
        except Exception as e:
            self.logger.error(f"Error during geofabric delineation: {str(e)}")
            return None
    
    def delineate_coastal(self) -> Optional[Path]:
        """Delineate coastal watersheds."""
        self.logger.info("Starting coastal delineation")
        try:
            delineator = GeofabricDelineator(self.config, self.logger)
            return delineator.delineate_coastal()
        except Exception as e:
            self.logger.error(f"Error during coastal delineation: {str(e)}")
            return None