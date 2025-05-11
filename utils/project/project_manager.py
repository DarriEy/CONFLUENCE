# In utils/project_utils/project_manager.py

from pathlib import Path
import logging
from typing import Dict, Any, Optional

from utils.dataHandling_utils.data_utils import ProjectInitialisation


class ProjectManager:
    """Manages project-level operations including initialization and setup."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the Project Manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Initialize project initialization utility
        self.project_initialisation = ProjectInitialisation(self.config, self.logger)
    
    def setup_project(self) -> Path:
        """
        Set up the project directory structure.
        
        Returns:
            Path to the project directory
        """
        self.logger.info(f"Setting up project for domain: {self.domain_name}")
        
        project_dir = self.project_initialisation.setup_project()
        
        self.logger.info(f"Project directory created at: {project_dir}")
        self.logger.info(f"Shapefiles directories created")
        
        return project_dir
    
    def create_pour_point(self) -> Optional[Path]:
        """
        Create pour point shapefile from coordinates if needed.
        
        Returns:
            Path to created shapefile or None if using existing shapefile
        """
        if self.config.get('POUR_POINT_COORDS', 'default').lower() == 'default':
            self.logger.info("Using user-provided pour point shapefile")
            return None
        
        output_file = self.project_initialisation.create_pourPoint()
        
        if output_file:
            self.logger.info(f"Pour point shapefile created successfully: {output_file}")
        else:
            self.logger.error("Failed to create pour point shapefile")
        
        return output_file
    
    def validate_project_structure(self) -> bool:
        """
        Validate that the project structure is properly set up.
        
        Returns:
            True if project structure is valid, False otherwise
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
        
        Returns:
            Dictionary containing project information
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