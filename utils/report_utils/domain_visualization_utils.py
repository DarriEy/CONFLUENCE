# In utils/report_utils/domain_visualization_utils.py

from pathlib import Path
import logging
from typing import Optional


class DomainVisualizer:
    """Handles domain-related visualizations."""
    
    def __init__(self, config: dict, logger: logging.Logger, reporter):
        self.config = config
        self.logger = logger
        self.reporter = reporter
    
    def plot_domain(self) -> Optional[Path]:
        """Plot the domain visualization."""
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, no domain to plot")
            return None
        
        self.logger.info("Creating domain visualization...")
        domain_plot = self.reporter.plot_domain()
        if domain_plot:
            self.logger.info(f"Domain visualization created: {domain_plot}")
        else:
            self.logger.warning("Could not create domain visualization")
        return domain_plot
    
    def plot_discretized_domain(self) -> Optional[Path]:
        """Plot the discretized domain visualization."""
        if self.config.get('SPATIAL_MODE') == 'Point':
            self.logger.info("Spatial mode: Point simulations, discretisation not required")
            return None
        
        discretization_method = self.config.get('DOMAIN_DISCRETIZATION')
        self.logger.info("Creating discretization visualization...")
        discretization_plot = self.reporter.plot_discretized_domain(discretization_method)
        if discretization_plot:
            self.logger.info(f"Discretization visualization created: {discretization_plot}")
        else:
            self.logger.warning("Could not create discretization visualization")
        return discretization_plot