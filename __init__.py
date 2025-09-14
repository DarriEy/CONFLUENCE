# confluence/__init__.py
"""
CONFLUENCE - Community Optimization and Numerical Framework for 
Large-domain Understanding of Environmental Networks and Computational Exploration

A comprehensive hydrological modeling platform for watershed analysis.
"""

__version__ = "1.0.0"
__author__ = "Darri Eythorsson"
__email__ = "darri.eythorsson@ucalgary.ca"

# Main API imports for convenience
from .utils.project.project_manager import ProjectManager
from .utils.project.workflow_orchestrator import WorkflowOrchestrator
from .utils.cli.cli_argument_manager import CLIArgumentManager

__all__ = [
    "ProjectManager", 
    "WorkflowOrchestrator", 
    "CLIArgumentManager"
]
