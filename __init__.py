"""
SYMFLUENCE - SYnergistic Modelling Framework for Linking and Unifying
Earth-system Nexii for Computational Exploration

A comprehensive hydrological modeling platform for watershed analysis.

Note: This package was previously known as CONFLUENCE. For backward compatibility,
the old name is still supported but deprecated.
"""
from symfluence_version import __version__

__author__ = "Darri Eythorsson"
__email__ = "darri.eythorsson@ucalgary.ca"

# Main API imports for convenience
from utils.project.project_manager import ProjectManager
from utils.project.workflow_orchestrator import WorkflowOrchestrator
from utils.cli.cli_argument_manager import CLIArgumentManager

__all__ = [
    "__version__",
    "ProjectManager", 
    "WorkflowOrchestrator", 
    "CLIArgumentManager"
]

# Backward compatibility warning
import warnings
warnings.warn(
    "The 'confluence' package name is deprecated. Please update your imports to use 'symfluence'.",
    DeprecationWarning,
    stacklevel=2
)
