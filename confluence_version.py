"""
DEPRECATED: This file is maintained for backward compatibility.
Please use symfluence_version instead.

This will be removed in a future version.
"""
import warnings

warnings.warn(
    "confluence_version is deprecated. Please use symfluence_version instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from symfluence_version import __version__

__all__ = ['__version__']
