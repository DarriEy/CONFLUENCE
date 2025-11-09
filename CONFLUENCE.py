#!/usr/bin/env python3
"""
DEPRECATED: This file is maintained for backward compatibility.
Please use symfluence.py instead.

This wrapper imports and re-exports SYMFLUENCE as CONFLUENCE to maintain
backward compatibility with existing code. This will be removed in a future version.
"""
import warnings
import sys

warnings.warn(
    "CONFLUENCE.py is deprecated. Please use symfluence.py instead. "
    "The CONFLUENCE name has been rebranded to SYMFLUENCE.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from symfluence import SYMFLUENCE as CONFLUENCE, main

# Make CONFLUENCE available under its old name
__all__ = ['CONFLUENCE', 'main']

if __name__ == "__main__":
    print("\n⚠️  WARNING: CONFLUENCE.py is deprecated.")
    print("   Please use 'python symfluence.py' or './symfluence' instead.")
    print("   The project has been rebranded to SYMFLUENCE.\n")
    main()
