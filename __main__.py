"""
Entry point for symfluence command-line interface.
This allows the package to be run with: python -m symfluence

Note: For backward compatibility, 'python -m confluence' is also supported
but will show a deprecation warning.
"""

from symfluence import main

if __name__ == "__main__":
    main()
