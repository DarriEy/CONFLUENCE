# src/symfluence/__init__.py
try:
    from symfluence_version import __version__
except Exception:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("symfluence")
    except PackageNotFoundError:
        __version__ = "0.0.0"
