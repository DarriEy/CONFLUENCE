Installation
============

Overview
--------
SYMFLUENCE supports a one-command installer that creates an isolated Python 3.11 environment and installs required packages. HPC environments vary; use the module recipes on your cluster as needed.

Quick Install
-------------
Run the built-in installer from the project root:

.. code-block:: bash

   ./symfluence --install

What this does:
- Creates/updates ``.venv/`` (Python 3.11 recommended)
- Installs Python dependencies with ``pip``
- Reuses the environment on subsequent runs

Backward Compatibility
----------------------
For users transitioning from CONFLUENCE, the old command still works:

.. code-block:: bash

   ./confluence --install

This wrapper will be deprecated in a future release. Please update to ``./symfluence``.

Manual Setup (Optional)
-----------------------
If you prefer to manage the environment yourself:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

System Prerequisites
--------------------
- Build toolchain: GCC/Clang (C/C++), gfortran; CMake ≥ 3.20; MPI (OpenMPI/MPICH)
- Core libs: GDAL, HDF5, NetCDF (C + Fortran), BLAS/LAPACK
- Optional tools: R, CDO (when applicable)

HPC Notes
---------
Use your site's module system, then run the installer:

.. code-block:: bash

   # Example (conceptual): load compiler, MPI, GDAL, NetCDF, Python 3.11
   module load gcc openmpi gdal netcdf-c netcdf-fortran python/3.11
   ./symfluence --install

Verification
------------
.. code-block:: bash

   ./symfluence --help

Next Steps
----------
- :doc:`getting_started` — your first run
- :doc:`configuration` — YAML structure and options
- :doc:`examples` — progressive tutorials
