Configuration Guide
===================

CONFLUENCE uses YAML configuration files to control all aspects of the modeling workflow.

Configuration Structure
-----------------------

The configuration file is organized into several main sections:

Global Settings
~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Main paths
   CONFLUENCE_DATA_DIR: "/path/to/data"
   CONFLUENCE_CODE_DIR: "/path/to/code"
   
   # Experiment settings
   DOMAIN_NAME: "your_domain"
   EXPERIMENT_ID: "run_1"
   
   # Computational settings
   MPI_PROCESSES: 40
   FORCE_RUN_ALL_STEPS: False

Geospatial Settings
~~~~~~~~~~~~~~~~~~~

Define your watershed domain and discretization:

.. code-block:: yaml

   # Coordinate settings
   POUR_POINT_COORDS: lat/lon
   BOUNDING_BOX_COORDS: lat_max/lon_min/lat_min/lon_max
   
   # Domain representation
   DOMAIN_DEFINITION_METHOD: delineate  # subset, delineate, or lumped
   GEOFABRIC_TYPE: TDX                  # Merit, TDX, or NWS
   STREAM_THRESHOLD: 7500               # Flow accumulation threshold
   
   # Discretization
   DOMAIN_DISCRETIZATION: GRUs          # elevation, soilclass, landclass, radiation, GRUs
   ELEVATION_BAND_SIZE: 400             # meters
   MIN_HRU_SIZE: 5                      # km²

Model Settings
~~~~~~~~~~~~~~

Configure hydrological and routing models:

.. code-block:: yaml

   # Model selection
   HYDROLOGICAL_MODEL: SUMMA
   ROUTING_MODEL: mizuRoute
   
   # SUMMA settings
   SETTINGS_SUMMA_CONNECT_HRUS: yes
   SETTINGS_SUMMA_TRIALPARAM_N: 1
   
   # mizuRoute settings
   SETTINGS_MIZU_WITHIN_BASIN: 0
   SETTINGS_MIZU_ROUTING_DT: 3600

Data and Forcings
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Forcing data
   FORCING_DATASET: "ERA5"
   FORCING_START_YEAR: "2018"
   FORCING_END_YEAR: "2018"
   FORCING_VARIABLES: longitude,latitude,time,LWRadAtm,SWRadAtm,pptrate,airpres,airtemp,spechum,windspd
   
   # Data acquisition
   DATA_ACQUIRE: HPC        # HPC or supplied
   FORCING_TIME_STEP_SIZE: 3600  # seconds

Optimization and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Optimization settings
   CALIBRATION_PERIOD: 2018-01-01, 2018-06-30
   EVALUATION_PERIOD: 2018-07-01, 2018-12-31
   OPTIMIZATION_ALGORITHM: NSGA-II
   OPTIMIZATION_METRIC: KGE
   
   # Parameters to calibrate
   PARAMS_TO_CALIBRATE: newSnowDenMin,newSnowDenMultTemp,Fcapil,k_snow

Advanced Configuration
----------------------

Custom Paths
~~~~~~~~~~~~

Override default paths for specific components:

.. code-block:: yaml

   # Custom installation paths
   SUMMA_INSTALL_PATH: /custom/path/to/summa
   MIZUROUTE_INSTALL_PATH: /custom/path/to/mizuroute
   
   # Custom output paths
   EXPERIMENT_OUTPUT_SUMMA: /custom/output/summa
   EXPERIMENT_OUTPUT_MIZUROUTE: /custom/output/mizuroute

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Configure parallel execution:

.. code-block:: yaml

   # Parallel SUMMA settings
   SETTINGS_SUMMA_USE_PARALLEL_SUMMA: True
   SETTINGS_SUMMA_CPUS_PER_TASK: 32
   SETTINGS_SUMMA_GRU_PER_JOB: 10
   SETTINGS_SUMMA_TIME_LIMIT: 01:00:00

Configuration Best Practices
----------------------------

1. **Version Control**: Keep configuration files in version control but exclude `config_active.yaml`
2. **Documentation**: Comment your configuration files extensively
3. **Validation**: Use the configuration validator before running
4. **Templates**: Start from templates for new projects
5. **Backup**: Save successful configurations for reproducibility

Example Configurations
----------------------

CONFLUENCE provides several example configurations:

- `config_North_America.yaml`: Large domain example
- `config_template.yaml`: Basic template
- Additional examples in the `examples/` directory

See Also
--------

- :doc:`workflows` for workflow-specific configuration
- :doc:`examples` for complete configuration examples
- :doc:`api` for programmatic configuration
