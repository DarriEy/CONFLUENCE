Configuration
=============

Overview
--------
All CONFLUENCE workflows are driven by a single YAML configuration file.  
This file defines domain setup, model selection, data sources, optimization strategy, and output behavior.  
Configurations are modular, validated at runtime, and fully reproducible.

---

Structure
---------
Configurations are organized into logical blocks:

1. **Global and Logging Settings** — Experiment metadata and runtime controls  
2. **Geospatial Definition** — Domain delineation and discretization  
3. **Data and Forcings** — Input datasets and acquisition options  
4. **Model Configuration** — Model selection and parameters  
5. **Routing** — Flow routing with mizuRoute  
6. **Calibration and Optimization** — Parameter estimation and metrics  
7. **Emulation** — Differentiable emulators for large-domain workflows  
8. **Paths and Resources** — Custom paths, file structure, and parallelism

---

Global and Logging Settings
---------------------------
Define experiment identifiers, paths, and computational options.

.. code-block:: yaml

   CONFLUENCE_DATA_DIR: "/path/to/data"
   CONFLUENCE_CODE_DIR: "/path/to/code"
   DOMAIN_NAME: "bow_river"
   EXPERIMENT_ID: "baseline_01"
   EXPERIMENT_TIME_START: "2018-01-01"
   EXPERIMENT_TIME_END: "2019-12-31"
   MPI_PROCESSES: 40
   LOG_LEVEL: INFO
   LOG_TO_FILE: True
   LOG_FORMAT: detailed

---

Geospatial Definition
---------------------
Configure watershed delineation, thresholds, and HRU discretization.

.. code-block:: yaml

   POUR_POINT_COORDS: 51.17/-115.57
   DOMAIN_DEFINITION_METHOD: delineate    # delineate | subset | lumped
   GEOFABRIC_TYPE: TDX                    # TDX | MERIT | NWS
   STREAM_THRESHOLD: 7500
   MULTI_SCALE_THRESHOLDS: [2500, 7500, 15000]
   USE_DROP_ANALYSIS: True
   DROP_ANALYSIS_NUM_THRESHOLDS: 5
   DE LINEATE_COASTAL_WATERSHEDS: False
   DOMAIN_DISCRETIZATION: elevation
   ELEVATION_BAND_SIZE: 400
   MIN_HRU_SIZE: 5
   RADIATION_CLASS_NUMBER: 8
   ASPECT_CLASS_NUMBER: 4

These parameters control how the domain is delineated and discretized before model setup.

---

Data and Forcings
-----------------
Forcing data and meteorological drivers are defined here.

.. code-block:: yaml

   FORCING_DATASET: ERA5
   FORCING_VARIABLES:
     - airtemp
     - windspd
     - pptrate
     - spechum
     - SWRadAtm
     - LWRadAtm
   FORCING_TIME_STEP_SIZE: 3600
   APPLY_LAPSE_RATE: True
   LAPSE_RATE: -6.5
   DATA_ACQUIRE: HPC                     # HPC | supplied | local

Optional extensions:
- **EM-Earth** integration for high-resolution precipitation/temperature downscaling  
- **Supplemental forcing** and derived variables (e.g., PET via Priestley–Taylor)

---

Model Configuration
-------------------
Select hydrologic and routing models, and configure per-model parameters.

.. code-block:: yaml

   HYDROLOGICAL_MODEL: SUMMA            # SUMMA | FUSE | GR | LSTM | NextGen
   ROUTING_MODEL: mizuRoute

### SUMMA
.. code-block:: yaml

   SETTINGS_SUMMA_CONNECT_HRUS: yes
   SETTINGS_SUMMA_TRIALPARAM_N: 1
   SETTINGS_SUMMA_USE_PARALLEL_SUMMA: True
   SETTINGS_SUMMA_CPUS_PER_TASK: 32
   SETTINGS_SUMMA_GRU_PER_JOB: 10

### FUSE
.. code-block:: yaml

   FUSE_SPATIAL_MODE: distributed
   FUSE_DECISION_OPTIONS: default
   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: [alpha, beta, k_storage]

### NextGen
.. code-block:: yaml

   NGEN_BMI_MODULES: [cfe, noah, pet]
   NGEN_NOAH_PARAMS_TO_CALIBRATE: [bexp, dksat, psisat, refkdt]
   NGEN_ACTIVE_CATCHMENT_ID: 1002

### GR4J and LSTM
.. code-block:: yaml

   GR_SPATIAL_MODE: lumped
   FLASH_HIDDEN_SIZE: 256
   FLASH_EPOCHS: 100
   FLASH_USE_ATTENTION: True

---

Routing
-------
mizuRoute parameters for flow routing and network operations.

.. code-block:: yaml

   SETTINGS_MIZU_ROUTING_DT: 3600
   SETTINGS_MIZU_ROUTING_UNITS: seconds
   SETTINGS_MIZU_WITHIN_BASIN: 0
   SETTINGS_MIZU_OUTPUT_FREQ: daily
   SETTINGS_MIZU_OUTPUT_VARS: [Qout, Qsim, storage]

---

Calibration and Optimization
-----------------------------
Define calibration period, optimization algorithms, and objective metrics.

.. code-block:: yaml

   CALIBRATION_PERIOD: "2018-01-01,2018-06-30"
   EVALUATION_PERIOD: "2018-07-01,2018-12-31"
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, newSnowDenMin]
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30

Supported algorithms:
- **DE** – Differential Evolution  
- **DDS** – Dynamically Dimensioned Search  
- **PSO** – Particle Swarm Optimization  
- **NSGA-II** – Multi-objective optimization  

---


Paths and Resources
-------------------
Set paths for custom installations and parallel runtime behavior.

.. code-block:: yaml

   SUMMA_INSTALL_PATH: /opt/summa
   MIZUROUTE_INSTALL_PATH: /opt/mizuroute
   OUTPUT_DIR: /scratch/confluence/output
   SETTINGS_SUMMA_TIME_LIMIT: 02:00:00
   SETTINGS_SUMMA_MEM: 12G

---

Validation and Best Practices
-----------------------------
1. Validate before execution:
   .. code-block:: bash
      ./confluence --validate_config my_project.yaml
2. Comment custom values and rationale within YAML.
3. Version-control all configuration files (except `config_active.yaml`).
4. Use template as baseline and document deviations clearly.

---