API Reference
=============

Overview
--------
The CONFLUENCE Python API mirrors the CLI and provides programmatic access to the full workflow.
The primary entry point is the ``CONFLUENCE`` class, which coordinates manager components and the
``WorkflowOrchestrator``-driven step sequence.

Quick Start
-----------
.. code-block:: python

   from CONFLUENCE import CONFLUENCE

   conf = CONFLUENCE("my_project.yaml")
   conf.run_workflow()  # executes the orchestrated end-to-end pipeline

Managers and Responsibilities
-----------------------------
Internally, CONFLUENCE composes several managers, coordinated by the workflow orchestrator:

- ``project`` — project structure, pour point creation
- ``domain`` — domain definition, discretization
- ``data`` — attributes/forcing acquisition, observed data, model-agnostic preprocessing
- ``model`` — model-specific preprocessing, simulation runs, post-processing
- ``optimization`` — calibration and emulation
- ``analysis`` — benchmarking, decision and sensitivity analyses

Core Methods (by Stage)
-----------------------

Project Setup
~~~~~~~~~~~~~
.. code-block:: python

   conf.setup_project()          # project manager
   conf.create_pour_point()      # project manager

Domain and Data
~~~~~~~~~~~~~~~
.. code-block:: python

   conf.acquire_attributes()     # data manager
   conf.define_domain()          # domain manager
   conf.discretize_domain()      # domain manager

Observed/Forcing Data & Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.process_observed_data()                 # data manager
   conf.acquire_forcings()                      # data manager
   conf.run_model_agnostic_preprocessing()      # data manager

Model Execution and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.preprocess_models()      # model manager
   conf.run_models()             # model manager
   conf.postprocess_results()    # model manager

Optimization and Emulation
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.calibrate_model()        # optimization manager
   conf.run_emulation()          # optimization manager

Analyses (Optional)
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.run_benchmarking()       # analysis manager
   conf.run_decision_analysis()  # analysis manager
   conf.run_sensitivity_analysis()# analysis manager

End-to-End Orchestration
------------------------
The orchestrated pipeline executes steps in order, skipping completed steps unless forced.

.. code-block:: python

   conf.run_workflow(force_run=False)  # force_run=True to recompute all steps

Configuration flags affecting orchestration include:
- ``FORCE_RUN_ALL_STEPS`` (YAML) and ``force_run`` (API) to recompute outputs
- ``STOP_ON_ERROR`` to stop or continue on failure
- ``MPI_PROCESSES`` and model-level parallel settings for scalable runs

Status and Monitoring
---------------------
Programmatically query progress using the orchestrator’s status:

.. code-block:: python

   status = conf.get_workflow_status()
   # returns {total_steps, completed_steps, pending_steps, step_details: [...]}

Logging
-------
Set verbosity in YAML (e.g., ``LOG_LEVEL: INFO``) and inspect logs under
``_workLog_<domain_name>/`` for step-by-step diagnostics.

Extending CONFLUENCE
--------------------
- Add new model adapters under ``utils/models/`` and register them in preprocessing/run steps
- Add optimization strategies under ``optimization`` and expose via configuration
- Create new analyses in ``analysis`` and wire into the orchestrated sequence

References
----------
- :doc:`workflows` — high-level orchestration and usage
- :doc:`configuration` — configuration schema and examples
- :doc:`examples` — runnable tutorials
