Workflows
=========

CONFLUENCE provides both automated workflows and step-by-step control for hydrological modeling.

Automated Workflow
------------------

The simplest way to run CONFLUENCE is using the automated workflow:

.. code-block:: python

   from CONFLUENCE import CONFLUENCE
   
   # Initialize and run complete workflow
   confluence = CONFLUENCE('config_active.yaml')
   confluence.run_workflow()

This automatically executes all steps in sequence:

1. Project setup
2. Domain definition
3. Data acquisition
4. Preprocessing
5. Model execution
6. Postprocessing
7. Visualization

Step-by-Step Workflow
---------------------

For more control, execute individual steps:

.. code-block:: python

   from CONFLUENCE import CONFLUENCE
   
   confluence = CONFLUENCE('config_active.yaml')
   
   # Step 1: Setup project structure
   confluence.setup_project()
   
   # Step 2: Define domain
   confluence.create_pourPoint()
   confluence.define_domain()
   confluence.discretize_domain()
   
   # Step 3: Acquire data
   confluence.acquire_attributes()
   confluence.acquire_forcings()
   
   # Step 4: Preprocess data
   confluence.model_agnostic_pre_processing()
   confluence.model_specific_pre_processing()
   
   # Step 5: Run models
   confluence.run_models()
   
   # Step 6: Analyze results
   confluence.run_postprocessing()
   confluence.visualise_model_output()

Parallel Workflows
------------------

CONFLUENCE supports parallel execution for:

- Multiple GRU processing
- Parameter optimization
- Ensemble simulations

.. code-block:: python

   # Configure parallel execution
   config['SETTINGS_SUMMA_USE_PARALLEL_SUMMA'] = True
   config['MPI_PROCESSES'] = 40
   
   # Run parallel optimization
   confluence.run_parallel_optimization()

Custom Workflows
----------------

Create custom workflows by combining components:

.. code-block:: python

   # Example: Focus on optimization
   confluence.run_models()
   confluence.calibrate_model()
   confluence.run_sensitivity_analysis()
   confluence.run_decision_analysis()

Workflow Configuration
----------------------

Control workflow behavior through configuration:

.. code-block:: yaml

   # Skip steps if outputs exist
   FORCE_RUN_ALL_STEPS: False
   
   # Enable specific analyses
   RUN_SENSITIVITY_ANALYSIS: True
   RUN_DECISION_ANALYSIS: True
   RUN_BENCHMARKING: True

Error Handling
--------------

CONFLUENCE provides robust error handling:

.. code-block:: python

   try:
       confluence.run_workflow()
   except Exception as e:
       logger.error(f"Workflow failed: {str(e)}")
       # Check logs for details

Monitoring Progress
-------------------

Track workflow progress through:

- Console output
- Log files in `_workLog_domain_name/`
- Status files in output directories

Next Steps
----------

- Explore specific :doc:`modules/index`
- See :doc:`examples` for complete workflows
- Check :doc:`api` for detailed reference
