Examples
========

Basic Workflow Example
----------------------

This example demonstrates setting up a basic hydrological modeling workflow:

.. code-block:: python

   from CONFLUENCE import CONFLUENCE
   
   # Initialize with configuration
   confluence = CONFLUENCE('config_active.yaml')
   
   # Run the complete workflow
   confluence.run_workflow()

North America Domain Example
----------------------------

Based on your config_North_America.yaml, here's how to set up a North American domain:

.. literalinclude:: ../../0_config_files/config_North_America.yaml
   :language: yaml
   :caption: North America Configuration

[Continue with detailed examples...]
