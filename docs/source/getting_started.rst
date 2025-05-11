Getting Started
===============

This guide will walk you through setting up your first hydrological model with CONFLUENCE.

Quick Start
-----------

1. **Prepare your configuration file**

   Copy the template configuration and modify it for your domain:

   .. code-block:: bash

      cp 0_config_files/config_template.yaml config_active.yaml

2. **Define your study area**

   Specify your watershed using either:
   
   - Pour point coordinates
   - Bounding box
   - Existing shapefile

   .. code-block:: yaml

      # Using pour point coordinates
      POUR_POINT_COORDS: 62.09444/136.27223
      
      # Or using bounding box (lat_max/lon_min/lat_min/lon_max)
      BOUNDING_BOX_COORDS: 83.0/-169.0/7.0/-52.0

3. **Run the workflow**

   .. code-block:: bash

      python CONFLUENCE.py --config config_active.yaml

Example: North America Domain
-----------------------------

Here's a complete example using the provided North America configuration:

.. literalinclude:: ../../0_config_files/config_North_America.yaml
   :language: yaml
   :caption: North America Configuration Example
   :lines: 1-50

Running the Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python CONFLUENCE.py --config 0_config_files/config_North_America.yaml

This will:

1. Define the watershed domain
2. Download and preprocess forcing data
3. Set up the SUMMA model
4. Run hydrological simulations
5. Route streamflow with mizuRoute
6. Generate visualization outputs

Understanding the Workflow
--------------------------

CONFLUENCE follows these main steps:

1. **Domain Definition**: Delineate or subset your watershed
2. **Data Acquisition**: Download forcing data (ERA5, etc.)
3. **Preprocessing**: Prepare data for modeling
4. **Model Setup**: Configure hydrological models
5. **Simulation**: Run the models
6. **Postprocessing**: Analyze and visualize results

Next Steps
----------

- Explore different :doc:`configuration` options
- Learn about available :doc:`workflows`
- Check out more :doc:`examples`
