Getting Started
===============

This guide walks you through setting up and running your first SYMFLUENCE workflow — from creating a configuration file to executing a full hydrological simulation.

---

Quick Start
-----------

1. **Create your configuration**

   Copy and edit the provided template:

   .. code-block:: bash

      cp 0_config_files/config_template.yaml my_project.yaml

2. **Define your study domain**

   You can specify:
   - A pour point (latitude/longitude)
   - A bounding box
   - An existing shapefile

   Example (YAML):

   .. code-block:: yaml

      # Pour point coordinates
      POUR_POINT_COORDS: 51.1722/-115.5717

      # Or bounding box (lat_max/lon_min/lat_min/lon_max)
      BOUNDING_BOX_COORDS: 52.0/-116.0/50.0/-114.0

3. **Run the workflow**

   .. code-block:: bash

      ./symfluence --config my_project.yaml --setup_project
      ./symfluence --config my_project.yaml

   **Note:** For backward compatibility, ``./confluence`` still works but will be deprecated.

---

Example: Bow River Watershed
----------------------------

.. literalinclude:: ../../0_config_files/config_template.yaml
   :language: yaml
   :caption: Example configuration (excerpt)
   :lines: 1-40

This example performs:
1. Domain delineation and forcing data setup  
2. Model configuration (e.g., SUMMA, FUSE, NextGen, GR4J, LSTM)  
3. Simulation execution  
4. Routing using mizuRoute  
5. Output evaluation and visualization

---

Understanding the Workflow
--------------------------

Each SYMFLUENCE run follows a structured pipeline:

1. **Domain Definition** — delineate watershed or region  
2. **Data Acquisition** — retrieve and preprocess forcing datasets (ERA5, Daymet, etc.)  
3. **Model Setup** — configure supported models  
4. **Simulation** — execute model runs  
5. **Routing & Evaluation** — route flows and compute diagnostics  
6. **Reporting** — generate plots, metrics, and summaries  

Each step can be called individually using CLI options such as ``--setup_project`` or ``--calibrate_model``.

---

Next Steps
----------

- Explore the :doc:`configuration` structure in detail  
- Try the progressive :doc:`examples` for advanced applications  
- Visit :doc:`troubleshooting` for setup or runtime guidance
