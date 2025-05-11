Examples
========

This section provides practical examples of using CONFLUENCE for various hydrological modeling tasks. The examples are based on the Jupyter notebooks available in the ``jupyter_notebooks/`` directory of the repository.

.. note::
   All notebooks can be found in the ``jupyter_notebooks/`` folder of the CONFLUENCE repository. 
   You can run them interactively to follow along with the tutorials.

Lumped Basin Workflow (Bow River at Banff)
------------------------------------------

This example demonstrates the complete CONFLUENCE workflow for a lumped basin model.

**Notebook**: ``01_lumped_basin_workflow.ipynb``

Key Steps
~~~~~~~~~

1. **Project Setup and Configuration**:

   .. code-block:: python

      from CONFLUENCE import CONFLUENCE
      
      # Initialize with configuration
      confluence = CONFLUENCE('config_active.yaml')
      
      # Setup project directories
      confluence.setup_project()

2. **Create Pour Point**:

   .. code-block:: python

      # Create pour point shapefile from coordinates
      confluence.create_pourPoint()
      
      # Coordinates from config: 51.4215/-115.3593 (Bow River at Banff)

3. **Delineate Lumped Watershed**:

   .. code-block:: python

      # Delineate using lumped method
      confluence.define_domain()
      
      # Create single HRU for lumped model
      confluence.discretize_domain()

4. **Run Complete Workflow**:

   .. code-block:: python

      # Process observed data
      confluence.process_observed_data()
      
      # Acquire forcing data
      confluence.acquire_forcings()
      
      # Preprocess data
      confluence.model_agnostic_pre_processing()
      confluence.model_specific_pre_processing()
      
      # Run model
      confluence.run_models()
      
      # Visualize results
      confluence.visualise_model_output()

Distributed Basin Workflow
--------------------------

This notebook demonstrates distributed modeling with multiple GRUs (Grouped Response Units).

**Notebook**: ``02_semi_distributed_basin_workflow.ipynb``

Key Differences from Lumped Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Configuration changes for distributed model
   DOMAIN_NAME: 'Bow_at_Banff_distributed'
   DOMAIN_DEFINITION_METHOD: 'delineate'  # Changed from 'lumped'
   STREAM_THRESHOLD: 5000                 # Creates more sub-basins
   SPATIAL_MODE: 'Distributed'            # Changed from 'Lumped'

Delineation with Stream Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Delineate distributed watershed
   confluence.define_domain()
   
   # This creates:
   # - Multiple sub-basins (GRUs)
   # - River network connecting them
   # - Routing topology for mizuRoute

Visualization of Results
~~~~~~~~~~~~~~~~~~~~~~~~

The notebook includes code to compare lumped vs distributed domains:

.. code-block:: python

   # Load and visualize basins and river network
   basins = gpd.read_file(basin_files[0])
   rivers = gpd.read_file(network_files[0])
   
   # Plot with different colors for each GRU
   basins.plot(ax=ax, column='GRU_ID', cmap='viridis')
   rivers.plot(ax=ax, color='blue', linewidth=2)

Regional Domain Workflow (Iceland)
----------------------------------

This example shows how to model an entire region with multiple independent watersheds.

**Notebook**: ``04_regional_domain_workflow.ipynb``

Regional Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Key settings for regional modeling
   DOMAIN_NAME: 'Iceland'
   BOUNDING_BOX_COORDS: 68.0/-26.0/62.5/-11.0
   DELINEATE_BY_POURPOINT: False          # Full region, not single outlet
   DELINEATE_COASTAL_WATERSHEDS: True     # Include coastal drainage
   DOMAIN_DEFINITION_METHOD: 'delineate'

Regional Delineation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Delineate entire region
   confluence.define_domain()
   
   # This creates multiple independent drainage basins
   # including watersheds that drain directly to ocean

Analysis of Regional Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The notebook includes analysis code for regional statistics:

.. code-block:: python

   # Analyze watershed characteristics
   areas_km2 = basins.geometry.area / 1e6
   
   # Create summary statistics
   stats_data = {
       'Total Watersheds': len(basins),
       'Total Area (km²)': f"{areas_km2.sum():,.0f}",
       'Mean Area (km²)': f"{areas_km2.mean():.1f}",
       'Largest (km²)': f"{areas_km2.max():.1f}"
   }

Interactive Notebooks
---------------------

The CONFLUENCE repository includes several Jupyter notebooks for hands-on learning:

1. **01_lumped_basin_workflow.ipynb**: Complete lumped model tutorial
2. **02_semi_distributed_basin_workflow.ipynb**: Distributed modeling with GRUs
3. **03_subset_from_geofabric.ipynb**: Using existing geofabric data
4. **04_regional_domain_workflow.ipynb**: Regional-scale modeling
5. **05_point_simulation_single.ipynb**: Point-scale simulations
6. **06_point_simulation_batch.ipynb**: Batch point simulations
7. **07_ensemble_project.ipynb**: Ensemble modeling techniques

Running the Notebooks
~~~~~~~~~~~~~~~~~~~~~

1. Set up your environment:

   .. code-block:: bash

      # Navigate to notebooks directory
      cd jupyter_notebooks/
      
      # Start Jupyter
      jupyter notebook

2. Open any notebook and run cells sequentially
3. Modify configurations to experiment with different settings

Key Configuration Examples
--------------------------

The notebooks demonstrate various configuration patterns:

Lumped Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff_lumped"
   POUR_POINT_COORDS: 51.4215/-115.3593
   DOMAIN_DEFINITION_METHOD: lumped
   LUMPED_WATERSHED_METHOD: pysheds
   SPATIAL_MODE: Lumped

Distributed Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff_distributed"
   DOMAIN_DEFINITION_METHOD: delineate
   STREAM_THRESHOLD: 5000
   DOMAIN_DISCRETIZATION: GRUs
   SPATIAL_MODE: Distributed

Regional Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   DOMAIN_NAME: "Iceland"
   BOUNDING_BOX_COORDS: 68.0/-26.0/62.5/-11.0
   DELINEATE_BY_POURPOINT: False
   DELINEATE_COASTAL_WATERSHEDS: True

Point Simulation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   DOMAIN_NAME: "HJ_Andrews_point"
   EXPERIMENT_ID: "point_sim"
   SPATIAL_MODE: Point
   DATA_ACQUIRE: supplied

Common Tasks and Solutions
--------------------------

Creating Custom Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the lumped basin notebook:

.. code-block:: python

   # Visualize pour point with basemap
   import contextily as cx
   
   gdf = gpd.read_file(pour_point_path)
   gdf_web = gdf.to_crs(epsg=3857)
   
   fig, ax = plt.subplots(figsize=(12, 10))
   gdf_web.plot(ax=ax, color='red', markersize=200)
   cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

Comparing Model Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the distributed basin notebook:

.. code-block:: python

   # Create comparison visualization
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
   
   # Lumped model
   lumped_basin.plot(ax=ax1, color='lightblue')
   ax1.set_title('Lumped Model (1 Unit)')
   
   # Distributed model
   distributed_basins.plot(ax=ax2, column='GRU_ID', cmap='viridis')
   ax2.set_title(f'Distributed Model ({len(basins)} GRUs)')

Handling Large Domains
~~~~~~~~~~~~~~~~~~~~~~

From the regional modeling notebook:

.. code-block:: python

   # Process data in chunks for large domains
   config['MPI_PROCESSES'] = 40  # Use parallel processing
   config['SETTINGS_SUMMA_GRU_PER_JOB'] = 10  # Batch GRUs

Best Practices from Examples
----------------------------

The notebooks demonstrate several best practices:

1. **Incremental Development**: Start with lumped, then distributed
2. **Visualization at Each Step**: Always plot domains and results
3. **Configuration Management**: Use YAML files for reproducibility
4. **Error Checking**: Verify outputs before proceeding

.. code-block:: python

   # Example of error checking from notebooks
   if pour_point_path.exists():
       print("✓ Pour point created successfully")
   else:
       print("Error: Pour point not created")

Learn More
----------

- Full notebook code: `jupyter_notebooks/ <https://github.com/DarriEy/CONFLUENCE/tree/main/jupyter_notebooks>`_
- Example configurations: `0_config_files/ <https://github.com/DarriEy/CONFLUENCE/tree/main/0_config_files>`_
- Tutorial videos: Coming soon

Next Steps
----------

1. Run through ``01_lumped_basin_workflow.ipynb`` for a complete introduction
2. Experiment with distributed modeling in ``02_semi_distributed_basin_workflow.ipynb``
3. Try regional modeling with ``04_regional_domain_workflow.ipynb``
4. Explore advanced features in the other notebooks
