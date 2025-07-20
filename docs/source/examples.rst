Examples
========

This section provides comprehensive examples demonstrating CONFLUENCE applications across the complete spectrum of hydrological modeling scales, from single-point simulations to continental-scale modeling and large-sample comparative studies. The examples are organized into two complementary series: spatial scaling demonstrations and large-sample methodological applications.

.. note::
   All notebooks can be found in the ``jupyter_notebooks/`` folder of the CONFLUENCE repository. 
   You can run them interactively to follow along with the tutorials.

Tutorial Series Overview
------------------------

CONFLUENCE examples showcase the framework's seamless scalability across different spatial scales and analytical approaches. Each tutorial builds systematically on previous concepts while introducing new spatial scales or methodological approaches.

Spatial Scaling Series: Point to Continental Applications
=========================================================

The spatial scaling series demonstrates CONFLUENCE's capability to handle hydrological modeling across the complete spatial hierarchy, from detailed process validation at individual points to comprehensive continental water resources assessment.

Point Scale Validation (Tutorial 01)
-------------------------------------

Point-scale simulations focus on detailed process validation at specific locations such as flux towers and weather stations, emphasizing vertical process representation without spatial discretization.

Tutorial 01a: SNOTEL Snow and Soil Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``01a_point_scale_snotel.ipynb``

This tutorial demonstrates point-scale snow and soil moisture validation using Paradise SNOTEL station data.

**Configuration Setup**:

.. code-block:: python

   from CONFLUENCE import CONFLUENCE
   import yaml
   
   # Load point-scale configuration template
   with open('config_point_template.yaml', 'r') as f:
       config_dict = yaml.safe_load(f)
   
   # Update for SNOTEL point modeling
   config_updates = {
       'DOMAIN_NAME': 'paradise',
       'POUR_POINT_COORDS': '46.78/-121.75',  # Paradise SNOTEL
       'DOMAIN_DEFINITION_METHOD': 'point',
       'DOWNLOAD_SNOTEL': 'True',
       'EXPERIMENT_TIME_START': '1981-01-01 01:00',
       'EXPERIMENT_TIME_END': '2019-12-31 23:00'
   }

**Key Workflow Steps**:

.. code-block:: python

   # Initialize CONFLUENCE
   confluence = CONFLUENCE(config_path)
   
   # Setup project structure
   project_dir = confluence.managers['project'].setup_project()
   confluence.managers['project'].create_pour_point()
   
   # Execute point-scale workflow
   confluence.managers['data'].process_observed_data()
   confluence.managers['data'].run_model_agnostic_preprocessing()
   confluence.managers['model'].preprocess_models()
   confluence.managers['model'].run_models()

**Performance Evaluation**:

.. code-block:: python

   # Snow Water Equivalent evaluation
   rmse = np.sqrt(((obs_valid - sim_valid) ** 2).mean())
   nse = 1 - ((obs_valid - sim_valid) ** 2).sum() / ((obs_valid - obs_valid.mean()) ** 2).sum()
   kge = 1 - np.sqrt((kge_corr - 1)**2 + (kge_bias - 1)**2 + (kge_var - 1)**2)

Tutorial 01b: FLUXNET Energy Balance Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``01b_point_scale_fluxnet.ipynb``

This tutorial focuses on energy balance validation using FLUXNET eddy covariance tower data for evapotranspiration processes.

Basin-Scale Modeling Progression (Tutorials 02a-02c)
----------------------------------------------------

The Bow River at Banff case study demonstrates the complete progression of basin-scale modeling approaches within a consistent 2,600 km² watershed framework.

Tutorial 02a: Lumped Basin Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``02a_basin_lumped.ipynb``

This tutorial establishes lumped modeling fundamentals using traditional single-unit representation of the entire watershed.

**Lumped Basin Configuration**:

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff_lumped"
   POUR_POINT_COORDS: "51.1722/-115.5717"  # Banff gauging station
   DOMAIN_DEFINITION_METHOD: "lumped"       # Single computational unit
   DOMAIN_DISCRETIZATION: "GRUS"            # Single HRU for entire watershed
   STATION_ID: "05BB001"                     # WSC streamflow station
   DOWNLOAD_WSC_DATA: True

**Workflow Execution**:

.. code-block:: python

   # Basin-scale setup
   confluence = CONFLUENCE(temp_config_path)
   
   # Execute lumped watershed workflow
   confluence.managers['domain'].define_domain()      # Watershed delineation
   confluence.managers['domain'].discretize_domain()  # Single HRU creation
   confluence.managers['data'].process_observed_data()  # Streamflow data
   confluence.managers['model'].run_models()          # SUMMA + mizuRoute

**Streamflow Performance Assessment**:

.. code-block:: python

   # Load observed and simulated streamflow
   obs_daily = obs_df['discharge_cms'].resample('D').mean()
   sim_daily = sim_df.resample('D').mean()
   
   # Calculate performance metrics
   nse = 1 - ((obs_valid - sim_valid) ** 2).sum() / ((obs_valid - obs_valid.mean()) ** 2).sum()
   kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

Tutorial 02b: Semi-Distributed Basin Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``02b_basin_semi_distributed.ipynb``

This tutorial advances to semi-distributed modeling with approximately 15 sub-basin units that capture spatial heterogeneity while maintaining computational efficiency.

**Semi-Distributed Configuration Changes**:

.. code-block:: yaml

   DOMAIN_NAME: "Bow_at_Banff_distributed"
   DOMAIN_DEFINITION_METHOD: "delineate"    # Changed from "lumped"
   STREAM_THRESHOLD: 5000                   # Creates multiple sub-basins
   DOMAIN_DISCRETIZATION: "GRUs"            # Multiple computational units

Tutorial 02c: Elevation-Based HRU Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``02c_basin_elevation_bands.ipynb``

This tutorial culminates with elevation-based HRU discretization that subdivides each sub-basin into elevation bands, creating detailed spatial representation.

**Elevation Band Configuration**:

.. code-block:: yaml

   DOMAIN_DISCRETIZATION: "elevation"       # Elevation-based HRUs
   ELEVATION_BAND_SIZE: 200                 # 200m elevation bands
   MIN_HRU_SIZE: 4                          # Minimum HRU size in km²

Regional and Continental Applications (Tutorials 03a-03b)
---------------------------------------------------------

Tutorial 03a: Regional Domain Modeling (Iceland)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``03a_domain_regional.ipynb``

This tutorial demonstrates regional domain modeling using Iceland as a comprehensive case study for national-scale water resources assessment across multiple independent drainage systems.

**Regional Configuration**:

.. code-block:: yaml

   DOMAIN_NAME: "Iceland"
   BOUNDING_BOX_COORDS: "65.5/-25.0/63.0/-13.0"  # Iceland boundaries
   DOMAIN_DEFINITION_METHOD: "delineate"
   DELINEATE_BY_POURPOINT: False              # Full region, not single outlet
   DELINEATE_COASTAL_WATERSHEDS: True         # Include coastal drainage

**Regional Workflow**:

.. code-block:: python

   # Regional multi-watershed delineation
   confluence.managers['domain'].define_domain()      # Multiple independent watersheds
   confluence.managers['domain'].discretize_domain()  # Regional computational units
   
   # Regional analysis
   basins_gdf = gpd.read_file(basin_files[0])
   total_area = basins_gdf.geometry.area.sum() / 1e6  # Total coverage in km²
   watershed_count = len(basins_gdf)                   # Number of independent basins

Tutorial 03b: Continental Domain Scale (North America)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``03b_domain_continental.ipynb``

This tutorial extends to continental-scale modeling across North America, representing the ultimate computational challenge requiring high-performance computing infrastructure.

Large-Sample Methodological Series
==================================

The large-sample series demonstrates CONFLUENCE's power for comparative hydrology and systematic model evaluation across diverse environmental conditions.

Tutorial 04a: Multi-Site Energy Balance Validation (FLUXNET)
------------------------------------------------------------

**Notebook**: ``04a_large_sample_fluxnet.ipynb``

This tutorial demonstrates systematic energy balance validation across the global network of eddy covariance towers, establishing large-sample methodology for land-atmosphere interaction modeling.

**Large Sample Configuration**:

.. code-block:: python

   # Load FLUXNET site database
   fluxnet_df = pd.read_csv('fluxnet_towers.csv')
   
   # Automated configuration generation for multiple sites
   for _, site in fluxnet_df.iterrows():
       site_config = base_config.copy()
       site_config.update({
           'DOMAIN_NAME': site['DOMAIN_NAME'],
           'POUR_POINT_COORDS': site['POUR_POINT_COORDS'],
           'DOMAIN_DEFINITION_METHOD': 'point'
       })

**Batch Processing Execution**:

.. code-block:: python

   # Execute across multiple FLUXNET sites
   script_success = run_fluxnet_script_from_notebook()
   
   # Multi-site results aggregation
   et_results, processing_summary = extract_et_results_from_domains(completed_domains)
   fluxnet_obs, obs_summary = load_fluxnet_observations()

**Comparative Analysis**:

.. code-block:: python

   # Performance across environmental gradients
   climate_stats = {}
   for site in common_sites:
       climate = site['climate']
       if climate not in climate_stats:
           climate_stats[climate] = {'correlations': [], 'rmses': [], 'biases': []}
       climate_stats[climate]['correlations'].append(site['correlation'])

Tutorial 04b: Snow Process Validation (NorSWE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``04b_large_sample_norswe.ipynb``

This tutorial focuses on systematic snow hydrology validation across northern hemisphere environmental gradients using the NorSWE snow observation network.

Tutorial 04c: Distributed Basin Comparison (CAMELS-Spat)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook**: ``04c_large_sample_camels.ipynb**

This tutorial showcases large-sample distributed watershed modeling across hundreds of basins throughout the continental United States using CAMELS-Spat applications.

Workflow Orchestration and Best Practices
==========================================

All tutorials demonstrate key CONFLUENCE principles:

Configuration Management
------------------------

.. code-block:: python

   # Template-based configuration
   with open(config_template_path, 'r') as f:
       config_dict = yaml.safe_load(f)
   
   # Site-specific updates
   config_dict.update(config_updates)
   
   # Save and initialize
   with open(temp_config_path, 'w') as f:
       yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

Workflow Orchestration
----------------------

.. code-block:: python

   # Initialize CONFLUENCE system
   confluence = CONFLUENCE(config_path)
   
   # Execute complete workflow
   confluence.managers['project'].setup_project()
   confluence.managers['domain'].define_domain()
   confluence.managers['domain'].discretize_domain()
   confluence.managers['data'].run_model_agnostic_preprocessing()
   confluence.managers['model'].preprocess_models()
   confluence.managers['model'].run_models()

Visualization and Analysis
--------------------------

.. code-block:: python

   # Standard visualization patterns
   fig, axes = plt.subplots(2, 2, figsize=(16, 12))
   
   # Time series comparison
   ax1.plot(obs_valid.index, obs_valid.values, 'b-', label='Observed', linewidth=1.5)
   ax1.plot(sim_valid.index, sim_valid.values, 'r-', label='Simulated', linewidth=1.5)
   
   # Performance metrics display
   metrics_text = f'NSE: {nse:.3f}\nKGE: {kge:.3f}\nBias: {pbias:+.1f}%'
   ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes)

Common Configuration Patterns
=============================

Point Scale Configuration
-------------------------

.. code-block:: yaml

   DOMAIN_NAME: "point_site"
   DOMAIN_DEFINITION_METHOD: "point"
   BOUNDING_BOX_COORDS: "lat_max/lon_min/lat_min/lon_max"
   POUR_POINT_COORDS: "lat/lon"

Basin Scale Configuration
------------------------

.. code-block:: yaml

   DOMAIN_NAME: "watershed_name"
   POUR_POINT_COORDS: "lat/lon"
   DOMAIN_DEFINITION_METHOD: "lumped"  # or "delineate"
   DOMAIN_DISCRETIZATION: "GRUs"       # or "elevation"
   STATION_ID: "gauge_station_id"

Regional Scale Configuration
---------------------------

.. code-block:: yaml

   DOMAIN_NAME: "region_name"
   BOUNDING_BOX_COORDS: "lat_max/lon_min/lat_min/lon_max"
   DOMAIN_DEFINITION_METHOD: "delineate"
   DELINEATE_BY_POURPOINT: False
   DELINEATE_COASTAL_WATERSHEDS: True

Large Sample Configuration
--------------------------

.. code-block:: yaml

   # Base template for systematic multi-site generation
   DOMAIN_NAME: "template_name"
   EXPERIMENT_ID: "batch_experiment"
   FORCE_RUN_ALL_STEPS: False
   MPI_PROCESSES: 4

Getting Started with Examples
=============================

Prerequisites
-------------

- Complete CONFLUENCE installation with geospatial packages
- Access to relevant datasets (SNOTEL, FLUXNET, streamflow data)
- Computational resources appropriate for chosen tutorial scale
- Python environment with Jupyter notebook support

Tutorial Execution Sequence
---------------------------

1. **Start with Point Scale**: Begin with ``01a_point_scale_snotel.ipynb`` to understand fundamental concepts
2. **Progress Through Basin Scale**: Follow tutorials 02a → 02b → 02c for complete basin modeling progression
3. **Advance to Regional Scale**: Explore 03a for regional applications
4. **Master Large Sample Methods**: Use 04a for comparative hydrology approaches

Running the Notebooks
---------------------

.. code-block:: bash

   # Navigate to notebooks directory
   cd jupyter_notebooks/
   
   # Start Jupyter
   jupyter notebook
   
   # Open desired tutorial and run cells sequentially

Best Practices from Examples
============================

The tutorials demonstrate several best practices:

1. **Incremental Complexity**: Start simple (lumped) then advance (distributed)
2. **Systematic Validation**: Always compare simulated vs observed data
3. **Comprehensive Visualization**: Plot domains, forcing, and results
4. **Configuration Management**: Use templates for reproducible science
5. **Error Checking**: Verify outputs at each workflow step

.. code-block:: python

   # Example validation pattern from tutorials
   if output_path.exists():
       print("✓ Output generated successfully")
       # Load and analyze results
   else:
       print("❌ Output generation failed")
       # Troubleshoot workflow

Troubleshooting and Support
===========================

- **Configuration Issues**: Review template files and update paths
- **Data Requirements**: Check tutorial-specific data availability
- **Performance Issues**: Adjust computational settings for your system
- **Workflow Errors**: Examine log files for detailed error information

Learn More
----------

- Full notebook code: `jupyter_notebooks/ <https://github.com/DarriEy/CONFLUENCE/tree/main/jupyter_notebooks>`_
- Configuration templates: `0_config_files/ <https://github.com/DarriEy/CONFLUENCE/tree/main/0_config_files>`_
- Main documentation: :doc:`index`

Next Steps
----------

1. Run ``01a_point_scale_snotel.ipynb`` for comprehensive introduction
2. Progress through basin-scale tutorials (02a-02c) for spatial modeling
3. Explore regional applications with ``03a_domain_regional.ipynb``
4. Master large-sample methods with ``04a_large_sample_fluxnet.ipynb``
5. Adapt examples to your specific research applications