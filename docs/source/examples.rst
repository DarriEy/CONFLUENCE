Examples
========

Overview
--------
This section introduces complete, ready-to-run examples that demonstrate CONFLUENCE’s full range of workflows — from point-scale validation to large-domain and large-sample modeling.  
Each example includes configuration templates, Jupyter notebooks, and batch scripts located in the ``examples/`` directory.

---

Tutorial Structure
------------------
The examples are organized progressively to guide users from simple to advanced workflows:

1. **Point Scale (01)** — Snow and energy balance validation (SNOTEL, FLUXNET)  
2. **Basin Scale (02)** — Bow River case studies (lumped, semi-distributed, elevation-based)  
3. **Regional and Continental (03)** — Iceland and North America workflows  
4. **Large Sample (04)** — Comparative studies (FLUXNET, NorSWE, CAMELS, LamaH, CARAVAN)

Each directory contains a configuration file, notebook, and optional SLURM script.

---

Running the Examples
--------------------
1. Install CONFLUENCE and activate your environment:
   .. code-block:: bash

      ./confluence --install
      source .venv/bin/activate

2. Navigate to the example directory:
   .. code-block:: bash

      cd examples/02_basin_scale/
      jupyter notebook 02b_basin_semi_distributed.ipynb

3. Run the notebook or script as described inside.

---

Learning Path
-------------
- **Start simple:** ``01a_point_scale_snotel.ipynb`` — understand configuration and validation  
- **Progress spatially:** ``02a–02c`` — from lumped to elevation-band modeling  
- **Scale up:** ``03a–03b`` — regional and continental workflows  
- **Generalize:** ``04a–04c`` — multi-site, global datasets and comparative analysis  

---

Best Practices
--------------
- Always validate configuration before execution.  
- Follow the order: setup → run → evaluate.  
- Use logs and plots to verify intermediate outputs.  
- Adapt example configurations for your domain and models.

---

References
----------
- Example notebooks: `jupyter_notebooks/ <https://github.com/DarriEy/CONFLUENCE/tree/main/jupyter_notebooks>`_  
- Configuration templates: `0_config_files/ <https://github.com/DarriEy/CONFLUENCE/tree/main/0_config_files>`_  
- :doc:`configuration` — Configuration reference  