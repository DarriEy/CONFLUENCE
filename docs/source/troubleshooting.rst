Troubleshooting
===============

Overview
--------
This guide outlines how to diagnose and resolve common issues in CONFLUENCE related to configuration, installation, runtime execution, and HPC workflows.

---

Configuration Issues
--------------------
**Validation errors**
- Review reported line numbers and missing keys.
- Ensure YAML syntax uses spaces, not tabs.

**Common causes**
- Incorrect indentation or duplicated keys.
- Invalid parameter names or deprecated fields.
- Missing domain coordinates or file paths.

**Tip:** Always start from `config_template.yaml` and comment custom changes.

---

Installation and Environment
----------------------------
**Environment not found**
- Ensure you installed using the built-in installer:
  .. code-block:: bash

     ./confluence --install
- Activate environment:
  .. code-block:: bash

     source .venv/bin/activate

**Missing libraries**
- Verify GDAL, NetCDF, and HDF5 are installed and accessible.
- Check ``which gdalinfo`` and ``ldd <libnetcdf.so>`` on Linux.

**Version conflicts**
- Use Python 3.11. Other versions are not guaranteed to be supported.
- Reinstall dependencies cleanly:
  .. code-block:: bash

     rm -rf .venv
     ./confluence --install

---

Runtime and Workflow
--------------------
**Model execution stops prematurely**
- Inspect logs in ``_workLog_<domain_name>/``.
- Common issues:
  - Missing forcing data files.
  - Invalid HRU discretization thresholds.
  - Incorrect relative paths in model setup.

**Parallel job hangs**
- Check ``MPI_PROCESSES`` matches available cores.
- Reduce thread counts if memory limits are reached.
- Ensure the cluster scheduler allows requested resources.

**Routing or calibration failure**
- Confirm mizuRoute input files are generated.
- Verify all calibration parameters exist in the model setup.
- Run model once without calibration to isolate the issue.

---

Optimization and Emulation
--------------------------
**Slow convergence or stagnation**
- Reduce parameter bounds or start with fewer variables.
- Use smaller population size for initial testing.
- Consider switching algorithms (DE → DDS).

---

HPC and Cluster Use
-------------------
**Job submission fails**
- Check SLURM script formatting and environment variables.
- Ensure ``--slurm_job`` flag is used for automated submission.

**Missing output**
- Check for out-of-memory (OOM) events in SLURM logs.
- Confirm output directory write permissions.

---

Logging and Debugging
---------------------
All major steps produce detailed logs: