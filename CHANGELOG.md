# Changelog

All notable changes to SYMFLUENCE are documented here.

---

## [0.5.2] - 2025-11-12

### Major: Formal Initial Release with End-to-End CI Validation

**This release marks the first fully reproducible SYMFLUENCE workflow with continuous integration.**

### Added
- **End-to-End CI Pipeline (Example Notebook 2a Equivalent)**  
  Integrated a comprehensive GitHub Actions workflow that builds, validates, and runs SYMFLUENCE automatically on every commit to `main`.  
  - Compiles all hydrologic model dependencies (TauDEM, mizuRoute, FUSE, NGEN).  
  - Validates MPI, NetCDF, GDAL, and HDF5 environments.  
  - Executes key steps (`setup_project`, `create_pour_point`, `define_domain`, `discretize_domain`, `model_agnostic_preprocessing`, `run_model`, `calibrate_model`, `run_benchmarking`).  
  - Confirms reproducible outputs under `SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff`.  
  - Runs both wrapper (`./symfluence`) and direct Python entrypoints equivalently.  

### Changed
- Updated `external_tools_config.py` to include automatic path resolution for TauDEM binaries (e.g., `moveoutletstostrms → moveoutletstostreams`).  
- Expanded logging and run summaries for CI visibility.  
- Protected `main` branch to require successful CI validation before merge.  

### Notes
This release formalizes SYMFLUENCE’s **reproducibility framework**, guaranteeing that all supported workflows can be rebuilt and validated automatically on clean systems.

---

## [0.5.0] - 2025-01-09

### Major: CONFLUENCE → SYMFLUENCE Rebranding

**This is the rebranding release.** The project is now SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration).

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: github.com/DarriEy/CONFLUENCE → github.com/DarriEy/SYMFLUENCE

### Deprecated
- All CONFLUENCE naming (will be removed in v1.0.0)
- Legacy names still work but show warnings

### Migration
```bash
# Update command
./confluence --install  # old (still works)
./symfluence --install  # new

# Update imports
from CONFLUENCE import CONFLUENCE  # old (still works)
from symfluence import SYMFLUENCE  # new

# Update config
CONFLUENCE_DATA_DIR → SYMFLUENCE_DATA_DIR
```

---

## Links
- PyPI: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- Docs: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- GitHub: [github.com/DarriEy/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)
