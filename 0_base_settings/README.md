# Base Settings

This directory contains the default configuration templates and parameter files for all supported hydrological models in SYMFLUENCE.  
These serve as **clean, validated starting points** for new projects and ensure reproducible, consistent model setup across workflows.

---

## 1. Structure

Each subdirectory includes model-specific base templates:

- **SUMMA/** — Parameter files, decision file, and file manager templates for SUMMA model setup  
  → [CH-Earth SUMMA Repository](https://github.com/CH-Earth/SUMMA)

- **FUSE/** — Configuration templates and parameter definitions for FUSE model applications  
  → [CH-Earth FUSE Repository](https://github.com/CH-Earth/FUSE)

- **mizuRoute/** — Network topology, routing parameter files, and control templates for mizuRoute  
  → [CH-Earth mizuRoute Repository](https://github.com/CH-Earth/mizuRoute)

- **NOAH/** — Parameter files for the NOAH Land Surface Model used in **NextGen-style** workflows.
  → [CH-Earth NOAH Repository](https://github.com/CH-Earth/noah-lsm)

---

## 2. Usage

Base settings are automatically copied into each project during model preprocessing:
```bash
./symfluence --preprocess_models --config my_project.yaml
```

Copied files can then be safely modified without affecting the base templates.  
This ensures:
- Clean defaults for every new project  
- Reproducibility across model setups  
- Isolation between experiments

---

## 3. Customization Guidelines

- **Avoid editing files here directly.**  
  Changes will affect all future projects.  
  Instead, modify the copies within your project directory.

- Use version control to track any changes for transparency.
- When adding support for new models, follow the directory structure and naming conventions used here.

---

## 4. References

For full model documentation and parameter details:
- [SUMMA – Structure for Unifying Multiple Modeling Alternatives](https://github.com/CH-Earth/SUMMA)  
- [FUSE – Framework for Understanding Structural Errors](https://github.com/CH-Earth/FUSE)  
- [mizuRoute – River Network Routing Tool](https://github.com/CH-Earth/mizuRoute)  
- [NOAH – Land Surface Model](https://github.com/CH-Earth/noah-lsm)
