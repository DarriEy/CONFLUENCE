# Configuration Files

This directory contains the configuration templates that control all aspects of CONFLUENCE workflows.  
Configurations define experiment setup, model selection, data paths, and analysis parameters across the workflow—from conceptualization to evaluation.

---

## 1. Configuration Structure

All configurations are written in **YAML** and follow the logical sequence of the workflow.  
Each section feeds into the next to ensure consistency and reproducibility.

### Sections
- **Global Settings** — Define directories, domain names, and experiment IDs.  
- **Geospatial** — Configure watershed delineation, coordinate systems, and spatial discretization.  
- **Model-Agnostic** — Specify preprocessing, forcing data, and format standardization.  
- **Model-Specific** — Provide setup parameters for SUMMA, FUSE, NEXTGEN, GR, LSTM.  
- **Evaluation** — Define performance metrics, benchmarking methods, and visualization settings.  
- **Optimization** — Configure calibration targets, objective functions, and algorithms.

---

## 2. Working with Configurations

### Creating a New Configuration
Start from the provided template:
```bash
cp config_template.yaml config_active.yaml
```
Edit from top to bottom:
1. Define global paths and domain details.  
2. Set model and calibration parameters.  
3. Verify settings using built-in validation tools.

### Hierarchy and Dependencies
Each section uses values defined in earlier ones.  
This ensures parameter inheritance, consistent file paths, and workflow synchronization.

### Path Management
Default paths align with CONFLUENCE’s directory structure, but you can override them to match existing data or cluster environments.

### Multi-Model Workflows
You can run several models within a single configuration for intercomparison or ensemble experiments.  
Each model block operates independently while sharing the global setup.

---

## 3. Best Practices

### Configuration Management
- Maintain one configuration per experiment or domain.  
- Use descriptive experiment identifiers.  
- Avoid editing active configurations—create copies for new runs.

### Documentation
Comment on any non-default values directly within YAML files.  
This supports reproducibility and provides clarity for future analysis.

---

## 4. Troubleshooting
- Review error messages carefully; they include direct hints for correction.  
- The configuration template provides defaults and recommended values.
