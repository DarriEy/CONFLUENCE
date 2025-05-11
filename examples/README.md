# CONFLUENCE Examples

This directory contains example workflows demonstrating various CONFLUENCE applications, from single-point simulations to continental-scale modeling. Examples are organized into two main categories: domain-specific setups and large-sample analyses.

## Overview

CONFLUENCE examples showcase the framework's flexibility across different spatial scales and modeling approaches. Each example includes:
- Configuration files (YAML)
- Jupyter notebooks with step-by-step tutorials
- Shell scripts for batch processing
- Documentation explaining the specific use case

## 1. Domain-Specific Setups

These examples demonstrate CONFLUENCE workflows for specific locations at various spatial scales.

### 1.1 Point Scale
- **Location**: Single point simulations
- **Use Case**: Testing model physics at specific locations (e.g., flux towers, weather stations)
- **Key Features**: No spatial discretization, focuses on vertical processes
- **Example**: `01_point_scale.ipynb`

### 1.2 Lumped Basin - Bow at Banff
- **Location**: Bow River at Banff, Alberta, Canada
- **Scale**: ~2,600 km²
- **Use Case**: Traditional lumped hydrological modeling
- **Key Features**: Single HRU, entire watershed as one unit
- **Tutorial**: `01_lumped_basin_workflow.ipynb`

### 1.3 Semi-Distributed Basin - Bow at Banff
- **Location**: Same as lumped, but with multiple sub-basins
- **Scale**: ~2,600 km² divided into ~50 GRUs
- **Use Case**: Capturing some spatial heterogeneity while maintaining computational efficiency
- **Key Features**: Multiple GRUs with routing, balance between complexity and performance
- **Tutorial**: `02_semi_distributed_basin.ipynb`

### 1.4 Distributed Basin - Bow at Banff
- **Location**: Same watershed with detailed spatial discretization
- **Scale**: ~2,600 km² divided into 100+ HRUs
- **Use Case**: High-resolution modeling capturing detailed spatial variability
- **Key Features**: HRUs based on elevation bands
- **Tutorial**: `03_distributed_basin.ipynb`

### 1.5 Full Domain - Iceland
- **Location**: Entire country of Iceland
- **Scale**: ~103,000 km²
- **Use Case**: National-scale water resources assessment
- **Key Features**: Multiple watersheds, complex terrain
- **Tutorial**: `04_iceland_domain.ipynb`

### 1.6 Continental Domain - North America
- **Location**: North American continent
- **Scale**: ~24.7 million km²
- **Use Case**: Continental water balance, climate impact studies
- **Key Features**: Large-scale modeling, HPC requirements
- **Tutorial**: `05_north_america_domain.ipynb`

## 2. Large-Sample Setups

These examples demonstrate CONFLUENCE's capability to handle multiple sites/basins systematically for comparative hydrology and large-sample studies.

### 2.1 CARAVAN - Lumped Basins
- **Dataset**: CARAVAN (Catchment Attributes and Meteorology for Large-sample Studies)
- **Scale**: 10.000s of basins globally
- **Use Case**: Comparative hydrology, model benchmarking
- **Key Features**: Standardized data processing, batch execution
- **Script**: `run_watersheds_caravan.py`

### 2.2 NorSWE - Point Scale Snow
- **Dataset**: Northern Hemisphere Snow Water Equivalent dataset
- **Scale**: 10.000s snow monitoring sites across NH
- **Use Case**: Snow model evaluation, multi-site calibration
- **Script**: `run_sites_norswe.py`

### 2.3 CAMELSSPAT - Distributed Basins
- **Dataset**: CAMELS with spatial attributes
- **Scale**: 100s basins across the continental US
- **Use Case**: Distributed modeling comparison, spatial pattern analysis
- **Script**: `run_watersheds_camelsspat.py`

### 2.4 FLUXNET - Point Scale Met Fluxes
- **Dataset**: FLUXNET eddy covariance towers
- **Scale**: Global network of 200 + flux towers
- **Use Case**: Land-atmosphere interaction modeling, flux validation
- **Script**: `run_towers_fluxnet.py`

## Getting Started

### Prerequisites
- CONFLUENCE installation (see main README)
- Python environment with required packages
- Access to relevant datasets (see individual examples)

### Running Examples

1. **Interactive Jupyter Notebooks**: Best for learning and exploration
   ```bash
   jupyter notebook 01_lumped_basin.ipynb
    ```

2. **Batch Scripts**: To process multiple sites/basins at a time for large sample experiments
   ```python
   python run_watersheds_caravan.py --dataset camels --max-watersheds 10
    ```

## Key Learning Objectives
Each example is designed to teach specific CONFLUENCE concepts:

### Configuration Management: How to set up YAML configs for different use cases
### Spatial Discretization: From point to continental scales
### Workflow Orchestration: Managing complex modeling workflows
### Batch Processing: Handling multiple sites/basins efficiently
### Performance Optimization: Scaling from laptop to HPC

## Tips for New Users

Start with the lumped basin tutorial (01_lumped_basin_workflow.ipynb)
Understand the configuration structure before moving to complex examples
Check data requirements for each example before running
Review log files for troubleshooting
Use dry-run options when testing batch scripts

## Contributing
To add new examples:

Follow the existing directory structure
Include comprehensive documentation
Provide both interactive (notebook) and batch (script) versions
Add appropriate configuration templates
Update this README with the new example

## Support
For questions about specific examples:

Check the example-specific documentation
Review the main CONFLUENCE documentation
Open an issue on the GitHub repository


Last updated: January 2025