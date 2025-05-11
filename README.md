# CONFLUENCE

<p align="center">
  <img src="https://github.com/DarriEy/CONFLUENCE/blob/main/docs/source/_static/Conf.jpg" alt="CONFLUENCE Logo" width="600">
</p>

<p align="center">
  <a href="https://github.com/DarriEy/CONFLUENCE/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-GPL%20v3-blue.svg" alt="License">
  </a>
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">
    <img src="https://img.shields.io/github/issues/DarriEy/CONFLUENCE.svg" alt="Issues">
  </a>
  <a href="https://github.com/DarriEy/CONFLUENCE/commits/main">
    <img src="https://img.shields.io/github/last-commit/DarriEy/CONFLUENCE.svg" alt="Last Commit">
  </a>
</p>

<h3 align="center">Community Optimization and Numerical Framework for Large-domain<br>Understanding of Environmental Networks and Computational Exploration</h3>

---

## Overview

CONFLUENCE is a  hydrological modeling platform designed to streamline the  modeling workflow—from conceptualization to evaluation. It provides a unified framework for running multiple hydrological models, handling data preprocessing, model calibration, and result visualization across various scales and regions.

> **Note**: CONFLUENCE is currently under active development. Features and interfaces may change.

## Key Features

- **Unified Workflow**: Integrated pipeline from data acquisition to result visualization
- **Multi-Model Support**: SUMMA, FUSE, MESH, HYPE, GR models with mizuRoute routing
- **Automated Domain Processing**: Geofabric delineation and discretization
- **Advanced Visualization**: Comprehensive plotting and reporting tools
- **Parallel Processing**: HPC-ready with MPI support
- **Configuration-Based**: YAML-based configuration for reproducible workflows
- **AI Integration**: Optional INDRA system for AI-assisted workflow guidance

## Quick Start

### Running on Shared Infrastructure

CONFLUENCE is designed to run on HPC environments. Use the provided batch script for easy execution:

```bash
# Clone the repository
git clone https://github.com/DarriEy/CONFLUENCE.git
cd CONFLUENCE

# Edit your configuration
cp config_template.yaml config_active.yaml
# Modify config_active.yaml with your settings

# Submit the job
sbatch run_CONFLUENCE_batch.sh
```

The `run_CONFLUENCE_batch.sh` script is pre-configured for the shared University of Calgary ARC infrastructure with:
- Appropriate SLURM settings
- Module loading for required dependencies

### Interactive Development

For development and testing, you can use the provided Jupyter notebooks:

## Prerequisites

### Required Software
- Python 3.11
- NetCDF libraries
- GDAL/OGR tools

### Required Models and tools (compile separately)
- [SUMMA](https://github.com/CH-Earth/summa.git)
- [FUSE](https://github.com/CyrilThebault/fuse)
- [MESH](https://github.com/MESH-Model/MESH-Dev)
- [mizuRoute](https://github.com/ESCOMP/mizuRoute.git)
- [TauDEM](https://github.com/dtarb/TauDEM.git)

## Configuration

CONFLUENCE uses YAML configuration files to control all aspects of the workflow:

```yaml
# Example configuration structure
CONFLUENCE_DATA_DIR: "/path/to/data"
DOMAIN_NAME: "my_watershed"
EXPERIMENT_ID: "test_run"
HYDROLOGICAL_MODEL: "SUMMA"
ROUTING_MODEL: "mizuRoute"
```

Key configuration sections:
- **Global Settings**: Paths, domain names, experiment IDs
- **Geospatial Settings**: Domain definition, discretization methods
- **Model Settings**: Model-specific parameters
- **Optimization Settings**: Calibration algorithms and metrics

See `config_template.yaml` for a complete example.

## Project Structure

```
CONFLUENCE/
├── 0_config_files/          # Configuration templates
├── examples/                # Jupyter notebook examples
├── utils/                   # Core utility modules
│   ├── configHandling_utils/    # Configuration management
│   ├── dataHandling_utils/      # Data processing
│   ├── models_utils/            # Model-specific utilities
│   ├── geospatial_utils/        # Domain processing
│   └── optimization_utils/      # Calibration tools
├── CONFLUENCE.py            # Main execution script
├── run_CONFLUENCE_batch.sh  # HPC batch submission script
└── requirements.txt         # Python dependencies
```


## Example Workflows

Browse the `jupyter_notebooks/` directory for comprehensive tutorials demonstrating:

1. **[01_lumped_basin_workflow.ipynb](exemples/01_lumped_basin_workflow.ipynb)** - Complete lumped basin modeling tutorial (Bow River at Banff)
2. **[02_semi_distributed_basin_workflow.ipynb](exemples/02_semi_distributed_basin_workflow.ipynb)** - Semi-Distributed modeling with multiple GRUs
3. **[03_distributed_basin_workflow.ipynb](exemples/03_distributed_basin_workflow.ipynb)** - Distributed modeling with multiple GRUs
4. **[04_regional_domain_workflow.ipynb](exemples/04_regional_domain_workflow.ipynb)** - Regional-scale modeling (Iceland example)
5. **[05_continental_domain_workflow.ipynb](examples/05_continental_North_America_workflow.ipynb)** - Continental-scale modeling (North America example)

Each notebook includes step-by-step instructions and can be run interactively to learn CONFLUENCE's capabilities.

### Getting Started with Examples

```bash
# Navigate to the notebooks directory
cd examples/

# Start Jupyter
jupyter notebook
```

## Documentation

📚 **Full documentation is available at [confluence.readthedocs.io](https://confluence.readthedocs.io/)**

[![Documentation Status](https://readthedocs.org/projects/confluence/badge/?version=latest)](https://confluence.readthedocs.io/en/latest/?badge=latest)

The documentation includes:
- **[Getting Started Guide](https://confluence.readthedocs.io/en/latest/getting_started.html)** - Quick introduction to CONFLUENCE
- **[Installation Instructions](https://confluence.readthedocs.io/en/latest/installation.html)** - Detailed setup guide
- **[Configuration Guide](https://confluence.readthedocs.io/en/latest/configuration.html)** - Complete configuration reference
- **[Examples](https://confluence.readthedocs.io/en/latest/examples.html)** - Practical tutorials based on Jupyter notebooks
- **[API Reference](https://confluence.readthedocs.io/en/latest/api.html)** - Detailed API documentation

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Submitting pull requests
- Reporting issues
- Adding new features

## Citation

If you use CONFLUENCE in your research, please cite:

```bibtex
@article{eythorsson2025,
  title={Toward Automated Scientific Discovery in Hydrology: The Opportunities and Dangers of AI Augmented Research Frameworks},
  author={Eythorsson, Darri and Clark, Martyn},
  journal={Hydrological Processes},
  volume={39},
  number={1},
  pages={e70065},
  year={2025},
  doi={10.1002/hyp.70065}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by NOAA (grant no. NA22NWS4320003). Computational resources provided by the Digital Research Alliance of Canada and the University of Calgary.

---

<p align="center">
  Developed at the University of Calgary<br>
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Report Bug</a> ·
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Request Feature</a>
</p>
