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

---

## Overview

CONFLUENCE is a  hydrological modeling platform designed to streamline the  modeling workflow—from conceptualization to evaluation. It provides a unified framework for running multiple hydrological models, handling data preprocessing, model calibration, and result visualization across various scales and regions.

> **Note**: CONFLUENCE is currently under active development. Features and interfaces may change.

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

Browse the `jupyter_notebooks/` directory for comprehensive tutorials.

Each notebook includes step-by-step instructions and can be run interactively.

### Getting Started with Examples

```bash
# Navigate to the notebooks directory
cd examples/

# Start Jupyter
jupyter notebook
```

> **Note**: University of Calgary ARC users can get quick started using the **[OpenOnDemand Service](https://ood-arc.rcs.ucalgary.ca/pun/sys/dashboard/batch_connect/sys/uc_jupyter/session_contexts/new)** Please use the following script path /work/comphyd_lab/data/CONFLUENCE_data/installs/jupyter-conf.sh as the Jupyter initialization batch script


## Documentation

📚 **Documentation is available at [confluence.readthedocs.io](https://confluence.readthedocs.io/)**

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

This work is intended to fuse a collection of community assets in an integrated workflow framework. 

This work primarily builds on the work presented in the [Community Workflows to Advance Reproducibility in Hydrological Modelling, CWARHM, Knoben et al., 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031753) and later advances on those workflows in the [Model Agnostic Framework, Keshavarz et al., 2025](https://github.com/CH-Earth/agnostic-orchestrator). 

This work was supported by NOAA (grant no. NA22NWS4320003). Computational resources provided by the Digital Research Alliance of Canada and the University of Calgary.

---

<p align="center">
  Developed at the University of Calgary<br>
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Report Bug</a> ·
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Request Feature</a>
</p>
