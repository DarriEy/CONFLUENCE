# CONFLUENCE

<p align="center">
  <a href="https://github.com/DarriEy/CONFLUENCE/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">
    <img src="https://img.shields.io/github/issues/DarriEy/CONFLUENCE.svg" alt="Issues">
  </a>
  <a href="https://github.com/DarriEy/CONFLUENCE/commits/main">
    <img src="https://img.shields.io/github/last-commit/DarriEy/CONFLUENCE.svg" alt="Last Commit">
  </a>
</p>

<p align="center">
<strong>Community Optimization Nexus for Leveraging Understanding of<br>Environmental Networks in Computational Exploration</strong>
</p>

---

## Overview

CONFLUENCE is a computational framework designed to support hydrological modeling workflows from domain conceptualization through model evaluation. The platform integrates multiple modeling approaches and provides tools for parameter optimization, sensitivity analysis, and model comparison across various spatial scales.

The framework aims to reduce technical barriers in hydrological modeling by providing standardized workflows while maintaining flexibility for research applications. CONFLUENCE is actively developed and should be considered experimental software.

## Features

- **Workflow Integration**: Coordinates data preprocessing, model setup, execution, and analysis
- **Multi-Model Support**: Interfaces with SUMMA, FUSE, GR models and mizuRoute routing
- **Domain Processing**: Tools for watershed delineation and spatial discretization
- **Analysis Tools**: Built-in capabilities for benchmarking, sensitivity analysis, and visualization
- **Flexible Execution**: Support for local and high-performance computing environments
- **Configuration-Based**: YAML-driven configuration for reproducible workflows

## Installation

### Prerequisites

- Python 3.8 or later
- Git for repository access
- Internet connection for downloading external dependencies

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DarriEy/CONFLUENCE.git
   cd CONFLUENCE
   ```

2. **Run the installation script:**
   ```bash
   ./confluence --install
   ```

   This command will:
   - Create a Python virtual environment if needed
   - Install required Python packages
   - Download and configure external modeling tools
   - Validate the installation

3. **Verify the installation:**
   ```bash
   ./confluence --wrapper-info
   ./confluence --validate_binaries
   ```

## Usage

CONFLUENCE provides a command-line interface through the `confluence` shell wrapper:

```bash
# Display available options
./confluence --help

# Run complete workflow with default settings
./confluence

# Execute specific workflow components
./confluence --calibrate_model --run_benchmarking

# Configure domain from coordinates
./confluence --pour_point 51.1722/-115.5717 --domain_def delineate

# Check current workflow status
./confluence --status

# Use custom configuration file
./confluence --config /path/to/config.yaml
```

### Basic Workflow

1. **Prepare configuration:**
   ```bash
   cp 0_config_files/config_template.yaml my_config.yaml
   # Edit configuration file with project-specific settings
   ```

2. **Initialize project:**
   ```bash
   ./confluence --config my_config.yaml --setup_project
   ```

3. **Execute workflow:**
   ```bash
   ./confluence --config my_config.yaml
   ```

## Documentation

- **Installation Guide**: Detailed setup instructions for various environments
- **Configuration Reference**: Complete documentation of configuration parameters
- **User Guide**: Step-by-step workflow tutorials
- **API Reference**: Documentation for developers and advanced users
- **Examples**: Sample configurations and case studies

## Examples

The `examples/` directory contains example workflows demonstrating various modeling applications:

- Basic catchment modeling workflows
- Multi-model comparison studies
- Parameter calibration and optimization
- Sensitivity analysis procedures
- Large-scale domain processing

Each example includes documentation and sample datasets for reproducibility.

## Advanced Usage

### Command Line Options

```bash
# Run individual workflow steps
./confluence --setup_project --define_domain --calibrate_model

# Enable debug output
./confluence --debug --force_rerun

# Preview workflow without execution
./confluence --dry_run

# Domain setup with specific parameters
./confluence --pour_point "lat/lon" --domain_def delineate --domain_name "StudyArea"
```

### Python Interface

For programmatic access and integration:

```python
from pathlib import Path
from CONFLUENCE import CONFLUENCE

# Initialize framework
config_path = Path('config.yaml')
confluence = CONFLUENCE(config_path)

# Execute specific workflow components
confluence.run_individual_steps(['setup_project', 'calibrate_model'])

```

## Configuration

CONFLUENCE uses YAML configuration files to specify:
- Spatial domain boundaries and discretization parameters
- Model selection and parameterization
- Optimization targets and constraints
- Output formatting and analysis options

The template configuration file `0_config_files/config_template.yaml` provides detailed documentation of all available parameters.

## Project Structure

```
CONFLUENCE_CODE_DIR/
├── CONFLUENCE.py          # Main application entry point
├── confluence             # Command-line wrapper script
├── utils/                 # Core framework modules
│   ├── project/           # Project and workflow management
│   ├── geospatial/        # Spatial processing utilities
│   ├── models/            # Model interface modules
│   ├── cli/               # Command line interface modules
│   ├── data/              # Data processing modules
│   ├── reporting/         # Reporting and vizualisation modules
│   ├── optimization/      # Calibration and optimization algorithms
│   └── evaluation/        # Analysis and evaluation tools
├── 0_config_files/        # Configuration templates and examples
├── 0_base_settings/       # Default external tool configurations 
├── examples/              # Example workflows and case studies
├── docs/                  # Documentation source files
CONFLUENCE_DATA_DIR/
├── installs/              # External tool installations 
│   ├── summa/             # See https://github.com/CH-Earth/summa
│   ├── mizuRoute/         # See https://github.com/ESCOMP/mizuRoute 
│   ├── gistool/           # See https://github.com/CH-Earth/gistool.git
│   ├── datatool/          # See https://github.com/CH-Earth/datatool.git
│   ├── taudem/            # Se https://github.com/dtarb/TauDEM.git
│   └── fuse/              # See https://github.com/CyrilThebault/fuse.git
└── domain_directories     # Data directories for confluence modelling domains

```

## Contributing

Contributions are welcome through GitHub issues and pull requests. Please refer to the contributing guidelines for information on:
- Code formatting and documentation standards
- Testing procedures
- Issue reporting protocols

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for complete terms.

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

## Support

- **Documentation**: Available in the `docs/` directory and online
- **Issues**: Report bugs and request features through GitHub Issues
- **Community**: Join discussions through GitHub Discussions

## Acknowledgments

This work builds upon contributions from the hydrological modeling community and integrates tools developed by numerous researchers and institutions. We acknowledge the developers of the underlying modeling frameworks and the broader scientific software ecosystem that enables this work.

---

<p align="center">
<em>CONFLUENCE is research software under active development</em>
</p>
s