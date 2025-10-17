# CONFLUENCE

<p align="center">
  <img src="https://github.com/DarriEy/CONFLUENCE/blob/main/docs/source/_static/Conf.jpg" alt="CONFLUENCE Logo" width="600">
</p>

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

<h3 align="center">Community Optimization Nexus for Leveraging Understanding of<br>Environmental Networks in Computational Exploration</h3>

---

## 🌊 Overview

CONFLUENCE is a comprehensive computational environmental modeling platform that streamlines the entire hydrological modeling workflow—from domain conceptualization to model evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across various spatial scales.

**Key Capabilities:**
- Unified workflow management for complex environmental modeling
- Multi-model integration and comparison
- Automated domain setup and discretization  
- Advanced parameter optimization and calibration
- Comprehensive result analysis and visualization

## ✨ Key Features

- **🔄 Complete Workflow Integration**: End-to-end pipeline from data acquisition to evaluation
- **🎯 Multi-Model Framework**: Support for SUMMA, FUSE, HYPE, GR models with mizuRoute routing
- **🗺️ Automated Domain Processing**: Intelligent watershed delineation and discretization
- **⚙️ Smart Installation**: Automated environment setup and dependency management
- **📊 Advanced Analytics**: Built-in benchmarking, sensitivity analysis, and visualization tools
- **⚡ Scalable Execution**: Support for local, cluster, and cloud computing environments
- **🤖 AI Integration**: Optional INDRA system for intelligent workflow assistance

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** 
- **Git** for cloning the repository
- **Network access** for downloading external tools and datasets

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DarriEy/CONFLUENCE.git
   cd CONFLUENCE
   ```

2. **Run the automated installer:**
   ```bash
   ./confluence --install
   ```

   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Download and install external modeling tools
   - Validate the installation

3. **Verify installation:**
   ```bash
   ./confluence --wrapper-info
   ./confluence --validate_binaries
   ```

### Basic Usage

CONFLUENCE provides a convenient shell wrapper for easy execution:

```bash
# Show available options
./confluence --help

# Run complete workflow with default configuration  
./confluence

# Run specific workflow steps
./confluence --calibrate_model --run_benchmarking

# Set up a new domain from pour point coordinates
./confluence --pour_point 51.1722/-115.5717 --domain_def delineate

# Check workflow status
./confluence --status

# Run with custom configuration
./confluence --config /path/to/your/config.yaml
```

### Your First CONFLUENCE Project

1. **Create a new project configuration:**
   ```bash
   cp 0_config_files/config_template.yaml my_project_config.yaml
   # Edit configuration with your project details
   ```

2. **Set up the project:**
   ```bash
   ./confluence --config my_project_config.yaml --setup_project
   ```

3. **Run the workflow:**
   ```bash
   ./confluence --config my_project_config.yaml
   ```

## 📖 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Configuration Reference](docs/configuration.md)** - Complete parameter documentation  
- **[User Guide](docs/user_guide.md)** - Step-by-step tutorials
- **[API Reference](docs/api.md)** - Developer documentation
- **[Examples](examples/)** - Example workflows and case studies

## 📚 Example Workflows

Explore the `examples/` directory for comprehensive tutorials:

- **Basic watershed modeling** - Complete workflow for a single catchment
- **Multi-model comparison** - Comparing different model structures
- **Parameter calibration** - Automated optimization workflows  
- **Sensitivity analysis** - Understanding parameter importance
- **Regional modeling** - Large-scale domain processing

Each example includes step-by-step instructions and sample datasets.

## 🛠️ Advanced Usage

### Command Line Interface

The `confluence` wrapper provides extensive CLI options:

```bash
# Individual workflow steps
./confluence --setup_project --define_domain --calibrate_model

# Debug mode with detailed logging
./confluence --debug --force_rerun

# Dry run to preview workflow steps
./confluence --dry_run

# Pour point-based domain setup
./confluence --pour_point "lat/lon" --domain_def delineate --domain_name "MyWatershed"
```

### Python API

For advanced users and integration with other tools:

```python
from pathlib import Path
from CONFLUENCE import CONFLUENCE

# Initialize CONFLUENCE
config_path = Path('my_config.yaml')
confluence = CONFLUENCE(config_path)

# Run specific workflow components
confluence.run_individual_steps(['setup_project', 'calibrate_model'])

# Access workflow results
results = confluence.get_results()
```

## 🔧 Configuration

CONFLUENCE uses YAML configuration files to define:
- Domain boundaries and discretization settings
- Model selection and parameters
- Optimization and calibration targets
- Output and visualization preferences

See `0_config_files/config_template.yaml` for a complete configuration template with detailed comments.

## 🏗️ Project Structure

```
CONFLUENCE/
├── CONFLUENCE.py           # Main entry point
├── confluence              # Shell wrapper script  
├── utils/                  # Core framework modules
│   ├── project/           # Project management
│   ├── geospatial/        # Domain processing
│   ├── models/            # Model interfaces
│   ├── optimization/      # Calibration algorithms
│   └── evaluation/        # Analysis tools
├── 0_config_files/        # Configuration templates
├── examples/              # Example workflows
├── docs/                  # Documentation
└── installs/              # External tools (auto-generated)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements  
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🆘 Support

- **Documentation**: [https://confluence.readthedocs.io](https://confluence.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/DarriEy/CONFLUENCE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DarriEy/CONFLUENCE/discussions)

## 🙏 Acknowledgments

CONFLUENCE builds upon decades of advances in hydrological modeling and computational frameworks. We acknowledge the contributions of the broader hydrological modeling community and the developers of the integrated modeling tools.

---

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

<p align="center">
  Developed at the University of Calgary<br>
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Report Bug</a> ·
  <a href="https://github.com/DarriEy/CONFLUENCE/issues">Request Feature</a>
</p>
