# CONFLUENCE
**Community Optimization Nexus for Leveraging Understanding of Environmental Networks in Computational Exploration**

---

## Overview
**CONFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Installation
Environment setup varies by system.  
Use the built-in installer whenever possible:

```bash
./confluence --install
```

This creates a clean Python 3.11 virtual environment, installs dependencies, and registers local binaries.  
For detailed instructions (ARC, FIR, Anvil, macOS), see [INSTALL.md](INSTALL.md).

---

## Quick Start

### Basic CLI Usage
```bash
# Show options
./confluence --help

# Run default workflow
./confluence

# Run specific steps
./confluence --setup_project --calibrate_model

# Define domain from pour point
./confluence --pour_point 51.1722/-115.5717 --domain_def delineate

# Preview workflow
./confluence --dry_run
```

### First Project
```bash
cp 0_config_files/config_template.yaml my_project.yaml
./confluence --config my_project.yaml --setup_project
./confluence --config my_project.yaml
```
---

## Python API
For programmatic control or integration:

```python
from pathlib import Path
from CONFLUENCE import CONFLUENCE

cfg = Path('my_config.yaml')
confluence = CONFLUENCE(cfg)
confluence.run_individual_steps(['setup_project', 'calibrate_model'])
```

---

## Configuration
YAML configuration files define:
- Domain boundaries and discretization
- Model selection and parameters
- Optimization targets
- Output and visualization options

See [`0_config_files/config_template.yaml`](0_config_files/config_template.yaml) for a full example.

---

## Project Structure
```
CONFLUENCE/
├── CONFLUENCE.py         # Main entry point
├── confluence            # Shell wrapper
├── utils/                # Core framework modules
│   ├── project/
│   ├── geospatial/
│   ├── models/
│   ├── optimization/
│   └── evaluation/
├── 0_config_files/       # Configuration templates
├── examples/             # Example workflows
├── docs/                 # Documentation
└── installs/             # Auto-generated tool installs
```

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards and testing
- Branching and pull request process
- Issue reporting

---

## License
Licensed under the MIT License.  
See [LICENSE](LICENSE) for details.

---

## Support
- Documentation: [confluence.readthedocs.io](https://confluence.readthedocs.io)  
- Issues: [GitHub Issues](https://github.com/DarriEy/CONFLUENCE/issues)  
- Discussions: [GitHub Discussions](https://github.com/DarriEy/CONFLUENCE/discussions)

---

```
