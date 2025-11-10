# SYMFLUENCE  
**SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii**  

[![Build Status](https://img.shields.io/github/actions/workflow/status/DarriEy/SYMFLUENCE/ci.yml?branch=main)](https://github.com/DarriEy/SYMFLUENCE/actions)  
[![PyPI Version](https://img.shields.io/pypi/v/symfluence)](https://pypi.org/project/symfluence/)  
[![License](https://img.shields.io/badge/License-GPLv3-only-blue.svg)](LICENSE)  

## 🚀 Quick Links  
- [Documentation](https://symfluence.readthedocs.io)  
- [symfluence.org](https://symfluence.org)
- [GitHub Discussions](https://github.com/DarriEy/SYMFLUENCE/discussions)
- [GitHub Issues](https://github.com/DarriEy/SYMFLUENCE/issues)
- [Contributing Guide](CONTRIBUTING.md)  
- [Code of Conduct](CODE_OF_CONDUCT.md)  
- [CHANGELOG](CHANGELOG.md)  
---

## Overview
**SYMFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Installation
Environment setup varies by system.  
Use the built-in installer whenever possible:

```bash
./symfluence --install
```

This creates a clean Python 3.11 virtual environment, installs dependencies, and registers local binaries.  
For detailed instructions (ARC, FIR, Anvil, macOS), see [INSTALL.md](INSTALL.md).

---

## Quick Start

### Basic CLI Usage
```bash
# Show options
./symfluence --help

# Run default workflow
./symfluence

# Run specific steps
./symfluence --setup_project --calibrate_model

# Define domain from pour point
./symfluence --pour_point 51.1722/-115.5717 --domain_def delineate

# Preview workflow
./symfluence --dry_run
```

### First Project
```bash
cp 0_config_files/config_template.yaml my_project.yaml
./symfluence --config my_project.yaml --setup_project
./symfluence --config my_project.yaml
```
---

## Python API
For programmatic control or integration:

```python
from pathlib import Path
from symfluence import SYMFLUENCE

cfg = Path('my_config.yaml')
symfluence = SYMFLUENCE(cfg)
symfluence.run_individual_steps(['setup_project', 'calibrate_model'])
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
SYMFLUENCE/
├── SYMFLUENCE.py         # Main entry point
├── symfluence            # Shell wrapper
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

## Branching Strategy  
- **main**: Stable releases only — every commit is a published version.  
- **develop**: Ongoing integration — merges from feature branches and then tested before release.  
- Feature branches: `feature/<description>`, PR to `develop`.

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards and testing
- Branching and pull request process
- Issue reporting

---

## License
Licensed under the GPL-3.0 License.  
See [LICENSE](LICENSE) for details.

---

Happy modelling!  
The SYMFLUENCE Team  