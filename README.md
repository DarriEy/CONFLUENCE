# SYMFLUENCE
**SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration**

[![PyPI version](https://badge.fury.io/py/symfluence.svg)](https://badge.fury.io/py/symfluence)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-symfluence.org-brightgreen)](https://symfluence.readthedocs.io)


---

## Overview
**SYMFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Quick Links

- **Install:** `pip install symfluence`
- **Documentation:** [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **Website:** [symfluence.org](https://symfluence.org)
- **Discussions:** [GitHub Discussions](https://github.com/DarriEy/SYMFLUENCE/discussions)
- **Issues:** [GitHub Issues](https://github.com/DarriEy/SYMFLUENCE/issues)

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

## Support
- Documentation: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)  
- Issues: [GitHub Issues](https://github.com/DarriEy/SYMFLUENCE/issues)  
- Discussions: [GitHub Discussions](https://github.com/DarriEy/SYMFLUENCE/discussions)
