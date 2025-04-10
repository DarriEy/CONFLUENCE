# CONFLUENCE
Community Optimization and Numerical Framework for Large-domain Understanding of Environmental Networks and Computational Exploration

## Overview

CONFLUENCE is an advanced hydrological modeling platform designed to facilitate comprehensive modeling and analysis across various scales and regions. It integrates multiple components for data management, model setup, optimization, uncertainty analysis, forecasting, visualization, and workflow management.

NOTE: CONFLUENCE is currently in development

## Features

- Data preprocessing and acquisition 
- Hydrological model setup and initialization
- Model optimization and calibration
- Result visualization and reporting
- Workflow management for complex modeling tasks

## Installation
1. Clone the repository:

```    
git clone https://github.com/DarriEy/CONFLUENCE.git
``` 

2. Install the required dependencies:


```
cd CONFLUENCE 

pip install -r requirements.txt
```

3. Clone and compile/install the desired/required binaries

- SUMMA (https://github.com/CH-Earth/summa.git)
- FUSE (https://github.com/CyrilThebault/fuse)
- MESH (https://github.com/MESH-Model/MESH-Dev)
- mizuRoute (https://github.com/ESCOMP/mizuRoute.git)
- TauDEM (https://github.com/dtarb/TauDEM.git)
- Ostrich (http://www.civil.uwaterloo.ca/envmodelling/Ostrich.html)

## Configuration
- Copy `config_template.yaml` to `config_active.yaml` or any other name
- Modify the copy according to your needs
- All config files except `config_template.yaml` are ignored by git

## Usage

1. Set up your project configuration in `config_active.yaml`
2. Run the main CONFLUENCE script:

python CONFLUENCE.py --config path/to/your/config_active.yaml

or

3. Run the scripts in /jupyter notebooks for a stepwise introduction to CONFLUENCE

## Contributing

We welcome contributions to CONFLUENCE! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

Please make sure to update tests as appropriate and adhere to the project's coding standards.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

