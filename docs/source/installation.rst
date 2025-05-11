Installation
============

Requirements
------------

CONFLUENCE requires Python 3.8 or higher and several scientific computing libraries.

System Dependencies
~~~~~~~~~~~~~~~~~~~

Before installing CONFLUENCE, you'll need to install and compile several hydrological modeling tools:

- **SUMMA** - Structure for Unifying Multiple Modeling Alternatives
  
  .. code-block:: bash
  
     git clone https://github.com/CH-Earth/summa.git
     cd summa
     # Follow SUMMA installation instructions

- **mizuRoute** - Network routing model
  
  .. code-block:: bash
  
     git clone https://github.com/ESCOMP/mizuRoute.git
     cd mizuRoute
     # Follow mizuRoute installation instructions

- **FUSE** - Framework for Understanding Structural Errors
  
  .. code-block:: bash
  
     git clone https://github.com/CyrilThebault/fuse
     cd fuse
     # Follow FUSE installation instructions

- **MESH** - Modelling Environmental Systems in an Integrated manner
- **TauDEM** - Terrain Analysis Using Digital Elevation Models
- **Ostrich** - Optimization Software Tool for Research Involving Computational Heuristics

Installation Steps
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/DarriEy/CONFLUENCE.git
      cd CONFLUENCE

2. Install Python dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Configure your environment:

   .. code-block:: bash

      cp config_template.yaml config_active.yaml
      # Edit config_active.yaml with your settings

4. Verify installation:

   .. code-block:: bash

      python CONFLUENCE.py --help

Next Steps
----------

Once installed, proceed to the :doc:`getting_started` guide to set up your first watershed model.
