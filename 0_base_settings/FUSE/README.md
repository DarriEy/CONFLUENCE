# FUSE Base Settings

This directory contains template configuration files for the FUSE (Framework for Understanding Structural Errors) hydrological model. These templates define model structure decisions, parameter constraints, numerical methods, and execution settings that are copied to project directories during preprocessing.

## Core Files

**fm_catch.txt** - File manager template that specifies paths, input/output settings, simulation dates, evaluation metrics, and optimization parameters. CONFLUENCE automatically updates paths and dates from the main configuration during preprocessing.

**input_info.txt** - Defines the structure and variables in forcing data files, ensuring consistent data interpretation across different input sources and formats.

**fuse_zConstraints_snow.txt** - Specifies parameter bounds and constraints for model calibration, providing physically reasonable limits for optimization algorithms.

**fuse_zNumerix.txt** - Controls numerical solution methods including time step settings, convergence criteria, and solver options for stable model execution.

**fuse_zDecisions_902.txt** - Defines model structure decisions for different hydrological processes. The file is automatically renamed during preprocessing to match the experiment ID (e.g., `fuse_zDecisions_run_1.txt`).

## Model Structure Decisions

FUSE's flexibility comes from multiple options for representing key hydrological processes:

**Error Handling (RFERR)** - Controls how precipitation errors are treated (additive vs multiplicative)  
**Architecture (ARCH1/ARCH2)** - Defines soil moisture accounting structure and configuration  
**Surface Runoff (QSURF)** - Selects surface runoff generation mechanism (Arno, PRMS, TOPMODEL variants)  
**Percolation (QPERC)** - Controls how water moves between soil layers  
**Evapotranspiration (ESOIL)** - Determines soil evaporation calculation approach  
**Interflow (QINTF)** - Enables or disables lateral subsurface flow  
**Routing (Q_TDH)** - Selects catchment routing method (gamma function or none)  
**Snow Module (SNOWM)** - Enables temperature-index snow modeling or disables snow processes  

## Usage Notes

These templates provide well-tested default configurations that work across diverse hydrological conditions. The decision file (902) represents a balanced model structure suitable for most applications, while other files provide conservative parameter bounds and stable numerical settings.

During preprocessing, paths are automatically updated to match your project structure, dates are set from the main configuration, and optimization settings are applied from calibration parameters. The original templates remain unchanged while copied versions can be customized for specific modeling requirements.