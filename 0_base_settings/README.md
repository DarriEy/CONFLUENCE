# CONFLUENCE Base Settings

This directory contains default template files for supported hydrological models. These base settings provide clean, well-tested starting points that are copied to project-specific directories during preprocessing, ensuring consistent model setup while enabling project-specific customization.

## Structure

Each model has a dedicated subdirectory containing its required template files:

**SUMMA/** - Parameter files, decision options, output controls, and lookup tables for SUMMA model setup  
**FUSE/** - Configuration templates and parameter definitions for FUSE model applications  
**mizuRoute** - Topology definition, parameter and control file templates

## Usage

These base settings are automatically copied to project directories during model preprocessing. The original templates remain unchanged, while copied versions can be modified for specific applications without affecting other projects. This approach maintains clean defaults while supporting extensive customization for different modeling scenarios.

## Customization

Modify base settings carefully, as changes affect all future projects using these templates. For project-specific modifications, edit the copied files in the project settings directory rather than these base templates. This preserves reproducibility and ensures consistent starting points across different modeling applications.