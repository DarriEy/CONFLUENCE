# CONFLUENCE Configuration Files

This directory contains template and active configuration files that control all aspects of CONFLUENCE workflows. Configuration files provide the central coordination point for experiment settings, model selection, data paths, and analysis parameters across the complete modeling pipeline from conceptualization to evaluation.

## Configuration Structure

CONFLUENCE uses YAML configuration files organized into logical sections that mirror the workflow progression. Each section contains settings that become available to subsequent workflow components, creating a natural dependency structure that prevents configuration conflicts while enabling flexible experiment design.

### Global Settings
The foundation configuration establishes directory paths, domain names, and experiment identifiers used throughout the workflow. These settings define the physical organization of data and results while providing unique identifiers for tracking experiments and managing multiple concurrent projects.

### Geospatial Configuration
Spatial domain settings control watershed delineation, coordinate systems, and spatial discretization approaches. These parameters determine how CONFLUENCE translates geographic information into computational domains, supporting scales from individual points to continental applications through consistent spatial frameworks.

### Model-Agnostic Settings
Data preprocessing configuration manages forcing data acquisition, quality control, and format standardization before model-specific processing. These settings ensure consistent data handling across different model applications while supporting diverse input sources and temporal resolutions.

### Model-Specific Settings
Individual model configuration sections contain all parameters needed for model setup, execution, and output processing. Each supported model (SUMMA, FUSE, GR, HYPE, FLASH, MESH) has dedicated configuration sections with appropriate parameter defaults and path specifications that integrate seamlessly with the broader workflow.

### Evaluation Configuration
Analysis settings control performance assessment, benchmarking approaches, and visualization options. These parameters determine how model results are evaluated against observations and processed into meaningful scientific insights through statistical analysis and graphical summaries.

### Optimization Configuration
Calibration settings manage parameter estimation approaches, objective function definitions, and optimization algorithm selection. These parameters control both traditional iterative optimization and modern gradient-based approaches while supporting parallel processing and convergence monitoring.

## Working with Configurations

### Creating New Configurations
Start with the provided template file and customize settings for your specific application. The template includes comprehensive parameter documentation and default values that work for most applications. Focus first on global settings and spatial domain definition before proceeding to model-specific parameters.

### Configuration Hierarchy
Settings follow a logical progression where early sections provide values used in later sections. Global paths and experiment identifiers flow through all workflow components, while spatial domain settings inform model setup and evaluation approaches. This hierarchy ensures consistency while enabling flexible parameter combinations.

### Path Management
CONFLUENCE supports both default and custom path specifications throughout the configuration. Default paths follow established conventions and integrate naturally with the project directory structure, while custom paths enable integration with existing data management systems or specialized computational environments.

### Multi-Model Workflows
Single configuration files can specify multiple models for comparative studies or ensemble applications. Model-specific sections operate independently while sharing common settings from earlier configuration sections, enabling systematic comparison across different modeling approaches.

## Best Practices

### Configuration Management
Maintain separate configuration files for different experiments or domains rather than modifying existing configurations. This approach preserves reproducibility while enabling systematic parameter exploration. Use descriptive experiment identifiers that clearly indicate the purpose and characteristics of each modeling setup.

### Parameter Documentation
Document any deviations from default parameters within configuration files using YAML comments. This documentation provides valuable context for future analysis and helps maintain understanding of experimental design decisions over time.

### Validation Approach
Use the configuration validation tools to check parameter consistency before initiating workflows. The validation process identifies common configuration errors and provides guidance for resolution, preventing computational resource waste and ensuring reliable results.

### Version Control
Track configuration files alongside analysis scripts and documentation to maintain complete experimental records. Configuration files provide the essential metadata needed for reproducing results and understanding experimental design decisions.

## Support and Troubleshooting

Configuration validation tools provide immediate feedback on parameter consistency and path availability. Error messages include specific guidance for resolving common configuration issues, while the template file provides comprehensive parameter documentation with recommended value ranges.

The hierarchical configuration structure ensures that changes in early sections propagate appropriately to dependent components, while the modular design enables incremental configuration development that matches workflow progression from conceptualization through evaluation.

Understanding configuration structure provides the foundation for effective CONFLUENCE application across diverse hydrological modeling challenges while maintaining the reproducibility and scientific rigor essential for reliable environmental predictions.