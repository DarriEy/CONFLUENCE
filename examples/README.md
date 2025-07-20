# CONFLUENCE Examples

This directory contains comprehensive example workflows demonstrating CONFLUENCE applications across the complete spectrum of hydrological modeling scales, from single-point simulations to continental-scale modeling and large-sample comparative studies. The examples are organized into two complementary series: spatial scaling demonstrations and large-sample methodological applications.

## Tutorial Series Overview

CONFLUENCE examples showcase the framework's seamless scalability across different spatial scales and analytical approaches. Each tutorial includes complete configuration files, step-by-step Jupyter notebooks with detailed scientific explanations, automated batch processing scripts, and comprehensive documentation explaining specific use cases and applications.

## 1. Spatial Scaling Series: Point to Continental Applications

The spatial scaling series demonstrates CONFLUENCE's capability to handle hydrological modeling across the complete spatial hierarchy, from detailed process validation at individual points to comprehensive continental water resources assessment.

### Point Scale Validation (Tutorial 01)
Point-scale simulations focus on detailed process validation at specific locations such as flux towers and weather stations. These applications emphasize vertical process representation without spatial discretization, enabling detailed validation of hydrological physics and energy balance components against high-quality observational data.

### Basin-Scale Modeling Progression (Tutorials 02a-02c)
The Bow River at Banff case study demonstrates the complete progression of basin-scale modeling approaches within a consistent 2,600 km² watershed framework. Tutorial 02a establishes lumped modeling fundamentals using traditional single-unit representation of the entire watershed. Tutorial 02b advances to semi-distributed modeling with approximately 15 sub-basin units that capture spatial heterogeneity while maintaining computational efficiency. Tutorial 02c culminates with elevation-based HRU discretization that subdivides each sub-basin into elevation bands, creating detailed spatial representation that captures critical altitudinal controls on mountain hydrology.

### Regional and Continental Applications (Tutorials 03a-03b)
Tutorial 03a demonstrates regional domain modeling using Iceland as a comprehensive case study for national-scale water resources assessment across multiple independent drainage systems. Tutorial 03b extends to continental-scale modeling across North America, representing the ultimate computational challenge that requires high-performance computing infrastructure and demonstrates CONFLUENCE's capability for Earth system science applications.

## 2. Large-Sample Methodological Series

The large-sample series demonstrates CONFLUENCE's power for comparative hydrology and systematic model evaluation across diverse environmental conditions, enabling scientific insights that emerge only from multi-site analysis.

### Multi-Site Energy Balance Validation (Tutorial 04a)
FLUXNET tower applications demonstrate systematic energy balance validation across the global network of eddy covariance towers. This tutorial establishes large-sample methodology for land-atmosphere interaction modeling and provides comprehensive flux validation across diverse ecosystem types and climate regimes.

### Snow Process Validation (Tutorial 04b)
NorSWE snow observation network applications focus on systematic snow hydrology validation across northern hemisphere environmental gradients. This tutorial demonstrates multi-variable snow assessment integrating snow water equivalent and snow depth observations across elevation and climate gradients, enabling comprehensive evaluation of snow physics representation.

### Distributed Basin Comparison (Tutorial 04c)
CAMELS-Spat applications demonstrate large-sample distributed watershed modeling across hundreds of basins throughout the continental United States. This tutorial showcases spatial pattern analysis and distributed modeling comparison capabilities that reveal regional hydrological process controls and model transferability patterns.

## Getting Started

### Prerequisites
CONFLUENCE installation requires a complete Python environment with geospatial and scientific computing packages. Users should have access to relevant datasets for their intended applications and appropriate computational resources ranging from desktop systems for basin-scale work to high-performance computing clusters for continental applications.

### Tutorial Execution Approaches

**Interactive Learning**: Jupyter notebooks provide optimal environments for learning and exploration, offering step-by-step guidance with detailed scientific explanations and immediate visual feedback for understanding CONFLUENCE concepts and capabilities.

**Production Workflows**: Batch processing scripts enable systematic multi-site execution for large-sample experiments, providing automated configuration generation, parallel job submission, and comprehensive result aggregation for research applications.

## Key Learning Objectives

The tutorial series provides comprehensive training in fundamental CONFLUENCE concepts including configuration management across diverse spatial scales and use cases, spatial discretization principles from point-scale through continental applications, workflow orchestration for complex modeling sequences, batch processing methodologies for handling multiple sites efficiently, and performance optimization strategies for scaling from desktop to high-performance computing environments.

Each tutorial builds systematically on previous concepts while introducing new spatial scales or methodological approaches, ensuring users develop both depth of understanding and breadth of application capabilities.

## Best Practices for New Users

Begin with the lumped basin tutorial to establish fundamental CONFLUENCE concepts before advancing to complex multi-scale applications. Understand the configuration structure and workflow principles before attempting large-sample studies. Verify data requirements for each tutorial before execution and always review log files for troubleshooting guidance. Utilize dry-run options when testing batch processing scripts to avoid computational resource waste.

The tutorial progression is designed to build expertise systematically, ensuring users develop both technical proficiency and scientific understanding necessary for successful hydrological modeling applications across the complete spectrum of spatial scales and analytical approaches.

## Contributing and Support

New tutorial contributions should follow the established directory structure and documentation standards. Include both interactive notebook versions for learning and batch script versions for production applications. Provide comprehensive configuration templates and update this documentation with new example descriptions.

For specific tutorial questions, consult the individual tutorial documentation, review the main CONFLUENCE documentation, or open an issue on the GitHub repository for community support and developer assistance.