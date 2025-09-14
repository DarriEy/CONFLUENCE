# CONFLUENCE Examples

This directory contains practical example workflows that demonstrate CONFLUENCE's capabilities across different scales and applications in hydrological modeling. The examples progress from simple point-scale setups to complex large-sample studies, providing hands-on experience with the complete modeling workflow.

## Structure and Learning Path

The examples are organized into four progressive sections, each building on concepts from the previous section.

### 01. Point-Scale Validation
**Start here for CONFLUENCE fundamentals**

The point-scale tutorials introduce core CONFLUENCE concepts through focused validation studies. Tutorial 01a demonstrates snow physics validation using SNOTEL station data, while Tutorial 01b covers energy balance validation using FLUXNET tower observations. These examples establish fundamental skills in configuration, model setup, and single-point validation techniques.

### 02. Watershed Modeling
**Progress from simple to complex spatial representations**

The watershed modeling series uses the Bow River at Banff (2,600 km²) to demonstrate spatial discretization approaches. Tutorial 02a begins with lumped single-unit watershed modeling, Tutorial 02b advances to semi-distributed modeling with approximately 15 sub-basin units, and Tutorial 02c implements full elevation-based HRU discretization. This progression illustrates the trade-offs between computational complexity and spatial representation.

### 03. Large Domain Applications  
**Scale up to regional and continental modeling**

These tutorials address computational considerations and data management for large-scale applications. Tutorial 03a demonstrates national-scale modeling using Iceland as a comprehensive case study, while Tutorial 03b extends to continental-scale modeling across North America. Users learn about high-performance computing requirements and strategies for managing complex, large-domain simulations.

### 04. Large Sample Studies
**Apply CONFLUENCE across multiple sites for comparative analysis**

The large-sample series demonstrates systematic model evaluation across diverse environmental conditions. These tutorials cover energy balance validation across FLUXNET towers (04a), snow hydrology across the NorSWE network (04b), and distributed modeling across CAMELS watersheds (04c). Additional examples include global hydrology using the CARAVAN dataset (04d), European applications via LamaH-CE (04e), New Zealand case studies (04f), groundwater network analysis (04g), and soil moisture validation using ISMN (04h). This comprehensive suite teaches batch processing, parallel execution, and systematic model evaluation techniques.

## Getting Started

CONFLUENCE installation with required dependencies provides the foundation for these tutorials. Users should have basic Python and Jupyter notebook experience, along with access to relevant datasets (download instructions are provided in each notebook). Computational resources should match the scale of chosen examples, ranging from desktop systems for point-scale work to high-performance computing for continental applications.

The recommended learning path follows the numbered sequence, as each tutorial builds on previous concepts. Interactive execution using Jupyter notebooks provides the best learning environment, while the provided batch scripts enable production workflows once users become comfortable with the framework.

## Learning Outcomes

These tutorials provide comprehensive training in configuration management across different scales and use cases, spatial discretization strategies and their applications, workflow orchestration for complex modeling sequences, and batch processing for efficient multi-site analysis. Users also gain experience with performance optimization from desktop to HPC environments and develop skills in model evaluation techniques and validation approaches.

Each example includes complete configuration files ready to run, step-by-step Jupyter notebooks with explanations, batch processing scripts for automated execution, documentation explaining use cases and scientific context, and guidance for interpreting expected outputs.

## Best Practices and Support

Success with these tutorials requires starting with smaller examples before attempting large-scale applications. Users should review configuration files to understand parameter choices, check data availability and download requirements before starting, and use dry-run options when testing new configurations. Consulting log files provides valuable troubleshooting guidance, and verifying computational resources match tutorial requirements prevents common issues.

For support, users should first review individual tutorial documentation, then check the main CONFLUENCE documentation, and finally open GitHub issues for specific problems. Contributors should follow established directory structure and naming conventions, include both interactive notebooks and batch scripts, provide complete configuration templates, and update this README when adding new examples.

The goal of these examples is to provide practical experience building toward reliable, large-scale hydrological predictions. Each tutorial contributes specific skills while maintaining focus on real-world applications and scientific rigor.