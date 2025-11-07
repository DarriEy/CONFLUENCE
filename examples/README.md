# CONFLUENCE Examples

This directory provides complete, ready-to-run examples that demonstrate **CONFLUENCE** across spatial scales — from single-site validation to continental-scale modeling.  
Each example includes configuration files, Jupyter notebooks, and optional batch scripts to reproduce the workflows described in the main documentation.

---

## Learning Path

Examples are organized into three progressive stages. Follow them sequentially to build understanding and confidence with CONFLUENCE.

### 01. Point-Scale Validation

Focuses on single-site process studies:

- **01a** - Snow physics validation using SNOTEL data  
- **01b** - Energy balance validation with FLUXNET observations  

Covers configuration structure, model setup, and controlled single-point testing.

### 02. Watershed Modeling

Demonstrates transition from lumped to distributed modeling at the basin scale using the Bow River at Banff (~2,600 km²):

- **02a** – Lumped model  
- **02b** – Semi-distributed sub-basins (~15 units)  
- **02c** – Elevation-based HRU discretization  

Highlights trade-offs between spatial complexity and computational cost.

### 03. Large-Domain Applications

Scales up to national and continental domains:

- **03a** – Iceland  
- **03b** – North America  

Focuses on high-performance execution, large datasets, and scaling efficiency.

---

## Quick Start

1. **Install dependencies**
   ```bash
   ./confluence --install
   ```

2. **Launch an example notebook** directly from the CLI:
   ```bash
   ./confluence --example_notebook 1a
   ```
   This command automatically:

   - Locates the corresponding notebook (e.g., `examples/01_point_vertical_flux_estimation/01a_point_scale_snotel.ipynb`)  
   - Opens it in Jupyter Lab  
   - Initializes it using the **root CONFLUENCE virtual environment**  

   You can substitute any example ID (e.g., `2b`, `3a`) to launch the corresponding workflow.

3. **Run complete workflows** via configuration:
   ```bash
   ./confluence --config config.yaml
   ```

---

## Data Access

Example data for **01a – 02c** are provided as a single ~200 MB bundle for quick testing.

- **Download:** [GitHub Release – Example Data (01a–02c)](https://github.com/DarriEy/CONFLUENCE/releases/download/examples-data-v0.1/confluence_examples_01a-02c_data.zip)
- **Automatic setup:** Example 01a automatically downloads this bundle if missing.

If you have access to institutional storage (e.g. **FIR** or **UCalgary ARC**),  
you may instead point the confluence config paths to your local MAF dataset.

> Large-domain examples (**03a–03b**) use multi-terabyte datasets and are **not included** in this bundle.  

---

## Example Contents

Each example directory includes:

- A complete and reproducible configuration file  
- A Jupyter notebook with contextual explanations  
- Optional batch script for automated runs  
- Notes on expected results and dataset access  

---

## Learning Outcomes

By completing these examples, you will learn to:

- Configure and manage CONFLUENCE workflows  
- Apply spatial discretization and calibration strategies  
- Execute simulations locally or on HPC systems  
- Automate large-sample experiments  
- Evaluate results statistically and visually  
- Transition confidently from validation studies to large-scale modeling  
