# Example Workflows

This directory contains practical examples demonstrating CONFLUENCE across scales—from single-site validation to continental modeling.  
Each example provides a complete, ready-to-run workflow with configuration files, Jupyter notebooks, and batch scripts.

---

## 1. Structure and Learning Path

Examples are organized into four progressive stages. Follow them in order for the best learning experience.

### **01. Point-Scale Validation**
**Foundation level**  
Introduces core concepts using focused validation studies:
- Snow physics validation with SNOTEL data  
- Energy balance validation with FLUXNET tower observations  
Teaches configuration structure, model setup, and single-point validation.

### **02. Watershed Modeling**
**Spatial scaling**  
Demonstrates modeling from lumped to distributed representations using the Bow River at Banff (≈2,600 km²):  
- 02a: Single-unit lumped model  
- 02b: Semi-distributed sub-basin model (~15 units)  
- 02c: Elevation-based HRU discretization  
Shows trade-offs between spatial detail and computational cost.

### **03. Large-Domain Applications**
**Scaling up**  
Explores regional to continental applications:  
- 03a: National-scale modeling (Iceland)  
- 03b: Continental modeling (North America)  
Covers HPC execution, large data handling, and resource scaling.

### **04. Large-Sample Studies**
**Comparative and ensemble workflows**  
Demonstrates multi-site and global analyses using:
- FLUXNET, NorSWE, CAMELS, CARAVAN, LamaH-CE, NZ datasets, ISMN  
Focuses on automation, parallel execution, and systematic evaluation.

---

## 2. Getting Started

1. Ensure CONFLUENCE is installed with all dependencies:
   ```bash
   ./confluence --install
   ```

2. Open the desired notebook in the example folder.

3. Follow the step-by-step instructions within each notebook or run:
   ```bash
   ./confluence --config config.yaml
   ```

**Recommended path:** begin with 01 tutorials, then move upward as you gain experience.

Each example includes:
- A complete configuration file  
- Jupyter notebook with explanations  
- Optional batch script for automated runs  
- Notes on expected results and dataset access

---

## 3. Learning Outcomes
Through these tutorials you will:
- Manage configurations and workflow dependencies  
- Apply spatial discretization strategies  
- Orchestrate complete model sequences  
- Automate batch runs and multi-site experiments  
- Evaluate models statistically and visually  
- Transition from local runs to HPC environments confidently
