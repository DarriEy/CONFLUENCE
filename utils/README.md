# Utilities

Core utility modules that power model integration, optimization, data handling, and workflow orchestration in CONFLUENCE. These modules expose clear interfaces and remain loosely coupled to support reliability and extensibility across scales.

---

## 1. Architecture

Utilities are organized by capability. Each module provides a focused interface and integrates through configuration and shared services (logging, project management).

### Project Management (`project/`)
Project initialization, directory layout, and logging infrastructure for reproducible runs. Establishes file organization and detailed execution records for debugging and auditability.

### Model Integration (`models/`)
Unified preprocessor → runner → postprocessor pattern. Central selection/dispatch with model-specific utilities for configuration generation, data formatting, execution, and output processing. Supports SUMMA, FUSE, GR, HYPE, FLASH, and MESH.

### Optimization (`optimization/`)
Two complementary paths:
- **Iterative** (e.g., differential evolution, particle swarm) with parallel execution.
- **Differentiable** parameter emulation for gradient-based optimization.
Shared interfaces enable consistent parameter handling, objective evaluation, and convergence monitoring.

### Command Line Interface (`cli/`)
Argument parsing and validation for interactive and batch workflows. Manages complex parameter combinations with clear error messages and usage output.

### Evaluation (`evaluation/`)
Performance assessment: sensitivity analysis, benchmarking, and decision-analysis utilities. Works with optimization results for systematic model evaluation.

### Reporting (`reporting/`)
Summary tables and figures for single-run and large-sample studies. Produces publication-ready visuals and comparative summaries.

### Data (`data/`)
Standardized interfaces for preprocessing, QC, and format conversion across common hydrologic data sources. Preserves metadata and consistency throughout the workflow.

---

## 2. Extending CONFLUENCE

### Add a Model
Follow the preprocessor → runner → postprocessor pattern. Implement model utilities under `models/` and register them with the dispatcher. Handle configuration generation, data formatting, execution, and output parsing.

### Add an Optimization Algorithm
Choose iterative or differentiable frameworks. Extend base optimizer interfaces; implement parameter sampling, objective evaluation, update rules, and convergence checks. Reuse logging and configuration services.

### Customize Workflows
Compose new steps using existing logging, error handling, and progress reporting utilities. Integrate via configuration without modifying core functionality.

---

## 3. Development Guidelines
- Maintain consistent error handling and logging.
- Keep module interfaces small and well-documented.
- Prefer configuration-driven behavior over hard-coded paths.
- Ensure additions work across small (point) to large (continental) scales.