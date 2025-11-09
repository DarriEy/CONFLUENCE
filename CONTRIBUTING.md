# Contributing to SYMFLUENCE

We welcome all contributions — from bug fixes and documentation improvements to new model integrations and performance optimizations. This guide outlines how to get started and how to collaborate effectively.

---

## 1. Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/SYMFLUENCE.git
   cd SYMFLUENCE
   git remote add upstream https://github.com/DarriEy/SYMFLUENCE.git
   ```

2. **Set up your environment**
   ```bash
   ./symfluence --install
   ```
   This will create and manage a `.venv` automatically.  
   If you prefer manual setup:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   ./symfluence --help
   ```

---

## 2. Making Changes

### Code Style
- Follow **PEP 8** and use clear, descriptive variable names.
- Include **type hints** and short, informative **docstrings**.
- Keep functions focused and testable.

Example:
```python
def calculate_runoff(precip: float, area: float) -> float:
    """Compute runoff (m³/s) from precipitation rate and catchment area."""
    return (precip / 1000) * area * 1000 / 3600
```

### Commit Messages
Use concise, descriptive messages:
- `Add FUSE model interface`
- `Fix NetCDF write bug`
- `Update optimization documentation`

---

## 3. Submitting Your Work

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-update
   ```

2. **Commit and push**
   ```bash
   git commit -m "Describe your change"
   git push origin feature/my-update
   ```

3. **Open a Pull Request (PR)**  
   Include:
   - **Description:** what and why  
   - **Type:** new feature, fix, documentation, etc.  
   - **Testing:** how it was verified  
   - **Related issues:** e.g., “Closes #42”

Example:
```
## Description
Adds support for SUMMA-MIZU coupled runs.

## Type
- [x] Feature
- [ ] Fix
- [ ] Docs

## Testing
Validated on example domain; all tests pass.

## Related Issues
Closes #117
```

---

## 4. Code Review
All submissions are reviewed by maintainers. Expect constructive feedback — discussions help keep the codebase consistent and maintainable.  

Please be responsive and open to suggestions.

---

## 5. Reporting Issues
When reporting, include:
- **Description:** what went wrong  
- **Steps to reproduce**  
- **Expected vs actual behavior**  
- **Environment:** OS, Python version, SYMFLUENCE commit/branch  

Example:
```
## Description
Model setup fails on FIR cluster with NetCDF 4.9.2.

## Steps
1. Run ./symfluence --setup_project
2. Error during model initialization

## Expected
Setup completes successfully

## Actual
KeyError: 'MESH_PARAM_FILE'

## Environment
OS: Rocky Linux 8
Python: 3.11.7
Branch: main
```

---

## 6. Feature Requests
We value ideas for improvement. When proposing features:
- Describe the functionality and motivation.
- Suggest how it might fit into existing workflows.
- Include example usage if possible.

---

## 7. Contribution Types
We welcome:
- Bug fixes  
- Documentation updates  
- Example projects or tutorials  
- New model or data integrations  
- Performance improvements  
- Visualization or reporting enhancements

---

## 8. Questions
If you’re unsure where to start:
- Open a GitHub discussion or issue  
- Review existing docs at [symfluence.readthedocs.io](https://symfluence.readthedocs.io)

Thank you for helping improve SYMFLUENCE.
