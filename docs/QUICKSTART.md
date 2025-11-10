# Quickstart (smoke path)

This quickstart reproduces the CI smoke test on a tiny dataset.

## Steps
1. Install:
```bash
python -m pip install --upgrade pip
pip install -e .[dev,docs]
```
2. Configure (defaults auto-create `../SYMFLUENCE_data`):
```bash
symfluence init --accept-defaults
```
3. Run the smoke example:
```bash
pytest -q -k smoke
```
4. Build docs locally (optional):
```bash
make -C docs html
open docs/build/html/index.html
```
