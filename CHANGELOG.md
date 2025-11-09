# Changelog

All notable changes to SYMFLUENCE are documented here.

---

## [0.5.0] - 2025-01-09

### Major: CONFLUENCE → SYMFLUENCE Rebranding

**This is the rebranding release.** The project is now SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration).

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: github.com/DarriEy/CONFLUENCE → github.com/DarriEy/SYMFLUENCE

### Deprecated
- All CONFLUENCE naming (will be removed in v1.0.0)
- Legacy names still work but show warnings

### Migration
```bash
# Update command
./confluence --install  # old (still works)
./symfluence --install  # new

# Update imports
from CONFLUENCE import CONFLUENCE  # old (still works)
from symfluence import SYMFLUENCE  # new

# Update config
CONFLUENCE_DATA_DIR → SYMFLUENCE_DATA_DIR
```

---

## Links
- PyPI: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- Docs: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- GitHub: [github.com/DarriEy/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)

