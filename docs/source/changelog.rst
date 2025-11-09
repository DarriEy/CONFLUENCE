Changelog
=========

All notable changes to this project are documented here.
This file follows a concise, category-based format (Added / Changed / Removed / Fixed).

Version 0.2.0 (Unreleased)
--------------------------

Changed
-------
- **Major rebrand:** CONFLUENCE renamed to SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration)
- All documentation updated to reference SYMFLUENCE
- Repository moved to https://github.com/DarriEy/SYMFLUENCE
- ReadTheDocs URL updated to symfluence.readthedocs.io
- Configuration file parameters updated: ``CONFLUENCE_DATA_DIR`` → ``SYMFLUENCE_DATA_DIR``, ``CONFLUENCE_CODE_DIR`` → ``SYMFLUENCE_CODE_DIR``
- Main Python class renamed: ``CONFLUENCE`` → ``SYMFLUENCE``
- CLI command updated: ``./confluence`` → ``./symfluence``

Added
-----
- Backward compatibility wrappers for ``./confluence`` command and ``CONFLUENCE`` Python class
- Deprecation notices in documentation for old naming conventions
- Migration guide for users transitioning from CONFLUENCE

Deprecated
----------
- ``./confluence`` CLI command (use ``./symfluence`` instead)
- ``CONFLUENCE`` Python class (use ``SYMFLUENCE`` instead)
- ``CONFLUENCE_DATA_DIR`` and ``CONFLUENCE_CODE_DIR`` configuration parameters (use ``SYMFLUENCE_*`` equivalents)

---

Version 0.1.0 (Previous)
------------------------

Added
-----
- New documentation pages: ``installation.rst``, ``getting_started.rst``, ``workflows.rst``,
  ``examples.rst``, ``troubleshooting.rst``, and ``api.rst`` (concise, production tone).
- Base settings documentation now references CH-Earth repositories for model details (SUMMA, FUSE, mizuRoute, NOAH).
- Examples documentation aligned with repository examples and learning path.

Changed
-------
- Root ``README.md`` rewritten for clarity (one-command install, HPC notes, minimal structure).
- ``INSTALL.md`` updated to reflect built-in venv behavior (Python 3.11) and HPC module recipes.
- ``CONTRIBUTING.md`` simplified and made flexible (PEP 8, type hints, clear PR expectations).
- ``0_config_files/README.md`` clarified structure and best practices (validation, versioning).
- ``configuration.rst`` aligned to the current configuration template schema (global/logging, geospatial,
  data/forcings, models: SUMMA/FUSE/NextGen/GR4J/LSTM + mizuRoute, optimization, emulation, paths/resources).
- Sphinx ``index.rst`` modernized to mirror README sections and reflect the supported models list.

Removed
-------
- The ``modules`` section from the docs (deferred until autodoc is enabled).
- Tool-specific READMEs within ``0_base_settings/<tool>/`` (now link to upstream repos).

Fixed
-----
- Outdated references to older Python versions or legacy options in docs.
- Inconsistent terminology across installation, configuration, and examples pages.
