Changelog
=========

All notable changes to this project are documented here.
This file follows a concise, category-based format (Added / Changed / Removed / Fixed).

Unreleased
----------

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


