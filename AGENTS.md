# Agent Instructions

## General Rules
- Strive for clean, modular code.
- Ensure all docstrings follow the Google style.
- Documentation should be updated in both German (`docs/de/`) and English (`docs/en/`).

## Pre-submission Checklist
- **Cleanup**: Temporary files (e.g., build artifacts like `site/`, log files, `changes.diff`) must be deleted before creating a Pull Request or submitting.
- **Documentation**: If the navigation structure in `mkdocs.yml` is changed, ensure that `nav_translations` in the `i18n` plugin are updated and landing pages (`index.md`) are synchronized.
- **Testing**: Run relevant unit tests to ensure no regressions are introduced.
