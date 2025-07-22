# CrackSeg GUI

This directory contains the source code for the Streamlit-based graphical user
interface (GUI) of the CrackSeg project.

## Overview

The GUI provides an interactive, user-friendly way to configure experiments,
run training, monitor progress, and visualize results. It is designed with a
modular architecture to facilitate maintenance and extensibility.

## Directory Structure

- `app.py`: The main entry point for the Streamlit application.
- `pages/`: Each `.py` file here represents a distinct page in the multi-page
    application (e.g., Home, Config, Train).
- `components/`: Reusable UI elements (e.g., file browser, header, config
    editor) that are used across different pages.
- `utils/`: Helper functions and classes that support the GUI, such as state
    management, process handling, and configuration I/O.
- `services/`: Backend services that the GUI relies on (e.g., fetching system
    info).
- `assets/`: Static assets like CSS, images, and fonts. See the
    [assets/README.md](assets/README.md) for details on the asset management
    system.
- `styles/`: Contains CSS files for theming and custom styling.

## How to Run

To launch the GUI, run the following command from the project root:

```bash
streamlit run gui/app.py
```

## Development

- **Modularity**: When adding new features, consider whether they belong in a
    new component, a new page, or a new utility.
- **State Management**: All session state is managed centrally through the
    `SessionStateManager` in `utils/session_state.py`. Use this manager to
    get and set state variables to ensure consistency.
- **Styling**: Apply styles using the `ThemeComponent` and custom CSS files in
    the `styles/` and `assets/` directories. Avoid inline styles where possible.

For a complete guide on the recent refactoring, see
[README_REFACTORING.md](README_REFACTORING.md).
