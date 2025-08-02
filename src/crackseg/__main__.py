#!/usr/bin/env python
"""
Main entry point for running the CrackSeg project from the command line.
"""

import os
import sys

# Ensure the package root is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust if __main__.py is deeper
package_root = os.path.abspath(os.path.join(current_dir, ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from main import main  # noqa: E402

if __name__ == "__main__":
    # This allows the package to be run as a script
    # Example: python -m src
    # Hydra will automatically provide the cfg parameter when called from
    # command line
    # For direct execution, we need to handle this differently
    import sys

    if len(sys.argv) > 1:
        # If arguments are provided, let Hydra handle them
        main()
    else:
        # For direct execution without arguments, we need to provide a default
        # config
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).parent.parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            cfg = compose(config_name="base")
            main(cfg)
