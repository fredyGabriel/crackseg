#!/usr/bin/env python3
"""
Debug Artifacts Compatibility Wrapper. This script maintains
compatibility with existing workflows while delegating to the
refactored modular debug artifacts system. For new development, import
directly from scripts.debug module: from scripts.debug import
ArtifactDiagnostics, ArtifactFixer
"""

import sys
from pathlib import Path

# Add debug directory to path for import s
sys.path.insert(0, str(Path(__file__).parent / "debug"))

# Import from the refactored modular system
from main import main

if __name__ == "__main__":
    main()
