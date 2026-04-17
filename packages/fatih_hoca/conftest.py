"""Conftest for fatih_hoca tests to ensure src is in path."""
import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
