"""Pytest configuration for fatih_hoca tests.

Adds the tests/ directory to sys.path so that sim.state (and future
tests/sim/* modules) are importable without polluting the runtime package.
"""
import sys
import pathlib

# Allow `from sim.state import ...` in tests/sim/
_tests_dir = pathlib.Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
