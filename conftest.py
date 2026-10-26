"""
conftest.py
===========
Pytest configuration file placed at the project root.

This ensures that:
- pytest can discover tests in the 'tests/' directory.
- The project root is on sys.path so 'src' and other modules can be imported.
"""
import pathlib
import sys

# Ensure the project root is always importable regardless of where pytest is invoked from
sys.path.insert(0, str(pathlib.Path(__file__).parent))
