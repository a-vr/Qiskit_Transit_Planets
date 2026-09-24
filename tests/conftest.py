"""pytest configuration: path setup and non-interactive matplotlib backend."""

import sys
import os

# Make `src` importable without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
