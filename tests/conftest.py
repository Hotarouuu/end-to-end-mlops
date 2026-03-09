"""Pytest configuration for the farm-detection test suite.

Inserts the project root directory into ``sys.path`` so that top-level modules
(e.g. ``app.py``) are importable from within the ``tests/`` directory without
requiring an editable install.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
