"""
Ensure the repo root is on sys.path so tests can import pipeline.* modules.
"""

import sys
import os

# The repo root is one level above this tests/ directory.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
