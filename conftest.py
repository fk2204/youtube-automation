"""
Root conftest — adds src/ to sys.path so all imports resolve
without installing the package.
"""

import sys
import os

# Allow `from social.xxx import Xxx` and `from registry.xxx import Xxx`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
