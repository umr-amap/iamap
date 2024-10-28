import sys
import os

PYTHON_VERSION = sys.version_info
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT_DIR = os.path.realpath(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
PACKAGES_INSTALL_DIR = os.path.join(
    PLUGIN_ROOT_DIR, f"python{PYTHON_VERSION.major}.{PYTHON_VERSION.minor}"
)

sys.path.append(PACKAGES_INSTALL_DIR)  # TODO: check for a less intrusive way to do this
