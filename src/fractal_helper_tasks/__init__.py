"""Collection of Fractal helper tasks"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-helper-tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"
