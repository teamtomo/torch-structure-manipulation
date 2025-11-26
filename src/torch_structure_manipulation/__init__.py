"""Module for loading molecular structures with bond information."""

from importlib.metadata import PackageNotFoundError, version

from .load_structure import load_model_bonds

try:
    __version__ = version("torch-structure-manipulation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Davide Torre"
__email__ = "davidetorre99@gmail.com"


__all__ = [
    "load_model_bonds",
]
