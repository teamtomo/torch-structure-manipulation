from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-structure-manipulation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Davide Torre"
__email__ = "davidetorre99@gmail.com"

from .fast_cif_parser import FastCIFBondParser
from .fast_atom_environments import FastAtomEnvironmentMapper
from .structure_transforms import StructureTransforms

__all__ = [
    "FastCIFBondParser",
    "FastAtomEnvironmentMapper",
    "StructureTransforms",
]
