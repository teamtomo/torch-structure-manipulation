"""A python package to extract bonding environments from cif files and perform a few basic structure transformation (centering at a desired point, translation, rotation, removing atoms within/outisde a radius, removing sidechains, separating proteins and RNA) """

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-structure-manipulation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Davide Torre"
__email__ = "davidetorre99@gmail.com"
