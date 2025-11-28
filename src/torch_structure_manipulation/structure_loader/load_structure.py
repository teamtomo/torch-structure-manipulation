"""Module for loading molecular structures with bond information.

This module provides an API that returns a pandas DataFrame instead of
separate tensors and lists.
"""

import pathlib
from dataclasses import dataclass

import pandas as pd

from torch_structure_manipulation.structure_transforms import center_structure

from .load_structure_utils import (
    get_bonded_atom_ids_and_molecule_types,
    load_df,
)


@dataclass
class StructureLoadOptions:
    """Configuration options for loading the structure.

    Attributes
    ----------
    center_atoms : bool
        Whether to center the atoms. Default is True.
    center_atoms_by_mass : bool
        If True, uses center of mass. If False, uses geometric center.
        Default is False (geometric center).
    center_point : tuple[float, float, float] | None
        Target center point. If None, centers at origin. Default is None.
    include_hydrogens : bool
        Whether to include hydrogen atoms in bonded atom ids. Default is True.
        If False, hydrogen atoms are excluded from the bonded element lists.
    load_bonded_environment : bool
        Whether to compute bonded atom ids and molecule types. Default is True.
        If False, bonded_environment and molecule_type columns will be None.
    """

    center_atoms: bool = True
    center_atoms_by_mass: bool = False
    center_point: tuple[float, float, float] | None = None
    include_hydrogens: bool = True
    load_bonded_environment: bool = True

    def __post_init__(self) -> None:
        """Validate option consistency after dataclass initialization.

        Ensures that center_atoms_by_mass and center_point are only used
        when center_atoms is True, and validates center_point format.
        """
        # Enforce consistency: center_atoms must be True to use these options.
        if not self.center_atoms:
            if self.center_atoms_by_mass:
                raise ValueError(
                    "center_atoms_by_mass=True requires center_atoms=True."
                )
            if self.center_point is not None:
                raise ValueError("center_point cannot be set when center_atoms=False.")

        # Optional: validate tuple length if provided
        if self.center_point is not None:
            if len(self.center_point) != 3:
                raise ValueError("center_point must be a 3-tuple of floats.")


def load_structure(
    file_path: str | pathlib.Path,
    options: StructureLoadOptions | None = None,
) -> pd.DataFrame:
    """Load molecular structure and return original mmdf dataframe.

    Returns DataFrame with computed bonding and molecule type information.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the PDB/mmCIF file.
    options : StructureLoadOptions | None
        Configuration options for loading. If None, uses default options.

    Returns
    -------
    pd.DataFrame
        DataFrame with original mmdf columns plus:
        - 'bonded_environment': Bonding environment strings like "C(CNO)"
        - 'molecule_type': Molecule type per atom ("protein" or "rna")
        - Updated 'x', 'y', 'z' columns if center_atoms is True
    """
    if options is None:
        options = StructureLoadOptions()

    # Load the base DataFrame
    df = load_df(file_path)

    # Track if we need to copy the DataFrame
    needs_copy = False

    # Add centered coordinates if requested
    if options.center_atoms:
        df = center_structure(
            df,
            center_point=options.center_point,
            use_center_of_mass=options.center_atoms_by_mass,
        )
        needs_copy = True

    # Add bonding information if requested
    if options.load_bonded_environment:
        bonded_ids, molecule_types = get_bonded_atom_ids_and_molecule_types(
            df=df, include_hydrogens=options.include_hydrogens
        )
        if not needs_copy:
            df = df.copy()
        df["bonded_environment"] = bonded_ids
        df["molecule_type"] = molecule_types
    else:
        if not needs_copy:
            df = df.copy()
        df["bonded_environment"] = None
        df["molecule_type"] = None

    return df
