"""Module for loading molecular structures with bond information.

This module provides an API that returns a pandas DataFrame with computed
bonding and molecule type information.
"""

import pandas as pd

from .load_structure_utils import get_bonded_atom_ids_and_molecule_types


def load_structure(
    df: pd.DataFrame,
    include_hydrogens: bool = True,
) -> pd.DataFrame:
    """Add bonding and molecule type information to a structure DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with mmdf columns (x, y, z, element, residue, etc.)
    include_hydrogens : bool, default=True
        Whether to include hydrogen atoms in bonded atom ids. If False,
        hydrogen atoms are excluded from the bonded element lists.

    Returns
    -------
    pd.DataFrame
        DataFrame with original mmdf columns plus:
        - 'bonded_environment': Bonding environment strings like "C(CNO)"
        - 'molecule_type': Molecule type per atom ("protein" or "rna")
    """
    df = df.copy()

    # Add bonding information
    bonded_ids, molecule_types = get_bonded_atom_ids_and_molecule_types(
        df=df, include_hydrogens=include_hydrogens
    )
    df["bonded_environment"] = bonded_ids
    df["molecule_type"] = molecule_types

    return df
