"""Functions for selecting and filtering atoms."""

import warnings

import pandas as pd
import torch

from .utils import df_to_atomzyx, get_nucleic_acid_residues, get_protein_residues


def return_atoms_by_radius(
    df: pd.DataFrame, center_point: tuple[float, float, float], radius: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return atoms inside and outside a specified radius.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with z, y, x coordinates
    center_point : tuple[float, float, float]
        Center point for radius calculation in (z, y, x) order
    radius : float
        Radius in Angstroms

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (atoms_inside, atoms_outside) DataFrames
    """
    if len(df) == 0:
        empty_df = df.copy()
        return empty_df, empty_df

    atomzyx = df_to_atomzyx(df)
    inside_mask, outside_mask = return_atoms_by_radius_from_atomzyx(
        atomzyx, center_point, radius
    )

    atoms_inside = df[inside_mask.cpu().numpy()].copy()
    atoms_outside = df[outside_mask.cpu().numpy()].copy()

    return atoms_inside, atoms_outside


def return_atoms_by_radius_from_atomzyx(
    atomzyx: torch.Tensor, center_point: tuple[float, float, float], radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for atoms inside and outside a specified radius from atomzyx tensor.

    Parameters
    ----------
    atomzyx : torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    center_point : tuple[float, float, float]
        Center point for radius calculation in (z, y, x) order
    radius : float
        Radius in Angstroms

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple of (inside_mask, outside_mask) boolean tensors
    """
    center = torch.tensor(center_point, dtype=torch.float32, device=atomzyx.device)

    distances = torch.norm(atomzyx - center, dim=1)
    inside_mask = distances <= radius
    outside_mask = distances > radius

    return inside_mask, outside_mask


def remove_sidechains(
    df: pd.DataFrame, keep_backbone_atoms: list[str] | None = None
) -> pd.DataFrame:
    """Remove sidechain atoms, keeping only backbone atoms.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with 'atom' column containing atom names
    keep_backbone_atoms : List[str] | None
        List of atom names to keep. If None, uses standard protein backbone atoms.

    Returns
    -------
    pd.DataFrame
        DataFrame with only backbone atoms
    """
    if keep_backbone_atoms is None:
        # Standard protein backbone atoms
        keep_backbone_atoms = ["N", "CA", "C", "O", "H", "HA", "OXT"]
        # Add nucleic acid backbone atoms
        keep_backbone_atoms.extend(
            ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
        )

    # Filter atoms
    backbone_mask = df["atom"].isin(keep_backbone_atoms)

    return df[backbone_mask].copy()


def separate_protein_rna(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate protein and RNA/DNA components.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with 'residue' column

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (protein_df, nucleic_acid_df)
    """
    amino_acids = get_protein_residues()
    nucleic_acids = get_nucleic_acid_residues()

    # Separate based on residue names
    protein_mask = df["residue"].isin(amino_acids)
    nucleic_mask = df["residue"].isin(nucleic_acids)

    protein_df = df[protein_mask].copy()
    nucleic_df = df[nucleic_mask].copy()

    # Handle remaining residues (warn if significant amount)
    remaining = df[~(protein_mask | nucleic_mask)]
    if len(remaining) > 0:
        warnings.warn(
            f"Found {len(remaining)} atoms in {remaining['residue'].nunique()} "
            f"unrecognized residue types: {set(remaining['residue'].unique())}",
            stacklevel=2,
        )

    return protein_df, nucleic_df
