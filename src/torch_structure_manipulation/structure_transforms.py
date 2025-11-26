"""Module containing functions for transforming and analyzing molecular structures."""

import warnings

import numpy as np
import pandas as pd
import roma
import torch


def df_to_atomzyx(df: pd.DataFrame) -> torch.Tensor:
    """Extract atom coordinates from DataFrame as torch tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with z, y, x columns

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    """
    return torch.tensor(df[["z", "y", "x"]].values, dtype=torch.float32)


def center_structure(
    df: pd.DataFrame,
    center_point: tuple[float, float, float] | None = None,
    use_center_of_mass: bool = False,
    atom_selection: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Center DataFrame at specified point or origin.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with z, y, x coordinates
    center_point : tuple[float, float, float] | None
        Target center point in (z, y, x) order. If None, centers at origin.
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    atom_selection : pd.Series | None
        Boolean series to select atoms for center calculation

    Returns
    -------
    pd.DataFrame
        Centered DataFrame
    """
    if len(df) == 0:
        return df.copy()

    atomzyx = df_to_atomzyx(df)

    # Extract selection mask from DataFrame if provided
    selection_mask = None
    if atom_selection is not None:
        selection_mask = torch.tensor(atom_selection.values, dtype=torch.bool)

    # Extract masses from DataFrame if needed for center of mass
    masses = None
    if use_center_of_mass:
        if "atomic_weight" in df.columns:
            masses = torch.tensor(df["atomic_weight"].values, dtype=torch.float32)
        elif "atomic_number" in df.columns:
            masses = torch.tensor(df["atomic_number"].values, dtype=torch.float32)

    centered_atomzyx = center_structure_from_atomzyx(
        atomzyx, center_point, use_center_of_mass, selection_mask, masses
    )

    df_result = df.copy()
    df_result[["z", "y", "x"]] = centered_atomzyx.numpy()
    return df_result


def center_structure_from_atomzyx(
    atomzyx: torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
    use_center_of_mass: bool = False,
    selection_mask: torch.Tensor | None = None,
    masses: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center structure from atomzyx tensor to specified center point.

    Parameters
    ----------
    atomzyx : torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    center_point : tuple[float, float, float] | None
        Target center point in (z, y, x) order. If None, centers at origin.
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    selection_mask : torch.Tensor | None
        Boolean tensor to select atoms for center calculation
    masses : torch.Tensor | None
        Tensor of atomic masses for center of mass calculation.
        Required if use_center_of_mass is True.

    Returns
    -------
    torch.Tensor
        Centered atomzyx tensor
    """
    if selection_mask is not None:
        selected_coords = atomzyx[selection_mask]
        selected_masses = masses[selection_mask] if masses is not None else None
    else:
        selected_coords = atomzyx
        selected_masses = masses

    if center_point is None:
        target_center = torch.zeros(3, dtype=torch.float32, device=atomzyx.device)
    else:
        target_center = torch.tensor(
            center_point, dtype=torch.float32, device=atomzyx.device
        )

    current_center = calculate_center_from_tensors(
        selected_coords, use_center_of_mass, selected_masses
    )

    translation = target_center - current_center
    centered_atomzyx = atomzyx + translation

    return centered_atomzyx


def calculate_center_from_tensors(
    coords: torch.Tensor, use_center_of_mass: bool, masses: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate center from coordinates and masses tensors.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (n_atoms, 3) containing coordinates
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    masses : torch.Tensor | None
        Tensor of atomic masses. Required if use_center_of_mass is True.

    Returns
    -------
    torch.Tensor
        Center point as tensor of shape (3,)
    """
    if not use_center_of_mass:
        return torch.mean(coords, dim=0)

    if masses is None:
        # Fallback to geometric center if no masses provided
        warnings.warn(
            "use_center_of_mass is True but masses is None. "
            "Falling back to geometric center.",
            stacklevel=2,
        )
        return torch.mean(coords, dim=0)

    # Ensure masses are on same device as coords
    masses = masses.to(coords.device)

    total_mass = torch.sum(masses)
    if total_mass == 0:
        warnings.warn(
            "use_center_of_mass is True but total_mass is 0. "
            "Falling back to geometric center.",
            stacklevel=2,
        )
        return torch.mean(coords, dim=0)

    center_of_mass = torch.sum(coords * masses.unsqueeze(1), dim=0) / total_mass
    return center_of_mass


def apply_rotation(
    df: pd.DataFrame,
    rotation_matrix: np.ndarray | torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """Apply a rotation matrix to the structure.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with z, y, x coordinates
    rotation_matrix : np.ndarray | torch.Tensor
        3x3 rotation matrix (applied to z, y, x coordinates)
    center_point : tuple[float, float, float] | None
        Point to rotate around in (z, y, x) order. If None, rotates around origin

    Returns
    -------
    pd.DataFrame
        DataFrame with rotated coordinates
    """
    atomzyx = df_to_atomzyx(df)
    rotated_atomzyx = apply_rotation_to_atomzyx(atomzyx, rotation_matrix, center_point)

    df = df.copy()
    df[["z", "y", "x"]] = rotated_atomzyx.cpu().numpy()
    return df


def apply_rotation_to_atomzyx(
    atomzyx: torch.Tensor,
    rotation_matrix: np.ndarray | torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Apply a rotation matrix to atomzyx tensor.

    Parameters
    ----------
    atomzyx : torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    rotation_matrix : np.ndarray | torch.Tensor
        3x3 rotation matrix designed for (x, y, z) coordinates.
    center_point : tuple[float, float, float] | None
        Point to rotate around in (z, y, x) order. If None, rotates around origin

    Returns
    -------
    torch.Tensor
        Rotated atomzyx tensor
    """
    # Convert to torch tensor if needed
    if isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = torch.from_numpy(rotation_matrix).float()

    # Ensure rotation_matrix is on same device as atomzyx
    rotation_matrix = rotation_matrix.to(atomzyx.device)

    # Center coordinates if specified (in zyx)
    if center_point is not None:
        center = torch.tensor(center_point, dtype=torch.float32, device=atomzyx.device)
        atomzyx = atomzyx - center

    # Convert zyx coordinates to xyz for rotation
    # zyx: [z, y, x] -> xyz: [x, y, z]
    atomxyz = atomzyx[:, [2, 1, 0]]

    # Apply rotation (rotation_matrix is designed for xyz)
    rotated_xyz = torch.matmul(atomxyz, rotation_matrix.T)

    # Convert back from xyz to zyx
    # xyz: [x, y, z] -> zyx: [z, y, x]
    rotated_atomzyx = rotated_xyz[:, [2, 1, 0]]

    # Translate back if centered
    if center_point is not None:
        rotated_atomzyx = rotated_atomzyx + center

    return rotated_atomzyx


def create_rotation_matrix_from_euler(
    angles: torch.Tensor,
    order: str = "ZYZ",
    degrees: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create rotation matrix from Euler angles using roma.

    Parameters
    ----------
    angles : torch.Tensor
        Euler angles as a tensor.
        Shape can be (3,) for single rotation or (..., 3) for batch.
        The last dimension must be 3, corresponding to (alpha, beta, gamma).
    order : str
        Rotation order convention (e.g., 'xyz', 'zyx', 'ZYZ', 'zyz').
        Uppercase letters indicate intrinsic rotations.
        Lowercase indicate extrinsic rotations.
    degrees : bool
        If True, input angles are in degrees. If False, angles are in radians.
        Default is True (degrees).
    device : torch.device | None
        Device on which to perform computation.
        If None, uses the device of the input tensor.

    Returns
    -------
    torch.Tensor
        Rotation matrix.
        Shape is (3, 3) for single rotation or (..., 3, 3) for batch.
    """
    # Move to device if specified
    if device is not None:
        angles = angles.to(device)
    else:
        device = angles.device

    # Use roma to construct rotation matrix
    # roma uses uppercase for intrinsic, lowercase for extrinsic rotations
    rot_mat = roma.euler_to_rotmat(order, angles, degrees=degrees, device=device)

    return rot_mat


def apply_translation(
    df: pd.DataFrame,
    translation_vector: tuple[float, float, float] | np.ndarray | torch.Tensor,
) -> pd.DataFrame:
    """Apply a translation to the structure.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with z, y, x coordinates
    translation_vector : tuple[float, float, float] | np.ndarray | torch.Tensor
        Translation vector in (dz, dy, dx) order

    Returns
    -------
    pd.DataFrame
        DataFrame with translated coordinates
    """
    atomzyx = df_to_atomzyx(df)
    translated_atomzyx = apply_translation_to_atomzyx(atomzyx, translation_vector)

    df = df.copy()
    df[["z", "y", "x"]] = translated_atomzyx.cpu().numpy()
    return df


def apply_translation_to_atomzyx(
    atomzyx: torch.Tensor,
    translation_vector: tuple[float, float, float] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Apply a translation to atomzyx tensor.

    Parameters
    ----------
    atomzyx : torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    translation_vector : tuple[float, float, float] | np.ndarray | torch.Tensor
        Translation vector in (dz, dy, dx) order

    Returns
    -------
    torch.Tensor
        Translated atomzyx tensor
    """
    if isinstance(translation_vector, (tuple, list)):
        translation = torch.tensor(
            translation_vector, dtype=torch.float32, device=atomzyx.device
        )
    elif isinstance(translation_vector, np.ndarray):
        translation = torch.from_numpy(translation_vector).float().to(atomzyx.device)
    else:
        translation = translation_vector.to(atomzyx.device)

    return atomzyx + translation


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


def get_protein_residues() -> set[str]:
    """Get set of standard and non-standard amino acid residue names.

    Returns
    -------
    set[str]
        Set of amino acid residue names (uppercase).
    """
    return {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        # Non-standard amino acids
        "SEC",
        "PYL",
        "MSE",
        "HYP",
        "NLE",
    }


def get_nucleic_acid_residues() -> set[str]:
    """Get set of nucleic acid (RNA/DNA) residue names.

    Returns
    -------
    set[str]
        Set of nucleic acid residue names (uppercase).
    """
    return {
        "A",
        "T",
        "G",
        "C",
        "U",  # Standard bases
        "DA",
        "DT",
        "DG",
        "DC",  # DNA
        "ADE",
        "THY",
        "GUA",
        "CYT",
        "URA",  # Full names
        # Modified bases
        "PSU",
        "I",
        "M7G",
        "M2G",
        "M22G",
        "YYG",
        "H2U",
        "OMC",
        "OMG",
    }


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
