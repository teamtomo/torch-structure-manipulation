"""Functions for centering molecular structures."""

import warnings

import pandas as pd
import torch

from .utils import df_to_atomzyx


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
