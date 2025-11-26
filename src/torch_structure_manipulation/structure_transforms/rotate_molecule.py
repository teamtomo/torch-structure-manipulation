"""Functions for rotating molecular structures."""

import numpy as np
import pandas as pd
import roma
import torch

from .utils import df_to_atomzyx


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
