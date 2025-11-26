"""Functions for translating molecular structures."""

import numpy as np
import pandas as pd
import torch

from .utils import df_to_atomzyx


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
