import torch
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union, List
import warnings


class StructureTransforms:
    """Structure transformations using PyTorch tensors."""
    
    def __init__(self):
        pass
    
    def center_structure(self, 
                        df: pd.DataFrame, 
                        center_point: Optional[Tuple[float, float, float]] = None,
                        use_center_of_mass: bool = False,
                        atom_selection: Optional[pd.Series] = None) -> pd.DataFrame:
        """Center structure at specified point or origin."""
        if len(df) == 0:
            return df.copy()
            
        df_result = df.copy()
        
        coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)
        
        if atom_selection is not None:
            selection_mask = torch.tensor(atom_selection.values, dtype=torch.bool)
            selected_coords = coords[selection_mask]
            selected_df = df[atom_selection]
        else:
            selected_coords = coords
            selected_df = df
        
        target_center = torch.zeros(3, dtype=torch.float32) if center_point is None else \
                       torch.tensor(center_point, dtype=torch.float32)
        
        current_center = self._calculate_center_vectorized(selected_coords, selected_df, use_center_of_mass)
        
        translation = target_center - current_center
        coords += translation
        
        df_result[['x', 'y', 'z']] = coords.numpy()
        
        return df_result
    
    def apply_rotation(self, 
                      df: pd.DataFrame, 
                      rotation_matrix: Union[np.ndarray, torch.Tensor],
                      center_point: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """Apply a rotation matrix to the structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Structure DataFrame with x, y, z coordinates
        rotation_matrix : Union[np.ndarray, torch.Tensor]
            3x3 rotation matrix
        center_point : Optional[Tuple[float, float, float]]
            Point to rotate around. If None, rotates around origin
            
        Returns
        -------
        pd.DataFrame
            DataFrame with rotated coordinates
        """
        df = df.copy()
        
        # Convert to torch tensor if needed
        if isinstance(rotation_matrix, np.ndarray):
            rotation_matrix = torch.from_numpy(rotation_matrix).float()
        
        # Get coordinates
        coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)
        
        # Center coordinates if specified
        if center_point is not None:
            center = torch.tensor(center_point, dtype=torch.float32)
            coords = coords - center
        
        # Apply rotation
        rotated_coords = torch.matmul(coords, rotation_matrix.T)
        
        # Translate back if centered
        if center_point is not None:
            rotated_coords = rotated_coords + center
        
        # Update DataFrame
        df[['x', 'y', 'z']] = rotated_coords.cpu().numpy()
        
        return df
    
    def apply_translation(self, 
                         df: pd.DataFrame, 
                         translation_vector: Union[Tuple[float, float, float], np.ndarray, torch.Tensor]) -> pd.DataFrame:
        """Apply a translation to the structure.
        
        Parameters
        ----------
        df : pd.DataFrame
            Structure DataFrame with x, y, z coordinates
        translation_vector : Union[Tuple[float, float, float], np.ndarray, torch.Tensor]
            Translation vector (dx, dy, dz)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with translated coordinates
        """
        df = df.copy()
        
        if isinstance(translation_vector, (tuple, list)):
            translation_vector = np.array(translation_vector)
        elif isinstance(translation_vector, torch.Tensor):
            translation_vector = translation_vector.cpu().numpy()
        
        df[['x', 'y', 'z']] += translation_vector
        
        return df
    
    def remove_atoms_by_radius(self, 
                              df: pd.DataFrame, 
                              center_point: Tuple[float, float, float],
                              radius: float,
                              keep_inside: bool = True) -> pd.DataFrame:
        """Remove atoms within or outside a specified radius.
        
        Parameters
        ----------
        df : pd.DataFrame
            Structure DataFrame
        center_point : Tuple[float, float, float]
            Center point for radius calculation
        radius : float
            Radius in Angstroms
        keep_inside : bool
            If True, keeps atoms inside the radius. If False, keeps atoms outside.
            
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        if len(df) == 0:
            return df.copy()
        
        coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32, )
        center = torch.tensor(center_point, dtype=torch.float32, )
        
        distances = torch.norm(coords - center, dim=1)
        mask = (distances <= radius) if keep_inside else (distances > radius)
        
        return df[mask.cpu().numpy()].copy()
    
    def remove_sidechains(self, df: pd.DataFrame, keep_backbone_atoms: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove sidechain atoms, keeping only backbone atoms.
        
        Parameters
        ----------
        df : pd.DataFrame
            Structure DataFrame with 'atom' column containing atom names
        keep_backbone_atoms : Optional[List[str]]
            List of atom names to keep. If None, uses standard protein backbone atoms.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with only backbone atoms
        """
        if keep_backbone_atoms is None:
            # Standard protein backbone atoms
            keep_backbone_atoms = ['N', 'CA', 'C', 'O', 'H', 'HA']
            # Add nucleic acid backbone atoms
            keep_backbone_atoms.extend(["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"])
        
        # Filter atoms
        backbone_mask = df['atom'].isin(keep_backbone_atoms)
        
        return df[backbone_mask].copy()
    
    def separate_protein_rna(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate protein and RNA/DNA components.
        
        Parameters
        ----------
        df : pd.DataFrame
            Structure DataFrame with 'residue' column
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (protein_df, nucleic_acid_df)
        """
        # Standard amino acid residues
        amino_acids = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            # Non-standard amino acids
            'SEC', 'PYL', 'MSE', 'HYP', 'NLE'
        }
        
        # Nucleic acid residues
        nucleic_acids = {
            'A', 'T', 'G', 'C', 'U',  # Standard bases
            'DA', 'DT', 'DG', 'DC',   # DNA
            'ADE', 'THY', 'GUA', 'CYT', 'URA',  # Full names
            # Modified bases
            'PSU', 'I', 'M7G', 'M2G', 'M22G', 'YYG', 'H2U', 'OMC', 'OMG'
        }
        
        # Separate based on residue names
        protein_mask = df['residue'].isin(amino_acids)
        nucleic_mask = df['residue'].isin(nucleic_acids)
        
        protein_df = df[protein_mask].copy()
        nucleic_df = df[nucleic_mask].copy()
        
        # Handle remaining residues (warn if significant amount)
        remaining = df[~(protein_mask | nucleic_mask)]
        if len(remaining) > 0:
            warnings.warn(f"Found {len(remaining)} atoms in {remaining['residue'].nunique()} "
                         f"unrecognized residue types: {set(remaining['residue'].unique())}")
        
        return protein_df, nucleic_df
    
    
    def _calculate_center_of_mass(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate center of mass using atomic masses."""
        coords = df[['x', 'y', 'z']].values
        
        # Use atomic weights if available, otherwise use atomic numbers
        if 'atomic_weight' in df.columns:
            masses = df['atomic_weight'].values
        elif 'atomic_number' in df.columns:
            masses = df['atomic_number'].values
        else:
            # Fallback to equal masses (geometric center)
            masses = np.ones(len(df))
        
        total_mass = np.sum(masses)
        center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        
        return center_of_mass
    
    def _calculate_center_vectorized(self, coords: torch.Tensor, df: pd.DataFrame, 
                                    use_center_of_mass: bool) -> torch.Tensor:
        """Calculate center."""
        if not use_center_of_mass:
            return torch.mean(coords, dim=0)
        
        # Vectorized mass assignment and center of mass calculation
        if 'atomic_weight' in df.columns:
            masses = torch.tensor(df['atomic_weight'].values, dtype=torch.float32)
        elif 'atomic_number' in df.columns:
            masses = torch.tensor(df['atomic_number'].values, dtype=torch.float32)
        else:
            # Fallback to geometric center for equal masses
            return torch.mean(coords, dim=0)
        
        # Single vectorized center of mass calculation
        total_mass = torch.sum(masses)
        if total_mass == 0:
            return torch.mean(coords, dim=0)
        
        center_of_mass = torch.sum(coords * masses.unsqueeze(1), dim=0) / total_mass
        return center_of_mass
    
    def _calculate_geometric_center(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate geometric center (centroid)."""
        coords = df[['x', 'y', 'z']].values
        return np.mean(coords, axis=0)
    
    def create_rotation_matrix_from_euler(self, 
                                        angles: Tuple[float, float, float],
                                        order: str = 'xyz') -> torch.Tensor:
        """Create rotation matrix from Euler angles.
        
        Parameters
        ----------
        angles : Tuple[float, float, float]
            Euler angles in radians (alpha, beta, gamma)
        order : str
            Rotation order (e.g., 'xyz', 'zyx')
            
        Returns
        -------
        torch.Tensor
            3x3 rotation matrix
        """
        alpha, beta, gamma = angles
        
        # Convert angles to tensors
        alpha_t = torch.tensor(alpha, dtype=torch.float32, )
        beta_t = torch.tensor(beta, dtype=torch.float32, )
        gamma_t = torch.tensor(gamma, dtype=torch.float32, )
        
        # Create individual rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(alpha_t), -torch.sin(alpha_t)],
            [0, torch.sin(alpha_t), torch.cos(alpha_t)]
        ], dtype=torch.float32, )
        
        Ry = torch.tensor([
            [torch.cos(beta_t), 0, torch.sin(beta_t)],
            [0, 1, 0],
            [-torch.sin(beta_t), 0, torch.cos(beta_t)]
        ], dtype=torch.float32, )
        
        Rz = torch.tensor([
            [torch.cos(gamma_t), -torch.sin(gamma_t), 0],
            [torch.sin(gamma_t), torch.cos(gamma_t), 0],
            [0, 0, 1]
        ], dtype=torch.float32, )
        
        # Combine rotations based on order
        rotation_matrices = {'x': Rx, 'y': Ry, 'z': Rz}
        
        R = torch.eye(3, dtype=torch.float32, )
        for axis in order.lower():
            R = torch.matmul(R, rotation_matrices[axis])
        
        return R
