import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict


class FastAtomEnvironmentMapper:
    """Map atom environments based on covalent connectivity."""
    
    def __init__(self):
        pass
    
    def map_environments(self, mmdf_df: pd.DataFrame, bonds_df: pd.DataFrame) -> pd.DataFrame:
        """Map atom environments based on bonded neighbors.
        
        Returns mmdf DataFrame with added environment columns:
        environment_id, coordination_number, bonded_elements.
        """
        n_atoms = len(mmdf_df)
        if n_atoms == 0:
            return mmdf_df.copy()
        
        if len(bonds_df) == 0:
            # No bonds - all atoms are isolated
            environments = [f"{elem}()" for elem in mmdf_df['element']]
            coordination = [0] * n_atoms
            bonded_elements = [[]] * n_atoms
        else:
            environments, coordination, bonded_elements = self._fast_vectorized_environments(
                mmdf_df, bonds_df, n_atoms
            )
        
        # Add environment data efficiently
        env_data = pd.DataFrame({
            'environment_id': environments,
            'coordination_number': coordination,
            'bonded_elements': bonded_elements
        })
        
        result_df = pd.concat([mmdf_df.reset_index(drop=True), env_data], axis=1)
        return result_df
    
    def _fast_vectorized_environments(self, mmdf_df: pd.DataFrame, bonds_df: pd.DataFrame, n_atoms: int):
        """Ultra-fast environment calculation using PyTorch tensors."""
        # Convert to tensors for speed
        atom1_indices = torch.from_numpy(bonds_df['atom1_idx'].values).long()
        atom2_indices = torch.from_numpy(bonds_df['atom2_idx'].values).long()
        elements = mmdf_df['element'].values
        
        # Create bidirectional adjacency using scatter operations
        all_src = torch.cat([atom1_indices, atom2_indices])
        all_dst = torch.cat([atom2_indices, atom1_indices])
        
        # Use scatter to count neighbors efficiently
        coordination_counts = torch.zeros(n_atoms, dtype=torch.long)
        coordination_counts.scatter_add_(0, all_src, torch.ones_like(all_src))
        
        # Build adjacency lists efficiently using sorting
        sorted_indices = torch.argsort(all_src)
        sorted_src = all_src[sorted_indices]
        sorted_dst = all_dst[sorted_indices]
        
        # Split into per-atom neighbor lists
        split_sizes = coordination_counts.tolist()
        neighbor_lists = torch.split(sorted_dst, split_sizes)
        
        # Generate environment strings efficiently
        environments = []
        coordination = coordination_counts.numpy().tolist()
        bonded_elements_list = []
        
        for i in range(n_atoms):
            central_element = elements[i]
            
            if coordination[i] == 0:
                environments.append(f"{central_element}()")
                bonded_elements_list.append([])
            else:
                # Get neighbor elements
                neighbor_indices = neighbor_lists[i].numpy()
                neighbor_elements = [elements[j] for j in neighbor_indices]
                neighbor_elements.sort()  # Sort for consistent representation
                
                bonded_str = ''.join(neighbor_elements)
                environments.append(f"{central_element}({bonded_str})")
                bonded_elements_list.append(neighbor_elements)
        
        return environments, coordination, bonded_elements_list
    
    def get_environment_statistics(self, df_with_envs: pd.DataFrame) -> pd.DataFrame:
        """Fast environment statistics using vectorized operations."""
        if len(df_with_envs) == 0:
            return pd.DataFrame(columns=['environment_id', 'count', 'percentage'])
        
        total_atoms = len(df_with_envs)
        
        # Use pandas value_counts for speed
        env_counts = df_with_envs['environment_id'].value_counts()
        
        stats = pd.DataFrame({
            'environment_id': env_counts.index,
            'count': env_counts.values,
            'percentage': (env_counts.values / total_atoms * 100).round(2)
        })
        
        return stats.reset_index(drop=True)
