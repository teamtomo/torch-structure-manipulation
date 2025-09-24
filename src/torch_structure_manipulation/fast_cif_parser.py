import gemmi
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class FastCIFBondParser:
    """Parse covalent bonds from CIF files using gemmi."""
    
    def __init__(self):
        """Initialize bond order mapping."""
        self.bond_order_map = {
            'sing': 1.0,
            'doub': 2.0,
            'trip': 3.0,
            'quad': 4.0,
            'arom': 1.5,
            'delo': 0.5,
            'pi': 1.5
        }
    
    def extract_bonds_for_mmdf(self, cif_file: str, mmdf_df: pd.DataFrame) -> pd.DataFrame:
        """Extract covalent bonds from cif file and map to mmdf indices.
        
        Uses _chem_comp_bond for template bonds and _struct_conn for 
        inter-residue bonds, then expands templates to all residues.
        """
        try:
            doc = gemmi.cif.read_file(cif_file)
            block = doc.sole_block()
        except Exception as e:
            print(f"Error reading CIF: {e}")
            return pd.DataFrame(columns=['atom1_idx', 'atom2_idx', 'bond_order', 
                                       'atom1_element', 'atom2_element', 'aromatic_flag', 'stereo_config'])
        
        # Extract template bonds and inter-residue bonds
        chem_comp_bonds = self._extract_chem_comp_bonds(block)
        struct_conn_bonds = self._extract_struct_conn_bonds(block)
        
        # Expand templates to all residues in structure
        expanded_bonds = self._expand_template_bonds(chem_comp_bonds, mmdf_df)
        all_bonds = expanded_bonds + struct_conn_bonds
        
        if not all_bonds:
            return pd.DataFrame(columns=['atom1_idx', 'atom2_idx', 'bond_order', 
                                       'atom1_element', 'atom2_element', 'aromatic_flag', 'stereo_config'])
        
        cif_bonds = pd.DataFrame(all_bonds)
        return self._fast_map_bonds(mmdf_df, cif_bonds)
    
    def _fast_map_bonds(self, mmdf_df: pd.DataFrame, cif_bonds: pd.DataFrame) -> pd.DataFrame:
        """Map CIF bond identifiers to mmdf DataFrame indices."""
        atom_to_element = mmdf_df['element'].values
        
        # Direct atom name lookup
        atom_to_idx = {atom: idx for idx, atom in enumerate(mmdf_df['atom'])}
        
        # Composite key lookup (chain_residue_residueId_atom)
        chains = mmdf_df['chain'].values
        residues = mmdf_df['residue'].values
        residue_ids = mmdf_df['residue_id'].values
        atoms = mmdf_df['atom'].values
        
        composite_keys = [f"{chain}_{residue}_{residue_id}_{atom}" 
                         for chain, residue, residue_id, atom in zip(chains, residues, residue_ids, atoms)]
        composite_to_idx = {key: idx for idx, key in enumerate(composite_keys)}
        
        atom1_indices = []
        atom2_indices = []
        atom1_elements = []
        atom2_elements = []
        bond_orders = []
        aromatic_flags = []
        stereo_configs = []
        
        def find_atom_index(atom_id):
            """Try multiple strategies to find atom index."""
            # Strategy 1: Direct atom name lookup
            if atom_id in atom_to_idx:
                return atom_to_idx[atom_id]
            
            # Strategy 2: Composite key lookup
            if atom_id in composite_to_idx:
                return composite_to_idx[atom_id]
            
            return None
        
        for _, bond in cif_bonds.iterrows():
            atom1_id = bond['atom_id_1']
            atom2_id = bond['atom_id_2']
            
            idx1 = find_atom_index(atom1_id)
            idx2 = find_atom_index(atom2_id)
            
            if idx1 is not None and idx2 is not None:
                atom1_indices.append(idx1)
                atom2_indices.append(idx2)
                atom1_elements.append(atom_to_element[idx1])
                atom2_elements.append(atom_to_element[idx2])
                bond_orders.append(bond['bond_order'])
                aromatic_flags.append(bond['aromatic_flag'])
                stereo_configs.append(bond['stereo_config'])
        
        # Create result DataFrame
        result_bonds = pd.DataFrame({
            'atom1_idx': atom1_indices,
            'atom2_idx': atom2_indices,
            'bond_order': bond_orders,
            'atom1_element': atom1_elements,
            'atom2_element': atom2_elements,
            'aromatic_flag': aromatic_flags,
            'stereo_config': stereo_configs
        })
        
        print(f"Mapped {len(result_bonds)} bonds to mmdf indices")
        return result_bonds
    
    def create_bond_tensors(self, bonds_df: pd.DataFrame, n_atoms: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create tensors from bond DataFrame.
        
        Parameters
        ----------
        bonds_df : pd.DataFrame
            Bond DataFrame
        n_atoms : int
            Total number of atoms
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (edge_index, edge_attr) tensors
        """
        if len(bonds_df) == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3), dtype=torch.float32)
        
        # Convert to tensors
        atom1_indices = torch.from_numpy(bonds_df['atom1_idx'].values).long()
        atom2_indices = torch.from_numpy(bonds_df['atom2_idx'].values).long()
        
        # Create bidirectional edge index
        edge_index = torch.stack([
            torch.cat([atom1_indices, atom2_indices]),
            torch.cat([atom2_indices, atom1_indices])
        ])
        
        # Create edge attributes
        bond_orders = torch.from_numpy(bonds_df['bond_order'].values).float()
        aromatic_flags = torch.from_numpy(bonds_df['aromatic_flag'].values).float()
        stereo_flags = torch.tensor(
            [1.0 if x and x != 'N' else 0.0 for x in bonds_df['stereo_config'].fillna('N')], 
            dtype=torch.float32
        )
        
        edge_attr_single = torch.stack([bond_orders, aromatic_flags, stereo_flags], dim=1)
        edge_attr = torch.cat([edge_attr_single, edge_attr_single], dim=0)
        
        return edge_index, edge_attr
    
    def _extract_chem_comp_bonds(self, block) -> List[Dict]:
        """Extract template bonds from _chem_comp_bond table."""
        bonds = []
        
        for item in block:
            if hasattr(item, 'loop') and item.loop:
                if any('_chem_comp_bond' in tag for tag in item.loop.tags):
                    tags = item.loop.tags
                    comp_id_col = next(i for i, tag in enumerate(tags) if 'comp_id' in tag)
                    atom1_col = next(i for i, tag in enumerate(tags) if 'atom_id_1' in tag)
                    atom2_col = next(i for i, tag in enumerate(tags) if 'atom_id_2' in tag)
                    order_col = next(i for i, tag in enumerate(tags) if 'value_order' in tag)
                    aromatic_col = next((i for i, tag in enumerate(tags) if 'aromatic_flag' in tag), None)
                    stereo_col = next((i for i, tag in enumerate(tags) if 'stereo_config' in tag), None)
                    
                    for row_idx in range(item.loop.length()):
                        comp_id = item.loop[row_idx, comp_id_col]
                        atom1_id = item.loop[row_idx, atom1_col].strip('"')
                        atom2_id = item.loop[row_idx, atom2_col].strip('"')
                        value_order = item.loop[row_idx, order_col]
                        aromatic = item.loop[row_idx, aromatic_col] if aromatic_col is not None else 'N'
                        stereo = item.loop[row_idx, stereo_col] if stereo_col is not None else None
                        
                        bonds.append({
                            'comp_id': comp_id,
                            'atom_id_1': atom1_id,
                            'atom_id_2': atom2_id,
                            'bond_order': self.bond_order_map.get(value_order, 1.0),
                            'aromatic_flag': aromatic == 'Y',
                            'stereo_config': stereo
                        })
                    break
        
        return bonds
    
    def _extract_struct_conn_bonds(self, block) -> List[Dict]:
        """Extract inter-residue covalent bonds from _struct_conn table."""
        bonds = []
        
        for item in block:
            if hasattr(item, 'loop') and item.loop:
                if any('_struct_conn' in tag for tag in item.loop.tags):
                    tags = item.loop.tags
                    conn_type_col = next(i for i, tag in enumerate(tags) if 'conn_type_id' in tag)
                    
                    # First atom columns
                    p1_asym_col = next(i for i, tag in enumerate(tags) if 'ptnr1_label_asym_id' in tag)
                    p1_comp_col = next(i for i, tag in enumerate(tags) if 'ptnr1_label_comp_id' in tag)
                    p1_seq_col = next(i for i, tag in enumerate(tags) if 'ptnr1_label_seq_id' in tag)
                    p1_atom_col = next(i for i, tag in enumerate(tags) if 'ptnr1_label_atom_id' in tag)
                    
                    # second atom columns
                    p2_asym_col = next(i for i, tag in enumerate(tags) if 'ptnr2_label_asym_id' in tag)
                    p2_comp_col = next(i for i, tag in enumerate(tags) if 'ptnr2_label_comp_id' in tag)
                    p2_seq_col = next(i for i, tag in enumerate(tags) if 'ptnr2_label_seq_id' in tag)
                    p2_atom_col = next(i for i, tag in enumerate(tags) if 'ptnr2_label_atom_id' in tag)
                    
                    for row_idx in range(item.loop.length()):
                        conn_type = item.loop[row_idx, conn_type_col]
                        
                        # Only process covalent bonds
                        if conn_type == 'covale':
                            p1_asym = item.loop[row_idx, p1_asym_col]
                            p1_comp = item.loop[row_idx, p1_comp_col]
                            p1_seq = item.loop[row_idx, p1_seq_col]
                            p1_atom = item.loop[row_idx, p1_atom_col].strip('"')
                            
                            p2_asym = item.loop[row_idx, p2_asym_col]
                            p2_comp = item.loop[row_idx, p2_comp_col]
                            p2_seq = item.loop[row_idx, p2_seq_col]
                            p2_atom = item.loop[row_idx, p2_atom_col].strip('"')
                            
                            atom1_key = f"{p1_asym}_{p1_comp}_{p1_seq}_{p1_atom}"
                            atom2_key = f"{p2_asym}_{p2_comp}_{p2_seq}_{p2_atom}"
                            
                            bonds.append({
                                'comp_id': f"{p1_comp}-{p2_comp}",
                                'atom_id_1': atom1_key,
                                'atom_id_2': atom2_key,
                                'bond_order': 1.0,
                                'aromatic_flag': False,
                                'stereo_config': None
                            })
                    break
        
        return bonds
    
    def _expand_template_bonds(self, template_bonds: List[Dict], mmdf_df: pd.DataFrame) -> List[Dict]:
        """Expand template bonds to all matching residues in the structure."""
        expanded = []
        
        # Group templates by component
        templates_by_comp = {}
        for bond in template_bonds:
            comp_id = bond['comp_id']
            if comp_id not in templates_by_comp:
                templates_by_comp[comp_id] = []
            templates_by_comp[comp_id].append(bond)
        
        # Apply templates to each residue in the structure
        for comp_id, comp_templates in templates_by_comp.items():
            # Find all residues of this type
            matching_residues = mmdf_df[mmdf_df['residue'] == comp_id]
            residue_groups = matching_residues.groupby(['chain', 'residue_id'])
            
            for (chain, residue_id), residue_atoms in residue_groups:
                # Apply each template bond to this residue
                for template in comp_templates:
                    atom1_name = template['atom_id_1']
                    atom2_name = template['atom_id_2']
                    
                    # Create full atom keys
                    atom1_key = f"{chain}_{comp_id}_{residue_id}_{atom1_name}"
                    atom2_key = f"{chain}_{comp_id}_{residue_id}_{atom2_name}"
                    
                    expanded.append({
                        'comp_id': comp_id,
                        'atom_id_1': atom1_key,
                        'atom_id_2': atom2_key,
                        'bond_order': template['bond_order'],
                        'aromatic_flag': template['aromatic_flag'],
                        'stereo_config': template['stereo_config']
                    })
        
        return expanded
