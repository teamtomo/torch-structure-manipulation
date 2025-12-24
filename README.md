# torch-structure-manipulation

[![License](https://img.shields.io/pypi/l/torch-structure-manipulation.svg?color=green)](https://github.com/teamtomo/torch-structure-manipulation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-structure-manipulation.svg?color=green)](https://pypi.org/project/torch-structure-manipulation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-structure-manipulation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-structure-manipulation/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-structure-manipulation)

A python package to extract bonding environments from cif/pdb files and perform basic structure transformation (centering at a desired point, translation, rotation, removing atoms within/outside a radius, removing sidechains, separating proteins and RNA).


## Basic example usage

### Loading a structure

```python
from torch_structure_manipulation.structure_loader import load_structure, StructureLoadOptions

# Load structure with default options (centered, with bonding info)
df = load_structure('structure.cif')

# Or customize loading options
options = StructureLoadOptions(
    center_atoms=True,
    center_atoms_by_mass=False,  # Use geometric center
    center_point=(0.0, 0.0, 0.0),  # Center at origin
    include_hydrogens=True,
    load_bonded_environment=True,
)
df = load_structure('structure.cif', options=options)

# The DataFrame includes original mmdf columns plus:
print(f"Bonded environments: {df['bonded_environment'].head()}")
print(f"Molecule types: {df['molecule_type'].unique()}")
# Example output:
# Bonded environments: 0    C(CAO)
#                      1    N(CA)
#                      2    C(HHCN)
#                      ...
# Molecule types: ['protein' 'rna']
```

### Structure transformations

All transformation functions support both `zyx` (z, y, x) and `xyz` (x, y, z) coordinate systems via the `zyx` parameter (default: `True` for zyx). When `zyx=True`, DataFrames must have `z`, `y`, `x` columns and coordinate tuples are in (z, y, x) order. When `zyx=False`, DataFrames must have `x`, `y`, `z` columns and coordinate tuples are in (x, y, z) order.

```python
from torch_structure_manipulation.structure_transforms import (
    find_atoms_in_ball,
    ball_query_atoms,
    center_structure,
    apply_rotation,
    apply_translation,
    separate_protein_rna,
)

# Select atoms within a ball (sphere) from origin
# Using zyx coordinates (default)
inside, outside = find_atoms_in_ball(
    df, center=(0.0, 0.0, 0.0), radius=50.0, zyx=True
)

# Using xyz coordinates
inside, outside = find_atoms_in_ball(
    df, center=(0.0, 0.0, 0.0), radius=50.0, zyx=False
)

# Get boolean mask for easier composition of multiple queries
inside_mask = ball_query_atoms(df, center=(0.0, 0.0, 0.0), radius=50.0, zyx=True)
print(f"Atoms inside radius: {len(inside)}")
print(f"Atoms outside radius: {len(outside)}")

# Center structure at a specific point (zyx coordinates, default)
centered_df = center_structure(
    df, center_point=(10.0, 20.0, 30.0), use_center_of_mass=False, zyx=True
)

# Apply translation (zyx coordinates, default)
# Translation vector in (dz, dy, dx) order when zyx=True
translated_df = apply_translation(df, translation=(1.0, 2.0, 3.0), zyx=True)

# Apply rotation (zyx coordinates, default)
# Rotation matrix is designed for (x, y, z) coordinates
import torch
from torch_structure_manipulation.structure_transforms import create_rotation_matrix_from_euler
angles = torch.tensor([0.0, 0.0, 90.0])
rotation_matrix = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
rotated_df = apply_rotation(df, rotation_matrix, zyx=True)

# Separate protein and RNA components
protein_df, rna_df = separate_protein_rna(df)
```

## Requirements

- Python 3.10+
- PyTorch
- pandas
- gemmi
- mmdf 

