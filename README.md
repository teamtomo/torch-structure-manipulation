# torch-structure-manipulation

[![License](https://img.shields.io/pypi/l/torch-structure-manipulation.svg?color=green)](https://github.com/teamtomo/torch-structure-manipulation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-structure-manipulation.svg?color=green)](https://pypi.org/project/torch-structure-manipulation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-structure-manipulation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-structure-manipulation/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-structure-manipulation)

A python package to extract bonding environments from cif files and perform basic structure transformation (centering at a desired point, translation, rotation, removing atoms within/outisde a radius, removing sidechains, separating proteins and RNA).


## Basix example usage

```python
import mmdf
from torch_structure_manipulation import FastCIFBondParser, FastAtomEnvironmentMapper

# Read structure
mmdf_df = mmdf.read('structure.cif')

# Extract covalent bonds
parser = FastCIFBondParser()
bonds_df = parser.extract_bonds_for_mmdf('structure.cif', mmdf_df)

# Map atom environments
env_mapper = FastAtomEnvironmentMapper()
mmdf_with_envs = env_mapper.map_environments(mmdf_df, bonds_df)
```

## Output

- Original mmdf DataFrame with added environment columns
- Edge index dataframe for all bonding
- Bonding environment

## Requirements

- Python 3.10+
- PyTorch
- pandas
- gemmi
- mmdf 

