# torch-structure-manipulation

[![License](https://img.shields.io/pypi/l/torch-structure-manipulation.svg?color=green)](https://github.com/teamtomo/torch-structure-manipulation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-structure-manipulation.svg?color=green)](https://pypi.org/project/torch-structure-manipulation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-structure-manipulation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-structure-manipulation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-structure-manipulation/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-structure-manipulation)

A python package to extract bonding environments from cif files and perform a few basic structure transformation (centering at a desired point, translation, rotation, removing atoms within/outisde a radius, removing sidechains, separating proteins and RNA) 

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork teamtomo/torch-structure-manipulation --clone
# or just
# gh repo clone teamtomo/torch-structure-manipulation
cd torch-structure-manipulation
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
