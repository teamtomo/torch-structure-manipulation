"""Tests for structure_loader module."""

from pathlib import Path

import mmdf
import pandas as pd
import pytest

from torch_structure_manipulation.structure_loader import (
    get_bonded_atom_ids_and_molecule_types,
    load_structure,
)


@pytest.fixture(scope="session")
def test_file():
    """Get the test file path."""
    file_path = Path(__file__).parent / "4V6X.cif"
    if not file_path.exists():
        pytest.skip("Test file 4V6X.cif not found")
    return file_path


@pytest.fixture(scope="session")
def base_df(test_file):
    """Load base DataFrame once for all tests."""
    return mmdf.read(test_file)


@pytest.fixture(scope="session")
def loaded_structure_df(base_df):
    """Load structure with default options once for all tests."""
    return load_structure(base_df)


class TestLoadStructure:
    """Tests for load_structure function."""

    def test_load_with_defaults(self, loaded_structure_df):
        """Test loading structure with default options."""
        assert isinstance(loaded_structure_df, pd.DataFrame)
        assert len(loaded_structure_df) > 0
        assert "bonded_environment" in loaded_structure_df.columns
        assert "molecule_type" in loaded_structure_df.columns
        assert loaded_structure_df["bonded_environment"].iloc[0] is not None
        assert loaded_structure_df["molecule_type"].iloc[0] is not None

    def test_load_with_include_hydrogens_false(self, base_df):
        """Test loading structure without hydrogens in bonded environment."""
        df = load_structure(base_df, include_hydrogens=False)
        assert isinstance(df, pd.DataFrame)
        assert "bonded_environment" in df.columns
        assert "molecule_type" in df.columns
        # Check that no hydrogen atoms appear in bonded IDs
        for bid in df["bonded_environment"]:
            if "H" in bid:
                # H should only appear as the central atom, not in the bonded list
                bonded_elements = bid.split("(")[1].rstrip(")") if "(" in bid else ""
                assert not bid.startswith("H(") or "H" not in bonded_elements

    def test_load_preserves_original_columns(self, base_df):
        """Test that load_structure preserves original DataFrame columns."""
        original_columns = set(base_df.columns)
        df = load_structure(base_df)
        # Should have all original columns plus new ones
        assert original_columns.issubset(set(df.columns))
        assert "bonded_environment" in df.columns
        assert "molecule_type" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns


class TestGetBondedAtomIdsAndMoleculeTypes:
    """Tests for get_bonded_atom_ids_and_molecule_types function."""

    def test_compute_bonded_ids(self, base_df):
        """Test computing bonded atom IDs."""
        bonded_ids, molecule_types = get_bonded_atom_ids_and_molecule_types(
            base_df, include_hydrogens=True
        )

        assert isinstance(bonded_ids, list)
        assert isinstance(molecule_types, list)
        assert len(bonded_ids) == len(base_df)
        assert len(molecule_types) == len(base_df)
        assert all(isinstance(bid, str) for bid in bonded_ids)
        assert all(mt in ("protein", "rna") for mt in molecule_types)

    def test_without_hydrogens(self, base_df):
        """Test computing bonded IDs without hydrogens."""
        bonded_ids, _ = get_bonded_atom_ids_and_molecule_types(
            base_df, include_hydrogens=False
        )

        assert isinstance(bonded_ids, list)
        assert len(bonded_ids) == len(base_df)
        # Check that no hydrogen atoms appear in bonded IDs
        for bid in bonded_ids:
            if "H" in bid:
                # H should only appear as the central atom, not in the bonded list
                bonded_elements = bid.split("(")[1].rstrip(")") if "(" in bid else ""
                assert not bid.startswith("H(") or "H" not in bonded_elements
