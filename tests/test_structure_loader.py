"""Tests for structure_loader module."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from torch_structure_manipulation.structure_loader import (
    StructureLoadOptions,
    df_params_to_tensors,
    get_bonded_atom_ids_and_molecule_types,
    get_zyx_coords,
    load_df,
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
    return load_df(test_file)


@pytest.fixture(scope="session")
def loaded_structure_df(test_file):
    """Load structure with default options once for all tests."""
    return load_structure(test_file)


class TestLoadDf:
    """Tests for load_df function."""

    def test_load_cif_file(self, base_df):
        """Test loading a CIF file."""
        assert isinstance(base_df, pd.DataFrame)
        assert len(base_df) > 0
        assert "x" in base_df.columns
        assert "y" in base_df.columns
        assert "z" in base_df.columns
        assert "element" in base_df.columns


class TestGetZyxCoords:
    """Tests for get_zyx_coords function."""

    def test_extract_coordinates(self):
        """Test extracting coordinates from DataFrame."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        result = get_zyx_coords(df)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)
        expected = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        assert torch.allclose(result, expected.float())


class TestStructureLoadOptions:
    """Tests for StructureLoadOptions dataclass."""

    def test_default_options(self):
        """Test default options."""
        options = StructureLoadOptions()
        assert options.center_atoms is True
        assert options.center_atoms_by_mass is False
        assert options.center_point is None
        assert options.include_hydrogens is True
        assert options.load_bonded_environment is True

    def test_custom_options(self):
        """Test custom options."""
        options = StructureLoadOptions(
            center_atoms=False,
            include_hydrogens=False,
            load_bonded_environment=False,
        )
        assert options.center_atoms is False
        assert options.include_hydrogens is False
        assert options.load_bonded_environment is False

    def test_validation_center_atoms_by_mass(self):
        """Test validation when center_atoms_by_mass=True but center_atoms=False."""
        match_msg = "center_atoms_by_mass=True requires center_atoms=True"
        with pytest.raises(ValueError, match=match_msg):
            StructureLoadOptions(center_atoms=False, center_atoms_by_mass=True)

    def test_validation_center_point(self):
        """Test validation when center_point is set but center_atoms is False."""
        match_msg = "center_point cannot be set when center_atoms=False"
        with pytest.raises(ValueError, match=match_msg):
            StructureLoadOptions(center_atoms=False, center_point=(0.0, 0.0, 0.0))

    def test_validation_center_point_length(self):
        """Test validation of center_point tuple length."""
        with pytest.raises(ValueError, match="center_point must be a 3-tuple"):
            StructureLoadOptions(center_point=(0.0, 0.0))  # Only 2 elements


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

    def test_load_without_bonded_environment(self, test_file):
        """Test loading structure without bonded environment."""
        options = StructureLoadOptions(load_bonded_environment=False)
        df = load_structure(test_file, options=options)
        assert isinstance(df, pd.DataFrame)
        # Check that columns exist but are None
        assert "bonded_environment" in df.columns
        assert "molecule_type" in df.columns
        assert df["bonded_environment"].iloc[0] is None
        assert df["molecule_type"].iloc[0] is None

    def test_load_without_centering(self, test_file):
        """Test loading structure without centering."""
        options = StructureLoadOptions(center_atoms=False)
        df = load_structure(test_file, options=options)
        assert isinstance(df, pd.DataFrame)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns


class TestDfParamsToTensors:
    """Tests for df_params_to_tensors function."""

    def test_extract_all_params(self, loaded_structure_df):
        """Test extracting all parameters from DataFrame."""
        # Extract parameters
        atom_zyx, atom_id, atom_b_factor, atom_bonded_id, molecule_type = (
            df_params_to_tensors(loaded_structure_df)
        )

        assert isinstance(atom_zyx, torch.Tensor)
        assert atom_zyx.shape[1] == 3  # z, y, x
        assert isinstance(atom_id, list)
        assert len(atom_id) == len(loaded_structure_df)
        assert isinstance(atom_b_factor, torch.Tensor)
        assert isinstance(atom_bonded_id, list)
        assert isinstance(molecule_type, list)
        assert len(atom_bonded_id) == len(loaded_structure_df)
        assert len(molecule_type) == len(loaded_structure_df)


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
