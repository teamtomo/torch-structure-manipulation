"""Tests for structure_transforms module."""

import numpy as np
import pandas as pd
import pytest
import torch

from torch_structure_manipulation.structure_transforms import (
    apply_rotation,
    apply_rotation_to_atomzyx,
    apply_translation,
    apply_translation_to_atomzyx,
    calculate_center_from_tensors,
    center_structure,
    center_structure_from_atomzyx,
    create_rotation_matrix_from_euler,
    df_to_atomzyx,
    get_nucleic_acid_residues,
    get_protein_residues,
    remove_sidechains,
    return_atoms_by_radius,
    return_atoms_by_radius_from_atomzyx,
    separate_protein_rna,
)


class TestDfToAtomzyx:
    """Tests for df_to_atomzyx function."""

    def test_basic_extraction(self):
        """Test basic coordinate extraction."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        result = df_to_atomzyx(df)
        expected = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        assert torch.allclose(result, expected.float())


class TestCenterStructure:
    """Tests for center_structure functions."""

    def test_center_at_origin(self):
        """Test centering at origin."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        result = center_structure(df)
        centered_coords = result[["z", "y", "x"]].values
        # Mean should be at origin
        assert np.allclose(centered_coords.mean(axis=0), [0, 0, 0])

    def test_center_at_specific_point(self):
        """Test centering at a specific point."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        center_point = (2.0, 5.0, 8.0)  # (z, y, x)
        result = center_structure(df, center_point=center_point)
        centered_coords = result[["z", "y", "x"]].values
        # Mean should be at center_point
        assert np.allclose(centered_coords.mean(axis=0), center_point)

    def test_center_structure_from_atomzyx(self):
        """Test centering from atomzyx tensor."""
        atomzyx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = center_structure_from_atomzyx(atomzyx)
        # Mean should be at origin
        assert torch.allclose(result.mean(dim=0), torch.zeros(3))

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["z", "y", "x"])
        result = center_structure(df)
        assert len(result) == 0


class TestCalculateCenterFromTensors:
    """Tests for calculate_center_from_tensors function."""

    def test_geometric_center(self):
        """Test geometric center calculation."""
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        center = calculate_center_from_tensors(coords, use_center_of_mass=False)
        expected = torch.tensor([4.0, 5.0, 6.0])  # Mean of coordinates
        assert torch.allclose(center, expected)

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        masses = torch.tensor([1.0, 2.0, 3.0])
        center = calculate_center_from_tensors(
            coords, use_center_of_mass=True, masses=masses
        )
        # Weighted average: (0*1 + 1*2 + 2*3) / 6 = 8/6 = 4/3
        expected = torch.tensor([4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0])
        assert torch.allclose(center, expected)


class TestRotation:
    """Tests for rotation functions."""

    def test_create_rotation_matrix_90_deg_z(self):
        """Test 90-degree rotation around z-axis."""
        angles = torch.tensor([0.0, 0.0, 90.0])  # ZYZ order
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        # 90 deg rotation around z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert torch.allclose(R, expected, atol=1e-6)

    def test_apply_rotation_to_atomzyx(self):
        """Test rotation application to atomzyx coordinates."""
        # Create 90-degree rotation around z-axis
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        # Point at [0, 1, 0] in zyx (z=0, y=1, x=0)
        # In xyz: [0, 1, 0], after 90 deg z rotation: [-1, 0, 0]
        # Back to zyx: [0, 0, -1]
        atomzyx = torch.tensor([[0.0, 1.0, 0.0]])
        rotated = apply_rotation_to_atomzyx(atomzyx, R)
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_apply_rotation_with_center(self):
        """Test rotation around a center point."""
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        atomzyx = torch.tensor([[1.0, 1.0, 0.0]])
        center_point = (1.0, 1.0, 0.0)  # Rotate around itself
        rotated = apply_rotation_to_atomzyx(atomzyx, R, center_point=center_point)
        # Should return to original position
        assert torch.allclose(rotated, atomzyx, atol=1e-5)

    def test_apply_rotation_dataframe(self):
        """Test rotation on DataFrame."""
        df = pd.DataFrame(
            {
                "z": [0.0, 1.0],
                "y": [1.0, 0.0],
                "x": [0.0, 0.0],
            }
        )
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        result = apply_rotation(df, R)
        # Check that coordinates changed
        original_coords = df[["z", "y", "x"]].values
        rotated_coords = result[["z", "y", "x"]].values
        assert not np.allclose(rotated_coords, original_coords)


class TestTranslation:
    """Tests for translation functions."""

    def test_apply_translation_to_atomzyx(self):
        """Test translation of atomzyx coordinates."""
        atomzyx = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        translation = (1.0, 2.0, 3.0)  # (dz, dy, dx)
        result = apply_translation_to_atomzyx(atomzyx, translation)
        expected = torch.tensor([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
        assert torch.allclose(result, expected)

    def test_apply_translation_dataframe(self):
        """Test translation on DataFrame."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0],
                "y": [3.0, 4.0],
                "x": [5.0, 6.0],
            }
        )
        translation = (1.0, 1.0, 1.0)  # (dz, dy, dx)
        result = apply_translation(df, translation)
        expected = df[["z", "y", "x"]].values + np.array([1.0, 1.0, 1.0])
        result_coords = result[["z", "y", "x"]].values
        assert np.allclose(result_coords, expected)


class TestReturnAtomsByRadius:
    """Tests for return_atoms_by_radius functions."""

    def test_return_atoms_by_radius_from_atomzyx(self):
        """Test radius filtering from atomzyx tensor."""
        atomzyx = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        center_point = (0.0, 0.0, 0.0)  # (z, y, x)
        radius = 1.5
        inside_mask, outside_mask = return_atoms_by_radius_from_atomzyx(
            atomzyx, center_point, radius
        )
        # First two points should be inside, last two outside
        assert inside_mask.tolist() == [True, True, False, False]
        assert outside_mask.tolist() == [False, False, True, True]

    def test_return_atoms_by_radius_dataframe(self):
        """Test radius filtering on DataFrame."""
        df = pd.DataFrame(
            {
                "z": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "x": [0.0, 0.0, 0.0, 0.0],
                "element": ["C", "C", "C", "C"],
            }
        )
        center_point = (0.0, 0.0, 0.0)
        radius = 1.5
        inside_df, outside_df = return_atoms_by_radius(df, center_point, radius)
        assert len(inside_df) == 2
        assert len(outside_df) == 2


class TestRemoveSidechains:
    """Tests for remove_sidechains function."""

    def test_remove_sidechains_default(self):
        """Test removing sidechains with default backbone atoms."""
        df = pd.DataFrame(
            {
                "atom": ["N", "CA", "C", "O", "CB", "CG"],
                "z": [0.0] * 6,
                "y": [0.0] * 6,
                "x": [0.0] * 6,
            }
        )
        result = remove_sidechains(df)
        # Should keep N, CA, C, O, but not CB, CG
        assert len(result) == 4
        assert set(result["atom"]) == {"N", "CA", "C", "O"}

    def test_remove_sidechains_custom(self):
        """Test removing sidechains with custom backbone atoms."""
        df = pd.DataFrame(
            {
                "atom": ["N", "CA", "C", "O", "CB"],
                "z": [0.0] * 5,
                "y": [0.0] * 5,
                "x": [0.0] * 5,
            }
        )
        keep_atoms = ["N", "CA", "CB"]
        result = remove_sidechains(df, keep_backbone_atoms=keep_atoms)
        assert len(result) == 3
        assert set(result["atom"]) == {"N", "CA", "CB"}


class TestResidueFunctions:
    """Tests for residue-related functions."""

    def test_get_protein_residues(self):
        """Test getting protein residues."""
        residues = get_protein_residues()
        assert isinstance(residues, set)
        assert "ALA" in residues
        assert "GLY" in residues
        assert "MSE" in residues  # Non-standard amino acid

    def test_get_nucleic_acid_residues(self):
        """Test getting nucleic acid residues."""
        residues = get_nucleic_acid_residues()
        assert isinstance(residues, set)
        assert "A" in residues
        assert "U" in residues
        assert "DA" in residues  # DNA variant

    def test_separate_protein_rna(self):
        """Test separating protein and RNA components."""
        df = pd.DataFrame(
            {
                "residue": ["ALA", "GLY", "A", "U", "UNK"],
                "z": [0.0] * 5,
                "y": [0.0] * 5,
                "x": [0.0] * 5,
            }
        )
        with pytest.warns(UserWarning, match="unrecognized residue types"):
            protein_df, nucleic_df = separate_protein_rna(df)
        assert len(protein_df) == 2
        assert len(nucleic_df) == 2
        assert set(protein_df["residue"]) == {"ALA", "GLY"}
        assert set(nucleic_df["residue"]) == {"A", "U"}
