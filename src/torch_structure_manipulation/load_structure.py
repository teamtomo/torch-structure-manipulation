"""Module for loading molecular structures with bond information."""

import json
import pathlib

import mmdf
import pandas as pd
import torch

from .structure_transforms import (
    center_structure_from_atomzyx,
    get_nucleic_acid_residues,
    get_protein_residues,
)


def _load_bonding_data() -> tuple[
    dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]
]:
    """
    Load protein and RNA bonding dictionaries from JSON file.

    Returns
    -------
    tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]
        Tuple of (protein_bonding, rna_bonding).
        Each is a dictionary with residue names as keys and atom names as keys.
        The values are lists of bonded atom names.
    """
    json_path = pathlib.Path(__file__).parent / "bonding_data.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["protein"], data["rna"]


# Load bonding dictionaries from JSON
_PROTEIN_BONDING, _RNA_BONDING = _load_bonding_data()


def load_df(file_path: str | pathlib.Path) -> pd.DataFrame:
    """
    Load a pdb/mmcif file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the pdb file.

    Returns
    -------
    pd.DataFrame
        DataFrame read from pdb/mmcif file.
    """
    return mmdf.read(file_path)


def load_model_bonds(
    file_path: str | pathlib.Path,
    center_atoms: bool = True,
    center_atoms_by_mass: bool = False,
    center_point: tuple[float, float, float] | None = None,
    include_hydrogens: bool = True,
    load_bonded_environment: bool = True,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str] | None, list[str] | None]:
    """Pdb/cif file to atom coordinates, ids, B factors, bonded atoms, molecule types.

    Loads a pdb from `file_path` and returns the atom coordinates (in Angstroms),
    atom ids as a list of strings, B factors (in Angstroms^2), bonded atom ids
    in format like "C(HHCN)", and molecule type per atom ('protein' or 'rna').

    O(C) bonds are categorized as:
    - O(C, carboxyl) for carboxyl groups (sidechain in ASP/GLU, C-terminal OXT)
    - O(C, amide) for backbone O in peptide bonds
    - O(C) for all other O(C) bonds

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the pdb file.
    center_atoms : bool
        Whether to center the atoms. Default is True.
    center_atoms_by_mass : bool
        If True, uses center of mass. If False, uses geometric center.
        Default is False (geometric center).
    center_point : tuple[float, float, float] | None
        Target center point. If None, centers at origin. Default is None.
    include_hydrogens : bool
        Whether to include hydrogen atoms in bonded atom ids. Default is True.
        If False, hydrogen atoms are excluded from the bonded element lists.
    load_bonded_environment : bool
        Whether to compute bonded atom ids and molecule types. Default is True.
        If False, returns None for bonded_id and molecule_type.

    Returns
    -------
    tuple[torch.Tensor, list[str], torch.Tensor, list[str] | None, list[str] | None]
        Atom coordinates, atom ids, B factors, bonded atom ids (or None),
        molecule type per atom (or None).
    """
    df = load_df(file_path)
    atom_zyx = torch.tensor(df[["z", "y", "x"]].to_numpy()).float()  # (n_atoms, 3)

    if center_atoms:
        # Extract masses if needed for center of mass
        masses = None
        if center_atoms_by_mass:
            if "atomic_weight" in df.columns:
                masses = torch.tensor(df["atomic_weight"].values, dtype=torch.float32)
            elif "atomic_number" in df.columns:
                masses = torch.tensor(df["atomic_number"].values, dtype=torch.float32)

        # Center using structure_transforms function (now uses z, y, x order)
        atom_zyx = center_structure_from_atomzyx(
            atom_zyx,
            center_point=center_point,
            use_center_of_mass=center_atoms_by_mass,
            masses=masses,
        )

    atom_id = df["element"].str.upper().tolist()
    atom_b_factor = torch.tensor(df["b_isotropic"].to_numpy()).float()

    if load_bonded_environment:
        # Get bonded atom ids and molecule types using lookup tables
        atom_bonded_id, molecule_type = _get_bonded_atom_ids_and_molecule_types(
            df=df, include_hydrogens=include_hydrogens
        )
    else:
        atom_bonded_id = None
        molecule_type = None

    return atom_zyx, atom_id, atom_b_factor, atom_bonded_id, molecule_type


def _get_bonded_atom_ids_and_molecule_types(
    df: pd.DataFrame, include_hydrogens: bool = True
) -> tuple[list[str], list[str]]:
    """
    Get bonded atom IDs and molecule types using lookup tables.

    This function efficiently computes both bonded IDs and molecule types, combining
    multiple operations into fewer loops and using faster pandas operations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with atom data from mmdf.read()
    include_hydrogens : bool
        Whether to include hydrogen atoms in bonded atom ids.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of (bonded_ids, molecule_types).
        Each is a list with one entry per atom.
    """
    # Pre-compute residue ordering for inter-residue bonds
    next_residue, prev_residue = _build_residue_order(df)

    # Pre-extract columns to lists for faster access
    # This avoids repeated getattr calls in itertuples
    residue_col = df["residue"].str.upper().tolist()
    atom_col = df["atom"].str.strip().tolist()
    element_col = df["element"].str.upper().str.strip().tolist()
    residue_id_col = df["residue_id"].astype(str).tolist()
    chain_col = df["chain"].astype(str).tolist()

    # Build residue lookup and identify peptide bonds
    residue_lookup, peptide_bond_c_atoms = _build_residue_lookup_and_peptide_bonds(
        df, residue_col, atom_col, element_col, residue_id_col, chain_col, next_residue
    )

    # Compute molecule types using vectorized pandas operations
    molecule_types = _compute_molecule_types(df)

    # Pre-compute bonded atom names cache
    # Many atoms share the same (residue, atom_name) pairs, so cache the results
    bonded_names_cache: dict[tuple[str, str], list[str]] = {}

    # Compute bonded IDs for all atoms
    bonded_ids = []
    for i in range(len(df)):
        bonded_id = _compute_bonded_id_for_atom(
            i,
            residue_col,
            atom_col,
            element_col,
            residue_id_col,
            chain_col,
            residue_lookup,
            next_residue,
            prev_residue,
            peptide_bond_c_atoms,
            bonded_names_cache,
            include_hydrogens,
        )
        bonded_ids.append(bonded_id)

    return bonded_ids, molecule_types


def _build_residue_lookup_and_peptide_bonds(
    df: pd.DataFrame,
    residue_col: list[str],
    atom_col: list[str],
    element_col: list[str],
    residue_id_col: list[str],
    chain_col: list[str],
    next_residue: dict[tuple[str, str], str],
) -> tuple[dict[tuple[str, str], dict[str, str]], set[tuple[str, str, str]]]:
    """
    Build residue lookup dictionary and identify peptide bond C atoms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with atom data.
    residue_col : list[str]
        Pre-extracted residue column values.
    atom_col : list[str]
        Pre-extracted atom column values.
    element_col : list[str]
        Pre-extracted element column values.
    residue_id_col : list[str]
        Pre-extracted residue_id column values.
    chain_col : list[str]
        Pre-extracted chain column values.
    next_residue : dict[tuple[str, str], str]
        Dictionary mapping (chain, residue_id) to next residue_id.

    Returns
    -------
    tuple[dict[tuple[str, str], dict[str, str]], set[tuple[str, str, str]]]
        Tuple of (residue_lookup, peptide_bond_c_atoms).
        residue_lookup maps (chain, residue_id) to {atom_name: element}.
        peptide_bond_c_atoms is a set of (chain, residue_id, atom_name) tuples.
    """
    residue_lookup: dict[tuple[str, str], dict[str, str]] = {}
    peptide_bond_c_atoms: set[tuple[str, str, str]] = set()

    for i in range(len(df)):
        residue_id = residue_id_col[i]
        chain = chain_col[i]
        atom_name = atom_col[i]
        element = element_col[i]
        residue = residue_col[i]

        key = (chain, residue_id)
        if key not in residue_lookup:
            residue_lookup[key] = {}
        residue_lookup[key][atom_name] = element

        # Identify peptide bonds on-the-fly (C atoms bonded to N of next residue)
        if residue in _PROTEIN_BONDING and atom_name == "C":
            next_residue_id = next_residue.get((chain, residue_id))
            if next_residue_id:
                next_residue_atoms = residue_lookup.get((chain, next_residue_id), {})
                n_element = _find_atom_element("N", next_residue_atoms)
                if n_element:
                    peptide_bond_c_atoms.add((chain, residue_id, "C"))

    return residue_lookup, peptide_bond_c_atoms


def _compute_molecule_types(df: pd.DataFrame) -> list[str]:
    """
    Compute molecule type for each atom based on residue.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'residue' column.

    Returns
    -------
    list[str]
        List of molecule types, one per atom ('protein' or 'rna').
    """
    protein_residues = get_protein_residues()
    rna_residues = get_nucleic_acid_residues()

    residue_upper = df["residue"].str.upper()
    molecule_type_map: dict[str, str] = {}
    for res in rna_residues:
        molecule_type_map[res] = "rna"
    for res in protein_residues:
        molecule_type_map[res] = "protein"
    # Use map with default 'protein' for unknown residues
    result: list[str] = residue_upper.map(molecule_type_map).fillna("protein").tolist()
    return result


def _get_intra_residue_bonded_elements(
    bonded_atom_names: list[str],
    residue_atoms: dict[str, str],
    include_hydrogens: bool,
) -> list[str]:
    """
    Get bonded elements from atoms within the same residue.

    Parameters
    ----------
    bonded_atom_names : list[str]
        List of bonded atom names from lookup table.
    residue_atoms : dict[str, str]
        Dictionary of atom names to elements in the residue.
    include_hydrogens : bool
        Whether to include hydrogen atoms.

    Returns
    -------
    list[str]
        List of bonded element symbols.
    """
    bonded_elements = []
    for bonded_atom_name in bonded_atom_names:
        # Handle atom name variations (e.g., "O5'" vs "O5*")
        bonded_element = _find_atom_element(bonded_atom_name, residue_atoms)

        if bonded_element:
            # Atom exists in PDB - add it (filtering by include_hydrogens)
            if include_hydrogens or bonded_element != "H":
                bonded_elements.append(bonded_element)
        else:
            # Atom doesn't exist in PDB - check if it's an expected hydrogen
            if _is_hydrogen_atom_name(bonded_atom_name):
                # This is a hydrogen atom that should be bonded but isn't in PDB
                # Include it if include_hydrogens is True
                if include_hydrogens:
                    bonded_elements.append("H")

    return bonded_elements


def _format_bonded_id(
    element: str,
    bonded_elements: list[str],
    peptide_bond_c_atoms: set[tuple[str, str, str]],
    residue: str,
    atom_name: str,
    chain: str,
    residue_id: str,
) -> str:
    """
    Format bonded ID string from element and bonded elements.

    Parameters
    ----------
    element : str
        Element symbol of the central atom.
    bonded_elements : list[str]
        List of bonded element symbols.
    peptide_bond_c_atoms : set[tuple[str, str, str]]
        Set of (chain, residue_id, atom_name) tuples for peptide bond C atoms.
    residue : str
        Residue name.
    atom_name : str
        Atom name.
    chain : str
        Chain identifier.
    residue_id : str
        Residue ID.

    Returns
    -------
    str
        Formatted bonded ID string (e.g., "C(HHCN)" or "O(C, carboxyl)").
    """
    # Sort bonded elements for consistent representation
    bonded_elements_sorted = sorted(bonded_elements)
    bonded_str = "".join(bonded_elements_sorted)

    # Special handling for O(C) bonds: categorize as carboxyl, amide, or other
    if element == "O" and bonded_str == "C":
        o_category = _categorize_o_c_bond(
            residue, atom_name, chain, residue_id, peptide_bond_c_atoms
        )
        if o_category:
            return f"{element}({bonded_str}, {o_category})"
        return f"{element}({bonded_str})"
    return f"{element}({bonded_str})"


def _compute_bonded_id_for_atom(
    i: int,
    residue_col: list[str],
    atom_col: list[str],
    element_col: list[str],
    residue_id_col: list[str],
    chain_col: list[str],
    residue_lookup: dict[tuple[str, str], dict[str, str]],
    next_residue: dict[tuple[str, str], str],
    prev_residue: dict[tuple[str, str], str],
    peptide_bond_c_atoms: set[tuple[str, str, str]],
    bonded_names_cache: dict[tuple[str, str], list[str]],
    include_hydrogens: bool,
) -> str:
    """
    Compute bonded ID for a single atom.

    Parameters
    ----------
    i : int
        Index of the atom in the DataFrame.
    residue_col : list[str]
        Pre-extracted residue column values.
    atom_col : list[str]
        Pre-extracted atom column values.
    element_col : list[str]
        Pre-extracted element column values.
    residue_id_col : list[str]
        Pre-extracted residue_id column values.
    chain_col : list[str]
        Pre-extracted chain column values.
    residue_lookup : dict[tuple[str, str], dict[str, str]]
        Dictionary mapping (chain, residue_id) to {atom_name: element}.
    next_residue : dict[tuple[str, str], str]
        Dictionary mapping (chain, residue_id) to next residue_id.
    prev_residue : dict[tuple[str, str], str]
        Dictionary mapping (chain, residue_id) to previous residue_id.
    peptide_bond_c_atoms : set[tuple[str, str, str]]
        Set of (chain, residue_id, atom_name) tuples for peptide bond C atoms.
    bonded_names_cache : dict[tuple[str, str], list[str]]
        Cache for bonded atom names lookups.
    include_hydrogens : bool
        Whether to include hydrogen atoms.

    Returns
    -------
    str
        Formatted bonded ID string.
    """
    residue = residue_col[i]
    atom_name = atom_col[i]
    element = element_col[i]
    residue_id = residue_id_col[i]
    chain = chain_col[i]

    # Get bonded atom names from lookup table (intra-residue)
    # Use cache to avoid repeated lookups
    cache_key = (residue, atom_name)
    if cache_key not in bonded_names_cache:
        bonded_names_cache[cache_key] = _get_bonded_atom_names(residue, atom_name)
    bonded_atom_names = bonded_names_cache[cache_key]

    # Look up elements of bonded atoms from the same residue
    residue_atoms = residue_lookup.get((chain, residue_id), {})
    bonded_elements = _get_intra_residue_bonded_elements(
        bonded_atom_names, residue_atoms, include_hydrogens
    )

    # Add inter-residue bonds
    inter_residue_elements = _get_inter_residue_bonded_elements_from_values(
        residue=residue,
        atom_name=atom_name,
        residue_id=residue_id,
        chain=chain,
        residue_lookup=residue_lookup,
        next_residue=next_residue,
        prev_residue=prev_residue,
        include_hydrogens=include_hydrogens,
    )
    bonded_elements.extend(inter_residue_elements)

    # Format the bonded ID
    return _format_bonded_id(
        element,
        bonded_elements,
        peptide_bond_c_atoms,
        residue,
        atom_name,
        chain,
        residue_id,
    )


def _categorize_o_c_bond(
    residue: str,
    atom_name: str,
    chain: str,
    residue_id: str,
    peptide_bond_c_atoms: set[tuple[str, str, str]],
) -> str | None:
    """
    Categorize an O(C) bond as 'carboxyl' or 'amide', or return None for other.

    Parameters
    ----------
    residue : str
        Residue name (e.g., 'ASP', 'GLU', 'ALA')
    atom_name : str
        Atom name (e.g., 'O', 'OXT', 'OD1', 'OD2', 'OE1', 'OE2')
    chain : str
        Chain identifier
    residue_id : str
        Residue ID
    peptide_bond_c_atoms : set[tuple[str, str, str]]
        Set of (chain, residue_id, atom_name) tuples for C atoms in peptide bonds

    Returns
    -------
    str | None
        'carboxyl', 'amide', or None
    """
    # C-terminal oxygen (OXT) is carboxyl
    if atom_name == "OXT":
        return "carboxyl"

    # Sidechain carboxyl groups
    if residue == "ASP" and atom_name in ("OD1", "OD2"):
        return "carboxyl"
    if residue == "GLU" and atom_name in ("OE1", "OE2"):
        return "carboxyl"

    # Backbone O in peptide bond (amide)
    if atom_name == "O" and residue in _PROTEIN_BONDING:
        if (chain, residue_id, "C") in peptide_bond_c_atoms:
            return "amide"

    # Everything else
    return None


def _is_hydrogen_atom_name(atom_name: str) -> bool:
    """
    Check if an atom name represents a hydrogen atom.

    Parameters
    ----------
    atom_name : str
        Atom name to check.

    Returns
    -------
    bool
        True if the atom name represents a hydrogen atom, False otherwise.
    """
    # Common hydrogen atom name patterns
    atom_name_upper = atom_name.upper()
    # Single H, or starts with H followed by a letter/number
    if atom_name_upper == "H" or atom_name_upper.startswith("H"):
        # Exclude heavy atoms that might start with H (like HE, HG in some contexts)
        # But include common hydrogen patterns
        if len(atom_name_upper) == 1:
            return True
        # Check for common hydrogen patterns: HA, HB, HG, HD, HE, H1, H2, etc.
        if atom_name_upper in ["HA", "HB", "HG", "HD", "HE", "HZ", "HH"]:
            return True
        # Pattern like H1, H2, H3, H11, H12, etc.
        if (
            atom_name_upper[1:].isdigit()
            or atom_name_upper[1:].replace("'", "").isdigit()
        ):
            return True
        # Pattern like H5', H5'', H2', etc. (with quotes)
        if "'" in atom_name_upper or "''" in atom_name_upper:
            if atom_name_upper.startswith("H"):
                return True
    return False


def _find_atom_element(
    atom_name: str,
    residue_atoms: dict[str, str],
) -> str:
    """
    Find element for an atom name, handling variations.

    Parameters
    ----------
    atom_name : str
        Atom name to find the element for.
    residue_atoms : dict[str, str]
        Dictionary of atom names and their elements.

    Returns
    -------
    str
        Element of the atom name.
        If the atom name is not found, returns an empty string.
    """
    # Direct lookup
    if atom_name in residue_atoms:
        return residue_atoms[atom_name]

    # Try without quotes/stars
    atom_name_clean = atom_name.replace("'", "").replace("*", "")
    for key, element in residue_atoms.items():
        if key.replace("'", "").replace("*", "") == atom_name_clean:
            return element

    return ""


def _build_residue_order(
    df: pd.DataFrame,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Build mapping of (chain, residue_id) -> next/prev_residue_id for inter-res bonds.

    Only considers residues that are sequentially adjacent (residue_id differs by 1)
    to avoid false bonds across chain breaks or missing residues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with chain and residue_id columns.

    Returns
    -------
    tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]
        Tuple of (next_residue, prev_residue).
        Each is a dictionary with:
        (chain, residue_id) as keys
        the next/previous residue_id as values.
    """
    # Group by chain and sort by residue_id
    next_residue = {}
    prev_residue = {}
    for chain in df["chain"].unique():
        chain_str = str(chain)  # Ensure chain is a string for consistent key matching
        chain_df = df[df["chain"] == chain]
        # Convert residue_ids to numeric for proper sorting and comparison
        residue_ids_numeric = []
        residue_id_map = {}  # numeric -> original string

        for res_id in chain_df["residue_id"].unique():
            try:
                # Try to convert to int/float
                numeric_id = float(res_id)
                residue_ids_numeric.append(numeric_id)
                residue_id_map[numeric_id] = str(res_id)
            except (ValueError, TypeError):
                # If not numeric, skip inter-residue bond inference
                continue

        residue_ids_numeric = sorted(residue_ids_numeric)

        for i, numeric_id in enumerate(residue_ids_numeric):
            residue_id = residue_id_map[numeric_id]
            residue_key = (chain_str, residue_id)

            # Only add if next residue is sequential (differs by 1)
            if i < len(residue_ids_numeric) - 1:
                next_numeric = residue_ids_numeric[i + 1]
                if abs(next_numeric - numeric_id) <= 1.1:  # Allow 1.0 or 1 (int/float)
                    next_residue[residue_key] = residue_id_map[next_numeric]

            # Only add if previous residue is sequential (differs by 1)
            if i > 0:
                prev_numeric = residue_ids_numeric[i - 1]
                if abs(numeric_id - prev_numeric) <= 1.1:  # Allow 1.0 or 1 (int/float)
                    prev_residue[residue_key] = residue_id_map[prev_numeric]

    return next_residue, prev_residue


def _get_inter_residue_bonded_elements_from_values(
    residue: str,
    atom_name: str,
    residue_id: str,
    chain: str,
    residue_lookup: dict[tuple[str, str], dict[str, str]],
    next_residue: dict[tuple[str, str], str],
    prev_residue: dict[tuple[str, str], str],
    include_hydrogens: bool,
) -> list[str]:
    """Get bonded elements from adjacent residues (inter-residue bonds).

    This version takes pre-extracted values for better performance.

    Parameters
    ----------
    residue : str
        Residue name (e.g., 'ASP', 'GLU', 'ALA')
    atom_name : str
        Atom name (e.g., 'O', 'OXT', 'OD1', 'OD2', 'OE1', 'OE2')
    residue_id : str
        Residue ID
    chain : str
        Chain identifier
    residue_lookup : dict[tuple[str, str], dict[str, str]]
        Dictionary of residue names and their atoms.
    next_residue : dict[tuple[str, str], str]
        Dictionary of next residue ids.
    prev_residue : dict[tuple[str, str], str]
        Dictionary of previous residue ids.
    include_hydrogens : bool
        Whether to include hydrogen atoms in bonded elements.

    Returns
    -------
    list[str]
        List of bonded elements.
    """
    bonded_elements = []

    # Protein: C (carbonyl) bonds to N (amide) of next residue
    if residue in _PROTEIN_BONDING and atom_name == "C":
        next_residue_id = next_residue.get((chain, residue_id))
        if next_residue_id:
            next_residue_atoms = residue_lookup.get((chain, next_residue_id), {})
            n_element = _find_atom_element("N", next_residue_atoms)
            if n_element and (include_hydrogens or n_element != "H"):
                bonded_elements.append(n_element)

    # Protein: N (amide) bonds to C (carbonyl) of previous residue
    if residue in _PROTEIN_BONDING and atom_name == "N":
        prev_residue_id = prev_residue.get((chain, residue_id))
        if prev_residue_id:
            prev_residue_atoms = residue_lookup.get((chain, prev_residue_id), {})
            c_element = _find_atom_element("C", prev_residue_atoms)
            if c_element and (include_hydrogens or c_element != "H"):
                bonded_elements.append(c_element)

    # RNA: O3' bonds to P of next residue (if not terminal)
    # If terminal, O3' will bond to HO3' instead (handled by lookup table)
    if residue in _RNA_BONDING and atom_name == "O3'":
        next_residue_id = next_residue.get((chain, residue_id))
        if next_residue_id:
            # Not terminal - bond to P of next residue
            next_residue_atoms = residue_lookup.get((chain, next_residue_id), {})
            p_element = _find_atom_element("P", next_residue_atoms)
            if p_element and (include_hydrogens or p_element != "H"):
                bonded_elements.append(p_element)
        # If terminal (no next_residue_id), HO3' will be added from lookup table

    # RNA: P bonds to O3' of previous residue (if not first residue)
    # If terminal/last residue, P will bond to OP3 instead (handled by lookup table)
    if residue in _RNA_BONDING and atom_name == "P":
        prev_residue_id = prev_residue.get((chain, residue_id))
        if prev_residue_id:
            # Not first residue - bond to O3' of previous residue
            prev_residue_atoms = residue_lookup.get((chain, prev_residue_id), {})
            o3_element = _find_atom_element("O3'", prev_residue_atoms)
            if o3_element and (include_hydrogens or o3_element != "H"):
                bonded_elements.append(o3_element)
        # If first residue or terminal (no prev_residue_id) then
        # OP3 will be added from lookup table if present

    return bonded_elements


def _get_bonded_atom_names(residue: str, atom_name: str) -> list[str]:
    """
    Get bonded atom names for a given residue and atom name.

    Parameters
    ----------
    residue : str
        Residue name (e.g., 'ASP', 'GLU', 'ALA')
    atom_name : str
        Atom name (e.g., 'O', 'OXT', 'OD1', 'OD2', 'OE1', 'OE2')

    Returns
    -------
    list[str]
        List of bonded atom names.
    """
    # Check if it's a protein residue
    if residue in _PROTEIN_BONDING:
        if atom_name in _PROTEIN_BONDING[residue]:
            return _PROTEIN_BONDING[residue][atom_name]

    # Check if it's an RNA residue
    if residue in _RNA_BONDING:
        if atom_name in _RNA_BONDING[residue]:
            return _RNA_BONDING[residue][atom_name]

    # Fallback: return empty list if not found
    return []


def determine_molecule_type(df: pd.DataFrame) -> str:
    """
    Determine if the structure is protein or RNA based on residues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'residue' column containing residue names.

    Returns
    -------
    str
        'protein' or 'rna' or 'rna+protein'.
    """
    protein_residues = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

    rna_residues = {"A", "U", "G", "C", "DA", "DT", "DG", "DC"}

    unique_residues = set(df["residue"].str.upper().unique())

    has_protein = bool(unique_residues & protein_residues)
    has_rna = bool(unique_residues & rna_residues)

    if has_protein and has_rna:
        return "rna+protein"
    if has_protein:
        return "protein"
    if has_rna:
        return "rna"
    # Default to protein if unknown
    return "protein"


def get_atom_molecule_types(df: pd.DataFrame) -> list[str]:
    """Determine molecule type for each atom based on its residue.

    Returns 'rna' if the atom is part of an RNA/DNA nucleotide,
    'protein' if part of an amino acid, or 'protein' as default.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'residue' column containing residue names.

    Returns
    -------
    list[str]
        List of molecule types, one per atom. Each is either 'rna' or 'protein'.
    """
    protein_residues = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

    rna_residues = {"A", "U", "G", "C", "DA", "DT", "DG", "DC"}

    molecule_types = []
    for _, row in df.iterrows():
        residue = row.get("residue", "").upper()
        if residue in rna_residues:
            molecule_types.append("rna")
        elif residue in protein_residues:
            molecule_types.append("protein")
        else:
            # Default to protein if unknown
            molecule_types.append("protein")

    return molecule_types
