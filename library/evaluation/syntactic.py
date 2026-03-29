from typing import List

import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

rdkit.RDLogger.DisableLog("rdApp.*")


def clean_design(smiles: str) -> str:
    """
    Cleans a given SMILES string by performing the following steps:
    1. Converts the SMILES string to a molecule object using RDKit.
    2. Removes any charges from the molecule.
    3. Sanitizes the molecule by checking for any errors or inconsistencies.
    4. Converts the sanitized molecule back to a canonical SMILES string.

    Parameters
    ----------
    smiles : str
        A SMILES design that possibly represents a chemical compound.

    Returns
    -------
    str
        A cleaned and canonicalized SMILES string representing a chemical compound.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    sanitization_flag = Chem.SanitizeMol(
        mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
    )
    # SANITIZE_NONE is the "no error" flag of rdkit!
    if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    if can_smiles is None or len(can_smiles) == 0:
        return None

    return can_smiles


def get_valid_designs(design_list: List[str]) -> List[str]:
    """
    Filters a list of SMILES strings to only keep the valid ones.
    Applies the `clean_design` function to each SMILES string in the list.
    So, uncharging, sanitization, and canonicalization are performed on each SMILES string.

    Parameters
    ----------
    design_list : List[str]
        A list of SMILES designs representing chemical compounds.

    Returns
    -------
    List[str]
        A list of valid SMILES strings representing chemical compounds.
    """
    cleaned_designs = [clean_design(design) for design in design_list]
    return [design for design in cleaned_designs if design is not None]
