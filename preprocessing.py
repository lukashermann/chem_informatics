import numpy as np
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from tqdm import tqdm

from dimorphite_dl.dimorphite_dl import DimorphiteDL
from rdkit import Chem

from rdkit.Chem import AddHs, Kekulize, MolToSmiles, MolFromSmiles
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds, _CalcCrippenContribs


def apply_filters(mol):
    """
    Apply filters to exclude compounds with less than four atoms, more than nine rings, or more than forty
    rotatable bonds.

    Args:
        mol: Molecule

    Returns:
        True if molecule passes filters, False otherwise.

    """
    if mol.GetNumAtoms() < 4:
        return False
    if CalcNumRings(mol) > 9:
        return False
    if CalcNumRotatableBonds(mol) > 40:
        return False
    return True


def set_protonation(mol):
    """
    Set protonation states at pH 7.4

    Args:
        mol: Molecule.

    Returns:
        Molecule with set protonation states.
    """
    dimorphite = DimorphiteDL(min_ph=7.4, max_ph=7.4, pka_precision=0)
    protonated_mols_smiles = dimorphite.protonate(MolToSmiles(mol))

    if len(protonated_mols_smiles) > 0:
        return MolFromSmiles(protonated_mols_smiles[0])
    return mol


def embed_3d(mol):
    """
    Generate 3D coordinates and optimize with Merck molecular force field (MMFF).

    Args:
        mol: Molecule.

    Returns:
        True if successfully embedded in 3D, False otherwise.

    """
    if EmbedMolecule(mol, randomSeed=0) != 0:
        return False
    if MMFFOptimizeMolecule(mol, maxIters=1000) != 0:
        return False
    return True


def compute_gasteiger_charges(mol):
    """
    Compute Gasteiger-Marsili partial charges.

    Args:
        mol: Molecule.

    Returns:
        True if charges are computed successfully, False otherwise.
    """
    ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    # if any atom charge is nan or inf, skip this molecule.
    if np.any(np.isnan(charges)) or np.any(np.isinf(charges)):
        return False
    return True


def compute_atomic_log_p(tauts):
    """
    Compute atomic log P contribution for every atom using Wildman-Crippen approach.
    (in-place).

    Args:
        tauts: List of tautomers.
    """
    for taut in tauts:
        for atom, atomic_log_p in zip(taut.GetAtoms(), _CalcCrippenContribs(taut)):
            atom.SetProp("logP", str(atomic_log_p[0]))


def enumerate_and_preprocess_tautomers(mol, enumerator, num_tauts=-1):
    """
    Enumerate and preprocess num_tauts tautomers for molecule. If num_tauts == -1, all tautomers are enumerated.

    Args:
        mol: Molecule
        enumerator: Tautomer enumerator.
        num_tauts: Maximum number of tautomers for molecule.

    Returns:
        List of preprocessed tautomers.
    """
    try:
        tauts = enumerator.Enumerate(mol)
    except Chem.rdchem.KekulizeException:
        return []
    tauts = list(tauts)
    # Shuffle tautomers to get random subset.
    np.random.shuffle(tauts)
    processed_tauts = []
    for taut in tauts:
        # Set protonation states at pH 7.4
        set_protonation(taut)
        taut = AddHs(taut, addCoords=True)
        # Compute 3D Coordinates
        if not embed_3d(taut):
            continue
        if not compute_gasteiger_charges(taut):
            continue
        processed_tauts.append(taut)
        # Only enumerate num_tauts tautomers
        if num_tauts != -1 and len(processed_tauts) >= num_tauts:
            break
    return processed_tauts


def preprocess_dataset(path, num_compounds, num_tauts=-1):
    """
    Load and preprocess dataset.

    Args:
        path: Path to SDF dataset.
        num_compounds: Number of compounds to preprocess.
        num_tauts: Max number of tautomers per compound.

    Returns:
        mols: List of molecules (unpreprocessed).
        all_tauts: List of lists of tautomers for all molecules (preprocessed).
    """
    print("Preprocessing molecules:")
    print()
    np.random.seed(0)
    suppl = Chem.SDMolSupplier(path)

    salt_remover = SaltRemover()
    enumerator = TautomerEnumerator()

    mols = []
    all_tauts = []
    hash_table = {}

    i = 0
    with tqdm(total=num_compounds) as pbar:
        for j, mol in enumerate(suppl):
            stripped_mol = salt_remover.StripMol(mol)
            Kekulize(stripped_mol, clearAromaticFlags=True)
            # Check for duplicates
            smiles = Chem.MolToSmiles(stripped_mol)
            smiles_hash = hash(smiles)
            if smiles_hash in hash_table:
                if smiles == Chem.MolToSmiles(mols[hash_table[smiles_hash]]):
                    continue
            if not apply_filters(stripped_mol):
                continue

            tauts = enumerate_and_preprocess_tautomers(stripped_mol, enumerator, num_tauts)
            if len(tauts) == 0:
                continue

            compute_atomic_log_p(tauts)

            mols.append(stripped_mol)
            all_tauts.append(tauts)
            hash_table[smiles_hash] = i
            i += 1
            pbar.update(1)
            if i >= num_compounds:
                break
    print()
    print(f"Successfully preprocessed {i} of {j + 1} molecules.")
    print(f"Conversion rate {i / (j + 1) * 100:.0f} %.")
    return mols, all_tauts


