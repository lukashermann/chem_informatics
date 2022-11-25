import numpy as np
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from tqdm import tqdm

from repos.dimorphite_dl.dimorphite_dl import run_with_mol_list
from rdkit import Chem

from rdkit.Chem import AddHs, Kekulize, Draw
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds, _CalcCrippenContribs

from utils import timeit, viz_3d


def custom_filters(mol):
    if mol.GetNumAtoms() < 4:
        return False
    if CalcNumRings(mol) > 9:
        return False
    if CalcNumRotatableBonds(mol) > 40:
        return False
    return True


def protonate(mol):
    protonated_mols = run_with_mol_list(
        [mol],
        min_ph=7.4,
        max_ph=7.4,
        pka_precision=0,
        silent=True
    )
    if len(protonated_mols) > 0:
        return protonated_mols[0]
    print("Can't protonate")
    return mol


def embed_3d(mol):
    if EmbedMolecule(mol, randomSeed=0) != 0:
        return False
    if MMFFOptimizeMolecule(mol, maxIters=1000) != 0:
        return False
    return True


def compute_gasteiger_charges(mol):
    ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    # if any atom charge is nan or inf, skip this tautomer
    if np.any(np.isnan(charges)) or np.any(np.isinf(charges)):
        return False
    return True


def compute_atomic_log_p(tauts):
    for taut in tauts:
        for atom, atomic_log_p in zip(taut.GetAtoms(), _CalcCrippenContribs(taut)):
            atom.SetProp("logP", str(atomic_log_p[0]))


def get_tautomers(mol, enumerator, num_tauts=-1):
    Kekulize(mol, clearAromaticFlags=True)
    try:
        tauts = enumerator.Enumerate(mol)
    except Chem.rdchem.KekulizeException:
        print("can't kekulize")
        return []
    processed_tauts = []
    for taut in tauts:
        protonate(taut)
        taut = AddHs(taut, addCoords=True)
        if not embed_3d(taut):
            continue
        if not compute_gasteiger_charges(taut):
            continue
        processed_tauts.append(taut)
    if num_tauts != -1 and len(processed_tauts) > num_tauts:
        # sample num_tauts tautomers
        processed_tauts = list(np.random.choice(np.array(processed_tauts, dtype=object), size=num_tauts, replace=False))
    return processed_tauts


@timeit
def compound_preprocessing(path, num_compounds, num_tauts=-1):
    print("load sdf")
    suppl = Chem.SDMolSupplier(path)

    remover = SaltRemover()
    enumerator = TautomerEnumerator()

    mols = []
    all_tauts = []
    hash_table = {}

    i = 0
    with tqdm(total=num_compounds) as pbar:
        for mol in suppl:
            stripped_mol = remover.StripMol(mol)
            smiles = Chem.MolToSmiles(stripped_mol)
            smiles_hash = hash(smiles)
            if smiles_hash in hash_table:
                if smiles == Chem.MolToSmiles(mols[hash_table[smiles_hash]]):
                    # print("duplicate")
                    continue
            if not custom_filters(stripped_mol):
                continue

            tauts = get_tautomers(stripped_mol, enumerator, num_tauts)
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
    return mols, all_tauts


