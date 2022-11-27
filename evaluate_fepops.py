from itertools import chain

import numpy as np
from rdkit.Chem import PyMol, Draw, MolToSmiles

from fepops import get_features
from utils import load_mols, load_fepops, visualize_results_2d


def corr(X, y):
    return X @ y.T / np.sqrt(np.sum(X ** 2, axis=1)[:, None] @ np.sum(y ** 2, axis=1)[None])


def fepops_similarity(X, y):
    X = np.reshape(X, (-1, X.shape[-1]))
    # y = np.reshape(y, -1)
    x_mean = np.mean(X, axis=0)
    # TODO: Is this scaling correct?
    X = X - x_mean
    y = y - x_mean
    x_std = np.std(X, axis=0)
    X /= x_std
    y /= x_std

    r = corr(X, y)

    r = np.reshape(r, (r.shape[0] // y.shape[0], y.shape[0], -1))
    best = np.max(r, axis=(1, 2))

    return best


def remove_same_compounds(r, mol_index):
    for mol_i in np.unique(mol_index):
        taut_i = np.where(mol_index==mol_i)[0]
        comb = [(x * 7, y * 7) for x in taut_i for y in taut_i if x <= y]
        for x, y in comb:
            r[x:x + 7, y:y + 7] = -1
            r[y:y + 7, x:x + 7] = -1


def fepops_similarity_matrix(X, mol_index):
    X = np.reshape(X, (-1, X.shape[-1]))
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    r = corr(X, X)
    remove_same_compounds(r, mol_index)
    r[np.tril_indices(r.shape[0])] = -1

    best = np.dstack(np.unravel_index(np.argsort(r.ravel()), r.shape))[0][::-1]
    return r, best


def get_most_similar_conformers(mol1, mol2):
    feat1 = np.array(get_features(mol1))
    feat2 = np.array(get_features(mol2))
    feat1 = feat1 - np.mean(feat1, axis=0)
    feat2 = feat2 - np.mean(feat2, axis=0)
    r = corr(feat1, feat2)
    i, j = np.where(r == r.max())
    print(r.max())
    return int(i), int(j)


def visualize_results_3d(mol_1, mol_2):
    conf_id_1, conf_id_2 = get_most_similar_conformers(mol_1, mol_2)
    v= PyMol.MolViewer()
    v.DeleteAll()
    v.server.do('set grid_mode, on')
    v.ShowMol(mol_1, confId=conf_id_1, name="mol_1", showOnly=False)
    v.ShowMol(mol_2, confId=conf_id_2, name='mol_2', showOnly=False)


def evaluate_probes(mols, mols_probe, tauts, tauts_probe, mol_index, mol_index_probe, fepops, fepops_probe):
    print(fepops.shape)
    print(fepops_probe.shape)

    for i, probe in enumerate(fepops_probe):
        sims = fepops_similarity(fepops, probe)
        print(f"Max similarity of r={np.max(sims)}")
        order = np.argsort(sims)[::-1]
        best_index = order[0]
        vis = input("Visualize? (y/N)")
        if vis == "y":
            visualize_results_2d(mols_probe[mol_index_probe[i]], mols[mol_index[best_index]])
            # visualize_results_3d(tauts_probe[i], tauts[best_index])


def evaluate_dataset(mols, tauts, mol_index, fepops):
    r_matrix, best_ids = fepops_similarity_matrix(fepops, mol_index)
    # visualize_matrix(r_matrix)
    # histogram_matrix(r_matrix)
    top_1_percent = int(len(np.where(r_matrix > -1)[0]) * 0.01)
    best_ids = best_ids[:top_1_percent]
    best_ids_tauts = (best_ids // 7).tolist()
    best_ids_mols = [[mol_index[taut1], mol_index[taut2]] for taut1, taut2 in best_ids_tauts]
    for (fep1, fep2), (taut_i_1, taut_i_2), (i, (mol_i_1, mol_i_2)) in zip(best_ids, best_ids_tauts, enumerate(best_ids_mols)):
        mol1 = mols[mol_i_1]
        mol2 = mols[mol_i_2]
        if MolToSmiles(mol1) == MolToSmiles(mol2):
            continue
        if best_ids_mols.index([mol_i_1, mol_i_2]) < i:
            continue
        print(f"Similarity = {r_matrix[fep1, fep2]}")
        visualize_results_2d(mol1, mol2)
        vis = input("Visualize in 3D? (y/N)")
        if vis == "y":
            visualize_results_3d(tauts[taut_i_1], tauts[taut_i_2])


def split_test_set(mols, tauts, fepops, n_test=1):
    # split data
    mols_probe = mols[-n_test:]
    tauts_probe = tauts[-n_test:]
    mols = mols[:-n_test]
    tauts = tauts[:-n_test]
    fepops_probe = fepops[-n_test:]
    fepops = fepops[:-n_test]

    mol_index = np.repeat(np.arange(len(tauts)), [len(t) for t in tauts])
    mol_index_probe = np.repeat(np.arange(len(tauts_probe)), [len(t) for t in tauts_probe])
    flat_tauts = np.array(list(chain(*tauts)))
    flat_tauts_probe = np.array(list(chain(*tauts_probe)))
    # flat_tauts_probe = np.array([t[0] for t in tauts_probe])

    fepops_probe = np.array(list(chain(*fepops_probe)))
    # fepops_probe = np.array([t[0] for t in fepops_probe])
    fepops = np.array(list(chain(*fepops)))

    return mols, mols_probe, flat_tauts, flat_tauts_probe, mol_index, mol_index_probe, fepops, fepops_probe


def main():
    name_suffix = 500
    mols, tauts = load_mols(name_suffix=name_suffix)
    fepops = load_fepops(name_suffix=name_suffix)

    mols, mols_probe, tauts, tauts_probe, mol_index, mol_index_probe, fepops, fepops_probe = split_test_set(mols, tauts, fepops, n_test=1)

    evaluate_probes(mols, mols_probe, tauts, tauts_probe, mol_index, mol_index_probe, fepops, fepops_probe)
    # evaluate_dataset(mols, tauts, mol_index, fepops)


if __name__ == "__main__":
    main()