from functools import partial
import pickle as pkl
from itertools import chain

import numpy as np
from rdkit.Chem import PyMol, Draw

from pangea_case_study import get_features
from utils import load_mols, load_fepops


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


def get_most_similar_conformers(mol1, mol2):
    feat1 = np.array(get_features(mol1))
    feat2 = np.array(get_features(mol2))
    feat1 = feat1 - np.mean(feat1, axis=0)
    feat2 = feat2 - np.mean(feat2, axis=0)
    r = corr(feat1, feat2)
    i, j = np.where(r == r.max())
    print(r.max())
    return int(i), int(j)


def visualize_results_3d(mol_probe, mol_dataset):
    conf_id_probe, conf_id_dataset = get_most_similar_conformers(mol_probe, mol_dataset)

    v= PyMol.MolViewer()
    v.DeleteAll()
    v.server.do('set grid_mode, on')
    v.ShowMol(mol_probe, confId=conf_id_probe, name="mol_probe", showOnly=False)
    v.ShowMol(mol_dataset, confId=conf_id_dataset, name='mol_dataset', showOnly=False)


def visualize_results_2d(mol_probe, mol_dataset):
    Draw.MolToImage(mol_probe, size=(500, 500)).show()
    Draw.MolToImage(mol_dataset, size=(500, 500)).show()


def evaluate_probes(mols, mols_probe, tauts, tauts_probe, fepops, fepops_probe):
    print(fepops.shape)
    print(fepops_probe.shape)

    flat_tauts = np.array(list(chain(*tauts)))
    flat_tauts_probe = np.array(list(chain(*tauts_probe)))

    for i, probe in enumerate(fepops_probe):
        sims = fepops_similarity(fepops, probe)
        print(f"Max similarity of r={np.max(sims)}")
        order = np.argsort(sims)[::-1]
        visualize_results_3d(flat_tauts_probe[i], flat_tauts[order[0]])


def main():
    mols, tauts = load_mols()
    fepops = load_fepops()

    n_test = 1
    mols_probe = mols[-n_test:]
    tauts_probe = tauts[-n_test:]
    mols = mols[:-n_test]
    tauts = tauts[:-n_test]
    fepops_probe = fepops[-n_test:]
    fepops = fepops[:-n_test]
    fepops_probe = np.array(list(chain(*fepops_probe)))
    fepops = np.array(list(chain(*fepops)))

    evaluate_probes(mols, mols_probe, tauts, tauts_probe, fepops,fepops_probe)

    # a = np.random.random((5, 2, 3))
    # b = np.random.random((2, 3))

    # print(a.shape)
    # print(b.shape)

    # r = pearson_r(a, b)
    # print(r)


if __name__ == "__main__":
    main()