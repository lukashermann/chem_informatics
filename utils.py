import os
import time
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import PyMol, Draw
from rdkit.Chem.Lipinski import HDonorSmarts, HAcceptorSmarts


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def viz_3d(mol, clear=True, grid=True, name="mol"):
    v = PyMol.MolViewer()
    if clear:
        v.DeleteAll()
    v.server.do(f"set grid_mode, {'on' if grid else 'off'}")
    v.ShowMol(mol, confId=-1, name=name, showOnly=False)


def visualize_matrix(r, num_rows=500):
    """
    Plot the correlation matrix.

    Args:
        r: Correlation matrix.
        num_rows: Show only num_rows x num_rows as subset of matrix
    """
    data = r[:num_rows, :num_rows]
    data -= data.min()
    data /= data.max()
    plt.figure(figsize=(20, 20))
    plt.imshow(data)
    plt.show()


def histogram_matrix(r_matrix):
    """
    Plot a histogram of correlations.

    Args:
        r_matrix: Correlation matrix.
    """
    plt.hist(r_matrix.ravel(), bins=30, range=(-0.9999, 1))
    plt.show()


def visualize_results_2d(mol_1, mol_2):
    """
    Visualize two molecules.
    """
    Draw.MolToImage(mol_1, size=(500, 500)).show()
    Draw.MolToImage(mol_2, size=(500, 500)).show()


def get_hbd(mol):
    """
    Get number of hydrogen bond donors in molecule.
    """
    f = lambda x, y=HDonorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def get_hba(mol):
    """
    Get number of hydrogen bond acceptors in molecule.
    """
    f = lambda x, y=HAcceptorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def save_mols(mols, tauts, name_suffix=""):
    os.makedirs("data", exist_ok=True)
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(rf"data/mols_{name_suffix}.pkl", "wb") as file:
        pkl.dump(mols, file)
    with open(rf"data/tauts_{name_suffix}.pkl", "wb") as file:
        pkl.dump(tauts, file)


def load_mols(name_suffix=""):
    with open(rf"data/mols_{name_suffix}.pkl", "rb") as file:
        mols = pkl.load(file)
    with open(rf"data/tauts_{name_suffix}.pkl", "rb") as file:
        tauts = pkl.load(file)
    return mols, tauts


def save_fepops(fepops, name_suffix=""):
    os.makedirs("data", exist_ok=True)
    np.save(f"data/fepops_{name_suffix}.npy", np.array(fepops, dtype=object))


def load_fepops(name_suffix=""):
    fepops = np.load(f"data/fepops_{name_suffix}.npy", allow_pickle=True)
    return fepops

#  v.AddPharmacophore(conf.GetPositions()[hbd].tolist(), [(1, 0, 0) for _ in range(len(hbd))], label="HBD")