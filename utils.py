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


def visualize_matrix(r):
    data = r[:500, :500]
    data -= data.min()
    data /= data.max()
    plt.figure(figsize=(20, 20))
    plt.imshow(data)
    plt.show()


def histogram_matrix(r_matrix):
    plt.hist(r_matrix.ravel(), bins=30, range=(-0.9999, 1))
    plt.show()


def visualize_results_2d(mol_1, mol_2):
    Draw.MolToImage(mol_1, size=(500, 500)).show()
    Draw.MolToImage(mol_2, size=(500, 500)).show()


def get_hbd(mol):
    f = lambda x, y=HDonorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def get_hba(mol):
    f = lambda x, y=HAcceptorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def save_mols(mols, tauts, name_suffix=""):
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
    np.save(f"data/fepops_{name_suffix}.npy", fepops)


def load_fepops(name_suffix=""):
    fepops = np.load(f"data/fepops_{name_suffix}.npy", allow_pickle=True)
    return fepops

#  v.AddPharmacophore(conf.GetPositions()[hbd].tolist(), [(1, 0, 0) for _ in range(len(hbd))], label="HBD")