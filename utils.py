import time
import pickle as pkl

import numpy as np
from rdkit import Chem
from rdkit.Chem import PyMol
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


def get_hbd(mol):
    f = lambda x, y=HDonorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def get_hba(mol):
    f = lambda x, y=HAcceptorSmarts: x.GetSubstructMatches(y, uniquify=1)
    return [x[0] for x in f(mol)]


def save_mols(mols, tauts):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(r"data/mols_300.pkl", "wb") as file:
        pkl.dump(mols, file)
    with open(r"data/tauts_300.pkl", "wb") as file:
        pkl.dump(tauts, file)


def load_mols():
    with open(r"data/mols_300.pkl", "rb") as file:
        mols = pkl.load(file)
    with open(r"data/tauts_300.pkl", "rb") as file:
        tauts = pkl.load(file)
    return mols, tauts


def load_fepops():
    fepops = np.load("data/fepops_300.npy", allow_pickle=True)
    return fepops

#  v.AddPharmacophore(conf.GetPositions()[hbd].tolist(), [(1, 0, 0) for _ in range(len(hbd))], label="HBD")