from functools import partial
from itertools import chain
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from rdkit.Chem import rdDistGeom

from preprocessing import compound_preprocessing
from utils import timeit, get_hbd, get_hba, save_mols


@timeit
def generate_conformers(all_tauts):
    print()
    print("generating conformers")
    print()
    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xa700f
    etkdg.verbose = False
    etkdg.numThreads = 0
    conformer_num = 1024
    etkdg.useRandomCoords = True  # CalcNumRotatableBonds(taut) > 5

    for tauts in tqdm(all_tauts):
        for taut in tauts:
            rdDistGeom.EmbedMultipleConfs(taut, numConfs=conformer_num, params=etkdg)


def get_features(mol, num_centroids=4):
    charges, log_ps, hbd, hba = get_properties(mol)
    conf_pos = [conf.GetPositions() for conf in mol.GetConformers()]
    return list(map(partial(cluster_features, charges, log_ps, hbd, hba, num_centroids), conf_pos))


def get_properties(mol):
    charges = np.array([float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()])
    log_ps = np.array([float(atom.GetProp('logP')) for atom in mol.GetAtoms()])
    hbd = get_hbd(mol)
    hba = get_hba(mol)
    return charges, log_ps, hbd, hba


def compute_distance(cs):
    d1 = np.linalg.norm(cs[0] - cs[3])
    d2 = np.linalg.norm(cs[0] - cs[1])
    d3 = np.linalg.norm(cs[1] - cs[2])
    d4 = np.linalg.norm(cs[2] - cs[3])
    d5 = np.linalg.norm(cs[0] - cs[2])
    d6 = np.linalg.norm(cs[1] - cs[3])
    return [d1, d2, d3, d4, d5, d6]


def cluster_features(charges, log_ps, hbd, hba, n_centr, positions):
    kmeans = KMeans(n_clusters=4, random_state=0).fit(positions)
    cluster_q = [np.sum(charges[np.where(kmeans.labels_ == i)]) for i in range(n_centr)]
    cluster_L = [np.sum(log_ps[np.where(kmeans.labels_ == i)]) for i in range(n_centr)]
    cluster_HD = [float(len(np.where(kmeans.labels_[hbd] == i)[0]) > 0) for i in range(n_centr)]
    cluster_HA = [float(len(np.where(kmeans.labels_[hba] == i)[0]) > 0) for i in range(n_centr)]

    order = np.argsort(cluster_q)

    centroids = kmeans.cluster_centers_[order]
    dists = compute_distance(centroids)
    cluster_q = np.array(cluster_q)[order]
    cluster_L = np.array(cluster_L)[order]
    cluster_HD = np.array(cluster_HD)[order]
    cluster_HA = np.array(cluster_HA)[order]
    feat = list(chain(*zip(dists[:n_centr], cluster_q, cluster_L, cluster_HD, cluster_HA))) + dists[n_centr:]
    return feat


def get_medoids(conf_feats):
    conf_feats = np.array(conf_feats)
    if conf_feats.shape[0] < 7:
        return conf_feats
    kmedo = KMedoids(n_clusters=7, random_state=0).fit(conf_feats)
    return kmedo.cluster_centers_


@timeit
def cluster_conformers(all_tauts):
    print()
    print("clustering conformers")
    print()
    all_descriptors = []

    for tauts in tqdm(all_tauts):
        descriptors = []
        for taut in tauts:
            conf_feats = get_features(taut)
            fepops = get_medoids(conf_feats)
            descriptors.append(fepops)
        all_descriptors.append(descriptors)
    return all_descriptors


def main():
    dataset_path = '/home/lukas/Downloads/chembl25.sdf'

    mols, tauts = compound_preprocessing(dataset_path, 300, num_tauts=5)
    generate_conformers(tauts)
    save_mols(mols, tauts)

    descriptors = cluster_conformers(tauts)
    np.save("data/fepops_300.npy", descriptors)


if __name__ == "__main__":
    main()
