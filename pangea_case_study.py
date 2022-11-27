import argparse
from functools import partial
from itertools import chain
import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

from rdkit.Chem import rdDistGeom

from preprocessing import preprocess_dataset
from utils import timeit, get_hbd, get_hba, save_mols, save_fepops


def generate_conformers(all_tauts, num_conformers=1024):
    """
    Compute conformers for every molecule in all_tauts. Molecules get modified in place.

    Args:
        all_tauts: List of lists of tautomers.
        num_conformers: How many conformers per tautomer.
    """
    print()
    print("Generating conformers:")
    print()
    # Parameters for conformer generation
    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xa700f
    etkdg.verbose = False
    etkdg.numThreads = 0
    etkdg.optimizerForceTol = 0.0135  # experimental (see https://greglandrum.github.io/rdkit-blog/3d/conformers/optimization/2022/09/29/optimizing-conformer-generation-parameters.html)

    for tauts in tqdm(all_tauts):
        for taut in tauts:
            # If molecules have more than 5 rotatable bonds, sample conformers at random.
            etkdg.useRandomCoords = CalcNumRotatableBonds(taut) > 5
            # Compute conformers
            rdDistGeom.EmbedMultipleConfs(taut, numConfs=num_conformers, params=etkdg)


def get_features(mol, num_centroids=4):
    """
    Get FEPOPS features for every conformer of a molecule.

    Args:
        mol: Molecule to compute features for.
        num_centroids: For K-means clustering.

    Returns:
        List of features for every conformer.
    """
    charges, log_ps, hbd, hba = get_properties(mol)
    conf_pos = [conf.GetPositions() for conf in mol.GetConformers()]
    return list(map(partial(cluster_features, charges, log_ps, hbd, hba, num_centroids), conf_pos))


def get_properties(mol):
    """
    Compute properties of molecule for FEPOPS descriptors.

    Args:
        mol: Input molecule.

    Returns:
        charges: Gasteiger-Marsili partial charges for every atom.
        log_ps: Atomic log P contribution for every atom using Wildman-Crippen approach.
        hbd: Number of hydrogen bond donors.
        hba: Number of hydrogen bond acceptors.
    """
    charges = np.array([float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()])
    log_ps = np.array([float(atom.GetProp('logP')) for atom in mol.GetAtoms()])
    hbd = get_hbd(mol)
    hba = get_hba(mol)
    return charges, log_ps, hbd, hba


def compute_distance(cluster_centers):
    """
    Compute distances between cluster centers in specific order.

    Args:
        cluster_centers: Cluster centers sorted by partial charges in ascending order.

    Returns:
        List of distances
    """
    d1 = np.linalg.norm(cluster_centers[0] - cluster_centers[3])
    d2 = np.linalg.norm(cluster_centers[0] - cluster_centers[1])
    d3 = np.linalg.norm(cluster_centers[1] - cluster_centers[2])
    d4 = np.linalg.norm(cluster_centers[2] - cluster_centers[3])
    d5 = np.linalg.norm(cluster_centers[0] - cluster_centers[2])
    d6 = np.linalg.norm(cluster_centers[1] - cluster_centers[3])
    return [d1, d2, d3, d4, d5, d6]


def cluster_features(charges, log_ps, hbd, hba, n_centroids, positions):
    """
    Cluster conformer positions and compute features for each cluster.

    Args:
        charges: Gasteiger-Marsili partial charges for every atom.
        log_ps: Atomic log P contribution for every atom using Wildman-Crippen approach.
        hbd: Number of hydrogen bond donors.
        hba: Number of hydrogen bond acceptors.
        n_centroids: Num centroids for K-means clustering
        positions: List of 3D positions of atoms in conformer.

    Returns:
        List of 22 FEPOPS feature point for one conformer.

    """
    kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(positions)
    #  Sum of Gasteiger-Marsili partial charges per cluster.
    cluster_q = [np.sum(charges[np.where(kmeans.labels_ == i)]) for i in range(n_centroids)]
    # Sum of atomic log P values per cluster.
    cluster_L = [np.sum(log_ps[np.where(kmeans.labels_ == i)]) for i in range(n_centroids)]
    # Binary flag indicating the presence of hydrogen bond donors per cluster.
    cluster_HD = [float(len(np.where(kmeans.labels_[hbd] == i)[0]) > 0) for i in range(n_centroids)]
    # Binary flag indicating the presence of hydrogen bond acceptors per cluster.
    cluster_HA = [float(len(np.where(kmeans.labels_[hba] == i)[0]) > 0) for i in range(n_centroids)]

    # Compute order of cluster by partial charges.
    order = np.argsort(cluster_q)

    centroids = kmeans.cluster_centers_[order]
    dists = compute_distance(centroids)
    cluster_q = np.array(cluster_q)[order]
    cluster_L = np.array(cluster_L)[order]
    cluster_HD = np.array(cluster_HD)[order]
    cluster_HA = np.array(cluster_HA)[order]
    # Combine features to a feature vector of lengths 22
    # d1, q1, L1, HD1, HA1, d2, q2, L2, HD2, HA2, d3, q3, L3, HD3, HA3, d4, q4, L4, HD4, HA4, d5, d6
    feat = list(chain(*zip(dists[:n_centroids], cluster_q, cluster_L, cluster_HD, cluster_HA))) + dists[n_centroids:]
    return feat


def get_medoids(conf_feats):
    """
    Cluster conformer features with K-medoids.

    Args:
        conf_feats: List of features for every conformer. (1024 x 22)

    Returns:
        7 Medoids of conformer features. (7 x 22)
    """
    conf_feats = np.array(conf_feats)
    assert conf_feats.shape[0] >= 7
    kmedo = KMedoids(n_clusters=7, random_state=0).fit(conf_feats)
    return kmedo.cluster_centers_


def compute_fepops(all_tauts):
    """
    Compute FEPOPS descriptors for list of lists of tautomers.

    Args:
        all_tauts: [[mol1_taut1, mol1_taut2, ...], [mol2_taut1, ...], ...]

    Returns:
        FEPOPS for every tautomer.
    """
    print()
    print("Computing FEPOPS descriptors:")
    print()
    all_fepops = []

    for tauts in tqdm(all_tauts):
        fepops_tauts = []
        for taut in tauts:
            conf_feats = get_features(taut)
            fepops = get_medoids(conf_feats)
            fepops_tauts.append(fepops)
        all_fepops.append(fepops_tauts)
    return all_fepops


def main():
    """
    Computes FEPOPS descriptors for a given dataset.
    """
    parser = argparse.ArgumentParser(description="Compute FEPOPS descriptors for dataset.")
    parser.add_argument("-p", "--dataset_path", type=str, help="Path to chemical dataset in SDF format.")
    parser.add_argument("-n", "--num_molecules", type=int, default=500, help="Number of molecules to process.")
    parser.add_argument("--num_tautomers", type=int, default=5, help="Max number of tautomers per molecule.")
    parser.add_argument("-s", "--suffix", type=str, default="test", help="Save files with name suffix")

    args = parser.parse_args()

    mols, tauts = preprocess_dataset(args.dataset_path, args.num_molecules, num_tauts=args.num_tautomers)
    generate_conformers(tauts)
    save_mols(mols, tauts, name_suffix=args.suffix)

    fepops = compute_fepops(tauts)
    save_fepops(fepops, name_suffix=args.suffix)


if __name__ == "__main__":
    main()
