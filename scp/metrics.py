import numpy as np
from scipy.stats import spearmanr, pearsonr


def get_coverage(x):
    mask = ~np.isnan(x)
    return mask.sum() / x.size


def r_squared(x, y):
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def mse(x, y):
    return np.mean((x - y) ** 2)


def spearman(x, y):
    return spearmanr(x, y)[0]


def pearson(x, y):
    return pearsonr(x, y)[0]


def compare_intensities_protein_wise(
    x1, x2, metrics=["pearson", "spearman", "mse"], idx_proteins=None, n_min_overlap=2
):
    assert n_min_overlap >= 2

    if idx_proteins is None:
        idx_proteins = np.arange(x1.shape[1])

    overlap_mask = ~np.isnan(x1) & ~np.isnan(x2)
    n_intensities_per_protein = overlap_mask.sum(axis=0)
    idx_proteins_ok = idx_proteins[n_intensities_per_protein[idx_proteins] >= n_min_overlap]

    result = {metric: [] for metric in metrics}
    result["protein_idx"] = idx_proteins_ok

    metric_fns = {metric: globals()[metric] for metric in metrics}

    for protein_idx in idx_proteins_ok:
        row_overlap = overlap_mask[:, protein_idx]

        x1_protein = x1[row_overlap, protein_idx]
        x2_protein = x2[row_overlap, protein_idx]

        for metric, fn in metric_fns.items():
            value = fn(x1_protein, x2_protein)
            result[metric].append(value)

    return result
