import numpy as np
import scanpy as sc
import pandas as pd


#################################################
# Reproducibility
#################################################

_seed_set = False


def fix_seed(seed=0):
    global _seed_set
    _seed_set = True

    np.random.seed(seed)


def ensure_seed_set():
    assert _seed_set, "Seed not set. Call fix_seed() first."


#################################################
# Data Simulation
#################################################

def simulate_two_groups(
    n_group1=500,
    n_group2=500,
    n_proteins=900,
    n_de_proteins=300,
    log2_fold_change=1,
):
    """
    Simulate two groups of cells with differentially expressed proteins.
    """

    ensure_seed_set()

    assert n_de_proteins <= n_proteins, "n_de_proteins can't be larger than n_proteins"

    n_cells = n_group1 + n_group2

    idx_group_1 = np.arange(n_group1)
    idx_group_2 = np.arange(n_group1, n_cells)

    idx_de_proteins = np.arange(n_de_proteins)

    mean_protein = np.random.uniform(5, 12, n_proteins)
    var_protein = 0.3 * np.ones(n_proteins)

    x = np.random.normal(mean_protein, var_protein, (n_cells, n_proteins))
    x[np.ix_(idx_group_1, idx_de_proteins)] += log2_fold_change

    #x_protein = np.mean(x, axis=0)
    #prob = logit_linear(x_protein, b0=-6.0, b1=0.8)
    #prob = np.tile(prob, (n_cells, 1))
    prob = logit_linear(x, b0=-6.0, b1=0.8)

    mask = make_sampled_mask(prob)

    adata = create_dataset(x, prob, mask)

    adata.obs["group"] = ""
    adata.obs.iloc[idx_group_1, 0] = "g1"
    adata.obs.iloc[idx_group_2, 0] = "g2"

    return adata


#################################################
# Create detection probabilities from intensities
#################################################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit_linear(x, b0, b1):
    return sigmoid(b0 + b1 * x)


#############################################
# Create masks from detection probabilities
#############################################


def make_fixed_mask(prob, threshold=0.5):
    return np.ones_like(prob) * (prob > threshold)


def make_sampled_mask(prob):
    return np.random.binomial(1, prob)


#############################################
# Other utilities
#############################################


def create_dataset(intensities, detection_probabilities, mask):
    n_cells, n_proteins = intensities.shape

    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names, columns=["group"])

    var_names = [f"protein_{i}" for i in range(n_proteins)]
    var = pd.DataFrame(index=var_names)

    adata = sc.AnnData(intensities, obs=obs, var=var)
    adata.layers["intensity"] = adata.X.copy()
    adata.layers["detection_probability"] = detection_probabilities.copy()
    adata.layers["detected"] = mask == 1

    return adata
