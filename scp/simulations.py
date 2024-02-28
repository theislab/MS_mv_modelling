import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d

#################################################
# Reproducibility
#################################################

seed_set = False
rng = np.random.default_rng()


def fix_seed(seed=0):
    global rng, seed_set
    seed_set = seed is not None
    rng = np.random.default_rng(seed)


def ensure_seed_set():
    assert seed_set, "Seed not set. Call fix_seed() first."


#################################################
# Data Simulation
#################################################


def simulate_group(
    n_cells=2000,
    n_proteins=1000,
):
    """
    Simulate a group of cells.
    """

    ensure_seed_set()

    # cell-type protein distribution
    cell_type_signature = rng.uniform(5, 12, n_proteins)

    # cell-specific variation
    cell_variation = rng.normal(0, 1, n_cells)

    intensity = cell_type_signature.reshape(1, -1) + cell_variation.reshape(-1, 1)
    measurement = rng.normal(intensity, 0.1)

    # MNAR
    # prob = logit_linear(measurement, b0=-6.0, b1=0.8)
    # prob = logistic(measurement, k=2, x0=7)
    prob = sigmoid(2 * measurement - 14)
    mask = create_sampled_mask(prob)

    adata = create_dataset(measurement, prob, mask)
    adata.obs["group"] = "g1"

    return adata


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

    idx_de_up = np.arange(n_de_proteins // 2)
    idx_de_down = np.arange(n_de_proteins // 2, n_de_proteins)

    mean_protein = rng.uniform(5, 12, n_proteins)
    var_protein = 0.3 * np.ones(n_proteins)

    x = rng.normal(mean_protein, var_protein, (n_cells, n_proteins))
    x[np.ix_(idx_group_1, idx_de_up)] += log2_fold_change
    x[np.ix_(idx_group_1, idx_de_down)] -= log2_fold_change

    # x_protein = np.mean(x, axis=0)
    # prob = logit_linear(x_protein, b0=-6.0, b1=0.8)
    # prob = np.tile(prob, (n_cells, 1))
    prob = logit_linear(x, b0=-6.0, b1=0.8)

    mask = create_sampled_mask(prob)

    adata = create_dataset(x, prob, mask)
    adata.obs["group"] = ""
    adata.obs["group"].iloc[idx_group_1] = "g1"
    adata.obs["group"].iloc[idx_group_2] = "g2"

    return adata


def simulate_group_advanced(
    n_cells=2000,
    n_proteins=1000,
    mcar_prob=0.0,
    mnar_cell_sd=0.0,
    mnar_protein_sd=0.0,
):
    """
    Simulate a group of cells.
    """

    ensure_seed_set()

    # cell-type protein distribution
    cell_type_signature = rng.uniform(5, 12, n_proteins)

    # cell-specific variation
    cell_variation = rng.normal(0, 1, n_cells)

    intensity = cell_type_signature.reshape(1, -1) + cell_variation.reshape(-1, 1)
    measurement = rng.normal(intensity, 0.1)

    ## MNAR
    mnar_x = measurement.copy()

    if mnar_cell_sd > 0:
        cell_specific_mnar = rng.normal(0, mnar_cell_sd, n_cells)
        mnar_x += cell_specific_mnar.reshape(-1, 1)

    if mnar_protein_sd > 0:
        protein_specific_mnar = rng.normal(0, mnar_protein_sd, n_proteins)
        mnar_x += protein_specific_mnar.reshape(1, -1)

    prob = sigmoid(2 * mnar_x - 14)
    mnar_mask = create_sampled_mask(prob)

    ## MCAR
    mcar_mask = create_mcar_mask(np.ones_like(prob) * (1 - mcar_prob))

    mask = mcar_mask * mnar_mask
    adata = create_dataset(measurement, prob, mask)
    adata.obs["group"] = "g1"

    return adata


#################################################
# Create detection probabilities from intensities
#################################################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit_linear(x, b0, b1):
    return sigmoid(b0 + b1 * x)


def logistic(x, k=1, x0=0, L=1):
    return L / (1 + np.exp(-k * (x - x0)))


#############################################
# Create masks
#############################################


def create_fixed_mask(prob, threshold=0.5):
    return np.ones_like(prob) * (prob > threshold)


def create_sampled_mask(prob):
    return rng.binomial(1, prob)


def create_mcar_mask(prob):
    return rng.binomial(1, prob)


#############################################
# Other utilities
#############################################


def create_dataset(intensities, detection_probabilities, mask):
    n_cells, n_proteins = intensities.shape

    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names)

    var_names = [f"protein_{i}" for i in range(n_proteins)]
    var = pd.DataFrame(index=var_names)

    adata = sc.AnnData(intensities, obs=obs, var=var)
    adata.layers["intensity"] = adata.X.copy()
    adata.layers["detection_probability"] = detection_probabilities.copy()
    adata.layers["detected"] = mask == 1

    return adata



def add_train_test_set(adata, train_mask, layer=None):
    x = adata.layers[layer] if layer is not None else adata.X

    train_mask = train_mask.astype(bool)

    adata.layers["train"] = x.copy()
    adata.layers["train"][~train_mask] = np.nan

    adata.layers["test"] = x.copy()
    adata.layers["test"][train_mask] = np.nan


def create_mnar_mcar_mask(mv_rate, mnar_proportion, x, seed=None, sd_scale=1.0):
    """
    Create a mask for missing values that are MNAR and MCAR.

    Parameters
    ----------
    mv_rate : float
        The proportion of missing values in the dataset.

    mnar_proportion : float
        The proportion of missing values that are MNAR.
    
    """

    if seed is not None:
        np.random.seed(seed)


    # MNAR
    def expected_mask_sum(xx, mu, sd):
        return 1 - np.nanmean(norm.cdf(xx, loc=mu, scale=sd))
    
    sd = np.nanstd(x) / 2 * sd_scale

    q_low = np.nanquantile(x, q=mv_rate * 0.0001)
    q_high = np.nanquantile(x, q=mv_rate)
    qs = np.linspace(q_low, q_high, num=50)
    expec_sum_m = [expected_mask_sum(x[~np.isnan(x)], q, sd) for q in qs]
    inv_mnar_curve = interp1d(expec_sum_m, qs)

    p = np.clip(mnar_proportion * mv_rate, inv_mnar_curve.x[0], inv_mnar_curve.x[-1])
    q = inv_mnar_curve(p)
    threshold = np.random.normal(q, sd, size=x.shape)
    m_mnar = x < threshold if mnar_proportion > 0 else np.zeros_like(x).astype(bool)

    # MCAR
    p_mcar = mv_rate * (1 - mnar_proportion) / (1 - mv_rate * mnar_proportion)
    m_mcar = np.random.binomial(1, p_mcar, size=x.shape).astype(bool)

    return m_mnar, m_mcar