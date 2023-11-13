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

def simulate_group(
    n_cells=1000,
    n_proteins=900,
):
    """
    Simulate a group of cells.
    """

    ensure_seed_set()

    # cell-type protein distribution
    cell_type_signature = np.random.uniform(5, 12, n_proteins)

    # cell-specific variation
    cell_variation = np.random.normal(0, 1, n_cells)

    intensity = cell_type_signature.reshape(1, -1) + cell_variation.reshape(-1, 1)
    meassurement = np.random.normal(intensity, 0.1)

    prob = logit_linear(meassurement, b0=-6.0, b1=0.8)
    mask = make_sampled_mask(prob)

    adata = create_dataset(meassurement, prob, mask)

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

    # idx_de_proteins = np.arange(n_de_proteins)
    idx_de_proteins = np.random.choice(range(n_proteins), n_de_proteins)
    de_direction = np.random.choice(np.array((-1,1)), len(idx_de_proteins), p=np.array((0.5,0.5)))


    mean_protein_g1 = np.random.uniform(5, 12, n_proteins)
    # mean_protein_g2 = mean_protein_g1.copy()
    # mean_protein_g2[idx_de_proteins] += log2_fold_change*de_direction
    
    var_protein = 0.3 * np.ones(n_proteins)

    # x_g1 = np.random.normal(mean_protein_g1, var_protein, (n_group1, n_proteins))
    # x_g2 = np.random.normal(mean_protein_g2, var_protein, (n_group2, n_proteins))
    # x  = np.concatenate((x_g1,x_g2), axis = 0)

    x = np.random.normal(mean_protein_g1, var_protein, (n_cells, n_proteins))
    x[np.ix_(idx_group_1, idx_de_proteins)] += log2_fold_change

    x_protein = np.mean(x, axis=0)
    prob = logit_linear(x_protein, b0=-8.87, b1=1.26)
    mask = make_mask(n_cells, prob) # this
   
    #prob = np.tile(prob, (n_cells, 1))

    
    # prob = logit_linear(x, b0=-6.0, b1=0.8)
    # mask = make_sampled_mask(prob)
    
   
    # mask = make_mask2(n_cells, prob)



    adata = create_dataset(x, prob, mask)

    adata.obs["group"] = ""
    adata.obs.iloc[idx_group_1, 0] = "g1"
    adata.obs.iloc[idx_group_2, 0] = "g2"

    adata.var["is_de"] = 0
    adata.var.iloc[idx_de_proteins,0] = de_direction

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


# def make_mask(n_rep, prob):
#     resamples = [np.random.binomial(n=1, p=prob) for i in range(n_rep)]
#     mask = np.concatenate(resamples,axis=0)
#     return mask.reshape(n_rep, len(prob))

def make_mask(n_rep, prob):
    return np.random.binomial(1, prob, size = (n_rep,len(prob)))

def make_mask2(n_rep, prob):
    detected_cells_per_prot = np.random.binomial(n=n_rep, p=prob)
    mask = np.zeros((n_rep, len(prob)))
    # detected_idx = np.arange(len(prob)*n_rep).reshape(n_rep,len(prob))
    for i, k in enumerate(detected_cells_per_prot):
        mask[np.random.choice(range(n_rep), k), i] = 1
    return mask


                    
                    
    

#############################################
# Other utilities
#############################################


def create_dataset(intensities, detection_probabilities, mask):
    n_cells, n_proteins = intensities.shape

    obs_names = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(index=obs_names, columns=["group"])

    var_names = [f"protein_{i}" for i in range(n_proteins)]
    var = pd.DataFrame(index=var_names, columns=["is_de","detection_probability"])

    adata = sc.AnnData(intensities, obs=obs, var=var)
    adata.layers["intensity"] = adata.X.copy()
    # adata.var["detection_probability"] = detection_probabilities.copy()
    adata.layers["detected"] = mask == 1

    return adata
