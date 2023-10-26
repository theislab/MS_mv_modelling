import numpy as np
import pandas as pd
import scanpy as sc


## AnnData
def filter_by_detection_proportion(adata, min_coverage=0.1):
    detection_proportion = np.mean(~np.isnan(adata.X), axis=0)
    mask = detection_proportion >= min_coverage
    adata._inplace_subset_var(mask)


def reshape_anndata_like(adata, adata_like, sanity_check=True, verbose=True):
    """
    Reshape adata.X into the shape of adata_like.X.
    Annotations in .obs and .var are copied to the result.
    """

    def intersection(a, b):
        return list(set(a).intersection(b))

    row_overlap = intersection(adata.obs.index, adata_like.obs.index)
    col_overlap = intersection(adata.var.index, adata_like.var.index)

    if verbose:
        print(
            f"transfering {len(row_overlap)} cells and {len(col_overlap)} proteins during reshape."
        )

    x = pd.DataFrame(np.nan, index=adata_like.obs.index, columns=adata_like.var.index)
    x.loc[row_overlap, col_overlap] = adata[row_overlap, col_overlap].X.copy()

    obs = pd.DataFrame(
        np.nan,
        index=adata_like.obs.index,
        columns=adata.obs.columns,
    )
    obs.loc[row_overlap, :] = adata[row_overlap, :].obs.copy()

    var = pd.DataFrame(
        np.nan,
        index=adata_like.var.index,
        columns=adata.var.columns,
    )
    var.loc[col_overlap, :] = adata[:, col_overlap].var.copy()

    result = sc.AnnData(
        x,
        obs=obs,
        var=var,
    )

    if sanity_check:
        assert np.all(result.obs.index == adata_like.obs.index)
        assert np.all(result.var.index == adata_like.var.index)

        assert np.all(result.obs.columns == adata.obs.columns)
        assert np.all(result.var.columns == adata.var.columns)

        a = result[row_overlap, col_overlap].copy()
        a.X[np.isnan(a.X)] = 0

        b = adata[row_overlap, col_overlap].copy()
        b.X[np.isnan(b.X)] = 0

        assert np.all(a.X == b.X)

    return result


def prepare_anndata_for_R(adata):
    """
    Convert AnnData object to a format that is compatible with R.

    int64 values in .obs are not supported when converting the AnnData object to R, so convert them to int32
    (.X and .layers may contain float64 values)
    """

    adata = adata.copy()

    adata.obs = pd.DataFrame.from_dict(
        {
            column: adata.obs[column].astype(np.int32)
            if adata.obs[column].dtype == np.int64
            else adata.obs[column]
            for column in adata.obs.columns
        }
    )

    for column in adata.obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[column]):
            adata.obs.drop(column, axis=1, inplace=True)
            # adata.obs[column] = adata.obs[column].astype(str)

    return adata


## numpy
def get_coverage(x):
    mask = ~np.isnan(x)
    return mask.sum() / x.size


def fill_if_nan(a, b):
    # c:= a <- b
    mask = np.isnan(a) & ~np.isnan(b)
    c = a.copy()
    c[mask] = b[mask]
    return c


def r_squared(x, y):
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


## other imputation models


def impute_downshifted_normal_sample(
    x,
    scale=0.3,
    shift=1.8,
):
    x = x.copy()
    missing_indices = np.where(np.isnan(x))

    mean = np.nanmean(x, axis=1)
    std = np.nanstd(x, axis=1)
    mean_shifted = mean - shift * std
    std_shifted = scale * std

    np.random.seed(42)
    m = np.take(mean_shifted, missing_indices[0])
    s = np.take(std_shifted, missing_indices[0])
    draws = np.random.normal(m, s)
    x[missing_indices] = draws

    return x


def impute_downshifted_normal_global(
    x,
    scale=0.3,
    shift=1.8,
):
    x = x.copy()
    missing_indices = np.where(np.isnan(x))

    mean = np.nanmean(x)
    std = np.nanstd(x)

    np.random.seed(42)
    draws = np.random.normal(
        loc=mean - shift * std, scale=scale * std, size=len(missing_indices[0])
    )
    x[missing_indices] = draws

    return x


def impute_downshifted_normal_local(
    x,
    scale=0.3,
    shift=1.8,
):
    x = x.copy()
    missing_indices = np.where(np.isnan(x))

    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean_shifted = mean - shift * std
    std_shifted = scale * std

    np.random.seed(42)
    m = np.take(mean_shifted, missing_indices[1])
    s = np.take(std_shifted, missing_indices[1])
    draws = np.random.normal(m, s)
    x[missing_indices] = draws

    return x


def impute_sample_min(x):
    x = x.copy()
    missing_indices = np.where(np.isnan(x))

    sample_min = np.nanmin(x, axis=1)
    x[missing_indices] = np.take(sample_min, missing_indices[0])

    return x


# @TODO: add these too
"""
def impute_ppca(
    adata,
    ncomp=2,
):
    
    adata = adata.copy()

    genes_before = adata.shape[1]
    min_entry = np.nanmin(adata.X) - 1
    adata.X -= min_entry
    sc.pp.filter_genes(adata, min_cells=ncomp)
    adata.X += min_entry
    genes_after = adata.shape[1]
    print(
        "%d genes were filtered for the probabilistic PCA! %d genes left!"
        % (genes_before - genes_after, genes_after)
    )

    import statsmodels.api as sm

    pc = sm.multivariate.PCA(
        data=adata.X,
        ncomp=ncomp,
        standardize=True,
        normalize=True,
        # whether to normalize the factors to have unit inner product, otherwise loadings (eigenvectors) will have unit inner product
        method="svd",
        missing="fill-em",
        tol_em=5e-08,
        max_em_iter=100,
    )
    imputed_data = pc._adjusted_data
    return imputed_data


def impute_ppca_batch(
    self,
    batch="cohort",
    ncomp=5,
):
    import statsmodels.api as sm

    self.adata.X[self.missing_indices] = np.nan
    genes_before = self.adata.shape[1]
    min_entry = np.nanmin(self.adata.X) - 1
    self.adata.X -= min_entry
    protein_filter = np.array([True] * self.adata.shape[1])
    for b in np.unique(self.adata.obs[batch]):
        batch_data = self.adata[self.adata.obs[batch] == b]
        batch_filter, _ = sc.pp.filter_genes(batch_data, min_cells=ncomp, inplace=False)
        protein_filter = protein_filter & batch_filter
    self.adata = self.adata[:, protein_filter]
    self.adata.X += min_entry
    self.missing_indices = np.where(np.isnan(self.adata.X))
    genes_after = self.adata.shape[1]
    print(
        "%d genes were filtered for the probabilistic PCA! %d genes left!"
        % (genes_before - genes_after, genes_after)
    )

    for b in np.unique(self.adata.obs[batch]):
        batch_adata = self.adata[self.adata.obs[batch] == b]
        pc = sm.multivariate.PCA(
            data=batch_adata.X,
            ncomp=ncomp,
            standardize=True,
            normalize=True,
            # whether to normalize the factors to have unit inner product, otherwise loadings (eigenvectors) will have unit inner product
            method="svd",
            missing="fill-em",
            tol_em=5e-08,
            max_em_iter=100,
        )
        assert batch_adata.is_view
        batch_adata.X = pc._adjusted_data


def impute_knn(
    self,
    n_neighbors=5,
    weights="uniform",
):
    self.adata.X[self.missing_indices] = np.nan
    from sklearn.impute import KNNImputer

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        copy=False,
    )
    imputer.fit_transform(self.adata.X)


def impute_iterative(
    self,
    n_nearest_features,
    initial_strategy="constant",
):
    # At each step, a feature column is designated as output y and the other feature columns are treated as inputs X,
    # then a regressor is fit on (X, y) for known y and missing values are predicted. This is done multiple times
    # in an iterated fashion.
    self.adata.X[self.missing_indices] = np.nan
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        n_nearest_features=n_nearest_features,
        initial_strategy=initial_strategy,
        random_state=42,
        verbose=2,
    )
    self.adata.X = imputer.fit_transform(self.adata.X)
    del imputer
"""
