import numpy as np
import scanpy as sc

from scp.utils import (
    prepare_anndata_for_R,
)


def impute_downshifted_normal_sample(
    adata,
    layer=None,
    scale=0.3,
    shift=1.8,
):
    if layer is None:
        x = adata.X.copy()
    else:
        x = adata.layers[layer].copy()

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
    adata,
    layer=None,
    scale=0.3,
    shift=1.8,
):
    if layer is None:
        x = adata.X.copy()
    else:
        x = adata.layers[layer].copy()

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
    adata,
    layer=None,
    scale=0.3,
    shift=1.8,
):
    if layer is None:
        x = adata.X.copy()
    else:
        x = adata.layers[layer].copy()

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


def impute_sample_min(adata, layer=None):
    if layer is None:
        x = adata.X.copy()
    else:
        x = adata.layers[layer].copy()

    missing_indices = np.where(np.isnan(x))

    sample_min = np.nanmin(x, axis=1)
    x[missing_indices] = np.take(sample_min, missing_indices[0])

    return x


def impute_knn(
    adata,
    layer=None,
    n_neighbors=5,
    weights="uniform",
):
    from sklearn.impute import KNNImputer

    adata = adata.copy()

    if layer is not None:
        adata.X = adata.layers[layer].copy()

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        copy=False,
    )
    imputer.fit_transform(adata.X)

    return adata.X


def impute_iterative(
    adata,
    layer=None,
    n_nearest_features=5,
    initial_strategy="constant",
):
    # At each step, a feature column is designated as output y and the other feature columns are treated as inputs X,
    # then a regressor is fit on (X, y) for known y and missing values are predicted. This is done multiple times
    # in an iterated fashion.

    adata = adata.copy()

    if layer is not None:
        adata.X = adata.layers[layer].copy()

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
    adata.X = imputer.fit_transform(adata.X)
    del imputer

    return adata.X


def run_protDP(adata, layer=None):
    import anndata2ri
    from rpy2.robjects import r
    from rpy2.robjects.vectors import ListVector
    
    r_adata = prepare_anndata_for_R(adata)

    anndata2ri.activate()
    r.assign("r_adata", r_adata)

    r(f"""
        library(protDP)

        X <- assay(r_adata, "{layer}")
        dpcfit <- dpc(X)
        """
    )
    dpcFit = r("dpcfit")

    def listvector_to_dict(r_listvector):
        py_dict = dict(zip(r_listvector.names, map(convert_r_to_python, r_listvector)))
        return py_dict

    def convert_r_to_python(r_object):
        if isinstance(r_object, ListVector):
            return listvector_to_dict(r_object)
        else:
            return r_object

    return convert_r_to_python(dpcFit)