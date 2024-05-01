import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import os


# files
def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_dict(dict, path):
    dir = os.path.dirname(path)
    ensure_dir_exists(dir)

    with open(path, "wb") as f:
        pickle.dump(dict, f)


def load_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# filtering
def filter_by_detection_proportion(adata, min_coverage=0.1):
    detection_proportion = np.mean(~np.isnan(adata.X), axis=0)
    mask = detection_proportion >= min_coverage
    adata._inplace_subset_var(mask)


## AnnData
def get_missingness_per_protein(adata, layer=None):
    x = adata.X if layer is None else adata.layers[layer]
    return np.mean(np.isnan(x), axis=0)


def sort_anndata_by_missingness(adata, layer=None):
    missingness = pd.DataFrame(
        get_missingness_per_protein(adata, layer=layer), columns=["missingness"]
    )
    missingness.sort_values(by="missingness", ascending=True, inplace=True)
    return adata[:, missingness.index]


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
            f"transferring {len(row_overlap)} cells and {len(col_overlap)} proteins during reshape."
        )

    x = pd.DataFrame(np.nan, index=adata_like.obs.index, columns=adata_like.var.index)
    x.loc[row_overlap, col_overlap] = adata[row_overlap, col_overlap].X.copy()

    obs = pd.DataFrame(
        np.nan,
        index=adata_like.obs.index,
        columns=adata.obs.columns,
    )
    obs = obs.astype(object)
    obs.loc[row_overlap, :] = adata[row_overlap, :].obs.copy()

    var = pd.DataFrame(
        np.nan,
        index=adata_like.var.index,
        columns=adata.var.columns,
    )
    var = var.astype(object)
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
            #adata.obs[column] = adata.obs[column].astype(str)

    return adata


## numpy
def fill_if_nan(a, b):
    # c:= a <- b
    mask = np.isnan(a) & ~np.isnan(b)
    c = a.copy()
    c[mask] = b[mask]
    return c
