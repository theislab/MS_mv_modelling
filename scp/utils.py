import numpy as np
import pandas as pd
import scanpy as sc

## AnnData
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
        print(f"transfering {len(row_overlap)} cells and {len(col_overlap)} proteins during reshape.")

    x = pd.DataFrame(
        np.nan,
        index=adata_like.obs.index,
        columns=adata_like.var.index
    )
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
