import numpy as np
import pandas as pd

## metrics
def r_squared(x, y):
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


## R
def prepare_anndata_for_R(adata):
    # int64 values in .obs are not supported when converting the AnnData object to R, so convert them to int32
    # (.X and .layers may contain float64 values)

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
