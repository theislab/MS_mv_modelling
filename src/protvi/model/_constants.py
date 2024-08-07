from typing import NamedTuple


class _EXTRA_KEYS_NT(NamedTuple):
    PRIOR_CAT_COVS_KEY: str = "prior_categorical_covs"
    PRIOR_CONT_COVS_KEY: str = "prior_continuous_covs"
    NORM_CONT_COVS_KEY: str = "norm_continuous_covs"
    TREND_BATCH_KEY: str = "detection_trend_key"
    BOOSTER_PROTEINVAR_KEY: str = "booster_protein_variances_key"


EXTRA_KEYS = _EXTRA_KEYS_NT()
