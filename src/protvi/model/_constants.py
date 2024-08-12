from typing import NamedTuple


class _EXTRA_KEYS_NT(NamedTuple):
    PRIOR_CAT_COVS_KEY: str = "prior_categorical_covs"
    PRIOR_CONT_COVS_KEY: str = "prior_continuous_covs"
    TREND_BATCH_KEY: str = "detection_trend_key"
    MULTILEVEL_COV_KEY: str = "multilevel_cov_key"


EXTRA_KEYS = _EXTRA_KEYS_NT()
