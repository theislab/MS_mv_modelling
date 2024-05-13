import logging
import warnings
from collections.abc import Sequence
from functools import partial
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model.base._utils import _de_core
from torch import nn as nn

from ._constants import EXTRA_KEYS
from ._protvae import PROTVAE

logger = logging.getLogger(__name__)


def scprotein_raw_counts_properties(
    adata_manager: AnnDataManager,
    idx1: list[int] | np.ndarray,
    idx2: list[int] | np.ndarray,
    var_idx: Optional[list[int] | np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Computes and returns some statistics on the raw counts of two sub-populations.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object setup with :class:`~scvi.model.SCVI`.
    idx1
        subset of indices describing the first population.
    idx2
        subset of indices describing the second population.
    var_idx
        subset of variables to extract properties from. if None, all variables are used.

    Returns
    -------
    type
        Dict of ``np.ndarray`` containing, by pair (one for each sub-population).

    """
    data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
    data1 = data[idx1]
    data2 = data[idx2]
    if var_idx is not None:
        data1 = data1[:, var_idx]
        data2 = data2[:, var_idx]
    mean1 = np.asarray((data1 > 0).mean(axis=0)).ravel()
    mean2 = np.asarray((data2 > 0).mean(axis=0)).ravel()
    properties = {"emp_mean1": mean1, "emp_mean2": mean2, "emp_effect": (mean1 - mean2)}
    return properties


class PROTVI(
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """ProtVI model.

    Parameters
    ----------
    adata:
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.

    n_hidden:
        Number of nodes per hidden layer.

    n_latent:
        Dimensionality of the latent space.

    n_layers:
        Number of hidden layers used for encoder and decoder NNs.

    dropout_rate:
        Dropout rate for neural networks.

    latent_distribution:
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)

    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        latent_distribution: Literal["normal", "ln"] = "normal",
        x_variance: Literal["protein", "protein-cell"] = "protein",
        log_variational: bool = True,
        decoder_type: Literal["selection", "conjunction", "hybrid"] = "selection",
        loss_type: Literal["elbo", "iwae"] = "elbo",
        n_samples: int = 1,
        max_loss_dropout: float = 0.0,
        use_x_mix=False,
        encode_norm_factors=False,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        n_prior_cats_per_cov = (
            self.adata_manager.get_state_registry(EXTRA_KEYS.PRIOR_CAT_COVS_KEY).n_cats_per_key
            if EXTRA_KEYS.PRIOR_CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        self.module = PROTVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            x_variance=x_variance,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            log_variational=log_variational,
            decoder_type=decoder_type,
            loss_type=loss_type,
            n_samples=n_samples,
            max_loss_dropout=max_loss_dropout,
            use_x_mix=use_x_mix,
            encode_norm_factors=encode_norm_factors,
            n_prior_continuous_cov=self.summary_stats.get("n_prior_continuous_covs", 0),
            n_prior_cats_per_cov=n_prior_cats_per_cov,
            **model_kwargs,
        )

        self._model_summary_string = f"PROTVI Model with the following params: \nn_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: {dropout_rate}"

        self.init_params_ = self._get_init_params(locals())

    @torch.inference_mode()
    def impute(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 32,
        loss_type: Optional[Literal["elbo", "iwae"]] = None,
        replace_with_obs: bool = False,
    ):
        """Imputes the protein intensities (including the missing intensities) and detection probabilities for the given indices.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData
            If None, defaults to the AnnData object used to initialize the model

        indices
            Indices of cells to use for imputation

        batch_size
            Batch size to use for imputation

        loss_type
            Loss type to use for imputation

        replace_with_obs
            Whether to replace the imputated values with the observed values

        n_samples
            number of samples to use for IWAE estimate


        Returns
        -------
        imputated
            Tuple of imputed protein intensities, imputed detection probabilities based on the `indices` provided

        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)  # type: ignore

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
        )

        if loss_type is None:
            loss_type = self.module.loss_type

        xs_list = []
        ps_list = []
        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            generative_kwargs = {"use_x_mix": replace_with_obs}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                compute_loss=False,
                generative_kwargs=generative_kwargs,
                inference_kwargs=inference_kwargs,
            )

            x_mean = generative_outputs["x_mean"].cpu().numpy()
            m_prob = generative_outputs["m_prob"].cpu().numpy()

            if loss_type == "elbo":
                x_imp = np.mean(x_mean, axis=0)
                p_imp = np.mean(m_prob, axis=0)
            elif loss_type == "iwae":
                distributions = self.module._get_distributions(inference_outputs, generative_outputs)
                x = tensors[REGISTRY_KEYS.X_KEY].to(inference_outputs["z"].device)
                mask = (x != 0).type(torch.float32)
                scoring_mask = mask
                lw = self.module._get_importance_weights(
                    x,
                    inference_outputs["z"],
                    mask,
                    scoring_mask,
                    **distributions,
                )
                lw = lw.cpu().numpy()

                e_x = np.exp(lw - np.max(lw, axis=0))
                w = e_x / np.sum(e_x, axis=0)

                x_imp = np.sum(x_mean * w[..., None], axis=0)
                p_imp = np.sum(m_prob * w[..., None], axis=0)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")

            xs_list.append(x_imp)
            ps_list.append(p_imp)

        xs = np.concatenate(xs_list, axis=0)
        ps = np.concatenate(ps_list, axis=0)

        if replace_with_obs:
            x_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
            m = x_data != 0
            xs = x_data * m + xs * ~m

        return xs, ps

    @torch.inference_mode()
    def get_normalized_abundance(
        self,
        adata: AnnData | None = None,
        indices: list[int] | np.ndarray | None = None,
        # transform_batch: list[Number | str] | None = None,
        # gene_list: list[str] | None = None,
        # library_size: float | Literal["latent"] = 1,
        n_samples: int = 1,  # @TODO: make test
        n_samples_overall: int = None,
        # weights: Literal["uniform", "importance"] | None = None,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        # **importance_weighting_kwargs,
        **kwargs,  # @TODO: remove
    ) -> np.ndarray | pd.DataFrame:
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            assert n_samples == 1  # default value
            n_samples = n_samples_overall // len(indices) + 1

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # @TODO: test returning dataframe, numpy array with more than 1 sample. assert on shape.
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True

        norm_abuns = []
        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            generative_kwargs = {"use_x_mix": False}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                generative_kwargs=generative_kwargs,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            norm_abuns.append(generative_outputs["x_norm"].squeeze().cpu())

        cell_axis = 1 if n_samples > 1 else 0
        norm_abuns = np.concatenate(norm_abuns, axis=cell_axis)

        if n_samples_overall is not None:
            norm_abuns = norm_abuns.reshape(-1, norm_abuns.shape[-1])
            n_samples_ = norm_abuns.shape[0]
            ind_ = np.random.choice(n_samples_, n_samples_overall, p=None, replace=True)
            norm_abuns = norm_abuns[ind_]
        elif n_samples > 1 and return_mean:
            norm_abuns = norm_abuns.mean(0)

        if (return_numpy is None) or (return_numpy is False):
            return pd.DataFrame(
                norm_abuns,
                columns=adata.var_names,
                index=adata.obs_names[indices],
            )
        else:
            return norm_abuns

    def differential_abundance(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: list[str] | None = None,
        group2: str | None = None,
        idx1: list[int] | list[bool] | str | None = None,
        idx2: list[int] | list[bool] | str | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: list[str] | None = None,
        batchid2: list[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        # weights: Literal["uniform", "importance"] | None = "uniform",
        # filter_outlier_cells: bool = False,
        # importance_weighting_kwargs: dict | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        adata = self._validate_anndata(adata)
        col_names = adata.var_names

        model_fn = partial(
            self.get_normalized_abundance,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            mode=mode,
            delta=delta,
            all_stats=all_stats,
            all_stats_fn=scprotein_raw_counts_properties,
            batch_correction=batch_correction,
            batchid1=batchid1,
            batchid2=batchid2,
            col_names=col_names,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )

        return result

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        prior_continuous_covariate_keys: Optional[list[str]] = None,
        prior_categorical_covariate_keys: Optional[list[str]] = None,
        norm_continuous_covariate_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        """Set up :class:`~anndata.AnnData` object for PROTVI.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.

        layer
            Name of the layer for which to extract the data.

        batch_key
            Key in ``adata.obs`` for batches/cell types/categories.

        categorical_covariate_keys
            List of keys in ``adata.obs`` for categorical covariates.

        continuous_covariate_keys
            List of keys in ``adata.obs`` for continuous covariates.

        prior_categorical_covariate_keys
            List of keys in ``adata.obs`` for prior categorical covariates. batch_key is *not* automatically added for prior covariates.

        prior_continuous_covariate_keys
            List of keys in ``adata.obs`` for prior continuous covariates.

        norm_continuous_covariate_keys
            List of keys in ``adata.obs`` for normalized continuous covariates.

        **kwargs
            Keyword args for AnnDataManager.register_fields

        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            CategoricalJointObsField(EXTRA_KEYS.PRIOR_CAT_COVS_KEY, prior_categorical_covariate_keys),
            NumericalJointObsField(EXTRA_KEYS.PRIOR_CONT_COVS_KEY, prior_continuous_covariate_keys),
            NumericalJointObsField(EXTRA_KEYS.NORM_CONT_COVS_KEY, norm_continuous_covariate_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
