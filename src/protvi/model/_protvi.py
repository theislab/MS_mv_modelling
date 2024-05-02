import logging
from typing import Callable, Iterable, List, Literal, Optional, Sequence  # noqa: UP035

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from torch import nn as nn

from ._extra_keys import EXTRA_KEYS
from ._protvae import PROTVAE


class ProteinMixin:
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
            Whether to replace the imptuated values with the observed values

        n_samples
            number of samples to use for IWAE estimate

        Returns
        -------
        imputated
            Tuple of imputed protein intensities, imputed detection probabilities based on the `indices` provided

        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

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
                distributions = self.module._get_distributions(
                    inference_outputs, generative_outputs
                )
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


class PROTVI(
    VAEMixin,
    ProteinMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """

    Parameters
    ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
        n_hidden
            Number of nodes per hidden layer.
        n_latent
            Dimensionality of the latent space.
        n_layers
            Number of hidden layers used for encoder and decoder NNs.
        dropout_rate
            Dropout rate for neural networks.
        latent_distribution
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
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch

        n_prior_cats_per_cov = (
            self.adata_manager.get_state_registry(
                EXTRA_KEYS.PRIOR_CAT_COVS_KEY
            ).n_cats_per_key
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

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        prior_continuous_covariate_keys: Optional[List[str]] = None,
        prior_categorical_covariate_keys: Optional[List[str]] = None,
        norm_continuous_covariate_keys: Optional[List[str]] = None,
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

        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            CategoricalJointObsField(
                EXTRA_KEYS.PRIOR_CAT_COVS_KEY, prior_categorical_covariate_keys
            ),
            NumericalJointObsField(
                EXTRA_KEYS.PRIOR_CONT_COVS_KEY, prior_continuous_covariate_keys
            ),
            NumericalJointObsField(
                EXTRA_KEYS.NORM_CONT_COVS_KEY, norm_continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
