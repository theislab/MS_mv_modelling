
import logging
from typing import Callable, Iterable, Literal, Optional, List

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.nn import (
    Decoder,
    Encoder,
    FCLayers,
)
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)

from anndata import AnnData

logger = logging.getLogger(__name__)


class NormalCdf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


class PROTVAE(BaseModuleClass):
    """Variational auto-encoder for proteomics data.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covariates
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to use covariates in the encoder.
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        log_variational: Tunable[bool] = True,
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        var_activation: Tunable[Callable] = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder = Encoder(
            n_input=n_input_encoder,
            n_output=n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = Decoder(
            n_input=n_input_decoder,
            n_output=n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        self.prob_net = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_layers=n_layers,
                n_hidden=n_hidden,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            ),
            nn.Linear(n_hidden, n_input),
            nn.Sigmoid(), #NormalCdf(),
        )

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }

    @auto_move_data
    def inference(
        self, 
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz, z = self.encoder(encoder_input, batch_index, *categorical_input)

        return {
            "qz": qz,
            "z": z,
        }

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {
            "z": z,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }

    @auto_move_data
    def generative(
        self, 
        z,
        batch_index,
        cont_covs=None,
        cat_covs=None,
    ):
        """Runs the generative model."""

        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        px_mean, px_var = self.decoder(
            decoder_input,
            batch_index,
            *categorical_input
        )
        prob_detection = self.prob_net(px_mean)

        return {
            "px_mean": px_mean,
            "px_std": torch.sqrt(px_var),
            "prob_detection": prob_detection,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]

        px_mean = generative_outputs["px_mean"]
        px_std = generative_outputs["px_std"]
        prob_detection = generative_outputs["prob_detection"]
        
        qz = inference_outputs["qz"]
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))

        likelihood = self._likelihood(prob_detection, px_mean, px_std, x)

        kl_divergence = kl(qz, pz).sum(dim=-1)
        weighted_kl_local = kl_weight * kl_divergence
        reconst_loss = -likelihood.sum(-1)

        loss = torch.mean(reconst_loss + weighted_kl_local)

        return LossOutput(
            loss=loss, 
            reconstruction_loss=reconst_loss, 
            kl_local=kl_divergence
        )
    
    def _likelihood(self, prob_detection, px_mean, px_std, x, eps=1e-6):
        px = Normal(px_mean, px_std)

        # @TODO: don't do unneccessary computation
        t1 = torch.log(torch.clamp(1 - prob_detection, min=eps))
        t2 = torch.log(torch.clamp(prob_detection, min=eps)) + px.log_prob(x)

        x_sig = (x != 0)
        likelihood = torch.empty_like(x)
        likelihood[~x_sig] = t1[~x_sig]
        likelihood[x_sig] = t2[x_sig]

        return likelihood


class PROTVI(
    VAEMixin,
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
        log_variational: bool = True,
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

        self.module = PROTVAE(
            n_input=self.summary_stats.n_vars,
            n_batch = n_batch,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            log_variational=log_variational,
        )

        self._model_summary_string = (
            "PROTVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: , latent_distribution: {}"
        ).format(
            n_hidden, n_latent, n_layers, dropout_rate, latent_distribution)
        
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
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
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata)
        cls.register_manager(adata_manager)
