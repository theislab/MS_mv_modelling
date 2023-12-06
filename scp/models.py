import logging
from typing import Callable, Iterable, Literal, Optional, List, Sequence

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal, Bernoulli
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


## nn modules
class GlobalLinear(nn.Module):
    def __init__(self, n_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_features = n_features

        self.weight = nn.Parameter(torch.empty(1, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.weight

        if self.bias is not None:
            out += self.bias

        return out


class ElementwiseLinear(nn.Module):
    def __init__(self, n_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_features = n_features

        self.weight = nn.Parameter(torch.empty(n_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.weight

        if self.bias is not None:
            out += self.bias

        return out


## Decoder
class ConjunctionDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell"] = "protein",
        **kwargs,
    ):
        super().__init__()

        self.x_variance = x_variance

        # z -> px
        self.h_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.x_mean_decoder = nn.Linear(n_hidden, n_output)

        if self.x_variance == "protein-cell":
            self.x_var_decoder = nn.Linear(n_hidden, n_output)
        elif self.x_variance == "protein":
            self.x_var = nn.Parameter(torch.randn(n_output))

        # z -> p
        self.m_logit = GlobalLinear(n_output)
        self.m_prob_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            self.m_logit,
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, x_data: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        x_data
            tensor with shape ``(n_output,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """

        # z -> px
        h = self.h_decoder(z, *cat_list)
        x_mean = self.x_mean_decoder(h)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(h)
        elif self.x_variance == "protein":
            x_var = self.x_var

        x_var = torch.exp(x_var)

        # z -> p
        m_prob = self.m_prob_decoder(h)

        return x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        weight = self.m_logit.weight.detach().cpu().numpy()

        bias = None
        if self.m_logit.bias is not None:
            bias = self.m_logit.bias.detach().cpu().numpy()

        return weight, bias
    

class HybridDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell"] = "protein",
        **kwargs,
    ):
        super().__init__()

        self.x_variance = x_variance

        # z -> px
        self.h_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.x_mean_decoder = nn.Linear(n_hidden, n_output)

        if self.x_variance == "protein-cell":
            self.x_var_decoder = nn.Linear(n_hidden, n_output)
        elif self.x_variance == "protein":
            self.x_var = nn.Parameter(torch.randn(n_output))

        # z -> p, x -> p
        self.x_p = GlobalLinear(n_output)
        self.z_p = nn.Linear(n_hidden, 1)

    def forward(self, z: torch.Tensor, x_data: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        x_data
            tensor with shape ``(n_output,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """

        # z -> x
        h = self.h_decoder(z, *cat_list)
        x_mean = self.x_mean_decoder(h)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(h)
        elif self.x_variance == "protein":
            x_var = self.x_var

        x_var = torch.exp(x_var)

        # z -> p, x -> p
        z_p = self.z_p(h)
        x_p = self.x_p(x_mean)

        m_prob = torch.sigmoid(z_p + x_p)

        return x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        return None, None


class SelectionDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell"] = "protein",
        **kwargs,
    ):
        super().__init__()

        self.x_variance = x_variance

        # z -> x
        self.x_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.x_mean_decoder = nn.Linear(n_hidden, n_output)

        if self.x_variance == "protein-cell":
            self.x_var_decoder = nn.Linear(n_hidden, n_output)
        elif self.x_variance == "protein":
            self.x_var = nn.Parameter(torch.randn(n_output))

        # x -> p
        self.m_logit = GlobalLinear(n_output)
        self.m_prob_decoder = nn.Sequential(
            self.m_logit,
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, x_data: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        x_data
            tensor with shape ``(n_output,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """

        # z -> x
        x_h = self.x_decoder(z, *cat_list)
        x_mean = self.x_mean_decoder(x_h)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(x_h)
        elif self.x_variance == "protein":
            x_var = self.x_var

        x_var = torch.exp(x_var)

        # x -> p
        if x_data is None:
            x_p = x_mean
        else:
            m = x_data != 0
            x_p = x_data * m + x_mean * (~m)

        m_prob = self.m_prob_decoder(x_p)

        return x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        weight = self.m_logit.weight.detach().cpu().numpy()

        bias = None
        if self.m_logit.bias is not None:
            bias = self.m_logit.bias.detach().cpu().numpy()

        return weight, bias


## module
class PROTVAE(BaseModuleClass):
    """Variational auto-encoder for proteomics data.

    Parameters
    ----------
    n_input
        Number of input proteins
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
        x_variance: Literal["protein", "protein-cell"] = "protein",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        var_activation: Tunable[Callable] = None,
        decoder_type: Tunable[Literal["selection", "conjunction", "hybrid"]] = "selection",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.x_variance = x_variance
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

        # @TODO: if these have the same class signature, can we just pass the class?
        if decoder_type == "selection":
            self.decoder = SelectionDecoderPROTVI(
                n_input=n_input_decoder,
                n_output=n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                x_variance=x_variance,
            )
        elif decoder_type == "conjunction":
            self.decoder = ConjunctionDecoderPROTVI(
                n_input=n_input_decoder,
                n_output=n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                x_variance=x_variance,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
            )
        elif decoder_type == "hybrid":
            self.decoder = HybridDecoderPROTVI(
                n_input=n_input_decoder,
                n_output=n_input,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                x_variance=x_variance,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
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

        if self.log_variational:
            x = torch.log(1 + x)

        if (cont_covs is not None) and (self.encode_covariates):
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x

        if (cat_covs is not None) and (self.encode_covariates):
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

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {
            "x": x,
            "z": z,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }

    @auto_move_data
    def generative(
        self,
        x,
        z,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        use_x=True,
    ):
        """Runs the generative model."""

        if self.log_variational:
            x = torch.log(1 + x)

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

        x_d = x if use_x else None
        x_mean, x_var, m_prob = self.decoder(
            decoder_input, x_d, batch_index, *categorical_input
        )
        return {
            "x_mean": x_mean,
            "x_var": x_var,
            "m_prob": m_prob,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        mechanism_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]

        x_mean = generative_outputs["x_mean"]
        x_var = generative_outputs["x_var"]
        m_prob = generative_outputs["m_prob"]

        qz = inference_outputs["qz"]
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))

        kl_divergence = kl(qz, pz).sum(dim=-1)
        weighted_kl = kl_weight * kl_divergence
        reconst_loss = self._get_reconstruction_loss(
            x, x_mean, x_var, m_prob, mechanism_weight=mechanism_weight
        )

        loss = torch.mean(reconst_loss + weighted_kl)

        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_divergence
        )

    def sample(self):
        raise NotImplementedError

    def _get_reconstruction_loss(self, x, x_mean, x_var, m_prob, mechanism_weight=1.0):
        m_obs = x != 0
        m_miss = ~m_obs

        x_std = torch.sqrt(x_var)
        ll_x = Normal(x_mean, x_std).log_prob(x)
        ll_m = mechanism_weight * Bernoulli(m_prob).log_prob(m_obs.type(torch.float32))

        ll = torch.empty_like(x)
        ll[m_obs] = ll_x[m_obs] + ll_m[m_obs]
        ll[m_miss] = ll_m[m_miss]

        nll = -torch.sum(ll, dim=-1)

        return nll


## downstream utils
class ProteinMixin:
    @torch.inference_mode()
    def impute(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Imputes the protein intensities (including the missing intensities) and detection probabilities for the given indices.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData
            If None, defaults to the AnnData object used to initialize the model

        indices
            Indices of cells to use for imputation

        batch_size
            Batch size to use for imputation

        Returns
        -------
        imputated
            Tuple of imputed protein intensities, imputed detection probabilities based on the `indices` provided

        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        if batch_size is None:
            batch_size = adata.n_obs

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
        )

        intensity_list = []
        detection_probability_list = []
        for tensors in scdl:
            generative_kwargs = {"use_x": True}
            _, generative_outputs = self.module.forward(
                tensors=tensors, compute_loss=False, generative_kwargs=generative_kwargs
            )

            x_mean = generative_outputs["x_mean"].cpu().numpy()
            m_prob = generative_outputs["m_prob"].cpu().numpy()

            intensity_list.append(x_mean)
            detection_probability_list.append(m_prob)

        intensities = np.concatenate(intensity_list, axis=0)
        detection_probabilities = np.concatenate(detection_probability_list, axis=0)

        return intensities, detection_probabilities


## model
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
        decoder_type: Tunable[Literal["selection", "conjunction", "hybrid"]] = "selection",
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
        )

        self._model_summary_string = (
            "PROTVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: , latent_distribution: {}"
        ).format(n_hidden, n_latent, n_layers, dropout_rate, latent_distribution)

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