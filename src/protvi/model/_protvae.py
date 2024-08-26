import logging
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)
from scvi.nn import (
    Encoder,
    FCLayers,
    one_hot,
)
from torch import nn as nn
from torch.distributions import Bernoulli, Normal, kl_divergence

from ._constants import EXTRA_KEYS

logger = logging.getLogger(__name__)


class BatchEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.batch_encoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self,
        input,
    ):
        return self.batch_encoder(input)


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


class BatchGlobalLinear(nn.Module):
    def __init__(self, n_batch: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_batch = n_batch

        self.weight = nn.Parameter(torch.empty(n_batch, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_batch, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, batch_index) -> torch.Tensor:
        out = x * torch.index_select(self.weight, 0, batch_index[:, 0].long()).unsqueeze(1)  # batch x n_output

        if self.bias is not None:
            out += torch.index_select(self.bias, 0, batch_index[:, 0].long()).unsqueeze(1)

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


class DecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_extra_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell", "protein-batch"] = "protein",
        n_batch: int = None,
        n_batch_var: int = None,
        n_negative_control: int = None,
        batch_embedding_type: Literal["one-hot", "embedding", "encoder"] = "one-hot",
        batch_dim: int = None,
        batch_continous_info: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__()

        self.x_variance = x_variance
        self.n_batch = n_batch
        self.n_batch_var = n_batch_var
        self.batch_embedding_type = batch_embedding_type
        self.batch_continous_info = batch_continous_info

        n_cat_list = list([] if n_extra_cat_list is None else n_extra_cat_list)

        batch_input_dim = batch_continous_info.shape[-1] if batch_continous_info is not None else n_negative_control

        if batch_embedding_type == "one-hot":
            h_input = n_input + n_batch
        elif batch_embedding_type == "embedding":
            h_input = n_input + batch_dim
            if batch_dim is None:
                raise ValueError("`n_embedding` must be provided when using batch embedding")
            self.batch_embedding = nn.Embedding(n_batch, batch_dim)  # TO DO: replace with nn.Embedding
        elif batch_embedding_type == "encoder":
            h_input = n_input + batch_dim
            self.batch_encoder = BatchEncoder(
                n_input=batch_input_dim,
                n_output=batch_dim,
                n_hidden=n_hidden,
                n_layers=n_layers,
                # n_input=batch_input_dim, n_output=batch_dim, n_hidden=128, n_layers=1
            )
        else:
            raise ValueError("Invalid batch embedding type")

        self.h_decoder = FCLayers(
            n_in=h_input,
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
        elif self.x_variance == "protein-batch":
            self.x_var = nn.Parameter(torch.randn(n_output, n_batch_var))

    def forward(
        self,
        z: torch.Tensor,
        size_factor: torch.Tensor,
        batch_index: torch.Tensor,
        batch_var_index: torch.Tensor = None,  # trend_batch_index will be recycled to be used for batch (covariate)-specific variances
        batch_negative_control: torch.Tensor = None,
        *extra_cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        size_factor
            normalization factor
        batch_index
            special case categorical covariate for batch index
        extra_cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """
        # if self.batch_continous_info is None:
        #     batch_input = one_hot(batch_index, self.n_batch)
        # else:
        #     batch_input = self.batch_continous_info[batch_index].squeeze()
        #     # batch_input = self.batch_continous_info.squeeze()

        if self.batch_continous_info is not None:
            # batch_input = self.batch_continous_info[batch_index].squeeze()
            batch_input = self.batch_continous_info.squeeze()
        elif batch_negative_control is not None:
            batch_input = batch_negative_control
        else:
            batch_input = None

        if self.batch_embedding_type == "one-hot":
            batch_encoding = one_hot(batch_index, self.n_batch)
        elif self.batch_embedding_type == "embedding":
            batch_encoding = self.batch_embedding(batch_index.long()).squeeze(1)  # required .long() ?
        elif self.batch_embedding_type == "encoder":
            batch_encoding = self.batch_encoder(torch.tensor(batch_input, dtype=z.dtype, device=z.device))
            batch_encoding = torch.index_select(batch_encoding, 0, batch_index[:, 0].long())  # batch x latent dim
        else:
            raise ValueError("Invalid batch embedding type")

        hz = torch.cat((z, batch_encoding), dim=-1)

        h = self.h_decoder(hz, *extra_cat_list)
        x_norm = self.x_mean_decoder(h).squeeze()

        x_mean = x_norm + size_factor
        # x_mean = x_norm

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(h)
        elif self.x_variance == "protein":
            x_var = self.x_var.expand(x_mean.shape)

        x_var = torch.exp(x_var)

        return x_norm, x_mean, x_var, h


class ConjunctionDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_extra_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell", "protein-batch"] = "protein",
        n_batch: int = None,
        n_negative_control: int = None,
        batch_embedding_type: Literal["one-hot", "embedding", "encoder"] = "one-hot",
        batch_dim: int = None,
        batch_continous_info: torch.Tensor = None,
        detection_trend: Literal["global", "per-batch"] = "global",
        n_trend_batch: int = None,
        **kwargs,
    ):
        super().__init__()

        self.base_nn = DecoderPROTVI(
            n_input=n_input,
            n_output=n_output,
            n_extra_cat_list=n_extra_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            x_variance=x_variance,
            n_batch=n_batch,
            n_negative_control=n_negative_control,
            batch_embedding_type=batch_embedding_type,
            batch_dim=batch_dim,
            batch_continous_info=batch_continous_info,
            **kwargs,
        )

        self.m_prob_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            # GlobalLinear(n_output),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        x_obs: torch.Tensor,
        size_factor: torch.Tensor,
        batch_index: torch.Tensor,
        trend_batch_index: torch.Tensor = None,
        batch_negative_control: torch.Tensor = None,
        *extra_cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        size_factor
            normalization factor
        batch_index
            special case categorical covariate for batch index
        extra_cat_list
            list of category membership(s) for this sample


        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """
        trend_batch_index = None  # override to ensure trend is not used
        x_norm, x_mean, x_var, h = self.base_nn(
            z, size_factor, batch_index, trend_batch_index, batch_negative_control, *extra_cat_list
        )

        m_prob = self.m_prob_decoder(h)

        return x_norm, x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        return None, None


class HybridDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_extra_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell", "protein-batch"] = "protein",
        n_batch: int = None,
        n_negative_control: int = None,
        batch_embedding_type: Literal["one-hot", "embedding", "encoder"] = "one-hot",
        batch_dim: int = None,
        batch_continous_info: torch.Tensor = None,
        detection_trend: Literal["global", "per-batch"] = "global",
        n_trend_batch: int = None,
        **kwargs,
    ):
        super().__init__()

        self.base_nn = DecoderPROTVI(
            n_input=n_input,
            n_output=n_output,
            n_extra_cat_list=n_extra_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            x_variance=x_variance,
            n_batch=n_batch,
            n_negative_control=n_negative_control,
            batch_embedding_type=batch_embedding_type,
            batch_dim=batch_dim,
            batch_continous_info=batch_continous_info,
            **kwargs,
        )

        # z -> p, x -> p
        self.x_p = GlobalLinear(n_output)
        self.z_p = nn.Linear(n_hidden, 1)

    def forward(
        self,
        z: torch.Tensor,
        x_obs: torch.Tensor,
        size_factor: torch.Tensor,
        batch_index: torch.Tensor,
        trend_batch_index: torch.Tensor,
        batch_negative_control: torch.Tensor = None,
        *extra_cat_list: int,
        **kwargs,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        x_obs
            if set, use x_obs instead of x_mean as input for the detection probability
        size_factor
            normalization factor
        batch_index
            special case categorical covariate for batch index
        extra_cat_list
            list of category membership(s) for this sample


        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """
        x_norm, x_mean, x_var, h = self.base_nn(
            z, size_factor, batch_index, trend_batch_index, batch_negative_control, *extra_cat_list
        )

        # x_mix
        if x_obs is None:
            x_mix = x_mean
        else:
            m = x_obs != 0
            x_mix = x_obs * m + x_mean * ~m

        z_p = self.z_p(h)
        x_p = self.x_p(x_mix)

        m_prob = torch.sigmoid(z_p + x_p)

        return x_norm, x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        return None, None


class SelectionDecoderPROTVI(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_extra_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        x_variance: Literal["protein", "protein-cell", "protein-batch"] = "protein",
        n_batch: int = None,
        n_negative_control: int = None,
        batch_embedding_type: Literal["one-hot", "embedding", "encoder"] = "one-hot",
        batch_dim: int = None,
        batch_continous_info: torch.Tensor = None,
        detection_trend: Literal["global", "per-batch"] = "global",
        n_trend_batch: int = None,
        **kwargs,
    ):
        super().__init__()

        self.detection_trend = detection_trend
        self.base_nn = DecoderPROTVI(
            n_input=n_input,
            n_output=n_output,
            n_extra_cat_list=n_extra_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            x_variance=x_variance,
            n_batch=n_batch,
            n_batch_var=n_trend_batch[0] if n_trend_batch is not None else None,
            n_negative_control=n_negative_control,
            batch_embedding_type=batch_embedding_type,
            batch_dim=batch_dim,
            batch_continous_info=batch_continous_info,
            **kwargs,
        )

        if detection_trend == "global":
            self.m_logit = GlobalLinear(n_output)
            self.m_prob_decoder = nn.Sequential(
                self.m_logit,
                nn.Sigmoid(),
            )
        else:  # per batch modeling
            self.m_logit = BatchGlobalLinear(n_trend_batch)
            self.m_prob_decoder = nn.Sigmoid()

    def forward(
        self,
        z: torch.Tensor,
        x_obs: torch.Tensor,
        size_factor: torch.Tensor,
        batch_index: torch.Tensor,
        trend_batch_index: torch.Tensor,
        batch_negative_control: torch.Tensor = None,
        *extra_cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        x_obs
            if set, use x_obs instead of x_mean as input for the detection probability
        size_factor
            normalization factor
        batch_index
            special case categorical covariate for batch index
        extra_cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            mean, variance, and detection probability tensors

        """
        # z -> x
        x_norm, x_mean, x_var, _ = self.base_nn(
            z, size_factor, batch_index, trend_batch_index, batch_negative_control, *extra_cat_list
        )

        # x_mix
        if x_obs is None:
            x_mix = x_mean
        else:
            m = x_obs != 0
            x_mix = x_obs * m + x_mean * ~m

        if self.detection_trend == "global":
            m_prob = self.m_prob_decoder(x_mix)
        else:
            m_prob = self.m_prob_decoder(self.m_logit(x_mix, trend_batch_index))

        return x_norm, x_mean, x_var, m_prob

    def get_mask_logit_weights(self):
        weight = self.m_logit.weight.detach().cpu().numpy()

        bias = None
        if self.m_logit.bias is not None:
            bias = self.m_logit.bias.detach().cpu().numpy()

        return weight, bias


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
    decoder_type
        One of

        * ``'selection'`` - Selection decoder
        * ``'conjunction'`` - Conjunction decoder
        * ``'hybrid'`` - Hybrid decoder

    loss_type
        One of

        * ``'elbo'`` - Evidence lower bound
        * ``'iwae'`` - Importance weighted autoencoder

    n_samples
        Number of samples to use for importance weighted autoencoder loss.
    max_loss_dropout
        Maximum fraction of features to use when computing the loss. This acts as a dropout mask.
    n_prior_continuous_cov
        Number of continuous covariates for the prior.
    n_prior_cats_per_cov
        Number of categories for each extra categorical covariate for the prior.
    batch_continous_info
        Information for batch continuous covariates. Assumes that the batch continuous covariates are ordered by batches also sorted by name.
        See `scvi.data.setup_anndata` for more information on the ordering of batches.

    """

    def __init__(
        self,
        n_input: int,
        n_batch: int,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] = None,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        x_variance: Literal["protein", "protein-cell", "protein-batch"] = "protein",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Callable = None,
        decoder_type: Literal["selection", "conjunction", "hybrid"] = "selection",
        loss_type: Literal["elbo", "iwae"] = "elbo",
        batch_embedding_type: Literal["one-hot", "embedding", "encoder"] = "one-hot",
        batch_dim: int = None,
        n_samples: int = 1,
        max_loss_dropout: float = 0.0,
        use_x_mix=False,
        encode_norm_factors=False,
        n_prior_continuous_cov: int = 0,
        n_prior_cats_per_cov: Iterable[int] = None,
        batch_continous_info: torch.Tensor = None,
        detection_trend: Literal["global", "per-batch"] = "global",
        n_trend_batch: int = None,
        negative_control_indices: list[int] = None,
        n_multilevel_batch: int = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_trend_batch = n_trend_batch
        self.n_multilevel_batch = n_multilevel_batch
        self.log_variational = log_variational
        self.x_variance = x_variance
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.loss_type = loss_type
        self.n_samples = n_samples
        self.max_loss_dropout = max_loss_dropout
        self.use_x_mix = use_x_mix
        self.encode_norm_factors = encode_norm_factors
        losses = {
            "elbo": self._elbo_loss,
            "iwae": self._iwae_loss,
        }
        self.loss_fn = losses[loss_type]
        self.decoder_type = decoder_type
        self.negative_control_indices = negative_control_indices
        self.n_negative_control = len(negative_control_indices) if negative_control_indices is not None else None

        self.n_prior_cats_per_cov = n_prior_cats_per_cov

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        ## Encoder
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

        self.l_encoder = Encoder(
            n_input=n_input_encoder,
            n_output=1,
            n_cat_list=encoder_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        ## Decoder
        modules = {
            "conjunction": ConjunctionDecoderPROTVI,
            "hybrid": HybridDecoderPROTVI,
            "selection": SelectionDecoderPROTVI,
        }
        module = modules[decoder_type]

        n_extra_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = module(
            n_input=n_input_decoder,
            n_output=n_input,
            n_extra_cat_list=n_extra_cat_list,
            n_batch=n_batch,
            n_negative_control=self.n_negative_control,
            n_layers=n_layers,
            n_hidden=n_hidden,
            x_variance=x_variance,
            batch_embedding_type=batch_embedding_type,
            batch_dim=batch_dim,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            batch_continous_info=batch_continous_info,
            detection_trend=detection_trend,
            n_trend_batch=n_trend_batch,
        )

        ## Latent prior encoder

        n_prior_cats_per_cov = n_prior_cats_per_cov[0] if n_prior_cats_per_cov is not None else 0

        self.n_prior_cats_per_cov = n_prior_cats_per_cov
        self.prior_cat_embedding = (
            nn.Embedding(self.n_prior_cats_per_cov, batch_dim) if n_prior_cats_per_cov > 0 else None
        )

        # n_input_prior_encoder = n_prior_continuous_cov + n_prior_cats_per_cov
        # cat_list = list([] if n_prior_cats_per_cov is None else n_prior_cats_per_cov)

        # assuming the nn.Embedding as default implementation for all categorical data
        # or can add additional batch_dim_prior, or can have a statement to choose between one-hot or nn.Embedding
        # but all one-hot should potentially be replaced with nn.Embedding

        # TO DO: Need to deal with this properly if using embedding for categorical covariates
        # batch_dim = 0 if batch_dim is None else batch_dim

        if n_prior_cats_per_cov == 0:
            n_input_prior_encoder = n_prior_continuous_cov
        else:
            n_input_prior_encoder = n_prior_continuous_cov + batch_dim

        self.n_input_prior_encoder = n_input_prior_encoder

        self.prior_encoder = Encoder(
            n_input=n_input_prior_encoder,
            n_output=n_latent,
            # n_cat_list=cat_list,
            n_layers=n_layers,  # this should be set to 1?
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=True,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        # self.size_factor = nn.Parameter(torch.randn(n_input))
        self.size_factor = nn.Parameter(torch.randn(n_input, n_batch))
        # self.size_factor = nn.Parameter(torch.randn(n_trend_batch))

        self.alpha = (
            nn.Parameter(torch.rand(n_multilevel_batch[0], batch_dim), requires_grad=False)
            if self.n_multilevel_batch is not None
            else None
        )  # torch.rand values between 0-1, torch.full to initialize all with a single value
        self.shift_embedding = (
            nn.Embedding(self.n_multilevel_batch[0], batch_dim) if self.n_multilevel_batch is not None else None
        )

        # self.mixture_prior = False
        # if n_prior_cats_per_cov > 1:
        #     self.b_prior_logits = torch.nn.Parameter(torch.zeros(n_prior_cats_per_cov))
        #     self.b_prior_means = torch.nn.Parameter(torch.randn(10, n_prior_cats_per_cov))
        #     self.b_prior_scales = torch.nn.Parameter(torch.zeros(10, n_prior_cats_per_cov))
        #     self.mixture_prior = True

    def _get_inference_input(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        prior_cont_key = EXTRA_KEYS.PRIOR_CONT_COVS_KEY
        prior_cont_covs = tensors[prior_cont_key] if prior_cont_key in tensors.keys() else None

        prior_cat_key = EXTRA_KEYS.PRIOR_CAT_COVS_KEY
        prior_cat_covs = tensors[prior_cat_key] if prior_cat_key in tensors.keys() else None

        return {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "prior_cont_covs": prior_cont_covs,
            "prior_cat_covs": prior_cat_covs,
        }

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        prior_cont_covs=None,
        prior_cat_covs=None,
        n_samples=None,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        ## latent prior
        if prior_cont_covs is not None:
            prior_continous_input = prior_cont_covs.to(x.device)
        else:
            prior_continous_input = torch.empty(0).to(x.device)

        if prior_cat_covs is not None:
            prior_categorical_input = self.prior_cat_embedding(
                prior_cat_covs.long()
            )  # TODO extend to multiple categorical covariate
        else:
            prior_categorical_input = torch.empty(0).to(x.device)

        #### original
        # if (prior_cont_covs is not None) or (prior_cat_covs is not None):
        #     latent_model_input = torch.cat((prior_continous_input, prior_categorical_input), dim=-1)
        #     # pz, _ = self.prior_encoder(prior_continous_input, *prior_categorical_input)
        #     pz, _ = self.prior_encoder(latent_model_input)
        # else:
        #     pz = Normal(loc=0.0, scale=1.0)

        if (prior_cont_covs is not None) or (prior_cat_covs is not None):
            latent_model_input = torch.cat((prior_continous_input, prior_categorical_input), dim=-1)

            latent_model_input = latent_model_input.squeeze(1)
            # pz, _ = self.prior_encoder(prior_continous_input, *prior_categorical_input)
            pz, _ = self.prior_encoder(latent_model_input)

        # elif (prior_cat_covs is not None) and (prior_cont_covs is None):
        #     offset = (
        #         10.0 * F.one_hot(prior_cat_covs, num_classes=self.n_prior_cats_per_cov).float()
        #         if self.n_prior_cats_per_cov >= 2
        #         else 0.0
        #     )
        #     cats =torch.distributions.Categorical(logits=self.b_prior_logits + offset)
        #     normal_dists = torch.distributions.Normal(self.b_prior_means, torch.exp(self.b_prior_scales))
        #     pz = torch.distributions.MixtureSameFamily(cats, normal_dists)

        else:
            pz = Normal(loc=0.0, scale=1.0)

        ## latent posterior
        if n_samples is None:
            n_samples = self.n_samples

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

        qz, _ = self.encoder(encoder_input, batch_index, *categorical_input)

        # shape: (n_samples, n_batch, n_latent)
        z = qz.rsample(torch.Size([n_samples]))

        ql = None
        library = None
        if self.encode_norm_factors:
            ql, mean_library = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = ql.sample((n_samples,))

        else:
            library = F.linear(one_hot(batch_index, self.n_batch), self.size_factor)

        return {
            "pz": pz,
            "qz": qz,
            "z": z,
            "library": library,
        }

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        size_factor = inference_outputs["library"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        trend_batch_key = EXTRA_KEYS.TREND_BATCH_KEY
        trend_batch_index = tensors[trend_batch_key] if trend_batch_key in tensors.keys() else None

        multilevel_batch_key = EXTRA_KEYS.MULTILEVEL_COV_KEY
        multilevel_batch_index = tensors[multilevel_batch_key] if multilevel_batch_key in tensors.keys() else None

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        nc_idx = self.negative_control_indices
        if nc_idx is not None:
            x_nc = x[:, nc_idx]
        else:
            x_nc = None

        return {
            "x": x,
            "z": z,
            "size_factor": size_factor,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "trend_batch_index": trend_batch_index,
            "multilevel_batch_index": multilevel_batch_index,
            "x_nc": x_nc,
        }

    @auto_move_data
    def generative(
        self,
        x,
        z,
        size_factor,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        trend_batch_index=None,
        multilevel_batch_index=None,
        x_nc=None,
        use_x_mix=False,
    ):
        """Runs the generative model."""
        n_samples, n_batch = z.size(0), z.size(1)
        use_x_mix = use_x_mix if self.use_x_mix is not None else self.use_x_mix

        packed_shape = (n_samples * n_batch, -1)
        unpacked_shape = (n_samples, n_batch, -1)

        if self.log_variational:
            x = torch.log(1 + x)

        if batch_index is not None:
            # shape: (n_batch, 1) -> (n_samples * n_batch, 1)
            batch_index = batch_index.repeat(n_samples, 1)

        if trend_batch_index is not None:
            # shape: (n_batch, 1) -> (n_samples * n_batch, 1)
            trend_batch_index = trend_batch_index.repeat(n_samples, 1)

        mu_a = a = pa = None
        if multilevel_batch_index is not None:
            # shape: (n_batch, 1) -> (n_samples * n_batch, 1)
            multilevel_batch_index = multilevel_batch_index.repeat(n_samples, 1)
            shift = self.shift_embedding(multilevel_batch_index.long())
            mean_embedding = (
                self.shift_embedding.weight.clone()
            )  # it could be that the shift has to be sampled from a normal distribution
            self.mean_embedding = mean_embedding

            latent_offset = torch.index_select(
                self.alpha, 0, multilevel_batch_index[:, 0].long()
            )  # batch_size x batch_dim
            size_factor = latent_offset.sum(dim=-1)  # batch_size x 1
            size_factor = size_factor.unsqueeze(1)

            mu_a = torch.index_select(self.mean_embedding, 0, multilevel_batch_index[:, 0].long())
            a = torch.index_select(self.alpha, 0, multilevel_batch_index[:, 0].long())
            pa = Normal(a, torch.ones_like(a))

        if cat_covs is not None:
            # shape: (n_batch, n_cat_covs) -> (n_samples * n_batch, n_cat_covs)
            cat_covs_repeat = cat_covs.repeat(n_samples, 1)
            categorical_input = torch.split(cat_covs_repeat, 1, dim=1)
        else:
            categorical_input = ()

        if cont_covs is not None:
            # shape: (n_samples, n_batch, n_input) -> (n_samples * n_batch, n_input + n_cont_covs)
            cont_conv_repeat = cont_covs.repeat(n_samples, 1)
            decoder_input = torch.cat([z.view(packed_shape), cont_conv_repeat], dim=-1)
        else:
            # (n_samples, n_batch, n_input) -> (n_samples * n_batch, n_input)
            decoder_input = z.view(packed_shape)

        # shape: (n_batch, n_input) -> (n_samples * n_batch, n_input)
        x_obs = x.repeat(n_samples, 1) if use_x_mix else None

        x_norm, x_mean, x_var, m_prob = self.decoder(
            decoder_input, x_obs, size_factor, batch_index, trend_batch_index, x_nc, *categorical_input
        )  # double check this

        # shape: (n_samples * n_batch, n_input) -> (n_samples, n_batch, n_input)
        x_norm = x_norm.view(unpacked_shape)
        x_mean = x_mean.view(unpacked_shape)
        x_var = x_var.view(unpacked_shape)
        m_prob = m_prob.view(unpacked_shape)

        return {
            "x": x,
            "x_norm": x_norm,
            "x_mean": x_mean,
            "x_var": x_var,
            "m_prob": m_prob,
            "pa": pa,
            "mu_a": mu_a,
        }

    def _get_distributions(self, inference_outputs, generative_outputs):
        qz = inference_outputs["qz"]
        pz = inference_outputs["pz"]
        # size_factor = inference_outputs["library"]

        pa = generative_outputs["pa"]
        mu_a = generative_outputs["mu_a"]

        x_mean = generative_outputs["x_mean"]
        x_var = generative_outputs["x_var"]
        m_prob = generative_outputs["m_prob"]
        # x = generative_outputs["x"]
        # m = x != 0
        # x_mix = x * m + x_mean * ~m

        # if self.encode_norm_factors:
        #      px = Normal(loc=(x_mean - size_factor), scale=torch.sqrt(x_var))
        # else:
        #      px = Normal(loc= x_mean, scale=torch.sqrt(x_var))

        px = Normal(loc=x_mean, scale=torch.sqrt(x_var))

        pm = Bernoulli(probs=m_prob)

        return {
            "qz": qz,
            "pz": pz,
            "px": px,
            "pm": pm,
            "pa": pa,
            "mu_a": mu_a,
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
        z = inference_outputs["z"]

        distributions = self._get_distributions(inference_outputs, generative_outputs)

        mask = (x != 0).type(torch.float32)
        scoring_mask = self._get_scoring_mask(mask, max_loss_dropout=self.max_loss_dropout)

        return self.loss_fn(  # type: ignore
            x=x,
            z=z,
            mask=mask,
            scoring_mask=scoring_mask,
            **distributions,
            kl_weight=kl_weight,
            mechanism_weight=mechanism_weight,
        )

    def _get_scoring_mask(self, mask, max_loss_dropout: float):
        # the scoring mask acts as a dropout mask when computing the loss of the reconstructed features
        # In each sample a fraction of the features are used to compute the loss
        p_missing = torch.rand(mask.shape[0], 1) * max_loss_dropout
        scoring_mask = torch.bernoulli(1.0 - p_missing.expand_as(mask)).to(mask.dtype).to(mask.device)
        return scoring_mask

    def _elbo_loss(
        self,
        x,
        z,
        mask,
        scoring_mask,
        qz,
        pz,
        px,
        pm,
        pa,
        mu_a,
        kl_weight: float = 1.0,
        mechanism_weight: float = 1.0,
        **kwargs,
    ) -> LossOutput:
        log_px = mask * px.log_prob(x)
        log_pm = mechanism_weight * pm.log_prob(mask)

        log_alpha = torch.tensor(0.0)
        if mu_a is not None:
            log_alpha = -pa.log_prob(mu_a).sum(dim=-1)  # batch_size

        # log_obs = scoring_mask * (log_px + log_pm)
        log_obs = 1.0 * (log_px + log_pm)

        # (n_samples, n_batch, n_features) -> (n_batch, n_features)
        log_pd = log_obs.sum(dim=0)

        # (n_batch, n_features) -> (n_batch,)
        reconstruction_loss = -log_pd.sum(dim=-1)

        # KL
        # (n_batch, n_latent) -> (n_batch,)
        kl = kl_divergence(qz, pz).sum(dim=-1)

        # if self.mixture_prior:
        #     kl = qz.log_prob(z) - pz.log_prob(z)
        #     kl = kl.sum(dim=-1)
        # else:
        #      kl = kl_divergence(qz, pz).sum(dim=-1)

        weighted_kl = kl * kl_weight

        ## ELBO
        loss = (reconstruction_loss + log_alpha + weighted_kl).mean()

        return LossOutput(loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl)

    def _get_log_importance_weights(self, x, z, mask, scoring_mask, qz, pz, px, pm, mechanism_weight=1.0):
        log_px = mask * px.log_prob(x)
        log_pm = mechanism_weight * pm.log_prob(mask)

        log_obs = scoring_mask * (log_px + log_pm)

        # (n_samples, n_batch, n_features) -> (n_samples, n_batch)
        log_obs = log_obs.sum(dim=-1)
        log_pz = pz.log_prob(z).sum(dim=-1)
        log_qz = qz.log_prob(z).sum(dim=-1)

        log_weights = log_obs + log_pz - log_qz

        return log_weights

    def _iwae_loss(
        self,
        x,
        z,
        mask,
        scoring_mask,
        qz,
        pz,
        px,
        pm,
        mechanism_weight: float = 1.0,
        **kwargs,
    ) -> LossOutput:
        log_weights = self._get_log_importance_weights(x, z, mask, scoring_mask, qz, pz, px, pm, mechanism_weight)

        # (n_samples, n_batch) -> (n_batch,)
        log_weight_sum = log_weights.logsumexp(dim=0)

        # Truncated Importance Sampling
        log_weight_sum -= np.log(log_weights.size(0))

        loss = -log_weight_sum.mean()

        return LossOutput(loss=loss, n_obs_minibatch=x.size(0))

    def sample(self):
        raise NotImplementedError
