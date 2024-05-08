import logging
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    auto_move_data,
)
from scvi.nn import (
    Encoder,
    FCLayers,
)
from torch import nn as nn
from torch.distributions import Bernoulli, Normal, kl_divergence

from ._constants import EXTRA_KEYS

logger = logging.getLogger(__name__)


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

    def forward(
        self,
        z: torch.Tensor,
        x_data: torch.Tensor,
        size_factor: torch.Tensor,
        *cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``

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
        # x_mean *= size_factor
        x_mean = x_mean - size_factor
        x_mean = torch.squeeze(x_mean)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(h)
        elif self.x_variance == "protein":
            x_var = self.x_var.expand(x_mean.shape)

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

    def forward(
        self,
        z: torch.Tensor,
        x_data: torch.Tensor,
        size_factor: torch.Tensor,
        *cat_list: list[int],
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
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
        x_mean = x_mean - size_factor
        x_mean = torch.squeeze(x_mean)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(h)
        elif self.x_variance == "protein":
            x_var = self.x_var.expand(x_mean.shape)

        x_var = torch.exp(x_var)

        # x_mix
        if x_data is None:
            x_mix = x_mean
        else:
            m = x_data != 0
            x_mix = x_data * m + x_mean * ~m

        # z -> p, x -> p
        z_p = self.z_p(h)
        x_p = self.x_p(x_mix)

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

    def forward(
        self,
        z: torch.Tensor,
        x_data: torch.Tensor,
        size_factor: torch.Tensor,
        *cat_list: int,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
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
        x_mean = x_mean - size_factor
        x_mean = torch.squeeze(x_mean)

        if self.x_variance == "protein-cell":
            x_var = self.x_var_decoder(x_h)
        elif self.x_variance == "protein":
            x_var = self.x_var.expand(x_mean.shape)

        x_var = torch.exp(x_var)

        # x_mix
        if x_data is None:
            x_mix = x_mean
        else:
            m = x_data != 0
            x_mix = x_data * m + x_mean * ~m

        m_prob = self.m_prob_decoder(x_mix)

        return x_mean, x_var, m_prob

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
        x_variance: Literal["protein", "protein-cell"] = "protein",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        var_activation: Callable = None,
        decoder_type: Literal["selection", "conjunction", "hybrid"] = "selection",
        loss_type: Literal["elbo", "iwae"] = "elbo",
        n_samples: int = 1,
        max_loss_dropout: float = 0.0,
        use_x_mix=False,
        encode_norm_factors=False,
        use_norm_factors=False,  # @TODO: unused
        n_prior_continuous_cov: int = 0,
        n_prior_cats_per_cov: Optional[Iterable[int]] = None,
        n_norm_continuous_cov: int = 0,  # @TODO: unused
    ):
        super().__init__()
        self.n_latent = n_latent
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
            "selection": SelectionDecoderPROTVI,
            "conjunction": ConjunctionDecoderPROTVI,
            "hybrid": HybridDecoderPROTVI,
        }
        module = modules[decoder_type]

        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = module(
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

        ## Latent prior encoder
        n_input_prior_encoder = n_prior_continuous_cov
        cat_list = list([] if n_prior_cats_per_cov is None else n_prior_cats_per_cov)
        self.prior_encoder = Encoder(
            n_input=n_input_prior_encoder,
            n_output=n_latent,
            n_cat_list=cat_list,
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

        self.size_factor = nn.Parameter(torch.randn(n_input))

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

        norm_cont_key = EXTRA_KEYS.NORM_CONT_COVS_KEY
        norm_cont_covs = tensors[norm_cont_key] if norm_cont_key in tensors.keys() else None

        return {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "prior_cont_covs": prior_cont_covs,
            "prior_cat_covs": prior_cat_covs,
            "norm_cont_covs": norm_cont_covs,
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
        norm_cont_covs=None,
        n_samples=None,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        ## csf adjustment
        if norm_cont_covs is not None:
            norm_continous_input = norm_cont_covs.to(x.device)
            self.csf_offset = True
        else:
            norm_continous_input = torch.empty(0).to(x.device)
            self.csf_offset = False

        ## latent prior
        if prior_cont_covs is not None:
            prior_continous_input = prior_cont_covs.to(x.device)
        else:
            prior_continous_input = torch.empty(0).to(x.device)

        if prior_cat_covs is not None:
            prior_categorical_input = torch.split(prior_cat_covs, 1, dim=1)
        else:
            prior_categorical_input = ()

        if (prior_cont_covs is not None) or (prior_cat_covs is not None):
            pz, _ = self.prior_encoder(prior_continous_input, *prior_categorical_input)
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
        elif self.csf_offset:
            library = norm_continous_input
        else:
            library = self.size_factor

        return {
            "pz": pz,
            "qz": qz,
            "z": z,
            "library": library,
        }

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        size_factor = inference_outputs["library"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        return {
            "x": x,
            "z": z,
            "size_factor": size_factor,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
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

        # x_mean, x_var, m_prob = self.decoder(
        #     decoder_input, x_obs, self.size_factor, batch_index, *categorical_input
        # )

        x_mean, x_var, m_prob = self.decoder(decoder_input, x_obs, size_factor, batch_index, *categorical_input)

        # shape: (n_samples * n_batch, n_input) -> (n_samples, n_batch, n_input)
        x_mean = x_mean.view(unpacked_shape)
        x_var = x_var.view(unpacked_shape)
        m_prob = m_prob.view(unpacked_shape)

        return {
            "x_mean": x_mean,
            "x_var": x_var,
            "m_prob": m_prob,
            "x": x,
        }

    def _get_distributions(self, inference_outputs, generative_outputs):
        qz = inference_outputs["qz"]
        pz = inference_outputs["pz"]
        # size_factor = inference_outputs["library"]

        x_mean = generative_outputs["x_mean"]
        x_var = generative_outputs["x_var"]
        m_prob = generative_outputs["m_prob"]
        x = generative_outputs["x"]
        m = x != 0
        x_mix = x * m + x_mean * ~m

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
        mask,
        scoring_mask,
        qz,
        pz,
        px,
        pm,
        kl_weight: float = 1.0,
        mechanism_weight: float = 1.0,
        **kwargs,
    ) -> LossOutput:
        lpx = mask * px.log_prob(x)
        lpm = mechanism_weight * pm.log_prob(mask)

        ll = scoring_mask * (lpx + lpm)

        # average over samples, (n_samples, n_batch, n_features) -> (n_batch, n_features)
        lpd = ll.sum(dim=0)

        # sum over features: (n_batch, n_features) -> (n_batch,)
        reconstruction_loss = -lpd.sum(dim=-1)

        ## KL
        # (n_batch, n_latent) -> (n_batch,)
        kl = kl_divergence(qz, pz).sum(dim=-1)
        weighted_kl = kl * kl_weight

        ## ELBO
        loss = (reconstruction_loss + weighted_kl).mean()

        return LossOutput(loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl)

    def _get_importance_weights(self, x, z, mask, scoring_mask, qz, pz, px, pm, mechanism_weight=1.0):
        lpx = mask * px.log_prob(x)
        lpm = mechanism_weight * pm.log_prob(mask)

        ll = scoring_mask * (lpx + lpm)

        # sum over features: (n_samples, n_batch, n_features) -> (n_samples, n_batch)
        ll = ll.sum(dim=-1)
        lpz = pz.log_prob(z).sum(dim=-1)
        lqz = qz.log_prob(z).sum(dim=-1)

        lw = ll + lpz - lqz

        return lw

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
        lw = self._get_importance_weights(x, z, mask, scoring_mask, qz, pz, px, pm, mechanism_weight)

        # sum over samples: (n_samples, n_batch) -> (n_batch,)
        lw_sum = lw.logsumexp(dim=0) - np.log(lw.size(0))

        loss = -lw_sum.mean()

        return LossOutput(loss=loss, n_obs_minibatch=x.size(0))

    def sample(self):
        raise NotImplementedError
