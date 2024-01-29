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

    def forward(self, z: torch.Tensor, x_data: torch.Tensor, *cat_list: int):
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

    def forward(self, z: torch.Tensor, x_data: torch.Tensor, *cat_list: int):
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
        decoder_type: Tunable[
            Literal["selection", "conjunction", "hybrid"]
        ] = "selection",
        loss_type: Tunable[Literal["elbo", "iwae"]] = "elbo",
        n_samples: Tunable[int] = 1,
        max_p_train_dropoout: Tunable[float] = 0.0,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.x_variance = x_variance
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.loss_type = loss_type
        self.n_samples = n_samples
        self.max_p_train_dropoout = max_p_train_dropoout

        if loss_type == "elbo":
            self.loss_fn = self._elbo_loss
        elif loss_type == "iwae":
            self.loss_fn = self._iwae_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

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
        n_samples=None,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """

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

        # @TODO: create protvi encoder
        qz, _ = self.encoder(encoder_input, batch_index, *categorical_input)

        # shape: (n_samples, n_batch, n_latent)
        z = qz.rsample(torch.Size([n_samples]))

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
        use_x_mix=True,
    ):
        """Runs the generative model."""

        n_samples, n_batch = z.size(0), z.size(1)

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

        x_mean, x_var, m_prob = self.decoder(
            decoder_input, x_obs, batch_index, *categorical_input
        )

        # shape: (n_samples * n_batch, n_input) -> (n_samples, n_batch, n_input)
        x_mean = x_mean.view(unpacked_shape)
        x_var = x_var.view(unpacked_shape)
        m_prob = m_prob.view(unpacked_shape)

        return {
            "x_mean": x_mean,
            "x_var": x_var,
            "m_prob": m_prob,
        }

    def _get_distributions(self, inference_outputs, generative_outputs):
        qz = inference_outputs["qz"]
        pz = Normal(loc=0.0, scale=1.0)

        x_mean = generative_outputs["x_mean"]
        x_var = generative_outputs["x_var"]
        m_prob = generative_outputs["m_prob"]

        px = Normal(x_mean, torch.sqrt(x_var))
        pm = Bernoulli(m_prob)

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
        scoring_mask = self._get_scoring_mask(mask, max_p_train_dropoout=self.max_p_train_dropoout)

        return self.loss_fn(
            x=x,
            z=z,
            mask=mask,
            scoring_mask=scoring_mask,
            **distributions,
            kl_weight=kl_weight,
            mechanism_weight=mechanism_weight,
        )

    def _get_scoring_mask(self, mask, max_p_train_dropoout: float):
        # each cell has a different dropout probability with max: max_p_train_dropoout
        p_missing = torch.rand(mask.shape[0], 1) * max_p_train_dropoout 
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
    ):

        lpx = mask * px.log_prob(x)
        lpm = mechanism_weight * pm.log_prob(mask)

        ll = scoring_mask * (lpx + lpm)

        # average over samples, (n_samples, n_batch, n_input) -> (n_batch, n_input)
        lpd = torch.mean(ll, dim=0)

        # sum over features: (n_batch, n_input) -> (n_batch,)
        reconstruction_loss = -torch.sum(lpd, dim=-1)

        ## KL
        # (n_batch, n_latent) -> (n_batch,)
        kl_divergence = kl(qz, pz).sum(dim=-1)
        weighted_kl = kl_divergence * kl_weight

        ## ELBO
        loss = torch.mean(reconstruction_loss + weighted_kl)

        return LossOutput(
            loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_divergence
        )

    def _get_importance_weights(self, x, z, mask, scoring_mask, qz, pz, px, pm, mechanism_weight=1.0):
        # sum over features: (n_samples, n_batch, n_input) -> (n_samples, n_batch)
        lpx = (mask * px.log_prob(x)).sum(dim=-1)
        lpm = mechanism_weight * pm.log_prob(mask).sum(dim=-1)

        ll = scoring_mask * (lpx + lpm)

        lpz = pz.log_prob(z).sum(dim=-1)
        lqz = qz.log_prob(z).sum(dim=-1)

        lw = ll + lpz - lqz

        return lw

    def _iwae_loss(
        self,
        x,
        z,
        qz,
        pz,
        px,
        pm,
        mechanism_weight: float = 1.0,
        **kwargs,
    ):
        lw = self._get_importance_weights(x, z, qz, pz, px, pm, mechanism_weight)

        # sum over samples: (n_samples, n_batch) -> (n_batch,)
        lw_sum = torch.logsumexp(lw, dim=0) - np.log(lw.size(0))

        loss = -torch.mean(lw_sum)

        return LossOutput(loss=loss, n_obs_minibatch=x.size(0))

    def sample(self):
        raise NotImplementedError


## downstream utils
class ProteinMixin:
    @torch.inference_mode()
    def impute(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = None,
        batch_size: int = 32,
        loss_type: Optional[Literal["elbo", "iwae"]] = None,
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
            generative_kwargs = {}
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

                lw = self.module._get_importance_weights(
                    tensors[REGISTRY_KEYS.X_KEY],
                    inference_outputs["z"],
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

        return xs, ps


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
        decoder_type: Tunable[
            Literal["selection", "conjunction", "hybrid"]
        ] = "selection",
        loss_type: Tunable[Literal["elbo", "iwae"]] = "elbo",
        n_samples: Tunable[int] = 1,
        max_p_train_dropoout: Tunable[float] = 0.0,
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
            loss_type=loss_type,
            n_samples=n_samples,
            max_p_train_dropoout=max_p_train_dropoout,
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
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
