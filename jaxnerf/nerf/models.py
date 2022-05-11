# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
from typing import Any, Callable
from flax import linen as nn
from jax import random
import jax.numpy as jnp

from nerf import model_utils
from nerf import utils
from nerf.nerf_transformer import get_nerf_transformer


def get_model(key, example_batch, args):
    """A helper function that wraps around a 'model zoo'."""
    model_dict = {
        "nerf": construct_nerf,
        "next": construct_next,
    }
    return model_dict[args.model](key, example_batch, args)


class NeXT(nn.Module):
    num_coarse_samples: int  # The number of samples for the coarse nerf.
    num_fine_samples: int  # The number of samples for the fine nerf.
    use_viewdirs: bool  # If True, use viewdirs as an input.
    near: float  # The distance to the near plane
    far: float  # The distance to the far plane
    noise_std: float  # The std dev of noise added to raw sigma.
    white_bkgd: bool  # If True, use a white background.
    min_deg_point: int  # The minimum degree of positional encoding for positions.
    max_deg_point: int  # The maximum degree of positional encoding for positions.
    deg_view: int  # The degree of positional encoding for viewdirs.
    lindisp: bool  # If True, sample linearly in disparity rather than in depth.
    rgb_activation: Callable[Ellipsis, Any]  # Output RGB activation.
    sigma_activation: Callable[Ellipsis, Any]  # Output sigma activation.
    legacy_posenc_order: bool  # Keep the same ordering as the original tf code.
    embed_dim: int = 256
    depth: int = 2
    output_c: int = 3
    num_heads: int = 8
    mlp_ratio: float = 1.
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    attn_drop_rate: float = 0.
    norm_layer: Any = nn.LayerNorm
    act_layer: Any = nn.relu
    skips: str = '0,1'
    window_size: int = 0
    backbone: str = 'next_b'
    shift_size: int = 0

    @nn.compact
    def __call__(self, rng_0, rng_1, rays, randomized, train):
        """NeXT Model.

        Args:
          rng_0: jnp.ndarray, random number generator for coarse model sampling.
          rng_1: jnp.ndarray, random number generator for fine model sampling.
          rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
          randomized: bool, use randomized stratified sampling.

        Returns:
          ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
        """

        # Stratified sampling along rays
        key, rng_0 = random.split(rng_0)
        z_vals, samples = model_utils.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )
        samples_enc = model_utils.posenc(
            samples,
            self.min_deg_point,
            self.max_deg_point,
            self.legacy_posenc_order,
        )

        # Construct the "coarse" model.
        kwargs = {
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'output_c': self.output_c,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'norm_layer': self.norm_layer,
            'act_layer': self.act_layer,
            'skips': self.skips,
            'drop_path_rate': self.drop_path_rate,
            'window_size': self.window_size,
            'shift_size': self.shift_size
        }

        coarse_model = get_nerf_transformer(name=self.backbone, **kwargs)

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = model_utils.posenc(
                rays.viewdirs,
                0,
                self.deg_view,
                self.legacy_posenc_order,
            )
            raw_rgb, raw_sigma = coarse_model(samples_enc, viewdirs_enc, train=train)
        else:
            raw_rgb, raw_sigma = coarse_model(samples_enc, train=train)
        # Add noises to regularize the density predictions if needed
        key, rng_0 = random.split(rng_0)
        raw_sigma = model_utils.add_gaussian_noise(
            key,
            raw_sigma,
            self.noise_std,
            randomized,
        )
        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)
        # Volumetric rendering.
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]
        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            key, rng_1 = random.split(rng_1)
            z_vals, samples = model_utils.sample_pdf(
                key,
                z_vals_mid,
                weights[Ellipsis, 1:-1],
                rays.origins,
                rays.directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )
            samples_enc = model_utils.posenc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
                self.legacy_posenc_order,
            )

            # Construct the "fine" model.
            fine_model = get_nerf_transformer(name=self.backbone, **kwargs)

            if self.use_viewdirs:
                raw_rgb, raw_sigma = fine_model(samples_enc, viewdirs_enc, train=train)
            else:
                raw_rgb, raw_sigma = fine_model(samples_enc, train=train)
            key, rng_1 = random.split(rng_1)
            raw_sigma = model_utils.add_gaussian_noise(
                key,
                raw_sigma,
                self.noise_std,
                randomized,
            )
            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)
            comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                rays.directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))
        return ret


class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""
    num_coarse_samples: int  # The number of samples for the coarse nerf.
    num_fine_samples: int  # The number of samples for the fine nerf.
    use_viewdirs: bool  # If True, use viewdirs as an input.
    near: float  # The distance to the near plane
    far: float  # The distance to the far plane
    noise_std: float  # The std dev of noise added to raw sigma.
    net_depth: int  # The depth of the first part of MLP.
    net_width: int  # The width of the first part of MLP.
    net_depth_condition: int  # The depth of the second part of MLP.
    net_width_condition: int  # The width of the second part of MLP.
    net_activation: Callable[Ellipsis, Any]  # MLP activation
    skip_layer: int  # How often to add skip connections.
    num_rgb_channels: int  # The number of RGB channels.
    num_sigma_channels: int  # The number of density channels.
    white_bkgd: bool  # If True, use a white background.
    min_deg_point: int  # The minimum degree of positional encoding for positions.
    max_deg_point: int  # The maximum degree of positional encoding for positions.
    deg_view: int  # The degree of positional encoding for viewdirs.
    lindisp: bool  # If True, sample linearly in disparity rather than in depth.
    rgb_activation: Callable[Ellipsis, Any]  # Output RGB activation.
    sigma_activation: Callable[Ellipsis, Any]  # Output sigma activation.
    legacy_posenc_order: bool  # Keep the same ordering as the original tf code.

    @nn.compact
    def __call__(self, rng_0, rng_1, rays, randomized, train=True):
        """Nerf Model.

        Args:
          rng_0: jnp.ndarray, random number generator for coarse model sampling.
          rng_1: jnp.ndarray, random number generator for fine model sampling.
          rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
          randomized: bool, use randomized stratified sampling.

        Returns:
          ret: list, [(rgb_coarse, disp_coarse, acc_coarse), (rgb, disp, acc)]
        """
        # Stratified sampling along rays
        key, rng_0 = random.split(rng_0)
        z_vals, samples = model_utils.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            self.num_coarse_samples,
            self.near,
            self.far,
            randomized,
            self.lindisp,
        )
        samples_enc = model_utils.posenc(
            samples,
            self.min_deg_point,
            self.max_deg_point,
            self.legacy_posenc_order,
        )

        # Construct the "coarse" MLP.
        coarse_mlp = model_utils.MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            net_depth_condition=self.net_depth_condition,
            net_width_condition=self.net_width_condition,
            net_activation=self.net_activation,
            skip_layer=self.skip_layer,
            num_rgb_channels=self.num_rgb_channels,
            num_sigma_channels=self.num_sigma_channels)

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = model_utils.posenc(
                rays.viewdirs,
                0,
                self.deg_view,
                self.legacy_posenc_order,
            )
            raw_rgb, raw_sigma = coarse_mlp(samples_enc, viewdirs_enc)
        else:
            raw_rgb, raw_sigma = coarse_mlp(samples_enc)
        # Add noises to regularize the density predictions if needed
        key, rng_0 = random.split(rng_0)
        raw_sigma = model_utils.add_gaussian_noise(
            key,
            raw_sigma,
            self.noise_std,
            randomized,
        )
        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)
        # Volumetric rendering.
        comp_rgb, disp, acc, weights = model_utils.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            rays.directions,
            white_bkgd=self.white_bkgd,
        )
        ret = [
            (comp_rgb, disp, acc),
        ]
        # Hierarchical sampling based on coarse predictions
        if self.num_fine_samples > 0:
            z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
            key, rng_1 = random.split(rng_1)
            z_vals, samples = model_utils.sample_pdf(
                key,
                z_vals_mid,
                weights[Ellipsis, 1:-1],
                rays.origins,
                rays.directions,
                z_vals,
                self.num_fine_samples,
                randomized,
            )
            samples_enc = model_utils.posenc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
                self.legacy_posenc_order,
            )

            # Construct the "fine" MLP.
            fine_mlp = model_utils.MLP(
                net_depth=self.net_depth,
                net_width=self.net_width,
                net_depth_condition=self.net_depth_condition,
                net_width_condition=self.net_width_condition,
                net_activation=self.net_activation,
                skip_layer=self.skip_layer,
                num_rgb_channels=self.num_rgb_channels,
                num_sigma_channels=self.num_sigma_channels)

            if self.use_viewdirs:
                raw_rgb, raw_sigma = fine_mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_sigma = fine_mlp(samples_enc)
            key, rng_1 = random.split(rng_1)
            raw_sigma = model_utils.add_gaussian_noise(
                key,
                raw_sigma,
                self.noise_std,
                randomized,
            )
            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)
            comp_rgb, disp, acc, unused_weights = model_utils.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                rays.directions,
                white_bkgd=self.white_bkgd,
            )
            ret.append((comp_rgb, disp, acc))
        return ret


def construct_nerf(key, example_batch, args):
    """Construct a Neural Radiance Field.

    Args:
      key: jnp.ndarray. Random number generator.
      example_batch: dict, an example of a batch of data.
      args: FLAGS class. Hyperparameters of nerf.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    net_activation = getattr(nn, str(args.net_activation))
    rgb_activation = getattr(nn, str(args.rgb_activation))
    sigma_activation = getattr(nn, str(args.sigma_activation))

    # Assert that rgb_activation always produces outputs in [0, 1], and
    # sigma_activation always produce non-negative outputs.
    x = jnp.exp(jnp.linspace(-90, 90, 1024))
    x = jnp.concatenate([-x[::-1], x], 0)

    rgb = rgb_activation(x)
    if jnp.any(rgb < 0) or jnp.any(rgb > 1):
        raise NotImplementedError(
            "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
                .format(args.rgb_activation))

    sigma = sigma_activation(x)
    if jnp.any(sigma < 0):
        raise NotImplementedError(
            "Choice of sigma_activation `{}` produces negative densities".format(
                args.sigma_activation))

    model = NerfModel(
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        use_viewdirs=args.use_viewdirs,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        net_depth=args.net_depth,
        net_width=args.net_width,
        net_depth_condition=args.net_depth_condition,
        net_width_condition=args.net_width_condition,
        skip_layer=args.skip_layer,
        num_rgb_channels=args.num_rgb_channels,
        num_sigma_channels=args.num_sigma_channels,
        lindisp=args.lindisp,
        net_activation=net_activation,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order)
    rays = example_batch["rays"]
    key1, key2, key3 = random.split(key, num=3)

    init_variables = model.init(
        key1,
        rng_0=key2,
        rng_1=key3,
        rays=utils.namedtuple_map(lambda x: x[0], rays),
        randomized=args.randomized)

    return model, init_variables


def construct_next(key, example_batch, args):
    """Construct a Neural Radiance Field.

    Args:
      key: jnp.ndarray. Random number generator.
      example_batch: dict, an example of a batch of data.
      args: FLAGS class. Hyperparameters of nerf.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    net_activation = getattr(nn, str(args.net_activation))
    rgb_activation = getattr(nn, str(args.rgb_activation))
    sigma_activation = getattr(nn, str(args.sigma_activation))

    # Assert that rgb_activation always produces outputs in [0, 1], and
    # sigma_activation always produce non-negative outputs.
    x = jnp.exp(jnp.linspace(-90, 90, 1024))
    x = jnp.concatenate([-x[::-1], x], 0)

    rgb = rgb_activation(x)
    if jnp.any(rgb < 0) or jnp.any(rgb > 1):
        raise NotImplementedError(
            "Choice of rgb_activation `{}` produces colors outside of [0, 1]"
                .format(args.rgb_activation))

    sigma = sigma_activation(x)
    if jnp.any(sigma < 0):
        raise NotImplementedError(
            "Choice of sigma_activation `{}` produces negative densities".format(
                args.sigma_activation))

    model = NeXT(
        backbone=args.backbone,
        shift_size=args.shift_size,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        output_c=args.output_c,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        attn_drop_rate=args.attn_drop_rate,
        norm_layer=nn.LayerNorm,
        act_layer=net_activation,
        use_viewdirs=args.use_viewdirs,
        skips=args.skips,
        min_deg_point=args.min_deg_point,
        max_deg_point=args.max_deg_point,
        deg_view=args.deg_view,
        num_coarse_samples=args.num_coarse_samples,
        num_fine_samples=args.num_fine_samples,
        near=args.near,
        far=args.far,
        noise_std=args.noise_std,
        white_bkgd=args.white_bkgd,
        lindisp=args.lindisp,
        rgb_activation=rgb_activation,
        sigma_activation=sigma_activation,
        legacy_posenc_order=args.legacy_posenc_order)
    rays = example_batch["rays"]
    key1, key2, key3 = random.split(key, num=3)

    init_variables = model.init(
        key1,
        rng_0=key2,
        rng_1=key3,
        rays=utils.namedtuple_map(lambda x: x[0], rays),
        randomized=args.randomized,
        train=True
    )

    return model, init_variables
