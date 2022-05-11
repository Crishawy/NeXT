from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from functools import partial
import jax
import numpy as np

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
dense_layer = partial(nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
droppath_prngkey = jax.random.PRNGKey(2020)


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)], dtype=np.float32)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    sinusoid_table = jnp.array(sinusoid_table, dtype=jnp.float32)
    return jnp.reshape(sinusoid_table, (1, sinusoid_table.shape[0], sinusoid_table.shape[1]))


class SinusoidPositionEmbs(nn.Module):
    num_samples: int = 192
    embed_dim: int = 256

    def setup(self):
        self.pos_embed = get_sinusoid_encoding_table(self.num_samples, self.embed_dim)

    @nn.compact
    def __call__(self, inputs):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        return inputs + self.pos_embed


def window_partition(x, window_size):
    B, n, C = x.shape
    x = jnp.reshape(x, (B, n // window_size, window_size, C))
    windows = jnp.reshape(x, (-1, window_size, C))
    return windows


def window_reverse(windows, window_size, n):
    B = int(windows.shape[0] / (n / window_size))
    x = jnp.reshape(windows, (B, n // window_size, window_size, -1))
    x = jnp.reshape(x, (B, n, -1))
    return x


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.
    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.
        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init.
        Args:
          inputs: Inputs to the layer.
        Returns:
          Output tensor with shape `(bs, timesteps, in_dim)`.
        """
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                          Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
        output = nn.Dropout(
            rate=self.dropout_rate)(
            output, deterministic=deterministic)
        return output


class DropPath(nn.Module):
    """Create a dropout layer.

      Args:
        rate: the dropout probability.  (_not_ the keep rate!)
      """
    rate: float

    @nn.compact
    def __call__(self, inputs, deterministic=False, rng=None):
        if self.rate == 0. or deterministic:
            return inputs
        keep_prob = 1. - self.rate
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        if rng is None:
            rng = self.make_rng('params')
        random_tensor = keep_prob + jax.random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        output = random_tensor * inputs / keep_prob
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.
    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    window_size: int = 8
    drop_path_rate: float = 0.
    shift_size: int = 0

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Encoder1DBlock module.
        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.
        Returns:
          output after transformer encoder block.
        """
        b, n, c = inputs.shape

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)

        # cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(x, -self.shift_size, axis=1)

        # partition windows
        x_windows = window_partition(x, self.window_size)
        # attn
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads)(
            x_windows, x_windows)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # merge windows
        x = window_reverse(x, self.window_size, n)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(x, self.shift_size, axis=1)

        if self.drop_path_rate > 0.:
            x = DropPath(rate=self.drop_path_rate)(x, deterministic=deterministic, rng=droppath_prngkey)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)
        if self.drop_path_rate > 0.:
            y = DropPath(rate=self.drop_path_rate)(y, deterministic=deterministic, rng=droppath_prngkey)

        return x + y


class LocalNerfTransformer(nn.Module):
    embed_dim: int = 256
    depth: int = 2
    output_c: int = 3
    num_heads: int = 8
    mlp_ratio: float = 1.
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    attn_drop_rate: float = 0.
    norm_layer: Any = nn.LayerNorm
    act_layer: Any = nn.gelu
    use_viewdirs: bool = True
    skips: str = '0,1'
    window_size: int = 0
    shift_size: int = 0

    def forward_features(self, input_pts, train=True):
        x = input_pts

        x = dense_layer(self.embed_dim)(x)
        x = self.norm_layer()(x)

        x = SinusoidPositionEmbs(num_samples=x.shape[1], embed_dim=self.embed_dim)(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not train)

        skips = list(map(int, self.skips.split(',')))
        for i in range(self.depth):
            x = Encoder1DBlock(
                mlp_dim=int(self.embed_dim * self.mlp_ratio),
                dropout_rate=self.drop_rate,
                attention_dropout_rate=self.attn_drop_rate,
                name=f'encoderblock_{i}',
                num_heads=self.num_heads,
                window_size=self.window_size,
                drop_path_rate=self.drop_path_rate,
                shift_size=0 if i % 2 == 0 else self.shift_size
            )(x, deterministic=not train)
            if i in skips:
                x = jnp.concatenate([input_pts, x], -1)
                x = dense_layer(self.embed_dim)(x)
                x = self.act_layer(x)

        x = self.norm_layer()(x)
        return x

    @nn.compact
    def __call__(self, input_pts, input_views=None, train=True):
        h = self.forward_features(input_pts, train)

        if self.use_viewdirs:
            assert input_views is not None
            if len(input_views.shape) < 3:
                input_views = jnp.repeat(jnp.expand_dims(input_views, 1), input_pts.shape[1], 1)
            alpha = dense_layer(1)(h)
            feature = dense_layer(self.embed_dim)(h)
            h = jnp.concatenate([feature, input_views], -1)
            rgb = dense_layer(self.embed_dim // 2)(h)
            rgb = self.act_layer(rgb)
            rgb = dense_layer(3)(rgb)
            return rgb, alpha
        else:
            outputs = dense_layer(self.embed_dim // 2)
            outputs = self.act_layer(outputs)
            outputs = dense_layer(self.output_c)(outputs)

        return outputs


def get_nerf_transformer(name, **kwargs):
    if name == 'next_s':
        default_kwargs = {
            'embed_dim': 192,
            'depth': 2,
            'skips': '0,1',
            'window_size': 64
        }
    elif name == 'next_b':
        default_kwargs = {
            'embed_dim': 256,
            'depth': 2,
            'skips': '0,1',
            'window_size': 64
        }
    elif name == 'next_l':
        default_kwargs = {
            'embed_dim': 256,
            'depth': 4,
            'skips': '0,1,2,3',
            'window_size': 64
        }
    else:
        raise NotImplementedError(name)
    kwargs.update(default_kwargs)
    return LocalNerfTransformer(**kwargs)
