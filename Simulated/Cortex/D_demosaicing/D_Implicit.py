"""Local implicit-field demosaicing with Fourier coordinate features."""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class LocalImplicitField(eqx.Module):
    """Continuous residual field conditioned on local cone-activation context."""

    context_conv: eqx.nn.Conv2d
    input_projection: eqx.nn.Conv2d
    hidden_projection: eqx.nn.Conv2d
    output_projection: eqx.nn.Conv2d
    latent_dim: int = eqx.field(static=True)
    num_frequencies: int = eqx.field(static=True)
    omega0: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    context_kernel_size: int = eqx.field(static=True)
    hidden_kernel_size: int = eqx.field(static=True)
    conditioning: str = eqx.field(static=True)
    gaussian_kernel_size: int = eqx.field(static=True)
    gaussian_sigma: float = eqx.field(static=True)
    gaussian_epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 8,
        context_channels: int = 16,
        context_kernel_size: int = 5,
        hidden_channels: int = 32,
        hidden_kernel_size: int = 1,
        num_frequencies: int = 6,
        omega0: float = 10.0,
        activation: str = "sine",
        conditioning: str = "none",
        gaussian_kernel_size: int = 9,
        gaussian_sigma: float = 2.0,
        gaussian_epsilon: float = 1e-3,
        *,
        key: jax.Array,
    ):
        if hidden_channels < latent_dim:
            raise ValueError("hidden_channels must be at least latent_dim")
        if num_frequencies < 0:
            raise ValueError("num_frequencies cannot be negative")
        if context_channels < latent_dim:
            raise ValueError("context_channels must be at least latent_dim")
        if context_kernel_size < 1 or context_kernel_size % 2 == 0:
            raise ValueError("context_kernel_size must be a positive odd integer")
        if activation not in {"sine", "relu"}:
            raise ValueError("activation must be 'sine' or 'relu'")
        if hidden_kernel_size < 1 or hidden_kernel_size % 2 == 0:
            raise ValueError("hidden_kernel_size must be a positive odd integer")
        if conditioning not in {"none", "gaussian"}:
            raise ValueError("conditioning must be 'none' or 'gaussian'")
        if gaussian_kernel_size < 1 or gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if gaussian_epsilon <= 0:
            raise ValueError("gaussian_epsilon must be positive")

        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.omega0 = omega0
        self.activation = activation
        self.context_kernel_size = context_kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.conditioning = conditioning
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_epsilon = gaussian_epsilon

        keys = jax.random.split(key, 4)
        self.context_conv = eqx.nn.Conv2d(
            latent_dim,
            context_channels,
            kernel_size=context_kernel_size,
            padding=0,
            use_bias=False,
            key=keys[0],
        )

        fourier_channels = 4 * num_frequencies
        conditioning_channels = 2 * latent_dim if conditioning == "gaussian" else 0
        input_channels = (
            latent_dim + context_channels + fourier_channels + conditioning_channels
        )
        self.input_projection = eqx.nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=1,
            key=keys[1],
        )
        self.hidden_projection = eqx.nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=hidden_kernel_size,
            padding=0,
            key=keys[2],
        )
        self.output_projection = eqx.nn.Conv2d(
            hidden_channels,
            latent_dim,
            kernel_size=1,
            key=keys[3],
        )

        if activation == "sine":
            first_bound = 1.0 / input_channels
            hidden_bound = math.sqrt(6.0 / hidden_channels) / omega0
        else:
            first_bound = math.sqrt(6.0 / input_channels)
            hidden_bound = math.sqrt(6.0 / hidden_channels)
        self.input_projection = eqx.tree_at(
            lambda layer: layer.weight,
            self.input_projection,
            jax.random.uniform(
                keys[1],
                self.input_projection.weight.shape,
                minval=-first_bound,
                maxval=first_bound,
            ),
        )
        self.hidden_projection = eqx.tree_at(
            lambda layer: layer.weight,
            self.hidden_projection,
            jax.random.uniform(
                keys[2],
                self.hidden_projection.weight.shape,
                minval=-hidden_bound,
                maxval=hidden_bound,
            ),
        )
        self.output_projection = eqx.tree_at(
            lambda layer: layer.weight,
            self.output_projection,
            jnp.zeros_like(self.output_projection.weight),
        )
        if self.output_projection.bias is not None:
            self.output_projection = eqx.tree_at(
                lambda layer: layer.bias,
                self.output_projection,
                jnp.zeros_like(self.output_projection.bias),
            )

    def _fourier_features(self, height: int, width: int, dtype) -> jax.Array:
        y = jnp.linspace(-1.0, 1.0, height, dtype=dtype)
        x = jnp.linspace(-1.0, 1.0, width, dtype=dtype)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        if self.num_frequencies == 0:
            return jnp.empty((0, height, width), dtype=dtype)

        frequencies = 2.0 ** jnp.arange(self.num_frequencies, dtype=dtype)
        x_phase = jnp.pi * frequencies[:, None, None] * xx[None]
        y_phase = jnp.pi * frequencies[:, None, None] * yy[None]
        return jnp.concatenate(
            (jnp.sin(x_phase), jnp.cos(x_phase), jnp.sin(y_phase), jnp.cos(y_phase)),
            axis=0,
        )

    def _gaussian_blur(self, value: jax.Array) -> jax.Array:
        """Apply the same fixed Gaussian independently to every latent channel."""
        radius = self.gaussian_kernel_size // 2
        axis = jnp.arange(-radius, radius + 1, dtype=value.dtype)
        kernel_1d = jnp.exp(-(axis ** 2) / (2 * self.gaussian_sigma ** 2))
        kernel_1d = kernel_1d / jnp.sum(kernel_1d)

        # Express the separable filter as weighted shifts. Grouped/depthwise
        # convolution has very high compile cost on jax-mps for this shape,
        # while these static slices fuse into two small Metal kernels.
        horizontal_input = jnp.pad(
            value,
            ((0, 0), (0, 0), (radius, radius)),
            mode="reflect",
        )
        width = value.shape[-1]
        horizontal = sum(
            kernel_1d[offset]
            * horizontal_input[:, :, offset:offset + width]
            for offset in range(self.gaussian_kernel_size)
        )
        vertical_input = jnp.pad(
            horizontal,
            ((0, 0), (radius, radius), (0, 0)),
            mode="reflect",
        )
        height = value.shape[-2]
        return sum(
            kernel_1d[offset]
            * vertical_input[:, offset:offset + height, :]
            for offset in range(self.gaussian_kernel_size)
        )

    def __call__(
        self,
        injected_activation: Float[Array, "latent_dim height width"],
        cone_identity: Float[Array, "latent_dim height width"] | None = None,
        cone_support: Float[Array, "latent_dim height width"] | None = None,
    ) -> Float[Array, "latent_dim height width"]:
        height, width = injected_activation.shape[-2:]
        gaussian_seed = None
        gaussian_support = None
        context_input = injected_activation
        if self.conditioning == "gaussian":
            if cone_identity is None and cone_support is None:
                raise ValueError(
                    "gaussian conditioning requires cone_identity or cone_support"
                )
            raw_support = (
                cone_identity ** 2 if cone_support is None else cone_support
            )
            gaussian_support = self._gaussian_blur(raw_support)
            gaussian_seed = self._gaussian_blur(injected_activation) / jnp.maximum(
                gaussian_support, self.gaussian_epsilon
            )
            context_input = gaussian_seed

        def activate(value):
            if self.activation == "sine":
                return jnp.sin(self.omega0 * value)
            return jax.nn.relu(value)

        context_radius = self.context_kernel_size // 2
        padded = jnp.pad(
            context_input,
            (
                (0, 0),
                (context_radius, context_radius),
                (context_radius, context_radius),
            ),
            mode="reflect",
        )
        context = self.context_conv(padded)
        feature_parts = [injected_activation, context]
        if gaussian_seed is not None:
            feature_parts.extend((gaussian_seed, gaussian_support))
        feature_parts.append(
            self._fourier_features(height, width, injected_activation.dtype)
        )
        features = jnp.concatenate(feature_parts, axis=0)
        hidden = activate(self.input_projection(features))
        if self.hidden_kernel_size > 1:
            radius = self.hidden_kernel_size // 2
            hidden = jnp.pad(
                hidden,
                ((0, 0), (radius, radius), (radius, radius)),
                mode="reflect",
            )
        hidden = activate(self.hidden_projection(hidden))
        residual = self.output_projection(hidden)
        base = gaussian_seed if gaussian_seed is not None else injected_activation
        return base + residual


class ImplicitDemosaicing(eqx.Module):
    """Drop-in D_demosaicing implementation backed by a local implicit field."""

    demosaicing: LocalImplicitField
    latent_dim: int = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 8,
        context_channels: int = 16,
        context_kernel_size: int = 5,
        hidden_channels: int = 32,
        hidden_kernel_size: int = 1,
        num_frequencies: int = 6,
        omega0: float = 10.0,
        activation: str = "sine",
        conditioning: str = "none",
        gaussian_kernel_size: int = 9,
        gaussian_sigma: float = 2.0,
        gaussian_epsilon: float = 1e-3,
        compute_dtype: str = "float32",
        *,
        key: jax.Array,
    ):
        if compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "compute_dtype must be 'float32', 'float16', or 'bfloat16'"
            )
        self.latent_dim = latent_dim
        self.compute_dtype = compute_dtype
        self.demosaicing = LocalImplicitField(
            latent_dim=latent_dim,
            context_channels=context_channels,
            context_kernel_size=context_kernel_size,
            hidden_channels=hidden_channels,
            hidden_kernel_size=hidden_kernel_size,
            num_frequencies=num_frequencies,
            omega0=omega0,
            activation=activation,
            conditioning=conditioning,
            gaussian_kernel_size=gaussian_kernel_size,
            gaussian_sigma=gaussian_sigma,
            gaussian_epsilon=gaussian_epsilon,
            key=key,
        )

    def demosaic(
        self,
        injected_activation: jax.Array,
        cone_identity: jax.Array | None = None,
        cone_support: jax.Array | None = None,
    ) -> jax.Array:
        if cone_identity is None:
            identity = None
            identity_in_axes = None
        elif cone_identity.shape[0] == 1:
            identity = cone_identity[0]
            identity_in_axes = None
        else:
            identity = cone_identity
            identity_in_axes = 0
        if cone_support is None:
            support = None
            support_in_axes = None
        elif cone_support.shape[0] == 1:
            support = cone_support[0]
            support_in_axes = None
        else:
            support = cone_support
            support_in_axes = 0
        if self.compute_dtype == "float32":
            return jax.vmap(
                self.demosaicing,
                in_axes=(0, identity_in_axes, support_in_axes),
            )(
                injected_activation, identity, support
            )

        dtype = jnp.float16 if self.compute_dtype == "float16" else jnp.bfloat16
        compute_model = jax.tree.map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            self.demosaicing,
        )
        compute_identity = None if identity is None else identity.astype(dtype)
        compute_support = None if support is None else support.astype(dtype)
        output = jax.vmap(
            compute_model,
            in_axes=(0, identity_in_axes, support_in_axes),
        )(
            injected_activation.astype(dtype), compute_identity, compute_support
        )
        return output.astype(injected_activation.dtype)

    def __call__(self, injected_activation: jax.Array) -> jax.Array:
        return self.demosaic(injected_activation)
