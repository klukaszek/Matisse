"""JAX/Equinox implementation of UNet demosaicing module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Optional


class DoubleConv(eqx.Module):
    """(convolution => ReLU) * 2"""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (defaults to out_channels)
            key: JAX random key for initialization
        """
        if mid_channels is None:
            mid_channels = out_channels

        key1, key2 = jax.random.split(key)

        # Conv2d with reflection padding, no bias
        # Note: Equinox Conv2d doesn't have built-in padding modes,
        # so we'll handle reflection padding in the forward pass
        self.conv1 = eqx.nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=3,
            padding=0,  # We'll handle padding manually
            use_bias=False,
            key=key1
        )
        self.conv2 = eqx.nn.Conv2d(
            mid_channels, out_channels,
            kernel_size=3,
            padding=0,  # We'll handle padding manually
            use_bias=False,
            key=key2
        )

    def __call__(self, x: Float[Array, "channels height width"]) -> Float[Array, "out_channels height width"]:
        """Forward pass with reflection padding.

        Note: Expects input without batch dimension. Use vmap for batched inputs.
        """
        # Apply reflection padding (pad=1 for kernel_size=3)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), mode='reflect')
        x = self.conv1(x)
        x = jax.nn.relu(x)

        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), mode='reflect')
        x = self.conv2(x)
        x = jax.nn.relu(x)

        return x


class Down(eqx.Module):
    """Downscaling with maxpool then double conv"""
    double_conv: DoubleConv

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.random.PRNGKey):
        """Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: JAX random key for initialization
        """
        self.double_conv = DoubleConv(in_channels, out_channels, key=key)

    def __call__(self, x: Float[Array, "channels height width"]) -> Float[Array, "out_channels height/2 width/2"]:
        """Forward pass with max pooling.

        Note: Expects input without batch dimension. Use vmap for batched inputs.
        """
        # Max pool with kernel_size=2, stride=2
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 2, 2),
            window_strides=(1, 2, 2),
            padding='VALID'
        )
        x = self.double_conv(x)
        return x


class Up(eqx.Module):
    """Upscaling then double conv"""
    up_conv: eqx.nn.ConvTranspose2d
    double_conv: DoubleConv

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, *, key: jax.random.PRNGKey):
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling (unused, kept for compatibility)
            key: JAX random key for initialization
        """
        key1, key2 = jax.random.split(key)

        # Transpose convolution for upsampling
        self.up_conv = eqx.nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            key=key1
        )

        # Double conv on concatenated features
        self.double_conv = DoubleConv(in_channels, out_channels, key=key2)

    def __call__(
        self,
        x1: Float[Array, "channels height width"],
        x2: Float[Array, "channels/2 height*2 width*2"]
    ) -> Float[Array, "out_channels height*2 width*2"]:
        """Forward pass with skip connection.

        Note: Expects inputs without batch dimension. Use vmap for batched inputs.

        Args:
            x1: Upsampled features from lower level
            x2: Skip connection features from encoder

        Returns:
            Concatenated and processed features
        """
        x1 = self.up_conv(x1)

        # Handle size mismatches with padding
        diff_y = x2.shape[1] - x1.shape[1]
        diff_x = x2.shape[2] - x1.shape[2]

        if diff_y != 0 or diff_x != 0:
            # Pad x1 to match x2 size
            pad_top = diff_y // 2
            pad_bottom = diff_y - pad_top
            pad_left = diff_x // 2
            pad_right = diff_x - pad_left

            x1 = jnp.pad(
                x1,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant'
            )

        # Concatenate along channel dimension
        x = jnp.concatenate([x2, x1], axis=0)

        return self.double_conv(x)


class OutConv(eqx.Module):
    """Output convolution layer"""
    conv: eqx.nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.random.PRNGKey):
        """Initialize output convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: JAX random key for initialization
        """
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=key)

    def __call__(self, x: Float[Array, "channels height width"]) -> Float[Array, "out_channels height width"]:
        """Forward pass.

        Note: Expects input without batch dimension. Use vmap for batched inputs.
        """
        return self.conv(x)


class UNet(eqx.Module):
    """UNet architecture for demosaicing.

    Standard UNet with 4 levels of downsampling and upsampling,
    with skip connections between corresponding levels.
    """
    inc: DoubleConv
    down1: Down
    down2: Down
    down3: Down
    up1: Up
    up2: Up
    up3: Up
    outc: OutConv
    base_channels: int = eqx.field(static=True)

    def __init__(
        self,
        dim_latent: int = 8,
        base_channels: int = 16,
        *,
        key: jax.random.PRNGKey,
    ):
        """Initialize UNet.

        Args:
            dim_latent: Number of latent channels (input and output)
            base_channels: Width of the first U-Net stage
            key: JAX random key for initialization
        """
        if base_channels < 2:
            raise ValueError("base_channels must be at least 2")

        keys = jax.random.split(key, 8)
        self.base_channels = base_channels

        self.inc = DoubleConv(dim_latent, base_channels, key=keys[0])
        self.down1 = Down(base_channels, base_channels * 2, key=keys[1])
        self.down2 = Down(base_channels * 2, base_channels * 4, key=keys[2])
        self.down3 = Down(base_channels * 4, base_channels * 8, key=keys[3])
        self.up1 = Up(base_channels * 8, base_channels * 4, False, key=keys[4])
        self.up2 = Up(base_channels * 4, base_channels * 2, False, key=keys[5])
        self.up3 = Up(base_channels * 2, base_channels, False, key=keys[6])
        self.outc = OutConv(base_channels, dim_latent, key=keys[7])

    def __call__(self, x: Float[Array, "latent_dim height width"]) -> Float[Array, "latent_dim height width"]:
        """Forward pass through UNet.

        Note: Expects input without batch dimension (channels, height, width).
        Use vmap for batched inputs: jax.vmap(unet)(batched_input)

        Args:
            x: Input tensor of shape (latent_dim, height, width)

        Returns:
            Demosaiced output of same shape as input
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x


class DefaultDemosaicing(eqx.Module):
    """Wrapper for UNet demosaicing."""
    demosaicing: UNet
    latent_dim: int = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 16,
        compute_dtype: str = "float32",
        *,
        key: jax.random.PRNGKey,
    ):
        """Initialize demosaicing module.

        Args:
            latent_dim: Number of latent channels
            base_channels: Width of the first U-Net stage
            compute_dtype: Convolution dtype; parameters remain float32
            key: JAX random key for initialization
        """
        if compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "compute_dtype must be 'float32', 'float16', or 'bfloat16'"
            )
        self.latent_dim = latent_dim
        self.compute_dtype = compute_dtype
        self.demosaicing = UNet(
            dim_latent=latent_dim,
            base_channels=base_channels,
            key=key,
        )

    def demosaic(
        self,
        C_encoded_pa: Float[Array, "batch latent_dim height width"],
        cone_identity: Float[Array, "1 latent_dim height width"] | None = None,
        cone_support: Float[Array, "batch latent_dim height width"] | None = None,
    ) -> Float[Array, "batch latent_dim height width"]:
        """Apply demosaicing.

        Args:
            C_encoded_pa: Cone-encoded input with batch dimension
            cone_identity: Unused by the U-Net; accepted for a common interface
            cone_support: Unused by the U-Net; accepted for a common interface

        Returns:
            Demosaiced internal percept
        """
        if self.compute_dtype == "float32":
            return jax.vmap(self.demosaicing)(C_encoded_pa)

        dtype = jnp.float16 if self.compute_dtype == "float16" else jnp.bfloat16
        compute_model = jax.tree.map(
            lambda leaf: leaf.astype(dtype) if eqx.is_inexact_array(leaf) else leaf,
            self.demosaicing,
        )
        output = jax.vmap(compute_model)(C_encoded_pa.astype(dtype))
        return output.astype(C_encoded_pa.dtype)

    def __call__(self, x: Float[Array, "batch latent_dim height width"]) -> Float[Array, "batch latent_dim height width"]:
        """Forward pass (alias for demosaic)."""
        return self.demosaic(x)
