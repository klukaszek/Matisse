"""JAX/Equinox implementation of spatial sampling with foveation (MIP-mapping)."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from .helper import get_cone_sampling_map


class DefaultSpatialSampling(eqx.Module):
    """Spatial sampling with foveation using MIP-mapping.

    Implements multi-resolution sampling that simulates the non-uniform
    spatial sampling of the retina, with higher resolution at the fovea
    and lower resolution in the periphery.
    """
    cone_locs: Float[Array, "height width 2"]
    mip_level: Float[Array, "height width"]
    ll: Int[Array, "max_mip_level height width"]
    ul: Int[Array, "max_mip_level height width"]
    dl: Float[Array, "1 1 1 height width"]
    du: Float[Array, "1 1 1 height width"]
    grid: Float[Array, "1 height width 2"]
    max_mip_level: int = eqx.field(static=True)
    optic_nerve_signal_dim: int = eqx.field(static=True)
    max_shift_size: int = eqx.field(static=True)
    required_image_resolution: int = eqx.field(static=True)

    def __init__(
        self,
        simulation_size: int = 256,
        max_shift_size: int = 128,
        cone_distribution_type: str = 'Human',
        root_dir: str = None
    ):
        """Initialize spatial sampling module.

        Args:
            simulation_size: Dimension of optic nerve signal
            max_shift_size: Maximum shift size for eye movements
            cone_distribution_type: Type of cone distribution
            root_dir: Root directory for loading cone data
        """
        self.optic_nerve_signal_dim = simulation_size
        self.max_shift_size = max_shift_size

        # Get cone sampling map with MIP levels
        cone_locs_np, mip_level_np, required_image_resolution = get_cone_sampling_map(
            simulation_size,
            cone_distribution_type,
            root_dir
        )

        self.required_image_resolution = required_image_resolution
        self.cone_locs = jnp.array(cone_locs_np)
        self.mip_level = jnp.array(mip_level_np)

        # Create grid for sampling (add batch dimension)
        self.grid = self.cone_locs[None, ...]  # (1, H, W, 2)

        # Compute lower and upper MIP levels
        ll = jnp.floor(self.mip_level).astype(jnp.int32)
        ul = jnp.ceil(self.mip_level).astype(jnp.int32)
        self.max_mip_level = int(jnp.max(ul)) + 1

        # Compute interpolation weights
        self.du = self.mip_level - ll
        self.dl = ul - self.mip_level

        # Handle edge case where dl + du != 1
        self.dl = jnp.where(self.dl + self.du != 1, 1.0, self.dl)

        # Add dimensions for broadcasting: (1, 1, 1, H, W)
        self.dl = self.dl[None, None, None, :, :]
        self.du = self.du[None, None, None, :, :]

        # One-hot encode MIP levels for selection
        # Shape: (max_mip_level, H, W)
        self.ll = jax.nn.one_hot(ll, num_classes=self.max_mip_level, axis=0).astype(jnp.int32)
        self.ul = jax.nn.one_hot(ul, num_classes=self.max_mip_level, axis=0).astype(jnp.int32)

    def __call__(
        self,
        im0: Float[Array, "batch timesteps channels height width"]
    ) -> Float[Array, "batch timesteps channels ons_dim ons_dim"]:
        """Apply spatial sampling with MIP-mapping.

        Args:
            im0: Input image pyramid base level

        Returns:
            Spatially sampled image matching ONS resolution
        """
        BS, N, C, Hm, Wm = im0.shape
        D = self.optic_nerve_signal_dim

        # Build MIP pyramid
        mip = []
        DIV = 1

        for level in range(self.max_mip_level):
            # Current level dimensions
            H_curr = Hm // DIV
            W_curr = Wm // DIV

            # Reshape for sampling: (BS*N, C, H, W)
            im_reshaped = im0.reshape(BS * N, C, H_curr, W_curr)

            # Grid sample using JAX
            # Need to convert grid from [-1, 1] to pixel coordinates
            grid_pixel = (self.grid + 1) * jnp.array([[[[(W_curr - 1) / 2, (H_curr - 1) / 2]]]]) + jnp.array([[[[0, 0]]]])

            # Sample all channels for all batch items
            def sample_single(img, grid_pix):
                """Sample a single image with given grid."""
                from jax.scipy.ndimage import map_coordinates
                # grid_pix is (H, W, 2) with (x, y) coords
                # map_coordinates wants (2, H, W) with (y, x) order
                coords = jnp.stack([grid_pix[:, :, 1], grid_pix[:, :, 0]], axis=0)

                # Sample all channels
                sampled_channels = []
                for c_idx in range(img.shape[0]):
                    sampled = map_coordinates(img[c_idx], coords, order=1, mode='constant', cval=0)
                    sampled_channels.append(sampled)
                return jnp.stack(sampled_channels, axis=0)

            # Apply to all batch items
            sampled = jax.vmap(sample_single, in_axes=(0, None))(im_reshaped, grid_pixel[0])

            # Reshape back to (BS, N, C, D, D)
            sampled = sampled.reshape(BS, N, C, D, D)
            mip.append(sampled)

            # Downsample for next level (MIP-map)
            im0 = mipmap_jax(im0, 2)
            DIV *= 2

        # Stack MIP levels: (max_mip_level, BS, N, C, D, D)
        mip = jnp.stack(mip, axis=0)

        # Select appropriate MIP level using one-hot encoding
        # Reshape one-hot tensors for broadcasting
        # ll: (max_mip_level, H, W) -> (max_mip_level, 1, 1, 1, H, W)
        ll_broadcast = self.ll[:, None, None, None, :, :]
        ul_broadcast = self.ul[:, None, None, None, :, :]

        # Weighted sum over MIP levels
        l = jnp.sum(ll_broadcast * mip, axis=0)  # (BS, N, C, D, D)
        u = jnp.sum(ul_broadcast * mip, axis=0)  # (BS, N, C, D, D)

        # Linear interpolation between levels
        rgb1 = l * self.dl + u * self.du

        return rgb1


def mipmap_jax(orig: Float[Array, "..."], N: int) -> Float[Array, "..."]:
    """Create MIP-map by downsampling with averaging.

    Args:
        orig: Input array (BS, D, C, H, W)
        N: Downsampling factor

    Returns:
        Downsampled array (BS, D, C, H//N, W//N)
    """
    BS, D, C, H, W = orig.shape
    nH = H // N
    nW = W // N

    # Handle case where dimensions aren't evenly divisible
    if H % N != 0 or W % N != 0:
        # Crop to make dimensions divisible
        H_crop = nH * N
        W_crop = nW * N
        orig = orig[:, :, :, :H_crop, :W_crop]

    # Reshape to separate into blocks and average
    reshaped = orig.reshape(BS, D, C, nH, N, nW, N)
    return jnp.mean(reshaped, axis=(4, 6))
