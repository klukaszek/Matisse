"""JAX/Equinox implementation of CortexModel combining all cortex modules."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Tuple

from .C_cone_spectral_type import DefaultConeSpectralType
from .D_demosaicing import DefaultDemosaicing, ImplicitDemosaicing
from .W_lateral_inhibition_weights import DefaultLateralInhibitionWeights
from .P_cell_position import DefaultCellPosition
from .M_global_movement import DefaultGlobalMovement
from ..NeuralScope import NS_cone_mosaic, NS_internal_percept
from ..Retina.FV_spatial_sampling.helper import compute_required_image_resolution


class CortexModel(eqx.Module):
    """Complete cortical model combining all learnable inverse components.

    This model learns to invert the retinal processing, reconstructing
    internal percepts from optic nerve signals.
    """
    C_cone_spectral_type: DefaultConeSpectralType
    D_demosaicing: DefaultDemosaicing | ImplicitDemosaicing
    W_lateral_inhibition_weights: DefaultLateralInhibitionWeights
    P_cell_position: DefaultCellPosition
    M_global_movement: DefaultGlobalMovement
    ns_ip: NS_internal_percept
    ns_cm: NS_cone_mosaic
    learn_ip_ns: bool = eqx.field(static=True)
    temporal_fusion: str = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 8,
        simulation_size: int = 256,
        required_image_resolution: int = None,
        simulating_tetra: bool = False,
        demosaicing_type: str = "Default",
        demosaicing_base_channels: int = 16,
        demosaicing_compute_dtype: str = "float32",
        demosaicing_context_channels: int = 16,
        demosaicing_context_kernel_size: int = 5,
        demosaicing_hidden_channels: int = 32,
        demosaicing_hidden_kernel_size: int = 1,
        demosaicing_num_frequencies: int = 6,
        demosaicing_omega0: float = 10.0,
        demosaicing_activation: str = "sine",
        demosaicing_conditioning: str = "none",
        demosaicing_gaussian_kernel_size: int = 9,
        demosaicing_gaussian_sigma: float = 2.0,
        demosaicing_gaussian_epsilon: float = 1e-3,
        temporal_fusion: str = "none",
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize cortex model.

        Args:
            latent_dim: Number of latent channels
            simulation_size: Spatial dimension of simulation
            required_image_resolution: Retina field-of-view resolution used by
                the global movement pyramid
            simulating_tetra: Whether simulating tetrachromacy
            demosaicing_type: Demosaicing implementation name
            demosaicing_base_channels: Width of the first U-Net stage
            demosaicing_compute_dtype: Compute dtype for U-Net convolutions
            key: JAX random key for initialization
        """
        keys = jax.random.split(key, 7)
        if temporal_fusion not in {"none", "oracle"}:
            raise ValueError("temporal_fusion must be 'none' or 'oracle'")
        self.temporal_fusion = temporal_fusion

        # Initialize all submodules
        self.C_cone_spectral_type = DefaultConeSpectralType(
            latent_dim=latent_dim,
            simulation_size=simulation_size,
            key=keys[0]
        )

        if demosaicing_type.lower() == "default":
            self.D_demosaicing = DefaultDemosaicing(
                latent_dim=latent_dim,
                base_channels=demosaicing_base_channels,
                compute_dtype=demosaicing_compute_dtype,
                key=keys[1],
            )
        elif demosaicing_type.lower() in {"implicit", "siren"}:
            self.D_demosaicing = ImplicitDemosaicing(
                latent_dim=latent_dim,
                context_channels=demosaicing_context_channels,
                context_kernel_size=demosaicing_context_kernel_size,
                hidden_channels=demosaicing_hidden_channels,
                hidden_kernel_size=demosaicing_hidden_kernel_size,
                num_frequencies=demosaicing_num_frequencies,
                omega0=demosaicing_omega0,
                activation=demosaicing_activation,
                conditioning=demosaicing_conditioning,
                gaussian_kernel_size=demosaicing_gaussian_kernel_size,
                gaussian_sigma=demosaicing_gaussian_sigma,
                gaussian_epsilon=demosaicing_gaussian_epsilon,
                compute_dtype=demosaicing_compute_dtype,
                key=keys[1],
            )
        else:
            raise ValueError(
                f"Unknown demosaicing type '{demosaicing_type}' "
                "(use 'Default' or 'Implicit')"
            )

        self.W_lateral_inhibition_weights = DefaultLateralInhibitionWeights(
            simulation_size=simulation_size,
            key=keys[2]
        )

        self.P_cell_position = DefaultCellPosition(
            simulation_size=simulation_size,
            latent_dim=16,
            num_layers=8,
            key=keys[3]
        )

        self.M_global_movement = DefaultGlobalMovement(
            simulation_size,
            required_image_resolution=required_image_resolution
        )

        # NeuralScope modules
        self.ns_ip = NS_internal_percept(
            latent_dim=latent_dim,
            output_dim=3,
            key=keys[4]
        )

        self.ns_cm = NS_cone_mosaic(
            latent_dim=latent_dim,
            output_dim=4,
            key=keys[5]
        )

        # Disable neural scope for internal percept when simulating tetrachromacy
        self.learn_ip_ns = not simulating_tetra

    def decode(
        self,
        ons: Float[Array, "batch 1 height width"]
    ) -> Float[Array, "batch latent_dim height width"]:
        """Decode optic nerve signal to internal percept.

        Args:
            ons: Optic nerve signal

        Returns:
            Internal percept in latent space
        """
        # Deconvolve lateral inhibition
        pa = self.W_lateral_inhibition_weights.deconvolve(ons)

        # Inject cone identity
        cone_identity = self.C_cone_spectral_type.get_cone_identity_function()
        C_injected_pa = pa * cone_identity

        # Demosaic to reconstruct internal percept
        ip = self.D_demosaicing.demosaic(C_injected_pa, cone_identity)

        return ip

    def decode_fused(
        self,
        ons1: Float[Array, "batch 1 height width"],
        ons2: Float[Array, "batch 1 height width"],
        dxy_1_to_2: Float[Array, "batch 2"],
    ) -> tuple[
        Float[Array, "batch latent_dim height width"],
        Float[Array, "batch 1 height width"],
    ]:
        """Register and fuse two cone-sample fields before demosaicing."""
        cone_identity = self.C_cone_spectral_type.get_cone_identity_function()
        pa1 = self.W_lateral_inhibition_weights.deconvolve(ons1)
        pa2 = self.W_lateral_inhibition_weights.deconvolve(ons2)
        injected1 = pa1 * cone_identity
        injected2 = pa2 * cone_identity

        warped_injected1, valid_mask = self.P_cell_position.efficient_warping(
            injected1, dxy_1_to_2
        )
        support = jnp.broadcast_to(cone_identity ** 2, injected1.shape)
        warped_support1, _ = self.P_cell_position.efficient_warping(
            support, dxy_1_to_2
        )

        combined_activation = injected2 + warped_injected1 * valid_mask
        combined_support = support + warped_support1 * valid_mask
        percept2 = self.D_demosaicing.demosaic(
            combined_activation,
            cone_identity,
            cone_support=combined_support,
        )
        return percept2, valid_mask

    def encode(
        self,
        ip: Float[Array, "batch latent_dim height width"]
    ) -> Float[Array, "batch 1 height width"]:
        """Encode internal percept to optic nerve signal.

        Args:
            ip: Internal percept in latent space

        Returns:
            Optic nerve signal
        """
        # Sample from cone identity
        pa = self.C_cone_spectral_type.C_sampling(ip)

        # Convolve with lateral inhibition
        ons = self.W_lateral_inhibition_weights.convolve(pa)

        return ons

    def main_train(
        self,
        ons1: Float[Array, "batch 1 height width"],
        ons2: Float[Array, "batch 1 height width"],
        linsRGB1: Float[Array, "batch 3 height width"],
        true_dxy: Float[Array, "batch 2"],
        cone_mosaic: Float[Array, "batch 4 height width"],
        kernel_size: int
    ) -> Tuple[float, float, float, Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """Main training forward pass.

        Args:
            ons1: First optic nerve signal
            ons2: Second optic nerve signal
            linsRGB1: Ground truth linear RGB for first frame
            true_dxy: True eye movement
            cone_mosaic: Ground truth cone mosaic
            kernel_size: Kernel size for loss masking

        Returns:
            Tuple of (main_loss, ns_cm_loss, ns_ip_loss, pred_cone_mosaic, pred_ons2, pred_dxy)
        """
        # Decode first frame
        warped_ip1 = self.decode(ons1)

        # Predict eye movement
        P = kernel_size // 2
        pred_dxy = self.M_global_movement(ons1, ons2, self.P_cell_position, true_dxy)

        # Warp internal percept according to predicted movement
        pred_warped_ip2, mask2 = self.P_cell_position.efficient_warping(
            warped_ip1,
            pred_dxy[:, 0, :]  # Remove middle dimension
        )

        # Encode to predict second ONS
        ons2_pred = self.encode(pred_warped_ip2)

        # Main loss: masked MSE between predicted and actual ONS
        # Crop boundaries to avoid edge effects
        ons2_crop = ons2[:, :, P:-P, P:-P]
        ons2_pred_crop = ons2_pred[:, :, P:-P, P:-P]
        mask2_crop = mask2[:, :, P:-P, P:-P]

        main_loss = jnp.sum(
            jnp.sum(
                ((ons2_pred_crop - ons2_crop) ** 2) * mask2_crop,
                axis=0
            ) / (jnp.sum(mask2_crop, axis=0) + 1)
        )

        # Neural scope: cone mosaic loss
        C = jax.lax.stop_gradient(self.C_cone_spectral_type.get_cone_identity_function())
        
        # Apply NS directly (ns_cm handles shape (B, latent_dim, H, W))
        # C is (1, latent_dim, H, W), so we can pass it directly
        pred_cone_mosaic = self.ns_cm(C)

        ns_cm_loss = jnp.sum(
            (pred_cone_mosaic - cone_mosaic)[:, :, P:-P, P:-P] ** 2
        )

        # Neural scope: internal percept loss
        if self.learn_ip_ns:
            warped_ip1_detached = jax.lax.stop_gradient(warped_ip1)
            # ns_ip expects (B, latent_dim, H, W), so pass directly
            pred_warped_linsRGB1 = self.ns_ip(warped_ip1_detached)

            ns_ip_loss = jnp.sum(
                (pred_warped_linsRGB1[:, :3] - linsRGB1[:, :3])[:, :, P:-P, P:-P] ** 2
            )
        else:
            ns_ip_loss = 0.0

        return (
            main_loss,
            ns_cm_loss,
            ns_ip_loss,
            pred_cone_mosaic,
            ons2_pred * mask2,
            pred_dxy
        )

    def get_unwarped_percept(
        self,
        warped_ip_sRGB: Float[Array, "batch channels height width"],
        required_image_resolution: int = None
    ) -> Tuple[Float[Array, "batch height width channels"], Float[Array, "height width"]]:
        """Get unwarped percept by applying inverse cell position transform.

        Args:
            warped_ip_sRGB: Warped internal percept in sRGB space
            required_image_resolution: Output grid resolution. MUST be supplied
                (as a static value) when this method is wrapped in jit: it is
                used as an array shape, and the fallback computes it via
                ``compute_required_image_resolution`` which calls ``int()`` on a
                traced value and cannot run under jit. Callers should pass the
                retina's known resolution (``retina.required_image_resolution``).

        Returns:
            Tuple of (internal_percept_sRGB, invalid_regions)
        """
        # Get XY locations
        xy_full = self.P_cell_position.get_XY_default_locations()
        if required_image_resolution is None:
            required_image_resolution = compute_required_image_resolution(xy_full)

        # Generate grid
        grid = self.M_global_movement.generate_grid_fixed(
            xy_full[0, :, 0, 0],
            xy_full[0, :, -1, 0],
            xy_full[0, :, 0, -1],
            xy_full[0, :, -1, -1],
            required_image_resolution
        )

        # Get UV locations
        uvs = self.P_cell_position.get_UV_locations(
            jnp.transpose(grid, (2, 0, 1))[None, ...]
        )
        uvs = jnp.tile(uvs, (len(warped_ip_sRGB), 1, 1, 1))

        # Grid sample to unwarp
        from jax.scipy.ndimage import map_coordinates

        def grid_sample(img, grid_uv):
            """Sample image (B, C, H, W) at UV grid (B, 2, H', W') in [-1, 1]."""
            H, W = img.shape[2], img.shape[3]
            # (B, 2, H', W') -> (B, H', W', 2), then [-1,1] -> source pixel coords
            # (align_corners=True convention, scaled by the *source* image size).
            grid_perm = jnp.transpose(grid_uv, (0, 2, 3, 1))
            grid_pixel = (grid_perm + 1) * jnp.array([[[[(W - 1) / 2, (H - 1) / 2]]]])

            def sample_batch_item(img_single, grid_single):
                coords = jnp.stack([grid_single[:, :, 1], grid_single[:, :, 0]], axis=0)
                # vmap over channels rather than a Python loop: keeps the traced
                # graph small so XLA can fuse the gathers and reuse buffers.
                return jax.vmap(
                    lambda ch: map_coordinates(ch, coords, order=1, mode='constant', cval=0)
                )(img_single)

            return jax.vmap(sample_batch_item)(img, grid_pixel)

        internal_percept_sRGB = grid_sample(warped_ip_sRGB, uvs)

        # Transpose to (batch, height, width, channels) and clip
        internal_percept_sRGB = jnp.transpose(internal_percept_sRGB, (0, 2, 3, 1))
        internal_percept_sRGB = jnp.clip(internal_percept_sRGB, 0, 1)

        # Compute invalid regions (outside [-1, 1] range)
        invalid_regions = ((uvs <= -1) | (uvs >= 1)).sum(axis=1) > 0
        invalid_regions = invalid_regions[0].astype(jnp.int32)

        return internal_percept_sRGB, invalid_regions
