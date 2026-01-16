"""JAX/Equinox implementation of global movement estimation via pyramid matching."""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class DefaultGlobalMovement(eqx.Module):
    """Global movement estimation using multi-scale pyramid matching.

    Estimates eye movement between two optic nerve signals using a coarse-to-fine
    pyramid search strategy. Non-differentiable during training (uses stop_gradient).
    """
    shift_range: tuple = eqx.field(static=True)
    simulation_size: int = eqx.field(static=True)
    shifts: tuple = eqx.field(static=True)

    def __init__(self, simulation_size: int = 256):
        """Initialize global movement module."""
        # Pre-compute constants
        self.simulation_size = simulation_size
        self.shift_range = tuple(range(-2, 3))
        self.shifts = tuple((dx, dy) for dx in self.shift_range for dy in self.shift_range)

    def gaussian_blur(self, ons: Float[Array, "batch 1 height width"]) -> Float[Array, "batch 1 height width"]:
        """Apply Gaussian blur with reflection padding.

        Args:
            ons: Optic nerve signal

        Returns:
            Blurred signal
        """
        # Pad with reflection
        ons = jnp.pad(ons, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='reflect')

        # Apply Gaussian blur (kernel size 9, sigma=2.0)
        kernel_size = 9
        sigma = 2.0
        x = jnp.arange(kernel_size) - (kernel_size - 1) / 2
        gaussian_1d = jnp.exp(-x ** 2 / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / jnp.sum(gaussian_1d)
        gaussian_2d = jnp.outer(gaussian_1d, gaussian_1d)
        gaussian_2d = gaussian_2d / jnp.sum(gaussian_2d)

        # Apply convolution using JAX
        def convolve_single(img):
            return jax.scipy.signal.convolve2d(img[0], gaussian_2d, mode='same')[None, ...]

        blurred = jax.vmap(convolve_single)(ons)

        # Remove padding
        blurred = blurred[:, :, 4:-4, 4:-4]

        return blurred

    def generate_grid_fixed(
        self,
        top_left: Float[Array, "2"],
        bottom_left: Float[Array, "2"],
        top_right: Float[Array, "2"],
        bottom_right: Float[Array, "2"],
        required_image_resolution: int
    ) -> Float[Array, "resolution resolution 2"]:
        """Generate bilinear interpolation grid from corner points.

        Args:
            top_left, bottom_left, top_right, bottom_right: Corner coordinates
            required_image_resolution: Grid resolution

        Returns:
            Interpolated grid
        """
        # Generate meshgrid
        y_coords, x_coords = jnp.meshgrid(
            jnp.linspace(0, 1, required_image_resolution),
            jnp.linspace(0, 1, required_image_resolution),
            indexing='ij'
        )

        # Bilinear interpolation
        top_weights = 1 - y_coords
        bottom_weights = y_coords

        top_interpolated = (1 - x_coords)[:, :, None] * top_left + x_coords[:, :, None] * top_right
        bottom_interpolated = (1 - x_coords)[:, :, None] * bottom_left + x_coords[:, :, None] * bottom_right

        grid = top_weights[:, :, None] * top_interpolated + bottom_weights[:, :, None] * bottom_interpolated

        return grid

    def _shift_single_image(
        self,
        img: Float[Array, "1 height width"],
        cdx: int,
        cdy: int,
        padding: int,
        target_res: int
    ) -> Float[Array, "1 height width"]:
        """Shift a single image by the given amount.

        Args:
            img: Single image (1, H, W)
            cdx: Shift in x direction
            cdy: Shift in y direction
            padding: Padding size
            target_res: Target resolution after shift

        Returns:
            Shifted image
        """
        # Pad the image
        padded = jnp.pad(img, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

        # Extract shifted region using dynamic_slice
        # Note: cdx and cdy can be negative, so we add padding to make indices positive
        start_y = cdx + padding
        start_x = cdy + padding

        return jax.lax.dynamic_slice(
            padded,
            (0, start_y, start_x),
            (1, target_res, target_res)
        )

    def __call__(
        self,
        ons1: Float[Array, "batch 1 height width"],
        ons2: Float[Array, "batch 1 height width"],
        P_cell_position,
        true_dxy: Float[Array, "batch 2"] = None
    ) -> Float[Array, "batch 1 2"]:
        """Estimate global movement between two ONS frames.

        Args:
            ons1: First optic nerve signal
            ons2: Second optic nerve signal
            P_cell_position: Cell position module for unwarping
            true_dxy: Ground truth movement (unused, for compatibility)

        Returns:
            Predicted movement (dx, dy) normalized to [-1, 1]
        """
        # Stop gradients (non-differentiable optimization)
        ons1 = jax.lax.stop_gradient(ons1)
        ons2 = jax.lax.stop_gradient(ons2)

        # Gaussian blur
        blurred_ons1 = self.gaussian_blur(ons1)
        blurred_ons2 = self.gaussian_blur(ons2)

        # Unwarp the blurred ONS
        xy_full = P_cell_position.get_XY_default_locations()

        # Use simulation_size as required resolution
        # Note: compute_required_image_resolution can't be used inside JIT because
        # it requires concrete int conversion. At initialization, RealNVP is near-identity
        # so simulation_size is a good approximation.
        required_image_resolution = self.simulation_size

        # Generate unwarping grid
        grid = self.generate_grid_fixed(
            xy_full[0, :, 0, 0],
            xy_full[0, :, -1, 0],
            xy_full[0, :, 0, -1],
            xy_full[0, :, -1, -1],
            required_image_resolution
        )

        # Get UV locations for grid
        uvs = P_cell_position.get_UV_locations(
            jnp.transpose(grid, (2, 0, 1))[None, ...]
        )
        uvs = jnp.tile(uvs, (len(blurred_ons1), 1, 1, 1))

        # Grid sample to unwarp (use map_coordinates)
        from jax.scipy.ndimage import map_coordinates

        def grid_sample(img, grid_uv):
            """Sample image using UV grid."""
            # Convert from [-1, 1] to pixel coordinates
            H, W = img.shape[2], img.shape[3]
            grid_uv_permuted = jnp.transpose(grid_uv, (0, 2, 3, 1))  # (B, H, W, 2)
            grid_pixel = (grid_uv_permuted + 1) * jnp.array([[[[(W - 1) / 2, (H - 1) / 2]]]])

            # Sample for each batch item
            def sample_batch_item(img_single, grid_single):
                coords = jnp.stack([grid_single[:, :, 1], grid_single[:, :, 0]], axis=0)
                return map_coordinates(img_single[0], coords, order=1, mode='constant', cval=0)[None, ...]

            return jax.vmap(sample_batch_item)(img, grid_pixel)

        full_blurred_ons1 = grid_sample(blurred_ons1, uvs)
        full_blurred_ons2 = grid_sample(blurred_ons2, uvs)

        # Create mask
        mask = jnp.ones_like(blurred_ons1)
        full_mask = grid_sample(mask, uvs)
        full_mask = (full_mask > 0.5).astype(jnp.float32)  # Nearest-neighbor-like

        # Generate image pyramid
        levels = int(np.log2(required_image_resolution)) - 2
        pyramid_blurred_ons1 = [full_blurred_ons1]
        pyramid_blurred_ons2 = [full_blurred_ons2]
        pyramid_mask = [full_mask]

        for i in range(levels):
            # Average pooling
            pyramid_blurred_ons1.append(
                jax.lax.reduce_window(
                    pyramid_blurred_ons1[-1],
                    0.0,
                    jax.lax.add,
                    (1, 1, 2, 2),
                    (1, 1, 2, 2),
                    'VALID'
                ) / 4.0
            )
            pyramid_blurred_ons2.append(
                jax.lax.reduce_window(
                    pyramid_blurred_ons2[-1],
                    0.0,
                    jax.lax.add,
                    (1, 1, 2, 2),
                    (1, 1, 2, 2),
                    'VALID'
                ) / 4.0
            )
            # Max pooling for mask
            pyramid_mask.append(
                jax.lax.reduce_window(
                    pyramid_mask[-1],
                    -jnp.inf,
                    jax.lax.max,
                    (1, 1, 2, 2),
                    (1, 1, 2, 2),
                    'VALID'
                )
            )

        # Reverse pyramids (coarse to fine)
        pyramid_blurred_ons1 = pyramid_blurred_ons1[::-1]
        pyramid_blurred_ons2 = pyramid_blurred_ons2[::-1]
        pyramid_mask = pyramid_mask[::-1]

        # Pyramid matching (coarse to fine)
        batch_size = len(full_blurred_ons1)
        aggregate_shifts = jnp.zeros((batch_size, 2))

        for i in range(levels + 1):
            current_blurred_ons1 = pyramid_blurred_ons1[i]
            current_blurred_ons2 = pyramid_blurred_ons2[i]
            current_mask = pyramid_mask[i]
            current_res = current_blurred_ons1.shape[2]

            # Pad for shifting
            padded_current_blurred_ons1 = jnp.pad(
                current_blurred_ons1,
                ((0, 0), (0, 0), (2, 2), (2, 2)),
                mode='constant'
            )
            padded_current_mask = jnp.pad(
                current_mask,
                ((0, 0), (0, 0), (2, 2), (2, 2)),
                mode='constant'
            )

            # Generate all shifted versions
            shifted_ons1_list = []
            shifted_mask_list = []
            for dx, dy in self.shifts:
                shifted_ons = jax.lax.dynamic_slice(
                    padded_current_blurred_ons1,
                    (0, 0, dx + 2, dy + 2),
                    (batch_size, 1, current_res, current_res)
                )
                shifted_mask = jax.lax.dynamic_slice(
                    padded_current_mask,
                    (0, 0, dx + 2, dy + 2),
                    (batch_size, 1, current_res, current_res)
                )
                shifted_ons1_list.append(shifted_ons)
                shifted_mask_list.append(shifted_mask)

            # Compute NCC scores for all shifts
            ncc_scores = jnp.ones((batch_size, len(self.shifts))) * 1000

            for j, (shifted_ons1, shifted_mask) in enumerate(zip(shifted_ons1_list, shifted_mask_list)):
                combined_mask = shifted_mask * current_mask
                error = jnp.sum(
                    jnp.sqrt((shifted_ons1 - current_blurred_ons2) ** 2) * combined_mask,
                    axis=(2, 3)
                ) / (jnp.sum(combined_mask, axis=(2, 3)) + 1e-6)
                ncc_scores = ncc_scores.at[:, j].set(error.squeeze())

            # Find best shift
            max_indices = jnp.argmin(ncc_scores, axis=1)
            dx = max_indices % len(self.shift_range) - 2
            dy = max_indices // len(self.shift_range) - 2
            aggregate_shifts = aggregate_shifts + jnp.stack([dx, dy], axis=-1) * (2 ** (levels - i))

            # Shift upper pyramid levels based on found shift
            # For each upper level j, shift by (dy * 2^(j-i), dx * 2^(j-i))
            for j in range(i + 1, levels + 1):
                scale = 2 ** (j - i)
                padding = scale * 2  # P = 2^(j-i) * 2

                # Scaled shifts for this level (per batch element)
                # Note: PyTorch code swaps dx/dy here: cdx = dy, cdy = dx
                cdx_batch = dy * scale  # (batch,)
                cdy_batch = dx * scale  # (batch,)

                target_res = pyramid_blurred_ons1[j].shape[2]

                # Shift ons1 pyramid - vmap over batch
                def shift_single_ons(args):
                    img, cdx_val, cdy_val = args
                    return self._shift_single_image(img, cdx_val, cdy_val, padding, target_res)

                pyramid_blurred_ons1[j] = jax.vmap(shift_single_ons)(
                    (pyramid_blurred_ons1[j], cdx_batch, cdy_batch)
                )

                # Shift mask pyramid
                pyramid_mask[j] = jax.vmap(shift_single_ons)(
                    (pyramid_mask[j], cdx_batch, cdy_batch)
                )

        # Normalize predicted displacement
        pred_dxy = aggregate_shifts.reshape([-1, 2]) / (required_image_resolution / 2)
        pred_dxy = pred_dxy[:, None, :]  # Add dimension: (B, 1, 2)

        return pred_dxy
