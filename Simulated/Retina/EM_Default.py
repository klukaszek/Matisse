"""JAX/Equinox implementation of eye motion simulation."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int


class DefaultEyeMotion(eqx.Module):
    """Eye motion simulation with random walk.

    Simulates eye movements by randomly sampling crops from a larger image
    across multiple timesteps. Each timestep involves a random shift within
    a maximum shift size.
    """
    timesteps_per_image: int = eqx.field(static=True)
    max_shift_size: int = eqx.field(static=True)

    def __init__(self, timesteps_per_image: int = 8, max_shift_size: int = 128):
        """Initialize eye motion module.

        Args:
            timesteps_per_image: Number of timesteps to sample per image
            max_shift_size: Maximum shift in pixels for each step
        """
        self.timesteps_per_image = timesteps_per_image
        self.max_shift_size = max_shift_size

    def required_full_field(self, resolution: int) -> int:
        """Full-field size the dataset must supply for this policy.

        The uniform walk can shift up to ``max_shift_size`` on each of the T-1
        steps, so it may travel ``(T-1) * max_shift_size`` from centre in either
        direction; the field must be large enough that no crop runs off-image.
        """
        return resolution + (self.timesteps_per_image - 1) * 2 * self.max_shift_size

    def __call__(
        self,
        LMS_full_field: Float[Array, "batch channels height width"],
        required_image_resolution: int,
        *,
        key: jax.random.PRNGKey
    ) -> tuple[
        Float[Array, "batch timesteps channels resolution resolution"],
        Float[Array, "batch timesteps-1 2"]
    ]:
        """Simulate eye movements and extract crops.

        Args:
            LMS_full_field: Full field LMS image (larger than required resolution)
            required_image_resolution: Size of the crop to extract
            key: JAX random key for random movements

        Returns:
            Tuple of:
            - batch_LMS_current_FoV: Cropped images at each timestep
            - batch_true_dxy: True displacement (dx, dy) between timesteps
        """
        batch_size, channels, H, W = LMS_full_field.shape
        MSS = self.max_shift_size
        T = self.timesteps_per_image

        # Initial crop (centered)
        initial_crop = jax.lax.dynamic_slice(
            LMS_full_field,
            (0, 0, MSS, MSS),
            (batch_size, channels, required_image_resolution, required_image_resolution)
        )

        # Generate and apply all batch trajectories together. The previous
        # implementation looped over batch items and repeatedly scattered into
        # a full (B,T,C,H,W) output, serializing otherwise independent crops.
        proposed_shifts = jax.random.randint(
            key,
            (T - 1, batch_size, 2),
            -MSS,
            MSS,
        )

        def trajectory_step(position, shift):
            x, y = position
            proposed_x = x + shift[:, 0]
            proposed_y = y + shift[:, 1]
            new_x = jnp.clip(proposed_x, 0, W - required_image_resolution)
            new_y = jnp.clip(proposed_y, 0, H - required_image_resolution)

            def crop_one(image, crop_x, crop_y):
                return jax.lax.dynamic_slice(
                    image,
                    (0, crop_y, crop_x),
                    (channels, required_image_resolution, required_image_resolution),
                )

            crops = jax.vmap(crop_one)(LMS_full_field, new_x, new_y)
            actual_shift = jnp.stack((new_x - x, new_y - y), axis=-1)
            return (new_x, new_y), (crops, actual_shift.astype(jnp.float32))

        initial_position = (
            jnp.full((batch_size,), MSS, dtype=jnp.int32),
            jnp.full((batch_size,), MSS, dtype=jnp.int32),
        )
        _, (moved_crops, true_dxy) = jax.lax.scan(
            trajectory_step,
            initial_position,
            proposed_shifts,
        )

        moved_crops = jnp.swapaxes(moved_crops, 0, 1)
        batch_LMS_current_FoV = jnp.concatenate(
            (initial_crop[:, None], moved_crops),
            axis=1,
        )
        return batch_LMS_current_FoV, jnp.swapaxes(true_dxy, 0, 1)
