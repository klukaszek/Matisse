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

        # Initialize outputs
        batch_LMS_current_FoV = jnp.zeros(
            (batch_size, T, channels, required_image_resolution, required_image_resolution)
        )
        batch_true_dxy = jnp.zeros((batch_size, T - 1, 2))

        # Initial crop (centered)
        initial_crop = jax.lax.dynamic_slice(
            LMS_full_field,
            (0, 0, MSS, MSS),
            (batch_size, channels, required_image_resolution, required_image_resolution)
        )
        batch_LMS_current_FoV = batch_LMS_current_FoV.at[:, 0].set(initial_crop)

        # Simulate eye movements for each batch item
        def process_single_image(i, carry):
            fov_array, dxy_array, key = carry
            lms_image = LMS_full_field[i]
            key, subkey = jax.random.split(key)

            # Simulate trajectory for this image
            x, y = MSS, MSS
            fovs = [fov_array[i, 0]]  # Start with initial crop
            dxys = []

            for t in range(T - 1):
                key, subkey = jax.random.split(key)
                # Random shift
                dx, dy = jax.random.randint(subkey, (2,), -MSS, MSS)

                # New position with clamping
                new_x = jnp.clip(x + dx, 0, W - required_image_resolution)
                new_y = jnp.clip(y + dy, 0, H - required_image_resolution)

                # Actual displacement after clamping
                dx = new_x - x
                dy = new_y - y

                # Extract crop at new position
                crop = jax.lax.dynamic_slice(
                    lms_image,
                    (0, new_y, new_x),
                    (channels, required_image_resolution, required_image_resolution)
                )

                fovs.append(crop)
                dxys.append(jnp.array([dx, dy], dtype=jnp.float32))

                x, y = new_x, new_y

            # Update arrays
            fov_array = fov_array.at[i, 1:].set(jnp.stack(fovs[1:]))
            dxy_array = dxy_array.at[i].set(jnp.stack(dxys))

            return (fov_array, dxy_array, key)

        # Process all images in batch
        init_carry = (batch_LMS_current_FoV, batch_true_dxy, key)
        final_fov, final_dxy, _ = jax.lax.fori_loop(
            0, batch_size, process_single_image, init_carry
        )

        return final_fov, final_dxy
