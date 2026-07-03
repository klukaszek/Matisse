"""JAX/Equinox implementation of biologically-grounded fixational eye motion.

`DefaultEyeMotion` draws an i.i.d. uniform jump in [-MSS, MSS] at every
timestep. That is not how an eye behaves during fixation: real inter-saccadic
motion is dominated by slow, temporally-correlated *ocular drift* (a persistent
random walk, ~a few arcmin), punctuated by occasional *microsaccades* (fast,
larger jumps that are statistically biased back toward the fixation locus).
See Rucci & Victor (2015), Engbert & Kliegl (2004), Kuang et al. (2012).

This module reproduces that structure so the cortical motion-estimator
(`M_global_movement`) is trained on a realistic, mostly-small displacement
distribution with a heavy microsaccadic tail, rather than uniform noise. The
call signature is identical to `DefaultEyeMotion` so it is a drop-in
replacement selected from config.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class FixationalEyeMotion(eqx.Module):
    """Fixational drift + microsaccade eye-motion generator.

    The trajectory is a continuous 2-D position (in full-field pixels) that is
    rounded to integer crop offsets at each timestep:

      * Ocular drift -- a persistent (momentum) random walk. The per-step
        velocity is ``drift_persistence * v + N(0, drift_std)``, which gives the
        temporal correlation (smoothly varying direction) seen in measured
        drift, unlike memoryless white-noise jumps.
      * Microsaccade -- with probability ``microsaccade_rate`` per step, a jump
        of amplitude ``U(microsaccade_min, microsaccade_max)`` is added. Its
        direction mixes a random unit vector with the unit vector pointing back
        to the fixation centre, weighted by ``microsaccade_center_bias`` to
        reproduce the corrective (centre-seeking) tendency of microsaccades.

    Positions are clamped to the valid crop range ``[0, full - resolution]`` and
    the reported displacement ``true_dxy`` is the *actual* integer pixel shift
    that occurred between consecutive timesteps -- matching the supervision
    target convention of `DefaultEyeMotion`.
    """

    timesteps_per_image: int = eqx.field(static=True)
    max_shift_size: int = eqx.field(static=True)
    drift_std: float = eqx.field(static=True)
    drift_persistence: float = eqx.field(static=True)
    microsaccade_rate: float = eqx.field(static=True)
    microsaccade_min: float = eqx.field(static=True)
    microsaccade_max: float = eqx.field(static=True)
    microsaccade_center_bias: float = eqx.field(static=True)

    def __init__(
        self,
        timesteps_per_image: int = 8,
        max_shift_size: int = 128,
        drift_std: float = 1.5,
        drift_persistence: float = 0.7,
        microsaccade_rate: float = 0.3,
        microsaccade_min: float = 4.0,
        microsaccade_max: float | None = None,
        microsaccade_center_bias: float = 0.5,
    ):
        """Initialize fixational eye motion.

        Args:
            timesteps_per_image: Number of timesteps to sample per image.
            max_shift_size: Half-width of the valid crop window (pixels). Also
                the default microsaccade amplitude ceiling.
            drift_std: Std of the per-step drift velocity increment (pixels).
            drift_persistence: Velocity momentum in [0, 1); larger = slower,
                more correlated drift.
            microsaccade_rate: Per-timestep probability of a microsaccade.
            microsaccade_min: Minimum microsaccade amplitude (pixels).
            microsaccade_max: Maximum microsaccade amplitude (pixels). Defaults
                to ``max_shift_size`` so jumps stay inside the crop window.
            microsaccade_center_bias: Fraction (0..1) of microsaccade direction
                pointed back toward the fixation centre (corrective tendency).
        """
        self.timesteps_per_image = timesteps_per_image
        self.max_shift_size = max_shift_size
        self.drift_std = float(drift_std)
        self.drift_persistence = float(drift_persistence)
        self.microsaccade_rate = float(microsaccade_rate)
        self.microsaccade_min = float(microsaccade_min)
        self.microsaccade_max = (
            float(max_shift_size) if microsaccade_max is None else float(microsaccade_max)
        )
        self.microsaccade_center_bias = float(microsaccade_center_bias)

    def required_full_field(self, resolution: int) -> int:
        """Full-field size the dataset must supply for this policy.

        Drift and microsaccades are clamped to +/- ``max_shift_size`` of the
        fixation centre every step, so the excursion is bounded independent of
        the number of timesteps -- the field never grows with T (unlike the
        uniform random walk). This is what keeps the cached crops small.
        """
        return resolution + 2 * self.max_shift_size

    def __call__(
        self,
        LMS_full_field: Float[Array, "batch channels height width"],
        required_image_resolution: int,
        *,
        key: jax.random.PRNGKey,
    ) -> tuple[
        Float[Array, "batch timesteps channels resolution resolution"],
        Float[Array, "batch timesteps-1 2"],
    ]:
        """Simulate fixational eye movements and extract crops.

        Args:
            LMS_full_field: Full field LMS image (larger than required resolution).
            required_image_resolution: Size of the crop to extract.
            key: JAX random key for the stochastic trajectory.

        Returns:
            Tuple of:
            - batch_LMS_current_FoV: Cropped images at each timestep.
            - batch_true_dxy: True integer displacement (dx, dy) between timesteps.
        """
        batch_size, channels, H, W = LMS_full_field.shape
        MSS = self.max_shift_size
        T = self.timesteps_per_image
        res = required_image_resolution

        # Valid integer crop-offset range and the fixation centre within it.
        max_x = float(W - res)
        max_y = float(H - res)
        center = jnp.array([float(MSS), float(MSS)], dtype=jnp.float32)  # (x, y)

        # Initial crop (centred at the fixation locus).
        initial_crop = jax.lax.dynamic_slice(
            LMS_full_field,
            (0, 0, MSS, MSS),
            (batch_size, channels, res, res),
        )

        init_pos = jnp.broadcast_to(center[None, :], (batch_size, 2))
        init_vel = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        init_int = jnp.broadcast_to(
            jnp.array([MSS, MSS], dtype=jnp.int32)[None, :], (batch_size, 2)
        )

        step_keys = jax.random.split(key, T - 1)

        def trajectory_step(carry, step_key):
            pos, vel, prev_int = carry
            k_drift, k_occ, k_amp, k_dir = jax.random.split(step_key, 4)

            # --- Ocular drift: persistent (correlated) random walk ---
            drift_noise = jax.random.normal(k_drift, (batch_size, 2)) * self.drift_std
            vel = self.drift_persistence * vel + drift_noise
            pos = pos + vel

            # --- Microsaccade: occasional, centre-biased jump ---
            occur = jax.random.bernoulli(
                k_occ, self.microsaccade_rate, (batch_size, 1)
            ).astype(jnp.float32)
            amp = jax.random.uniform(
                k_amp,
                (batch_size, 1),
                minval=self.microsaccade_min,
                maxval=max(self.microsaccade_min, self.microsaccade_max),
            )
            rand_dir = jax.random.normal(k_dir, (batch_size, 2))
            rand_dir = rand_dir / (
                jnp.linalg.norm(rand_dir, axis=-1, keepdims=True) + 1e-8
            )
            to_center = center[None, :] - pos
            to_center = to_center / (
                jnp.linalg.norm(to_center, axis=-1, keepdims=True) + 1e-8
            )
            micro_dir = (
                (1.0 - self.microsaccade_center_bias) * rand_dir
                + self.microsaccade_center_bias * to_center
            )
            micro_dir = micro_dir / (
                jnp.linalg.norm(micro_dir, axis=-1, keepdims=True) + 1e-8
            )
            pos = pos + occur * amp * micro_dir

            # --- Clamp to the valid crop window and discretise ---
            pos_x = jnp.clip(pos[:, 0], 0.0, max_x)
            pos_y = jnp.clip(pos[:, 1], 0.0, max_y)
            pos = jnp.stack([pos_x, pos_y], axis=-1)
            int_x = jnp.round(pos_x).astype(jnp.int32)
            int_y = jnp.round(pos_y).astype(jnp.int32)

            def crop_one(image, crop_x, crop_y):
                return jax.lax.dynamic_slice(
                    image,
                    (0, crop_y, crop_x),
                    (channels, res, res),
                )

            crops = jax.vmap(crop_one)(LMS_full_field, int_x, int_y)
            actual_shift = jnp.stack(
                (int_x - prev_int[:, 0], int_y - prev_int[:, 1]), axis=-1
            ).astype(jnp.float32)
            new_int = jnp.stack([int_x, int_y], axis=-1)
            return (pos, vel, new_int), (crops, actual_shift)

        _, (moved_crops, true_dxy) = jax.lax.scan(
            trajectory_step,
            (init_pos, init_vel, init_int),
            step_keys,
        )

        moved_crops = jnp.swapaxes(moved_crops, 0, 1)
        batch_LMS_current_FoV = jnp.concatenate(
            (initial_crop[:, None], moved_crops),
            axis=1,
        )
        return batch_LMS_current_FoV, jnp.swapaxes(true_dxy, 0, 1)
