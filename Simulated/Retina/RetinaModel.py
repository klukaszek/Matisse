"""JAX/Equinox implementation of RetinaModel combining all retina modules."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Tuple, Optional, Union

from .EM_Default import DefaultEyeMotion
from .EM_Fixational import FixationalEyeMotion
from .FV_spatial_sampling import DefaultSpatialSampling
from .SS_spectral_sampling import DefaultSpectralSampling
from .LI_Default import DefaultLateralInhibition
from .SC_Default import DefaultSpikeConversion
from .helper.ColorSpaceTransform import ColorSpaceTransform


class RetinaModel(eqx.Module):
    """Complete retina model combining all forward processing components.

    This model simulates the forward processing of the retina from RGB input
    to optic nerve signals, including eye movements, spatial sampling,
    spectral sampling, lateral inhibition, and spike conversion.
    """
    EyeMotion: Union[DefaultEyeMotion, FixationalEyeMotion]
    SpatialSampling: DefaultSpatialSampling
    SpectralSampling: DefaultSpectralSampling
    LateralInhibition: DefaultLateralInhibition
    SpikeConversion: DefaultSpikeConversion
    CST: ColorSpaceTransform
    required_image_resolution: int = eqx.field(static=True)

    def __init__(
        self,
        simulation_size: int = 256,
        timesteps_per_image: int = 8,
        max_shift_size: int = 128,
        cone_types_str: str = 'LMS',
        cone_distribution_type: str = 'Human',
        simulating_tetra: bool = False,
        cone_fundamentals_params: Optional[dict] = None,
        cone_gain_adaptation: str = 'none',
        eye_motion_type: str = 'Default',
        eye_motion_params: Optional[dict] = None,
        root_dir: Optional[str] = None
    ):
        """Initialize retina model.

        Args:
            simulation_size: Dimension of optic nerve signal
            timesteps_per_image: Number of timesteps per image
            max_shift_size: Maximum shift size for eye movements
            cone_types_str: Type of cone configuration
            cone_distribution_type: Type of cone spatial distribution
            simulating_tetra: Whether simulating tetrachromacy
            cone_fundamentals_params: Peak wavelengths for tetrachromacy
            eye_motion_type: Eye-motion generator, 'Default' (uniform random
                walk) or 'Fixational' (ocular drift + microsaccades)
            eye_motion_params: Extra keyword params forwarded to the selected
                eye-motion generator (e.g. drift_std, microsaccade_rate)
            root_dir: Root directory for loading data files
        """
        # Initialize all submodules. The eye-motion generator is selectable:
        # 'Default' draws i.i.d. uniform jumps; 'Fixational' produces
        # literature-aligned ocular drift + microsaccades.
        em_params = dict(eye_motion_params or {})
        if eye_motion_type == 'Fixational':
            self.EyeMotion = FixationalEyeMotion(
                timesteps_per_image=timesteps_per_image,
                max_shift_size=max_shift_size,
                **em_params,
            )
        elif eye_motion_type in ('Default', None):
            self.EyeMotion = DefaultEyeMotion(
                timesteps_per_image=timesteps_per_image,
                max_shift_size=max_shift_size
            )
        else:
            raise ValueError(
                f"Unknown eye_motion_type '{eye_motion_type}' "
                "(use 'Default' or 'Fixational')"
            )

        self.SpatialSampling = DefaultSpatialSampling(
            simulation_size=simulation_size,
            max_shift_size=max_shift_size,
            cone_distribution_type=cone_distribution_type,
            root_dir=root_dir
        )

        self.SpectralSampling = DefaultSpectralSampling(
            simulation_size=simulation_size,
            cone_types_str=cone_types_str,
            simulating_tetra=simulating_tetra,
            cone_fundamentals_params=cone_fundamentals_params,
            root_dir=root_dir
        )

        self.LateralInhibition = DefaultLateralInhibition(
            simulation_size=simulation_size
        )

        self.SpikeConversion = DefaultSpikeConversion()

        # Store required image resolution
        self.required_image_resolution = self.SpatialSampling.required_image_resolution

        # Initialize color space transform
        cone_fundamentals = self.SpectralSampling.get_cone_fundamentals()
        self.CST = ColorSpaceTransform(cone_fundamentals, root_dir=root_dir)
        if cone_gain_adaptation not in {'none', 'white'}:
            raise ValueError("cone_gain_adaptation must be 'none' or 'white'")
        if cone_gain_adaptation == 'white':
            white_response = self.CST.white_point.reshape(-1)
            active = jnp.max(cone_fundamentals, axis=0) > 0
            response_gain = jnp.where(
                active,
                1.0 / jnp.maximum(white_response, 1e-6),
                1.0,
            )
            self.SpectralSampling = eqx.tree_at(
                lambda module: module.cone_response_gain,
                self.SpectralSampling,
                response_gain,
            )

    def __call__(
        self,
        batch_LMS_full_field: Float[Array, "batch 4 height width"],
        *,
        key: jax.random.PRNGKey,
        intermediate_outputs: bool = False
    ) -> Tuple:
        """Forward pass through retina model.

        Args:
            batch_LMS_full_field: Full-field LMS input images
            key: JAX random key for stochastic processes
            intermediate_outputs: Whether to return intermediate outputs

        Returns:
            If intermediate_outputs=False:
                (batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV)
            If intermediate_outputs=True:
                (batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV,
                 batch_pa, batch_bipolar_signals, batch_LMS_current_FoV)
        """
        keys = jax.random.split(key, 4)

        # Eye motion: sample current field of view with eye movements
        batch_LMS_current_FoV, batch_true_dxy = self.EyeMotion(
            batch_LMS_full_field,
            self.required_image_resolution,
            key=keys[0]
        )

        # Normalize true eye motion to range [-1, 1]
        batch_true_dxy = batch_true_dxy.astype(jnp.float32) / (self.required_image_resolution / 2)

        # Spatial sampling: apply foveation effect via MIP-mapping
        batch_warped_LMS_current_FoV = self.SpatialSampling(batch_LMS_current_FoV)

        # Spectral sampling: convert LMS to photoreceptor activation
        batch_pa = self.SpectralSampling(batch_warped_LMS_current_FoV, key=keys[1])

        # Match the reference's single LI call so all batch items and timesteps
        # share one sampled noise ratio.
        batch_size, timesteps = batch_pa.shape[0], batch_pa.shape[1]
        pa_flat = batch_pa.reshape(
            batch_size * timesteps,
            *batch_pa.shape[2:]
        )
        batch_bipolar_signals = self.LateralInhibition(
            pa_flat,
            key=keys[2]
        ).reshape(batch_pa.shape)

        # Spike conversion: convert bipolar signals to spikes (currently identity)
        batch_ons = self.SpikeConversion(batch_bipolar_signals)

        if not intermediate_outputs:
            return batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV
        else:
            return (
                batch_ons,
                batch_true_dxy,
                batch_warped_LMS_current_FoV,
                batch_pa,
                batch_bipolar_signals,
                batch_LMS_current_FoV
            )

    def get_cone_mosaic(self) -> Float[Array, "1 4 height width"]:
        """Get the cone mosaic from spectral sampling module."""
        return self.SpectralSampling.get_cone_mosaic()

    def get_cone_fundamentals(self) -> Float[Array, "301 4"]:
        """Get the cone spectral sensitivities."""
        return self.SpectralSampling.get_cone_fundamentals()
