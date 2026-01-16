"""JAX/Equinox implementation of spectral sampling module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
import numpy as np

from .helper.generate_cone_fundamentals import generate_cone_fundamentals_from_peak_frequencies
from .helper.generate_cone_mosaic import generate_default_cone_mosaic


class DefaultSpectralSampling(eqx.Module):
    """Spectral sampling with cone mosaic and Poisson-like noise.

    Multiplies LMS input by cone mosaic to get photoreceptor activation,
    then adds noise proportional to sqrt(activation) (Poisson-like).
    """
    cone_mosaic: Float[Array, "1 1 4 height width"]
    cone_fundamentals: Float[Array, "301 4"]
    cone_cell_noise_std: float = eqx.field(static=True)
    cone_types: tuple = eqx.field(static=True)

    def __init__(
        self,
        simulation_size: int = 256,
        cone_types_str: str = 'LMS',
        simulating_tetra: bool = False,
        cone_fundamentals_params: dict = None,
        root_dir: str = None
    ):
        """Initialize spectral sampling module.

        Args:
            simulation_size: Spatial dimension of simulation
            cone_types_str: Type of cone configuration ('LMS', 'L', 'M', 'S', etc.)
            simulating_tetra: Whether simulating tetrachromacy
            cone_fundamentals_params: Dict mapping cone types to peak wavelengths
            root_dir: Root directory for loading lens data
        """
        # Generate cone mosaic
        cone_mosaic_np = generate_default_cone_mosaic(simulation_size, cone_types_str)
        # Convert to (1, 1, 4, H, W) format
        cone_mosaic = jnp.array(cone_mosaic_np).transpose(2, 0, 1)[None, None, ...]
        self.cone_mosaic = cone_mosaic

        # Cone cell activation noise
        self.cone_cell_noise_std = 0.01

        # Default cone fundamentals peaks
        if not simulating_tetra:
            cone_fundamental_peaks = [560, 530, 419]
            cone_types_list = ['L', 'M', 'S']
        else:  # tetrachromacy
            if cone_fundamentals_params is None:
                raise ValueError("cone_fundamentals_params required for tetrachromacy")
            cone_fundamental_peaks = list(cone_fundamentals_params.values())
            cone_types_list = list(cone_fundamentals_params.keys())

        self.cone_types = tuple(cone_types_list)

        # Generate cone fundamentals
        cone_fundamentals_np = generate_cone_fundamentals_from_peak_frequencies(
            cone_fundamental_peaks,
            root_dir=root_dir
        )
        self.cone_fundamentals = jnp.array(cone_fundamentals_np)

    def __call__(
        self,
        lms: Float[Array, "batch timesteps 4 height width"],
        *,
        key: jax.random.PRNGKey
    ) -> Float[Array, "batch timesteps 1 height width"]:
        """Apply spectral sampling with noise.

        Args:
            lms: Input LMS image
            key: JAX random key for noise generation

        Returns:
            Photoreceptor activation with noise
        """
        # Element-wise multiplication: (B, T, 4, H, W) * (1, 1, 4, H, W) -> (B, T, 4, H, W)
        # Then sum over spectral dimension (axis=2)
        photoreceptor_activation = jnp.sum(lms * self.cone_mosaic, axis=2, keepdims=True)

        # Add Poisson-like noise
        # Generate random ratio between 0 and 1
        key1, key2 = jax.random.split(key)
        ratio = jax.random.uniform(key1, shape=()) * 1.0  # 0.0 to 1.0

        # Generate Gaussian noise with std proportional to sqrt(activation)
        noise_std = jnp.sqrt(jnp.abs(photoreceptor_activation)) * self.cone_cell_noise_std * ratio + 1e-10
        noise = jax.random.normal(key2, shape=photoreceptor_activation.shape)
        noisy_pa = photoreceptor_activation + noise * noise_std

        return noisy_pa

    def get_cone_fundamentals(self) -> Float[Array, "301 4"]:
        """Get cone fundamentals."""
        return self.cone_fundamentals

    def get_cone_mosaic(self) -> Float[Array, "1 4 height width"]:
        """Get cone mosaic (without batch dimension)."""
        return self.cone_mosaic[0]
