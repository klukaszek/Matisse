"""JAX/Equinox implementation of cone spectral type module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class DefaultConeSpectralType(eqx.Module):
    """Learns cone identity function with normalization.

    This module learns which cone type (L, M, S) each position in the mosaic
    represents. The learned parameters are normalized to ensure they form
    valid probability distributions.
    """
    raw_cone_identity_function: Float[Array, "1 latent_dim height width"]
    latent_dim: int = eqx.field(static=True)
    simulation_size: int = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int = 8,
        simulation_size: int = 256,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize the cone spectral type module.

        Args:
            latent_dim: Number of latent channels
            simulation_size: Spatial dimension of simulation
            key: JAX random key (unused, but kept for consistency)
        """
        self.latent_dim = latent_dim
        self.simulation_size = simulation_size

        # Initialize cone identity function
        # Set first channel to 1, rest to 0 (similar to PyTorch version)
        cone_identity_function = jnp.zeros((1, latent_dim, simulation_size, simulation_size))
        cone_identity_function = cone_identity_function.at[0, 0, :, :].set(1.0)

        self.raw_cone_identity_function = cone_identity_function

    def get_cone_identity_function(self) -> Float[Array, "1 latent_dim height width"]:
        """Get normalized cone identity function.

        Returns:
            Normalized cone identity function with unit norm along channel dimension.
        """
        C = self.raw_cone_identity_function

        # Match PyTorch: epsilon is added after the square root.
        norm = jnp.sqrt(jnp.sum(C * C, axis=1, keepdims=True)) + 1e-5
        C = C / norm

        return C

    def C_injection(
        self,
        pa: Float[Array, "batch 1 height width"]
    ) -> Float[Array, "batch latent_dim height width"]:
        """Inject cone identity into single-channel input.

        Multiplies the input by the normalized cone identity function to
        distribute the signal across latent channels according to cone types.

        Args:
            pa: Input tensor of shape (batch, 1, height, width)

        Returns:
            C-injected tensor of shape (batch, latent_dim, height, width)
        """
        C = self.get_cone_identity_function()
        C_injected_pa = pa * C
        return C_injected_pa

    def C_sampling(
        self,
        C_injected_pa: Float[Array, "batch latent_dim height width"]
    ) -> Float[Array, "batch 1 height width"]:
        """Sample from cone-injected representation.

        Performs weighted sum across latent channels using the cone identity
        function to reconstruct a single-channel output.

        Args:
            C_injected_pa: Input tensor of shape (batch, latent_dim, height, width)

        Returns:
            Sampled tensor of shape (batch, 1, height, width)
        """
        C = self.get_cone_identity_function()
        pa = jnp.sum(C_injected_pa * C, axis=1, keepdims=True)
        return pa

    def __call__(
        self,
        x: Float[Array, "batch channels height width"],
        operation: str = "injection"
    ) -> Float[Array, "batch output_channels height width"]:
        """Convenience forward method supporting both operations.

        Args:
            x: Input tensor
            operation: Either "injection" or "sampling"

        Returns:
            Result of the specified operation
        """
        if operation == "injection":
            return self.C_injection(x)
        elif operation == "sampling":
            return self.C_sampling(x)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'injection' or 'sampling'.")
