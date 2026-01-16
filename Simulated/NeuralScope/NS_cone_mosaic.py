"""JAX/Equinox implementation of NeuralScope cone mosaic module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class NS_cone_mosaic(eqx.Module):
    """Simple linear projection from latent_dim to output_dim.

    Initialized with zeros for auxiliary interpretability loss.
    """
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int = 8, output_dim: int = 3, *, key: jax.random.PRNGKey):
        """Initialize the neural scope cone mosaic module.

        Args:
            latent_dim: Number of latent channels (default: 8)
            output_dim: Number of output channels (default: 3)
            key: JAX random key for initialization
        """
        # Create linear layer
        self.linear = eqx.nn.Linear(latent_dim, output_dim, key=key)

        # Zero-initialize weights and biases
        self.linear = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            self.linear,
            (jnp.zeros_like(self.linear.weight), jnp.zeros_like(self.linear.bias))
        )

    def __call__(
        self,
        C: Float[Array, "batch latent_dim height width"]
    ) -> Float[Array, "batch output_dim height width"]:
        """Forward pass.

        Args:
            C: Input tensor of shape (batch, latent_dim, height, width)

        Returns:
            Output tensor of shape (batch, output_dim, height, width)
        """
        # Permute from (B, C, H, W) to (B, H, W, C) for linear layer
        C_permuted = jnp.transpose(C, (0, 2, 3, 1))

        # Apply linear transformation: (B, H, W, latent_dim) -> (B, H, W, output_dim)
        output = jax.vmap(jax.vmap(jax.vmap(self.linear)))(C_permuted)

        # Permute back to (B, C, H, W)
        output = jnp.transpose(output, (0, 3, 1, 2))

        return output
