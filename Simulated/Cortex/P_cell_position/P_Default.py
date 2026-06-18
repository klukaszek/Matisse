"""JAX/Equinox implementation of RealNVP normalizing flow for cell position.

This module implements an invertible coordinate transformation using RealNVP
normalizing flows. It learns to warp a regular grid to represent the irregular
spatial distribution of cone photoreceptors in the retina.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import List
import numpy as np


class MLP(eqx.Module):
    """Multi-layer perceptron for affine coupling.

    Used in the affine coupling block to predict scale and shift parameters.
    """
    layers: List[eqx.nn.Linear]
    hidden_dims: List[int] = eqx.field(static=True)

    def __init__(self, hidden_dims: List[int], init_zeros: bool = True, *, key: jax.random.PRNGKey):
        """Initialize MLP.

        Args:
            hidden_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            init_zeros: Whether to initialize output layer with zeros
            key: JAX random key for initialization
        """
        self.hidden_dims = hidden_dims
        num_layers = len(hidden_dims) - 1
        keys = jax.random.split(key, num_layers)

        layers = []
        for i in range(num_layers):
            layer = eqx.nn.Linear(
                hidden_dims[i],
                hidden_dims[i + 1],
                key=keys[i]
            )

            # Zero-initialize last layer if requested
            if init_zeros and i == num_layers - 1:
                layer = eqx.tree_at(
                    lambda m: (m.weight, m.bias),
                    layer,
                    (jnp.zeros_like(layer.weight), jnp.zeros_like(layer.bias))
                )

            layers.append(layer)

        self.layers = layers

    def __call__(self, x: Float[Array, "... input_dim"]) -> Float[Array, "... output_dim"]:
        """Forward pass through MLP."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU to all layers except the last
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x


class AffineCouplingBlock(eqx.Module):
    """Affine coupling block for RealNVP.

    Splits input in half, uses first half to predict affine transformation
    parameters for second half.
    """
    param_map: MLP
    dim: int = eqx.field(static=True)

    def __init__(self, param_map: MLP):
        """Initialize affine coupling block.

        Args:
            param_map: MLP that maps from half the dimensions to scale and shift parameters
        """
        self.param_map = param_map
        self.dim = 2  # Always working with 2D coordinates

    def forward(self, x: Float[Array, "... 2"]) -> tuple[Float[Array, "... 2"], Float[Array, "..."]]:
        """Forward pass (regular grid -> warped grid).

        Args:
            x: Input coordinates

        Returns:
            Transformed coordinates and log determinant of Jacobian
        """
        # Split input: first channel stays, second channel gets transformed
        x1, x2 = x[..., 0:1], x[..., 1:2]

        # Predict scale and shift from first channel
        params = self.param_map(x1)
        log_scale, shift = params[..., 0:1], params[..., 1:2]

        # Clamp log_scale for numerical stability
        log_scale = jnp.clip(log_scale, -10, 10)

        # Affine transformation: x2' = x2 * exp(log_scale) + shift
        x2_transformed = x2 * jnp.exp(log_scale) + shift

        # Concatenate back
        z = jnp.concatenate([x1, x2_transformed], axis=-1)

        # Log determinant of Jacobian
        log_det = jnp.sum(log_scale, axis=-1)

        return z, log_det

    def inverse(self, z: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
        """Inverse pass (warped grid -> regular grid).

        Args:
            z: Transformed coordinates

        Returns:
            Original coordinates
        """
        # Split input
        z1, z2 = z[..., 0:1], z[..., 1:2]

        # Predict scale and shift from first channel (same as forward)
        params = self.param_map(z1)
        log_scale, shift = params[..., 0:1], params[..., 1:2]

        # Clamp log_scale for numerical stability
        log_scale = jnp.clip(log_scale, -10, 10)

        # Inverse affine transformation: x2 = (z2 - shift) / exp(log_scale)
        x2 = (z2 - shift) * jnp.exp(-log_scale)

        # Concatenate back
        x = jnp.concatenate([z1, x2], axis=-1)

        return x


class Permute(eqx.Module):
    """Permutation layer for RealNVP.

    Swaps the two coordinates to ensure both get transformed.
    """
    def forward(self, x: Float[Array, "... 2"]) -> tuple[Float[Array, "... 2"], Float[Array, "..."]]:
        """Forward pass: swap coordinates.

        Returns:
            Permuted coordinates and zero log determinant (permutation is volume-preserving)
        """
        # Swap x and y coordinates
        z = jnp.stack([x[..., 1], x[..., 0]], axis=-1)
        log_det = jnp.zeros(x.shape[:-1])
        return z, log_det

    def inverse(self, z: Float[Array, "... 2"]) -> Float[Array, "... 2"]:
        """Inverse pass: swap coordinates (same as forward for swap)."""
        return jnp.stack([z[..., 1], z[..., 0]], axis=-1)


class RealNVP(eqx.Module):
    """RealNVP normalizing flow.

    Stack of affine coupling blocks and permutations for invertible
    coordinate transformations.
    """
    flows: List  # List of flow layers (AffineCouplingBlock or Permute)

    def __init__(
        self,
        latent_dim: int = 16,
        num_layers: int = 8,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize RealNVP flow.

        Args:
            latent_dim: Hidden dimension for MLPs
            num_layers: Number of coupling layers
            key: JAX random key for initialization
        """
        keys = jax.random.split(key, num_layers)

        flows = []
        for i in range(num_layers):
            # Create MLP for this layer: [1, latent_dim, latent_dim, 2]
            param_map = MLP([1, latent_dim, latent_dim, 2], init_zeros=True, key=keys[i])

            # Add affine coupling block
            flows.append(AffineCouplingBlock(param_map))

            # Add permutation
            flows.append(Permute())

        self.flows = flows

    def forward(
        self,
        x: Float[Array, "... 2"]
    ) -> Float[Array, "... 2"]:
        """Forward pass: regular grid -> warped grid.

        Args:
            x: Input coordinates in normalized space [-1, 1]

        Returns:
            Transformed coordinates (warped positions)
        """
        z = x
        for flow in self.flows:
            z, _ = flow.forward(z)
        return z

    def inverse(
        self,
        z: Float[Array, "... 2"]
    ) -> Float[Array, "... 2"]:
        """Inverse pass: warped grid -> regular grid.

        Args:
            z: Transformed coordinates (warped positions)

        Returns:
            Original coordinates in normalized space [-1, 1]
        """
        x = z
        # Apply inverse transformations in reverse order
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x


class DefaultCellPosition(eqx.Module):
    """Cell position module using RealNVP for coordinate transformation.

    Maps between regular grid UV coordinates and warped XY cone positions.
    """
    RealNVP: RealNVP
    regular_grid_uv: Float[Array, "1 2 height width"]
    simulation_size: int = eqx.field(static=True)

    def __init__(
        self,
        simulation_size: int = 256,
        latent_dim: int = 16,
        num_layers: int = 8,
        *,
        key: jax.random.PRNGKey
    ):
        """Initialize cell position module.

        Args:
            simulation_size: Spatial dimension of simulation
            latent_dim: Hidden dimension for RealNVP MLPs
            num_layers: Number of RealNVP layers
            key: JAX random key for initialization
        """
        self.simulation_size = simulation_size

        # Create regular grid in normalized UV space [-1, 1]
        x_ons_dim = jnp.linspace(0, simulation_size - 1, simulation_size)
        y_ons_dim = jnp.linspace(0, simulation_size - 1, simulation_size)
        xx_ons_dim, yy_ons_dim = jnp.meshgrid(x_ons_dim, y_ons_dim, indexing='xy')
        regular_grid = jnp.stack([xx_ons_dim, yy_ons_dim], axis=0)[None, ...]  # (1, 2, H, W)

        # Normalize to [-1, 1]
        self.regular_grid_uv = (regular_grid - (simulation_size - 1) / 2) / ((simulation_size - 1) / 2)

        # Initialize RealNVP
        self.RealNVP = RealNVP(latent_dim=latent_dim, num_layers=num_layers, key=key)

    def get_XY_default_locations(self) -> Float[Array, "1 2 height width"]:
        """Get cone locations in retina XY space.

        Returns:
            Warped XY coordinates of shape (1, 2, H, W)
        """
        # Permute to (..., 2) format for RealNVP.
        # regular_grid_uv is a FIXED base lattice (the torch reference stores it
        # as a plain tensor attribute, not an nn.Parameter, so the optimizer
        # never updates it). In equinox it is an array field => a differentiable
        # leaf, and train.py marks the whole P_cell_position subtree trainable,
        # so without this stop_gradient the base grid drifts under Adam and the
        # learned warp becomes wavy (over-warped percepts).
        grid_flat = jnp.transpose(jax.lax.stop_gradient(self.regular_grid_uv)[0], (1, 2, 0))  # (H, W, 2)

        # Apply forward transformation
        xy_flat = jax.vmap(jax.vmap(self.RealNVP.forward))(grid_flat)  # (H, W, 2)

        # Permute back to (1, 2, H, W)
        xy = jnp.transpose(xy_flat, (2, 0, 1))[None, ...]

        return xy

    def get_UV_locations(
        self,
        xy_locations: Float[Array, "batch 2 height width"]
    ) -> Float[Array, "batch 2 height width"]:
        """Get UV coordinates for given XY locations.

        Args:
            xy_locations: Warped XY coordinates

        Returns:
            Regular grid UV coordinates
        """
        batch_size = xy_locations.shape[0]

        # Permute to (..., 2) format
        xy_flat = jnp.transpose(xy_locations, (0, 2, 3, 1))  # (B, H, W, 2)

        # Apply inverse transformation
        uv_flat = jax.vmap(jax.vmap(jax.vmap(self.RealNVP.inverse)))(xy_flat)  # (B, H, W, 2)

        # Permute back to (B, 2, H, W)
        uv = jnp.transpose(uv_flat, (0, 3, 1, 2))

        return uv

    def efficient_warping(
        self,
        ip1: Float[Array, "batch channels height width"],
        pred_dxy: Float[Array, "batch 2"]
    ) -> tuple[Float[Array, "batch channels height width"], Float[Array, "batch 1 height width"]]:
        """Efficiently warp internal percept based on predicted eye movement.

        Args:
            ip1: Internal percept at time t
            pred_dxy: Predicted eye movement (dx, dy)

        Returns:
            Warped internal percept and mask
        """
        # Get current cone locations
        xy = self.get_XY_default_locations()  # (1, 2, H, W)

        # Add predicted movement
        translated1 = xy + pred_dxy.reshape(-1, 2, 1, 1)  # (B, 2, H, W)

        # Get corresponding UV coordinates
        grid1 = self.get_UV_locations(translated1)  # (B, 2, H, W)

        # Permute for grid_sample: (B, H, W, 2)
        grid1_permuted = jnp.transpose(grid1, (0, 2, 3, 1))

        # Use JAX's map_coordinates for grid sampling
        # Note: JAX uses different convention - coordinates are in pixel space
        # Convert from [-1, 1] to [0, H-1] and [0, W-1]
        H, W = ip1.shape[2], ip1.shape[3]
        grid1_pixel = (grid1_permuted + 1) * jnp.array([[[[(W - 1) / 2, (H - 1) / 2]]]]) + jnp.array([[[[0, 0]]]])

        # Grid sample using bilinear interpolation
        def grid_sample_channel(img_channel, grid):
            """Sample a single channel using bilinear interpolation."""
            from jax.scipy.ndimage import map_coordinates
            # grid is (H, W, 2), need to split into y and x coordinates
            coords = jnp.stack([grid[..., 1], grid[..., 0]], axis=0)  # (2, H, W)
            return map_coordinates(img_channel, coords, order=1, mode='constant', cval=0)

        # Apply to all batch items and channels
        ip2_pred = jax.vmap(
            lambda img, grid: jax.vmap(lambda ch: grid_sample_channel(ch, grid))(img)
        )(ip1, grid1_pixel)

        # Compute mask: samples outside [-1, 1] UV range are invalid
        # grid1 is in UV space (normalized [-1, 1])
        # Mark as 0 where coordinates are outside valid range (matches PyTorch padding_mode='zeros')
        valid_u = (grid1_permuted[..., 0] >= -1) & (grid1_permuted[..., 0] <= 1)
        valid_v = (grid1_permuted[..., 1] >= -1) & (grid1_permuted[..., 1] <= 1)
        mask2 = (valid_u & valid_v).astype(jnp.float32)[:, None, :, :]  # (B, 1, H, W)

        return ip2_pred, mask2

    def __call__(self, operation: str = "get_xy", **kwargs):
        """Convenience forward method supporting multiple operations.

        Args:
            operation: One of "get_xy", "get_uv", "warp"
            **kwargs: Additional arguments for the operation

        Returns:
            Result of the specified operation
        """
        if operation == "get_xy":
            return self.get_XY_default_locations()
        elif operation == "get_uv":
            return self.get_UV_locations(kwargs.get("xy_locations"))
        elif operation == "warp":
            return self.efficient_warping(kwargs.get("ip1"), kwargs.get("pred_dxy"))
        else:
            raise ValueError(f"Unknown operation: {operation}")
