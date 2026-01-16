"""Simulated: JAX/Equinox/Penzai implementation of Matisse retina simulation.

This package provides a JAX reimplementation of the Matisse model with:
- Equinox for PyTorch-like neural network API
- Penzai for interactive model exploration and visualization
- Full compatibility with JAX transformations (jit, grad, vmap, etc.)

Modules:
--------
- NeuralScope: Auxiliary neural networks for interpretability
- Cortex: Learnable inverse model components
- Retina: Fixed forward model components (non-learnable)

Example:
--------
>>> import jax
>>> from Simulated.Cortex.D_demosaicing import UNet
>>>
>>> key = jax.random.PRNGKey(42)
>>> model = UNet(dim_latent=8, key=key)
>>> x = jax.random.normal(key, (2, 8, 256, 256))
>>> output = model(x)
"""

__version__ = "0.1.0"

# Import main modules for convenience
from . import NeuralScope
from . import Cortex
from . import Retina

__all__ = [
    "NeuralScope",
    "Cortex",
    "Retina",
]
