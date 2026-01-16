# Matisse JAX/Equinox/Penzai Implementation

This directory contains the JAX/Equinox reimplementation of the Matisse retina simulation model, with Penzai integration for interactive model exploration and visualization.

## Overview

The original Matisse model is implemented in PyTorch. This JAX version provides:

- **Equinox**: PyTorch-like API for building neural networks in JAX
- **Penzai**: Interactive model visualization and exploration
- **JAX**: Automatic differentiation, JIT compilation, and functional programming
- **Orbax**: Checkpointing and model serialization

## Directory Structure

```
Simulated/
├── NeuralScope/              # Auxiliary neural networks
│   ├── NS_cone_mosaic.py     # Linear projection for cone mosaic
│   └── NS_internal_percept.py # Linear projection for internal percept
├── Cortex/                   # Learnable inverse model components
│   ├── C_cone_spectral_type/ # Cone identity learning
│   ├── D_demosaicing/        # UNet demosaicing network
│   ├── W_lateral_inhibition_weights/  # FFT-based learnable convolution
│   ├── P_cell_position/      # RealNVP normalizing flow for cone positions
│   └── M_global_movement/    # Eye movement estimation (TODO)
└── Retina/                   # Fixed forward model components
    └── LI_Default.py         # DoG lateral inhibition (non-learnable)
```

## Modules Converted

### ✅ Completed

1. **NeuralScope Modules** (`NS_cone_mosaic`, `NS_internal_percept`)
   - Simple linear projections
   - Zero-initialized for auxiliary loss

2. **Cone Spectral Type** (`DefaultConeSpectralType`)
   - Learns cone identity function
   - Normalized parameter learning
   - C_injection and C_sampling operations

3. **UNet Demosaicing** (`DefaultDemosaicing`, `UNet`)
   - Full UNet with skip connections
   - 4 levels of encoding/decoding (16→32→64→128)
   - Reflection padding for convolutions
   - ~890K parameters

4. **FFT-based Modules**
   - `DefaultLateralInhibitionWeights`: Learnable kernel in frequency domain
   - `DefaultLateralInhibition`: Fixed DoG (Difference of Gaussians) filter
   - Complex-valued parameters
   - Efficient convolution/deconvolution via FFT

5. **RealNVP Normalizing Flow** (`DefaultCellPosition`)
   - Custom JAX implementation (replaces `normflows` dependency)
   - 8 layers of affine coupling + permutation
   - Invertible coordinate transformation
   - Grid warping with bilinear interpolation

### ⏳ TODO

6. **Global Movement** (`M_Default`)
   - Multi-scale pyramid matching
   - Eye movement estimation
   - Non-differentiable optimization

7. **Spatial Sampling** (`FV_Default`)
   - MIP-mapping for foveation
   - Multi-level interpolation

8. **Spectral Sampling** (`SS_Default`)
   - Cone mosaic multiplication
   - Poisson-like noise

9. **Training Loop**
   - Port to JAX/Optax
   - Multiple optimizers for different param groups
   - Masked losses

## Key Differences from PyTorch

Matisse JAX is a ground-up reimplementation focused on functional purity and performance.

### 1. Random Number Generation
JAX uses explicit PRNG state management, unlike PyTorch's global state.
```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=x.shape)
```

### 2. Module Definition
We use **Equinox**, which treats modules as PyTrees. This makes them compatible with all JAX transformations (`jit`, `grad`, `vmap`).
```python
class MyModule(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, *, key):
        self.linear = eqx.nn.Linear(10, 10, key=key)
    def __call__(self, x):
        return self.linear(x)
```

### 3. Complex Numbers
JAX has native complex number support, eliminating the need for `view_as_complex`.
```python
complex_param = real + 1j * imag
```

### 4. Data Pipeline
The pipeline is fully JAX-focused using **Google Grain** and **NumPy** (`.npy`) for storage, ensuring zero dependency on PyTorch for training.


### 5. Grid Sampling

**PyTorch:**
```python
output = F.grid_sample(
    input, grid,
    align_corners=True,
    mode='bilinear',
    padding_mode='zeros'
)
```

**JAX:**
```python
from jax.scipy.ndimage import map_coordinates
# Convert grid from [-1, 1] to pixel coordinates
output = map_coordinates(
    input, coords,
    order=1,  # bilinear
    mode='constant', cval=0
)
```

## Usage Examples

### Basic Model Instantiation

```python
import jax
import jax.numpy as jnp
from Simulated.Cortex.D_demosaicing import UNet

# Initialize random key
key = jax.random.PRNGKey(42)

# Create model
model = UNet(dim_latent=8, key=key)

# Create dummy input
batch_size, latent_dim, height, width = 2, 8, 256, 256
x = jax.random.normal(key, (batch_size, latent_dim, height, width))

# Forward pass
output = model(x)
print(output.shape)  # (2, 8, 256, 256)
```

### JIT Compilation

```python
import equinox as eqx

# JIT compile for performance (10-100x speedup)
model_jit = eqx.filter_jit(model)
output = model_jit(x)
```

### Computing Gradients

```python
# Define loss function
def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)

# Compute gradients
grads = jax.grad(loss_fn)(model, x, y)
```

### Penzai Visualization

```python
import treescope

# Setup interactive visualization
treescope.basic_interactive_setup()

# View model structure (in Jupyter/IPython)
model  # This will show an interactive tree view

# View specific components
model.demosaicing.inc  # View input convolution

# View arrays with nice formatting
treescope.render_to_text(model.linear.weight)
```

### Model Surgery with Penzai

```python
from penzai import pz

# Find all Linear layers
def find_layers(model, layer_type):
    """Recursively find all layers of a specific type."""
    layers = []
    # ... (see penzai_demo.py for full implementation)
    return layers

# Modify specific components
# (Advanced usage - see Penzai documentation)
```

## Penzai Features

Penzai provides powerful tools for neural network introspection:

1. **Interactive Visualization**: Explore model structure in Jupyter notebooks
2. **Treescope**: Pretty-print PyTrees with syntax highlighting
3. **Model Surgery**: Select and modify specific components
4. **Activation Inspection**: Visualize intermediate computations
5. **Shape Tracking**: Automatic shape inference and validation

### Example: Visualizing Activations

```python
import penzai.treescope as ts

# Capture intermediate activations
def model_with_activations(model, x):
    activations = {}

    # Manual activation capture (Penzai can automate this)
    x1 = model.inc(x)
    activations['inc'] = x1

    x2 = model.down1(x1)
    activations['down1'] = x2

    # ... etc

    return output, activations

output, acts = model_with_activations(model, x)

# Visualize with nice formatting
ts.render_array(acts['inc'])
```

## Installation

Dependencies are managed with `uv`. To install:

```bash
# Install JAX ecosystem
uv add jax jaxlib equinox penzai optax orbax-checkpoint

# For GPU support (optional)
# See JAX installation guide: https://github.com/google/jax#installation
```

## Performance Considerations

### JIT Compilation

Always JIT compile your models for production:

```python
@eqx.filter_jit
def train_step(model, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    # ... optimizer update
    return loss, model, opt_state

# First call will compile (slow)
loss, model, opt_state = train_step(model, opt_state, x, y)

# Subsequent calls are fast (10-100x speedup)
loss, model, opt_state = train_step(model, opt_state, x, y)
```

### Memory Format

JAX doesn't have explicit `channels_last` memory format. Instead:

1. Use standard NCHW format throughout
2. Rely on XLA compiler optimizations
3. For CPU, consider manually transposing if needed

### Complex Numbers

JAX has native complex number support:

```python
# PyTorch
complex_tensor = torch.view_as_complex(
    torch.stack([real, imag], -1)
)

# JAX (simpler!)
complex_tensor = real + 1j * imag
```

## Testing

Run the demo to verify installation:

```bash
python penzai_demo.py
```

This will:
1. Instantiate all converted models
2. Run forward passes
3. Show Penzai visualization examples
4. Demonstrate JIT compilation

## Next Steps

1. **Convert Remaining Modules**: Eye movement, spatial sampling, spectral sampling
2. **Port Training Loop**: Convert to JAX/Optax with multiple optimizers
3. **Add Checkpointing**: Use Orbax for model serialization
4. **Implement Data Pipeline**: Convert PyTorch DataLoader to JAX
5. **Performance Benchmarks**: Compare JAX vs PyTorch speed and memory
6. **Penzai Advanced Features**: Implement automatic activation capture and model surgery

## Resources

- **Equinox**: https://docs.kidger.site/equinox/
- **JAX**: https://jax.readthedocs.io/
- **Penzai**: https://penzai.readthedocs.io/
- **Optax**: https://optax.readthedocs.io/
- **Orbax**: https://github.com/google/orbax

## Known Issues

1. **Grid Sampling**: JAX's `map_coordinates` has slightly different behavior than PyTorch's `grid_sample`. May need fine-tuning for exact equivalence.

2. **Normflows Dependency**: RealNVP was reimplemented from scratch in JAX since `normflows` is PyTorch-only. Behavior should match but hasn't been numerically validated yet.

3. **Random Number Generation**: JAX requires explicit key management. Need to thread keys through training loop carefully.

## Contributing

When adding new modules:

1. Follow the Equinox module pattern with type annotations
2. Use `jaxtyping` for array shape documentation
3. Add docstrings explaining the module's purpose
4. Include example usage in docstring
5. Test with `penzai_demo.py` or create new tests

## License

Same as the original Matisse project.
