# JAX/Grain Data Loading Pipeline

This directory contains the JAX/Grain-based data loading pipeline for the Matisse retina simulation project.

## Overview

The pipeline has been ported from PyTorch's `torch.utils.data.DataLoader` to Google's [Grain](https://github.com/google/grain), a high-performance data loading library designed specifically for JAX training loops.

## Why Grain?

- **JAX-Native**: Designed for JAX training workflows
- **High Performance**: Efficient multi-threaded data loading
- **Deterministic**: Reproducible shuffling with seed control
- **Flexible**: Supports custom data sources and transformations

## Structure

```
Dataset/
├── __init__.py          # Registry and dataset factory
├── Abstract.py          # Base class for all datasets
├── NTIRE.py             # NTIRE hyperspectral dataset
└── test_grain_dataloader.py  # Tests with synthetic data
```

## Quick Start

### 1. Basic Usage

```python
import jax
from Simulated.Retina import RetinaModel
from Dataset import create_dataset
from Dataset.NTIRE import create_dataloader

# Initialize retina model
retina = RetinaModel(simulation_size=256, timesteps_per_image=8)

# Configure dataset
params = {
    'Experiment': {'timesteps_per_image': 8, 'simulating_tetra': False},
    'RetinaModel': {'max_shift_size': 128, 'retina_spectral_sampling': {'cone_types': 'LMS'}},
    'Dataset': {'dataset_name': 'NTIRE', 'batch_size': 16}
}

# Create dataset and dataloader
dataset = create_dataset('NTIRE', params, retina)
loader = create_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    seed=42
)

# Iterate through batches
for batch_LMS_full_field in loader:
    # batch_LMS_full_field has shape (B, H, W, 4)
    # Transpose to (B, C, H, W) for processing
    batch_LMS = jnp.transpose(batch_LMS_full_field, (0, 3, 1, 2))

    # Process through retina...
    key = jax.random.PRNGKey(0)
    batch_ons, batch_dxy, batch_warped_LMS = retina(batch_LMS, key=key)
```

### 2. Integration with Training Loop

```python
import jax
import optax
import equinox as eqx
from Simulated.Cortex import CortexModel
from Simulated.Retina import RetinaModel
from Dataset import create_dataset
from Dataset.NTIRE import create_dataloader

# Initialize models
key = jax.random.PRNGKey(0)
retina = RetinaModel(simulation_size=256, timesteps_per_image=8)
cortex = CortexModel(latent_dim=8, simulation_size=256, key=key)

# Setup data
dataset = create_dataset('NTIRE', params, retina)
loader = create_dataloader(dataset, batch_size=16, shuffle=True)

# Setup optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(cortex, eqx.is_array))

# Training loop
for epoch in range(num_epochs):
    for batch_LMS_full_field in loader:
        # Prepare data
        batch_LMS = jnp.transpose(batch_LMS_full_field, (0, 3, 1, 2))

        # Forward through retina (no gradients)
        key, subkey = jax.random.split(key)
        batch_ons, batch_true_dxy, batch_warped_LMS = retina(
            batch_LMS, key=subkey
        )

        # Extract timesteps
        ons1 = batch_ons[:, 0]
        ons2 = batch_ons[:, 1]
        true_dxy = batch_true_dxy[:, 0]

        # Compute loss and gradients
        def loss_fn(cortex_model):
            main_loss, ns_cm_loss, ns_ip_loss, *_ = cortex_model.main_train(
                ons1, ons2, linsRGB1, true_dxy, cone_mosaic, kernel_size=21
            )
            return main_loss + 0.01 * ns_cm_loss + 0.01 * ns_ip_loss

        loss, grads = jax.value_and_grad(loss_fn)(cortex)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        cortex = eqx.apply_updates(cortex, updates)

        print(f"Loss: {loss:.4f}")
```

## Creating Custom Datasets

To create a custom dataset, inherit from `GrainDataset`:

```python
from Dataset.Abstract import GrainDataset
from Dataset import register_class
import jax.numpy as jnp

@register_class("MyDataset")
class MyDataset(GrainDataset):
    def __init__(self, params: dict, retina):
        super().__init__(params, retina)
        # Initialize your dataset here
        self.data = ...  # Load/prepare data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> jnp.ndarray:
        # Return a single sample as JAX array
        # Shape should be (H, W, 4) for LMS images
        return self.data[index]
```

Then use it like any other dataset:

```python
dataset = create_dataset('MyDataset', params, retina)
loader = create_dataloader(dataset, batch_size=16)
```

## NTIRE Dataset

The NTIRE dataset implementation handles:

1. **Spectral Interpolation**: Increases spectral resolution of hyperspectral data
2. **LMS Conversion**: Converts hyperspectral cubes to LMS using cone fundamentals
3. **White Balance**: Applies white world normalization
4. **Random Cropping**: Generates multiple crops from each image
5. **Caching**: Saves preprocessed data to disk for fast loading

### Dataset Preparation

The dataset is automatically prepared on first use:

1. Place NTIRE 2022 hyperspectral data in `Dataset/NTIRE2022_original/data/`
2. Run training - preprocessing happens automatically
3. Preprocessed data is saved in `Dataset/NTIRE_{dim}_{type}/LMS/data/`

### File Format

Data is stored as PyTorch tensors (`.pt` files) for compatibility:
- Each file contains a single LMS image
- Shape: `(H, W, 4)` where channels are [L, M, S, Q] (Q=0 for trichromatic)
- Format: Float32, range approximately [0, 1]

## Performance Tips

1. **Worker Count**: Set `num_workers` to match CPU cores (8-16 typical)
2. **Batch Size**: Larger batches (16-32) improve GPU utilization
3. **Prefetching**: Grain automatically prefetches 2x batch_size
4. **Determinism**: Use fixed `seed` for reproducible training

## Testing

Run the test suite to verify the dataloader:

```bash
uv run python test_grain_dataloader.py
```

This tests:
- Dataset creation
- Batch loading
- Shape verification
- Integration with RetinaModel

## Differences from PyTorch

| Feature | PyTorch | Grain |
|---------|---------|-------|
| Base class | `torch.utils.data.Dataset` | `pygrain.RandomAccessDataSource` |
| DataLoader | `DataLoader` | `pygrain.DataLoader` |
| Shuffling | Per-epoch | Continuous with seed |
| Batching | Built-in | Via operations |
| Workers | Process-based | Thread-based |

## API Reference

### `create_dataset(dataset_name, params, retina)`

Factory function to create a dataset instance.

**Parameters:**
- `dataset_name` (str): Name of registered dataset (e.g., 'NTIRE')
- `params` (dict): Configuration parameters
- `retina` (RetinaModel): Retina model for getting image dimensions

**Returns:**
- `GrainDataset`: Dataset instance

### `create_dataloader(dataset, batch_size, shuffle=True, num_workers=8, seed=0)`

Create a Grain DataLoader from a dataset.

**Parameters:**
- `dataset` (GrainDataset): Dataset instance
- `batch_size` (int): Number of samples per batch
- `shuffle` (bool): Whether to shuffle data
- `num_workers` (int): Number of worker threads
- `seed` (int): Random seed for shuffling

**Returns:**
- `pygrain.DataLoader`: Configured data loader

## Examples

See `example_grain_dataloader.py` for complete working examples.

## Notes

- Data is returned in `(B, H, W, C)` format (PyTorch uses `(B, C, H, W)`)
- Transpose before passing to models: `jnp.transpose(batch, (0, 3, 1, 2))`
- Grain uses infinite epochs by default (keeps iterating)
- All data loading is deterministic with fixed seed

## Future Enhancements

Potential improvements:
- [ ] Add data augmentation operations
- [ ] Support for other hyperspectral datasets
- [ ] TFRecord format support for even faster loading
- [ ] Distributed training with sharding
- [ ] Online data generation

## References

- [Grain Documentation](https://github.com/google/grain)
- [JAX Data Loading Guide](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html)
