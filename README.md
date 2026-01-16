# Matisse (JAX Implementation)

**Matisse** is a biologically plausible model of the retina and visual cortex, now fully implemented in **JAX**, **Equinox**, and **Penzai** for high-performance simulation and interpretability research.

This repository contains the simulation code for the retina (forward model) and the cortex (inverse model/reconstruction), along with training scripts and data pipelines.

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **uv** (Universal Python Package Manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Matisse.git
    cd Matisse
    ```

2.  **Install dependencies with uv:**
    ```bash
    uv sync
    ```
    This will set up a virtual environment and install all required packages defined in `pyproject.toml`.

### Data Setup

The model requires hyperspectral training data (e.g., ARAD_1K). 
The `generate_training_data.py` script helps verify the configuration and setup the dataset cache.

```bash
uv run python generate_training_data.py
```

## 🏋️‍♀️ Training

To train the cortical model:

```bash
uv run python train_jax.py --config_filename Default/LMS
```

**Options:**
- `--config_filename`: Path to the YAML config file in `Experiment/Config/` (default: `Default/LMS`).
- `--checkpoint_dir`: Custom directory to save checkpoints (optional).
- `--resume_from`: Path to a checkpoint to resume from (optional).

### Checkpointing

The training script automatically saves:
- **Orbax Checkpoints**: For robust state saving/resuming (folders `0/`, `100/`, etc.).
- **Equinox files (`.eqx`)**: Lightweight model weights for easy visualization in Penzai (`model_100.eqx`).

## 🧪 Testing

Run the integration tests to verify the pipeline:

```bash
uv run python test_integration.py
```

Run unit tests for individual modules:

```bash
uv run python test_jax_modules.py
```

## 📂 Project Structure

- **`Simulated/`**: JAX implementation of the biological models.
    - **`Retina/`**: Forward model (Eye Motion, Spatial/Spectral Sampling, Lateral Inhibition).
    - **`Cortex/`**: Inverse model (Demosaicing, Cell Position, etc.).
- **`Dataset/`**: Data loading pipelines using `grain`.
- **`Assets/`**: Shared static data files (cone locations, spectral data).
- **`Experiment/`**: Configuration files and output directory for learned weights.
- **`train_jax.py`**: Main training script.

## 🛠 Tech Stack

- **JAX**: High-performance numerical computing.
- **Equinox**: Neural network library for JAX.
- **Optax**: Optimization library.
- **Orbax**: Checkpointing.
- **Grain**: Efficient data loading.
- **Penzai**: Visualization and interpretability.
- **uv**: Dependency management.