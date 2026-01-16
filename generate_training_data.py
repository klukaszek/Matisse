"""Generate full ARAD_1K training dataset (50k samples)."""
import os
import sys
sys.path.append('.')

from Simulated.Retina import RetinaModel
from Dataset import create_dataset

print("="*70)
print("Generating Full ARAD_1K Training Dataset")
print("="*70)

# Configuration for full dataset (matching original PyTorch config)
params = {
    'Experiment': {
        'timesteps_per_image': 2,  # Original uses 2
        'simulating_tetra': False
    },
    'RetinaModel': {
        'max_shift_size': 15,  # Original uses 15
        'retina_spectral_sampling': {
            'cone_types': 'LMS'
        }
    },
    'Dataset': {
        'dataset_name': 'NTIRE',
        'batch_size': 8,
        'size': 10000  # MPS/CPU uses 10k, not 50k!
    },
    'root_dir': os.path.dirname(os.path.abspath(__file__))
}

print("\nConfiguration:")
print(f"  Dataset size: {params['Dataset']['size']} samples")
print(f"  Timesteps per image: {params['Experiment']['timesteps_per_image']}")
print(f"  Max shift size: {params['RetinaModel']['max_shift_size']}")

# Initialize retina
print("\nInitializing RetinaModel...")
retina = RetinaModel(
    simulation_size=256,
    timesteps_per_image=params['Experiment']['timesteps_per_image'],
    max_shift_size=params['RetinaModel']['max_shift_size'],
    cone_types_str='LMS',
    cone_distribution_type='Human',
    simulating_tetra=False
)
print(f"✓ Required image resolution: {retina.required_image_resolution}")

# Calculate expected data size
dim_image = (
    retina.required_image_resolution +
    (params['Experiment']['timesteps_per_image'] - 1) *
    2 * params['RetinaModel']['max_shift_size']
)
dataset_size = params['Dataset']['size']
expected_gb = dataset_size * dim_image * dim_image * 4 * 4 / 1e9

print(f"\nDataset details:")
print(f"  Image dimension: {dim_image}x{dim_image}")
print(f"  Number of samples: {dataset_size:,}")
print(f"  Estimated size: ~{expected_gb:.1f} GB")
print(f"  Source images: 900 ARAD hyperspectral files")

# Check disk space
import shutil
stat = shutil.disk_usage(params['root_dir'])
free_gb = stat.free / 1e9
print(f"  Free disk space: {free_gb:.1f} GB")

if free_gb < expected_gb * 1.2:
    print(f"\n⚠️  WARNING: Low disk space!")
    print(f"     Need: ~{expected_gb * 1.2:.1f} GB (with margin)")
    print(f"     Have: {free_gb:.1f} GB")
    print("\n  Continuing anyway (data will be cached incrementally)...")
else:
    print(f"\n✓ Sufficient disk space available")

print("\n" + "="*70)
print("Starting preprocessing...")
print("This will take approximately 5-10 minutes")
print("="*70)

# Create dataset (this triggers preprocessing)
dataset = create_dataset('NTIRE', params, retina)

print("\n" + "="*70)
print("Dataset Generation Complete!")
print("="*70)
print(f"✓ Created {len(dataset):,} training samples")
print(f"✓ Sample shape: {dataset[0].shape}")

# Show final disk usage
data_dir = f'{params["root_dir"]}/Dataset/ARAD_{dim_image}_LMS/LMS/data'
if os.path.exists(data_dir):
    import subprocess
    result = subprocess.run(
        ['du', '-sh', data_dir],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        size = result.stdout.split()[0]
        print(f"✓ Disk usage: {size}")

print("\n🎉 Ready for training!")
print("\nNext steps:")
print("  1. Dataset is cached and ready to use")
print("  2. Run training script with: uv run python train_jax.py")
print("  3. Subsequent runs will use cached data (fast)")
