"""Generate full ARAD_1K training dataset (preprocesses only full_LMS, crops on-demand).

This script triggers the NTIRE dataset preprocessing which:
1. Validates ARAD1K .mat files against JSON manifests
2. Spectrally interpolates 31 bands -> 301 bands (band-by-band linear)
3. Converts hyperspectral -> LMS with white balance
4. Saves ONLY full_LMS images (no pre-cached crops)

Crops are generated on-demand at training time, saving ~250GB of disk space
compared to pre-caching all crops.
"""
import os
import sys
import json
sys.path.append('.')

from Simulated.Retina import RetinaModel
from Dataset import create_dataset

print("="*70)
print("Generating ARAD_1K Training Dataset (full_LMS only)")
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
print(f"  Virtual training samples: {params['Dataset']['size']:,} (crops generated on-demand)")
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

# Calculate expected data size (only full_LMS, no pre-cached crops). The
# eye-motion policy owns the field size: bounded fixational drift needs only
# max_shift_size of margin regardless of timesteps, so raising T no longer
# inflates the cache (the uniform walk still requests (T-1)*max_shift_size).
dim_image = retina.EyeMotion.required_full_field(retina.required_image_resolution)

# 900 source images, each ~ (482, 512, 4) float32 = ~4MB
# After interpolation: (482, 512, 301) -> LMS (482, 512, 4) = ~4MB each
# Total: ~900 * 4MB = ~3.6GB
num_source_images = 900
bytes_per_image = 482 * 512 * 4 * 4  # H * W * 4 channels * float32
expected_gb = num_source_images * bytes_per_image / 1e9

print(f"\nDataset details:")
print(f"  Source images: {num_source_images} ARAD hyperspectral files")
print(f"  Full LMS cache: ~{expected_gb:.1f} GB (no pre-cached crops)")
print(f"  Crops: generated on-demand at training time")
print(f"  Disk savings vs pre-cached crops: ~{params['Dataset']['size'] * dim_image * dim_image * 4 * 4 / 1e9 - expected_gb:.0f} GB")

# Check disk space
import shutil
stat = shutil.disk_usage(params['root_dir'])
free_gb = stat.free / 1e9
print(f"  Free disk space: {free_gb:.1f} GB")

if free_gb < expected_gb * 1.5:
    print(f"\n⚠️  WARNING: Low disk space!")
    print(f"     Need: ~{expected_gb * 1.5:.1f} GB (with margin)")
    print(f"     Have: {free_gb:.1f} GB")
    print("\n  Continuing anyway (data will be cached incrementally)...")
else:
    print(f"\n✓ Sufficient disk space available")

print("\n" + "="*70)
print("Starting preprocessing...")
print("This will take approximately 5-10 minutes")
print("="*70)

# Create dataset (this triggers preprocessing of full_LMS only)
dataset = create_dataset('NTIRE', params, retina)

print("\n" + "="*70)
print("Dataset Generation Complete!")
print("="*70)
print(f"✓ full_LMS cache: {len(dataset.file_list):,} source images")
print(f"✓ Virtual training samples: {len(dataset):,} (crops generated on-demand at train time)")
print(f"✓ Sample shape: {dataset[0].shape}")

# Show final disk usage of full_LMS cache
import glob
data_dir = f'{params["root_dir"]}/Dataset/ARAD_{dim_image}_LMS/full_LMS/data'
if os.path.exists(data_dir):
    npy_files = glob.glob(f'{data_dir}/*.npy')
    total_bytes = sum(os.path.getsize(f) for f in npy_files)
    total_gb = total_bytes / 1e9
    print(f"✓ full_LMS cache: {len(npy_files)} files, {total_gb:.1f} GB")

print("\n🎉 Ready for training!")
print("\nNext steps:")
print("  1. Dataset full_LMS cache is ready to use")
print("  2. Run training script with: uv run python train.py")
print("  3. Crops are generated on-demand during training (fast)")
