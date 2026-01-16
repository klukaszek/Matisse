"""JAX/Grain implementation of NTIRE/ARAD_1K hyperspectral dataset."""
import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm.auto import tqdm
import grain.python as pygrain
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from dataclasses import dataclass

from DatasetJAX.Abstract import GrainDataset
from DatasetJAX import register_class


@dataclass
class RandomAugmentation(pygrain.MapTransform):
    """Random augmentation transform for LMS images.

    Applies random horizontal flip, vertical flip, and 90-degree rotations.
    These augmentations preserve spectral properties while increasing data diversity.
    """

    def map(self, element: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an image.

        Args:
            element: LMS image of shape (H, W, 4)

        Returns:
            Augmented image of shape (H, W, 4)
        """
        # Generate random augmentation flags
        # Use numpy random for efficiency (this runs in worker threads)
        flip_h = np.random.random() > 0.5
        flip_v = np.random.random() > 0.5
        rot_k = np.random.randint(0, 4)  # 0, 1, 2, or 3 times 90 degrees

        img = element

        # Apply horizontal flip
        if flip_h:
            img = np.flip(img, axis=1)

        # Apply vertical flip
        if flip_v:
            img = np.flip(img, axis=0)

        # Apply rotation (k * 90 degrees)
        if rot_k > 0:
            img = np.rot90(img, k=rot_k, axes=(0, 1))

        # Ensure contiguous array
        return np.ascontiguousarray(img)


@register_class("NTIRE")
class NTIRE(GrainDataset):
    """NTIRE/ARAD_1K hyperspectral dataset for JAX/Grain.

    Loads preprocessed LMS images from disk. Images are converted from
    hyperspectral data and preprocessed with white balance normalization.

    Supports both:
    - ARAD_1K_Mirror (900 training images, generates 50k+ crops)
    - NTIRE2022 (if available)
    """

    def __init__(self, params: dict, retina):
        """Initialize NTIRE dataset.

        Args:
            params: Configuration parameters
            retina: RetinaModel instance for getting required image resolution
        """
        super().__init__(params, retina)

        # Calculate required image dimension
        dim_image = (
            retina.required_image_resolution +
            (params['Experiment']['timesteps_per_image'] - 1) *
            2 * params['RetinaModel']['max_shift_size']
        )

        self.dataset_name = params['Dataset']['dataset_name']
        dataset_type = 'LMS'

        # Handle custom cone fundamentals
        if 'cone_fundamentals' in params['RetinaModel']['retina_spectral_sampling']:
            cone_fundamentals_params = params['RetinaModel']['retina_spectral_sampling']['cone_fundamentals']
        else:
            cone_fundamentals_params = {'L': 560, 'M': 530, 'S': 419}

        # Append cone peaks to dataset type string
        for key in cone_fundamentals_params:
            if key == 'L' and cone_fundamentals_params[key] != 560:
                dataset_type += f'_L{cone_fundamentals_params[key]}'
            elif key == 'M' and cone_fundamentals_params[key] != 530:
                dataset_type += f'_M{cone_fundamentals_params[key]}'
            elif key == 'S' and cone_fundamentals_params[key] != 419:
                dataset_type += f'_S{cone_fundamentals_params[key]}'
            elif key == 'Q':
                dataset_type += f'_Q{cone_fundamentals_params[key]}'

        # Get root directory from params or use default
        if 'root_dir' in params:
            root_dir = params['root_dir']
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Dataset size from params or default (50k for GPU, 10k for CPU/MPS)
        dataset_size = params.get('Dataset', {}).get('size', 50000)

        # Determine which dataset to use (prefer ARAD_1K_Mirror)
        arad_path = f'{root_dir}/Dataset/ARAD_1K_Mirror/Train_spectral'
        ntire_path = f'{root_dir}/Dataset/NTIRE2022_interpolated/data'

        if os.path.exists(arad_path):
            print(f'Using ARAD_1K dataset from {arad_path}')
            self.source_dataset = 'ARAD_1K'
        elif os.path.exists(ntire_path):
            print(f'Using NTIRE2022 dataset from {ntire_path}')
            self.source_dataset = 'NTIRE2022'
        else:
            raise RuntimeError(
                f"No hyperspectral data found. Please place data in either:\n"
                f"  - {arad_path}\n"
                f"  - {ntire_path}"
            )

        # Check if data needs preprocessing (check if directory exists and has files)
        data_dir = f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data'
        existing_files = []
        if os.path.exists(data_dir):
            existing_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

        if len(existing_files) == 0:
            print(f'=== Preprocessing {self.source_dataset} hyperspectral data... ===')
            print(f'    Generating {dataset_size} crops of size {dim_image}x{dim_image}')
            print(f'    This will create ~{dataset_size * dim_image * dim_image * 4 * 4 / 1e9:.1f}GB of data')

            if self.source_dataset == 'ARAD_1K':
                preprocess_ARAD_hyperspectral_data(
                    dim_image, dataset_type, retina.CST, root_dir, dataset_size
                )
            else:
                # First interpolate if needed
                interpolated_check = f'{root_dir}/Dataset/NTIRE2022_interpolated/data/0899.pt'
                if not os.path.exists(interpolated_check):
                    print('=== Spectrally interpolating NTIRE hyperspectral data... ===')
                    interpolate_NTIRE_hyperspectral_data(root_dir)
                    print('=== Done! ===')

                preprocess_NTIRE_hyperspectral_data(
                    dim_image, dataset_type, retina.CST, root_dir, dataset_size
                )
            print('=== Preprocessing complete! ===')

        # Store dataset path
        self.data_dir = f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data'

        # Cache file list (use all existing files, like PyTorch DatasetFolder)
        self.file_list = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.pt')
        ])
        self.dataset_size = len(self.file_list)

        if len(self.file_list) == 0:
            raise RuntimeError(f"No data files found in {self.data_dir}")

        print(f'Dataset ready: {len(self.file_list)} samples')

    def __getitem__(self, index: int) -> jnp.ndarray:
        """Get a single LMS image.

        Args:
            index: Index of the image to retrieve

        Returns:
            LMS image as JAX array of shape (H, W, 4)
        """
        index = index % len(self.file_list)
        file_path = self.file_list[index]

        # Load using pickle (PyTorch tensors saved with torch.save)
        with open(file_path, 'rb') as f:
            import torch
            data = torch.load(f, map_location='cpu', weights_only=True)

        # Convert to JAX array
        data_np = data.numpy()
        return jnp.array(data_np)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_list)


def create_dataloader(
    dataset: GrainDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    seed: int = 0,
    augment: bool = True
) -> pygrain.DataLoader:
    """Create a Grain DataLoader from a dataset.

    Args:
        dataset: GrainDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads
        seed: Random seed for shuffling
        augment: Whether to apply random augmentation (flips/rotations)

    Returns:
        Grain DataLoader instance
    """
    # Create sampler
    if shuffle:
        sampler = pygrain.IndexSampler(
            num_records=len(dataset),
            shuffle=True,
            seed=seed,
            num_epochs=None,  # Infinite epochs
            shard_options=pygrain.NoSharding()
        )
    else:
        sampler = pygrain.IndexSampler(
            num_records=len(dataset),
            shuffle=False,
            num_epochs=None,
            shard_options=pygrain.NoSharding()
        )

    # Build operations list
    operations = []

    # Add augmentation if enabled (applied per-sample before batching)
    if augment:
        operations.append(RandomAugmentation())

    # Add batching
    operations.append(pygrain.Batch(batch_size=batch_size, drop_remainder=True))

    # Create loader
    loader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        worker_count=num_workers,
        worker_buffer_size=2,
        read_options=pygrain.ReadOptions(
            num_threads=1,
            prefetch_buffer_size=batch_size * 2
        ),
        operations=operations
    )

    return loader


def interpolate_NTIRE_hyperspectral_data(root_dir: str):
    """Interpolate NTIRE hyperspectral data to increase spectral resolution.

    Args:
        root_dir: Root directory containing the dataset
    """
    import h5py
    import torch

    os.makedirs(f'{root_dir}/Dataset/NTIRE2022_interpolated/data', exist_ok=True)
    data_folder = f'{root_dir}/Dataset/NTIRE2022_original/data'

    def interpolate(index):
        if os.path.exists(f'{data_folder}/ARAD_1K_{index:04d}.mat'):
            if not os.path.exists(f'{root_dir}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt'):
                # Load mat file
                mat_contents = h5py.File(f'{data_folder}/ARAD_1K_{index:04d}.mat', 'r')

                bands = np.asarray(mat_contents['bands'])[:, 0]
                cube = np.asarray(mat_contents['cube'])

                new_cube = []
                for i in range(len(bands) - 1):
                    for j in range(10):
                        image1 = cube[i]
                        image2 = cube[i + 1]
                        new_image = image1 * (1 - j / 10) + image2 * (j / 10)
                        new_cube.append(new_image)

                new_cube.append(cube[-1])
                new_cube = np.asarray(new_cube)
                new_cube = torch.FloatTensor(new_cube)
                new_cube = new_cube.permute(2, 1, 0)

                torch.save(new_cube, f'{root_dir}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(interpolate, i) for i in range(1, 1001)]
    print('Done!')


def preprocess_ARAD_hyperspectral_data(
    dim_image: int,
    dataset_type: str,
    CST,
    root_dir: str,
    dataset_size: int
):
    """Preprocess ARAD_1K hyperspectral data to LMS format.

    Generates multiple random crops from each hyperspectral image to create
    a large training dataset (e.g., 50,000 crops from 900 images).

    Args:
        dim_image: Required image dimension for crops
        dataset_type: Type string (e.g., 'LMS', 'LMS_L560_M530')
        CST: ColorSpaceTransform instance
        root_dir: Root directory
        dataset_size: Number of crops to generate (e.g., 50000)
    """
    import torch
    import torch.nn.functional as F
    import h5py

    # Get white point (convert to numpy)
    defined_white_point = np.array(CST.white_point)
    cone_fundamentals_np = np.array(CST.cone_fundamentals)

    os.makedirs(f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data/', exist_ok=True)
    os.makedirs(f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    # Load all ARAD files
    arad_dir = f'{root_dir}/Dataset/ARAD_1K_Mirror/Train_spectral'
    arad_files = sorted([
        f for f in os.listdir(arad_dir)
        if f.endswith('.mat')
    ])

    print(f'Found {len(arad_files)} ARAD hyperspectral images')
    print(f'Will generate {dataset_size} crops ({dataset_size // len(arad_files)} per image)')

    # Step 1: Convert hyperspectral to LMS
    def hyperspectral_cube_to_lms(index):
        """Convert a single ARAD hyperspectral cube to LMS."""
        output_path = f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{index}.pt'
        if os.path.exists(output_path):
            return  # Already processed

        file_path = os.path.join(arad_dir, arad_files[index])

        try:
            # Load .mat file
            mat_contents = h5py.File(file_path, 'r')

            # ARAD format: cube is (H, W, bands)
            cube = np.array(mat_contents['cube'])  # Shape might be (bands, H, W) or (H, W, bands)

            # Check shape and transpose if needed
            if cube.shape[0] == 31:  # bands first
                cube = np.transpose(cube, (1, 2, 0))  # (31, H, W) -> (H, W, 31)

            # Interpolate from 31 bands to 301 bands (10nm resolution)
            # ARAD has 31 bands from 400-700nm (10nm steps already)
            # For compatibility with cone fundamentals (400-700nm, 1nm steps = 301 points)
            # We need to interpolate
            H, W, bands = cube.shape

            if bands == 31:
                # Interpolate to 301 bands
                cube_torch = torch.from_numpy(cube).float()
                # Reshape for interpolation: (H*W, 31) -> (H*W, 1, 31) -> (H*W, 1, 301)
                cube_flat = cube_torch.reshape(-1, 31)
                cube_interp = F.interpolate(
                    cube_flat.unsqueeze(1),
                    size=301,
                    mode='linear',
                    align_corners=True
                )
                cube = cube_interp.squeeze(1).reshape(H, W, 301).numpy()

            # Convert to LMS: (H, W, 301) @ (301, 4) = (H, W, 4)
            lms_np = np.matmul(cube, cone_fundamentals_np)

            # Upsample if needed
            if H < dim_image or W < dim_image:
                if H < W:
                    multiplier = dim_image / H
                else:
                    multiplier = dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

                lms_torch = torch.from_numpy(lms_np).permute(2, 0, 1).unsqueeze(0)
                lms_torch = F.interpolate(
                    lms_torch, size=(nH, nW), mode='bilinear', align_corners=False
                )
                lms_np = lms_torch.squeeze().permute(1, 2, 0).numpy()

            # White world white balance
            current_white_point = lms_np.reshape(-1, lms_np.shape[-1]).max(0) + 1e-10
            lms_np = lms_np / current_white_point[None, None, :]
            lms_np *= defined_white_point

            # Save
            lms_save = torch.from_numpy(lms_np.astype(np.float32))
            torch.save(lms_save, output_path)

        except Exception as e:
            print(f"\n⚠️  Skipping corrupted file {arad_files[index]}: {e}")
            # Create a dummy file to mark it as processed (will skip in cropping)
            torch.save(torch.zeros(1, 1, 4), output_path)

    print('\n[1/2] Converting hyperspectral data to LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(hyperspectral_cube_to_lms, range(len(arad_files))),
            total=len(arad_files),
            desc='Converting to LMS'
        ))
    print('Done!')

    # Step 2: Generate random crops
    def crop_image(crop_index):
        """Generate a single random crop."""
        # Determine which source image to use
        source_index = crop_index % len(arad_files)

        # Load full LMS image
        lms = torch.load(
            f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{source_index}.pt',
            weights_only=True
        )

        # Skip if this is a dummy file (from corrupted source)
        if lms.shape[0] == 1 and lms.shape[1] == 1:
            # Try next image
            source_index = (source_index + 1) % len(arad_files)
            lms = torch.load(
                f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{source_index}.pt',
                weights_only=True
            )

        H, W = lms.shape[0], lms.shape[1]

        # Random crop
        if H > dim_image and W > dim_image:
            # Use crop_index as seed for determinism
            rng = np.random.RandomState(crop_index)
            x = rng.randint(0, H - dim_image)
            y = rng.randint(0, W - dim_image)
        elif H == dim_image and W > dim_image:
            rng = np.random.RandomState(crop_index)
            x = 0
            y = rng.randint(0, W - dim_image)
        elif H > dim_image and W == dim_image:
            rng = np.random.RandomState(crop_index)
            x = rng.randint(0, H - dim_image)
            y = 0
        elif H == dim_image and W == dim_image:
            x = 0
            y = 0
        else:
            raise ValueError(f'Image too small: {H}x{W} < {dim_image}x{dim_image}')

        lms_crop = lms[x:x + dim_image, y:y + dim_image]

        # Save
        torch.save(
            lms_crop,
            f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data/{crop_index}.pt'
        )

    print('\n[2/2] Generating random crops...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(crop_image, range(dataset_size)),
            total=dataset_size,
            desc='Cropping images'
        ))
    print('Done!')
    print(f'\nCreated {dataset_size} training samples in:')
    print(f'  {root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data/')


def preprocess_NTIRE_hyperspectral_data(
    dim_image: int,
    dataset_type: str,
    CST,
    root_dir: str,
    dataset_size: int
):
    """Preprocess NTIRE hyperspectral data to LMS format.

    Args:
        dim_image: Required image dimension
        dataset_type: Type string (e.g., 'LMS', 'LMS_L560_M530')
        CST: ColorSpaceTransform instance
        root_dir: Root directory
        dataset_size: Number of samples to generate
    """
    import torch
    import torch.nn.functional as F

    # Get white point (convert to numpy)
    defined_white_point = np.array(CST.white_point)

    os.makedirs(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/', exist_ok=True)
    os.makedirs(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    # Load all interpolated data
    interpolated_dir = f'{root_dir}/Dataset/NTIRE2022_interpolated/data'
    file_list = sorted([f for f in os.listdir(interpolated_dir) if f.endswith('.pt')])

    # Convert hyperspectral to LMS
    def hyperspectral_cube_to_lms(index):
        if not os.path.exists(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.pt'):
            file_path = os.path.join(interpolated_dir, file_list[index])
            cube = torch.load(file_path, weights_only=True, map_location='cpu')

            # Convert to numpy for processing
            cube_np = cube.numpy()
            cone_fundamentals_np = np.array(CST.cone_fundamentals)

            # Matrix multiply: (H, W, 301) @ (301, 4) = (H, W, 4)
            lms_np = np.matmul(cube_np, cone_fundamentals_np)

            H, W, _ = lms_np.shape
            if H < dim_image or W < dim_image:
                # Need to upsample
                if H < W:
                    multiplier = dim_image / H
                else:
                    multiplier = dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

                # Use torch for interpolation
                lms_torch = torch.from_numpy(lms_np).permute(2, 0, 1).unsqueeze(0)
                lms_torch = F.interpolate(
                    lms_torch, size=(nH, nW), mode='bilinear', align_corners=False
                )
                lms_np = lms_torch.squeeze().permute(1, 2, 0).numpy()

            # White world white balance
            current_white_point = lms_np.reshape(-1, lms_np.shape[-1]).max(0) + 1e-10
            lms_np = lms_np / current_white_point[None, None, :]
            lms_np *= defined_white_point

            # Save as PyTorch tensor for compatibility
            lms_save = torch.from_numpy(lms_np)
            torch.save(lms_save, f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.pt')

    print('First, converting hyperspectral data to LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(hyperspectral_cube_to_lms, i) for i in range(len(file_list))]
    print('Done!')

    # Crop images
    def crop_image(index):
        if index % 1000 == 0:
            print(f'{index} / {dataset_size}')
        i = index % len(file_list)
        lms = torch.load(
            f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{i}.pt',
            weights_only=True
        )

        W, H = lms.shape[0], lms.shape[1]
        if W > dim_image:
            if H > dim_image:
                x = np.random.randint(0, W - dim_image)
                y = np.random.randint(0, H - dim_image)
            elif H == dim_image:
                x = np.random.randint(0, W - dim_image)
                y = 0
            else:
                raise ValueError('Hyperspectral image smaller than required resolution')
        elif W == dim_image:
            if H > dim_image:
                x = 0
                y = np.random.randint(0, H - dim_image)
            elif H == dim_image:
                x = 0
                y = 0
            else:
                raise ValueError('Hyperspectral image smaller than required resolution')

        lms = lms[x:x + dim_image, y:y + dim_image]
        torch.save(lms, f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/{index}.pt')

    print('Next, cropping the images...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(crop_image, i) for i in range(dataset_size)]
    print('Done!')
