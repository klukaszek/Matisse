"""JAX/Grain implementation of NTIRE/ARAD_1K hyperspectral dataset."""
import os
import io
import zipfile
import json
import numpy as np
import pickle
from tqdm.auto import tqdm
import grain.python as pygrain
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from dataclasses import dataclass
import h5py
import jax
import jax.numpy as jnp

from Dataset.Abstract import GrainDataset
from Dataset import register_class


def _numeric_path_key(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


def _is_valid_cached_image(path: str, dim_image: int) -> bool:
    try:
        image = np.load(path, mmap_mode='r')
        return (
            image.ndim == 3
            and image.shape[0] >= dim_image
            and image.shape[1] >= dim_image
            and image.shape[2] == 4
        )
    except (OSError, ValueError):
        return False


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
    - ARAD_1K_Mirror (900 training images, crops generated on demand)
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
        arad_zip_path = f'{root_dir}/Dataset/ARAD_1K_Mirror/Train_spectral.zip'
        arad_json_train = f'{root_dir}/Dataset/ARAD_1K_Mirror/.arad_good_ids_train.json'
        arad_json_valid = f'{root_dir}/Dataset/ARAD_1K_Mirror/.arad_good_ids_valid.json'
        ntire_path = f'{root_dir}/Dataset/NTIRE2022_interpolated/data'

        # Validate ARAD1K dataset against JSON manifests if available
        expected_train_ids = None
        expected_valid_ids = None
        if os.path.exists(arad_json_train):
            with open(arad_json_train, 'r') as f:
                expected_train_ids = set(json.load(f))
        if os.path.exists(arad_json_valid):
            with open(arad_json_valid, 'r') as f:
                expected_valid_ids = set(json.load(f))

        if os.path.exists(arad_path):
            print(f'Using ARAD_1K dataset from {arad_path}')
            self.source_dataset = 'ARAD_1K'
            found_files = [
                name for name in os.listdir(arad_path)
                if name.endswith('.mat')
            ]
            source_count = len(found_files)
            # Validate against JSON manifest if available
            if expected_train_ids is not None:
                found_ids = {os.path.splitext(f)[0] for f in found_files}
                missing = expected_train_ids - found_ids
                if missing:
                    print(f'  ⚠️  Warning: {len(missing)} expected training files missing from extracted directory')
                else:
                    print(f'  ✓ All {len(expected_train_ids)} expected training files present')
        elif os.path.exists(arad_zip_path):
            print(f'Using ARAD_1K dataset from {arad_zip_path}')
            self.source_dataset = 'ARAD_1K'
            with zipfile.ZipFile(arad_zip_path) as archive:
                found_files = [
                    name for name in archive.namelist()
                    if name.endswith('.mat')
                ]
                source_count = len(found_files)
            # Validate against JSON manifest if available
            if expected_train_ids is not None:
                found_ids = {os.path.splitext(os.path.basename(f))[0] for f in found_files}
                missing = expected_train_ids - found_ids
                if missing:
                    print(f'  ⚠️  Warning: {len(missing)} expected training files missing from zip archive')
                else:
                    print(f'  ✓ All {len(expected_train_ids)} expected training files present in zip')
        elif os.path.exists(ntire_path):
            print(f'Using NTIRE2022 dataset from {ntire_path}')
            self.source_dataset = 'NTIRE2022'
            source_count = dataset_size
        else:
            raise RuntimeError(
                f"No hyperspectral data found. Please place data in either:\n"
                f"  - {arad_path} or {arad_zip_path}\n"
                f"  - {ntire_path}"
            )

        self.dim_image = dim_image
        self.dataset_size = dataset_size

        if self.source_dataset == 'ARAD_1K':
            data_dir = (
                f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}'
                '/full_LMS/data'
            )
        else:
            data_dir = f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data'

        existing_files = []
        invalid_markers = []
        if os.path.exists(data_dir):
            existing_files = [
                f for f in os.listdir(data_dir)
                if f.endswith('.npy')
                and _is_valid_cached_image(os.path.join(data_dir, f), dim_image)
            ]
            invalid_markers = [
                f for f in os.listdir(data_dir) if f.endswith('.invalid')
            ]

        if len(existing_files) + len(invalid_markers) < source_count:
            print(f'=== Preprocessing {self.source_dataset} hyperspectral data... ===')

            if self.source_dataset == 'ARAD_1K':
                print('    Caching 900 full LMS images; crops are generated on demand')
                preprocess_ARAD_hyperspectral_data(
                    dim_image,
                    dataset_type,
                    retina.CST,
                    root_dir,
                    dataset_size,
                    generate_crops=False,
                )
            else:
                print(f'    Generating {dataset_size} crops of size {dim_image}x{dim_image}')
                # First interpolate if needed
                interpolated_check = f'{root_dir}/Dataset/NTIRE2022_interpolated/data/0899.npy'
                if not os.path.exists(interpolated_check):
                    print('=== Spectrally interpolating NTIRE hyperspectral data... ===')
                    interpolate_NTIRE_hyperspectral_data(root_dir)
                    print('=== Done! ===')

                preprocess_NTIRE_hyperspectral_data(
                    dim_image, dataset_type, retina.CST, root_dir, dataset_size
                )
            print('=== Preprocessing complete! ===')

        self.data_dir = data_dir

        self.file_list = sorted(
            (
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.endswith('.npy')
                and _is_valid_cached_image(
                    os.path.join(self.data_dir, f),
                    dim_image,
                )
            ),
            key=_numeric_path_key,
        )
        excluded_count = source_count - len(self.file_list)

        if len(self.file_list) == 0:
            raise RuntimeError(f"No data files found in {self.data_dir}")

        print(
            f'Dataset ready: {self.dataset_size} samples '
            f'from {len(self.file_list)} cached images'
        )
        if excluded_count:
            print(f'Excluded {excluded_count} corrupt source image(s)')

    def __getitem__(self, index) -> np.ndarray:
        """Get a single LMS image.

        Args:
            index: Index of the image to retrieve

        Returns:
            LMS image as numpy array of shape (H, W, 4)
        """
        source_index = index % len(self.file_list)
        # Use memory-mapped loading to avoid loading full image into RAM.
        # Only the cropped region is actually read into memory.
        image = np.load(self.file_list[source_index], mmap_mode='r')

        if self.source_dataset != 'ARAD_1K':
            return image

        height, width = image.shape[:2]
        if height < self.dim_image or width < self.dim_image:
            raise ValueError(
                f'Cached image is too small: {height}x{width} '
                f'< {self.dim_image}x{self.dim_image}'
            )

        # Use the global numpy random state to match the reference's
        # np.random.randint behavior (non-deterministic across runs).
        x = np.random.randint(0, height - self.dim_image) if height > self.dim_image else 0
        y = np.random.randint(0, width - self.dim_image) if width > self.dim_image else 0
        return image[x:x + self.dim_image, y:y + self.dim_image]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.dataset_size


def create_dataloader(
    dataset: GrainDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    seed: int = 0,
    augment: bool = False
) -> pygrain.DataLoader:
    """Create a Grain DataLoader from a dataset.

    Args:
        dataset: GrainDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads
        seed: Random seed for shuffling
        augment: Whether to apply random augmentation (disabled by default to
            match the reference training pipeline)

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
        worker_buffer_size=2,  # Minimal buffering — crops are cheap to generate
        read_options=pygrain.ReadOptions(
            num_threads=1,
            prefetch_buffer_size=batch_size * 2  # Only prefetch 2 batches ahead
        ),
        operations=operations
    )

    return loader


from functools import partial


def _augment_one(img, key):
    """Random h-flip, v-flip, and 0/90/180/270 rotation of a square crop.

    Mirrors RandomAugmentation (and the augmentation the torch reference bakes
    into its precomputed crop set). Requires a square crop so rotation preserves
    shape.
    """
    k_h, k_v, k_r = jax.random.split(key, 3)
    flip_h = jax.random.bernoulli(k_h)
    flip_v = jax.random.bernoulli(k_v)
    rot_k = jax.random.randint(k_r, (), 0, 4)

    img = jnp.where(flip_h, jnp.flip(img, axis=1), img)
    img = jnp.where(flip_v, jnp.flip(img, axis=0), img)
    img = jax.lax.switch(
        rot_k,
        [
            lambda im: im,
            lambda im: jnp.rot90(im, 1, axes=(0, 1)),
            lambda im: jnp.rot90(im, 2, axes=(0, 1)),
            lambda im: jnp.rot90(im, 3, axes=(0, 1)),
        ],
        img,
    )
    return img


@partial(jax.jit, static_argnames=('batch_size', 'crop', 'augment'))
def _sample_crops(images, key, batch_size, crop, augment=True):
    """Gather `batch_size` random crops of size (crop, crop) from a device-
    resident image stack (N, H, W, C). Faithful to NTIRE.__getitem__: integer
    crop offsets, with the offset pinned to 0 along any axis that equals `crop`.

    When `augment` is set, each crop also gets a random flip/rotation — this is
    the runtime equivalent of the augmentation the torch reference precomputes
    into its crop set, and it is what gives the cone-identity parameter enough
    data diversity to actually differentiate (otherwise its gradient is mostly
    per-batch noise and it stays frozen at init).
    """
    N, H, W, C = images.shape
    k_idx, k_x, k_y, k_aug = jax.random.split(key, 4)
    idx = jax.random.randint(k_idx, (batch_size,), 0, N)
    # randint's maxval is exclusive; match np.random.randint(0, dim - crop).
    if H > crop:
        xs = jax.random.randint(k_x, (batch_size,), 0, H - crop)
    else:
        xs = jnp.zeros((batch_size,), jnp.int32)
    if W > crop:
        ys = jax.random.randint(k_y, (batch_size,), 0, W - crop)
    else:
        ys = jnp.zeros((batch_size,), jnp.int32)

    def crop_one(i, x, y):
        # dynamic_slice on the index axis avoids materializing whole images.
        return jax.lax.dynamic_slice(images, (i, x, y, 0), (1, crop, crop, C))[0]

    crops = jax.vmap(crop_one)(idx, xs, ys)
    if augment:
        crops = jax.vmap(_augment_one)(crops, jax.random.split(k_aug, batch_size))
    return crops


class DeviceResidentCropLoader:
    """Infinite crop sampler that keeps the entire (uniformly-shaped) image set
    resident in device memory and generates random crops on-device.

    This removes the CPU crop work *and* the per-step host->device transfer from
    the training hot loop — the whole batch never leaves the accelerator. Only
    usable when every cached image has the same shape (true for ARAD_1K_Mirror,
    ~4.2GB). Augmentation (flip/rotation) is applied on-device by default: the
    torch reference bakes augmentation into its precomputed 50k-crop set, and
    without it the cone-identity parameter never gets enough data diversity to
    differentiate (its gradient stays dominated by per-batch noise).
    """

    def __init__(self, dataset: GrainDataset, batch_size: int, crop_size: int,
                 seed: int = 0, augment: bool = True):
        file_list = dataset.file_list
        first = np.load(file_list[0], mmap_mode='r')
        shape, dtype = first.shape, first.dtype
        if shape[0] < crop_size or shape[1] < crop_size:
            raise ValueError(
                f'Image {shape} smaller than crop {crop_size}; cannot use '
                f'device-resident loader.'
            )

        # Stack into one contiguous host buffer, then move to device once.
        host = np.empty((len(file_list),) + shape, dtype)
        for i, path in enumerate(file_list):
            img = np.load(path, mmap_mode='r')
            if img.shape != shape:
                raise ValueError(
                    f'Ragged image shapes ({img.shape} != {shape}); '
                    f'device-resident loader requires uniform shapes.'
                )
            host[i] = img
        gib = host.nbytes / 1024 ** 3
        print(f'  Loading {len(file_list)} images ({gib:.1f} GiB) into device memory...')
        self.images = jax.device_put(jnp.asarray(host))
        del host

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.augment = augment
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        return self

    def __next__(self):
        self.key, k = jax.random.split(self.key)
        return _sample_crops(
            self.images, k, self.batch_size, self.crop_size, self.augment
        )


def interpolate_NTIRE_hyperspectral_data(root_dir: str):
    """Interpolate NTIRE hyperspectral data to increase spectral resolution.

    Args:
        root_dir: Root directory containing the dataset
    """
    os.makedirs(f'{root_dir}/Dataset/NTIRE2022_interpolated/data', exist_ok=True)
    data_folder = f'{root_dir}/Dataset/NTIRE2022_original/data'

    def interpolate(index):
        if os.path.exists(f'{data_folder}/ARAD_1K_{index:04d}.mat'):
            if not os.path.exists(f'{root_dir}/Dataset/NTIRE2022_interpolated/data/{index:04d}.npy'):
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
                # Transpose to (H, W, C)
                new_cube = np.transpose(new_cube, (2, 1, 0))

                np.save(f'{root_dir}/Dataset/NTIRE2022_interpolated/data/{index:04d}.npy', new_cube)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(interpolate, i) for i in range(1, 1001)]
    print('Done!')


def preprocess_ARAD_hyperspectral_data(
    dim_image: int,
    dataset_type: str,
    CST,
    root_dir: str,
    dataset_size: int,
    generate_crops: bool = True,
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
        generate_crops: Whether to materialize crops instead of creating them
            on demand
    """
    # Get white point (convert to numpy)
    defined_white_point = np.array(CST.white_point)
    cone_fundamentals_np = np.array(CST.cone_fundamentals)

    os.makedirs(f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data/', exist_ok=True)
    os.makedirs(f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    # Load all ARAD files. The official mirror may be left compressed.
    arad_dir = f'{root_dir}/Dataset/ARAD_1K_Mirror/Train_spectral'
    arad_zip_path = f'{root_dir}/Dataset/ARAD_1K_Mirror/Train_spectral.zip'
    if os.path.isdir(arad_dir):
        arad_files = sorted(
            f for f in os.listdir(arad_dir) if f.endswith('.mat')
        )
        source_is_zip = False
    elif os.path.isfile(arad_zip_path):
        with zipfile.ZipFile(arad_zip_path) as archive:
            arad_files = sorted(
                f for f in archive.namelist() if f.endswith('.mat')
            )
        source_is_zip = True
    else:
        raise RuntimeError(
            f'ARAD training data not found at {arad_dir} or {arad_zip_path}'
        )

    print(f'Found {len(arad_files)} ARAD hyperspectral images')
    print(f'Will generate {dataset_size} crops ({dataset_size // len(arad_files)} per image)')

    # Step 1: Convert hyperspectral to LMS
    def hyperspectral_cube_to_lms(index):
        """Convert a single ARAD hyperspectral cube to LMS."""
        output_path = f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{index}.npy'
        invalid_path = f'{output_path}.invalid'
        if os.path.exists(output_path):
            if _is_valid_cached_image(output_path, dim_image):
                return
            os.remove(output_path)
        if os.path.exists(invalid_path):
            return

        try:
            if source_is_zip:
                with zipfile.ZipFile(arad_zip_path) as archive:
                    mat_bytes = io.BytesIO(archive.read(arad_files[index]))
                mat_source = mat_bytes
            else:
                mat_source = os.path.join(arad_dir, arad_files[index])

            with h5py.File(mat_source, 'r') as mat_contents:
                cube = np.asarray(mat_contents['cube'])

            # h5py exposes the MATLAB cube as (bands, width, height). Match
            # the reference's permute(2, 1, 0), yielding (482, 512, bands).
            if cube.shape[0] == 31:
                cube = np.transpose(cube, (2, 1, 0))
            elif cube.shape[-1] != 31:
                raise ValueError(f'Unexpected ARAD cube shape: {cube.shape}')

            # Interpolate from 31 bands to 301 bands (10nm resolution)
            # Match the reference implementation: band-by-band linear interpolation
            H, W, bands = cube.shape

            if bands == 31:
                new_cube = []
                for i in range(bands - 1):
                    for j in range(10):
                        image1 = cube[:, :, i]
                        image2 = cube[:, :, i + 1]
                        new_image = image1 * (1 - j / 10) + image2 * (j / 10)
                        new_cube.append(new_image)
                new_cube.append(cube[:, :, -1])
                cube = np.stack(new_cube, axis=-1)  # (H, W, 301)

            # Convert to LMS: (H, W, 301) @ (301, 4) = (H, W, 4)
            lms_np = np.matmul(cube, cone_fundamentals_np)

            # Upsample if needed
            if H < dim_image or W < dim_image:
                if H < W:
                    multiplier = dim_image / H
                else:
                    multiplier = dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

                # Use jax.image.resize with bilinear interpolation to match
                # the reference's F.interpolate(..., mode='bilinear', align_corners=False).
                # Both JAX and PyTorch use half-centered pixel conventions by default.
                lms_jax = jnp.array(lms_np)
                lms_jax = jax.image.resize(lms_jax, (nH, nW, lms_jax.shape[-1]), method='bilinear', antialias=False)
                lms_np = np.array(lms_jax)

            # White world white balance
            current_white_point = lms_np.reshape(-1, lms_np.shape[-1]).max(0) + 1e-10
            lms_np = lms_np / current_white_point[None, None, :]
            lms_np *= defined_white_point

            # Save
            np.save(output_path, lms_np.astype(np.float32))

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            with open(invalid_path, 'w', encoding='utf-8') as marker:
                marker.write(f'{arad_files[index]}: {e}\n')
            print(f"\nSkipping corrupt file {arad_files[index]}: {e}")

    print('\n[1/2] Converting hyperspectral data to LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(hyperspectral_cube_to_lms, range(len(arad_files))),
            total=len(arad_files),
            desc='Converting to LMS'
        ))
    print('Done!')

    if not generate_crops:
        return

    # Step 2: Generate random crops
    def crop_image(crop_index):
        """Generate a single random crop."""
        # Determine which source image to use
        source_index = crop_index % len(arad_files)

        # Load full LMS image
        lms = np.load(
            f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{source_index}.npy'
        )

        # Skip if this is a dummy file (from corrupted source)
        if lms.shape[0] == 1 and lms.shape[1] == 1:
            # Try next image
            source_index = (source_index + 1) % len(arad_files)
            lms = np.load(
                f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{source_index}.npy'
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
        np.save(
            f'{root_dir}/Dataset/ARAD_{dim_image}_{dataset_type}/LMS/data/{crop_index}.npy',
            lms_crop
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
    # Get white point (convert to numpy)
    defined_white_point = np.array(CST.white_point)

    os.makedirs(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/', exist_ok=True)
    os.makedirs(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    # Load all interpolated data
    interpolated_dir = f'{root_dir}/Dataset/NTIRE2022_interpolated/data'
    file_list = sorted([f for f in os.listdir(interpolated_dir) if f.endswith('.npy')])

    # Convert hyperspectral to LMS
    def hyperspectral_cube_to_lms(index):
        if not os.path.exists(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.npy'):
            file_path = os.path.join(interpolated_dir, file_list[index])
            cube = np.load(file_path)

            # Convert to numpy for processing
            cone_fundamentals_np = np.array(CST.cone_fundamentals)

            # Matrix multiply: (H, W, 301) @ (301, 4) = (H, W, 4)
            lms_np = np.matmul(cube, cone_fundamentals_np)

            H, W, _ = lms_np.shape
            if H < dim_image or W < dim_image:
                # Need to upsample
                if H < W:
                    multiplier = dim_image / H
                else:
                    multiplier = dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

            # Use jax.image.resize with bilinear interpolation to match
                # the reference's F.interpolate(..., mode='bilinear', align_corners=False).
                # Both JAX and PyTorch use half-centered pixel conventions by default.
                lms_jax = jnp.array(lms_np)
                lms_jax = jax.image.resize(lms_jax, (nH, nW, lms_jax.shape[-1]), method='bilinear', antialias=False)
                lms_np = np.array(lms_jax)

            # White world white balance
            current_white_point = lms_np.reshape(-1, lms_np.shape[-1]).max(0) + 1e-10
            lms_np = lms_np / current_white_point[None, None, :]
            lms_np *= defined_white_point

            # Save
            np.save(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{index}.npy', lms_np.astype(np.float32))

    print('First, converting hyperspectral data to LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(hyperspectral_cube_to_lms, i) for i in range(len(file_list))]
    print('Done!')

    # Crop images
    def crop_image(index):
        if index % 1000 == 0:
            print(f'{index} / {dataset_size}')
        i = index % len(file_list)
        lms = np.load(
            f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/full_LMS/data/{i}.npy'
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
        np.save(f'{root_dir}/Dataset/NTIRE_{dim_image}_{dataset_type}/LMS/data/{index}.npy', lms)

    print('Next, cropping the images...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(crop_image, i) for i in range(dataset_size)]
    print('Done!')
