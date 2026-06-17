import os
import io
import json
import zipfile
import torch
import numpy as np
from root_config import *
from tqdm.auto import tqdm
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from Dataset.Abstract import Dataset
from Dataset import register_class


def _numeric_path_key(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def _is_valid_cached_image(path, dim_image):
    """Cheap validity check on a cached full LMS image via the mmap header."""
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


def _list_arad_sources(arad_dir, arad_zip):
    """Return (sorted .mat names, source_is_zip). The official mirror may leave
    the spectral data compressed, so accept either an extracted directory or the
    raw zip."""
    if os.path.isdir(arad_dir):
        return sorted(f for f in os.listdir(arad_dir) if f.endswith('.mat')), False
    if os.path.isfile(arad_zip):
        with zipfile.ZipFile(arad_zip) as archive:
            return sorted(n for n in archive.namelist() if n.endswith('.mat')), True
    raise RuntimeError(
        'ARAD_1K training data not found. Expected raw .mat files at:\n'
        f'  {arad_dir}/  (extracted)  or\n'
        f'  {arad_zip}  (zip archive)'
    )


@register_class("NTIRE")
class NTIRE(Dataset):
    """NTIRE/ARAD_1K hyperspectral dataset with runtime random crops.

    Full-resolution LMS images are converted directly from the raw ARAD_1K
    ``.mat`` cubes (read from ``ARAD_1K_Mirror/Train_spectral`` or its zip) and
    cached as ``ARAD_{dim}_{type}/full_LMS/data/*.npy``. Crops are generated on
    demand in ``__getitem__`` by memory-mapping a full image and reading only
    the cropped region. This mirrors the JAX/Grain pipeline and avoids both the
    ~500GB of precomputed crops and the ``NTIRE2022_interpolated`` ``.pt``
    intermediate the original implementation depended on.
    """

    def __init__(self, params, retina):
        super(NTIRE, self).__init__(params, retina)

        dim_image = retina.required_image_resolution + (params['Experiment']['timesteps_per_image'] - 1) * 2 * params['RetinaModel']['max_shift_size']

        self.dataset_name = params['Dataset']['dataset_name']
        self.dim_image = dim_image
        self.dataset_size = DATASET_SIZE

        dataset_type = 'LMS'

        # if the peak frequencies are not the default values, add them to the dataset type (+ regenerate the dataset)
        if 'cone_fundamentals' in params['RetinaModel']['retina_spectral_sampling']:
            cone_fundamentals_params = params['RetinaModel']['retina_spectral_sampling']['cone_fundamentals']
        else:
            cone_fundamentals_params = {'L': 560, 'M': 530, 'S': 419}

        for key in cone_fundamentals_params:
            if key == 'L':
                L_peak = cone_fundamentals_params[key]
                if L_peak != 560: # default peak of the L-cone at 560 nm
                    dataset_type += f'_L{L_peak}'
            elif key == 'M':
                M_peak = cone_fundamentals_params[key]
                if M_peak != 530: # default peak of the M-cone at 530 nm
                    dataset_type += f'_M{M_peak}'
            elif key == 'S':
                S_peak = cone_fundamentals_params[key]
                if S_peak != 419: # default peak of the S-cone at 419 nm
                    dataset_type += f'_S{S_peak}'
            elif key == 'Q':
                Q_peak = cone_fundamentals_params[key]
                dataset_type += f'_Q{Q_peak}'

        # Raw ARAD_1K hyperspectral source (NTIRE 2022 spectral track).
        arad_dir = f'{ROOT_DIR}/Dataset/ARAD_1K_Mirror/Train_spectral'
        arad_zip = f'{ROOT_DIR}/Dataset/ARAD_1K_Mirror/Train_spectral.zip'
        arad_json_train = f'{ROOT_DIR}/Dataset/ARAD_1K_Mirror/.arad_good_ids_train.json'

        source_files, _ = _list_arad_sources(arad_dir, arad_zip)
        source_count = len(source_files)

        # Optional sanity check against the published "good ids" manifest.
        if os.path.exists(arad_json_train):
            with open(arad_json_train, 'r') as f:
                expected_train_ids = set(json.load(f))
            found_ids = {os.path.splitext(os.path.basename(f))[0] for f in source_files}
            missing = expected_train_ids - found_ids
            if missing:
                print(f'  Warning: {len(missing)} expected training file(s) missing from the ARAD source')
            else:
                print(f'  All {len(expected_train_ids)} expected training files present')

        # Converted full LMS images live under ARAD_{dim}_{type} (matching the
        # JAX pipeline). Crops are generated on demand from these.
        self.data_dir = f'{ROOT_DIR}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data'

        def _list_cache():
            if not os.path.exists(self.data_dir):
                return [], []
            entries = os.listdir(self.data_dir)
            valid = [
                f for f in entries
                if f.endswith('.npy')
                and _is_valid_cached_image(os.path.join(self.data_dir, f), dim_image)
            ]
            invalid = [f for f in entries if f.endswith('.invalid')]
            return valid, invalid

        cached_files, invalid_markers = _list_cache()

        # Convert from the raw hyperspectral cubes only if the cache is
        # incomplete. A source image is considered handled once it has either a
        # valid cached .npy or an .invalid marker (corrupt source).
        if len(cached_files) + len(invalid_markers) < source_count:
            print('=== Preprocessing ARAD_1K hyperspectral data... ===')
            print('    Caching full LMS images; crops are generated on demand')
            preprocess_ARAD_hyperspectral_data(
                dim_image, dataset_type, retina.CST, arad_dir, arad_zip,
            )
            print('=== Done! ===')
            cached_files, invalid_markers = _list_cache()

        self.file_list = sorted(
            (os.path.join(self.data_dir, f) for f in cached_files),
            key=_numeric_path_key,
        )

        if len(self.file_list) == 0:
            raise RuntimeError(f'No cached LMS images found in {self.data_dir}')

        excluded_count = source_count - len(self.file_list)
        print (
            f'Dataset ready: {self.dataset_size} samples '
            f'from {len(self.file_list)} cached images'
        )
        if excluded_count:
            print(f'Excluded {excluded_count} corrupt source image(s)')


    def __getitem__(self, index):
        source_index = index % len(self.file_list)

        # Memory-mapped load so only the cropped region is read into memory.
        image = np.load(self.file_list[source_index], mmap_mode='r')

        H, W = image.shape[0], image.shape[1]
        if H < self.dim_image or W < self.dim_image:
            raise ValueError(
                f'Cached image is too small: {H}x{W} '
                f'< {self.dim_image}x{self.dim_image}'
            )

        # Match the reference's np.random.randint(0, dim - crop) behavior,
        # pinning the offset to 0 along any axis equal to the crop size.
        x = np.random.randint(0, H - self.dim_image) if H > self.dim_image else 0
        y = np.random.randint(0, W - self.dim_image) if W > self.dim_image else 0

        crop = np.ascontiguousarray(image[x:x + self.dim_image, y:y + self.dim_image])
        return torch.from_numpy(crop)


    def __len__(self):
        return self.dataset_size


def preprocess_ARAD_hyperspectral_data(dim_image, dataset_type, CST, arad_dir, arad_zip):
    """Convert raw ARAD_1K hyperspectral cubes to full-resolution LMS images.

    Reads each ``.mat`` cube directly (no ``NTIRE2022_interpolated`` ``.pt``
    intermediate), spectrally interpolates 31 -> 301 bands band-by-band,
    projects onto the cone fundamentals, white-balances, and caches one full LMS
    ``.npy`` per source cube. Crops are produced on demand at train time.

    The band interpolation, cone projection and white balance run in numpy
    (thread-safe under the ThreadPoolExecutor); only the optional upsample uses
    torch ``F.interpolate`` to match the reference's bilinear resize.
    """

    out_dir = f'{ROOT_DIR}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data'
    os.makedirs(out_dir, exist_ok=True)

    # Pull the colour-space tensors to host once; the per-image work is numpy.
    cone_fundamentals = CST.cone_fundamentals.detach().cpu().numpy()   # (301, 4)
    defined_white_point = CST.white_point.detach().cpu().numpy()        # (1, 1, 4)

    source_files, source_is_zip = _list_arad_sources(arad_dir, arad_zip)

    def hyperspectral_cube_to_lms(index):
        import h5py

        output_path = f'{out_dir}/{index}.npy'
        invalid_path = f'{output_path}.invalid'
        if os.path.exists(output_path):
            if _is_valid_cached_image(output_path, dim_image):
                return
            os.remove(output_path)
        if os.path.exists(invalid_path):
            return

        try:
            if source_is_zip:
                with zipfile.ZipFile(arad_zip) as archive:
                    mat_source = io.BytesIO(archive.read(source_files[index]))
            else:
                mat_source = os.path.join(arad_dir, source_files[index])

            with h5py.File(mat_source, 'r') as mat_contents:
                cube = np.asarray(mat_contents['cube'])

            # h5py exposes the MATLAB cube as (bands, width, height). Match the
            # reference's permute(2, 1, 0), yielding (H, W, bands).
            if cube.shape[0] == 31:
                cube = np.transpose(cube, (2, 1, 0))
            elif cube.shape[-1] != 31:
                raise ValueError(f'Unexpected ARAD cube shape: {cube.shape}')

            H, W, bands = cube.shape

            # Spectrally interpolate 31 -> 301 bands (band-by-band linear), as in
            # the original NTIRE interpolation step.
            if bands == 31:
                new_cube = []
                for i in range(bands - 1):
                    for j in range(10):
                        new_cube.append(cube[:, :, i] * (1 - j / 10) + cube[:, :, i + 1] * (j / 10))
                new_cube.append(cube[:, :, -1])
                cube = np.stack(new_cube, axis=-1)   # (H, W, 301)

            # Project onto the cone fundamentals: (H, W, 301) @ (301, 4).
            lms = np.matmul(cube, cone_fundamentals)   # (H, W, 4)

            # Upsample if the image is smaller than the required crop size,
            # using the reference's bilinear resize.
            if H < dim_image or W < dim_image:
                multiplier = dim_image / H if H < W else dim_image / W
                nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))
                lms_t = torch.from_numpy(lms).permute(2, 0, 1).unsqueeze(0)
                lms_t = F.interpolate(lms_t, size=(nH, nW), mode='bilinear', align_corners=False)
                lms = lms_t.squeeze(0).permute(1, 2, 0).numpy()

            # White-world white balance.
            current_white_point = lms.reshape(-1, lms.shape[-1]).max(0) + 1e-10
            lms = lms / current_white_point[None, None, :]
            lms = lms * defined_white_point

            np.save(output_path, lms.astype(np.float32))

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            with open(invalid_path, 'w', encoding='utf-8') as marker:
                marker.write(f'{source_files[index]}: {e}\n')
            print(f'\nSkipping corrupt file {source_files[index]}: {e}')

    print('Converting hyperspectral data to full-resolution LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(hyperspectral_cube_to_lms, range(len(source_files))),
            total=len(source_files),
            desc='Converting to LMS',
        ))
    print('Done!')
