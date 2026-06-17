import os
import torch
import numpy as np
from root_config import *
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
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


@register_class("NTIRE")
class NTIRE(Dataset):
    """NTIRE/ARAD_1K hyperspectral dataset with runtime random crops.

    Only the full-resolution LMS images are cached on disk
    (``full_LMS/data/*.npy``). Crops are generated on demand in
    ``__getitem__`` by memory-mapping a full image and reading only the
    cropped region. This mirrors the JAX/Grain pipeline and avoids
    materializing the ~50k precomputed crops (~500GB) that the previous
    implementation wrote to disk.
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

        # ARAD_1K is the NTIRE 2022 hyperspectral dataset; its converted full
        # LMS images live under ARAD_{dim}_{type} (matching the JAX pipeline).
        self.data_dir = f'{ROOT_DIR}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data'

        def _list_cached():
            if not os.path.exists(self.data_dir):
                return []
            return [
                f for f in os.listdir(self.data_dir)
                if f.endswith('.npy')
                and _is_valid_cached_image(os.path.join(self.data_dir, f), dim_image)
            ]

        cached_files = _list_cached()

        # Only convert from the raw hyperspectral data if no cache exists yet.
        # When the full LMS images are already cached, crops are generated on
        # demand and no interpolation/preprocessing is needed.
        if len(cached_files) == 0:
            if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/0899.pt'):
                print ('=== Spectrally interpolating NTIRE hyperspectral data... ===')
                interpolate_NTIRE_hyperspectral_data()
                print ('=== Done! ===')

            print ('=== Preprocessing NTIRE hyperspectral data... ===')
            print ('    Caching full LMS images; crops are generated on demand')
            preprocess_NTIRE_hyperspectral_data(dim_image, dataset_type, retina.CST)
            print ('=== Done! ===')
            cached_files = _list_cached()

        self.file_list = sorted(
            (os.path.join(self.data_dir, f) for f in cached_files),
            key=_numeric_path_key,
        )

        if len(self.file_list) == 0:
            raise RuntimeError(f'No cached LMS images found in {self.data_dir}')

        print (
            f'Dataset ready: {self.dataset_size} samples '
            f'from {len(self.file_list)} cached images'
        )


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


def interpolate_NTIRE_hyperspectral_data():
    import h5py

    os.makedirs(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data', exist_ok=True)
    data_folder = f'{ROOT_DIR}/Dataset/NTIRE2022_original/data'

    def interpolate(index):
        if os.path.exists(f'{data_folder}/ARAD_1K_{index:04d}.mat'):
            if not os.path.exists(f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt'):
                # load mat file
                mat_contents = h5py.File(f'{data_folder}/ARAD_1K_{index:04d}.mat', 'r')

                bands = np.asarray(mat_contents['bands'])[:,0]
                cube = np.asarray(mat_contents['cube'])

                new_cube = []
                for i in range(len(bands)-1):
                    for j in range(10):
                        image1 = cube[i]
                        image2 = cube[i+1]
                        new_image = image1 * (1-j/10) + image2 * (j/10)
                        new_cube.append(new_image)

                new_cube.append(cube[-1])
                new_cube = np.asarray(new_cube)
                new_cube = torch.FloatTensor(new_cube)
                new_cube = new_cube.permute(2,1,0)

                torch.save(new_cube, f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated/data/{index:04d}.pt')


    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        _ = [executor.submit(interpolate, i) for i in range(1, 1001)]
    print ('Done!')


def preprocess_NTIRE_hyperspectral_data(dim_image, dataset_type, CST):
    """Convert interpolated hyperspectral cubes to full-resolution LMS images.

    Crops are no longer materialized to disk — only one full LMS image per
    source cube is cached (as ``.npy`` so ``__getitem__`` can memory-map it and
    read just the cropped region at runtime).
    """

    defined_white_point = CST.white_point
    device = CST.device

    os.makedirs(f'{ROOT_DIR}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/', exist_ok=True)

    def local_loader(path):
        return torch.load(path, weights_only=True, map_location=device)

    all_data = DatasetFolder(root=f'{ROOT_DIR}/Dataset/NTIRE2022_interpolated', loader=local_loader, extensions='.pt')
    L = len(all_data)

    def hyperspectral_cube_to_lms(index):
        output_path = f'{ROOT_DIR}/Dataset/ARAD_{dim_image}_{dataset_type}/full_LMS/data/{index}.npy'
        if os.path.exists(output_path):
            return

        cube = all_data[index][0]
        cube = cube.to(device)

        lms = torch.matmul(cube, CST.cone_fundamentals)

        (H, W, _) = lms.shape
        if H < dim_image or W < dim_image:
            if H < W:
                multiplier = dim_image / H
            else:
                multiplier = dim_image / W
            nH, nW = int(np.ceil(H * multiplier)), int(np.ceil(W * multiplier))

            lms = F.interpolate((lms).permute(2,0,1).unsqueeze(0), size=(nH, nW), mode='bilinear', align_corners=False).squeeze().permute(1,2,0)

        # white world white balance
        current_white_point = (lms.reshape(-1, lms.shape[-1]).max(0)[0] + 1e-10) # (8, 3)
        lms = lms / current_white_point[None,None,:]
        lms *= defined_white_point

        lms_save = lms.detach().cpu().numpy().astype(np.float32)
        np.save(output_path, lms_save)

    print ('Converting hyperspectral data to full-resolution LMS...')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(
            executor.map(hyperspectral_cube_to_lms, range(L)),
            total=L,
            desc='Converting to LMS',
        ))
    print ('Done!')
