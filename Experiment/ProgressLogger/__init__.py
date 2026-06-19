"""JAX/Equinox implementation of Local Progress Logger.

Mirrors the original PyTorch Local logger but adapted for JAX/Equinox.
Local file-based logging only (no external experiment-tracker backends).

Logs:
- main_loss.png: Log-scale loss curve over gradient updates
- IP{id}_current.png: Latest side-by-side comparison (original vs predicted)
- IP{id}/{step}.png: Historical snapshots at each logging timestep
- IP{id}.gif: Progress video generated from historical snapshots
"""
import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import jax
import jax.numpy as jnp
import equinox as eqx

from Simulated.Retina.FV_spatial_sampling.helper import compute_required_image_resolution


# Try to load Futura font like the original
_FONT_PATH = None
for candidate in [
    'Tutorials/data/Futura.ttc',
    'Matisse_OG/Tutorials/data/Futura.ttc',
]:
    if os.path.exists(candidate):
        _FONT_PATH = candidate
        break

if _FONT_PATH:
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_FONT_PATH)
    _prop = font_manager.FontProperties(fname=_FONT_PATH)
    plt.rcParams['font.family'] = 'sans-serif'
else:
    _prop = None


def largest_valid_region_square(matrix: np.ndarray) -> tuple:
    """Find the largest square region of zeros in a binary matrix.

    Returns (y_min, x_min, y_max, x_max) coordinates.
    """
    rows, cols = matrix.shape
    dp = np.zeros_like(matrix, dtype=int)

    max_size = 0
    bottom_right = (0, 0)

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                if i == 0 or j == 0:
                    dp[i, j] = 1
                else:
                    dp[i, j] = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + 1

                if dp[i, j] > max_size:
                    max_size = dp[i, j]
                    bottom_right = (i, j)

    y_max, x_max = bottom_right
    y_min = y_max - max_size + 1
    x_min = x_max - max_size + 1

    return (y_min, x_min, y_max, x_max)


class LocalProgressLogger:
    """Local file-based progress logger for JAX training.

    Saves loss curves, side-by-side image comparisons, and progress GIFs.
    """

    def __init__(
        self,
        experiment_name: str,
        required_image_resolution: int,
        root_dir: str = '.',
        test_images_dir: str = 'Experiment/ProgressLogger/test_images',
    ):
        """Initialize logger.

        Args:
            experiment_name: Name of the experiment (used for directory naming)
            required_image_resolution: Spatial resolution to resize test images to
            root_dir: Project root directory
            test_images_dir: Directory containing test PNG images
        """
        self.experiment_name = experiment_name
        self.required_image_resolution = required_image_resolution
        self.root_dir = root_dir

        # Load test images
        test_images = []
        test_dir = os.path.join(root_dir, test_images_dir)
        if os.path.isdir(test_dir):
            for file in sorted(os.listdir(test_dir)):
                if file.endswith('.png'):
                    img_path = os.path.join(test_dir, file)
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize((required_image_resolution, required_image_resolution))
                    test_images.append(np.array(image) / 255.0)

        self.test_images = np.asarray(test_images) if test_images else np.zeros((1, required_image_resolution, required_image_resolution, 3))
        self.num_test_images = self.test_images.shape[0]

        # Create logging directories
        self.log_dir = os.path.join(root_dir, 'Experiment', 'Logging', experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        for i in range(self.num_test_images):
            os.makedirs(os.path.join(self.log_dir, f'IP{i}'), exist_ok=True)

        # Loss tracking
        self.num_gradient_updates_list = []
        self.main_loss_list = []
        self.ns_cm_list = []
        self.ns_ip_list = []

        # Cached jitted unwarp. Compiling once is what makes unwarping affordable:
        # run eagerly the pipeline (nested vmap of RealNVP.inverse over a full
        # res*res grid + per-channel map_coordinates) materializes every
        # intermediate and peaked at ~40GB. Under jit XLA fuses and reuses
        # buffers, and the compile is reused across every logging step because
        # the model structure/shapes never change. Resolution is closed over as
        # a static value (it is used as an array shape).
        _res = self.required_image_resolution

        @eqx.filter_jit
        def _unwarp(cortex, warped_ip_chw):
            return cortex.get_unwarped_percept(
                warped_ip_chw, required_image_resolution=_res
            )

        self._unwarp_fn = _unwarp

    def log_progress(
        self,
        simulating_tetra: bool,
        retina,
        cortex,
        num_gradient_updates: int,
        main_loss: float,
        ns_cm_loss: float,
        ns_ip_loss: float,
    ):
        """Log training progress at a checkpoint timestep.

        Args:
            simulating_tetra: Whether tetrachromacy is being simulated
            retina: RetinaModel instance (Equinox module)
            cortex: CortexModel instance (Equinox module)
            num_gradient_updates: Current training step
            main_loss: Main reconstruction loss value
            ns_cm_loss: Neural scope cone mosaic loss
            ns_ip_loss: Neural scope internal percept loss
        """
        # Track losses
        self.num_gradient_updates_list.append(num_gradient_updates)
        self.main_loss_list.append(float(main_loss))

        # Plot loss curve
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.num_gradient_updates_list, self.main_loss_list, label='Main Loss')
        plt.yscale('log')
        plt.xlabel('Number of Gradient Updates')
        plt.ylabel('Main Loss')
        plt.title('Main Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'main_loss.png'))
        plt.close()

        if not simulating_tetra and self.num_test_images > 0:
            # Simulate retina + cortex on test images
            pred_internal_percept_sRGB = self._run_inference(retina, cortex)
            pred_internal_percept_sRGB = np.clip(pred_internal_percept_sRGB, 0, 1)

            # Plot side-by-side comparisons
            for image_id, image in enumerate(pred_internal_percept_sRGB):
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(self.test_images[image_id])
                ax[0].set_title('Original')
                ax[0].axis('off')
                ax[1].imshow(image)
                ax[1].set_title('Predicted')
                ax[1].axis('off')
                fig.suptitle(f'After {num_gradient_updates:06d} gradient updates')

                # Save at two locations: current + historical
                plt.savefig(os.path.join(self.log_dir, f'IP{image_id}_current.png'))
                plt.savefig(os.path.join(self.log_dir, f'IP{image_id}', f'{num_gradient_updates}.png'))
                plt.close()

    def _run_inference(self, retina, cortex) -> np.ndarray:
        """Run retina + cortex inference on test images one at a time.

        Processes each test image individually to avoid memory overflow.
        Returns predicted internal percept in sRGB space, shape (N, H, W, 3).
        """
        import gc
        results = []
        crop_box = None  # largest valid square; identical across images, computed once

        for img_idx in range(self.num_test_images):
            # Single image: (1, H, W, 3)
            test_sRGB = jnp.array(self.test_images[img_idx:img_idx + 1])
            test_linsRGB = retina.CST.sRGB_to_linsRGB(test_sRGB)
            test_LMS = retina.CST.linsRGB_to_LMS(test_linsRGB)
            # (1, H, W, 4) -> (1, 4, H, W) -> (1, 1, 4, H, W)
            test_LMS = jnp.transpose(test_LMS, (0, 3, 1, 2))[:, None, ...]

            # Retina forward (single timestep, no eye motion)
            key = jax.random.PRNGKey(0)
            test_warped = retina.SpatialSampling(test_LMS)
            test_pa = retina.SpectralSampling(test_warped, key=key)
            test_pa = test_pa[:, 0, ...]  # Squeeze timestep: (1, 1, H, W)
            test_ons = retina.LateralInhibition(test_pa, key=key)
            test_ons = retina.SpikeConversion(test_ons)

            # Cortex decode
            warped_ip = cortex.decode(test_ons)

            # Neural scope -> sRGB (still in warped cone-space here)
            warped_linsRGB = cortex.ns_ip(warped_ip)
            warped_linsRGB = jnp.transpose(warped_linsRGB, (0, 2, 3, 1))
            warped_sRGB = retina.CST.linsRGB_to_sRGB(warped_linsRGB)
            warped_sRGB = jnp.clip(warped_sRGB, 0, 1)

            # Unwarp back to image space, matching the torch reference. The
            # warped percept lives in cone-sampling space where demosaicing
            # artifacts are exaggerated; the unwarp rectifies it. Uses the cached
            # jitted unwarp (compiled once) so this no longer blows up memory.
            warped_chw = jnp.transpose(warped_sRGB, (0, 3, 1, 2))  # (1, 3, H, W)
            unwarped_sRGB, invalid_regions = self._unwarp_fn(cortex, warped_chw)  # (1,H',W',3),(H',W')

            # Crop to the largest fully-valid square, discarding the distorted
            # warped borders — this is what the torch logger does
            # (Local.py: largest_valid_region_square(invalid_regions) then crop).
            # invalid_regions depends only on the (shared) cell-position warp, not
            # on the image, so compute the crop box once and reuse it.
            if crop_box is None:
                crop_box = largest_valid_region_square(np.asarray(invalid_regions))
            y0, x0, y1, x1 = crop_box
            pred = np.array(unwarped_sRGB[0])[y0:y1, x0:x1]  # (h, w, 3)
            results.append(pred)

            # Aggressive cleanup after each image
            del test_sRGB, test_linsRGB, test_LMS, test_warped, test_pa
            del test_ons, warped_ip, warped_linsRGB, warped_sRGB, warped_chw, unwarped_sRGB, pred
            gc.collect()

        return np.stack(results, axis=0)

    def generate_progress_video(self):
        """Generate GIF progress videos from logged snapshots."""
        for image_id in range(self.num_test_images):
            ip_dir = os.path.join(self.log_dir, f'IP{image_id}')
            if not os.path.isdir(ip_dir):
                continue

            images = []
            for filename in sorted(os.listdir(ip_dir)):
                if filename.endswith('.png'):
                    img_path = os.path.join(ip_dir, filename)
                    images.append(Image.open(img_path))

            if images:
                gif_path = os.path.join(self.log_dir, f'IP{image_id}.gif')
                imageio.mimsave(gif_path, images, fps=20)
                print(f"  Generated {gif_path}")
