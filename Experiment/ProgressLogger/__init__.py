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


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Canonical sRGB EOTF (IEC 61966-2-1) -> linear radiance."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR in sRGB space (dB). Returns inf on an exact match."""
    mse = float(np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2))
    if mse <= 0.0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def psnr_linear(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR on linear radiance (dB) -- the domain the model reconstructs in."""
    mse = float(np.mean((_srgb_to_linear(pred) - _srgb_to_linear(target)) ** 2))
    if mse <= 0.0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def ssim_score(pred: np.ndarray, target: np.ndarray):
    """SSIM index (higher = better). None if scikit-image is unavailable."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # noqa: BLE001
        return None
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape:
        return None
    return float(
        structural_similarity(
            a, b, data_range=1.0, channel_axis=-1,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
    )


_LPIPS = None
_LPIPS_ERR = None


def lpips_score(pred: np.ndarray, target: np.ndarray):
    """LPIPS distance (lower = better). None if torch/lpips is unavailable.

    Runs on CPU and is only ever invoked on the sparse logging schedule, so it
    stays off the JAX/MPS training hot path.
    """
    global _LPIPS, _LPIPS_ERR
    if _LPIPS_ERR is not None:
        return None
    if _LPIPS is None:
        try:
            import torch  # noqa: F401
            import lpips
            _LPIPS = lpips.LPIPS(net='alex', verbose=False).eval()
            for p in _LPIPS.parameters():
                p.requires_grad_(False)
        except Exception as e:  # noqa: BLE001
            _LPIPS_ERR = str(e)
            return None
    import torch
    x = torch.from_numpy(np.asarray(pred)[None].transpose(0, 3, 1, 2).astype('float32')) * 2.0 - 1.0
    y = torch.from_numpy(np.asarray(target)[None].transpose(0, 3, 1, 2).astype('float32')) * 2.0 - 1.0
    with torch.no_grad():
        d = _LPIPS(x, y)
    return float(d.item())


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

        # Reconstruction-quality tracking (mean over the test set, per logged
        # step). Populated in log_progress from the unwarped predictions.
        self.eval_steps_list = []
        self.psnr_list = []
        self.psnr_lin_list = []
        self.ssim_list = []
        self.lpips_list = []

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
            pred_internal_percept_sRGB, crop_box = self._run_inference(retina, cortex)
            pred_internal_percept_sRGB = np.clip(pred_internal_percept_sRGB, 0, 1)

            # --- Reconstruction-quality metrics on the unwarped predictions ---
            # Targets are cropped to the same largest-valid square the
            # predictions were cropped to, so PSNR/SSIM/LPIPS compare aligned
            # content (and exclude the distorted warped borders).
            self._log_eval_metrics(
                pred_internal_percept_sRGB, crop_box, num_gradient_updates
            )

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

    def _run_inference(self, retina, cortex):
        """Run retina + cortex inference on test images one at a time.

        Processes each test image individually to avoid memory overflow.
        Returns ``(predictions, crop_box)`` where ``predictions`` is the
        unwarped internal percept in sRGB space, shape (N, h, w, 3), cropped to
        the largest valid square ``crop_box = (y0, x0, y1, x1)``.
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

            key = jax.random.PRNGKey(0)
            if cortex.temporal_fusion == 'oracle':
                margin = retina.EyeMotion.max_shift_size
                full_field = jnp.pad(
                    test_LMS[:, 0],
                    ((0, 0), (0, 0), (margin, margin), (margin, margin)),
                    mode='reflect',
                )
                test_ons_pair, true_dxy, test_warped = retina(
                    full_field, key=key
                )
                warped_ip, _ = cortex.decode_fused(
                    test_ons_pair[:, 0],
                    test_ons_pair[:, 1],
                    true_dxy[:, 0],
                )
                # Fusion is aligned to the second view; return to the centered
                # first view used by the side-by-side reference image.
                warped_ip, _ = cortex.P_cell_position.efficient_warping(
                    warped_ip, -true_dxy[:, 0]
                )
                test_pa = test_ons = test_ons_pair
            else:
                # Retina forward (single timestep, no eye motion)
                test_warped = retina.SpatialSampling(test_LMS)
                test_pa = retina.SpectralSampling(test_warped, key=key)
                test_pa = test_pa[:, 0, ...]
                test_ons = retina.LateralInhibition(test_pa, key=key)
                test_ons = retina.SpikeConversion(test_ons)
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

        return np.stack(results, axis=0), crop_box

    def _log_eval_metrics(self, preds: np.ndarray, crop_box, num_gradient_updates: int):
        """Compute + persist PSNR/SSIM/LPIPS for the unwarped predictions.

        Writes one JSON line per logged step to ``eval_metrics.jsonl`` (per-image
        and mean values) and refreshes ``eval_metrics.png`` with the curves over
        training so reconstruction quality is tracked alongside the loss.
        """
        y0, x0, y1, x1 = crop_box
        per_image = []
        for i, pred in enumerate(preds):
            target = np.clip(self.test_images[i][y0:y1, x0:x1], 0.0, 1.0)
            pred = np.clip(pred, 0.0, 1.0)
            per_image.append({
                'psnr': psnr(pred, target),
                'psnr_lin': psnr_linear(pred, target),
                'ssim': ssim_score(pred, target),
                'lpips': lpips_score(pred, target),
            })

        def _mean(field):
            vals = [m[field] for m in per_image if m[field] is not None and np.isfinite(m[field])]
            return float(np.mean(vals)) if vals else None

        mean_psnr = _mean('psnr')
        mean_psnr_lin = _mean('psnr_lin')
        mean_ssim = _mean('ssim')
        mean_lpips = _mean('lpips')

        record = {
            'step': num_gradient_updates,
            'mean_psnr': mean_psnr,
            'mean_psnr_lin': mean_psnr_lin,
            'mean_ssim': mean_ssim,
            'mean_lpips': mean_lpips,
            'per_image': per_image,
        }
        with open(os.path.join(self.log_dir, 'eval_metrics.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        self.eval_steps_list.append(num_gradient_updates)
        self.psnr_list.append(mean_psnr)
        self.psnr_lin_list.append(mean_psnr_lin)
        self.ssim_list.append(mean_ssim)
        self.lpips_list.append(mean_lpips)

        def _fmt(v, suffix=''):
            return f'{v:.4f}{suffix}' if v is not None else 'n/a'

        print(
            f"  [eval @ {num_gradient_updates}] "
            f"PSNR {_fmt(mean_psnr, ' dB')} | PSNR-lin {_fmt(mean_psnr_lin, ' dB')} | "
            f"SSIM {_fmt(mean_ssim)} | LPIPS {_fmt(mean_lpips)}"
        )

        # Curves: PSNR (left axis) and SSIM/LPIPS (right axis), masking n/a.
        def _series(values):
            xs = [s for s, v in zip(self.eval_steps_list, values) if v is not None]
            ys = [v for v in values if v is not None]
            return xs, ys

        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = ax.twinx()
        plotted = False
        for values, label, color, axis in [
            (self.psnr_list, 'PSNR (dB)', 'crimson', ax),
            (self.psnr_lin_list, 'PSNR-lin (dB)', 'darkorange', ax),
        ]:
            xs, ys = _series(values)
            if xs:
                axis.plot(xs, ys, 'o-', color=color, label=label)
                plotted = True
        for values, label, color in [
            (self.ssim_list, 'SSIM', 'steelblue'),
            (self.lpips_list, 'LPIPS', 'seagreen'),
        ]:
            xs, ys = _series(values)
            if xs:
                ax2.plot(xs, ys, 's--', color=color, alpha=0.8, label=label)
                plotted = True
        if plotted:
            ax.set_xlabel('Number of Gradient Updates')
            ax.set_ylabel('PSNR (dB)')
            ax2.set_ylabel('SSIM / LPIPS')
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=8)
            ax.set_title('Reconstruction quality vs training step')
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, 'eval_metrics.png'))
        plt.close(fig)

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
