import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Must be set before JAX is imported anywhere.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import re
    import glob
    import time

    import yaml
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import orbax.checkpoint as ocp
    import matplotlib.pyplot as plt
    from PIL import Image

    import marimo as mo
    from Simulated.Retina import RetinaModel
    from Simulated.Cortex import CortexModel

    ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    CONFIG_DIR = os.path.join(ROOT, "Experiment", "Config", "Default")
    WEIGHTS_DIR = os.path.join(ROOT, "Experiment", "LearnedWeights")
    TEST_IMAGES_DIR = os.path.join(ROOT, "Experiment", "ProgressLogger", "test_images")

    mo.md("# Matisse — Inference, PSNR/LPIPS Comparison")
    return (
        CONFIG_DIR,
        CortexModel,
        Image,
        ROOT,
        RetinaModel,
        TEST_IMAGES_DIR,
        WEIGHTS_DIR,
        eqx,
        glob,
        jax,
        jnp,
        mo,
        np,
        ocp,
        os,
        plt,
        re,
        time,
        yaml,
    )


@app.cell
def _(
    CONFIG_DIR,
    CortexModel,
    Image,
    ROOT,
    RetinaModel,
    TEST_IMAGES_DIR,
    WEIGHTS_DIR,
    eqx,
    glob,
    jax,
    jnp,
    np,
    ocp,
    os,
    re,
    time,
    yaml,
):
    def experiment_name(config_path):
        with open(config_path, "r") as f:
            params = yaml.safe_load(f)
        return params.get("Experiment", {}).get("name")


    def configs_with_weights():
        rows = []
        for path in sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))):
            name = experiment_name(path)
            if name and os.path.isdir(os.path.join(WEIGHTS_DIR, name)):
                rows.append({"label": f"{name}", "path": path, "name": name})
        return rows


    def list_weights(name):
        """Aggregate every saved checkpoint for an experiment.

        Prefers `.eqx` files (one per logging timestep, directly loadable);
        falls back to Orbax step directories. Returns (step:int, path:str)
        sorted by step.
        """
        exp_dir = os.path.join(WEIGHTS_DIR, name) if name else ""
        if not exp_dir or not os.path.isdir(exp_dir):
            return []
        weights = []
        for path in glob.glob(os.path.join(exp_dir, "model_*.eqx")):
            m = re.search(r"model_(\d+)\.eqx$", os.path.basename(path))
            if m:
                weights.append((int(m.group(1)), path))
        if not weights:
            for entry in os.listdir(exp_dir):
                step_dir = os.path.join(exp_dir, entry)
                if entry.isdigit() and os.path.isdir(step_dir):
                    weights.append((int(entry), step_dir))
        weights.sort(key=lambda t: t[0])
        return weights


    def build_models(params):
        root_dir = params.get("root_dir", ROOT)
        experiment = params["Experiment"]
        simulating_tetra = experiment["simulating_tetra"]
        simulation_size = experiment.get("simulation_size") or params.get(
            "RetinaModel", {}
        ).get("simulation_size", 256)

        retina = RetinaModel(
            simulation_size=simulation_size,
            timesteps_per_image=experiment["timesteps_per_image"],
            max_shift_size=params["RetinaModel"]["max_shift_size"],
            cone_types_str=params["RetinaModel"]["retina_spectral_sampling"]["cone_types"],
            cone_distribution_type=params.get("RetinaModel", {})
            .get("retina_spatial_sampling", {})
            .get("cone_distribution", "Human"),
            simulating_tetra=simulating_tetra,
            cone_fundamentals_params=params["RetinaModel"]["retina_spectral_sampling"].get(
                "cone_fundamentals"
            ),
            cone_gain_adaptation=params["RetinaModel"]["retina_spectral_sampling"].get(
                "gain_adaptation", "none"
            ),
            root_dir=root_dir,
        )

        cortical_cfg = params.get("CorticalModel", params.get("CortexModel", {}))
        dm = cortical_cfg.get("cortex_learn_demosaicing", {})
        cortex = CortexModel(
            latent_dim=cortical_cfg.get("latent_dim", 8),
            simulation_size=simulation_size,
            required_image_resolution=retina.required_image_resolution,
            simulating_tetra=simulating_tetra,
            demosaicing_type=dm.get("type", "Default"),
            demosaicing_base_channels=dm.get("base_channels", 16),
            demosaicing_compute_dtype=dm.get("compute_dtype", "float32"),
            demosaicing_context_channels=dm.get("context_channels", 16),
            demosaicing_context_kernel_size=dm.get("context_kernel_size", 5),
            demosaicing_hidden_channels=dm.get("hidden_channels", 32),
            demosaicing_hidden_kernel_size=dm.get("hidden_kernel_size", 1),
            demosaicing_num_frequencies=dm.get("num_frequencies", 6),
            demosaicing_omega0=dm.get("omega0", 10.0),
            demosaicing_activation=dm.get("activation", "sine"),
            demosaicing_conditioning=dm.get("conditioning", "none"),
            demosaicing_gaussian_kernel_size=dm.get("gaussian_kernel_size", 9),
            demosaicing_gaussian_sigma=dm.get("gaussian_sigma", 2.0),
            demosaicing_gaussian_epsilon=dm.get("gaussian_epsilon", 1e-3),
            temporal_fusion=cortical_cfg.get("temporal_fusion", "none"),
            key=jax.random.PRNGKey(42),
        )
        return retina, cortex


    def load_weights(cortex, weight_path, step):
        if weight_path.endswith(".eqx"):
            return eqx.tree_deserialise_leaves(weight_path, cortex)
        manager = ocp.CheckpointManager(os.path.dirname(weight_path.rstrip("/")))
        restored = manager.restore(step, args=ocp.args.StandardRestore({"model": cortex}))
        return restored["model"]


    @eqx.filter_jit
    def compute_warp_grid(cortex, res):
        """Precompute the static UV sampling grid + invalid mask for a model.

        The unwarp's UV grid depends only on trained model state (cone positions
        + RealNVP weights), never on pixel content -- so it can be evaluated
        once per model load and reused across every inference frame. This takes
        the RealNVP forward+inverse (~11 ms at res 512) out of the hot path.

        Returns (cached_uvs, invalid_regions, xy_grid, xy_full) where:
        - cached_uvs   has shape (1, 2, H', W')  -- UV sampling grid per
                       output pixel;
        - invalid_regions has shape (H', W') as an int mask of OOB samples;
        - xy_grid      has shape (H', W', 2)  -- retinotopic XY location of
                       each unwarped output pixel (column 0 = X, 1 = Y), in
                       [-1, 1]. Used for per-pixel eccentricity analysis;
        - xy_full      has shape (1, 2, H_c, W_c)  -- warped XY positions of
                       every cone cell from the RealNVP forward pass. Used
                       to build a cone-density map for eccentricity overlay.
        """
        xy_full = cortex.P_cell_position.get_XY_default_locations()
        grid = cortex.M_global_movement.generate_grid_fixed(
            xy_full[0, :, 0, 0],
            xy_full[0, :, -1, 0],
            xy_full[0, :, 0, -1],
            xy_full[0, :, -1, -1],
            res,
        )
        uvs = cortex.P_cell_position.get_UV_locations(
            jnp.transpose(grid, (2, 0, 1))[None, ...]
        )  # (1, 2, H', W')
        invalid = ((uvs <= -1) | (uvs >= 1)).sum(axis=1) > 0
        invalid = jnp.asarray(invalid[0]).astype(jnp.int32)
        return uvs, invalid, grid, xy_full


    def _grid_sample(img, grid_uv):
        """Bilinear resample of (B, C, H, W) at UV grid (B, 2, H', W') in [-1, 1].

        Inlined mirror of the grid sampler inside CortexModel.get_unwarped_percept
        so the cached grid can be reused without re-entering the RealNVP path.
        """
        from jax.scipy.ndimage import map_coordinates

        H, W = img.shape[2], img.shape[3]
        grid_perm = jnp.transpose(grid_uv, (0, 2, 3, 1))
        grid_pixel = (grid_perm + 1) * jnp.array([[[[(W - 1) / 2, (H - 1) / 2]]]])

        def sample_batch_item(img_single, grid_single):
            coords = jnp.stack([grid_single[:, :, 1], grid_single[:, :, 0]], axis=0)
            return jax.vmap(
                lambda ch: map_coordinates(ch, coords, order=1, mode='constant', cval=0)
            )(img_single)

        return jax.vmap(sample_batch_item)(img, grid_pixel)


    def make_infer_cached(cortex, cached_uvs):
        """Build a fused jitted inference function closing over `cached_uvs`.

        Closing over the precomputed UV grid (rather than passing it as a jitted
        argument) bakes it into the XLA program as a constant, so the RealNVP
        inverse never runs inside the timed region. The grid is identical for
        every frame of a fixed trained model -- computed once per model load.
        """
        _uvs_const = cached_uvs

        @eqx.filter_jit
        def _infer(retina, cortex, sRGB):
            linsRGB = retina.CST.sRGB_to_linsRGB(sRGB)
            lms = retina.CST.linsRGB_to_LMS(linsRGB)
            lms = jnp.transpose(lms, (0, 3, 1, 2))[:, None, ...]
            warped = retina.SpatialSampling(lms)
            pa = retina.SpectralSampling(warped, key=jax.random.PRNGKey(0))[:, 0, ...]
            ons = retina.SpikeConversion(retina.LateralInhibition(pa, key=jax.random.PRNGKey(0)))
            warped_ip = cortex.decode(ons)
            warped_linsRGB = jnp.transpose(cortex.ns_ip(warped_ip), (0, 2, 3, 1))
            warped_sRGB = jnp.clip(retina.CST.linsRGB_to_sRGB(warped_linsRGB), 0.0, 1.0)
            warped_chw = jnp.transpose(warped_sRGB, (0, 3, 1, 2))
            B = warped_chw.shape[0]
            uvs = jnp.broadcast_to(_uvs_const, (B, *_uvs_const.shape[1:]))
            unwarped = _grid_sample(warped_chw, uvs)
            unwarped = jnp.transpose(unwarped, (0, 2, 3, 1))
            unwarped = jnp.clip(unwarped, 0.0, 1.0)
            return unwarped

        return _infer


    def largest_valid_square(matrix):
        rows, cols = matrix.shape
        dp = np.zeros_like(matrix, dtype=int)
        max_size, br = 0, (0, 0)
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == 0:
                    dp[i, j] = 1 if (i == 0 or j == 0) else (
                        min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + 1
                    )
                    if dp[i, j] > max_size:
                        max_size, br = dp[i, j], (i, j)
        return (br[0] - max_size + 1, br[1] - max_size + 1, br[0], br[1])


    def load_test_images(res):
        imgs = []
        for fn in sorted(os.listdir(TEST_IMAGES_DIR)):
            if fn.endswith(".png"):
                imgs.append(
                    np.asarray(
                        Image.open(os.path.join(TEST_IMAGES_DIR, fn))
                        .convert("RGB")
                        .resize((res, res))
                    )
                    / 255.0
                )
        return np.asarray(imgs)


    def psnr(pred, target, max_val=1.0):
        a = np.asarray(pred, dtype=np.float64)
        b = np.asarray(target, dtype=np.float64)
        mse = float(np.mean((a - b) ** 2))
        if mse <= 0.0:
            return float("inf")
        return 10.0 * np.log10((max_val ** 2) / mse)


    def _srgb_to_linear(c):
        # Canonical sRGB EOTF (IEC 61966-2-1). Operating in linear radiance
        # matches the convention used by most demosaic/reconstruction papers:
        # errors in shadows are weighted by their true radiance, not the
        # gamma-compressed sRGB value, so the number reflects what the model
        # actually has to reconstruct rather than what the display shows.
        c = np.asarray(c, dtype=np.float64)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


    def psnr_linear(pred_srgb, target_srgb):
        """PSNR on linear radiance (sRGB EOTF applied to both sides).

        Both inputs are sRGB in [0, 1]. The decoded linear values for sRGB
        clipped at 1.0 are also in [0, 1], so the signal range stays 1.0.
        """
        a = _srgb_to_linear(pred_srgb)
        b = _srgb_to_linear(target_srgb)
        mse = float(np.mean((a - b) ** 2))
        if mse <= 0.0:
            return float("inf")
        return 10.0 * np.log10(1.0 / mse)


    _LPIPS = None
    _LPIPS_ERR = None


    def lpips_score(pred, target):
        """LPIPS distance (lower = more perceptually similar).

        Lazily builds an AlexNet-backed LPIPS model on CPU. Returns None if the
        dependency is unavailable or weight download fails.
        """
        global _LPIPS, _LPIPS_ERR
        if _LPIPS_ERR is not None:
            return None
        if _LPIPS is None:
            try:
                import torch
                import lpips

                _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
                for p in _LPIPS.parameters():
                    p.requires_grad_(False)
            except Exception as e:  # noqa: BLE001
                _LPIPS_ERR = str(e)
                return None
        import torch

        x = torch.from_numpy(np.asarray(pred)[None].transpose(0, 3, 1, 2).astype("float32")) * 2.0 - 1.0
        y = torch.from_numpy(np.asarray(target)[None].transpose(0, 3, 1, 2).astype("float32")) * 2.0 - 1.0
        with torch.no_grad():
            d = _LPIPS(x, y)
        return float(d.item())


    def ssim_score(pred, target):
        """SSIM index (higher = more structurally similar; 1.0 = identical).

        Standard Wang et al. SSIM over decoupled luminance/contrast/structure,
        averaged across channels via `channel_axis`. Returns None if
        scikit-image is unavailable.
        """
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
                a,
                b,
                data_range=1.0,
                channel_axis=-1,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        )


    def run_selection(config_value, weight_value):
        """Build models for `config_value`, load `weight_value`, run timed
        inference over all test images. Returns a result dict.
        """
        if not config_value or not weight_value:
            return None
        with open(config_value["path"], "r") as f:
            params = yaml.safe_load(f)
        step, weight_path = weight_value

        retina, cortex = build_models(params)
        cortex = load_weights(cortex, weight_path, step)
        res = retina.required_image_resolution
        originals = load_test_images(res)

        # Precompute the static UV unwarp grid ONCE: it depends only on trained
        # model state, never on pixel content. This skips the per-frame RealNVP
        # forward+inverse (~11 ms at res 512) and the per-frame crop-box search.
        cached_uvs, invalid_static, xy_grid, xy_full = compute_warp_grid(cortex, res)
        crop_box = largest_valid_square(np.asarray(invalid_static))
        infer_cached = make_infer_cached(cortex, cached_uvs)

        # Per-pixel retinotopic XY of each unwarped output pixel and the
        # cone-position scatter, used by the eccentricity-resolved error plot
        # at the end of the notebook. Computed once per model load alongside
        # the cached UV grid -- never on the timed frame path.
        xy_grid_np = np.asarray(xy_grid)  # (res, res, 2)  [x, y] in [-1, 1]
        cone_xy = np.asarray(xy_full[0]).reshape(2, -1).T  # (n_cones, 2)

        # Warm up the JIT cache so steady-state timing excludes compilation.
        warmup = infer_cached(retina, cortex, jnp.asarray(originals[:1]))
        jax.block_until_ready(warmup)

        # --- Timed inference loop (pure JAX, no torch/CPU work between iters) ---
        # Interleaving torch LPIPS calls between JAX iterations perturbs the MPS
        # dispatch queue and ~doubles per-frame time from ~9ms to ~20ms. So run
        # all inference first in a tight JAX-only loop, collect predictions, then
        # compute PSNR/LPIPS/SSIM on the cached crops afterward -- outside timing.
        run_times, preds = [], []
        y0, x0, y1, x1 = crop_box
        for i in range(len(originals)):
            t0 = time.perf_counter()
            unwarped = infer_cached(retina, cortex, jnp.asarray(originals[i : i + 1]))
            jax.block_until_ready(unwarped)
            run_times.append(time.perf_counter() - t0)
            unwarped = np.asarray(unwarped[0])
            preds.append(np.clip(unwarped[y0:y1, x0:x1], 0.0, 1.0))

        # --- Metrics (computed after timing, on cached crops) ---
        psnrs, psnrs_lin, lpipss, ssims = [], [], [], []
        for i, pred_crop in enumerate(preds):
            orig_crop = originals[i][y0:y1, x0:x1]
            psnrs.append(psnr(pred_crop, orig_crop))
            psnrs_lin.append(psnr_linear(pred_crop, orig_crop))
            lpipss.append(lpips_score(pred_crop, orig_crop))
            ssims.append(ssim_score(pred_crop, orig_crop))

        return {
            "name": config_value["name"],
            "step": step,
            "res": res,
            "preds": preds,
            "originals": originals,
            "crop_box": crop_box,
            "psnr": psnrs,
            "psnr_lin": psnrs_lin,
            "lpips": lpipss,
            "ssim": ssims,
            "total_s": float(np.sum(run_times)),
            "per_img_ms": float(np.mean(run_times) * 1000.0),
            "backend": jax.default_backend(),
            "xy_grid": xy_grid_np,
            "cone_xy": cone_xy,
        }

    return configs_with_weights, list_weights, run_selection


@app.cell
def _(configs_with_weights, mo):
    _rows = configs_with_weights()
    config_dropdown_a = mo.ui.dropdown(
        options={r["label"]: r for r in _rows} or {"(none)": None},
        label="Config A (experiment name)",
        searchable=True,
    )
    config_dropdown_a
    return (config_dropdown_a,)


@app.cell
def _(config_dropdown_a, list_weights, mo):
    name_a = config_dropdown_a.value["name"] if config_dropdown_a.value else None
    weights_a = list_weights(name_a) if name_a else []
    opts_a = {
        f"step {step}" + ("  (latest)" if i == len(weights_a) - 1 else ""): (step, path)
        for i, (step, path) in enumerate(weights_a)
    }
    weight_dropdown_a = mo.ui.dropdown(
        options=opts_a or {"(none)": None},
        label="Weights A (step)",
        searchable=True,
        value=list(opts_a.keys())[-1] if opts_a else None,
    )
    weight_dropdown_a
    return (weight_dropdown_a,)


@app.cell
def _(configs_with_weights, mo):
    _rows = configs_with_weights()
    config_dropdown_b = mo.ui.dropdown(
        options={r["label"]: r for r in _rows} or {"(none)": None},
        label="Config B (experiment name)",
        searchable=True,
    )
    config_dropdown_b
    return (config_dropdown_b,)


@app.cell
def _(config_dropdown_b, list_weights, mo):
    name_b = config_dropdown_b.value["name"] if config_dropdown_b.value else None
    weights_b = list_weights(name_b) if name_b else []
    opts_b = {
        f"step {step}" + ("  (latest)" if i == len(weights_b) - 1 else ""): (step, path)
        for i, (step, path) in enumerate(weights_b)
    }
    weight_dropdown_b = mo.ui.dropdown(
        options=opts_b or {"(none)": None},
        label="Weights B (step)",
        searchable=True,
        value=list(opts_b.keys())[-1] if opts_b else None,
    )
    weight_dropdown_b
    return (weight_dropdown_b,)


@app.cell
def _(config_dropdown_a, mo, run_selection, weight_dropdown_a):
    result_a = None
    _msg_a = mo.md("")
    if config_dropdown_a.value and weight_dropdown_a.value:
        try:
            result_a = run_selection(config_dropdown_a.value, weight_dropdown_a.value)
        except Exception as e:  # noqa: BLE001
            _msg_a = mo.md(f"**Run A failed:** `{e}`").callout(kind="danger")
    else:
        _msg_a = mo.md("Pick a config + weights for A.")
    return (result_a,)


@app.cell
def _(config_dropdown_b, mo, run_selection, weight_dropdown_b):
    result_b = None
    _msg_b = mo.md("")
    if config_dropdown_b.value and weight_dropdown_b.value:
        try:
            result_b = run_selection(config_dropdown_b.value, weight_dropdown_b.value)
        except Exception as e:  # noqa: BLE001
            _msg_b = mo.md(f"**Run B failed:** `{e}`").callout(kind="danger")
    else:
        _msg_b = mo.md("Pick a config + weights for B.")
    return (result_b,)


@app.cell
def _(mo, np, plt, result_a, result_b):
    def _fmt_lpips_mean(xs):
        vals = [v for v in (xs or []) if v is not None]
        if not vals:
            return "n/a"
        return f"{float(np.mean(vals)):.4f}"

    def _fmt_psnr_mean(xs):
        vals = [v for v in (xs or []) if np.isfinite(v)]
        if not vals:
            return "n/a"
        return f"{float(np.mean(vals)):.2f} dB"

    def _fmt_ssim_mean(xs):
        vals = [v for v in (xs or []) if v is not None]
        if not vals:
            return "n/a"
        return f"{float(np.mean(vals)):.4f}"

    def _per_row(r):
        return {
            "config": r["name"] if r else "—",
            "step": r["step"] if r else "—",
            "PSNR (mean)": _fmt_psnr_mean(r["psnr"]) if r else "—",
            "PSNR-lin (mean)": _fmt_psnr_mean(r["psnr_lin"]) if r else "—",
            "SSIM (mean)": _fmt_ssim_mean(r["ssim"]) if r else "—",
            "LPIPS (mean)": _fmt_lpips_mean(r["lpips"]) if r else "—",
            "wall (s)": f"{r['total_s']:.3f}" if r else "—",
            "per image (ms)": f"{r['per_img_ms']:.2f}" if r else "—",
            "backend": r["backend"] if r else "—",
        }

    _rows = [_per_row(result_a), _per_row(result_b)]
    table = mo.ui.table(_rows, label="Summary")

    diff_psnr = None
    diff_psnr_lin = None
    diff_lpips = None
    diff_ssim = None
    if result_a and result_b:
        pa = [v for v in result_a["psnr"] if np.isfinite(v)]
        pb = [v for v in result_b["psnr"] if np.isfinite(v)]
        if pa and pb:
            diff_psnr = float(np.mean(pa)) - float(np.mean(pb))
        la_lin = [v for v in result_a["psnr_lin"] if np.isfinite(v)]
        lb_lin = [v for v in result_b["psnr_lin"] if np.isfinite(v)]
        if la_lin and lb_lin:
            diff_psnr_lin = float(np.mean(la_lin)) - float(np.mean(lb_lin))
        la = [v for v in result_a["lpips"] if v is not None]
        lb = [v for v in result_b["lpips"] if v is not None]
        if la and lb:
            diff_lpips = float(np.mean(la)) - float(np.mean(lb))
        sa = [v for v in result_a["ssim"] if v is not None]
        sb = [v for v in result_b["ssim"] if v is not None]
        if sa and sb:
            diff_ssim = float(np.mean(sa)) - float(np.mean(sb))

    deltas = mo.md(
        (f"**\u0394 PSNR (A - B):** `{diff_psnr:+.2f} dB`  "
         f"| higher is better for A") if diff_psnr is not None else ""
    ).callout(
        kind="neutral"
    ) if diff_psnr is not None else mo.md("")
    deltas_psnr_lin = mo.md(
        (f"**\u0394 PSNR-lin (A - B):** `{diff_psnr_lin:+.2f} dB`  "
         f"(linear radiance | higher is better for A)") if diff_psnr_lin is not None else ""
    ).callout(
        kind="neutral"
    ) if diff_psnr_lin is not None else mo.md("")
    deltas_ssim = mo.md(
        (f"**\u0394 SSIM (A - B):** `{diff_ssim:+.4f}`  "
         f"| higher is better for A") if diff_ssim is not None else ""
    ).callout(
        kind="neutral"
    ) if diff_ssim is not None else mo.md("")
    deltas_lpips = mo.md(
        (f"**\u0394 LPIPS (A - B):** `{diff_lpips:+.4f}`  "
         f"| lower is better for A") if diff_lpips is not None else ""
    ).callout(
        kind="neutral"
    ) if diff_lpips is not None else mo.md("")

    # Side-by-side comparison: Original | A | B per image, cropped to the
    # shared valid square. Three columns at a moderate DPI for legibility.
    if result_a is None and result_b is None:
        fig_view = mo.md(
            "Select configs + weights for both A and B to run inference."
        ).callout(kind="info")
    else:
        a = result_a or result_b
        originals = a["originals"]
        n = len(originals)
        preds_a = result_a["preds"] if result_a else [None] * n
        preds_b = result_b["preds"] if result_b else [None] * n
        has_a = result_a is not None
        has_b = result_b is not None

        def _metric_str(r, i):
            p = r['psnr'][i]
            p_lin = r['psnr_lin'][i] if i < len(r['psnr_lin']) else None
            s = r['ssim'][i] if i < len(r['ssim']) else None
            l = r['lpips'][i] if i < len(r['lpips']) else None
            p_str = f"{p:.1f}dB" if np.isfinite(p) else "—"
            pl_str = (
                f"lin{p_lin:.1f}" if (p_lin is not None and np.isfinite(p_lin)) else ""
            )
            s_str = f"S{s:.3f}" if s is not None else ""
            l_str = f"L{l:.3f}" if l is not None else ""
            parts = [x for x in (p_str, pl_str, s_str, l_str) if x]
            return " ".join(parts)

        fig, axes = plt.subplots(n, 3, figsize=(7.2, 1.5 * n), dpi=110)
        if n == 1:
            axes = axes[None, :]
        for i in range(n):
            y0, x0, y1, x1 = (result_a or result_b)["crop_box"]
            orig_crop = np.clip(originals[i][y0:y1, x0:x1], 0.0, 1.0)

            axes[i, 0].imshow(orig_crop)
            axes[i, 0].set_title("Original", fontsize=8)
            axes[i, 0].axis("off")

            if has_a:
                axes[i, 1].imshow(np.clip(preds_a[i], 0.0, 1.0))
                axes[i, 1].set_title(f"A: {result_a['name']}\n{_metric_str(result_a, i)}", fontsize=7)
            else:
                axes[i, 1].set_title("A: (none)", fontsize=7)
            axes[i, 1].axis("off")

            if has_b:
                axes[i, 2].imshow(np.clip(preds_b[i], 0.0, 1.0))
                axes[i, 2].set_title(f"B: {result_b['name']}\n{_metric_str(result_b, i)}", fontsize=7)
            else:
                axes[i, 2].set_title("B: (none)", fontsize=7)
            axes[i, 2].axis("off")
        fig.tight_layout(pad=0.3)
        fig_view = fig

    mo.vstack([
        mo.md("### Summary"),
        table,
        deltas,
        deltas_psnr_lin,
        deltas_ssim,
        deltas_lpips,
        mo.md("### Side-by-side (cropped to valid region)"),
        fig_view,
    ])
    return


@app.cell
def _(mo, np, plt, result_a, result_b):
    """Eccentricity-resolved reconstruction error (confound-controlled).

    An earlier version of this plot was confounded in three ways flagged in
    review, all corrected here:

      1. Raw linear-radiance MSE rewards a smooth/dark periphery for being
         low-detail, not for being well reconstructed. We now also report
         error normalized by local input contrast (gradient magnitude of the
         target's linear luminance) so the question becomes "for matched
         detail, does the periphery reconstruct worse?". Both the raw and
         normalized curves are shown relative to their foveal value, so their
         *shapes* can be compared directly.
      2. The companion sampling curve was "cones per bin" = density x annulus
         area, which is non-monotonic (it peaks at mid-eccentricity from
         geometry alone). We now plot local cone *density* (cones per unit
         retinotopic area), which falls monotonically from the fovea -- the
         variable the sampling-bound argument actually needs.
      3. The outer bins averaged in vignetted / zero-density padding. Pixels
         and bins beyond cone coverage (99.5th-percentile cone eccentricity)
         are masked out of every curve and greyed out in the error map.

    Reading the corrected panel 3:
      * If the bold (contrast-normalized) curve climbs with eccentricity, the
        periphery genuinely reconstructs *matched* detail worse -> a real
        sampling-bound headroom that saccades could cash in, with the local
        density curve as a rough ceiling.
      * If the bold curve stays flat while the faint (raw) curve falls, the
        apparent peripheral "win" was just smoother content -> the
        edge-everywhere error in panel 1 is decoder/re-encoder-bound and a
        saccade policy optimizes around a wall.

    Caveat: this is still a population plot over whatever content happens to
    sit at each eccentricity. It cannot fully separate "the periphery is bad"
    from "the periphery of these images is smooth". The decisive test is to
    hold a patch of content fixed and move the eye -- reconstruct the same
    patch foveated vs at several eccentricities; that foveal-minus-peripheral
    gap on identical content is the per-saccade headroom an RL policy would
    cash in. That experiment needs fixation control in the inference path and
    is left as the recommended follow-up.
    """
    r = result_a if result_a is not None else result_b
    if r is None:
        ecc_view = mo.md(
            "Select a config + weights for A (or B) to view the eccentricity analysis."
        ).callout(kind="info")
    else:
        _xy_grid = r["xy_grid"]   # (res, res, 2) per-pixel retinotopic XY in [-1, 1]
        _cone_xy = r["cone_xy"]   # (n_cones, 2)
        _y0, _x0, _y1, _x1 = r["crop_box"]
        _preds = r["preds"]
        _originals = r["originals"]
        _n = len(_preds)

        # Per-output-pixel eccentricity restricted to the cropped valid region.
        _e_crop = _xy_grid[_y0:_y1, _x0:_x1, :]               # (ch, cw, 2)
        _ecc = np.sqrt(_e_crop[..., 0] ** 2 + _e_crop[..., 1] ** 2)
        _cone_ecc = np.sqrt(_cone_xy[:, 0] ** 2 + _cone_xy[:, 1] ** 2)

        # Eccentricity beyond which the retina has essentially no cones. Past
        # this radius the output is vignetted padding and must not enter the
        # error statistics (fix #3).
        _ecc_outer = float(np.percentile(_cone_ecc, 99.5))
        _e_max = _ecc_outer * 1.02

        _n_bins = 16
        _bins = np.linspace(0.0, _e_max, _n_bins + 1)
        _centers = 0.5 * (_bins[:-1] + _bins[1:])
        _cone_count_per_bin, _ = np.histogram(_cone_ecc, bins=_bins)

        # Local cone DENSITY (cones per unit retinotopic area). Unlike
        # cones-per-bin = density x annulus-area, this is monotonic in
        # eccentricity, so it can actually back the sampling-bound read (fix #2).
        _annulus_area = np.pi * (_bins[1:] ** 2 - _bins[:-1] ** 2)
        _cone_density_per_bin = _cone_count_per_bin / np.maximum(_annulus_area, 1e-12)

        # Bins with no sampled cones are padding -> dropped from every curve.
        _bin_valid = _cone_count_per_bin > 0

        # Static cone-density map: log 2D histogram over the [-1, 1] retinotopic
        # square. The fovea is the dense cluster at the origin; the periphery
        # thins out, mirroring the local-density curve in panel 3.
        _cone_hist, _, _ = np.histogram2d(
            _cone_xy[:, 0], _cone_xy[:, 1], bins=240, range=[[-1, 1], [-1, 1]]
        )
        _theta = np.linspace(0.0, 2.0 * np.pi, 200)

        def _srgb_to_linear_arr(c):
            c = np.asarray(c, dtype=np.float64)
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

        _err_cmap = plt.cm.inferno.copy()
        _err_cmap.set_bad("0.15")

        _fig, _axes = plt.subplots(_n, 3, figsize=(14, 3.0 * _n), dpi=110)
        if _n == 1:
            _axes = _axes[None, :]

        for _i in range(_n):
            _pred = np.clip(_preds[_i], 0.0, 1.0)
            _orig = np.clip(_originals[_i][_y0:_y1, _x0:_x1], 0.0, 1.0)
            _lin_pred = _srgb_to_linear_arr(_pred)
            _lin_orig = _srgb_to_linear_arr(_orig)
            _err_map = np.mean((_lin_pred - _lin_orig) ** 2, axis=-1)

            # Local input contrast = gradient magnitude of the target's linear
            # luminance. Normalizing error by this controls for "the periphery
            # is just smoother", asking instead about matched detail (fix #1).
            _lum = _lin_orig.mean(axis=-1)
            _gy, _gx = np.gradient(_lum)
            _contrast_map = np.sqrt(_gx ** 2 + _gy ** 2)

            # Drop padding pixels (beyond cone coverage) from the statistics.
            _pix_valid = _ecc <= _ecc_outer
            _flat_ecc = _ecc[_pix_valid]
            _flat_err = _err_map[_pix_valid]
            _flat_con = _contrast_map[_pix_valid]

            _err_sum, _ = np.histogram(_flat_ecc, bins=_bins, weights=_flat_err)
            _con_sum, _ = np.histogram(_flat_ecc, bins=_bins, weights=_flat_con)
            _cnt, _ = np.histogram(_flat_ecc, bins=_bins)

            _raw_err = np.where(_cnt > 0, _err_sum / np.maximum(_cnt, 1), np.nan)
            # Error per unit local contrast: total error / total contrast in bin.
            _norm_err = np.where(
                _con_sum > 1e-9, _err_sum / np.maximum(_con_sum, 1e-9), np.nan
            )
            _raw_err = np.where(_bin_valid, _raw_err, np.nan)
            _norm_err = np.where(_bin_valid, _norm_err, np.nan)
            _dens = np.where(_bin_valid, _cone_density_per_bin, np.nan)

            # Express both error curves relative to their foveal (innermost
            # valid) value so their shapes are directly comparable.
            _ok = np.where(_bin_valid & np.isfinite(_raw_err) & np.isfinite(_norm_err))[0]
            if len(_ok) and _raw_err[_ok[0]] > 0 and _norm_err[_ok[0]] > 0:
                _f = _ok[0]
                _raw_rel = _raw_err / _raw_err[_f]
                _norm_rel = _norm_err / _norm_err[_f]
            else:
                _raw_rel = _raw_err
                _norm_rel = _norm_err

            # --- Panel 1: error heatmap (padding greyed out) ---
            _ax_err = _axes[_i, 0]
            _err_disp = np.where(_pix_valid, _err_map, np.nan)
            _err_p95 = max(1e-6, float(np.percentile(_err_map[_pix_valid], 95)))
            _im_err = _ax_err.imshow(_err_disp, cmap=_err_cmap, vmin=0.0, vmax=_err_p95)
            _ax_err.set_title(f"Error  |  {r['name']}  img {_i}", fontsize=8)
            _ax_err.axis("off")
            _fig.colorbar(_im_err, ax=_ax_err, fraction=0.046, pad=0.02, label="lin. MSE")

            # --- Panel 2: cone-density map + fixation + coverage ring ---
            _ax_den = _axes[_i, 1]
            _im_den = _ax_den.imshow(
                np.log1p(_cone_hist.T),
                extent=[-1, 1, -1, 1],
                origin="lower",
                cmap="viridis",
                aspect="auto",
            )
            _ax_den.plot(0.0, 0.0, "r+", markersize=12, mew=2, label="fixation")
            _ax_den.plot(
                _ecc_outer * np.cos(_theta), _ecc_outer * np.sin(_theta),
                "r--", lw=1.0, label="cone coverage",
            )
            _ax_den.set_xlim(-1, 1)
            _ax_den.set_ylim(-1, 1)
            _ax_den.set_title("Cone density (log) + coverage ring", fontsize=8)
            _ax_den.axis("off")
            _ax_den.legend(fontsize=6, loc="upper right")
            _fig.colorbar(_im_den, ax=_ax_den, fraction=0.046, pad=0.02, label="log(1+count)")

            # --- Panel 3: relative error curves vs local cone density ---
            _ax_ee = _axes[_i, 2]
            _ln_raw = _ax_ee.plot(
                _centers, _raw_rel, "o--", color="crimson", alpha=0.45,
                label="Raw MSE (rel. fovea)",
            )
            _ln_norm = _ax_ee.plot(
                _centers, _norm_rel, "o-", color="crimson",
                label="Contrast-norm. err (rel. fovea)",
            )
            _ax_ee.axhline(1.0, color="0.6", lw=0.8, ls=":")
            _ax_ee.set_xlabel("Eccentricity (normalized retinotopic units)", fontsize=8)
            _ax_ee.set_ylabel("Error relative to fovea (×)", color="crimson", fontsize=8)
            _ax_ee.tick_params(axis="y", labelcolor="crimson")
            _ax_ee.set_xlim(0.0, _e_max)
            _ax_ee.set_ylim(bottom=0.0)

            _ax_cc = _ax_ee.twinx()
            _ln_cc = _ax_cc.plot(
                _centers, _dens, "s--", color="steelblue", alpha=0.7,
                label="Local cone density",
            )
            _ax_cc.set_ylabel("Cones per unit area", color="steelblue", fontsize=8)
            _ax_cc.tick_params(axis="y", labelcolor="steelblue")
            _ax_cc.set_ylim(bottom=0.0)

            _lns = _ln_raw + _ln_norm + _ln_cc
            _ax_ee.legend(_lns, [_l.get_label() for _l in _lns], fontsize=6, loc="upper left")
            _ax_ee.set_title(
                "Normalized error vs local density  "
                "(rising norm. err ⇒ true peripheral loss; flat ⇒ content artifact)",
                fontsize=6.5,
            )

        _fig.suptitle(
            f"Eccentricity-resolved error (confound-controlled)  —  {r['name']}  (step {r['step']})",
            fontsize=10,
        )
        _fig.tight_layout(pad=0.6, rect=[0, 0, 1, 0.98])
        ecc_view = _fig

    mo.vstack([
        mo.md("### Eccentricity-resolved reconstruction error (confound-controlled)"),
        mo.md(
            "Raw MSE (faint, dashed) vs **contrast-normalized** error (bold), both "
            "relative to the fovea, plotted against **local cone density** "
            "(monotonic). If the bold curve rises with eccentricity the periphery "
            "genuinely reconstructs matched detail worse (sampling-bound headroom); "
            "if it stays flat while the faint curve falls, the apparent peripheral "
            "gain was just smoother content (decoder-bound). Padding beyond the "
            "cone-coverage ring is masked out of every curve. The decisive "
            "content-fixed eye-movement test is described in the cell docstring."
        ).callout(kind="info"),
        ecc_view,
    ])
    return


if __name__ == "__main__":
    app.run()
