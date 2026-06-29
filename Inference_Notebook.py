import marimo

__generated_with = "0.19.4"
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
        RetinaModel,
        ROOT,
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


# ---- Helpers: build models, list weights, run inference, metrics ------------
@app.cell
def _(
    CONFIG_DIR,
    CortexModel,
    RetinaModel,
    ROOT,
    TEST_IMAGES_DIR,
    WEIGHTS_DIR,
    eqx,
    glob,
    jax,
    jnp,
    os,
    re,
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
    def infer(retina, cortex, sRGB):
        res = retina.required_image_resolution
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
        unwarped, invalid = cortex.get_unwarped_percept(
            warped_chw, required_image_resolution=res
        )
        return unwarped, invalid


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

        # Warm up the JIT cache so steady-state timing excludes compilation.
        warmup, inv0 = infer(retina, cortex, jnp.asarray(originals[:1]))
        jax.block_until_ready(warmup)

        run_times, preds, psnrs, psnrs_lin, lpipss, ssims = [], [], [], [], [], []
        crop_box = None
        for i in range(len(originals)):
            t0 = time.perf_counter()
            unwarped, invalid = infer(retina, cortex, jnp.asarray(originals[i : i + 1]))
            jax.block_until_ready(unwarped)
            run_times.append(time.perf_counter() - t0)
            unwarped = np.asarray(unwarped[0])
            if crop_box is None:
                crop_box = largest_valid_square(np.asarray(invalid))
            y0, x0, y1, x1 = crop_box
            pred_crop = np.clip(unwarped[y0:y1, x0:x1], 0.0, 1.0)
            orig_crop = originals[i][y0:y1, x0:x1]
            preds.append(pred_crop)
            psnrs.append(psnr(pred_crop, orig_crop))
            psnrs_lin.append(psnr_linear(pred_crop, orig_crop))
            lpipss.append(lpips_score(pred_crop, orig_crop))
            ssims.append(ssim_score(pred_crop, orig_crop))

        import time as _time  # already imported

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
        }
    return (
        build_models,
        configs_with_weights,
        experiment_name,
        infer,
        largest_valid_square,
        list_weights,
        load_test_images,
        lpips_score,
        psnr,
        psnr_linear,
        run_selection,
        ssim_score,
    )


# ---- UI: config A -----------------------------------------------------------
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


# ---- UI: weights A ----------------------------------------------------------
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


# ---- UI: config B -----------------------------------------------------------
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


# ---- UI: weights B ----------------------------------------------------------
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


# ---- Run A ------------------------------------------------------------------
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


# ---- Run B ------------------------------------------------------------------
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


# ---- Comparison view --------------------------------------------------------
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


if __name__ == "__main__":
    app.run()