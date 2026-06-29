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
    import optax
    import matplotlib.pyplot as plt

    import marimo as mo

    # Import the jit'd training step + helpers from train.py. train.py only
    # defines helpers + the loop driver at top level (the main block is
    # guarded), so importing it has no side effects.
    import train as T
    from train import OptimizerStates, create_optimizers, prepare_batch, train_step

    from Simulated.Retina import RetinaModel
    from Simulated.Cortex import CortexModel

    ROOT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    CONFIG_DIR = os.path.join(ROOT, "Experiment", "Config", "Default")

    mo.md("# Matisse — Per-Step Train-Time Benchmark")
    return (
        CONFIG_DIR,
        CortexModel,
        OptimizerStates,
        RetinaModel,
        T,
        create_optimizers,
        eqx,
        glob,
        jax,
        jnp,
        mo,
        np,
        optax,
        os,
        plt,
        prepare_batch,
        re,
        time,
        train_step,
        yaml,
    )


# ---- Helpers: build models, build optimizers, representative batch, bench -----
@app.cell
def _(CONFIG_DIR, CortexModel, OptimizerStates, RetinaModel, ROOT, create_optimizers, eqx, glob, jax, jnp, np, os, yaml):
    def configs():
        paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
        rows = []
        for path in paths:
            with open(path, "r") as f:
                params = yaml.safe_load(f)
            name = params.get("Experiment", {}).get("name")
            if name:
                rows.append({"label": name, "path": path, "name": name})
        return rows


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

        cc = params.get("CorticalModel", params.get("CortexModel", {}))
        dm = cc.get("cortex_learn_demosaicing", {})
        cortex = CortexModel(
            latent_dim=cc.get("latent_dim", 8),
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
            temporal_fusion=cc.get("temporal_fusion", "none"),
            key=jax.random.PRNGKey(42),
        )
        return retina, cortex


    def build_optimizers(cortex, learning_rate):
        """Three optimizers + the OptimizerStates container, mirroring train.py."""
        main_opt, ns_cm_opt, ns_ip_opt = create_optimizers(
            learning_rate, lr_schedule="constant"
        )

        main_filter = jax.tree.map(lambda _: False, cortex)
        main_filter = eqx.tree_at(
            lambda m: (
                m.C_cone_spectral_type,
                m.D_demosaicing,
                m.W_lateral_inhibition_weights,
                m.P_cell_position,
            ),
            main_filter,
            replace=(True, True, True, True),
        )
        cm_filter = eqx.tree_at(
            lambda m: m.ns_cm, jax.tree.map(lambda _: False, cortex), replace=True
        )
        ip_filter = eqx.tree_at(
            lambda m: m.ns_ip, jax.tree.map(lambda _: False, cortex), replace=True
        )
        opt_states = OptimizerStates(
            main_opt.init(eqx.filter(cortex, main_filter)),
            ns_cm_opt.init(eqx.filter(cortex, cm_filter)),
            ns_ip_opt.init(eqx.filter(cortex, ip_filter)),
        )
        return (main_opt, ns_cm_opt, ns_ip_opt), opt_states


    def representative_batch(params, retina):
        """Synthesize an LMS full-field batch with the same shape the NTIRE
        loader yields. Random content is fine: train_step / prepare_batch are
        shape-specialized by XLA, so timing depends only on shape + dtype, not
        pixel values.
        """
        bsz = params["Dataset"]["batch_size"]
        timesteps = params["Experiment"]["timesteps_per_image"]
        max_shift = params["RetinaModel"]["max_shift_size"]
        dim_image = retina.required_image_resolution + (timesteps - 1) * 2 * max_shift
        rng = np.random.default_rng(0)
        # 4 channels = LMS + (zero-pad stub), matching the dataset's layout for
        # non-tetrachromat runs.
        field = rng.uniform(0.0, 1.0, size=(bsz, dim_image, dim_image, 4)).astype("float32")
        return jnp.asarray(field)


    def bench_config(params, n_iter, n_warmup, full_update):
        """Build models + optimizers, warm up both kernels, then time n_iter
        consecutive train_step calls. Returns a result dict.

        Each timed call wraps the cortex gradient update with all submodels
        active (or all but ns_cm/ns_ip if `full_update` is False) so configs are
        compared apples-to-apples regardless of their per-cycle update settings.
        """
        retina, cortex = build_models(params)
        training_cfg = params.get("Training", {})
        optimizers, opt_states = build_optimizers(
            cortex, training_cfg.get("learning_rate", 1e-3)
        )
        batch = representative_batch(params, retina)
        simulating_tetra = params["Experiment"]["simulating_tetra"]
        cone_mosaic = retina.SpectralSampling.get_cone_mosaic()
        kernel_size = retina.LateralInhibition.get_kernel_size()
        ns_ip_loss_kind = training_cfg.get("ns_ip_loss", "l2")
        ons_loss_kind = training_cfg.get("ons_loss", "l2")
        ons_huber_delta = training_cfg.get("ons_huber_delta", 0.01)

        key = jax.random.PRNGKey(42)

        # Warm up prepare_batch + train_step until both are compiled and steady.
        for _ in range(n_warmup):
            key, subkey = jax.random.split(key)
            ons1, ons2, linsRGB1, true_dxy = prepare_batch(retina, batch, subkey)
            cortex, opt_states, losses = train_step(
                cortex, opt_states, optimizers,
                ons1, ons2, linsRGB1, true_dxy,
                cone_mosaic, kernel_size, simulating_tetra,
                ns_ip_loss_kind=ns_ip_loss_kind,
                update_cell_position=full_update,
                update_neural_scope=full_update,
                ons_loss_kind=ons_loss_kind,
                ons_huber_delta=ons_huber_delta,
            )
        jax.block_until_ready(losses["total"])

        # Timed region: each iteration runs prepare_batch (cheap, jitted) then
        # the gradient step. block_until_ready on the loss forces a device
        # sync so perf_counter measures true device wall time per step.
        step_times = []
        prep_times = []
        for _ in range(n_iter):
            key, subkey = jax.random.split(key)
            t0 = time.perf_counter()
            ons1, ons2, linsRGB1, true_dxy = prepare_batch(retina, batch, subkey)
            jax.block_until_ready(ons1)
            prep_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            cortex, opt_states, losses = train_step(
                cortex, opt_states, optimizers,
                ons1, ons2, linsRGB1, true_dxy,
                cone_mosaic, kernel_size, simulating_tetra,
                ns_ip_loss_kind=ns_ip_loss_kind,
                update_cell_position=full_update,
                update_neural_scope=full_update,
                ons_loss_kind=ons_loss_kind,
                ons_huber_delta=ons_huber_delta,
            )
            jax.block_until_ready(losses["total"])
            step_times.append(time.perf_counter() - t0)

        import numpy as _np

        return {
            "name": params["Experiment"]["name"],
            "n_iter": n_iter,
            "res": retina.required_image_resolution,
            "batch_size": params["Dataset"]["batch_size"],
            "demosaicing_type": params.get("CorticalModel", {})
            .get("cortex_learn_demosaicing", {})
            .get("type", "Default"),
            "step_times_ms": _np.asarray(step_times) * 1000.0,
            "prep_times_ms": _np.asarray(prep_times) * 1000.0,
            "mean_ms": float(_np.mean(step_times) * 1000.0),
            "median_ms": float(_np.median(step_times) * 1000.0),
            "min_ms": float(_np.min(step_times) * 1000.0),
            "max_ms": float(_np.max(step_times) * 1000.0),
            "std_ms": float(_np.std(step_times) * 1000.0),
            "prep_mean_ms": float(_np.mean(prep_times) * 1000.0),
            "steps_per_sec": float(1.0 / _np.mean(step_times)) if _np.mean(step_times) > 0 else 0.0,
            "backend": jax.default_backend(),
            "full_update": full_update,
        }
    return (
        bench_config,
        build_models,
        build_optimizers,
        configs,
        representative_batch,
    )


# ---- UI: config A and config B ----------------------------------------------
@app.cell
def _(configs, mo):
    _rows = configs()
    opts = {r["label"]: r for r in _rows}
    config_a = mo.ui.dropdown(options=opts, label="Config A", searchable=True)
    config_b = mo.ui.dropdown(options=opts, label="Config B", searchable=True)
    return (config_a, config_b)


@app.cell
def _(config_a, config_b, mo):
    n_iter = mo.ui.number(start=1, stop=200, step=1, value=10, label="Timed steps (post-JIT)")
    n_warmup = mo.ui.number(start=1, stop=20, step=1, value=3, label="Warmup steps")
    full_update = mo.ui.checkbox(label="All submodels updating (worst case per step)", value=True)

    mo.vstack([
        mo.md("### Benchmark settings"),
        config_a,
        config_b,
        n_iter,
        n_warmup,
        full_update,
    ])
    return (full_update, n_iter, n_warmup)


@app.cell
def _(bench_config, config_a, config_b, full_update, mo, n_iter, n_warmup, yaml):
    result_a = None
    result_b = None
    err_a = None
    err_b = None

    def _run(config_value):
        if not config_value:
            return None, None
        import yaml as _yaml

        with open(config_value["path"], "r") as f:
            params = _yaml.safe_load(f)
        try:
            return bench_config(
                params, int(n_iter.value), int(n_warmup.value), bool(full_update.value)
            ), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    result_a, err_a = _run(config_a.value)
    result_b, err_b = _run(config_b.value)

    msgs = []
    if err_a:
        msgs.append(mo.md(f"**Run A failed:** `{err_a}`").callout(kind="danger"))
    if err_b:
        msgs.append(mo.md(f"**Run B failed:** `{err_b}`").callout(kind="danger"))
    errors = mo.vstack(msgs) if msgs else mo.md("")
    return (result_a, result_b)


# ---- Comparison view --------------------------------------------------------
@app.cell
def _(mo, np, plt, result_a, result_b):
    def _row(r):
        if r is None:
            return {k: "—" for k in [
                "config", "demosaicing", "res", "bs", "mean (ms)", "median (ms)",
                "min (ms)", "std (ms)", "prep (ms)", "steps/sec", "backend",
            ]}
        return {
            "config": r["name"],
            "demosaicing": r["demosaicing_type"],
            "res": r["res"],
            "bs": r["batch_size"],
            "mean (ms)": f"{r['mean_ms']:.2f}",
            "median (ms)": f"{r['median_ms']:.2f}",
            "min (ms)": f"{r['min_ms']:.2f}",
            "std (ms)": f"{r['std_ms']:.2f}",
            "prep (ms)": f"{r['prep_mean_ms']:.2f}",
            "steps/sec": f"{r['steps_per_sec']:.2f}",
            "backend": r["backend"],
        }

    table_rows = [_row(result_a), _row(result_b)]
    table = mo.ui.table(table_rows, label="Per-step timing")

    # Bar chart of mean train_step time. Small + low DPI to stay well under the
    # marimo output cap.
    have_a = result_a is not None
    have_b = result_b is not None
    if have_a or have_b:
        names = [
            r["name"] if r else "A" for r in (result_a, result_b) if r is not None
        ]
        means = [
            r["mean_ms"] for r in (result_a, result_b) if r is not None
        ]
        meds = [
            r["median_ms"] for r in (result_a, result_b) if r is not None
        ]
        fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=90)
        x = np.arange(len(names))
        ax.bar(x - 0.15, means, width=0.3, label="mean")
        ax.bar(x + 0.15, meds, width=0.3, label="median")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("train_step (ms)")
        ax.set_title("Cortex gradient step (post-JIT)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout(pad=0.3)
        chart = fig
    else:
        chart = mo.md("Pick two configs to benchmark.")

    speedup = None
    delta_ms = None
    if have_a and have_b:
        delta_ms = result_a["mean_ms"] - result_b["mean_ms"]
        if result_b["mean_ms"] > 0:
            speedup = result_a["mean_ms"] / result_b["mean_ms"]

    delta_callout = mo.md(
        f"**A - B (mean):** `{delta_ms:+.2f} ms`"
        + (f"  |  speedup A/B: `{speedup:.2f}x`" if speedup is not None else "")
    ).callout(kind="neutral") if delta_ms is not None else mo.md("")

    mo.vstack([
        mo.md("### Summary"),
        table,
        delta_callout,
        mo.md("### Mean / median train_step time"),
        chart,
        mo.md(
            "_Timed region is the cortex gradient step (`train_step`); the "
            "retina forward (`prepare_batch`) is reported separately as `prep "
            "(ms)`. Random LMS batches — shapes match the real loader so XLA "
            "specializes identically._"
        ),
    ])
    return


if __name__ == "__main__":
    app.run()