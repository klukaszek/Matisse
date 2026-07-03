"JAX/Equinox/Optax training loop for Matisse cortical model."
import os
import json

# Disable JAX GPU memory preallocation - must be set before importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import orbax.checkpoint as ocp
import time
from tqdm.auto import tqdm
from typing import Tuple, Dict, Any, NamedTuple
import resource

# Increase file descriptor limit for Grain multi-processing
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
except Exception as e:
    print(f"Warning: Could not increase file descriptor limit: {e}")

from Simulated.Retina import RetinaModel
from Simulated.Cortex import CortexModel
from Dataset import create_dataset
from Dataset.NTIRE import create_dataloader, DeviceResidentCropLoader
from Experiment.ProgressLogger import LocalProgressLogger


# Define a container for the three optimizer states
class OptimizerStates(NamedTuple):
    main: optax.OptState
    ns_cm: optax.OptState
    ns_ip: optax.OptState


def tree_l2_norm(tree) -> jax.Array:
    """Compute an L2 norm over all array leaves in a parameter subtree."""
    leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


def ons_reconstruction_penalty(
    residual: jax.Array,
    loss_kind: str = 'l2',
    huber_delta: float = 0.01,
) -> jax.Array:
    """Elementwise ONS penalty, preserving L2 scale near zero."""
    if loss_kind == 'l2':
        return residual ** 2
    if loss_kind == 'huber':
        absolute = jnp.abs(residual)
        return jnp.where(
            absolute <= huber_delta,
            residual ** 2,
            2 * huber_delta * absolute - huber_delta ** 2,
        )
    raise ValueError(f"Unknown ons_loss '{loss_kind}' (use 'l2' or 'huber')")


def create_optimizers(
    learning_rate: float = 1e-3,
    # Per-group clip thresholds. These are spike-catchers, NOT a tight clip:
    # measured gradient global-norms at a healthy mid-training checkpoint are
    # main ~2.4e2, ns_cm ~2.8e2, ns_ip ~6e4 (the losses are summed, not meaned),
    # so the thresholds sit ~10x above normal. Normal steps pass through
    # unclipped (preserving parity with the unclipped torch dynamics); only a
    # genuine explosion gets bounded. Raise/lower per experiment if needed.
    max_grad_norm: Tuple[float, float, float] = (2.5e3, 2.5e3, 1.0e6),
    # Learning-rate schedule. The torch reference uses a constant lr, which makes
    # the loss wander in the back half (the optimizer keeps taking full-size
    # steps and never settles into the basin). 'warmup_cosine' warms up over
    # warmup_steps then cosine-decays to end_learning_rate over max_gradient_updates,
    # which tames the early oscillation and lets the loss settle late. Set
    # lr_schedule='constant' to restore exact torch parity.
    lr_schedule: str = 'constant',
    max_gradient_updates: int = 100_000,
    warmup_steps: int = 1_000,
    end_learning_rate: float = 1e-5,
) -> Tuple[optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation]:
    """Create optimizers for different parameter groups.

    Each optimizer is hardened against the FFT-deconvolution instability that
    used to poison the whole model with NaNs:
      * apply_if_finite skips the update entirely when a non-finite gradient
        appears, so a bad step is dropped instead of corrupting every parameter
        permanently. This is the primary guard (Adam already bounds its step to
        ~lr, so the poison only ever enters via inf/NaN gradients).
      * clip_by_global_norm is a secondary backstop that keeps a rare
        finite-but-huge gradient from corrupting Adam's second-moment estimate.
    """
    if lr_schedule == 'warmup_cosine':
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_gradient_updates,
            end_value=end_learning_rate,
        )
    elif lr_schedule == 'constant':
        lr = learning_rate
    else:
        raise ValueError(f"Unknown lr_schedule '{lr_schedule}' (use 'constant' or 'warmup_cosine')")

    def _make(clip_norm: float) -> optax.GradientTransformation:
        return optax.apply_if_finite(
            optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adam(lr),
            ),
            # apply_if_finite *accepts* the update after this many consecutive
            # non-finite gradients, so keep it high: we never want to let a NaN
            # through. A genuinely stuck run shows up as a frozen loss instead.
            max_consecutive_errors=1_000_000,
        )

    return _make(max_grad_norm[0]), _make(max_grad_norm[1]), _make(max_grad_norm[2])


@eqx.filter_jit
def retina_forward(model: RetinaModel, x: jax.Array, key: jax.Array) -> Tuple:
    """JIT-compiled forward pass for the fixed retina model."""
    return model(x, key=key)


@eqx.filter_jit
def prepare_batch(
    retina: RetinaModel,
    batch_LMS_full_field: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Retina forward + slicing + CST color conversion, fused into one kernel.

    Previously the retina forward was jitted but the transpose/slice/CST steps
    ran eagerly between two jit boundaries, adding a dispatch round-trip (and on
    accelerators a host sync) every step. Doing it all in one compiled function
    removes that boundary and lets XLA fuse the transposes into the conversion.

    Returns (batch_ons, warped_linsRGB1, true_dxy) where batch_ons has shape
    (B, T, 1, H, W) -- every eye-motion timestep, consumed pairwise by the
    multi-frame reconstruction loss. warped_linsRGB1 is the frame-0 warped image
    used as the ns_ip RGB probe target.
    """
    # (B, H, W, 4) -> (B, 4, H, W)
    batch_LMS_full = jnp.transpose(batch_LMS_full_field, (0, 3, 1, 2))

    batch_ons, batch_true_dxy, batch_warped_LMS = retina(batch_LMS_full, key=key)

    warped_LMS1 = batch_warped_LMS[:, 0]

    # LMS -> linear sRGB (CST works channel-last)
    warped_linsRGB1 = retina.CST.LMS_to_linsRGB(jnp.transpose(warped_LMS1, (0, 2, 3, 1)))
    warped_linsRGB1 = jnp.transpose(warped_linsRGB1, (0, 3, 1, 2))

    return batch_ons, warped_linsRGB1, batch_true_dxy


def masked_latent_consistency_loss(
    pred_ip: jax.Array,
    target_ip: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Channel-count-independent latent equivariance loss."""
    error = (pred_ip - target_ip) ** 2 * mask
    valid_per_pixel = jnp.sum(mask, axis=0) + 1
    return jnp.sum(jnp.sum(error, axis=0) / valid_per_pixel) / pred_ip.shape[1]


def zero_cell_position_updates(updates: CortexModel) -> CortexModel:
    """Clear position-subtree updates while preserving optimizer tree shape."""
    zero_position_updates = jax.tree.map(
        jnp.zeros_like, updates.P_cell_position
    )
    return eqx.tree_at(
        lambda tree: tree.P_cell_position,
        updates,
        replace=zero_position_updates,
    )


@eqx.filter_jit
def train_step(
    cortex: CortexModel,
    opt_states: OptimizerStates,
    optimizers: Tuple[optax.GradientTransformation, ...],
    batch_ons: jax.Array,
    linsRGB1: jax.Array,
    true_dxy: jax.Array,
    cone_mosaic: jax.Array,
    kernel_size: int,
    simulating_tetra: bool,
    ns_ip_loss_kind: str = 'l2',
    latent_consistency_weight: float = 0.0,
    latent_consistency_batch_size: int = 0,
    update_cell_position: bool = True,
    update_neural_scope: bool = True,
    ons_loss_kind: str = 'l2',
    ons_huber_delta: float = 0.01,
) -> Tuple[CortexModel, OptimizerStates, Dict[str, float]]:
    """Three separate backward passes to match the reference PyTorch implementation.

    The original PyTorch code does:
        1. main_loss.backward() + main_optimizer.step()
        2. ns_cm_loss.backward() + ns_cm_optimizer.step()
        3. ns_ip_loss.backward() + ns_ip_optimizer.step()

    We replicate this by computing three separate filter_value_and_grad calls,
    each with its own loss function, and applying updates sequentially.

    Only the main loss needs the full forward (decode + global-movement pyramid
    + warp + encode). The neural-scope losses operate on stop_gradient'd inputs:
    ns_cm recomputes only its tiny projection, while ns_ip reuses the decoded
    percept from the main forward and trains only its projection head. This
    avoids running the expensive global-movement pyramid or decoder twice.
    """

    main_opt, ns_cm_opt, ns_ip_opt = optimizers
    P = kernel_size // 2

    # --- Main Loss Backward Pass ---
    def main_loss_fn(model):
        cell_position = model.P_cell_position
        if not update_cell_position:
            cell_position = jax.tree.map(
                lambda leaf: (
                    jax.lax.stop_gradient(leaf)
                    if eqx.is_inexact_array(leaf)
                    else leaf
                ),
                cell_position,
            )

        # Multi-frame objective over consecutive timestep pairs. The eye is
        # always moving, so each pair (t, t+1) is a self-supervision constraint:
        # decode frame t, warp its percept by the inter-frame eye motion,
        # re-encode, and match frame t+1's ONS. We warp by the simulated
        # ground-truth `true_dxy` rather than running the block-matching
        # M_global_movement estimator: that estimator is a fixed, non-learned
        # algorithm and was ~half the per-pair cost, yet in training the motion
        # is something we generate and therefore already know exactly. (The
        # estimator is left untouched and still runs at multi-frame inference.)
        # Summing over the trajectory, averaged by the pair count, keeps the
        # loss scale independent of trajectory length.
        n_frames = batch_ons.shape[1]
        n_pairs = n_frames - 1
        res_half = model.M_global_movement.required_image_resolution / 2

        # Frame-0 decode feeds the ns_ip RGB probe and the first pair's source.
        warped_ip1 = model.decode(batch_ons[:, 0])

        main_loss = jnp.array(0.0)
        normalized_ons_mse = jnp.array(0.0)
        valid_mask_fraction = jnp.array(0.0)
        latent_consistency_loss = jnp.array(0.0)

        for t in range(n_pairs):
            ons_a = batch_ons[:, t]
            ons_b = batch_ons[:, t + 1]
            dxy_t = true_dxy[:, t, :]

            if model.temporal_fusion == 'oracle':
                pred_warped_ip, mask = model.decode_fused(ons_a, ons_b, dxy_t)
            else:
                warped_ip_a = warped_ip1 if t == 0 else model.decode(ons_a)
                pred_warped_ip, mask = cell_position.efficient_warping(
                    warped_ip_a, dxy_t
                )
            ons_b_pred = model.encode(pred_warped_ip)

            ons_b_crop = ons_b[:, :, P:-P, P:-P]
            ons_b_pred_crop = ons_b_pred[:, :, P:-P, P:-P]
            mask_crop = mask[:, :, P:-P, P:-P]

            residual = ons_b_pred_crop - ons_b_crop
            reconstruction_penalty = ons_reconstruction_penalty(
                residual,
                loss_kind=ons_loss_kind,
                huber_delta=ons_huber_delta,
            )
            main_loss = main_loss + jnp.sum(
                jnp.sum(
                    reconstruction_penalty * mask_crop,
                    axis=0,
                ) / (jnp.sum(mask_crop, axis=0) + 1)
            )
            squared_error = (ons_b_pred_crop - ons_b_crop) ** 2 * mask_crop
            normalized_ons_mse = normalized_ons_mse + jnp.sum(squared_error) / jnp.maximum(
                jnp.sum(mask_crop), 1.0
            )
            valid_mask_fraction = valid_mask_fraction + jnp.mean(mask_crop)

            # The ONS reconstruction only observes the local projection of the
            # latent percept selected by the cone mosaic. Enforce equivariance
            # across eye movements in every latent channel so the decoder cannot
            # hide unstable structure in directions the encoder does not see.
            if latent_consistency_weight:
                consistency_batch_size = (
                    ons_b.shape[0]
                    if latent_consistency_batch_size <= 0
                    else min(latent_consistency_batch_size, ons_b.shape[0])
                )
                ip_b = model.decode(ons_b[:consistency_batch_size])
                ip_b_crop = ip_b[:, :, P:-P, P:-P]
                pred_ip_b_crop = pred_warped_ip[
                    :consistency_batch_size, :, P:-P, P:-P
                ]
                consistency_mask = mask_crop[:consistency_batch_size]
                latent_consistency_loss = latent_consistency_loss + masked_latent_consistency_loss(
                    pred_ip_b_crop, ip_b_crop, consistency_mask
                )

        # Average across pairs so the loss scale (and thus LR / grad-norm
        # expectations) is independent of trajectory length.
        inv_pairs = 1.0 / n_pairs
        main_loss = main_loss * inv_pairs
        normalized_ons_mse = normalized_ons_mse * inv_pairs
        valid_mask_fraction = valid_mask_fraction * inv_pairs
        latent_consistency_loss = latent_consistency_loss * inv_pairs

        # Motion is taken as ground truth (no estimator), so this slot now
        # reports the mean eye-motion magnitude in pixels rather than an
        # estimator error -- a useful check that the eye is actually moving.
        movement_mae_pixels = jnp.mean(jnp.abs(true_dxy)) * res_half

        objective = main_loss + latent_consistency_weight * latent_consistency_loss
        aux = (
            jax.lax.stop_gradient(warped_ip1),
            jax.lax.stop_gradient(main_loss),
            jax.lax.stop_gradient(latent_consistency_loss),
            jax.lax.stop_gradient(normalized_ons_mse),
            jax.lax.stop_gradient(movement_mae_pixels),
            jax.lax.stop_gradient(valid_mask_fraction),
        )
        return objective, aux

    main_filter = jax.tree.map(lambda _: False, cortex)
    main_filter = eqx.tree_at(
        lambda m: (m.C_cone_spectral_type, m.D_demosaicing, m.W_lateral_inhibition_weights, m.P_cell_position),
        main_filter, replace=(True, True, True, True)
    )

    (l_main_objective, main_aux), main_grads = eqx.filter_value_and_grad(
        main_loss_fn,
        has_aux=True,
    )(cortex)
    (
        warped_ip1_detached,
        l_main,
        l_latent_consistency,
        l_normalized_ons_mse,
        movement_mae_pixels,
        valid_mask_fraction,
    ) = main_aux
    main_grads = eqx.filter(main_grads, main_filter)
    grad_norm_cone = tree_l2_norm(main_grads.C_cone_spectral_type)
    grad_norm_demosaicing = tree_l2_norm(main_grads.D_demosaicing)
    grad_norm_lateral_inhibition = tree_l2_norm(
        main_grads.W_lateral_inhibition_weights
    )
    grad_norm_position = tree_l2_norm(main_grads.P_cell_position)
    main_updates, new_main_state = main_opt.update(main_grads, opt_states.main, eqx.filter(cortex, main_filter))
    if not update_cell_position:
        main_updates = zero_cell_position_updates(main_updates)
    cortex = eqx.apply_updates(cortex, main_updates)

    # --- NS Cone Mosaic Loss Backward Pass ---
    # Mirrors the ns_cm branch of main_train exactly (cone identity is detached
    # there too), without the rest of the forward pass.
    if update_neural_scope:
        def ns_cm_loss_fn(model):
            C = jax.lax.stop_gradient(model.C_cone_spectral_type.get_cone_identity_function())
            pred_cone_mosaic = model.ns_cm(C)
            return jnp.sum((pred_cone_mosaic - cone_mosaic)[:, :, P:-P, P:-P] ** 2)

        cm_filter = jax.tree.map(lambda _: False, cortex)
        cm_filter = eqx.tree_at(lambda m: m.ns_cm, cm_filter, replace=True)

        l_cm, cm_grads = eqx.filter_value_and_grad(ns_cm_loss_fn)(cortex)
        cm_grads = eqx.filter(cm_grads, cm_filter)
        cm_updates, new_cm_state = ns_cm_opt.update(cm_grads, opt_states.ns_cm, eqx.filter(cortex, cm_filter))
        cortex = eqx.apply_updates(cortex, cm_updates)
    else:
        l_cm = jnp.array(0.0)
        new_cm_state = opt_states.ns_cm

    # --- NS Internal Percept Loss Backward Pass ---
    # In the original, ns_ip_loss is only backpropagated when not simulating_tetra
    if not simulating_tetra and update_neural_scope:
        # Mirrors the ns_ip branch of main_train: reuse the detached decode from
        # the main forward and only train the ns_ip projection.
        def ns_ip_loss_fn(model):
            pred_warped_linsRGB1 = model.ns_ip(warped_ip1_detached)
            err = (pred_warped_linsRGB1[:, :3] - linsRGB1[:, :3])[:, :, P:-P, P:-P]
            # L1 vs L2 readout loss. L2 (the torch default) is mean-seeking and
            # desaturates vivid colors by ~15-20%; L1 preserves the mode and
            # recovers most of that saturation at a negligible MSE cost (validated
            # by re-fitting the readout on the frozen latent). 'l2' restores parity.
            if ns_ip_loss_kind == 'l1':
                return jnp.sum(jnp.abs(err))
            return jnp.sum(err ** 2)

        ip_filter = jax.tree.map(lambda _: False, cortex)
        ip_filter = eqx.tree_at(lambda m: m.ns_ip, ip_filter, replace=True)

        l_ip, ip_grads = eqx.filter_value_and_grad(ns_ip_loss_fn)(cortex)
        ip_grads = eqx.filter(ip_grads, ip_filter)
        ip_updates, new_ip_state = ns_ip_opt.update(ip_grads, opt_states.ns_ip, eqx.filter(cortex, ip_filter))
        cortex = eqx.apply_updates(cortex, ip_updates)
    else:
        l_ip = jnp.array(0.0)
        new_ip_state = opt_states.ns_ip

    new_opt_states = OptimizerStates(new_main_state, new_cm_state, new_ip_state)

    losses = {
        'main': l_main,
        'main_objective': l_main_objective,
        'normalized_ons_mse': l_normalized_ons_mse,
        'movement_mae_pixels': movement_mae_pixels,
        'valid_mask_fraction': valid_mask_fraction,
        'grad_norm_cone': grad_norm_cone,
        'grad_norm_demosaicing': grad_norm_demosaicing,
        'grad_norm_lateral_inhibition': grad_norm_lateral_inhibition,
        'grad_norm_position': grad_norm_position,
        'latent_consistency': l_latent_consistency,
        'ns_cm': l_cm,
        'ns_ip': l_ip,
        'total': l_main_objective + l_cm + l_ip
    }

    return cortex, new_opt_states, losses


def train_cortical_model(
    params: Dict[str, Any],
    checkpoint_dir: str = None,
    resume_from: str = None,
    num_workers: int = None
):
    """Main training loop for cortical model."""
    if num_workers is None:
        # Only used by the Grain fallback loader. The default path is the
        # device-resident crop loader (no CPU workers, no per-step transfer); if
        # it is disabled or unavailable, give Grain a few workers so loading at
        # least overlaps with compute instead of blocking it (worker_count=0 is
        # fully synchronous and pins throughput regardless of GPU speed).
        num_workers = 4

    print("="*70)
    print("JAX/Equinox/Optax Training - Matisse Cortical Model (Optimized with Orbax)")
    print("="*70)

    print(f"\nJAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    experiment_name = params['Experiment']['name']
    root_dir = params.get('root_dir', os.path.dirname(os.path.abspath(__file__)))

    if checkpoint_dir is None:
        checkpoint_dir = f'{root_dir}/Experiment/LearnedWeights/{experiment_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Orbax Checkpointing Setup ---
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    manager_options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    # New API: Do not pass checkpointer. It is inferred from save args.
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, 
        options=manager_options
    )

    max_gradient_updates = params['Training']['max_gradient_updates']
    learning_rate = params['Training']['learning_rate']
    batch_size = params['Dataset']['batch_size']
    simulating_tetra = params['Experiment']['simulating_tetra']

    print(f"\nExperiment: {experiment_name}")
    print(f"Max gradient updates: {max_gradient_updates:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # --- Initialization ---
    key = jax.random.PRNGKey(42)

    simulation_size = params.get('Experiment', {}).get('simulation_size') or \
                      params.get('RetinaModel', {}).get('simulation_size', 256)
    cone_distribution = params.get('RetinaModel', {}).get('retina_spatial_sampling', {}).get('cone_distribution', 'Human')
    latent_dim = params.get('CortexModel', {}).get('latent_dim') or \
                 params.get('CorticalModel', {}).get('latent_dim', 8)
    cortical_cfg = params.get('CorticalModel', params.get('CortexModel', {}))
    demosaicing_cfg = cortical_cfg.get('cortex_learn_demosaicing', {})

    # Eye-motion policy: read the `retina_eye_motion` block. `type` selects the
    # generator; every other key is forwarded as a parameter to it (e.g. drift
    # / microsaccade settings for the Fixational policy).
    eye_motion_cfg = params['RetinaModel'].get('retina_eye_motion') or {}
    eye_motion_type = eye_motion_cfg.get('type', 'Default')
    eye_motion_params = {
        key: value for key, value in eye_motion_cfg.items() if key != 'type'
    }

    # Retina
    retina = RetinaModel(
        simulation_size=simulation_size,
        timesteps_per_image=params['Experiment']['timesteps_per_image'],
        max_shift_size=params['RetinaModel']['max_shift_size'],
        cone_types_str=params['RetinaModel']['retina_spectral_sampling']['cone_types'],
        cone_distribution_type=cone_distribution,
        simulating_tetra=simulating_tetra,
        cone_fundamentals_params=params['RetinaModel']['retina_spectral_sampling'].get(
            'cone_fundamentals'
        ),
        cone_gain_adaptation=params['RetinaModel']['retina_spectral_sampling'].get(
            'gain_adaptation', 'none'
        ),
        eye_motion_type=eye_motion_type,
        eye_motion_params=eye_motion_params,
        root_dir=root_dir
    )
    print(
        f"✓ Retina initialized (image resolution: {retina.required_image_resolution}, "
        f"eye_motion={type(retina.EyeMotion).__name__})"
    )

    # Cortex
    key, subkey = jax.random.split(key)
    cortex = CortexModel(
        latent_dim=latent_dim,
        simulation_size=simulation_size,
        required_image_resolution=retina.required_image_resolution,
        simulating_tetra=simulating_tetra,
        demosaicing_type=demosaicing_cfg.get('type', 'Default'),
        demosaicing_base_channels=demosaicing_cfg.get('base_channels', 16),
        demosaicing_compute_dtype=demosaicing_cfg.get('compute_dtype', 'float32'),
        demosaicing_context_channels=demosaicing_cfg.get('context_channels', 16),
        demosaicing_context_kernel_size=demosaicing_cfg.get('context_kernel_size', 5),
        demosaicing_hidden_channels=demosaicing_cfg.get('hidden_channels', 32),
        demosaicing_hidden_kernel_size=demosaicing_cfg.get('hidden_kernel_size', 1),
        demosaicing_num_frequencies=demosaicing_cfg.get('num_frequencies', 6),
        demosaicing_omega0=demosaicing_cfg.get('omega0', 10.0),
        demosaicing_activation=demosaicing_cfg.get('activation', 'sine'),
        demosaicing_conditioning=demosaicing_cfg.get('conditioning', 'none'),
        demosaicing_gaussian_kernel_size=demosaicing_cfg.get('gaussian_kernel_size', 9),
        demosaicing_gaussian_sigma=demosaicing_cfg.get('gaussian_sigma', 2.0),
        demosaicing_gaussian_epsilon=demosaicing_cfg.get('gaussian_epsilon', 1e-3),
        temporal_fusion=cortical_cfg.get('temporal_fusion', 'none'),
        key=subkey
    )
    print(
        "✓ Cortex initialized "
        f"(demosaicing={type(cortex.D_demosaicing).__name__}, "
        f"compute_dtype={cortex.D_demosaicing.compute_dtype})"
    )

    # Dataset
    print("\nLoading dataset...")
    dataset = create_dataset(params['Dataset']['dataset_name'], params, retina)
    print(f"✓ Dataset loaded: {len(dataset):,} samples")

    # Prefer the device-resident crop loader: it keeps the whole image set in
    # accelerator memory and crops on-device, removing CPU cropping and the
    # per-step host->device transfer from the hot loop. Falls back to the Grain
    # loader if shapes are ragged or it doesn't fit / errors out.
    use_device_resident = params.get('Dataset', {}).get('device_resident', True)
    loader = None
    if use_device_resident and getattr(dataset, 'file_list', None):
        try:
            loader = DeviceResidentCropLoader(
                dataset,
                batch_size=batch_size,
                crop_size=dataset.dim_image,
                seed=42,
            )
            print(f"✓ Device-resident crop loader created (crop={dataset.dim_image})")
        except Exception as e:
            print(f"  Device-resident loader unavailable ({e}); using Grain loader.")
            loader = None
    if loader is None:
        loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=42,
            augment=True,  # runtime augmentation (the reference precomputes it)
        )
        print(f"✓ DataLoader created (num_workers={num_workers}, augment=True)")

    # Optimizers. Schedule settings come from the Training config; defaults keep
    # torch parity (constant lr) unless 'lr_schedule: warmup_cosine' is set.
    training_cfg = params.get('Training', {})
    cell_position_update_cycle = training_cfg.get('cell_position_update_cycle', 1)
    if cell_position_update_cycle < 1:
        raise ValueError('cell_position_update_cycle must be at least 1')
    neural_scope_update_cycle = training_cfg.get('neural_scope_update_cycle', 1)
    if neural_scope_update_cycle < 1:
        raise ValueError('neural_scope_update_cycle must be at least 1')
    checkpoint_cycle = training_cfg.get('checkpoint_cycle')
    if checkpoint_cycle is not None and checkpoint_cycle < 1:
        raise ValueError('checkpoint_cycle must be at least 1')
    latent_consistency_batch_size = training_cfg.get(
        'latent_consistency_batch_size', 0
    )
    if latent_consistency_batch_size < 0:
        raise ValueError('latent_consistency_batch_size cannot be negative')
    main_opt, ns_cm_opt, ns_ip_opt = create_optimizers(
        learning_rate,
        lr_schedule=training_cfg.get('lr_schedule', 'constant'),
        max_gradient_updates=max_gradient_updates,
        warmup_steps=training_cfg.get('warmup_steps', 1_000),
        end_learning_rate=training_cfg.get('end_learning_rate', 1e-5),
    )
    optimizers = (main_opt, ns_cm_opt, ns_ip_opt)

    # Initial Optimizer States
    main_filter = jax.tree.map(lambda _: False, cortex)
    main_filter = eqx.tree_at(
        lambda m: (m.C_cone_spectral_type, m.D_demosaicing, m.W_lateral_inhibition_weights, m.P_cell_position),
        main_filter, replace=(True, True, True, True)
    )
    cm_filter = jax.tree.map(lambda _: False, cortex)
    cm_filter = eqx.tree_at(lambda m: m.ns_cm, cm_filter, replace=True)
    ip_filter = jax.tree.map(lambda _: False, cortex)
    ip_filter = eqx.tree_at(lambda m: m.ns_ip, ip_filter, replace=True)

    opt_states = OptimizerStates(
        main=main_opt.init(eqx.filter(cortex, main_filter)),
        ns_cm=ns_cm_opt.init(eqx.filter(cortex, cm_filter)),
        ns_ip=ns_ip_opt.init(eqx.filter(cortex, ip_filter))
    )
    print("✓ Optimizers initialized")

    # Ground truth
    true_LI_kernel_size = retina.LateralInhibition.get_kernel_size()
    true_cone_mosaic = retina.SpectralSampling.get_cone_mosaic()

    logging_timesteps = set()
    for i in range(0, 2000, 100): logging_timesteps.add(i)
    for i in range(2000, 10000, 1000): logging_timesteps.add(i)
    for i in range(10000, max_gradient_updates + 1, 10000): logging_timesteps.add(i)

    # --- Progress Logger Setup ---
    learning_progress_logging = params.get('Training', {}).get('learning_progress_logging', False)
    logger = None
    if learning_progress_logging:
        logger = LocalProgressLogger(
            experiment_name=experiment_name,
            required_image_resolution=retina.required_image_resolution,
            root_dir=root_dir,
            test_images_dir='Experiment/ProgressLogger/test_images',
        )
        print(f"✓ Progress logger initialized")
        print(f"  Logging to: {logger.log_dir}")
        print(f"  Test images: {logger.num_test_images}")

    # --- Training Loop ---
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    num_gradient_updates = 0
    if resume_from is not None:
        if resume_from == 'latest':
            restore_step = checkpoint_manager.latest_step()
        else:
            restore_step = int(resume_from)
        if restore_step is None:
            raise RuntimeError(f'No checkpoint available in {checkpoint_dir}')
        restored = checkpoint_manager.restore(
            restore_step,
            args=ocp.args.StandardRestore({
                'model': cortex,
                'opt_states': opt_states
            })
        )
        cortex = restored['model']
        opt_states = restored['opt_states']
        num_gradient_updates = restore_step
        print(f"Resumed model and optimizer state from step {restore_step}")

    bar = tqdm(total=max_gradient_updates, desc="Training")
    bar.update(num_gradient_updates)
    start_time = time.time()

    # Initial checkpoint for a fresh run.
    if resume_from is None and num_gradient_updates in logging_timesteps:
        checkpoint_manager.save(
            num_gradient_updates, 
            args=ocp.args.StandardSave({'model': cortex, 'opt_states': opt_states})
        )
        checkpoint_manager.wait_until_finished()
        # Also save separate .eqx file for Penzai/visualization
        eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)

    while num_gradient_updates < max_gradient_updates:
        for batch_LMS_full_field in loader:
            if num_gradient_updates >= max_gradient_updates:
                break

            # Retina forward + slice + CST, fused into one jitted kernel.
            key, subkey = jax.random.split(key)
            batch_ons, batch_warped_linsRGB1, batch_true_dxy = prepare_batch(
                retina, batch_LMS_full_field, subkey
            )

            # Cortex Train Step
            cortex, opt_states, losses = train_step(
                cortex, opt_states, optimizers,
                batch_ons, batch_warped_linsRGB1,
                batch_true_dxy, true_cone_mosaic, true_LI_kernel_size,
                simulating_tetra,
                ns_ip_loss_kind=training_cfg.get('ns_ip_loss', 'l2'),
                latent_consistency_weight=training_cfg.get('latent_consistency_weight', 0.0),
                latent_consistency_batch_size=latent_consistency_batch_size,
                update_cell_position=(
                    (num_gradient_updates + 1) % cell_position_update_cycle == 0
                ),
                update_neural_scope=(
                    (num_gradient_updates + 1) % neural_scope_update_cycle == 0
                    or (num_gradient_updates + 1) % 25 == 0
                    or (num_gradient_updates + 1) in logging_timesteps
                ),
                ons_loss_kind=training_cfg.get('ons_loss', 'l2'),
                ons_huber_delta=training_cfg.get('ons_huber_delta', 0.01),
            )

            num_gradient_updates += 1
            bar.update(1)
            # Pulling the losses to host with float() forces a device sync, which
            # serializes JAX's async dispatch. Do it sparsely (and on logging
            # steps) so the device can run ahead between updates.
            if num_gradient_updates % 25 == 0 or num_gradient_updates in logging_timesteps:
                scalar_metrics = {
                    'main': f'{float(losses["main"]):.4f}',
                    'latent': f'{float(losses["latent_consistency"]):.4f}',
                    'ns_cm': f'{float(losses["ns_cm"]):.4f}',
                    'ns_ip': f'{float(losses["ns_ip"]):.4f}',
                    'move_px': f'{float(losses["movement_mae_pixels"]):.3f}',
                    'g_D': f'{float(losses["grad_norm_demosaicing"]):.2e}',
                }
                bar.set_postfix(scalar_metrics)
                if logger is not None:
                    metric_record = {
                        'step': num_gradient_updates,
                        **{key: float(value) for key, value in losses.items()},
                    }
                    with open(
                        os.path.join(logger.log_dir, 'training_metrics.jsonl'),
                        'a',
                        encoding='utf-8',
                    ) as metrics_file:
                        metrics_file.write(json.dumps(metric_record) + '\n')

            should_checkpoint = (
                num_gradient_updates in logging_timesteps
                or (
                    checkpoint_cycle is not None
                    and num_gradient_updates % checkpoint_cycle == 0
                )
            )
            if should_checkpoint:
                checkpoint_manager.save(
                    num_gradient_updates, 
                    args=ocp.args.StandardSave({'model': cortex, 'opt_states': opt_states})
                )
                checkpoint_manager.wait_until_finished()
                # Also save separate .eqx file for Penzai/visualization
                eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)

                # Image logging remains on the sparse logging schedule; it is
                # intentionally decoupled from lightweight recovery checkpoints.
                if logger is not None and num_gradient_updates in logging_timesteps:
                    logger.log_progress(
                        simulating_tetra=simulating_tetra,
                        retina=retina,
                        cortex=cortex,
                        num_gradient_updates=num_gradient_updates,
                        main_loss=losses['main'],
                        ns_cm_loss=losses['ns_cm'],
                        ns_ip_loss=losses['ns_ip'],
                    )

    bar.close()
    
    # Generate progress videos at the end of training
    if logger is not None:
        print("\nGenerating progress videos...")
        logger.generate_progress_video()
    
    # Save final model if not already saved
    if num_gradient_updates not in logging_timesteps:
        print(f"Saving final checkpoint at step {num_gradient_updates}...")
        checkpoint_manager.save(
            num_gradient_updates,
            args=ocp.args.StandardSave({'model': cortex, 'opt_states': opt_states})
        )
        checkpoint_manager.wait_until_finished()
        eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)
        
        # Also log progress at final step if not already logged
        if logger is not None:
            logger.log_progress(
                simulating_tetra=simulating_tetra,
                retina=retina,
                cortex=cortex,
                num_gradient_updates=num_gradient_updates,
                main_loss=losses['main'],
                ns_cm_loss=losses['ns_cm'],
                ns_ip_loss=losses['ns_ip'],
            )
    
    checkpoint_manager.close()
    
    elapsed = time.time() - start_time
    print(f"\nTraining Complete! Time: {elapsed/60:.1f}m, Speed: {max_gradient_updates/elapsed:.2f} it/s")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    if logger is not None:
        print(f"Logs saved to: {logger.log_dir}")

if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_filename', default='Default/LMS')
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{root_dir}/Experiment/Config/{args.config_filename}.yaml'

    if not os.path.exists(config_path):
        print(f"Config file not found, using default.")
        params = {
            'Experiment': {'name': 'JAX_LMS_Default', 'timesteps_per_image': 2, 'simulating_tetra': False},
            'RetinaModel': {'simulation_size': 256, 'max_shift_size': 15, 'retina_spatial_sampling': {'cone_distribution': 'Human'}, 'retina_spectral_sampling': {'cone_types': 'LMS'}},
            'CortexModel': {'latent_dim': 8},
            'Dataset': {'dataset_name': 'NTIRE', 'batch_size': 8},
            'Training': {'max_gradient_updates': 100000, 'learning_rate': 1e-3, 'learning_progress_logging': True},
            'root_dir': root_dir
        }
    else:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
            params['root_dir'] = root_dir

    train_cortical_model(params, args.checkpoint_dir, args.resume_from, args.num_workers)
