"JAX/Equinox/Optax training loop for Matisse cortical model."
import os

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


def create_optimizers(
    learning_rate: float = 1e-3,
    # Per-group clip thresholds. These are spike-catchers, NOT a tight clip:
    # measured gradient global-norms at a healthy mid-training checkpoint are
    # main ~2.4e2, ns_cm ~2.8e2, ns_ip ~6e4 (the losses are summed, not meaned),
    # so the thresholds sit ~10x above normal. Normal steps pass through
    # unclipped (preserving parity with the unclipped torch dynamics); only a
    # genuine explosion gets bounded. Raise/lower per experiment if needed.
    max_grad_norm: Tuple[float, float, float] = (2.5e3, 2.5e3, 1.0e6),
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
    def _make(clip_norm: float) -> optax.GradientTransformation:
        return optax.apply_if_finite(
            optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adam(learning_rate),
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

    Returns (ons1, ons2, warped_linsRGB1, true_dxy).
    """
    # (B, H, W, 4) -> (B, 4, H, W)
    batch_LMS_full = jnp.transpose(batch_LMS_full_field, (0, 3, 1, 2))

    batch_ons, batch_true_dxy, batch_warped_LMS = retina(batch_LMS_full, key=key)

    ons1 = batch_ons[:, 0]
    ons2 = batch_ons[:, 1]
    warped_LMS1 = batch_warped_LMS[:, 0]

    # LMS -> linear sRGB (CST works channel-last)
    warped_linsRGB1 = retina.CST.LMS_to_linsRGB(jnp.transpose(warped_LMS1, (0, 2, 3, 1)))
    warped_linsRGB1 = jnp.transpose(warped_linsRGB1, (0, 3, 1, 2))

    return ons1, ons2, warped_linsRGB1, batch_true_dxy


@eqx.filter_jit
def train_step(
    cortex: CortexModel,
    opt_states: OptimizerStates,
    optimizers: Tuple[optax.GradientTransformation, ...],
    ons1: jax.Array,
    ons2: jax.Array,
    linsRGB1: jax.Array,
    true_dxy: jax.Array,
    cone_mosaic: jax.Array,
    kernel_size: int,
    simulating_tetra: bool
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
        # Inline the main-loss path so we can reuse the decoded percept for the
        # ns_ip branch below. Recomputing decode(ons1) there costs ~30ms/step on
        # MPS and also differs from the Torch reference, which reuses the
        # single forward graph for all three losses.
        warped_ip1 = model.decode(ons1)

        pred_dxy = model.M_global_movement(
            ons1, ons2, model.P_cell_position, true_dxy
        )
        pred_warped_ip2, mask2 = model.P_cell_position.efficient_warping(
            warped_ip1,
            pred_dxy[:, 0, :],
        )
        ons2_pred = model.encode(pred_warped_ip2)

        ons2_crop = ons2[:, :, P:-P, P:-P]
        ons2_pred_crop = ons2_pred[:, :, P:-P, P:-P]
        mask2_crop = mask2[:, :, P:-P, P:-P]

        main_loss = jnp.sum(
            jnp.sum(
                ((ons2_pred_crop - ons2_crop) ** 2) * mask2_crop,
                axis=0,
            ) / (jnp.sum(mask2_crop, axis=0) + 1)
        )
        return main_loss, jax.lax.stop_gradient(warped_ip1)

    main_filter = jax.tree.map(lambda _: False, cortex)
    main_filter = eqx.tree_at(
        lambda m: (m.C_cone_spectral_type, m.D_demosaicing, m.W_lateral_inhibition_weights, m.P_cell_position),
        main_filter, replace=(True, True, True, True)
    )

    (l_main, warped_ip1_detached), main_grads = eqx.filter_value_and_grad(
        main_loss_fn,
        has_aux=True,
    )(cortex)
    main_grads = eqx.filter(main_grads, main_filter)
    main_updates, new_main_state = main_opt.update(main_grads, opt_states.main, eqx.filter(cortex, main_filter))
    cortex = eqx.apply_updates(cortex, main_updates)

    # --- NS Cone Mosaic Loss Backward Pass ---
    # Mirrors the ns_cm branch of main_train exactly (cone identity is detached
    # there too), without the rest of the forward pass.
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

    # --- NS Internal Percept Loss Backward Pass ---
    # In the original, ns_ip_loss is only backpropagated when not simulating_tetra
    if not simulating_tetra:
        # Mirrors the ns_ip branch of main_train: reuse the detached decode from
        # the main forward and only train the ns_ip projection.
        def ns_ip_loss_fn(model):
            pred_warped_linsRGB1 = model.ns_ip(warped_ip1_detached)
            return jnp.sum(
                (pred_warped_linsRGB1[:, :3] - linsRGB1[:, :3])[:, :, P:-P, P:-P] ** 2
            )

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
        'ns_cm': l_cm,
        'ns_ip': l_ip,
        'total': l_main + l_cm + l_ip
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
        root_dir=root_dir
    )
    print(f"✓ Retina initialized (image resolution: {retina.required_image_resolution})")

    # Cortex
    key, subkey = jax.random.split(key)
    cortex = CortexModel(
        latent_dim=latent_dim,
        simulation_size=simulation_size,
        required_image_resolution=retina.required_image_resolution,
        simulating_tetra=simulating_tetra,
        key=subkey
    )
    print(f"✓ Cortex initialized")

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

    # Optimizers
    main_opt, ns_cm_opt, ns_ip_opt = create_optimizers(learning_rate)
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
            batch_ons1, batch_ons2, batch_warped_linsRGB1, batch_true_dxy = prepare_batch(
                retina, batch_LMS_full_field, subkey
            )

            # Cortex Train Step
            cortex, opt_states, losses = train_step(
                cortex, opt_states, optimizers,
                batch_ons1, batch_ons2, batch_warped_linsRGB1,
                batch_true_dxy, true_cone_mosaic, true_LI_kernel_size,
                simulating_tetra
            )

            num_gradient_updates += 1
            bar.update(1)
            # Pulling the losses to host with float() forces a device sync, which
            # serializes JAX's async dispatch. Do it sparsely (and on logging
            # steps) so the device can run ahead between updates.
            if num_gradient_updates % 25 == 0 or num_gradient_updates in logging_timesteps:
                bar.set_postfix({
                    'main': f'{float(losses["main"]):.4f}',
                    'ns_cm': f'{float(losses["ns_cm"]):.4f}',
                    'ns_ip': f'{float(losses["ns_ip"]):.4f}'
                })

            if num_gradient_updates in logging_timesteps:
                checkpoint_manager.save(
                    num_gradient_updates, 
                    args=ocp.args.StandardSave({'model': cortex, 'opt_states': opt_states})
                )
                checkpoint_manager.wait_until_finished()
                # Also save separate .eqx file for Penzai/visualization
                eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)

                # --- Log progress if enabled ---
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
