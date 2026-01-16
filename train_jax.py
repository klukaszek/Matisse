"JAX/Equinox/Optax training loop for Matisse cortical model."
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import orbax.checkpoint as ocp
import os
import time
from tqdm.auto import tqdm
from typing import Tuple, Dict, Any, NamedTuple

from SimulatedJAX.Retina import RetinaModel
from SimulatedJAX.Cortex import CortexModel
from DatasetJAX import create_dataset
from DatasetJAX.NTIRE import create_dataloader

# Define a container for the three optimizer states
class OptimizerStates(NamedTuple):
    main: optax.OptState
    ns_cm: optax.OptState
    ns_ip: optax.OptState

def create_optimizers(
    learning_rate: float = 1e-3
) -> Tuple[optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation]:
    """Create optimizers for different parameter groups."""
    main_optimizer = optax.adam(learning_rate)
    ns_cm_optimizer = optax.adam(learning_rate)
    ns_ip_optimizer = optax.adam(learning_rate)
    return main_optimizer, ns_cm_optimizer, ns_ip_optimizer

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
    """Single unified training step."""
    
    main_opt, ns_cm_opt, ns_ip_opt = optimizers
    
    def loss_fn(model):
        main_loss, ns_cm_loss, ns_ip_loss, _, _, _ = model.main_train(
            ons1, ons2, linsRGB1, true_dxy, cone_mosaic, kernel_size
        )
        total_loss = main_loss + ns_cm_loss + ns_ip_loss
        return total_loss, (main_loss, ns_cm_loss, ns_ip_loss)

    # Compute gradients for all parameters
    (total_loss_val, (l_main, l_cm, l_ip)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(cortex)

    # --- Update Main Parameters ---
    main_filter = jax.tree.map(lambda _: False, cortex)
    main_filter = eqx.tree_at(
        lambda m: (m.C_cone_spectral_type, m.D_demosaicing, m.W_lateral_inhibition_weights, m.P_cell_position),
        main_filter, replace=(True, True, True, True)
    )
    main_grads = eqx.filter(grads, main_filter)
    main_updates, new_main_state = main_opt.update(main_grads, opt_states.main, eqx.filter(cortex, main_filter))
    cortex = eqx.apply_updates(cortex, main_updates)

    # --- Update NS Cone Mosaic ---
    cm_filter = jax.tree.map(lambda _: False, cortex)
    cm_filter = eqx.tree_at(lambda m: m.ns_cm, cm_filter, replace=True)
    cm_grads = eqx.filter(grads, cm_filter)
    cm_updates, new_cm_state = ns_cm_opt.update(cm_grads, opt_states.ns_cm, eqx.filter(cortex, cm_filter))
    cortex = eqx.apply_updates(cortex, cm_updates)

    # --- Update NS Internal Percept ---
    ip_filter = jax.tree.map(lambda _: False, cortex)
    ip_filter = eqx.tree_at(lambda m: m.ns_ip, ip_filter, replace=True)
    ip_grads = eqx.filter(grads, ip_filter)
    ip_updates, new_ip_state = ns_ip_opt.update(ip_grads, opt_states.ns_ip, eqx.filter(cortex, ip_filter))
    cortex = eqx.apply_updates(cortex, ip_updates)

    new_opt_states = OptimizerStates(new_main_state, new_cm_state, new_ip_state)
    
    losses = {
        'main': l_main,
        'ns_cm': l_cm,
        'ns_ip': l_ip,
        'total': total_loss_val
    }
    
    return cortex, new_opt_states, losses

def train_cortical_model(
    params: Dict[str, Any],
    checkpoint_dir: str = None,
    resume_from: str = None,
    num_workers: int = 4
):
    """Main training loop for cortical model."""
    print("="*70)
    print("JAX/Equinox/Optax Training - Matisse Cortical Model (Optimized with Orbax)")
    print("="*70)

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
        root_dir=root_dir
    )
    print(f"✓ Retina initialized (image resolution: {retina.required_image_resolution})")

    # Cortex
    key, subkey = jax.random.split(key)
    cortex = CortexModel(
        latent_dim=latent_dim,
        simulation_size=simulation_size,
        simulating_tetra=simulating_tetra,
        key=subkey
    )
    print(f"✓ Cortex initialized")

    # Dataset
    print("\nLoading dataset...")
    dataset = create_dataset(params['Dataset']['dataset_name'], params, retina)
    print(f"✓ Dataset loaded: {len(dataset):,} samples")

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=42
    )
    print(f"✓ DataLoader created (num_workers={num_workers})")

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

    # --- Training Loop ---
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    num_gradient_updates = 0
    bar = tqdm(total=max_gradient_updates, desc="Training")
    start_time = time.time()

    # Initial checkpoint
    if num_gradient_updates in logging_timesteps:
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

            # Data prep: (B, H, W, 4) -> (B, 4, H, W)
            batch_LMS_full = jnp.transpose(batch_LMS_full_field, (0, 3, 1, 2))

            # Retina Forward (no grad)
            key, subkey = jax.random.split(key)
            batch_ons, batch_true_dxy, batch_warped_LMS_current_FoV = retina(
                batch_LMS_full, key=subkey
            )

            # Slicing
            batch_ons1 = batch_ons[:, 0]
            batch_ons2 = batch_ons[:, 1]
            batch_warped_LMS1 = batch_warped_LMS_current_FoV[:, 0]
            
            # CST
            batch_warped_linsRGB1 = retina.CST.LMS_to_linsRGB(
                jnp.transpose(batch_warped_LMS1, (0, 2, 3, 1))
            )
            batch_warped_linsRGB1 = jnp.transpose(batch_warped_linsRGB1, (0, 3, 1, 2))

            # Cortex Train Step
            cortex, opt_states, losses = train_step(
                cortex, opt_states, optimizers,
                batch_ons1, batch_ons2, batch_warped_linsRGB1,
                batch_true_dxy, true_cone_mosaic, true_LI_kernel_size,
                simulating_tetra
            )

            num_gradient_updates += 1
            bar.update(1)
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
                # Also save separate .eqx file
                eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)

    bar.close()
    
    # Save final model if not already saved
    if num_gradient_updates not in logging_timesteps:
        print(f"Saving final checkpoint at step {num_gradient_updates}...")
        checkpoint_manager.save(
            num_gradient_updates,
            args=ocp.args.StandardSave({'model': cortex, 'opt_states': opt_states})
        )
        checkpoint_manager.wait_until_finished()
        eqx.tree_serialise_leaves(f"{checkpoint_dir}/model_{num_gradient_updates}.eqx", cortex)
    
    checkpoint_manager.close()
    
    elapsed = time.time() - start_time
    print(f"\nTraining Complete! Time: {elapsed/60:.1f}m, Speed: {max_gradient_updates/elapsed:.2f} it/s")
    print(f"Checkpoints saved to: {checkpoint_dir}")

if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_filename', default='Default/LMS')
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--num_workers', type=int, default=4)
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
            'Training': {'max_gradient_updates': 100000, 'learning_rate': 1e-3},
            'root_dir': root_dir
        }
    else:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
            params['root_dir'] = root_dir

    train_cortical_model(params, args.checkpoint_dir, args.resume_from, args.num_workers)
