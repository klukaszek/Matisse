"""Integration tests for full Matisse JAX models."""
import jax
import jax.numpy as jnp
import sys
sys.path.append('.')

from Simulated.Cortex import CortexModel
from Simulated.Retina import RetinaModel


def test_cortex_model():
    """Test CortexModel end-to-end."""
    print("\n" + "="*60)
    print("Testing CortexModel")
    print("="*60)

    # Initialize
    key = jax.random.PRNGKey(42)
    latent_dim = 8
    simulation_size = 256

    print("Initializing CortexModel...")
    cortex = CortexModel(
        latent_dim=latent_dim,
        simulation_size=simulation_size,
        simulating_tetra=False,
        key=key
    )
    print("✓ CortexModel initialized")

    # Test decode
    print("\nTesting decode...")
    key, subkey = jax.random.split(key)
    ons = jax.random.normal(subkey, (2, 1, simulation_size, simulation_size))
    ip = cortex.decode(ons)
    print(f"  ONS shape: {ons.shape}")
    print(f"  IP shape: {ip.shape}")
    assert ip.shape == (2, latent_dim, simulation_size, simulation_size)
    print("✓ Decode works")

    # Test encode
    print("\nTesting encode...")
    ons_reconstructed = cortex.encode(ip)
    print(f"  IP shape: {ip.shape}")
    print(f"  ONS reconstructed shape: {ons_reconstructed.shape}")
    assert ons_reconstructed.shape == ons.shape
    print("✓ Encode works")

    # Test main_train
    print("\nTesting main_train...")
    key, subkey = jax.random.split(key)
    ons1 = jax.random.normal(subkey, (2, 1, simulation_size, simulation_size))
    ons2 = jax.random.normal(subkey, (2, 1, simulation_size, simulation_size))
    linsRGB1 = jax.random.uniform(subkey, (2, 3, simulation_size, simulation_size))
    true_dxy = jax.random.normal(subkey, (2, 2)) * 0.1
    cone_mosaic = jax.random.uniform(subkey, (2, 4, simulation_size, simulation_size))
    kernel_size = 21

    main_loss, ns_cm_loss, ns_ip_loss, pred_cone_mosaic, pred_ons2, pred_dxy = cortex.main_train(
        ons1, ons2, linsRGB1, true_dxy, cone_mosaic, kernel_size
    )

    print(f"  Main loss: {main_loss:.4f}")
    print(f"  NS cone mosaic loss: {ns_cm_loss:.4f}")
    print(f"  NS internal percept loss: {ns_ip_loss:.4f}")
    print(f"  Predicted dxy shape: {pred_dxy.shape}")
    print("✓ Main train works")

    return True


def test_retina_model():
    """Test RetinaModel end-to-end."""
    print("\n" + "="*60)
    print("Testing RetinaModel")
    print("="*60)

    # Initialize
    simulation_size = 256
    timesteps_per_image = 4  # Reduced for faster testing
    max_shift_size = 128

    print("Initializing RetinaModel...")
    print("  (This may take a moment - generating cone mosaics and MIP maps...)")
    retina = RetinaModel(
        simulation_size=simulation_size,
        timesteps_per_image=timesteps_per_image,
        max_shift_size=max_shift_size,
        cone_types_str='LMS',
        cone_distribution_type='Human',
        simulating_tetra=False
    )
    print("✓ RetinaModel initialized")
    print(f"  Required image resolution: {retina.required_image_resolution}")

    # Create input
    key = jax.random.PRNGKey(123)
    batch_size = 2
    full_field_size = retina.required_image_resolution + 2 * max_shift_size

    print(f"\nCreating input (full field size: {full_field_size})...")
    key, subkey = jax.random.split(key)
    batch_LMS_full_field = jax.random.uniform(
        subkey,
        (batch_size, 4, full_field_size, full_field_size)
    )

    # Forward pass
    print("\nRunning forward pass...")
    key, subkey = jax.random.split(key)
    batch_ons, batch_true_dxy, batch_warped_LMS = retina(
        batch_LMS_full_field,
        key=subkey,
        intermediate_outputs=False
    )

    print(f"  Input LMS shape: {batch_LMS_full_field.shape}")
    print(f"  Output ONS shape: {batch_ons.shape}")
    print(f"  True dxy shape: {batch_true_dxy.shape}")
    print(f"  Warped LMS shape: {batch_warped_LMS.shape}")

    assert batch_ons.shape == (batch_size, timesteps_per_image, 1, simulation_size, simulation_size)
    assert batch_true_dxy.shape == (batch_size, timesteps_per_image - 1, 2)
    assert batch_warped_LMS.shape == (batch_size, timesteps_per_image, 4, simulation_size, simulation_size)
    print("✓ Forward pass works")

    # Test with intermediate outputs
    print("\nTesting with intermediate outputs...")
    key, subkey = jax.random.split(key)
    outputs = retina(
        batch_LMS_full_field,
        key=subkey,
        intermediate_outputs=True
    )

    batch_ons, batch_true_dxy, batch_warped_LMS, batch_pa, batch_bipolar, batch_LMS_fov = outputs
    print(f"  Photoreceptor activation shape: {batch_pa.shape}")
    print(f"  Bipolar signals shape: {batch_bipolar.shape}")
    print(f"  LMS FoV shape: {batch_LMS_fov.shape}")
    print("✓ Intermediate outputs work")

    # Test helper methods
    print("\nTesting helper methods...")
    cone_mosaic = retina.get_cone_mosaic()
    cone_fundamentals = retina.get_cone_fundamentals()
    print(f"  Cone mosaic shape: {cone_mosaic.shape}")
    print(f"  Cone fundamentals shape: {cone_fundamentals.shape}")
    print("✓ Helper methods work")

    return True


def test_end_to_end():
    """Test full pipeline: Retina -> Cortex."""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)

    simulation_size = 256
    latent_dim = 8
    timesteps_per_image = 4
    max_shift_size = 128

    # Initialize models
    key = jax.random.PRNGKey(999)

    print("Initializing Retina...")
    retina = RetinaModel(
        simulation_size=simulation_size,
        timesteps_per_image=timesteps_per_image,
        max_shift_size=max_shift_size
    )

    print("Initializing Cortex...")
    key, subkey = jax.random.split(key)
    cortex = CortexModel(
        latent_dim=latent_dim,
        simulation_size=simulation_size,
        key=subkey
    )

    # Create input
    batch_size = 1
    full_field_size = retina.required_image_resolution + 2 * max_shift_size
    key, subkey = jax.random.split(key)
    batch_LMS_full_field = jax.random.uniform(
        subkey,
        (batch_size, 4, full_field_size, full_field_size)
    )

    # Forward through retina
    print("\nForward pass through Retina...")
    key, subkey = jax.random.split(key)
    batch_ons, batch_true_dxy, _ = retina(batch_LMS_full_field, key=subkey)

    # Extract first two timesteps
    ons1 = batch_ons[:, 0, :, :, :]  # (batch, 1, H, W)
    ons2 = batch_ons[:, 1, :, :, :]  # (batch, 1, H, W)
    true_dxy = batch_true_dxy[:, 0, :]  # (batch, 2)

    print(f"  ONS1 shape: {ons1.shape}")
    print(f"  ONS2 shape: {ons2.shape}")

    # Decode through cortex
    print("\nDecoding through Cortex...")
    ip = cortex.decode(ons1)
    print(f"  Decoded IP shape: {ip.shape}")

    # Encode back
    print("\nEncoding back to ONS...")
    ons_reconstructed = cortex.encode(ip)
    print(f"  Reconstructed ONS shape: {ons_reconstructed.shape}")

    # Check reconstruction error
    reconstruction_error = jnp.mean((ons1 - ons_reconstructed) ** 2)
    print(f"  Reconstruction MSE: {reconstruction_error:.6f}")

    print("✓ End-to-end pipeline works")

    return True


def main():
    """Run all integration tests."""
    print("="*60)
    print("JAX/Equinox Matisse Integration Tests")
    print("="*60)

    results = {}

    # Test CortexModel
    try:
        results['CortexModel'] = test_cortex_model()
    except Exception as e:
        print(f"\n✗ CortexModel test failed: {e}")
        results['CortexModel'] = False
        import traceback
        traceback.print_exc()

    # Test RetinaModel
    try:
        results['RetinaModel'] = test_retina_model()
    except Exception as e:
        print(f"\n✗ RetinaModel test failed: {e}")
        results['RetinaModel'] = False
        import traceback
        traceback.print_exc()

    # Test end-to-end
    try:
        results['End-to-End'] = test_end_to_end()
    except Exception as e:
        print(f"\n✗ End-to-end test failed: {e}")
        results['End-to-End'] = False
        import traceback
        traceback.print_exc()

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
