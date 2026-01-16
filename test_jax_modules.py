"""Quick test script to verify JAX/Equinox modules work correctly."""
import jax
import jax.numpy as jnp
import sys
sys.path.append('.')

from SimulatedJAX.NeuralScope import NS_cone_mosaic, NS_internal_percept
from SimulatedJAX.Cortex import (
    DefaultConeSpectralType,
    UNet,
    DefaultLateralInhibitionWeights,
    DefaultCellPosition,
)
from SimulatedJAX.Retina import DefaultLateralInhibition


def test_module(name, module, input_shape, key):
    """Test a single module."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Create input
    x = jax.random.normal(key, input_shape)
    print(f"Input shape: {x.shape}")

    # Forward pass
    try:
        if name == "DefaultLateralInhibition":
            # This module needs a key for noise
            key, subkey = jax.random.split(key)
            output = module(x, key=subkey)
        else:
            output = module(x)
        print(f"Output shape: {output.shape}")
        print(f"✓ Forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def main():
    """Run all module tests."""
    print("="*60)
    print("JAX/Equinox Module Tests")
    print("="*60)

    # Initialize random key
    key = jax.random.PRNGKey(42)
    results = {}

    # Test parameters
    batch_size = 2
    latent_dim = 8
    height, width = 256, 256

    # Test 1: NS_cone_mosaic
    key, subkey = jax.random.split(key)
    ns_cone = NS_cone_mosaic(latent_dim=latent_dim, output_dim=3, key=subkey)
    key, subkey = jax.random.split(key)
    results['NS_cone_mosaic'] = test_module(
        "NS_cone_mosaic",
        ns_cone,
        (batch_size, latent_dim, height, width),
        subkey
    )

    # Test 2: NS_internal_percept
    key, subkey = jax.random.split(key)
    ns_percept = NS_internal_percept(latent_dim=latent_dim, output_dim=3, key=subkey)
    key, subkey = jax.random.split(key)
    results['NS_internal_percept'] = test_module(
        "NS_internal_percept",
        ns_percept,
        (batch_size, latent_dim, height, width),
        subkey
    )

    # Test 3: DefaultConeSpectralType
    key, subkey = jax.random.split(key)
    cone_spectral = DefaultConeSpectralType(
        latent_dim=latent_dim,
        simulation_size=height,
        key=subkey
    )
    key, subkey = jax.random.split(key)
    # Test C_injection (1 -> latent_dim)
    print(f"\n{'='*60}")
    print(f"Testing: DefaultConeSpectralType (C_injection)")
    print(f"{'='*60}")
    x = jax.random.normal(subkey, (batch_size, 1, height, width))
    print(f"Input shape: {x.shape}")
    output = cone_spectral.C_injection(x)
    print(f"Output shape: {output.shape}")
    results['ConeSpectralType_injection'] = output.shape[1] == latent_dim

    # Test C_sampling (latent_dim -> 1)
    print(f"\n{'='*60}")
    print(f"Testing: DefaultConeSpectralType (C_sampling)")
    print(f"{'='*60}")
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batch_size, latent_dim, height, width))
    print(f"Input shape: {x.shape}")
    output = cone_spectral.C_sampling(x)
    print(f"Output shape: {output.shape}")
    results['ConeSpectralType_sampling'] = output.shape[1] == 1

    # Test 4: UNet (needs vmap for batches)
    key, subkey = jax.random.split(key)
    unet = UNet(dim_latent=latent_dim, key=subkey)
    key, subkey = jax.random.split(key)

    print(f"\n{'='*60}")
    print(f"Testing: UNet")
    print(f"{'='*60}")
    x = jax.random.normal(subkey, (batch_size, latent_dim, height, width))
    print(f"Input shape: {x.shape}")
    try:
        # Use vmap for batched inputs
        unet_batched = jax.vmap(unet)
        output = unet_batched(x)
        print(f"Output shape: {output.shape}")
        print(f"✓ Forward pass successful")
        results['UNet'] = True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        results['UNet'] = False

    # Test 5: DefaultLateralInhibitionWeights
    key, subkey = jax.random.split(key)
    li_weights = DefaultLateralInhibitionWeights(simulation_size=height, key=subkey)

    # Test deconvolve
    print(f"\n{'='*60}")
    print(f"Testing: DefaultLateralInhibitionWeights (deconvolve)")
    print(f"{'='*60}")
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batch_size, 1, height, width))
    print(f"Input shape: {x.shape}")
    output = li_weights.deconvolve(x)
    print(f"Output shape: {output.shape}")
    results['LI_weights_deconvolve'] = output.shape == x.shape

    # Test convolve
    print(f"\n{'='*60}")
    print(f"Testing: DefaultLateralInhibitionWeights (convolve)")
    print(f"{'='*60}")
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (batch_size, 1, height, width))
    print(f"Input shape: {x.shape}")
    output = li_weights.convolve(x)
    print(f"Output shape: {output.shape}")
    results['LI_weights_convolve'] = output.shape == x.shape

    # Test 6: DefaultLateralInhibition (Retina)
    li_retina = DefaultLateralInhibition(simulation_size=height)
    key, subkey = jax.random.split(key)
    results['DefaultLateralInhibition'] = test_module(
        "DefaultLateralInhibition",
        li_retina,
        (batch_size, 1, height, width),
        subkey
    )

    # Test 7: DefaultCellPosition
    key, subkey = jax.random.split(key)
    cell_position = DefaultCellPosition(
        simulation_size=height,
        latent_dim=16,
        num_layers=8,
        key=subkey
    )

    # Test get_XY_default_locations
    print(f"\n{'='*60}")
    print(f"Testing: DefaultCellPosition (get_XY)")
    print(f"{'='*60}")
    xy = cell_position.get_XY_default_locations()
    print(f"XY positions shape: {xy.shape}")
    results['CellPosition_get_xy'] = xy.shape == (1, 2, height, width)

    # Test get_UV_locations
    print(f"\n{'='*60}")
    print(f"Testing: DefaultCellPosition (get_UV)")
    print(f"{'='*60}")
    key, subkey = jax.random.split(key)
    xy_test = jax.random.normal(subkey, (batch_size, 2, height, width))
    uv = cell_position.get_UV_locations(xy_test)
    print(f"UV positions shape: {uv.shape}")
    results['CellPosition_get_uv'] = uv.shape == (batch_size, 2, height, width)

    # Test efficient_warping
    print(f"\n{'='*60}")
    print(f"Testing: DefaultCellPosition (efficient_warping)")
    print(f"{'='*60}")
    key, subkey = jax.random.split(key)
    ip1 = jax.random.normal(subkey, (batch_size, latent_dim, height, width))
    pred_dxy = jax.random.normal(subkey, (batch_size, 2))
    ip2, mask2 = cell_position.efficient_warping(ip1, pred_dxy)
    print(f"Warped IP shape: {ip2.shape}")
    print(f"Mask shape: {mask2.shape}")
    results['CellPosition_warp'] = (
        ip2.shape == (batch_size, latent_dim, height, width) and
        mask2.shape == (batch_size, 1, height, width)
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:40s} {status}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
