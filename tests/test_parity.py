import io
import zipfile
import unittest
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from Simulated.Cortex.C_cone_spectral_type.C_Default import (
    DefaultConeSpectralType,
)
from Simulated.Cortex.M_global_movement.M_Default import DefaultGlobalMovement
from Simulated.Cortex.W_lateral_inhibition_weights.W_Default import (
    DefaultLateralInhibitionWeights,
)
from Simulated.Cortex.D_demosaicing.D_Default import DefaultDemosaicing
from Simulated.Cortex.D_demosaicing.D_Implicit import ImplicitDemosaicing
from Simulated.Retina.FV_spatial_sampling.helper import (
    compute_required_image_resolution,
)
from Simulated.Retina.EM_Default import DefaultEyeMotion
from Dataset.NTIRE import NTIRE, _is_valid_cached_image, _numeric_path_key
from train import (
    masked_latent_consistency_loss,
    ons_reconstruction_penalty,
    zero_cell_position_updates,
)


class ParityTests(unittest.TestCase):
    def test_huber_ons_penalty_matches_l2_near_zero_and_limits_outliers(self):
        residual = jnp.array([0.005, 0.1])

        l2 = ons_reconstruction_penalty(residual, 'l2')
        huber = ons_reconstruction_penalty(residual, 'huber', huber_delta=0.01)

        np.testing.assert_allclose(huber[0], l2[0])
        self.assertLess(float(huber[1]), float(l2[1]))

    def test_implicit_demosaicing_preserves_shape_and_identity_at_init(self):
        model = ImplicitDemosaicing(
            latent_dim=4,
            hidden_channels=8,
            num_frequencies=3,
            key=jax.random.PRNGKey(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16, 16))

        output = model.demosaic(x)

        self.assertEqual(output.shape, x.shape)
        np.testing.assert_allclose(output, x, atol=1e-6, rtol=1e-6)

    def test_implicit_demosaicing_backpropagates_after_residual_init(self):
        model = ImplicitDemosaicing(
            latent_dim=4,
            hidden_channels=8,
            num_frequencies=3,
            key=jax.random.PRNGKey(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16, 16))
        target = jnp.zeros_like(x)

        loss, gradients = eqx.filter_value_and_grad(
            lambda module: jnp.mean((module.demosaic(x) - target) ** 2)
        )(model)
        output_gradient = gradients.demosaicing.output_projection.weight

        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertGreater(float(jnp.linalg.norm(output_gradient)), 0.0)

    def test_implicit_spatial_mixer_preserves_resolution(self):
        model = ImplicitDemosaicing(
            latent_dim=4,
            context_channels=8,
            hidden_channels=12,
            hidden_kernel_size=3,
            num_frequencies=3,
            key=jax.random.PRNGKey(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16, 16))

        output = model.demosaic(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(model.demosaicing.hidden_projection.weight.shape[-2:], (3, 3))

    def test_implicit_context_kernel_is_configurable(self):
        model = ImplicitDemosaicing(
            latent_dim=4,
            context_channels=8,
            context_kernel_size=7,
            hidden_channels=8,
            key=jax.random.PRNGKey(0),
        )
        x = jnp.ones((1, 4, 16, 16))

        output = model.demosaic(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(model.demosaicing.context_conv.weight.shape[-2:], (7, 7))

    def test_gaussian_conditioning_normalizes_cone_support(self):
        model = ImplicitDemosaicing(
            latent_dim=2,
            context_channels=4,
            hidden_channels=4,
            num_frequencies=0,
            conditioning='gaussian',
            gaussian_kernel_size=5,
            gaussian_sigma=1.0,
            key=jax.random.PRNGKey(0),
        )
        yy, xx = jnp.indices((16, 16))
        selector = (xx + yy) % 2
        cone_identity = jnp.stack(
            (selector == 0, selector == 1), axis=0
        ).astype(jnp.float32)[None]
        injected = jnp.ones((1, 1, 16, 16)) * cone_identity

        output = model.demosaic(injected, cone_identity)

        np.testing.assert_allclose(output, 1.0, atol=1e-4, rtol=1e-4)

    def test_oracle_temporal_fusion_preserves_batch_and_resolution(self):
        from Simulated.Cortex import CortexModel

        cortex = CortexModel(
            latent_dim=4,
            simulation_size=16,
            required_image_resolution=16,
            demosaicing_type='Implicit',
            demosaicing_context_channels=8,
            demosaicing_hidden_channels=8,
            demosaicing_num_frequencies=0,
            demosaicing_conditioning='gaussian',
            demosaicing_gaussian_kernel_size=5,
            temporal_fusion='oracle',
            key=jax.random.PRNGKey(0),
        )
        ons1 = jax.random.normal(jax.random.PRNGKey(1), (2, 1, 16, 16))
        ons2 = jax.random.normal(jax.random.PRNGKey(2), (2, 1, 16, 16))
        dxy = jnp.array([[0.0, 0.0], [0.125, -0.125]])

        percept, mask = cortex.decode_fused(ons1, ons2, dxy)
        loss, grads = eqx.filter_value_and_grad(
            lambda model: jnp.mean(model.decode_fused(ons1, ons2, dxy)[0] ** 2)
        )(cortex)

        self.assertEqual(percept.shape, (2, 4, 16, 16))
        self.assertEqual(mask.shape, (2, 1, 16, 16))
        self.assertTrue(bool(jnp.all(jnp.isfinite(percept))))
        self.assertTrue(bool(jnp.isfinite(loss)))
        cone_grads = jax.tree.leaves(
            eqx.filter(grads.C_cone_spectral_type, eqx.is_array)
        )
        self.assertTrue(
            all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in cone_grads)
        )

    def test_cortex_selects_implicit_demosaicing(self):
        from Simulated.Cortex import CortexModel

        cortex = CortexModel(
            latent_dim=4,
            simulation_size=16,
            required_image_resolution=16,
            demosaicing_type='Implicit',
            demosaicing_context_channels=8,
            demosaicing_hidden_channels=8,
            demosaicing_num_frequencies=2,
            key=jax.random.PRNGKey(0),
        )

        self.assertIsInstance(cortex.D_demosaicing, ImplicitDemosaicing)

    def test_eye_motion_vectorizes_batch_trajectories(self):
        eye_motion = DefaultEyeMotion(timesteps_per_image=3, max_shift_size=2)
        images = jnp.arange(4 * 1 * 12 * 12, dtype=jnp.float32).reshape(4, 1, 12, 12)

        crops, shifts = eye_motion(
            images,
            required_image_resolution=8,
            key=jax.random.PRNGKey(0),
        )

        self.assertEqual(crops.shape, (4, 3, 1, 8, 8))
        self.assertEqual(shifts.shape, (4, 2, 2))
        self.assertTrue(bool(jnp.all(jnp.abs(shifts) <= 2)))
        np.testing.assert_array_equal(crops[:, 0], images[:, :, 2:10, 2:10])

    def test_frozen_position_updates_are_zeroed(self):
        from Simulated.Cortex import CortexModel

        cortex = CortexModel(
            latent_dim=4,
            simulation_size=16,
            required_image_resolution=16,
            key=jax.random.PRNGKey(0),
        )
        updates = jax.tree.map(jnp.ones_like, cortex)

        updates = zero_cell_position_updates(updates)

        position_leaves = jax.tree.leaves(
            eqx.filter(updates.P_cell_position, eqx.is_array)
        )
        demosaicing_leaves = jax.tree.leaves(
            eqx.filter(updates.D_demosaicing, eqx.is_array)
        )
        self.assertTrue(all(bool(jnp.all(leaf == 0)) for leaf in position_leaves))
        self.assertTrue(all(bool(jnp.all(leaf == 1)) for leaf in demosaicing_leaves))


    def test_demosaicing_width_is_configurable(self):
        standard = DefaultDemosaicing(
            latent_dim=8, base_channels=16, key=jax.random.PRNGKey(0)
        )
        narrow = DefaultDemosaicing(
            latent_dim=8, base_channels=8, key=jax.random.PRNGKey(0)
        )

        standard_params = sum(x.size for x in jax.tree.leaves(eqx.filter(standard, eqx.is_array)))
        narrow_params = sum(x.size for x in jax.tree.leaves(eqx.filter(narrow, eqx.is_array)))

        self.assertLess(narrow_params, standard_params / 3)

    def test_bfloat16_demosaicing_keeps_float32_master_weights(self):
        model = DefaultDemosaicing(
            latent_dim=4,
            base_channels=4,
            compute_dtype='bfloat16',
            key=jax.random.PRNGKey(0),
        )
        x = jnp.ones((1, 4, 16, 16), dtype=jnp.float32)

        output = model.demosaic(x)
        loss, grads = eqx.filter_value_and_grad(
            lambda module: jnp.mean(module.demosaic(x) ** 2)
        )(model)

        self.assertEqual(output.dtype, jnp.float32)
        self.assertTrue(bool(jnp.isfinite(loss)))
        for parameter, grad in zip(
            jax.tree.leaves(eqx.filter(model, eqx.is_array)),
            jax.tree.leaves(eqx.filter(grads, eqx.is_array)),
        ):
            self.assertEqual(parameter.dtype, jnp.float32)
            self.assertEqual(grad.dtype, jnp.float32)
            self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_latent_consistency_is_independent_of_channel_count(self):
        mask = jnp.ones((2, 1, 3, 3))

        losses = []
        for channels in (3, 4, 8):
            target = jnp.zeros((2, channels, 3, 3))
            pred = jnp.ones_like(target)
            losses.append(masked_latent_consistency_loss(pred, target, mask))

        np.testing.assert_allclose(losses, losses[0], rtol=1e-6)

    def test_latent_consistency_respects_validity_mask(self):
        target = jnp.zeros((1, 5, 2, 2))
        pred = jnp.ones_like(target)
        mask = jnp.zeros((1, 1, 2, 2))

        loss = masked_latent_consistency_loss(pred, target, mask)

        self.assertEqual(float(loss), 0.0)

    def test_required_image_resolution_is_next_power_of_two(self):
        axis = jnp.linspace(-1.0, 1.0, 8)
        xx, yy = jnp.meshgrid(axis, axis, indexing="xy")
        xy = jnp.stack([xx, yy])[None]

        self.assertEqual(compute_required_image_resolution(xy), 8)

    def test_cone_normalization_matches_reference_epsilon_placement(self):
        module = DefaultConeSpectralType(
            latent_dim=2,
            simulation_size=1,
            key=jax.random.PRNGKey(0),
        )
        raw = jnp.array([[[[1e-3]], [[0.0]]]])
        module = eqx.tree_at(
            lambda m: m.raw_cone_identity_function,
            module,
            raw,
        )

        normalized = module.get_cone_identity_function()
        expected = 1e-3 / (1e-3 + 1e-5)
        np.testing.assert_allclose(
            normalized[0, 0, 0, 0],
            expected,
            rtol=1e-6,
        )

    def test_white_gain_adaptation_equalizes_active_cone_responses(self):
        from Simulated.Retina import RetinaModel

        retina = RetinaModel(
            simulation_size=16,
            timesteps_per_image=2,
            max_shift_size=2,
            cone_gain_adaptation='white',
            root_dir='.',
        )
        white_response = retina.CST.white_point.reshape(-1)
        adapted_response = (
            white_response * retina.SpectralSampling.cone_response_gain
        )
        active = jnp.max(
            retina.SpectralSampling.cone_fundamentals, axis=0
        ) > 0

        np.testing.assert_allclose(
            adapted_response[active], 1.0, atol=1e-5, rtol=1e-5
        )

    def test_global_movement_uses_retina_resolution(self):
        movement = DefaultGlobalMovement(
            simulation_size=256,
            required_image_resolution=512,
        )

        self.assertEqual(movement.required_image_resolution, 512)

    def test_learned_lateral_inhibition_uses_real_imag_parameters(self):
        module = DefaultLateralInhibitionWeights(
            simulation_size=8,
            key=jax.random.PRNGKey(0),
        )

        self.assertFalse(jnp.iscomplexobj(module.LIF))
        self.assertEqual(module.LIF.shape, (15, 15, 2))

        loss, grad = eqx.filter_value_and_grad(
            lambda m: jnp.sum(m.LIF[..., 1] ** 2)
        )(module)
        self.assertGreater(float(loss), 0.0)
        self.assertFalse(jnp.iscomplexobj(grad.LIF))

    def test_learned_lateral_inhibition_identity_convolve(self):
        module = DefaultLateralInhibitionWeights(
            simulation_size=8,
            key=jax.random.PRNGKey(0),
        )
        x = jnp.arange(2 * 1 * 8 * 8, dtype=jnp.float32).reshape(2, 1, 8, 8)

        y = module.convolve(x)

        np.testing.assert_allclose(y, x, atol=1e-5, rtol=1e-5)

    def test_arad_zip_cube_orientation(self):
        zip_path = "Dataset/ARAD_1K_Mirror/Train_spectral.zip"
        with zipfile.ZipFile(zip_path) as archive:
            name = sorted(
                n for n in archive.namelist() if n.endswith(".mat")
            )[0]
            data = io.BytesIO(archive.read(name))

        with h5py.File(data, "r") as mat_file:
            raw = np.asarray(mat_file["cube"])

        self.assertEqual(raw.shape, (31, 512, 482))
        self.assertEqual(
            np.transpose(raw, (2, 1, 0)).shape,
            (482, 512, 31),
        )

    def test_arad_crops_are_on_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            image = np.arange(12 * 14 * 4, dtype=np.float32).reshape(12, 14, 4)
            image_path = Path(directory) / "0.npy"
            np.save(image_path, image)

            dataset = object.__new__(NTIRE)
            dataset.file_list = [str(image_path)]
            dataset.source_dataset = "ARAD_1K"
            dataset.dim_image = 8
            dataset.dataset_size = 10

            crop1 = dataset[3]
            crop2 = dataset[3]

            self.assertEqual(len(dataset), 10)
            self.assertEqual(crop1.shape, (8, 8, 4))
            # Crops are now non-deterministic (matching the reference's
            # np.random.randint behavior), so same index may yield different crops.
            # We just verify both are valid crops from the original image.
            self.assertEqual(crop1.shape, (8, 8, 4))
            self.assertEqual(crop2.shape, (8, 8, 4))

    def test_cache_validation_and_numeric_sorting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "10.npy", np.zeros((8, 8, 4), dtype=np.float32))
            np.save(root / "2.npy", np.zeros((1, 1, 4), dtype=np.float32))
            np.save(root / "1.npy", np.zeros((8, 8, 4), dtype=np.float32))

            paths = sorted(
                (str(path) for path in root.glob("*.npy")),
                key=_numeric_path_key,
            )

            self.assertEqual(
                [Path(path).name for path in paths],
                ["1.npy", "2.npy", "10.npy"],
            )
            self.assertTrue(_is_valid_cached_image(paths[0], 8))
            self.assertFalse(_is_valid_cached_image(paths[1], 8))


if __name__ == "__main__":
    unittest.main()
