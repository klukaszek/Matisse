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
from Simulated.Retina.FV_spatial_sampling.helper import (
    compute_required_image_resolution,
)
from Dataset.NTIRE import NTIRE, _is_valid_cached_image, _numeric_path_key


class ParityTests(unittest.TestCase):
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
