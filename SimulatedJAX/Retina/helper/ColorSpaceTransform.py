"""JAX implementation of color space transformations."""
import jax.numpy as jnp
import pickle
import numpy as np
from jaxtyping import Array, Float
import os


class ColorSpaceTransform:
    """Color space transformations between sRGB, linear RGB, XYZ, and LMS.

    Handles conversion between color spaces using cone fundamentals and
    standard color matching functions. Supports both trichromatic and
    tetrachromatic vision.
    """

    def __init__(self, cone_fundamentals: Float[Array, "301 4"], root_dir: str = None):
        """Initialize color space transform.

        Args:
            cone_fundamentals: Cone spectral sensitivities (301 wavelengths x 4 cone types)
            root_dir: Root directory for loading CIEXYZ data
        """
        self.cone_fundamentals = cone_fundamentals
        self.Q = cone_fundamentals[:, 3]
        self.is_tetrachromatic = jnp.max(self.Q) != 0

        # Load XYZ color matching functions (CIE 1931)
        if root_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, "data", "CIEXYZ.cpkl")
        else:
            data_path = f"{root_dir}/Simulated/Retina/helper/data/CIEXYZ.cpkl"

        with open(data_path, 'rb') as f:
            CIEXYZ_dict = pickle.load(f)

        CIEXYZ = []
        for i in range(400, 701):
            CIEXYZ.append(CIEXYZ_dict[i])
        CIEXYZ = np.asarray(CIEXYZ)
        self.CIEXYZ = jnp.array(CIEXYZ)

        # Compute transformation matrices
        # CIEXYZ to LMS: (CIEXYZ^T @ CIEXYZ)^-1 @ CIEXYZ^T @ cone_fundamentals
        XYZ_T_XYZ = self.CIEXYZ.T @ self.CIEXYZ
        XYZ_T_XYZ_inv = jnp.linalg.inv(XYZ_T_XYZ)
        self.CIEXYZ_to_LMS_matrix = XYZ_T_XYZ_inv @ self.CIEXYZ.T @ cone_fundamentals

        # Linear RGB to CIEXYZ (assumes D65 illuminant)
        self.linsRGB_to_CIEXYZ_matrix = jnp.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ])

        # Linear RGB to LMS
        self.linsRGB_to_LMS_matrix = (
            self.CIEXYZ_to_LMS_matrix.T @ self.linsRGB_to_CIEXYZ_matrix
        ).T

        # Compute white point
        one = jnp.ones((1, 1, 3))
        self.white_point = one @ self.linsRGB_to_LMS_matrix

        # LMS to linear RGB (inverse, only for trichromatic)
        if not self.is_tetrachromatic:
            LMS_matrix_3x3 = self.linsRGB_to_LMS_matrix[:, :3]
            self.LMS_to_linsRGB_matrix = jnp.linalg.inv(LMS_matrix_3x3)
            # Pad with zeros (3x3 -> 4x3 matrix)
            zeros = jnp.zeros((1, 3))
            self.LMS_to_linsRGB_matrix = jnp.concatenate(
                [self.LMS_to_linsRGB_matrix, zeros], axis=0
            )
        else:
            self.LMS_to_linsRGB_matrix = None

    def sRGB_to_linsRGB(
        self,
        sRGB: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """Convert sRGB to linear RGB using gamma correction.

        Args:
            sRGB: sRGB values in range [0, 1]

        Returns:
            Linear RGB values
        """
        linRGB = sRGB / 12.92
        mask = sRGB > 0.04045
        linRGB = jnp.where(mask, ((sRGB + 0.055) / 1.055) ** 2.4, linRGB)
        return linRGB

    def linsRGB_to_sRGB(
        self,
        linRGB: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """Convert linear RGB to sRGB using inverse gamma correction.

        Args:
            linRGB: Linear RGB values

        Returns:
            sRGB values in range [0, 1]
        """
        sRGB = 12.92 * linRGB
        mask = linRGB > 0.0031308
        sRGB = jnp.where(mask, 1.055 * linRGB ** (1 / 2.4) - 0.055, sRGB)
        return sRGB

    def linsRGB_to_LMS(
        self,
        linsRGB: Float[Array, "... 3"]
    ) -> Float[Array, "... 4"]:
        """Convert linear RGB to LMS cone space.

        Args:
            linsRGB: Linear RGB values (last dimension is 3)

        Returns:
            LMS values (last dimension is 4, with Q channel)
        """
        lms = linsRGB @ self.linsRGB_to_LMS_matrix
        return lms

    def LMS_to_linsRGB(
        self,
        lms: Float[Array, "... 4"]
    ) -> Float[Array, "... 3or4"]:
        """Convert LMS cone space to linear RGB.

        Args:
            lms: LMS values (last dimension is 4)

        Returns:
            Linear RGB values (last dimension is 3), or LMS if tetrachromatic
        """
        if self.is_tetrachromatic:
            # Cannot convert tetrachromatic to RGB
            return lms
        else:
            linsRGB = lms @ self.LMS_to_linsRGB_matrix
            return linsRGB

    def sRGB_to_LMS(
        self,
        sRGB: Float[Array, "... 3"]
    ) -> Float[Array, "... 4"]:
        """Convert sRGB directly to LMS.

        Args:
            sRGB: sRGB values in range [0, 1]

        Returns:
            LMS values
        """
        linsRGB = self.sRGB_to_linsRGB(sRGB)
        return self.linsRGB_to_LMS(linsRGB)

    def LMS_to_sRGB(
        self,
        lms: Float[Array, "... 4"]
    ) -> Float[Array, "... 3or4"]:
        """Convert LMS directly to sRGB.

        Args:
            lms: LMS values

        Returns:
            sRGB values in range [0, 1], or LMS if tetrachromatic
        """
        if self.is_tetrachromatic:
            return lms
        else:
            linsRGB = self.LMS_to_linsRGB(lms)
            return self.linsRGB_to_sRGB(linsRGB)
