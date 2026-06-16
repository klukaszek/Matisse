"""JAX/Equinox implementation of retina lateral inhibition module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex
import numpy as np


def gaussian_kernel(kernel_size: int = 5, sig: float = 1.0) -> np.ndarray:
    """Create a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sig: Standard deviation of the Gaussian

    Returns:
        Normalized 2D Gaussian kernel
    """
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


class DefaultLateralInhibition(eqx.Module):
    """Fixed lateral inhibition using DoG (Difference of Gaussians) kernel.

    Applies a center-surround receptive field pattern using FFT-based
    convolution. This is a non-learnable module that simulates the
    lateral inhibition in the retina.
    """
    LI_kernel_in_freq_domain: Complex[Array, "fft_dim fft_dim"]
    LI_kernel: Float[Array, "kernel_size kernel_size"]
    ONS_DIM: int = eqx.field(static=True)
    FFT_DIM: int = eqx.field(static=True)
    L_PAD: int = eqx.field(static=True)
    R_PAD: int = eqx.field(static=True)
    PAD: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    LI_noise_std: float = eqx.field(static=True)

    def __init__(
        self,
        simulation_size: int = 256,
        LI_center_r: float = 0.15,
        LI_surround_r: float = 0.90,
        LI_center_surround_ratio: float = 0.91,
        LI_noise_std: float = 0.01,
    ):
        """Initialize lateral inhibition module.

        Args:
            simulation_size: Spatial dimension of simulation
            LI_center_r: Standard deviation of center Gaussian
            LI_surround_r: Standard deviation of surround Gaussian
            LI_center_surround_ratio: Weight ratio between center and surround
            LI_noise_std: Standard deviation of multiplicative noise
        """
        self.ONS_DIM = simulation_size
        self.FFT_DIM = self.ONS_DIM * 2 - 1
        self.LI_noise_std = LI_noise_std

        # Calculate kernel size
        kernel_size = int(np.ceil(LI_surround_r * 6))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        # Create DoG kernel
        center = gaussian_kernel(kernel_size=kernel_size, sig=LI_center_r)
        surround = gaussian_kernel(kernel_size=kernel_size, sig=LI_surround_r)
        kernel = center - LI_center_surround_ratio * surround

        self.LI_kernel = jnp.array(kernel)

        # Convert kernel to frequency domain
        self.PAD = kernel_size // 2
        self.L_PAD = (self.FFT_DIM - self.ONS_DIM) // 2
        self.R_PAD = (self.FFT_DIM - self.ONS_DIM) - self.L_PAD
        self.P = (self.FFT_DIM - kernel_size) // 2

        # Pad and transform kernel
        padded_LI_kernel = jnp.pad(self.LI_kernel, ((self.P, self.P), (self.P, self.P)))
        LI_kernel_in_freq_domain = jnp.fft.fft2(padded_LI_kernel)
        self.LI_kernel_in_freq_domain = jnp.fft.fftshift(LI_kernel_in_freq_domain)

    def __call__(
        self,
        pa: Float[Array, "batch channels height width"],
        *,
        key: jax.random.PRNGKey
    ) -> Float[Array, "batch channels height width"]:
        """Apply lateral inhibition with noise.

        Args:
            pa: Photoreceptor activation
            key: JAX random key for noise generation

        Returns:
            Optic nerve signal with lateral inhibition and noise
        """
        # Pad input
        pa = jnp.pad(pa, ((0, 0), (0, 0), (self.L_PAD, self.R_PAD), (self.L_PAD, self.R_PAD)))

        # Transform to frequency domain
        pa_fft = jnp.fft.fft2(pa, axes=(2, 3))
        pa_fft = jnp.fft.fftshift(pa_fft, axes=(2, 3))

        # Apply lateral inhibition kernel
        ons_fft = pa_fft * self.LI_kernel_in_freq_domain

        # Transform back to spatial domain
        ons_fft = jnp.fft.ifftshift(ons_fft, axes=(2, 3))
        ons = jnp.fft.ifft2(ons_fft, axes=(2, 3))
        ons = jnp.fft.ifftshift(ons, axes=(2, 3))

        # Crop to original size and take real part
        ons = ons[:, :, self.L_PAD:-self.R_PAD, self.L_PAD:-self.R_PAD].real

        # The reference samples one discrete noise level for the whole tensor.
        key1, key2 = jax.random.split(key)
        ratio = jax.random.randint(key1, shape=(), minval=0, maxval=11) * 0.1

        # Add noise: ons + noise * ons * std * ratio
        noise = jax.random.normal(key2, shape=ons.shape)
        noisy_ons = ons + noise * ons * self.LI_noise_std * ratio

        return noisy_ons

    def get_kernel_size(self) -> int:
        """Get the size of the lateral inhibition kernel."""
        return self.kernel_size

    def get_LI_kernel(self) -> Float[Array, "kernel_size kernel_size"]:
        """Get the lateral inhibition kernel in spatial domain."""
        return self.LI_kernel
