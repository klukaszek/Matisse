"""JAX/Equinox implementation of lateral inhibition weights module."""
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex


class DefaultLateralInhibitionWeights(eqx.Module):
    """Learnable lateral inhibition weights using FFT-based convolution/deconvolution.

    Learns a kernel in the frequency domain for efficient spatial convolution
    and deconvolution operations. The kernel is stored as a complex-valued
    parameter in the Fourier domain.
    """
    LIF: Complex[Array, "fft_dim fft_dim"]  # Lateral Inhibition Filter in frequency domain
    FFT_DIM: int = eqx.field(static=True)
    L_PAD: int = eqx.field(static=True)
    R_PAD: int = eqx.field(static=True)
    simulation_size: int = eqx.field(static=True)

    def __init__(self, simulation_size: int = 256, *, key: jax.random.PRNGKey):
        """Initialize lateral inhibition weights.

        Args:
            simulation_size: Spatial dimension of simulation
            key: JAX random key (unused, but kept for consistency)
        """
        self.simulation_size = simulation_size
        self.FFT_DIM = simulation_size * 2 - 1

        # Left and right padding for the kernel
        self.L_PAD = (self.FFT_DIM - simulation_size) // 2
        self.R_PAD = (self.FFT_DIM - simulation_size) - self.L_PAD

        # Initialize kernel as delta function in spatial domain
        kernel = jnp.zeros((self.FFT_DIM, self.FFT_DIM))
        kernel = kernel.at[self.FFT_DIM // 2, self.FFT_DIM // 2].set(1.0)

        # Transform to frequency domain
        kernel_fft = jnp.fft.fft2(kernel)
        kernel_fft_shift = jnp.fft.fftshift(kernel_fft)

        # Store as complex array (this will be a learnable parameter)
        self.LIF = kernel_fft_shift

    def deconvolve(
        self,
        ons: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        """Deconvolve optic nerve signal to get photoreceptor activation.

        Performs division in the frequency domain (deconvolution in spatial domain).

        Args:
            ons: Optic nerve signal

        Returns:
            Photoreceptor activation (deconvolved signal)
        """
        # Pad input
        ons = jnp.pad(ons, ((0, 0), (0, 0), (self.L_PAD, self.R_PAD), (self.L_PAD, self.R_PAD)))

        # Transform to frequency domain
        ons_fft = jnp.fft.fft2(ons, axes=(2, 3))
        ons_fft = jnp.fft.fftshift(ons_fft, axes=(2, 3))

        # Deconvolution: division in frequency domain
        # Add small epsilon to avoid division by zero
        pa_fft = ons_fft / (self.LIF + 1e-10)

        # Transform back to spatial domain
        pa_fft = jnp.fft.ifftshift(pa_fft, axes=(2, 3))
        pa = jnp.fft.ifft2(pa_fft, axes=(2, 3))
        pa = jnp.fft.fftshift(pa, axes=(2, 3))

        # Crop to original size and take real part
        pa = pa[:, :, self.L_PAD:-self.R_PAD, self.L_PAD:-self.R_PAD].real

        return pa

    def convolve(
        self,
        pa: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        """Convolve photoreceptor activation to get optic nerve signal.

        Performs multiplication in the frequency domain (convolution in spatial domain).

        Args:
            pa: Photoreceptor activation

        Returns:
            Optic nerve signal (convolved signal)
        """
        # Pad input
        pa = jnp.pad(pa, ((0, 0), (0, 0), (self.L_PAD, self.R_PAD), (self.L_PAD, self.R_PAD)))

        # Transform to frequency domain
        pa_fft = jnp.fft.fft2(pa, axes=(2, 3))
        pa_fft = jnp.fft.fftshift(pa_fft, axes=(2, 3))

        # Convolution: multiplication in frequency domain
        ons_fft = pa_fft * self.LIF

        # Transform back to spatial domain
        ons_fft = jnp.fft.ifftshift(ons_fft, axes=(2, 3))
        ons = jnp.fft.ifft2(ons_fft, axes=(2, 3))
        ons = jnp.fft.ifftshift(ons, axes=(2, 3))

        # Crop to original size and take real part
        ons = ons[:, :, self.L_PAD:-self.R_PAD, self.L_PAD:-self.R_PAD].real

        return ons

    def get_predicted_kernel(self, kernel_size: int) -> Float[Array, "kernel_size kernel_size"]:
        """Get the spatial domain kernel for visualization.

        Args:
            kernel_size: Size of the kernel to extract

        Returns:
            Spatial domain kernel
        """
        # Transform back to spatial domain
        kernel_spatial = jnp.fft.ifft2(jnp.fft.ifftshift(self.LIF))
        kernel_spatial = kernel_spatial.real

        # Crop to desired size
        padding = (self.FFT_DIM - kernel_size) // 2
        pred_kernel = kernel_spatial[padding:-padding, padding:-padding]

        return pred_kernel

    def __call__(
        self,
        x: Float[Array, "batch channels height width"],
        operation: str = "deconvolve"
    ) -> Float[Array, "batch channels height width"]:
        """Convenience forward method supporting both operations.

        Args:
            x: Input tensor
            operation: Either "deconvolve" or "convolve"

        Returns:
            Result of the specified operation
        """
        if operation == "deconvolve":
            return self.deconvolve(x)
        elif operation == "convolve":
            return self.convolve(x)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'deconvolve' or 'convolve'.")
