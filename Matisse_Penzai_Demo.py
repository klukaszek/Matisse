import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import penzai
    from penzai import pz
    import treescope
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import os
    import sys
    import io
    import base64
    import marimo as mo
    import h5py
    import scipy.ndimage
    import imageio.v3 as iio

    # Add project root to path
    sys.path.append('.')

    from Simulated.Retina import RetinaModel
    from Simulated.Cortex import CortexModel
    from Simulated.Retina.helper.ColorSpaceTransform import ColorSpaceTransform
    return (
        ColorSpaceTransform,
        CortexModel,
        RetinaModel,
        base64,
        eqx,
        h5py,
        iio,
        io,
        jax,
        jnp,
        mcolors,
        mo,
        np,
        os,
        penzai,
        plt,
        pz,
        scipy,
        sys,
        treescope,
    )


@app.cell
def _(base64, iio, io, mo, np, treescope):
    # Helper function to render with ArrayAutovisualizer for pretty array printing
    def render_treescope(obj, height="600px"):
        """Render object with treescope ArrayAutovisualizer in a marimo iframe."""
        with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
            html = treescope.render_to_html(obj)
        return mo.iframe(html, height=height)

    # Helper to fix orientation for display
    def orient(img):
        """Rotate 90 deg clockwise then flip horizontally to match expected orientation."""
        return np.fliplr(np.rot90(img, k=3))

    # Helper to create animated GIF from list of images and display in marimo
    def create_gif(images, fps=2, loop=0):
        """
        Create an animated GIF from a list of numpy arrays and return marimo Html.

        Args:
            images: List of numpy arrays (H, W, 3) with values in [0, 1]
            fps: Frames per second
            loop: Number of loops (0 = infinite)

        Returns:
            mo.Html displaying the animated GIF
        """
        # Convert to uint8
        frames = [(img * 255).astype(np.uint8) for img in images]

        # Write GIF to bytes buffer
        buf = io.BytesIO()
        iio.imwrite(buf, frames, extension=".gif", fps=fps, loop=loop)
        buf.seek(0)

        # Encode to base64 and create HTML
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        html = f'<img src="data:image/gif;base64,{b64}" style="max-width:100%;" />'
        return mo.Html(html)

    return create_gif, orient, render_treescope


@app.cell
def _(mo):
    mo.md(
        """
        # Matisse: Retina-Cortex Model Interpretability

        **A colorimetry and HCI research notebook for understanding simulated visual processing**

        This notebook provides deep inspection of the Matisse model internals for:
        - **Color science**: Spectral sensitivities, cone mosaics, chromatic adaptation
        - **Visual neuroscience**: Foveation, lateral inhibition, eye movements
        - **HCI applications**: Display quality evaluation, CVD simulation, environmental lighting effects

        ---

        ## Notebook Structure

        **Part A: Retina Model** - Forward visual processing simulation
        1. Color Space Foundations (LMS, XYZ, sRGB transformations)
        2. Cone Spectral Sensitivities & Mosaic Distribution
        3. Spatial Sampling & Foveation (MIP-mapped sampling)
        4. Lateral Inhibition (Center-surround, DoG kernel)
        5. Complete Retina Pipeline Visualization

        **Part B: Cortex Model** - Learned inverse model
        1. Learned Cone Identity Function
        2. Learned LI Deconvolution Kernel
        3. Eye Movement Estimation (Coarse-to-fine pyramid)
        4. RealNVP Coordinate Transforms
        5. Internal Percept & Multi-timestep Integration

        **Part C: End-to-End Analysis**
        1. Validation on Real Hyperspectral Images
        2. Quality Metrics & Reconstruction Fidelity
        """
    )
    return


@app.cell
def _(mo):
    mo.md("---\n# Configuration & Model Initialization")
    return


@app.cell
def _(CortexModel, RetinaModel, eqx, jax, mo, os):
    # Configuration
    key = jax.random.PRNGKey(42)

    # Retina Configuration
    SIMULATION_SIZE = 256
    TIMESTEPS_PER_IMAGE = 4  # Multiple timesteps for eye movement analysis
    MAX_SHIFT_SIZE = 15
    LATENT_DIM = 8

    print("Initializing RetinaModel...")
    retina = RetinaModel(
        simulation_size=SIMULATION_SIZE,
        timesteps_per_image=TIMESTEPS_PER_IMAGE,
        max_shift_size=MAX_SHIFT_SIZE,
        cone_types_str='LMS',
        cone_distribution_type='Human',
        simulating_tetra=False
    )

    # Initialize random Cortex for comparison
    print("Initializing random CortexModel...")
    key, _subkey = jax.random.split(key)
    cortex_random = CortexModel(
        latent_dim=LATENT_DIM,
        simulation_size=SIMULATION_SIZE,
        simulating_tetra=False,
        key=_subkey
    )

    # Load trained Cortex if available
    _model_path = "model_100000.eqx"
    if os.path.exists(_model_path):
        print(f"Loading trained CortexModel from {_model_path}...")
        _key = jax.random.PRNGKey(0)
        cortex_trained = CortexModel(
            latent_dim=LATENT_DIM,
            simulation_size=SIMULATION_SIZE,
            simulating_tetra=False,
            key=_key
        )
        cortex_trained = eqx.tree_deserialise_leaves(_model_path, cortex_trained)
        print("Trained model loaded successfully!")
    else:
        print(f"Trained model not found at {_model_path}")
        cortex_trained = None

    config_summary = mo.md(f"""
    **Configuration Summary:**
    - Simulation Size: {SIMULATION_SIZE}x{SIMULATION_SIZE}
    - Timesteps per Image: {TIMESTEPS_PER_IMAGE}
    - Max Eye Shift: {MAX_SHIFT_SIZE} pixels
    - Latent Dimension: {LATENT_DIM}
    - Trained Model: {'Loaded' if cortex_trained else 'Not Found'}
    """)
    config_summary
    return (
        LATENT_DIM,
        MAX_SHIFT_SIZE,
        SIMULATION_SIZE,
        TIMESTEPS_PER_IMAGE,
        config_summary,
        cortex_random,
        cortex_trained,
        key,
        retina,
    )


# =============================================================================
# PART A: RETINA MODEL
# =============================================================================


@app.cell
def _(mo):
    mo.md(
        """
        ---
        # Part A: Retina Model Internals

        The retina model simulates forward visual processing:

        ```
        Input Image → Eye Motion → Spatial Sampling (Foveation) →
        Spectral Sampling (Cone Mosaic) → Lateral Inhibition → Optic Nerve Signal
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## A.1 Model Architecture Overview")
    return


@app.cell
def _(render_treescope, retina):
    render_treescope(retina, height="700px")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A.2 Color Space Foundations

        The ColorSpaceTransform module handles conversions between:
        - **sRGB**: Standard display color space (gamma-corrected)
        - **Linear RGB**: Linear light values (no gamma)
        - **CIE XYZ**: Device-independent color space
        - **LMS**: Cone response space (Long, Medium, Short wavelength cones)

        These transformations are critical for accurate color rendering in HCI applications.
        """
    )
    return


@app.cell
def _(mo, np, plt, render_treescope, retina):
    # Extract color space transformation matrices
    _cst = retina.CST

    _linsrgb_to_lms = np.array(_cst.linsRGB_to_LMS_matrix)
    _lms_to_linsrgb = np.array(_cst.LMS_to_linsRGB_matrix)
    _xyz_to_lms = np.array(_cst.CIEXYZ_to_LMS_matrix)

    # Create visualization
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    # Linear RGB to LMS
    _im0 = _axes[0].imshow(_linsrgb_to_lms, cmap='RdBu', vmin=-1, vmax=1)
    _axes[0].set_title("Linear RGB → LMS")
    _axes[0].set_xticks([0, 1, 2])
    _axes[0].set_xticklabels(['R', 'G', 'B'])
    _axes[0].set_yticks([0, 1, 2, 3])
    _axes[0].set_yticklabels(['L', 'M', 'S', 'Q'])
    plt.colorbar(_im0, ax=_axes[0])

    # Add values as text
    for _i in range(_linsrgb_to_lms.shape[0]):
        for _j in range(_linsrgb_to_lms.shape[1]):
            _axes[0].text(_j, _i, f'{_linsrgb_to_lms[_i, _j]:.2f}',
                         ha='center', va='center', fontsize=9)

    # LMS to Linear RGB
    _im1 = _axes[1].imshow(_lms_to_linsrgb[:, :3], cmap='RdBu', vmin=-2, vmax=2)
    _axes[1].set_title("LMS → Linear RGB")
    _axes[1].set_xticks([0, 1, 2])
    _axes[1].set_xticklabels(['L', 'M', 'S'])
    _axes[1].set_yticks([0, 1, 2])
    _axes[1].set_yticklabels(['R', 'G', 'B'])
    plt.colorbar(_im1, ax=_axes[1])

    for _i in range(3):
        for _j in range(3):
            _axes[1].text(_j, _i, f'{_lms_to_linsrgb[_i, _j]:.2f}',
                         ha='center', va='center', fontsize=9)

    # XYZ to LMS (Hunt-Pointer-Estevez)
    _im2 = _axes[2].imshow(_xyz_to_lms[:3, :], cmap='RdBu', vmin=-1, vmax=1)
    _axes[2].set_title("CIE XYZ → LMS (HPE)")
    _axes[2].set_xticks([0, 1, 2])
    _axes[2].set_xticklabels(['X', 'Y', 'Z'])
    _axes[2].set_yticks([0, 1, 2])
    _axes[2].set_yticklabels(['L', 'M', 'S'])
    plt.colorbar(_im2, ax=_axes[2])

    for _i in range(3):
        for _j in range(3):
            _axes[2].text(_j, _i, f'{_xyz_to_lms[_i, _j]:.2f}',
                         ha='center', va='center', fontsize=9)

    plt.suptitle("Color Space Transformation Matrices", fontsize=14, fontweight='bold')
    plt.tight_layout()

    _cst_output = mo.vstack([
        plt.gca(),
        mo.md("### Matrix Details (Treescope)"),
        render_treescope({
            "linsRGB_to_LMS": _linsrgb_to_lms,
            "LMS_to_linsRGB": _lms_to_linsrgb,
            "CIEXYZ_to_LMS": _xyz_to_lms,
            "white_point": np.array(_cst.white_point),
        }, height="350px")
    ])
    _cst_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A.3 Cone Spectral Sensitivities

        The cone fundamentals define how each cone type responds to different wavelengths.
        Based on the Neitz & Neitz model with physiological parameters:
        - **L cones**: Peak ~560nm (red-sensitive)
        - **M cones**: Peak ~530nm (green-sensitive)
        - **S cones**: Peak ~419nm (blue-sensitive)

        Understanding these is critical for CVD simulation and display calibration.
        """
    )
    return


@app.cell
def _(mo, np, plt, retina):
    # Get cone fundamentals
    fundamentals = retina.SpectralSampling.get_cone_fundamentals()
    wavelengths = np.array(list(range(400, 701)))

    # Create detailed visualization
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard plot
    _axes[0].plot(wavelengths, fundamentals[:, 0], 'r-', linewidth=2, label='L Cone (560nm)')
    _axes[0].plot(wavelengths, fundamentals[:, 1], 'g-', linewidth=2, label='M Cone (530nm)')
    _axes[0].plot(wavelengths, fundamentals[:, 2], 'b-', linewidth=2, label='S Cone (419nm)')
    if fundamentals.shape[1] > 3:
        _axes[0].plot(wavelengths, fundamentals[:, 3], 'm--', linewidth=2, label='Q Cone (tetrachromat)')
    _axes[0].set_xlabel("Wavelength (nm)", fontsize=12)
    _axes[0].set_ylabel("Normalized Sensitivity", fontsize=12)
    _axes[0].set_title("Cone Spectral Sensitivities", fontsize=14)
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)
    _axes[0].set_xlim(400, 700)

    # Log scale to see tails
    _axes[1].semilogy(wavelengths, fundamentals[:, 0] + 1e-10, 'r-', linewidth=2, label='L Cone')
    _axes[1].semilogy(wavelengths, fundamentals[:, 1] + 1e-10, 'g-', linewidth=2, label='M Cone')
    _axes[1].semilogy(wavelengths, fundamentals[:, 2] + 1e-10, 'b-', linewidth=2, label='S Cone')
    _axes[1].set_xlabel("Wavelength (nm)", fontsize=12)
    _axes[1].set_ylabel("Log Sensitivity", fontsize=12)
    _axes[1].set_title("Log Scale (Tail Visibility)", fontsize=14)
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)
    _axes[1].set_xlim(400, 700)

    plt.tight_layout()

    # Compute statistics
    _l_peak = wavelengths[np.argmax(fundamentals[:, 0])]
    _m_peak = wavelengths[np.argmax(fundamentals[:, 1])]
    _s_peak = wavelengths[np.argmax(fundamentals[:, 2])]

    _fund_stats = mo.md(f"""
    **Peak Sensitivities:**
    - L Cone: {_l_peak}nm (max={fundamentals[:, 0].max():.4f})
    - M Cone: {_m_peak}nm (max={fundamentals[:, 1].max():.4f})
    - S Cone: {_s_peak}nm (max={fundamentals[:, 2].max():.4f})
    - L-M Separation: {_l_peak - _m_peak}nm
    """)

    mo.vstack([plt.gca(), _fund_stats])
    return fundamentals, wavelengths


@app.cell
def _(mo):
    mo.md(
        """
        ## A.4 Cone Mosaic Distribution

        The spatial arrangement of cones varies across the retina:
        - **Fovea**: Dense packing, mostly L and M cones
        - **Periphery**: Sparser, more S cones relatively

        The L:M ratio varies between individuals (~1.5:1 to 3:1).
        """
    )
    return


@app.cell
def _(mo, np, plt, retina):
    # Get cone mosaic
    mosaic = retina.get_cone_mosaic()[0]  # (4, H, W)
    mosaic_np = np.array(mosaic)

    # Create comprehensive visualization
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))

    # Full mosaic (RGB = LMS)
    _mosaic_rgb = mosaic_np[:3, :, :].transpose(1, 2, 0)
    _axes[0, 0].imshow(_mosaic_rgb)
    _axes[0, 0].set_title("Cone Mosaic (L=R, M=G, S=B)")
    _axes[0, 0].axis('off')

    # Individual cone channels
    _axes[0, 1].imshow(mosaic_np[0], cmap='Reds')
    _axes[0, 1].set_title("L Cones")
    _axes[0, 1].axis('off')

    _axes[0, 2].imshow(mosaic_np[1], cmap='Greens')
    _axes[0, 2].set_title("M Cones")
    _axes[0, 2].axis('off')

    _axes[1, 0].imshow(mosaic_np[2], cmap='Blues')
    _axes[1, 0].set_title("S Cones")
    _axes[1, 0].axis('off')

    # Center crop (fovea detail)
    _center = mosaic_np.shape[1] // 2
    _crop_size = 64
    _fovea_crop = mosaic_np[:3, _center-_crop_size:_center+_crop_size,
                            _center-_crop_size:_center+_crop_size].transpose(1, 2, 0)
    _axes[1, 1].imshow(_fovea_crop, interpolation='nearest')
    _axes[1, 1].set_title("Fovea Detail (128x128 center)")
    _axes[1, 1].axis('off')

    # Cone density histogram
    _l_density = mosaic_np[0].mean()
    _m_density = mosaic_np[1].mean()
    _s_density = mosaic_np[2].mean()
    _total = _l_density + _m_density + _s_density

    _bars = _axes[1, 2].bar(['L', 'M', 'S'], [_l_density/_total, _m_density/_total, _s_density/_total],
                           color=['red', 'green', 'blue'], alpha=0.7)
    _axes[1, 2].set_ylabel("Proportion")
    _axes[1, 2].set_title(f"Cone Ratios (L:M = {_l_density/_m_density:.2f}:1)")
    _axes[1, 2].set_ylim(0, 0.7)

    plt.suptitle("Cone Mosaic Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    _mosaic_stats = mo.md(f"""
    **Mosaic Statistics:**
    - Shape: {mosaic_np.shape}
    - L Cone Density: {_l_density:.4f} ({100*_l_density/_total:.1f}%)
    - M Cone Density: {_m_density:.4f} ({100*_m_density/_total:.1f}%)
    - S Cone Density: {_s_density:.4f} ({100*_s_density/_total:.1f}%)
    - L:M:S Ratio: {_l_density/_s_density:.1f}:{_m_density/_s_density:.1f}:1
    """)

    mo.vstack([plt.gca(), _mosaic_stats])
    return mosaic, mosaic_np


@app.cell
def _(mo):
    mo.md(
        """
        ## A.5 Mosaic-Weighted Spectral Response

        The effective spectral sensitivity depends on both cone fundamentals AND mosaic composition.
        This is what determines color discrimination ability for a given retina configuration.
        """
    )
    return


@app.cell
def _(fundamentals, mo, mosaic_np, np, plt, pz, render_treescope, wavelengths):
    # Compute mosaic-weighted response
    _cone_weights = mosaic_np.mean(axis=(1, 2))
    _cone_weights = _cone_weights / (_cone_weights.sum() + 1e-12)

    _fund_np = np.array(fundamentals)
    _tuning = _fund_np * _cone_weights[None, :]
    _tuning_total = _tuning.sum(axis=1)

    _cone_labels = ["L", "M", "S", "Q"]

    # Create visualization
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual weighted responses
    for _idx, (_label, _color) in enumerate(zip(_cone_labels[:3], ['red', 'green', 'blue'])):
        _axes[0].plot(wavelengths, _tuning[:, _idx], color=_color, linewidth=2,
                     label=f"{_label} (w={_cone_weights[_idx]:.3f})")
    _axes[0].plot(wavelengths, _tuning_total, 'k--', linewidth=2, label="Total Response")
    _axes[0].set_xlabel("Wavelength (nm)")
    _axes[0].set_ylabel("Weighted Response")
    _axes[0].set_title("Mosaic-Weighted Spectral Sensitivity")
    _axes[0].legend()
    _axes[0].grid(True, alpha=0.3)

    # Normalized comparison
    _axes[1].plot(wavelengths, _fund_np[:, 0], 'r-', alpha=0.5, label='L (raw)')
    _axes[1].plot(wavelengths, _fund_np[:, 1], 'g-', alpha=0.5, label='M (raw)')
    _axes[1].plot(wavelengths, _fund_np[:, 2], 'b-', alpha=0.5, label='S (raw)')
    _axes[1].plot(wavelengths, _tuning_total / _tuning_total.max(), 'k-', linewidth=2,
                 label='Total (normalized)')
    _axes[1].set_xlabel("Wavelength (nm)")
    _axes[1].set_ylabel("Normalized Response")
    _axes[1].set_title("Raw vs Mosaic-Weighted")
    _axes[1].legend()
    _axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Create named arrays for treescope
    _tuning_named = pz.nx.wrap(_tuning, "wavelength", "cone")
    _weights_named = pz.nx.wrap(_cone_weights, "cone")

    _spectral_output = mo.vstack([
        plt.gca(),
        mo.md("### Penzai Named Arrays"),
        render_treescope({
            "cone_weights": _weights_named,
            "weighted_tuning": _tuning_named,
        }, height="350px"),
    ])
    _spectral_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A.6 Lateral Inhibition (Center-Surround)

        The retina performs local contrast enhancement via center-surround receptive fields,
        implemented as a Difference of Gaussians (DoG) filter. This is critical for edge detection
        and spatial frequency processing.
        """
    )
    return


@app.cell
def _(mo, np, plt, render_treescope, retina):
    # Get LI kernel (no arguments - kernel size is determined at init)
    _li = retina.LateralInhibition
    li_kernel = np.array(_li.get_LI_kernel())
    _kernel_size = _li.get_kernel_size()

    # Visualization
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    # 2D kernel
    _im = _axes[0].imshow(li_kernel, cmap='RdBu', vmin=-li_kernel.max(), vmax=li_kernel.max())
    _axes[0].set_title("DoG Kernel (2D)")
    plt.colorbar(_im, ax=_axes[0])

    # 1D cross-section
    _center = _kernel_size // 2
    _profile = li_kernel[_center, :]
    _x = np.arange(_kernel_size) - _center
    _axes[1].plot(_x, _profile, 'b-', linewidth=2)
    _axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    _axes[1].fill_between(_x, _profile, 0, where=_profile > 0, alpha=0.3, color='blue')
    _axes[1].fill_between(_x, _profile, 0, where=_profile < 0, alpha=0.3, color='red')
    _axes[1].set_xlabel("Position (pixels)")
    _axes[1].set_ylabel("Kernel Value")
    _axes[1].set_title("Center Cross-Section")
    _axes[1].grid(True, alpha=0.3)

    # Frequency response (MTF)
    _fft_kernel = np.fft.fft2(li_kernel)
    _fft_kernel = np.fft.fftshift(_fft_kernel)
    _mtf = np.abs(_fft_kernel)
    _mtf = _mtf / _mtf.max()

    _axes[2].imshow(np.log10(_mtf + 1e-6), cmap='viridis')
    _axes[2].set_title("MTF (Log Frequency Response)")
    _axes[2].axis('off')

    plt.suptitle("Lateral Inhibition Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()

    _li_stats = mo.md(f"""
    **Lateral Inhibition Parameters:**
    - Kernel Size: {_kernel_size}x{_kernel_size}
    - Noise Std: {_li.LI_noise_std}
    - FFT Dimension: {_li.FFT_DIM}
    """)

    mo.vstack([
        plt.gca(),
        _li_stats,
        mo.md("### Kernel Details"),
        render_treescope({"LI_kernel": li_kernel}, height="250px")
    ])
    return (li_kernel,)


# =============================================================================
# PART B: CORTEX MODEL
# =============================================================================


@app.cell
def _(mo):
    mo.md(
        """
        ---
        # Part B: Cortex Model Internals

        The cortex model learns to invert retinal processing:

        ```
        ONS → Deconvolve LI → Inject Cone Identity → Demosaic (UNet) → Internal Percept
        ```

        Key learned components:
        - **Cone Identity Function**: Which cone type is at each location
        - **LI Deconvolution Kernel**: Inverse of lateral inhibition
        - **Eye Movement Estimation**: Predicts gaze shifts between frames
        - **RealNVP Coordinate Transform**: Maps regular grid to cone positions
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## B.1 Cortex Architecture Overview")
    return


@app.cell
def _(cortex_trained, mo, render_treescope):
    if cortex_trained is not None:
        _b1_output = render_treescope(cortex_trained, height="800px")
    else:
        _b1_output = mo.md("*Trained cortex model not available.*")
    _b1_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## B.2 Learned Cone Identity Function

        The cortex learns a latent representation of cone identity at each spatial location.
        This is a (latent_dim, H, W) tensor that encodes which cone type is present.
        """
    )
    return


@app.cell
def _(cortex_trained, mo, np, plt, render_treescope):
    if cortex_trained is not None:
        # Get learned cone identity
        _cone_id = cortex_trained.C_cone_spectral_type.get_cone_identity_function()
        _cone_id_np = np.array(_cone_id[0])  # (latent_dim, H, W)

        _latent_dim = _cone_id_np.shape[0]
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))

        for _i in range(_latent_dim):
            _ax = _axes[_i // 4, _i % 4]
            _im = _ax.imshow(_cone_id_np[_i], cmap='viridis')
            _ax.set_title(f"Latent Channel {_i}")
            _ax.axis('off')
            plt.colorbar(_im, ax=_ax, fraction=0.046)

        plt.suptitle("Learned Cone Identity Function (8 Latent Channels)", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _b2_output = mo.vstack([
            plt.gca(),
            mo.md("### Cone Identity Statistics"),
            mo.md(f"- Shape: {_cone_id_np.shape}"),
            mo.md(f"- Value Range: [{_cone_id_np.min():.4f}, {_cone_id_np.max():.4f}]"),
            render_treescope({"cone_identity": _cone_id}, height="300px")
        ])
    else:
        _b2_output = mo.md("*Trained cortex model not available.*")
    _b2_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## B.3 Learned vs Ground Truth LI Kernel

        The cortex learns to deconvolve the lateral inhibition applied by the retina.
        Comparing the learned kernel to the ground truth reveals how well the model
        has recovered the retinal processing.
        """
    )
    return


@app.cell
def _(cortex_trained, li_kernel, mo, np, plt, retina):
    if cortex_trained is not None:
        # Get kernel size from retina's LI module
        _kernel_size = retina.LateralInhibition.get_kernel_size()

        # Get learned LI kernel at same size
        _learned_kernel = np.array(
            cortex_trained.W_lateral_inhibition_weights.get_predicted_kernel(_kernel_size)
        )

        # Ground truth from retina
        _gt_kernel = li_kernel

        # Normalize for comparison
        _learned_norm = _learned_kernel / (np.abs(_learned_kernel).max() + 1e-10)
        _gt_norm = _gt_kernel / (np.abs(_gt_kernel).max() + 1e-10)

        _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))

        # 2D kernels
        _vmax = max(np.abs(_gt_norm).max(), np.abs(_learned_norm).max())
        _axes[0, 0].imshow(_gt_norm, cmap='RdBu', vmin=-_vmax, vmax=_vmax)
        _axes[0, 0].set_title("Ground Truth LI Kernel")
        _axes[0, 0].axis('off')

        _axes[0, 1].imshow(_learned_norm, cmap='RdBu', vmin=-_vmax, vmax=_vmax)
        _axes[0, 1].set_title("Learned Deconvolution Kernel")
        _axes[0, 1].axis('off')

        _diff = _learned_norm - _gt_norm
        _im = _axes[0, 2].imshow(_diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        _axes[0, 2].set_title("Difference (Learned - GT)")
        _axes[0, 2].axis('off')
        plt.colorbar(_im, ax=_axes[0, 2])

        # 1D cross-sections
        _center = _kernel_size // 2
        _x = np.arange(_kernel_size) - _center

        _axes[1, 0].plot(_x, _gt_norm[_center, :], 'b-', linewidth=2, label='Ground Truth')
        _axes[1, 0].plot(_x, _learned_norm[_center, :], 'r--', linewidth=2, label='Learned')
        _axes[1, 0].set_xlabel("Position (pixels)")
        _axes[1, 0].set_ylabel("Normalized Value")
        _axes[1, 0].set_title("Horizontal Cross-Section")
        _axes[1, 0].legend()
        _axes[1, 0].grid(True, alpha=0.3)

        _axes[1, 1].plot(_x, _gt_norm[:, _center], 'b-', linewidth=2, label='Ground Truth')
        _axes[1, 1].plot(_x, _learned_norm[:, _center], 'r--', linewidth=2, label='Learned')
        _axes[1, 1].set_xlabel("Position (pixels)")
        _axes[1, 1].set_ylabel("Normalized Value")
        _axes[1, 1].set_title("Vertical Cross-Section")
        _axes[1, 1].legend()
        _axes[1, 1].grid(True, alpha=0.3)

        # Frequency response comparison
        _gt_fft = np.abs(np.fft.fftshift(np.fft.fft2(_gt_norm)))
        _learned_fft = np.abs(np.fft.fftshift(np.fft.fft2(_learned_norm)))

        _axes[1, 2].plot(_gt_fft[_center, :], 'b-', linewidth=2, label='GT MTF')
        _axes[1, 2].plot(_learned_fft[_center, :], 'r--', linewidth=2, label='Learned MTF')
        _axes[1, 2].set_xlabel("Frequency")
        _axes[1, 2].set_ylabel("Magnitude")
        _axes[1, 2].set_title("MTF Comparison")
        _axes[1, 2].legend()
        _axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle("LI Kernel: Ground Truth vs Learned", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Compute correlation
        _corr = np.corrcoef(_gt_norm.flatten(), _learned_norm.flatten())[0, 1]
        _mse = np.mean((_gt_norm - _learned_norm) ** 2)

        _kernel_stats = mo.md(f"""
        **Kernel Comparison Metrics:**
        - Pearson Correlation: {_corr:.4f}
        - MSE: {_mse:.6f}
        """)

        _b3_output = mo.vstack([plt.gca(), _kernel_stats])
    else:
        _b3_output = mo.md("*Trained cortex model not available.*")
    _b3_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## B.4 RealNVP Coordinate Transform

        The cortex uses a RealNVP normalizing flow to learn the mapping from
        a regular grid to the actual cone positions. This is invertible,
        allowing both forward (grid → cones) and inverse (cones → grid) transforms.
        """
    )
    return


@app.cell
def _(SIMULATION_SIZE, cortex_trained, mo, np, plt):
    if cortex_trained is not None:
        # Get cell positions
        _p = cortex_trained.P_cell_position
        _xy_locs = np.array(_p.get_XY_default_locations())  # (H*W, 2)

        # Reshape to grid
        _xy_grid = _xy_locs.reshape(SIMULATION_SIZE, SIMULATION_SIZE, 2)

        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

        # X coordinates
        _im0 = _axes[0].imshow(_xy_grid[:, :, 0], cmap='viridis')
        _axes[0].set_title("Learned X Coordinates")
        _axes[0].axis('off')
        plt.colorbar(_im0, ax=_axes[0])

        # Y coordinates
        _im1 = _axes[1].imshow(_xy_grid[:, :, 1], cmap='viridis')
        _axes[1].set_title("Learned Y Coordinates")
        _axes[1].axis('off')
        plt.colorbar(_im1, ax=_axes[1])

        # Displacement from regular grid
        _regular_x = np.linspace(-1, 1, SIMULATION_SIZE)
        _regular_y = np.linspace(-1, 1, SIMULATION_SIZE)
        _regular_grid = np.stack(np.meshgrid(_regular_x, _regular_y), axis=-1)

        _displacement = np.sqrt(np.sum((_xy_grid - _regular_grid) ** 2, axis=-1))
        _im2 = _axes[2].imshow(_displacement, cmap='hot')
        _axes[2].set_title("Displacement from Regular Grid")
        _axes[2].axis('off')
        plt.colorbar(_im2, ax=_axes[2])

        plt.suptitle("RealNVP Learned Coordinate Transform", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _coord_stats = mo.md(f"""
        **Coordinate Transform Statistics:**
        - Grid Size: {SIMULATION_SIZE}x{SIMULATION_SIZE}
        - X Range: [{_xy_grid[:,:,0].min():.4f}, {_xy_grid[:,:,0].max():.4f}]
        - Y Range: [{_xy_grid[:,:,1].min():.4f}, {_xy_grid[:,:,1].max():.4f}]
        - Mean Displacement: {_displacement.mean():.4f}
        - Max Displacement: {_displacement.max():.4f}
        """)

        _b4_output = mo.vstack([plt.gca(), _coord_stats])
    else:
        _b4_output = mo.md("*Trained cortex model not available.*")
    _b4_output
    return


# =============================================================================
# PART C: END-TO-END ANALYSIS
# =============================================================================


@app.cell
def _(mo):
    mo.md(
        """
        ---
        # Part C: End-to-End Pipeline Analysis

        Now we run complete forward+inverse passes to analyze:
        1. Retina processing stages
        2. Eye movement tracking across timesteps
        3. Cortex decoding quality
        4. Multi-timestep integration
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## C.1 Validation Image Selection")
    return


@app.cell
def _(mo):
    # Image selector for validation set
    image_selector = mo.ui.slider(
        start=901, stop=950, step=1, value=901,
        label="Validation Image ID (ARAD_1K_0XXX)"
    )
    image_selector
    return (image_selector,)


@app.cell
def _(MAX_SHIFT_SIZE, h5py, image_selector, jnp, mo, np, os, retina, scipy):
    # Load and process validation image
    _image_id = image_selector.value
    _spectral_path = f"Dataset/ARAD_1K_Mirror/Valid_spectral/ARAD_1K_{_image_id:04d}.mat"
    _rgb_path = f"Dataset/ARAD_1K_Mirror/Valid_RGB/Valid_RGB/ARAD_1K_{_image_id:04d}.jpg"

    if os.path.exists(_spectral_path):
        # Load hyperspectral data
        _mat = h5py.File(_spectral_path, 'r')
        _cube = np.array(_mat['cube'])

        # Transpose if bands first
        if _cube.shape[0] == 31:
            _cube = np.transpose(_cube, (1, 2, 0))

        # Interpolate from 31 bands to 301 bands
        _H, _W, _bands = _cube.shape
        if _bands == 31:
            _zoom_factor = (1, 1, 301 / 31)
            _cube = scipy.ndimage.zoom(_cube, _zoom_factor, order=1)

        # Get cone fundamentals from retina's ColorSpaceTransform
        _cone_fundamentals = np.array(retina.CST.cone_fundamentals)
        _white_point = np.array(retina.CST.white_point)

        # Convert to LMS
        _lms = np.matmul(_cube, _cone_fundamentals)

        # White balance
        _current_wp = _lms.reshape(-1, 4).max(0) + 1e-10
        _lms = _lms / _current_wp[None, None, :]
        _lms *= _white_point

        # Crop to required size
        _required_size = retina.required_image_resolution + 2 * MAX_SHIFT_SIZE
        _start_h = (_lms.shape[0] - _required_size) // 2
        _start_w = (_lms.shape[1] - _required_size) // 2
        _lms_crop = _lms[_start_h:_start_h + _required_size, _start_w:_start_w + _required_size]

        # Convert to JAX and add batch dimension (B, C, H, W)
        valid_lms = jnp.array(_lms_crop.transpose(2, 0, 1)[None, ...])
        rgb_image_path = _rgb_path if os.path.exists(_rgb_path) else None
    else:
        valid_lms = None
        rgb_image_path = None
        print(f"Spectral file not found: {_spectral_path}")

    # Display the reference RGB image
    if rgb_image_path:
        _ref_output = mo.vstack([
            mo.md(f"**Reference RGB Image: ARAD_1K_{_image_id:04d}**"),
            mo.image(src=rgb_image_path)
        ])
    else:
        _ref_output = mo.md("*Reference RGB image not found.*")

    _ref_output
    return rgb_image_path, valid_lms


@app.cell
def _(mo):
    mo.md("## C.2 Retina Processing Pipeline")
    return


@app.cell
def _(jax, key, mo, np, orient, plt, retina, valid_lms):
    # Process validation image through retina with all intermediate outputs
    if valid_lms is not None:
        _key, _subkey = jax.random.split(key)

        # Get all intermediate outputs
        (
            retina_ons,
            retina_dxy,
            retina_warped_LMS,
            retina_pa,
            retina_bipolar,
            retina_LMS_FoV
        ) = retina(valid_lms, key=_subkey, intermediate_outputs=True)

        # Convert to numpy for visualization
        _ons_np = np.array(retina_ons)
        _dxy_np = np.array(retina_dxy)
        _n_timesteps = _ons_np.shape[1]

        # Visualize first timestep processing stages
        _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Color images
        _lms_fov = np.array(retina_LMS_FoV[0, 0].transpose(1, 2, 0))
        _srgb_fov = np.clip(np.array(retina.CST.LMS_to_sRGB(_lms_fov)), 0, 1)
        _axes[0, 0].imshow(orient(_srgb_fov))
        _axes[0, 0].set_title("1. Current FoV (sRGB)")
        _axes[0, 0].axis('off')

        _warped_lms = np.array(retina_warped_LMS[0, 0].transpose(1, 2, 0))
        _srgb_warped = np.clip(np.array(retina.CST.LMS_to_sRGB(_warped_lms)), 0, 1)
        _axes[0, 1].imshow(orient(_srgb_warped))
        _axes[0, 1].set_title("2. Spatially Sampled (Foveated)")
        _axes[0, 1].axis('off')

        _pa = np.array(retina_pa[0, 0, 0])
        _axes[0, 2].imshow(orient(_pa), cmap='viridis')
        _axes[0, 2].set_title("3. Photoreceptor Activation")
        _axes[0, 2].axis('off')

        # Row 2: Signal processing
        _bipolar = np.array(retina_bipolar[0, 0, 0])
        _axes[1, 0].imshow(orient(_bipolar), cmap='gray')
        _axes[1, 0].set_title("4. Bipolar (After LI)")
        _axes[1, 0].axis('off')

        _ons = np.array(retina_ons[0, 0, 0])
        _axes[1, 1].imshow(orient(_ons), cmap='gray')
        _axes[1, 1].set_title("5. Optic Nerve Signal")
        _axes[1, 1].axis('off')

        # ONS histogram
        _axes[1, 2].hist(_ons.flatten(), bins=50, color='steelblue', edgecolor='black')
        _axes[1, 2].set_title("ONS Distribution")
        _axes[1, 2].set_xlabel("Signal Value")
        _axes[1, 2].set_ylabel("Frequency")

        plt.suptitle("Retina Processing Pipeline (Timestep 1)", fontsize=14, fontweight='bold')
        plt.tight_layout()

        retina_output = mo.vstack([plt.gca()])
    else:
        retina_ons = None
        retina_dxy = None
        retina_output = mo.md("*Validation data not available.*")

    retina_output
    return (
        retina_bipolar,
        retina_dxy,
        retina_LMS_FoV,
        retina_ons,
        retina_output,
        retina_pa,
        retina_warped_LMS,
    )


@app.cell
def _(mo):
    mo.md("## C.3 Eye Movement Analysis (Multi-Timestep)")
    return


@app.cell
def _(TIMESTEPS_PER_IMAGE, create_gif, mo, np, orient, plt, retina, retina_LMS_FoV, retina_dxy, retina_ons):
    if retina_ons is not None:
        _dxy_np = np.array(retina_dxy[0])  # (T-1, 2)
        _n_moves = _dxy_np.shape[0]
        _n_timesteps = retina_ons.shape[1]

        # Create GIF frames for FoV and ONS
        _fov_frames = []
        _ons_frames = []
        for _t in range(_n_timesteps):
            # FoV frame
            _lms_t = np.array(retina_LMS_FoV[0, _t].transpose(1, 2, 0))
            _srgb_t = np.clip(np.array(retina.CST.LMS_to_sRGB(_lms_t)), 0, 1)
            _fov_frames.append(orient(_srgb_t))

            # ONS frame (convert grayscale to RGB for GIF)
            _ons_t = np.array(retina_ons[0, _t, 0])
            _ons_oriented = orient(_ons_t)
            _ons_norm = (_ons_oriented - _ons_oriented.min()) / (_ons_oriented.max() - _ons_oriented.min() + 1e-8)
            _ons_rgb = np.stack([_ons_norm, _ons_norm, _ons_norm], axis=-1)
            _ons_frames.append(_ons_rgb)

        # Create GIFs
        _fov_gif = create_gif(_fov_frames, fps=2)
        _ons_gif = create_gif(_ons_frames, fps=2)

        # Create static analysis plots
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

        # Eye movement trajectory
        _cumulative_x = np.concatenate([[0], np.cumsum(_dxy_np[:, 0])])
        _cumulative_y = np.concatenate([[0], np.cumsum(_dxy_np[:, 1])])
        _axes[0].plot(_cumulative_x, _cumulative_y, 'b-o', markersize=8, linewidth=2)
        _axes[0].plot(0, 0, 'go', markersize=12, label='Start')
        _axes[0].plot(_cumulative_x[-1], _cumulative_y[-1], 'ro', markersize=12, label='End')
        for _i in range(len(_cumulative_x)):
            _axes[0].annotate(f't{_i}', (_cumulative_x[_i], _cumulative_y[_i]),
                             textcoords="offset points", xytext=(5, 5))
        _axes[0].set_xlabel("Cumulative dx")
        _axes[0].set_ylabel("Cumulative dy")
        _axes[0].set_title("Eye Movement Trajectory")
        _axes[0].legend()
        _axes[0].grid(True, alpha=0.3)
        _axes[0].axis('equal')

        # Movement magnitudes
        _magnitudes = np.sqrt(_dxy_np[:, 0]**2 + _dxy_np[:, 1]**2)
        _axes[1].bar(range(1, _n_moves + 1), _magnitudes, color='steelblue')
        _axes[1].set_xlabel("Movement")
        _axes[1].set_ylabel("Magnitude")
        _axes[1].set_title("Movement Magnitudes")

        # dx, dy over time
        _axes[2].plot(range(1, _n_moves + 1), _dxy_np[:, 0], 'r-o', label='dx')
        _axes[2].plot(range(1, _n_moves + 1), _dxy_np[:, 1], 'b-o', label='dy')
        _axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        _axes[2].set_xlabel("Movement")
        _axes[2].set_ylabel("Displacement")
        _axes[2].set_title("dx, dy Components")
        _axes[2].legend()
        _axes[2].grid(True, alpha=0.3)

        plt.suptitle("Eye Movement Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _eye_stats = mo.md(f"""
        **Eye Movement Statistics:**
        - Number of timesteps: {_n_timesteps}
        - Number of movements: {_n_moves}
        - Total displacement: ({_cumulative_x[-1]:.4f}, {_cumulative_y[-1]:.4f})
        - Mean magnitude: {_magnitudes.mean():.4f}
        - Max magnitude: {_magnitudes.max():.4f}
        """)

        _c3_output = mo.vstack([
            mo.md("### Animated Field of View (FoV) Across Timesteps"),
            _fov_gif,
            mo.md("### Animated Optic Nerve Signal (ONS) Across Timesteps"),
            _ons_gif,
            mo.md("### Eye Movement Analysis"),
            plt.gca(),
            _eye_stats
        ])
    else:
        _c3_output = mo.md("*Retina outputs not available.*")
    _c3_output
    return


@app.cell
def _(mo):
    mo.md("## C.4 Cortex Decoding (Multi-Timestep)")
    return


@app.cell
def _(cortex_trained, create_gif, mo, np, orient, plt, render_treescope, retina, retina_ons):
    if retina_ons is not None and cortex_trained is not None:
        # Decode each timestep
        _n_timesteps = retina_ons.shape[1]
        _decoded_ips = []
        _decoded_rgbs = []
        _decoded_rgb_frames = []

        for _t in range(_n_timesteps):
            _ons_t = retina_ons[:, _t, :, :, :]  # (B, 1, H, W)
            _ip_t = cortex_trained.decode(_ons_t)
            _rgb_t = cortex_trained.ns_ip(_ip_t)

            _decoded_ips.append(_ip_t)
            _decoded_rgbs.append(_rgb_t)

            # Create frame for GIF
            _rgb_np = np.array(_rgb_t[0, :3].transpose(1, 2, 0))
            _srgb = np.clip(np.array(retina.CST.linsRGB_to_sRGB(_rgb_np)), 0, 1)
            _decoded_rgb_frames.append(orient(_srgb))

        # Create GIF of decoded RGB
        _decoded_gif = create_gif(_decoded_rgb_frames, fps=2)

        # Visualize all timesteps side by side
        _n_show = min(_n_timesteps, 4)
        _fig, _axes = plt.subplots(2, _n_show, figsize=(4 * _n_show, 8))

        for _t in range(_n_show):
            # ONS input
            _ons_t = np.array(retina_ons[0, _t, 0])
            _axes[0, _t].imshow(orient(_ons_t), cmap='gray')
            _axes[0, _t].set_title(f"ONS t={_t}")
            _axes[0, _t].axis('off')

            # Decoded RGB
            _axes[1, _t].imshow(_decoded_rgb_frames[_t])
            _axes[1, _t].set_title(f"Decoded RGB t={_t}")
            _axes[1, _t].axis('off')

        plt.suptitle("Cortex Decoding: ONS → Internal Percept → RGB", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Build treescope dict with all timesteps
        _treescope_data = {}
        for _t in range(_n_timesteps):
            _treescope_data[f"t{_t}"] = {
                "Input_ONS": retina_ons[:, _t],
                "Internal_Percept": _decoded_ips[_t],
                "RGB_Prediction": _decoded_rgbs[_t],
            }

        _c4_output = mo.vstack([
            mo.md("### Animated Cortex Reconstruction"),
            _decoded_gif,
            mo.md("### Static Comparison (All Timesteps)"),
            plt.gca(),
            mo.md("### Decoded Tensors (All Timesteps)"),
            render_treescope(_treescope_data, height="600px")
        ])
    else:
        _c4_output = mo.md("*Cortex or retina outputs not available.*")
    _c4_output
    return


@app.cell
def _(mo):
    mo.md("## C.5 Internal Percept Latent Space")
    return


@app.cell
def _(LATENT_DIM, cortex_trained, mo, np, orient, plt, retina_ons):
    if retina_ons is not None and cortex_trained is not None:
        # Get internal percept for first timestep
        _ons1 = retina_ons[:, 0, :, :, :]
        _ip = cortex_trained.decode(_ons1)
        _ip_np = np.array(_ip[0])  # (latent_dim, H, W)

        # Visualize all latent channels
        _fig, _axes = plt.subplots(2, 4, figsize=(16, 8))

        for _i in range(LATENT_DIM):
            _ax = _axes[_i // 4, _i % 4]
            _channel = _ip_np[_i]
            _im = _ax.imshow(orient(_channel), cmap='viridis')
            _ax.set_title(f"Latent {_i}: [{_channel.min():.2f}, {_channel.max():.2f}]")
            _ax.axis('off')
            plt.colorbar(_im, ax=_ax, fraction=0.046)

        plt.suptitle("Internal Percept Latent Channels", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _latent_stats = mo.md(f"""
        **Latent Space Statistics:**
        - Shape: {_ip_np.shape}
        - Overall Range: [{_ip_np.min():.4f}, {_ip_np.max():.4f}]
        - Mean: {_ip_np.mean():.4f}
        - Std: {_ip_np.std():.4f}
        """)

        _c5_output = mo.vstack([plt.gca(), _latent_stats])
    else:
        _c5_output = mo.md("*Data not available.*")
    _c5_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## C.6 Reconstruction Quality Analysis

        Compare the ground truth FoV to the cortex reconstruction.
        """
    )
    return


@app.cell
def _(cortex_trained, mo, np, orient, plt, retina, retina_ons, retina_warped_LMS):
    if retina_ons is not None and cortex_trained is not None:
        # Ground truth: Spatially sampled LMS (same size as simulation: 256x256)
        # Using retina_warped_LMS instead of retina_LMS_FoV to match cortex output size
        _gt_lms = np.array(retina_warped_LMS[0, 0].transpose(1, 2, 0))
        _gt_srgb = np.clip(np.array(retina.CST.LMS_to_sRGB(_gt_lms)), 0, 1)

        # Reconstruction: Cortex decoded
        _ons1 = retina_ons[:, 0, :, :, :]
        _ip = cortex_trained.decode(_ons1)
        _rgb_pred = cortex_trained.ns_ip(_ip)
        _pred_np = np.array(_rgb_pred[0, :3].transpose(1, 2, 0))
        _pred_srgb = np.clip(np.array(retina.CST.linsRGB_to_sRGB(_pred_np)), 0, 1)

        # Compute error
        _gt_oriented = orient(_gt_srgb)
        _pred_oriented = orient(_pred_srgb)
        _error = np.abs(_gt_oriented - _pred_oriented)
        _mse = np.mean(_error ** 2)
        _psnr = 10 * np.log10(1.0 / (_mse + 1e-10))

        _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))

        _axes[0].imshow(_gt_oriented)
        _axes[0].set_title("Ground Truth (Foveated)")
        _axes[0].axis('off')

        _axes[1].imshow(_pred_oriented)
        _axes[1].set_title("Cortex Reconstruction")
        _axes[1].axis('off')

        _im = _axes[2].imshow(_error.mean(axis=2), cmap='hot', vmin=0, vmax=0.3)
        _axes[2].set_title("Absolute Error")
        _axes[2].axis('off')
        plt.colorbar(_im, ax=_axes[2])

        # Per-channel error histogram
        _axes[3].hist(_error[:, :, 0].flatten(), bins=50, alpha=0.5, color='red', label='R')
        _axes[3].hist(_error[:, :, 1].flatten(), bins=50, alpha=0.5, color='green', label='G')
        _axes[3].hist(_error[:, :, 2].flatten(), bins=50, alpha=0.5, color='blue', label='B')
        _axes[3].set_xlabel("Error")
        _axes[3].set_ylabel("Frequency")
        _axes[3].set_title("Error Distribution")
        _axes[3].legend()

        plt.suptitle("Reconstruction Quality (vs Foveated Input)", fontsize=14, fontweight='bold')
        plt.tight_layout()

        _quality_stats = mo.md(f"""
        **Quality Metrics:**
        - MSE: {_mse:.6f}
        - PSNR: {_psnr:.2f} dB
        - Mean Absolute Error: {_error.mean():.4f}
        - Max Error: {_error.max():.4f}

        *Note: Comparing cortex reconstruction against the foveated (spatially sampled) input,
        not the full FoV, since the cortex operates at simulation resolution ({_gt_oriented.shape[0]}x{_gt_oriented.shape[1]}).*
        """)

        _c6_output = mo.vstack([plt.gca(), _quality_stats])
    else:
        _c6_output = mo.md("*Data not available.*")
    _c6_output
    return


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## Summary

        This notebook provides comprehensive inspection of the Matisse retina-cortex model:

        **Retina (Forward Model):**
        - Color space transformations (sRGB ↔ LMS ↔ XYZ)
        - Cone spectral sensitivities and spatial mosaic distribution
        - Foveated spatial sampling with MIP-mapped resolution falloff
        - Lateral inhibition via DoG center-surround processing

        **Cortex (Inverse Model):**
        - Learned cone identity function in latent space
        - Learned LI deconvolution kernel (comparison to ground truth)
        - RealNVP coordinate transforms for cone position mapping
        - Multi-timestep eye movement tracking

        **For HCI/Display Research:**
        - The model provides a differentiable pipeline from display output to perceived image
        - Can be used to evaluate display quality under different viewing conditions
        - Supports CVD simulation by modifying cone mosaic composition
        - Eye movement integration reveals temporal aspects of perception
        """
    )
    return


if __name__ == "__main__":
    app.run()
