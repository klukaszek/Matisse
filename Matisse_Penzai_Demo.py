import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import penzai
    from penzai import pz
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import sys
    
    # Add project root to path
    sys.path.append('.')
    
    from Simulated.Retina import RetinaModel
    from Simulated.Cortex import CortexModel
    from Simulated.Retina.helper.ColorSpaceTransform import ColorSpaceTransform
    return (
        ColorSpaceTransform,
        CortexModel,
        RetinaModel,
        eqx,
        jax,
        jnp,
        np,
        os,
        penzai,
        plt,
        pz,
        sys,
    )


@app.cell
def _(pz):
    pz.ts.active_interactive_context.enable()
    return


@app.cell
def _(marimo):
    marimo.md(
        """
        # 🎨 Matisse: JAX/Penzai Interpretability Demo
        
        This notebook demonstrates the **Matisse** model (Retina + Cortex) implemented in JAX, visualized using **Penzai** for structural interpretability.
        """
    )
    return


@app.cell
def _(CortexModel, RetinaModel, jax):
    # Initialize Models
    key = jax.random.PRNGKey(42)
    
    # Retina Configuration
    simulation_size = 256
    timesteps_per_image = 2
    max_shift_size = 15
    
    print("Initializing RetinaModel...")
    retina = RetinaModel(
        simulation_size=simulation_size,
        timesteps_per_image=timesteps_per_image,
        max_shift_size=max_shift_size,
        cone_types_str='LMS',
        cone_distribution_type='Human',
        simulating_tetra=False
    )
    
    # Cortex Configuration
    print("Initializing CortexModel...")
    key, subkey = jax.random.split(key)
    cortex = CortexModel(
        latent_dim=8,
        simulation_size=simulation_size,
        simulating_tetra=False,
        key=subkey
    )
    
    print("Models initialized successfully!")
    return (
        cortex,
        key,
        max_shift_size,
        retina,
        simulation_size,
        subkey,
        timesteps_per_image,
    )


@app.cell
def _(marimo):
    marimo.md("## 👁️ Retina Model Structure")
    return


@app.cell
def _(pz, retina):
    # Visualize Retina Structure
    pz.show(retina)
    return


@app.cell
def _(marimo):
    marimo.md("## 🧠 Cortex Model Structure")
    return


@app.cell
def _(cortex, pz):
    # Visualize Cortex Structure
    pz.show(cortex)
    return


@app.cell
def _(marimo):
    marimo.md("## 🌈 Cone Fundamentals & Mosaic")
    return


@app.cell
def _(plt, retina):
    # Plot Cone Fundamentals
    fundamentals = retina.SpectralSampling.get_cone_fundamentals()
    wavelengths = list(range(400, 701))
    
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, fundamentals[:, 0], 'r', label='L Cone')
    plt.plot(wavelengths, fundamentals[:, 1], 'g', label='M Cone')
    plt.plot(wavelengths, fundamentals[:, 2], 'b', label='S Cone')
    plt.title("Cone Spectral Sensitivities")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca()
    return (fundamentals, wavelengths)


@app.cell
def _(plt, retina):
    # Visualize Cone Mosaic
    mosaic = retina.get_cone_mosaic()[0] # (4, H, W)
    
    # Combine into RGB for visualization
    # R=L, G=M, B=S
    mosaic_img = mosaic[:3, :, :].transpose(1, 2, 0)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(mosaic_img)
    plt.title("Cone Mosaic Distribution (L=Red, M=Green, S=Blue)")
    plt.axis('off')
    plt.gca()
    return (mosaic, mosaic_img)


@app.cell
def _(marimo):
    marimo.md("## 🔄 End-to-End Processing Demo")
    return


@app.cell
def _(
    cortex,
    jax,
    key,
    max_shift_size,
    pz,
    retina,
    simulation_size,
    timesteps_per_image,
):
    # Generate Synthetic Input
    full_field_size = retina.required_image_resolution + 2 * max_shift_size
    key, subkey = jax.random.split(key)
    
    # Random colored pattern
    input_image = jax.random.uniform(
        subkey, 
        (1, 4, full_field_size, full_field_size) # Batch, Channels (LMSQ), H, W
    )
    
    # Forward Pass through Retina
    key, subkey = jax.random.split(key)
    batch_ons, batch_true_dxy, batch_warped_LMS = retina(
        input_image, 
        key=subkey,
        intermediate_outputs=False
    )
    
    # Extract first timestep ONS
    ons1 = batch_ons[:, 0, :, :, :] # (B, 1, H, W)
    
    # Decode with Cortex
    decoded_ip = cortex.decode(ons1)
    
    # Penzai visualizer for tensors
    pz.show({
        "Input (LMS)": input_image,
        "Optic Nerve Signal": ons1,
        "Decoded Internal Percept": decoded_ip,
        "Eye Movement (dxy)": batch_true_dxy
    })
    return (
        batch_ons,
        batch_true_dxy,
        batch_warped_LMS,
        decoded_ip,
        full_field_size,
        input_image,
        ons1,
    )


if __name__ == "__main__":
    app.run()
