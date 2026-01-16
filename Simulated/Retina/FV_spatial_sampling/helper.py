"""Helper functions for spatial sampling in JAX."""
import math
import jax
import jax.numpy as jnp
import pickle
import numpy as np
from jaxtyping import Array, Float
import os


def compute_required_image_resolution(xy: Float[Array, "1 2 height width"]) -> int:
    """Compute required image resolution from XY coordinate map.

    Args:
        xy: Coordinate tensor of shape (1, 2, H, W), values in range [-1, 1]

    Returns:
        Required image resolution as integer
    """
    # Normalize to [0, 1]
    xy_norm = xy.at[:, 0].set(xy[:, 0] - jnp.min(xy[:, 0]))
    xy_norm = xy_norm.at[:, 1].set(xy_norm[:, 1] - jnp.min(xy_norm[:, 1]))
    xy_norm = xy_norm.at[:, 0].set(xy_norm[:, 0] / jnp.max(xy_norm[:, 0]))
    xy_norm = xy_norm.at[:, 1].set(xy_norm[:, 1] / jnp.max(xy_norm[:, 1]))

    # Compute gradients
    du = jnp.sqrt(jnp.sum((xy_norm[0, :, 1:, :] - xy_norm[0, :, :-1, :]) ** 2, axis=0))
    dv = jnp.sqrt(jnp.sum((xy_norm[0, :, :, 1:] - xy_norm[0, :, :, :-1]) ** 2, axis=0))

    # Find minimum average of smallest 10 distances
    du_sorted = jnp.sort(du.flatten())[:10]
    dv_sorted = jnp.sort(dv.flatten())[:10]

    multiplier = 1.0 / jnp.minimum(jnp.mean(du_sorted), jnp.mean(dv_sorted))
    # Convert to Python int for use in static fields
    required_image_resolution = int(jnp.ceil(multiplier))

    return required_image_resolution


def rand_perlin_2d(shape, res, fade_fn=None):
    """Generate 2D Perlin noise.

    From https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57

    Args:
        shape: Output shape (H, W)
        res: Resolution tuple (res_h, res_w)
        fade_fn: Fade function (default: smoothstep)

    Returns:
        Perlin noise array
    """
    if fade_fn is None:
        fade_fn = lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid_x, grid_y = jnp.meshgrid(
        jnp.arange(0, res[0], delta[0]),
        jnp.arange(0, res[1], delta[1]),
        indexing='ij'
    )
    grid = jnp.stack([grid_x, grid_y], axis=-1) % 1

    # Generate random angles (use numpy for random generation during initialization)
    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    gradients = jnp.array(gradients)

    def tile_grads(slice1, slice2):
        grad_slice = gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]]
        # Repeat along axes
        return jnp.repeat(jnp.repeat(grad_slice, d[0], axis=0), d[1], axis=1)

    def dot(grad, shift):
        shifted_grid = jnp.stack([
            grid[:shape[0], :shape[1], 0] + shift[0],
            grid[:shape[0], :shape[1], 1] + shift[1]
        ], axis=-1)
        return jnp.sum(shifted_grid * grad[:shape[0], :shape[1]], axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    t = fade_fn(grid[:shape[0], :shape[1]])
    # Manual linear interpolation (JAX doesn't have jnp.lerp)
    lerp1 = n00 + (n10 - n00) * t[..., 0]
    lerp2 = n01 + (n11 - n01) * t[..., 0]
    result = math.sqrt(2) * (lerp1 + (lerp2 - lerp1) * t[..., 1])

    return result


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5, seed=0):
    """Generate multi-octave Perlin noise.

    Args:
        shape: Output shape (H, W)
        res: Base resolution tuple (res_h, res_w)
        octaves: Number of octaves
        persistence: Amplitude decay factor
        seed: Random seed

    Returns:
        Multi-octave Perlin noise
    """
    np.random.seed(seed)
    noise = jnp.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        noise = noise + amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise


def get_cone_sampling_map(
    simulation_size: int,
    cone_distribution_type: str = 'Human',
    root_dir: str = None
):
    """Get cone sampling map with MIP levels.

    Args:
        simulation_size: Dimension of simulation
        cone_distribution_type: Type of cone distribution ('Human', 'HorizontalStreak', 'TwoFoveas')
        root_dir: Root directory path

    Returns:
        Tuple of (cone_locs, mip_level, required_image_resolution)
    """
    # Load cone locations from pickle file
    if root_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            current_dir,
            "../../../Assets/Retina",
            f"cone_locs_{cone_distribution_type}_512.cpkl"
        )
    else:
        data_path = f"{root_dir}/Assets/Retina/cone_locs_{cone_distribution_type}_512.cpkl"

    with open(data_path, 'rb') as f:
        cone_locs_in_ecc = pickle.load(f)

    height = cone_locs_in_ecc.shape[0]

    D = simulation_size
    N = height
    P = (height - D) // 2

    cone_locs_in_ecc = cone_locs_in_ecc / np.max(np.abs(cone_locs_in_ecc))

    # Generate Perlin noise for realistic variation
    noise_x = rand_perlin_2d_octaves((N, N), (2, 2), 6, 0.8, seed=0)
    noise_y = rand_perlin_2d_octaves((N, N), (2, 2), 6, 0.8, seed=1)

    noise = jnp.stack([noise_x, noise_y], axis=-1)
    # Pad to handle boundaries
    noise = jnp.pad(noise[1:-1, 1:-1], ((1, 1), (1, 1), (0, 0)), mode='constant')

    # Create base location grid
    x = np.linspace(0, N - 1, N)
    y = np.linspace(0, N - 1, N)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    base_loc = np.stack([xx, yy], axis=-1)

    base_loc2 = base_loc + np.array(noise)
    base_loc_norm = ((base_loc / (N - 1)) - 0.5) * 2
    base_loc2_norm = ((base_loc2 / (N - 1)) - 0.5) * 2

    # Resample cone locations with noise using grid sampling
    from jax.scipy.ndimage import map_coordinates

    # Convert coordinates for map_coordinates
    # base_loc2_norm is in [-1, 1], convert to pixel coords [0, N-1]
    coords_pixel = (base_loc2_norm + 1) * (N - 1) / 2

    # Resample each channel
    cone_locs_in_ecc_jax = jnp.array(cone_locs_in_ecc)
    noisy_cone_locs_list = []
    for c in range(cone_locs_in_ecc_jax.shape[2]):
        channel = cone_locs_in_ecc_jax[:, :, c]
        # map_coordinates expects (ndim, ...) shape for coordinates
        coords = jnp.stack([coords_pixel[:, :, 1], coords_pixel[:, :, 0]], axis=0)
        resampled = map_coordinates(channel, coords, order=1, mode='constant', cval=0)
        noisy_cone_locs_list.append(resampled)

    noisy_cone_locs_in_ecc = jnp.stack(noisy_cone_locs_list, axis=-1)

    # Crop to simulation size
    if P > 0:
        noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc[P:-P, P:-P]

    # Normalize to [0, 1]
    noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc - jnp.min(noisy_cone_locs_in_ecc)
    noisy_cone_locs_in_ecc = noisy_cone_locs_in_ecc / jnp.max(noisy_cone_locs_in_ecc)

    # Compute required image resolution
    noisy_cone_locs_tensor = jnp.transpose(noisy_cone_locs_in_ecc, (2, 0, 1))[None, ...]
    required_image_resolution = compute_required_image_resolution(noisy_cone_locs_tensor)

    # Convert to [-1, 1] for grid sampling
    cone_locs = (noisy_cone_locs_in_ecc - 0.5) * 2.0

    # Compute MIP level
    tmp_cone_locs = noisy_cone_locs_in_ecc * required_image_resolution
    du = jnp.sqrt(jnp.sum((tmp_cone_locs[1:, :] - tmp_cone_locs[:-1, :]) ** 2, axis=-1))
    dv = jnp.sqrt(jnp.sum((tmp_cone_locs[:, 1:] - tmp_cone_locs[:, :-1]) ** 2, axis=-1))
    du = jnp.pad(du, ((0, 1), (0, 0)), mode='reflect')
    dv = jnp.pad(dv, ((0, 0), (0, 1)), mode='reflect')
    mip_level = jnp.log2(jnp.maximum(du, dv))

    return np.array(cone_locs), np.array(mip_level), required_image_resolution
