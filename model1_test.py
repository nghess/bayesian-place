"""
Bayesian place field estimation with temporal stability.

Approach: Kernel density estimation of spatial firing rate maps
with hierarchical structure across time blocks. Stability is
quantified via posterior correlation between successive maps.

Expects:
    - position: (N, 3) array of [timestamp, x, y]
    - spike_times: (S,) array of spike timestamps

You'll need to adapt the data loading to your format.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


# --- Core estimation functions ---

def compute_occupancy(position, x_edges, y_edges, dt):
    """
    Compute time-spent-per-bin (occupancy map) from position data.

    Parameters
    ----------
    position : (N, 3) array — [timestamp, x, y]
    x_edges, y_edges : bin edges for the 2D histogram
    dt : float — sampling interval of position data (seconds)

    Returns
    -------
    occupancy : 2D array, time in seconds per spatial bin
    """
    occ, _, _ = np.histogram2d(
        position[:, 1], position[:, 2],
        bins=[x_edges, y_edges]
    )
    return occ * dt


def compute_spike_map(spike_times, position, x_edges, y_edges):
    """
    Bin spike counts into spatial bins by interpolating spike positions.

    Parameters
    ----------
    spike_times : (S,) array of spike timestamps
    position : (N, 3) array — [timestamp, x, y]
    x_edges, y_edges : bin edges

    Returns
    -------
    spike_map : 2D array, spike counts per spatial bin
    """
    # Interpolate animal position at each spike time
    spike_x = np.interp(spike_times, position[:, 0], position[:, 1])
    spike_y = np.interp(spike_times, position[:, 0], position[:, 2])

    spike_map, _, _ = np.histogram2d(
        spike_x, spike_y,
        bins=[x_edges, y_edges]
    )
    return spike_map


def estimate_rate_map(spike_map, occupancy, smooth_sigma=2.0, min_occupancy=0.1):
    """
    Compute smoothed firing rate map.

    Parameters
    ----------
    spike_map : 2D spike count array
    occupancy : 2D occupancy array (seconds)
    smooth_sigma : Gaussian smoothing kernel width (in bins)
    min_occupancy : minimum occupancy (seconds) to include a bin

    Returns
    -------
    rate_map : 2D array, firing rate (Hz). Unvisited bins are NaN.
    """
    # Smooth both maps before dividing (adaptive smoothing alternative exists)
    smooth_spikes = gaussian_filter(spike_map.astype(float), sigma=smooth_sigma)
    smooth_occ = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)

    rate_map = np.full_like(smooth_spikes, np.nan)
    valid = smooth_occ > min_occupancy
    rate_map[valid] = smooth_spikes[valid] / smooth_occ[valid]

    return rate_map


# --- Temporal block analysis ---

def split_into_blocks(position, spike_times, n_blocks):
    """
    Split session into temporal blocks.

    Returns
    -------
    blocks : list of (block_position, block_spike_times) tuples
    """
    t_start, t_end = position[0, 0], position[-1, 0]
    edges = np.linspace(t_start, t_end, n_blocks + 1)

    blocks = []
    for i in range(n_blocks):
        t0, t1 = edges[i], edges[i + 1]
        pos_mask = (position[:, 0] >= t0) & (position[:, 0] < t1)
        spk_mask = (spike_times >= t0) & (spike_times < t1)
        blocks.append((position[pos_mask], spike_times[spk_mask]))

    return blocks


def compute_block_rate_maps(blocks, x_edges, y_edges, dt, smooth_sigma=2.0):
    """Compute a rate map for each temporal block."""
    rate_maps = []
    for pos_block, spk_block in blocks:
        if len(pos_block) < 10 or len(spk_block) < 1:
            rate_maps.append(None)
            continue
        occ = compute_occupancy(pos_block, x_edges, y_edges, dt)
        spk = compute_spike_map(spk_block, pos_block, x_edges, y_edges)
        rate_maps.append(estimate_rate_map(spk, occ, smooth_sigma))
    return rate_maps


# --- Stability metrics ---

def pairwise_spatial_correlation(rate_maps):
    """
    Compute Pearson correlation between successive rate maps.
    Only includes bins that are valid (non-NaN) in both maps.

    Returns
    -------
    correlations : list of (r, p) tuples for consecutive block pairs
    """
    correlations = []
    for i in range(len(rate_maps) - 1):
        if rate_maps[i] is None or rate_maps[i + 1] is None:
            correlations.append((np.nan, np.nan))
            continue

        m1, m2 = rate_maps[i].ravel(), rate_maps[i + 1].ravel()
        valid = np.isfinite(m1) & np.isfinite(m2)

        if valid.sum() < 5:
            correlations.append((np.nan, np.nan))
        else:
            r, p = pearsonr(m1[valid], m2[valid])
            correlations.append((r, p))

    return correlations


def bayesian_stability_estimate(correlations, prior_mean=0.5, prior_var=0.25):
    """
    Simple conjugate Bayesian update for the 'stability' parameter,
    treating Fisher-z-transformed correlations as approximately normal.

    This gives you a posterior distribution over mean stability rather
    than just point estimates.

    Parameters
    ----------
    correlations : list of (r, p) tuples
    prior_mean : prior mean for Fisher-z stability
    prior_var : prior variance

    Returns
    -------
    post_mean, post_var : posterior mean and variance in Fisher-z space
    rs_valid : the raw correlation values used
    """
    rs = np.array([r for r, p in correlations if np.isfinite(r)])
    if len(rs) == 0:
        return prior_mean, prior_var, rs

    # Fisher z-transform for approximate normality
    z = np.arctanh(np.clip(rs, -0.999, 0.999))

    # Observation variance: 1/(n_bins - 3) per observation, but we
    # approximate with a fixed value — adjust if you have variable bin counts
    obs_var = 0.1  # you can refine this based on your actual bin counts

    # Conjugate normal-normal update
    prior_precision = 1.0 / prior_var
    obs_precision = len(z) / obs_var
    post_precision = prior_precision + obs_precision
    post_mean = (prior_precision * prior_mean + obs_precision * z.mean()) / post_precision
    post_var = 1.0 / post_precision

    return post_mean, post_var, rs


# --- Visualization ---

def plot_results(rate_maps, correlations, post_mean, post_var, x_edges, y_edges):
    """Plot rate maps per block, correlation time course, and posterior."""
    n_blocks = len(rate_maps)
    valid_maps = [rm for rm in rate_maps if rm is not None]

    if not valid_maps:
        print("No valid rate maps to plot.")
        return

    vmax = np.nanpercentile(np.concatenate([rm.ravel() for rm in valid_maps]), 98)

    fig, axes = plt.subplots(2, max(n_blocks, 2), figsize=(4 * n_blocks, 8))

    # Top row: rate maps
    for i, rm in enumerate(rate_maps):
        ax = axes[0, i] if n_blocks > 1 else axes[0]
        if rm is not None:
            ax.imshow(rm.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                      cmap='hot')
        ax.set_title(f'Block {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Hide unused axes in top row
    for i in range(n_blocks, axes.shape[1]):
        axes[0, i].set_visible(False)

    # Bottom left: correlation time course
    ax_corr = axes[1, 0]
    rs = [r for r, p in correlations]
    ax_corr.plot(range(1, len(rs) + 1), rs, 'ko-')
    ax_corr.set_xlabel('Block transition')
    ax_corr.set_ylabel('Spatial correlation (r)')
    ax_corr.set_ylim(-0.2, 1.05)
    ax_corr.set_title('Pairwise stability')
    ax_corr.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Bottom right: posterior on stability
    ax_post = axes[1, 1]
    z_range = np.linspace(-1, 2, 200)
    posterior = np.exp(-0.5 * (z_range - post_mean) ** 2 / post_var)
    posterior /= posterior.sum() * (z_range[1] - z_range[0])
    # Convert z back to r for interpretability
    r_range = np.tanh(z_range)
    ax_post.plot(r_range, posterior, 'b-', lw=2)
    ax_post.fill_between(r_range, posterior, alpha=0.2)
    ax_post.set_xlabel('Stability (r)')
    ax_post.set_ylabel('Posterior density')
    ax_post.set_title(f'Posterior: mean r ≈ {np.tanh(post_mean):.2f}')
    ax_post.set_xlim(-0.5, 1.05)

    for i in range(2, axes.shape[1]):
        axes[1, i].set_visible(False)

    plt.tight_layout()
    plt.savefig('place_field_stability.png', dpi=150)
    plt.show()


# --- Main pipeline ---

def run_analysis(position, spike_times, n_bins=30, n_blocks=4, smooth_sigma=2.0):
    """
    Full pipeline.

    Parameters
    ----------
    position : (N, 3) array [timestamp, x, y]
    spike_times : (S,) array
    n_bins : spatial bins per dimension
    n_blocks : number of temporal blocks
    smooth_sigma : Gaussian smoothing width in bins
    """
    dt = np.median(np.diff(position[:, 0]))

    # Spatial bin edges
    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    # Full-session rate map
    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    full_rate_map = estimate_rate_map(spk, occ, smooth_sigma)

    print(f"Session duration: {position[-1, 0] - position[0, 0]:.1f} s")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Peak rate: {np.nanmax(full_rate_map):.1f} Hz")

    # Temporal blocks
    blocks = split_into_blocks(position, spike_times, n_blocks)
    block_maps = compute_block_rate_maps(blocks, x_edges, y_edges, dt, smooth_sigma)

    # Stability
    corrs = pairwise_spatial_correlation(block_maps)
    post_mean, post_var, rs = bayesian_stability_estimate(corrs)

    print(f"\nPairwise correlations: {[f'{r:.3f}' for r, p in corrs]}")
    print(f"Posterior mean stability (r): {np.tanh(post_mean):.3f}")
    print(f"Posterior 95% CI: [{np.tanh(post_mean - 1.96*np.sqrt(post_var)):.3f}, "
          f"{np.tanh(post_mean + 1.96*np.sqrt(post_var)):.3f}]")

    plot_results(block_maps, corrs, post_mean, post_var, x_edges, y_edges)

    return full_rate_map, block_maps, corrs, (post_mean, post_var)


# --- Example with synthetic data ---

if __name__ == '__main__':
    np.random.seed(42)

    # Simulate 10 min of random foraging in a 1m x 1m box
    duration = 600  # seconds
    fs_pos = 30  # position sampling rate
    n_samples = duration * fs_pos
    t = np.linspace(0, duration, n_samples)

    # Random walk position
    dx = np.cumsum(np.random.randn(n_samples) * 0.002)
    dy = np.cumsum(np.random.randn(n_samples) * 0.002)
    x = np.clip(0.5 + dx, 0, 1)
    y = np.clip(0.5 + dy, 0, 1)
    position = np.column_stack([t, x, y])

    # Simulate a place cell: Gaussian field centered at (0.6, 0.4)
    # with slight drift in preferred location over time
    peak_rate = 15  # Hz
    field_width = 0.15

    # Drift: center moves from (0.6, 0.4) to (0.65, 0.42) over session
    cx = 0.6 + 0.05 * (t / duration)
    cy = 0.4 + 0.02 * (t / duration)

    rate = peak_rate * np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * field_width ** 2)
    )

    # Generate spikes as inhomogeneous Poisson process
    spike_prob = rate / fs_pos
    spikes_bool = np.random.rand(n_samples) < spike_prob
    spike_times = t[spikes_bool]

    print(f"Generated {len(spike_times)} spikes\n")

    # Run
    full_map, block_maps, corrs, posterior = run_analysis(
        position, spike_times, n_bins=25, n_blocks=5, smooth_sigma=2.0
    )