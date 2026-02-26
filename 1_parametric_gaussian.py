"""
Bayesian place field estimation with temporal stability.

Approach: Kernel density estimation of spatial firing rate maps
with hierarchical structure across time blocks. Stability is
quantified via spatial correlation and center-of-mass shift.

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


# --- Bayesian rate estimation ---

def compute_marginal_posteriors(spike_map, occupancy, x_edges, y_edges,
                                 prior_alpha=0.5, prior_beta=0.1):
    """
    Compute posterior distributions for marginal firing rates using Gamma-Poisson conjugacy.

    For each x (or y) position, we marginalize across the other dimension and compute
    a posterior distribution over the firing rate.

    Parameters
    ----------
    spike_map : 2D array of spike counts per bin
    occupancy : 2D array of time spent per bin (seconds)
    x_edges, y_edges : bin edges
    prior_alpha, prior_beta : Gamma prior parameters (shape, rate)
        Default prior is weakly informative: Gamma(0.5, 0.1) -> mean=5, wide variance

    Returns
    -------
    x_posterior : dict with 'centers', 'alpha', 'beta' (posterior Gamma parameters)
    y_posterior : dict with 'centers', 'alpha', 'beta'
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # X marginal: sum spikes and occupancy across y for each x
    x_spikes = np.sum(spike_map, axis=1)
    x_occupancy = np.sum(occupancy, axis=1)

    # Posterior: Gamma(alpha + spikes, beta + occupancy)
    x_post_alpha = prior_alpha + x_spikes
    x_post_beta = prior_beta + x_occupancy

    x_posterior = {
        'centers': x_centers,
        'alpha': x_post_alpha,
        'beta': x_post_beta
    }

    # Y marginal: sum spikes and occupancy across x for each y
    y_spikes = np.sum(spike_map, axis=0)
    y_occupancy = np.sum(occupancy, axis=0)

    y_post_alpha = prior_alpha + y_spikes
    y_post_beta = prior_beta + y_occupancy

    y_posterior = {
        'centers': y_centers,
        'alpha': y_post_alpha,
        'beta': y_post_beta
    }

    return x_posterior, y_posterior


def gamma_credible_interval(alpha, beta, ci=0.95):
    """
    Compute mean and credible interval for Gamma(alpha, beta) distribution.

    Parameters
    ----------
    alpha, beta : arrays of Gamma parameters (shape, rate)
    ci : credible interval width (default 95%)

    Returns
    -------
    mean, lower, upper : arrays
    """
    from scipy.stats import gamma

    mean = alpha / beta
    lower = gamma.ppf((1 - ci) / 2, alpha, scale=1/beta)
    upper = gamma.ppf((1 + ci) / 2, alpha, scale=1/beta)

    return mean, lower, upper


# --- Parametric Gaussian place field model ---

def gaussian_rate_map(x_centers, y_centers, mu_x, mu_y, sigma_x, sigma_y, peak, baseline=0):
    """
    Compute a 2D Gaussian rate map.

    Parameters
    ----------
    x_centers, y_centers : 1D arrays of bin centers
    mu_x, mu_y : field center coordinates
    sigma_x, sigma_y : field widths
    peak : peak firing rate above baseline
    baseline : baseline firing rate

    Returns
    -------
    rate_map : 2D array of shape (len(x_centers), len(y_centers))
    """
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    rate = baseline + peak * np.exp(
        -((x_grid - mu_x)**2 / (2 * sigma_x**2) +
          (y_grid - mu_y)**2 / (2 * sigma_y**2))
    )
    return rate


def gaussian_log_likelihood(params, spike_map, occupancy, x_centers, y_centers):
    """
    Compute log-likelihood of spike counts under Gaussian place field model.

    Parameters
    ----------
    params : [mu_x, mu_y, sigma_x, sigma_y, peak, baseline]
    spike_map : 2D array of spike counts
    occupancy : 2D array of time spent per bin (seconds)

    Returns
    -------
    log_lik : float
    """
    mu_x, mu_y, sigma_x, sigma_y, peak, baseline = params

    # Compute expected rates
    rate_map = gaussian_rate_map(x_centers, y_centers, mu_x, mu_y,
                                  sigma_x, sigma_y, peak, baseline)

    # Expected spike counts = rate * occupancy
    expected = rate_map * occupancy

    # Poisson log-likelihood (ignoring constant terms)
    # log P(n | λ) = n*log(λ) - λ - log(n!)
    # We ignore log(n!) as it doesn't depend on parameters
    valid = (occupancy > 0) & (expected > 0)
    log_lik = np.sum(
        spike_map[valid] * np.log(expected[valid]) - expected[valid]
    )

    return log_lik


def gaussian_log_prior(params, x_range, y_range):
    """
    Compute log-prior for Gaussian place field parameters.

    Priors:
    - mu_x, mu_y: Uniform over spatial range
    - sigma_x, sigma_y: Half-normal with scale = range/4
    - peak: Exponential with mean = 10 Hz
    - baseline: Exponential with mean = 1 Hz

    Parameters
    ----------
    params : [mu_x, mu_y, sigma_x, sigma_y, peak, baseline]
    x_range, y_range : tuples of (min, max) for spatial bounds
    """
    mu_x, mu_y, sigma_x, sigma_y, peak, baseline = params

    log_prior = 0.0

    # Uniform priors on location (return -inf if outside bounds)
    if not (x_range[0] <= mu_x <= x_range[1]):
        return -np.inf
    if not (y_range[0] <= mu_y <= y_range[1]):
        return -np.inf

    # Positive constraints on other parameters
    if sigma_x <= 0 or sigma_y <= 0 or peak <= 0 or baseline < 0:
        return -np.inf

    # Half-normal prior on sigmas (scale = range/4)
    sigma_scale_x = (x_range[1] - x_range[0]) / 4
    sigma_scale_y = (y_range[1] - y_range[0]) / 4
    log_prior += -0.5 * (sigma_x / sigma_scale_x)**2
    log_prior += -0.5 * (sigma_y / sigma_scale_y)**2

    # Exponential prior on peak (mean = 10 Hz)
    log_prior += -peak / 10.0

    # Exponential prior on baseline (mean = 1 Hz)
    log_prior += -baseline / 1.0

    return log_prior


def fit_gaussian_place_field(spike_map, occupancy, x_edges, y_edges,
                              n_samples=2000, burn_in=500, seed=None):
    """
    Fit a Gaussian place field model using MCMC (Metropolis-Hastings).

    Parameters
    ----------
    spike_map : 2D array of spike counts
    occupancy : 2D array of time spent per bin
    x_edges, y_edges : bin edges
    n_samples : number of MCMC samples
    burn_in : number of samples to discard
    seed : random seed

    Returns
    -------
    samples : dict with arrays for each parameter
    acceptance_rate : float
    """
    if seed is not None:
        np.random.seed(seed)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_range = (x_edges[0], x_edges[-1])
    y_range = (y_edges[0], y_edges[-1])

    # Initialize parameters from data
    rate_map = spike_map / np.maximum(occupancy, 1e-10)
    peak_idx = np.unravel_index(np.nanargmax(rate_map), rate_map.shape)
    init_mu_x = x_centers[peak_idx[0]]
    init_mu_y = y_centers[peak_idx[1]]
    init_sigma = (x_range[1] - x_range[0]) / 6
    init_peak = np.nanmax(rate_map)
    init_baseline = np.nanpercentile(rate_map[occupancy > 0], 10)

    params = np.array([init_mu_x, init_mu_y, init_sigma, init_sigma,
                       max(init_peak, 1.0), max(init_baseline, 0.1)])

    # Proposal standard deviations
    proposal_std = np.array([
        (x_range[1] - x_range[0]) / 20,  # mu_x
        (y_range[1] - y_range[0]) / 20,  # mu_y
        (x_range[1] - x_range[0]) / 40,  # sigma_x
        (y_range[1] - y_range[0]) / 40,  # sigma_y
        init_peak / 10,                   # peak
        0.5                               # baseline
    ])

    # Storage
    all_samples = np.zeros((n_samples, 6))
    n_accepted = 0

    # Current log-posterior
    current_log_post = (
        gaussian_log_likelihood(params, spike_map, occupancy, x_centers, y_centers) +
        gaussian_log_prior(params, x_range, y_range)
    )

    # MCMC loop
    for i in range(n_samples):
        # Propose new parameters
        proposal = params + proposal_std * np.random.randn(6)

        # Compute log-posterior for proposal
        prop_log_post = (
            gaussian_log_likelihood(proposal, spike_map, occupancy, x_centers, y_centers) +
            gaussian_log_prior(proposal, x_range, y_range)
        )

        # Accept/reject
        log_alpha = prop_log_post - current_log_post
        if np.log(np.random.rand()) < log_alpha:
            params = proposal
            current_log_post = prop_log_post
            n_accepted += 1

        all_samples[i] = params

    # Discard burn-in
    samples = all_samples[burn_in:]
    acceptance_rate = n_accepted / n_samples

    return {
        'mu_x': samples[:, 0],
        'mu_y': samples[:, 1],
        'sigma_x': samples[:, 2],
        'sigma_y': samples[:, 3],
        'peak': samples[:, 4],
        'baseline': samples[:, 5],
        'x_centers': x_centers,
        'y_centers': y_centers
    }, acceptance_rate


def plot_gaussian_posterior(samples, full_rate_map, x_edges, y_edges, n_synthetic=4):
    """
    Plot posterior distributions and synthetic place fields from Gaussian model.

    Parameters
    ----------
    samples : dict from fit_gaussian_place_field
    full_rate_map : observed rate map
    x_edges, y_edges : bin edges
    n_synthetic : number of synthetic fields to show
    """
    x_centers = samples['x_centers']
    y_centers = samples['y_centers']

    # Compute posterior means
    post_means = {k: np.mean(samples[k]) for k in ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'peak', 'baseline']}

    # Generate posterior mean field
    mean_field = gaussian_rate_map(
        x_centers, y_centers,
        post_means['mu_x'], post_means['mu_y'],
        post_means['sigma_x'], post_means['sigma_y'],
        post_means['peak'], post_means['baseline']
    )

    # Generate synthetic fields from posterior samples
    n_total = len(samples['mu_x'])
    sample_indices = np.random.choice(n_total, n_synthetic, replace=False)
    synthetic_fields = []
    for idx in sample_indices:
        field = gaussian_rate_map(
            x_centers, y_centers,
            samples['mu_x'][idx], samples['mu_y'][idx],
            samples['sigma_x'][idx], samples['sigma_y'][idx],
            samples['peak'][idx], samples['baseline'][idx]
        )
        synthetic_fields.append(field)

    # Plot
    fig = plt.figure(figsize=(16, 10))

    # Top row: rate maps
    n_cols = n_synthetic + 2
    vmax = np.nanpercentile(full_rate_map.ravel(), 98)

    ax = plt.subplot(2, n_cols, 1)
    ax.imshow(full_rate_map.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot')
    ax.set_title('Observed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = plt.subplot(2, n_cols, 2)
    ax.imshow(mean_field.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot')
    ax.set_title('Posterior mean')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for i, field in enumerate(synthetic_fields):
        ax = plt.subplot(2, n_cols, i + 3)
        ax.imshow(field.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='hot')
        ax.set_title(f'Sample {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Bottom row: parameter posteriors
    param_names = ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'peak', 'baseline']
    param_labels = ['μx', 'μy', 'σx', 'σy', 'Peak (Hz)', 'Baseline (Hz)']

    for i, (name, label) in enumerate(zip(param_names, param_labels)):
        ax = plt.subplot(2, n_cols, n_cols + i + 1)
        ax.hist(samples[name], bins=30, density=True, alpha=0.7, color='steelblue')
        ax.axvline(post_means[name], color='k', linestyle='--', lw=2, label='Mean')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label}: {post_means[name]:.3f}')

    plt.suptitle('Gaussian place field model: posterior distributions', y=1.02)
    plt.tight_layout()
    plt.savefig('place_field_gaussian_model.png', dpi=150)
    plt.show()

    return mean_field, synthetic_fields


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


def compute_center_of_mass(rate_map, x_edges, y_edges):
    """
    Compute the center of mass (centroid) of a rate map.

    Parameters
    ----------
    rate_map : 2D array with NaN for unvisited bins
    x_edges, y_edges : bin edges

    Returns
    -------
    (cx, cy) : center of mass coordinates, or (nan, nan) if invalid
    """
    if rate_map is None:
        return np.nan, np.nan

    # Bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Replace NaN with 0 for calculation
    rm = np.nan_to_num(rate_map, nan=0.0)
    total = rm.sum()

    if total == 0:
        return np.nan, np.nan

    # Weighted average of bin centers
    cx = np.sum(rm * x_centers[:, np.newaxis]) / total
    cy = np.sum(rm * y_centers[np.newaxis, :]) / total

    return cx, cy


def pairwise_com_shift(rate_maps, x_edges, y_edges):
    """
    Compute center-of-mass shift between successive rate maps.

    Returns
    -------
    shifts : list of floats, Euclidean distance between COMs (in spatial units)
    """
    # Compute COM for each rate map
    coms = [compute_center_of_mass(rm, x_edges, y_edges) for rm in rate_maps]

    shifts = []
    for i in range(len(coms) - 1):
        cx1, cy1 = coms[i]
        cx2, cy2 = coms[i + 1]

        if np.isnan(cx1) or np.isnan(cx2):
            shifts.append(np.nan)
        else:
            shift = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            shifts.append(shift)

    return shifts, coms


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

def plot_results(rate_maps, correlations, com_shifts, coms, post_mean, post_var,
                 x_edges, y_edges, x_posterior=None, y_posterior=None, blocks=None):
    """Plot rate maps, correlation, COM shift, posterior, and marginal posteriors."""
    n_blocks = len(rate_maps)
    valid_maps = [rm for rm in rate_maps if rm is not None]

    if not valid_maps:
        print("No valid rate maps to plot.")
        return

    vmax = np.nanpercentile(np.concatenate([rm.ravel() for rm in valid_maps]), 98)

    n_cols = max(n_blocks, 5)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8))

    # Top row: rate maps with trajectory and COM overlay
    for i, rm in enumerate(rate_maps):
        ax = axes[0, i]
        if rm is not None:
            ax.imshow(rm.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                      cmap='hot')
        # Overlay trajectory if available
        if blocks is not None and i < len(blocks):
            pos_block, _ = blocks[i]
            if len(pos_block) > 0:
                ax.plot(pos_block[:, 1], pos_block[:, 2],
                        'b-', alpha=0.3, linewidth=0.5)
        # Mark center of mass
        cx, cy = coms[i]
        if not np.isnan(cx):
            ax.plot(cx, cy, 'c+', markersize=12, markeredgewidth=2)
        ax.set_title(f'Block {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

    # Hide unused axes in top row
    for i in range(n_blocks, n_cols):
        axes[0, i].set_visible(False)

    # Bottom row: metrics and marginal distributions
    # 1. Correlation time course
    ax_corr = axes[1, 0]
    rs = [r for r, p in correlations]
    ax_corr.plot(range(1, len(rs) + 1), rs, 'ko-')
    ax_corr.set_xlabel('Block transition')
    ax_corr.set_ylabel('Spatial correlation (r)')
    ax_corr.set_ylim(-0.2, 1.05)
    ax_corr.set_title('Pairwise correlation')
    ax_corr.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 2. COM shift time course
    ax_com = axes[1, 1]
    ax_com.plot(range(1, len(com_shifts) + 1), com_shifts, 'ko-')
    ax_com.set_xlabel('Block transition')
    ax_com.set_ylabel('COM shift (spatial units)')
    ax_com.set_ylim(0, max(com_shifts) * 1.2 if any(s > 0 for s in com_shifts if not np.isnan(s)) else 0.1)
    ax_com.set_title('Center of mass shift')
    ax_com.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # 3. Posterior on correlation
    ax_post = axes[1, 2]
    z_range = np.linspace(-1, 2, 200)
    posterior = np.exp(-0.5 * (z_range - post_mean) ** 2 / post_var)
    posterior /= posterior.sum() * (z_range[1] - z_range[0])
    r_range = np.tanh(z_range)
    ax_post.plot(r_range, posterior, 'b-', lw=2)
    ax_post.fill_between(r_range, posterior, alpha=0.2)
    ax_post.set_xlabel('Spatial correlation (r)')
    ax_post.set_ylabel('Posterior density')
    ax_post.set_title(f'Posterior: mean r ≈ {np.tanh(post_mean):.2f}')
    ax_post.set_xlim(-0.5, 1.05)

    # 4 & 5. Posterior marginal firing rate distributions (x and y)
    if x_posterior is not None and y_posterior is not None:
        # X marginal posterior
        x_mean, x_lower, x_upper = gamma_credible_interval(
            x_posterior['alpha'], x_posterior['beta'], ci=0.95
        )
        ax_x = axes[1, 3]
        ax_x.plot(x_posterior['centers'], x_mean, 'k-', lw=2, label='Posterior mean')
        ax_x.fill_between(x_posterior['centers'], x_lower, x_upper, alpha=0.3, label='95% CI')
        ax_x.set_xlabel('x position')
        ax_x.set_ylabel('Firing rate (Hz)')
        ax_x.set_title('X marginal posterior')
        ax_x.set_xlim(x_edges[0], x_edges[-1])
        ax_x.set_ylim(0, None)

        # Y marginal posterior
        y_mean, y_lower, y_upper = gamma_credible_interval(
            y_posterior['alpha'], y_posterior['beta'], ci=0.95
        )
        ax_y = axes[1, 4]
        ax_y.plot(y_posterior['centers'], y_mean, 'k-', lw=2, label='Posterior mean')
        ax_y.fill_between(y_posterior['centers'], y_lower, y_upper, alpha=0.3, label='95% CI')
        ax_y.set_xlabel('y position')
        ax_y.set_ylabel('Firing rate (Hz)')
        ax_y.set_title('Y marginal posterior')
        ax_y.set_xlim(y_edges[0], y_edges[-1])
        ax_y.set_ylim(0, None)
    else:
        axes[1, 3].set_visible(False)
        axes[1, 4].set_visible(False)

    # Hide any remaining unused axes
    for i in range(5, n_cols):
        axes[1, i].set_visible(False)

    plt.tight_layout()
    plt.savefig('place_field_stability.png', dpi=150)
    plt.show()


def plot_synthetic_place_fields(full_rate_map, x_posterior, y_posterior, x_edges, y_edges,
                                 n_samples=4, seed=None):
    """
    Generate and plot synthetic place fields by sampling from the posterior distributions.

    Assumes separability: rate(x,y) ∝ rate_x(x) * rate_y(y)
    This is a reasonable approximation for Gaussian-like place fields.

    Parameters
    ----------
    full_rate_map : 2D array, observed rate map for comparison
    x_posterior, y_posterior : dicts with Gamma posterior parameters
    x_edges, y_edges : bin edges
    n_samples : number of synthetic fields to generate
    seed : random seed for reproducibility
    """
    from scipy.stats import gamma

    if seed is not None:
        np.random.seed(seed)

    # Sample from marginal posteriors
    x_samples = gamma.rvs(x_posterior['alpha'], scale=1/x_posterior['beta'],
                          size=(n_samples, len(x_posterior['alpha'])))
    y_samples = gamma.rvs(y_posterior['alpha'], scale=1/y_posterior['beta'],
                          size=(n_samples, len(y_posterior['alpha'])))

    # Compute posterior means and credible intervals
    x_mean, x_lower, x_upper = gamma_credible_interval(
        x_posterior['alpha'], x_posterior['beta'], ci=0.95
    )
    y_mean, y_lower, y_upper = gamma_credible_interval(
        y_posterior['alpha'], y_posterior['beta'], ci=0.95
    )

    # Create synthetic 2D rate maps (outer product, normalized)
    synthetic_maps = []
    for i in range(n_samples):
        # Outer product to create 2D field
        rate_2d = np.outer(x_samples[i], y_samples[i])
        # Normalize to match observed peak rate
        if np.nanmax(full_rate_map) > 0 and rate_2d.max() > 0:
            rate_2d = rate_2d * (np.nanmax(full_rate_map) / rate_2d.max())
        synthetic_maps.append(rate_2d)

    # Posterior mean field
    mean_field = np.outer(x_mean, y_mean)
    if np.nanmax(full_rate_map) > 0 and mean_field.max() > 0:
        mean_field = mean_field * (np.nanmax(full_rate_map) / mean_field.max())

    # Plot: 2 rows - top for 2D maps, bottom for marginal posteriors
    n_cols = n_samples + 2  # observed + mean + samples
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7),
                             gridspec_kw={'height_ratios': [1.2, 1]})

    vmax = np.nanpercentile(full_rate_map.ravel(), 98)

    # Define colors for samples
    sample_colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

    # --- Top row: 2D rate maps ---
    # Observed rate map
    ax = axes[0, 0]
    ax.imshow(full_rate_map.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
              cmap='hot')
    ax.set_title('Observed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Posterior mean field
    ax = axes[0, 1]
    ax.imshow(mean_field.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
              cmap='hot')
    ax.set_title('Posterior mean')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Sampled fields
    for i, syn_map in enumerate(synthetic_maps):
        ax = axes[0, i + 2]
        ax.imshow(syn_map.T, origin='lower', aspect='auto', vmin=0, vmax=vmax,
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                  cmap='hot')
        ax.set_title(f'Sample {i + 1}', color=sample_colors[i])
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # --- Bottom row: Marginal posteriors with samples ---
    # X marginal posterior
    ax_x = axes[1, 0]
    ax_x.fill_between(x_posterior['centers'], x_lower, x_upper, alpha=0.3, color='gray',
                      label='95% CI')
    ax_x.plot(x_posterior['centers'], x_mean, 'k-', lw=2, label='Posterior mean')
    for i in range(n_samples):
        ax_x.plot(x_posterior['centers'], x_samples[i], '-', color=sample_colors[i],
                  alpha=0.7, lw=1.5, label=f'Sample {i+1}')
    ax_x.set_xlabel('x position')
    ax_x.set_ylabel('Firing rate (Hz)')
    ax_x.set_title('X marginal posterior')
    ax_x.set_xlim(x_edges[0], x_edges[-1])
    ax_x.set_ylim(0, None)

    # Y marginal posterior
    ax_y = axes[1, 1]
    ax_y.fill_between(y_posterior['centers'], y_lower, y_upper, alpha=0.3, color='gray',
                      label='95% CI')
    ax_y.plot(y_posterior['centers'], y_mean, 'k-', lw=2, label='Posterior mean')
    for i in range(n_samples):
        ax_y.plot(y_posterior['centers'], y_samples[i], '-', color=sample_colors[i],
                  alpha=0.7, lw=1.5, label=f'Sample {i+1}')
    ax_y.set_xlabel('y position')
    ax_y.set_ylabel('Firing rate (Hz)')
    ax_y.set_title('Y marginal posterior')
    ax_y.set_xlim(y_edges[0], y_edges[-1])
    ax_y.set_ylim(0, None)

    # Add legend to one of the marginal plots
    ax_y.legend(loc='upper left', fontsize=8)

    # Hide unused axes in bottom row
    for i in range(2, n_cols):
        axes[1, i].set_visible(False)

    plt.suptitle('Generative model: synthetic place fields from posterior', y=1.02)
    plt.tight_layout()
    plt.savefig('place_field_synthetic.png', dpi=150)
    plt.show()

    return synthetic_maps, mean_field


# --- Main pipeline ---

def run_analysis(position, spike_times, n_bins=30, n_blocks=4, smooth_sigma=2.0):
    """
    Full pipeline computing spatial correlation, COM shift, and marginal posteriors.

    Parameters
    ----------
    position : (N, 3) array [timestamp, x, y]
    spike_times : (S,) array
    n_bins : spatial bins per dimension
    n_blocks : number of temporal blocks
    smooth_sigma : Gaussian smoothing width in bins

    Returns
    -------
    full_rate_map : rate map for entire session
    block_maps : list of rate maps per block
    correlations : list of (r, p) tuples
    com_shifts : list of COM shift distances
    corr_posterior : (post_mean, post_var) for correlation
    marginal_posteriors : (x_posterior, y_posterior) dicts with Gamma parameters
    """
    dt = np.median(np.diff(position[:, 0]))

    # Spatial bin edges
    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    # Full-session maps (raw counts for Bayesian inference)
    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    full_rate_map = estimate_rate_map(spk, occ, smooth_sigma)

    # Compute marginal posteriors from raw spike counts and occupancy
    x_posterior, y_posterior = compute_marginal_posteriors(spk, occ, x_edges, y_edges)

    print(f"Session duration: {position[-1, 0] - position[0, 0]:.1f} s")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Peak rate: {np.nanmax(full_rate_map):.1f} Hz")

    # Temporal blocks
    blocks = split_into_blocks(position, spike_times, n_blocks)
    block_maps = compute_block_rate_maps(blocks, x_edges, y_edges, dt, smooth_sigma)

    # Stability metrics
    corrs = pairwise_spatial_correlation(block_maps)
    com_shifts, coms = pairwise_com_shift(block_maps, x_edges, y_edges)
    post_mean, post_var, rs = bayesian_stability_estimate(corrs)

    print(f"\nPairwise correlations: {[f'{r:.3f}' for r, p in corrs]}")
    print(f"Pairwise COM shifts: {[f'{s:.3f}' for s in com_shifts]}")
    print(f"Posterior mean correlation: {np.tanh(post_mean):.3f}")
    print(f"Posterior 95% CI: [{np.tanh(post_mean - 1.96*np.sqrt(post_var)):.3f}, "
          f"{np.tanh(post_mean + 1.96*np.sqrt(post_var)):.3f}]")

    plot_results(block_maps, corrs, com_shifts, coms, post_mean, post_var,
                 x_edges, y_edges, x_posterior=x_posterior, y_posterior=y_posterior,
                 blocks=blocks)

    return {
        'full_rate_map': full_rate_map,
        'block_maps': block_maps,
        'correlations': corrs,
        'com_shifts': com_shifts,
        'corr_posterior': (post_mean, post_var),
        'x_posterior': x_posterior,
        'y_posterior': y_posterior,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'spike_map': spk,
        'occupancy': occ
    }


def simulate_foraging_trajectory(duration, fs_pos, bounds=(0, 1), speed=0.05):
    """
    Simulate uniform foraging using an Ornstein-Uhlenbeck process.

    This gives a random walk that mean-reverts to center, providing
    more uniform spatial coverage than a pure random walk.

    Parameters
    ----------
    duration : float — session duration in seconds
    fs_pos : int — position sampling rate (Hz)
    bounds : tuple — (min, max) for x and y
    speed : float — movement speed parameter

    Returns
    -------
    position : (N, 3) array of [timestamp, x, y]
    """
    n_samples = int(duration * fs_pos)
    dt = 1.0 / fs_pos
    t = np.linspace(0, duration, n_samples)

    # OU process parameters
    theta = 0.1  # mean reversion rate
    sigma = speed * np.sqrt(2 * theta)  # diffusion scaled for target speed
    center = (bounds[0] + bounds[1]) / 2

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    x[0] = center
    y[0] = center

    for i in range(1, n_samples):
        # OU update with reflection at boundaries
        x[i] = x[i-1] + theta * (center - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        y[i] = y[i-1] + theta * (center - y[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()

        # Reflect at boundaries
        x[i] = np.clip(x[i], bounds[0], bounds[1])
        y[i] = np.clip(y[i], bounds[0], bounds[1])

    return np.column_stack([t, x, y])


# --- Example with synthetic data ---

if __name__ == '__main__':
    # np.random.seed(42)

    # # Simulate foraging in a 1m x 1m box
    # duration = 600  # seconds
    # fs_pos = 100  # position sampling rate

    # # Use OU process for uniform exploration
    # position = simulate_foraging_trajectory(duration, fs_pos, bounds=(0, 1), speed=0.5)
    # t, x, y = position[:, 0], position[:, 1], position[:, 2]
    # n_samples = len(t)

    # # Simulate a place cell: Gaussian field centered at (0.6, 0.4)
    # # with optional drift in preferred location over time
    # peak_rate = 30  # Hz
    # field_width = 0.25

    # # Drift parameters (set to 0 for stable field)
    # x_drift = 1.0
    # y_drift = 1.0
    # cx = 0 + x_drift * (t / duration)
    # cy = 0 + y_drift * (t / duration)

    # rate = peak_rate * np.exp(
    #     -((x - cx) ** 2 + (y - cy) ** 2) / (2 * field_width ** 2)
    # )

    # # Generate spikes as inhomogeneous Poisson process
    # spike_prob = rate / fs_pos
    # spikes_bool = np.random.rand(n_samples) < spike_prob
    # spike_times = t[spikes_bool]

    # print(f"Generated {len(spike_times)} spikes\n")

    # === Real data example ===
    from SessionData import SessionData

    base_path = "d:/data/"
    mouse_id = "7012"
    session_id = "m10"
    experiment = "clickbait-motivate"
    cluster_id = 19

    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True)

    position = data.events[['timestamp_ms', 'nose_x', 'nose_y']].to_numpy()
    spike_times = data.clusters[cluster_id]['spike_times']

    # === Run analysis ===
    results = run_analysis(
        position, spike_times, n_bins=32, n_blocks=5, smooth_sigma=2.0
    )

    # Generate synthetic place fields from non-parametric posterior
    synthetic_maps, mean_field = plot_synthetic_place_fields(
        results['full_rate_map'],
        results['x_posterior'],
        results['y_posterior'],
        results['x_edges'],
        results['y_edges'],
        n_samples=4,
        seed=123
    )

    # Fit parametric Gaussian place field model
    print("\nFitting Gaussian place field model...")
    gaussian_samples, acceptance_rate = fit_gaussian_place_field(
        results['spike_map'],
        results['occupancy'],
        results['x_edges'],
        results['y_edges'],
        n_samples=12000,
        burn_in=2000,
        seed=42
    )
    print(f"MCMC acceptance rate: {acceptance_rate:.2%}")

    # Plot Gaussian model posterior
    gaussian_mean_field, gaussian_synthetic = plot_gaussian_posterior(
        gaussian_samples,
        results['full_rate_map'],
        results['x_edges'],
        results['y_edges'],
        n_synthetic=4
    )


