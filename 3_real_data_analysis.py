"""
Real data analysis with Gaussian Process place field model.

Key differences from synthetic data:
- Timestamps may be in ms (converted to seconds)
- Spatial coordinates normalized to [0,1] for GP fitting
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# --- Core estimation functions ---

def compute_occupancy(position, x_edges, y_edges, dt):
    """Compute time-spent-per-bin (occupancy map) from position data."""
    occ, _, _ = np.histogram2d(
        position[:, 1], position[:, 2],
        bins=[x_edges, y_edges]
    )
    return occ * dt


def compute_spike_map(spike_times, position, x_edges, y_edges):
    """Bin spike counts into spatial bins by interpolating spike positions."""
    spike_x = np.interp(spike_times, position[:, 0], position[:, 1])
    spike_y = np.interp(spike_times, position[:, 0], position[:, 2])
    spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
    return spike_map


def estimate_rate_map(spike_map, occupancy, smooth_sigma=2.0, min_occupancy=0.1):
    """Compute smoothed firing rate map."""
    smooth_spikes = gaussian_filter(spike_map.astype(float), sigma=smooth_sigma)
    smooth_occ = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)
    rate_map = np.full_like(smooth_spikes, np.nan)
    valid = smooth_occ > min_occupancy
    rate_map[valid] = smooth_spikes[valid] / smooth_occ[valid]
    return rate_map


# --- Gaussian Process model ---

def rbf_kernel(X1, X2, length_scale, amplitude=1.0):
    """Radial Basis Function kernel."""
    sq_dist = cdist(X1, X2, metric='sqeuclidean')
    return amplitude**2 * np.exp(-sq_dist / (2 * length_scale**2))


def rbf_kernel_with_noise(X1, X2, length_scale, amplitude=1.0, noise=1e-6):
    """RBF kernel with diagonal noise term."""
    K = rbf_kernel(X1, X2, length_scale, amplitude)
    if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
        K += noise * np.eye(K.shape[0])
    return K


class GaussianProcessPlaceField:
    """
    Gaussian Process model for place field estimation.
    Models log-firing rate as a GP: log(rate(x, y)) ~ GP(mean, k(x, x'))
    """

    def __init__(self, length_scale=0.15, amplitude=2.0, noise=0.5, mean=0.0):
        self.length_scale = length_scale
        self.amplitude = amplitude
        self.noise = noise
        self.mean = mean

        self.X_train = None
        self.y_train = None
        self.L = None
        self.lower = None
        self.alpha = None

    def fit(self, positions, rates, min_rate=0.1):
        """Fit the GP to observed firing rates."""
        self.X_train = positions
        self.y_train = np.log(np.maximum(rates, min_rate))

        K = rbf_kernel_with_noise(
            self.X_train, self.X_train,
            self.length_scale, self.amplitude, self.noise**2
        )

        self.L, self.lower = cho_factor(K, lower=True)
        self.alpha = cho_solve((self.L, self.lower), self.y_train - self.mean)

        return self

    def predict(self, positions, return_std=False):
        """Predict firing rates at new positions."""
        K_star = rbf_kernel(positions, self.X_train, self.length_scale, self.amplitude)
        log_mean = self.mean + K_star @ self.alpha

        if return_std:
            K_ss = rbf_kernel_with_noise(
                positions, positions,
                self.length_scale, self.amplitude, 1e-6
            )
            v = cho_solve((self.L, self.lower), K_star.T)
            log_var = np.diag(K_ss) - np.sum(K_star.T * v, axis=0)
            log_std = np.sqrt(np.maximum(log_var, 1e-10))

            mean_rate = np.exp(log_mean)
            std_rate = mean_rate * log_std

            return mean_rate, std_rate

        return np.exp(log_mean)

    def sample(self, positions, n_samples=1):
        """Sample from the posterior distribution."""
        K_star = rbf_kernel(positions, self.X_train, self.length_scale, self.amplitude)
        log_mean = self.mean + K_star @ self.alpha

        K_ss = rbf_kernel_with_noise(
            positions, positions,
            self.length_scale, self.amplitude, 1e-6
        )
        v = cho_solve((self.L, self.lower), K_star.T)
        log_cov = K_ss - K_star @ v

        log_cov = (log_cov + log_cov.T) / 2
        eigvals = np.linalg.eigvalsh(log_cov)
        if eigvals.min() < 0:
            log_cov += (1e-6 - eigvals.min()) * np.eye(log_cov.shape[0])

        log_samples = np.random.multivariate_normal(log_mean, log_cov, size=n_samples)

        return np.exp(log_samples)

    def optimize_hyperparameters(self, positions, rates, min_rate=0.1,
                                  length_scale_bounds=(0.05, 0.5),
                                  amplitude_bounds=(0.5, 5.0),
                                  noise_bounds=(0.1, 2.0)):
        """Optimize hyperparameters by maximizing marginal likelihood."""
        y = np.log(np.maximum(rates, min_rate))
        n = len(y)

        def neg_log_marginal_likelihood(params):
            ls, amp, noise = params
            K = rbf_kernel_with_noise(positions, positions, ls, amp, noise**2)

            try:
                L, lower = cho_factor(K, lower=True)
                alpha = cho_solve((L, lower), y - self.mean)

                log_ml = -0.5 * (y - self.mean) @ alpha
                log_ml -= np.sum(np.log(np.diag(L)))
                log_ml -= 0.5 * n * np.log(2 * np.pi)

                return -log_ml
            except np.linalg.LinAlgError:
                return np.inf

        x0 = [self.length_scale, self.amplitude, self.noise]
        bounds = [length_scale_bounds, amplitude_bounds, noise_bounds]

        result = minimize(neg_log_marginal_likelihood, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.length_scale, self.amplitude, self.noise = result.x
            print(f"Optimized hyperparameters: length_scale={self.length_scale:.3f}, "
                  f"amplitude={self.amplitude:.3f}, noise={self.noise:.3f}")
        else:
            print("Hyperparameter optimization did not converge, using initial values.")

        return self.fit(positions, rates, min_rate)


def fit_gp_place_field(spike_map, occupancy, x_edges, y_edges,
                       optimize=True, min_occupancy=0.1, **gp_kwargs):
    """
    Fit a GP place field model to binned spike data.

    IMPORTANT: Normalizes positions to [0,1] range for GP fitting.
    """
    # Compute bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create grid of all positions
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions_raw = np.column_stack([xx.ravel(), yy.ravel()])

    # Normalize positions to [0, 1] for GP fitting
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    # Use the larger dimension to normalize (preserves aspect ratio)
    scale = max(x_max - x_min, y_max - y_min)

    grid_positions_norm = np.column_stack([
        (grid_positions_raw[:, 0] - x_min) / scale,
        (grid_positions_raw[:, 1] - y_min) / scale
    ])

    # Compute rates
    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    # Get valid positions and rates for fitting
    valid_flat = valid.ravel()
    train_positions = grid_positions_norm[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    print(f"Training GP on {len(train_positions)} valid bins")
    print(f"Rate range: {train_rates.min():.2f} - {train_rates.max():.2f} Hz")

    # Create and fit GP
    gp = GaussianProcessPlaceField(**gp_kwargs)

    if optimize:
        gp.optimize_hyperparameters(train_positions, train_rates)
    else:
        gp.fit(train_positions, train_rates)

    return gp, grid_positions_norm, grid_positions_raw, valid.ravel(), scale


def predict_rate_map(gp, grid_positions, shape, return_std=False):
    """Predict a full rate map from a fitted GP."""
    if return_std:
        mean, std = gp.predict(grid_positions, return_std=True)
        return mean.reshape(shape), std.reshape(shape)
    else:
        mean = gp.predict(grid_positions)
        return mean.reshape(shape)


def sample_rate_maps(gp, grid_positions, shape, n_samples=4):
    """Sample rate maps from the GP posterior."""
    raw_samples = gp.sample(grid_positions, n_samples)
    return [s.reshape(shape) for s in raw_samples]


# --- Visualization ---

def plot_gp_results(observed_rate_map, gp, grid_positions, shape, x_edges, y_edges,
                    n_samples=4, seed=None):
    """Plot GP place field results."""
    if seed is not None:
        np.random.seed(seed)

    mean_map, std_map = predict_rate_map(gp, grid_positions, shape, return_std=True)
    sample_maps = sample_rate_maps(gp, grid_positions, shape, n_samples)

    n_cols = n_samples + 3
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7),
                             gridspec_kw={'height_ratios': [1.2, 1]})

    vmax = np.nanpercentile(observed_rate_map.ravel(), 98)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # Top row: rate maps
    ax = axes[0, 0]
    im = ax.imshow(observed_rate_map.T, origin='lower', aspect='auto',
                   vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('Observed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes[0, 1]
    ax.imshow(mean_map.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('GP posterior mean')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes[0, 2]
    im_std = ax.imshow(std_map.T, origin='lower', aspect='auto', extent=extent, cmap='viridis')
    ax.set_title('Uncertainty (std)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im_std, ax=ax, fraction=0.046)

    for i, sample_map in enumerate(sample_maps):
        ax = axes[0, i + 3]
        ax.imshow(sample_map.T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title(f'Sample {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Bottom row: marginal slices
    peak_idx = np.unravel_index(np.nanargmax(mean_map), mean_map.shape)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    ax = axes[1, 0]
    ax.plot(x_centers, observed_rate_map[:, peak_idx[1]], 'k.', alpha=0.5, label='Observed')
    ax.plot(x_centers, mean_map[:, peak_idx[1]], 'b-', lw=2, label='GP mean')
    ax.fill_between(x_centers,
                    mean_map[:, peak_idx[1]] - 2*std_map[:, peak_idx[1]],
                    mean_map[:, peak_idx[1]] + 2*std_map[:, peak_idx[1]],
                    alpha=0.3, label='95% CI')
    for sample_map in sample_maps:
        ax.plot(x_centers, sample_map[:, peak_idx[1]], '-', alpha=0.5, lw=1)
    ax.set_xlabel('x position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'X slice at y={y_centers[peak_idx[1]]:.1f}')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(y_centers, observed_rate_map[peak_idx[0], :], 'k.', alpha=0.5, label='Observed')
    ax.plot(y_centers, mean_map[peak_idx[0], :], 'b-', lw=2, label='GP mean')
    ax.fill_between(y_centers,
                    mean_map[peak_idx[0], :] - 2*std_map[peak_idx[0], :],
                    mean_map[peak_idx[0], :] + 2*std_map[peak_idx[0], :],
                    alpha=0.3, label='95% CI')
    for sample_map in sample_maps:
        ax.plot(y_centers, sample_map[peak_idx[0], :], '-', alpha=0.5, lw=1)
    ax.set_xlabel('y position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Y slice at x={x_centers[peak_idx[0]]:.1f}')

    ax = axes[1, 2]
    ax.axis('off')
    hp_text = (f"GP Hyperparameters:\n\n"
               f"Length scale: {gp.length_scale:.3f}\n"
               f"Amplitude: {gp.amplitude:.3f}\n"
               f"Noise: {gp.noise:.3f}\n\n"
               f"Peak observed: {np.nanmax(observed_rate_map):.1f} Hz\n"
               f"Peak GP mean: {np.max(mean_map):.1f} Hz")
    ax.text(0.1, 0.5, hp_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes)

    for i in range(3, n_cols):
        axes[1, i].set_visible(False)

    plt.suptitle('Gaussian Process place field model', y=1.02)
    plt.tight_layout()
    plt.savefig('real_data_gp_model.png', dpi=150)
    plt.show()

    return mean_map, std_map, sample_maps


# --- Main ---

if __name__ == '__main__':
    from SessionData import SessionData

    # Data configuration
    base_path = "d:/data/"
    mouse_id = "7012"
    session_id = "m10"
    experiment = "clickbait-motivate"
    cluster_id = 19

    print(f"Loading data: {mouse_id}/{session_id}, cluster {cluster_id}")
    print("-" * 50)

    # Load data
    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True
    )

    # Extract position and spike times
    # Convert timestamps from ms to seconds
    position = data.events[['timestamp_ms', 'nose_x', 'nose_y']].to_numpy()
    position[:, 0] = position[:, 0] / 1000.0  # ms -> seconds

    spike_times = data.clusters[cluster_id]['spike_times']
    spike_times = spike_times / 1000.0  # ms -> seconds

    print("-" * 50)
    print()

    # Analysis parameters
    n_bins = 25
    smooth_sigma = 1.0

    # Compute dt in seconds
    dt = np.median(np.diff(position[:, 0]))

    # Spatial bin edges
    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    # Compute maps
    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    observed_rate_map = estimate_rate_map(spk, occ, smooth_sigma)

    print(f"Session duration: {position[-1, 0] - position[0, 0]:.1f} s")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Peak rate: {np.nanmax(observed_rate_map):.1f} Hz")
    print()

    # Fit GP model
    print("Fitting GP place field model...")
    gp, grid_pos_norm, grid_pos_raw, valid_mask, scale = fit_gp_place_field(
        spk, occ, x_edges, y_edges,
        optimize=True,
        length_scale=0.15,
        amplitude=2.0,
        noise=0.5
    )

    # Plot results
    shape = (n_bins, n_bins)
    mean_map, std_map, samples = plot_gp_results(
        observed_rate_map, gp, grid_pos_norm, shape, x_edges, y_edges,
        n_samples=4, seed=123
    )

    print("\nDone!")
