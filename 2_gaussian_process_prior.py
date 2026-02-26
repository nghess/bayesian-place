"""
Gaussian Process prior for place field estimation.

This model uses a GP prior to enforce smooth spatial relationships between bins,
allowing it to capture arbitrarily-shaped place fields while maintaining smoothness.

The key advantage over parametric Gaussian models: flexibility to fit any shape
(multiple fields, elongated fields, crescent shapes, etc.) while still being smooth.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# --- Kernel functions ---

def rbf_kernel(X1, X2, length_scale, amplitude=1.0):
    """
    Radial Basis Function (squared exponential) kernel.

    k(x, x') = amplitude² * exp(-||x - x'||² / (2 * length_scale²))

    Parameters
    ----------
    X1, X2 : arrays of shape (n1, d) and (n2, d)
    length_scale : characteristic length scale
    amplitude : signal variance (amplitude²)

    Returns
    -------
    K : covariance matrix of shape (n1, n2)
    """
    sq_dist = cdist(X1, X2, metric='sqeuclidean')
    return amplitude**2 * np.exp(-sq_dist / (2 * length_scale**2))


def rbf_kernel_with_noise(X1, X2, length_scale, amplitude=1.0, noise=1e-6):
    """RBF kernel with diagonal noise term for numerical stability."""
    K = rbf_kernel(X1, X2, length_scale, amplitude)
    if X1.shape[0] == X2.shape[0] and np.allclose(X1, X2):
        K += noise * np.eye(K.shape[0])
    return K


# --- Gaussian Process class ---

class GaussianProcessPlaceField:
    """
    Gaussian Process model for place field estimation.

    Models log-firing rate as a GP:
        log(rate(x, y)) ~ GP(mean, k(x, x'))

    This ensures positive rates and allows flexible field shapes.
    """

    def __init__(self, length_scale=0.15, amplitude=2.0, noise=0.5, mean=0.0):
        """
        Parameters
        ----------
        length_scale : spatial smoothness (in same units as position)
        amplitude : signal standard deviation in log-rate space
        noise : observation noise standard deviation
        mean : prior mean in log-rate space
        """
        self.length_scale = length_scale
        self.amplitude = amplitude
        self.noise = noise
        self.mean = mean

        # Fitted attributes
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None

    def fit(self, positions, rates, min_rate=0.1):
        """
        Fit the GP to observed firing rates.

        Parameters
        ----------
        positions : (n, 2) array of (x, y) bin centers
        rates : (n,) array of firing rates
        min_rate : minimum rate to avoid log(0)
        """
        self.X_train = positions

        # Work in log space for positivity
        self.y_train = np.log(np.maximum(rates, min_rate))

        # Compute kernel matrix
        K = rbf_kernel_with_noise(
            self.X_train, self.X_train,
            self.length_scale, self.amplitude, self.noise**2
        )

        # Cholesky decomposition for efficient solving
        self.L, self.lower = cho_factor(K, lower=True)
        self.alpha = cho_solve((self.L, self.lower), self.y_train - self.mean)

        return self

    def predict(self, positions, return_std=False):
        """
        Predict firing rates at new positions.

        Parameters
        ----------
        positions : (m, 2) array of positions to predict
        return_std : whether to return standard deviation

        Returns
        -------
        mean_rate : (m,) array of predicted rates (in original space)
        std_rate : (m,) array of standard deviations (if return_std=True)
        """
        # Cross-covariance
        K_star = rbf_kernel(positions, self.X_train, self.length_scale, self.amplitude)

        # Posterior mean in log space
        log_mean = self.mean + K_star @ self.alpha

        if return_std:
            # Posterior variance
            K_ss = rbf_kernel_with_noise(
                positions, positions,
                self.length_scale, self.amplitude, 1e-6
            )
            v = cho_solve((self.L, self.lower), K_star.T)
            log_var = np.diag(K_ss) - np.sum(K_star.T * v, axis=0)
            log_std = np.sqrt(np.maximum(log_var, 1e-10))

            # Transform to rate space (approximate)
            mean_rate = np.exp(log_mean)
            # Delta method approximation for std in rate space
            std_rate = mean_rate * log_std

            return mean_rate, std_rate

        return np.exp(log_mean)

    def sample(self, positions, n_samples=1):
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        positions : (m, 2) array of positions
        n_samples : number of samples to draw

        Returns
        -------
        samples : (n_samples, m) array of rate samples
        """
        # Cross-covariance
        K_star = rbf_kernel(positions, self.X_train, self.length_scale, self.amplitude)

        # Posterior mean
        log_mean = self.mean + K_star @ self.alpha

        # Posterior covariance
        K_ss = rbf_kernel_with_noise(
            positions, positions,
            self.length_scale, self.amplitude, 1e-6
        )
        v = cho_solve((self.L, self.lower), K_star.T)
        log_cov = K_ss - K_star @ v

        # Ensure positive definiteness
        log_cov = (log_cov + log_cov.T) / 2
        eigvals = np.linalg.eigvalsh(log_cov)
        if eigvals.min() < 0:
            log_cov += (1e-6 - eigvals.min()) * np.eye(log_cov.shape[0])

        # Sample from multivariate normal in log space
        log_samples = np.random.multivariate_normal(log_mean, log_cov, size=n_samples)

        # Transform to rate space
        return np.exp(log_samples)

    def optimize_hyperparameters(self, positions, rates, min_rate=0.1,
                                  length_scale_bounds=(0.05, 0.5),
                                  amplitude_bounds=(0.5, 5.0),
                                  noise_bounds=(0.1, 2.0)):
        """
        Optimize hyperparameters by maximizing marginal likelihood.

        Parameters
        ----------
        positions : (n, 2) array of bin centers
        rates : (n,) array of firing rates
        min_rate : minimum rate for log transform
        length_scale_bounds, amplitude_bounds, noise_bounds : parameter bounds

        Returns
        -------
        self : fitted model with optimized hyperparameters
        """
        y = np.log(np.maximum(rates, min_rate))
        n = len(y)

        def neg_log_marginal_likelihood(params):
            ls, amp, noise = params

            K = rbf_kernel_with_noise(positions, positions, ls, amp, noise**2)

            try:
                L, lower = cho_factor(K, lower=True)
                alpha = cho_solve((L, lower), y - self.mean)

                # Log marginal likelihood
                log_ml = -0.5 * (y - self.mean) @ alpha
                log_ml -= np.sum(np.log(np.diag(L)))
                log_ml -= 0.5 * n * np.log(2 * np.pi)

                return -log_ml
            except np.linalg.LinAlgError:
                return np.inf

        # Optimize
        x0 = [self.length_scale, self.amplitude, self.noise]
        bounds = [length_scale_bounds, amplitude_bounds, noise_bounds]

        result = minimize(neg_log_marginal_likelihood, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.length_scale, self.amplitude, self.noise = result.x
            print(f"Optimized hyperparameters: length_scale={self.length_scale:.3f}, "
                  f"amplitude={self.amplitude:.3f}, noise={self.noise:.3f}")
        else:
            print("Hyperparameter optimization did not converge, using initial values.")

        # Fit with optimized parameters
        return self.fit(positions, rates, min_rate)


def fit_gp_place_field(spike_map, occupancy, x_edges, y_edges,
                       optimize=True, min_occupancy=0.1, **gp_kwargs):
    """
    Fit a GP place field model to binned spike data.

    Parameters
    ----------
    spike_map : 2D array of spike counts
    occupancy : 2D array of time per bin (seconds)
    x_edges, y_edges : bin edges
    optimize : whether to optimize hyperparameters
    min_occupancy : minimum occupancy to include a bin
    **gp_kwargs : passed to GaussianProcessPlaceField

    Returns
    -------
    gp : fitted GaussianProcessPlaceField
    grid_positions : (n_bins, 2) array of all bin centers
    valid_mask : boolean mask for valid bins
    """
    # Compute bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create grid of all positions
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute rates
    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    # Get valid positions and rates for fitting
    valid_flat = valid.ravel()
    train_positions = grid_positions[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    # Create and fit GP
    gp = GaussianProcessPlaceField(**gp_kwargs)

    if optimize:
        gp.optimize_hyperparameters(train_positions, train_rates)
    else:
        gp.fit(train_positions, train_rates)

    return gp, grid_positions, valid.ravel()


def predict_rate_map(gp, grid_positions, shape, return_std=False):
    """
    Predict a full rate map from a fitted GP.

    Parameters
    ----------
    gp : fitted GaussianProcessPlaceField
    grid_positions : (n, 2) array of bin centers
    shape : tuple, shape of output rate map
    return_std : whether to return uncertainty map

    Returns
    -------
    rate_map : 2D array of predicted rates
    std_map : 2D array of uncertainties (if return_std=True)
    """
    if return_std:
        mean, std = gp.predict(grid_positions, return_std=True)
        return mean.reshape(shape), std.reshape(shape)
    else:
        mean = gp.predict(grid_positions)
        return mean.reshape(shape)


def sample_rate_maps(gp, grid_positions, shape, n_samples=4):
    """
    Sample rate maps from the GP posterior.

    Parameters
    ----------
    gp : fitted GaussianProcessPlaceField
    grid_positions : (n, 2) array of bin centers
    shape : tuple, shape of output rate maps
    n_samples : number of samples

    Returns
    -------
    samples : list of 2D rate map arrays
    """
    raw_samples = gp.sample(grid_positions, n_samples)
    return [s.reshape(shape) for s in raw_samples]


# --- Visualization ---

def plot_gp_results(observed_rate_map, gp, grid_positions, shape, x_edges, y_edges,
                    n_samples=4, seed=None):
    """
    Plot GP place field results: observed, posterior mean, uncertainty, and samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get predictions
    mean_map, std_map = predict_rate_map(gp, grid_positions, shape, return_std=True)
    sample_maps = sample_rate_maps(gp, grid_positions, shape, n_samples)

    # Plot
    n_cols = n_samples + 3  # observed + mean + std + samples
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

    # Bottom row: marginal slices through the field center
    # Find the peak location
    peak_idx = np.unravel_index(np.nanargmax(mean_map), mean_map.shape)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # X slice at peak y
    ax = axes[1, 0]
    ax.plot(x_centers, observed_rate_map[:, peak_idx[1]], 'k.', alpha=0.5, label='Observed')
    ax.plot(x_centers, mean_map[:, peak_idx[1]], 'b-', lw=2, label='GP mean')
    ax.fill_between(x_centers,
                    mean_map[:, peak_idx[1]] - 2*std_map[:, peak_idx[1]],
                    mean_map[:, peak_idx[1]] + 2*std_map[:, peak_idx[1]],
                    alpha=0.3, label='95% CI')
    for i, sample_map in enumerate(sample_maps):
        ax.plot(x_centers, sample_map[:, peak_idx[1]], '-', alpha=0.5, lw=1)
    ax.set_xlabel('x position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'X slice at y={y_centers[peak_idx[1]]:.2f}')
    ax.legend(fontsize=8)
    ax.set_xlim(x_edges[0], x_edges[-1])

    # Y slice at peak x
    ax = axes[1, 1]
    ax.plot(y_centers, observed_rate_map[peak_idx[0], :], 'k.', alpha=0.5, label='Observed')
    ax.plot(y_centers, mean_map[peak_idx[0], :], 'b-', lw=2, label='GP mean')
    ax.fill_between(y_centers,
                    mean_map[peak_idx[0], :] - 2*std_map[peak_idx[0], :],
                    mean_map[peak_idx[0], :] + 2*std_map[peak_idx[0], :],
                    alpha=0.3, label='95% CI')
    for i, sample_map in enumerate(sample_maps):
        ax.plot(y_centers, sample_map[peak_idx[0], :], '-', alpha=0.5, lw=1)
    ax.set_xlabel('y position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Y slice at x={x_centers[peak_idx[0]]:.2f}')
    ax.set_xlim(y_edges[0], y_edges[-1])

    # Hyperparameters text
    ax = axes[1, 2]
    ax.axis('off')
    hp_text = (f"GP Hyperparameters:\n\n"
               f"Length scale: {gp.length_scale:.3f}\n"
               f"Amplitude: {gp.amplitude:.3f}\n"
               f"Noise: {gp.noise:.3f}")
    ax.text(0.1, 0.5, hp_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes)

    # Hide remaining axes
    for i in range(3, n_cols):
        axes[1, i].set_visible(False)

    plt.suptitle('Gaussian Process place field model', y=1.02)
    plt.tight_layout()
    plt.savefig('place_field_gp_model.png', dpi=150)
    plt.show()

    return mean_map, std_map, sample_maps


# --- Helper functions (copied from model1_test.py to avoid import issues) ---

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
    from scipy.ndimage import gaussian_filter
    smooth_spikes = gaussian_filter(spike_map.astype(float), sigma=smooth_sigma)
    smooth_occ = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)
    rate_map = np.full_like(smooth_spikes, np.nan)
    valid = smooth_occ > min_occupancy
    rate_map[valid] = smooth_spikes[valid] / smooth_occ[valid]
    return rate_map


def simulate_foraging_trajectory(duration, fs_pos, bounds=(0, 1), speed=0.05):
    """Simulate uniform foraging using an Ornstein-Uhlenbeck process."""
    n_samples = int(duration * fs_pos)
    dt = 1.0 / fs_pos
    t = np.linspace(0, duration, n_samples)

    theta = 0.1
    sigma = speed * np.sqrt(2 * theta)
    center = (bounds[0] + bounds[1]) / 2

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    x[0] = center
    y[0] = center

    for i in range(1, n_samples):
        x[i] = x[i-1] + theta * (center - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        y[i] = y[i-1] + theta * (center - y[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        x[i] = np.clip(x[i], bounds[0], bounds[1])
        y[i] = np.clip(y[i], bounds[0], bounds[1])

    return np.column_stack([t, x, y])


# --- Demo with synthetic non-Gaussian place field ---

if __name__ == '__main__':
    np.random.seed(42)

    # Simulate foraging
    duration = 600  # seconds
    fs_pos = 100
    position = simulate_foraging_trajectory(duration, fs_pos, bounds=(0, 1), speed=0.5)
    t, x, y = position[:, 0], position[:, 1], position[:, 2]
    n_samples = len(t)

    # Create a NON-GAUSSIAN place field: two peaks (could be crescent, elongated, etc.)
    peak_rate = 25  # Hz

    # Two-peaked place field
    field1 = peak_rate * np.exp(-((x - 0.3)**2 + (y - 0.6)**2) / (2 * 0.1**2))
    field2 = peak_rate * 0.7 * np.exp(-((x - 0.7)**2 + (y - 0.4)**2) / (2 * 0.12**2))
    rate = field1 + field2 + 0.5  # baseline

    # Generate spikes
    spike_prob = rate / fs_pos
    spikes_bool = np.random.rand(n_samples) < spike_prob
    spike_times = t[spikes_bool]

    print(f"Generated {len(spike_times)} spikes\n")

    # Bin the data
    n_bins = 25
    dt = 1.0 / fs_pos
    x_edges = np.linspace(0, 1, n_bins + 1)
    y_edges = np.linspace(0, 1, n_bins + 1)

    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    observed_rate_map = estimate_rate_map(spk, occ, smooth_sigma=1.0)

    print("Fitting GP place field model...")

    # Fit GP model
    gp, grid_positions, valid_mask = fit_gp_place_field(
        spk, occ, x_edges, y_edges,
        optimize=True,
        length_scale=0.15,
        amplitude=2.0,
        noise=0.5
    )

    # Plot results
    shape = (n_bins, n_bins)
    mean_map, std_map, samples = plot_gp_results(
        observed_rate_map, gp, grid_positions, shape, x_edges, y_edges,
        n_samples=4, seed=123
    )

    print("\nGP model fitting complete!")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")
    print(f"Peak predicted rate: {np.max(mean_map):.1f} Hz")
