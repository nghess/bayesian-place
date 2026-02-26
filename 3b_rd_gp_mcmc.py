"""
Real data analysis with Bayesian GP place field model (MCMC).

Extends 3_real_data_analysis.py by using PyMC's NUTS sampler to obtain
posterior distributions over kernel hyperparameters, then marginalizing
over these when predicting firing rate maps.

Option 1: MCMC over hyperparameters only; GP latent function marginalized
analytically (fully Bayesian GP).

Requirements:
    pip install pymc arviz

Usage:
    python 4_real_data_mcmc.py
"""

import numpy as np
import pymc as pm
import arviz as az
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# Core estimation functions (unchanged from 3_real_data_analysis.py)
# =============================================================================

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


# =============================================================================
# Kernel functions
# =============================================================================

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


# =============================================================================
# Analytical GP posterior (for use with each hyperparameter sample)
# =============================================================================

def gp_posterior(X_train, y_train, X_pred, length_scale, amplitude, noise, mean=0.0):
    """
    Compute the analytical GP posterior for a single set of hyperparameters.

    Returns
    -------
    pred_mean : (m,) posterior mean at prediction positions
    pred_var : (m,) posterior marginal variance at prediction positions
    """
    K = rbf_kernel_with_noise(X_train, X_train, length_scale, amplitude, noise**2)
    K_star = rbf_kernel(X_pred, X_train, length_scale, amplitude)
    K_ss_diag = amplitude**2 * np.ones(X_pred.shape[0])

    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y_train - mean)
        v = cho_solve((L, lower), K_star.T)

        pred_mean = mean + K_star @ alpha
        pred_var = K_ss_diag - np.sum(K_star.T * v, axis=0)
        pred_var = np.maximum(pred_var, 1e-10)

    except np.linalg.LinAlgError:
        pred_mean = np.full(X_pred.shape[0], mean)
        pred_var = np.full(X_pred.shape[0], amplitude**2)

    return pred_mean, pred_var


# =============================================================================
# Original GP class (kept for MAP comparison)
# =============================================================================

class GaussianProcessPlaceField:
    """
    Standard GP model with MAP hyperparameter estimation.
    Identical to 3_real_data_analysis.py — included for side-by-side comparison.
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
        K_star = rbf_kernel(positions, self.X_train, self.length_scale, self.amplitude)
        log_mean = self.mean + K_star @ self.alpha
        if return_std:
            K_ss = rbf_kernel_with_noise(
                positions, positions, self.length_scale, self.amplitude, 1e-6
            )
            v = cho_solve((self.L, self.lower), K_star.T)
            log_var = np.diag(K_ss) - np.sum(K_star.T * v, axis=0)
            log_std = np.sqrt(np.maximum(log_var, 1e-10))
            mean_rate = np.exp(log_mean)
            std_rate = mean_rate * log_std
            return mean_rate, std_rate
        return np.exp(log_mean)

    def optimize_hyperparameters(self, positions, rates, min_rate=0.1,
                                  length_scale_bounds=(0.05, 0.5),
                                  amplitude_bounds=(0.5, 5.0),
                                  noise_bounds=(0.1, 2.0)):
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
            print(f"  MAP hyperparameters: length_scale={self.length_scale:.3f}, "
                  f"amplitude={self.amplitude:.3f}, noise={self.noise:.3f}")
        else:
            print("  Hyperparameter optimization did not converge, using initial values.")

        return self.fit(positions, rates, min_rate)


# =============================================================================
# MCMC GP class
# =============================================================================

class GaussianProcessPlaceFieldMCMC:
    """
    Fully Bayesian Gaussian Process model for place field estimation.

    Uses PyMC's NUTS sampler to draw posterior samples of the GP hyperparameters
    (length_scale, amplitude, noise, mean), then computes the analytical GP
    posterior for each sample to marginalize over hyperparameter uncertainty.
    """

    def __init__(self,
                 length_scale_alpha=5.0,
                 length_scale_beta=0.5,
                 amplitude_sigma=2.0,
                 noise_sigma=1.0,
                 mean_sigma=2.0):
        """
        Parameters
        ----------
        length_scale_alpha, length_scale_beta : InverseGamma prior params for length_scale.
            Default alpha=5, beta=0.5 → mode ~0.125, appropriate for [0,1]-normalized coords.
        amplitude_sigma : HalfNormal sigma for amplitude (signal std in log-rate space).
        noise_sigma : HalfNormal sigma for observation noise std.
        mean_sigma : Normal sigma for prior mean in log-rate space.
        """
        self.length_scale_alpha = length_scale_alpha
        self.length_scale_beta = length_scale_beta
        self.amplitude_sigma = amplitude_sigma
        self.noise_sigma = noise_sigma
        self.mean_sigma = mean_sigma

        self.trace = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.summary = None

    def fit(self, positions, rates, min_rate=0.1,
            n_samples=1000, n_tune=1000, n_chains=2,
            target_accept=0.9, random_seed=42, verbose=True):
        """
        Run MCMC to sample the posterior over GP hyperparameters.

        Parameters
        ----------
        positions : (n, 2) array of spatial bin centers (normalized to [0,1])
        rates : (n,) array of firing rates (Hz)
        min_rate : floor to avoid log(0)
        n_samples : posterior draws per chain
        n_tune : warmup/tuning draws (discarded)
        n_chains : independent chains
        target_accept : NUTS target acceptance (increase if divergences occur)
        random_seed : for reproducibility
        verbose : print progress and summary

        Returns
        -------
        self
        """
        self.X_train = np.asarray(positions, dtype=np.float64)
        self.y_train = np.log(np.maximum(rates, min_rate)).astype(np.float64)
        n = len(self.y_train)

        if verbose:
            print(f"  Fitting Bayesian GP with MCMC ({n_chains} chains × "
                  f"{n_samples} draws, {n_tune} tuning)...")
            print(f"  Training points: {n}")

        with pm.Model() as model:
            # --- Priors ---
            length_scale = pm.InverseGamma("length_scale",
                                           alpha=self.length_scale_alpha,
                                           beta=self.length_scale_beta)
            amplitude = pm.HalfNormal("amplitude", sigma=self.amplitude_sigma)
            noise = pm.HalfNormal("noise", sigma=self.noise_sigma)
            mean = pm.Normal("mean", mu=0, sigma=self.mean_sigma)

            # --- GP with analytical marginal likelihood ---
            cov_func = amplitude**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=length_scale)
            mean_func = pm.gp.mean.Constant(mean)
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

            gp.marginal_likelihood("y_obs", X=self.X_train, y=self.y_train,
                                   sigma=noise)

            # --- Sample ---
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                cores=min(n_chains, 4),
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=verbose,
            )

        self.model = model
        self.trace = trace

        self.summary = az.summary(trace, var_names=["length_scale", "amplitude",
                                                     "noise", "mean"])
        if verbose:
            print("\n  --- Posterior summary ---")
            print(self.summary.to_string())
            print()

            rhat_max = self.summary["r_hat"].max()
            ess_min = self.summary["ess_bulk"].min()
            print(f"  Max R-hat: {rhat_max:.3f} (want < 1.01)")
            print(f"  Min ESS (bulk): {ess_min:.0f} (want > 400)")

            n_div = self.trace.sample_stats["diverging"].sum().values
            if n_div > 0:
                print(f"  ⚠ {n_div} divergent transitions! Consider increasing "
                      f"target_accept or adjusting priors.")
            else:
                print("  No divergent transitions.")

        return self

    def get_hyperparameter_samples(self, n=None):
        """
        Extract posterior hyperparameter samples (flattened across chains).

        Parameters
        ----------
        n : int or None — if given, randomly subsample n draws.

        Returns
        -------
        dict with keys 'length_scale', 'amplitude', 'noise', 'mean'
        """
        posterior = self.trace.posterior
        ls = posterior["length_scale"].values.flatten()
        amp = posterior["amplitude"].values.flatten()
        nse = posterior["noise"].values.flatten()
        mn = posterior["mean"].values.flatten()

        if n is not None and n < len(ls):
            idx = np.random.choice(len(ls), size=n, replace=False)
            ls, amp, nse, mn = ls[idx], amp[idx], nse[idx], mn[idx]

        return {"length_scale": ls, "amplitude": amp, "noise": nse, "mean": mn}

    def predict(self, positions, n_posterior_samples=200, return_samples=False):
        """
        Predict rates, marginalizing over hyperparameter uncertainty.
        Draws one GP conditional sample per hyperparameter draw (full uncertainty).

        Returns
        -------
        mean_rate : (m,) marginalized mean rate
        std_rate : (m,) marginalized std
        rate_samples : (n_posterior_samples, m) [only if return_samples=True]
        """
        positions = np.asarray(positions, dtype=np.float64)
        m = positions.shape[0]
        hp = self.get_hyperparameter_samples(n=n_posterior_samples)

        log_rate_samples = np.zeros((n_posterior_samples, m))
        for i in range(n_posterior_samples):
            pred_mean, pred_var = gp_posterior(
                self.X_train, self.y_train, positions,
                hp["length_scale"][i], hp["amplitude"][i],
                hp["noise"][i], hp["mean"][i]
            )
            log_rate_samples[i] = pred_mean + np.sqrt(pred_var) * np.random.randn(m)

        rate_samples = np.exp(log_rate_samples)
        mean_rate = np.mean(rate_samples, axis=0)
        std_rate = np.std(rate_samples, axis=0)

        if return_samples:
            return mean_rate, std_rate, rate_samples
        return mean_rate, std_rate

    def predict_mean_only(self, positions, n_posterior_samples=200):
        """
        Predict using GP posterior means only (no conditional sampling).
        Gives smoother maps — uncertainty reflects hyperparameter ambiguity only.

        Returns
        -------
        mean_rate : (m,) mean predicted rate
        std_rate : (m,) std across hyperparameter draws
        """
        positions = np.asarray(positions, dtype=np.float64)
        m = positions.shape[0]
        hp = self.get_hyperparameter_samples(n=n_posterior_samples)

        log_rate_means = np.zeros((n_posterior_samples, m))
        for i in range(n_posterior_samples):
            pred_mean, _ = gp_posterior(
                self.X_train, self.y_train, positions,
                hp["length_scale"][i], hp["amplitude"][i],
                hp["noise"][i], hp["mean"][i]
            )
            log_rate_means[i] = pred_mean

        rate_means = np.exp(log_rate_means)
        return np.mean(rate_means, axis=0), np.std(rate_means, axis=0)


# =============================================================================
# Fitting wrappers
# =============================================================================

def fit_gp_place_field(spike_map, occupancy, x_edges, y_edges,
                       optimize=True, min_occupancy=0.1, **gp_kwargs):
    """
    Fit the MAP GP model. Normalizes positions to [0,1].

    Returns
    -------
    gp, grid_positions_norm, grid_positions_raw, valid_mask, scale
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions_raw = np.column_stack([xx.ravel(), yy.ravel()])

    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]
    scale = max(x_max - x_min, y_max - y_min)

    grid_positions_norm = np.column_stack([
        (grid_positions_raw[:, 0] - x_min) / scale,
        (grid_positions_raw[:, 1] - y_min) / scale
    ])

    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    valid_flat = valid.ravel()
    train_positions = grid_positions_norm[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    print(f"  Training on {len(train_positions)} valid bins")
    print(f"  Rate range: {train_rates.min():.2f} – {train_rates.max():.2f} Hz")

    gp = GaussianProcessPlaceField(**gp_kwargs)
    if optimize:
        gp.optimize_hyperparameters(train_positions, train_rates)
    else:
        gp.fit(train_positions, train_rates)

    return gp, grid_positions_norm, grid_positions_raw, valid_flat, scale


def fit_gp_place_field_mcmc(spike_map, occupancy, x_edges, y_edges,
                             min_occupancy=0.1, **mcmc_kwargs):
    """
    Fit the Bayesian GP model via MCMC. Normalizes positions to [0,1]
    using the same convention as fit_gp_place_field.

    Parameters
    ----------
    spike_map : 2D array of spike counts per bin
    occupancy : 2D array of time spent per bin (seconds)
    x_edges, y_edges : bin edges (in original coordinates, e.g. pixels or cm)
    min_occupancy : minimum occupancy to include a bin
    **mcmc_kwargs : passed to GaussianProcessPlaceFieldMCMC.fit()
        Useful keys: n_samples, n_tune, n_chains, target_accept

    Returns
    -------
    gp : fitted GaussianProcessPlaceFieldMCMC
    grid_positions_norm : (n_total_bins, 2) normalized bin centers
    grid_positions_raw : (n_total_bins, 2) original-coordinate bin centers
    valid_mask : boolean mask for bins with sufficient occupancy
    scale : normalization scale factor (for interpreting length_scale)
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions_raw = np.column_stack([xx.ravel(), yy.ravel()])

    # Normalize using the same convention as the MAP version
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]
    scale = max(x_max - x_min, y_max - y_min)

    grid_positions_norm = np.column_stack([
        (grid_positions_raw[:, 0] - x_min) / scale,
        (grid_positions_raw[:, 1] - y_min) / scale
    ])

    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    valid_flat = valid.ravel()
    train_positions = grid_positions_norm[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    print(f"  Training on {len(train_positions)} valid bins")
    print(f"  Rate range: {train_rates.min():.2f} – {train_rates.max():.2f} Hz")
    print(f"  Normalization scale: {scale:.1f} (length_scale × {scale:.1f} = real units)")

    gp = GaussianProcessPlaceFieldMCMC()
    gp.fit(train_positions, train_rates, **mcmc_kwargs)

    return gp, grid_positions_norm, grid_positions_raw, valid_flat, scale


# =============================================================================
# Prediction helpers
# =============================================================================

def predict_rate_map(gp, grid_positions, shape, return_std=False):
    """Predict rate map from a MAP GP."""
    if return_std:
        mean, std = gp.predict(grid_positions, return_std=True)
        return mean.reshape(shape), std.reshape(shape)
    return gp.predict(grid_positions).reshape(shape)


def predict_rate_map_mcmc(gp, grid_positions, shape,
                           n_posterior_samples=200, method="mean_only"):
    """Predict rate map from the MCMC GP, marginalizing over hyperparameters."""
    if method == "mean_only":
        mean, std = gp.predict_mean_only(grid_positions, n_posterior_samples)
    else:
        mean, std = gp.predict(grid_positions, n_posterior_samples)
    return mean.reshape(shape), std.reshape(shape)


# =============================================================================
# Visualization
# =============================================================================

def plot_mcmc_diagnostics(gp, figsize=(14, 8)):
    """Plot posterior distributions and trace plots for MCMC diagnostics."""
    var_names = ["length_scale", "amplitude", "noise", "mean"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    posterior = gp.trace.posterior

    for i, var in enumerate(var_names):
        samples = posterior[var].values  # (chains, draws)
        n_chains = samples.shape[0]

        # Posterior density
        ax_hist = fig.add_subplot(gs[0, i])
        for chain in range(n_chains):
            ax_hist.hist(samples[chain], bins=40, alpha=0.5,
                         density=True, label=f"Chain {chain+1}")
        ax_hist.set_title(var, fontsize=11, fontweight='bold')
        ax_hist.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax_hist.legend(fontsize=8)

        all_samples = samples.flatten()
        median = np.median(all_samples)
        hdi = az.hdi(gp.trace, var_names=[var], hdi_prob=0.94)[var].values
        ax_hist.axvline(median, color='k', ls='--', lw=1.5)
        ax_hist.axvspan(hdi[0], hdi[1], alpha=0.15, color='steelblue')
        ax_hist.set_xlabel(f"Median: {median:.3f}\n94% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]",
                           fontsize=8)

        # Trace plot
        ax_trace = fig.add_subplot(gs[1, i])
        for chain in range(n_chains):
            ax_trace.plot(samples[chain], alpha=0.6, lw=0.5)
        ax_trace.set_title(f"Trace: {var}", fontsize=10)
        ax_trace.set_xlabel("Draw")
        ax_trace.set_ylabel("Value" if i == 0 else "")

    fig.suptitle("MCMC Diagnostics: Posterior Distributions and Trace Plots",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("mcmc_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_gp_mcmc_results(observed_rate_map, gp_mcmc, gp_map, grid_positions,
                          shape, x_edges, y_edges, scale,
                          n_posterior_samples=200, n_gp_samples=4, seed=None):
    """
    Plot comprehensive results comparing MAP and MCMC GP models.

    Row 0: Observed | MAP mean | MCMC mean | Uncertainty (hp) |
           Uncertainty (full) | GP samples
    Row 1: Marginal slices through peak with both MAP and MCMC CIs
    Row 2: Hyperparameter summary + posterior scatter

    Parameters
    ----------
    observed_rate_map : 2D array of observed rates
    gp_mcmc : fitted GaussianProcessPlaceFieldMCMC
    gp_map : fitted GaussianProcessPlaceField (MAP), or None to skip comparison
    grid_positions : (n, 2) normalized bin centers
    shape : (nx, ny) map shape
    x_edges, y_edges : bin edges (original coordinates)
    scale : normalization scale factor
    n_posterior_samples : draws to marginalize over
    n_gp_samples : number of posterior sample maps
    seed : random seed
    """
    if seed is not None:
        np.random.seed(seed)

    # --- MCMC predictions ---
    mean_map_mcmc, std_map_hp = predict_rate_map_mcmc(
        gp_mcmc, grid_positions, shape, n_posterior_samples, method="mean_only"
    )
    _, std_map_full = predict_rate_map_mcmc(
        gp_mcmc, grid_positions, shape, n_posterior_samples, method="full"
    )

    # --- MAP predictions (if available) ---
    if gp_map is not None:
        mean_map_map, std_map_map = predict_rate_map(
            gp_map, grid_positions, shape, return_std=True
        )
    else:
        mean_map_map = None
        std_map_map = None

    # --- GP posterior samples (each with different hyperparameter draw) ---
    hp = gp_mcmc.get_hyperparameter_samples(n=n_gp_samples)
    sample_maps = []
    for i in range(n_gp_samples):
        pred_mean, pred_var = gp_posterior(
            gp_mcmc.X_train, gp_mcmc.y_train, grid_positions,
            hp["length_scale"][i], hp["amplitude"][i],
            hp["noise"][i], hp["mean"][i]
        )
        log_sample = pred_mean + np.sqrt(pred_var) * np.random.randn(len(pred_mean))
        sample_maps.append(np.exp(log_sample).reshape(shape))

    # --- Layout ---
    has_map = gp_map is not None
    n_cols = (5 if has_map else 4) + n_gp_samples
    fig = plt.figure(figsize=(3.2 * n_cols, 10.5))
    gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1.2, 1, 0.7],
                  hspace=0.35, wspace=0.35)

    vmax = np.nanpercentile(observed_rate_map.ravel(), 98)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # ===== Row 0: Rate maps =====
    col = 0

    ax = fig.add_subplot(gs[0, col]); col += 1
    ax.imshow(observed_rate_map.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('Observed', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    if has_map:
        ax = fig.add_subplot(gs[0, col]); col += 1
        ax.imshow(mean_map_map.T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title('MAP GP mean', fontsize=10, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, col]); col += 1
    ax.imshow(mean_map_mcmc.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('MCMC GP mean', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, col]); col += 1
    im1 = ax.imshow(std_map_hp.T, origin='lower', aspect='auto',
                     extent=extent, cmap='viridis')
    ax.set_title('Uncertainty\n(hyperparams only)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im1, ax=ax, fraction=0.046, label='Hz')

    ax = fig.add_subplot(gs[0, col]); col += 1
    im2 = ax.imshow(std_map_full.T, origin='lower', aspect='auto',
                     extent=extent, cmap='viridis')
    ax.set_title('Uncertainty\n(full: hp + GP)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im2, ax=ax, fraction=0.046, label='Hz')

    for i, smap in enumerate(sample_maps):
        ax = fig.add_subplot(gs[0, col]); col += 1
        ax.imshow(smap.T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    # ===== Row 1: Marginal slices =====
    peak_idx = np.unravel_index(np.nanargmax(mean_map_mcmc), mean_map_mcmc.shape)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # X-slice at peak y
    ax = fig.add_subplot(gs[1, :n_cols // 2])
    ax.plot(x_centers, observed_rate_map[:, peak_idx[1]], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(x_centers, mean_map_mcmc[:, peak_idx[1]], 'b-', lw=2,
            label='MCMC mean')

    if has_map:
        ax.plot(x_centers, mean_map_map[:, peak_idx[1]], 'r--', lw=1.5,
                label='MAP mean', alpha=0.8)

    # Full CI
    low = mean_map_mcmc[:, peak_idx[1]] - 2 * std_map_full[:, peak_idx[1]]
    high = mean_map_mcmc[:, peak_idx[1]] + 2 * std_map_full[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low, 0), high,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')

    # HP-only CI
    low_hp = mean_map_mcmc[:, peak_idx[1]] - 2 * std_map_hp[:, peak_idx[1]]
    high_hp = mean_map_mcmc[:, peak_idx[1]] + 2 * std_map_hp[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        ax.plot(x_centers, smap[:, peak_idx[1]], '-', alpha=0.4, lw=1,
                color='gray', label='GP samples' if i == 0 else None)

    ax.set_xlabel('x position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'X-slice at y = {y_centers[peak_idx[1]]:.1f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(x_edges[0], x_edges[-1])

    # Y-slice at peak x
    ax = fig.add_subplot(gs[1, n_cols // 2:])
    ax.plot(y_centers, observed_rate_map[peak_idx[0], :], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(y_centers, mean_map_mcmc[peak_idx[0], :], 'b-', lw=2,
            label='MCMC mean')

    if has_map:
        ax.plot(y_centers, mean_map_map[peak_idx[0], :], 'r--', lw=1.5,
                label='MAP mean', alpha=0.8)

    low = mean_map_mcmc[peak_idx[0], :] - 2 * std_map_full[peak_idx[0], :]
    high = mean_map_mcmc[peak_idx[0], :] + 2 * std_map_full[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low, 0), high,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')

    low_hp = mean_map_mcmc[peak_idx[0], :] - 2 * std_map_hp[peak_idx[0], :]
    high_hp = mean_map_mcmc[peak_idx[0], :] + 2 * std_map_hp[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        ax.plot(y_centers, smap[peak_idx[0], :], '-', alpha=0.4, lw=1,
                color='gray', label='GP samples' if i == 0 else None)

    ax.set_xlabel('y position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Y-slice at x = {x_centers[peak_idx[0]]:.1f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(y_edges[0], y_edges[-1])

    # ===== Row 2: Hyperparameter summary + pair plot =====
    ax = fig.add_subplot(gs[2, :n_cols // 2])
    ax.axis('off')

    summary = gp_mcmc.summary
    hp_text = "MCMC Hyperparameter posteriors (mean ± sd  [94% HDI]):\n\n"
    for var_name in ["length_scale", "amplitude", "noise", "mean"]:
        row = summary.loc[var_name]
        hp_text += (f"  {var_name:14s}: {row['mean']:.4f} ± {row['sd']:.4f}   "
                    f"[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")
        # Show length_scale in original units
        if var_name == "length_scale":
            hp_text += f"  (= {row['mean'] * scale:.1f} original units)"
        hp_text += "\n"

    if has_map:
        hp_text += (f"\n  MAP comparison: ls={gp_map.length_scale:.4f}, "
                    f"amp={gp_map.amplitude:.4f}, noise={gp_map.noise:.4f}")

    hp_text += f"\n\n  R-hat (max): {summary['r_hat'].max():.4f}"
    hp_text += f"  |  ESS bulk (min): {summary['ess_bulk'].min():.0f}"
    n_div = gp_mcmc.trace.sample_stats["diverging"].sum().values
    hp_text += f"  |  Divergences: {n_div}"

    ax.text(0.05, 0.5, hp_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Pair plot
    ax_pair = fig.add_subplot(gs[2, n_cols // 2:])
    hp_samples = gp_mcmc.get_hyperparameter_samples(n=500)
    sc = ax_pair.scatter(hp_samples["length_scale"], hp_samples["amplitude"],
                         c=hp_samples["noise"], cmap='coolwarm', s=8, alpha=0.5)
    ax_pair.set_xlabel("length_scale (normalized)", fontsize=9)
    ax_pair.set_ylabel("amplitude", fontsize=9)
    ax_pair.set_title("Hyperparameter posterior samples", fontsize=10)
    plt.colorbar(sc, ax=ax_pair, label='noise', fraction=0.046)

    # Mark MAP estimate if available
    if has_map:
        ax_pair.plot(gp_map.length_scale, gp_map.amplitude, 'r*', ms=15,
                     markeredgecolor='k', markeredgewidth=0.5, label='MAP', zorder=10)
        ax_pair.legend(fontsize=8)

    fig.suptitle('Bayesian GP Place Field Model — MCMC vs MAP', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('real_data_gp_mcmc.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mean_map_mcmc, std_map_hp, std_map_full, sample_maps


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    from SessionData import SessionData

    # ---- Data configuration ----
    base_path = "d:/data/"
    mouse_id = "7012"
    session_id = "m10"
    experiment = "clickbait-motivate"
    cluster_id = 19

    print(f"Loading data: {mouse_id}/{session_id}, cluster {cluster_id}")
    print("=" * 60)

    # ---- Load data ----
    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True
    )

    # Extract position and spike times (ms -> seconds)
    position = data.events[['timestamp_ms', 'nose_x', 'nose_y']].to_numpy()
    position[:, 0] = position[:, 0] / 1000.0

    spike_times = data.clusters[cluster_id]['spike_times']
    spike_times = spike_times / 1000.0

    print("=" * 60)

    # ---- Analysis parameters ----
    n_bins = 25
    smooth_sigma = 1.0
    dt = np.median(np.diff(position[:, 0]))

    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    # ---- Compute maps ----
    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    observed_rate_map = estimate_rate_map(spk, occ, smooth_sigma)
    shape = (n_bins, n_bins)

    print(f"\nSession duration: {position[-1, 0] - position[0, 0]:.1f} s")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")

    # ---- Fit MAP GP (for comparison) ----
    print("\n--- Fitting MAP GP ---")
    gp_map, grid_pos_norm, grid_pos_raw, valid_mask, scale = fit_gp_place_field(
        spk, occ, x_edges, y_edges,
        optimize=True,
        length_scale=0.15,
        amplitude=2.0,
        noise=0.5
    )

    # ---- Fit MCMC GP ----
    print("\n--- Fitting MCMC GP ---")
    gp_mcmc, grid_pos_norm_mcmc, grid_pos_raw_mcmc, valid_mask_mcmc, scale_mcmc = \
        fit_gp_place_field_mcmc(
            spk, occ, x_edges, y_edges,
            n_samples=1000,
            n_tune=1000,
            n_chains=2,
            target_accept=0.9,
        )

    # ---- Plot diagnostics ----
    print("\nPlotting MCMC diagnostics...")
    plot_mcmc_diagnostics(gp_mcmc)

    # ---- Plot comparison results ----
    print("Plotting results...")
    mean_map, std_hp, std_full, samples = plot_gp_mcmc_results(
        observed_rate_map, gp_mcmc, gp_map, grid_pos_norm, shape,
        x_edges, y_edges, scale,
        n_posterior_samples=200, n_gp_samples=4, seed=123
    )

    print("\nDone!")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")
    print(f"Peak MCMC mean rate: {np.max(mean_map):.1f} Hz")