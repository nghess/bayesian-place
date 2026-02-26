"""
Bayesian GP place field model with MCMC and temporal block cross-validation.

Fully Bayesian GP: MCMC over hyperparameters, analytical GP conditional.
Cross-validation uses contiguous time blocks to preserve temporal structure.

Requirements:
    pip install pymc arviz

Usage:
    python 5_real_data_mcmc_cv.py
"""

import numpy as np
import pymc as pm
import arviz as az
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings


# =============================================================================
# Core estimation functions
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
# Analytical GP posterior
# =============================================================================

def gp_posterior(X_train, y_train, X_pred, length_scale, amplitude, noise, mean=0.0):
    """
    Compute the analytical GP posterior for a single set of hyperparameters.

    Returns
    -------
    pred_mean : (m,) posterior mean
    pred_var : (m,) posterior marginal variance
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
# MCMC GP class
# =============================================================================

class GaussianProcessPlaceFieldMCMC:
    """
    Fully Bayesian GP for place field estimation via MCMC.

    Samples the posterior over (length_scale, amplitude, noise, mean)
    using PyMC's NUTS, then marginalizes analytically over the GP conditional.
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
        length_scale_alpha, length_scale_beta : InverseGamma prior on length_scale.
            Mode ~ beta / (alpha + 1) ≈ 0.083 for defaults. Appropriate for
            [0,1]-normalized coordinates.
        amplitude_sigma : HalfNormal sigma for amplitude (log-rate signal std).
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
        """
        self.X_train = np.asarray(positions, dtype=np.float64)
        self.y_train = np.log(np.maximum(rates, min_rate)).astype(np.float64)
        n = len(self.y_train)

        if verbose:
            print(f"  MCMC: {n_chains} chains × {n_samples} draws "
                  f"({n_tune} tuning), {n} training points")

        with pm.Model() as model:
            length_scale = pm.InverseGamma("length_scale",
                                           alpha=self.length_scale_alpha,
                                           beta=self.length_scale_beta)
            amplitude = pm.HalfNormal("amplitude", sigma=self.amplitude_sigma)
            noise = pm.HalfNormal("noise", sigma=self.noise_sigma)
            mean = pm.Normal("mean", mu=0, sigma=self.mean_sigma)

            cov_func = amplitude**2 * pm.gp.cov.ExpQuad(input_dim=2, ls=length_scale)
            mean_func = pm.gp.mean.Constant(mean)
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)

            gp.marginal_likelihood("y_obs", X=self.X_train, y=self.y_train,
                                   sigma=noise)

            trace = pm.sample(
                draws=n_samples, tune=n_tune,
                chains=n_chains, cores=min(n_chains, 4),
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=verbose,
            )

        self.model = model
        self.trace = trace
        self.summary = az.summary(trace, var_names=["length_scale", "amplitude",
                                                     "noise", "mean"])

        if verbose:
            print(f"\n  Posterior summary:")
            print(self.summary.to_string())
            rhat_max = self.summary["r_hat"].max()
            ess_min = self.summary["ess_bulk"].min()
            n_div = self.trace.sample_stats["diverging"].sum().values
            print(f"  R-hat max: {rhat_max:.3f} | ESS min: {ess_min:.0f} | "
                  f"Divergences: {n_div}")

        return self

    def get_hyperparameter_samples(self, n=None):
        """Extract posterior samples (flattened across chains)."""
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
        Predict rates with full uncertainty (hyperparams + GP conditional).
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
        Smoother maps; uncertainty = hyperparameter ambiguity only.
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

    def predict_log_rate(self, positions, n_posterior_samples=200):
        """
        Predict log-rates (useful for Poisson log-likelihood scoring).

        Returns
        -------
        log_rate_mean : (m,) mean predicted log-rate
        log_rate_std : (m,) std of predicted log-rate
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

        return np.mean(log_rate_means, axis=0), np.std(log_rate_means, axis=0)


# =============================================================================
# Spatial binning and normalization
# =============================================================================

def make_grid(x_edges, y_edges):
    """
    Build the spatial grid and normalization parameters.

    Returns
    -------
    grid_positions_norm : (n_bins, 2) normalized to [0, ~1]
    grid_positions_raw : (n_bins, 2) original coordinates
    scale : normalization factor (max dimension span)
    x_min, y_min : offsets used for normalization
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

    return grid_positions_norm, grid_positions_raw, scale, x_min, y_min


def compute_train_data(spike_map, occupancy, grid_positions_norm, min_occupancy=0.1):
    """
    Extract valid training positions and rates from binned data.

    Returns
    -------
    train_positions : (n_valid, 2) normalized positions
    train_rates : (n_valid,) firing rates in Hz
    valid_mask : (n_total,) boolean mask
    """
    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    valid_flat = valid.ravel()
    train_positions = grid_positions_norm[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    return train_positions, train_rates, valid_flat


# =============================================================================
# Fitting wrapper
# =============================================================================

def fit_gp_mcmc(spike_map, occupancy, x_edges, y_edges,
                min_occupancy=0.1, **mcmc_kwargs):
    """
    Fit Bayesian GP to binned spike data.

    Returns
    -------
    gp : fitted GaussianProcessPlaceFieldMCMC
    grid_norm : (n_bins, 2) normalized bin centers
    grid_raw : (n_bins, 2) original bin centers
    valid_mask : boolean mask
    scale : normalization factor
    """
    grid_norm, grid_raw, scale, x_min, y_min = make_grid(x_edges, y_edges)
    train_pos, train_rates, valid_mask = compute_train_data(
        spike_map, occupancy, grid_norm, min_occupancy
    )

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    print(f"  Grid: {n_x} × {n_y} = {n_x * n_y} bins "
          f"({valid_mask.sum()} valid)")
    print(f"  Rate range: {train_rates.min():.2f} – {train_rates.max():.2f} Hz")
    print(f"  Scale: {scale:.1f} (length_scale × scale = original units)")

    gp = GaussianProcessPlaceFieldMCMC()
    gp.fit(train_pos, train_rates, **mcmc_kwargs)

    return gp, grid_norm, grid_raw, valid_mask, scale


# =============================================================================
# Prediction helpers
# =============================================================================

def predict_rate_map(gp, grid_positions, shape,
                     n_posterior_samples=200, method="mean_only"):
    """
    Predict rate map from MCMC GP.

    Parameters
    ----------
    method : "mean_only" (smooth, hp uncertainty) or "full" (hp + GP conditional)
    """
    if method == "mean_only":
        mean, std = gp.predict_mean_only(grid_positions, n_posterior_samples)
    else:
        mean, std = gp.predict(grid_positions, n_posterior_samples)
    return mean.reshape(shape), std.reshape(shape)


# =============================================================================
# Temporal block cross-validation
# =============================================================================

class TemporalBlockCV:
    """
    Blocked time-series cross-validation for GP place field models.

    Splits the session into K contiguous time blocks. For each fold,
    one block is held out and the model is trained on the remaining blocks.

    Evaluation metrics:
    - Poisson log-likelihood of held-out spike counts
    - Pearson correlation between predicted and observed rate maps
    - Mean squared error in rate space
    """

    def __init__(self, n_folds=5, min_occupancy=0.1):
        self.n_folds = n_folds
        self.min_occupancy = min_occupancy
        self.results = None

    def run(self, position, spike_times, x_edges, y_edges,
            n_samples=500, n_tune=500, n_chains=2,
            target_accept=0.9, n_pred_samples=100,
            verbose=True):
        """
        Run temporal block cross-validation.

        Parameters
        ----------
        position : (n, 3) array [time, x, y] in seconds
        spike_times : (n_spikes,) array in seconds
        x_edges, y_edges : spatial bin edges
        n_samples, n_tune, n_chains : MCMC parameters (reduced for CV speed)
        target_accept : NUTS target acceptance
        n_pred_samples : posterior draws for prediction per fold
        verbose : print per-fold progress

        Returns
        -------
        results : dict with per-fold and aggregate metrics
        """
        t_start = position[0, 0]
        t_end = position[-1, 0]
        dt = np.median(np.diff(position[:, 0]))

        # Define temporal block boundaries
        block_edges = np.linspace(t_start, t_end, self.n_folds + 1)

        grid_norm, grid_raw, scale, x_min, y_min = make_grid(x_edges, y_edges)
        n_x = len(x_edges) - 1
        n_y = len(y_edges) - 1
        shape = (n_x, n_y)

        fold_results = []

        for fold in range(self.n_folds):
            t_lo = block_edges[fold]
            t_hi = block_edges[fold + 1]

            if verbose:
                print(f"\n{'='*60}")
                print(f"Fold {fold+1}/{self.n_folds}: "
                      f"held-out block [{t_lo:.1f}s, {t_hi:.1f}s]")
                print(f"{'='*60}")

            # --- Split position data ---
            test_mask_pos = (position[:, 0] >= t_lo) & (position[:, 0] < t_hi)
            train_mask_pos = ~test_mask_pos

            pos_train = position[train_mask_pos]
            pos_test = position[test_mask_pos]

            # --- Split spike times ---
            test_mask_spk = (spike_times >= t_lo) & (spike_times < t_hi)
            train_mask_spk = ~test_mask_spk

            spk_train = spike_times[train_mask_spk]
            spk_test = spike_times[test_mask_spk]

            if verbose:
                print(f"  Train: {len(pos_train)} pos samples, "
                      f"{len(spk_train)} spikes")
                print(f"  Test:  {len(pos_test)} pos samples, "
                      f"{len(spk_test)} spikes")

            # --- Compute train maps ---
            occ_train = compute_occupancy(pos_train, x_edges, y_edges, dt)
            spk_map_train = compute_spike_map(spk_train, pos_train, x_edges, y_edges)

            train_positions, train_rates, valid_train = compute_train_data(
                spk_map_train, occ_train, grid_norm, self.min_occupancy
            )

            if len(train_positions) < 10:
                warnings.warn(f"Fold {fold+1}: only {len(train_positions)} valid "
                              f"training bins — skipping.")
                fold_results.append(None)
                continue

            # --- Compute test maps ---
            occ_test = compute_occupancy(pos_test, x_edges, y_edges, dt)
            spk_map_test = compute_spike_map(spk_test, pos_test, x_edges, y_edges)

            # --- Fit MCMC on training data ---
            gp = GaussianProcessPlaceFieldMCMC()
            gp.fit(train_positions, train_rates,
                   n_samples=n_samples, n_tune=n_tune, n_chains=n_chains,
                   target_accept=target_accept, verbose=verbose)

            # --- Predict on full grid ---
            pred_mean_map, pred_std_map = predict_rate_map(
                gp, grid_norm, shape, n_pred_samples, method="mean_only"
            )

            # --- Evaluate on held-out data ---
            metrics = self._evaluate_fold(
                pred_mean_map, gp, grid_norm, shape,
                spk_map_test, occ_test, n_pred_samples
            )

            metrics["fold"] = fold + 1
            metrics["t_lo"] = t_lo
            metrics["t_hi"] = t_hi
            metrics["n_train_spikes"] = len(spk_train)
            metrics["n_test_spikes"] = len(spk_test)
            metrics["pred_mean_map"] = pred_mean_map
            metrics["pred_std_map"] = pred_std_map
            metrics["test_rate_map"] = self._make_rate_map(spk_map_test, occ_test)
            metrics["gp"] = gp

            fold_results.append(metrics)

            if verbose:
                print(f"\n  Fold {fold+1} metrics:")
                print(f"    Poisson LL/spike: {metrics['poisson_ll_per_spike']:.3f}")
                print(f"    Correlation:      {metrics['correlation']:.3f}")
                print(f"    RMSE:             {metrics['rmse']:.2f} Hz")

        # --- Aggregate ---
        valid_folds = [r for r in fold_results if r is not None]
        agg = {}
        for key in ["poisson_ll_per_spike", "correlation", "rmse"]:
            vals = [r[key] for r in valid_folds if np.isfinite(r[key])]
            agg[f"{key}_mean"] = np.mean(vals) if vals else np.nan
            agg[f"{key}_std"] = np.std(vals) if vals else np.nan
            agg[f"{key}_per_fold"] = vals

        self.results = {
            "folds": fold_results,
            "aggregate": agg,
            "n_folds": self.n_folds,
            "block_edges": block_edges,
        }

        if verbose:
            print(f"\n{'='*60}")
            print("Cross-validation summary")
            print(f"{'='*60}")
            print(f"  Poisson LL/spike: {agg['poisson_ll_per_spike_mean']:.3f} "
                  f"± {agg['poisson_ll_per_spike_std']:.3f}")
            print(f"  Correlation:      {agg['correlation_mean']:.3f} "
                  f"± {agg['correlation_std']:.3f}")
            print(f"  RMSE:             {agg['rmse_mean']:.2f} "
                  f"± {agg['rmse_std']:.2f} Hz")

        return self.results

    def _make_rate_map(self, spike_map, occupancy):
        """Compute rate map without smoothing (for honest test evaluation)."""
        rate_map = np.full_like(spike_map, np.nan, dtype=float)
        valid = occupancy > self.min_occupancy
        rate_map[valid] = spike_map[valid] / occupancy[valid]
        return rate_map

    def _evaluate_fold(self, pred_mean_map, gp, grid_norm, shape,
                       spk_map_test, occ_test, n_pred_samples):
        """
        Compute held-out evaluation metrics.

        Metrics
        -------
        poisson_ll_per_spike : mean Poisson log-likelihood per spike.
            Higher is better. Evaluates how well the predicted rate map
            explains the held-out spike counts.
        correlation : Pearson r between predicted and observed rate maps
            (over bins with sufficient test occupancy).
        rmse : root-mean-squared error in Hz.
        """
        # Observed test rate map (unsmoothed)
        test_rate_map = self._make_rate_map(spk_map_test, occ_test)

        # Bins valid in both predicted and test
        valid_test = np.isfinite(test_rate_map.ravel())
        valid_pred = np.isfinite(pred_mean_map.ravel())
        valid = valid_test & valid_pred

        if valid.sum() < 5:
            return {"poisson_ll_per_spike": np.nan,
                    "correlation": np.nan, "rmse": np.nan}

        pred_flat = pred_mean_map.ravel()[valid]
        test_flat = test_rate_map.ravel()[valid]

        # --- Correlation ---
        if np.std(pred_flat) > 0 and np.std(test_flat) > 0:
            correlation = np.corrcoef(pred_flat, test_flat)[0, 1]
        else:
            correlation = np.nan

        # --- RMSE ---
        rmse = np.sqrt(np.mean((pred_flat - test_flat)**2))

        # --- Poisson log-likelihood ---
        # For each valid bin: spike_count ~ Poisson(predicted_rate * occupancy)
        spike_counts = spk_map_test.ravel()[valid]
        occ_flat = occ_test.ravel()[valid]

        # Predicted expected count = rate * occupancy
        expected_counts = np.maximum(pred_flat * occ_flat, 1e-10)

        # Poisson LL: k * log(lambda) - lambda - log(k!)
        from scipy.special import gammaln
        poisson_ll = (spike_counts * np.log(expected_counts)
                      - expected_counts
                      - gammaln(spike_counts + 1))
        total_ll = np.sum(poisson_ll)
        total_spikes = spk_map_test.sum()
        poisson_ll_per_spike = total_ll / max(total_spikes, 1)

        return {
            "poisson_ll_per_spike": poisson_ll_per_spike,
            "correlation": correlation,
            "rmse": rmse,
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_mcmc_diagnostics(gp, figsize=(14, 8)):
    """Plot posterior distributions and trace plots."""
    var_names = ["length_scale", "amplitude", "noise", "mean"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)
    posterior = gp.trace.posterior

    for i, var in enumerate(var_names):
        samples = posterior[var].values
        n_chains = samples.shape[0]

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

        ax_trace = fig.add_subplot(gs[1, i])
        for chain in range(n_chains):
            ax_trace.plot(samples[chain], alpha=0.6, lw=0.5)
        ax_trace.set_title(f"Trace: {var}", fontsize=10)
        ax_trace.set_xlabel("Draw")
        ax_trace.set_ylabel("Value" if i == 0 else "")

    fig.suptitle("MCMC Diagnostics", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("mcmc_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_gp_results(observed_rate_map, gp, grid_positions, shape,
                    x_edges, y_edges, scale,
                    n_posterior_samples=200, n_gp_samples=4, seed=None):
    """
    Plot GP results (MCMC only, no MAP comparison).

    Row 0: Observed | MCMC mean | Uncertainty (hp) | Uncertainty (full) | Samples
    Row 1: Marginal slices with credible intervals
    Row 2: Hyperparameter summary + posterior scatter
    """
    if seed is not None:
        np.random.seed(seed)

    # Predictions
    mean_map, std_map_hp = predict_rate_map(
        gp, grid_positions, shape, n_posterior_samples, method="mean_only"
    )
    _, std_map_full = predict_rate_map(
        gp, grid_positions, shape, n_posterior_samples, method="full"
    )

    # GP samples
    hp = gp.get_hyperparameter_samples(n=n_gp_samples)
    sample_maps = []
    for i in range(n_gp_samples):
        pred_mean, pred_var = gp_posterior(
            gp.X_train, gp.y_train, grid_positions,
            hp["length_scale"][i], hp["amplitude"][i],
            hp["noise"][i], hp["mean"][i]
        )
        log_sample = pred_mean + np.sqrt(pred_var) * np.random.randn(len(pred_mean))
        sample_maps.append(np.exp(log_sample).reshape(shape))

    # Layout
    n_cols = 4 + n_gp_samples
    fig = plt.figure(figsize=(3.2 * n_cols, 10.5))
    gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1.2, 1, 0.7],
                  hspace=0.35, wspace=0.35)

    vmax = np.nanpercentile(observed_rate_map.ravel(), 98)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # Row 0: Rate maps
    col = 0

    ax = fig.add_subplot(gs[0, col]); col += 1
    ax.imshow(observed_rate_map.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('Observed', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, col]); col += 1
    ax.imshow(mean_map.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('MCMC mean', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, col]); col += 1
    im1 = ax.imshow(std_map_hp.T, origin='lower', aspect='auto',
                     extent=extent, cmap='viridis')
    ax.set_title('Uncertainty\n(hyperparams)', fontsize=10, fontweight='bold')
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

    # Row 1: Marginal slices
    peak_idx = np.unravel_index(np.nanargmax(mean_map), mean_map.shape)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # X-slice
    ax = fig.add_subplot(gs[1, :n_cols // 2])
    ax.plot(x_centers, observed_rate_map[:, peak_idx[1]], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(x_centers, mean_map[:, peak_idx[1]], 'b-', lw=2, label='MCMC mean')

    low = mean_map[:, peak_idx[1]] - 2 * std_map_full[:, peak_idx[1]]
    high = mean_map[:, peak_idx[1]] + 2 * std_map_full[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low, 0), high,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')
    low_hp = mean_map[:, peak_idx[1]] - 2 * std_map_hp[:, peak_idx[1]]
    high_hp = mean_map[:, peak_idx[1]] + 2 * std_map_hp[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        ax.plot(x_centers, smap[:, peak_idx[1]], '-', alpha=0.4, lw=1,
                color='gray', label='GP samples' if i == 0 else None)

    ax.set_xlabel('x position'); ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'X-slice at y = {y_centers[peak_idx[1]]:.1f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(x_edges[0], x_edges[-1])

    # Y-slice
    ax = fig.add_subplot(gs[1, n_cols // 2:])
    ax.plot(y_centers, observed_rate_map[peak_idx[0], :], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(y_centers, mean_map[peak_idx[0], :], 'b-', lw=2, label='MCMC mean')

    low = mean_map[peak_idx[0], :] - 2 * std_map_full[peak_idx[0], :]
    high = mean_map[peak_idx[0], :] + 2 * std_map_full[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low, 0), high,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')
    low_hp = mean_map[peak_idx[0], :] - 2 * std_map_hp[peak_idx[0], :]
    high_hp = mean_map[peak_idx[0], :] + 2 * std_map_hp[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        ax.plot(y_centers, smap[peak_idx[0], :], '-', alpha=0.4, lw=1,
                color='gray', label='GP samples' if i == 0 else None)

    ax.set_xlabel('y position'); ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Y-slice at x = {x_centers[peak_idx[0]]:.1f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(y_edges[0], y_edges[-1])

    # Row 2: Summary + pair plot
    ax = fig.add_subplot(gs[2, :n_cols // 2])
    ax.axis('off')
    summary = gp.summary
    hp_text = "Hyperparameter posteriors (mean ± sd  [94% HDI]):\n\n"
    for var_name in ["length_scale", "amplitude", "noise", "mean"]:
        row = summary.loc[var_name]
        hp_text += (f"  {var_name:14s}: {row['mean']:.4f} ± {row['sd']:.4f}   "
                    f"[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]")
        if var_name == "length_scale":
            hp_text += f"  (= {row['mean'] * scale:.1f} original units)"
        hp_text += "\n"
    hp_text += f"\n  R-hat (max): {summary['r_hat'].max():.4f}"
    hp_text += f"  |  ESS bulk (min): {summary['ess_bulk'].min():.0f}"
    n_div = gp.trace.sample_stats["diverging"].sum().values
    hp_text += f"  |  Divergences: {n_div}"
    ax.text(0.05, 0.5, hp_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax_pair = fig.add_subplot(gs[2, n_cols // 2:])
    hp_samples = gp.get_hyperparameter_samples(n=500)
    sc = ax_pair.scatter(hp_samples["length_scale"], hp_samples["amplitude"],
                         c=hp_samples["noise"], cmap='coolwarm', s=8, alpha=0.5)
    ax_pair.set_xlabel("length_scale (normalized)", fontsize=9)
    ax_pair.set_ylabel("amplitude", fontsize=9)
    ax_pair.set_title("Hyperparameter posterior samples", fontsize=10)
    plt.colorbar(sc, ax=ax_pair, label='noise', fraction=0.046)

    fig.suptitle('Bayesian GP Place Field Model (MCMC)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('real_data_gp_mcmc.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mean_map, std_map_hp, std_map_full, sample_maps


def plot_cv_results(cv_results, x_edges, y_edges, figsize=None):
    """
    Plot cross-validation results.

    Top row: per-fold predicted rate maps
    Middle row: per-fold held-out observed rate maps
    Bottom: aggregate metrics summary
    """
    folds = [r for r in cv_results["folds"] if r is not None]
    n_folds = len(folds)

    if figsize is None:
        figsize = (4 * n_folds, 10)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, n_folds, figure=fig, height_ratios=[1, 1, 0.5],
                  hspace=0.35, wspace=0.3)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # Collect vmax across all folds
    all_rates = []
    for r in folds:
        all_rates.append(np.nanmax(r["pred_mean_map"]))
        all_rates.append(np.nanmax(r["test_rate_map"]))
    vmax = np.nanpercentile(all_rates, 95)

    for i, r in enumerate(folds):
        # Top row: predicted
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(r["pred_mean_map"].T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title(f'Fold {r["fold"]} predicted\n'
                     f'[{r["t_lo"]:.0f}–{r["t_hi"]:.0f}s held out]',
                     fontsize=9)
        if i == 0:
            ax.set_ylabel('y')
        ax.set_xlabel('x')

        # Middle row: held-out observed
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(r["test_rate_map"].T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title(f'Fold {r["fold"]} observed\n'
                     f'r={r["correlation"]:.2f}, '
                     f'LL={r["poisson_ll_per_spike"]:.2f}',
                     fontsize=9)
        if i == 0:
            ax.set_ylabel('y')
        ax.set_xlabel('x')

    # Bottom: summary
    agg = cv_results["aggregate"]
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')

    # Per-fold bar chart overlaid as text
    summary_text = (
        f"Cross-validation summary ({n_folds} temporal blocks)\n"
        f"{'─'*55}\n"
        f"  Poisson LL/spike:  {agg['poisson_ll_per_spike_mean']:.3f} "
        f"± {agg['poisson_ll_per_spike_std']:.3f}   "
        f"(per fold: {', '.join(f'{v:.3f}' for v in agg['poisson_ll_per_spike_per_fold'])})\n"
        f"  Correlation:       {agg['correlation_mean']:.3f} "
        f"± {agg['correlation_std']:.3f}   "
        f"(per fold: {', '.join(f'{v:.3f}' for v in agg['correlation_per_fold'])})\n"
        f"  RMSE:              {agg['rmse_mean']:.2f} "
        f"± {agg['rmse_std']:.2f} Hz   "
        f"(per fold: {', '.join(f'{v:.2f}' for v in agg['rmse_per_fold'])})"
    )
    ax.text(0.05, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

    fig.suptitle('Temporal Block Cross-Validation', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    from SessionData import SessionData

    plt.ion()  # interactive mode on

    # ---- Data configuration ----
    base_path = "d:/data/"
    mouse_id = "7012"
    session_id = "m10"
    experiment = "clickbait-motivate"
    cluster_id = 19

    print(f"Loading data: {mouse_id}/{session_id}, cluster {cluster_id}")
    print("=" * 60)

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
    # Asymmetric grid: 9 columns (x) × 20 rows (y) to match arena aspect ratio
    n_bins_x = 9
    n_bins_y = 20
    smooth_sigma = 1.0
    dt = np.median(np.diff(position[:, 0]))

    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins_x + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins_y + 1)

    shape = (n_bins_x, n_bins_y)

    # ---- Compute maps (full session) ----
    occ = compute_occupancy(position, x_edges, y_edges, dt)
    spk = compute_spike_map(spike_times, position, x_edges, y_edges)
    observed_rate_map = estimate_rate_map(spk, occ, smooth_sigma)

    print(f"\nSession duration: {position[-1, 0] - position[0, 0]:.1f} s")
    print(f"Total spikes: {len(spike_times)}")
    print(f"Grid: {n_bins_x} × {n_bins_y} = {n_bins_x * n_bins_y} bins")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")

    # ---- Fit full MCMC model ----
    print("\n" + "=" * 60)
    print("Fitting full MCMC model")
    print("=" * 60)

    gp, grid_norm, grid_raw, valid_mask, scale = fit_gp_mcmc(
        spk, occ, x_edges, y_edges,
        n_samples=1000,
        n_tune=1000,
        n_chains=2,
        target_accept=0.9,
    )

    # ---- Plot diagnostics ----
    print("\nPlotting MCMC diagnostics...")
    plot_mcmc_diagnostics(gp)

    # ---- Plot results ----
    print("Plotting results...")
    mean_map, std_hp, std_full, samples = plot_gp_results(
        observed_rate_map, gp, grid_norm, shape,
        x_edges, y_edges, scale,
        n_posterior_samples=200, n_gp_samples=4, seed=123
    )

    # ---- Cross-validation ----
    print("\n" + "=" * 60)
    print("Running temporal block cross-validation (5 folds)")
    print("=" * 60)

    cv = TemporalBlockCV(n_folds=5)
    cv_results = cv.run(
        position, spike_times, x_edges, y_edges,
        n_samples=500,   # reduced for CV speed
        n_tune=500,
        n_chains=2,
        target_accept=0.9,
        n_pred_samples=100,
    )

    # ---- Plot CV results ----
    print("\nPlotting cross-validation results...")
    plot_cv_results(cv_results, x_edges, y_edges)

    print("\nDone!")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")
    print(f"Peak MCMC mean rate: {np.max(mean_map):.1f} Hz")