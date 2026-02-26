"""
Gaussian Process place field estimation with MCMC inference over hyperparameters.

Extends the GP model from 2_gaussian_process_prior.py by using PyMC's NUTS sampler
to obtain posterior distributions over the kernel hyperparameters (length_scale,
amplitude, noise), then marginalizing over these when predicting firing rate maps.

This gives proper uncertainty estimates that account for hyperparameter uncertainty,
rather than conditioning on a single point estimate from marginal likelihood optimization.

Option 1 approach: MCMC over hyperparameters only; the GP latent function is
marginalized analytically (standard fully Bayesian GP).

Requirements:
    pip install pymc arviz

Usage:
    python 3_gaussian_process_mcmc.py
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve


# =============================================================================
# Kernel functions (same as original)
# =============================================================================

def rbf_kernel(X1, X2, length_scale, amplitude=1.0):
    """RBF (squared exponential) kernel."""
    sq_dist = cdist(X1, X2, metric='sqeuclidean')
    return amplitude**2 * np.exp(-sq_dist / (2 * length_scale**2))


def rbf_kernel_with_noise(X1, X2, length_scale, amplitude=1.0, noise=1e-6):
    """RBF kernel with diagonal noise for numerical stability."""
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

    Parameters
    ----------
    X_train : (n, 2) training positions
    y_train : (n,) training log-rates
    X_pred : (m, 2) prediction positions
    length_scale, amplitude, noise : kernel hyperparameters
    mean : prior mean in log-rate space

    Returns
    -------
    pred_mean : (m,) posterior mean at prediction positions
    pred_var : (m,) posterior marginal variance at prediction positions
    """
    K = rbf_kernel_with_noise(X_train, X_train, length_scale, amplitude, noise**2)
    K_star = rbf_kernel(X_pred, X_train, length_scale, amplitude)
    K_ss_diag = amplitude**2 * np.ones(X_pred.shape[0])  # diagonal of K(X*, X*)

    try:
        L, lower = cho_factor(K, lower=True)
        alpha = cho_solve((L, lower), y_train - mean)
        v = cho_solve((L, lower), K_star.T)

        pred_mean = mean + K_star @ alpha
        pred_var = K_ss_diag - np.sum(K_star.T * v, axis=0)
        pred_var = np.maximum(pred_var, 1e-10)

    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails for this hyperparameter draw
        pred_mean = np.full(X_pred.shape[0], mean)
        pred_var = np.full(X_pred.shape[0], amplitude**2)

    return pred_mean, pred_var


# =============================================================================
# MCMC GP class
# =============================================================================

class GaussianProcessPlaceFieldMCMC:
    """
    Fully Bayesian Gaussian Process model for place field estimation.

    Uses PyMC's NUTS sampler to draw posterior samples of the GP hyperparameters
    (length_scale, amplitude, noise, mean), then computes the analytical GP
    posterior for each sample to marginalize over hyperparameter uncertainty.

    This is "Option 1": MCMC over hyperparameters, analytical GP conditional.
    """

    def __init__(self,
                 # Prior parameters — tune these to your arena/data
                 length_scale_alpha=5.0,
                 length_scale_beta=0.5,
                 amplitude_sigma=2.0,
                 noise_sigma=1.0,
                 mean_sigma=2.0):
        """
        Parameters
        ----------
        length_scale_alpha, length_scale_beta : InverseGamma prior params for length_scale.
            Default alpha=5, beta=0.5 gives a mode around 0.1 with a reasonable
            spread — appropriate for arenas normalized to [0, 1]. Scale beta
            proportionally if your arena is in cm or other units.
        amplitude_sigma : HalfNormal sigma for the amplitude (signal std in log-rate space).
        noise_sigma : HalfNormal sigma for observation noise std.
        mean_sigma : Normal sigma for the prior mean in log-rate space.
        """
        self.length_scale_alpha = length_scale_alpha
        self.length_scale_beta = length_scale_beta
        self.amplitude_sigma = amplitude_sigma
        self.noise_sigma = noise_sigma
        self.mean_sigma = mean_sigma

        # Filled after fitting
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
        positions : (n, 2) array of spatial bin centers
        rates : (n,) array of firing rates (Hz)
        min_rate : floor to avoid log(0)
        n_samples : posterior draws per chain
        n_tune : warmup/tuning draws per chain (discarded)
        n_chains : number of independent chains
        target_accept : NUTS target acceptance rate (increase toward 1.0 if
                        you get divergent transitions)
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
            print(f"Fitting Bayesian GP with MCMC ({n_chains} chains × "
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

            # Marginal likelihood: integrates out the latent function analytically
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

        # Compute summary
        self.summary = az.summary(trace, var_names=["length_scale", "amplitude",
                                                     "noise", "mean"])
        if verbose:
            print("\n--- Posterior summary ---")
            print(self.summary)
            print()

            # Convergence diagnostics
            rhat_max = self.summary["r_hat"].max()
            ess_min = self.summary["ess_bulk"].min()
            print(f"Max R-hat: {rhat_max:.3f} (want < 1.01)")
            print(f"Min ESS (bulk): {ess_min:.0f} (want > 400)")

            # Check for divergences
            n_div = trace.sample_stats["diverging"].sum().values
            if n_div > 0:
                print(f"⚠ {n_div} divergent transitions detected! Consider "
                      f"increasing target_accept or reparameterizing.")
            else:
                print("No divergent transitions detected.")

        return self

    def get_hyperparameter_samples(self, n=None):
        """
        Extract hyperparameter posterior samples from the trace.

        Parameters
        ----------
        n : int or None
            If given, randomly subsample n draws. If None, return all draws.

        Returns
        -------
        dict with keys 'length_scale', 'amplitude', 'noise', 'mean',
        each a 1D array of posterior samples.
        """
        posterior = self.trace.posterior
        # Flatten across chains
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
        Predict firing rates at new positions, marginalizing over hyperparameter
        uncertainty by averaging the analytical GP posterior across MCMC draws.

        Parameters
        ----------
        positions : (m, 2) array of positions to predict at
        n_posterior_samples : number of hyperparameter draws to use
        return_samples : if True, also return individual rate map samples

        Returns
        -------
        mean_rate : (m,) mean predicted rate (marginalized over hyperparams)
        std_rate : (m,) std of predicted rate
        rate_samples : (n_posterior_samples, m) array [only if return_samples=True]
        """
        positions = np.asarray(positions, dtype=np.float64)
        m = positions.shape[0]
        hp = self.get_hyperparameter_samples(n=n_posterior_samples)

        # For each hyperparameter draw, compute the GP posterior mean and sample
        # one function draw from the posterior GP
        log_rate_samples = np.zeros((n_posterior_samples, m))

        for i in range(n_posterior_samples):
            pred_mean, pred_var = gp_posterior(
                self.X_train, self.y_train, positions,
                hp["length_scale"][i], hp["amplitude"][i],
                hp["noise"][i], hp["mean"][i]
            )
            # Draw one sample from the GP posterior for this hyperparameter setting
            log_rate_samples[i] = pred_mean + np.sqrt(pred_var) * np.random.randn(m)

        # Transform to rate space
        rate_samples = np.exp(log_rate_samples)

        mean_rate = np.mean(rate_samples, axis=0)
        std_rate = np.std(rate_samples, axis=0)

        if return_samples:
            return mean_rate, std_rate, rate_samples
        return mean_rate, std_rate

    def predict_mean_only(self, positions, n_posterior_samples=200):
        """
        Predict using only the GP posterior mean (no sampling from the GP
        conditional). This gives a smoother estimate — useful for comparing
        to the MAP/MLE result.

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
# Convenience wrappers (parallel to original)
# =============================================================================

def fit_gp_place_field_mcmc(spike_map, occupancy, x_edges, y_edges,
                             min_occupancy=0.1, **mcmc_kwargs):
    """
    Fit the Bayesian GP place field model to binned spike data.

    Parameters
    ----------
    spike_map : 2D array of spike counts per bin
    occupancy : 2D array of time spent per bin (seconds)
    x_edges, y_edges : bin edges
    min_occupancy : minimum occupancy to include a bin
    **mcmc_kwargs : passed to GaussianProcessPlaceFieldMCMC.fit()

    Returns
    -------
    gp : fitted GaussianProcessPlaceFieldMCMC
    grid_positions : (n_total_bins, 2) array of all bin centers
    valid_mask : boolean mask for bins with sufficient occupancy
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_positions = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute observed rates
    rate_map = np.zeros_like(spike_map, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_map[valid] / occupancy[valid]
    rate_map[~valid] = np.nan

    valid_flat = valid.ravel()
    train_positions = grid_positions[valid_flat]
    train_rates = rate_map.ravel()[valid_flat]

    gp = GaussianProcessPlaceFieldMCMC()
    gp.fit(train_positions, train_rates, **mcmc_kwargs)

    return gp, grid_positions, valid_flat


def predict_rate_map_mcmc(gp, grid_positions, shape,
                           n_posterior_samples=200, method="mean_only"):
    """
    Predict a full rate map from the Bayesian GP.

    Parameters
    ----------
    gp : fitted GaussianProcessPlaceFieldMCMC
    grid_positions : (n, 2) bin centers
    shape : (nx, ny) output shape
    n_posterior_samples : hyperparameter draws to marginalize over
    method : "mean_only" for smoother maps, "full" to also sample from GP conditional

    Returns
    -------
    mean_map, std_map : 2D arrays of predicted rate and uncertainty
    """
    if method == "mean_only":
        mean, std = gp.predict_mean_only(grid_positions, n_posterior_samples)
    else:
        mean, std = gp.predict(grid_positions, n_posterior_samples)

    return mean.reshape(shape), std.reshape(shape)


# =============================================================================
# Visualization — extended with MCMC diagnostics
# =============================================================================

def plot_mcmc_diagnostics(gp, figsize=(14, 8)):
    """
    Plot MCMC diagnostics: posterior distributions and trace plots.

    Parameters
    ----------
    gp : fitted GaussianProcessPlaceFieldMCMC
    """
    var_names = ["length_scale", "amplitude", "noise", "mean"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    posterior = gp.trace.posterior

    for i, var in enumerate(var_names):
        samples = posterior[var].values  # shape: (chains, draws)
        n_chains = samples.shape[0]

        # Top row: posterior density
        ax_hist = fig.add_subplot(gs[0, i])
        for chain in range(n_chains):
            ax_hist.hist(samples[chain], bins=40, alpha=0.5,
                         density=True, label=f"Chain {chain+1}")
        ax_hist.set_title(var, fontsize=11, fontweight='bold')
        ax_hist.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax_hist.legend(fontsize=8)

        # Add summary stats
        all_samples = samples.flatten()
        median = np.median(all_samples)
        hdi = az.hdi(gp.trace, var_names=[var], hdi_prob=0.94)[var].values
        ax_hist.axvline(median, color='k', ls='--', lw=1.5, label='Median')
        ax_hist.axvspan(hdi[0], hdi[1], alpha=0.15, color='steelblue')
        ax_hist.set_xlabel(f"Median: {median:.3f}\n94% HDI: [{hdi[0]:.3f}, {hdi[1]:.3f}]",
                           fontsize=8)

        # Bottom row: trace plot
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


def plot_gp_mcmc_results(observed_rate_map, gp, grid_positions, shape,
                          x_edges, y_edges, n_posterior_samples=200,
                          n_gp_samples=4, seed=None):
    """
    Plot comprehensive GP MCMC results:
      - Row 1: Observed | MCMC posterior mean | Uncertainty (hyperparam only) |
               Uncertainty (full) | GP samples
      - Row 2: Marginal slices through peak with credible intervals
      - Row 3: Hyperparameter info

    Parameters
    ----------
    observed_rate_map : 2D array of observed rates
    gp : fitted GaussianProcessPlaceFieldMCMC
    grid_positions : (n, 2) bin centers
    shape : (nx, ny) map shape
    x_edges, y_edges : bin edges
    n_posterior_samples : hyperparameter draws for prediction
    n_gp_samples : number of GP sample maps to show
    seed : random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Compute predictions ---
    # "mean_only": uncertainty only from hyperparameters
    mean_map_smooth, std_map_hp = predict_rate_map_mcmc(
        gp, grid_positions, shape, n_posterior_samples, method="mean_only"
    )
    # "full": uncertainty from hyperparameters + GP conditional
    mean_map_full, std_map_full = predict_rate_map_mcmc(
        gp, grid_positions, shape, n_posterior_samples, method="full"
    )

    # GP posterior samples (each with a different hyperparameter draw)
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

    # --- Layout ---
    n_sample_cols = n_gp_samples
    n_cols = 4 + n_sample_cols  # observed, mean, std_hp, std_full, + samples
    fig = plt.figure(figsize=(3.2 * n_cols, 10))
    gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1.2, 1, 0.7],
                  hspace=0.35, wspace=0.35)

    vmax = np.nanpercentile(observed_rate_map.ravel(), 98)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # ---- Row 0: Rate maps ----
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(observed_rate_map.T, origin='lower', aspect='auto',
                   vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('Observed', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mean_map_smooth.T, origin='lower', aspect='auto',
              vmin=0, vmax=vmax, extent=extent, cmap='hot')
    ax.set_title('MCMC posterior mean', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    ax = fig.add_subplot(gs[0, 2])
    im_std1 = ax.imshow(std_map_hp.T, origin='lower', aspect='auto',
                         extent=extent, cmap='viridis')
    ax.set_title('Uncertainty\n(hyperparams only)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im_std1, ax=ax, fraction=0.046, label='Hz')

    ax = fig.add_subplot(gs[0, 3])
    im_std2 = ax.imshow(std_map_full.T, origin='lower', aspect='auto',
                         extent=extent, cmap='viridis')
    ax.set_title('Uncertainty\n(full: hp + GP)', fontsize=10, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.colorbar(im_std2, ax=ax, fraction=0.046, label='Hz')

    for i, smap in enumerate(sample_maps):
        ax = fig.add_subplot(gs[0, 4 + i])
        ax.imshow(smap.T, origin='lower', aspect='auto',
                  vmin=0, vmax=vmax, extent=extent, cmap='hot')
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    # ---- Row 1: Marginal slices ----
    peak_idx = np.unravel_index(np.nanargmax(mean_map_smooth), mean_map_smooth.shape)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # X-slice at peak y
    ax = fig.add_subplot(gs[1, :n_cols // 2])
    ax.plot(x_centers, observed_rate_map[:, peak_idx[1]], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(x_centers, mean_map_smooth[:, peak_idx[1]], 'b-', lw=2,
            label='MCMC mean')

    # Credible interval from full posterior
    low_full = mean_map_full[:, peak_idx[1]] - 2 * std_map_full[:, peak_idx[1]]
    high_full = mean_map_full[:, peak_idx[1]] + 2 * std_map_full[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low_full, 0), high_full,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')

    # Credible interval from hyperparameter uncertainty only
    low_hp = mean_map_smooth[:, peak_idx[1]] - 2 * std_map_hp[:, peak_idx[1]]
    high_hp = mean_map_smooth[:, peak_idx[1]] + 2 * std_map_hp[:, peak_idx[1]]
    ax.fill_between(x_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        label = 'GP samples' if i == 0 else None
        ax.plot(x_centers, smap[:, peak_idx[1]], '-', alpha=0.4, lw=1,
                color='gray', label=label)

    ax.set_xlabel('x position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'X-slice at y = {y_centers[peak_idx[1]]:.2f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(x_edges[0], x_edges[-1])

    # Y-slice at peak x
    ax = fig.add_subplot(gs[1, n_cols // 2:])
    ax.plot(y_centers, observed_rate_map[peak_idx[0], :], 'k.', ms=5,
            alpha=0.6, label='Observed')
    ax.plot(y_centers, mean_map_smooth[peak_idx[0], :], 'b-', lw=2,
            label='MCMC mean')

    low_full = mean_map_full[peak_idx[0], :] - 2 * std_map_full[peak_idx[0], :]
    high_full = mean_map_full[peak_idx[0], :] + 2 * std_map_full[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low_full, 0), high_full,
                    alpha=0.15, color='steelblue', label='~95% CI (full)')

    low_hp = mean_map_smooth[peak_idx[0], :] - 2 * std_map_hp[peak_idx[0], :]
    high_hp = mean_map_smooth[peak_idx[0], :] + 2 * std_map_hp[peak_idx[0], :]
    ax.fill_between(y_centers, np.maximum(low_hp, 0), high_hp,
                    alpha=0.25, color='orange', label='~95% CI (hp only)')

    for i, smap in enumerate(sample_maps):
        label = 'GP samples' if i == 0 else None
        ax.plot(y_centers, smap[peak_idx[0], :], '-', alpha=0.4, lw=1,
                color='gray', label=label)

    ax.set_xlabel('y position')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Y-slice at x = {x_centers[peak_idx[0]]:.2f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(y_edges[0], y_edges[-1])

    # ---- Row 2: Hyperparameter posterior summaries ----
    ax = fig.add_subplot(gs[2, :n_cols // 2])
    ax.axis('off')

    summary = gp.summary
    hp_text = "Hyperparameter posteriors (median [94% HDI]):\n\n"
    for var_name in ["length_scale", "amplitude", "noise", "mean"]:
        row = summary.loc[var_name]
        hp_text += (f"  {var_name:14s}: {row['mean']:.4f} ± {row['sd']:.4f}   "
                    f"[{row['hdi_3%']:.4f}, {row['hdi_97%']:.4f}]\n")
    hp_text += f"\n  R-hat (max): {summary['r_hat'].max():.4f}"
    hp_text += f"\n  ESS bulk (min): {summary['ess_bulk'].min():.0f}"

    # Check divergences
    n_div = gp.trace.sample_stats["diverging"].sum().values
    hp_text += f"\n  Divergences: {n_div}"

    ax.text(0.05, 0.5, hp_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Pair plot (mini)
    ax_pair = fig.add_subplot(gs[2, n_cols // 2:])
    hp_samples = gp.get_hyperparameter_samples(n=500)
    sc = ax_pair.scatter(hp_samples["length_scale"], hp_samples["amplitude"],
                         c=hp_samples["noise"], cmap='coolwarm', s=8, alpha=0.5)
    ax_pair.set_xlabel("length_scale", fontsize=9)
    ax_pair.set_ylabel("amplitude", fontsize=9)
    ax_pair.set_title("Hyperparameter posterior samples", fontsize=10)
    plt.colorbar(sc, ax=ax_pair, label='noise', fraction=0.046)

    fig.suptitle('Bayesian GP Place Field Model (MCMC)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('place_field_gp_mcmc.png', dpi=150, bbox_inches='tight')
    plt.show()

    return mean_map_smooth, std_map_hp, std_map_full, sample_maps


# =============================================================================
# Helper functions (same as original — included to keep this self-contained)
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


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # --- Simulate data (identical to original for comparison) ---
    duration = 600  # seconds
    fs_pos = 100
    position = simulate_foraging_trajectory(duration, fs_pos, bounds=(0, 1), speed=0.5)
    t, x, y = position[:, 0], position[:, 1], position[:, 2]
    n_samples = len(t)

    # Two-peaked place field
    peak_rate = 25  # Hz
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

    # --- Fit MCMC model ---
    gp, grid_positions, valid_mask = fit_gp_place_field_mcmc(
        spk, occ, x_edges, y_edges,
        n_samples=1000,   # posterior draws per chain
        n_tune=1000,       # warmup draws
        n_chains=2,
        target_accept=0.9,
    )

    # --- Plot diagnostics ---
    plot_mcmc_diagnostics(gp)

    # --- Plot full results ---
    shape = (n_bins, n_bins)
    mean_map, std_hp, std_full, samples = plot_gp_mcmc_results(
        observed_rate_map, gp, grid_positions, shape, x_edges, y_edges,
        n_posterior_samples=200, n_gp_samples=4, seed=123
    )

    print("\nMCMC GP model fitting complete!")
    print(f"Peak observed rate: {np.nanmax(observed_rate_map):.1f} Hz")
    print(f"Peak predicted rate: {np.max(mean_map):.1f} Hz")