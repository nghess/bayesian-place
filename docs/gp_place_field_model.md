# Gaussian Process Place Field Model

## Overview

This document describes the mathematical foundation of our Gaussian Process (GP) model for place field estimation. The goal is to infer the spatial firing rate function of a neuron from discrete spike observations, while enforcing smoothness and quantifying uncertainty.

## 1. Problem Setup

### Observed Data

We observe:
- **Position trajectory**: $(x_t, y_t)$ for $t = 1, \ldots, T$ sampled at rate $f_s$
- **Spike times**: $\{s_1, s_2, \ldots, s_N\}$

### Goal

Estimate the firing rate function $\lambda(x, y)$ that describes how the neuron's activity varies across space.

### Binned Representation

We discretize space into a grid of $B \times B$ bins. For each bin $i$ with center $(x_i, y_i)$:
- $n_i$ = number of spikes in bin $i$
- $\tau_i$ = total time spent in bin $i$ (occupancy, in seconds)

The observed firing rate in bin $i$ is:
$$\hat{r}_i = \frac{n_i}{\tau_i}$$

Bins with insufficient occupancy ($\tau_i < \tau_{\min}$) are excluded from fitting.

## 2. Generative Model

### Poisson Likelihood

We assume spikes in each bin follow a Poisson process:
$$n_i \sim \text{Poisson}(\lambda_i \cdot \tau_i)$$

where $\lambda_i = \lambda(x_i, y_i)$ is the true firing rate at bin $i$.

### Log-Rate Parameterization

To ensure positive firing rates, we model the **log firing rate** as a Gaussian Process:
$$f(x, y) = \log \lambda(x, y)$$
$$f \sim \mathcal{GP}(m, k)$$

where:
- $m$ is the prior mean function (we use $m = 0$, implying prior mean rate of 1 Hz)
- $k(x, x')$ is the covariance (kernel) function

This gives us:
$$\lambda(x, y) = \exp(f(x, y))$$

### Why Log-Rate?

1. **Positivity**: $\exp(f) > 0$ for any $f \in \mathbb{R}$
2. **Multiplicative effects**: Additive changes in $f$ correspond to multiplicative changes in rate
3. **Approximate Gaussian likelihood**: For bins with sufficient spikes, $\log \hat{r}_i \approx \mathcal{N}(f_i, \sigma^2_i)$

## 3. The Gaussian Process Prior

### Definition

A Gaussian Process is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is fully specified by:
- Mean function: $m(x) = \mathbb{E}[f(x)]$
- Covariance function: $k(x, x') = \text{Cov}[f(x), f(x')]$

For any finite set of locations $X = \{x_1, \ldots, x_n\}$:
$$\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top \sim \mathcal{N}(\mathbf{m}, K)$$

where $K_{ij} = k(x_i, x_j)$.

### Radial Basis Function (RBF) Kernel

We use the squared exponential (RBF) kernel:
$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

**Hyperparameters**:
- $\ell$ (length scale): Controls smoothness. Points closer than $\ell$ are strongly correlated.
- $\sigma_f$ (amplitude): Controls the magnitude of variation in $f$.

**Properties**:
- Infinitely differentiable (very smooth)
- Stationary (depends only on $x - x'$)
- Isotropic (same in all directions)

### Observation Noise

We add an observation noise term to account for:
- Poisson variability in spike counts
- Measurement noise
- Model misspecification

The full kernel for training points is:
$$k_y(x, x') = k(x, x') + \sigma_n^2 \delta(x, x')$$

where $\delta$ is the Kronecker delta (noise only on diagonal).

## 4. Gaussian Process Inference

### Training

Given $n$ training points with positions $X$ and observed log-rates $\mathbf{y} = [\log \hat{r}_1, \ldots, \log \hat{r}_n]^\top$:

1. Compute the kernel matrix: $K = k(X, X) + \sigma_n^2 I$
2. Solve: $\boldsymbol{\alpha} = K^{-1}(\mathbf{y} - \mathbf{m})$

In practice, we use Cholesky decomposition for numerical stability:
$$K = LL^\top$$
$$\boldsymbol{\alpha} = L^\top \backslash (L \backslash (\mathbf{y} - \mathbf{m}))$$

### Prediction

For a new location $x_*$, the posterior distribution is:
$$f_* | X, \mathbf{y}, x_* \sim \mathcal{N}(\mu_*, \sigma_*^2)$$

**Posterior mean**:
$$\mu_* = m(x_*) + k(x_*, X) \boldsymbol{\alpha}$$

**Posterior variance**:
$$\sigma_*^2 = k(x_*, x_*) - k(x_*, X) K^{-1} k(X, x_*)$$

The predicted firing rate is:
$$\hat{\lambda}_* = \exp(\mu_*)$$

### Uncertainty in Rate Space

Using the delta method approximation:
$$\text{std}(\lambda_*) \approx \lambda_* \cdot \sigma_*$$

This gives us credible intervals for the firing rate.

## 5. Hyperparameter Optimization

### Log Marginal Likelihood

We optimize hyperparameters $\boldsymbol{\theta} = (\ell, \sigma_f, \sigma_n)$ by maximizing the log marginal likelihood:

$$\log p(\mathbf{y} | X, \boldsymbol{\theta}) = -\frac{1}{2}(\mathbf{y} - \mathbf{m})^\top K^{-1}(\mathbf{y} - \mathbf{m}) - \frac{1}{2}\log|K| - \frac{n}{2}\log(2\pi)$$

**Interpretation of terms**:
1. **Data fit**: $-\frac{1}{2}(\mathbf{y} - \mathbf{m})^\top K^{-1}(\mathbf{y} - \mathbf{m})$ — how well the model explains the data
2. **Complexity penalty**: $-\frac{1}{2}\log|K|$ — penalizes overly flexible models
3. **Normalization**: $-\frac{n}{2}\log(2\pi)$

### Optimization

We use L-BFGS-B with bounds:
- Length scale: $[0.05, 0.5]$ (in normalized coordinates)
- Amplitude: $[0.5, 5.0]$
- Noise: $[0.1, 2.0]$

These bounds are chosen for normalized position coordinates in $[0, 1]$.

## 6. Posterior Sampling

To generate samples from the posterior (for visualization and uncertainty propagation):

1. Compute posterior mean $\boldsymbol{\mu}_*$ and covariance $\Sigma_*$ at grid points
2. Sample: $\mathbf{f}_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)$
3. Transform: $\boldsymbol{\lambda}_* = \exp(\mathbf{f}_*)$

The posterior covariance is:
$$\Sigma_* = K_{**} - K_{*X} K^{-1} K_{X*}$$

where $K_{**} = k(X_*, X_*)$ and $K_{*X} = k(X_*, X)$.

## 7. Implementation Notes

### Coordinate Normalization

Real data may have coordinates in pixels (e.g., 0-800). We normalize to $[0, 1]$ to ensure hyperparameter bounds are meaningful:

$$x_{\text{norm}} = \frac{x - x_{\min}}{\max(x_{\text{range}}, y_{\text{range}})}$$

Using the maximum range preserves aspect ratio.

### Numerical Stability

- Add small jitter ($10^{-6}$) to diagonal for Cholesky decomposition
- Use Cholesky solve instead of explicit matrix inverse
- Clip rates to minimum value before log transform

### Minimum Rate

We use $\lambda_{\min} = 0.1$ Hz to avoid $\log(0)$:
$$\mathbf{y} = \log(\max(\hat{\mathbf{r}}, \lambda_{\min}))$$

## 8. Model Assumptions and Limitations

### Assumptions

1. **Stationarity**: The smoothness properties are the same everywhere in the arena
2. **Isotropy**: The correlation structure is the same in all directions
3. **Log-Gaussian rates**: The log firing rate is approximately Gaussian
4. **Independence**: Spikes in different bins are conditionally independent given the rate

### Limitations

1. **Computational cost**: $O(n^3)$ for $n$ bins (Cholesky decomposition)
2. **Stationary kernel**: Cannot capture spatially-varying smoothness
3. **Single length scale**: Same smoothness in $x$ and $y$ directions
4. **Point estimate of hyperparameters**: Does not account for hyperparameter uncertainty

### Potential Extensions

1. **Automatic Relevance Determination (ARD)**: Separate length scales for $x$ and $y$
2. **Sparse GPs**: For computational efficiency with many bins
3. **Non-stationary kernels**: For spatially-varying smoothness
4. **Full Bayesian treatment**: MCMC over hyperparameters

## 9. References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Brown, E. N., Frank, L. M., Tang, D., Quirk, M. C., & Wilson, M. A. (1998). A statistical paradigm for neural spike train decoding applied to position prediction from ensemble firing patterns of rat hippocampal place cells. *Journal of Neuroscience*, 18(18), 7411-7425.
- Park, M., Weller, J. P., Bhattacharyya, P., Bhalla, U. S., & Bhattacharyya, S. (2014). Bayesian active learning for parameter estimation of receptive fields. *BMC Neuroscience*, 15(1), 1-2.
