"""
Place field center of mass (COM) drift analysis.

Uses a sliding window (10-min bins, 1-min steps) to estimate the firing rate
map in each window and track how the place field COM moves over the session.
Plots both the absolute COM trajectory and displacement from the initial
window as a stability metric.

Stripped-down rate map pipeline (no GP model):
  - Occupancy map via histogram2d weighted by dt
  - Spike map via histogram2d with interpolated spike positions
  - Gaussian-smoothed rate map (spikes / occupancy)
  - Center of mass as rate-weighted mean of bin center coordinates
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --- Rate map functions ---

def compute_occupancy(position, x_edges, y_edges, dt):
    """Time spent per bin (seconds). position columns: [t, x, y]."""
    occ, _, _ = np.histogram2d(
        position[:, 1], position[:, 2],
        bins=[x_edges, y_edges]
    )
    return occ * dt


def compute_spike_map(spike_times, position, x_edges, y_edges):
    """Spike counts per bin, with spike positions interpolated from trajectory."""
    spike_x = np.interp(spike_times, position[:, 0], position[:, 1])
    spike_y = np.interp(spike_times, position[:, 0], position[:, 2])
    spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
    return spike_map


def estimate_rate_map(spike_map, occupancy, smooth_sigma=1.5, min_occupancy=0.1):
    """Smoothed firing rate map (Hz). Bins below min_occupancy are set to nan."""
    smooth_spikes = gaussian_filter(spike_map.astype(float), sigma=smooth_sigma)
    smooth_occ = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)
    rate_map = np.full_like(smooth_spikes, np.nan)
    valid = smooth_occ > min_occupancy
    rate_map[valid] = smooth_spikes[valid] / smooth_occ[valid]
    return rate_map


# --- Center of mass ---

def compute_com(rate_map, x_centers, y_centers):
    """
    Rate-weighted center of mass of a 2D firing rate map.

    Parameters
    ----------
    rate_map : (nx, ny) array — output of estimate_rate_map
    x_centers, y_centers : bin center coordinates

    Returns
    -------
    com_x, com_y : floats, or (nan, nan) if no valid firing
    """
    valid = ~np.isnan(rate_map) & (rate_map > 0)
    total = rate_map[valid].sum()
    if total == 0:
        return np.nan, np.nan

    # meshgrid with 'ij' indexing matches histogram2d convention
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    com_x = (rate_map[valid] * xx[valid]).sum() / total
    com_y = (rate_map[valid] * yy[valid]).sum() / total
    return com_x, com_y


# --- Sliding window analysis ---

def sliding_window_com(position, spike_times, x_edges, y_edges,
                       window_sec=600, step_sec=60, dt=None,
                       smooth_sigma=1.5, min_occupancy=0.1):
    """
    Compute place field COM in successive sliding time windows.

    Bin edges are fixed across all windows so COM coordinates are comparable.
    Windows with fewer than 10 position samples or no spikes yield nan COM.

    Parameters
    ----------
    position : (N, 3) array of [timestamp_s, x, y]
    spike_times : (S,) array of spike timestamps in seconds
    x_edges, y_edges : bin edges (define once from the full session)
    window_sec : window width in seconds (default 600 = 10 min)
    step_sec : step size in seconds (default 60 = 1 min)
    dt : position sampling interval in seconds; computed from data if None
    smooth_sigma : Gaussian smoothing in bins for rate map estimation
    min_occupancy : minimum occupancy threshold (seconds)

    Returns
    -------
    window_centers : (W,) array — center time of each window (seconds)
    com_x : (W,) array of COM x positions
    com_y : (W,) array of COM y positions
    """
    t = position[:, 0]
    t_start, t_end = t[0], t[-1]

    if dt is None:
        dt = np.median(np.diff(t))

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    window_starts = np.arange(t_start, t_end - window_sec, step_sec)

    window_centers, com_x_list, com_y_list = [], [], []

    for t0 in window_starts:
        t1 = t0 + window_sec
        t_center = t0 + window_sec / 2

        pos_mask = (t >= t0) & (t < t1)
        spk_mask = (spike_times >= t0) & (spike_times < t1)
        pos_win = position[pos_mask]
        spk_win = spike_times[spk_mask]

        if pos_win.shape[0] < 10 or spk_win.size == 0:
            window_centers.append(t_center)
            com_x_list.append(np.nan)
            com_y_list.append(np.nan)
            continue

        occ = compute_occupancy(pos_win, x_edges, y_edges, dt)
        spk_map = compute_spike_map(spk_win, pos_win, x_edges, y_edges)
        rate_map = estimate_rate_map(spk_map, occ, smooth_sigma, min_occupancy)

        cx, cy = compute_com(rate_map, x_centers, y_centers)
        window_centers.append(t_center)
        com_x_list.append(cx)
        com_y_list.append(cy)

    return np.array(window_centers), np.array(com_x_list), np.array(com_y_list)


# --- Plotting ---

def plot_com_drift(window_centers, com_x, com_y, title='Place field COM drift'):
    """
    Four-panel figure summarising COM drift over the session.

    Top-left  : 2D COM trajectory in space, colored by time.
    Top-right : Euclidean displacement from the first valid window's COM.
    Bottom-left  : COM x position over time.
    Bottom-right : COM y position over time.

    Parameters
    ----------
    window_centers : (W,) array of window center times (seconds)
    com_x, com_y : (W,) arrays of COM coordinates
    title : figure suptitle

    Returns
    -------
    fig : matplotlib Figure
    """
    t_min = window_centers / 60.0          # seconds → minutes
    valid = ~np.isnan(com_x) & ~np.isnan(com_y)

    if not valid.any():
        print("No valid COM estimates — nothing to plot.")
        return None

    # Reference COM: first valid window
    ref_idx = np.where(valid)[0][0]
    ref_x, ref_y = com_x[ref_idx], com_y[ref_idx]

    displacement = np.full(len(com_x), np.nan)
    displacement[valid] = np.sqrt(
        (com_x[valid] - ref_x) ** 2 + (com_y[valid] - ref_y) ** 2
    )

    # Color scale for time (0 = start, 1 = end)
    norm_time = (t_min - t_min[valid][0]) / (t_min[valid][-1] - t_min[valid][0])
    colors = cm.plasma(norm_time[valid])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14)

    # --- Top-left: 2D COM trajectory ---
    ax = axes[0, 0]
    sc = ax.scatter(com_x[valid], com_y[valid], c=norm_time[valid],
                    cmap='plasma', s=40, zorder=3)
    ax.plot(com_x[valid], com_y[valid], '-', color='gray', lw=0.8,
            alpha=0.5, zorder=2)
    ax.plot(com_x[ref_idx], com_y[ref_idx], 'o', color='green',
            ms=10, zorder=4, label='start')
    last_valid = np.where(valid)[0][-1]
    ax.plot(com_x[last_valid], com_y[last_valid], 's', color='red',
            ms=10, zorder=4, label='end')
    plt.colorbar(sc, ax=ax, label='Normalised time')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')
    ax.set_title('COM trajectory in space')
    ax.legend(fontsize=9)

    # --- Top-right: displacement from initial COM ---
    ax = axes[0, 1]
    ax.plot(t_min[valid], displacement[valid], 'k.-', ms=5)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Displacement from initial COM')
    ax.set_title('Euclidean COM displacement')

    # --- Bottom-left: COM x over time ---
    ax = axes[1, 0]
    ax.plot(t_min[valid], com_x[valid], '.-', color='steelblue', ms=5)
    ax.axhline(ref_x, color='green', lw=0.8, ls='--', label='initial x')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('COM x position')
    ax.set_title('COM x over time')
    ax.legend(fontsize=9)

    # --- Bottom-right: COM y over time ---
    ax = axes[1, 1]
    ax.plot(t_min[valid], com_y[valid], '.-', color='tomato', ms=5)
    ax.axhline(ref_y, color='green', lw=0.8, ls='--', label='initial y')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('COM y position')
    ax.set_title('COM y over time')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('com_drift.png', dpi=150)
    plt.show()

    return fig


# --- Entry point ---

if __name__ == '__main__':
    from SessionData import SessionData

    # ------------------------------------------------------------------ #
    #  Data configuration — edit to match your session                    #
    # ------------------------------------------------------------------ #
    base_path  = "d:/data/"
    mouse_id   = "7012"
    session_id = "m10"
    experiment = "clickbait-motivate"
    cluster_id = 19

    window_sec   = 600   # 10-minute windows
    step_sec     = 60    # 1-minute steps
    n_bins       = 25
    smooth_sigma = 1.5
    # ------------------------------------------------------------------ #

    print(f"Loading data: mouse {mouse_id} / session {session_id} / cluster {cluster_id}")
    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True
    )

    # Position array: [timestamp_s, x, y]
    position = data.events[['timestamp_ms', 'nose_x', 'nose_y']].to_numpy(dtype=float)
    position[:, 0] /= 1000.0  # ms → seconds

    # Spike times in seconds
    spike_times = data.clusters[cluster_id]['spike_times'].astype(float) / 1000.0

    session_duration = position[-1, 0] - position[0, 0]
    print(f"Session duration : {session_duration / 60:.1f} min")
    print(f"Total spikes     : {len(spike_times)}")
    print(f"Mean firing rate : {len(spike_times) / session_duration:.2f} Hz")

    if session_duration < window_sec:
        raise ValueError(
            f"Session ({session_duration:.0f} s) is shorter than one window ({window_sec} s)."
        )

    # Bin edges from full-session spatial range (consistent across all windows)
    dt = np.median(np.diff(position[:, 0]))
    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    print(f"\nRunning sliding window COM analysis "
          f"({window_sec // 60}-min windows, {step_sec // 60}-min steps)...")

    window_centers, com_x, com_y = sliding_window_com(
        position, spike_times, x_edges, y_edges,
        window_sec=window_sec, step_sec=step_sec, dt=dt,
        smooth_sigma=smooth_sigma
    )

    n_valid = (~np.isnan(com_x)).sum()
    print(f"Windows computed : {len(window_centers)}")
    print(f"Valid windows    : {n_valid}")

    plot_com_drift(
        window_centers, com_x, com_y,
        title=f'Place field COM drift — mouse {mouse_id}, session {session_id}, '
              f'cluster {cluster_id}'
    )

    print("\nDone. Saved com_drift.png")
