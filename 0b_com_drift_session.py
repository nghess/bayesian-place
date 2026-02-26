"""
Session-level place field COM drift analysis.

Runs the sliding window COM analysis from 0_com_drift.py over every cluster
in a session and superimposes the resulting curves in a single figure.
Individual cell curves are shown in light color; a bold mean curve is
overlaid. This is the starting point for channel-based segregation
(see next iteration).

Data format (from SessionData):
  position[:, 0]  — timestamp in seconds (converted from ms)
  position[:, 1]  — x coordinate
  position[:, 2]  — y coordinate
  data.clusters[idx]['spike_times']  — spike times in ms
  data.clusters[idx]['cluster_id']   — original Kilosort cluster ID
  data.clusters[idx]['best_channel'] — channel with max waveform amplitude
  data.clusters[idx]['n_spikes']     — spike count
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# --- Rate map functions (self-contained, same as 0_com_drift.py) ---

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
    """Smoothed firing rate map (Hz). Bins below min_occupancy are nan."""
    smooth_spikes = gaussian_filter(spike_map.astype(float), sigma=smooth_sigma)
    smooth_occ = gaussian_filter(occupancy.astype(float), sigma=smooth_sigma)
    rate_map = np.full_like(smooth_spikes, np.nan)
    valid = smooth_occ > min_occupancy
    rate_map[valid] = smooth_spikes[valid] / smooth_occ[valid]
    return rate_map


def compute_com(rate_map, x_centers, y_centers):
    """Rate-weighted center of mass. Returns (nan, nan) if no valid firing."""
    valid = ~np.isnan(rate_map) & (rate_map > 0)
    total = rate_map[valid].sum()
    if total == 0:
        return np.nan, np.nan
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    com_x = (rate_map[valid] * xx[valid]).sum() / total
    com_y = (rate_map[valid] * yy[valid]).sum() / total
    return com_x, com_y


def sliding_window_com(position, spike_times, x_edges, y_edges,
                       window_sec=600, step_sec=60, dt=None,
                       smooth_sigma=1.5, min_occupancy=0.1):
    """
    COM in successive sliding windows. Bin edges are fixed (passed in from
    the caller) so all cells share the same spatial reference frame.
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
        pos_mask = (t >= t0) & (t < t1)
        spk_mask = (spike_times >= t0) & (spike_times < t1)
        pos_win = position[pos_mask]
        spk_win = spike_times[spk_mask]

        t_center = t0 + window_sec / 2
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


# --- Session-level analysis ---

def run_session_com_analysis(data, position, x_edges, y_edges, dt,
                             window_sec=600, step_sec=60,
                             smooth_sigma=1.5, min_occupancy=0.1):
    """
    Run sliding window COM analysis for every cluster in data.clusters.

    Parameters
    ----------
    data : SessionData object
    position : (N, 3) array [timestamp_s, x, y]
    x_edges, y_edges : bin edges (shared across all clusters)
    dt : position sampling interval in seconds
    window_sec, step_sec : window and step size in seconds
    smooth_sigma, min_occupancy : rate map parameters

    Returns
    -------
    results : dict keyed by cluster index, each entry containing:
        'cluster_id'   : original Kilosort cluster ID
        'best_channel' : channel with max waveform amplitude
        'n_spikes'     : spike count
        'window_centers': (W,) array of window center times (seconds)
        'com_x', 'com_y': (W,) arrays of COM positions
        'displacement'  : (W,) Euclidean displacement from the preceding valid COM (step-to-step)
    """
    results = {}

    for idx, cluster in data.clusters.items():
        spike_times = cluster['spike_times'].astype(float) / 1000.0  # ms → s

        window_centers, com_x, com_y = sliding_window_com(
            position, spike_times, x_edges, y_edges,
            window_sec=window_sec, step_sec=step_sec, dt=dt,
            smooth_sigma=smooth_sigma, min_occupancy=min_occupancy
        )

        valid = ~np.isnan(com_x) & ~np.isnan(com_y)
        displacement = np.full(len(com_x), np.nan)
        valid_idx = np.where(valid)[0]
        for j in range(1, len(valid_idx)):
            i_prev, i_curr = valid_idx[j - 1], valid_idx[j]
            displacement[i_curr] = np.sqrt(
                (com_x[i_curr] - com_x[i_prev]) ** 2 +
                (com_y[i_curr] - com_y[i_prev]) ** 2
            )

        results[idx] = {
            'cluster_id':    cluster['cluster_id'],
            'best_channel':  cluster['best_channel'],
            'n_spikes':      cluster['n_spikes'],
            'window_centers': window_centers,
            'com_x':         com_x,
            'com_y':         com_y,
            'displacement':  displacement,
        }

        n_valid = valid.sum()
        print(f"  cluster {cluster['cluster_id']:4d} "
              f"(ch {cluster['best_channel']:3d}, "
              f"{cluster['n_spikes']:5d} spikes) — "
              f"{n_valid}/{len(window_centers)} valid windows")

    return results


# --- Plotting ---

def _get_true_spans(t, mask):
    """
    Return list of (t_start, t_end) for each contiguous True block in mask.

    Parameters
    ----------
    t : (N,) array of times (any units)
    mask : (N,) boolean array

    Returns
    -------
    spans : list of (t_start, t_end) tuples
    """
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]   # index of first True in each block
    ends   = np.where(diff == -1)[0]  # index just past last True in each block
    return [(t[s], t[e - 1]) for s, e in zip(starts, ends)]


def _plot_displacement_panel(ax, group_results, group_label, flip_spans=None):
    """
    Plot superimposed displacement curves for one group of clusters onto ax.

    Individual cluster curves are thin and semi-transparent; a bold group
    mean is overlaid. Returns early (no error) if the group is empty.
    """
    if not group_results:
        ax.text(0.5, 0.5, f'No clusters in {group_label}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(group_label)
        return

    n = len(group_results)
    cmap = cm.get_cmap('tab20' if n <= 20 else 'viridis', n)

    all_disp = []
    ref_t_min = None

    for i, (idx, res) in enumerate(group_results.items()):
        t_min = res['window_centers'] / 60.0
        if ref_t_min is None:
            ref_t_min = t_min

        disp = res['displacement']
        valid = ~np.isnan(disp)
        label = f"cl {res['cluster_id']} ch{res['best_channel']}"

        ax.plot(t_min[valid], disp[valid],
                color=cmap(i), lw=1.0, alpha=0.55, label=label)
        all_disp.append(disp)

    # Group mean
    if n > 1:
        mean_disp = np.nanmean(all_disp, axis=0)
        ax.plot(ref_t_min, mean_disp, 'k-', lw=2.5, label='mean', zorder=5)

    # Shade flip_state True periods behind all curves
    if flip_spans:
        for t_start, t_end in flip_spans:
            ax.axvspan(t_start, t_end, color='steelblue', alpha=0.15, zorder=0)

    ax.axhline(0, color='gray', lw=0.7, ls='--')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Step-to-step COM displacement')
    ax.set_title(f'{group_label} (n={n})')

    legend_fontsize = max(5, min(9, 120 // n))
    ax.legend(loc='upper left', fontsize=legend_fontsize,
              ncol=max(1, n // 10), framealpha=0.6)


def plot_session_com_drift(results, title='Session COM drift',
                           channel_threshold=16,
                           flip_t_min=None, flip_state=None):
    """
    Two-panel figure separating clusters by brain region.

    Top    : OB clusters  (best_channel <= channel_threshold)
    Bottom : HC clusters  (best_channel >  channel_threshold)

    Each panel shows one Euclidean-displacement curve per cluster
    (thin, semi-transparent) plus a bold group mean. Periods where
    flip_state is True are shaded as a light background bar.

    Parameters
    ----------
    results : dict returned by run_session_com_analysis
    title : figure suptitle
    channel_threshold : channel boundary between OB and HC (default 16)
    flip_t_min : (N,) array of event timestamps in minutes (from data.events)
    flip_state : (N,) boolean array aligned to flip_t_min

    Returns
    -------
    fig : matplotlib Figure
    """
    if not results:
        print("No clusters to plot.")
        return None

    ob = {k: v for k, v in results.items() if v['best_channel'] <= channel_threshold}
    hc = {k: v for k, v in results.items() if v['best_channel'] >  channel_threshold}

    flip_spans = None
    if flip_t_min is not None and flip_state is not None:
        flip_spans = _get_true_spans(flip_t_min, flip_state)

    fig, (ax_ob, ax_hc) = plt.subplots(2, 1, figsize=(14, 10),
                                        sharex=True,
                                        gridspec_kw={'hspace': 0.35})

    _plot_displacement_panel(ax_ob, ob, 'OB Clusters', flip_spans=flip_spans)
    _plot_displacement_panel(ax_hc, hc, 'HC Clusters', flip_spans=flip_spans)

    fig.suptitle(title, fontsize=13)
    plt.savefig('com_drift_session.png', dpi=150)
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

    window_sec   = 300   # 5-minute windows
    step_sec     = 10    # 10-second steps
    n_bins       = 25
    smooth_sigma = 1.5
    # ------------------------------------------------------------------ #

    print(f"Loading data: mouse {mouse_id} / session {session_id}")
    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True
    )

    print(f"\n{data.n_clusters} clusters loaded.")

    # Position array: [timestamp_s, x, y]
    position = data.events[['timestamp_ms', 'nose_x', 'nose_y']].to_numpy(dtype=float)
    position[:, 0] /= 1000.0  # ms → seconds

    session_duration = position[-1, 0] - position[0, 0]
    print(f"Session duration : {session_duration / 60:.1f} min")

    if session_duration < window_sec:
        raise ValueError(
            f"Session ({session_duration:.0f} s) is shorter than one window ({window_sec} s)."
        )

    # Shared bin edges and dt — defined once from the full session
    dt = np.median(np.diff(position[:, 0]))
    x_edges = np.linspace(position[:, 1].min(), position[:, 1].max(), n_bins + 1)
    y_edges = np.linspace(position[:, 2].min(), position[:, 2].max(), n_bins + 1)

    print(f"\nRunning sliding window COM analysis "
          f"({window_sec // 60}-min windows, {step_sec // 60}-min steps) "
          f"across {data.n_clusters} clusters...")

    results = run_session_com_analysis(
        data, position, x_edges, y_edges, dt,
        window_sec=window_sec, step_sec=step_sec,
        smooth_sigma=smooth_sigma
    )

    # flip_state: convert event timestamps to minutes for alignment with COM plots
    flip_t_min  = data.events['timestamp_ms'].to_numpy(dtype=float) / 60000.0
    flip_state  = data.events['flip_state'].to_numpy(dtype=bool)

    plot_session_com_drift(
        results,
        title=f'Place field COM drift — mouse {mouse_id}, session {session_id}',
        flip_t_min=flip_t_min,
        flip_state=flip_state,
    )

    print("\nDone. Saved com_drift_session.png")
