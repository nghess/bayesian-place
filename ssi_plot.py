"""
ssi_plot.py - Skaggs spatial information and SSI for SessionData clusters.

Computes occupancy/spike histograms and a null SI distribution via circular
shuffle permutation for every cluster in a loaded session. Saves one figure
per cluster and appends metrics to a cumulative CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.ndimage import zoom
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Spike alignment
# ---------------------------------------------------------------------------

def align_spikes(spike_times_flat, mtrack):
    """Return DataFrame of spike times with their closest frame positions."""
    frame_ms = mtrack['frame_ms'].values
    idx = np.searchsorted(frame_ms, spike_times_flat)
    idx = np.clip(idx, 1, len(frame_ms) - 1)
    left_diff = np.abs(spike_times_flat - frame_ms[idx - 1])
    right_diff = np.abs(spike_times_flat - frame_ms[idx])
    closest = np.where(left_diff < right_diff, idx - 1, idx)
    return pd.DataFrame({
        'spike_time': spike_times_flat,
        'frame_ms':   frame_ms[closest],
        'head_x':     mtrack['head_x'].values[closest],
        'head_y':     mtrack['head_y'].values[closest],
    })


# ---------------------------------------------------------------------------
# Spatial information (Skaggs)
# ---------------------------------------------------------------------------

def skaggs_si(spike_rate_flat, p_i):
    """
    Vectorized Skaggs spatial information calculation.
    Returns SI in bits/spike, or 0 if the mean rate is zero.
    """
    valid = ~np.isnan(spike_rate_flat)
    denom = np.sum(p_i[valid])
    if denom == 0:
        return 0.0
    r_mean = np.sum(spike_rate_flat[valid] * p_i[valid]) / denom
    if r_mean == 0:
        return 0.0
    active = valid & (spike_rate_flat > 0)
    ratio = spike_rate_flat[active] / r_mean
    return float(np.sum(p_i[active] * ratio * np.log2(ratio)))


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def run_permutations(spike_times_flat, mtrack, occupancy_time,
                     x_edges, y_edges, p_i, min_occupancy_s,
                     n_permutations=1000):
    """Circular shuffle permutation test; returns null SI distribution."""
    frame_ms = mtrack['frame_ms'].values
    t_min = mtrack['frame_ms'].min()
    t_max = mtrack['frame_ms'].max()
    recording_duration = t_max - t_min

    null_si = np.zeros(n_permutations)

    for perm in range(n_permutations):
        shift = np.random.uniform(0, recording_duration)
        shifted = spike_times_flat + shift
        shifted = np.where(shifted > t_max, shifted - recording_duration, shifted)

        idx = np.searchsorted(frame_ms, shifted)
        idx = np.clip(idx, 1, len(frame_ms) - 1)
        ld = np.abs(shifted - frame_ms[idx - 1])
        rd = np.abs(shifted - frame_ms[idx])
        closest = np.where(ld < rd, idx - 1, idx)

        sh_hist, _, _ = np.histogram2d(
            mtrack['head_x'].values[closest],
            mtrack['head_y'].values[closest],
            bins=[x_edges, y_edges]
        )
        sh_rate = np.divide(
            sh_hist, occupancy_time,
            out=np.full_like(sh_hist, np.nan, dtype=float),
            where=occupancy_time >= min_occupancy_s
        )
        null_si[perm] = skaggs_si(sh_rate.flatten(), p_i)

    return null_si


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_figure(mouse_id, session_id, cluster,
                mtrack, spike_positions_df, spike_times_flat,
                n_permutations=1000, grid_shape=(9, 20),
                min_occupancy_ms=1000, upsample_factor=100,
                best_channel=None, n_spikes=None):
    """
    Build the full 2x3 subplot figure for one cell.

    Subplots:
        [0,0] Trajectory + spike positions
        [0,1] Occupancy histogram
        [0,2] Spike count histogram
        [1,0] Spike rate map
        [1,1] Spike rate map (upsampled)
        [1,2] Null SI distribution
    """
    FS = 15        # 1.5 × default 10 — body text, labels, ticks, legend
    SUPTITLE_FS = 26  # 2 × previous suptitle 13

    # --- Grid edges ---
    x_edges = np.linspace(mtrack['head_x'].min(), mtrack['head_x'].max(), grid_shape[0] + 1)
    y_edges = np.linspace(mtrack['head_y'].min(), mtrack['head_y'].max(), grid_shape[1] + 1)

    # --- Histograms ---
    occupancy_hist, _, _ = np.histogram2d(
        mtrack['head_x'], mtrack['head_y'], bins=[x_edges, y_edges]
    )
    spike_hist, _, _ = np.histogram2d(
        spike_positions_df['head_x'], spike_positions_df['head_y'],
        bins=[x_edges, y_edges]
    )

    # --- Occupancy in seconds ---
    median_interval_ms = np.median(np.diff(mtrack['frame_ms'].values[:100]))
    time_per_frame_s = median_interval_ms / 1000
    occupancy_time = occupancy_hist * time_per_frame_s
    min_occupancy_s = min_occupancy_ms / 1000

    # --- Spike rate map ---
    spike_rate = np.divide(
        spike_hist, occupancy_time,
        out=np.full_like(spike_hist, np.nan, dtype=float),
        where=occupancy_time >= min_occupancy_s
    )

    # --- Spatial information ---
    occ_flat = occupancy_time.flatten()
    p_i = occ_flat / np.sum(occ_flat)
    spatial_info = skaggs_si(spike_rate.flatten(), p_i)

    # --- Null distribution ---
    print(f"  Running {n_permutations} permutations...")
    null_si = run_permutations(
        spike_times_flat, mtrack, occupancy_time,
        x_edges, y_edges, p_i, min_occupancy_s, n_permutations
    )
    null_mean = np.mean(null_si)
    null_std = np.std(null_si)
    z_score = (spatial_info - null_mean) / null_std if null_std > 0 else 0.0
    p_value = float(1 - norm.cdf(z_score))
    p_value_empirical = float(np.sum(null_si >= spatial_info) / n_permutations)

    # --- Upsampled rate map ---
    valid_mask = ~np.isnan(spike_rate.T)
    rate_filled = np.nan_to_num(spike_rate.T, nan=0.0)
    rate_upsampled = zoom(rate_filled, upsample_factor, order=3)
    mask_upsampled = zoom(valid_mask.astype(float), upsample_factor, order=3)
    rate_upsampled[mask_upsampled < 0.5] = np.nan

    # -----------------------------------------------------------------------
    # Build figure
    # -----------------------------------------------------------------------
    title_parts = [f"Mouse {mouse_id}, Session {session_id} - Cl{cluster}"]
    if best_channel is not None:
        title_parts.append(f"ch{best_channel}")
    if n_spikes is not None:
        title_parts.append(f"{n_spikes} spikes")
    title_base = "   ".join(title_parts)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # [0,0] Trajectory + spikes
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(mtrack['head_x'], mtrack['head_y'], 'o', ms=1, alpha=0.1, color='C0')
    ax0.scatter(
        spike_positions_df['head_x'], spike_positions_df['head_y'],
        s=2, color='red', alpha=1, zorder=2
    )
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('Head X', fontsize=FS)
    ax0.set_ylabel('Head Y', fontsize=FS)
    ax0.set_title('Trajectory & Spike Positions', fontsize=FS)
    ax0.tick_params(labelsize=FS)

    # [0,1] Occupancy histogram
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        occupancy_time.T, origin='lower', aspect='equal',
        cmap='viridis', extent=extent
    )
    ax1.set_xlabel('Head X', fontsize=FS)
    ax1.set_ylabel('Head Y', fontsize=FS)
    ax1.set_title('Occupancy Histogram', fontsize=FS)
    ax1.tick_params(labelsize=FS)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Seconds')
    cbar1.ax.yaxis.label.set_size(FS)
    cbar1.ax.tick_params(labelsize=FS)

    # [0,2] Spike count histogram
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(
        spike_hist.T, origin='lower', aspect='equal',
        cmap='hot', extent=extent
    )
    ax2.set_xlabel('Head X', fontsize=FS)
    ax2.set_ylabel('Head Y', fontsize=FS)
    ax2.set_title('Spike Count Histogram', fontsize=FS)
    ax2.tick_params(labelsize=FS)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Spike Counts')
    cbar2.ax.yaxis.label.set_size(FS)
    cbar2.ax.tick_params(labelsize=FS)

    # [1,0] Spike rate map
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        spike_rate.T, origin='lower', aspect='equal',
        cmap='jet', extent=extent
    )
    ax3.set_xlabel('Head X', fontsize=FS)
    ax3.set_ylabel('Head Y', fontsize=FS)
    ax3.set_title('Spike Rate Map', fontsize=FS)
    ax3.tick_params(labelsize=FS)
    cbar3 = plt.colorbar(im3, ax=ax3, label='Spikes/Sec')
    cbar3.ax.yaxis.label.set_size(FS)
    cbar3.ax.tick_params(labelsize=FS)

    # [1,1] Upsampled spike rate map
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(
        rate_upsampled, origin='lower', aspect='equal',
        cmap='jet', extent=extent, interpolation='bilinear'
    )
    ax4.set_xlabel('Head X', fontsize=FS)
    ax4.set_ylabel('Head Y', fontsize=FS)
    ax4.set_title(f'Spike Rate Map (Upsampled {upsample_factor}x)', fontsize=FS)
    ax4.tick_params(labelsize=FS)
    cbar4 = plt.colorbar(im4, ax=ax4, label='Spikes/Sec')
    cbar4.ax.yaxis.label.set_size(FS)
    cbar4.ax.tick_params(labelsize=FS)

    # [1,2] Null SI distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(null_si, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Null Distribution')
    ax5.axvline(spatial_info, color='red', lw=2, ls='--',
                label=f'SI = {spatial_info:.3f}')
    ax5.axvline(null_mean, color='blue', lw=1, ls=':',
                label=f'Null Mean = {null_mean:.3f}')
    ax5.axvline(null_mean + 3 * null_std, color='green', lw=1, ls=':',
                label=f'3σ = {null_mean + 3 * null_std:.3f}')
    ax5.set_xlabel('Spatial Information (bits/spike)', fontsize=FS)
    ax5.set_ylabel('Frequency', fontsize=FS)
    ax5.set_title(f'SSI Null Distribution\n(z={z_score:.2f}, p={p_value:.4f})', fontsize=FS)
    ax5.tick_params(labelsize=FS)
    ax5.legend(fontsize=FS)

    fig.suptitle(title_base, fontsize=SUPTITLE_FS, fontweight='bold')
    return fig, spatial_info, z_score, p_value_empirical


# ---------------------------------------------------------------------------
# SessionData entry point
# ---------------------------------------------------------------------------

def run_session_ssi(data, mouse_id, session_id,
                    n_permutations=1000,
                    grid_shape=(9, 20),
                    min_occupancy_ms=1000,
                    upsample_factor=100,
                    plots_root='plots',
                    results_csv='plots/ssi_results.csv'):
    """
    Compute SI and SSI for every cluster in a SessionData object.

    Saves one figure per cluster to:
        <plots_root>/<mouse_id>/<session_id>/<cluster_id>.png

    Appends results to <results_csv> (creates the file if it doesn't exist).

    Parameters
    ----------
    data : SessionData
    mouse_id, session_id : str — used for labelling and file paths
    n_permutations : number of circular-shuffle permutations
    grid_shape : (nx, ny) spatial bin counts (default 9x20 for 888x1968 arena)
    min_occupancy_ms : bins visited fewer than this many ms are excluded
    upsample_factor : zoom factor for the upsampled rate map panel
    plots_root : root directory for saved figures
    results_csv : path to the cumulative results CSV

    Returns
    -------
    records : list of dicts with keys
        mouse_id, session_id, cluster_id, best_channel, n_spikes, si, ssi, p_value
    """
    # --- Prepare mtrack: rename columns to match existing function signatures ---
    # align_spikes / run_permutations / make_figure all expect
    # frame_ms, head_x, head_y
    mtrack = (
        data.events[['timestamp_ms', 'nose_x', 'nose_y']]
        .rename(columns={'timestamp_ms': 'frame_ms',
                         'nose_x':       'head_x',
                         'nose_y':       'head_y'})
        .reset_index(drop=True)
    )

    # --- Output directory ---
    out_dir = Path(plots_root) / str(mouse_id) / str(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Loop over clusters ---
    records = []
    n_total = data.n_clusters

    for i, (idx, cluster) in enumerate(data.clusters.items()):
        cluster_id = cluster['cluster_id']
        print(f"[{i + 1}/{n_total}] cluster {cluster_id} "
              f"(ch {cluster['best_channel']}, {cluster['n_spikes']} spikes)")

        spike_times_flat = cluster['spike_times'].astype(float)  # ms

        # Filter to tracking window.  align_spikes uses np.clip, so any spike
        # outside [t_min, t_max] would be clamped to the first/last frame and
        # accumulate at that position, creating a spurious hotspot shared by
        # every cluster in the session.
        t_min_ms = float(mtrack['frame_ms'].min())
        t_max_ms = float(mtrack['frame_ms'].max())
        in_window = (spike_times_flat >= t_min_ms) & (spike_times_flat <= t_max_ms)
        n_dropped = int((~in_window).sum())
        if n_dropped:
            print(f"  WARNING: {n_dropped} spikes outside tracking window dropped")
        spike_times_flat = spike_times_flat[in_window]

        spike_positions_df = align_spikes(spike_times_flat, mtrack)

        fig, si, z_score, p_value = make_figure(
            mouse_id, session_id, cluster_id,
            mtrack, spike_positions_df, spike_times_flat,
            n_permutations=n_permutations,
            grid_shape=grid_shape,
            min_occupancy_ms=min_occupancy_ms,
            upsample_factor=upsample_factor,
            best_channel=cluster['best_channel'],
            n_spikes=cluster['n_spikes'],
        )

        out_path = out_dir / f"{cluster_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  SI={si:.4f}  SSI(z)={z_score:.3f}  p={p_value:.4f}  → {out_path}")

        records.append({
            'mouse_id':     mouse_id,
            'session_id':   session_id,
            'cluster_id':   cluster_id,
            'best_channel': cluster['best_channel'],
            'n_spikes':     cluster['n_spikes'],
            'si':           si,
            'ssi':          z_score,
            'p_value':      p_value,
        })

    # --- Append to cumulative CSV ---
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if results_path.exists():
        existing = pd.read_csv(results_path)
        # Drop any rows already present for this mouse/session to avoid duplicates
        existing = existing[
            ~((existing['mouse_id'].astype(str) == str(mouse_id)) &
              (existing['session_id'].astype(str) == str(session_id)))
        ]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}  ({len(combined)} total rows)")

    return records


if __name__ == '__main__':
    from SessionData import SessionData

    # ------------------------------------------------------------------ #
    #  Data configuration — edit to match your session                    #
    # ------------------------------------------------------------------ #
    base_path    = "d:/data/"
    mouse_id     = "7012"
    session_id   = "m10"
    experiment   = "clickbait-motivate"
    n_perms      = 1000
    # ------------------------------------------------------------------ #

    print(f"Loading data: mouse {mouse_id} / session {session_id}")
    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True,
    )
    print(f"{data.n_clusters} clusters loaded.\n")

    records = run_session_ssi(
        data, mouse_id, session_id,
        n_permutations=n_perms,
    )

    print(f"\nDone. {len(records)} clusters processed.")
