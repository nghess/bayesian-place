"""
shuffle_frames.py — Visualise the circular shuffle procedure for a single unit.

Generates one PNG per evenly-spaced shuffle step, suitable for assembling
into a video with ffmpeg or similar tools.  Each frame shows three panels:

  1. Trajectory + shifted spike positions
  2. Raw spike rate map
  3. Upsampled rate map

Colour bars are omitted for clean video rendering.  The colour scale is fixed
to the peak of the *unshuffled* rate map so that the field can be seen to
dissolve and reappear as the shift advances through the full session duration.

Callable as a function so it can be imported from a notebook:

    from shuffle_frames import make_shuffle_frames
    make_shuffle_frames(data, cluster_idx=0,
                        mouse_id='7012', session_id='m10')

Files are saved as:
    <output_dir>/<mouse_id>/<session_id>/<cluster_id>/<step:04d>.png

so that the frame sequence is immediately usable with e.g.:
    ffmpeg -framerate 30 -i %04d.png -c:v libx264 shuffle.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom

from ssi_plot import align_spikes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_rate_map(spike_df, occupancy_time, x_edges, y_edges,
                      min_occupancy_s):
    """Bin spikes spatially and divide by occupancy; under-visited bins → nan."""
    sh, _, _ = np.histogram2d(
        spike_df['head_x'], spike_df['head_y'],
        bins=[x_edges, y_edges]
    )
    return np.divide(
        sh, occupancy_time,
        out=np.full_like(sh, np.nan, dtype=float),
        where=occupancy_time >= min_occupancy_s
    )


def _upsample(rate_map, factor):
    """Smooth-zoom a rate map for display; NaN regions are preserved."""
    valid  = ~np.isnan(rate_map.T)
    filled = np.nan_to_num(rate_map.T, nan=0.0)
    up     = zoom(filled, factor, order=3)
    mask   = zoom(valid.astype(float), factor, order=3)
    up[mask < 0.5] = np.nan
    return up


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def make_shuffle_frames(
    data,
    cluster_idx,
    mouse_id,
    session_id,
    n_steps=1000,
    output_dir='shuffle_frames',
    grid_shape=(9, 20),
    min_occupancy_ms=1000,
    upsample_factor=10,
    figsize=(14, 5),
    dpi=120,
):
    """
    Generate one PNG per shuffle step showing how circular time-shifting
    moves a place cell's spikes around the arena.

    Parameters
    ----------
    data : SessionData
    cluster_idx : int
        Key into data.clusters (sequential index, not Kilosort cluster ID).
    mouse_id, session_id : str
        Used for the suptitle and output path.
    n_steps : int
        Number of evenly-spaced steps spanning the full session duration.
        Step i applies a shift of i * (duration / n_steps).
        Step 0 is always the unshuffled original.
    output_dir : str or Path
        Root directory; frames go to <output_dir>/<mouse_id>/<session_id>/
        <cluster_id>/<step:04d>.png.
    grid_shape : (nx, ny)
        Spatial bin counts (default 9 × 20 for the 888 × 1968 arena).
    min_occupancy_ms : float
        Bins visited fewer than this many milliseconds are masked.
    upsample_factor : int
        Zoom factor applied to the rate map for the third panel.
    figsize : (w, h)
        Matplotlib figure size in inches.
    dpi : int
        Output resolution.  120 dpi at (14, 5) inches → 1680 × 600 px.

    Returns
    -------
    out_dir : Path
        Directory containing the saved frames.
    """
    cluster    = data.clusters[cluster_idx]
    cluster_id = cluster['cluster_id']

    # --- Position data (same column convention as ssi_plot.py) ---
    mtrack = (
        data.events[['timestamp_ms', 'nose_x', 'nose_y']]
        .rename(columns={'timestamp_ms': 'frame_ms',
                         'nose_x':       'head_x',
                         'nose_y':       'head_y'})
        .reset_index(drop=True)
    )

    t_min_ms    = float(mtrack['frame_ms'].min())
    t_max_ms    = float(mtrack['frame_ms'].max())
    duration_ms = t_max_ms - t_min_ms

    # --- Filter spikes to tracking window (same fix as ssi_plot.py) ---
    spike_times = cluster['spike_times'].astype(float)   # ms
    in_window   = (spike_times >= t_min_ms) & (spike_times <= t_max_ms)
    n_dropped   = int((~in_window).sum())
    if n_dropped:
        print(f"  WARNING: {n_dropped} spikes outside tracking window dropped")
    spike_times = spike_times[in_window]

    # --- Spatial grid ---
    x_edges = np.linspace(mtrack['head_x'].min(), mtrack['head_x'].max(),
                          grid_shape[0] + 1)
    y_edges = np.linspace(mtrack['head_y'].min(), mtrack['head_y'].max(),
                          grid_shape[1] + 1)
    extent  = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # --- Occupancy (fixed — does not change with shuffle) ---
    median_interval_ms = float(np.median(np.diff(mtrack['frame_ms'].values)))
    time_per_frame_s   = median_interval_ms / 1000.0
    min_occupancy_s    = min_occupancy_ms / 1000.0

    occ_hist, _, _ = np.histogram2d(
        mtrack['head_x'], mtrack['head_y'], bins=[x_edges, y_edges]
    )
    occupancy_time = occ_hist * time_per_frame_s   # seconds per bin

    # --- Unshuffled rate map → fixes vmax for the entire sequence ---
    rate0 = _compute_rate_map(
        align_spikes(spike_times, mtrack),
        occupancy_time, x_edges, y_edges, min_occupancy_s
    )
    vmax = float(np.nanmax(rate0))

    # --- Output directory ---
    out_dir = (Path(output_dir) / str(mouse_id)
               / str(session_id) / str(cluster_id))
    out_dir.mkdir(parents=True, exist_ok=True)

    step_ms = duration_ms / n_steps

    print(f"Cluster {cluster_id}  |  {len(spike_times)} spikes  |  "
          f"{duration_ms / 60_000:.1f}-min session")
    print(f"Step size : {step_ms / 1000:.2f} s   "
          f"vmax fixed at {vmax:.1f} Hz")
    print(f"Saving {n_steps} frames to {out_dir}\n")

    traj_x = mtrack['head_x'].values
    traj_y = mtrack['head_y'].values

    for step in range(n_steps):
        shift_ms = step * step_ms
        shift_s  = shift_ms / 1000.0

        # Circular shift
        shifted  = spike_times + shift_ms
        shifted  = np.where(shifted > t_max_ms,
                            shifted - duration_ms, shifted)

        spk_pos  = align_spikes(shifted, mtrack)
        rate_map = _compute_rate_map(spk_pos, occupancy_time,
                                     x_edges, y_edges, min_occupancy_s)
        rate_up  = _upsample(rate_map, upsample_factor)

        # --- Figure (fixed size — no bbox_inches='tight' to keep dims stable) ---
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: trajectory + shifted spike positions
        ax = axes[0]
        ax.plot(traj_x, traj_y, 'o', ms=1, alpha=0.08,
                color='C0', rasterized=True)
        ax.scatter(spk_pos['head_x'], spk_pos['head_y'],
                   s=2, color='red', alpha=0.6, zorder=2, rasterized=True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory & Spikes')

        # Panel 2: raw rate map
        ax = axes[1]
        ax.imshow(rate_map.T, origin='lower', aspect='equal',
                  cmap='jet', extent=extent, vmin=0, vmax=vmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Rate Map')

        # Panel 3: upsampled rate map
        ax = axes[2]
        ax.imshow(rate_up, origin='lower', aspect='equal',
                  cmap='jet', extent=extent, vmin=0, vmax=vmax,
                  interpolation='bilinear')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Rate Map (×{upsample_factor})')

        fig.suptitle(
            f'Mouse {mouse_id}   Session {session_id}   Cl {cluster_id}'
            f'        shift = {shift_s:8.1f} s  ({shift_s / 60:5.2f} min)'
            f'        frame {step + 1:04d} / {n_steps}',
            fontsize=10
        )
        plt.tight_layout()
        fig.savefig(out_dir / f'{step:04d}.png', dpi=dpi)
        plt.close(fig)

        if step == 0 or (step + 1) % 100 == 0:
            print(f'  [{step + 1:4d}/{n_steps}]  shift = {shift_s:7.1f} s')

    print(f'\nDone. {n_steps} frames → {out_dir}')
    return out_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from SessionData import SessionData

    # ------------------------------------------------------------------ #
    base_path   = 'd:/data/'
    mouse_id    = '7011'
    session_id  = 'm11'
    experiment  = 'clickbait-motivate'
    cluster_idx = 30       # index into data.clusters (not Kilosort ID)
    n_steps     = 1000
    # ------------------------------------------------------------------ #

    data = SessionData(
        base_path=base_path,
        mouse_id=mouse_id,
        session_id=session_id,
        experiment=experiment,
        min_spikes=50,
        verbose=True,
    )

    make_shuffle_frames(
        data, cluster_idx,
        mouse_id=mouse_id,
        session_id=session_id,
        n_steps=n_steps,
    )
