"""
Spatial rate map and occupancy map utilities for electrophysiology analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter
from pathlib import Path


def _compute_percentile_limits(data, lower_percentile=5, upper_percentile=95):
    """
    Compute percentile-based limits for colormap scaling.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of data values (can contain NaNs)
    lower_percentile : float
        Lower percentile for vmin (default: 5)
    upper_percentile : float
        Upper percentile for vmax (default: 95)
        
    Returns:
    --------
    vmin : float
        Lower percentile value
    vmax : float
        Upper percentile value
    """
    # Get non-NaN values
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return 0, 1
    
    vmin = np.percentile(valid_data, lower_percentile)
    vmax = np.percentile(valid_data, upper_percentile)
    
    # Ensure vmin < vmax
    if vmin >= vmax:
        vmax = vmin + 1
    
    return vmin, vmax


def _upsample_map(data, zoom_factor=5):
    """
    Upsample spatial map with smooth interpolation for visualization.
    
    This is for visualization only - does not affect calculations.
    Uses bilinear interpolation to create smooth, continuous-looking heatmaps.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array to upsample (can contain NaNs)
    zoom_factor : float
        How much to upsample (default: 5 = 5x resolution)
        
    Returns:
    --------
    upsampled : numpy.ndarray
        Upsampled array with smooth interpolation between bins
    """
    from scipy.ndimage import zoom
    
    # Use bilinear interpolation (order=1) for smooth appearance
    # This creates smooth transitions between bins
    upsampled = zoom(data, zoom_factor, order=1)
    
    return upsampled


def compute_occupancy_map(position_times, positions, bin_size=50.7, 
                         fixed_arena_bounds=None, min_occupancy=0.1,
                         return_metadata=True):
    """
    Compute occupancy map from position data.
    
    Parameters:
    -----------
    position_times : array-like
        Position timestamps in milliseconds
    positions : DataFrame
        Position data with centroid_x, centroid_y columns
    bin_size : float
        Spatial bin size in pixels (default: 50.7, which equals 1 cm)
    fixed_arena_bounds : tuple, optional
        Fixed arena bounds (x_min, x_max, y_min, y_max)
    min_occupancy : float
        Minimum occupancy threshold in seconds (default: 0.1)
    return_metadata : bool
        Whether to return metadata dict along with occupancy map (default: True)
        
    Returns:
    --------
    occupancy_map : numpy.ndarray
        2D array of occupancy times (in seconds) for each spatial bin
    metadata : dict (if return_metadata=True)
        Dictionary containing:
        - 'x_edges': bin edges for x dimension
        - 'y_edges': bin edges for y dimension
        - 'x_min', 'x_max', 'y_min', 'y_max': arena bounds
        - 'bin_size': spatial bin size used
        - 'dt': median time step in seconds
        - 'total_time': total time in valid bins (seconds)
        - 'valid_bins': boolean mask of bins meeting min_occupancy threshold
        - 'n_valid_bins': number of valid bins
        
    Examples:
    ---------
    >>> # Basic usage
    >>> occupancy_map, metadata = compute_occupancy_map(pos_times, positions)
    >>> 
    >>> # Get occupancy map only (without metadata)
    >>> occupancy_map = compute_occupancy_map(pos_times, positions, 
    ...                                       return_metadata=False)
    """
    
    # Convert to arrays
    if 'centroid_x' in positions.columns:
        positions_array = positions[['centroid_x', 'centroid_y']].values
    else:
        raise ValueError("Position DataFrame must contain 'centroid_x' and 'centroid_y' columns")
    
    pos_times_array = position_times.values if hasattr(position_times, 'values') else np.array(position_times)
    
    # Calculate arena bounds
    if fixed_arena_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_arena_bounds
    else:
        x_min, x_max = np.nanmin(positions_array[:, 0]), np.nanmax(positions_array[:, 0])
        y_min, y_max = np.nanmin(positions_array[:, 1]), np.nanmax(positions_array[:, 1])
        margin = bin_size * 0.5
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin
    
    # Create spatial bins
    x_range = [x_min, x_max]
    y_range = [y_min, y_max]
    x_bins = int(np.ceil((x_range[1] - x_range[0]) / bin_size))
    y_bins = int(np.ceil((y_range[1] - y_range[0]) / bin_size))
    
    x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], y_bins + 1)
    
    # Calculate occupancy map
    dt = np.median(np.asarray(np.diff(pos_times_array), dtype=np.float64)) / 1000.0
    occupancy_map, _, _ = np.histogram2d(
        positions_array[:, 0], positions_array[:, 1], 
        bins=[x_edges, y_edges]
    )
    occupancy_map = occupancy_map * dt
    
    if return_metadata:
        # Calculate valid bins and total time
        valid_bins = occupancy_map >= min_occupancy
        total_time = np.sum(occupancy_map[valid_bins])
        n_valid_bins = np.sum(valid_bins)
        
        metadata = {
            'x_edges': x_edges,
            'y_edges': y_edges,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'bin_size': bin_size,
            'dt': dt,
            'total_time': total_time,
            'valid_bins': valid_bins,
            'n_valid_bins': n_valid_bins
        }
        
        return occupancy_map, metadata
    else:
        return occupancy_map


def plot_occupancy_map(mouse_id, session_id,
                       events_path,
                       position_times=None, positions=None,
                       bin_size=50.7, min_occupancy=0.1,
                       fixed_arena_bounds=None,
                       title=None,
                       xlabel='X Position (cm)',
                       ylabel='Y Position (cm)',
                       cbar_label='Occupancy (s)',
                       cmap='viridis',
                       vmin=None,
                       vmax=None,
                       use_percentile_limits=False,
                       sigma=0,
                       upsample=False,
                       log_scale=False,
                       min_max_ticks_only=False,
                       cbar_min_max_only=True,
                       cbar_shrink=0.76,
                       hide_all_ticks=False,
                       hide_axis_labels=False,
                       figsize=(6, 6), save_plot=False, output_dir=None):
    """
    Create occupancy map with units in cm and 90-degree rotation.
    
    Parameters:
    -----------
    use_percentile_limits : bool
        if True, automatically set vmin and vmax to 5th and 95th percentiles (default: False).
        Overrides vmin and vmax if they are None.
    sigma : float
        Gaussian smoothing sigma for visualization (default: 0 = no smoothing).
        Applied after calculation, for display only.
    upsample : bool
        if True, upsample the display map to native pixel resolution (default: False).
        For visualization only, does not affect calculations.
    """
    px_per_cm = 50.7
    
    # Load data if not provided
    if position_times is None or positions is None:
        events = pd.read_csv(events_path)
        position_times = events['timestamp_ms']
        positions = events[['centroid_x', 'centroid_y']]
    
    # Convert to arrays
    positions_array = positions[['centroid_x', 'centroid_y']].values
    if hasattr(position_times, 'values'):
        pos_times_array = position_times.values
    else:
        pos_times_array = np.array(position_times)
    
    # Calculate arena bounds
    if fixed_arena_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_arena_bounds
    else:
        # Start from 0 and extend to max of data plus margin
        x_max = np.nanmax(positions_array[:, 0])
        y_max = np.nanmax(positions_array[:, 1])
        margin = bin_size * 0.5
        x_min, x_max = 0, x_max + margin
        y_min, y_max = 0, y_max + margin
    
    # Create spatial bins
    x_bins = int(np.ceil((x_max - x_min) / bin_size))
    y_bins = int(np.ceil((y_max - y_min) / bin_size))
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    
    # Calculate occupancy map (swap x and y for 90-degree rotation)
    dt = np.median(np.asarray(np.diff(pos_times_array), dtype=np.float64)) / 1000.0
    occupancy_map, _, _ = np.histogram2d(
        positions_array[:, 1], positions_array[:, 0],  # Swapped: y, x instead of x, y
        bins=[y_edges, x_edges]  # Swapped: y_edges, x_edges
    )
    occupancy_map = occupancy_map * dt
    
    # For display: transpose to get correct orientation
    occupancy_map_display = occupancy_map.T
    
    # Apply visualization-only transformations (smoothing and upsampling)
    # These do NOT affect the original occupancy_map used for calculations
    if sigma > 0:
        occupancy_map_display = gaussian_filter(occupancy_map_display, sigma=sigma)
    
    if upsample:
        occupancy_map_display = _upsample_map(occupancy_map_display, zoom_factor=5)
    
    # Convert extent to cm - swapped for rotation: y range on x-axis, x range on y-axis
    extent_cm = [y_min / px_per_cm, y_max / px_per_cm, 
                 x_min / px_per_cm, x_max / px_per_cm]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Apply log scale if requested
    plot_data = occupancy_map_display.copy()
    if log_scale:
        plot_data = np.log10(plot_data + 1e-10)  # Add small value to avoid log(0)
        if cbar_label == 'Occupancy (s)':  # Update label if using default
            cbar_label = 'Log10 Occupancy (s)'
    
    # Use percentile limits if requested (and vmin/vmax not explicitly set)
    if use_percentile_limits:
        if vmin is None or vmax is None:
            percentile_vmin, percentile_vmax = _compute_percentile_limits(plot_data)
            if vmin is None:
                vmin = percentile_vmin
            if vmax is None:
                vmax = percentile_vmax
    
    # Create x and y coordinates for the heatmap
    x_coords = np.linspace(extent_cm[0], extent_cm[1], plot_data.shape[1])
    y_coords = np.linspace(extent_cm[2], extent_cm[3], plot_data.shape[0])
    
    # Plot using seaborn heatmap for better SVG rendering
    im = sns.heatmap(plot_data, ax=ax, cmap=cmap, 
                     vmin=vmin, vmax=vmax,
                     cbar_kws={'label': cbar_label, 'shrink': cbar_shrink},
                     xticklabels=False, yticklabels=False,
                     square=True)
    
    # Get the colorbar to modify ticks
    cbar = im.collections[0].colorbar
    
    # Set colorbar ticks to min/max only if requested
    if cbar_min_max_only:
        actual_vmin, actual_vmax = cbar.vmin, cbar.vmax
        cbar.set_ticks([actual_vmin, actual_vmax])
    
    # Add black border around the heatmap
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)
    
    # Manually set the extent since seaborn doesn't use extent parameter
    ax.set_xlim(0, plot_data.shape[1])
    ax.set_ylim(0, plot_data.shape[0])
    
    # Create custom tick positions for extent in cm
    n_xticks = 2  # min and max
    n_yticks = 2
    xtick_positions = np.linspace(0, plot_data.shape[1], n_xticks)
    ytick_positions = np.linspace(0, plot_data.shape[0], n_yticks)
    xtick_labels = [f'{extent_cm[0]:.1f}', f'{extent_cm[1]:.1f}']
    ytick_labels = [f'{extent_cm[2]:.1f}', f'{extent_cm[3]:.1f}']
    
    # Set axis labels (unless hidden)
    if not hide_axis_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)
    
    ax.set_aspect('equal', adjustable='box')  # Force equal aspect ratio
    
    # Set min/max ticks only if requested (overrides default)
    if min_max_ticks_only:
        ax.set_xticks(xtick_positions)
        ax.set_yticks(ytick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticklabels(ytick_labels)
    
    # Hide all ticks and tick labels if requested
    if hide_all_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    if title is None:
        ax.set_title(f'{mouse_id} {session_id}\nOccupancy Map', pad=20)
    else:
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    
    # Save if requested
    if save_plot:
        if output_dir is None:
            output_dir = "."
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        plot_filename = f"{output_dir}/{mouse_id}_{session_id}_occupancy.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename}")
    
    return fig, {
        'occupancy_map': occupancy_map,
        'occupancy_map_display': occupancy_map_display
    }


def plot_rate_map(mouse_id, session_id, cluster_id,
                  kilosort_path, events_path,
                  spike_times_array=None, position_times=None, positions=None,
                  bin_size=50.7, sigma=0, min_occupancy=0.1,
                  fixed_arena_bounds=None,
                  title=None,
                  xlabel='X Position (cm)',
                  ylabel='Y Position (cm)',
                  cbar_label='Spikes/sec',
                  cmap='viridis',
                  vmin=None,
                  vmax=None,
                  use_percentile_limits=False,
                  upsample=False,
                  min_max_ticks_only=False,
                  cbar_min_max_only=True,
                  cbar_shrink=0.76,
                  hide_all_ticks=False,
                  hide_axis_labels=False,
                  figsize=(6, 6), save_plot=False, output_dir=None):
    """
    Create rate map with units in cm and 90-degree rotation.
    
    Can either load data from paths OR use provided arrays.
    
    Parameters:
    -----------
    use_percentile_limits : bool
        if True, automatically set vmin and vmax to 5th and 95th percentiles (default: False).
        Overrides vmin and vmax if they are None.
    sigma : float
        Gaussian smoothing sigma for visualization (default: 0 = no smoothing).
        Applied after rate calculation, before display.
    upsample : bool
        if True, upsample the display map to native pixel resolution (default: False).
        For visualization only, does not affect spatial information calculation.
    """
    px_per_cm = 50.7
    
    # Load data if not provided
    if spike_times_array is None or position_times is None or positions is None:
        spike_times = np.load(f"{kilosort_path}\\spike_times.npy")
        spike_templates = np.load(f"{kilosort_path}\\spike_templates.npy")
        
        events = pd.read_csv(events_path)
        position_times = events['timestamp_ms']
        positions = events[['centroid_x', 'centroid_y']]
        
        cl_spikes = np.where(spike_templates == cluster_id)[0]
        cl_spike_times = spike_times[cl_spikes] / 30.0
    else:
        cl_spike_times = spike_times_array
    
    # Convert to arrays
    positions_array = positions[['centroid_x', 'centroid_y']].values
    if hasattr(position_times, 'values'):
        pos_times_array = position_times.values
    else:
        pos_times_array = np.array(position_times)
    spike_times_array = np.array(cl_spike_times)
    
    # Calculate arena bounds
    if fixed_arena_bounds is not None:
        x_min, x_max, y_min, y_max = fixed_arena_bounds
    else:
        # Start from 0 and extend to max of data plus margin
        x_max = np.nanmax(positions_array[:, 0])
        y_max = np.nanmax(positions_array[:, 1])
        margin = bin_size * 0.5
        x_min, x_max = 0, x_max + margin
        y_min, y_max = 0, y_max + margin
    
    # Create spatial bins
    x_bins = int(np.ceil((x_max - x_min) / bin_size))
    y_bins = int(np.ceil((y_max - y_min) / bin_size))
    x_edges = np.linspace(x_min, x_max, x_bins + 1)
    y_edges = np.linspace(y_min, y_max, y_bins + 1)
    
    # Calculate occupancy map (swap x and y for 90-degree rotation)
    dt = np.median(np.asarray(np.diff(pos_times_array), dtype=np.float64)) / 1000.0
    occupancy_map, _, _ = np.histogram2d(
        positions_array[:, 1], positions_array[:, 0],  # Swapped: y, x instead of x, y
        bins=[y_edges, x_edges]  # Swapped: y_edges, x_edges
    )
    occupancy_map = occupancy_map * dt
    
    # Filter spikes by time range
    valid_spikes = (spike_times_array >= pos_times_array[0]) & (spike_times_array <= pos_times_array[-1])
    filtered_spike_times = spike_times_array[valid_spikes]
    
    # Calculate spike positions and spike map (swap x and y)
    if len(filtered_spike_times) > 0:
        spike_x = np.interp(filtered_spike_times, np.asarray(pos_times_array, dtype=np.float64), 
                           positions_array[:, 0])
        spike_y = np.interp(filtered_spike_times, np.asarray(pos_times_array, dtype=np.float64), 
                           positions_array[:, 1])
        spike_map, _, _ = np.histogram2d(spike_y, spike_x, bins=[y_edges, x_edges])  # Swapped
    else:
        spike_map = np.zeros_like(occupancy_map)
    
    # Calculate rate map
    valid_bins = occupancy_map >= min_occupancy
    rate_map = np.zeros_like(occupancy_map)
    rate_map[valid_bins] = spike_map[valid_bins] / occupancy_map[valid_bins]
    
    # Calculate spatial information BEFORE smoothing (using unsmoothed rate_map)
    # This ensures spatial info calculation is not affected by visualization parameters
    total_spikes = len(filtered_spike_times)
    total_time = np.sum(occupancy_map[valid_bins])
    mean_rate = total_spikes / total_time if total_time > 0 else 0
    
    if mean_rate > 0:
        rate_map_array = np.asarray(rate_map, dtype=np.float64)
        valid_rate_bins = valid_bins & (rate_map_array > 0)
        if np.any(valid_rate_bins):
            p_i = occupancy_map[valid_rate_bins] / total_time
            r_i = rate_map_array[valid_rate_bins]
            spatial_info = np.sum(p_i * (r_i / float(mean_rate)) * np.log2(r_i / float(mean_rate)))
        else:
            spatial_info = 0.0
    else:
        spatial_info = 0.0
    
    # Now apply visualization-only transformations (smoothing and upsampling)
    # These do NOT affect spatial information calculation
    if sigma > 0:
        rate_map = gaussian_filter(rate_map, sigma=sigma)
    
    # Create visualization array
    rate_map_viz = np.copy(rate_map)
    rate_map_viz[~valid_bins] = np.nan
    
    # For display: transpose to get correct orientation  
    rate_map_display = rate_map_viz.T
    
    # Apply upsampling for visualization if requested
    if upsample:
        rate_map_display = _upsample_map(rate_map_display, zoom_factor=5)
    
    # Use percentile limits if requested (and vmin/vmax not explicitly set)
    if use_percentile_limits:
        if vmin is None or vmax is None:
            percentile_vmin, percentile_vmax = _compute_percentile_limits(rate_map_display)
            if vmin is None:
                vmin = percentile_vmin
            if vmax is None:
                vmax = percentile_vmax
    else:
        # Default behavior: vmax from 95th percentile if not provided
        if vmax is None:
            if np.any(~np.isnan(rate_map_display)):
                valid_rates = rate_map_display[~np.isnan(rate_map_display)]
                vmax = np.percentile(np.asarray(valid_rates, dtype=np.float64), 95) if len(valid_rates) > 0 else 1
            else:
                vmax = 1
        
        # Set vmin to 0 if not provided
        if vmin is None:
            vmin = 0
    
    # Convert extent to cm - swapped for rotation: y range on x-axis, x range on y-axis
    extent_cm = [y_min / px_per_cm, y_max / px_per_cm, 
                 x_min / px_per_cm, x_max / px_per_cm]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot using seaborn heatmap for better SVG rendering
    im = sns.heatmap(rate_map_display, ax=ax, cmap=cmap, 
                     vmin=vmin, vmax=vmax,
                     cbar_kws={'label': cbar_label, 'shrink': cbar_shrink},
                     xticklabels=False, yticklabels=False,
                     square=True)
    
    # Get the colorbar to modify ticks
    cbar = im.collections[0].colorbar
    
    # Set colorbar ticks to min/max only if requested
    if cbar_min_max_only:
        cbar.set_ticks([vmin, vmax])
    
    # Add black border around the heatmap
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)
    
    # Manually set the extent since seaborn doesn't use extent parameter
    ax.set_xlim(0, rate_map_display.shape[1])
    ax.set_ylim(0, rate_map_display.shape[0])
    
    # Create custom tick positions for extent in cm
    n_xticks = 2  # min and max
    n_yticks = 2
    xtick_positions = np.linspace(0, rate_map_display.shape[1], n_xticks)
    ytick_positions = np.linspace(0, rate_map_display.shape[0], n_yticks)
    xtick_labels = [f'{extent_cm[0]:.1f}', f'{extent_cm[1]:.1f}']
    ytick_labels = [f'{extent_cm[2]:.1f}', f'{extent_cm[3]:.1f}']
    
    # Set axis labels (unless hidden)
    if not hide_axis_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    ax.set_aspect('equal', adjustable='box')  # Force equal aspect ratio

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)
    
    # Set min/max ticks only if requested (overrides default)
    if min_max_ticks_only:
        ax.set_xticks(xtick_positions)
        ax.set_yticks(ytick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticklabels(ytick_labels)
    
    # Hide all ticks and tick labels if requested
    if hide_all_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    if title is None:
        ax.set_title(f'{mouse_id} {session_id} - Cluster {cluster_id}\nRate Map (SI={spatial_info:.3f} bits/spike)', pad=20)
    else:
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    
    # Save if requested
    if save_plot:
        if output_dir is None:
            output_dir = "."
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        plot_filename = f"{output_dir}/{mouse_id}_{session_id}_cluster{cluster_id}_ratemap.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {plot_filename}")
    
    return fig, {
        'spatial_info': spatial_info,
        'rate_map': rate_map_viz,
        'rate_map_display': rate_map_display
    }