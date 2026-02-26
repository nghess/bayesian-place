"""
SessionData class for storing and manipulating preprocessed neural and behavioral data.

This module provides a standalone class to encapsulate preprocessed neural data from a single recording session,
including spike times, cluster information, and simple population activity binning tools.

Code by Nate Gonzales-Hess, August 2025.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, Union, List
from scipy.stats import zscore
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import cv2
import os
import operator
import copy

from sldata.behavior_utils import get_file_paths


class SessionData:
    """
    A class to store and filter Kilosort and Bonsai data for a single recording session.
    
    Attributes:
    -----------
    mouse_id : str
        Mouse identifier
    session_id : str  
        Session identifier
    clusters : dict
        Dictionary with sequential indices (0, 1, 2, ...) as keys, each containing:
        - 'cluster_id': original Kilosort cluster ID
        - 'best_channel': channel with maximum waveform amplitude
        - 'spike_times': spike times in milliseconds
        - 'waveform_template': mean waveform template
        - 'n_spikes': number of spikes
    sampling_rate : float
        Neural data sampling rate in Hz
    n_clusters : int
        Number of clusters loaded (after filtering)
    sniff : np.ndarray
        Sniff signal data (if available)
    sniff_times : np.ndarray
        Peak times of sniff signal in milliseconds (if available)
    """
    
    def __init__(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\",
                 sampling_rate: float = 30000.0, min_spikes: int = 250,
                 exclude_noise: bool = True, verbose: bool = True):
        """
        Initialize SessionData object by loading preprocessed neural data.

        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        base_path : str
            Base directory path for preprocessed data files
        sampling_rate : float
            Neural data sampling rate in Hz
        min_spikes : int
            Minimum number of spikes required to include a cluster
        exclude_noise : bool
            Whether to exclude clusters labeled as 'noise' (default: True)
        verbose : bool
            Whether to print loading information
        """
        self.mouse_id = mouse_id
        self.session_id = session_id
        self.experiment = experiment
        self.base_path = base_path
        self.video_path = get_file_paths(f"{base_path}/{experiment}", 'avi', f"{mouse_id}_{session_id}", session_type='', print_paths=False, verbose=False)[0]
        self.video = cv2.VideoCapture()
        self.sampling_rate = sampling_rate
        self.min_spikes = min_spikes
        self.exclude_noise = exclude_noise
        self.verbose = verbose

        # Initialize containers
        self.clusters = {}
        self.raw_data = {}
        self.cluster_labels = {}  # Maps cluster_id -> label
        self.n_clusters = 0
        self.events = pd.DataFrame()
        self.duration = np.int32()

        # Initialize signal traces
        self.sniff = np.array([])
        self.sniff_times = np.array([])
        self.iti = np.array([])
        self.reward = np.array([])
        

        # Load the data
        self._load_ephys_data()
        self._load_events_data()
        self._load_video()
        self._load_cluster_labels()
        self._process_clusters()
        self._process_signals()
        self.find_sniff_peaks(prominence=1000, distance=50)

        if 'nose.x' in self.events:
            self.position_x = 'nose.x'
            self.position_y = 'nose.y'
        else:
            self.position_x = 'bonsai_centroid_x'
            self.position_y = 'bonsai_centroid_y'
        
        
        if verbose:
            print(f"Loaded {self.n_clusters} clusters for {mouse_id}_{session_id}")
            if len(self.sniff) > 0:
                print(f"Loaded sniff data: {len(self.sniff)} samples")
            if len(self.sniff_times) > 0:
                print(f"Found {len(self.sniff_times)} sniff events")
    
    def _get_file_path(self, target_filename: str) -> Optional[Path]:
        """
        Find the path to a target file by searching flexibly within the experiment directory.
        
        Parameters:
        -----------
        target_filename : str
            Target filename to search for (e.g., 'sniff.npy', 'spike_times.npy')
            
        Returns:
        --------
        file_path : Path or None
            Path to the target file, or None if not found
        """
        # Search from experiment directory
        search_dir = Path(self.base_path) / self.experiment
        
        if not search_dir.exists():
            if self.verbose:
                print(f"Experiment directory does not exist: {search_dir}")
            return None
        
        # Search for the target file within the experiment directory
        search_pattern = f"**/{target_filename}"
        matching_files = list(search_dir.glob(search_pattern))
        
        # Filter for files that have mouse_id/session_id in their path
        # Use Path.parts to avoid OS-specific separator issues
        for file_path in matching_files:
            path_parts = file_path.parts
            # Check if both mouse_id and session_id appear in the path parts
            if self.mouse_id in path_parts and self.session_id in path_parts:
                return file_path
        
        if self.verbose:
            print(f"Warning: {target_filename} not found for {self.mouse_id}/{self.session_id} in {search_dir}")
        return None
    
    def _load_video(self):
        # Load video
        self.video = cv2.VideoCapture(f"{self.video_path}")

        #  Video properties
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get offset between video and event data start
        self.video_offset = self.total_frames - len(self.events)

        print(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")


    def _load_ephys_data(self):
        """Load preprocessed neural data files."""
        try:
            self.raw_data = self._load_kilosort_files(
                self.mouse_id, 
                self.session_id,
                self.experiment, 
                base_path=self.base_path,
                files_needed=['spike_times', 'spike_templates', 'templates', 'sniff']
            )
            
            if self.verbose:
                print(f"Loaded data: {list(self.raw_data.keys())}")
                
        except Exception as e:
            raise ValueError(f"Failed to load preprocessed neural data: {e}")
    
    def _load_kilosort_files(self, mouse_id: str, session_id: str, experiment: str, base_path: str = "S:\\", 
                            files_needed: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load preprocessed spike sorting data for a given mouse and session using flexible path finding.
        
        Parameters:
        -----------
        mouse_id : str
            Mouse identifier (e.g., '7001')
        session_id : str
            Session identifier (e.g., 'v1')
        experiment : str
            Experiment name
        base_path : str
            Base data directory path (default: 'S:\\')
        files_needed : list of str, optional
            Specific files to load. If None, loads all standard files.
            Options: ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                     'whitening_mat_inv', 'sniff', 'events']
            
        Returns:
        --------
        data : dict
            Dictionary containing loaded data arrays/dataframes
        """
        # Default files to load
        if files_needed is None:
            files_needed = ['spike_times', 'spike_templates', 'templates', 'amplitudes', 
                           'whitening_mat_inv', 'sniff', 'events']
        
        data = {}
        
        # Define filename mapping
        target_files = {
            'spike_times': 'spike_times.npy',
            'spike_templates': 'spike_templates.npy', 
            'templates': 'templates.npy',
            'amplitudes': 'amplitudes.npy',
            'whitening_mat_inv': 'whitening_mat_inv.npy',
            'sniff': 'sniff.npy',
            'events': 'events.csv'
        }
        
        # Load each requested file using flexible path finding
        for key in files_needed:
            if key in target_files:
                target_filename = target_files[key]
                file_path = self._get_file_path(target_filename)
                
                if file_path is not None:
                    try:
                        data[key] = np.load(file_path)
                        if self.verbose:
                            print(f"Successfully loaded {key} from {file_path}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading {key} from {file_path}: {e}")
                else:
                    if self.verbose:
                        print(f"Warning: {target_filename} not found")
        
        return data

    def _load_events_data(self) -> pd.DataFrame:
        """
        Load behavioral events data for a given mouse and session using flexible path finding.

        Returns:
        --------
        events : pd.DataFrame
            Events dataframe with behavioral data from Bonsai
        """
        events_path = self._get_file_path("events.csv")

        if events_path is None:
            if self.verbose:
                print("Warning: events.csv not found")
            return

        try:
            self.events = pd.read_csv(events_path)
            self.duration = (self.events['timestamp_ms'].iloc[-1] - self.events['timestamp_ms'].iloc[0])
            if self.verbose:
                print(f"Successfully loaded events from {events_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading events from {events_path}: {e}")

    def _load_cluster_labels(self):
        """
        Load cluster labels from cluster_group.tsv file.

        This file contains cluster quality labels (good, mua, noise).
        Looks for label column with header 'group' first, then 'KSLabel' as fallback.
        Cluster ID is always assumed to be in the first column.
        """
        cluster_group_path = self._get_file_path("cluster_group.tsv")

        if cluster_group_path is None:
            if self.verbose:
                print("Warning: cluster_group.tsv not found. All clusters will be labeled as 'unknown'.")
            return

        try:
            # Read the TSV file
            with open(cluster_group_path, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                if self.verbose:
                    print("Warning: cluster_group.tsv is empty or has no data rows")
                return

            # Parse header to find the label column
            header = lines[0].strip().split('\t')
            label_col_idx = None

            # First try to find 'group' column
            for idx, col_name in enumerate(header):
                if col_name.lower() == 'group':
                    label_col_idx = idx
                    break

            # If 'group' not found, try 'KSLabel'
            if label_col_idx is None:
                for idx, col_name in enumerate(header):
                    if col_name.lower() == 'kslabel':
                        label_col_idx = idx
                        break

            # If still not found, use column 1 as default
            if label_col_idx is None:
                label_col_idx = 1
                if self.verbose:
                    print("Warning: Neither 'group' nor 'KSLabel' column found. Using column 1 for labels.")

            # Parse data lines
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) > label_col_idx:
                    try:
                        cluster_id = int(parts[0])  # Column 0: cluster_id (always)
                        label = parts[label_col_idx].strip()  # Label column (found from header)
                        self.cluster_labels[cluster_id] = label
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue

            if self.verbose:
                n_good = sum(1 for label in self.cluster_labels.values() if label == 'good')
                n_mua = sum(1 for label in self.cluster_labels.values() if label == 'mua')
                n_noise = sum(1 for label in self.cluster_labels.values() if label == 'noise')
                print(f"Loaded cluster labels: {n_good} good, {n_mua} mua, {n_noise} noise")

        except Exception as e:
            if self.verbose:
                print(f"Error loading cluster labels from {cluster_group_path}: {e}")
    
    def _get_cluster_spike_times(self, spike_times: np.ndarray, spike_templates: np.ndarray, 
                                cluster_id: int, position_times: Optional[np.ndarray] = None,
                                sampling_rate: float = 30000.0, 
                                time_filter: bool = True) -> np.ndarray:
        """
        Extract spike times for a specific cluster with optional time filtering.
        
        Parameters:
        -----------
        spike_times : np.ndarray
            All spike times from kilosort (in samples)
        spike_templates : np.ndarray
            Cluster assignments for each spike
        cluster_id : int
            Target cluster ID
        position_times : np.ndarray, optional
            Position timestamps for filtering (in ms)
        sampling_rate : float
            Sampling rate in Hz (default: 30000)
        time_filter : bool
            Whether to filter spikes to position recording period
            
        Returns:
        --------
        cluster_spike_times : np.ndarray
            Spike times for the cluster (in milliseconds)
        """
        # Get spike indices for this cluster
        cluster_spikes = np.where(spike_templates == cluster_id)[0]
        
        # Extract spike times and convert to milliseconds
        cluster_spike_times = spike_times[cluster_spikes] / (sampling_rate / 1000.0)
        
        # Filter to position recording period if requested
        if time_filter and position_times is not None:
            time_mask = ((cluster_spike_times >= position_times[0]) & 
                        (cluster_spike_times <= position_times[-1]))
            cluster_spike_times = cluster_spike_times[time_mask]
        
        return cluster_spike_times
    
    def _process_clusters(self):
        """Process clusters and extract relevant information."""
        if 'spike_times' not in self.raw_data or 'spike_templates' not in self.raw_data:
            raise ValueError("Missing required spike data")

        spike_times = self.raw_data['spike_times']
        spike_templates = self.raw_data['spike_templates']
        templates = self.raw_data.get('templates')

        # Get unique cluster IDs
        unique_clusters = np.unique(spike_templates)

        # Collect valid clusters first (those meeting min_spikes threshold and not noise)
        valid_clusters = []
        n_excluded_noise = 0
        n_excluded_spikes = 0

        for cluster_id in unique_clusters:
            # Get label for this cluster
            label = self.cluster_labels.get(cluster_id, 'unknown')

            # Skip noise clusters if exclude_noise is True
            if self.exclude_noise and label == 'noise':
                n_excluded_noise += 1
                continue

            # Get spike times for this cluster using internal method
            cluster_spike_times = self._get_cluster_spike_times(
                spike_times, spike_templates, cluster_id,
                sampling_rate=self.sampling_rate, time_filter=False
            )

            # Apply minimum spike threshold
            if len(cluster_spike_times) < self.min_spikes:
                n_excluded_spikes += 1
                continue

            # Get best channel and waveform template
            best_channel = None
            waveform_template = None

            if templates is not None and cluster_id < len(templates):
                template = templates[cluster_id]
                # Best channel is the one with maximum absolute amplitude
                best_channel = np.argmax(np.max(np.abs(template), axis=0))
                # Extract waveform for best channel
                waveform_template = template[:, best_channel]

            # Store cluster information with original cluster_id and label
            cluster_info = {
                'cluster_id': int(cluster_id),  # Original cluster ID from Kilosort
                'label': label,  # Cluster quality label (good, mua, noise, unknown)
                'best_channel': best_channel,
                'spike_times': cluster_spike_times,
                'waveform_template': waveform_template,
                'n_spikes': len(cluster_spike_times)
            }

            valid_clusters.append(cluster_info)

        # Now store clusters with sequential indices (no gaps)
        self.clusters = {}
        for i, cluster_info in enumerate(valid_clusters):
            self.clusters[i] = cluster_info

        self.n_clusters = len(self.clusters)

        if self.verbose:
            if n_excluded_noise > 0:
                print(f"Excluded {n_excluded_noise} noise clusters")
            if n_excluded_spikes > 0:
                print(f"Excluded {n_excluded_spikes} clusters with < {self.min_spikes} spikes")
    
    def _process_signals(self):
        """Process signal traces (sniff, etc.)."""
        # Load sniff data if available
        if 'sniff' in self.raw_data:
            self.sniff = self.raw_data['sniff']
        else:
            self.sniff = np.array([])
            if self.verbose:
                print("No sniff data found")
    
    def _compute_percentile_limits(self, data, lower_percentile=5, upper_percentile=95):
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

    def _upsample_map(self, data, zoom_factor=5):
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

    def get_cluster_ids(self) -> list:
        """
        Get list of cluster indices (sequential 0-based indices).

        Returns:
        --------
        cluster_indices : list
            List of cluster indices (0, 1, 2, ...)
        """
        return list(self.clusters.keys())
    
    def get_original_cluster_ids(self) -> list:
        """
        Get list of original Kilosort cluster IDs.

        Returns:
        --------
        original_cluster_ids : list
            List of original cluster IDs from Kilosort data
        """
        return [cluster_info['cluster_id'] for cluster_info in self.clusters.values()]

    def get_clusters_by_label(self, label: str) -> list:
        """
        Get list of cluster indices with a specific label.

        Parameters:
        -----------
        label : str
            Cluster quality label to filter by (e.g., 'good', 'mua', 'noise')

        Returns:
        --------
        cluster_indices : list
            List of cluster indices matching the specified label

        Examples:
        ---------
        >>> # Get only 'good' clusters
        >>> good_clusters = session.get_clusters_by_label('good')
        >>> print(f"Found {len(good_clusters)} good clusters")
        """
        return [i for i, cluster in self.clusters.items() if cluster['label'] == label]

    def get_cluster_label_counts(self) -> dict:
        """
        Get count of clusters for each label.

        Returns:
        --------
        label_counts : dict
            Dictionary mapping label -> count

        Examples:
        ---------
        >>> counts = session.get_cluster_label_counts()
        >>> print(f"Good: {counts.get('good', 0)}, MUA: {counts.get('mua', 0)}")
        """
        label_counts = {}
        for cluster in self.clusters.values():
            label = cluster['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def get_cluster_info(self, cluster_index: int) -> Dict:
        """
        Get information for a specific cluster.
        
        Parameters:
        -----------
        cluster_index : int
            Cluster index (0-based sequential index, not original Kilosort ID)
            
        Returns:
        --------
        cluster_info : dict
            Dictionary with cluster information
        """
        if cluster_index not in self.clusters:
            raise ValueError(f"Cluster index {cluster_index} not found")
        
        return self.clusters[cluster_index].copy()
    
    def get_spike_times(self, cluster_index: int, 
                       start_time: float = 0, 
                       end_time: float = np.inf) -> np.ndarray:
        """
        Get spike times for a cluster within a time window.
        
        Parameters:
        -----------
        cluster_index : int
            Cluster index (0-based sequential index, not original Kilosort ID)
        start_time : float
            Start time in milliseconds
        end_time : float
            End time in milliseconds
            
        Returns:
        --------
        spike_times : np.ndarray
            Filtered spike times
        """
        if cluster_index not in self.clusters:
            raise ValueError(f"Cluster index {cluster_index} not found")
        
        spike_times = self.clusters[cluster_index]['spike_times']
        
        # Apply time filter
        time_mask = (spike_times >= start_time) & (spike_times <= end_time)
        return spike_times[time_mask]
    
    def create_population_raster(self, start_time: float = 0, 
                                end_time: float = np.inf,
                                bin_size_ms: float = 50.0,
                                zscore_neurons: bool = True,
                                cluster_ids: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Create time-binned population activity matrix.
        
        Parameters:
        -----------
        start_time : float
            Start time in milliseconds (default: 0)
        end_time : float
            End time in milliseconds (default: inf, uses session duration)
        bin_size_ms : float
            Time bin size in milliseconds (default: 50)
        zscore_neurons : bool
            Whether to z-score each neuron's activity (default: True)
        cluster_ids : list, optional
            Specific cluster indices to include. If None, uses all clusters.
            
        Returns:
        --------
        population_matrix : np.ndarray
            Population activity matrix [neurons x time_bins]
        time_bins : np.ndarray
            Time bin centers in milliseconds
        included_clusters : list
            List of cluster indices included in the matrix
        """
        # Determine which clusters to include
        if cluster_ids is None:
            included_clusters = self.get_cluster_ids()
        else:
            included_clusters = [cid for cid in cluster_ids if cid in self.clusters]
        
        if len(included_clusters) == 0:
            raise ValueError("No valid clusters found")
        
        # Determine time range
        if end_time == np.inf:
            # Find maximum spike time across all clusters
            max_time = 0
            for cluster_index in included_clusters:
                spike_times = self.clusters[cluster_index]['spike_times']
                if len(spike_times) > 0:
                    max_time = max(max_time, np.max(spike_times))
            end_time = max_time
        
        # Create time bins
        time_edges = np.arange(start_time, end_time + bin_size_ms, bin_size_ms)
        time_bins = time_edges[:-1] + bin_size_ms / 2  # Bin centers
        n_time_bins = len(time_bins)
        
        # Initialize population matrix
        n_neurons = len(included_clusters)
        population_matrix = np.zeros((n_neurons, n_time_bins))
        
        # Fill matrix with spike counts
        for i, cluster_index in enumerate(included_clusters):
            spike_times = self.get_spike_times(cluster_index, start_time, end_time)
            
            if len(spike_times) > 0:
                # Count spikes in each time bin
                spike_counts, _ = np.histogram(spike_times, bins=time_edges)
                population_matrix[i, :] = spike_counts
        
        # Convert to firing rates (spikes/second)
        bin_size_sec = bin_size_ms / 1000.0
        population_matrix = population_matrix / bin_size_sec
        
        # Apply z-scoring if requested
        if zscore_neurons:
            for i in range(n_neurons):
                # Only z-score if there's variance
                if np.std(population_matrix[i, :]) > 0:
                    population_matrix[i, :] = zscore(population_matrix[i, :])
        
        if self.verbose:
            print(f"Created population matrix: {n_neurons} neurons x {n_time_bins} time bins")
            print(f"Time range: {start_time:.1f} - {end_time:.1f} ms")
            print(f"Bin size: {bin_size_ms} ms")
            if zscore_neurons:
                print("Applied z-scoring to neurons")
        
        return population_matrix, time_bins, included_clusters
    
    def create_event_colormap(self, time_bins: np.ndarray, event_column: str, 
                             aggregation: str = 'mean', 
                             timestamp_column: str = 'timestamp_ms') -> np.ndarray:
        """
        Create a color map for time bins based on event data.
        
        Parameters:
        -----------
        time_bins : np.ndarray
            Time bin centers in milliseconds (typically from create_population_raster)
        event_column : str
            Column name from events dataframe to use for color mapping
        aggregation : str
            How to aggregate events within each time bin:
            - 'mean': average value (for continuous variables like speed)
            - 'max': maximum value 
            - 'min': minimum value
            - 'any': True if any event in bin is True (for boolean variables)
            - 'all': True if all events in bin are True (for boolean variables)
            - 'sum': sum of values in bin
            - 'count': number of events in bin
        timestamp_column : str
            Column name containing timestamps (default: 'timestamp_ms')
            
        Returns:
        --------
        color_values : np.ndarray
            Array of values for each time bin, same length as time_bins
            
        Examples:
        ---------
        # Get population raster
        pop_matrix, time_bins, clusters = session.create_population_raster()
        
        # Create color map based on reward state (boolean)
        reward_colors = session.create_event_colormap(time_bins, 'reward_state', 'any')
        
        # Create color map based on average speed
        speed_colors = session.create_event_colormap(time_bins, 'speed', 'mean')
        """
        if self.events.empty:
            if self.verbose:
                print("Warning: No events data available")
            return np.zeros(len(time_bins))
        
        if event_column not in self.events.columns:
            raise ValueError(f"Column '{event_column}' not found in events data. "
                           f"Available columns: {list(self.events.columns)}")
        
        if timestamp_column not in self.events.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in events data")
        
        # Convert timestamps to milliseconds if needed
        if pd.api.types.is_datetime64_any_dtype(self.events[timestamp_column]):
            # Convert datetime to milliseconds from start
            start_time = self.events[timestamp_column].iloc[0]
            event_times_ms = (self.events[timestamp_column] - start_time).dt.total_seconds() * 1000
        else:
            # Assume already in milliseconds - convert to numeric to handle any string values
            event_times_ms = pd.to_numeric(self.events[timestamp_column], errors='coerce')
        
        # Remove any NaN values that might result from conversion
        valid_mask = ~pd.isna(event_times_ms)
        if not valid_mask.all():
            if self.verbose:
                print(f"Warning: {(~valid_mask).sum()} invalid timestamps found and will be ignored")
            event_times_ms = event_times_ms[valid_mask]
            # Also filter the events dataframe to match
            events_subset = self.events[valid_mask]
        else:
            events_subset = self.events
        
        # Calculate time bin edges from bin centers
        bin_size_ms = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 50.0
        time_edges = time_bins - bin_size_ms / 2
        time_edges = np.append(time_edges, time_edges[-1] + bin_size_ms)
        
        # Initialize color values array
        color_values = np.zeros(len(time_bins))
        
        # For each time bin, find events that fall within it and aggregate
        for i, (start_edge, end_edge) in enumerate(zip(time_edges[:-1], time_edges[1:])):
            # Find events within this time bin
            mask = (event_times_ms >= start_edge) & (event_times_ms < end_edge)
            bin_events = events_subset.loc[events_subset.index[mask], event_column]
            
            if len(bin_events) == 0:
                # No events in this bin
                color_values[i] = 0 if aggregation in ['mean', 'max', 'min', 'sum'] else False
                continue
            
            # Apply aggregation function
            if aggregation == 'mean':
                color_values[i] = bin_events.mean()
            elif aggregation == 'max':
                color_values[i] = bin_events.max()
            elif aggregation == 'min':
                color_values[i] = bin_events.min()
            elif aggregation == 'any':
                color_values[i] = bin_events.any()
            elif aggregation == 'all':
                color_values[i] = bin_events.all()
            elif aggregation == 'sum':
                color_values[i] = bin_events.sum()
            elif aggregation == 'count':
                color_values[i] = len(bin_events)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}. "
                               f"Supported: 'mean', 'max', 'min', 'any', 'all', 'sum', 'count'")
        
        if self.verbose:
            print(f"Created color map from '{event_column}' using '{aggregation}' aggregation")
            print(f"Value range: {np.min(color_values):.3f} to {np.max(color_values):.3f}")
        
        return color_values
    
    def find_sniff_peaks(self, height: Optional[float] = None, 
                        prominence: Optional[float] = None, 
                        distance: Optional[int] = None,
                        width: Optional[Union[int, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Find peaks in the sniff signal and store peak times in sniff_times attribute.
        
        Parameters:
        -----------
        height : float, optional
            Required minimum height of peaks. If None, uses default scipy behavior.
        prominence : float, optional
            Required prominence of peaks. If None, uses default scipy behavior.
        distance : int, optional
            Required minimal horizontal distance between peaks in samples.
        width : int or tuple of ints, optional
            Required width of peaks in samples.
            
        Returns:
        --------
        peak_times : np.ndarray
            Array of peak times in milliseconds
            
        Examples:
        ---------
        # Find all peaks with default parameters
        peak_times = session.find_sniff_peaks()
        
        # Find prominent peaks with minimum distance
        peak_times = session.find_sniff_peaks(prominence=0.5, distance=100)
        
        # Find peaks with specific height threshold
        peak_times = session.find_sniff_peaks(height=0.2)
        """
        if len(self.sniff) == 0:
            if self.verbose:
                print("Warning: No sniff data available for peak detection")
            self.sniff_times = np.array([])
            return self.sniff_times
        
        # Determine effective sampling rate after any processing
        effective_sampling_rate = self.sampling_rate
        
        # Check if sniff signal needs filtering and decimation
        if len(self.sniff) > self.duration * 1.5:  # Signal is at higher sample rate than milliseconds
            if self.verbose:
                print("Filtering and decimating sniff signal from 30kHz to 1kHz...")
            
            # Apply 60Hz notch filter
            b, a = signal.iirnotch(60, 10, 30_000)
            self.sniff = signal.filtfilt(b, a, self.sniff)
            
            # Apply low-pass filter (20Hz cutoff)
            sos = signal.butter(3, 40, 'low', fs=30_000, output='sos')
            self.sniff = signal.sosfiltfilt(sos, self.sniff)
            
            # Remove DC offset
            self.sniff -= np.median(self.sniff)
            
            # Decimate by factor of 30 (30kHz -> 1kHz)
            self.sniff = signal.decimate(self.sniff, 30)
            self.sniff = self.sniff.astype(np.int16)
            
            # Update effective sampling rate
            effective_sampling_rate = 1000.0
        
        # Build parameters dictionary for find_peaks
        peak_params = {}
        if height is not None:
            peak_params['height'] = height
        if prominence is not None:
            peak_params['prominence'] = prominence
        if distance is not None:
            peak_params['distance'] = distance
        if width is not None:
            peak_params['width'] = width
        
        # Find peaks in the sniff signal
        peak_indices, _ = find_peaks(self.sniff, **peak_params)
        
        # Convert peak indices to times in milliseconds using correct sampling rate
        self.sniff_times = peak_indices * (1000.0 / effective_sampling_rate)
        
        if self.verbose:
            print(f"Found {len(self.sniff_times)} peaks in sniff signal")
            if len(self.sniff_times) > 0:
                print(f"Peak times range: {self.sniff_times[0]:.1f} - {self.sniff_times[-1]:.1f} ms")
        
        return self.sniff_times
        
    def get_sniff_data(self, start_time_ms: float = 0, 
                      end_time_ms: float = np.inf) -> np.ndarray:
        """
        Get sniff data within a time window.
        
        Parameters:
        -----------
        start_time_ms : float
            Start time in milliseconds (default: 0)
        end_time_ms : float
            End time in milliseconds (default: inf, uses full duration)
            
        Returns:
        --------
        sniff_segment : np.ndarray
            Sniff data for the specified time window
        """
        if len(self.sniff) == 0:
            return np.array([])
        
        # Convert time to samples
        start_sample = int(start_time_ms * self.sampling_rate / 1000.0)
        
        if end_time_ms == np.inf:
            end_sample = len(self.sniff)
        else:
            end_sample = int(end_time_ms * self.sampling_rate / 1000.0)
        
        # Apply bounds checking
        start_sample = max(0, start_sample)
        end_sample = min(len(self.sniff), end_sample)
        
        return self.sniff[start_sample:end_sample]
    
    def has_sniff_data(self) -> bool:
        """
        Check if sniff data is available.
        
        Returns:
        --------
        has_sniff : bool
            True if sniff data is loaded, False otherwise
        """
        return len(self.sniff) > 0
    
    def filter_clusters(self, filter_expr: str) -> 'SessionData':
        """
        Filter clusters based on a condition and return a new SessionData object.
        
        Parameters:
        -----------
        filter_expr : str
            Filter expression string (e.g., 'best_channel > 16', 'n_spikes >= 100')
            Supported operators: >, <, >=, <=, ==, !=
            Supported keys: any key in cluster dictionary
            
        Returns:
        --------
        filtered_session : SessionData
            New SessionData object containing only clusters that meet the criteria
            
        Examples:
        ---------
        # Filter for olfactory bulb clusters (channels > 16)
        ob_session = session.filter_clusters('best_channel > 16')
        
        # Filter for highly active clusters
        active_session = session.filter_clusters('n_spikes >= 200')
        
        # Filter for specific channel
        ch5_session = session.filter_clusters('best_channel == 5')
        """
        
        # Parse the filter expression
        # Order operators by length (longest first) to avoid partial matches
        operators = [
            ('>=', operator.ge),
            ('<=', operator.le),
            ('==', operator.eq),
            ('!=', operator.ne),
            ('>', operator.gt),
            ('<', operator.lt)
        ]
        
        # Find the operator in the expression
        op_found = None
        op_symbol = None
        
        for symbol, op_func in operators:
            if symbol in filter_expr:
                op_found = op_func
                op_symbol = symbol
                break
        
        if op_found is None:
            raise ValueError(f"No valid operator found in filter expression: {filter_expr}")
        
        # Split the expression
        parts = filter_expr.split(op_symbol)
        if len(parts) != 2:
            raise ValueError(f"Invalid filter expression format: {filter_expr}")
        
        key = parts[0].strip()
        value_str = parts[1].strip()
        
        # Convert value to appropriate type
        try:
            # Try to convert to number first
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # If not a number, treat as string
            value = value_str.strip('\'"')
        
        # Filter clusters
        filtered_clusters = {}
        filtered_indices = []
        
        for cluster_index, cluster_info in self.clusters.items():
            if key not in cluster_info:
                if self.verbose:
                    print(f"Warning: Key '{key}' not found in cluster {cluster_index}")
                continue
            
            cluster_value = cluster_info[key]
            
            # Handle None values
            if cluster_value is None:
                if op_symbol in ['==', '!=']:
                    if (op_symbol == '==' and value is None) or (op_symbol == '!=' and value is not None):
                        filtered_indices.append(cluster_index)
                continue
            
            # Apply the filter
            try:
                if op_found(cluster_value, value):
                    filtered_indices.append(cluster_index)
            except TypeError as e:
                if self.verbose:
                    print(f"Warning: Cannot compare {cluster_value} with {value} for cluster {cluster_index}: {e}")
                continue
        
        # Create new SessionData object
        filtered_session = copy.copy(self)
        filtered_session.clusters = {}
        
        # Re-index filtered clusters sequentially
        for new_index, old_index in enumerate(filtered_indices):
            filtered_session.clusters[new_index] = copy.deepcopy(self.clusters[old_index])
        
        filtered_session.n_clusters = len(filtered_session.clusters)
        
        # Copy signal data (sniff, etc.)
        filtered_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        filtered_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        
        if self.verbose:
            print(f"Filtered from {self.n_clusters} to {filtered_session.n_clusters} clusters using: {filter_expr}")
        
        return filtered_session
    
    def filter_events(self, filter_expr: str, timestamp_column: str = 'timestamp_ms', 
                     return_false_condition: bool = False) -> Union['SessionData', Tuple['SessionData', 'SessionData']]:
        """
        Filter spike times based on events data and return a new SessionData object.
        
        Parameters:
        -----------
        filter_expr : str
            Filter expression string (e.g., 'reward_state == 1', 'speed > 5.0')
            Supported operators: >, <, >=, <=, ==, !=
            Supported keys: any column name in events dataframe
        timestamp_column : str
            Column name containing timestamps (default: 'timestamp_ms')
        return_false_condition : bool
            If True, also return a SessionData object containing spikes where condition was False
            
        Returns:
        --------
        filtered_session : SessionData or tuple of SessionData
            New SessionData object(s) with spike times filtered based on events condition.
            If return_false_condition=True, returns (true_condition_session, false_condition_session)
            
        Examples:
        ---------
        # Filter spikes during reward periods
        reward_session = session.filter_events('reward_state == 1')
        
        # Filter spikes during high speed periods, also get low speed periods
        high_speed, low_speed = session.filter_events('speed > 5.0', return_false_condition=True)
        """
        
        if self.events.empty:
            raise ValueError("No events data available for filtering")
        
        if timestamp_column not in self.events.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in events data")
        
        # Parse the filter expression (reuse logic from filter_clusters)
        operators = [
            ('>=', operator.ge),
            ('<=', operator.le),
            ('==', operator.eq),
            ('!=', operator.ne),
            ('>', operator.gt),
            ('<', operator.lt)
        ]
        
        op_found = None
        op_symbol = None
        
        for symbol, op_func in operators:
            if symbol in filter_expr:
                op_found = op_func
                op_symbol = symbol
                break
        
        if op_found is None:
            raise ValueError(f"No valid operator found in filter expression: {filter_expr}")
        
        parts = filter_expr.split(op_symbol)
        if len(parts) != 2:
            raise ValueError(f"Invalid filter expression format: {filter_expr}")
        
        key = parts[0].strip()
        value_str = parts[1].strip()
        
        if key not in self.events.columns:
            raise ValueError(f"Column '{key}' not found in events data. "
                           f"Available columns: {list(self.events.columns)}")
        
        # Convert value to appropriate type
        try:
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # Handle boolean values
            value_cleaned = value_str.strip('\'"').lower()
            if value_cleaned in ['true', '1']:
                value = True
            elif value_cleaned in ['false', '0']:
                value = False
            else:
                value = value_str.strip('\'"')
        
        # Convert timestamps to milliseconds if needed
        if pd.api.types.is_datetime64_any_dtype(self.events[timestamp_column]):
            start_time = self.events[timestamp_column].iloc[0]
            event_times_ms = (self.events[timestamp_column] - start_time).dt.total_seconds() * 1000
        else:
            event_times_ms = pd.to_numeric(self.events[timestamp_column], errors='coerce')
        
        # Remove any NaN values
        valid_mask = ~pd.isna(event_times_ms)
        if not valid_mask.all():
            if self.verbose:
                print(f"Warning: {(~valid_mask).sum()} invalid timestamps found and will be ignored")
            event_times_ms = event_times_ms[valid_mask]
            events_subset = self.events[valid_mask].copy()
        else:
            events_subset = self.events.copy()
            
        # Apply the condition to events
        try:
            condition_mask = op_found(events_subset[key], value)
        except TypeError as e:
            raise ValueError(f"Cannot apply condition '{filter_expr}': {e}")
        
        
        def _filter_spikes_by_condition(condition_mask):
            """Helper function to filter spike times based on condition periods."""
            filtered_clusters = {}
            
            # Find time periods where condition is true
            if not condition_mask.any():
                # No periods where condition is true
                for cluster_index, cluster_info in self.clusters.items():
                    new_cluster_info = copy.deepcopy(cluster_info)
                    new_cluster_info['spike_times'] = np.array([])
                    new_cluster_info['n_spikes'] = 0
                    filtered_clusters[cluster_index] = new_cluster_info
                return filtered_clusters
            
            # Get time periods where condition is true
            true_times = event_times_ms[condition_mask].values
            
            # Create time intervals where condition is true
            # For continuous data, we need to group consecutive true periods
            time_intervals = []
            if len(true_times) > 0:
                # Sort times to be safe
                true_times = np.sort(true_times)
                
                # Group consecutive time points (within reasonable gap)
                # Assume events are sampled regularly, use median gap as threshold
                if len(true_times) > 1:
                    gaps = np.diff(true_times)
                    median_gap = np.median(gaps)
                    max_gap = median_gap * 2  # Allow up to 2x median gap
                else:
                    max_gap = 100  # Default 100ms gap
                
                start_time = true_times[0]
                end_time = true_times[0]
                
                for i in range(1, len(true_times)):
                    if true_times[i] - end_time <= max_gap:
                        # Extend current interval
                        end_time = true_times[i]
                    else:
                        # Save current interval and start new one
                        time_intervals.append((start_time, end_time))
                        start_time = true_times[i]
                        end_time = true_times[i]
                
                # Don't forget the last interval
                time_intervals.append((start_time, end_time))
            
            # Filter spikes for each cluster
            for cluster_index, cluster_info in self.clusters.items():
                spike_times = cluster_info['spike_times']
                
                if len(spike_times) == 0 or len(time_intervals) == 0:
                    filtered_spike_times = np.array([])
                else:
                    # Find spikes that fall within any of the true time intervals
                    valid_spikes = []
                    for start_time, end_time in time_intervals:
                        # Include spikes within this interval
                        mask = (spike_times >= start_time) & (spike_times <= end_time)
                        valid_spikes.extend(spike_times[mask])
                    
                    filtered_spike_times = np.array(valid_spikes)
                
                # Create new cluster info with filtered spike times
                new_cluster_info = copy.deepcopy(cluster_info)
                new_cluster_info['spike_times'] = filtered_spike_times
                new_cluster_info['n_spikes'] = len(filtered_spike_times)
                filtered_clusters[cluster_index] = new_cluster_info
            
            return filtered_clusters
        
        # Create filtered session for true condition
        true_session = copy.copy(self)
        true_session.clusters = _filter_spikes_by_condition(condition_mask)
        true_session.n_clusters = len(true_session.clusters)
        
        # Copy signal data
        true_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        true_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        true_session.events = self.events.copy()
        
        if self.verbose:
            total_true_spikes = sum(cluster['n_spikes'] for cluster in true_session.clusters.values())
            total_original_spikes = sum(cluster['n_spikes'] for cluster in self.clusters.values())
            print(f"Filtered spikes with condition '{filter_expr}': {total_true_spikes}/{total_original_spikes} spikes retained")
        
        if not return_false_condition:
            return true_session
        
        # Create filtered session for false condition
        false_session = copy.copy(self)
        false_session.clusters = _filter_spikes_by_condition(~condition_mask)
        false_session.n_clusters = len(false_session.clusters)
        
        # Copy signal data
        false_session.sniff = self.sniff.copy() if len(self.sniff) > 0 else np.array([])
        false_session.sniff_times = self.sniff_times.copy() if len(self.sniff_times) > 0 else np.array([])
        false_session.events = self.events.copy()
        
        if self.verbose:
            total_false_spikes = sum(cluster['n_spikes'] for cluster in false_session.clusters.values())
            print(f"False condition spikes: {total_false_spikes}/{total_original_spikes} spikes")
        
        return true_session, false_session
    
    def get_session_summary(self) -> Dict:
        """
        Get a summary of the session data.
        
        Returns:
        --------
        summary : dict
            Dictionary containing basic session information
        """
        # Basic session info
        summary = {
            'mouse_id': self.mouse_id,
            'session_id': self.session_id,
            'experiment': self.experiment,
            'n_clusters': self.n_clusters,
        }
        
        # Event columns
        if not self.events.empty:
            summary['event_columns'] = list(self.events.columns)
            
            # Calculate session duration from timestamp_ms column
            if 'timestamp_ms' in self.events.columns:
                timestamps = self.events['timestamp_ms']
                duration_ms = timestamps.max() - timestamps.min()
                summary['session_duration_min'] = round(duration_ms / (1000 * 60), 2)
            else:
                summary['session_duration_min'] = 'N/A (no timestamp_ms column)'
        else:
            summary['event_columns'] = []
            summary['session_duration_min'] = 'N/A (no events data)'
        
        return summary
    
    def create_sniff_locked_raster(self, cluster_idx: int, window_ms: float = 500.0, 
                                  n_spikes_for_sorting: int = 3) -> np.ndarray:
        """
        Create sniff-locked raster for a specific cluster.
        
        Parameters:
        -----------
        cluster_idx : int
            Index of the cluster to analyze
        window_ms : float
            Half-window size around sniff events in milliseconds (default: 500ms = ±500ms window)
        n_spikes_for_sorting : int
            Number of spikes to average for sorting by latency (default: 3)
            
        Returns:
        --------
        raster_matrix : np.ndarray
            Sniff-locked raster matrix [n_sniffs x n_timesteps], sorted by spike latency
        """
        if len(self.sniff_times) == 0:
            if self.verbose:
                print("Warning: No sniff events found for sniff-locked analysis")
            return np.array([])
        
        if cluster_idx not in self.clusters:
            raise ValueError(f"Cluster {cluster_idx} not found")
        
        spike_times = self.clusters[cluster_idx]['spike_times']
        
        # Create time bins for the window (1ms resolution)
        time_bins = np.arange(-window_ms, window_ms + 1, 1.0)  # +1 to include endpoint
        n_timesteps = len(time_bins)
        n_sniffs = len(self.sniff_times)
        
        # Initialize raster matrix
        raster_matrix = np.zeros((n_sniffs, n_timesteps))
        latencies = np.full(n_sniffs, np.inf)  # For sorting
        
        # Process each sniff event
        for sniff_idx, sniff_time in enumerate(self.sniff_times):
            # Find spikes within the window around this sniff
            start_time = sniff_time - window_ms
            end_time = sniff_time + window_ms
            
            # Get spikes in window
            window_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
            
            # Convert spike times to relative times (sniff at 0)
            relative_spike_times = window_spikes - sniff_time
            
            # Fill raster for this sniff
            for spike_time in relative_spike_times:
                # Find closest time bin (1ms resolution)
                bin_idx = np.argmin(np.abs(time_bins - spike_time))
                if 0 <= bin_idx < n_timesteps:
                    raster_matrix[sniff_idx, bin_idx] = 1
            
            # Calculate latency for sorting (mean time of next n spikes after sniff)
            post_sniff_spikes = relative_spike_times[relative_spike_times > 0]
            if len(post_sniff_spikes) >= n_spikes_for_sorting:
                latencies[sniff_idx] = np.mean(post_sniff_spikes[:n_spikes_for_sorting])
            elif len(post_sniff_spikes) > 0:
                # Use available spikes if fewer than n_spikes_for_sorting
                latencies[sniff_idx] = np.mean(post_sniff_spikes)
        
        # Sort raster matrix by latency (shortest latency first)
        # Handle inf values (no post-sniff spikes) by putting them at the end
        valid_latencies = latencies[latencies != np.inf]
        invalid_indices = np.where(latencies == np.inf)[0]
        
        if len(valid_latencies) > 0:
            valid_indices = np.where(latencies != np.inf)[0]
            sorted_valid_indices = valid_indices[np.argsort(latencies[valid_indices])]
            # Combine sorted valid indices with invalid indices at the end
            sorted_indices = np.concatenate([sorted_valid_indices, invalid_indices])
        else:
            # All latencies are inf, keep original order
            sorted_indices = np.arange(n_sniffs)
        
        raster_matrix = raster_matrix[sorted_indices]
        
        # Store the result in the cluster data
        self.clusters[cluster_idx]['sniff_locked_raster'] = {
            'raster_matrix': raster_matrix,
            'time_bins': time_bins,
            'window_ms': window_ms,
            'n_spikes_for_sorting': n_spikes_for_sorting,
            'sorted_indices': sorted_indices,
            'latencies': latencies[sorted_indices]
        }
        
        if self.verbose:
            print(f"Created sniff-locked raster for cluster {cluster_idx}: "
                  f"{raster_matrix.shape} (sniffs x timesteps)")
        
        return raster_matrix
    
    def get_sniff_locked_scatter_data(self, cluster_indices: Optional[List[int]] = None, 
                                    window_ms: float = 500.0, 
                                    n_spikes_for_sorting: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sparse scatter plot data for sniff-locked analysis.
        
        Much more memory efficient than full raster matrices - only stores actual spike times.
        Sniff events are sorted by spike latency for better visualization.
        
        Parameters:
        -----------
        cluster_indices : List[int], optional
            List of cluster indices to include. If None, uses all clusters.
        window_ms : float
            Half-window size around sniff events in milliseconds (default: 500ms)
        n_spikes_for_sorting : int
            Number of spikes to average for sorting by latency (default: 3)
            
        Returns:
        --------
        sniff_indices : np.ndarray
            Array of sniff event indices (for y-axis of scatter plot) - sorted by latency
        relative_spike_times : np.ndarray  
            Array of spike times relative to sniff events (for x-axis of scatter plot)
        cluster_ids : np.ndarray
            Array of cluster indices corresponding to each spike (for coloring/grouping)
        """
        if len(self.sniff_times) == 0:
            if self.verbose:
                print("Warning: No sniff events found for scatter plot")
            return np.array([]), np.array([]), np.array([])
        
        if cluster_indices is None:
            cluster_indices = list(self.clusters.keys())
        
        # First, calculate latencies for each sniff event for sorting
        n_sniffs = len(self.sniff_times)
        latencies = np.full(n_sniffs, np.inf)
        
        # Calculate latencies across all specified clusters
        for sniff_idx, sniff_time in enumerate(self.sniff_times):
            all_post_sniff_spikes = []
            
            # Collect post-sniff spikes from all clusters for this sniff
            for cluster_idx in cluster_indices:
                if cluster_idx not in self.clusters:
                    continue
                
                spike_times = self.clusters[cluster_idx]['spike_times']
                start_time = sniff_time - window_ms
                end_time = sniff_time + window_ms
                
                window_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
                relative_times = window_spikes - sniff_time
                post_sniff_spikes = relative_times[relative_times > 0]
                
                all_post_sniff_spikes.extend(post_sniff_spikes)
            
            # Calculate mean latency for sorting
            if len(all_post_sniff_spikes) >= n_spikes_for_sorting:
                all_post_sniff_spikes = sorted(all_post_sniff_spikes)
                latencies[sniff_idx] = np.mean(all_post_sniff_spikes[:n_spikes_for_sorting])
            elif len(all_post_sniff_spikes) > 0:
                latencies[sniff_idx] = np.mean(all_post_sniff_spikes)
        
        # Create sorted order (shortest latency first, inf values at end)
        valid_latencies = latencies[latencies != np.inf]
        invalid_indices = np.where(latencies == np.inf)[0]
        
        if len(valid_latencies) > 0:
            valid_indices = np.where(latencies != np.inf)[0]
            sorted_valid_indices = valid_indices[np.argsort(-latencies[valid_indices])]
            sorted_sniff_order = sorted_valid_indices#np.concatenate([sorted_valid_indices, invalid_indices])
        else:
            sorted_sniff_order = np.arange(n_sniffs)
        
        # Now collect scatter data using sorted sniff order
        all_sniff_indices = []
        all_relative_times = []
        all_cluster_ids = []
        
        # Process each cluster
        for cluster_idx in cluster_indices:
            if cluster_idx not in self.clusters:
                continue
                
            spike_times = self.clusters[cluster_idx]['spike_times']
            
            # Process each sniff event in sorted order
            for new_sniff_idx, original_sniff_idx in enumerate(sorted_sniff_order):
                sniff_time = self.sniff_times[original_sniff_idx]
                
                # Find spikes within the window around this sniff
                start_time = sniff_time - window_ms
                end_time = sniff_time + window_ms
                
                # Get spikes in window
                window_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
                
                if len(window_spikes) > 0:
                    # Convert to relative times (sniff at 0)
                    relative_times = window_spikes - sniff_time
                    
                    # Add to scatter data (use new sorted index)
                    all_sniff_indices.extend([new_sniff_idx] * len(relative_times))
                    all_relative_times.extend(relative_times)
                    all_cluster_ids.extend([cluster_idx] * len(relative_times))
        
        # Convert to numpy arrays
        sniff_indices = np.array(all_sniff_indices)
        relative_spike_times = np.array(all_relative_times)
        cluster_ids = np.array(all_cluster_ids)
        
        if self.verbose:
            print(f"Created scatter data: {len(relative_spike_times)} spikes across "
                  f"{len(cluster_indices)} clusters and {len(self.sniff_times)} sniff events")
        
        return sniff_indices, relative_spike_times, cluster_ids
    
    def plot_sniff_locked_scatter(self, cluster_indices: Optional[List[int]] = None,
                                window_ms: float = 500.0, n_spikes_for_sorting: int = 5,
                                figsize: Tuple[float, float] = (8, 8)):
        """
        Create a scatter plot of sniff-locked spikes.
        
        Parameters:
        -----------
        cluster_indices : List[int], optional
            List of cluster indices to include. If None, uses all clusters.
        window_ms : float
            Half-window size around sniff events in milliseconds
        n_spikes_for_sorting : int
            Number of spikes to average for sorting by latency
        figsize : Tuple[float, float]
            Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return
        
        # Get scatter data
        sniff_indices, relative_times, cluster_ids = self.get_sniff_locked_scatter_data(
            cluster_indices, window_ms, n_spikes_for_sorting)
        
        if len(relative_times) == 0:
            print("No spike data found for plotting")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # If multiple clusters, color by cluster
        if len(np.unique(cluster_ids)) > 1:
            scatter = ax.scatter(relative_times, sniff_indices, c=cluster_ids, 
                               alpha=0.6, s=1, cmap='tab10')
            plt.colorbar(scatter, label='Cluster ID')
        else:
            ax.scatter(relative_times, sniff_indices, alpha=0.05, s=.5, color='k')
        
        # Formatting
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time Relative to Sniff (ms)')
        ax.set_ylabel('Sniff Event Index')
        ax.set_title(f'Sniff-Locked Spike Raster (±{window_ms}ms window)')
        ax.grid(False, alpha=0)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_trajectory(self):
        # Create plot
        fig, ax = plt.subplots(figsize=(self.width, self.height))

    def compute_occupancy_map(self, bin_size=50.7, fixed_arena_bounds=None,
                             min_occupancy=0.1, flip_state=None, return_metadata=True):
        """
        Compute occupancy map from position data.

        Parameters:
        -----------
        bin_size : float
            Spatial bin size in pixels (default: 50.7, which equals 1 cm)
        fixed_arena_bounds : tuple, optional
            Fixed arena bounds (x_min, x_max, y_min, y_max)
        min_occupancy : float
            Minimum occupancy threshold in seconds (default: 0.1)
        flip_state : bool, optional
            Filter by flip_state column. If None, use all data. If True or False,
            only include events where flip_state matches this value.
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
        >>> occupancy_map, metadata = session.compute_occupancy_map()
        >>>
        >>> # Get occupancy map only (without metadata)
        >>> occupancy_map = session.compute_occupancy_map(return_metadata=False)
        >>>
        >>> # Filter by flip_state
        >>> occ_map_flip_true, _ = session.compute_occupancy_map(flip_state=True)
        """
        if self.events.empty:
            raise ValueError("No events data available. Cannot compute occupancy map.")

        if 'timestamp_ms' not in self.events.columns:
            raise ValueError("Events DataFrame must contain 'timestamp_ms' column")

        # Filter events by flip_state if requested
        if flip_state is not None:
            if 'flip_state' not in self.events.columns:
                raise ValueError("Events DataFrame must contain 'flip_state' column for flip_state filtering")
            events_filtered = self.events[self.events['flip_state'] == flip_state].copy()
            if len(events_filtered) == 0:
                raise ValueError(f"No events found with flip_state={flip_state}")
        else:
            events_filtered = self.events

        # Extract position data
        positions_array = events_filtered[[self.position_x, self.position_y]].values
        pos_times_array = events_filtered['timestamp_ms'].values

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

    def plot_occupancy_map(self, bin_size=50.7, min_occupancy=0.1,
                          fixed_arena_bounds=None,
                          flip_state=None,
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
                          figsize=(6, 6),
                          save_plot=False,
                          output_dir=None):
        """
        Create occupancy map with units in cm and 90-degree rotation.

        Parameters:
        -----------
        bin_size : float
            Spatial bin size in pixels (default: 50.7 = 1 cm)
        min_occupancy : float
            Minimum occupancy threshold in seconds
        flip_state : bool, optional
            Filter by flip_state column. If None, use all data.
        use_percentile_limits : bool
            if True, automatically set vmin and vmax to 5th and 95th percentiles (default: False).
            Overrides vmin and vmax if they are None.
        sigma : float
            Gaussian smoothing sigma for visualization (default: 0 = no smoothing).
            Applied after calculation, for display only.
        upsample : bool
            if True, upsample the display map to native pixel resolution (default: False).
            For visualization only, does not affect calculations.
        save_plot : bool
            Whether to save the plot to disk
        output_dir : str
            Directory to save plot (if save_plot=True)

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        data : dict
            Dictionary containing 'occupancy_map' and 'occupancy_map_display'
        """
        px_per_cm = 50.7

        # Filter events by flip_state if requested
        if flip_state is not None:
            if 'flip_state' not in self.events.columns:
                raise ValueError("Events DataFrame must contain 'flip_state' column for flip_state filtering")
            events_filtered = self.events[self.events['flip_state'] == flip_state].copy()
            if len(events_filtered) == 0:
                raise ValueError(f"No events found with flip_state={flip_state}")
        else:
            events_filtered = self.events

        # Extract position data
        positions_array = events_filtered[[self.position_x, self.position_y]].values
        pos_times_array = events_filtered['timestamp_ms'].values

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
            occupancy_map_display = self._upsample_map(occupancy_map_display, zoom_factor=5)

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
                percentile_vmin, percentile_vmax = self._compute_percentile_limits(plot_data)
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
            ax.set_title(f'{self.mouse_id} {self.session_id}\nOccupancy Map', pad=20)
        else:
            ax.set_title(title, pad=20)

        plt.tight_layout()

        # Save if requested
        if save_plot:
            if output_dir is None:
                output_dir = "."
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            plot_filename = f"{output_dir}/{self.mouse_id}_{self.session_id}_occupancy.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved: {plot_filename}")

        return fig, {
            'occupancy_map': occupancy_map,
            'occupancy_map_display': occupancy_map_display
        }

    def compute_rate_map(self, cluster_index, bin_size=50.7, sigma=0, min_occupancy=0.1,
                        fixed_arena_bounds=None, flip_state=None, filter_flip_state=True,
                        return_metadata=True):
        """
        Compute spatial firing rate map for a specific cluster.

        Parameters:
        -----------
        cluster_index : int
            Index of the cluster to analyze
        bin_size : float
            Spatial bin size in pixels (default: 50.7 = 1 cm)
        sigma : float
            Gaussian smoothing sigma (default: 0 = no smoothing)
        min_occupancy : float
            Minimum occupancy threshold in seconds
        fixed_arena_bounds : tuple, optional
            Fixed arena bounds (x_min, x_max, y_min, y_max)
        flip_state : bool, optional
            Filter by flip_state column. If None, use all data. If True or False,
            only include events and spikes where flip_state matches this value.
            Only applies if filter_flip_state=True.
        filter_flip_state : bool
            Whether to apply flip_state filtering (default: True). If False,
            ignores flip_state parameter and uses all data without filtering.
        return_metadata : bool
            Whether to return metadata dict (default: True)

        Returns:
        --------
        rate_map : numpy.ndarray
            2D array of firing rates (spikes/sec) for each spatial bin
        metadata : dict (if return_metadata=True)
            Dictionary containing:
            - 'spatial_info': spatial information in bits/spike
            - 'mean_rate': mean firing rate
            - 'total_spikes': total number of spikes
            - 'total_time': total time in valid bins
            - 'occupancy_map': occupancy map used
            - 'spike_map': spike count map
            - 'valid_bins': boolean mask of valid bins
            - 'x_edges', 'y_edges': bin edges
            - 'x_min', 'x_max', 'y_min', 'y_max': arena bounds

        Examples:
        ---------
        >>> # Compute rate map for cluster 0
        >>> rate_map, metadata = session.compute_rate_map(0)
        >>> print(f"Spatial info: {metadata['spatial_info']:.3f} bits/spike")
        >>>
        >>> # Compute rate map for flip_state=True only
        >>> rate_map_flip, metadata = session.compute_rate_map(0, flip_state=True)
        >>>
        >>> # Get true combined rate map (no filtering)
        >>> rate_map_all, metadata = session.compute_rate_map(0, filter_flip_state=False)
        """
        if cluster_index not in self.clusters:
            raise ValueError(f"Cluster index {cluster_index} not found")

        if self.events.empty:
            raise ValueError("No events data available")

        # Extract all position data first (needed for spike filtering)
        all_positions_array = self.events[[self.position_x, self.position_y]].values
        all_pos_times_array = self.events['timestamp_ms'].values
        spike_times_array = self.clusters[cluster_index]['spike_times']

        # Filter spikes by flip_state if requested
        if filter_flip_state and flip_state is not None:
            if 'flip_state' not in self.events.columns:
                raise ValueError("Events DataFrame must contain 'flip_state' column for flip_state filtering")

            # For each spike, find the nearest event time and check its flip_state
            # Use searchsorted to find the nearest event index for each spike
            spike_event_indices = np.searchsorted(all_pos_times_array, spike_times_array, side='left')
            # Clip to valid range
            spike_event_indices = np.clip(spike_event_indices, 0, len(all_pos_times_array) - 1)

            # Get flip_state values for these events
            spike_flip_states = self.events['flip_state'].values[spike_event_indices]

            # Filter spikes to only those matching the desired flip_state
            spike_mask = spike_flip_states == flip_state
            spike_times_array = spike_times_array[spike_mask]

            # Also filter events for position data
            events_filtered = self.events[self.events['flip_state'] == flip_state].copy()
            if len(events_filtered) == 0:
                raise ValueError(f"No events found with flip_state={flip_state}")
        else:
            events_filtered = self.events

        # Extract position data from filtered events
        positions_array = events_filtered[[self.position_x, self.position_y]].values
        pos_times_array = events_filtered['timestamp_ms'].values

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
        x_bins = int(np.ceil((x_max - x_min) / bin_size))
        y_bins = int(np.ceil((y_max - y_min) / bin_size))
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)

        # Calculate occupancy map
        dt = np.median(np.asarray(np.diff(pos_times_array), dtype=np.float64)) / 1000.0
        occupancy_map, _, _ = np.histogram2d(
            positions_array[:, 0], positions_array[:, 1],
            bins=[x_edges, y_edges]
        )
        occupancy_map = occupancy_map * dt

        # Filter spikes by time range
        valid_spikes = (spike_times_array >= pos_times_array[0]) & (spike_times_array <= pos_times_array[-1])
        filtered_spike_times = spike_times_array[valid_spikes]

        # Calculate spike positions and spike map
        if len(filtered_spike_times) > 0:
            spike_x = np.interp(filtered_spike_times, np.asarray(pos_times_array, dtype=np.float64),
                               positions_array[:, 0])
            spike_y = np.interp(filtered_spike_times, np.asarray(pos_times_array, dtype=np.float64),
                               positions_array[:, 1])
            spike_map, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
        else:
            spike_map = np.zeros_like(occupancy_map)

        # Calculate rate map
        valid_bins = occupancy_map >= min_occupancy
        rate_map = np.zeros_like(occupancy_map)
        rate_map[valid_bins] = spike_map[valid_bins] / occupancy_map[valid_bins]

        # Calculate spatial information BEFORE smoothing
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

        # Apply smoothing if requested (after spatial info calculation)
        if sigma > 0:
            rate_map = gaussian_filter(rate_map, sigma=sigma)

        if return_metadata:
            metadata = {
                'spatial_info': spatial_info,
                'mean_rate': mean_rate,
                'total_spikes': total_spikes,
                'total_time': total_time,
                'occupancy_map': occupancy_map,
                'spike_map': spike_map,
                'valid_bins': valid_bins,
                'x_edges': x_edges,
                'y_edges': y_edges,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'bin_size': bin_size
            }
            return rate_map, metadata
        else:
            return rate_map

    def plot_rate_map(self, cluster_index, bin_size=50.7, sigma=0, min_occupancy=0.1,
                     fixed_arena_bounds=None,
                     flip_state=None,
                     filter_flip_state=True,
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
                     figsize=(6, 6),
                     save_plot=False,
                     output_dir=None):
        """
        Create rate map plot with units in cm and 90-degree rotation.

        Parameters:
        -----------
        cluster_index : int
            Index of the cluster to analyze
        bin_size : float
            Spatial bin size in pixels (default: 50.7 = 1 cm)
        sigma : float
            Gaussian smoothing sigma for visualization (default: 0 = no smoothing).
            Applied after rate calculation, before display.
        min_occupancy : float
            Minimum occupancy threshold in seconds
        flip_state : bool, optional
            Filter by flip_state column. If None, use all data.
            Only applies if filter_flip_state=True.
        filter_flip_state : bool
            Whether to apply flip_state filtering (default: True). If False,
            ignores flip_state parameter and uses all data without filtering.
        use_percentile_limits : bool
            if True, automatically set vmin and vmax to 5th and 95th percentiles (default: False).
        upsample : bool
            if True, upsample the display map to native pixel resolution (default: False).
            For visualization only, does not affect spatial information calculation.
        save_plot : bool
            Whether to save the plot to disk
        output_dir : str
            Directory to save plot (if save_plot=True)

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object
        data : dict
            Dictionary containing 'spatial_info', 'rate_map', and 'rate_map_display'
        """
        px_per_cm = 50.7

        # Get cluster ID for title
        original_cluster_id = self.clusters[cluster_index]['cluster_id']

        # Extract all position data first (needed for spike filtering)
        all_positions_array = self.events[[self.position_x, self.position_y]].values
        all_pos_times_array = self.events['timestamp_ms'].values
        spike_times_array = self.clusters[cluster_index]['spike_times']

        # Filter spikes by flip_state if requested
        if filter_flip_state and flip_state is not None:
            if 'flip_state' not in self.events.columns:
                raise ValueError("Events DataFrame must contain 'flip_state' column for flip_state filtering")

            # For each spike, find the nearest event time and check its flip_state
            # Use searchsorted to find the nearest event index for each spike
            spike_event_indices = np.searchsorted(all_pos_times_array, spike_times_array, side='left')
            # Clip to valid range
            spike_event_indices = np.clip(spike_event_indices, 0, len(all_pos_times_array) - 1)

            # Get flip_state values for these events
            spike_flip_states = self.events['flip_state'].values[spike_event_indices]

            # Filter spikes to only those matching the desired flip_state
            spike_mask = spike_flip_states == flip_state
            spike_times_array = spike_times_array[spike_mask]

            # Also filter events for position data
            events_filtered = self.events[self.events['flip_state'] == flip_state].copy()
            if len(events_filtered) == 0:
                raise ValueError(f"No events found with flip_state={flip_state}")
        else:
            events_filtered = self.events

        # Extract position data from filtered events
        positions_array = events_filtered[[self.position_x, self.position_y]].values
        pos_times_array = events_filtered['timestamp_ms'].values

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
        if sigma > 0:
            rate_map = gaussian_filter(rate_map, sigma=sigma)

        # Create visualization array
        rate_map_viz = np.copy(rate_map)
        rate_map_viz[~valid_bins] = np.nan

        # For display: transpose to get correct orientation
        rate_map_display = rate_map_viz.T

        # Apply upsampling for visualization if requested
        if upsample:
            rate_map_display = self._upsample_map(rate_map_display, zoom_factor=5)

        # Use percentile limits if requested (and vmin/vmax not explicitly set)
        if use_percentile_limits:
            if vmin is None or vmax is None:
                percentile_vmin, percentile_vmax = self._compute_percentile_limits(rate_map_display)
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

        # Convert extent to cm - swapped for rotation
        extent_cm = [y_min / px_per_cm, y_max / px_per_cm,
                     x_min / px_per_cm, x_max / px_per_cm]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot using seaborn heatmap
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

        # Manually set the extent
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

        ax.set_aspect('equal', adjustable='box')

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
            ax.set_title(f'{self.mouse_id} {self.session_id} - Cluster {original_cluster_id}\nRate Map (SI={spatial_info:.3f} bits/spike)', pad=20)
        else:
            ax.set_title(title, pad=20)

        plt.tight_layout()

        # Save if requested
        if save_plot:
            if output_dir is None:
                output_dir = "."
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            plot_filename = f"{output_dir}/{self.mouse_id}_{self.session_id}_cluster{original_cluster_id}_ratemap.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Saved: {plot_filename}")

        return fig, {
            'spatial_info': spatial_info,
            'rate_map': rate_map_viz,
            'rate_map_display': rate_map_display
        }

    def __repr__(self) -> str:
        """String representation of SessionData object."""
        return f"SessionData({self.mouse_id}_{self.session_id}, {self.n_clusters} clusters)"