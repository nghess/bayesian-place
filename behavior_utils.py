import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

"""
Helper Functions Below
"""

# Get the name of the last two directories in data path
def get_savedirs(path):
    path = str(path)
    parts = path.split(os.path.sep)
    return os.path.sep.join(parts[-3:-1])

def natural_sort_key(path):
    """
    Create a key for sorting strings with numbers naturally.
    Converts "1", "2", "10" into proper numeric order instead of lexicographic order.
    """
    # Convert path to string if it's a Path object
    path_str = str(path)
    # Split string into chunks of numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', path_str)]

# Get all files matching extension and keyword in a directory and its subdirectories
def get_file_paths(directory: str = '', extension: str = '', keyword: str = '', session_type: str = '', print_paths=False, print_n=np.inf, verbose=True) -> list:
    paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name]
    # Sort paths using natural sorting
    paths = sorted(paths, key=natural_sort_key)
    if verbose:
        print(f'Found {len(paths)} {keyword}.{extension} files')
    if print_paths:
            show_paths(paths, print_n)
    return paths

# Print collected paths and their indices
def show_paths(data_paths, print_n=np.inf):
    for ii, path in enumerate(data_paths[:min(print_n, len(data_paths))]):
        print(f"{ii} {path}")

# Paths are valid only their final two parent directories (mouse and session ids) match the reference paths
def filter_paths(paths_to_filter, paths_to_reference):
    filtered_paths = []
    for path in paths_to_filter:
        for ref_path in paths_to_reference:
            if (path.parents[1].name == ref_path.parents[1].name and 
                path.parents[0].name == ref_path.parents[0].name):
                filtered_paths.append(path)
                break
    return filtered_paths

"""
Functions for camera TTL
"""
def process_ttl(ttl_path, ttl_floor=-32000, min_frame_duration=6, min_frame_spacing=6):
    sample_hz=30000
    resample_hz=1000

    data = np.load(ttl_path)
    ttl_resample = data[::sample_hz//resample_hz]

    ttl_bool = clean_camera_ttl(ttl_resample, 
                           threshold=ttl_floor,
                           min_frame_duration=min_frame_duration,
                           min_frame_spacing=min_frame_spacing)

    return ttl_bool


def crop_ttl(ttl_bool):
    # Find all low-to-high transitions
    valid_transitions = np.where(np.diff(ttl_bool) == 1)[0]
    # Calculate duration between transitions by taking difference
    transition_durations = np.diff(valid_transitions)
    # Add the last duration by duplicating the final duration
    transition_durations = np.append(transition_durations, transition_durations[-1])
    # Keep only transitions that are at least 10 samples apart
    valid_transitions = valid_transitions[transition_durations >= 10]
    # Apply crop to ttl_bool
    ttl_crop = ttl_bool[valid_transitions[0]:valid_transitions[-1]]
        
    return ttl_crop, valid_transitions


def ttl_bool(data_path: str, results_path: str, sample_hz=30000, resample_hz=1000, save=True):
    data = np.load(data_path)

    #Resample data to 1000 Hz
    ttl_resample = data[::sample_hz//resample_hz]

    # Normalize to 0-1 range
    normalized = (ttl_resample - np.min(ttl_resample)) / (np.max(ttl_resample) - np.min(ttl_resample))
    
    # Rebuild ttl signal as boolean
    ttl_bool = ttl_resample > -30000

    if save:
        np.save(results_path, ttl_bool)
    return ttl_bool


def clean_camera_ttl(signal, threshold=-30000, min_frame_duration=10, min_frame_spacing=10, echo=False):
    # Initial threshold
    binary = (signal < threshold).astype(int)
    if echo:
        print("Number of samples below threshold:", np.sum(binary))

    # Find potential frame boundaries
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    if echo:
        print("Number of starts found:", len(starts))
        print("Number of ends found:", len(ends))
        if len(starts) > 0:
            print("First few start indices:", starts[:5])
        if len(ends) > 0:
            print("First few end indices:", ends[:5])
    
    # Ensure we have matching starts and ends
    if len(starts) == 0 or len(ends) == 0:
        if echo:
            print("No valid transitions found")
        return np.zeros_like(signal, dtype=int)
    
    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:-1]

    # Filter based on duration and spacing
    valid_frames = []
    last_valid_end = -min_frame_spacing
    
    for start, end in zip(starts, ends):
        duration = end - start
        spacing = start - last_valid_end
        
        if duration >= min_frame_duration and spacing >= min_frame_spacing:
            valid_frames.append((start, end))
            last_valid_end = end
    
    # Create cleaned signal
    cleaned = np.zeros_like(signal, dtype=int)
    for start, end in valid_frames:
        cleaned[start:end] = 1
        
    return cleaned


def analyze_ttl_timing(signal, threshold=-25000):
    binary = (signal < threshold).astype(int)
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    if len(starts) > 0 and len(ends) > 0:
        # Make sure we have matching pairs
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
            
        # Now calculate durations and spacings
        durations = ends - starts  # How long the signal is "on"
        spacings = starts[1:] - ends[:-1]  # Time between pulses
        
        print(f"Average pulse duration: {np.mean(durations):.2f} samples ({np.mean(durations)/1000*1000:.2f} ms)")
        print(f"Average spacing between pulses: {np.mean(spacings):.2f} samples ({np.mean(spacings)/1000*1000:.2f} ms)")
        print(f"Frame rate: {1000/np.mean(starts[1:] - starts[:-1]):.2f} fps")
        
        # Additional diagnostic info
        print(f"\nNumber of pulses analyzed: {len(durations)}")
        print(f"Duration range: {np.min(durations):.2f} to {np.max(durations):.2f} samples")
        print(f"Spacing range: {np.min(spacings):.2f} to {np.max(spacings):.2f} samples")


def convert_timestamps_to_ms(timestamps_df):
    """
    Convert a dataframe of datetime strings to milliseconds from start.
    
    Args:
        timestamps_df: DataFrame or Series containing datetime strings
        
    Returns:
        Series containing millisecond timestamps relative to first entry
    """
    # Convert strings to datetime objects if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(timestamps_df):
        timestamps = pd.to_datetime(timestamps_df)
    else:
        timestamps = timestamps_df
        
    # Get the first timestamp
    start_time = timestamps.iloc[0]
    
    # Convert to milliseconds from start
    timestamps_ms = (timestamps - start_time).dt.total_seconds() * 1000
    
    return timestamps_ms


# Clickbait event dataframe handling:
def process_events(idx, event_paths_a, event_paths_b, columns: dict):
    """
    Process event data from paired CSV files.
    
    Args:
        event_paths_a: List of paths to eventsA CSV files
        event_paths_b: List of paths to eventsB CSV files
        idx: Index of the session to process
        columns: Dictionary of variable names and datatypes
        
    Returns:
        DataFrame containing processed event data
    """
    # Load events .csv part A (This .csv is always 7 cols wide, but we don't hardcode in case Bonsai WriteCsv improves.)
    event_data_a = pd.read_csv(event_paths_a[idx])
    col_names_a = list(columns.keys())[:len(event_data_a.columns)]
    event_data_a.columns = col_names_a
    pd.to_datetime(event_data_a['timestamp'])

    # Load events .csv part B (col number varies by experimental condition, so we need to load in col names flexibly.)
    event_data_b = pd.read_csv(event_paths_b[idx])
    col_names_b = list(columns.keys())[len(event_data_a.columns):]
    event_data_b.columns = col_names_b

    # Concatenate eventsA and eventsB dataframes
    if len(event_data_a) == len(event_data_b):
        event_data = pd.concat([event_data_a, event_data_b], axis=1)
    else:
        print("Event dataframes must contain same number of rows")
        min_length = min(len(event_data_a), len(event_data_b))
        max_length = max(len(event_data_a), len(event_data_b))
        print(f"Trimmed long dataframe by {max_length-min_length} rows.")
        event_data_a = event_data_a.iloc[:min_length]
        event_data_b = event_data_b.iloc[:min_length]
        event_data = pd.concat([event_data_a, event_data_b], axis=1)

    # Set columns to appropriate types as specified in columns dictionary
    event_data = event_data.astype(columns)

    # Calculate speed and direction columns
    event_data['speed'] = calculate_speed(event_data)
    event_data['direction'] = calculate_direction(event_data)
    
    # Rebuild 'reward_state' as periods between click onset and poke events
    # reward_state = True from when 'click' goes False->True until either 'poke_left' or 'poke_right' goes False->True
    
    reward_state = np.zeros(len(event_data), dtype=bool)
    
    # Find click transitions from False to True
    click_onsets = np.where((event_data['click'].shift(1) == False) & 
                           (event_data['click'] == True))[0]
    
    # Find poke transitions from False to True (either left or right)
    poke_left_onsets = np.where((event_data['poke_left'].shift(1) == False) & 
                               (event_data['poke_left'] == True))[0]
    poke_right_onsets = np.where((event_data['poke_right'].shift(1) == False) & 
                                (event_data['poke_right'] == True))[0]
    
    # Combine and sort all poke onsets
    all_poke_onsets = np.sort(np.concatenate([poke_left_onsets, poke_right_onsets]))
    
    # For each click onset, find the next poke onset and mark the period as reward_state
    for click_idx in click_onsets:
        # Find the next poke onset after this click onset
        next_pokes = all_poke_onsets[all_poke_onsets > click_idx]
        if len(next_pokes) > 0:
            poke_idx = next_pokes[0]
            # Mark reward_state period from click onset to poke onset
            reward_state[click_idx:poke_idx] = True
    
    event_data['reward_state'] = reward_state
    
    # Add 'drinking' column (period between reward initiation poke and start of ITI)
    event_data['drinking'] = calculate_drinking(event_data)

    return event_data

# Calculate speed and direction
def calculate_speed(data):
    """Calculate speed from x,y coordinates"""
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y']) 
    speed = np.sqrt(dx**2 + dy**2)
    # Add 0 at start to match length
    return np.concatenate(([0], speed))

def calculate_direction(data):
    """Calculate movement direction in radians from x,y coordinates"""
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y'])
    direction = np.arctan2(dy, dx)
    # Add 0 at start to match length 
    return np.concatenate(([0], direction))

def calculate_drinking(data):
    """Calculate drinking periods between reward_state True->False and ITI False->True"""
    drinking = np.zeros(len(data), dtype=bool)
    
    # Find reward_state transitions from True to False
    reward_transitions = np.where((data['reward_state'].shift(1) == True) & 
                                 (data['reward_state'] == False))[0]
    
    # Find iti transitions from False to True
    iti_transitions = np.where((data['iti'].shift(1) == False) & 
                              (data['iti'] == True))[0]
    
    # For each reward_state True->False transition, find the next iti False->True transition
    for reward_idx in reward_transitions:
        # Find the next iti transition after this reward transition
        next_iti = iti_transitions[iti_transitions > reward_idx]
        if len(next_iti) > 0:
            iti_idx = next_iti[0]
            # Mark drinking period from reward transition to iti transition
            drinking[reward_idx:iti_idx] = True
    
    return drinking

