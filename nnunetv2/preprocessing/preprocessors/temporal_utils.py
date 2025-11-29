"""
Temporal file grouping utilities for SA_TA_UNet.
Handles identification of temporal sequences using PatientID_FrameNumber pattern.
"""
import re
import os
from typing import List, Dict, Tuple, Optional


def group_temporal_files(file_lists: List[List[str]], 
                        identifiers: Optional[List[str]] = None,
                        verbose: bool = True) -> Dict[int, List[str]]:
    """
    Group temporal files and match each file to its previous frame.
    
    Args:
        file_lists: List of file lists (each file list represents one case's channels)
        identifiers: Optional list of case identifiers (basenames without extensions)
                    If None, will extract from first file in each file_list
        verbose: Whether to print warnings for non-compliant filenames
    
    Returns:
        previous_file_mapping: Dict mapping original index to previous frame's file list
        
    Pattern:
        PatientID_FrameNumber (e.g., IVUS_001_0000, IVUS_001_0001)
        - Same PatientID = same temporal sequence
        - FrameNumber = temporal order
        - First frame (or non-compliant): previous = current
        
    Example:
        >>> file_lists = [['IVUS_001_0000.nii.gz'], ['IVUS_001_0001.nii.gz']]
        >>> mapping = group_temporal_files(file_lists)
        >>> mapping[0]  # First frame -> self
        ['IVUS_001_0000.nii.gz']
        >>> mapping[1]  # Frame 1 -> Frame 0
        ['IVUS_001_0000.nii.gz']
    """
    pattern = re.compile(r'(.*)_(\d+)$')
    
    # Group by PatientID
    grouped_by_id = {}
    
    for idx, file_list in enumerate(file_lists):
        # Extract identifier (basename without extension)
        if identifiers is not None:
            identifier = identifiers[idx]
        else:
            basename = os.path.basename(file_list[0])
            # Remove all extensions (.nii.gz -> remove .gz then .nii)
            identifier = basename.split('.')[0] if '.' in basename else basename
        
        match = pattern.match(identifier)
        
        if not match:
            # Non-compliant filename: treat as standalone (previous = current)
            if verbose:
                print(f"\n{'='*80}")
                print(f"WARNING: Filename '{identifier}' does not match 'PatientID_FrameNumber' pattern")
                print(f"         Previous frame will be set to current (first frame behavior)")
                print(f"{'='*80}\n")
            
            # Use special group for standalone files
            if 'standalone' not in grouped_by_id:
                grouped_by_id['standalone'] = []
            grouped_by_id['standalone'].append((0, idx, file_list))
            continue
        
        patient_id = match.group(1)
        frame_num = int(match.group(2))
        
        if patient_id not in grouped_by_id:
            grouped_by_id[patient_id] = []
        grouped_by_id[patient_id].append((frame_num, idx, file_list))
    
    # Create previous frame mapping
    previous_file_mapping = {}
    
    for patient_id, frames in grouped_by_id.items():
        # Sort by frame number
        frames.sort(key=lambda x: x[0])
        
        for i, (frame_num, original_idx, file_list) in enumerate(frames):
            if i == 0:
                # First frame or standalone: previous = current
                previous_file_mapping[original_idx] = file_list
            else:
                # Subsequent frames: previous = previous frame
                prev_frame_idx = frames[i-1][1]
                previous_file_mapping[original_idx] = file_lists[prev_frame_idx]
    
    return previous_file_mapping


def get_previous_file_list(file_list: List[str], 
                          all_file_lists: List[List[str]],
                          all_identifiers: Optional[List[str]] = None,
                          cache: Optional[Dict[int, List[str]]] = None) -> List[str]:
    """
    Get the previous frame's file list for a given file list.
    
    Args:
        file_list: Current file list
        all_file_lists: All file lists in the dataset
        all_identifiers: Optional identifiers for all cases
        cache: Optional pre-computed mapping (from group_temporal_files)
    
    Returns:
        Previous frame's file list
    """
    if cache is not None:
        # Use pre-computed mapping
        try:
            idx = all_file_lists.index(file_list)
            return cache[idx]
        except (ValueError, KeyError):
            # Fallback: return current as previous
            return file_list
    else:
        # Compute on-the-fly (not recommended for batch processing)
        mapping = group_temporal_files(all_file_lists, all_identifiers, verbose=False)
        idx = all_file_lists.index(file_list)
        return mapping.get(idx, file_list)


def group_temporal_identifiers(identifiers: List[str], verbose: bool = True) -> Dict[int, str]:
    """
    Simplified version that works with case identifiers directly.
    Returns mapping from index to previous case identifier.
    
    Args:
        identifiers: List of case identifiers (e.g., ['IVUS_001_0000', 'IVUS_001_0001'])
        verbose: Whether to print warnings
        
    Returns:
        Mapping from original index to previous identifier
    """
    pattern = re.compile(r'(.*)_(\d+)$')
    grouped_by_id = {}
    
    for idx, identifier in enumerate(identifiers):
        match = pattern.match(identifier)
        
        if not match:
            if verbose:
                print(f"\n{'='*80}")
                print(f"WARNING: Identifier '{identifier}' does not match 'PatientID_FrameNumber' pattern")
                print(f"         Previous frame will be set to current (first frame behavior)")
                print(f"{'='*80}\n")
            if 'standalone' not in grouped_by_id:
                grouped_by_id['standalone'] = []
            grouped_by_id['standalone'].append((0, idx, identifier))
            continue
        
        patient_id = match.group(1)
        frame_num = int(match.group(2))
        
        if patient_id not in grouped_by_id:
            grouped_by_id[patient_id] = []
        grouped_by_id[patient_id].append((frame_num, idx, identifier))
    
    # Create mapping
    previous_identifier_mapping = {}
    
    for patient_id, frames in grouped_by_id.items():
        frames.sort(key=lambda x: x[0])
        
        for i, (frame_num, original_idx, identifier) in enumerate(frames):
            if i == 0:
                previous_identifier_mapping[original_idx] = identifier
            else:
                prev_frame_idx = frames[i-1][1]
                previous_identifier_mapping[original_idx] = identifiers[prev_frame_idx]
    
    return previous_identifier_mapping


def create_temporal_mapping_for_dataset(dataset: dict, verbose: bool = True) -> Dict[str, str]:
    """
    Create temporal mapping for preprocessing stage.
    Maps each case_id to its previous_case_id for temporal sequence handling.
    
    This function is designed for the preprocessing stage where we need to know
    which previous case to load for each current case before multiprocessing.
    
    Args:
        dataset: Dataset dict from get_filenames_of_train_images_and_targets
                 Format: {'case_id': {'images': [...], 'label': '...'}, ...}
                 Example: {'IVUS_001_0000': {'images': ['...'], 'label': '...'},
                          'IVUS_001_0001': {'images': ['...'], 'label': '...'}}
        verbose: Whether to print warnings for non-compliant filenames
    
    Returns:
        temporal_mapping: Dict mapping case_id to previous_case_id
                         Format: {'case_id': 'previous_case_id', ...}
                         First frames and standalone files map to themselves
    
    Example:
        >>> dataset = {
        ...     'IVUS_001_0000': {'images': [...], 'label': '...'},
        ...     'IVUS_001_0001': {'images': [...], 'label': '...'},
        ...     'IVUS_001_0002': {'images': [...], 'label': '...'},
        ... }
        >>> mapping = create_temporal_mapping_for_dataset(dataset)
        >>> mapping['IVUS_001_0000']  # First frame -> self
        'IVUS_001_0000'
        >>> mapping['IVUS_001_0001']  # Frame 1 -> Frame 0
        'IVUS_001_0000'
        >>> mapping['IVUS_001_0002']  # Frame 2 -> Frame 1
        'IVUS_001_0001'
    """
    # Get list of case_ids (preserve order from dataset.keys())
    case_ids = list(dataset.keys())
    
    # Use existing group_temporal_identifiers to get index-based mapping
    # Returns: {idx: previous_identifier}
    idx_to_previous_id = group_temporal_identifiers(case_ids, verbose=verbose)
    
    # Convert index-based mapping to case_id-based mapping
    # This makes it easy for workers to look up: temporal_mapping[case_id] = previous_case_id
    temporal_mapping = {}
    for idx, case_id in enumerate(case_ids):
        previous_case_id = idx_to_previous_id[idx]
        temporal_mapping[case_id] = previous_case_id
    
    return temporal_mapping
