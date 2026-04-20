"""
Submission formatting utilities.

Handles conversion between numpy arrays and submission strings.
Format: 0.xxxxxx-0.xxxxxx-... (comma-separated coordinates, one per query point)
"""

import numpy as np
from typing import List, Tuple, Dict


def array_to_submission_string(query_array: np.ndarray, precision: int = 6) -> str:
    """
    Convert a single query point (numpy array) to submission string format.
    
    Parameters
    ----------
    query_array : np.ndarray
        1D array of coordinates [x1, x2, ..., xd]
    precision : int, default=6
        Number of decimal places
    
    Returns
    -------
    str
        Formatted string: "0.xxxxxx-0.xxxxxx-..."
        
    Example
    -------
    >>> array_to_submission_string(np.array([0.5, 0.5]))
    '0.500000-0.500000'
    """
    formatted = "-".join(f"{x:.{precision}f}" for x in query_array)
    return formatted


def submission_string_to_array(query_string: str) -> np.ndarray:
    """
    Convert submission string format to numpy array.
    
    Parameters
    ----------
    query_string : str
        Formatted string: "0.xxxxxx-0.xxxxxx-..."
    
    Returns
    -------
    np.ndarray
        1D array of coordinates
        
    Example
    -------
    >>> submission_string_to_array('0.500000-0.500000')
    array([0.5, 0.5])
    """
    coords = [float(x) for x in query_string.split("-")]
    return np.array(coords)


def format_submission_query(
    query_arrays: List[np.ndarray],
    precision: int = 6
) -> Dict[int, str]:
    """
    Format a batch of query arrays (one per function) into submission format.
    
    Parameters
    ----------
    query_arrays : List[np.ndarray]
        List of 8 query arrays, one per function (F1-F8)
    precision : int, default=6
        Number of decimal places
    
    Returns
    -------
    Dict[int, str]
        Dictionary mapping function_id (1-8) to formatted query strings
        
    Example
    -------
    >>> queries = [np.array([0.5, 0.5]), np.array([0.3, 0.7]), ...]
    >>> formatted = format_submission_query(queries)
    >>> formatted[1]
    '0.500000-0.500000'
    """
    from typing import Dict
    
    formatted = {}
    for func_id, query_array in enumerate(query_arrays, start=1):
        formatted[func_id] = array_to_submission_string(query_array, precision)
    
    return formatted


def validate_submission(
    query_arrays: List[np.ndarray],
    expected_dims: List[int] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that submission meets dimensionality and range constraints.
    
    Parameters
    ----------
    query_arrays : List[np.ndarray]
        List of query arrays to validate
    expected_dims : List[int], optional
        Expected dimensionality for each function. 
        If None, uses defaults: [2, 2, 3, 4, 4, 5, 6, 8]
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_error_messages)
    """
    if expected_dims is None:
        expected_dims = [2, 2, 3, 4, 4, 5, 6, 8]
    
    errors = []
    
    if len(query_arrays) != 8:
        errors.append(f"Expected 8 functions, got {len(query_arrays)}")
        return False, errors
    
    for func_id, (query, expected_dim) in enumerate(zip(query_arrays, expected_dims), start=1):
        # Check dimensionality
        if query.ndim != 1:
            errors.append(f"F{func_id}: Expected 1D array, got shape {query.shape}")
        elif len(query) != expected_dim:
            errors.append(f"F{func_id}: Expected {expected_dim}D, got {len(query)}D")
        
        # Check range [0, 1]
        if np.any(query < 0) or np.any(query > 1):
            out_of_range = query[(query < 0) | (query > 1)]
            errors.append(
                f"F{func_id}: Out-of-range coordinates found. "
                f"Valid range: [0, 1]. Got: {out_of_range}"
            )
    
    return len(errors) == 0, errors
