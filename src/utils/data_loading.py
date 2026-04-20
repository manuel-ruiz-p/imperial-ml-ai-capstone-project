"""
Data loading utilities for the capstone project.

Functions:
- load_function_data: Load initial inputs/outputs for a specific function
- load_all_functions: Load data for all 8 functions from a given directory
- load_weekly_results: Load Week N query results from submission directory
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def load_function_data(
    function_id: int,
    data_dir: Optional[Path] = None,
    week: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load input and output data for a specific function.
    
    Parameters
    ----------
    function_id : int
        Function number (1-8)
    data_dir : Path, optional
        Path to data directory. If None, uses default data/raw/
    week : int, optional
        Week number to load. If None, loads all historical data.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (inputs, outputs) arrays
        
    Raises
    ------
    FileNotFoundError
        If data files do not exist for the specified function
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    func_dir = data_dir / f"function_{function_id}"
    
    # Load initial .npy files
    inputs_path = func_dir / "initial_inputs.npy"
    outputs_path = func_dir / "initial_outputs.npy"
    
    if not inputs_path.exists() or not outputs_path.exists():
        raise FileNotFoundError(
            f"Missing data files for function {function_id} in {func_dir}"
        )
    
    inputs = np.load(inputs_path)
    outputs = np.load(outputs_path)
    
    return inputs, outputs


def load_all_functions(
    data_dir: Optional[Path] = None
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load input and output data for all 8 functions.
    
    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory. If None, uses default data/raw/
    
    Returns
    -------
    Dict[int, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping function_id -> (inputs, outputs)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    all_data = {}
    for func_id in range(1, 9):
        try:
            inputs, outputs = load_function_data(func_id, data_dir)
            all_data[func_id] = (inputs, outputs)
        except FileNotFoundError:
            print(f"Warning: Could not load data for function {func_id}")
    
    return all_data


def load_weekly_results(
    week: int,
    submissions_dir: Optional[Path] = None
) -> Dict[int, Tuple[np.ndarray, float]]:
    """
    Load Week N submission queries and results.
    
    Parameters
    ----------
    week : int
        Week number (1, 2, 3, ...)
    submissions_dir : Path, optional
        Path to submissions directory. If None, uses default submissions/
    
    Returns
    -------
    Dict[int, Tuple[np.ndarray, float]]
        Dictionary mapping function_id -> (query_point, output_value)
    """
    if submissions_dir is None:
        submissions_dir = Path(__file__).parent.parent.parent / "submissions"
    
    week_dir = submissions_dir / f"week_{week:02d}"
    
    if not week_dir.exists():
        raise FileNotFoundError(f"No submission directory for week {week}")
    
    results = {}
    
    # Try to load queries.txt and results.npy if they exist
    queries_file = week_dir / "queries.txt"
    results_file = week_dir / "results.txt"
    
    # TODO: Implement parsing logic for submission files
    # This depends on the exact submission format chosen
    
    return results


# TODO: Add function to convert between raw numpy data and submission format
