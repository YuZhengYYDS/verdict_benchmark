"""
NLLS (Non-Linear Least Squares) Parameter Processing Utility

This module provides functions to load and process NLLS fitted parameters from MATLAB files,
specifically for the AstroSticks-dvZeppelin-Sphere model used in VERDICT MRI analysis.

Functions:
    load_nlls_parameters: Load and process NLLS parameters from MATLAB file
    extract_parameter_statistics: Calculate comprehensive statistics for NLLS parameters
    create_parameter_dataframe: Create pandas DataFrame with processed parameters
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, Tuple, Optional, Union
import os
from pathlib import Path


def load_nlls_parameters(
    mat_file_path: Union[str, Path],
    target_params: Optional[list] = None,
    apply_unit_conversion: bool = True,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load and process NLLS fitted parameters from MATLAB file.
    
    Args:
        mat_file_path (str or Path): Path to the MATLAB file containing NLLS fitted parameters
        target_params (list, optional): List of target parameter names to extract.
                                      Defaults to ['fic', 'fee', 'Dic', 'R']
        apply_unit_conversion (bool): Whether to apply SI unit conversions 
                                    (Dic: μm²/ms → m²/s, R: μm → m)
        verbose (bool): Whether to print processing information
    
    Returns:
        Dict[str, np.ndarray]: Dictionary containing processed parameter arrays
                              Keys: 'fic', 'fee', 'Dic', 'R' (and others if specified)
                              Values: numpy arrays with parameter values
    
    Raises:
        FileNotFoundError: If the MATLAB file doesn't exist
        KeyError: If required keys are missing from the MATLAB file
        ValueError: If parameter extraction fails
    
    Example:
        >>> params = load_nlls_parameters('Patient05_FittedParams.mat')
        >>> print(f"fic values shape: {params['fic'].shape}")
        >>> print(f"Dic range: {params['Dic'].min():.2e} - {params['Dic'].max():.2e}")
    """
    
    # Default target parameters for VERDICT NLLS analysis
    if target_params is None:
        target_params = ['fic', 'fee', 'Dic', 'R']
    
    # Validate file path
    mat_file_path = Path(mat_file_path)
    if not mat_file_path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {mat_file_path}")
    
    if verbose:
        print("=== Loading NLLS fitted parameters ===")
        print(f"File: {mat_file_path}")
    
    try:
        # Load MATLAB file
        nlls_mat = sio.loadmat(str(mat_file_path))
        
        if verbose:
            print(f"Available keys in MATLAB file: {list(nlls_mat.keys())}")
        
        # Filter out MATLAB metadata keys
        data_keys = [key for key in nlls_mat.keys() if not key.startswith('__')]
        if verbose:
            print(f"Data keys: {data_keys}")
        
        # Extract required structures
        required_keys = ['model', 'gsps', 'mlps']
        for key in required_keys:
            if key not in nlls_mat:
                raise KeyError(f"Required key '{key}' not found in MATLAB file")
        
        model = nlls_mat['model']
        gsps = nlls_mat['gsps']
        mlps = nlls_mat['mlps']
        
        if verbose:
            print(f"Model parameters shape: {mlps.shape}")
            print(f"Grid search parameters shape: {gsps.shape}")
        
        # Extract parameter names from model structure
        try:
            param_names = [param[0] for param in model[0][0][2][0]]
            if verbose:
                print(f"Available parameter names: {param_names}")
        except (IndexError, TypeError) as e:
            if verbose:
                print(f"Warning: Could not extract parameter names from model structure: {e}")
            param_names = []
        
        # Parameter mapping for AstroSticks-dvZeppelin-Sphere model
        # Based on typical VERDICT parameter ordering
        param_mapping = {
            'fic': 0,    # ficvf - intracellular fraction
            'fee': 1,    # fee - extracellular fraction  
            'R': 3,      # rads - cell radius (μm)
            'Dic': 4,    # di - intracellular diffusivity (μm²/ms)
        }
        
        # Extract parameters
        parameters = {}
        
        for param_name in target_params:
            if param_name in param_mapping:
                col_idx = param_mapping[param_name]
                if col_idx < mlps.shape[1]:
                    parameters[param_name] = mlps[:, col_idx].copy()
                    if verbose:
                        print(f"Extracted {param_name} from column {col_idx}")
                else:
                    raise ValueError(f"Column index {col_idx} for parameter '{param_name}' exceeds data dimensions")
            else:
                raise ValueError(f"Unknown parameter '{param_name}'. Available: {list(param_mapping.keys())}")
        
        # Apply unit conversions to SI units
        if apply_unit_conversion:
            if 'Dic' in parameters:
                # Convert Dic from μm²/ms to m²/s: 1 μm²/ms = 1e-9 m²/s
                parameters['Dic'] = parameters['Dic'] * 1e-9
                if verbose:
                    print("Converted Dic: μm²/ms → m²/s")
            
            if 'R' in parameters:
                # Convert R from μm to m: 1 μm = 1e-6 m
                parameters['R'] = parameters['R'] * 1e-6
                if verbose:
                    print("Converted R: μm → m")
        
        if verbose:
            print(f"Successfully extracted {len(parameters)} parameters")
            print(f"Total voxels: {len(next(iter(parameters.values())))}")
        
        return parameters
        
    except Exception as e:
        raise ValueError(f"Error processing MATLAB file: {str(e)}") from e


def extract_parameter_statistics(
    parameters: Dict[str, np.ndarray],
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive statistics for NLLS parameters.
    
    Args:
        parameters (Dict[str, np.ndarray]): Dictionary of parameter arrays
        verbose (bool): Whether to print statistics table
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary with statistics
                                   Format: {param_name: {stat_name: value}}
    
    Example:
        >>> stats = extract_parameter_statistics(params)
        >>> fic_mean = stats['fic']['mean']
        >>> dic_std = stats['Dic']['std']
    """
    
    statistics = {}
    
    if verbose:
        print("\n=== NLLS Parameter Statistics ===")
        print("Parameter | Mean        | Std         | Min         | Max         | Zeros")
        print("----------|-------------|-------------|-------------|-------------|-------")
    
    for param_name, param_values in parameters.items():
        # Calculate statistics
        stats = {
            'mean': float(np.mean(param_values)),
            'std': float(np.std(param_values)),
            'min': float(np.min(param_values)),
            'max': float(np.max(param_values)),
            'median': float(np.median(param_values)),
            'q25': float(np.percentile(param_values, 25)),
            'q75': float(np.percentile(param_values, 75)),
            'zeros': int(np.sum(param_values == 0)),
            'non_zeros': int(np.sum(param_values != 0)),
            'total': len(param_values)
        }
        
        statistics[param_name] = stats
        
        if verbose:
            # Use scientific notation for very small values (Dic and R)
            if param_name in ['Dic', 'R']:
                print(f"{param_name:9s} | {stats['mean']:11.2e} | {stats['std']:11.2e} | "
                      f"{stats['min']:11.2e} | {stats['max']:11.2e} | {stats['zeros']:7d}")
            else:
                print(f"{param_name:9s} | {stats['mean']:11.4f} | {stats['std']:11.4f} | "
                      f"{stats['min']:11.4f} | {stats['max']:11.4f} | {stats['zeros']:7d}")
    
    if verbose:
        total_voxels = len(next(iter(parameters.values())))
        print(f"\nTotal voxels: {total_voxels}")
        print("Note: Dic values are in m²/s, R values are in m")
    
    return statistics


def create_parameter_dataframe(
    parameters: Dict[str, np.ndarray],
    include_metadata: bool = True
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with processed NLLS parameters.
    
    Args:
        parameters (Dict[str, np.ndarray]): Dictionary of parameter arrays
        include_metadata (bool): Whether to include metadata columns
    
    Returns:
        pd.DataFrame: DataFrame with parameter columns and optional metadata
    
    Example:
        >>> df = create_parameter_dataframe(params)
        >>> print(df.describe())
        >>> print(f"DataFrame shape: {df.shape}")
    """
    
    # Create base DataFrame
    df = pd.DataFrame(parameters)
    
    if include_metadata:
        # Add voxel index
        df['voxel_id'] = range(len(df))
        
        # Add parameter validity flags
        for param_name in parameters.keys():
            df[f'{param_name}_valid'] = (df[param_name] > 0) & np.isfinite(df[param_name])
        
        # Add overall validity flag (all parameters valid)
        valid_cols = [col for col in df.columns if col.endswith('_valid')]
        df['all_valid'] = df[valid_cols].all(axis=1)
    
    return df


def process_nlls_file(
    mat_file_path: Union[str, Path],
    target_params: Optional[list] = None,
    apply_unit_conversion: bool = True,
    return_statistics: bool = False,
    return_dataframe: bool = True,
    verbose: bool = True
) -> Union[Dict, Tuple]:
    """
    Complete NLLS file processing pipeline.
    
    This is the main function that combines all processing steps for convenience.
    
    Args:
        mat_file_path (str or Path): Path to the MATLAB file
        target_params (list, optional): Target parameter names
        apply_unit_conversion (bool): Whether to apply SI unit conversions
        return_statistics (bool): Whether to return parameter statistics
        return_dataframe (bool): Whether to return pandas DataFrame
        verbose (bool): Whether to print processing information
    
    Returns:
        Dict or Tuple: Depending on return flags:
                      - Just parameters dict if both return flags are False
                      - Tuple with (parameters, statistics) if return_statistics=True
                      - Tuple with (parameters, dataframe) if return_dataframe=True
                      - Tuple with (parameters, statistics, dataframe) if both are True
    
    Example:
        >>> # Basic usage - returns parameters dict and DataFrame
        >>> params, df = process_nlls_file('Patient05_FittedParams.mat')
        >>> 
        >>> # Full analysis with statistics
        >>> params, stats, df = process_nlls_file(
        ...     'Patient05_FittedParams.mat',
        ...     return_statistics=True,
        ...     return_dataframe=True
        ... )
    """
    
    # Load and process parameters
    parameters = load_nlls_parameters(
        mat_file_path=mat_file_path,
        target_params=target_params,
        apply_unit_conversion=apply_unit_conversion,
        verbose=verbose
    )
    
    # Prepare return values
    results = [parameters]
    
    if return_statistics:
        statistics = extract_parameter_statistics(parameters, verbose=verbose)
        results.append(statistics)
    
    if return_dataframe:
        dataframe = create_parameter_dataframe(parameters, include_metadata=True)
        results.append(dataframe)
        if verbose:
            print(f"\nDataFrame created with shape: {dataframe.shape}")
            print(f"Columns: {list(dataframe.columns)}")
    
    # Return appropriate format
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


# Example usage and testing
if __name__ == "__main__":
    # Example usage (uncomment and modify path as needed)
    """
    # Basic usage
    mat_file = r'D:\\AiProjects\\UCLmaster\\OneDrive_1_2025-5-2\\data\\Patient05\\FittedParams_AstroSticksdvZeppelinSphereB0T2_FWE_fixdv_40.mat'
    
    # Process NLLS file
    params, df = process_nlls_file(mat_file, verbose=True)
    
    # Display results
    print(f"\nProcessed parameters: {list(params.keys())}")
    print(f"DataFrame shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Get full analysis
    params, stats, df = process_nlls_file(
        mat_file,
        return_statistics=True,
        return_dataframe=True,
        verbose=True
    )
    """
    
    print("NLLS processor module loaded successfully!")
    print("Main function: process_nlls_file()")
    print("Use help(process_nlls_file) for detailed documentation.")