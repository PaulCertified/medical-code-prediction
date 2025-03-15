"""
Data utility functions for loading and saving data.
"""
import os
import json
import yaml
import pickle
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return {}


def load_data(data_path: str, file_format: Optional[str] = None) -> Any:
    """
    Load data from a file.
    
    Args:
        data_path: Path to the data file
        file_format: Format of the data file ('csv', 'json', 'pickle', 'txt', etc.)
                    If None, inferred from the file extension
        
    Returns:
        Loaded data
    """
    if file_format is None:
        file_format = os.path.splitext(data_path)[1][1:].lower()
    
    try:
        if file_format == 'csv':
            return pd.read_csv(data_path)
        elif file_format == 'json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif file_format == 'pickle':
            with open(data_path, 'rb') as f:
                return pickle.load(f)
        elif file_format == 'txt':
            with open(data_path, 'r') as f:
                return f.read()
        elif file_format == 'yaml' or file_format == 'yml':
            with open(data_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None


def save_data(data: Any, data_path: str, file_format: Optional[str] = None) -> bool:
    """
    Save data to a file.
    
    Args:
        data: Data to save
        data_path: Path to save the data to
        file_format: Format of the data file ('csv', 'json', 'pickle', 'txt', etc.)
                    If None, inferred from the file extension
        
    Returns:
        True if successful, False otherwise
    """
    if file_format is None:
        file_format = os.path.splitext(data_path)[1][1:].lower()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
        
        if file_format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(data_path, index=False)
            else:
                pd.DataFrame(data).to_csv(data_path, index=False)
        elif file_format == 'json':
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif file_format == 'pickle':
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
        elif file_format == 'txt':
            with open(data_path, 'w') as f:
                f.write(str(data))
        elif file_format == 'yaml' or file_format == 'yml':
            with open(data_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return True
    except Exception as e:
        print(f"Error saving data to {data_path}: {e}")
        return False


def split_data(data: Union[pd.DataFrame, List], 
              train_ratio: float = 0.8, 
              val_ratio: float = 0.1,
              test_ratio: float = 0.1,
              random_seed: int = 42) -> Dict[str, Any]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data: Data to split (DataFrame or list)
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing the split data
    """
    import numpy as np
    
    # Check that ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Train, validation, and test ratios must sum to 1")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    if isinstance(data, pd.DataFrame):
        # Shuffle the DataFrame
        data = data.sample(frac=1).reset_index(drop=True)
        
        # Calculate split indices
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split the data
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
    else:
        # Convert to list if not already
        data = list(data)
        
        # Shuffle the list
        np.random.shuffle(data)
        
        # Calculate split indices
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split the data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    } 