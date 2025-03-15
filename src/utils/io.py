"""
IO utility functions for loading and saving data.
"""
import os
import yaml
import csv
import json
from typing import Dict, List, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        # Return default configuration
        return {
            "preprocessing": {
                "lowercase": True,
                "remove_punctuation": False,
                "expand_abbreviations": True,
                "max_length": 512
            },
            "ner": {
                "model": "biobert-base-cased-v1.1",
                "labels": ["DIAGNOSIS", "PROCEDURE", "MEDICATION", "SYMPTOM", "ANATOMY"]
            },
            "model": {
                "name": "biobert-base-cased-v1.1",
                "max_length": 512
            }
        }


def load_icd10_codes(file_path: str) -> Dict[str, str]:
    """
    Load ICD-10 codes from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing ICD-10 codes
        
    Returns:
        Dictionary mapping ICD-10 codes to descriptions
    """
    codes = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    code, description = row[0], row[1]
                    codes[code] = description
        print(f"Loaded {len(codes)} ICD-10 codes from {file_path}")
    except Exception as e:
        print(f"Error loading ICD-10 codes from {file_path}: {e}")
    
    return codes


def load_cpt_codes(file_path: str) -> Dict[str, str]:
    """
    Load CPT codes from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing CPT codes
        
    Returns:
        Dictionary mapping CPT codes to descriptions
    """
    codes = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    code, description = row[0], row[1]
                    codes[code] = description
        print(f"Loaded {len(codes)} CPT codes from {file_path}")
    except Exception as e:
        print(f"Error loading CPT codes from {file_path}: {e}")
    
    return codes


def save_predictions(predictions: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save predictions to a JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the predictions to
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {output_path}")
    except Exception as e:
        print(f"Error saving predictions to {output_path}: {e}")


def load_text_file(file_path: str) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content of the file
    """
    try:
        with open(file_path, 'r') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Error loading text from {file_path}: {e}")
        return "" 