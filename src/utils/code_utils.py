"""
Utility functions for working with medical codes (ICD-10 and CPT).
"""
import os
import csv
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def load_icd10_codes(file_path: str) -> Dict[str, str]:
    """
    Load ICD-10 codes from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing ICD-10 codes
        
    Returns:
        Dictionary mapping ICD-10 codes to their descriptions
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        if 'code' in df.columns and 'description' in df.columns:
            return dict(zip(df['code'], df['description']))
        else:
            # Try to infer columns based on common naming patterns
            code_col = next((col for col in df.columns if 'code' in col.lower()), df.columns[0])
            desc_col = next((col for col in df.columns if 'desc' in col.lower()), df.columns[1])
            
            return dict(zip(df[code_col], df[desc_col]))
    except Exception as e:
        print(f"Error loading ICD-10 codes from {file_path}: {e}")
        return {}


def load_cpt_codes(file_path: str) -> Dict[str, str]:
    """
    Load CPT codes from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing CPT codes
        
    Returns:
        Dictionary mapping CPT codes to their descriptions
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        if 'code' in df.columns and 'description' in df.columns:
            return dict(zip(df['code'], df['description']))
        else:
            # Try to infer columns based on common naming patterns
            code_col = next((col for col in df.columns if 'code' in col.lower()), df.columns[0])
            desc_col = next((col for col in df.columns if 'desc' in col.lower()), df.columns[1])
            
            return dict(zip(df[code_col], df[desc_col]))
    except Exception as e:
        print(f"Error loading CPT codes from {file_path}: {e}")
        return {}


def code_to_description(code: str, 
                       icd10_codes: Optional[Dict[str, str]] = None,
                       cpt_codes: Optional[Dict[str, str]] = None,
                       icd10_path: Optional[str] = None,
                       cpt_path: Optional[str] = None) -> str:
    """
    Get the description for a medical code.
    
    Args:
        code: The medical code to look up
        icd10_codes: Dictionary mapping ICD-10 codes to descriptions
        cpt_codes: Dictionary mapping CPT codes to descriptions
        icd10_path: Path to the ICD-10 codes file (used if icd10_codes is None)
        cpt_path: Path to the CPT codes file (used if cpt_codes is None)
        
    Returns:
        The description for the code, or "Unknown code" if not found
    """
    # Load code dictionaries if not provided
    if icd10_codes is None and icd10_path:
        icd10_codes = load_icd10_codes(icd10_path)
    
    if cpt_codes is None and cpt_path:
        cpt_codes = load_cpt_codes(cpt_path)
    
    # Check if the code is in the ICD-10 dictionary
    if icd10_codes and code in icd10_codes:
        return icd10_codes[code]
    
    # Check if the code is in the CPT dictionary
    if cpt_codes and code in cpt_codes:
        return cpt_codes[code]
    
    # If not found in either dictionary, return "Unknown code"
    return "Unknown code"


def is_valid_icd10(code: str) -> bool:
    """
    Check if a code is a valid ICD-10 code format.
    
    Args:
        code: The code to check
        
    Returns:
        True if the code is a valid ICD-10 format, False otherwise
    """
    import re
    
    # ICD-10 format: Letter followed by 2 digits, optionally followed by a period and more digits
    pattern = r'^[A-Z]\d{2}(\.\d+)?$'
    
    return bool(re.match(pattern, code))


def is_valid_cpt(code: str) -> bool:
    """
    Check if a code is a valid CPT code format.
    
    Args:
        code: The code to check
        
    Returns:
        True if the code is a valid CPT format, False otherwise
    """
    import re
    
    # CPT format: 5 digits
    pattern = r'^\d{5}$'
    
    return bool(re.match(pattern, code))


def categorize_icd10(code: str) -> str:
    """
    Categorize an ICD-10 code into its chapter/category.
    
    Args:
        code: The ICD-10 code to categorize
        
    Returns:
        The category of the ICD-10 code
    """
    if not is_valid_icd10(code):
        return "Invalid ICD-10 code"
    
    # Extract the first character (letter) from the code
    chapter_letter = code[0]
    
    # ICD-10 chapters
    chapters = {
        'A': 'Certain infectious and parasitic diseases',
        'B': 'Certain infectious and parasitic diseases',
        'C': 'Neoplasms',
        'D': 'Neoplasms / Diseases of the blood and blood-forming organs',
        'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental and behavioral disorders',
        'G': 'Diseases of the nervous system',
        'H': 'Diseases of the eye and adnexa / Diseases of the ear and mastoid process',
        'I': 'Diseases of the circulatory system',
        'J': 'Diseases of the respiratory system',
        'K': 'Diseases of the digestive system',
        'L': 'Diseases of the skin and subcutaneous tissue',
        'M': 'Diseases of the musculoskeletal system and connective tissue',
        'N': 'Diseases of the genitourinary system',
        'O': 'Pregnancy, childbirth and the puerperium',
        'P': 'Certain conditions originating in the perinatal period',
        'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings',
        'S': 'Injury, poisoning and certain other consequences of external causes',
        'T': 'Injury, poisoning and certain other consequences of external causes',
        'V': 'External causes of morbidity',
        'W': 'External causes of morbidity',
        'X': 'External causes of morbidity',
        'Y': 'External causes of morbidity',
        'Z': 'Factors influencing health status and contact with health services'
    }
    
    return chapters.get(chapter_letter, "Unknown chapter")


def categorize_cpt(code: str) -> str:
    """
    Categorize a CPT code into its section/category.
    
    Args:
        code: The CPT code to categorize
        
    Returns:
        The category of the CPT code
    """
    if not is_valid_cpt(code):
        return "Invalid CPT code"
    
    # Convert code to integer for range checking
    code_int = int(code)
    
    # CPT code ranges and their categories
    if 0 <= code_int <= 9999:
        return "Evaluation and Management"
    elif 10000 <= code_int <= 19999:
        return "Anesthesia"
    elif 20000 <= code_int <= 29999:
        return "Surgery (Integumentary System)"
    elif 30000 <= code_int <= 39999:
        return "Surgery (Respiratory, Cardiovascular, Hemic/Lymphatic Systems)"
    elif 40000 <= code_int <= 49999:
        return "Surgery (Digestive System)"
    elif 50000 <= code_int <= 59999:
        return "Surgery (Urinary, Male/Female Genital, Maternity Care Systems)"
    elif 60000 <= code_int <= 69999:
        return "Surgery (Endocrine, Nervous, Eye, Auditory Systems)"
    elif 70000 <= code_int <= 79999:
        return "Radiology"
    elif 80000 <= code_int <= 89999:
        return "Pathology and Laboratory"
    elif 90000 <= code_int <= 99999:
        return "Medicine"
    else:
        return "Unknown category" 