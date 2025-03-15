#!/usr/bin/env python
"""
Simple script to demonstrate medical code prediction.
"""
import os
import sys
import csv
import random
from typing import Dict, List, Any

# Sample medical terms to look for
MEDICAL_TERMS = [
    "acute coronary syndrome", "myocardial infarction", "nstemi", "stemi",
    "hypertension", "high blood pressure",
    "diabetes", "type 2 diabetes",
    "chronic kidney disease",
    "heart failure",
    "chest pain",
    "shortness of breath", "dyspnea",
    "ecg", "ekg", "electrocardiogram",
    "echocardiogram",
    "cardiac catheterization", "coronary angiography",
    "chest x-ray",
    "troponin", "cardiac enzymes",
    "aspirin", "clopidogrel",
    "atorvastatin",
    "metoprolol",
    "heparin"
]

def load_codes(file_path):
    """Load codes from a CSV file."""
    codes = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    code, description = row[0], row[1]
                    codes[code] = description
        print(f"Loaded {len(codes)} codes from {file_path}")
    except Exception as e:
        print(f"Error loading codes: {e}")
    return codes

def extract_key_terms(text):
    """Extract key medical terms from text."""
    text_lower = text.lower()
    matches = []
    for term in MEDICAL_TERMS:
        if term in text_lower:
            matches.append(term)
    return matches

def predict_codes(text, icd10_codes, cpt_codes):
    """Predict ICD-10 and CPT codes from text."""
    key_terms = extract_key_terms(text)
    predictions = []
    
    # Predict ICD-10 codes
    if "acute coronary syndrome" in key_terms or "nstemi" in key_terms:
        predictions.append({
            "code": "I21.4",
            "description": icd10_codes.get("I21.4", "Non-ST elevation myocardial infarction"),
            "confidence": random.uniform(0.85, 0.95),
            "type": "ICD-10"
        })
    
    if "hypertension" in key_terms or "high blood pressure" in key_terms:
        predictions.append({
            "code": "I10",
            "description": icd10_codes.get("I10", "Essential (primary) hypertension"),
            "confidence": random.uniform(0.80, 0.90),
            "type": "ICD-10"
        })
    
    if "diabetes" in key_terms or "type 2 diabetes" in key_terms:
        predictions.append({
            "code": "E11.9",
            "description": icd10_codes.get("E11.9", "Type 2 diabetes mellitus without complications"),
            "confidence": random.uniform(0.80, 0.90),
            "type": "ICD-10"
        })
    
    if "chronic kidney disease" in key_terms:
        predictions.append({
            "code": "N18.9",
            "description": icd10_codes.get("N18.9", "Chronic kidney disease, unspecified"),
            "confidence": random.uniform(0.75, 0.85),
            "type": "ICD-10"
        })
    
    # Predict CPT codes
    if "ecg" in key_terms or "ekg" in key_terms or "electrocardiogram" in key_terms:
        predictions.append({
            "code": "93000",
            "description": cpt_codes.get("93000", "Electrocardiogram complete"),
            "confidence": random.uniform(0.80, 0.90),
            "type": "CPT"
        })
    
    if "cardiac catheterization" in key_terms or "coronary angiography" in key_terms:
        predictions.append({
            "code": "93454",
            "description": cpt_codes.get("93454", "Coronary angiography"),
            "confidence": random.uniform(0.85, 0.95),
            "type": "CPT"
        })
    
    if "chest x-ray" in key_terms:
        predictions.append({
            "code": "71046",
            "description": cpt_codes.get("71046", "Chest X-ray 2 views"),
            "confidence": random.uniform(0.75, 0.85),
            "type": "CPT"
        })
    
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    return predictions

def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python simple_predict.py <clinical_note_file> <icd10_codes_file> <cpt_codes_file>")
        return
    
    clinical_note_file = sys.argv[1]
    icd10_codes_file = sys.argv[2]
    cpt_codes_file = sys.argv[3]
    
    # Load codes
    icd10_codes = load_codes(icd10_codes_file)
    cpt_codes = load_codes(cpt_codes_file)
    
    # Load clinical note
    try:
        with open(clinical_note_file, 'r') as f:
            clinical_text = f.read()
    except Exception as e:
        print(f"Error reading clinical note: {e}")
        return
    
    # Extract key terms
    key_terms = extract_key_terms(clinical_text)
    print("\nExtracted Key Terms:")
    for term in key_terms:
        print(f"  - {term}")
    
    # Predict codes
    print("\nPredicting codes...")
    predictions = predict_codes(clinical_text, icd10_codes, cpt_codes)
    
    # Print predictions
    print("\nPredicted Codes:")
    for pred in predictions:
        print(f"  - {pred['code']} ({pred['type']}): {pred['description']} (Confidence: {pred['confidence']:.2f})")

if __name__ == "__main__":
    main() 