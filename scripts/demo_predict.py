#!/usr/bin/env python
"""
Demo script for medical code prediction.
"""
import sys
import csv
import random

def load_codes(file_path):
    """Load codes from a CSV file."""
    codes = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                code, description = row[0], row[1]
                codes[code] = description
    print(f"Loaded {len(codes)} codes from {file_path}")
    return codes

def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: python demo_predict.py <clinical_note_file> <icd10_codes_file> <cpt_codes_file>")
        sys.exit(1)
    
    clinical_note_file = sys.argv[1]
    icd10_codes_file = sys.argv[2]
    cpt_codes_file = sys.argv[3]
    
    # Load codes
    icd10_codes = load_codes(icd10_codes_file)
    cpt_codes = load_codes(cpt_codes_file)
    
    # Load clinical note
    with open(clinical_note_file, 'r') as f:
        clinical_text = f.read()
    
    print("\nClinical note loaded. Length:", len(clinical_text), "characters")
    
    # Simulate predictions
    print("\nPredicting codes...")
    
    # Predefined predictions based on the sample clinical note
    predictions = [
        {
            "code": "I21.4",
            "description": icd10_codes.get("I21.4", "Non-ST elevation myocardial infarction"),
            "confidence": 0.92,
            "type": "ICD-10"
        },
        {
            "code": "I10",
            "description": icd10_codes.get("I10", "Essential (primary) hypertension"),
            "confidence": 0.89,
            "type": "ICD-10"
        },
        {
            "code": "E11.9",
            "description": icd10_codes.get("E11.9", "Type 2 diabetes mellitus without complications"),
            "confidence": 0.87,
            "type": "ICD-10"
        },
        {
            "code": "N18.2",
            "description": icd10_codes.get("N18.2", "Chronic kidney disease, stage 2 (mild)"),
            "confidence": 0.83,
            "type": "ICD-10"
        },
        {
            "code": "K21.9",
            "description": icd10_codes.get("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
            "confidence": 0.76,
            "type": "ICD-10"
        },
        {
            "code": "93000",
            "description": cpt_codes.get("93000", "Electrocardiogram complete"),
            "confidence": 0.91,
            "type": "CPT"
        },
        {
            "code": "93454",
            "description": cpt_codes.get("93454", "Coronary angiography"),
            "confidence": 0.88,
            "type": "CPT"
        },
        {
            "code": "71046",
            "description": cpt_codes.get("71046", "Chest X-ray 2 views"),
            "confidence": 0.85,
            "type": "CPT"
        },
        {
            "code": "99291",
            "description": cpt_codes.get("99291", "Critical care first hour"),
            "confidence": 0.82,
            "type": "CPT"
        },
        {
            "code": "80053",
            "description": cpt_codes.get("80053", "Comprehensive metabolic panel"),
            "confidence": 0.79,
            "type": "CPT"
        }
    ]
    
    # Print predictions
    print("\nPredicted Codes:")
    for pred in predictions:
        print(f"  - {pred['code']} ({pred['type']}): {pred['description']} (Confidence: {pred['confidence']:.2f})")
    
    # Explain top prediction
    top_pred = predictions[0]
    print(f"\nExplanation for top prediction ({top_pred['code']}):")
    print(f"  Code: {top_pred['code']}")
    print(f"  Description: {top_pred['description']}")
    print(f"  Confidence: {top_pred['confidence']:.2f}")
    
    print("  Relevant text segments:")
    print('    - "Mr. Smith is a 68-year-old male with a history of hypertension, hyperlipidemia, and type 2 diabetes mellitus who presents to the emergency department with a 2-hour history of substernal chest pain and shortness of breath."')
    print('    - "ECG: Normal sinus rhythm, 1mm ST depression in leads V3-V6, T wave inversions in leads II, III, aVF"')
    print('    - "Assessment and Plan: 1. Acute coronary syndrome, likely NSTEMI"')
    
    print("  Feature importance:")
    print("    - ECG findings: 0.35")
    print("    - Chest pain description: 0.30")
    print("    - Troponin elevation: 0.25")
    print("    - Risk factors: 0.10")

if __name__ == "__main__":
    main() 