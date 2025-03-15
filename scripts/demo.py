#!/usr/bin/env python

"""
Simple demo script for medical code prediction.
"""

import sys

def main():
    """Main function."""
    print("Medical Code Prediction Demo")
    print("============================")
    
    if len(sys.argv) < 2:
        print("Usage: python demo.py <clinical_note_file>")
        return
    
    clinical_note_file = sys.argv[1]
    
    # Load clinical note
    try:
        with open(clinical_note_file, "r") as f:
            clinical_text = f.read()
        print(f"\nLoaded clinical note from {clinical_note_file}")
        print(f"Length: {len(clinical_text)} characters")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Simulate predictions
    print("\nPredicting codes...")
    
    # Predefined predictions
    predictions = [
        {"code": "I21.4", "description": "Non-ST elevation myocardial infarction", "confidence": 0.92, "type": "ICD-10"},
        {"code": "I10", "description": "Essential (primary) hypertension", "confidence": 0.89, "type": "ICD-10"},
        {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "confidence": 0.87, "type": "ICD-10"},
        {"code": "93000", "description": "Electrocardiogram complete", "confidence": 0.91, "type": "CPT"},
        {"code": "93454", "description": "Coronary angiography", "confidence": 0.88, "type": "CPT"}
    ]
    
    # Print predictions
    print("\nPredicted Codes:")
    for pred in predictions:
        print(f"  - {pred['code']} ({pred['type']}): {pred['description']} (Confidence: {pred['confidence']:.2f})")

if __name__ == "__main__":
    main() 