#!/usr/bin/env python
"""
Script to predict ICD-10 and CPT codes from clinical text.
"""
import os
import sys
import argparse
from typing import Dict, List, Optional

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_text, extract_entities
from src.models import CodePredictionModel
from src.utils import load_config, load_icd10_codes, load_cpt_codes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict ICD-10 and CPT codes from clinical text.')
    parser.add_argument('--text', type=str, help='Clinical text to predict codes for')
    parser.add_argument('--file', type=str, help='Path to a file containing clinical text')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, help='Path to a trained model')
    parser.add_argument('--icd10_codes', type=str, help='Path to ICD-10 codes reference file')
    parser.add_argument('--cpt_codes', type=str, help='Path to CPT codes reference file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--code_type', type=str, default='both', choices=['icd10', 'cpt', 'both'], help='Type of codes to predict')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Get clinical text
    if args.text:
        clinical_text = args.text
    elif args.file:
        try:
            with open(args.file, "r") as f:
                clinical_text = f.read()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            return
    else:
        print("Please provide either --text or --file")
        return
    
    # Preprocess the text
    print("Preprocessing clinical text...")
    preprocessed_text = preprocess_text(
        clinical_text,
        lowercase=config["preprocessing"]["lowercase"],
        remove_punct=config["preprocessing"]["remove_punctuation"],
        expand_abbrev=config["preprocessing"]["expand_abbreviations"]
    )
    
    # Extract entities
    print("Extracting medical entities...")
    entities = extract_entities(preprocessed_text, entity_types=config["ner"]["labels"])
    
    # Print extracted entities
    print("\nExtracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"\n{entity_type}:")
            for entity in entity_list:
                print(f"  - {entity}")
    
    # Initialize the model
    print("\nInitializing code prediction model...")
    model = CodePredictionModel(
        model_name=config["model"]["name"],
        max_length=config["model"]["max_length"],
        icd10_codes_path=args.icd10_codes,
        cpt_codes_path=args.cpt_codes
    )
    
    # Load a trained model if specified
    if args.model_path:
        model.load(args.model_path)
    
    # Predict codes
    print("Predicting codes...")
    predictions = model.predict(
        preprocessed_text,
        threshold=args.threshold,
        top_k=args.top_k,
        code_type=args.code_type
    )
    
    # Print predictions
    print("\nPredicted Codes:")
    for pred in predictions:
        print(f"  - {pred['code']} ({pred['type']}): {pred['description']} (Confidence: {pred['confidence']:.2f})")
    
    # Explain the top prediction if available
    if predictions:
        top_code = predictions[0]['code']
        print(f"\nExplanation for top prediction ({top_code}):")
        explanation = model.explain(preprocessed_text, top_code)
        
        print(f"  Code: {explanation['code']}")
        print(f"  Description: {explanation['description']}")
        print(f"  Confidence: {explanation['confidence']:.2f}")
        
        print("  Relevant text segments:")
        for segment in explanation['relevant_text']:
            print(f'    - "{segment}"')
        
        print("  Feature importance:")
        for feature, importance in explanation['feature_importance'].items():
            print(f"    - {feature}: {importance:.2f}")


if __name__ == "__main__":
    main() 