#!/usr/bin/env python
"""
Script to invoke the SageMaker endpoint for medical code predictions.
"""
import os
import sys
import argparse
import json
from typing import Dict, Any, List

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config,
    load_icd10_codes,
    load_cpt_codes,
    get_aws_session,
    invoke_endpoint
)
from src.preprocessing import preprocess_text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Invoke SageMaker endpoint for medical code predictions.')
    parser.add_argument('--text', type=str, help='Clinical text to predict codes for')
    parser.add_argument('--file', type=str, help='Path to a file containing clinical text')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--endpoint-name', type=str, required=True, help='Name of the SageMaker endpoint')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    parser.add_argument('--icd10-codes', type=str, help='Path to ICD-10 codes reference file')
    parser.add_argument('--cpt-codes', type=str, help='Path to CPT codes reference file')
    parser.add_argument('--output', type=str, help='Path to save the predictions to')
    
    return parser.parse_args()


def format_predictions(predictions: List[Dict[str, Any]], 
                      icd10_codes: Dict[str, str] = None,
                      cpt_codes: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Format the predictions with code descriptions.
    
    Args:
        predictions: List of prediction dictionaries
        icd10_codes: Dictionary mapping ICD-10 codes to descriptions
        cpt_codes: Dictionary mapping CPT codes to descriptions
        
    Returns:
        List of formatted prediction dictionaries
    """
    formatted_predictions = []
    
    for pred in predictions:
        code = pred['code']
        code_type = pred['type']
        
        # Get the description from the appropriate code dictionary
        if code_type == 'ICD-10' and icd10_codes and code in icd10_codes:
            description = icd10_codes[code]
        elif code_type == 'CPT' and cpt_codes and code in cpt_codes:
            description = cpt_codes[code]
        else:
            description = pred.get('description', 'Unknown')
        
        # Create a formatted prediction
        formatted_pred = {
            'code': code,
            'type': code_type,
            'description': description,
            'confidence': pred['confidence']
        }
        
        formatted_predictions.append(formatted_pred)
    
    return formatted_predictions


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Load code dictionaries if provided
    icd10_codes = None
    cpt_codes = None
    
    if args.icd10_codes:
        icd10_codes = load_icd10_codes(args.icd10_codes)
        print(f"Loaded {len(icd10_codes)} ICD-10 codes from {args.icd10_codes}")
    
    if args.cpt_codes:
        cpt_codes = load_cpt_codes(args.cpt_codes)
        print(f"Loaded {len(cpt_codes)} CPT codes from {args.cpt_codes}")
    
    # Get clinical text
    if args.text:
        clinical_text = args.text
    elif args.file:
        try:
            with open(args.file, "r") as f:
                clinical_text = f.read()
            print(f"Loaded clinical text from {args.file}")
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
    
    # Create the input data for the endpoint
    input_data = {
        "text": preprocessed_text,
        "threshold": config["prediction"]["threshold"],
        "top_k": config["prediction"]["top_k"],
        "code_type": "both"  # Predict both ICD-10 and CPT codes
    }
    
    # Convert the input data to JSON
    input_json = json.dumps(input_data)
    
    # Invoke the endpoint
    print(f"Invoking endpoint {args.endpoint_name}...")
    try:
        # Create an AWS session
        session = get_aws_session(region_name=args.region)
        
        # Invoke the endpoint
        response = invoke_endpoint(
            endpoint_name=args.endpoint_name,
            input_data=input_json,
            session=session
        )
        
        # Parse the response
        if response['statusCode'] == 200:
            predictions = json.loads(response['body'])
            
            # Format the predictions
            formatted_predictions = format_predictions(
                predictions,
                icd10_codes=icd10_codes,
                cpt_codes=cpt_codes
            )
            
            # Print the predictions
            print("\nPredicted Codes:")
            for pred in formatted_predictions:
                print(f"  - {pred['code']} ({pred['type']}): {pred['description']} (Confidence: {pred['confidence']:.2f})")
            
            # Save the predictions if an output path is provided
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(formatted_predictions, f, indent=2)
                print(f"Saved predictions to {args.output}")
        else:
            print(f"Error: {response}")
    except Exception as e:
        print(f"Error invoking endpoint: {e}")


if __name__ == "__main__":
    main() 