#!/usr/bin/env python
"""
Script to run the FastAPI application for medical code prediction.
"""
import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the FastAPI application for medical code prediction.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--icd10-codes', type=str, help='Path to ICD-10 codes reference file')
    parser.add_argument('--cpt-codes', type=str, help='Path to CPT codes reference file')
    parser.add_argument('--endpoint-name', type=str, help='Name of the SageMaker endpoint')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set environment variables for the API
    os.environ['CONFIG_PATH'] = args.config
    
    if args.icd10_codes:
        os.environ['ICD10_CODES_PATH'] = args.icd10_codes
    
    if args.cpt_codes:
        os.environ['CPT_CODES_PATH'] = args.cpt_codes
    
    if args.endpoint_name:
        os.environ['ENDPOINT_NAME'] = args.endpoint_name
    
    os.environ['AWS_REGION'] = args.region
    os.environ['API_HOST'] = args.host
    os.environ['API_PORT'] = str(args.port)
    
    # Run the API
    import uvicorn
    
    print(f"Starting API on {args.host}:{args.port}")
    print(f"Configuration: {args.config}")
    
    if args.endpoint_name:
        print(f"Using SageMaker endpoint: {args.endpoint_name} in region {args.region}")
    else:
        print("No SageMaker endpoint specified. Using local model for demo purposes.")
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 