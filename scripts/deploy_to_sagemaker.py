#!/usr/bin/env python
"""
Script to deploy the medical code prediction model to AWS SageMaker.
"""
import os
import sys
import argparse
import json
import time
from typing import Dict, Any

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    load_config,
    get_aws_session,
    create_sagemaker_client,
    create_s3_client,
    upload_to_s3,
    create_sagemaker_model,
    create_sagemaker_endpoint_config,
    create_sagemaker_endpoint,
    get_endpoint_status
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy medical code prediction model to AWS SageMaker.')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model artifacts')
    parser.add_argument('--s3-bucket', type=str, required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--s3-prefix', type=str, default='medical-code-prediction', help='S3 key prefix')
    parser.add_argument('--model-name', type=str, default='medical-code-prediction', help='Name for the SageMaker model')
    parser.add_argument('--endpoint-name', type=str, default='medical-code-prediction', help='Name for the SageMaker endpoint')
    parser.add_argument('--instance-type', type=str, default='ml.m5.xlarge', help='Instance type for deployment')
    parser.add_argument('--instance-count', type=int, default=1, help='Number of instances')
    parser.add_argument('--role-arn', type=str, required=True, help='ARN of the IAM role for SageMaker')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    
    return parser.parse_args()


def upload_model_to_s3(model_path: str, bucket: str, prefix: str) -> str:
    """
    Upload model artifacts to S3.
    
    Args:
        model_path: Path to the model artifacts
        bucket: S3 bucket name
        prefix: S3 key prefix
        
    Returns:
        str: S3 URI for the uploaded model artifacts
    """
    # Create a unique S3 key for the model
    model_name = os.path.basename(model_path)
    s3_key = f"{prefix}/models/{model_name}"
    
    # Upload the model to S3
    print(f"Uploading model to s3://{bucket}/{s3_key}...")
    upload_to_s3(model_path, bucket, s3_key)
    
    # Return the S3 URI
    return f"s3://{bucket}/{s3_key}"


def deploy_model(model_name: str, model_s3_uri: str, image_uri: str, role_arn: str,
                instance_type: str, instance_count: int, endpoint_name: str,
                region: str) -> Dict[str, Any]:
    """
    Deploy a model to SageMaker.
    
    Args:
        model_name: Name for the SageMaker model
        model_s3_uri: S3 URI for the model artifacts
        image_uri: Docker image URI for the model
        role_arn: ARN of the IAM role for SageMaker
        instance_type: Instance type for deployment
        instance_count: Number of instances
        endpoint_name: Name for the SageMaker endpoint
        region: AWS region
        
    Returns:
        Dict: Deployment information
    """
    # Create a SageMaker client
    sagemaker_client = create_sagemaker_client(region_name=region)
    
    # Create a SageMaker model
    print(f"Creating SageMaker model: {model_name}...")
    create_sagemaker_model(
        model_name=model_name,
        model_artifacts=model_s3_uri,
        image_uri=image_uri,
        role_arn=role_arn,
        sagemaker_client=sagemaker_client
    )
    
    # Create an endpoint configuration
    config_name = f"{model_name}-config"
    print(f"Creating endpoint configuration: {config_name}...")
    create_sagemaker_endpoint_config(
        config_name=config_name,
        model_name=model_name,
        instance_type=instance_type,
        instance_count=instance_count,
        sagemaker_client=sagemaker_client
    )
    
    # Create an endpoint
    print(f"Creating endpoint: {endpoint_name}...")
    create_sagemaker_endpoint(
        endpoint_name=endpoint_name,
        config_name=config_name,
        sagemaker_client=sagemaker_client
    )
    
    # Wait for the endpoint to be in service
    print("Waiting for endpoint to be in service...")
    status = "Creating"
    while status != "InService":
        time.sleep(30)
        status = get_endpoint_status(endpoint_name, sagemaker_client)
        print(f"Endpoint status: {status}")
        
        if status == "Failed":
            raise Exception("Endpoint creation failed")
    
    return {
        "model_name": model_name,
        "endpoint_name": endpoint_name,
        "status": status
    }


def get_image_uri(region: str) -> str:
    """
    Get the Docker image URI for the model.
    
    Args:
        region: AWS region
        
    Returns:
        str: Docker image URI
    """
    # For a custom container, you would use your own ECR repository
    # For this example, we'll use a pre-built SageMaker container for PyTorch
    account_id_map = {
        'us-east-1': '763104351884',
        'us-east-2': '763104351884',
        'us-west-1': '763104351884',
        'us-west-2': '763104351884',
        'eu-west-1': '763104351884',
        'eu-central-1': '763104351884',
        'ap-northeast-1': '763104351884',
        'ap-northeast-2': '763104351884',
        'ap-southeast-1': '763104351884',
        'ap-southeast-2': '763104351884'
    }
    
    account_id = account_id_map.get(region, '763104351884')
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:1.8.1-cpu-py36-ubuntu18.04"


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Upload model to S3
    model_s3_uri = upload_model_to_s3(args.model_path, args.s3_bucket, args.s3_prefix)
    
    # Get the Docker image URI
    image_uri = get_image_uri(args.region)
    
    # Deploy the model
    deployment_info = deploy_model(
        model_name=args.model_name,
        model_s3_uri=model_s3_uri,
        image_uri=image_uri,
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        endpoint_name=args.endpoint_name,
        region=args.region
    )
    
    # Print deployment information
    print("\nDeployment Information:")
    print(json.dumps(deployment_info, indent=2))
    
    print("\nModel successfully deployed to SageMaker!")
    print(f"Endpoint Name: {args.endpoint_name}")
    print(f"Region: {args.region}")
    print("\nYou can now use this endpoint to make predictions.")


if __name__ == "__main__":
    main() 