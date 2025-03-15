"""
AWS utility functions for SageMaker deployment and S3 integration.
"""
import os
import boto3
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_aws_session(region_name: Optional[str] = None) -> boto3.Session:
    """
    Get an AWS session using the configured credentials.
    
    Args:
        region_name: AWS region name. If None, uses the default from AWS config
                    or environment variables.
    
    Returns:
        boto3.Session: Configured AWS session
    """
    try:
        # This will use credentials from:
        # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # 2. Shared credential file (~/.aws/credentials)
        # 3. EC2/ECS IAM role
        session = boto3.Session(region_name=region_name)
        return session
    except Exception as e:
        logger.error(f"Error creating AWS session: {e}")
        raise


def create_sagemaker_client(session: Optional[boto3.Session] = None, 
                           region_name: Optional[str] = None) -> boto3.client:
    """
    Create a SageMaker client.
    
    Args:
        session: AWS session. If None, creates a new session.
        region_name: AWS region name. If None, uses the session's region.
    
    Returns:
        boto3.client: SageMaker client
    """
    if session is None:
        session = get_aws_session(region_name)
    
    return session.client('sagemaker')


def create_s3_client(session: Optional[boto3.Session] = None,
                    region_name: Optional[str] = None) -> boto3.client:
    """
    Create an S3 client.
    
    Args:
        session: AWS session. If None, creates a new session.
        region_name: AWS region name. If None, uses the session's region.
    
    Returns:
        boto3.client: S3 client
    """
    if session is None:
        session = get_aws_session(region_name)
    
    return session.client('s3')


def upload_to_s3(local_path: str, bucket: str, s3_key: str, 
                session: Optional[boto3.Session] = None) -> bool:
    """
    Upload a file to S3.
    
    Args:
        local_path: Path to the local file
        bucket: S3 bucket name
        s3_key: S3 object key (path within the bucket)
        session: AWS session. If None, creates a new session.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        s3 = create_s3_client(session)
        s3.upload_file(local_path, bucket, s3_key)
        logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_path} to S3: {e}")
        return False


def download_from_s3(bucket: str, s3_key: str, local_path: str,
                    session: Optional[boto3.Session] = None) -> bool:
    """
    Download a file from S3.
    
    Args:
        bucket: S3 bucket name
        s3_key: S3 object key (path within the bucket)
        local_path: Path to save the file locally
        session: AWS session. If None, creates a new session.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        
        s3 = create_s3_client(session)
        s3.download_file(bucket, s3_key, local_path)
        logger.info(f"Successfully downloaded s3://{bucket}/{s3_key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return False


def create_sagemaker_model(model_name: str, model_artifacts: str, 
                          image_uri: str, role_arn: str,
                          env_vars: Optional[Dict[str, str]] = None,
                          sagemaker_client: Optional[boto3.client] = None) -> Dict[str, Any]:
    """
    Create a SageMaker model.
    
    Args:
        model_name: Name for the SageMaker model
        model_artifacts: S3 path to model artifacts (tar.gz file)
        image_uri: Docker image URI for model deployment
        role_arn: ARN of the IAM role for SageMaker
        env_vars: Environment variables for the model container
        sagemaker_client: SageMaker client. If None, creates a new client.
    
    Returns:
        Dict: Response from SageMaker CreateModel API
    """
    if sagemaker_client is None:
        sagemaker_client = create_sagemaker_client()
    
    # Default environment variables
    if env_vars is None:
        env_vars = {}
    
    try:
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_artifacts,
                'Environment': env_vars
            },
            ExecutionRoleArn=role_arn
        )
        logger.info(f"Successfully created SageMaker model: {model_name}")
        return response
    except Exception as e:
        logger.error(f"Error creating SageMaker model: {e}")
        raise


def create_sagemaker_endpoint_config(config_name: str, model_name: str,
                                    instance_type: str, instance_count: int = 1,
                                    sagemaker_client: Optional[boto3.client] = None) -> Dict[str, Any]:
    """
    Create a SageMaker endpoint configuration.
    
    Args:
        config_name: Name for the endpoint configuration
        model_name: Name of the SageMaker model
        instance_type: Type of instance for deployment (e.g., 'ml.m5.xlarge')
        instance_count: Number of instances
        sagemaker_client: SageMaker client. If None, creates a new client.
    
    Returns:
        Dict: Response from SageMaker CreateEndpointConfig API
    """
    if sagemaker_client is None:
        sagemaker_client = create_sagemaker_client()
    
    try:
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count
                }
            ]
        )
        logger.info(f"Successfully created endpoint configuration: {config_name}")
        return response
    except Exception as e:
        logger.error(f"Error creating endpoint configuration: {e}")
        raise


def create_sagemaker_endpoint(endpoint_name: str, config_name: str,
                             sagemaker_client: Optional[boto3.client] = None) -> Dict[str, Any]:
    """
    Create a SageMaker endpoint.
    
    Args:
        endpoint_name: Name for the endpoint
        config_name: Name of the endpoint configuration
        sagemaker_client: SageMaker client. If None, creates a new client.
    
    Returns:
        Dict: Response from SageMaker CreateEndpoint API
    """
    if sagemaker_client is None:
        sagemaker_client = create_sagemaker_client()
    
    try:
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        logger.info(f"Creating endpoint: {endpoint_name}")
        return response
    except Exception as e:
        logger.error(f"Error creating endpoint: {e}")
        raise


def get_endpoint_status(endpoint_name: str,
                       sagemaker_client: Optional[boto3.client] = None) -> str:
    """
    Get the status of a SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        sagemaker_client: SageMaker client. If None, creates a new client.
    
    Returns:
        str: Endpoint status
    """
    if sagemaker_client is None:
        sagemaker_client = create_sagemaker_client()
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except Exception as e:
        logger.error(f"Error getting endpoint status: {e}")
        raise


def invoke_endpoint(endpoint_name: str, input_data: str,
                   content_type: str = 'application/json',
                   accept: str = 'application/json',
                   session: Optional[boto3.Session] = None) -> Dict[str, Any]:
    """
    Invoke a SageMaker endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        input_data: Input data for the model
        content_type: Content type of the input data
        accept: Expected content type of the output
        session: AWS session. If None, creates a new session.
    
    Returns:
        Dict: Response from the endpoint
    """
    if session is None:
        session = get_aws_session()
    
    runtime = session.client('sagemaker-runtime')
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Accept=accept,
            Body=input_data
        )
        
        # Parse the response body
        result = response['Body'].read().decode('utf-8')
        return {
            'statusCode': response['ResponseMetadata']['HTTPStatusCode'],
            'body': result
        }
    except Exception as e:
        logger.error(f"Error invoking endpoint: {e}")
        raise 