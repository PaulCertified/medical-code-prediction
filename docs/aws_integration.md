# AWS Integration for Medical Code Prediction

This document provides instructions for integrating the Medical Code Prediction system with AWS services, particularly SageMaker for model deployment.

## Prerequisites

Before you begin, make sure you have:

1. An AWS account with appropriate permissions
2. AWS CLI installed and configured
3. Python 3.8+ with boto3 installed

## Setting Up AWS Credentials

You can set up your AWS credentials using the provided script:

```bash
python scripts/setup_aws_credentials.py --access-key YOUR_ACCESS_KEY --secret-key YOUR_SECRET_KEY --region us-west-2
```

Alternatively, you can set up your credentials manually:

1. Create the AWS credentials file:

```bash
mkdir -p ~/.aws
touch ~/.aws/credentials
touch ~/.aws/config
```

2. Add your credentials to `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

3. Add your region to `~/.aws/config`:

```ini
[default]
region = us-west-2
```

## IAM Role for SageMaker

You need to create an IAM role that SageMaker can assume to access other AWS services:

1. Go to the IAM console in AWS
2. Create a new role with the following permissions:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess
   - AmazonECR-FullAccess
3. Note the ARN of the role (e.g., `arn:aws:iam::123456789012:role/SageMakerRole`)

## Deploying the Model to SageMaker

To deploy your trained model to SageMaker, use the provided script:

```bash
python scripts/deploy_to_sagemaker.py \
  --model-path models/medical_code_prediction.tar.gz \
  --s3-bucket your-bucket-name \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --region us-west-2
```

This script will:
1. Upload your model artifacts to S3
2. Create a SageMaker model
3. Create an endpoint configuration
4. Create and deploy an endpoint

## Invoking the SageMaker Endpoint

Once your model is deployed, you can invoke it using the provided script:

```bash
python scripts/invoke_endpoint.py \
  --file data/raw/sample_clinical_note.txt \
  --endpoint-name medical-code-prediction \
  --region us-west-2 \
  --icd10-codes data/reference/icd10_codes.csv \
  --cpt-codes data/reference/cpt_codes.csv
```

## Running the API with SageMaker Integration

To run the FastAPI application with SageMaker integration:

```bash
python scripts/run_api.py \
  --endpoint-name medical-code-prediction \
  --region us-west-2 \
  --icd10-codes data/reference/icd10_codes.csv \
  --cpt-codes data/reference/cpt_codes.csv
```

## Monitoring and Maintenance

### CloudWatch Logs

SageMaker automatically logs to CloudWatch. You can view these logs in the CloudWatch console:

1. Go to the CloudWatch console
2. Navigate to Logs > Log groups
3. Look for `/aws/sagemaker/Endpoints/medical-code-prediction`

### Updating the Model

To update your model with a new version:

1. Train a new model
2. Package the model artifacts
3. Deploy the new model using the same script as above, but with a new model path

### Cleaning Up Resources

To avoid incurring unnecessary charges, remember to clean up your resources when they're no longer needed:

```bash
# Delete the SageMaker endpoint
aws sagemaker delete-endpoint --endpoint-name medical-code-prediction

# Delete the endpoint configuration
aws sagemaker delete-endpoint-config --endpoint-config-name medical-code-prediction-config

# Delete the SageMaker model
aws sagemaker delete-model --model-name medical-code-prediction
```

## Troubleshooting

### Common Issues

1. **Insufficient permissions**: Make sure your IAM role has the necessary permissions.
2. **Model artifacts not found**: Ensure your model artifacts are correctly uploaded to S3.
3. **Endpoint creation failure**: Check the CloudWatch logs for error messages.

### Getting Help

If you encounter issues, check:
1. SageMaker documentation: https://docs.aws.amazon.com/sagemaker/
2. AWS boto3 documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
3. Open an issue in the project repository