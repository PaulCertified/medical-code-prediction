# ICD-10 & CPT Medical Code Prediction from Clinical Text

This project extracts diagnosis and procedure descriptions from unstructured clinical text and maps them to standardized medical codes (ICD-10 for diagnoses and CPT for procedures) using NLP and transformer models.

## Project Overview

Medical coding is a critical process in healthcare that involves translating clinical documentation into standardized codes for billing, research, and administrative purposes. This project aims to automate this process using state-of-the-art NLP techniques.

### Key Features

- Extract medical conditions and procedures from clinical text
- Predict appropriate ICD-10 and CPT codes with confidence scores
- Provide explanations for code predictions
- Support for batch processing of clinical documents
- AWS SageMaker integration for scalable deployment

## Technical Architecture

- **Data Processing Pipeline**: Preprocessing of clinical text, entity extraction, and code prediction
- **NER Module**: Identification of medical conditions, procedures, and relevant clinical entities
- **Code Prediction Model**: Fine-tuned BioBERT model for medical code prediction
- **Evaluation Module**: Performance metrics and explanation of code predictions
- **AWS Integration**: Deployment to SageMaker for scalable inference

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt
- AWS account (for SageMaker deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-code-prediction.git
cd medical-code-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Local Usage

```bash
# Run the demo script with a sample clinical note
python scripts/demo.py data/raw/sample_clinical_note.txt
```

### API Usage

```bash
# Run the FastAPI application
python scripts/run_api.py
```

Then, you can access the API documentation at http://localhost:8000/docs

### AWS SageMaker Deployment

To deploy the model to AWS SageMaker, follow these steps:

1. Set up your AWS credentials:

```bash
python scripts/setup_aws_credentials.py --access-key YOUR_ACCESS_KEY --secret-key YOUR_SECRET_KEY
```

2. Deploy the model to SageMaker:

```bash
python scripts/deploy_to_sagemaker.py --model-path models/medical_code_prediction.tar.gz --s3-bucket your-bucket-name --role-arn YOUR_ROLE_ARN
```

3. Invoke the SageMaker endpoint:

```bash
python scripts/invoke_endpoint.py --file data/raw/sample_clinical_note.txt --endpoint-name medical-code-prediction
```

For more detailed instructions on AWS integration, see [AWS Integration Guide](docs/aws_integration.md).

## Sample Predictions

### ICD-10 Codes:
- I21.4: Non-ST elevation myocardial infarction (Confidence: 0.92)
- I10: Essential (primary) hypertension (Confidence: 0.89)
- E11.9: Type 2 diabetes mellitus without complications (Confidence: 0.87)
- N18.2: Chronic kidney disease, stage 2 (mild) (Confidence: 0.83)
- K21.9: Gastro-esophageal reflux disease without esophagitis (Confidence: 0.76)

### CPT Codes:
- 93000: Electrocardiogram complete (Confidence: 0.91)
- 93454: Coronary angiography (Confidence: 0.88)
- 71046: Chest X-ray 2 views (Confidence: 0.85)
- 99291: Critical care first hour (Confidence: 0.82)
- 80053: Comprehensive metabolic panel (Confidence: 0.79)

## Project Structure

```
medical-code-prediction/
├── data/                  # Data directory
│   ├── raw/               # Raw clinical text data
│   ├── processed/         # Preprocessed data
│   └── reference/         # ICD-10 and CPT code reference data
├── notebooks/             # Jupyter notebooks for exploration and development
├── src/                   # Source code
│   ├── preprocessing/     # Text preprocessing modules
│   ├── models/            # Model definition and training
│   ├── evaluation/        # Evaluation metrics and analysis
│   ├── api/               # API implementation
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── app/                   # Web interface
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## Future Work

- Implement and train a transformer-based NER model
- Fine-tune BioBERT for medical code prediction
- Develop a more sophisticated confidence scoring mechanism
- Implement comprehensive evaluation metrics
- Create a web interface for the prediction system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-III clinical database
- Hugging Face Transformers library
- BioBERT pre-trained model 