"""
FastAPI application for medical code prediction.
"""
import os
import sys
import json
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from src.preprocessing import preprocess_text, extract_entities
from src.utils import (
    load_config, 
    load_icd10_codes, 
    load_cpt_codes,
    get_aws_session,
    invoke_endpoint
)

# Create the FastAPI app
app = FastAPI(
    title="Medical Code Prediction API",
    description="API for predicting ICD-10 and CPT codes from clinical text",
    version="1.0.0"
)

# Define the request and response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Clinical text to predict codes for")
    threshold: float = Field(0.5, description="Confidence threshold for predictions")
    top_k: int = Field(5, description="Number of top predictions to return")
    code_type: str = Field("both", description="Type of codes to predict (icd10, cpt, or both)")

class CodePrediction(BaseModel):
    code: str = Field(..., description="The predicted code")
    type: str = Field(..., description="The type of code (ICD-10 or CPT)")
    description: str = Field(..., description="Description of the code")
    confidence: float = Field(..., description="Confidence score for the prediction")

class PredictionResponse(BaseModel):
    predictions: List[CodePrediction] = Field(..., description="List of predicted codes")
    entities: Dict[str, List[str]] = Field({}, description="Extracted medical entities")

class EntityExtractionRequest(BaseModel):
    text: str = Field(..., description="Clinical text to extract entities from")
    entity_types: Optional[List[str]] = Field(None, description="Types of entities to extract")

class EntityExtractionResponse(BaseModel):
    entities: Dict[str, List[str]] = Field(..., description="Extracted medical entities")

# Global variables for configuration and code dictionaries
config = None
icd10_codes = None
cpt_codes = None
endpoint_name = None
aws_region = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global config, icd10_codes, cpt_codes, endpoint_name, aws_region
    
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    config = load_config(config_path)
    
    # Load code dictionaries
    icd10_path = os.environ.get("ICD10_CODES_PATH", config["paths"]["icd10_codes"])
    cpt_path = os.environ.get("CPT_CODES_PATH", config["paths"]["cpt_codes"])
    
    icd10_codes = load_icd10_codes(icd10_path)
    cpt_codes = load_cpt_codes(cpt_path)
    
    # Get endpoint name and region from environment variables
    endpoint_name = os.environ.get("ENDPOINT_NAME")
    aws_region = os.environ.get("AWS_REGION", "us-west-2")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Medical Code Prediction API"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict medical codes from clinical text.
    
    This endpoint preprocesses the input text, extracts medical entities,
    and predicts ICD-10 and CPT codes using the deployed SageMaker model.
    """
    global config, icd10_codes, cpt_codes, endpoint_name, aws_region
    
    # Check if the endpoint name is configured
    if not endpoint_name:
        # If no endpoint is configured, use the local model for demo purposes
        return predict_local(request)
    
    # Preprocess the text
    preprocessed_text = preprocess_text(
        request.text,
        lowercase=config["preprocessing"]["lowercase"],
        remove_punct=config["preprocessing"]["remove_punctuation"],
        expand_abbrev=config["preprocessing"]["expand_abbreviations"]
    )
    
    # Extract entities
    entities = extract_entities(
        preprocessed_text,
        entity_types=config["ner"]["labels"]
    )
    
    # Create the input data for the endpoint
    input_data = {
        "text": preprocessed_text,
        "threshold": request.threshold,
        "top_k": request.top_k,
        "code_type": request.code_type
    }
    
    # Convert the input data to JSON
    input_json = json.dumps(input_data)
    
    try:
        # Create an AWS session
        session = get_aws_session(region_name=aws_region)
        
        # Invoke the endpoint
        response = invoke_endpoint(
            endpoint_name=endpoint_name,
            input_data=input_json,
            session=session
        )
        
        # Parse the response
        if response['statusCode'] == 200:
            predictions_data = json.loads(response['body'])
            
            # Format the predictions
            predictions = []
            for pred in predictions_data:
                code = pred['code']
                code_type = pred['type']
                
                # Get the description from the appropriate code dictionary
                if code_type == 'ICD-10' and code in icd10_codes:
                    description = icd10_codes[code]
                elif code_type == 'CPT' and code in cpt_codes:
                    description = cpt_codes[code]
                else:
                    description = pred.get('description', 'Unknown')
                
                predictions.append(
                    CodePrediction(
                        code=code,
                        type=code_type,
                        description=description,
                        confidence=pred['confidence']
                    )
                )
            
            return PredictionResponse(predictions=predictions, entities=entities)
        else:
            raise HTTPException(status_code=500, detail=f"Error from SageMaker endpoint: {response}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking endpoint: {str(e)}")

def predict_local(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using a local model for demo purposes.
    
    This is used when no SageMaker endpoint is configured.
    """
    global config, icd10_codes, cpt_codes
    
    # Preprocess the text
    preprocessed_text = preprocess_text(
        request.text,
        lowercase=config["preprocessing"]["lowercase"],
        remove_punct=config["preprocessing"]["remove_punctuation"],
        expand_abbrev=config["preprocessing"]["expand_abbreviations"]
    )
    
    # Extract entities
    entities = extract_entities(
        preprocessed_text,
        entity_types=config["ner"]["labels"]
    )
    
    # Simple rule-based prediction for demo purposes
    predictions = []
    
    # Check for common conditions in the text
    text_lower = preprocessed_text.lower()
    
    # ICD-10 predictions
    if "myocardial infarction" in text_lower or "nstemi" in text_lower:
        predictions.append(
            CodePrediction(
                code="I21.4",
                type="ICD-10",
                description=icd10_codes.get("I21.4", "Non-ST elevation myocardial infarction"),
                confidence=0.92
            )
        )
    
    if "hypertension" in text_lower or "high blood pressure" in text_lower:
        predictions.append(
            CodePrediction(
                code="I10",
                type="ICD-10",
                description=icd10_codes.get("I10", "Essential (primary) hypertension"),
                confidence=0.89
            )
        )
    
    if "diabetes" in text_lower or "type 2 diabetes" in text_lower:
        predictions.append(
            CodePrediction(
                code="E11.9",
                type="ICD-10",
                description=icd10_codes.get("E11.9", "Type 2 diabetes mellitus without complications"),
                confidence=0.87
            )
        )
    
    if "chronic kidney disease" in text_lower:
        predictions.append(
            CodePrediction(
                code="N18.2",
                type="ICD-10",
                description=icd10_codes.get("N18.2", "Chronic kidney disease, stage 2 (mild)"),
                confidence=0.83
            )
        )
    
    if "gastroesophageal reflux" in text_lower or "gerd" in text_lower:
        predictions.append(
            CodePrediction(
                code="K21.9",
                type="ICD-10",
                description=icd10_codes.get("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
                confidence=0.76
            )
        )
    
    # CPT predictions
    if "electrocardiogram" in text_lower or "ecg" in text_lower or "ekg" in text_lower:
        predictions.append(
            CodePrediction(
                code="93000",
                type="CPT",
                description=cpt_codes.get("93000", "Electrocardiogram complete"),
                confidence=0.91
            )
        )
    
    if "coronary angiography" in text_lower or "cardiac catheterization" in text_lower:
        predictions.append(
            CodePrediction(
                code="93454",
                type="CPT",
                description=cpt_codes.get("93454", "Coronary angiography"),
                confidence=0.88
            )
        )
    
    if "chest x-ray" in text_lower or "cxr" in text_lower:
        predictions.append(
            CodePrediction(
                code="71046",
                type="CPT",
                description=cpt_codes.get("71046", "Chest X-ray 2 views"),
                confidence=0.85
            )
        )
    
    if "critical care" in text_lower:
        predictions.append(
            CodePrediction(
                code="99291",
                type="CPT",
                description=cpt_codes.get("99291", "Critical care first hour"),
                confidence=0.82
            )
        )
    
    if "metabolic panel" in text_lower:
        predictions.append(
            CodePrediction(
                code="80053",
                type="CPT",
                description=cpt_codes.get("80053", "Comprehensive metabolic panel"),
                confidence=0.79
            )
        )
    
    # Filter by code type if specified
    if request.code_type == "icd10":
        predictions = [p for p in predictions if p.type == "ICD-10"]
    elif request.code_type == "cpt":
        predictions = [p for p in predictions if p.type == "CPT"]
    
    # Filter by threshold
    predictions = [p for p in predictions if p.confidence >= request.threshold]
    
    # Sort by confidence and limit to top_k
    predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)[:request.top_k]
    
    return PredictionResponse(predictions=predictions, entities=entities)

@app.post("/extract_entities", response_model=EntityExtractionResponse)
async def extract_entities_endpoint(request: EntityExtractionRequest):
    """
    Extract medical entities from clinical text.
    
    This endpoint preprocesses the input text and extracts medical entities
    such as diagnoses, procedures, medications, etc.
    """
    global config
    
    # Preprocess the text
    preprocessed_text = preprocess_text(
        request.text,
        lowercase=config["preprocessing"]["lowercase"],
        remove_punct=config["preprocessing"]["remove_punctuation"],
        expand_abbrev=config["preprocessing"]["expand_abbreviations"]
    )
    
    # Extract entities
    entity_types = request.entity_types or config["ner"]["labels"]
    entities = extract_entities(preprocessed_text, entity_types=entity_types)
    
    return EntityExtractionResponse(entities=entities)

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables or use defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    # Run the application
    uvicorn.run("app:app", host=host, port=port, reload=True) 