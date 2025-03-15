from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import uvicorn
import os
import pathlib
import time
import re
from botocore.config import Config

# Get the current directory
BASE_DIR = pathlib.Path(__file__).parent.absolute()
STATIC_DIR = os.path.join(BASE_DIR, "static")

print(f"Base directory: {BASE_DIR}")
print(f"Static directory: {STATIC_DIR}")
print(f"Static directory exists: {os.path.exists(STATIC_DIR)}")
if os.path.exists(STATIC_DIR):
    print(f"Static directory contents: {os.listdir(STATIC_DIR)}")

app = FastAPI(title="Medical Code Prediction API")

# Mount the static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize SageMaker runtime client with timeouts
boto_config = Config(
    connect_timeout=5,  # 5 seconds connection timeout
    read_timeout=5,     # 5 seconds read timeout
    retries={"max_attempts": 0}  # No retries
)
runtime = boto3.client('sagemaker-runtime', region_name='us-west-1', config=boto_config)

# Flag to force using mock predictions
USE_MOCK_PREDICTIONS = True

class ClinicalNoteRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: list

# Mock predictions by scenario
MOCK_PREDICTIONS_CARDIAC = [
    {"code": "I21.4", "type": "ICD-10", "description": "Non-ST elevation myocardial infarction", "confidence": 0.92},
    {"code": "I10", "type": "ICD-10", "description": "Essential (primary) hypertension", "confidence": 0.89},
    {"code": "E11.9", "type": "ICD-10", "description": "Type 2 diabetes mellitus without complications", "confidence": 0.87},
    {"code": "R06.02", "type": "ICD-10", "description": "Shortness of breath", "confidence": 0.84},
    {"code": "R07.9", "type": "ICD-10", "description": "Chest pain, unspecified", "confidence": 0.81},
    {"code": "93000", "type": "CPT", "description": "Electrocardiogram complete", "confidence": 0.91},
    {"code": "93454", "type": "CPT", "description": "Coronary angiography", "confidence": 0.88},
    {"code": "71045", "type": "CPT", "description": "Chest X-ray, single view", "confidence": 0.85},
    {"code": "80053", "type": "CPT", "description": "Comprehensive metabolic panel", "confidence": 0.82},
    {"code": "82550", "type": "CPT", "description": "Creatine kinase (CK), (CPK) total", "confidence": 0.78}
]

MOCK_PREDICTIONS_STROKE = [
    {"code": "I63.9", "type": "ICD-10", "description": "Cerebral infarction, unspecified", "confidence": 0.94},
    {"code": "I10", "type": "ICD-10", "description": "Essential (primary) hypertension", "confidence": 0.90},
    {"code": "G81.9", "type": "ICD-10", "description": "Hemiplegia, unspecified", "confidence": 0.87},
    {"code": "R47.1", "type": "ICD-10", "description": "Dysarthria and anarthria", "confidence": 0.85},
    {"code": "E11.9", "type": "ICD-10", "description": "Type 2 diabetes mellitus without complications", "confidence": 0.82},
    {"code": "70551", "type": "CPT", "description": "MRI brain without contrast", "confidence": 0.93},
    {"code": "70496", "type": "CPT", "description": "CT angiography head", "confidence": 0.90},
    {"code": "93000", "type": "CPT", "description": "Electrocardiogram complete", "confidence": 0.87},
    {"code": "80053", "type": "CPT", "description": "Comprehensive metabolic panel", "confidence": 0.84},
    {"code": "85025", "type": "CPT", "description": "Complete blood count with differential", "confidence": 0.82}
]

MOCK_PREDICTIONS_RESPIRATORY = [
    {"code": "J18.9", "type": "ICD-10", "description": "Pneumonia, unspecified organism", "confidence": 0.93},
    {"code": "J44.9", "type": "ICD-10", "description": "Chronic obstructive pulmonary disease, unspecified", "confidence": 0.88},
    {"code": "R06.02", "type": "ICD-10", "description": "Shortness of breath", "confidence": 0.86},
    {"code": "R05", "type": "ICD-10", "description": "Cough", "confidence": 0.84},
    {"code": "J06.9", "type": "ICD-10", "description": "Acute upper respiratory infection, unspecified", "confidence": 0.78},
    {"code": "71045", "type": "CPT", "description": "Chest X-ray, single view", "confidence": 0.92},
    {"code": "94640", "type": "CPT", "description": "Respiratory inhaled treatments", "confidence": 0.89},
    {"code": "94060", "type": "CPT", "description": "Pulmonary function test (PFT)", "confidence": 0.87},
    {"code": "85025", "type": "CPT", "description": "Complete blood count with differential", "confidence": 0.83},
    {"code": "87430", "type": "CPT", "description": "Infectious agent antigen detection", "confidence": 0.80}
]

MOCK_PREDICTIONS_NEURO = [
    {"code": "G44.209", "type": "ICD-10", "description": "Tension-type headache, unspecified", "confidence": 0.91},
    {"code": "G43.909", "type": "ICD-10", "description": "Migraine, unspecified, not intractable", "confidence": 0.88},
    {"code": "R51", "type": "ICD-10", "description": "Headache", "confidence": 0.87},
    {"code": "H53.8", "type": "ICD-10", "description": "Other visual disturbances", "confidence": 0.82},
    {"code": "R55", "type": "ICD-10", "description": "Syncope and collapse", "confidence": 0.79},
    {"code": "70551", "type": "CPT", "description": "MRI brain without contrast", "confidence": 0.92},
    {"code": "95812", "type": "CPT", "description": "Electroencephalogram (EEG)", "confidence": 0.89},
    {"code": "80053", "type": "CPT", "description": "Comprehensive metabolic panel", "confidence": 0.85},
    {"code": "85025", "type": "CPT", "description": "Complete blood count with differential", "confidence": 0.83},
    {"code": "80076", "type": "CPT", "description": "Hepatic function panel", "confidence": 0.78}
]

MOCK_PREDICTIONS_GI = [
    {"code": "K29.70", "type": "ICD-10", "description": "Gastritis, unspecified, without bleeding", "confidence": 0.90},
    {"code": "K21.9", "type": "ICD-10", "description": "Gastro-esophageal reflux disease without esophagitis", "confidence": 0.87},
    {"code": "R10.9", "type": "ICD-10", "description": "Unspecified abdominal pain", "confidence": 0.85},
    {"code": "K52.9", "type": "ICD-10", "description": "Non-infective gastroenteritis and colitis, unspecified", "confidence": 0.81},
    {"code": "R11.2", "type": "ICD-10", "description": "Nausea with vomiting, unspecified", "confidence": 0.78},
    {"code": "43235", "type": "CPT", "description": "Upper gastrointestinal endoscopy", "confidence": 0.91},
    {"code": "74150", "type": "CPT", "description": "CT abdomen without contrast", "confidence": 0.88},
    {"code": "80053", "type": "CPT", "description": "Comprehensive metabolic panel", "confidence": 0.86},
    {"code": "85025", "type": "CPT", "description": "Complete blood count with differential", "confidence": 0.83},
    {"code": "81001", "type": "CPT", "description": "Urinalysis, automated with microscopy", "confidence": 0.79}
]

MOCK_PREDICTIONS_DEFAULT = [
    {"code": "Z00.00", "type": "ICD-10", "description": "Encounter for general adult medical examination without abnormal findings", "confidence": 0.95},
    {"code": "Z13.9", "type": "ICD-10", "description": "Encounter for screening, unspecified", "confidence": 0.90},
    {"code": "Z71.89", "type": "ICD-10", "description": "Other specified counseling", "confidence": 0.85},
    {"code": "Z68.32", "type": "ICD-10", "description": "Body mass index (BMI) 32.0-32.9, adult", "confidence": 0.82},
    {"code": "Z23", "type": "ICD-10", "description": "Encounter for immunization", "confidence": 0.79},
    {"code": "99213", "type": "CPT", "description": "Office/outpatient visit, established patient, 20-29 minutes", "confidence": 0.93},
    {"code": "80053", "type": "CPT", "description": "Comprehensive metabolic panel", "confidence": 0.89},
    {"code": "85025", "type": "CPT", "description": "Complete blood count with differential", "confidence": 0.87},
    {"code": "81002", "type": "CPT", "description": "Urinalysis, non-automated, without microscopy", "confidence": 0.83},
    {"code": "36415", "type": "CPT", "description": "Venipuncture, routine", "confidence": 0.81}
]

def get_mock_predictions_by_text(clinical_text):
    """
    Return different mock predictions based on the content of the clinical text
    """
    clinical_text = clinical_text.lower()
    
    # Define keyword patterns for different medical scenarios
    cardiac_pattern = re.compile(r'chest pain|shortness of breath|st depression|heart|troponin|cardiac|myocardial|coronary|heart failure|cardiovascular')
    stroke_pattern = re.compile(r'stroke|facial droop|weakness|slurred speech|difficulty speaking|numbness|sudden onset|hemiparesis|hemiplegia|cerebral|paralysis')
    respiratory_pattern = re.compile(r'cough|shortness of breath|sob|pneumonia|copd|asthma|respiratory|wheezing|pleural|bronchitis|lung')
    neuro_pattern = re.compile(r'headache|seizure|migraine|dizziness|confusion|altered mental status|syncope|vertigo|neuropathy|paresthesia')
    gi_pattern = re.compile(r'abdominal pain|nausea|vomiting|diarrhea|constipation|indigestion|heartburn|gastritis|gerd|reflux|gastroenteritis')
    
    # Check which pattern has the most matches
    scenarios = [
        (cardiac_pattern.findall(clinical_text), MOCK_PREDICTIONS_CARDIAC),
        (stroke_pattern.findall(clinical_text), MOCK_PREDICTIONS_STROKE),
        (respiratory_pattern.findall(clinical_text), MOCK_PREDICTIONS_RESPIRATORY),
        (neuro_pattern.findall(clinical_text), MOCK_PREDICTIONS_NEURO),
        (gi_pattern.findall(clinical_text), MOCK_PREDICTIONS_GI)
    ]
    
    # Return predictions for the scenario with the most matching keywords
    max_matches = 0
    best_predictions = MOCK_PREDICTIONS_DEFAULT
    
    for matches, predictions in scenarios:
        if len(matches) > max_matches:
            max_matches = len(matches)
            best_predictions = predictions
    
    return best_predictions

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ClinicalNoteRequest):
    print(f"Received prediction request for text: {request.text[:50]}...")
    
    # Add a small delay to simulate processing
    time.sleep(1)
    
    # If forcing mock predictions, return scenario-based predictions
    if USE_MOCK_PREDICTIONS:
        print("Using mock predictions based on clinical scenario")
        predictions = get_mock_predictions_by_text(request.text)
        return {"predictions": predictions}
    
    try:
        try:
            # Try to invoke the SageMaker endpoint with a timeout
            print("Attempting to call SageMaker endpoint...")
            response = runtime.invoke_endpoint(
                EndpointName='medical-code-prediction-v3',
                ContentType='application/json',
                Body=json.dumps({"text": request.text})
            )
            
            # Parse the response
            result = json.loads(response['Body'].read().decode())
            print(f"SageMaker response received: {result}")
            return {"predictions": result}
            
        except Exception as sagemaker_error:
            # Log the SageMaker error
            print(f"SageMaker error: {str(sagemaker_error)}")
            print("Falling back to mock predictions")
            
            # Return scenario-based mock predictions
            predictions = get_mock_predictions_by_text(request.text)
            return {"predictions": predictions}
            
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        # Return mock predictions even on other errors
        print("Error occurred, still returning mock predictions")
        predictions = get_mock_predictions_by_text(request.text)
        return {"predictions": predictions}

@app.get("/")
async def root():
    # Redirect to the UI HTML file
    return RedirectResponse(url="/ui2")

# Add a direct endpoint for the static file
@app.get("/ui", response_class=HTMLResponse)
async def ui():
    try:
        static_file = os.path.join(STATIC_DIR, "index.html")
        with open(static_file, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error loading UI: {str(e)}"

# Add a direct endpoint for the ui.html file
@app.get("/ui2", response_class=HTMLResponse)
async def ui2():
    try:
        ui_file = os.path.join(BASE_DIR, "ui.html")
        with open(ui_file, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error loading UI: {str(e)}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 