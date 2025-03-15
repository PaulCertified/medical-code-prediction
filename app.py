from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import uvicorn
import os

app = FastAPI(title="Medical Code Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-1')

class ClinicalNoteRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ClinicalNoteRequest):
    try:
        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName='medical-code-prediction-v3',
            ContentType='application/json',
            Body=json.dumps({"text": request.text})
        )
        
        # Parse the response
        result = json.loads(response['Body'].read().decode())
        
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 