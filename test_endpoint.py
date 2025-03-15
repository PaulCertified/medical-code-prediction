import boto3
import json

# Initialize SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-west-1')

# Sample clinical text
sample_text = """
68-year-old male presenting with chest pain and shortness of breath for the past 2 days.
Patient has a history of hypertension and type 2 diabetes.
ECG shows ST depression in leads V3-V5.
Troponin I elevated at 0.8 ng/mL.
"""

# Prepare the request payload
payload = {
    "text": sample_text
}

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName='medical-code-prediction-v3',
    ContentType='application/json',
    Body=json.dumps(payload)
)

# Parse the response
result = json.loads(response['Body'].read().decode())

# Print the results
print("Predicted Medical Codes:")
for code in result:
    print(f"  - {code['code']} ({code['type']}): {code['description']} (Confidence: {code['confidence']})") 