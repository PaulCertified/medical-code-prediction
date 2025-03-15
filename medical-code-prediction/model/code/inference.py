import json

def model_fn(model_dir):
    # Return a dummy model
    return {"ready": True}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Return dummy predictions
    text = input_data.get("text", "")
    
    predictions = [
        {"code": "I21.4", "type": "ICD-10", "description": "Non-ST elevation myocardial infarction", "confidence": 0.92},
        {"code": "I10", "type": "ICD-10", "description": "Essential (primary) hypertension", "confidence": 0.89},
        {"code": "93000", "type": "CPT", "description": "Electrocardiogram complete", "confidence": 0.91}
    ]
    
    return predictions

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
