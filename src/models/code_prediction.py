"""
Code prediction model for ICD-10 and CPT codes.
"""
import os
import csv
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd


class CodePredictionModel:
    """
    Model for predicting ICD-10 and CPT codes from clinical text.
    
    This is a simplified implementation for demonstration purposes.
    In a real-world scenario, this would use a trained transformer model.
    """
    
    def __init__(
        self,
        model_name: str = "biobert-base-cased-v1.1",
        max_length: int = 512,
        icd10_codes_path: Optional[str] = None,
        cpt_codes_path: Optional[str] = None
    ):
        """
        Initialize the code prediction model.
        
        Args:
            model_name: Name of the pretrained model to use
            max_length: Maximum sequence length for the model
            icd10_codes_path: Path to the ICD-10 codes reference file
            cpt_codes_path: Path to the CPT codes reference file
        """
        self.model_name = model_name
        self.max_length = max_length
        self.icd10_codes = {}
        self.cpt_codes = {}
        
        # Load code reference files if provided
        if icd10_codes_path and os.path.exists(icd10_codes_path):
            self.load_icd10_codes(icd10_codes_path)
        
        if cpt_codes_path and os.path.exists(cpt_codes_path):
            self.load_cpt_codes(cpt_codes_path)
        
        # In a real implementation, this would load a trained model
        print(f"Initialized code prediction model with {model_name}")
    
    def load_icd10_codes(self, file_path: str) -> None:
        """
        Load ICD-10 codes from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing ICD-10 codes
        """
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        code, description = row[0], row[1]
                        self.icd10_codes[code] = description
            print(f"Loaded {len(self.icd10_codes)} ICD-10 codes from {file_path}")
        except Exception as e:
            print(f"Error loading ICD-10 codes: {e}")
    
    def load_cpt_codes(self, file_path: str) -> None:
        """
        Load CPT codes from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing CPT codes
        """
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader)
                for row in reader:
                    if len(row) >= 2:
                        code, description = row[0], row[1]
                        self.cpt_codes[code] = description
            print(f"Loaded {len(self.cpt_codes)} CPT codes from {file_path}")
        except Exception as e:
            print(f"Error loading CPT codes: {e}")
    
    def load(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        # In a real implementation, this would load model weights
        print(f"Loaded model from {model_path}")
    
    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        top_k: int = 5,
        code_type: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Predict ICD-10 and/or CPT codes from clinical text.
        
        Args:
            text: Clinical text to predict codes for
            threshold: Confidence threshold for predictions
            top_k: Number of top predictions to return
            code_type: Type of codes to predict ("icd10", "cpt", or "both")
            
        Returns:
            List of dictionaries containing predicted codes, descriptions, and confidence scores
        """
        # This is a simplified implementation for demonstration purposes
        # In a real-world scenario, this would use a trained model for prediction
        
        predictions = []
        
        # Extract key terms from the text for rule-based matching
        key_terms = self._extract_key_terms(text)
        
        # Predict ICD-10 codes
        if code_type in ["icd10", "both"]:
            icd10_predictions = self._predict_icd10_codes(text, key_terms)
            predictions.extend(icd10_predictions)
        
        # Predict CPT codes
        if code_type in ["cpt", "both"]:
            cpt_predictions = self._predict_cpt_codes(text, key_terms)
            predictions.extend(cpt_predictions)
        
        # Sort predictions by confidence score
        predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        
        # Filter by threshold and limit to top_k
        predictions = [p for p in predictions if p["confidence"] >= threshold][:top_k]
        
        return predictions
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key medical terms from the text.
        
        Args:
            text: Clinical text
            
        Returns:
            List of key medical terms
        """
        # This is a simplified implementation
        # In a real-world scenario, this would use NLP techniques
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Define key medical terms to look for
        medical_terms = [
            "myocardial infarction", "heart attack", "mi", "nstemi", "stemi",
            "acute coronary syndrome", "acs",
            "hypertension", "high blood pressure", "htn",
            "diabetes", "diabetes mellitus", "type 2 diabetes", "t2dm",
            "chronic kidney disease", "ckd",
            "heart failure", "hf", "chf",
            "coronary artery disease", "cad",
            "chest pain", "angina",
            "shortness of breath", "sob", "dyspnea",
            "echocardiogram", "echo",
            "electrocardiogram", "ecg", "ekg",
            "cardiac catheterization", "cath",
            "coronary angiography", "angiogram",
            "aspirin", "clopidogrel", "plavix",
            "atorvastatin", "lipitor",
            "metoprolol", "lopressor", "toprol",
            "lisinopril", "prinivil", "zestril",
            "metformin", "glucophage",
            "insulin",
            "heparin",
            "troponin", "ck-mb", "cardiac enzymes",
            "lipid panel", "cholesterol",
            "complete blood count", "cbc",
            "basic metabolic panel", "bmp",
            "comprehensive metabolic panel", "cmp",
        ]
        
        # Find matches in the text
        matches = []
        for term in medical_terms:
            if term in text_lower:
                matches.append(term)
        
        return matches
    
    def _predict_icd10_codes(self, text: str, key_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Predict ICD-10 codes based on text and key terms.
        
        Args:
            text: Clinical text
            key_terms: Extracted key terms from the text
            
        Returns:
            List of dictionaries containing predicted ICD-10 codes
        """
        predictions = []
        
        # This is a simplified rule-based implementation for demonstration
        # In a real-world scenario, this would use a trained model
        
        # Check for acute coronary syndrome / NSTEMI
        if any(term in key_terms for term in ["acute coronary syndrome", "acs", "nstemi", "non-st elevation myocardial infarction"]):
            predictions.append({
                "code": "I21.4",
                "description": self.icd10_codes.get("I21.4", "Non-ST elevation myocardial infarction"),
                "confidence": random.uniform(0.85, 0.95),
                "type": "ICD-10"
            })
        
        # Check for STEMI
        if any(term in key_terms for term in ["stemi", "st elevation myocardial infarction"]):
            predictions.append({
                "code": "I21.3",
                "description": self.icd10_codes.get("I21.3", "ST elevation myocardial infarction of unspecified site"),
                "confidence": random.uniform(0.85, 0.95),
                "type": "ICD-10"
            })
        
        # Check for hypertension
        if any(term in key_terms for term in ["hypertension", "htn", "high blood pressure"]):
            predictions.append({
                "code": "I10",
                "description": self.icd10_codes.get("I10", "Essential (primary) hypertension"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "ICD-10"
            })
        
        # Check for type 2 diabetes
        if any(term in key_terms for term in ["diabetes", "diabetes mellitus", "type 2 diabetes", "t2dm"]):
            predictions.append({
                "code": "E11.9",
                "description": self.icd10_codes.get("E11.9", "Type 2 diabetes mellitus without complications"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "ICD-10"
            })
        
        # Check for chronic kidney disease
        if any(term in key_terms for term in ["chronic kidney disease", "ckd"]):
            predictions.append({
                "code": "N18.9",
                "description": self.icd10_codes.get("N18.9", "Chronic kidney disease, unspecified"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "ICD-10"
            })
            
            # Check for stage 2 CKD
            if "stage 2" in text.lower():
                predictions.append({
                    "code": "N18.2",
                    "description": self.icd10_codes.get("N18.2", "Chronic kidney disease, stage 2 (mild)"),
                    "confidence": random.uniform(0.80, 0.90),
                    "type": "ICD-10"
                })
        
        # Check for heart failure
        if any(term in key_terms for term in ["heart failure", "hf", "chf"]):
            predictions.append({
                "code": "I50.9",
                "description": self.icd10_codes.get("I50.9", "Heart failure, unspecified"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "ICD-10"
            })
        
        # Check for GERD
        if "gerd" in key_terms or "gastroesophageal reflux disease" in key_terms:
            predictions.append({
                "code": "K21.9",
                "description": self.icd10_codes.get("K21.9", "Gastro-esophageal reflux disease without esophagitis"),
                "confidence": random.uniform(0.70, 0.80),
                "type": "ICD-10"
            })
        
        # Check for hyperlipidemia
        if any(term in key_terms for term in ["hyperlipidemia", "high cholesterol", "lipid"]):
            predictions.append({
                "code": "E78.5",
                "description": self.icd10_codes.get("E78.5", "Hyperlipidemia, unspecified"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "ICD-10"
            })
        
        # Check for chest pain
        if "chest pain" in key_terms:
            predictions.append({
                "code": "R07.9",
                "description": self.icd10_codes.get("R07.9", "Chest pain, unspecified"),
                "confidence": random.uniform(0.70, 0.80),
                "type": "ICD-10"
            })
        
        # Check for shortness of breath
        if any(term in key_terms for term in ["shortness of breath", "sob", "dyspnea"]):
            predictions.append({
                "code": "R06.02",
                "description": self.icd10_codes.get("R06.02", "Shortness of breath"),
                "confidence": random.uniform(0.70, 0.80),
                "type": "ICD-10"
            })
        
        return predictions
    
    def _predict_cpt_codes(self, text: str, key_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Predict CPT codes based on text and key terms.
        
        Args:
            text: Clinical text
            key_terms: Extracted key terms from the text
            
        Returns:
            List of dictionaries containing predicted CPT codes
        """
        predictions = []
        
        # This is a simplified rule-based implementation for demonstration
        # In a real-world scenario, this would use a trained model
        
        # Check for ECG/EKG
        if any(term in key_terms for term in ["electrocardiogram", "ecg", "ekg"]):
            predictions.append({
                "code": "93000",
                "description": self.cpt_codes.get("93000", "Electrocardiogram complete"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "CPT"
            })
        
        # Check for echocardiogram
        if any(term in key_terms for term in ["echocardiogram", "echo"]):
            predictions.append({
                "code": "93306",
                "description": self.cpt_codes.get("93306", "Echocardiography complete with spectral and color flow Doppler"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "CPT"
            })
        
        # Check for cardiac catheterization / coronary angiography
        if any(term in key_terms for term in ["cardiac catheterization", "cath", "coronary angiography", "angiogram"]):
            predictions.append({
                "code": "93454",
                "description": self.cpt_codes.get("93454", "Coronary angiography"),
                "confidence": random.uniform(0.85, 0.95),
                "type": "CPT"
            })
        
        # Check for chest X-ray
        if "chest x-ray" in key_terms or "cxr" in key_terms:
            predictions.append({
                "code": "71046",
                "description": self.cpt_codes.get("71046", "Chest X-ray 2 views"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "CPT"
            })
        
        # Check for comprehensive metabolic panel
        if "comprehensive metabolic panel" in key_terms or "cmp" in key_terms:
            predictions.append({
                "code": "80053",
                "description": self.cpt_codes.get("80053", "Comprehensive metabolic panel"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "CPT"
            })
        
        # Check for basic metabolic panel
        if "basic metabolic panel" in key_terms or "bmp" in key_terms:
            predictions.append({
                "code": "80048",
                "description": self.cpt_codes.get("80048", "Basic metabolic panel"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "CPT"
            })
        
        # Check for lipid panel
        if "lipid panel" in key_terms or "cholesterol" in key_terms:
            predictions.append({
                "code": "80061",
                "description": self.cpt_codes.get("80061", "Lipid panel"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "CPT"
            })
        
        # Check for CBC
        if "complete blood count" in key_terms or "cbc" in key_terms:
            predictions.append({
                "code": "85025",
                "description": self.cpt_codes.get("85025", "Complete CBC with auto diff WBC"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "CPT"
            })
        
        # Check for troponin
        if "troponin" in key_terms or "cardiac enzymes" in key_terms:
            predictions.append({
                "code": "84484",
                "description": self.cpt_codes.get("84484", "Troponin quantitative"),
                "confidence": random.uniform(0.80, 0.90),
                "type": "CPT"
            })
        
        # Check for initial hospital care
        if "admit" in text.lower() or "admission" in text.lower():
            predictions.append({
                "code": "99223",
                "description": self.cpt_codes.get("99223", "Initial hospital care per day level 3"),
                "confidence": random.uniform(0.70, 0.80),
                "type": "CPT"
            })
        
        # Check for critical care
        if "ccu" in key_terms or "cardiac care unit" in key_terms or "critical care" in text.lower():
            predictions.append({
                "code": "99291",
                "description": self.cpt_codes.get("99291", "Critical care first hour"),
                "confidence": random.uniform(0.75, 0.85),
                "type": "CPT"
            })
        
        # Check for IV infusion
        if "iv" in key_terms or "intravenous" in key_terms:
            predictions.append({
                "code": "96365",
                "description": self.cpt_codes.get("96365", "IV infusion therapy initial up to 1 hour"),
                "confidence": random.uniform(0.70, 0.80),
                "type": "CPT"
            })
        
        return predictions
    
    def explain(self, text: str, code: str) -> Dict[str, Any]:
        """
        Provide an explanation for a predicted code.
        
        Args:
            text: Clinical text
            code: The code to explain
            
        Returns:
            Dictionary containing explanation details
        """
        # This is a simplified implementation for demonstration purposes
        # In a real-world scenario, this would use model interpretability techniques
        
        explanation = {
            "code": code,
            "description": "",
            "confidence": 0.0,
            "relevant_text": [],
            "feature_importance": {}
        }
        
        # Determine if it's an ICD-10 or CPT code
        if code in self.icd10_codes:
            explanation["description"] = self.icd10_codes[code]
            explanation["type"] = "ICD-10"
        elif code in self.cpt_codes:
            explanation["description"] = self.cpt_codes[code]
            explanation["type"] = "CPT"
        else:
            explanation["description"] = "Unknown code"
            explanation["type"] = "Unknown"
        
        # Extract relevant text segments (simplified)
        text_lower = text.lower()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # ICD-10 code explanations
        if code == "I21.4":  # NSTEMI
            keywords = ["nstemi", "non-st elevation", "acute coronary syndrome", "myocardial infarction", "troponin", "chest pain"]
            explanation["confidence"] = random.uniform(0.85, 0.95)
            explanation["feature_importance"] = {
                "troponin elevation": 0.35,
                "chest pain": 0.25,
                "ECG changes": 0.20,
                "clinical presentation": 0.15,
                "risk factors": 0.05
            }
        
        elif code == "I10":  # Hypertension
            keywords = ["hypertension", "high blood pressure", "htn", "elevated blood pressure"]
            explanation["confidence"] = random.uniform(0.80, 0.90)
            explanation["feature_importance"] = {
                "blood pressure readings": 0.40,
                "medication history": 0.30,
                "clinical history": 0.20,
                "risk factors": 0.10
            }
        
        elif code == "E11.9":  # Type 2 diabetes
            keywords = ["diabetes", "type 2", "t2dm", "hyperglycemia", "glucose", "hba1c"]
            explanation["confidence"] = random.uniform(0.80, 0.90)
            explanation["feature_importance"] = {
                "diabetes history": 0.35,
                "glucose levels": 0.25,
                "HbA1c": 0.20,
                "medications": 0.15,
                "symptoms": 0.05
            }
        
        # CPT code explanations
        elif code == "93000":  # ECG
            keywords = ["ecg", "ekg", "electrocardiogram"]
            explanation["confidence"] = random.uniform(0.80, 0.90)
            explanation["feature_importance"] = {
                "procedure mention": 0.60,
                "clinical indication": 0.30,
                "context": 0.10
            }
        
        elif code == "93454":  # Coronary angiography
            keywords = ["coronary angiography", "angiogram", "cardiac catheterization", "cath"]
            explanation["confidence"] = random.uniform(0.85, 0.95)
            explanation["feature_importance"] = {
                "procedure mention": 0.50,
                "clinical indication": 0.30,
                "context": 0.20
            }
        
        elif code == "80053":  # Comprehensive metabolic panel
            keywords = ["comprehensive metabolic panel", "cmp", "metabolic panel"]
            explanation["confidence"] = random.uniform(0.80, 0.90)
            explanation["feature_importance"] = {
                "test mention": 0.60,
                "clinical indication": 0.25,
                "context": 0.15
            }
        
        else:
            keywords = []
            explanation["confidence"] = random.uniform(0.60, 0.70)
            explanation["feature_importance"] = {
                "unknown factors": 1.0
            }
        
        # Find relevant text segments containing keywords
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                explanation["relevant_text"].append(sentence)
        
        # Limit to top 3 most relevant segments
        explanation["relevant_text"] = explanation["relevant_text"][:3]
        
        return explanation 