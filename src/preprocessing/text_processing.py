"""
Text processing utilities for medical text.
"""
import re
import string
from typing import Dict, List, Optional, Set, Union

# Sample medical abbreviations dictionary
MEDICAL_ABBREVIATIONS = {
    "MI": "myocardial infarction",
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "CKD": "chronic kidney disease",
    "CABG": "coronary artery bypass graft",
    "CAD": "coronary artery disease",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CVA": "cerebrovascular accident",
    "DVT": "deep vein thrombosis",
    "PE": "pulmonary embolism",
    "GERD": "gastroesophageal reflux disease",
    "HLD": "hyperlipidemia",
    "HF": "heart failure",
    "ACS": "acute coronary syndrome",
    "NSTEMI": "non-ST elevation myocardial infarction",
    "STEMI": "ST elevation myocardial infarction",
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "SpO2": "oxygen saturation",
    "Temp": "temperature",
    "JVD": "jugular venous distention",
    "SOB": "shortness of breath",
    "CP": "chest pain",
    "HA": "headache",
    "N/V": "nausea and vomiting",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "PRN": "as needed",
    "PO": "by mouth",
    "IV": "intravenous",
    "IM": "intramuscular",
    "SL": "sublingual",
    "SC": "subcutaneous",
    "CCU": "cardiac care unit",
    "ICU": "intensive care unit",
    "ED": "emergency department",
    "ECG": "electrocardiogram",
    "EKG": "electrocardiogram",
    "CXR": "chest X-ray",
    "CBC": "complete blood count",
    "BMP": "basic metabolic panel",
    "CMP": "comprehensive metabolic panel",
    "LFT": "liver function test",
    "BUN": "blood urea nitrogen",
    "Cr": "creatinine",
    "K": "potassium",
    "Na": "sodium",
    "Cl": "chloride",
    "CO2": "carbon dioxide",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "WBC": "white blood cell count",
    "Plt": "platelet count",
    "PT": "prothrombin time",
    "INR": "international normalized ratio",
    "PTT": "partial thromboplastin time",
    "BNP": "brain natriuretic peptide",
    "CK-MB": "creatine kinase-MB",
    "LDL": "low-density lipoprotein",
    "HDL": "high-density lipoprotein",
    "TG": "triglycerides",
    "HbA1c": "hemoglobin A1c",
}


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_punct: bool = False,
    expand_abbrev: bool = True
) -> str:
    """
    Preprocess clinical text for NLP tasks.
    
    Args:
        text: The input clinical text
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
        expand_abbrev: Whether to expand medical abbreviations
        
    Returns:
        Preprocessed text
    """
    # Make a copy of the original text
    processed_text = text
    
    # Convert to lowercase if specified
    if lowercase:
        processed_text = processed_text.lower()
    
    # Expand abbreviations if specified
    if expand_abbrev:
        # Create a regex pattern for whole word matching of abbreviations
        abbrev_pattern = r'\b(' + '|'.join(re.escape(abbr) for abbr in MEDICAL_ABBREVIATIONS.keys()) + r')\b'
        
        # Function to replace abbreviations with their expansions
        def expand_match(match):
            abbr = match.group(0)
            # Preserve case if the original text wasn't lowercased
            if not lowercase and abbr.isupper():
                return MEDICAL_ABBREVIATIONS[abbr.upper()]
            return MEDICAL_ABBREVIATIONS[abbr]
        
        # Replace abbreviations in the text
        processed_text = re.sub(abbrev_pattern, expand_match, processed_text, flags=re.IGNORECASE)
    
    # Remove punctuation if specified
    if remove_punct:
        processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
    
    return processed_text


def extract_entities(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract medical entities from clinical text.
    
    This is a simplified implementation for demonstration purposes.
    In a real-world scenario, this would use a trained NER model.
    
    Args:
        text: The input clinical text
        entity_types: Types of entities to extract (e.g., DIAGNOSIS, PROCEDURE)
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    # Default entity types if none provided
    if entity_types is None:
        entity_types = ["DIAGNOSIS", "PROCEDURE", "MEDICATION", "SYMPTOM", "ANATOMY"]
    
    # Initialize results dictionary
    entities = {entity_type: [] for entity_type in entity_types}
    
    # Simple rule-based extraction for demonstration
    # In a real implementation, this would use a trained NER model
    
    # Extract diagnoses
    if "DIAGNOSIS" in entity_types:
        diagnosis_patterns = [
            r"(?i)acute coronary syndrome",
            r"(?i)myocardial infarction",
            r"(?i)NSTEMI",
            r"(?i)STEMI",
            r"(?i)hypertension",
            r"(?i)type 2 diabetes mellitus",
            r"(?i)diabetes mellitus",
            r"(?i)chronic kidney disease",
            r"(?i)heart failure",
            r"(?i)GERD",
            r"(?i)gastroesophageal reflux disease",
            r"(?i)hyperlipidemia",
            r"(?i)coronary artery disease",
        ]
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, text)
            entities["DIAGNOSIS"].extend(matches)
    
    # Extract procedures
    if "PROCEDURE" in entity_types:
        procedure_patterns = [
            r"(?i)cardiac catheterization",
            r"(?i)coronary angiography",
            r"(?i)echocardiography",
            r"(?i)electrocardiogram",
            r"(?i)ECG",
            r"(?i)EKG",
            r"(?i)chest X-ray",
            r"(?i)CXR",
            r"(?i)CABG",
            r"(?i)coronary artery bypass graft",
        ]
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text)
            entities["PROCEDURE"].extend(matches)
    
    # Extract medications
    if "MEDICATION" in entity_types:
        medication_patterns = [
            r"(?i)aspirin",
            r"(?i)clopidogrel",
            r"(?i)atorvastatin",
            r"(?i)lisinopril",
            r"(?i)metoprolol",
            r"(?i)metformin",
            r"(?i)insulin",
            r"(?i)nitroglycerin",
            r"(?i)heparin",
            r"(?i)omeprazole",
            r"(?i)amlodipine",
        ]
        for pattern in medication_patterns:
            matches = re.findall(pattern, text)
            entities["MEDICATION"].extend(matches)
    
    # Extract symptoms
    if "SYMPTOM" in entity_types:
        symptom_patterns = [
            r"(?i)chest pain",
            r"(?i)shortness of breath",
            r"(?i)dyspnea",
            r"(?i)nausea",
            r"(?i)vomiting",
            r"(?i)diaphoresis",
            r"(?i)fatigue",
            r"(?i)dizziness",
            r"(?i)syncope",
            r"(?i)palpitations",
        ]
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            entities["SYMPTOM"].extend(matches)
    
    # Extract anatomy
    if "ANATOMY" in entity_types:
        anatomy_patterns = [
            r"(?i)heart",
            r"(?i)lung",
            r"(?i)kidney",
            r"(?i)liver",
            r"(?i)coronary artery",
            r"(?i)left ventricle",
            r"(?i)right ventricle",
            r"(?i)atrium",
        ]
        for pattern in anatomy_patterns:
            matches = re.findall(pattern, text)
            entities["ANATOMY"].extend(matches)
    
    # Extract tests
    if "TEST" in entity_types:
        test_patterns = [
            r"(?i)troponin",
            r"(?i)CK-MB",
            r"(?i)BNP",
            r"(?i)CBC",
            r"(?i)complete blood count",
            r"(?i)BMP",
            r"(?i)basic metabolic panel",
            r"(?i)lipid panel",
            r"(?i)HbA1c",
            r"(?i)hemoglobin A1c",
        ]
        for pattern in test_patterns:
            matches = re.findall(pattern, text)
            entities["TEST"].extend(matches)
    
    # Extract treatments
    if "TREATMENT" in entity_types:
        treatment_patterns = [
            r"(?i)statin therapy",
            r"(?i)antiplatelet therapy",
            r"(?i)anticoagulation",
            r"(?i)beta-blocker",
            r"(?i)ACE inhibitor",
            r"(?i)ARB",
            r"(?i)diuretic",
            r"(?i)insulin therapy",
            r"(?i)oral hypoglycemic",
        ]
        for pattern in treatment_patterns:
            matches = re.findall(pattern, text)
            entities["TREATMENT"].extend(matches)
    
    # Remove duplicates and empty lists
    for entity_type in entity_types:
        entities[entity_type] = list(set([match.lower() for match in entities[entity_type]]))
        if not entities[entity_type]:
            entities[entity_type] = []
    
    return entities 