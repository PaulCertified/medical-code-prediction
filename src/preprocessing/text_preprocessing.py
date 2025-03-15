"""
Text preprocessing module for clinical text.
"""
import re
import string
from typing import Dict, List, Optional, Union

# Dictionary of common medical abbreviations
MEDICAL_ABBREVIATIONS = {
    "pt": "patient",
    "pts": "patients",
    "dx": "diagnosis",
    "hx": "history",
    "tx": "treatment",
    "sx": "symptoms",
    "fx": "fracture",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "gerd": "gastroesophageal reflux disease",
    "bph": "benign prostatic hyperplasia",
    "ckd": "chronic kidney disease",
    "esrd": "end-stage renal disease",
    "afib": "atrial fibrillation",
    "hld": "hyperlipidemia",
    "osa": "obstructive sleep apnea",
    "ra": "rheumatoid arthritis",
    "sle": "systemic lupus erythematosus",
    "gib": "gastrointestinal bleeding",
    "uc": "ulcerative colitis",
    "ibs": "irritable bowel syndrome",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uti": "urinary tract infection",
    "ards": "acute respiratory distress syndrome",
    "aki": "acute kidney injury",
    "hf": "heart failure",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "n/v": "nausea and vomiting",
    "c/o": "complains of",
    "s/p": "status post",
    "h/o": "history of",
    "f/u": "follow up",
    "yo": "year old",
    "y/o": "year old",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rr": "respiratory rate",
    "t": "temperature",
    "o2": "oxygen",
    "spo2": "oxygen saturation",
    "wbc": "white blood cell count",
    "rbc": "red blood cell count",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet count",
    "bun": "blood urea nitrogen",
    "cr": "creatinine",
    "gfr": "glomerular filtration rate",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    "tbili": "total bilirubin",
    "a1c": "hemoglobin a1c",
    "tsh": "thyroid stimulating hormone",
    "ua": "urinalysis",
    "cxr": "chest x-ray",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "us": "ultrasound",
    "ekg": "electrocardiogram",
    "echo": "echocardiogram",
    "endo": "endoscopy",
    "colo": "colonoscopy",
    "egd": "esophagogastroduodenoscopy",
    "cabg": "coronary artery bypass graft",
    "ptca": "percutaneous transluminal coronary angioplasty",
    "pci": "percutaneous coronary intervention",
    "tka": "total knee arthroplasty",
    "tha": "total hip arthroplasty",
    "orif": "open reduction internal fixation",
    "lap": "laparoscopic",
    "lap chole": "laparoscopic cholecystectomy",
    "appy": "appendectomy",
    "po": "by mouth",
    "pr": "per rectum",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "sl": "sublingual",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qd": "once daily",
    "qod": "every other day",
    "prn": "as needed",
    "q4h": "every 4 hours",
    "q6h": "every 6 hours",
    "q8h": "every 8 hours",
    "q12h": "every 12 hours",
    "qhs": "at bedtime",
    "ac": "before meals",
    "pc": "after meals",
    "w/": "with",
    "w/o": "without",
    "b/l": "bilateral",
    "r/o": "rule out",
    "d/c": "discharge or discontinue",
    "f/c": "fever and chills",
    "n/a": "not applicable",
    "neg": "negative",
    "pos": "positive",
    "wt": "weight",
    "ht": "height",
    "bmi": "body mass index",
    "cc": "chief complaint",
    "pmh": "past medical history",
    "psh": "past surgical history",
    "fh": "family history",
    "sh": "social history",
    "meds": "medications",
    "all": "allergies",
    "ros": "review of systems",
    "pe": "physical examination",
    "vs": "vital signs",
    "labs": "laboratory results",
    "a/p": "assessment and plan",
    "icu": "intensive care unit",
    "ed": "emergency department",
    "or": "operating room",
    "pacu": "post-anesthesia care unit",
    "snf": "skilled nursing facility",
    "ltc": "long-term care",
    "rehab": "rehabilitation",
    "pt": "physical therapy",
    "ot": "occupational therapy",
    "st": "speech therapy",
    "rt": "respiratory therapy",
    "sw": "social work",
    "md": "medical doctor",
    "np": "nurse practitioner",
    "pa": "physician assistant",
    "rn": "registered nurse",
    "lpn": "licensed practical nurse",
    "cna": "certified nursing assistant",
    "doa": "dead on arrival",
    "dnr": "do not resuscitate",
    "cpr": "cardiopulmonary resuscitation",
    "adl": "activities of daily living",
    "iadl": "instrumental activities of daily living",
    "loc": "level of consciousness",
    "gcs": "glasgow coma scale",
    "mmse": "mini-mental state examination",
    "moca": "montreal cognitive assessment",
    "bmp": "basic metabolic panel",
    "cmp": "comprehensive metabolic panel",
    "cbc": "complete blood count",
    "lft": "liver function tests",
    "abg": "arterial blood gas",
    "pft": "pulmonary function test",
    "lfts": "liver function tests",
    "uti": "urinary tract infection",
    "ubs": "urinalysis with reflex to culture",
    "c&s": "culture and sensitivity",
    "mrsa": "methicillin-resistant staphylococcus aureus",
    "vre": "vancomycin-resistant enterococcus",
    "cdiff": "clostridium difficile",
    "hiv": "human immunodeficiency virus",
    "hcv": "hepatitis c virus",
    "hbv": "hepatitis b virus",
    "tb": "tuberculosis",
    "ca": "cancer",
    "mets": "metastasis",
    "chemo": "chemotherapy",
    "rt": "radiation therapy",
    "nsaid": "non-steroidal anti-inflammatory drug",
    "ppi": "proton pump inhibitor",
    "acei": "angiotensin-converting enzyme inhibitor",
    "arb": "angiotensin receptor blocker",
    "ccb": "calcium channel blocker",
    "bb": "beta blocker",
    "abx": "antibiotics",
    "vanco": "vancomycin",
    "levo": "levofloxacin",
    "cipro": "ciprofloxacin",
    "amox": "amoxicillin",
    "augmentin": "amoxicillin-clavulanate",
    "pcn": "penicillin",
    "ceph": "cephalosporin",
    "asa": "aspirin",
    "coumadin": "warfarin",
    "heparin": "heparin",
    "lmwh": "low molecular weight heparin",
    "doac": "direct oral anticoagulant",
    "noac": "novel oral anticoagulant",
    "statin": "HMG-CoA reductase inhibitor",
    "ssri": "selective serotonin reuptake inhibitor",
    "snri": "serotonin-norepinephrine reuptake inhibitor",
    "tca": "tricyclic antidepressant",
    "maoi": "monoamine oxidase inhibitor",
    "benzo": "benzodiazepine",
    "opiate": "opioid",
    "apap": "acetaminophen",
    "tylenol": "acetaminophen",
    "motrin": "ibuprofen",
    "advil": "ibuprofen",
    "aleve": "naproxen",
    "lasix": "furosemide",
    "hctz": "hydrochlorothiazide",
    "digoxin": "digoxin",
    "synthroid": "levothyroxine",
    "insulin": "insulin",
    "metformin": "metformin",
    "glipizide": "glipizide",
    "glyburide": "glyburide",
    "januvia": "sitagliptin",
    "jardiance": "empagliflozin",
    "ozempic": "semaglutide",
    "trulicity": "dulaglutide",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novolog": "insulin aspart",
    "levemir": "insulin detemir",
    "tresiba": "insulin degludec",
    "toujeo": "insulin glargine u-300",
    "basaglar": "insulin glargine",
    "admelog": "insulin lispro",
    "fiasp": "insulin aspart",
    "afrezza": "insulin human",
    "humulin": "insulin human",
    "novolin": "insulin human",
    "nph": "neutral protamine hagedorn insulin",
    "regular": "regular insulin",
    "70/30": "70% NPH insulin and 30% regular insulin",
    "75/25": "75% insulin lispro protamine and 25% insulin lispro",
    "50/50": "50% NPH insulin and 50% regular insulin",
    "70/30": "70% insulin aspart protamine and 30% insulin aspart",
    "lisinopril": "lisinopril",
    "enalapril": "enalapril",
    "captopril": "captopril",
    "losartan": "losartan",
    "valsartan": "valsartan",
    "irbesartan": "irbesartan",
    "amlodipine": "amlodipine",
    "diltiazem": "diltiazem",
    "verapamil": "verapamil",
    "metoprolol": "metoprolol",
    "atenolol": "atenolol",
    "carvedilol": "carvedilol",
    "propranolol": "propranolol",
    "hydralazine": "hydralazine",
    "clonidine": "clonidine",
    "spironolactone": "spironolactone",
    "aldactone": "spironolactone",
    "bumex": "bumetanide",
    "torsemide": "torsemide",
    "demadex": "torsemide",
    "zaroxolyn": "metolazone",
    "dyazide": "hydrochlorothiazide-triamterene",
    "maxzide": "hydrochlorothiazide-triamterene",
    "aldactazide": "hydrochlorothiazide-spironolactone",
    "moduretic": "hydrochlorothiazide-amiloride",
    "inspra": "eplerenone",
    "midamor": "amiloride",
    "microzide": "hydrochlorothiazide",
    "chlorthalidone": "chlorthalidone",
    "indapamide": "indapamide",
    "lozol": "indapamide",
    "edecrin": "ethacrynic acid",
    "diuril": "chlorothiazide",
    "diamox": "acetazolamide",
    "mannitol": "mannitol",
    "osmitrol": "mannitol",
    "isordil": "isosorbide dinitrate",
    "imdur": "isosorbide mononitrate",
    "nitro": "nitroglycerin",
    "nitrostat": "nitroglycerin",
    "nitro-dur": "nitroglycerin",
    "nitro-bid": "nitroglycerin",
    "nitrolingual": "nitroglycerin",
    "nitromist": "nitroglycerin",
    "nitrodisc": "nitroglycerin",
    "minitran": "nitroglycerin",
    "transderm-nitro": "nitroglycerin",
    "nitrek": "nitroglycerin",
    "deponit": "nitroglycerin",
    "nitro-time": "nitroglycerin",
    "nitrogard": "nitroglycerin",
    "nitrocot": "nitroglycerin",
    "nitroglyn": "nitroglycerin",
    "nitrol": "nitroglycerin",
    "nitropress": "nitroprusside",
    "nipride": "nitroprusside",
    "brilinta": "ticagrelor",
    "plavix": "clopidogrel",
    "effient": "prasugrel",
    "aggrenox": "aspirin-dipyridamole",
    "persantine": "dipyridamole",
    "eliquis": "apixaban",
    "xarelto": "rivaroxaban",
    "pradaxa": "dabigatran",
    "savaysa": "edoxaban",
    "lovenox": "enoxaparin",
    "fragmin": "dalteparin",
    "innohep": "tinzaparin",
    "arixtra": "fondaparinux",
    "angiomax": "bivalirudin",
    "refludan": "lepirudin",
    "argatroban": "argatroban",
    "acova": "argatroban",
    "activase": "alteplase",
    "retavase": "reteplase",
    "tnkase": "tenecteplase",
    "streptase": "streptokinase",
    "abbokinase": "urokinase",
    "kinlytic": "urokinase",
    "reopro": "abciximab",
    "integrilin": "eptifibatide",
    "aggrastat": "tirofiban",
    "lipitor": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    "pravachol": "pravastatin",
    "lescol": "fluvastatin",
    "livalo": "pitavastatin",
    "mevacor": "lovastatin",
    "altoprev": "lovastatin",
    "caduet": "amlodipine-atorvastatin",
    "vytorin": "ezetimibe-simvastatin",
    "zetia": "ezetimibe",
    "welchol": "colesevelam",
    "colestid": "colestipol",
    "questran": "cholestyramine",
    "prevalite": "cholestyramine",
    "locholest": "cholestyramine",
    "tricor": "fenofibrate",
    "trilipix": "fenofibric acid",
    "antara": "fenofibrate",
    "fenoglide": "fenofibrate",
    "fibricor": "fenofibric acid",
    "lipofen": "fenofibrate",
    "lofibra": "fenofibrate",
    "triglide": "fenofibrate",
    "lopid": "gemfibrozil",
    "niaspan": "niacin",
    "niacor": "niacin",
    "slo-niacin": "niacin",
    "advicor": "niacin-lovastatin",
    "simcor": "niacin-simvastatin",
    "lovaza": "omega-3-acid ethyl esters",
    "vascepa": "icosapent ethyl",
    "epanova": "omega-3-carboxylic acids",
    "omtryg": "omega-3-acid ethyl esters",
    "juxtapid": "lomitapide",
    "kynamro": "mipomersen",
    "praluent": "alirocumab",
    "repatha": "evolocumab",
    "nexletol": "bempedoic acid",
    "nexlizet": "bempedoic acid-ezetimibe",
    "evkeeza": "evinacumab",
    "leqvio": "inclisiran",
    "zypitamag": "pitavastatin",
    "roszet": "ezetimibe-rosuvastatin",
    "ezallor": "rosuvastatin",
    "flolipid": "simvastatin",
}


def clean_text(text: str, lowercase: bool = True, remove_punct: bool = False) -> str:
    """
    Clean clinical text by removing extra whitespace, optionally lowercasing,
    and optionally removing punctuation.
    
    Args:
        text: The input clinical text
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()
    
    # Remove punctuation if specified
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


def expand_abbreviations(text: str, abbreviations: Optional[Dict[str, str]] = None) -> str:
    """
    Expand medical abbreviations in the text.
    
    Args:
        text: The input clinical text
        abbreviations: Dictionary of abbreviations to expand. If None, uses the default MEDICAL_ABBREVIATIONS
        
    Returns:
        Text with expanded abbreviations
    """
    if abbreviations is None:
        abbreviations = MEDICAL_ABBREVIATIONS
    
    # Create a regex pattern for word boundaries to match whole words only
    pattern = r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviations.keys()) + r')\b'
    
    # Replace abbreviations with their expanded forms
    expanded_text = re.sub(pattern, lambda match: abbreviations[match.group(0)], text, flags=re.IGNORECASE)
    
    return expanded_text


def preprocess_text(text: str, 
                   lowercase: bool = True, 
                   remove_punct: bool = False, 
                   expand_abbrev: bool = True,
                   abbreviations: Optional[Dict[str, str]] = None) -> str:
    """
    Preprocess clinical text by cleaning and optionally expanding abbreviations.
    
    Args:
        text: The input clinical text
        lowercase: Whether to convert text to lowercase
        remove_punct: Whether to remove punctuation
        expand_abbrev: Whether to expand medical abbreviations
        abbreviations: Dictionary of abbreviations to expand. If None, uses the default MEDICAL_ABBREVIATIONS
        
    Returns:
        Preprocessed text
    """
    # Clean the text
    cleaned_text = clean_text(text, lowercase=lowercase, remove_punct=remove_punct)
    
    # Expand abbreviations if specified
    if expand_abbrev:
        cleaned_text = expand_abbreviations(cleaned_text, abbreviations=abbreviations)
    
    return cleaned_text


def segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences, accounting for medical abbreviations.
    
    Args:
        text: The input clinical text
        
    Returns:
        List of sentences
    """
    # Handle common abbreviations that might confuse sentence splitting
    text = re.sub(r'(\b[A-Za-z]\.)(\s)', r'\1<POINT>\2', text)  # Handle single letter abbreviations
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    
    # Restore periods in abbreviations
    sentences = [re.sub(r'<POINT>', '.', s) for s in sentences]
    
    return sentences


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text into words, handling medical terms appropriately.
    
    Args:
        text: The input clinical text
        
    Returns:
        List of tokens
    """
    # Handle hyphenated terms and preserve them
    text = re.sub(r'(\w+)-(\w+)', r'\1_HYPHEN_\2', text)
    
    # Tokenize on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    
    # Restore hyphenated terms
    tokens = [re.sub(r'_HYPHEN_', '-', token) for token in tokens]
    
    return tokens 