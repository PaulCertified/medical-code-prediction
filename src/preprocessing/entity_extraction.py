"""
Entity extraction module for identifying medical entities in clinical text.
"""
from typing import Dict, List, Optional, Tuple, Union

import re
from .text_preprocessing import segment_sentences


# Regular expressions for common medical entities
REGEX_PATTERNS = {
    "DIAGNOSIS": [
        r'\b(?:diagnosed with|diagnosis of|assessment of|impression of|assessment:|impression:|diagnosis:|dx:)\s+([\w\s\-\,]+)',
        r'\b(?:suffers from|suffering from|known case of|has a history of)\s+([\w\s\-\,]+)',
        r'\b(?:presented with|presents with|complains of|complained of|reports|reported)\s+([\w\s\-\,]+)',
    ],
    "PROCEDURE": [
        r'\b(?:underwent|undergoing|scheduled for|performed|will undergo|had|has had)\s+([\w\s\-\,]+(?:surgery|procedure|operation|repair|replacement|resection|biopsy|implantation|removal|excision|amputation|transplantation|bypass|angioplasty|catheterization|endoscopy|colonoscopy|bronchoscopy|arthroscopy))',
        r'\b(?:status post|s/p)\s+([\w\s\-\,]+(?:surgery|procedure|operation|repair|replacement|resection|biopsy|implantation|removal|excision|amputation|transplantation|bypass|angioplasty|catheterization|endoscopy|colonoscopy|bronchoscopy|arthroscopy))',
    ],
    "MEDICATION": [
        r'\b(?:prescribed|taking|started on|continues on|maintained on|given|administered|received)\s+([\w\s\-\,]+(?:mg|mcg|g|ml|units|tabs|capsules|pills|patch|injection|infusion|solution|suspension|syrup|cream|ointment|gel|spray|inhaler|nebulizer))',
        r'\b(?:medication:|medications:|meds:|current medications:|med list:|medication list:)\s+([\w\s\-\,]+)',
    ],
    "ANATOMY": [
        r'\b(?:in the|of the|at the|on the|involving the|affecting the)\s+([\w\s\-\,]+(?:heart|lung|liver|kidney|brain|spine|spinal cord|stomach|intestine|colon|rectum|bladder|uterus|ovary|testicle|prostate|breast|skin|muscle|bone|joint|artery|vein|nerve|eye|ear|nose|throat|mouth|tongue|esophagus|trachea|bronchus|pancreas|gallbladder|adrenal|thyroid|pituitary|hypothalamus|cerebellum|cerebrum|cortex|ventricle|atrium|aorta|carotid|femoral|radial|ulnar|tibial|fibular|humerus|radius|ulna|femur|tibia|fibula|patella|calcaneus|talus|metatarsal|phalanx|cranium|mandible|maxilla|clavicle|scapula|sternum|rib|vertebra|pelvis|ilium|ischium|pubis|sacrum|coccyx))',
    ],
    "SEVERITY": [
        r'\b(mild|moderate|severe|critical|extreme|minimal|significant|marked|pronounced|substantial|considerable|extensive|profound|slight|minor|major)\s+([\w\s\-\,]+)',
    ],
    "DURATION": [
        r'\b(?:for|over|during|throughout|within|after|before|since|lasting|persisting)\s+([\w\s\-\,]+(?:day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|second|seconds))',
        r'\b(acute|chronic|subacute|recurrent|persistent|intermittent|transient|episodic|paroxysmal|constant|continuous|ongoing|longstanding)\s+([\w\s\-\,]+)',
    ],
}


def extract_entities_with_regex(text: str, entity_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract medical entities from text using regular expressions.
    
    Args:
        text: The input clinical text
        entity_types: List of entity types to extract. If None, extracts all entity types.
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    if entity_types is None:
        entity_types = list(REGEX_PATTERNS.keys())
    
    # Split text into sentences for better context
    sentences = segment_sentences(text)
    
    # Initialize results dictionary
    entities = {entity_type: [] for entity_type in entity_types}
    
    # Process each sentence
    for sentence in sentences:
        for entity_type in entity_types:
            if entity_type in REGEX_PATTERNS:
                for pattern in REGEX_PATTERNS[entity_type]:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        # Extract the entity (group 1 contains the actual entity)
                        entity = match.group(1).strip()
                        if entity and entity not in entities[entity_type]:
                            entities[entity_type].append(entity)
    
    return entities


def extract_entities(text: str, entity_types: Optional[List[str]] = None, 
                    use_transformer: bool = False, model_name: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Extract medical entities from text using either regex patterns or a transformer model.
    
    Args:
        text: The input clinical text
        entity_types: List of entity types to extract. If None, extracts all entity types.
        use_transformer: Whether to use a transformer model for extraction
        model_name: Name of the transformer model to use. Required if use_transformer is True.
        
    Returns:
        Dictionary mapping entity types to lists of extracted entities
    """
    if use_transformer:
        if model_name is None:
            raise ValueError("model_name must be provided when use_transformer is True")
        
        # This is a placeholder for transformer-based entity extraction
        # In a real implementation, this would load and use a transformer model
        # such as BioBERT fine-tuned for medical NER
        print(f"Using transformer model {model_name} for entity extraction")
        print("Note: Transformer-based extraction is not implemented yet")
        
        # For now, fall back to regex-based extraction
        return extract_entities_with_regex(text, entity_types)
    else:
        return extract_entities_with_regex(text, entity_types)


def filter_entities(entities: Dict[str, List[str]], min_length: int = 3, 
                   exclude_words: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Filter extracted entities based on length and excluded words.
    
    Args:
        entities: Dictionary mapping entity types to lists of extracted entities
        min_length: Minimum length of entities to keep
        exclude_words: List of words to exclude from entities
        
    Returns:
        Filtered dictionary of entities
    """
    if exclude_words is None:
        exclude_words = ["the", "and", "with", "without", "from", "to", "in", "on", "at", "by", "for", "of", "a", "an"]
    
    filtered_entities = {}
    
    for entity_type, entity_list in entities.items():
        filtered_list = []
        for entity in entity_list:
            # Check if entity meets minimum length
            if len(entity) >= min_length:
                # Check if entity is not just an excluded word
                if entity.lower() not in exclude_words:
                    filtered_list.append(entity)
        
        filtered_entities[entity_type] = filtered_list
    
    return filtered_entities


def normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Normalize extracted entities by removing duplicates, standardizing format, etc.
    
    Args:
        entities: Dictionary mapping entity types to lists of extracted entities
        
    Returns:
        Normalized dictionary of entities
    """
    normalized_entities = {}
    
    for entity_type, entity_list in entities.items():
        normalized_list = []
        for entity in entity_list:
            # Convert to lowercase
            normalized_entity = entity.lower()
            
            # Remove trailing punctuation
            normalized_entity = re.sub(r'[.,;:!?]+$', '', normalized_entity)
            
            # Remove leading articles
            normalized_entity = re.sub(r'^(a|an|the)\s+', '', normalized_entity)
            
            # Add to list if not already present
            if normalized_entity and normalized_entity not in normalized_list:
                normalized_list.append(normalized_entity)
        
        normalized_entities[entity_type] = normalized_list
    
    return normalized_entities 