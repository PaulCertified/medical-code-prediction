"""
Named Entity Recognition (NER) module for identifying medical entities in clinical text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class MedicalEntityRecognizer:
    """
    Named Entity Recognition model for identifying medical entities in clinical text.
    """
    
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", device: str = None):
        """
        Initialize the NER model.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Define entity labels
        self.id2label = {
            0: "O",  # Outside of a named entity
            1: "B-DIAGNOSIS",  # Beginning of a diagnosis entity
            2: "I-DIAGNOSIS",  # Inside of a diagnosis entity
            3: "B-PROCEDURE",  # Beginning of a procedure entity
            4: "I-PROCEDURE",  # Inside of a procedure entity
            5: "B-MEDICATION",  # Beginning of a medication entity
            6: "I-MEDICATION",  # Inside of a medication entity
            7: "B-ANATOMICAL_SITE",  # Beginning of an anatomical site entity
            8: "I-ANATOMICAL_SITE",  # Inside of an anatomical site entity
            9: "B-SEVERITY",  # Beginning of a severity entity
            10: "I-SEVERITY"  # Inside of a severity entity
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify medical entities in the given text.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            List of identified entities with their types and positions
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert token predictions to entity spans
        entities = self._convert_predictions_to_entities(text, inputs, predictions[0])
        
        return entities
    
    def _convert_predictions_to_entities(
        self, text: str, inputs: Dict[str, torch.Tensor], predictions: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Convert token-level predictions to entity spans.
        
        Args:
            text: Original input text
            inputs: Tokenizer inputs
            predictions: Model predictions
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        input_ids = inputs["input_ids"][0].cpu().numpy()
        
        # Get token to character mapping
        token_to_chars = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, truncation=True
        )["offset_mapping"]
        
        # Process predictions
        current_entity = None
        
        for i, (prediction, token_id) in enumerate(zip(predictions.cpu().numpy(), input_ids)):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                continue
                
            # Get the predicted label
            label = self.id2label[prediction]
            
            # If it's not an entity, close any open entity
            if label == "O" and current_entity:
                entities.append(current_entity)
                current_entity = None
                continue
            elif label == "O":
                continue
            
            # Extract entity type and position (B- or I-)
            entity_position, entity_type = label.split("-")
            
            # Get character span for this token
            start_char, end_char = token_to_chars[i]
            
            # If it's the beginning of a new entity
            if entity_position == "B":
                # Close any open entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start a new entity
                current_entity = {
                    "type": entity_type,
                    "text": text[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "confidence": float(torch.softmax(self.model.logits[0, i], dim=0)[prediction].cpu().numpy())
                }
            
            # If it's inside an entity
            elif entity_position == "I" and current_entity and current_entity["type"] == entity_type:
                # Extend the current entity
                current_entity["text"] += text[start_char:end_char]
                current_entity["end"] = end_char
                
                # Update confidence (average)
                current_confidence = float(torch.softmax(self.model.logits[0, i], dim=0)[prediction].cpu().numpy())
                current_entity["confidence"] = (current_entity["confidence"] + current_confidence) / 2
        
        # Add the last entity if there is one
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def fine_tune(
        self, 
        train_texts: List[str], 
        train_annotations: List[List[Dict[str, Any]]],
        validation_texts: Optional[List[str]] = None,
        validation_annotations: Optional[List[List[Dict[str, Any]]]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the NER model on medical data.
        
        Args:
            train_texts: List of training texts
            train_annotations: List of annotations for each training text
            validation_texts: List of validation texts
            validation_annotations: List of annotations for each validation text
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary of training metrics
        """
        # This is a placeholder for the fine-tuning logic
        # In a real implementation, this would:
        # 1. Convert annotations to token-level labels
        # 2. Create a dataset and dataloader
        # 3. Set up an optimizer and training loop
        # 4. Train the model and track metrics
        
        print(f"Fine-tuning model on {len(train_texts)} examples...")
        print(f"Parameters: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Placeholder for training metrics
        metrics = {
            "train_loss": [0.5, 0.3, 0.2],  # Placeholder values
            "val_loss": [0.6, 0.4, 0.3] if validation_texts else []
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage (note: this would use a placeholder model since we haven't fine-tuned yet)
    sample_text = "Patient diagnosed with type 2 diabetes mellitus and hypertension. Prescribed metformin 500mg twice daily."
    
    # In a real scenario, you would load a fine-tuned model
    # For demonstration, we'll just initialize the model with the base BioBERT
    # This won't give meaningful NER results without fine-tuning
    ner = MedicalEntityRecognizer()
    
    print("Sample text:")
    print(sample_text)
    print("\nIdentified entities (placeholder - would require fine-tuning for real results):")
    entities = ner.predict(sample_text)
    for entity in entities:
        print(f"{entity['type']}: {entity['text']} (confidence: {entity['confidence']:.2f})") 