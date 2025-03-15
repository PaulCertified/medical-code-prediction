"""
Text preprocessing module for clinical text.
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict, Any, Optional

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class ClinicalTextProcessor:
    """
    Preprocesses clinical text for NLP tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text processor.
        
        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config or {}
        self.abbreviation_map = self._load_medical_abbreviations()
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """
        Load medical abbreviations mapping.
        
        Returns:
            Dictionary mapping abbreviations to their expanded forms
        """
        # This would typically load from a file, but for now we'll define a small set
        return {
            "pt": "patient",
            "dx": "diagnosis",
            "hx": "history",
            "tx": "treatment",
            "fx": "fracture",
            "sx": "symptoms",
            "abd": "abdominal",
            "afib": "atrial fibrillation",
            "bp": "blood pressure",
            "ca": "cancer",
            "chf": "congestive heart failure",
            "copd": "chronic obstructive pulmonary disease",
            "cva": "cerebrovascular accident",
            "dm": "diabetes mellitus",
            "mi": "myocardial infarction",
            "sob": "shortness of breath"
        }
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess clinical text.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase if specified in config
        if self.config.get('lowercase', False):
            text = text.lower()
        
        # Clean text
        text = self._clean_text(text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing unnecessary characters and normalizing whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations in the text.
        
        Args:
            text: Text with abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        words = word_tokenize(text)
        expanded_words = []
        
        for word in words:
            # Check if the word is in our abbreviation map
            lower_word = word.lower()
            if lower_word in self.abbreviation_map:
                # If the original word was capitalized, capitalize the expansion
                if word[0].isupper():
                    expanded = self.abbreviation_map[lower_word].capitalize()
                else:
                    expanded = self.abbreviation_map[lower_word]
                expanded_words.append(expanded)
            else:
                expanded_words.append(word)
        
        # Reconstruct the text
        expanded_text = ' '.join(expanded_words)
        
        return expanded_text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Text to segment
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)


if __name__ == "__main__":
    # Example usage
    sample_text = """Pt with hx of DM and CHF presented with SOB. 
    BP was elevated at 160/95. Abd exam was unremarkable.
    Dx: Acute exacerbation of CHF. Tx with diuretics initiated."""
    
    processor = ClinicalTextProcessor()
    processed_text = processor.preprocess(sample_text)
    
    print("Original text:")
    print(sample_text)
    print("\nProcessed text:")
    print(processed_text)
    
    print("\nSentences:")
    for i, sentence in enumerate(processor.segment_sentences(processed_text), 1):
        print(f"{i}. {sentence}") 