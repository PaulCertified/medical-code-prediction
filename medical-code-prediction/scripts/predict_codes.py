

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Get clinical text
    if args.text:
        clinical_text = args.text
    elif args.file:
        try:
            with open(args.file, "r") as f:
                clinical_text = f.read()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            return
    else:
        print("Please provide either --text or --file")
        return
    
    # Preprocess the text
    print("Preprocessing clinical text...")
    preprocessed_text = preprocess_text(
        clinical_text,
        lowercase=config["preprocessing"]["lowercase"],
        remove_punct=config["preprocessing"]["remove_punctuation"],
        expand_abbrev=config["preprocessing"]["expand_abbreviations"]
    )
    
    # Extract entities
    print("Extracting medical entities...")
    entities = extract_entities(preprocessed_text, entity_types=config["ner"]["labels"])
    
    # Print extracted entities
    print("
Extracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"
{entity_type}:")
            for entity in entity_list:
                print(f"  - {entity}")
    # Initialize the model
    print("
Initializing code prediction model...")
    model = CodePredictionModel(
        model_name=config["model"]["name"],
        max_length=config["model"]["max_length"],
        icd10_codes_path=args.icd10_codes,
        cpt_codes_path=args.cpt_codes
    )
    
    # Load a trained model if specified
    if args.model_path:
        model.load(args.model_path)
    
    # Predict codes
    print("Predicting codes...")
    predictions = model.predict(
        preprocessed_text,
        threshold=args.threshold,
        top_k=args.top_k,
        code_type=args.code_type
    )
    
    # Print predictions
    print("
Predicted Codes:")
    for pred in predictions:
        print(f"  - {pred["code"]} ({pred["type"]}): {pred["description"]} (Confidence: {pred["confidence"]:.2f})")
    # Explain the top prediction if available
    if predictions:
        top_code = predictions[0]["code"]
        print(f"
Explanation for top prediction ({top_code}):")
        explanation = model.explain(preprocessed_text, top_code)
        
        print(f"  Code: {explanation["code"]}")
        print(f"  Description: {explanation["description"]}")
        print(f"  Confidence: {explanation["confidence"]:.2f}")
        
        print("  Relevant text segments:")
        for segment in explanation["relevant_text"]:
            print(f"    - "{segment}"")
        
        print("  Feature importance:")
        for feature, importance in explanation["feature_importance"].items():
            print(f"    - {feature}: {importance:.2f}")


if __name__ == "__main__":
    main()
