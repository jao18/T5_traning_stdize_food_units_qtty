#!/usr/bin/env python3
"""
Script to evaluate the T5 model's performance on ingredient standardization.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.t5_extractor import T5Extractor

def evaluate_model(model_path=None):
    """
    Evaluate the T5 model on ingredient standardization tasks.
    
    Args:
        model_path: Optional path to a fine-tuned model
    """
    # Initialize model
    if model_path:
        print(f"Loading fine-tuned model from {model_path}...")
        model = T5Extractor(model_name=model_path)
    else:
        print("Using pre-trained T5 model...")
        model = T5Extractor()
    
    # Test examples for evaluation
    test_examples = [
        "1 cup chopped tomatoes",
        "2 tbsp olive oil",
        "3 cloves of garlic, minced",
        "1/2 pound ground beef",
        "1 teaspoon salt",
        "2 large eggs, beaten",
        "1/4 cup shredded cheddar cheese",
        "3 tablespoons butter, melted",
        "one medium onion, diced",
        "2 cups chicken broth, low-sodium"
    ]
    
    # Evaluate on test examples
    print("\nEvaluating on test examples:")
    for example in test_examples:
        result = model.extract_fields(example)
        formatted = f"{{ qty: {result.get('quantity')} , unit: {result.get('portion') or 'count'} , item: {result.get('food_name')} , mod: {result.get('modifier') or 'None'} }}"
        print(f"\nInput: {example}")
        print(f"Output: {formatted}")
        print(f"Parsed: {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate T5 model for ingredient standardization")
    parser.add_argument("--model-path", type=str, help="Path to fine-tuned model directory")
    
    args = parser.parse_args()
    evaluate_model(args.model_path)
