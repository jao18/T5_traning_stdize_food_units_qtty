#!/usr/bin/env python3
"""
Script to diagnose resource issues and set up initial data for ingredient standardization.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.data_utils import extract_and_save_resources, generate_synthetic_examples

def diagnose_and_fix_resources():
    """Check for resource issues and fix them."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    resources_dir = base_dir / "resources"
    
    # Check if resources directory exists
    if not resources_dir.exists():
        print(f"Creating resources directory at {resources_dir}")
        os.makedirs(resources_dir, exist_ok=True)
    
    # Check for required resource files
    required_files = ['ingredients.json', 'modifiers.json', 'units.json', 'quantities.json']
    missing_files = []
    
    for filename in required_files:
        filepath = resources_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            # Check if file is valid JSON and has content
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    print(f"✓ {filename}: Found with {len(data) if isinstance(data, list) else 'valid'} items")
            except Exception as e:
                print(f"✗ {filename}: Error reading file: {e}")
                missing_files.append(filename)
    
    # If any files are missing, try to extract from notebook
    if missing_files:
        print(f"\nMissing resource files: {', '.join(missing_files)}")
        notebook_path = base_dir / "ingredient-standardization-via-machine-translation.ipynb"
        
        if notebook_path.exists():
            print(f"\nAttempting to extract resources from notebook: {notebook_path}")
            try:
                extract_and_save_resources(str(notebook_path), str(resources_dir))
                print("Resources extracted successfully")
            except Exception as e:
                print(f"Error extracting resources from notebook: {e}")
                create_default_resources(resources_dir, missing_files)
        else:
            print(f"Notebook not found at {notebook_path}")
            create_default_resources(resources_dir, missing_files)
    
    # Check if train.json exists and extract ingredients if needed
    train_json_path = base_dir / "app" / "input" / "train.json"
    if train_json_path.exists():
        print(f"\nFound train.json at {train_json_path}")
        print("Checking if we can extract additional ingredients...")
        
        ingredients_path = resources_dir / "ingredients.json"
        existing_ingredients = []
        
        # Load existing ingredients
        if ingredients_path.exists():
            try:
                with open(ingredients_path, 'r') as f:
                    existing_ingredients = json.load(f)
                print(f"Loaded {len(existing_ingredients)} existing ingredients")
            except:
                print("Error loading existing ingredients")
        
        # Extract ingredients from train.json
        try:
            with open(train_json_path, 'r') as f:
                recipes = json.load(f)
            
            raw_ingredients = set(existing_ingredients)
            
            # Extract ingredients from each recipe
            for recipe in recipes:
                if 'ingredients' in recipe and isinstance(recipe['ingredients'], list):
                    for ingredient in recipe['ingredients']:
                        if ingredient and isinstance(ingredient, str):
                            raw_ingredients.add(ingredient)
            
            # If we found new ingredients, save them
            raw_ingredients_list = list(raw_ingredients)
            if len(raw_ingredients_list) > len(existing_ingredients):
                print(f"Found {len(raw_ingredients_list) - len(existing_ingredients)} new ingredients")
                with open(ingredients_path, 'w') as f:
                    json.dump(raw_ingredients_list, f, indent=2)
                print(f"Saved {len(raw_ingredients_list)} total ingredients to {ingredients_path}")
            else:
                print("No new ingredients found")
                
        except Exception as e:
            print(f"Error processing train.json: {e}")
    
    # Generate example data
    print("\nGenerating example ingredient data:")
    try:
        examples = generate_synthetic_examples(3, str(resources_dir))
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")
    except Exception as e:
        print(f"Error generating examples: {e}")
    
    print("\nResource diagnosis complete.")
    print("""
Next steps:
1. Make sure all required resources are available
2. Fine-tune the model using: python app/scripts/fine_tune_check.py
3. Test the model using: python app/scripts/model_evaluation.py
4. Start the API: ./run.sh
""")

def create_default_resources(resources_dir, missing_files):
    """Create default resource files if extraction failed."""
    print("\nCreating default resource files...")
    
    resources = {
        'ingredients.json': [
            "onion", "garlic", "tomato", "potato", "carrot", "bell pepper", 
            "chicken breast", "ground beef", "rice", "pasta", "olive oil", 
            "salt", "pepper", "cumin", "oregano", "basil"
        ],
        'modifiers.json': [
            "diced", "chopped", "minced", "sliced", "grated", "shredded",
            "baked", "fried", "roasted", "grilled", "steamed", "boiled"
        ],
        'units.json': [
            "cup", "tablespoon", "teaspoon", "pound", "ounce", "gram", 
            "kilogram", "liter", "milliliter", "count", "pinch", "dash"
        ],
        'quantities.json': {
            "1/2": 0.5, "1/4": 0.25, "1/3": 0.333, "2/3": 0.666, "3/4": 0.75,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "half": 0.5, "quarter": 0.25
        }
    }
    
    for filename in missing_files:
        if filename in resources:
            filepath = resources_dir / filename
            with open(filepath, 'w') as f:
                json.dump(resources[filename], f, indent=2)
            print(f"Created default {filename} with {len(resources[filename])} items")

if __name__ == "__main__":
    diagnose_and_fix_resources()
