#!/usr/bin/env python
# examples/process_ingredients.py

"""
Example script showing how to use the ingredient processor directly
without going through the FastAPI interface.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path so we can import the app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model.t5_extractor import T5Extractor
from app.database.loader import FoodDatabase
from app.database.searcher import FoodSearcher
from app.core.processor import IngredientProcessor

def process_examples():
    """Process some example ingredients in both English and French"""
    
    # Paths to database files
    data_dir = os.environ.get("DATA_DIR", "./data")
    food_path = os.path.join(data_dir, "foods.json")
    conversion_path = os.path.join(data_dir, "conversion_factors.json")
    measure_path = os.path.join(data_dir, "measures.json")
    
    # Initialize components
    print("Loading T5 model...")
    t5_extractor = T5Extractor(model_name="google/mt5-small")
    
    print(f"Loading food database from {data_dir}...")
    food_database = FoodDatabase(
        food_path=food_path,
        conversion_path=conversion_path,
        measure_path=measure_path
    )
    
    food_searcher = FoodSearcher(food_database)
    ingredient_processor = IngredientProcessor(t5_extractor, food_searcher)
    
    # Example ingredients in English
    english_examples = [
        "2 cups of cold tomato juice",
        "1 tablespoon olive oil",
        "3 large eggs",
        "1/2 cup diced onions",
        "250g lean ground beef"
    ]
    
    # Example ingredients in French
    french_examples = [
        "2 tasses de jus de tomate froid",
        "1 cuillère à soupe d'huile d'olive",
        "3 gros œufs",
        "1/2 tasse d'oignons coupés en dés",
        "250g de bœuf haché maigre"
    ]
    
    # Process English examples
    print("\nProcessing English examples...")
    english_results = ingredient_processor.process_ingredients(english_examples, language="en")
    formatted_english = [ingredient_processor.format_matched_ingredient(result) for result in english_results]
    
    # Process French examples
    print("\nProcessing French examples...")
    french_results = ingredient_processor.process_ingredients(french_examples, language="fr")
    formatted_french = [ingredient_processor.format_matched_ingredient(result) for result in french_results]
    
    # Print results
    print("\n===== ENGLISH RESULTS =====")
    for i, (example, result) in enumerate(zip(english_examples, formatted_english)):
        print(f"\nExample {i+1}: {example}")
        print(json.dumps(result, indent=2))
    
    print("\n===== FRENCH RESULTS =====")
    for i, (example, result) in enumerate(zip(french_examples, formatted_french)):
        print(f"\nExample {i+1}: {example}")
        print(json.dumps(result, indent=2))
        
    # Example recipe
    example_recipe = """
    Classic Pancakes
    
    Ingredients:
    - 1 cup all-purpose flour
    - 2 tablespoons sugar
    - 1 teaspoon baking powder
    - 1/2 teaspoon salt
    - 1 cup milk
    - 1 large egg
    - 2 tablespoons unsalted butter, melted
    
    Instructions:
    1. Whisk together dry ingredients...
    """
    
    print("\n===== RECIPE PROCESSING =====")
    recipe_result = ingredient_processor.process_recipe(example_recipe)
    formatted_recipe = {
        "raw_text": recipe_result["raw_text"],
        "language": recipe_result["language"],
        "ingredients": [ingredient_processor.format_matched_ingredient(ing) for ing in recipe_result["ingredients"]]
    }
    print(json.dumps(formatted_recipe, indent=2))

if __name__ == "__main__":
    process_examples()