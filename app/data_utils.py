#!/usr/bin/env python3
"""
Utility script to extract ingredient data from the notebook and save as resources for the app.
"""

import os
import json
import random
from typing import Dict, List, Any

def extract_and_save_resources(notebook_path: str, output_dir: str) -> None:
    """
    Extract ingredient data from the notebook and save as JSON files
    
    Args:
        notebook_path: Path to the ingredient-standardization-via-machine-translation.ipynb
        output_dir: Directory to save the extracted resources
    """
    import nbformat
    from nbformat import NotebookNode
    
    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the notebook
    print(f"Loading notebook from {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Initialize variables to extract
    raw_ingredients_l = []
    mods_l = []
    units_l = []
    qty_dict = {}
    
    # Helper function to extract variables from cell outputs
    def extract_var_from_cell(cell: NotebookNode, var_name: str) -> Any:
        """Extract a variable assignment from a code cell."""
        if cell.cell_type != 'code':
            return None
            
        source = cell.source
        if f"{var_name} = " in source or f"{var_name}=" in source:
            # Try to find and extract the variable
            exec_globals = {}
            try:
                exec(source, exec_globals)
                if var_name in exec_globals:
                    return exec_globals[var_name]
            except Exception as e:
                print(f"Error extracting {var_name}: {e}")
        return None
    
    # Process cells for variable extraction
    for cell in notebook.cells:
        # Try to extract each variable from cells
        if raw_ingredients_l == [] and 'raw_ingredients_l' in cell.source:
            extracted = extract_var_from_cell(cell, 'raw_ingredients_l')
            if extracted:
                raw_ingredients_l = extracted
                
        if mods_l == [] and 'mods_l' in cell.source:
            extracted = extract_var_from_cell(cell, 'mods_l')
            if extracted:
                mods_l = extracted
                
        if units_l == [] and 'units_l' in cell.source:
            extracted = extract_var_from_cell(cell, 'units_l')
            if extracted:
                units_l = extracted
                
        if not qty_dict and 'qty_dict' in cell.source:
            extracted = extract_var_from_cell(cell, 'qty_dict')
            if extracted:
                qty_dict = extracted
    
    # If extraction didn't work, create basic versions of these resources
    if not raw_ingredients_l:
        print("Could not extract ingredients, creating basic list")
        raw_ingredients_l = [
            "onion", "garlic", "tomato", "potato", "carrot", "bell pepper", 
            "chicken breast", "ground beef", "rice", "pasta", "olive oil", 
            "salt", "pepper", "cumin", "oregano", "basil"
        ]
    
    if not mods_l:
        print("Could not extract modifiers, creating comprehensive list")
        # First list of modifiers
        mods_l_1 = ['baked', 'blanched', 'blackened', 'braised', 'breaded', 'broiled', 'caramelized', 'charred', 'fermented', 'fried',
                 'glazed', 'infused', 'marinated', 'poached', 'roasted', 'sauteed', 'seared', 'smoked', 'whipped']
        
        # Second list of modifiers (more comprehensive)
        mods_l_2 = ['diced', 'battered', 'blackened', 'blanched', 'blended', 'boiled', 'boned', 'braised', 'brewed', 'broiled',
                   'browned', 'butterflied', 'candied', 'canned', 'caramelized', 'charred', 'chilled', 'chopped', 'clarified',
                   'condensed', 'creamed', 'crystalized', 'curdled', 'cured', 'curried', 'dehydrated', 'deviled', 'diluted',
                   'dredged', 'drenched', 'dried', 'drizzled', 'dry roasted', 'dusted', 'escalloped', 'evaporated', 'fermented',
                   'filled', 'folded', 'freeze dried', 'fricaseed', 'fried', 'glazed', 'granulated', 'grated', 'griddled', 'grilled',
                   'hardboiled', 'homogenized', 'kneaded', 'malted', 'mashed', 'minced', 'mixed', 'medium', 'small', 'large',
                   'packed', 'pan-fried', 'parboiled', 'parched', 'pasteurized', 'peppered', 'pickled', 'powdered', 'preserved',
                   'pulverized', 'pureed', 'redolent', 'reduced', 'refrigerated', 'chilled', 'roasted', 'rolled', 'salted',
                   'saturated', 'scalded', 'scorched', 'scrambled', 'seared', 'seasoned', 'shredded', 'skimmed', 'sliced',
                   'slivered', 'smothered', 'soaked', 'soft-boiled', 'hard-boiled', 'stewed', 'stuffed', 'toasted', 'whipped',
                   'wilted', 'wrapped']
        
        # Combine and deduplicate
        mods_set = set()
        for a_list in [mods_l_1, mods_l_2]:
            for mod in a_list:
                mods_set.add(mod)
        
        # Convert to list
        mods_l = list(mods_set)
    
    if not units_l:
        print("Could not extract units, creating basic list")
        units_l = [
            "cup", "tablespoon", "teaspoon", "pound", "ounce", "gram", 
            "kilogram", "liter", "milliliter", "count", "pinch", "dash"
        ]
    
    if not qty_dict:
        print("Could not extract quantity dictionary, creating basic version")
        qty_dict = {
            "1/2": 0.5, "1/4": 0.25, "1/3": 0.333, "2/3": 0.666, "3/4": 0.75,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "half": 0.5, "quarter": 0.25
        }
    
    # Save extracted data
    save_paths = {
        'ingredients.json': raw_ingredients_l,
        'modifiers.json': mods_l,
        'units.json': units_l,
        'quantities.json': qty_dict
    }
    
    for filename, data in save_paths.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} items to {filepath}")
    
    print(f"All resources extracted and saved to {output_dir}")

def generate_synthetic_examples(num_examples: int = 10, 
                              resources_dir: str = './resources') -> List[Dict]:
    """
    Generate synthetic examples for testing the T5 model
    
    Args:
        num_examples: Number of examples to generate
        resources_dir: Directory containing resource files
        
    Returns:
        List of dictionaries with input and expected output fields
    """
    # Load resources
    resources = {}
    for resource_name in ['ingredients', 'modifiers', 'units', 'quantities']:
        try:
            with open(f"{resources_dir}/{resource_name}.json", 'r') as f:
                resources[resource_name] = json.load(f)
        except Exception as e:
            print(f"Error loading {resource_name}: {e}")
            resources[resource_name] = []
    
    if not resources['ingredients'] or not resources['units']:
        raise ValueError("Cannot generate examples: resources not loaded")
    
    examples = []
    for _ in range(num_examples):
        # Pick random components
        qty_str, qty_val = random.choice(list(resources['quantities'].items()))
        unit = random.choice(resources['units'])
        ingredient = random.choice(resources['ingredients'])
        
        # Decide if we'll use a modifier
        use_mod = random.choice([True, False, False])
        if use_mod and resources['modifiers']:
            mod = random.choice(resources['modifiers'])
        else:
            mod = None
        
        # Decide if we'll put modifier at the end
        mod_at_end = random.choice([True, False])
        
        # Decide if we'll include units
        use_units = random.choice([True, True, True, True, False])
        if not use_units:
            unit_for_output = 'count'
        else:
            unit_for_output = unit
        
        # Build the raw text input
        if mod and use_units:
            if mod_at_end:
                raw_text = f"{qty_str} {unit} {ingredient}, {mod}"
            else:
                raw_text = f"{qty_str} {unit} {mod} {ingredient}"
        elif mod and not use_units:
            if mod_at_end:
                raw_text = f"{qty_str} {ingredient}, {mod}"
            else:
                raw_text = f"{qty_str} {mod} {ingredient}"
        elif not mod and use_units:
            raw_text = f"{qty_str} {unit} {ingredient}"
        else:
            raw_text = f"{qty_str} {ingredient}"
        
        # Build the standardized output
        if mod:
            std_output = f"{{ qty: {qty_val} , unit: {unit_for_output} , item: {ingredient} , mod: {mod} }}"
        else:
            std_output = f"{{ qty: {qty_val} , unit: {unit_for_output} , item: {ingredient} , mod: None }}"
        
        # Create the example
        examples.append({
            "input": raw_text,
            "output": std_output,
            "parsed": {
                "food_name": ingredient,
                "quantity": qty_val,
                "portion": unit_for_output if unit_for_output != 'count' else None,
                "modifier": mod
            }
        })
    
    return examples

def load_test_data(file_path=None):
    """
    Load test data from test.json file
    
    Args:
        file_path: Path to test.json file, or None to use default location
        
    Returns:
        List of dictionaries with ingredient data
    """
    if file_path is None:
        # Use default location: base_dir/app/input/test.json
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "app", "input", "test.json")
    
    if not os.path.exists(file_path):
        print(f"Warning: Test data file not found at {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} test examples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def generate_training_data_from_json(num_samples=1000, include_real_data=True, real_data_ratio=0.3):
    """
    Generate training data by combining synthetic and real examples from test.json
    
    Args:
        num_samples: Total number of samples to generate
        include_real_data: Whether to include real data from test.json
        real_data_ratio: Ratio of real data to include (0.0-1.0)
        
    Returns:
        Tuple of (X, Y) where X is raw ingredient text and Y is standardized format
    """
    # Initialize empty lists
    X = []
    Y = []
    
    # Load real data if requested
    real_data = []
    if include_real_data:
        real_data = load_test_data()
    
    # Determine how many real vs synthetic examples to use
    num_real = min(int(num_samples * real_data_ratio), len(real_data)) if real_data else 0
    num_synthetic = num_samples - num_real
    
    # Process real data if available
    if num_real > 0:
        print(f"Using {num_real} real examples from test.json")
        for i, item in enumerate(real_data[:num_real]):
            raw_text = item.get('raw_ingredient', '')
            if not raw_text:
                continue
                
            # Extract fields from the item
            quantity = item.get('quantity')
            unit = item.get('unit', 'count')
            food = item.get('food', '')
            modifier = item.get('comment', None)
            
            # Skip incomplete items
            if not food:
                continue
                
            # Format standardized output
            if modifier:
                Y.append(f"{{ qty: {quantity} , unit: {unit} , item: {food} , mod: {modifier} }}")
            else:
                Y.append(f"{{ qty: {quantity} , unit: {unit} , item: {food} , mod: None }}")
                
            X.append(raw_text)
    
    # Generate synthetic data for the remainder
    if num_synthetic > 0:
        print(f"Generating {num_synthetic} synthetic examples")
        # Load resources
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resources_dir = os.path.join(base_dir, "resources")
        
        # Make sure resources directory exists
        os.makedirs(resources_dir, exist_ok=True)
        
        # Load ingredient resources
        resources = {}
        for resource_name in ['ingredients', 'modifiers', 'units', 'quantities']:
            try:
                with open(f"{resources_dir}/{resource_name}.json", 'r') as f:
                    resources[resource_name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {resource_name}.json: {e}")
                
        # If resources are missing, use basic lists
        if not resources.get('ingredients'):
            resources['ingredients'] = ["tomato", "onion", "garlic", "chicken", "beef", "potato", "carrot"]
        
        if not resources.get('modifiers'):
            resources['modifiers'] = ["chopped", "diced", "minced", "sliced", "grated"]
            
        if not resources.get('units'):
            resources['units'] = ["cup", "tablespoon", "teaspoon", "pound", "ounce", "count"]
            
        if not resources.get('quantities'):
            resources['quantities'] = {
                "1/2": 0.5, "1/4": 0.25, "1/3": 0.333, "2/3": 0.666, "3/4": 0.75,
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
                "half": 0.5, "quarter": 0.25
            }
        
        # Generate synthetic examples
        for i in range(num_synthetic):
            # Pick random components
            qty_str, qty_val = random.choice(list(resources['quantities'].items()))
            unit = random.choice(resources['units'])
            ingredient = random.choice(resources['ingredients'])
            
            # Decide if we'll use a modifier
            use_mod = random.choice([True, False, False])
            if use_mod and resources['modifiers']:
                mod = random.choice(resources['modifiers'])
            else:
                mod = None
            
            # Decide if we'll put modifier at the end
            mod_at_end = random.choice([True, False])
            
            # Decide if we'll include units
            use_units = random.choice([True, True, True, True, False])
            if not use_units:
                unit_for_output = 'count'
            else:
                unit_for_output = unit
            
            # Build the raw text input
            if mod and use_units:
                if mod_at_end:
                    raw_text = f"{qty_str} {unit} {ingredient}, {mod}"
                else:
                    raw_text = f"{qty_str} {unit} {mod} {ingredient}"
            elif mod and not use_units:
                if mod_at_end:
                    raw_text = f"{qty_str} {ingredient}, {mod}"
                else:
                    raw_text = f"{qty_str} {mod} {ingredient}"
            elif not mod and use_units:
                raw_text = f"{qty_str} {unit} {ingredient}"
            else:
                raw_text = f"{qty_str} {ingredient}"
            
            # Build the standardized output
            if mod:
                std_output = f"{{ qty: {qty_val} , unit: {unit_for_output} , item: {ingredient} , mod: {mod} }}"
            else:
                std_output = f"{{ qty: {qty_val} , unit: {unit_for_output} , item: {ingredient} , mod: None }}"
            
            # Add to dataset
            X.append(raw_text)
            Y.append(std_output)
    
    print(f"Generated total of {len(X)} training examples ({num_real} real, {len(X)-num_real} synthetic)")
    return X, Y

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ingredient data from notebook")
    parser.add_argument('--notebook', type=str, required=True, 
                        help='Path to ingredient-standardization-via-machine-translation.ipynb')
    parser.add_argument('--output', type=str, default='./resources',
                        help='Directory to save the extracted resources')
    
    args = parser.parse_args()
    extract_and_save_resources(args.notebook, args.output)
    
    # Generate and print some example data
    print("\nGenerating 3 example ingredient texts:")
    examples = generate_synthetic_examples(3, args.output)
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Input: {example['input']}")
        print(f"Output: {example['output']}")
        print(f"Parsed: {example['parsed']}")
