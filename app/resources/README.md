# Ingredient Standardization Resources

This directory contains resources needed for ingredient standardization using the T5 model:

## Required Files

1. **ingredients.json** - List of ingredient names used for synthetic data generation
2. **modifiers.json** - List of ingredient modifiers (e.g., "diced", "chopped")
3. **units.json** - List of measurement units (e.g., "cup", "tablespoon")
4. **quantities.json** - Dictionary mapping quantity strings to numeric values

## Format

All files should be in JSON format:

- `ingredients.json`: A JSON array of strings
- `modifiers.json`: A JSON array of strings
- `units.json`: A JSON array of strings
- `quantities.json`: A JSON object mapping strings to numeric values

## Example

To create these files, you can extract them from the notebook using code like:

```python
import json

# Save ingredients
with open('ingredients.json', 'w') as f:
    json.dump(raw_ingredients_l, f)

# Save modifiers
with open('modifiers.json', 'w') as f:
    json.dump(mods_l, f)

# Save units
with open('units.json', 'w') as f:
    json.dump(units_l, f)

# Save quantities
with open('quantities.json', 'w') as f:
    json.dump(qty_dict, f)
```

The T5Extractor will automatically load these resources if they exist, and will use basic fallback values if they don't.
