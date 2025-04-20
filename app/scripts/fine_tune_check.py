#!/usr/bin/env python3
"""
Script to fine-tune the T5 model and evaluate its performance on ingredient standardization.
Memory-optimized version for machines with limited resources.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.t5_extractor import T5Extractor
from app.data_utils import generate_training_data_from_json

def fine_tune_and_evaluate():
    """Fine-tune the T5 model with memory-efficient settings and evaluate its performance."""
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    resources_dir = base_dir / "resources"
    output_dir = base_dir / "fine_tuned_models" / "t5_ingredients"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model with a smaller variant
    print("Initializing T5 model (smaller variant for memory efficiency)...")
    model = T5Extractor(model_name="google/t5-small")
    
    # Generate training data mixing real and synthetic examples
    print("Generating training data (with real ingredients from test.json)...")
    train_samples = 1000  # Reduced from 5000 for memory efficiency
    x, y = generate_training_data_from_json(
        num_samples=train_samples,
        include_real_data=True,
        real_data_ratio=0.4  # Use 40% real data if available
    )
    
    # Format into required structure for fine-tuning
    train_data = [
        {"input": f"standardize ingredient: {x_i}", "output": y_i}
        for x_i, y_i in zip(x, y)
    ]
    
    # Split into train/eval datasets
    train_size = int(0.9 * len(train_data))
    eval_data = train_data[train_size:]
    train_data = train_data[:train_size]
    
    print(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples")
    
    # Fine-tune the model with memory-efficient settings
    print("Starting memory-efficient fine-tuning...")
    model.fine_tune(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=str(output_dir),
        epochs=2,  # Reduced epochs for faster training
        batch_size=2,  # Smaller batch size for lower memory usage
        gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batches
        save_steps=50,  # Save more frequently to avoid losing progress
        mixed_precision="fp16"  # Use mixed precision to save memory
    )
    
    # Test examples for evaluation
    test_examples = [
        "1 cup chopped tomatoes",
        "2 tbsp olive oil",
        "3 cloves of garlic, minced",
        "1/2 pound ground beef"
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
    fine_tune_and_evaluate()
