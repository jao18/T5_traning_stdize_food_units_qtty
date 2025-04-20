#!/usr/bin/env python3
"""
Script for memory-efficient fine-tuning of T5 models on limited hardware.
Uses various optimization techniques to reduce memory usage:
- 8-bit quantization
- Gradient accumulation
- GPU optimization for T4
- DeepSpeed optimization
- Smaller model variant (t5-small)
"""

import sys
import os
import json
from pathlib import Path
import torch

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.t5_extractor import T5Extractor
from app.data_utils import generate_training_data_from_json

def optimized_fine_tune(
    batch_size=4,  # Increased for GPU
    gradient_accumulation_steps=4,  # Reduced for GPU
    train_samples=2000,  # Increased for better training
    model_name="google/t5-small",
    epochs=3,
    save_steps=100,
    use_8bit=True,
    use_deepspeed=True,
    use_real_data=True
):
    """Fine-tune T5 with memory-efficient settings optimized for Tesla T4 GPU."""
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    resources_dir = base_dir / "resources"
    output_dir = base_dir / "fine_tuned_models" / "t5_ingredients_optimized"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU and print device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    else:
        print("No GPU detected, using CPU")
    
    # Print optimization configuration
    print(f"Starting optimized training with the following configuration:")
    print(f"- Model: {model_name}")
    print(f"- Device: {device}")
    print(f"- Batch size: {batch_size}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"- Training samples: {train_samples}")
    print(f"- Using real data from test.json: {use_real_data}")
    print(f"- 8-bit quantization: {use_8bit}")
    print(f"- DeepSpeed optimization: {use_deepspeed}")
    print(f"- Saving checkpoints every {save_steps} steps")
    
    # Initialize model with optimizations
    print("\nInitializing T5 model with memory optimizations...")
    model = T5Extractor(
        model_name=model_name, 
        use_8bit=use_8bit,
        use_deepspeed=use_deepspeed
    )
    
    # Generate training data (mix of real and synthetic)
    print(f"\nGenerating {train_samples} training samples...")
    x, y = generate_training_data_from_json(
        num_samples=train_samples,
        include_real_data=use_real_data,
        real_data_ratio=0.4 if use_real_data else 0  # Use 40% real data if available
    )
    
    # Format data for training
    train_data = [
        {"input": f"standardize ingredient: {x_i}", "output": y_i}
        for x_i, y_i in zip(x, y)
    ]
    
    # Split into train/eval datasets (90/10 split)
    train_size = int(0.9 * len(train_data))
    eval_data = train_data[train_size:]
    train_data = train_data[:train_size]
    
    print(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples")
    
    # Configure DeepSpeed specifically for T4 GPU if available
    deepspeed_config = None
    if use_deepspeed and device == "cuda":
        deepspeed_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 50,
            "wall_clock_breakdown": False
        }
    
    # Fine-tune with GPU optimizations
    print("\nStarting GPU-optimized fine-tuning...")
    model.fine_tune(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        mixed_precision="fp16",  # Use mixed precision for speed and memory savings
        deepspeed_config=deepspeed_config  # Pass custom config for T4
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
    
    print(f"\nModel fine-tuned and saved to {output_dir}")
    print("\nTo use this model:")
    print(f"python app/scripts/model_evaluation.py --model-path {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-efficient T5 fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (use 4-8 for T4 GPU)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--samples", type=int, default=2000, help="Number of training samples to generate")
    parser.add_argument("--model", type=str, default="google/t5-small", help="Model name (t5-small recommended)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")
    parser.add_argument("--no-deepspeed", action="store_true", help="Disable DeepSpeed optimization")
    parser.add_argument("--no-real-data", action="store_true", help="Don't use real data from test.json")
    
    args = parser.parse_args()
    
    optimized_fine_tune(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        train_samples=args.samples,
        model_name=args.model,
        epochs=args.epochs,
        save_steps=args.save_steps,
        use_8bit=not args.no_8bit,
        use_deepspeed=not args.no_deepspeed,
        use_real_data=not args.no_real_data
    )
#!/usr/bin/env python3
"""
Script for memory-efficient fine-tuning of T5 models on limited hardware.
Uses various optimization techniques to reduce memory usage:
- 8-bit quantization
- Gradient accumulation
- GPU optimization for T4
- DeepSpeed optimization
- Smaller model variant (t5-small)
"""

import sys
import os
import json
from pathlib import Path
import torch

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.model.t5_extractor import T5Extractor
from app.data_utils import generate_synthetic_examples

def optimized_fine_tune(
    batch_size=4,  # Increased for GPU
    gradient_accumulation_steps=4,  # Reduced for GPU
    train_samples=2000,  # Increased for better training
    model_name="google/t5-small",
    epochs=3,
    save_steps=100,
    use_8bit=True,
    use_deepspeed=True
):
    """Fine-tune T5 with memory-efficient settings optimized for Tesla T4 GPU."""
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    resources_dir = base_dir / "resources"
    output_dir = base_dir / "fine_tuned_models" / "t5_ingredients_optimized"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU and print device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    else:
        print("No GPU detected, using CPU")
    
    # Print optimization configuration
    print(f"Starting optimized training with the following configuration:")
    print(f"- Model: {model_name}")
    print(f"- Device: {device}")
    print(f"- Batch size: {batch_size}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"- Training samples: {train_samples}")
    print(f"- 8-bit quantization: {use_8bit}")
    print(f"- DeepSpeed optimization: {use_deepspeed}")
    print(f"- Saving checkpoints every {save_steps} steps")
    
    # Initialize model with optimizations
    print("\nInitializing T5 model with memory optimizations...")
    model = T5Extractor(
        model_name=model_name, 
        use_8bit=use_8bit,
        use_deepspeed=use_deepspeed
    )
    
    # Generate synthetic training data (using fewer samples)
    print(f"\nGenerating {train_samples} synthetic training samples...")
    x, y = model.generate_synthetic_data(train_samples)
    
    # Format data for training
    train_data = [
        {"input": f"standardize ingredient: {x_i}", "output": y_i}
        for x_i, y_i in zip(x, y)
    ]
    
    # Split into train/eval datasets (90/10 split)
    train_size = int(0.9 * len(train_data))
    eval_data = train_data[train_size:]
    train_data = train_data[:train_size]
    
    print(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples")
    
    # Configure DeepSpeed specifically for T4 GPU if available
    deepspeed_config = None
    if use_deepspeed and device == "cuda":
        deepspeed_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 50,
            "wall_clock_breakdown": False
        }
    
    # Fine-tune with GPU optimizations
    print("\nStarting GPU-optimized fine-tuning...")
    model.fine_tune(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        mixed_precision="fp16",  # Use mixed precision for speed and memory savings
        deepspeed_config=deepspeed_config  # Pass custom config for T4
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
    
    print(f"\nModel fine-tuned and saved to {output_dir}")
    print("\nTo use this model:")
    print(f"python app/scripts/model_evaluation.py --model-path {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory-efficient T5 fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (use 4-8 for T4 GPU)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--samples", type=int, default=2000, help="Number of training samples to generate")
    parser.add_argument("--model", type=str, default="google/t5-small", help="Model name (t5-small recommended)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")
    parser.add_argument("--no-deepspeed", action="store_true", help="Disable DeepSpeed optimization")
    
    args = parser.parse_args()
    
    optimized_fine_tune(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        train_samples=args.samples,
        model_name=args.model,
        epochs=args.epochs,
        save_steps=args.save_steps,
        use_8bit=not args.no_8bit,
        use_deepspeed=not args.no_deepspeed
    )
