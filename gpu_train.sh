#!/bin/bash
# Script to run T5 model training on GPU with optimized settings

echo "Setting up environment for GPU training..."
pip install -q torch==2.0.0 accelerate>=0.26.0 deepspeed>=0.8.0 bitsandbytes>=0.37.0 datasets>=2.10.0

# Check for GPU
if [ -z "$(nvidia-smi 2>/dev/null)" ]; then
    echo "Warning: No GPU detected. Training will be slow."
    DEVICE="cpu"
else
    echo "GPU detected. Using CUDA for training."
    DEVICE="cuda"
    # Print GPU info
    nvidia-smi
fi

# Create directories
mkdir -p fine_tuned_models/t5_ingredients_gpu
mkdir -p resources

echo "Preparing resources..."
# Extract resources if needed
if [ ! -f "resources/ingredients.json" ]; then
    echo "Generating basic resources..."
    python -m app.scripts.diagnose_resources
fi

# Check if test.json exists, create input directory if it doesn't
if [ ! -d "app/input" ]; then
    echo "Creating input directory..."
    mkdir -p app/input
fi

# Create a basic test.json if it doesn't exist
if [ ! -f "app/input/test.json" ]; then
    echo "Creating basic test.json with sample data..."
    cat > app/input/test.json << EOF
[
  {
    "raw_ingredient": "1 cup diced tomatoes",
    "quantity": 1,
    "unit": "cup",
    "food": "tomatoes",
    "comment": "diced"
  },
  {
    "raw_ingredient": "2 tablespoons olive oil",
    "quantity": 2,
    "unit": "tablespoon",
    "food": "olive oil"
  },
  {
    "raw_ingredient": "3 cloves garlic, minced",
    "quantity": 3,
    "unit": "count",
    "food": "garlic",
    "comment": "minced"
  }
]
EOF
    echo "Created sample test.json"
fi

echo "Starting optimized T5 training on $DEVICE..."
python -m app.scripts.optimized_training \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --samples 2000 \
    --model "google/t5-small" \
    --epochs 3 \
    --save-steps 100

echo "Training complete. Evaluating model..."
python -m app.scripts.model_evaluation --model-path fine_tuned_models/t5_ingredients_optimized

echo "Done!"
