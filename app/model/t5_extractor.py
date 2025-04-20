# app/model/t5_extractor.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import random
import pandas as pd
import os
import json
from typing import List, Dict, Tuple, Union, Optional

class T5Extractor:
    def __init__(self, model_name="google/mt5-small", device=None, use_8bit=False, use_deepspeed=False):
        """
        Initialize the T5 model for ingredient extraction
                pip install -r requirements.txt
        Args:
            model_name: The T5 model variant to use (default: google/mt5-small)
            device: Device to use (cuda/cpu), will auto-detect if None
            use_8bit: Whether to use 8-bit quantization for memory efficiency
            use_deepspeed: Whether to use DeepSpeed optimization
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Apply optimizations if requested
        self.use_8bit = use_8bit
        self.use_deepspeed = use_deepspeed
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load model with memory optimizations if requested
        if use_8bit:
            try:
                import bitsandbytes as bnb
                print("Using 8-bit quantization for memory efficiency")
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, 
                    load_in_8bit=True,
                    device_map="auto"
                )
                # Set device to auto when using 8-bit quantization
                self.device = "auto"
            except ImportError:
                print("Warning: bitsandbytes not available, falling back to default precision")
                self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Language-specific prompts
        self.prompts = {
            "en": "standardize ingredient: ",
            "fr": "standardiser l'ingrédient: "
        }
        
        # Load ingredient data, modifiers, units and quantities if available
        self.raw_ingredients_l = []
        self.mods_l = []
        self.units_l = []
        self.qty_dict = {}
        
        # Try to load the data resources
        self._load_resources()

    def _load_resources(self):
        """Load ingredient resources if available"""
        try:
            # Paths to resource files (adjust as needed)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            resources_dir = os.path.join(base_dir, 'resources')
            
            # Load ingredients if available
            ingredients_path = os.path.join(resources_dir, 'ingredients.json')
            if os.path.exists(ingredients_path):
                with open(ingredients_path, 'r') as f:
                    self.raw_ingredients_l = json.load(f)
            
            # Load modifiers if available
            modifiers_path = os.path.join(resources_dir, 'modifiers.json')
            if os.path.exists(modifiers_path):
                with open(modifiers_path, 'r') as f:
                    self.mods_l = json.load(f)
            
            # Load units if available
            units_path = os.path.join(resources_dir, 'units.json')
            if os.path.exists(units_path):
                with open(units_path, 'r') as f:
                    self.units_l = json.load(f)
            
            # Load quantities if available
            quantities_path = os.path.join(resources_dir, 'quantities.json')
            if os.path.exists(quantities_path):
                with open(quantities_path, 'r') as f:
                    self.qty_dict = json.load(f)
            
            print(f"Loaded resources: {len(self.raw_ingredients_l)} ingredients, {len(self.mods_l)} modifiers, {len(self.units_l)} units")
        except Exception as e:
            print(f"Error loading resources: {e}")
            # Initialize with some basic values if resources aren't available
            self._init_basic_resources()
    
    def _init_basic_resources(self):
        """Initialize basic resources when full data isn't available"""
        # Basic units
        self.units_l = ['cup', 'tablespoon', 'teaspoon', 'pound', 'ounce', 'count']
        
        # Comprehensive list of modifiers for food preparation
        mods_l_1 = ['baked', 'blanched', 'blackened', 'braised', 'breaded', 'broiled', 'caramelized', 'charred', 'fermented', 'fried',
                 'glazed', 'infused', 'marinated', 'poached', 'roasted', 'sauteed', 'seared', 'smoked', 'whipped']
        
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
        
        # Create a set to eliminate duplicates
        mods_set = set()
        for a_list in [mods_l_1, mods_l_2]:
            for mod in a_list:
                mods_set.add(mod)
        
        # Convert back to list
        self.mods_l = list(mods_set)
        
        # Basic quantity mappings
        self.qty_dict = {
            "1/2": 0.5, "1/4": 0.25, "1/3": 0.333, "2/3": 0.666, "3/4": 0.75,
            "half": 0.5, "quarter": 0.25, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
        }

    def extract_fields(self, text: str, language: str = "en") -> dict:
        """
        Extract structured fields from raw ingredient text
        
        Args:
            text: Raw ingredient text (e.g. "2 cups of cold tomato juice")
            language: Language code ("en" or "fr")
            
        Returns:
            Dict with food_name, quantity, portion, and modifier fields
        """
        prompt = f"{self.prompts.get(language, self.prompts['en'])}{text}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate structured output
        outputs = self.model.generate(
            **inputs, 
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse the standardized output
        return self._parse_standardized_output(decoded)

    def _parse_standardized_output(self, decoded_text: str) -> dict:
        """
        Parse the standardized output from T5 into structured fields
        
        Args:
            decoded_text: The text output from the T5 model
            
        Returns:
            Dict with extracted fields
        """
        # Default empty result
        result = {
            "food_name": None,
            "quantity": None,
            "portion": None,
            "modifier": None
        }
        
        # Try to parse JSON-like structure from the notebook format
        # Example: "{ qty: 1 , unit: cup , item: red bell pepper , mod: diced }"
        try:
            # Extract components using regex
            qty_match = re.search(r'qty:\s*([0-9.]+)', decoded_text)
            unit_match = re.search(r'unit:\s*([^,}]+)', decoded_text)
            item_match = re.search(r'item:\s*([^,}]+)', decoded_text)
            mod_match = re.search(r'mod:\s*([^,}]+)', decoded_text)
            
            if qty_match:
                result["quantity"] = float(qty_match.group(1).strip())
            
            if unit_match:
                unit = unit_match.group(1).strip()
                if unit.lower() == 'none':
                    unit = None
                result["portion"] = unit
            
            if item_match:
                result["food_name"] = item_match.group(1).strip()
            
            if mod_match:
                mod = mod_match.group(1).strip()
                if mod.lower() == 'none':
                    mod = None
                result["modifier"] = mod
                
        except Exception as e:
            print(f"Error parsing standardized output: {e}")
            # Fallback to basic parsing
            parts = decoded_text.split(',')
            if len(parts) >= 1:
                result["food_name"] = parts[0].strip()
                
        # If still no food name, use the whole text
        if not result["food_name"]:
            result["food_name"] = decoded_text.strip()
            
        return result

    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[List[str], List[str]]:
        """
        Generate synthetic training data for ingredient standardization
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (X, Y) where X is raw ingredient text and Y is standardized format
        """
        # Similar to gen_ing_data from the notebook
        if not self.raw_ingredients_l or not self.units_l or not self.mods_l:
            raise ValueError("Cannot generate synthetic data: resources not loaded")
            
        X = [0] * num_samples
        Y = [0] * num_samples
        
        for i in range(num_samples):
            # Pick random components
            rnd_qty_str, rnd_qty_int = random.choice(list(self.qty_dict.items()))
            rnd_unit = random.choice(self.units_l)
            rnd_mod_present = random.choice([None, None, True])  # 1/3 chance
            rnd_mod = random.choice(self.mods_l) if rnd_mod_present else None
            
            # If we don't have ingredient data, generate placeholder ingredients
            if self.raw_ingredients_l:
                rnd_ing = random.choice(self.raw_ingredients_l)
            else:
                basic_ingredients = ["tomato", "onion", "garlic", "chicken", "beef", "potato", "carrot"]
                rnd_ing = random.choice(basic_ingredients)
            
            # Randomly decide if we'll use units or just count
            no_units_present = random.choice([False, False, False, False, True])  # 1/5 chance
            if no_units_present:
                rnd_unit = 'count'  # For Y output
            
            # Build Y (standardized output)
            if rnd_mod_present:
                Y[i] = f"{{ qty: {rnd_qty_int} , unit: {rnd_unit} , item: {rnd_ing} , mod: {rnd_mod} }}"
            else:
                Y[i] = f"{{ qty: {rnd_qty_int} , unit: {rnd_unit} , item: {rnd_ing} , mod: None }}"
            
            # Build X (raw input)
            # Randomly decide if modifier goes at end or before ingredient
            rnd_mod_at_end = random.choice([False, True])
            
            if rnd_mod_present:
                if no_units_present:
                    if rnd_mod_at_end:
                        X[i] = f'{rnd_qty_str} {rnd_ing} , {rnd_mod}'
                    else:
                        X[i] = f'{rnd_qty_str} {rnd_mod} {rnd_ing}'
                else:
                    if rnd_mod_at_end:
                        X[i] = f'{rnd_qty_str} {rnd_unit} {rnd_ing} , {rnd_mod}'
                    else:
                        X[i] = f'{rnd_qty_str} {rnd_unit} {rnd_mod} {rnd_ing}'
            else:
                if no_units_present:
                    X[i] = f'{rnd_qty_str} {rnd_ing}'
                else:
                    X[i] = f'{rnd_qty_str} {rnd_unit} {rnd_ing}'
        
        return X, Y
    
    def fine_tune(self, train_data=None, eval_data=None, output_dir="./fine_tuned_t5", 
                 epochs=3, batch_size=8, synthetic_data_size=1000, gradient_accumulation_steps=1,
                 save_steps=100, mixed_precision=None, deepspeed_config=None):
        """
        Fine-tune the T5 model on ingredient extraction data with memory optimizations
        
        Args:
            train_data: List of dicts with 'input' and 'output' keys, or None to use synthetic data
            eval_data: Optional evaluation data in the same format
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size (smaller values use less memory)
            synthetic_data_size: Number of synthetic examples to generate if train_data is None
            gradient_accumulation_steps: Number of steps to accumulate gradients (helps with memory)
            save_steps: Save a checkpoint every N steps
            mixed_precision: Use mixed precision training ("fp16" or "bf16")
            deepspeed_config: Custom DeepSpeed configuration (overrides default)
        """
        from transformers import Trainer, TrainingArguments
        from transformers import DataCollatorForSeq2Seq
        import torch
        import numpy as np
        from datasets import Dataset
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate synthetic data if no training data provided
        if train_data is None:
            print(f"Generating {synthetic_data_size} synthetic training examples...")
            x, y = self.generate_synthetic_data(synthetic_data_size)
            
            # Format into required structure
            train_data = [
                {"input": f"standardize ingredient: {x_i}", "output": y_i}
                for x_i, y_i in zip(x, y)
            ]
            
            # Create a small eval set from the generated data
            if eval_data is None:
                train_size = int(0.9 * len(train_data))
                eval_data = train_data[train_size:]
                train_data = train_data[:train_size]
        
        # More efficient data preparation using datasets library
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            
            # Set up the labels (target sequences)
            labels = self.tokenizer(
                examples["output"],
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Convert to Datasets format for memory efficiency
        train_dataset = Dataset.from_list(train_data)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        if eval_data:
            eval_dataset = Dataset.from_list(eval_data)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        else:
            eval_dataset = None
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if mixed_precision else None
        )
        
        # DeepSpeed configuration if enabled
        if self.use_deepspeed:
            try:
                import deepspeed
                print("Using DeepSpeed optimization")
                # If custom config provided, use it, otherwise create default
                if deepspeed_config is None:
                    # Basic DeepSpeed config
                    deepspeed_config = {
                        "zero_optimization": {
                            "stage": 2,
                            "offload_optimizer": {
                                "device": "cpu",
                                "pin_memory": True
                            },
                            "allgather_partitions": True,
                            "allgather_bucket_size": 2e8,
                            "reduce_scatter": True,
                            "reduce_bucket_size": 2e8,
                            "overlap_comm": True,
                            "contiguous_gradients": True
                        },
                        "train_batch_size": batch_size * gradient_accumulation_steps,
                        "fp16": {
                            "enabled": mixed_precision == "fp16"
                        },
                    }
                else:
                    print("Using custom DeepSpeed configuration")
                    
                # Save DeepSpeed config
                with open(os.path.join(output_dir, "ds_config.json"), "w") as f:
                    json.dump(deepspeed_config, f, indent=4)
            except ImportError:
                print("Warning: deepspeed not available, falling back to standard training")
        
        # Optimized training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=3,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            report_to="none",  # Don't report to any tracking system
            fp16=(mixed_precision == "fp16"),
            bf16=(mixed_precision == "bf16"),
            load_best_model_at_end=True if eval_dataset else False,
            deepspeed=os.path.join(output_dir, "ds_config.json") if deepspeed_config else None,
            # Add additional memory-saving configurations
            dataloader_num_workers=2,
            remove_unused_columns=True,
            disable_tqdm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        print(f"Starting fine-tuning with batch size {batch_size} and gradient accumulation {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        trainer.train()
        
        # Save the fine-tuned model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model fine-tuned and saved to {output_dir}")
        
        # Update the model and tokenizer to use the fine-tuned version
        # Only do this with non-8bit models as reloading quantized models requires special handling
        if not self.use_8bit:
            self.model = T5ForConditionalGeneration.from_pretrained(output_dir).to(self.device)
            self.tokenizer = T5Tokenizer.from_pretrained(output_dir)