#!/usr/bin/env python3
"""
FastAPI application for ingredient standardization using the T5 model.
"""
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uvicorn

from model.t5_extractor import T5Extractor
from data_utils import generate_synthetic_examples

# Initialize the T5 model
model = T5Extractor()

# Create FastAPI app
app = FastAPI(
    title="Ingredient Standardization API",
    description="API for standardizing ingredient text using T5 model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngredientRequest(BaseModel):
    text: str
    language: str = "en"

class IngredientResponse(BaseModel):
    food_name: Optional[str] = None
    quantity: Optional[float] = None
    portion: Optional[str] = None
    modifier: Optional[str] = None
    original_text: str
    standardized_format: Optional[str] = None

class BatchIngredientRequest(BaseModel):
    ingredients: List[IngredientRequest]

@app.get("/")
async def root():
    """Root endpoint that provides basic information about the API."""
    return {
        "message": "Ingredient Standardization API",
        "version": "1.0.0",
        "endpoints": {
            "/standardize": "Standardize a single ingredient",
            "/batch-standardize": "Standardize multiple ingredients",
            "/examples": "Generate example ingredient texts",
            "/health": "Check API health"
        }
    }

@app.post("/standardize", response_model=IngredientResponse)
async def standardize_ingredient(request: IngredientRequest):
    """
    Standardize a single ingredient text into structured format.
    
    Args:
        request: IngredientRequest with text and language
        
    Returns:
        Structured ingredient data
    """
    try:
        result = model.extract_fields(request.text, request.language)
        # Create response with both parsed data and original text
        return IngredientResponse(
            food_name=result.get("food_name"),
            quantity=result.get("quantity"),
            portion=result.get("portion"),
            modifier=result.get("modifier"),
            original_text=request.text,
            standardized_format=f"{{ qty: {result.get('quantity')} , unit: {result.get('portion') or 'count'} , " +
                               f"item: {result.get('food_name')} , mod: {result.get('modifier') or 'None'} }}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ingredient: {str(e)}")

@app.post("/batch-standardize", response_model=List[IngredientResponse])
async def batch_standardize(requests: BatchIngredientRequest):
    """
    Standardize multiple ingredient texts.
    
    Args:
        requests: Batch of ingredient texts
        
    Returns:
        List of structured ingredient data
    """
    results = []
    for request in requests.ingredients:
        try:
            result = model.extract_fields(request.text, request.language)
            # Create response with both parsed data and original text
            results.append(IngredientResponse(
                food_name=result.get("food_name"),
                quantity=result.get("quantity"),
                portion=result.get("portion"),
                modifier=result.get("modifier"),
                original_text=request.text,
                standardized_format=f"{{ qty: {result.get('quantity')} , unit: {result.get('portion') or 'count'} , " +
                                   f"item: {result.get('food_name')} , mod: {result.get('modifier') or 'None'} }}"
            ))
        except Exception as e:
            # Add error information for this ingredient
            results.append(IngredientResponse(
                original_text=request.text,
                standardized_format=f"Error: {str(e)}"
            ))
    return results

@app.get("/examples", response_model=List[Dict[str, Any]])
async def get_examples(count: int = Query(5, description="Number of examples to generate")):
    """
    Generate example ingredient texts and their standardized formats.
    
    Args:
        count: Number of examples to generate
        
    Returns:
        List of example ingredient texts and standardized formats
    """
    try:
        resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
        examples = generate_synthetic_examples(count, resources_dir)
        return examples
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating examples: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    # Run the API with uvicorn when script is executed directly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)