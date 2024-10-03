import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict, List
import uvicorn
from contextlib import asynccontextmanager

# Define request model
class TextRequest(BaseModel):
    text: str

# Define response model
class PredictionResponse(BaseModel):
    predicted_label: str
    class_probabilities: Dict[str, float]

# Helper function to get embeddings
def get_embeddings(texts: List[str]) -> np.ndarray:
    embeddings = use_model(texts)
    return embeddings.numpy()

# Initialize the FastAPI app with lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global use_model, lr_model
    
    # Load Universal Sentence Encoder
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Load your pre-trained logistic regression model and label encoder
    with open('lr_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    
    # Yield control back to the application
    yield

app = FastAPI(title="Text Classification API", lifespan=lifespan)

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    # Get embeddings for the input text
    embedding = get_embeddings([request.text])
    
    # Get prediction probabilities
    probabilities = lr_model.predict_proba(embedding)[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = lr_model.classes_[predicted_class_idx]
    
    # Create dictionary of class probabilities
    class_probs = {
        label: float(prob) 
        for label, prob in zip(lr_model.classes_, probabilities)
    }
    
    return PredictionResponse(
        predicted_label=predicted_label,
        class_probabilities=class_probs
    )

