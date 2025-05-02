import os

import joblib
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

app = FastAPI(
    title="SentimentHub API",
    description="Predict sentiment (positive/negative) for a given text.",
    version="1.0",
)


class TextIn(BaseModel):
    text: str


try:
    model = joblib.load(MODEL_PATH)
    vecorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    model = None
    vecorizer = None
    print(f"Error loading model/vectorizer: {e}")


@app.post("/predict/")
def predict_sentiment(input: TextIn):
    if model is None or vecorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer not loaded.")
    try:
        vec = vecorizer.transform([input.text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0][pred]
        return {
            "sentiment": "positive" if pred == 1 else "negative",
            "confidence": float(proba),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Welcome to SentimentHub API! Go to /docs for interactive docs."}
