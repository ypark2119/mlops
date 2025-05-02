from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

@app.get("/")
def read_root():
    return {"message": "This is a model for classifying Reddit comments"}

class RequestBody(BaseModel):  # Capitalized class name!
    reddit_comment: str

@app.on_event("startup")
def load_model():
    global model_pipeline
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")

@app.post("/predict")
def predict(data: RequestBody):
    prediction = model_pipeline.predict_proba([data.reddit_comment])
    return {"Predictions": prediction.tolist()}
