from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species using a model loaded from a run artifact",
    version="0.1",
)

class IrisRequest(BaseModel):
    features: list

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/iris_final_model/1" 
    model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/predict")
def predict(request: IrisRequest):
    input_array = np.array(request.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
