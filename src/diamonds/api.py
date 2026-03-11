from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import diamonds.registry
import pandas as pd


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform any startup tasks here (e.g., connect to a database)
    print("Starting up the application...")
    global preprocessor
    preprocessor = diamonds.registry.load_model("preprocessor")
    global model
    model = diamonds.registry.load_model("model")
    yield
    # Perform any shutdown tasks here (e.g., close database connections)
    print("Shutting down the application...")

class Diamond(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

app = FastAPI(lifespan = lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(diamond: Diamond)->dict:
    X = pd.DataFrame([diamond.model_dump(exclude = {'price'})])
    X_preprocessed = preprocessor.transform(X)
    y = model.predict(X_preprocessed)
    return {"predicted price": y[0]}

@app.post("/predict_batch")
def predict_batch(diamonds: list[Diamond])->list:
    X = pd.DataFrame([d.model_dump(exclude = {'price'}) for d in diamonds])
    X_preprocessed = preprocessor.transform(X)
    y = model.predict(X_preprocessed)
    
    return [{"predicted_price": p} for p in y]
    
    