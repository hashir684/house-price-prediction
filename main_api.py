from fastapi import FastAPI
import joblib
import numpy as np
import os
from llm_help import extract_details_from_query

app = FastAPI()

project_folder = r"f:\house-price-prediction"

model_path = os.path.join(project_folder, "house_price_model.pkl")
columns_path = os.path.join(project_folder, "model_columns.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: dict):
    location = data["location"]
    sqft = data["total_sqft"]
    bath = data["bath"]
    bhk = data["bhk"]

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk


    if location in columns:
        loc_index = np.where(columns == location)[0][0]
        x[loc_index] = 1

    price = model.predict([x])[0]

    return {
        "predicted_price": round(float(price), 2)
    }


@app.post("/ask_and_predict")
def ask_and_predict(data: dict):
    user_query = data["query"]

    details = extract_details_from_query(user_query)

    required_keys = ["location", "total_sqft", "bath", "bhk"]

    for key in required_keys:
        if key not in details:
            return {
                "error": f"Missing field: {key}",
                "llm_output": details
            }

    return predict(details)
