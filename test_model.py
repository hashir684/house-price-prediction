import joblib
import numpy as np

# Load saved model and feature columns
model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")

# Sample prediction function
def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if location in columns:
        loc_index = np.where(columns == location)[0][0]
        x[loc_index] = 1
    return model.predict([x])[0]

print(predict_price("Indira Nagar", 1000, 2, 2))
