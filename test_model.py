import joblib
import numpy as np
import os

project_folder = r"f:\house-price-prediction"
model_path = os.path.join(project_folder, "house_price_model.pkl")
columns_path = os.path.join(project_folder, "model_columns.pkl")

try:
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    print("Model and columns loaded successfully!")

except FileNotFoundError:
    print("Error: .pkl files not found. Run your training script first to create them.")

    exit()

def predict_price(location, total_sqft, bath, bhk):
    x = np.zeros(len(columns))

    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk

    # One-hot encode the location
    if location in columns:
        loc_index = np.where(columns == location)[0][0]
        x[loc_index] = 1

    # Predict price
    price = model.predict([x])[0]
    return price

if __name__ == "__main__":
    test_location = "Indira Nagar"
    test_sqft = 1000
    test_bath = 2
    test_bhk = 2

    predicted_price = predict_price(test_location, test_sqft, test_bath, test_bhk)
    print(f"Predicted price for {test_sqft} sqft, {test_bhk} BHK, {test_bath} bath in {test_location}: {predicted_price:.2f} Lakhs")
