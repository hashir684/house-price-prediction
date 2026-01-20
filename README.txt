HOUSE PRICE PREDICTION PROJECT

Overview
This project predicts house prices using a trained Machine Learning model and also allows users to ask questions in normal English using a Local LLM (TinyLlama via Ollama).

The system works in two modes:
1. Normal API prediction using structured JSON input
2. Natural language query using LLM to extract details automatically

------------------------------------------------------------

Technologies Used

- Python
- FastAPI
- Machine Learning (scikit-learn)
- Joblib
- LangChain
- Ollama (TinyLlama)
- NumPy

------------------------------------------------------------

Project Files

main_api.py        -> Main FastAPI backend file
llm_help.py        -> LLM processing and query extraction
house_price_model.pkl -> Trained ML model
model_columns.pkl  -> Feature columns used in model
test_model.py      -> Testing script

------------------------------------------------------------

How to Run the Project

1. Clone the repository

git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

------------------------------------------------------------

2. Create and activate virtual environment

python -m venv venv

On Windows:
venv\Scripts\activate

------------------------------------------------------------

3. Install dependencies

pip install -r requirements.txt

------------------------------------------------------------

4. Install Ollama and TinyLlama

Download Ollama from:
https://ollama.com

Then run:

ollama pull tinyllama

------------------------------------------------------------

5. Run FastAPI Server

uvicorn main_api:app --reload

Server will run at:

http://127.0.0.1:8000

------------------------------------------------------------

Available API Endpoints

1. Check API Status

GET /

Example:
http://127.0.0.1:8000/

Response:
{
  "message": "House Price Prediction API is running"
}

------------------------------------------------------------

2. Normal Prediction

POST /predict

Input Example:

{
  "location": "Whitefield",
  "total_sqft": 1200,
  "bath": 2,
  "bhk": 2
}

------------------------------------------------------------

3. Natural Language Prediction

POST /ask_and_predict

Input Example:

{
  "query": "Predict price for 2 BHK house in Whitefield with 2 bathrooms and 1200 sqft"
}

The LLM will automatically extract details and return predicted price.

------------------------------------------------------------

Future Improvements

- Add a web-based frontend
- Support more flexible queries
- Improve accuracy of LLM extraction
- Deploy the system online

------------------------------------------------------------

Author

Muhammad Hashir Khan

------------------------------------------------------------

End of Documentation
