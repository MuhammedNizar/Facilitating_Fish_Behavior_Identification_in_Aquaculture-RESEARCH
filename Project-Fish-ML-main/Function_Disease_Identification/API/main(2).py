from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import uvicorn

app = FastAPI()

# Allowing all origins, methods, and headers for simplicity.
# You may want to restrict these based on your deployment needs.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the scaler and model
loaded_scaler = joblib.load(r"C:\Users\theekshana\Desktop\ML\scaler_filename.joblib")
new_model_ = tf.keras.models.load_model(r"C:\Users\theekshana\Desktop\ML\new_model_")

class InputData(BaseModel):
    Temperature: float
    Turbidity: float
    Dissolved_Oxygen: float
    PH: float
    Nitrate: float
    Ammonia: float
    Salinity: float

# Assuming you have a dictionary that maps class indices to their labels
class_labels = {0: "Fin Rot", 1: "Red Spot", 2: "White Spot (Ich)"}

@app.post('/predict')
def predict(data: InputData):
    # Convert input data to a numpy array
    input_data = np.array([[data.Temperature, data.Turbidity, data.Dissolved_Oxygen,
                            data.PH, data.Nitrate, data.Ammonia, data.Salinity]])

    # Scale the input data
    input_data_scaled = loaded_scaler.transform(input_data)

    # Make a prediction
    prediction = new_model_.predict(input_data_scaled)

    # Convert the predictions to percentage without rounding
    percentage_values = np.round(prediction.flatten() * 100, 4)

    # Format the percentage values without scientific notation
    formatted_percentage_values = [f'{val:.4f}' for val in percentage_values]

    # Prepare the response with class labels
    response = {"predictions": {label: percentage for label, percentage in zip(class_labels.values(), formatted_percentage_values)}}

    return response

# Run the FastAPI app using Uvicorn
nest_asyncio.apply()
uvicorn.run(app, host='127.0.0.1', port=8002)