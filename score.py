# score.py

import json
import joblib
import os
import numpy as np

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)
        prediction = model.predict(np.array(input_data["data"]))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
