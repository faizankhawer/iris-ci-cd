import json
import joblib
import os
import numpy as np

def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # Robust path handling
    model_path = None
    for root, dirs, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_path = os.path.join(root, "model.pkl")
            break

    if model_path is None:
        raise FileNotFoundError("model.pkl not found in AZUREML_MODEL_DIR")

    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")


def run(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)

        if "data" not in data:
            return {"error": "Missing 'data' key in input"}

        input_array = np.array(data["data"])

        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

        prediction = model.predict(input_array)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
