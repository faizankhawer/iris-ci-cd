import joblib
import json
import numpy as np

def init():
    global model
    model = joblib.load("model.pkl")

def run(raw_data):
    data = json.loads(raw_data)
    prediction = model.predict(np.array(data))
    return prediction.tolist()