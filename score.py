
import os
import json
import joblib
import numpy as np


def init():
    global model

    model_dir = os.environ.get("AZUREML_MODEL_DIR")

    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    model_path = os.path.join(model_dir, "model.pkl")

    if not os.path.exists(model_path):
        # Support deployments where the model is mounted
        # inside a version-specific subdirectory.
        matches = []
        for root, _, files in os.walk(model_dir):
            if "model.pkl" in files:
                matches.append(os.path.join(root, "model.pkl"))

        if not matches:
            raise FileNotFoundError(
                f"model.pkl not found under {model_dir}"
            )

        model_path = matches[0]

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")


def run(raw_data):
    try:
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        data = raw_data.get("data", raw_data)
        features = np.asarray(data, dtype=float)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.ndim != 2 or features.shape[1] != 4:
            return {
                "error": "Expected 4 features per Iris sample."
            }

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        iris_names = [
            "Iris-setosa",
            "Iris-versicolor",
            "Iris-virginica",
        ]

        results = []

        for prediction, probability in zip(
            predictions, probabilities
        ):
            class_id = int(prediction)

            results.append({
                "class_id": class_id,
                "species": iris_names[class_id],
                "probabilities": {
                    iris_names[i]: round(float(probability[i]), 4)
                    for i in range(len(iris_names))
                },
            })

        return {"predictions": results}

    except Exception as exc:
        return {"error": str(exc)}
