# train.py

import os
import joblib
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="iris-model")
    args = parser.parse_args()

    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")

    # Register model in Azure ML
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential)

    registered_model = Model(
        path="outputs/model.pkl",
        name=args.model_name,
        description="Iris classification model",
        type="custom_model",
    )

    ml_client.models.create_or_update(registered_model)
    print("Model registered successfully!")

if __name__ == "__main__":
    main()
