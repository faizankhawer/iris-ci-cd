# train.py

import os
import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def get_ml_client():
    """
    Create MLClient using environment variables (CI/CD compatible).
    """

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )

    return ml_client


def train_model():

    print("Loading Iris dataset...")

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Evaluating model...")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    return model, acc


def save_model(model):

    os.makedirs("outputs", exist_ok=True)

    model_path = "outputs/model.pkl"

    joblib.dump(model, model_path)

    print("Model saved at:", model_path)

    return model_path


def register_model(model_path, model_name):

    print("Connecting to Azure ML...")

    ml_client = get_ml_client()

    print("Registering model...")

    model = Model(
        path=model_path,
        name=model_name,
        description="Iris classification model",
        type="custom_model",
    )

    registered_model = ml_client.models.create_or_update(model)

    print("Model registered successfully!")

    return registered_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="iris-model")

    args = parser.parse_args()

    model, acc = train_model()

    model_path = save_model(model)

    register_model(model_path, args.model_name)


if __name__ == "__main__":
    main()
