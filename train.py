
import os
import argparse
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def get_ml_client():
    required = [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_WORKSPACE_NAME",
    ]

    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {missing}"
        )

    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )


def train_model(model_name):
    ml_client = get_ml_client()

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/model.pkl"
    joblib.dump(model, model_path)

    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:2],
        )

    registered_model = ml_client.models.create_or_update(
        Model(
            path=model_path,
            name=model_name,
            description="Random Forest Iris classification model",
            type="custom_model",
        )
    )

    print(
        f"Registered model: {registered_model.name}, "
        f"version: {registered_model.version}"
    )

    return registered_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="iris-model")
    args = parser.parse_args()

    train_model(args.model_name)
