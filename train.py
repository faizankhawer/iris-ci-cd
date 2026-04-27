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


# -----------------------------
# Azure ML Client
# -----------------------------
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)


# -----------------------------
# Validate Environment Variables
# -----------------------------
def validate_env():
    required_vars = [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_WORKSPACE_NAME",
    ]

    missing = [var for var in required_vars if var not in os.environ]

    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")

    print("✅ Environment variables loaded successfully")


# -----------------------------
# Register Model to Azure ML
# -----------------------------
def register_model_to_azure(model_path, model_name):
    print("📦 Registering model in Azure ML...")

    model = Model(
        path=model_path,
        name=model_name,
        description="Iris classification model",
        type="custom_model",
    )

    registered_model = ml_client.models.create_or_update(model)

    print(
        f"✅ Model registered in Azure ML → Name: {registered_model.name}, Version: {registered_model.version}"
    )

    return registered_model


# -----------------------------
# Train Model
# -----------------------------
def train_model(model_name):
    print("📊 Loading Iris dataset...")

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():

        print("🚀 Training model...")

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        print("📈 Evaluating model...")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"✅ Accuracy: {acc}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        # Signature
        signature = infer_signature(X_train, preds)

        # Log model to MLflow (optional but fine)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train[:2],
        )

    # -----------------------------
    # Save model locally
    # -----------------------------
    model_path = "model.pkl"
    joblib.dump(model, model_path)

    print(f"💾 Model saved locally at {model_path}")

    # -----------------------------
    # Register in Azure ML
    # -----------------------------
    register_model_to_azure(model_path, model_name)

    return model


# -----------------------------
# Main
# -----------------------------
def main():
    validate_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="iris-model")

    args = parser.parse_args()

    train_model(args.model_name)


if __name__ == "__main__":
    main()
