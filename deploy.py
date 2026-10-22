
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential


MODEL_NAME = "iris-model"
ENDPOINT_NAME = "iris-endpoint"
DEPLOYMENT_NAME = "blue"

INSTANCE_TYPE = os.getenv(
    "AZUREML_INSTANCE_TYPE", "Standard_DS2_v2"
)


def main():
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
    )

    # Find the latest registered version.
    models = list(ml_client.models.list(name=MODEL_NAME))

    if not models:
        raise RuntimeError(
            f"No registered model found: {MODEL_NAME}"
        )

    model = max(models, key=lambda item: int(item.version))

    print(
        f"Deploying {model.name}, version {model.version}"
    )

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Iris classification endpoint",
        auth_mode="key",
    )

    ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()

    environment = Environment(
        name="iris-inference-env",
        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file="conda.yaml",
    )

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()

    # Route all endpoint traffic to the new deployment.
    endpoint = ml_client.online_endpoints.get(
        name=ENDPOINT_NAME
    )
    endpoint.traffic = {DEPLOYMENT_NAME: 100}

    ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()

    print(f"Deployment completed: {ENDPOINT_NAME}")

    keys = ml_client.online_endpoints.get_keys(
        name=ENDPOINT_NAME
    )

    print("Endpoint URL:", endpoint.scoring_uri)
    print("Primary key retrieved; not displaying it.")


if __name__ == "__main__":
    main()
