from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.entities import Model, Environment, CodeConfiguration

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential,
    "SUBSCRIPTION_ID",
    "RESOURCE_GROUP",
    "WORKSPACE_NAME",
)

# Register model
model = ml_client.models.create_or_update(
    Model(
        path="outputs/model.pkl",
        name="iris-model"
    )
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="iris-endpoint",
    auth_mode="key"
)

ml_client.begin_create_or_update(endpoint).wait()

# Deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="iris-endpoint",
    model=model,
    environment=Environment(
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    ),
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1
)

ml_client.begin_create_or_update(deployment).wait()