import os
import uuid

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential


# -----------------------------
# Authenticate & Initialize Client
# -----------------------------
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_WORKSPACE_NAME"],
)

print("Connected to Azure ML workspace")

# -----------------------------
# Names
# -----------------------------
endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
deployment_name = "blue"

# -----------------------------
# Get Latest Model
# -----------------------------
print("Fetching latest model...")
models = list(ml_client.models.list(name="iris-model"))

if not models:
    raise Exception("❌ No models found in Azure ML")

model = max(models, key=lambda m: int(m.version))

print(f"Using model version: {model.version}")

# -----------------------------
# Create Endpoint
# -----------------------------
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",
)

print(f"Creating endpoint: {endpoint_name}")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# -----------------------------
# Create Environment
# -----------------------------
print("Creating environment...")

env = Environment(
    name="iris-env",
    description="Iris inference environment",
    conda_file="conda.yaml",   # make sure this file exists
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

env = ml_client.environments.create_or_update(env)

# -----------------------------
# Create Deployment
# -----------------------------
print("Deploying model...")

deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model.id,
    environment=env.id,
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py",   # make sure this file exists
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# -----------------------------
# Route Traffic
# -----------------------------
endpoint.traffic = {deployment_name: 100}

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"✅ Deployment successful!")
print(f"Endpoint name: {endpoint_name}")
