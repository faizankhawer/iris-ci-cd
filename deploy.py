from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

import uuid

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
deployment_name = "blue"

# Get latest registered model safely
model = ml_client.models.get(name="iris-model", label="latest")

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",
)

print("Creating endpoint...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create environment (versioned)
env = Environment(
    name="iris-env",
    description="Iris inference environment",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

env = ml_client.environments.create_or_update(env)

# Create deployment
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model.id,
    environment=env.id,
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py",
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

print("Deploying model...")
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route traffic
endpoint.traffic = {deployment_name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Deployment successful! Endpoint: {endpoint_name}")
