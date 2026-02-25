# deploy.py

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential

import uuid

endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
deployment_name = "blue"

credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key",
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create environment
env = Environment(
    name="iris-env",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

ml_client.environments.create_or_update(env)

# Create deployment
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model="iris-model:latest",
    environment=env,
    code_configuration=CodeConfiguration(
        code="./",
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Set traffic
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(
        name=endpoint_name,
        traffic={"blue": 100}
    )
).result()

print("Deployment successful!")
