import mlflow
from mlflow import MlflowClient


def get_production_model_version(model_name: str) -> str:
    """
    Get the production model version from the MLflow Model Registry.

    Parameters
    ----------
    model_name:
        Registered MLflow model name.

    Returns
    -------
    str:
        Model identifier formatted as:
        <model_name>:v<version>
    """
    client = MlflowClient()
    mlflow.set_tracking_uri("sqlite:////Users/graham/Projects/xG_model/mlflow.db")

    model_version = client.get_model_version_by_alias(
        name=model_name,
        alias="production",
    )

    return f"{model_name}:v{model_version.version}"
