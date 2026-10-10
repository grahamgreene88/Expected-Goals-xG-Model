"""Export the registry's production model for deployment.

Run after (re)assigning the `production` alias:
    python -m model.export_model
"""

import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = REPO_ROOT / "production_model"
MODEL_NAME = "nhl_xg_xgboost"
ALIAS = "production"


def export() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)

    model_dir = EXPORT_DIR / "model"
    shutil.rmtree(EXPORT_DIR, ignore_errors=True)
    model_dir.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(model_dir)
    )
    if not (model_dir / "MLmodel").exists():
        raise RuntimeError(f"No MLmodel file in {model_dir}; check the download layout")

    meta = {
        "model_version": f"{MODEL_NAME}:v{mv.version}",
        "run_id": mv.run_id,
    }
    (EXPORT_DIR / "model_version.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {meta['model_version']} to {EXPORT_DIR}")


if __name__ == "__main__":
    export()
